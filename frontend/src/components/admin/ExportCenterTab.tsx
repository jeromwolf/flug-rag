import { useState } from "react";
import {
  Box,
  Typography,
  Card,
  CardContent,
  Grid,
  Alert,
  Chip,
  TextField,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Snackbar,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Divider,
  Stack,
  Avatar,
} from "@mui/material";
import ThumbUpIcon from "@mui/icons-material/ThumbUp";
import SpeedIcon from "@mui/icons-material/Speed";
import AccessTimeIcon from "@mui/icons-material/AccessTime";
import QuestionAnswerIcon from "@mui/icons-material/QuestionAnswer";
import DownloadIcon from "@mui/icons-material/Download";
import GroupIcon from "@mui/icons-material/Group";
import { feedbackApi, authApi, logsApi } from "../../api/client";

interface ExportType {
  id: string;
  title: string;
  description: string;
  estimatedRows: number;
  lastExport: string;
  icon: React.ReactNode;
  color: string;
}

interface ExportHistoryItem {
  id: string;
  name: string;
  format: string;
  rows: number;
  size: string;
  date: string;
}

async function fetchAndGenerateFeedbackCSV(): Promise<string> {
  const resp = await feedbackApi.list(500);
  const feedbacks = resp.data?.feedbacks || [];
  const headers = ["id", "session_id", "message_id", "rating", "comment", "correction_text", "created_at", "username"];
  const BOM = "\uFEFF";
  const escape = (v: string) => `"${(v ?? "").replace(/"/g, '""')}"`;
  const rows = feedbacks.map((f: any) =>
    headers.map((h) => escape(String(f[h] ?? ""))).join(",")
  );
  return BOM + [headers.map(escape).join(","), ...rows].join("\n");
}

async function fetchAndGenerateUsersCSV(): Promise<string> {
  const resp = await authApi.listUsers();
  const users = resp.data?.users || resp.data || [];
  const headers = ["username", "email", "role", "department", "is_active", "last_login", "created_at"];
  const BOM = "\uFEFF";
  const escape = (v: string) => `"${(v ?? "").replace(/"/g, '""')}"`;
  const rows = (Array.isArray(users) ? users : []).map((u: any) =>
    headers.map((h) => escape(String(u[h] ?? ""))).join(",")
  );
  return BOM + [headers.map(escape).join(","), ...rows].join("\n");
}

async function fetchAndGenerateConversationsCSV(): Promise<string> {
  const resp = await logsApi.searchQueries({ page: 1, page_size: 100 });
  const queries = resp.data?.queries || [];
  const headers = ["timestamp", "session_id", "question", "answer", "confidence", "latency_ms", "model_used"];
  const BOM = "\uFEFF";
  const escape = (v: string) => `"${(v ?? "").replace(/"/g, '""')}"`;
  const rows = queries.map((q: any) =>
    [q.timestamp, q.session_id, q.content, q.answer, q.confidence ?? "", q.latency_ms ?? "", q.model_used ?? ""]
      .map((v) => escape(String(v ?? "")))
      .join(",")
  );
  return BOM + [headers.map(escape).join(","), ...rows].join("\n");
}

async function fetchAndGenerateStatsCSV(): Promise<string> {
  const resp = await logsApi.searchAccess({ page: 1, page_size: 100 });
  const logs = resp.data?.logs || [];
  const headers = ["timestamp", "username", "action", "details", "ip_address"];
  const BOM = "\uFEFF";
  const escape = (v: string) => `"${(v ?? "").replace(/"/g, '""')}"`;
  const rows = logs.map((l: any) =>
    headers.map((h) => escape(String(l[h] ?? ""))).join(",")
  );
  return BOM + [headers.map(escape).join(","), ...rows].join("\n");
}

async function fetchAndGenerateJSON(exportId: string): Promise<string> {
  let data: any[] = [];
  if (exportId === "feedback") {
    const resp = await feedbackApi.list(500);
    data = resp.data?.feedbacks || [];
  } else if (exportId === "stats") {
    const resp = await logsApi.searchAccess({ page: 1, page_size: 100 });
    data = resp.data?.logs || [];
  } else if (exportId === "users") {
    const resp = await authApi.listUsers();
    data = resp.data?.users || resp.data || [];
    if (!Array.isArray(data)) data = [];
  } else {
    const resp = await logsApi.searchQueries({ page: 1, page_size: 100 });
    data = resp.data?.queries || [];
  }
  return JSON.stringify(data, null, 2);
}

function ExportCenterTab() {
  const [dialogOpen, setDialogOpen] = useState(false);
  const [selectedExport, setSelectedExport] = useState<ExportType | null>(null);
  const [dateFrom, setDateFrom] = useState("2026-01-01");
  const [dateTo, setDateTo] = useState("2026-03-05");
  const [format, setFormat] = useState<"csv" | "json">("csv");
  const [snackOpen, setSnackOpen] = useState(false);
  const [exportHistory, setExportHistory] = useState<ExportHistoryItem[]>([]);

  const exportTypes: ExportType[] = [
    {
      id: "feedback",
      title: "피드백 데이터",
      description: "모든 피드백 (질문, 답변, 평점, 수정의견, 날짜 포함)",
      estimatedRows: 0,
      lastExport: "—",
      icon: <ThumbUpIcon sx={{ fontSize: 36 }} />,
      color: "#1976d2",
    },
    {
      id: "stats",
      title: "접근 로그",
      description: "사용자 접근 로그 (timestamp, username, action, ip_address)",
      estimatedRows: 0,
      lastExport: "—",
      icon: <SpeedIcon sx={{ fontSize: 36 }} />,
      color: "#388e3c",
    },
    {
      id: "users",
      title: "사용자 목록",
      description: "사용자 정보, 역할, 부서, 최근 로그인 날짜",
      estimatedRows: 0,
      lastExport: "—",
      icon: <GroupIcon sx={{ fontSize: 36 }} />,
      color: "#7b1fa2",
    },
    {
      id: "conversations",
      title: "대화 로그",
      description: "쿼리 로그 (timestamp, session_id, 질문, 답변, confidence, latency)",
      estimatedRows: 0,
      lastExport: "—",
      icon: <QuestionAnswerIcon sx={{ fontSize: 36 }} />,
      color: "#e65100",
    },
  ];

  function handleCardClick(et: ExportType) {
    setSelectedExport(et);
    setDialogOpen(true);
  }

  async function handleExport() {
    if (!selectedExport) return;

    let content: string;
    let mimeType: string;
    let ext: string;

    try {
      if (format === "csv") {
        if (selectedExport.id === "feedback") content = await fetchAndGenerateFeedbackCSV();
        else if (selectedExport.id === "stats") content = await fetchAndGenerateStatsCSV();
        else if (selectedExport.id === "users") content = await fetchAndGenerateUsersCSV();
        else content = await fetchAndGenerateConversationsCSV();
        mimeType = "text/csv;charset=utf-8";
        ext = "csv";
      } else {
        content = await fetchAndGenerateJSON(selectedExport.id);
        mimeType = "application/json";
        ext = "json";
      }
    } catch (err) {
      console.error("Export failed:", err);
      alert("내보내기에 실패했습니다. 잠시 후 다시 시도해주세요.");
      return;
    }

    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    const fileName = `${selectedExport.id}_${dateFrom}_${dateTo}.${ext}`;
    a.download = fileName;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    const rows = content.split("\n").length - 1; // minus header
    const sizeKb = Math.round(blob.size / 1024);
    const now = new Date();
    const dateStr = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, "0")}-${String(now.getDate()).padStart(2, "0")} ${String(now.getHours()).padStart(2, "0")}:${String(now.getMinutes()).padStart(2, "0")}`;
    const newItem: ExportHistoryItem = {
      id: `h${Date.now()}`,
      name: selectedExport.title,
      format: format.toUpperCase(),
      rows,
      size: `${sizeKb} KB`,
      date: dateStr,
    };
    setExportHistory((prev) => [newItem, ...prev].slice(0, 10));
    setDialogOpen(false);
    setSnackOpen(true);
  }

  return (
    <Box>
      <Typography variant="h6" fontWeight={700} mb={1}>
        데이터 내보내기
      </Typography>
      <Typography variant="body2" color="text.secondary" mb={3}>
        원하는 데이터 유형을 선택하고 날짜 범위와 형식을 지정하여 내보내세요.
      </Typography>

      {/* Export type cards */}
      <Grid container spacing={2} mb={4}>
        {exportTypes.map((et) => (
          <Grid key={et.id} size={{ xs: 12, sm: 6, md: 3 }}>
            <Card
              sx={{
                cursor: "pointer",
                borderRadius: 2,
                border: "1px solid",
                borderColor: "divider",
                transition: "box-shadow 0.2s, transform 0.15s",
                "&:hover": {
                  boxShadow: 4,
                  transform: "translateY(-2px)",
                },
              }}
              onClick={() => handleCardClick(et)}
            >
              <CardContent>
                <Box display="flex" alignItems="center" gap={1.5} mb={1.5}>
                  <Avatar sx={{ bgcolor: `${et.color}22`, color: et.color, width: 52, height: 52 }}>
                    {et.icon}
                  </Avatar>
                  <Box>
                    <Typography variant="subtitle1" fontWeight={700}>
                      {et.title}
                    </Typography>
                    <Chip
                      label={`약 ${et.estimatedRows.toLocaleString()}행`}
                      size="small"
                      sx={{ bgcolor: `${et.color}18`, color: et.color, fontWeight: 600, fontSize: "0.7rem" }}
                    />
                  </Box>
                </Box>
                <Typography variant="body2" color="text.secondary" mb={1.5} sx={{ minHeight: 40 }}>
                  {et.description}
                </Typography>
                <Divider sx={{ mb: 1 }} />
                <Box display="flex" alignItems="center" gap={0.5}>
                  <AccessTimeIcon sx={{ fontSize: 14, color: "text.disabled" }} />
                  <Typography variant="caption" color="text.disabled">
                    최근 내보내기: {et.lastExport}
                  </Typography>
                </Box>
                <Box display="flex" justifyContent="flex-end" mt={1}>
                  <Button
                    size="small"
                    variant="outlined"
                    startIcon={<DownloadIcon />}
                    sx={{ borderRadius: 5, textTransform: "none", fontSize: "0.75rem" }}
                    onClick={(e) => {
                      e.stopPropagation();
                      handleCardClick(et);
                    }}
                  >
                    내보내기
                  </Button>
                </Box>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>

      {/* Export history */}
      <Typography variant="subtitle1" fontWeight={700} mb={1.5}>
        최근 내보내기 이력
      </Typography>
      <TableContainer component={Paper} variant="outlined" sx={{ borderRadius: 2 }}>
        <Table size="small">
          <TableHead>
            <TableRow sx={{ bgcolor: "action.hover" }}>
              <TableCell sx={{ fontWeight: 700 }}>데이터 유형</TableCell>
              <TableCell sx={{ fontWeight: 700 }}>형식</TableCell>
              <TableCell sx={{ fontWeight: 700 }} align="right">행 수</TableCell>
              <TableCell sx={{ fontWeight: 700 }} align="right">크기</TableCell>
              <TableCell sx={{ fontWeight: 700 }}>내보낸 시각</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {exportHistory.map((item) => (
              <TableRow key={item.id} hover>
                <TableCell>{item.name}</TableCell>
                <TableCell>
                  <Chip
                    label={item.format}
                    size="small"
                    color={item.format === "CSV" ? "primary" : "secondary"}
                    variant="outlined"
                  />
                </TableCell>
                <TableCell align="right">{item.rows.toLocaleString()}</TableCell>
                <TableCell align="right">{item.size}</TableCell>
                <TableCell>{item.date}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>

      {/* Export dialog */}
      <Dialog open={dialogOpen} onClose={() => setDialogOpen(false)} maxWidth="xs" fullWidth>
        <DialogTitle sx={{ display: "flex", alignItems: "center", gap: 1 }}>
          <DownloadIcon color="primary" />
          {selectedExport?.title} 내보내기
        </DialogTitle>
        <DialogContent>
          <Stack spacing={2.5} mt={0.5}>
            <Box>
              <Typography variant="body2" fontWeight={600} mb={1}>
                날짜 범위
              </Typography>
              <Stack direction="row" spacing={1} alignItems="center">
                <TextField
                  label="시작일"
                  type="date"
                  size="small"
                  value={dateFrom}
                  onChange={(e) => setDateFrom(e.target.value)}
                  InputLabelProps={{ shrink: true }}
                  fullWidth
                />
                <Typography variant="body2" color="text.secondary" sx={{ whiteSpace: "nowrap" }}>
                  ~
                </Typography>
                <TextField
                  label="종료일"
                  type="date"
                  size="small"
                  value={dateTo}
                  onChange={(e) => setDateTo(e.target.value)}
                  InputLabelProps={{ shrink: true }}
                  fullWidth
                />
              </Stack>
            </Box>
            <Box>
              <Typography variant="body2" fontWeight={600} mb={1}>
                내보내기 형식
              </Typography>
              <Stack direction="row" spacing={1}>
                {(["csv", "json"] as const).map((f) => (
                  <Button
                    key={f}
                    variant={format === f ? "contained" : "outlined"}
                    size="small"
                    onClick={() => setFormat(f)}
                    sx={{ textTransform: "uppercase", fontWeight: 700, borderRadius: 5, minWidth: 80 }}
                  >
                    {f}
                  </Button>
                ))}
              </Stack>
            </Box>
            <Alert severity="info" sx={{ fontSize: "0.8rem" }}>
              약 <strong>{selectedExport?.estimatedRows.toLocaleString()}행</strong>의 데이터가 포함됩니다.
              {format === "csv" && " CSV는 Excel 호환 BOM 인코딩(UTF-8)을 사용합니다."}
            </Alert>
          </Stack>
        </DialogContent>
        <DialogActions sx={{ px: 3, pb: 2 }}>
          <Button onClick={() => setDialogOpen(false)} sx={{ textTransform: "none" }}>
            취소
          </Button>
          <Button
            onClick={handleExport}
            variant="contained"
            startIcon={<DownloadIcon />}
            sx={{ textTransform: "none", borderRadius: 5 }}
          >
            내보내기
          </Button>
        </DialogActions>
      </Dialog>

      {/* Success snackbar */}
      <Snackbar
        open={snackOpen}
        autoHideDuration={3500}
        onClose={() => setSnackOpen(false)}
        anchorOrigin={{ vertical: "bottom", horizontal: "center" }}
      >
        <Alert
          onClose={() => setSnackOpen(false)}
          severity="success"
          variant="filled"
          sx={{ width: "100%", borderRadius: 2 }}
        >
          파일이 다운로드되었습니다.
        </Alert>
      </Snackbar>
    </Box>
  );
}

export default ExportCenterTab;
