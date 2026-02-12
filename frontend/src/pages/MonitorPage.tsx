import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  Box,
  List,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Typography,
  Card,
  CardContent,
  Grid,
  CircularProgress,
  Alert,
  Chip,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Tab,
  Tabs,
  TextField,
  Button,
  Select,
  MenuItem,
  FormControl,
  TablePagination,
} from "@mui/material";
import ThumbUpIcon from "@mui/icons-material/ThumbUp";
import ThumbDownIcon from "@mui/icons-material/ThumbDown";
import RemoveIcon from "@mui/icons-material/Remove";
import FeedbackIcon from "@mui/icons-material/Feedback";
import TrendingUpIcon from "@mui/icons-material/TrendingUp";
import TrendingDownIcon from "@mui/icons-material/TrendingDown";
import FolderIcon from "@mui/icons-material/Folder";
import CheckCircleIcon from "@mui/icons-material/CheckCircle";
import BuildIcon from "@mui/icons-material/Build";
import DownloadIcon from "@mui/icons-material/Download";
import SearchIcon from "@mui/icons-material/Search";
import BarChartIcon from "@mui/icons-material/BarChart";
import { PieChart, Pie, Cell, Legend, Tooltip as RechartsTooltip, ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid } from "recharts";
import api, { feedbackApi, adminApi, mcpApi, statsApi, logsApi } from "../api/client";
import Layout from "../components/Layout";

const PIE_COLORS = ["#4caf50", "#f44336", "#9e9e9e"];

// ── Section: Key Metrics ──
function KeyMetricsSection() {
  const { data: statsData, isLoading: statsLoading } = useQuery({
    queryKey: ["feedback-stats"],
    queryFn: () => feedbackApi.stats(),
  });

  const { data: infoData, isLoading: infoLoading } = useQuery({
    queryKey: ["admin-info"],
    queryFn: () => adminApi.getInfo(),
  });

  if (statsLoading || infoLoading) return <CircularProgress />;

  const stats = statsData?.data;
  const info = infoData?.data;
  const total = stats?.total ?? 0;
  const positive = stats?.positive ?? 0;
  const negative = stats?.negative ?? 0;
  const positiveRate = total > 0 ? ((positive / total) * 100).toFixed(1) : "0";
  const negativeRate = total > 0 ? ((negative / total) * 100).toFixed(1) : "0";

  const metrics = [
    {
      label: "총 피드백 수",
      value: String(total),
      icon: <FeedbackIcon sx={{ fontSize: 40 }} color="primary" />,
    },
    {
      label: "긍정 비율",
      value: `${positiveRate}%`,
      icon: <TrendingUpIcon sx={{ fontSize: 40, color: "#4caf50" }} />,
    },
    {
      label: "부정 비율",
      value: `${negativeRate}%`,
      icon: <TrendingDownIcon sx={{ fontSize: 40, color: "#f44336" }} />,
    },
    {
      label: "문서 수",
      value: String(info?.document_count ?? 0),
      icon: <FolderIcon sx={{ fontSize: 40 }} color="primary" />,
    },
  ];

  return (
    <Grid container spacing={2}>
      {metrics.map((m) => (
        <Grid key={m.label} size={{ xs: 12, sm: 6, md: 3 }}>
          <Card variant="outlined">
            <CardContent sx={{ display: "flex", alignItems: "center", gap: 2, py: 2 }}>
              {m.icon}
              <Box>
                <Typography variant="body2" color="text.secondary">
                  {m.label}
                </Typography>
                <Typography variant="h5" sx={{ fontWeight: 700 }}>
                  {m.value}
                </Typography>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      ))}
    </Grid>
  );
}

// ── Section: Feedback Pie Chart ──
function FeedbackChartSection() {
  const { data, isLoading, error } = useQuery({
    queryKey: ["feedback-stats"],
    queryFn: () => feedbackApi.stats(),
  });

  if (isLoading) return <CircularProgress />;
  if (error) return <Alert severity="error">피드백 통계를 불러올 수 없습니다.</Alert>;

  const stats = data?.data;
  const chartData = [
    { name: "긍정", value: stats?.positive ?? 0 },
    { name: "부정", value: stats?.negative ?? 0 },
    { name: "보통", value: stats?.neutral ?? 0 },
  ];

  const hasData = chartData.some((d) => d.value > 0);

  return (
    <Card variant="outlined">
      <CardContent>
        <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>
          피드백 통계
        </Typography>
        {hasData ? (
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie data={chartData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={100} label>
                {chartData.map((_, index) => (
                  <Cell key={`cell-${index}`} fill={PIE_COLORS[index % PIE_COLORS.length]} />
                ))}
              </Pie>
              <RechartsTooltip />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        ) : (
          <Box sx={{ height: 300, display: "flex", alignItems: "center", justifyContent: "center" }}>
            <Typography color="text.secondary">피드백 데이터가 없습니다.</Typography>
          </Box>
        )}
      </CardContent>
    </Card>
  );
}

// ── Section: Recent Feedback Table ──
interface FeedbackItem {
  id: string;
  message_id: string;
  session_id: string;
  rating: number;
  comment?: string;
  corrected_answer?: string;
  created_at: string;
}

function RecentFeedbackSection() {
  const { data, isLoading, error } = useQuery({
    queryKey: ["feedback-list"],
    queryFn: () => feedbackApi.list(20),
  });

  if (isLoading) return <CircularProgress />;
  if (error) return <Alert severity="error">피드백 목록을 불러올 수 없습니다.</Alert>;

  const feedbacks: FeedbackItem[] = data?.data?.feedbacks ?? [];

  const getRatingIcon = (rating: number) => {
    if (rating === 1) return <ThumbUpIcon sx={{ color: "#4caf50" }} fontSize="small" />;
    if (rating === -1) return <ThumbDownIcon sx={{ color: "#f44336" }} fontSize="small" />;
    return <RemoveIcon sx={{ color: "#9e9e9e" }} fontSize="small" />;
  };

  return (
    <Card variant="outlined">
      <CardContent>
        <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>
          최근 피드백
        </Typography>
        {feedbacks.length > 0 ? (
          <TableContainer component={Paper} variant="outlined">
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>메시지 ID</TableCell>
                  <TableCell align="center">평점</TableCell>
                  <TableCell>코멘트</TableCell>
                  <TableCell>일시</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {feedbacks.map((fb) => (
                  <TableRow key={fb.id}>
                    <TableCell>
                      <Typography variant="body2" sx={{ fontFamily: "monospace", fontSize: 12 }}>
                        {fb.message_id.slice(0, 12)}...
                      </Typography>
                    </TableCell>
                    <TableCell align="center">{getRatingIcon(fb.rating)}</TableCell>
                    <TableCell>
                      <Typography variant="body2" noWrap sx={{ maxWidth: 300 }}>
                        {fb.comment || "-"}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Typography variant="body2" color="text.secondary">
                        {new Date(fb.created_at).toLocaleString("ko-KR")}
                      </Typography>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        ) : (
          <Typography color="text.secondary">피드백이 없습니다.</Typography>
        )}
      </CardContent>
    </Card>
  );
}

// ── Section: System Health ──
function SystemHealthSection() {
  const { data, isLoading, error } = useQuery({
    queryKey: ["health"],
    queryFn: () => api.get("/health"),
    refetchInterval: 30000,
  });

  if (isLoading) return <CircularProgress />;
  if (error)
    return (
      <Card variant="outlined">
        <CardContent>
          <Typography variant="h6" sx={{ mb: 1, fontWeight: 600 }}>
            시스템 상태
          </Typography>
          <Alert severity="error">시스템에 연결할 수 없습니다.</Alert>
        </CardContent>
      </Card>
    );

  const health = data?.data;

  return (
    <Card variant="outlined">
      <CardContent>
        <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>
          시스템 상태
        </Typography>
        <Box sx={{ display: "flex", alignItems: "center", gap: 2, mb: 1 }}>
          <CheckCircleIcon sx={{ color: "#4caf50", fontSize: 28 }} />
          <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
            {health?.app_name ?? "Flux RAG"}
          </Typography>
          <Chip label="정상" color="success" size="small" />
        </Box>
        <Typography variant="body2" color="text.secondary">
          버전: {health?.version ?? "-"}
        </Typography>
        <Typography variant="body2" color="text.secondary">
          상태: {health?.status ?? "running"}
        </Typography>
      </CardContent>
    </Card>
  );
}

// ── Section: MCP Tools Status ──
function McpToolsStatusSection() {
  const { data, isLoading, error } = useQuery({
    queryKey: ["mcp-tools"],
    queryFn: () => mcpApi.listTools(),
  });

  if (isLoading) return <CircularProgress />;
  if (error) return <Alert severity="error">MCP 도구 상태를 불러올 수 없습니다.</Alert>;

  const tools: Array<{ name: string; description: string }> = data?.data?.tools ?? [];

  return (
    <Card variant="outlined">
      <CardContent>
        <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>
          MCP 도구 현황
        </Typography>
        {tools.length > 0 ? (
          <List dense disablePadding>
            {tools.map((tool) => (
              <ListItemButton key={tool.name} disableRipple sx={{ cursor: "default", py: 0.5 }}>
                <ListItemIcon sx={{ minWidth: 36 }}>
                  <BuildIcon fontSize="small" color="action" />
                </ListItemIcon>
                <ListItemText
                  primary={tool.name}
                  secondary={tool.description}
                  primaryTypographyProps={{ variant: "body2", fontWeight: 600 }}
                  secondaryTypographyProps={{ variant: "caption" }}
                />
              </ListItemButton>
            ))}
          </List>
        ) : (
          <Typography color="text.secondary" variant="body2">
            등록된 MCP 도구가 없습니다.
          </Typography>
        )}
      </CardContent>
    </Card>
  );
}

// ── Section: Usage Stats (일/주/월 사용량 차트) ──
function UsageStatsSection() {
  const [period, setPeriod] = useState<"day" | "week" | "month">("week");
  const { data, isLoading } = useQuery({
    queryKey: ["usage-stats", period],
    queryFn: () => statsApi.getUsage(period),
  });

  if (isLoading) return <CircularProgress />;
  const stats = data?.data;

  return (
    <Card variant="outlined">
      <CardContent>
        <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center", mb: 2 }}>
          <Typography variant="h6" sx={{ fontWeight: 600 }}>
            사용량 통계
          </Typography>
          <FormControl size="small" sx={{ minWidth: 100 }}>
            <Select value={period} onChange={(e) => setPeriod(e.target.value as "day" | "week" | "month")}>
              <MenuItem value="day">일간</MenuItem>
              <MenuItem value="week">주간</MenuItem>
              <MenuItem value="month">월간</MenuItem>
            </Select>
          </FormControl>
        </Box>
        <Grid container spacing={2} sx={{ mb: 2 }}>
          {[
            { label: "총 세션", value: stats?.total_sessions ?? 0 },
            { label: "총 질의", value: stats?.total_queries ?? 0 },
            { label: "고유 사용자", value: stats?.unique_users ?? 0 },
            { label: "평균 질의/세션", value: stats?.avg_queries_per_session?.toFixed(1) ?? "0" },
          ].map((m) => (
            <Grid key={m.label} size={{ xs: 6, sm: 3 }}>
              <Box sx={{ textAlign: "center" }}>
                <Typography variant="h5" sx={{ fontWeight: 700 }}>{m.value}</Typography>
                <Typography variant="caption" color="text.secondary">{m.label}</Typography>
              </Box>
            </Grid>
          ))}
        </Grid>
        {stats?.daily_breakdown?.length > 0 && (
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={stats.daily_breakdown}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" tick={{ fontSize: 11 }} />
              <YAxis />
              <RechartsTooltip />
              <Bar dataKey="queries" fill="#1976d2" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        )}
      </CardContent>
    </Card>
  );
}

// ── Section: Keyword Stats ──
function KeywordStatsSection() {
  const { data, isLoading } = useQuery({
    queryKey: ["keyword-stats"],
    queryFn: () => statsApi.getKeywords("week", 20),
  });

  if (isLoading) return <CircularProgress />;
  const keywords = data?.data?.keywords ?? [];

  return (
    <Card variant="outlined">
      <CardContent>
        <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>
          인기 키워드
        </Typography>
        <Box sx={{ display: "flex", flexWrap: "wrap", gap: 1 }}>
          {keywords.map((kw: { keyword: string; count: number }, i: number) => (
            <Chip
              key={kw.keyword}
              label={`${kw.keyword} (${kw.count})`}
              size={i < 5 ? "medium" : "small"}
              color={i < 3 ? "primary" : i < 10 ? "default" : "default"}
              variant={i < 3 ? "filled" : "outlined"}
            />
          ))}
          {keywords.length === 0 && (
            <Typography color="text.secondary" variant="body2">키워드 데이터가 없습니다.</Typography>
          )}
        </Box>
      </CardContent>
    </Card>
  );
}

// ── Section: Excel Export ──
function ExcelExportSection() {
  const [period, setPeriod] = useState<"day" | "week" | "month">("month");
  const [downloading, setDownloading] = useState(false);

  const handleExport = async () => {
    setDownloading(true);
    try {
      const res = await statsApi.exportExcel(period);
      const url = window.URL.createObjectURL(new Blob([res.data]));
      const a = document.createElement("a");
      a.href = url;
      a.download = `flux-rag-stats-${period}.xlsx`;
      a.click();
      window.URL.revokeObjectURL(url);
    } catch {
      // ignore
    } finally {
      setDownloading(false);
    }
  };

  return (
    <Card variant="outlined">
      <CardContent sx={{ display: "flex", alignItems: "center", gap: 2 }}>
        <BarChartIcon color="primary" />
        <Typography variant="subtitle1" sx={{ fontWeight: 600, flex: 1 }}>
          통계 Excel 다운로드
        </Typography>
        <FormControl size="small" sx={{ minWidth: 100 }}>
          <Select value={period} onChange={(e) => setPeriod(e.target.value as "day" | "week" | "month")}>
            <MenuItem value="day">일간</MenuItem>
            <MenuItem value="week">주간</MenuItem>
            <MenuItem value="month">월간</MenuItem>
          </Select>
        </FormControl>
        <Button variant="contained" startIcon={<DownloadIcon />} onClick={handleExport} disabled={downloading} size="small">
          {downloading ? "다운로드 중..." : "다운로드"}
        </Button>
      </CardContent>
    </Card>
  );
}

// ── Section: Query Log Search ──
function QueryLogSection() {
  const [keyword, setKeyword] = useState("");
  const [page, setPage] = useState(0);
  const [searchTerm, setSearchTerm] = useState("");

  const { data, isLoading } = useQuery({
    queryKey: ["query-logs", searchTerm, page],
    queryFn: () => logsApi.searchQueries({ keyword: searchTerm || undefined, page: page + 1, page_size: 10 }),
    enabled: true,
  });

  const logs = data?.data;

  return (
    <Card variant="outlined">
      <CardContent>
        <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>
          질의 이력
        </Typography>
        <Box sx={{ display: "flex", gap: 1, mb: 2 }}>
          <TextField
            size="small"
            placeholder="키워드 검색..."
            value={keyword}
            onChange={(e) => setKeyword(e.target.value)}
            onKeyDown={(e) => { if (e.key === "Enter") { setSearchTerm(keyword); setPage(0); } }}
            sx={{ flex: 1 }}
          />
          <Button variant="outlined" startIcon={<SearchIcon />} onClick={() => { setSearchTerm(keyword); setPage(0); }}>
            검색
          </Button>
        </Box>
        {isLoading ? <CircularProgress /> : (
          <>
            <TableContainer component={Paper} variant="outlined">
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>일시</TableCell>
                    <TableCell>세션</TableCell>
                    <TableCell>질의 내용</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {(logs?.queries ?? []).map((q: { timestamp: string; session_id: string; content: string }, i: number) => (
                    <TableRow key={i}>
                      <TableCell sx={{ whiteSpace: "nowrap" }}>
                        <Typography variant="caption">{new Date(q.timestamp).toLocaleString("ko-KR")}</Typography>
                      </TableCell>
                      <TableCell>
                        <Typography variant="caption" sx={{ fontFamily: "monospace" }}>{q.session_id?.slice(0, 8)}</Typography>
                      </TableCell>
                      <TableCell>
                        <Typography variant="body2" noWrap sx={{ maxWidth: 400 }}>{q.content}</Typography>
                      </TableCell>
                    </TableRow>
                  ))}
                  {(logs?.queries ?? []).length === 0 && (
                    <TableRow><TableCell colSpan={3}><Typography color="text.secondary" variant="body2" align="center">결과 없음</Typography></TableCell></TableRow>
                  )}
                </TableBody>
              </Table>
            </TableContainer>
            {(logs?.total ?? 0) > 10 && (
              <TablePagination
                component="div"
                count={logs?.total ?? 0}
                page={page}
                onPageChange={(_, p) => setPage(p)}
                rowsPerPage={10}
                rowsPerPageOptions={[10]}
                labelDisplayedRows={({ from, to, count }) => `${from}-${to} / ${count}`}
              />
            )}
          </>
        )}
      </CardContent>
    </Card>
  );
}

// ── Main: MonitorPage ──
export default function MonitorPage() {
  const [tab, setTab] = useState(0);

  return (
    <Layout title="모니터링 대시보드">
      <Box sx={{ borderBottom: 1, borderColor: "divider", mb: 3 }}>
        <Tabs value={tab} onChange={(_, v) => setTab(v)}>
          <Tab label="개요" />
          <Tab label="사용 통계" />
          <Tab label="로그" />
        </Tabs>
      </Box>

      {tab === 0 && (
        <>
          {/* Section 1: Key Metrics */}
          <Box sx={{ mb: 3 }}>
            <Typography variant="h6" sx={{ mb: 1.5, fontWeight: 600 }}>
              주요 지표
            </Typography>
            <KeyMetricsSection />
          </Box>

          {/* Section 2 & 3: Charts + Table */}
          <Grid container spacing={3} sx={{ mb: 3 }}>
            <Grid size={{ xs: 12, md: 6 }}>
              <FeedbackChartSection />
            </Grid>
            <Grid size={{ xs: 12, md: 6 }}>
              <RecentFeedbackSection />
            </Grid>
          </Grid>

          {/* Section 4 & 5: System Health + MCP Tools */}
          <Grid container spacing={3}>
            <Grid size={{ xs: 12, md: 6 }}>
              <SystemHealthSection />
            </Grid>
            <Grid size={{ xs: 12, md: 6 }}>
              <McpToolsStatusSection />
            </Grid>
          </Grid>
        </>
      )}

      {tab === 1 && (
        <>
          <Box sx={{ mb: 3 }}>
            <ExcelExportSection />
          </Box>
          <Grid container spacing={3} sx={{ mb: 3 }}>
            <Grid size={{ xs: 12, md: 7 }}>
              <UsageStatsSection />
            </Grid>
            <Grid size={{ xs: 12, md: 5 }}>
              <KeywordStatsSection />
            </Grid>
          </Grid>
        </>
      )}

      {tab === 2 && (
        <QueryLogSection />
      )}
    </Layout>
  );
}
