import { useState, useCallback, useEffect } from "react";
import {
  Box,
  Typography,
  Grid,
  CircularProgress,
  Chip,
  TextField,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Paper,
  IconButton,
  Tooltip,
  Divider,
  Stack,
  Avatar,
  InputAdornment,
} from "@mui/material";
import PersonIcon from "@mui/icons-material/Person";
import DeleteIcon from "@mui/icons-material/Delete";
import SettingsIcon from "@mui/icons-material/Settings";
import DescriptionIcon from "@mui/icons-material/Description";
import ThumbUpIcon from "@mui/icons-material/ThumbUp";
import DownloadIcon from "@mui/icons-material/Download";
import HistoryIcon from "@mui/icons-material/History";
import ComputerIcon from "@mui/icons-material/Computer";
import ShieldIcon from "@mui/icons-material/Shield";
import LoginIcon from "@mui/icons-material/Login";
import LogoutIcon from "@mui/icons-material/Logout";
import SearchIcon from "@mui/icons-material/Search";
import RefreshIcon from "@mui/icons-material/Refresh";
import { logsApi } from "../../api/client";

type AuditSeverity = "INFO" | "WARNING" | "CRITICAL";
type AuditEventType =
  | "login"
  | "logout"
  | "doc_upload"
  | "doc_delete"
  | "feedback"
  | "settings"
  | "security"
  | "system";

interface AuditEntry {
  id: string;
  timestamp: string;
  user: string;
  eventType: AuditEventType;
  action: string;
  details: string;
  severity: AuditSeverity;
  ip?: string;
  resource?: string;
}

const AUDIT_EVENT_CONFIG: Record<
  AuditEventType,
  { label: string; color: string; bgColor: string; icon: React.ReactNode }
> = {
  login: {
    label: "로그인",
    color: "#1976d2",
    bgColor: "#e3f2fd",
    icon: <LoginIcon fontSize="small" />,
  },
  logout: {
    label: "로그아웃",
    color: "#5c6bc0",
    bgColor: "#e8eaf6",
    icon: <LogoutIcon fontSize="small" />,
  },
  doc_upload: {
    label: "문서 업로드",
    color: "#388e3c",
    bgColor: "#e8f5e9",
    icon: <DescriptionIcon fontSize="small" />,
  },
  doc_delete: {
    label: "문서 삭제",
    color: "#c62828",
    bgColor: "#ffebee",
    icon: <DeleteIcon fontSize="small" />,
  },
  feedback: {
    label: "피드백",
    color: "#f57c00",
    bgColor: "#fff8e1",
    icon: <ThumbUpIcon fontSize="small" />,
  },
  settings: {
    label: "설정 변경",
    color: "#ef6c00",
    bgColor: "#fff3e0",
    icon: <SettingsIcon fontSize="small" />,
  },
  security: {
    label: "보안 이벤트",
    color: "#b71c1c",
    bgColor: "#ffebee",
    icon: <ShieldIcon fontSize="small" />,
  },
  system: {
    label: "시스템",
    color: "#546e7a",
    bgColor: "#eceff1",
    icon: <ComputerIcon fontSize="small" />,
  },
};


function mapActionToEventType(action: string): AuditEventType {
  const upper = action.toUpperCase();
  if (upper.includes("LOGIN")) return "login";
  if (upper.includes("LOGOUT")) return "logout";
  if (upper.includes("UPLOAD") || upper.includes("INGEST")) return "doc_upload";
  if (upper.includes("DELETE") || upper.includes("REMOVE")) return "doc_delete";
  if (upper.includes("FEEDBACK")) return "feedback";
  if (upper.includes("SETTINGS") || upper.includes("CHANGE") || upper.includes("ROLE")) return "settings";
  if (upper.includes("SECURITY") || upper.includes("GUARDRAIL") || upper.includes("BLOCK")) return "security";
  return "system";
}

function mapActionToSeverity(action: string): AuditSeverity {
  const upper = action.toUpperCase();
  if (upper.includes("FAIL") || upper.includes("BLOCK") || upper.includes("VIOLATION")) return "CRITICAL";
  if (upper.includes("DELETE") || upper.includes("CHANGE") || upper.includes("ROLLBACK")) return "WARNING";
  return "INFO";
}


function AuditLogTab() {
  const PAGE_SIZE = 10;
  const [entries, setEntries] = useState<AuditEntry[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(false);
  const [searchText, setSearchText] = useState("");
  const [userFilter, setUserFilter] = useState("");
  const [eventTypeFilter, setEventTypeFilter] = useState<AuditEventType[]>([]);
  const [severityFilter, setSeverityFilter] = useState<AuditSeverity[]>([]);
  const [dateFrom, setDateFrom] = useState("");
  const [dateTo, setDateTo] = useState("");
  const [page, setPage] = useState(0);
  const [detailEntry, setDetailEntry] = useState<AuditEntry | null>(null);

  const eventTypes = Object.keys(AUDIT_EVENT_CONFIG) as AuditEventType[];
  const severities: AuditSeverity[] = ["INFO", "WARNING", "CRITICAL"];

  const fetchLogs = useCallback(async () => {
    setLoading(true);
    try {
      const resp = await logsApi.searchAccess({
        username: userFilter || undefined,
        start_date: dateFrom || undefined,
        end_date: dateTo || undefined,
        page: page + 1,
        page_size: 100,
      });
      const data = resp.data;
      const mapped: AuditEntry[] = (data.logs || []).map((log: any, idx: number) => ({
        id: `a${idx}`,
        timestamp: log.timestamp,
        user: log.username || log.user_id || "system",
        eventType: mapActionToEventType(log.action),
        action: log.action,
        details: log.details || "",
        severity: mapActionToSeverity(log.action),
        ip: log.ip_address || "",
        resource: log.resource || "",
      }));
      setEntries(mapped);
      setTotal(data.total || mapped.length);
    } catch (err) {
      console.error("Failed to fetch audit logs:", err);
      setEntries([]);
      setTotal(0);
    } finally {
      setLoading(false);
    }
  }, [userFilter, dateFrom, dateTo, page]);

  useEffect(() => { fetchLogs(); }, [fetchLogs]);

  // Client-side filtering (search text, event type, severity)
  const filtered = entries.filter((e) => {
    if (
      searchText &&
      ![e.user, e.action, e.details, e.ip ?? "", e.resource ?? ""]
        .join(" ")
        .toLowerCase()
        .includes(searchText.toLowerCase())
    )
      return false;
    if (eventTypeFilter.length > 0 && !eventTypeFilter.includes(e.eventType))
      return false;
    if (severityFilter.length > 0 && !severityFilter.includes(e.severity))
      return false;
    return true;
  });

  const totalPages = Math.ceil(filtered.length / PAGE_SIZE);
  const pageItems = filtered.slice(page * PAGE_SIZE, page * PAGE_SIZE + PAGE_SIZE);

  const resetFilters = () => {
    setSearchText("");
    setUserFilter("");
    setEventTypeFilter([]);
    setSeverityFilter([]);
    setDateFrom("");
    setDateTo("");
    setPage(0);
  };

  const exportCsv = () => {
    const header = "id,timestamp,user,eventType,action,details,severity,ip,resource\n";
    const rows = filtered
      .map((e) =>
        [
          e.id,
          e.timestamp,
          e.user,
          e.eventType,
          `"${e.action}"`,
          `"${e.details.replace(/"/g, '""')}"`,
          e.severity,
          e.ip ?? "",
          e.resource ?? "",
        ].join(",")
      )
      .join("\n");
    const blob = new Blob([header + rows], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `audit_log_${new Date().toISOString().slice(0, 10)}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const formatTs = (ts: string) => {
    const d = new Date(ts);
    return d.toLocaleString("ko-KR", {
      month: "2-digit",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });
  };

  const toggleEventType = (et: AuditEventType) => {
    setPage(0);
    setEventTypeFilter((prev) =>
      prev.includes(et) ? prev.filter((x) => x !== et) : [...prev, et]
    );
  };

  const toggleSeverity = (s: AuditSeverity) => {
    setPage(0);
    setSeverityFilter((prev) =>
      prev.includes(s) ? prev.filter((x) => x !== s) : [...prev, s]
    );
  };

  return (
    <Box>
      {/* Header */}
      <Box sx={{ display: "flex", alignItems: "center", gap: 2, mb: 3 }}>
        <HistoryIcon color="primary" />
        <Typography variant="h6" fontWeight={700}>
          감사 로그
        </Typography>
        <Chip
          label={`${filtered.length}건`}
          color="primary"
          size="small"
          sx={{ fontWeight: 700 }}
        />
        {loading && <CircularProgress size={18} />}
        <Box sx={{ flex: 1 }} />
        <Button
          startIcon={<DownloadIcon />}
          variant="outlined"
          size="small"
          onClick={exportCsv}
        >
          CSV 내보내기
        </Button>
        <Tooltip title="새로고침">
          <IconButton size="small" onClick={() => { resetFilters(); fetchLogs(); }}>
            <RefreshIcon />
          </IconButton>
        </Tooltip>
      </Box>

      {/* Filters */}
      <Paper variant="outlined" sx={{ p: 2, mb: 3, borderRadius: 2 }}>
        <Grid container spacing={2} alignItems="flex-start">
          {/* Search */}
          <Grid size={{ xs: 12, md: 4 }}>
            <TextField
              size="small"
              fullWidth
              placeholder="전체 검색..."
              value={searchText}
              onChange={(e) => { setSearchText(e.target.value); setPage(0); }}
              InputProps={{
                startAdornment: (
                  <InputAdornment position="start">
                    <SearchIcon fontSize="small" />
                  </InputAdornment>
                ),
              }}
            />
          </Grid>
          {/* User filter */}
          <Grid size={{ xs: 12, md: 3 }}>
            <TextField
              size="small"
              fullWidth
              placeholder="사용자 필터..."
              value={userFilter}
              onChange={(e) => { setUserFilter(e.target.value); setPage(0); }}
              InputProps={{
                startAdornment: (
                  <InputAdornment position="start">
                    <PersonIcon fontSize="small" />
                  </InputAdornment>
                ),
              }}
            />
          </Grid>
          {/* Date from */}
          <Grid size={{ xs: 6, md: 2 }}>
            <TextField
              size="small"
              fullWidth
              type="date"
              label="시작일"
              value={dateFrom}
              onChange={(e) => { setDateFrom(e.target.value); setPage(0); }}
              InputLabelProps={{ shrink: true }}
            />
          </Grid>
          {/* Date to */}
          <Grid size={{ xs: 6, md: 2 }}>
            <TextField
              size="small"
              fullWidth
              type="date"
              label="종료일"
              value={dateTo}
              onChange={(e) => { setDateTo(e.target.value); setPage(0); }}
              InputLabelProps={{ shrink: true }}
            />
          </Grid>
        </Grid>

        {/* Event type chips */}
        <Box sx={{ mt: 2 }}>
          <Typography variant="caption" color="text.secondary" sx={{ mr: 1 }}>
            이벤트 유형:
          </Typography>
          {eventTypes.map((et) => {
            const cfg = AUDIT_EVENT_CONFIG[et];
            const active = eventTypeFilter.includes(et);
            return (
              <Chip
                key={et}
                label={cfg.label}
                size="small"
                icon={
                  <Box
                    component="span"
                    sx={{ color: active ? "#fff" : cfg.color, display: "flex" }}
                  >
                    {cfg.icon}
                  </Box>
                }
                onClick={() => toggleEventType(et)}
                sx={{
                  mr: 0.5,
                  mb: 0.5,
                  fontWeight: 600,
                  bgcolor: active ? cfg.color : cfg.bgColor,
                  color: active ? "#fff" : cfg.color,
                  border: `1px solid ${cfg.color}`,
                  cursor: "pointer",
                  "&:hover": { opacity: 0.85 },
                }}
              />
            );
          })}
        </Box>

        {/* Severity chips */}
        <Box sx={{ mt: 1 }}>
          <Typography variant="caption" color="text.secondary" sx={{ mr: 1 }}>
            심각도:
          </Typography>
          {severities.map((s) => {
            const active = severityFilter.includes(s);
            const colorMap: Record<AuditSeverity, string> = {
              INFO: "#1976d2",
              WARNING: "#f57c00",
              CRITICAL: "#b71c1c",
            };
            const bgMap: Record<AuditSeverity, string> = {
              INFO: "#e3f2fd",
              WARNING: "#fff8e1",
              CRITICAL: "#ffebee",
            };
            return (
              <Chip
                key={s}
                label={s}
                size="small"
                onClick={() => toggleSeverity(s)}
                sx={{
                  mr: 0.5,
                  fontWeight: 700,
                  fontSize: "0.7rem",
                  bgcolor: active ? colorMap[s] : bgMap[s],
                  color: active ? "#fff" : colorMap[s],
                  border: `1px solid ${colorMap[s]}`,
                  cursor: "pointer",
                  "&:hover": { opacity: 0.85 },
                }}
              />
            );
          })}
        </Box>
      </Paper>

      {/* Timeline */}
      <Box>
        {loading ? (
          <Box sx={{ textAlign: "center", py: 6 }}>
            <CircularProgress size={32} />
            <Typography color="text.secondary" sx={{ mt: 1 }}>감사 로그를 불러오는 중...</Typography>
          </Box>
        ) : pageItems.length === 0 ? (
          <Paper variant="outlined" sx={{ p: 4, textAlign: "center", borderRadius: 2 }}>
            <HistoryIcon sx={{ fontSize: 48, color: "text.disabled", mb: 1 }} />
            <Typography color="text.secondary">감사 로그가 없습니다</Typography>
            <Typography variant="caption" color="text.disabled">시스템 사용 시 자동으로 기록됩니다</Typography>
          </Paper>
        ) : (
          pageItems.map((entry, idx) => {
            const cfg = AUDIT_EVENT_CONFIG[entry.eventType];
            const isLast = idx === pageItems.length - 1;
            return (
              <Box
                key={entry.id}
                sx={{ display: "flex", gap: 0, mb: isLast ? 0 : 0 }}
              >
                {/* Timeline line + dot */}
                <Box
                  sx={{
                    display: "flex",
                    flexDirection: "column",
                    alignItems: "center",
                    mr: 2,
                    minWidth: 36,
                  }}
                >
                  <Avatar
                    sx={{
                      width: 36,
                      height: 36,
                      bgcolor: cfg.bgColor,
                      color: cfg.color,
                      border: `2px solid ${cfg.color}`,
                      flexShrink: 0,
                    }}
                  >
                    {cfg.icon}
                  </Avatar>
                  {!isLast && (
                    <Box
                      sx={{
                        width: 2,
                        flex: 1,
                        minHeight: 16,
                        bgcolor: "divider",
                        my: 0.5,
                      }}
                    />
                  )}
                </Box>

                {/* Content */}
                <Paper
                  variant="outlined"
                  sx={{
                    flex: 1,
                    p: 1.5,
                    mb: 1.5,
                    borderRadius: 2,
                    cursor: "pointer",
                    borderColor:
                      entry.severity === "CRITICAL"
                        ? "#b71c1c"
                        : entry.severity === "WARNING"
                        ? "#f57c00"
                        : "divider",
                    bgcolor:
                      entry.severity === "CRITICAL"
                        ? "#fff5f5"
                        : entry.severity === "WARNING"
                        ? "#fffbf0"
                        : "background.paper",
                    "&:hover": { boxShadow: 2 },
                    transition: "box-shadow 0.15s",
                  }}
                  onClick={() => setDetailEntry(entry)}
                >
                  <Box sx={{ display: "flex", alignItems: "center", gap: 1, flexWrap: "wrap" }}>
                    <Chip
                      label={cfg.label}
                      size="small"
                      sx={{
                        bgcolor: cfg.bgColor,
                        color: cfg.color,
                        fontWeight: 700,
                        fontSize: "0.7rem",
                        height: 20,
                      }}
                    />
                    <Chip
                      label={entry.severity}
                      size="small"
                      sx={{
                        bgcolor: entry.severity === "CRITICAL" ? "#b71c1c" : entry.severity === "WARNING" ? "#f57c00" : "#e3f2fd",
                        color: entry.severity === "INFO" ? "#1976d2" : "#fff",
                        fontWeight: 700,
                        fontSize: "0.65rem",
                        height: 18,
                      }}
                    />
                    <Typography variant="body2" fontWeight={700} sx={{ flex: 1, minWidth: 0 }}>
                      {entry.action}
                    </Typography>
                    <Typography variant="caption" color="text.secondary" sx={{ whiteSpace: "nowrap" }}>
                      {formatTs(entry.timestamp)}
                    </Typography>
                  </Box>
                  <Box sx={{ display: "flex", alignItems: "center", gap: 1, mt: 0.5 }}>
                    <PersonIcon sx={{ fontSize: 13, color: "text.secondary" }} />
                    <Typography variant="caption" color="text.secondary" fontWeight={600}>
                      {entry.user}
                    </Typography>
                    {entry.ip && (
                      <>
                        <Typography variant="caption" color="text.disabled">
                          •
                        </Typography>
                        <Typography variant="caption" color="text.disabled">
                          {entry.ip}
                        </Typography>
                      </>
                    )}
                    {entry.resource && (
                      <>
                        <Typography variant="caption" color="text.disabled">
                          •
                        </Typography>
                        <Typography
                          variant="caption"
                          color="primary"
                          sx={{ fontStyle: "italic" }}
                        >
                          {entry.resource}
                        </Typography>
                      </>
                    )}
                  </Box>
                  <Typography
                    variant="caption"
                    color="text.secondary"
                    sx={{ display: "block", mt: 0.25, lineHeight: 1.4 }}
                  >
                    {entry.details}
                  </Typography>
                </Paper>
              </Box>
            );
          })
        )}
      </Box>

      {/* Pagination */}
      {totalPages > 1 && (
        <Box sx={{ display: "flex", justifyContent: "center", alignItems: "center", gap: 1, mt: 2 }}>
          <Button
            size="small"
            disabled={page === 0}
            onClick={() => setPage((p) => p - 1)}
          >
            이전
          </Button>
          {Array.from({ length: totalPages }, (_, i) => (
            <Button
              key={i}
              size="small"
              variant={page === i ? "contained" : "outlined"}
              onClick={() => setPage(i)}
              sx={{ minWidth: 36 }}
            >
              {i + 1}
            </Button>
          ))}
          <Button
            size="small"
            disabled={page >= totalPages - 1}
            onClick={() => setPage((p) => p + 1)}
          >
            다음
          </Button>
        </Box>
      )}

      {/* Detail Dialog */}
      <Dialog
        open={!!detailEntry}
        onClose={() => setDetailEntry(null)}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle sx={{ display: "flex", alignItems: "center", gap: 1 }}>
          <HistoryIcon color="primary" />
          감사 로그 상세
        </DialogTitle>
        <DialogContent dividers>
          {detailEntry && (() => {
            const cfg = AUDIT_EVENT_CONFIG[detailEntry.eventType];
            return (
              <Stack spacing={1.5}>
                <Box sx={{ display: "flex", gap: 1 }}>
                  <Chip
                    label={cfg.label}
                    size="small"
                    sx={{ bgcolor: cfg.bgColor, color: cfg.color, fontWeight: 700 }}
                  />
                  <Chip
                    label={detailEntry.severity}
                    size="small"
                    color={
                      detailEntry.severity === "CRITICAL"
                        ? "error"
                        : detailEntry.severity === "WARNING"
                        ? "warning"
                        : "info"
                    }
                    sx={{ fontWeight: 700 }}
                  />
                </Box>
                <Divider />
                {[
                  { label: "로그 ID", value: detailEntry.id },
                  {
                    label: "타임스탬프",
                    value: new Date(detailEntry.timestamp).toLocaleString("ko-KR"),
                  },
                  { label: "사용자", value: detailEntry.user },
                  { label: "액션", value: detailEntry.action },
                  { label: "IP 주소", value: detailEntry.ip ?? "N/A" },
                  { label: "대상 리소스", value: detailEntry.resource ?? "N/A" },
                ].map(({ label, value }) => (
                  <Box key={label} sx={{ display: "flex", gap: 1 }}>
                    <Typography
                      variant="caption"
                      color="text.secondary"
                      sx={{ minWidth: 100, fontWeight: 600 }}
                    >
                      {label}
                    </Typography>
                    <Typography variant="caption" sx={{ wordBreak: "break-all" }}>
                      {value}
                    </Typography>
                  </Box>
                ))}
                <Divider />
                <Box>
                  <Typography variant="caption" color="text.secondary" fontWeight={600}>
                    상세 내용
                  </Typography>
                  <Typography
                    variant="body2"
                    sx={{
                      mt: 0.5,
                      p: 1.5,
                      bgcolor: "grey.50",
                      borderRadius: 1,
                      fontSize: "0.8125rem",
                      lineHeight: 1.6,
                      whiteSpace: "pre-wrap",
                    }}
                  >
                    {detailEntry.details}
                  </Typography>
                </Box>
              </Stack>
            );
          })()}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDetailEntry(null)}>닫기</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}

export default AuditLogTab;
