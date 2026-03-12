import { useState, useEffect, useCallback } from "react";
import { useSearchParams } from "react-router-dom";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  Box,
  Typography,
  Card,
  CardContent,
  Grid,
  CircularProgress,
  Alert,
  Chip,
  TextField,
  Button,
  Accordion,
  AccordionSummary,
  AccordionDetails,
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
  IconButton,
  Select,
  MenuItem,
  Switch,
  FormControlLabel,
  FormControl,
  InputLabel,
  Tooltip,
  Slider,
  Divider,
  LinearProgress,
  Stack,
  Avatar,
  Checkbox,
  TableSortLabel,
  InputAdornment,
  Collapse,
  Badge,
  Popover,
  List,
  ListItem,
  ListItemAvatar,
  ListItemText,
  ListSubheader,
  ListItemButton,
  ListItemIcon,
} from "@mui/material";
import InfoIcon from "@mui/icons-material/Info";
import HelpOutlineIcon from "@mui/icons-material/HelpOutline";
import StorageIcon from "@mui/icons-material/Storage";
import AppRegistrationIcon from "@mui/icons-material/AppRegistration";
import FolderIcon from "@mui/icons-material/Folder";
import GroupWorkIcon from "@mui/icons-material/GroupWork";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import BuildIcon from "@mui/icons-material/Build";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import AccountTreeIcon from "@mui/icons-material/AccountTree";
import StarIcon from "@mui/icons-material/Star";
import AddIcon from "@mui/icons-material/Add";
import EditIcon from "@mui/icons-material/Edit";
import DeleteIcon from "@mui/icons-material/Delete";
import CheckCircleIcon from "@mui/icons-material/CheckCircle";
import ErrorIcon from "@mui/icons-material/Error";
import SecurityIcon from "@mui/icons-material/Security";
import MemoryIcon from "@mui/icons-material/Memory";
import PersonIcon from "@mui/icons-material/Person";
import CachedIcon from "@mui/icons-material/Cached";
import DeleteSweepIcon from "@mui/icons-material/DeleteSweep";
import ScienceIcon from "@mui/icons-material/Science";
import SpeedIcon from "@mui/icons-material/Speed";
import ContentCopyIcon from "@mui/icons-material/ContentCopy";
import DashboardIcon from "@mui/icons-material/Dashboard";
import DescriptionIcon from "@mui/icons-material/Description";
import ThumbUpIcon from "@mui/icons-material/ThumbUp";
import ThumbDownIcon from "@mui/icons-material/ThumbDown";
import RemoveCircleOutlineIcon from "@mui/icons-material/RemoveCircleOutline";
import WarningAmberIcon from "@mui/icons-material/WarningAmber";
import AccessTimeIcon from "@mui/icons-material/AccessTime";
import QuestionAnswerIcon from "@mui/icons-material/QuestionAnswer";
import SearchIcon from "@mui/icons-material/Search";
import FilterListIcon from "@mui/icons-material/FilterList";
import LockResetIcon from "@mui/icons-material/LockReset";
import GroupIcon from "@mui/icons-material/Group";
import RefreshIcon from "@mui/icons-material/Refresh";
import VisibilityIcon from "@mui/icons-material/Visibility";
import BlockIcon from "@mui/icons-material/Block";
import CheckIcon from "@mui/icons-material/Check";
import KeyIcon from "@mui/icons-material/Key";
import SettingsIcon from "@mui/icons-material/Settings";
import NotificationsIcon from "@mui/icons-material/Notifications";
import NotificationsNoneIcon from "@mui/icons-material/NotificationsNone";
import FiberManualRecordIcon from "@mui/icons-material/FiberManualRecord";
import DownloadIcon from "@mui/icons-material/Download";
import HistoryIcon from "@mui/icons-material/History";
import ComputerIcon from "@mui/icons-material/Computer";
import ShieldIcon from "@mui/icons-material/Shield";
import LoginIcon from "@mui/icons-material/Login";
import LogoutIcon from "@mui/icons-material/Logout";
import DocumentScannerIcon from "@mui/icons-material/DocumentScanner";
import TuneIcon from "@mui/icons-material/Tune";
import ScienceOutlinedIcon from "@mui/icons-material/ScienceOutlined";
import { adminApi, mcpApi, workflowsApi, guardrailsApi, authApi, governanceApi, feedbackApi, statsApi, ocrApi, logsApi } from "../api/client";
import Layout from "../components/Layout";
import CustomToolBuilder from "../components/CustomToolBuilder";
import ContentManager from "../components/ContentManager";
import { PasswordConfirmDialog } from "../components/chat/ChatDialogs";
import PromptSimulatorTab from "../components/admin/PromptSimulatorTab";
import BatchEvaluatorTab from "../components/admin/BatchEvaluatorTab";
import NewSystemSettingsTab from "../components/admin/SystemSettingsTab";
import FeedbackDrilldownDialog from "../components/admin/FeedbackDrilldownDialog";
import ReportTemplateTab from "../components/admin/ReportTemplateTab";
import CacheManagementTab from "../components/admin/CacheManagementTab";
import LLMPlaygroundTab from "../components/admin/LLMPlaygroundTab";
import OcrTestTab from "../components/admin/OcrTestTab";
import AuditLogTab from "../components/admin/AuditLogTab";
import ExportCenterTab from "../components/admin/ExportCenterTab";

// ── Tab: Dashboard ──
function DashboardTab() {
  const { data: infoData, isLoading: infoLoading } = useQuery({
    queryKey: ["admin-info"],
    queryFn: () => adminApi.getInfo(),
  });

  const { data: usersData, isLoading: usersLoading } = useQuery({
    queryKey: ["auth-users"],
    queryFn: () => authApi.listUsers(),
  });

  const { data: metricsData, isLoading: metricsLoading } = useQuery({
    queryKey: ["admin-system-metrics"],
    queryFn: () => adminApi.getSystemMetrics(),
    refetchInterval: 30_000,
  });

  const { data: feedbackData } = useQuery({
    queryKey: ["feedback-list-dashboard"],
    queryFn: () => feedbackApi.list(5),
  });

  const { data: usageData } = useQuery({
    queryKey: ["stats-usage-dashboard"],
    queryFn: () => statsApi.getUsage("day"),
  });

  const info = infoData?.data as {
    app_name?: string;
    document_count?: number;
    session_count?: number;
  } | undefined;

  const users: unknown[] = usersData?.data ?? [];
  const metrics = (metricsData?.data as { metrics?: Record<string, unknown> })?.metrics as {
    cpu?: number;
    memory?: number;
    disk?: number;
    avg_response_time_ms?: number;
    cache_hit_rate?: number;
    active_sessions?: number;
    status?: string;
  } | undefined;

  // Today's query count from usage stats
  const todayQueries: number = (() => {
    const usage = usageData?.data as { daily_breakdown?: Array<{ date?: string; queries?: number }> } | undefined;
    if (usage?.daily_breakdown && usage.daily_breakdown.length > 0) {
      const today = new Date().toISOString().slice(0, 10);
      const todayEntry = usage.daily_breakdown.find((d) => d.date === today);
      if (todayEntry) return todayEntry.queries ?? 0;
      // fallback: last entry
      return usage.daily_breakdown[usage.daily_breakdown.length - 1]?.queries ?? 0;
    }
    return 0;
  })();

  const systemStatus: "healthy" | "warning" | "error" = (() => {
    const cpu = metrics?.cpu ?? 0;
    const mem = metrics?.memory ?? 0;
    if (cpu > 90 || mem > 90) return "error";
    if (cpu > 70 || mem > 70) return "warning";
    return "healthy";
  })();

  const statusLabels: Record<string, string> = {
    healthy: "정상",
    warning: "경고",
    error: "오류",
  };

  const summaryCards = [
    {
      label: "총 문서 수",
      value: infoLoading ? "..." : String(info?.document_count ?? 0),
      icon: <DescriptionIcon sx={{ fontSize: 32 }} />,
      color: "#1976d2",
      bg: "#e3f2fd",
    },
    {
      label: "총 사용자 수",
      value: usersLoading ? "..." : String(users.length),
      icon: <PersonIcon sx={{ fontSize: 32 }} />,
      color: "#388e3c",
      bg: "#e8f5e9",
    },
    {
      label: "오늘 질의 수",
      value: String(todayQueries),
      icon: <QuestionAnswerIcon sx={{ fontSize: 32 }} />,
      color: "#f57c00",
      bg: "#fff3e0",
    },
    {
      label: "시스템 상태",
      value: statusLabels[systemStatus],
      icon: metricsLoading ? (
        <CircularProgress size={28} />
      ) : systemStatus === "healthy" ? (
        <CheckCircleIcon sx={{ fontSize: 32 }} />
      ) : systemStatus === "warning" ? (
        <WarningAmberIcon sx={{ fontSize: 32 }} />
      ) : (
        <ErrorIcon sx={{ fontSize: 32 }} />
      ),
      color:
        systemStatus === "healthy"
          ? "#2e7d32"
          : systemStatus === "warning"
          ? "#e65100"
          : "#c62828",
      bg:
        systemStatus === "healthy"
          ? "#e8f5e9"
          : systemStatus === "warning"
          ? "#fff3e0"
          : "#ffebee",
    },
  ];

  const recentFeedbacks: Array<{
    id: string;
    rating: number;
    comment?: string;
    query?: string;
    created_at?: string;
  }> = (() => {
    const raw = feedbackData?.data;
    if (Array.isArray(raw)) return raw.slice(0, 5);
    if (raw?.feedbacks && Array.isArray(raw.feedbacks)) return raw.feedbacks.slice(0, 5);
    return [];
  })();

  const { data: errorsData } = useQuery({
    queryKey: ["admin-recent-errors"],
    queryFn: () => adminApi.getRecentErrors(5),
    refetchInterval: 60_000,
  });

  const recentErrors: Array<{ timestamp: string; message: string; level: string }> = (() => {
    const raw = errorsData?.data;
    if (raw?.errors && Array.isArray(raw.errors)) return raw.errors.slice(0, 5);
    return [];
  })();

  // Feedback drilldown dialog state
  const [drilldownFeedbackId, setDrilldownFeedbackId] = useState<string | null>(null);

  const avgResponseMs = metrics?.avg_response_time_ms ?? null;
  const cacheHitRate = metrics?.cache_hit_rate ?? null;
  const activeSessions = metrics?.active_sessions ?? info?.session_count ?? null;

  const quickStats = [
    {
      label: "평균 응답시간",
      value: avgResponseMs !== null ? `${avgResponseMs.toFixed(0)} ms` : "—",
      // NOTE: avg_response_time_ms needs to be added to /admin/system-metrics response
      icon: <SpeedIcon fontSize="small" color="action" />,
    },
    {
      label: "캐시 히트율",
      value: cacheHitRate !== null ? `${(cacheHitRate * 100).toFixed(1)}%` : "—",
      // NOTE: cache_hit_rate needs to be added to /admin/system-metrics response
      icon: <CachedIcon fontSize="small" color="action" />,
    },
    {
      label: "활성 세션",
      value: activeSessions !== null ? String(activeSessions) : "—",
      icon: <GroupWorkIcon fontSize="small" color="action" />,
    },
  ];

  return (
    <Box>
      {/* Summary Cards */}
      <Grid container spacing={2} sx={{ mb: 3 }}>
        {summaryCards.map((card) => (
          <Grid key={card.label} size={{ xs: 12, sm: 6, md: 3 }}>
            <Card
              variant="outlined"
              sx={{
                borderLeft: `4px solid ${card.color}`,
                transition: "box-shadow 0.2s",
                "&:hover": { boxShadow: 4 },
              }}
            >
              <CardContent sx={{ display: "flex", alignItems: "center", gap: 2, py: 2 }}>
                <Avatar
                  sx={{
                    bgcolor: card.bg,
                    color: card.color,
                    width: 52,
                    height: 52,
                  }}
                >
                  {card.icon}
                </Avatar>
                <Box>
                  <Typography variant="body2" color="text.secondary" sx={{ mb: 0.25 }}>
                    {card.label}
                  </Typography>
                  <Typography variant="h5" fontWeight={700} color={card.color}>
                    {card.value}
                  </Typography>
                </Box>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>

      {/* Quick Stats Row */}
      <Card variant="outlined" sx={{ mb: 3, "&:hover": { boxShadow: 2 }, transition: "box-shadow 0.2s" }}>
        <CardContent>
          <Typography variant="subtitle1" fontWeight={600} gutterBottom>
            빠른 통계
          </Typography>
          <Grid container spacing={2}>
            {quickStats.map((stat) => (
              <Grid key={stat.label} size={{ xs: 12, sm: 4 }}>
                <Box
                  sx={{
                    display: "flex",
                    alignItems: "center",
                    gap: 1.5,
                    p: 1.5,
                    borderRadius: 1,
                    bgcolor: "action.hover",
                  }}
                >
                  {stat.icon}
                  <Box>
                    <Typography variant="caption" color="text.secondary">
                      {stat.label}
                    </Typography>
                    <Typography variant="body1" fontWeight={600}>
                      {stat.value}
                    </Typography>
                  </Box>
                </Box>
              </Grid>
            ))}
          </Grid>

          {/* Resource usage bars */}
          {metrics && (
            <Box sx={{ mt: 2, display: "flex", flexDirection: "column", gap: 1 }}>
              <Divider sx={{ mb: 1 }} />
              {[
                { label: "CPU", value: metrics.cpu ?? 0 },
                { label: "메모리", value: metrics.memory ?? 0 },
                { label: "디스크", value: metrics.disk ?? 0 },
              ].map((res) => (
                <Box key={res.label} sx={{ display: "flex", alignItems: "center", gap: 2 }}>
                  <Typography variant="caption" sx={{ width: 48, flexShrink: 0, color: "text.secondary" }}>
                    {res.label}
                  </Typography>
                  <LinearProgress
                    variant="determinate"
                    value={Math.min(res.value, 100)}
                    color={res.value > 85 ? "error" : res.value > 65 ? "warning" : "primary"}
                    sx={{ flex: 1, height: 8, borderRadius: 4 }}
                  />
                  <Typography variant="caption" sx={{ width: 40, textAlign: "right", color: "text.secondary" }}>
                    {res.value.toFixed(0)}%
                  </Typography>
                </Box>
              ))}
            </Box>
          )}
        </CardContent>
      </Card>

      {/* Recent Activity */}
      <Grid container spacing={2}>
        {/* Recent Feedback */}
        <Grid size={{ xs: 12, md: 6 }}>
          <Card
            variant="outlined"
            sx={{ height: "100%", "&:hover": { boxShadow: 2 }, transition: "box-shadow 0.2s" }}
          >
            <CardContent>
              <Typography variant="subtitle1" fontWeight={600} gutterBottom>
                최근 피드백 (5건)
              </Typography>
              {recentFeedbacks.length === 0 ? (
                <Typography variant="body2" color="text.secondary" sx={{ py: 2, textAlign: "center" }}>
                  피드백 데이터가 없습니다.
                </Typography>
              ) : (
                <Stack spacing={1}>
                  {recentFeedbacks.map((fb, i) => {
                    const ratingIcon =
                      fb.rating > 0 ? (
                        <ThumbUpIcon fontSize="small" color="success" />
                      ) : fb.rating < 0 ? (
                        <ThumbDownIcon fontSize="small" color="error" />
                      ) : (
                        <RemoveCircleOutlineIcon fontSize="small" color="action" />
                      );
                    return (
                      <Box
                        key={fb.id ?? i}
                        onClick={() => fb.id && setDrilldownFeedbackId(fb.id)}
                        sx={{
                          display: "flex",
                          alignItems: "flex-start",
                          gap: 1.5,
                          p: 1,
                          borderRadius: 1,
                          bgcolor: "action.hover",
                          cursor: fb.id ? "pointer" : "default",
                          "&:hover": fb.id ? { bgcolor: "action.selected" } : {},
                        }}
                      >
                        {ratingIcon}
                        <Box sx={{ flex: 1, minWidth: 0 }}>
                          <Typography
                            variant="body2"
                            sx={{
                              overflow: "hidden",
                              textOverflow: "ellipsis",
                              whiteSpace: "nowrap",
                            }}
                          >
                            {fb.query ?? fb.comment ?? "(내용 없음)"}
                          </Typography>
                          {fb.created_at && (
                            <Typography variant="caption" color="text.secondary">
                              <AccessTimeIcon sx={{ fontSize: 11, verticalAlign: "middle", mr: 0.3 }} />
                              {new Date(fb.created_at).toLocaleString("ko-KR", {
                                month: "2-digit",
                                day: "2-digit",
                                hour: "2-digit",
                                minute: "2-digit",
                              })}
                            </Typography>
                          )}
                        </Box>
                        <Chip
                          label={fb.rating > 0 ? "긍정" : fb.rating < 0 ? "부정" : "중립"}
                          size="small"
                          color={fb.rating > 0 ? "success" : fb.rating < 0 ? "error" : "default"}
                          variant="outlined"
                        />
                      </Box>
                    );
                  })}
                </Stack>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Recent Error Logs */}
        <Grid size={{ xs: 12, md: 6 }}>
          <Card
            variant="outlined"
            sx={{ height: "100%", "&:hover": { boxShadow: 2 }, transition: "box-shadow 0.2s" }}
          >
            <CardContent>
              <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", mb: 1 }}>
                <Typography variant="subtitle1" fontWeight={600}>
                  최근 에러 로그 (5건)
                </Typography>
              </Box>
              {recentErrors.length === 0 ? (
                <Typography variant="body2" color="text.secondary" sx={{ py: 2, textAlign: "center" }}>
                  최근 에러 로그가 없습니다.
                </Typography>
              ) : (
                <Stack spacing={1}>
                  {recentErrors.map((err, i) => {
                    const levelColor =
                      err.level === "ERROR"
                        ? "error"
                        : err.level === "WARNING"
                        ? "warning"
                        : "default";
                    return (
                      <Box
                        key={i}
                        sx={{
                          display: "flex",
                          alignItems: "flex-start",
                          gap: 1.5,
                          p: 1,
                          borderRadius: 1,
                          bgcolor: "action.hover",
                        }}
                      >
                        {err.level === "ERROR" ? (
                          <ErrorIcon fontSize="small" color="error" />
                        ) : err.level === "WARNING" ? (
                          <WarningAmberIcon fontSize="small" color="warning" />
                        ) : (
                          <InfoIcon fontSize="small" color="info" />
                        )}
                        <Box sx={{ flex: 1, minWidth: 0 }}>
                          <Typography
                            variant="body2"
                            sx={{
                              overflow: "hidden",
                              textOverflow: "ellipsis",
                              whiteSpace: "nowrap",
                            }}
                          >
                            {err.message}
                          </Typography>
                          <Typography variant="caption" color="text.secondary">
                            <AccessTimeIcon sx={{ fontSize: 11, verticalAlign: "middle", mr: 0.3 }} />
                            {new Date(err.timestamp).toLocaleString("ko-KR", {
                              month: "2-digit",
                              day: "2-digit",
                              hour: "2-digit",
                              minute: "2-digit",
                            })}
                          </Typography>
                        </Box>
                        <Chip label={err.level} size="small" color={levelColor} variant="outlined" />
                      </Box>
                    );
                  })}
                </Stack>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Feedback Drilldown Dialog */}
      <FeedbackDrilldownDialog
        open={!!drilldownFeedbackId}
        feedbackId={drilldownFeedbackId}
        onClose={() => setDrilldownFeedbackId(null)}
      />
    </Box>
  );
}

// ── Tab: User Management ──
// ── Helpers for UserManagementTab ──
type UserRole = "admin" | "manager" | "expert" | "user" | "viewer";

interface UserRecord {
  id: string;
  username: string;
  full_name?: string;
  email?: string;
  department?: string;
  role: UserRole;
  is_active: boolean;
  last_login?: string;
  created_at?: string;
}

const ROLE_META: Record<string, { label: string; color: "error" | "warning" | "info" | "success" | "default" | "primary" | "secondary" }> = {
  admin:   { label: "관리자",  color: "error" },
  manager: { label: "매니저",  color: "warning" },
  expert:  { label: "전문가",  color: "info" },
  user:    { label: "사용자",  color: "success" },
  viewer:  { label: "뷰어",    color: "default" },
};

const ALL_ROLES: UserRole[] = ["admin", "manager", "expert", "user", "viewer"];

function generateTempPassword(): string {
  const upper = "ABCDEFGHJKLMNPQRSTUVWXYZ";
  const lower = "abcdefghjkmnpqrstuvwxyz";
  const digits = "23456789";
  const special = "@#$!%*?";
  const allChars = upper + lower + digits + special;
  let pw = upper[Math.floor(Math.random() * upper.length)]
    + lower[Math.floor(Math.random() * lower.length)]
    + digits[Math.floor(Math.random() * digits.length)]
    + special[Math.floor(Math.random() * special.length)];
  for (let i = 0; i < 8; i++) pw += allChars[Math.floor(Math.random() * allChars.length)];
  return pw.split("").sort(() => Math.random() - 0.5).join("");
}

type SortKey = "username" | "full_name" | "department" | "role" | "is_active";
type SortDir = "asc" | "desc";

function UserManagementTab() {
  const queryClient = useQueryClient();

  // UI state
  const [searchText, setSearchText]         = useState("");
  const [roleFilter, setRoleFilter]         = useState<string>("all");
  const [sortKey, setSortKey]               = useState<SortKey>("username");
  const [sortDir, setSortDir]               = useState<SortDir>("asc");
  const [selected, setSelected]             = useState<string[]>([]);
  const [bulkRoleOpen, setBulkRoleOpen]     = useState(false);
  const [bulkRole, setBulkRole]             = useState<UserRole>("user");

  // Dialogs
  const [createOpen, setCreateOpen]             = useState(false);
  const [detailUser, setDetailUser]             = useState<UserRecord | null>(null);
  const [deleteConfirmId, setDeleteConfirmId]   = useState<string | null>(null);
  const [resetPwUserId, setResetPwUserId]       = useState<string | null>(null);
  const [generatedPw, setGeneratedPw]           = useState<string>("");

  // Create form
  const [formData, setFormData] = useState({
    username: "", email: "", full_name: "", department: "", role: "user" as UserRole,
  });
  const [formGeneratedPw, setFormGeneratedPw]   = useState("");
  const [createdPwVisible, setCreatedPwVisible] = useState(false);

  const [snack, setSnack] = useState<{ open: boolean; message: string; severity: "success" | "error" }>({
    open: false, message: "", severity: "success",
  });
  const showSnack = (message: string, severity: "success" | "error" = "success") =>
    setSnack({ open: true, message, severity });

  // Data
  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ["auth-users"],
    queryFn: () => authApi.listUsers(),
  });

  const rawUsers: UserRecord[] = (data?.data ?? []) as UserRecord[];

  // Mutations
  const createMutation = useMutation({
    mutationFn: (payload: { username: string; password: string; email?: string; full_name?: string; department?: string; role?: string }) =>
      authApi.createUser(payload),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["auth-users"] });
      setCreateOpen(false);
      setCreatedPwVisible(true);
      showSnack("사용자가 생성되었습니다.");
    },
    onError: () => showSnack("사용자 생성에 실패했습니다.", "error"),
  });

  const updateRoleMutation = useMutation({
    mutationFn: ({ userId, role }: { userId: string; role: string }) => authApi.updateUserRole(userId, role),
    onSuccess: () => { queryClient.invalidateQueries({ queryKey: ["auth-users"] }); showSnack("역할이 변경되었습니다."); },
    onError: () => showSnack("역할 변경에 실패했습니다.", "error"),
  });

  const toggleActiveMutation = useMutation({
    mutationFn: ({ userId, isActive }: { userId: string; isActive: boolean }) => authApi.toggleUserActive(userId, isActive),
    onSuccess: () => { queryClient.invalidateQueries({ queryKey: ["auth-users"] }); },
    onError: () => showSnack("상태 변경에 실패했습니다.", "error"),
  });

  const deleteMutation = useMutation({
    mutationFn: (userId: string) => authApi.deleteUser(userId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["auth-users"] });
      setDeleteConfirmId(null);
      setSelected((s) => s.filter((id) => id !== deleteConfirmId));
      showSnack("사용자가 삭제되었습니다.");
    },
    onError: () => showSnack("사용자 삭제에 실패했습니다.", "error"),
  });

  // Bulk role change — sequential mutations
  const handleBulkRoleChange = async () => {
    for (const userId of selected) {
      await authApi.updateUserRole(userId, bulkRole);
    }
    queryClient.invalidateQueries({ queryKey: ["auth-users"] });
    setBulkRoleOpen(false);
    setSelected([]);
    showSnack(`${selected.length}명의 역할이 변경되었습니다.`);
  };

  const handleBulkActiveToggle = async (active: boolean) => {
    for (const userId of selected) {
      await authApi.toggleUserActive(userId, active);
    }
    queryClient.invalidateQueries({ queryKey: ["auth-users"] });
    setSelected([]);
    showSnack(`${selected.length}명의 상태가 변경되었습니다.`);
  };

  // Sort & filter
  const handleSort = (key: SortKey) => {
    if (sortKey === key) setSortDir((d) => d === "asc" ? "desc" : "asc");
    else { setSortKey(key); setSortDir("asc"); }
  };

  const filteredUsers = rawUsers
    .filter((u) => {
      const q = searchText.toLowerCase();
      if (q && !u.username.toLowerCase().includes(q)
            && !(u.full_name ?? "").toLowerCase().includes(q)
            && !(u.department ?? "").toLowerCase().includes(q)) return false;
      if (roleFilter !== "all" && u.role !== roleFilter) return false;
      return true;
    })
    .sort((a, b) => {
      let av: string | boolean = a[sortKey] ?? "";
      let bv: string | boolean = b[sortKey] ?? "";
      if (typeof av === "boolean") av = av ? "1" : "0";
      if (typeof bv === "boolean") bv = bv ? "1" : "0";
      return sortDir === "asc" ? (av as string).localeCompare(bv as string) : (bv as string).localeCompare(av as string);
    });

  // Checkbox helpers
  const allSelected = filteredUsers.length > 0 && filteredUsers.every((u) => selected.includes(u.id));
  const someSelected = filteredUsers.some((u) => selected.includes(u.id));
  const toggleAll = () => {
    if (allSelected) setSelected((s) => s.filter((id) => !filteredUsers.find((u) => u.id === id)));
    else setSelected((s) => Array.from(new Set([...s, ...filteredUsers.map((u) => u.id)])));
  };
  const toggleOne = (id: string) =>
    setSelected((s) => s.includes(id) ? s.filter((x) => x !== id) : [...s, id]);

  // Create helpers
  const handleOpenCreate = () => {
    const pw = generateTempPassword();
    setFormData({ username: "", email: "", full_name: "", department: "", role: "user" });
    setFormGeneratedPw(pw);
    setCreatedPwVisible(false);
    setCreateOpen(true);
  };
  const handleCreate = () => {
    createMutation.mutate({
      username: formData.username,
      password: formGeneratedPw,
      email: formData.email || undefined,
      full_name: formData.full_name || undefined,
      department: formData.department || undefined,
      role: formData.role,
    });
  };

  if (isLoading) return <CircularProgress />;
  if (error) return <Alert severity="error">사용자 목록을 불러올 수 없습니다.</Alert>;

  const roleCounts = ALL_ROLES.reduce<Record<string, number>>((acc, r) => {
    acc[r] = rawUsers.filter((u) => u.role === r).length;
    return acc;
  }, {});

  return (
    <>
      {/* Toolbar */}
      <Box sx={{ display: "flex", gap: 1.5, mb: 2, alignItems: "center", flexWrap: "wrap" }}>
        <TextField
          size="small"
          placeholder="사용자명, 이름, 부서 검색..."
          value={searchText}
          onChange={(e) => setSearchText(e.target.value)}
          sx={{ minWidth: 240 }}
          slotProps={{
            input: {
              startAdornment: (
                <InputAdornment position="start">
                  <SearchIcon fontSize="small" />
                </InputAdornment>
              ),
            },
          }}
        />
        <Box sx={{ flex: 1 }} />
        <Tooltip title="새로고침">
          <IconButton size="small" onClick={() => refetch()}>
            <RefreshIcon fontSize="small" />
          </IconButton>
        </Tooltip>
        <Button variant="contained" startIcon={<AddIcon />} onClick={handleOpenCreate} size="small">
          사용자 추가
        </Button>
      </Box>

      {/* Role filter chips */}
      <Box sx={{ display: "flex", gap: 1, mb: 2, flexWrap: "wrap", alignItems: "center" }}>
        <FilterListIcon fontSize="small" sx={{ color: "text.secondary" }} />
        <Chip
          label={`전체 (${rawUsers.length})`}
          size="small"
          color={roleFilter === "all" ? "primary" : "default"}
          variant={roleFilter === "all" ? "filled" : "outlined"}
          onClick={() => setRoleFilter("all")}
          sx={{ fontWeight: roleFilter === "all" ? 700 : 400 }}
        />
        {ALL_ROLES.map((r) => (
          <Chip
            key={r}
            label={`${ROLE_META[r].label} (${roleCounts[r] ?? 0})`}
            size="small"
            color={roleFilter === r ? ROLE_META[r].color : "default"}
            variant={roleFilter === r ? "filled" : "outlined"}
            onClick={() => setRoleFilter(r === roleFilter ? "all" : r)}
            sx={{ fontWeight: roleFilter === r ? 700 : 400 }}
          />
        ))}
      </Box>

      {/* Bulk action bar */}
      <Collapse in={selected.length > 0}>
        <Paper
          variant="outlined"
          sx={{ mb: 2, p: 1.5, display: "flex", gap: 1, alignItems: "center", bgcolor: "action.selected", flexWrap: "wrap" }}
        >
          <Badge badgeContent={selected.length} color="primary">
            <GroupIcon fontSize="small" />
          </Badge>
          <Typography variant="body2" sx={{ fontWeight: 600, mr: 1 }}>
            {selected.length}명 선택됨
          </Typography>
          <Button size="small" variant="outlined" startIcon={<EditIcon />} onClick={() => setBulkRoleOpen(true)}>
            역할 일괄 변경
          </Button>
          <Button size="small" variant="outlined" color="success" startIcon={<CheckIcon />}
            onClick={() => handleBulkActiveToggle(true)}>
            일괄 활성화
          </Button>
          <Button size="small" variant="outlined" color="warning" startIcon={<BlockIcon />}
            onClick={() => handleBulkActiveToggle(false)}>
            일괄 비활성화
          </Button>
          <Box sx={{ flex: 1 }} />
          <Button size="small" onClick={() => setSelected([])}>선택 해제</Button>
        </Paper>
      </Collapse>

      {/* Table */}
      <TableContainer component={Paper} variant="outlined">
        <Table size="small">
          <TableHead>
            <TableRow sx={{ "& th": { fontWeight: 700, bgcolor: "action.hover" } }}>
              <TableCell padding="checkbox">
                <Checkbox
                  size="small"
                  indeterminate={someSelected && !allSelected}
                  checked={allSelected}
                  onChange={toggleAll}
                />
              </TableCell>
              <TableCell>
                <TableSortLabel
                  active={sortKey === "username"} direction={sortKey === "username" ? sortDir : "asc"}
                  onClick={() => handleSort("username")}
                >사용자명</TableSortLabel>
              </TableCell>
              <TableCell>
                <TableSortLabel
                  active={sortKey === "full_name"} direction={sortKey === "full_name" ? sortDir : "asc"}
                  onClick={() => handleSort("full_name")}
                >이름</TableSortLabel>
              </TableCell>
              <TableCell>
                <TableSortLabel
                  active={sortKey === "department"} direction={sortKey === "department" ? sortDir : "asc"}
                  onClick={() => handleSort("department")}
                >부서</TableSortLabel>
              </TableCell>
              <TableCell>
                <TableSortLabel
                  active={sortKey === "role"} direction={sortKey === "role" ? sortDir : "asc"}
                  onClick={() => handleSort("role")}
                >역할</TableSortLabel>
              </TableCell>
              <TableCell align="center">
                <TableSortLabel
                  active={sortKey === "is_active"} direction={sortKey === "is_active" ? sortDir : "asc"}
                  onClick={() => handleSort("is_active")}
                >활성</TableSortLabel>
              </TableCell>
              <TableCell align="center">액션</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {filteredUsers.map((user) => (
              <TableRow
                key={user.id}
                hover
                selected={selected.includes(user.id)}
                sx={{ cursor: "pointer", "&.Mui-selected": { bgcolor: "action.selected" } }}
              >
                <TableCell padding="checkbox" onClick={(e) => e.stopPropagation()}>
                  <Checkbox size="small" checked={selected.includes(user.id)} onChange={() => toggleOne(user.id)} />
                </TableCell>
                <TableCell onClick={() => setDetailUser(user)} sx={{ fontWeight: 600 }}>
                  <Stack direction="row" alignItems="center" gap={1}>
                    <Avatar sx={{ width: 28, height: 28, fontSize: 13, bgcolor: user.is_active ? "primary.main" : "grey.400" }}>
                      {user.username.charAt(0).toUpperCase()}
                    </Avatar>
                    {user.username}
                  </Stack>
                </TableCell>
                <TableCell onClick={() => setDetailUser(user)}>{user.full_name ?? "—"}</TableCell>
                <TableCell onClick={() => setDetailUser(user)}>{user.department ?? "—"}</TableCell>
                <TableCell onClick={(e) => e.stopPropagation()}>
                  <Select
                    size="small"
                    value={user.role}
                    onChange={(e) => updateRoleMutation.mutate({ userId: user.id, role: e.target.value })}
                    sx={{ minWidth: 110, fontSize: 13 }}
                    renderValue={(v) => (
                      <Chip label={ROLE_META[v]?.label ?? v} size="small" color={ROLE_META[v]?.color ?? "default"} sx={{ fontSize: 12 }} />
                    )}
                  >
                    {ALL_ROLES.map((r) => (
                      <MenuItem key={r} value={r}>
                        <Chip label={ROLE_META[r].label} size="small" color={ROLE_META[r].color} sx={{ fontSize: 12, mr: 1 }} />
                        {r}
                      </MenuItem>
                    ))}
                  </Select>
                </TableCell>
                <TableCell align="center" onClick={(e) => e.stopPropagation()}>
                  <Switch
                    size="small"
                    checked={user.is_active}
                    onChange={(e) => toggleActiveMutation.mutate({ userId: user.id, isActive: e.target.checked })}
                    color="success"
                  />
                </TableCell>
                <TableCell align="center" onClick={(e) => e.stopPropagation()}>
                  <Tooltip title="상세보기">
                    <IconButton size="small" onClick={() => setDetailUser(user)}>
                      <VisibilityIcon fontSize="small" />
                    </IconButton>
                  </Tooltip>
                  <Tooltip title="비밀번호 초기화">
                    <IconButton size="small" color="warning"
                      onClick={() => { const pw = generateTempPassword(); setResetPwUserId(user.id); setGeneratedPw(pw); }}>
                      <LockResetIcon fontSize="small" />
                    </IconButton>
                  </Tooltip>
                  <Tooltip title="삭제">
                    <IconButton size="small" color="error" onClick={() => setDeleteConfirmId(user.id)}>
                      <DeleteIcon fontSize="small" />
                    </IconButton>
                  </Tooltip>
                </TableCell>
              </TableRow>
            ))}
            {filteredUsers.length === 0 && (
              <TableRow>
                <TableCell colSpan={7} align="center" sx={{ py: 4 }}>
                  <Typography color="text.secondary">
                    {searchText || roleFilter !== "all" ? "검색 결과가 없습니다." : "등록된 사용자가 없습니다."}
                  </Typography>
                </TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
      </TableContainer>
      <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: "block" }}>
        총 {rawUsers.length}명 중 {filteredUsers.length}명 표시
      </Typography>

      {/* ── Create User Dialog ── */}
      <Dialog open={createOpen} onClose={() => setCreateOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle sx={{ display: "flex", alignItems: "center", gap: 1 }}>
          <AddIcon /> 사용자 추가
        </DialogTitle>
        <DialogContent>
          {createdPwVisible ? (
            <Box sx={{ textAlign: "center", py: 3 }}>
              <CheckCircleIcon sx={{ fontSize: 48, color: "success.main", mb: 1 }} />
              <Typography variant="h6" gutterBottom>사용자가 생성되었습니다</Typography>
              <Typography variant="body2" color="text.secondary" gutterBottom>
                임시 비밀번호를 안전하게 전달하세요.
              </Typography>
              <Paper variant="outlined" sx={{ mt: 2, p: 2, bgcolor: "action.hover" }}>
                <Typography variant="body2" color="text.secondary" gutterBottom>임시 비밀번호</Typography>
                <Stack direction="row" alignItems="center" justifyContent="center" gap={1}>
                  <Typography variant="h6" sx={{ fontFamily: "monospace", letterSpacing: 2 }}>
                    {formGeneratedPw}
                  </Typography>
                  <Tooltip title="복사">
                    <IconButton size="small" onClick={() => navigator.clipboard.writeText(formGeneratedPw)}>
                      <ContentCopyIcon fontSize="small" />
                    </IconButton>
                  </Tooltip>
                </Stack>
              </Paper>
            </Box>
          ) : (
            <>
              <TextField
                label="사용자명 *"
                fullWidth
                margin="normal"
                value={formData.username}
                onChange={(e) => setFormData((f) => ({ ...f, username: e.target.value }))}
                autoFocus
              />
              <TextField
                label="이름"
                fullWidth
                margin="normal"
                value={formData.full_name}
                onChange={(e) => setFormData((f) => ({ ...f, full_name: e.target.value }))}
              />
              <TextField
                label="이메일"
                type="email"
                fullWidth
                margin="normal"
                value={formData.email}
                onChange={(e) => setFormData((f) => ({ ...f, email: e.target.value }))}
              />
              <TextField
                label="부서"
                fullWidth
                margin="normal"
                value={formData.department}
                onChange={(e) => setFormData((f) => ({ ...f, department: e.target.value }))}
              />
              <FormControl fullWidth margin="normal">
                <InputLabel>역할</InputLabel>
                <Select
                  value={formData.role}
                  label="역할"
                  onChange={(e) => setFormData((f) => ({ ...f, role: e.target.value as UserRole }))}
                >
                  {ALL_ROLES.map((r) => (
                    <MenuItem key={r} value={r}>
                      <Chip label={ROLE_META[r].label} size="small" color={ROLE_META[r].color} sx={{ mr: 1, fontSize: 12 }} />
                      {r}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
              <Paper variant="outlined" sx={{ mt: 2, p: 1.5, bgcolor: "action.hover" }}>
                <Stack direction="row" alignItems="center" gap={1}>
                  <KeyIcon fontSize="small" color="action" />
                  <Box sx={{ flex: 1 }}>
                    <Typography variant="caption" color="text.secondary">자동 생성된 임시 비밀번호</Typography>
                    <Typography variant="body2" sx={{ fontFamily: "monospace", letterSpacing: 1.5 }}>
                      {formGeneratedPw}
                    </Typography>
                  </Box>
                  <Tooltip title="재생성">
                    <IconButton size="small" onClick={() => setFormGeneratedPw(generateTempPassword())}>
                      <RefreshIcon fontSize="small" />
                    </IconButton>
                  </Tooltip>
                  <Tooltip title="복사">
                    <IconButton size="small" onClick={() => navigator.clipboard.writeText(formGeneratedPw)}>
                      <ContentCopyIcon fontSize="small" />
                    </IconButton>
                  </Tooltip>
                </Stack>
              </Paper>
            </>
          )}
        </DialogContent>
        <DialogActions>
          {createdPwVisible ? (
            <Button variant="contained" onClick={() => setCreateOpen(false)}>닫기</Button>
          ) : (
            <>
              <Button onClick={() => setCreateOpen(false)}>취소</Button>
              <Button
                variant="contained"
                onClick={handleCreate}
                disabled={createMutation.isPending || !formData.username}
              >
                {createMutation.isPending ? <CircularProgress size={18} /> : "추가"}
              </Button>
            </>
          )}
        </DialogActions>
      </Dialog>

      {/* ── User Detail Dialog ── */}
      <Dialog open={detailUser !== null} onClose={() => setDetailUser(null)} maxWidth="sm" fullWidth>
        {detailUser && (
          <>
            <DialogTitle sx={{ display: "flex", alignItems: "center", gap: 1.5 }}>
              <Avatar sx={{ bgcolor: "primary.main", width: 36, height: 36 }}>
                {detailUser.username.charAt(0).toUpperCase()}
              </Avatar>
              <Box>
                <Typography variant="subtitle1" fontWeight={700}>{detailUser.full_name ?? detailUser.username}</Typography>
                <Typography variant="caption" color="text.secondary">@{detailUser.username}</Typography>
              </Box>
            </DialogTitle>
            <DialogContent>
              <Grid container spacing={2}>
                <Grid size={{ xs: 12 }}>
                  <Divider sx={{ mb: 1 }}>
                    <Typography variant="caption" color="text.secondary">프로필</Typography>
                  </Divider>
                </Grid>
                {[
                  { label: "사용자명", value: detailUser.username },
                  { label: "이름", value: detailUser.full_name ?? "—" },
                  { label: "이메일", value: detailUser.email ?? "—" },
                  { label: "부서", value: detailUser.department ?? "—" },
                ].map(({ label, value }) => (
                  <Grid key={label} size={{ xs: 6 }}>
                    <Typography variant="caption" color="text.secondary">{label}</Typography>
                    <Typography variant="body2" fontWeight={500}>{value}</Typography>
                  </Grid>
                ))}
                <Grid size={{ xs: 6 }}>
                  <Typography variant="caption" color="text.secondary">역할</Typography>
                  <Box>
                    <Chip label={ROLE_META[detailUser.role]?.label ?? detailUser.role} size="small"
                      color={ROLE_META[detailUser.role]?.color ?? "default"} />
                  </Box>
                </Grid>
                <Grid size={{ xs: 6 }}>
                  <Typography variant="caption" color="text.secondary">상태</Typography>
                  <Box>
                    <Chip
                      label={detailUser.is_active ? "활성" : "비활성"}
                      size="small"
                      color={detailUser.is_active ? "success" : "default"}
                    />
                  </Box>
                </Grid>
                <Grid size={{ xs: 12 }}>
                  <Divider sx={{ mb: 1, mt: 1 }}>
                    <Typography variant="caption" color="text.secondary">통계</Typography>
                  </Divider>
                </Grid>
                {[
                  { icon: <AccessTimeIcon fontSize="small" />, label: "마지막 로그인", value: detailUser.last_login ?? "—" },
                  { icon: <QuestionAnswerIcon fontSize="small" />, label: "총 쿼리 수", value: "—" },
                  { icon: <ThumbUpIcon fontSize="small" />, label: "피드백 수", value: "—" },
                ].map(({ icon, label, value }) => (
                  <Grid key={label} size={{ xs: 4 }}>
                    <Paper variant="outlined" sx={{ p: 1.5, textAlign: "center" }}>
                      <Box sx={{ color: "text.secondary", mb: 0.5 }}>{icon}</Box>
                      <Typography variant="h6" fontWeight={700}>{value}</Typography>
                      <Typography variant="caption" color="text.secondary">{label}</Typography>
                    </Paper>
                  </Grid>
                ))}
              </Grid>
            </DialogContent>
            <DialogActions sx={{ flexWrap: "wrap", gap: 1, p: 2 }}>
              <Button
                size="small"
                variant="outlined"
                color="warning"
                startIcon={<LockResetIcon />}
                onClick={() => { const pw = generateTempPassword(); setResetPwUserId(detailUser.id); setGeneratedPw(pw); setDetailUser(null); }}
              >
                비밀번호 초기화
              </Button>
              <Button
                size="small"
                variant="outlined"
                color={detailUser.is_active ? "warning" : "success"}
                startIcon={detailUser.is_active ? <BlockIcon /> : <CheckIcon />}
                onClick={() => {
                  toggleActiveMutation.mutate({ userId: detailUser.id, isActive: !detailUser.is_active });
                  setDetailUser(null);
                }}
              >
                {detailUser.is_active ? "비활성화" : "활성화"}
              </Button>
              <Button
                size="small"
                variant="outlined"
                color="error"
                startIcon={<DeleteIcon />}
                onClick={() => { setDeleteConfirmId(detailUser.id); setDetailUser(null); }}
              >
                삭제
              </Button>
              <Box sx={{ flex: 1 }} />
              <Button onClick={() => setDetailUser(null)}>닫기</Button>
            </DialogActions>
          </>
        )}
      </Dialog>

      {/* ── Reset Password Dialog ── */}
      <Dialog open={resetPwUserId !== null} onClose={() => setResetPwUserId(null)} maxWidth="xs" fullWidth>
        <DialogTitle sx={{ display: "flex", alignItems: "center", gap: 1 }}>
          <LockResetIcon /> 비밀번호 초기화
        </DialogTitle>
        <DialogContent>
          <Typography variant="body2" color="text.secondary" gutterBottom>
            새로 생성된 임시 비밀번호를 사용자에게 안전하게 전달하세요.
          </Typography>
          <Paper variant="outlined" sx={{ mt: 2, p: 2, bgcolor: "action.hover", textAlign: "center" }}>
            <Typography variant="body2" color="text.secondary" gutterBottom>임시 비밀번호</Typography>
            <Stack direction="row" alignItems="center" justifyContent="center" gap={1}>
              <Typography variant="h6" sx={{ fontFamily: "monospace", letterSpacing: 2 }}>
                {generatedPw}
              </Typography>
              <Tooltip title="복사">
                <IconButton size="small" onClick={() => navigator.clipboard.writeText(generatedPw)}>
                  <ContentCopyIcon fontSize="small" />
                </IconButton>
              </Tooltip>
            </Stack>
          </Paper>
          <Alert severity="info" sx={{ mt: 2 }} icon={<KeyIcon />}>
            이 기능은 현재 UI 미리보기용입니다. 실제 비밀번호 초기화 API가 연결되면 자동 적용됩니다.
          </Alert>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setResetPwUserId(null)}>닫기</Button>
        </DialogActions>
      </Dialog>

      {/* ── Delete Confirm Dialog ── */}
      <Dialog open={deleteConfirmId !== null} onClose={() => setDeleteConfirmId(null)}>
        <DialogTitle sx={{ display: "flex", alignItems: "center", gap: 1, color: "error.main" }}>
          <DeleteIcon /> 사용자 삭제
        </DialogTitle>
        <DialogContent>
          <Typography>이 사용자를 영구적으로 삭제하시겠습니까? 이 작업은 되돌릴 수 없습니다.</Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDeleteConfirmId(null)}>취소</Button>
          <Button
            variant="contained"
            color="error"
            onClick={() => deleteConfirmId && deleteMutation.mutate(deleteConfirmId)}
            disabled={deleteMutation.isPending}
          >
            {deleteMutation.isPending ? <CircularProgress size={18} /> : "삭제"}
          </Button>
        </DialogActions>
      </Dialog>

      {/* ── Bulk Role Change Dialog ── */}
      <Dialog open={bulkRoleOpen} onClose={() => setBulkRoleOpen(false)} maxWidth="xs" fullWidth>
        <DialogTitle>역할 일괄 변경</DialogTitle>
        <DialogContent>
          <Typography variant="body2" color="text.secondary" gutterBottom>
            선택된 {selected.length}명의 역할을 변경합니다.
          </Typography>
          <FormControl fullWidth margin="normal">
            <InputLabel>새 역할</InputLabel>
            <Select value={bulkRole} label="새 역할" onChange={(e) => setBulkRole(e.target.value as UserRole)}>
              {ALL_ROLES.map((r) => (
                <MenuItem key={r} value={r}>
                  <Chip label={ROLE_META[r].label} size="small" color={ROLE_META[r].color} sx={{ mr: 1, fontSize: 12 }} />
                  {r}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setBulkRoleOpen(false)}>취소</Button>
          <Button variant="contained" onClick={handleBulkRoleChange}>변경</Button>
        </DialogActions>
      </Dialog>

      <Snackbar
        open={snack.open}
        autoHideDuration={3000}
        onClose={() => setSnack((s) => ({ ...s, open: false }))}
      >
        <Alert severity={snack.severity} onClose={() => setSnack((s) => ({ ...s, open: false }))} sx={{ width: "100%" }}>
          {snack.message}
        </Alert>
      </Snackbar>
    </>
  );
}

// ── Tab: System Info ──
function SystemInfoTab() {
  const queryClient = useQueryClient();
  const { data, isLoading, error } = useQuery({
    queryKey: ["admin-info"],
    queryFn: () => adminApi.getInfo(),
  });

  const { data: logLevelData } = useQuery({
    queryKey: ["admin-log-level"],
    queryFn: () => adminApi.getLogLevel(),
    select: (res) => res.data as { level: string },
  });

  const [selectedLevel, setSelectedLevel] = useState<string>("");
  const [logSnack, setLogSnack] = useState<{ open: boolean; message: string; severity: "success" | "error" }>({
    open: false, message: "", severity: "success",
  });

  const setLogLevelMutation = useMutation({
    mutationFn: (level: string) => adminApi.setLogLevel(level),
    onSuccess: (_res, level) => {
      setLogSnack({ open: true, message: `로그 레벨이 ${level}로 변경되었습니다.`, severity: "success" });
      queryClient.invalidateQueries({ queryKey: ["admin-log-level"] });
      setSelectedLevel("");
    },
    onError: () => setLogSnack({ open: true, message: "로그 레벨 변경에 실패했습니다.", severity: "error" }),
  });

  if (isLoading) return <CircularProgress />;
  if (error) return <Alert severity="error">시스템 정보를 불러올 수 없습니다.</Alert>;

  const info = data?.data;
  const currentLevel = logLevelData?.level ?? "INFO";
  const effectiveLevel = selectedLevel || currentLevel;

  const cards = [
    { label: "앱 이름", value: info?.app_name ?? "-", icon: <AppRegistrationIcon color="primary" /> },
    { label: "버전", value: info?.version ?? "-", icon: <InfoIcon color="primary" /> },
    { label: "기본 프로바이더", value: info?.default_provider ?? "-", icon: <StorageIcon color="primary" /> },
    { label: "문서 수", value: info?.document_count ?? 0, icon: <FolderIcon color="primary" /> },
    { label: "세션 수", value: info?.session_count ?? 0, icon: <GroupWorkIcon color="primary" /> },
  ];

  return (
    <Box>
      <Grid container spacing={2}>
        {cards.map((card) => (
          <Grid key={card.label} size={{ xs: 12, sm: 6, md: 4 }}>
            <Card variant="outlined">
              <CardContent sx={{ display: "flex", alignItems: "center", gap: 2 }}>
                {card.icon}
                <Box>
                  <Typography variant="body2" color="text.secondary">
                    {card.label}
                  </Typography>
                  <Typography variant="h6">{String(card.value)}</Typography>
                </Box>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>

      {/* Log Level Control */}
      <Box sx={{ mt: 3 }}>
        <Typography variant="subtitle1" fontWeight={600} gutterBottom>
          로그 레벨
        </Typography>
        <Box sx={{ display: "flex", alignItems: "center", gap: 2 }}>
          <FormControl size="small" sx={{ minWidth: 160 }}>
            <InputLabel>로그 레벨</InputLabel>
            <Select
              label="로그 레벨"
              value={effectiveLevel}
              onChange={(e) => setSelectedLevel(e.target.value)}
            >
              {["DEBUG", "INFO", "WARNING", "ERROR"].map((lvl) => (
                <MenuItem key={lvl} value={lvl}>{lvl}</MenuItem>
              ))}
            </Select>
          </FormControl>
          <Button
            variant="contained"
            size="small"
            disabled={!selectedLevel || setLogLevelMutation.isPending}
            onClick={() => setLogLevelMutation.mutate(effectiveLevel)}
          >
            저장
          </Button>
          {!selectedLevel && (
            <Typography variant="body2" color="text.secondary">현재: {currentLevel}</Typography>
          )}
        </Box>
      </Box>

      <Snackbar
        open={logSnack.open}
        autoHideDuration={3000}
        onClose={() => setLogSnack((s) => ({ ...s, open: false }))}
        anchorOrigin={{ vertical: "bottom", horizontal: "center" }}
      >
        <Alert severity={logSnack.severity} onClose={() => setLogSnack((s) => ({ ...s, open: false }))}>
          {logSnack.message}
        </Alert>
      </Snackbar>
    </Box>
  );
}

// ── Tab: LLM Providers ──
function ProvidersTab() {
  const queryClient = useQueryClient();
  const [pendingProvider, setPendingProvider] = useState<string | null>(null);
  const [confirmOpen, setConfirmOpen] = useState(false);
  const [snack, setSnack] = useState<{ open: boolean; message: string; severity: "success" | "error" }>({
    open: false,
    message: "",
    severity: "success",
  });

  const { data, isLoading, error } = useQuery({
    queryKey: ["admin-providers"],
    queryFn: () => adminApi.getProviders(),
  });

  const setDefaultMutation = useMutation({
    mutationFn: (provider: string) => adminApi.setDefaultProvider(provider),
    onSuccess: () => {
      setSnack({ open: true, message: "기본 프로바이더가 변경되었습니다.", severity: "success" });
      queryClient.invalidateQueries({ queryKey: ["admin-providers"] });
      queryClient.invalidateQueries({ queryKey: ["admin-info"] });
    },
    onError: () => {
      setSnack({ open: true, message: "프로바이더 변경에 실패했습니다.", severity: "error" });
    },
  });

  const handleSetDefaultRequest = (providerName: string) => {
    setPendingProvider(providerName);
    setConfirmOpen(true);
  };

  const handleConfirmed = () => {
    if (!pendingProvider) return;
    setConfirmOpen(false);
    setDefaultMutation.mutate(pendingProvider);
    setPendingProvider(null);
  };

  if (isLoading) return <CircularProgress />;
  if (error) return <Alert severity="error">프로바이더 목록을 불러올 수 없습니다.</Alert>;

  const providers: Array<{ name: string; is_default: boolean }> = data?.data ?? [];

  return (
    <>
      <Alert severity="info" sx={{ mb: 2 }}>
        기본 프로바이더를 변경하면 전체 시스템의 LLM 응답에 즉시 영향을 줍니다.
      </Alert>
      <Grid container spacing={2}>
        {providers.map((p) => (
          <Grid key={p.name} size={{ xs: 12, sm: 6, md: 4 }}>
            <Card
              variant="outlined"
              sx={{
                borderColor: p.is_default ? "primary.main" : "divider",
                borderWidth: p.is_default ? 2 : 1,
              }}
            >
              <CardContent sx={{ display: "flex", alignItems: "center", gap: 2 }}>
                <StorageIcon color={p.is_default ? "primary" : "action"} />
                <Box sx={{ flex: 1 }}>
                  <Typography variant="subtitle1" sx={{ fontWeight: p.is_default ? 700 : 400 }}>
                    {p.name}
                  </Typography>
                </Box>
                {p.is_default ? (
                  <Chip label="현재 기본" color="primary" size="small" icon={<StarIcon />} />
                ) : (
                  <Button
                    size="small"
                    variant="outlined"
                    onClick={() => handleSetDefaultRequest(p.name)}
                    disabled={setDefaultMutation.isPending}
                  >
                    기본으로 설정
                  </Button>
                )}
              </CardContent>
            </Card>
          </Grid>
        ))}
        {providers.length === 0 && (
          <Grid size={12}>
            <Typography color="text.secondary">등록된 프로바이더가 없습니다.</Typography>
          </Grid>
        )}
      </Grid>

      {/* Confirmation Dialog */}
      <Dialog open={confirmOpen} onClose={() => setConfirmOpen(false)} maxWidth="xs" fullWidth>
        <DialogTitle>프로바이더 변경 확인</DialogTitle>
        <DialogContent>
          <Typography>
            프로바이더 변경은 전체 시스템에 영향을 줍니다. 계속하시겠습니까?
          </Typography>
          {pendingProvider && (
            <Alert severity="warning" sx={{ mt: 2 }}>
              <strong>{pendingProvider}</strong> 를 기본 프로바이더로 설정합니다.
            </Alert>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setConfirmOpen(false)}>취소</Button>
          <Button
            variant="contained"
            color="warning"
            onClick={handleConfirmed}
            disabled={setDefaultMutation.isPending}
          >
            변경
          </Button>
        </DialogActions>
      </Dialog>

      <Snackbar
        open={snack.open}
        autoHideDuration={3000}
        onClose={() => setSnack((s) => ({ ...s, open: false }))}
        message={snack.message}
      />
    </>
  );
}

// ── Tab: Model Management ──
function ModelsTab() {
  const queryClient = useQueryClient();
  const [dialogOpen, setDialogOpen] = useState(false);
  const [editingModel, setEditingModel] = useState<{
    id?: string;
    name: string;
    provider: string;
    model_id: string;
    description: string;
    is_default: boolean;
  } | null>(null);
  const [snack, setSnack] = useState<{ open: boolean; message: string; severity: "success" | "error" }>({
    open: false,
    message: "",
    severity: "success",
  });
  const [deleteConfirm, setDeleteConfirm] = useState<string | null>(null);
  const [testResults, setTestResults] = useState<Record<string, boolean>>({});

  const { data, isLoading, error } = useQuery({
    queryKey: ["admin-models"],
    queryFn: () => adminApi.listModels(),
  });

  const createMutation = useMutation({
    mutationFn: (model: { name: string; provider: string; model_id: string; description: string; is_default: boolean }) =>
      adminApi.createModel(model),
    onSuccess: () => {
      setSnack({ open: true, message: "모델이 추가되었습니다.", severity: "success" });
      queryClient.invalidateQueries({ queryKey: ["admin-models"] });
      setDialogOpen(false);
      setEditingModel(null);
    },
    onError: () => {
      setSnack({ open: true, message: "모델 추가에 실패했습니다.", severity: "error" });
    },
  });

  const updateMutation = useMutation({
    mutationFn: ({ id, model }: { id: string; model: { name: string; provider: string; model_id: string; description: string; is_default: boolean } }) =>
      adminApi.updateModel(id, model),
    onSuccess: () => {
      setSnack({ open: true, message: "모델이 수정되었습니다.", severity: "success" });
      queryClient.invalidateQueries({ queryKey: ["admin-models"] });
      setDialogOpen(false);
      setEditingModel(null);
    },
    onError: () => {
      setSnack({ open: true, message: "모델 수정에 실패했습니다.", severity: "error" });
    },
  });

  const deleteMutation = useMutation({
    mutationFn: (id: string) => adminApi.deleteModel(id),
    onSuccess: () => {
      setSnack({ open: true, message: "모델이 삭제되었습니다.", severity: "success" });
      queryClient.invalidateQueries({ queryKey: ["admin-models"] });
      setDeleteConfirm(null);
    },
    onError: () => {
      setSnack({ open: true, message: "모델 삭제에 실패했습니다.", severity: "error" });
    },
  });

  const testMutation = useMutation({
    mutationFn: (id: string) => adminApi.testModel(id),
    onSuccess: (_, id) => {
      setTestResults((prev) => ({ ...prev, [id]: true }));
      setSnack({ open: true, message: "테스트 성공", severity: "success" });
    },
    onError: (_, id) => {
      setTestResults((prev) => ({ ...prev, [id]: false }));
      setSnack({ open: true, message: "테스트 실패", severity: "error" });
    },
  });

  const handleOpenCreate = () => {
    setEditingModel({ name: "", provider: "ollama", model_id: "", description: "", is_default: false });
    setDialogOpen(true);
  };

  const handleOpenEdit = (model: { id: string; name: string; provider: string; model_id: string; description: string; is_default: boolean }) => {
    setEditingModel(model);
    setDialogOpen(true);
  };

  const handleSave = () => {
    if (!editingModel) return;
    if (editingModel.id) {
      updateMutation.mutate({ id: editingModel.id, model: editingModel });
    } else {
      createMutation.mutate(editingModel);
    }
  };

  const handleDelete = (id: string) => {
    deleteMutation.mutate(id);
  };

  const handleTest = (id: string) => {
    testMutation.mutate(id);
  };

  if (isLoading) return <CircularProgress />;
  if (error) return <Alert severity="error">모델 목록을 불러올 수 없습니다.</Alert>;

  const models: Array<{ id: string; name: string; provider: string; model_id: string; description: string; is_default: boolean; is_active: boolean; status: string }> = data?.data?.models ?? [];

  return (
    <>
      <Box sx={{ mb: 2, display: "flex", justifyContent: "flex-end" }}>
        <Button variant="contained" startIcon={<AddIcon />} onClick={handleOpenCreate}>
          모델 추가
        </Button>
      </Box>

      <TableContainer component={Paper} variant="outlined">
        <Table>
          <TableHead>
            <TableRow>
              <TableCell>이름</TableCell>
              <TableCell>프로바이더</TableCell>
              <TableCell>모델 ID</TableCell>
              <TableCell>기본</TableCell>
              <TableCell>활성</TableCell>
              <TableCell>헬스체크</TableCell>
              <TableCell align="right">작업</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {models.map((model) => (
              <TableRow key={model.id}>
                <TableCell>{model.name}</TableCell>
                <TableCell>
                  <Chip
                    label={model.provider}
                    size="small"
                    variant="outlined"
                    color={
                      model.provider === "ollama" ? "default"
                      : model.provider === "openai" ? "success"
                      : model.provider === "anthropic" ? "secondary"
                      : "info"
                    }
                  />
                </TableCell>
                <TableCell sx={{ fontFamily: "monospace", fontSize: 13 }}>{model.model_id}</TableCell>
                <TableCell>
                  {model.is_default && <Chip label="기본" color="primary" size="small" icon={<StarIcon />} />}
                </TableCell>
                <TableCell>
                  {model.is_active !== false ? (
                    <Chip label="활성" color="success" size="small" icon={<CheckCircleIcon />} />
                  ) : (
                    <Chip label="비활성" color="default" size="small" />
                  )}
                </TableCell>
                <TableCell>
                  {testResults[model.id] === true && <Chip label="정상" color="success" size="small" icon={<CheckCircleIcon />} />}
                  {testResults[model.id] === false && <Chip label="실패" color="error" size="small" icon={<ErrorIcon />} />}
                  {testResults[model.id] === undefined && <Typography variant="body2" color="text.secondary">-</Typography>}
                </TableCell>
                <TableCell align="right">
                  <Tooltip title="헬스체크">
                    <IconButton size="small" onClick={() => handleTest(model.id)}>
                      <PlayArrowIcon />
                    </IconButton>
                  </Tooltip>
                  <Tooltip title="수정">
                    <IconButton size="small" onClick={() => handleOpenEdit(model)}>
                      <EditIcon />
                    </IconButton>
                  </Tooltip>
                  <Tooltip title="삭제">
                    <IconButton size="small" onClick={() => setDeleteConfirm(model.id)}>
                      <DeleteIcon />
                    </IconButton>
                  </Tooltip>
                </TableCell>
              </TableRow>
            ))}
            {models.length === 0 && (
              <TableRow>
                <TableCell colSpan={7} align="center">
                  <Typography color="text.secondary">등록된 모델이 없습니다.</Typography>
                </TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
      </TableContainer>

      <Dialog open={dialogOpen} onClose={() => setDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>{editingModel?.id ? "모델 수정" : "모델 추가"}</DialogTitle>
        <DialogContent>
          <TextField
            label="이름"
            fullWidth
            margin="normal"
            value={editingModel?.name ?? ""}
            onChange={(e) => setEditingModel((m) => m ? { ...m, name: e.target.value } : null)}
          />
          <Select
            fullWidth
            value={editingModel?.provider ?? "ollama"}
            onChange={(e) => setEditingModel((m) => m ? { ...m, provider: e.target.value } : null)}
            sx={{ mt: 2 }}
          >
            <MenuItem value="ollama">Ollama</MenuItem>
            <MenuItem value="openai">OpenAI</MenuItem>
            <MenuItem value="anthropic">Anthropic</MenuItem>
            <MenuItem value="vllm">vLLM</MenuItem>
          </Select>
          <TextField
            label="모델 ID"
            fullWidth
            margin="normal"
            value={editingModel?.model_id ?? ""}
            onChange={(e) => setEditingModel((m) => m ? { ...m, model_id: e.target.value } : null)}
          />
          <TextField
            label="설명"
            fullWidth
            margin="normal"
            multiline
            rows={2}
            value={editingModel?.description ?? ""}
            onChange={(e) => setEditingModel((m) => m ? { ...m, description: e.target.value } : null)}
          />
          <FormControlLabel
            control={
              <Switch
                checked={editingModel?.is_default ?? false}
                onChange={(e) => setEditingModel((m) => m ? { ...m, is_default: e.target.checked } : null)}
              />
            }
            label="기본 모델로 설정"
            sx={{ mt: 1 }}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDialogOpen(false)}>취소</Button>
          <Button variant="contained" onClick={handleSave} disabled={createMutation.isPending || updateMutation.isPending}>
            저장
          </Button>
        </DialogActions>
      </Dialog>

      <Dialog open={deleteConfirm !== null} onClose={() => setDeleteConfirm(null)}>
        <DialogTitle>모델 삭제</DialogTitle>
        <DialogContent>
          <Typography>이 모델을 삭제하시겠습니까?</Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDeleteConfirm(null)}>취소</Button>
          <Button variant="contained" color="error" onClick={() => deleteConfirm && handleDelete(deleteConfirm)}>
            삭제
          </Button>
        </DialogActions>
      </Dialog>

      <Snackbar
        open={snack.open}
        autoHideDuration={3000}
        onClose={() => setSnack((s) => ({ ...s, open: false }))}
        message={snack.message}
      />
    </>
  );
}

// ── Tab: Guardrails ──
function GuardrailsTab() {
  const queryClient = useQueryClient();
  const [dialogOpen, setDialogOpen] = useState(false);
  const [testDialogOpen, setTestDialogOpen] = useState(false);
  const [testInput, setTestInput] = useState("");
  const [testRuleId, setTestRuleId] = useState("");
  const [testResult, setTestResult] = useState<string | null>(null);
  const [editingRule, setEditingRule] = useState<{
    id?: string;
    name: string;
    rule_type: string;
    pattern: string;
    action: string;
    message: string;
    is_active: boolean;
  } | null>(null);
  const [snack, setSnack] = useState<{ open: boolean; message: string; severity: "success" | "error" }>({
    open: false,
    message: "",
    severity: "success",
  });
  const [deleteConfirm, setDeleteConfirm] = useState<string | null>(null);

  const { data: rulesData, isLoading: rulesLoading, error: rulesError } = useQuery({
    queryKey: ["guardrails-rules"],
    queryFn: () => guardrailsApi.list(),
  });

  const { data: logsData, isLoading: logsLoading } = useQuery({
    queryKey: ["guardrails-logs"],
    queryFn: () => guardrailsApi.getLogs(),
  });

  const createMutation = useMutation({
    mutationFn: (rule: { name: string; rule_type: string; pattern: string; action: string; message: string; is_active: boolean }) =>
      guardrailsApi.create(rule),
    onSuccess: () => {
      setSnack({ open: true, message: "규칙이 추가되었습니다.", severity: "success" });
      queryClient.invalidateQueries({ queryKey: ["guardrails-rules"] });
      setDialogOpen(false);
      setEditingRule(null);
    },
    onError: () => {
      setSnack({ open: true, message: "규칙 추가에 실패했습니다.", severity: "error" });
    },
  });

  const updateMutation = useMutation({
    mutationFn: ({ id, rule }: { id: string; rule: { name: string; rule_type: string; pattern: string; action: string; message: string; is_active: boolean } }) =>
      guardrailsApi.update(id, rule),
    onSuccess: () => {
      setSnack({ open: true, message: "규칙이 수정되었습니다.", severity: "success" });
      queryClient.invalidateQueries({ queryKey: ["guardrails-rules"] });
      setDialogOpen(false);
      setEditingRule(null);
    },
    onError: () => {
      setSnack({ open: true, message: "규칙 수정에 실패했습니다.", severity: "error" });
    },
  });

  const deleteMutation = useMutation({
    mutationFn: (id: string) => guardrailsApi.delete(id),
    onSuccess: () => {
      setSnack({ open: true, message: "규칙이 삭제되었습니다.", severity: "success" });
      queryClient.invalidateQueries({ queryKey: ["guardrails-rules"] });
      setDeleteConfirm(null);
    },
    onError: () => {
      setSnack({ open: true, message: "규칙 삭제에 실패했습니다.", severity: "error" });
    },
  });

  const testMutation = useMutation({
    mutationFn: (data: { rule_id: string; test_input: string }) => guardrailsApi.test(data),
    onSuccess: (res) => {
      setTestResult(JSON.stringify(res.data, null, 2));
    },
    onError: () => {
      setTestResult("테스트 실패");
    },
  });

  const handleOpenCreate = () => {
    setEditingRule({ name: "", rule_type: "input_keyword", pattern: "", action: "block", message: "", is_active: true });
    setDialogOpen(true);
  };

  const handleOpenEdit = (rule: { id: string; name: string; rule_type: string; pattern: string; action: string; message: string; is_active: boolean }) => {
    setEditingRule(rule);
    setDialogOpen(true);
  };

  const handleSave = () => {
    if (!editingRule) return;
    if (editingRule.id) {
      updateMutation.mutate({ id: editingRule.id, rule: editingRule });
    } else {
      createMutation.mutate(editingRule);
    }
  };

  const handleDelete = (id: string) => {
    deleteMutation.mutate(id);
  };

  const handleTest = () => {
    testMutation.mutate({ rule_id: testRuleId, test_input: testInput });
  };

  if (rulesLoading) return <CircularProgress />;
  if (rulesError) return <Alert severity="error">가드레일 규칙을 불러올 수 없습니다.</Alert>;

  const rules: Array<{ id: string; name: string; rule_type: string; pattern: string; action: string; message: string; is_active: boolean }> = rulesData?.data?.rules ?? [];
  const logs: Array<{ timestamp: string; rule_name: string; input_preview: string; action_taken: string }> = logsData?.data?.logs ?? [];

  return (
    <>
      <Box sx={{ mb: 2, display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <Typography variant="h6">가드레일 규칙</Typography>
        <Box>
          <Button variant="outlined" startIcon={<PlayArrowIcon />} onClick={() => setTestDialogOpen(true)} sx={{ mr: 1 }}>
            테스트
          </Button>
          <Button variant="contained" startIcon={<AddIcon />} onClick={handleOpenCreate}>
            규칙 추가
          </Button>
        </Box>
      </Box>

      <TableContainer component={Paper} variant="outlined" sx={{ mb: 4 }}>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell>이름</TableCell>
              <TableCell>유형</TableCell>
              <TableCell>패턴</TableCell>
              <TableCell>액션</TableCell>
              <TableCell>활성</TableCell>
              <TableCell align="right">작업</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {rules.map((rule) => (
              <TableRow key={rule.id}>
                <TableCell>{rule.name}</TableCell>
                <TableCell>
                  <Chip label={rule.rule_type} size="small" />
                </TableCell>
                <TableCell sx={{ fontFamily: "monospace", fontSize: 13, maxWidth: 200, overflow: "hidden", textOverflow: "ellipsis" }}>
                  {rule.pattern}
                </TableCell>
                <TableCell>
                  <Chip
                    label={rule.action}
                    size="small"
                    color={rule.action === "block" ? "error" : rule.action === "warn" ? "warning" : "default"}
                  />
                </TableCell>
                <TableCell>
                  {rule.is_active ? (
                    <Chip label="활성" color="success" size="small" icon={<CheckCircleIcon />} />
                  ) : (
                    <Chip label="비활성" size="small" />
                  )}
                </TableCell>
                <TableCell align="right">
                  <IconButton size="small" onClick={() => handleOpenEdit(rule)}>
                    <EditIcon />
                  </IconButton>
                  <IconButton size="small" onClick={() => setDeleteConfirm(rule.id)}>
                    <DeleteIcon />
                  </IconButton>
                </TableCell>
              </TableRow>
            ))}
            {rules.length === 0 && (
              <TableRow>
                <TableCell colSpan={6} align="center">
                  <Typography color="text.secondary">등록된 규칙이 없습니다.</Typography>
                </TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
      </TableContainer>

      <Typography variant="h6" sx={{ mb: 2 }}>필터링 로그 (최근 20개)</Typography>
      <TableContainer component={Paper} variant="outlined">
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell>시간</TableCell>
              <TableCell>규칙</TableCell>
              <TableCell>입력 미리보기</TableCell>
              <TableCell>액션</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {logsLoading ? (
              <TableRow>
                <TableCell colSpan={4} align="center"><CircularProgress size={20} /></TableCell>
              </TableRow>
            ) : logs.length > 0 ? (
              logs.slice(0, 20).map((log, idx) => (
                <TableRow key={idx}>
                  <TableCell sx={{ fontSize: 12 }}>{new Date(log.timestamp).toLocaleString()}</TableCell>
                  <TableCell>{log.rule_name}</TableCell>
                  <TableCell sx={{ fontFamily: "monospace", fontSize: 12, maxWidth: 300, overflow: "hidden", textOverflow: "ellipsis" }}>
                    {log.input_preview}
                  </TableCell>
                  <TableCell>
                    <Chip label={log.action_taken} size="small" />
                  </TableCell>
                </TableRow>
              ))
            ) : (
              <TableRow>
                <TableCell colSpan={4} align="center">
                  <Typography color="text.secondary">로그가 없습니다.</Typography>
                </TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
      </TableContainer>

      <Dialog open={dialogOpen} onClose={() => setDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>{editingRule?.id ? "규칙 수정" : "규칙 추가"}</DialogTitle>
        <DialogContent>
          <TextField
            label="이름"
            fullWidth
            margin="normal"
            value={editingRule?.name ?? ""}
            onChange={(e) => setEditingRule((r) => r ? { ...r, name: e.target.value } : null)}
          />
          <Select
            fullWidth
            value={editingRule?.rule_type ?? "input_keyword"}
            onChange={(e) => setEditingRule((r) => r ? { ...r, rule_type: e.target.value } : null)}
            sx={{ mt: 2 }}
          >
            <MenuItem value="input_keyword">입력 키워드 필터</MenuItem>
            <MenuItem value="input_pattern">입력 정규식 필터</MenuItem>
            <MenuItem value="output_keyword">출력 키워드 필터</MenuItem>
            <MenuItem value="output_pattern">출력 정규식 필터</MenuItem>
          </Select>
          <TextField
            label="패턴"
            fullWidth
            margin="normal"
            value={editingRule?.pattern ?? ""}
            onChange={(e) => setEditingRule((r) => r ? { ...r, pattern: e.target.value } : null)}
          />
          <Select
            fullWidth
            value={editingRule?.action ?? "block"}
            onChange={(e) => setEditingRule((r) => r ? { ...r, action: e.target.value } : null)}
            sx={{ mt: 2 }}
          >
            <MenuItem value="block">Block</MenuItem>
            <MenuItem value="warn">Warn</MenuItem>
            <MenuItem value="mask">Mask</MenuItem>
          </Select>
          <TextField
            label="메시지 (선택)"
            fullWidth
            margin="normal"
            multiline
            rows={2}
            value={editingRule?.message ?? ""}
            onChange={(e) => setEditingRule((r) => r ? { ...r, message: e.target.value } : null)}
          />
          <FormControlLabel
            control={
              <Switch
                checked={editingRule?.is_active ?? true}
                onChange={(e) => setEditingRule((r) => r ? { ...r, is_active: e.target.checked } : null)}
              />
            }
            label="활성화"
            sx={{ mt: 1 }}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDialogOpen(false)}>취소</Button>
          <Button variant="contained" onClick={handleSave} disabled={createMutation.isPending || updateMutation.isPending}>
            저장
          </Button>
        </DialogActions>
      </Dialog>

      <Dialog open={deleteConfirm !== null} onClose={() => setDeleteConfirm(null)}>
        <DialogTitle>규칙 삭제</DialogTitle>
        <DialogContent>
          <Typography>이 규칙을 삭제하시겠습니까?</Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDeleteConfirm(null)}>취소</Button>
          <Button variant="contained" color="error" onClick={() => deleteConfirm && handleDelete(deleteConfirm)}>
            삭제
          </Button>
        </DialogActions>
      </Dialog>

      <Dialog open={testDialogOpen} onClose={() => setTestDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>가드레일 테스트</DialogTitle>
        <DialogContent>
          <FormControl fullWidth margin="normal">
            <InputLabel>테스트 규칙</InputLabel>
            <Select value={testRuleId} onChange={(e) => setTestRuleId(e.target.value)} label="테스트 규칙">
              {rules.map((r) => (
                <MenuItem key={r.id} value={r.id}>{r.name} ({r.rule_type})</MenuItem>
              ))}
            </Select>
          </FormControl>
          <TextField
            label="테스트 입력"
            fullWidth
            margin="normal"
            multiline
            rows={4}
            value={testInput}
            onChange={(e) => setTestInput(e.target.value)}
          />
          {testResult && (
            <Box sx={{ mt: 2, p: 1, bgcolor: "grey.100", borderRadius: 1, maxHeight: 200, overflow: "auto" }}>
              <Typography variant="body2" component="pre" sx={{ fontFamily: "monospace", fontSize: 12, whiteSpace: "pre-wrap" }}>
                {testResult}
              </Typography>
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setTestDialogOpen(false)}>닫기</Button>
          <Button variant="contained" onClick={handleTest} disabled={testMutation.isPending}>
            {testMutation.isPending ? <CircularProgress size={20} /> : "테스트"}
          </Button>
        </DialogActions>
      </Dialog>

      <Snackbar
        open={snack.open}
        autoHideDuration={3000}
        onClose={() => setSnack((s) => ({ ...s, open: false }))}
        message={snack.message}
      />
    </>
  );
}

// ── Tab: Prompt Management ──
function PromptsTab() {
  const queryClient = useQueryClient();
  const [editValues, setEditValues] = useState<Record<string, string>>({});
  const [pendingSaveName, setPendingSaveName] = useState<string | null>(null);
  const [snack, setSnack] = useState<{ open: boolean; message: string; severity: "success" | "error" }>({
    open: false,
    message: "",
    severity: "success",
  });

  const { data, isLoading, error } = useQuery({
    queryKey: ["admin-prompts"],
    queryFn: () => adminApi.getPrompts(),
  });

  const mutation = useMutation({
    mutationFn: ({ name, content }: { name: string; content: string }) =>
      adminApi.updatePrompt(name, content),
    onSuccess: () => {
      setSnack({ open: true, message: "프롬프트가 저장되었습니다.", severity: "success" });
      queryClient.invalidateQueries({ queryKey: ["admin-prompts"] });
    },
    onError: () => {
      setSnack({ open: true, message: "프롬프트 저장에 실패했습니다.", severity: "error" });
    },
  });

  if (isLoading) return <CircularProgress />;
  if (error) return <Alert severity="error">프롬프트를 불러올 수 없습니다.</Alert>;

  const prompts: Record<string, string> = data?.data?.prompts ?? {};

  const handleChange = (name: string, value: string) => {
    setEditValues((prev) => ({ ...prev, [name]: value }));
  };

  const handleSaveRequest = (name: string) => {
    setPendingSaveName(name);
  };

  const handleSaveConfirmed = () => {
    if (!pendingSaveName) return;
    const name = pendingSaveName;
    const content = editValues[name] ?? prompts[name] ?? "";
    setPendingSaveName(null);
    mutation.mutate({ name, content });
  };

  return (
    <>
      <Alert severity="warning" sx={{ mb: 2 }}>
        프롬프트 변경 시 모든 RAG 응답에 즉시 반영됩니다. 변경 전 시뮬레이터 테스트를 권장합니다.
      </Alert>
      {Object.entries(prompts).map(([name, defaultValue]) => (
        <Accordion key={name} sx={{ mb: 1 }}>
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
              {name}
            </Typography>
          </AccordionSummary>
          <AccordionDetails>
            <TextField
              multiline
              fullWidth
              minRows={4}
              maxRows={16}
              value={editValues[name] ?? defaultValue}
              onChange={(e) => handleChange(name, e.target.value)}
              slotProps={{
                input: {
                  sx: { fontFamily: "monospace", fontSize: 13 },
                },
              }}
            />
            <Box sx={{ mt: 1, display: "flex", justifyContent: "flex-end" }}>
              <Button
                variant="contained"
                size="small"
                onClick={() => handleSaveRequest(name)}
                disabled={mutation.isPending}
              >
                저장
              </Button>
            </Box>
          </AccordionDetails>
        </Accordion>
      ))}
      {Object.keys(prompts).length === 0 && (
        <Typography color="text.secondary">등록된 프롬프트가 없습니다.</Typography>
      )}
      <Snackbar
        open={snack.open}
        autoHideDuration={3000}
        onClose={() => setSnack((s) => ({ ...s, open: false }))}
        message={snack.message}
      />
      <PasswordConfirmDialog
        open={pendingSaveName !== null}
        onClose={() => setPendingSaveName(null)}
        onConfirm={handleSaveConfirmed}
        title="프롬프트 변경 확인"
        description="프롬프트 변경은 모든 RAG 응답에 즉시 반영됩니다. 계속하시려면 비밀번호를 입력하세요."
      />
    </>
  );
}

// ── Tab: MCP Tools ──
function McpToolsTab() {
  const [testDialog, setTestDialog] = useState<{ open: boolean; toolName: string }>({
    open: false,
    toolName: "",
  });
  const [argsInput, setArgsInput] = useState("{}");
  const [result, setResult] = useState<string | null>(null);
  const [callError, setCallError] = useState<string | null>(null);

  const { data, isLoading, error } = useQuery({
    queryKey: ["mcp-tools"],
    queryFn: () => mcpApi.listTools(),
  });

  const callMutation = useMutation({
    mutationFn: ({ toolName, args }: { toolName: string; args: Record<string, unknown> }) =>
      mcpApi.callTool(toolName, args),
    onSuccess: (res) => {
      setResult(JSON.stringify(res.data, null, 2));
      setCallError(null);
    },
    onError: (err) => {
      setCallError(String(err));
      setResult(null);
    },
  });

  const handleOpenTest = (toolName: string) => {
    setTestDialog({ open: true, toolName });
    setResult(null);
    setCallError(null);

    // Pre-populate JSON template from tool schema
    const tool = (data?.data?.tools ?? []).find((t: McpTool) => t.name === toolName) as McpTool | undefined;
    if (tool?.inputSchema?.properties) {
      const template: Record<string, unknown> = {};
      const required = tool.inputSchema.required ?? [];
      for (const [pName, pDef] of Object.entries(tool.inputSchema.properties)) {
        if (required.includes(pName)) {
          // Populate required params with placeholder values
          if (pDef.default !== undefined && pDef.default !== null && pDef.default !== "") {
            template[pName] = pDef.default;
          } else if (pDef.enum && pDef.enum.length > 0) {
            template[pName] = pDef.enum[0];
          } else if (pDef.type === "integer") {
            template[pName] = 0;
          } else if (pDef.type === "boolean") {
            template[pName] = false;
          } else if (pDef.type === "array") {
            template[pName] = [];
          } else if (pDef.type === "object") {
            template[pName] = {};
          } else {
            template[pName] = "";
          }
        }
      }
      setArgsInput(JSON.stringify(template, null, 2));
    } else {
      setArgsInput("{}");
    }
  };

  const handleCall = () => {
    try {
      const parsed = JSON.parse(argsInput) as Record<string, unknown>;
      callMutation.mutate({ toolName: testDialog.toolName, args: parsed });
    } catch {
      setCallError("올바른 JSON 형식이 아닙니다.");
    }
  };

  if (isLoading) return <CircularProgress />;
  if (error) return <Alert severity="error">MCP 도구를 불러올 수 없습니다.</Alert>;

  interface McpTool {
    name: string;
    description: string;
    help_text?: string;
    category?: string;
    inputSchema?: {
      type: string;
      properties?: Record<string, { type: string; description: string; enum?: string[]; default?: unknown }>;
      required?: string[];
    };
  }

  const tools: McpTool[] = data?.data?.tools ?? [];

  const CATEGORY_LABELS: Record<string, string> = {
    utility: "유틸리티",
    analytics: "분석",
    analysis: "분석",
    retrieval: "검색",
    management: "관리",
    nlp: "언어처리",
    document: "문서생성",
    generation: "생성",
    general: "일반",
  };

  return (
    <>
      {tools.map((tool) => {
        const paramEntries = tool.inputSchema?.properties
          ? Object.entries(tool.inputSchema.properties)
          : [];
        const requiredParams = tool.inputSchema?.required ?? [];

        return (
        <Card key={tool.name} variant="outlined" sx={{ mb: 1.5 }}>
          <CardContent>
            <Box sx={{ display: "flex", alignItems: "flex-start", gap: 2 }}>
              <BuildIcon color="action" sx={{ mt: 0.5 }} />
              <Box sx={{ flex: 1 }}>
                <Box sx={{ display: "flex", alignItems: "center", gap: 1, flexWrap: "wrap", mb: 0.5 }}>
                  <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
                    {tool.name}
                  </Typography>
                  {tool.category && (
                    <Chip
                      label={CATEGORY_LABELS[tool.category] ?? tool.category}
                      size="small"
                      variant="outlined"
                      sx={{ height: 20, fontSize: 11 }}
                    />
                  )}
                  {tool.help_text && (
                    <Tooltip
                      title={
                        <Box sx={{ whiteSpace: "pre-line", fontSize: 12 }}>
                          {tool.help_text}
                        </Box>
                      }
                      arrow
                      placement="top"
                    >
                      <HelpOutlineIcon sx={{ fontSize: 16, color: "text.secondary", cursor: "help" }} />
                    </Tooltip>
                  )}
                </Box>
                <Typography variant="body2" color="text.secondary" sx={{ mb: paramEntries.length > 0 ? 1 : 0 }}>
                  {tool.description}
                </Typography>
                {paramEntries.length > 0 && (
                  <Box sx={{ display: "flex", flexWrap: "wrap", gap: 0.5 }}>
                    {paramEntries.map(([pName, pDef]) => {
                      const isRequired = requiredParams.includes(pName);
                      return (
                        <Tooltip
                          key={pName}
                          title={
                            <Box sx={{ fontSize: 12 }}>
                              <strong>{pName}</strong> ({pDef.type}){isRequired ? " *필수" : " (선택)"}
                              {pDef.description && <><br />{pDef.description}</>}
                              {pDef.enum && <><br />선택값: {pDef.enum.join(", ")}</>}
                              {pDef.default !== undefined && pDef.default !== null && pDef.default !== "" && (
                                <><br />기본값: {String(pDef.default)}</>
                              )}
                            </Box>
                          }
                          arrow
                          placement="bottom"
                        >
                          <Chip
                            label={pName + (isRequired ? " *" : "")}
                            size="small"
                            color={isRequired ? "primary" : "default"}
                            variant={isRequired ? "filled" : "outlined"}
                            sx={{ height: 20, fontSize: 11, cursor: "help" }}
                          />
                        </Tooltip>
                      );
                    })}
                  </Box>
                )}
              </Box>
              <Button variant="outlined" size="small" onClick={() => handleOpenTest(tool.name)} sx={{ flexShrink: 0 }}>
                테스트
              </Button>
            </Box>
          </CardContent>
        </Card>
        );
      })}
      {tools.length === 0 && (
        <Typography color="text.secondary">등록된 MCP 도구가 없습니다.</Typography>
      )}

      <Dialog
        open={testDialog.open}
        onClose={() => setTestDialog({ open: false, toolName: "" })}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>도구 테스트: {testDialog.toolName}</DialogTitle>
        <DialogContent>
          {(() => {
            const activeTool = (data?.data?.tools ?? []).find((t: McpTool) => t.name === testDialog.toolName) as McpTool | undefined;
            if (!activeTool) return null;
            const props = activeTool.inputSchema?.properties ?? {};
            const required = activeTool.inputSchema?.required ?? [];
            const paramList = Object.entries(props);
            if (paramList.length === 0) return null;
            return (
              <Box sx={{ mb: 1.5, mt: 1, p: 1.5, bgcolor: "grey.50", borderRadius: 1, border: "1px solid", borderColor: "divider" }}>
                <Typography variant="caption" color="text.secondary" sx={{ fontWeight: 600, display: "block", mb: 0.5 }}>
                  파라미터 안내
                </Typography>
                {paramList.map(([pName, pDef]) => (
                  <Box key={pName} sx={{ display: "flex", gap: 0.5, alignItems: "baseline", mb: 0.25 }}>
                    <Chip
                      label={required.includes(pName) ? `${pName} *` : pName}
                      size="small"
                      color={required.includes(pName) ? "primary" : "default"}
                      variant={required.includes(pName) ? "filled" : "outlined"}
                      sx={{ height: 18, fontSize: 10 }}
                    />
                    <Typography variant="caption" color="text.secondary">
                      {pDef.type}{pDef.enum ? ` [${pDef.enum.join(" | ")}]` : ""} — {pDef.description}
                    </Typography>
                  </Box>
                ))}
              </Box>
            );
          })()}
          <Typography variant="body2" sx={{ mb: 1 }}>
            인자 (JSON)
          </Typography>
          <TextField
            multiline
            fullWidth
            minRows={3}
            maxRows={10}
            value={argsInput}
            onChange={(e) => setArgsInput(e.target.value)}
            slotProps={{
              input: {
                sx: { fontFamily: "monospace", fontSize: 13 },
              },
            }}
          />
          {callError && (
            <Alert severity="error" sx={{ mt: 1 }}>
              {callError}
            </Alert>
          )}
          {result && (
            <Box sx={{ mt: 1, p: 1, bgcolor: "grey.100", borderRadius: 1, maxHeight: 200, overflow: "auto" }}>
              <Typography variant="body2" component="pre" sx={{ fontFamily: "monospace", fontSize: 12, whiteSpace: "pre-wrap" }}>
                {result}
              </Typography>
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setTestDialog({ open: false, toolName: "" })}>닫기</Button>
          <Button variant="contained" onClick={handleCall} disabled={callMutation.isPending}>
            {callMutation.isPending ? <CircularProgress size={20} /> : "실행"}
          </Button>
        </DialogActions>
      </Dialog>
    </>
  );
}

// ── Tab: Workflows ──
function WorkflowsTab() {
  const [runDialog, setRunDialog] = useState<{ open: boolean; preset: string }>({
    open: false,
    preset: "",
  });
  const [inputData, setInputData] = useState("{}");
  const [result, setResult] = useState<string | null>(null);
  const [runError, setRunError] = useState<string | null>(null);

  const { data, isLoading, error } = useQuery({
    queryKey: ["workflow-presets"],
    queryFn: () => workflowsApi.listPresets(),
  });

  const runMutation = useMutation({
    mutationFn: ({ preset, input }: { preset: string; input: Record<string, unknown> }) =>
      workflowsApi.run(input, { preset }),
    onSuccess: (res) => {
      setResult(JSON.stringify(res.data, null, 2));
      setRunError(null);
    },
    onError: (err) => {
      setRunError(String(err));
      setResult(null);
    },
  });

  const handleOpenRun = (preset: string) => {
    setRunDialog({ open: true, preset });
    setInputData("{}");
    setResult(null);
    setRunError(null);
  };

  const handleRun = () => {
    try {
      const parsed = JSON.parse(inputData) as Record<string, unknown>;
      runMutation.mutate({ preset: runDialog.preset, input: parsed });
    } catch {
      setRunError("올바른 JSON 형식이 아닙니다.");
    }
  };

  if (isLoading) return <CircularProgress />;
  if (error) return <Alert severity="error">워크플로우를 불러올 수 없습니다.</Alert>;

  const presets: Array<{ name: string; description: string; node_count: number }> = data?.data?.presets ?? [];

  return (
    <>
      {presets.map((preset) => (
        <Card key={preset.name} variant="outlined" sx={{ mb: 1.5 }}>
          <CardContent sx={{ display: "flex", alignItems: "center", gap: 2 }}>
            <AccountTreeIcon color="action" />
            <Box sx={{ flex: 1 }}>
              <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
                {preset.name}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                {preset.description}
              </Typography>
              <Chip label={`노드 ${preset.node_count}개`} size="small" sx={{ mt: 0.5 }} />
            </Box>
            <Button
              variant="outlined"
              size="small"
              startIcon={<PlayArrowIcon />}
              onClick={() => handleOpenRun(preset.name)}
            >
              실행
            </Button>
          </CardContent>
        </Card>
      ))}
      {presets.length === 0 && (
        <Typography color="text.secondary">등록된 워크플로우가 없습니다.</Typography>
      )}

      <Dialog
        open={runDialog.open}
        onClose={() => setRunDialog({ open: false, preset: "" })}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>워크플로우 실행: {runDialog.preset}</DialogTitle>
        <DialogContent>
          <Typography variant="body2" sx={{ mb: 1, mt: 1 }}>
            입력 데이터 (JSON)
          </Typography>
          <TextField
            multiline
            fullWidth
            minRows={3}
            maxRows={10}
            value={inputData}
            onChange={(e) => setInputData(e.target.value)}
            slotProps={{
              input: {
                sx: { fontFamily: "monospace", fontSize: 13 },
              },
            }}
          />
          {runError && (
            <Alert severity="error" sx={{ mt: 1 }}>
              {runError}
            </Alert>
          )}
          {result && (
            <Box sx={{ mt: 1, p: 1, bgcolor: "grey.100", borderRadius: 1, maxHeight: 200, overflow: "auto" }}>
              <Typography variant="body2" component="pre" sx={{ fontFamily: "monospace", fontSize: 12, whiteSpace: "pre-wrap" }}>
                {result}
              </Typography>
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setRunDialog({ open: false, preset: "" })}>닫기</Button>
          <Button variant="contained" onClick={handleRun} disabled={runMutation.isPending}>
            {runMutation.isPending ? <CircularProgress size={20} /> : "실행"}
          </Button>
        </DialogActions>
      </Dialog>
    </>
  );
}

// ── Tab: Chunking Config ──
function ChunkingConfigTab() {
  const queryClient = useQueryClient();
  const [form, setForm] = useState<{ chunk_strategy: string; chunk_size: number; chunk_overlap: number } | null>(null);
  const [snack, setSnack] = useState<{ open: boolean; message: string; severity: "success" | "error" }>({
    open: false,
    message: "",
    severity: "success",
  });

  const { data, isLoading, error } = useQuery({
    queryKey: ["admin-chunking-config"],
    queryFn: () => adminApi.getChunkingConfig(),
    select: (res) => res.data as { chunk_strategy: string; chunk_size: number; chunk_overlap: number },
  });

  // Sync form when data arrives
  const config = data;
  const currentForm = form ?? (config ?? null);

  const updateMutation = useMutation({
    mutationFn: (payload: { chunk_strategy?: string; chunk_size?: number; chunk_overlap?: number }) =>
      adminApi.updateChunkingConfig(payload),
    onSuccess: () => {
      setSnack({ open: true, message: "청킹 설정이 저장되었습니다.", severity: "success" });
      queryClient.invalidateQueries({ queryKey: ["admin-chunking-config"] });
      setForm(null);
    },
    onError: () => {
      setSnack({ open: true, message: "저장에 실패했습니다.", severity: "error" });
    },
  });

  if (isLoading) return <CircularProgress />;
  if (error || !currentForm)
    return <Alert severity="error">청킹 설정을 불러올 수 없습니다.</Alert>;

  const handleSave = () => {
    if (!currentForm) return;
    updateMutation.mutate({
      chunk_strategy: currentForm.chunk_strategy,
      chunk_size: currentForm.chunk_size,
      chunk_overlap: currentForm.chunk_overlap,
    });
  };

  const strategies = ["adaptive", "recursive", "semantic", "table", "hierarchical"];

  return (
    <>
      <Typography variant="h6" gutterBottom>
        청킹 파라미터 설정
      </Typography>
      <Alert severity="info" sx={{ mb: 3 }}>
        변경된 설정은 새로 인제스트하는 문서부터 적용됩니다. 기존 문서에 적용하려면 재인제스트가 필요합니다.
      </Alert>
      <Card variant="outlined" sx={{ maxWidth: 560 }}>
        <CardContent sx={{ display: "flex", flexDirection: "column", gap: 3 }}>
          <Box>
            <Box sx={{ display: "flex", alignItems: "center", gap: 0.5, mb: 0.5 }}>
              <Typography variant="body2" color="text.secondary">청킹 전략</Typography>
              <Tooltip title="adaptive: 문서 구조에 따라 자동 분할 (권장) | recursive: 고정 크기 반복 분할 | semantic: 의미 단위 분할 | table: 표 구조 보존 분할" arrow>
                <HelpOutlineIcon sx={{ fontSize: 16, color: "text.secondary", cursor: "help" }} />
              </Tooltip>
            </Box>
            <FormControl fullWidth>
              <InputLabel id="chunk-strategy-label">청킹 전략 (Chunk Strategy)</InputLabel>
              <Select
                labelId="chunk-strategy-label"
                label="청킹 전략 (Chunk Strategy)"
                value={currentForm.chunk_strategy}
                onChange={(e) =>
                  setForm({ ...currentForm, chunk_strategy: e.target.value })
                }
              >
                {strategies.map((s) => (
                  <MenuItem key={s} value={s}>
                    {s}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Box>

          <TextField
            label="청크 크기 (Chunk Size)"
            helperText="100 ~ 4000 자"
            type="number"
            inputProps={{ min: 100, max: 4000 }}
            value={currentForm.chunk_size}
            onChange={(e) =>
              setForm({ ...currentForm, chunk_size: Number(e.target.value) })
            }
            fullWidth
          />

          <TextField
            label="청크 오버랩 (Chunk Overlap)"
            helperText="0 ~ 500 자"
            type="number"
            inputProps={{ min: 0, max: 500 }}
            value={currentForm.chunk_overlap}
            onChange={(e) =>
              setForm({ ...currentForm, chunk_overlap: Number(e.target.value) })
            }
            fullWidth
          />

          <Button
            variant="contained"
            onClick={handleSave}
            disabled={updateMutation.isPending}
            startIcon={updateMutation.isPending ? <CircularProgress size={18} /> : undefined}
          >
            저장
          </Button>
        </CardContent>
      </Card>

      <Snackbar
        open={snack.open}
        autoHideDuration={3000}
        onClose={() => setSnack({ ...snack, open: false })}
        message={snack.message}
      />
    </>
  );
}

// ── Tab: Governance ──
interface GovernanceChange {
  id: string;
  change_type: string;
  description: string;
  current_value: Record<string, unknown>;
  proposed_value: Record<string, unknown>;
  status: string;
  created_by: string;
  created_at: string;
  test_results: {
    tested_at: string;
    tested_by: string;
    baseline_accuracy: number;
    proposed_accuracy: number;
    sample_size: number;
    passed: boolean;
    details: string;
  } | null;
  approved_by: string | null;
  approved_at: string | null;
  applied_at: string | null;
  comment: string;
}

const STATUS_LABELS: Record<string, string> = {
  draft: "초안",
  testing: "테스트 중",
  pending_approval: "승인 대기",
  approved: "승인됨",
  applied: "적용됨",
  rejected: "거부됨",
};

const STATUS_COLORS: Record<string, "default" | "info" | "warning" | "success" | "primary" | "error"> = {
  draft: "default",
  testing: "info",
  pending_approval: "warning",
  approved: "success",
  applied: "primary",
  rejected: "error",
};

const CHANGE_TYPE_LABELS: Record<string, string> = {
  prompt: "프롬프트",
  chunking: "청킹",
  model: "모델",
  retriever: "검색기",
};

function GovernanceTab() {
  const queryClient = useQueryClient();
  const [statusFilter, setStatusFilter] = useState<string | undefined>(undefined);
  const [createDialogOpen, setCreateDialogOpen] = useState(false);
  const [approveDialogId, setApproveDialogId] = useState<string | null>(null);
  const [approveDecision, setApproveDecision] = useState(true);
  const [approveComment, setApproveComment] = useState("");
  const [detailId, setDetailId] = useState<string | null>(null);
  const [snack, setSnack] = useState<{ open: boolean; message: string; severity: "success" | "error" }>({
    open: false,
    message: "",
    severity: "success",
  });

  const [newForm, setNewForm] = useState({
    change_type: "prompt",
    description: "",
    current_value: "",
    proposed_value: "",
  });

  const { data, isLoading, error } = useQuery({
    queryKey: ["governance-changes", statusFilter],
    queryFn: () => governanceApi.list(statusFilter),
    select: (res) => res.data as { changes: GovernanceChange[]; total: number },
    refetchInterval: 10000,
  });

  const changes = data?.changes ?? [];

  const createMutation = useMutation({
    mutationFn: (payload: { change_type: string; description: string; current_value: Record<string, unknown>; proposed_value: Record<string, unknown> }) =>
      governanceApi.create(payload),
    onSuccess: () => {
      setSnack({ open: true, message: "변경 요청이 생성되었습니다.", severity: "success" });
      setCreateDialogOpen(false);
      setNewForm({ change_type: "prompt", description: "", current_value: "", proposed_value: "" });
      queryClient.invalidateQueries({ queryKey: ["governance-changes"] });
    },
    onError: () => setSnack({ open: true, message: "생성에 실패했습니다.", severity: "error" }),
  });

  const simulateMutation = useMutation({
    mutationFn: (id: string) => governanceApi.simulate(id),
    onSuccess: () => {
      setSnack({ open: true, message: "시뮬레이션 완료. 승인 대기 상태로 이동했습니다.", severity: "success" });
      queryClient.invalidateQueries({ queryKey: ["governance-changes"] });
    },
    onError: () => setSnack({ open: true, message: "시뮬레이션 실패.", severity: "error" }),
  });

  const approveMutation = useMutation({
    mutationFn: ({ id, approved, comment }: { id: string; approved: boolean; comment: string }) =>
      governanceApi.approve(id, { approved, comment }),
    onSuccess: (_, vars) => {
      setSnack({ open: true, message: vars.approved ? "승인되었습니다." : "거부되었습니다.", severity: "success" });
      setApproveDialogId(null);
      setApproveComment("");
      queryClient.invalidateQueries({ queryKey: ["governance-changes"] });
    },
    onError: () => setSnack({ open: true, message: "처리에 실패했습니다.", severity: "error" }),
  });

  const applyMutation = useMutation({
    mutationFn: (id: string) => governanceApi.apply(id),
    onSuccess: () => {
      setSnack({ open: true, message: "운영 환경에 적용되었습니다.", severity: "success" });
      queryClient.invalidateQueries({ queryKey: ["governance-changes"] });
    },
    onError: () => setSnack({ open: true, message: "적용에 실패했습니다.", severity: "error" }),
  });

  const handleCreate = () => {
    let currentVal: Record<string, unknown>;
    let proposedVal: Record<string, unknown>;
    try {
      currentVal = newForm.current_value ? JSON.parse(newForm.current_value) : {};
      proposedVal = newForm.proposed_value ? JSON.parse(newForm.proposed_value) : {};
    } catch {
      setSnack({ open: true, message: "현재/제안값은 JSON 형식이어야 합니다.", severity: "error" });
      return;
    }
    createMutation.mutate({
      change_type: newForm.change_type,
      description: newForm.description,
      current_value: currentVal,
      proposed_value: proposedVal,
    });
  };

  const detailChange = changes.find((c) => c.id === detailId);

  const STATUS_FILTERS = [
    { label: "전체", value: undefined },
    { label: "초안", value: "draft" },
    { label: "테스트 중", value: "testing" },
    { label: "승인 대기", value: "pending_approval" },
    { label: "승인됨", value: "approved" },
    { label: "적용됨", value: "applied" },
    { label: "거부됨", value: "rejected" },
  ];

  return (
    <>
      <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", mb: 2 }}>
        <Box sx={{ display: "flex", alignItems: "center", gap: 0.5 }}>
          <Typography variant="h6">RAG 설정 변경 관리</Typography>
          <Tooltip title="변경 관리 절차: 초안 작성 → 시뮬레이션 테스트 → 관리자 승인 → 운영 적용. 승인 없이는 운영 환경에 반영되지 않습니다." arrow>
            <HelpOutlineIcon sx={{ fontSize: 17, color: "text.secondary", cursor: "help" }} />
          </Tooltip>
        </Box>
        <Button variant="contained" startIcon={<AddIcon />} onClick={() => setCreateDialogOpen(true)}>
          새 변경 요청
        </Button>
      </Box>

      <Alert severity="info" sx={{ mb: 2 }}>
        RAG 설정 변경은 시뮬레이션 테스트 후 관리자 승인을 거쳐 운영에 적용됩니다.
      </Alert>

      {/* Status filter chips */}
      <Box sx={{ display: "flex", gap: 1, flexWrap: "wrap", mb: 2 }}>
        {STATUS_FILTERS.map(({ label, value }) => (
          <Chip
            key={label}
            label={label}
            clickable
            color={statusFilter === value ? "primary" : "default"}
            variant={statusFilter === value ? "filled" : "outlined"}
            onClick={() => setStatusFilter(value)}
          />
        ))}
      </Box>

      {isLoading && <CircularProgress />}
      {error && <Alert severity="error">변경 목록을 불러올 수 없습니다.</Alert>}

      {!isLoading && !error && (
        <TableContainer component={Paper} variant="outlined">
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>ID</TableCell>
                <TableCell>유형</TableCell>
                <TableCell>설명</TableCell>
                <TableCell>상태</TableCell>
                <TableCell>요청자</TableCell>
                <TableCell>요청일</TableCell>
                <TableCell align="right">액션</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {changes.length === 0 && (
                <TableRow>
                  <TableCell colSpan={7} align="center" sx={{ color: "text.secondary", py: 4 }}>
                    변경 요청이 없습니다.
                  </TableCell>
                </TableRow>
              )}
              {changes.map((c) => (
                <TableRow key={c.id} hover>
                  <TableCell>
                    <Tooltip title={c.id}>
                      <Typography variant="body2" sx={{ fontFamily: "monospace", fontSize: "0.75rem" }}>
                        {c.id.slice(0, 8)}…
                      </Typography>
                    </Tooltip>
                  </TableCell>
                  <TableCell>{CHANGE_TYPE_LABELS[c.change_type] ?? c.change_type}</TableCell>
                  <TableCell sx={{ maxWidth: 240 }}>
                    <Typography variant="body2" noWrap>
                      {c.description}
                    </Typography>
                  </TableCell>
                  <TableCell>
                    <Chip
                      label={STATUS_LABELS[c.status] ?? c.status}
                      color={STATUS_COLORS[c.status] ?? "default"}
                      size="small"
                    />
                  </TableCell>
                  <TableCell>{c.created_by}</TableCell>
                  <TableCell>
                    <Typography variant="body2" noWrap>
                      {new Date(c.created_at).toLocaleDateString("ko-KR")}
                    </Typography>
                  </TableCell>
                  <TableCell align="right">
                    <Box sx={{ display: "flex", gap: 0.5, justifyContent: "flex-end" }}>
                      <Tooltip title="상세 보기">
                        <IconButton size="small" onClick={() => setDetailId(c.id)}>
                          <InfoIcon fontSize="small" />
                        </IconButton>
                      </Tooltip>
                      {c.status === "draft" && (
                        <Tooltip title="시뮬레이션 실행">
                          <span>
                            <IconButton
                              size="small"
                              color="info"
                              onClick={() => simulateMutation.mutate(c.id)}
                              disabled={simulateMutation.isPending}
                            >
                              <PlayArrowIcon fontSize="small" />
                            </IconButton>
                          </span>
                        </Tooltip>
                      )}
                      {c.status === "pending_approval" && (
                        <>
                          <Tooltip title="승인">
                            <IconButton
                              size="small"
                              color="success"
                              onClick={() => { setApproveDialogId(c.id); setApproveDecision(true); setApproveComment(""); }}
                            >
                              <CheckCircleIcon fontSize="small" />
                            </IconButton>
                          </Tooltip>
                          <Tooltip title="거부">
                            <IconButton
                              size="small"
                              color="error"
                              onClick={() => { setApproveDialogId(c.id); setApproveDecision(false); setApproveComment(""); }}
                            >
                              <ErrorIcon fontSize="small" />
                            </IconButton>
                          </Tooltip>
                        </>
                      )}
                      {c.status === "approved" && (
                        <Tooltip title="운영 적용">
                          <span>
                            <IconButton
                              size="small"
                              color="primary"
                              onClick={() => applyMutation.mutate(c.id)}
                              disabled={applyMutation.isPending}
                            >
                              <CheckCircleIcon fontSize="small" />
                            </IconButton>
                          </span>
                        </Tooltip>
                      )}
                    </Box>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      )}

      {/* Create Dialog */}
      <Dialog open={createDialogOpen} onClose={() => setCreateDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>새 변경 요청</DialogTitle>
        <DialogContent sx={{ display: "flex", flexDirection: "column", gap: 2, pt: 2 }}>
          <FormControl fullWidth>
            <InputLabel id="gov-type-label">변경 유형</InputLabel>
            <Select
              labelId="gov-type-label"
              label="변경 유형"
              value={newForm.change_type}
              onChange={(e) => setNewForm({ ...newForm, change_type: e.target.value })}
            >
              <MenuItem value="prompt">프롬프트</MenuItem>
              <MenuItem value="chunking">청킹 파라미터</MenuItem>
              <MenuItem value="model">모델 설정</MenuItem>
              <MenuItem value="retriever">검색기 설정</MenuItem>
            </Select>
          </FormControl>
          <TextField
            label="변경 설명"
            multiline
            rows={2}
            value={newForm.description}
            onChange={(e) => setNewForm({ ...newForm, description: e.target.value })}
            fullWidth
            required
            placeholder="어떤 변경이며 왜 필요한지 설명해 주세요"
          />
          <TextField
            label="현재 값 (JSON)"
            multiline
            rows={3}
            value={newForm.current_value}
            onChange={(e) => setNewForm({ ...newForm, current_value: e.target.value })}
            fullWidth
            placeholder='예: {"system_prompt": "기존 프롬프트 내용..."}'
            helperText="JSON 형식으로 입력"
          />
          <TextField
            label="제안 값 (JSON)"
            multiline
            rows={3}
            value={newForm.proposed_value}
            onChange={(e) => setNewForm({ ...newForm, proposed_value: e.target.value })}
            fullWidth
            placeholder='예: {"system_prompt": "개선된 프롬프트 내용..."}'
            helperText="JSON 형식으로 입력"
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setCreateDialogOpen(false)}>취소</Button>
          <Button
            variant="contained"
            onClick={handleCreate}
            disabled={createMutation.isPending || !newForm.description}
          >
            {createMutation.isPending ? <CircularProgress size={18} /> : "생성"}
          </Button>
        </DialogActions>
      </Dialog>

      {/* Approve/Reject Dialog */}
      <Dialog open={!!approveDialogId} onClose={() => setApproveDialogId(null)} maxWidth="xs" fullWidth>
        <DialogTitle>{approveDecision ? "변경 요청 승인" : "변경 요청 거부"}</DialogTitle>
        <DialogContent sx={{ pt: 2 }}>
          <TextField
            label="코멘트 (선택)"
            multiline
            rows={3}
            fullWidth
            value={approveComment}
            onChange={(e) => setApproveComment(e.target.value)}
            placeholder={approveDecision ? "승인 사유를 입력하세요" : "거부 사유를 입력하세요"}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setApproveDialogId(null)}>취소</Button>
          <Button
            variant="contained"
            color={approveDecision ? "success" : "error"}
            onClick={() => {
              if (approveDialogId) {
                approveMutation.mutate({ id: approveDialogId, approved: approveDecision, comment: approveComment });
              }
            }}
            disabled={approveMutation.isPending}
          >
            {approveMutation.isPending ? <CircularProgress size={18} /> : approveDecision ? "승인" : "거부"}
          </Button>
        </DialogActions>
      </Dialog>

      {/* Detail Dialog */}
      {detailChange && (
        <Dialog open={!!detailId} onClose={() => setDetailId(null)} maxWidth="md" fullWidth>
          <DialogTitle>변경 요청 상세</DialogTitle>
          <DialogContent>
            <Box sx={{ display: "flex", flexDirection: "column", gap: 2, pt: 1 }}>
              <Box sx={{ display: "flex", gap: 1, alignItems: "center" }}>
                <Chip label={STATUS_LABELS[detailChange.status] ?? detailChange.status} color={STATUS_COLORS[detailChange.status] ?? "default"} />
                <Chip label={CHANGE_TYPE_LABELS[detailChange.change_type] ?? detailChange.change_type} variant="outlined" />
              </Box>
              <Typography variant="body1">{detailChange.description}</Typography>
              <Typography variant="caption" color="text.secondary">
                요청자: {detailChange.created_by} | {new Date(detailChange.created_at).toLocaleString("ko-KR")}
              </Typography>
              {detailChange.approved_by && (
                <Typography variant="caption" color="text.secondary">
                  {detailChange.status === "rejected" ? "거부자" : "승인자"}: {detailChange.approved_by} | {detailChange.approved_at ? new Date(detailChange.approved_at).toLocaleString("ko-KR") : ""}
                </Typography>
              )}
              {detailChange.comment && (
                <Alert severity={detailChange.status === "rejected" ? "error" : "info"}>
                  코멘트: {detailChange.comment}
                </Alert>
              )}
              <Box>
                <Typography variant="subtitle2" gutterBottom>현재 값</Typography>
                <Paper variant="outlined" sx={{ p: 1.5, fontFamily: "monospace", fontSize: "0.8rem", whiteSpace: "pre-wrap", maxHeight: 150, overflow: "auto" }}>
                  {JSON.stringify(detailChange.current_value, null, 2)}
                </Paper>
              </Box>
              <Box>
                <Typography variant="subtitle2" gutterBottom>제안 값</Typography>
                <Paper variant="outlined" sx={{ p: 1.5, fontFamily: "monospace", fontSize: "0.8rem", whiteSpace: "pre-wrap", maxHeight: 150, overflow: "auto" }}>
                  {JSON.stringify(detailChange.proposed_value, null, 2)}
                </Paper>
              </Box>
              {detailChange.test_results && (
                <Box>
                  <Typography variant="subtitle2" gutterBottom>시뮬레이션 결과</Typography>
                  <Card variant="outlined">
                    <CardContent>
                      <Box sx={{ display: "flex", gap: 3, flexWrap: "wrap" }}>
                        <Box>
                          <Typography variant="caption" color="text.secondary">기준 정확도</Typography>
                          <Typography variant="h6">{detailChange.test_results.baseline_accuracy}%</Typography>
                        </Box>
                        <Box>
                          <Typography variant="caption" color="text.secondary">제안 정확도</Typography>
                          <Typography variant="h6" color={detailChange.test_results.proposed_accuracy >= detailChange.test_results.baseline_accuracy ? "success.main" : "error.main"}>
                            {detailChange.test_results.proposed_accuracy}%
                          </Typography>
                        </Box>
                        <Box>
                          <Typography variant="caption" color="text.secondary">샘플 수</Typography>
                          <Typography variant="h6">{detailChange.test_results.sample_size}</Typography>
                        </Box>
                        <Box>
                          <Typography variant="caption" color="text.secondary">결과</Typography>
                          <Chip
                            label={detailChange.test_results.passed ? "통과" : "실패"}
                            color={detailChange.test_results.passed ? "success" : "error"}
                            size="small"
                          />
                        </Box>
                      </Box>
                      <Typography variant="body2" sx={{ mt: 1 }} color="text.secondary">
                        {detailChange.test_results.details}
                      </Typography>
                    </CardContent>
                  </Card>
                </Box>
              )}
              {detailChange.applied_at && (
                <Typography variant="caption" color="success.main">
                  적용 완료: {new Date(detailChange.applied_at).toLocaleString("ko-KR")}
                </Typography>
              )}
            </Box>
          </DialogContent>
          <DialogActions>
            <Button onClick={() => setDetailId(null)}>닫기</Button>
          </DialogActions>
        </Dialog>
      )}

      <Snackbar
        open={snack.open}
        autoHideDuration={3000}
        onClose={() => setSnack({ ...snack, open: false })}
        message={snack.message}
      />
    </>
  );
}

// ── Notification Bell ──

type NotificationType = "feedback" | "document" | "system" | "user";

interface Notification {
  id: number;
  type: NotificationType;
  title: string;
  timestamp: Date;
  read: boolean;
}

function relativeTime(date: Date): string {
  const diffMs = Date.now() - date.getTime();
  const diffSec = Math.floor(diffMs / 1000);
  if (diffSec < 60) return "방금 전";
  const diffMin = Math.floor(diffSec / 60);
  if (diffMin < 60) return `${diffMin}분 전`;
  const diffHr = Math.floor(diffMin / 60);
  if (diffHr < 24) return `${diffHr}시간 전`;
  const diffDay = Math.floor(diffHr / 24);
  return `${diffDay}일 전`;
}

const NOTIFICATION_TYPE_META: Record<NotificationType, { icon: React.ReactNode; color: string }> = {
  feedback: { icon: <ThumbUpIcon fontSize="small" />, color: "#4caf50" },
  document: { icon: <DescriptionIcon fontSize="small" />, color: "#1976d2" },
  system: { icon: <WarningAmberIcon fontSize="small" />, color: "#f57c00" },
  user: { icon: <PersonIcon fontSize="small" />, color: "#9c27b0" },
};

function NotificationBell() {
  const [notifications, setNotifications] = useState<Notification[]>([]);
  const [anchorEl, setAnchorEl] = useState<HTMLButtonElement | null>(null);

  useEffect(() => {
    logsApi.searchAccess({ page: 1, page_size: 10 })
      .then((resp) => {
        const logs = resp.data?.logs || [];
        const mapped: Notification[] = logs.map((log: any, idx: number) => {
          let type: NotificationType = "system";
          const action = (log.action || "").toUpperCase();
          if (action.includes("LOGIN") || action.includes("LOGOUT") || action.includes("USER") || action.includes("ROLE")) type = "user";
          else if (action.includes("DOCUMENT") || action.includes("UPLOAD") || action.includes("INGEST")) type = "document";
          else if (action.includes("FEEDBACK")) type = "feedback";
          return {
            id: idx + 1,
            type,
            title: log.action + (log.details ? ` — ${String(log.details).slice(0, 50)}` : ""),
            timestamp: new Date(log.timestamp),
            read: idx >= 3,
          };
        });
        setNotifications(mapped);
      })
      .catch(() => {
        setNotifications([]);
      });
  }, []);

  const unreadCount = notifications.filter((n) => !n.read).length;
  const open = Boolean(anchorEl);

  const handleOpen = (event: React.MouseEvent<HTMLButtonElement>) => {
    setAnchorEl(event.currentTarget);
  };

  const handleClose = () => {
    setAnchorEl(null);
  };

  const markAllRead = () => {
    setNotifications((prev) => prev.map((n) => ({ ...n, read: true })));
  };

  const markRead = (id: number) => {
    setNotifications((prev) =>
      prev.map((n) => (n.id === id ? { ...n, read: true } : n))
    );
  };

  return (
    <>
      <Tooltip title="알림">
        <IconButton onClick={handleOpen} size="medium" color="inherit">
          <Badge
            badgeContent={unreadCount}
            color="error"
            max={99}
            sx={{
              "& .MuiBadge-badge": {
                fontSize: "0.65rem",
                minWidth: 16,
                height: 16,
                padding: "0 4px",
              },
            }}
          >
            {unreadCount > 0 ? (
              <NotificationsIcon />
            ) : (
              <NotificationsNoneIcon />
            )}
          </Badge>
        </IconButton>
      </Tooltip>

      <Popover
        open={open}
        anchorEl={anchorEl}
        onClose={handleClose}
        anchorOrigin={{ vertical: "bottom", horizontal: "right" }}
        transformOrigin={{ vertical: "top", horizontal: "right" }}
        slotProps={{
          paper: {
            sx: {
              width: 380,
              maxHeight: 520,
              display: "flex",
              flexDirection: "column",
              borderRadius: 2,
              boxShadow: 8,
            },
          },
        }}
      >
        <Box
          sx={{
            px: 2,
            py: 1.5,
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            borderBottom: "1px solid",
            borderColor: "divider",
            flexShrink: 0,
          }}
        >
          <Typography variant="subtitle1" sx={{ fontWeight: 700 }}>
            알림
            {unreadCount > 0 && (
              <Chip
                label={unreadCount}
                size="small"
                color="error"
                sx={{ ml: 1, height: 18, fontSize: "0.65rem" }}
              />
            )}
          </Typography>
          {unreadCount > 0 && (
            <Button
              size="small"
              variant="text"
              onClick={markAllRead}
              sx={{ fontSize: "0.75rem", textTransform: "none" }}
            >
              모두 읽음 처리
            </Button>
          )}
        </Box>

        <Box sx={{ overflowY: "auto", flex: 1 }}>
          {notifications.length === 0 ? (
            <Box
              sx={{
                py: 6,
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                gap: 1,
                color: "text.secondary",
              }}
            >
              <NotificationsNoneIcon sx={{ fontSize: 40, opacity: 0.4 }} />
              <Typography variant="body2">알림 없음</Typography>
            </Box>
          ) : (
            <List disablePadding>
              {notifications.map((notif, idx) => {
                const meta = NOTIFICATION_TYPE_META[notif.type];
                return (
                  <Box key={notif.id}>
                    <ListItem
                      onClick={() => markRead(notif.id)}
                      sx={{
                        px: 2,
                        py: 1.2,
                        cursor: "pointer",
                        backgroundColor: notif.read ? "transparent" : "action.hover",
                        "&:hover": { backgroundColor: "action.selected" },
                        transition: "background-color 0.15s",
                      }}
                    >
                      <ListItemAvatar sx={{ minWidth: 44 }}>
                        <Avatar
                          sx={{
                            width: 34,
                            height: 34,
                            bgcolor: meta.color + "20",
                            color: meta.color,
                          }}
                        >
                          {meta.icon}
                        </Avatar>
                      </ListItemAvatar>
                      <ListItemText
                        primary={
                          <Typography
                            variant="body2"
                            sx={{
                              fontWeight: notif.read ? 400 : 600,
                              lineHeight: 1.4,
                              color: notif.read ? "text.secondary" : "text.primary",
                              pr: 2,
                            }}
                          >
                            {notif.title}
                          </Typography>
                        }
                        secondary={
                          <Typography variant="caption" color="text.disabled">
                            {relativeTime(notif.timestamp)}
                          </Typography>
                        }
                      />
                      {!notif.read && (
                        <Box sx={{ display: "flex", alignItems: "center", alignSelf: "center", ml: 0.5 }}>
                          <FiberManualRecordIcon
                            sx={{ fontSize: 10, color: "primary.main" }}
                          />
                        </Box>
                      )}
                    </ListItem>
                    {idx < notifications.length - 1 && (
                      <Divider variant="inset" component="li" sx={{ ml: 7 }} />
                    )}
                  </Box>
                );
              })}
            </List>
          )}
        </Box>
      </Popover>
    </>
  );
}

// ── Sidebar navigation structure ──

const NAV_SECTIONS = [
  {
    label: "개요",
    items: [
      { key: "dashboard", label: "대시보드", icon: <DashboardIcon fontSize="small" /> },
    ],
  },
  {
    label: "AI / RAG",
    items: [
      { key: "rag-settings", label: "시스템 설정", icon: <SettingsIcon fontSize="small" /> },
      { key: "providers", label: "LLM 프로바이더", icon: <StorageIcon fontSize="small" /> },
      { key: "models", label: "모델 관리", icon: <MemoryIcon fontSize="small" /> },
      { key: "prompts", label: "프롬프트 관리", icon: <DescriptionIcon fontSize="small" /> },
      { key: "prompt-simulator", label: "프롬프트 시뮬레이터", icon: <ScienceOutlinedIcon fontSize="small" /> },
      { key: "batch-evaluator", label: "배치 평가기", icon: <SpeedIcon fontSize="small" /> },
      { key: "chunking", label: "청킹 설정", icon: <TuneIcon fontSize="small" /> },
      { key: "playground", label: "LLM 테스트", icon: <ScienceIcon fontSize="small" /> },
      { key: "ocr-test", label: "OCR 테스트", icon: <DocumentScannerIcon fontSize="small" /> },
    ],
  },
  {
    label: "도구",
    items: [
      { key: "mcp-tools", label: "MCP 도구", icon: <BuildIcon fontSize="small" /> },
      { key: "custom-tools", label: "커스텀 도구", icon: <AppRegistrationIcon fontSize="small" /> },
      { key: "report-templates", label: "보고서 템플릿", icon: <DescriptionIcon fontSize="small" /> },
      { key: "workflows", label: "워크플로우", icon: <AccountTreeIcon fontSize="small" /> },
      { key: "content", label: "콘텐츠 관리", icon: <FolderIcon fontSize="small" /> },
    ],
  },
  {
    label: "보안",
    items: [
      { key: "users", label: "사용자 관리", icon: <PersonIcon fontSize="small" /> },
      { key: "guardrails", label: "가드레일", icon: <SecurityIcon fontSize="small" /> },
      { key: "governance", label: "변경 관리", icon: <ShieldIcon fontSize="small" /> },
      { key: "audit-log", label: "감사 로그", icon: <HistoryIcon fontSize="small" /> },
    ],
  },
  {
    label: "시스템",
    items: [
      { key: "system-info", label: "시스템 정보", icon: <ComputerIcon fontSize="small" /> },
      { key: "cache", label: "캐시 관리", icon: <CachedIcon fontSize="small" /> },
      { key: "export", label: "데이터 내보내기", icon: <DownloadIcon fontSize="small" /> },
    ],
  },
];

const ADMIN_NAV_WIDTH = 220;

const TAB_COMPONENTS: Record<string, React.ComponentType> = {
  "dashboard": DashboardTab,
  "rag-settings": NewSystemSettingsTab,
  "providers": ProvidersTab,
  "models": ModelsTab,
  "prompts": PromptsTab,
  "prompt-simulator": PromptSimulatorTab,
  "batch-evaluator": BatchEvaluatorTab,
  "chunking": ChunkingConfigTab,
  "playground": LLMPlaygroundTab,
  "ocr-test": OcrTestTab,
  "mcp-tools": McpToolsTab,
  "custom-tools": CustomToolBuilder,
  "report-templates": ReportTemplateTab,
  "workflows": WorkflowsTab,
  "content": ContentManager,
  "users": UserManagementTab,
  "guardrails": GuardrailsTab,
  "governance": GovernanceTab,
  "audit-log": AuditLogTab,
  "system-info": SystemInfoTab,
  "cache": CacheManagementTab,
  "export": ExportCenterTab,
};

export default function AdminPage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const [activeKey, setActiveKey] = useState(() => {
    const tab = searchParams.get("tab") || "dashboard";
    return TAB_COMPONENTS[tab] ? tab : "dashboard";
  });

  const handleTabChange = (key: string) => {
    setActiveKey(key);
    setSearchParams({ tab: key }, { replace: true });
  };

  const ActiveComponent = TAB_COMPONENTS[activeKey] ?? DashboardTab;

  return (
    <Layout title="관리자 설정" noPadding headerActions={<NotificationBell />}>
      <Box sx={{ display: "flex", height: "100%" }}>
        {/* Sidebar Navigation */}
        <Box
          sx={{
            width: ADMIN_NAV_WIDTH,
            flexShrink: 0,
            borderRight: "1px solid",
            borderColor: "divider",
            overflowY: "auto",
            bgcolor: "background.paper",
          }}
        >
          <List disablePadding dense>
            {NAV_SECTIONS.map((section) => (
              <Box key={section.label}>
                <ListSubheader
                  sx={{
                    lineHeight: "32px",
                    fontSize: "0.7rem",
                    fontWeight: 700,
                    textTransform: "uppercase",
                    color: "text.disabled",
                    letterSpacing: 0.5,
                    bgcolor: "background.paper",
                  }}
                >
                  {section.label}
                </ListSubheader>
                {section.items.map((item) => (
                  <ListItemButton
                    key={item.key}
                    selected={activeKey === item.key}
                    onClick={() => handleTabChange(item.key)}
                    sx={{
                      py: 0.6,
                      px: 2,
                      minHeight: 36,
                      "&.Mui-selected": {
                        bgcolor: "primary.main",
                        color: "primary.contrastText",
                        "&:hover": { bgcolor: "primary.dark" },
                        "& .MuiListItemIcon-root": { color: "inherit" },
                      },
                    }}
                  >
                    <ListItemIcon sx={{ minWidth: 32 }}>{item.icon}</ListItemIcon>
                    <ListItemText
                      primary={item.label}
                      primaryTypographyProps={{ fontSize: "0.82rem" }}
                    />
                  </ListItemButton>
                ))}
              </Box>
            ))}
          </List>
        </Box>

        {/* Content Area */}
        <Box sx={{ flex: 1, overflow: "auto", p: 3 }}>
          <ActiveComponent />
        </Box>
      </Box>
    </Layout>
  );
}
