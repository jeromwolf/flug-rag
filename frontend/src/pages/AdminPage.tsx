import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  Box,
  Tabs,
  Tab,
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
import { adminApi, mcpApi, workflowsApi, guardrailsApi, authApi, governanceApi, feedbackApi, statsApi, ocrApi } from "../api/client";
import Layout from "../components/Layout";
import CustomToolBuilder from "../components/CustomToolBuilder";
import ContentManager from "../components/ContentManager";
import { PasswordConfirmDialog } from "../components/chat/ChatDialogs";

interface TabPanelProps {
  children: React.ReactNode;
  value: number;
  index: number;
}

function TabPanel({ children, value, index }: TabPanelProps) {
  if (value !== index) return null;
  return <Box sx={{ py: 3 }}>{children}</Box>;
}

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

  const users: unknown[] = usersData?.data?.users ?? usersData?.data ?? [];
  const metrics = metricsData?.data as {
    cpu?: number;
    memory?: number;
    disk?: number;
    avg_response_time_ms?: number;
    cache_hit_rate?: number;
    active_sessions?: number;
    status?: string;
  } | undefined;

  // Today's query count from usage stats (or mock if not available)
  const todayQueries: number = (() => {
    const usage = usageData?.data as { daily?: Array<{ count?: number }> } | undefined;
    if (usage?.daily && usage.daily.length > 0) {
      return usage.daily[usage.daily.length - 1]?.count ?? 0;
    }
    // NOTE: needs backend /stats/usage?period=day endpoint to return daily breakdown
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

  // Mock recent error logs — NOTE: needs backend /logs/errors endpoint
  const recentErrors: Array<{ timestamp: string; message: string; level: string }> = [
    { timestamp: new Date(Date.now() - 5 * 60_000).toISOString(), message: "LLM 응답 타임아웃 (32s)", level: "ERROR" },
    { timestamp: new Date(Date.now() - 18 * 60_000).toISOString(), message: "임베딩 배치 처리 실패 (doc_1234)", level: "WARNING" },
    { timestamp: new Date(Date.now() - 42 * 60_000).toISOString(), message: "벡터 검색 응답 지연 (4.2s)", level: "WARNING" },
    { timestamp: new Date(Date.now() - 70 * 60_000).toISOString(), message: "가드레일 트리거: 키워드 필터", level: "INFO" },
    { timestamp: new Date(Date.now() - 120 * 60_000).toISOString(), message: "캐시 미스율 임계값 초과 (68%)", level: "WARNING" },
  ];

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
                        sx={{
                          display: "flex",
                          alignItems: "flex-start",
                          gap: 1.5,
                          p: 1,
                          borderRadius: 1,
                          bgcolor: "action.hover",
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
                <Chip label="Mock" size="small" variant="outlined" color="default" />
                {/* NOTE: replace mock data with real /logs/errors or /logs/access?action=error endpoint */}
              </Box>
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
            </CardContent>
          </Card>
        </Grid>
      </Grid>
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

  const rawUsers: UserRecord[] = (data?.data?.users ?? data?.data ?? []) as UserRecord[];

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

// ── Tab: Cache Management ──
interface CacheCategory {
  key: string;
  label: string;
  description: string;
  unit: string;
}

interface CacheConfigData {
  cache_enabled: boolean;
  cache_type: string;
  ttl: Record<string, number>;
  categories: CacheCategory[];
}

function CacheManagementTab() {
  const queryClient = useQueryClient();
  const [localTtl, setLocalTtl] = useState<Record<string, number> | null>(null);
  const [snack, setSnack] = useState<{ open: boolean; message: string; severity: "success" | "error" }>({
    open: false,
    message: "",
    severity: "success",
  });
  const [clearConfirmCategory, setClearConfirmCategory] = useState<string | null>(null);

  const { data, isLoading, error } = useQuery({
    queryKey: ["admin-cache-config"],
    queryFn: () => adminApi.getCacheConfig(),
    select: (res) => res.data as CacheConfigData,
  });

  const config = data;
  const effectiveTtl = localTtl ?? config?.ttl ?? {};

  const updateMutation = useMutation({
    mutationFn: (payload: Record<string, number>) => adminApi.updateCacheConfig(payload),
    onSuccess: () => {
      setSnack({ open: true, message: "캐시 TTL이 저장되었습니다.", severity: "success" });
      queryClient.invalidateQueries({ queryKey: ["admin-cache-config"] });
      setLocalTtl(null);
    },
    onError: () => {
      setSnack({ open: true, message: "저장에 실패했습니다.", severity: "error" });
    },
  });

  const clearMutation = useMutation({
    mutationFn: (category?: string) => adminApi.clearCache(category),
    onSuccess: (_, category) => {
      setSnack({
        open: true,
        message: category ? `'${category}' 캐시가 초기화되었습니다.` : "전체 캐시가 초기화되었습니다.",
        severity: "success",
      });
      setClearConfirmCategory(null);
    },
    onError: () => {
      setSnack({ open: true, message: "캐시 초기화에 실패했습니다.", severity: "error" });
      setClearConfirmCategory(null);
    },
  });

  const handleSave = () => {
    if (!localTtl) return;
    const numeric: Record<string, number> = {};
    for (const [k, v] of Object.entries(localTtl)) {
      numeric[k] = Number(v);
    }
    updateMutation.mutate(numeric);
  };

  const formatTtl = (seconds: number): string => {
    if (seconds === 0) return "비활성화";
    if (seconds < 60) return `${seconds}초`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)}분 ${seconds % 60}초`;
    if (seconds < 86400) return `${Math.floor(seconds / 3600)}시간 ${Math.floor((seconds % 3600) / 60)}분`;
    return `${Math.floor(seconds / 86400)}일`;
  };

  if (isLoading) return <CircularProgress />;
  if (error || !config) return <Alert severity="error">캐시 설정을 불러올 수 없습니다.</Alert>;

  return (
    <>
      <Box sx={{ mb: 3, display: "flex", alignItems: "center", gap: 2 }}>
        <Box>
          <Typography variant="h6" fontWeight={600}>캐시 관리</Typography>
          <Typography variant="body2" color="text.secondary">
            캐시 유형: <strong>{config.cache_type === "memory" ? "인메모리" : "Redis"}</strong>
            {" · "}
            캐시 활성화: <strong>{config.cache_enabled ? "활성" : "비활성"}</strong>
          </Typography>
        </Box>
        {!config.cache_enabled && (
          <Alert severity="info" sx={{ flex: 1 }}>
            캐시가 비활성화 상태입니다. TTL 설정은 저장되지만 캐시가 활성화될 때 적용됩니다.
          </Alert>
        )}
      </Box>

      {/* TTL Settings per Category */}
      <Typography variant="subtitle1" fontWeight={600} gutterBottom>
        카테고리별 TTL 설정
      </Typography>
      <Card variant="outlined" sx={{ mb: 3 }}>
        <CardContent sx={{ display: "flex", flexDirection: "column", gap: 2 }}>
          {(config.categories ?? []).map((cat) => (
            <Box key={cat.key} sx={{ display: "flex", alignItems: "flex-start", gap: 2, flexWrap: "wrap" }}>
              <Box sx={{ flex: "1 1 220px" }}>
                <Typography variant="body2" fontWeight={500}>{cat.label}</Typography>
                <Typography variant="caption" color="text.secondary">{cat.description}</Typography>
              </Box>
              <Box sx={{ display: "flex", alignItems: "center", gap: 1, flex: "0 0 auto" }}>
                <TextField
                  size="small"
                  type="number"
                  label={`TTL (${cat.unit})`}
                  value={effectiveTtl[cat.key] ?? 0}
                  onChange={(e) =>
                    setLocalTtl({ ...effectiveTtl, [cat.key]: Number(e.target.value) })
                  }
                  inputProps={{ min: 0 }}
                  sx={{ width: 140 }}
                />
                <Typography variant="caption" color="text.secondary" sx={{ minWidth: 80 }}>
                  {formatTtl(effectiveTtl[cat.key] ?? 0)}
                </Typography>
                <Tooltip title={`'${cat.label}' 캐시만 초기화`}>
                  <IconButton
                    size="small"
                    color="warning"
                    onClick={() => setClearConfirmCategory(cat.key)}
                    disabled={clearMutation.isPending}
                  >
                    <DeleteSweepIcon fontSize="small" />
                  </IconButton>
                </Tooltip>
              </Box>
            </Box>
          ))}

          <Box sx={{ display: "flex", gap: 1, mt: 1 }}>
            <Button
              variant="contained"
              onClick={handleSave}
              disabled={!localTtl || updateMutation.isPending}
              startIcon={updateMutation.isPending ? <CircularProgress size={18} /> : undefined}
            >
              저장
            </Button>
            <Button
              variant="outlined"
              onClick={() => setLocalTtl(null)}
              disabled={!localTtl}
            >
              초기화
            </Button>
          </Box>
        </CardContent>
      </Card>

      {/* Global Cache Clear */}
      <Typography variant="subtitle1" fontWeight={600} gutterBottom>
        전체 캐시 초기화
      </Typography>
      <Card variant="outlined">
        <CardContent sx={{ display: "flex", alignItems: "center", gap: 2 }}>
          <Box sx={{ flex: 1 }}>
            <Typography variant="body2">
              모든 캐시 항목을 즉시 삭제합니다. 이후 요청은 캐시 없이 처리됩니다.
            </Typography>
          </Box>
          <Button
            variant="outlined"
            color="error"
            startIcon={<CachedIcon />}
            onClick={() => setClearConfirmCategory("__all__")}
            disabled={clearMutation.isPending}
          >
            전체 캐시 초기화
          </Button>
        </CardContent>
      </Card>

      {/* Confirm Dialog */}
      <Dialog
        open={clearConfirmCategory !== null}
        onClose={() => setClearConfirmCategory(null)}
        maxWidth="xs"
        fullWidth
      >
        <DialogTitle>캐시 초기화 확인</DialogTitle>
        <DialogContent>
          <Typography>
            {clearConfirmCategory === "__all__"
              ? "전체 캐시를 초기화하시겠습니까?"
              : `'${config.categories.find((c) => c.key === clearConfirmCategory)?.label ?? clearConfirmCategory}' 캐시를 초기화하시겠습니까?`}
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setClearConfirmCategory(null)}>취소</Button>
          <Button
            variant="contained"
            color="error"
            disabled={clearMutation.isPending}
            onClick={() => {
              if (clearConfirmCategory === "__all__") {
                clearMutation.mutate(undefined);
              } else if (clearConfirmCategory) {
                clearMutation.mutate(clearConfirmCategory);
              }
            }}
          >
            {clearMutation.isPending ? <CircularProgress size={18} /> : "초기화"}
          </Button>
        </DialogActions>
      </Dialog>

      <Snackbar
        open={snack.open}
        autoHideDuration={3000}
        onClose={() => setSnack((s) => ({ ...s, open: false }))}
        anchorOrigin={{ vertical: "bottom", horizontal: "center" }}
      >
        <Alert severity={snack.severity} onClose={() => setSnack((s) => ({ ...s, open: false }))}>
          {snack.message}
        </Alert>
      </Snackbar>
    </>
  );
}

// ── Tab: LLM Playground ──
function LLMPlaygroundTab() {
  const [systemPrompt, setSystemPrompt] = useState(
    "당신은 도움이 되는 AI 어시스턴트입니다. 질문에 명확하고 간결하게 답변하세요."
  );
  const [userPrompt, setUserPrompt] = useState("");
  const [temperature, setTemperature] = useState(0.2);
  const [maxTokens, setMaxTokens] = useState(1024);
  const [selectedModel, setSelectedModel] = useState("");
  const [result, setResult] = useState<{
    response: string;
    latency_ms: number;
    model_used: string;
    tokens_used?: number;
  } | null>(null);
  const [copied, setCopied] = useState(false);

  const { data: modelsData } = useQuery({
    queryKey: ["admin-models"],
    queryFn: () => adminApi.listModels(),
  });

  const models: Array<{ model_id: string; name: string; provider: string }> =
    modelsData?.data?.models ?? [];

  const playgroundMutation = useMutation({
    mutationFn: () =>
      adminApi.playground({
        prompt: userPrompt,
        model: selectedModel || undefined,
        temperature,
        max_tokens: maxTokens,
        system_prompt: systemPrompt || undefined,
      }),
    onSuccess: (res) => {
      setResult(res.data);
    },
  });

  const handleCopy = () => {
    if (result?.response) {
      navigator.clipboard.writeText(result.response);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  return (
    <Box sx={{ maxWidth: 900 }}>
      <Typography variant="h6" gutterBottom>
        LLM 테스트
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
        RAG 파이프라인 없이 LLM에 직접 프롬프트를 전송합니다. 프롬프트 엔지니어링 및 모델 동작 테스트에 사용하세요.
      </Typography>

      <Grid container spacing={3}>
        {/* Left: Inputs */}
        <Grid size={{ xs: 12, md: 5 }}>
          <Card variant="outlined">
            <CardContent sx={{ display: "flex", flexDirection: "column", gap: 2 }}>
              <Typography variant="subtitle2" color="text.secondary">
                파라미터
              </Typography>

              {/* Model selector */}
              <FormControl size="small" fullWidth>
                <InputLabel>모델 (기본값 사용 시 비워두기)</InputLabel>
                <Select
                  value={selectedModel}
                  onChange={(e) => setSelectedModel(e.target.value)}
                  label="모델 (기본값 사용 시 비워두기)"
                >
                  <MenuItem value="">
                    <em>기본 모델</em>
                  </MenuItem>
                  {models.map((m) => (
                    <MenuItem key={m.model_id} value={m.model_id}>
                      [{m.provider}] {m.name}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>

              {/* Temperature */}
              <Box>
                <Typography variant="caption" color="text.secondary">
                  Temperature: {temperature.toFixed(1)}
                </Typography>
                <Slider
                  value={temperature}
                  min={0}
                  max={1}
                  step={0.1}
                  onChange={(_, v) => setTemperature(v as number)}
                  marks={[
                    { value: 0, label: "0" },
                    { value: 0.5, label: "0.5" },
                    { value: 1, label: "1" },
                  ]}
                  size="small"
                />
              </Box>

              {/* Max tokens */}
              <TextField
                label="Max Tokens"
                type="number"
                size="small"
                value={maxTokens}
                onChange={(e) => {
                  const v = parseInt(e.target.value);
                  if (!isNaN(v)) setMaxTokens(Math.min(4096, Math.max(64, v)));
                }}
                inputProps={{ min: 64, max: 4096 }}
                helperText="64 ~ 4096"
              />

              {/* System prompt */}
              <TextField
                label="시스템 프롬프트 (선택)"
                multiline
                rows={4}
                value={systemPrompt}
                onChange={(e) => setSystemPrompt(e.target.value)}
                placeholder="시스템 프롬프트를 입력하세요..."
                size="small"
              />
            </CardContent>
          </Card>
        </Grid>

        {/* Right: Prompt & Result */}
        <Grid size={{ xs: 12, md: 7 }}>
          <Box sx={{ display: "flex", flexDirection: "column", gap: 2, height: "100%" }}>
            {/* User prompt */}
            <TextField
              label="프롬프트"
              multiline
              rows={6}
              value={userPrompt}
              onChange={(e) => setUserPrompt(e.target.value)}
              placeholder="LLM에 전송할 프롬프트를 입력하세요..."
              fullWidth
              required
            />

            <Button
              variant="contained"
              startIcon={
                playgroundMutation.isPending ? <CircularProgress size={18} color="inherit" /> : <PlayArrowIcon />
              }
              disabled={playgroundMutation.isPending || !userPrompt.trim()}
              onClick={() => playgroundMutation.mutate()}
              sx={{ alignSelf: "flex-start" }}
            >
              {playgroundMutation.isPending ? "실행 중..." : "실행"}
            </Button>

            {/* Error */}
            {playgroundMutation.isError && (
              <Alert severity="error">
                {(playgroundMutation.error as { response?: { data?: { detail?: string } } })?.response?.data?.detail ?? "LLM 요청 실패"}
              </Alert>
            )}

            {/* Result */}
            {result && (
              <Card variant="outlined">
                <CardContent>
                  {/* Meta info */}
                  <Box sx={{ display: "flex", gap: 1, flexWrap: "wrap", mb: 1.5 }}>
                    <Chip
                      icon={<SpeedIcon />}
                      label={`${result.latency_ms.toLocaleString()} ms`}
                      size="small"
                      color="primary"
                      variant="outlined"
                    />
                    <Chip
                      label={result.model_used}
                      size="small"
                      variant="outlined"
                    />
                    {result.tokens_used != null && (
                      <Chip
                        label={`${result.tokens_used} tokens`}
                        size="small"
                        variant="outlined"
                      />
                    )}
                    <Box sx={{ flex: 1 }} />
                    <Tooltip title={copied ? "복사됨!" : "응답 복사"}>
                      <IconButton size="small" onClick={handleCopy}>
                        <ContentCopyIcon fontSize="small" />
                      </IconButton>
                    </Tooltip>
                  </Box>

                  <Divider sx={{ mb: 1.5 }} />

                  {/* Response text */}
                  <Box
                    component="pre"
                    sx={{
                      whiteSpace: "pre-wrap",
                      wordBreak: "break-word",
                      fontFamily: "inherit",
                      fontSize: "0.875rem",
                      lineHeight: 1.7,
                      m: 0,
                      maxHeight: 400,
                      overflowY: "auto",
                    }}
                  >
                    {result.response}
                  </Box>
                </CardContent>
              </Card>
            )}
          </Box>
        </Grid>
      </Grid>
    </Box>
  );
}

// ── Tab: System Settings ──
function SystemSettingsTab() {
  const [llmSettings, setLlmSettings] = useState({
    provider: "ollama" as "ollama" | "vllm" | "openai" | "anthropic",
    model: "qwen2.5:7b",
    temperature: 0.2,
    maxTokens: 512,
  });

  const [ragSettings, setRagSettings] = useState({
    chunkSize: 800,
    chunkOverlap: 100,
    topK: 10,
    rerankTopN: 5,
    multiQueryEnabled: false,
    selfRagEnabled: true,
  });

  const [systemSettings, setSystemSettings] = useState({
    authEnabled: false,
    cacheEnabled: false,
    debugMode: false,
  });

  const [snackbar, setSnackbar] = useState<{ open: boolean; message: string; severity: "success" | "error" }>({
    open: false,
    message: "",
    severity: "success",
  });

  const defaultLlm = { provider: "ollama" as const, model: "qwen2.5:7b", temperature: 0.2, maxTokens: 512 };
  const defaultRag = { chunkSize: 800, chunkOverlap: 100, topK: 10, rerankTopN: 5, multiQueryEnabled: false, selfRagEnabled: true };
  const defaultSystem = { authEnabled: false, cacheEnabled: false, debugMode: false };

  const handleSave = () => {
    // Mock save — no backend call
    setSnackbar({ open: true, message: "설정이 저장되었습니다.", severity: "success" });
  };

  const handleReset = () => {
    setLlmSettings(defaultLlm);
    setRagSettings(defaultRag);
    setSystemSettings(defaultSystem);
    setSnackbar({ open: true, message: "기본값으로 초기화되었습니다.", severity: "success" });
  };

  return (
    <Box>
      <Typography variant="h5" fontWeight={700} gutterBottom>
        시스템 설정
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
        LLM, RAG 파이프라인, 시스템 전반 설정을 관리합니다.
      </Typography>

      <Grid container spacing={3}>
        {/* LLM Settings */}
        <Grid size={{ xs: 12, md: 6 }}>
          <Card variant="outlined">
            <CardContent>
              <Typography variant="h6" fontWeight={600} gutterBottom sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                <MemoryIcon fontSize="small" color="primary" />
                LLM 설정
              </Typography>
              <Divider sx={{ mb: 2 }} />

              <Stack spacing={2.5}>
                <FormControl fullWidth size="small">
                  <InputLabel>기본 프로바이더</InputLabel>
                  <Select
                    value={llmSettings.provider}
                    label="기본 프로바이더"
                    onChange={(e) =>
                      setLlmSettings((prev) => ({
                        ...prev,
                        provider: e.target.value as typeof llmSettings.provider,
                      }))
                    }
                  >
                    <MenuItem value="ollama">Ollama (로컬)</MenuItem>
                    <MenuItem value="vllm">vLLM (운영)</MenuItem>
                    <MenuItem value="openai">OpenAI</MenuItem>
                    <MenuItem value="anthropic">Anthropic</MenuItem>
                  </Select>
                </FormControl>

                <TextField
                  label="기본 모델명"
                  size="small"
                  fullWidth
                  value={llmSettings.model}
                  onChange={(e) =>
                    setLlmSettings((prev) => ({ ...prev, model: e.target.value }))
                  }
                  placeholder="예: qwen2.5:7b"
                />

                <Box>
                  <Typography variant="body2" color="text.secondary" gutterBottom>
                    Temperature: <strong>{llmSettings.temperature.toFixed(1)}</strong>
                  </Typography>
                  <Slider
                    value={llmSettings.temperature}
                    min={0.0}
                    max={1.0}
                    step={0.1}
                    marks
                    valueLabelDisplay="auto"
                    onChange={(_, v) =>
                      setLlmSettings((prev) => ({ ...prev, temperature: v as number }))
                    }
                    sx={{ color: "primary.main" }}
                  />
                  <Box sx={{ display: "flex", justifyContent: "space-between" }}>
                    <Typography variant="caption" color="text.secondary">0.0 (결정적)</Typography>
                    <Typography variant="caption" color="text.secondary">1.0 (창의적)</Typography>
                  </Box>
                </Box>

                <TextField
                  label="최대 토큰 수"
                  size="small"
                  fullWidth
                  type="number"
                  value={llmSettings.maxTokens}
                  onChange={(e) =>
                    setLlmSettings((prev) => ({
                      ...prev,
                      maxTokens: Math.max(256, Math.min(4096, Number(e.target.value))),
                    }))
                  }
                  inputProps={{ min: 256, max: 4096, step: 64 }}
                  helperText="범위: 256 ~ 4096"
                />
              </Stack>
            </CardContent>
          </Card>
        </Grid>

        {/* RAG Settings */}
        <Grid size={{ xs: 12, md: 6 }}>
          <Card variant="outlined">
            <CardContent>
              <Typography variant="h6" fontWeight={600} gutterBottom sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                <StorageIcon fontSize="small" color="primary" />
                RAG 설정
              </Typography>
              <Divider sx={{ mb: 2 }} />

              <Stack spacing={2.5}>
                <Grid container spacing={2}>
                  <Grid size={6}>
                    <TextField
                      label="청크 크기"
                      size="small"
                      fullWidth
                      type="number"
                      value={ragSettings.chunkSize}
                      onChange={(e) =>
                        setRagSettings((prev) => ({
                          ...prev,
                          chunkSize: Math.max(100, Number(e.target.value)),
                        }))
                      }
                      inputProps={{ min: 100, max: 4000, step: 50 }}
                    />
                  </Grid>
                  <Grid size={6}>
                    <TextField
                      label="청크 오버랩"
                      size="small"
                      fullWidth
                      type="number"
                      value={ragSettings.chunkOverlap}
                      onChange={(e) =>
                        setRagSettings((prev) => ({
                          ...prev,
                          chunkOverlap: Math.max(0, Number(e.target.value)),
                        }))
                      }
                      inputProps={{ min: 0, max: 500, step: 10 }}
                    />
                  </Grid>
                  <Grid size={6}>
                    <TextField
                      label="Top-K 검색 수"
                      size="small"
                      fullWidth
                      type="number"
                      value={ragSettings.topK}
                      onChange={(e) =>
                        setRagSettings((prev) => ({
                          ...prev,
                          topK: Math.max(1, Math.min(50, Number(e.target.value))),
                        }))
                      }
                      inputProps={{ min: 1, max: 50, step: 1 }}
                    />
                  </Grid>
                  <Grid size={6}>
                    <TextField
                      label="Rerank Top-N"
                      size="small"
                      fullWidth
                      type="number"
                      value={ragSettings.rerankTopN}
                      onChange={(e) =>
                        setRagSettings((prev) => ({
                          ...prev,
                          rerankTopN: Math.max(1, Math.min(20, Number(e.target.value))),
                        }))
                      }
                      inputProps={{ min: 1, max: 20, step: 1 }}
                    />
                  </Grid>
                </Grid>

                <Divider />

                <FormControlLabel
                  control={
                    <Switch
                      checked={ragSettings.multiQueryEnabled}
                      onChange={(e) =>
                        setRagSettings((prev) => ({ ...prev, multiQueryEnabled: e.target.checked }))
                      }
                      color="primary"
                    />
                  }
                  label={
                    <Box>
                      <Typography variant="body2" fontWeight={500}>Multi-Query</Typography>
                      <Typography variant="caption" color="text.secondary">
                        다중 관점 쿼리 생성으로 검색 품질 향상
                      </Typography>
                    </Box>
                  }
                />

                <FormControlLabel
                  control={
                    <Switch
                      checked={ragSettings.selfRagEnabled}
                      onChange={(e) =>
                        setRagSettings((prev) => ({ ...prev, selfRagEnabled: e.target.checked }))
                      }
                      color="primary"
                    />
                  }
                  label={
                    <Box>
                      <Typography variant="body2" fontWeight={500}>Self-RAG</Typography>
                      <Typography variant="caption" color="text.secondary">
                        응답 생성 후 근거성 자동 평가
                      </Typography>
                    </Box>
                  }
                />
              </Stack>
            </CardContent>
          </Card>
        </Grid>

        {/* System Settings */}
        <Grid size={{ xs: 12, md: 6 }}>
          <Card variant="outlined">
            <CardContent>
              <Typography variant="h6" fontWeight={600} gutterBottom sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                <SettingsIcon fontSize="small" color="primary" />
                시스템 설정
              </Typography>
              <Divider sx={{ mb: 2 }} />

              <Stack spacing={1.5}>
                <FormControlLabel
                  control={
                    <Switch
                      checked={systemSettings.authEnabled}
                      onChange={(e) =>
                        setSystemSettings((prev) => ({ ...prev, authEnabled: e.target.checked }))
                      }
                      color="primary"
                    />
                  }
                  label={
                    <Box>
                      <Typography variant="body2" fontWeight={500}>인증 활성화</Typography>
                      <Typography variant="caption" color="text.secondary">
                        JWT 기반 인증 및 RBAC 권한 제어
                      </Typography>
                    </Box>
                  }
                />
                <Divider />
                <FormControlLabel
                  control={
                    <Switch
                      checked={systemSettings.cacheEnabled}
                      onChange={(e) =>
                        setSystemSettings((prev) => ({ ...prev, cacheEnabled: e.target.checked }))
                      }
                      color="primary"
                    />
                  }
                  label={
                    <Box>
                      <Typography variant="body2" fontWeight={500}>캐시 활성화</Typography>
                      <Typography variant="caption" color="text.secondary">
                        Redis/InMemory 응답 캐시로 성능 향상
                      </Typography>
                    </Box>
                  }
                />
                <Divider />
                <FormControlLabel
                  control={
                    <Switch
                      checked={systemSettings.debugMode}
                      onChange={(e) =>
                        setSystemSettings((prev) => ({ ...prev, debugMode: e.target.checked }))
                      }
                      color="warning"
                    />
                  }
                  label={
                    <Box>
                      <Typography variant="body2" fontWeight={500}>디버그 모드</Typography>
                      <Typography variant="caption" color="text.secondary">
                        상세 로그 출력 (운영 환경에서 비활성화 권장)
                      </Typography>
                    </Box>
                  }
                />
              </Stack>
            </CardContent>
          </Card>
        </Grid>

        {/* Current Values Summary */}
        <Grid size={{ xs: 12, md: 6 }}>
          <Card variant="outlined" sx={{ bgcolor: "action.hover" }}>
            <CardContent>
              <Typography variant="h6" fontWeight={600} gutterBottom sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                <InfoIcon fontSize="small" color="primary" />
                현재 설정 요약
              </Typography>
              <Divider sx={{ mb: 2 }} />
              <Stack spacing={1}>
                {[
                  ["LLM 프로바이더", llmSettings.provider],
                  ["모델명", llmSettings.model],
                  ["Temperature", llmSettings.temperature.toFixed(1)],
                  ["Max Tokens", String(llmSettings.maxTokens)],
                  ["청크 크기", String(ragSettings.chunkSize)],
                  ["오버랩", String(ragSettings.chunkOverlap)],
                  ["Top-K", String(ragSettings.topK)],
                  ["Rerank Top-N", String(ragSettings.rerankTopN)],
                  ["Multi-Query", ragSettings.multiQueryEnabled ? "ON" : "OFF"],
                  ["Self-RAG", ragSettings.selfRagEnabled ? "ON" : "OFF"],
                  ["인증", systemSettings.authEnabled ? "활성" : "비활성"],
                  ["캐시", systemSettings.cacheEnabled ? "활성" : "비활성"],
                  ["디버그", systemSettings.debugMode ? "ON" : "OFF"],
                ].map(([key, val]) => (
                  <Box key={key} sx={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                    <Typography variant="caption" color="text.secondary">{key}</Typography>
                    <Chip label={val} size="small" variant="outlined" sx={{ fontSize: "0.7rem", height: 20 }} />
                  </Box>
                ))}
              </Stack>
            </CardContent>
          </Card>
        </Grid>

        {/* Action Buttons */}
        <Grid size={12}>
          <Box sx={{ display: "flex", gap: 2, justifyContent: "flex-end", pt: 1 }}>
            <Button
              variant="outlined"
              color="inherit"
              startIcon={<RefreshIcon />}
              onClick={handleReset}
            >
              기본값으로 초기화
            </Button>
            <Button
              variant="contained"
              color="primary"
              startIcon={<CheckIcon />}
              onClick={handleSave}
              sx={{ minWidth: 120 }}
            >
              설정 저장
            </Button>
          </Box>
        </Grid>
      </Grid>

      <Snackbar
        open={snackbar.open}
        autoHideDuration={3000}
        onClose={() => setSnackbar((prev) => ({ ...prev, open: false }))}
        anchorOrigin={{ vertical: "bottom", horizontal: "center" }}
      >
        <Alert
          severity={snackbar.severity}
          onClose={() => setSnackbar((prev) => ({ ...prev, open: false }))}
          sx={{ width: "100%" }}
        >
          {snackbar.message}
        </Alert>
      </Snackbar>
    </Box>
  );
}

// ── Main: AdminPage ──
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

const MOCK_NOTIFICATIONS: Notification[] = [
  {
    id: 1,
    type: "feedback",
    title: "새로운 부정확 피드백이 접수되었습니다",
    timestamp: new Date(Date.now() - 2 * 60 * 1000),
    read: false,
  },
  {
    id: 2,
    type: "document",
    title: "문서 '내부규정_제5장.hwp' 인제스트 완료",
    timestamp: new Date(Date.now() - 8 * 60 * 1000),
    read: false,
  },
  {
    id: 3,
    type: "system",
    title: "시스템 CPU 사용률 85% 초과",
    timestamp: new Date(Date.now() - 15 * 60 * 1000),
    read: false,
  },
  {
    id: 4,
    type: "user",
    title: "새 사용자 'kim_manager' 가입 승인 대기",
    timestamp: new Date(Date.now() - 30 * 60 * 1000),
    read: false,
  },
  {
    id: 5,
    type: "document",
    title: "문서 'ALIO_공시_2025Q4.pdf' 인제스트 완료",
    timestamp: new Date(Date.now() - 55 * 60 * 1000),
    read: true,
  },
  {
    id: 6,
    type: "feedback",
    title: "부분정확 피드백 3건이 누적되었습니다",
    timestamp: new Date(Date.now() - 2 * 3600 * 1000),
    read: true,
  },
  {
    id: 7,
    type: "system",
    title: "메모리 사용률 78% — 모니터링 권장",
    timestamp: new Date(Date.now() - 4 * 3600 * 1000),
    read: true,
  },
  {
    id: 8,
    type: "user",
    title: "사용자 'park_expert' 역할 변경 요청",
    timestamp: new Date(Date.now() - 6 * 3600 * 1000),
    read: true,
  },
  {
    id: 9,
    type: "document",
    title: "문서 동기화 작업 완료 — 12개 갱신됨",
    timestamp: new Date(Date.now() - 24 * 3600 * 1000),
    read: true,
  },
  {
    id: 10,
    type: "system",
    title: "가드레일 규칙 위반 시도 감지됨",
    timestamp: new Date(Date.now() - 2 * 24 * 3600 * 1000),
    read: true,
  },
];

function NotificationBell() {
  const [notifications, setNotifications] = useState<Notification[]>(MOCK_NOTIFICATIONS);
  const [anchorEl, setAnchorEl] = useState<HTMLButtonElement | null>(null);

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
        {/* Header */}
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

        {/* Notification list */}
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
                      alignItems="flex-start"
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

// ── Tab: Export Center ──
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

function generateFeedbackCSV(): string {
  const headers = ["messageId", "question", "answer", "rating", "correctionText", "date"];
  const rows: string[][] = [];
  const ratings = [1, 0, -1];
  const questions = [
    "안전관리 기준은 무엇인가요?",
    "가스 누출 시 대처 방법은?",
    "정기 점검 주기는 어떻게 되나요?",
    "설비 운전 기준을 알려주세요.",
    "긴급 차단 절차를 설명해 주세요.",
  ];
  for (let i = 0; i < 50; i++) {
    const d = new Date(2026, 0, 1 + (i % 60));
    rows.push([
      `msg-${1000 + i}`,
      questions[i % questions.length],
      `해당 규정에 따르면 ${i + 1}번 항목을 참고하시기 바랍니다.`,
      String(ratings[i % 3]),
      i % 5 === 0 ? "답변 내용이 불명확합니다." : "",
      d.toISOString().split("T")[0],
    ]);
  }
  const BOM = "\uFEFF";
  const escape = (v: string) => `"${v.replace(/"/g, '""')}"`;
  return BOM + [headers.map(escape).join(","), ...rows.map((r) => r.map(escape).join(","))].join("\n");
}

function generateStatsCSV(): string {
  const headers = ["date", "queryCount", "avgResponseTime", "cacheHitRate"];
  const rows: string[][] = [];
  for (let i = 0; i < 30; i++) {
    const d = new Date(2026, 1, 3 - i);
    rows.push([
      d.toISOString().split("T")[0],
      String(Math.floor(50 + Math.random() * 200)),
      String((1.2 + Math.random() * 3).toFixed(2)),
      String((0.4 + Math.random() * 0.5).toFixed(2)),
    ]);
  }
  const BOM = "\uFEFF";
  const escape = (v: string) => `"${v.replace(/"/g, '""')}"`;
  return BOM + [headers.map(escape).join(","), ...rows.map((r) => r.map(escape).join(","))].join("\n");
}

function generateUsersCSV(): string {
  const headers = ["username", "email", "role", "department", "lastLogin"];
  const roles = ["admin", "manager", "user", "viewer"];
  const departments = ["IT팀", "안전관리팀", "운영팀", "경영지원팀", "기술팀"];
  const rows: string[][] = [];
  for (let i = 0; i < 20; i++) {
    const d = new Date(2026, 1, 1 + (i % 28));
    rows.push([
      `user${String(i + 1).padStart(3, "0")}`,
      `user${i + 1}@example.com`,
      roles[i % 4],
      departments[i % 5],
      d.toISOString().split("T")[0],
    ]);
  }
  const BOM = "\uFEFF";
  const escape = (v: string) => `"${v.replace(/"/g, '""')}"`;
  return BOM + [headers.map(escape).join(","), ...rows.map((r) => r.map(escape).join(","))].join("\n");
}

function generateConversationsCSV(): string {
  const headers = ["sessionId", "userId", "messageIndex", "role", "content", "timestamp"];
  const rows: string[][] = [];
  for (let s = 0; s < 10; s++) {
    const sessionId = `sess-${2000 + s}`;
    const userId = `user${String(s + 1).padStart(3, "0")}`;
    const msgCount = 4 + (s % 4) * 2;
    for (let m = 0; m < msgCount; m++) {
      const d = new Date(2026, 1, 1 + s, 9 + m);
      rows.push([
        sessionId,
        userId,
        String(m),
        m % 2 === 0 ? "user" : "assistant",
        m % 2 === 0
          ? "안전 점검 절차에 대해 알려주세요."
          : "안전 점검 절차는 다음과 같습니다. 첫째, 설비 상태를 확인하고...",
        d.toISOString(),
      ]);
    }
  }
  const BOM = "\uFEFF";
  const escape = (v: string) => `"${v.replace(/"/g, '""')}"`;
  return BOM + [headers.map(escape).join(","), ...rows.map((r) => r.map(escape).join(","))].join("\n");
}

function generateJSON(exportId: string): string {
  if (exportId === "feedback") {
    const ratings = [1, 0, -1];
    const data = Array.from({ length: 50 }, (_, i) => ({
      messageId: `msg-${1000 + i}`,
      question: `안전관리 관련 질문 ${i + 1}`,
      answer: `답변 내용 ${i + 1}`,
      rating: ratings[i % 3],
      correctionText: i % 5 === 0 ? "답변 내용이 불명확합니다." : null,
      date: new Date(2026, 0, 1 + (i % 60)).toISOString().split("T")[0],
    }));
    return JSON.stringify(data, null, 2);
  }
  if (exportId === "stats") {
    const data = Array.from({ length: 30 }, (_, i) => ({
      date: new Date(2026, 1, 3 - i).toISOString().split("T")[0],
      queryCount: Math.floor(50 + Math.random() * 200),
      avgResponseTime: parseFloat((1.2 + Math.random() * 3).toFixed(2)),
      cacheHitRate: parseFloat((0.4 + Math.random() * 0.5).toFixed(2)),
    }));
    return JSON.stringify(data, null, 2);
  }
  if (exportId === "users") {
    const roles = ["admin", "manager", "user", "viewer"];
    const departments = ["IT팀", "안전관리팀", "운영팀", "경영지원팀", "기술팀"];
    const data = Array.from({ length: 20 }, (_, i) => ({
      username: `user${String(i + 1).padStart(3, "0")}`,
      email: `user${i + 1}@example.com`,
      role: roles[i % 4],
      department: departments[i % 5],
      lastLogin: new Date(2026, 1, 1 + (i % 28)).toISOString().split("T")[0],
    }));
    return JSON.stringify(data, null, 2);
  }
  // conversations
  const sessions = Array.from({ length: 10 }, (_, s) => ({
    sessionId: `sess-${2000 + s}`,
    userId: `user${String(s + 1).padStart(3, "0")}`,
    messages: Array.from({ length: 4 + (s % 4) * 2 }, (__, m) => ({
      index: m,
      role: m % 2 === 0 ? "user" : "assistant",
      content: m % 2 === 0 ? "안전 점검 절차에 대해 알려주세요." : "안전 점검 절차는 다음과 같습니다.",
      timestamp: new Date(2026, 1, 1 + s, 9 + m).toISOString(),
    })),
  }));
  return JSON.stringify(sessions, null, 2);
}

// ── Tab: Audit Log ──
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


function generateMockAuditData(): AuditEntry[] {
  const now = new Date("2026-03-05T10:00:00");
  const entries: AuditEntry[] = [
    {
      id: "a001",
      timestamp: new Date(now.getTime() - 5 * 60 * 1000).toISOString(),
      user: "admin",
      eventType: "login",
      action: "로그인 성공",
      details: "관리자 계정으로 로그인. 브라우저: Chrome 121",
      severity: "INFO",
      ip: "192.168.1.10",
    },
    {
      id: "a002",
      timestamp: new Date(now.getTime() - 12 * 60 * 1000).toISOString(),
      user: "user1",
      eventType: "doc_upload",
      action: "문서 업로드",
      details: "파일: 안전관리규정_2026.pdf (2.3 MB), 컬렉션: 내부규정",
      severity: "INFO",
      ip: "192.168.1.22",
      resource: "안전관리규정_2026.pdf",
    },
    {
      id: "a003",
      timestamp: new Date(now.getTime() - 25 * 60 * 1000).toISOString(),
      user: "system",
      eventType: "system",
      action: "자동 동기화 실행",
      details: "문서 동기화 스케줄러 트리거. 변경 감지: 3개 파일",
      severity: "INFO",
    },
    {
      id: "a004",
      timestamp: new Date(now.getTime() - 45 * 60 * 1000).toISOString(),
      user: "unknown",
      eventType: "security",
      action: "로그인 실패 (5회 연속)",
      details: "계정: admin, IP: 203.0.113.55. 브루트포스 시도 의심. 계정 잠금 적용.",
      severity: "CRITICAL",
      ip: "203.0.113.55",
    },
    {
      id: "a005",
      timestamp: new Date(now.getTime() - 70 * 60 * 1000).toISOString(),
      user: "manager",
      eventType: "settings",
      action: "가드레일 규칙 추가",
      details: "새 키워드 필터 규칙 추가: '기밀정보', '내부자료'. 규칙 ID: gr-2026-031",
      severity: "INFO",
      ip: "192.168.1.5",
    },
    {
      id: "a006",
      timestamp: new Date(now.getTime() - 90 * 60 * 1000).toISOString(),
      user: "user2",
      eventType: "feedback",
      action: "피드백 제출",
      details: "응답 평가: 부정확 (-1). 세션 ID: sess-2044. 수정 의견 포함.",
      severity: "INFO",
      ip: "192.168.1.33",
    },
    {
      id: "a007",
      timestamp: new Date(now.getTime() - 2 * 60 * 60 * 1000).toISOString(),
      user: "admin",
      eventType: "doc_delete",
      action: "문서 삭제",
      details: "파일: 구_안전규정_2024.pdf 삭제. 벡터스토어에서 234 청크 제거.",
      severity: "WARNING",
      ip: "192.168.1.10",
      resource: "구_안전규정_2024.pdf",
    },
    {
      id: "a008",
      timestamp: new Date(now.getTime() - 3 * 60 * 60 * 1000).toISOString(),
      user: "manager",
      eventType: "login",
      action: "로그인 성공",
      details: "관리자(매니저) 계정 로그인. IP: 192.168.1.5",
      severity: "INFO",
      ip: "192.168.1.5",
    },
    {
      id: "a009",
      timestamp: new Date(now.getTime() - 3.5 * 60 * 60 * 1000).toISOString(),
      user: "system",
      eventType: "system",
      action: "임베딩 작업 완료",
      details: "문서 5건 임베딩 완료. 신규 청크: 1,247개. 소요시간: 42초.",
      severity: "INFO",
    },
    {
      id: "a010",
      timestamp: new Date(now.getTime() - 4 * 60 * 60 * 1000).toISOString(),
      user: "admin",
      eventType: "settings",
      action: "LLM 프로바이더 변경",
      details: "기본 LLM: ollama/qwen2.5:7b → ollama/qwen2.5:14b. 변경자: admin",
      severity: "WARNING",
      ip: "192.168.1.10",
    },
    {
      id: "a011",
      timestamp: new Date(now.getTime() - 5 * 60 * 60 * 1000).toISOString(),
      user: "user1",
      eventType: "logout",
      action: "로그아웃",
      details: "세션 정상 종료. 세션 지속 시간: 2시간 15분",
      severity: "INFO",
      ip: "192.168.1.22",
    },
    {
      id: "a012",
      timestamp: new Date(now.getTime() - 6 * 60 * 60 * 1000).toISOString(),
      user: "system",
      eventType: "security",
      action: "비정상 접근 패턴 감지",
      details: "IP 203.0.113.88에서 단시간 내 API 호출 횟수 초과 (500회/분). 자동 차단 적용.",
      severity: "CRITICAL",
      ip: "203.0.113.88",
    },
    {
      id: "a013",
      timestamp: new Date(now.getTime() - 7 * 60 * 60 * 1000).toISOString(),
      user: "user2",
      eventType: "doc_upload",
      action: "문서 업로드",
      details: "파일: ALIO_공시자료_Q4.xlsx (1.1 MB), 컬렉션: ALIO공시",
      severity: "INFO",
      ip: "192.168.1.33",
      resource: "ALIO_공시자료_Q4.xlsx",
    },
    {
      id: "a014",
      timestamp: new Date(now.getTime() - 8 * 60 * 60 * 1000).toISOString(),
      user: "admin",
      eventType: "settings",
      action: "사용자 권한 변경",
      details: "user2 역할 변경: viewer → user. 승인자: admin",
      severity: "WARNING",
      ip: "192.168.1.10",
    },
    {
      id: "a015",
      timestamp: new Date(now.getTime() - 10 * 60 * 60 * 1000).toISOString(),
      user: "system",
      eventType: "system",
      action: "데이터베이스 백업 완료",
      details: "SQLite 백업 완료: memory.db, audit.db, users.db. 총 크기: 18.4 MB",
      severity: "INFO",
    },
    {
      id: "a016",
      timestamp: new Date(now.getTime() - 1 * 24 * 60 * 60 * 1000).toISOString(),
      user: "manager",
      eventType: "feedback",
      action: "피드백 검토",
      details: "최근 50건 피드백 검토 완료. 부정 피드백 3건 메모 추가.",
      severity: "INFO",
      ip: "192.168.1.5",
    },
    {
      id: "a017",
      timestamp: new Date(now.getTime() - 1.2 * 24 * 60 * 60 * 1000).toISOString(),
      user: "admin",
      eventType: "security",
      action: "IP 블랙리스트 추가",
      details: "IP 203.0.113.55 영구 차단 등록. 사유: 반복적 브루트포스 시도.",
      severity: "WARNING",
      ip: "192.168.1.10",
    },
    {
      id: "a018",
      timestamp: new Date(now.getTime() - 1.5 * 24 * 60 * 60 * 1000).toISOString(),
      user: "user1",
      eventType: "login",
      action: "로그인 성공",
      details: "일반 사용자 로그인. 기기: MacBook Pro",
      severity: "INFO",
      ip: "192.168.1.22",
    },
    {
      id: "a019",
      timestamp: new Date(now.getTime() - 2 * 24 * 60 * 60 * 1000).toISOString(),
      user: "system",
      eventType: "system",
      action: "Milvus 인덱스 최적화",
      details: "벡터 인덱스 재구성 완료. 39,739 청크. 소요시간: 8분 22초.",
      severity: "INFO",
    },
    {
      id: "a020",
      timestamp: new Date(now.getTime() - 2.2 * 24 * 60 * 60 * 1000).toISOString(),
      user: "admin",
      eventType: "doc_upload",
      action: "대량 문서 업로드",
      details: "일괄 업로드: 국외출장_보고서 20건. 총 크기: 45.2 MB. 인제스트 파이프라인 실행 중.",
      severity: "INFO",
      ip: "192.168.1.10",
      resource: "국외출장_보고서 (20건)",
    },
    {
      id: "a021",
      timestamp: new Date(now.getTime() - 2.5 * 24 * 60 * 60 * 1000).toISOString(),
      user: "manager",
      eventType: "settings",
      action: "프롬프트 버전 롤백",
      details: "시스템 프롬프트 v1.3 → v1.2 롤백. 사유: 응답 품질 저하 탐지.",
      severity: "WARNING",
      ip: "192.168.1.5",
    },
    {
      id: "a022",
      timestamp: new Date(now.getTime() - 3 * 24 * 60 * 60 * 1000).toISOString(),
      user: "system",
      eventType: "security",
      action: "가드레일 트리거",
      details: "입력 가드레일 활성화. 사용자: user2, 쿼리에 민감 키워드 포함. 요청 차단.",
      severity: "WARNING",
    },
    {
      id: "a023",
      timestamp: new Date(now.getTime() - 3.3 * 24 * 60 * 60 * 1000).toISOString(),
      user: "user2",
      eventType: "login",
      action: "로그인 실패",
      details: "잘못된 비밀번호 입력 (2회). IP: 192.168.1.33",
      severity: "WARNING",
      ip: "192.168.1.33",
    },
    {
      id: "a024",
      timestamp: new Date(now.getTime() - 4 * 24 * 60 * 60 * 1000).toISOString(),
      user: "admin",
      eventType: "settings",
      action: "캐시 초기화",
      details: "LLM 응답 캐시 전체 삭제. 삭제된 항목: 1,240개. 메모리 해제: 128 MB.",
      severity: "INFO",
      ip: "192.168.1.10",
    },
    {
      id: "a025",
      timestamp: new Date(now.getTime() - 4.5 * 24 * 60 * 60 * 1000).toISOString(),
      user: "manager",
      eventType: "doc_delete",
      action: "문서 삭제",
      details: "파일: 홍보물_2023_outdated.pdf 삭제. 관련 벡터 512개 제거.",
      severity: "INFO",
      ip: "192.168.1.5",
      resource: "홍보물_2023_outdated.pdf",
    },
    {
      id: "a026",
      timestamp: new Date(now.getTime() - 5 * 24 * 60 * 60 * 1000).toISOString(),
      user: "system",
      eventType: "system",
      action: "시스템 재시작",
      details: "RunPod 인스턴스 재시작. 서비스 복구 시간: 45초. 모든 서비스 정상 확인.",
      severity: "WARNING",
    },
    {
      id: "a027",
      timestamp: new Date(now.getTime() - 5.5 * 24 * 60 * 60 * 1000).toISOString(),
      user: "user1",
      eventType: "feedback",
      action: "피드백 제출",
      details: "응답 평가: 정확 (+1). 세션 ID: sess-2031. '상세한 답변 감사합니다' 코멘트.",
      severity: "INFO",
      ip: "192.168.1.22",
    },
    {
      id: "a028",
      timestamp: new Date(now.getTime() - 6 * 24 * 60 * 60 * 1000).toISOString(),
      user: "admin",
      eventType: "security",
      action: "감사 로그 내보내기",
      details: "감사 로그 CSV 내보내기 실행. 기간: 2026-02-01 ~ 2026-03-05. 항목: 2,341건.",
      severity: "INFO",
      ip: "192.168.1.10",
    },
    {
      id: "a029",
      timestamp: new Date(now.getTime() - 6.5 * 24 * 60 * 60 * 1000).toISOString(),
      user: "manager",
      eventType: "login",
      action: "로그인 성공",
      details: "매니저 계정 로그인. 위치: 서울 사무소",
      severity: "INFO",
      ip: "192.168.1.5",
    },
    {
      id: "a030",
      timestamp: new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000).toISOString(),
      user: "system",
      eventType: "system",
      action: "정기 점검 완료",
      details: "주간 시스템 점검 완료. CPU: 정상, 메모리: 정상, 디스크: 정상, 네트워크: 정상.",
      severity: "INFO",
    },
  ];
  return entries;
}

function AuditLogTab() {
  const ALL_ENTRIES = generateMockAuditData();
  const PAGE_SIZE = 10;

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

  const filtered = ALL_ENTRIES.filter((e) => {
    if (
      searchText &&
      ![e.user, e.action, e.details, e.ip ?? "", e.resource ?? ""]
        .join(" ")
        .toLowerCase()
        .includes(searchText.toLowerCase())
    )
      return false;
    if (userFilter && !e.user.toLowerCase().includes(userFilter.toLowerCase()))
      return false;
    if (eventTypeFilter.length > 0 && !eventTypeFilter.includes(e.eventType))
      return false;
    if (severityFilter.length > 0 && !severityFilter.includes(e.severity))
      return false;
    if (dateFrom && new Date(e.timestamp) < new Date(dateFrom)) return false;
    if (dateTo && new Date(e.timestamp) > new Date(dateTo + "T23:59:59")) return false;
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
        <Box sx={{ flex: 1 }} />
        <Button
          startIcon={<DownloadIcon />}
          variant="outlined"
          size="small"
          onClick={exportCsv}
        >
          CSV 내보내기
        </Button>
        <Tooltip title="필터 초기화">
          <IconButton size="small" onClick={resetFilters}>
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
        {pageItems.length === 0 ? (
          <Box sx={{ textAlign: "center", py: 6 }}>
            <HistoryIcon sx={{ fontSize: 48, color: "text.disabled", mb: 1 }} />
            <Typography color="text.secondary">조건에 맞는 감사 로그가 없습니다.</Typography>
          </Box>
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

function ExportCenterTab() {
  const [dialogOpen, setDialogOpen] = useState(false);
  const [selectedExport, setSelectedExport] = useState<ExportType | null>(null);
  const [dateFrom, setDateFrom] = useState("2026-01-01");
  const [dateTo, setDateTo] = useState("2026-03-05");
  const [format, setFormat] = useState<"csv" | "json">("csv");
  const [snackOpen, setSnackOpen] = useState(false);
  const [exportHistory, setExportHistory] = useState<ExportHistoryItem[]>([
    { id: "h1", name: "피드백 데이터", format: "CSV", rows: 50, size: "12 KB", date: "2026-03-04 14:22" },
    { id: "h2", name: "사용 통계", format: "JSON", rows: 30, size: "8 KB", date: "2026-03-03 09:15" },
    { id: "h3", name: "사용자 목록", format: "CSV", rows: 20, size: "5 KB", date: "2026-03-01 16:40" },
  ]);

  const exportTypes: ExportType[] = [
    {
      id: "feedback",
      title: "피드백 데이터",
      description: "모든 피드백 (질문, 답변, 평점, 수정의견, 날짜 포함)",
      estimatedRows: 50,
      lastExport: "2026-03-04",
      icon: <ThumbUpIcon sx={{ fontSize: 36 }} />,
      color: "#1976d2",
    },
    {
      id: "stats",
      title: "사용 통계",
      description: "일별 쿼리 수, 평균 응답시간, 캐시 히트율",
      estimatedRows: 30,
      lastExport: "2026-03-03",
      icon: <SpeedIcon sx={{ fontSize: 36 }} />,
      color: "#388e3c",
    },
    {
      id: "users",
      title: "사용자 목록",
      description: "사용자 정보, 역할, 부서, 최근 로그인 날짜",
      estimatedRows: 20,
      lastExport: "2026-03-01",
      icon: <GroupIcon sx={{ fontSize: 36 }} />,
      color: "#7b1fa2",
    },
    {
      id: "conversations",
      title: "대화 로그",
      description: "세션별 대화 메시지 전체 (role, content, timestamp 포함)",
      estimatedRows: 87,
      lastExport: "2026-02-28",
      icon: <QuestionAnswerIcon sx={{ fontSize: 36 }} />,
      color: "#e65100",
    },
  ];

  function handleCardClick(et: ExportType) {
    setSelectedExport(et);
    setDialogOpen(true);
  }

  function handleExport() {
    if (!selectedExport) return;

    let content: string;
    let mimeType: string;
    let ext: string;
    const rows =
      selectedExport.id === "feedback"
        ? 50
        : selectedExport.id === "stats"
        ? 30
        : selectedExport.id === "users"
        ? 20
        : 87;

    if (format === "csv") {
      if (selectedExport.id === "feedback") content = generateFeedbackCSV();
      else if (selectedExport.id === "stats") content = generateStatsCSV();
      else if (selectedExport.id === "users") content = generateUsersCSV();
      else content = generateConversationsCSV();
      mimeType = "text/csv;charset=utf-8";
      ext = "csv";
    } else {
      content = generateJSON(selectedExport.id);
      mimeType = "application/json";
      ext = "json";
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

function OcrTestTab() {
  const [file, setFile] = useState<File | null>(null);
  const [provider, setProvider] = useState<string>("cloud");
  const [enhanced, setEnhanced] = useState(false);
  const [loading, setLoading] = useState(false);
  const [healthLoading, setHealthLoading] = useState(false);
  const [result, setResult] = useState<{
    text: string;
    confidence: number;
    page_count: number;
    table_count: number;
    provider: string;
    enhanced: boolean;
    metadata: object;
  } | null>(null);
  const [health, setHealth] = useState<{ cloud: boolean; onprem: boolean } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragOver(false);
    const dropped = e.dataTransfer.files[0];
    if (dropped) {
      setFile(dropped);
      setResult(null);
      setError(null);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selected = e.target.files?.[0];
    if (selected) {
      setFile(selected);
      setResult(null);
      setError(null);
    }
  };

  const handleProcess = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await ocrApi.process(file, provider, enhanced);
      setResult(res.data as { text: string; confidence: number; page_count: number; table_count: number; provider: string; enhanced: boolean; metadata: object });
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "OCR 처리 중 오류가 발생했습니다.";
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  const handleHealthCheck = async () => {
    setHealthLoading(true);
    setError(null);
    try {
      const res = await ocrApi.health();
      setHealth(res.data as { cloud: boolean; onprem: boolean });
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "헬스 체크 실패";
      setError(msg);
    } finally {
      setHealthLoading(false);
    }
  };

  return (
    <Box>
      <Typography variant="h6" gutterBottom fontWeight={600}>
        OCR 테스트
      </Typography>
      <Typography variant="body2" color="text.secondary" mb={3}>
        문서 파일을 업로드하여 OCR 처리 결과를 확인합니다.
      </Typography>

      <Grid container spacing={3}>
        {/* Upload area */}
        <Grid size={{ xs: 12, md: 6 }}>
          <Card variant="outlined">
            <CardContent>
              <Typography variant="subtitle2" fontWeight={600} gutterBottom>
                파일 업로드
              </Typography>
              <Box
                onDrop={handleDrop}
                onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
                onDragLeave={() => setDragOver(false)}
                sx={{
                  border: "2px dashed",
                  borderColor: dragOver ? "primary.main" : "divider",
                  borderRadius: 2,
                  p: 4,
                  textAlign: "center",
                  bgcolor: dragOver ? "action.hover" : "background.default",
                  cursor: "pointer",
                  transition: "all 0.2s",
                  mb: 2,
                }}
                onClick={() => document.getElementById("ocr-file-input")?.click()}
              >
                <DocumentScannerIcon sx={{ fontSize: 48, color: "text.secondary", mb: 1 }} />
                <Typography variant="body2" color="text.secondary">
                  파일을 드래그하거나 클릭하여 선택
                </Typography>
                <Typography variant="caption" color="text.disabled">
                  PDF, PNG, JPG, JPEG, TIFF, BMP 지원
                </Typography>
              </Box>
              <input
                id="ocr-file-input"
                type="file"
                accept=".pdf,.png,.jpg,.jpeg,.tiff,.tif,.bmp"
                style={{ display: "none" }}
                onChange={handleFileChange}
              />
              {file && (
                <Alert severity="info" sx={{ mb: 2 }}>
                  선택된 파일: <strong>{file.name}</strong> ({(file.size / 1024).toFixed(1)} KB)
                </Alert>
              )}

              <Stack spacing={2}>
                <FormControl size="small" fullWidth>
                  <InputLabel>OCR 프로바이더</InputLabel>
                  <Select
                    value={provider}
                    label="OCR 프로바이더"
                    onChange={(e) => setProvider(e.target.value)}
                  >
                    <MenuItem value="cloud">Cloud (Upstage)</MenuItem>
                    <MenuItem value="onprem">On-Premise</MenuItem>
                  </Select>
                </FormControl>

                <FormControlLabel
                  control={
                    <Switch
                      checked={enhanced}
                      onChange={(e) => setEnhanced(e.target.checked)}
                    />
                  }
                  label="Enhanced 모드 (고품질 처리)"
                />

                <Button
                  variant="contained"
                  startIcon={loading ? <CircularProgress size={16} color="inherit" /> : <DocumentScannerIcon />}
                  onClick={handleProcess}
                  disabled={!file || loading}
                  fullWidth
                >
                  {loading ? "처리 중..." : "OCR 처리"}
                </Button>
              </Stack>
            </CardContent>
          </Card>
        </Grid>

        {/* Health check */}
        <Grid size={{ xs: 12, md: 6 }}>
          <Card variant="outlined">
            <CardContent>
              <Typography variant="subtitle2" fontWeight={600} gutterBottom>
                OCR 서비스 상태
              </Typography>
              <Button
                variant="outlined"
                size="small"
                startIcon={healthLoading ? <CircularProgress size={14} /> : <RefreshIcon />}
                onClick={handleHealthCheck}
                disabled={healthLoading}
                sx={{ mb: 2 }}
              >
                헬스 체크
              </Button>
              {health && (
                <Stack spacing={1}>
                  <Box display="flex" alignItems="center" gap={1}>
                    {health.cloud ? (
                      <CheckCircleIcon color="success" fontSize="small" />
                    ) : (
                      <ErrorIcon color="error" fontSize="small" />
                    )}
                    <Typography variant="body2">
                      Cloud (Upstage): {health.cloud ? "정상" : "비활성"}
                    </Typography>
                  </Box>
                  <Box display="flex" alignItems="center" gap={1}>
                    {health.onprem ? (
                      <CheckCircleIcon color="success" fontSize="small" />
                    ) : (
                      <ErrorIcon color="error" fontSize="small" />
                    )}
                    <Typography variant="body2">
                      On-Premise: {health.onprem ? "정상" : "비활성"}
                    </Typography>
                  </Box>
                </Stack>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Error */}
        {error && (
          <Grid size={{ xs: 12 }}>
            <Alert severity="error">{error}</Alert>
          </Grid>
        )}

        {/* Results */}
        {result && (
          <Grid size={{ xs: 12 }}>
            <Card variant="outlined">
              <CardContent>
                <Typography variant="subtitle2" fontWeight={600} gutterBottom>
                  OCR 결과
                </Typography>
                <Grid container spacing={2} mb={2}>
                  <Grid size={{ xs: 6, sm: 3 }}>
                    <Typography variant="caption" color="text.secondary">신뢰도</Typography>
                    <Typography variant="h6" color="primary">
                      {(result.confidence * 100).toFixed(1)}%
                    </Typography>
                  </Grid>
                  <Grid size={{ xs: 6, sm: 3 }}>
                    <Typography variant="caption" color="text.secondary">페이지 수</Typography>
                    <Typography variant="h6">{result.page_count}</Typography>
                  </Grid>
                  <Grid size={{ xs: 6, sm: 3 }}>
                    <Typography variant="caption" color="text.secondary">표 수</Typography>
                    <Typography variant="h6">{result.table_count}</Typography>
                  </Grid>
                  <Grid size={{ xs: 6, sm: 3 }}>
                    <Typography variant="caption" color="text.secondary">프로바이더</Typography>
                    <Typography variant="h6" sx={{ textTransform: "capitalize" }}>
                      {result.provider}
                    </Typography>
                  </Grid>
                </Grid>
                <Typography variant="caption" color="text.secondary" display="block" mb={1}>
                  추출된 텍스트
                </Typography>
                <Paper
                  variant="outlined"
                  sx={{
                    p: 2,
                    maxHeight: 400,
                    overflow: "auto",
                    fontFamily: "monospace",
                    fontSize: "0.8rem",
                    whiteSpace: "pre-wrap",
                    bgcolor: "background.default",
                  }}
                >
                  {result.text || "(텍스트 없음)"}
                </Paper>
              </CardContent>
            </Card>
          </Grid>
        )}
      </Grid>
    </Box>
  );
}

export default function AdminPage() {
  const [tabIndex, setTabIndex] = useState(0);

  return (
    <Layout title="관리자 설정" headerActions={<NotificationBell />}>
      <Tabs
        value={tabIndex}
        onChange={(_, v: number) => setTabIndex(v)}
        sx={{ borderBottom: 1, borderColor: "divider", mb: 1 }}
        variant="scrollable"
        scrollButtons="auto"
      >
        <Tab label="대시보드" icon={<DashboardIcon />} iconPosition="start" />
        <Tab label="사용자 관리" icon={<PersonIcon />} iconPosition="start" />
        <Tab label="시스템 정보" />
        <Tab label="LLM 프로바이더" />
        <Tab label="모델 관리" icon={<MemoryIcon />} iconPosition="start" />
        <Tab label="가드레일" icon={<SecurityIcon />} iconPosition="start" />
        <Tab label="프롬프트 관리" />
        <Tab label="MCP 도구" />
        <Tab label="커스텀 도구" />
        <Tab label="워크플로우" />
        <Tab label="콘텐츠 관리" />
        <Tab label="청킹 설정" />
        <Tab label="변경 관리" icon={<AccountTreeIcon />} iconPosition="start" />
        <Tab label="캐시 관리" icon={<CachedIcon />} iconPosition="start" />
        <Tab label="LLM 테스트" icon={<ScienceIcon />} iconPosition="start" />
        <Tab label="시스템 설정" icon={<SettingsIcon />} iconPosition="start" />
        <Tab label="데이터 내보내기" icon={<DownloadIcon />} iconPosition="start" />
        <Tab label="감사 로그" icon={<HistoryIcon />} iconPosition="start" />
        <Tab label="OCR 테스트" icon={<DocumentScannerIcon />} iconPosition="start" />
      </Tabs>

      <TabPanel value={tabIndex} index={0}>
        <DashboardTab />
      </TabPanel>
      <TabPanel value={tabIndex} index={1}>
        <UserManagementTab />
      </TabPanel>
      <TabPanel value={tabIndex} index={2}>
        <SystemInfoTab />
      </TabPanel>
      <TabPanel value={tabIndex} index={3}>
        <ProvidersTab />
      </TabPanel>
      <TabPanel value={tabIndex} index={4}>
        <ModelsTab />
      </TabPanel>
      <TabPanel value={tabIndex} index={5}>
        <GuardrailsTab />
      </TabPanel>
      <TabPanel value={tabIndex} index={6}>
        <PromptsTab />
      </TabPanel>
      <TabPanel value={tabIndex} index={7}>
        <McpToolsTab />
      </TabPanel>
      <TabPanel value={tabIndex} index={8}>
        <CustomToolBuilder />
      </TabPanel>
      <TabPanel value={tabIndex} index={9}>
        <WorkflowsTab />
      </TabPanel>
      <TabPanel value={tabIndex} index={10}>
        <ContentManager />
      </TabPanel>
      <TabPanel value={tabIndex} index={11}>
        <ChunkingConfigTab />
      </TabPanel>
      <TabPanel value={tabIndex} index={12}>
        <GovernanceTab />
      </TabPanel>
      <TabPanel value={tabIndex} index={13}>
        <CacheManagementTab />
      </TabPanel>
      <TabPanel value={tabIndex} index={14}>
        <LLMPlaygroundTab />
      </TabPanel>
      <TabPanel value={tabIndex} index={15}>
        <SystemSettingsTab />
      </TabPanel>
      <TabPanel value={tabIndex} index={16}>
        <ExportCenterTab />
      </TabPanel>
      <TabPanel value={tabIndex} index={17}>
        <AuditLogTab />
      </TabPanel>
      <TabPanel value={tabIndex} index={18}>
        <OcrTestTab />
      </TabPanel>
    </Layout>
  );
}
