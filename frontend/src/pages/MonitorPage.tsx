import { useState, useMemo } from "react";
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
  InputLabel,
  TablePagination,
  LinearProgress,
  Tooltip,
  ToggleButton,
  ToggleButtonGroup,
  useTheme,
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
import LoginIcon from "@mui/icons-material/Login";
import QuestionAnswerIcon from "@mui/icons-material/QuestionAnswer";
import SettingsIcon from "@mui/icons-material/Settings";
import StorageIcon from "@mui/icons-material/Storage";
import InsertDriveFileIcon from "@mui/icons-material/InsertDriveFile";
import WarningAmberIcon from "@mui/icons-material/WarningAmber";
import {
  PieChart,
  Pie,
  Cell,
  Legend,
  Tooltip as RechartsTooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  LineChart,
  Line,
  AreaChart,
  Area,
  ReferenceLine,
} from "recharts";
import type { PieLabelRenderProps } from "recharts";
import api, { feedbackApi, adminApi, mcpApi, statsApi, logsApi, storageApi } from "../api/client";
import Layout from "../components/Layout";

const PIE_COLORS = ["#4caf50", "#f44336", "#9e9e9e"];

// ── Section: System Metrics (Real-time) ──
interface GpuInfo {
  index: number;
  name: string;
  utilization: number;
  memory_used_gb: number;
  memory_total_gb: number;
  memory_percent: number;
}

interface SystemMetrics {
  cpu?: number;
  cpu_count?: number;
  memory?: number;
  memory_used_gb?: number;
  memory_total_gb?: number;
  memory_available_gb?: number;
  disk?: number;
  disk_used_gb?: number;
  disk_total_gb?: number;
  disk_free_gb?: number;
  process_count?: number;
  uptime_seconds?: number;
  uptime_human?: string;
  gpu?: GpuInfo[];
}

function ResourceGauge({ label, value, detail, color }: { label: string; value: number; detail: string; color: string }) {
  return (
    <Box sx={{ textAlign: "center" }}>
      <Box sx={{ position: "relative", display: "inline-flex" }}>
        <CircularProgress
          variant="determinate"
          value={Math.min(value, 100)}
          size={80}
          thickness={6}
          sx={{ color }}
        />
        <Box sx={{ position: "absolute", top: 0, left: 0, bottom: 0, right: 0, display: "flex", alignItems: "center", justifyContent: "center" }}>
          <Typography variant="body2" sx={{ fontWeight: 700 }}>
            {value.toFixed(0)}%
          </Typography>
        </Box>
      </Box>
      <Typography variant="subtitle2" sx={{ mt: 1, fontWeight: 600 }}>
        {label}
      </Typography>
      <Typography variant="caption" color="text.secondary">
        {detail}
      </Typography>
    </Box>
  );
}

function SystemMetricsSection() {
  const { data, isLoading, error } = useQuery({
    queryKey: ["system-metrics"],
    queryFn: () => adminApi.getSystemMetrics(),
    refetchInterval: 30000, // 30 second auto refresh
  });

  if (isLoading) return <CircularProgress />;
  if (error) return <Alert severity="warning">시스템 메트릭을 불러올 수 없습니다. (관리자 권한 필요)</Alert>;

  const metrics: SystemMetrics = data?.data?.metrics ?? {};
  const hasAnyData = metrics.cpu !== undefined || metrics.memory !== undefined || metrics.disk !== undefined;
  if (!hasAnyData) return null;

  const coreItems = [
    {
      label: "CPU",
      value: metrics.cpu ?? 0,
      detail: `${metrics.cpu_count ?? "-"} 코어`,
      color: (metrics.cpu ?? 0) > 80 ? "#f44336" : (metrics.cpu ?? 0) > 50 ? "#ff9800" : "#4caf50",
    },
    {
      label: "메모리",
      value: metrics.memory ?? 0,
      detail: `${metrics.memory_used_gb ?? 0} / ${metrics.memory_total_gb ?? 0} GB`,
      color: (metrics.memory ?? 0) > 80 ? "#f44336" : (metrics.memory ?? 0) > 60 ? "#ff9800" : "#4caf50",
    },
    {
      label: "디스크",
      value: metrics.disk ?? 0,
      detail: `${metrics.disk_used_gb ?? 0} / ${metrics.disk_total_gb ?? 0} GB`,
      color: (metrics.disk ?? 0) > 85 ? "#f44336" : (metrics.disk ?? 0) > 70 ? "#ff9800" : "#4caf50",
    },
  ];

  const gpuList: GpuInfo[] = metrics.gpu ?? [];

  return (
    <Card variant="outlined" sx={{ mb: 3 }}>
      <CardContent>
        <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center", mb: 2 }}>
          <Typography variant="h6" sx={{ fontWeight: 600 }}>
            시스템 리소스
          </Typography>
          <Box sx={{ display: "flex", gap: 2 }}>
            {metrics.uptime_human && (
              <Typography variant="caption" color="text.secondary">
                업타임: {metrics.uptime_human}
              </Typography>
            )}
            {metrics.process_count !== undefined && (
              <Typography variant="caption" color="text.secondary">
                프로세스: {metrics.process_count}
              </Typography>
            )}
          </Box>
        </Box>

        {/* CPU / Memory / Disk gauges */}
        <Grid container spacing={3}>
          {coreItems.map((item) => (
            <Grid key={item.label} size={{ xs: 12, sm: 4 }}>
              <ResourceGauge {...item} />
            </Grid>
          ))}
        </Grid>

        {/* GPU section */}
        {gpuList.length > 0 && (
          <>
            <Typography variant="subtitle2" sx={{ mt: 3, mb: 1.5, fontWeight: 600, color: "text.secondary" }}>
              GPU
            </Typography>
            <Grid container spacing={2}>
              {gpuList.map((gpu) => (
                <Grid key={gpu.index} size={{ xs: 12, sm: 6, md: 4 }}>
                  <Card variant="outlined" sx={{ p: 1.5 }}>
                    <Typography variant="body2" sx={{ fontWeight: 600, mb: 1 }} noWrap title={gpu.name}>
                      GPU {gpu.index}: {gpu.name}
                    </Typography>
                    <Box sx={{ display: "flex", gap: 2, alignItems: "center" }}>
                      <Box sx={{ textAlign: "center" }}>
                        <Box sx={{ position: "relative", display: "inline-flex" }}>
                          <CircularProgress
                            variant="determinate"
                            value={gpu.utilization}
                            size={56}
                            thickness={5}
                            sx={{ color: gpu.utilization > 80 ? "#f44336" : gpu.utilization > 50 ? "#ff9800" : "#4caf50" }}
                          />
                          <Box sx={{ position: "absolute", top: 0, left: 0, bottom: 0, right: 0, display: "flex", alignItems: "center", justifyContent: "center" }}>
                            <Typography variant="caption" sx={{ fontWeight: 700, fontSize: "0.65rem" }}>
                              {gpu.utilization}%
                            </Typography>
                          </Box>
                        </Box>
                        <Typography variant="caption" display="block" color="text.secondary">연산</Typography>
                      </Box>
                      <Box sx={{ textAlign: "center" }}>
                        <Box sx={{ position: "relative", display: "inline-flex" }}>
                          <CircularProgress
                            variant="determinate"
                            value={gpu.memory_percent}
                            size={56}
                            thickness={5}
                            sx={{ color: gpu.memory_percent > 85 ? "#f44336" : gpu.memory_percent > 65 ? "#ff9800" : "#7b61ff" }}
                          />
                          <Box sx={{ position: "absolute", top: 0, left: 0, bottom: 0, right: 0, display: "flex", alignItems: "center", justifyContent: "center" }}>
                            <Typography variant="caption" sx={{ fontWeight: 700, fontSize: "0.65rem" }}>
                              {gpu.memory_percent}%
                            </Typography>
                          </Box>
                        </Box>
                        <Typography variant="caption" display="block" color="text.secondary">VRAM</Typography>
                      </Box>
                      <Box>
                        <Typography variant="caption" color="text.secondary" display="block">
                          {gpu.memory_used_gb} / {gpu.memory_total_gb} GB
                        </Typography>
                      </Box>
                    </Box>
                  </Card>
                </Grid>
              ))}
            </Grid>
          </>
        )}
      </CardContent>
    </Card>
  );
}

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
            {health?.app_name ?? "AI 어시스턴트"}
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
      a.download = `ai-qa-stats-${period}.xlsx`;
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

// ── Section: Log Management (Access / Query / Operations) ──

function getDefaultDateRange() {
  const end = new Date();
  const start = new Date();
  start.setDate(start.getDate() - 7);
  const fmt = (d: Date) => d.toISOString().slice(0, 10);
  return { start: fmt(start), end: fmt(end) };
}

const OP_ACTION_OPTIONS = [
  { value: "all", label: "전체" },
  { value: "DOCUMENT_UPLOAD", label: "문서 업로드" },
  { value: "DOCUMENT_DELETE", label: "문서 삭제" },
  { value: "DOCUMENT_ACCESS", label: "문서 접근" },
  { value: "SETTINGS_CHANGE", label: "설정 변경" },
  { value: "ROLE_CHANGE", label: "역할 변경" },
  { value: "USER_CREATED", label: "사용자 생성" },
  { value: "USER_DEACTIVATED", label: "사용자 비활성화" },
] as const;

interface AccessLogEntry {
  timestamp: string;
  username: string;
  action: string;
  resource: string;
  ip_address: string;
}
interface QueryLogEntry {
  timestamp: string;
  session_id: string;
  content: string;
  cached?: boolean;
}
interface OperationLogEntry {
  timestamp: string;
  username: string;
  action: string;
  resource: string;
  details: string;
}

function LogManagementSection() {
  const defaults = getDefaultDateRange();

  // Shared date filters
  const [startDate, setStartDate] = useState(defaults.start);
  const [endDate, setEndDate] = useState(defaults.end);

  // Sub-tab state
  const [logTab, setLogTab] = useState(0);

  // Access log state
  const [accessUsername, setAccessUsername] = useState("");
  const [accessIp, setAccessIp] = useState("");
  const [accessPage, setAccessPage] = useState(0);
  const [accessSearchUsername, setAccessSearchUsername] = useState("");
  const [accessSearchIp, setAccessSearchIp] = useState("");

  // Query log state
  const [queryKeyword, setQueryKeyword] = useState("");
  const [querySessionId, setQuerySessionId] = useState("");
  const [queryPage, setQueryPage] = useState(0);
  const [querySearchKeyword, setQuerySearchKeyword] = useState("");
  const [querySearchSessionId, setQuerySearchSessionId] = useState("");

  // Operation log state
  const [opActionFilter, setOpActionFilter] = useState("all");
  const [opPage, setOpPage] = useState(0);

  const PAGE_SIZE = 15;

  // ── Data fetching ──

  const { data: accessData, isLoading: accessLoading } = useQuery({
    queryKey: ["access-logs", accessSearchUsername, accessSearchIp, startDate, endDate, accessPage],
    queryFn: () =>
      logsApi.searchAccess({
        username: accessSearchUsername || undefined,
        ip_address: accessSearchIp || undefined,
        start_date: startDate || undefined,
        end_date: endDate || undefined,
        page: accessPage + 1,
        page_size: PAGE_SIZE,
      }),
    enabled: logTab === 0,
  });

  const { data: queryData, isLoading: queryLoading } = useQuery({
    queryKey: ["query-logs", querySearchKeyword, querySearchSessionId, startDate, endDate, queryPage],
    queryFn: () =>
      logsApi.searchQueries({
        keyword: querySearchKeyword || undefined,
        session_id: querySearchSessionId || undefined,
        start_date: startDate || undefined,
        end_date: endDate || undefined,
        page: queryPage + 1,
        page_size: PAGE_SIZE,
      }),
    enabled: logTab === 1,
  });

  const { data: opData, isLoading: opLoading } = useQuery({
    queryKey: ["operation-logs", opActionFilter, opPage],
    queryFn: () =>
      logsApi.getOperations({
        action_filter: opActionFilter || undefined,
        page: opPage + 1,
        page_size: PAGE_SIZE,
      }),
    enabled: logTab === 2,
  });

  const accessLogs = accessData?.data;
  const queryLogs = queryData?.data;
  const opLogs = opData?.data;

  // ── Search handlers ──

  const handleAccessSearch = () => {
    setAccessSearchUsername(accessUsername);
    setAccessSearchIp(accessIp);
    setAccessPage(0);
  };

  const handleQuerySearch = () => {
    setQuerySearchKeyword(queryKeyword);
    setQuerySearchSessionId(querySessionId);
    setQueryPage(0);
  };

  return (
    <>
      {/* Summary Cards */}
      <Box sx={{ display: "flex", gap: 2, mb: 3, flexWrap: "wrap" }}>
        <Card variant="outlined" sx={{ flex: "1 1 180px", minWidth: 180 }}>
          <CardContent sx={{ display: "flex", alignItems: "center", gap: 1.5, py: 1.5, "&:last-child": { pb: 1.5 } }}>
            <LoginIcon sx={{ fontSize: 32, color: "#1976d2" }} />
            <Box>
              <Typography variant="caption" color="text.secondary">접속 로그</Typography>
              <Typography variant="h6" sx={{ fontWeight: 700 }}>{accessLogs?.total ?? "-"}</Typography>
            </Box>
          </CardContent>
        </Card>
        <Card variant="outlined" sx={{ flex: "1 1 180px", minWidth: 180 }}>
          <CardContent sx={{ display: "flex", alignItems: "center", gap: 1.5, py: 1.5, "&:last-child": { pb: 1.5 } }}>
            <QuestionAnswerIcon sx={{ fontSize: 32, color: "#2e7d32" }} />
            <Box>
              <Typography variant="caption" color="text.secondary">총 질의</Typography>
              <Typography variant="h6" sx={{ fontWeight: 700 }}>{queryLogs?.total ?? "-"}</Typography>
            </Box>
          </CardContent>
        </Card>
        <Card variant="outlined" sx={{ flex: "1 1 180px", minWidth: 180 }}>
          <CardContent sx={{ display: "flex", alignItems: "center", gap: 1.5, py: 1.5, "&:last-child": { pb: 1.5 } }}>
            <SettingsIcon sx={{ fontSize: 32, color: "#ed6c02" }} />
            <Box>
              <Typography variant="caption" color="text.secondary">작업 로그</Typography>
              <Typography variant="h6" sx={{ fontWeight: 700 }}>{opLogs?.total ?? "-"}</Typography>
            </Box>
          </CardContent>
        </Card>
      </Box>

      {/* Date Range Filters (shared) */}
      <Card variant="outlined" sx={{ mb: 2 }}>
        <CardContent sx={{ display: "flex", alignItems: "center", gap: 2, py: 1.5, "&:last-child": { pb: 1.5 }, flexWrap: "wrap" }}>
          <Typography variant="body2" sx={{ fontWeight: 600, minWidth: 60 }}>기간</Typography>
          <TextField
            type="date"
            size="small"
            label="시작일"
            value={startDate}
            onChange={(e) => setStartDate(e.target.value)}
            slotProps={{ inputLabel: { shrink: true } }}
            sx={{ width: 170 }}
          />
          <Typography variant="body2" color="text.secondary">~</Typography>
          <TextField
            type="date"
            size="small"
            label="종료일"
            value={endDate}
            onChange={(e) => setEndDate(e.target.value)}
            slotProps={{ inputLabel: { shrink: true } }}
            sx={{ width: 170 }}
          />
        </CardContent>
      </Card>

      {/* Sub-tabs */}
      <Box sx={{ borderBottom: 1, borderColor: "divider", mb: 2 }}>
        <Tabs value={logTab} onChange={(_, v) => setLogTab(v)} variant="fullWidth">
          <Tab label="접속 로그" />
          <Tab label="질의 이력" />
          <Tab label="작업 로그" />
        </Tabs>
      </Box>

      {/* ── Access Logs Tab ── */}
      {logTab === 0 && (
        <Card variant="outlined">
          <CardContent>
            <Box sx={{ display: "flex", gap: 1, mb: 2, flexWrap: "wrap" }}>
              <TextField
                size="small"
                placeholder="사용자명..."
                value={accessUsername}
                onChange={(e) => setAccessUsername(e.target.value)}
                onKeyDown={(e) => { if (e.key === "Enter") handleAccessSearch(); }}
                sx={{ flex: "1 1 160px", minWidth: 140 }}
              />
              <TextField
                size="small"
                placeholder="IP 주소..."
                value={accessIp}
                onChange={(e) => setAccessIp(e.target.value)}
                onKeyDown={(e) => { if (e.key === "Enter") handleAccessSearch(); }}
                sx={{ flex: "1 1 160px", minWidth: 140 }}
              />
              <Button variant="outlined" startIcon={<SearchIcon />} onClick={handleAccessSearch}>
                검색
              </Button>
            </Box>
            {accessLoading ? <CircularProgress /> : (
              <>
                <TableContainer component={Paper} variant="outlined">
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell>시각</TableCell>
                        <TableCell>사용자</TableCell>
                        <TableCell>액션</TableCell>
                        <TableCell>리소스</TableCell>
                        <TableCell>IP</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {(accessLogs?.logs ?? []).map((log: AccessLogEntry, i: number) => (
                        <TableRow key={i} hover>
                          <TableCell sx={{ whiteSpace: "nowrap" }}>
                            <Typography variant="caption">{new Date(log.timestamp).toLocaleString("ko-KR")}</Typography>
                          </TableCell>
                          <TableCell>
                            <Typography variant="body2">{log.username || "-"}</Typography>
                          </TableCell>
                          <TableCell>
                            <Chip label={log.action} size="small" variant="outlined" />
                          </TableCell>
                          <TableCell>
                            <Typography variant="body2" noWrap sx={{ maxWidth: 200, overflow: "hidden", textOverflow: "ellipsis" }}>
                              {log.resource || "-"}
                            </Typography>
                          </TableCell>
                          <TableCell>
                            <Typography variant="caption" sx={{ fontFamily: "monospace" }}>{log.ip_address || "-"}</Typography>
                          </TableCell>
                        </TableRow>
                      ))}
                      {(accessLogs?.logs ?? []).length === 0 && (
                        <TableRow>
                          <TableCell colSpan={5}>
                            <Typography color="text.secondary" variant="body2" align="center">결과 없음</Typography>
                          </TableCell>
                        </TableRow>
                      )}
                    </TableBody>
                  </Table>
                </TableContainer>
                {(accessLogs?.total ?? 0) > PAGE_SIZE && (
                  <TablePagination
                    component="div"
                    count={accessLogs?.total ?? 0}
                    page={accessPage}
                    onPageChange={(_, p) => setAccessPage(p)}
                    rowsPerPage={PAGE_SIZE}
                    rowsPerPageOptions={[PAGE_SIZE]}
                    labelDisplayedRows={({ from, to, count }) => `${from}-${to} / ${count}`}
                  />
                )}
              </>
            )}
          </CardContent>
        </Card>
      )}

      {/* ── Query Logs Tab ── */}
      {logTab === 1 && (
        <Card variant="outlined">
          <CardContent>
            <Box sx={{ display: "flex", gap: 1, mb: 2, flexWrap: "wrap" }}>
              <TextField
                size="small"
                placeholder="키워드 검색..."
                value={queryKeyword}
                onChange={(e) => setQueryKeyword(e.target.value)}
                onKeyDown={(e) => { if (e.key === "Enter") handleQuerySearch(); }}
                sx={{ flex: "1 1 200px", minWidth: 160 }}
              />
              <TextField
                size="small"
                placeholder="세션 ID..."
                value={querySessionId}
                onChange={(e) => setQuerySessionId(e.target.value)}
                onKeyDown={(e) => { if (e.key === "Enter") handleQuerySearch(); }}
                sx={{ flex: "1 1 160px", minWidth: 140 }}
              />
              <Button variant="outlined" startIcon={<SearchIcon />} onClick={handleQuerySearch}>
                검색
              </Button>
            </Box>
            {queryLoading ? <CircularProgress /> : (
              <>
                <TableContainer component={Paper} variant="outlined">
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell>시각</TableCell>
                        <TableCell>세션 ID</TableCell>
                        <TableCell>질의 내용</TableCell>
                        <TableCell>캐시</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {(queryLogs?.queries ?? []).map((q: QueryLogEntry, i: number) => (
                        <TableRow key={i} hover>
                          <TableCell sx={{ whiteSpace: "nowrap" }}>
                            <Typography variant="caption">{new Date(q.timestamp).toLocaleString("ko-KR")}</Typography>
                          </TableCell>
                          <TableCell>
                            <Typography variant="caption" sx={{ fontFamily: "monospace" }}>{q.session_id?.slice(0, 8)}...</Typography>
                          </TableCell>
                          <TableCell>
                            <Typography variant="body2" noWrap sx={{ maxWidth: 420, overflow: "hidden", textOverflow: "ellipsis" }}>
                              {q.content}
                            </Typography>
                          </TableCell>
                          <TableCell>
                            {q.cached && (
                              <Chip label="캐시" size="small" color="success" variant="outlined" sx={{ fontSize: "0.7rem" }} />
                            )}
                          </TableCell>
                        </TableRow>
                      ))}
                      {(queryLogs?.queries ?? []).length === 0 && (
                        <TableRow>
                          <TableCell colSpan={4}>
                            <Typography color="text.secondary" variant="body2" align="center">결과 없음</Typography>
                          </TableCell>
                        </TableRow>
                      )}
                    </TableBody>
                  </Table>
                </TableContainer>
                {(queryLogs?.total ?? 0) > PAGE_SIZE && (
                  <TablePagination
                    component="div"
                    count={queryLogs?.total ?? 0}
                    page={queryPage}
                    onPageChange={(_, p) => setQueryPage(p)}
                    rowsPerPage={PAGE_SIZE}
                    rowsPerPageOptions={[PAGE_SIZE]}
                    labelDisplayedRows={({ from, to, count }) => `${from}-${to} / ${count}`}
                  />
                )}
              </>
            )}
          </CardContent>
        </Card>
      )}

      {/* ── Operation Logs Tab ── */}
      {logTab === 2 && (
        <Card variant="outlined">
          <CardContent>
            <Box sx={{ display: "flex", gap: 1, mb: 2, flexWrap: "wrap" }}>
              <FormControl size="small" sx={{ minWidth: 200 }}>
                <InputLabel>액션 유형</InputLabel>
                <Select
                  value={opActionFilter}
                  label="액션 유형"
                  onChange={(e) => { setOpActionFilter(e.target.value); setOpPage(0); }}
                >
                  {OP_ACTION_OPTIONS.map((opt) => (
                    <MenuItem key={opt.value} value={opt.value}>{opt.label}</MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Box>
            {opLoading ? <CircularProgress /> : (
              <>
                <TableContainer component={Paper} variant="outlined">
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell>시각</TableCell>
                        <TableCell>사용자</TableCell>
                        <TableCell>액션</TableCell>
                        <TableCell>리소스</TableCell>
                        <TableCell>상세</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {(opLogs?.logs ?? []).map((log: OperationLogEntry, i: number) => (
                        <TableRow key={i} hover>
                          <TableCell sx={{ whiteSpace: "nowrap" }}>
                            <Typography variant="caption">{new Date(log.timestamp).toLocaleString("ko-KR")}</Typography>
                          </TableCell>
                          <TableCell>
                            <Typography variant="body2">{log.username || "-"}</Typography>
                          </TableCell>
                          <TableCell>
                            <Chip label={log.action} size="small" color="info" variant="outlined" />
                          </TableCell>
                          <TableCell>
                            <Typography variant="body2" noWrap sx={{ maxWidth: 200, overflow: "hidden", textOverflow: "ellipsis" }}>
                              {log.resource || "-"}
                            </Typography>
                          </TableCell>
                          <TableCell>
                            <Typography variant="body2" noWrap sx={{ maxWidth: 250, overflow: "hidden", textOverflow: "ellipsis" }}>
                              {log.details || "-"}
                            </Typography>
                          </TableCell>
                        </TableRow>
                      ))}
                      {(opLogs?.logs ?? []).length === 0 && (
                        <TableRow>
                          <TableCell colSpan={5}>
                            <Typography color="text.secondary" variant="body2" align="center">결과 없음</Typography>
                          </TableCell>
                        </TableRow>
                      )}
                    </TableBody>
                  </Table>
                </TableContainer>
                {(opLogs?.total ?? 0) > PAGE_SIZE && (
                  <TablePagination
                    component="div"
                    count={opLogs?.total ?? 0}
                    page={opPage}
                    onPageChange={(_, p) => setOpPage(p)}
                    rowsPerPage={PAGE_SIZE}
                    rowsPerPageOptions={[PAGE_SIZE]}
                    labelDisplayedRows={({ from, to, count }) => `${from}-${to} / ${count}`}
                  />
                )}
              </>
            )}
          </CardContent>
        </Card>
      )}
    </>
  );
}

// ── Section: Feedback Analytics ──
function FeedbackAnalyticsSection() {
  const { data, isLoading, error } = useQuery({
    queryKey: ["feedback-analytics"],
    queryFn: () => feedbackApi.analytics(),
  });

  if (isLoading) return <CircularProgress />;
  if (error) return <Alert severity="error">피드백 분석 데이터를 불러올 수 없습니다.</Alert>;

  const analytics = data?.data;
  const dailyTrend = analytics?.daily_trend ?? [];
  const errorBreakdown = analytics?.error_breakdown ?? [];
  const totalErrors = analytics?.total_errors ?? 0;

  const ERROR_COLORS: Record<string, string> = {
    incorrect_answer: "#f44336",
    hallucination: "#ff9800",
    offensive: "#9c27b0",
    outdated: "#2196f3",
    other: "#9e9e9e",
  };

  return (
    <>
      {/* Daily Trend Chart */}
      <Card variant="outlined" sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>
            일별 피드백 추이 (최근 30일)
          </Typography>
          {dailyTrend.length > 0 ? (
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={dailyTrend}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" tick={{ fontSize: 11 }} />
                <YAxis />
                <RechartsTooltip />
                <Legend />
                <Bar dataKey="positive" name="긍정" fill="#4caf50" stackId="a" radius={[0, 0, 0, 0]} />
                <Bar dataKey="neutral" name="보통" fill="#9e9e9e" stackId="a" />
                <Bar dataKey="negative" name="부정" fill="#f44336" stackId="a" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <Box sx={{ height: 200, display: "flex", alignItems: "center", justifyContent: "center" }}>
              <Typography color="text.secondary">피드백 데이터가 없습니다.</Typography>
            </Box>
          )}
        </CardContent>
      </Card>

      {/* Error Reports Breakdown */}
      <Card variant="outlined">
        <CardContent>
          <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>
            오류 신고 유형 분류 (총 {totalErrors}건)
          </Typography>
          {errorBreakdown.length > 0 ? (
            <Grid container spacing={2}>
              {errorBreakdown.map((e: { type: string; label: string; count: number }) => (
                <Grid key={e.type} size={{ xs: 6, sm: 4, md: 2.4 }}>
                  <Card variant="outlined" sx={{ textAlign: "center", p: 2 }}>
                    <Typography variant="h4" sx={{ fontWeight: 700, color: ERROR_COLORS[e.type] ?? "#9e9e9e" }}>
                      {e.count}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      {e.label}
                    </Typography>
                  </Card>
                </Grid>
              ))}
            </Grid>
          ) : (
            <Typography color="text.secondary">오류 신고가 없습니다.</Typography>
          )}
        </CardContent>
      </Card>
    </>
  );
}

// ── Section: Storage Management ──
function formatBytes(bytes: number): string {
  if (bytes === 0) return "0 B";
  const k = 1024;
  const sizes = ["B", "KB", "MB", "GB", "TB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(1))} ${sizes[i]}`;
}

interface StorageFileRecord {
  filename: string;
  size_bytes: number;
  extension: string;
  uploaded_at: string;
  age_days: number;
}

interface StorageUserRecord {
  user: string;
  file_count: number;
  bytes: number;
}

interface StorageTypeRecord {
  extension: string;
  count: number;
  bytes: number;
}

interface StorageQuota {
  max_file_size_mb: number;
  max_user_storage_mb: number;
  max_total_storage_gb: number;
  file_retention_days: number;
}

interface StorageStats {
  total_bytes: number;
  total_files: number;
  by_user: StorageUserRecord[];
  by_type: StorageTypeRecord[];
  largest_files: StorageFileRecord[];
  expiring_soon: StorageFileRecord[];
  quota: StorageQuota;
}

function StorageManagementSection() {
  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ["storage-stats"],
    queryFn: () => storageApi.getStats(),
    refetchInterval: 60000, // refresh every minute
  });

  if (isLoading) return <CircularProgress />;
  if (error) return <Alert severity="warning">저장소 통계를 불러올 수 없습니다. (관리자 권한 필요)</Alert>;

  const stats: StorageStats = data?.data ?? {
    total_bytes: 0,
    total_files: 0,
    by_user: [],
    by_type: [],
    largest_files: [],
    expiring_soon: [],
    quota: { max_file_size_mb: 50, max_user_storage_mb: 500, max_total_storage_gb: 10, file_retention_days: 365 },
  };

  const quota = stats.quota;
  const totalUsedGB = stats.total_bytes / (1024 ** 3);
  const totalPct = Math.min((totalUsedGB / quota.max_total_storage_gb) * 100, 100);
  const totalColor = totalPct > 85 ? "error" : totalPct > 65 ? "warning" : "primary";

  // File type distribution for pie chart
  const typeChartData = stats.by_type.slice(0, 8).map((t) => ({
    name: t.extension || "unknown",
    value: t.count,
    bytes: t.bytes,
  }));
  const TYPE_COLORS = ["#4caf50", "#2196f3", "#ff9800", "#9c27b0", "#f44336", "#00bcd4", "#795548", "#607d8b"];

  return (
    <>
      {/* Header */}
      <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center", mb: 3 }}>
        <Typography variant="h6" sx={{ fontWeight: 600 }}>
          저장소 관리
        </Typography>
        <Button variant="outlined" size="small" onClick={() => refetch()}>
          새로고침
        </Button>
      </Box>

      {/* Summary Cards */}
      <Grid container spacing={2} sx={{ mb: 3 }}>
        <Grid size={{ xs: 12, sm: 6, md: 3 }}>
          <Card variant="outlined">
            <CardContent sx={{ display: "flex", alignItems: "center", gap: 2 }}>
              <StorageIcon sx={{ fontSize: 40, color: "primary.main" }} />
              <Box>
                <Typography variant="body2" color="text.secondary">총 사용량</Typography>
                <Typography variant="h6" sx={{ fontWeight: 700 }}>{formatBytes(stats.total_bytes)}</Typography>
                <Typography variant="caption" color="text.secondary">/ {quota.max_total_storage_gb} GB 한도</Typography>
              </Box>
            </CardContent>
          </Card>
        </Grid>
        <Grid size={{ xs: 12, sm: 6, md: 3 }}>
          <Card variant="outlined">
            <CardContent sx={{ display: "flex", alignItems: "center", gap: 2 }}>
              <InsertDriveFileIcon sx={{ fontSize: 40, color: "info.main" }} />
              <Box>
                <Typography variant="body2" color="text.secondary">파일 수</Typography>
                <Typography variant="h6" sx={{ fontWeight: 700 }}>{stats.total_files.toLocaleString()}</Typography>
                <Typography variant="caption" color="text.secondary">{stats.by_type.length}가지 유형</Typography>
              </Box>
            </CardContent>
          </Card>
        </Grid>
        <Grid size={{ xs: 12, sm: 6, md: 3 }}>
          <Card variant="outlined">
            <CardContent sx={{ display: "flex", alignItems: "center", gap: 2 }}>
              <SettingsIcon sx={{ fontSize: 40, color: "secondary.main" }} />
              <Box>
                <Typography variant="body2" color="text.secondary">파일 크기 한도</Typography>
                <Typography variant="h6" sx={{ fontWeight: 700 }}>{quota.max_file_size_mb} MB</Typography>
                <Typography variant="caption" color="text.secondary">파일당 최대 크기</Typography>
              </Box>
            </CardContent>
          </Card>
        </Grid>
        <Grid size={{ xs: 12, sm: 6, md: 3 }}>
          <Card variant="outlined">
            <CardContent sx={{ display: "flex", alignItems: "center", gap: 2 }}>
              <WarningAmberIcon sx={{ fontSize: 40, color: stats.expiring_soon.length > 0 ? "warning.main" : "text.secondary" }} />
              <Box>
                <Typography variant="body2" color="text.secondary">보존 기간 임박</Typography>
                <Typography variant="h6" sx={{ fontWeight: 700, color: stats.expiring_soon.length > 0 ? "warning.main" : "inherit" }}>
                  {stats.expiring_soon.length}건
                </Typography>
                <Typography variant="caption" color="text.secondary">만료 30일 이내</Typography>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Storage Usage Bar */}
      <Card variant="outlined" sx={{ mb: 3 }}>
        <CardContent>
          <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center", mb: 1 }}>
            <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
              전체 저장소 사용률
            </Typography>
            <Typography variant="body2" color="text.secondary">
              {formatBytes(stats.total_bytes)} / {quota.max_total_storage_gb} GB ({totalPct.toFixed(1)}%)
            </Typography>
          </Box>
          <LinearProgress
            variant="determinate"
            value={totalPct}
            color={totalColor as "primary" | "warning" | "error"}
            sx={{ height: 12, borderRadius: 6, mb: 1 }}
          />
          <Grid container spacing={2} sx={{ mt: 1 }}>
            <Grid size={{ xs: 12, sm: 4 }}>
              <Typography variant="caption" color="text.secondary">파일당 최대: {quota.max_file_size_mb} MB</Typography>
            </Grid>
            <Grid size={{ xs: 12, sm: 4 }}>
              <Typography variant="caption" color="text.secondary">사용자당 최대: {quota.max_user_storage_mb} MB</Typography>
            </Grid>
            <Grid size={{ xs: 12, sm: 4 }}>
              <Typography variant="caption" color="text.secondary">보존 기간: {quota.file_retention_days}일</Typography>
            </Grid>
          </Grid>
        </CardContent>
      </Card>

      {/* File Type Distribution + User Breakdown */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        {/* File Type Pie Chart */}
        <Grid size={{ xs: 12, md: 5 }}>
          <Card variant="outlined" sx={{ height: "100%" }}>
            <CardContent>
              <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 2 }}>
                파일 유형 분포
              </Typography>
              {typeChartData.length > 0 ? (
                <ResponsiveContainer width="100%" height={280}>
                  <PieChart>
                    <Pie
                      data={typeChartData}
                      dataKey="value"
                      nameKey="name"
                      cx="50%"
                      cy="50%"
                      outerRadius={90}
                      label={(props: PieLabelRenderProps) => `${String(props.name ?? "")} ${(((props.percent) ?? 0) * 100).toFixed(0)}%`}
                      labelLine={false}
                    >
                      {typeChartData.map((_, index) => (
                        <Cell key={`cell-${index}`} fill={TYPE_COLORS[index % TYPE_COLORS.length]} />
                      ))}
                    </Pie>
                    <RechartsTooltip formatter={(value: unknown, name: unknown, props: { payload?: { bytes?: number } }) => [
                      `${String(value ?? "")}개 (${formatBytes((props.payload?.bytes) ?? 0)})`, String(name ?? "")
                    ] as [string, string]} />
                    <Legend />
                  </PieChart>
                </ResponsiveContainer>
              ) : (
                <Box sx={{ height: 200, display: "flex", alignItems: "center", justifyContent: "center" }}>
                  <Typography color="text.secondary">파일이 없습니다.</Typography>
                </Box>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Per-User Storage Breakdown */}
        <Grid size={{ xs: 12, md: 7 }}>
          <Card variant="outlined" sx={{ height: "100%" }}>
            <CardContent>
              <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 2 }}>
                사용자별 저장소 사용량
              </Typography>
              {stats.by_user.length > 0 ? (
                <TableContainer>
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell>사용자</TableCell>
                        <TableCell align="right">파일 수</TableCell>
                        <TableCell align="right">사용량</TableCell>
                        <TableCell sx={{ minWidth: 120 }}>한도 대비</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {stats.by_user.map((u) => {
                        const userPct = Math.min((u.bytes / (1024 ** 2) / quota.max_user_storage_mb) * 100, 100);
                        const uColor = userPct > 85 ? "error" : userPct > 65 ? "warning" : "primary";
                        return (
                          <TableRow key={u.user} hover>
                            <TableCell>
                              <Typography variant="body2" sx={{ fontWeight: 500 }}>{u.user}</Typography>
                            </TableCell>
                            <TableCell align="right">
                              <Typography variant="body2">{u.file_count}</Typography>
                            </TableCell>
                            <TableCell align="right">
                              <Typography variant="body2">{formatBytes(u.bytes)}</Typography>
                            </TableCell>
                            <TableCell>
                              <Tooltip title={`${userPct.toFixed(1)}% (한도: ${quota.max_user_storage_mb} MB)`}>
                                <LinearProgress
                                  variant="determinate"
                                  value={userPct}
                                  color={uColor as "primary" | "warning" | "error"}
                                  sx={{ height: 8, borderRadius: 4 }}
                                />
                              </Tooltip>
                            </TableCell>
                          </TableRow>
                        );
                      })}
                    </TableBody>
                  </Table>
                </TableContainer>
              ) : (
                <Typography color="text.secondary" variant="body2">
                  사용자별 데이터가 없습니다. (개인 지식공간 사용 시 표시됩니다.)
                </Typography>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Largest Files */}
      <Card variant="outlined" sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 2 }}>
            대용량 파일 TOP 10
          </Typography>
          {stats.largest_files.length > 0 ? (
            <TableContainer>
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>파일명</TableCell>
                    <TableCell>유형</TableCell>
                    <TableCell align="right">크기</TableCell>
                    <TableCell>업로드 일시</TableCell>
                    <TableCell align="right">경과일</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {stats.largest_files.map((f, i) => (
                    <TableRow key={i} hover>
                      <TableCell>
                        <Typography variant="body2" noWrap sx={{ maxWidth: 280 }} title={f.filename}>
                          {f.filename}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Chip label={f.extension || "-"} size="small" variant="outlined" />
                      </TableCell>
                      <TableCell align="right">
                        <Typography variant="body2" sx={{ fontWeight: 600 }}>{formatBytes(f.size_bytes)}</Typography>
                      </TableCell>
                      <TableCell>
                        <Typography variant="caption">
                          {f.uploaded_at ? new Date(f.uploaded_at).toLocaleDateString("ko-KR") : "-"}
                        </Typography>
                      </TableCell>
                      <TableCell align="right">
                        <Typography variant="body2">{f.age_days}일</Typography>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          ) : (
            <Typography color="text.secondary" variant="body2">파일이 없습니다.</Typography>
          )}
        </CardContent>
      </Card>

      {/* Files Expiring Soon */}
      {stats.expiring_soon.length > 0 && (
        <Card variant="outlined" sx={{ border: "1px solid", borderColor: "warning.main" }}>
          <CardContent>
            <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 2 }}>
              <WarningAmberIcon color="warning" />
              <Typography variant="subtitle1" sx={{ fontWeight: 600, color: "warning.main" }}>
                보존 기간 임박 파일 ({stats.expiring_soon.length}건)
              </Typography>
            </Box>
            <Alert severity="warning" sx={{ mb: 2 }}>
              아래 파일은 보존 기간({quota.file_retention_days}일) 만료 30일 이내입니다. 필요한 경우 재업로드하거나 관리자에게 문의하세요.
            </Alert>
            <TableContainer>
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>파일명</TableCell>
                    <TableCell>유형</TableCell>
                    <TableCell align="right">크기</TableCell>
                    <TableCell align="right">경과일</TableCell>
                    <TableCell align="right">잔여일</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {stats.expiring_soon.map((f, i) => {
                    const remaining = quota.file_retention_days - f.age_days;
                    return (
                      <TableRow key={i} hover>
                        <TableCell>
                          <Typography variant="body2" noWrap sx={{ maxWidth: 280 }} title={f.filename}>
                            {f.filename}
                          </Typography>
                        </TableCell>
                        <TableCell>
                          <Chip label={f.extension || "-"} size="small" variant="outlined" />
                        </TableCell>
                        <TableCell align="right">
                          <Typography variant="body2">{formatBytes(f.size_bytes)}</Typography>
                        </TableCell>
                        <TableCell align="right">
                          <Typography variant="body2">{f.age_days}일</Typography>
                        </TableCell>
                        <TableCell align="right">
                          <Chip
                            label={`${remaining}일`}
                            size="small"
                            color={remaining < 14 ? "error" : "warning"}
                          />
                        </TableCell>
                      </TableRow>
                    );
                  })}
                </TableBody>
              </Table>
            </TableContainer>
          </CardContent>
        </Card>
      )}
    </>
  );
}

// ── Section: Trend Charts ──
// NOTE: Backend time-series API endpoints needed:
//   GET /api/stats/timeseries?period=1h|6h|24h|7d  → { queries: [{time, count}], response_time: [{time, avg_ms}], cache_hit: [{time, rate}] }
//   GET /api/stats/model-usage  → { models: [{name, count}] }
// Currently uses mock data generated with useMemo.

type TrendPeriod = "1h" | "6h" | "24h" | "7d";

function generateQueryData(period: TrendPeriod) {
  const now = Date.now();
  const intervals: Record<TrendPeriod, { count: number; stepMs: number; label: (t: number) => string }> = {
    "1h":  { count: 12, stepMs: 5 * 60 * 1000,      label: (t) => new Date(t).toLocaleTimeString("ko-KR", { hour: "2-digit", minute: "2-digit" }) },
    "6h":  { count: 12, stepMs: 30 * 60 * 1000,     label: (t) => new Date(t).toLocaleTimeString("ko-KR", { hour: "2-digit", minute: "2-digit" }) },
    "24h": { count: 24, stepMs: 60 * 60 * 1000,     label: (t) => new Date(t).toLocaleTimeString("ko-KR", { hour: "2-digit", minute: "2-digit" }) },
    "7d":  { count: 14, stepMs: 12 * 60 * 60 * 1000, label: (t) => new Date(t).toLocaleDateString("ko-KR", { month: "short", day: "numeric", hour: "2-digit" }) },
  };
  const cfg = intervals[period];
  return Array.from({ length: cfg.count }, (_, i) => {
    const t = now - (cfg.count - 1 - i) * cfg.stepMs;
    const base = 15 + Math.sin(i * 0.8) * 8;
    return {
      time: cfg.label(t),
      queries: Math.max(0, Math.round(base + Math.random() * 10)),
      avg_ms: Math.max(500, Math.round(2200 + Math.sin(i * 0.6) * 1200 + Math.random() * 600)),
      cache_hit: Math.min(100, Math.max(30, Math.round(65 + Math.sin(i * 0.4) * 15 + Math.random() * 10))),
    };
  });
}

const MODEL_COLORS = ["#10a37f", "#1976d2", "#9c27b0", "#ff9800", "#f44336", "#00bcd4"];

function TrendChartsSection() {
  const theme = useTheme();
  const isDark = theme.palette.mode === "dark";
  const gridColor = isDark ? "rgba(255,255,255,0.08)" : "rgba(0,0,0,0.08)";
  const textColor = theme.palette.text.secondary;

  const [period, setPeriod] = useState<TrendPeriod>("24h");

  // Mock time-series data (replace with real API call when backend supports time-series endpoint)
  const timeSeriesData = useMemo(() => generateQueryData(period), [period]);

  // Mock model usage data (replace with statsApi.getModelUsage() when available)
  const modelUsageData = useMemo(() => [
    { name: "qwen2.5:14b", count: 312 },
    { name: "qwen2.5:7b", count: 187 },
    { name: "qwen2.5:32b", count: 94 },
    { name: "gpt-4o", count: 42 },
    { name: "claude-3-5", count: 18 },
  ], []);

  const totalModelUsage = modelUsageData.reduce((s, m) => s + m.count, 0);

  // useQuery stub — swap queryFn body with real API once backend exposes time-series data
  useQuery({
    queryKey: ["trend-timeseries", period],
    queryFn: async () => {
      // TODO: replace with: return statsApi.getTimeseries(period);
      return null;
    },
    refetchInterval: 30000,
    enabled: false, // disabled until real API is available
  });

  return (
    <Box>
      {/* Period selector */}
      <Box sx={{ display: "flex", alignItems: "center", gap: 2, mb: 3 }}>
        <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
          조회 기간
        </Typography>
        <ToggleButtonGroup
          value={period}
          exclusive
          size="small"
          onChange={(_, v: TrendPeriod | null) => { if (v) setPeriod(v); }}
        >
          <ToggleButton value="1h">1시간</ToggleButton>
          <ToggleButton value="6h">6시간</ToggleButton>
          <ToggleButton value="24h">24시간</ToggleButton>
          <ToggleButton value="7d">7일</ToggleButton>
        </ToggleButtonGroup>
        <Chip label="30초 자동 새로고침" size="small" variant="outlined" color="success" />
      </Box>

      <Grid container spacing={3}>

        {/* A: 시간별 질의 수 */}
        <Grid size={{ xs: 12, md: 6 }}>
          <Card variant="outlined">
            <CardContent>
              <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>
                시간별 질의 수
              </Typography>
              <ResponsiveContainer width="100%" height={260}>
                <AreaChart data={timeSeriesData} margin={{ top: 5, right: 10, left: -10, bottom: 0 }}>
                  <defs>
                    <linearGradient id="queryGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor={theme.palette.primary.main} stopOpacity={0.3} />
                      <stop offset="95%" stopColor={theme.palette.primary.main} stopOpacity={0.02} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke={gridColor} />
                  <XAxis dataKey="time" tick={{ fontSize: 11, fill: textColor }} tickLine={false} />
                  <YAxis tick={{ fontSize: 11, fill: textColor }} tickLine={false} axisLine={false} />
                  <RechartsTooltip
                    contentStyle={{
                      backgroundColor: theme.palette.background.paper,
                      border: `1px solid ${theme.palette.divider}`,
                      borderRadius: 8,
                      fontSize: 12,
                    }}
                    formatter={(v: number | undefined) => [`${(v ?? 0).toLocaleString()}건`, "질의 수"]}
                  />
                  <Area
                    type="monotone"
                    dataKey="queries"
                    stroke={theme.palette.primary.main}
                    strokeWidth={2}
                    fill="url(#queryGradient)"
                    dot={false}
                    activeDot={{ r: 4 }}
                  />
                </AreaChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>

        {/* B: 평균 응답 시간 */}
        <Grid size={{ xs: 12, md: 6 }}>
          <Card variant="outlined">
            <CardContent>
              <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>
                평균 응답 시간 (ms)
              </Typography>
              <ResponsiveContainer width="100%" height={260}>
                <LineChart data={timeSeriesData} margin={{ top: 5, right: 10, left: -10, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke={gridColor} />
                  <XAxis dataKey="time" tick={{ fontSize: 11, fill: textColor }} tickLine={false} />
                  <YAxis tick={{ fontSize: 11, fill: textColor }} tickLine={false} axisLine={false} />
                  <RechartsTooltip
                    contentStyle={{
                      backgroundColor: theme.palette.background.paper,
                      border: `1px solid ${theme.palette.divider}`,
                      borderRadius: 8,
                      fontSize: 12,
                    }}
                    formatter={(v: number | undefined) => [`${(v ?? 0).toLocaleString()} ms`, "평균 응답 시간"]}
                  />
                  {/* Warning threshold at 5000ms */}
                  <ReferenceLine
                    y={5000}
                    stroke="#f44336"
                    strokeDasharray="6 3"
                    strokeWidth={1.5}
                    label={{ value: "5s 임계값", position: "insideTopRight", fill: "#f44336", fontSize: 11 }}
                  />
                  <Line
                    type="monotone"
                    dataKey="avg_ms"
                    stroke="#ff9800"
                    strokeWidth={2}
                    dot={false}
                    activeDot={{ r: 4 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>

        {/* C: 모델별 사용 비율 */}
        <Grid size={{ xs: 12, md: 6 }}>
          <Card variant="outlined">
            <CardContent>
              <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>
                모델별 사용 비율
              </Typography>
              <ResponsiveContainer width="100%" height={280}>
                <PieChart>
                  <Pie
                    data={modelUsageData}
                    dataKey="count"
                    nameKey="name"
                    cx="50%"
                    cy="45%"
                    innerRadius={60}
                    outerRadius={100}
                    paddingAngle={3}
                  >
                    {modelUsageData.map((_, index) => (
                      <Cell key={`model-cell-${index}`} fill={MODEL_COLORS[index % MODEL_COLORS.length]} />
                    ))}
                  </Pie>
                  <RechartsTooltip
                    contentStyle={{
                      backgroundColor: theme.palette.background.paper,
                      border: `1px solid ${theme.palette.divider}`,
                      borderRadius: 8,
                      fontSize: 12,
                    }}
                    formatter={(v: number | undefined, name: string | undefined) => [
                      `${v ?? 0}건 (${(((v ?? 0) / totalModelUsage) * 100).toFixed(1)}%)`,
                      name ?? "",
                    ]}
                  />
                  <Legend
                    formatter={(value) => (
                      <Typography component="span" variant="caption" sx={{ color: "text.primary" }}>
                        {value}
                      </Typography>
                    )}
                  />
                  {/* Center label */}
                  <text x="50%" y="43%" textAnchor="middle" dominantBaseline="middle" style={{ fontSize: 22, fontWeight: 700, fill: theme.palette.text.primary }}>
                    {totalModelUsage}
                  </text>
                  <text x="50%" y="49%" textAnchor="middle" dominantBaseline="middle" style={{ fontSize: 11, fill: textColor }}>
                    총 질의
                  </text>
                </PieChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>

        {/* D: 캐시 히트율 트렌드 */}
        <Grid size={{ xs: 12, md: 6 }}>
          <Card variant="outlined">
            <CardContent>
              <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>
                캐시 히트율 트렌드 (%)
              </Typography>
              <ResponsiveContainer width="100%" height={280}>
                <AreaChart data={timeSeriesData} margin={{ top: 5, right: 10, left: -10, bottom: 0 }}>
                  <defs>
                    <linearGradient id="cacheGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#4caf50" stopOpacity={0.35} />
                      <stop offset="95%" stopColor="#4caf50" stopOpacity={0.02} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke={gridColor} />
                  <XAxis dataKey="time" tick={{ fontSize: 11, fill: textColor }} tickLine={false} />
                  <YAxis domain={[0, 100]} tick={{ fontSize: 11, fill: textColor }} tickLine={false} axisLine={false} tickFormatter={(v) => `${v}%`} />
                  <RechartsTooltip
                    contentStyle={{
                      backgroundColor: theme.palette.background.paper,
                      border: `1px solid ${theme.palette.divider}`,
                      borderRadius: 8,
                      fontSize: 12,
                    }}
                    formatter={(v: number | undefined) => [`${v ?? 0}%`, "캐시 히트율"]}
                  />
                  <Area
                    type="monotone"
                    dataKey="cache_hit"
                    stroke="#4caf50"
                    strokeWidth={2}
                    fill="url(#cacheGradient)"
                    dot={false}
                    activeDot={{ r: 4 }}
                  />
                </AreaChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>

      </Grid>
    </Box>
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
          <Tab label="피드백 분석" />
          <Tab label="저장소 관리" icon={<StorageIcon sx={{ fontSize: 16 }} />} iconPosition="start" />
          <Tab label="트렌드" icon={<BarChartIcon sx={{ fontSize: 16 }} />} iconPosition="start" />
        </Tabs>
      </Box>

      {tab === 0 && (
        <>
          {/* NEW: System Metrics at top */}
          <SystemMetricsSection />

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
        <LogManagementSection />
      )}

      {tab === 3 && (
        <FeedbackAnalyticsSection />
      )}

      {tab === 4 && (
        <StorageManagementSection />
      )}

      {tab === 5 && (
        <TrendChartsSection />
      )}
    </Layout>
  );
}
