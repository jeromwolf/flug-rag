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
import { PieChart, Pie, Cell, Legend, Tooltip, ResponsiveContainer } from "recharts";
import api, { feedbackApi, adminApi, mcpApi } from "../api/client";
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
              <Tooltip />
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

// ── Main: MonitorPage ──
export default function MonitorPage() {
  return (
    <Layout title="모니터링 대시보드">
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
    </Layout>
  );
}
