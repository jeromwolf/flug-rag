import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  Box,
  Card,
  CardContent,
  Typography,
  Tabs,
  Tab,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TablePagination,
  Paper,
  Chip,
  Button,
  IconButton,
  CircularProgress,
  Alert,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  LinearProgress,
  Grid,
  TextField,
  MenuItem,
  Select,
  InputLabel,
  FormControl,
  InputAdornment,
  Tooltip as MuiTooltip,
} from "@mui/material";
import RefreshIcon from "@mui/icons-material/Refresh";
import DeleteIcon from "@mui/icons-material/Delete";
import ReplayIcon from "@mui/icons-material/Replay";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import ExpandLessIcon from "@mui/icons-material/ExpandLess";
import EditIcon from "@mui/icons-material/Edit";
import AddIcon from "@mui/icons-material/Add";
import SearchIcon from "@mui/icons-material/Search";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import CheckCircleIcon from "@mui/icons-material/CheckCircle";
import CancelIcon from "@mui/icons-material/Cancel";
import RateReviewIcon from "@mui/icons-material/RateReview";
import HelpOutlineIcon from "@mui/icons-material/HelpOutline";
import {
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";
import { qualityApi, goldenDataApi, correctionsApi } from "../api/client";
import Layout from "../components/Layout";

// ── Types ──

interface DocumentStatusSummary {
  total: number;
  by_status: Record<string, number>;
  by_file_type: Record<string, number>;
}

interface DocumentChange {
  filename: string;
  change_type: string;
  file_path: string;
}

interface ChunkMetrics {
  total_chunks: number;
  avg_length: number;
  duplicate_count: number;
  empty_count: number;
  table_count: number;
  semantic_completeness: number;
  length_histogram: Record<string, number>;
}

interface DocumentChunkStats {
  document_id: string;
  filename: string;
  chunk_count: number;
  avg_length: number;
  min_length: number;
  max_length: number;
}

interface ChunkPreviewItem {
  id: string;
  content_preview: string;
  length: number;
  page_number: number | null;
}

interface EmbeddingStatus {
  total_jobs: number;
  success_count: number;
  failure_count: number;
  success_rate: number;
}

interface EmbeddingHistoryItem {
  filename: string;
  total_chunks: number;
  success_count: number;
  failure_count: number;
  status: string;
  started_at: string;
}

interface EmbeddingFailedItem {
  filename: string;
  error: string;
  started_at: string;
}

interface VectorDistribution {
  total_vectors: number;
  dimensions: number;
  estimated_size_mb: number;
  index_type: string;
  norm_stats: {
    min: number;
    max: number;
    mean: number;
    std: number;
  };
  outlier_count: number;
  outlier_ids: string[];
}

interface ReprocessStats {
  total: number;
  pending: number;
  processing: number;
  completed: number;
  failed: number;
}

interface ReprocessQueueItem {
  id: string;
  filename: string;
  error_message: string;
  status: string;
  retry_count: number;
  max_retries: number;
  created_at: string;
}

// ── Helpers ──

const BAR_COLOR = "#1976d2";

interface TabPanelProps {
  children: React.ReactNode;
  value: number;
  index: number;
}

function TabPanel({ children, value, index }: TabPanelProps) {
  if (value !== index) return null;
  return <Box sx={{ py: 3 }}>{children}</Box>;
}

function StatusChip({ status }: { status: string }) {
  const colorMap: Record<string, "success" | "error" | "warning" | "info"> = {
    completed: "success",
    success: "success",
    failed: "error",
    error: "error",
    pending: "warning",
    processing: "info",
    running: "info",
  };
  return <Chip label={status} color={colorMap[status] ?? "default"} size="small" />;
}

function ChangeTypeChip({ type }: { type: string }) {
  const colorMap: Record<string, "success" | "error" | "warning" | "info"> = {
    added: "success",
    modified: "warning",
    deleted: "error",
    unchanged: "info",
  };
  return <Chip label={type} color={colorMap[type] ?? "default"} size="small" />;
}

function MetricCard({
  label,
  value,
  color,
}: {
  label: string;
  value: string | number;
  color?: string;
}) {
  return (
    <Card variant="outlined">
      <CardContent sx={{ py: 2 }}>
        <Typography variant="body2" color="text.secondary">
          {label}
        </Typography>
        <Typography variant="h5" sx={{ fontWeight: 700, color: color ?? "text.primary" }}>
          {value}
        </Typography>
      </CardContent>
    </Card>
  );
}

// ── Tab 0: Document Processing ──

const HIDDEN_FILES = [".DS_Store", ".omc", "Store", ".gitkeep", "Thumbs.db", "__MACOSX"];

function DocumentProcessingTab() {
  const [changesPage, setChangesPage] = useState(0);
  const CHANGES_ROWS_PER_PAGE = 20;

  const { data: statusData, isLoading: statusLoading, error: statusError } = useQuery({
    queryKey: ["quality-doc-status"],
    queryFn: () => qualityApi.getDocumentStatus(),
    refetchInterval: 30000,
  });

  const { data: changesData, isLoading: changesLoading, error: changesError } = useQuery({
    queryKey: ["quality-doc-changes"],
    queryFn: () => qualityApi.getDocumentChanges(),
  });

  if (statusLoading || changesLoading) return <CircularProgress />;
  if (statusError) return <Alert severity="error">문서 상태를 불러올 수 없습니다.</Alert>;
  if (changesError) return <Alert severity="error">문서 변경사항을 불러올 수 없습니다.</Alert>;

  const summary: DocumentStatusSummary = statusData?.data?.summary ?? {
    total: 0,
    by_status: {},
    by_file_type: {},
  };
  const allChanges: DocumentChange[] = changesData?.data?.changes ?? [];
  const changes = allChanges.filter(
    (d) => !HIDDEN_FILES.includes(d.filename) && !d.filename.startsWith(".")
  );
  const pagedChanges = changes.slice(
    changesPage * CHANGES_ROWS_PER_PAGE,
    changesPage * CHANGES_ROWS_PER_PAGE + CHANGES_ROWS_PER_PAGE
  );

  const statusCards = [
    { label: "전체 문서", value: summary.total, color: undefined },
    { label: "처리 완료", value: summary.by_status["completed"] ?? 0, color: "success.main" },
    { label: "처리 실패", value: summary.by_status["failed"] ?? 0, color: "error.main" },
    { label: "대기중", value: summary.by_status["pending"] ?? 0, color: "warning.main" },
  ];

  const fileTypeData = Object.entries(summary.by_file_type).map(([name, value]) => ({
    name,
    count: value,
  }));

  return (
    <>
      {/* Summary cards */}
      <Grid container spacing={2} sx={{ mb: 3 }}>
        {statusCards.map((card) => (
          <Grid key={card.label} size={{ xs: 6, sm: 3 }}>
            <MetricCard label={card.label} value={card.value} color={card.color} />
          </Grid>
        ))}
      </Grid>

      {/* File type distribution */}
      <Card variant="outlined" sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>
            파일 유형 분포
          </Typography>
          {fileTypeData.length > 0 ? (
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={fileTypeData}>
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" fill={BAR_COLOR} name="문서 수" />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <Alert severity="info">데이터가 없습니다</Alert>
          )}
        </CardContent>
      </Card>

      {/* Document changes table */}
      <Card variant="outlined">
        <CardContent>
          <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>
            문서 변경 감지
          </Typography>
          {changes.length > 0 ? (
            <>
              <TableContainer component={Paper} variant="outlined">
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>파일명</TableCell>
                      <TableCell>변경 유형</TableCell>
                      <TableCell>파일 경로</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {pagedChanges.map((change, idx) => (
                      <TableRow key={`${change.filename}-${idx}`}>
                        <TableCell>{change.filename}</TableCell>
                        <TableCell>
                          <ChangeTypeChip type={change.change_type} />
                        </TableCell>
                        <TableCell>
                          <Typography variant="body2" sx={{ fontFamily: "monospace", fontSize: 12 }}>
                            {change.file_path}
                          </Typography>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
              <TablePagination
                component="div"
                count={changes.length}
                page={changesPage}
                onPageChange={(_, p) => setChangesPage(p)}
                rowsPerPage={CHANGES_ROWS_PER_PAGE}
                rowsPerPageOptions={[CHANGES_ROWS_PER_PAGE]}
                labelDisplayedRows={({ from, to, count }) => `${from}-${to} / ${count}`}
              />
            </>
          ) : (
            <Alert severity="info">변경 감지된 문서가 없습니다</Alert>
          )}
        </CardContent>
      </Card>
    </>
  );
}

// ── Tab 1: Chunk Quality ──

function ChunkQualityTab() {
  const [previewDocId, setPreviewDocId] = useState<string | null>(null);

  const { data: metricsData, isLoading: metricsLoading, error: metricsError } = useQuery({
    queryKey: ["quality-chunk-metrics"],
    queryFn: () => qualityApi.getChunkMetrics(),
  });

  const { data: byDocData, isLoading: byDocLoading, error: byDocError } = useQuery({
    queryKey: ["quality-chunks-by-doc"],
    queryFn: () => qualityApi.getChunksByDocument(),
  });

  const { data: previewData, isLoading: previewLoading } = useQuery({
    queryKey: ["quality-chunk-preview", previewDocId],
    queryFn: () => qualityApi.getChunkPreview(previewDocId!),
    enabled: !!previewDocId,
  });

  if (metricsLoading || byDocLoading) return <CircularProgress />;
  if (metricsError) return <Alert severity="error">청크 메트릭을 불러올 수 없습니다.</Alert>;
  if (byDocError) return <Alert severity="error">문서별 청크 정보를 불러올 수 없습니다.</Alert>;

  const metrics: ChunkMetrics = metricsData?.data ?? {
    total_chunks: 0,
    avg_length: 0,
    duplicate_count: 0,
    empty_count: 0,
    table_count: 0,
    semantic_completeness: 0,
    length_histogram: {},
  };
  const docChunks: DocumentChunkStats[] = byDocData?.data?.documents ?? [];
  const previewChunks: ChunkPreviewItem[] = previewData?.data?.chunks ?? [];

  const metricCards = [
    { label: "총 청크 수", value: metrics.total_chunks },
    { label: "평균 길이", value: Math.round(metrics.avg_length) },
    { label: "중복 수", value: metrics.duplicate_count, color: "warning.main" },
    { label: "빈 청크", value: metrics.empty_count, color: "error.main" },
    { label: "테이블 청크", value: metrics.table_count, color: "info.main" },
    { label: "의미완결성", value: `${(metrics.semantic_completeness * 100).toFixed(1)}%`, color: "success.main" },
  ];

  const bucketOrder = ["0-200", "200-400", "400-600", "600-800", "800+"];
  const histogramData = bucketOrder.map((bucket) => ({
    name: bucket,
    count: metrics.length_histogram[bucket] ?? 0,
  }));

  return (
    <>
      {/* Metric cards */}
      <Grid container spacing={2} sx={{ mb: 3 }}>
        {metricCards.map((card) => (
          <Grid key={card.label} size={{ xs: 6, sm: 4, md: 2 }}>
            <MetricCard label={card.label} value={card.value} color={card.color} />
          </Grid>
        ))}
      </Grid>

      {/* Length distribution chart */}
      <Card variant="outlined" sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>
            청크 길이 분포
          </Typography>
          {histogramData.some((d) => d.count > 0) ? (
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={histogramData}>
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" fill={BAR_COLOR} name="청크 수" />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <Alert severity="info">데이터가 없습니다</Alert>
          )}
        </CardContent>
      </Card>

      {/* Document chunk stats table */}
      <Card variant="outlined">
        <CardContent>
          <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>
            문서별 청크 통계
          </Typography>
          {docChunks.length > 0 ? (
            <TableContainer component={Paper} variant="outlined">
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>문서명</TableCell>
                    <TableCell align="right">청크 수</TableCell>
                    <TableCell align="right">평균 길이</TableCell>
                    <TableCell align="right">최소 길이</TableCell>
                    <TableCell align="right">최대 길이</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {docChunks.map((doc) => (
                    <TableRow
                      key={doc.document_id}
                      hover
                      sx={{ cursor: "pointer" }}
                      onClick={() => setPreviewDocId(doc.document_id)}
                    >
                      <TableCell>{doc.filename}</TableCell>
                      <TableCell align="right">{doc.chunk_count}</TableCell>
                      <TableCell align="right">{Math.round(doc.avg_length)}</TableCell>
                      <TableCell align="right">{doc.min_length}</TableCell>
                      <TableCell align="right">{doc.max_length}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          ) : (
            <Alert severity="info">데이터가 없습니다</Alert>
          )}
        </CardContent>
      </Card>

      {/* Chunk preview dialog */}
      <Dialog
        open={!!previewDocId}
        onClose={() => setPreviewDocId(null)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>청크 미리보기</DialogTitle>
        <DialogContent>
          {previewLoading ? (
            <Box sx={{ display: "flex", justifyContent: "center", py: 4 }}>
              <CircularProgress />
            </Box>
          ) : previewChunks.length > 0 ? (
            <TableContainer component={Paper} variant="outlined" sx={{ mt: 1 }}>
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>ID</TableCell>
                    <TableCell>내용 미리보기</TableCell>
                    <TableCell align="right">길이</TableCell>
                    <TableCell align="right">페이지</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {previewChunks.map((chunk) => (
                    <TableRow key={chunk.id}>
                      <TableCell>
                        <Typography variant="body2" sx={{ fontFamily: "monospace", fontSize: 11 }}>
                          {chunk.id.slice(0, 12)}...
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Typography variant="body2" noWrap sx={{ maxWidth: 400 }}>
                          {chunk.content_preview.length > 100
                            ? `${chunk.content_preview.slice(0, 100)}...`
                            : chunk.content_preview}
                        </Typography>
                      </TableCell>
                      <TableCell align="right">{chunk.length}</TableCell>
                      <TableCell align="right">{chunk.page_number ?? "-"}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          ) : (
            <Alert severity="info" sx={{ mt: 1 }}>
              청크 데이터가 없습니다
            </Alert>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setPreviewDocId(null)}>닫기</Button>
        </DialogActions>
      </Dialog>
    </>
  );
}

// ── Tab 2: Embedding Status ──

function EmbeddingStatusTab() {
  const { data: statusData, isLoading: statusLoading, error: statusError } = useQuery({
    queryKey: ["quality-embedding-status"],
    queryFn: () => qualityApi.getEmbeddingStatus(),
    refetchInterval: 30000,
  });

  const { data: historyData, isLoading: historyLoading, error: historyError } = useQuery({
    queryKey: ["quality-embedding-history"],
    queryFn: () => qualityApi.getEmbeddingHistory(50),
  });

  const { data: failedData } = useQuery({
    queryKey: ["quality-embedding-failed"],
    queryFn: () => qualityApi.getEmbeddingFailed(),
  });

  if (statusLoading || historyLoading) return <CircularProgress />;
  if (statusError) return <Alert severity="error">임베딩 상태를 불러올 수 없습니다.</Alert>;
  if (historyError) return <Alert severity="error">임베딩 이력을 불러올 수 없습니다.</Alert>;

  const status: EmbeddingStatus = statusData?.data ?? {
    total_jobs: 0,
    success_count: 0,
    failure_count: 0,
    success_rate: 0,
  };
  const history: EmbeddingHistoryItem[] = historyData?.data?.history ?? [];
  const failedJobs: EmbeddingFailedItem[] = failedData?.data?.failed ?? [];

  const statusCards = [
    { label: "전체 작업", value: status.total_jobs },
    { label: "성공", value: status.success_count, color: "success.main" },
    { label: "실패", value: status.failure_count, color: "error.main" },
    { label: "성공률", value: `${(status.success_rate * 100).toFixed(1)}%`, color: "success.main" },
  ];

  const pieData = [
    { name: "성공", value: status.success_count },
    { name: "실패", value: status.failure_count },
  ];
  const hasPieData = pieData.some((d) => d.value > 0);

  return (
    <>
      {/* Status cards */}
      <Grid container spacing={2} sx={{ mb: 3 }}>
        {statusCards.map((card) => (
          <Grid key={card.label} size={{ xs: 6, sm: 3 }}>
            <MetricCard label={card.label} value={card.value} color={card.color} />
          </Grid>
        ))}
      </Grid>

      {/* Success rate pie chart */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid size={{ xs: 12, md: 5 }}>
          <Card variant="outlined" sx={{ height: "100%" }}>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>
                성공률
              </Typography>
              {hasPieData ? (
                <ResponsiveContainer width="100%" height={250}>
                  <PieChart>
                    <Pie
                      data={pieData}
                      dataKey="value"
                      nameKey="name"
                      cx="50%"
                      cy="50%"
                      outerRadius={90}
                      label
                    >
                      <Cell fill="#4caf50" />
                      <Cell fill="#f44336" />
                    </Pie>
                    <Tooltip />
                    <Legend />
                  </PieChart>
                </ResponsiveContainer>
              ) : (
                <Box sx={{ height: 250, display: "flex", alignItems: "center", justifyContent: "center" }}>
                  <Typography color="text.secondary">데이터가 없습니다</Typography>
                </Box>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Recent history table */}
        <Grid size={{ xs: 12, md: 7 }}>
          <Card variant="outlined" sx={{ height: "100%" }}>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>
                최근 임베딩 이력
              </Typography>
              {history.length > 0 ? (
                <TableContainer component={Paper} variant="outlined" sx={{ maxHeight: 300 }}>
                  <Table size="small" stickyHeader>
                    <TableHead>
                      <TableRow>
                        <TableCell>파일명</TableCell>
                        <TableCell align="right">청크</TableCell>
                        <TableCell align="right">성공</TableCell>
                        <TableCell align="right">실패</TableCell>
                        <TableCell>상태</TableCell>
                        <TableCell>시작 시간</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {history.map((item, idx) => (
                        <TableRow key={`${item.filename}-${idx}`}>
                          <TableCell>
                            <Typography variant="body2" noWrap sx={{ maxWidth: 150 }}>
                              {item.filename}
                            </Typography>
                          </TableCell>
                          <TableCell align="right">{item.total_chunks}</TableCell>
                          <TableCell align="right">{item.success_count}</TableCell>
                          <TableCell align="right">{item.failure_count}</TableCell>
                          <TableCell>
                            <StatusChip status={item.status} />
                          </TableCell>
                          <TableCell>
                            <Typography variant="body2" color="text.secondary" sx={{ fontSize: 12 }}>
                              {new Date(item.started_at).toLocaleString("ko-KR")}
                            </Typography>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              ) : (
                <Alert severity="info">데이터가 없습니다</Alert>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Failed jobs alert */}
      {failedJobs.length > 0 && (
        <Alert severity="error" sx={{ mb: 2 }}>
          <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 1 }}>
            실패한 임베딩 작업: {failedJobs.length}건
          </Typography>
          {failedJobs.map((job, idx) => (
            <Typography key={`failed-${idx}`} variant="body2" sx={{ ml: 1 }}>
              - {job.filename}: {job.error}
            </Typography>
          ))}
        </Alert>
      )}
    </>
  );
}

// ── Tab 3: Vector Distribution ──

function VectorDistributionTab() {
  const [outlierExpanded, setOutlierExpanded] = useState(false);

  const { data: distData, isLoading: distLoading, error: distError } = useQuery({
    queryKey: ["quality-vector-distribution"],
    queryFn: () => qualityApi.getVectorDistribution(),
  });

  const { data: healthData, isLoading: healthLoading, error: healthError } = useQuery({
    queryKey: ["quality-vector-health"],
    queryFn: () => qualityApi.getVectorHealth(),
    refetchInterval: 30000,
  });

  if (distLoading || healthLoading) return <CircularProgress />;
  if (distError) return <Alert severity="error">벡터 분포를 불러올 수 없습니다.</Alert>;
  if (healthError) return <Alert severity="error">벡터 상태를 불러올 수 없습니다.</Alert>;

  const dist: VectorDistribution = distData?.data ?? {
    total_vectors: 0,
    dimensions: 0,
    estimated_size_mb: 0,
    index_type: "-",
    norm_stats: { min: 0, max: 0, mean: 0, std: 0 },
    outlier_count: 0,
    outlier_ids: [],
  };

  const health = healthData?.data ?? {};
  const healthStatus = (health as Record<string, string>).status ?? "unknown";

  const collectionCards = [
    { label: "전체 벡터", value: dist.total_vectors.toLocaleString() },
    { label: "차원수", value: dist.dimensions },
    { label: "예상 크기(MB)", value: dist.estimated_size_mb.toFixed(1) },
    { label: "인덱스 타입", value: dist.index_type },
  ];

  const normCards = [
    { label: "최소값", value: dist.norm_stats.min.toFixed(4) },
    { label: "최대값", value: dist.norm_stats.max.toFixed(4) },
    { label: "평균값", value: dist.norm_stats.mean.toFixed(4) },
    { label: "표준편차", value: dist.norm_stats.std.toFixed(4) },
  ];

  return (
    <>
      {/* Health status */}
      <Box sx={{ mb: 3 }}>
        <Alert severity={healthStatus === "healthy" ? "success" : "warning"}>
          벡터 저장소 상태: <strong>{healthStatus === "healthy" ? "정상" : healthStatus}</strong>
        </Alert>
      </Box>

      {/* Collection info cards */}
      <Typography variant="h6" sx={{ mb: 1.5, fontWeight: 600 }}>
        컬렉션 정보
      </Typography>
      <Grid container spacing={2} sx={{ mb: 3 }}>
        {collectionCards.map((card) => (
          <Grid key={card.label} size={{ xs: 6, sm: 3 }}>
            <MetricCard label={card.label} value={card.value} />
          </Grid>
        ))}
      </Grid>

      {/* Norm distribution cards */}
      <Typography variant="h6" sx={{ mb: 1.5, fontWeight: 600 }}>
        Norm 분포
      </Typography>
      <Grid container spacing={2} sx={{ mb: 3 }}>
        {normCards.map((card) => (
          <Grid key={card.label} size={{ xs: 6, sm: 3 }}>
            <MetricCard label={card.label} value={card.value} />
          </Grid>
        ))}
      </Grid>

      {/* Outliers */}
      {dist.outlier_count > 0 && (
        <Alert
          severity="warning"
          action={
            <IconButton
              size="small"
              onClick={() => setOutlierExpanded((prev) => !prev)}
            >
              {outlierExpanded ? <ExpandLessIcon /> : <ExpandMoreIcon />}
            </IconButton>
          }
        >
          <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
            이상치 벡터: {dist.outlier_count}개
          </Typography>
          {outlierExpanded && (
            <Box sx={{ mt: 1, maxHeight: 200, overflow: "auto" }}>
              {dist.outlier_ids.map((id) => (
                <Typography key={id} variant="body2" sx={{ fontFamily: "monospace", fontSize: 11 }}>
                  {id}
                </Typography>
              ))}
            </Box>
          )}
        </Alert>
      )}

      {dist.outlier_count === 0 && (
        <Alert severity="success">이상치 벡터가 없습니다</Alert>
      )}
    </>
  );
}

// ── Tab 4: Reprocess Queue ──

function ReprocessQueueTab() {
  const queryClient = useQueryClient();

  const { data: statsData, isLoading: statsLoading, error: statsError } = useQuery({
    queryKey: ["quality-reprocess-stats"],
    queryFn: () => qualityApi.getReprocessStats(),
    refetchInterval: 30000,
  });

  const { data: queueData, isLoading: queueLoading, error: queueError } = useQuery({
    queryKey: ["quality-reprocess-queue"],
    queryFn: () => qualityApi.getReprocessQueue(),
    refetchInterval: 30000,
  });

  const retryMutation = useMutation({
    mutationFn: (queueId: string) => qualityApi.retryItem(queueId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["quality-reprocess-queue"] });
      queryClient.invalidateQueries({ queryKey: ["quality-reprocess-stats"] });
    },
  });

  const retryAllMutation = useMutation({
    mutationFn: () => qualityApi.retryAllFailed(),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["quality-reprocess-queue"] });
      queryClient.invalidateQueries({ queryKey: ["quality-reprocess-stats"] });
    },
  });

  const deleteMutation = useMutation({
    mutationFn: (queueId: string) => qualityApi.deleteQueueItem(queueId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["quality-reprocess-queue"] });
      queryClient.invalidateQueries({ queryKey: ["quality-reprocess-stats"] });
    },
  });

  if (statsLoading || queueLoading) return <CircularProgress />;
  if (statsError) return <Alert severity="error">재처리 통계를 불러올 수 없습니다.</Alert>;
  if (queueError) return <Alert severity="error">재처리 큐를 불러올 수 없습니다.</Alert>;

  const stats: ReprocessStats = statsData?.data ?? {
    total: 0,
    pending: 0,
    processing: 0,
    completed: 0,
    failed: 0,
  };
  const queue: ReprocessQueueItem[] = queueData?.data?.queue ?? [];

  const statsCards = [
    { label: "전체", value: stats.total },
    { label: "대기", value: stats.pending, color: "warning.main" },
    { label: "처리중", value: stats.processing, color: "info.main" },
    { label: "완료", value: stats.completed, color: "success.main" },
    { label: "실패", value: stats.failed, color: "error.main" },
  ];

  return (
    <>
      {/* Stats cards */}
      <Grid container spacing={2} sx={{ mb: 3 }}>
        {statsCards.map((card) => (
          <Grid key={card.label} size={{ xs: 6, sm: 2.4 }}>
            <MetricCard label={card.label} value={card.value} color={card.color} />
          </Grid>
        ))}
      </Grid>

      {/* Action buttons */}
      <Box sx={{ mb: 3, display: "flex", gap: 1 }}>
        <Button
          variant="contained"
          color="warning"
          startIcon={<RefreshIcon />}
          onClick={() => retryAllMutation.mutate()}
          disabled={retryAllMutation.isPending || stats.failed === 0}
        >
          {retryAllMutation.isPending ? <CircularProgress size={20} /> : "전체 재시도"}
        </Button>
      </Box>

      {/* Progress bar for processing */}
      {stats.processing > 0 && (
        <Box sx={{ mb: 2 }}>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 0.5 }}>
            처리중: {stats.processing}건
          </Typography>
          <LinearProgress />
        </Box>
      )}

      {/* Queue table */}
      <Card variant="outlined">
        <CardContent>
          <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>
            재처리 큐
          </Typography>
          {queue.length > 0 ? (
            <TableContainer component={Paper} variant="outlined">
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>파일명</TableCell>
                    <TableCell>오류 메시지</TableCell>
                    <TableCell>상태</TableCell>
                    <TableCell align="center">재시도</TableCell>
                    <TableCell>생성일</TableCell>
                    <TableCell align="center">작업</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {queue.map((item) => (
                    <TableRow key={item.id}>
                      <TableCell>
                        <Typography variant="body2" noWrap sx={{ maxWidth: 150 }}>
                          {item.filename}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Typography variant="body2" noWrap sx={{ maxWidth: 250 }}>
                          {item.error_message.length > 80
                            ? `${item.error_message.slice(0, 80)}...`
                            : item.error_message}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <StatusChip status={item.status} />
                      </TableCell>
                      <TableCell align="center">
                        <Typography variant="body2">
                          {item.retry_count}/{item.max_retries}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Typography variant="body2" color="text.secondary" sx={{ fontSize: 12 }}>
                          {new Date(item.created_at).toLocaleString("ko-KR")}
                        </Typography>
                      </TableCell>
                      <TableCell align="center">
                        <Box sx={{ display: "flex", gap: 0.5, justifyContent: "center" }}>
                          <IconButton
                            size="small"
                            color="primary"
                            onClick={() => retryMutation.mutate(item.id)}
                            disabled={retryMutation.isPending || item.status === "processing"}
                            title="재시도"
                          >
                            <ReplayIcon fontSize="small" />
                          </IconButton>
                          <IconButton
                            size="small"
                            color="error"
                            onClick={() => deleteMutation.mutate(item.id)}
                            disabled={deleteMutation.isPending}
                            title="삭제"
                          >
                            <DeleteIcon fontSize="small" />
                          </IconButton>
                        </Box>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          ) : (
            <Alert severity="info">재처리 대기 항목이 없습니다</Alert>
          )}
        </CardContent>
      </Card>
    </>
  );
}

// ── Tab 5: Golden Dataset ──

interface GoldenEntry {
  id: string;
  question: string;
  answer: string;
  category: string;
  evaluation_tag: string;
  source_message_id: string;
  source_session_id: string;
  created_by: string;
  created_at: string;
  updated_at: string;
  is_active: boolean;
}

const CATEGORY_OPTIONS = [
  { value: "factual", label: "Factual" },
  { value: "inference", label: "Inference" },
  { value: "negative", label: "Negative" },
  { value: "multi_hop", label: "Multi-hop" },
];

const DIFFICULTY_OPTIONS = [
  { value: "easy", label: "쉬움" },
  { value: "medium", label: "보통" },
  { value: "hard", label: "어려움" },
];

function difficultyColor(d: string): "success" | "warning" | "error" | "default" {
  if (d === "easy") return "success";
  if (d === "medium") return "warning";
  if (d === "hard") return "error";
  return "default";
}

interface GoldenFormState {
  question: string;
  expected_answer: string;
  category: string;
  difficulty: string;
  source: string;
  keywords: string;
}

const EMPTY_FORM: GoldenFormState = {
  question: "",
  expected_answer: "",
  category: "factual",
  difficulty: "medium",
  source: "",
  keywords: "",
};

const BENCHMARK_RESULTS = [
  { dataset: "관련 법률", total: 50, success: 50, rate: 100 },
  { dataset: "내부규정", total: 60, success: 59, rate: 98.3 },
  { dataset: "인쇄홍보물", total: 20, success: 20, rate: 100 },
  { dataset: "ALIO 공시", total: 20, success: 20, rate: 100 },
  { dataset: "국외출장 보고서", total: 20, success: 16, rate: 80 },
];

function GoldenDatasetTab() {
  const queryClient = useQueryClient();
  const [page, setPage] = useState(0);
  const ROWS_PER_PAGE = 10;
  const [searchText, setSearchText] = useState("");
  const [dialogOpen, setDialogOpen] = useState(false);
  const [editEntry, setEditEntry] = useState<GoldenEntry | null>(null);
  const [form, setForm] = useState<GoldenFormState>(EMPTY_FORM);
  const [deleteConfirmId, setDeleteConfirmId] = useState<string | null>(null);

  const { data, isLoading, error } = useQuery({
    queryKey: ["golden-data"],
    queryFn: () => goldenDataApi.list(),
  });

  const createMutation = useMutation({
    mutationFn: (payload: Record<string, unknown>) => goldenDataApi.create(payload),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["golden-data"] });
      setDialogOpen(false);
      setForm(EMPTY_FORM);
    },
  });

  const updateMutation = useMutation({
    mutationFn: ({ id, payload }: { id: string; payload: Record<string, unknown> }) =>
      goldenDataApi.update(id, payload),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["golden-data"] });
      setDialogOpen(false);
      setEditEntry(null);
      setForm(EMPTY_FORM);
    },
  });

  const deleteMutation = useMutation({
    mutationFn: (id: string) => goldenDataApi.delete(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["golden-data"] });
      setDeleteConfirmId(null);
    },
  });

  if (isLoading) return <CircularProgress />;
  if (error) return <Alert severity="error">골든 데이터를 불러올 수 없습니다.</Alert>;

  const rawEntries: GoldenEntry[] = data?.data?.entries ?? [];

  // Client-side search
  const filtered = searchText.trim()
    ? rawEntries.filter((e) =>
        e.question.toLowerCase().includes(searchText.toLowerCase())
      )
    : rawEntries;

  const pagedEntries = filtered.slice(page * ROWS_PER_PAGE, page * ROWS_PER_PAGE + ROWS_PER_PAGE);

  // Summary counts
  const categoryCount: Record<string, number> = {};
  const difficultyCount: Record<string, number> = {};
  for (const e of rawEntries) {
    const cat = e.category || "기타";
    categoryCount[cat] = (categoryCount[cat] ?? 0) + 1;
    const diff = e.evaluation_tag || "medium";
    difficultyCount[diff] = (difficultyCount[diff] ?? 0) + 1;
  }

  function openAdd() {
    setEditEntry(null);
    setForm(EMPTY_FORM);
    setDialogOpen(true);
  }

  function openEdit(entry: GoldenEntry) {
    setEditEntry(entry);
    setForm({
      question: entry.question,
      expected_answer: entry.answer,
      category: entry.category || "factual",
      difficulty: entry.evaluation_tag || "medium",
      source: entry.source_message_id || "",
      keywords: "",
    });
    setDialogOpen(true);
  }

  function handleSave() {
    const payload = {
      question: form.question,
      answer: form.expected_answer,
      category: form.category,
      evaluation_tag: form.difficulty,
      source_message_id: form.source,
    };
    if (editEntry) {
      updateMutation.mutate({ id: editEntry.id, payload });
    } else {
      createMutation.mutate(payload);
    }
  }

  const isSaving = createMutation.isPending || updateMutation.isPending;

  const summaryCards = [
    { label: "전체", value: rawEntries.length },
    { label: "Factual", value: categoryCount["factual"] ?? 0, color: "primary.main" },
    { label: "Inference", value: categoryCount["inference"] ?? 0, color: "secondary.main" },
    { label: "Negative", value: categoryCount["negative"] ?? 0, color: "error.main" },
    { label: "Multi-hop", value: categoryCount["multi_hop"] ?? 0, color: "warning.main" },
    { label: "쉬움", value: difficultyCount["easy"] ?? 0, color: "success.main" },
    { label: "보통", value: difficultyCount["medium"] ?? 0, color: "warning.main" },
    { label: "어려움", value: difficultyCount["hard"] ?? 0, color: "error.main" },
  ];

  return (
    <>
      {/* Summary Cards */}
      <Grid container spacing={2} sx={{ mb: 3 }}>
        {summaryCards.map((card) => (
          <Grid key={card.label} size={{ xs: 6, sm: 3, md: 1.5 }}>
            <MetricCard label={card.label} value={card.value} color={card.color} />
          </Grid>
        ))}
      </Grid>

      {/* Toolbar */}
      <Box sx={{ mb: 2, display: "flex", gap: 2, alignItems: "center", flexWrap: "wrap" }}>
        <TextField
          size="small"
          placeholder="질문 검색..."
          value={searchText}
          onChange={(e) => { setSearchText(e.target.value); setPage(0); }}
          slotProps={{
            input: {
              startAdornment: (
                <InputAdornment position="start">
                  <SearchIcon fontSize="small" />
                </InputAdornment>
              ),
            },
          }}
          sx={{ minWidth: 260 }}
        />
        <Button
          variant="contained"
          startIcon={<AddIcon />}
          onClick={openAdd}
          size="small"
        >
          추가
        </Button>
      </Box>

      {/* Table */}
      <Card variant="outlined" sx={{ mb: 4 }}>
        <CardContent sx={{ p: 0 }}>
          {filtered.length > 0 ? (
            <>
              <TableContainer component={Paper} variant="outlined">
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell sx={{ minWidth: 200 }}>질문</TableCell>
                      <TableCell sx={{ minWidth: 220 }}>기대 답변</TableCell>
                      <TableCell>카테고리</TableCell>
                      <TableCell>난이도</TableCell>
                      <TableCell>출처</TableCell>
                      <TableCell>등록일</TableCell>
                      <TableCell align="center">작업</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {pagedEntries.map((entry) => (
                      <TableRow key={entry.id} hover>
                        <TableCell>
                          <Typography variant="body2" sx={{ maxWidth: 280 }}>
                            {entry.question.length > 80
                              ? `${entry.question.slice(0, 80)}...`
                              : entry.question}
                          </Typography>
                        </TableCell>
                        <TableCell>
                          <Typography variant="body2" color="text.secondary" sx={{ maxWidth: 300 }}>
                            {entry.answer.length > 100
                              ? `${entry.answer.slice(0, 100)}...`
                              : entry.answer}
                          </Typography>
                        </TableCell>
                        <TableCell>
                          {entry.category ? (
                            <Chip label={entry.category} size="small" variant="outlined" />
                          ) : (
                            <Typography variant="body2" color="text.disabled">-</Typography>
                          )}
                        </TableCell>
                        <TableCell>
                          {entry.evaluation_tag ? (
                            <Chip
                              label={entry.evaluation_tag}
                              size="small"
                              color={difficultyColor(entry.evaluation_tag)}
                            />
                          ) : (
                            <Typography variant="body2" color="text.disabled">-</Typography>
                          )}
                        </TableCell>
                        <TableCell>
                          <Typography variant="body2" noWrap sx={{ maxWidth: 120, fontSize: 12 }}>
                            {entry.source_message_id || "-"}
                          </Typography>
                        </TableCell>
                        <TableCell>
                          <Typography variant="body2" color="text.secondary" sx={{ fontSize: 12 }}>
                            {entry.created_at
                              ? new Date(entry.created_at).toLocaleDateString("ko-KR")
                              : "-"}
                          </Typography>
                        </TableCell>
                        <TableCell align="center">
                          <Box sx={{ display: "flex", gap: 0.5, justifyContent: "center" }}>
                            <IconButton
                              size="small"
                              color="primary"
                              onClick={() => openEdit(entry)}
                              title="편집"
                            >
                              <EditIcon fontSize="small" />
                            </IconButton>
                            <IconButton
                              size="small"
                              color="error"
                              onClick={() => setDeleteConfirmId(entry.id)}
                              title="삭제"
                            >
                              <DeleteIcon fontSize="small" />
                            </IconButton>
                          </Box>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
              <TablePagination
                component="div"
                count={filtered.length}
                page={page}
                onPageChange={(_, p) => setPage(p)}
                rowsPerPage={ROWS_PER_PAGE}
                rowsPerPageOptions={[ROWS_PER_PAGE]}
                labelDisplayedRows={({ from, to, count }) => `${from}-${to} / ${count}`}
              />
            </>
          ) : (
            <Box sx={{ p: 3 }}>
              <Alert severity="info">
                {searchText ? "검색 결과가 없습니다." : "등록된 골든 데이터가 없습니다."}
              </Alert>
            </Box>
          )}
        </CardContent>
      </Card>

      {/* Benchmark Results */}
      <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>
        벤치마크 결과
      </Typography>
      <Grid container spacing={2} sx={{ mb: 2 }}>
        <Grid size={{ xs: 12, sm: 6, md: 4 }}>
          <MetricCard
            label="전체 정확도"
            value="95.8% (115/120)"
            color="success.main"
          />
        </Grid>
        {BENCHMARK_RESULTS.map((r) => (
          <Grid key={r.dataset} size={{ xs: 12, sm: 6, md: 4 }}>
            <Card variant="outlined">
              <CardContent sx={{ py: 1.5 }}>
                <Typography variant="body2" color="text.secondary">
                  {r.dataset}
                </Typography>
                <Box sx={{ display: "flex", alignItems: "center", gap: 1, mt: 0.5 }}>
                  <Typography
                    variant="h6"
                    sx={{
                      fontWeight: 700,
                      color: r.rate >= 95 ? "success.main" : r.rate >= 85 ? "warning.main" : "error.main",
                    }}
                  >
                    {r.rate}%
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    ({r.success}/{r.total})
                  </Typography>
                </Box>
                <LinearProgress
                  variant="determinate"
                  value={r.rate}
                  color={r.rate >= 95 ? "success" : r.rate >= 85 ? "warning" : "error"}
                  sx={{ mt: 1, height: 6, borderRadius: 3 }}
                />
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>
      <Box sx={{ mt: 2 }}>
        <MuiTooltip title="시뮬레이터 환경에서만 실행 가능">
          <span>
            <Button
              variant="outlined"
              startIcon={<PlayArrowIcon />}
              disabled
            >
              벤치마크 실행
            </Button>
          </span>
        </MuiTooltip>
      </Box>

      {/* Add / Edit Dialog */}
      <Dialog
        open={dialogOpen}
        onClose={() => { setDialogOpen(false); setEditEntry(null); }}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>{editEntry ? "골든 데이터 편집" : "골든 데이터 추가"}</DialogTitle>
        <DialogContent>
          <Box sx={{ display: "flex", flexDirection: "column", gap: 2, pt: 1 }}>
            <TextField
              label="질문 *"
              multiline
              minRows={3}
              value={form.question}
              onChange={(e) => setForm((f) => ({ ...f, question: e.target.value }))}
              fullWidth
            />
            <TextField
              label="기대 답변 *"
              multiline
              minRows={4}
              value={form.expected_answer}
              onChange={(e) => setForm((f) => ({ ...f, expected_answer: e.target.value }))}
              fullWidth
            />
            <FormControl fullWidth size="small">
              <InputLabel>카테고리</InputLabel>
              <Select
                label="카테고리"
                value={form.category}
                onChange={(e) => setForm((f) => ({ ...f, category: e.target.value }))}
              >
                {CATEGORY_OPTIONS.map((o) => (
                  <MenuItem key={o.value} value={o.value}>{o.label}</MenuItem>
                ))}
              </Select>
            </FormControl>
            <FormControl fullWidth size="small">
              <InputLabel>난이도</InputLabel>
              <Select
                label="난이도"
                value={form.difficulty}
                onChange={(e) => setForm((f) => ({ ...f, difficulty: e.target.value }))}
              >
                {DIFFICULTY_OPTIONS.map((o) => (
                  <MenuItem key={o.value} value={o.value}>{o.label}</MenuItem>
                ))}
              </Select>
            </FormControl>
            <TextField
              label="출처"
              value={form.source}
              onChange={(e) => setForm((f) => ({ ...f, source: e.target.value }))}
              fullWidth
              size="small"
            />
            <TextField
              label="키워드 (쉼표로 구분)"
              value={form.keywords}
              onChange={(e) => setForm((f) => ({ ...f, keywords: e.target.value }))}
              fullWidth
              size="small"
              helperText="예: 가스안전, 내부규정, 출장비"
            />
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => { setDialogOpen(false); setEditEntry(null); }}>취소</Button>
          <Button
            variant="contained"
            onClick={handleSave}
            disabled={!form.question.trim() || !form.expected_answer.trim() || isSaving}
          >
            {isSaving ? <CircularProgress size={20} /> : (editEntry ? "저장" : "추가")}
          </Button>
        </DialogActions>
      </Dialog>

      {/* Delete Confirm Dialog */}
      <Dialog
        open={!!deleteConfirmId}
        onClose={() => setDeleteConfirmId(null)}
        maxWidth="xs"
        fullWidth
      >
        <DialogTitle>골든 데이터 삭제</DialogTitle>
        <DialogContent>
          <Typography>이 항목을 삭제하시겠습니까? 이 작업은 취소할 수 없습니다.</Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDeleteConfirmId(null)}>취소</Button>
          <Button
            variant="contained"
            color="error"
            onClick={() => deleteConfirmId && deleteMutation.mutate(deleteConfirmId)}
            disabled={deleteMutation.isPending}
          >
            {deleteMutation.isPending ? <CircularProgress size={20} /> : "삭제"}
          </Button>
        </DialogActions>
      </Dialog>
    </>
  );
}

// ── Tab 6: Correction Review Queue ──

interface CorrectionEntry {
  id: string;
  message_id: string;
  session_id: string;
  query: string;
  original_answer: string;
  corrected_answer: string;
  correction_reason: string;
  category: string;
  difficulty: string;
  status: "pending" | "approved" | "rejected";
  submitted_by: string;
  submitted_at: string;
  reviewed_by: string | null;
  reviewed_at: string | null;
  reviewer_comment: string;
  golden_data_id: string | null;
}

interface CorrectionStats {
  total: number;
  pending: number;
  approved: number;
  rejected: number;
}

function correctionStatusColor(
  status: string
): "warning" | "success" | "error" | "default" {
  if (status === "pending") return "warning";
  if (status === "approved") return "success";
  if (status === "rejected") return "error";
  return "default";
}

function correctionStatusLabel(status: string): string {
  if (status === "pending") return "대기중";
  if (status === "approved") return "승인";
  if (status === "rejected") return "거부";
  return status;
}

function CorrectionQueueTab() {
  const queryClient = useQueryClient();
  const [statusFilter, setStatusFilter] = useState<string | undefined>(undefined);
  const [reviewTarget, setReviewTarget] = useState<CorrectionEntry | null>(null);
  const [reviewComment, setReviewComment] = useState("");
  const [reviewSubmitting, setReviewSubmitting] = useState(false);

  const { data: statsData, isLoading: statsLoading } = useQuery({
    queryKey: ["corrections-stats"],
    queryFn: () => correctionsApi.stats(),
    refetchInterval: 30000,
  });

  const { data: listData, isLoading: listLoading, error: listError } = useQuery({
    queryKey: ["corrections-list", statusFilter],
    queryFn: () => correctionsApi.list(statusFilter),
    refetchInterval: 30000,
  });

  const reviewMutation = useMutation({
    mutationFn: ({ id, approved, comment }: { id: string; approved: boolean; comment: string }) =>
      correctionsApi.review(id, { approved, reviewer_comment: comment }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["corrections-list"] });
      queryClient.invalidateQueries({ queryKey: ["corrections-stats"] });
      queryClient.invalidateQueries({ queryKey: ["golden-data"] });
      setReviewTarget(null);
      setReviewComment("");
      setReviewSubmitting(false);
    },
    onError: () => {
      setReviewSubmitting(false);
    },
  });

  const stats: CorrectionStats = statsData?.data ?? {
    total: 0,
    pending: 0,
    approved: 0,
    rejected: 0,
  };
  const corrections: CorrectionEntry[] = listData?.data?.corrections ?? [];

  const statsCards = [
    { label: "전체", value: stats.total },
    { label: "대기 중", value: stats.pending, color: "warning.main" },
    { label: "승인", value: stats.approved, color: "success.main" },
    { label: "거부", value: stats.rejected, color: "error.main" },
  ];

  function handleReview(approved: boolean) {
    if (!reviewTarget) return;
    setReviewSubmitting(true);
    reviewMutation.mutate({ id: reviewTarget.id, approved, comment: reviewComment });
  }

  if (statsLoading || listLoading) return <CircularProgress />;
  if (listError) return <Alert severity="error">교정 목록을 불러올 수 없습니다.</Alert>;

  return (
    <>
      {/* Header with help tooltip */}
      <Box sx={{ display: "flex", alignItems: "center", gap: 0.5, mb: 2 }}>
        <Typography variant="h6">답변 교정 관리</Typography>
        <MuiTooltip title="교정 절차: 전문가가 AI 답변 교정 → 검토 대기열 → 관리자 승인 → 골든 데이터셋 자동 추가. 승인된 교정만 품질 향상에 반영됩니다." arrow>
          <HelpOutlineIcon sx={{ fontSize: 17, color: "text.secondary", cursor: "help" }} />
        </MuiTooltip>
      </Box>

      {/* Stats cards */}
      <Grid container spacing={2} sx={{ mb: 3 }}>
        {statsCards.map((card) => (
          <Grid key={card.label} size={{ xs: 6, sm: 3 }}>
            <MetricCard label={card.label} value={card.value} color={card.color} />
          </Grid>
        ))}
      </Grid>

      {/* Status filter chips */}
      <Box sx={{ mb: 2, display: "flex", gap: 1, flexWrap: "wrap" }}>
        {[
          { value: undefined, label: "전체" },
          { value: "pending", label: "대기중" },
          { value: "approved", label: "승인" },
          { value: "rejected", label: "거부" },
        ].map((opt) => (
          <Chip
            key={opt.label}
            label={opt.label}
            variant={statusFilter === opt.value ? "filled" : "outlined"}
            color={
              opt.value === "pending"
                ? "warning"
                : opt.value === "approved"
                ? "success"
                : opt.value === "rejected"
                ? "error"
                : "default"
            }
            onClick={() => setStatusFilter(opt.value)}
            clickable
          />
        ))}
      </Box>

      {/* Table */}
      <Card variant="outlined">
        <CardContent sx={{ p: 0 }}>
          {corrections.length > 0 ? (
            <TableContainer component={Paper} variant="outlined">
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell sx={{ minWidth: 160 }}>질문</TableCell>
                    <TableCell sx={{ minWidth: 160 }}>원본 답변</TableCell>
                    <TableCell sx={{ minWidth: 160 }}>교정 답변</TableCell>
                    <TableCell>교정 사유</TableCell>
                    <TableCell>제출자</TableCell>
                    <TableCell>상태</TableCell>
                    <TableCell>제출일</TableCell>
                    <TableCell align="center">액션</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {corrections.map((item) => (
                    <TableRow key={item.id} hover>
                      <TableCell>
                        <Typography variant="body2" sx={{ maxWidth: 200 }}>
                          {item.query.length > 60
                            ? `${item.query.slice(0, 60)}...`
                            : item.query}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Typography variant="body2" color="text.secondary" sx={{ maxWidth: 200 }}>
                          {item.original_answer.length > 60
                            ? `${item.original_answer.slice(0, 60)}...`
                            : item.original_answer}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Typography variant="body2" color="text.secondary" sx={{ maxWidth: 200 }}>
                          {item.corrected_answer.length > 60
                            ? `${item.corrected_answer.slice(0, 60)}...`
                            : item.corrected_answer}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Typography variant="body2" noWrap sx={{ maxWidth: 120 }}>
                          {item.correction_reason || "-"}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Typography variant="body2">{item.submitted_by}</Typography>
                      </TableCell>
                      <TableCell>
                        <Chip
                          label={correctionStatusLabel(item.status)}
                          color={correctionStatusColor(item.status)}
                          size="small"
                        />
                      </TableCell>
                      <TableCell>
                        <Typography variant="body2" color="text.secondary" sx={{ fontSize: 12 }}>
                          {new Date(item.submitted_at).toLocaleDateString("ko-KR")}
                        </Typography>
                      </TableCell>
                      <TableCell align="center">
                        {item.status === "pending" ? (
                          <MuiTooltip title="검토">
                            <IconButton
                              size="small"
                              color="primary"
                              onClick={() => {
                                setReviewTarget(item);
                                setReviewComment("");
                              }}
                            >
                              <RateReviewIcon fontSize="small" />
                            </IconButton>
                          </MuiTooltip>
                        ) : item.status === "approved" ? (
                          <Box sx={{ display: "flex", alignItems: "center", gap: 0.5 }}>
                            <CheckCircleIcon fontSize="small" color="success" />
                            {item.golden_data_id && (
                              <MuiTooltip title={`골든 ID: ${item.golden_data_id.slice(0, 8)}...`}>
                                <Typography variant="caption" color="success.main" sx={{ cursor: "help" }}>
                                  골든
                                </Typography>
                              </MuiTooltip>
                            )}
                          </Box>
                        ) : (
                          <CancelIcon fontSize="small" color="error" />
                        )}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          ) : (
            <Box sx={{ p: 3 }}>
              <Alert severity="info">
                {statusFilter
                  ? `'${correctionStatusLabel(statusFilter)}' 상태의 교정이 없습니다.`
                  : "등록된 교정이 없습니다."}
              </Alert>
            </Box>
          )}
        </CardContent>
      </Card>

      {/* Review Dialog */}
      <Dialog
        open={!!reviewTarget}
        onClose={() => !reviewSubmitting && setReviewTarget(null)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>교정 검토</DialogTitle>
        <DialogContent>
          {reviewTarget && (
            <Box sx={{ display: "flex", flexDirection: "column", gap: 2, pt: 1 }}>
              <Box>
                <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                  질문
                </Typography>
                <Typography variant="body1" sx={{ p: 1.5, bgcolor: "action.hover", borderRadius: 1 }}>
                  {reviewTarget.query}
                </Typography>
              </Box>
              <Grid container spacing={2}>
                <Grid size={{ xs: 12, md: 6 }}>
                  <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                    원본 답변
                  </Typography>
                  <Typography
                    variant="body2"
                    sx={{
                      p: 1.5,
                      bgcolor: "error.50",
                      border: "1px solid",
                      borderColor: "error.200",
                      borderRadius: 1,
                      minHeight: 120,
                      whiteSpace: "pre-wrap",
                    }}
                  >
                    {reviewTarget.original_answer}
                  </Typography>
                </Grid>
                <Grid size={{ xs: 12, md: 6 }}>
                  <Typography variant="subtitle2" color="success.main" gutterBottom>
                    교정 답변
                  </Typography>
                  <Typography
                    variant="body2"
                    sx={{
                      p: 1.5,
                      bgcolor: "success.50",
                      border: "1px solid",
                      borderColor: "success.200",
                      borderRadius: 1,
                      minHeight: 120,
                      whiteSpace: "pre-wrap",
                    }}
                  >
                    {reviewTarget.corrected_answer}
                  </Typography>
                </Grid>
              </Grid>
              {reviewTarget.correction_reason && (
                <Box>
                  <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                    교정 사유
                  </Typography>
                  <Typography variant="body2" sx={{ p: 1, bgcolor: "action.hover", borderRadius: 1 }}>
                    {reviewTarget.correction_reason}
                  </Typography>
                </Box>
              )}
              <Box sx={{ display: "flex", gap: 1 }}>
                <Chip label={`카테고리: ${reviewTarget.category}`} size="small" variant="outlined" />
                <Chip label={`제출자: ${reviewTarget.submitted_by}`} size="small" variant="outlined" />
                <Chip
                  label={`제출일: ${new Date(reviewTarget.submitted_at).toLocaleDateString("ko-KR")}`}
                  size="small"
                  variant="outlined"
                />
              </Box>
              <TextField
                label="검토 의견 (선택)"
                multiline
                minRows={2}
                value={reviewComment}
                onChange={(e) => setReviewComment(e.target.value)}
                fullWidth
                placeholder="승인/거부 이유를 입력하세요..."
              />
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button
            onClick={() => setReviewTarget(null)}
            disabled={reviewSubmitting}
          >
            취소
          </Button>
          <Button
            variant="outlined"
            color="error"
            startIcon={reviewSubmitting ? <CircularProgress size={16} /> : <CancelIcon />}
            onClick={() => handleReview(false)}
            disabled={reviewSubmitting}
          >
            거부
          </Button>
          <Button
            variant="contained"
            color="success"
            startIcon={reviewSubmitting ? <CircularProgress size={16} /> : <CheckCircleIcon />}
            onClick={() => handleReview(true)}
            disabled={reviewSubmitting}
          >
            승인 및 골든 데이터 추가
          </Button>
        </DialogActions>
      </Dialog>
    </>
  );
}

// ── Error Wrapper ──

function TabErrorBoundary({ children }: { children: React.ReactNode }) {
  try {
    return <>{children}</>;
  } catch {
    return <Alert severity="error">이 탭을 렌더링하는 중 오류가 발생했습니다.</Alert>;
  }
}

// ── Main: QualityDashboardPage ──

export default function QualityDashboardPage() {
  const [tabIndex, setTabIndex] = useState(0);

  return (
    <Layout title="RAG 품질 관리">
      <Tabs
        value={tabIndex}
        onChange={(_, v: number) => setTabIndex(v)}
        sx={{ borderBottom: 1, borderColor: "divider", mb: 1 }}
        variant="scrollable"
        scrollButtons="auto"
      >
        <Tab label="문서 처리 현황" />
        <Tab label="청크 품질" />
        <Tab label="임베딩 상태" />
        <Tab label="벡터 분포" />
        <Tab label="재처리 큐" />
        <Tab label="골든 데이터셋" />
        <Tab label="답변 교정" />
      </Tabs>

      <TabPanel value={tabIndex} index={0}>
        <TabErrorBoundary>
          <DocumentProcessingTab />
        </TabErrorBoundary>
      </TabPanel>
      <TabPanel value={tabIndex} index={1}>
        <TabErrorBoundary>
          <ChunkQualityTab />
        </TabErrorBoundary>
      </TabPanel>
      <TabPanel value={tabIndex} index={2}>
        <TabErrorBoundary>
          <EmbeddingStatusTab />
        </TabErrorBoundary>
      </TabPanel>
      <TabPanel value={tabIndex} index={3}>
        <TabErrorBoundary>
          <VectorDistributionTab />
        </TabErrorBoundary>
      </TabPanel>
      <TabPanel value={tabIndex} index={4}>
        <TabErrorBoundary>
          <ReprocessQueueTab />
        </TabErrorBoundary>
      </TabPanel>
      <TabPanel value={tabIndex} index={5}>
        <TabErrorBoundary>
          <GoldenDatasetTab />
        </TabErrorBoundary>
      </TabPanel>
      <TabPanel value={tabIndex} index={6}>
        <TabErrorBoundary>
          <CorrectionQueueTab />
        </TabErrorBoundary>
      </TabPanel>
    </Layout>
  );
}
