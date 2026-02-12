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
} from "@mui/material";
import RefreshIcon from "@mui/icons-material/Refresh";
import DeleteIcon from "@mui/icons-material/Delete";
import ReplayIcon from "@mui/icons-material/Replay";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import ExpandLessIcon from "@mui/icons-material/ExpandLess";
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
import { qualityApi } from "../api/client";
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

function DocumentProcessingTab() {
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
  const changes: DocumentChange[] = changesData?.data?.changes ?? [];

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
                  {changes.map((change, idx) => (
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
    </Layout>
  );
}
