import { useState, useRef, useCallback, useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  Box,
  Typography,
  Button,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Card,
  CardContent,
  Chip,
  CircularProgress,
  Alert,
  Divider,
  Stack,
  Grid,
  LinearProgress,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Collapse,
  IconButton,
  TextField,
  Tooltip,
} from "@mui/material";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import StopIcon from "@mui/icons-material/Stop";
import DownloadIcon from "@mui/icons-material/Download";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import ExpandLessIcon from "@mui/icons-material/ExpandLess";
import CheckCircleIcon from "@mui/icons-material/CheckCircle";
import CancelIcon from "@mui/icons-material/Cancel";
import HistoryIcon from "@mui/icons-material/History";
import { benchmarkApi, API_BASE, getAuthHeaders } from "../../api/client";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface BenchmarkResultRow {
  id: number;
  category: string;
  difficulty: string;
  question: string;
  expected_answer: string;
  actual_answer: string;
  confidence: number;
  sources_count: number;
  latency_ms: number;
  success: boolean;
  semantic_similarity: number;
  rouge_1: number;
  rouge_l: number;
  composite_score: number;
  grade: string;
  length_penalty: number;
  category_score: number;
}

interface EvalSummary {
  total_questions: number;
  success_count: number;
  success_rate: number;
  avg_confidence: number;
  avg_latency_ms: number;
  total_time_sec: number;
  avg_composite_score: number;
  avg_semantic_similarity: number;
  avg_rouge_1: number;
  avg_rouge_l: number;
  grade_distribution: Record<string, number>;
}

interface CategoryStat {
  total: number;
  success: number;
  success_rate: number;
  avg_confidence: number;
  avg_composite_score: number;
  avg_semantic_similarity: number;
  avg_rouge_l: number;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const GRADE_COLORS: Record<string, string> = {
  A: "#22c55e",
  B: "#3b82f6",
  C: "#f59e0b",
  D: "#f97316",
  F: "#ef4444",
};

const CATEGORY_LABELS: Record<string, string> = {
  factual: "사실 확인",
  inference: "추론",
  multi_hop: "복합 추론",
  negative: "부정 질문",
};

function formatTime(seconds: number): string {
  if (seconds < 60) return `${Math.round(seconds)}초`;
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  return `${m}분 ${s}초`;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function BatchEvaluatorTab() {
  // Dataset selection
  const [selectedDataset, setSelectedDataset] = useState("");
  const [limit, setLimit] = useState<number | "">("");

  // Evaluation state
  const [isRunning, setIsRunning] = useState(false);
  const [progress, setProgress] = useState<{ done: number; total: number } | null>(null);
  const [results, setResults] = useState<BenchmarkResultRow[]>([]);
  const [summary, setSummary] = useState<EvalSummary | null>(null);
  const [categoryStats, setCategoryStats] = useState<Record<string, CategoryStat> | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [startTime, setStartTime] = useState<number | null>(null);

  // UI state
  const [expandedRow, setExpandedRow] = useState<number | null>(null);
  const [filterCategory, setFilterCategory] = useState("all");
  const [filterResult, setFilterResult] = useState("all");
  const [selectedHistory, setSelectedHistory] = useState("");

  const abortRef = useRef<AbortController | null>(null);

  // Fetch dataset list
  const { data: datasetsData } = useQuery({
    queryKey: ["batch-eval-datasets"],
    queryFn: () => benchmarkApi.listDatasets().then((r) => r.data),
  });

  // Fetch history
  const { data: historyData, refetch: refetchHistory } = useQuery({
    queryKey: ["batch-eval-history"],
    queryFn: () => benchmarkApi.listHistory().then((r) => r.data),
  });

  const datasets = datasetsData?.datasets ?? [];
  const history = historyData?.history ?? [];

  // ETA calculation
  const eta = (() => {
    if (!progress || !startTime || progress.done === 0) return null;
    const elapsed = (Date.now() - startTime) / 1000;
    const perQuestion = elapsed / progress.done;
    const remaining = (progress.total - progress.done) * perQuestion;
    return remaining;
  })();

  // ---------------------------------------------------------------------------
  // Run evaluation
  // ---------------------------------------------------------------------------
  const handleRun = useCallback(async () => {
    if (!selectedDataset) return;

    setIsRunning(true);
    setError(null);
    setResults([]);
    setSummary(null);
    setCategoryStats(null);
    setProgress(null);
    setExpandedRow(null);
    setStartTime(Date.now());

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const params = new URLSearchParams({ dataset: selectedDataset });
      if (limit && Number(limit) > 0) params.set("limit", String(limit));

      const response = await fetch(`${API_BASE}/admin/batch-evaluate/stream?${params}`, {
        headers: { ...getAuthHeaders() },
        signal: controller.signal,
      });

      if (!response.ok) {
        const text = await response.text();
        throw new Error(`HTTP ${response.status}: ${text.slice(0, 200)}`);
      }

      const reader = response.body!.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let currentEvent = "message";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (line.startsWith("event:")) {
            currentEvent = line.slice(6).trim();
          } else if (line.startsWith("data:")) {
            const jsonStr = line.slice(5).trim();
            if (!jsonStr) continue;

            try {
              const data = JSON.parse(jsonStr);

              if (currentEvent === "init") {
                setProgress({ done: 0, total: data.total });
              } else if (currentEvent === "progress") {
                setProgress({ done: data.done, total: data.total });
                setResults((prev) => [...prev, data.result as BenchmarkResultRow]);
              } else if (currentEvent === "complete") {
                setSummary(data.summary as EvalSummary);
                setCategoryStats(data.category_stats as Record<string, CategoryStat>);
              } else if (currentEvent === "error") {
                setError(data.message || "알 수 없는 오류");
              }
            } catch {
              // skip malformed JSON
            }
            currentEvent = "message";
          }
        }
      }
    } catch (e: unknown) {
      if ((e as Error).name !== "AbortError") {
        setError((e as Error).message || "평가 중 오류 발생");
      }
    } finally {
      setIsRunning(false);
      abortRef.current = null;
      refetchHistory();
    }
  }, [selectedDataset, limit, refetchHistory]);

  const handleStop = useCallback(() => {
    abortRef.current?.abort();
    setIsRunning(false);
  }, []);

  // ---------------------------------------------------------------------------
  // Load history result
  // ---------------------------------------------------------------------------
  const handleLoadHistory = useCallback(async (filename: string) => {
    if (!filename) return;
    try {
      const { data } = await benchmarkApi.getHistoryResult(filename);
      const d = data as Record<string, unknown>;
      setSummary(d.summary as EvalSummary);
      setCategoryStats(d.category_stats as Record<string, CategoryStat>);
      setResults((d.results as BenchmarkResultRow[]) ?? []);
      setProgress({
        done: (d.summary as EvalSummary)?.total_questions ?? 0,
        total: (d.summary as EvalSummary)?.total_questions ?? 0,
      });
      setError(null);
    } catch {
      setError("이전 결과 로드 실패");
    }
  }, []);

  useEffect(() => {
    if (selectedHistory) handleLoadHistory(selectedHistory);
  }, [selectedHistory, handleLoadHistory]);

  // ---------------------------------------------------------------------------
  // Excel export
  // ---------------------------------------------------------------------------
  const handleExport = useCallback(() => {
    if (results.length === 0) return;
    const headers = [
      "ID", "카테고리", "난이도", "질문", "기대답변", "실제답변",
      "결과", "등급", "신뢰도", "카테고리점수", "종합점수",
      "의미유사도", "ROUGE-L", "ROUGE-1", "길이페널티", "응답시간(ms)",
    ];
    const rows = results.map((r) => [
      r.id, r.category, r.difficulty, `"${r.question.replace(/"/g, '""')}"`,
      `"${r.expected_answer.replace(/"/g, '""')}"`,
      `"${r.actual_answer.replace(/"/g, '""')}"`,
      r.success ? "통과" : "실패", r.grade, r.confidence, r.category_score,
      r.composite_score, r.semantic_similarity, r.rouge_l, r.rouge_1,
      r.length_penalty, Math.round(r.latency_ms),
    ]);
    const csv = "\uFEFF" + [headers.join(","), ...rows.map((r) => r.join(","))].join("\n");
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `batch_eval_${selectedDataset || "results"}_${new Date().toISOString().slice(0, 10)}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }, [results, selectedDataset]);

  // ---------------------------------------------------------------------------
  // Filtered results
  // ---------------------------------------------------------------------------
  const filteredResults = results.filter((r) => {
    if (filterCategory !== "all" && r.category !== filterCategory) return false;
    if (filterResult === "pass" && !r.success) return false;
    if (filterResult === "fail" && r.success) return false;
    return true;
  });

  // Live stats from results
  const liveSuccessCount = results.filter((r) => r.success).length;
  const liveSuccessRate = results.length > 0 ? liveSuccessCount / results.length : 0;
  const liveAvgLatency = results.length > 0
    ? results.reduce((a, r) => a + r.latency_ms, 0) / results.length
    : 0;
  const liveAvgScore = results.length > 0
    ? results.reduce((a, r) => a + r.composite_score, 0) / results.length
    : 0;

  // Use final summary if available, otherwise live stats
  const displayRate = summary ? summary.success_rate : liveSuccessRate;
  const displaySuccess = summary ? summary.success_count : liveSuccessCount;
  const displayFail = summary
    ? summary.total_questions - summary.success_count
    : results.length - liveSuccessCount;
  const displayLatency = summary ? summary.avg_latency_ms : liveAvgLatency;
  const displayScore = summary ? summary.avg_composite_score : liveAvgScore;

  // Live grade distribution
  const gradeDist = summary?.grade_distribution ??
    results.reduce<Record<string, number>>((acc, r) => {
      acc[r.grade] = (acc[r.grade] || 0) + 1;
      return acc;
    }, {});

  // Live category stats
  const displayCatStats = categoryStats ?? (() => {
    const cats: Record<string, { total: number; success: number }> = {};
    for (const r of results) {
      if (!cats[r.category]) cats[r.category] = { total: 0, success: 0 };
      cats[r.category].total++;
      if (r.success) cats[r.category].success++;
    }
    const out: Record<string, CategoryStat> = {};
    for (const [cat, s] of Object.entries(cats)) {
      out[cat] = {
        total: s.total,
        success: s.success,
        success_rate: s.total > 0 ? s.success / s.total : 0,
        avg_confidence: 0,
        avg_composite_score: 0,
        avg_semantic_similarity: 0,
        avg_rouge_l: 0,
      };
    }
    return Object.keys(out).length > 0 ? out : null;
  })();

  // ---------------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------------
  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        배치 평가기
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
        골든 데이터셋으로 RAG 파이프라인 품질을 일괄 평가합니다.
        각 질문을 실제 RAG에 질의하고, 기대 답변과 비교하여 점수를 산출합니다.
      </Typography>

      {/* ---- Input Area ---- */}
      <Card variant="outlined" sx={{ mb: 3 }}>
        <CardContent>
          <Grid container spacing={2} alignItems="center">
            <Grid size={{ xs: 12, sm: 5 }}>
              <FormControl fullWidth size="small">
                <InputLabel>데이터셋</InputLabel>
                <Select
                  value={selectedDataset}
                  label="데이터셋"
                  onChange={(e) => setSelectedDataset(e.target.value)}
                  disabled={isRunning}
                >
                  {datasets.map((ds) => (
                    <MenuItem key={ds.name} value={ds.name}>
                      {ds.name} ({ds.question_count}문항)
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
            <Grid size={{ xs: 6, sm: 2 }}>
              <TextField
                label="최대 문항"
                type="number"
                size="small"
                fullWidth
                value={limit}
                onChange={(e) => setLimit(e.target.value ? Number(e.target.value) : "")}
                disabled={isRunning}
                placeholder="전체"
                slotProps={{ htmlInput: { min: 1, max: 500 } }}
              />
            </Grid>
            <Grid size={{ xs: 6, sm: 5 }}>
              <Stack direction="row" spacing={1}>
                <Button
                  variant="contained"
                  startIcon={isRunning ? <CircularProgress size={16} color="inherit" /> : <PlayArrowIcon />}
                  onClick={handleRun}
                  disabled={isRunning || !selectedDataset}
                >
                  {isRunning ? "평가 중..." : "평가 실행"}
                </Button>
                {isRunning && (
                  <Button
                    variant="outlined"
                    color="error"
                    startIcon={<StopIcon />}
                    onClick={handleStop}
                  >
                    중단
                  </Button>
                )}
                {results.length > 0 && !isRunning && (
                  <Button
                    variant="outlined"
                    startIcon={<DownloadIcon />}
                    onClick={handleExport}
                  >
                    CSV
                  </Button>
                )}
              </Stack>
            </Grid>
          </Grid>
        </CardContent>
      </Card>

      {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}

      {/* ---- Progress Bar ---- */}
      {progress && (
        <Card variant="outlined" sx={{ mb: 3 }}>
          <CardContent sx={{ py: 1.5 }}>
            <Stack direction="row" alignItems="center" spacing={2}>
              <Box sx={{ flex: 1 }}>
                <LinearProgress
                  variant="determinate"
                  value={(progress.done / progress.total) * 100}
                  sx={{ height: 8, borderRadius: 4 }}
                />
              </Box>
              <Typography variant="body2" color="text.secondary" sx={{ minWidth: 120, textAlign: "right" }}>
                {progress.done} / {progress.total} ({Math.round((progress.done / progress.total) * 100)}%)
              </Typography>
              {eta !== null && isRunning && (
                <Typography variant="body2" color="text.secondary" sx={{ minWidth: 100, textAlign: "right" }}>
                  {formatTime(eta)} 남음
                </Typography>
              )}
            </Stack>
          </CardContent>
        </Card>
      )}

      {/* ---- Summary Cards ---- */}
      {results.length > 0 && (
        <>
          <Grid container spacing={2} sx={{ mb: 3 }}>
            {[
              { label: "정확도", value: `${(displayRate * 100).toFixed(1)}%`, color: displayRate >= 0.8 ? "#22c55e" : displayRate >= 0.6 ? "#f59e0b" : "#ef4444" },
              { label: "통과", value: String(displaySuccess), color: "#22c55e" },
              { label: "실패", value: String(displayFail), color: "#ef4444" },
              { label: "평균 응답시간", value: `${(displayLatency / 1000).toFixed(1)}s`, color: "#3b82f6" },
              { label: "평균 점수", value: displayScore.toFixed(3), color: "#8b5cf6" },
            ].map((card) => (
              <Grid size={{ xs: 6, sm: 2.4 }} key={card.label}>
                <Card variant="outlined" sx={{ textAlign: "center", py: 1.5 }}>
                  <Typography variant="caption" color="text.secondary">{card.label}</Typography>
                  <Typography variant="h5" sx={{ color: card.color, fontWeight: 700 }}>
                    {card.value}
                  </Typography>
                </Card>
              </Grid>
            ))}
          </Grid>

          {/* ---- Category Bars ---- */}
          {displayCatStats && (
            <Card variant="outlined" sx={{ mb: 3 }}>
              <CardContent>
                <Typography variant="subtitle2" gutterBottom>카테고리별 성공률</Typography>
                <Stack spacing={1.5}>
                  {Object.entries(displayCatStats).map(([cat, stat]) => {
                    const rate = stat.success_rate;
                    return (
                      <Box key={cat}>
                        <Stack direction="row" justifyContent="space-between" alignItems="center">
                          <Typography variant="body2" sx={{ minWidth: 80 }}>
                            {CATEGORY_LABELS[cat] || cat}
                          </Typography>
                          <Box sx={{ flex: 1, mx: 2 }}>
                            <LinearProgress
                              variant="determinate"
                              value={rate * 100}
                              sx={{
                                height: 12,
                                borderRadius: 6,
                                bgcolor: "action.hover",
                                "& .MuiLinearProgress-bar": {
                                  bgcolor: rate >= 0.8 ? "#22c55e" : rate >= 0.6 ? "#f59e0b" : "#ef4444",
                                  borderRadius: 6,
                                },
                              }}
                            />
                          </Box>
                          <Typography variant="body2" sx={{ minWidth: 100, textAlign: "right" }}>
                            {(rate * 100).toFixed(1)}% ({stat.success}/{stat.total})
                          </Typography>
                        </Stack>
                      </Box>
                    );
                  })}
                </Stack>
              </CardContent>
            </Card>
          )}

          {/* ---- Grade Distribution ---- */}
          <Card variant="outlined" sx={{ mb: 3 }}>
            <CardContent>
              <Typography variant="subtitle2" gutterBottom>등급 분포</Typography>
              <Stack direction="row" spacing={1} flexWrap="wrap">
                {["A", "B", "C", "D", "F"].map((g) => (
                  <Chip
                    key={g}
                    label={`${g}: ${gradeDist[g] || 0}`}
                    sx={{
                      bgcolor: GRADE_COLORS[g] + "22",
                      color: GRADE_COLORS[g],
                      fontWeight: 700,
                      fontSize: "0.85rem",
                      border: `1px solid ${GRADE_COLORS[g]}44`,
                    }}
                  />
                ))}
              </Stack>
            </CardContent>
          </Card>

          <Divider sx={{ mb: 2 }} />

          {/* ---- Filters ---- */}
          <Stack direction="row" spacing={2} sx={{ mb: 2 }} alignItems="center">
            <Typography variant="subtitle2">상세 결과</Typography>
            <FormControl size="small" sx={{ minWidth: 120 }}>
              <InputLabel>카테고리</InputLabel>
              <Select
                value={filterCategory}
                label="카테고리"
                onChange={(e) => setFilterCategory(e.target.value)}
              >
                <MenuItem value="all">전체</MenuItem>
                {Object.entries(CATEGORY_LABELS).map(([k, v]) => (
                  <MenuItem key={k} value={k}>{v}</MenuItem>
                ))}
              </Select>
            </FormControl>
            <FormControl size="small" sx={{ minWidth: 100 }}>
              <InputLabel>결과</InputLabel>
              <Select
                value={filterResult}
                label="결과"
                onChange={(e) => setFilterResult(e.target.value)}
              >
                <MenuItem value="all">전체</MenuItem>
                <MenuItem value="pass">통과</MenuItem>
                <MenuItem value="fail">실패</MenuItem>
              </Select>
            </FormControl>
            <Typography variant="body2" color="text.secondary">
              {filteredResults.length}건
            </Typography>
          </Stack>

          {/* ---- Results Table ---- */}
          <TableContainer component={Paper} variant="outlined" sx={{ mb: 3 }}>
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell width={30}>#</TableCell>
                  <TableCell width={80}>카테고리</TableCell>
                  <TableCell>질문</TableCell>
                  <TableCell width={50} align="center">등급</TableCell>
                  <TableCell width={60} align="center">결과</TableCell>
                  <TableCell width={70} align="right">점수</TableCell>
                  <TableCell width={70} align="right">신뢰도</TableCell>
                  <TableCell width={60} align="right">시간</TableCell>
                  <TableCell width={30} />
                </TableRow>
              </TableHead>
              <TableBody>
                {filteredResults.map((r) => (
                  <>
                    <TableRow
                      key={r.id}
                      hover
                      onClick={() => setExpandedRow(expandedRow === r.id ? null : r.id)}
                      sx={{ cursor: "pointer", "& td": { borderBottom: expandedRow === r.id ? "none" : undefined } }}
                    >
                      <TableCell>{r.id}</TableCell>
                      <TableCell>
                        <Chip label={CATEGORY_LABELS[r.category] || r.category} size="small" variant="outlined" />
                      </TableCell>
                      <TableCell sx={{ maxWidth: 300, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                        {r.question}
                      </TableCell>
                      <TableCell align="center">
                        <Chip
                          label={r.grade}
                          size="small"
                          sx={{
                            bgcolor: (GRADE_COLORS[r.grade] || "#999") + "22",
                            color: GRADE_COLORS[r.grade] || "#999",
                            fontWeight: 700,
                            minWidth: 32,
                          }}
                        />
                      </TableCell>
                      <TableCell align="center">
                        {r.success ? (
                          <CheckCircleIcon sx={{ color: "#22c55e", fontSize: 20 }} />
                        ) : (
                          <CancelIcon sx={{ color: "#ef4444", fontSize: 20 }} />
                        )}
                      </TableCell>
                      <TableCell align="right">
                        <Tooltip title={`카테고리: ${r.category_score.toFixed(3)} / 종합: ${r.composite_score.toFixed(3)}`}>
                          <span>{r.category_score.toFixed(2)}</span>
                        </Tooltip>
                      </TableCell>
                      <TableCell align="right">{r.confidence.toFixed(2)}</TableCell>
                      <TableCell align="right">{(r.latency_ms / 1000).toFixed(1)}s</TableCell>
                      <TableCell>
                        <IconButton size="small">
                          {expandedRow === r.id ? <ExpandLessIcon /> : <ExpandMoreIcon />}
                        </IconButton>
                      </TableCell>
                    </TableRow>
                    <TableRow key={`${r.id}-detail`}>
                      <TableCell colSpan={9} sx={{ py: 0, px: 2 }}>
                        <Collapse in={expandedRow === r.id}>
                          <Box sx={{ py: 2 }}>
                            <Grid container spacing={2}>
                              <Grid size={{ xs: 12, md: 6 }}>
                                <Typography variant="caption" color="text.secondary">기대 답변</Typography>
                                <Typography variant="body2" sx={{ mt: 0.5, p: 1.5, bgcolor: "action.hover", borderRadius: 1, whiteSpace: "pre-wrap", maxHeight: 200, overflow: "auto" }}>
                                  {r.expected_answer}
                                </Typography>
                              </Grid>
                              <Grid size={{ xs: 12, md: 6 }}>
                                <Typography variant="caption" color="text.secondary">실제 답변</Typography>
                                <Typography variant="body2" sx={{ mt: 0.5, p: 1.5, bgcolor: r.success ? "success.main" : "error.main", color: "#fff", borderRadius: 1, whiteSpace: "pre-wrap", maxHeight: 200, overflow: "auto", opacity: 0.9 }}>
                                  {r.actual_answer}
                                </Typography>
                              </Grid>
                            </Grid>
                            <Stack direction="row" spacing={1} sx={{ mt: 1.5 }} flexWrap="wrap">
                              <Chip label={`sem=${r.semantic_similarity.toFixed(3)}`} size="small" variant="outlined" />
                              <Chip label={`rouge_l=${r.rouge_l.toFixed(3)}`} size="small" variant="outlined" />
                              <Chip label={`rouge_1=${r.rouge_1.toFixed(3)}`} size="small" variant="outlined" />
                              <Chip label={`cat_score=${r.category_score.toFixed(3)}`} size="small" variant="outlined" />
                              <Chip label={`len_pen=${r.length_penalty.toFixed(2)}`} size="small" variant="outlined" />
                              <Chip label={`sources=${r.sources_count}`} size="small" variant="outlined" />
                            </Stack>
                          </Box>
                        </Collapse>
                      </TableCell>
                    </TableRow>
                  </>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </>
      )}

      {/* ---- History ---- */}
      {history.length > 0 && (
        <Card variant="outlined" sx={{ mt: 2 }}>
          <CardContent>
            <Stack direction="row" spacing={2} alignItems="center">
              <HistoryIcon color="action" />
              <Typography variant="subtitle2">이전 평가 기록</Typography>
              <FormControl size="small" sx={{ minWidth: 300 }}>
                <Select
                  value={selectedHistory}
                  displayEmpty
                  onChange={(e) => setSelectedHistory(e.target.value)}
                >
                  <MenuItem value="">선택...</MenuItem>
                  {history.map((h) => (
                    <MenuItem key={h.filename} value={h.filename}>
                      {h.dataset} — {(h.success_rate * 100).toFixed(1)}% ({h.total_questions}문항)
                      {h.created_at ? ` — ${h.created_at.slice(0, 10)}` : ""}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Stack>
          </CardContent>
        </Card>
      )}
    </Box>
  );
}
