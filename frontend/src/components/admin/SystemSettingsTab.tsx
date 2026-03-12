import { useState, useEffect } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  Box,
  Typography,
  Card,
  CardContent,
  Grid,
  CircularProgress,
  Alert,
  TextField,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Snackbar,
  Switch,
  FormControlLabel,
  Slider,
  Divider,
  Stack,
} from "@mui/material";
import MemoryIcon from "@mui/icons-material/Memory";
import StorageIcon from "@mui/icons-material/Storage";
import TuneIcon from "@mui/icons-material/Tune";
import { adminApi } from "../../api/client";

interface RagSettings {
  context_max_chunks: number;
  llm_max_tokens: number;
  llm_temperature: number;
  use_rerank: boolean;
  retrieval_top_k: number;
  rerank_top_n: number;
  multi_query_enabled: boolean;
  self_rag_enabled: boolean;
  agentic_rag_enabled: boolean;
  bm25_weight: number;
  vector_weight: number;
  few_shot_max_examples: number;
  confidence_low: number;
  confidence_high: number;
}

const DEFAULT_SETTINGS: RagSettings = {
  context_max_chunks: 7,
  llm_max_tokens: 512,
  llm_temperature: 0.2,
  use_rerank: false,
  retrieval_top_k: 10,
  rerank_top_n: 5,
  multi_query_enabled: false,
  self_rag_enabled: false,
  agentic_rag_enabled: false,
  bm25_weight: 0.3,
  vector_weight: 0.7,
  few_shot_max_examples: 2,
  confidence_low: 0.3,
  confidence_high: 0.8,
};

export default function SystemSettingsTab() {
  const queryClient = useQueryClient();

  const [form, setForm] = useState<RagSettings>(DEFAULT_SETTINGS);
  const [confirmOpen, setConfirmOpen] = useState(false);
  const [snackbar, setSnackbar] = useState<{
    open: boolean;
    message: string;
    severity: "success" | "error";
  }>({ open: false, message: "", severity: "success" });

  const { data, isLoading, error } = useQuery({
    queryKey: ["admin-rag-settings"],
    queryFn: async () => {
      const res = await adminApi.getRagSettings();
      return res.data as RagSettings;
    },
  });

  useEffect(() => {
    if (data) {
      setForm(data);
    }
  }, [data]);

  const updateMutation = useMutation({
    mutationFn: async (settings: Partial<RagSettings>) => {
      const res = await adminApi.updateRagSettings(
        settings as Record<string, unknown>,
      );
      return res.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["admin-rag-settings"] });
      setSnackbar({
        open: true,
        message: "RAG 설정이 저장되었습니다.",
        severity: "success",
      });
    },
    onError: () => {
      setSnackbar({
        open: true,
        message: "설정 저장에 실패했습니다.",
        severity: "error",
      });
    },
  });

  const handleSave = () => {
    setConfirmOpen(true);
  };

  const handleConfirmSave = () => {
    setConfirmOpen(false);
    updateMutation.mutate(form);
  };

  const handleReset = () => {
    queryClient.invalidateQueries({ queryKey: ["admin-rag-settings"] });
    if (data) {
      setForm(data);
    }
    setSnackbar({
      open: true,
      message: "서버의 현재 설정값으로 초기화했습니다.",
      severity: "success",
    });
  };

  const setField = <K extends keyof RagSettings>(key: K, value: RagSettings[K]) => {
    setForm((prev) => ({ ...prev, [key]: value }));
  };

  if (isLoading) {
    return (
      <Box sx={{ display: "flex", justifyContent: "center", py: 8 }}>
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Alert severity="error" sx={{ mb: 2 }}>
        설정을 불러오는 데 실패했습니다. 잠시 후 다시 시도해 주세요.
      </Alert>
    );
  }

  return (
    <Box>
      <Typography variant="h5" fontWeight={700} gutterBottom>
        시스템 설정
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
        RAG 파이프라인 및 LLM 런타임 파라미터를 조정합니다.
      </Typography>

      <Alert severity="warning" sx={{ mb: 3 }}>
        런타임 변경만 적용됩니다. 서버 재시작 시 .env 기본값으로 초기화됩니다.
      </Alert>

      <Grid container spacing={3}>
        {/* Left column: LLM 설정 */}
        <Grid size={{ xs: 12, md: 6 }}>
          <Card variant="outlined">
            <CardContent>
              <Typography
                variant="h6"
                fontWeight={600}
                gutterBottom
                sx={{ display: "flex", alignItems: "center", gap: 1 }}
              >
                <MemoryIcon fontSize="small" color="primary" />
                LLM 설정
              </Typography>
              <Divider sx={{ mb: 2 }} />

              <Stack spacing={3}>
                {/* Temperature slider */}
                <Box>
                  <Typography variant="body2" color="text.secondary" gutterBottom>
                    Temperature:{" "}
                    <strong>{form.llm_temperature.toFixed(1)}</strong>
                  </Typography>
                  <Slider
                    value={form.llm_temperature}
                    min={0.0}
                    max={1.0}
                    step={0.1}
                    marks
                    valueLabelDisplay="auto"
                    onChange={(_, v) => setField("llm_temperature", v as number)}
                    sx={{ color: "primary.main" }}
                  />
                  <Box sx={{ display: "flex", justifyContent: "space-between" }}>
                    <Typography variant="caption" color="text.secondary">
                      결정적
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      창의적
                    </Typography>
                  </Box>
                </Box>

                {/* Max Tokens */}
                <TextField
                  label="최대 토큰 수 (LLM Max Tokens)"
                  size="small"
                  fullWidth
                  type="number"
                  value={form.llm_max_tokens}
                  onChange={(e) => {
                    const v = Math.max(64, Math.min(8192, Number(e.target.value)));
                    setField("llm_max_tokens", v);
                  }}
                  inputProps={{ min: 64, max: 8192, step: 64 }}
                  helperText="범위: 64 ~ 8192"
                />
              </Stack>
            </CardContent>
          </Card>
        </Grid>

        {/* Right column: RAG 파이프라인 설정 */}
        <Grid size={{ xs: 12, md: 6 }}>
          <Card variant="outlined">
            <CardContent>
              <Typography
                variant="h6"
                fontWeight={600}
                gutterBottom
                sx={{ display: "flex", alignItems: "center", gap: 1 }}
              >
                <StorageIcon fontSize="small" color="primary" />
                RAG 파이프라인 설정
              </Typography>
              <Divider sx={{ mb: 2 }} />

              <Stack spacing={2.5}>
                <Grid container spacing={2}>
                  <Grid size={6}>
                    <TextField
                      label="컨텍스트 최대 청크 수"
                      size="small"
                      fullWidth
                      type="number"
                      value={form.context_max_chunks}
                      onChange={(e) => {
                        const v = Math.max(1, Math.min(30, Number(e.target.value)));
                        setField("context_max_chunks", v);
                      }}
                      inputProps={{ min: 1, max: 30, step: 1 }}
                      helperText="1 ~ 30"
                    />
                  </Grid>
                  <Grid size={6}>
                    <TextField
                      label="검색 Top-K"
                      size="small"
                      fullWidth
                      type="number"
                      value={form.retrieval_top_k}
                      onChange={(e) => {
                        const v = Math.max(1, Math.min(100, Number(e.target.value)));
                        setField("retrieval_top_k", v);
                      }}
                      inputProps={{ min: 1, max: 100, step: 1 }}
                      helperText="1 ~ 100"
                    />
                  </Grid>
                  <Grid size={6}>
                    <TextField
                      label="Rerank Top-N"
                      size="small"
                      fullWidth
                      type="number"
                      value={form.rerank_top_n}
                      onChange={(e) => {
                        const v = Math.max(1, Math.min(30, Number(e.target.value)));
                        setField("rerank_top_n", v);
                      }}
                      inputProps={{ min: 1, max: 30, step: 1 }}
                      helperText="1 ~ 30"
                    />
                  </Grid>
                  <Grid size={6}>
                    <TextField
                      label="Few-Shot 예시 수"
                      size="small"
                      fullWidth
                      type="number"
                      value={form.few_shot_max_examples}
                      onChange={(e) => {
                        const v = Math.max(0, Math.min(10, Number(e.target.value)));
                        setField("few_shot_max_examples", v);
                      }}
                      inputProps={{ min: 0, max: 10, step: 1 }}
                      helperText="0 ~ 10"
                    />
                  </Grid>
                  <Grid size={6}>
                    <TextField
                      label="Confidence Low"
                      size="small"
                      fullWidth
                      type="number"
                      value={form.confidence_low}
                      onChange={(e) => {
                        const v = Math.max(0.0, Math.min(1.0, Number(e.target.value)));
                        setField("confidence_low", parseFloat(v.toFixed(2)));
                      }}
                      inputProps={{ min: 0.0, max: 1.0, step: 0.05 }}
                      helperText="0.0 ~ 1.0"
                    />
                  </Grid>
                  <Grid size={6}>
                    <TextField
                      label="Confidence High"
                      size="small"
                      fullWidth
                      type="number"
                      value={form.confidence_high}
                      onChange={(e) => {
                        const v = Math.max(0.0, Math.min(1.0, Number(e.target.value)));
                        setField("confidence_high", parseFloat(v.toFixed(2)));
                      }}
                      inputProps={{ min: 0.0, max: 1.0, step: 0.05 }}
                      helperText="0.0 ~ 1.0"
                    />
                  </Grid>
                </Grid>

                {/* Rerank switch */}
                <FormControlLabel
                  control={
                    <Switch
                      checked={form.use_rerank}
                      onChange={(e) => setField("use_rerank", e.target.checked)}
                      color="primary"
                    />
                  }
                  label={
                    <Box>
                      <Typography variant="body2" fontWeight={500}>
                        Reranker 사용
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        교차 인코더 기반 재순위화 (Off시 더 빠름)
                      </Typography>
                    </Box>
                  }
                />

                {/* BM25 weight slider */}
                <Box>
                  <Typography variant="body2" color="text.secondary" gutterBottom>
                    BM25 가중치: <strong>{form.bm25_weight.toFixed(2)}</strong>
                  </Typography>
                  <Slider
                    value={form.bm25_weight}
                    min={0.0}
                    max={1.0}
                    step={0.05}
                    marks
                    valueLabelDisplay="auto"
                    onChange={(_, v) => setField("bm25_weight", v as number)}
                    sx={{ color: "secondary.main" }}
                  />
                  <Box sx={{ display: "flex", justifyContent: "space-between" }}>
                    <Typography variant="caption" color="text.secondary">
                      0.0 (벡터만)
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      1.0 (BM25만)
                    </Typography>
                  </Box>
                </Box>

                {/* Vector weight slider */}
                <Box>
                  <Typography variant="body2" color="text.secondary" gutterBottom>
                    Vector 가중치: <strong>{form.vector_weight.toFixed(2)}</strong>
                  </Typography>
                  <Slider
                    value={form.vector_weight}
                    min={0.0}
                    max={1.0}
                    step={0.05}
                    marks
                    valueLabelDisplay="auto"
                    onChange={(_, v) => setField("vector_weight", v as number)}
                    sx={{ color: "primary.main" }}
                  />
                  <Box sx={{ display: "flex", justifyContent: "space-between" }}>
                    <Typography variant="caption" color="text.secondary">
                      0.0 (BM25만)
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      1.0 (벡터만)
                    </Typography>
                  </Box>
                </Box>
              </Stack>
            </CardContent>
          </Card>
        </Grid>

        {/* Full width: 고급 RAG 기능 */}
        <Grid size={{ xs: 12 }}>
          <Card variant="outlined">
            <CardContent>
              <Typography
                variant="h6"
                fontWeight={600}
                gutterBottom
                sx={{ display: "flex", alignItems: "center", gap: 1 }}
              >
                <TuneIcon fontSize="small" color="primary" />
                고급 RAG 기능
              </Typography>
              <Divider sx={{ mb: 2 }} />

              <Stack spacing={1.5}>
                <FormControlLabel
                  control={
                    <Switch
                      checked={form.multi_query_enabled}
                      onChange={(e) =>
                        setField("multi_query_enabled", e.target.checked)
                      }
                      color="primary"
                    />
                  }
                  label={
                    <Box>
                      <Typography variant="body2" fontWeight={500}>
                        Multi-Query
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        다중 관점 쿼리 생성으로 검색 품질 향상
                      </Typography>
                    </Box>
                  }
                />

                <FormControlLabel
                  control={
                    <Switch
                      checked={form.self_rag_enabled}
                      onChange={(e) =>
                        setField("self_rag_enabled", e.target.checked)
                      }
                      color="primary"
                    />
                  }
                  label={
                    <Box>
                      <Typography variant="body2" fontWeight={500}>
                        Self-RAG
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        응답 생성 후 근거성 자동 평가 및 재시도
                      </Typography>
                    </Box>
                  }
                />

                <FormControlLabel
                  control={
                    <Switch
                      checked={form.agentic_rag_enabled}
                      onChange={(e) =>
                        setField("agentic_rag_enabled", e.target.checked)
                      }
                      color="primary"
                    />
                  }
                  label={
                    <Box>
                      <Typography variant="body2" fontWeight={500}>
                        Agentic RAG
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        동적 전략 라우팅 (factual / inference / negative 자동 분류)
                      </Typography>
                    </Box>
                  }
                />
              </Stack>
            </CardContent>
          </Card>
        </Grid>

        {/* Action buttons */}
        <Grid size={{ xs: 12 }}>
          <Stack direction="row" spacing={2} justifyContent="flex-end">
            <Button variant="outlined" color="inherit" onClick={handleReset}>
              초기화
            </Button>
            <Button
              variant="contained"
              onClick={handleSave}
              disabled={updateMutation.isPending}
              startIcon={
                updateMutation.isPending ? (
                  <CircularProgress size={16} color="inherit" />
                ) : undefined
              }
            >
              저장
            </Button>
          </Stack>
        </Grid>
      </Grid>

      {/* Confirm Dialog */}
      <Dialog open={confirmOpen} onClose={() => setConfirmOpen(false)}>
        <DialogTitle>RAG 설정 변경 확인</DialogTitle>
        <DialogContent>
          <Typography variant="body2">
            RAG 설정을 변경하시겠습니까? 운영 중인 파이프라인에 즉시 반영됩니다.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setConfirmOpen(false)}>취소</Button>
          <Button variant="contained" onClick={handleConfirmSave}>
            확인
          </Button>
        </DialogActions>
      </Dialog>

      {/* Snackbar */}
      <Snackbar
        open={snackbar.open}
        autoHideDuration={4000}
        onClose={() => setSnackbar((s) => ({ ...s, open: false }))}
        anchorOrigin={{ vertical: "bottom", horizontal: "center" }}
      >
        <Alert
          onClose={() => setSnackbar((s) => ({ ...s, open: false }))}
          severity={snackbar.severity}
          sx={{ width: "100%" }}
        >
          {snackbar.message}
        </Alert>
      </Snackbar>
    </Box>
  );
}
