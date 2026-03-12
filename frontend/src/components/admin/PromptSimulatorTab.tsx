import { useState } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import {
  Box,
  Typography,
  TextField,
  Button,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Card,
  CardContent,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Chip,
  CircularProgress,
  Alert,
  Divider,
  Stack,
  Grid,
} from "@mui/material";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import PreviewIcon from "@mui/icons-material/Preview";
import { adminApi } from "../../api/client";

interface SimulateResult {
  system_prompt: string;
  user_prompt: string;
  detected_doc_type: string;
  prompt_length_chars: number;
  system_prompt_length: number;
  user_prompt_length: number;
}

interface PlaygroundResult {
  response: string;
  latency_ms: number;
  model_used: string;
  tokens_used?: number;
}

const MODEL_HINT_OPTIONS = [
  { value: "", label: "기본 (자동)" },
  { value: "7B", label: "7B" },
  { value: "14B", label: "14B" },
  { value: "32B", label: "32B" },
  { value: "70B", label: "70B" },
];

const DOC_TYPE_LABELS: Record<string, string> = {
  legal: "법률/규정",
  technical: "기술",
  general: "일반",
};

export default function PromptSimulatorTab() {
  const [query, setQuery] = useState("");
  const [promptName, setPromptName] = useState("");
  const [modelHint, setModelHint] = useState("");
  const [contextChunksJson, setContextChunksJson] = useState("");
  const [simulateResult, setSimulateResult] = useState<SimulateResult | null>(null);
  const [playgroundResult, setPlaygroundResult] = useState<PlaygroundResult | null>(null);
  const [jsonError, setJsonError] = useState<string | null>(null);

  // Fetch prompt names
  const { data: promptsData, isLoading: promptsLoading } = useQuery({
    queryKey: ["admin-prompts"],
    queryFn: () => adminApi.getPrompts(),
  });

  const promptKeys: string[] = (() => {
    const raw = promptsData?.data as { prompts?: Record<string, string> } | undefined;
    if (!raw?.prompts) return [];
    return Object.keys(raw.prompts).filter(
      (k) =>
        k.includes("system") ||
        k === "rag_system" ||
        k === "direct_system" ||
        k === "legal_system" ||
        k === "technical_system",
    );
  })();

  // Simulate mutation
  const simulateMutation = useMutation({
    mutationFn: (data: Parameters<typeof adminApi.simulatePrompt>[0]) =>
      adminApi.simulatePrompt(data),
    onSuccess: (res) => {
      setSimulateResult(res.data);
    },
  });

  // Playground mutation
  const playgroundMutation = useMutation({
    mutationFn: (data: Parameters<typeof adminApi.playground>[0]) =>
      adminApi.playground(data),
    onSuccess: (res) => {
      setPlaygroundResult(res.data);
    },
  });

  function parseContextChunks():
    | Array<{ content: string; metadata?: Record<string, unknown> }>
    | null {
    const trimmed = contextChunksJson.trim();
    if (!trimmed) return [];
    try {
      const parsed = JSON.parse(trimmed);
      if (!Array.isArray(parsed)) {
        setJsonError("JSON 배열 형식이어야 합니다. 예: [{\"content\": \"...\"}]");
        return null;
      }
      setJsonError(null);
      return parsed;
    } catch {
      setJsonError("JSON 파싱 오류: 올바른 JSON 배열을 입력하세요.");
      return null;
    }
  }

  function buildSimulatePayload() {
    const chunks = parseContextChunks();
    if (chunks === null) return null;
    return {
      query,
      ...(promptName ? { prompt_name: promptName } : {}),
      ...(chunks.length > 0 ? { context_chunks: chunks } : {}),
      ...(modelHint ? { model_hint: modelHint } : {}),
    };
  }

  async function handlePreview() {
    if (!query.trim()) return;
    const payload = buildSimulatePayload();
    if (!payload) return;
    setPlaygroundResult(null);
    simulateMutation.mutate(payload);
  }

  async function handleLLMTest() {
    if (!query.trim()) return;
    const payload = buildSimulatePayload();
    if (!payload) return;

    // First simulate to get assembled prompts
    try {
      const simRes = await adminApi.simulatePrompt(payload);
      const sim = simRes.data;
      setSimulateResult(sim);
      // Then call playground with assembled prompts
      playgroundMutation.mutate({
        prompt: sim.user_prompt,
        system_prompt: sim.system_prompt,
        ...(modelHint ? { model: modelHint } : {}),
      });
    } catch {
      simulateMutation.mutate(payload); // triggers error state
    }
  }

  const isSimulating = simulateMutation.isPending;
  const isTesting = playgroundMutation.isPending;
  const anyLoading = isSimulating || isTesting;

  const simulateError = simulateMutation.error as Error | null;
  const playgroundError = playgroundMutation.error as Error | null;

  return (
    <Box>
      <Typography variant="h6" fontWeight={600} mb={0.5}>
        프롬프트 시뮬레이터
      </Typography>
      <Typography variant="body2" color="text.secondary" mb={3}>
        프롬프트가 어떻게 조립되는지 미리보기 + LLM 테스트
      </Typography>

      <Card variant="outlined">
        <CardContent>
          {/* Top row: prompt selector + model hint */}
          <Grid container spacing={2} mb={2}>
            <Grid size={{ xs: 12, sm: 6 }}>
              <FormControl fullWidth size="small">
                <InputLabel>프롬프트 선택</InputLabel>
                <Select
                  value={promptName}
                  label="프롬프트 선택"
                  onChange={(e) => setPromptName(e.target.value)}
                  disabled={promptsLoading}
                >
                  <MenuItem value="">
                    <em>기본 (자동 감지)</em>
                  </MenuItem>
                  {promptKeys.map((k) => (
                    <MenuItem key={k} value={k}>
                      {k}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
            <Grid size={{ xs: 12, sm: 6 }}>
              <FormControl fullWidth size="small">
                <InputLabel>모델 힌트</InputLabel>
                <Select
                  value={modelHint}
                  label="모델 힌트"
                  onChange={(e) => setModelHint(e.target.value)}
                >
                  {MODEL_HINT_OPTIONS.map((opt) => (
                    <MenuItem key={opt.value} value={opt.value}>
                      {opt.label}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
          </Grid>

          {/* Query input */}
          <TextField
            fullWidth
            label="질문"
            placeholder="예: 연차휴가는 어떻게 부여되나요?"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            size="small"
            sx={{ mb: 2 }}
          />

          {/* Context chunks */}
          <TextField
            fullWidth
            label="컨텍스트 (선택)"
            placeholder={`[{"content": "참고 문서 내용...", "metadata": {"filename": "규정.hwp"}}]`}
            value={contextChunksJson}
            onChange={(e) => {
              setContextChunksJson(e.target.value);
              setJsonError(null);
            }}
            multiline
            rows={4}
            size="small"
            error={!!jsonError}
            helperText={
              jsonError ||
              "JSON 배열 형식. 비워두면 실제 RAG 검색 없이 빈 컨텍스트로 조립합니다."
            }
            sx={{ mb: 2 }}
          />

          {/* Action buttons */}
          <Stack direction="row" spacing={2}>
            <Button
              variant="outlined"
              startIcon={isSimulating ? <CircularProgress size={16} /> : <PreviewIcon />}
              onClick={handlePreview}
              disabled={!query.trim() || anyLoading}
            >
              미리보기
            </Button>
            <Button
              variant="contained"
              startIcon={isTesting ? <CircularProgress size={16} color="inherit" /> : <PlayArrowIcon />}
              onClick={handleLLMTest}
              disabled={!query.trim() || anyLoading}
            >
              LLM 테스트 실행
            </Button>
          </Stack>
        </CardContent>
      </Card>

      {/* Errors */}
      {simulateError && (
        <Alert severity="error" sx={{ mt: 2 }}>
          시뮬레이션 오류: {simulateError.message}
        </Alert>
      )}
      {playgroundError && (
        <Alert severity="error" sx={{ mt: 2 }}>
          LLM 테스트 오류: {playgroundError.message}
        </Alert>
      )}

      {/* Simulate results */}
      {simulateResult && (
        <Box mt={2}>
          <Divider sx={{ mb: 2 }} />

          {/* Meta info row */}
          <Stack direction="row" spacing={1} alignItems="center" mb={2} flexWrap="wrap">
            <Chip
              label={`문서 유형: ${DOC_TYPE_LABELS[simulateResult.detected_doc_type] ?? simulateResult.detected_doc_type}`}
              color="primary"
              size="small"
              variant="outlined"
            />
            <Chip
              label={`전체 ${simulateResult.prompt_length_chars.toLocaleString()}자`}
              size="small"
              variant="outlined"
            />
          </Stack>

          {/* System prompt accordion */}
          <Accordion defaultExpanded>
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Stack direction="row" spacing={1} alignItems="center">
                <Typography fontWeight={600}>시스템 프롬프트</Typography>
                <Chip
                  label={`${simulateResult.system_prompt_length.toLocaleString()}자`}
                  size="small"
                  color="secondary"
                />
              </Stack>
            </AccordionSummary>
            <AccordionDetails sx={{ p: 0 }}>
              <Box
                component="pre"
                sx={{
                  m: 0,
                  p: 2,
                  bgcolor: "grey.50",
                  borderTop: 1,
                  borderColor: "divider",
                  fontSize: "0.75rem",
                  fontFamily: "monospace",
                  whiteSpace: "pre-wrap",
                  wordBreak: "break-word",
                  maxHeight: 400,
                  overflowY: "auto",
                }}
              >
                {simulateResult.system_prompt}
              </Box>
            </AccordionDetails>
          </Accordion>

          {/* User prompt accordion */}
          <Accordion defaultExpanded sx={{ mt: 1 }}>
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Stack direction="row" spacing={1} alignItems="center">
                <Typography fontWeight={600}>유저 프롬프트</Typography>
                <Chip
                  label={`${simulateResult.user_prompt_length.toLocaleString()}자`}
                  size="small"
                  color="info"
                />
              </Stack>
            </AccordionSummary>
            <AccordionDetails sx={{ p: 0 }}>
              <Box
                component="pre"
                sx={{
                  m: 0,
                  p: 2,
                  bgcolor: "grey.50",
                  borderTop: 1,
                  borderColor: "divider",
                  fontSize: "0.75rem",
                  fontFamily: "monospace",
                  whiteSpace: "pre-wrap",
                  wordBreak: "break-word",
                  maxHeight: 300,
                  overflowY: "auto",
                }}
              >
                {simulateResult.user_prompt}
              </Box>
            </AccordionDetails>
          </Accordion>
        </Box>
      )}

      {/* Playground result */}
      {playgroundResult && (
        <Box mt={2}>
          <Divider sx={{ mb: 2 }} />
          <Accordion defaultExpanded>
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Stack direction="row" spacing={1} alignItems="center" flexWrap="wrap">
                <Typography fontWeight={600}>LLM 응답</Typography>
                <Chip
                  label={`${playgroundResult.latency_ms.toLocaleString()}ms`}
                  size="small"
                  color="success"
                  variant="outlined"
                />
                <Chip
                  label={playgroundResult.model_used}
                  size="small"
                  variant="outlined"
                />
                {playgroundResult.tokens_used != null && (
                  <Chip
                    label={`${playgroundResult.tokens_used} tokens`}
                    size="small"
                    variant="outlined"
                  />
                )}
              </Stack>
            </AccordionSummary>
            <AccordionDetails sx={{ p: 0 }}>
              <Box
                sx={{
                  m: 0,
                  p: 2,
                  bgcolor: "grey.50",
                  borderTop: 1,
                  borderColor: "divider",
                  fontSize: "0.875rem",
                  whiteSpace: "pre-wrap",
                  wordBreak: "break-word",
                  maxHeight: 400,
                  overflowY: "auto",
                }}
              >
                {playgroundResult.response}
              </Box>
            </AccordionDetails>
          </Accordion>
        </Box>
      )}

      {/* Loading indicator while testing */}
      {isTesting && !playgroundResult && (
        <Box mt={2} display="flex" alignItems="center" gap={1}>
          <CircularProgress size={20} />
          <Typography variant="body2" color="text.secondary">
            LLM 응답 대기 중...
          </Typography>
        </Box>
      )}
    </Box>
  );
}
