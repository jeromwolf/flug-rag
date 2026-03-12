import { useState } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
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
  Chip,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Tooltip,
  Slider,
  Divider,
  IconButton,
} from "@mui/material";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import SpeedIcon from "@mui/icons-material/Speed";
import ContentCopyIcon from "@mui/icons-material/ContentCopy";
import { adminApi } from "../../api/client";

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

export default LLMPlaygroundTab;
