import { useState, useEffect, useRef, useCallback } from "react";
import {
  Box,
  Paper,
  Typography,
  Chip,
  Divider,
  LinearProgress,
  Alert,
  Stack,
} from "@mui/material";
import {
  Speed as SpeedIcon,
  Psychology as PsychologyIcon,
  CheckCircle as CheckCircleIcon,
  RadioButtonUnchecked as PendingIcon,
} from "@mui/icons-material";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { API_BASE, getAuthHeaders } from "../../api/client";

interface ModelResult {
  content: string;
  confidence: number;
  latencyMs: number;
  done: boolean;
  error: string | null;
}

interface CompareViewProps {
  question: string;
  isActive: boolean;
  mainModel?: string;
  lightModel?: string;
}

const DEFAULT_MAIN_MODEL = "qwen2.5:14b";
const DEFAULT_LIGHT_MODEL = "qwen2.5:7b";

const MODEL_COLORS = {
  main: { primary: "#1565c0", bg: "#e3f2fd", chip: "primary" as const },
  light: { primary: "#2e7d32", bg: "#e8f5e9", chip: "success" as const },
};

async function streamModel(
  question: string,
  modelName: string,
  signal: AbortSignal,
  onChunk: (chunk: string) => void,
  onDone: (confidence: number, latencyMs: number) => void,
  onError: (msg: string) => void,
) {
  const body: Record<string, unknown> = {
    message: question,
    mode: "auto",
    temperature: 0.2,
  };

  // Only set model if it's not a generic label
  if (modelName && modelName !== "default") {
    body.model = modelName;
  }

  let response: Response;
  try {
    response = await fetch(`${API_BASE}/chat/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json", ...getAuthHeaders() },
      body: JSON.stringify(body),
      signal,
    });
  } catch (err: unknown) {
    if (err instanceof DOMException && err.name === "AbortError") return;
    onError("네트워크 오류가 발생했습니다.");
    return;
  }

  if (!response.ok) {
    onError(`서버 오류: HTTP ${response.status}`);
    return;
  }

  const reader = response.body!.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let currentEvent = "";
  let confidence = 0;
  let latencyMs = 0;

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";

      for (const line of lines) {
        if (line.startsWith("event:")) {
          currentEvent = line.slice(6).trim();
          continue;
        }
        if (line.startsWith("data:")) {
          const raw = line.slice(5).trim();
          if (!raw) continue;
          try {
            const data = JSON.parse(raw);
            switch (currentEvent) {
              case "chunk":
                onChunk(data.content ?? "");
                break;
              case "end":
                confidence = data.confidence_score ?? data.confidence ?? 0;
                latencyMs = data.latency_ms ?? 0;
                break;
              case "error":
                onError(data.message ?? "응답 생성 중 오류가 발생했습니다.");
                break;
            }
          } catch {
            // ignore malformed JSON
          }
        }
      }
    }
  } catch (err: unknown) {
    if (err instanceof DOMException && err.name === "AbortError") return;
    onError("스트리밍 중 오류가 발생했습니다.");
    return;
  }

  onDone(confidence, latencyMs);
}

function ConfidenceBar({ score }: { score: number }) {
  const pct = Math.round(score * 100);
  const color =
    score >= 0.8 ? "success" : score >= 0.5 ? "warning" : "error";
  return (
    <Box sx={{ mt: 0.5 }}>
      <Box sx={{ display: "flex", justifyContent: "space-between", mb: 0.25 }}>
        <Typography variant="caption" color="text.secondary">
          신뢰도
        </Typography>
        <Typography variant="caption" fontWeight={600}>
          {pct}%
        </Typography>
      </Box>
      <LinearProgress
        variant="determinate"
        value={pct}
        color={color}
        sx={{ height: 5, borderRadius: 3 }}
      />
    </Box>
  );
}

function ModelColumn({
  label,
  modelId,
  result,
  colorKey,
}: {
  label: string;
  modelId: string;
  result: ModelResult;
  colorKey: "main" | "light";
}) {
  const palette = MODEL_COLORS[colorKey];

  return (
    <Paper
      elevation={0}
      sx={{
        flex: 1,
        minWidth: 0,
        display: "flex",
        flexDirection: "column",
        border: "1px solid",
        borderColor: result.done && !result.error
          ? palette.primary + "40"
          : "divider",
        borderRadius: 2,
        overflow: "hidden",
        transition: "border-color 0.3s ease",
      }}
    >
      {/* Column header */}
      <Box
        sx={{
          display: "flex",
          alignItems: "center",
          gap: 1,
          px: 2,
          py: 1.25,
          bgcolor: result.done && !result.error ? palette.bg : "action.hover",
          borderBottom: "1px solid",
          borderColor: "divider",
          transition: "background-color 0.3s ease",
        }}
      >
        <Chip
          label={label}
          color={palette.chip}
          size="small"
          icon={
            result.done ? (
              result.error ? undefined : <CheckCircleIcon />
            ) : (
              <PendingIcon />
            )
          }
          sx={{ fontWeight: 700, fontSize: "0.7rem" }}
        />
        <Typography
          variant="caption"
          color="text.secondary"
          sx={{
            fontFamily: "monospace",
            fontSize: "0.7rem",
            overflow: "hidden",
            textOverflow: "ellipsis",
            whiteSpace: "nowrap",
          }}
        >
          {modelId}
        </Typography>
      </Box>

      {/* Streaming progress bar */}
      {!result.done && !result.error && (
        <LinearProgress
          sx={{ height: 2, borderRadius: 0 }}
          color={palette.chip}
        />
      )}

      {/* Content area */}
      <Box
        sx={{
          flex: 1,
          overflowY: "auto",
          px: 2.5,
          py: 2,
          "& p": { mt: 0, mb: 1, lineHeight: 1.7 },
          "& code": {
            bgcolor: "action.selected",
            px: 0.5,
            py: 0.25,
            borderRadius: 0.5,
            fontSize: "0.82em",
            fontFamily: "monospace",
          },
          "& pre": {
            bgcolor: "action.selected",
            p: 1.5,
            borderRadius: 1,
            overflowX: "auto",
            "& code": { bgcolor: "transparent", p: 0 },
          },
          "& ul, & ol": { pl: 2.5, mb: 1 },
          "& li": { mb: 0.25 },
          "& strong": { fontWeight: 700 },
          "& blockquote": {
            borderLeft: "3px solid",
            borderColor: palette.primary,
            pl: 1.5,
            ml: 0,
            color: "text.secondary",
          },
        }}
      >
        {result.error ? (
          <Alert severity="error" sx={{ mt: 1 }}>
            {result.error}
          </Alert>
        ) : result.content ? (
          <ReactMarkdown remarkPlugins={[remarkGfm]}>
            {result.content}
          </ReactMarkdown>
        ) : (
          <Typography
            variant="body2"
            color="text.disabled"
            sx={{ fontStyle: "italic" }}
          >
            응답 생성 중...
          </Typography>
        )}
      </Box>

      {/* Metadata footer */}
      {result.done && !result.error && (
        <Box
          sx={{
            px: 2.5,
            py: 1.25,
            borderTop: "1px solid",
            borderColor: "divider",
            bgcolor: "action.hover",
          }}
        >
          <ConfidenceBar score={result.confidence} />
          <Box
            sx={{
              display: "flex",
              alignItems: "center",
              gap: 0.5,
              mt: 0.75,
            }}
          >
            <SpeedIcon sx={{ fontSize: 13, color: "text.disabled" }} />
            <Typography variant="caption" color="text.secondary">
              응답시간:{" "}
              <strong>
                {result.latencyMs >= 1000
                  ? `${(result.latencyMs / 1000).toFixed(1)}s`
                  : `${result.latencyMs}ms`}
              </strong>
            </Typography>
          </Box>
        </Box>
      )}
    </Paper>
  );
}

function CompareSummary({
  mainResult,
  lightResult,
  mainModel,
  lightModel,
}: {
  mainResult: ModelResult;
  lightResult: ModelResult;
  mainModel: string;
  lightModel: string;
}) {
  const fasterModel =
    mainResult.latencyMs <= lightResult.latencyMs ? mainModel : lightModel;
  const fasterMs = Math.min(mainResult.latencyMs, lightResult.latencyMs);
  const slowerMs = Math.max(mainResult.latencyMs, lightResult.latencyMs);
  const speedDiff =
    slowerMs > 0
      ? `${((slowerMs - fasterMs) / 1000).toFixed(1)}s 빠름`
      : "";

  const higherConfidence =
    mainResult.confidence >= lightResult.confidence ? mainModel : lightModel;
  const confDiff = Math.abs(mainResult.confidence - lightResult.confidence);

  return (
    <Paper
      elevation={0}
      sx={{
        mt: 2,
        px: 3,
        py: 2,
        border: "1px solid",
        borderColor: "divider",
        borderRadius: 2,
        bgcolor: "action.hover",
      }}
    >
      <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 1.5 }}>
        <PsychologyIcon sx={{ fontSize: 18, color: "primary.main" }} />
        <Typography variant="subtitle2" fontWeight={700}>
          비교 요약
        </Typography>
      </Box>
      <Stack direction="row" spacing={3} flexWrap="wrap">
        <Box>
          <Typography variant="caption" color="text.secondary">
            응답 속도
          </Typography>
          <Typography variant="body2" fontWeight={600}>
            {fasterModel}{" "}
            <Typography
              component="span"
              variant="caption"
              color="success.main"
            >
              {speedDiff}
            </Typography>
          </Typography>
        </Box>
        <Divider orientation="vertical" flexItem />
        <Box>
          <Typography variant="caption" color="text.secondary">
            신뢰도
          </Typography>
          <Typography variant="body2" fontWeight={600}>
            {higherConfidence}{" "}
            <Typography
              component="span"
              variant="caption"
              color="info.main"
            >
              +{(confDiff * 100).toFixed(0)}%p
            </Typography>
          </Typography>
        </Box>
        <Divider orientation="vertical" flexItem />
        <Box>
          <Typography variant="caption" color="text.secondary">
            Main 신뢰도
          </Typography>
          <Typography variant="body2" fontWeight={600}>
            {(mainResult.confidence * 100).toFixed(1)}%
          </Typography>
        </Box>
        <Box>
          <Typography variant="caption" color="text.secondary">
            Light 신뢰도
          </Typography>
          <Typography variant="body2" fontWeight={600}>
            {(lightResult.confidence * 100).toFixed(1)}%
          </Typography>
        </Box>
      </Stack>
    </Paper>
  );
}

export function CompareView({
  question,
  isActive,
  mainModel = DEFAULT_MAIN_MODEL,
  lightModel = DEFAULT_LIGHT_MODEL,
}: CompareViewProps) {
  const [mainResult, setMainResult] = useState<ModelResult>({
    content: "",
    confidence: 0,
    latencyMs: 0,
    done: false,
    error: null,
  });
  const [lightResult, setLightResult] = useState<ModelResult>({
    content: "",
    confidence: 0,
    latencyMs: 0,
    done: false,
    error: null,
  });

  const abortRef = useRef<AbortController | null>(null);
  const lastQuestion = useRef<string>("");

  const runComparison = useCallback(async () => {
    if (!question || question === lastQuestion.current) return;
    lastQuestion.current = question;

    // Abort any in-flight requests
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    // Reset results
    setMainResult({ content: "", confidence: 0, latencyMs: 0, done: false, error: null });
    setLightResult({ content: "", confidence: 0, latencyMs: 0, done: false, error: null });

    const mainStream = streamModel(
      question,
      mainModel,
      controller.signal,
      (chunk) => setMainResult((prev) => ({ ...prev, content: prev.content + chunk })),
      (confidence, latencyMs) =>
        setMainResult((prev) => ({ ...prev, confidence, latencyMs, done: true })),
      (error) => setMainResult((prev) => ({ ...prev, error, done: true })),
    );

    const lightStream = streamModel(
      question,
      lightModel,
      controller.signal,
      (chunk) => setLightResult((prev) => ({ ...prev, content: prev.content + chunk })),
      (confidence, latencyMs) =>
        setLightResult((prev) => ({ ...prev, confidence, latencyMs, done: true })),
      (error) => setLightResult((prev) => ({ ...prev, error, done: true })),
    );

    await Promise.all([mainStream, lightStream]);
  }, [question, mainModel, lightModel]);

  useEffect(() => {
    if (isActive && question) {
      runComparison();
    }
    return () => {
      abortRef.current?.abort();
    };
  }, [isActive, question, runComparison]);

  const bothDone = mainResult.done && lightResult.done;
  const neitherHasError = !mainResult.error && !lightResult.error;

  if (!question) {
    return (
      <Box
        sx={{
          flex: 1,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          flexDirection: "column",
          gap: 1.5,
          opacity: 0.5,
        }}
      >
        <PsychologyIcon sx={{ fontSize: 48, color: "text.disabled" }} />
        <Typography color="text.secondary" variant="body2">
          질문을 입력하면 두 모델의 답변을 동시에 비교합니다.
        </Typography>
        <Stack direction="row" spacing={1}>
          <Chip label={mainModel} size="small" color="primary" variant="outlined" />
          <Typography variant="caption" color="text.disabled" alignSelf="center">
            vs
          </Typography>
          <Chip label={lightModel} size="small" color="success" variant="outlined" />
        </Stack>
      </Box>
    );
  }

  return (
    <Box
      sx={{
        flex: 1,
        display: "flex",
        flexDirection: "column",
        p: 2,
        gap: 1.5,
        overflowY: "auto",
        minHeight: 0,
      }}
    >
      {/* Question header */}
      <Paper
        elevation={0}
        sx={{
          px: 2.5,
          py: 1.5,
          borderRadius: 2,
          border: "1px solid",
          borderColor: "divider",
          bgcolor: "action.selected",
          flexShrink: 0,
        }}
      >
        <Typography variant="caption" color="text.secondary" fontWeight={600}>
          질문
        </Typography>
        <Typography variant="body2" sx={{ mt: 0.25, fontWeight: 500 }}>
          {question}
        </Typography>
      </Paper>

      {/* Two-column compare layout */}
      <Box
        sx={{
          display: "flex",
          gap: 1.5,
          flex: 1,
          minHeight: 0,
        }}
      >
        <ModelColumn
          label="Main Model"
          modelId={mainModel}
          result={mainResult}
          colorKey="main"
        />
        <ModelColumn
          label="Light Model"
          modelId={lightModel}
          result={lightResult}
          colorKey="light"
        />
      </Box>

      {/* Summary (only after both complete without errors) */}
      {bothDone && neitherHasError && (
        <CompareSummary
          mainResult={mainResult}
          lightResult={lightResult}
          mainModel={mainModel}
          lightModel={lightModel}
        />
      )}
    </Box>
  );
}
