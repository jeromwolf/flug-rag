import { useState, useCallback, useMemo, useEffect, useRef } from "react";
import {
  Box,
  Drawer,
  List,
  ListItemButton,
  ListItemText,
  ListItemIcon,
  Typography,
  Button,
  Paper,
  Chip,
  IconButton,
  Tooltip,
  Divider,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TablePagination,
  LinearProgress,
  CircularProgress,
  Snackbar,
  Alert,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogContentText,
  DialogActions,
  Collapse,
  TextField,
  InputAdornment,
  Stack,
  ToggleButtonGroup,
  ToggleButton,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Grid,
  Card,
  CardContent,
  CardActionArea,
  Checkbox,
} from "@mui/material";
import {
  Chat as ChatIcon,
  Description as DescriptionIcon,
  AdminPanelSettings as AdminIcon,
  Monitor as MonitorIcon,
  CloudUpload as UploadIcon,
  Delete as DeleteIcon,
  Download as DownloadIcon,
  InsertDriveFile as FileIcon,
  Close as CloseIcon,
  FolderOpen as FolderOpenIcon,
  CheckCircle as CheckCircleIcon,
  Error as ErrorIcon,
  HourglassEmpty as PendingIcon,
  ExpandMore as ExpandMoreIcon,
  ExpandLess as ExpandLessIcon,
  Search as SearchIcon,
  ViewList as ViewListIcon,
  ViewModule as ViewModuleIcon,
  ArrowUpward as ArrowUpIcon,
  ArrowDownward as ArrowDownIcon,
  Refresh as RefreshIcon,
} from "@mui/icons-material";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { useDropzone } from "react-dropzone";
import { documentsApi } from "../api/client";
import { useAppStore } from "../stores/appStore";

// ---------------------------------------------------------------------------
// IngestProgressTracker — types, constants, and component
// ---------------------------------------------------------------------------

type IngestStepId = "upload" | "ocr" | "chunk" | "embed" | "done";
type StepState = "pending" | "active" | "complete" | "error";

interface IngestStep {
  id: IngestStepId;
  emoji: string;
  label: string;
  description: string;
  durationMs: number;
}

const INGEST_STEPS: IngestStep[] = [
  { id: "upload", emoji: "☁️", label: "업로드 중",      description: "서버로 파일 전송 중...",          durationMs: 1400 },
  { id: "ocr",    emoji: "📄", label: "OCR 처리 중",    description: "문서 텍스트 추출 중...",           durationMs: 2000 },
  { id: "chunk",  emoji: "✂️", label: "청킹 중",        description: "문서를 청크로 분할 중...",         durationMs: 1600 },
  { id: "embed",  emoji: "🧮", label: "임베딩 생성 중", description: "벡터 임베딩 계산 중...",           durationMs: 1800 },
  { id: "done",   emoji: "✅", label: "완료",            description: "인덱싱이 완료되었습니다.",          durationMs: 0    },
];

const STEP_INDICES: Record<IngestStepId, number> = {
  upload: 0,
  ocr:    1,
  chunk:  2,
  embed:  3,
  done:   4,
};

// 5% error probability per step (for demo purposes)
const ERROR_CHANCE = 0.05;

interface FileIngestState {
  fileName: string;
  currentStep: IngestStepId;
  stepStates: Record<IngestStepId, StepState>;
  progress: number;          // 0–100
  errorStep: IngestStepId | null;
  isComplete: boolean;
}

function makeInitialState(fileName: string): FileIngestState {
  return {
    fileName,
    currentStep: "upload",
    stepStates: {
      upload: "active",
      ocr:    "pending",
      chunk:  "pending",
      embed:  "pending",
      done:   "pending",
    },
    progress: 0,
    errorStep: null,
    isComplete: false,
  };
}

function makeRetryState(prev: FileIngestState): FileIngestState {
  if (!prev.errorStep) return prev;
  // Reset errorStep and everything after it back to pending/active
  const errorIdx = STEP_INDICES[prev.errorStep];
  const next: FileIngestState = {
    ...prev,
    errorStep: null,
    isComplete: false,
    currentStep: prev.errorStep,
    stepStates: { ...prev.stepStates },
    progress: Math.round((errorIdx / (INGEST_STEPS.length - 1)) * 100),
  };
  for (let i = errorIdx; i < INGEST_STEPS.length; i++) {
    const sid = INGEST_STEPS[i].id;
    next.stepStates[sid] = i === errorIdx ? "active" : "pending";
  }
  return next;
}

// ---------------------------------------------------------------------------
// IngestProgressTracker Dialog component
// ---------------------------------------------------------------------------

interface IngestProgressTrackerProps {
  open: boolean;
  files: File[];
  onClose: () => void;
  /** Called when all files (including retries) have been fully ingested */
  onAllComplete?: () => void;
}

function IngestProgressTracker({ open, files, onClose, onAllComplete }: IngestProgressTrackerProps) {
  const [states, setStates] = useState<FileIngestState[]>([]);
  const [activeIndex, setActiveIndex] = useState(0);
  const [allDone, setAllDone] = useState(false);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Reset state whenever `files` prop changes (new upload batch)
  useEffect(() => {
    if (!open || files.length === 0) return;
    setStates(files.map((f) => makeInitialState(f.name)));
    setActiveIndex(0);
    setAllDone(false);
  }, [open, files]);

  // Advance a single file's ingest simulation
  const advanceFile = useCallback((fileIdx: number, fromStep: IngestStepId) => {
    const stepIdx = STEP_INDICES[fromStep];
    const step = INGEST_STEPS[stepIdx];

    if (!step || step.id === "done") return;

    // Simulate error with 5% chance (skip on "done" step)
    const willError = Math.random() < ERROR_CHANCE;

    timerRef.current = setTimeout(() => {
      if (willError) {
        setStates((prev) => {
          const next = [...prev];
          const fs = { ...next[fileIdx] };
          fs.stepStates = { ...fs.stepStates, [fromStep]: "error" };
          fs.errorStep = fromStep;
          next[fileIdx] = fs;
          return next;
        });
        return; // stop advancing — user must retry
      }

      // Mark current step complete, advance to next
      const nextStepIdx = stepIdx + 1;
      const nextStep = INGEST_STEPS[nextStepIdx];

      setStates((prev) => {
        const next = [...prev];
        const fs = { ...next[fileIdx] };
        fs.stepStates = {
          ...fs.stepStates,
          [fromStep]: "complete",
          [nextStep.id]: nextStep.id === "done" ? "complete" : "active",
        };
        fs.currentStep = nextStep.id;
        fs.progress = Math.round((nextStepIdx / (INGEST_STEPS.length - 1)) * 100);
        fs.isComplete = nextStep.id === "done";
        next[fileIdx] = fs;
        return next;
      });

      if (nextStep.id !== "done") {
        advanceFile(fileIdx, nextStep.id);
      }
    }, step.durationMs + Math.random() * 400);
  }, []);

  // Kick off simulation when states are initialized
  useEffect(() => {
    if (states.length === 0) return;
    // Only start the first file automatically; others start after
    // previous completes or after a stagger delay
    states.forEach((fs, idx) => {
      if (fs.currentStep === "upload" && fs.stepStates.upload === "active") {
        setTimeout(() => advanceFile(idx, "upload"), idx * 300);
      }
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [states.length > 0 ? states[0]?.fileName : null]);

  // Watch for all-done
  useEffect(() => {
    if (states.length === 0) return;
    const done = states.every((s) => s.isComplete);
    if (done && !allDone) {
      setAllDone(true);
      onAllComplete?.();
      timerRef.current = setTimeout(() => {
        onClose();
      }, 2000);
    }
  }, [states, allDone, onAllComplete, onClose]);

  // Cleanup timers on unmount
  useEffect(() => () => { if (timerRef.current) clearTimeout(timerRef.current); }, []);

  const handleRetry = (fileIdx: number) => {
    setStates((prev) => {
      const next = [...prev];
      const retried = makeRetryState(next[fileIdx]);
      next[fileIdx] = retried;
      return next;
    });
    // Re-start simulation from the retried step
    const retried = makeRetryState(states[fileIdx]);
    advanceFile(fileIdx, retried.currentStep);
  };

  // Display the currently "active" file (the one being viewed in the tracker)
  const displayIdx = Math.min(activeIndex, states.length - 1);
  const displayState = states[displayIdx] ?? null;

  // Count how many are complete vs errored vs in progress
  const completeCount = states.filter((s) => s.isComplete).length;
  const errorCount = states.filter((s) => s.errorStep !== null).length;

  return (
    <Dialog
      open={open}
      onClose={allDone ? onClose : undefined}
      maxWidth="sm"
      fullWidth
      PaperProps={{
        sx: {
          background: "linear-gradient(160deg, #0f1923 0%, #141e2b 100%)",
          border: "1px solid rgba(255,255,255,0.08)",
          borderRadius: 3,
          overflow: "hidden",
        },
      }}
    >
      {/* Header */}
      <DialogTitle
        sx={{
          px: 3,
          py: 2,
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          borderBottom: "1px solid rgba(255,255,255,0.06)",
        }}
      >
        <Box sx={{ display: "flex", alignItems: "center", gap: 1.5 }}>
          <Box
            sx={{
              width: 8,
              height: 8,
              borderRadius: "50%",
              bgcolor: allDone ? "#10a37f" : errorCount > 0 ? "#ef5350" : "#ffb300",
              boxShadow: allDone
                ? "0 0 10px #10a37f"
                : errorCount > 0
                ? "0 0 10px #ef5350"
                : "0 0 10px #ffb300",
              animation: !allDone && errorCount === 0 ? "pulseGlow 1.5s infinite" : "none",
              "@keyframes pulseGlow": {
                "0%, 100%": { opacity: 1, transform: "scale(1)" },
                "50%": { opacity: 0.5, transform: "scale(1.5)" },
              },
            }}
          />
          <Typography
            variant="subtitle1"
            fontWeight={700}
            sx={{
              color: "#e8edf2",
              fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
              letterSpacing: "0.02em",
            }}
          >
            문서 인제스트 파이프라인
          </Typography>
        </Box>

        <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
          <Chip
            size="small"
            label={`${completeCount} / ${states.length} 완료`}
            sx={{
              bgcolor: "rgba(16,163,127,0.15)",
              color: "#10a37f",
              fontFamily: "'JetBrains Mono', monospace",
              fontSize: "0.72rem",
              fontWeight: 700,
              border: "1px solid rgba(16,163,127,0.3)",
            }}
          />
          {allDone && (
            <IconButton size="small" onClick={onClose} sx={{ color: "rgba(255,255,255,0.5)" }}>
              <CloseIcon fontSize="small" />
            </IconButton>
          )}
        </Box>
      </DialogTitle>

      <DialogContent sx={{ p: 0 }}>
        {/* File selector tabs (only shown when >1 file) */}
        {states.length > 1 && (
          <Box
            sx={{
              display: "flex",
              gap: 0.5,
              px: 2.5,
              pt: 2,
              pb: 0,
              flexWrap: "wrap",
            }}
          >
            {states.map((fs, idx) => (
              <Chip
                key={idx}
                label={fs.fileName.length > 20 ? `${fs.fileName.slice(0, 18)}…` : fs.fileName}
                size="small"
                onClick={() => setActiveIndex(idx)}
                sx={{
                  cursor: "pointer",
                  fontFamily: "'JetBrains Mono', monospace",
                  fontSize: "0.68rem",
                  bgcolor:
                    idx === displayIdx
                      ? "rgba(16,163,127,0.18)"
                      : "rgba(255,255,255,0.05)",
                  color:
                    idx === displayIdx
                      ? "#10a37f"
                      : fs.errorStep
                      ? "#ef5350"
                      : fs.isComplete
                      ? "rgba(255,255,255,0.7)"
                      : "rgba(255,255,255,0.45)",
                  border: `1px solid ${
                    idx === displayIdx
                      ? "rgba(16,163,127,0.4)"
                      : fs.errorStep
                      ? "rgba(239,83,80,0.3)"
                      : "rgba(255,255,255,0.08)"
                  }`,
                  transition: "all 0.2s",
                  "&:hover": { bgcolor: "rgba(16,163,127,0.12)" },
                }}
              />
            ))}
          </Box>
        )}

        {displayState && (
          <Box sx={{ px: 3, pt: states.length > 1 ? 2 : 3, pb: 3 }}>
            {/* File name display */}
            <Box
              sx={{
                display: "flex",
                alignItems: "center",
                gap: 1,
                mb: 2.5,
                p: 1.5,
                borderRadius: 2,
                bgcolor: "rgba(255,255,255,0.04)",
                border: "1px solid rgba(255,255,255,0.06)",
              }}
            >
              <FileIcon sx={{ fontSize: 16, color: "#64b5f6", flexShrink: 0 }} />
              <Typography
                variant="body2"
                sx={{
                  color: "#b0bec5",
                  fontFamily: "'JetBrains Mono', monospace",
                  fontSize: "0.8rem",
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                  whiteSpace: "nowrap",
                  flex: 1,
                }}
              >
                {displayState.fileName}
              </Typography>
            </Box>

            {/* Master progress bar */}
            <Box sx={{ mb: 3 }}>
              <Box sx={{ display: "flex", justifyContent: "space-between", mb: 0.75 }}>
                <Typography
                  variant="caption"
                  sx={{
                    color: "rgba(255,255,255,0.4)",
                    fontFamily: "'JetBrains Mono', monospace",
                    fontSize: "0.7rem",
                    textTransform: "uppercase",
                    letterSpacing: "0.06em",
                  }}
                >
                  전체 진행률
                </Typography>
                <Typography
                  variant="caption"
                  sx={{
                    color: displayState.errorStep
                      ? "#ef5350"
                      : displayState.isComplete
                      ? "#10a37f"
                      : "#ffb300",
                    fontFamily: "'JetBrains Mono', monospace",
                    fontSize: "0.72rem",
                    fontWeight: 700,
                  }}
                >
                  {displayState.progress}%
                </Typography>
              </Box>
              <Box
                sx={{
                  height: 6,
                  borderRadius: 3,
                  bgcolor: "rgba(255,255,255,0.06)",
                  overflow: "hidden",
                  position: "relative",
                }}
              >
                <Box
                  sx={{
                    position: "absolute",
                    left: 0,
                    top: 0,
                    height: "100%",
                    width: `${displayState.progress}%`,
                    borderRadius: 3,
                    bgcolor: displayState.errorStep
                      ? "#ef5350"
                      : displayState.isComplete
                      ? "#10a37f"
                      : "#ffb300",
                    transition: "width 0.5s ease, background-color 0.3s",
                    boxShadow: displayState.errorStep
                      ? "0 0 8px rgba(239,83,80,0.5)"
                      : displayState.isComplete
                      ? "0 0 8px rgba(16,163,127,0.5)"
                      : "0 0 8px rgba(255,179,0,0.5)",
                  }}
                />
              </Box>
            </Box>

            {/* Step pipeline */}
            <Box sx={{ display: "flex", flexDirection: "column", gap: 0 }}>
              {INGEST_STEPS.map((step, idx) => {
                const state = displayState.stepStates[step.id];
                const isLast = idx === INGEST_STEPS.length - 1;

                const nodeColor =
                  state === "complete" ? "#10a37f"
                  : state === "active"  ? "#ffb300"
                  : state === "error"   ? "#ef5350"
                  : "rgba(255,255,255,0.12)";

                const labelColor =
                  state === "complete" ? "#b2dfdb"
                  : state === "active"  ? "#ffe082"
                  : state === "error"   ? "#ef9a9a"
                  : "rgba(255,255,255,0.28)";

                const descColor =
                  state === "active"  ? "rgba(255,224,130,0.6)"
                  : state === "complete" ? "rgba(178,223,219,0.5)"
                  : state === "error"   ? "rgba(239,154,154,0.6)"
                  : "rgba(255,255,255,0.18)";

                return (
                  <Box key={step.id} sx={{ display: "flex", alignItems: "stretch", gap: 0 }}>
                    {/* Connector column */}
                    <Box
                      sx={{
                        display: "flex",
                        flexDirection: "column",
                        alignItems: "center",
                        width: 32,
                        flexShrink: 0,
                      }}
                    >
                      {/* Node dot */}
                      <Box
                        sx={{
                          width: 14,
                          height: 14,
                          borderRadius: "50%",
                          bgcolor:
                            state === "active" || state === "error"
                              ? "transparent"
                              : nodeColor,
                          border: `2px solid ${nodeColor}`,
                          flexShrink: 0,
                          mt: "13px",
                          zIndex: 1,
                          position: "relative",
                          boxShadow:
                            state === "complete"
                              ? "0 0 6px rgba(16,163,127,0.6)"
                              : state === "active"
                              ? "0 0 8px rgba(255,179,0,0.7)"
                              : state === "error"
                              ? "0 0 8px rgba(239,83,80,0.7)"
                              : "none",
                          // Spinning ring for active
                          "&::before":
                            state === "active"
                              ? {
                                  content: '""',
                                  position: "absolute",
                                  inset: -4,
                                  borderRadius: "50%",
                                  border: "2px solid transparent",
                                  borderTopColor: "#ffb300",
                                  animation: "spinRing 0.9s linear infinite",
                                }
                              : {},
                          "@keyframes spinRing": {
                            from: { transform: "rotate(0deg)" },
                            to: { transform: "rotate(360deg)" },
                          },
                        }}
                      >
                        {/* Inner fill for active state */}
                        {state === "active" && (
                          <Box
                            sx={{
                              position: "absolute",
                              inset: 2,
                              borderRadius: "50%",
                              bgcolor: "#ffb300",
                              animation: "pulseInner 1.2s ease-in-out infinite",
                              "@keyframes pulseInner": {
                                "0%, 100%": { opacity: 1 },
                                "50%": { opacity: 0.4 },
                              },
                            }}
                          />
                        )}
                      </Box>
                      {/* Connector line below */}
                      {!isLast && (
                        <Box
                          sx={{
                            width: 2,
                            flex: 1,
                            minHeight: 20,
                            bgcolor:
                              state === "complete"
                                ? "rgba(16,163,127,0.4)"
                                : "rgba(255,255,255,0.07)",
                            transition: "background-color 0.4s",
                          }}
                        />
                      )}
                    </Box>

                    {/* Step content */}
                    <Box
                      sx={{
                        flex: 1,
                        py: 1.25,
                        pb: isLast ? 0 : 1.25,
                        pl: 1.5,
                        display: "flex",
                        alignItems: "flex-start",
                        justifyContent: "space-between",
                        gap: 1,
                      }}
                    >
                      <Box>
                        <Box sx={{ display: "flex", alignItems: "center", gap: 0.75, mb: 0.25 }}>
                          <Typography
                            component="span"
                            sx={{ fontSize: "1rem", lineHeight: 1, userSelect: "none" }}
                          >
                            {step.emoji}
                          </Typography>
                          <Typography
                            variant="body2"
                            sx={{
                              color: labelColor,
                              fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
                              fontSize: "0.82rem",
                              fontWeight: state === "active" ? 700 : 500,
                              transition: "color 0.3s",
                            }}
                          >
                            {step.label}
                          </Typography>
                          {state === "active" && (
                            <CircularProgress
                              size={10}
                              thickness={5}
                              sx={{ color: "#ffb300", ml: 0.5 }}
                            />
                          )}
                        </Box>
                        {(state === "active" || state === "error") && (
                          <Typography
                            variant="caption"
                            sx={{
                              color: descColor,
                              fontFamily: "'JetBrains Mono', monospace",
                              fontSize: "0.68rem",
                              display: "block",
                            }}
                          >
                            {state === "error" ? "처리 중 오류가 발생했습니다." : step.description}
                          </Typography>
                        )}
                      </Box>

                      {/* Right-side state indicator */}
                      <Box sx={{ flexShrink: 0, pt: 0.5 }}>
                        {state === "complete" && (
                          <CheckCircleIcon sx={{ fontSize: 18, color: "#10a37f" }} />
                        )}
                        {state === "error" && (
                          <Box sx={{ display: "flex", alignItems: "center", gap: 0.5 }}>
                            <ErrorIcon sx={{ fontSize: 18, color: "#ef5350" }} />
                            <Button
                              size="small"
                              startIcon={<RefreshIcon sx={{ fontSize: 13 }} />}
                              onClick={() => handleRetry(displayIdx)}
                              sx={{
                                color: "#ef9a9a",
                                fontSize: "0.7rem",
                                fontFamily: "'JetBrains Mono', monospace",
                                fontWeight: 700,
                                textTransform: "none",
                                bgcolor: "rgba(239,83,80,0.1)",
                                border: "1px solid rgba(239,83,80,0.3)",
                                borderRadius: 1,
                                px: 1,
                                py: 0.25,
                                minWidth: 0,
                                lineHeight: 1.4,
                                "&:hover": {
                                  bgcolor: "rgba(239,83,80,0.2)",
                                  borderColor: "rgba(239,83,80,0.5)",
                                },
                              }}
                            >
                              재시도
                            </Button>
                          </Box>
                        )}
                        {state === "pending" && (
                          <PendingIcon sx={{ fontSize: 16, color: "rgba(255,255,255,0.15)" }} />
                        )}
                      </Box>
                    </Box>
                  </Box>
                );
              })}
            </Box>

            {/* All-done banner */}
            {displayState.isComplete && (
              <Box
                sx={{
                  mt: 3,
                  p: 1.5,
                  borderRadius: 2,
                  bgcolor: "rgba(16,163,127,0.1)",
                  border: "1px solid rgba(16,163,127,0.3)",
                  display: "flex",
                  alignItems: "center",
                  gap: 1,
                  animation: "fadeInBanner 0.4s ease",
                  "@keyframes fadeInBanner": {
                    from: { opacity: 0, transform: "translateY(6px)" },
                    to: { opacity: 1, transform: "translateY(0)" },
                  },
                }}
              >
                <CheckCircleIcon sx={{ fontSize: 18, color: "#10a37f" }} />
                <Typography
                  variant="body2"
                  sx={{
                    color: "#80cbc4",
                    fontFamily: "'JetBrains Mono', monospace",
                    fontSize: "0.78rem",
                    fontWeight: 600,
                  }}
                >
                  문서가 지식베이스에 성공적으로 등록되었습니다.
                </Typography>
              </Box>
            )}
          </Box>
        )}
      </DialogContent>

      {/* Footer */}
      <Box
        sx={{
          px: 3,
          py: 1.5,
          borderTop: "1px solid rgba(255,255,255,0.06)",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
        }}
      >
        <Typography
          variant="caption"
          sx={{
            color: "rgba(255,255,255,0.22)",
            fontFamily: "'JetBrains Mono', monospace",
            fontSize: "0.68rem",
          }}
        >
          {allDone ? "자동으로 창이 닫힙니다..." : "처리가 완료될 때까지 기다려 주세요"}
        </Typography>
        {(allDone || errorCount > 0) && (
          <Button
            size="small"
            onClick={onClose}
            sx={{
              color: "rgba(255,255,255,0.5)",
              fontSize: "0.72rem",
              fontFamily: "'JetBrains Mono', monospace",
              textTransform: "none",
              "&:hover": { color: "rgba(255,255,255,0.8)" },
            }}
          >
            닫기
          </Button>
        )}
      </Box>
    </Dialog>
  );
}

// ---------------------------------------------------------------------------
// Main page constants
// ---------------------------------------------------------------------------

const SIDEBAR_WIDTH = 280;
const ROWS_PER_PAGE = 20;
const CARDS_PER_PAGE = 24;

// Hidden/system filenames that should not be shown
const HIDDEN_PATTERNS = [
  /^\.DS_Store$/i,
  /^__MACOSX/,
  /^\.gitkeep$/i,
  /^\.gitignore$/i,
  /^thumbs\.db$/i,
  /^\./,           // any dotfile
];

function isHiddenFile(filename: string): boolean {
  return HIDDEN_PATTERNS.some((re) => re.test(filename));
}

interface UploadQueueItem {
  file: File;
  status: "pending" | "uploading" | "done" | "error";
  errorMessage?: string;
}

const ACCEPTED_TYPES: Record<string, string[]> = {
  "application/pdf": [".pdf"],
  "application/vnd.hancom.hwp": [".hwp"],
  "application/vnd.openxmlformats-officedocument.wordprocessingml.document": [
    ".docx",
  ],
  "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": [
    ".xlsx",
  ],
  "application/vnd.openxmlformats-officedocument.presentationml.presentation": [
    ".pptx",
  ],
  "text/plain": [".txt"],
};

const FILE_TYPE_FILTERS = ["전체", "PDF", "HWP", "DOCX", "XLSX", "PPTX", "TXT", "기타"];
const STATUS_FILTERS = ["전체", "완료", "처리중", "실패", "대기"];

type SortField = "date" | "name" | "size" | "chunks";
type SortDir = "asc" | "desc";
type ViewMode = "list" | "card";

// File type colors (matching SourcesPanel convention)
const FILE_TYPE_COLORS: Record<string, string> = {
  PDF:  "#e53935",
  HWP:  "#1e88e5",
  DOCX: "#43a047",
  XLSX: "#00897b",
  PPTX: "#f4511e",
  TXT:  "#8e24aa",
};

function getFileTypeColor(ext: string): string {
  return FILE_TYPE_COLORS[ext.toUpperCase()] ?? "#757575";
}

function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatDate(dateStr: string): string {
  const d = new Date(dateStr);
  return d.toLocaleDateString("ko-KR", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

interface DocumentRow {
  id: string;
  filename: string;
  file_type?: string;
  fileType?: string;
  file_size?: number;
  fileSize?: number;
  chunk_count?: number;
  chunkCount?: number;
  uploaded_at?: string;
  uploadDate?: string;
  status?: string;
  metadata?: Record<string, unknown>;
}

function getFileExt(doc: DocumentRow): string {
  const ft = doc.file_type ?? doc.fileType;
  if (ft) return ft.toUpperCase();
  const parts = doc.filename.split(".");
  return parts.length > 1 ? parts.pop()!.toUpperCase() : "";
}

function normalizeStatus(status: string | undefined): string {
  switch (status) {
    case "completed": return "완료";
    case "processing": return "처리중";
    case "failed":    return "실패";
    case "pending":   return "대기";
    default:          return "완료";
  }
}

function getStatusChip(status: string | undefined) {
  switch (status) {
    case "completed":
      return <Chip label="완료" color="success" size="small" />;
    case "processing":
      return <Chip label="처리 중" color="info" size="small" />;
    case "failed":
      return <Chip label="실패" color="error" size="small" />;
    case "pending":
      return <Chip label="대기" color="default" size="small" />;
    default:
      return <Chip label="완료" color="success" size="small" />;
  }
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export default function DocumentsPage() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const { sidebarOpen } = useAppStore();

  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadQueue, setUploadQueue] = useState<UploadQueueItem[]>([]);
  const [queueExpanded, setQueueExpanded] = useState(true);

  // Ingest progress tracker
  const [ingestTrackerOpen, setIngestTrackerOpen] = useState(false);
  const [ingestFiles, setIngestFiles] = useState<File[]>([]);
  const [deleteDialogId, setDeleteDialogId] = useState<string | null>(null);
  const [bulkDeleteDialogOpen, setBulkDeleteDialogOpen] = useState(false);
  const [detailDoc, setDetailDoc] = useState<DocumentRow | null>(null);
  const [snackbar, setSnackbar] = useState<{
    open: boolean;
    message: string;
    severity: "success" | "error" | "info";
  }>({ open: false, message: "", severity: "info" });

  // Filter / search state
  const [searchText, setSearchText] = useState("");
  const [typeFilter, setTypeFilter] = useState("전체");
  const [statusFilter, setStatusFilter] = useState("전체");

  // Sorting state
  const [sortField, setSortField] = useState<SortField>("date");
  const [sortDir, setSortDir] = useState<SortDir>("desc");

  // View mode
  const [viewMode, setViewMode] = useState<ViewMode>("list");

  // Bulk selection
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());

  // Pagination state
  const [page, setPage] = useState(0);

  // -----------------------------------------------------------------------
  // Queries
  // -----------------------------------------------------------------------

  const { data: docsData, isLoading } = useQuery({
    queryKey: ["documents"],
    queryFn: async () => {
      const res = await documentsApi.list();
      return res.data;
    },
  });

  const rawDocuments: DocumentRow[] = docsData?.documents ?? [];

  // Frontend system-file filter
  const documents: DocumentRow[] = useMemo(
    () => rawDocuments.filter((d) => !isHiddenFile(d.filename)),
    [rawDocuments]
  );

  // Apply search + type + status filters then sort
  const filteredDocuments: DocumentRow[] = useMemo(() => {
    const filtered = documents.filter((doc) => {
      const matchesSearch =
        searchText.trim() === "" ||
        doc.filename.toLowerCase().includes(searchText.trim().toLowerCase());

      const ext = getFileExt(doc);
      const matchesType =
        typeFilter === "전체" ||
        (typeFilter === "기타"
          ? !["PDF", "HWP", "DOCX", "XLSX", "PPTX", "TXT"].includes(ext)
          : ext === typeFilter);

      const docStatusLabel = normalizeStatus(doc.status);
      const matchesStatus =
        statusFilter === "전체" || docStatusLabel === statusFilter;

      return matchesSearch && matchesType && matchesStatus;
    });

    // Sort
    filtered.sort((a, b) => {
      let cmp = 0;
      if (sortField === "date") {
        const da = new Date(a.uploaded_at ?? a.uploadDate ?? 0).getTime();
        const db = new Date(b.uploaded_at ?? b.uploadDate ?? 0).getTime();
        cmp = da - db;
      } else if (sortField === "name") {
        cmp = a.filename.localeCompare(b.filename, "ko");
      } else if (sortField === "size") {
        const sa = a.file_size ?? a.fileSize ?? 0;
        const sb = b.file_size ?? b.fileSize ?? 0;
        cmp = sa - sb;
      } else if (sortField === "chunks") {
        const ca = a.chunk_count ?? a.chunkCount ?? 0;
        const cb = b.chunk_count ?? b.chunkCount ?? 0;
        cmp = ca - cb;
      }
      return sortDir === "asc" ? cmp : -cmp;
    });

    return filtered;
  }, [documents, searchText, typeFilter, statusFilter, sortField, sortDir]);

  // Summary stats (over all visible docs, not just current page)
  const totalCount = filteredDocuments.length;
  const totalSize = useMemo(
    () => filteredDocuments.reduce((sum, d) => sum + (d.file_size ?? d.fileSize ?? 0), 0),
    [filteredDocuments]
  );

  const rowsPerPage = viewMode === "list" ? ROWS_PER_PAGE : CARDS_PER_PAGE;

  // Paginated slice
  const pagedDocuments = useMemo(
    () => filteredDocuments.slice(page * rowsPerPage, (page + 1) * rowsPerPage),
    [filteredDocuments, page, rowsPerPage]
  );

  // Reset page when filters / view changes
  const handleSearchChange = (val: string) => {
    setSearchText(val);
    setPage(0);
  };

  const handleTypeFilter = (type: string) => {
    setTypeFilter(type);
    setPage(0);
    setSelectedIds(new Set());
  };

  const handleStatusFilter = (status: string) => {
    setStatusFilter(status);
    setPage(0);
    setSelectedIds(new Set());
  };

  const handleSortFieldChange = (field: SortField) => {
    if (sortField === field) {
      setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    } else {
      setSortField(field);
      setSortDir("desc");
    }
    setPage(0);
  };

  const handleViewModeChange = (_: React.MouseEvent, val: ViewMode | null) => {
    if (val) {
      setViewMode(val);
      setPage(0);
      setSelectedIds(new Set());
    }
  };

  // -----------------------------------------------------------------------
  // Selection helpers
  // -----------------------------------------------------------------------

  const toggleSelect = (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const toggleSelectAll = () => {
    if (selectedIds.size === pagedDocuments.length) {
      setSelectedIds(new Set());
    } else {
      setSelectedIds(new Set(pagedDocuments.map((d) => d.id)));
    }
  };

  const clearSelection = () => setSelectedIds(new Set());

  // -----------------------------------------------------------------------
  // Mutations
  // -----------------------------------------------------------------------

  const deleteMutation = useMutation({
    mutationFn: (id: string) => documentsApi.delete(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["documents"] });
      setDeleteDialogId(null);
      setSnackbar({
        open: true,
        message: "문서가 삭제되었습니다.",
        severity: "success",
      });
    },
    onError: () => {
      setSnackbar({
        open: true,
        message: "문서 삭제에 실패했습니다.",
        severity: "error",
      });
    },
  });

  const bulkDeleteMutation = useMutation({
    mutationFn: async (ids: string[]) => {
      for (const id of ids) {
        await documentsApi.delete(id);
      }
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["documents"] });
      setBulkDeleteDialogOpen(false);
      setSnackbar({
        open: true,
        message: `${selectedIds.size}개 문서가 삭제되었습니다.`,
        severity: "success",
      });
      setSelectedIds(new Set());
    },
    onError: () => {
      setSnackbar({
        open: true,
        message: "일부 문서 삭제에 실패했습니다.",
        severity: "error",
      });
    },
  });

  // -----------------------------------------------------------------------
  // Upload
  // -----------------------------------------------------------------------

  const handleUpload = useCallback(
    async (files: File[]) => {
      if (files.length === 0) return;

      // Open the ingest progress tracker dialog
      setIngestFiles(files);
      setIngestTrackerOpen(true);

      const initialQueue: UploadQueueItem[] = files.map((f) => ({
        file: f,
        status: "pending",
      }));
      setUploadQueue(initialQueue);
      setQueueExpanded(true);
      setUploading(true);
      setUploadProgress(0);

      let completed = 0;
      let failed = 0;

      for (let i = 0; i < files.length; i++) {
        setUploadQueue((prev) =>
          prev.map((item, idx) =>
            idx === i ? { ...item, status: "uploading" } : item
          )
        );

        try {
          await documentsApi.upload(files[i]);
          completed++;
          setUploadQueue((prev) =>
            prev.map((item, idx) =>
              idx === i ? { ...item, status: "done" } : item
            )
          );
        } catch (err: unknown) {
          failed++;
          const errorMessage =
            err instanceof Error ? err.message : "업로드 실패";
          setUploadQueue((prev) =>
            prev.map((item, idx) =>
              idx === i ? { ...item, status: "error", errorMessage } : item
            )
          );
        }

        setUploadProgress(
          Math.round(((completed + failed) / files.length) * 100)
        );
      }

      setUploading(false);
      queryClient.invalidateQueries({ queryKey: ["documents"] });

      if (failed > 0) {
        setSnackbar({
          open: true,
          message: `${completed}개 업로드 완료, ${failed}개 실패`,
          severity: failed === files.length ? "error" : "info",
        });
      } else {
        setSnackbar({
          open: true,
          message: `${completed}개 문서가 업로드되었습니다.`,
          severity: "success",
        });
        setTimeout(() => setUploadQueue([]), 3000);
      }
    },
    [queryClient]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop: handleUpload,
    accept: ACCEPTED_TYPES,
    disabled: uploading,
  });

  // -----------------------------------------------------------------------
  // Sidebar
  // -----------------------------------------------------------------------

  const sidebarContent = (
    <Box
      sx={{
        width: SIDEBAR_WIDTH,
        height: "100%",
        display: "flex",
        flexDirection: "column",
        bgcolor: "grey.50",
      }}
    >
      <Box sx={{ p: 2, display: "flex", alignItems: "center", gap: 1 }}>
        <DescriptionIcon color="primary" />
        <Typography variant="h6" fontWeight={700}>
          AI 어시스턴트
        </Typography>
      </Box>

      <Divider />
      <Box sx={{ flex: 1 }} />
      <Divider />

      <List dense>
        <ListItemButton
          sx={{ borderRadius: 1, mx: 1 }}
          onClick={() => navigate("/chat")}
        >
          <ListItemIcon sx={{ minWidth: 36 }}>
            <ChatIcon fontSize="small" />
          </ListItemIcon>
          <ListItemText primary="채팅" primaryTypographyProps={{ variant: "body2" }} />
        </ListItemButton>
        <ListItemButton selected sx={{ borderRadius: 1, mx: 1 }}>
          <ListItemIcon sx={{ minWidth: 36 }}>
            <DescriptionIcon fontSize="small" />
          </ListItemIcon>
          <ListItemText primary="문서" primaryTypographyProps={{ variant: "body2" }} />
        </ListItemButton>
        <ListItemButton sx={{ borderRadius: 1, mx: 1 }} onClick={() => navigate("/admin")}>
          <ListItemIcon sx={{ minWidth: 36 }}>
            <AdminIcon fontSize="small" />
          </ListItemIcon>
          <ListItemText primary="관리" primaryTypographyProps={{ variant: "body2" }} />
        </ListItemButton>
        <ListItemButton sx={{ borderRadius: 1, mx: 1 }} onClick={() => navigate("/monitor")}>
          <ListItemIcon sx={{ minWidth: 36 }}>
            <MonitorIcon fontSize="small" />
          </ListItemIcon>
          <ListItemText primary="모니터링" primaryTypographyProps={{ variant: "body2" }} />
        </ListItemButton>
      </List>
    </Box>
  );

  // -----------------------------------------------------------------------
  // Sort direction icon helper
  // -----------------------------------------------------------------------

  const SortIcon = ({ field }: { field: SortField }) =>
    sortField === field ? (
      sortDir === "asc" ? (
        <ArrowUpIcon sx={{ fontSize: 14, ml: 0.5 }} />
      ) : (
        <ArrowDownIcon sx={{ fontSize: 14, ml: 0.5 }} />
      )
    ) : null;

  // -----------------------------------------------------------------------
  // Card view
  // -----------------------------------------------------------------------

  const DocumentCard = ({ doc }: { doc: DocumentRow }) => {
    const ext = getFileExt(doc);
    const borderColor = getFileTypeColor(ext);
    const fileSize = doc.file_size ?? doc.fileSize ?? 0;
    const chunkCount = doc.chunk_count ?? doc.chunkCount ?? 0;
    const uploadDate = doc.uploaded_at ?? doc.uploadDate ?? "";
    const isSelected = selectedIds.has(doc.id);

    return (
      <Card
        variant="outlined"
        sx={{
          borderLeft: `4px solid ${borderColor}`,
          borderColor: isSelected ? "primary.main" : undefined,
          outline: isSelected ? "2px solid" : "none",
          outlineColor: "primary.main",
          position: "relative",
          transition: "box-shadow 0.15s",
          "&:hover": { boxShadow: 3 },
        }}
      >
        {/* Selection checkbox overlay */}
        <Box
          sx={{
            position: "absolute",
            top: 6,
            right: 6,
            zIndex: 1,
          }}
        >
          <Checkbox
            size="small"
            checked={isSelected}
            onClick={(e) => toggleSelect(doc.id, e)}
          />
        </Box>

        <CardActionArea onClick={() => setDetailDoc(doc)} sx={{ p: 0 }}>
          <CardContent sx={{ pr: 5 }}>
            {/* File type badge */}
            <Chip
              label={ext || "FILE"}
              size="small"
              sx={{
                bgcolor: borderColor,
                color: "#fff",
                fontWeight: 700,
                fontSize: "0.7rem",
                mb: 1,
              }}
            />

            {/* Filename */}
            <Typography
              variant="body2"
              fontWeight={600}
              sx={{
                mb: 1,
                display: "-webkit-box",
                WebkitLineClamp: 2,
                WebkitBoxOrient: "vertical",
                overflow: "hidden",
                lineHeight: 1.4,
                minHeight: "2.8em",
              }}
            >
              {doc.filename}
            </Typography>

            {/* Meta row */}
            <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap sx={{ mb: 0.5 }}>
              {fileSize > 0 && (
                <Typography variant="caption" color="text.secondary">
                  {formatFileSize(fileSize)}
                </Typography>
              )}
              {chunkCount > 0 && (
                <Typography variant="caption" color="text.secondary">
                  {chunkCount}청크
                </Typography>
              )}
            </Stack>

            {/* Date */}
            {uploadDate && (
              <Typography variant="caption" color="text.secondary" display="block" sx={{ mb: 0.5 }}>
                {formatDate(uploadDate)}
              </Typography>
            )}

            {/* Status */}
            {getStatusChip(doc.status)}
          </CardContent>
        </CardActionArea>
      </Card>
    );
  };

  // -----------------------------------------------------------------------
  // Main render
  // -----------------------------------------------------------------------

  return (
    <Box sx={{ display: "flex", height: "100vh" }}>
      {/* Sidebar */}
      <Drawer
        variant="persistent"
        open={sidebarOpen}
        sx={{
          width: sidebarOpen ? SIDEBAR_WIDTH : 0,
          flexShrink: 0,
          "& .MuiDrawer-paper": {
            width: SIDEBAR_WIDTH,
            boxSizing: "border-box",
            borderRight: "1px solid",
            borderColor: "divider",
          },
        }}
      >
        {sidebarContent}
      </Drawer>

      {/* Main content */}
      <Box sx={{ flex: 1, display: "flex", flexDirection: "column", minWidth: 0, overflow: "auto" }}>

        {/* Header */}
        <Box
          sx={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            px: 3,
            py: 2,
            borderBottom: "1px solid",
            borderColor: "divider",
            bgcolor: "background.paper",
          }}
        >
          <Typography variant="h5" fontWeight={600}>
            문서 관리
          </Typography>
          <Button
            variant="contained"
            startIcon={<UploadIcon />}
            component="label"
            disabled={uploading}
          >
            파일 업로드 (다중 선택 가능)
            <input
              type="file"
              hidden
              multiple
              accept=".pdf,.hwp,.docx,.xlsx,.pptx,.txt"
              onChange={(e) => {
                if (e.target.files) {
                  handleUpload(Array.from(e.target.files));
                  e.target.value = "";
                }
              }}
            />
          </Button>
        </Box>

        {/* Drop zone */}
        <Box sx={{ px: 3, pt: 2 }}>
          <Paper
            {...getRootProps()}
            variant="outlined"
            sx={{
              p: 4,
              textAlign: "center",
              cursor: uploading ? "default" : "pointer",
              borderStyle: "dashed",
              borderWidth: 2,
              borderColor: isDragActive ? "primary.main" : "divider",
              bgcolor: isDragActive ? "primary.50" : "transparent",
              transition: "all 0.2s",
              "&:hover": uploading
                ? {}
                : { borderColor: "primary.main", bgcolor: "grey.50" },
            }}
          >
            <input {...getInputProps()} />
            <UploadIcon sx={{ fontSize: 48, color: "text.secondary", mb: 1 }} />
            <Typography variant="body1" color="text.secondary">
              {isDragActive
                ? "파일을 여기에 놓으세요"
                : "파일을 드래그하거나 클릭하여 선택 (여러 파일 동시 업로드 가능)"}
            </Typography>
            <Typography variant="caption" color="text.secondary">
              지원 형식: PDF, HWP, DOCX, XLSX, PPTX, TXT
            </Typography>
          </Paper>
        </Box>

        {/* Upload queue */}
        {uploadQueue.length > 0 && (
          <Box sx={{ px: 3, pt: 2 }}>
            <Paper variant="outlined" sx={{ overflow: "hidden" }}>
              <Box
                sx={{
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "space-between",
                  px: 2,
                  py: 1,
                  bgcolor: "grey.50",
                  borderBottom: queueExpanded ? "1px solid" : "none",
                  borderColor: "divider",
                  cursor: "pointer",
                  userSelect: "none",
                }}
                onClick={() => setQueueExpanded((v) => !v)}
              >
                <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                  <Typography variant="body2" fontWeight={600}>
                    업로드 현황
                  </Typography>
                  <Chip
                    size="small"
                    label={`${uploadQueue.filter((q) => q.status === "done").length} / ${uploadQueue.length}`}
                    color={
                      uploading
                        ? "info"
                        : uploadQueue.some((q) => q.status === "error")
                        ? "warning"
                        : "success"
                    }
                  />
                </Box>
                <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                  {!uploading && (
                    <Button
                      size="small"
                      variant="text"
                      onClick={(e) => {
                        e.stopPropagation();
                        setUploadQueue([]);
                      }}
                    >
                      닫기
                    </Button>
                  )}
                  <IconButton size="small">
                    {queueExpanded ? (
                      <ExpandLessIcon fontSize="small" />
                    ) : (
                      <ExpandMoreIcon fontSize="small" />
                    )}
                  </IconButton>
                </Box>
              </Box>

              {uploading && (
                <LinearProgress
                  variant="determinate"
                  value={uploadProgress}
                  sx={{ height: 3 }}
                />
              )}

              <Collapse in={queueExpanded}>
                <Box sx={{ maxHeight: 240, overflowY: "auto" }}>
                  {uploadQueue.map((item, idx) => (
                    <Box
                      key={idx}
                      sx={{
                        display: "flex",
                        alignItems: "center",
                        gap: 1.5,
                        px: 2,
                        py: 0.75,
                        borderBottom: idx < uploadQueue.length - 1 ? "1px solid" : "none",
                        borderColor: "divider",
                        bgcolor:
                          item.status === "error"
                            ? "error.50"
                            : item.status === "done"
                            ? "success.50"
                            : "transparent",
                      }}
                    >
                      <Box sx={{ flexShrink: 0, display: "flex" }}>
                        {item.status === "pending" && (
                          <PendingIcon fontSize="small" sx={{ color: "text.disabled" }} />
                        )}
                        {item.status === "uploading" && (
                          <CircularProgress size={16} thickness={5} />
                        )}
                        {item.status === "done" && (
                          <CheckCircleIcon fontSize="small" color="success" />
                        )}
                        {item.status === "error" && (
                          <ErrorIcon fontSize="small" color="error" />
                        )}
                      </Box>

                      <Box sx={{ flex: 1, minWidth: 0 }}>
                        <Typography
                          variant="body2"
                          noWrap
                          sx={{
                            color: item.status === "error" ? "error.main" : "text.primary",
                          }}
                        >
                          {item.file.name}
                        </Typography>
                        {item.errorMessage && (
                          <Typography variant="caption" color="error" noWrap>
                            {item.errorMessage}
                          </Typography>
                        )}
                      </Box>

                      <Typography variant="caption" color="text.secondary" sx={{ flexShrink: 0 }}>
                        {formatFileSize(item.file.size)}
                      </Typography>

                      <Chip
                        size="small"
                        label={
                          item.status === "pending"
                            ? "대기"
                            : item.status === "uploading"
                            ? "업로드 중"
                            : item.status === "done"
                            ? "완료"
                            : "실패"
                        }
                        color={
                          item.status === "pending"
                            ? "default"
                            : item.status === "uploading"
                            ? "info"
                            : item.status === "done"
                            ? "success"
                            : "error"
                        }
                        sx={{ flexShrink: 0, minWidth: 64 }}
                      />
                    </Box>
                  ))}
                </Box>
              </Collapse>
            </Paper>
          </Box>
        )}

        {/* Documents section */}
        <Box sx={{ px: 3, py: 2, flex: 1 }}>
          {isLoading ? (
            <Box sx={{ display: "flex", justifyContent: "center", py: 4 }}>
              <LinearProgress sx={{ width: "50%" }} />
            </Box>
          ) : documents.length === 0 ? (
            <Box
              sx={{
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                justifyContent: "center",
                py: 8,
                gap: 2,
              }}
            >
              <FolderOpenIcon sx={{ fontSize: 64, color: "text.secondary", opacity: 0.4 }} />
              <Typography variant="h6" color="text.secondary">
                등록된 문서가 없습니다
              </Typography>
              <Typography variant="body2" color="text.secondary">
                위의 업로드 영역에 파일을 드래그하여 문서를 추가하세요.
              </Typography>
            </Box>
          ) : (
            <>
              {/* ---- Toolbar ---- */}
              <Box sx={{ mb: 2, display: "flex", flexDirection: "column", gap: 1.5 }}>

                {/* Row 1: Search + Sort + View toggle */}
                <Box sx={{ display: "flex", alignItems: "center", gap: 1.5, flexWrap: "wrap" }}>
                  <TextField
                    size="small"
                    placeholder="파일명 검색..."
                    value={searchText}
                    onChange={(e) => handleSearchChange(e.target.value)}
                    sx={{ flexGrow: 1, maxWidth: 360 }}
                    slotProps={{
                      input: {
                        startAdornment: (
                          <InputAdornment position="start">
                            <SearchIcon fontSize="small" color="action" />
                          </InputAdornment>
                        ),
                        endAdornment: searchText ? (
                          <InputAdornment position="end">
                            <IconButton
                              size="small"
                              onClick={() => handleSearchChange("")}
                              edge="end"
                            >
                              <CloseIcon fontSize="small" />
                            </IconButton>
                          </InputAdornment>
                        ) : null,
                      },
                    }}
                  />

                  {/* Sort field */}
                  <FormControl size="small" sx={{ minWidth: 140 }}>
                    <InputLabel>정렬 기준</InputLabel>
                    <Select
                      label="정렬 기준"
                      value={sortField}
                      onChange={(e) => handleSortFieldChange(e.target.value as SortField)}
                    >
                      <MenuItem value="date">날짜 (최신순)</MenuItem>
                      <MenuItem value="name">이름 (가나다)</MenuItem>
                      <MenuItem value="size">크기 (큰 순)</MenuItem>
                      <MenuItem value="chunks">청크 수</MenuItem>
                    </Select>
                  </FormControl>

                  {/* Sort direction */}
                  <Tooltip title={sortDir === "asc" ? "오름차순" : "내림차순"}>
                    <IconButton
                      size="small"
                      onClick={() => setSortDir((d) => (d === "asc" ? "desc" : "asc"))}
                    >
                      {sortDir === "asc" ? (
                        <ArrowUpIcon fontSize="small" />
                      ) : (
                        <ArrowDownIcon fontSize="small" />
                      )}
                    </IconButton>
                  </Tooltip>

                  {/* Bulk delete button */}
                  {selectedIds.size > 0 && (
                    <Button
                      variant="contained"
                      color="error"
                      size="small"
                      startIcon={<DeleteIcon />}
                      onClick={() => setBulkDeleteDialogOpen(true)}
                    >
                      선택 삭제 ({selectedIds.size})
                    </Button>
                  )}
                  {selectedIds.size > 0 && (
                    <Button size="small" variant="text" onClick={clearSelection}>
                      선택 해제
                    </Button>
                  )}

                  <Box sx={{ flex: 1 }} />

                  {/* View toggle */}
                  <ToggleButtonGroup
                    size="small"
                    value={viewMode}
                    exclusive
                    onChange={handleViewModeChange}
                  >
                    <ToggleButton value="list">
                      <Tooltip title="목록 보기">
                        <ViewListIcon fontSize="small" />
                      </Tooltip>
                    </ToggleButton>
                    <ToggleButton value="card">
                      <Tooltip title="카드 보기">
                        <ViewModuleIcon fontSize="small" />
                      </Tooltip>
                    </ToggleButton>
                  </ToggleButtonGroup>
                </Box>

                {/* Row 2: Status filter chips + Type filter chips + summary */}
                <Box sx={{ display: "flex", alignItems: "center", flexWrap: "wrap", gap: 1 }}>
                  {/* Status filter */}
                  <Box sx={{ display: "flex", alignItems: "center", gap: 0.5 }}>
                    <Typography variant="caption" color="text.secondary" sx={{ mr: 0.5 }}>
                      상태:
                    </Typography>
                    {STATUS_FILTERS.map((s) => (
                      <Chip
                        key={s}
                        label={s}
                        size="small"
                        clickable
                        color={statusFilter === s ? "primary" : "default"}
                        variant={statusFilter === s ? "filled" : "outlined"}
                        onClick={() => handleStatusFilter(s)}
                      />
                    ))}
                  </Box>

                  <Divider orientation="vertical" flexItem sx={{ mx: 0.5 }} />

                  {/* File type filter */}
                  <Box sx={{ display: "flex", alignItems: "center", gap: 0.5, flexWrap: "wrap" }}>
                    <Typography variant="caption" color="text.secondary" sx={{ mr: 0.5 }}>
                      유형:
                    </Typography>
                    {FILE_TYPE_FILTERS.map((ft) => (
                      <Chip
                        key={ft}
                        label={ft}
                        size="small"
                        clickable
                        color={typeFilter === ft ? "primary" : "default"}
                        variant={typeFilter === ft ? "filled" : "outlined"}
                        onClick={() => handleTypeFilter(ft)}
                      />
                    ))}
                  </Box>

                  <Divider orientation="vertical" flexItem sx={{ mx: 0.5 }} />

                  {/* Summary */}
                  <Stack direction="row" spacing={1}>
                    <Chip
                      size="small"
                      label={`총 ${totalCount}건`}
                      color="default"
                      sx={{ fontWeight: 600 }}
                    />
                    {totalSize > 0 && (
                      <Chip
                        size="small"
                        label={formatFileSize(totalSize)}
                        color="default"
                        variant="outlined"
                      />
                    )}
                  </Stack>
                </Box>
              </Box>

              {/* ---- Content ---- */}
              {filteredDocuments.length === 0 ? (
                <Box sx={{ textAlign: "center", py: 6 }}>
                  <Typography color="text.secondary">검색 결과가 없습니다.</Typography>
                </Box>
              ) : viewMode === "list" ? (
                /* ---- List / Table view ---- */
                <Paper variant="outlined">
                  <TableContainer>
                    <Table size="small">
                      <TableHead>
                        <TableRow>
                          <TableCell padding="checkbox">
                            <Checkbox
                              size="small"
                              indeterminate={
                                selectedIds.size > 0 && selectedIds.size < pagedDocuments.length
                              }
                              checked={
                                pagedDocuments.length > 0 &&
                                selectedIds.size === pagedDocuments.length
                              }
                              onChange={toggleSelectAll}
                            />
                          </TableCell>
                          <TableCell
                            sx={{ cursor: "pointer", userSelect: "none" }}
                            onClick={() => handleSortFieldChange("name")}
                          >
                            <Box sx={{ display: "flex", alignItems: "center" }}>
                              파일명 <SortIcon field="name" />
                            </Box>
                          </TableCell>
                          <TableCell>유형</TableCell>
                          <TableCell
                            align="right"
                            sx={{ cursor: "pointer", userSelect: "none" }}
                            onClick={() => handleSortFieldChange("size")}
                          >
                            <Box sx={{ display: "flex", alignItems: "center", justifyContent: "flex-end" }}>
                              크기 <SortIcon field="size" />
                            </Box>
                          </TableCell>
                          <TableCell
                            align="right"
                            sx={{ cursor: "pointer", userSelect: "none" }}
                            onClick={() => handleSortFieldChange("chunks")}
                          >
                            <Box sx={{ display: "flex", alignItems: "center", justifyContent: "flex-end" }}>
                              청크수 <SortIcon field="chunks" />
                            </Box>
                          </TableCell>
                          <TableCell
                            sx={{ cursor: "pointer", userSelect: "none" }}
                            onClick={() => handleSortFieldChange("date")}
                          >
                            <Box sx={{ display: "flex", alignItems: "center" }}>
                              업로드일 <SortIcon field="date" />
                            </Box>
                          </TableCell>
                          <TableCell>상태</TableCell>
                          <TableCell align="center">액션</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {pagedDocuments.map((doc) => {
                          const fileType = getFileExt(doc);
                          const fileSize = doc.file_size ?? doc.fileSize ?? 0;
                          const chunkCount = doc.chunk_count ?? doc.chunkCount ?? 0;
                          const uploadDate = doc.uploaded_at ?? doc.uploadDate ?? "";
                          const isSelected = selectedIds.has(doc.id);

                          return (
                            <TableRow
                              key={doc.id}
                              hover
                              selected={isSelected}
                              sx={{ cursor: "pointer" }}
                              onClick={() => setDetailDoc(doc)}
                            >
                              <TableCell padding="checkbox">
                                <Checkbox
                                  size="small"
                                  checked={isSelected}
                                  onClick={(e) => toggleSelect(doc.id, e)}
                                />
                              </TableCell>
                              <TableCell>
                                <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                                  <Box
                                    sx={{
                                      width: 4,
                                      height: 28,
                                      borderRadius: 1,
                                      bgcolor: getFileTypeColor(fileType),
                                      flexShrink: 0,
                                    }}
                                  />
                                  <FileIcon fontSize="small" color="action" />
                                  <Typography variant="body2" noWrap sx={{ maxWidth: 280 }}>
                                    {doc.filename}
                                  </Typography>
                                </Box>
                              </TableCell>
                              <TableCell>
                                <Chip label={fileType || "-"} size="small" variant="outlined" />
                              </TableCell>
                              <TableCell align="right">
                                {fileSize > 0 ? formatFileSize(fileSize) : "-"}
                              </TableCell>
                              <TableCell align="right">{chunkCount}</TableCell>
                              <TableCell>{uploadDate ? formatDate(uploadDate) : "-"}</TableCell>
                              <TableCell>{getStatusChip(doc.status)}</TableCell>
                              <TableCell align="center">
                                <Box sx={{ display: "flex", gap: 0.5, justifyContent: "center" }}>
                                  <Tooltip title="다운로드">
                                    <IconButton
                                      size="small"
                                      color="primary"
                                      onClick={(e) => {
                                        e.stopPropagation();
                                        documentsApi.download(doc.id);
                                      }}
                                    >
                                      <DownloadIcon fontSize="small" />
                                    </IconButton>
                                  </Tooltip>
                                  <Tooltip title="삭제">
                                    <IconButton
                                      size="small"
                                      color="error"
                                      onClick={(e) => {
                                        e.stopPropagation();
                                        setDeleteDialogId(doc.id);
                                      }}
                                    >
                                      <DeleteIcon fontSize="small" />
                                    </IconButton>
                                  </Tooltip>
                                </Box>
                              </TableCell>
                            </TableRow>
                          );
                        })}
                      </TableBody>
                    </Table>
                  </TableContainer>
                  <TablePagination
                    component="div"
                    count={filteredDocuments.length}
                    page={page}
                    onPageChange={(_e, newPage) => setPage(newPage)}
                    rowsPerPage={ROWS_PER_PAGE}
                    rowsPerPageOptions={[ROWS_PER_PAGE]}
                    labelDisplayedRows={({ from, to, count }) =>
                      `${from}–${to} / ${count}건`
                    }
                  />
                </Paper>
              ) : (
                /* ---- Card / Grid view ---- */
                <>
                  <Grid container spacing={2}>
                    {pagedDocuments.map((doc) => (
                      <Grid key={doc.id} size={{ xs: 12, sm: 6, md: 4, lg: 3 }}>
                        <DocumentCard doc={doc} />
                      </Grid>
                    ))}
                  </Grid>
                  <TablePagination
                    component="div"
                    count={filteredDocuments.length}
                    page={page}
                    onPageChange={(_e, newPage) => setPage(newPage)}
                    rowsPerPage={CARDS_PER_PAGE}
                    rowsPerPageOptions={[CARDS_PER_PAGE]}
                    labelDisplayedRows={({ from, to, count }) =>
                      `${from}–${to} / ${count}건`
                    }
                  />
                </>
              )}
            </>
          )}
        </Box>
      </Box>

      {/* ----------------------------------------------------------------- */}
      {/* Detail Dialog                                                      */}
      {/* ----------------------------------------------------------------- */}
      <Dialog
        open={detailDoc !== null}
        onClose={() => setDetailDoc(null)}
        maxWidth="sm"
        fullWidth
      >
        {detailDoc && (
          <>
            <DialogTitle
              sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", pb: 1 }}
            >
              <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                <Box
                  sx={{
                    width: 6,
                    height: 32,
                    borderRadius: 1,
                    bgcolor: getFileTypeColor(getFileExt(detailDoc)),
                    flexShrink: 0,
                  }}
                />
                <Typography variant="h6">문서 상세</Typography>
              </Box>
              <IconButton onClick={() => setDetailDoc(null)} size="small">
                <CloseIcon />
              </IconButton>
            </DialogTitle>

            <Divider />

            <DialogContent>
              <Box sx={{ display: "flex", flexDirection: "column", gap: 2, pt: 1 }}>
                {/* Filename */}
                <Box>
                  <Typography variant="caption" color="text.secondary">파일명</Typography>
                  <Typography variant="body2" sx={{ wordBreak: "break-all", fontWeight: 500 }}>
                    {detailDoc.filename}
                  </Typography>
                </Box>

                <Grid container spacing={2}>
                  <Grid size={{ xs: 6 }}>
                    <Typography variant="caption" color="text.secondary">파일 유형</Typography>
                    <Box sx={{ mt: 0.5 }}>
                      <Chip
                        label={getFileExt(detailDoc) || "-"}
                        size="small"
                        sx={{
                          bgcolor: getFileTypeColor(getFileExt(detailDoc)),
                          color: "#fff",
                          fontWeight: 700,
                        }}
                      />
                    </Box>
                  </Grid>
                  <Grid size={{ xs: 6 }}>
                    <Typography variant="caption" color="text.secondary">상태</Typography>
                    <Box sx={{ mt: 0.5 }}>
                      {getStatusChip(detailDoc.status)}
                    </Box>
                  </Grid>
                  <Grid size={{ xs: 6 }}>
                    <Typography variant="caption" color="text.secondary">파일 크기</Typography>
                    <Typography variant="body2">
                      {formatFileSize(detailDoc.file_size ?? detailDoc.fileSize ?? 0)}
                    </Typography>
                  </Grid>
                  <Grid size={{ xs: 6 }}>
                    <Typography variant="caption" color="text.secondary">청크 수</Typography>
                    <Typography variant="body2">
                      {detailDoc.chunk_count ?? detailDoc.chunkCount ?? 0}개
                    </Typography>
                  </Grid>
                  <Grid size={{ xs: 12 }}>
                    <Typography variant="caption" color="text.secondary">업로드일</Typography>
                    <Typography variant="body2">
                      {(detailDoc.uploaded_at ?? detailDoc.uploadDate)
                        ? formatDate(detailDoc.uploaded_at ?? detailDoc.uploadDate ?? "")
                        : "-"}
                    </Typography>
                  </Grid>
                  {!!detailDoc.metadata?.department && (
                    <Grid size={{ xs: 6 }}>
                      <Typography variant="caption" color="text.secondary">부서</Typography>
                      <Typography variant="body2">
                        {String(detailDoc.metadata.department)}
                      </Typography>
                    </Grid>
                  )}
                  {!!detailDoc.metadata?.category && (
                    <Grid size={{ xs: 6 }}>
                      <Typography variant="caption" color="text.secondary">카테고리</Typography>
                      <Typography variant="body2">
                        {String(detailDoc.metadata.category)}
                      </Typography>
                    </Grid>
                  )}
                  {detailDoc.metadata?.ocr_applied !== undefined && (
                    <Grid size={{ xs: 6 }}>
                      <Typography variant="caption" color="text.secondary">OCR 적용</Typography>
                      <Typography variant="body2">
                        {detailDoc.metadata.ocr_applied ? "예" : "아니오"}
                      </Typography>
                    </Grid>
                  )}
                </Grid>

                {detailDoc.metadata && Object.keys(detailDoc.metadata).length > 0 && (
                  <Box>
                    <Typography variant="caption" color="text.secondary">메타데이터</Typography>
                    <Paper variant="outlined" sx={{ p: 1.5, mt: 0.5, fontSize: "0.82rem" }}>
                      <pre style={{ margin: 0, whiteSpace: "pre-wrap", wordBreak: "break-all" }}>
                        {JSON.stringify(detailDoc.metadata, null, 2)}
                      </pre>
                    </Paper>
                  </Box>
                )}
              </Box>
            </DialogContent>

            <Divider />

            <DialogActions sx={{ px: 3, py: 2, gap: 1 }}>
              <Button
                variant="outlined"
                color="primary"
                startIcon={<DownloadIcon />}
                onClick={() => documentsApi.download(detailDoc.id)}
              >
                다운로드
              </Button>
              <Button
                variant="outlined"
                color="error"
                startIcon={<DeleteIcon />}
                onClick={() => {
                  setDeleteDialogId(detailDoc.id);
                  setDetailDoc(null);
                }}
              >
                삭제
              </Button>
              <Box sx={{ flex: 1 }} />
              <Button variant="contained" onClick={() => setDetailDoc(null)}>
                닫기
              </Button>
            </DialogActions>
          </>
        )}
      </Dialog>

      {/* ----------------------------------------------------------------- */}
      {/* Single delete confirmation                                          */}
      {/* ----------------------------------------------------------------- */}
      <Dialog open={deleteDialogId !== null} onClose={() => setDeleteDialogId(null)}>
        <DialogTitle>문서 삭제</DialogTitle>
        <DialogContent>
          <DialogContentText>
            이 문서를 삭제하시겠습니까? 관련된 모든 청크도 함께 삭제됩니다.
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDeleteDialogId(null)}>취소</Button>
          <Button
            color="error"
            variant="contained"
            onClick={() => {
              if (deleteDialogId) deleteMutation.mutate(deleteDialogId);
            }}
          >
            삭제
          </Button>
        </DialogActions>
      </Dialog>

      {/* ----------------------------------------------------------------- */}
      {/* Bulk delete confirmation                                            */}
      {/* ----------------------------------------------------------------- */}
      <Dialog open={bulkDeleteDialogOpen} onClose={() => setBulkDeleteDialogOpen(false)}>
        <DialogTitle>선택 문서 일괄 삭제</DialogTitle>
        <DialogContent>
          <DialogContentText>
            선택한 {selectedIds.size}개 문서를 삭제하시겠습니까?{" "}
            관련된 모든 청크도 함께 삭제됩니다. 이 작업은 되돌릴 수 없습니다.
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setBulkDeleteDialogOpen(false)}>취소</Button>
          <Button
            color="error"
            variant="contained"
            disabled={bulkDeleteMutation.isPending}
            onClick={() => bulkDeleteMutation.mutate(Array.from(selectedIds))}
          >
            {bulkDeleteMutation.isPending ? "삭제 중..." : `${selectedIds.size}개 삭제`}
          </Button>
        </DialogActions>
      </Dialog>

      {/* ----------------------------------------------------------------- */}
      {/* Ingest Progress Tracker                                             */}
      {/* ----------------------------------------------------------------- */}
      <IngestProgressTracker
        open={ingestTrackerOpen}
        files={ingestFiles}
        onClose={() => setIngestTrackerOpen(false)}
        onAllComplete={() => {
          queryClient.invalidateQueries({ queryKey: ["documents"] });
        }}
      />

      {/* ----------------------------------------------------------------- */}
      {/* Snackbar                                                            */}
      {/* ----------------------------------------------------------------- */}
      <Snackbar
        open={snackbar.open}
        autoHideDuration={3000}
        onClose={() => setSnackbar((prev) => ({ ...prev, open: false }))}
        anchorOrigin={{ vertical: "bottom", horizontal: "center" }}
      >
        <Alert
          severity={snackbar.severity}
          onClose={() => setSnackbar((prev) => ({ ...prev, open: false }))}
        >
          {snackbar.message}
        </Alert>
      </Snackbar>
    </Box>
  );
}
