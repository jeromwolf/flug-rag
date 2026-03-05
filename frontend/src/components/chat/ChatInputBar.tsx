import { useRef, useState, useCallback, useEffect } from "react";
import {
  Box,
  TextField,
  IconButton,
  Tooltip,
  Chip,
  Typography,
  useMediaQuery,
  useTheme,
  Popover,
  List,
  ListItemButton,
  ListItemText,
  ListItemIcon,
  Divider,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Snackbar,
  Alert,
} from "@mui/material";
import {
  ArrowUpward as ArrowUpwardIcon,
  StopCircle as StopCircleIcon,
  AttachFile as AttachFileIcon,
  Close as CloseIcon,
  AutoFixHigh as AutoFixHighIcon,
  Add as AddIcon,
  Delete as DeleteIcon,
  Summarize as SummarizeIcon,
  CompareArrows as CompareArrowsIcon,
  BarChart as BarChartIcon,
  Search as SearchIcon,
  Description as DescriptionIcon,
  Star as StarIcon,
  Mic as MicIcon,
} from "@mui/icons-material";
import { useDropzone } from "react-dropzone";
import { useLanguage } from "../../contexts/LanguageContext";

// ─── Web Speech API type declarations ────────────────────────────────────────

interface SpeechRecognitionEvent extends Event {
  readonly resultIndex: number;
  readonly results: SpeechRecognitionResultList;
}

interface SpeechRecognitionResultList {
  readonly length: number;
  item(index: number): SpeechRecognitionResult;
  [index: number]: SpeechRecognitionResult;
}

interface SpeechRecognitionResult {
  readonly isFinal: boolean;
  readonly length: number;
  item(index: number): SpeechRecognitionAlternative;
  [index: number]: SpeechRecognitionAlternative;
}

interface SpeechRecognitionAlternative {
  readonly transcript: string;
  readonly confidence: number;
}

interface SpeechRecognitionErrorEvent extends Event {
  readonly error: string;
  readonly message: string;
}

interface ISpeechRecognition extends EventTarget {
  lang: string;
  continuous: boolean;
  interimResults: boolean;
  maxAlternatives: number;
  onstart: ((this: ISpeechRecognition, ev: Event) => void) | null;
  onend: ((this: ISpeechRecognition, ev: Event) => void) | null;
  onerror: ((this: ISpeechRecognition, ev: SpeechRecognitionErrorEvent) => void) | null;
  onresult: ((this: ISpeechRecognition, ev: SpeechRecognitionEvent) => void) | null;
  start(): void;
  stop(): void;
  abort(): void;
}

interface ISpeechRecognitionConstructor {
  new (): ISpeechRecognition;
}

declare global {
  interface Window {
    SpeechRecognition: ISpeechRecognitionConstructor | undefined;
    webkitSpeechRecognition: ISpeechRecognitionConstructor | undefined;
  }
}

const SpeechRecognitionAPI: ISpeechRecognitionConstructor | undefined =
  typeof window !== "undefined"
    ? window.SpeechRecognition ?? window.webkitSpeechRecognition
    : undefined;

// ─── Prompt Template types & helpers ────────────────────────────────────────

export interface PromptTemplate {
  id: string;
  label: string;
  template: string;
  icon?: string;
  isDefault?: boolean;
}

const STORAGE_KEY = "kogas-ai-templates";

const DEFAULT_TEMPLATES: PromptTemplate[] = [
  {
    id: "default-summarize",
    label: "문서 요약",
    template: "다음 문서의 핵심 내용을 요약해 주세요: {{문서명}}",
    icon: "Summarize",
    isDefault: true,
  },
  {
    id: "default-compare",
    label: "비교 분석",
    template: "{{항목1}}과(와) {{항목2}}의 차이점을 비교 분석해 주세요",
    icon: "CompareArrows",
    isDefault: true,
  },
  {
    id: "default-data",
    label: "데이터 분석",
    template: "{{기간}} 동안의 {{데이터명}} 데이터를 분석해 주세요",
    icon: "BarChart",
    isDefault: true,
  },
  {
    id: "default-regulation",
    label: "규정 검색",
    template: "{{규정명}}에서 {{키워드}}에 대한 규정을 찾아 주세요",
    icon: "Search",
    isDefault: true,
  },
  {
    id: "default-report",
    label: "보고서 작성",
    template: "{{주제}}에 대한 보고서를 작성해 주세요. 포함사항: {{포함사항}}",
    icon: "Description",
    isDefault: true,
  },
];

function loadCustomTemplates(): PromptTemplate[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    return JSON.parse(raw) as PromptTemplate[];
  } catch {
    return [];
  }
}

function saveCustomTemplates(templates: PromptTemplate[]) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(templates));
}

const TEMPLATE_ICON_MAP: Record<string, React.ReactElement> = {
  Summarize: <SummarizeIcon fontSize="small" />,
  CompareArrows: <CompareArrowsIcon fontSize="small" />,
  BarChart: <BarChartIcon fontSize="small" />,
  Search: <SearchIcon fontSize="small" />,
  Description: <DescriptionIcon fontSize="small" />,
  Star: <StarIcon fontSize="small" />,
};

function getTemplateIcon(icon?: string): React.ReactElement {
  if (icon && TEMPLATE_ICON_MAP[icon]) return TEMPLATE_ICON_MAP[icon];
  return <StarIcon fontSize="small" />;
}

// ─── ChatInputBar ────────────────────────────────────────────────────────────

interface ChatInputBarProps {
  inputValue: string;
  onInputChange: (value: string) => void;
  onSend: () => void;
  onStop: () => void;
  onKeyDown: (e: React.KeyboardEvent) => void;
  isStreaming: boolean;
  attachedFiles: File[];
  onFilesAttached: (files: File[]) => void;
  onRemoveFile: (index: number) => void;
}

const ACCEPTED_TYPES = {
  "application/pdf": [".pdf"],
  "application/msword": [".doc"],
  "application/vnd.openxmlformats-officedocument.wordprocessingml.document": [".docx"],
  "application/haansofthwp": [".hwp"],
  "text/plain": [".txt"],
  "application/vnd.ms-excel": [".xls"],
  "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": [".xlsx"],
  "application/vnd.ms-powerpoint": [".ppt"],
  "application/vnd.openxmlformats-officedocument.presentationml.presentation": [".pptx"],
};

export function ChatInputBar({
  inputValue,
  onInputChange,
  onSend,
  onStop,
  onKeyDown,
  isStreaming,
  attachedFiles,
  onFilesAttached,
  onRemoveFile,
}: ChatInputBarProps) {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down("sm"));
  const { t } = useLanguage();

  const attachButtonRef = useRef<HTMLButtonElement>(null);
  const templateButtonRef = useRef<HTMLButtonElement>(null);
  const textFieldRef = useRef<HTMLTextAreaElement | null>(null);

  // ── Voice input ───────────────────────────────────────────────────────────
  const recognitionRef = useRef<ISpeechRecognition | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [voiceError, setVoiceError] = useState<string | null>(null);
  const isSpeechSupported = Boolean(SpeechRecognitionAPI);
  // Accumulate committed (final) transcript so we can append to existing input
  const committedTranscriptRef = useRef("");

  const handleMicClick = useCallback(() => {
    if (!SpeechRecognitionAPI) return;

    // If already recording, stop
    if (isRecording) {
      recognitionRef.current?.stop();
      return;
    }

    // Save current input value as base before we start appending
    committedTranscriptRef.current = "";

    const rec = new SpeechRecognitionAPI();
    rec.lang = "ko-KR";
    rec.continuous = true;
    rec.interimResults = true;

    const baseValue = inputValue;

    rec.onstart = () => {
      setIsRecording(true);
    };

    rec.onresult = (ev: SpeechRecognitionEvent) => {
      let interim = "";
      let finalSegment = "";

      for (let i = ev.resultIndex; i < ev.results.length; i++) {
        const result = ev.results[i];
        if (result.isFinal) {
          finalSegment += result[0].transcript;
        } else {
          interim += result[0].transcript;
        }
      }

      if (finalSegment) {
        committedTranscriptRef.current += finalSegment;
      }

      // Show committed + interim in the input
      const displayed = committedTranscriptRef.current + interim;
      const separator = baseValue && displayed ? " " : "";
      onInputChange(baseValue + separator + displayed);
    };

    rec.onerror = (ev: SpeechRecognitionErrorEvent) => {
      const errorMap: Record<string, string> = {
        "not-allowed": "마이크 접근 권한이 없습니다. 브라우저 설정에서 허용해 주세요.",
        "no-speech": "음성이 감지되지 않았습니다. 다시 시도해 주세요.",
        "audio-capture": "마이크를 찾을 수 없습니다.",
        "network": "네트워크 오류가 발생했습니다.",
        "aborted": "",
      };
      const msg = errorMap[ev.error] ?? `음성 인식 오류: ${ev.error}`;
      if (msg) setVoiceError(msg);
      setIsRecording(false);
    };

    rec.onend = () => {
      setIsRecording(false);
      recognitionRef.current = null;
    };

    recognitionRef.current = rec;
    rec.start();
  }, [isRecording, inputValue, onInputChange]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      recognitionRef.current?.abort();
    };
  }, []);

  // ── Dropzone ──────────────────────────────────────────────────────────────
  const { getRootProps, getInputProps, open } = useDropzone({
    accept: ACCEPTED_TYPES,
    noClick: true,
    noKeyboard: true,
    onDrop: (acceptedFiles) => {
      onFilesAttached(acceptedFiles);
    },
  });

  // ── Character counter ─────────────────────────────────────────────────────
  const charCount = inputValue.length;
  const MAX_CHARS = 5000;
  const showCounter = charCount > 100;
  const isWarning = charCount > 4000;
  const isError = charCount > 4500;

  // ── Template popover state ────────────────────────────────────────────────
  const [templateAnchor, setTemplateAnchor] = useState<HTMLButtonElement | null>(null);
  const [customTemplates, setCustomTemplates] = useState<PromptTemplate[]>(loadCustomTemplates);

  const handleOpenTemplates = useCallback(
    (e: React.MouseEvent<HTMLButtonElement>) => {
      setTemplateAnchor(e.currentTarget);
    },
    [],
  );
  const handleCloseTemplates = useCallback(() => setTemplateAnchor(null), []);

  // Insert template text: replace current input, position cursor at first variable
  const handleSelectTemplate = useCallback(
    (template: PromptTemplate) => {
      onInputChange(template.template);
      handleCloseTemplates();
      // Move cursor to first {{...}} variable after React re-renders
      setTimeout(() => {
        const el = textFieldRef.current;
        if (!el) return;
        el.focus();
        const firstVar = template.template.indexOf("{{");
        if (firstVar !== -1) {
          const endVar = template.template.indexOf("}}", firstVar);
          const start = firstVar;
          const end = endVar !== -1 ? endVar + 2 : firstVar;
          el.setSelectionRange(start, end);
        }
      }, 0);
    },
    [onInputChange, handleCloseTemplates],
  );

  // ── Add/Delete custom template dialog ─────────────────────────────────────
  const [addDialogOpen, setAddDialogOpen] = useState(false);
  const [newLabel, setNewLabel] = useState("");
  const [newTemplateText, setNewTemplateText] = useState("");
  const [addError, setAddError] = useState("");

  const handleOpenAddDialog = useCallback(() => {
    setNewLabel("");
    setNewTemplateText("");
    setAddError("");
    setAddDialogOpen(true);
  }, []);

  const handleCloseAddDialog = useCallback(() => {
    setAddDialogOpen(false);
  }, []);

  const handleSaveTemplate = useCallback(() => {
    if (!newLabel.trim()) {
      setAddError("템플릿 이름을 입력해 주세요.");
      return;
    }
    if (!newTemplateText.trim()) {
      setAddError("템플릿 내용을 입력해 주세요.");
      return;
    }
    const newT: PromptTemplate = {
      id: `custom-${Date.now()}`,
      label: newLabel.trim(),
      template: newTemplateText.trim(),
      icon: "Star",
      isDefault: false,
    };
    const updated = [...customTemplates, newT];
    setCustomTemplates(updated);
    saveCustomTemplates(updated);
    setAddDialogOpen(false);
  }, [newLabel, newTemplateText, customTemplates]);

  const handleDeleteTemplate = useCallback(
    (id: string) => {
      const updated = customTemplates.filter((t) => t.id !== id);
      setCustomTemplates(updated);
      saveCustomTemplates(updated);
    },
    [customTemplates],
  );

  const templatePopoverOpen = Boolean(templateAnchor);

  return (
    <Box
      sx={{
        px: { xs: 1, sm: 2 },
        py: { xs: 1, sm: 2 },
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        bgcolor: "background.default",
      }}
    >
      <Box
        {...getRootProps()}
        sx={{
          maxWidth: 768,
          width: "100%",
          border: "1px solid",
          borderColor: "divider",
          borderRadius: 3,
          bgcolor: "background.paper",
          display: "flex",
          flexDirection: "column",
          transition: "border-color 0.2s",
          "&:focus-within": { borderColor: "text.secondary" },
        }}
      >
        <input {...getInputProps()} />

        {/* Attached files row */}
        {attachedFiles.length > 0 && (
          <Box
            sx={{
              display: "flex",
              flexWrap: "wrap",
              gap: 0.75,
              px: 1.5,
              pt: 1.25,
            }}
          >
            {attachedFiles.map((file, index) => (
              <Chip
                key={`${file.name}-${index}`}
                label={file.name}
                size="small"
                onDelete={() => onRemoveFile(index)}
                deleteIcon={<CloseIcon sx={{ fontSize: 14 }} />}
                sx={{
                  maxWidth: 200,
                  "& .MuiChip-label": {
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                    fontSize: "0.75rem",
                  },
                }}
              />
            ))}
          </Box>
        )}

        {/* Main input row */}
        <Box
          sx={{
            display: "flex",
            alignItems: "flex-end",
            px: 0.5,
            py: 0.5,
            gap: 0.5,
          }}
        >
          {/* File attach button */}
          <Tooltip title={t("chat.attachFile")}>
            <IconButton
              ref={attachButtonRef}
              onClick={open}
              size="small"
              sx={{
                mb: 0.5,
                color: "text.secondary",
                "&:hover": { color: "text.primary" },
              }}
              aria-label="파일 첨부"
            >
              <AttachFileIcon fontSize="small" />
            </IconButton>
          </Tooltip>

          {/* Prompt template button */}
          <Tooltip title="프롬프트 템플릿">
            <IconButton
              ref={templateButtonRef}
              onClick={handleOpenTemplates}
              size="small"
              sx={{
                mb: 0.5,
                color: templatePopoverOpen ? "primary.main" : "text.secondary",
                "&:hover": { color: "primary.main" },
              }}
              aria-label="프롬프트 템플릿"
            >
              <AutoFixHighIcon fontSize="small" />
            </IconButton>
          </Tooltip>

          {/* Voice input button */}
          <Tooltip
            title={
              !isSpeechSupported
                ? "이 브라우저에서는 음성 입력을 지원하지 않습니다"
                : isRecording
                  ? "녹음 중단"
                  : "음성 입력"
            }
          >
            <span>
              <IconButton
                onClick={handleMicClick}
                size="small"
                disabled={!isSpeechSupported}
                sx={{
                  mb: 0.5,
                  position: "relative",
                  color: isRecording ? "error.main" : "text.secondary",
                  "&:hover": { color: isRecording ? "error.dark" : "text.primary" },
                  "@keyframes micPulse": {
                    "0%": { boxShadow: "0 0 0 0 rgba(211, 47, 47, 0.5)" },
                    "70%": { boxShadow: "0 0 0 8px rgba(211, 47, 47, 0)" },
                    "100%": { boxShadow: "0 0 0 0 rgba(211, 47, 47, 0)" },
                  },
                  ...(isRecording && {
                    animation: "micPulse 1.2s ease-out infinite",
                    borderRadius: "50%",
                  }),
                }}
                aria-label={isRecording ? "음성 입력 중단" : "음성 입력 시작"}
              >
                <MicIcon fontSize="small" />
              </IconButton>
            </span>
          </Tooltip>

          {/* Recording status indicator */}
          {isRecording && (
            <Box
              sx={{
                display: "flex",
                alignItems: "center",
                mb: 0.5,
                gap: 0.5,
                "@keyframes blink": {
                  "0%, 100%": { opacity: 1 },
                  "50%": { opacity: 0.2 },
                },
              }}
            >
              <Box
                sx={{
                  width: 7,
                  height: 7,
                  borderRadius: "50%",
                  bgcolor: "error.main",
                  animation: "blink 1s ease-in-out infinite",
                }}
              />
              <Typography
                variant="caption"
                sx={{ fontSize: "0.65rem", color: "error.main", lineHeight: 1, whiteSpace: "nowrap" }}
              >
                녹음 중...
              </Typography>
            </Box>
          )}

          <TextField
            fullWidth
            multiline
            minRows={1}
            maxRows={8}
            placeholder={isMobile ? t("chat.inputPlaceholder") : t("chat.inputPlaceholderDesktop")}
            value={inputValue}
            onChange={(e) => onInputChange(e.target.value)}
            onKeyDown={onKeyDown}
            disabled={isStreaming}
            size="small"
            inputRef={textFieldRef}
            sx={{
              "& .MuiOutlinedInput-root": {
                "& fieldset": { border: "none" },
                padding: "8px 4px",
              },
              "& .MuiInputBase-inputMultiline": {
                lineHeight: 1.6,
                maxHeight: 200,
                overflowY: "auto",
              },
            }}
            inputProps={{ "aria-label": "메시지 입력" }}
          />

          <Box sx={{ display: "flex", flexDirection: "column", alignItems: "flex-end", mb: 0.5, gap: 0.25 }}>
            {showCounter && (
              <Typography
                variant="caption"
                sx={{
                  fontSize: "0.65rem",
                  color: isError ? "error.main" : isWarning ? "warning.main" : "text.disabled",
                  lineHeight: 1,
                  pr: 0.5,
                }}
              >
                {charCount.toLocaleString()} / {MAX_CHARS.toLocaleString()}
              </Typography>
            )}
            {isStreaming ? (
              <Tooltip title={t("chat.stopGeneration")}>
                <IconButton
                  onClick={onStop}
                  sx={{
                    width: 32,
                    height: 32,
                    bgcolor: "error.main",
                    color: "white",
                    "&:hover": { bgcolor: "error.dark" },
                    borderRadius: 1.5,
                  }}
                  aria-label="생성 중단"
                >
                  <StopCircleIcon sx={{ fontSize: 18 }} />
                </IconButton>
              </Tooltip>
            ) : (
              <Tooltip title={`${t("chat.send")} (Enter)`}>
                <span>
                  <IconButton
                    onClick={onSend}
                    disabled={!inputValue.trim()}
                    sx={{
                      width: 32,
                      height: 32,
                      bgcolor: inputValue.trim() ? "primary.main" : "action.disabledBackground",
                      color: "white",
                      "&:hover": { bgcolor: inputValue.trim() ? "primary.dark" : "action.disabledBackground" },
                      "&.Mui-disabled": {
                        bgcolor: "action.disabledBackground",
                        color: "action.disabled",
                      },
                      borderRadius: 1.5,
                      transition: "background-color 0.2s",
                    }}
                    aria-label="메시지 전송"
                  >
                    <ArrowUpwardIcon sx={{ fontSize: 18 }} />
                  </IconButton>
                </span>
              </Tooltip>
            )}
          </Box>
        </Box>
      </Box>

      {/* Keyboard shortcut hint — hidden on mobile */}
      {!isMobile && (
        <Typography
          variant="caption"
          sx={{
            display: "block",
            textAlign: "center",
            mt: 0.75,
            fontSize: "0.7rem",
            color: "text.disabled",
            opacity: 0.6,
            letterSpacing: "0.01em",
            userSelect: "none",
          }}
        >
          {t("chat.keyboardHint")}
        </Typography>
      )}

      {/* ── Template Popover ──────────────────────────────────────────────── */}
      <Popover
        open={templatePopoverOpen}
        anchorEl={templateAnchor}
        onClose={handleCloseTemplates}
        anchorOrigin={{ vertical: "top", horizontal: "left" }}
        transformOrigin={{ vertical: "bottom", horizontal: "left" }}
        PaperProps={{
          sx: {
            width: 320,
            maxHeight: 460,
            display: "flex",
            flexDirection: "column",
            borderRadius: 2,
            boxShadow: 4,
          },
        }}
      >
        {/* Header */}
        <Box
          sx={{
            px: 2,
            py: 1.25,
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            borderBottom: "1px solid",
            borderColor: "divider",
          }}
        >
          <Typography variant="subtitle2" fontWeight={600}>
            프롬프트 템플릿
          </Typography>
          <IconButton size="small" onClick={handleCloseTemplates} aria-label="닫기">
            <CloseIcon fontSize="small" />
          </IconButton>
        </Box>

        {/* Template list */}
        <Box sx={{ flex: 1, overflowY: "auto" }}>
          {/* Default templates */}
          <Typography
            variant="caption"
            sx={{ px: 2, pt: 1.5, pb: 0.5, display: "block", color: "text.secondary", fontWeight: 600 }}
          >
            기본 템플릿
          </Typography>
          <List dense disablePadding>
            {DEFAULT_TEMPLATES.map((t) => (
              <ListItemButton
                key={t.id}
                onClick={() => handleSelectTemplate(t)}
                sx={{ px: 2, py: 0.75 }}
              >
                <ListItemIcon sx={{ minWidth: 32, color: "primary.main" }}>
                  {getTemplateIcon(t.icon)}
                </ListItemIcon>
                <ListItemText
                  primary={
                    <Typography variant="body2" fontWeight={500}>
                      {t.label}
                    </Typography>
                  }
                  secondary={
                    <Typography
                      variant="caption"
                      sx={{
                        color: "text.secondary",
                        display: "-webkit-box",
                        WebkitLineClamp: 1,
                        WebkitBoxOrient: "vertical",
                        overflow: "hidden",
                      }}
                    >
                      {t.template}
                    </Typography>
                  }
                />
              </ListItemButton>
            ))}
          </List>

          {/* Custom templates */}
          {customTemplates.length > 0 && (
            <>
              <Divider sx={{ mx: 2, my: 0.5 }} />
              <Typography
                variant="caption"
                sx={{ px: 2, pt: 0.5, pb: 0.5, display: "block", color: "text.secondary", fontWeight: 600 }}
              >
                내 템플릿
              </Typography>
              <List dense disablePadding>
                {customTemplates.map((t) => (
                  <ListItemButton
                    key={t.id}
                    onClick={() => handleSelectTemplate(t)}
                    sx={{ px: 2, py: 0.75, pr: 1 }}
                  >
                    <ListItemIcon sx={{ minWidth: 32, color: "warning.main" }}>
                      {getTemplateIcon(t.icon)}
                    </ListItemIcon>
                    <ListItemText
                      primary={
                        <Typography variant="body2" fontWeight={500}>
                          {t.label}
                        </Typography>
                      }
                      secondary={
                        <Typography
                          variant="caption"
                          sx={{
                            color: "text.secondary",
                            display: "-webkit-box",
                            WebkitLineClamp: 1,
                            WebkitBoxOrient: "vertical",
                            overflow: "hidden",
                          }}
                        >
                          {t.template}
                        </Typography>
                      }
                    />
                    <IconButton
                      size="small"
                      onClick={(e) => {
                        e.stopPropagation();
                        handleDeleteTemplate(t.id);
                      }}
                      aria-label="템플릿 삭제"
                      sx={{ ml: 0.5, color: "text.disabled", "&:hover": { color: "error.main" } }}
                    >
                      <DeleteIcon sx={{ fontSize: 16 }} />
                    </IconButton>
                  </ListItemButton>
                ))}
              </List>
            </>
          )}
        </Box>

        {/* Add template button */}
        <Box sx={{ borderTop: "1px solid", borderColor: "divider", p: 1 }}>
          <Button
            fullWidth
            size="small"
            startIcon={<AddIcon />}
            onClick={handleOpenAddDialog}
            variant="text"
            sx={{ justifyContent: "flex-start", color: "text.secondary", fontSize: "0.8rem" }}
          >
            템플릿 추가
          </Button>
        </Box>
      </Popover>

      {/* ── Voice error Snackbar ──────────────────────────────────────────── */}
      <Snackbar
        open={Boolean(voiceError)}
        autoHideDuration={4000}
        onClose={() => setVoiceError(null)}
        anchorOrigin={{ vertical: "bottom", horizontal: "center" }}
      >
        <Alert onClose={() => setVoiceError(null)} severity="warning" variant="filled" sx={{ width: "100%" }}>
          {voiceError}
        </Alert>
      </Snackbar>

      {/* ── Add Template Dialog ───────────────────────────────────────────── */}
      <Dialog
        open={addDialogOpen}
        onClose={handleCloseAddDialog}
        maxWidth="xs"
        fullWidth
        PaperProps={{ sx: { borderRadius: 2 } }}
      >
        <DialogTitle sx={{ pb: 1, fontWeight: 600, fontSize: "1rem" }}>새 템플릿 추가</DialogTitle>
        <DialogContent sx={{ pt: 1 }}>
          <TextField
            label="템플릿 이름"
            value={newLabel}
            onChange={(e) => {
              setNewLabel(e.target.value);
              setAddError("");
            }}
            fullWidth
            size="small"
            sx={{ mb: 2 }}
            placeholder="예: 회의록 요약"
            autoFocus
          />
          <TextField
            label="템플릿 내용"
            value={newTemplateText}
            onChange={(e) => {
              setNewTemplateText(e.target.value);
              setAddError("");
            }}
            fullWidth
            size="small"
            multiline
            minRows={3}
            maxRows={6}
            placeholder={"예: {{문서명}}의 주요 내용을 요약해 주세요.\n\n변수는 {{변수명}} 형식으로 입력하세요."}
          />
          {addError && (
            <Typography variant="caption" color="error" sx={{ mt: 0.75, display: "block" }}>
              {addError}
            </Typography>
          )}
          <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: "block" }}>
            변수는 {"{{변수명}}"} 형식으로 입력하면 클릭 시 자동으로 커서가 이동합니다.
          </Typography>
        </DialogContent>
        <DialogActions sx={{ px: 3, pb: 2 }}>
          <Button onClick={handleCloseAddDialog} size="small" color="inherit">
            취소
          </Button>
          <Button onClick={handleSaveTemplate} size="small" variant="contained" disableElevation>
            저장
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}
