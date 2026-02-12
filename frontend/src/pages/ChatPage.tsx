import { useState, useRef, useEffect, useCallback } from "react";
import {
  Box,
  Drawer,
  List,
  ListItemButton,
  ListItemText,
  ListItemIcon,
  IconButton,
  Typography,
  TextField,
  Button,
  Paper,
  Chip,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Tooltip,
  Divider,
  Select,
  MenuItem,
  ToggleButtonGroup,
  ToggleButton,
  CircularProgress,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogContentText,
  DialogActions,
  Snackbar,
  Alert,
  Slider,
} from "@mui/material";
import type { SelectChangeEvent } from "@mui/material";
import {
  Chat as ChatIcon,
  Description as DescriptionIcon,
  AdminPanelSettings as AdminIcon,
  Monitor as MonitorIcon,
  Add as AddIcon,
  Send as SendIcon,
  Stop as StopIcon,
  Delete as DeleteIcon,
  ExpandMore as ExpandMoreIcon,
  ThumbUp as ThumbUpIcon,
  ThumbDown as ThumbDownIcon,
  ContentCopy as CopyIcon,
  Menu as MenuIcon,
  SmartToy as BotIcon,
  ReportProblem as ReportProblemIcon,
  Edit as EditIcon,
} from "@mui/icons-material";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { useAppStore } from "../stores/appStore";
import { sessionsApi, feedbackApi, qualityApi, contentApi, API_BASE, getAuthHeaders } from "../api/client";
import type { Message, Source } from "../types";
import { getConfidenceLevel } from "../types";

const SIDEBAR_WIDTH = 280;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function groupSessionsByDate(
  sessions: Array<{
    id: string;
    title: string;
    created_at?: string;
    createdAt?: string;
    message_count?: number;
    messageCount?: number;
    updated_at?: string;
    updatedAt?: string;
  }>
) {
  const now = new Date();
  const todayStart = new Date(now.getFullYear(), now.getMonth(), now.getDate());
  const weekStart = new Date(todayStart);
  weekStart.setDate(weekStart.getDate() - weekStart.getDay());

  const groups: {
    label: string;
    items: typeof sessions;
  }[] = [
    { label: "오늘", items: [] },
    { label: "이번 주", items: [] },
    { label: "이전", items: [] },
  ];

  for (const s of sessions) {
    const d = new Date(s.created_at ?? s.createdAt ?? "");
    if (d >= todayStart) groups[0].items.push(s);
    else if (d >= weekStart) groups[1].items.push(s);
    else groups[2].items.push(s);
  }

  return groups.filter((g) => g.items.length > 0);
}

const SAMPLE_QUESTIONS = [
  "가스기술공사의 안전 점검 기준은 무엇인가요?",
  "LNG 저장탱크 검사 절차에 대해 알려주세요.",
  "배관 누출 탐지 방법을 설명해주세요.",
  "가스 시설 설치 기준에 대해 알려주세요.",
];

// ---------------------------------------------------------------------------
// Confidence badge
// ---------------------------------------------------------------------------

function ConfidenceBadge({ score }: { score: number }) {
  const level = getConfidenceLevel(score);
  const colorMap = { high: "success", medium: "warning", low: "error" } as const;
  const labelMap = { high: "높음", medium: "중간", low: "낮음" } as const;
  return (
    <Chip
      label={`신뢰도: ${labelMap[level]} (${Math.round(score * 100)}%)`}
      color={colorMap[level]}
      size="small"
      variant="outlined"
    />
  );
}

// ---------------------------------------------------------------------------
// Sources accordion
// ---------------------------------------------------------------------------

function SourcesPanel({ sources }: { sources: Source[] }) {
  if (sources.length === 0) return null;
  return (
    <Accordion
      disableGutters
      sx={{ mt: 1, "&:before": { display: "none" }, boxShadow: "none" }}
    >
      <AccordionSummary expandIcon={<ExpandMoreIcon />}>
        <Typography variant="body2" color="text.secondary">
          참고 문서 ({sources.length}건)
        </Typography>
      </AccordionSummary>
      <AccordionDetails sx={{ p: 0 }}>
        {sources.map((src, idx) => (
          <Box
            key={src.chunkId || idx}
            sx={{
              p: 1.5,
              borderTop: idx > 0 ? "1px solid" : "none",
              borderColor: "divider",
            }}
          >
            <Typography variant="subtitle2">
              {src.filename}
              {src.page != null && ` (p.${src.page})`}
            </Typography>
            <Typography variant="caption" color="text.secondary">
              유사도: {Math.round(src.score * 100)}%
            </Typography>
            <Typography
              variant="body2"
              sx={{
                mt: 0.5,
                whiteSpace: "pre-wrap",
                maxHeight: 120,
                overflow: "auto",
                fontSize: "0.82rem",
                color: "text.secondary",
              }}
            >
              {src.content}
            </Typography>
          </Box>
        ))}
      </AccordionDetails>
    </Accordion>
  );
}

// ---------------------------------------------------------------------------
// Message bubble
// ---------------------------------------------------------------------------

interface MessageBubbleProps {
  msg: Message;
  onFeedback: (messageId: string, rating: number) => void;
  onErrorReport: (messageId: string) => void;
  onEditAnswer: (messageId: string, currentContent: string) => void;
  sessionId: string | null;
  userRole?: string;
}

function MessageBubble({ msg, onFeedback, onErrorReport, onEditAnswer, sessionId, userRole }: MessageBubbleProps) {
  const isUser = msg.role === "user";

  const handleCopy = () => {
    navigator.clipboard.writeText(msg.content);
  };

  const canEdit = !isUser && (userRole === "admin" || userRole === "manager");

  return (
    <Box
      sx={{
        display: "flex",
        justifyContent: isUser ? "flex-end" : "flex-start",
        mb: 2,
        px: 2,
      }}
    >
      <Box sx={{ maxWidth: "75%", minWidth: 80 }}>
        {!isUser && (
          <Box sx={{ display: "flex", alignItems: "center", gap: 0.5, mb: 0.5 }}>
            <BotIcon fontSize="small" color="primary" />
            <Typography variant="caption" color="text.secondary">
              Flux RAG
            </Typography>
          </Box>
        )}

        <Paper
          elevation={isUser ? 0 : 1}
          sx={{
            p: 2,
            borderRadius: 2,
            bgcolor: isUser ? "primary.main" : "grey.50",
            color: isUser ? "primary.contrastText" : "text.primary",
          }}
        >
          {isUser ? (
            <Typography variant="body1" sx={{ whiteSpace: "pre-wrap" }}>
              {msg.content}
            </Typography>
          ) : (
            <Box
              sx={{
                "& p": { m: 0, mb: 1 },
                "& p:last-child": { mb: 0 },
                "& pre": {
                  bgcolor: "grey.100",
                  p: 1.5,
                  borderRadius: 1,
                  overflow: "auto",
                  fontSize: "0.85rem",
                },
                "& code": { fontSize: "0.85rem" },
                "& table": {
                  borderCollapse: "collapse",
                  width: "100%",
                  my: 1,
                },
                "& th, & td": {
                  border: "1px solid",
                  borderColor: "divider",
                  p: 0.5,
                  fontSize: "0.85rem",
                },
              }}
            >
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {msg.content}
              </ReactMarkdown>
            </Box>
          )}
        </Paper>

        {/* Meta row for assistant messages */}
        {!isUser && (
          <Box sx={{ mt: 0.5 }}>
            <Box
              sx={{
                display: "flex",
                alignItems: "center",
                gap: 1,
                flexWrap: "wrap",
              }}
            >
              {msg.confidenceScore != null && (
                <ConfidenceBadge score={msg.confidenceScore} />
              )}
              {msg.latencyMs != null && (
                <Typography variant="caption" color="text.secondary">
                  {(msg.latencyMs / 1000).toFixed(1)}s
                </Typography>
              )}
              {msg.modelUsed && (
                <Typography variant="caption" color="text.secondary">
                  {msg.modelUsed}
                </Typography>
              )}
            </Box>

            {/* Sources */}
            {msg.sources && msg.sources.length > 0 && (
              <SourcesPanel sources={msg.sources} />
            )}

            {/* Action buttons */}
            <Box sx={{ display: "flex", gap: 0.5, mt: 0.5 }}>
              {sessionId && (
                <>
                  <Tooltip title="도움이 됐어요">
                    <IconButton
                      size="small"
                      onClick={() => onFeedback(msg.id, 1)}
                      aria-label="긍정 평가"
                    >
                      <ThumbUpIcon fontSize="small" />
                    </IconButton>
                  </Tooltip>
                  <Tooltip title="도움이 안 됐어요">
                    <IconButton
                      size="small"
                      onClick={() => onFeedback(msg.id, -1)}
                      aria-label="부정 평가"
                    >
                      <ThumbDownIcon fontSize="small" />
                    </IconButton>
                  </Tooltip>
                  <Tooltip title="오류 신고">
                    <IconButton
                      size="small"
                      onClick={() => onErrorReport(msg.id)}
                      aria-label="오류 신고"
                    >
                      <ReportProblemIcon fontSize="small" />
                    </IconButton>
                  </Tooltip>
                  {canEdit && (
                    <Tooltip title="답변 수정">
                      <IconButton
                        size="small"
                        onClick={() => onEditAnswer(msg.id, msg.content)}
                        aria-label="답변 수정"
                      >
                        <EditIcon fontSize="small" />
                      </IconButton>
                    </Tooltip>
                  )}
                </>
              )}
              <Tooltip title="복사">
                <IconButton size="small" onClick={handleCopy}>
                  <CopyIcon fontSize="small" />
                </IconButton>
              </Tooltip>
            </Box>
          </Box>
        )}
      </Box>
    </Box>
  );
}

// ---------------------------------------------------------------------------
// Main ChatPage
// ---------------------------------------------------------------------------

export default function ChatPage() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();

  // Store
  const {
    currentSessionId,
    setCurrentSessionId,
    responseMode,
    setResponseMode,
    selectedModel,
    setSelectedModel,
    sidebarOpen,
    toggleSidebar,
  } = useAppStore();

  // Local state
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [streamingContent, setStreamingContent] = useState("");
  const [deleteDialogId, setDeleteDialogId] = useState<string | null>(null);
  const [temperature, setTemperature] = useState(0.7);
  const [errorReportDialog, setErrorReportDialog] = useState<{
    open: boolean;
    messageId: string | null;
  }>({ open: false, messageId: null });
  const [errorDescription, setErrorDescription] = useState("");
  const [snackbar, setSnackbar] = useState<{
    open: boolean;
    message: string;
    severity: "success" | "error" | "info";
  }>({ open: false, message: "", severity: "info" });
  const [editDialogOpen, setEditDialogOpen] = useState(false);
  const [editingMessageId, setEditingMessageId] = useState("");
  const [editedAnswer, setEditedAnswer] = useState("");
  const [evaluationTag, setEvaluationTag] = useState("accurate");
  const [userRole, setUserRole] = useState<string>("");

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  // -----------------------------------------------------------------------
  // Queries
  // -----------------------------------------------------------------------

  const { data: sessionsData } = useQuery({
    queryKey: ["sessions"],
    queryFn: async () => {
      const res = await sessionsApi.list();
      return res.data;
    },
  });

  const sessions = sessionsData?.sessions ?? [];

  // Fetch announcements
  const { data: announcementsData } = useQuery({
    queryKey: ["announcements"],
    queryFn: () => contentApi.listAnnouncements(),
    staleTime: 60000,
  });

  // Get user role
  useEffect(() => {
    const user = localStorage.getItem("flux_user");
    if (user) {
      try {
        const parsed = JSON.parse(user);
        setUserRole(parsed.role || "");
      } catch {
        // ignore
      }
    }
  }, []);

  // Load messages when session changes
  useEffect(() => {
    if (!currentSessionId) {
      setMessages([]);
      return;
    }
    sessionsApi.getMessages(currentSessionId).then((res) => {
      const raw = res.data as
        | Array<{
            id: string;
            role: string;
            content: string;
            metadata?: Record<string, unknown>;
            created_at?: string;
          }>
        | { messages?: Array<Record<string, unknown>> };

      const arr = Array.isArray(raw)
        ? raw
        : Array.isArray((raw as { messages?: unknown[] }).messages)
          ? ((raw as { messages: Array<Record<string, unknown>> }).messages)
          : [];

      const mapped: Message[] = arr.map((m: Record<string, unknown>) => ({
        id: (m.id as string) ?? crypto.randomUUID(),
        role: (m.role as Message["role"]) ?? "assistant",
        content: (m.content as string) ?? "",
        sources: (m.metadata as Record<string, unknown> | undefined)?.sources as
          | Source[]
          | undefined,
        confidenceScore: (m.metadata as Record<string, unknown> | undefined)
          ?.confidence as number | undefined,
        modelUsed: (m.metadata as Record<string, unknown> | undefined)
          ?.model as string | undefined,
        latencyMs: (m.metadata as Record<string, unknown> | undefined)
          ?.latency_ms as number | undefined,
        createdAt: (m.created_at as string) ?? new Date().toISOString(),
      }));

      setMessages(mapped);
    });
  }, [currentSessionId]);

  // Auto-scroll
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, streamingContent]);

  // -----------------------------------------------------------------------
  // Mutations
  // -----------------------------------------------------------------------

  const deleteSessionMutation = useMutation({
    mutationFn: (id: string) => sessionsApi.delete(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["sessions"] });
      if (deleteDialogId === currentSessionId) {
        setCurrentSessionId(null);
        setMessages([]);
      }
      setDeleteDialogId(null);
    },
  });

  // -----------------------------------------------------------------------
  // Streaming chat
  // -----------------------------------------------------------------------

  const handleSend = useCallback(async () => {
    const text = inputValue.trim();
    if (!text || isStreaming) return;

    setInputValue("");

    // Add user message optimistically
    const userMsg: Message = {
      id: crypto.randomUUID(),
      role: "user",
      content: text,
      createdAt: new Date().toISOString(),
    };
    setMessages((prev) => [...prev, userMsg]);
    setIsStreaming(true);
    setStreamingContent("");

    const abortController = new AbortController();
    abortRef.current = abortController;

    try {
      const body: Record<string, unknown> = {
        message: text,
        mode: responseMode === "direct" ? "direct" : "auto",
      };
      if (currentSessionId) body.session_id = currentSessionId;
      if (selectedModel && selectedModel !== "default") {
        body.model = selectedModel;
      }
      body.temperature = temperature;

      const response = await fetch(`${API_BASE}/chat/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json", ...getAuthHeaders() },
        body: JSON.stringify(body),
        signal: abortController.signal,
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const reader = response.body!.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let content = "";
      let sources: Source[] = [];
      let currentEvent = "";
      let messageId = "";
      let newSessionId = currentSessionId;
      let confidence = 0;
      let latencyMs = 0;

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
                case "start":
                  messageId = data.message_id ?? "";
                  if (data.session_id) newSessionId = data.session_id;
                  break;
                case "source":
                  sources.push({
                    chunkId: data.chunk_id ?? "",
                    filename: data.filename ?? "",
                    page: data.page,
                    content: data.content ?? "",
                    score: data.score ?? 0,
                  });
                  break;
                case "chunk":
                  content += data.content ?? "";
                  setStreamingContent(content);
                  break;
                case "end":
                  confidence = data.confidence_score ?? data.confidence ?? 0;
                  latencyMs = data.latency_ms ?? 0;
                  break;
              }
            } catch {
              // ignore malformed JSON lines
            }
          }
        }
      }

      // Finalize assistant message
      const assistantMsg: Message = {
        id: messageId || crypto.randomUUID(),
        role: "assistant",
        content,
        sources: sources.length > 0 ? sources : undefined,
        confidenceScore: confidence || undefined,
        latencyMs: latencyMs || undefined,
        createdAt: new Date().toISOString(),
      };

      setMessages((prev) => [...prev, assistantMsg]);
      setStreamingContent("");

      // If we got a new session id, update
      if (newSessionId && newSessionId !== currentSessionId) {
        setCurrentSessionId(newSessionId);
        queryClient.invalidateQueries({ queryKey: ["sessions"] });
      } else {
        // Refresh session list to update message count
        queryClient.invalidateQueries({ queryKey: ["sessions"] });
      }
    } catch (err: unknown) {
      if (err instanceof DOMException && err.name === "AbortError") {
        // user cancelled
        if (streamingContent) {
          setMessages((prev) => [
            ...prev,
            {
              id: crypto.randomUUID(),
              role: "assistant",
              content: streamingContent + "\n\n_(생성 중단됨)_",
              createdAt: new Date().toISOString(),
            },
          ]);
        }
      } else {
        setSnackbar({
          open: true,
          message: "메시지 전송에 실패했습니다.",
          severity: "error",
        });
      }
    } finally {
      setIsStreaming(false);
      setStreamingContent("");
      abortRef.current = null;
    }
  }, [
    inputValue,
    isStreaming,
    currentSessionId,
    responseMode,
    selectedModel,
    streamingContent,
    setCurrentSessionId,
    queryClient,
  ]);

  const handleStopGeneration = () => {
    abortRef.current?.abort();
  };

  const handleNewChat = () => {
    setCurrentSessionId(null);
    setMessages([]);
    setStreamingContent("");
  };

  const handleFeedback = (messageId: string, rating: number) => {
    if (!currentSessionId) return;
    feedbackApi
      .submit({
        message_id: messageId,
        session_id: currentSessionId,
        rating,
      })
      .then(() => {
        setSnackbar({
          open: true,
          message: "피드백이 전송되었습니다.",
          severity: "success",
        });
      })
      .catch(() => {
        setSnackbar({
          open: true,
          message: "피드백 전송에 실패했습니다.",
          severity: "error",
        });
      });
  };

  const handleErrorReport = (messageId: string) => {
    setErrorReportDialog({ open: true, messageId });
    setErrorDescription("");
  };

  const handleEditAnswer = (messageId: string, currentContent: string) => {
    setEditingMessageId(messageId);
    setEditedAnswer(currentContent);
    setEvaluationTag("accurate");
    setEditDialogOpen(true);
  };

  const handleSaveGoldenData = async () => {
    if (!currentSessionId || !editingMessageId) return;

    // Find the user message before this assistant message
    const msgIndex = messages.findIndex((m) => m.id === editingMessageId);
    let userQuestion = "";
    if (msgIndex > 0 && messages[msgIndex - 1].role === "user") {
      userQuestion = messages[msgIndex - 1].content;
    }

    try {
      await qualityApi.createGoldenData({
        question: userQuestion,
        answer: editedAnswer,
        source_message_id: editingMessageId,
        source_session_id: currentSessionId,
        evaluation_tag: evaluationTag,
      });
      setSnackbar({
        open: true,
        message: "답변이 Golden Data로 저장되었습니다.",
        severity: "success",
      });
      setEditDialogOpen(false);
    } catch {
      setSnackbar({
        open: true,
        message: "Golden Data 저장에 실패했습니다.",
        severity: "error",
      });
    }
  };

  const handleSubmitErrorReport = () => {
    if (!currentSessionId || !errorReportDialog.messageId) return;
    feedbackApi
      .submitErrorReport({
        message_id: errorReportDialog.messageId,
        session_id: currentSessionId,
        description: errorDescription,
      })
      .then(() => {
        setSnackbar({
          open: true,
          message: "오류 신고가 전송되었습니다.",
          severity: "success",
        });
        setErrorReportDialog({ open: false, messageId: null });
      })
      .catch(() => {
        setSnackbar({
          open: true,
          message: "오류 신고 전송에 실패했습니다.",
          severity: "error",
        });
      });
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
      e.preventDefault();
      handleSend();
    }
  };

  // -----------------------------------------------------------------------
  // Render helpers
  // -----------------------------------------------------------------------

  const sessionGroups = groupSessionsByDate(sessions);

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
      {/* Logo */}
      <Box sx={{ p: 2, display: "flex", alignItems: "center", gap: 1 }}>
        <BotIcon color="primary" />
        <Typography variant="h6" fontWeight={700}>
          Flux RAG
        </Typography>
      </Box>

      {/* New chat button */}
      <Box sx={{ px: 2, pb: 1 }}>
        <Button
          variant="outlined"
          fullWidth
          startIcon={<AddIcon />}
          onClick={handleNewChat}
        >
          새 대화
        </Button>
      </Box>

      <Divider />

      {/* Session list */}
      <Box sx={{ flex: 1, overflow: "auto", px: 1, py: 1 }}>
        {sessionGroups.map((group) => (
          <Box key={group.label} sx={{ mb: 1 }}>
            <Typography
              variant="caption"
              color="text.secondary"
              sx={{ px: 1, fontWeight: 600 }}
            >
              {group.label}
            </Typography>
            <List dense disablePadding>
              {group.items.map((s) => (
                <ListItemButton
                  key={s.id}
                  selected={s.id === currentSessionId}
                  onClick={() => setCurrentSessionId(s.id)}
                  sx={{ borderRadius: 1, mb: 0.25 }}
                >
                  <ListItemText
                    primary={s.title || "새 대화"}
                    secondary={`${s.message_count ?? s.messageCount ?? 0}개 메시지`}
                    primaryTypographyProps={{
                      noWrap: true,
                      variant: "body2",
                    }}
                    secondaryTypographyProps={{
                      variant: "caption",
                    }}
                  />
                  <Tooltip title="삭제">
                    <IconButton
                      size="small"
                      onClick={(e) => {
                        e.stopPropagation();
                        setDeleteDialogId(s.id);
                      }}
                    >
                      <DeleteIcon fontSize="small" />
                    </IconButton>
                  </Tooltip>
                </ListItemButton>
              ))}
            </List>
          </Box>
        ))}
        {sessions.length === 0 && (
          <Typography
            variant="body2"
            color="text.secondary"
            sx={{ p: 2, textAlign: "center" }}
          >
            대화 내역이 없습니다
          </Typography>
        )}
      </Box>

      <Divider />

      {/* Bottom nav */}
      <List dense>
        <ListItemButton selected sx={{ borderRadius: 1, mx: 1 }}>
          <ListItemIcon sx={{ minWidth: 36 }}>
            <ChatIcon fontSize="small" />
          </ListItemIcon>
          <ListItemText primary="채팅" primaryTypographyProps={{ variant: "body2" }} />
        </ListItemButton>
        <ListItemButton
          sx={{ borderRadius: 1, mx: 1 }}
          onClick={() => navigate("/documents")}
        >
          <ListItemIcon sx={{ minWidth: 36 }}>
            <DescriptionIcon fontSize="small" />
          </ListItemIcon>
          <ListItemText primary="문서" primaryTypographyProps={{ variant: "body2" }} />
        </ListItemButton>
        <ListItemButton
          sx={{ borderRadius: 1, mx: 1 }}
          onClick={() => navigate("/admin")}
        >
          <ListItemIcon sx={{ minWidth: 36 }}>
            <AdminIcon fontSize="small" />
          </ListItemIcon>
          <ListItemText primary="관리" primaryTypographyProps={{ variant: "body2" }} />
        </ListItemButton>
        <ListItemButton
          sx={{ borderRadius: 1, mx: 1 }}
          onClick={() => navigate("/monitor")}
        >
          <ListItemIcon sx={{ minWidth: 36 }}>
            <MonitorIcon fontSize="small" />
          </ListItemIcon>
          <ListItemText
            primary="모니터링"
            primaryTypographyProps={{ variant: "body2" }}
          />
        </ListItemButton>
      </List>
    </Box>
  );

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
      <Box
        sx={{
          flex: 1,
          display: "flex",
          flexDirection: "column",
          minWidth: 0,
        }}
      >
        {/* Top bar */}
        <Box
          sx={{
            display: "flex",
            alignItems: "center",
            gap: 2,
            px: 2,
            py: 1,
            borderBottom: "1px solid",
            borderColor: "divider",
            bgcolor: "background.paper",
          }}
        >
          <IconButton onClick={toggleSidebar}>
            <MenuIcon />
          </IconButton>

          <Select
            size="small"
            value={selectedModel}
            onChange={(e: SelectChangeEvent) => setSelectedModel(e.target.value)}
            sx={{ minWidth: 160 }}
            aria-label="AI 모델 선택"
          >
            <MenuItem value="default">기본 모델</MenuItem>
            <MenuItem value="vllm">vLLM</MenuItem>
            <MenuItem value="ollama">Ollama</MenuItem>
            <MenuItem value="openai">OpenAI</MenuItem>
            <MenuItem value="anthropic">Anthropic</MenuItem>
          </Select>

          <Box sx={{ display: "flex", alignItems: "center", gap: 1, minWidth: 200 }}>
            <Typography variant="caption" color="text.secondary">
              Temperature:
            </Typography>
            <Slider
              value={temperature}
              onChange={(_, val) => setTemperature(val as number)}
              min={0}
              max={2}
              step={0.1}
              valueLabelDisplay="auto"
              size="small"
              sx={{ width: 120 }}
              aria-label="Temperature 조절"
            />
            <Typography variant="caption" color="text.secondary">
              {temperature}
            </Typography>
          </Box>

          <ToggleButtonGroup
            size="small"
            exclusive
            value={responseMode}
            onChange={(_e, val: "rag" | "direct" | null) => {
              if (val) setResponseMode(val);
            }}
          >
            <ToggleButton value="rag" aria-label="RAG 모드">RAG</ToggleButton>
            <ToggleButton value="direct" aria-label="직접 모드">직접 응답</ToggleButton>
          </ToggleButtonGroup>

          <Box sx={{ flex: 1 }} />

          {currentSessionId && (
            <Typography variant="caption" color="text.secondary">
              세션: {currentSessionId.slice(0, 8)}...
            </Typography>
          )}
        </Box>

        {/* Chat messages area */}
        <Box sx={{ flex: 1, overflow: "auto", py: 2 }}>
          {/* Announcement banners */}
          {announcementsData?.data?.announcements?.filter((a: any) => a.is_pinned && a.is_active).map((a: any) => (
            <Box key={a.id} sx={{ px: 2, mb: 1 }}>
              <Alert severity="info">
                <strong>{a.title}</strong> — {a.content.slice(0, 100)}{a.content.length > 100 ? "..." : ""}
              </Alert>
            </Box>
          ))}

          {/* Empty state */}
          {messages.length === 0 && !streamingContent && (
            <Box
              sx={{
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                justifyContent: "center",
                height: "100%",
                gap: 3,
              }}
            >
              <BotIcon sx={{ fontSize: 64, color: "primary.main", opacity: 0.5 }} />
              <Typography variant="h5" color="text.secondary">
                무엇이든 물어보세요!
              </Typography>
              <Typography variant="body2" color="text.secondary">
                한국가스기술공사 문서 기반 AI 어시스턴트입니다.
              </Typography>
              <Box
                sx={{
                  display: "flex",
                  flexWrap: "wrap",
                  gap: 1,
                  justifyContent: "center",
                  maxWidth: 600,
                }}
              >
                {SAMPLE_QUESTIONS.map((q) => (
                  <Chip
                    key={q}
                    label={q}
                    variant="outlined"
                    clickable
                    onClick={() => {
                      setInputValue(q);
                    }}
                    sx={{ fontSize: "0.82rem" }}
                  />
                ))}
              </Box>
            </Box>
          )}

          {/* Message list */}
          {messages.map((msg) => (
            <MessageBubble
              key={msg.id}
              msg={msg}
              onFeedback={handleFeedback}
              onErrorReport={handleErrorReport}
              onEditAnswer={handleEditAnswer}
              sessionId={currentSessionId}
              userRole={userRole}
            />
          ))}

          {/* Streaming message */}
          {isStreaming && streamingContent && (
            <Box
              sx={{
                display: "flex",
                justifyContent: "flex-start",
                mb: 2,
                px: 2,
              }}
            >
              <Box sx={{ maxWidth: "75%" }}>
                <Box
                  sx={{
                    display: "flex",
                    alignItems: "center",
                    gap: 0.5,
                    mb: 0.5,
                  }}
                >
                  <BotIcon fontSize="small" color="primary" />
                  <Typography variant="caption" color="text.secondary">
                    Flux RAG
                  </Typography>
                </Box>
                <Paper
                  elevation={1}
                  sx={{ p: 2, borderRadius: 2, bgcolor: "grey.50" }}
                >
                  <Box
                    sx={{
                      "& p": { m: 0, mb: 1 },
                      "& p:last-child": { mb: 0 },
                    }}
                  >
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>
                      {streamingContent}
                    </ReactMarkdown>
                  </Box>
                </Paper>
              </Box>
            </Box>
          )}

          {/* Typing indicator */}
          {isStreaming && !streamingContent && (
            <Box
              sx={{
                display: "flex",
                justifyContent: "flex-start",
                mb: 2,
                px: 2,
              }}
            >
              <Box sx={{ display: "flex", alignItems: "center", gap: 1, p: 2 }}>
                <CircularProgress size={16} />
                <Typography variant="body2" color="text.secondary">
                  응답 생성 중...
                </Typography>
              </Box>
            </Box>
          )}

          <div ref={messagesEndRef} />
        </Box>

        {/* Input area */}
        <Box
          sx={{
            p: 2,
            borderTop: "1px solid",
            borderColor: "divider",
            bgcolor: "background.paper",
          }}
        >
          <Box
            sx={{
              display: "flex",
              gap: 1,
              alignItems: "flex-end",
              maxWidth: 900,
              mx: "auto",
            }}
          >
            <TextField
              fullWidth
              multiline
              maxRows={4}
              placeholder="메시지를 입력하세요... (Cmd+Enter로 전송)"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={isStreaming}
              size="small"
              sx={{
                "& .MuiOutlinedInput-root": {
                  borderRadius: 2,
                },
              }}
              inputProps={{ "aria-label": "메시지 입력" }}
            />
            {isStreaming ? (
              <IconButton
                color="error"
                onClick={handleStopGeneration}
                sx={{
                  bgcolor: "error.light",
                  color: "white",
                  "&:hover": { bgcolor: "error.main" },
                }}
                aria-label="생성 중단"
              >
                <StopIcon />
              </IconButton>
            ) : (
              <IconButton
                color="primary"
                onClick={handleSend}
                disabled={!inputValue.trim()}
                sx={{
                  bgcolor: "primary.main",
                  color: "white",
                  "&:hover": { bgcolor: "primary.dark" },
                  "&.Mui-disabled": {
                    bgcolor: "grey.300",
                    color: "grey.500",
                  },
                }}
                aria-label="메시지 전송"
              >
                <SendIcon />
              </IconButton>
            )}
          </Box>
        </Box>
      </Box>

      {/* Delete confirmation dialog */}
      <Dialog
        open={deleteDialogId !== null}
        onClose={() => setDeleteDialogId(null)}
      >
        <DialogTitle>대화 삭제</DialogTitle>
        <DialogContent>
          <DialogContentText>
            이 대화를 삭제하시겠습니까? 삭제된 대화는 복구할 수 없습니다.
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDeleteDialogId(null)}>취소</Button>
          <Button
            color="error"
            variant="contained"
            onClick={() => {
              if (deleteDialogId) deleteSessionMutation.mutate(deleteDialogId);
            }}
          >
            삭제
          </Button>
        </DialogActions>
      </Dialog>

      {/* Error report dialog */}
      <Dialog
        open={errorReportDialog.open}
        onClose={() => setErrorReportDialog({ open: false, messageId: null })}
      >
        <DialogTitle>오류 신고</DialogTitle>
        <DialogContent>
          <DialogContentText>
            발견하신 오류를 설명해주세요.
          </DialogContentText>
          <TextField
            autoFocus
            margin="dense"
            multiline
            rows={4}
            fullWidth
            placeholder="오류 내용을 입력하세요..."
            value={errorDescription}
            onChange={(e) => setErrorDescription(e.target.value)}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setErrorReportDialog({ open: false, messageId: null })}>
            취소
          </Button>
          <Button
            color="primary"
            variant="contained"
            onClick={handleSubmitErrorReport}
          >
            신고
          </Button>
        </DialogActions>
      </Dialog>

      {/* Snackbar */}
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

      {/* Edit Answer Dialog for Expert Evaluation */}
      <Dialog
        open={editDialogOpen}
        onClose={() => setEditDialogOpen(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>답변 수정 (Golden Data 저장)</DialogTitle>
        <DialogContent>
          <DialogContentText>
            답변을 수정하여 Golden Data로 저장하세요. 이 데이터는 향후 모델 평가 및 개선에 사용됩니다.
          </DialogContentText>
          <TextField
            autoFocus
            margin="dense"
            multiline
            rows={8}
            fullWidth
            placeholder="수정된 답변을 입력하세요..."
            value={editedAnswer}
            onChange={(e) => setEditedAnswer(e.target.value)}
            sx={{ mt: 2 }}
          />
          <Box sx={{ mt: 2 }}>
            <Typography variant="body2" color="text.secondary" gutterBottom>
              평가 태그:
            </Typography>
            <Select
              size="small"
              value={evaluationTag}
              onChange={(e: SelectChangeEvent) => setEvaluationTag(e.target.value)}
              fullWidth
            >
              <MenuItem value="accurate">정확함 (Accurate)</MenuItem>
              <MenuItem value="partial">부분적으로 정확 (Partial)</MenuItem>
              <MenuItem value="inaccurate">부정확 (Inaccurate)</MenuItem>
              <MenuItem value="hallucination">환각 (Hallucination)</MenuItem>
            </Select>
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setEditDialogOpen(false)}>취소</Button>
          <Button
            color="primary"
            variant="contained"
            onClick={handleSaveGoldenData}
            disabled={!editedAnswer.trim()}
          >
            저장
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}
