import { useState, useCallback, useEffect, useMemo } from "react";
import {
  Box,
  Button,
  Chip,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  Divider,
  Drawer,
  List,
  ListItem,
  ListItemText,
  Typography,
  useMediaQuery,
  useTheme,
} from "@mui/material";
import { Summarize as SummarizeIcon } from "@mui/icons-material";
import { useQuery } from "@tanstack/react-query";
import { useAppStore } from "../stores/appStore";
import { useAuth } from "../contexts/AuthContext";
import { contentApi, sessionsApi } from "../api/client";
import { CompareView } from "../components/chat/CompareView";

// Hooks
import { useSnackbar } from "../hooks/useSnackbar";
import { useSessions } from "../hooks/useSessions";
import { useFeedback } from "../hooks/useFeedback";
import { useGoldenData } from "../hooks/useGoldenData";
import { useStreamingChat } from "../hooks/useStreamingChat";

// Components
import { ChatSidebar } from "../components/chat/ChatSidebar";
import { ChatTopBar } from "../components/chat/ChatTopBar";
import { ChatMessageList } from "../components/chat/ChatMessageList";
import { ChatInputBar } from "../components/chat/ChatInputBar";
import {
  DeleteSessionDialog,
  ErrorReportDialog,
  FeedbackCommentDialog,
  GoldenDataEditDialog,
  NotificationSnackbar,
} from "../components/chat/ChatDialogs";
import { OnboardingTour } from "../components/OnboardingTour";

const SIDEBAR_WIDTH = 260;

export default function ChatPage() {
  const { user } = useAuth();
  const userRole = user?.role ?? "";

  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down("md"));

  const {
    currentSessionId,
    setCurrentSessionId,
    responseMode,
    setResponseMode,
    selectedModel,
    setSelectedModel,
    temperature,
    setTemperature,
    sidebarOpen,
    toggleSidebar,
    darkMode,
    toggleDarkMode,
    compareMode,
    toggleCompareMode,
  } = useAppStore();

  // Hooks
  const { snackbar, showSnackbar, closeSnackbar } = useSnackbar();

  // Compare mode: track the latest question sent while compare is active
  const [compareQuestion, setCompareQuestion] = useState("");

  // File attachments for ChatInputBar
  const [attachedFiles, setAttachedFiles] = useState<File[]>([]);

  const handleFilesAttached = (files: File[]) => {
    setAttachedFiles((prev) => [...prev, ...files]);
  };

  const handleRemoveFile = (index: number) => {
    setAttachedFiles((prev) => prev.filter((_, i) => i !== index));
  };

  const {
    sessions,
    sessionGroups,
    deleteDialogId,
    setDeleteDialogId,
    deleteSession,
    renameSession,
    isLoading: isSessionsLoading,
  } = useSessions(currentSessionId, setCurrentSessionId);

  const {
    messages,
    setMessages,
    inputValue,
    setInputValue,
    isStreaming,
    isEditing,
    streamingContent,
    messagesEndRef,
    handleSend,
    handleStopGeneration,
    handleNewChat,
    handleKeyDown,
    handleEditUserMessage,
    handleRegenerate,
    toolInProgress,
  } = useStreamingChat({
    currentSessionId,
    setCurrentSessionId,
    responseMode,
    selectedModel,
    temperature,
    showSnackbar,
    attachedFiles,
    onFilesSent: () => setAttachedFiles([]),
  });

  const {
    handleFeedback,
    errorReportDialog,
    errorDescription,
    setErrorDescription,
    openErrorReport,
    closeErrorReport,
    submitErrorReport,
    feedbackDialog,
    feedbackComment,
    setFeedbackComment,
    closeFeedbackDialog,
    submitFeedbackWithComment,
  } = useFeedback(currentSessionId, messages, showSnackbar);

  const {
    editDialogOpen,
    editedAnswer,
    setEditedAnswer,
    evaluationTag,
    setEvaluationTag,
    openEditDialog,
    closeEditDialog,
    saveGoldenData,
  } = useGoldenData(currentSessionId, messages, showSnackbar);

  // Reactions state: messageId → emoji[]
  const [reactions, setReactions] = useState<Map<string, string[]>>(new Map());

  // Load reactions from localStorage when session changes
  useEffect(() => {
    if (!currentSessionId) {
      setReactions(new Map());
      return;
    }
    try {
      const stored = localStorage.getItem(`kogas-ai-reactions-${currentSessionId}`);
      if (stored) {
        const parsed: Record<string, string[]> = JSON.parse(stored);
        setReactions(new Map(Object.entries(parsed)));
      } else {
        setReactions(new Map());
      }
    } catch {
      setReactions(new Map());
    }
  }, [currentSessionId]);

  const handleReact = useCallback(
    (messageId: string, emoji: string) => {
      setReactions((prev) => {
        const next = new Map(prev);
        const current = next.get(messageId) ?? [];
        const idx = current.indexOf(emoji);
        const updated =
          idx >= 0
            ? current.filter((e) => e !== emoji) // toggle off
            : [...current, emoji];               // toggle on
        if (updated.length === 0) {
          next.delete(messageId);
        } else {
          next.set(messageId, updated);
        }
        // Persist to localStorage
        if (currentSessionId) {
          const plain: Record<string, string[]> = {};
          next.forEach((v, k) => {
            plain[k] = v;
          });
          try {
            localStorage.setItem(
              `kogas-ai-reactions-${currentSessionId}`,
              JSON.stringify(plain),
            );
          } catch {
            // storage quota exceeded — ignore
          }
        }
        return next;
      });
    },
    [currentSessionId],
  );

  // Multi-select state
  const [selectMode, setSelectMode] = useState(false);
  const [selectedMessageIds, setSelectedMessageIds] = useState<Set<string>>(new Set());

  // Pinned messages state (per session, resets on session change)
  const [pinnedMessageIds, setPinnedMessageIds] = useState<Set<string>>(new Set());

  // Reset pins when session changes
  useEffect(() => {
    setPinnedMessageIds(new Set());
  }, [currentSessionId]);

  const MAX_PINS = 5;

  const handleTogglePin = useCallback((messageId: string) => {
    setPinnedMessageIds((prev) => {
      const next = new Set(prev);
      if (next.has(messageId)) {
        next.delete(messageId);
      } else {
        if (next.size >= MAX_PINS) {
          showSnackbar(`최대 ${MAX_PINS}개까지 고정할 수 있습니다.`, "info");
          return prev;
        }
        next.add(messageId);
      }
      return next;
    });
  }, [showSnackbar]);

  const handleToggleSelectMode = useCallback(() => {
    setSelectMode((prev) => {
      if (prev) {
        // Exiting select mode: clear selection
        setSelectedMessageIds(new Set());
      }
      return !prev;
    });
  }, []);

  const handleExitSelectMode = useCallback(() => {
    setSelectMode(false);
    setSelectedMessageIds(new Set());
  }, []);

  const handleToggleMessageSelect = useCallback((messageId: string) => {
    setSelectedMessageIds((prev) => {
      const next = new Set(prev);
      if (next.has(messageId)) {
        next.delete(messageId);
      } else {
        next.add(messageId);
      }
      return next;
    });
  }, []);

  // Search state (placed after `messages` is available from useStreamingChat)
  const [searchBarOpen, setSearchBarOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [currentSearchIndex, setCurrentSearchIndex] = useState(0);

  const visibleMessages = useMemo(
    () => messages.filter((m) => m.role !== "system"),
    [messages],
  );

  const searchResults = useMemo<number[]>(() => {
    if (!searchQuery.trim()) return [];
    const lower = searchQuery.toLowerCase();
    return visibleMessages
      .map((m, i) => (m.content.toLowerCase().includes(lower) ? i : -1))
      .filter((i) => i !== -1);
  }, [searchQuery, visibleMessages]);

  // Keep currentSearchIndex in bounds when results change
  useEffect(() => {
    setCurrentSearchIndex(0);
  }, [searchResults.length]);

  const handleSearchNavigate = useCallback(
    (direction: "up" | "down") => {
      if (searchResults.length === 0) return;
      setCurrentSearchIndex((prev) => {
        if (direction === "down") return (prev + 1) % searchResults.length;
        return (prev - 1 + searchResults.length) % searchResults.length;
      });
    },
    [searchResults.length],
  );

  const handleSearchClose = useCallback(() => {
    setSearchQuery("");
    setCurrentSearchIndex(0);
  }, []);

  const handleSearchOpen = useCallback(() => {
    setSearchBarOpen(true);
  }, []);

  const handleSendWithClear = useCallback(() => {
    if (compareMode) {
      const text = inputValue.trim();
      if (text) {
        setCompareQuestion(text);
        setInputValue("");
      }
      setAttachedFiles([]);
    } else {
      handleSend();
      // Files are cleared by onFilesSent callback from the hook after capture
    }
  }, [compareMode, handleSend, inputValue, setInputValue]);

  // Copy conversation — mirrors ChatTopBar logic, needed for keyboard shortcut
  const handleCopyConversation = useCallback(async () => {
    if (messages.length === 0) {
      showSnackbar("복사할 대화가 없습니다.", "info");
      return;
    }
    const text = messages
      .filter((m) => m.role !== "system")
      .map((m) => {
        const label = m.role === "user" ? "사용자" : "AI 어시스턴트";
        return `${label}: ${m.content}`;
      })
      .join("\n\n");
    try {
      await navigator.clipboard.writeText(text);
      showSnackbar("대화가 클립보드에 복사되었습니다.", "success");
    } catch {
      showSnackbar("복사에 실패했습니다.", "error");
    }
  }, [messages, showSnackbar]);

  // Download conversation — mirrors ChatTopBar logic, needed for keyboard shortcut
  const handleDownloadConversation = useCallback(() => {
    if (messages.length === 0) {
      showSnackbar("내보낼 대화가 없습니다.", "info");
      return;
    }
    const now = new Date();
    const dateStr = now.toLocaleDateString("ko-KR", {
      year: "numeric",
      month: "2-digit",
      day: "2-digit",
    });
    const formatTime = (isoStr: string) => {
      const d = new Date(isoStr);
      if (isNaN(d.getTime())) return "";
      return d.toLocaleTimeString("ko-KR", { hour: "2-digit", minute: "2-digit" });
    };
    const lines: string[] = ["# AI 대화 기록", `날짜: ${dateStr}`, "", "---"];
    messages
      .filter((m) => m.role !== "system")
      .forEach((m) => {
        lines.push("");
        const label = m.role === "user" ? "사용자" : "AI 어시스턴트";
        const timeStr = formatTime(m.createdAt);
        lines.push(`## ${label}${timeStr ? ` (${timeStr})` : ""}`);
        lines.push(m.content);
        if (m.role === "assistant") {
          if (m.sources && m.sources.length > 0) {
            const srcList = m.sources
              .map((s) => `${s.filename}${s.page != null ? ` (p.${s.page})` : ""}`)
              .join(", ");
            lines.push(`**출처:** ${srcList}`);
          }
          const metaParts: string[] = [];
          if (m.confidenceScore != null) metaParts.push(`**신뢰도:** ${m.confidenceScore.toFixed(2)}`);
          if (m.modelUsed) metaParts.push(`**모델:** ${m.modelUsed}`);
          if (m.latencyMs != null) metaParts.push(`**응답시간:** ${(m.latencyMs / 1000).toFixed(1)}s`);
          if (metaParts.length > 0) lines.push(metaParts.join(" | "));
        }
        lines.push("");
        lines.push("---");
      });
    const blob = new Blob([lines.join("\n")], { type: "text/markdown;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const fileDate = now
      .toLocaleDateString("ko-KR", { year: "numeric", month: "2-digit", day: "2-digit" })
      .replace(/\./g, "")
      .replace(/\s/g, "_");
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `AI_대화_${fileDate}.md`;
    anchor.click();
    URL.revokeObjectURL(url);
    showSnackbar("대화가 파일로 저장되었습니다.", "success");
  }, [messages, showSnackbar]);

  // Fork conversation at a given message: create new session with messages up to and including that message
  const handleForkConversation = useCallback(
    async (messageId: string) => {
      if (isStreaming) return;

      const visibleMessages = messages.filter((m) => m.role !== "system");
      const forkIndex = visibleMessages.findIndex((m) => m.id === messageId);
      if (forkIndex === -1) return;

      const forkedMessages = visibleMessages.slice(0, forkIndex + 1);

      // Determine title: use original session title if available, append "(분기)"
      const currentSession = sessions.find((s) => s.id === currentSessionId);
      const baseTitle = currentSession?.title ?? "대화";
      const forkTitle = `${baseTitle} (분기)`;

      try {
        // Create a new session on the backend
        const res = await sessionsApi.create(forkTitle);
        const newSessionId: string = res.data?.session_id ?? res.data?.id ?? "";
        if (!newSessionId) throw new Error("세션 생성 실패");

        // Switch to the new session (skip reload since we set messages manually)
        setCurrentSessionId(newSessionId);
        setMessages(forkedMessages);
        showSnackbar(`분기 대화가 생성되었습니다: "${forkTitle}"`, "success");
      } catch {
        showSnackbar("분기 생성에 실패했습니다.", "error");
      }
    },
    [isStreaming, messages, sessions, currentSessionId, setCurrentSessionId, setMessages, showSnackbar],
  );

  const handleSearchQueryChange = useCallback(
    (q: string) => {
      setSearchQuery(q);
      if (!searchBarOpen) setSearchBarOpen(true);
    },
    [searchBarOpen],
  );

  const handleSearchCloseWithBar = useCallback(() => {
    handleSearchClose();
    setSearchBarOpen(false);
  }, [handleSearchClose]);

  // Summary dialog state
  const [summaryDialogOpen, setSummaryDialogOpen] = useState(false);

  const handleSummarize = useCallback(() => {
    setSummaryDialogOpen(true);
  }, []);

  const handleCloseSummaryDialog = useCallback(() => {
    setSummaryDialogOpen(false);
  }, []);

  // Derived summary data (computed on demand inside dialog, but memoised here for perf)
  const summaryData = useMemo(() => {
    const visible = messages.filter((m) => m.role !== "system");
    const userMessages = visible.filter((m) => m.role === "user");
    const assistantMessages = visible.filter((m) => m.role === "assistant");

    // Duration
    let durationStr = "-";
    if (visible.length >= 2) {
      const first = new Date(visible[0].createdAt);
      const last = new Date(visible[visible.length - 1].createdAt);
      if (!isNaN(first.getTime()) && !isNaN(last.getTime())) {
        const diffMs = last.getTime() - first.getTime();
        const diffMin = Math.floor(diffMs / 60000);
        const diffSec = Math.floor((diffMs % 60000) / 1000);
        if (diffMin > 0) {
          durationStr = `${diffMin}분 ${diffSec}초`;
        } else {
          durationStr = `${diffSec}초`;
        }
      }
    }

    // Topics: first 5 words (Korean-aware) of each user question
    const topics = userMessages.map((m) => {
      const words = m.content.trim().split(/\s+/).slice(0, 5).join(" ");
      return words.length < m.content.trim().length ? words + "…" : words;
    });

    // Unique source filenames
    const sourceSet = new Set<string>();
    assistantMessages.forEach((m) => {
      (m.sources ?? []).forEach((s) => {
        if (s.filename) sourceSet.add(s.filename);
      });
    });
    const uniqueSources = Array.from(sourceSet);

    // Average confidence
    const scores = assistantMessages
      .map((m) => m.confidenceScore)
      .filter((s): s is number => s != null);
    const avgConfidence =
      scores.length > 0
        ? scores.reduce((a, b) => a + b, 0) / scores.length
        : null;

    return {
      totalCount: visible.length,
      userCount: userMessages.length,
      assistantCount: assistantMessages.length,
      durationStr,
      topics,
      uniqueSources,
      avgConfidence,
    };
  }, [messages]);

  const handleCopySummary = useCallback(async () => {
    const lines: string[] = [
      "=== 대화 요약 ===",
      `전체 메시지: ${summaryData.totalCount}개 (사용자 ${summaryData.userCount}개 / AI ${summaryData.assistantCount}개)`,
      `대화 시간: ${summaryData.durationStr}`,
    ];
    if (summaryData.avgConfidence != null) {
      lines.push(`평균 신뢰도: ${(summaryData.avgConfidence * 100).toFixed(1)}%`);
    }
    if (summaryData.topics.length > 0) {
      lines.push("", "주요 질문:");
      summaryData.topics.forEach((t, i) => lines.push(`  ${i + 1}. ${t}`));
    }
    if (summaryData.uniqueSources.length > 0) {
      lines.push("", "참조 문서:");
      summaryData.uniqueSources.forEach((s) => lines.push(`  - ${s}`));
    }
    try {
      await navigator.clipboard.writeText(lines.join("\n"));
      showSnackbar("요약이 클립보드에 복사되었습니다.", "success");
    } catch {
      showSnackbar("복사에 실패했습니다.", "error");
    }
  }, [summaryData, showSnackbar]);

  const handleAiSummaryRequest = useCallback(() => {
    setSummaryDialogOpen(false);
    setInputValue("지금까지의 대화를 요약해 주세요");
    // Small delay to let dialog close before auto-send feels natural
    setTimeout(() => {
      handleSend();
    }, 100);
  }, [setInputValue, handleSend]);

  // Global keyboard shortcuts
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const isMac = e.metaKey;
      const isCtrl = e.ctrlKey;
      const tag = (document.activeElement as HTMLElement | null)?.tagName?.toLowerCase();
      const inInput = tag === "input" || tag === "textarea";

      // Ctrl+N / Cmd+N — new chat (works everywhere)
      if ((isCtrl || isMac) && !e.shiftKey && e.key === "n") {
        e.preventDefault();
        handleNewChat();
        setAttachedFiles([]);
        return;
      }

      // Escape — exit select mode, close search if open, or stop generation
      if (e.key === "Escape") {
        if (selectMode) {
          handleExitSelectMode();
          return;
        }
        if (searchBarOpen) {
          handleSearchCloseWithBar();
          return;
        }
        if (isStreaming) {
          handleStopGeneration();
          return;
        }
      }

      // Ctrl+F / Cmd+F — open search bar
      if ((isCtrl || isMac) && !e.shiftKey && e.key.toLowerCase() === "f") {
        e.preventDefault();
        setSearchBarOpen(true);
        return;
      }

      // Remaining shortcuts: skip when focus is in input/textarea
      if (inInput) return;

      // Ctrl+Shift+C / Cmd+Shift+C — copy conversation
      if ((isCtrl || isMac) && e.shiftKey && e.key.toLowerCase() === "c") {
        e.preventDefault();
        handleCopyConversation();
        return;
      }

      // Ctrl+Shift+S / Cmd+Shift+S — download/export conversation
      if ((isCtrl || isMac) && e.shiftKey && e.key.toLowerCase() === "s") {
        e.preventDefault();
        handleDownloadConversation();
        return;
      }
    };

    document.addEventListener("keydown", handler);
    return () => document.removeEventListener("keydown", handler);
  }, [
    handleNewChat,
    handleStopGeneration,
    handleCopyConversation,
    handleDownloadConversation,
    isStreaming,
    searchBarOpen,
    handleSearchCloseWithBar,
    selectMode,
    handleExitSelectMode,
  ]);

  // Announcements
  const { data: announcementsData } = useQuery({
    queryKey: ["announcements"],
    queryFn: () => contentApi.listAnnouncements(),
    staleTime: 60000,
  });

  return (
    <Box sx={{ display: "flex", height: "100vh" }}>
      {/* Sidebar */}
      <Drawer
        variant={isMobile ? "temporary" : "persistent"}
        open={isMobile ? sidebarOpen : sidebarOpen}
        onClose={toggleSidebar}
        ModalProps={{ keepMounted: true }}
        sx={{
          width: !isMobile && sidebarOpen ? SIDEBAR_WIDTH : 0,
          flexShrink: 0,
          transition: "width 0.2s",
          "& .MuiDrawer-paper": {
            width: SIDEBAR_WIDTH,
            boxSizing: "border-box",
            border: "none",
            bgcolor: "#171717",
            overflowX: "hidden",
          },
        }}
      >
        <ChatSidebar
          sessionGroups={sessionGroups}
          sessionsEmpty={sessions.length === 0}
          isSessionsLoading={isSessionsLoading}
          currentSessionId={currentSessionId}
          onSelectSession={(id) => {
            setCurrentSessionId(id);
            // Close drawer on mobile after selecting a session
            if (isMobile && sidebarOpen) toggleSidebar();
          }}
          onNewChat={() => {
            handleNewChat();
            setAttachedFiles([]);
            if (isMobile && sidebarOpen) toggleSidebar();
          }}
          onDeleteSession={setDeleteDialogId}
          onRenameSession={renameSession}
        />
      </Drawer>

      {/* Main content */}
      <Box sx={{ flex: 1, display: "flex", flexDirection: "column", minWidth: 0 }}>
        <ChatTopBar
          onToggleSidebar={toggleSidebar}
          selectedModel={selectedModel}
          onModelChange={setSelectedModel}
          temperature={temperature}
          onTemperatureChange={setTemperature}
          responseMode={responseMode}
          onModeChange={setResponseMode}
          darkMode={darkMode}
          onToggleDarkMode={toggleDarkMode}
          messages={messages}
          onShowSnackbar={showSnackbar}
          compareMode={compareMode}
          onToggleCompareMode={toggleCompareMode}
          searchOpen={searchBarOpen}
          onSearchOpen={handleSearchOpen}
          searchQuery={searchQuery}
          onSearchQueryChange={handleSearchQueryChange}
          searchResults={searchResults}
          currentSearchIndex={currentSearchIndex}
          onSearchNavigate={handleSearchNavigate}
          onSearchClose={handleSearchCloseWithBar}
          sessionId={currentSessionId}
          selectMode={selectMode}
          onToggleSelectMode={handleToggleSelectMode}
          onSummarize={handleSummarize}
        />

        {compareMode ? (
          <CompareView
            question={compareQuestion}
            isActive={compareMode}
            mainModel={selectedModel !== "default" ? selectedModel : undefined}
          />
        ) : (
          <ChatMessageList
            messages={messages}
            streamingContent={streamingContent}
            isStreaming={isStreaming}
            isEditing={isEditing}
            toolInProgress={toolInProgress}
            announcements={announcementsData}
            onSampleClick={setInputValue}
            onFeedback={handleFeedback}
            onErrorReport={openErrorReport}
            onEditAnswer={openEditDialog}
            onEditUserMessage={handleEditUserMessage}
            onRegenerate={handleRegenerate}
            currentSessionId={currentSessionId}
            userRole={userRole}
            messagesEndRef={messagesEndRef}
            searchHighlightIndices={searchResults}
            currentSearchHighlightIndex={currentSearchIndex}
            selectMode={selectMode}
            selectedMessageIds={selectedMessageIds}
            onToggleSelect={handleToggleMessageSelect}
            onExitSelectMode={handleExitSelectMode}
            onShowSnackbar={showSnackbar}
            pinnedMessageIds={pinnedMessageIds}
            onTogglePin={handleTogglePin}
            onFork={handleForkConversation}
            reactions={reactions}
            onReact={handleReact}
          />
        )}

        {/* TODO: attach files to chat request once backend /chat/stream supports multipart upload */}
        <ChatInputBar
          inputValue={inputValue}
          onInputChange={setInputValue}
          onSend={handleSendWithClear}
          onStop={handleStopGeneration}
          onKeyDown={handleKeyDown}
          isStreaming={isStreaming}
          attachedFiles={attachedFiles}
          onFilesAttached={handleFilesAttached}
          onRemoveFile={handleRemoveFile}
        />
      </Box>

      {/* Dialogs */}
      <DeleteSessionDialog
        open={deleteDialogId !== null}
        onClose={() => setDeleteDialogId(null)}
        onConfirm={() => {
          if (deleteDialogId) deleteSession(deleteDialogId);
        }}
      />

      <ErrorReportDialog
        open={errorReportDialog.open}
        description={errorDescription}
        onDescriptionChange={setErrorDescription}
        onClose={closeErrorReport}
        onSubmit={submitErrorReport}
      />

      <FeedbackCommentDialog
        open={feedbackDialog.open}
        rating={feedbackDialog.rating}
        comment={feedbackComment}
        onCommentChange={setFeedbackComment}
        onClose={closeFeedbackDialog}
        onSubmit={submitFeedbackWithComment}
      />

      <GoldenDataEditDialog
        open={editDialogOpen}
        editedAnswer={editedAnswer}
        evaluationTag={evaluationTag}
        onAnswerChange={setEditedAnswer}
        onTagChange={setEvaluationTag}
        onClose={closeEditDialog}
        onSave={saveGoldenData}
      />

      <NotificationSnackbar snackbar={snackbar} onClose={closeSnackbar} />

      {/* Onboarding tour — shown automatically on first visit */}
      <OnboardingTour />

      {/* Summary Dialog */}
      <Dialog
        open={summaryDialogOpen}
        onClose={handleCloseSummaryDialog}
        maxWidth="sm"
        fullWidth
        PaperProps={{ sx: { borderRadius: 3 } }}
      >
        <DialogTitle sx={{ pb: 1 }}>
          <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
            <SummarizeIcon color="primary" fontSize="small" />
            <Typography variant="h6" component="span" fontWeight={600}>
              대화 요약
            </Typography>
          </Box>
        </DialogTitle>

        <DialogContent dividers sx={{ p: 3 }}>
          {/* Stats section */}
          <Box sx={{ mb: 2.5 }}>
            <Typography
              variant="subtitle2"
              fontWeight={600}
              gutterBottom
              color="text.secondary"
              sx={{ textTransform: "uppercase", fontSize: "0.7rem", letterSpacing: 0.8 }}
            >
              통계
            </Typography>
            <Box
              sx={{
                display: "grid",
                gridTemplateColumns: "1fr 1fr",
                gap: 1.5,
              }}
            >
              <Box
                sx={{
                  p: 1.5,
                  borderRadius: 2,
                  bgcolor: "action.hover",
                  display: "flex",
                  flexDirection: "column",
                  gap: 0.25,
                }}
              >
                <Typography variant="caption" color="text.secondary">
                  전체 메시지
                </Typography>
                <Typography variant="body1" fontWeight={600}>
                  {summaryData.totalCount}개
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  사용자 {summaryData.userCount} · AI {summaryData.assistantCount}
                </Typography>
              </Box>
              <Box
                sx={{
                  p: 1.5,
                  borderRadius: 2,
                  bgcolor: "action.hover",
                  display: "flex",
                  flexDirection: "column",
                  gap: 0.25,
                }}
              >
                <Typography variant="caption" color="text.secondary">
                  대화 시간
                </Typography>
                <Typography variant="body1" fontWeight={600}>
                  {summaryData.durationStr}
                </Typography>
              </Box>
              {summaryData.avgConfidence != null && (
                <Box
                  sx={{
                    p: 1.5,
                    borderRadius: 2,
                    bgcolor: "action.hover",
                    display: "flex",
                    flexDirection: "column",
                    gap: 0.25,
                  }}
                >
                  <Typography variant="caption" color="text.secondary">
                    평균 신뢰도
                  </Typography>
                  <Typography
                    variant="body1"
                    fontWeight={600}
                    color={
                      summaryData.avgConfidence >= 0.8
                        ? "success.main"
                        : summaryData.avgConfidence >= 0.3
                          ? "warning.main"
                          : "error.main"
                    }
                  >
                    {(summaryData.avgConfidence * 100).toFixed(1)}%
                  </Typography>
                </Box>
              )}
            </Box>
          </Box>

          {/* Topics section */}
          {summaryData.topics.length > 0 && (
            <>
              <Divider sx={{ my: 2 }} />
              <Box sx={{ mb: 2.5 }}>
                <Typography
                  variant="subtitle2"
                  fontWeight={600}
                  gutterBottom
                  color="text.secondary"
                  sx={{ textTransform: "uppercase", fontSize: "0.7rem", letterSpacing: 0.8, mb: 1 }}
                >
                  주요 질문
                </Typography>
                <List dense disablePadding>
                  {summaryData.topics.map((topic, i) => (
                    <ListItem
                      key={i}
                      disableGutters
                      sx={{ py: 0.25, alignItems: "flex-start" }}
                    >
                      <ListItemText
                        primary={
                          <Box sx={{ display: "flex", gap: 1, alignItems: "flex-start" }}>
                            <Typography
                              component="span"
                              variant="caption"
                              color="primary.main"
                              fontWeight={600}
                              sx={{ mt: 0.1, flexShrink: 0 }}
                            >
                              {i + 1}.
                            </Typography>
                            <Typography variant="body2" component="span">
                              {topic}
                            </Typography>
                          </Box>
                        }
                      />
                    </ListItem>
                  ))}
                </List>
              </Box>
            </>
          )}

          {/* Sources section */}
          {summaryData.uniqueSources.length > 0 && (
            <>
              <Divider sx={{ my: 2 }} />
              <Box>
                <Typography
                  variant="subtitle2"
                  fontWeight={600}
                  gutterBottom
                  color="text.secondary"
                  sx={{ textTransform: "uppercase", fontSize: "0.7rem", letterSpacing: 0.8, mb: 1 }}
                >
                  참조 문서 ({summaryData.uniqueSources.length}개)
                </Typography>
                <Box sx={{ display: "flex", flexWrap: "wrap", gap: 0.75 }}>
                  {summaryData.uniqueSources.map((src) => (
                    <Chip
                      key={src}
                      label={src}
                      size="small"
                      variant="outlined"
                      sx={{ fontSize: "0.75rem", maxWidth: 260 }}
                    />
                  ))}
                </Box>
              </Box>
            </>
          )}
        </DialogContent>

        <DialogActions sx={{ px: 3, py: 2, gap: 1 }}>
          <Button
            variant="contained"
            size="small"
            onClick={handleAiSummaryRequest}
            startIcon={<SummarizeIcon fontSize="small" />}
          >
            AI 요약 요청
          </Button>
          <Button variant="outlined" size="small" onClick={handleCopySummary}>
            복사
          </Button>
          <Button size="small" onClick={handleCloseSummaryDialog}>
            닫기
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}
