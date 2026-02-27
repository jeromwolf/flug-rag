import { useState, useCallback } from "react";
import { Box, Drawer, useMediaQuery, useTheme } from "@mui/material";
import { useQuery } from "@tanstack/react-query";
import { useAppStore } from "../stores/appStore";
import { useAuth } from "../contexts/AuthContext";
import { contentApi } from "../api/client";
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
  GoldenDataEditDialog,
  NotificationSnackbar,
} from "../components/chat/ChatDialogs";

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
  } = useSessions(currentSessionId, setCurrentSessionId);

  const {
    messages,
    inputValue,
    setInputValue,
    isStreaming,
    streamingContent,
    messagesEndRef,
    handleSend,
    handleStopGeneration,
    handleNewChat,
    handleKeyDown,
    handleEditUserMessage,
  } = useStreamingChat({
    currentSessionId,
    setCurrentSessionId,
    responseMode,
    selectedModel,
    temperature,
    showSnackbar,
  });

  const {
    handleFeedback,
    errorReportDialog,
    errorDescription,
    setErrorDescription,
    openErrorReport,
    closeErrorReport,
    submitErrorReport,
  } = useFeedback(currentSessionId, showSnackbar);

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

  // Clear files after send (files are display-only until backend supports upload)
  const handleSendWithClear = useCallback(() => {
    if (compareMode) {
      // In compare mode, capture question and trigger compare view
      const text = inputValue.trim();
      if (text) {
        setCompareQuestion(text);
        setInputValue("");
      }
    } else {
      handleSend();
    }
    setAttachedFiles([]);
  }, [compareMode, handleSend, inputValue, setInputValue]);

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
        open={sidebarOpen}
        onClose={toggleSidebar}
        sx={{
          width: sidebarOpen ? SIDEBAR_WIDTH : 0,
          flexShrink: 0,
          "& .MuiDrawer-paper": {
            width: SIDEBAR_WIDTH,
            boxSizing: "border-box",
            border: "none",
            bgcolor: "#171717",
          },
        }}
      >
        <ChatSidebar
          sessionGroups={sessionGroups}
          sessionsEmpty={sessions.length === 0}
          currentSessionId={currentSessionId}
          onSelectSession={setCurrentSessionId}
          onNewChat={handleNewChat}
          onDeleteSession={setDeleteDialogId}
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
            announcements={announcementsData}
            onSampleClick={setInputValue}
            onFeedback={handleFeedback}
            onErrorReport={openErrorReport}
            onEditAnswer={openEditDialog}
            onEditUserMessage={handleEditUserMessage}
            currentSessionId={currentSessionId}
            userRole={userRole}
            messagesEndRef={messagesEndRef}
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
    </Box>
  );
}
