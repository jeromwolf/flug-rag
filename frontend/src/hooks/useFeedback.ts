import { useState } from "react";
import { feedbackApi } from "../api/client";
import type { Message } from "../types";

export function useFeedback(
  currentSessionId: string | null,
  messages: Message[],
  showSnackbar: (message: string, severity: "success" | "error" | "info") => void,
) {
  const [errorReportDialog, setErrorReportDialog] = useState<{
    open: boolean;
    messageId: string | null;
  }>({ open: false, messageId: null });
  const [errorDescription, setErrorDescription] = useState("");

  // Feedback comment dialog state (for negative/partial feedback)
  const [feedbackDialog, setFeedbackDialog] = useState<{
    open: boolean;
    messageId: string | null;
    rating: number;
    query: string;
    answer: string;
  }>({ open: false, messageId: null, rating: 0, query: "", answer: "" });
  const [feedbackComment, setFeedbackComment] = useState("");

  /**
   * Find the user question that preceded a given AI message.
   */
  const findQueryAnswer = (messageId: string): { query: string; answer: string } => {
    const idx = messages.findIndex((m) => m.id === messageId);
    if (idx < 0) return { query: "", answer: "" };

    const aiMsg = messages[idx];
    const answer = aiMsg.content || "";

    // Walk backwards to find the preceding user message
    let query = "";
    for (let i = idx - 1; i >= 0; i--) {
      if (messages[i].role === "user") {
        query = messages[i].content || "";
        break;
      }
    }

    return { query, answer };
  };

  const handleFeedback = (messageId: string, rating: number) => {
    if (!currentSessionId) return;

    const { query, answer } = findQueryAnswer(messageId);

    if (rating === 1) {
      // Positive feedback: submit immediately
      feedbackApi
        .submit({
          message_id: messageId,
          session_id: currentSessionId,
          rating,
          query,
          answer,
        })
        .then(() => {
          showSnackbar("피드백이 전송되었습니다.", "success");
        })
        .catch(() => {
          showSnackbar("피드백 전송에 실패했습니다.", "error");
        });
    } else {
      // Negative or partial feedback: open comment dialog
      setFeedbackDialog({
        open: true,
        messageId,
        rating,
        query,
        answer,
      });
      setFeedbackComment("");
    }
  };

  const closeFeedbackDialog = () => {
    setFeedbackDialog({ open: false, messageId: null, rating: 0, query: "", answer: "" });
    setFeedbackComment("");
  };

  const submitFeedbackWithComment = () => {
    if (!currentSessionId || !feedbackDialog.messageId) return;

    feedbackApi
      .submit({
        message_id: feedbackDialog.messageId,
        session_id: currentSessionId,
        rating: feedbackDialog.rating,
        comment: feedbackComment,
        query: feedbackDialog.query,
        answer: feedbackDialog.answer,
      })
      .then(() => {
        showSnackbar("피드백이 전송되었습니다.", "success");
        closeFeedbackDialog();
      })
      .catch(() => {
        showSnackbar("피드백 전송에 실패했습니다.", "error");
      });
  };

  const openErrorReport = (messageId: string) => {
    setErrorReportDialog({ open: true, messageId });
    setErrorDescription("");
  };

  const closeErrorReport = () => {
    setErrorReportDialog({ open: false, messageId: null });
  };

  const submitErrorReport = () => {
    if (!currentSessionId || !errorReportDialog.messageId) return;
    feedbackApi
      .submitErrorReport({
        message_id: errorReportDialog.messageId,
        session_id: currentSessionId,
        description: errorDescription,
      })
      .then(() => {
        showSnackbar("오류 신고가 전송되었습니다.", "success");
        closeErrorReport();
      })
      .catch(() => {
        showSnackbar("오류 신고 전송에 실패했습니다.", "error");
      });
  };

  return {
    handleFeedback,
    // Error report dialog
    errorReportDialog,
    errorDescription,
    setErrorDescription,
    openErrorReport,
    closeErrorReport,
    submitErrorReport,
    // Feedback comment dialog
    feedbackDialog,
    feedbackComment,
    setFeedbackComment,
    closeFeedbackDialog,
    submitFeedbackWithComment,
  };
}
