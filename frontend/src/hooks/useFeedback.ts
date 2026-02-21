import { useState } from "react";
import { feedbackApi } from "../api/client";

export function useFeedback(
  currentSessionId: string | null,
  showSnackbar: (message: string, severity: "success" | "error" | "info") => void,
) {
  const [errorReportDialog, setErrorReportDialog] = useState<{
    open: boolean;
    messageId: string | null;
  }>({ open: false, messageId: null });
  const [errorDescription, setErrorDescription] = useState("");

  const handleFeedback = (messageId: string, rating: number) => {
    if (!currentSessionId) return;
    feedbackApi
      .submit({
        message_id: messageId,
        session_id: currentSessionId,
        rating,
      })
      .then(() => {
        showSnackbar("피드백이 전송되었습니다.", "success");
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
    errorReportDialog,
    errorDescription,
    setErrorDescription,
    openErrorReport,
    closeErrorReport,
    submitErrorReport,
  };
}
