import { useState } from "react";
import { qualityApi } from "../api/client";
import type { Message } from "../types";

export function useGoldenData(
  currentSessionId: string | null,
  messages: Message[],
  showSnackbar: (message: string, severity: "success" | "error" | "info") => void,
) {
  const [editDialogOpen, setEditDialogOpen] = useState(false);
  const [editingMessageId, setEditingMessageId] = useState("");
  const [editedAnswer, setEditedAnswer] = useState("");
  const [evaluationTag, setEvaluationTag] = useState("accurate");

  const openEditDialog = (messageId: string, currentContent: string) => {
    setEditingMessageId(messageId);
    setEditedAnswer(currentContent);
    setEvaluationTag("accurate");
    setEditDialogOpen(true);
  };

  const closeEditDialog = () => {
    setEditDialogOpen(false);
  };

  const saveGoldenData = async () => {
    if (!currentSessionId || !editingMessageId) return;

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
      showSnackbar("답변이 Golden Data로 저장되었습니다.", "success");
      setEditDialogOpen(false);
    } catch {
      showSnackbar("Golden Data 저장에 실패했습니다.", "error");
    }
  };

  return {
    editDialogOpen,
    editedAnswer,
    setEditedAnswer,
    evaluationTag,
    setEvaluationTag,
    openEditDialog,
    closeEditDialog,
    saveGoldenData,
  };
}
