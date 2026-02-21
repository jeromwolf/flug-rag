import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { sessionsApi } from "../api/client";

interface SessionItem {
  id: string;
  title: string;
  created_at?: string;
  createdAt?: string;
  message_count?: number;
  messageCount?: number;
  updated_at?: string;
  updatedAt?: string;
}

function groupSessionsByDate(sessions: SessionItem[]) {
  const now = new Date();
  const todayStart = new Date(now.getFullYear(), now.getMonth(), now.getDate());
  const weekStart = new Date(todayStart);
  weekStart.setDate(weekStart.getDate() - weekStart.getDay());

  const groups: { label: string; items: SessionItem[] }[] = [
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

export function useSessions(
  currentSessionId: string | null,
  setCurrentSessionId: (id: string | null) => void,
) {
  const queryClient = useQueryClient();
  const [deleteDialogId, setDeleteDialogId] = useState<string | null>(null);

  const { data: sessionsData } = useQuery({
    queryKey: ["sessions"],
    queryFn: async () => {
      const res = await sessionsApi.list();
      return res.data;
    },
  });

  const sessions: SessionItem[] = sessionsData?.sessions ?? [];
  const sessionGroups = groupSessionsByDate(sessions);

  const deleteSessionMutation = useMutation({
    mutationFn: (id: string) => sessionsApi.delete(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["sessions"] });
      if (deleteDialogId === currentSessionId) {
        setCurrentSessionId(null);
      }
      setDeleteDialogId(null);
    },
  });

  const deleteSession = (id: string) => {
    deleteSessionMutation.mutate(id);
  };

  return {
    sessions,
    sessionGroups,
    deleteDialogId,
    setDeleteDialogId,
    deleteSession,
    isDeleting: deleteSessionMutation.isPending,
  };
}
