import { useState, useRef, useEffect, useCallback } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { sessionsApi, API_BASE, getAuthHeaders } from "../api/client";
import type { Message, Source } from "../types";

interface UseStreamingChatOptions {
  currentSessionId: string | null;
  setCurrentSessionId: (id: string | null) => void;
  responseMode: "rag" | "direct";
  selectedModel: string;
  temperature: number;
  showSnackbar: (message: string, severity: "success" | "error" | "info") => void;
}

export function useStreamingChat({
  currentSessionId,
  setCurrentSessionId,
  responseMode,
  selectedModel,
  temperature,
  showSnackbar,
}: UseStreamingChatOptions) {
  const queryClient = useQueryClient();
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [streamingContent, setStreamingContent] = useState("");

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);
  const streamingContentRef = useRef("");

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
    }).catch(() => {
      setMessages([]);
    });
  }, [currentSessionId]);

  // Auto-scroll
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, streamingContent]);

  const handleSend = useCallback(async () => {
    const text = inputValue.trim();
    if (!text || isStreaming) return;

    setInputValue("");

    const userMsg: Message = {
      id: crypto.randomUUID(),
      role: "user",
      content: text,
      createdAt: new Date().toISOString(),
    };
    setMessages((prev) => [...prev, userMsg]);
    setIsStreaming(true);
    setStreamingContent("");
    streamingContentRef.current = "";

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
                  streamingContentRef.current = content;
                  setStreamingContent(content);
                  break;
                case "end":
                  confidence = data.confidence_score ?? data.confidence ?? 0;
                  latencyMs = data.latency_ms ?? 0;
                  break;
                case "error":
                  showSnackbar(data.message ?? "응답 생성 중 오류가 발생했습니다.", "error");
                  break;
              }
            } catch {
              // ignore malformed JSON lines
            }
          }
        }
      }

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

      if (newSessionId && newSessionId !== currentSessionId) {
        setCurrentSessionId(newSessionId);
      }
      queryClient.invalidateQueries({ queryKey: ["sessions"] });
    } catch (err: unknown) {
      if (err instanceof DOMException && err.name === "AbortError") {
        if (streamingContentRef.current) {
          setMessages((prev) => [
            ...prev,
            {
              id: crypto.randomUUID(),
              role: "assistant",
              content: streamingContentRef.current + "\n\n_(생성 중단됨)_",
              createdAt: new Date().toISOString(),
            },
          ]);
        }
      } else {
        showSnackbar("메시지 전송에 실패했습니다.", "error");
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
    temperature,
    setCurrentSessionId,
    queryClient,
    showSnackbar,
  ]);

  const handleStopGeneration = () => {
    abortRef.current?.abort();
  };

  const handleNewChat = () => {
    setCurrentSessionId(null);
    setMessages([]);
    setStreamingContent("");
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleEditUserMessage = useCallback(
    (messageId: string, content: string) => {
      if (isStreaming) return;
      const idx = messages.findIndex((m) => m.id === messageId);
      if (idx === -1) return;
      setMessages(messages.slice(0, idx));
      setInputValue(content);
    },
    [messages, isStreaming]
  );

  return {
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
  };
}
