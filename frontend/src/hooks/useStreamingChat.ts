import { useState, useRef, useEffect, useCallback } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { sessionsApi, ocrApi, API_BASE, getAuthHeaders } from "../api/client";
import type { Message, Source } from "../types";

interface UseStreamingChatOptions {
  currentSessionId: string | null;
  setCurrentSessionId: (id: string | null) => void;
  responseMode: "rag" | "direct";
  selectedModel: string;
  temperature: number;
  showSnackbar: (message: string, severity: "success" | "error" | "info") => void;
  attachedFiles?: File[];
  onFilesSent?: () => void;
}

export function useStreamingChat({
  currentSessionId,
  setCurrentSessionId,
  responseMode,
  selectedModel,
  temperature,
  showSnackbar,
  attachedFiles,
  onFilesSent,
}: UseStreamingChatOptions) {
  const queryClient = useQueryClient();
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [streamingContent, setStreamingContent] = useState("");
  const [toolInProgress, setToolInProgress] = useState<string | null>(null);
  const [isOcrProcessing, setIsOcrProcessing] = useState(false);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);
  const streamingContentRef = useRef("");
  const skipNextReloadRef = useRef(false);
  // OCR 텍스트를 세션 내에서 유지 (후속 질문에서도 사용)
  const ocrContextRef = useRef("");

  // Load messages when session changes (skip if we just set it from streaming)
  useEffect(() => {
    if (skipNextReloadRef.current) {
      skipNextReloadRef.current = false;
      return;
    }
    if (!currentSessionId) {
      setMessages([]);
      ocrContextRef.current = "";
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
    setIsEditing(false);

    // Capture files before clearing
    const filesToProcess = attachedFiles?.length ? [...attachedFiles] : [];
    onFilesSent?.();

    const userMsg: Message = {
      id: crypto.randomUUID(),
      role: "user",
      content: text,
      attachedFileNames: filesToProcess.length > 0 ? filesToProcess.map(f => f.name) : undefined,
      createdAt: new Date().toISOString(),
    };
    setMessages((prev) => [...prev, userMsg]);
    setIsStreaming(true);
    setStreamingContent("");
    streamingContentRef.current = "";

    const abortController = new AbortController();
    abortRef.current = abortController;

    try {
      // Process OCR for newly attached files
      let newOcrContext = "";
      if (filesToProcess.length > 0) {
        setIsOcrProcessing(true);
        setToolInProgress("첨부파일 OCR 처리 중...");
        for (const file of filesToProcess) {
          try {
            const res = await ocrApi.process(file);
            const ocrText = res.data?.text || "";
            if (ocrText) {
              newOcrContext += `\n\n[첨부파일: ${file.name}]\n${ocrText}`;
            } else {
              showSnackbar(`${file.name}: OCR 결과가 비어있습니다.`, "info");
            }
          } catch {
            showSnackbar(`${file.name} OCR 처리 실패`, "error");
          }
        }
        setIsOcrProcessing(false);
        setToolInProgress(null);
        // 새 OCR 텍스트를 세션 컨텍스트에 누적 (최대 30,000자 — 초과 시 최신 파일만 유지)
        const MAX_OCR_CHARS = 30000;
        if (ocrContextRef.current.length + newOcrContext.length > MAX_OCR_CHARS) {
          ocrContextRef.current = newOcrContext;
        } else {
          ocrContextRef.current += newOcrContext;
        }
      }

      // 현재 세션에 OCR 컨텍스트가 있으면 사용 (새 파일 또는 이전 파일)
      const activeOcrContext = ocrContextRef.current;
      const hasOcr = activeOcrContext.length > 0;
      const messageWithOcr = hasOcr
        ? `다음은 사용자가 첨부한 문서의 OCR 텍스트입니다. 이 내용을 기반으로 질문에 답변하세요.\n\n--- 첨부파일 내용 ---${activeOcrContext}\n\n--- 사용자 질문 ---\n${text}`
        : text;

      const body: Record<string, unknown> = {
        message: messageWithOcr,
        // OCR 컨텍스트가 있으면 direct 모드 (벡터DB 검색 없이 OCR 텍스트만으로 답변)
        mode: hasOcr ? "direct" : (responseMode === "direct" ? "direct" : "auto"),
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
      let ttftMs: number | undefined;
      let tps: number | undefined;
      let suggestedQuestions: string[] = [];

      // SSE parser: accumulate data lines per event, dispatch on blank line
      let dataLines: string[] = [];

      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      function handleSSEEvent(eventName: string, data: any) {
        switch (eventName) {
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
            if (data.ttft_ms != null) ttftMs = data.ttft_ms as number;
            if (data.tps != null) tps = data.tps as number;
            break;
          case "tool_start":
            setToolInProgress(data.message ?? data.tool_name ?? "도구 실행 중...");
            break;
          case "tool_end":
            setToolInProgress(null);
            break;
          case "self_rag_warning":
            showSnackbar(data.message ?? "근거 검증 주의: 원문을 확인해주세요.", "info");
            break;
          case "guardrail_warning":
            showSnackbar(data.message ?? "안전 필터가 적용되었습니다.", "info");
            break;
          case "suggested_questions":
            suggestedQuestions = data.questions ?? [];
            break;
          case "error":
            showSnackbar(data.message ?? "응답 생성 중 오류가 발생했습니다.", "error");
            break;
        }
      }

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        // Normalize CRLF → LF (sse-starlette uses \r\n)
        buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n").replace(/\r/g, "\n");
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (line.startsWith("event:")) {
            // Dispatch any pending data before switching event
            if (dataLines.length > 0) {
              const pendingRaw = dataLines.join("\n").trim();
              dataLines = [];
              if (pendingRaw) {
                try {
                  const pendingData = JSON.parse(pendingRaw);
                  handleSSEEvent(currentEvent, pendingData);
                } catch {
                  // ignore
                }
              }
            }
            currentEvent = line.slice(6).trim();
            continue;
          }
          if (line.startsWith("data:")) {
            dataLines.push(line.slice(5));
            continue;
          }
          if (line.startsWith(":")) {
            continue;
          }
          // Blank line = end of SSE event
          if (line === "" && dataLines.length > 0) {
            const raw = dataLines.join("\n").trim();
            dataLines = [];
            if (!raw) continue;
            try {
              const data = JSON.parse(raw);
              handleSSEEvent(currentEvent, data);
            } catch {
              // ignore malformed JSON
            }
          }
        }
      }

      // Flush remaining data
      if (dataLines.length > 0) {
        const raw = dataLines.join("\n").trim();
        if (raw) {
          try {
            const data = JSON.parse(raw);
            handleSSEEvent(currentEvent, data);
          } catch {
            // ignore
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
        ttftMs,
        tps,
        suggestedQuestions: suggestedQuestions.length > 0 ? suggestedQuestions : undefined,
        createdAt: new Date().toISOString(),
      };

      setMessages((prev) => [...prev, assistantMsg]);
      setStreamingContent("");

      if (newSessionId && newSessionId !== currentSessionId) {
        skipNextReloadRef.current = true;
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
      setToolInProgress(null);
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
    attachedFiles,
    onFilesSent,
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

  const [isEditing, setIsEditing] = useState(false);

  const handleEditUserMessage = useCallback(
    (messageId: string, content: string) => {
      if (isStreaming) return;
      const idx = messages.findIndex((m) => m.id === messageId);
      if (idx === -1) return;
      setIsEditing(true);
      setMessages(messages.slice(0, idx));
      setInputValue(content);
    },
    [messages, isStreaming]
  );

  const handleRegenerate = useCallback(
    async (assistantMessageId: string) => {
      if (isStreaming) return;

      // Find the assistant message index
      const assistantIdx = messages.findIndex((m) => m.id === assistantMessageId);
      if (assistantIdx === -1) return;

      // Find the user message immediately before it
      const userMsg = messages
        .slice(0, assistantIdx)
        .reverse()
        .find((m) => m.role === "user");
      if (!userMsg) return;

      const userContent = userMsg.content;

      // Remove the assistant message (keep messages up to but not including it)
      setMessages(messages.slice(0, assistantIdx));
      setIsStreaming(true);
      setStreamingContent("");
      streamingContentRef.current = "";

      const abortController = new AbortController();
      abortRef.current = abortController;

      try {
        const body: Record<string, unknown> = {
          message: userContent,
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
        let ttftMs: number | undefined;
        let tps: number | undefined;
        let suggestedQuestions: string[] = [];
        let dataLines: string[] = [];

        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        function handleSSEEvent(eventName: string, data: any) {
          switch (eventName) {
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
              if (data.ttft_ms != null) ttftMs = data.ttft_ms as number;
              if (data.tps != null) tps = data.tps as number;
              break;
            case "tool_start":
              setToolInProgress(data.message ?? data.tool_name ?? "도구 실행 중...");
              break;
            case "tool_end":
              setToolInProgress(null);
              break;
            case "self_rag_warning":
              showSnackbar(data.message ?? "근거 검증 주의: 원문을 확인해주세요.", "info");
              break;
            case "guardrail_warning":
              showSnackbar(data.message ?? "안전 필터가 적용되었습니다.", "info");
              break;
            case "suggested_questions":
              suggestedQuestions = data.questions ?? [];
              break;
            case "error":
              showSnackbar(data.message ?? "응답 생성 중 오류가 발생했습니다.", "error");
              break;
          }
        }

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n").replace(/\r/g, "\n");
          const lines = buffer.split("\n");
          buffer = lines.pop() || "";

          for (const line of lines) {
            if (line.startsWith("event:")) {
              if (dataLines.length > 0) {
                const pendingRaw = dataLines.join("\n").trim();
                dataLines = [];
                if (pendingRaw) {
                  try {
                    const pendingData = JSON.parse(pendingRaw);
                    handleSSEEvent(currentEvent, pendingData);
                  } catch { /* ignore */ }
                }
              }
              currentEvent = line.slice(6).trim();
              continue;
            }
            if (line.startsWith("data:")) {
              dataLines.push(line.slice(5));
              continue;
            }
            if (line.startsWith(":")) continue;
            if (line === "" && dataLines.length > 0) {
              const raw = dataLines.join("\n").trim();
              dataLines = [];
              if (!raw) continue;
              try {
                const data = JSON.parse(raw);
                handleSSEEvent(currentEvent, data);
              } catch { /* ignore */ }
            }
          }
        }

        if (dataLines.length > 0) {
          const raw = dataLines.join("\n").trim();
          if (raw) {
            try {
              const data = JSON.parse(raw);
              handleSSEEvent(currentEvent, data);
            } catch { /* ignore */ }
          }
        }

        const assistantMsg: Message = {
          id: messageId || crypto.randomUUID(),
          role: "assistant",
          content,
          sources: sources.length > 0 ? sources : undefined,
          confidenceScore: confidence || undefined,
          latencyMs: latencyMs || undefined,
          ttftMs,
          tps,
          suggestedQuestions: suggestedQuestions.length > 0 ? suggestedQuestions : undefined,
          createdAt: new Date().toISOString(),
        };

        setMessages((prev) => [...prev, assistantMsg]);
        setStreamingContent("");

        if (newSessionId && newSessionId !== currentSessionId) {
          skipNextReloadRef.current = true;
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
          showSnackbar("재생성에 실패했습니다.", "error");
        }
      } finally {
        setIsStreaming(false);
        setStreamingContent("");
        setToolInProgress(null);
        abortRef.current = null;
      }
    },
    [
      messages,
      isStreaming,
      currentSessionId,
      responseMode,
      selectedModel,
      temperature,
      setCurrentSessionId,
      queryClient,
      showSnackbar,
    ]
  );

  return {
    messages,
    setMessages,
    inputValue,
    setInputValue,
    isStreaming,
    isEditing,
    isOcrProcessing,
    streamingContent,
    messagesEndRef,
    handleSend,
    handleStopGeneration,
    handleNewChat,
    handleKeyDown,
    handleEditUserMessage,
    handleRegenerate,
    toolInProgress,
  };
}
