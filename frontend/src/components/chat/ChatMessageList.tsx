import {
  Box,
  Typography,
  Avatar,
  Alert,
  Fab,
  IconButton,
  Tooltip,
  Button,
  CircularProgress,
} from "@mui/material";
import {
  AutoAwesome as AutoAwesomeIcon,
  KeyboardArrowDown as KeyboardArrowDownIcon,
  Summarize as SummarizeIcon,
  EditNote as EditNoteIcon,
  Analytics as AnalyticsIcon,
  Gavel as GavelIcon,
  ContentCopy as ContentCopyIcon,
  Bookmark as BookmarkIcon,
  Close as CloseIcon,
  PushPin as PushPinIcon,
  ExpandMore as ExpandMoreIcon,
  ExpandLess as ExpandLessIcon,
  ExpandLess as ExpandUpIcon,
  DragIndicator as DragIndicatorIcon,
  AccountBalance as AccountBalanceIcon,
  School as SchoolIcon,
  Payments as PaymentsIcon,
  Lightbulb as LightbulbIcon,
  Policy as PolicyIcon,
  Shield as ShieldIcon,
} from "@mui/icons-material";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { Message } from "../../types";
import { MessageBubble } from "./MessageBubble";
import { useRef, useState, useCallback, useEffect, useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { bookmarksApi } from "../../api/client";
import { useAuth } from "../../contexts/AuthContext";

/** Maximum number of messages rendered in the DOM at one time. */
const VISIBLE_MESSAGE_LIMIT = 50;
/** How many more messages to reveal per "load more" action. */
const LOAD_MORE_STEP = 30;

const SUGGESTION_CARDS = [
  {
    label: "여비규정에서 국내 출장 시 일비 지급 기준은?",
    Icon: GavelIcon,
    color: "#10a37f",
    bg: "rgba(16, 163, 127, 0.08)",
  },
  {
    label: "직장 내 갑질 행위의 정의와 금지 행위 유형은?",
    Icon: AnalyticsIcon,
    color: "#5436da",
    bg: "rgba(84, 54, 218, 0.08)",
  },
  {
    label: "업무상 재해가 발생했을 때 보상 절차는 어떻게 되나요?",
    Icon: EditNoteIcon,
    color: "#d97706",
    bg: "rgba(217, 119, 6, 0.08)",
  },
  {
    label: "징계의 종류에는 어떤 것들이 있나요?",
    Icon: AccountBalanceIcon,
    color: "#dc2626",
    bg: "rgba(220, 38, 38, 0.08)",
  },
  {
    label: "신입사원 수습기간은 얼마인가요?",
    Icon: SchoolIcon,
    color: "#0891b2",
    bg: "rgba(8, 145, 178, 0.08)",
  },
  {
    label: "퇴직금 지급 기준은 어떻게 되나요?",
    Icon: PaymentsIcon,
    color: "#7c3aed",
    bg: "rgba(124, 58, 237, 0.08)",
  },
  {
    label: "직무발명에 대한 보상 규정이 있나요?",
    Icon: LightbulbIcon,
    color: "#ea580c",
    bg: "rgba(234, 88, 12, 0.08)",
  },
  {
    label: "부정청탁 금지 위반 시 처리 절차는?",
    Icon: PolicyIcon,
    color: "#be185d",
    bg: "rgba(190, 24, 93, 0.08)",
  },
  {
    label: "내부 고발자(공익신고자) 보호 규정이 있나요?",
    Icon: ShieldIcon,
    color: "#059669",
    bg: "rgba(5, 150, 105, 0.08)",
  },
  {
    label: "연차휴가는 어떻게 부여되나요?",
    Icon: SummarizeIcon,
    color: "#4f46e5",
    bg: "rgba(79, 70, 229, 0.08)",
  },
];

interface ChatMessageListProps {
  messages: Message[];
  streamingContent: string;
  isStreaming: boolean;
  isEditing?: boolean;
  toolInProgress?: string | null;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  announcements: any;
  onSampleClick: (question: string) => void;
  onFeedback: (messageId: string, rating: number) => void;
  onErrorReport: (messageId: string) => void;
  onEditAnswer: (messageId: string, currentContent: string) => void;
  onEditUserMessage?: (messageId: string, content: string) => void;
  onRegenerate?: (messageId: string) => void;
  currentSessionId: string | null;
  userRole: string;
  messagesEndRef: React.RefObject<HTMLDivElement | null>;
  // Search highlight
  searchHighlightIndices?: number[];
  currentSearchHighlightIndex?: number;
  // Multi-select
  selectMode?: boolean;
  selectedMessageIds?: Set<string>;
  onToggleSelect?: (messageId: string) => void;
  onExitSelectMode?: () => void;
  onShowSnackbar?: (msg: string, severity: "success" | "error" | "info") => void;
  // Pin
  pinnedMessageIds?: Set<string>;
  onTogglePin?: (messageId: string) => void;
  // Fork
  onFork?: (messageId: string) => void;
  // Reactions
  reactions?: Map<string, string[]>;
  onReact?: (messageId: string, emoji: string) => void;
}

export function ChatMessageList({
  messages,
  streamingContent,
  isStreaming,
  isEditing,
  toolInProgress,
  announcements,
  onSampleClick,
  onFeedback,
  onErrorReport,
  onEditAnswer,
  onEditUserMessage,
  onRegenerate,
  currentSessionId,
  userRole,
  messagesEndRef,
  searchHighlightIndices = [],
  currentSearchHighlightIndex = 0,
  selectMode = false,
  selectedMessageIds,
  onToggleSelect,
  onExitSelectMode,
  onShowSnackbar,
  pinnedMessageIds,
  onTogglePin,
  onFork,
  reactions,
  onReact,
}: ChatMessageListProps) {
  const { user } = useAuth();
  const displayName = user?.full_name || user?.username || null;

  const scrollContainerRef = useRef<HTMLDivElement | null>(null);
  const isNearBottom = useRef(true);
  const [showScrollBtn, setShowScrollBtn] = useState(false);

  // Per-message refs for search scroll-to (by index)
  const messageRowRefs = useRef<Map<number, HTMLDivElement>>(new Map());

  const setMessageRowRef = useCallback((idx: number, el: HTMLDivElement | null) => {
    if (el) {
      messageRowRefs.current.set(idx, el);
    } else {
      messageRowRefs.current.delete(idx);
    }
  }, []);

  // Per-message refs for pin scroll-to (by message id)
  const messageIdRefs = useRef<Map<string, HTMLDivElement>>(new Map());

  const setMessageIdRef = useCallback((id: string, el: HTMLDivElement | null) => {
    if (el) {
      messageIdRefs.current.set(id, el);
    } else {
      messageIdRefs.current.delete(id);
    }
  }, []);

  // Pinned panel collapsed state
  const [pinnedPanelCollapsed, setPinnedPanelCollapsed] = useState(false);

  // ── Pinned message drag-and-drop reorder ──────────────────────────────────
  // Maintain a stable ordered list of pinned message IDs.
  // Source of truth: pinnedMessageIds (Set) + localStorage for saved order.
  const localStorageKey = currentSessionId
    ? `flux-rag-pin-order-${currentSessionId}`
    : null;

  // Load persisted order from localStorage, falling back to insertion order.
  const buildInitialPinOrder = useCallback(
    (ids: Set<string>): string[] => {
      if (!localStorageKey) return Array.from(ids);
      try {
        const raw = localStorage.getItem(localStorageKey);
        if (raw) {
          const saved: string[] = JSON.parse(raw);
          // Keep only IDs that are still actually pinned, preserve saved order,
          // then append any newly pinned IDs at the end.
          const savedFiltered = saved.filter((id) => ids.has(id));
          const missing = Array.from(ids).filter((id) => !savedFiltered.includes(id));
          return [...savedFiltered, ...missing];
        }
      } catch {
        // ignore malformed localStorage value
      }
      return Array.from(ids);
    },
    [localStorageKey],
  );

  const [pinnedOrder, setPinnedOrder] = useState<string[]>(() =>
    buildInitialPinOrder(pinnedMessageIds ?? new Set()),
  );

  // Sync pinnedOrder whenever the pinned set changes (pin/unpin events).
  useEffect(() => {
    const ids = pinnedMessageIds ?? new Set<string>();
    setPinnedOrder((prev) => {
      const prevFiltered = prev.filter((id) => ids.has(id));
      const missing = Array.from(ids).filter((id) => !prevFiltered.includes(id));
      return [...prevFiltered, ...missing];
    });
  }, [pinnedMessageIds]);

  // Persist order to localStorage whenever it changes.
  useEffect(() => {
    if (!localStorageKey || pinnedOrder.length === 0) return;
    try {
      localStorage.setItem(localStorageKey, JSON.stringify(pinnedOrder));
    } catch {
      // quota exceeded or private browsing — silently ignore
    }
  }, [pinnedOrder, localStorageKey]);

  // Drag state
  const dragSrcIdRef = useRef<string | null>(null);
  const [dragOverId, setDragOverId] = useState<string | null>(null);
  const [dropPosition, setDropPosition] = useState<"above" | "below">("below");
  const [isDraggingId, setIsDraggingId] = useState<string | null>(null);

  const handleDragStart = useCallback(
    (e: React.DragEvent<HTMLDivElement>, messageId: string) => {
      dragSrcIdRef.current = messageId;
      setIsDraggingId(messageId);
      e.dataTransfer.effectAllowed = "move";
      // Use a transparent 1×1 image as drag ghost so the item itself shows the
      // opacity fade instead of a platform-default ghost.
      const ghost = document.createElement("div");
      ghost.style.position = "fixed";
      ghost.style.top = "-9999px";
      document.body.appendChild(ghost);
      e.dataTransfer.setDragImage(ghost, 0, 0);
      requestAnimationFrame(() => document.body.removeChild(ghost));
    },
    [],
  );

  const handleDragOver = useCallback(
    (e: React.DragEvent<HTMLDivElement>, messageId: string) => {
      e.preventDefault();
      e.dataTransfer.dropEffect = "move";
      if (messageId === dragSrcIdRef.current) return;
      const rect = e.currentTarget.getBoundingClientRect();
      const midY = rect.top + rect.height / 2;
      setDropPosition(e.clientY < midY ? "above" : "below");
      setDragOverId(messageId);
    },
    [],
  );

  const handleDragLeave = useCallback(
    (e: React.DragEvent<HTMLDivElement>, messageId: string) => {
      // Only clear if leaving to outside the list entirely (relatedTarget outside)
      if (!e.currentTarget.contains(e.relatedTarget as Node | null)) {
        setDragOverId((prev) => (prev === messageId ? null : prev));
      }
    },
    [],
  );

  const handleDrop = useCallback(
    (e: React.DragEvent<HTMLDivElement>, targetId: string) => {
      e.preventDefault();
      const srcId = dragSrcIdRef.current;
      if (!srcId || srcId === targetId) {
        setDragOverId(null);
        setIsDraggingId(null);
        return;
      }
      setPinnedOrder((prev) => {
        const next = prev.filter((id) => id !== srcId);
        const targetIdx = next.indexOf(targetId);
        if (targetIdx === -1) return prev;
        const insertAt = dropPosition === "above" ? targetIdx : targetIdx + 1;
        next.splice(insertAt, 0, srcId);
        return next;
      });
      setDragOverId(null);
      setIsDraggingId(null);
      dragSrcIdRef.current = null;
    },
    [dropPosition],
  );

  const handleDragEnd = useCallback(() => {
    setIsDraggingId(null);
    setDragOverId(null);
    dragSrcIdRef.current = null;
  }, []);

  // Touch-based drag support (pointer events)
  const touchDragStateRef = useRef<{
    srcId: string;
    startY: number;
    currentY: number;
    active: boolean;
  } | null>(null);

  const handlePointerDown = useCallback(
    (e: React.PointerEvent<HTMLDivElement>, messageId: string) => {
      // Only respond to touch/pen (not mouse — mouse uses HTML5 DnD)
      if (e.pointerType === "mouse") return;
      e.currentTarget.setPointerCapture(e.pointerId);
      touchDragStateRef.current = {
        srcId: messageId,
        startY: e.clientY,
        currentY: e.clientY,
        active: false,
      };
    },
    [],
  );

  const handlePointerMove = useCallback(
    (e: React.PointerEvent<HTMLDivElement>) => {
      const state = touchDragStateRef.current;
      if (!state) return;
      const delta = Math.abs(e.clientY - state.startY);
      if (!state.active && delta > 8) {
        state.active = true;
        setIsDraggingId(state.srcId);
      }
      if (!state.active) return;
      state.currentY = e.clientY;
      dragSrcIdRef.current = state.srcId;

      // Find which pinned item the pointer is over
      const el = document.elementFromPoint(e.clientX, e.clientY);
      const row = el?.closest<HTMLElement>("[data-pinned-id]");
      if (row) {
        const targetId = row.dataset.pinnedId!;
        if (targetId !== state.srcId) {
          const rect = row.getBoundingClientRect();
          setDropPosition(e.clientY < rect.top + rect.height / 2 ? "above" : "below");
          setDragOverId(targetId);
        }
      }
    },
    [],
  );

  const handlePointerUp = useCallback(
    () => {
      const state = touchDragStateRef.current;
      if (state?.active && dragSrcIdRef.current && dragOverId) {
        const srcId = dragSrcIdRef.current;
        const targetId = dragOverId;
        setPinnedOrder((prev) => {
          const next = prev.filter((id) => id !== srcId);
          const targetIdx = next.indexOf(targetId);
          if (targetIdx === -1) return prev;
          const insertAt = dropPosition === "above" ? targetIdx : targetIdx + 1;
          next.splice(insertAt, 0, srcId);
          return next;
        });
      }
      touchDragStateRef.current = null;
      setIsDraggingId(null);
      setDragOverId(null);
      dragSrcIdRef.current = null;
    },
    [dragOverId, dropPosition],
  );
  // ──────────────────────────────────────────────────────────────────────────

  const handleScrollToMessage = useCallback((messageId: string) => {
    const el = messageIdRefs.current.get(messageId);
    if (el) {
      el.scrollIntoView({ behavior: "smooth", block: "center" });
    }
  }, []);

  const { data: bookmarksData } = useQuery({
    queryKey: ["bookmarks"],
    queryFn: () => bookmarksApi.list(),
    staleTime: 30000,
  });

  const bookmarkedIds = useMemo<Set<string>>(() => {
    const items: { message_id: string }[] = bookmarksData?.data?.bookmarks ?? [];
    return new Set(items.map((b) => b.message_id));
  }, [bookmarksData]);

  // Visible messages (no system messages) — defined early so select callbacks can use it
  // This is the FULL list; windowing only affects what's rendered, not what's searchable.
  const visibleMessages = useMemo(
    () => messages.filter((m) => m.role !== "system"),
    [messages],
  );

  // ── Windowing ──────────────────────────────────────────────────────────────
  // `visibleStartIndex` tracks which index of `visibleMessages` is the first
  // rendered one.  It starts so that we always show the LAST VISIBLE_MESSAGE_LIMIT
  // messages.  When the user scrolls to the top (or clicks "load more") we
  // decrease it by LOAD_MORE_STEP.
  const [visibleStartIndex, setVisibleStartIndex] = useState(() =>
    Math.max(0, visibleMessages.length - VISIBLE_MESSAGE_LIMIT),
  );

  // Keep visibleStartIndex sane when new messages arrive: if we were already
  // showing the tail, keep showing the tail (don't suddenly hide new messages).
  const prevMsgCountRef = useRef(visibleMessages.length);
  useEffect(() => {
    const prev = prevMsgCountRef.current;
    const curr = visibleMessages.length;
    if (curr > prev) {
      // New messages were appended — slide the window down if we were at the end.
      const wasShowingAll = visibleStartIndex <= Math.max(0, prev - VISIBLE_MESSAGE_LIMIT) + 1;
      if (wasShowingAll) {
        setVisibleStartIndex(Math.max(0, curr - VISIBLE_MESSAGE_LIMIT));
      }
    }
    prevMsgCountRef.current = curr;
  }, [visibleMessages.length, visibleStartIndex]);

  // When a search match is outside the current window, expand the window to
  // include it.
  useEffect(() => {
    if (searchHighlightIndices.length === 0) return;
    const targetIdx = searchHighlightIndices[currentSearchHighlightIndex];
    if (targetIdx == null) return;
    if (targetIdx < visibleStartIndex) {
      setVisibleStartIndex(Math.max(0, targetIdx - 5));
    }
  }, [searchHighlightIndices, currentSearchHighlightIndex, visibleStartIndex]);

  const hiddenCount = visibleStartIndex;
  const [isLoadingMore, setIsLoadingMore] = useState(false);

  const loadMoreMessages = useCallback(() => {
    if (visibleStartIndex === 0 || isLoadingMore) return;
    setIsLoadingMore(true);

    // Capture current scroll height before expanding so we can restore position.
    const container = scrollContainerRef.current;
    const scrollHeightBefore = container?.scrollHeight ?? 0;
    const scrollTopBefore = container?.scrollTop ?? 0;

    setVisibleStartIndex((prev) => Math.max(0, prev - LOAD_MORE_STEP));

    // After the DOM updates, push scrollTop down by the delta so the user's
    // viewport stays in the same place (content prepended above).
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        if (container) {
          const delta = container.scrollHeight - scrollHeightBefore;
          container.scrollTop = scrollTopBefore + delta;
        }
        setIsLoadingMore(false);
      });
    });
  }, [visibleStartIndex, isLoadingMore]);

  // The slice that is actually rendered.
  const renderedMessages = useMemo(
    () => visibleMessages.slice(visibleStartIndex),
    [visibleMessages, visibleStartIndex],
  );

  // Sentinel ref at the top — IntersectionObserver triggers load-more when it
  // becomes visible (i.e. user scrolled to the top of the rendered list).
  const topSentinelRef = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    const sentinel = topSentinelRef.current;
    if (!sentinel || visibleStartIndex === 0) return;
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries[0].isIntersecting) {
          loadMoreMessages();
        }
      },
      { root: scrollContainerRef.current, threshold: 0.1 },
    );
    observer.observe(sentinel);
    return () => observer.disconnect();
  }, [visibleStartIndex, loadMoreMessages]);
  // ──────────────────────────────────────────────────────────────────────────

  // Multi-select: copy selected messages to clipboard
  const handleCopySelected = useCallback(async () => {
    if (!selectedMessageIds || selectedMessageIds.size === 0) return;
    const selected = visibleMessages
      .filter((m) => selectedMessageIds.has(m.id))
      // sort by natural order (they're already in order in the array)
      .map((m) => {
        const label = m.role === "user" ? "사용자" : "AI 어시스턴트";
        return `${label}: ${m.content}`;
      })
      .join("\n\n");
    try {
      await navigator.clipboard.writeText(selected);
      onShowSnackbar?.(`${selectedMessageIds.size}개 메시지가 복사되었습니다.`, "success");
    } catch {
      onShowSnackbar?.("복사에 실패했습니다.", "error");
    }
  }, [selectedMessageIds, visibleMessages, onShowSnackbar]);

  // Multi-select: bookmark all selected (fire-and-forget per message)
  const handleBookmarkSelected = useCallback(async () => {
    if (!selectedMessageIds || selectedMessageIds.size === 0 || !currentSessionId) return;
    const toBookmark = visibleMessages.filter(
      (m) => selectedMessageIds.has(m.id) && !bookmarkedIds.has(m.id),
    );
    if (toBookmark.length === 0) {
      onShowSnackbar?.("선택된 메시지가 이미 모두 북마크되어 있습니다.", "info");
      return;
    }
    await Promise.allSettled(
      toBookmark.map((m) =>
        bookmarksApi.add({
          message_id: m.id,
          session_id: currentSessionId,
          content: m.content,
          role: m.role,
        }),
      ),
    );
    onShowSnackbar?.(`${toBookmark.length}개 메시지를 북마크했습니다.`, "success");
  }, [selectedMessageIds, visibleMessages, bookmarkedIds, currentSessionId, onShowSnackbar]);

  const scrollToBottom = useCallback((setNearBottom = false) => {
    if (scrollContainerRef.current) {
      if (setNearBottom) {
        isNearBottom.current = true;
      }
      scrollContainerRef.current.scrollTo({
        top: scrollContainerRef.current.scrollHeight,
        behavior: "smooth",
      });
    }
  }, []);

  const handleScroll = useCallback(() => {
    if (!scrollContainerRef.current) return;
    const { scrollHeight, scrollTop, clientHeight } =
      scrollContainerRef.current;
    const distanceFromBottom = scrollHeight - scrollTop - clientHeight;
    isNearBottom.current = distanceFromBottom <= 100;
    setShowScrollBtn(distanceFromBottom > 200);
  }, []);

  useEffect(() => {
    const container = scrollContainerRef.current;
    if (!container) return;
    container.addEventListener("scroll", handleScroll);
    return () => container.removeEventListener("scroll", handleScroll);
  }, [handleScroll]);

  // Auto-scroll during streaming when user is near bottom
  useEffect(() => {
    if (isNearBottom.current) {
      scrollToBottom();
    }
  }, [streamingContent, scrollToBottom]);

  // Auto-scroll when a new message is added (user sends or AI finishes)
  useEffect(() => {
    if (isNearBottom.current) {
      scrollToBottom();
    }
  }, [messages.length, scrollToBottom]);

  // Scroll to highlighted search result
  useEffect(() => {
    if (searchHighlightIndices.length === 0) return;
    const targetIdx = searchHighlightIndices[currentSearchHighlightIndex];
    if (targetIdx == null) return;
    const el = messageRowRefs.current.get(targetIdx);
    if (el) {
      el.scrollIntoView({ behavior: "smooth", block: "center" });
    }
  }, [searchHighlightIndices, currentSearchHighlightIndex]);

  return (
    <Box
      ref={scrollContainerRef}
      role="log"
      aria-live="polite"
      aria-label="대화 메시지"
      sx={{ flex: 1, overflow: "auto", py: 2, position: "relative" }}
    >
      {/* Selection action bar */}
      {selectMode && selectedMessageIds && selectedMessageIds.size > 0 && (
        <Box
          sx={{
            position: "sticky",
            top: 0,
            zIndex: 20,
            display: "flex",
            alignItems: "center",
            gap: 1,
            px: 2,
            py: 0.75,
            bgcolor: (theme) =>
              theme.palette.mode === "dark"
                ? "rgba(16,163,127,0.15)"
                : "rgba(16,163,127,0.1)",
            backdropFilter: "blur(8px)",
            borderBottom: "1px solid",
            borderColor: "rgba(16,163,127,0.25)",
          }}
        >
          <Typography variant="body2" sx={{ fontWeight: 600, flex: 1, color: "primary.main" }}>
            {selectedMessageIds.size}개 선택됨
          </Typography>
          <Tooltip title="선택된 메시지 복사">
            <IconButton size="small" onClick={handleCopySelected} aria-label="선택 복사">
              <ContentCopyIcon fontSize="small" />
            </IconButton>
          </Tooltip>
          {currentSessionId && (
            <Tooltip title="선택된 메시지 북마크">
              <IconButton size="small" onClick={handleBookmarkSelected} aria-label="선택 북마크">
                <BookmarkIcon fontSize="small" />
              </IconButton>
            </Tooltip>
          )}
          <Tooltip title="선택 모드 종료">
            <IconButton size="small" onClick={onExitSelectMode} aria-label="선택 모드 종료">
              <CloseIcon fontSize="small" />
            </IconButton>
          </Tooltip>
        </Box>
      )}

      {/* Pinned Messages Panel */}
      {pinnedMessageIds && pinnedMessageIds.size > 0 && (
        <Box
          sx={{
            position: "sticky",
            top: 0,
            zIndex: 15,
            mx: 0,
            borderBottom: "1px solid",
            borderColor: (theme) =>
              theme.palette.mode === "dark"
                ? "rgba(16,163,127,0.2)"
                : "rgba(16,163,127,0.18)",
            bgcolor: (theme) =>
              theme.palette.mode === "dark"
                ? "rgba(16,163,127,0.08)"
                : "rgba(16,163,127,0.05)",
            backdropFilter: "blur(8px)",
          }}
        >
          {/* Header row */}
          <Box
            sx={{
              display: "flex",
              alignItems: "center",
              gap: 1,
              px: 2,
              py: 0.75,
              cursor: "pointer",
              userSelect: "none",
            }}
            onClick={() => setPinnedPanelCollapsed((v) => !v)}
          >
            <PushPinIcon sx={{ fontSize: 14, color: "primary.main" }} />
            <Typography
              variant="caption"
              sx={{ fontWeight: 600, color: "primary.main", flex: 1 }}
            >
              고정된 메시지 {pinnedMessageIds.size}개
            </Typography>
            <IconButton size="small" sx={{ p: 0.25 }}>
              {pinnedPanelCollapsed ? (
                <ExpandMoreIcon sx={{ fontSize: 16 }} />
              ) : (
                <ExpandLessIcon sx={{ fontSize: 16 }} />
              )}
            </IconButton>
          </Box>

          {/* Pinned items list — drag-and-drop reorderable */}
          {!pinnedPanelCollapsed && (
            <Box sx={{ pb: 0.5 }}>
              {pinnedOrder
                .filter((id) => pinnedMessageIds.has(id))
                .map((id) => {
                  const m = visibleMessages.find((msg) => msg.id === id);
                  if (!m) return null;
                  const preview =
                    m.content.length > 120
                      ? m.content.slice(0, 120) + "\u2026"
                      : m.content;
                  const roleLabel = m.role === "user" ? "사용자" : "AI 어시스턴트";
                  const isBeingDragged = isDraggingId === m.id;
                  const isDropTarget = dragOverId === m.id;

                  return (
                    <Box
                      key={m.id}
                      data-pinned-id={m.id}
                      draggable
                      onDragStart={(e) => handleDragStart(e, m.id)}
                      onDragOver={(e) => handleDragOver(e, m.id)}
                      onDragLeave={(e) => handleDragLeave(e, m.id)}
                      onDrop={(e) => handleDrop(e, m.id)}
                      onDragEnd={handleDragEnd}
                      onPointerDown={(e) => handlePointerDown(e, m.id)}
                      onPointerMove={handlePointerMove}
                      onPointerUp={handlePointerUp}
                      sx={{
                        position: "relative",
                        display: "flex",
                        alignItems: "flex-start",
                        gap: 1,
                        px: 2,
                        py: 0.5,
                        cursor: isBeingDragged ? "grabbing" : "pointer",
                        opacity: isBeingDragged ? 0.4 : 1,
                        transition: "opacity 0.15s ease, box-shadow 0.15s ease, transform 0.15s ease",
                        boxShadow: isBeingDragged
                          ? (theme) =>
                              theme.palette.mode === "dark"
                                ? "0 4px 16px rgba(0,0,0,0.5)"
                                : "0 4px 16px rgba(0,0,0,0.15)"
                          : "none",
                        transform: isBeingDragged ? "scale(0.98)" : "scale(1)",
                        // Drop indicator lines
                        "&::before": {
                          content: '""',
                          position: "absolute",
                          left: 8,
                          right: 8,
                          top: -1,
                          height: 2,
                          borderRadius: 1,
                          bgcolor: "primary.main",
                          opacity: isDropTarget && dropPosition === "above" ? 1 : 0,
                          transition: "opacity 0.1s ease",
                          pointerEvents: "none",
                          zIndex: 2,
                        },
                        "&::after": {
                          content: '""',
                          position: "absolute",
                          left: 8,
                          right: 8,
                          bottom: -1,
                          height: 2,
                          borderRadius: 1,
                          bgcolor: "primary.main",
                          opacity: isDropTarget && dropPosition === "below" ? 1 : 0,
                          transition: "opacity 0.1s ease",
                          pointerEvents: "none",
                          zIndex: 2,
                        },
                        "&:hover": {
                          bgcolor: (theme) =>
                            theme.palette.mode === "dark"
                              ? "rgba(255,255,255,0.04)"
                              : "rgba(0,0,0,0.03)",
                          "& .pin-drag-handle": {
                            opacity: 1,
                          },
                        },
                      }}
                      onClick={() => {
                        if (!isDraggingId) handleScrollToMessage(m.id);
                      }}
                    >
                      {/* Drag handle — visible on row hover */}
                      <Box
                        className="pin-drag-handle"
                        sx={{
                          opacity: 0,
                          flexShrink: 0,
                          display: "flex",
                          alignItems: "center",
                          mt: 0.35,
                          cursor: "grab",
                          color: "text.disabled",
                          transition: "opacity 0.15s ease",
                          "&:active": { cursor: "grabbing" },
                          // Always show if actively dragging this item
                          ...(isBeingDragged && { opacity: 1 }),
                        }}
                        onMouseDown={(e) => e.stopPropagation()} // prevent click-scroll on handle mousedown
                      >
                        <DragIndicatorIcon sx={{ fontSize: 14 }} />
                      </Box>

                      <Box sx={{ flex: 1, minWidth: 0 }}>
                        <Typography
                          variant="caption"
                          sx={{
                            fontWeight: 600,
                            color: "text.secondary",
                            display: "block",
                            fontSize: "0.7rem",
                          }}
                        >
                          {roleLabel}
                        </Typography>
                        <Typography
                          variant="caption"
                          sx={{
                            color: "text.primary",
                            display: "-webkit-box",
                            WebkitLineClamp: 2,
                            WebkitBoxOrient: "vertical",
                            overflow: "hidden",
                            lineHeight: 1.45,
                            fontSize: "0.8rem",
                          }}
                        >
                          {preview}
                        </Typography>
                      </Box>
                      <Tooltip title="고정 해제">
                        <IconButton
                          size="small"
                          sx={{ p: 0.25, flexShrink: 0, mt: 0.25 }}
                          onClick={(e) => {
                            e.stopPropagation();
                            onTogglePin?.(m.id);
                          }}
                          aria-label="고정 해제"
                        >
                          <CloseIcon sx={{ fontSize: 14 }} />
                        </IconButton>
                      </Tooltip>
                    </Box>
                  );
                })}
            </Box>
          )}
        </Box>
      )}

      {/* Announcement banners */}
      {announcements?.data?.announcements
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        ?.filter((a: any) => a.is_pinned && a.is_active)
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        .map((a: any) => (
          <Box key={a.id} sx={{ px: 2, mb: 1 }}>
            <Alert severity="info">
              <strong>{a.title}</strong> — {a.content.slice(0, 100)}
              {a.content.length > 100 ? "..." : ""}
            </Alert>
          </Box>
        ))}

      {/* Welcome Screen */}
      {messages.length === 0 && !streamingContent && !isEditing && (
        <Box
          sx={{
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
            height: "100%",
            gap: 4,
            px: 3,
            position: "relative",
            overflow: "hidden",
          }}
        >
          {/* Background atmospheric decoration */}
          <Box
            aria-hidden="true"
            sx={{
              position: "absolute",
              top: "10%",
              left: "5%",
              width: 320,
              height: 320,
              borderRadius: "50%",
              background: (theme) =>
                theme.palette.mode === "dark"
                  ? "radial-gradient(circle, rgba(16,163,127,0.05) 0%, transparent 70%)"
                  : "radial-gradient(circle, rgba(16,163,127,0.07) 0%, transparent 70%)",
              pointerEvents: "none",
              animation: "driftA 12s ease-in-out infinite",
              "@keyframes driftA": {
                "0%, 100%": { transform: "translate(0, 0) scale(1)" },
                "33%": { transform: "translate(12px, -18px) scale(1.04)" },
                "66%": { transform: "translate(-8px, 10px) scale(0.97)" },
              },
            }}
          />
          <Box
            aria-hidden="true"
            sx={{
              position: "absolute",
              bottom: "8%",
              right: "4%",
              width: 260,
              height: 260,
              borderRadius: "50%",
              background: (theme) =>
                theme.palette.mode === "dark"
                  ? "radial-gradient(circle, rgba(84,54,218,0.04) 0%, transparent 70%)"
                  : "radial-gradient(circle, rgba(84,54,218,0.06) 0%, transparent 70%)",
              pointerEvents: "none",
              animation: "driftB 16s ease-in-out infinite",
              "@keyframes driftB": {
                "0%, 100%": { transform: "translate(0, 0) scale(1)" },
                "40%": { transform: "translate(-14px, 20px) scale(1.06)" },
                "75%": { transform: "translate(10px, -8px) scale(0.96)" },
              },
            }}
          />
          <Box
            aria-hidden="true"
            sx={{
              position: "absolute",
              top: "45%",
              right: "12%",
              width: 160,
              height: 160,
              borderRadius: "50%",
              background: (theme) =>
                theme.palette.mode === "dark"
                  ? "radial-gradient(circle, rgba(217,119,6,0.03) 0%, transparent 70%)"
                  : "radial-gradient(circle, rgba(217,119,6,0.05) 0%, transparent 70%)",
              pointerEvents: "none",
              animation: "driftC 20s ease-in-out infinite",
              "@keyframes driftC": {
                "0%, 100%": { transform: "translate(0, 0)" },
                "50%": { transform: "translate(-16px, -12px)" },
              },
            }}
          />

          {/* Icon + greeting */}
          <Box sx={{ textAlign: "center", position: "relative", zIndex: 1 }}>
            {/* Animated floating icon */}
            <Box
              sx={{
                display: "inline-flex",
                alignItems: "center",
                justifyContent: "center",
                width: 80,
                height: 80,
                borderRadius: "24px",
                background: "linear-gradient(135deg, #10a37f 0%, #0d8f6f 50%, #5436da 100%)",
                boxShadow: (theme) =>
                  theme.palette.mode === "dark"
                    ? "0 8px 32px rgba(16,163,127,0.35), 0 2px 8px rgba(0,0,0,0.4)"
                    : "0 8px 32px rgba(16,163,127,0.25), 0 2px 8px rgba(0,0,0,0.12)",
                mb: 3,
                animation: "floatIcon 3s ease-in-out infinite",
                "@keyframes floatIcon": {
                  "0%, 100%": { transform: "translateY(0)" },
                  "50%": { transform: "translateY(-8px)" },
                },
              }}
            >
              <AutoAwesomeIcon sx={{ fontSize: 36, color: "#ffffff" }} />
            </Box>

            {/* Personalized greeting */}
            <Typography
              variant="h4"
              sx={{
                fontWeight: 700,
                mb: 1,
                letterSpacing: "-0.02em",
                lineHeight: 1.2,
              }}
            >
              {displayName ? (
                <>
                  안녕하세요,{" "}
                  <Box
                    component="span"
                    sx={{
                      background: "linear-gradient(90deg, #10a37f, #5436da)",
                      WebkitBackgroundClip: "text",
                      WebkitTextFillColor: "transparent",
                      backgroundClip: "text",
                    }}
                  >
                    {displayName}
                  </Box>
                  님!
                </>
              ) : (
                "AI 어시스턴트"
              )}
            </Typography>

            <Typography
              variant="body1"
              color="text.secondary"
              sx={{ fontWeight: 400, letterSpacing: "0.01em" }}
            >
              {displayName ? "오늘은 무엇을 도와드릴까요?" : "무엇이든 물어보세요"}
            </Typography>
          </Box>

          {/* 2x5 suggestion cards */}
          <Box
            sx={{
              display: "grid",
              gridTemplateColumns: "1fr 1fr",
              gap: 1.2,
              maxWidth: 640,
              width: "100%",
              position: "relative",
              zIndex: 1,
            }}
          >
            {SUGGESTION_CARDS.map(({ label, Icon, color, bg }) => (
              <Box
                key={label}
                role="button"
                tabIndex={0}
                onClick={() => onSampleClick(label)}
                onKeyDown={(e) => e.key === "Enter" && onSampleClick(label)}
                sx={{
                  border: "1px solid",
                  borderColor: "divider",
                  borderRadius: "12px",
                  p: 1.5,
                  cursor: "pointer",
                  transition: "transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease",
                  bgcolor: "background.paper",
                  display: "flex",
                  flexDirection: "column",
                  gap: 1.25,
                  "&:hover": {
                    transform: "scale(1.02)",
                    borderColor: color,
                    boxShadow: (theme) =>
                      theme.palette.mode === "dark"
                        ? `0 6px 20px rgba(0,0,0,0.3)`
                        : `0 6px 20px rgba(0,0,0,0.09)`,
                  },
                  "&:focus-visible": {
                    outline: `2px solid ${color}`,
                    outlineOffset: 2,
                  },
                }}
              >
                {/* Icon badge */}
                <Box
                  sx={{
                    width: 36,
                    height: 36,
                    borderRadius: "10px",
                    bgcolor: bg,
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    flexShrink: 0,
                  }}
                >
                  <Icon sx={{ fontSize: 20, color }} />
                </Box>
                <Typography
                  variant="body2"
                  sx={{
                    fontWeight: 500,
                    lineHeight: 1.45,
                    color: "text.primary",
                  }}
                >
                  {label}
                </Typography>
              </Box>
            ))}
          </Box>
        </Box>
      )}

      {/* ── Message windowing ── */}
      {/* Sentinel element: when it enters the viewport, load more messages */}
      {hiddenCount > 0 && (
        <Box ref={topSentinelRef} sx={{ px: 2, pb: 0.5 }}>
          <Button
            fullWidth
            variant="text"
            size="small"
            startIcon={
              isLoadingMore ? (
                <CircularProgress size={14} />
              ) : (
                <ExpandUpIcon fontSize="small" />
              )
            }
            onClick={loadMoreMessages}
            disabled={isLoadingMore}
            sx={{
              color: "text.secondary",
              fontSize: "0.8rem",
              textTransform: "none",
              py: 0.75,
              borderRadius: 2,
              "&:hover": { bgcolor: "action.hover" },
            }}
          >
            {isLoadingMore
              ? "불러오는 중…"
              : `이전 메시지 ${hiddenCount}개 더 보기`}
          </Button>
        </Box>
      )}

      {/* Message list — only the visible window is rendered */}
      {renderedMessages.map((msg, relIdx) => {
        // `globalIdx` is the index in the FULL visibleMessages array — used for
        // search highlighting refs so they stay consistent.
        const globalIdx = visibleStartIndex + relIdx;

        // For assistant messages, find the most recent preceding user message
        // to use as the highlight query (search the full array).
        let query: string | undefined;
        if (msg.role === "assistant") {
          for (let i = globalIdx - 1; i >= 0; i--) {
            if (visibleMessages[i].role === "user") {
              query = visibleMessages[i].content;
              break;
            }
          }
        }

        const isSearchMatch = searchHighlightIndices.includes(globalIdx);
        const isActiveMatch =
          isSearchMatch &&
          searchHighlightIndices[currentSearchHighlightIndex] === globalIdx;

        return (
          <Box
            key={msg.id}
            ref={(el: HTMLDivElement | null) => {
              setMessageRowRef(globalIdx, el);
              setMessageIdRef(msg.id, el);
            }}
            sx={
              isActiveMatch
                ? {
                    borderLeft: "3px solid",
                    borderColor: "primary.main",
                    bgcolor: "rgba(16,163,127,0.07)",
                    borderRadius: "0 4px 4px 0",
                    transition: "background-color 0.2s ease",
                  }
                : isSearchMatch
                  ? {
                      borderLeft: "3px solid",
                      borderColor: "rgba(16,163,127,0.35)",
                      bgcolor: "rgba(16,163,127,0.03)",
                      borderRadius: "0 4px 4px 0",
                    }
                  : undefined
            }
          >
            <MessageBubble
              msg={msg}
              onFeedback={onFeedback}
              onErrorReport={onErrorReport}
              onEditAnswer={onEditAnswer}
              onEditUserMessage={onEditUserMessage}
              onRegenerate={msg.role === "assistant" ? onRegenerate : undefined}
              onSuggestedClick={onSampleClick}
              sessionId={currentSessionId}
              userRole={userRole}
              isBookmarked={bookmarkedIds.has(msg.id)}
              query={query}
              selectMode={selectMode}
              isSelected={selectedMessageIds?.has(msg.id) ?? false}
              onToggleSelect={onToggleSelect}
              isPinned={pinnedMessageIds?.has(msg.id) ?? false}
              onTogglePin={onTogglePin}
              onFork={msg.role === "assistant" ? onFork : undefined}
              reactions={reactions?.get(msg.id) ?? []}
              onReact={onReact}
            />
          </Box>
        );
      })}

      {/* Streaming message */}
      {isStreaming && streamingContent && (
        <Box
          sx={{
            display: "flex",
            justifyContent: "center",
            py: 1.5,
            px: 2,
          }}
        >
          <Box
            sx={{
              maxWidth: 768,
              width: "100%",
              display: "flex",
              gap: 2,
            }}
          >
            <Avatar
              sx={{
                width: 28,
                height: 28,
                bgcolor: "primary.main",
                flexShrink: 0,
                mt: 0.25,
              }}
            >
              <AutoAwesomeIcon sx={{ fontSize: 16 }} />
            </Avatar>
            <Box sx={{ flex: 1, minWidth: 0 }}>
              <Typography
                variant="subtitle2"
                sx={{ fontWeight: 600, mb: 0.5 }}
              >
                AI 어시스턴트
              </Typography>
              <Box
                sx={{
                  "& p": { m: 0, mb: 1 },
                  "& p:last-child": { mb: 0 },
                }}
              >
                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                  {streamingContent}
                </ReactMarkdown>
                {/* Blinking cursor */}
                <Box
                  component="span"
                  sx={{
                    animation: "blink 1s step-end infinite",
                    "@keyframes blink": {
                      "0%, 100%": { opacity: 1 },
                      "50%": { opacity: 0 },
                    },
                    color: "primary.main",
                    fontWeight: 300,
                  }}
                >
                  ▌
                </Box>
              </Box>
            </Box>
          </Box>
        </Box>
      )}

      {/* Bouncing dots typing indicator (with optional tool-in-progress label) */}
      {isStreaming && !streamingContent && (
        <Box
          sx={{
            display: "flex",
            justifyContent: "center",
            py: 1.5,
            px: 2,
          }}
        >
          <Box
            sx={{
              maxWidth: 768,
              width: "100%",
              display: "flex",
              gap: 2,
            }}
          >
            <Avatar
              sx={{
                width: 28,
                height: 28,
                bgcolor: "primary.main",
                flexShrink: 0,
                mt: 0.25,
              }}
            >
              <AutoAwesomeIcon sx={{ fontSize: 16 }} />
            </Avatar>
            <Box sx={{ flex: 1, minWidth: 0, pt: 0.25 }}>
              <Typography
                variant="subtitle2"
                sx={{ fontWeight: 600, mb: 0.75 }}
              >
                AI 어시스턴트
              </Typography>
              {toolInProgress && (
                <Typography
                  variant="body2"
                  sx={{
                    color: "primary.main",
                    fontWeight: 500,
                    mb: 0.5,
                    animation: "pulse 2s ease-in-out infinite",
                    "@keyframes pulse": {
                      "0%, 100%": { opacity: 1 },
                      "50%": { opacity: 0.6 },
                    },
                  }}
                >
                  {toolInProgress}
                </Typography>
              )}
              <Box sx={{ display: "flex", gap: 0.75, alignItems: "center" }}>
                {[0, 1, 2].map((i) => (
                  <Box
                    key={i}
                    sx={{
                      width: 7,
                      height: 7,
                      borderRadius: "50%",
                      bgcolor: "primary.main",
                      animation: "bounce 1.4s ease-in-out infinite",
                      animationDelay: `${i * 0.16}s`,
                      "@keyframes bounce": {
                        "0%, 80%, 100%": {
                          transform: "scale(0.6)",
                          opacity: 0.4,
                        },
                        "40%": { transform: "scale(1)", opacity: 1 },
                      },
                    }}
                  />
                ))}
              </Box>
            </Box>
          </Box>
        </Box>
      )}

      <div ref={messagesEndRef} />

      {/* Scroll to bottom FAB */}
      {showScrollBtn && (
        <Fab
          size="small"
          onClick={() => scrollToBottom(true)}
          aria-label="맨 아래로 이동"
          sx={{
            position: "sticky",
            bottom: 16,
            left: "50%",
            transform: "translateX(-50%)",
            zIndex: 10,
            bgcolor: "background.paper",
            boxShadow: 2,
            "&:hover": { bgcolor: "action.hover" },
          }}
        >
          <KeyboardArrowDownIcon />
        </Fab>
      )}
    </Box>
  );
}
