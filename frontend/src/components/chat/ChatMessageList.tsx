import {
  Box,
  Typography,
  Avatar,
  Alert,
  Fab,
} from "@mui/material";
import {
  AutoAwesome as AutoAwesomeIcon,
  KeyboardArrowDown as KeyboardArrowDownIcon,
} from "@mui/icons-material";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { Message } from "../../types";
import { MessageBubble } from "./MessageBubble";
import { useRef, useState, useCallback, useEffect, useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { bookmarksApi } from "../../api/client";

const SUGGESTION_CHIPS = [
  "문서 내용을 요약해 주세요",
  "보고서 작성을 도와주세요",
  "데이터를 분석해 주세요",
  "규정에 대해 설명해 주세요",
];

interface ChatMessageListProps {
  messages: Message[];
  streamingContent: string;
  isStreaming: boolean;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  announcements: any;
  onSampleClick: (question: string) => void;
  onFeedback: (messageId: string, rating: number) => void;
  onErrorReport: (messageId: string) => void;
  onEditAnswer: (messageId: string, currentContent: string) => void;
  onEditUserMessage?: (messageId: string, content: string) => void;
  currentSessionId: string | null;
  userRole: string;
  messagesEndRef: React.RefObject<HTMLDivElement | null>;
}

export function ChatMessageList({
  messages,
  streamingContent,
  isStreaming,
  announcements,
  onSampleClick,
  onFeedback,
  onErrorReport,
  onEditAnswer,
  onEditUserMessage,
  currentSessionId,
  userRole,
  messagesEndRef,
}: ChatMessageListProps) {
  const scrollContainerRef = useRef<HTMLDivElement | null>(null);
  const [showScrollBtn, setShowScrollBtn] = useState(false);

  const { data: bookmarksData } = useQuery({
    queryKey: ["bookmarks"],
    queryFn: () => bookmarksApi.list(),
    staleTime: 30000,
  });

  const bookmarkedIds = useMemo<Set<string>>(() => {
    const items: { message_id: string }[] = bookmarksData?.data?.bookmarks ?? [];
    return new Set(items.map((b) => b.message_id));
  }, [bookmarksData]);

  const scrollToBottom = useCallback(() => {
    if (scrollContainerRef.current) {
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
    setShowScrollBtn(distanceFromBottom > 200);
  }, []);

  useEffect(() => {
    const container = scrollContainerRef.current;
    if (!container) return;
    container.addEventListener("scroll", handleScroll);
    return () => container.removeEventListener("scroll", handleScroll);
  }, [handleScroll]);

  return (
    <Box
      ref={scrollContainerRef}
      sx={{ flex: 1, overflow: "auto", py: 2, position: "relative" }}
    >
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
      {messages.length === 0 && !streamingContent && (
        <Box
          sx={{
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
            height: "100%",
            gap: 3,
            px: 3,
          }}
        >
          {/* Icon + title */}
          <Box sx={{ textAlign: "center" }}>
            <AutoAwesomeIcon
              sx={{ fontSize: 48, color: "primary.main", mb: 2 }}
            />
            <Typography
              variant="h4"
              sx={{ fontWeight: 700, mb: 1 }}
            >
              Flux AI
            </Typography>
            <Typography variant="body1" color="text.secondary">
              무엇이든 물어보세요
            </Typography>
          </Box>

          {/* 2x2 suggestion chips */}
          <Box
            sx={{
              display: "grid",
              gridTemplateColumns: "1fr 1fr",
              gap: 1.5,
              maxWidth: 560,
              width: "100%",
            }}
          >
            {SUGGESTION_CHIPS.map((chip) => (
              <Box
                key={chip}
                onClick={() => onSampleClick(chip)}
                sx={{
                  border: "1px solid",
                  borderColor: "divider",
                  borderRadius: 2,
                  p: 2,
                  cursor: "pointer",
                  transition: "all 0.2s ease",
                  "&:hover": {
                    borderColor: "primary.main",
                    bgcolor: "action.hover",
                  },
                }}
              >
                <Typography variant="body2" sx={{ fontWeight: 500 }}>
                  {chip}
                </Typography>
              </Box>
            ))}
          </Box>
        </Box>
      )}

      {/* Message list */}
      {messages.map((msg) => (
        <MessageBubble
          key={msg.id}
          msg={msg}
          onFeedback={onFeedback}
          onErrorReport={onErrorReport}
          onEditAnswer={onEditAnswer}
          onEditUserMessage={onEditUserMessage}
          sessionId={currentSessionId}
          userRole={userRole}
          isBookmarked={bookmarkedIds.has(msg.id)}
        />
      ))}

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
                Flux AI
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
              </Box>
            </Box>
          </Box>
        </Box>
      )}

      {/* Bouncing dots typing indicator */}
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
                Flux AI
              </Typography>
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
          onClick={scrollToBottom}
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
