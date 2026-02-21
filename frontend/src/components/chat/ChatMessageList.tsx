import {
  Box,
  Typography,
  Paper,
  Alert,
  Fab,
  Card,
  CardContent,
} from "@mui/material";
import {
  SmartToy as BotIcon,
  KeyboardArrowDown as KeyboardArrowDownIcon,
  Gavel,
  Engineering,
  Assessment,
} from "@mui/icons-material";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { Message } from "../../types";
import { MessageBubble } from "./MessageBubble";
import { useRef, useState, useCallback, useEffect } from "react";

interface CategoryCard {
  icon: React.ElementType;
  title: string;
  questions: string[];
}

const CATEGORY_CARDS: CategoryCard[] = [
  {
    icon: Gavel,
    title: "법률/규정",
    questions: ["감사규정의 목적은?", "징계의 종류는?"],
  },
  {
    icon: Engineering,
    title: "기술문서",
    questions: ["LNG 저장탱크 검사 절차는?", "배관 누출 탐지 방법은?"],
  },
  {
    icon: Assessment,
    title: "공시/보고서",
    questions: ["유동자산은 얼마인가요?", "출장의 주요 업무는?"],
  },
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
  currentSessionId,
  userRole,
  messagesEndRef,
}: ChatMessageListProps) {
  const scrollContainerRef = useRef<HTMLDivElement | null>(null);
  const [showScrollBtn, setShowScrollBtn] = useState(false);

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

    const { scrollHeight, scrollTop, clientHeight } = scrollContainerRef.current;
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

      {/* Enhanced Welcome Screen */}
      {messages.length === 0 && !streamingContent && (
        <Box
          sx={{
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
            height: "100%",
            gap: 4,
            px: 3,
          }}
        >
          <Box sx={{ textAlign: "center" }}>
            <Typography
              variant="h4"
              sx={{
                fontWeight: 600,
                background: "linear-gradient(135deg, #1976d2 0%, #42a5f5 100%)",
                backgroundClip: "text",
                WebkitBackgroundClip: "text",
                WebkitTextFillColor: "transparent",
                mb: 1,
              }}
            >
              한국가스기술공사 AI 어시스턴트
            </Typography>
            <Typography variant="body1" color="text.secondary">
              문서 기반으로 정확한 답변을 제공합니다
            </Typography>
          </Box>

          <Box
            sx={{
              display: "flex",
              gap: 2,
              flexWrap: "wrap",
              justifyContent: "center",
              maxWidth: 900,
              width: "100%",
            }}
          >
            {CATEGORY_CARDS.map((category, idx) => {
              const IconComponent = category.icon;
              return (
                <Box key={idx} sx={{ flex: "1 1 250px", maxWidth: 300 }}>
                  <Card
                    elevation={2}
                    sx={{
                      height: "100%",
                      transition: "all 0.3s ease",
                      "&:hover": {
                        elevation: 6,
                        transform: "translateY(-4px)",
                      },
                    }}
                  >
                    <CardContent>
                      <Box
                        sx={{
                          display: "flex",
                          alignItems: "center",
                          gap: 1,
                          mb: 2,
                        }}
                      >
                        <IconComponent color="primary" />
                        <Typography variant="h6" sx={{ fontWeight: 600 }}>
                          {category.title}
                        </Typography>
                      </Box>
                      <Box sx={{ display: "flex", flexDirection: "column", gap: 1 }}>
                        {category.questions.map((question, qIdx) => (
                          <Typography
                            key={qIdx}
                            variant="body2"
                            onClick={() => onSampleClick(question)}
                            sx={{
                              cursor: "pointer",
                              px: 1.5,
                              py: 1,
                              borderRadius: 1,
                              transition: "all 0.2s ease",
                              "&:hover": {
                                color: "primary.main",
                                bgcolor: "action.hover",
                              },
                            }}
                          >
                            {question}
                          </Typography>
                        ))}
                      </Box>
                    </CardContent>
                  </Card>
                </Box>
              );
            })}
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
          sessionId={currentSessionId}
          userRole={userRole}
        />
      ))}

      {/* Streaming message */}
      {isStreaming && streamingContent && (
        <Box
          sx={{
            display: "flex",
            justifyContent: "flex-start",
            mb: 2,
            px: 2,
          }}
        >
          <Box sx={{ maxWidth: "75%" }}>
            <Box
              sx={{
                display: "flex",
                alignItems: "center",
                gap: 0.5,
                mb: 0.5,
              }}
            >
              <BotIcon fontSize="small" color="primary" />
              <Typography variant="caption" color="text.secondary">
                Flux RAG
              </Typography>
            </Box>
            <Paper
              elevation={1}
              sx={{ p: 2, borderRadius: 2, bgcolor: "grey.50" }}
            >
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
            </Paper>
          </Box>
        </Box>
      )}

      {/* Bouncing dots typing indicator */}
      {isStreaming && !streamingContent && (
        <Box
          sx={{
            display: "flex",
            justifyContent: "flex-start",
            mb: 2,
            px: 2,
          }}
        >
          <Box sx={{ display: "flex", gap: 0.5, alignItems: "center", p: 2 }}>
            <BotIcon fontSize="small" color="primary" />
            <Typography variant="body2" color="text.secondary" sx={{ mr: 1 }}>
              답변 준비 중
            </Typography>
            {[0, 1, 2].map((i) => (
              <Box
                key={i}
                sx={{
                  width: 8,
                  height: 8,
                  borderRadius: "50%",
                  bgcolor: "primary.main",
                  animation: "bounce 1.4s ease-in-out infinite",
                  animationDelay: `${i * 0.16}s`,
                  "@keyframes bounce": {
                    "0%, 80%, 100%": { transform: "scale(0.6)", opacity: 0.4 },
                    "40%": { transform: "scale(1)", opacity: 1 },
                  },
                }}
              />
            ))}
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
