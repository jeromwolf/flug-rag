import {
  Box,
  Avatar,
  Typography,
  Tooltip,
  IconButton,
  useTheme,
} from "@mui/material";
import {
  CheckCircle as CheckCircleIcon,
  RemoveCircleOutline as RemoveCircleOutlineIcon,
  Cancel as CancelIcon,
  ContentCopy as CopyIcon,
  ReportProblem as ReportProblemIcon,
  Edit as EditIcon,
  AutoAwesome as AutoAwesomeIcon,
  Person as PersonIcon,
  Bookmark as BookmarkIcon,
  BookmarkBorder as BookmarkBorderIcon,
} from "@mui/icons-material";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { memo, useRef, useState, useCallback } from "react";
import type { Message } from "../../types";
import { ConfidenceBadge } from "./ConfidenceBadge";
import { SourcesPanel } from "./SourcesPanel";
import { bookmarksApi } from "../../api/client";

// Code block with language header bar
const CodeBlockWrapper = ({
  children,
  language,
}: {
  children?: React.ReactNode;
  language: string;
}) => {
  const theme = useTheme();
  const isDark = theme.palette.mode === "dark";
  const ref = useRef<HTMLPreElement>(null);
  const [copied, setCopied] = useState(false);

  const lang = language;

  const handleCopy = () => {
    if (ref.current) {
      navigator.clipboard.writeText(ref.current.textContent || "");
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    }
  };

  return (
    <Box
      sx={{
        borderRadius: 1,
        overflow: "hidden",
        border: "1px solid",
        borderColor: isDark ? "rgba(255,255,255,0.1)" : "rgba(0,0,0,0.1)",
        my: 1.5,
        fontFamily:
          '"JetBrains Mono", "Fira Code", "Cascadia Code", monospace',
      }}
    >
      {/* Header bar */}
      <Box
        sx={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          px: 2,
          py: 0.75,
          bgcolor: isDark ? "#1a1a1a" : "#e8e8e8",
        }}
      >
        <Typography
          variant="caption"
          sx={{
            fontFamily: "inherit",
            color: isDark ? "rgba(255,255,255,0.5)" : "rgba(0,0,0,0.5)",
            fontSize: "0.75rem",
          }}
        >
          {lang || "code"}
        </Typography>
        <Tooltip title={copied ? "복사됨!" : "복사"}>
          <IconButton
            size="small"
            onClick={handleCopy}
            sx={{
              color: isDark ? "rgba(255,255,255,0.5)" : "rgba(0,0,0,0.5)",
              "&:hover": {
                color: isDark ? "rgba(255,255,255,0.9)" : "rgba(0,0,0,0.8)",
              },
            }}
          >
            <CopyIcon sx={{ fontSize: 14 }} />
          </IconButton>
        </Tooltip>
      </Box>
      {/* Code body */}
      <Box
        component="pre"
        ref={ref}
        sx={{
          m: 0,
          p: 2,
          bgcolor: isDark ? "#1e1e1e" : "#f6f8fa",
          overflow: "auto",
          fontSize: "0.85rem",
          lineHeight: 1.6,
          fontFamily: "inherit",
        }}
      >
        {children}
      </Box>
    </Box>
  );
};

// Inline code
const InlineCode = ({ children }: { children?: React.ReactNode }) => {
  const theme = useTheme();
  const isDark = theme.palette.mode === "dark";
  return (
    <Box
      component="code"
      sx={{
        px: 0.6,
        py: 0.2,
        borderRadius: 0.5,
        bgcolor: isDark ? "rgba(255,255,255,0.1)" : "rgba(0,0,0,0.07)",
        fontFamily:
          '"JetBrains Mono", "Fira Code", "Cascadia Code", monospace',
        fontSize: "0.875em",
      }}
    >
      {children}
    </Box>
  );
};

interface MessageBubbleProps {
  msg: Message;
  onFeedback: (messageId: string, rating: number) => void;
  onErrorReport: (messageId: string) => void;
  onEditAnswer: (messageId: string, currentContent: string) => void;
  onEditUserMessage?: (messageId: string, content: string) => void;
  sessionId: string | null;
  userRole?: string;
  isBookmarked?: boolean;
}

export const MessageBubble = memo(function MessageBubble({
  msg,
  onFeedback,
  onErrorReport,
  onEditAnswer,
  onEditUserMessage,
  sessionId,
  userRole,
  isBookmarked = false,
}: MessageBubbleProps) {
  const isUser = msg.role === "user";
  const theme = useTheme();
  const isDark = theme.palette.mode === "dark";

  const [bookmarked, setBookmarked] = useState(isBookmarked);
  const [bookmarkLoading, setBookmarkLoading] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(msg.content);
  };

  const handleBookmarkToggle = useCallback(async () => {
    if (bookmarkLoading || !sessionId) return;
    setBookmarkLoading(true);
    try {
      if (bookmarked) {
        await bookmarksApi.remove(msg.id);
        setBookmarked(false);
      } else {
        await bookmarksApi.add({
          message_id: msg.id,
          session_id: sessionId,
          content: msg.content,
          role: msg.role,
        });
        setBookmarked(true);
      }
    } catch {
      // 실패 시 상태 롤백 없이 조용히 처리
    } finally {
      setBookmarkLoading(false);
    }
  }, [bookmarked, bookmarkLoading, msg.id, msg.content, msg.role, sessionId]);

  const canEdit = !isUser && (userRole === "admin" || userRole === "manager");

  return (
    <Box
      sx={{
        display: "flex",
        justifyContent: "center",
        py: 1.5,
        px: 2,
        bgcolor: isUser
          ? isDark
            ? "rgba(255,255,255,0.03)"
            : "rgba(0,0,0,0.02)"
          : "transparent",
        animation: "fadeInUp 0.3s ease-out",
        "&:hover .action-buttons": { opacity: 1 },
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
        {/* Avatar */}
        <Avatar
          sx={{
            width: 28,
            height: 28,
            bgcolor: isUser ? "#5436DA" : "primary.main",
            flexShrink: 0,
            mt: 0.25,
          }}
        >
          {isUser ? (
            <PersonIcon sx={{ fontSize: 16 }} />
          ) : (
            <AutoAwesomeIcon sx={{ fontSize: 16 }} />
          )}
        </Avatar>

        {/* Content column */}
        <Box sx={{ flex: 1, minWidth: 0 }}>
          {/* Name label */}
          <Typography
            variant="subtitle2"
            sx={{ fontWeight: 600, mb: 0.5, lineHeight: 1.4 }}
          >
            {isUser ? "사용자" : "Flux AI"}
          </Typography>

          {/* Message content */}
          {isUser ? (
            <Typography variant="body1" sx={{ whiteSpace: "pre-wrap" }}>
              {msg.content}
            </Typography>
          ) : (
            <Box
              sx={{
                "& p": { m: 0, mb: 1 },
                "& p:last-child": { mb: 0 },
                "& ul, & ol": { pl: 2.5, mb: 1 },
                "& li": { mb: 0.5 },
                "& blockquote": {
                  borderLeft: "3px solid",
                  borderColor: "divider",
                  pl: 2,
                  ml: 0,
                  my: 1,
                  color: "text.secondary",
                },
                "& table": {
                  borderCollapse: "collapse",
                  width: "100%",
                  my: 1.5,
                  fontSize: "0.875rem",
                },
                "& th, & td": {
                  border: "1px solid",
                  borderColor: "divider",
                  px: 1.5,
                  py: 0.75,
                },
                "& th": {
                  bgcolor: isDark
                    ? "rgba(255,255,255,0.05)"
                    : "rgba(0,0,0,0.04)",
                  fontWeight: 600,
                },
                "& hr": { borderColor: "divider", my: 2 },
                "& a": { color: "primary.main" },
              }}
            >
              <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                components={{
                  pre: ({ children }: { children?: React.ReactNode }) => (
                    <Box sx={{ my: 1 }}>{children}</Box>
                  ),
                  code({ className, children }: { className?: string; children?: React.ReactNode }) {
                    const match = /language-(\w+)/.exec(className || "");
                    const isInline = !match && !String(children).includes("\n");
                    if (isInline) {
                      return <InlineCode>{children}</InlineCode>;
                    }
                    const language = match?.[1] ?? "";
                    return (
                      <CodeBlockWrapper language={language}>
                        {children}
                      </CodeBlockWrapper>
                    );
                  },
                }}
              >
                {msg.content}
              </ReactMarkdown>
            </Box>
          )}

          {/* Metadata + actions for assistant */}
          {!isUser && (
            <Box sx={{ mt: 1 }}>
              {/* Metadata row */}
              <Box
                sx={{
                  display: "flex",
                  alignItems: "center",
                  gap: 1,
                  flexWrap: "wrap",
                  mb: 0.5,
                }}
              >
                {msg.confidenceScore != null && (
                  <ConfidenceBadge score={msg.confidenceScore} />
                )}
                {msg.latencyMs != null && (
                  <Typography variant="caption" sx={{ opacity: 0.55 }}>
                    {(msg.latencyMs / 1000).toFixed(1)}s
                  </Typography>
                )}
                {msg.modelUsed && (
                  <Typography variant="caption" sx={{ opacity: 0.55 }}>
                    {msg.modelUsed}
                  </Typography>
                )}
              </Box>

              {/* Sources */}
              {msg.sources && msg.sources.length > 0 && (
                <SourcesPanel sources={msg.sources} />
              )}

              {/* Action buttons — visible on hover */}
              <Box
                className="action-buttons"
                sx={{
                  display: "flex",
                  gap: 0.25,
                  mt: 0.5,
                  opacity: 0,
                  transition: "opacity 0.2s",
                }}
              >
                {sessionId && (
                  <>
                    <Tooltip title="정확">
                      <IconButton
                        size="small"
                        onClick={() => onFeedback(msg.id, 1)}
                        aria-label="정확"
                        sx={{ color: "success.main" }}
                      >
                        <CheckCircleIcon sx={{ fontSize: 16 }} />
                      </IconButton>
                    </Tooltip>
                    <Tooltip title="부분정확">
                      <IconButton
                        size="small"
                        onClick={() => onFeedback(msg.id, 0)}
                        aria-label="부분정확"
                        sx={{ color: "warning.main" }}
                      >
                        <RemoveCircleOutlineIcon sx={{ fontSize: 16 }} />
                      </IconButton>
                    </Tooltip>
                    <Tooltip title="부정확">
                      <IconButton
                        size="small"
                        onClick={() => onFeedback(msg.id, -1)}
                        aria-label="부정확"
                        sx={{ color: "error.main" }}
                      >
                        <CancelIcon sx={{ fontSize: 16 }} />
                      </IconButton>
                    </Tooltip>
                    <Tooltip title="오류 신고">
                      <IconButton
                        size="small"
                        onClick={() => onErrorReport(msg.id)}
                        aria-label="오류 신고"
                        sx={{ color: "text.secondary" }}
                      >
                        <ReportProblemIcon sx={{ fontSize: 16 }} />
                      </IconButton>
                    </Tooltip>
                    {canEdit && (
                      <Tooltip title="답변 수정">
                        <IconButton
                          size="small"
                          onClick={() => onEditAnswer(msg.id, msg.content)}
                          aria-label="답변 수정"
                          sx={{ color: "text.secondary" }}
                        >
                          <EditIcon sx={{ fontSize: 16 }} />
                        </IconButton>
                      </Tooltip>
                    )}
                  </>
                )}
                <Tooltip title="복사">
                  <IconButton
                    size="small"
                    onClick={handleCopy}
                    sx={{ color: "text.secondary" }}
                  >
                    <CopyIcon sx={{ fontSize: 16 }} />
                  </IconButton>
                </Tooltip>
                {sessionId && (
                  <Tooltip title={bookmarked ? "북마크 해제" : "북마크"}>
                    <IconButton
                      size="small"
                      onClick={handleBookmarkToggle}
                      disabled={bookmarkLoading}
                      aria-label={bookmarked ? "북마크 해제" : "북마크"}
                      sx={{
                        color: bookmarked ? "warning.main" : "text.secondary",
                        "&:hover": { color: "warning.main" },
                      }}
                    >
                      {bookmarked ? (
                        <BookmarkIcon sx={{ fontSize: 16 }} />
                      ) : (
                        <BookmarkBorderIcon sx={{ fontSize: 16 }} />
                      )}
                    </IconButton>
                  </Tooltip>
                )}
              </Box>
            </Box>
          )}

          {/* Edit button for user messages */}
          {isUser && onEditUserMessage && (
            <Box
              className="action-buttons"
              sx={{
                display: "flex",
                mt: 0.5,
                opacity: 0,
                transition: "opacity 0.2s",
              }}
            >
              <Tooltip title="메시지 수정 후 재전송">
                <IconButton
                  size="small"
                  onClick={() => onEditUserMessage(msg.id, msg.content)}
                  aria-label="메시지 수정"
                  sx={{ color: "text.secondary" }}
                >
                  <EditIcon sx={{ fontSize: 16 }} />
                </IconButton>
              </Tooltip>
            </Box>
          )}
        </Box>
      </Box>
    </Box>
  );
});
