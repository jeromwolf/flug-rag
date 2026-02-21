import { Box, Paper, Typography, Tooltip, IconButton } from "@mui/material";
import {
  ThumbUp as ThumbUpIcon,
  ThumbDown as ThumbDownIcon,
  ContentCopy as CopyIcon,
  SmartToy as BotIcon,
  ReportProblem as ReportProblemIcon,
  Edit as EditIcon,
} from "@mui/icons-material";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { useRef } from "react";
import type { Message } from "../../types";
import { ConfidenceBadge } from "./ConfidenceBadge";
import { SourcesPanel } from "./SourcesPanel";

// Code block component with copy button
const CodeBlock = ({ children, ...props }: any) => {
  const ref = useRef<HTMLPreElement>(null);
  const handleCopy = () => {
    if (ref.current) {
      navigator.clipboard.writeText(ref.current.textContent || '');
    }
  };
  return (
    <Box sx={{ position: 'relative', '&:hover .copy-code-btn': { opacity: 1 } }}>
      <Tooltip title="복사">
        <IconButton
          className="copy-code-btn"
          size="small"
          onClick={handleCopy}
          sx={{
            position: 'absolute',
            top: 4,
            right: 4,
            opacity: 0,
            transition: 'opacity 0.2s',
            bgcolor: 'background.paper',
            boxShadow: 1,
            '&:hover': { bgcolor: 'action.hover' },
          }}
        >
          <CopyIcon sx={{ fontSize: 14 }} />
        </IconButton>
      </Tooltip>
      <pre ref={ref} {...props}>{children}</pre>
    </Box>
  );
};

interface MessageBubbleProps {
  msg: Message;
  onFeedback: (messageId: string, rating: number) => void;
  onErrorReport: (messageId: string) => void;
  onEditAnswer: (messageId: string, currentContent: string) => void;
  sessionId: string | null;
  userRole?: string;
}

export function MessageBubble({ msg, onFeedback, onErrorReport, onEditAnswer, sessionId, userRole }: MessageBubbleProps) {
  const isUser = msg.role === "user";

  const handleCopy = () => {
    navigator.clipboard.writeText(msg.content);
  };

  const canEdit = !isUser && (userRole === "admin" || userRole === "manager");

  return (
    <Box
      sx={{
        display: "flex",
        justifyContent: isUser ? "flex-end" : "flex-start",
        mb: 2,
        px: 2,
      }}
    >
      <Box
        sx={{
          maxWidth: "75%",
          minWidth: 80,
          '&:hover .action-buttons': { opacity: 1 }
        }}
      >
        {!isUser && (
          <Box sx={{ display: "flex", alignItems: "center", gap: 0.5, mb: 0.5 }}>
            <BotIcon fontSize="small" color="primary" />
            <Typography variant="caption" color="text.secondary">
              Flux RAG
            </Typography>
          </Box>
        )}

        <Paper
          elevation={0}
          sx={{
            p: 2,
            borderRadius: isUser ? 2.5 : 2,
            borderBottomRightRadius: isUser ? 0.5 : undefined,
            bgcolor: isUser ? "primary.main" : "action.hover",
            color: isUser ? "primary.contrastText" : "text.primary",
            boxShadow: isUser ? 'none' : '0 1px 3px rgba(0,0,0,0.08)',
            borderLeft: isUser ? 'none' : '3px solid',
            borderColor: isUser ? 'transparent' : 'primary.main',
          }}
        >
          {isUser ? (
            <Typography variant="body1" sx={{ whiteSpace: "pre-wrap" }}>
              {msg.content}
            </Typography>
          ) : (
            <Box
              sx={{
                "& p": { m: 0, mb: 1 },
                "& p:last-child": { mb: 0 },
                "& pre": {
                  bgcolor: "grey.100",
                  p: 1.5,
                  borderRadius: 1,
                  overflow: "auto",
                  fontSize: "0.85rem",
                },
                "& code": { fontSize: "0.85rem" },
                "& table": {
                  borderCollapse: "collapse",
                  width: "100%",
                  my: 1,
                },
                "& th, & td": {
                  border: "1px solid",
                  borderColor: "divider",
                  p: 0.5,
                  fontSize: "0.85rem",
                },
              }}
            >
              <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                components={{ pre: CodeBlock }}
              >
                {msg.content}
              </ReactMarkdown>
            </Box>
          )}
        </Paper>

        {!isUser && (
          <Box sx={{ mt: 0.5 }}>
            <Box
              sx={{
                display: "flex",
                alignItems: "center",
                gap: 1,
                flexWrap: "wrap",
              }}
            >
              {msg.confidenceScore != null && (
                <ConfidenceBadge score={msg.confidenceScore} />
              )}
              {msg.latencyMs != null && (
                <Typography variant="caption" sx={{ opacity: 0.7 }}>
                  {(msg.latencyMs / 1000).toFixed(1)}s
                </Typography>
              )}
              {msg.modelUsed && (
                <Typography variant="caption" sx={{ opacity: 0.7 }}>
                  {msg.modelUsed}
                </Typography>
              )}
            </Box>

            {msg.sources && msg.sources.length > 0 && (
              <SourcesPanel sources={msg.sources} />
            )}

            <Box
              className="action-buttons"
              sx={{
                display: "flex",
                gap: 0.5,
                mt: 0.5,
                opacity: 0,
                transition: 'opacity 0.2s'
              }}
            >
              {sessionId && (
                <>
                  <Tooltip title="도움이 됐어요">
                    <IconButton
                      size="small"
                      onClick={() => onFeedback(msg.id, 1)}
                      aria-label="긍정 평가"
                    >
                      <ThumbUpIcon fontSize="small" />
                    </IconButton>
                  </Tooltip>
                  <Tooltip title="도움이 안 됐어요">
                    <IconButton
                      size="small"
                      onClick={() => onFeedback(msg.id, -1)}
                      aria-label="부정 평가"
                    >
                      <ThumbDownIcon fontSize="small" />
                    </IconButton>
                  </Tooltip>
                  <Tooltip title="오류 신고">
                    <IconButton
                      size="small"
                      onClick={() => onErrorReport(msg.id)}
                      aria-label="오류 신고"
                    >
                      <ReportProblemIcon fontSize="small" />
                    </IconButton>
                  </Tooltip>
                  {canEdit && (
                    <Tooltip title="답변 수정">
                      <IconButton
                        size="small"
                        onClick={() => onEditAnswer(msg.id, msg.content)}
                        aria-label="답변 수정"
                      >
                        <EditIcon fontSize="small" />
                      </IconButton>
                    </Tooltip>
                  )}
                </>
              )}
              <Tooltip title="복사">
                <IconButton size="small" onClick={handleCopy}>
                  <CopyIcon fontSize="small" />
                </IconButton>
              </Tooltip>
            </Box>
          </Box>
        )}
      </Box>
    </Box>
  );
}
