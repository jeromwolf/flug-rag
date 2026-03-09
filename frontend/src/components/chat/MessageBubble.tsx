import {
  Box,
  Avatar,
  Checkbox,
  Chip,
  Collapse,
  CircularProgress,
  Typography,
  Tooltip,
  IconButton,
  useTheme,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Popover,
} from "@mui/material";
import {
  CheckCircle as CheckCircleIcon,
  RemoveCircleOutline as RemoveCircleOutlineIcon,
  Cancel as CancelIcon,
  ContentCopy as CopyIcon,
  Check as CheckIcon,
  ReportProblem as ReportProblemIcon,
  Edit as EditIcon,
  AutoAwesome as AutoAwesomeIcon,
  Person as PersonIcon,
  Bookmark as BookmarkIcon,
  BookmarkBorder as BookmarkBorderIcon,
  Replay as ReplayIcon,
  Search as SearchIcon,
  SmartToy as SmartToyIcon,
  CallMerge as MergeIcon,
  ModelTraining as ModelTrainingIcon,
  Speed as SpeedIcon,
  Source as SourceIcon,
  PushPin as PushPinIcon,
  PushPinOutlined as PushPinOutlinedIcon,
  CallSplit as ForkIcon,
  AddReaction as AddReactionIcon,
  Translate as TranslateIcon,
  GTranslate as GTranslateIcon,
} from "@mui/icons-material";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import {
  oneDark,
  oneLight,
} from "react-syntax-highlighter/dist/esm/styles/prism";
import { memo, useState, useCallback, isValidElement, useEffect, useRef } from "react";
import type { Message } from "../../types";
import { SourcesPanel } from "./SourcesPanel";
import { ImageLightbox } from "./ImageLightbox";
import { bookmarksApi } from "../../api/client";

// ── Confidence Gauge ──────────────────────────────────────────────────────────
const ConfidenceGauge = ({ score }: { score: number }) => {
  const pct = Math.round(score * 100);
  // Color thresholds
  const color =
    score >= 0.8 ? "#4caf50" : score >= 0.5 ? "#ff9800" : "#f44336";

  // SVG donut params
  const size = 28;
  const stroke = 3;
  const r = (size - stroke * 2) / 2;
  const circ = 2 * Math.PI * r;
  const dashOffset = circ * (1 - score);
  const cx = size / 2;
  const cy = size / 2;

  const levelLabel =
    score >= 0.8 ? "높음" : score >= 0.5 ? "중간" : "낮음";

  return (
    <Tooltip
      title={
        <Box>
          <Typography variant="caption" sx={{ fontWeight: 600, display: "block" }}>
            신뢰도: {pct}% ({levelLabel})
          </Typography>
        </Box>
      }
      arrow
    >
      <Box
        sx={{
          position: "relative",
          width: size,
          height: size,
          flexShrink: 0,
          cursor: "default",
        }}
      >
        <svg
          width={size}
          height={size}
          style={{ transform: "rotate(-90deg)" }}
        >
          {/* Track */}
          <circle
            cx={cx}
            cy={cy}
            r={r}
            fill="none"
            stroke="currentColor"
            strokeWidth={stroke}
            style={{ color: "rgba(128,128,128,0.18)" }}
          />
          {/* Fill arc */}
          <circle
            cx={cx}
            cy={cy}
            r={r}
            fill="none"
            stroke={color}
            strokeWidth={stroke}
            strokeDasharray={circ}
            strokeDashoffset={dashOffset}
            strokeLinecap="round"
            style={{ transition: "stroke-dashoffset 0.6s ease" }}
          />
        </svg>
        {/* Score label */}
        <Typography
          sx={{
            position: "absolute",
            inset: 0,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            fontSize: "0.55rem",
            fontWeight: 700,
            lineHeight: 1,
            color,
            letterSpacing: "-0.02em",
          }}
        >
          {pct}
        </Typography>
      </Box>
    </Tooltip>
  );
};

// ── Response Mode Badge ───────────────────────────────────────────────────────
const ResponseModeBadge = ({
  mode,
}: {
  mode: "rag" | "direct" | "hybrid";
}) => {
  const config = {
    rag: {
      label: "RAG",
      icon: <SearchIcon sx={{ fontSize: 11 }} />,
      color: "#2e7d32",
      borderColor: "#4caf50",
      bgColor: "rgba(76,175,80,0.08)",
    },
    direct: {
      label: "Direct",
      icon: <SmartToyIcon sx={{ fontSize: 11 }} />,
      color: "#1565c0",
      borderColor: "#2196f3",
      bgColor: "rgba(33,150,243,0.08)",
    },
    hybrid: {
      label: "Hybrid",
      icon: <MergeIcon sx={{ fontSize: 11 }} />,
      color: "#6a1b9a",
      borderColor: "#9c27b0",
      bgColor: "rgba(156,39,176,0.08)",
    },
  } as const;

  const { label, icon, color, borderColor, bgColor } = config[mode];

  return (
    <Box
      sx={{
        display: "inline-flex",
        alignItems: "center",
        gap: "3px",
        px: "6px",
        py: "2px",
        borderRadius: "999px",
        border: `1px solid ${borderColor}`,
        bgcolor: bgColor,
        color,
        fontSize: "0.68rem",
        fontWeight: 600,
        letterSpacing: "0.03em",
        lineHeight: 1.5,
        cursor: "default",
        userSelect: "none",
      }}
    >
      {icon}
      {label}
    </Box>
  );
};

// ── Model Chip ────────────────────────────────────────────────────────────────
const ModelChip = ({ model }: { model: string }) => {
  // Abbreviate long model names for display
  const displayName = model.length > 20 ? model.slice(0, 18) + "…" : model;
  return (
    <Tooltip title={`모델: ${model}`} arrow>
      <Box
        sx={{
          display: "inline-flex",
          alignItems: "center",
          gap: "3px",
          px: "6px",
          py: "2px",
          borderRadius: "999px",
          border: "1px solid",
          borderColor: "rgba(128,128,128,0.28)",
          bgcolor: "rgba(128,128,128,0.06)",
          color: "text.secondary",
          fontSize: "0.68rem",
          fontWeight: 500,
          letterSpacing: "0.02em",
          lineHeight: 1.5,
          cursor: "default",
          userSelect: "none",
        }}
      >
        <ModelTrainingIcon sx={{ fontSize: 11, opacity: 0.75 }} />
        {displayName}
      </Box>
    </Tooltip>
  );
};

// ── Performance Indicator (v3 - always visible, no tooltip needed) ────────────
const PerfIndicator = ({
  ttftMs,
  tps,
  latencyMs,
}: {
  ttftMs?: number;
  tps?: number;
  latencyMs?: number;
}) => {
  if (latencyMs == null && ttftMs == null && tps == null) return null;

  // Primary display: total response time in seconds — ALWAYS visible
  const latencyText = latencyMs != null ? `${(latencyMs / 1000).toFixed(1)}초` : null;

  const latencyColor =
    latencyMs == null
      ? "#9e9e9e"
      : latencyMs < 3000
        ? "#4caf50"
        : latencyMs < 8000
          ? "#ff9800"
          : "#f44336";

  return (
    <Box
      sx={{
        display: "inline-flex",
        alignItems: "center",
        gap: "4px",
        px: "8px",
        py: "3px",
        borderRadius: "999px",
        border: "1px solid",
        borderColor: latencyColor,
        bgcolor: `${latencyColor}14`,
        cursor: "default",
      }}
    >
      <SpeedIcon sx={{ fontSize: 14, color: latencyColor }} />
      {latencyText && (
        <Typography
          variant="caption"
          sx={{
            fontSize: "0.78rem",
            fontWeight: 700,
            color: latencyColor,
            lineHeight: 1,
          }}
        >
          {latencyText}
        </Typography>
      )}
      {tps != null && (
        <Typography
          variant="caption"
          sx={{
            fontSize: "0.68rem",
            fontWeight: 500,
            color: "text.secondary",
            lineHeight: 1,
            ml: "2px",
          }}
        >
          {tps.toFixed(0)}t/s
        </Typography>
      )}
    </Box>
  );
};

// ── Sources Count Badge ───────────────────────────────────────────────────────
const SourcesCountBadge = ({ count }: { count: number }) => {
  if (count === 0) return null;
  return (
    <Box
      sx={{
        display: "inline-flex",
        alignItems: "center",
        gap: "3px",
        px: "6px",
        py: "2px",
        borderRadius: "999px",
        border: "1px solid",
        borderColor: "primary.main",
        bgcolor: "rgba(16,163,127,0.08)",
        color: "primary.main",
        fontSize: "0.68rem",
        fontWeight: 600,
        lineHeight: 1.5,
        cursor: "default",
        userSelect: "none",
      }}
    >
      <SourceIcon sx={{ fontSize: 11 }} />
      {count} 출처
    </Box>
  );
};

// Normalize language alias to Prism-supported names
const normalizeLanguage = (lang: string): string => {
  const aliases: Record<string, string> = {
    js: "javascript",
    ts: "typescript",
    jsx: "jsx",
    tsx: "tsx",
    py: "python",
    sh: "bash",
    shell: "bash",
    zsh: "bash",
    yml: "yaml",
    md: "markdown",
    rs: "rust",
    rb: "ruby",
    cs: "csharp",
    cpp: "cpp",
    "c++": "cpp",
    kt: "kotlin",
    go: "go",
    java: "java",
    sql: "sql",
    html: "html",
    css: "css",
    scss: "scss",
    json: "json",
    xml: "xml",
    dockerfile: "docker",
    Dockerfile: "docker",
  };
  return aliases[lang] ?? lang;
};

// Code block with syntax highlighting and header bar
const CodeBlockWrapper = ({
  code,
  language,
}: {
  code: string;
  language: string;
}) => {
  const theme = useTheme();
  const isDark = theme.palette.mode === "dark";
  const [copied, setCopied] = useState(false);

  const lang = normalizeLanguage(language);
  const displayLang = language || "text";

  const handleCopy = () => {
    navigator.clipboard.writeText(code).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    });
  };

  return (
    <Box
      sx={{
        borderRadius: "8px",
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
          bgcolor: isDark ? "#1a1a1a" : "#e2e4e8",
          borderBottom: "1px solid",
          borderColor: isDark ? "rgba(255,255,255,0.06)" : "rgba(0,0,0,0.08)",
        }}
      >
        <Typography
          variant="caption"
          sx={{
            fontFamily: "inherit",
            color: isDark ? "rgba(255,255,255,0.5)" : "rgba(0,0,0,0.55)",
            fontSize: "0.75rem",
            textTransform: "lowercase",
            letterSpacing: "0.03em",
          }}
        >
          {displayLang}
        </Typography>
        <Tooltip title={copied ? "복사됨!" : "복사"}>
          <IconButton
            size="small"
            onClick={handleCopy}
            aria-label="코드 복사"
            sx={{
              p: 0.4,
              color: isDark ? "rgba(255,255,255,0.45)" : "rgba(0,0,0,0.45)",
              "&:hover": {
                color: isDark ? "rgba(255,255,255,0.9)" : "rgba(0,0,0,0.8)",
                bgcolor: isDark
                  ? "rgba(255,255,255,0.08)"
                  : "rgba(0,0,0,0.06)",
              },
              transition: "color 0.15s, background-color 0.15s",
            }}
          >
            {copied ? (
              <CheckIcon sx={{ fontSize: 14, color: "success.main" }} />
            ) : (
              <CopyIcon sx={{ fontSize: 14 }} />
            )}
          </IconButton>
        </Tooltip>
      </Box>

      {/* Syntax-highlighted code body */}
      <SyntaxHighlighter
        language={lang || "text"}
        style={isDark ? oneDark : oneLight}
        customStyle={{
          margin: 0,
          borderRadius: 0,
          fontSize: "0.84rem",
          lineHeight: 1.65,
          fontFamily:
            '"JetBrains Mono", "Fira Code", "Cascadia Code", monospace',
          padding: "1rem 1.25rem",
          background: isDark ? "#1e1e2e" : "#f8f8fa",
          overflowX: "auto",
        }}
        codeTagProps={{
          style: {
            fontFamily:
              '"JetBrains Mono", "Fira Code", "Cascadia Code", monospace',
          },
        }}
        wrapLongLines={false}
        showLineNumbers={code.split("\n").length > 8}
        lineNumberStyle={{
          minWidth: "2.5em",
          paddingRight: "1em",
          color: isDark ? "rgba(255,255,255,0.2)" : "rgba(0,0,0,0.25)",
          userSelect: "none",
          fontSize: "0.78rem",
        }}
      >
        {code}
      </SyntaxHighlighter>
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

// ── Poll Widget ───────────────────────────────────────────────────────────────

const POLL_STORAGE_KEY = "flux-rag-polls";

interface PollData {
  id: string;
  question: string;
  options: string[];
  mockCounts: number[];
}

interface PollVoteStore {
  [pollId: string]: number; // index of chosen option
}

function loadPollVotes(): PollVoteStore {
  try {
    const raw = localStorage.getItem(POLL_STORAGE_KEY);
    return raw ? (JSON.parse(raw) as PollVoteStore) : {};
  } catch {
    return {};
  }
}

function savePollVote(pollId: string, optionIndex: number): void {
  try {
    const votes = loadPollVotes();
    votes[pollId] = optionIndex;
    localStorage.setItem(POLL_STORAGE_KEY, JSON.stringify(votes));
  } catch {
    // ignore storage errors
  }
}

const PollWidget = ({
  poll,
  isDark,
}: {
  poll: PollData;
  isDark: boolean;
}) => {
  const [votes, setVotes] = useState<PollVoteStore>(loadPollVotes);
  const votedIndex = votes[poll.id] ?? null;
  const hasVoted = votedIndex !== null;

  // Per-option local vote counts: mockCounts + 1 for the chosen option
  const displayCounts = poll.options.map((_, i) => {
    const base = poll.mockCounts[i] ?? 0;
    return hasVoted && votedIndex === i ? base + 1 : base;
  });
  const totalVotes = displayCounts.reduce((a, b) => a + b, 0);

  // Animate result bars in after voting
  const [barsReady, setBarsReady] = useState(hasVoted);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, []);

  const handleVote = (idx: number) => {
    if (hasVoted) return;
    const next = { ...votes, [poll.id]: idx };
    setVotes(next);
    savePollVote(poll.id, idx);
    // Delay bar animation slightly so state settles
    timerRef.current = setTimeout(() => setBarsReady(true), 30);
  };

  const borderColor = isDark ? "rgba(255,255,255,0.10)" : "rgba(0,0,0,0.10)";
  const cardBg = isDark ? "rgba(255,255,255,0.04)" : "rgba(0,0,0,0.025)";

  return (
    <Box
      sx={{
        my: 1.5,
        borderRadius: "12px",
        border: `1px solid ${borderColor}`,
        bgcolor: cardBg,
        overflow: "hidden",
        maxWidth: 480,
      }}
    >
      {/* Question header */}
      <Box
        sx={{
          px: 2,
          py: 1.25,
          borderBottom: `1px solid ${borderColor}`,
          display: "flex",
          alignItems: "center",
          gap: 1,
        }}
      >
        <Box
          sx={{
            width: 20,
            height: 20,
            borderRadius: "50%",
            bgcolor: "primary.main",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            flexShrink: 0,
          }}
        >
          <Typography sx={{ fontSize: "0.6rem", color: "#fff", fontWeight: 800, lineHeight: 1 }}>
            ?
          </Typography>
        </Box>
        <Typography
          variant="body2"
          sx={{
            fontWeight: 600,
            fontSize: "0.875rem",
            lineHeight: 1.4,
            color: "text.primary",
          }}
        >
          {poll.question}
        </Typography>
      </Box>

      {/* Options */}
      <Box sx={{ px: 1.5, py: 1 }}>
        {poll.options.map((option, idx) => {
          const isChosen = votedIndex === idx;
          const pct =
            totalVotes > 0
              ? Math.round((displayCounts[idx] / totalVotes) * 100)
              : 0;

          return (
            <Box
              key={idx}
              component={hasVoted ? "div" : "button"}
              onClick={hasVoted ? undefined : () => handleVote(idx)}
              role={hasVoted ? undefined : "radio"}
              aria-checked={isChosen}
              tabIndex={hasVoted ? undefined : 0}
              onKeyDown={
                hasVoted
                  ? undefined
                  : (e: React.KeyboardEvent) => {
                      if (e.key === "Enter" || e.key === " ") {
                        e.preventDefault();
                        handleVote(idx);
                      }
                    }
              }
              sx={{
                display: "block",
                width: "100%",
                position: "relative",
                mb: 0.75,
                borderRadius: "8px",
                border: "1px solid",
                borderColor: isChosen
                  ? "primary.main"
                  : isDark
                    ? "rgba(255,255,255,0.10)"
                    : "rgba(0,0,0,0.10)",
                bgcolor: isChosen
                  ? isDark
                    ? "rgba(16,163,127,0.14)"
                    : "rgba(16,163,127,0.08)"
                  : isDark
                    ? "rgba(255,255,255,0.03)"
                    : "rgba(0,0,0,0.02)",
                overflow: "hidden",
                cursor: hasVoted ? "default" : "pointer",
                p: 0,
                textAlign: "left",
                fontFamily: "inherit",
                transition: "border-color 0.18s, background-color 0.18s",
                "&:hover": hasVoted
                  ? {}
                  : {
                      borderColor: "primary.main",
                      bgcolor: isDark
                        ? "rgba(16,163,127,0.10)"
                        : "rgba(16,163,127,0.06)",
                    },
                "&:last-child": { mb: 0 },
              }}
            >
              {/* Result fill bar (shown after voting) */}
              {hasVoted && (
                <Box
                  sx={{
                    position: "absolute",
                    inset: 0,
                    width: barsReady ? `${pct}%` : "0%",
                    background: isChosen
                      ? `linear-gradient(90deg, rgba(16,163,127,0.22), rgba(16,163,127,0.10))`
                      : isDark
                        ? "rgba(255,255,255,0.05)"
                        : "rgba(0,0,0,0.04)",
                    transition: "width 0.55s cubic-bezier(0.25,0.46,0.45,0.94)",
                    borderRadius: "8px",
                    pointerEvents: "none",
                  }}
                />
              )}

              {/* Option row content */}
              <Box
                sx={{
                  position: "relative",
                  display: "flex",
                  alignItems: "center",
                  gap: 1,
                  px: 1.5,
                  py: 1,
                  zIndex: 1,
                }}
              >
                {/* Radio circle or checkmark */}
                <Box
                  sx={{
                    width: 18,
                    height: 18,
                    borderRadius: "50%",
                    border: "1.5px solid",
                    borderColor: isChosen ? "primary.main" : "text.disabled",
                    bgcolor: isChosen ? "primary.main" : "transparent",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    flexShrink: 0,
                    transition: "background-color 0.18s, border-color 0.18s",
                  }}
                >
                  {isChosen && (
                    <Box
                      sx={{
                        width: 8,
                        height: 8,
                        borderRadius: "50%",
                        bgcolor: "#fff",
                      }}
                    />
                  )}
                </Box>

                {/* Option label */}
                <Typography
                  variant="body2"
                  sx={{
                    flex: 1,
                    fontSize: "0.84rem",
                    color: isChosen ? "primary.main" : "text.primary",
                    fontWeight: isChosen ? 600 : 400,
                    lineHeight: 1.4,
                    transition: "color 0.18s",
                  }}
                >
                  {option}
                </Typography>

                {/* Percentage (shown after voting) */}
                {hasVoted && (
                  <Typography
                    variant="caption"
                    sx={{
                      fontSize: "0.78rem",
                      fontWeight: isChosen ? 700 : 500,
                      color: isChosen ? "primary.main" : "text.secondary",
                      minWidth: 34,
                      textAlign: "right",
                      opacity: barsReady ? 1 : 0,
                      transition: "opacity 0.35s 0.2s",
                      flexShrink: 0,
                    }}
                  >
                    {pct}%
                  </Typography>
                )}
              </Box>
            </Box>
          );
        })}
      </Box>

      {/* Footer: vote count */}
      <Box
        sx={{
          px: 2,
          py: 0.75,
          borderTop: `1px solid ${borderColor}`,
          display: "flex",
          alignItems: "center",
          gap: 0.5,
        }}
      >
        <Typography
          variant="caption"
          sx={{
            fontSize: "0.72rem",
            color: "text.disabled",
            fontStyle: "italic",
          }}
        >
          {hasVoted
            ? `총 ${totalVotes.toLocaleString()}표`
            : `${totalVotes.toLocaleString()}명 참여 중 · 클릭하여 투표`}
        </Typography>
      </Box>
    </Box>
  );
};

// ── Poll content parser ────────────────────────────────────────────────────────

interface ContentSegmentText {
  type: "text";
  content: string;
}

interface ContentSegmentPoll {
  type: "poll";
  poll: PollData;
}

type ContentSegment = ContentSegmentText | ContentSegmentPoll;

/**
 * Splits message content into text and [poll:...][/poll] segments.
 * Generates deterministic mock vote counts from the poll id.
 */
function parsePollBlocks(content: string): ContentSegment[] {
  const POLL_RE = /\[poll:([^\]]+)\]([\s\S]*?)\[\/poll\]/g;
  const segments: ContentSegment[] = [];
  let lastIndex = 0;
  let match: RegExpExecArray | null;

  while ((match = POLL_RE.exec(content)) !== null) {
    const [fullMatch, question, body] = match;
    const start = match.index;

    if (start > lastIndex) {
      segments.push({ type: "text", content: content.slice(lastIndex, start) });
    }

    // Parse option lines: lines starting with "- " or "* "
    const options = body
      .split("\n")
      .map((l) => l.trim())
      .filter((l) => l.startsWith("- ") || l.startsWith("* "))
      .map((l) => l.slice(2).trim())
      .filter(Boolean);

    // Deterministic mock counts using a seeded hash of the poll id
    const pollId = `poll-${encodeURIComponent(question.trim())}`;
    const mockCounts = options.map((opt, i) => {
      let h = 0;
      const seed = pollId + opt + i;
      for (let j = 0; j < seed.length; j++) {
        h = (Math.imul(31, h) + seed.charCodeAt(j)) | 0;
      }
      return 10 + (Math.abs(h) % 41); // 10–50
    });

    segments.push({
      type: "poll",
      poll: { id: pollId, question: question.trim(), options, mockCounts },
    });

    lastIndex = start + fullMatch.length;
  }

  if (lastIndex < content.length) {
    segments.push({ type: "text", content: content.slice(lastIndex) });
  }

  return segments.length > 0 ? segments : [{ type: "text", content }];
}

// ── MessageBubble interface ───────────────────────────────────────────────────

interface MessageBubbleProps {
  msg: Message;
  onFeedback: (messageId: string, rating: number) => void;
  onErrorReport: (messageId: string) => void;
  onEditAnswer: (messageId: string, currentContent: string) => void;
  onEditUserMessage?: (messageId: string, content: string) => void;
  onRegenerate?: (messageId: string) => void;
  onSuggestedClick?: (question: string) => void;
  sessionId: string | null;
  userRole?: string;
  isBookmarked?: boolean;
  /** The user's query that produced this assistant message — used for keyword highlighting in source snippets. */
  query?: string;
  // Multi-select
  selectMode?: boolean;
  isSelected?: boolean;
  onToggleSelect?: (messageId: string) => void;
  // Pin
  isPinned?: boolean;
  onTogglePin?: (messageId: string) => void;
  // Fork
  onFork?: (messageId: string) => void;
  // Reactions
  reactions?: string[];
  onReact?: (messageId: string, emoji: string) => void;
}

const formatTime = (iso: string) => {
  const d = new Date(iso);
  return d.toLocaleTimeString("ko-KR", {
    hour: "numeric",
    minute: "2-digit",
    hour12: true,
  });
};

export const MessageBubble = memo(function MessageBubble({
  msg,
  onFeedback,
  onErrorReport,
  onEditAnswer,
  onEditUserMessage,
  onRegenerate,
  onSuggestedClick,
  sessionId,
  userRole,
  isBookmarked = false,
  query,
  selectMode = false,
  isSelected = false,
  onToggleSelect,
  isPinned = false,
  onTogglePin,
  onFork,
  reactions = [],
  onReact,
}: MessageBubbleProps) {
  const isUser = msg.role === "user";
  const theme = useTheme();
  const isDark = theme.palette.mode === "dark";

  const [bookmarked, setBookmarked] = useState(isBookmarked);
  const [bookmarkLoading, setBookmarkLoading] = useState(false);

  // Image lightbox state
  const [lightbox, setLightbox] = useState<{
    open: boolean;
    src: string;
    alt: string;
  }>({ open: false, src: "", alt: "" });

  const openLightbox = useCallback((src: string, alt: string) => {
    setLightbox({ open: true, src, alt });
  }, []);

  const closeLightbox = useCallback(() => {
    setLightbox((prev) => ({ ...prev, open: false }));
  }, []);

  // Emoji reaction picker
  const EMOJI_LIST = ["👍", "👎", "❤️", "😊", "🤔"];
  const [emojiAnchor, setEmojiAnchor] = useState<HTMLElement | null>(null);

  const handleOpenEmojiPicker = useCallback((e: React.MouseEvent<HTMLElement>) => {
    e.stopPropagation();
    setEmojiAnchor(e.currentTarget);
  }, []);

  const handleCloseEmojiPicker = useCallback(() => {
    setEmojiAnchor(null);
  }, []);

  const handleSelectEmoji = useCallback((emoji: string) => {
    onReact?.(msg.id, emoji);
    setEmojiAnchor(null);
  }, [onReact, msg.id]);

  // ── Translation state ─────────────────────────────────────────────────────
  const [translationState, setTranslationState] = useState<
    "idle" | "loading" | "done"
  >("idle");
  const [translatedText, setTranslatedText] = useState<string | null>(null);
  const [showTranslation, setShowTranslation] = useState(false);

  /** Returns true if the string contains Korean characters (Hangul block). */
  const isKorean = (text: string): boolean =>
    /[\uAC00-\uD7A3\u1100-\u11FF\u3130-\u318F]/.test(text);

  const mockTranslate = useCallback(
    (text: string): string => {
      const korean = isKorean(text);
      const preview = text.slice(0, 50);
      if (korean) {
        return `[English Translation]\n${preview}... (translated content)`;
      }
      return `[한국어 번역]\n${preview}... (번역된 내용)`;
    },
    []
  );

  const handleTranslateToggle = useCallback(async () => {
    if (translationState === "loading") return;

    if (showTranslation) {
      // Toggle OFF — hide without resetting cached text
      setShowTranslation(false);
      return;
    }

    if (translatedText !== null) {
      // Already translated — just show again
      setShowTranslation(true);
      return;
    }

    // Kick off mock translation
    setTranslationState("loading");
    const delay = 1000 + Math.random() * 800; // 1.0–1.8s mock delay
    await new Promise((res) => setTimeout(res, delay));
    const result = mockTranslate(msg.content);
    setTranslatedText(result);
    setTranslationState("done");
    setShowTranslation(true);
  }, [translationState, showTranslation, translatedText, mockTranslate, msg.content]);

  const translationDirection = isKorean(msg.content)
    ? "한국어 → English"
    : "English → 한국어";

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

  const senderLabel = isUser ? "사용자" : "AI 어시스턴트";
  const timeLabel = msg.createdAt ? ` (${formatTime(msg.createdAt)})` : "";

  const handleRowClick = () => {
    if (selectMode && onToggleSelect) {
      onToggleSelect(msg.id);
    }
  };

  return (
    <Box
      role="article"
      aria-label={`${senderLabel} 메시지${timeLabel}`}
      onClick={handleRowClick}
      sx={{
        display: "flex",
        justifyContent: "center",
        py: { xs: 1, sm: 1.5 },
        px: { xs: 1, sm: 2 },
        bgcolor: isSelected
          ? isDark
            ? "rgba(16,163,127,0.12)"
            : "rgba(16,163,127,0.08)"
          : isUser
            ? isDark
              ? "rgba(255,255,255,0.03)"
              : "rgba(0,0,0,0.02)"
            : "transparent",
        animation: "fadeInUp 0.3s ease-out",
        "&:hover .action-buttons": { opacity: selectMode ? 0 : 1 },
        cursor: selectMode ? "pointer" : "default",
        transition: "background-color 0.15s ease",
        outline: isSelected ? `1px solid rgba(16,163,127,0.35)` : "none",
        outlineOffset: -1,
      }}
    >
      {/* Checkbox for select mode */}
      {selectMode && (
        <Box
          sx={{
            display: "flex",
            alignItems: "flex-start",
            pt: 0.5,
            mr: 0.5,
            flexShrink: 0,
          }}
          onClick={(e) => e.stopPropagation()}
        >
          <Checkbox
            checked={isSelected}
            onChange={() => onToggleSelect?.(msg.id)}
            size="small"
            sx={{
              p: 0.25,
              color: "text.secondary",
              "&.Mui-checked": { color: "primary.main" },
            }}
            aria-label={`메시지 ${isSelected ? "선택 해제" : "선택"}`}
          />
        </Box>
      )}

      <Box
        sx={{
          maxWidth: { xs: "100%", sm: 768 },
          width: "100%",
          display: "flex",
          gap: { xs: 1, sm: 2 },
        }}
      >
        {/* Avatar */}
        <Avatar
          sx={{
            width: { xs: 24, sm: 28 },
            height: { xs: 24, sm: 28 },
            bgcolor: isUser ? "#5436DA" : "primary.main",
            flexShrink: 0,
            mt: 0.25,
          }}
        >
          {isUser ? (
            <PersonIcon sx={{ fontSize: { xs: 14, sm: 16 } }} />
          ) : (
            <AutoAwesomeIcon sx={{ fontSize: { xs: 14, sm: 16 } }} />
          )}
        </Avatar>

        {/* Content column */}
        <Box sx={{ flex: 1, minWidth: 0 }}>
          {/* Name label */}
          <Typography
            variant="subtitle2"
            sx={{ fontWeight: 600, mb: 0.5, lineHeight: 1.4 }}
          >
            {isUser ? (
              <>
                사용자
                {msg.createdAt && (
                  <Box
                    component="span"
                    sx={{ opacity: 0.5, fontWeight: 400, fontSize: "0.8rem", ml: 1 }}
                  >
                    {formatTime(msg.createdAt)}
                  </Box>
                )}
              </>
            ) : "AI 어시스턴트"}
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
                "& hr": { borderColor: "divider", my: 2 },
                "& a": { color: "primary.main" },
              }}
            >
              {parsePollBlocks(msg.content).map((seg, segIdx) =>
                seg.type === "poll" ? (
                  <PollWidget key={`poll-${segIdx}`} poll={seg.poll} isDark={isDark} />
                ) : (
                  <ReactMarkdown
                    key={`md-${segIdx}`}
                    remarkPlugins={[remarkGfm]}
                    components={{
                  // Suppress the default <pre> wrapper — CodeBlockWrapper is self-contained
                  pre: ({ children }: { children?: React.ReactNode }) => (
                    <>{children}</>
                  ),
                  code({
                    className,
                    children,
                  }: {
                    className?: string;
                    children?: React.ReactNode;
                  }) {
                    const match = /language-(\w+)/.exec(className || "");
                    // Extract raw text from children (handles string or ReactNode tree)
                    const extractText = (node: React.ReactNode): string => {
                      if (typeof node === "string") return node;
                      if (typeof node === "number") return String(node);
                      if (Array.isArray(node))
                        return node.map(extractText).join("");
                      if (isValidElement(node)) {
                        const el = node as React.ReactElement<{
                          children?: React.ReactNode;
                        }>;
                        return extractText(el.props.children);
                      }
                      return "";
                    };
                    const rawCode = extractText(children).replace(/\n$/, "");

                    // Inline code: no language class and no newlines
                    const isInline = !className && !rawCode.includes("\n");
                    if (isInline) {
                      return <InlineCode>{children}</InlineCode>;
                    }

                    const language = match?.[1] ?? "";
                    return (
                      <CodeBlockWrapper code={rawCode} language={language} />
                    );
                  },
                  table: ({ children }: { children?: React.ReactNode }) => (
                    <TableContainer
                      component={Paper}
                      variant="outlined"
                      sx={{ my: 1, borderRadius: 1, overflow: "auto" }}
                    >
                      <Table size="small">{children}</Table>
                    </TableContainer>
                  ),
                  thead: ({ children }: { children?: React.ReactNode }) => (
                    <TableHead>{children}</TableHead>
                  ),
                  tbody: ({ children }: { children?: React.ReactNode }) => (
                    <TableBody>{children}</TableBody>
                  ),
                  tr: ({ children }: { children?: React.ReactNode }) => (
                    <TableRow
                      sx={{
                        "&:nth-of-type(odd)": { bgcolor: "action.hover" },
                      }}
                    >
                      {children}
                    </TableRow>
                  ),
                  th: ({ children }: { children?: React.ReactNode }) => (
                    <TableCell
                      sx={{
                        fontWeight: 600,
                        bgcolor: isDark
                          ? "rgba(255,255,255,0.06)"
                          : "grey.100",
                        whiteSpace: "nowrap",
                      }}
                    >
                      {children}
                    </TableCell>
                  ),
                  td: ({ children }: { children?: React.ReactNode }) => (
                    <TableCell sx={{ whiteSpace: "nowrap" }}>
                      {children}
                    </TableCell>
                  ),
                  img: ({
                    src,
                    alt,
                  }: {
                    src?: string;
                    alt?: string;
                  }) => {
                    if (!src) return null;
                    const label = alt || src.split("/").pop() || "이미지";
                    return (
                      <Box
                        component="span"
                        sx={{
                          display: "inline-block",
                          position: "relative",
                          cursor: "zoom-in",
                          lineHeight: 0,
                          borderRadius: "6px",
                          overflow: "hidden",
                          my: 0.5,
                          "&:hover .md-img-overlay": { opacity: 1 },
                          "&:hover img": { transform: "scale(1.03)" },
                        }}
                        onClick={() => openLightbox(src, label)}
                        role="button"
                        tabIndex={0}
                        aria-label={`이미지 크게 보기: ${label}`}
                        onKeyDown={(e) => {
                          if (e.key === "Enter" || e.key === " ") {
                            e.preventDefault();
                            openLightbox(src, label);
                          }
                        }}
                      >
                        <Box
                          component="img"
                          src={src}
                          alt={label}
                          sx={{
                            maxWidth: "100%",
                            maxHeight: 400,
                            display: "block",
                            borderRadius: "6px",
                            border: "1px solid",
                            borderColor: isDark
                              ? "rgba(255,255,255,0.1)"
                              : "rgba(0,0,0,0.1)",
                            transition: "transform 0.22s ease",
                            objectFit: "contain",
                          }}
                        />
                        {/* Hover overlay hint */}
                        <Box
                          className="md-img-overlay"
                          sx={{
                            position: "absolute",
                            inset: 0,
                            bgcolor: "rgba(0,0,0,0.32)",
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "center",
                            opacity: 0,
                            transition: "opacity 0.18s",
                            borderRadius: "6px",
                          }}
                        >
                          <Typography
                            variant="caption"
                            sx={{
                              color: "#fff",
                              fontSize: "0.75rem",
                              fontWeight: 600,
                              letterSpacing: "0.04em",
                              textShadow: "0 1px 4px rgba(0,0,0,0.5)",
                            }}
                          >
                            클릭하여 크게 보기
                          </Typography>
                        </Box>
                      </Box>
                    );
                  },
                    }}
                  >
                    {seg.content}
                  </ReactMarkdown>
                )
              )}
            </Box>
          )}

          {/* Translation loading indicator */}
          {!isUser && translationState === "loading" && (
            <Box
              sx={{
                display: "flex",
                alignItems: "center",
                gap: 1,
                mt: 1.25,
                px: 1.5,
                py: 1,
                borderRadius: "8px",
                bgcolor: isDark
                  ? "rgba(255,255,255,0.04)"
                  : "rgba(0,0,0,0.03)",
                border: "1px dashed",
                borderColor: isDark
                  ? "rgba(255,255,255,0.12)"
                  : "rgba(0,0,0,0.12)",
              }}
            >
              <CircularProgress
                size={14}
                thickness={4}
                sx={{ color: "primary.main", flexShrink: 0 }}
              />
              <Typography
                variant="caption"
                sx={{
                  color: "text.secondary",
                  fontSize: "0.78rem",
                  fontStyle: "italic",
                }}
              >
                번역 중...
              </Typography>
            </Box>
          )}

          {/* Translation result panel */}
          {!isUser && translatedText !== null && (
            <Collapse in={showTranslation} timeout={260}>
              <Box
                sx={{
                  mt: 1.25,
                  pl: 1.5,
                  pr: 1.5,
                  pt: 1,
                  pb: 1.25,
                  borderTop: "2px dashed",
                  borderColor: isDark
                    ? "rgba(16,163,127,0.35)"
                    : "rgba(16,163,127,0.40)",
                  borderRadius: "0 0 8px 8px",
                  bgcolor: isDark
                    ? "rgba(16,163,127,0.06)"
                    : "rgba(16,163,127,0.04)",
                  position: "relative",
                  ml: 0,
                }}
              >
                {/* Header row: direction label + badge */}
                <Box
                  sx={{
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "space-between",
                    mb: 0.75,
                  }}
                >
                  <Box
                    sx={{
                      display: "inline-flex",
                      alignItems: "center",
                      gap: "4px",
                      color: "primary.main",
                      opacity: 0.75,
                    }}
                  >
                    <GTranslateIcon sx={{ fontSize: 13 }} />
                    <Typography
                      variant="caption"
                      sx={{
                        fontSize: "0.72rem",
                        fontWeight: 500,
                        letterSpacing: "0.02em",
                        color: "primary.main",
                        opacity: 0.9,
                      }}
                    >
                      {translationDirection}
                    </Typography>
                  </Box>
                  <Chip
                    label="번역됨"
                    size="small"
                    sx={{
                      height: 18,
                      fontSize: "0.67rem",
                      fontWeight: 600,
                      letterSpacing: "0.04em",
                      bgcolor: isDark
                        ? "rgba(16,163,127,0.18)"
                        : "rgba(16,163,127,0.12)",
                      color: "primary.main",
                      border: "1px solid",
                      borderColor: isDark
                        ? "rgba(16,163,127,0.35)"
                        : "rgba(16,163,127,0.30)",
                      "& .MuiChip-label": { px: "6px" },
                    }}
                  />
                </Box>

                {/* Translated content */}
                <Typography
                  variant="body2"
                  sx={{
                    fontStyle: "italic",
                    fontSize: "0.9rem",
                    lineHeight: 1.7,
                    color: isDark
                      ? "rgba(255,255,255,0.72)"
                      : "rgba(0,0,0,0.68)",
                    whiteSpace: "pre-wrap",
                  }}
                >
                  {translatedText}
                </Typography>
              </Box>
            </Collapse>
          )}

          {/* Reaction pills — shown for all messages when reactions exist */}
          {reactions.length > 0 && (
            <Box sx={{ display: "flex", flexWrap: "wrap", gap: 0.5, mt: 0.75 }}>
              {/* Count each unique emoji */}
              {Array.from(new Set(reactions)).map((emoji) => {
                const count = reactions.filter((e) => e === emoji).length;
                return (
                  <Box
                    key={emoji}
                    component="button"
                    onClick={() => onReact?.(msg.id, emoji)}
                    sx={{
                      display: "inline-flex",
                      alignItems: "center",
                      gap: "3px",
                      height: 20,
                      px: "6px",
                      border: "1px solid",
                      borderColor: isDark ? "rgba(255,255,255,0.18)" : "rgba(0,0,0,0.14)",
                      borderRadius: "10px",
                      bgcolor: isDark ? "rgba(255,255,255,0.07)" : "rgba(0,0,0,0.05)",
                      cursor: "pointer",
                      fontSize: "11px",
                      lineHeight: 1,
                      fontFamily: "inherit",
                      color: "text.primary",
                      transition: "background-color 0.15s",
                      "&:hover": {
                        bgcolor: isDark ? "rgba(255,255,255,0.13)" : "rgba(0,0,0,0.09)",
                      },
                    }}
                    aria-label={`${emoji} ${count}개, 클릭하여 제거`}
                  >
                    <span style={{ fontSize: 12 }}>{emoji}</span>
                    {count > 1 && (
                      <span style={{ fontSize: 11, opacity: 0.7 }}>{count}</span>
                    )}
                  </Box>
                );
              })}
            </Box>
          )}

          {/* Metadata + actions for assistant */}
          {!isUser && (
            <Box sx={{ mt: 1 }}>
              {/* Rich metadata row */}
              <Box
                sx={{
                  display: "flex",
                  alignItems: "center",
                  gap: "8px",
                  flexWrap: "wrap",
                  mb: 0.75,
                }}
              >
                {/* Timestamp */}
                {msg.createdAt && (
                  <Typography
                    variant="caption"
                    sx={{ opacity: 0.45, fontSize: "0.72rem", flexShrink: 0 }}
                  >
                    {formatTime(msg.createdAt)}
                  </Typography>
                )}

                {/* Divider dot */}
                {msg.createdAt &&
                  (msg.confidenceScore != null ||
                    msg.responseMode ||
                    msg.modelUsed ||
                    (msg.sources && msg.sources.length > 0)) && (
                    <Box
                      component="span"
                      sx={{
                        width: 3,
                        height: 3,
                        borderRadius: "50%",
                        bgcolor: "text.disabled",
                        flexShrink: 0,
                      }}
                    />
                  )}

                {/* Confidence SVG gauge */}
                {msg.confidenceScore != null && (
                  <ConfidenceGauge score={msg.confidenceScore} />
                )}

                {/* Response mode badge */}
                {msg.responseMode && (
                  <ResponseModeBadge mode={msg.responseMode} />
                )}

                {/* Model chip */}
                {msg.modelUsed && <ModelChip model={msg.modelUsed} />}

                {/* Performance icon with tooltip */}
                {(msg.ttftMs != null ||
                  msg.tps != null ||
                  msg.latencyMs != null) && (
                  <PerfIndicator
                    ttftMs={msg.ttftMs}
                    tps={msg.tps}
                    latencyMs={msg.latencyMs}
                  />
                )}

                {/* Sources count badge */}
                {msg.sources &&
                  msg.sources.length > 0 &&
                  (msg.confidenceScore == null ||
                    msg.confidenceScore >= 0.3) && (
                    <SourcesCountBadge count={msg.sources.length} />
                  )}
              </Box>

              {/* Sources panel — hide when confidence is very low */}
              {msg.sources && msg.sources.length > 0 && (msg.confidenceScore == null || msg.confidenceScore >= 0.3) && (
                <SourcesPanel sources={msg.sources} query={query} />
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
                    aria-label="답변 복사"
                    sx={{ color: "text.secondary" }}
                  >
                    <CopyIcon sx={{ fontSize: 16 }} />
                  </IconButton>
                </Tooltip>
                <Tooltip title={showTranslation ? "원문 보기" : "번역"}>
                  <IconButton
                    size="small"
                    onClick={handleTranslateToggle}
                    aria-label={showTranslation ? "원문 보기" : "번역"}
                    disabled={translationState === "loading"}
                    sx={{
                      color: showTranslation
                        ? "primary.main"
                        : "text.secondary",
                      "&:hover": { color: "primary.main" },
                      transition: "color 0.15s",
                    }}
                  >
                    <TranslateIcon sx={{ fontSize: 16 }} />
                  </IconButton>
                </Tooltip>
                {onRegenerate && (
                  <Tooltip title="다시 생성">
                    <IconButton
                      size="small"
                      onClick={() => onRegenerate(msg.id)}
                      aria-label="다시 생성"
                      sx={{ color: "text.secondary" }}
                    >
                      <ReplayIcon sx={{ fontSize: 16 }} />
                    </IconButton>
                  </Tooltip>
                )}
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
                {onTogglePin && (
                  <Tooltip title={isPinned ? "고정 해제" : "고정"}>
                    <IconButton
                      size="small"
                      onClick={() => onTogglePin(msg.id)}
                      aria-label={isPinned ? "고정 해제" : "고정"}
                      sx={{
                        color: isPinned ? "primary.main" : "text.secondary",
                        "&:hover": { color: "primary.main" },
                      }}
                    >
                      {isPinned ? (
                        <PushPinIcon sx={{ fontSize: 16 }} />
                      ) : (
                        <PushPinOutlinedIcon sx={{ fontSize: 16 }} />
                      )}
                    </IconButton>
                  </Tooltip>
                )}
                {onFork && (
                  <Tooltip title="여기서 분기">
                    <IconButton
                      size="small"
                      onClick={() => onFork(msg.id)}
                      aria-label="여기서 분기"
                      sx={{ color: "text.secondary", "&:hover": { color: "info.main" } }}
                    >
                      <ForkIcon sx={{ fontSize: 16 }} />
                    </IconButton>
                  </Tooltip>
                )}
                {onReact && (
                  <Tooltip title="이모지 반응">
                    <IconButton
                      size="small"
                      onClick={handleOpenEmojiPicker}
                      aria-label="이모지 반응 추가"
                      sx={{ color: "text.secondary", "&:hover": { color: "primary.main" } }}
                    >
                      <AddReactionIcon sx={{ fontSize: 16 }} />
                    </IconButton>
                  </Tooltip>
                )}
              </Box>
            </Box>
          )}

          {/* Suggested follow-up questions */}
          {!isUser && msg.suggestedQuestions && msg.suggestedQuestions.length > 0 && onSuggestedClick && (
            <Box sx={{ display: "flex", flexWrap: "wrap", gap: 0.75, mt: 1 }}>
              {msg.suggestedQuestions.map((q, i) => (
                <Chip
                  key={i}
                  label={q}
                  size="small"
                  variant="outlined"
                  onClick={() => onSuggestedClick(q)}
                  sx={{
                    cursor: "pointer",
                    borderColor: "primary.main",
                    color: "primary.main",
                    fontSize: "0.8rem",
                    maxWidth: 320,
                    "&:hover": {
                      bgcolor: "primary.main",
                      color: "primary.contrastText",
                    },
                  }}
                />
              ))}
            </Box>
          )}

          {/* Action buttons for user messages */}
          {isUser && (
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
              <Tooltip title="복사">
                <IconButton
                  size="small"
                  onClick={handleCopy}
                  aria-label="복사"
                  sx={{ color: "text.secondary" }}
                >
                  <CopyIcon sx={{ fontSize: 16 }} />
                </IconButton>
              </Tooltip>
              {onEditUserMessage && (
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
              )}
              {onTogglePin && (
                <Tooltip title={isPinned ? "고정 해제" : "고정"}>
                  <IconButton
                    size="small"
                    onClick={() => onTogglePin(msg.id)}
                    aria-label={isPinned ? "고정 해제" : "고정"}
                    sx={{
                      color: isPinned ? "primary.main" : "text.secondary",
                      "&:hover": { color: "primary.main" },
                    }}
                  >
                    {isPinned ? (
                      <PushPinIcon sx={{ fontSize: 16 }} />
                    ) : (
                      <PushPinOutlinedIcon sx={{ fontSize: 16 }} />
                    )}
                  </IconButton>
                </Tooltip>
              )}
              {onReact && (
                <Tooltip title="이모지 반응">
                  <IconButton
                    size="small"
                    onClick={handleOpenEmojiPicker}
                    aria-label="이모지 반응 추가"
                    sx={{ color: "text.secondary", "&:hover": { color: "primary.main" } }}
                  >
                    <AddReactionIcon sx={{ fontSize: 16 }} />
                  </IconButton>
                </Tooltip>
              )}
            </Box>
          )}
        </Box>
      </Box>

      {/* Image lightbox */}
      <ImageLightbox
        open={lightbox.open}
        src={lightbox.src}
        alt={lightbox.alt}
        onClose={closeLightbox}
      />

      {/* Emoji picker popover */}
      <Popover
        open={Boolean(emojiAnchor)}
        anchorEl={emojiAnchor}
        onClose={handleCloseEmojiPicker}
        anchorOrigin={{ vertical: "top", horizontal: "left" }}
        transformOrigin={{ vertical: "bottom", horizontal: "left" }}
        slotProps={{
          paper: {
            sx: {
              p: "6px 8px",
              borderRadius: "10px",
              boxShadow: isDark
                ? "0 4px 16px rgba(0,0,0,0.5)"
                : "0 4px 16px rgba(0,0,0,0.15)",
              border: "1px solid",
              borderColor: isDark ? "rgba(255,255,255,0.1)" : "rgba(0,0,0,0.08)",
            },
          },
        }}
      >
        <Box sx={{ display: "flex", gap: "2px" }}>
          {EMOJI_LIST.map((emoji) => {
            const isActive = reactions.includes(emoji);
            return (
              <Box
                key={emoji}
                component="button"
                onClick={() => handleSelectEmoji(emoji)}
                aria-label={`${emoji} 반응${isActive ? " (선택됨)" : ""}`}
                sx={{
                  width: 32,
                  height: 32,
                  border: "1px solid",
                  borderColor: isActive
                    ? "primary.main"
                    : "transparent",
                  borderRadius: "6px",
                  bgcolor: isActive
                    ? isDark ? "rgba(16,163,127,0.18)" : "rgba(16,163,127,0.1)"
                    : "transparent",
                  cursor: "pointer",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  fontSize: "18px",
                  lineHeight: 1,
                  transition: "background-color 0.12s, transform 0.1s",
                  fontFamily: "inherit",
                  "&:hover": {
                    bgcolor: isDark ? "rgba(255,255,255,0.1)" : "rgba(0,0,0,0.07)",
                    transform: "scale(1.2)",
                  },
                }}
              >
                {emoji}
              </Box>
            );
          })}
        </Box>
      </Popover>
    </Box>
  );
});
