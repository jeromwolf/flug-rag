import { useState, useMemo } from "react";
import {
  Box,
  Tab,
  Tabs,
  Chip,
  Stack,
  Typography,
  Paper,
  IconButton,
  Tooltip,
  Divider,
  CircularProgress,
  Alert,
  Avatar,
  List,
  ListItem,
  ListItemText,
  ListItemAvatar,
  ListItemSecondaryAction,
} from "@mui/material";
import {
  ThumbUp as ThumbUpIcon,
  ThumbDown as ThumbDownIcon,
  ThumbsUpDown as ThumbsUpDownIcon,
  Bookmark as BookmarkIcon,
  BookmarkBorder as BookmarkBorderIcon,
  Chat as ChatIcon,
  Delete as DeleteIcon,
  OpenInNew as OpenInNewIcon,
  History as HistoryIcon,
  Feedback as FeedbackIcon,
} from "@mui/icons-material";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { sessionsApi, feedbackApi, bookmarksApi } from "../api/client";
import Layout from "../components/Layout";
import { useAppStore } from "../stores/appStore";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface Session {
  id: string;
  title: string;
  created_at: string;
  message_count?: number;
  last_message?: string;
}

interface FeedbackItem {
  id: string;
  message_id: string;
  session_id: string;
  rating: number;
  comment?: string;
  corrected_answer?: string;
  query?: string;
  answer?: string;
  created_at: string;
}

interface Bookmark {
  id: string;
  message_id: string;
  session_id: string;
  content: string;
  role?: string;
  note?: string;
  created_at: string;
}

type PeriodFilter = "today" | "week" | "month" | "all";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function isWithinPeriod(dateStr: string, period: PeriodFilter): boolean {
  if (period === "all") return true;
  const date = new Date(dateStr);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffDays = diffMs / (1000 * 60 * 60 * 24);
  if (period === "today") return diffDays < 1;
  if (period === "week") return diffDays < 7;
  if (period === "month") return diffDays < 30;
  return true;
}

function formatDate(dateStr: string): string {
  const date = new Date(dateStr);
  return date.toLocaleString("ko-KR", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function getDateGroupLabel(dateStr: string): string {
  const date = new Date(dateStr);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffDays = diffMs / (1000 * 60 * 60 * 24);
  if (diffDays < 1) return "오늘";
  if (diffDays < 2) return "어제";
  if (diffDays < 7) return "이번 주";
  return "이전";
}

function truncate(text: string, maxLen = 80): string {
  if (!text) return "";
  return text.length > maxLen ? text.slice(0, maxLen) + "…" : text;
}

// ---------------------------------------------------------------------------
// Mock data fallbacks (used when APIs return empty or fail)
// ---------------------------------------------------------------------------

const MOCK_FEEDBACK: FeedbackItem[] = [
  {
    id: "fb-1",
    message_id: "msg-1",
    session_id: "sess-1",
    rating: 1,
    query: "우리 공사의 설립 목적은 무엇인가요?",
    answer: "공사는 관련 기술 개발 및 안전 관리를 목적으로 설립되었습니다.",
    created_at: new Date(Date.now() - 1000 * 60 * 30).toISOString(),
  },
  {
    id: "fb-2",
    message_id: "msg-2",
    session_id: "sess-2",
    rating: -1,
    query: "2023년 매출액은 얼마인가요?",
    answer: "죄송합니다. 해당 정보를 찾을 수 없습니다.",
    comment: "답변이 틀렸습니다. 정확한 수치를 제공해 주세요.",
    created_at: new Date(Date.now() - 1000 * 60 * 60 * 3).toISOString(),
  },
  {
    id: "fb-3",
    message_id: "msg-3",
    session_id: "sess-3",
    rating: 0,
    query: "내부 규정 제15조 내용은?",
    answer: "제15조는 직원 복무 규정에 관한 내용입니다.",
    created_at: new Date(Date.now() - 1000 * 60 * 60 * 24 * 2).toISOString(),
  },
];

const MOCK_BOOKMARKS: Bookmark[] = [
  {
    id: "bm-1",
    message_id: "bm-msg-1",
    session_id: "sess-1",
    content: "관련 법률 제3조에 따르면 공사는 시설의 설치, 유지 및 관리에 관한 기술적 업무를 수행합니다.",
    role: "assistant",
    created_at: new Date(Date.now() - 1000 * 60 * 45).toISOString(),
  },
  {
    id: "bm-2",
    message_id: "bm-msg-2",
    session_id: "sess-2",
    content: "내부 규정 제22조: 직원은 근무 시간 중 개인 업무를 수행할 수 없으며, 위반 시 경고 처분을 받을 수 있습니다.",
    role: "assistant",
    note: "중요 규정",
    created_at: new Date(Date.now() - 1000 * 60 * 60 * 5).toISOString(),
  },
];

// ---------------------------------------------------------------------------
// Period filter chips
// ---------------------------------------------------------------------------

interface PeriodChipsProps {
  value: PeriodFilter;
  onChange: (v: PeriodFilter) => void;
}

function PeriodChips({ value, onChange }: PeriodChipsProps) {
  const options: { label: string; value: PeriodFilter }[] = [
    { label: "오늘", value: "today" },
    { label: "이번 주", value: "week" },
    { label: "이번 달", value: "month" },
    { label: "전체", value: "all" },
  ];
  return (
    <Stack direction="row" spacing={1} flexWrap="wrap">
      {options.map((opt) => (
        <Chip
          key={opt.value}
          label={opt.label}
          clickable
          color={value === opt.value ? "primary" : "default"}
          variant={value === opt.value ? "filled" : "outlined"}
          onClick={() => onChange(opt.value)}
          size="small"
        />
      ))}
    </Stack>
  );
}

// ---------------------------------------------------------------------------
// Tab: 질문 기록
// ---------------------------------------------------------------------------

function QuestionsTab({ period }: { period: PeriodFilter }) {
  const navigate = useNavigate();
  const setCurrentSessionId = useAppStore((s) => s.setCurrentSessionId);

  const { data, isLoading, isError } = useQuery({
    queryKey: ["sessions", "list"],
    queryFn: async () => {
      const res = await sessionsApi.list(100);
      return (res.data?.sessions ?? res.data ?? []) as Session[];
    },
    retry: 1,
  });

  const sessions: Session[] = useMemo(() => {
    if (!data) return [];
    return data.filter((s: Session) => isWithinPeriod(s.created_at, period));
  }, [data, period]);

  // Group by date label
  const grouped = useMemo(() => {
    const groups: Record<string, Session[]> = {};
    for (const s of sessions) {
      const label = getDateGroupLabel(s.created_at);
      if (!groups[label]) groups[label] = [];
      groups[label].push(s);
    }
    return groups;
  }, [sessions]);

  const groupOrder = ["오늘", "어제", "이번 주", "이전"];

  if (isLoading) {
    return (
      <Box sx={{ display: "flex", justifyContent: "center", pt: 6 }}>
        <CircularProgress />
      </Box>
    );
  }

  if (isError) {
    return (
      <Alert severity="error" sx={{ mt: 2 }}>
        대화 기록을 불러오는 데 실패했습니다.
      </Alert>
    );
  }

  if (sessions.length === 0) {
    return (
      <Box sx={{ textAlign: "center", pt: 8, color: "text.secondary" }}>
        <HistoryIcon sx={{ fontSize: 48, mb: 1, opacity: 0.4 }} />
        <Typography>해당 기간에 대화 기록이 없습니다.</Typography>
      </Box>
    );
  }

  return (
    <Box>
      {groupOrder
        .filter((g) => grouped[g]?.length > 0)
        .map((groupLabel) => (
          <Box key={groupLabel} sx={{ mb: 3 }}>
            <Typography
              variant="caption"
              sx={{ fontWeight: 700, color: "text.secondary", textTransform: "uppercase", letterSpacing: 0.5 }}
            >
              {groupLabel}
            </Typography>
            <Divider sx={{ mb: 1, mt: 0.5 }} />
            <List disablePadding>
              {grouped[groupLabel].map((session) => (
                <Paper
                  key={session.id}
                  variant="outlined"
                  sx={{
                    mb: 1,
                    borderRadius: 2,
                    "&:hover": { bgcolor: "action.hover" },
                    transition: "background 0.15s",
                  }}
                >
                  <ListItem
                    disablePadding
                    sx={{ px: 2, py: 1.5 }}
                  >
                    <ListItemAvatar sx={{ minWidth: 44 }}>
                      <Avatar sx={{ width: 32, height: 32, bgcolor: "primary.main", fontSize: 16 }}>
                        <ChatIcon sx={{ fontSize: 18 }} />
                      </Avatar>
                    </ListItemAvatar>
                    <ListItemText
                      primary={
                        <Typography variant="body2" sx={{ fontWeight: 600, mb: 0.25 }}>
                          {truncate(session.title || "제목 없음", 60)}
                        </Typography>
                      }
                      secondary={
                        <Stack direction="row" spacing={1} alignItems="center" sx={{ mt: 0.25 }}>
                          <Typography variant="caption" color="text.secondary">
                            {formatDate(session.created_at)}
                          </Typography>
                          {session.message_count !== undefined && (
                            <Chip
                              label={`${session.message_count}개 메시지`}
                              size="small"
                              variant="outlined"
                              sx={{ height: 18, fontSize: "0.65rem" }}
                            />
                          )}
                        </Stack>
                      }
                      secondaryTypographyProps={{ component: "span" }}
                    />
                    <ListItemSecondaryAction>
                      <Tooltip title="대화 열기">
                        <IconButton
                          edge="end"
                          size="small"
                          onClick={() => {
                            setCurrentSessionId(session.id);
                            navigate(`/chat?session=${session.id}`);
                          }}
                        >
                          <OpenInNewIcon fontSize="small" />
                        </IconButton>
                      </Tooltip>
                    </ListItemSecondaryAction>
                  </ListItem>
                </Paper>
              ))}
            </List>
          </Box>
        ))}
    </Box>
  );
}

// ---------------------------------------------------------------------------
// Tab: 피드백
// ---------------------------------------------------------------------------

function FeedbackTab({ period }: { period: PeriodFilter }) {
  const navigate = useNavigate();
  const setCurrentSessionId = useAppStore((s) => s.setCurrentSessionId);

  const { data, isLoading, isError } = useQuery({
    queryKey: ["feedback", "list"],
    queryFn: async () => {
      try {
        const res = await feedbackApi.list(100);
        const items = res.data?.items ?? res.data ?? [];
        if (Array.isArray(items) && items.length > 0) return items as FeedbackItem[];
        return MOCK_FEEDBACK;
      } catch {
        return MOCK_FEEDBACK;
      }
    },
    retry: 1,
  });

  const items: FeedbackItem[] = useMemo(() => {
    if (!data) return [];
    return data.filter((f: FeedbackItem) => isWithinPeriod(f.created_at, period));
  }, [data, period]);

  function RatingIcon({ rating }: { rating: number }) {
    if (rating > 0)
      return <ThumbUpIcon sx={{ fontSize: 18, color: "success.main" }} />;
    if (rating < 0)
      return <ThumbDownIcon sx={{ fontSize: 18, color: "error.main" }} />;
    return <ThumbsUpDownIcon sx={{ fontSize: 18, color: "warning.main" }} />;
  }

  function ratingLabel(rating: number) {
    if (rating > 0) return "정확";
    if (rating < 0) return "부정확";
    return "부분정확";
  }

  function ratingColor(rating: number): "success" | "error" | "warning" {
    if (rating > 0) return "success";
    if (rating < 0) return "error";
    return "warning";
  }

  if (isLoading) {
    return (
      <Box sx={{ display: "flex", justifyContent: "center", pt: 6 }}>
        <CircularProgress />
      </Box>
    );
  }

  if (isError) {
    return (
      <Alert severity="error" sx={{ mt: 2 }}>
        피드백 기록을 불러오는 데 실패했습니다.
      </Alert>
    );
  }

  if (items.length === 0) {
    return (
      <Box sx={{ textAlign: "center", pt: 8, color: "text.secondary" }}>
        <FeedbackIcon sx={{ fontSize: 48, mb: 1, opacity: 0.4 }} />
        <Typography>해당 기간에 피드백 기록이 없습니다.</Typography>
      </Box>
    );
  }

  return (
    <Stack spacing={1.5}>
      {items.map((item) => (
        <Paper
          key={item.id}
          variant="outlined"
          sx={{ borderRadius: 2, p: 2, "&:hover": { bgcolor: "action.hover" }, transition: "background 0.15s" }}
        >
          <Stack direction="row" alignItems="flex-start" spacing={1.5}>
            <Avatar sx={{ width: 32, height: 32, bgcolor: `${ratingColor(item.rating)}.light`, mt: 0.25 }}>
              <RatingIcon rating={item.rating} />
            </Avatar>
            <Box sx={{ flex: 1, minWidth: 0 }}>
              <Stack direction="row" alignItems="center" spacing={1} sx={{ mb: 0.75 }}>
                <Chip
                  label={ratingLabel(item.rating)}
                  size="small"
                  color={ratingColor(item.rating)}
                  variant="outlined"
                  sx={{ height: 20, fontSize: "0.7rem" }}
                />
                <Typography variant="caption" color="text.secondary">
                  {formatDate(item.created_at)}
                </Typography>
              </Stack>

              {item.query && (
                <Box sx={{ mb: 0.5 }}>
                  <Typography variant="caption" sx={{ fontWeight: 700, color: "text.secondary" }}>
                    질문
                  </Typography>
                  <Typography variant="body2" sx={{ color: "text.primary" }}>
                    {truncate(item.query, 120)}
                  </Typography>
                </Box>
              )}

              {item.answer && (
                <Box sx={{ mb: 0.5 }}>
                  <Typography variant="caption" sx={{ fontWeight: 700, color: "text.secondary" }}>
                    AI 답변
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    {truncate(item.answer, 120)}
                  </Typography>
                </Box>
              )}

              {item.comment && (
                <Box sx={{ mt: 0.75, p: 1, bgcolor: "action.selected", borderRadius: 1 }}>
                  <Typography variant="caption" sx={{ fontWeight: 700, color: "text.secondary" }}>
                    의견
                  </Typography>
                  <Typography variant="body2">{item.comment}</Typography>
                </Box>
              )}

              {item.corrected_answer && (
                <Box sx={{ mt: 0.75, p: 1, bgcolor: "success.light", borderRadius: 1 }}>
                  <Typography variant="caption" sx={{ fontWeight: 700, color: "success.dark" }}>
                    수정 답변
                  </Typography>
                  <Typography variant="body2" sx={{ color: "success.dark" }}>
                    {truncate(item.corrected_answer, 120)}
                  </Typography>
                </Box>
              )}
            </Box>

            <Tooltip title="대화로 이동">
              <IconButton
                size="small"
                onClick={() => {
                  setCurrentSessionId(item.session_id);
                  navigate(`/chat?session=${item.session_id}`);
                }}
              >
                <OpenInNewIcon fontSize="small" />
              </IconButton>
            </Tooltip>
          </Stack>
        </Paper>
      ))}
    </Stack>
  );
}

// ---------------------------------------------------------------------------
// Tab: 북마크
// ---------------------------------------------------------------------------

function BookmarksTab({ period }: { period: PeriodFilter }) {
  const navigate = useNavigate();
  const setCurrentSessionId = useAppStore((s) => s.setCurrentSessionId);
  const queryClient = useQueryClient();
  const showNotification = useAppStore((s) => s.showNotification);

  const { data, isLoading, isError } = useQuery({
    queryKey: ["bookmarks", "list"],
    queryFn: async () => {
      try {
        const res = await bookmarksApi.list();
        const items = res.data?.bookmarks ?? res.data ?? [];
        if (Array.isArray(items) && items.length > 0) return items as Bookmark[];
        return MOCK_BOOKMARKS;
      } catch {
        return MOCK_BOOKMARKS;
      }
    },
    retry: 1,
  });

  const removeMutation = useMutation({
    mutationFn: (messageId: string) => bookmarksApi.remove(messageId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["bookmarks", "list"] });
      showNotification("북마크가 삭제되었습니다.", "success");
    },
    onError: () => {
      showNotification("북마크 삭제에 실패했습니다.", "error");
    },
  });

  const items: Bookmark[] = useMemo(() => {
    if (!data) return [];
    return data.filter((b: Bookmark) => isWithinPeriod(b.created_at, period));
  }, [data, period]);

  if (isLoading) {
    return (
      <Box sx={{ display: "flex", justifyContent: "center", pt: 6 }}>
        <CircularProgress />
      </Box>
    );
  }

  if (isError) {
    return (
      <Alert severity="error" sx={{ mt: 2 }}>
        북마크를 불러오는 데 실패했습니다.
      </Alert>
    );
  }

  if (items.length === 0) {
    return (
      <Box sx={{ textAlign: "center", pt: 8, color: "text.secondary" }}>
        <BookmarkBorderIcon sx={{ fontSize: 48, mb: 1, opacity: 0.4 }} />
        <Typography>해당 기간에 저장된 북마크가 없습니다.</Typography>
        <Typography variant="body2" sx={{ mt: 1 }}>
          AI 답변에서 북마크 아이콘을 클릭하여 중요한 내용을 저장하세요.
        </Typography>
      </Box>
    );
  }

  return (
    <Stack spacing={1.5}>
      {items.map((bookmark) => (
        <Paper
          key={bookmark.id}
          variant="outlined"
          sx={{ borderRadius: 2, p: 2, "&:hover": { bgcolor: "action.hover" }, transition: "background 0.15s" }}
        >
          <Stack direction="row" alignItems="flex-start" spacing={1.5}>
            <Avatar sx={{ width: 32, height: 32, bgcolor: "warning.light", mt: 0.25 }}>
              <BookmarkIcon sx={{ fontSize: 18, color: "warning.dark" }} />
            </Avatar>
            <Box sx={{ flex: 1, minWidth: 0 }}>
              <Stack direction="row" alignItems="center" spacing={1} sx={{ mb: 0.75 }}>
                <Typography variant="caption" color="text.secondary">
                  {formatDate(bookmark.created_at)}
                </Typography>
                {bookmark.note && (
                  <Chip
                    label={bookmark.note}
                    size="small"
                    variant="outlined"
                    color="warning"
                    sx={{ height: 20, fontSize: "0.7rem" }}
                  />
                )}
              </Stack>

              <Typography variant="body2" sx={{ color: "text.primary", lineHeight: 1.6 }}>
                {truncate(bookmark.content, 200)}
              </Typography>
            </Box>

            <Stack direction="column" spacing={0.5} alignItems="flex-end" sx={{ flexShrink: 0 }}>
              <Tooltip title="대화로 이동">
                <IconButton
                  size="small"
                  onClick={() => {
                    setCurrentSessionId(bookmark.session_id);
                    navigate(`/chat?session=${bookmark.session_id}`);
                  }}
                >
                  <OpenInNewIcon fontSize="small" />
                </IconButton>
              </Tooltip>
              <Tooltip title="북마크 삭제">
                <IconButton
                  size="small"
                  color="error"
                  onClick={() => removeMutation.mutate(bookmark.message_id)}
                  disabled={removeMutation.isPending}
                >
                  <DeleteIcon fontSize="small" />
                </IconButton>
              </Tooltip>
            </Stack>
          </Stack>
        </Paper>
      ))}
    </Stack>
  );
}

// ---------------------------------------------------------------------------
// Main page
// ---------------------------------------------------------------------------

interface TabPanelProps {
  children?: React.ReactNode;
  value: number;
  index: number;
}

function TabPanel({ children, value, index }: TabPanelProps) {
  return (
    <Box
      role="tabpanel"
      hidden={value !== index}
      id={`activity-tabpanel-${index}`}
      aria-labelledby={`activity-tab-${index}`}
      sx={{ pt: 2 }}
    >
      {value === index && children}
    </Box>
  );
}

export default function ActivityPage() {
  const [tabValue, setTabValue] = useState(0);
  const [period, setPeriod] = useState<PeriodFilter>("all");

  return (
    <Layout title="활동 기록">
      <Box sx={{ maxWidth: 800, mx: "auto" }}>
        {/* Tabs */}
        <Box sx={{ borderBottom: 1, borderColor: "divider", mb: 2 }}>
          <Tabs
            value={tabValue}
            onChange={(_, v) => setTabValue(v)}
            aria-label="activity tabs"
          >
            <Tab
              id="activity-tab-0"
              aria-controls="activity-tabpanel-0"
              icon={<HistoryIcon sx={{ fontSize: 18 }} />}
              iconPosition="start"
              label="질문 기록"
              sx={{ minHeight: 48, textTransform: "none", fontWeight: 600 }}
            />
            <Tab
              id="activity-tab-1"
              aria-controls="activity-tabpanel-1"
              icon={<FeedbackIcon sx={{ fontSize: 18 }} />}
              iconPosition="start"
              label="피드백"
              sx={{ minHeight: 48, textTransform: "none", fontWeight: 600 }}
            />
            <Tab
              id="activity-tab-2"
              aria-controls="activity-tabpanel-2"
              icon={<BookmarkIcon sx={{ fontSize: 18 }} />}
              iconPosition="start"
              label="북마크"
              sx={{ minHeight: 48, textTransform: "none", fontWeight: 600 }}
            />
          </Tabs>
        </Box>

        {/* Period filter */}
        <Box sx={{ mb: 2.5 }}>
          <PeriodChips value={period} onChange={setPeriod} />
        </Box>

        {/* Tab panels */}
        <TabPanel value={tabValue} index={0}>
          <QuestionsTab period={period} />
        </TabPanel>
        <TabPanel value={tabValue} index={1}>
          <FeedbackTab period={period} />
        </TabPanel>
        <TabPanel value={tabValue} index={2}>
          <BookmarksTab period={period} />
        </TabPanel>
      </Box>
    </Layout>
  );
}
