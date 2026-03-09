import { useState, useRef, useEffect, useCallback, useMemo } from "react";
import {
  Box,
  List,
  ListItemButton,
  ListItemText,
  ListItemIcon,
  IconButton,
  Typography,
  Button,
  Divider,
  Tooltip,
  TextField,
  InputAdornment,
  Collapse,
  Popover,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Chip,
  Stack,
  Menu,
  MenuItem,
} from "@mui/material";
import {
  Chat as ChatIcon,
  Description as DescriptionIcon,
  AdminPanelSettings as AdminIcon,
  Monitor as MonitorIcon,
  Add as AddIcon,
  Delete as DeleteIcon,
  AutoAwesome as AutoAwesomeIcon,
  Search as SearchIcon,
  Bookmark as BookmarkIcon,
  ExpandLess,
  ExpandMore,
  BookmarkRemove as BookmarkRemoveIcon,
  HelpOutline as HelpOutlineIcon,
  Close as CloseIcon,
  Edit as EditIcon,
  CallSplit as ForkIcon,
  Label as LabelIcon,
  Settings as SettingsIcon,
  Check as CheckIcon,
  Sort as SortIcon,
  StarBorder as StarBorderIcon,
  Star as StarIcon,
  Logout as LogoutIcon,
} from "@mui/icons-material";
import { useNavigate, useLocation } from "react-router-dom";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { bookmarksApi } from "../../api/client";
import { useAuth } from "../../contexts/AuthContext";
import { HelpDialog } from "./HelpDialog";
import LoadingSkeleton from "../LoadingSkeleton";
import { useSessionTags, PRESET_TAGS } from "../../hooks/useSessionTags";
import type { SessionTag } from "../../hooks/useSessionTags";
import { useLanguage } from "../../contexts/LanguageContext";

// ---------------------------------------------------------------------------
// Sort / Filter types and localStorage persistence helpers
// ---------------------------------------------------------------------------

type SortOption = "newest" | "oldest" | "most_messages" | "alpha";
type DateFilter = "today" | "week" | "month" | "all";

const LS_SORT_KEY = "flux-rag-sort";
const LS_DATE_FILTER_KEY = "flux-rag-date-filter";
const LS_FAVORITES_KEY = "flux-rag-favorites";

function lsGet<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(key);
    return raw !== null ? (JSON.parse(raw) as T) : fallback;
  } catch {
    return fallback;
  }
}

function lsSet(key: string, value: unknown): void {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    /* ignore */
  }
}

const SORT_LABELS: Record<SortOption, string> = {
  newest: "최신순",
  oldest: "오래된순",
  most_messages: "메시지 수 많은순",
  alpha: "제목 가나다순",
};

const DATE_FILTER_LABELS: Record<DateFilter, string> = {
  today: "오늘",
  week: "이번 주",
  month: "이번 달",
  all: "전체",
};

// ---------------------------------------------------------------------------
// Color palette for custom tag creation
// ---------------------------------------------------------------------------

const COLOR_SWATCHES = [
  "#ef4444", // red
  "#f97316", // orange
  "#f59e0b", // amber
  "#84cc16", // lime
  "#10b981", // emerald
  "#06b6d4", // cyan
  "#3b82f6", // blue
  "#8b5cf6", // violet
  "#ec4899", // pink
  "#6b7280", // gray
  "#1e293b", // slate
  "#78350f", // brown
];

// ---------------------------------------------------------------------------
// Interfaces
// ---------------------------------------------------------------------------

interface SessionItem {
  id: string;
  title: string;
  message_count?: number;
  messageCount?: number;
  created_at?: string;
  createdAt?: string;
  updated_at?: string;
  updatedAt?: string;
}

// ---------------------------------------------------------------------------
// Smart title truncation
// ---------------------------------------------------------------------------

const TRAILING_PARTICLES = /[은는이가을를에의로와과도에서]+$/;
const TRAILING_SUFFIXES = /\s*(에\s*대해|에\s*대한|알려\s*주세요|설명해\s*주세요)[?？]*$/;
const TRAILING_PUNCTUATION = /[?？\s]+$/;

function smartTruncateTitle(title: string, maxLen = 30): string {
  if (!title) return "새 대화";
  let t = title
    .replace(TRAILING_SUFFIXES, "")
    .replace(TRAILING_PUNCTUATION, "")
    .replace(TRAILING_PARTICLES, "")
    .trim();

  if (t.length <= maxLen) return t || title.slice(0, maxLen);

  const sub = t.slice(0, maxLen);
  const lastSpace = sub.lastIndexOf(" ");
  const cut = lastSpace > maxLen * 0.5 ? sub.slice(0, lastSpace) : sub;

  return cut.replace(TRAILING_PARTICLES, "").replace(TRAILING_PUNCTUATION, "").trim() + "...";
}

// ---------------------------------------------------------------------------
// Relative date label
// ---------------------------------------------------------------------------

function getRelativeDate(dateStr?: string): string {
  if (!dateStr) return "";
  const d = new Date(dateStr);
  if (isNaN(d.getTime())) return "";

  const now = new Date();
  const todayStart = new Date(now.getFullYear(), now.getMonth(), now.getDate());
  const yesterdayStart = new Date(todayStart);
  yesterdayStart.setDate(yesterdayStart.getDate() - 1);
  const weekStart = new Date(todayStart);
  weekStart.setDate(weekStart.getDate() - 6);

  if (d >= todayStart) return "오늘";
  if (d >= yesterdayStart) return "어제";
  if (d >= weekStart) {
    const days = ["일", "월", "화", "수", "목", "금", "토"];
    return days[d.getDay()] + "요일";
  }
  const m = d.getMonth() + 1;
  const day = d.getDate();
  return `${m}/${day}`;
}

// ---------------------------------------------------------------------------
// Date filter utility
// ---------------------------------------------------------------------------

function isWithinDateFilter(dateStr: string | undefined, filter: DateFilter): boolean {
  if (filter === "all") return true;
  if (!dateStr) return false;
  const d = new Date(dateStr);
  if (isNaN(d.getTime())) return false;

  const now = new Date();
  const todayStart = new Date(now.getFullYear(), now.getMonth(), now.getDate());

  if (filter === "today") {
    return d >= todayStart;
  }
  if (filter === "week") {
    const weekStart = new Date(todayStart);
    weekStart.setDate(weekStart.getDate() - 6);
    return d >= weekStart;
  }
  if (filter === "month") {
    const monthStart = new Date(now.getFullYear(), now.getMonth(), 1);
    return d >= monthStart;
  }
  return true;
}

// ---------------------------------------------------------------------------
// Bookmark type
// ---------------------------------------------------------------------------

interface Bookmark {
  message_id: string;
  session_id: string;
  content: string;
  role: string;
  note: string;
  user_id: string;
  created_at: string;
}

// ---------------------------------------------------------------------------
// TagDots — small colored circles shown on session items
// ---------------------------------------------------------------------------

function TagDots({ tags }: { tags: SessionTag[] }) {
  if (tags.length === 0) return null;

  const visible = tags.slice(0, 4);
  const more = tags.length - 4;

  return (
    <Box
      sx={{
        display: "flex",
        alignItems: "center",
        gap: "2px",
        flexShrink: 0,
        ml: 0.25,
      }}
    >
      {visible.map((tag) => (
        <Tooltip key={tag.id} title={tag.name} placement="top" arrow enterDelay={400}>
          <Box
            sx={{
              width: 7,
              height: 7,
              borderRadius: "50%",
              bgcolor: tag.color,
              flexShrink: 0,
              boxShadow: `0 0 0 1px rgba(0,0,0,0.3)`,
            }}
          />
        </Tooltip>
      ))}
      {more > 0 && (
        <Typography sx={{ fontSize: "0.6rem", color: "#8e8ea0", ml: "1px" }}>
          +{more}
        </Typography>
      )}
    </Box>
  );
}

// ---------------------------------------------------------------------------
// TagContextMenu — popover shown on right-click for assigning tags
// ---------------------------------------------------------------------------

interface TagContextMenuProps {
  anchorEl: HTMLElement | null;
  sessionId: string;
  allTags: SessionTag[];
  hasTag: (sessionId: string, tagId: string) => boolean;
  onToggle: (sessionId: string, tagId: string) => void;
  onClose: () => void;
}

function TagContextMenu({
  anchorEl,
  sessionId,
  allTags,
  hasTag,
  onToggle,
  onClose,
}: TagContextMenuProps) {
  return (
    <Popover
      open={Boolean(anchorEl)}
      anchorEl={anchorEl}
      onClose={onClose}
      anchorOrigin={{ vertical: "center", horizontal: "right" }}
      transformOrigin={{ vertical: "center", horizontal: "left" }}
      slotProps={{
        paper: {
          sx: {
            bgcolor: "#1e1e1e",
            border: "1px solid rgba(255,255,255,0.1)",
            borderRadius: 2,
            boxShadow: "0 8px 32px rgba(0,0,0,0.4)",
            minWidth: 160,
            p: 0.5,
          },
        },
      }}
    >
      <Typography
        variant="caption"
        sx={{
          display: "block",
          px: 1.5,
          pt: 0.75,
          pb: 0.5,
          color: "#8e8ea0",
          fontWeight: 600,
          fontSize: "0.65rem",
          textTransform: "uppercase",
          letterSpacing: "0.05em",
        }}
      >
        태그 설정
      </Typography>
      {allTags.map((tag) => {
        const checked = hasTag(sessionId, tag.id);
        return (
          <Box
            key={tag.id}
            onClick={() => onToggle(sessionId, tag.id)}
            sx={{
              display: "flex",
              alignItems: "center",
              gap: 1,
              px: 1.5,
              py: 0.6,
              borderRadius: 1,
              cursor: "pointer",
              userSelect: "none",
              "&:hover": { bgcolor: "rgba(255,255,255,0.08)" },
            }}
          >
            {/* Color dot */}
            <Box
              sx={{
                width: 9,
                height: 9,
                borderRadius: "50%",
                bgcolor: tag.color,
                flexShrink: 0,
                boxShadow: checked
                  ? `0 0 0 2px rgba(255,255,255,0.35), 0 0 0 1px ${tag.color}`
                  : "0 0 0 1px rgba(0,0,0,0.3)",
              }}
            />
            <Typography
              variant="body2"
              sx={{
                flex: 1,
                fontSize: "0.82rem",
                color: "#ececec",
                fontWeight: checked ? 600 : 400,
              }}
            >
              {tag.name}
            </Typography>
            {checked && <CheckIcon sx={{ fontSize: 13, color: "#10b981" }} />}
          </Box>
        );
      })}
    </Popover>
  );
}

// ---------------------------------------------------------------------------
// TagManageDialog — create/delete custom tags
// ---------------------------------------------------------------------------

interface TagManageDialogProps {
  open: boolean;
  onClose: () => void;
  customTags: SessionTag[];
  onAddTag: (name: string, color: string) => void;
  onDeleteTag: (tagId: string) => void;
}

function TagManageDialog({
  open,
  onClose,
  customTags,
  onAddTag,
  onDeleteTag,
}: TagManageDialogProps) {
  const [newName, setNewName] = useState("");
  const [newColor, setNewColor] = useState(COLOR_SWATCHES[0]);
  const nameInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (open) {
      setNewName("");
      setNewColor(COLOR_SWATCHES[0]);
      setTimeout(() => nameInputRef.current?.focus(), 80);
    }
  }, [open]);

  const handleAdd = () => {
    const trimmed = newName.trim();
    if (!trimmed) return;
    onAddTag(trimmed, newColor);
    setNewName("");
    setNewColor(COLOR_SWATCHES[0]);
    nameInputRef.current?.focus();
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      e.preventDefault();
      handleAdd();
    }
  };

  const allTagsDisplay: SessionTag[] = [...PRESET_TAGS, ...customTags];

  return (
    <Dialog
      open={open}
      onClose={onClose}
      maxWidth="xs"
      fullWidth
      slotProps={{
        paper: {
          sx: {
            bgcolor: "#1a1a1a",
            border: "1px solid rgba(255,255,255,0.1)",
            borderRadius: 2.5,
            color: "#ececec",
          },
        },
      }}
    >
      <DialogTitle
        sx={{
          fontSize: "0.95rem",
          fontWeight: 700,
          pb: 1,
          display: "flex",
          alignItems: "center",
          gap: 1,
        }}
      >
        <LabelIcon sx={{ fontSize: 18, color: "#10a37f" }} />
        태그 관리
        <IconButton
          size="small"
          onClick={onClose}
          sx={{ ml: "auto", color: "#8e8ea0" }}
        >
          <CloseIcon sx={{ fontSize: 16 }} />
        </IconButton>
      </DialogTitle>

      <DialogContent sx={{ pt: 0 }}>
        {/* Existing tags */}
        <Typography
          variant="caption"
          sx={{
            display: "block",
            mb: 1,
            color: "#8e8ea0",
            fontWeight: 600,
            fontSize: "0.65rem",
            textTransform: "uppercase",
            letterSpacing: "0.05em",
          }}
        >
          현재 태그
        </Typography>
        <Box sx={{ display: "flex", flexWrap: "wrap", gap: 0.75, mb: 2.5 }}>
          {allTagsDisplay.map((tag) => (
            <Chip
              key={tag.id}
              label={tag.name}
              size="small"
              onDelete={tag.isPreset ? undefined : () => onDeleteTag(tag.id)}
              sx={{
                bgcolor: `${tag.color}22`,
                color: tag.color,
                border: `1px solid ${tag.color}55`,
                fontSize: "0.75rem",
                fontWeight: 600,
                height: 24,
                "& .MuiChip-deleteIcon": {
                  color: `${tag.color}99`,
                  fontSize: 14,
                  "&:hover": { color: tag.color },
                },
                "&::before": {
                  content: '""',
                  display: "inline-block",
                  width: 6,
                  height: 6,
                  borderRadius: "50%",
                  bgcolor: tag.color,
                  mr: 0.5,
                  flexShrink: 0,
                },
              }}
            />
          ))}
        </Box>

        {/* Create new tag */}
        <Typography
          variant="caption"
          sx={{
            display: "block",
            mb: 1,
            color: "#8e8ea0",
            fontWeight: 600,
            fontSize: "0.65rem",
            textTransform: "uppercase",
            letterSpacing: "0.05em",
          }}
        >
          새 태그 추가
        </Typography>

        <Stack spacing={1.5}>
          <TextField
            inputRef={nameInputRef}
            size="small"
            fullWidth
            placeholder="태그 이름..."
            value={newName}
            onChange={(e) => setNewName(e.target.value)}
            onKeyDown={handleKeyDown}
            slotProps={{
              input: {
                sx: {
                  bgcolor: "rgba(255,255,255,0.05)",
                  color: "#ececec",
                  borderRadius: 1.5,
                  fontSize: "0.875rem",
                  "& fieldset": {
                    borderColor: "rgba(255,255,255,0.15)",
                  },
                  "&:hover fieldset": {
                    borderColor: "rgba(255,255,255,0.25)",
                  },
                  "&.Mui-focused fieldset": {
                    borderColor: "#10a37f",
                  },
                  "&::placeholder": {
                    color: "#8e8ea0",
                    opacity: 1,
                  },
                },
              },
            }}
            sx={{
              "& input::placeholder": { color: "#8e8ea0" },
            }}
          />

          {/* Color swatches */}
          <Box>
            <Typography
              variant="caption"
              sx={{ color: "#8e8ea0", fontSize: "0.7rem", display: "block", mb: 0.75 }}
            >
              색상 선택
            </Typography>
            <Box sx={{ display: "flex", flexWrap: "wrap", gap: 0.75 }}>
              {COLOR_SWATCHES.map((color) => (
                <Box
                  key={color}
                  onClick={() => setNewColor(color)}
                  sx={{
                    width: 22,
                    height: 22,
                    borderRadius: "50%",
                    bgcolor: color,
                    cursor: "pointer",
                    boxShadow:
                      newColor === color
                        ? `0 0 0 2px #1a1a1a, 0 0 0 4px ${color}`
                        : "0 0 0 1px rgba(0,0,0,0.3)",
                    transition: "box-shadow 0.15s",
                    "&:hover": {
                      boxShadow: `0 0 0 2px #1a1a1a, 0 0 0 3px ${color}`,
                      transform: "scale(1.1)",
                    },
                  }}
                />
              ))}
            </Box>
          </Box>

          {/* Preview */}
          {newName.trim() && (
            <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
              <Typography variant="caption" sx={{ color: "#8e8ea0", fontSize: "0.7rem" }}>
                미리보기:
              </Typography>
              <Chip
                label={newName.trim()}
                size="small"
                sx={{
                  bgcolor: `${newColor}22`,
                  color: newColor,
                  border: `1px solid ${newColor}55`,
                  fontSize: "0.75rem",
                  fontWeight: 600,
                  height: 22,
                }}
              />
            </Box>
          )}
        </Stack>
      </DialogContent>

      <DialogActions sx={{ px: 3, pb: 2, pt: 0.5 }}>
        <Button
          onClick={onClose}
          size="small"
          sx={{
            color: "#8e8ea0",
            textTransform: "none",
            "&:hover": { color: "#ececec" },
          }}
        >
          닫기
        </Button>
        <Button
          onClick={handleAdd}
          disabled={!newName.trim()}
          size="small"
          variant="contained"
          sx={{
            bgcolor: "#10a37f",
            color: "#fff",
            textTransform: "none",
            borderRadius: 1.5,
            "&:hover": { bgcolor: "#0e8f6f" },
            "&:disabled": { bgcolor: "rgba(255,255,255,0.1)", color: "#8e8ea0" },
          }}
        >
          추가
        </Button>
      </DialogActions>
    </Dialog>
  );
}

// ---------------------------------------------------------------------------
// TagFilterBar — horizontal row of tag filter chips
// ---------------------------------------------------------------------------

interface TagFilterBarProps {
  allTags: SessionTag[];
  activeTagId: string | null;
  onFilterChange: (tagId: string | null) => void;
  onManage: () => void;
}

function TagFilterBar({
  allTags,
  activeTagId,
  onFilterChange,
  onManage,
}: TagFilterBarProps) {
  return (
    <Box
      sx={{
        px: 1,
        pb: 1,
        display: "flex",
        flexDirection: "column",
        gap: 0.5,
      }}
    >
      <Box
        sx={{
          display: "flex",
          alignItems: "center",
          gap: 0.5,
          px: 0.5,
        }}
      >
        <LabelIcon sx={{ fontSize: 12, color: "#8e8ea0", flexShrink: 0 }} />
        <Typography
          variant="caption"
          sx={{
            flex: 1,
            fontWeight: 600,
            fontSize: "0.62rem",
            textTransform: "uppercase",
            letterSpacing: "0.05em",
            color: "#8e8ea0",
          }}
        >
          태그 필터
        </Typography>
        <Tooltip title="태그 관리" placement="right">
          <IconButton
            size="small"
            onClick={onManage}
            sx={{
              p: 0.3,
              color: "#8e8ea0",
              "&:hover": { color: "#ececec", bgcolor: "rgba(255,255,255,0.08)" },
            }}
          >
            <SettingsIcon sx={{ fontSize: 12 }} />
          </IconButton>
        </Tooltip>
      </Box>

      {/* Scrollable chip row */}
      <Box
        sx={{
          display: "flex",
          flexWrap: "wrap",
          gap: 0.4,
          maxHeight: 72,
          overflowY: "auto",
          "&::-webkit-scrollbar": { width: 3 },
          "&::-webkit-scrollbar-thumb": {
            bgcolor: "rgba(255,255,255,0.1)",
            borderRadius: 2,
          },
          scrollbarWidth: "thin",
          scrollbarColor: "rgba(255,255,255,0.1) transparent",
        }}
      >
        {/* "전체" chip */}
        <TagFilterChip
          label="전체"
          color="#8e8ea0"
          active={activeTagId === null}
          onClick={() => onFilterChange(null)}
        />
        {allTags.map((tag) => (
          <TagFilterChip
            key={tag.id}
            label={tag.name}
            color={tag.color}
            active={activeTagId === tag.id}
            onClick={() => onFilterChange(activeTagId === tag.id ? null : tag.id)}
          />
        ))}
      </Box>
    </Box>
  );
}

interface TagFilterChipProps {
  label: string;
  color: string;
  active: boolean;
  onClick: () => void;
}

function TagFilterChip({ label, color, active, onClick }: TagFilterChipProps) {
  return (
    <Box
      onClick={onClick}
      sx={{
        display: "inline-flex",
        alignItems: "center",
        gap: "3px",
        px: 0.75,
        py: "2px",
        borderRadius: 10,
        cursor: "pointer",
        userSelect: "none",
        fontSize: "0.68rem",
        fontWeight: active ? 700 : 500,
        color: active ? "#fff" : "#adadad",
        bgcolor: active ? `${color}cc` : "rgba(255,255,255,0.06)",
        border: `1px solid ${active ? color : "rgba(255,255,255,0.08)"}`,
        transition: "all 0.15s ease",
        "&:hover": {
          bgcolor: active ? `${color}dd` : "rgba(255,255,255,0.1)",
          color: "#fff",
          borderColor: active ? color : "rgba(255,255,255,0.15)",
        },
      }}
    >
      {label !== "전체" && (
        <Box
          sx={{
            width: 5,
            height: 5,
            borderRadius: "50%",
            bgcolor: active ? "#fff" : color,
            flexShrink: 0,
          }}
        />
      )}
      {label}
    </Box>
  );
}

// ---------------------------------------------------------------------------
// SessionListItem — with tag dots and context menu trigger
// ---------------------------------------------------------------------------

function SessionListItem({
  s,
  isActive,
  onSelect,
  onDelete,
  onRename,
  sessionTags,
  allTags,
  hasTag,
  onToggleTag,
  isFavorite,
  onToggleFavorite,
}: {
  s: SessionItem;
  isActive: boolean;
  onSelect: () => void;
  onDelete: (e: React.MouseEvent) => void;
  onRename: (id: string, title: string) => void;
  sessionTags: SessionTag[];
  allTags: SessionTag[];
  hasTag: (sessionId: string, tagId: string) => boolean;
  onToggleTag: (sessionId: string, tagId: string) => void;
  isFavorite: boolean;
  onToggleFavorite: (id: string, e: React.MouseEvent) => void;
}) {
  const [isEditing, setIsEditing] = useState(false);
  const [editValue, setEditValue] = useState(s.title || "새 대화");
  const inputRef = useRef<HTMLInputElement>(null);

  // Context menu anchor
  const [ctxAnchor, setCtxAnchor] = useState<HTMLElement | null>(null);
  // Long-press timer
  const longPressTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    if (isEditing && inputRef.current) {
      inputRef.current.focus();
      inputRef.current.select();
    }
  }, [isEditing]);

  const startEdit = (e: React.MouseEvent) => {
    e.stopPropagation();
    setEditValue(s.title || "새 대화");
    setIsEditing(true);
  };

  const commitEdit = () => {
    const trimmed = editValue.trim();
    if (trimmed && trimmed !== (s.title || "새 대화")) {
      onRename(s.id, trimmed);
    }
    setIsEditing(false);
  };

  const cancelEdit = () => {
    setEditValue(s.title || "새 대화");
    setIsEditing(false);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      e.preventDefault();
      commitEdit();
    } else if (e.key === "Escape") {
      e.preventDefault();
      cancelEdit();
    }
  };

  const handleContextMenu = (e: React.MouseEvent<HTMLElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setCtxAnchor(e.currentTarget);
  };

  const handleTouchStart = (e: React.TouchEvent<HTMLElement>) => {
    const target = e.currentTarget;
    longPressTimer.current = setTimeout(() => {
      setCtxAnchor(target);
    }, 500);
  };

  const handleTouchEnd = () => {
    if (longPressTimer.current) {
      clearTimeout(longPressTimer.current);
      longPressTimer.current = null;
    }
  };

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (longPressTimer.current) clearTimeout(longPressTimer.current);
    };
  }, []);

  if (isEditing) {
    return (
      <Box
        sx={{
          borderRadius: 1.5,
          mb: 0.25,
          px: 1,
          py: 0.5,
          bgcolor: "rgba(255,255,255,0.15)",
          display: "flex",
          alignItems: "center",
          gap: 0.5,
        }}
      >
        <TextField
          inputRef={inputRef}
          size="small"
          value={editValue}
          onChange={(e) => setEditValue(e.target.value)}
          onBlur={commitEdit}
          onKeyDown={handleKeyDown}
          fullWidth
          variant="standard"
          slotProps={{
            input: {
              disableUnderline: false,
              sx: {
                color: "#ececec",
                fontSize: "0.875rem",
                "&::before": { borderBottomColor: "rgba(255,255,255,0.3)" },
                "&::after": { borderBottomColor: "#10a37f" },
              },
            },
          }}
          sx={{ flex: 1 }}
        />
        <IconButton
          size="small"
          onMouseDown={(e) => {
            e.preventDefault();
            cancelEdit();
          }}
          sx={{ color: "#8e8ea0", p: 0.25 }}
        >
          <CloseIcon sx={{ fontSize: 14 }} />
        </IconButton>
      </Box>
    );
  }

  const rawTitle = s.title || "새 대화";
  const displayTitle = smartTruncateTitle(rawTitle);
  const isTruncated = displayTitle !== rawTitle;
  const relativeDate = getRelativeDate(s.updated_at ?? s.updatedAt ?? s.created_at ?? s.createdAt);
  const isForked = rawTitle.includes("(분기)");

  return (
    <>
      <Tooltip
        title={isTruncated ? rawTitle : ""}
        placement="right"
        arrow
        disableHoverListener={!isTruncated}
        enterDelay={600}
      >
        <ListItemButton
          selected={isActive}
          onClick={onSelect}
          onDoubleClick={startEdit}
          onContextMenu={handleContextMenu}
          onTouchStart={handleTouchStart}
          onTouchEnd={handleTouchEnd}
          onTouchCancel={handleTouchEnd}
          sx={{
            borderRadius: 1.5,
            mb: 0.25,
            py: 0.6,
            color: "#ececec",
            "&:hover": {
              bgcolor: "rgba(255,255,255,0.1)",
            },
            "&:hover .session-actions": { opacity: 1 },
            "&.Mui-selected": {
              bgcolor: "rgba(255,255,255,0.15)",
              color: "#ececec",
              "&:hover": {
                bgcolor: "rgba(255,255,255,0.2)",
              },
              "& .MuiListItemText-secondary": {
                color: "#8e8ea0",
              },
            },
          }}
        >
          {/* Title + tag dots stacked */}
          <Box sx={{ flex: 1, minWidth: 0 }}>
            <Box sx={{ display: "flex", alignItems: "center", gap: 0.5, minWidth: 0 }}>
              {isForked && (
                <Tooltip title="분기된 대화">
                  <ForkIcon sx={{ fontSize: 11, color: "rgba(100,180,255,0.7)", flexShrink: 0 }} />
                </Tooltip>
              )}
              <Typography
                noWrap
                variant="body2"
                sx={{
                  color: "#ececec",
                  fontSize: "0.875rem",
                  fontWeight: isActive ? 600 : 400,
                  minWidth: 0,
                  flex: 1,
                }}
              >
                {displayTitle}
              </Typography>
              <TagDots tags={sessionTags} />
            </Box>
            <Typography
              noWrap
              variant="caption"
              sx={{ color: "text.disabled", fontSize: "0.7rem", display: "block" }}
            >
              {relativeDate || `${s.message_count ?? s.messageCount ?? 0}개 메시지`}
            </Typography>
          </Box>

          {/* Action buttons — shown on hover (star always visible when favorited) */}
          <Box
            className="session-actions"
            sx={{
              display: "flex",
              alignItems: "center",
              flexShrink: 0,
              opacity: isFavorite ? 1 : 0,
              transition: "opacity 0.2s",
            }}
          >
            <Tooltip title={isFavorite ? "즐겨찾기 해제" : "즐겨찾기 추가"}>
              <IconButton
                size="small"
                onClick={(e) => onToggleFavorite(s.id, e)}
                aria-label={isFavorite ? "즐겨찾기 해제" : "즐겨찾기 추가"}
                sx={{
                  color: isFavorite ? "#f59e0b" : "#8e8ea0",
                  "&:hover": { color: "#f59e0b" },
                  p: 0.25,
                }}
              >
                {isFavorite ? (
                  <StarIcon sx={{ fontSize: 13 }} />
                ) : (
                  <StarBorderIcon sx={{ fontSize: 13 }} />
                )}
              </IconButton>
            </Tooltip>
            <Tooltip title="태그 설정">
              <IconButton
                size="small"
                onClick={(e) => {
                  e.stopPropagation();
                  setCtxAnchor(e.currentTarget);
                }}
                aria-label="태그 설정"
                sx={{
                  color: "#8e8ea0",
                  "&:hover": { color: "#ececec" },
                  p: 0.25,
                }}
              >
                <LabelIcon sx={{ fontSize: 13 }} />
              </IconButton>
            </Tooltip>
            <Tooltip title="이름 변경 (더블클릭)">
              <IconButton
                size="small"
                onClick={startEdit}
                aria-label="대화 이름 변경"
                sx={{
                  color: "#8e8ea0",
                  "&:hover": { color: "#ececec" },
                  p: 0.25,
                }}
              >
                <EditIcon sx={{ fontSize: 13 }} />
              </IconButton>
            </Tooltip>
            <Tooltip title="삭제">
              <IconButton
                size="small"
                onClick={onDelete}
                aria-label="대화 삭제"
                sx={{
                  color: "#8e8ea0",
                  "&:hover": { color: "#ececec" },
                  p: 0.25,
                }}
              >
                <DeleteIcon sx={{ fontSize: 14 }} />
              </IconButton>
            </Tooltip>
          </Box>
        </ListItemButton>
      </Tooltip>

      {/* Tag context menu */}
      <TagContextMenu
        anchorEl={ctxAnchor}
        sessionId={s.id}
        allTags={allTags}
        hasTag={hasTag}
        onToggle={onToggleTag}
        onClose={() => setCtxAnchor(null)}
      />
    </>
  );
}

// ---------------------------------------------------------------------------
// ChatSidebarProps
// ---------------------------------------------------------------------------

interface ChatSidebarProps {
  sessionGroups: { label: string; items: SessionItem[] }[];
  sessionsEmpty: boolean;
  isSessionsLoading?: boolean;
  currentSessionId: string | null;
  onSelectSession: (id: string) => void;
  onNewChat: () => void;
  onDeleteSession: (id: string) => void;
  onRenameSession: (id: string, title: string) => void;
}

const SIDEBAR_WIDTH = 260;

// ---------------------------------------------------------------------------
// ChatSidebar
// ---------------------------------------------------------------------------

export function ChatSidebar({
  sessionGroups,
  sessionsEmpty,
  isSessionsLoading = false,
  currentSessionId,
  onSelectSession,
  onNewChat,
  onDeleteSession,
  onRenameSession,
}: ChatSidebarProps) {
  const navigate = useNavigate();
  const location = useLocation();
  const { t } = useLanguage();
  const { logout } = useAuth();
  const [searchQuery, setSearchQuery] = useState("");
  const [bookmarksOpen, setBookmarksOpen] = useState(true);
  const [helpOpen, setHelpOpen] = useState(false);
  const [manageOpen, setManageOpen] = useState(false);
  const [activeTagFilter, setActiveTagFilter] = useState<string | null>(null);
  const queryClient = useQueryClient();

  // Sort + date filter + favorites — persisted to localStorage
  const [sortOption, setSortOption] = useState<SortOption>(() =>
    lsGet<SortOption>(LS_SORT_KEY, "newest"),
  );
  const [dateFilter, setDateFilter] = useState<DateFilter>(() =>
    lsGet<DateFilter>(LS_DATE_FILTER_KEY, "all"),
  );
  const [favorites, setFavorites] = useState<string[]>(() =>
    lsGet<string[]>(LS_FAVORITES_KEY, []),
  );
  const [showFavoritesOnly, setShowFavoritesOnly] = useState(false);

  // Sort menu anchor
  const [sortAnchorEl, setSortAnchorEl] = useState<HTMLElement | null>(null);

  const { allTags, customTags, addTag, deleteTag, getSessionTags, hasTag, toggleTag } =
    useSessionTags();

  const { data: bookmarksData } = useQuery({
    queryKey: ["bookmarks"],
    queryFn: () => bookmarksApi.list().then((r) => r.data),
    staleTime: 30_000,
  });

  const bookmarks: Bookmark[] = bookmarksData?.bookmarks ?? [];

  const removeMutation = useMutation({
    mutationFn: (messageId: string) => bookmarksApi.remove(messageId),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["bookmarks"] }),
  });

  // Persist sort preference
  const handleSortChange = useCallback((opt: SortOption) => {
    setSortOption(opt);
    lsSet(LS_SORT_KEY, opt);
    setSortAnchorEl(null);
  }, []);

  // Persist date filter
  const handleDateFilter = useCallback((df: DateFilter) => {
    setDateFilter(df);
    lsSet(LS_DATE_FILTER_KEY, df);
  }, []);

  // Toggle favorite
  const handleToggleFavorite = useCallback(
    (id: string, e: React.MouseEvent) => {
      e.stopPropagation();
      setFavorites((prev) => {
        const next = prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id];
        lsSet(LS_FAVORITES_KEY, next);
        return next;
      });
    },
    [],
  );

  // Flat list of all sessions for sort+filter (groups come from parent, already time-grouped)
  const allSessions = useMemo(
    () => sessionGroups.flatMap((g) => g.items),
    [sessionGroups],
  );

  // Apply search + tag + date + favorites + sort, then re-group by date label
  const filteredGroups = useMemo(() => {
    // 1. Filter
    let items = allSessions.filter((s) => {
      const titleMatch = (s.title || "새 대화")
        .toLowerCase()
        .includes(searchQuery.toLowerCase());
      const tagMatch = activeTagFilter === null || hasTag(s.id, activeTagFilter);
      const dateMatch = isWithinDateFilter(
        s.updated_at ?? s.updatedAt ?? s.created_at ?? s.createdAt,
        dateFilter,
      );
      const favMatch = !showFavoritesOnly || favorites.includes(s.id);
      return titleMatch && tagMatch && dateMatch && favMatch;
    });

    // 2. Sort
    items = [...items].sort((a, b) => {
      if (sortOption === "newest") {
        const da = new Date(a.updated_at ?? a.updatedAt ?? a.created_at ?? a.createdAt ?? 0).getTime();
        const db = new Date(b.updated_at ?? b.updatedAt ?? b.created_at ?? b.createdAt ?? 0).getTime();
        return db - da;
      }
      if (sortOption === "oldest") {
        const da = new Date(a.updated_at ?? a.updatedAt ?? a.created_at ?? a.createdAt ?? 0).getTime();
        const db = new Date(b.updated_at ?? b.updatedAt ?? b.created_at ?? b.createdAt ?? 0).getTime();
        return da - db;
      }
      if (sortOption === "most_messages") {
        return (b.message_count ?? b.messageCount ?? 0) - (a.message_count ?? a.messageCount ?? 0);
      }
      if (sortOption === "alpha") {
        return (a.title || "새 대화").localeCompare(b.title || "새 대화", "ko");
      }
      return 0;
    });

    // 3. Re-group (only when sort is "newest" keep original time-based groups)
    if (sortOption === "newest" || sortOption === "oldest") {
      // Re-bucket by relative date
      const buckets: Record<string, SessionItem[]> = {};
      const bucketOrder: string[] = [];
      items.forEach((s) => {
        const label = getRelativeDate(s.updated_at ?? s.updatedAt ?? s.created_at ?? s.createdAt) || "이전";
        if (!buckets[label]) {
          buckets[label] = [];
          bucketOrder.push(label);
        }
        buckets[label].push(s);
      });
      return bucketOrder
        .map((label) => ({ label, items: buckets[label] }))
        .filter((g) => g.items.length > 0);
    }

    // For other sorts, show as flat single group
    if (items.length === 0) return [];
    return [{ label: SORT_LABELS[sortOption], items }];
  }, [
    allSessions,
    searchQuery,
    activeTagFilter,
    dateFilter,
    showFavoritesOnly,
    favorites,
    sortOption,
    hasTag,
  ]);

  const handleTagFilter = useCallback((tagId: string | null) => {
    setActiveTagFilter(tagId);
  }, []);

  return (
    <Box
      component="nav"
      role="navigation"
      aria-label="사이드바 탐색"
      sx={{
        width: SIDEBAR_WIDTH,
        height: "100%",
        display: "flex",
        flexDirection: "column",
        bgcolor: "#171717",
        color: "#ececec",
      }}
    >
      {/* Logo */}
      <Box sx={{ p: 2, pb: 1, display: "flex", alignItems: "center", gap: 1 }}>
        <AutoAwesomeIcon sx={{ color: "#10a37f", fontSize: 22 }} />
        <Typography variant="h6" fontWeight={700} sx={{ color: "#ececec" }}>
          AI 어시스턴트
        </Typography>
      </Box>

      {/* New chat button */}
      <Box sx={{ px: 2, pb: 1.5 }}>
        <Button
          variant="outlined"
          fullWidth
          startIcon={<AddIcon />}
          onClick={onNewChat}
          aria-label="새 대화 시작"
          sx={{
            borderRadius: 2,
            textTransform: "none",
            py: 1,
            color: "#ececec",
            borderColor: "rgba(255,255,255,0.2)",
            "&:hover": {
              borderColor: "rgba(255,255,255,0.4)",
              bgcolor: "rgba(255,255,255,0.05)",
            },
          }}
        >
          {t("sidebar.newChat")}
        </Button>
      </Box>

      {/* Search + Sort */}
      <Box sx={{ px: 2, pb: 0.75, display: "flex", gap: 0.5, alignItems: "center" }}>
        <TextField
          size="small"
          fullWidth
          placeholder={t("sidebar.searchPlaceholder")}
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          inputProps={{ "aria-label": "대화 검색" }}
          slotProps={{
            input: {
              startAdornment: (
                <InputAdornment position="start">
                  <SearchIcon fontSize="small" sx={{ color: "#8e8ea0" }} />
                </InputAdornment>
              ),
              endAdornment: searchQuery ? (
                <InputAdornment position="end">
                  <IconButton
                    size="small"
                    onClick={() => setSearchQuery("")}
                    edge="end"
                    aria-label="검색어 지우기"
                    sx={{ color: "#8e8ea0", p: 0.25 }}
                  >
                    <CloseIcon sx={{ fontSize: 14 }} />
                  </IconButton>
                </InputAdornment>
              ) : undefined,
            },
          }}
          sx={{
            "& .MuiOutlinedInput-root": {
              borderRadius: 2,
              bgcolor: "rgba(255,255,255,0.05)",
              color: "#ececec",
              "& fieldset": { border: "none" },
              "&:hover fieldset": { border: "none" },
              "&.Mui-focused fieldset": { border: "none" },
            },
            "& input::placeholder": {
              color: "#8e8ea0",
              opacity: 1,
            },
          }}
        />
        <Tooltip title={`정렬: ${SORT_LABELS[sortOption]}`} placement="right">
          <IconButton
            size="small"
            onClick={(e) => setSortAnchorEl(e.currentTarget)}
            aria-label="정렬 옵션"
            sx={{
              color: sortOption !== "newest" ? "#10a37f" : "#8e8ea0",
              bgcolor: sortOption !== "newest" ? "rgba(16,163,127,0.12)" : "rgba(255,255,255,0.05)",
              borderRadius: 1.5,
              p: 0.75,
              flexShrink: 0,
              "&:hover": {
                color: "#ececec",
                bgcolor: "rgba(255,255,255,0.1)",
              },
              transition: "all 0.15s ease",
            }}
          >
            <SortIcon sx={{ fontSize: 18 }} />
          </IconButton>
        </Tooltip>
      </Box>

      {/* Sort menu */}
      <Menu
        anchorEl={sortAnchorEl}
        open={Boolean(sortAnchorEl)}
        onClose={() => setSortAnchorEl(null)}
        anchorOrigin={{ vertical: "bottom", horizontal: "right" }}
        transformOrigin={{ vertical: "top", horizontal: "right" }}
        slotProps={{
          paper: {
            sx: {
              bgcolor: "#1e1e1e",
              border: "1px solid rgba(255,255,255,0.1)",
              borderRadius: 2,
              boxShadow: "0 8px 32px rgba(0,0,0,0.4)",
              minWidth: 180,
              mt: 0.5,
            },
          },
        }}
      >
        {(Object.entries(SORT_LABELS) as [SortOption, string][]).map(([key, label]) => (
          <MenuItem
            key={key}
            onClick={() => handleSortChange(key)}
            sx={{
              py: 0.75,
              px: 1.5,
              fontSize: "0.82rem",
              color: sortOption === key ? "#10a37f" : "#ececec",
              fontWeight: sortOption === key ? 700 : 400,
              display: "flex",
              alignItems: "center",
              gap: 1,
              "&:hover": { bgcolor: "rgba(255,255,255,0.08)" },
            }}
          >
            <Box sx={{ width: 16, display: "flex", alignItems: "center" }}>
              {sortOption === key && <CheckIcon sx={{ fontSize: 14, color: "#10a37f" }} />}
            </Box>
            {label}
          </MenuItem>
        ))}
      </Menu>

      {/* Date filter + Favorites chips */}
      <Box
        sx={{
          px: 1.5,
          pb: 0.75,
          display: "flex",
          flexWrap: "wrap",
          gap: 0.4,
          alignItems: "center",
        }}
      >
        {(Object.entries(DATE_FILTER_LABELS) as [DateFilter, string][]).map(([key, label]) => {
          const isActive = dateFilter === key;
          return (
            <Box
              key={key}
              onClick={() => handleDateFilter(key)}
              sx={{
                display: "inline-flex",
                alignItems: "center",
                px: 0.9,
                py: "2px",
                borderRadius: 10,
                cursor: "pointer",
                userSelect: "none",
                fontSize: "0.68rem",
                fontWeight: isActive ? 700 : 500,
                color: isActive ? "#fff" : "#8e8ea0",
                bgcolor: isActive ? "#10a37f" : "rgba(255,255,255,0.06)",
                border: `1px solid ${isActive ? "#10a37f" : "rgba(255,255,255,0.08)"}`,
                transition: "all 0.15s ease",
                "&:hover": {
                  bgcolor: isActive ? "#0e8f6f" : "rgba(255,255,255,0.1)",
                  color: "#fff",
                },
              }}
            >
              {label}
            </Box>
          );
        })}
        {/* Favorites chip */}
        <Box
          onClick={() => setShowFavoritesOnly((prev) => !prev)}
          sx={{
            display: "inline-flex",
            alignItems: "center",
            gap: "3px",
            px: 0.9,
            py: "2px",
            borderRadius: 10,
            cursor: "pointer",
            userSelect: "none",
            fontSize: "0.68rem",
            fontWeight: showFavoritesOnly ? 700 : 500,
            color: showFavoritesOnly ? "#f59e0b" : "#8e8ea0",
            bgcolor: showFavoritesOnly ? "rgba(245,158,11,0.15)" : "rgba(255,255,255,0.06)",
            border: `1px solid ${showFavoritesOnly ? "rgba(245,158,11,0.5)" : "rgba(255,255,255,0.08)"}`,
            transition: "all 0.15s ease",
            "&:hover": {
              bgcolor: showFavoritesOnly ? "rgba(245,158,11,0.2)" : "rgba(255,255,255,0.1)",
              color: "#f59e0b",
            },
          }}
        >
          {showFavoritesOnly ? (
            <StarIcon sx={{ fontSize: 10 }} />
          ) : (
            <StarBorderIcon sx={{ fontSize: 10 }} />
          )}
          즐겨찾기
        </Box>
      </Box>

      {/* Tag filter bar */}
      <Divider sx={{ borderColor: "rgba(255,255,255,0.08)", mb: 0.5 }} />
      <TagFilterBar
        allTags={allTags}
        activeTagId={activeTagFilter}
        onFilterChange={handleTagFilter}
        onManage={() => setManageOpen(true)}
      />
      <Divider sx={{ borderColor: "rgba(255,255,255,0.08)" }} />

      {/* Session list */}
      <Box
        sx={{
          flex: 1,
          overflow: "auto",
          px: 1,
          py: 1,
          "&::-webkit-scrollbar": { width: 6 },
          "&::-webkit-scrollbar-track": { bgcolor: "transparent" },
          "&::-webkit-scrollbar-thumb": {
            bgcolor: "rgba(255,255,255,0.15)",
            borderRadius: 3,
            "&:hover": { bgcolor: "rgba(255,255,255,0.25)" },
          },
          scrollbarWidth: "thin",
          scrollbarColor: "rgba(255,255,255,0.15) transparent",
        }}
      >
        {isSessionsLoading ? (
          <LoadingSkeleton variant="list" count={6} />
        ) : (
          <>
            {/* Active tag filter indicator */}
            {activeTagFilter !== null && (
              <Box
                sx={{
                  display: "flex",
                  alignItems: "center",
                  gap: 0.5,
                  px: 1,
                  mb: 0.75,
                }}
              >
                <Typography variant="caption" sx={{ color: "#8e8ea0", fontSize: "0.68rem" }}>
                  필터:
                </Typography>
                {(() => {
                  const t = allTags.find((tg) => tg.id === activeTagFilter);
                  if (!t) return null;
                  return (
                    <Box
                      sx={{
                        display: "inline-flex",
                        alignItems: "center",
                        gap: "3px",
                        px: 0.75,
                        py: "1px",
                        borderRadius: 10,
                        bgcolor: `${t.color}22`,
                        border: `1px solid ${t.color}55`,
                        fontSize: "0.68rem",
                        fontWeight: 600,
                        color: t.color,
                      }}
                    >
                      <Box
                        sx={{
                          width: 5,
                          height: 5,
                          borderRadius: "50%",
                          bgcolor: t.color,
                        }}
                      />
                      {t.name}
                    </Box>
                  );
                })()}
                <IconButton
                  size="small"
                  onClick={() => setActiveTagFilter(null)}
                  sx={{ p: 0.2, color: "#8e8ea0", "&:hover": { color: "#ececec" } }}
                >
                  <CloseIcon sx={{ fontSize: 11 }} />
                </IconButton>
              </Box>
            )}

            {filteredGroups.map((group) => (
              <Box key={group.label} sx={{ mb: 1 }}>
                <Typography
                  variant="caption"
                  sx={{
                    px: 1,
                    fontWeight: 600,
                    fontSize: "0.65rem",
                    textTransform: "uppercase",
                    letterSpacing: "0.05em",
                    color: "#8e8ea0",
                    display: "block",
                    mb: 0.5,
                  }}
                >
                  {group.label}
                </Typography>
                <List dense disablePadding>
                  {group.items.map((s) => (
                    <SessionListItem
                      key={s.id}
                      s={s}
                      isActive={s.id === currentSessionId}
                      onSelect={() => onSelectSession(s.id)}
                      onDelete={(e) => {
                        e.stopPropagation();
                        onDeleteSession(s.id);
                      }}
                      onRename={onRenameSession}
                      sessionTags={getSessionTags(s.id)}
                      allTags={allTags}
                      hasTag={hasTag}
                      onToggleTag={toggleTag}
                      isFavorite={favorites.includes(s.id)}
                      onToggleFavorite={handleToggleFavorite}
                    />
                  ))}
                </List>
              </Box>
            ))}
            {(sessionsEmpty || filteredGroups.length === 0) && !isSessionsLoading && (
              <Typography
                variant="body2"
                sx={{ p: 2, textAlign: "center", color: "#8e8ea0", fontSize: "0.82rem" }}
              >
                {searchQuery
                  ? t("sidebar.noSearchResults")
                  : showFavoritesOnly
                    ? "즐겨찾기한 대화가 없습니다"
                    : activeTagFilter !== null
                      ? t("sidebar.noTaggedChats")
                      : dateFilter !== "all"
                        ? "해당 기간에 대화가 없습니다"
                        : t("sidebar.noHistory")}
              </Typography>
            )}
          </>
        )}
      </Box>

      <Divider sx={{ borderColor: "rgba(255,255,255,0.1)" }} />

      {/* Bookmarks section */}
      <Box sx={{ flexShrink: 0 }}>
        {/* Header row */}
        <Box
          onClick={() => setBookmarksOpen((prev) => !prev)}
          role="button"
          aria-expanded={bookmarksOpen}
          aria-label={bookmarksOpen ? "저장한 답변 접기" : "저장한 답변 펼치기"}
          // i18n-managed text below
          tabIndex={0}
          onKeyDown={(e) => {
            if (e.key === "Enter" || e.key === " ") {
              e.preventDefault();
              setBookmarksOpen((prev) => !prev);
            }
          }}
          sx={{
            px: 2,
            py: 1,
            display: "flex",
            alignItems: "center",
            gap: 0.75,
            cursor: "pointer",
            userSelect: "none",
            "&:hover": { bgcolor: "rgba(255,255,255,0.05)" },
          }}
        >
          <BookmarkIcon sx={{ fontSize: 15, color: "#8e8ea0" }} />
          <Typography
            variant="caption"
            sx={{
              flex: 1,
              fontWeight: 600,
              fontSize: "0.65rem",
              textTransform: "uppercase",
              letterSpacing: "0.05em",
              color: "#8e8ea0",
            }}
          >
            {t("sidebar.savedAnswers")} {bookmarks.length > 0 ? `(${bookmarks.length})` : ""}
          </Typography>
          {bookmarksOpen ? (
            <ExpandLess sx={{ fontSize: 14, color: "#8e8ea0" }} />
          ) : (
            <ExpandMore sx={{ fontSize: 14, color: "#8e8ea0" }} />
          )}
        </Box>

        <Collapse in={bookmarksOpen}>
          <Box
            sx={{
              maxHeight: 200,
              overflowY: "auto",
              px: 1,
              pb: 0.5,
              "&::-webkit-scrollbar": { width: 4 },
              "&::-webkit-scrollbar-track": { bgcolor: "transparent" },
              "&::-webkit-scrollbar-thumb": {
                bgcolor: "rgba(255,255,255,0.15)",
                borderRadius: 2,
              },
              scrollbarWidth: "thin",
              scrollbarColor: "rgba(255,255,255,0.15) transparent",
            }}
          >
            {bookmarks.length === 0 ? (
              <Typography
                variant="caption"
                sx={{
                  display: "block",
                  px: 1,
                  py: 1,
                  color: "#8e8ea0",
                  textAlign: "center",
                }}
              >
                {t("sidebar.noSavedAnswers")}
              </Typography>
            ) : (
              <List dense disablePadding>
                {bookmarks.map((bm) => (
                  <Tooltip
                    key={bm.message_id}
                    title={
                      <Box
                        sx={{ maxWidth: 320, whiteSpace: "pre-wrap", fontSize: "0.78rem" }}
                      >
                        {bm.content}
                      </Box>
                    }
                    placement="right"
                    arrow
                  >
                    <ListItemButton
                      sx={{
                        borderRadius: 1.5,
                        mb: 0.25,
                        py: 0.6,
                        color: "#ececec",
                        "&:hover": { bgcolor: "rgba(255,255,255,0.1)" },
                        "&:hover .bm-remove-btn": { opacity: 1 },
                      }}
                    >
                      <ListItemText
                        primary={
                          bm.content.slice(0, 40) + (bm.content.length > 40 ? "…" : "")
                        }
                        secondary={new Date(bm.created_at).toLocaleDateString("ko-KR", {
                          month: "short",
                          day: "numeric",
                        })}
                        primaryTypographyProps={{
                          noWrap: true,
                          variant: "body2",
                          fontSize: "0.78rem",
                          sx: { color: "#ececec" },
                        }}
                        secondaryTypographyProps={{
                          variant: "caption",
                          sx: { color: "#8e8ea0", fontSize: "0.65rem" },
                        }}
                      />
                      <IconButton
                        className="bm-remove-btn"
                        size="small"
                        onClick={(e) => {
                          e.stopPropagation();
                          removeMutation.mutate(bm.message_id);
                        }}
                        aria-label="북마크 삭제"
                        sx={{
                          opacity: 0,
                          transition: "opacity 0.2s",
                          color: "#8e8ea0",
                          "&:hover": { color: "#ececec" },
                          flexShrink: 0,
                        }}
                      >
                        <BookmarkRemoveIcon sx={{ fontSize: 14 }} />
                      </IconButton>
                    </ListItemButton>
                  </Tooltip>
                ))}
              </List>
            )}
          </Box>
        </Collapse>
      </Box>

      <Divider sx={{ borderColor: "rgba(255,255,255,0.1)" }} />

      {/* Bottom nav */}
      <List dense sx={{ py: 0.5 }}>
        {[
          { label: "채팅", icon: <ChatIcon fontSize="small" />, path: "/chat" },
          { label: "문서", icon: <DescriptionIcon fontSize="small" />, path: "/documents" },
          { label: "관리", icon: <AdminIcon fontSize="small" />, path: "/admin" },
          { label: "모니터링", icon: <MonitorIcon fontSize="small" />, path: "/monitor" },
        ].map((item) => (
          <ListItemButton
            key={item.path}
            selected={location.pathname === item.path}
            onClick={() => navigate(item.path)}
            sx={{
              borderRadius: 1,
              mx: 1,
              py: 0.5,
              color: "#ececec",
              "&:hover": { bgcolor: "rgba(255,255,255,0.1)" },
              "&.Mui-selected": {
                bgcolor: "rgba(255,255,255,0.15)",
                color: "#ececec",
                "&:hover": { bgcolor: "rgba(255,255,255,0.2)" },
              },
            }}
          >
            <ListItemIcon sx={{ minWidth: 32, color: "inherit" }}>
              {item.icon}
            </ListItemIcon>
            <ListItemText
              primary={item.label}
              primaryTypographyProps={{ variant: "body2", fontSize: "0.85rem" }}
            />
          </ListItemButton>
        ))}

        {/* Help / Notice button */}
        <ListItemButton
          onClick={() => setHelpOpen(true)}
          sx={{
            borderRadius: 1,
            mx: 1,
            py: 0.5,
            color: "#ececec",
            "&:hover": { bgcolor: "rgba(255,255,255,0.1)" },
          }}
        >
          <ListItemIcon sx={{ minWidth: 32, color: "inherit" }}>
            <HelpOutlineIcon fontSize="small" />
          </ListItemIcon>
          <ListItemText
            primary="도움말 / 공지"
            primaryTypographyProps={{ variant: "body2", fontSize: "0.85rem" }}
          />
        </ListItemButton>

        {/* Logout button — always show (sidebar is behind PrivateRoute) */}
        <ListItemButton
          onClick={async () => {
            await logout();
            window.location.href = "/login";
          }}
          sx={{
            borderRadius: 1,
            mx: 1,
            py: 0.5,
            color: "#ff6b6b",
            "&:hover": { bgcolor: "rgba(255,107,107,0.15)" },
          }}
        >
          <ListItemIcon sx={{ minWidth: 32, color: "inherit" }}>
            <LogoutIcon fontSize="small" />
          </ListItemIcon>
          <ListItemText
            primary="로그아웃"
            primaryTypographyProps={{ variant: "body2", fontSize: "0.85rem" }}
          />
        </ListItemButton>
      </List>

      <HelpDialog open={helpOpen} onClose={() => setHelpOpen(false)} />

      {/* Tag management dialog */}
      <TagManageDialog
        open={manageOpen}
        onClose={() => setManageOpen(false)}
        customTags={customTags}
        onAddTag={addTag}
        onDeleteTag={deleteTag}
      />
    </Box>
  );
}
