import { useState } from "react";
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
} from "@mui/icons-material";
import { useNavigate, useLocation } from "react-router-dom";

interface SessionItem {
  id: string;
  title: string;
  message_count?: number;
  messageCount?: number;
}

interface ChatSidebarProps {
  sessionGroups: { label: string; items: SessionItem[] }[];
  sessionsEmpty: boolean;
  currentSessionId: string | null;
  onSelectSession: (id: string) => void;
  onNewChat: () => void;
  onDeleteSession: (id: string) => void;
}

const SIDEBAR_WIDTH = 260;

export function ChatSidebar({
  sessionGroups,
  sessionsEmpty,
  currentSessionId,
  onSelectSession,
  onNewChat,
  onDeleteSession,
}: ChatSidebarProps) {
  const navigate = useNavigate();
  const location = useLocation();
  const [searchQuery, setSearchQuery] = useState("");

  const filteredGroups = sessionGroups
    .map((group) => ({
      ...group,
      items: group.items.filter((s) =>
        (s.title || "새 대화").toLowerCase().includes(searchQuery.toLowerCase())
      ),
    }))
    .filter((group) => group.items.length > 0);

  return (
    <Box
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
          Flux AI
        </Typography>
      </Box>

      {/* New chat button */}
      <Box sx={{ px: 2, pb: 1.5 }}>
        <Button
          variant="outlined"
          fullWidth
          startIcon={<AddIcon />}
          onClick={onNewChat}
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
          새 대화
        </Button>
      </Box>

      {/* Search */}
      <Box sx={{ px: 2, pb: 1 }}>
        <TextField
          size="small"
          fullWidth
          placeholder="대화 검색..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          slotProps={{
            input: {
              startAdornment: (
                <InputAdornment position="start">
                  <SearchIcon fontSize="small" sx={{ color: "#8e8ea0" }} />
                </InputAdornment>
              ),
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
      </Box>

      <Divider sx={{ borderColor: "rgba(255,255,255,0.1)" }} />

      {/* Session list */}
      <Box sx={{ flex: 1, overflow: "auto", px: 1, py: 1 }}>
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
                <ListItemButton
                  key={s.id}
                  selected={s.id === currentSessionId}
                  onClick={() => onSelectSession(s.id)}
                  sx={{
                    borderRadius: 1.5,
                    mb: 0.25,
                    py: 0.75,
                    color: "#ececec",
                    "&:hover": {
                      bgcolor: "rgba(255,255,255,0.1)",
                    },
                    "&:hover .delete-btn": { opacity: 1 },
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
                  <ListItemText
                    primary={s.title || "새 대화"}
                    secondary={`${s.message_count ?? s.messageCount ?? 0}개 메시지`}
                    primaryTypographyProps={{
                      noWrap: true,
                      variant: "body2",
                      fontWeight: s.id === currentSessionId ? 600 : 400,
                      sx: { color: "#ececec" },
                    }}
                    secondaryTypographyProps={{
                      variant: "caption",
                      noWrap: true,
                      sx: { color: "#8e8ea0" },
                    }}
                  />
                  <Tooltip title="삭제">
                    <IconButton
                      className="delete-btn"
                      size="small"
                      onClick={(e) => {
                        e.stopPropagation();
                        onDeleteSession(s.id);
                      }}
                      sx={{
                        opacity: 0,
                        transition: "opacity 0.2s",
                        color: "#8e8ea0",
                        "&:hover": { color: "#ececec" },
                      }}
                    >
                      <DeleteIcon fontSize="small" />
                    </IconButton>
                  </Tooltip>
                </ListItemButton>
              ))}
            </List>
          </Box>
        ))}
        {(sessionsEmpty || (searchQuery && filteredGroups.length === 0)) && (
          <Typography
            variant="body2"
            sx={{ p: 2, textAlign: "center", color: "#8e8ea0" }}
          >
            {searchQuery ? "검색 결과가 없습니다" : "대화 내역이 없습니다"}
          </Typography>
        )}
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
              "&:hover": {
                bgcolor: "rgba(255,255,255,0.1)",
              },
              "&.Mui-selected": {
                bgcolor: "rgba(255,255,255,0.15)",
                color: "#ececec",
                "&:hover": {
                  bgcolor: "rgba(255,255,255,0.2)",
                },
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
      </List>
    </Box>
  );
}
