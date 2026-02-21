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
  SmartToy as BotIcon,
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

const SIDEBAR_WIDTH = 280;

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
        bgcolor: "background.default",
      }}
    >
      {/* Logo */}
      <Box sx={{ p: 2, pb: 1, display: "flex", alignItems: "center", gap: 1 }}>
        <BotIcon color="primary" />
        <Typography variant="h6" fontWeight={700}>
          Flux RAG
        </Typography>
      </Box>

      {/* New chat button */}
      <Box sx={{ px: 2, pb: 1.5 }}>
        <Button
          variant="contained"
          fullWidth
          startIcon={<AddIcon />}
          onClick={onNewChat}
          sx={{
            borderRadius: 2,
            textTransform: "none",
            boxShadow: 1,
            py: 1,
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
                  <SearchIcon fontSize="small" sx={{ color: "text.secondary" }} />
                </InputAdornment>
              ),
            },
          }}
          sx={{
            "& .MuiOutlinedInput-root": {
              borderRadius: 2,
              bgcolor: "action.hover",
              "& fieldset": { border: "none" },
            },
          }}
        />
      </Box>

      <Divider />

      {/* Session list */}
      <Box sx={{ flex: 1, overflow: "auto", px: 1, py: 1 }}>
        {filteredGroups.map((group) => (
          <Box key={group.label} sx={{ mb: 1 }}>
            <Typography
              variant="caption"
              color="text.secondary"
              sx={{ px: 1, fontWeight: 600, fontSize: "0.7rem", textTransform: "uppercase" }}
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
                    "&:hover .delete-btn": { opacity: 1 },
                    "&.Mui-selected": {
                      bgcolor: "primary.main",
                      color: "primary.contrastText",
                      "&:hover": { bgcolor: "primary.dark" },
                      "& .MuiListItemText-secondary": {
                        color: "primary.contrastText",
                        opacity: 0.7,
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
                    }}
                    secondaryTypographyProps={{
                      variant: "caption",
                      noWrap: true,
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
                        color: "inherit",
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
            color="text.secondary"
            sx={{ p: 2, textAlign: "center" }}
          >
            {searchQuery ? "검색 결과가 없습니다" : "대화 내역이 없습니다"}
          </Typography>
        )}
      </Box>

      <Divider />

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
            sx={{ borderRadius: 1, mx: 1, py: 0.5 }}
          >
            <ListItemIcon sx={{ minWidth: 32 }}>{item.icon}</ListItemIcon>
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
