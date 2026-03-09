import { useState } from "react";
import {
  Box,
  Drawer,
  List,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Typography,
  Divider,
  IconButton,
  Tooltip,
  Chip,
} from "@mui/material";
import {
  Chat as ChatIcon,
  Description as DocsIcon,
  Settings as AdminIcon,
  BarChart as MonitorIcon,
  SmartToy as AgentIcon,
  FactCheck as QualityIcon,
  Person as PersonIcon,
  History as HistoryIcon,
  ChevronLeft as ChevronLeftIcon,
  Menu as MenuIcon,
  Logout as LogoutIcon,
} from "@mui/icons-material";
import { useNavigate, useLocation } from "react-router-dom";
import { useAuth } from "../contexts/AuthContext";

const DRAWER_WIDTH = 240;

const navItems = [
  { path: "/chat", label: "채팅", icon: <ChatIcon />, permission: "chat:read", roleLabel: null },
  { path: "/personal-knowledge", label: "개인 지식공간", icon: <PersonIcon />, permission: "chat:read", roleLabel: null },
  { path: "/activity", label: "활동 기록", icon: <HistoryIcon />, permission: "chat:read", roleLabel: null },
  { path: "/documents", label: "문서 관리", icon: <DocsIcon />, permission: "documents:write", roleLabel: "관리자" },
  { path: "/admin", label: "관리자", icon: <AdminIcon />, permission: "admin:read", roleLabel: "관리자" },
  { path: "/monitor", label: "모니터링", icon: <MonitorIcon />, permission: "monitor:read", roleLabel: "관리자" },
  { path: "/agent-builder", label: "에이전트 빌더", icon: <AgentIcon />, permission: "agent-builder:read", roleLabel: "관리자" },
  { path: "/quality", label: "RAG 품질 관리", icon: <QualityIcon />, permission: "quality:read", roleLabel: "전문가" },
];

const ROLE_LABELS: Record<string, string> = {
  admin: "관리자",
  manager: "매니저",
  expert: "전문가",
  user: "사용자",
  viewer: "뷰어",
};

interface LayoutProps {
  children: React.ReactNode;
  title?: string;
  noPadding?: boolean;
  headerActions?: React.ReactNode;
}

export default function Layout({ children, title, noPadding, headerActions }: LayoutProps) {
  const navigate = useNavigate();
  const location = useLocation();
  const [open, setOpen] = useState(true);
  const { user, logout, hasPermission, authEnabled } = useAuth();

  // Filter nav items by permission
  const visibleNav = navItems.filter(
    (item) => !authEnabled || hasPermission(item.permission),
  );

  const handleLogout = async () => {
    await logout();
    navigate("/login", { replace: true });
  };

  return (
    <Box sx={{ display: "flex", height: "100vh" }}>
      <Drawer
        variant="permanent"
        sx={{
          width: open ? DRAWER_WIDTH : 64,
          flexShrink: 0,
          transition: "width 0.2s",
          "& .MuiDrawer-paper": {
            width: open ? DRAWER_WIDTH : 64,
            boxSizing: "border-box",
            overflowX: "hidden",
            transition: "width 0.2s",
          },
        }}
      >
        <Box sx={{ p: open ? 2 : 1, display: "flex", alignItems: "center", justifyContent: open ? "space-between" : "center" }}>
          {open ? (
            <>
              <Box>
                <Typography variant="h6" sx={{ fontWeight: 700, lineHeight: 1.2 }}>
                  AI 어시스턴트
                </Typography>
              </Box>
              <IconButton size="small" onClick={() => setOpen(false)}>
                <ChevronLeftIcon />
              </IconButton>
            </>
          ) : (
            <IconButton size="small" onClick={() => setOpen(true)}>
              <MenuIcon />
            </IconButton>
          )}
        </Box>
        <Divider />
        <List sx={{ flex: 1 }}>
          {visibleNav.map((item) => (
            <Tooltip key={item.path} title={open ? "" : item.label} placement="right">
              <ListItemButton
                selected={location.pathname === item.path}
                onClick={() => navigate(item.path)}
                sx={{ px: open ? 2 : 2.5 }}
              >
                <ListItemIcon sx={{ minWidth: open ? 40 : "unset" }}>
                  {item.icon}
                </ListItemIcon>
                {open && (
                  <ListItemText
                    primary={item.label}
                    secondary={
                      item.roleLabel ? (
                        <Typography
                          component="span"
                          variant="caption"
                          sx={{ color: "text.disabled", fontSize: "0.68rem" }}
                        >
                          {item.roleLabel}
                        </Typography>
                      ) : null
                    }
                    secondaryTypographyProps={{ component: "span" }}
                  />
                )}
              </ListItemButton>
            </Tooltip>
          ))}
        </List>

        {/* User info + logout at bottom */}
        <Divider />
        <Box sx={{ p: open ? 1.5 : 1, display: "flex", alignItems: "center", gap: 1 }}>
          {open && user ? (
            <Box sx={{ flex: 1, minWidth: 0 }}>
              <Typography variant="body2" noWrap sx={{ fontWeight: 600 }}>
                {user.full_name || user.username}
              </Typography>
              <Box sx={{ display: "flex", alignItems: "center", gap: 0.5 }}>
                <Chip
                  label={ROLE_LABELS[user.role] || user.role}
                  size="small"
                  color={user.role === "admin" ? "error" : user.role === "manager" ? "warning" : user.role === "expert" ? "info" : "default"}
                  sx={{ height: 20, fontSize: "0.7rem" }}
                />
                {user.department && (
                  <Typography variant="caption" color="text.secondary" noWrap>
                    {user.department}
                  </Typography>
                )}
              </Box>
            </Box>
          ) : null}
          <Tooltip title="로그아웃" placement="right">
            <IconButton size="small" onClick={handleLogout}>
              <LogoutIcon fontSize="small" />
            </IconButton>
          </Tooltip>
        </Box>
      </Drawer>

      <Box sx={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden" }}>
        {title && (
          <Box sx={{ px: 3, py: 2, borderBottom: "1px solid", borderColor: "divider", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
            <Typography variant="h5" sx={{ fontWeight: 700 }}>
              {title}
            </Typography>
            {headerActions && (
              <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                {headerActions}
              </Box>
            )}
          </Box>
        )}
        <Box sx={{ flex: 1, overflow: "auto", ...(noPadding ? {} : { p: 3 }) }}>
          {children}
        </Box>
      </Box>
    </Box>
  );
}
