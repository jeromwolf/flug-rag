import { useState, useRef, useEffect, useCallback } from "react";
import { useNetworkStatus } from "../../hooks/useNetworkStatus";
import {
  Alert,
  Avatar,
  Box,
  Button,
  Chip,
  Collapse,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  Divider,
  FormControlLabel,
  IconButton,
  InputAdornment,
  InputBase,
  MenuItem,
  Popover,
  Select,
  Slider,
  Snackbar,
  Switch,
  TextField,
  ToggleButton,
  ToggleButtonGroup,
  Tooltip,
  Typography,
  useMediaQuery,
  useTheme as useMuiTheme,
} from "@mui/material";
import type { SelectChangeEvent } from "@mui/material";
import {
  Checklist as ChecklistIcon,
  CheckCircle as CheckCircleIcon,
  Close as CloseIcon,
  CompareArrows as CompareArrowsIcon,
  ContentCopy as ContentCopyIcon,
  DarkMode as DarkModeIcon,
  Download as DownloadIcon,
  ExpandLess as ExpandLessIcon,
  ExpandMore as ExpandMoreIcon,
  FormatSize as FormatSizeIcon,
  HelpOutline as HelpOutlineIcon,
  KeyboardArrowDown as KeyboardArrowDownIcon,
  KeyboardArrowUp as KeyboardArrowUpIcon,
  LaptopMac as LaptopMacIcon,
  LightMode as LightModeIcon,
  Lock as LockIcon,
  Menu as MenuIcon,
  Notifications as NotificationsIcon,
  Palette as PaletteIcon,
  Person as PersonIcon,
  Print as PrintIcon,
  Search as SearchIcon,
  Settings as SettingsIcon,
  Share as ShareIcon,
  Summarize as SummarizeIcon,
  Logout as LogoutIcon,
  Visibility as VisibilityIcon,
  VisibilityOff as VisibilityOffIcon,
} from "@mui/icons-material";
import { useQuery } from "@tanstack/react-query";
import { adminApi, authApi } from "../../api/client";
import { useAuth } from "../../contexts/AuthContext";
import { useAppStore } from "../../stores/appStore";
import type { ThemeMode, FontSize } from "../../stores/appStore";
import type { Message } from "../../types";
import { useLanguage } from "../../contexts/LanguageContext";
import type { Language } from "../../contexts/LanguageContext";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const FALLBACK_MODELS = [
  { id: "default", name: "기본 모델" },
  { id: "vllm", name: "vLLM" },
  { id: "ollama", name: "Ollama" },
  { id: "openai", name: "OpenAI" },
  { id: "anthropic", name: "Anthropic" },
];

const ROLE_LABELS: Record<string, string> = {
  admin: "관리자",
  manager: "매니저",
  expert: "전문가",
  user: "일반 사용자",
  viewer: "조회자",
};

const ROLE_COLORS: Record<
  string,
  "error" | "warning" | "info" | "success" | "default"
> = {
  admin: "error",
  manager: "warning",
  expert: "info",
  user: "success",
  viewer: "default",
};

const PASSWORD_REGEX =
  /^(?=.*[A-Z])(?=.*\d)(?=.*[!@#$%^&*()_+\-=[\]{};':"\\|,.<>/?]).{8,}$/;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function validatePassword(pw: string): string {
  if (pw.length < 8) return "최소 8자 이상이어야 합니다.";
  if (!/[A-Z]/.test(pw)) return "대문자를 1자 이상 포함해야 합니다.";
  if (!/\d/.test(pw)) return "숫자를 1자 이상 포함해야 합니다.";
  if (!/[!@#$%^&*()_+\-=[\]{};':"\\|,.<>/?]/.test(pw))
    return "특수문자를 1자 이상 포함해야 합니다.";
  return "";
}

function getInitials(name: string): string {
  if (!name) return "?";
  const parts = name.trim().split(/\s+/);
  if (parts.length >= 2) return (parts[0][0] + parts[1][0]).toUpperCase();
  return name.slice(0, 2).toUpperCase();
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface ChatTopBarProps {
  onToggleSidebar: () => void;
  selectedModel: string;
  onModelChange: (model: string) => void;
  temperature: number;
  onTemperatureChange: (temp: number) => void;
  responseMode: "rag" | "direct";
  onModeChange: (mode: "rag" | "direct") => void;
  darkMode: boolean;
  onToggleDarkMode: () => void;
  messages: Message[];
  onShowSnackbar: (msg: string, severity: "success" | "error" | "info") => void;
  compareMode: boolean;
  onToggleCompareMode: () => void;
  // Search props
  searchOpen: boolean;
  onSearchOpen: () => void;
  searchQuery: string;
  onSearchQueryChange: (q: string) => void;
  searchResults: number[];
  currentSearchIndex: number;
  onSearchNavigate: (direction: "up" | "down") => void;
  onSearchClose: () => void;
  // Session id for share feature
  sessionId?: string | null;
  // Select mode
  selectMode?: boolean;
  onToggleSelectMode?: () => void;
  // Summarize
  onSummarize?: () => void;
}

interface AuthUserLike {
  username: string;
  email: string;
  full_name: string;
  department: string;
  role: string;
}

// ---------------------------------------------------------------------------
// UserSettingsDialog (separate component to keep ChatTopBar manageable)
// ---------------------------------------------------------------------------

interface UserSettingsDialogProps {
  open: boolean;
  onClose: () => void;
  user: AuthUserLike | null;
  onLogout: () => Promise<void>;
}

function UserSettingsDialog({ open, onClose, user, onLogout }: UserSettingsDialogProps) {
  const {
    themeMode,
    setThemeMode,
    fontSize,
    setFontSize,
    notificationsEnabled,
    setNotificationsEnabled,
  } = useAppStore();
  const { language, setLanguage, t } = useLanguage();

  // Password section
  const [pwExpanded, setPwExpanded] = useState(false);
  const [currentPw, setCurrentPw] = useState("");
  const [newPw, setNewPw] = useState("");
  const [confirmPw, setConfirmPw] = useState("");
  const [showCurrentPw, setShowCurrentPw] = useState(false);
  const [showNewPw, setShowNewPw] = useState(false);
  const [showConfirmPw, setShowConfirmPw] = useState(false);
  const [pwLoading, setPwLoading] = useState(false);
  const [pwError, setPwError] = useState("");

  // Snackbar
  const [snackbar, setSnackbar] = useState<{
    open: boolean;
    message: string;
    severity: "success" | "error" | "info";
  }>({ open: false, message: "", severity: "success" });

  const showSnackbar = useCallback(
    (message: string, severity: "success" | "error" | "info") => {
      setSnackbar({ open: true, message, severity });
    },
    [],
  );

  const resetPwFields = () => {
    setCurrentPw("");
    setNewPw("");
    setConfirmPw("");
    setPwError("");
    setPwExpanded(false);
  };

  const handleClose = () => {
    resetPwFields();
    onClose();
  };

  const handlePasswordChange = async () => {
    const err = validatePassword(newPw);
    if (err) {
      setPwError(err);
      return;
    }
    if (newPw !== confirmPw) {
      setPwError("새 비밀번호와 확인이 일치하지 않습니다.");
      return;
    }
    setPwError("");
    setPwLoading(true);
    try {
      await authApi.changePassword({
        current_password: currentPw,
        new_password: newPw,
      });
      showSnackbar("비밀번호가 성공적으로 변경되었습니다.", "success");
      resetPwFields();
    } catch (e: unknown) {
      const msg =
        (e as { response?: { data?: { detail?: string } } })?.response?.data
          ?.detail ?? "비밀번호 변경에 실패했습니다.";
      setPwError(msg);
      showSnackbar(msg, "error");
    } finally {
      setPwLoading(false);
    }
  };

  const handleNotificationToggle = async (checked: boolean) => {
    if (checked && "Notification" in window) {
      const perm = await Notification.requestPermission();
      if (perm !== "granted") {
        showSnackbar("브라우저 알림 권한이 거부되었습니다.", "error");
        return;
      }
    }
    setNotificationsEnabled(checked);
    showSnackbar(
      checked
        ? "브라우저 알림이 활성화되었습니다."
        : "브라우저 알림이 비활성화되었습니다.",
      "info",
    );
  };

  const sectionSx = { display: "flex", alignItems: "center", gap: 1, mb: 1.5 };

  return (
    <>
      <Dialog
        open={open}
        onClose={handleClose}
        maxWidth="sm"
        fullWidth
        PaperProps={{ sx: { borderRadius: 3 } }}
      >
        <DialogTitle sx={{ pb: 1, fontWeight: 600 }}>{t("settings.title")}</DialogTitle>

        <DialogContent
          dividers
          sx={{ p: 3, display: "flex", flexDirection: "column", gap: 3 }}
        >
          {/* A. 프로필 */}
          <Box>
            <Box sx={sectionSx}>
              <PersonIcon fontSize="small" color="primary" />
              <Typography variant="subtitle2" fontWeight={600}>
                프로필
              </Typography>
            </Box>
            <Box
              sx={{ display: "flex", alignItems: "center", gap: 2, mb: 2 }}
            >
              <Avatar
                sx={{
                  width: 56,
                  height: 56,
                  fontSize: "1.2rem",
                  bgcolor: "primary.main",
                  flexShrink: 0,
                }}
              >
                {user ? getInitials(user.full_name || user.username) : "?"}
              </Avatar>
              <Box sx={{ flex: 1, minWidth: 0 }}>
                <Typography variant="body1" fontWeight={600} noWrap>
                  {user?.full_name || user?.username || "-"}
                </Typography>
                <Typography variant="body2" color="text.secondary" noWrap>
                  {user?.email || "-"}
                </Typography>
                {user?.role && (
                  <Chip
                    label={ROLE_LABELS[user.role] ?? user.role}
                    size="small"
                    color={ROLE_COLORS[user.role] ?? "default"}
                    sx={{ mt: 0.5, height: 20, fontSize: "0.7rem" }}
                  />
                )}
              </Box>
            </Box>
            <Box
              sx={{
                display: "grid",
                gridTemplateColumns: "1fr 1fr",
                gap: 1.5,
              }}
            >
              <TextField
                label="사용자명"
                value={user?.username || ""}
                size="small"
                slotProps={{ input: { readOnly: true } }}
              />
              <TextField
                label="부서"
                value={user?.department || "-"}
                size="small"
                slotProps={{ input: { readOnly: true } }}
              />
            </Box>
          </Box>

          <Divider />

          {/* B. 비밀번호 변경 */}
          <Box>
            <Box
              sx={{
                ...sectionSx,
                cursor: "pointer",
                userSelect: "none",
                "&:hover": { opacity: 0.8 },
              }}
              onClick={() => setPwExpanded((v) => !v)}
            >
              <LockIcon fontSize="small" color="primary" />
              <Typography
                variant="subtitle2"
                fontWeight={600}
                sx={{ flex: 1 }}
              >
                비밀번호 변경
              </Typography>
              {pwExpanded ? (
                <ExpandLessIcon fontSize="small" />
              ) : (
                <ExpandMoreIcon fontSize="small" />
              )}
            </Box>
            <Collapse in={pwExpanded}>
              <Box
                sx={{
                  display: "flex",
                  flexDirection: "column",
                  gap: 1.5,
                  pt: 0.5,
                }}
              >
                <TextField
                  label="현재 비밀번호"
                  type={showCurrentPw ? "text" : "password"}
                  value={currentPw}
                  onChange={(e) => setCurrentPw(e.target.value)}
                  size="small"
                  fullWidth
                  slotProps={{
                    input: {
                      endAdornment: (
                        <InputAdornment position="end">
                          <IconButton
                            size="small"
                            onClick={() => setShowCurrentPw((v) => !v)}
                            edge="end"
                          >
                            {showCurrentPw ? (
                              <VisibilityOffIcon fontSize="small" />
                            ) : (
                              <VisibilityIcon fontSize="small" />
                            )}
                          </IconButton>
                        </InputAdornment>
                      ),
                    },
                  }}
                />
                <TextField
                  label="새 비밀번호"
                  type={showNewPw ? "text" : "password"}
                  value={newPw}
                  onChange={(e) => {
                    setNewPw(e.target.value);
                    if (pwError) setPwError("");
                  }}
                  size="small"
                  fullWidth
                  helperText="최소 8자, 대문자, 숫자, 특수문자 각 1자 이상"
                  error={
                    Boolean(pwError) &&
                    !PASSWORD_REGEX.test(newPw) &&
                    newPw.length > 0
                  }
                  slotProps={{
                    input: {
                      endAdornment: (
                        <InputAdornment position="end">
                          <IconButton
                            size="small"
                            onClick={() => setShowNewPw((v) => !v)}
                            edge="end"
                          >
                            {showNewPw ? (
                              <VisibilityOffIcon fontSize="small" />
                            ) : (
                              <VisibilityIcon fontSize="small" />
                            )}
                          </IconButton>
                        </InputAdornment>
                      ),
                    },
                  }}
                />
                <TextField
                  label="새 비밀번호 확인"
                  type={showConfirmPw ? "text" : "password"}
                  value={confirmPw}
                  onChange={(e) => {
                    setConfirmPw(e.target.value);
                    if (pwError) setPwError("");
                  }}
                  size="small"
                  fullWidth
                  error={Boolean(confirmPw) && confirmPw !== newPw}
                  helperText={
                    confirmPw && confirmPw !== newPw
                      ? "비밀번호가 일치하지 않습니다."
                      : ""
                  }
                  slotProps={{
                    input: {
                      endAdornment: (
                        <InputAdornment position="end">
                          <IconButton
                            size="small"
                            onClick={() => setShowConfirmPw((v) => !v)}
                            edge="end"
                          >
                            {showConfirmPw ? (
                              <VisibilityOffIcon fontSize="small" />
                            ) : (
                              <VisibilityIcon fontSize="small" />
                            )}
                          </IconButton>
                        </InputAdornment>
                      ),
                    },
                  }}
                />
                {pwError && (
                  <Alert severity="error" sx={{ py: 0.5, fontSize: "0.8rem" }}>
                    {pwError}
                  </Alert>
                )}
                <Button
                  variant="contained"
                  size="small"
                  disabled={!currentPw || !newPw || !confirmPw || pwLoading}
                  onClick={handlePasswordChange}
                  sx={{ alignSelf: "flex-start" }}
                >
                  {pwLoading ? "변경 중..." : "비밀번호 변경"}
                </Button>
              </Box>
            </Collapse>
          </Box>

          <Divider />

          {/* C. 테마 설정 */}
          <Box>
            <Box sx={sectionSx}>
              <PaletteIcon fontSize="small" color="primary" />
              <Typography variant="subtitle2" fontWeight={600}>
                테마 설정
              </Typography>
            </Box>
            <ToggleButtonGroup
              exclusive
              fullWidth
              size="small"
              value={themeMode}
              onChange={(_e, val: ThemeMode | null) => {
                if (val) setThemeMode(val);
              }}
            >
              <ToggleButton value="light" aria-label="라이트 모드">
                <LightModeIcon fontSize="small" sx={{ mr: 0.5 }} />
                라이트
              </ToggleButton>
              <ToggleButton value="dark" aria-label="다크 모드">
                <DarkModeIcon fontSize="small" sx={{ mr: 0.5 }} />
                다크
              </ToggleButton>
              <ToggleButton value="system" aria-label="시스템 설정 따름">
                <LaptopMacIcon fontSize="small" sx={{ mr: 0.5 }} />
                시스템
              </ToggleButton>
            </ToggleButtonGroup>
            <Typography
              variant="caption"
              color="text.secondary"
              sx={{ mt: 1, display: "block" }}
            >
              {themeMode === "system"
                ? "운영체제 설정을 따릅니다."
                : themeMode === "dark"
                  ? "항상 다크 모드를 사용합니다."
                  : "항상 라이트 모드를 사용합니다."}
            </Typography>
          </Box>

          <Divider />

          {/* D. 글꼴 크기 */}
          <Box>
            <Box sx={sectionSx}>
              <FormatSizeIcon fontSize="small" color="primary" />
              <Typography variant="subtitle2" fontWeight={600}>
                글꼴 크기
              </Typography>
            </Box>
            <ToggleButtonGroup
              exclusive
              fullWidth
              size="small"
              value={fontSize}
              onChange={(_e, val: FontSize | null) => {
                if (val) setFontSize(val);
              }}
            >
              <ToggleButton value="small" aria-label="작게">
                <Typography sx={{ fontSize: "0.75rem", lineHeight: 1 }}>
                  가
                </Typography>
                <Typography variant="caption" sx={{ ml: 0.5 }}>
                  작게
                </Typography>
              </ToggleButton>
              <ToggleButton value="medium" aria-label="보통">
                <Typography sx={{ fontSize: "0.9rem", lineHeight: 1 }}>
                  가
                </Typography>
                <Typography variant="caption" sx={{ ml: 0.5 }}>
                  보통
                </Typography>
              </ToggleButton>
              <ToggleButton value="large" aria-label="크게">
                <Typography sx={{ fontSize: "1.1rem", lineHeight: 1 }}>
                  가
                </Typography>
                <Typography variant="caption" sx={{ ml: 0.5 }}>
                  크게
                </Typography>
              </ToggleButton>
            </ToggleButtonGroup>
            <Typography
              variant="caption"
              color="text.secondary"
              sx={{ mt: 1, display: "block" }}
            >
              현재:{" "}
              {fontSize === "small"
                ? "14px (작게)"
                : fontSize === "large"
                  ? "18px (크게)"
                  : "16px (보통)"}
            </Typography>
          </Box>

          <Divider />

          {/* E. 알림 설정 */}
          <Box>
            <Box sx={sectionSx}>
              <NotificationsIcon fontSize="small" color="primary" />
              <Typography variant="subtitle2" fontWeight={600}>
                알림 설정
              </Typography>
            </Box>
            <FormControlLabel
              control={
                <Switch
                  checked={notificationsEnabled}
                  onChange={(e) => handleNotificationToggle(e.target.checked)}
                  size="small"
                />
              }
              label={
                <Box>
                  <Typography variant="body2">브라우저 알림</Typography>
                  <Typography variant="caption" color="text.secondary">
                    {notificationsEnabled
                      ? "알림이 활성화되어 있습니다."
                      : "AI 응답 완료 시 알림을 받습니다."}
                  </Typography>
                </Box>
              }
              sx={{ alignItems: "flex-start", ml: 0 }}
            />
            {"Notification" in window &&
              Notification.permission === "denied" && (
                <Alert
                  severity="warning"
                  sx={{ mt: 1, py: 0.5, fontSize: "0.8rem" }}
                >
                  브라우저 알림 권한이 차단되어 있습니다. 브라우저 설정에서
                  허용해 주세요.
                </Alert>
              )}
          </Box>

          <Divider />

          {/* F. 언어 설정 */}
          <Box>
            <Box sx={sectionSx}>
              <Typography variant="subtitle2" fontWeight={600} sx={{ flex: 1 }}>
                {t("settings.language")}
              </Typography>
            </Box>
            <ToggleButtonGroup
              exclusive
              fullWidth
              size="small"
              value={language}
              onChange={(_e, val: Language | null) => {
                if (val) setLanguage(val);
              }}
            >
              <ToggleButton value="ko" aria-label="한국어">
                🇰🇷 {t("settings.korean")}
              </ToggleButton>
              <ToggleButton value="en" aria-label="English">
                🇺🇸 {t("settings.english")}
              </ToggleButton>
            </ToggleButtonGroup>
          </Box>
        </DialogContent>

        <DialogActions sx={{ px: 3, py: 2, justifyContent: "space-between" }}>
          <Button
            color="error"
            size="small"
            onClick={async () => {
              onClose();
              await onLogout();
              window.location.href = "/login";
            }}
          >
            로그아웃
          </Button>
          <Button onClick={handleClose} variant="outlined" size="small">
            {t("settings.close")}
          </Button>
        </DialogActions>
      </Dialog>

      <Snackbar
        open={snackbar.open}
        autoHideDuration={4000}
        onClose={() => setSnackbar((s) => ({ ...s, open: false }))}
        anchorOrigin={{ vertical: "bottom", horizontal: "center" }}
      >
        <Alert
          onClose={() => setSnackbar((s) => ({ ...s, open: false }))}
          severity={snackbar.severity}
          variant="filled"
          sx={{ width: "100%" }}
        >
          {snackbar.message}
        </Alert>
      </Snackbar>
    </>
  );
}

// ---------------------------------------------------------------------------
// ShareDialog component
// ---------------------------------------------------------------------------

type ShareExpiry = "1d" | "7d" | "30d" | "unlimited";

const EXPIRY_LABELS: Record<ShareExpiry, string> = {
  "1d": "1일",
  "7d": "7일",
  "30d": "30일",
  unlimited: "무제한",
};

interface ShareDialogProps {
  open: boolean;
  onClose: () => void;
  sessionId: string;
}

function ShareDialog({ open, onClose, sessionId }: ShareDialogProps) {
  const muiTheme = useMuiTheme();
  const isMobile = useMediaQuery(muiTheme.breakpoints.down("sm"));

  const shareUrl = `${window.location.origin}/shared/${sessionId}`;

  const [includeSources, setIncludeSources] = useState(true);
  const [expiry, setExpiry] = useState<ShareExpiry>("7d");
  const [copied, setCopied] = useState(false);
  const copyTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Reset state whenever the dialog opens
  useEffect(() => {
    if (open) {
      setCopied(false);
      setIncludeSources(true);
      setExpiry("7d");
    }
  }, [open]);

  // Clean up timeout on unmount
  useEffect(() => {
    return () => {
      if (copyTimeoutRef.current) clearTimeout(copyTimeoutRef.current);
    };
  }, []);

  const handleCopyUrl = async () => {
    try {
      await navigator.clipboard.writeText(shareUrl);
      setCopied(true);
      if (copyTimeoutRef.current) clearTimeout(copyTimeoutRef.current);
      copyTimeoutRef.current = setTimeout(() => setCopied(false), 2000);
    } catch {
      // Fallback for environments without clipboard API
      const el = document.createElement("textarea");
      el.value = shareUrl;
      el.style.position = "fixed";
      el.style.opacity = "0";
      document.body.appendChild(el);
      el.select();
      document.execCommand("copy");
      document.body.removeChild(el);
      setCopied(true);
      if (copyTimeoutRef.current) clearTimeout(copyTimeoutRef.current);
      copyTimeoutRef.current = setTimeout(() => setCopied(false), 2000);
    }
  };

  return (
    <Dialog
      open={open}
      onClose={onClose}
      maxWidth="xs"
      fullWidth
      fullScreen={isMobile}
      PaperProps={{
        sx: {
          borderRadius: isMobile ? 0 : 3,
          overflow: "hidden",
        },
      }}
    >
      {/* Header */}
      <DialogTitle
        sx={{
          pb: 1,
          pt: 2.5,
          px: 3,
          fontWeight: 700,
          fontSize: "1rem",
          display: "flex",
          alignItems: "center",
          gap: 1,
        }}
      >
        <ShareIcon fontSize="small" color="primary" />
        대화 공유
        <Box sx={{ flex: 1 }} />
        <IconButton
          size="small"
          onClick={onClose}
          aria-label="닫기"
          sx={{ mr: -0.5 }}
        >
          <CloseIcon fontSize="small" />
        </IconButton>
      </DialogTitle>

      <DialogContent sx={{ px: 3, pb: 0, pt: 1 }}>
        {/* URL display + copy */}
        <Typography
          variant="caption"
          color="text.secondary"
          sx={{ display: "block", mb: 0.75, fontWeight: 500 }}
        >
          공유 링크
        </Typography>
        <Box
          sx={{
            display: "flex",
            alignItems: "center",
            gap: 1,
            mb: 3,
            p: 1.25,
            borderRadius: 2,
            border: "1px solid",
            borderColor: "divider",
            bgcolor: (theme) =>
              theme.palette.mode === "dark"
                ? "rgba(255,255,255,0.04)"
                : "rgba(0,0,0,0.03)",
          }}
        >
          <Typography
            variant="body2"
            sx={{
              flex: 1,
              overflow: "hidden",
              textOverflow: "ellipsis",
              whiteSpace: "nowrap",
              fontSize: "0.8rem",
              color: "text.secondary",
              fontFamily: "monospace",
            }}
            title={shareUrl}
          >
            {shareUrl}
          </Typography>
          <Tooltip title={copied ? "복사됨!" : "링크 복사"}>
            <IconButton
              size="small"
              onClick={handleCopyUrl}
              aria-label="공유 링크 복사"
              sx={{
                flexShrink: 0,
                color: copied ? "success.main" : "primary.main",
                transition: "color 0.2s ease",
              }}
            >
              {copied ? (
                <CheckCircleIcon fontSize="small" />
              ) : (
                <ContentCopyIcon fontSize="small" />
              )}
            </IconButton>
          </Tooltip>
        </Box>

        {copied && (
          <Box
            sx={{
              display: "flex",
              alignItems: "center",
              gap: 0.75,
              mb: 2.5,
              mt: -2,
              px: 0.5,
            }}
          >
            <CheckCircleIcon sx={{ fontSize: 14, color: "success.main" }} />
            <Typography
              variant="caption"
              sx={{ color: "success.main", fontWeight: 500 }}
            >
              복사됨!
            </Typography>
          </Box>
        )}

        {/* Options */}
        <Typography
          variant="caption"
          color="text.secondary"
          sx={{ display: "block", mb: 1.5, fontWeight: 500 }}
        >
          공유 옵션
        </Typography>

        {/* Include sources toggle */}
        <Box
          sx={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            mb: 2,
            py: 1,
            px: 1.5,
            borderRadius: 2,
            border: "1px solid",
            borderColor: "divider",
          }}
        >
          <Box>
            <Typography variant="body2" fontWeight={500} sx={{ lineHeight: 1.3 }}>
              소스 포함
            </Typography>
            <Typography variant="caption" color="text.secondary">
              {includeSources ? "참조 문서 출처를 함께 공유합니다" : "메시지만 공유합니다"}
            </Typography>
          </Box>
          <Switch
            checked={includeSources}
            onChange={(e) => setIncludeSources(e.target.checked)}
            size="small"
            color="primary"
            inputProps={{ "aria-label": "소스 포함 여부" }}
          />
        </Box>

        {/* Expiry selector */}
        <Box sx={{ mb: 2 }}>
          <Typography
            variant="body2"
            fontWeight={500}
            sx={{ mb: 1, display: "block" }}
          >
            링크 유효 기간
          </Typography>
          <ToggleButtonGroup
            exclusive
            fullWidth
            size="small"
            value={expiry}
            onChange={(_e, val: ShareExpiry | null) => {
              if (val) setExpiry(val);
            }}
            aria-label="링크 유효 기간 선택"
          >
            {(Object.entries(EXPIRY_LABELS) as [ShareExpiry, string][]).map(
              ([value, label]) => (
                <ToggleButton
                  key={value}
                  value={value}
                  aria-label={label}
                  sx={{ fontSize: "0.78rem", py: 0.75 }}
                >
                  {label}
                </ToggleButton>
              )
            )}
          </ToggleButtonGroup>
        </Box>
      </DialogContent>

      <DialogActions sx={{ px: 3, py: 2.5, gap: 1, flexDirection: "column" }}>
        <Button
          variant="contained"
          fullWidth
          onClick={handleCopyUrl}
          startIcon={copied ? <CheckCircleIcon /> : <ContentCopyIcon />}
          color={copied ? "success" : "primary"}
          sx={{
            borderRadius: 2,
            fontWeight: 600,
            transition: "background-color 0.2s ease",
          }}
        >
          {copied ? "복사됨!" : "링크 복사"}
        </Button>
      </DialogActions>
    </Dialog>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export function ChatTopBar({
  onToggleSidebar,
  selectedModel,
  onModelChange,
  temperature,
  onTemperatureChange,
  responseMode,
  onModeChange,
  darkMode,
  onToggleDarkMode,
  messages,
  onShowSnackbar,
  compareMode,
  onToggleCompareMode,
  searchOpen,
  onSearchOpen,
  searchQuery,
  onSearchQueryChange,
  searchResults,
  currentSearchIndex,
  onSearchNavigate,
  onSearchClose,
  sessionId,
  selectMode = false,
  onToggleSelectMode,
  onSummarize,
}: ChatTopBarProps) {
  const muiTheme = useMuiTheme();
  const isMobile = useMediaQuery(muiTheme.breakpoints.down("sm"));
  const { language, setLanguage, t } = useLanguage();

  const { user, logout } = useAuth();
  const { isOnline } = useNetworkStatus();

  const [anchorEl, setAnchorEl] = useState<HTMLButtonElement | null>(null);
  const settingsOpen = Boolean(anchorEl);

  const [settingsDialogOpen, setSettingsDialogOpen] = useState(false);
  const [shareDialogOpen, setShareDialogOpen] = useState(false);

  const hasSession = Boolean(sessionId);
  const hasMessages = messages.filter((m) => m.role !== "system").length > 0;

  const searchInputRef = useRef<HTMLInputElement | null>(null);

  const handleCloseSearch = () => {
    onSearchClose();
  };

  useEffect(() => {
    if (searchOpen) {
      const t = setTimeout(() => searchInputRef.current?.focus(), 50);
      return () => clearTimeout(t);
    }
  }, [searchOpen]);

  const resultLabel =
    searchResults.length === 0
      ? "0 / 0"
      : `${currentSearchIndex + 1} / ${searchResults.length}`;

  const { data: modelsData } = useQuery({
    queryKey: ["models"],
    queryFn: async () => {
      const res = await adminApi.listModels();
      return (res.data?.models ?? []) as Array<{ id: string; name: string }>;
    },
    staleTime: 300000,
  });

  const modelOptions =
    modelsData && modelsData.length > 0 ? modelsData : FALLBACK_MODELS;

  const handleCopyConversation = async () => {
    if (messages.length === 0) {
      onShowSnackbar("복사할 대화가 없습니다.", "info");
      return;
    }
    const text = messages
      .filter((m) => m.role !== "system")
      .map((m) => {
        const label = m.role === "user" ? "사용자" : "AI 어시스턴트";
        return `${label}: ${m.content}`;
      })
      .join("\n\n");

    try {
      await navigator.clipboard.writeText(text);
      onShowSnackbar("대화가 클립보드에 복사되었습니다.", "success");
    } catch {
      onShowSnackbar("복사에 실패했습니다.", "error");
    }
  };

  const handleDownloadConversation = () => {
    if (messages.length === 0) {
      onShowSnackbar("내보낼 대화가 없습니다.", "info");
      return;
    }

    const now = new Date();
    const dateStr = now.toLocaleDateString("ko-KR", {
      year: "numeric",
      month: "2-digit",
      day: "2-digit",
    });

    const formatTime = (isoStr: string) => {
      const d = new Date(isoStr);
      if (isNaN(d.getTime())) return "";
      return d.toLocaleTimeString("ko-KR", {
        hour: "2-digit",
        minute: "2-digit",
      });
    };

    const lines: string[] = [
      "# AI 대화 기록",
      `날짜: ${dateStr}`,
      "",
      "---",
    ];

    const visibleMessages = messages.filter((m) => m.role !== "system");

    visibleMessages.forEach((m) => {
      lines.push("");
      const label = m.role === "user" ? "사용자" : "AI 어시스턴트";
      const timeStr = formatTime(m.createdAt);
      lines.push(`## ${label}${timeStr ? ` (${timeStr})` : ""}`);
      lines.push(m.content);

      if (m.role === "assistant") {
        const metaParts: string[] = [];

        if (m.sources && m.sources.length > 0) {
          const srcList = m.sources
            .map(
              (s) =>
                `${s.filename}${s.page != null ? ` (p.${s.page})` : ""}`,
            )
            .join(", ");
          lines.push(`**출처:** ${srcList}`);
        }

        if (m.confidenceScore != null) {
          metaParts.push(`**신뢰도:** ${m.confidenceScore.toFixed(2)}`);
        }
        if (m.modelUsed) {
          metaParts.push(`**모델:** ${m.modelUsed}`);
        }
        if (m.latencyMs != null) {
          metaParts.push(`**응답시간:** ${(m.latencyMs / 1000).toFixed(1)}s`);
        }

        if (metaParts.length > 0) {
          lines.push(metaParts.join(" | "));
        }
      }

      lines.push("");
      lines.push("---");
    });

    const markdownContent = lines.join("\n");
    const blob = new Blob([markdownContent], {
      type: "text/markdown;charset=utf-8",
    });
    const url = URL.createObjectURL(blob);

    const fileDate = now
      .toLocaleDateString("ko-KR", {
        year: "numeric",
        month: "2-digit",
        day: "2-digit",
      })
      .replace(/\./g, "")
      .replace(/\s/g, "_");
    const filename = `AI_대화_${fileDate}.md`;

    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = filename;
    anchor.click();
    URL.revokeObjectURL(url);

    onShowSnackbar("대화가 파일로 저장되었습니다.", "success");
  };

  const handlePrint = () => {
    if (messages.filter((m) => m.role !== "system").length === 0) {
      onShowSnackbar("인쇄할 대화가 없습니다.", "info");
      return;
    }
    window.print();
  };

  return (
    <Box
      sx={{
        borderBottom: "1px solid",
        borderColor: darkMode
          ? "rgba(255,255,255,0.05)"
          : "rgba(0,0,0,0.05)",
        bgcolor: "transparent",
      }}
    >
      {/* Main bar */}
      <Box
        sx={{
          display: "flex",
          alignItems: "center",
          gap: 0.5,
          px: 1.5,
          minHeight: 48,
        }}
      >
        {/* Left: menu toggle with offline indicator */}
        <Tooltip title={isOnline ? "" : "오프라인 상태"}>
          <Box sx={{ position: "relative", display: "inline-flex" }}>
            <IconButton
              onClick={onToggleSidebar}
              size="small"
              aria-label="사이드바 열기/닫기"
            >
              <MenuIcon fontSize="small" />
            </IconButton>
            {!isOnline && (
              <Box
                aria-hidden="true"
                sx={{
                  position: "absolute",
                  top: 4,
                  right: 4,
                  width: 8,
                  height: 8,
                  borderRadius: "50%",
                  bgcolor: "error.main",
                  border: "1.5px solid",
                  borderColor: "background.default",
                  pointerEvents: "none",
                }}
              />
            )}
          </Box>
        </Tooltip>

        <Box sx={{ flex: 1 }} />

        {/* Center: model selector */}
        <Select
          size="small"
          value={selectedModel || "default"}
          onChange={(e: SelectChangeEvent) => onModelChange(e.target.value)}
          aria-label="AI 모델 선택"
          sx={{
            borderRadius: 2,
            minWidth: isMobile ? 100 : 140,
            "& .MuiOutlinedInput-notchedOutline": { borderColor: "divider" },
            "& .MuiSelect-select": {
              py: 0.75,
              fontSize: isMobile ? "0.8rem" : "0.875rem",
            },
          }}
        >
          {modelOptions.map((m) => (
            <MenuItem key={m.id} value={m.id} sx={{ fontSize: "0.875rem" }}>
              {isMobile
                ? m.name.length > 10
                  ? m.name.slice(0, 10) + "…"
                  : m.name
                : m.name}
            </MenuItem>
          ))}
        </Select>

        <Box sx={{ flex: 1 }} />

        {/* Select mode toggle */}
        {onToggleSelectMode && (
          <Tooltip title={selectMode ? "선택 모드 종료" : "메시지 선택"}>
            <IconButton
              onClick={onToggleSelectMode}
              size="small"
              aria-label={selectMode ? "선택 모드 종료" : "메시지 선택 모드"}
              sx={{
                color: selectMode ? "primary.main" : "text.secondary",
                bgcolor: selectMode ? "primary.main" + "1a" : "transparent",
                border: selectMode ? "1px solid" : "1px solid transparent",
                borderColor: selectMode ? "primary.main" : "transparent",
                borderRadius: 1.5,
                transition: "all 0.2s ease",
              }}
            >
              <ChecklistIcon fontSize="small" />
            </IconButton>
          </Tooltip>
        )}

        {/* Search */}
        <Tooltip title="대화 내 검색 (Ctrl+F)">
          <IconButton
            onClick={onSearchOpen}
            size="small"
            aria-label="대화 내 검색"
            sx={{
              color: searchOpen ? "primary.main" : "text.secondary",
              bgcolor: searchOpen ? "primary.main" + "1a" : "transparent",
            }}
          >
            <SearchIcon fontSize="small" />
          </IconButton>
        </Tooltip>

        <Tooltip title="대화 복사">
          <IconButton
            onClick={handleCopyConversation}
            size="small"
            aria-label="대화 복사"
            sx={{ display: { xs: "none", sm: "inline-flex" } }}
          >
            <ContentCopyIcon fontSize="small" />
          </IconButton>
        </Tooltip>

        <Tooltip title="대화 내보내기">
          <IconButton
            onClick={handleDownloadConversation}
            size="small"
            aria-label="대화 내보내기"
            sx={{ display: { xs: "none", sm: "inline-flex" } }}
          >
            <DownloadIcon fontSize="small" />
          </IconButton>
        </Tooltip>

        <Tooltip
          title={
            !hasSession
              ? "활성 대화가 없습니다"
              : !hasMessages
                ? "공유할 메시지가 없습니다"
                : "대화 공유"
          }
        >
          <span>
            <IconButton
              onClick={() => setShareDialogOpen(true)}
              size="small"
              aria-label="대화 공유"
              disabled={!hasSession || !hasMessages}
              sx={{ display: { xs: "none", sm: "inline-flex" } }}
            >
              <ShareIcon fontSize="small" />
            </IconButton>
          </span>
        </Tooltip>

        <Tooltip title="대화 인쇄">
          <IconButton
            onClick={handlePrint}
            size="small"
            aria-label="대화 인쇄"
            className="no-print"
            sx={{ display: { xs: "none", sm: "inline-flex" } }}
          >
            <PrintIcon fontSize="small" />
          </IconButton>
        </Tooltip>

        <Tooltip title="대화 요약">
          <span>
            <IconButton
              onClick={onSummarize}
              size="small"
              aria-label="대화 요약"
              disabled={messages.filter((m) => m.role !== "system").length < 5 || !onSummarize}
              sx={{ display: { xs: "none", sm: "inline-flex" } }}
            >
              <SummarizeIcon fontSize="small" />
            </IconButton>
          </span>
        </Tooltip>

        <Tooltip title="모델 비교 모드">
          <IconButton
            onClick={onToggleCompareMode}
            size="small"
            aria-label="모델 비교 모드"
            sx={{
              color: compareMode ? "primary.main" : "text.secondary",
              bgcolor: compareMode ? "primary.main" + "1a" : "transparent",
              border: compareMode ? "1px solid" : "1px solid transparent",
              borderColor: compareMode ? "primary.main" : "transparent",
              borderRadius: 1.5,
              transition: "all 0.2s ease",
              "&:hover": {
                bgcolor: compareMode ? "primary.main" + "26" : "action.hover",
              },
            }}
          >
            <CompareArrowsIcon fontSize="small" />
          </IconButton>
        </Tooltip>

        {/* Language toggle */}
        <Tooltip title={t("topbar.language")}>
          <IconButton
            onClick={() => setLanguage(language === "ko" ? "en" : "ko")}
            size="small"
            aria-label="언어 전환"
            sx={{
              fontSize: "0.7rem",
              fontWeight: 700,
              width: 32,
              height: 32,
              borderRadius: 1.5,
              color: "text.secondary",
              "&:hover": { color: "text.primary" },
            }}
          >
            {language === "ko" ? "EN" : "한"}
          </IconButton>
        </Tooltip>

        <Tooltip title={darkMode ? "라이트 모드" : "다크 모드"}>
          <IconButton
            onClick={onToggleDarkMode}
            size="small"
            aria-label={
              darkMode ? "라이트 모드로 전환" : "다크 모드로 전환"
            }
          >
            {darkMode ? (
              <LightModeIcon fontSize="small" />
            ) : (
              <DarkModeIcon fontSize="small" />
            )}
          </IconButton>
        </Tooltip>

        {/* AI Settings popover */}
        <Tooltip title="AI 설정">
          <IconButton
            onClick={(e) => setAnchorEl(e.currentTarget)}
            size="small"
            aria-label="AI 설정 열기"
            sx={{ bgcolor: settingsOpen ? "action.selected" : "transparent" }}
          >
            <SettingsIcon fontSize="small" />
          </IconButton>
        </Tooltip>

        {/* User Settings dialog */}
        <Tooltip title="사용자 설정">
          <IconButton
            onClick={() => setSettingsDialogOpen(true)}
            size="small"
            aria-label="사용자 설정 열기"
            sx={{
              bgcolor: settingsDialogOpen ? "action.selected" : "transparent",
            }}
          >
            {user ? (
              <Avatar
                sx={{
                  width: 24,
                  height: 24,
                  fontSize: "0.65rem",
                  bgcolor: "primary.main",
                }}
              >
                {getInitials(user.full_name || user.username)}
              </Avatar>
            ) : (
              <PersonIcon fontSize="small" />
            )}
          </IconButton>
        </Tooltip>

        {/* Logout button — always visible */}
        <Tooltip title="로그아웃">
          <IconButton
            onClick={logout}
            size="small"
            aria-label="로그아웃"
            sx={{ color: "error.main" }}
          >
            <LogoutIcon fontSize="small" />
          </IconButton>
        </Tooltip>

        {/* AI Settings popover content */}
        <Popover
          open={settingsOpen}
          anchorEl={anchorEl}
          onClose={() => setAnchorEl(null)}
          anchorOrigin={{ vertical: "bottom", horizontal: "right" }}
          transformOrigin={{ vertical: "top", horizontal: "right" }}
          slotProps={{ paper: { sx: { p: 2.5, width: 280, mt: 1 } } }}
        >
          <Typography variant="subtitle2" sx={{ mb: 1.5 }}>
            AI 설정
          </Typography>

          <Typography variant="caption" color="text.secondary">
            Temperature: {temperature}
          </Typography>
          <Slider
            value={temperature}
            onChange={(_, val) => onTemperatureChange(val as number)}
            min={0}
            max={2}
            step={0.1}
            valueLabelDisplay="auto"
            size="small"
            sx={{ mb: 2, mt: 0.5 }}
            aria-label="Temperature 조절"
          />

          <Box
            sx={{ display: "flex", alignItems: "center", gap: 0.5, mb: 0.5 }}
          >
            <Typography variant="caption" color="text.secondary">
              응답 모드
            </Typography>
            <Tooltip
              title="auto: 질문에 따라 자동 선택 (권장) | rag: 문서 기반 검색 후 답변 | direct: 문서 검색 없이 LLM 직접 답변"
              arrow
            >
              <HelpOutlineIcon
                sx={{ fontSize: 16, color: "text.secondary", cursor: "help" }}
              />
            </Tooltip>
          </Box>
          <ToggleButtonGroup
            size="small"
            exclusive
            fullWidth
            value={responseMode}
            onChange={(_e, val: "rag" | "direct" | null) => {
              if (val) onModeChange(val);
            }}
          >
            <ToggleButton value="rag" aria-label="RAG 모드">
              RAG
            </ToggleButton>
            <ToggleButton value="direct" aria-label="직접 응답">
              직접 응답
            </ToggleButton>
          </ToggleButtonGroup>
        </Popover>
      </Box>

      {/* Collapsible search bar */}
      <Collapse in={searchOpen} unmountOnExit>
        <Box
          sx={{
            display: "flex",
            alignItems: "center",
            gap: 1,
            px: 2,
            py: 0.75,
            borderTop: "1px solid",
            borderColor: darkMode
              ? "rgba(255,255,255,0.06)"
              : "rgba(0,0,0,0.06)",
            bgcolor: darkMode
              ? "rgba(255,255,255,0.03)"
              : "rgba(0,0,0,0.02)",
          }}
          onKeyDown={(e) => {
            if (e.key === "Escape") handleCloseSearch();
            if (e.key === "Enter") {
              if (e.shiftKey) {
                onSearchNavigate("up");
              } else {
                onSearchNavigate("down");
              }
            }
          }}
        >
          <SearchIcon
            sx={{ fontSize: 16, color: "text.secondary", flexShrink: 0 }}
          />
          <InputBase
            inputRef={searchInputRef}
            placeholder="대화 내 검색..."
            value={searchQuery}
            onChange={(e) => onSearchQueryChange(e.target.value)}
            sx={{ flex: 1, fontSize: "0.875rem", "& input": { p: 0 } }}
            inputProps={{ "aria-label": "대화 내 검색" }}
          />
          <Typography
            variant="caption"
            color="text.secondary"
            sx={{ flexShrink: 0, minWidth: 40, textAlign: "center" }}
          >
            {resultLabel}
          </Typography>
          <Tooltip title="이전 결과 (Shift+Enter)">
            <span>
              <IconButton
                size="small"
                onClick={() => onSearchNavigate("up")}
                disabled={searchResults.length === 0}
                aria-label="이전 검색 결과"
              >
                <KeyboardArrowUpIcon fontSize="small" />
              </IconButton>
            </span>
          </Tooltip>
          <Tooltip title="다음 결과 (Enter)">
            <span>
              <IconButton
                size="small"
                onClick={() => onSearchNavigate("down")}
                disabled={searchResults.length === 0}
                aria-label="다음 검색 결과"
              >
                <KeyboardArrowDownIcon fontSize="small" />
              </IconButton>
            </span>
          </Tooltip>
          <Tooltip title="검색 닫기 (Esc)">
            <IconButton
              size="small"
              onClick={handleCloseSearch}
              aria-label="검색 닫기"
            >
              <CloseIcon fontSize="small" />
            </IconButton>
          </Tooltip>
        </Box>
      </Collapse>

      {/* User Settings Dialog */}
      <UserSettingsDialog
        open={settingsDialogOpen}
        onClose={() => setSettingsDialogOpen(false)}
        user={user}
        onLogout={logout}
      />

      {/* Share Dialog */}
      {sessionId && (
        <ShareDialog
          open={shareDialogOpen}
          onClose={() => setShareDialogOpen(false)}
          sessionId={sessionId}
        />
      )}
    </Box>
  );
}
