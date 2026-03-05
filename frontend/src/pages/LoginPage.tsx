import { useState, useEffect, type FormEvent } from "react";
import {
  Box,
  TextField,
  Button,
  Typography,
  InputAdornment,
  IconButton,
  CircularProgress,
  Collapse,
  Checkbox,
  FormControlLabel,
  useMediaQuery,
  useTheme,
} from "@mui/material";
import {
  Visibility,
  VisibilityOff,
  AutoAwesome as SparkleIcon,
  SearchRounded as SearchIcon,
  ArticleRounded as ReportIcon,
  LockRounded as LockIcon,
  ExpandMore as ExpandMoreIcon,
  ExpandLess as ExpandLessIcon,
  PersonRounded as PersonIcon,
  KeyRounded as KeyIcon,
} from "@mui/icons-material";
import { useAuth } from "../contexts/AuthContext";
import { useNavigate } from "react-router-dom";

// ---------------------------------------------------------------------------
// Keyframe injection — done once via a style tag
// ---------------------------------------------------------------------------

const STYLES = `
  @keyframes loginFadeInRight {
    from { opacity: 0; transform: translateX(32px); }
    to   { opacity: 1; transform: translateX(0); }
  }
  @keyframes loginFadeInUp {
    from { opacity: 0; transform: translateY(24px); }
    to   { opacity: 1; transform: translateY(0); }
  }
  @keyframes loginShake {
    0%, 100% { transform: translateX(0); }
    15%  { transform: translateX(-8px); }
    30%  { transform: translateX( 8px); }
    45%  { transform: translateX(-6px); }
    60%  { transform: translateX( 6px); }
    75%  { transform: translateX(-3px); }
    90%  { transform: translateX( 3px); }
  }
  @keyframes loginFloat {
    0%, 100% { transform: translateY(0px); }
    50%      { transform: translateY(-12px); }
  }
  @keyframes loginPulse {
    0%, 100% { opacity: 0.15; }
    50%      { opacity: 0.45; }
  }
  @keyframes loginOrbit {
    from { transform: rotate(0deg) translateX(var(--r)) rotate(0deg); }
    to   { transform: rotate(360deg) translateX(var(--r)) rotate(-360deg); }
  }
  @keyframes loginDotPulse {
    0%, 100% { opacity: 0.08; transform: scale(1); }
    50%      { opacity: 0.3; transform: scale(1.6); }
  }
`;

function injectStyles() {
  if (!document.getElementById("login-keyframes")) {
    const el = document.createElement("style");
    el.id = "login-keyframes";
    el.textContent = STYLES;
    document.head.appendChild(el);
  }
}

// ---------------------------------------------------------------------------
// Particle dot grid — CSS-only animation, purely decorative
// ---------------------------------------------------------------------------

interface DotProps {
  x: number;
  y: number;
  delay: number;
  size: number;
}

function Dot({ x, y, delay, size }: DotProps) {
  return (
    <Box
      sx={{
        position: "absolute",
        left: `${x}%`,
        top: `${y}%`,
        width: size,
        height: size,
        borderRadius: "50%",
        bgcolor: "rgba(255,255,255,0.6)",
        animation: `loginDotPulse ${2.8 + delay * 0.7}s ${delay}s ease-in-out infinite`,
        pointerEvents: "none",
      }}
    />
  );
}

function ParticleField() {
  const dots: DotProps[] = [
    { x: 8,  y: 12, delay: 0.0, size: 3 },
    { x: 22, y: 28, delay: 0.4, size: 2 },
    { x: 38, y: 8,  delay: 0.8, size: 4 },
    { x: 55, y: 20, delay: 1.2, size: 2 },
    { x: 70, y: 38, delay: 0.2, size: 3 },
    { x: 85, y: 15, delay: 1.6, size: 2 },
    { x: 12, y: 55, delay: 0.6, size: 2 },
    { x: 32, y: 68, delay: 1.0, size: 3 },
    { x: 48, y: 50, delay: 0.3, size: 2 },
    { x: 62, y: 72, delay: 1.4, size: 4 },
    { x: 78, y: 58, delay: 0.7, size: 2 },
    { x: 92, y: 45, delay: 1.8, size: 3 },
    { x: 18, y: 82, delay: 0.5, size: 2 },
    { x: 42, y: 88, delay: 1.1, size: 3 },
    { x: 68, y: 88, delay: 0.9, size: 2 },
    { x: 88, y: 78, delay: 1.5, size: 4 },
    { x: 5,  y: 38, delay: 1.9, size: 2 },
    { x: 95, y: 62, delay: 0.1, size: 3 },
    { x: 25, y: 45, delay: 1.3, size: 2 },
    { x: 75, y: 25, delay: 0.8, size: 3 },
  ];

  return (
    <Box
      sx={{
        position: "absolute",
        inset: 0,
        overflow: "hidden",
        pointerEvents: "none",
      }}
    >
      {dots.map((d, i) => (
        <Dot key={i} {...d} />
      ))}

      {/* Geometric ring accents */}
      <Box
        sx={{
          position: "absolute",
          width: 340,
          height: 340,
          border: "1px solid rgba(255,255,255,0.07)",
          borderRadius: "50%",
          bottom: -80,
          right: -80,
          animation: "loginPulse 5s 0s ease-in-out infinite",
        }}
      />
      <Box
        sx={{
          position: "absolute",
          width: 220,
          height: 220,
          border: "1px solid rgba(255,255,255,0.05)",
          borderRadius: "50%",
          bottom: -20,
          right: -20,
          animation: "loginPulse 5s 1s ease-in-out infinite",
        }}
      />
      <Box
        sx={{
          position: "absolute",
          width: 180,
          height: 180,
          border: "1px solid rgba(16,163,127,0.18)",
          borderRadius: "50%",
          top: 60,
          left: -40,
          animation: "loginPulse 6s 2s ease-in-out infinite",
        }}
      />
    </Box>
  );
}

// ---------------------------------------------------------------------------
// Feature highlight row
// ---------------------------------------------------------------------------

interface FeatureRowProps {
  icon: React.ReactNode;
  label: string;
  delay: number;
}

function FeatureRow({ icon, label, delay }: FeatureRowProps) {
  return (
    <Box
      sx={{
        display: "flex",
        alignItems: "center",
        gap: 1.5,
        opacity: 0,
        animation: `loginFadeInUp 0.6s ${delay}s ease forwards`,
      }}
    >
      <Box
        sx={{
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          width: 36,
          height: 36,
          borderRadius: "10px",
          bgcolor: "rgba(16,163,127,0.18)",
          border: "1px solid rgba(16,163,127,0.3)",
          color: "#10a37f",
          flexShrink: 0,
        }}
      >
        {icon}
      </Box>
      <Typography
        sx={{
          color: "rgba(255,255,255,0.82)",
          fontSize: "0.875rem",
          fontWeight: 400,
          letterSpacing: "0.01em",
          fontFamily: "'Pretendard Variable', sans-serif",
        }}
      >
        {label}
      </Typography>
    </Box>
  );
}

// ---------------------------------------------------------------------------
// Test accounts accordion
// ---------------------------------------------------------------------------

const TEST_ACCOUNTS = [
  { role: "관리자", username: "admin",   password: "admin123" },
  { role: "매니저", username: "manager", password: "manager123" },
  { role: "사용자", username: "user",    password: "user123" },
  { role: "뷰어",   username: "viewer",  password: "viewer123" },
];

function TestAccountsHint({
  onSelect,
}: {
  onSelect: (u: string, p: string) => void;
}) {
  const [open, setOpen] = useState(false);

  return (
    <Box sx={{ mt: 3 }}>
      <Button
        size="small"
        variant="text"
        onClick={() => setOpen((v) => !v)}
        endIcon={open ? <ExpandLessIcon /> : <ExpandMoreIcon />}
        sx={{
          color: "text.secondary",
          fontSize: "0.78rem",
          textTransform: "none",
          p: 0,
          minWidth: 0,
          "&:hover": { bgcolor: "transparent", color: "primary.main" },
        }}
      >
        테스트 계정 보기
      </Button>
      <Collapse in={open}>
        <Box
          sx={{
            mt: 1.5,
            p: 1.5,
            borderRadius: 2,
            bgcolor: "action.hover",
            border: "1px solid",
            borderColor: "divider",
          }}
        >
          {TEST_ACCOUNTS.map((acct) => (
            <Box
              key={acct.username}
              onClick={() => onSelect(acct.username, acct.password)}
              sx={{
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                py: 0.75,
                px: 1,
                borderRadius: 1.5,
                cursor: "pointer",
                transition: "background 0.15s",
                "&:hover": { bgcolor: "action.selected" },
              }}
            >
              <Typography
                variant="caption"
                sx={{ fontWeight: 600, color: "text.secondary", minWidth: 44 }}
              >
                {acct.role}
              </Typography>
              <Typography
                variant="caption"
                sx={{ color: "text.primary", fontFamily: "monospace", flex: 1, px: 1 }}
              >
                {acct.username}
              </Typography>
              <Typography
                variant="caption"
                sx={{ color: "text.disabled", fontFamily: "monospace" }}
              >
                {acct.password}
              </Typography>
            </Box>
          ))}
          <Typography
            variant="caption"
            sx={{ display: "block", mt: 1, color: "text.disabled", fontSize: "0.68rem" }}
          >
            클릭하면 자동 입력됩니다
          </Typography>
        </Box>
      </Collapse>
    </Box>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export default function LoginPage() {
  const { login } = useAuth();
  const navigate = useNavigate();
  const theme = useTheme();
  const isDesktop = useMediaQuery(theme.breakpoints.up("md"));

  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [rememberMe, setRememberMe] = useState(false);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [shaking, setShaking] = useState(false);

  useEffect(() => {
    injectStyles();
    const remembered = localStorage.getItem("app_remember_user");
    if (remembered) {
      setUsername(remembered);
      setRememberMe(true);
    }
  }, []);

  const triggerShake = () => {
    setShaking(true);
    setTimeout(() => setShaking(false), 600);
  };

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (!username.trim() || !password.trim()) {
      setError("사용자명과 비밀번호를 입력해주세요.");
      triggerShake();
      return;
    }

    setLoading(true);
    setError("");

    try {
      await login(username, password);
      if (rememberMe) {
        localStorage.setItem("app_remember_user", username);
      } else {
        localStorage.removeItem("app_remember_user");
      }
      navigate("/chat", { replace: true });
    } catch (err: unknown) {
      const axiosErr = err as {
        response?: { status?: number; data?: { detail?: string } };
      };
      if (axiosErr.response?.status === 429) {
        setError("로그인 시도 횟수가 초과되었습니다. 잠시 후 다시 시도해주세요.");
      } else if (axiosErr.response?.status === 401) {
        setError("사용자명 또는 비밀번호가 올바르지 않습니다.");
      } else {
        setError("로그인 중 오류가 발생했습니다. 다시 시도해주세요.");
      }
      triggerShake();
    } finally {
      setLoading(false);
    }
  };

  const handleTestSelect = (u: string, p: string) => {
    setUsername(u);
    setPassword(p);
    setError("");
  };

  // ── Left branding panel ──────────────────────────────────────────────────

  const BrandingPanel = (
    <Box
      sx={{
        position: "relative",
        flex: "0 0 58%",
        display: isDesktop ? "flex" : "none",
        flexDirection: "column",
        justifyContent: "center",
        alignItems: "flex-start",
        px: 7,
        py: 6,
        overflow: "hidden",
        background:
          "linear-gradient(135deg, #0a1628 0%, #1a365d 52%, #0d4a3a 100%)",
      }}
    >
      <ParticleField />

      {/* Top-left wordmark */}
      <Box
        sx={{
          position: "absolute",
          top: 28,
          left: 36,
          display: "flex",
          alignItems: "center",
          gap: 1,
          opacity: 0,
          animation: "loginFadeInUp 0.5s 0.1s ease forwards",
        }}
      >
        <Box
          sx={{
            width: 28,
            height: 28,
            borderRadius: "7px",
            bgcolor: "#10a37f",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
          }}
        >
          <SparkleIcon sx={{ fontSize: 16, color: "#fff" }} />
        </Box>
        <Typography
          sx={{
            fontFamily: "'Pretendard Variable', sans-serif",
            fontSize: "0.8rem",
            fontWeight: 700,
            color: "rgba(255,255,255,0.6)",
            letterSpacing: "0.08em",
            textTransform: "uppercase",
          }}
        >
          AI Platform
        </Typography>
      </Box>

      {/* Main hero content */}
      <Box sx={{ position: "relative", zIndex: 1, maxWidth: 440 }}>
        {/* Icon badge */}
        <Box
          sx={{
            display: "inline-flex",
            alignItems: "center",
            justifyContent: "center",
            width: 72,
            height: 72,
            borderRadius: "20px",
            background:
              "linear-gradient(135deg, rgba(16,163,127,0.25) 0%, rgba(16,163,127,0.08) 100%)",
            border: "1px solid rgba(16,163,127,0.35)",
            mb: 3.5,
            opacity: 0,
            animation: "loginFadeInUp 0.6s 0.2s ease forwards",
          }}
        >
          <SparkleIcon
            sx={{
              fontSize: 38,
              color: "#10a37f",
              filter: "drop-shadow(0 0 12px rgba(16,163,127,0.6))",
            }}
          />
        </Box>

        <Typography
          sx={{
            fontFamily: "'Pretendard Variable', sans-serif",
            fontSize: "clamp(2.2rem, 3.5vw, 3rem)",
            fontWeight: 800,
            color: "#ffffff",
            lineHeight: 1.1,
            letterSpacing: "-0.02em",
            mb: 1,
            opacity: 0,
            animation: "loginFadeInUp 0.6s 0.35s ease forwards",
          }}
        >
          AI 어시스턴트
        </Typography>

        <Typography
          sx={{
            fontFamily: "'Pretendard Variable', sans-serif",
            fontSize: "1.05rem",
            fontWeight: 400,
            color: "rgba(255,255,255,0.55)",
            mb: 5,
            opacity: 0,
            animation: "loginFadeInUp 0.6s 0.5s ease forwards",
          }}
        >
          생성형 AI 플랫폼
        </Typography>

        {/* Divider line */}
        <Box
          sx={{
            width: 48,
            height: 2,
            bgcolor: "#10a37f",
            borderRadius: 1,
            mb: 4,
            opacity: 0,
            animation: "loginFadeInUp 0.6s 0.6s ease forwards",
          }}
        />

        {/* Feature highlights */}
        <Box
          sx={{
            display: "flex",
            flexDirection: "column",
            gap: 2,
          }}
        >
          <FeatureRow
            icon={<SearchIcon sx={{ fontSize: 18 }} />}
            label="지능형 문서 검색 및 분석"
            delay={0.72}
          />
          <FeatureRow
            icon={<ReportIcon sx={{ fontSize: 18 }} />}
            label="AI 기반 보고서 자동 생성"
            delay={0.88}
          />
          <FeatureRow
            icon={<LockIcon sx={{ fontSize: 18 }} />}
            label="안전한 기업 데이터 관리"
            delay={1.04}
          />
        </Box>
      </Box>

      {/* Bottom badge */}
      <Box
        sx={{
          position: "absolute",
          bottom: 28,
          left: 36,
          display: "flex",
          alignItems: "center",
          gap: 1,
          opacity: 0,
          animation: "loginFadeInUp 0.6s 1.2s ease forwards",
        }}
      >
        <Box
          sx={{
            px: 1.5,
            py: 0.5,
            borderRadius: "20px",
            bgcolor: "rgba(16,163,127,0.12)",
            border: "1px solid rgba(16,163,127,0.2)",
          }}
        >
          <Typography
            sx={{
              fontSize: "0.7rem",
              color: "#10a37f",
              fontFamily: "'Pretendard Variable', sans-serif",
              fontWeight: 600,
              letterSpacing: "0.05em",
            }}
          >
            v0.1.0 — Secure Enterprise AI
          </Typography>
        </Box>
      </Box>
    </Box>
  );

  // ── Right login form ─────────────────────────────────────────────────────

  const LoginForm = (
    <Box
      sx={{
        flex: 1,
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        px: { xs: 3, sm: 6, md: 5, lg: 8 },
        py: 6,
        bgcolor: "background.paper",
        position: "relative",
        animation: "loginFadeInRight 0.55s 0.1s ease both",
      }}
    >
      {/* Mobile-only wordmark */}
      {!isDesktop && (
        <Box
          sx={{
            display: "flex",
            alignItems: "center",
            gap: 1.5,
            mb: 5,
          }}
        >
          <Box
            sx={{
              width: 40,
              height: 40,
              borderRadius: "11px",
              bgcolor: "primary.main",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
            }}
          >
            <SparkleIcon sx={{ fontSize: 22, color: "#fff" }} />
          </Box>
          <Box>
            <Typography
              sx={{
                fontFamily: "'Pretendard Variable', sans-serif",
                fontWeight: 800,
                fontSize: "1.25rem",
                lineHeight: 1,
                color: "text.primary",
              }}
            >
              AI 어시스턴트
            </Typography>
            <Typography
              sx={{
                fontSize: "0.72rem",
                color: "text.secondary",
                fontFamily: "'Pretendard Variable', sans-serif",
              }}
            >
              생성형 AI 플랫폼
            </Typography>
          </Box>
        </Box>
      )}

      <Box sx={{ width: "100%", maxWidth: 360 }}>
        {/* Heading */}
        <Box sx={{ mb: 4 }}>
          <Typography
            sx={{
              fontFamily: "'Pretendard Variable', sans-serif",
              fontSize: "1.75rem",
              fontWeight: 700,
              color: "text.primary",
              lineHeight: 1.2,
              mb: 0.75,
            }}
          >
            로그인
          </Typography>
          <Typography
            sx={{
              fontSize: "0.875rem",
              color: "text.secondary",
              fontFamily: "'Pretendard Variable', sans-serif",
            }}
          >
            계정 정보를 입력하여 시작하세요
          </Typography>
        </Box>

        {/* Error */}
        <Box
          sx={{
            overflow: "hidden",
            mb: error ? 2 : 0,
            animation: shaking ? "loginShake 0.55s ease" : "none",
          }}
        >
          {error && (
            <Box
              role="alert"
              aria-live="assertive"
              sx={{
                px: 2,
                py: 1.25,
                borderRadius: 2,
                bgcolor: "error.main",
                background:
                  "linear-gradient(135deg, rgba(211,47,47,0.1) 0%, rgba(211,47,47,0.06) 100%)",
                border: "1px solid",
                borderColor: "error.main",
                borderOpacity: 0.4,
                display: "flex",
                alignItems: "flex-start",
                gap: 1,
              }}
            >
              <Box
                sx={{
                  mt: 0.15,
                  width: 6,
                  height: 6,
                  borderRadius: "50%",
                  bgcolor: "error.main",
                  flexShrink: 0,
                }}
              />
              <Typography
                sx={{
                  fontSize: "0.82rem",
                  color: "error.main",
                  fontFamily: "'Pretendard Variable', sans-serif",
                  lineHeight: 1.4,
                }}
              >
                {error}
              </Typography>
            </Box>
          )}
        </Box>

        {/* Form */}
        <Box component="form" onSubmit={handleSubmit} noValidate>
          {/* Username */}
          <TextField
            fullWidth
            label="사용자명"
            placeholder="사용자명"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            disabled={loading}
            autoFocus
            inputProps={{ "aria-label": "사용자명" }}
            sx={{
              mb: 2,
              "& .MuiOutlinedInput-root": {
                borderRadius: "10px",
                fontSize: "0.9rem",
                fontFamily: "'Pretendard Variable', sans-serif",
                transition: "box-shadow 0.2s",
                "&.Mui-focused": {
                  boxShadow: "0 0 0 3px rgba(16,163,127,0.15)",
                },
                "&.Mui-focused .MuiOutlinedInput-notchedOutline": {
                  borderColor: "primary.main",
                  borderWidth: "1.5px",
                },
              },
              "& .MuiInputLabel-root.Mui-focused": {
                color: "primary.main",
              },
            }}
            slotProps={{
              inputLabel: { shrink: true },
              input: {
                startAdornment: (
                  <InputAdornment position="start">
                    <PersonIcon
                      sx={{ fontSize: 18, color: "text.disabled" }}
                    />
                  </InputAdornment>
                ),
              },
            }}
          />

          {/* Password */}
          <TextField
            fullWidth
            label="비밀번호"
            placeholder="비밀번호"
            type={showPassword ? "text" : "password"}
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            disabled={loading}
            inputProps={{ "aria-label": "비밀번호" }}
            sx={{
              mb: 1.5,
              "& .MuiOutlinedInput-root": {
                borderRadius: "10px",
                fontSize: "0.9rem",
                fontFamily: "'Pretendard Variable', sans-serif",
                transition: "box-shadow 0.2s",
                "&.Mui-focused": {
                  boxShadow: "0 0 0 3px rgba(16,163,127,0.15)",
                },
                "&.Mui-focused .MuiOutlinedInput-notchedOutline": {
                  borderColor: "primary.main",
                  borderWidth: "1.5px",
                },
              },
              "& .MuiInputLabel-root.Mui-focused": {
                color: "primary.main",
              },
            }}
            slotProps={{
              inputLabel: { shrink: true },
              input: {
                startAdornment: (
                  <InputAdornment position="start">
                    <KeyIcon sx={{ fontSize: 18, color: "text.disabled" }} />
                  </InputAdornment>
                ),
                endAdornment: (
                  <InputAdornment position="end">
                    <IconButton
                      onClick={() => setShowPassword(!showPassword)}
                      edge="end"
                      size="small"
                      tabIndex={-1}
                      aria-label={showPassword ? "비밀번호 숨기기" : "비밀번호 표시"}
                      sx={{ color: "text.disabled", "&:hover": { color: "text.secondary" } }}
                    >
                      {showPassword ? (
                        <VisibilityOff sx={{ fontSize: 18 }} />
                      ) : (
                        <Visibility sx={{ fontSize: 18 }} />
                      )}
                    </IconButton>
                  </InputAdornment>
                ),
              },
            }}
          />

          {/* Remember me */}
          <FormControlLabel
            control={
              <Checkbox
                checked={rememberMe}
                onChange={(e) => setRememberMe(e.target.checked)}
                size="small"
                sx={{
                  color: "text.disabled",
                  "&.Mui-checked": { color: "primary.main" },
                  p: 0.5,
                }}
              />
            }
            label={
              <Typography
                sx={{
                  fontSize: "0.82rem",
                  color: "text.secondary",
                  fontFamily: "'Pretendard Variable', sans-serif",
                }}
              >
                사용자명 기억하기
              </Typography>
            }
            sx={{ mb: 2.5, ml: 0 }}
          />

          {/* Submit */}
          <Button
            type="submit"
            fullWidth
            variant="contained"
            size="large"
            disabled={loading}
            disableElevation
            sx={{
              py: 1.4,
              fontWeight: 700,
              fontSize: "0.92rem",
              fontFamily: "'Pretendard Variable', sans-serif",
              letterSpacing: "0.02em",
              borderRadius: "10px",
              bgcolor: "primary.main",
              transition: "all 0.2s ease",
              boxShadow: "0 1px 2px rgba(0,0,0,0.08)",
              "&:hover": {
                bgcolor: "primary.dark",
                boxShadow:
                  "0 4px 16px rgba(16,163,127,0.35), 0 2px 6px rgba(0,0,0,0.12)",
                transform: "translateY(-1px)",
              },
              "&:active": {
                transform: "translateY(0)",
              },
              "&.Mui-disabled": {
                bgcolor: "action.disabledBackground",
              },
            }}
          >
            {loading ? (
              <CircularProgress size={22} sx={{ color: "#fff" }} />
            ) : (
              "로그인"
            )}
          </Button>
        </Box>

        {/* Test accounts */}
        <TestAccountsHint onSelect={handleTestSelect} />

        {/* Version */}
        <Typography
          sx={{
            display: "block",
            textAlign: "center",
            mt: 4,
            fontSize: "0.7rem",
            color: "text.disabled",
            fontFamily: "'Pretendard Variable', sans-serif",
            letterSpacing: "0.04em",
          }}
        >
          AI Platform — v1.0
        </Typography>
      </Box>
    </Box>
  );

  // ── Root layout ──────────────────────────────────────────────────────────

  return (
    <Box
      sx={{
        display: "flex",
        minHeight: "100vh",
        height: "100vh",
        overflow: "hidden",
      }}
    >
      {BrandingPanel}
      {LoginForm}
    </Box>
  );
}
