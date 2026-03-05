import { useState, useEffect, useCallback, useRef } from "react";
import { Box, Typography, Button, useTheme } from "@mui/material";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface SpotlightRegion {
  top: number;
  left: number;
  width: number;
  height: number;
  borderRadius?: number;
}

interface TourStep {
  id: string;
  title: string;
  description: string;
  spotlight: SpotlightRegion | null;
  popoverAnchor: {
    vertical: "top" | "bottom" | "center";
    horizontal: "left" | "right" | "center";
  };
}

// ---------------------------------------------------------------------------
// Layout constants — must match ChatPage.tsx / component sizing
// ---------------------------------------------------------------------------

const SIDEBAR_WIDTH = 260;
const TOPBAR_HEIGHT = 48;

// ---------------------------------------------------------------------------
// Step definitions
// ---------------------------------------------------------------------------

function buildSteps(vw: number, vh: number): TourStep[] {
  // Input bar: centered at bottom, max-width 768px, px=16 on md+
  const inputBarHPad = vw >= 600 ? 16 : 8;
  const inputBarWidth = Math.min(768, vw - inputBarHPad * 2);
  const inputBarLeft = (vw - inputBarWidth) / 2;
  // The input bar container has py: { xs: 1, sm: 2 } which is 8px or 16px
  const inputBarPy = vw >= 600 ? 16 : 8;
  // Approximate height: TextField + buttons ≈ 52px, container ≈ 52 + 2*py
  const inputBarHeight = 52 + inputBarPy * 2;
  const inputBarTop = vh - inputBarHeight - 28; // subtract keyboard hint text height

  // Sidebar: always at left, full height
  const sidebarSpotlight: SpotlightRegion = {
    top: 0,
    left: 0,
    width: SIDEBAR_WIDTH,
    height: vh,
    borderRadius: 0,
  };

  // Settings (gear icon): in topbar, right side
  // Topbar right icons: roughly last 48px from the right at top
  const settingsButtonRight = 48 * 2; // roughly 2nd from right
  const settingsSpotlight: SpotlightRegion = {
    top: 4,
    left: vw - settingsButtonRight - 36,
    width: 40,
    height: 40,
    borderRadius: 20,
  };

  // TopBar: full width, top
  const topbarSpotlight: SpotlightRegion = {
    top: 0,
    left: 0,
    width: vw,
    height: TOPBAR_HEIGHT,
    borderRadius: 0,
  };

  return [
    {
      id: "input",
      title: "채팅 입력",
      description: "질문을 입력하고 전송하세요. Enter로 전송, Shift+Enter로 줄바꿈합니다.",
      spotlight: {
        top: inputBarTop,
        left: inputBarLeft - 8,
        width: inputBarWidth + 16,
        height: inputBarHeight,
        borderRadius: 16,
      },
      popoverAnchor: { vertical: "top", horizontal: "center" },
    },
    {
      id: "mode",
      title: "응답 모드",
      description: "RAG 모드는 사내 문서를 검색하여 정확한 답변을 제공합니다. 설정 아이콘에서 응답 모드와 Temperature를 조절하세요.",
      spotlight: settingsSpotlight,
      popoverAnchor: { vertical: "bottom", horizontal: "right" },
    },
    {
      id: "sidebar",
      title: "사이드바",
      description: "대화 기록을 관리하고 검색할 수 있습니다. 이전 대화를 클릭하면 이어서 대화할 수 있습니다.",
      spotlight: sidebarSpotlight,
      popoverAnchor: { vertical: "center", horizontal: "right" },
    },
    {
      id: "settings",
      title: "AI 설정",
      description: "오른쪽 상단의 설정 버튼에서 Temperature와 AI 모델을 조정할 수 있습니다.",
      spotlight: topbarSpotlight,
      popoverAnchor: { vertical: "bottom", horizontal: "right" },
    },
    {
      id: "sources",
      title: "출처 확인",
      description: "AI 답변 아래에서 참조한 출처 문서를 확인하세요. 문서명과 관련도 점수가 표시됩니다.",
      spotlight: null,
      popoverAnchor: { vertical: "center", horizontal: "center" },
    },
    {
      id: "help",
      title: "환영합니다!",
      description: "AI 어시스턴트 사용이 준비되었습니다. 궁금한 점은 언제든지 질문하세요. 여러분의 업무를 함께 돕겠습니다.",
      spotlight: null,
      popoverAnchor: { vertical: "center", horizontal: "center" },
    },
  ];
}

// ---------------------------------------------------------------------------
// Popover positioning
// ---------------------------------------------------------------------------

const POPOVER_W = 320;
const POPOVER_H = 200; // approx, actual is auto
const POPOVER_GAP = 16;

function computePopoverStyle(
  step: TourStep,
  vw: number,
  vh: number,
): React.CSSProperties {
  const { spotlight, popoverAnchor } = step;

  if (!spotlight) {
    // Centered overlay
    return {
      position: "fixed",
      top: "50%",
      left: "50%",
      transform: "translate(-50%, -50%)",
      width: POPOVER_W,
      zIndex: 10001,
    };
  }

  const { vertical, horizontal } = popoverAnchor;
  let top = 0;
  let left = 0;

  // Vertical placement
  if (vertical === "top") {
    top = spotlight.top - POPOVER_H - POPOVER_GAP;
    if (top < POPOVER_GAP) top = spotlight.top + spotlight.height + POPOVER_GAP;
  } else if (vertical === "bottom") {
    top = spotlight.top + spotlight.height + POPOVER_GAP;
    if (top + POPOVER_H > vh - POPOVER_GAP) {
      top = spotlight.top - POPOVER_H - POPOVER_GAP;
    }
  } else {
    // center
    top = spotlight.top + spotlight.height / 2 - POPOVER_H / 2;
  }

  // Horizontal placement
  if (horizontal === "right") {
    left = spotlight.left + spotlight.width + POPOVER_GAP;
    if (left + POPOVER_W > vw - POPOVER_GAP) {
      left = spotlight.left - POPOVER_W - POPOVER_GAP;
    }
  } else if (horizontal === "left") {
    left = spotlight.left - POPOVER_W - POPOVER_GAP;
    if (left < POPOVER_GAP) {
      left = spotlight.left + spotlight.width + POPOVER_GAP;
    }
  } else {
    // center
    left = spotlight.left + spotlight.width / 2 - POPOVER_W / 2;
  }

  // Clamp to viewport
  left = Math.max(POPOVER_GAP, Math.min(left, vw - POPOVER_W - POPOVER_GAP));
  top = Math.max(POPOVER_GAP, Math.min(top, vh - POPOVER_GAP - 10));

  return {
    position: "fixed",
    top,
    left,
    width: POPOVER_W,
    zIndex: 10001,
  };
}

// ---------------------------------------------------------------------------
// Storage key
// ---------------------------------------------------------------------------

const STORAGE_KEY = "kogas-ai-onboarding-done";

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export function OnboardingTour() {
  const theme = useTheme();
  const isDark = theme.palette.mode === "dark";

  const [visible, setVisible] = useState(false);
  const [stepIndex, setStepIndex] = useState(0);
  const [dims, setDims] = useState({
    vw: window.innerWidth,
    vh: window.innerHeight,
  });
  // Animate in/out
  const [entering, setEntering] = useState(false);
  const enterTimeout = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Track viewport size
  useEffect(() => {
    const onResize = () =>
      setDims({ vw: window.innerWidth, vh: window.innerHeight });
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);

  // Check localStorage on mount
  useEffect(() => {
    const done = localStorage.getItem(STORAGE_KEY);
    if (!done) {
      // Small delay so the chat page renders first
      const t = setTimeout(() => {
        setVisible(true);
        setEntering(true);
        const tt = setTimeout(() => setEntering(false), 50);
        enterTimeout.current = tt;
      }, 800);
      return () => clearTimeout(t);
    }
  }, []);

  const dismiss = useCallback(() => {
    localStorage.setItem(STORAGE_KEY, "1");
    setVisible(false);
  }, []);

  const handleSkip = useCallback(() => {
    dismiss();
  }, [dismiss]);

  const handleNext = useCallback(() => {
    const steps = buildSteps(dims.vw, dims.vh);
    if (stepIndex < steps.length - 1) {
      setStepIndex((i) => i + 1);
    } else {
      dismiss();
    }
  }, [stepIndex, dims, dismiss]);

  const handleDotClick = useCallback((index: number) => {
    setStepIndex(index);
  }, []);

  // Cleanup timeout on unmount
  useEffect(() => {
    return () => {
      if (enterTimeout.current) clearTimeout(enterTimeout.current);
    };
  }, []);

  if (!visible) return null;

  const steps = buildSteps(dims.vw, dims.vh);
  const step = steps[stepIndex];
  const isLast = stepIndex === steps.length - 1;
  const popoverStyle = computePopoverStyle(step, dims.vw, dims.vh);

  // Overlay color
  const overlayColor = isDark
    ? "rgba(0, 0, 0, 0.72)"
    : "rgba(0, 0, 0, 0.60)";

  // Popover colors
  const popoverBg = isDark ? "#1e1e1e" : "#ffffff";
  const popoverBorder = isDark ? "rgba(255,255,255,0.08)" : "rgba(0,0,0,0.10)";
  const titleColor = isDark ? "#ececec" : "#1a1a2e";
  const descColor = isDark ? "#b4b4b4" : "#4a4a5a";
  const dotActive = "#10a37f";
  const dotInactive = isDark ? "rgba(255,255,255,0.18)" : "rgba(0,0,0,0.15)";
  const skipColor = isDark ? "#8e8ea0" : "#9090a0";
  const accentGreen = "#10a37f";

  return (
    <>
      {/* ── Backdrop ─────────────────────────────────────────────────────── */}
      <Box
        onClick={handleSkip}
        sx={{
          position: "fixed",
          inset: 0,
          zIndex: 9999,
          pointerEvents: "all",
          opacity: entering ? 0 : 1,
          transition: "opacity 0.35s ease",
        }}
      >
        {step.spotlight ? (
          // SVG-based cutout overlay for pixel-perfect spotlight
          <svg
            width={dims.vw}
            height={dims.vh}
            style={{ display: "block", position: "absolute", inset: 0 }}
          >
            <defs>
              <mask id="spotlight-mask">
                {/* White = show overlay, black = cut out (reveal UI) */}
                <rect width={dims.vw} height={dims.vh} fill="white" />
                <rect
                  x={step.spotlight.left}
                  y={step.spotlight.top}
                  width={step.spotlight.width}
                  height={step.spotlight.height}
                  rx={step.spotlight.borderRadius ?? 12}
                  ry={step.spotlight.borderRadius ?? 12}
                  fill="black"
                />
              </mask>
            </defs>
            <rect
              width={dims.vw}
              height={dims.vh}
              fill={overlayColor}
              mask="url(#spotlight-mask)"
            />
          </svg>
        ) : (
          // Full overlay with no cutout
          <Box
            sx={{
              position: "absolute",
              inset: 0,
              bgcolor: overlayColor,
            }}
          />
        )}

        {/* Spotlight highlight ring */}
        {step.spotlight && (
          <Box
            sx={{
              position: "absolute",
              top: step.spotlight.top - 3,
              left: step.spotlight.left - 3,
              width: step.spotlight.width + 6,
              height: step.spotlight.height + 6,
              borderRadius: (step.spotlight.borderRadius ?? 12) + 3,
              border: `2px solid ${accentGreen}`,
              boxShadow: `0 0 0 1px ${accentGreen}40, 0 0 24px ${accentGreen}60`,
              pointerEvents: "none",
              "@keyframes pulseRing": {
                "0%, 100%": { opacity: 1, transform: "scale(1)" },
                "50%": { opacity: 0.7, transform: "scale(1.008)" },
              },
              animation: "pulseRing 2.2s ease-in-out infinite",
            }}
          />
        )}
      </Box>

      {/* ── Popover card ─────────────────────────────────────────────────── */}
      <Box
        onClick={(e) => e.stopPropagation()}
        sx={{
          ...popoverStyle,
          bgcolor: popoverBg,
          border: `1px solid ${popoverBorder}`,
          borderRadius: "16px",
          boxShadow: isDark
            ? "0 24px 64px rgba(0,0,0,0.6), 0 4px 16px rgba(0,0,0,0.4)"
            : "0 24px 64px rgba(0,0,0,0.18), 0 4px 16px rgba(0,0,0,0.10)",
          overflow: "hidden",
          opacity: entering ? 0 : 1,
          transform: entering ? "translateY(6px)" : "translateY(0)",
          transition: "opacity 0.35s ease, transform 0.35s ease",
        }}
      >
        {/* Header accent stripe */}
        <Box
          sx={{
            height: 3,
            background: `linear-gradient(90deg, ${accentGreen}, #0d8a6b)`,
          }}
        />

        {/* Content */}
        <Box sx={{ p: "20px 24px 16px" }}>
          {/* Step badge + title */}
          <Box
            sx={{
              display: "flex",
              alignItems: "center",
              gap: 1.5,
              mb: 1.25,
            }}
          >
            <Box
              sx={{
                width: 28,
                height: 28,
                borderRadius: "50%",
                bgcolor: `${accentGreen}1a`,
                border: `1.5px solid ${accentGreen}40`,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                flexShrink: 0,
              }}
            >
              <Typography
                sx={{
                  fontSize: "0.7rem",
                  fontWeight: 700,
                  color: accentGreen,
                  lineHeight: 1,
                  letterSpacing: "-0.01em",
                }}
              >
                {stepIndex + 1}
              </Typography>
            </Box>
            <Typography
              sx={{
                fontSize: "0.95rem",
                fontWeight: 700,
                color: titleColor,
                letterSpacing: "-0.02em",
                lineHeight: 1.3,
              }}
            >
              {step.title}
            </Typography>
          </Box>

          {/* Description */}
          <Typography
            sx={{
              fontSize: "0.85rem",
              color: descColor,
              lineHeight: 1.65,
              mb: 2.25,
              pl: "43px", // align with title text
            }}
          >
            {step.description}
          </Typography>

          {/* Progress dots + buttons */}
          <Box
            sx={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
            }}
          >
            {/* Progress dots */}
            <Box sx={{ display: "flex", gap: 0.625, alignItems: "center" }}>
              {steps.map((_, i) => (
                <Box
                  key={i}
                  onClick={() => handleDotClick(i)}
                  sx={{
                    width: i === stepIndex ? 18 : 6,
                    height: 6,
                    borderRadius: 3,
                    bgcolor: i === stepIndex ? dotActive : dotInactive,
                    cursor: "pointer",
                    transition: "all 0.25s ease",
                    flexShrink: 0,
                    "&:hover": {
                      bgcolor: i === stepIndex ? dotActive : isDark ? "rgba(255,255,255,0.35)" : "rgba(0,0,0,0.3)",
                    },
                  }}
                />
              ))}
            </Box>

            {/* Action buttons */}
            <Box sx={{ display: "flex", gap: 1, alignItems: "center" }}>
              {!isLast && (
                <Button
                  size="small"
                  onClick={handleSkip}
                  sx={{
                    color: skipColor,
                    fontSize: "0.78rem",
                    fontWeight: 500,
                    px: 1,
                    minWidth: 0,
                    textTransform: "none",
                    letterSpacing: 0,
                    "&:hover": {
                      bgcolor: isDark ? "rgba(255,255,255,0.06)" : "rgba(0,0,0,0.04)",
                      color: descColor,
                    },
                  }}
                >
                  건너뛰기
                </Button>
              )}
              <Button
                size="small"
                variant="contained"
                onClick={handleNext}
                disableElevation
                sx={{
                  bgcolor: accentGreen,
                  color: "#ffffff",
                  fontSize: "0.82rem",
                  fontWeight: 600,
                  px: 2,
                  py: 0.6,
                  minWidth: isLast ? 80 : 60,
                  borderRadius: "8px",
                  textTransform: "none",
                  letterSpacing: "-0.01em",
                  "&:hover": {
                    bgcolor: "#0d8a6b",
                  },
                }}
              >
                {isLast ? "시작하기" : "다음"}
              </Button>
            </Box>
          </Box>
        </Box>
      </Box>
    </>
  );
}

// ---------------------------------------------------------------------------
// Hook to manually re-trigger tour (for help button / testing)
// ---------------------------------------------------------------------------

export function resetOnboardingTour() {
  localStorage.removeItem(STORAGE_KEY);
}
