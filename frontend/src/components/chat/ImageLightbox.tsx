import { useEffect, useCallback, useState, useRef } from "react";
import {
  Dialog,
  DialogContent,
  Box,
  IconButton,
  Typography,
  Fade,
  Tooltip,
} from "@mui/material";
import {
  Close as CloseIcon,
  ZoomIn as ZoomInIcon,
  ZoomOut as ZoomOutIcon,
  RestartAlt as ResetIcon,
} from "@mui/icons-material";

interface ImageLightboxProps {
  open: boolean;
  src: string;
  alt?: string;
  filename?: string;
  onClose: () => void;
}

const MIN_ZOOM = 0.25;
const MAX_ZOOM = 5;
const ZOOM_STEP = 0.25;

export function ImageLightbox({
  open,
  src,
  alt,
  filename,
  onClose,
}: ImageLightboxProps) {
  const [zoom, setZoom] = useState(1);
  const [dragging, setDragging] = useState(false);
  const [translate, setTranslate] = useState({ x: 0, y: 0 });
  const dragStart = useRef<{ mx: number; my: number; tx: number; ty: number } | null>(null);
  const imgRef = useRef<HTMLImageElement>(null);

  // Reset zoom/pan when dialog opens
  useEffect(() => {
    if (open) {
      setZoom(1);
      setTranslate({ x: 0, y: 0 });
    }
  }, [open]);

  // ESC key handler
  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      if (!open) return;
      if (e.key === "Escape") {
        onClose();
      } else if (e.key === "+" || e.key === "=") {
        setZoom((z) => Math.min(MAX_ZOOM, +(z + ZOOM_STEP).toFixed(2)));
      } else if (e.key === "-") {
        setZoom((z) => Math.max(MIN_ZOOM, +(z - ZOOM_STEP).toFixed(2)));
      } else if (e.key === "0") {
        setZoom(1);
        setTranslate({ x: 0, y: 0 });
      }
    },
    [open, onClose]
  );

  useEffect(() => {
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [handleKeyDown]);

  // Scroll-wheel zoom
  const handleWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault();
    const delta = e.deltaY < 0 ? ZOOM_STEP : -ZOOM_STEP;
    setZoom((z) => {
      const next = +(z + delta).toFixed(2);
      return Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, next));
    });
  }, []);

  // Drag-to-pan handlers
  const handleMouseDown = useCallback(
    (e: React.MouseEvent) => {
      if (zoom <= 1) return;
      e.preventDefault();
      setDragging(true);
      dragStart.current = {
        mx: e.clientX,
        my: e.clientY,
        tx: translate.x,
        ty: translate.y,
      };
    },
    [zoom, translate]
  );

  const handleMouseMove = useCallback(
    (e: React.MouseEvent) => {
      if (!dragging || !dragStart.current) return;
      const dx = e.clientX - dragStart.current.mx;
      const dy = e.clientY - dragStart.current.my;
      setTranslate({
        x: dragStart.current.tx + dx,
        y: dragStart.current.ty + dy,
      });
    },
    [dragging]
  );

  const handleMouseUp = useCallback(() => {
    setDragging(false);
    dragStart.current = null;
  }, []);

  const handleZoomIn = () =>
    setZoom((z) => Math.min(MAX_ZOOM, +(z + ZOOM_STEP).toFixed(2)));
  const handleZoomOut = () =>
    setZoom((z) => Math.max(MIN_ZOOM, +(z - ZOOM_STEP).toFixed(2)));
  const handleReset = () => {
    setZoom(1);
    setTranslate({ x: 0, y: 0 });
  };

  const displayName = filename || alt || src.split("/").pop() || "이미지";

  return (
    <Dialog
      open={open}
      onClose={onClose}
      maxWidth={false}
      fullWidth
      TransitionComponent={Fade}
      TransitionProps={{ timeout: 220 }}
      aria-label="이미지 뷰어"
      sx={{
        "& .MuiDialog-paper": {
          bgcolor: "transparent",
          boxShadow: "none",
          m: 0,
          maxHeight: "100vh",
          maxWidth: "100vw",
          width: "100vw",
          height: "100vh",
          borderRadius: 0,
          overflow: "hidden",
        },
        "& .MuiBackdrop-root": {
          bgcolor: "rgba(0,0,0,0.92)",
          backdropFilter: "blur(4px)",
        },
      }}
    >
      <DialogContent
        sx={{
          p: 0,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          width: "100%",
          height: "100%",
          position: "relative",
          overflow: "hidden",
          "&:focus": { outline: "none" },
        }}
        onWheel={handleWheel}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
      >
        {/* Close button — top right */}
        <Tooltip title="닫기 (Esc)" placement="left">
          <IconButton
            onClick={onClose}
            aria-label="닫기"
            sx={{
              position: "absolute",
              top: 16,
              right: 16,
              zIndex: 10,
              bgcolor: "rgba(255,255,255,0.12)",
              color: "#fff",
              backdropFilter: "blur(8px)",
              border: "1px solid rgba(255,255,255,0.18)",
              width: 40,
              height: 40,
              transition: "background-color 0.15s, transform 0.15s",
              "&:hover": {
                bgcolor: "rgba(255,255,255,0.22)",
                transform: "scale(1.08)",
              },
            }}
          >
            <CloseIcon sx={{ fontSize: 20 }} />
          </IconButton>
        </Tooltip>

        {/* Image container */}
        <Box
          onMouseDown={handleMouseDown}
          sx={{
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            width: "100%",
            height: "100%",
            cursor: zoom > 1 ? (dragging ? "grabbing" : "grab") : "default",
            userSelect: "none",
          }}
        >
          <Box
            component="img"
            ref={imgRef}
            src={src}
            alt={alt || displayName}
            draggable={false}
            sx={{
              maxWidth: zoom === 1 ? "90vw" : "none",
              maxHeight: zoom === 1 ? "85vh" : "none",
              width: zoom === 1 ? "auto" : "auto",
              height: zoom === 1 ? "auto" : "auto",
              transform: `translate(${translate.x}px, ${translate.y}px) scale(${zoom})`,
              transformOrigin: "center center",
              transition: dragging ? "none" : "transform 0.18s ease",
              borderRadius: zoom === 1 ? "6px" : 0,
              boxShadow:
                zoom === 1
                  ? "0 32px 80px rgba(0,0,0,0.6), 0 0 0 1px rgba(255,255,255,0.06)"
                  : "none",
              pointerEvents: "none",
              display: "block",
              objectFit: "contain",
            }}
            onError={(e) => {
              (e.currentTarget as HTMLImageElement).style.display = "none";
            }}
          />
        </Box>

        {/* Bottom bar — filename + zoom controls */}
        <Box
          sx={{
            position: "absolute",
            bottom: 0,
            left: 0,
            right: 0,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            gap: 1,
            py: 1.75,
            px: 3,
            background:
              "linear-gradient(to top, rgba(0,0,0,0.7) 0%, transparent 100%)",
            backdropFilter: "blur(0px)",
          }}
        >
          {/* Filename */}
          <Typography
            variant="caption"
            sx={{
              color: "rgba(255,255,255,0.65)",
              fontSize: "0.78rem",
              letterSpacing: "0.02em",
              maxWidth: 320,
              overflow: "hidden",
              textOverflow: "ellipsis",
              whiteSpace: "nowrap",
              mr: 2,
            }}
          >
            {displayName}
          </Typography>

          {/* Zoom controls */}
          <Box
            sx={{
              display: "flex",
              alignItems: "center",
              gap: 0.5,
              bgcolor: "rgba(255,255,255,0.1)",
              borderRadius: "24px",
              border: "1px solid rgba(255,255,255,0.14)",
              px: 1,
              py: 0.25,
              backdropFilter: "blur(12px)",
            }}
          >
            <Tooltip title="축소 (-)">
              <span>
                <IconButton
                  size="small"
                  onClick={handleZoomOut}
                  disabled={zoom <= MIN_ZOOM}
                  aria-label="축소"
                  sx={{
                    color: zoom <= MIN_ZOOM ? "rgba(255,255,255,0.3)" : "#fff",
                    p: 0.5,
                    "&:hover": { bgcolor: "rgba(255,255,255,0.12)" },
                  }}
                >
                  <ZoomOutIcon sx={{ fontSize: 18 }} />
                </IconButton>
              </span>
            </Tooltip>

            {/* Zoom level label — click to reset */}
            <Tooltip title="원래 크기로 (0)">
              <Typography
                component="button"
                onClick={handleReset}
                sx={{
                  color: "#fff",
                  fontSize: "0.75rem",
                  fontWeight: 600,
                  letterSpacing: "0.03em",
                  minWidth: 44,
                  textAlign: "center",
                  cursor: "pointer",
                  bgcolor: "transparent",
                  border: "none",
                  fontFamily: "inherit",
                  lineHeight: 1,
                  py: 0.5,
                  px: 0.25,
                  borderRadius: 1,
                  transition: "background-color 0.12s",
                  "&:hover": { bgcolor: "rgba(255,255,255,0.1)" },
                }}
                aria-label="줌 초기화"
              >
                {Math.round(zoom * 100)}%
              </Typography>
            </Tooltip>

            <Tooltip title="확대 (+)">
              <span>
                <IconButton
                  size="small"
                  onClick={handleZoomIn}
                  disabled={zoom >= MAX_ZOOM}
                  aria-label="확대"
                  sx={{
                    color: zoom >= MAX_ZOOM ? "rgba(255,255,255,0.3)" : "#fff",
                    p: 0.5,
                    "&:hover": { bgcolor: "rgba(255,255,255,0.12)" },
                  }}
                >
                  <ZoomInIcon sx={{ fontSize: 18 }} />
                </IconButton>
              </span>
            </Tooltip>

            <Tooltip title="초기화 (0)">
              <span>
                <IconButton
                  size="small"
                  onClick={handleReset}
                  disabled={zoom === 1 && translate.x === 0 && translate.y === 0}
                  aria-label="초기화"
                  sx={{
                    color:
                      zoom === 1 && translate.x === 0 && translate.y === 0
                        ? "rgba(255,255,255,0.3)"
                        : "#fff",
                    p: 0.5,
                    "&:hover": { bgcolor: "rgba(255,255,255,0.12)" },
                  }}
                >
                  <ResetIcon sx={{ fontSize: 18 }} />
                </IconButton>
              </span>
            </Tooltip>
          </Box>
        </Box>
      </DialogContent>
    </Dialog>
  );
}
