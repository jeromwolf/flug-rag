import { useState, useMemo } from "react";
import { Box, Typography, Chip, IconButton, Tooltip } from "@mui/material";
import {
  PictureAsPdf,
  Description,
  Article,
  InsertDriveFile,
  TableChart,
  Slideshow,
  TextSnippet,
  ExpandMore,
  ExpandLess,
  Star,
  StarHalf,
  StarBorder,
  ImageSearch as ImageSearchIcon,
} from "@mui/icons-material";
import type { Source } from "../../types";
import { API_BASE } from "../../api/client";
import { ImageLightbox } from "./ImageLightbox";

// Korean particles and stop-words to filter out when extracting keywords
const KOREAN_STOP_WORDS = new Set([
  "은", "는", "이", "가", "을", "를", "에", "의", "로", "와", "과",
  "도", "만", "에서", "으로", "까지", "부터", "에게", "한테", "께",
  "이나", "나", "든지", "라도", "이라도", "하고", "이고", "고", "며",
  "이며", "면서", "이면서", "지만", "이지만", "그리고", "그러나",
  "하지만", "또는", "혹은", "및", "또", "대해", "대한", "위해",
  "위한", "통해", "통한", "관한", "관해", "대하여", "있는", "있어",
  "있을", "없는", "없어", "없을", "하는", "한다", "합니다", "했",
  "했을", "할", "하여", "해서", "되는", "된다", "됩니다", "되었",
  "무엇", "어떤", "어느", "누가", "누구", "언제", "어디", "왜", "어떻게",
  "의해", "것", "수", "등", "및", "즉", "각", "전", "후", "중",
]);

/**
 * Extract meaningful keywords from a query string.
 * Splits on whitespace and punctuation, filters short tokens and stop-words.
 */
function extractKeywords(query: string): string[] {
  const tokens = query
    .split(/[\s,?.!;:()\[\]{}"'\/\\·\-]+/)
    .map((t) => t.trim())
    .filter((t) => {
      if (t.length < 2) return false;
      if (KOREAN_STOP_WORDS.has(t)) return false;
      if (/^\d{1,2}$/.test(t)) return false;
      return true;
    });
  return [...new Set(tokens)];
}

/**
 * Highlight all keyword occurrences in `text` by wrapping them in <mark> spans.
 */
function highlightKeywords(
  text: string,
  keywords: string[]
): React.ReactNode[] {
  if (!keywords.length) return [text];

  const escaped = keywords.map((kw) =>
    kw.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")
  );
  const pattern = new RegExp(`(${escaped.join("|")})`, "gi");

  const parts = text.split(pattern);
  return parts.map((part, i) => {
    if (i % 2 === 1) {
      return (
        <mark
          key={i}
          style={{
            background: "rgba(16,163,127,0.2)",
            fontWeight: 600,
            borderRadius: "2px",
            padding: "0 2px",
            color: "inherit",
          }}
        >
          {part}
        </mark>
      );
    }
    return part;
  });
}

// Image extensions that can be previewed in the lightbox
const IMAGE_EXTENSIONS = new Set([
  "png", "jpg", "jpeg", "gif", "svg", "webp", "bmp", "ico", "avif",
]);

function isImageFile(filename: string): boolean {
  const ext = filename.split(".").pop()?.toLowerCase() ?? "";
  return IMAGE_EXTENSIONS.has(ext);
}

/** File type metadata: icon component, color, label */
interface FileTypeMeta {
  icon: React.ReactNode;
  color: string;
}

function getFileTypeMeta(filename: string): FileTypeMeta {
  const ext = filename.split(".").pop()?.toLowerCase();
  const iconSx = { fontSize: 16 };
  switch (ext) {
    case "pdf":
      return {
        icon: <PictureAsPdf sx={{ ...iconSx, color: "#e53935" }} />,
        color: "#e53935",
      };
    case "hwp":
      return {
        icon: <Description sx={{ ...iconSx, color: "#1e88e5" }} />,
        color: "#1e88e5",
      };
    case "xlsx":
    case "xls":
      return {
        icon: <TableChart sx={{ ...iconSx, color: "#43a047" }} />,
        color: "#43a047",
      };
    case "docx":
    case "doc":
      return {
        icon: <Article sx={{ ...iconSx, color: "#1565c0" }} />,
        color: "#1565c0",
      };
    case "pptx":
    case "ppt":
      return {
        icon: <Slideshow sx={{ ...iconSx, color: "#ef6c00" }} />,
        color: "#ef6c00",
      };
    case "txt":
      return {
        icon: <TextSnippet sx={{ ...iconSx, color: "#757575" }} />,
        color: "#9e9e9e",
      };
    case "png":
    case "jpg":
    case "jpeg":
    case "gif":
    case "svg":
    case "webp":
    case "bmp":
    case "avif":
      return {
        icon: <ImageSearchIcon sx={{ ...iconSx, color: "#8e24aa" }} />,
        color: "#8e24aa",
      };
    default:
      return {
        icon: <InsertDriveFile sx={{ ...iconSx, color: "#757575" }} />,
        color: "#9e9e9e",
      };
  }
}

/** Star rating display for score (0.0 – 1.0 → 0–5 stars) */
function ScoreStars({ score }: { score: number }) {
  const stars = Math.round(score * 5 * 2) / 2; // round to nearest 0.5
  const full = Math.floor(stars);
  const half = stars % 1 >= 0.5 ? 1 : 0;
  const empty = 5 - full - half;

  const starColor =
    score > 0.8 ? "#43a047" : score > 0.5 ? "#ffa726" : "#bdbdbd";
  const starSx = { fontSize: 12, color: starColor };

  return (
    <Tooltip title={`관련도 ${Math.round(score * 100)}%`} placement="top">
      <Box sx={{ display: "flex", alignItems: "center", gap: 0 }}>
        {Array.from({ length: full }).map((_, i) => (
          <Star key={`f-${i}`} sx={starSx} />
        ))}
        {half === 1 && <StarHalf sx={starSx} />}
        {Array.from({ length: empty }).map((_, i) => (
          <StarBorder key={`e-${i}`} sx={starSx} />
        ))}
      </Box>
    </Tooltip>
  );
}

/** Score bar visualization */
function ScoreBar({ score, color }: { score: number; color: string }) {
  return (
    <Box
      sx={{
        height: 3,
        borderRadius: 2,
        bgcolor: "action.hover",
        overflow: "hidden",
        mt: 0.75,
        mb: 0.5,
      }}
    >
      <Box
        sx={{
          height: "100%",
          width: `${Math.round(score * 100)}%`,
          bgcolor: color,
          borderRadius: 2,
          transition: "width 0.4s ease",
        }}
      />
    </Box>
  );
}

/** Renders source content with keyword highlights, collapsed by default */
function HighlightedContent({
  content,
  keywords,
  expanded,
}: {
  content: string;
  keywords: string[];
  expanded: boolean;
}) {
  const highlighted = useMemo(
    () => highlightKeywords(content, keywords),
    [content, keywords]
  );

  return (
    <Typography
      variant="body2"
      component="div"
      sx={{
        mt: 0.5,
        fontSize: "0.78rem",
        color: "text.secondary",
        lineHeight: 1.5,
        whiteSpace: "pre-wrap",
        animation: "fadeInUp 0.2s ease-out",
        ...(expanded
          ? {}
          : {
              display: "-webkit-box",
              WebkitLineClamp: 3,
              WebkitBoxOrient: "vertical",
              overflow: "hidden",
            }),
      }}
    >
      {highlighted}
    </Typography>
  );
}

interface LightboxState {
  open: boolean;
  src: string;
  filename: string;
}

export function SourcesPanel({
  sources,
  query,
}: {
  sources: Source[];
  query?: string;
}) {
  const [expandedIdx, setExpandedIdx] = useState<Set<number>>(new Set());
  const [lightbox, setLightbox] = useState<LightboxState>({
    open: false,
    src: "",
    filename: "",
  });

  const keywords = useMemo(
    () => (query ? extractKeywords(query) : []),
    [query]
  );

  if (!sources || sources.length === 0) return null;

  const toggleExpand = (idx: number) => {
    setExpandedIdx((prev) => {
      const next = new Set(prev);
      if (next.has(idx)) {
        next.delete(idx);
      } else {
        next.add(idx);
      }
      return next;
    });
  };

  const openLightbox = (src: string, filename: string) => {
    setLightbox({ open: true, src, filename });
  };

  const closeLightbox = () => {
    setLightbox((prev) => ({ ...prev, open: false }));
  };

  return (
    <>
      <Box sx={{ mt: 1.5 }}>
        <Typography
          variant="caption"
          color="text.secondary"
          sx={{ mb: 1, display: "block" }}
        >
          참고 문서 {sources.length}건
        </Typography>
        <Box sx={{ display: "flex", gap: 1, flexWrap: "wrap" }}>
          {sources.map((src, idx) => {
            const { icon, color } = getFileTypeMeta(src.filename);
            const isExpanded = expandedIdx.has(idx);
            const isImage = isImageFile(src.filename);
            // Build image URL: prefer source_url, otherwise cannot preview
            const imageUrl = src.source_url
              ? `${API_BASE}${src.source_url}`
              : null;

            return (
              <Box
                key={src.chunkId || idx}
                sx={{
                  flex: "1 1 200px",
                  maxWidth: 300,
                  borderRadius: 2,
                  border: "1px solid",
                  borderColor: "divider",
                  borderLeft: `3px solid ${color}`,
                  overflow: "hidden",
                  transition: "all 0.2s",
                  "&:hover": {
                    borderColor: color,
                    borderLeftColor: color,
                    bgcolor: "action.hover",
                  },
                }}
              >
                {/* Image thumbnail — only for image sources with a URL */}
                {isImage && imageUrl && (
                  <Box
                    onClick={() => openLightbox(imageUrl, src.filename)}
                    sx={{
                      position: "relative",
                      width: "100%",
                      height: 120,
                      overflow: "hidden",
                      cursor: "zoom-in",
                      bgcolor: "rgba(0,0,0,0.04)",
                      "&:hover .img-overlay": { opacity: 1 },
                    }}
                  >
                    <Box
                      component="img"
                      src={imageUrl}
                      alt={src.filename}
                      sx={{
                        width: "100%",
                        height: "100%",
                        objectFit: "cover",
                        display: "block",
                        transition: "transform 0.25s ease",
                        "&:hover": { transform: "scale(1.04)" },
                      }}
                      onError={(e) => {
                        // Hide thumbnail row on load error
                        const parent = (e.currentTarget as HTMLElement)
                          .parentElement;
                        if (parent) parent.style.display = "none";
                      }}
                    />
                    {/* Hover overlay */}
                    <Box
                      className="img-overlay"
                      sx={{
                        position: "absolute",
                        inset: 0,
                        bgcolor: "rgba(0,0,0,0.38)",
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        opacity: 0,
                        transition: "opacity 0.18s",
                      }}
                    >
                      <ImageSearchIcon
                        sx={{ color: "#fff", fontSize: 28, opacity: 0.9 }}
                      />
                    </Box>
                  </Box>
                )}

                {/* Card header — clickable to expand/collapse */}
                <Box
                  onClick={() => toggleExpand(idx)}
                  sx={{
                    p: 1.5,
                    cursor: "pointer",
                    pb: src.content ? 1 : 1.5,
                  }}
                >
                  {/* Top row: icon + filename + page badge + source link */}
                  <Box
                    sx={{
                      display: "flex",
                      alignItems: "center",
                      gap: 0.5,
                      mb: 0.25,
                    }}
                  >
                    {icon}
                    <Typography
                      variant="subtitle2"
                      sx={{ fontSize: "0.78rem", flex: 1, lineHeight: 1.3 }}
                      noWrap
                      title={src.filename}
                    >
                      {src.filename}
                    </Typography>
                    {src.page != null && (
                      <Chip
                        label={`p.${src.page}`}
                        size="small"
                        variant="outlined"
                        sx={{
                          height: 18,
                          fontSize: "0.65rem",
                          borderColor: color,
                          color: color,
                          flexShrink: 0,
                          "& .MuiChip-label": { px: 0.75 },
                        }}
                      />
                    )}
                    {/* For image sources, show "미리보기" chip if we have a URL */}
                    {isImage && imageUrl && (
                      <Chip
                        label="미리보기"
                        size="small"
                        variant="outlined"
                        onClick={(e: React.MouseEvent) => {
                          e.stopPropagation();
                          openLightbox(imageUrl, src.filename);
                        }}
                        sx={{
                          height: 18,
                          fontSize: "0.65rem",
                          borderColor: "#8e24aa",
                          color: "#8e24aa",
                          flexShrink: 0,
                          cursor: "zoom-in",
                          "& .MuiChip-label": { px: 0.75 },
                          "&:hover": {
                            bgcolor: "rgba(142,36,170,0.08)",
                          },
                        }}
                      />
                    )}
                    {src.source_url && !isImage && (
                      <Chip
                        label="원본"
                        size="small"
                        component="a"
                        href={`${API_BASE}${src.source_url}`}
                        target="_blank"
                        rel="noopener noreferrer"
                        clickable
                        variant="outlined"
                        onClick={(e: React.MouseEvent) => e.stopPropagation()}
                        sx={{
                          height: 18,
                          fontSize: "0.65rem",
                          flexShrink: 0,
                          "& .MuiChip-label": { px: 0.75 },
                        }}
                      />
                    )}
                  </Box>

                  {/* Score row: bar + stars */}
                  <ScoreBar score={src.score} color={color} />
                  <Box
                    sx={{
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "space-between",
                    }}
                  >
                    <ScoreStars score={src.score} />
                    <Typography
                      variant="caption"
                      sx={{ fontSize: "0.65rem", color: "text.disabled" }}
                    >
                      {Math.round(src.score * 100)}%
                    </Typography>
                  </Box>
                </Box>

                {/* Expandable content */}
                {src.content && (
                  <Box sx={{ px: 1.5, pb: 1 }}>
                    <HighlightedContent
                      content={src.content}
                      keywords={keywords}
                      expanded={isExpanded}
                    />
                    <Box
                      sx={{
                        display: "flex",
                        justifyContent: "flex-end",
                        mt: 0.5,
                      }}
                    >
                      <IconButton
                        size="small"
                        onClick={(e) => {
                          e.stopPropagation();
                          toggleExpand(idx);
                        }}
                        sx={{ p: 0.25 }}
                        aria-label={isExpanded ? "접기" : "더 보기"}
                      >
                        {isExpanded ? (
                          <ExpandLess sx={{ fontSize: 16 }} />
                        ) : (
                          <ExpandMore sx={{ fontSize: 16 }} />
                        )}
                      </IconButton>
                      <Typography
                        variant="caption"
                        onClick={(e) => {
                          e.stopPropagation();
                          toggleExpand(idx);
                        }}
                        sx={{
                          fontSize: "0.7rem",
                          color: "primary.main",
                          cursor: "pointer",
                          lineHeight: "26px",
                          userSelect: "none",
                          "&:hover": { textDecoration: "underline" },
                        }}
                      >
                        {isExpanded ? "접기" : "더 보기"}
                      </Typography>
                    </Box>
                  </Box>
                )}
              </Box>
            );
          })}
        </Box>
      </Box>

      {/* Global lightbox (rendered outside card flow to avoid stacking context issues) */}
      <ImageLightbox
        open={lightbox.open}
        src={lightbox.src}
        filename={lightbox.filename}
        onClose={closeLightbox}
      />
    </>
  );
}
