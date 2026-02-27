import { useState } from "react";
import { Box, Typography, Chip } from "@mui/material";
import {
  PictureAsPdf,
  Description,
  Article,
  InsertDriveFile,
} from "@mui/icons-material";
import type { Source } from "../../types";

function getFileIcon(filename: string) {
  const ext = filename.split(".").pop()?.toLowerCase();
  const iconSx = { fontSize: 16 };
  switch (ext) {
    case "pdf":
      return <PictureAsPdf sx={{ ...iconSx, color: "#e53935" }} />;
    case "hwp":
      return <Description sx={{ ...iconSx, color: "#1565c0" }} />;
    case "docx":
    case "doc":
      return <Article sx={{ ...iconSx, color: "#1976d2" }} />;
    default:
      return <InsertDriveFile sx={{ ...iconSx, color: "#757575" }} />;
  }
}

export function SourcesPanel({ sources }: { sources: Source[] }) {
  const [expandedIdx, setExpandedIdx] = useState<number | null>(null);

  if (!sources || sources.length === 0) return null;

  return (
    <Box sx={{ mt: 1.5 }}>
      <Typography
        variant="caption"
        color="text.secondary"
        sx={{ mb: 1, display: "block" }}
      >
        참고 문서 {sources.length}건
      </Typography>
      <Box sx={{ display: "flex", gap: 1, flexWrap: "wrap" }}>
        {sources.map((src, idx) => (
          <Box
            key={src.chunkId || idx}
            onClick={() =>
              setExpandedIdx(expandedIdx === idx ? null : idx)
            }
            sx={{
              flex: "1 1 200px",
              maxWidth: 280,
              p: 1.5,
              borderRadius: 2,
              cursor: "pointer",
              border: "1px solid",
              borderColor: "divider",
              transition: "all 0.2s",
              "&:hover": {
                borderColor: "text.secondary",
                bgcolor: "action.hover",
              },
            }}
          >
            <Box sx={{ display: "flex", alignItems: "center", gap: 0.5, mb: 0.5 }}>
              {getFileIcon(src.filename)}
              <Typography
                variant="subtitle2"
                sx={{ fontSize: "0.78rem", flex: 1 }}
                noWrap
              >
                {src.filename}
                {src.page ? ` (p.${src.page})` : ""}
              </Typography>
              <Chip
                label={`${Math.round(src.score * 100)}%`}
                size="small"
                color={
                  src.score > 0.8
                    ? "success"
                    : src.score > 0.5
                    ? "warning"
                    : "default"
                }
                sx={{ height: 20, fontSize: "0.7rem" }}
              />
            </Box>
            {expandedIdx === idx && src.content && (
              <Typography
                variant="body2"
                sx={{
                  mt: 1,
                  fontSize: "0.78rem",
                  color: "text.secondary",
                  lineHeight: 1.5,
                  whiteSpace: "pre-wrap",
                  animation: "fadeInUp 0.2s ease-out",
                }}
              >
                {src.content}
              </Typography>
            )}
          </Box>
        ))}
      </Box>
    </Box>
  );
}
