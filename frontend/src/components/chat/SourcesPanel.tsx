import {
  Box,
  Typography,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  LinearProgress,
  Chip,
} from "@mui/material";
import {
  ExpandMore as ExpandMoreIcon,
  PictureAsPdf,
  Description,
  Article,
  InsertDriveFile,
} from "@mui/icons-material";
import type { Source } from "../../types";

function getFileIcon(filename: string) {
  const ext = filename.split('.').pop()?.toLowerCase();
  switch (ext) {
    case 'pdf': return <PictureAsPdf fontSize="small" color="error" />;
    case 'hwp': return <Description fontSize="small" color="info" />;
    case 'docx': case 'doc': return <Article fontSize="small" color="primary" />;
    default: return <InsertDriveFile fontSize="small" color="action" />;
  }
}

export function SourcesPanel({ sources }: { sources: Source[] }) {
  if (sources.length === 0) return null;
  return (
    <Accordion
      disableGutters
      sx={{ mt: 1, "&:before": { display: "none" }, boxShadow: "none", bgcolor: "transparent" }}
    >
      <AccordionSummary expandIcon={<ExpandMoreIcon />} sx={{ minHeight: 32, px: 0 }}>
        <Chip
          label={`참고 문서 ${sources.length}건`}
          size="small"
          variant="outlined"
          sx={{ fontSize: '0.75rem' }}
        />
      </AccordionSummary>
      <AccordionDetails sx={{ p: 0 }}>
        {sources.map((src, idx) => (
          <Box
            key={src.chunkId || idx}
            sx={{
              p: 1.5,
              borderRadius: 1,
              bgcolor: "action.hover",
              mb: 0.5,
            }}
          >
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, mb: 0.5 }}>
              {getFileIcon(src.filename)}
              <Typography variant="subtitle2" sx={{ fontSize: '0.8rem' }}>
                {src.filename}
                {src.page != null && ` (p.${src.page})`}
              </Typography>
            </Box>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.5 }}>
              <LinearProgress
                variant="determinate"
                value={Math.round(src.score * 100)}
                sx={{ flex: 1, height: 4, borderRadius: 2 }}
              />
              <Typography variant="caption" color="text.secondary" sx={{ minWidth: 32 }}>
                {Math.round(src.score * 100)}%
              </Typography>
            </Box>
            {src.content && (
              <Typography
                variant="body2"
                sx={{
                  whiteSpace: "pre-wrap",
                  maxHeight: 80,
                  overflow: "hidden",
                  fontSize: "0.78rem",
                  color: "text.secondary",
                  lineHeight: 1.4,
                }}
              >
                {src.content}
              </Typography>
            )}
          </Box>
        ))}
      </AccordionDetails>
    </Accordion>
  );
}
