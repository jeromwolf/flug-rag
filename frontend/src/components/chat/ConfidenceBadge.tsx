import { Chip } from "@mui/material";
import { getConfidenceLevel } from "../../types";

export function ConfidenceBadge({ score }: { score: number }) {
  const level = getConfidenceLevel(score);
  const colorMap = { high: "success", medium: "warning", low: "error" } as const;
  const labelMap = { high: "높음", medium: "중간", low: "낮음" } as const;
  return (
    <Chip
      label={`신뢰도: ${labelMap[level]} (${Math.round(score * 100)}%)`}
      color={colorMap[level]}
      size="small"
      variant="outlined"
    />
  );
}
