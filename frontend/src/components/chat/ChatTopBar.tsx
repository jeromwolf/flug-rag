import { useState } from "react";
import {
  Box,
  IconButton,
  Typography,
  Select,
  MenuItem,
  Slider,
  ToggleButtonGroup,
  ToggleButton,
  Popover,
  Tooltip,
} from "@mui/material";
import type { SelectChangeEvent } from "@mui/material";
import {
  Menu as MenuIcon,
  Settings as SettingsIcon,
  DarkMode as DarkModeIcon,
  LightMode as LightModeIcon,
} from "@mui/icons-material";

interface ChatTopBarProps {
  onToggleSidebar: () => void;
  selectedModel: string;
  onModelChange: (model: string) => void;
  temperature: number;
  onTemperatureChange: (temp: number) => void;
  responseMode: "rag" | "direct";
  onModeChange: (mode: "rag" | "direct") => void;
  currentSessionId: string | null;
  darkMode: boolean;
  onToggleDarkMode: () => void;
}

export function ChatTopBar({
  onToggleSidebar,
  selectedModel,
  onModelChange,
  temperature,
  onTemperatureChange,
  responseMode,
  onModeChange,
  currentSessionId,
  darkMode,
  onToggleDarkMode,
}: ChatTopBarProps) {
  const [anchorEl, setAnchorEl] = useState<HTMLButtonElement | null>(null);
  const settingsOpen = Boolean(anchorEl);

  return (
    <Box
      sx={{
        display: "flex",
        alignItems: "center",
        gap: 1,
        px: 2,
        py: 1,
        borderBottom: "1px solid",
        borderColor: "divider",
        bgcolor: "background.paper",
        minHeight: 52,
      }}
    >
      <IconButton onClick={onToggleSidebar} size="small">
        <MenuIcon />
      </IconButton>

      <Typography variant="subtitle1" fontWeight={600} sx={{ ml: 0.5 }}>
        Flux RAG
      </Typography>

      <Box sx={{ flex: 1 }} />

      {currentSessionId && (
        <Typography variant="caption" color="text.secondary" sx={{ mr: 1 }}>
          {currentSessionId.slice(0, 8)}
        </Typography>
      )}

      <Tooltip title={darkMode ? "라이트 모드" : "다크 모드"}>
        <IconButton onClick={onToggleDarkMode} size="small">
          {darkMode ? <LightModeIcon fontSize="small" /> : <DarkModeIcon fontSize="small" />}
        </IconButton>
      </Tooltip>

      <Tooltip title="설정">
        <IconButton
          onClick={(e) => setAnchorEl(e.currentTarget)}
          size="small"
          sx={{
            bgcolor: settingsOpen ? "action.selected" : "transparent",
          }}
        >
          <SettingsIcon fontSize="small" />
        </IconButton>
      </Tooltip>

      <Popover
        open={settingsOpen}
        anchorEl={anchorEl}
        onClose={() => setAnchorEl(null)}
        anchorOrigin={{ vertical: "bottom", horizontal: "right" }}
        transformOrigin={{ vertical: "top", horizontal: "right" }}
        slotProps={{
          paper: {
            sx: { p: 2.5, width: 300, mt: 1 },
          },
        }}
      >
        <Typography variant="subtitle2" sx={{ mb: 1.5 }}>
          AI 설정
        </Typography>

        <Typography variant="caption" color="text.secondary">
          모델
        </Typography>
        <Select
          size="small"
          fullWidth
          value={selectedModel}
          onChange={(e: SelectChangeEvent) => onModelChange(e.target.value)}
          sx={{ mb: 2, mt: 0.5 }}
          aria-label="AI 모델 선택"
        >
          <MenuItem value="default">기본 모델</MenuItem>
          <MenuItem value="vllm">vLLM</MenuItem>
          <MenuItem value="ollama">Ollama</MenuItem>
          <MenuItem value="openai">OpenAI</MenuItem>
          <MenuItem value="anthropic">Anthropic</MenuItem>
        </Select>

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

        <Typography variant="caption" color="text.secondary" sx={{ mb: 0.5, display: "block" }}>
          응답 모드
        </Typography>
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
  );
}
