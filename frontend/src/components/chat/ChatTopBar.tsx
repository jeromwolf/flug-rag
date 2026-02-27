import { useState } from "react";
import {
  Box,
  IconButton,
  Select,
  MenuItem,
  Slider,
  ToggleButtonGroup,
  ToggleButton,
  Popover,
  Tooltip,
  Typography,
} from "@mui/material";
import type { SelectChangeEvent } from "@mui/material";
import {
  Menu as MenuIcon,
  Settings as SettingsIcon,
  DarkMode as DarkModeIcon,
  LightMode as LightModeIcon,
  ContentCopy as ContentCopyIcon,
} from "@mui/icons-material";
import { useQuery } from "@tanstack/react-query";
import { adminApi } from "../../api/client";
import type { Message } from "../../types";

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
}

const FALLBACK_MODELS = [
  { id: "default", name: "기본 모델" },
  { id: "vllm", name: "vLLM" },
  { id: "ollama", name: "Ollama" },
  { id: "openai", name: "OpenAI" },
  { id: "anthropic", name: "Anthropic" },
];

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
}: ChatTopBarProps) {
  const [anchorEl, setAnchorEl] = useState<HTMLButtonElement | null>(null);
  const settingsOpen = Boolean(anchorEl);

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
        const label = m.role === "user" ? "사용자" : "Flux AI";
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

  return (
    <Box
      sx={{
        display: "flex",
        alignItems: "center",
        gap: 0.5,
        px: 1.5,
        borderBottom: "1px solid",
        borderColor: darkMode ? "rgba(255,255,255,0.05)" : "rgba(0,0,0,0.05)",
        bgcolor: "transparent",
        minHeight: 48,
      }}
    >
      {/* Left: menu toggle */}
      <IconButton onClick={onToggleSidebar} size="small">
        <MenuIcon fontSize="small" />
      </IconButton>

      {/* Left spacer */}
      <Box sx={{ flex: 1 }} />

      {/* Center: model selector */}
      <Select
        size="small"
        value={selectedModel || "default"}
        onChange={(e: SelectChangeEvent) => onModelChange(e.target.value)}
        aria-label="AI 모델 선택"
        sx={{
          borderRadius: 2,
          minWidth: 140,
          "& .MuiOutlinedInput-notchedOutline": {
            borderColor: "divider",
          },
          "& .MuiSelect-select": {
            py: 0.75,
            fontSize: "0.875rem",
          },
        }}
      >
        {modelOptions.map((m) => (
          <MenuItem key={m.id} value={m.id} sx={{ fontSize: "0.875rem" }}>
            {m.name}
          </MenuItem>
        ))}
      </Select>

      {/* Right spacer */}
      <Box sx={{ flex: 1 }} />

      {/* Right: action buttons */}
      <Tooltip title="대화 복사">
        <IconButton onClick={handleCopyConversation} size="small">
          <ContentCopyIcon fontSize="small" />
        </IconButton>
      </Tooltip>

      <Tooltip title={darkMode ? "라이트 모드" : "다크 모드"}>
        <IconButton onClick={onToggleDarkMode} size="small">
          {darkMode ? (
            <LightModeIcon fontSize="small" />
          ) : (
            <DarkModeIcon fontSize="small" />
          )}
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
            sx: { p: 2.5, width: 280, mt: 1 },
          },
        }}
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

        <Typography
          variant="caption"
          color="text.secondary"
          sx={{ mb: 0.5, display: "block" }}
        >
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
