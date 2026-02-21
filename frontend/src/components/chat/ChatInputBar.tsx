import { Box, TextField, IconButton, Tooltip } from "@mui/material";
import { Send as SendIcon, Stop as StopIcon } from "@mui/icons-material";

interface ChatInputBarProps {
  inputValue: string;
  onInputChange: (value: string) => void;
  onSend: () => void;
  onStop: () => void;
  onKeyDown: (e: React.KeyboardEvent) => void;
  isStreaming: boolean;
}

export function ChatInputBar({
  inputValue,
  onInputChange,
  onSend,
  onStop,
  onKeyDown,
  isStreaming,
}: ChatInputBarProps) {
  return (
    <Box
      sx={{
        p: 2,
        borderTop: "1px solid",
        borderColor: "divider",
        bgcolor: "background.paper",
      }}
    >
      <Box
        sx={{
          display: "flex",
          gap: 1.5,
          alignItems: "flex-end",
          maxWidth: 900,
          mx: "auto",
        }}
      >
        <TextField
          fullWidth
          multiline
          maxRows={6}
          placeholder="메시지를 입력하세요... (Shift+Enter로 줄바꿈)"
          value={inputValue}
          onChange={(e) => onInputChange(e.target.value)}
          onKeyDown={onKeyDown}
          disabled={isStreaming}
          size="small"
          sx={{
            "& .MuiOutlinedInput-root": {
              borderRadius: 3,
              bgcolor: "action.hover",
              "&:hover": {
                bgcolor: "action.selected",
              },
              "&.Mui-focused": {
                bgcolor: "background.paper",
              },
            },
          }}
          inputProps={{ "aria-label": "메시지 입력" }}
        />
        {isStreaming ? (
          <Tooltip title="생성 중단">
            <IconButton
              onClick={onStop}
              sx={{
                width: 42,
                height: 42,
                bgcolor: "error.main",
                color: "white",
                "&:hover": { bgcolor: "error.dark" },
                boxShadow: 1,
              }}
              aria-label="생성 중단"
            >
              <StopIcon />
            </IconButton>
          </Tooltip>
        ) : (
          <Tooltip title="전송 (Enter)">
            <span>
              <IconButton
                onClick={onSend}
                disabled={!inputValue.trim()}
                sx={{
                  width: 42,
                  height: 42,
                  bgcolor: "primary.main",
                  color: "white",
                  "&:hover": { bgcolor: "primary.dark" },
                  "&.Mui-disabled": {
                    bgcolor: "grey.300",
                    color: "grey.500",
                  },
                  boxShadow: 1,
                  transition: "all 0.2s",
                }}
                aria-label="메시지 전송"
              >
                <SendIcon />
              </IconButton>
            </span>
          </Tooltip>
        )}
      </Box>
    </Box>
  );
}
