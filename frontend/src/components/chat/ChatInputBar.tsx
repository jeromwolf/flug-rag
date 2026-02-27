import { useRef } from "react";
import { Box, TextField, IconButton, Tooltip, Chip, Typography } from "@mui/material";
import {
  ArrowUpward as ArrowUpwardIcon,
  StopCircle as StopCircleIcon,
  AttachFile as AttachFileIcon,
  Close as CloseIcon,
} from "@mui/icons-material";
import { useDropzone } from "react-dropzone";

interface ChatInputBarProps {
  inputValue: string;
  onInputChange: (value: string) => void;
  onSend: () => void;
  onStop: () => void;
  onKeyDown: (e: React.KeyboardEvent) => void;
  isStreaming: boolean;
  attachedFiles: File[];
  onFilesAttached: (files: File[]) => void;
  onRemoveFile: (index: number) => void;
}

const ACCEPTED_TYPES = {
  "application/pdf": [".pdf"],
  "application/msword": [".doc"],
  "application/vnd.openxmlformats-officedocument.wordprocessingml.document": [".docx"],
  "application/haansofthwp": [".hwp"],
  "text/plain": [".txt"],
  "application/vnd.ms-excel": [".xls"],
  "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": [".xlsx"],
  "application/vnd.ms-powerpoint": [".ppt"],
  "application/vnd.openxmlformats-officedocument.presentationml.presentation": [".pptx"],
};

export function ChatInputBar({
  inputValue,
  onInputChange,
  onSend,
  onStop,
  onKeyDown,
  isStreaming,
  attachedFiles,
  onFilesAttached,
  onRemoveFile,
}: ChatInputBarProps) {
  const attachButtonRef = useRef<HTMLButtonElement>(null);

  const { getRootProps, getInputProps, open } = useDropzone({
    accept: ACCEPTED_TYPES,
    noClick: true,
    noKeyboard: true,
    onDrop: (acceptedFiles) => {
      onFilesAttached(acceptedFiles);
    },
  });

  const charCount = inputValue.length;
  const showCounter = charCount > 5000;
  const isNearLimit = charCount > 9500;

  return (
    <Box
      sx={{
        px: 2,
        py: 2,
        display: "flex",
        justifyContent: "center",
        bgcolor: "background.default",
      }}
    >
      <Box
        {...getRootProps()}
        sx={{
          maxWidth: 768,
          width: "100%",
          border: "1px solid",
          borderColor: "divider",
          borderRadius: 3,
          bgcolor: "background.paper",
          display: "flex",
          flexDirection: "column",
          transition: "border-color 0.2s",
          "&:focus-within": { borderColor: "text.secondary" },
        }}
      >
        <input {...getInputProps()} />

        {/* Attached files row */}
        {attachedFiles.length > 0 && (
          <Box
            sx={{
              display: "flex",
              flexWrap: "wrap",
              gap: 0.75,
              px: 1.5,
              pt: 1.25,
            }}
          >
            {attachedFiles.map((file, index) => (
              <Chip
                key={`${file.name}-${index}`}
                label={file.name}
                size="small"
                onDelete={() => onRemoveFile(index)}
                deleteIcon={<CloseIcon sx={{ fontSize: 14 }} />}
                sx={{
                  maxWidth: 200,
                  "& .MuiChip-label": {
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                    fontSize: "0.75rem",
                  },
                }}
              />
            ))}
          </Box>
        )}

        {/* Main input row */}
        <Box
          sx={{
            display: "flex",
            alignItems: "flex-end",
            px: 0.5,
            py: 0.5,
            gap: 0.5,
          }}
        >
          <Tooltip title="파일 첨부">
            <IconButton
              ref={attachButtonRef}
              onClick={open}
              size="small"
              sx={{
                mb: 0.5,
                color: "text.secondary",
                "&:hover": { color: "text.primary" },
              }}
              aria-label="파일 첨부"
            >
              <AttachFileIcon fontSize="small" />
            </IconButton>
          </Tooltip>

          <TextField
            fullWidth
            multiline
            maxRows={12}
            placeholder="메시지를 입력하세요..."
            value={inputValue}
            onChange={(e) => onInputChange(e.target.value)}
            onKeyDown={onKeyDown}
            disabled={isStreaming}
            size="small"
            sx={{
              "& .MuiOutlinedInput-root": {
                "& fieldset": { border: "none" },
                padding: "8px 4px",
              },
              "& .MuiInputBase-inputMultiline": {
                lineHeight: 1.6,
              },
            }}
            inputProps={{ "aria-label": "메시지 입력" }}
          />

          <Box sx={{ display: "flex", flexDirection: "column", alignItems: "flex-end", mb: 0.5, gap: 0.25 }}>
            {showCounter && (
              <Typography
                variant="caption"
                sx={{
                  fontSize: "0.65rem",
                  color: isNearLimit ? "warning.main" : "text.disabled",
                  lineHeight: 1,
                  pr: 0.5,
                }}
              >
                {charCount.toLocaleString()}
              </Typography>
            )}
            {isStreaming ? (
              <Tooltip title="생성 중단">
                <IconButton
                  onClick={onStop}
                  sx={{
                    width: 32,
                    height: 32,
                    bgcolor: "error.main",
                    color: "white",
                    "&:hover": { bgcolor: "error.dark" },
                    borderRadius: 1.5,
                  }}
                  aria-label="생성 중단"
                >
                  <StopCircleIcon sx={{ fontSize: 18 }} />
                </IconButton>
              </Tooltip>
            ) : (
              <Tooltip title="전송 (Enter)">
                <span>
                  <IconButton
                    onClick={onSend}
                    disabled={!inputValue.trim()}
                    sx={{
                      width: 32,
                      height: 32,
                      bgcolor: inputValue.trim() ? "primary.main" : "action.disabledBackground",
                      color: "white",
                      "&:hover": { bgcolor: inputValue.trim() ? "primary.dark" : "action.disabledBackground" },
                      "&.Mui-disabled": {
                        bgcolor: "action.disabledBackground",
                        color: "action.disabled",
                      },
                      borderRadius: 1.5,
                      transition: "background-color 0.2s",
                    }}
                    aria-label="메시지 전송"
                  >
                    <ArrowUpwardIcon sx={{ fontSize: 18 }} />
                  </IconButton>
                </span>
              </Tooltip>
            )}
          </Box>
        </Box>
      </Box>
    </Box>
  );
}
