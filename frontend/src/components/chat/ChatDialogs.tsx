import {
  Box,
  Typography,
  TextField,
  Button,
  Select,
  MenuItem,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogContentText,
  DialogActions,
  Snackbar,
  Alert,
} from "@mui/material";
import type { SelectChangeEvent } from "@mui/material";

// ---------------------------------------------------------------------------
// Delete Session Dialog
// ---------------------------------------------------------------------------

interface DeleteSessionDialogProps {
  open: boolean;
  onClose: () => void;
  onConfirm: () => void;
}

export function DeleteSessionDialog({ open, onClose, onConfirm }: DeleteSessionDialogProps) {
  return (
    <Dialog open={open} onClose={onClose}>
      <DialogTitle>대화 삭제</DialogTitle>
      <DialogContent>
        <DialogContentText>
          이 대화를 삭제하시겠습니까? 삭제된 대화는 복구할 수 없습니다.
        </DialogContentText>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>취소</Button>
        <Button color="error" variant="contained" onClick={onConfirm}>
          삭제
        </Button>
      </DialogActions>
    </Dialog>
  );
}

// ---------------------------------------------------------------------------
// Error Report Dialog
// ---------------------------------------------------------------------------

interface ErrorReportDialogProps {
  open: boolean;
  description: string;
  onDescriptionChange: (value: string) => void;
  onClose: () => void;
  onSubmit: () => void;
}

export function ErrorReportDialog({
  open,
  description,
  onDescriptionChange,
  onClose,
  onSubmit,
}: ErrorReportDialogProps) {
  return (
    <Dialog open={open} onClose={onClose}>
      <DialogTitle>오류 신고</DialogTitle>
      <DialogContent>
        <DialogContentText>
          발견하신 오류를 설명해주세요.
        </DialogContentText>
        <TextField
          autoFocus
          margin="dense"
          multiline
          rows={4}
          fullWidth
          placeholder="오류 내용을 입력하세요..."
          value={description}
          onChange={(e) => onDescriptionChange(e.target.value)}
        />
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>취소</Button>
        <Button color="primary" variant="contained" onClick={onSubmit}>
          신고
        </Button>
      </DialogActions>
    </Dialog>
  );
}

// ---------------------------------------------------------------------------
// Golden Data Edit Dialog
// ---------------------------------------------------------------------------

interface GoldenDataEditDialogProps {
  open: boolean;
  editedAnswer: string;
  evaluationTag: string;
  onAnswerChange: (value: string) => void;
  onTagChange: (value: string) => void;
  onClose: () => void;
  onSave: () => void;
}

export function GoldenDataEditDialog({
  open,
  editedAnswer,
  evaluationTag,
  onAnswerChange,
  onTagChange,
  onClose,
  onSave,
}: GoldenDataEditDialogProps) {
  return (
    <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth>
      <DialogTitle>답변 수정 (Golden Data 저장)</DialogTitle>
      <DialogContent>
        <DialogContentText>
          답변을 수정하여 Golden Data로 저장하세요. 이 데이터는 향후 모델 평가 및 개선에 사용됩니다.
        </DialogContentText>
        <TextField
          autoFocus
          margin="dense"
          multiline
          rows={8}
          fullWidth
          placeholder="수정된 답변을 입력하세요..."
          value={editedAnswer}
          onChange={(e) => onAnswerChange(e.target.value)}
          sx={{ mt: 2 }}
        />
        <Box sx={{ mt: 2 }}>
          <Typography variant="body2" color="text.secondary" gutterBottom>
            평가 태그:
          </Typography>
          <Select
            size="small"
            value={evaluationTag}
            onChange={(e: SelectChangeEvent) => onTagChange(e.target.value)}
            fullWidth
          >
            <MenuItem value="accurate">정확함 (Accurate)</MenuItem>
            <MenuItem value="partial">부분적으로 정확 (Partial)</MenuItem>
            <MenuItem value="inaccurate">부정확 (Inaccurate)</MenuItem>
            <MenuItem value="hallucination">환각 (Hallucination)</MenuItem>
          </Select>
        </Box>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>취소</Button>
        <Button
          color="primary"
          variant="contained"
          onClick={onSave}
          disabled={!editedAnswer.trim()}
        >
          저장
        </Button>
      </DialogActions>
    </Dialog>
  );
}

// ---------------------------------------------------------------------------
// Notification Snackbar
// ---------------------------------------------------------------------------

interface NotificationSnackbarProps {
  snackbar: { open: boolean; message: string; severity: "success" | "error" | "info" };
  onClose: () => void;
}

export function NotificationSnackbar({ snackbar, onClose }: NotificationSnackbarProps) {
  return (
    <Snackbar
      open={snackbar.open}
      autoHideDuration={3000}
      onClose={onClose}
      anchorOrigin={{ vertical: "bottom", horizontal: "center" }}
    >
      <Alert severity={snackbar.severity} onClose={onClose}>
        {snackbar.message}
      </Alert>
    </Snackbar>
  );
}
