import { useState } from "react";
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
import { authApi } from "../../api/client";

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
// Feedback Comment Dialog (for negative/partial feedback)
// ---------------------------------------------------------------------------

interface FeedbackCommentDialogProps {
  open: boolean;
  rating: number;
  comment: string;
  onCommentChange: (value: string) => void;
  onClose: () => void;
  onSubmit: () => void;
}

export function FeedbackCommentDialog({
  open,
  rating,
  comment,
  onCommentChange,
  onClose,
  onSubmit,
}: FeedbackCommentDialogProps) {
  const ratingLabel = rating === -1 ? "부정확" : "부분적으로 정확";
  return (
    <Dialog open={open} onClose={onClose}>
      <DialogTitle>피드백 - {ratingLabel}</DialogTitle>
      <DialogContent>
        <DialogContentText>
          어떤 부분이 {rating === -1 ? "부정확" : "부족"}했는지 알려주세요. (선택사항)
        </DialogContentText>
        <TextField
          autoFocus
          margin="dense"
          multiline
          rows={3}
          fullWidth
          placeholder="예: 답변에 포함된 날짜가 틀렸습니다..."
          value={comment}
          onChange={(e) => onCommentChange(e.target.value)}
        />
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>취소</Button>
        <Button color="primary" variant="contained" onClick={onSubmit}>
          전송
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
// Password Confirm Dialog (for sensitive admin operations)
// ---------------------------------------------------------------------------

interface PasswordConfirmDialogProps {
  open: boolean;
  onClose: () => void;
  onConfirm: (password: string) => void;
  title?: string;
  description?: string;
}

export function PasswordConfirmDialog({
  open,
  onClose,
  onConfirm,
  title = "설정 변경 확인",
  description = "이 작업은 시스템에 즉시 반영됩니다. 계속하시려면 비밀번호를 입력하세요.",
}: PasswordConfirmDialogProps) {
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const handleClose = () => {
    setPassword("");
    setError("");
    onClose();
  };

  const handleConfirm = async () => {
    if (!password) {
      setError("비밀번호를 입력해주세요.");
      return;
    }
    setLoading(true);
    setError("");
    try {
      await authApi.verifyPassword(password);
      onConfirm(password);
      setPassword("");
      setError("");
    } catch {
      setError("비밀번호가 올바르지 않습니다.");
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") {
      void handleConfirm();
    }
  };

  return (
    <Dialog open={open} onClose={handleClose} maxWidth="xs" fullWidth>
      <DialogTitle>{title}</DialogTitle>
      <DialogContent>
        <DialogContentText sx={{ mb: 2 }}>{description}</DialogContentText>
        <TextField
          autoFocus
          type="password"
          label="비밀번호"
          fullWidth
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          onKeyDown={handleKeyDown}
          error={!!error}
          helperText={error}
          size="small"
        />
      </DialogContent>
      <DialogActions>
        <Button onClick={handleClose} disabled={loading}>
          취소
        </Button>
        <Button
          color="primary"
          variant="contained"
          onClick={() => void handleConfirm()}
          disabled={loading}
        >
          {loading ? "확인 중..." : "확인"}
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
