import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Typography,
  Box,
  Chip,
  CircularProgress,
  Divider,
  Stack,
  Alert,
} from '@mui/material';
import { useQuery } from '@tanstack/react-query';
import { feedbackApi } from '../../api/client';

interface FeedbackDrilldownDialogProps {
  open: boolean;
  feedbackId: string | null;
  onClose: () => void;
}

export default function FeedbackDrilldownDialog({
  open,
  feedbackId,
  onClose,
}: FeedbackDrilldownDialogProps) {
  const { data: feedback, isLoading, error } = useQuery({
    queryKey: ['feedback-detail', feedbackId],
    queryFn: () => feedbackApi.getDetail(feedbackId!).then((r) => r.data),
    enabled: !!feedbackId && open,
  });

  const getRatingLabel = (rating: number) => {
    if (rating > 0) return '긍정';
    if (rating < 0) return '부정';
    return '중립';
  };

  const getRatingColor = (rating: number) => {
    if (rating > 0) return 'success';
    if (rating < 0) return 'error';
    return 'default';
  };

  const formatDate = (isoString: string) => {
    if (!isoString) return '';
    const date = new Date(isoString);
    return date.toLocaleString('ko-KR', {
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
    });
  };

  const truncateSessionId = (id: string) => {
    if (!id) return '';
    return id.length > 8 ? `${id.substring(0, 8)}...` : id;
  };

  return (
    <Dialog open={open} maxWidth="md" fullWidth onClose={onClose}>
      <DialogTitle
        sx={{
          fontWeight: 'bold',
          display: 'flex',
          alignItems: 'center',
          gap: 2,
        }}
      >
        피드백 상세
        {feedback && (
          <Chip
            label={getRatingLabel(feedback.rating || 0)}
            color={getRatingColor(feedback.rating || 0) as any}
            size="small"
          />
        )}
      </DialogTitle>

      <DialogContent dividers>
        {isLoading ? (
          <Box display="flex" justifyContent="center" p={4}>
            <CircularProgress />
          </Box>
        ) : error ? (
          <Alert severity="error">피드백을 불러올 수 없습니다.</Alert>
        ) : feedback ? (
          <Stack spacing={2}>
            {/* 질문 */}
            <Box>
              <Typography variant="subtitle2" sx={{ fontWeight: 'bold', mb: 1 }}>
                질문
              </Typography>
              <Box
                sx={{
                  p: 2,
                  bgcolor: 'grey.100',
                  borderRadius: 1,
                  border: '1px solid',
                  borderColor: 'grey.300',
                }}
              >
                <Typography variant="body2" sx={{ whiteSpace: 'pre-wrap' }}>
                  {feedback.query || '(없음)'}
                </Typography>
              </Box>
            </Box>

            <Divider />

            {/* AI 응답 */}
            <Box>
              <Typography variant="subtitle2" sx={{ fontWeight: 'bold', mb: 1 }}>
                AI 응답
              </Typography>
              <Box
                sx={{
                  p: 2,
                  bgcolor: 'grey.100',
                  borderRadius: 1,
                  border: '1px solid',
                  borderColor: 'grey.300',
                  maxHeight: 300,
                  overflow: 'auto',
                }}
              >
                <Typography variant="body2" sx={{ whiteSpace: 'pre-wrap' }}>
                  {feedback.answer || '(없음)'}
                </Typography>
              </Box>
            </Box>

            <Divider />

            {/* 사용자 코멘트 */}
            {feedback.comment && (
              <>
                <Box>
                  <Typography variant="subtitle2" sx={{ fontWeight: 'bold', mb: 1 }}>
                    사용자 코멘트
                  </Typography>
                  <Typography variant="body2" sx={{ whiteSpace: 'pre-wrap' }}>
                    {feedback.comment}
                  </Typography>
                </Box>
                <Divider />
              </>
            )}

            {/* 수정된 답변 */}
            {feedback.corrected_answer && (
              <>
                <Box>
                  <Typography variant="subtitle2" sx={{ fontWeight: 'bold', mb: 1 }}>
                    수정된 답변
                  </Typography>
                  <Box
                    sx={{
                      p: 2,
                      bgcolor: '#e8f5e9',
                      borderRadius: 1,
                      border: '1px solid',
                      borderColor: '#4caf50',
                      maxHeight: 300,
                      overflow: 'auto',
                    }}
                  >
                    <Typography variant="body2" sx={{ whiteSpace: 'pre-wrap' }}>
                      {feedback.corrected_answer}
                    </Typography>
                  </Box>
                </Box>
                <Divider />
              </>
            )}

            {/* 메타 정보 */}
            <Box>
              <Typography variant="subtitle2" sx={{ fontWeight: 'bold', mb: 1 }}>
                메타 정보
              </Typography>
              <Stack spacing={1}>
                <Box display="flex" gap={2}>
                  <Typography variant="body2" sx={{ color: 'text.secondary' }}>
                    사용자:
                  </Typography>
                  <Typography variant="body2">{feedback.username || '(알 수 없음)'}</Typography>
                </Box>
                <Box display="flex" gap={2}>
                  <Typography variant="body2" sx={{ color: 'text.secondary' }}>
                    생성 시간:
                  </Typography>
                  <Typography variant="body2">{formatDate(feedback.created_at || '')}</Typography>
                </Box>
                <Box display="flex" gap={2}>
                  <Typography variant="body2" sx={{ color: 'text.secondary' }}>
                    세션 ID:
                  </Typography>
                  <Typography
                    variant="body2"
                    sx={{
                      fontFamily: 'monospace',
                      fontSize: '0.85rem',
                      color: 'text.secondary',
                    }}
                    title={feedback.session_id || ''}
                  >
                    {truncateSessionId(feedback.session_id || '')}
                  </Typography>
                </Box>
                {feedback.message_id && (
                  <Box display="flex" gap={2}>
                    <Typography variant="body2" sx={{ color: 'text.secondary' }}>
                      메시지 ID:
                    </Typography>
                    <Typography
                      variant="body2"
                      sx={{
                        fontFamily: 'monospace',
                        fontSize: '0.85rem',
                        color: 'text.secondary',
                      }}
                      title={feedback.message_id}
                    >
                      {truncateSessionId(feedback.message_id)}
                    </Typography>
                  </Box>
                )}
              </Stack>
            </Box>
          </Stack>
        ) : null}
      </DialogContent>

      <DialogActions>
        <Button onClick={onClose} variant="outlined">
          닫기
        </Button>
      </DialogActions>
    </Dialog>
  );
}
