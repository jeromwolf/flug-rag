import { useState } from 'react';
import {
  Box,
  Button,
  Chip,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  FormControl,
  InputLabel,
  MenuItem,
  Select,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TextField,
  Typography,
  Alert,
} from '@mui/material';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import api from '../api/client';

// ===== User: Access Request Form =====

export function AccessRequestForm() {
  const [open, setOpen] = useState(false);
  const [role, setRole] = useState('manager');
  const [reason, setReason] = useState('');
  const queryClient = useQueryClient();

  const { data: myRequests } = useQuery({
    queryKey: ['my-access-requests'],
    queryFn: () => api.get('/auth/access-request/my').then(r => r.data),
  });

  const createMutation = useMutation({
    mutationFn: () => api.post('/auth/access-request', {
      requested_role: role,
      reason,
    }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['my-access-requests'] });
      setOpen(false);
      setReason('');
    },
  });

  const statusColor = (status: string) => {
    if (status === 'approved') return 'success';
    if (status === 'rejected') return 'error';
    return 'warning';
  };

  const statusLabel = (status: string) => {
    if (status === 'approved') return '승인';
    if (status === 'rejected') return '거절';
    return '대기 중';
  };

  return (
    <Box>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
        <Typography variant="h6">내 권한 신청</Typography>
        <Button variant="contained" size="small" onClick={() => setOpen(true)}>
          권한 신청
        </Button>
      </Box>

      {myRequests?.requests?.length > 0 ? (
        <TableContainer>
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>요청 역할</TableCell>
                <TableCell>사유</TableCell>
                <TableCell>상태</TableCell>
                <TableCell>신청일</TableCell>
                <TableCell>검토 의견</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {myRequests.requests.map((r: any) => (
                <TableRow key={r.id}>
                  <TableCell>{r.requested_role}</TableCell>
                  <TableCell>{r.reason}</TableCell>
                  <TableCell>
                    <Chip label={statusLabel(r.status)} color={statusColor(r.status)} size="small" />
                  </TableCell>
                  <TableCell>{new Date(r.created_at).toLocaleDateString('ko-KR')}</TableCell>
                  <TableCell>{r.reviewer_comment || '-'}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      ) : (
        <Typography color="text.secondary">신청 내역이 없습니다.</Typography>
      )}

      <Dialog open={open} onClose={() => setOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>권한 신청</DialogTitle>
        <DialogContent>
          <FormControl fullWidth sx={{ mt: 1, mb: 2 }}>
            <InputLabel>요청 역할</InputLabel>
            <Select value={role} onChange={(e) => setRole(e.target.value)} label="요청 역할">
              <MenuItem value="manager">Manager</MenuItem>
              <MenuItem value="admin">Admin</MenuItem>
            </Select>
          </FormControl>
          <TextField
            label="신청 사유"
            multiline
            rows={3}
            fullWidth
            value={reason}
            onChange={(e) => setReason(e.target.value)}
            helperText="최소 10자 이상 입력해 주세요"
          />
          {createMutation.isError && (
            <Alert severity="error" sx={{ mt: 1 }}>
              {(createMutation.error as any)?.response?.data?.detail || '신청 실패'}
            </Alert>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setOpen(false)}>취소</Button>
          <Button
            variant="contained"
            disabled={reason.length < 10 || createMutation.isPending}
            onClick={() => createMutation.mutate()}
          >
            {createMutation.isPending ? '처리 중...' : '신청'}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}


// ===== Admin: Access Request Management =====

export function AdminAccessRequestPanel() {
  const [reviewDialog, setReviewDialog] = useState<{ id: string; username: string } | null>(null);
  const [decision, setDecision] = useState<'approved' | 'rejected'>('approved');
  const [comment, setComment] = useState('');
  const queryClient = useQueryClient();

  const { data: pendingRequests } = useQuery({
    queryKey: ['admin-access-requests'],
    queryFn: () => api.get('/auth/admin/access-requests').then(r => r.data),
    refetchInterval: 30000,
  });

  const reviewMutation = useMutation({
    mutationFn: ({ id, decision, comment }: { id: string; decision: string; comment: string }) =>
      api.put(`/auth/admin/access-requests/${id}`, { decision, comment }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['admin-access-requests'] });
      setReviewDialog(null);
      setComment('');
    },
  });

  return (
    <Box>
      <Typography variant="h6" gutterBottom>권한 승인 관리</Typography>

      {!pendingRequests?.requests?.length ? (
        <Alert severity="info">대기 중인 신청이 없습니다.</Alert>
      ) : (
        <TableContainer>
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>사용자</TableCell>
                <TableCell>현재 역할</TableCell>
                <TableCell>요청 역할</TableCell>
                <TableCell>사유</TableCell>
                <TableCell>신청일</TableCell>
                <TableCell>작업</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {pendingRequests.requests.map((r: any) => (
                <TableRow key={r.id}>
                  <TableCell>{r.username}</TableCell>
                  <TableCell>{r.current_role}</TableCell>
                  <TableCell>
                    <Chip label={r.requested_role} color="primary" size="small" />
                  </TableCell>
                  <TableCell sx={{ maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis' }}>
                    {r.reason}
                  </TableCell>
                  <TableCell>{new Date(r.created_at).toLocaleDateString('ko-KR')}</TableCell>
                  <TableCell>
                    <Button
                      size="small"
                      variant="outlined"
                      onClick={() => setReviewDialog({ id: r.id, username: r.username })}
                    >
                      검토
                    </Button>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      )}

      <Dialog open={!!reviewDialog} onClose={() => setReviewDialog(null)} maxWidth="sm" fullWidth>
        <DialogTitle>권한 신청 검토 - {reviewDialog?.username}</DialogTitle>
        <DialogContent>
          <FormControl fullWidth sx={{ mt: 1, mb: 2 }}>
            <InputLabel>결정</InputLabel>
            <Select value={decision} onChange={(e) => setDecision(e.target.value as any)} label="결정">
              <MenuItem value="approved">승인</MenuItem>
              <MenuItem value="rejected">거절</MenuItem>
            </Select>
          </FormControl>
          <TextField
            label="검토 의견"
            multiline
            rows={2}
            fullWidth
            value={comment}
            onChange={(e) => setComment(e.target.value)}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setReviewDialog(null)}>취소</Button>
          <Button
            variant="contained"
            color={decision === 'approved' ? 'success' : 'error'}
            disabled={reviewMutation.isPending}
            onClick={() => {
              if (reviewDialog) {
                reviewMutation.mutate({
                  id: reviewDialog.id,
                  decision,
                  comment,
                });
              }
            }}
          >
            {decision === 'approved' ? '승인' : '거절'}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}
