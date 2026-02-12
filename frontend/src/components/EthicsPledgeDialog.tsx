import { useState } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Checkbox,
  FormControlLabel,
  Typography,
  Box,
  Alert,
  CircularProgress,
} from '@mui/material';
import { useMutation, useQuery } from '@tanstack/react-query';
import api from '../api/client';

interface EthicsPledgeDialogProps {
  open: boolean;
  onAgreed: () => void;
}

export default function EthicsPledgeDialog({ open, onAgreed }: EthicsPledgeDialogProps) {
  const [checked, setChecked] = useState(false);

  const { data: pledge, isLoading } = useQuery({
    queryKey: ['ethics-pledge'],
    queryFn: () => api.get('/auth/ethics-pledge').then(r => r.data),
    enabled: open,
  });

  const agreeMutation = useMutation({
    mutationFn: () => api.post('/auth/ethics-pledge/agree'),
    onSuccess: () => onAgreed(),
  });

  return (
    <Dialog open={open} maxWidth="md" fullWidth disableEscapeKeyDown>
      <DialogTitle sx={{ fontWeight: 'bold' }}>
        AI 윤리 서약
      </DialogTitle>
      <DialogContent dividers>
        {isLoading ? (
          <Box display="flex" justifyContent="center" p={4}>
            <CircularProgress />
          </Box>
        ) : (
          <>
            <Alert severity="info" sx={{ mb: 2 }}>
              서비스 이용을 위해 아래 윤리 서약에 동의해 주세요.
            </Alert>
            <Box
              sx={{
                maxHeight: 400,
                overflow: 'auto',
                p: 2,
                bgcolor: 'grey.50',
                borderRadius: 1,
                whiteSpace: 'pre-wrap',
                fontFamily: 'inherit',
                fontSize: '0.9rem',
                lineHeight: 1.8,
              }}
            >
              <Typography variant="body2" component="div">
                {pledge?.content || ''}
              </Typography>
            </Box>
            <FormControlLabel
              control={
                <Checkbox
                  checked={checked}
                  onChange={(e) => setChecked(e.target.checked)}
                />
              }
              label="위 내용을 읽고 동의합니다"
              sx={{ mt: 2 }}
            />
          </>
        )}
      </DialogContent>
      <DialogActions>
        <Button
          variant="contained"
          disabled={!checked || agreeMutation.isPending}
          onClick={() => agreeMutation.mutate()}
        >
          {agreeMutation.isPending ? '처리 중...' : '동의합니다'}
        </Button>
      </DialogActions>
    </Dialog>
  );
}
