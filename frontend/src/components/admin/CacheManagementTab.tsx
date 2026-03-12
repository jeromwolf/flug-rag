import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  Box,
  Typography,
  Card,
  CardContent,
  CircularProgress,
  Alert,
  TextField,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Snackbar,
  IconButton,
  Tooltip,
} from "@mui/material";
import CachedIcon from "@mui/icons-material/Cached";
import DeleteSweepIcon from "@mui/icons-material/DeleteSweep";
import { adminApi } from "../../api/client";

interface CacheCategory {
  key: string;
  label: string;
  description: string;
  unit: string;
}

interface CacheConfigData {
  cache_enabled: boolean;
  cache_type: string;
  ttl: Record<string, number>;
  categories: CacheCategory[];
}

function CacheManagementTab() {
  const queryClient = useQueryClient();
  const [localTtl, setLocalTtl] = useState<Record<string, number> | null>(null);
  const [snack, setSnack] = useState<{ open: boolean; message: string; severity: "success" | "error" }>({
    open: false,
    message: "",
    severity: "success",
  });
  const [clearConfirmCategory, setClearConfirmCategory] = useState<string | null>(null);

  const { data, isLoading, error } = useQuery({
    queryKey: ["admin-cache-config"],
    queryFn: () => adminApi.getCacheConfig(),
    select: (res) => res.data as CacheConfigData,
  });

  const config = data;
  const effectiveTtl = localTtl ?? config?.ttl ?? {};

  const updateMutation = useMutation({
    mutationFn: (payload: Record<string, number>) => adminApi.updateCacheConfig(payload),
    onSuccess: () => {
      setSnack({ open: true, message: "캐시 TTL이 저장되었습니다.", severity: "success" });
      queryClient.invalidateQueries({ queryKey: ["admin-cache-config"] });
      setLocalTtl(null);
    },
    onError: () => {
      setSnack({ open: true, message: "저장에 실패했습니다.", severity: "error" });
    },
  });

  const clearMutation = useMutation({
    mutationFn: (category?: string) => adminApi.clearCache(category),
    onSuccess: (_, category) => {
      setSnack({
        open: true,
        message: category ? `'${category}' 캐시가 초기화되었습니다.` : "전체 캐시가 초기화되었습니다.",
        severity: "success",
      });
      setClearConfirmCategory(null);
    },
    onError: () => {
      setSnack({ open: true, message: "캐시 초기화에 실패했습니다.", severity: "error" });
      setClearConfirmCategory(null);
    },
  });

  const handleSave = () => {
    if (!localTtl) return;
    const numeric: Record<string, number> = {};
    for (const [k, v] of Object.entries(localTtl)) {
      numeric[k] = Number(v);
    }
    updateMutation.mutate(numeric);
  };

  const formatTtl = (seconds: number): string => {
    if (seconds === 0) return "비활성화";
    if (seconds < 60) return `${seconds}초`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)}분 ${seconds % 60}초`;
    if (seconds < 86400) return `${Math.floor(seconds / 3600)}시간 ${Math.floor((seconds % 3600) / 60)}분`;
    return `${Math.floor(seconds / 86400)}일`;
  };

  if (isLoading) return <CircularProgress />;
  if (error || !config) return <Alert severity="error">캐시 설정을 불러올 수 없습니다.</Alert>;

  return (
    <>
      <Box sx={{ mb: 3, display: "flex", alignItems: "center", gap: 2 }}>
        <Box>
          <Typography variant="h6" fontWeight={600}>캐시 관리</Typography>
          <Typography variant="body2" color="text.secondary">
            캐시 유형: <strong>{config.cache_type === "memory" ? "인메모리" : "Redis"}</strong>
            {" · "}
            캐시 활성화: <strong>{config.cache_enabled ? "활성" : "비활성"}</strong>
          </Typography>
        </Box>
        {!config.cache_enabled && (
          <Alert severity="info" sx={{ flex: 1 }}>
            캐시가 비활성화 상태입니다. TTL 설정은 저장되지만 캐시가 활성화될 때 적용됩니다.
          </Alert>
        )}
      </Box>

      {/* TTL Settings per Category */}
      <Typography variant="subtitle1" fontWeight={600} gutterBottom>
        카테고리별 TTL 설정
      </Typography>
      <Card variant="outlined" sx={{ mb: 3 }}>
        <CardContent sx={{ display: "flex", flexDirection: "column", gap: 2 }}>
          {(config.categories ?? []).map((cat) => (
            <Box key={cat.key} sx={{ display: "flex", alignItems: "flex-start", gap: 2, flexWrap: "wrap" }}>
              <Box sx={{ flex: "1 1 220px" }}>
                <Typography variant="body2" fontWeight={500}>{cat.label}</Typography>
                <Typography variant="caption" color="text.secondary">{cat.description}</Typography>
              </Box>
              <Box sx={{ display: "flex", alignItems: "center", gap: 1, flex: "0 0 auto" }}>
                <TextField
                  size="small"
                  type="number"
                  label={`TTL (${cat.unit})`}
                  value={effectiveTtl[cat.key] ?? 0}
                  onChange={(e) =>
                    setLocalTtl({ ...effectiveTtl, [cat.key]: Number(e.target.value) })
                  }
                  inputProps={{ min: 0 }}
                  sx={{ width: 140 }}
                />
                <Typography variant="caption" color="text.secondary" sx={{ minWidth: 80 }}>
                  {formatTtl(effectiveTtl[cat.key] ?? 0)}
                </Typography>
                <Tooltip title={`'${cat.label}' 캐시만 초기화`}>
                  <IconButton
                    size="small"
                    color="warning"
                    onClick={() => setClearConfirmCategory(cat.key)}
                    disabled={clearMutation.isPending}
                  >
                    <DeleteSweepIcon fontSize="small" />
                  </IconButton>
                </Tooltip>
              </Box>
            </Box>
          ))}

          <Box sx={{ display: "flex", gap: 1, mt: 1 }}>
            <Button
              variant="contained"
              onClick={handleSave}
              disabled={!localTtl || updateMutation.isPending}
              startIcon={updateMutation.isPending ? <CircularProgress size={18} /> : undefined}
            >
              저장
            </Button>
            <Button
              variant="outlined"
              onClick={() => setLocalTtl(null)}
              disabled={!localTtl}
            >
              초기화
            </Button>
          </Box>
        </CardContent>
      </Card>

      {/* Global Cache Clear */}
      <Typography variant="subtitle1" fontWeight={600} gutterBottom>
        전체 캐시 초기화
      </Typography>
      <Card variant="outlined">
        <CardContent sx={{ display: "flex", alignItems: "center", gap: 2 }}>
          <Box sx={{ flex: 1 }}>
            <Typography variant="body2">
              모든 캐시 항목을 즉시 삭제합니다. 이후 요청은 캐시 없이 처리됩니다.
            </Typography>
          </Box>
          <Button
            variant="outlined"
            color="error"
            startIcon={<CachedIcon />}
            onClick={() => setClearConfirmCategory("__all__")}
            disabled={clearMutation.isPending}
          >
            전체 캐시 초기화
          </Button>
        </CardContent>
      </Card>

      {/* Confirm Dialog */}
      <Dialog
        open={clearConfirmCategory !== null}
        onClose={() => setClearConfirmCategory(null)}
        maxWidth="xs"
        fullWidth
      >
        <DialogTitle>캐시 초기화 확인</DialogTitle>
        <DialogContent>
          <Typography>
            {clearConfirmCategory === "__all__"
              ? "전체 캐시를 초기화하시겠습니까?"
              : `'${config.categories.find((c) => c.key === clearConfirmCategory)?.label ?? clearConfirmCategory}' 캐시를 초기화하시겠습니까?`}
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setClearConfirmCategory(null)}>취소</Button>
          <Button
            variant="contained"
            color="error"
            disabled={clearMutation.isPending}
            onClick={() => {
              if (clearConfirmCategory === "__all__") {
                clearMutation.mutate(undefined);
              } else if (clearConfirmCategory) {
                clearMutation.mutate(clearConfirmCategory);
              }
            }}
          >
            {clearMutation.isPending ? <CircularProgress size={18} /> : "초기화"}
          </Button>
        </DialogActions>
      </Dialog>

      <Snackbar
        open={snack.open}
        autoHideDuration={3000}
        onClose={() => setSnack((s) => ({ ...s, open: false }))}
        anchorOrigin={{ vertical: "bottom", horizontal: "center" }}
      >
        <Alert severity={snack.severity} onClose={() => setSnack((s) => ({ ...s, open: false }))}>
          {snack.message}
        </Alert>
      </Snackbar>
    </>
  );
}

export default CacheManagementTab;
