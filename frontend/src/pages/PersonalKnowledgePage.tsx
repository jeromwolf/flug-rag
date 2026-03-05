import { useState, useCallback } from "react";
import {
  Box,
  Button,
  Paper,
  Chip,
  IconButton,
  Tooltip,
  Typography,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  LinearProgress,
  CircularProgress,
  Snackbar,
  Alert,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogContentText,
  DialogActions,
  Collapse,
  Stack,
  Divider,
} from "@mui/material";
import {
  CloudUpload as UploadIcon,
  Delete as DeleteIcon,
  InsertDriveFile as FileIcon,
  FolderOpen as FolderOpenIcon,
  CheckCircle as CheckCircleIcon,
  Error as ErrorIcon,
  HourglassEmpty as PendingIcon,
  ExpandMore as ExpandMoreIcon,
  ExpandLess as ExpandLessIcon,
  Policy as PolicyIcon,
  Warning as WarningIcon,
} from "@mui/icons-material";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useDropzone } from "react-dropzone";
import { personalKnowledgeApi } from "../api/client";
import Layout from "../components/Layout";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface PersonalDoc {
  document_id: string;
  filename: string;
  file_type: string;
  chunk_count: number;
  uploaded_at: string;
  pii_warnings: string[];
}

interface UploadQueueItem {
  file: File;
  status: "pending" | "uploading" | "done" | "error";
  errorMessage?: string;
  piiWarnings?: string[];
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const ACCEPTED_TYPES: Record<string, string[]> = {
  "application/pdf": [".pdf"],
  "application/vnd.hancom.hwp": [".hwp"],
  "application/vnd.openxmlformats-officedocument.wordprocessingml.document": [".docx"],
  "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": [".xlsx"],
  "application/vnd.openxmlformats-officedocument.presentationml.presentation": [".pptx"],
  "text/plain": [".txt"],
};

function formatDate(dateStr: string): string {
  const d = new Date(dateStr);
  if (isNaN(d.getTime())) return "-";
  return d.toLocaleDateString("ko-KR", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function getFileTypeChipColor(ext: string): "default" | "primary" | "secondary" | "error" | "info" | "success" | "warning" {
  const upper = ext.toUpperCase().replace(".", "");
  switch (upper) {
    case "PDF": return "error";
    case "HWP": return "primary";
    case "DOCX": return "info";
    case "XLSX": return "success";
    case "PPTX": return "warning";
    default: return "default";
  }
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export default function PersonalKnowledgePage() {
  const queryClient = useQueryClient();

  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadQueue, setUploadQueue] = useState<UploadQueueItem[]>([]);
  const [queueExpanded, setQueueExpanded] = useState(true);
  const [deleteDialogId, setDeleteDialogId] = useState<string | null>(null);
  const [piiDialogDoc, setPiiDialogDoc] = useState<PersonalDoc | null>(null);
  const [snackbar, setSnackbar] = useState<{
    open: boolean;
    message: string;
    severity: "success" | "error" | "info" | "warning";
  }>({ open: false, message: "", severity: "info" });

  // -----------------------------------------------------------------------
  // Queries
  // -----------------------------------------------------------------------

  const { data, isLoading } = useQuery({
    queryKey: ["personal-knowledge"],
    queryFn: async () => {
      const res = await personalKnowledgeApi.list();
      return res.data as { documents: PersonalDoc[]; total: number };
    },
  });

  const documents: PersonalDoc[] = data?.documents ?? [];

  // -----------------------------------------------------------------------
  // Mutations
  // -----------------------------------------------------------------------

  const deleteMutation = useMutation({
    mutationFn: (docId: string) => personalKnowledgeApi.delete(docId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["personal-knowledge"] });
      setDeleteDialogId(null);
      setSnackbar({ open: true, message: "문서가 삭제되었습니다.", severity: "success" });
    },
    onError: () => {
      setSnackbar({ open: true, message: "문서 삭제에 실패했습니다.", severity: "error" });
    },
  });

  const piiScanMutation = useMutation({
    mutationFn: (docId: string) => personalKnowledgeApi.piiScan(docId),
    onSuccess: (res) => {
      setSnackbar({
        open: true,
        message: res.data.has_pii
          ? `PII 탐지: ${res.data.total_matches}건 발견`
          : "PII 탐지: 개인정보 없음",
        severity: res.data.has_pii ? "warning" : "success",
      });
    },
    onError: () => {
      setSnackbar({ open: true, message: "PII 스캔에 실패했습니다.", severity: "error" });
    },
  });

  // -----------------------------------------------------------------------
  // Upload
  // -----------------------------------------------------------------------

  const handleUpload = useCallback(
    async (files: File[]) => {
      if (files.length === 0) return;

      const initialQueue: UploadQueueItem[] = files.map((f) => ({
        file: f,
        status: "pending",
      }));
      setUploadQueue(initialQueue);
      setQueueExpanded(true);
      setUploading(true);
      setUploadProgress(0);

      let completed = 0;
      let failed = 0;

      for (let i = 0; i < files.length; i++) {
        setUploadQueue((prev) =>
          prev.map((item, idx) => (idx === i ? { ...item, status: "uploading" } : item))
        );

        try {
          const res = await personalKnowledgeApi.upload(files[i]);
          completed++;
          const piiWarnings: string[] = res.data?.pii_warnings ?? [];
          setUploadQueue((prev) =>
            prev.map((item, idx) =>
              idx === i ? { ...item, status: "done", piiWarnings } : item
            )
          );
        } catch (err: unknown) {
          failed++;
          const errorMessage = err instanceof Error ? err.message : "업로드 실패";
          setUploadQueue((prev) =>
            prev.map((item, idx) =>
              idx === i ? { ...item, status: "error", errorMessage } : item
            )
          );
        }

        setUploadProgress(Math.round(((completed + failed) / files.length) * 100));
      }

      setUploading(false);
      queryClient.invalidateQueries({ queryKey: ["personal-knowledge"] });

      if (failed > 0) {
        setSnackbar({
          open: true,
          message: `${completed}개 업로드 완료, ${failed}개 실패`,
          severity: failed === files.length ? "error" : "info",
        });
      } else {
        setSnackbar({
          open: true,
          message: `${completed}개 문서가 업로드되었습니다.`,
          severity: "success",
        });
        setTimeout(() => setUploadQueue([]), 3000);
      }
    },
    [queryClient]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop: handleUpload,
    accept: ACCEPTED_TYPES,
    disabled: uploading,
  });

  // -----------------------------------------------------------------------
  // Render
  // -----------------------------------------------------------------------

  return (
    <Layout title="개인 지식공간">
      {/* Header actions */}
      <Box sx={{ display: "flex", justifyContent: "flex-end", mb: 2 }}>
        <Button
          variant="contained"
          startIcon={<UploadIcon />}
          component="label"
          disabled={uploading}
        >
          파일 업로드
          <input
            type="file"
            hidden
            multiple
            accept=".pdf,.hwp,.docx,.xlsx,.pptx,.txt"
            onChange={(e) => {
              if (e.target.files) {
                handleUpload(Array.from(e.target.files));
                e.target.value = "";
              }
            }}
          />
        </Button>
      </Box>

      {/* Info banner */}
      <Paper
        variant="outlined"
        sx={{
          mb: 2,
          p: 1.5,
          display: "flex",
          alignItems: "center",
          gap: 1,
          bgcolor: "info.50",
          borderColor: "info.200",
        }}
      >
        <PolicyIcon fontSize="small" color="info" />
        <Typography variant="body2" color="text.secondary">
          개인 지식공간에 업로드한 문서는 본인만 접근할 수 있습니다. 민감한 개인정보(PII)는 자동으로 탐지됩니다.
        </Typography>
      </Paper>

      {/* Drop zone */}
      <Paper
        {...getRootProps()}
        variant="outlined"
        sx={{
          mb: 2,
          p: 4,
          textAlign: "center",
          cursor: uploading ? "default" : "pointer",
          borderStyle: "dashed",
          borderWidth: 2,
          borderColor: isDragActive ? "primary.main" : "divider",
          bgcolor: isDragActive ? "action.hover" : "transparent",
          transition: "all 0.2s",
          "&:hover": uploading ? {} : { borderColor: "primary.main", bgcolor: "action.hover" },
        }}
      >
        <input {...getInputProps()} />
        <UploadIcon sx={{ fontSize: 48, color: "text.secondary", mb: 1 }} />
        <Typography variant="body1" color="text.secondary">
          {isDragActive
            ? "파일을 여기에 놓으세요"
            : "파일을 드래그하거나 클릭하여 선택"}
        </Typography>
        <Typography variant="caption" color="text.secondary">
          지원 형식: PDF, HWP, DOCX, XLSX, PPTX, TXT
        </Typography>
      </Paper>

      {/* Upload queue */}
      {uploadQueue.length > 0 && (
        <Box sx={{ mb: 2 }}>
          <Paper variant="outlined" sx={{ overflow: "hidden" }}>
            <Box
              sx={{
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                px: 2,
                py: 1,
                bgcolor: "grey.50",
                borderBottom: queueExpanded ? "1px solid" : "none",
                borderColor: "divider",
                cursor: "pointer",
                userSelect: "none",
              }}
              onClick={() => setQueueExpanded((v) => !v)}
            >
              <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                <Typography variant="body2" fontWeight={600}>
                  업로드 현황
                </Typography>
                <Chip
                  size="small"
                  label={`${uploadQueue.filter((q) => q.status === "done").length} / ${uploadQueue.length}`}
                  color={
                    uploading
                      ? "info"
                      : uploadQueue.some((q) => q.status === "error")
                      ? "warning"
                      : "success"
                  }
                />
              </Box>
              <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                {!uploading && (
                  <Button
                    size="small"
                    variant="text"
                    onClick={(e) => {
                      e.stopPropagation();
                      setUploadQueue([]);
                    }}
                  >
                    닫기
                  </Button>
                )}
                <IconButton size="small">
                  {queueExpanded ? <ExpandLessIcon fontSize="small" /> : <ExpandMoreIcon fontSize="small" />}
                </IconButton>
              </Box>
            </Box>

            {uploading && (
              <LinearProgress variant="determinate" value={uploadProgress} sx={{ height: 3 }} />
            )}

            <Collapse in={queueExpanded}>
              <Box sx={{ maxHeight: 240, overflowY: "auto" }}>
                {uploadQueue.map((item, idx) => (
                  <Box
                    key={idx}
                    sx={{
                      display: "flex",
                      alignItems: "center",
                      gap: 1.5,
                      px: 2,
                      py: 0.75,
                      borderBottom: idx < uploadQueue.length - 1 ? "1px solid" : "none",
                      borderColor: "divider",
                      bgcolor:
                        item.status === "error"
                          ? "error.50"
                          : item.status === "done"
                          ? "success.50"
                          : "transparent",
                    }}
                  >
                    <Box sx={{ flexShrink: 0, display: "flex" }}>
                      {item.status === "pending" && <PendingIcon fontSize="small" sx={{ color: "text.disabled" }} />}
                      {item.status === "uploading" && <CircularProgress size={16} thickness={5} />}
                      {item.status === "done" && <CheckCircleIcon fontSize="small" color="success" />}
                      {item.status === "error" && <ErrorIcon fontSize="small" color="error" />}
                    </Box>
                    <Box sx={{ flex: 1, minWidth: 0 }}>
                      <Typography
                        variant="body2"
                        noWrap
                        sx={{ color: item.status === "error" ? "error.main" : "text.primary" }}
                      >
                        {item.file.name}
                      </Typography>
                      {item.errorMessage && (
                        <Typography variant="caption" color="error" noWrap>
                          {item.errorMessage}
                        </Typography>
                      )}
                      {item.piiWarnings && item.piiWarnings.length > 0 && (
                        <Typography variant="caption" color="warning.main" noWrap>
                          PII 경고: {item.piiWarnings.join(", ")}
                        </Typography>
                      )}
                    </Box>
                    <Chip
                      size="small"
                      label={
                        item.status === "pending"
                          ? "대기"
                          : item.status === "uploading"
                          ? "업로드 중"
                          : item.status === "done"
                          ? "완료"
                          : "실패"
                      }
                      color={
                        item.status === "pending"
                          ? "default"
                          : item.status === "uploading"
                          ? "info"
                          : item.status === "done"
                          ? "success"
                          : "error"
                      }
                      sx={{ flexShrink: 0, minWidth: 64 }}
                    />
                  </Box>
                ))}
              </Box>
            </Collapse>
          </Paper>
        </Box>
      )}

      {/* Summary */}
      {!isLoading && documents.length > 0 && (
        <Box sx={{ mb: 1.5 }}>
          <Stack direction="row" spacing={1} alignItems="center">
            <Chip size="small" label={`총 ${documents.length}개 문서`} color="default" sx={{ fontWeight: 600 }} />
            <Chip
              size="small"
              label={`${documents.reduce((s, d) => s + d.chunk_count, 0)} 청크`}
              color="default"
              variant="outlined"
            />
          </Stack>
        </Box>
      )}

      {/* Document list */}
      {isLoading ? (
        <Box sx={{ display: "flex", justifyContent: "center", py: 4 }}>
          <LinearProgress sx={{ width: "50%" }} />
        </Box>
      ) : documents.length === 0 ? (
        <Box
          sx={{
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
            py: 8,
            gap: 2,
          }}
        >
          <FolderOpenIcon sx={{ fontSize: 64, color: "text.secondary", opacity: 0.4 }} />
          <Typography variant="h6" color="text.secondary">
            개인 문서가 없습니다
          </Typography>
          <Typography variant="body2" color="text.secondary">
            위의 업로드 영역에 파일을 드래그하여 나만의 지식을 추가하세요.
          </Typography>
        </Box>
      ) : (
        <Paper variant="outlined">
          <TableContainer>
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>파일명</TableCell>
                  <TableCell>유형</TableCell>
                  <TableCell align="right">청크수</TableCell>
                  <TableCell>업로드일</TableCell>
                  <TableCell>PII</TableCell>
                  <TableCell align="center">액션</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {documents.map((doc) => {
                  const extLabel = (doc.file_type || "").toUpperCase().replace(".", "");
                  const hasPii = doc.pii_warnings && doc.pii_warnings.length > 0;
                  return (
                    <TableRow key={doc.document_id} hover>
                      <TableCell>
                        <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                          <FileIcon fontSize="small" color="action" />
                          <Typography variant="body2" noWrap sx={{ maxWidth: 300 }}>
                            {doc.filename}
                          </Typography>
                        </Box>
                      </TableCell>
                      <TableCell>
                        <Chip
                          label={extLabel || "-"}
                          size="small"
                          color={getFileTypeChipColor(doc.file_type)}
                          variant="outlined"
                        />
                      </TableCell>
                      <TableCell align="right">{doc.chunk_count}</TableCell>
                      <TableCell>{formatDate(doc.uploaded_at)}</TableCell>
                      <TableCell>
                        {hasPii ? (
                          <Tooltip title={doc.pii_warnings.join(", ")}>
                            <Chip
                              icon={<WarningIcon />}
                              label="경고"
                              size="small"
                              color="warning"
                              variant="outlined"
                            />
                          </Tooltip>
                        ) : (
                          <Chip label="없음" size="small" color="success" variant="outlined" />
                        )}
                      </TableCell>
                      <TableCell align="center">
                        <Box sx={{ display: "flex", gap: 0.5, justifyContent: "center" }}>
                          <Tooltip title="PII 재스캔">
                            <IconButton
                              size="small"
                              color="info"
                              disabled={piiScanMutation.isPending}
                              onClick={() => {
                                piiScanMutation.mutate(doc.document_id);
                                setPiiDialogDoc(doc);
                              }}
                            >
                              <PolicyIcon fontSize="small" />
                            </IconButton>
                          </Tooltip>
                          <Tooltip title="삭제">
                            <IconButton
                              size="small"
                              color="error"
                              onClick={() => setDeleteDialogId(doc.document_id)}
                            >
                              <DeleteIcon fontSize="small" />
                            </IconButton>
                          </Tooltip>
                        </Box>
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          </TableContainer>
        </Paper>
      )}

      {/* PII dialog result */}
      <Dialog
        open={piiDialogDoc !== null && !piiScanMutation.isPending && piiScanMutation.isSuccess}
        onClose={() => {
          setPiiDialogDoc(null);
          piiScanMutation.reset();
        }}
      >
        <DialogTitle>PII 스캔 결과</DialogTitle>
        <DialogContent>
          {piiScanMutation.data && (
            <Box sx={{ display: "flex", flexDirection: "column", gap: 1.5 }}>
              <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                <Typography variant="body2" fontWeight={600}>
                  파일:
                </Typography>
                <Typography variant="body2">{piiDialogDoc?.filename}</Typography>
              </Box>
              <Divider />
              <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                {piiScanMutation.data.data.has_pii ? (
                  <WarningIcon color="warning" />
                ) : (
                  <CheckCircleIcon color="success" />
                )}
                <Typography variant="body2">
                  {piiScanMutation.data.data.has_pii
                    ? `개인정보 ${piiScanMutation.data.data.total_matches}건 탐지 (${piiScanMutation.data.data.chunks_scanned} 청크 검사)`
                    : `개인정보 없음 (${piiScanMutation.data.data.chunks_scanned} 청크 검사 완료)`}
                </Typography>
              </Box>
              {piiScanMutation.data.data.warnings && piiScanMutation.data.data.warnings.length > 0 && (
                <Box>
                  <Typography variant="caption" color="text.secondary" gutterBottom>
                    탐지 유형:
                  </Typography>
                  <Stack direction="row" flexWrap="wrap" gap={0.5} sx={{ mt: 0.5 }}>
                    {piiScanMutation.data.data.warnings.map((w: string) => (
                      <Chip key={w} label={w} size="small" color="warning" variant="outlined" />
                    ))}
                  </Stack>
                </Box>
              )}
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button
            onClick={() => {
              setPiiDialogDoc(null);
              piiScanMutation.reset();
            }}
          >
            닫기
          </Button>
        </DialogActions>
      </Dialog>

      {/* Delete confirmation */}
      <Dialog open={deleteDialogId !== null} onClose={() => setDeleteDialogId(null)}>
        <DialogTitle>문서 삭제</DialogTitle>
        <DialogContent>
          <DialogContentText>
            이 문서를 삭제하시겠습니까? 관련된 모든 청크도 함께 삭제됩니다.
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDeleteDialogId(null)}>취소</Button>
          <Button
            color="error"
            variant="contained"
            onClick={() => {
              if (deleteDialogId) deleteMutation.mutate(deleteDialogId);
            }}
          >
            삭제
          </Button>
        </DialogActions>
      </Dialog>

      {/* Snackbar */}
      <Snackbar
        open={snackbar.open}
        autoHideDuration={3000}
        onClose={() => setSnackbar((prev) => ({ ...prev, open: false }))}
        anchorOrigin={{ vertical: "bottom", horizontal: "center" }}
      >
        <Alert
          severity={snackbar.severity}
          onClose={() => setSnackbar((prev) => ({ ...prev, open: false }))}
        >
          {snackbar.message}
        </Alert>
      </Snackbar>
    </Layout>
  );
}
