import { useState, useCallback } from "react";
import {
  Box,
  Drawer,
  List,
  ListItemButton,
  ListItemText,
  ListItemIcon,
  Typography,
  Button,
  Paper,
  Chip,
  IconButton,
  Tooltip,
  Divider,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  LinearProgress,
  Snackbar,
  Alert,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogContentText,
  DialogActions,
} from "@mui/material";
import {
  Chat as ChatIcon,
  Description as DescriptionIcon,
  AdminPanelSettings as AdminIcon,
  Monitor as MonitorIcon,
  CloudUpload as UploadIcon,
  Delete as DeleteIcon,
  InsertDriveFile as FileIcon,
  Close as CloseIcon,
  FolderOpen as FolderOpenIcon,
} from "@mui/icons-material";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { useDropzone } from "react-dropzone";
import { documentsApi } from "../api/client";
import { useAppStore } from "../stores/appStore";

const SIDEBAR_WIDTH = 280;

const ACCEPTED_TYPES: Record<string, string[]> = {
  "application/pdf": [".pdf"],
  "application/vnd.hancom.hwp": [".hwp"],
  "application/vnd.openxmlformats-officedocument.wordprocessingml.document": [
    ".docx",
  ],
  "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": [
    ".xlsx",
  ],
  "application/vnd.openxmlformats-officedocument.presentationml.presentation": [
    ".pptx",
  ],
  "text/plain": [".txt"],
};

function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatDate(dateStr: string): string {
  const d = new Date(dateStr);
  return d.toLocaleDateString("ko-KR", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

interface DocumentRow {
  id: string;
  filename: string;
  file_type?: string;
  fileType?: string;
  file_size?: number;
  fileSize?: number;
  chunk_count?: number;
  chunkCount?: number;
  uploaded_at?: string;
  uploadDate?: string;
  status?: string;
  metadata?: Record<string, unknown>;
}

function getStatusChip(status: string | undefined) {
  switch (status) {
    case "completed":
      return <Chip label="완료" color="success" size="small" />;
    case "processing":
      return <Chip label="처리 중" color="info" size="small" />;
    case "failed":
      return <Chip label="실패" color="error" size="small" />;
    case "pending":
      return <Chip label="대기" color="default" size="small" />;
    default:
      return <Chip label="완료" color="success" size="small" />;
  }
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export default function DocumentsPage() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const { sidebarOpen } = useAppStore();

  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [deleteDialogId, setDeleteDialogId] = useState<string | null>(null);
  const [detailDoc, setDetailDoc] = useState<DocumentRow | null>(null);
  const [snackbar, setSnackbar] = useState<{
    open: boolean;
    message: string;
    severity: "success" | "error" | "info";
  }>({ open: false, message: "", severity: "info" });

  // -----------------------------------------------------------------------
  // Queries
  // -----------------------------------------------------------------------

  const { data: docsData, isLoading } = useQuery({
    queryKey: ["documents"],
    queryFn: async () => {
      const res = await documentsApi.list();
      return res.data;
    },
  });

  const documents: DocumentRow[] = docsData?.documents ?? [];

  // -----------------------------------------------------------------------
  // Mutations
  // -----------------------------------------------------------------------

  const deleteMutation = useMutation({
    mutationFn: (id: string) => documentsApi.delete(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["documents"] });
      setDeleteDialogId(null);
      setSnackbar({
        open: true,
        message: "문서가 삭제되었습니다.",
        severity: "success",
      });
    },
    onError: () => {
      setSnackbar({
        open: true,
        message: "문서 삭제에 실패했습니다.",
        severity: "error",
      });
    },
  });

  // -----------------------------------------------------------------------
  // Upload
  // -----------------------------------------------------------------------

  const handleUpload = useCallback(
    async (files: File[]) => {
      if (files.length === 0) return;
      setUploading(true);
      setUploadProgress(0);

      let completed = 0;
      let failed = 0;

      for (const file of files) {
        try {
          await documentsApi.upload(file);
          completed++;
        } catch {
          failed++;
        }
        setUploadProgress(Math.round(((completed + failed) / files.length) * 100));
      }

      setUploading(false);
      setUploadProgress(0);
      queryClient.invalidateQueries({ queryKey: ["documents"] });

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
  // Sidebar (shared structure with ChatPage)
  // -----------------------------------------------------------------------

  const sidebarContent = (
    <Box
      sx={{
        width: SIDEBAR_WIDTH,
        height: "100%",
        display: "flex",
        flexDirection: "column",
        bgcolor: "grey.50",
      }}
    >
      {/* Logo */}
      <Box sx={{ p: 2, display: "flex", alignItems: "center", gap: 1 }}>
        <DescriptionIcon color="primary" />
        <Typography variant="h6" fontWeight={700}>
          Flux RAG
        </Typography>
      </Box>

      <Divider />

      <Box sx={{ flex: 1 }} />

      <Divider />

      {/* Bottom nav */}
      <List dense>
        <ListItemButton
          sx={{ borderRadius: 1, mx: 1 }}
          onClick={() => navigate("/chat")}
        >
          <ListItemIcon sx={{ minWidth: 36 }}>
            <ChatIcon fontSize="small" />
          </ListItemIcon>
          <ListItemText
            primary="채팅"
            primaryTypographyProps={{ variant: "body2" }}
          />
        </ListItemButton>
        <ListItemButton selected sx={{ borderRadius: 1, mx: 1 }}>
          <ListItemIcon sx={{ minWidth: 36 }}>
            <DescriptionIcon fontSize="small" />
          </ListItemIcon>
          <ListItemText
            primary="문서"
            primaryTypographyProps={{ variant: "body2" }}
          />
        </ListItemButton>
        <ListItemButton
          sx={{ borderRadius: 1, mx: 1 }}
          onClick={() => navigate("/admin")}
        >
          <ListItemIcon sx={{ minWidth: 36 }}>
            <AdminIcon fontSize="small" />
          </ListItemIcon>
          <ListItemText
            primary="관리"
            primaryTypographyProps={{ variant: "body2" }}
          />
        </ListItemButton>
        <ListItemButton
          sx={{ borderRadius: 1, mx: 1 }}
          onClick={() => navigate("/monitor")}
        >
          <ListItemIcon sx={{ minWidth: 36 }}>
            <MonitorIcon fontSize="small" />
          </ListItemIcon>
          <ListItemText
            primary="모니터링"
            primaryTypographyProps={{ variant: "body2" }}
          />
        </ListItemButton>
      </List>
    </Box>
  );

  // -----------------------------------------------------------------------
  // Main render
  // -----------------------------------------------------------------------

  return (
    <Box sx={{ display: "flex", height: "100vh" }}>
      {/* Sidebar */}
      <Drawer
        variant="persistent"
        open={sidebarOpen}
        sx={{
          width: sidebarOpen ? SIDEBAR_WIDTH : 0,
          flexShrink: 0,
          "& .MuiDrawer-paper": {
            width: SIDEBAR_WIDTH,
            boxSizing: "border-box",
            borderRight: "1px solid",
            borderColor: "divider",
          },
        }}
      >
        {sidebarContent}
      </Drawer>

      {/* Main content */}
      <Box
        sx={{
          flex: 1,
          display: "flex",
          flexDirection: "column",
          minWidth: 0,
          overflow: "auto",
        }}
      >
        {/* Header */}
        <Box
          sx={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            px: 3,
            py: 2,
            borderBottom: "1px solid",
            borderColor: "divider",
            bgcolor: "background.paper",
          }}
        >
          <Typography variant="h5" fontWeight={600}>
            문서 관리
          </Typography>
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

        {/* Upload progress */}
        {uploading && (
          <Box sx={{ px: 3, pt: 1 }}>
            <LinearProgress variant="determinate" value={uploadProgress} />
            <Typography variant="caption" color="text.secondary">
              업로드 중... {uploadProgress}%
            </Typography>
          </Box>
        )}

        {/* Drop zone */}
        <Box sx={{ px: 3, pt: 2 }}>
          <Paper
            {...getRootProps()}
            variant="outlined"
            sx={{
              p: 4,
              textAlign: "center",
              cursor: uploading ? "default" : "pointer",
              borderStyle: "dashed",
              borderWidth: 2,
              borderColor: isDragActive ? "primary.main" : "divider",
              bgcolor: isDragActive ? "primary.50" : "transparent",
              transition: "all 0.2s",
              "&:hover": uploading
                ? {}
                : { borderColor: "primary.main", bgcolor: "grey.50" },
            }}
          >
            <input {...getInputProps()} />
            <UploadIcon
              sx={{ fontSize: 48, color: "text.secondary", mb: 1 }}
            />
            <Typography variant="body1" color="text.secondary">
              {isDragActive
                ? "파일을 여기에 놓으세요"
                : "파일을 드래그하거나 클릭하여 업로드"}
            </Typography>
            <Typography variant="caption" color="text.secondary">
              지원 형식: PDF, HWP, DOCX, XLSX, PPTX, TXT
            </Typography>
          </Paper>
        </Box>

        {/* Documents table */}
        <Box sx={{ px: 3, py: 2, flex: 1 }}>
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
              <FolderOpenIcon
                sx={{ fontSize: 64, color: "text.secondary", opacity: 0.4 }}
              />
              <Typography variant="h6" color="text.secondary">
                등록된 문서가 없습니다
              </Typography>
              <Typography variant="body2" color="text.secondary">
                위의 업로드 영역에 파일을 드래그하여 문서를 추가하세요.
              </Typography>
            </Box>
          ) : (
            <TableContainer component={Paper} variant="outlined">
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>파일명</TableCell>
                    <TableCell>유형</TableCell>
                    <TableCell align="right">크기</TableCell>
                    <TableCell align="right">청크수</TableCell>
                    <TableCell>업로드일</TableCell>
                    <TableCell>상태</TableCell>
                    <TableCell align="center">액션</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {documents.map((doc) => {
                    const fileType =
                      doc.file_type ?? doc.fileType ?? "";
                    const fileSize =
                      doc.file_size ?? doc.fileSize ?? 0;
                    const chunkCount =
                      doc.chunk_count ?? doc.chunkCount ?? 0;
                    const uploadDate =
                      doc.uploaded_at ?? doc.uploadDate ?? "";

                    return (
                      <TableRow
                        key={doc.id}
                        hover
                        sx={{ cursor: "pointer" }}
                        onClick={() => setDetailDoc(doc)}
                      >
                        <TableCell>
                          <Box
                            sx={{
                              display: "flex",
                              alignItems: "center",
                              gap: 1,
                            }}
                          >
                            <FileIcon
                              fontSize="small"
                              color="action"
                            />
                            <Typography
                              variant="body2"
                              noWrap
                              sx={{ maxWidth: 300 }}
                            >
                              {doc.filename}
                            </Typography>
                          </Box>
                        </TableCell>
                        <TableCell>
                          <Chip
                            label={
                              fileType.toUpperCase() ||
                              doc.filename
                                .split(".")
                                .pop()
                                ?.toUpperCase() ||
                              "-"
                            }
                            size="small"
                            variant="outlined"
                          />
                        </TableCell>
                        <TableCell align="right">
                          {fileSize > 0
                            ? formatFileSize(fileSize)
                            : "-"}
                        </TableCell>
                        <TableCell align="right">
                          {chunkCount}
                        </TableCell>
                        <TableCell>
                          {uploadDate
                            ? formatDate(uploadDate)
                            : "-"}
                        </TableCell>
                        <TableCell>
                          {getStatusChip(doc.status)}
                        </TableCell>
                        <TableCell align="center">
                          <Tooltip title="삭제">
                            <IconButton
                              size="small"
                              color="error"
                              onClick={(e) => {
                                e.stopPropagation();
                                setDeleteDialogId(doc.id);
                              }}
                            >
                              <DeleteIcon fontSize="small" />
                            </IconButton>
                          </Tooltip>
                        </TableCell>
                      </TableRow>
                    );
                  })}
                </TableBody>
              </Table>
            </TableContainer>
          )}
        </Box>
      </Box>

      {/* Detail side panel */}
      <Drawer
        anchor="right"
        open={detailDoc !== null}
        onClose={() => setDetailDoc(null)}
        sx={{
          "& .MuiDrawer-paper": { width: 360, p: 3 },
        }}
      >
        {detailDoc && (
          <Box>
            <Box
              sx={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                mb: 2,
              }}
            >
              <Typography variant="h6">문서 상세</Typography>
              <IconButton onClick={() => setDetailDoc(null)}>
                <CloseIcon />
              </IconButton>
            </Box>
            <Divider sx={{ mb: 2 }} />

            <Box sx={{ display: "flex", flexDirection: "column", gap: 2 }}>
              <Box>
                <Typography variant="caption" color="text.secondary">
                  파일명
                </Typography>
                <Typography variant="body2" sx={{ wordBreak: "break-all" }}>
                  {detailDoc.filename}
                </Typography>
              </Box>
              <Box>
                <Typography variant="caption" color="text.secondary">
                  파일 유형
                </Typography>
                <Typography variant="body2">
                  {(
                    detailDoc.file_type ??
                    detailDoc.fileType ??
                    detailDoc.filename.split(".").pop() ??
                    "-"
                  ).toUpperCase()}
                </Typography>
              </Box>
              <Box>
                <Typography variant="caption" color="text.secondary">
                  파일 크기
                </Typography>
                <Typography variant="body2">
                  {formatFileSize(
                    detailDoc.file_size ?? detailDoc.fileSize ?? 0
                  )}
                </Typography>
              </Box>
              <Box>
                <Typography variant="caption" color="text.secondary">
                  청크 수
                </Typography>
                <Typography variant="body2">
                  {detailDoc.chunk_count ?? detailDoc.chunkCount ?? 0}개
                </Typography>
              </Box>
              <Box>
                <Typography variant="caption" color="text.secondary">
                  업로드일
                </Typography>
                <Typography variant="body2">
                  {(detailDoc.uploaded_at ?? detailDoc.uploadDate)
                    ? formatDate(
                        detailDoc.uploaded_at ?? detailDoc.uploadDate ?? ""
                      )
                    : "-"}
                </Typography>
              </Box>
              <Box>
                <Typography variant="caption" color="text.secondary">
                  상태
                </Typography>
                <Box sx={{ mt: 0.5 }}>
                  {getStatusChip(detailDoc.status)}
                </Box>
              </Box>
              {detailDoc.metadata && (
                <Box>
                  <Typography variant="caption" color="text.secondary">
                    메타데이터
                  </Typography>
                  <Paper
                    variant="outlined"
                    sx={{ p: 1.5, mt: 0.5, fontSize: "0.82rem" }}
                  >
                    <pre
                      style={{
                        margin: 0,
                        whiteSpace: "pre-wrap",
                        wordBreak: "break-all",
                      }}
                    >
                      {JSON.stringify(detailDoc.metadata, null, 2)}
                    </pre>
                  </Paper>
                </Box>
              )}
            </Box>

            <Divider sx={{ my: 2 }} />
            <Button
              variant="outlined"
              color="error"
              fullWidth
              startIcon={<DeleteIcon />}
              onClick={() => {
                setDeleteDialogId(detailDoc.id);
                setDetailDoc(null);
              }}
            >
              문서 삭제
            </Button>
          </Box>
        )}
      </Drawer>

      {/* Delete confirmation */}
      <Dialog
        open={deleteDialogId !== null}
        onClose={() => setDeleteDialogId(null)}
      >
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
    </Box>
  );
}
