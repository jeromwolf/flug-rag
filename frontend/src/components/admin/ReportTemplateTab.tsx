import { useState, useRef } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  Box,
  Typography,
  Button,
  TextField,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  IconButton,
  Chip,
  CircularProgress,
  Alert,
  Stack,
  Checkbox,
  FormControlLabel,
  Collapse,
  Divider,
} from "@mui/material";
import AddIcon from "@mui/icons-material/Add";
import DeleteIcon from "@mui/icons-material/Delete";
import EditIcon from "@mui/icons-material/Edit";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import ExpandLessIcon from "@mui/icons-material/ExpandLess";
import CloudUploadIcon from "@mui/icons-material/CloudUpload";
import DescriptionIcon from "@mui/icons-material/Description";
import { reportTemplatesApi } from "../../api/client";

interface Section {
  title: string;
  description: string;
  required: boolean;
}

interface ReportTemplate {
  id: string;
  name: string;
  description?: string;
  sections?: Section[];
  jinja_template?: string;
  updated_at?: string;
  created_at?: string;
}

const EMPTY_SECTION: Section = { title: "", description: "", required: false };

const EMPTY_FORM = {
  name: "",
  description: "",
  sections: [{ ...EMPTY_SECTION }] as Section[],
  jinja_template: "",
};

export default function ReportTemplateTab() {
  const queryClient = useQueryClient();

  const [editOpen, setEditOpen] = useState(false);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [form, setForm] = useState({ ...EMPTY_FORM, sections: [{ ...EMPTY_SECTION }] });
  const [advancedOpen, setAdvancedOpen] = useState(false);

  const [imageDialogOpen, setImageDialogOpen] = useState(false);
  const [imageLoading, setImageLoading] = useState(false);
  const [imageError, setImageError] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [deleteConfirmId, setDeleteConfirmId] = useState<string | null>(null);
  const [snackError, setSnackError] = useState<string | null>(null);

  const { data, isLoading, error } = useQuery({
    queryKey: ["report-templates"],
    queryFn: async () => {
      const res = await reportTemplatesApi.list();
      return res.data as { templates?: ReportTemplate[] } | ReportTemplate[];
    },
  });

  const templates: ReportTemplate[] = Array.isArray(data)
    ? data
    : (data as { templates?: ReportTemplate[] })?.templates ?? [];

  const saveMutation = useMutation({
    mutationFn: async (payload: typeof form & { id?: string }) => {
      const { id, ...body } = payload;
      if (id) return reportTemplatesApi.update(id, body);
      return reportTemplatesApi.create(body);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["report-templates"] });
      setEditOpen(false);
    },
    onError: () => setSnackError("저장에 실패했습니다."),
  });

  const deleteMutation = useMutation({
    mutationFn: (id: string) => reportTemplatesApi.delete(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["report-templates"] });
      setDeleteConfirmId(null);
    },
    onError: () => setSnackError("삭제에 실패했습니다."),
  });

  function openNew() {
    setEditingId(null);
    setForm({ name: "", description: "", sections: [{ ...EMPTY_SECTION }], jinja_template: "" });
    setAdvancedOpen(false);
    setEditOpen(true);
  }

  function openEdit(t: ReportTemplate) {
    setEditingId(t.id);
    setForm({
      name: t.name,
      description: t.description ?? "",
      sections: t.sections?.length ? t.sections.map((s) => ({ ...s })) : [{ ...EMPTY_SECTION }],
      jinja_template: t.jinja_template ?? "",
    });
    setAdvancedOpen(false);
    setEditOpen(true);
  }

  function handleSave() {
    if (!form.name.trim()) return;
    saveMutation.mutate({ ...form, id: editingId ?? undefined });
  }

  function addSection() {
    setForm((f) => ({ ...f, sections: [...f.sections, { ...EMPTY_SECTION }] }));
  }

  function removeSection(idx: number) {
    setForm((f) => ({ ...f, sections: f.sections.filter((_, i) => i !== idx) }));
  }

  function updateSection(idx: number, field: keyof Section, value: string | boolean) {
    setForm((f) => {
      const sections = f.sections.map((s, i) => (i === idx ? { ...s, [field]: value } : s));
      return { ...f, sections };
    });
  }

  async function handleImageUpload(file: File) {
    setImageLoading(true);
    setImageError(null);
    try {
      const formData = new FormData();
      formData.append("file", file);
      const resp = await reportTemplatesApi.fromImage(formData);
      const generated = resp.data as Partial<ReportTemplate>;
      setImageDialogOpen(false);
      setEditingId(null);
      setForm({
        name: generated.name ?? "",
        description: generated.description ?? "",
        sections: generated.sections?.length
          ? generated.sections.map((s) => ({ ...s }))
          : [{ ...EMPTY_SECTION }],
        jinja_template: generated.jinja_template ?? "",
      });
      setAdvancedOpen(false);
      setEditOpen(true);
    } catch {
      setImageError("이미지 분석에 실패했습니다. 다시 시도해 주세요.");
    } finally {
      setImageLoading(false);
    }
  }

  function handleFileDrop(e: React.DragEvent) {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file) handleImageUpload(file);
  }

  function formatDate(dateStr?: string) {
    if (!dateStr) return "-";
    return new Date(dateStr).toLocaleDateString("ko-KR");
  }

  return (
    <Box>
      {/* Header */}
      <Stack direction="row" justifyContent="space-between" alignItems="center" mb={3}>
        <Box>
          <Typography variant="h6" fontWeight={600}>
            보고서 템플릿 관리
          </Typography>
          <Typography variant="body2" color="text.secondary">
            보고서 구조를 정의하고 이미지에서 자동으로 생성합니다.
          </Typography>
        </Box>
        <Stack direction="row" spacing={1}>
          <Button
            variant="outlined"
            startIcon={<CloudUploadIcon />}
            onClick={() => { setImageError(null); setImageDialogOpen(true); }}
          >
            이미지로 생성
          </Button>
          <Button variant="contained" startIcon={<AddIcon />} onClick={openNew}>
            새 템플릿
          </Button>
        </Stack>
      </Stack>

      {snackError && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setSnackError(null)}>
          {snackError}
        </Alert>
      )}

      {/* Table */}
      {isLoading ? (
        <Box display="flex" justifyContent="center" py={6}>
          <CircularProgress />
        </Box>
      ) : error ? (
        <Alert severity="error">템플릿 목록을 불러오는 데 실패했습니다.</Alert>
      ) : templates.length === 0 ? (
        <Paper variant="outlined" sx={{ p: 6, textAlign: "center" }}>
          <DescriptionIcon sx={{ fontSize: 48, color: "text.disabled", mb: 1 }} />
          <Typography color="text.secondary">등록된 템플릿이 없습니다.</Typography>
          <Button variant="outlined" startIcon={<AddIcon />} onClick={openNew} sx={{ mt: 2 }}>
            첫 템플릿 만들기
          </Button>
        </Paper>
      ) : (
        <TableContainer component={Paper} variant="outlined">
          <Table size="small">
            <TableHead>
              <TableRow sx={{ bgcolor: "grey.50" }}>
                <TableCell sx={{ fontWeight: 600 }}>이름</TableCell>
                <TableCell sx={{ fontWeight: 600 }}>설명</TableCell>
                <TableCell sx={{ fontWeight: 600 }} align="center">섹션 수</TableCell>
                <TableCell sx={{ fontWeight: 600 }}>수정일</TableCell>
                <TableCell sx={{ fontWeight: 600 }} align="right">액션</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {templates.map((t) => (
                <TableRow key={t.id} hover>
                  <TableCell>
                    <Typography variant="body2" fontWeight={500}>
                      {t.name}
                    </Typography>
                  </TableCell>
                  <TableCell>
                    <Typography variant="body2" color="text.secondary" noWrap sx={{ maxWidth: 260 }}>
                      {t.description || "-"}
                    </Typography>
                  </TableCell>
                  <TableCell align="center">
                    <Chip
                      label={t.sections?.length ?? 0}
                      size="small"
                      variant="outlined"
                    />
                  </TableCell>
                  <TableCell>
                    <Typography variant="body2" color="text.secondary">
                      {formatDate(t.updated_at ?? t.created_at)}
                    </Typography>
                  </TableCell>
                  <TableCell align="right">
                    <Stack direction="row" justifyContent="flex-end" spacing={0.5}>
                      <IconButton size="small" onClick={() => openEdit(t)}>
                        <EditIcon fontSize="small" />
                      </IconButton>
                      <IconButton
                        size="small"
                        color="error"
                        onClick={() => setDeleteConfirmId(t.id)}
                      >
                        <DeleteIcon fontSize="small" />
                      </IconButton>
                    </Stack>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      )}

      {/* Create/Edit Dialog */}
      <Dialog open={editOpen} onClose={() => setEditOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>{editingId ? "템플릿 편집" : "새 템플릿"}</DialogTitle>
        <DialogContent dividers>
          <Stack spacing={2.5}>
            <TextField
              label="이름"
              fullWidth
              size="small"
              value={form.name}
              onChange={(e) => setForm((f) => ({ ...f, name: e.target.value }))}
              required
              error={saveMutation.isError && !form.name.trim()}
              helperText={saveMutation.isError && !form.name.trim() ? "이름을 입력하세요." : ""}
            />
            <TextField
              label="설명"
              fullWidth
              size="small"
              multiline
              rows={2}
              value={form.description}
              onChange={(e) => setForm((f) => ({ ...f, description: e.target.value }))}
            />

            <Box>
              <Stack direction="row" justifyContent="space-between" alignItems="center" mb={1}>
                <Typography variant="body2" fontWeight={500}>
                  섹션 목록
                </Typography>
                <Button size="small" startIcon={<AddIcon />} onClick={addSection}>
                  섹션 추가
                </Button>
              </Stack>
              <Stack spacing={1.5}>
                {form.sections.map((s, idx) => (
                  <Paper key={idx} variant="outlined" sx={{ p: 1.5 }}>
                    <Stack spacing={1}>
                      <Stack direction="row" spacing={1} alignItems="center">
                        <TextField
                          label="제목"
                          size="small"
                          fullWidth
                          value={s.title}
                          onChange={(e) => updateSection(idx, "title", e.target.value)}
                        />
                        <IconButton
                          size="small"
                          color="error"
                          onClick={() => removeSection(idx)}
                          disabled={form.sections.length <= 1}
                        >
                          <DeleteIcon fontSize="small" />
                        </IconButton>
                      </Stack>
                      <TextField
                        label="설명"
                        size="small"
                        fullWidth
                        value={s.description}
                        onChange={(e) => updateSection(idx, "description", e.target.value)}
                      />
                      <FormControlLabel
                        control={
                          <Checkbox
                            size="small"
                            checked={s.required}
                            onChange={(e) => updateSection(idx, "required", e.target.checked)}
                          />
                        }
                        label={<Typography variant="body2">필수 섹션</Typography>}
                      />
                    </Stack>
                  </Paper>
                ))}
              </Stack>
            </Box>

            {/* Advanced: Jinja2 template */}
            <Box>
              <Button
                size="small"
                variant="text"
                color="inherit"
                endIcon={advancedOpen ? <ExpandLessIcon /> : <ExpandMoreIcon />}
                onClick={() => setAdvancedOpen((v) => !v)}
              >
                고급 설정
              </Button>
              <Collapse in={advancedOpen}>
                <Box mt={1}>
                  <Divider sx={{ mb: 1.5 }} />
                  <TextField
                    label="Jinja2 템플릿"
                    fullWidth
                    size="small"
                    multiline
                    rows={6}
                    value={form.jinja_template}
                    onChange={(e) => setForm((f) => ({ ...f, jinja_template: e.target.value }))}
                    placeholder={"# {{ title }}\n\n{% for section in sections %}\n## {{ section.title }}\n{{ section.content }}\n{% endfor %}"}
                    inputProps={{ style: { fontFamily: "monospace", fontSize: "0.8rem" } }}
                    helperText="Jinja2 문법으로 보고서 출력 형식을 커스터마이징합니다."
                  />
                </Box>
              </Collapse>
            </Box>
          </Stack>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setEditOpen(false)}>취소</Button>
          <Button
            variant="contained"
            onClick={handleSave}
            disabled={!form.name.trim() || saveMutation.isPending}
            startIcon={saveMutation.isPending ? <CircularProgress size={16} color="inherit" /> : undefined}
          >
            저장
          </Button>
        </DialogActions>
      </Dialog>

      {/* Image Upload Dialog */}
      <Dialog open={imageDialogOpen} onClose={() => !imageLoading && setImageDialogOpen(false)} maxWidth="xs" fullWidth>
        <DialogTitle>이미지로 템플릿 생성</DialogTitle>
        <DialogContent>
          <Typography variant="body2" color="text.secondary" mb={2}>
            보고서 양식 이미지를 업로드하면 섹션 구조를 자동으로 추출합니다.
          </Typography>
          <Box
            onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
            onDragLeave={() => setDragOver(false)}
            onDrop={handleFileDrop}
            onClick={() => fileInputRef.current?.click()}
            sx={{
              border: 2,
              borderStyle: "dashed",
              borderColor: dragOver ? "primary.main" : "divider",
              borderRadius: 2,
              p: 4,
              textAlign: "center",
              cursor: "pointer",
              bgcolor: dragOver ? "primary.50" : "grey.50",
              transition: "all 0.15s",
              "&:hover": { borderColor: "primary.main", bgcolor: "primary.50" },
            }}
          >
            {imageLoading ? (
              <Stack alignItems="center" spacing={1}>
                <CircularProgress size={32} />
                <Typography variant="body2" color="text.secondary">
                  분석 중...
                </Typography>
              </Stack>
            ) : (
              <Stack alignItems="center" spacing={1}>
                <CloudUploadIcon sx={{ fontSize: 40, color: "text.disabled" }} />
                <Typography variant="body2" fontWeight={500}>
                  클릭하거나 파일을 끌어다 놓으세요
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  PNG, JPG, PDF 지원
                </Typography>
              </Stack>
            )}
          </Box>
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*,application/pdf"
            style={{ display: "none" }}
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) handleImageUpload(file);
              e.target.value = "";
            }}
          />
          {imageError && (
            <Alert severity="error" sx={{ mt: 2 }}>
              {imageError}
            </Alert>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setImageDialogOpen(false)} disabled={imageLoading}>
            취소
          </Button>
        </DialogActions>
      </Dialog>

      {/* Delete Confirm Dialog */}
      <Dialog open={!!deleteConfirmId} onClose={() => setDeleteConfirmId(null)} maxWidth="xs">
        <DialogTitle>템플릿 삭제</DialogTitle>
        <DialogContent>
          <Typography variant="body2">
            이 템플릿을 삭제하시겠습니까? 이 작업은 되돌릴 수 없습니다.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDeleteConfirmId(null)}>취소</Button>
          <Button
            color="error"
            variant="contained"
            onClick={() => deleteConfirmId && deleteMutation.mutate(deleteConfirmId)}
            disabled={deleteMutation.isPending}
            startIcon={deleteMutation.isPending ? <CircularProgress size={16} color="inherit" /> : undefined}
          >
            삭제
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}
