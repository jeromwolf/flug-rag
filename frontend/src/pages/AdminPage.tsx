import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  Box,
  Tabs,
  Tab,
  Typography,
  Card,
  CardContent,
  Grid,
  CircularProgress,
  Alert,
  Chip,
  TextField,
  Button,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Snackbar,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  IconButton,
  Select,
  MenuItem,
  Switch,
  FormControlLabel,
  FormControl,
  InputLabel,
} from "@mui/material";
import InfoIcon from "@mui/icons-material/Info";
import StorageIcon from "@mui/icons-material/Storage";
import AppRegistrationIcon from "@mui/icons-material/AppRegistration";
import FolderIcon from "@mui/icons-material/Folder";
import GroupWorkIcon from "@mui/icons-material/GroupWork";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import BuildIcon from "@mui/icons-material/Build";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import AccountTreeIcon from "@mui/icons-material/AccountTree";
import StarIcon from "@mui/icons-material/Star";
import AddIcon from "@mui/icons-material/Add";
import EditIcon from "@mui/icons-material/Edit";
import DeleteIcon from "@mui/icons-material/Delete";
import CheckCircleIcon from "@mui/icons-material/CheckCircle";
import ErrorIcon from "@mui/icons-material/Error";
import SecurityIcon from "@mui/icons-material/Security";
import MemoryIcon from "@mui/icons-material/Memory";
import { adminApi, mcpApi, workflowsApi, guardrailsApi } from "../api/client";
import Layout from "../components/Layout";
import CustomToolBuilder from "../components/CustomToolBuilder";
import ContentManager from "../components/ContentManager";

interface TabPanelProps {
  children: React.ReactNode;
  value: number;
  index: number;
}

function TabPanel({ children, value, index }: TabPanelProps) {
  if (value !== index) return null;
  return <Box sx={{ py: 3 }}>{children}</Box>;
}

// ── Tab: System Info ──
function SystemInfoTab() {
  const { data, isLoading, error } = useQuery({
    queryKey: ["admin-info"],
    queryFn: () => adminApi.getInfo(),
  });

  if (isLoading) return <CircularProgress />;
  if (error) return <Alert severity="error">시스템 정보를 불러올 수 없습니다.</Alert>;

  const info = data?.data;

  const cards = [
    { label: "앱 이름", value: info?.app_name ?? "-", icon: <AppRegistrationIcon color="primary" /> },
    { label: "버전", value: info?.version ?? "-", icon: <InfoIcon color="primary" /> },
    { label: "기본 프로바이더", value: info?.default_provider ?? "-", icon: <StorageIcon color="primary" /> },
    { label: "문서 수", value: info?.document_count ?? 0, icon: <FolderIcon color="primary" /> },
    { label: "세션 수", value: info?.session_count ?? 0, icon: <GroupWorkIcon color="primary" /> },
  ];

  return (
    <Grid container spacing={2}>
      {cards.map((card) => (
        <Grid key={card.label} size={{ xs: 12, sm: 6, md: 4 }}>
          <Card variant="outlined">
            <CardContent sx={{ display: "flex", alignItems: "center", gap: 2 }}>
              {card.icon}
              <Box>
                <Typography variant="body2" color="text.secondary">
                  {card.label}
                </Typography>
                <Typography variant="h6">{String(card.value)}</Typography>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      ))}
    </Grid>
  );
}

// ── Tab: LLM Providers ──
function ProvidersTab() {
  const { data, isLoading, error } = useQuery({
    queryKey: ["admin-providers"],
    queryFn: () => adminApi.getProviders(),
  });

  if (isLoading) return <CircularProgress />;
  if (error) return <Alert severity="error">프로바이더 목록을 불러올 수 없습니다.</Alert>;

  const providers: Array<{ name: string; is_default: boolean }> = data?.data ?? [];

  return (
    <Grid container spacing={2}>
      {providers.map((p) => (
        <Grid key={p.name} size={{ xs: 12, sm: 6, md: 4 }}>
          <Card variant="outlined">
            <CardContent sx={{ display: "flex", alignItems: "center", gap: 2 }}>
              <StorageIcon color="action" />
              <Box sx={{ flex: 1 }}>
                <Typography variant="subtitle1">{p.name}</Typography>
              </Box>
              {p.is_default && <Chip label="기본" color="primary" size="small" icon={<StarIcon />} />}
            </CardContent>
          </Card>
        </Grid>
      ))}
      {providers.length === 0 && (
        <Grid size={12}>
          <Typography color="text.secondary">등록된 프로바이더가 없습니다.</Typography>
        </Grid>
      )}
    </Grid>
  );
}

// ── Tab: Model Management ──
function ModelsTab() {
  const queryClient = useQueryClient();
  const [dialogOpen, setDialogOpen] = useState(false);
  const [editingModel, setEditingModel] = useState<{
    id?: string;
    name: string;
    provider: string;
    model_id: string;
    description: string;
    is_default: boolean;
  } | null>(null);
  const [snack, setSnack] = useState<{ open: boolean; message: string; severity: "success" | "error" }>({
    open: false,
    message: "",
    severity: "success",
  });
  const [deleteConfirm, setDeleteConfirm] = useState<string | null>(null);
  const [testResults, setTestResults] = useState<Record<string, boolean>>({});

  const { data, isLoading, error } = useQuery({
    queryKey: ["admin-models"],
    queryFn: () => adminApi.listModels(),
  });

  const createMutation = useMutation({
    mutationFn: (model: { name: string; provider: string; model_id: string; description: string; is_default: boolean }) =>
      adminApi.createModel(model),
    onSuccess: () => {
      setSnack({ open: true, message: "모델이 추가되었습니다.", severity: "success" });
      queryClient.invalidateQueries({ queryKey: ["admin-models"] });
      setDialogOpen(false);
      setEditingModel(null);
    },
    onError: () => {
      setSnack({ open: true, message: "모델 추가에 실패했습니다.", severity: "error" });
    },
  });

  const updateMutation = useMutation({
    mutationFn: ({ id, model }: { id: string; model: { name: string; provider: string; model_id: string; description: string; is_default: boolean } }) =>
      adminApi.updateModel(id, model),
    onSuccess: () => {
      setSnack({ open: true, message: "모델이 수정되었습니다.", severity: "success" });
      queryClient.invalidateQueries({ queryKey: ["admin-models"] });
      setDialogOpen(false);
      setEditingModel(null);
    },
    onError: () => {
      setSnack({ open: true, message: "모델 수정에 실패했습니다.", severity: "error" });
    },
  });

  const deleteMutation = useMutation({
    mutationFn: (id: string) => adminApi.deleteModel(id),
    onSuccess: () => {
      setSnack({ open: true, message: "모델이 삭제되었습니다.", severity: "success" });
      queryClient.invalidateQueries({ queryKey: ["admin-models"] });
      setDeleteConfirm(null);
    },
    onError: () => {
      setSnack({ open: true, message: "모델 삭제에 실패했습니다.", severity: "error" });
    },
  });

  const testMutation = useMutation({
    mutationFn: (id: string) => adminApi.testModel(id),
    onSuccess: (_, id) => {
      setTestResults((prev) => ({ ...prev, [id]: true }));
      setSnack({ open: true, message: "테스트 성공", severity: "success" });
    },
    onError: (_, id) => {
      setTestResults((prev) => ({ ...prev, [id]: false }));
      setSnack({ open: true, message: "테스트 실패", severity: "error" });
    },
  });

  const handleOpenCreate = () => {
    setEditingModel({ name: "", provider: "ollama", model_id: "", description: "", is_default: false });
    setDialogOpen(true);
  };

  const handleOpenEdit = (model: { id: string; name: string; provider: string; model_id: string; description: string; is_default: boolean }) => {
    setEditingModel(model);
    setDialogOpen(true);
  };

  const handleSave = () => {
    if (!editingModel) return;
    if (editingModel.id) {
      updateMutation.mutate({ id: editingModel.id, model: editingModel });
    } else {
      createMutation.mutate(editingModel);
    }
  };

  const handleDelete = (id: string) => {
    deleteMutation.mutate(id);
  };

  const handleTest = (id: string) => {
    testMutation.mutate(id);
  };

  if (isLoading) return <CircularProgress />;
  if (error) return <Alert severity="error">모델 목록을 불러올 수 없습니다.</Alert>;

  const models: Array<{ id: string; name: string; provider: string; model_id: string; description: string; is_default: boolean; status: string }> = data?.data?.models ?? [];

  return (
    <>
      <Box sx={{ mb: 2, display: "flex", justifyContent: "flex-end" }}>
        <Button variant="contained" startIcon={<AddIcon />} onClick={handleOpenCreate}>
          모델 추가
        </Button>
      </Box>

      <TableContainer component={Paper} variant="outlined">
        <Table>
          <TableHead>
            <TableRow>
              <TableCell>이름</TableCell>
              <TableCell>프로바이더</TableCell>
              <TableCell>모델 ID</TableCell>
              <TableCell>기본</TableCell>
              <TableCell>상태</TableCell>
              <TableCell align="right">작업</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {models.map((model) => (
              <TableRow key={model.id}>
                <TableCell>{model.name}</TableCell>
                <TableCell>{model.provider}</TableCell>
                <TableCell sx={{ fontFamily: "monospace", fontSize: 13 }}>{model.model_id}</TableCell>
                <TableCell>
                  {model.is_default && <Chip label="기본" color="primary" size="small" icon={<StarIcon />} />}
                </TableCell>
                <TableCell>
                  {testResults[model.id] === true && <Chip label="정상" color="success" size="small" icon={<CheckCircleIcon />} />}
                  {testResults[model.id] === false && <Chip label="실패" color="error" size="small" icon={<ErrorIcon />} />}
                  {testResults[model.id] === undefined && model.status && <Chip label={model.status} size="small" />}
                </TableCell>
                <TableCell align="right">
                  <IconButton size="small" onClick={() => handleTest(model.id)}>
                    <PlayArrowIcon />
                  </IconButton>
                  <IconButton size="small" onClick={() => handleOpenEdit(model)}>
                    <EditIcon />
                  </IconButton>
                  <IconButton size="small" onClick={() => setDeleteConfirm(model.id)}>
                    <DeleteIcon />
                  </IconButton>
                </TableCell>
              </TableRow>
            ))}
            {models.length === 0 && (
              <TableRow>
                <TableCell colSpan={6} align="center">
                  <Typography color="text.secondary">등록된 모델이 없습니다.</Typography>
                </TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
      </TableContainer>

      <Dialog open={dialogOpen} onClose={() => setDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>{editingModel?.id ? "모델 수정" : "모델 추가"}</DialogTitle>
        <DialogContent>
          <TextField
            label="이름"
            fullWidth
            margin="normal"
            value={editingModel?.name ?? ""}
            onChange={(e) => setEditingModel((m) => m ? { ...m, name: e.target.value } : null)}
          />
          <Select
            fullWidth
            value={editingModel?.provider ?? "ollama"}
            onChange={(e) => setEditingModel((m) => m ? { ...m, provider: e.target.value } : null)}
            sx={{ mt: 2 }}
          >
            <MenuItem value="ollama">Ollama</MenuItem>
            <MenuItem value="openai">OpenAI</MenuItem>
            <MenuItem value="anthropic">Anthropic</MenuItem>
            <MenuItem value="vllm">vLLM</MenuItem>
          </Select>
          <TextField
            label="모델 ID"
            fullWidth
            margin="normal"
            value={editingModel?.model_id ?? ""}
            onChange={(e) => setEditingModel((m) => m ? { ...m, model_id: e.target.value } : null)}
          />
          <TextField
            label="설명"
            fullWidth
            margin="normal"
            multiline
            rows={2}
            value={editingModel?.description ?? ""}
            onChange={(e) => setEditingModel((m) => m ? { ...m, description: e.target.value } : null)}
          />
          <FormControlLabel
            control={
              <Switch
                checked={editingModel?.is_default ?? false}
                onChange={(e) => setEditingModel((m) => m ? { ...m, is_default: e.target.checked } : null)}
              />
            }
            label="기본 모델로 설정"
            sx={{ mt: 1 }}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDialogOpen(false)}>취소</Button>
          <Button variant="contained" onClick={handleSave} disabled={createMutation.isPending || updateMutation.isPending}>
            저장
          </Button>
        </DialogActions>
      </Dialog>

      <Dialog open={deleteConfirm !== null} onClose={() => setDeleteConfirm(null)}>
        <DialogTitle>모델 삭제</DialogTitle>
        <DialogContent>
          <Typography>이 모델을 삭제하시겠습니까?</Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDeleteConfirm(null)}>취소</Button>
          <Button variant="contained" color="error" onClick={() => deleteConfirm && handleDelete(deleteConfirm)}>
            삭제
          </Button>
        </DialogActions>
      </Dialog>

      <Snackbar
        open={snack.open}
        autoHideDuration={3000}
        onClose={() => setSnack((s) => ({ ...s, open: false }))}
        message={snack.message}
      />
    </>
  );
}

// ── Tab: Guardrails ──
function GuardrailsTab() {
  const queryClient = useQueryClient();
  const [dialogOpen, setDialogOpen] = useState(false);
  const [testDialogOpen, setTestDialogOpen] = useState(false);
  const [testInput, setTestInput] = useState("");
  const [testRuleId, setTestRuleId] = useState("");
  const [testResult, setTestResult] = useState<string | null>(null);
  const [editingRule, setEditingRule] = useState<{
    id?: string;
    name: string;
    rule_type: string;
    pattern: string;
    action: string;
    message: string;
    is_active: boolean;
  } | null>(null);
  const [snack, setSnack] = useState<{ open: boolean; message: string; severity: "success" | "error" }>({
    open: false,
    message: "",
    severity: "success",
  });
  const [deleteConfirm, setDeleteConfirm] = useState<string | null>(null);

  const { data: rulesData, isLoading: rulesLoading, error: rulesError } = useQuery({
    queryKey: ["guardrails-rules"],
    queryFn: () => guardrailsApi.list(),
  });

  const { data: logsData, isLoading: logsLoading } = useQuery({
    queryKey: ["guardrails-logs"],
    queryFn: () => guardrailsApi.getLogs(),
  });

  const createMutation = useMutation({
    mutationFn: (rule: { name: string; rule_type: string; pattern: string; action: string; message: string; is_active: boolean }) =>
      guardrailsApi.create(rule),
    onSuccess: () => {
      setSnack({ open: true, message: "규칙이 추가되었습니다.", severity: "success" });
      queryClient.invalidateQueries({ queryKey: ["guardrails-rules"] });
      setDialogOpen(false);
      setEditingRule(null);
    },
    onError: () => {
      setSnack({ open: true, message: "규칙 추가에 실패했습니다.", severity: "error" });
    },
  });

  const updateMutation = useMutation({
    mutationFn: ({ id, rule }: { id: string; rule: { name: string; rule_type: string; pattern: string; action: string; message: string; is_active: boolean } }) =>
      guardrailsApi.update(id, rule),
    onSuccess: () => {
      setSnack({ open: true, message: "규칙이 수정되었습니다.", severity: "success" });
      queryClient.invalidateQueries({ queryKey: ["guardrails-rules"] });
      setDialogOpen(false);
      setEditingRule(null);
    },
    onError: () => {
      setSnack({ open: true, message: "규칙 수정에 실패했습니다.", severity: "error" });
    },
  });

  const deleteMutation = useMutation({
    mutationFn: (id: string) => guardrailsApi.delete(id),
    onSuccess: () => {
      setSnack({ open: true, message: "규칙이 삭제되었습니다.", severity: "success" });
      queryClient.invalidateQueries({ queryKey: ["guardrails-rules"] });
      setDeleteConfirm(null);
    },
    onError: () => {
      setSnack({ open: true, message: "규칙 삭제에 실패했습니다.", severity: "error" });
    },
  });

  const testMutation = useMutation({
    mutationFn: (data: { rule_id: string; test_input: string }) => guardrailsApi.test(data),
    onSuccess: (res) => {
      setTestResult(JSON.stringify(res.data, null, 2));
    },
    onError: () => {
      setTestResult("테스트 실패");
    },
  });

  const handleOpenCreate = () => {
    setEditingRule({ name: "", rule_type: "keyword", pattern: "", action: "block", message: "", is_active: true });
    setDialogOpen(true);
  };

  const handleOpenEdit = (rule: { id: string; name: string; rule_type: string; pattern: string; action: string; message: string; is_active: boolean }) => {
    setEditingRule(rule);
    setDialogOpen(true);
  };

  const handleSave = () => {
    if (!editingRule) return;
    if (editingRule.id) {
      updateMutation.mutate({ id: editingRule.id, rule: editingRule });
    } else {
      createMutation.mutate(editingRule);
    }
  };

  const handleDelete = (id: string) => {
    deleteMutation.mutate(id);
  };

  const handleTest = () => {
    testMutation.mutate({ rule_id: testRuleId, test_input: testInput });
  };

  if (rulesLoading) return <CircularProgress />;
  if (rulesError) return <Alert severity="error">가드레일 규칙을 불러올 수 없습니다.</Alert>;

  const rules: Array<{ id: string; name: string; rule_type: string; pattern: string; action: string; message: string; is_active: boolean }> = rulesData?.data?.rules ?? [];
  const logs: Array<{ timestamp: string; rule_name: string; input_preview: string; action_taken: string }> = logsData?.data?.logs ?? [];

  return (
    <>
      <Box sx={{ mb: 2, display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <Typography variant="h6">가드레일 규칙</Typography>
        <Box>
          <Button variant="outlined" startIcon={<PlayArrowIcon />} onClick={() => setTestDialogOpen(true)} sx={{ mr: 1 }}>
            테스트
          </Button>
          <Button variant="contained" startIcon={<AddIcon />} onClick={handleOpenCreate}>
            규칙 추가
          </Button>
        </Box>
      </Box>

      <TableContainer component={Paper} variant="outlined" sx={{ mb: 4 }}>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell>이름</TableCell>
              <TableCell>유형</TableCell>
              <TableCell>패턴</TableCell>
              <TableCell>액션</TableCell>
              <TableCell>활성</TableCell>
              <TableCell align="right">작업</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {rules.map((rule) => (
              <TableRow key={rule.id}>
                <TableCell>{rule.name}</TableCell>
                <TableCell>
                  <Chip label={rule.rule_type} size="small" />
                </TableCell>
                <TableCell sx={{ fontFamily: "monospace", fontSize: 13, maxWidth: 200, overflow: "hidden", textOverflow: "ellipsis" }}>
                  {rule.pattern}
                </TableCell>
                <TableCell>
                  <Chip
                    label={rule.action}
                    size="small"
                    color={rule.action === "block" ? "error" : rule.action === "warn" ? "warning" : "default"}
                  />
                </TableCell>
                <TableCell>
                  {rule.is_active ? (
                    <Chip label="활성" color="success" size="small" icon={<CheckCircleIcon />} />
                  ) : (
                    <Chip label="비활성" size="small" />
                  )}
                </TableCell>
                <TableCell align="right">
                  <IconButton size="small" onClick={() => handleOpenEdit(rule)}>
                    <EditIcon />
                  </IconButton>
                  <IconButton size="small" onClick={() => setDeleteConfirm(rule.id)}>
                    <DeleteIcon />
                  </IconButton>
                </TableCell>
              </TableRow>
            ))}
            {rules.length === 0 && (
              <TableRow>
                <TableCell colSpan={6} align="center">
                  <Typography color="text.secondary">등록된 규칙이 없습니다.</Typography>
                </TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
      </TableContainer>

      <Typography variant="h6" sx={{ mb: 2 }}>필터링 로그 (최근 20개)</Typography>
      <TableContainer component={Paper} variant="outlined">
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell>시간</TableCell>
              <TableCell>규칙</TableCell>
              <TableCell>입력 미리보기</TableCell>
              <TableCell>액션</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {logsLoading ? (
              <TableRow>
                <TableCell colSpan={4} align="center"><CircularProgress size={20} /></TableCell>
              </TableRow>
            ) : logs.length > 0 ? (
              logs.slice(0, 20).map((log, idx) => (
                <TableRow key={idx}>
                  <TableCell sx={{ fontSize: 12 }}>{new Date(log.timestamp).toLocaleString()}</TableCell>
                  <TableCell>{log.rule_name}</TableCell>
                  <TableCell sx={{ fontFamily: "monospace", fontSize: 12, maxWidth: 300, overflow: "hidden", textOverflow: "ellipsis" }}>
                    {log.input_preview}
                  </TableCell>
                  <TableCell>
                    <Chip label={log.action_taken} size="small" />
                  </TableCell>
                </TableRow>
              ))
            ) : (
              <TableRow>
                <TableCell colSpan={4} align="center">
                  <Typography color="text.secondary">로그가 없습니다.</Typography>
                </TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
      </TableContainer>

      <Dialog open={dialogOpen} onClose={() => setDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>{editingRule?.id ? "규칙 수정" : "규칙 추가"}</DialogTitle>
        <DialogContent>
          <TextField
            label="이름"
            fullWidth
            margin="normal"
            value={editingRule?.name ?? ""}
            onChange={(e) => setEditingRule((r) => r ? { ...r, name: e.target.value } : null)}
          />
          <Select
            fullWidth
            value={editingRule?.rule_type ?? "keyword"}
            onChange={(e) => setEditingRule((r) => r ? { ...r, rule_type: e.target.value } : null)}
            sx={{ mt: 2 }}
          >
            <MenuItem value="keyword">Keyword</MenuItem>
            <MenuItem value="regex">Regex</MenuItem>
            <MenuItem value="pii">PII</MenuItem>
            <MenuItem value="prompt_injection">Prompt Injection</MenuItem>
          </Select>
          <TextField
            label="패턴"
            fullWidth
            margin="normal"
            value={editingRule?.pattern ?? ""}
            onChange={(e) => setEditingRule((r) => r ? { ...r, pattern: e.target.value } : null)}
          />
          <Select
            fullWidth
            value={editingRule?.action ?? "block"}
            onChange={(e) => setEditingRule((r) => r ? { ...r, action: e.target.value } : null)}
            sx={{ mt: 2 }}
          >
            <MenuItem value="block">Block</MenuItem>
            <MenuItem value="warn">Warn</MenuItem>
            <MenuItem value="mask">Mask</MenuItem>
          </Select>
          <TextField
            label="메시지 (선택)"
            fullWidth
            margin="normal"
            multiline
            rows={2}
            value={editingRule?.message ?? ""}
            onChange={(e) => setEditingRule((r) => r ? { ...r, message: e.target.value } : null)}
          />
          <FormControlLabel
            control={
              <Switch
                checked={editingRule?.is_active ?? true}
                onChange={(e) => setEditingRule((r) => r ? { ...r, is_active: e.target.checked } : null)}
              />
            }
            label="활성화"
            sx={{ mt: 1 }}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDialogOpen(false)}>취소</Button>
          <Button variant="contained" onClick={handleSave} disabled={createMutation.isPending || updateMutation.isPending}>
            저장
          </Button>
        </DialogActions>
      </Dialog>

      <Dialog open={deleteConfirm !== null} onClose={() => setDeleteConfirm(null)}>
        <DialogTitle>규칙 삭제</DialogTitle>
        <DialogContent>
          <Typography>이 규칙을 삭제하시겠습니까?</Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDeleteConfirm(null)}>취소</Button>
          <Button variant="contained" color="error" onClick={() => deleteConfirm && handleDelete(deleteConfirm)}>
            삭제
          </Button>
        </DialogActions>
      </Dialog>

      <Dialog open={testDialogOpen} onClose={() => setTestDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>가드레일 테스트</DialogTitle>
        <DialogContent>
          <FormControl fullWidth margin="normal">
            <InputLabel>테스트 규칙</InputLabel>
            <Select value={testRuleId} onChange={(e) => setTestRuleId(e.target.value)} label="테스트 규칙">
              {rules.map((r) => (
                <MenuItem key={r.id} value={r.id}>{r.name} ({r.rule_type})</MenuItem>
              ))}
            </Select>
          </FormControl>
          <TextField
            label="테스트 입력"
            fullWidth
            margin="normal"
            multiline
            rows={4}
            value={testInput}
            onChange={(e) => setTestInput(e.target.value)}
          />
          {testResult && (
            <Box sx={{ mt: 2, p: 1, bgcolor: "grey.100", borderRadius: 1, maxHeight: 200, overflow: "auto" }}>
              <Typography variant="body2" component="pre" sx={{ fontFamily: "monospace", fontSize: 12, whiteSpace: "pre-wrap" }}>
                {testResult}
              </Typography>
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setTestDialogOpen(false)}>닫기</Button>
          <Button variant="contained" onClick={handleTest} disabled={testMutation.isPending}>
            {testMutation.isPending ? <CircularProgress size={20} /> : "테스트"}
          </Button>
        </DialogActions>
      </Dialog>

      <Snackbar
        open={snack.open}
        autoHideDuration={3000}
        onClose={() => setSnack((s) => ({ ...s, open: false }))}
        message={snack.message}
      />
    </>
  );
}

// ── Tab: Prompt Management ──
function PromptsTab() {
  const queryClient = useQueryClient();
  const [editValues, setEditValues] = useState<Record<string, string>>({});
  const [snack, setSnack] = useState<{ open: boolean; message: string; severity: "success" | "error" }>({
    open: false,
    message: "",
    severity: "success",
  });

  const { data, isLoading, error } = useQuery({
    queryKey: ["admin-prompts"],
    queryFn: () => adminApi.getPrompts(),
  });

  const mutation = useMutation({
    mutationFn: ({ name, content }: { name: string; content: string }) =>
      adminApi.updatePrompt(name, content),
    onSuccess: () => {
      setSnack({ open: true, message: "프롬프트가 저장되었습니다.", severity: "success" });
      queryClient.invalidateQueries({ queryKey: ["admin-prompts"] });
    },
    onError: () => {
      setSnack({ open: true, message: "프롬프트 저장에 실패했습니다.", severity: "error" });
    },
  });

  if (isLoading) return <CircularProgress />;
  if (error) return <Alert severity="error">프롬프트를 불러올 수 없습니다.</Alert>;

  const prompts: Record<string, string> = data?.data?.prompts ?? {};

  const handleChange = (name: string, value: string) => {
    setEditValues((prev) => ({ ...prev, [name]: value }));
  };

  const handleSave = (name: string) => {
    const content = editValues[name] ?? prompts[name] ?? "";
    mutation.mutate({ name, content });
  };

  return (
    <>
      {Object.entries(prompts).map(([name, defaultValue]) => (
        <Accordion key={name} sx={{ mb: 1 }}>
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
              {name}
            </Typography>
          </AccordionSummary>
          <AccordionDetails>
            <TextField
              multiline
              fullWidth
              minRows={4}
              maxRows={16}
              value={editValues[name] ?? defaultValue}
              onChange={(e) => handleChange(name, e.target.value)}
              slotProps={{
                input: {
                  sx: { fontFamily: "monospace", fontSize: 13 },
                },
              }}
            />
            <Box sx={{ mt: 1, display: "flex", justifyContent: "flex-end" }}>
              <Button
                variant="contained"
                size="small"
                onClick={() => handleSave(name)}
                disabled={mutation.isPending}
              >
                저장
              </Button>
            </Box>
          </AccordionDetails>
        </Accordion>
      ))}
      {Object.keys(prompts).length === 0 && (
        <Typography color="text.secondary">등록된 프롬프트가 없습니다.</Typography>
      )}
      <Snackbar
        open={snack.open}
        autoHideDuration={3000}
        onClose={() => setSnack((s) => ({ ...s, open: false }))}
        message={snack.message}
      />
    </>
  );
}

// ── Tab: MCP Tools ──
function McpToolsTab() {
  const [testDialog, setTestDialog] = useState<{ open: boolean; toolName: string }>({
    open: false,
    toolName: "",
  });
  const [argsInput, setArgsInput] = useState("{}");
  const [result, setResult] = useState<string | null>(null);
  const [callError, setCallError] = useState<string | null>(null);

  const { data, isLoading, error } = useQuery({
    queryKey: ["mcp-tools"],
    queryFn: () => mcpApi.listTools(),
  });

  const callMutation = useMutation({
    mutationFn: ({ toolName, args }: { toolName: string; args: Record<string, unknown> }) =>
      mcpApi.callTool(toolName, args),
    onSuccess: (res) => {
      setResult(JSON.stringify(res.data, null, 2));
      setCallError(null);
    },
    onError: (err) => {
      setCallError(String(err));
      setResult(null);
    },
  });

  const handleOpenTest = (toolName: string) => {
    setTestDialog({ open: true, toolName });
    setArgsInput("{}");
    setResult(null);
    setCallError(null);
  };

  const handleCall = () => {
    try {
      const parsed = JSON.parse(argsInput) as Record<string, unknown>;
      callMutation.mutate({ toolName: testDialog.toolName, args: parsed });
    } catch {
      setCallError("올바른 JSON 형식이 아닙니다.");
    }
  };

  if (isLoading) return <CircularProgress />;
  if (error) return <Alert severity="error">MCP 도구를 불러올 수 없습니다.</Alert>;

  const tools: Array<{ name: string; description: string; parameters?: unknown }> = data?.data?.tools ?? [];

  return (
    <>
      {tools.map((tool) => (
        <Card key={tool.name} variant="outlined" sx={{ mb: 1.5 }}>
          <CardContent sx={{ display: "flex", alignItems: "center", gap: 2 }}>
            <BuildIcon color="action" />
            <Box sx={{ flex: 1 }}>
              <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
                {tool.name}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                {tool.description}
              </Typography>
            </Box>
            <Button variant="outlined" size="small" onClick={() => handleOpenTest(tool.name)}>
              테스트
            </Button>
          </CardContent>
        </Card>
      ))}
      {tools.length === 0 && (
        <Typography color="text.secondary">등록된 MCP 도구가 없습니다.</Typography>
      )}

      <Dialog
        open={testDialog.open}
        onClose={() => setTestDialog({ open: false, toolName: "" })}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>도구 테스트: {testDialog.toolName}</DialogTitle>
        <DialogContent>
          <Typography variant="body2" sx={{ mb: 1, mt: 1 }}>
            인자 (JSON)
          </Typography>
          <TextField
            multiline
            fullWidth
            minRows={3}
            maxRows={10}
            value={argsInput}
            onChange={(e) => setArgsInput(e.target.value)}
            slotProps={{
              input: {
                sx: { fontFamily: "monospace", fontSize: 13 },
              },
            }}
          />
          {callError && (
            <Alert severity="error" sx={{ mt: 1 }}>
              {callError}
            </Alert>
          )}
          {result && (
            <Box sx={{ mt: 1, p: 1, bgcolor: "grey.100", borderRadius: 1, maxHeight: 200, overflow: "auto" }}>
              <Typography variant="body2" component="pre" sx={{ fontFamily: "monospace", fontSize: 12, whiteSpace: "pre-wrap" }}>
                {result}
              </Typography>
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setTestDialog({ open: false, toolName: "" })}>닫기</Button>
          <Button variant="contained" onClick={handleCall} disabled={callMutation.isPending}>
            {callMutation.isPending ? <CircularProgress size={20} /> : "실행"}
          </Button>
        </DialogActions>
      </Dialog>
    </>
  );
}

// ── Tab: Workflows ──
function WorkflowsTab() {
  const [runDialog, setRunDialog] = useState<{ open: boolean; preset: string }>({
    open: false,
    preset: "",
  });
  const [inputData, setInputData] = useState("{}");
  const [result, setResult] = useState<string | null>(null);
  const [runError, setRunError] = useState<string | null>(null);

  const { data, isLoading, error } = useQuery({
    queryKey: ["workflow-presets"],
    queryFn: () => workflowsApi.listPresets(),
  });

  const runMutation = useMutation({
    mutationFn: ({ preset, input }: { preset: string; input: Record<string, unknown> }) =>
      workflowsApi.run(preset, input),
    onSuccess: (res) => {
      setResult(JSON.stringify(res.data, null, 2));
      setRunError(null);
    },
    onError: (err) => {
      setRunError(String(err));
      setResult(null);
    },
  });

  const handleOpenRun = (preset: string) => {
    setRunDialog({ open: true, preset });
    setInputData("{}");
    setResult(null);
    setRunError(null);
  };

  const handleRun = () => {
    try {
      const parsed = JSON.parse(inputData) as Record<string, unknown>;
      runMutation.mutate({ preset: runDialog.preset, input: parsed });
    } catch {
      setRunError("올바른 JSON 형식이 아닙니다.");
    }
  };

  if (isLoading) return <CircularProgress />;
  if (error) return <Alert severity="error">워크플로우를 불러올 수 없습니다.</Alert>;

  const presets: Array<{ name: string; description: string; node_count: number }> = data?.data?.presets ?? [];

  return (
    <>
      {presets.map((preset) => (
        <Card key={preset.name} variant="outlined" sx={{ mb: 1.5 }}>
          <CardContent sx={{ display: "flex", alignItems: "center", gap: 2 }}>
            <AccountTreeIcon color="action" />
            <Box sx={{ flex: 1 }}>
              <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
                {preset.name}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                {preset.description}
              </Typography>
              <Chip label={`노드 ${preset.node_count}개`} size="small" sx={{ mt: 0.5 }} />
            </Box>
            <Button
              variant="outlined"
              size="small"
              startIcon={<PlayArrowIcon />}
              onClick={() => handleOpenRun(preset.name)}
            >
              실행
            </Button>
          </CardContent>
        </Card>
      ))}
      {presets.length === 0 && (
        <Typography color="text.secondary">등록된 워크플로우가 없습니다.</Typography>
      )}

      <Dialog
        open={runDialog.open}
        onClose={() => setRunDialog({ open: false, preset: "" })}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>워크플로우 실행: {runDialog.preset}</DialogTitle>
        <DialogContent>
          <Typography variant="body2" sx={{ mb: 1, mt: 1 }}>
            입력 데이터 (JSON)
          </Typography>
          <TextField
            multiline
            fullWidth
            minRows={3}
            maxRows={10}
            value={inputData}
            onChange={(e) => setInputData(e.target.value)}
            slotProps={{
              input: {
                sx: { fontFamily: "monospace", fontSize: 13 },
              },
            }}
          />
          {runError && (
            <Alert severity="error" sx={{ mt: 1 }}>
              {runError}
            </Alert>
          )}
          {result && (
            <Box sx={{ mt: 1, p: 1, bgcolor: "grey.100", borderRadius: 1, maxHeight: 200, overflow: "auto" }}>
              <Typography variant="body2" component="pre" sx={{ fontFamily: "monospace", fontSize: 12, whiteSpace: "pre-wrap" }}>
                {result}
              </Typography>
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setRunDialog({ open: false, preset: "" })}>닫기</Button>
          <Button variant="contained" onClick={handleRun} disabled={runMutation.isPending}>
            {runMutation.isPending ? <CircularProgress size={20} /> : "실행"}
          </Button>
        </DialogActions>
      </Dialog>
    </>
  );
}

// ── Main: AdminPage ──
export default function AdminPage() {
  const [tabIndex, setTabIndex] = useState(0);

  return (
    <Layout title="관리자 설정">
      <Tabs
        value={tabIndex}
        onChange={(_, v: number) => setTabIndex(v)}
        sx={{ borderBottom: 1, borderColor: "divider", mb: 1 }}
      >
        <Tab label="시스템 정보" />
        <Tab label="LLM 프로바이더" />
        <Tab label="모델 관리" icon={<MemoryIcon />} iconPosition="start" />
        <Tab label="가드레일" icon={<SecurityIcon />} iconPosition="start" />
        <Tab label="프롬프트 관리" />
        <Tab label="MCP 도구" />
        <Tab label="커스텀 도구" />
        <Tab label="워크플로우" />
        <Tab label="콘텐츠 관리" />
      </Tabs>

      <TabPanel value={tabIndex} index={0}>
        <SystemInfoTab />
      </TabPanel>
      <TabPanel value={tabIndex} index={1}>
        <ProvidersTab />
      </TabPanel>
      <TabPanel value={tabIndex} index={2}>
        <ModelsTab />
      </TabPanel>
      <TabPanel value={tabIndex} index={3}>
        <GuardrailsTab />
      </TabPanel>
      <TabPanel value={tabIndex} index={4}>
        <PromptsTab />
      </TabPanel>
      <TabPanel value={tabIndex} index={5}>
        <McpToolsTab />
      </TabPanel>
      <TabPanel value={tabIndex} index={6}>
        <CustomToolBuilder />
      </TabPanel>
      <TabPanel value={tabIndex} index={7}>
        <WorkflowsTab />
      </TabPanel>
      <TabPanel value={tabIndex} index={8}>
        <ContentManager />
      </TabPanel>
    </Layout>
  );
}
