import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  Box,
  Button,
  Chip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  FormControl,
  Grid,
  IconButton,
  InputLabel,
  MenuItem,
  Select,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  TextField,
  Typography,
  Alert,
  Switch,
  FormControlLabel,
  CircularProgress,
} from "@mui/material";
import AddIcon from "@mui/icons-material/Add";
import DeleteIcon from "@mui/icons-material/Delete";
import EditIcon from "@mui/icons-material/Edit";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import { mcpApi } from "../api/client";

interface ToolParam {
  name: string;
  type: string;
  description: string;
  required: boolean;
}

interface CustomTool {
  id: string;
  name: string;
  description: string;
  parameters_schema: string;
  execution_type: string;
  api_url: string;
  api_method: string;
  api_headers: string;
  api_body_template: string;
  is_active: boolean;
  created_at: string;
  created_by: string;
}

const PARAM_TYPES = ["STRING", "INTEGER", "FLOAT", "BOOLEAN"];

export default function CustomToolBuilder() {
  const queryClient = useQueryClient();
  const [dialogOpen, setDialogOpen] = useState(false);
  const [testDialogOpen, setTestDialogOpen] = useState(false);
  const [editingTool, setEditingTool] = useState<CustomTool | null>(null);
  const [testToolId, setTestToolId] = useState("");
  const [testInput, setTestInput] = useState("{}");
  const [testResult, setTestResult] = useState("");

  // Form state
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [params, setParams] = useState<ToolParam[]>([]);
  const [apiUrl, setApiUrl] = useState("");
  const [apiMethod, setApiMethod] = useState("POST");
  const [apiHeaders, setApiHeaders] = useState("{}");
  const [apiBodyTemplate, setApiBodyTemplate] = useState("");

  const { data, isLoading } = useQuery({
    queryKey: ["custom-tools"],
    queryFn: () => mcpApi.listCustomTools(),
  });

  const createMutation = useMutation({
    mutationFn: (toolData: Record<string, unknown>) => mcpApi.createCustomTool(toolData as any),
    onSuccess: () => { queryClient.invalidateQueries({ queryKey: ["custom-tools"] }); setDialogOpen(false); resetForm(); },
  });

  const deleteMutation = useMutation({
    mutationFn: (id: string) => mcpApi.deleteCustomTool(id),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["custom-tools"] }),
  });

  const resetForm = () => {
    setName(""); setDescription(""); setParams([]); setApiUrl("");
    setApiMethod("POST"); setApiHeaders("{}"); setApiBodyTemplate("");
    setEditingTool(null);
  };

  const openCreate = () => { resetForm(); setDialogOpen(true); };

  const openEdit = (tool: CustomTool) => {
    setEditingTool(tool);
    setName(tool.name);
    setDescription(tool.description);
    setParams(JSON.parse(tool.parameters_schema || "[]"));
    setApiUrl(tool.api_url);
    setApiMethod(tool.api_method);
    setApiHeaders(tool.api_headers);
    setApiBodyTemplate(tool.api_body_template);
    setDialogOpen(true);
  };

  const handleSave = () => {
    const toolData = {
      name,
      description,
      parameters_schema: JSON.stringify(params),
      execution_type: "api",
      api_url: apiUrl,
      api_method: apiMethod,
      api_headers: apiHeaders,
      api_body_template: apiBodyTemplate,
    };
    if (editingTool) {
      mcpApi.updateCustomTool(editingTool.id, toolData).then(() => {
        queryClient.invalidateQueries({ queryKey: ["custom-tools"] });
        setDialogOpen(false);
        resetForm();
      });
    } else {
      createMutation.mutate(toolData);
    }
  };

  const addParam = () => setParams([...params, { name: "", type: "STRING", description: "", required: true }]);
  const removeParam = (i: number) => setParams(params.filter((_, idx) => idx !== i));
  const updateParam = (i: number, field: keyof ToolParam, value: string | boolean) => {
    const updated = [...params];
    (updated[i] as any)[field] = value;
    setParams(updated);
  };

  const handleTest = async () => {
    try {
      const input = JSON.parse(testInput);
      const res = await mcpApi.testCustomTool(testToolId, input);
      setTestResult(res.data.result || "No output");
    } catch (e: any) {
      setTestResult(`Error: ${e.message}`);
    }
  };

  const tools: CustomTool[] = data?.data?.tools ?? [];

  return (
    <Box>
      <Box sx={{ display: "flex", justifyContent: "space-between", mb: 2 }}>
        <Typography variant="h6" sx={{ fontWeight: 600 }}>커스텀 도구 관리</Typography>
        <Button variant="contained" startIcon={<AddIcon />} onClick={openCreate} size="small">
          도구 추가
        </Button>
      </Box>

      {isLoading ? <CircularProgress /> : (
        <TableContainer component={Paper} variant="outlined">
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>이름</TableCell>
                <TableCell>설명</TableCell>
                <TableCell>유형</TableCell>
                <TableCell>상태</TableCell>
                <TableCell align="right">작업</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {tools.map((tool) => (
                <TableRow key={tool.id}>
                  <TableCell><Typography variant="body2" sx={{ fontWeight: 600 }}>{tool.name}</Typography></TableCell>
                  <TableCell><Typography variant="body2" noWrap sx={{ maxWidth: 300 }}>{tool.description}</Typography></TableCell>
                  <TableCell><Chip label={tool.execution_type} size="small" /></TableCell>
                  <TableCell><Chip label={tool.is_active ? "활성" : "비활성"} color={tool.is_active ? "success" : "default"} size="small" /></TableCell>
                  <TableCell align="right">
                    <IconButton size="small" onClick={() => { setTestToolId(tool.id); setTestInput("{}"); setTestResult(""); setTestDialogOpen(true); }}>
                      <PlayArrowIcon fontSize="small" />
                    </IconButton>
                    <IconButton size="small" onClick={() => openEdit(tool)}><EditIcon fontSize="small" /></IconButton>
                    <IconButton size="small" onClick={() => deleteMutation.mutate(tool.id)}><DeleteIcon fontSize="small" /></IconButton>
                  </TableCell>
                </TableRow>
              ))}
              {tools.length === 0 && (
                <TableRow><TableCell colSpan={5} align="center"><Typography color="text.secondary" variant="body2">등록된 커스텀 도구가 없습니다.</Typography></TableCell></TableRow>
              )}
            </TableBody>
          </Table>
        </TableContainer>
      )}

      {/* Create/Edit Dialog */}
      <Dialog open={dialogOpen} onClose={() => setDialogOpen(false)} maxWidth="md" fullWidth>
        <DialogTitle>{editingTool ? "도구 수정" : "새 도구 추가"}</DialogTitle>
        <DialogContent>
          <Grid container spacing={2} sx={{ mt: 0.5 }}>
            <Grid size={{ xs: 6 }}>
              <TextField fullWidth label="도구 이름" value={name} onChange={(e) => setName(e.target.value)} size="small" />
            </Grid>
            <Grid size={{ xs: 6 }}>
              <TextField fullWidth label="API URL" value={apiUrl} onChange={(e) => setApiUrl(e.target.value)} size="small" />
            </Grid>
            <Grid size={{ xs: 12 }}>
              <TextField fullWidth label="설명" value={description} onChange={(e) => setDescription(e.target.value)} size="small" />
            </Grid>

            {/* Parameters */}
            <Grid size={{ xs: 12 }}>
              <Box sx={{ display: "flex", justifyContent: "space-between", mb: 1 }}>
                <Typography variant="subtitle2">파라미터</Typography>
                <Button size="small" onClick={addParam} startIcon={<AddIcon />}>추가</Button>
              </Box>
              {params.map((p, i) => (
                <Box key={i} sx={{ display: "flex", gap: 1, mb: 1 }}>
                  <TextField size="small" label="이름" value={p.name} onChange={(e) => updateParam(i, "name", e.target.value)} sx={{ flex: 1 }} />
                  <FormControl size="small" sx={{ minWidth: 100 }}>
                    <InputLabel>타입</InputLabel>
                    <Select value={p.type} label="타입" onChange={(e) => updateParam(i, "type", e.target.value)}>
                      {PARAM_TYPES.map((t) => <MenuItem key={t} value={t}>{t}</MenuItem>)}
                    </Select>
                  </FormControl>
                  <TextField size="small" label="설명" value={p.description} onChange={(e) => updateParam(i, "description", e.target.value)} sx={{ flex: 2 }} />
                  <FormControlLabel control={<Switch checked={p.required} onChange={(e) => updateParam(i, "required", e.target.checked)} size="small" />} label="필수" />
                  <IconButton size="small" onClick={() => removeParam(i)}><DeleteIcon fontSize="small" /></IconButton>
                </Box>
              ))}
            </Grid>

            <Grid size={{ xs: 4 }}>
              <FormControl fullWidth size="small">
                <InputLabel>HTTP 메소드</InputLabel>
                <Select value={apiMethod} label="HTTP 메소드" onChange={(e) => setApiMethod(e.target.value)}>
                  <MenuItem value="GET">GET</MenuItem>
                  <MenuItem value="POST">POST</MenuItem>
                  <MenuItem value="PUT">PUT</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid size={{ xs: 8 }}>
              <TextField fullWidth size="small" label="API 헤더 (JSON)" value={apiHeaders} onChange={(e) => setApiHeaders(e.target.value)} />
            </Grid>
            <Grid size={{ xs: 12 }}>
              <TextField fullWidth size="small" label="요청 바디 템플릿" value={apiBodyTemplate} onChange={(e) => setApiBodyTemplate(e.target.value)} multiline rows={3} placeholder='{"query": "{{query}}", "top_k": {{top_k}}}' />
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDialogOpen(false)}>취소</Button>
          <Button variant="contained" onClick={handleSave} disabled={!name || !description}>저장</Button>
        </DialogActions>
      </Dialog>

      {/* Test Dialog */}
      <Dialog open={testDialogOpen} onClose={() => setTestDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>도구 테스트</DialogTitle>
        <DialogContent>
          <TextField fullWidth label="테스트 입력 (JSON)" value={testInput} onChange={(e) => setTestInput(e.target.value)} multiline rows={3} sx={{ mt: 1, mb: 2 }} />
          <Button variant="contained" onClick={handleTest} startIcon={<PlayArrowIcon />} sx={{ mb: 2 }}>실행</Button>
          {testResult && (
            <Alert severity={testResult.startsWith("Error") ? "error" : "info"} sx={{ whiteSpace: "pre-wrap" }}>
              {testResult}
            </Alert>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setTestDialogOpen(false)}>닫기</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}
