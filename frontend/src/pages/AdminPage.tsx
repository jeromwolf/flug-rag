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
import { adminApi, mcpApi, workflowsApi } from "../api/client";
import Layout from "../components/Layout";

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
        <Tab label="프롬프트 관리" />
        <Tab label="MCP 도구" />
        <Tab label="워크플로우" />
      </Tabs>

      <TabPanel value={tabIndex} index={0}>
        <SystemInfoTab />
      </TabPanel>
      <TabPanel value={tabIndex} index={1}>
        <ProvidersTab />
      </TabPanel>
      <TabPanel value={tabIndex} index={2}>
        <PromptsTab />
      </TabPanel>
      <TabPanel value={tabIndex} index={3}>
        <McpToolsTab />
      </TabPanel>
      <TabPanel value={tabIndex} index={4}>
        <WorkflowsTab />
      </TabPanel>
    </Layout>
  );
}
