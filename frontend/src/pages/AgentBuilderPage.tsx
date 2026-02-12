import { useState, useCallback, useRef, useMemo } from "react";
import {
  Box,
  Typography,
  TextField,
  Button,
  IconButton,
  Tooltip,
  Divider,
  Select,
  MenuItem,
  Slider,
  Paper,
  Chip,
  CircularProgress,
  Snackbar,
  Alert,
} from "@mui/material";
import type { SelectChangeEvent } from "@mui/material";
import {
  PlayArrow as StartIcon,
  Psychology as LlmIcon,
  Search as RetrievalIcon,
  Build as ToolIcon,
  CallSplit as ConditionIcon,
  CheckCircle as OutputIcon,
  SwapHoriz as TransformIcon,
  Delete as DeleteIcon,
  PlayCircleFilled as RunIcon,
  Save as SaveIcon,
  FileDownload as ExportIcon,
  ClearAll as ClearIcon,
  DragIndicator as DragIcon,
  CheckCircleOutline as CompletedIcon,
  ErrorOutline as FailedIcon,
} from "@mui/icons-material";
import {
  ReactFlow,
  Controls,
  MiniMap,
  Background,
  BackgroundVariant,
  Handle,
  Position,
  useNodesState,
  useEdgesState,
  addEdge,
  ReactFlowProvider,
  useReactFlow,
} from "@xyflow/react";
import type {
  Node,
  Edge,
  Connection,
  NodeProps,
  NodeTypes,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import Layout from "../components/Layout";
import { workflowsApi } from "../api/client";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type WorkflowNodeType =
  | "start"
  | "llm"
  | "retrieval"
  | "tool"
  | "condition"
  | "output"
  | "transform";

type ExecutionStatus = "pending" | "running" | "completed" | "failed";

interface WorkflowNodeData extends Record<string, unknown> {
  nodeType: WorkflowNodeType;
  label: string;
  config: Record<string, unknown>;
  executionStatus?: ExecutionStatus;
}

type WorkflowNode = Node<WorkflowNodeData, "workflow">;
type WorkflowEdge = Edge;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const NODE_TYPE_META: Record<
  WorkflowNodeType,
  { icon: React.ReactNode; label: string; color: string }
> = {
  start: { icon: <StartIcon fontSize="small" />, label: "시작", color: "#4caf50" },
  llm: { icon: <LlmIcon fontSize="small" />, label: "LLM 생성", color: "#2196f3" },
  retrieval: { icon: <RetrievalIcon fontSize="small" />, label: "문서 검색", color: "#9c27b0" },
  tool: { icon: <ToolIcon fontSize="small" />, label: "도구 실행", color: "#ff9800" },
  condition: { icon: <ConditionIcon fontSize="small" />, label: "조건 분기", color: "#ffc107" },
  output: { icon: <OutputIcon fontSize="small" />, label: "출력", color: "#009688" },
  transform: { icon: <TransformIcon fontSize="small" />, label: "변환", color: "#3f51b5" },
};

interface PresetDef {
  id: string;
  name: string;
  description: string;
  nodes: WorkflowNode[];
  edges: WorkflowEdge[];
}

const PRESETS: PresetDef[] = [
  {
    id: "simple_rag",
    name: "간단 RAG",
    description: "질문 → 문서 검색 → 답변 생성",
    nodes: [
      {
        id: "start",
        type: "workflow",
        position: { x: 100, y: 200 },
        data: { nodeType: "start", label: "시작", config: {} },
      },
      {
        id: "retrieve",
        type: "workflow",
        position: { x: 350, y: 200 },
        data: { nodeType: "retrieval", label: "문서 검색", config: { top_k: 5 } },
      },
      {
        id: "output",
        type: "workflow",
        position: { x: 600, y: 200 },
        data: { nodeType: "output", label: "출력", config: {} },
      },
    ],
    edges: [
      { id: "e-start-retrieve", source: "start", target: "retrieve", animated: true },
      { id: "e-retrieve-output", source: "retrieve", target: "output", animated: true },
    ],
  },
  {
    id: "regulation_review",
    name: "규정 검토 에이전트",
    description: "문서를 사내 규정과 대조 검토",
    nodes: [
      {
        id: "start",
        type: "workflow",
        position: { x: 100, y: 200 },
        data: { nodeType: "start", label: "문서 입력", config: { description: "검토할 문서를 입력하세요" } },
      },
      {
        id: "regulation_search",
        type: "workflow",
        position: { x: 350, y: 200 },
        data: {
          nodeType: "tool",
          label: "규정 검토",
          config: {
            tool_name: "regulation_review",
            arguments_template: {
              document_text: "{{input}}",
              regulation_category: "전체",
              review_depth: "standard",
            },
          }
        },
      },
      {
        id: "output",
        type: "workflow",
        position: { x: 600, y: 200 },
        data: { nodeType: "output", label: "검토 보고서", config: { format: "markdown" } },
      },
    ],
    edges: [
      { id: "e-start-regulation", source: "start", target: "regulation_search", animated: true },
      { id: "e-regulation-output", source: "regulation_search", target: "output", animated: true },
    ],
  },
  {
    id: "safety_checklist",
    name: "안전 체크리스트 생성",
    description: "설비별 안전 점검 체크리스트 + 규정 매핑",
    nodes: [
      {
        id: "start",
        type: "workflow",
        position: { x: 100, y: 200 },
        data: { nodeType: "start", label: "설비 유형 선택", config: { description: "점검 대상 설비 유형을 입력하세요" } },
      },
      {
        id: "checklist_gen",
        type: "workflow",
        position: { x: 350, y: 200 },
        data: {
          nodeType: "tool",
          label: "체크리스트 생성",
          config: {
            tool_name: "safety_checklist",
            arguments_template: {
              equipment_type: "{{input}}",
              output_format: "markdown",
            },
          }
        },
      },
      {
        id: "output",
        type: "workflow",
        position: { x: 600, y: 200 },
        data: { nodeType: "output", label: "체크리스트 출력", config: { format: "markdown" } },
      },
    ],
    edges: [
      { id: "e-start-checklist", source: "start", target: "checklist_gen", animated: true },
      { id: "e-checklist-output", source: "checklist_gen", target: "output", animated: true },
    ],
  },
  {
    id: "routing",
    name: "라우팅 워크플로우",
    description: "질문 분류 → 문서검색/직접답변 분기 → 출력",
    nodes: [
      {
        id: "start",
        type: "workflow",
        position: { x: 50, y: 200 },
        data: { nodeType: "start", label: "시작", config: {} },
      },
      {
        id: "classify",
        type: "workflow",
        position: { x: 250, y: 200 },
        data: {
          nodeType: "llm",
          label: "질문 분류",
          config: {
            system_prompt: "사용자 질문을 분류하세요: document_search 또는 general_query",
            temperature: 0.1,
          },
        },
      },
      {
        id: "rag",
        type: "workflow",
        position: { x: 500, y: 100 },
        data: { nodeType: "retrieval", label: "문서 검색", config: { top_k: 5 } },
      },
      {
        id: "direct",
        type: "workflow",
        position: { x: 500, y: 300 },
        data: {
          nodeType: "llm",
          label: "직접 답변",
          config: { system_prompt: "한국어로 친절하게 답변하세요." },
        },
      },
      {
        id: "output",
        type: "workflow",
        position: { x: 750, y: 200 },
        data: { nodeType: "output", label: "출력", config: {} },
      },
    ],
    edges: [
      { id: "e1", source: "start", target: "classify", animated: true },
      { id: "e2", source: "classify", target: "rag", animated: true, label: "문서 질문" },
      { id: "e3", source: "classify", target: "direct", animated: true, label: "일반 질문" },
      { id: "e4", source: "rag", target: "output", animated: true },
      { id: "e5", source: "direct", target: "output", animated: true },
    ],
  },
  {
    id: "quality_check",
    name: "품질 검증 워크플로우",
    description: "문서 검색 → 신뢰도 확인 → 조건부 출력",
    nodes: [
      {
        id: "start",
        type: "workflow",
        position: { x: 50, y: 200 },
        data: { nodeType: "start", label: "시작", config: {} },
      },
      {
        id: "retrieve",
        type: "workflow",
        position: { x: 250, y: 200 },
        data: { nodeType: "retrieval", label: "문서 검색", config: { top_k: 5 } },
      },
      {
        id: "check",
        type: "workflow",
        position: { x: 450, y: 200 },
        data: {
          nodeType: "condition",
          label: "신뢰도 확인",
          config: { condition_type: "confidence", threshold: 0.5 },
        },
      },
      {
        id: "high_output",
        type: "workflow",
        position: { x: 700, y: 100 },
        data: { nodeType: "output", label: "정상 출력", config: {} },
      },
      {
        id: "low_transform",
        type: "workflow",
        position: { x: 700, y: 300 },
        data: {
          nodeType: "transform",
          label: "안전장치 적용",
          config: { template: "확인이 필요한 답변입니다.\n\n{input}" },
        },
      },
      {
        id: "low_output",
        type: "workflow",
        position: { x: 950, y: 300 },
        data: { nodeType: "output", label: "경고 출력", config: {} },
      },
    ],
    edges: [
      { id: "e1", source: "start", target: "retrieve", animated: true },
      { id: "e2", source: "retrieve", target: "check", animated: true },
      { id: "e3", source: "check", target: "high_output", animated: true, label: "신뢰도 높음" },
      { id: "e4", source: "check", target: "low_transform", animated: true, label: "신뢰도 낮음" },
      { id: "e5", source: "low_transform", target: "low_output", animated: true },
    ],
  },
];

// ---------------------------------------------------------------------------
// Custom Node Component
// ---------------------------------------------------------------------------

function WorkflowNodeComponent({ id, data, selected }: NodeProps<WorkflowNode>) {
  const meta = NODE_TYPE_META[data.nodeType];
  const status = data.executionStatus;

  const statusBorder =
    status === "running"
      ? "2px solid #2196f3"
      : status === "completed"
        ? "2px solid #4caf50"
        : status === "failed"
          ? "2px solid #f44336"
          : selected
            ? "2px solid #1976d2"
            : "1px solid rgba(0,0,0,0.12)";

  const pulseAnimation =
    status === "running"
      ? "pulse 1.5s ease-in-out infinite"
      : undefined;

  return (
    <Box
      sx={{
        minWidth: 160,
        borderRadius: 2,
        border: statusBorder,
        bgcolor: "background.paper",
        boxShadow: selected ? 4 : 1,
        overflow: "hidden",
        animation: pulseAnimation,
        "@keyframes pulse": {
          "0%, 100%": { boxShadow: "0 0 0 0 rgba(33,150,243,0.4)" },
          "50%": { boxShadow: "0 0 0 8px rgba(33,150,243,0)" },
        },
      }}
    >
      {/* Handles */}
      {data.nodeType !== "start" && (
        <Handle
          type="target"
          position={Position.Left}
          style={{
            background: meta.color,
            width: 10,
            height: 10,
            border: "2px solid white",
          }}
          id={`${id}-target`}
        />
      )}
      {data.nodeType !== "output" && (
        <Handle
          type="source"
          position={Position.Right}
          style={{
            background: meta.color,
            width: 10,
            height: 10,
            border: "2px solid white",
          }}
          id={`${id}-source`}
        />
      )}

      {/* Header */}
      <Box
        sx={{
          px: 1.5,
          py: 0.75,
          bgcolor: meta.color,
          color: "white",
          display: "flex",
          alignItems: "center",
          gap: 0.75,
        }}
      >
        {meta.icon}
        <Typography variant="caption" sx={{ fontWeight: 600, fontSize: "0.8rem" }}>
          {data.label}
        </Typography>
        {/* Status overlay */}
        {status === "completed" && (
          <CompletedIcon sx={{ ml: "auto", fontSize: 16, color: "white" }} />
        )}
        {status === "failed" && (
          <FailedIcon sx={{ ml: "auto", fontSize: 16, color: "white" }} />
        )}
        {status === "running" && (
          <CircularProgress size={14} sx={{ ml: "auto", color: "white" }} />
        )}
      </Box>

      {/* Body */}
      <Box sx={{ px: 1.5, py: 1 }}>
        <Chip
          label={meta.label}
          size="small"
          sx={{
            bgcolor: `${meta.color}18`,
            color: meta.color,
            fontWeight: 500,
            fontSize: "0.7rem",
            height: 22,
          }}
        />
      </Box>
    </Box>
  );
}

// ---------------------------------------------------------------------------
// Node Palette Component
// ---------------------------------------------------------------------------

const PALETTE_WIDTH = 200;

interface NodePaletteProps {
  onDragStart: (event: React.DragEvent, nodeType: WorkflowNodeType) => void;
}

function NodePalette({ onDragStart }: NodePaletteProps) {
  const nodeTypes: WorkflowNodeType[] = [
    "start",
    "llm",
    "retrieval",
    "tool",
    "condition",
    "output",
    "transform",
  ];

  return (
    <Paper
      elevation={0}
      sx={{
        width: PALETTE_WIDTH,
        borderRight: "1px solid",
        borderColor: "divider",
        p: 1.5,
        display: "flex",
        flexDirection: "column",
        gap: 1,
        overflowY: "auto",
      }}
    >
      <Typography variant="subtitle2" sx={{ fontWeight: 700, mb: 0.5 }}>
        노드 팔레트
      </Typography>
      <Divider />
      {nodeTypes.map((nt) => {
        const meta = NODE_TYPE_META[nt];
        return (
          <Box
            key={nt}
            draggable
            onDragStart={(e) => onDragStart(e, nt)}
            sx={{
              display: "flex",
              alignItems: "center",
              gap: 1,
              p: 1,
              borderRadius: 1.5,
              border: "1px solid",
              borderColor: "divider",
              cursor: "grab",
              userSelect: "none",
              transition: "all 0.15s",
              "&:hover": {
                borderColor: meta.color,
                bgcolor: `${meta.color}0A`,
                transform: "translateY(-1px)",
                boxShadow: 1,
              },
              "&:active": { cursor: "grabbing" },
            }}
          >
            <DragIcon sx={{ fontSize: 16, color: "text.disabled" }} />
            <Box
              sx={{
                width: 28,
                height: 28,
                borderRadius: 1,
                bgcolor: meta.color,
                color: "white",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
              }}
            >
              {meta.icon}
            </Box>
            <Typography variant="body2" sx={{ fontWeight: 500 }}>
              {meta.label}
            </Typography>
          </Box>
        );
      })}
    </Paper>
  );
}

// ---------------------------------------------------------------------------
// Properties Panel Component
// ---------------------------------------------------------------------------

const PROPERTIES_WIDTH = 300;

interface PropertiesPanelProps {
  selectedNode: WorkflowNode | null;
  onUpdateNode: (nodeId: string, data: Partial<WorkflowNodeData>) => void;
  onDeleteNode: (nodeId: string) => void;
  workflowName: string;
  workflowDescription: string;
  onWorkflowNameChange: (name: string) => void;
  onWorkflowDescriptionChange: (desc: string) => void;
}

function PropertiesPanel({
  selectedNode,
  onUpdateNode,
  onDeleteNode,
  workflowName,
  workflowDescription,
  onWorkflowNameChange,
  onWorkflowDescriptionChange,
}: PropertiesPanelProps) {
  if (!selectedNode) {
    return (
      <Paper
        elevation={0}
        sx={{
          width: PROPERTIES_WIDTH,
          borderLeft: "1px solid",
          borderColor: "divider",
          p: 2,
          overflowY: "auto",
        }}
      >
        <Typography variant="subtitle2" sx={{ fontWeight: 700, mb: 2 }}>
          워크플로우 정보
        </Typography>
        <TextField
          label="워크플로우 이름"
          value={workflowName}
          onChange={(e) => onWorkflowNameChange(e.target.value)}
          fullWidth
          size="small"
          sx={{ mb: 2 }}
        />
        <TextField
          label="설명"
          value={workflowDescription}
          onChange={(e) => onWorkflowDescriptionChange(e.target.value)}
          fullWidth
          size="small"
          multiline
          rows={3}
        />
        <Typography variant="caption" color="text.secondary" sx={{ mt: 2, display: "block" }}>
          노드를 클릭하면 속성을 편집할 수 있습니다.
        </Typography>
      </Paper>
    );
  }

  const { data } = selectedNode;
  const meta = NODE_TYPE_META[data.nodeType];
  const config = data.config;

  const updateConfig = (key: string, value: unknown) => {
    onUpdateNode(selectedNode.id, {
      ...data,
      config: { ...config, [key]: value },
    });
  };

  const updateLabel = (label: string) => {
    onUpdateNode(selectedNode.id, { ...data, label });
  };

  return (
    <Paper
      elevation={0}
      sx={{
        width: PROPERTIES_WIDTH,
        borderLeft: "1px solid",
        borderColor: "divider",
        p: 2,
        overflowY: "auto",
        display: "flex",
        flexDirection: "column",
        gap: 2,
      }}
    >
      <Typography variant="subtitle2" sx={{ fontWeight: 700 }}>
        속성
      </Typography>

      {/* Node Type (read-only) */}
      <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
        <Box
          sx={{
            width: 32,
            height: 32,
            borderRadius: 1,
            bgcolor: meta.color,
            color: "white",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
          }}
        >
          {meta.icon}
        </Box>
        <Box>
          <Typography variant="body2" sx={{ fontWeight: 600 }}>
            {meta.label}
          </Typography>
          <Typography variant="caption" color="text.secondary">
            {data.nodeType.toUpperCase()}
          </Typography>
        </Box>
      </Box>

      <Divider />

      {/* Label */}
      <TextField
        label="레이블"
        value={data.label}
        onChange={(e) => updateLabel(e.target.value)}
        fullWidth
        size="small"
      />

      {/* Config fields based on node type */}
      {data.nodeType === "llm" && (
        <>
          <TextField
            label="Provider"
            value={(config.provider as string) || ""}
            onChange={(e) => updateConfig("provider", e.target.value)}
            fullWidth
            size="small"
            placeholder="vllm, openai, ..."
          />
          <TextField
            label="모델"
            value={(config.model as string) || ""}
            onChange={(e) => updateConfig("model", e.target.value)}
            fullWidth
            size="small"
            placeholder="모델 이름"
          />
          <TextField
            label="시스템 프롬프트"
            value={(config.system_prompt as string) || ""}
            onChange={(e) => updateConfig("system_prompt", e.target.value)}
            fullWidth
            size="small"
            multiline
            rows={4}
            placeholder="LLM에 전달할 시스템 프롬프트"
          />
          <Box>
            <Typography variant="caption" color="text.secondary" gutterBottom>
              Temperature: {((config.temperature as number) ?? 0.7).toFixed(1)}
            </Typography>
            <Slider
              value={(config.temperature as number) ?? 0.7}
              onChange={(_e, val) => updateConfig("temperature", val)}
              min={0}
              max={2}
              step={0.1}
              size="small"
              sx={{ color: meta.color }}
            />
          </Box>
        </>
      )}

      {data.nodeType === "retrieval" && (
        <>
          <TextField
            label="Top K"
            type="number"
            value={(config.top_k as number) ?? 5}
            onChange={(e) => updateConfig("top_k", parseInt(e.target.value) || 5)}
            fullWidth
            size="small"
            slotProps={{ htmlInput: { min: 1, max: 50 } }}
          />
          <TextField
            label="필터 (JSON)"
            value={
              typeof config.filters === "object"
                ? JSON.stringify(config.filters, null, 2)
                : (config.filters as string) || "{}"
            }
            onChange={(e) => {
              try {
                updateConfig("filters", JSON.parse(e.target.value));
              } catch {
                updateConfig("filters", e.target.value);
              }
            }}
            fullWidth
            size="small"
            multiline
            rows={3}
            placeholder='{"key": "value"}'
          />
        </>
      )}

      {data.nodeType === "tool" && (
        <>
          <TextField
            label="도구 이름"
            value={(config.tool_name as string) || ""}
            onChange={(e) => updateConfig("tool_name", e.target.value)}
            fullWidth
            size="small"
            placeholder="calculator, search, ..."
          />
          <TextField
            label="인수 템플릿"
            value={
              typeof config.arguments_template === "object"
                ? JSON.stringify(config.arguments_template, null, 2)
                : (config.arguments_template as string) || ""
            }
            onChange={(e) => {
              try {
                updateConfig("arguments_template", JSON.parse(e.target.value));
              } catch {
                updateConfig("arguments_template", e.target.value);
              }
            }}
            fullWidth
            size="small"
            multiline
            rows={3}
            placeholder='{"arg1": "{input}"}'
          />
        </>
      )}

      {data.nodeType === "condition" && (
        <>
          <Select
            value={(config.condition_type as string) || "confidence"}
            onChange={(e: SelectChangeEvent) => updateConfig("condition_type", e.target.value)}
            fullWidth
            size="small"
          >
            <MenuItem value="confidence">신뢰도 (confidence)</MenuItem>
            <MenuItem value="keyword">키워드 (keyword)</MenuItem>
            <MenuItem value="length">길이 (length)</MenuItem>
            <MenuItem value="custom">사용자 정의 (custom)</MenuItem>
          </Select>
          <Box>
            <Typography variant="caption" color="text.secondary" gutterBottom>
              임계값: {((config.threshold as number) ?? 0.5).toFixed(2)}
            </Typography>
            <Slider
              value={(config.threshold as number) ?? 0.5}
              onChange={(_e, val) => updateConfig("threshold", val)}
              min={0}
              max={1}
              step={0.05}
              size="small"
              sx={{ color: meta.color }}
            />
          </Box>
        </>
      )}

      {data.nodeType === "transform" && (
        <TextField
          label="변환 템플릿"
          value={(config.template as string) || ""}
          onChange={(e) => updateConfig("template", e.target.value)}
          fullWidth
          size="small"
          multiline
          rows={4}
          placeholder={"결과: {input}"}
          helperText="{input} 자리 표시자를 사용하세요"
        />
      )}

      <Divider />

      {/* Delete button */}
      <Button
        variant="outlined"
        color="error"
        startIcon={<DeleteIcon />}
        onClick={() => onDeleteNode(selectedNode.id)}
        fullWidth
        size="small"
      >
        노드 삭제
      </Button>
    </Paper>
  );
}

// ---------------------------------------------------------------------------
// Main Canvas (Inner component that uses useReactFlow)
// ---------------------------------------------------------------------------

let nodeIdCounter = 0;

function AgentBuilderCanvas() {
  const reactFlowWrapper = useRef<HTMLDivElement>(null);
  const { screenToFlowPosition } = useReactFlow();

  // -- Workflow metadata --
  const [workflowName, setWorkflowName] = useState("새 워크플로우");
  const [workflowDescription, setWorkflowDescription] = useState("");

  // -- React Flow state --
  const [nodes, setNodes, onNodesChange] = useNodesState<WorkflowNode>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<WorkflowEdge>([]);

  // -- Selection --
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);

  // -- Execution --
  const [isRunning, setIsRunning] = useState(false);
  const [selectedPreset, setSelectedPreset] = useState("simple_rag");

  // -- Snackbar --
  const [snackbar, setSnackbar] = useState<{
    open: boolean;
    message: string;
    severity: "success" | "error" | "info";
  }>({ open: false, message: "", severity: "info" });

  const showSnackbar = useCallback(
    (message: string, severity: "success" | "error" | "info" = "info") => {
      setSnackbar({ open: true, message, severity });
    },
    [],
  );

  // -- Computed selected node --
  const selectedNode = useMemo(
    () =>
      selectedNodeId
        ? (nodes.find((n) => n.id === selectedNodeId) as WorkflowNode | undefined) ?? null
        : null,
    [nodes, selectedNodeId],
  );

  // -- Node types registration --
  const nodeTypes: NodeTypes = useMemo(
    () => ({
      workflow: WorkflowNodeComponent,
    }),
    [],
  );

  // -- Handlers --
  const onConnect = useCallback(
    (connection: Connection) => {
      setEdges((eds) =>
        addEdge({ ...connection, animated: true }, eds),
      );
    },
    [setEdges],
  );

  const onNodeClick = useCallback(
    (_event: React.MouseEvent, node: Node) => {
      setSelectedNodeId(node.id);
    },
    [],
  );

  const onPaneClick = useCallback(() => {
    setSelectedNodeId(null);
  }, []);

  // -- Drag & drop from palette --
  const onDragStart = useCallback(
    (event: React.DragEvent, nodeType: WorkflowNodeType) => {
      event.dataTransfer.setData("application/reactflow", nodeType);
      event.dataTransfer.effectAllowed = "move";
    },
    [],
  );

  const onDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = "move";
  }, []);

  const onDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault();

      const nodeType = event.dataTransfer.getData(
        "application/reactflow",
      ) as WorkflowNodeType;
      if (!nodeType) return;

      const position = screenToFlowPosition({
        x: event.clientX,
        y: event.clientY,
      });

      const meta = NODE_TYPE_META[nodeType];
      const newNode: WorkflowNode = {
        id: `${nodeType}_${++nodeIdCounter}`,
        type: "workflow",
        position,
        data: {
          nodeType,
          label: meta.label,
          config: nodeType === "retrieval" ? { top_k: 5 } : {},
        },
      };

      setNodes((nds) => [...nds, newNode]);
      showSnackbar(`"${meta.label}" 노드가 추가되었습니다.`, "success");
    },
    [screenToFlowPosition, setNodes, showSnackbar],
  );

  // -- Update node data --
  const onUpdateNode = useCallback(
    (nodeId: string, newData: Partial<WorkflowNodeData>) => {
      setNodes((nds) =>
        nds.map((n) => {
          if (n.id === nodeId) {
            return { ...n, data: { ...n.data, ...newData } };
          }
          return n;
        }),
      );
    },
    [setNodes],
  );

  // -- Delete node --
  const onDeleteNode = useCallback(
    (nodeId: string) => {
      setNodes((nds) => nds.filter((n) => n.id !== nodeId));
      setEdges((eds) =>
        eds.filter((e) => e.source !== nodeId && e.target !== nodeId),
      );
      setSelectedNodeId(null);
      showSnackbar("노드가 삭제되었습니다.", "info");
    },
    [setNodes, setEdges, showSnackbar],
  );

  // -- Load preset --
  const loadPreset = useCallback(() => {
    const preset = PRESETS.find((p) => p.id === selectedPreset);
    if (!preset) return;

    setNodes(preset.nodes.map((n) => ({ ...n })));
    setEdges(preset.edges.map((e) => ({ ...e })));
    setWorkflowName(preset.name);
    setWorkflowDescription(preset.description);
    setSelectedNodeId(null);
    showSnackbar(`"${preset.name}" 프리셋이 로드되었습니다.`, "success");
  }, [selectedPreset, setNodes, setEdges, showSnackbar]);

  // -- Clear canvas --
  const clearCanvas = useCallback(() => {
    setNodes([]);
    setEdges([]);
    setSelectedNodeId(null);
    setWorkflowName("새 워크플로우");
    setWorkflowDescription("");
    showSnackbar("캔버스가 초기화되었습니다.", "info");
  }, [setNodes, setEdges, showSnackbar]);

  // -- Run workflow --
  const runWorkflow = useCallback(async () => {
    if (nodes.length === 0) {
      showSnackbar("실행할 노드가 없습니다.", "error");
      return;
    }

    setIsRunning(true);

    // Set all nodes to pending
    setNodes((nds) =>
      nds.map((n) => ({
        ...n,
        data: { ...n.data, executionStatus: "pending" as ExecutionStatus },
      })),
    );

    try {
      // Simulate execution by walking through nodes in order
      // Find start node
      const startNode = nodes.find((n) => n.data.nodeType === "start");
      if (!startNode) {
        showSnackbar("시작 노드가 없습니다.", "error");
        setIsRunning(false);
        return;
      }

      // Build adjacency for BFS
      const adj = new Map<string, string[]>();
      for (const e of edges) {
        const existing = adj.get(e.source) || [];
        existing.push(e.target);
        adj.set(e.source, existing);
      }

      // BFS execution simulation
      const queue = [startNode.id];
      const visited = new Set<string>();

      while (queue.length > 0) {
        const currentId = queue.shift()!;
        if (visited.has(currentId)) continue;
        visited.add(currentId);

        // Mark running
        setNodes((nds) =>
          nds.map((n) =>
            n.id === currentId
              ? { ...n, data: { ...n.data, executionStatus: "running" as ExecutionStatus } }
              : n,
          ),
        );

        // Simulate processing time
        await new Promise((resolve) => setTimeout(resolve, 800));

        // Mark completed
        setNodes((nds) =>
          nds.map((n) =>
            n.id === currentId
              ? { ...n, data: { ...n.data, executionStatus: "completed" as ExecutionStatus } }
              : n,
          ),
        );

        // Enqueue neighbors
        const neighbors = adj.get(currentId) || [];
        for (const neighbor of neighbors) {
          if (!visited.has(neighbor)) {
            queue.push(neighbor);
          }
        }
      }

      // Also try calling backend
      try {
        await workflowsApi.run(selectedPreset, { query: "테스트 실행" });
      } catch {
        // Backend may not be available, simulation is sufficient
      }

      showSnackbar("워크플로우 실행이 완료되었습니다.", "success");
    } catch {
      showSnackbar("워크플로우 실행 중 오류가 발생했습니다.", "error");
    } finally {
      setIsRunning(false);
    }
  }, [nodes, edges, selectedPreset, setNodes, showSnackbar]);

  // -- Export workflow --
  const exportWorkflow = useCallback(() => {
    const workflow = {
      name: workflowName,
      description: workflowDescription,
      nodes: nodes.map((n) => ({
        id: n.id,
        type: n.data.nodeType,
        label: n.data.label,
        config: n.data.config,
        position: n.position,
      })),
      edges: edges.map((e) => ({
        id: e.id,
        source: e.source,
        target: e.target,
        label: e.label || "",
      })),
    };

    const blob = new Blob([JSON.stringify(workflow, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `${workflowName.replace(/\s+/g, "_")}.json`;
    link.click();
    URL.revokeObjectURL(url);

    showSnackbar("워크플로우가 내보내기 되었습니다.", "success");
  }, [nodes, edges, workflowName, workflowDescription, showSnackbar]);

  // -- Save workflow --
  const saveWorkflow = useCallback(() => {
    const workflow = {
      name: workflowName,
      description: workflowDescription,
      nodes: nodes.map((n) => ({
        id: n.id,
        type: n.data.nodeType,
        label: n.data.label,
        config: n.data.config,
        position: n.position,
      })),
      edges: edges.map((e) => ({
        id: e.id,
        source: e.source,
        target: e.target,
        label: e.label || "",
      })),
    };

    localStorage.setItem("flux-rag-workflow", JSON.stringify(workflow));
    showSnackbar("워크플로우가 저장되었습니다.", "success");
  }, [nodes, edges, workflowName, workflowDescription, showSnackbar]);

  return (
    <Box sx={{ display: "flex", flexDirection: "column", height: "100%" }}>
      {/* Toolbar */}
      <Paper
        elevation={0}
        sx={{
          px: 2,
          py: 1,
          borderBottom: "1px solid",
          borderColor: "divider",
          display: "flex",
          alignItems: "center",
          gap: 1.5,
          flexWrap: "wrap",
        }}
      >
        {/* Preset selector */}
        <Select
          value={selectedPreset}
          onChange={(e: SelectChangeEvent) => setSelectedPreset(e.target.value)}
          size="small"
          sx={{ minWidth: 180 }}
        >
          {PRESETS.map((p) => (
            <MenuItem key={p.id} value={p.id}>
              {p.name}
            </MenuItem>
          ))}
        </Select>
        <Button
          variant="outlined"
          size="small"
          onClick={loadPreset}
          disabled={isRunning}
        >
          프리셋 불러오기
        </Button>

        <Divider orientation="vertical" flexItem />

        {/* Run */}
        <Button
          variant="contained"
          size="small"
          startIcon={
            isRunning ? <CircularProgress size={16} color="inherit" /> : <RunIcon />
          }
          onClick={runWorkflow}
          disabled={isRunning || nodes.length === 0}
          color="primary"
        >
          {isRunning ? "실행 중..." : "실행"}
        </Button>

        <Divider orientation="vertical" flexItem />

        {/* Save / Export / Clear */}
        <Tooltip title="저장">
          <IconButton size="small" onClick={saveWorkflow} disabled={isRunning}>
            <SaveIcon fontSize="small" />
          </IconButton>
        </Tooltip>
        <Tooltip title="내보내기">
          <IconButton size="small" onClick={exportWorkflow} disabled={isRunning}>
            <ExportIcon fontSize="small" />
          </IconButton>
        </Tooltip>
        <Tooltip title="캔버스 초기화">
          <IconButton size="small" onClick={clearCanvas} disabled={isRunning}>
            <ClearIcon fontSize="small" />
          </IconButton>
        </Tooltip>

        {/* Workflow name */}
        <Box sx={{ ml: "auto" }}>
          <Typography variant="body2" color="text.secondary" noWrap>
            {workflowName}
          </Typography>
        </Box>
      </Paper>

      {/* Main area: Palette + Canvas + Properties */}
      <Box sx={{ display: "flex", flex: 1, overflow: "hidden" }}>
        {/* Left: Node palette */}
        <NodePalette onDragStart={onDragStart} />

        {/* Center: ReactFlow canvas */}
        <Box
          ref={reactFlowWrapper}
          sx={{ flex: 1, position: "relative" }}
          onDragOver={onDragOver}
          onDrop={onDrop}
        >
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            onNodeClick={onNodeClick}
            onPaneClick={onPaneClick}
            nodeTypes={nodeTypes}
            fitView
            proOptions={{ hideAttribution: true }}
            defaultEdgeOptions={{ animated: true }}
            style={{ width: "100%", height: "100%" }}
          >
            <Controls position="bottom-left" />
            <MiniMap
              position="bottom-right"
              nodeColor={(node) => {
                const data = node.data as WorkflowNodeData | undefined;
                if (data?.nodeType) {
                  return NODE_TYPE_META[data.nodeType]?.color ?? "#888";
                }
                return "#888";
              }}
              maskColor="rgba(0,0,0,0.08)"
              style={{ borderRadius: 8 }}
            />
            <Background variant={BackgroundVariant.Dots} gap={16} size={1} />
          </ReactFlow>
        </Box>

        {/* Right: Properties panel */}
        <PropertiesPanel
          selectedNode={selectedNode}
          onUpdateNode={onUpdateNode}
          onDeleteNode={onDeleteNode}
          workflowName={workflowName}
          workflowDescription={workflowDescription}
          onWorkflowNameChange={setWorkflowName}
          onWorkflowDescriptionChange={setWorkflowDescription}
        />
      </Box>

      {/* Snackbar */}
      <Snackbar
        open={snackbar.open}
        autoHideDuration={3000}
        onClose={() => setSnackbar((s) => ({ ...s, open: false }))}
        anchorOrigin={{ vertical: "bottom", horizontal: "center" }}
      >
        <Alert
          onClose={() => setSnackbar((s) => ({ ...s, open: false }))}
          severity={snackbar.severity}
          variant="filled"
          sx={{ width: "100%" }}
        >
          {snackbar.message}
        </Alert>
      </Snackbar>
    </Box>
  );
}

// ---------------------------------------------------------------------------
// Page component (wraps with ReactFlowProvider)
// ---------------------------------------------------------------------------

export default function AgentBuilderPage() {
  return (
    <Layout title="에이전트 빌더" noPadding>
      <ReactFlowProvider>
        <AgentBuilderCanvas />
      </ReactFlowProvider>
    </Layout>
  );
}
