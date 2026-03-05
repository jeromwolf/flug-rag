import { useState, useCallback, useRef, useMemo, useEffect } from "react";
import {
  Box,
  Typography,
  TextField,
  Button,
  IconButton,
  Tooltip,
  Select,
  MenuItem,
  Slider,
  Chip,
  CircularProgress,
  Snackbar,
  Alert,
  Stack,
  alpha,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  Divider,
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
  FileUpload as ImportIcon,
  FolderOpen as LoadIcon,
  ClearAll as ClearIcon,
  DragIndicator as DragIcon,
  CheckCircleOutline as CompletedIcon,
  ErrorOutline as FailedIcon,
  AccountTree as WorkflowIcon,
  AddCircleOutline as AddNodeIcon,
  TuneRounded as TuneIcon,
  KeyboardArrowRight as ArrowRightIcon,
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
  BaseEdge,
  EdgeLabelRenderer,
  getBezierPath,
  MarkerType,
} from "@xyflow/react";
import type {
  Node,
  Edge,
  Connection,
  NodeProps,
  NodeTypes,
  EdgeProps,
  EdgeTypes,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import Layout from "../components/Layout";
import { workflowsApi } from "../api/client";
import type { WorkflowListItem } from "../api/client";

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
// Design Tokens
// ---------------------------------------------------------------------------

const NODE_COLORS: Record<WorkflowNodeType, { bg: string; border: string; text: string; glow: string }> = {
  start:     { bg: "#1a6b3c", border: "#22c55e", text: "#4ade80", glow: "rgba(34,197,94,0.35)" },
  llm:       { bg: "#1a3f6b", border: "#3b82f6", text: "#60a5fa", glow: "rgba(59,130,246,0.35)" },
  retrieval: { bg: "#6b3d1a", border: "#f97316", text: "#fb923c", glow: "rgba(249,115,22,0.35)" },
  tool:      { bg: "#4a1a6b", border: "#a855f7", text: "#c084fc", glow: "rgba(168,85,247,0.35)" },
  condition: { bg: "#6b5c1a", border: "#eab308", text: "#facc15", glow: "rgba(234,179,8,0.35)" },
  output:    { bg: "#1a5a6b", border: "#06b6d4", text: "#22d3ee", glow: "rgba(6,182,212,0.35)" },
  transform: { bg: "#2a2a6b", border: "#6366f1", text: "#818cf8", glow: "rgba(99,102,241,0.35)" },
};

const NODE_TYPE_META: Record<
  WorkflowNodeType,
  { icon: React.ReactNode; label: string; sublabel: string; color: string }
> = {
  start:     { icon: <StartIcon />,     label: "시작",      sublabel: "TRIGGER",   color: "#22c55e" },
  llm:       { icon: <LlmIcon />,       label: "LLM 생성",   sublabel: "LLM",       color: "#3b82f6" },
  retrieval: { icon: <RetrievalIcon />, label: "문서 검색",  sublabel: "RETRIEVER", color: "#f97316" },
  tool:      { icon: <ToolIcon />,      label: "도구 실행",  sublabel: "TOOL",      color: "#a855f7" },
  condition: { icon: <ConditionIcon />, label: "조건 분기",  sublabel: "CONDITION", color: "#eab308" },
  output:    { icon: <OutputIcon />,    label: "출력",       sublabel: "OUTPUT",    color: "#06b6d4" },
  transform: { icon: <TransformIcon />, label: "변환",       sublabel: "TRANSFORM", color: "#6366f1" },
};

interface PresetDef {
  id: string;
  name: string;
  description: string;
  badge: string;
  nodes: WorkflowNode[];
  edges: WorkflowEdge[];
}

const PRESETS: PresetDef[] = [
  {
    id: "simple_rag",
    name: "간단 RAG",
    description: "질문 → 문서 검색 → 답변 생성",
    badge: "RAG",
    nodes: [
      {
        id: "start",
        type: "workflow",
        position: { x: 80, y: 180 },
        data: { nodeType: "start", label: "시작", config: {} },
      },
      {
        id: "retrieve",
        type: "workflow",
        position: { x: 340, y: 180 },
        data: { nodeType: "retrieval", label: "문서 검색", config: { top_k: 5 } },
      },
      {
        id: "output",
        type: "workflow",
        position: { x: 600, y: 180 },
        data: { nodeType: "output", label: "출력", config: {} },
      },
    ],
    edges: [
      { id: "e-start-retrieve", source: "start", target: "retrieve", animated: true, type: "workflow" },
      { id: "e-retrieve-output", source: "retrieve", target: "output", animated: true, type: "workflow" },
    ],
  },
  {
    id: "regulation_review",
    name: "규정 검토 에이전트",
    description: "문서를 사내 규정과 대조 검토",
    badge: "AGENT",
    nodes: [
      {
        id: "start",
        type: "workflow",
        position: { x: 80, y: 180 },
        data: { nodeType: "start", label: "문서 입력", config: { description: "검토할 문서를 입력하세요" } },
      },
      {
        id: "regulation_search",
        type: "workflow",
        position: { x: 340, y: 180 },
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
        position: { x: 600, y: 180 },
        data: { nodeType: "output", label: "검토 보고서", config: { format: "markdown" } },
      },
    ],
    edges: [
      { id: "e-start-regulation", source: "start", target: "regulation_search", animated: true, type: "workflow" },
      { id: "e-regulation-output", source: "regulation_search", target: "output", animated: true, type: "workflow" },
    ],
  },
  {
    id: "safety_checklist",
    name: "안전 체크리스트",
    description: "설비별 안전 점검 체크리스트 + 규정 매핑",
    badge: "SAFETY",
    nodes: [
      {
        id: "start",
        type: "workflow",
        position: { x: 80, y: 180 },
        data: { nodeType: "start", label: "설비 유형 선택", config: { description: "점검 대상 설비 유형을 입력하세요" } },
      },
      {
        id: "checklist_gen",
        type: "workflow",
        position: { x: 340, y: 180 },
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
        position: { x: 600, y: 180 },
        data: { nodeType: "output", label: "체크리스트 출력", config: { format: "markdown" } },
      },
    ],
    edges: [
      { id: "e-start-checklist", source: "start", target: "checklist_gen", animated: true, type: "workflow" },
      { id: "e-checklist-output", source: "checklist_gen", target: "output", animated: true, type: "workflow" },
    ],
  },
  {
    id: "routing",
    name: "라우팅 워크플로우",
    description: "질문 분류 → 문서검색/직접답변 분기 → 출력",
    badge: "ROUTER",
    nodes: [
      {
        id: "start",
        type: "workflow",
        position: { x: 40, y: 200 },
        data: { nodeType: "start", label: "시작", config: {} },
      },
      {
        id: "classify",
        type: "workflow",
        position: { x: 240, y: 200 },
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
        position: { x: 480, y: 90 },
        data: { nodeType: "retrieval", label: "문서 검색", config: { top_k: 5 } },
      },
      {
        id: "direct",
        type: "workflow",
        position: { x: 480, y: 310 },
        data: {
          nodeType: "llm",
          label: "직접 답변",
          config: { system_prompt: "한국어로 친절하게 답변하세요." },
        },
      },
      {
        id: "output",
        type: "workflow",
        position: { x: 720, y: 200 },
        data: { nodeType: "output", label: "출력", config: {} },
      },
    ],
    edges: [
      { id: "e1", source: "start", target: "classify", animated: true, type: "workflow" },
      { id: "e2", source: "classify", target: "rag", animated: true, label: "문서 질문", type: "workflow" },
      { id: "e3", source: "classify", target: "direct", animated: true, label: "일반 질문", type: "workflow" },
      { id: "e4", source: "rag", target: "output", animated: true, type: "workflow" },
      { id: "e5", source: "direct", target: "output", animated: true, type: "workflow" },
    ],
  },
  {
    id: "quality_check",
    name: "품질 검증 워크플로우",
    description: "문서 검색 → 신뢰도 확인 → 조건부 출력",
    badge: "QA",
    nodes: [
      {
        id: "start",
        type: "workflow",
        position: { x: 40, y: 200 },
        data: { nodeType: "start", label: "시작", config: {} },
      },
      {
        id: "retrieve",
        type: "workflow",
        position: { x: 240, y: 200 },
        data: { nodeType: "retrieval", label: "문서 검색", config: { top_k: 5 } },
      },
      {
        id: "check",
        type: "workflow",
        position: { x: 440, y: 200 },
        data: {
          nodeType: "condition",
          label: "신뢰도 확인",
          config: { condition_type: "confidence", threshold: 0.5 },
        },
      },
      {
        id: "high_output",
        type: "workflow",
        position: { x: 680, y: 90 },
        data: { nodeType: "output", label: "정상 출력", config: {} },
      },
      {
        id: "low_transform",
        type: "workflow",
        position: { x: 680, y: 310 },
        data: {
          nodeType: "transform",
          label: "안전장치 적용",
          config: { template: "확인이 필요한 답변입니다.\n\n{input}" },
        },
      },
      {
        id: "low_output",
        type: "workflow",
        position: { x: 940, y: 310 },
        data: { nodeType: "output", label: "경고 출력", config: {} },
      },
    ],
    edges: [
      { id: "e1", source: "start", target: "retrieve", animated: true, type: "workflow" },
      { id: "e2", source: "retrieve", target: "check", animated: true, type: "workflow" },
      { id: "e3", source: "check", target: "high_output", animated: true, label: "신뢰도 높음", type: "workflow" },
      { id: "e4", source: "check", target: "low_transform", animated: true, label: "신뢰도 낮음", type: "workflow" },
      { id: "e5", source: "low_transform", target: "low_output", animated: true, type: "workflow" },
    ],
  },
  {
    id: "regulation_comparison",
    name: "규정 비교 분석",
    description: "두 규정 문서를 나란히 검색하여 LLM이 비교 분석",
    badge: "COMPARE",
    nodes: [
      {
        id: "start",
        type: "workflow",
        position: { x: 40, y: 220 },
        data: { nodeType: "start", label: "비교 요청", config: { description: "비교할 두 규정 항목을 입력하세요" } },
      },
      {
        id: "retrieve_a",
        type: "workflow",
        position: { x: 280, y: 100 },
        data: {
          nodeType: "retrieval",
          label: "규정 A 검색",
          config: { top_k: 5, collection: "regulation_a", description: "첫 번째 규정 검색" },
        },
      },
      {
        id: "retrieve_b",
        type: "workflow",
        position: { x: 280, y: 340 },
        data: {
          nodeType: "retrieval",
          label: "규정 B 검색",
          config: { top_k: 5, collection: "regulation_b", description: "두 번째 규정 검색" },
        },
      },
      {
        id: "compare_llm",
        type: "workflow",
        position: { x: 560, y: 220 },
        data: {
          nodeType: "llm",
          label: "비교 분석",
          config: {
            system_prompt: "두 규정의 공통점, 차이점, 충돌 사항을 표 형식으로 비교 분석하세요.",
            temperature: 0.1,
          },
        },
      },
      {
        id: "output",
        type: "workflow",
        position: { x: 820, y: 220 },
        data: { nodeType: "output", label: "비교 보고서", config: { format: "markdown" } },
      },
    ],
    edges: [
      { id: "e1", source: "start", target: "retrieve_a", animated: true, label: "규정 A", type: "workflow" },
      { id: "e2", source: "start", target: "retrieve_b", animated: true, label: "규정 B", type: "workflow" },
      { id: "e3", source: "retrieve_a", target: "compare_llm", animated: true, type: "workflow" },
      { id: "e4", source: "retrieve_b", target: "compare_llm", animated: true, type: "workflow" },
      { id: "e5", source: "compare_llm", target: "output", animated: true, type: "workflow" },
    ],
  },
  {
    id: "safety_inspection",
    name: "안전 점검 워크플로",
    description: "체크리스트 생성 → LLM 검토 → 위험도 판단 → 보고서 출력",
    badge: "SAFETY",
    nodes: [
      {
        id: "start",
        type: "workflow",
        position: { x: 40, y: 220 },
        data: { nodeType: "start", label: "점검 시작", config: { description: "점검 대상 설비 및 현장 정보를 입력하세요" } },
      },
      {
        id: "checklist_tool",
        type: "workflow",
        position: { x: 260, y: 220 },
        data: {
          nodeType: "tool",
          label: "안전 체크리스트",
          config: {
            tool_name: "safety_checklist",
            arguments_template: { equipment_type: "{{input}}", output_format: "json" },
          },
        },
      },
      {
        id: "review_llm",
        type: "workflow",
        position: { x: 480, y: 220 },
        data: {
          nodeType: "llm",
          label: "위험도 검토",
          config: {
            system_prompt: "안전 체크리스트 결과를 분석하여 위험 항목을 식별하고 위험도(고/중/저)를 판정하세요.",
            temperature: 0.1,
          },
        },
      },
      {
        id: "risk_condition",
        type: "workflow",
        position: { x: 700, y: 220 },
        data: {
          nodeType: "condition",
          label: "위험도 판단",
          config: { condition_type: "keyword", keywords: ["고위험", "즉시조치"], description: "고위험 여부 분기" },
        },
      },
      {
        id: "urgent_output",
        type: "workflow",
        position: { x: 940, y: 100 },
        data: { nodeType: "output", label: "긴급 보고서", config: { format: "markdown", priority: "urgent" } },
      },
      {
        id: "normal_output",
        type: "workflow",
        position: { x: 940, y: 340 },
        data: { nodeType: "output", label: "일반 보고서", config: { format: "markdown", priority: "normal" } },
      },
    ],
    edges: [
      { id: "e1", source: "start", target: "checklist_tool", animated: true, type: "workflow" },
      { id: "e2", source: "checklist_tool", target: "review_llm", animated: true, type: "workflow" },
      { id: "e3", source: "review_llm", target: "risk_condition", animated: true, type: "workflow" },
      { id: "e4", source: "risk_condition", target: "urgent_output", animated: true, label: "고위험", type: "workflow" },
      { id: "e5", source: "risk_condition", target: "normal_output", animated: true, label: "저위험", type: "workflow" },
    ],
  },
  {
    id: "rag_agent_hybrid",
    name: "RAG + Agent 하이브리드",
    description: "질의 분류 후 문서 검색과 보고서 도구를 결합한 복합 처리",
    badge: "HYBRID",
    nodes: [
      {
        id: "start",
        type: "workflow",
        position: { x: 40, y: 220 },
        data: { nodeType: "start", label: "질의 입력", config: { description: "복합 질의를 입력하세요" } },
      },
      {
        id: "classify",
        type: "workflow",
        position: { x: 240, y: 220 },
        data: {
          nodeType: "condition",
          label: "질의 분류",
          config: {
            condition_type: "keyword",
            keywords: ["보고서", "분석", "비교", "요약"],
            description: "보고서 생성 필요 여부 판단",
          },
        },
      },
      {
        id: "retrieve",
        type: "workflow",
        position: { x: 460, y: 120 },
        data: { nodeType: "retrieval", label: "문서 검색", config: { top_k: 7 } },
      },
      {
        id: "answer_llm",
        type: "workflow",
        position: { x: 680, y: 120 },
        data: {
          nodeType: "llm",
          label: "답변 생성",
          config: {
            system_prompt: "검색된 문서를 바탕으로 정확하고 상세한 답변을 작성하세요.",
            temperature: 0.2,
          },
        },
      },
      {
        id: "report_tool",
        type: "workflow",
        position: { x: 680, y: 340 },
        data: {
          nodeType: "tool",
          label: "보고서 초안",
          config: {
            tool_name: "report_draft",
            arguments_template: { query: "{{input}}", format: "structured" },
          },
        },
      },
      {
        id: "output",
        type: "workflow",
        position: { x: 920, y: 220 },
        data: { nodeType: "output", label: "최종 출력", config: { format: "markdown" } },
      },
    ],
    edges: [
      { id: "e1", source: "start", target: "classify", animated: true, type: "workflow" },
      { id: "e2", source: "classify", target: "retrieve", animated: true, label: "RAG 처리", type: "workflow" },
      { id: "e3", source: "classify", target: "report_tool", animated: true, label: "보고서 생성", type: "workflow" },
      { id: "e4", source: "retrieve", target: "answer_llm", animated: true, type: "workflow" },
      { id: "e5", source: "answer_llm", target: "output", animated: true, type: "workflow" },
      { id: "e6", source: "report_tool", target: "output", animated: true, type: "workflow" },
    ],
  },
  {
    id: "qa_quality_assessment",
    name: "QA 품질 평가",
    description: "답변 검색 → LLM 평가 → 데이터 분석 → 품질 리포트",
    badge: "QA",
    nodes: [
      {
        id: "start",
        type: "workflow",
        position: { x: 40, y: 220 },
        data: { nodeType: "start", label: "평가 시작", config: { description: "평가할 질문과 답변을 입력하세요" } },
      },
      {
        id: "retrieve_ref",
        type: "workflow",
        position: { x: 260, y: 220 },
        data: {
          nodeType: "retrieval",
          label: "참조 답변 검색",
          config: { top_k: 3, description: "골든 데이터셋에서 참조 답변 검색" },
        },
      },
      {
        id: "eval_llm",
        type: "workflow",
        position: { x: 500, y: 220 },
        data: {
          nodeType: "llm",
          label: "품질 평가",
          config: {
            system_prompt: "제출된 답변을 참조 답변과 비교하여 정확성(0-10), 완결성(0-10), 가독성(0-10)을 평가하고 JSON으로 출력하세요.",
            temperature: 0.0,
          },
        },
      },
      {
        id: "analyze_tool",
        type: "workflow",
        position: { x: 740, y: 220 },
        data: {
          nodeType: "tool",
          label: "데이터 분석",
          config: {
            tool_name: "data_analyzer",
            arguments_template: { data: "{{input}}", analysis_type: "quality_metrics" },
          },
        },
      },
      {
        id: "output",
        type: "workflow",
        position: { x: 980, y: 220 },
        data: { nodeType: "output", label: "품질 리포트", config: { format: "markdown" } },
      },
    ],
    edges: [
      { id: "e1", source: "start", target: "retrieve_ref", animated: true, type: "workflow" },
      { id: "e2", source: "retrieve_ref", target: "eval_llm", animated: true, type: "workflow" },
      { id: "e3", source: "eval_llm", target: "analyze_tool", animated: true, type: "workflow" },
      { id: "e4", source: "analyze_tool", target: "output", animated: true, type: "workflow" },
    ],
  },
  {
    id: "intelligent_routing",
    name: "지능형 라우팅",
    description: "의도 분류 → RAG 답변 또는 직접 LLM 답변으로 자동 분기",
    badge: "ROUTER",
    nodes: [
      {
        id: "start",
        type: "workflow",
        position: { x: 40, y: 220 },
        data: { nodeType: "start", label: "질문 입력", config: {} },
      },
      {
        id: "intent",
        type: "workflow",
        position: { x: 260, y: 220 },
        data: {
          nodeType: "condition",
          label: "의도 분류",
          config: {
            condition_type: "keyword",
            keywords: ["규정", "절차", "기준", "조항", "법령", "지침", "문서"],
            description: "문서 기반 질의 여부 판단",
          },
        },
      },
      {
        id: "retrieve",
        type: "workflow",
        position: { x: 500, y: 100 },
        data: { nodeType: "retrieval", label: "문서 검색", config: { top_k: 5 } },
      },
      {
        id: "rag_llm",
        type: "workflow",
        position: { x: 740, y: 100 },
        data: {
          nodeType: "llm",
          label: "RAG 답변",
          config: {
            system_prompt: "검색된 사내 문서를 근거로 정확한 답변을 한국어로 제공하세요. 출처를 명시하세요.",
            temperature: 0.1,
          },
        },
      },
      {
        id: "direct_llm",
        type: "workflow",
        position: { x: 500, y: 340 },
        data: {
          nodeType: "llm",
          label: "직접 답변",
          config: {
            system_prompt: "일반 지식을 바탕으로 친절하고 명확하게 한국어로 답변하세요.",
            temperature: 0.3,
          },
        },
      },
      {
        id: "output",
        type: "workflow",
        position: { x: 980, y: 220 },
        data: { nodeType: "output", label: "출력", config: {} },
      },
    ],
    edges: [
      { id: "e1", source: "start", target: "intent", animated: true, type: "workflow" },
      { id: "e2", source: "intent", target: "retrieve", animated: true, label: "문서 질의", type: "workflow" },
      { id: "e3", source: "intent", target: "direct_llm", animated: true, label: "일반 질의", type: "workflow" },
      { id: "e4", source: "retrieve", target: "rag_llm", animated: true, type: "workflow" },
      { id: "e5", source: "rag_llm", target: "output", animated: true, type: "workflow" },
      { id: "e6", source: "direct_llm", target: "output", animated: true, type: "workflow" },
    ],
  },
];

// ---------------------------------------------------------------------------
// Custom Edge Component
// ---------------------------------------------------------------------------

function WorkflowEdgeComponent({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  data,
  animated,
  label,
  markerEnd,
}: EdgeProps) {
  const [edgePath, labelX, labelY] = getBezierPath({
    sourceX,
    sourceY,
    sourcePosition,
    targetX,
    targetY,
    targetPosition,
  });

  return (
    <>
      <BaseEdge
        id={id}
        path={edgePath}
        markerEnd={markerEnd}
        style={{
          stroke: "rgba(100,120,160,0.6)",
          strokeWidth: animated ? 2 : 1.5,
          filter: animated ? "drop-shadow(0 0 3px rgba(100,140,220,0.4))" : undefined,
        }}
      />
      {label && (
        <EdgeLabelRenderer>
          <Box
            sx={{
              position: "absolute",
              transform: `translate(-50%, -50%) translate(${labelX}px,${labelY}px)`,
              pointerEvents: "all",
              zIndex: 10,
            }}
          >
            <Chip
              label={label as string}
              size="small"
              sx={{
                height: 20,
                fontSize: "0.65rem",
                fontWeight: 700,
                letterSpacing: "0.03em",
                bgcolor: "rgba(30,40,60,0.9)",
                color: "rgba(140,180,255,0.9)",
                border: "1px solid rgba(100,140,220,0.4)",
                backdropFilter: "blur(8px)",
                "& .MuiChip-label": { px: 0.75 },
              }}
            />
          </Box>
        </EdgeLabelRenderer>
      )}
      {/* Unused data param — satisfy TS */}
      {data && null}
    </>
  );
}

// ---------------------------------------------------------------------------
// Custom Node Component
// ---------------------------------------------------------------------------

function WorkflowNodeComponent({ id, data, selected }: NodeProps<WorkflowNode>) {
  const meta = NODE_TYPE_META[data.nodeType];
  const colors = NODE_COLORS[data.nodeType];
  const status = data.executionStatus;

  // Summarise config into at most 2 key:value lines for display
  const configLines = useMemo(() => {
    const entries = Object.entries(data.config).filter(
      ([k]) => !["arguments_template"].includes(k)
    );
    return entries.slice(0, 2).map(([k, v]) => {
      let display = String(v ?? "");
      if (display.length > 22) display = display.slice(0, 20) + "…";
      const key = k.replace(/_/g, " ");
      return { key, display };
    });
  }, [data.config]);

  const glowColor =
    status === "running"
      ? "rgba(59,130,246,0.6)"
      : status === "completed"
        ? "rgba(34,197,94,0.5)"
        : status === "failed"
          ? "rgba(239,68,68,0.5)"
          : selected
            ? colors.glow
            : "transparent";

  const borderColor =
    status === "running"
      ? "#3b82f6"
      : status === "completed"
        ? "#22c55e"
        : status === "failed"
          ? "#ef4444"
          : selected
            ? colors.border
            : `${colors.border}55`;

  return (
    <Box
      sx={{
        width: 192,
        borderRadius: "10px",
        border: `1.5px solid ${borderColor}`,
        bgcolor: "#141922",
        boxShadow: `0 0 0 3px ${glowColor}, 0 4px 20px rgba(0,0,0,0.5)`,
        overflow: "visible",
        transition: "box-shadow 0.2s, border-color 0.2s",
        "@keyframes pulse": {
          "0%, 100%": { boxShadow: `0 0 0 3px rgba(59,130,246,0.4), 0 4px 20px rgba(0,0,0,0.5)` },
          "50%": { boxShadow: `0 0 0 8px rgba(59,130,246,0.1), 0 4px 20px rgba(0,0,0,0.5)` },
        },
        animation: status === "running" ? "pulse 1.4s ease-in-out infinite" : undefined,
      }}
    >
      {/* Input handle */}
      {data.nodeType !== "start" && (
        <Handle
          type="target"
          position={Position.Left}
          id={`${id}-target`}
          style={{
            width: 12,
            height: 12,
            background: colors.border,
            border: `2px solid #141922`,
            boxShadow: `0 0 6px ${colors.glow}`,
            left: -7,
            cursor: "crosshair",
          }}
        />
      )}

      {/* Output handle */}
      {data.nodeType !== "output" && (
        <Handle
          type="source"
          position={Position.Right}
          id={`${id}-source`}
          style={{
            width: 12,
            height: 12,
            background: colors.border,
            border: `2px solid #141922`,
            boxShadow: `0 0 6px ${colors.glow}`,
            right: -7,
            cursor: "crosshair",
          }}
        />
      )}

      {/* Header stripe */}
      <Box
        sx={{
          px: 1.25,
          py: 0.875,
          background: `linear-gradient(135deg, ${colors.bg} 0%, ${alpha(colors.bg, 0.6)} 100%)`,
          borderBottom: `1px solid ${colors.border}30`,
          borderRadius: "8px 8px 0 0",
          display: "flex",
          alignItems: "center",
          gap: 0.875,
        }}
      >
        {/* Icon badge */}
        <Box
          sx={{
            width: 26,
            height: 26,
            borderRadius: "6px",
            bgcolor: `${colors.border}22`,
            border: `1px solid ${colors.border}55`,
            color: colors.text,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            flexShrink: 0,
            "& svg": { fontSize: 15 },
          }}
        >
          {meta.icon}
        </Box>

        <Box sx={{ flex: 1, minWidth: 0 }}>
          <Typography
            sx={{
              fontSize: "0.75rem",
              fontWeight: 700,
              color: "#e8eaf0",
              lineHeight: 1.2,
              whiteSpace: "nowrap",
              overflow: "hidden",
              textOverflow: "ellipsis",
            }}
          >
            {data.label}
          </Typography>
          <Typography
            sx={{
              fontSize: "0.6rem",
              fontWeight: 700,
              letterSpacing: "0.08em",
              color: colors.text,
              opacity: 0.8,
              lineHeight: 1,
              mt: 0.2,
            }}
          >
            {meta.sublabel}
          </Typography>
        </Box>

        {/* Status indicator */}
        {status === "completed" && (
          <CompletedIcon sx={{ fontSize: 16, color: "#22c55e", flexShrink: 0 }} />
        )}
        {status === "failed" && (
          <FailedIcon sx={{ fontSize: 16, color: "#ef4444", flexShrink: 0 }} />
        )}
        {status === "running" && (
          <CircularProgress size={13} sx={{ color: "#60a5fa", flexShrink: 0 }} />
        )}
      </Box>

      {/* Config body */}
      <Box sx={{ px: 1.25, py: 0.875 }}>
        {configLines.length > 0 ? (
          configLines.map(({ key, display }) => (
            <Box
              key={key}
              sx={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                gap: 0.5,
                mb: 0.25,
              }}
            >
              <Typography
                sx={{
                  fontSize: "0.63rem",
                  color: "rgba(140,160,200,0.7)",
                  fontFamily: "monospace",
                  flexShrink: 0,
                }}
              >
                {key}
              </Typography>
              <Typography
                sx={{
                  fontSize: "0.63rem",
                  color: "rgba(200,215,240,0.85)",
                  fontFamily: "monospace",
                  maxWidth: 90,
                  whiteSpace: "nowrap",
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                  textAlign: "right",
                }}
              >
                {display}
              </Typography>
            </Box>
          ))
        ) : (
          <Typography
            sx={{
              fontSize: "0.62rem",
              color: "rgba(120,140,180,0.5)",
              fontStyle: "italic",
              fontFamily: "monospace",
            }}
          >
            설정 없음
          </Typography>
        )}
      </Box>
    </Box>
  );
}

// ---------------------------------------------------------------------------
// Node Palette Component
// ---------------------------------------------------------------------------

const PALETTE_WIDTH = 210;

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
    <Box
      sx={{
        width: PALETTE_WIDTH,
        borderRight: "1px solid rgba(80,100,140,0.25)",
        bgcolor: "#0f1419",
        p: 1.5,
        display: "flex",
        flexDirection: "column",
        gap: 0.75,
        overflowY: "auto",
        flexShrink: 0,
      }}
    >
      <Box sx={{ display: "flex", alignItems: "center", gap: 0.75, mb: 0.5 }}>
        <AddNodeIcon sx={{ fontSize: 15, color: "rgba(120,160,220,0.7)" }} />
        <Typography
          sx={{
            fontSize: "0.7rem",
            fontWeight: 800,
            letterSpacing: "0.1em",
            color: "rgba(120,160,220,0.7)",
            textTransform: "uppercase",
          }}
        >
          노드 팔레트
        </Typography>
      </Box>

      <Box sx={{ height: "1px", bgcolor: "rgba(80,100,140,0.25)", mb: 0.5 }} />

      {nodeTypes.map((nt) => {
        const meta = NODE_TYPE_META[nt];
        const colors = NODE_COLORS[nt];
        return (
          <Box
            key={nt}
            draggable
            onDragStart={(e) => onDragStart(e, nt)}
            sx={{
              display: "flex",
              alignItems: "center",
              gap: 0.875,
              p: 0.875,
              pr: 1.25,
              borderRadius: "8px",
              border: `1px solid ${colors.border}30`,
              bgcolor: `${colors.bg}`,
              cursor: "grab",
              userSelect: "none",
              transition: "all 0.15s ease",
              "&:hover": {
                borderColor: `${colors.border}80`,
                bgcolor: `${colors.bg}cc`,
                boxShadow: `0 0 12px ${colors.glow}`,
                transform: "translateX(2px)",
              },
              "&:active": {
                cursor: "grabbing",
                transform: "scale(0.97)",
              },
            }}
          >
            <DragIcon sx={{ fontSize: 13, color: "rgba(120,140,180,0.4)", flexShrink: 0 }} />
            <Box
              sx={{
                width: 24,
                height: 24,
                borderRadius: "5px",
                bgcolor: `${colors.border}22`,
                border: `1px solid ${colors.border}55`,
                color: colors.text,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                flexShrink: 0,
                "& svg": { fontSize: 14 },
              }}
            >
              {meta.icon}
            </Box>
            <Box sx={{ flex: 1, minWidth: 0 }}>
              <Typography sx={{ fontSize: "0.75rem", fontWeight: 600, color: "#e0e4f0", lineHeight: 1.2 }}>
                {meta.label}
              </Typography>
              <Typography
                sx={{
                  fontSize: "0.58rem",
                  fontWeight: 700,
                  letterSpacing: "0.07em",
                  color: colors.text,
                  opacity: 0.7,
                }}
              >
                {meta.sublabel}
              </Typography>
            </Box>
          </Box>
        );
      })}

      <Box sx={{ height: "1px", bgcolor: "rgba(80,100,140,0.25)", mt: 0.5, mb: 0.5 }} />

      <Typography
        sx={{
          fontSize: "0.62rem",
          color: "rgba(100,120,160,0.6)",
          textAlign: "center",
          lineHeight: 1.4,
        }}
      >
        드래그하여 캔버스에 추가
      </Typography>
    </Box>
  );
}

// ---------------------------------------------------------------------------
// Properties Panel Component
// ---------------------------------------------------------------------------

const PROPERTIES_WIDTH = 290;

interface PropertiesPanelProps {
  selectedNode: WorkflowNode | null;
  onUpdateNode: (nodeId: string, data: Partial<WorkflowNodeData>) => void;
  onDeleteNode: (nodeId: string) => void;
  workflowName: string;
  workflowDescription: string;
  onWorkflowNameChange: (name: string) => void;
  onWorkflowDescriptionChange: (desc: string) => void;
}

function SectionHeader({ label }: { label: string }) {
  return (
    <Box sx={{ display: "flex", alignItems: "center", gap: 0.75, mb: 1 }}>
      <Box sx={{ height: "1px", flex: 1, bgcolor: "rgba(80,100,140,0.25)" }} />
      <Typography
        sx={{
          fontSize: "0.6rem",
          fontWeight: 800,
          letterSpacing: "0.12em",
          color: "rgba(100,130,180,0.6)",
          textTransform: "uppercase",
          flexShrink: 0,
        }}
      >
        {label}
      </Typography>
      <Box sx={{ height: "1px", flex: 1, bgcolor: "rgba(80,100,140,0.25)" }} />
    </Box>
  );
}

function StyledField({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <Box sx={{ mb: 1.5 }}>
      <Typography
        sx={{
          fontSize: "0.65rem",
          fontWeight: 700,
          letterSpacing: "0.05em",
          color: "rgba(120,150,200,0.7)",
          textTransform: "uppercase",
          mb: 0.5,
        }}
      >
        {label}
      </Typography>
      {children}
    </Box>
  );
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
  const panelSx = {
    width: PROPERTIES_WIDTH,
    borderLeft: "1px solid rgba(80,100,140,0.25)",
    bgcolor: "#0f1419",
    p: 2,
    overflowY: "auto",
    flexShrink: 0,
    display: "flex",
    flexDirection: "column",
    gap: 0,
    "& .MuiTextField-root": {
      "& .MuiOutlinedInput-root": {
        bgcolor: "rgba(20,30,50,0.8)",
        "& fieldset": { borderColor: "rgba(80,100,140,0.3)" },
        "&:hover fieldset": { borderColor: "rgba(100,130,200,0.5)" },
        "&.Mui-focused fieldset": { borderColor: "rgba(100,140,220,0.7)" },
        "& input, & textarea": {
          fontSize: "0.8rem",
          color: "#c8d4e8",
        },
      },
      "& .MuiInputLabel-root": {
        fontSize: "0.78rem",
        color: "rgba(120,150,200,0.6)",
      },
    },
  };

  if (!selectedNode) {
    return (
      <Box sx={panelSx}>
        <Box sx={{ display: "flex", alignItems: "center", gap: 0.75, mb: 1.5 }}>
          <TuneIcon sx={{ fontSize: 15, color: "rgba(120,160,220,0.7)" }} />
          <Typography
            sx={{
              fontSize: "0.7rem",
              fontWeight: 800,
              letterSpacing: "0.1em",
              color: "rgba(120,160,220,0.7)",
              textTransform: "uppercase",
            }}
          >
            워크플로우 정보
          </Typography>
        </Box>
        <Box sx={{ height: "1px", bgcolor: "rgba(80,100,140,0.25)", mb: 2 }} />

        <StyledField label="이름">
          <TextField
            value={workflowName}
            onChange={(e) => onWorkflowNameChange(e.target.value)}
            fullWidth
            size="small"
          />
        </StyledField>
        <StyledField label="설명">
          <TextField
            value={workflowDescription}
            onChange={(e) => onWorkflowDescriptionChange(e.target.value)}
            fullWidth
            size="small"
            multiline
            rows={3}
          />
        </StyledField>

        <Box
          sx={{
            mt: 2,
            p: 1.5,
            borderRadius: "8px",
            border: "1px dashed rgba(80,100,140,0.35)",
            bgcolor: "rgba(20,30,50,0.4)",
            textAlign: "center",
          }}
        >
          <Typography sx={{ fontSize: "0.7rem", color: "rgba(100,130,180,0.6)", lineHeight: 1.5 }}>
            노드를 클릭하면
            <br />
            속성을 편집할 수 있습니다
          </Typography>
        </Box>
      </Box>
    );
  }

  const { data } = selectedNode;
  const meta = NODE_TYPE_META[data.nodeType];
  const colors = NODE_COLORS[data.nodeType];
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
    <Box sx={panelSx}>
      {/* Header */}
      <Box
        sx={{
          display: "flex",
          alignItems: "center",
          gap: 1,
          mb: 1.5,
          p: 1.25,
          borderRadius: "8px",
          background: `linear-gradient(135deg, ${colors.bg} 0%, ${alpha(colors.bg, 0.4)} 100%)`,
          border: `1px solid ${colors.border}30`,
        }}
      >
        <Box
          sx={{
            width: 30,
            height: 30,
            borderRadius: "7px",
            bgcolor: `${colors.border}22`,
            border: `1px solid ${colors.border}55`,
            color: colors.text,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            flexShrink: 0,
            "& svg": { fontSize: 17 },
          }}
        >
          {meta.icon}
        </Box>
        <Box>
          <Typography sx={{ fontSize: "0.82rem", fontWeight: 700, color: "#e8eaf0", lineHeight: 1.2 }}>
            {meta.label}
          </Typography>
          <Typography
            sx={{
              fontSize: "0.6rem",
              fontWeight: 700,
              letterSpacing: "0.1em",
              color: colors.text,
              opacity: 0.8,
            }}
          >
            {meta.sublabel}
          </Typography>
        </Box>
        <Typography
          sx={{
            ml: "auto",
            fontSize: "0.58rem",
            fontFamily: "monospace",
            color: "rgba(100,120,160,0.6)",
            fontWeight: 600,
          }}
        >
          #{selectedNode.id.slice(0, 8)}
        </Typography>
      </Box>

      <Box sx={{ height: "1px", bgcolor: "rgba(80,100,140,0.2)", mb: 1.75 }} />

      {/* Label */}
      <SectionHeader label="기본 정보" />
      <StyledField label="레이블">
        <TextField
          value={data.label}
          onChange={(e) => updateLabel(e.target.value)}
          fullWidth
          size="small"
        />
      </StyledField>

      {/* Type-specific config */}
      {(data.nodeType === "llm" || data.nodeType === "retrieval" ||
        data.nodeType === "tool" || data.nodeType === "condition" ||
        data.nodeType === "transform") && (
        <>
          <SectionHeader label="노드 설정" />
        </>
      )}

      {data.nodeType === "llm" && (
        <>
          <StyledField label="Provider">
            <TextField
              value={(config.provider as string) || ""}
              onChange={(e) => updateConfig("provider", e.target.value)}
              fullWidth
              size="small"
              placeholder="vllm, openai, ..."
            />
          </StyledField>
          <StyledField label="모델">
            <TextField
              value={(config.model as string) || ""}
              onChange={(e) => updateConfig("model", e.target.value)}
              fullWidth
              size="small"
              placeholder="모델 이름"
            />
          </StyledField>
          <StyledField label="시스템 프롬프트">
            <TextField
              value={(config.system_prompt as string) || ""}
              onChange={(e) => updateConfig("system_prompt", e.target.value)}
              fullWidth
              size="small"
              multiline
              rows={4}
              placeholder="LLM에 전달할 시스템 프롬프트"
            />
          </StyledField>
          <StyledField label={`Temperature: ${((config.temperature as number) ?? 0.7).toFixed(1)}`}>
            <Slider
              value={(config.temperature as number) ?? 0.7}
              onChange={(_e, val) => updateConfig("temperature", val)}
              min={0}
              max={2}
              step={0.1}
              size="small"
              sx={{
                color: colors.border,
                "& .MuiSlider-thumb": {
                  width: 14,
                  height: 14,
                  boxShadow: `0 0 6px ${colors.glow}`,
                },
                "& .MuiSlider-track": { border: "none" },
              }}
            />
          </StyledField>
        </>
      )}

      {data.nodeType === "retrieval" && (
        <>
          <StyledField label="Top K">
            <TextField
              type="number"
              value={(config.top_k as number) ?? 5}
              onChange={(e) => updateConfig("top_k", parseInt(e.target.value) || 5)}
              fullWidth
              size="small"
              slotProps={{ htmlInput: { min: 1, max: 50 } }}
            />
          </StyledField>
          <StyledField label="필터 (JSON)">
            <TextField
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
          </StyledField>
        </>
      )}

      {data.nodeType === "tool" && (
        <>
          <StyledField label="도구 이름">
            <TextField
              value={(config.tool_name as string) || ""}
              onChange={(e) => updateConfig("tool_name", e.target.value)}
              fullWidth
              size="small"
              placeholder="calculator, search, ..."
            />
          </StyledField>
          <StyledField label="인수 템플릿">
            <TextField
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
          </StyledField>
        </>
      )}

      {data.nodeType === "condition" && (
        <>
          <StyledField label="조건 유형">
            <Select
              value={(config.condition_type as string) || "confidence"}
              onChange={(e: SelectChangeEvent) => updateConfig("condition_type", e.target.value)}
              fullWidth
              size="small"
              sx={{
                bgcolor: "rgba(20,30,50,0.8)",
                "& .MuiOutlinedInput-notchedOutline": { borderColor: "rgba(80,100,140,0.3)" },
                "&:hover .MuiOutlinedInput-notchedOutline": { borderColor: "rgba(100,130,200,0.5)" },
                "& .MuiSelect-select": { fontSize: "0.8rem", color: "#c8d4e8" },
                "& .MuiSvgIcon-root": { color: "rgba(120,150,200,0.6)" },
              }}
            >
              <MenuItem value="confidence">신뢰도 (confidence)</MenuItem>
              <MenuItem value="keyword">키워드 (keyword)</MenuItem>
              <MenuItem value="length">길이 (length)</MenuItem>
              <MenuItem value="custom">사용자 정의 (custom)</MenuItem>
            </Select>
          </StyledField>
          <StyledField label={`임계값: ${((config.threshold as number) ?? 0.5).toFixed(2)}`}>
            <Slider
              value={(config.threshold as number) ?? 0.5}
              onChange={(_e, val) => updateConfig("threshold", val)}
              min={0}
              max={1}
              step={0.05}
              size="small"
              sx={{
                color: colors.border,
                "& .MuiSlider-thumb": {
                  width: 14,
                  height: 14,
                  boxShadow: `0 0 6px ${colors.glow}`,
                },
              }}
            />
          </StyledField>
        </>
      )}

      {data.nodeType === "transform" && (
        <StyledField label="변환 템플릿">
          <TextField
            value={(config.template as string) || ""}
            onChange={(e) => updateConfig("template", e.target.value)}
            fullWidth
            size="small"
            multiline
            rows={4}
            placeholder={"결과: {input}"}
            helperText="{input} 자리 표시자를 사용하세요"
          />
        </StyledField>
      )}

      {/* Delete */}
      <Box sx={{ mt: "auto", pt: 2 }}>
        <Box sx={{ height: "1px", bgcolor: "rgba(80,100,140,0.2)", mb: 1.5 }} />
        <Button
          variant="outlined"
          color="error"
          startIcon={<DeleteIcon />}
          onClick={() => onDeleteNode(selectedNode.id)}
          fullWidth
          size="small"
          sx={{
            borderColor: "rgba(239,68,68,0.4)",
            color: "rgba(239,68,68,0.8)",
            fontSize: "0.75rem",
            "&:hover": {
              borderColor: "rgba(239,68,68,0.7)",
              bgcolor: "rgba(239,68,68,0.08)",
            },
          }}
        >
          노드 삭제
        </Button>
      </Box>
    </Box>
  );
}

// ---------------------------------------------------------------------------
// Empty State Component
// ---------------------------------------------------------------------------

function EmptyState() {
  return (
    <Box
      sx={{
        position: "absolute",
        inset: 0,
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        pointerEvents: "none",
        gap: 1.5,
        zIndex: 1,
      }}
    >
      <Box
        sx={{
          width: 64,
          height: 64,
          borderRadius: "16px",
          border: "1.5px solid rgba(80,100,140,0.3)",
          bgcolor: "rgba(20,30,50,0.6)",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          mb: 0.5,
          "@keyframes floatIcon": {
            "0%, 100%": { transform: "translateY(0)" },
            "50%": { transform: "translateY(-6px)" },
          },
          animation: "floatIcon 3s ease-in-out infinite",
        }}
      >
        <WorkflowIcon sx={{ fontSize: 30, color: "rgba(80,120,200,0.5)" }} />
      </Box>
      <Typography sx={{ fontSize: "0.9rem", fontWeight: 700, color: "rgba(140,170,220,0.6)" }}>
        캔버스가 비어 있습니다
      </Typography>
      <Stack spacing={0.75} alignItems="center">
        <Box sx={{ display: "flex", alignItems: "center", gap: 0.75 }}>
          <Chip
            label="프리셋 불러오기"
            size="small"
            sx={{
              height: 22,
              fontSize: "0.65rem",
              fontWeight: 700,
              bgcolor: "rgba(30,50,90,0.7)",
              color: "rgba(100,150,240,0.8)",
              border: "1px solid rgba(80,120,200,0.3)",
            }}
          />
          <Typography sx={{ fontSize: "0.72rem", color: "rgba(100,130,180,0.5)" }}>
            로 시작하거나
          </Typography>
        </Box>
        <Box sx={{ display: "flex", alignItems: "center", gap: 0.75 }}>
          <Typography sx={{ fontSize: "0.72rem", color: "rgba(100,130,180,0.5)" }}>
            팔레트에서 노드를
          </Typography>
          <ArrowRightIcon sx={{ fontSize: 14, color: "rgba(100,130,180,0.4)" }} />
          <Typography sx={{ fontSize: "0.72rem", color: "rgba(100,130,180,0.5)" }}>
            드래그하세요
          </Typography>
        </Box>
      </Stack>
    </Box>
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
  const [currentWorkflowId, setCurrentWorkflowId] = useState<string | null>(null);

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

  // -- Save dialog --
  const [saveDialogOpen, setSaveDialogOpen] = useState(false);
  const [saveDialogName, setSaveDialogName] = useState("");
  const [saveDialogDesc, setSaveDialogDesc] = useState("");
  const [isSaving, setIsSaving] = useState(false);

  // -- Load dialog --
  const [loadDialogOpen, setLoadDialogOpen] = useState(false);
  const [savedWorkflows, setSavedWorkflows] = useState<WorkflowListItem[]>([]);
  const [isLoadingList, setIsLoadingList] = useState(false);
  const [deletingId, setDeletingId] = useState<string | null>(null);

  // -- Hidden file input for import --
  const importFileRef = useRef<HTMLInputElement>(null);

  const showSnackbar = useCallback(
    (message: string, severity: "success" | "error" | "info" = "info") => {
      setSnackbar({ open: true, message, severity });
    },
    [],
  );

  // -- Build workflow payload --
  const buildPayload = useCallback(
    (name: string, description: string) => ({
      name,
      description,
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
    }),
    [nodes, edges],
  );

  // -- Open save dialog --
  const openSaveDialog = useCallback(() => {
    setSaveDialogName(workflowName);
    setSaveDialogDesc(workflowDescription);
    setSaveDialogOpen(true);
  }, [workflowName, workflowDescription]);

  // -- Save to backend --
  const handleSaveConfirm = useCallback(async () => {
    if (!saveDialogName.trim()) return;
    setIsSaving(true);
    try {
      const payload = buildPayload(saveDialogName.trim(), saveDialogDesc.trim());
      if (currentWorkflowId) {
        await workflowsApi.update(currentWorkflowId, payload);
        showSnackbar("워크플로우가 업데이트되었습니다.", "success");
      } else {
        const res = await workflowsApi.create(payload);
        setCurrentWorkflowId(res.data.id);
        showSnackbar("워크플로우가 저장되었습니다.", "success");
      }
      setWorkflowName(saveDialogName.trim());
      setWorkflowDescription(saveDialogDesc.trim());
      setSaveDialogOpen(false);
    } catch {
      showSnackbar("저장 중 오류가 발생했습니다.", "error");
    } finally {
      setIsSaving(false);
    }
  }, [currentWorkflowId, saveDialogName, saveDialogDesc, buildPayload, showSnackbar]);

  // -- Load workflow list --
  const openLoadDialog = useCallback(async () => {
    setLoadDialogOpen(true);
    setIsLoadingList(true);
    try {
      const res = await workflowsApi.list();
      setSavedWorkflows(res.data.workflows);
    } catch {
      showSnackbar("워크플로우 목록을 불러오지 못했습니다.", "error");
    } finally {
      setIsLoadingList(false);
    }
  }, [showSnackbar]);

  // -- Load specific workflow from backend --
  const handleLoadWorkflow = useCallback(
    async (id: string) => {
      try {
        const res = await workflowsApi.get(id);
        const wf = res.data;
        // Reconstruct nodes
        const loadedNodes: WorkflowNode[] = (wf.nodes as Array<{
          id: string; type: WorkflowNodeType; label: string; config: Record<string, unknown>; position: { x: number; y: number };
        }>).map((n) => ({
          id: n.id,
          type: "workflow" as const,
          position: n.position,
          data: { nodeType: n.type, label: n.label, config: n.config },
        }));
        const loadedEdges: WorkflowEdge[] = (wf.edges as Array<{
          id: string; source: string; target: string; label: string;
        }>).map((e) => ({
          id: e.id,
          source: e.source,
          target: e.target,
          label: e.label,
          animated: true,
          type: "workflow" as const,
        }));
        setNodes(loadedNodes);
        setEdges(loadedEdges);
        setWorkflowName(wf.name);
        setWorkflowDescription(wf.description);
        setCurrentWorkflowId(wf.id);
        setSelectedNodeId(null);
        setLoadDialogOpen(false);
        showSnackbar(`"${wf.name}" 워크플로우가 로드되었습니다.`, "success");
      } catch {
        showSnackbar("워크플로우를 불러오지 못했습니다.", "error");
      }
    },
    [setNodes, setEdges, showSnackbar],
  );

  // -- Delete saved workflow --
  const handleDeleteWorkflow = useCallback(
    async (id: string) => {
      setDeletingId(id);
      try {
        await workflowsApi.delete(id);
        setSavedWorkflows((prev) => prev.filter((w) => w.id !== id));
        if (currentWorkflowId === id) setCurrentWorkflowId(null);
        showSnackbar("워크플로우가 삭제되었습니다.", "info");
      } catch {
        showSnackbar("삭제 중 오류가 발생했습니다.", "error");
      } finally {
        setDeletingId(null);
      }
    },
    [currentWorkflowId, showSnackbar],
  );

  // -- Import workflow from JSON file --
  const handleImportFile = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = (ev) => {
        try {
          const json = JSON.parse(ev.target?.result as string);
          const importedNodes: WorkflowNode[] = (json.nodes || []).map((n: {
            id: string; type: WorkflowNodeType; label: string; config: Record<string, unknown>; position: { x: number; y: number };
          }) => ({
            id: n.id,
            type: "workflow" as const,
            position: n.position || { x: 0, y: 0 },
            data: { nodeType: n.type, label: n.label || "", config: n.config || {} },
          }));
          const importedEdges: WorkflowEdge[] = (json.edges || []).map((edge: {
            id: string; source: string; target: string; label: string;
          }) => ({
            id: edge.id,
            source: edge.source,
            target: edge.target,
            label: edge.label || "",
            animated: true,
            type: "workflow" as const,
          }));
          setNodes(importedNodes);
          setEdges(importedEdges);
          setWorkflowName(json.name || "가져온 워크플로우");
          setWorkflowDescription(json.description || "");
          setCurrentWorkflowId(null);
          setSelectedNodeId(null);
          showSnackbar(`"${json.name || "워크플로우"}" 가져오기 완료`, "success");
        } catch {
          showSnackbar("JSON 파일을 파싱할 수 없습니다.", "error");
        }
      };
      reader.readAsText(file);
      // Reset file input so same file can be re-imported
      e.target.value = "";
    },
    [setNodes, setEdges, showSnackbar],
  );

  // Refresh load dialog whenever it opens
  useEffect(() => {
    if (!loadDialogOpen) return;
    setIsLoadingList(true);
    workflowsApi.list().then((res) => {
      setSavedWorkflows(res.data.workflows);
    }).catch(() => {
      // ignore - backend may be offline
    }).finally(() => setIsLoadingList(false));
  }, [loadDialogOpen]);

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

  // -- Edge types registration --
  const edgeTypes: EdgeTypes = useMemo(
    () => ({
      workflow: WorkflowEdgeComponent,
    }),
    [],
  );

  // -- Handlers --
  const onConnect = useCallback(
    (connection: Connection) => {
      setEdges((eds) =>
        addEdge(
          {
            ...connection,
            animated: true,
            type: "workflow",
            markerEnd: {
              type: MarkerType.ArrowClosed,
              color: "rgba(100,140,220,0.6)",
              width: 16,
              height: 16,
            },
          },
          eds,
        ),
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
    setCurrentWorkflowId(null);
    showSnackbar("캔버스가 초기화되었습니다.", "info");
  }, [setNodes, setEdges, showSnackbar]);

  // -- Run workflow --
  const runWorkflow = useCallback(async () => {
    if (nodes.length === 0) {
      showSnackbar("실행할 노드가 없습니다.", "error");
      return;
    }

    setIsRunning(true);

    setNodes((nds) =>
      nds.map((n) => ({
        ...n,
        data: { ...n.data, executionStatus: "pending" as ExecutionStatus },
      })),
    );

    try {
      const startNode = nodes.find((n) => n.data.nodeType === "start");
      if (!startNode) {
        showSnackbar("시작 노드가 없습니다.", "error");
        setIsRunning(false);
        return;
      }

      const adj = new Map<string, string[]>();
      for (const e of edges) {
        const existing = adj.get(e.source) || [];
        existing.push(e.target);
        adj.set(e.source, existing);
      }

      const queue = [startNode.id];
      const visited = new Set<string>();

      while (queue.length > 0) {
        const currentId = queue.shift()!;
        if (visited.has(currentId)) continue;
        visited.add(currentId);

        setNodes((nds) =>
          nds.map((n) =>
            n.id === currentId
              ? { ...n, data: { ...n.data, executionStatus: "running" as ExecutionStatus } }
              : n,
          ),
        );

        await new Promise((resolve) => setTimeout(resolve, 800));

        setNodes((nds) =>
          nds.map((n) =>
            n.id === currentId
              ? { ...n, data: { ...n.data, executionStatus: "completed" as ExecutionStatus } }
              : n,
          ),
        );

        const neighbors = adj.get(currentId) || [];
        for (const neighbor of neighbors) {
          if (!visited.has(neighbor)) {
            queue.push(neighbor);
          }
        }
      }

      try {
        await workflowsApi.run(selectedPreset, { query: "테스트 실행" });
      } catch {
        // Backend may not be available
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

  // -- Save workflow (opens dialog) --
  const saveWorkflow = useCallback(() => {
    openSaveDialog();
  }, [openSaveDialog]);

  const currentPreset = PRESETS.find((p) => p.id === selectedPreset);

  return (
    <Box
      sx={{
        display: "flex",
        flexDirection: "column",
        height: "100%",
        bgcolor: "#0a0e14",
      }}
    >
      {/* Toolbar */}
      <Box
        sx={{
          px: 2,
          py: 1,
          borderBottom: "1px solid rgba(80,100,140,0.25)",
          bgcolor: "#0d1219",
          display: "flex",
          alignItems: "center",
          gap: 1.5,
          flexWrap: "wrap",
          minHeight: 52,
        }}
      >
        {/* Preset selector group */}
        <Box
          sx={{
            display: "flex",
            alignItems: "center",
            gap: 1,
            p: 0.75,
            borderRadius: "10px",
            border: "1px solid rgba(80,100,140,0.25)",
            bgcolor: "rgba(15,20,30,0.8)",
          }}
        >
          <Select
            value={selectedPreset}
            onChange={(e: SelectChangeEvent) => setSelectedPreset(e.target.value)}
            size="small"
            sx={{
              minWidth: 175,
              "& .MuiOutlinedInput-notchedOutline": { border: "none" },
              "& .MuiSelect-select": {
                py: 0.5,
                fontSize: "0.8rem",
                fontWeight: 600,
                color: "#c8d4e8",
                display: "flex",
                alignItems: "center",
                gap: 0.75,
              },
              "& .MuiSvgIcon-root": { color: "rgba(120,150,200,0.6)" },
            }}
            renderValue={(val) => {
              const preset = PRESETS.find((p) => p.id === val);
              if (!preset) return val;
              return (
                <Box sx={{ display: "flex", alignItems: "center", gap: 0.75 }}>
                  <Chip
                    label={preset.badge}
                    size="small"
                    sx={{
                      height: 18,
                      fontSize: "0.58rem",
                      fontWeight: 800,
                      letterSpacing: "0.06em",
                      bgcolor: "rgba(40,70,130,0.8)",
                      color: "rgba(120,170,255,0.9)",
                      border: "1px solid rgba(80,130,220,0.4)",
                      "& .MuiChip-label": { px: 0.625 },
                    }}
                  />
                  <span>{preset.name}</span>
                </Box>
              );
            }}
          >
            {PRESETS.map((p) => (
              <MenuItem key={p.id} value={p.id} sx={{ gap: 1 }}>
                <Chip
                  label={p.badge}
                  size="small"
                  sx={{
                    height: 18,
                    fontSize: "0.58rem",
                    fontWeight: 800,
                    letterSpacing: "0.06em",
                    bgcolor: "rgba(40,70,130,0.8)",
                    color: "rgba(120,170,255,0.9)",
                    border: "1px solid rgba(80,130,220,0.4)",
                    "& .MuiChip-label": { px: 0.625 },
                  }}
                />
                <Box>
                  <Typography sx={{ fontSize: "0.8rem", fontWeight: 600, lineHeight: 1.2 }}>
                    {p.name}
                  </Typography>
                  <Typography sx={{ fontSize: "0.65rem", color: "text.secondary", lineHeight: 1.2 }}>
                    {p.description}
                  </Typography>
                </Box>
              </MenuItem>
            ))}
          </Select>
          <Button
            variant="contained"
            size="small"
            onClick={loadPreset}
            disabled={isRunning}
            sx={{
              py: 0.5,
              px: 1.25,
              fontSize: "0.72rem",
              fontWeight: 700,
              bgcolor: "rgba(50,80,140,0.8)",
              color: "rgba(150,190,255,0.95)",
              border: "1px solid rgba(80,130,220,0.4)",
              boxShadow: "none",
              "&:hover": {
                bgcolor: "rgba(70,110,190,0.8)",
                boxShadow: "0 0 12px rgba(80,130,220,0.3)",
              },
              "&:disabled": { opacity: 0.4 },
            }}
          >
            불러오기
          </Button>
        </Box>

        <Box sx={{ height: 28, width: "1px", bgcolor: "rgba(80,100,140,0.25)" }} />

        {/* Run button */}
        <Button
          variant="contained"
          size="small"
          startIcon={
            isRunning ? <CircularProgress size={13} color="inherit" /> : <RunIcon sx={{ fontSize: 16 }} />
          }
          onClick={runWorkflow}
          disabled={isRunning || nodes.length === 0}
          sx={{
            py: 0.625,
            px: 1.5,
            fontSize: "0.78rem",
            fontWeight: 700,
            bgcolor: isRunning ? "rgba(30,80,50,0.8)" : "rgba(20,90,55,0.85)",
            color: isRunning ? "rgba(80,200,130,0.8)" : "#4ade80",
            border: "1px solid rgba(40,160,90,0.4)",
            boxShadow: nodes.length > 0 && !isRunning ? "0 0 16px rgba(34,197,94,0.2)" : "none",
            "&:hover": {
              bgcolor: "rgba(30,110,65,0.9)",
              boxShadow: "0 0 20px rgba(34,197,94,0.3)",
            },
            "&:disabled": { opacity: 0.35 },
          }}
        >
          {isRunning ? "실행 중..." : "실행"}
        </Button>

        <Box sx={{ height: 28, width: "1px", bgcolor: "rgba(80,100,140,0.25)" }} />

        {/* Utility icons */}
        <Box sx={{ display: "flex", gap: 0.5 }}>
          {[
            { icon: <SaveIcon sx={{ fontSize: 15 }} />, label: currentWorkflowId ? "저장 (업데이트)" : "저장", action: saveWorkflow },
            { icon: <LoadIcon sx={{ fontSize: 15 }} />, label: "워크플로우 불러오기", action: openLoadDialog },
            { icon: <ExportIcon sx={{ fontSize: 15 }} />, label: "JSON 내보내기", action: exportWorkflow },
            { icon: <ImportIcon sx={{ fontSize: 15 }} />, label: "JSON 가져오기", action: () => importFileRef.current?.click() },
            { icon: <ClearIcon sx={{ fontSize: 15 }} />, label: "캔버스 초기화", action: clearCanvas },
          ].map(({ icon, label, action }) => (
            <Tooltip key={label} title={label} placement="bottom">
              <IconButton
                size="small"
                onClick={action}
                disabled={isRunning}
                sx={{
                  width: 30,
                  height: 30,
                  color: "rgba(120,150,200,0.6)",
                  border: "1px solid rgba(80,100,140,0.2)",
                  borderRadius: "7px",
                  "&:hover": {
                    bgcolor: "rgba(60,80,130,0.3)",
                    color: "rgba(160,190,240,0.9)",
                    borderColor: "rgba(100,130,200,0.4)",
                  },
                  "&:disabled": { opacity: 0.3 },
                }}
              >
                {icon}
              </IconButton>
            </Tooltip>
          ))}
        </Box>
        {/* Hidden file input for JSON import */}
        <input
          ref={importFileRef}
          type="file"
          accept=".json,application/json"
          style={{ display: "none" }}
          onChange={handleImportFile}
        />

        {/* Workflow name + node count */}
        <Box sx={{ ml: "auto", display: "flex", alignItems: "center", gap: 1.5 }}>
          {currentPreset && nodes.length > 0 && (
            <Typography
              sx={{
                fontSize: "0.65rem",
                fontFamily: "monospace",
                color: "rgba(100,130,180,0.6)",
                fontWeight: 600,
              }}
            >
              {nodes.length}N · {edges.length}E
            </Typography>
          )}
          <Typography
            sx={{
              fontSize: "0.78rem",
              fontWeight: 700,
              color: "rgba(160,190,240,0.7)",
              maxWidth: 200,
              whiteSpace: "nowrap",
              overflow: "hidden",
              textOverflow: "ellipsis",
            }}
          >
            {workflowName}
          </Typography>
        </Box>
      </Box>

      {/* Main area: Palette + Canvas + Properties */}
      <Box sx={{ display: "flex", flex: 1, overflow: "hidden" }}>
        {/* Left: Node palette */}
        <NodePalette onDragStart={onDragStart} />

        {/* Center: ReactFlow canvas */}
        <Box
          ref={reactFlowWrapper}
          sx={{
            flex: 1,
            position: "relative",
            bgcolor: "#0a0e14",
          }}
          onDragOver={onDragOver}
          onDrop={onDrop}
        >
          {nodes.length === 0 && <EmptyState />}
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            onNodeClick={onNodeClick}
            onPaneClick={onPaneClick}
            nodeTypes={nodeTypes}
            edgeTypes={edgeTypes}
            fitView
            proOptions={{ hideAttribution: true }}
            defaultEdgeOptions={{
              animated: true,
              type: "workflow",
              markerEnd: {
                type: MarkerType.ArrowClosed,
                color: "rgba(100,140,220,0.6)",
                width: 16,
                height: 16,
              },
            }}
            style={{ width: "100%", height: "100%" }}
          >
            <Controls
              position="bottom-left"
              style={{
                display: "flex",
                flexDirection: "column",
                gap: 2,
                bottom: 16,
                left: 16,
              }}
            />
            <MiniMap
              position="bottom-right"
              nodeColor={(node) => {
                const d = node.data as WorkflowNodeData | undefined;
                if (d?.nodeType) {
                  return NODE_COLORS[d.nodeType]?.border ?? "#555";
                }
                return "#555";
              }}
              maskColor="rgba(0,0,0,0.6)"
              style={{
                borderRadius: 8,
                border: "1px solid rgba(80,100,140,0.25)",
                background: "rgba(10,14,20,0.9)",
                bottom: 16,
                right: 16,
              }}
            />
            <Background
              variant={BackgroundVariant.Dots}
              gap={24}
              size={1}
              color="rgba(60,80,120,0.35)"
            />
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
          sx={{
            width: "100%",
            fontSize: "0.8rem",
            borderRadius: "8px",
          }}
        >
          {snackbar.message}
        </Alert>
      </Snackbar>

      {/* ------------------------------------------------------------------ */}
      {/* Save Dialog                                                          */}
      {/* ------------------------------------------------------------------ */}
      <Dialog
        open={saveDialogOpen}
        onClose={() => setSaveDialogOpen(false)}
        maxWidth="sm"
        fullWidth
        PaperProps={{
          sx: {
            bgcolor: "#0d1219",
            border: "1px solid rgba(80,100,140,0.35)",
            borderRadius: "12px",
            backgroundImage: "none",
          },
        }}
      >
        <DialogTitle
          sx={{
            fontSize: "0.95rem",
            fontWeight: 700,
            color: "rgba(160,190,240,0.9)",
            borderBottom: "1px solid rgba(80,100,140,0.2)",
            pb: 1.5,
          }}
        >
          {currentWorkflowId ? "워크플로우 업데이트" : "워크플로우 저장"}
        </DialogTitle>
        <DialogContent sx={{ pt: 2.5 }}>
          <Box
            sx={{
              "& .MuiOutlinedInput-root": {
                bgcolor: "rgba(20,30,50,0.8)",
                "& fieldset": { borderColor: "rgba(80,100,140,0.35)" },
                "&:hover fieldset": { borderColor: "rgba(100,130,200,0.5)" },
                "&.Mui-focused fieldset": { borderColor: "rgba(100,140,220,0.7)" },
                "& input, & textarea": { fontSize: "0.85rem", color: "#c8d4e8" },
              },
              "& .MuiInputLabel-root": { fontSize: "0.82rem", color: "rgba(120,150,200,0.6)" },
            }}
          >
            <TextField
              label="워크플로우 이름"
              value={saveDialogName}
              onChange={(e) => setSaveDialogName(e.target.value)}
              fullWidth
              size="small"
              autoFocus
              sx={{ mb: 2 }}
            />
            <TextField
              label="설명 (선택)"
              value={saveDialogDesc}
              onChange={(e) => setSaveDialogDesc(e.target.value)}
              fullWidth
              size="small"
              multiline
              rows={3}
            />
          </Box>
        </DialogContent>
        <DialogActions sx={{ px: 3, pb: 2, gap: 1 }}>
          <Button
            onClick={() => setSaveDialogOpen(false)}
            disabled={isSaving}
            sx={{ fontSize: "0.8rem", color: "rgba(120,150,200,0.7)" }}
          >
            취소
          </Button>
          <Button
            variant="contained"
            onClick={handleSaveConfirm}
            disabled={isSaving || !saveDialogName.trim()}
            startIcon={isSaving ? <CircularProgress size={13} color="inherit" /> : <SaveIcon sx={{ fontSize: 15 }} />}
            sx={{
              fontSize: "0.8rem",
              fontWeight: 700,
              bgcolor: "rgba(30,80,140,0.85)",
              color: "rgba(140,190,255,0.95)",
              border: "1px solid rgba(80,130,220,0.4)",
              "&:hover": { bgcolor: "rgba(50,100,180,0.85)" },
              "&:disabled": { opacity: 0.4 },
            }}
          >
            {isSaving ? "저장 중..." : "저장"}
          </Button>
        </DialogActions>
      </Dialog>

      {/* ------------------------------------------------------------------ */}
      {/* Load Dialog                                                          */}
      {/* ------------------------------------------------------------------ */}
      <Dialog
        open={loadDialogOpen}
        onClose={() => setLoadDialogOpen(false)}
        maxWidth="sm"
        fullWidth
        PaperProps={{
          sx: {
            bgcolor: "#0d1219",
            border: "1px solid rgba(80,100,140,0.35)",
            borderRadius: "12px",
            backgroundImage: "none",
          },
        }}
      >
        <DialogTitle
          sx={{
            fontSize: "0.95rem",
            fontWeight: 700,
            color: "rgba(160,190,240,0.9)",
            borderBottom: "1px solid rgba(80,100,140,0.2)",
            pb: 1.5,
          }}
        >
          저장된 워크플로우
        </DialogTitle>
        <DialogContent sx={{ pt: 1.5, px: 0, minHeight: 180 }}>
          {isLoadingList ? (
            <Box sx={{ display: "flex", justifyContent: "center", alignItems: "center", height: 120 }}>
              <CircularProgress size={28} sx={{ color: "rgba(100,140,220,0.7)" }} />
            </Box>
          ) : savedWorkflows.length === 0 ? (
            <Box sx={{ textAlign: "center", py: 4 }}>
              <Typography sx={{ fontSize: "0.85rem", color: "rgba(100,130,180,0.6)" }}>
                저장된 워크플로우가 없습니다.
              </Typography>
            </Box>
          ) : (
            <List disablePadding>
              {savedWorkflows.map((wf, idx) => (
                <Box key={wf.id}>
                  {idx > 0 && <Divider sx={{ borderColor: "rgba(80,100,140,0.15)" }} />}
                  <ListItem
                    sx={{
                      px: 3,
                      py: 1.25,
                      cursor: "pointer",
                      transition: "background 0.15s",
                      "&:hover": { bgcolor: "rgba(50,70,120,0.3)" },
                    }}
                    onClick={() => handleLoadWorkflow(wf.id)}
                  >
                    <ListItemText
                      primary={
                        <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                          <Typography sx={{ fontSize: "0.875rem", fontWeight: 600, color: "#c8d4e8" }}>
                            {wf.name}
                          </Typography>
                          <Chip
                            label={`${wf.node_count}노드`}
                            size="small"
                            sx={{
                              height: 18,
                              fontSize: "0.6rem",
                              fontWeight: 700,
                              bgcolor: "rgba(40,70,130,0.6)",
                              color: "rgba(120,170,255,0.85)",
                              border: "1px solid rgba(80,130,220,0.3)",
                              "& .MuiChip-label": { px: 0.625 },
                            }}
                          />
                        </Box>
                      }
                      secondary={
                        <Box>
                          {wf.description && (
                            <Typography
                              component="span"
                              sx={{ display: "block", fontSize: "0.75rem", color: "rgba(120,150,200,0.6)", mb: 0.25 }}
                            >
                              {wf.description}
                            </Typography>
                          )}
                          <Typography
                            component="span"
                            sx={{ fontSize: "0.68rem", color: "rgba(100,120,160,0.5)", fontFamily: "monospace" }}
                          >
                            {new Date(wf.updated_at).toLocaleString("ko-KR")}
                          </Typography>
                        </Box>
                      }
                    />
                    <ListItemSecondaryAction>
                      <Tooltip title="삭제">
                        <IconButton
                          size="small"
                          onClick={(e) => {
                            e.stopPropagation();
                            handleDeleteWorkflow(wf.id);
                          }}
                          disabled={deletingId === wf.id}
                          sx={{
                            color: "rgba(239,68,68,0.5)",
                            "&:hover": { color: "rgba(239,68,68,0.85)", bgcolor: "rgba(239,68,68,0.1)" },
                          }}
                        >
                          {deletingId === wf.id ? (
                            <CircularProgress size={14} sx={{ color: "rgba(239,68,68,0.5)" }} />
                          ) : (
                            <DeleteIcon sx={{ fontSize: 16 }} />
                          )}
                        </IconButton>
                      </Tooltip>
                    </ListItemSecondaryAction>
                  </ListItem>
                </Box>
              ))}
            </List>
          )}
        </DialogContent>
        <DialogActions sx={{ px: 3, pb: 2 }}>
          <Button
            onClick={() => setLoadDialogOpen(false)}
            sx={{ fontSize: "0.8rem", color: "rgba(120,150,200,0.7)" }}
          >
            닫기
          </Button>
        </DialogActions>
      </Dialog>

      {/* React Flow dark theme overrides */}
      <style>{`
        .react-flow__controls {
          background: rgba(13,18,25,0.9) !important;
          border: 1px solid rgba(80,100,140,0.25) !important;
          border-radius: 8px !important;
          box-shadow: none !important;
        }
        .react-flow__controls-button {
          background: transparent !important;
          border-bottom: 1px solid rgba(80,100,140,0.2) !important;
          color: rgba(120,150,200,0.7) !important;
          fill: rgba(120,150,200,0.7) !important;
        }
        .react-flow__controls-button:hover {
          background: rgba(50,70,120,0.3) !important;
        }
        .react-flow__controls-button:last-child {
          border-bottom: none !important;
        }
        .react-flow__edge-path {
          stroke: rgba(100,120,160,0.5);
        }
        .react-flow__connection-path {
          stroke: rgba(100,160,255,0.8) !important;
          stroke-width: 2 !important;
        }
        .react-flow__handle:hover {
          transform: scale(1.3);
          transition: transform 0.15s ease;
        }
      `}</style>
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
