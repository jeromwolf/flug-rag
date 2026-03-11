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
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import Layout from "../components/Layout";
import { workflowsApi, mcpApi } from "../api/client";
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
  { icon: React.ReactNode; label: string; sublabel: string; color: string; description: string }
> = {
  start:     { icon: <StartIcon />,     label: "시작",      sublabel: "TRIGGER",   color: "#22c55e", description: "워크플로우의 시작점. 사용자 입력을 받아 다음 노드로 전달합니다." },
  llm:       { icon: <LlmIcon />,       label: "LLM 생성",   sublabel: "LLM",       color: "#3b82f6", description: "AI 모델로 텍스트를 생성합니다.\n설정: Provider, 모델명, 시스템 프롬프트, Temperature" },
  retrieval: { icon: <RetrievalIcon />, label: "문서 검색",  sublabel: "RETRIEVER", color: "#f97316", description: "벡터DB에서 관련 문서를 검색합니다.\n설정: Top K (검색 수), 필터 조건 (JSON)" },
  tool:      { icon: <ToolIcon />,      label: "도구 실행",  sublabel: "TOOL",      color: "#a855f7", description: "MCP 도구를 실행합니다. (ERP, 이메일 등)\n설정: 도구 선택, 인수 템플릿 (JSON)" },
  condition: { icon: <ConditionIcon />, label: "조건 분기",  sublabel: "CONDITION", color: "#eab308", description: "조건에 따라 흐름을 분기합니다.\n설정: 조건 유형 (신뢰도/키워드/길이), 임계값" },
  output:    { icon: <OutputIcon />,    label: "출력",       sublabel: "OUTPUT",    color: "#06b6d4", description: "최종 결과를 사용자에게 출력합니다.\n설정: 출력 형식 (text/markdown/json)" },
  transform: { icon: <TransformIcon />, label: "변환",       sublabel: "TRANSFORM", color: "#6366f1", description: "데이터를 가공·변환합니다.\n설정: 변환 유형 (template/extract/merge), 템플릿" },
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
  // 1. 기술문서 검색·요약
  {
    id: "tech_doc_summary",
    name: "기술문서 검색·요약",
    description: "기술문서를 검색하고 신뢰도를 검증한 뒤 핵심 내용을 구조화된 요약 보고서로 생성",
    badge: "SEARCH",
    nodes: [
      {
        id: "start",
        type: "workflow",
        position: { x: 40, y: 220 },
        data: { nodeType: "start", label: "검색어 입력", config: { description: "요약할 기술문서 주제를 입력하세요 (예: 배관 용접 검사 기준)" } },
      },
      {
        id: "retrieve",
        type: "workflow",
        position: { x: 260, y: 220 },
        data: { nodeType: "retrieval", label: "기술문서 검색", config: { top_k: 7, description: "관련 기술문서 검색 (top_k=7)" } },
      },
      {
        id: "confidence_check",
        type: "workflow",
        position: { x: 480, y: 220 },
        data: {
          nodeType: "condition",
          label: "검색 품질 확인",
          config: { condition_type: "confidence", threshold: 0.3 },
        },
      },
      {
        id: "summarize",
        type: "workflow",
        position: { x: 700, y: 120 },
        data: {
          nodeType: "llm",
          label: "요약 보고서 생성",
          config: {
            system_prompt: "당신은 한국가스기술공사의 기술문서 분석 전문가입니다.\n검색된 기술문서를 바탕으로 다음 구조로 요약하세요:\n\n## 핵심 요약\n- 3줄 이내 핵심 내용\n\n## 주요 사항\n- 번호 매겨서 핵심 포인트 나열\n\n## 관련 규정\n- 적용되는 내부규정 및 조항 명시\n\n## 참고 사항\n- 추가 확인이 필요한 부분",
            temperature: 0.1,
          },
        },
      },
      {
        id: "fallback",
        type: "workflow",
        position: { x: 700, y: 340 },
        data: {
          nodeType: "transform",
          label: "검색 결과 부족 안내",
          config: { template: "검색된 문서의 신뢰도가 낮습니다.\n\n검색 결과:\n{input}\n\n더 구체적인 키워드로 다시 검색하거나, 관련 문서가 등록되어 있는지 확인해 주세요." },
        },
      },
      {
        id: "output",
        type: "workflow",
        position: { x: 940, y: 120 },
        data: { nodeType: "output", label: "요약 보고서", config: { format: "markdown" } },
      },
      {
        id: "output_low",
        type: "workflow",
        position: { x: 940, y: 340 },
        data: { nodeType: "output", label: "안내 출력", config: { format: "markdown" } },
      },
    ],
    edges: [
      { id: "e1", source: "start", target: "retrieve", animated: true, type: "workflow" },
      { id: "e2", source: "retrieve", target: "confidence_check", animated: true, type: "workflow" },
      { id: "e3", source: "confidence_check", target: "summarize", animated: true, label: "신뢰도 충분", type: "workflow" },
      { id: "e4", source: "confidence_check", target: "fallback", animated: true, label: "신뢰도 부족", type: "workflow" },
      { id: "e5", source: "summarize", target: "output", animated: true, type: "workflow" },
      { id: "e6", source: "fallback", target: "output_low", animated: true, type: "workflow" },
    ],
  },
  // 2. 규정 비교 분석
  {
    id: "regulation_comparison",
    name: "규정 비교 분석",
    description: "두 규정 문서를 병렬 검색하고 공통점·차이점·충돌 사항을 분석한 뒤 컴플라이언스 리뷰 수행",
    badge: "COMPARE",
    nodes: [
      {
        id: "start",
        type: "workflow",
        position: { x: 40, y: 220 },
        data: { nodeType: "start", label: "비교 요청 입력", config: { description: "비교할 두 규정 항목을 입력하세요 (예: 안전관리규정 vs 시설관리규정)" } },
      },
      {
        id: "retrieve_a",
        type: "workflow",
        position: { x: 280, y: 100 },
        data: {
          nodeType: "retrieval",
          label: "규정 A 검색",
          config: { top_k: 5, description: "첫 번째 규정 검색" },
        },
      },
      {
        id: "retrieve_b",
        type: "workflow",
        position: { x: 280, y: 340 },
        data: {
          nodeType: "retrieval",
          label: "규정 B 검색",
          config: { top_k: 5, description: "두 번째 규정 검색" },
        },
      },
      {
        id: "compare_llm",
        type: "workflow",
        position: { x: 540, y: 220 },
        data: {
          nodeType: "llm",
          label: "비교 분석",
          config: {
            system_prompt: "당신은 한국가스기술공사 규정 분석 전문가입니다.\n두 규정을 다음 형식으로 비교 분석하세요:\n\n## 규정 개요\n| 항목 | 규정 A | 규정 B |\n\n## 공통점\n- 두 규정이 동일하게 규정하는 사항\n\n## 차이점\n| 비교 항목 | 규정 A | 규정 B | 비고 |\n\n## 충돌 사항\n- 두 규정 간 상충되는 내용 (있을 경우)\n\n## 적용 시 유의사항\n- 실무에서 두 규정을 동시 적용 시 주의점",
            temperature: 0.1,
          },
        },
      },
      {
        id: "output",
        type: "workflow",
        position: { x: 780, y: 220 },
        data: { nodeType: "output", label: "비교 분석 보고서", config: { format: "markdown" } },
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
  // 3. 설비 점검이력 조회 및 리포팅
  {
    id: "equipment_inspection",
    name: "설비 점검이력 조회 및 리포팅",
    description: "EHSQ 시스템에서 설비 점검이력을 조회하고 안전 규정과 대조 분석하여 점검 리포트 자동 생성",
    badge: "SAFETY",
    nodes: [
      {
        id: "start",
        type: "workflow",
        position: { x: 40, y: 220 },
        data: { nodeType: "start", label: "설비 정보 입력", config: { description: "조회할 설비명 또는 시설을 입력하세요 (예: 인천 LNG 기지 배관)" } },
      },
      {
        id: "ehsq_lookup",
        type: "workflow",
        position: { x: 270, y: 100 },
        data: {
          nodeType: "tool",
          label: "EHSQ 점검이력 조회",
          config: {
            tool_name: "ehsq_lookup",
            arguments_template: { action: "incident_report", facility: "{input}" },
          },
        },
      },
      {
        id: "retrieve_safety",
        type: "workflow",
        position: { x: 270, y: 340 },
        data: {
          nodeType: "retrieval",
          label: "안전 규정 검색",
          config: { top_k: 5, description: "관련 안전 규정 검색" },
        },
      },
      {
        id: "analyze_llm",
        type: "workflow",
        position: { x: 520, y: 220 },
        data: {
          nodeType: "llm",
          label: "점검이력 분석",
          config: {
            system_prompt: "당신은 한국가스기술공사의 설비 안전 분석 전문가입니다.\nEHSQ 점검이력과 관련 안전 규정을 종합하여 분석하세요:\n\n## 점검 현황 요약\n- 총 점검 횟수, 최근 점검일, 점검 주기 준수 여부\n\n## 주요 결함 유형\n- 발견된 결함을 유형별로 분류하고 빈도 표시\n\n## 재발 패턴 분석\n- 동일 결함의 반복 발생 여부 및 원인 추정\n\n## 규정 대조 결과\n- 관련 안전 규정 준수 여부\n\n## 권고 조치사항\n- 즉시 조치, 단기(1개월), 중기(분기) 구분",
            temperature: 0.1,
          },
        },
      },
      {
        id: "report_tool",
        type: "workflow",
        position: { x: 770, y: 220 },
        data: {
          nodeType: "tool",
          label: "점검보고서 생성",
          config: {
            tool_name: "report_draft",
            arguments_template: { topic: "{input}", report_type: "점검보고서" },
          },
        },
      },
      {
        id: "output",
        type: "workflow",
        position: { x: 1010, y: 220 },
        data: { nodeType: "output", label: "점검 리포트", config: { format: "markdown" } },
      },
    ],
    edges: [
      { id: "e1", source: "start", target: "ehsq_lookup", animated: true, label: "EHSQ", type: "workflow" },
      { id: "e2", source: "start", target: "retrieve_safety", animated: true, label: "규정", type: "workflow" },
      { id: "e3", source: "ehsq_lookup", target: "analyze_llm", animated: true, type: "workflow" },
      { id: "e4", source: "retrieve_safety", target: "analyze_llm", animated: true, type: "workflow" },
      { id: "e5", source: "analyze_llm", target: "report_tool", animated: true, type: "workflow" },
      { id: "e6", source: "report_tool", target: "output", animated: true, type: "workflow" },
    ],
  },
  // 4. 경영평가 자료 자동 수집
  {
    id: "management_evaluation",
    name: "경영평가 자료 자동 수집",
    description: "ERP 시스템의 예산·실적 데이터와 내부규정을 결합하여 경영평가 자료를 자동으로 수집·분석",
    badge: "EVAL",
    nodes: [
      {
        id: "start",
        type: "workflow",
        position: { x: 40, y: 220 },
        data: { nodeType: "start", label: "평가 항목 입력", config: { description: "수집할 경영평가 항목을 입력하세요 (예: 2026년 상반기 안전관리 실적)" } },
      },
      {
        id: "erp_budget",
        type: "workflow",
        position: { x: 280, y: 100 },
        data: {
          nodeType: "tool",
          label: "ERP 예산·실적 조회",
          config: {
            tool_name: "erp_lookup",
            arguments_template: { query_type: "budget", keyword: "{input}" },
          },
        },
      },
      {
        id: "retrieve_regulation",
        type: "workflow",
        position: { x: 280, y: 340 },
        data: {
          nodeType: "retrieval",
          label: "평가 기준 규정 검색",
          config: { top_k: 5, description: "경영평가 기준 및 관련 규정 검색" },
        },
      },
      {
        id: "analyze_llm",
        type: "workflow",
        position: { x: 540, y: 220 },
        data: {
          nodeType: "llm",
          label: "종합 분석",
          config: {
            system_prompt: "당신은 한국가스기술공사의 경영평가 분석 전문가입니다.\nERP에서 수집한 경영 데이터와 내부규정을 종합하여 분석하세요:\n\n## 평가 항목 개요\n- 해당 경영평가 항목의 배경과 목적\n\n## 정량 실적\n| 지표 | 목표 | 실적 | 달성률 |\n- ERP 데이터 기반 수치 근거 명시\n\n## 정성 실적\n- 주요 추진 사항 및 성과\n\n## 규정 근거\n- 각 실적의 내부규정 근거 조항\n\n## 개선 과제\n- 미달 항목에 대한 개선 방안",
            temperature: 0.1,
          },
        },
      },
      {
        id: "report_tool",
        type: "workflow",
        position: { x: 780, y: 220 },
        data: {
          nodeType: "tool",
          label: "현황보고서 생성",
          config: {
            tool_name: "report_draft",
            arguments_template: { topic: "{input}", report_type: "현황보고서" },
          },
        },
      },
      {
        id: "output",
        type: "workflow",
        position: { x: 1020, y: 220 },
        data: { nodeType: "output", label: "경영평가 자료", config: { format: "markdown" } },
      },
    ],
    edges: [
      { id: "e1", source: "start", target: "erp_budget", animated: true, label: "ERP", type: "workflow" },
      { id: "e2", source: "start", target: "retrieve_regulation", animated: true, label: "규정", type: "workflow" },
      { id: "e3", source: "erp_budget", target: "analyze_llm", animated: true, type: "workflow" },
      { id: "e4", source: "retrieve_regulation", target: "analyze_llm", animated: true, type: "workflow" },
      { id: "e5", source: "analyze_llm", target: "report_tool", animated: true, type: "workflow" },
      { id: "e6", source: "report_tool", target: "output", animated: true, type: "workflow" },
    ],
  },
  // 5. 신규직원 온보딩 가이드 생성
  {
    id: "onboarding_guide",
    name: "신규직원 온보딩 가이드 생성",
    description: "내부규정을 검색하여 핵심 내용을 정리하고 신규직원 맞춤형 온보딩 가이드를 자동 생성",
    badge: "ONBOARD",
    nodes: [
      {
        id: "start",
        type: "workflow",
        position: { x: 40, y: 220 },
        data: { nodeType: "start", label: "부서·직무 입력", config: { description: "신규직원의 부서 및 직무를 입력하세요 (예: 시설안전부 배관검사)" } },
      },
      {
        id: "retrieve",
        type: "workflow",
        position: { x: 260, y: 220 },
        data: {
          nodeType: "retrieval",
          label: "내부규정 검색",
          config: { top_k: 7, description: "온보딩 관련 내부규정 및 절차 검색" },
        },
      },
      {
        id: "organize_llm",
        type: "workflow",
        position: { x: 480, y: 220 },
        data: {
          nodeType: "llm",
          label: "핵심 규정 정리",
          config: {
            system_prompt: "당신은 한국가스기술공사의 인사교육 담당자입니다.\n검색된 내부규정 중 신규직원이 반드시 알아야 할 핵심 내용을 정리하세요:\n\n1. 조직 및 직무 관련 규정\n2. 안전 관련 필수 규정\n3. 복무 및 근태 규정\n4. 보안 및 정보보호 규정\n\n각 항목에 해당 규정의 조항 번호를 명시하세요.",
            temperature: 0.2,
          },
        },
      },
      {
        id: "training_tool",
        type: "workflow",
        position: { x: 720, y: 220 },
        data: {
          nodeType: "tool",
          label: "온보딩 교육자료 생성",
          config: {
            tool_name: "training_material",
            arguments_template: { topic: "{input}", level: "신입", format: "종합" },
          },
        },
      },
      {
        id: "output",
        type: "workflow",
        position: { x: 960, y: 220 },
        data: { nodeType: "output", label: "온보딩 가이드", config: { format: "markdown" } },
      },
    ],
    edges: [
      { id: "e1", source: "start", target: "retrieve", animated: true, type: "workflow" },
      { id: "e2", source: "retrieve", target: "organize_llm", animated: true, type: "workflow" },
      { id: "e3", source: "organize_llm", target: "training_tool", animated: true, type: "workflow" },
      { id: "e4", source: "training_tool", target: "output", animated: true, type: "workflow" },
    ],
  },
  // 6. ISO 인증 대응 문서 초안
  {
    id: "iso_compliance",
    name: "ISO 인증 대응 문서 초안",
    description: "EHSQ 준수 현황과 ISO 관련 내부규정을 결합하여 Gap 분석 및 대응 문서 초안을 자동 생성",
    badge: "ISO",
    nodes: [
      {
        id: "start",
        type: "workflow",
        position: { x: 40, y: 220 },
        data: { nodeType: "start", label: "ISO 항목 입력", config: { description: "대응이 필요한 ISO 인증 항목을 입력하세요 (예: ISO 14001 환경경영 4.4 운영 계획)" } },
      },
      {
        id: "ehsq_compliance",
        type: "workflow",
        position: { x: 280, y: 100 },
        data: {
          nodeType: "tool",
          label: "준수 현황 조회",
          config: {
            tool_name: "ehsq_lookup",
            arguments_template: { action: "compliance_check", facility: "{input}" },
          },
        },
      },
      {
        id: "retrieve_iso",
        type: "workflow",
        position: { x: 280, y: 340 },
        data: {
          nodeType: "retrieval",
          label: "ISO 관련 규정 검색",
          config: { top_k: 5, description: "ISO 인증 관련 내부규정 검색" },
        },
      },
      {
        id: "gap_analysis",
        type: "workflow",
        position: { x: 540, y: 220 },
        data: {
          nodeType: "llm",
          label: "Gap 분석",
          config: {
            system_prompt: "당신은 한국가스기술공사의 ISO 인증 심사 대응 전문가입니다.\nEHSQ 준수 현황과 내부규정을 대조하여 Gap 분석을 수행하세요:\n\n## ISO 요건 요약\n- 해당 ISO 조항의 핵심 요구사항\n\n## 현재 준수 현황\n| 요구사항 | 현황 | 적합/부적합 |\n\n## Gap 분석\n- 미충족 항목과 원인 분석\n\n## 개선 계획\n| 개선 항목 | 담당부서 | 목표일 | 조치 내용 |\n\n## 필요 증적 자료 목록\n- 심사 시 제출해야 할 증빙 문서",
            temperature: 0.1,
          },
        },
      },
      {
        id: "report_tool",
        type: "workflow",
        position: { x: 800, y: 220 },
        data: {
          nodeType: "tool",
          label: "공식 대응 문서 생성",
          config: {
            tool_name: "report_draft",
            arguments_template: { topic: "{input}", report_type: "분석보고서" },
          },
        },
      },
      {
        id: "output",
        type: "workflow",
        position: { x: 1040, y: 220 },
        data: { nodeType: "output", label: "ISO 대응 문서", config: { format: "markdown" } },
      },
    ],
    edges: [
      { id: "e1", source: "start", target: "ehsq_compliance", animated: true, label: "EHSQ", type: "workflow" },
      { id: "e2", source: "start", target: "retrieve_iso", animated: true, label: "규정", type: "workflow" },
      { id: "e3", source: "ehsq_compliance", target: "gap_analysis", animated: true, type: "workflow" },
      { id: "e4", source: "retrieve_iso", target: "gap_analysis", animated: true, type: "workflow" },
      { id: "e5", source: "gap_analysis", target: "report_tool", animated: true, type: "workflow" },
      { id: "e6", source: "report_tool", target: "output", animated: true, type: "workflow" },
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
          <Tooltip
            key={nt}
            title={
              <Box sx={{ p: 0.5 }}>
                <Typography sx={{ fontSize: "0.75rem", fontWeight: 700, mb: 0.5, color: "#fff" }}>
                  {meta.label} ({meta.sublabel})
                </Typography>
                {meta.description.split("\n").map((line, i) => (
                  <Typography key={i} sx={{ fontSize: "0.68rem", color: "rgba(255,255,255,0.85)", lineHeight: 1.5 }}>
                    {line}
                  </Typography>
                ))}
              </Box>
            }
            placement="right"
            arrow
            enterDelay={400}
            slotProps={{
              tooltip: {
                sx: {
                  bgcolor: "rgba(15,20,30,0.95)",
                  border: "1px solid rgba(80,120,200,0.3)",
                  borderRadius: "8px",
                  boxShadow: "0 4px 20px rgba(0,0,0,0.5)",
                  maxWidth: 260,
                  p: 1,
                },
              },
              arrow: {
                sx: {
                  color: "rgba(15,20,30,0.95)",
                },
              },
            }}
          >
            <Box
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
          </Tooltip>
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
  // Fetch MCP tools list for tool node dropdown
  const [mcpTools, setMcpTools] = useState<{ name: string; description: string }[]>([]);
  useEffect(() => {
    mcpApi.listTools().then((res) => {
      const tools = (res.data as { tools?: { name: string; description: string }[] })?.tools ?? [];
      setMcpTools(tools);
    }).catch(() => {
      // Backend may be offline — use hardcoded fallback
      setMcpTools([
        { name: "search_documents", description: "문서 검색" },
        { name: "regulation_review", description: "규정 검토" },
        { name: "safety_checklist", description: "안전 체크리스트" },
        { name: "erp_lookup", description: "ERP 조회" },
        { name: "summarizer", description: "요약 생성" },
        { name: "translator", description: "번역" },
        { name: "report_draft", description: "보고서 초안" },
        { name: "calculator", description: "계산기" },
      ]);
    });
  }, []);

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
          <StyledField label="도구 선택">
            <Select
              value={(config.tool_name as string) || ""}
              onChange={(e: SelectChangeEvent) => updateConfig("tool_name", e.target.value)}
              fullWidth
              size="small"
              displayEmpty
              sx={{
                bgcolor: "rgba(20,30,50,0.8)",
                "& .MuiOutlinedInput-notchedOutline": { borderColor: "rgba(80,100,140,0.3)" },
                "&:hover .MuiOutlinedInput-notchedOutline": { borderColor: "rgba(100,130,200,0.5)" },
                "& .MuiSelect-select": { fontSize: "0.8rem", color: "#c8d4e8" },
                "& .MuiSvgIcon-root": { color: "rgba(120,150,200,0.6)" },
              }}
              MenuProps={{
                PaperProps: {
                  sx: {
                    bgcolor: "#0d1219",
                    border: "1px solid rgba(80,100,140,0.35)",
                    "& .MuiMenuItem-root": {
                      fontSize: "0.8rem",
                      color: "#c8d4e8",
                      "&:hover": { bgcolor: "rgba(50,70,120,0.3)" },
                      "&.Mui-selected": { bgcolor: "rgba(50,70,120,0.5)" },
                    },
                  },
                },
              }}
            >
              <MenuItem value="" disabled>
                <Typography sx={{ fontSize: "0.8rem", color: "rgba(100,130,180,0.5)" }}>
                  도구를 선택하세요
                </Typography>
              </MenuItem>
              {mcpTools.map((tool) => (
                <MenuItem key={tool.name} value={tool.name}>
                  <Box sx={{ display: "flex", flexDirection: "column" }}>
                    <Typography sx={{ fontSize: "0.8rem", fontWeight: 600 }}>
                      {tool.name}
                    </Typography>
                    <Typography sx={{ fontSize: "0.65rem", color: "rgba(120,150,200,0.6)" }}>
                      {tool.description}
                    </Typography>
                  </Box>
                </MenuItem>
              ))}
            </Select>
          </StyledField>
          <StyledField label="인수 템플릿 (JSON)">
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
              placeholder='{"arg1": "{{input}}"}'
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

  // -- Run dialog (query input + result) --
  const [runDialogOpen, setRunDialogOpen] = useState(false);
  const [runQuery, setRunQuery] = useState("");
  const [resultDialogOpen, setResultDialogOpen] = useState(false);
  const [runResult, setRunResult] = useState<{
    final_output: string;
    status: string;
    total_duration_ms: number;
    node_results: { node_id: string; status: string; duration_ms: number }[];
    error: string | null;
  } | null>(null);

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

  // -- Run workflow: open query input dialog --
  const runWorkflow = useCallback(() => {
    if (nodes.length === 0) {
      showSnackbar("실행할 노드가 없습니다.", "error");
      return;
    }
    const startNode = nodes.find((n) => n.data.nodeType === "start");
    if (!startNode) {
      showSnackbar("시작 노드가 없습니다.", "error");
      return;
    }
    setRunQuery("");
    setRunDialogOpen(true);
  }, [nodes, showSnackbar]);

  // -- Execute workflow after query input --
  const handleRunConfirm = useCallback(async () => {
    if (!runQuery.trim()) return;
    setRunDialogOpen(false);
    setIsRunning(true);
    setRunResult(null);

    // Reset all nodes to pending
    setNodes((nds) =>
      nds.map((n) => ({
        ...n,
        data: { ...n.data, executionStatus: "pending" as ExecutionStatus },
      })),
    );

    // Build adjacency for animation
    const adj = new Map<string, string[]>();
    for (const e of edges) {
      const existing = adj.get(e.source) || [];
      existing.push(e.target);
      adj.set(e.source, existing);
    }

    // Animate nodes in order (BFS)
    const startNode = nodes.find((n) => n.data.nodeType === "start");
    if (startNode) {
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
        await new Promise((resolve) => setTimeout(resolve, 600));
        const neighbors = adj.get(currentId) || [];
        for (const neighbor of neighbors) {
          if (!visited.has(neighbor)) queue.push(neighbor);
        }
      }
    }

    // Call backend
    try {
      let res;
      if (currentWorkflowId) {
        res = await workflowsApi.run({ query: runQuery.trim() }, { workflowId: currentWorkflowId });
      } else {
        // Send full inline workflow data instead of preset ID
        res = await workflowsApi.runInline({ query: runQuery.trim() }, nodes, edges);
      }
      const data = res.data;

      // Update node statuses from backend result
      const nodeStatusMap = new Map<string, string>();
      for (const nr of data.node_results || []) {
        nodeStatusMap.set(nr.node_id, nr.status);
      }
      setNodes((nds) =>
        nds.map((n) => ({
          ...n,
          data: {
            ...n.data,
            executionStatus: (nodeStatusMap.get(n.id) || (data.status === "completed" ? "completed" : "failed")) as ExecutionStatus,
          },
        })),
      );

      setRunResult(data);
      setResultDialogOpen(true);
      showSnackbar(
        data.status === "completed"
          ? `실행 완료 (${(data.total_duration_ms / 1000).toFixed(1)}초)`
          : "워크플로우 실행 중 오류가 발생했습니다.",
        data.status === "completed" ? "success" : "error",
      );
    } catch {
      // Mark all as completed anyway (animation only)
      setNodes((nds) =>
        nds.map((n) => ({
          ...n,
          data: { ...n.data, executionStatus: "completed" as ExecutionStatus },
        })),
      );
      showSnackbar("백엔드 연결 실패 — 애니메이션만 실행되었습니다.", "info");
    } finally {
      setIsRunning(false);
    }
  }, [runQuery, nodes, edges, selectedPreset, currentWorkflowId, setNodes, showSnackbar]);

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

      {/* ------------------------------------------------------------------ */}
      {/* Run Query Input Dialog                                             */}
      {/* ------------------------------------------------------------------ */}
      <Dialog
        open={runDialogOpen}
        onClose={() => setRunDialogOpen(false)}
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
          워크플로우 실행
        </DialogTitle>
        <DialogContent sx={{ pt: 2.5 }}>
          <Typography sx={{ fontSize: "0.8rem", color: "rgba(120,150,200,0.7)", mb: 2 }}>
            워크플로우에 전달할 입력 쿼리를 입력하세요.
          </Typography>
          <TextField
            value={runQuery}
            onChange={(e) => setRunQuery(e.target.value)}
            placeholder="예: 가스시설 안전점검 절차를 알려주세요"
            fullWidth
            multiline
            rows={3}
            autoFocus
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey && runQuery.trim()) {
                e.preventDefault();
                handleRunConfirm();
              }
            }}
            sx={{
              "& .MuiOutlinedInput-root": {
                bgcolor: "rgba(20,30,50,0.8)",
                "& fieldset": { borderColor: "rgba(80,100,140,0.35)" },
                "&:hover fieldset": { borderColor: "rgba(100,130,200,0.5)" },
                "&.Mui-focused fieldset": { borderColor: "rgba(34,197,94,0.7)" },
                "& textarea": { fontSize: "0.85rem", color: "#c8d4e8" },
              },
            }}
          />
        </DialogContent>
        <DialogActions sx={{ px: 3, pb: 2, gap: 1 }}>
          <Button
            onClick={() => setRunDialogOpen(false)}
            sx={{ fontSize: "0.8rem", color: "rgba(120,150,200,0.7)" }}
          >
            취소
          </Button>
          <Button
            variant="contained"
            onClick={handleRunConfirm}
            disabled={!runQuery.trim()}
            startIcon={<RunIcon sx={{ fontSize: 15 }} />}
            sx={{
              fontSize: "0.8rem",
              fontWeight: 700,
              bgcolor: "rgba(20,90,55,0.85)",
              color: "#4ade80",
              border: "1px solid rgba(40,160,90,0.4)",
              "&:hover": { bgcolor: "rgba(30,110,65,0.9)" },
              "&:disabled": { opacity: 0.4 },
            }}
          >
            실행
          </Button>
        </DialogActions>
      </Dialog>

      {/* ------------------------------------------------------------------ */}
      {/* Execution Result Dialog                                            */}
      {/* ------------------------------------------------------------------ */}
      <Dialog
        open={resultDialogOpen}
        onClose={() => setResultDialogOpen(false)}
        maxWidth="md"
        fullWidth
        PaperProps={{
          sx: {
            bgcolor: "#0d1219",
            border: "1px solid rgba(80,100,140,0.35)",
            borderRadius: "12px",
            backgroundImage: "none",
            maxHeight: "80vh",
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
            display: "flex",
            alignItems: "center",
            gap: 1,
          }}
        >
          {runResult?.status === "completed" ? (
            <CompletedIcon sx={{ fontSize: 18, color: "#22c55e" }} />
          ) : (
            <FailedIcon sx={{ fontSize: 18, color: "#ef4444" }} />
          )}
          실행 결과
          {runResult && (
            <Chip
              label={`${(runResult.total_duration_ms / 1000).toFixed(1)}초`}
              size="small"
              sx={{
                ml: "auto",
                height: 20,
                fontSize: "0.68rem",
                fontWeight: 700,
                bgcolor: "rgba(40,70,130,0.6)",
                color: "rgba(120,170,255,0.85)",
                border: "1px solid rgba(80,130,220,0.3)",
                "& .MuiChip-label": { px: 0.75 },
              }}
            />
          )}
        </DialogTitle>
        <DialogContent sx={{ pt: 2 }}>
          {runResult?.error && (
            <Alert
              severity="error"
              sx={{
                mb: 2,
                bgcolor: "rgba(239,68,68,0.1)",
                color: "#fca5a5",
                border: "1px solid rgba(239,68,68,0.3)",
                "& .MuiAlert-icon": { color: "#ef4444" },
              }}
            >
              {runResult.error}
            </Alert>
          )}

          {/* Node execution timeline */}
          {runResult?.node_results && runResult.node_results.length > 0 && (
            <Box sx={{ mb: 2 }}>
              <Typography sx={{ fontSize: "0.78rem", fontWeight: 600, color: "rgba(120,150,200,0.7)", mb: 1 }}>
                노드 실행 타임라인
              </Typography>
              <Box sx={{ display: "flex", gap: 0.75, flexWrap: "wrap" }}>
                {runResult.node_results.map((nr) => (
                  <Chip
                    key={nr.node_id}
                    icon={
                      nr.status === "completed" ? (
                        <CompletedIcon sx={{ fontSize: 13, color: "#22c55e !important" }} />
                      ) : (
                        <FailedIcon sx={{ fontSize: 13, color: "#ef4444 !important" }} />
                      )
                    }
                    label={`${nr.node_id} (${nr.duration_ms}ms)`}
                    size="small"
                    sx={{
                      height: 22,
                      fontSize: "0.68rem",
                      bgcolor: nr.status === "completed" ? "rgba(34,197,94,0.1)" : "rgba(239,68,68,0.1)",
                      color: nr.status === "completed" ? "rgba(120,220,160,0.9)" : "rgba(255,160,160,0.9)",
                      border: `1px solid ${nr.status === "completed" ? "rgba(34,197,94,0.3)" : "rgba(239,68,68,0.3)"}`,
                      "& .MuiChip-label": { px: 0.5 },
                    }}
                  />
                ))}
              </Box>
            </Box>
          )}

          {/* Query display */}
          {runQuery && (
            <Box sx={{ mb: 2, p: 1.5, bgcolor: "rgba(34,197,94,0.08)", border: "1px solid rgba(34,197,94,0.2)", borderRadius: "8px" }}>
              <Typography sx={{ fontSize: "0.72rem", fontWeight: 600, color: "rgba(34,197,94,0.7)", mb: 0.5 }}>
                입력 질문
              </Typography>
              <Typography sx={{ fontSize: "0.85rem", color: "#c8d4e8" }}>
                {runQuery}
              </Typography>
            </Box>
          )}

          {/* Final output */}
          <Typography sx={{ fontSize: "0.78rem", fontWeight: 600, color: "rgba(120,150,200,0.7)", mb: 1 }}>
            실행 출력
          </Typography>
          <Box
            sx={{
              bgcolor: "rgba(15,22,35,0.9)",
              border: "1px solid rgba(80,100,140,0.25)",
              borderRadius: "8px",
              p: 2,
              maxHeight: 400,
              overflow: "auto",
              wordBreak: "break-word",
              fontSize: "0.85rem",
              lineHeight: 1.8,
              color: "#c8d4e8",
              fontFamily: "'Noto Sans KR', sans-serif",
              "& strong": { color: "#93c5fd" },
            }}
          >
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              components={{
                h1: ({ children }) => <Typography variant="h6" sx={{ color: "#93c5fd", mt: 2, mb: 1 }}>{children}</Typography>,
                h2: ({ children }) => <Typography variant="subtitle1" sx={{ color: "#93c5fd", fontWeight: 700, mt: 2, mb: 1 }}>{children}</Typography>,
                h3: ({ children }) => <Typography variant="subtitle2" sx={{ color: "#93c5fd", fontWeight: 700, mt: 1.5, mb: 0.5 }}>{children}</Typography>,
                p: ({ children }) => <Typography variant="body2" sx={{ color: "#c8d4e8", mb: 1, lineHeight: 1.8 }}>{children}</Typography>,
                strong: ({ children }) => <strong style={{ color: "#93c5fd" }}>{children}</strong>,
                ul: ({ children }) => <Box component="ul" sx={{ pl: 2, mb: 1, "& li": { color: "#c8d4e8", mb: 0.3 } }}>{children}</Box>,
                ol: ({ children }) => <Box component="ol" sx={{ pl: 2, mb: 1, "& li": { color: "#c8d4e8", mb: 0.3 } }}>{children}</Box>,
                code: ({ children }) => <Box component="code" sx={{ bgcolor: "rgba(50,70,120,0.3)", px: 0.5, borderRadius: 0.5, fontSize: "0.8rem" }}>{children}</Box>,
                pre: ({ children }) => <Box component="pre" sx={{ bgcolor: "rgba(20,30,50,0.6)", p: 1.5, borderRadius: 1, overflow: "auto", fontSize: "0.8rem", mb: 1 }}>{children}</Box>,
                table: ({ children }) => <Box sx={{ overflow: "auto", bgcolor: "rgba(20,30,50,0.4)", borderRadius: 1, mb: 1 }}><Box component="table" sx={{ width: "100%", borderCollapse: "collapse", "& th, & td": { p: 1, borderBottom: "1px solid rgba(80,100,140,0.3)", color: "#c8d4e8", fontSize: "0.8rem" }, "& th": { color: "#93c5fd", fontWeight: 700 } }}>{children}</Box></Box>,
              }}
            >
              {typeof runResult?.final_output === "string"
                ? runResult.final_output
                : runResult?.final_output
                  ? JSON.stringify(runResult.final_output, null, 2)
                  : "출력 없음"}
            </ReactMarkdown>
          </Box>
        </DialogContent>
        <DialogActions sx={{ px: 3, pb: 2 }}>
          <Button
            onClick={() => setResultDialogOpen(false)}
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
