// Core types for flux-rag

export interface Message {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  sources?: Source[];
  confidenceScore?: number;
  responseMode?: "rag" | "direct" | "hybrid";
  modelUsed?: string;
  latencyMs?: number;
  createdAt: string;
}

export interface Source {
  chunkId: string;
  filename: string;
  page?: number;
  content: string;
  score: number;
}

export interface Session {
  id: string;
  sessionId: string;
  title: string;
  createdAt: string;
  updatedAt: string;
  messageCount: number;
}

export interface Document {
  id: string;
  filename: string;
  fileType: string;
  fileSize: number;
  uploadDate: string;
  department?: string;
  category?: string;
  status: "pending" | "processing" | "completed" | "failed";
  chunkCount: number;
  isPublic: boolean;
  ocrApplied: boolean;
}

export interface Model {
  id: string;
  name: string;
  provider: "vllm" | "ollama" | "openai" | "anthropic";
  modelId: string;
  config: {
    temperature: number;
    maxTokens: number;
    [key: string]: unknown;
  };
  isActive: boolean;
}

export interface Feedback {
  id: string;
  messageId: string;
  userId: string;
  rating: "up" | "down";
  correctionText?: string;
  reason?: string;
  isReviewed: boolean;
  createdAt: string;
}

export interface ChatRequest {
  message: string;
  sessionId?: string;
  mode: "rag" | "direct" | "hybrid";
  model?: string;
  filters?: {
    department?: string;
    dateFrom?: string;
    dateTo?: string;
  };
}

export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  message?: string;
  error?: {
    code: string;
    message: string;
    details?: unknown;
  };
}

// --- Auth ---
export type UserRole = "admin" | "manager" | "user" | "viewer";

export interface AuthUser {
  id: string;
  username: string;
  email: string;
  full_name: string;
  department: string;
  role: UserRole;
  is_active: boolean;
}

export interface AuthTokens {
  access_token: string;
  refresh_token: string;
  token_type: string;
  expires_at: string;
}

export type ConfidenceLevel = "high" | "medium" | "low";

export function getConfidenceLevel(score: number): ConfidenceLevel {
  if (score >= 0.8) return "high";
  if (score >= 0.5) return "medium";
  return "low";
}
