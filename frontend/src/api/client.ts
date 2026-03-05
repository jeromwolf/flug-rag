import axios from "axios";

const API_BASE = import.meta.env.VITE_API_URL || "/api";

const api = axios.create({
  baseURL: API_BASE,
  headers: {
    "Content-Type": "application/json",
  },
});

// ---------------------------------------------------------------------------
// Auth interceptors
// ---------------------------------------------------------------------------

const ACCESS_TOKEN_KEY = "app_access_token";
const REFRESH_TOKEN_KEY = "app_refresh_token";

// Request interceptor: attach Bearer token
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem(ACCESS_TOKEN_KEY);
    if (token && config.headers) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error),
);

// Response interceptor: handle 401 by attempting token refresh once
let isRefreshing = false;
let failedQueue: Array<{
  resolve: (token: string) => void;
  reject: (error: unknown) => void;
}> = [];

function processQueue(error: unknown, token: string | null) {
  for (const p of failedQueue) {
    if (error) {
      p.reject(error);
    } else if (token) {
      p.resolve(token);
    }
  }
  failedQueue = [];
}

api.interceptors.response.use(
  (response) => response,
  async (error) => {
    const originalRequest = error.config;

    // Skip auth endpoints to avoid infinite loops
    if (
      originalRequest?.url?.includes("/auth/login") ||
      originalRequest?.url?.includes("/auth/refresh")
    ) {
      return Promise.reject(error);
    }

    if (error.response?.status === 401 && !originalRequest._retry) {
      if (isRefreshing) {
        return new Promise<string>((resolve, reject) => {
          failedQueue.push({ resolve, reject });
        }).then((token) => {
          originalRequest.headers.Authorization = `Bearer ${token}`;
          return api(originalRequest);
        });
      }

      originalRequest._retry = true;
      isRefreshing = true;

      const refreshToken = localStorage.getItem(REFRESH_TOKEN_KEY);
      if (!refreshToken) {
        isRefreshing = false;
        // No refresh token -- redirect to login
        window.location.href = "/login";
        return Promise.reject(error);
      }

      try {
        const res = await axios.post(`${API_BASE}/auth/refresh`, {
          refresh_token: refreshToken,
        });
        const newAccess = res.data.access_token;
        const newRefresh = res.data.refresh_token;
        localStorage.setItem(ACCESS_TOKEN_KEY, newAccess);
        localStorage.setItem(REFRESH_TOKEN_KEY, newRefresh);

        processQueue(null, newAccess);
        originalRequest.headers.Authorization = `Bearer ${newAccess}`;
        return api(originalRequest);
      } catch (refreshError) {
        processQueue(refreshError, null);
        localStorage.removeItem(ACCESS_TOKEN_KEY);
        localStorage.removeItem(REFRESH_TOKEN_KEY);
        localStorage.removeItem("app_user");
        window.location.href = "/login";
        return Promise.reject(refreshError);
      } finally {
        isRefreshing = false;
      }
    }

    return Promise.reject(error);
  },
);

// ---------------------------------------------------------------------------
// Auth API
// ---------------------------------------------------------------------------

export const authApi = {
  login: (username: string, password: string) =>
    api.post("/auth/login", { username, password }),
  refresh: (refreshToken: string) =>
    api.post("/auth/refresh", { refresh_token: refreshToken }),
  logout: () => api.post("/auth/logout"),
  me: () => api.get("/auth/me"),
  listUsers: () => api.get("/auth/users"),
  updateUserRole: (userId: string, role: string) =>
    api.put(`/auth/users/${userId}/role`, { role }),
  getAuditLogs: (limit = 100) =>
    api.get("/auth/audit", { params: { limit } }),
  getEthicsPledge: () => api.get("/auth/ethics-pledge"),
  getEthicsPledgeStatus: () => api.get("/auth/ethics-pledge/status"),
  agreeEthicsPledge: () => api.post("/auth/ethics-pledge/agree"),
  createAccessRequest: (data: { requested_role: string; reason: string }) =>
    api.post("/auth/access-request", data),
  getMyAccessRequests: () => api.get("/auth/access-request/my"),
  getAdminAccessRequests: () => api.get("/auth/admin/access-requests"),
  reviewAccessRequest: (id: string, data: { decision: string; comment?: string }) =>
    api.put(`/auth/admin/access-requests/${id}`, data),
  createUser: (data: { username: string; password: string; email?: string; full_name?: string; department?: string; role?: string }) =>
    api.post("/auth/users", data),
  toggleUserActive: (userId: string, isActive: boolean) =>
    api.patch(`/auth/users/${userId}/active`, { is_active: isActive }),
  deleteUser: (userId: string) => api.delete(`/auth/users/${userId}`),
  verifyPassword: (password: string) =>
    api.post("/auth/verify-password", { password }),
  changePassword: (data: { current_password: string; new_password: string }) =>
    api.post("/auth/change-password", data),
};

// === Chat ===
export const chatApi = {
  send: (data: {
    message: string;
    session_id?: string;
    mode?: string;
    provider?: string;
    model?: string;
    filters?: Record<string, unknown>;
  }) => api.post("/chat", data),

  getStreamUrl: () => `${API_BASE}/chat/stream`,
};

// === Sessions ===
export const sessionsApi = {
  create: (title?: string) => api.post("/sessions", { title: title || "" }),
  list: (limit = 50, offset = 0) =>
    api.get("/sessions", { params: { limit, offset } }),
  getMessages: (sessionId: string, limit = 50) =>
    api.get(`/sessions/${sessionId}/messages`, { params: { limit } }),
  delete: (sessionId: string) => api.delete(`/sessions/${sessionId}`),
  update: (sessionId: string, title: string) =>
    api.patch(`/sessions/${sessionId}`, null, { params: { title } }),
};

// === Documents ===
export const documentsApi = {
  upload: (file: File) => {
    const formData = new FormData();
    formData.append("file", file);
    return api.post("/documents/upload", formData, {
      headers: { "Content-Type": "multipart/form-data" },
    });
  },
  list: () => api.get("/documents"),
  delete: (id: string) => api.delete(`/documents/${id}`),
  download: (id: string) => {
    window.open(`${API_BASE}/documents/${id}/download`, "_blank");
  },
};

// === Storage Management ===
export const storageApi = {
  getStats: () => api.get("/admin/storage-stats"),
};

// === Admin ===
export const adminApi = {
  getInfo: () => api.get("/admin/info"),
  getProviders: () => api.get("/admin/providers"),
  getPrompts: () => api.get("/admin/prompts"),
  updatePrompt: (name: string, content: string) =>
    api.put("/admin/prompts", { name, content }),
  // Models
  listModels: () => api.get("/admin/models"),
  createModel: (data: { name: string; provider: string; model_id: string; description?: string; is_default?: boolean }) =>
    api.post("/admin/models", data),
  updateModel: (id: string, data: Record<string, unknown>) =>
    api.put(`/admin/models/${id}`, data),
  deleteModel: (id: string) => api.delete(`/admin/models/${id}`),
  testModel: (id: string) => api.post(`/admin/models/${id}/test`),
  // Prompt versioning
  getPromptVersions: (name: string) => api.get(`/admin/prompts/versions/${name}`),
  savePromptVersion: (data: { name: string; content: string; description?: string }) =>
    api.post("/admin/prompts/versions", data),
  rollbackPrompt: (name: string, version: number) =>
    api.post(`/admin/prompts/rollback/${name}/${version}`),
  getSystemMetrics: () => api.get("/admin/system-metrics"),
  // Chunking config
  getChunkingConfig: () => api.get("/admin/chunking-config"),
  updateChunkingConfig: (data: { chunk_strategy?: string; chunk_size?: number; chunk_overlap?: number }) =>
    api.put("/admin/chunking-config", data),
  // Log level
  getLogLevel: () => api.get("/admin/log-level"),
  setLogLevel: (level: string) => api.put("/admin/log-level", null, { params: { level } }),
  // Cache management
  getCacheConfig: () => api.get("/admin/cache-config"),
  updateCacheConfig: (data: Partial<{ rag_query: number; llm_response: number; embeddings: number; documents: number }>) =>
    api.put("/admin/cache-config", data),
  clearCache: (category?: string) =>
    api.post("/admin/cache-clear", { category: category ?? null }),
  // Settings
  setDefaultProvider: (provider: string) =>
    api.put("/admin/settings", { default_provider: provider }),
  // LLM Playground
  playground: (data: {
    prompt: string;
    model?: string;
    temperature?: number;
    max_tokens?: number;
    system_prompt?: string;
  }) => api.post<{ response: string; latency_ms: number; model_used: string; tokens_used?: number }>("/admin/playground", data),
};

// === Feedback ===
export const feedbackApi = {
  submit: (data: {
    message_id: string;
    session_id: string;
    rating: number;
    comment?: string;
    corrected_answer?: string;
    query?: string;
    answer?: string;
  }) => api.post("/feedback", data),
  submitErrorReport: (data: {
    message_id: string;
    session_id: string;
    error_type?: string;
    description?: string;
  }) => api.post("/feedback/error-report", data),
  list: (limit = 50) => api.get("/feedback", { params: { limit } }),
  stats: () => api.get("/feedback/stats"),
  analytics: () => api.get("/feedback/analytics"),
};

// === MCP ===
export const mcpApi = {
  listTools: () => api.get("/mcp/tools"),
  callTool: (toolName: string, args: Record<string, unknown>) =>
    api.post("/mcp/call", { tool_name: toolName, arguments: args }),
  listCustomTools: () => api.get("/mcp/tools/custom"),
  createCustomTool: (data: {
    name: string;
    description: string;
    parameters_schema: string;
    execution_type: string;
    api_url?: string;
    api_method?: string;
    api_headers?: string;
    api_body_template?: string;
  }) => api.post("/mcp/tools/custom", data),
  updateCustomTool: (id: string, data: Record<string, unknown>) =>
    api.put(`/mcp/tools/custom/${id}`, data),
  deleteCustomTool: (id: string) => api.delete(`/mcp/tools/custom/${id}`),
  testCustomTool: (id: string, testInput: Record<string, unknown>) =>
    api.post(`/mcp/tools/custom/${id}/test`, testInput),
};

// === Workflows ===
export const workflowsApi = {
  // Presets (built-in)
  listPresets: () => api.get("/workflows/presets"),
  run: (preset: string, inputData: Record<string, unknown>) =>
    api.post("/workflows/run", { preset, input_data: inputData }),

  // User-saved workflows CRUD
  list: () => api.get<{ workflows: WorkflowListItem[] }>("/workflows"),
  create: (data: WorkflowSavePayload) =>
    api.post<WorkflowRecord>("/workflows", data),
  get: (id: string) => api.get<WorkflowRecord>(`/workflows/${id}`),
  update: (id: string, data: WorkflowSavePayload) =>
    api.put<WorkflowRecord>(`/workflows/${id}`, data),
  delete: (id: string) => api.delete(`/workflows/${id}`),
};

export interface WorkflowSavePayload {
  name: string;
  description?: string;
  nodes: unknown[];
  edges: unknown[];
}

export interface WorkflowListItem {
  id: string;
  name: string;
  description: string;
  created_at: string;
  updated_at: string;
  created_by: string | null;
  node_count: number;
}

export interface WorkflowRecord extends WorkflowListItem {
  nodes: unknown[];
  edges: unknown[];
}

// === Quality (RAG 파이프라인 품질 관리) ===
export const qualityApi = {
  // 문서 처리 현황
  getDocumentStatus: () => api.get("/quality/documents/status"),
  getDocumentChanges: () => api.get("/quality/documents/changes"),
  // 청크 품질
  getChunkMetrics: () => api.get("/quality/chunks/metrics"),
  getChunkPreview: (documentId: string) =>
    api.get(`/quality/chunks/preview/${documentId}`),
  getChunksByDocument: () => api.get("/quality/chunks/by-document"),
  // 임베딩 상태
  getEmbeddingStatus: () => api.get("/quality/embeddings/status"),
  getEmbeddingHistory: (limit = 50) =>
    api.get("/quality/embeddings/history", { params: { limit } }),
  getEmbeddingFailed: () => api.get("/quality/embeddings/failed"),
  // 벡터 분포
  getVectorDistribution: () => api.get("/quality/vectors/distribution"),
  getVectorHealth: () => api.get("/quality/vectors/health"),
  // 재처리 큐
  getReprocessQueue: (status?: string) =>
    api.get("/quality/reprocess/queue", { params: status ? { status } : {} }),
  getReprocessStats: () => api.get("/quality/reprocess/stats"),
  retryItem: (queueId: string) =>
    api.post(`/quality/reprocess/retry/${queueId}`),
  retryAllFailed: () => api.post("/quality/reprocess/retry-all"),
  deleteQueueItem: (queueId: string) =>
    api.delete(`/quality/reprocess/${queueId}`),
  // Golden Data (전문가 평가)
  listGoldenData: (limit = 50) =>
    api.get("/quality/golden-data", { params: { limit } }),
  createGoldenData: (data: {
    question: string;
    answer: string;
    source_message_id?: string;
    source_session_id?: string;
    category?: string;
    evaluation_tag?: string;
  }) => api.post("/quality/golden-data", data),
  updateGoldenData: (id: string, data: Record<string, unknown>) =>
    api.put(`/quality/golden-data/${id}`, data),
};

// === Golden Data ===
export const goldenDataApi = {
  list: () => api.get("/quality/golden-data"),
  create: (data: any) => api.post("/quality/golden-data", data),
  update: (id: string, data: any) => api.put(`/quality/golden-data/${id}`, data),
  delete: (id: string) => api.delete(`/quality/golden-data/${id}`),
};

// === Statistics ===
export const statsApi = {
  getUsage: (period: "day" | "week" | "month" = "week") =>
    api.get("/stats/usage", { params: { period } }),
  getKeywords: (period: "day" | "week" | "month" = "week", topN = 30) =>
    api.get("/stats/keywords", { params: { period, top_n: topN } }),
  getUserUsage: (period: "day" | "week" | "month" = "week") =>
    api.get("/stats/users", { params: { period } }),
  exportExcel: (period: "day" | "week" | "month" = "month") =>
    api.get("/stats/export", { params: { period }, responseType: "blob" }),
};

// === Logs ===
export const logsApi = {
  searchAccess: (params: {
    username?: string;
    action?: string;
    ip_address?: string;
    start_date?: string;
    end_date?: string;
    page?: number;
    page_size?: number;
  }) => api.get("/logs/access", { params }),
  searchQueries: (params: {
    keyword?: string;
    session_id?: string;
    start_date?: string;
    end_date?: string;
    page?: number;
    page_size?: number;
  }) => api.get("/logs/queries", { params }),
  getOperations: (params: {
    action_filter?: string;
    page?: number;
    page_size?: number;
  }) => api.get("/logs/operations", { params }),
};

// === Content Management ===
export const contentApi = {
  // Announcements
  listAnnouncements: () => api.get("/content/announcements"),
  createAnnouncement: (data: { title: string; content: string; is_pinned?: boolean }) =>
    api.post("/content/announcements", data),
  updateAnnouncement: (id: string, data: Record<string, unknown>) =>
    api.put(`/content/announcements/${id}`, data),
  deleteAnnouncement: (id: string) => api.delete(`/content/announcements/${id}`),
  // FAQ
  listFAQ: (category?: string) =>
    api.get("/content/faq", { params: category ? { category } : {} }),
  createFAQ: (data: { category: string; question: string; answer: string }) =>
    api.post("/content/faq", data),
  updateFAQ: (id: string, data: Record<string, unknown>) =>
    api.put(`/content/faq/${id}`, data),
  deleteFAQ: (id: string) => api.delete(`/content/faq/${id}`),
  // Surveys
  listSurveys: () => api.get("/content/surveys"),
  createSurvey: (data: { title: string; description?: string; questions: unknown[] }) =>
    api.post("/content/surveys", data),
  submitSurveyResponse: (id: string, answers: unknown[]) =>
    api.post(`/content/surveys/${id}/respond`, { answers }),
  getSurveyResults: (id: string) => api.get(`/content/surveys/${id}/results`),
};

// === Guardrails ===
export const guardrailsApi = {
  list: () => api.get("/guardrails/admin/guardrails"),
  create: (data: { name: string; rule_type: string; pattern: string; action: string; message?: string; is_active?: boolean }) =>
    api.post("/guardrails/admin/guardrails", data),
  update: (id: string, data: Record<string, unknown>) =>
    api.put(`/guardrails/admin/guardrails/${id}`, data),
  delete: (id: string) => api.delete(`/guardrails/admin/guardrails/${id}`),
  getLogs: (limit = 50) => api.get("/guardrails/admin/guardrails/logs", { params: { limit } }),
  test: (data: { rule_id: string; test_input: string }) =>
    api.post("/guardrails/admin/guardrails/test", data),
};

// === OCR ===
export const ocrApi = {
  health: () => api.get("/ocr/health"),
  process: (file: File) => {
    const formData = new FormData();
    formData.append("file", file);
    return api.post("/ocr/process", formData, { headers: { "Content-Type": "multipart/form-data" } });
  },
  switchProvider: (provider: string) => api.post("/ocr/switch-provider", { provider }),
  listTrainingData: (limit = 50, offset = 0) =>
    api.get("/ocr/training-data", { params: { limit, offset } }),
  getTrainingStats: () => api.get("/ocr/training-data/stats"),
  updateTrainingLabel: (id: string, data: Record<string, unknown>) =>
    api.put(`/ocr/training-data/${id}`, data),
  deleteTrainingData: (id: string) => api.delete(`/ocr/training-data/${id}`),
  exportTrainingData: (format = "jsonl") =>
    api.post("/ocr/training-data/export", { format }),
};

// === Personal Knowledge ===
export const personalKnowledgeApi = {
  upload: (file: File) => {
    const formData = new FormData();
    formData.append("file", file);
    return api.post("/my-knowledge/upload", formData, { headers: { "Content-Type": "multipart/form-data" } });
  },
  list: () => api.get("/my-knowledge"),
  delete: (docId: string) => api.delete(`/my-knowledge/${docId}`),
  piiScan: (docId: string) => api.post(`/my-knowledge/${docId}/pii-scan`),
};

// === Sync ===
export const syncApi = {
  status: () => api.get("/sync/status"),
  trigger: () => api.post("/sync/trigger"),
  schedule: (cron: string) => api.post("/sync/schedule", { cron_expression: cron }),
  stop: () => api.post("/sync/stop"),
  history: (limit = 20) => api.get("/sync/history", { params: { limit } }),
};

// === Bookmarks ===
export const bookmarksApi = {
  list: () => api.get("/bookmarks"),
  add: (data: {
    message_id: string;
    session_id: string;
    content: string;
    role?: string;
    note?: string;
  }) => api.post("/bookmarks", data),
  remove: (messageId: string) => api.delete(`/bookmarks/${messageId}`),
};

// === Folders ===
export const foldersApi = {
  list: () => api.get("/folders"),
  create: (data: { name: string; description?: string; access_level?: string }) =>
    api.post("/folders", data),
  get: (id: string) => api.get(`/folders/${id}`),
  update: (id: string, data: Record<string, unknown>) => api.put(`/folders/${id}`, data),
  delete: (id: string) => api.delete(`/folders/${id}`),
  getPermissions: (id: string) => api.get(`/folders/${id}/permissions`),
  setPermission: (id: string, data: { user_id: string; permission: string }) =>
    api.post(`/folders/${id}/permissions`, data),
  removePermission: (permId: string) => api.delete(`/folders/permissions/${permId}`),
};

// === Corrections ===
export const correctionsApi = {
  list: (status?: string) =>
    api.get("/corrections", { params: status ? { status } : {} }),
  stats: () => api.get("/corrections/stats"),
  submit: (data: {
    message_id: string;
    session_id: string;
    query: string;
    original_answer: string;
    corrected_answer: string;
    correction_reason?: string;
    category?: string;
    difficulty?: string;
  }) => api.post("/corrections", data),
  review: (id: string, data: { approved: boolean; reviewer_comment?: string }) =>
    api.post(`/corrections/${id}/review`, data),
};

// === Governance ===
export const governanceApi = {
  list: (status?: string) =>
    api.get("/governance/changes", { params: status ? { status } : {} }),
  create: (data: {
    change_type: string;
    description: string;
    current_value: Record<string, unknown>;
    proposed_value: Record<string, unknown>;
  }) => api.post("/governance/changes", data),
  simulate: (id: string) => api.post(`/governance/changes/${id}/simulate`),
  approve: (id: string, data: { approved: boolean; comment?: string }) =>
    api.post(`/governance/changes/${id}/approve`, data),
  apply: (id: string) => api.post(`/governance/changes/${id}/apply`),
};

/**
 * Helper to get auth header for non-axios requests (e.g., SSE fetch).
 */
export function getAuthHeaders(): Record<string, string> {
  const token = localStorage.getItem(ACCESS_TOKEN_KEY);
  if (token) {
    return { Authorization: `Bearer ${token}` };
  }
  return {};
}

export { API_BASE };
export default api;
