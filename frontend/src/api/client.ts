import axios from "axios";

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000/api";

const api = axios.create({
  baseURL: API_BASE,
  headers: {
    "Content-Type": "application/json",
  },
});

// ---------------------------------------------------------------------------
// Auth interceptors
// ---------------------------------------------------------------------------

const ACCESS_TOKEN_KEY = "flux_access_token";
const REFRESH_TOKEN_KEY = "flux_refresh_token";

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
        localStorage.removeItem("flux_user");
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
};

// === Admin ===
export const adminApi = {
  getInfo: () => api.get("/admin/info"),
  getProviders: () => api.get("/admin/providers"),
  getPrompts: () => api.get("/admin/prompts"),
  updatePrompt: (name: string, content: string) =>
    api.put("/admin/prompts", { name, content }),
};

// === Feedback ===
export const feedbackApi = {
  submit: (data: {
    message_id: string;
    session_id: string;
    rating: number;
    comment?: string;
    corrected_answer?: string;
  }) => api.post("/feedback", data),
  list: (limit = 50) => api.get("/feedback", { params: { limit } }),
  stats: () => api.get("/feedback/stats"),
};

// === MCP ===
export const mcpApi = {
  listTools: () => api.get("/mcp/tools"),
  callTool: (toolName: string, args: Record<string, unknown>) =>
    api.post("/mcp/call", { tool_name: toolName, arguments: args }),
};

// === Workflows ===
export const workflowsApi = {
  listPresets: () => api.get("/workflows/presets"),
  run: (preset: string, inputData: Record<string, unknown>) =>
    api.post("/workflows/run", { preset, input_data: inputData }),
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
