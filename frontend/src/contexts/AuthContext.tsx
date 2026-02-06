import {
  createContext,
  useContext,
  useState,
  useEffect,
  useCallback,
  useRef,
  type ReactNode,
} from "react";
import api from "../api/client";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

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

interface AuthState {
  user: AuthUser | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  authEnabled: boolean;
  login: (username: string, password: string) => Promise<void>;
  logout: () => Promise<void>;
  hasRole: (roles: UserRole[]) => boolean;
  hasPermission: (permission: string) => boolean;
}

// ---------------------------------------------------------------------------
// Permission matrix (mirrors backend)
// ---------------------------------------------------------------------------

const ROLE_PERMISSIONS: Record<UserRole, Set<string>> = {
  admin: new Set([
    "chat:read", "chat:write", "documents:read", "documents:write",
    "documents:delete", "admin:read", "admin:write", "monitor:read",
    "users:read", "users:write", "feedback:read", "feedback:write",
    "sessions:read", "sessions:write", "sessions:delete",
    "mcp:read", "mcp:execute", "workflows:read", "workflows:execute",
    "agent-builder:read", "agent-builder:write", "settings:read", "settings:write",
  ]),
  manager: new Set([
    "chat:read", "chat:write", "documents:read", "documents:write",
    "admin:read", "monitor:read", "feedback:read", "feedback:write",
    "sessions:read", "sessions:write", "sessions:delete",
    "mcp:read", "workflows:read", "workflows:execute", "agent-builder:read",
  ]),
  user: new Set([
    "chat:read", "chat:write", "documents:read",
    "sessions:read", "sessions:write", "sessions:delete",
    "feedback:write", "mcp:read", "workflows:read",
  ]),
  viewer: new Set(["chat:read", "sessions:read"]),
};

// ---------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------

const AuthContext = createContext<AuthState | undefined>(undefined);

// ---------------------------------------------------------------------------
// Token helpers
// ---------------------------------------------------------------------------

const ACCESS_TOKEN_KEY = "flux_access_token";
const REFRESH_TOKEN_KEY = "flux_refresh_token";
const USER_KEY = "flux_user";

function getStoredTokens() {
  return {
    accessToken: localStorage.getItem(ACCESS_TOKEN_KEY),
    refreshToken: localStorage.getItem(REFRESH_TOKEN_KEY),
  };
}

function storeTokens(accessToken: string, refreshToken: string) {
  localStorage.setItem(ACCESS_TOKEN_KEY, accessToken);
  localStorage.setItem(REFRESH_TOKEN_KEY, refreshToken);
}

function clearTokens() {
  localStorage.removeItem(ACCESS_TOKEN_KEY);
  localStorage.removeItem(REFRESH_TOKEN_KEY);
  localStorage.removeItem(USER_KEY);
}

function getStoredUser(): AuthUser | null {
  try {
    const raw = localStorage.getItem(USER_KEY);
    return raw ? (JSON.parse(raw) as AuthUser) : null;
  } catch {
    return null;
  }
}

function storeUser(user: AuthUser) {
  localStorage.setItem(USER_KEY, JSON.stringify(user));
}

// ---------------------------------------------------------------------------
// Provider
// ---------------------------------------------------------------------------

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<AuthUser | null>(getStoredUser);
  const [isLoading, setIsLoading] = useState(true);
  const [authEnabled, setAuthEnabled] = useState(true);
  const refreshTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Check whether auth is enabled by hitting the health endpoint
  // The health endpoint is at the server root, not under /api
  useEffect(() => {
    const baseUrl = (api.defaults.baseURL || "http://localhost:8000/api").replace(/\/api\/?$/, "");
    fetch(`${baseUrl}/health`)
      .then((r) => r.json())
      .catch(() => ({ auth_enabled: true }))
      .then((data) => {
        const enabled = data?.auth_enabled ?? true;
        setAuthEnabled(enabled);
        if (!enabled) {
          // Auth disabled -- auto-authenticate as dev admin
          const devUser: AuthUser = {
            id: "dev-admin",
            username: "dev",
            email: "dev@localhost",
            full_name: "Developer (auth disabled)",
            department: "Development",
            role: "admin",
            is_active: true,
          };
          setUser(devUser);
          setIsLoading(false);
        }
      });
  }, []);

  // On mount, validate stored token
  useEffect(() => {
    if (!authEnabled) return;

    const { accessToken } = getStoredTokens();
    if (!accessToken) {
      setIsLoading(false);
      return;
    }

    api
      .get("/auth/me", { headers: { Authorization: `Bearer ${accessToken}` } })
      .then((res) => {
        const u = res.data as AuthUser;
        setUser(u);
        storeUser(u);
        scheduleRefresh();
      })
      .catch(() => {
        clearTokens();
        setUser(null);
      })
      .finally(() => setIsLoading(false));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [authEnabled]);

  // Schedule token refresh ~1 minute before expiry (assume 30 min default)
  const scheduleRefresh = useCallback(() => {
    if (refreshTimerRef.current) clearTimeout(refreshTimerRef.current);
    // Refresh 2 minutes before expiry
    const delayMs = (28) * 60 * 1000; // 28 minutes
    refreshTimerRef.current = setTimeout(async () => {
      const { refreshToken } = getStoredTokens();
      if (!refreshToken) return;
      try {
        const res = await api.post("/auth/refresh", {
          refresh_token: refreshToken,
        });
        storeTokens(res.data.access_token, res.data.refresh_token);
        if (res.data.user) {
          const u = (typeof res.data.user === "object" ? res.data.user : JSON.parse(res.data.user)) as AuthUser;
          setUser(u);
          storeUser(u);
        }
        scheduleRefresh();
      } catch {
        clearTokens();
        setUser(null);
      }
    }, delayMs);
  }, []);

  const login = useCallback(
    async (username: string, password: string) => {
      const res = await api.post("/auth/login", { username, password });
      const { access_token, refresh_token, user: userData } = res.data;
      storeTokens(access_token, refresh_token);
      const u = userData as AuthUser;
      setUser(u);
      storeUser(u);
      scheduleRefresh();
    },
    [scheduleRefresh],
  );

  const logout = useCallback(async () => {
    const { accessToken } = getStoredTokens();
    try {
      if (accessToken) {
        await api.post("/auth/logout", null, {
          headers: { Authorization: `Bearer ${accessToken}` },
        });
      }
    } catch {
      // Ignore errors on logout
    }
    clearTokens();
    setUser(null);
    if (refreshTimerRef.current) clearTimeout(refreshTimerRef.current);
  }, []);

  const hasRole = useCallback(
    (roles: UserRole[]) => {
      if (!user) return false;
      return roles.includes(user.role);
    },
    [user],
  );

  const hasPermission = useCallback(
    (permission: string) => {
      if (!user) return false;
      return ROLE_PERMISSIONS[user.role]?.has(permission) ?? false;
    },
    [user],
  );

  return (
    <AuthContext.Provider
      value={{
        user,
        isAuthenticated: user !== null,
        isLoading,
        authEnabled,
        login,
        logout,
        hasRole,
        hasPermission,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

export function useAuth(): AuthState {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within AuthProvider");
  return ctx;
}
