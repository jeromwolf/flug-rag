import { useMemo } from "react";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { ThemeProvider, createTheme, CssBaseline } from "@mui/material";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import ErrorBoundary from "./components/ErrorBoundary";
import { AuthProvider } from "./contexts/AuthContext";
import PrivateRoute from "./components/PrivateRoute";
import { useAppStore } from "./stores/appStore";
import LoginPage from "./pages/LoginPage";
import ChatPage from "./pages/ChatPage";
import DocumentsPage from "./pages/DocumentsPage";
import AdminPage from "./pages/AdminPage";
import MonitorPage from "./pages/MonitorPage";
import AgentBuilderPage from "./pages/AgentBuilderPage";
import QualityDashboardPage from "./pages/QualityDashboardPage";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: { retry: 1, staleTime: 30000 },
  },
});

function AppContent() {
  const darkMode = useAppStore((s) => s.darkMode);

  const theme = useMemo(
    () =>
      createTheme({
        palette: {
          mode: darkMode ? "dark" : "light",
          primary: { main: "#10a37f" },
          secondary: { main: "#6e6e80" },
          ...(darkMode
            ? {
                background: {
                  default: "#212121",
                  paper: "#2f2f2f",
                },
                text: {
                  primary: "#ececec",
                  secondary: "#b4b4b4",
                },
              }
            : {
                background: {
                  default: "#ffffff",
                  paper: "#f7f7f8",
                },
                text: {
                  primary: "#343541",
                  secondary: "#6e6e80",
                },
              }),
        },
        typography: {
          fontFamily:
            '"Pretendard Variable", "Pretendard", "Noto Sans KR", -apple-system, BlinkMacSystemFont, sans-serif',
          body1: { fontSize: "0.9375rem", lineHeight: 1.7 },
          body2: { fontSize: "0.875rem", lineHeight: 1.6 },
        },
        shape: {
          borderRadius: 12,
        },
        components: {
          MuiCssBaseline: {
            styleOverrides: `
              @keyframes fadeInUp {
                from { opacity: 0; transform: translateY(8px); }
                to { opacity: 1; transform: translateY(0); }
              }
              body {
                scrollbar-width: thin;
              }
              body::-webkit-scrollbar {
                width: 6px;
              }
              body::-webkit-scrollbar-thumb {
                border-radius: 3px;
                background-color: ${darkMode ? "rgba(255,255,255,0.2)" : "rgba(0,0,0,0.2)"};
              }
            `,
          },
        },
      }),
    [darkMode]
  );

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <ErrorBoundary>
        <BrowserRouter>
          <AuthProvider>
            <Routes>
              <Route path="/login" element={<LoginPage />} />
              <Route path="/" element={<Navigate to="/chat" replace />} />
              <Route
                path="/chat"
                element={
                  <PrivateRoute>
                    <ChatPage />
                  </PrivateRoute>
                }
              />
              <Route
                path="/documents"
                element={
                  <PrivateRoute requiredPermission="documents:read">
                    <DocumentsPage />
                  </PrivateRoute>
                }
              />
              <Route
                path="/admin"
                element={
                  <PrivateRoute requiredPermission="admin:read">
                    <AdminPage />
                  </PrivateRoute>
                }
              />
              <Route
                path="/monitor"
                element={
                  <PrivateRoute requiredPermission="monitor:read">
                    <MonitorPage />
                  </PrivateRoute>
                }
              />
              <Route
                path="/agent-builder"
                element={
                  <PrivateRoute requiredPermission="agent-builder:read">
                    <AgentBuilderPage />
                  </PrivateRoute>
                }
              />
              <Route
                path="/quality"
                element={
                  <PrivateRoute requiredPermission="admin:read">
                    <QualityDashboardPage />
                  </PrivateRoute>
                }
              />
            </Routes>
          </AuthProvider>
        </BrowserRouter>
      </ErrorBoundary>
    </ThemeProvider>
  );
}

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <AppContent />
    </QueryClientProvider>
  );
}
