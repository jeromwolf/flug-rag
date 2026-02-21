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
          primary: { main: "#1976d2" },
          secondary: { main: "#dc004e" },
          ...(darkMode
            ? {
                background: {
                  default: "#1a1a2e",
                  paper: "#16213e",
                },
              }
            : {}),
        },
        typography: {
          fontFamily: '"Pretendard", "Noto Sans KR", sans-serif',
        },
        shape: {
          borderRadius: 8,
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
