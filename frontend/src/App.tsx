import { useMemo, lazy, Suspense } from "react";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { ThemeProvider, createTheme, CssBaseline, Snackbar, Alert, CircularProgress, Box } from "@mui/material";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import ErrorBoundary from "./components/ErrorBoundary";
import { AuthProvider } from "./contexts/AuthContext";
import { LanguageProvider } from "./contexts/LanguageContext";
import PrivateRoute from "./components/PrivateRoute";
import { useAppStore } from "./stores/appStore";
import NotificationStack from "./components/NotificationStack";
import NetworkStatusBanner from "./components/NetworkStatusBanner";
import LoginPage from "./pages/LoginPage";
import ChatPage from "./pages/ChatPage";

const AdminPage = lazy(() => import('./pages/AdminPage'));
const MonitorPage = lazy(() => import('./pages/MonitorPage'));
const DocumentsPage = lazy(() => import('./pages/DocumentsPage'));
const QualityDashboardPage = lazy(() => import('./pages/QualityDashboardPage'));
const AgentBuilderPage = lazy(() => import('./pages/AgentBuilderPage'));
const PersonalKnowledgePage = lazy(() => import('./pages/PersonalKnowledgePage'));
const ActivityPage = lazy(() => import('./pages/ActivityPage'));

function LoadingFallback() {
  return (
    <Box
      sx={{
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        height: "100vh",
      }}
    >
      <CircularProgress color="primary" />
    </Box>
  );
}

const queryClient = new QueryClient({
  defaultOptions: {
    queries: { retry: 1, staleTime: 30000 },
  },
});

function GlobalErrorSnackbar() {
  const globalError = useAppStore((s) => s.globalError);
  const setGlobalError = useAppStore((s) => s.setGlobalError);

  return (
    <Snackbar
      open={!!globalError}
      autoHideDuration={5000}
      onClose={() => setGlobalError(null)}
      anchorOrigin={{ vertical: "bottom", horizontal: "center" }}
    >
      <Alert
        severity="error"
        variant="filled"
        onClose={() => setGlobalError(null)}
        sx={{ width: "100%", maxWidth: 480 }}
      >
        {globalError}
      </Alert>
    </Snackbar>
  );
}

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
      {/* Network status banner — fixed top, above everything */}
      <NetworkStatusBanner />
      {/* Outer boundary catches catastrophic failures (theme provider, router init) */}
      <ErrorBoundary>
        <BrowserRouter>
          <AuthProvider>
            <Routes>
              <Route
                path="/login"
                element={
                  <ErrorBoundary>
                    <LoginPage />
                  </ErrorBoundary>
                }
              />
              <Route path="/" element={<Navigate to="/chat" replace />} />
              <Route
                path="/chat"
                element={
                  <ErrorBoundary>
                    <PrivateRoute>
                      <ChatPage />
                    </PrivateRoute>
                  </ErrorBoundary>
                }
              />
              <Route
                path="/documents"
                element={
                  <ErrorBoundary>
                    <PrivateRoute requiredPermission="documents:read">
                      <Suspense fallback={<LoadingFallback />}>
                        <DocumentsPage />
                      </Suspense>
                    </PrivateRoute>
                  </ErrorBoundary>
                }
              />
              <Route
                path="/admin"
                element={
                  <ErrorBoundary>
                    <PrivateRoute requiredPermission="admin:read">
                      <Suspense fallback={<LoadingFallback />}>
                        <AdminPage />
                      </Suspense>
                    </PrivateRoute>
                  </ErrorBoundary>
                }
              />
              <Route
                path="/monitor"
                element={
                  <ErrorBoundary>
                    <PrivateRoute requiredPermission="monitor:read">
                      <Suspense fallback={<LoadingFallback />}>
                        <MonitorPage />
                      </Suspense>
                    </PrivateRoute>
                  </ErrorBoundary>
                }
              />
              <Route
                path="/agent-builder"
                element={
                  <ErrorBoundary>
                    <PrivateRoute requiredPermission="agent-builder:read">
                      <Suspense fallback={<LoadingFallback />}>
                        <AgentBuilderPage />
                      </Suspense>
                    </PrivateRoute>
                  </ErrorBoundary>
                }
              />
              <Route
                path="/quality"
                element={
                  <ErrorBoundary>
                    <PrivateRoute requiredPermission="admin:read">
                      <Suspense fallback={<LoadingFallback />}>
                        <QualityDashboardPage />
                      </Suspense>
                    </PrivateRoute>
                  </ErrorBoundary>
                }
              />
              <Route
                path="/personal-knowledge"
                element={
                  <ErrorBoundary>
                    <PrivateRoute>
                      <Suspense fallback={<LoadingFallback />}>
                        <PersonalKnowledgePage />
                      </Suspense>
                    </PrivateRoute>
                  </ErrorBoundary>
                }
              />
              <Route
                path="/activity"
                element={
                  <ErrorBoundary>
                    <PrivateRoute>
                      <Suspense fallback={<LoadingFallback />}>
                        <ActivityPage />
                      </Suspense>
                    </PrivateRoute>
                  </ErrorBoundary>
                }
              />
            </Routes>
          </AuthProvider>
        </BrowserRouter>
      </ErrorBoundary>
      {/* Global error notification — reads from Zustand store */}
      <GlobalErrorSnackbar />
      {/* Centralized toast notification stack */}
      <NotificationStack />
    </ThemeProvider>
  );
}

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <LanguageProvider>
        <AppContent />
      </LanguageProvider>
    </QueryClientProvider>
  );
}
