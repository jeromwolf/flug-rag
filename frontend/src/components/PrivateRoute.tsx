import { Navigate } from "react-router-dom";
import { Box, CircularProgress, Typography } from "@mui/material";
import { useAuth, type UserRole } from "../contexts/AuthContext";
import type { ReactNode } from "react";

interface PrivateRouteProps {
  children: ReactNode;
  /** If set, user must hold one of these roles */
  requiredRoles?: UserRole[];
  /** If set, user must have this specific permission */
  requiredPermission?: string;
}

export default function PrivateRoute({
  children,
  requiredRoles,
  requiredPermission,
}: PrivateRouteProps) {
  const { isAuthenticated, isLoading, authEnabled, hasRole, hasPermission } =
    useAuth();

  // Still checking auth state
  if (isLoading) {
    return (
      <Box
        sx={{
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          height: "100vh",
          gap: 2,
        }}
      >
        <CircularProgress />
        <Typography variant="body2" color="text.secondary">
          인증 확인 중...
        </Typography>
      </Box>
    );
  }

  // Auth disabled -- allow everything
  if (!authEnabled) {
    return <>{children}</>;
  }

  // Not authenticated -- redirect to login
  if (!isAuthenticated) {
    return <Navigate to="/login" replace />;
  }

  // Role check — redirect to chat if unauthorized
  if (requiredRoles && !hasRole(requiredRoles)) {
    return <Navigate to="/chat" replace />;
  }

  // Permission check — redirect to chat if unauthorized
  if (requiredPermission && !hasPermission(requiredPermission)) {
    return <Navigate to="/chat" replace />;
  }

  return <>{children}</>;
}
