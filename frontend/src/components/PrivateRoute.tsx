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

  // Role check
  if (requiredRoles && !hasRole(requiredRoles)) {
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
        <Typography variant="h5" color="error">
          접근 권한이 없습니다
        </Typography>
        <Typography variant="body1" color="text.secondary">
          이 페이지에 접근하려면 적절한 권한이 필요합니다.
        </Typography>
      </Box>
    );
  }

  // Permission check
  if (requiredPermission && !hasPermission(requiredPermission)) {
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
        <Typography variant="h5" color="error">
          접근 권한이 없습니다
        </Typography>
        <Typography variant="body1" color="text.secondary">
          이 페이지에 접근할 수 있는 권한이 부족합니다.
        </Typography>
      </Box>
    );
  }

  return <>{children}</>;
}
