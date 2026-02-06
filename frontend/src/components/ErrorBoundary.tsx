import { Component } from "react";
import type { ReactNode, ErrorInfo } from "react";
import { Box, Typography, Button, Paper } from "@mui/material";
import { ErrorOutline as ErrorIcon } from "@mui/icons-material";

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

export default class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error("ErrorBoundary caught:", error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) return this.props.fallback;
      return (
        <Box sx={{ display: "flex", justifyContent: "center", alignItems: "center", height: "100%", p: 4 }}>
          <Paper sx={{ p: 4, textAlign: "center", maxWidth: 500 }}>
            <ErrorIcon sx={{ fontSize: 48, color: "error.main", mb: 2 }} />
            <Typography variant="h6" gutterBottom>
              오류가 발생했습니다
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
              {this.state.error?.message || "알 수 없는 오류가 발생했습니다."}
            </Typography>
            <Button variant="contained" onClick={() => this.setState({ hasError: false, error: null })}>
              다시 시도
            </Button>
          </Paper>
        </Box>
      );
    }
    return this.props.children;
  }
}
