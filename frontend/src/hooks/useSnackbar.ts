import { useState } from "react";

interface SnackbarState {
  open: boolean;
  message: string;
  severity: "success" | "error" | "info";
}

export function useSnackbar() {
  const [snackbar, setSnackbar] = useState<SnackbarState>({
    open: false,
    message: "",
    severity: "info",
  });

  const showSnackbar = (message: string, severity: SnackbarState["severity"]) => {
    setSnackbar({ open: true, message, severity });
  };

  const closeSnackbar = () => {
    setSnackbar((prev) => ({ ...prev, open: false }));
  };

  return { snackbar, showSnackbar, closeSnackbar };
}
