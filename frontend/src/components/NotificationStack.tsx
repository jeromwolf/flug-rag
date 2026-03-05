import { useEffect, useRef } from "react";
import { Box, Alert, IconButton, Slide, Collapse } from "@mui/material";
import { Close as CloseIcon } from "@mui/icons-material";
import { useAppStore, type Notification } from "../stores/appStore";

const DEFAULT_DURATION = 4000;
const MAX_VISIBLE = 3;

interface NotificationItemProps {
  notification: Notification;
  onDismiss: (id: string) => void;
}

function NotificationItem({ notification, onDismiss }: NotificationItemProps) {
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const duration = notification.duration ?? DEFAULT_DURATION;

  useEffect(() => {
    timerRef.current = setTimeout(() => {
      onDismiss(notification.id);
    }, duration);

    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, [notification.id, duration, onDismiss]);

  return (
    <Slide direction="up" in mountOnEnter unmountOnExit>
      <Alert
        severity={notification.severity}
        variant="filled"
        action={
          <IconButton
            size="small"
            aria-label="close"
            color="inherit"
            onClick={() => onDismiss(notification.id)}
          >
            <CloseIcon fontSize="small" />
          </IconButton>
        }
        sx={{
          width: "100%",
          maxWidth: 480,
          boxShadow: 4,
          "& .MuiAlert-message": { flex: 1 },
        }}
      >
        {notification.message}
      </Alert>
    </Slide>
  );
}

export default function NotificationStack() {
  const notifications = useAppStore((s) => s.notifications);
  const dismissNotification = useAppStore((s) => s.dismissNotification);

  // Show only the most recent MAX_VISIBLE; oldest overflow silently expires via timer
  const visible = notifications.slice(-MAX_VISIBLE);

  return (
    <Box
      aria-live="polite"
      aria-atomic="false"
      sx={{
        position: "fixed",
        bottom: 24,
        left: "50%",
        transform: "translateX(-50%)",
        zIndex: (theme) => theme.zIndex.snackbar,
        display: "flex",
        flexDirection: "column",
        gap: 1,
        alignItems: "center",
        pointerEvents: "none",
        "& > *": { pointerEvents: "auto" },
      }}
    >
      {visible.map((n) => (
        <Collapse key={n.id} in unmountOnExit>
          <NotificationItem notification={n} onDismiss={dismissNotification} />
        </Collapse>
      ))}
    </Box>
  );
}
