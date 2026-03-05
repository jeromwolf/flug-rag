import { Box, Slide, Typography } from "@mui/material";
import { Wifi as WifiIcon, WifiOff as WifiOffIcon } from "@mui/icons-material";
import { useNetworkStatus } from "../hooks/useNetworkStatus";

export default function NetworkStatusBanner() {
  const { isOnline, wasOffline } = useNetworkStatus();

  const showOffline = !isOnline;
  const showReconnected = isOnline && wasOffline;
  const showAny = showOffline || showReconnected;

  return (
    <Slide direction="down" in={showAny} mountOnEnter unmountOnExit>
      <Box
        role="status"
        aria-live="polite"
        sx={{
          position: "fixed",
          top: 0,
          left: 0,
          right: 0,
          zIndex: 9999,
          height: 36,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          gap: 0.75,
          bgcolor: showOffline ? "error.main" : "success.main",
          color: "#fff",
          transition: "background-color 0.3s ease",
        }}
      >
        {showOffline ? (
          <WifiOffIcon sx={{ fontSize: 16 }} />
        ) : (
          <WifiIcon sx={{ fontSize: 16 }} />
        )}
        <Typography variant="caption" sx={{ fontWeight: 600, fontSize: "0.8rem" }}>
          {showOffline
            ? "인터넷 연결이 끊어졌습니다"
            : "연결되었습니다"}
        </Typography>
      </Box>
    </Slide>
  );
}
