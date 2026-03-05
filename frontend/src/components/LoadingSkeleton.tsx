import { Box, Skeleton } from "@mui/material";

type SkeletonVariant = "chat" | "list" | "table" | "card";

interface LoadingSkeletonProps {
  variant?: SkeletonVariant;
  count?: number;
}

function ChatSkeleton() {
  return (
    <Box sx={{ display: "flex", flexDirection: "column", gap: 3, p: 3, maxWidth: 768, mx: "auto", width: "100%" }}>
      {/* AI message */}
      <Box sx={{ display: "flex", gap: 2, alignItems: "flex-start" }}>
        <Skeleton variant="circular" width={32} height={32} sx={{ flexShrink: 0 }} />
        <Box sx={{ flex: 1 }}>
          <Skeleton variant="text" width="90%" height={20} />
          <Skeleton variant="text" width="80%" height={20} />
          <Skeleton variant="text" width="60%" height={20} />
        </Box>
      </Box>
      {/* User message */}
      <Box sx={{ display: "flex", gap: 2, alignItems: "flex-start", flexDirection: "row-reverse" }}>
        <Skeleton variant="circular" width={32} height={32} sx={{ flexShrink: 0 }} />
        <Box sx={{ flex: 1, display: "flex", flexDirection: "column", alignItems: "flex-end" }}>
          <Skeleton variant="rounded" width="55%" height={40} />
        </Box>
      </Box>
      {/* AI message */}
      <Box sx={{ display: "flex", gap: 2, alignItems: "flex-start" }}>
        <Skeleton variant="circular" width={32} height={32} sx={{ flexShrink: 0 }} />
        <Box sx={{ flex: 1 }}>
          <Skeleton variant="text" width="95%" height={20} />
          <Skeleton variant="text" width="85%" height={20} />
          <Skeleton variant="text" width="70%" height={20} />
          <Skeleton variant="text" width="50%" height={20} />
        </Box>
      </Box>
    </Box>
  );
}

function ListSkeleton({ count }: { count: number }) {
  return (
    <Box sx={{ display: "flex", flexDirection: "column", gap: 0.5, px: 1, py: 1 }}>
      {Array.from({ length: count }).map((_, i) => (
        <Box key={i} sx={{ display: "flex", flexDirection: "column", px: 1, py: 0.75 }}>
          <Skeleton variant="text" width={`${65 + (i % 3) * 10}%`} height={18} />
          <Skeleton variant="text" width="40%" height={14} />
        </Box>
      ))}
    </Box>
  );
}

function TableSkeleton({ count }: { count: number }) {
  return (
    <Box sx={{ display: "flex", flexDirection: "column", gap: 1 }}>
      {/* Header */}
      <Box sx={{ display: "flex", gap: 2, px: 2, py: 1.5, borderBottom: "1px solid", borderColor: "divider" }}>
        <Skeleton variant="text" width={30} height={20} />
        <Skeleton variant="text" width={180} height={20} />
        <Skeleton variant="text" width={80} height={20} />
        <Skeleton variant="text" width={100} height={20} />
        <Skeleton variant="text" width={80} height={20} sx={{ ml: "auto" }} />
      </Box>
      {/* Rows */}
      {Array.from({ length: count }).map((_, i) => (
        <Box key={i} sx={{ display: "flex", gap: 2, px: 2, py: 1.25, alignItems: "center" }}>
          <Skeleton variant="rounded" width={20} height={20} />
          <Skeleton variant="text" width={`${140 + (i % 4) * 30}px`} height={20} />
          <Skeleton variant="rounded" width={60} height={22} sx={{ borderRadius: 3 }} />
          <Skeleton variant="text" width={90} height={20} />
          <Skeleton variant="text" width={70} height={20} sx={{ ml: "auto" }} />
        </Box>
      ))}
    </Box>
  );
}

function CardSkeleton({ count }: { count: number }) {
  return (
    <Box sx={{ display: "flex", flexWrap: "wrap", gap: 2 }}>
      {Array.from({ length: count }).map((_, i) => (
        <Box
          key={i}
          sx={{
            flex: "1 1 240px",
            maxWidth: 320,
            border: "1px solid",
            borderColor: "divider",
            borderRadius: 2,
            p: 2.5,
            display: "flex",
            flexDirection: "column",
            gap: 1.5,
          }}
        >
          <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
            <Skeleton variant="rounded" width={36} height={36} sx={{ borderRadius: 1.5 }} />
            <Skeleton variant="text" width={100} height={22} />
          </Box>
          <Skeleton variant="text" width="90%" height={18} />
          <Skeleton variant="text" width="70%" height={18} />
          <Skeleton variant="text" width={80} height={32} sx={{ mt: 0.5 }} />
        </Box>
      ))}
    </Box>
  );
}

export default function LoadingSkeleton({ variant = "list", count = 5 }: LoadingSkeletonProps) {
  switch (variant) {
    case "chat":
      return <ChatSkeleton />;
    case "table":
      return <TableSkeleton count={count} />;
    case "card":
      return <CardSkeleton count={count} />;
    case "list":
    default:
      return <ListSkeleton count={count} />;
  }
}
