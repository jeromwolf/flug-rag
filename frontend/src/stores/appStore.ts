import { create } from "zustand";

export type ThemeMode = "light" | "dark" | "system";
export type FontSize = "small" | "medium" | "large";

export interface Notification {
  id: string;
  message: string;
  severity: "success" | "error" | "warning" | "info";
  duration?: number; // ms, default 4000
}

interface AppState {
  // Current session
  currentSessionId: string | null;
  setCurrentSessionId: (id: string | null) => void;

  // Response mode
  responseMode: "rag" | "direct";
  setResponseMode: (mode: "rag" | "direct") => void;

  // Selected model
  selectedModel: string;
  setSelectedModel: (model: string) => void;

  // Temperature
  temperature: number;
  setTemperature: (temp: number) => void;

  // Sidebar
  sidebarOpen: boolean;
  toggleSidebar: () => void;

  // Theme (legacy toggle + new themeMode)
  darkMode: boolean;
  toggleDarkMode: () => void;
  themeMode: ThemeMode;
  setThemeMode: (mode: ThemeMode) => void;

  // Font size
  fontSize: FontSize;
  setFontSize: (size: FontSize) => void;

  // Browser notifications
  notificationsEnabled: boolean;
  setNotificationsEnabled: (enabled: boolean) => void;

  // Compare mode
  compareMode: boolean;
  toggleCompareMode: () => void;

  // Global error notification (legacy — kept for backward compat)
  globalError: string | null;
  setGlobalError: (message: string | null) => void;

  // Centralized toast notifications
  notifications: Notification[];
  showNotification: (
    message: string,
    severity: Notification["severity"],
    duration?: number
  ) => void;
  dismissNotification: (id: string) => void;
}

function resolveSystemDark(): boolean {
  return typeof window !== "undefined"
    ? window.matchMedia("(prefers-color-scheme: dark)").matches
    : false;
}

function computeDarkMode(mode: ThemeMode): boolean {
  if (mode === "system") return resolveSystemDark();
  return mode === "dark";
}

const storedThemeMode = (
  typeof window !== "undefined" ? localStorage.getItem("app_theme_mode") : null
) as ThemeMode | null;

const initialThemeMode: ThemeMode = storedThemeMode ?? "system";

// Legacy darkMode: prefer app_dark_mode if it was explicitly set before themeMode existed
const legacyDarkStr =
  typeof window !== "undefined" ? localStorage.getItem("app_dark_mode") : null;
const initialDarkMode =
  storedThemeMode !== null
    ? computeDarkMode(initialThemeMode)
    : legacyDarkStr !== null
      ? legacyDarkStr === "true"
      : resolveSystemDark();

const storedFontSize = (
  typeof window !== "undefined" ? localStorage.getItem("app_font_size") : null
) as FontSize | null;
const initialFontSize: FontSize = storedFontSize ?? "medium";

// Apply font size to document body on load
if (typeof document !== "undefined") {
  const fontSizePxMap: Record<FontSize, string> = {
    small: "14px",
    medium: "16px",
    large: "18px",
  };
  document.documentElement.style.fontSize = fontSizePxMap[initialFontSize];
}

export const useAppStore = create<AppState>((set) => ({
  currentSessionId: null,
  setCurrentSessionId: (id) => set({ currentSessionId: id }),

  responseMode: "rag",
  setResponseMode: (mode) => set({ responseMode: mode }),

  selectedModel: "default",
  setSelectedModel: (model) => set({ selectedModel: model }),

  temperature: parseFloat(
    (typeof window !== "undefined" && localStorage.getItem("app_temperature")) || "0.2"
  ),
  setTemperature: (temp) =>
    set(() => {
      localStorage.setItem("app_temperature", String(temp));
      return { temperature: temp };
    }),

  sidebarOpen: typeof window !== "undefined" ? window.innerWidth >= 900 : true,
  toggleSidebar: () => set((state) => ({ sidebarOpen: !state.sidebarOpen })),

  darkMode: initialDarkMode,
  toggleDarkMode: () =>
    set((state) => {
      const newMode = !state.darkMode;
      const newThemeMode: ThemeMode = newMode ? "dark" : "light";
      localStorage.setItem("app_dark_mode", String(newMode));
      localStorage.setItem("app_theme_mode", newThemeMode);
      return { darkMode: newMode, themeMode: newThemeMode };
    }),

  themeMode: initialThemeMode,
  setThemeMode: (mode) =>
    set(() => {
      localStorage.setItem("app_theme_mode", mode);
      const dark = computeDarkMode(mode);
      localStorage.setItem("app_dark_mode", String(dark));
      return { themeMode: mode, darkMode: dark };
    }),

  fontSize: initialFontSize,
  setFontSize: (size) =>
    set(() => {
      localStorage.setItem("app_font_size", size);
      const fontSizePxMap: Record<FontSize, string> = {
        small: "14px",
        medium: "16px",
        large: "18px",
      };
      if (typeof document !== "undefined") {
        document.documentElement.style.fontSize = fontSizePxMap[size];
      }
      return { fontSize: size };
    }),

  notificationsEnabled:
    typeof window !== "undefined" &&
    localStorage.getItem("app_notifications") === "true",
  setNotificationsEnabled: (enabled) =>
    set(() => {
      localStorage.setItem("app_notifications", String(enabled));
      return { notificationsEnabled: enabled };
    }),

  compareMode: false,
  toggleCompareMode: () => set((state) => ({ compareMode: !state.compareMode })),

  globalError: null,
  setGlobalError: (message) => set({ globalError: message }),

  notifications: [],
  showNotification: (message, severity, duration) =>
    set((state) => ({
      notifications: [
        ...state.notifications,
        {
          id: Date.now().toString(36) + Math.random().toString(36).slice(2),
          message,
          severity,
          duration,
        },
      ],
    })),
  dismissNotification: (id) =>
    set((state) => ({
      notifications: state.notifications.filter((n) => n.id !== id),
    })),
}));
