import { create } from "zustand";

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

  // Theme
  darkMode: boolean;
  toggleDarkMode: () => void;
}

export const useAppStore = create<AppState>((set) => ({
  currentSessionId: null,
  setCurrentSessionId: (id) => set({ currentSessionId: id }),

  responseMode: "rag",
  setResponseMode: (mode) => set({ responseMode: mode }),

  selectedModel: "default",
  setSelectedModel: (model) => set({ selectedModel: model }),

  temperature: parseFloat(
    (typeof window !== "undefined" && localStorage.getItem("flux_temperature")) || "0.7"
  ),
  setTemperature: (temp) =>
    set(() => {
      localStorage.setItem("flux_temperature", String(temp));
      return { temperature: temp };
    }),

  sidebarOpen: true,
  toggleSidebar: () => set((state) => ({ sidebarOpen: !state.sidebarOpen })),

  darkMode: typeof window !== 'undefined' && localStorage.getItem('flux_dark_mode') === 'true',
  toggleDarkMode: () => set((state) => {
    const newMode = !state.darkMode;
    localStorage.setItem('flux_dark_mode', String(newMode));
    return { darkMode: newMode };
  }),
}));
