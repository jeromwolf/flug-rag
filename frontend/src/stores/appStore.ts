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

  sidebarOpen: true,
  toggleSidebar: () => set((state) => ({ sidebarOpen: !state.sidebarOpen })),

  darkMode: false,
  toggleDarkMode: () => set((state) => ({ darkMode: !state.darkMode })),
}));
