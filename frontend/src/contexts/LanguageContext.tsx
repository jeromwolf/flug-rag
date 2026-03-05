import { createContext, useContext, useState, useCallback, useMemo } from "react";
import type { ReactNode } from "react";
import ko from "../i18n/ko";
import en from "../i18n/en";
import type { TranslationKey } from "../i18n/ko";

// ── Types ─────────────────────────────────────────────────────────────────────

export type Language = "ko" | "en";

const STORAGE_KEY = "flux-rag-language";

const TRANSLATIONS: Record<Language, Record<TranslationKey, string>> = {
  ko,
  en,
};

// ── Context shape ─────────────────────────────────────────────────────────────

interface LanguageContextValue {
  language: Language;
  setLanguage: (lang: Language) => void;
  t: (key: TranslationKey) => string;
}

// ── Context ───────────────────────────────────────────────────────────────────

const LanguageContext = createContext<LanguageContextValue | null>(null);

// ── Helpers ───────────────────────────────────────────────────────────────────

function loadLanguage(): Language {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored === "ko" || stored === "en") return stored;
  } catch {
    // localStorage may be unavailable in certain environments
  }
  // Default to Korean (primary audience)
  return "ko";
}

// ── Provider ──────────────────────────────────────────────────────────────────

export function LanguageProvider({ children }: { children: ReactNode }) {
  const [language, setLanguageState] = useState<Language>(loadLanguage);

  const setLanguage = useCallback((lang: Language) => {
    setLanguageState(lang);
    try {
      localStorage.setItem(STORAGE_KEY, lang);
    } catch {
      // ignore
    }
  }, []);

  const t = useCallback(
    (key: TranslationKey): string => {
      return TRANSLATIONS[language][key] ?? TRANSLATIONS["ko"][key] ?? key;
    },
    [language],
  );

  const value = useMemo<LanguageContextValue>(
    () => ({ language, setLanguage, t }),
    [language, setLanguage, t],
  );

  return (
    <LanguageContext.Provider value={value}>{children}</LanguageContext.Provider>
  );
}

// ── Hook ─────────────────────────────────────────────────────────────────────

export function useLanguage(): LanguageContextValue {
  const ctx = useContext(LanguageContext);
  if (!ctx) {
    throw new Error("useLanguage must be used within a LanguageProvider");
  }
  return ctx;
}
