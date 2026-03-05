import { useState, useCallback, useEffect } from "react";

// ---------------------------------------------------------------------------
// Data model
// ---------------------------------------------------------------------------

export interface SessionTag {
  id: string;
  name: string;
  color: string; // hex color
  isPreset: boolean; // preset tags cannot be deleted
}

// localStorage key for tag definitions
const TAG_DEFS_KEY = "flux-rag-tag-defs";
// localStorage key for session→tag assignments (Record<sessionId, tagId[]>)
const TAG_ASSIGNMENTS_KEY = "flux-rag-session-tags";

// ---------------------------------------------------------------------------
// Predefined tags
// ---------------------------------------------------------------------------

export const PRESET_TAGS: SessionTag[] = [
  { id: "preset-important", name: "중요", color: "#ef4444", isPreset: true },
  { id: "preset-work",      name: "업무", color: "#3b82f6", isPreset: true },
  { id: "preset-study",     name: "학습", color: "#10b981", isPreset: true },
  { id: "preset-ref",       name: "참고", color: "#f59e0b", isPreset: true },
  { id: "preset-personal",  name: "개인", color: "#8b5cf6", isPreset: true },
  { id: "preset-temp",      name: "임시", color: "#6b7280", isPreset: true },
];

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function readTagDefs(): SessionTag[] {
  try {
    const raw = localStorage.getItem(TAG_DEFS_KEY);
    if (!raw) return [];
    return JSON.parse(raw) as SessionTag[];
  } catch {
    return [];
  }
}

function writeTagDefs(defs: SessionTag[]): void {
  localStorage.setItem(TAG_DEFS_KEY, JSON.stringify(defs));
}

function readAssignments(): Record<string, string[]> {
  try {
    const raw = localStorage.getItem(TAG_ASSIGNMENTS_KEY);
    if (!raw) return {};
    return JSON.parse(raw) as Record<string, string[]>;
  } catch {
    return {};
  }
}

function writeAssignments(assignments: Record<string, string[]>): void {
  localStorage.setItem(TAG_ASSIGNMENTS_KEY, JSON.stringify(assignments));
}

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

export function useSessionTags() {
  // Custom tag definitions (in addition to PRESET_TAGS)
  const [customTags, setCustomTags] = useState<SessionTag[]>(() => readTagDefs());

  // All tags: preset + custom
  const allTags: SessionTag[] = [...PRESET_TAGS, ...customTags];

  // session→tagId[] assignments
  const [assignments, setAssignments] = useState<Record<string, string[]>>(
    () => readAssignments()
  );

  // Persist custom tags whenever they change
  useEffect(() => {
    writeTagDefs(customTags);
  }, [customTags]);

  // Persist assignments whenever they change
  useEffect(() => {
    writeAssignments(assignments);
  }, [assignments]);

  // ------------------------------------------------------------------
  // Tag CRUD
  // ------------------------------------------------------------------

  const addTag = useCallback((name: string, color: string) => {
    const id = `custom-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`;
    const newTag: SessionTag = { id, name, color, isPreset: false };
    setCustomTags((prev) => [...prev, newTag]);
    return newTag;
  }, []);

  const deleteTag = useCallback((tagId: string) => {
    // Don't allow deleting preset tags
    setCustomTags((prev) => prev.filter((t) => t.id !== tagId));
    // Also remove from all assignments
    setAssignments((prev) => {
      const next: Record<string, string[]> = {};
      for (const [sid, tags] of Object.entries(prev)) {
        const filtered = tags.filter((id) => id !== tagId);
        if (filtered.length > 0) next[sid] = filtered;
      }
      return next;
    });
  }, []);

  // ------------------------------------------------------------------
  // Assignment operations
  // ------------------------------------------------------------------

  /** Get tag ids assigned to a session */
  const getSessionTagIds = useCallback(
    (sessionId: string): string[] => assignments[sessionId] ?? [],
    [assignments]
  );

  /** Get full tag objects assigned to a session */
  const getSessionTags = useCallback(
    (sessionId: string): SessionTag[] => {
      const ids = assignments[sessionId] ?? [];
      return ids
        .map((id) => allTags.find((t) => t.id === id))
        .filter((t): t is SessionTag => t !== undefined);
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [assignments, customTags]
  );

  /** Toggle a tag on/off for a session */
  const toggleTag = useCallback((sessionId: string, tagId: string) => {
    setAssignments((prev) => {
      const current = prev[sessionId] ?? [];
      const hasTag = current.includes(tagId);
      const next = hasTag
        ? current.filter((id) => id !== tagId)
        : [...current, tagId];
      if (next.length === 0) {
        const { [sessionId]: _removed, ...rest } = prev;
        return rest;
      }
      return { ...prev, [sessionId]: next };
    });
  }, []);

  /** Check if a session has a specific tag */
  const hasTag = useCallback(
    (sessionId: string, tagId: string): boolean =>
      (assignments[sessionId] ?? []).includes(tagId),
    [assignments]
  );

  return {
    allTags,
    customTags,
    addTag,
    deleteTag,
    assignments,
    getSessionTagIds,
    getSessionTags,
    toggleTag,
    hasTag,
  };
}
