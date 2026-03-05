import type { TranslationKey } from "./ko";

const en: Record<TranslationKey, string> = {
  // ── Sidebar ──────────────────────────────────────────────────────────────
  "sidebar.newChat": "New Chat",
  "sidebar.searchPlaceholder": "Search chats...",
  "sidebar.chatList": "Chat List",
  "sidebar.savedAnswers": "Saved Answers",
  "sidebar.noSavedAnswers": "No saved answers",
  "sidebar.noHistory": "No chat history",
  "sidebar.noSearchResults": "No results found",
  "sidebar.noTaggedChats": "No chats with this tag",
  "sidebar.filter": "Filter:",
  "sidebar.tagFilter": "Tag Filter",
  "sidebar.tagManage": "Manage Tags",
  "sidebar.allTags": "All",

  // ── Chat ─────────────────────────────────────────────────────────────────
  "chat.inputPlaceholder": "Ask anything...",
  "chat.inputPlaceholderDesktop": "Ask anything... (Shift+Enter: new line)",
  "chat.send": "Send",
  "chat.stop": "Stop",
  "chat.regenerate": "Regenerate",
  "chat.copied": "Copied!",
  "chat.stopGeneration": "Stop generating",
  "chat.sendMessage": "Send message",
  "chat.recording": "Recording...",
  "chat.voiceInput": "Voice input",
  "chat.stopRecording": "Stop recording",
  "chat.attachFile": "Attach file",
  "chat.promptTemplate": "Prompt template",
  "chat.keyboardHint": "Ctrl+N new chat · Esc stop",

  // ── TopBar ───────────────────────────────────────────────────────────────
  "topbar.modelSelect": "Select model",
  "topbar.darkMode": "Dark mode",
  "topbar.lightMode": "Light mode",
  "topbar.settings": "AI Settings",
  "topbar.userSettings": "User Settings",
  "topbar.print": "Print chat",
  "topbar.share": "Share chat",
  "topbar.export": "Export chat",
  "topbar.copy": "Copy chat",
  "topbar.summarize": "Summarize chat",
  "topbar.compareMode": "Model comparison mode",
  "topbar.searchChat": "Search in chat (Ctrl+F)",
  "topbar.sidebar": "Toggle sidebar",
  "topbar.language": "Language",

  // ── Settings dialog ──────────────────────────────────────────────────────
  "settings.title": "User Settings",
  "settings.profile": "Profile",
  "settings.notifications": "Notifications",
  "settings.theme": "Theme",
  "settings.fontSize": "Font Size",
  "settings.language": "Language",
  "settings.save": "Save",
  "settings.close": "Close",
  "settings.changePassword": "Change Password",
  "settings.themeLight": "Light",
  "settings.themeDark": "Dark",
  "settings.themeSystem": "System",
  "settings.fontSmall": "Small",
  "settings.fontMedium": "Medium",
  "settings.fontLarge": "Large",
  "settings.browserNotifications": "Browser notifications",
  "settings.korean": "한국어",
  "settings.english": "English",

  // ── Auth ─────────────────────────────────────────────────────────────────
  "auth.login": "Login",
  "auth.logout": "Logout",
  "auth.changePassword": "Change Password",
  "auth.username": "Username",
  "auth.password": "Password",

  // ── Documents ────────────────────────────────────────────────────────────
  "documents.title": "Document Management",
  "documents.upload": "Upload",
  "documents.delete": "Delete",
  "documents.filter": "Filter",
  "documents.search": "Search",

  // ── Common ───────────────────────────────────────────────────────────────
  "common.confirm": "Confirm",
  "common.cancel": "Cancel",
  "common.close": "Close",
  "common.save": "Save",
  "common.delete": "Delete",
  "common.edit": "Edit",
  "common.search": "Search",
  "common.add": "Add",
  "common.loading": "Loading...",
  "common.error": "An error occurred",
};

export default en;
