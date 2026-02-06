"""Pydantic request/response models for API endpoints."""

from pydantic import BaseModel, Field


# --- Chat ---
class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None
    mode: str = "auto"  # "auto", "rag", "direct"
    provider: str | None = None
    model: str | None = None
    filters: dict | None = None


class ChatResponse(BaseModel):
    message_id: str
    content: str
    sources: list[dict] = []
    confidence: float = 0.0
    confidence_level: str = "low"
    safety_warning: str | None = None
    session_id: str
    metadata: dict = {}


class StreamEvent(BaseModel):
    event: str
    data: dict = {}


# --- Documents ---
class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    chunk_count: int
    status: str


class DocumentListItem(BaseModel):
    id: str
    filename: str
    file_type: str
    chunk_count: int
    uploaded_at: str
    metadata: dict = {}


class DocumentListResponse(BaseModel):
    documents: list[DocumentListItem]
    total: int


# --- Sessions ---
class SessionCreate(BaseModel):
    title: str = ""


class SessionResponse(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int = 0


class SessionListResponse(BaseModel):
    sessions: list[SessionResponse]


class MessageResponse(BaseModel):
    id: str
    role: str
    content: str
    metadata: dict = {}
    created_at: str


# --- Admin ---
class LLMProviderInfo(BaseModel):
    name: str
    is_default: bool


class PromptUpdateRequest(BaseModel):
    name: str
    content: str


class SystemInfoResponse(BaseModel):
    app_name: str
    version: str
    default_provider: str
    document_count: int
    session_count: int


# --- Feedback ---
class FeedbackRequest(BaseModel):
    message_id: str
    session_id: str
    rating: int = Field(ge=-1, le=1)  # -1=bad, 0=neutral, 1=good
    comment: str = ""
    corrected_answer: str | None = None


class FeedbackResponse(BaseModel):
    id: str
    status: str


# --- MCP ---
class MCPCallRequest(BaseModel):
    tool_name: str
    arguments: dict = {}


# --- Workflow ---
class WorkflowRunRequest(BaseModel):
    workflow_id: str | None = None
    preset: str | None = None
    input_data: dict
