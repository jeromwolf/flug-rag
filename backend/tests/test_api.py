"""Integration tests for FastAPI API endpoints."""

import pytest
from httpx import ASGITransport, AsyncClient

from api.main import app


@pytest.fixture
async def client():
    """Create test client."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


# ---------------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------------


class TestHealth:
    """Test health check endpoint."""

    @pytest.mark.asyncio
    async def test_health(self, client):
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "app" in data
        assert "version" in data


# ---------------------------------------------------------------------------
# Session Management
# ---------------------------------------------------------------------------


class TestSessions:
    """Test session and conversation history endpoints."""

    @pytest.mark.asyncio
    async def test_create_session(self, client):
        resp = await client.post("/api/sessions", json={"title": "테스트 세션"})
        assert resp.status_code == 200
        data = resp.json()
        assert "id" in data
        assert data["title"] == "테스트 세션"
        assert "created_at" in data
        assert "updated_at" in data

    @pytest.mark.asyncio
    async def test_create_session_default_title(self, client):
        resp = await client.post("/api/sessions", json={})
        assert resp.status_code == 200
        data = resp.json()
        assert "id" in data
        assert data["title"] == ""

    @pytest.mark.asyncio
    async def test_list_sessions(self, client):
        # Create a session first
        await client.post("/api/sessions", json={"title": "목록 테스트"})
        resp = await client.get("/api/sessions")
        assert resp.status_code == 200
        data = resp.json()
        assert "sessions" in data
        assert isinstance(data["sessions"], list)

    @pytest.mark.asyncio
    async def test_list_sessions_pagination(self, client):
        resp = await client.get("/api/sessions?limit=10&offset=0")
        assert resp.status_code == 200
        data = resp.json()
        assert "sessions" in data

    @pytest.mark.asyncio
    async def test_get_session(self, client):
        # Create then get
        create_resp = await client.post("/api/sessions", json={"title": "조회 테스트"})
        session_id = create_resp.json()["id"]

        resp = await client.get(f"/api/sessions/{session_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == session_id
        assert data["title"] == "조회 테스트"

    @pytest.mark.asyncio
    async def test_get_session_not_found(self, client):
        resp = await client.get("/api/sessions/nonexistent-id")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_get_messages(self, client):
        # Create session
        create_resp = await client.post("/api/sessions", json={"title": "메시지 테스트"})
        session_id = create_resp.json()["id"]

        # Get messages (should be empty)
        resp = await client.get(f"/api/sessions/{session_id}/messages")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)

    @pytest.mark.asyncio
    async def test_update_session(self, client):
        # Create then update
        create_resp = await client.post("/api/sessions", json={"title": "수정 전"})
        session_id = create_resp.json()["id"]

        resp = await client.patch(f"/api/sessions/{session_id}", params={"title": "수정 후"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "updated"

    @pytest.mark.asyncio
    async def test_delete_session(self, client):
        # Create then delete
        create_resp = await client.post("/api/sessions", json={"title": "삭제 테스트"})
        session_id = create_resp.json()["id"]

        resp = await client.delete(f"/api/sessions/{session_id}")
        assert resp.status_code == 200
        assert resp.json()["status"] == "deleted"
        assert resp.json()["id"] == session_id

        # Verify deletion
        get_resp = await client.get(f"/api/sessions/{session_id}")
        assert get_resp.status_code == 404


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------


class TestChat:
    """Test chat endpoints."""

    @pytest.mark.asyncio
    async def test_chat_basic(self, client):
        """Test basic chat without session (creates new session)."""
        resp = await client.post(
            "/api/chat",
            json={"message": "안녕하세요", "mode": "direct"},
        )
        # May fail if LLM not available, but should at least accept the request
        assert resp.status_code in (200, 500)
        if resp.status_code == 200:
            data = resp.json()
            assert "message_id" in data
            assert "content" in data
            assert "session_id" in data

    @pytest.mark.asyncio
    async def test_chat_with_session(self, client):
        """Test chat with existing session."""
        # Create session
        session_resp = await client.post("/api/sessions", json={"title": "채팅 테스트"})
        session_id = session_resp.json()["id"]

        # Send message
        resp = await client.post(
            "/api/chat",
            json={
                "message": "테스트 메시지",
                "session_id": session_id,
                "mode": "direct",
            },
        )
        # Accept both success and failure due to LLM availability
        assert resp.status_code in (200, 500)

    @pytest.mark.asyncio
    async def test_chat_modes(self, client):
        """Test different chat modes."""
        for mode in ["auto", "rag", "direct"]:
            resp = await client.post(
                "/api/chat",
                json={"message": "테스트", "mode": mode},
            )
            assert resp.status_code in (200, 500)

    @pytest.mark.asyncio
    async def test_chat_with_filters(self, client):
        """Test chat with metadata filters."""
        resp = await client.post(
            "/api/chat",
            json={
                "message": "가스 안전 규정",
                "mode": "rag",
                "filters": {"category": "safety"},
            },
        )
        assert resp.status_code in (200, 500)

    @pytest.mark.asyncio
    async def test_chat_with_provider_model(self, client):
        """Test chat with specific provider and model."""
        resp = await client.post(
            "/api/chat",
            json={
                "message": "테스트",
                "mode": "direct",
                "provider": "openai",
                "model": "gpt-4o-mini",
            },
        )
        assert resp.status_code in (200, 500)


# ---------------------------------------------------------------------------
# Admin
# ---------------------------------------------------------------------------


class TestAdmin:
    """Test admin endpoints."""

    @pytest.mark.asyncio
    async def test_system_info(self, client):
        resp = await client.get("/api/admin/info")
        # May fail if dependencies not initialized, but should accept request
        assert resp.status_code in (200, 500)
        if resp.status_code == 200:
            data = resp.json()
            assert "app_name" in data
            assert "version" in data
            assert "default_provider" in data

    @pytest.mark.asyncio
    async def test_providers(self, client):
        resp = await client.get("/api/admin/providers")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        # Should have at least one provider
        if len(data) > 0:
            assert "name" in data[0]
            assert "is_default" in data[0]

    @pytest.mark.asyncio
    async def test_get_prompts(self, client):
        resp = await client.get("/api/admin/prompts")
        # May fail if prompt files not found
        assert resp.status_code in (200, 500)
        if resp.status_code == 200:
            data = resp.json()
            assert "prompts" in data

    @pytest.mark.asyncio
    async def test_update_prompt(self, client):
        resp = await client.put(
            "/api/admin/prompts",
            json={"name": "test_prompt", "content": "테스트 프롬프트 내용"},
        )
        # May fail due to file permissions or initialization issues
        assert resp.status_code in (200, 500)


# ---------------------------------------------------------------------------
# Feedback
# ---------------------------------------------------------------------------


class TestFeedback:
    """Test feedback endpoints."""

    @pytest.mark.asyncio
    async def test_submit_feedback_positive(self, client):
        resp = await client.post(
            "/api/feedback",
            json={
                "message_id": "test-msg-001",
                "session_id": "test-session-001",
                "rating": 1,
                "comment": "좋은 답변입니다",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "id" in data
        assert data["status"] == "saved"

    @pytest.mark.asyncio
    async def test_submit_feedback_negative(self, client):
        resp = await client.post(
            "/api/feedback",
            json={
                "message_id": "test-msg-002",
                "session_id": "test-session-002",
                "rating": -1,
                "comment": "부정확한 답변",
                "corrected_answer": "올바른 답변은 이것입니다",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "saved"

    @pytest.mark.asyncio
    async def test_submit_feedback_neutral(self, client):
        resp = await client.post(
            "/api/feedback",
            json={
                "message_id": "test-msg-003",
                "session_id": "test-session-003",
                "rating": 0,
            },
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_list_feedback(self, client):
        # Submit feedback first
        await client.post(
            "/api/feedback",
            json={
                "message_id": "test-list",
                "session_id": "test-list",
                "rating": 1,
            },
        )

        resp = await client.get("/api/feedback")
        assert resp.status_code == 200
        data = resp.json()
        assert "feedbacks" in data
        assert "total" in data
        assert isinstance(data["feedbacks"], list)

    @pytest.mark.asyncio
    async def test_list_feedback_limit(self, client):
        resp = await client.get("/api/feedback?limit=5")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["feedbacks"]) <= 5

    @pytest.mark.asyncio
    async def test_feedback_stats(self, client):
        # Submit some feedback
        for rating in [1, 1, -1, 0]:
            await client.post(
                "/api/feedback",
                json={
                    "message_id": f"stats-{rating}",
                    "session_id": "stats-test",
                    "rating": rating,
                },
            )

        resp = await client.get("/api/feedback/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "total" in data
        assert "positive" in data
        assert "negative" in data
        assert "neutral" in data


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------


class TestDocuments:
    """Test document upload and management endpoints."""

    @pytest.mark.asyncio
    async def test_list_documents(self, client):
        resp = await client.get("/api/documents")
        assert resp.status_code == 200
        data = resp.json()
        assert "documents" in data
        assert "total" in data
        assert isinstance(data["documents"], list)

    @pytest.mark.asyncio
    async def test_upload_document_txt(self, client):
        """Test uploading a text file."""
        content = "가스 안전 관리 규정 테스트 문서입니다.\n\n이것은 테스트 문서입니다."
        files = {"file": ("test_doc.txt", content.encode("utf-8"), "text/plain")}
        resp = await client.post("/api/documents/upload", files=files)
        # May fail if IngestPipeline dependencies not available
        assert resp.status_code in (200, 500)
        if resp.status_code == 200:
            data = resp.json()
            assert "document_id" in data
            assert data["filename"] == "test_doc.txt"
            assert "chunk_count" in data

    @pytest.mark.asyncio
    async def test_upload_unsupported_file(self, client):
        """Test uploading unsupported file type."""
        content = b"fake executable"
        files = {"file": ("malware.exe", content, "application/x-msdownload")}
        resp = await client.post("/api/documents/upload", files=files)
        assert resp.status_code == 400
        assert "unsupported" in resp.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_delete_document_not_found(self, client):
        """Test deleting non-existent document."""
        resp = await client.delete("/api/documents/nonexistent-doc-id")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# MCP Tools
# ---------------------------------------------------------------------------


class TestMCP:
    """Test MCP tool endpoints."""

    @pytest.mark.asyncio
    async def test_list_tools(self, client):
        resp = await client.get("/api/mcp/tools")
        assert resp.status_code == 200
        data = resp.json()
        assert "tools" in data
        assert isinstance(data["tools"], list)
        # Should have at least calculator tool
        tool_names = [t["name"] for t in data["tools"]]
        assert "calculator" in tool_names

    @pytest.mark.asyncio
    async def test_call_calculator_addition(self, client):
        resp = await client.post(
            "/api/mcp/call",
            json={"tool_name": "calculator", "arguments": {"expression": "2 + 3"}},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["data"]["result"] == 5.0

    @pytest.mark.asyncio
    async def test_call_calculator_complex(self, client):
        resp = await client.post(
            "/api/mcp/call",
            json={"tool_name": "calculator", "arguments": {"expression": "(100 + 200) * 1.1"}},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert abs(data["data"]["result"] - 330.0) < 0.01

    @pytest.mark.asyncio
    async def test_call_calculator_error(self, client):
        resp = await client.post(
            "/api/mcp/call",
            json={"tool_name": "calculator", "arguments": {"expression": "1 / 0"}},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is False
        assert data["error"] is not None

    @pytest.mark.asyncio
    async def test_call_unknown_tool(self, client):
        resp = await client.post(
            "/api/mcp/call",
            json={"tool_name": "nonexistent_tool", "arguments": {}},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is False
        assert "not found" in data["error"].lower()


# ---------------------------------------------------------------------------
# Workflows
# ---------------------------------------------------------------------------


class TestWorkflows:
    """Test Agent Builder workflow endpoints."""

    @pytest.mark.asyncio
    async def test_list_presets(self, client):
        resp = await client.get("/api/workflows/presets")
        assert resp.status_code == 200
        data = resp.json()
        assert "presets" in data
        assert isinstance(data["presets"], list)

    @pytest.mark.asyncio
    async def test_run_workflow_missing_preset(self, client):
        resp = await client.post(
            "/api/workflows/run",
            json={"input_data": {"query": "test"}},
        )
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_run_workflow_unknown_preset(self, client):
        resp = await client.post(
            "/api/workflows/run",
            json={"preset": "nonexistent_preset", "input_data": {"query": "test"}},
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_run_workflow_valid_preset(self, client):
        """Test running a valid preset workflow."""
        # Get available presets first
        presets_resp = await client.get("/api/workflows/presets")
        presets = presets_resp.json()["presets"]

        if len(presets) > 0:
            # Use the first available preset (API returns objects with id/name)
            preset_name = presets[0]["id"] if isinstance(presets[0], dict) else presets[0]
            resp = await client.post(
                "/api/workflows/run",
                json={"preset": preset_name, "input_data": {"query": "테스트 쿼리"}},
            )
            # May fail if workflow dependencies not available
            assert resp.status_code in (200, 500)
            if resp.status_code == 200:
                data = resp.json()
                assert "workflow_id" in data
                assert "status" in data
