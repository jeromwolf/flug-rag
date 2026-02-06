"""Unit tests for Agent module: Router, Planner, Memory."""

import json

import pytest

from agent.memory import ConversationMemory
from agent.planner import ExecutionPlan, PlanStep, TaskPlanner
from agent.router import QueryCategory, QueryRouter, RoutingResult


# ---------------------------------------------------------------------------
# TestQueryRouter
# ---------------------------------------------------------------------------


class TestQueryRouter:
    """Tests for QueryRouter fallback and parsing logic."""

    def _make_router(self) -> QueryRouter:
        """Create a router without triggering LLM/PromptManager init."""
        router = object.__new__(QueryRouter)
        router.llm = None
        router.prompt_manager = None
        return router

    def test_fallback_document_search(self):
        router = self._make_router()
        result = router._fallback_route("가스 안전 규정에 대해 알려주세요")
        assert result.category == QueryCategory.DOCUMENT_SEARCH
        assert result.confidence == 0.6

    def test_fallback_chitchat(self):
        router = self._make_router()
        result = router._fallback_route("안녕하세요 ㅎㅎ")
        assert result.category == QueryCategory.CHITCHAT
        assert result.confidence == 0.8

    def test_fallback_complex_task(self):
        router = self._make_router()
        result = router._fallback_route("A와 B를 비교 분석하고 정리해 주세요")
        assert result.category == QueryCategory.COMPLEX_TASK
        assert result.confidence == 0.5

    def test_fallback_default(self):
        router = self._make_router()
        result = router._fallback_route("hello world")
        assert result.category == QueryCategory.GENERAL_QUERY
        assert result.confidence == 0.4

    def test_parse_valid_json(self):
        router = self._make_router()
        raw = json.dumps({
            "category": "document_search",
            "confidence": 0.95,
            "reasoning": "Mentions 규정",
        })
        result = router._parse_routing(raw)
        assert result.category == QueryCategory.DOCUMENT_SEARCH
        assert result.confidence == 0.95
        assert result.reasoning == "Mentions 규정"

    def test_parse_json_in_code_block(self):
        router = self._make_router()
        raw = '```json\n{"category": "chitchat", "confidence": 0.9, "reasoning": "greeting"}\n```'
        result = router._parse_routing(raw)
        assert result.category == QueryCategory.CHITCHAT
        assert result.confidence == 0.9

    def test_parse_invalid_json(self):
        router = self._make_router()
        result = router._parse_routing("this is not json at all - 규정 관련")
        # Falls back to keyword heuristic on the raw text
        assert result.category == QueryCategory.DOCUMENT_SEARCH

    def test_parse_invalid_json_no_keywords(self):
        router = self._make_router()
        result = router._parse_routing("totally random garbage xyz")
        assert result.category == QueryCategory.GENERAL_QUERY


# ---------------------------------------------------------------------------
# TestConversationMemory
# ---------------------------------------------------------------------------


class TestConversationMemory:
    """Tests for SQLite-backed ConversationMemory."""

    @pytest.fixture
    def memory(self, tmp_path):
        db_path = str(tmp_path / "test_memory.db")
        return ConversationMemory(db_path=db_path)

    async def test_create_session(self, memory):
        session_id = await memory.create_session(title="Test Session")
        assert session_id
        assert len(session_id) == 36  # UUID length

    async def test_add_and_get_history(self, memory):
        session_id = await memory.create_session(title="Chat")
        await memory.add_message(session_id, "user", "Hello")
        await memory.add_message(session_id, "assistant", "Hi there!")
        await memory.add_message(session_id, "user", "How are you?")

        history = await memory.get_history(session_id)
        assert len(history) == 3
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "Hello"
        assert history[1]["role"] == "assistant"
        assert history[1]["content"] == "Hi there!"
        assert history[2]["role"] == "user"
        assert history[2]["content"] == "How are you?"

    async def test_get_sessions(self, memory):
        await memory.create_session(title="Session A")
        sid_b = await memory.create_session(title="Session B")
        await memory.add_message(sid_b, "user", "msg")

        sessions = await memory.get_sessions()
        assert len(sessions) == 2
        # Most recently updated first
        titles = [s["title"] for s in sessions]
        assert "Session A" in titles
        assert "Session B" in titles

        # Session B has 1 message
        session_b = next(s for s in sessions if s["title"] == "Session B")
        assert session_b["message_count"] == 1

    async def test_get_session_by_id(self, memory):
        session_id = await memory.create_session(title="Find Me", metadata={"key": "value"})
        session = await memory.get_session(session_id)
        assert session is not None
        assert session["title"] == "Find Me"
        assert session["metadata"] == {"key": "value"}

    async def test_get_session_not_found(self, memory):
        session = await memory.get_session("nonexistent-id")
        assert session is None

    async def test_update_title(self, memory):
        session_id = await memory.create_session(title="Old Title")
        await memory.update_session_title(session_id, "New Title")

        session = await memory.get_session(session_id)
        assert session["title"] == "New Title"

    async def test_delete_session(self, memory):
        session_id = await memory.create_session(title="Delete Me")
        await memory.add_message(session_id, "user", "bye")

        await memory.delete_session(session_id)
        session = await memory.get_session(session_id)
        assert session is None

        history = await memory.get_history(session_id)
        assert len(history) == 0

    async def test_history_limit(self, memory):
        session_id = await memory.create_session(title="Limited")
        for i in range(10):
            await memory.add_message(session_id, "user", f"Message {i}")

        history = await memory.get_history(session_id, limit=3)
        assert len(history) == 3
        # Should be the 3 most recent, in chronological order
        assert history[0]["content"] == "Message 7"
        assert history[1]["content"] == "Message 8"
        assert history[2]["content"] == "Message 9"

    async def test_clear_all(self, memory):
        sid1 = await memory.create_session(title="S1")
        sid2 = await memory.create_session(title="S2")
        await memory.add_message(sid1, "user", "msg1")
        await memory.add_message(sid2, "user", "msg2")

        await memory.clear_all()

        sessions = await memory.get_sessions()
        assert len(sessions) == 0

    async def test_message_metadata(self, memory):
        session_id = await memory.create_session()
        await memory.add_message(
            session_id, "assistant", "response",
            metadata={"tokens": 42, "model": "test"},
        )

        history = await memory.get_history(session_id)
        assert len(history) == 1
        assert history[0]["metadata"]["tokens"] == 42
        assert history[0]["metadata"]["model"] == "test"


# ---------------------------------------------------------------------------
# TestTaskPlanner
# ---------------------------------------------------------------------------


class TestTaskPlanner:
    """Tests for TaskPlanner parsing logic."""

    def _make_planner(self) -> TaskPlanner:
        """Create a planner without triggering LLM init."""
        planner = object.__new__(TaskPlanner)
        planner.llm = None
        return planner

    def test_parse_valid_plan(self):
        planner = self._make_planner()
        raw = json.dumps({
            "steps": [
                {"step_id": 1, "action": "search", "description": "검색", "query": "가스 안전", "depends_on": []},
                {"step_id": 2, "action": "search", "description": "검색2", "query": "점검 절차", "depends_on": []},
                {"step_id": 3, "action": "summarize", "description": "종합", "depends_on": [1, 2]},
            ],
            "final_instruction": "결과를 종합하세요.",
        })
        plan = planner._parse_plan("원래 질문", raw)
        assert isinstance(plan, ExecutionPlan)
        assert plan.original_query == "원래 질문"
        assert len(plan.steps) == 3
        assert plan.steps[0].action == "search"
        assert plan.steps[0].query == "가스 안전"
        assert plan.steps[2].depends_on == [1, 2]
        assert plan.final_instruction == "결과를 종합하세요."

    def test_parse_plan_in_code_block(self):
        planner = self._make_planner()
        raw = '```json\n{"steps": [{"step_id": 1, "action": "search", "description": "d", "query": "q"}], "final_instruction": "f"}\n```'
        plan = planner._parse_plan("query", raw)
        assert len(plan.steps) == 1
        assert plan.steps[0].query == "q"

    def test_parse_invalid_json_fallback(self):
        planner = self._make_planner()
        plan = planner._parse_plan("테스트 질문", "invalid json response")
        assert isinstance(plan, ExecutionPlan)
        assert len(plan.steps) == 2
        assert plan.steps[0].action == "search"
        assert plan.steps[0].query == "테스트 질문"
        assert plan.steps[1].action == "generate"
        assert plan.steps[1].depends_on == [1]

    def test_plan_step_defaults(self):
        step = PlanStep(step_id=1, action="search", description="test")
        assert step.query is None
        assert step.depends_on == []
        assert step.result is None
        assert step.status == "pending"

    def test_execution_plan_defaults(self):
        plan = ExecutionPlan(original_query="q", steps=[])
        assert plan.final_instruction == "종합하여 하나의 답변을 생성하세요."
