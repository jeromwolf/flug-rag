"""Tests for multi-agent collaboration, chaining, new MCP tools, and monitoring."""

import asyncio
import json
import math

import pytest

from agent.collaboration.agent_pool import AgentPool
from agent.collaboration.coordinator import (
    AgentCoordinator,
    AgentStatus,
    TaskPriority,
    TaskStatus,
)
from agent.collaboration.protocols import (
    AgentMessage,
    MessageBus,
    MessageType,
    SharedContext,
)
from agent.collaboration.strategies import (
    AgentHandle,
    DebateStrategy,
    ParallelStrategy,
    SequentialStrategy,
    StrategyResult,
    VotingStrategy,
)
from agent.chaining.chain import AgentChain, ChainStep
from agent.chaining.templates import (
    CHAIN_TEMPLATES,
    create_analysis_chain,
    create_qa_chain,
    create_research_chain,
    create_translation_chain,
    list_chain_templates,
)
from agent.mcp.tools.base import ToolResult
from agent.mcp.tools.calculator_tool import CalculatorTool
from agent.mcp.tools.data_analyzer_tool import DataAnalyzerTool
from agent.monitor.dashboard_data import DashboardDataProvider
from agent.monitor.tracker import ExecutionTracker, ExecutionType, ExecutionState


# ===========================================================================
# Mock LLM for tool testing (avoids real API calls)
# ===========================================================================


class MockLLM:
    """Mock LLM that returns predictable responses."""

    def __init__(self, response_text: str = "Mock response"):
        self.response_text = response_text
        self.call_count = 0

    async def generate(self, prompt: str, system: str | None = None, temperature: float = 0.7, **kwargs):
        self.call_count += 1

        class MockResponse:
            def __init__(self, content):
                self.content = content
        return MockResponse(self.response_text)


# ===========================================================================
# Summarizer Tool Tests
# ===========================================================================


class TestSummarizerTool:
    """Tests for the SummarizerTool."""

    def setup_method(self):
        from agent.mcp.tools.summarizer_tool import SummarizerTool
        self.mock_llm = MockLLM("이것은 요약 결과입니다.")
        self.tool = SummarizerTool(llm=self.mock_llm)

    def test_definition(self):
        defn = self.tool.get_definition()
        assert defn.name == "document_summarizer"
        assert defn.category == "nlp"
        assert len(defn.parameters) == 4

    async def test_abstractive_summary(self):
        result = await self.tool.execute(text="긴 문서 내용...", mode="abstractive")
        assert result.success
        assert "summary" in result.data
        assert result.data["mode"] == "abstractive"
        assert self.mock_llm.call_count == 1

    async def test_extractive_summary(self):
        result = await self.tool.execute(text="핵심 문장이 포함된 문서", mode="extractive")
        assert result.success
        assert result.data["mode"] == "extractive"

    async def test_empty_text_error(self):
        result = await self.tool.execute(text="")
        assert not result.success
        assert "required" in result.error

    async def test_max_length(self):
        self.mock_llm.response_text = "A" * 1000
        result = await self.tool.execute(text="test", max_length=100)
        assert result.success
        assert len(result.data["summary"]) <= 100

    async def test_summarize_text_convenience(self):
        text = await self.tool.summarize_text("테스트 텍스트", max_length=500)
        assert isinstance(text, str)


# ===========================================================================
# Translator Tool Tests
# ===========================================================================


class TestTranslatorTool:
    """Tests for the TranslatorTool."""

    def setup_method(self):
        from agent.mcp.tools.translator_tool import TranslatorTool
        self.mock_llm = MockLLM("This is translated text.")
        self.tool = TranslatorTool(llm=self.mock_llm)

    def test_definition(self):
        defn = self.tool.get_definition()
        assert defn.name == "translator"
        assert defn.category == "nlp"
        assert len(defn.parameters) == 3

    async def test_translate_ko_en(self):
        result = await self.tool.execute(text="안녕하세요", source_lang="ko", target_lang="en")
        assert result.success
        assert "translated_text" in result.data
        assert result.data["source_lang"] == "ko"
        assert result.data["target_lang"] == "en"

    async def test_same_language(self):
        result = await self.tool.execute(text="Hello", source_lang="en", target_lang="en")
        assert result.success
        assert result.data["translated_text"] == "Hello"

    async def test_unsupported_pair(self):
        result = await self.tool.execute(text="Hello", source_lang="en", target_lang="ja")
        assert not result.success
        assert "Unsupported" in result.error

    async def test_missing_params(self):
        result = await self.tool.execute(text="Hello", source_lang="", target_lang="en")
        assert not result.success

    async def test_empty_text(self):
        result = await self.tool.execute(text="", source_lang="ko", target_lang="en")
        assert not result.success


# ===========================================================================
# Report Generator Tool Tests
# ===========================================================================


class TestReportGeneratorTool:
    """Tests for the ReportGeneratorTool."""

    def setup_method(self):
        from agent.mcp.tools.report_generator_tool import ReportGeneratorTool
        self.mock_llm = MockLLM("# 안전 점검 보고서\n\n## 점검 개요\n내용...")
        self.tool = ReportGeneratorTool(llm=self.mock_llm)

    def test_definition(self):
        defn = self.tool.get_definition()
        assert defn.name == "report_generator"
        assert defn.category == "document"
        assert any(p.name == "template_name" for p in defn.parameters)

    async def test_safety_inspection_report(self):
        result = await self.tool.execute(
            template_name="safety_inspection",
            data={"facility_name": "A공장", "inspector": "홍길동", "date": "2025-01-01", "findings": "이상 없음"},
            output_format="markdown",
        )
        assert result.success
        assert "report" in result.data
        assert result.data["template"] == "safety_inspection"

    async def test_monthly_summary_report(self):
        result = await self.tool.execute(
            template_name="monthly_summary",
            data={"department": "안전팀", "period": "2025-01", "activities": ["점검 10건"]},
        )
        assert result.success

    async def test_json_output(self):
        self.mock_llm.response_text = '```json\n{"title": "보고서", "content": "내용"}\n```'
        result = await self.tool.execute(
            template_name="incident_report",
            data={"incident_type": "누출", "location": "배관실", "date": "2025-01-15", "description": "미세 누출"},
            output_format="json",
        )
        assert result.success

    async def test_unknown_template(self):
        result = await self.tool.execute(template_name="nonexistent", data={"key": "val"})
        assert not result.success
        assert "Unknown template" in result.error

    async def test_empty_data(self):
        result = await self.tool.execute(template_name="safety_inspection", data={})
        assert not result.success


# ===========================================================================
# Email Composer Tool Tests
# ===========================================================================


class TestEmailComposerTool:
    """Tests for the EmailComposerTool."""

    def setup_method(self):
        from agent.mcp.tools.email_composer_tool import EmailComposerTool
        self.mock_llm = MockLLM("수신: 홍길동 팀장님\n\n안녕하세요.\n\n감사합니다.")
        self.tool = EmailComposerTool(llm=self.mock_llm)

    def test_definition(self):
        defn = self.tool.get_definition()
        assert defn.name == "email_composer"
        assert defn.category == "document"

    async def test_compose_formal_email(self):
        result = await self.tool.execute(
            subject="안전 점검 일정 안내",
            recipients=["홍길동 팀장"],
            body_context="3월 점검 일정 안내",
            tone="formal",
        )
        assert result.success
        assert "email" in result.data
        assert result.data["tone"] == "formal"

    async def test_compose_with_cc(self):
        result = await self.tool.execute(
            subject="회의 안내",
            recipients=["김철수 과장"],
            body_context="주간 회의",
            cc_list=["이영희 대리"],
        )
        assert result.success
        assert result.data["cc_list"] == ["이영희 대리"]

    async def test_missing_subject(self):
        result = await self.tool.execute(subject="", recipients=["A"], body_context="내용")
        assert not result.success

    async def test_missing_body(self):
        result = await self.tool.execute(subject="제목", body_context="")
        assert not result.success


# ===========================================================================
# Data Analyzer Tool Tests
# ===========================================================================


class TestDataAnalyzerTool:
    """Tests for the DataAnalyzerTool."""

    def setup_method(self):
        self.tool = DataAnalyzerTool()

    def test_definition(self):
        defn = self.tool.get_definition()
        assert defn.name == "data_analyzer"
        assert defn.category == "analytics"

    async def test_basic_statistics(self):
        result = await self.tool.execute(data=[1, 2, 3, 4, 5], analysis_type="statistics")
        assert result.success
        stats = result.data
        assert stats["count"] == 5
        assert stats["mean"] == 3.0
        assert stats["median"] == 3.0
        assert stats["min"] == 1.0
        assert stats["max"] == 5.0

    async def test_statistics_dict(self):
        result = await self.tool.execute(data={"a": 10, "b": 20, "c": 30}, analysis_type="statistics")
        assert result.success
        assert result.data["count"] == 3
        assert result.data["mean"] == 20.0

    async def test_empty_data_error(self):
        result = await self.tool.execute(data=None)
        assert not result.success

    async def test_bar_chart_data(self):
        result = await self.tool.execute(
            data={"A": 10, "B": 20, "C": 30},
            analysis_type="chart_data",
            chart_type="bar",
        )
        assert result.success
        assert result.data["chart_type"] == "bar"
        assert result.data["labels"] == ["A", "B", "C"]
        assert result.data["datasets"][0]["data"] == [10.0, 20.0, 30.0]

    async def test_pie_chart_data(self):
        result = await self.tool.execute(
            data={"사과": 40, "바나나": 30, "포도": 30},
            analysis_type="chart_data",
            chart_type="pie",
        )
        assert result.success
        assert result.data["chart_type"] == "pie"
        assert len(result.data["datasets"][0]["percentages"]) == 3

    async def test_scatter_chart_data(self):
        result = await self.tool.execute(
            data=[{"x": 1, "y": 2}, {"x": 3, "y": 4}],
            analysis_type="chart_data",
            chart_type="scatter",
        )
        assert result.success
        points = result.data["datasets"][0]["data"]
        assert len(points) == 2
        assert points[0] == {"x": 1.0, "y": 2.0}

    async def test_csv_summary(self):
        csv_data = [
            {"name": "A", "value": "10", "category": "X"},
            {"name": "B", "value": "20", "category": "Y"},
            {"name": "C", "value": "30", "category": "X"},
        ]
        result = await self.tool.execute(data=csv_data, analysis_type="csv_summary")
        assert result.success
        assert result.data["row_count"] == 3
        assert "column_stats" in result.data

    def test_compute_statistics_no_values(self):
        stats = self.tool.compute_statistics(["not", "numbers"])
        assert stats["count"] == 0

    def test_percentiles(self):
        stats = self.tool.compute_statistics(list(range(1, 101)))
        # For 100 values (1-100), verify percentiles are calculated correctly
        assert stats["percentiles"]["50"] == 50.5
        assert stats["percentiles"]["25"] == 25.75
        assert stats["percentiles"]["75"] == 75.25
        assert abs(stats["percentiles"]["90"] - 90.1) < 0.01


# ===========================================================================
# SharedContext Tests
# ===========================================================================


class TestSharedContext:
    def test_set_and_get(self):
        ctx = SharedContext()
        ctx.set("key1", "value1")
        assert ctx.get("key1") == "value1"
        assert ctx.get("missing") is None
        assert ctx.get("missing", "default") == "default"

    def test_has_and_keys(self):
        ctx = SharedContext()
        ctx.set("a", 1)
        ctx.set("b", 2)
        assert ctx.has("a")
        assert not ctx.has("c")
        assert sorted(ctx.keys()) == ["a", "b"]

    def test_delete_and_clear(self):
        ctx = SharedContext()
        ctx.set("x", 1)
        ctx.set("y", 2)
        ctx.delete("x")
        assert not ctx.has("x")
        ctx.clear()
        assert ctx.keys() == []

    def test_subscribe_notify(self):
        ctx = SharedContext()
        notifications = []
        ctx.subscribe("key", lambda k, v: notifications.append((k, v)))
        ctx.set("key", "val1")
        ctx.set("key", "val2")
        assert len(notifications) == 2
        assert notifications[0] == ("key", "val1")

    def test_history(self):
        ctx = SharedContext()
        ctx.set("a", 1, agent_id="agent1")
        history = ctx.get_history()
        assert len(history) == 1
        assert history[0]["key"] == "a"
        assert history[0]["agent_id"] == "agent1"


# ===========================================================================
# MessageBus Tests
# ===========================================================================


class TestMessageBus:
    def test_send_and_receive(self):
        bus = MessageBus()
        msg = AgentMessage(sender="a1", receiver="a2", content="hello")
        bus.send(msg)
        messages = bus.receive("a2")
        assert len(messages) == 1
        assert messages[0].content == "hello"
        # Should be empty after receive
        assert bus.receive("a2") == []

    def test_broadcast(self):
        bus = MessageBus()
        bus._queues["a1"] = []
        bus._queues["a2"] = []
        msg = AgentMessage(sender="system", receiver="*", content="broadcast", msg_type=MessageType.BROADCAST)
        bus.send(msg)
        assert len(bus.receive("a1")) == 1
        assert len(bus.receive("a2")) == 1

    def test_peek(self):
        bus = MessageBus()
        bus.send(AgentMessage(sender="x", receiver="y", content="1"))
        bus.send(AgentMessage(sender="x", receiver="y", content="2"))
        assert bus.peek("y") == 2
        assert bus.peek("z") == 0

    def test_create_response(self):
        msg = AgentMessage(sender="a1", receiver="a2", content="question")
        resp = msg.create_response("answer")
        assert resp.sender == "a2"
        assert resp.receiver == "a1"
        assert resp.msg_type == MessageType.RESPONSE
        assert resp.correlation_id == msg.id


# ===========================================================================
# AgentCoordinator Tests
# ===========================================================================


class TestAgentCoordinator:
    def setup_method(self):
        self.coordinator = AgentCoordinator()

    def test_register_and_list_agents(self):
        self.coordinator.register_agent("a1", "worker", capabilities=["search"])
        self.coordinator.register_agent("a2", "worker", capabilities=["generate"])
        agents = self.coordinator.list_agents()
        assert len(agents) == 2

    def test_create_task(self):
        task = self.coordinator.create_task("Test task", input_data={"key": "val"})
        assert task.status == TaskStatus.PENDING
        assert task.description == "Test task"

    def test_assign_task(self):
        self.coordinator.register_agent("a1", "worker")
        task = self.coordinator.create_task("Task 1")
        assert self.coordinator.assign_task("a1", task.id)
        assert task.status == TaskStatus.ASSIGNED
        assert task.assigned_agent == "a1"

    def test_assign_to_busy_agent_fails(self):
        self.coordinator.register_agent("a1", "worker")
        t1 = self.coordinator.create_task("Task 1")
        t2 = self.coordinator.create_task("Task 2")
        self.coordinator.assign_task("a1", t1.id)
        assert not self.coordinator.assign_task("a1", t2.id)

    def test_auto_assign(self):
        async def dummy(data):
            return data

        self.coordinator.register_agent("a1", "worker", execute_fn=dummy)
        self.coordinator.register_agent("a2", "worker", execute_fn=dummy)
        self.coordinator.create_task("T1")
        self.coordinator.create_task("T2")
        self.coordinator.create_task("T3")

        assignments = self.coordinator.auto_assign()
        assert len(assignments) == 2  # Only 2 agents available

    async def test_execute_task(self):
        async def double(data):
            return data * 2

        self.coordinator.register_agent("a1", "worker", execute_fn=double)
        task = self.coordinator.create_task("Double 5", input_data=5)
        self.coordinator.assign_task("a1", task.id)
        result = await self.coordinator.execute_task(task.id)
        assert result == 10
        assert task.status == TaskStatus.COMPLETED

    async def test_execute_task_failure(self):
        async def fail(data):
            raise ValueError("Test error")

        self.coordinator.register_agent("a1", "worker", execute_fn=fail)
        task = self.coordinator.create_task("Fail", input_data="x")
        self.coordinator.assign_task("a1", task.id)

        with pytest.raises(ValueError):
            await self.coordinator.execute_task(task.id)
        assert task.status == TaskStatus.FAILED
        assert "Test error" in task.error

    def test_cancel_task(self):
        self.coordinator.register_agent("a1", "worker")
        task = self.coordinator.create_task("Cancel me")
        assert self.coordinator.cancel_task(task.id)
        assert task.status == TaskStatus.CANCELLED

    async def test_execute_all_pending(self):
        async def inc(data):
            return data + 1

        self.coordinator.register_agent("a1", "worker", execute_fn=inc)
        self.coordinator.register_agent("a2", "worker", execute_fn=inc)
        self.coordinator.create_task("T1", input_data=10)
        self.coordinator.create_task("T2", input_data=20)

        results = await self.coordinator.execute_all_pending()
        values = [r for r in results.values() if not isinstance(r, dict)]
        assert sorted(values) == [11, 21]

    def test_collect_results(self):
        self.coordinator.register_agent("a1", "worker")
        task = self.coordinator.create_task("T1")
        task.status = TaskStatus.COMPLETED
        task.result = "done"
        task.assigned_agent = "a1"
        task.started_at = 1.0
        task.completed_at = 2.0

        results = self.coordinator.collect_results()
        assert task.id in results
        assert results[task.id]["result"] == "done"

    def test_priority_queue(self):
        self.coordinator.create_task("Low", priority=TaskPriority.LOW)
        self.coordinator.create_task("Critical", priority=TaskPriority.CRITICAL)
        self.coordinator.create_task("Normal", priority=TaskPriority.NORMAL)

        # The queue should have critical tasks first
        queue_tasks = [self.coordinator._tasks[tid] for tid in self.coordinator._task_queue]
        priorities = [t.priority for t in queue_tasks]
        assert priorities[0] == TaskPriority.CRITICAL

    def test_stats(self):
        self.coordinator.register_agent("a1", "worker")
        self.coordinator.create_task("T1")
        stats = self.coordinator.get_stats()
        assert stats["total_agents"] == 1
        assert stats["total_tasks"] == 1
        assert stats["pending_tasks"] == 1


# ===========================================================================
# Collaboration Strategies Tests
# ===========================================================================


class TestSequentialStrategy:
    async def test_sequential_execution(self):
        async def add_one(x):
            return x + 1

        async def double(x):
            return x * 2

        agents = [
            AgentHandle(agent_id="adder", execute_fn=add_one),
            AgentHandle(agent_id="doubler", execute_fn=double),
        ]
        strategy = SequentialStrategy()
        result = await strategy.execute(agents, 5)
        assert result.final_result == 12  # (5+1)*2
        assert result.agent_results["adder"] == 6
        assert result.agent_results["doubler"] == 12

    async def test_empty_agents(self):
        strategy = SequentialStrategy()
        result = await strategy.execute([], "input")
        assert result.final_result is None


class TestParallelStrategy:
    async def test_parallel_execution(self):
        async def upper(text):
            return str(text).upper()

        async def lower(text):
            return str(text).lower()

        agents = [
            AgentHandle(agent_id="upper", execute_fn=upper),
            AgentHandle(agent_id="lower", execute_fn=lower),
        ]
        strategy = ParallelStrategy()
        result = await strategy.execute(agents, "Hello")
        assert result.agent_results["upper"] == "HELLO"
        assert result.agent_results["lower"] == "hello"

    async def test_with_merge_fn(self):
        async def agent_a(x):
            return x + 1

        async def agent_b(x):
            return x + 2

        agents = [
            AgentHandle(agent_id="a", execute_fn=agent_a),
            AgentHandle(agent_id="b", execute_fn=agent_b),
        ]

        def merge(results):
            return sum(results.values())

        strategy = ParallelStrategy(merge_fn=merge)
        result = await strategy.execute(agents, 10)
        assert result.final_result == 23  # (10+1) + (10+2)

    async def test_handles_errors(self):
        async def fail(x):
            raise RuntimeError("fail!")

        async def ok(x):
            return "ok"

        agents = [
            AgentHandle(agent_id="fail", execute_fn=fail),
            AgentHandle(agent_id="ok", execute_fn=ok),
        ]
        strategy = ParallelStrategy()
        result = await strategy.execute(agents, "input")
        assert "error" in result.agent_results["fail"]
        assert result.agent_results["ok"] == "ok"


class TestDebateStrategy:
    async def test_single_round(self):
        async def agent_fn(x):
            return "answer"

        agents = [
            AgentHandle(agent_id="a1", execute_fn=agent_fn),
            AgentHandle(agent_id="a2", execute_fn=agent_fn),
        ]
        strategy = DebateStrategy(rounds=1)
        result = await strategy.execute(agents, "question")
        assert result.final_result == "answer"
        assert result.metadata["rounds"] == 1

    async def test_multi_round(self):
        call_count = {"a1": 0, "a2": 0}

        async def agent_fn(x):
            # Determine which agent based on round
            return f"round result"

        agents = [
            AgentHandle(agent_id="a1", execute_fn=agent_fn),
            AgentHandle(agent_id="a2", execute_fn=agent_fn),
        ]
        strategy = DebateStrategy(rounds=3)
        result = await strategy.execute(agents, "topic")
        assert len(result.metadata["all_rounds"]) == 3


class TestVotingStrategy:
    async def test_majority_vote(self):
        async def vote_a(x):
            return "answer_A"

        async def vote_b(x):
            return "answer_A"

        async def vote_c(x):
            return "answer_B"

        agents = [
            AgentHandle(agent_id="a1", execute_fn=vote_a),
            AgentHandle(agent_id="a2", execute_fn=vote_b),
            AgentHandle(agent_id="a3", execute_fn=vote_c),
        ]
        strategy = VotingStrategy()
        result = await strategy.execute(agents, "question")
        assert result.final_result == "answer_A"
        assert result.metadata["votes"]["answer_A"] == 2

    async def test_all_fail(self):
        async def fail(x):
            raise RuntimeError("error")

        agents = [AgentHandle(agent_id="a1", execute_fn=fail)]
        strategy = VotingStrategy()
        result = await strategy.execute(agents, "x")
        assert result.final_result is None
        assert "all_agents_failed" in result.metadata.get("error", "")


# ===========================================================================
# AgentPool Tests
# ===========================================================================


class TestAgentPool:
    def setup_method(self):
        self.pool = AgentPool(max_size=5)

    def test_register_and_list(self):
        self.pool.register_agent("a1", "worker", capabilities=["search"])
        self.pool.register_agent("a2", "worker", capabilities=["generate"])
        assert self.pool.size == 2
        assert len(self.pool.list_agents()) == 2

    def test_max_size(self):
        pool = AgentPool(max_size=2)
        assert pool.register_agent("a1", "worker")
        assert pool.register_agent("a2", "worker")
        assert not pool.register_agent("a3", "worker")

    def test_get_available_agent(self):
        self.pool.register_agent("a1", "worker", capabilities=["search", "summarize"])
        self.pool.register_agent("a2", "worker", capabilities=["translate"])

        agent = self.pool.get_available_agent(required_capabilities=["search"])
        assert agent is not None
        assert agent.agent_id == "a1"

        agent = self.pool.get_available_agent(required_capabilities=["translate"])
        assert agent.agent_id == "a2"

        agent = self.pool.get_available_agent(required_capabilities=["nonexistent"])
        assert agent is None

    def test_acquire_and_release(self):
        self.pool.register_agent("a1", "worker")
        assert self.pool.acquire("a1")
        assert self.pool.available_count == 0

        agent = self.pool.get_available_agent()
        assert agent is None

        assert self.pool.release("a1")
        assert self.pool.available_count == 1

    def test_unregister(self):
        self.pool.register_agent("a1", "worker")
        assert self.pool.unregister_agent("a1")
        assert self.pool.size == 0
        assert not self.pool.unregister_agent("nonexistent")

    def test_list_capabilities(self):
        self.pool.register_agent("a1", "w", capabilities=["a", "b"])
        self.pool.register_agent("a2", "w", capabilities=["b", "c"])
        caps = self.pool.list_capabilities()
        assert caps == ["a", "b", "c"]

    def test_stats(self):
        self.pool.register_agent("a1", "worker")
        self.pool.register_agent("a2", "analyzer")
        self.pool.acquire("a1")
        stats = self.pool.get_stats()
        assert stats["total"] == 2
        assert stats["available"] == 1
        assert stats["busy"] == 1
        assert stats["types"]["worker"] == 1
        assert stats["types"]["analyzer"] == 1


# ===========================================================================
# Agent Chain Tests
# ===========================================================================


class TestAgentChain:
    async def test_simple_chain(self):
        async def step1(data):
            return data + " -> step1"

        async def step2(data):
            return data + " -> step2"

        chain = AgentChain(name="test_chain")
        chain.add("s1", step1, name="Step 1")
        chain.add("s2", step2, name="Step 2")

        result = await chain.execute("start")
        assert result.success
        assert result.final_output == "start -> step1 -> step2"
        assert len(result.step_results) == 2

    async def test_chain_with_transform(self):
        async def process(data):
            return data * 2

        chain = AgentChain(name="transform_chain")
        chain.add(
            "processor",
            process,
            input_transform=lambda x: int(x),
            output_transform=lambda x: str(x),
        )

        result = await chain.execute("5")
        assert result.success
        assert result.final_output == "10"

    async def test_chain_with_condition(self):
        async def step_a(data):
            return "a_result"

        async def step_b(data):
            return "b_result"

        chain = AgentChain(name="conditional_chain")
        chain.add("a", step_a, condition=lambda x: x == "go")
        chain.add("b", step_b)

        # Condition False -> skip step a
        result = await chain.execute("nope")
        assert result.success
        assert "b_result" in result.final_output

        # Condition True -> run both
        result2 = await chain.execute("go")
        assert result2.success

    async def test_chain_with_fallback(self):
        async def failing_step(data):
            raise RuntimeError("primary failed")

        async def fallback_step(data):
            return "fallback_result"

        chain = AgentChain(name="fallback_chain")
        chain.add_step(ChainStep(
            agent_id="primary",
            name="Primary",
            execute_fn=failing_step,
            fallback_step=ChainStep(
                agent_id="fallback",
                name="Fallback",
                execute_fn=fallback_step,
            ),
        ))

        result = await chain.execute("input")
        assert result.success
        assert result.final_output == "fallback_result"
        assert result.step_results[0]["status"] == "fallback_completed"

    async def test_chain_failure_no_fallback(self):
        async def failing_step(data):
            raise RuntimeError("failed")

        chain = AgentChain(name="fail_chain")
        chain.add("fail", failing_step)

        result = await chain.execute("input")
        assert not result.success
        assert "failed" in result.error

    def test_chain_to_dict(self):
        chain = AgentChain(name="test", description="desc")
        chain.add_step(ChainStep(agent_id="s1", name="Step 1"))
        d = chain.to_dict()
        assert d["name"] == "test"
        assert len(d["steps"]) == 1


# ===========================================================================
# Chain Templates Tests
# ===========================================================================


class TestChainTemplates:
    def test_list_templates(self):
        templates = list_chain_templates()
        assert len(templates) >= 4
        names = [t["id"] for t in templates]
        assert "research_chain" in names
        assert "analysis_chain" in names
        assert "qa_chain" in names
        assert "translation_chain" in names

    def test_research_chain_creation(self):
        async def dummy(x):
            return x

        chain = create_research_chain(
            router_fn=dummy, retriever_fn=dummy,
            summarizer_fn=dummy, formatter_fn=dummy,
        )
        assert chain.name == "research_chain"
        assert chain.step_count == 4

    def test_analysis_chain_creation(self):
        async def dummy(x):
            return x

        chain = create_analysis_chain(
            analyzer_fn=dummy, reporter_fn=dummy, email_fn=dummy,
        )
        assert chain.name == "analysis_chain"
        assert chain.step_count == 3

    def test_qa_chain_creation(self):
        async def dummy(x):
            return x

        chain = create_qa_chain(
            router_fn=dummy, planner_fn=dummy,
            retriever_fn=dummy, merger_fn=dummy, quality_fn=dummy,
        )
        assert chain.name == "qa_chain"
        assert chain.step_count == 5

    def test_translation_chain_creation(self):
        async def dummy(x):
            return x

        chain = create_translation_chain(
            detector_fn=dummy, translator_fn=dummy, quality_fn=dummy,
        )
        assert chain.name == "translation_chain"
        assert chain.step_count == 3

    def test_template_registry(self):
        assert "research_chain" in CHAIN_TEMPLATES
        assert "factory" in CHAIN_TEMPLATES["research_chain"]


# ===========================================================================
# ExecutionTracker Tests
# ===========================================================================


class TestExecutionTracker:
    def setup_method(self):
        self.tracker = ExecutionTracker()

    def test_start_and_get(self):
        exec_id = self.tracker.start(ExecutionType.AGENT, "test_agent")
        record = self.tracker.get(exec_id)
        assert record is not None
        assert record.name == "test_agent"
        assert record.state == ExecutionState.RUNNING

    def test_complete(self):
        exec_id = self.tracker.start(ExecutionType.AGENT, "test")
        self.tracker.complete(exec_id)
        record = self.tracker.get(exec_id)
        assert record.state == ExecutionState.COMPLETED
        assert record.duration_ms is not None

    def test_fail(self):
        exec_id = self.tracker.start(ExecutionType.TOOL, "calc")
        self.tracker.fail(exec_id, "division by zero")
        record = self.tracker.get(exec_id)
        assert record.state == ExecutionState.FAILED
        assert record.error == "division by zero"

    def test_update_step(self):
        exec_id = self.tracker.start(ExecutionType.CHAIN, "my_chain", total_steps=3)
        self.tracker.update_step(exec_id, "step_1", completed_steps=1)
        record = self.tracker.get(exec_id)
        assert record.current_step == "step_1"
        assert record.completed_steps == 1

    def test_add_tokens(self):
        exec_id = self.tracker.start(ExecutionType.AGENT, "test")
        self.tracker.add_tokens(exec_id, 100)
        self.tracker.add_tokens(exec_id, 50)
        record = self.tracker.get(exec_id)
        assert record.tokens_used == 150

    def test_get_active(self):
        id1 = self.tracker.start(ExecutionType.AGENT, "a1")
        id2 = self.tracker.start(ExecutionType.AGENT, "a2")
        self.tracker.complete(id1)
        active = self.tracker.get_active()
        assert len(active) == 1
        assert active[0].id == id2

    def test_get_metrics(self):
        id1 = self.tracker.start(ExecutionType.AGENT, "a1")
        id2 = self.tracker.start(ExecutionType.TOOL, "t1")
        self.tracker.add_tokens(id1, 100)
        self.tracker.complete(id1)
        self.tracker.fail(id2, "err")

        metrics = self.tracker.get_metrics()
        assert metrics["total_executions"] == 2
        assert metrics["completed"] == 1
        assert metrics["failed"] == 1
        assert metrics["total_tokens"] == 100
        assert metrics["success_rate"] == 0.5

    def test_get_recent(self):
        for i in range(5):
            self.tracker.start(ExecutionType.AGENT, f"agent_{i}")
        recent = self.tracker.get_recent(limit=3)
        assert len(recent) == 3

    def test_callbacks(self):
        events = []
        self.tracker.on_update(lambda r: events.append(r.state))
        exec_id = self.tracker.start(ExecutionType.AGENT, "test")
        self.tracker.complete(exec_id)
        assert len(events) == 2  # start + complete
        assert events[0] == ExecutionState.RUNNING
        assert events[1] == ExecutionState.COMPLETED

    def test_clear(self):
        id1 = self.tracker.start(ExecutionType.AGENT, "active")
        id2 = self.tracker.start(ExecutionType.AGENT, "done")
        self.tracker.complete(id2)
        removed = self.tracker.clear(keep_active=True)
        assert removed == 1
        assert self.tracker.get(id1) is not None
        assert self.tracker.get(id2) is None


# ===========================================================================
# DashboardDataProvider Tests
# ===========================================================================


class TestDashboardDataProvider:
    def setup_method(self):
        self.tracker = ExecutionTracker()
        self.dashboard = DashboardDataProvider(self.tracker)

    def test_get_overview_empty(self):
        overview = self.dashboard.get_overview()
        assert overview["active_count"] == 0
        assert overview["metrics"]["total_executions"] == 0

    def test_get_agent_metrics(self):
        self.tracker.start(ExecutionType.AGENT, "a1")
        metrics = self.dashboard.get_agent_metrics()
        assert metrics["total"] == 1

    def test_get_chain_metrics_by_id(self):
        exec_id = self.tracker.start(ExecutionType.CHAIN, "my_chain", total_steps=5)
        self.tracker.update_step(exec_id, "step_2", completed_steps=2)
        metrics = self.dashboard.get_chain_metrics(chain_id=exec_id)
        assert metrics["name"] == "my_chain"
        assert metrics["completed_steps"] == 2

    def test_get_chain_metrics_not_found(self):
        metrics = self.dashboard.get_chain_metrics(chain_id="nonexistent")
        assert "error" in metrics

    def test_get_tool_usage_stats(self):
        id1 = self.tracker.start(ExecutionType.TOOL, "calculator")
        id2 = self.tracker.start(ExecutionType.TOOL, "calculator")
        id3 = self.tracker.start(ExecutionType.TOOL, "translator")
        self.tracker.complete(id1)
        self.tracker.complete(id2)
        self.tracker.fail(id3, "error")

        stats = self.dashboard.get_tool_usage_stats()
        assert stats["total_tool_calls"] == 3
        assert stats["unique_tools"] == 2
        calc = next(t for t in stats["tools"] if t["name"] == "calculator")
        assert calc["call_count"] == 2

    def test_get_active_executions(self):
        self.tracker.start(ExecutionType.AGENT, "running_agent")
        active = self.dashboard.get_active_executions()
        assert len(active) == 1
        assert active[0]["name"] == "running_agent"
