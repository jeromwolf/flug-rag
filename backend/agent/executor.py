"""Agent Plan Executor — runs ExecutionPlan steps in dependency order."""

import json
import logging
from typing import AsyncIterator

from .planner import ExecutionPlan, PlanStep
from .mcp.registry import get_registry, ToolRegistry

logger = logging.getLogger(__name__)


class PlanExecutor:
    """Execute plan steps produced by TaskPlanner."""

    def __init__(self, rag_chain, registry: ToolRegistry | None = None):
        self.rag = rag_chain
        self.registry = registry or get_registry()

    async def execute(
        self,
        plan: ExecutionPlan,
        filters: dict | None = None,
        provider: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
    ) -> str:
        """Execute all plan steps and return final aggregated answer."""
        results: dict[int, str] = {}

        for step in plan.steps:
            # Collect dependency context
            dep_context = "\n\n".join(
                results[d] for d in step.depends_on if d in results
            )
            try:
                result = await self._run_step(
                    step, dep_context, filters, provider, model, temperature
                )
                step.result = result
                step.status = "completed"
                results[step.step_id] = result
            except Exception as e:
                logger.error("Step %d failed: %s", step.step_id, e)
                step.status = "failed"
                step.result = f"오류: {e}"
                results[step.step_id] = step.result

        # Aggregate: use final_instruction or last step result
        if plan.final_instruction and results:
            all_results = "\n\n".join(
                f"[단계 {sid}] {r}" for sid, r in sorted(results.items())
            )
            prompt = f"{plan.final_instruction}\n\n참고 정보:\n{all_results}"
            resp = await self.rag.query(
                question=prompt, mode="direct",
                provider=provider, model=model, temperature=temperature,
            )
            return resp.content if hasattr(resp, "content") else str(resp)

        # Fallback: return last step
        if results:
            return results[max(results.keys())]
        return "실행할 단계가 없습니다."

    async def stream_execute(
        self,
        plan: ExecutionPlan,
        filters: dict | None = None,
        provider: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
    ) -> AsyncIterator[dict]:
        """Yield SSE-compatible events while executing plan steps."""
        results: dict[int, str] = {}

        for step in plan.steps:
            yield {
                "event": "tool_start",
                "data": {"message": f"단계 {step.step_id}: {step.description}"},
            }

            dep_context = "\n\n".join(
                results[d] for d in step.depends_on if d in results
            )
            try:
                result = await self._run_step(
                    step, dep_context, filters, provider, model, temperature
                )
                step.result = result
                step.status = "completed"
                results[step.step_id] = result
            except Exception as e:
                logger.error("Step %d failed: %s", step.step_id, e)
                step.status = "failed"
                step.result = f"오류: {e}"
                results[step.step_id] = step.result

            yield {"event": "tool_end", "data": {}}

        # Final aggregation via streaming
        if plan.final_instruction and results:
            all_results = "\n\n".join(
                f"[단계 {sid}] {r}" for sid, r in sorted(results.items())
            )
            prompt = f"{plan.final_instruction}\n\n참고 정보:\n{all_results}"
            async for evt in self.rag.stream_query(
                question=prompt, mode="direct",
                provider=provider, model=model, temperature=temperature,
            ):
                yield evt
        elif results:
            # Stream last result as chunks
            final = results[max(results.keys())]
            yield {"event": "chunk", "data": {"content": final}}
            yield {"event": "end", "data": {"confidence_score": 0.0}}

    async def _run_step(
        self,
        step: PlanStep,
        dep_context: str,
        filters: dict | None,
        provider: str | None,
        model: str | None,
        temperature: float | None,
    ) -> str:
        """Dispatch a single plan step by action type."""
        action = step.action

        if action == "search":
            resp = await self.rag.query(
                question=step.query or step.description,
                mode="rag",
                filters=filters,
                provider=provider,
                model=model,
                temperature=temperature,
            )
            return resp.content if hasattr(resp, "content") else str(resp)

        elif action == "generate":
            prompt = step.description
            if dep_context:
                prompt = f"{step.description}\n\n참고:\n{dep_context}"
            resp = await self.rag.query(
                question=prompt,
                mode="direct",
                provider=provider,
                model=model,
                temperature=temperature,
            )
            return resp.content if hasattr(resp, "content") else str(resp)

        elif action == "summarize":
            prompt = f"다음 내용을 요약해 주세요:\n\n{dep_context}"
            if step.query:
                prompt = f"{step.query}\n\n{dep_context}"
            resp = await self.rag.query(
                question=prompt,
                mode="direct",
                provider=provider,
                model=model,
                temperature=temperature,
            )
            return resp.content if hasattr(resp, "content") else str(resp)

        elif action == "compare":
            prompt = f"다음 내용을 비교 분석해 주세요:\n\n{dep_context}"
            resp = await self.rag.query(
                question=prompt,
                mode="direct",
                provider=provider,
                model=model,
                temperature=temperature,
            )
            return resp.content if hasattr(resp, "content") else str(resp)

        elif action == "tool":
            # Parse tool name from description if present
            tool_name = self._extract_tool_name(step.description)
            if tool_name and self.registry.get_tool(tool_name) is not None:
                result = await self.registry.execute(
                    tool_name, query=step.query or step.description
                )
                if result.success:
                    return self._format_tool_result(tool_name, result.data)
                return f"도구 실행 실패: {result.error}"
            # Fallback to search
            resp = await self.rag.query(
                question=step.query or step.description,
                mode="rag",
                filters=filters,
                provider=provider,
                model=model,
                temperature=temperature,
            )
            return resp.content if hasattr(resp, "content") else str(resp)

        else:
            # Unknown action — treat as generate
            resp = await self.rag.query(
                question=step.description,
                mode="direct",
                provider=provider,
                model=model,
                temperature=temperature,
            )
            return resp.content if hasattr(resp, "content") else str(resp)

    def _extract_tool_name(self, description: str) -> str | None:
        """Try to find a known tool name in the step description."""
        known = [
            "report_draft", "training_material", "regulation_review",
            "safety_checklist", "calculator", "data_analyzer",
            "document_search", "knowledge_base", "summarizer",
            "translator", "report_generator", "email_composer",
        ]
        desc_lower = description.lower()
        for name in known:
            if name in desc_lower:
                return name
        return None

    @staticmethod
    def _format_tool_result(tool_name: str, data) -> str:
        """Format ToolResult.data into displayable text."""
        if data is None:
            return ""
        if isinstance(data, str):
            return data
        if isinstance(data, dict):
            for key in ("draft", "material", "report"):
                if key in data:
                    return data[key]
            if "result" in data and "expression" in data:
                return f"계산 결과: {data['expression']} = {data['result']}"
            return json.dumps(data, ensure_ascii=False, indent=2)
        return str(data)
