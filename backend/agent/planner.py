"""Task planner for decomposing complex questions into execution steps."""

import json
from dataclasses import dataclass, field

from core.llm import BaseLLM, create_llm


@dataclass
class PlanStep:
    step_id: int
    action: str        # "search", "generate", "summarize", "compare", "tool"
    description: str
    query: str | None = None       # Query for this step
    depends_on: list[int] = field(default_factory=list)
    result: str | None = None
    status: str = "pending"  # pending, running, completed, failed


@dataclass
class ExecutionPlan:
    original_query: str
    steps: list[PlanStep]
    final_instruction: str = "종합하여 하나의 답변을 생성하세요."


PLANNER_SYSTEM = """당신은 복잡한 질문을 단계별 실행 계획으로 분해하는 플래너입니다.

질문을 분석하여 JSON 형식의 실행 계획을 작성하세요.

각 단계는 다음 중 하나의 action을 가집니다:
- "search": 문서 검색 (query 필수)
- "generate": LLM으로 텍스트 생성
- "summarize": 여러 결과 종합
- "compare": 두 결과 비교 분석

출력 형식:
{
  "steps": [
    {"step_id": 1, "action": "search", "description": "...", "query": "...", "depends_on": []},
    {"step_id": 2, "action": "search", "description": "...", "query": "...", "depends_on": []},
    {"step_id": 3, "action": "summarize", "description": "...", "depends_on": [1, 2]}
  ],
  "final_instruction": "..."
}"""


class TaskPlanner:
    """Decomposes complex questions into step-by-step execution plans."""

    def __init__(self, llm: BaseLLM | None = None):
        self.llm = llm or create_llm()

    async def plan(self, query: str, context: str = "") -> ExecutionPlan:
        """Create an execution plan for a complex query.

        Args:
            query: The complex user question.
            context: Optional additional context.

        Returns:
            ExecutionPlan with ordered steps.
        """
        prompt = f"질문: {query}"
        if context:
            prompt = f"배경: {context}\n\n{prompt}"

        response = await self.llm.generate(
            prompt=prompt,
            system=PLANNER_SYSTEM,
            temperature=0.2,
        )

        return self._parse_plan(query, response.content)

    def _parse_plan(self, original_query: str, response_text: str) -> ExecutionPlan:
        """Parse LLM response into ExecutionPlan."""
        try:
            text = response_text.strip()
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()

            data = json.loads(text)
            steps = []
            for s in data.get("steps", []):
                steps.append(PlanStep(
                    step_id=s["step_id"],
                    action=s["action"],
                    description=s["description"],
                    query=s.get("query"),
                    depends_on=s.get("depends_on", []),
                ))

            return ExecutionPlan(
                original_query=original_query,
                steps=steps,
                final_instruction=data.get("final_instruction", "종합하여 답변하세요."),
            )
        except (json.JSONDecodeError, KeyError, TypeError):
            # Fallback: single search + summarize
            return ExecutionPlan(
                original_query=original_query,
                steps=[
                    PlanStep(step_id=1, action="search", description="문서 검색", query=original_query),
                    PlanStep(step_id=2, action="generate", description="답변 생성", depends_on=[1]),
                ],
            )

    async def execute_plan(
        self,
        plan: ExecutionPlan,
        search_fn=None,
        generate_fn=None,
    ) -> str:
        """Execute a plan step by step.

        Args:
            plan: ExecutionPlan to execute.
            search_fn: async fn(query) -> str for search steps.
            generate_fn: async fn(prompt, context) -> str for generate/summarize steps.

        Returns:
            Final combined result string.
        """
        results = {}

        for step in plan.steps:
            # Check dependencies
            dep_context = "\n\n".join(
                f"[Step {d} 결과]\n{results[d]}"
                for d in step.depends_on
                if d in results
            )

            if step.action == "search" and search_fn:
                step.status = "running"
                step.result = await search_fn(step.query or plan.original_query)
                step.status = "completed"
            elif step.action in ("generate", "summarize", "compare") and generate_fn:
                step.status = "running"
                prompt = f"{step.description}\n\n{dep_context}" if dep_context else step.description
                step.result = await generate_fn(prompt, dep_context)
                step.status = "completed"
            else:
                step.status = "completed"
                step.result = dep_context or ""

            results[step.step_id] = step.result or ""

        # Return last step result or combine all
        if plan.steps:
            return plan.steps[-1].result or ""
        return ""
