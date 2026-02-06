"""Agent chain: defines and executes a chain of agents with input/output mappings."""

import time
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ChainStep:
    """A single step in an agent chain.

    Attributes:
        agent_id: Identifier for the agent to execute.
        input_transform: Optional function to transform input before this step.
        output_transform: Optional function to transform output after this step.
        condition: Optional function(current_data) -> bool to decide if step runs.
        fallback_step: Optional step to execute if this step fails.
        execute_fn: The async callable that does the work.
        name: Human-readable step name.
    """
    agent_id: str
    name: str = ""
    execute_fn: Callable | None = None  # async (input_data) -> result
    input_transform: Callable | None = None  # (data) -> transformed_data
    output_transform: Callable | None = None  # (result) -> transformed_result
    condition: Callable | None = None  # (data) -> bool
    fallback_step: "ChainStep | None" = None
    metadata: dict = field(default_factory=dict)


@dataclass
class ChainExecutionResult:
    """Result of executing an agent chain."""
    final_output: Any = None
    step_results: list[dict] = field(default_factory=list)
    total_duration_ms: int = 0
    success: bool = True
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "final_output": self.final_output,
            "step_results": self.step_results,
            "total_duration_ms": self.total_duration_ms,
            "success": self.success,
            "error": self.error,
        }


class AgentChain:
    """A chain of agents that execute in sequence with data flowing between them.

    Supports:
    - Input/output transforms between steps
    - Conditional step execution (branching)
    - Fallback chains on error
    - Step-by-step execution tracking
    """

    def __init__(self, name: str = "", description: str = ""):
        self.name = name
        self.description = description
        self._steps: list[ChainStep] = []

    def add_step(self, step: ChainStep) -> "AgentChain":
        """Add a step to the chain. Returns self for fluent chaining."""
        self._steps.append(step)
        return self

    def add(
        self,
        agent_id: str,
        execute_fn: Callable,
        name: str = "",
        input_transform: Callable | None = None,
        output_transform: Callable | None = None,
        condition: Callable | None = None,
    ) -> "AgentChain":
        """Convenience method to add a step with minimal boilerplate."""
        step = ChainStep(
            agent_id=agent_id,
            name=name or agent_id,
            execute_fn=execute_fn,
            input_transform=input_transform,
            output_transform=output_transform,
            condition=condition,
        )
        return self.add_step(step)

    @property
    def steps(self) -> list[ChainStep]:
        return list(self._steps)

    @property
    def step_count(self) -> int:
        return len(self._steps)

    async def execute(self, initial_input: Any) -> ChainExecutionResult:
        """Execute the chain starting with the given input.

        Args:
            initial_input: The initial data to feed into the first step.

        Returns:
            ChainExecutionResult with final output and per-step results.
        """
        result = ChainExecutionResult()
        start_time = time.time()
        current_data = initial_input

        for i, step in enumerate(self._steps):
            step_start = time.time()
            step_info = {
                "step_index": i,
                "agent_id": step.agent_id,
                "name": step.name,
                "status": "pending",
            }

            try:
                # Check condition
                if step.condition and not step.condition(current_data):
                    step_info["status"] = "skipped"
                    step_info["duration_ms"] = int((time.time() - step_start) * 1000)
                    result.step_results.append(step_info)
                    continue

                # Apply input transform
                step_input = current_data
                if step.input_transform:
                    step_input = step.input_transform(current_data)

                # Execute
                if not step.execute_fn:
                    raise ValueError(f"Step '{step.agent_id}' has no execute function")

                step_info["status"] = "running"
                step_output = await step.execute_fn(step_input)

                # Apply output transform
                if step.output_transform:
                    step_output = step.output_transform(step_output)

                current_data = step_output
                step_info["status"] = "completed"
                step_info["output_preview"] = str(step_output)[:200]

            except Exception as e:
                step_info["status"] = "failed"
                step_info["error"] = str(e)

                # Try fallback
                if step.fallback_step and step.fallback_step.execute_fn:
                    try:
                        fallback_input = current_data
                        if step.fallback_step.input_transform:
                            fallback_input = step.fallback_step.input_transform(current_data)
                        current_data = await step.fallback_step.execute_fn(fallback_input)
                        if step.fallback_step.output_transform:
                            current_data = step.fallback_step.output_transform(current_data)
                        step_info["status"] = "fallback_completed"
                        step_info["fallback_agent"] = step.fallback_step.agent_id
                    except Exception as fallback_err:
                        step_info["status"] = "fallback_failed"
                        step_info["fallback_error"] = str(fallback_err)
                        result.success = False
                        result.error = f"Step '{step.agent_id}' and fallback failed: {e} / {fallback_err}"
                        step_info["duration_ms"] = int((time.time() - step_start) * 1000)
                        result.step_results.append(step_info)
                        break
                else:
                    result.success = False
                    result.error = f"Step '{step.agent_id}' failed: {e}"
                    step_info["duration_ms"] = int((time.time() - step_start) * 1000)
                    result.step_results.append(step_info)
                    break

            step_info["duration_ms"] = int((time.time() - step_start) * 1000)
            result.step_results.append(step_info)

        result.final_output = current_data
        result.total_duration_ms = int((time.time() - start_time) * 1000)
        return result

    def to_dict(self) -> dict:
        """Serialize chain metadata."""
        return {
            "name": self.name,
            "description": self.description,
            "steps": [
                {
                    "agent_id": s.agent_id,
                    "name": s.name,
                    "has_condition": s.condition is not None,
                    "has_fallback": s.fallback_step is not None,
                    "metadata": s.metadata,
                }
                for s in self._steps
            ],
        }
