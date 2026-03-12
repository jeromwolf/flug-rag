"""Workflow execution engine: runs DAG-based agent workflows."""

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

from agent.builder.models import (
    Edge, ExecutionStatus, NodeConfig, NodeExecutionResult, NodeType,
    Workflow, WorkflowExecutionResult, WorkflowNode,
)
from agent.mcp.registry import ToolRegistry, create_default_registry
from core.llm import BaseLLM, create_llm
from rag import RAGChain


class WorkflowEngine:
    """Executes workflow DAGs step by step."""

    def __init__(
        self,
        llm: BaseLLM | None = None,
        rag_chain: RAGChain | None = None,
        tool_registry: ToolRegistry | None = None,
    ):
        self.llm = llm or create_llm()
        self.rag_chain = rag_chain
        self.tool_registry = tool_registry or create_default_registry()

    async def execute(
        self,
        workflow: Workflow,
        input_data: dict,
    ) -> WorkflowExecutionResult:
        """Execute a workflow with given input.

        Args:
            workflow: The workflow definition to execute.
            input_data: Initial input (e.g., {"query": "..."}).

        Returns:
            WorkflowExecutionResult with all node results.
        """
        start_time = time.time()
        result = WorkflowExecutionResult(
            workflow_id=workflow.id,
            status=ExecutionStatus.RUNNING,
        )

        # Find start node
        start_node = workflow.get_start_node()
        if not start_node:
            result.status = ExecutionStatus.FAILED
            result.error = "No start node found in workflow"
            return result

        # Execute DAG by traversal
        context = {"input": input_data, "results": {}}

        try:
            await self._execute_node(workflow, start_node, context, result)
            result.status = ExecutionStatus.COMPLETED
            result.final_output = context.get("final_output", context.get("last_output"))
        except Exception as e:
            result.status = ExecutionStatus.FAILED
            result.error = str(e)

        result.total_duration_ms = int((time.time() - start_time) * 1000)
        return result

    async def _execute_node(
        self,
        workflow: Workflow,
        node: WorkflowNode,
        context: dict,
        result: WorkflowExecutionResult,
    ):
        """Execute a single node and traverse to next nodes."""
        start_time = time.time()
        node_result = NodeExecutionResult(
            node_id=node.id,
            status=ExecutionStatus.RUNNING,
        )

        try:
            # LOOP nodes manage their own edge traversal
            if node.config.node_type == NodeType.LOOP:
                output = await self._run_loop_node(node, workflow, context, result)
                node_result.output = f"[Loop completed: {len(context.get('loop_results', []))} iterations]"
                node_result.status = ExecutionStatus.COMPLETED
                context["results"][node.id] = output
                context["last_output"] = output
            else:
                output = await self._run_node(node.config, context)
                node_result.output = output
                node_result.status = ExecutionStatus.COMPLETED
                context["results"][node.id] = output
                # Condition nodes should NOT overwrite last_output (keep retrieval data for LLM)
                if node.config.node_type != NodeType.CONDITION:
                    context["last_output"] = output
        except Exception as e:
            node_result.status = ExecutionStatus.FAILED
            node_result.error = str(e)
            raise
        finally:
            node_result.duration_ms = int((time.time() - start_time) * 1000)
            result.node_results.append(node_result)

        # LOOP nodes handle their own edge traversal — skip standard traversal
        if node.config.node_type == NodeType.LOOP:
            return

        # Traverse outgoing edges
        edges = workflow.get_outgoing_edges(node.id)

        # For condition nodes: filter edges based on result (True/False)
        if node.config.node_type == NodeType.CONDITION and isinstance(output, bool):
            filtered = []
            for edge in edges:
                label_lower = (edge.label or "").lower().strip()
                condition_lower = (edge.condition or "").lower().strip()
                combined = label_lower + " " + condition_lower
                # "신뢰도 충분", "true", "high" → True path
                is_true_edge = any(k in combined for k in ["충분", "true", "high", "yes", "pass"])
                # "신뢰도 부족", "false", "low" → False path
                is_false_edge = any(k in combined for k in ["부족", "false", "low", "no", "fail"])
                if output and is_true_edge:
                    filtered.append(edge)
                elif not output and is_false_edge:
                    filtered.append(edge)
                elif not is_true_edge and not is_false_edge:
                    filtered.append(edge)  # unlabeled edges always pass
            edges = filtered

        for edge in edges:
            # Check condition if present (for non-condition nodes)
            if edge.condition and not self._evaluate_condition(edge.condition, context):
                continue

            next_node = workflow.get_node(edge.target)
            if next_node:
                await self._execute_node(workflow, next_node, context, result)

    async def _run_node(self, config: NodeConfig, context: dict) -> Any:
        """Run a specific node type."""
        node_type = config.node_type

        if node_type == NodeType.START:
            input_data = context["input"]
            # If input is a dict with "query", extract the query string
            if isinstance(input_data, dict) and "query" in input_data:
                return input_data["query"]
            return input_data

        elif node_type == NodeType.LLM:
            return await self._run_llm_node(config, context)

        elif node_type == NodeType.RETRIEVAL:
            return await self._run_retrieval_node(config, context)

        elif node_type == NodeType.TOOL:
            return await self._run_tool_node(config, context)

        elif node_type == NodeType.CONDITION:
            return self._run_condition_node(config, context)

        elif node_type == NodeType.TRANSFORM:
            return self._run_transform_node(config, context)

        elif node_type == NodeType.OUTPUT:
            output = context.get("last_output", "")
            context["final_output"] = output
            return output

        return None

    async def _run_llm_node(self, config: NodeConfig, context: dict) -> str:
        """Execute an LLM generation node."""
        cfg = config.config
        prompt = cfg.get("prompt_template", "{input}")

        # Substitute context variables
        last_output = context.get("last_output", context.get("input", {}).get("query", ""))
        # Extract content from retrieval dict result
        if isinstance(last_output, dict) and "content" in last_output:
            input_text = str(last_output["content"])
        else:
            input_text = str(last_output)
        prompt = prompt.replace("{input}", input_text)

        system = cfg.get("system_prompt", "")
        if system:
            system += "\n\n[중요] 반드시 한국어로만 답변하세요. 중국어(简体/繁體)를 절대 사용하지 마세요."
        else:
            system = "반드시 한국어로만 답변하세요. 중국어(简体/繁體)를 절대 사용하지 마세요."
        temperature = cfg.get("temperature", 0.7)

        response = await self.llm.generate(
            prompt=prompt,
            system=system,
            temperature=temperature,
        )
        # Strip CJK characters (Chinese leak from Qwen), preserve newlines
        content = response.content
        import re
        content = re.sub(r'[\u4e00-\u9fff\u3400-\u4dbf]+', '', content)
        content = re.sub(r'[^\S\n]+', ' ', content)  # collapse spaces but NOT newlines
        content = re.sub(r'\n{3,}', '\n\n', content)  # max 2 consecutive newlines
        return content.strip()

    async def _run_retrieval_node(self, config: NodeConfig, context: dict) -> dict:
        """Execute a RAG retrieval node."""
        if not self.rag_chain:
            from rag import RAGChain
            self.rag_chain = RAGChain()

        cfg = config.config
        query = str(context.get("last_output", context.get("input", {}).get("query", "")))

        response = await self.rag_chain.query(
            question=query,
            mode="rag",
            filters=cfg.get("filters"),
        )

        return {
            "content": response.content,
            "sources": response.sources,
            "confidence": response.confidence,
        }

    async def _run_tool_node(self, config: NodeConfig, context: dict) -> Any:
        """Execute an MCP tool node."""
        cfg = config.config
        tool_name = cfg.get("tool_name", "")
        arguments = dict(cfg.get("arguments_template", {}))

        # Substitute {input} in argument values
        input_text = str(context.get("last_output", ""))
        for key, val in arguments.items():
            if isinstance(val, str) and "{input}" in val:
                arguments[key] = val.replace("{input}", input_text)

        result = await self.tool_registry.execute(tool_name, **arguments)
        return result.data if result.success else {"error": result.error}

    def _run_condition_node(self, config: NodeConfig, context: dict) -> bool:
        """Evaluate a condition node."""
        cfg = config.config
        condition_type = cfg.get("condition_type", "confidence")
        threshold = cfg.get("threshold", 0.5)

        if condition_type == "confidence":
            last_output = context.get("last_output", {})
            if isinstance(last_output, dict):
                return last_output.get("confidence", 0) >= threshold

        return True

    def _run_transform_node(self, config: NodeConfig, context: dict) -> str:
        """Apply a text transformation."""
        cfg = config.config
        template = cfg.get("template", "{input}")
        input_text = str(context.get("last_output", ""))
        return template.replace("{input}", input_text)

    def _evaluate_condition(self, condition: str, context: dict) -> bool:
        """Evaluate an edge condition. Simple string matching for now."""
        last_output = context.get("last_output")
        if condition == "true":
            return bool(last_output)
        if condition == "false":
            return not bool(last_output)
        if condition == "high_confidence":
            if isinstance(last_output, dict):
                return last_output.get("confidence", 0) >= 0.8
        if condition == "low_confidence":
            if isinstance(last_output, dict):
                return last_output.get("confidence", 0) < 0.5
        return True

    # ------------------------------------------------------------------
    # LOOP node execution
    # ------------------------------------------------------------------

    async def _run_loop_node(
        self,
        node: WorkflowNode,
        workflow: Workflow,
        context: dict,
        result: WorkflowExecutionResult,
    ) -> str:
        """Execute a LOOP node: run body sub-graph N times, then follow done edges."""
        cfg = node.config.config
        loop_type = cfg.get("loop_type", "count")
        max_iter = min(int(cfg.get("max_iterations", 10)), 50)  # hard limit 50

        # Classify outgoing edges into body vs done
        edges = workflow.get_outgoing_edges(node.id)
        body_edges: list[Edge] = []
        done_edges: list[Edge] = []

        for edge in edges:
            label = (edge.label or "").lower().strip()
            if any(k in label for k in ["body", "loop", "반복"]):
                body_edges.append(edge)
            elif any(k in label for k in ["done", "complete", "완료"]):
                done_edges.append(edge)
            else:
                # First unlabeled edge → body, subsequent → done
                if not body_edges:
                    body_edges.append(edge)
                else:
                    done_edges.append(edge)

        iterations_output: list[str] = []

        if loop_type == "count":
            count = min(int(cfg.get("count", 3)), max_iter)
            for i in range(count):
                context["loop_iteration"] = i
                context["loop_total"] = count
                logger.info("Loop [count] iteration %d/%d", i + 1, count)
                await self._execute_body(body_edges, workflow, context, result)
                iterations_output.append(str(context.get("last_output", "")))

        elif loop_type == "foreach":
            separator = cfg.get("separator", "\n")
            raw = str(context.get("last_output", ""))
            items = [x.strip() for x in raw.split(separator) if x.strip()]
            items = items[:max_iter]
            for i, item in enumerate(items):
                context["loop_iteration"] = i
                context["loop_item"] = item
                context["loop_total"] = len(items)
                context["last_output"] = item  # each item as input
                logger.info("Loop [foreach] item %d/%d: %s", i + 1, len(items), item[:50])
                await self._execute_body(body_edges, workflow, context, result)
                iterations_output.append(str(context.get("last_output", "")))

        elif loop_type == "while":
            for i in range(max_iter):
                context["loop_iteration"] = i
                context["loop_total"] = max_iter
                logger.info("Loop [while] iteration %d/%d", i + 1, max_iter)
                await self._execute_body(body_edges, workflow, context, result)
                iterations_output.append(str(context.get("last_output", "")))
                if self._evaluate_loop_condition(cfg, context):
                    logger.info("Loop [while] condition met at iteration %d", i + 1)
                    break

        # Store results
        context["loop_results"] = iterations_output
        combined = "\n\n".join(iterations_output)
        context["last_output"] = combined

        # Follow done edges
        for edge in done_edges:
            next_node = workflow.get_node(edge.target)
            if next_node:
                await self._execute_node(workflow, next_node, context, result)

        return combined

    async def _execute_body(
        self,
        body_edges: list[Edge],
        workflow: Workflow,
        context: dict,
        result: WorkflowExecutionResult,
    ) -> None:
        """Execute the body sub-graph of a loop once."""
        for edge in body_edges:
            next_node = workflow.get_node(edge.target)
            if next_node:
                await self._execute_node(workflow, next_node, context, result)

    def _evaluate_loop_condition(self, cfg: dict, context: dict) -> bool:
        """Evaluate whether a while-loop should stop (returns True to stop)."""
        condition_type = cfg.get("condition_type", "confidence")
        threshold = float(cfg.get("threshold", 0.8))
        last_output = context.get("last_output", "")

        if condition_type == "confidence":
            if isinstance(last_output, dict):
                return last_output.get("confidence", 0) >= threshold
            return False  # no confidence data → keep looping

        elif condition_type == "keyword":
            keyword = cfg.get("keyword", "")
            return keyword.lower() in str(last_output).lower() if keyword else True

        elif condition_type == "length":
            return len(str(last_output)) >= int(threshold * 1000)

        return True  # unknown type → stop
