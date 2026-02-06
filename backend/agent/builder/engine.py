"""Workflow execution engine: runs DAG-based agent workflows."""

import time
from typing import Any

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
            output = await self._run_node(node.config, context)
            node_result.output = output
            node_result.status = ExecutionStatus.COMPLETED
            context["results"][node.id] = output
            context["last_output"] = output
        except Exception as e:
            node_result.status = ExecutionStatus.FAILED
            node_result.error = str(e)
            raise
        finally:
            node_result.duration_ms = int((time.time() - start_time) * 1000)
            result.node_results.append(node_result)

        # Traverse outgoing edges
        edges = workflow.get_outgoing_edges(node.id)
        for edge in edges:
            # Check condition if present
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
        input_text = str(context.get("last_output", context.get("input", {}).get("query", "")))
        prompt = prompt.replace("{input}", input_text)

        system = cfg.get("system_prompt", "")
        temperature = cfg.get("temperature", 0.7)

        response = await self.llm.generate(
            prompt=prompt,
            system=system or None,
            temperature=temperature,
        )
        return response.content

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
