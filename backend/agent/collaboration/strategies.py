"""Collaboration strategies for multi-agent execution."""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

from agent.collaboration.protocols import AgentMessage, MessageType, SharedContext


@dataclass
class AgentHandle:
    """Lightweight handle to an agent's execute function."""
    agent_id: str
    execute_fn: Callable  # async (input_data) -> result
    capabilities: list[str] = field(default_factory=list)


@dataclass
class StrategyResult:
    """Result from a collaboration strategy execution."""
    final_result: Any
    agent_results: dict[str, Any] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


class CollaborationStrategy(ABC):
    """Base class for collaboration strategies."""

    @abstractmethod
    async def execute(self, agents: list[AgentHandle], task: Any) -> StrategyResult:
        """Execute the strategy with the given agents and task.

        Args:
            agents: List of agent handles to collaborate.
            task: The task input data.

        Returns:
            StrategyResult with combined output.
        """
        ...


class SequentialStrategy(CollaborationStrategy):
    """Agents execute in order, each receiving the previous agent's output.

    Agent 1 processes input -> Agent 2 processes Agent 1's output -> ...
    The final agent's output is the result.
    """

    async def execute(self, agents: list[AgentHandle], task: Any) -> StrategyResult:
        if not agents:
            return StrategyResult(final_result=None)

        current_input = task
        agent_results = {}

        for agent in agents:
            result = await agent.execute_fn(current_input)
            agent_results[agent.agent_id] = result
            current_input = result

        return StrategyResult(
            final_result=current_input,
            agent_results=agent_results,
            metadata={"strategy": "sequential", "agent_count": len(agents)},
        )


class ParallelStrategy(CollaborationStrategy):
    """Agents execute simultaneously on the same input.

    All agents receive the same task. Results are merged.
    An optional merge function can be provided; otherwise results are
    returned as a dict keyed by agent_id.
    """

    def __init__(self, merge_fn: Callable | None = None):
        """
        Args:
            merge_fn: Optional function(dict[agent_id, result]) -> merged_result.
        """
        self.merge_fn = merge_fn

    async def execute(self, agents: list[AgentHandle], task: Any) -> StrategyResult:
        if not agents:
            return StrategyResult(final_result=None)

        async def _run(agent: AgentHandle):
            try:
                return agent.agent_id, await agent.execute_fn(task)
            except Exception as e:
                return agent.agent_id, {"error": str(e)}

        results = await asyncio.gather(*[_run(a) for a in agents])
        agent_results = dict(results)

        if self.merge_fn:
            final = self.merge_fn(agent_results)
        else:
            final = agent_results

        return StrategyResult(
            final_result=final,
            agent_results=agent_results,
            metadata={"strategy": "parallel", "agent_count": len(agents)},
        )


class DebateStrategy(CollaborationStrategy):
    """Agents discuss and iterate toward consensus.

    Round 1: All agents independently process the task.
    Round 2..N: Each agent sees all previous results and refines.
    Final: A judge function or voting picks the best answer.

    Args:
        rounds: Number of debate rounds (default 2).
        judge_fn: Optional function(dict[agent_id, result]) -> winner_result.
    """

    def __init__(self, rounds: int = 2, judge_fn: Callable | None = None):
        self.rounds = max(1, rounds)
        self.judge_fn = judge_fn

    async def execute(self, agents: list[AgentHandle], task: Any) -> StrategyResult:
        if not agents:
            return StrategyResult(final_result=None)

        all_rounds: list[dict[str, Any]] = []

        # Round 1: independent
        round_results = {}
        for agent in agents:
            try:
                round_results[agent.agent_id] = await agent.execute_fn(task)
            except Exception as e:
                round_results[agent.agent_id] = {"error": str(e)}
        all_rounds.append(round_results)

        # Subsequent rounds: agents see previous results
        for round_num in range(1, self.rounds):
            context = {
                "original_task": task,
                "previous_round": all_rounds[-1],
                "round": round_num + 1,
            }
            round_results = {}
            for agent in agents:
                try:
                    round_results[agent.agent_id] = await agent.execute_fn(context)
                except Exception as e:
                    round_results[agent.agent_id] = {"error": str(e)}
            all_rounds.append(round_results)

        final_results = all_rounds[-1]

        # Judge or pick first non-error result
        if self.judge_fn:
            final = self.judge_fn(final_results)
        else:
            final = next(
                (v for v in final_results.values() if not isinstance(v, dict) or "error" not in v),
                final_results,
            )

        return StrategyResult(
            final_result=final,
            agent_results=final_results,
            metadata={
                "strategy": "debate",
                "rounds": self.rounds,
                "all_rounds": all_rounds,
            },
        )


class VotingStrategy(CollaborationStrategy):
    """Agents vote on the best answer.

    All agents independently produce a result, then each agent
    votes on which result is best.

    Args:
        vote_fn: Function(agent_result) -> vote_key used for tallying.
                 If None, results are compared by string equality.
    """

    def __init__(self, vote_fn: Callable | None = None):
        self.vote_fn = vote_fn or (lambda x: str(x))

    async def execute(self, agents: list[AgentHandle], task: Any) -> StrategyResult:
        if not agents:
            return StrategyResult(final_result=None)

        # Collect all results
        agent_results = {}
        for agent in agents:
            try:
                agent_results[agent.agent_id] = await agent.execute_fn(task)
            except Exception as e:
                agent_results[agent.agent_id] = {"error": str(e)}

        # Tally votes
        votes: dict[str, int] = {}
        vote_to_result: dict[str, Any] = {}
        for agent_id, result in agent_results.items():
            if isinstance(result, dict) and "error" in result:
                continue
            key = self.vote_fn(result)
            votes[key] = votes.get(key, 0) + 1
            vote_to_result[key] = result

        if not votes:
            # All failed
            return StrategyResult(
                final_result=None,
                agent_results=agent_results,
                metadata={"strategy": "voting", "votes": {}, "error": "all_agents_failed"},
            )

        # Winner is the result with the most votes
        winner_key = max(votes, key=lambda k: votes[k])
        winner_result = vote_to_result[winner_key]

        return StrategyResult(
            final_result=winner_result,
            agent_results=agent_results,
            metadata={
                "strategy": "voting",
                "votes": votes,
                "winner_key": winner_key,
                "agent_count": len(agents),
            },
        )
