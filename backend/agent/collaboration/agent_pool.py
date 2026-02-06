"""Agent pool manager for reusable agent instances."""

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class PooledAgent:
    """An agent managed by the pool."""
    agent_id: str
    agent_type: str
    capabilities: list[str] = field(default_factory=list)
    execute_fn: Callable | None = None  # async (input_data) -> result
    is_available: bool = True
    metadata: dict = field(default_factory=dict)


class AgentPool:
    """Manages a pool of reusable agent instances.

    Supports capability-based routing, pool sizing, and
    lifecycle management.
    """

    def __init__(self, max_size: int = 10):
        self._agents: dict[str, PooledAgent] = {}
        self._max_size = max_size

    def register_agent(
        self,
        agent_id: str,
        agent_type: str,
        capabilities: list[str] | None = None,
        execute_fn: Callable | None = None,
        **metadata,
    ) -> bool:
        """Register an agent in the pool.

        Returns:
            True if registered, False if pool is full.
        """
        if len(self._agents) >= self._max_size and agent_id not in self._agents:
            return False

        self._agents[agent_id] = PooledAgent(
            agent_id=agent_id,
            agent_type=agent_type,
            capabilities=capabilities or [],
            execute_fn=execute_fn,
            metadata=metadata,
        )
        return True

    def unregister_agent(self, agent_id: str) -> bool:
        """Remove an agent from the pool."""
        return self._agents.pop(agent_id, None) is not None

    def get_agent(self, agent_id: str) -> PooledAgent | None:
        """Get a specific agent by ID."""
        return self._agents.get(agent_id)

    def get_available_agent(self, required_capabilities: list[str] | None = None) -> PooledAgent | None:
        """Get the first available agent matching required capabilities.

        Args:
            required_capabilities: List of capabilities the agent must have.

        Returns:
            A matching available agent, or None.
        """
        required = required_capabilities or []
        for agent in self._agents.values():
            if not agent.is_available:
                continue
            if required and not all(cap in agent.capabilities for cap in required):
                continue
            return agent
        return None

    def get_all_available(self, required_capabilities: list[str] | None = None) -> list[PooledAgent]:
        """Get all available agents matching capabilities."""
        required = required_capabilities or []
        result = []
        for agent in self._agents.values():
            if not agent.is_available:
                continue
            if required and not all(cap in agent.capabilities for cap in required):
                continue
            result.append(agent)
        return result

    def acquire(self, agent_id: str) -> bool:
        """Mark an agent as busy (not available)."""
        agent = self._agents.get(agent_id)
        if agent and agent.is_available:
            agent.is_available = False
            return True
        return False

    def release(self, agent_id: str) -> bool:
        """Mark an agent as available again."""
        agent = self._agents.get(agent_id)
        if agent:
            agent.is_available = True
            return True
        return False

    def list_agents(self, agent_type: str | None = None) -> list[PooledAgent]:
        """List all agents, optionally filtered by type."""
        agents = list(self._agents.values())
        if agent_type:
            agents = [a for a in agents if a.agent_type == agent_type]
        return agents

    def list_capabilities(self) -> list[str]:
        """List all unique capabilities across all agents."""
        caps = set()
        for agent in self._agents.values():
            caps.update(agent.capabilities)
        return sorted(caps)

    @property
    def size(self) -> int:
        """Current pool size."""
        return len(self._agents)

    @property
    def available_count(self) -> int:
        """Number of available agents."""
        return sum(1 for a in self._agents.values() if a.is_available)

    @property
    def max_size(self) -> int:
        return self._max_size

    def resize(self, new_max: int) -> None:
        """Resize the pool max capacity."""
        self._max_size = max(1, new_max)

    def get_stats(self) -> dict:
        """Get pool statistics."""
        agents = list(self._agents.values())
        types = {}
        for a in agents:
            types[a.agent_type] = types.get(a.agent_type, 0) + 1

        return {
            "total": len(agents),
            "available": self.available_count,
            "busy": len(agents) - self.available_count,
            "max_size": self._max_size,
            "types": types,
            "capabilities": self.list_capabilities(),
        }
