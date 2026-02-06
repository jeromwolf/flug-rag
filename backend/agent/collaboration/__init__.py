"""Multi-agent collaboration framework."""

from .agent_pool import AgentPool
from .coordinator import AgentCoordinator, TaskPriority, TaskStatus
from .protocols import AgentMessage, MessageType, SharedContext
from .strategies import (
    CollaborationStrategy,
    DebateStrategy,
    ParallelStrategy,
    SequentialStrategy,
    VotingStrategy,
)

__all__ = [
    "AgentCoordinator",
    "AgentMessage",
    "AgentPool",
    "CollaborationStrategy",
    "DebateStrategy",
    "MessageType",
    "ParallelStrategy",
    "SequentialStrategy",
    "SharedContext",
    "TaskPriority",
    "TaskStatus",
    "VotingStrategy",
]
