"""Agent coordinator: manages multiple agent instances and task execution."""

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from agent.collaboration.protocols import AgentMessage, MessageBus, MessageType, SharedContext


class TaskPriority(int, Enum):
    """Task priority levels."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


class TaskStatus(str, Enum):
    """Task execution status."""
    PENDING = "pending"
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AgentStatus(str, Enum):
    """Agent runtime status."""
    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"
    OFFLINE = "offline"


@dataclass
class Task:
    """A unit of work to be assigned to an agent."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: str = ""
    input_data: Any = None
    priority: TaskPriority = TaskPriority.NORMAL
    status: TaskStatus = TaskStatus.PENDING
    assigned_agent: str | None = None
    result: Any = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def duration_ms(self) -> int | None:
        if self.started_at and self.completed_at:
            return int((self.completed_at - self.started_at) * 1000)
        return None


@dataclass
class AgentInfo:
    """Registered agent information."""
    agent_id: str
    agent_type: str
    capabilities: list[str] = field(default_factory=list)
    status: AgentStatus = AgentStatus.IDLE
    current_task_id: str | None = None
    execute_fn: Any = None  # async callable(input_data) -> result
    metadata: dict = field(default_factory=dict)


class AgentCoordinator:
    """Coordinates multiple agents, manages task queue and execution.

    Provides task assignment, result collection, parallel execution,
    and agent status tracking.
    """

    def __init__(self):
        self._agents: dict[str, AgentInfo] = {}
        self._tasks: dict[str, Task] = {}
        self._task_queue: list[str] = []  # task IDs sorted by priority
        self._context = SharedContext()
        self._message_bus = MessageBus()
        self._on_task_complete: list[Callable] = []

    # ---- Agent Management ----

    def register_agent(
        self,
        agent_id: str,
        agent_type: str,
        capabilities: list[str] | None = None,
        execute_fn=None,
        **metadata,
    ) -> None:
        """Register an agent with the coordinator."""
        self._agents[agent_id] = AgentInfo(
            agent_id=agent_id,
            agent_type=agent_type,
            capabilities=capabilities or [],
            execute_fn=execute_fn,
            metadata=metadata,
        )

    def unregister_agent(self, agent_id: str) -> None:
        """Remove an agent from the coordinator."""
        self._agents.pop(agent_id, None)

    def get_agent(self, agent_id: str) -> AgentInfo | None:
        """Get agent info by ID."""
        return self._agents.get(agent_id)

    def list_agents(self) -> list[AgentInfo]:
        """List all registered agents."""
        return list(self._agents.values())

    def get_idle_agents(self) -> list[AgentInfo]:
        """Get all idle agents."""
        return [a for a in self._agents.values() if a.status == AgentStatus.IDLE]

    # ---- Task Management ----

    def create_task(
        self,
        description: str,
        input_data: Any = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        **metadata,
    ) -> Task:
        """Create a new task and add it to the queue."""
        task = Task(
            description=description,
            input_data=input_data,
            priority=priority,
            metadata=metadata,
        )
        self._tasks[task.id] = task
        self._enqueue(task.id)
        return task

    def get_task(self, task_id: str) -> Task | None:
        """Get a task by ID."""
        return self._tasks.get(task_id)

    def list_tasks(self, status: TaskStatus | None = None) -> list[Task]:
        """List tasks, optionally filtered by status."""
        tasks = list(self._tasks.values())
        if status:
            tasks = [t for t in tasks if t.status == status]
        return tasks

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending or assigned task."""
        task = self._tasks.get(task_id)
        if not task:
            return False
        if task.status in (TaskStatus.PENDING, TaskStatus.ASSIGNED):
            task.status = TaskStatus.CANCELLED
            if task.id in self._task_queue:
                self._task_queue.remove(task.id)
            if task.assigned_agent:
                agent = self._agents.get(task.assigned_agent)
                if agent:
                    agent.status = AgentStatus.IDLE
                    agent.current_task_id = None
            return True
        return False

    # ---- Task Assignment ----

    def assign_task(self, agent_id: str, task_id: str) -> bool:
        """Assign a specific task to a specific agent."""
        task = self._tasks.get(task_id)
        agent = self._agents.get(agent_id)
        if not task or not agent:
            return False
        if agent.status != AgentStatus.IDLE:
            return False
        if task.status != TaskStatus.PENDING:
            return False

        task.status = TaskStatus.ASSIGNED
        task.assigned_agent = agent_id
        agent.status = AgentStatus.BUSY
        agent.current_task_id = task_id

        if task.id in self._task_queue:
            self._task_queue.remove(task.id)

        return True

    def auto_assign(self) -> list[tuple[str, str]]:
        """Auto-assign pending tasks to idle agents. Returns list of (agent_id, task_id) pairs."""
        assignments = []
        idle_agents = self.get_idle_agents()

        for agent in idle_agents:
            task_id = self._dequeue_for_agent(agent)
            if task_id:
                self.assign_task(agent.agent_id, task_id)
                assignments.append((agent.agent_id, task_id))
        return assignments

    # ---- Execution ----

    async def execute_task(self, task_id: str) -> Any:
        """Execute a single assigned task."""
        task = self._tasks.get(task_id)
        if not task or not task.assigned_agent:
            raise ValueError(f"Task {task_id} not assigned to any agent")

        agent = self._agents.get(task.assigned_agent)
        if not agent or not agent.execute_fn:
            raise ValueError(f"Agent {task.assigned_agent} has no execute function")

        task.status = TaskStatus.RUNNING
        task.started_at = time.time()

        try:
            result = await agent.execute_fn(task.input_data)
            task.result = result
            task.status = TaskStatus.COMPLETED
            task.completed_at = time.time()
            agent.status = AgentStatus.IDLE
            agent.current_task_id = None

            # Notify callbacks
            for callback in self._on_task_complete:
                try:
                    callback(task)
                except Exception:
                    pass

            return result
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.completed_at = time.time()
            agent.status = AgentStatus.ERROR
            agent.current_task_id = None
            raise

    async def execute_all_pending(self) -> dict[str, Any]:
        """Auto-assign and execute all pending tasks in parallel."""
        self.auto_assign()
        assigned_tasks = self.list_tasks(status=TaskStatus.ASSIGNED)

        if not assigned_tasks:
            return {}

        results = {}
        tasks_coro = []
        for task in assigned_tasks:
            tasks_coro.append(self._safe_execute(task.id))

        completed = await asyncio.gather(*tasks_coro, return_exceptions=True)
        for task, result in zip(assigned_tasks, completed):
            if isinstance(result, Exception):
                results[task.id] = {"error": str(result)}
            else:
                results[task.id] = result

        return results

    async def _safe_execute(self, task_id: str) -> Any:
        """Execute a task, catching exceptions for gather."""
        try:
            return await self.execute_task(task_id)
        except Exception as e:
            return e

    def collect_results(self, task_ids: list[str] | None = None) -> dict[str, Any]:
        """Collect results from completed tasks."""
        if task_ids is None:
            tasks = self.list_tasks(status=TaskStatus.COMPLETED)
        else:
            tasks = [self._tasks[tid] for tid in task_ids if tid in self._tasks]

        return {
            task.id: {
                "result": task.result,
                "status": task.status.value,
                "duration_ms": task.duration_ms,
                "agent": task.assigned_agent,
            }
            for task in tasks
        }

    # ---- Event Hooks ----

    def on_task_complete(self, callback: Callable) -> None:
        """Register a callback for task completion."""
        self._on_task_complete.append(callback)

    # ---- Properties ----

    @property
    def context(self) -> SharedContext:
        return self._context

    @property
    def message_bus(self) -> MessageBus:
        return self._message_bus

    # ---- Internal ----

    def _enqueue(self, task_id: str) -> None:
        """Insert task into priority queue."""
        task = self._tasks[task_id]
        # Higher priority first
        insert_idx = 0
        for i, tid in enumerate(self._task_queue):
            queued = self._tasks.get(tid)
            if queued and queued.priority.value >= task.priority.value:
                insert_idx = i + 1
            else:
                break
        self._task_queue.insert(insert_idx, task_id)

    def _dequeue_for_agent(self, agent: AgentInfo) -> str | None:
        """Get the next suitable task for an agent from the queue."""
        for task_id in self._task_queue:
            task = self._tasks.get(task_id)
            if not task or task.status != TaskStatus.PENDING:
                continue
            # Check capability match if task has required capabilities
            required = task.metadata.get("required_capabilities", [])
            if required and not all(c in agent.capabilities for c in required):
                continue
            return task_id
        return None

    def get_stats(self) -> dict:
        """Get coordinator statistics."""
        all_tasks = list(self._tasks.values())
        return {
            "total_agents": len(self._agents),
            "idle_agents": len(self.get_idle_agents()),
            "total_tasks": len(all_tasks),
            "pending_tasks": len([t for t in all_tasks if t.status == TaskStatus.PENDING]),
            "running_tasks": len([t for t in all_tasks if t.status == TaskStatus.RUNNING]),
            "completed_tasks": len([t for t in all_tasks if t.status == TaskStatus.COMPLETED]),
            "failed_tasks": len([t for t in all_tasks if t.status == TaskStatus.FAILED]),
        }
