"""Communication protocols for inter-agent messaging."""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class MessageType(str, Enum):
    """Types of inter-agent messages."""
    REQUEST = "request"
    RESPONSE = "response"
    BROADCAST = "broadcast"
    NOTIFY = "notify"
    ERROR = "error"


@dataclass
class AgentMessage:
    """Message for inter-agent communication."""
    sender: str
    receiver: str  # Use "*" for broadcast
    content: Any
    msg_type: MessageType = MessageType.REQUEST
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    correlation_id: str | None = None  # Links request to response
    metadata: dict = field(default_factory=dict)

    def create_response(self, content: Any, **metadata) -> "AgentMessage":
        """Create a response message linked to this request."""
        return AgentMessage(
            sender=self.receiver,
            receiver=self.sender,
            content=content,
            msg_type=MessageType.RESPONSE,
            correlation_id=self.id,
            metadata=metadata,
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "sender": self.sender,
            "receiver": self.receiver,
            "content": self.content,
            "msg_type": self.msg_type.value,
            "timestamp": self.timestamp,
            "correlation_id": self.correlation_id,
            "metadata": self.metadata,
        }


class SharedContext:
    """Shared blackboard for inter-agent data sharing.

    Agents can read/write to a shared key-value store with
    optional namespacing and event notifications.
    """

    def __init__(self):
        self._data: dict[str, Any] = {}
        self._history: list[dict] = []
        self._listeners: dict[str, list] = {}  # key -> list of callbacks

    def set(self, key: str, value: Any, agent_id: str = "") -> None:
        """Set a value in the shared context."""
        self._data[key] = value
        entry = {
            "action": "set",
            "key": key,
            "agent_id": agent_id,
            "timestamp": datetime.utcnow().isoformat(),
        }
        self._history.append(entry)
        self._notify_listeners(key, value)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from the shared context."""
        return self._data.get(key, default)

    def has(self, key: str) -> bool:
        """Check if a key exists."""
        return key in self._data

    def keys(self) -> list[str]:
        """List all keys."""
        return list(self._data.keys())

    def get_all(self) -> dict[str, Any]:
        """Get a copy of all data."""
        return dict(self._data)

    def delete(self, key: str) -> None:
        """Remove a key from context."""
        self._data.pop(key, None)

    def clear(self) -> None:
        """Clear all data."""
        self._data.clear()
        self._history.clear()

    def subscribe(self, key: str, callback) -> None:
        """Subscribe to changes on a key."""
        if key not in self._listeners:
            self._listeners[key] = []
        self._listeners[key].append(callback)

    def unsubscribe(self, key: str, callback) -> None:
        """Unsubscribe from changes on a key."""
        if key in self._listeners:
            self._listeners[key] = [cb for cb in self._listeners[key] if cb is not callback]

    def _notify_listeners(self, key: str, value: Any) -> None:
        """Notify listeners of a change."""
        for callback in self._listeners.get(key, []):
            try:
                callback(key, value)
            except Exception:
                pass  # Don't let listener errors break the context

    def get_history(self) -> list[dict]:
        """Get change history."""
        return list(self._history)


class MessageBus:
    """Simple in-memory message bus for agent communication."""

    def __init__(self):
        self._queues: dict[str, list[AgentMessage]] = {}
        self._broadcast_listeners: list = []

    def send(self, message: AgentMessage) -> None:
        """Send a message to a specific agent or broadcast."""
        if message.receiver == "*":
            for queue in self._queues.values():
                queue.append(message)
            for listener in self._broadcast_listeners:
                try:
                    listener(message)
                except Exception:
                    pass
        else:
            if message.receiver not in self._queues:
                self._queues[message.receiver] = []
            self._queues[message.receiver].append(message)

    def receive(self, agent_id: str) -> list[AgentMessage]:
        """Receive all pending messages for an agent."""
        messages = self._queues.pop(agent_id, [])
        return messages

    def peek(self, agent_id: str) -> int:
        """Check how many messages are pending for an agent."""
        return len(self._queues.get(agent_id, []))

    def on_broadcast(self, callback) -> None:
        """Register a callback for broadcast messages."""
        self._broadcast_listeners.append(callback)

    def clear(self, agent_id: str | None = None) -> None:
        """Clear messages for an agent or all agents."""
        if agent_id:
            self._queues.pop(agent_id, None)
        else:
            self._queues.clear()
