from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class Message:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)


class LLMClient(Protocol):
    def complete(self, messages: list[dict], tools: list[dict]) -> Message: ...


class FakeLLM:
    """Test double: returns scripted messages in order."""

    def __init__(self, scripted: list[Message]):
        self._scripted = list(scripted)

    def complete(self, messages: list[dict], tools: list[dict]) -> Message:
        return self._scripted.pop(0)
