import json
import os
from voicedesk.llm import Message, ToolCall, LLMError

DEFAULT_MODEL = "llama-3.3-70b-versatile"


def _to_message(choice) -> Message:
    m = choice.message
    calls = []
    for tc in (m.tool_calls or []):
        calls.append(ToolCall(
            id=tc.id,
            name=tc.function.name,
            arguments=json.loads(tc.function.arguments or "{}"),
        ))
    return Message(content=m.content, tool_calls=calls)


def _is_tool_use_failed(exc: Exception) -> bool:
    """True when Groq rejected a malformed tool call (code 'tool_use_failed').
    Detected without importing groq's exception types, so this stays testable
    and provider-agnostic. This failure is non-deterministic — a resample of
    the same request usually produces a valid structured tool call."""
    if getattr(exc, "code", None) == "tool_use_failed":
        return True
    return "tool_use_failed" in str(exc)


class GroqLLM:
    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        client=None,
        max_retries: int = 3,
    ):
        # Model is configurable via GROQ_MODEL so a different model can be tried
        # without code changes if tool-calling reliability is poor.
        self.model = model or os.environ.get("GROQ_MODEL", DEFAULT_MODEL)
        self.max_retries = max_retries
        if client is not None:
            self.client = client  # injected (used by tests — no network/key)
        else:
            from groq import Groq  # imported lazily so tests don't need the package
            self.client = Groq(api_key=api_key or os.environ["GROQ_API_KEY"])

    def complete(self, messages: list[dict], tools: list[dict]) -> Message:
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                )
                return _to_message(resp.choices[0])
            except Exception as e:  # noqa: BLE001 - translated to LLMError below
                last_exc = e
                # Retry only the transient malformed-tool-call case; fail fast
                # on everything else (auth, network, bad request, ...).
                if _is_tool_use_failed(e) and attempt < self.max_retries - 1:
                    continue
                raise LLMError(str(e)) from e
        raise LLMError(str(last_exc)) from last_exc
