import json
import os
import re
import time
from voicedesk.llm import Message, ToolCall, LLMError

DEFAULT_MODEL = "llama-3.3-70b-versatile"
MAX_BACKOFF_S = 60.0


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


def _is_rate_limited(exc: Exception) -> bool:
    """True when Groq rejected the request for exceeding the free-tier rate
    limit. Worth waiting out — unlike auth or bad-request errors."""
    if getattr(exc, "status_code", None) == 429:
        return True
    if getattr(exc, "code", None) == "rate_limit_exceeded":
        return True
    text = str(exc).lower()
    return "rate limit" in text or "429" in text


def _retry_after_seconds(exc: Exception) -> float | None:
    """How long Groq told us to wait, from the Retry-After header or the error
    body ("Please try again in 7.66s"). None when it didn't say."""
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if headers is not None:
        try:
            value = headers.get("retry-after")
        except Exception:  # noqa: BLE001 - headers may not be dict-like
            value = None
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    match = re.search(r"try again in ([\d.]+)s", str(exc))
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None


class GroqLLM:
    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        client=None,
        max_retries: int = 3,
        rate_limit_retries: int = 6,
        backoff_base: float = 2.0,
        on_retry: "callable | None" = None,
    ):
        # Model is configurable via GROQ_MODEL so a different model can be tried
        # without code changes if tool-calling reliability is poor.
        self.model = model or os.environ.get("GROQ_MODEL", DEFAULT_MODEL)
        self.max_retries = max_retries
        self.rate_limit_retries = rate_limit_retries
        self.backoff_base = backoff_base
        self.on_retry = on_retry
        if client is not None:
            self.client = client  # injected (used by tests — no network/key)
        else:
            from groq import Groq  # imported lazily so tests don't need the package
            self.client = Groq(api_key=api_key or os.environ["GROQ_API_KEY"])

    def complete(self, messages: list[dict], tools: list[dict]) -> Message:
        tool_attempts = 0
        rate_attempts = 0
        while True:
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                )
                return _to_message(resp.choices[0])
            except Exception as e:  # noqa: BLE001 - translated to LLMError below
                # Rate limits: wait exactly as long as Groq asked, then retry.
                if _is_rate_limited(e) and rate_attempts < self.rate_limit_retries:
                    rate_attempts += 1
                    wait = _retry_after_seconds(e)
                    if wait is None:
                        wait = self.backoff_base * (2 ** (rate_attempts - 1))
                    wait = min(wait, MAX_BACKOFF_S)
                    self._notify("rate_limited", wait, rate_attempts)
                    time.sleep(wait)
                    continue
                # Malformed tool calls are non-deterministic; resample at once.
                if _is_tool_use_failed(e) and tool_attempts < self.max_retries - 1:
                    tool_attempts += 1
                    self._notify("tool_use_failed", 0.0, tool_attempts)
                    continue
                # Everything else (auth, bad request, ...) fails fast.
                raise LLMError(str(e)) from e

    def _notify(self, reason: str, wait_s: float, attempt: int) -> None:
        if self.on_retry is None:
            return
        try:
            self.on_retry(reason, wait_s, attempt)
        except Exception:  # a broken callback must never break the run
            pass
