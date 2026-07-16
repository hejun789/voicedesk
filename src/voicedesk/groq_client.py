import json
import os
import re
import time
from voicedesk.llm import Message, ToolCall, LLMError, QuotaExhausted

DEFAULT_MODEL = "llama-3.3-70b-versatile"
MAX_BACKOFF_S = 900.0
TOKEN_HEADROOM = 3000  # roughly one agent call's worth of tokens
# A per-minute token bucket can, when deeply depleted, legitimately ask for a
# wait of a few minutes and WILL recover on its own — waiting it out is the
# right move. Only a wait beyond ~15 minutes is implausible for a per-minute
# bucket and instead indicates a long-window/daily quota, which retrying
# cannot fix. (Also used to detect that condition alongside the remaining-
# requests counter — see _remaining_requests.)
QUOTA_EXHAUSTED_WAIT_S = 900.0

_DURATION_RE = re.compile(r"(\d+(?:\.\d+)?)(ms|s|m|h)")


def _parse_duration(text: str | None) -> float:
    """Parse a Groq reset string ('370ms', '6s', '1m26.4s') into seconds.
    Returns 0.0 when it cannot be parsed."""
    if not text:
        return 0.0
    seconds = 0.0
    for amount, unit in _DURATION_RE.findall(str(text)):
        value = float(amount)
        seconds += {"ms": value / 1000, "s": value, "m": value * 60,
                    "h": value * 3600}[unit]
    return seconds


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


def _remaining_requests(exc: Exception) -> float | None:
    """Requests left in the provider's long-window (daily) budget, from the
    response headers. None when it did not say."""
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if headers is None:
        return None
    try:
        value = headers.get("x-ratelimit-remaining-requests")
    except Exception:  # noqa: BLE001 - headers may not be dict-like
        return None
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class GroqLLM:
    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        client=None,
        # Groq's Llama models intermittently emit a malformed <function=..> tool
        # call that the API itself rejects. It is non-deterministic, so a resample
        # usually fixes it — but two attempts is not enough in practice, and an
        # exhausted budget makes the agent give up mid-call.
        max_retries: int = 6,
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
        self._remaining_tokens: float | None = None
        self._tokens_reset_s: float = 0.0
        if client is not None:
            self.client = client  # injected (used by tests — no network/key)
        else:
            from groq import Groq  # imported lazily so tests don't need the package
            self.client = Groq(api_key=api_key or os.environ["GROQ_API_KEY"])

    def _update_limits(self, headers) -> None:
        """Record the rate-limit state Groq reported on the last response."""
        if not headers:
            return
        try:
            remaining = headers.get("x-ratelimit-remaining-tokens")
            reset = headers.get("x-ratelimit-reset-tokens")
        except Exception:  # noqa: BLE001 - headers may not be dict-like
            return
        if remaining is not None:
            try:
                self._remaining_tokens = float(remaining)
            except (TypeError, ValueError):
                self._remaining_tokens = None
        self._tokens_reset_s = _parse_duration(reset)

    def _throttle(self) -> None:
        """Wait BEFORE sending if the token bucket is nearly empty, so we never
        provoke a 429 (each 429 still burns a request from the daily quota)."""
        if self._remaining_tokens is None:
            return
        if self._remaining_tokens >= TOKEN_HEADROOM:
            return
        wait = min(self._tokens_reset_s + 0.5, MAX_BACKOFF_S)
        if wait <= 0:
            return
        self._notify("throttle", wait, 0)
        time.sleep(wait)
        self._remaining_tokens = None  # unknown until the next response

    def _create(self, messages: list[dict], tools: list[dict]):
        """Returns (choice, headers | None)."""
        kwargs = dict(model=self.model, messages=messages, tools=tools,
                      tool_choice="auto")
        raw = getattr(self.client.chat.completions, "with_raw_response", None)
        if raw is not None:
            response = raw.create(**kwargs)
            parsed = response.parse()
            return parsed.choices[0], response.headers
        resp = self.client.chat.completions.create(**kwargs)
        return resp.choices[0], None

    def complete(self, messages: list[dict], tools: list[dict]) -> Message:
        tool_attempts = 0
        rate_attempts = 0
        while True:
            try:
                self._throttle()
                choice, headers = self._create(messages, tools)
                self._update_limits(headers)
                return _to_message(choice)
            except Exception as e:  # noqa: BLE001 - translated to LLMError below
                # Rate limits: wait exactly as long as Groq asked, then retry.
                if _is_rate_limited(e) and rate_attempts < self.rate_limit_retries:
                    rate_attempts += 1
                    self._update_limits(getattr(getattr(e, "response", None), "headers", None))
                    requested = _retry_after_seconds(e)
                    remaining_reqs = _remaining_requests(e)
                    # A spent long-window (daily) request budget cannot be waited out.
                    if remaining_reqs is not None and remaining_reqs < 1:
                        raise QuotaExhausted(
                            f"Groq reports 0 requests remaining for {self.model!r} — "
                            f"the daily request quota is spent. Retrying will not help; "
                            f"try again later or switch GROQ_MODEL."
                        ) from e
                    # An implausibly long wait means a long-window limit, not the
                    # per-minute token bucket (which recovers in seconds-to-minutes).
                    if requested is not None and requested > QUOTA_EXHAUSTED_WAIT_S:
                        raise QuotaExhausted(
                            f"Groq asked for a {requested:.0f}s wait for {self.model!r} — "
                            f"longer than a per-minute bucket ever needs, so a long-window "
                            f"quota is spent. Retrying will not help; try again later or "
                            f"switch GROQ_MODEL."
                        ) from e
                    wait = requested
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
