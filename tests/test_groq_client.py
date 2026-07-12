import pytest
from types import SimpleNamespace
from voicedesk.groq_client import _to_message, GroqLLM
from voicedesk.llm import LLMError, QuotaExhausted


def _fake_choice(content=None, tool_calls=None):
    msg = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(message=msg)


def _tool_use_failed_error():
    e = Exception("400 tool call validation failed ... tool_use_failed")
    e.code = "tool_use_failed"
    return e


class _FakeGroqClient:
    """Stands in for the groq.Groq client. `behaviors` is a list of either an
    Exception to raise or a response object to return, consumed per create()."""

    def __init__(self, behaviors):
        self.behaviors = list(behaviors)
        self.calls = 0
        outer = self
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kw: outer._next())
        )

    def _next(self):
        b = self.behaviors[self.calls]
        self.calls += 1
        if isinstance(b, Exception):
            raise b
        return b


def test_to_message_plain_text():
    msg = _to_message(_fake_choice(content="hi there"))
    assert msg.content == "hi there"
    assert msg.tool_calls == []


def test_to_message_with_tool_call():
    tc = SimpleNamespace(
        id="call_1",
        function=SimpleNamespace(name="cancel", arguments='{"appointment_id": 3}'),
    )
    msg = _to_message(_fake_choice(content=None, tool_calls=[tc]))
    assert msg.tool_calls[0].name == "cancel"
    assert msg.tool_calls[0].arguments == {"appointment_id": 3}
    assert msg.tool_calls[0].id == "call_1"


def test_complete_retries_on_tool_use_failed_then_succeeds():
    good = SimpleNamespace(choices=[_fake_choice(content="booked")])
    client = _FakeGroqClient([_tool_use_failed_error(), good])
    llm = GroqLLM(client=client, max_retries=3)
    msg = llm.complete([], [])
    assert msg.content == "booked"
    assert client.calls == 2  # failed once, retried, succeeded


def test_complete_raises_llmerror_after_persistent_tool_use_failed():
    client = _FakeGroqClient([_tool_use_failed_error() for _ in range(3)])
    llm = GroqLLM(client=client, max_retries=3)
    with pytest.raises(LLMError):
        llm.complete([], [])
    assert client.calls == 3  # exhausted all retries


def test_complete_does_not_retry_other_errors():
    client = _FakeGroqClient([Exception("401 invalid api key"), SimpleNamespace(choices=[])])
    llm = GroqLLM(client=client, max_retries=3)
    with pytest.raises(LLMError):
        llm.complete([], [])
    assert client.calls == 1  # non-tool_use_failed errors fail fast, no retry


def _rate_limit_error():
    e = Exception("429 Too Many Requests: rate limit exceeded")
    e.status_code = 429
    return e


def test_complete_retries_on_rate_limit_then_succeeds():
    good = SimpleNamespace(choices=[_fake_choice(content="ok")])
    client = _FakeGroqClient([_rate_limit_error(), good])
    llm = GroqLLM(client=client, max_retries=3, backoff_base=0)
    msg = llm.complete([], [])
    assert msg.content == "ok"
    assert client.calls == 2  # backed off and retried


def test_complete_raises_llmerror_after_persistent_rate_limit():
    client = _FakeGroqClient([_rate_limit_error() for _ in range(3)])
    llm = GroqLLM(client=client, max_retries=3, rate_limit_retries=2, backoff_base=0)
    with pytest.raises(LLMError):
        llm.complete([], [])
    assert client.calls == 3


def test_retry_after_seconds_parses_header():
    from voicedesk.groq_client import _retry_after_seconds
    from types import SimpleNamespace as SN
    e = Exception("429 Too Many Requests")
    e.response = SN(headers={"retry-after": "3"})
    assert _retry_after_seconds(e) == 3.0


def test_retry_after_seconds_parses_message_body():
    from voicedesk.groq_client import _retry_after_seconds
    e = Exception("rate limit exceeded, please try again in 7.66s")
    assert _retry_after_seconds(e) == 7.66


def test_retry_after_seconds_returns_none_when_absent():
    from voicedesk.groq_client import _retry_after_seconds
    e = Exception("429 Too Many Requests: rate limit exceeded")
    assert _retry_after_seconds(e) is None


def test_complete_sleeps_for_retry_after_duration(monkeypatch):
    import voicedesk.groq_client as gc

    sleeps = []
    monkeypatch.setattr(gc.time, "sleep", lambda s: sleeps.append(s))

    e = Exception("rate limit exceeded, please try again in 5s")
    e.status_code = 429
    good = SimpleNamespace(choices=[_fake_choice(content="ok")])
    client = _FakeGroqClient([e, good])
    llm = GroqLLM(client=client, backoff_base=2.0)
    msg = llm.complete([], [])
    assert msg.content == "ok"
    assert sleeps == [5.0]


def test_complete_raises_quota_exhausted_instead_of_capping_huge_wait(monkeypatch):
    """A wait this long (9999s) means the daily quota is gone, not a
    per-minute rate limit — capping it to MAX_BACKOFF_S and retrying would
    just grind forever, so we must fail fast instead."""
    import voicedesk.groq_client as gc

    sleeps = []
    monkeypatch.setattr(gc.time, "sleep", lambda s: sleeps.append(s))

    e = Exception("rate limit exceeded, please try again in 9999s")
    e.status_code = 429
    good = SimpleNamespace(choices=[_fake_choice(content="ok")])
    client = _FakeGroqClient([e, good])
    llm = GroqLLM(client=client, backoff_base=2.0)
    with pytest.raises(QuotaExhausted):
        llm.complete([], [])
    assert sleeps == []
    assert client.calls == 1


def test_on_retry_called_on_rate_limit_with_wait_seconds(monkeypatch):
    import voicedesk.groq_client as gc

    monkeypatch.setattr(gc.time, "sleep", lambda s: None)

    e = Exception("rate limit exceeded, please try again in 5s")
    e.status_code = 429
    good = SimpleNamespace(choices=[_fake_choice(content="ok")])
    client = _FakeGroqClient([e, good])

    calls = []
    llm = GroqLLM(client=client, backoff_base=2.0,
                  on_retry=lambda reason, wait_s, attempt: calls.append((reason, wait_s, attempt)))
    msg = llm.complete([], [])
    assert msg.content == "ok"
    assert calls == [("rate_limited", 5.0, 1)]


def test_on_retry_called_on_tool_use_failed():
    good = SimpleNamespace(choices=[_fake_choice(content="booked")])
    client = _FakeGroqClient([_tool_use_failed_error(), good])

    calls = []
    llm = GroqLLM(client=client, max_retries=3,
                  on_retry=lambda reason, wait_s, attempt: calls.append((reason, wait_s, attempt)))
    msg = llm.complete([], [])
    assert msg.content == "booked"
    assert calls == [("tool_use_failed", 0.0, 1)]


def test_on_retry_exception_does_not_break_complete(monkeypatch):
    import voicedesk.groq_client as gc

    monkeypatch.setattr(gc.time, "sleep", lambda s: None)

    e = Exception("rate limit exceeded, please try again in 5s")
    e.status_code = 429
    good = SimpleNamespace(choices=[_fake_choice(content="ok")])
    client = _FakeGroqClient([e, good])

    def _broken(reason, wait_s, attempt):
        raise RuntimeError("boom")

    llm = GroqLLM(client=client, backoff_base=2.0, on_retry=_broken)
    msg = llm.complete([], [])
    assert msg.content == "ok"


def test_complete_works_without_on_retry_callback():
    good = SimpleNamespace(choices=[_fake_choice(content="ok")])
    client = _FakeGroqClient([good])
    llm = GroqLLM(client=client)
    msg = llm.complete([], [])
    assert msg.content == "ok"


# --- Fix 1 & 2: honor full Retry-After, proactive throttling from headers ---

def test_parse_duration_milliseconds():
    from voicedesk.groq_client import _parse_duration
    assert _parse_duration("370ms") == pytest.approx(0.37)


def test_parse_duration_seconds():
    from voicedesk.groq_client import _parse_duration
    assert _parse_duration("6s") == 6.0


def test_parse_duration_minutes_and_seconds():
    from voicedesk.groq_client import _parse_duration
    assert _parse_duration("1m26.4s") == pytest.approx(86.4)


def test_parse_duration_none():
    from voicedesk.groq_client import _parse_duration
    assert _parse_duration(None) == 0.0


def test_parse_duration_garbage():
    from voicedesk.groq_client import _parse_duration
    assert _parse_duration("garbage") == 0.0


class _FakeRawResponse:
    def __init__(self, choice, headers):
        self._choice = choice
        self.headers = headers

    def parse(self):
        return SimpleNamespace(choices=[self._choice])


class _FakeGroqClientWithRawResponse:
    """Exposes chat.completions.with_raw_response.create(...), like the real
    groq SDK, so we can capture rate-limit headers on every response."""

    def __init__(self, behaviors):
        self.behaviors = list(behaviors)
        self.calls = 0
        outer = self

        def _create(**kw):
            return outer._next()

        self.chat = SimpleNamespace(
            completions=SimpleNamespace(
                with_raw_response=SimpleNamespace(create=_create),
            )
        )

    def _next(self):
        b = self.behaviors[self.calls]
        self.calls += 1
        if isinstance(b, Exception):
            raise b
        return b


def test_complete_records_limits_from_raw_response_headers():
    choice = _fake_choice(content="ok")
    headers = {"x-ratelimit-remaining-tokens": "500", "x-ratelimit-reset-tokens": "2s"}
    raw = _FakeRawResponse(choice, headers)
    client = _FakeGroqClientWithRawResponse([raw])
    llm = GroqLLM(client=client)
    msg = llm.complete([], [])
    assert msg.content == "ok"
    assert llm._remaining_tokens == 500.0
    assert llm._tokens_reset_s == 2.0


def test_throttle_sleeps_when_bucket_low(monkeypatch):
    import voicedesk.groq_client as gc

    sleeps = []
    monkeypatch.setattr(gc.time, "sleep", lambda s: sleeps.append(s))

    good = SimpleNamespace(choices=[_fake_choice(content="ok")])
    client = _FakeGroqClient([good])
    calls = []
    llm = GroqLLM(client=client,
                  on_retry=lambda reason, wait_s, attempt: calls.append((reason, wait_s, attempt)))
    llm._remaining_tokens = 100
    llm._tokens_reset_s = 2.0
    msg = llm.complete([], [])
    assert msg.content == "ok"
    assert sleeps == [pytest.approx(2.5)]
    assert calls == [("throttle", pytest.approx(2.5), 0)]


def test_throttle_does_not_sleep_when_bucket_high(monkeypatch):
    import voicedesk.groq_client as gc

    sleeps = []
    monkeypatch.setattr(gc.time, "sleep", lambda s: sleeps.append(s))

    good = SimpleNamespace(choices=[_fake_choice(content="ok")])
    client = _FakeGroqClient([good])
    llm = GroqLLM(client=client)
    llm._remaining_tokens = 10000
    llm._tokens_reset_s = 2.0
    msg = llm.complete([], [])
    assert msg.content == "ok"
    assert sleeps == []


def test_complete_honors_retry_after_longer_than_old_cap(monkeypatch):
    import voicedesk.groq_client as gc

    sleeps = []
    monkeypatch.setattr(gc.time, "sleep", lambda s: sleeps.append(s))

    e = Exception("rate limit exceeded, please try again in 120s")
    e.status_code = 429
    good = SimpleNamespace(choices=[_fake_choice(content="ok")])
    client = _FakeGroqClient([e, good])
    llm = GroqLLM(client=client, backoff_base=2.0)
    msg = llm.complete([], [])
    assert msg.content == "ok"
    assert sleeps == [120.0]


# --- Fix: fail fast on daily-quota-exhaustion waits, don't grind on retries ---

def test_complete_raises_quota_exhausted_for_long_wait_no_sleep(monkeypatch):
    import voicedesk.groq_client as gc

    sleeps = []
    monkeypatch.setattr(gc.time, "sleep", lambda s: sleeps.append(s))

    e = Exception("rate limit exceeded, please try again in 300s")
    e.status_code = 429
    good = SimpleNamespace(choices=[_fake_choice(content="ok")])
    client = _FakeGroqClient([e, good])
    llm = GroqLLM(client=client, backoff_base=2.0)
    with pytest.raises(QuotaExhausted) as exc_info:
        llm.complete([], [])
    assert sleeps == []
    assert client.calls == 1
    assert "daily quota" in str(exc_info.value)


def test_complete_still_retries_short_rate_limit_wait(monkeypatch):
    import voicedesk.groq_client as gc

    sleeps = []
    monkeypatch.setattr(gc.time, "sleep", lambda s: sleeps.append(s))

    e = Exception("rate limit exceeded, please try again in 5s")
    e.status_code = 429
    good = SimpleNamespace(choices=[_fake_choice(content="ok")])
    client = _FakeGroqClient([e, good])
    llm = GroqLLM(client=client, backoff_base=2.0)
    msg = llm.complete([], [])
    assert msg.content == "ok"
    assert sleeps == [5.0]
    assert client.calls == 2


def test_quota_exhausted_is_subclass_of_llm_error():
    assert issubclass(QuotaExhausted, LLMError)
