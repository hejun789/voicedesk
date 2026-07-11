import pytest
from types import SimpleNamespace
from voicedesk.groq_client import _to_message, GroqLLM
from voicedesk.llm import LLMError


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


def test_complete_caps_retry_after_at_max_backoff(monkeypatch):
    import voicedesk.groq_client as gc

    sleeps = []
    monkeypatch.setattr(gc.time, "sleep", lambda s: sleeps.append(s))

    e = Exception("rate limit exceeded, please try again in 9999s")
    e.status_code = 429
    good = SimpleNamespace(choices=[_fake_choice(content="ok")])
    client = _FakeGroqClient([e, good])
    llm = GroqLLM(client=client, backoff_base=2.0)
    msg = llm.complete([], [])
    assert msg.content == "ok"
    assert sleeps == [gc.MAX_BACKOFF_S]


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
