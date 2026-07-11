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
    llm = GroqLLM(client=client, max_retries=3, backoff_base=0)
    with pytest.raises(LLMError):
        llm.complete([], [])
    assert client.calls == 3
