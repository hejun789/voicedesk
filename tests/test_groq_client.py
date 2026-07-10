from types import SimpleNamespace
from voicedesk.groq_client import _to_message


def _fake_choice(content=None, tool_calls=None):
    msg = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(message=msg)


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
