from fastapi.testclient import TestClient

from voicedesk.agent import Agent
from voicedesk.llm import FakeLLM, Message
from voicedesk.voice.server import create_app
from voicedesk.voice.session import SessionStore
from voicedesk.voice.stt import FakeSTT

AUDIO = b"x" * 2000  # over MIN_AUDIO_BYTES so it is not rejected as a stray tap


def _client(db, replies):
    """An app whose agent returns `replies` in order, sharing one SessionStore
    so a test can inspect conversation history after each turn."""
    sessions = SessionStore(
        lambda lang: Agent(db, FakeLLM([Message(content=r, tool_calls=[])
                                        for r in replies]))
    )
    app = create_app(FakeSTT(["first question", "second question"]), sessions)
    return TestClient(app), sessions


def _post(client, **extra):
    data = {"session_id": "s1", "lang": "en", **extra}
    return client.post("/turn", data=data,
                       files={"audio": ("turn.webm", AUDIO, "audio/webm")})


def test_heard_chars_truncates_the_previous_reply(db):
    client, sessions = _client(db, ["Booked for Monday at nine. Anything else?",
                                    "Sure."])
    _post(client)
    _post(client, heard_chars=26)
    agent = sessions.get_or_create("s1", "en")
    contents = [m.get("content") for m in agent.messages if m.get("role") == "assistant"]
    assert contents[0] == "Booked for Monday at nine. [interrupted]"


def test_omitting_heard_chars_leaves_history_untouched(db):
    # Regression guard: existing clients that never send the field must behave
    # exactly as they do today.
    full = "Booked for Monday at nine. Anything else?"
    client, sessions = _client(db, [full, "Sure."])
    _post(client)
    _post(client)
    agent = sessions.get_or_create("s1", "en")
    contents = [m.get("content") for m in agent.messages if m.get("role") == "assistant"]
    assert contents[0] == full


def test_heard_chars_zero_drops_the_previous_reply(db):
    client, sessions = _client(db, ["Booked for Monday.", "Sure."])
    _post(client)
    _post(client, heard_chars=0)
    agent = sessions.get_or_create("s1", "en")
    contents = [m.get("content") for m in agent.messages if m.get("role") == "assistant"]
    assert contents == ["Sure."]


def test_response_shape_is_unchanged_when_heard_chars_is_sent(db):
    client, _ = _client(db, ["Booked.", "Sure."])
    _post(client)
    res = _post(client, heard_chars=3)
    assert res.status_code == 200
    body = res.json()
    assert set(body) >= {"transcript", "reply", "timings", "lang"}
