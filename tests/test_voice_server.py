import sqlite3
import pytest
from fastapi.testclient import TestClient

from voicedesk.db import init_db
from voicedesk.agent import Agent
from voicedesk.llm import FakeLLM, Message, ToolCall
from voicedesk.tools import lookup_appt
from voicedesk.voice.stt import FakeSTT, STTError
from voicedesk.voice.session import SessionStore
from voicedesk.voice.server import create_app, DIDNT_CATCH, STT_FAILED


class _RaisingSTT:
    def transcribe(self, audio, filename="audio.webm"):
        raise STTError("429 rate limit")


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:", check_same_thread=False)
    c.row_factory = sqlite3.Row
    init_db(c)
    yield c
    c.close()


def _client(conn, stt, llm):
    sessions = SessionStore(lambda: Agent(conn, llm))
    return TestClient(create_app(stt, sessions))


def _post(client, audio=b"fakeaudio", session_id="s1"):
    return client.post(
        "/turn",
        data={"session_id": session_id},
        files={"audio": ("turn.webm", audio, "audio/webm")},
    )


def test_voice_turn_books_a_real_appointment(conn):
    stt = FakeSTT(["Book me Monday July 13th 2026 at 9am, Jane Doe, 5551234, cleaning"])
    llm = FakeLLM([
        Message(content=None, tool_calls=[
            ToolCall(id="1", name="book", arguments={
                "patient_name": "Jane Doe", "phone": "5551234",
                "slot_iso": "2026-07-13T09:00", "reason": "cleaning"})]),
        Message(content="You're booked for Monday at 9am.", tool_calls=[]),
    ])
    r = _post(_client(conn, stt, llm))
    assert r.status_code == 200
    body = r.json()
    assert "Jane Doe" in body["transcript"]
    assert "booked" in body["reply"].lower()
    # the side effect really happened:
    assert lookup_appt(conn, phone="5551234")[0]["slot_iso"] == "2026-07-13T09:00"


def test_response_carries_latency_breakdown(conn):
    stt = FakeSTT(["what are your hours"])
    llm = FakeLLM([Message(content="Weekdays 9 to 5.", tool_calls=[])])
    body = _post(_client(conn, stt, llm)).json()
    t = body["timings"]
    assert set(t) == {"stt_ms", "agent_ms", "total_ms"}
    assert all(isinstance(v, int) and v >= 0 for v in t.values())


def test_empty_transcript_does_not_call_the_agent(conn):
    stt = FakeSTT([""])
    llm = FakeLLM([])  # any agent call would IndexError -> proves it wasn't called
    body = _post(_client(conn, stt, llm)).json()
    assert body["reply"] == DIDNT_CATCH
    assert body["transcript"] == ""


def test_whitespace_transcript_does_not_call_the_agent(conn):
    stt = FakeSTT(["   "])
    llm = FakeLLM([])
    body = _post(_client(conn, stt, llm)).json()
    assert body["reply"] == DIDNT_CATCH


def test_stt_failure_degrades_gracefully(conn):
    llm = FakeLLM([])
    r = _post(_client(conn, _RaisingSTT(), llm))
    assert r.status_code == 200          # never a stack trace mid-call
    body = r.json()
    assert body["reply"] == STT_FAILED
    assert "429" in body["error"]


def test_same_session_id_keeps_conversation_history(conn):
    stt = FakeSTT(["I'd like an appointment", "Jane Doe, 5551234"])
    llm = FakeLLM([
        Message(content="Sure — what's your name and number?", tool_calls=[]),
        Message(content="Thanks Jane, what day suits you?", tool_calls=[]),
    ])
    client = _client(conn, stt, llm)
    first = _post(client, session_id="s1").json()
    second = _post(client, session_id="s1").json()
    assert "name" in first["reply"]
    assert "Jane" in second["reply"]


def test_index_page_is_served(conn):
    client = _client(conn, FakeSTT([]), FakeLLM([]))
    r = client.get("/")
    assert r.status_code == 200
    assert "Hold to talk" in r.text


def test_static_app_js_is_served(conn):
    client = _client(conn, FakeSTT([]), FakeLLM([]))
    r = client.get("/static/app.js")
    assert r.status_code == 200
    assert "speechSynthesis" in r.text
