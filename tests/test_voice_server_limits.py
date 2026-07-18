import sqlite3
import pytest
from fastapi.testclient import TestClient

from voicedesk.db import init_db
from voicedesk.agent import Agent
from voicedesk.llm import FakeLLM, Message
from voicedesk.voice.stt import FakeSTT
from voicedesk.voice.session import SessionStore
from voicedesk.voice.limits import RateLimiter
from voicedesk.voice.server import create_app, DEMO_LIMIT, DEMO_LIMIT_ZH


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:", check_same_thread=False)
    c.row_factory = sqlite3.Row
    init_db(c)
    yield c
    c.close()


def _client(conn, stt, llm, limiter):
    sessions = SessionStore(lambda lang: Agent(conn, llm))
    return TestClient(create_app(stt, sessions, limiter=limiter))


def _post(client, lang="en", xff=None):
    headers = {"X-Forwarded-For": xff} if xff else {}
    return client.post(
        "/turn",
        data={"session_id": "s1", "lang": lang},
        files={"audio": ("turn.webm", b"x" * 2000, "audio/webm")},
        headers=headers,
    )


def test_over_limit_returns_the_demo_message_without_calling_stt_or_agent(conn):
    # per_ip_limit=0 => every turn is over the limit. FakeSTT([]) / FakeLLM([])
    # would IndexError if reached, proving neither is called.
    limiter = RateLimiter(per_ip_limit=0, global_limit=100)
    body = _post(_client(conn, FakeSTT([]), FakeLLM([]), limiter)).json()
    assert body["reply"] == DEMO_LIMIT
    assert body["error"] == "rate_limited"
    assert body["lang"] == "en"


def test_over_limit_message_is_localized(conn):
    limiter = RateLimiter(per_ip_limit=0, global_limit=100)
    body = _post(_client(conn, FakeSTT([]), FakeLLM([]), limiter), lang="zh").json()
    assert body["reply"] == DEMO_LIMIT_ZH
    assert body["lang"] == "zh"


def test_under_the_limit_the_turn_is_processed_normally(conn):
    limiter = RateLimiter(per_ip_limit=5, global_limit=100)
    stt = FakeSTT(["what are your hours"])
    llm = FakeLLM([Message(content="Weekdays 9 to 5.", tool_calls=[])])
    body = _post(_client(conn, stt, llm, limiter)).json()
    assert body["reply"] == "Weekdays 9 to 5."


def test_per_ip_limit_blocks_the_next_turn_from_the_same_ip(conn):
    limiter = RateLimiter(per_ip_limit=1, global_limit=100)
    stt = FakeSTT(["hi"])                    # only the first (allowed) turn reaches STT
    llm = FakeLLM([Message(content="Hello!", tool_calls=[])])
    client = _client(conn, stt, llm, limiter)
    assert _post(client).json()["reply"] == "Hello!"
    assert _post(client).json()["reply"] == DEMO_LIMIT   # same IP, second turn blocked


def test_x_forwarded_for_identifies_the_visitor(conn):
    # Two different forwarded IPs get independent per-IP budgets.
    limiter = RateLimiter(per_ip_limit=1, global_limit=100)
    stt = FakeSTT(["a", "b"])
    llm = FakeLLM([Message(content="one", tool_calls=[]),
                   Message(content="two", tool_calls=[])])
    client = _client(conn, stt, llm, limiter)
    assert _post(client, xff="9.9.9.9").json()["reply"] == "one"
    assert _post(client, xff="9.9.9.9").json()["reply"] == DEMO_LIMIT   # IP1 spent
    assert _post(client, xff="8.8.8.8").json()["reply"] == "two"        # IP2 fresh
