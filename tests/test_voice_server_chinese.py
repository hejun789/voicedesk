import sqlite3
import pytest
from fastapi.testclient import TestClient

from voicedesk.db import init_db
from voicedesk.agent import Agent, build_system_prompt
from voicedesk.lang import faq_doc_for
from voicedesk.llm import FakeLLM, Message
from voicedesk.voice.stt import FakeSTT
from voicedesk.voice.session import SessionStore
from voicedesk.voice.server import create_app, DIDNT_CATCH_ZH


class _RecordingSTT:
    """Records the language it was asked to transcribe in."""

    def __init__(self, text="我要预约"):
        self.text = text
        self.languages = []

    def transcribe(self, audio, filename="audio.webm", language="en"):
        self.languages.append(language)
        return self.text


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:", check_same_thread=False)
    c.row_factory = sqlite3.Row
    init_db(c)
    yield c
    c.close()


def _client(conn, stt, llm):
    from datetime import date
    sessions = SessionStore(lambda lang: Agent(
        conn, llm,
        system_prompt=build_system_prompt(date(2026, 7, 10), lang),
        faq_doc_path=faq_doc_for(lang),
    ))
    return TestClient(create_app(stt, sessions))


def _post(client, lang="en", session_id="s1"):
    data = {"session_id": session_id}
    if lang is not None:
        data["lang"] = lang
    return client.post("/turn", data=data,
                       files={"audio": ("turn.webm", b"x" * 2000, "audio/webm")})


def test_lang_reaches_the_transcriber_and_is_echoed_back(conn):
    stt = _RecordingSTT()
    llm = FakeLLM([Message(content="好的，请问您的姓名？", tool_calls=[])])
    body = _post(_client(conn, stt, llm), lang="zh").json()
    assert stt.languages == ["zh"]
    assert body["lang"] == "zh"
    assert body["reply"] == "好的，请问您的姓名？"


def test_lang_defaults_to_english_when_absent(conn):
    stt = _RecordingSTT(text="hello")
    llm = FakeLLM([Message(content="Hi!", tool_calls=[])])
    body = _post(_client(conn, stt, llm), lang=None).json()
    assert stt.languages == ["en"]
    assert body["lang"] == "en"


def test_unknown_lang_falls_back_to_english(conn):
    stt = _RecordingSTT(text="hello")
    llm = FakeLLM([Message(content="Hi!", tool_calls=[])])
    body = _post(_client(conn, stt, llm), lang="klingon").json()
    assert stt.languages == ["en"]
    assert body["lang"] == "en"


def test_didnt_catch_is_spoken_in_chinese(conn):
    # A stray tap must apologise in the caller's language.
    stt = FakeSTT([])   # never called — the audio is too small
    llm = FakeLLM([])
    client = _client(conn, stt, llm)
    r = client.post("/turn", data={"session_id": "s1", "lang": "zh"},
                    files={"audio": ("turn.webm", b"tiny", "audio/webm")})
    body = r.json()
    assert body["reply"] == DIDNT_CATCH_ZH
    assert body["lang"] == "zh"


def test_same_session_id_in_two_languages_is_two_conversations(conn):
    stt = _RecordingSTT()
    llm = FakeLLM([Message(content="a", tool_calls=[]),
                   Message(content="b", tool_calls=[])])
    client = _client(conn, stt, llm)
    _post(client, lang="en", session_id="s1")
    _post(client, lang="zh", session_id="s1")
    # Two separate agents were built — a language switch is a new context, not
    # a mixed-language history.
    assert stt.languages == ["en", "zh"]
