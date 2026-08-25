from fastapi.testclient import TestClient

from voicedesk.agent import Agent
from voicedesk.llm import FakeLLM
from voicedesk.voice.server import create_app, MAX_TTS_CHARS
from voicedesk.voice.session import SessionStore
from voicedesk.voice.stt import FakeSTT
from voicedesk.voice.tts import FakeTTS, TTSError


def _client(db, tts=None, tts_limiter=None):
    sessions = SessionStore(lambda lang: Agent(db, FakeLLM([])))
    app = create_app(FakeSTT([]), sessions, tts=tts, tts_limiter=tts_limiter)
    return TestClient(app)


def test_tts_returns_wav_bytes_from_the_injected_client(db):
    tts = FakeTTS(wav=b"RIFF-test-wav")
    client = _client(db, tts=tts)
    res = client.post("/tts", data={"text": "Booked for Monday.", "lang": "en"})
    assert res.status_code == 200
    assert res.headers["content-type"] == "audio/wav"
    assert res.content == b"RIFF-test-wav"
    assert tts.calls == [("Booked for Monday.", "en")]


def test_tts_normalizes_language_before_calling_the_client(db):
    tts = FakeTTS()
    client = _client(db, tts=tts)
    client.post("/tts", data={"text": "你好", "lang": "zh-CN"})
    assert tts.calls == [("你好", "zh")]


def test_tts_503_when_no_client_configured(db):
    client = _client(db, tts=None)
    res = client.post("/tts", data={"text": "hi", "lang": "en"})
    assert res.status_code == 503
    assert res.json()["detail"] == "tts_unavailable"


class _RaisingTTS:
    def synthesize(self, text, language="en"):
        raise TTSError("synthesis backend down")


def test_tts_503_when_synthesis_fails(db):
    client = _client(db, tts=_RaisingTTS())
    res = client.post("/tts", data={"text": "hi", "lang": "en"})
    assert res.status_code == 503
    assert res.json()["detail"] == "tts_failed"


def test_tts_413_when_text_exceeds_the_char_cap(db):
    tts = FakeTTS()
    client = _client(db, tts=tts)
    oversize = "x" * (MAX_TTS_CHARS + 1)
    res = client.post("/tts", data={"text": oversize, "lang": "en"})
    assert res.status_code == 413
    assert res.json()["detail"] == "text_too_long"
    assert tts.calls == []   # rejected before it ever reached the client


class _RefusingLimiter:
    def allow(self, ip):
        return False


def test_tts_429_when_the_limiter_refuses(db):
    tts = FakeTTS()
    client = _client(db, tts=tts, tts_limiter=_RefusingLimiter())
    res = client.post("/tts", data={"text": "hi", "lang": "en"})
    assert res.status_code == 429
    assert res.json()["detail"] == "rate_limited"
    assert tts.calls == []   # rejected before it ever reached the client
