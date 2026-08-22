from fastapi.testclient import TestClient

from voicedesk.agent import Agent
from voicedesk.llm import FakeLLM
from voicedesk.voice.server import create_app
from voicedesk.voice.session import SessionStore
from voicedesk.voice.stt import FakeSTT
from voicedesk.voice.tts import FakeTTS, TTSError


def _client(db, tts=None):
    sessions = SessionStore(lambda lang: Agent(db, FakeLLM([])))
    app = create_app(FakeSTT([]), sessions, tts=tts)
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
