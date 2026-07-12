import pytest
from types import SimpleNamespace
from voicedesk.voice.stt import FakeSTT, GroqWhisper, STTError, DEFAULT_STT_MODEL


class _FakeAudioClient:
    """Stands in for groq.Groq: exposes .audio.transcriptions.create(...).
    `behavior` is either an Exception to raise or the object to return."""

    def __init__(self, behavior):
        self.behavior = behavior
        self.calls = []
        outer = self
        self.audio = SimpleNamespace(
            transcriptions=SimpleNamespace(create=lambda **kw: outer._next(kw))
        )

    def _next(self, kwargs):
        self.calls.append(kwargs)
        if isinstance(self.behavior, Exception):
            raise self.behavior
        return self.behavior


def test_fake_stt_returns_scripted_transcripts():
    stt = FakeSTT(["book me monday", "jane doe"])
    assert stt.transcribe(b"x") == "book me monday"
    assert stt.transcribe(b"x") == "jane doe"


def test_groq_whisper_returns_stripped_text():
    client = _FakeAudioClient(SimpleNamespace(text="  book me monday at 9am  "))
    stt = GroqWhisper(client=client)
    assert stt.transcribe(b"audiobytes", "turn.webm") == "book me monday at 9am"


def test_groq_whisper_sends_file_and_model():
    client = _FakeAudioClient(SimpleNamespace(text="hi"))
    stt = GroqWhisper(client=client)
    stt.transcribe(b"audiobytes", "turn.webm")
    sent = client.calls[0]
    assert sent["file"] == ("turn.webm", b"audiobytes")
    assert sent["model"] == DEFAULT_STT_MODEL


def test_groq_whisper_empty_text_is_empty_string():
    client = _FakeAudioClient(SimpleNamespace(text=None))
    stt = GroqWhisper(client=client)
    assert stt.transcribe(b"x") == ""


def test_groq_whisper_wraps_api_errors_in_stterror():
    client = _FakeAudioClient(Exception("429 rate limit"))
    stt = GroqWhisper(client=client)
    with pytest.raises(STTError):
        stt.transcribe(b"x")
