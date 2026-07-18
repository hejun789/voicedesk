from types import SimpleNamespace
from voicedesk.voice.stt import (
    FakeSTT, GroqWhisper, TRANSCRIPTION_PROMPTS, is_silence_hallucination,
)


class _FakeAudioClient:
    def __init__(self, text="好的"):
        self.calls = []
        outer = self
        self.audio = SimpleNamespace(
            transcriptions=SimpleNamespace(
                create=lambda **kw: outer._next(kw, text))
        )

    def _next(self, kwargs, text):
        self.calls.append(kwargs)
        return SimpleNamespace(text=text)


def test_transcribe_sends_chinese_language_and_prompt():
    client = _FakeAudioClient()
    GroqWhisper(client=client).transcribe(b"x", "turn.webm", language="zh")
    sent = client.calls[0]
    assert sent["language"] == "zh"
    assert sent["prompt"] == TRANSCRIPTION_PROMPTS["zh"]
    assert "牙科" in sent["prompt"]


def test_transcribe_defaults_to_english():
    client = _FakeAudioClient(text="hello")
    GroqWhisper(client=client).transcribe(b"x")
    assert client.calls[0]["language"] == "en"
    assert client.calls[0]["prompt"] == TRANSCRIPTION_PROMPTS["en"]


def test_transcribe_normalizes_an_unknown_language():
    client = _FakeAudioClient()
    GroqWhisper(client=client).transcribe(b"x", "turn.webm", language="klingon")
    assert client.calls[0]["language"] == "en"


def test_explicit_prompt_overrides_both_languages():
    client = _FakeAudioClient()
    GroqWhisper(client=client, prompt="custom").transcribe(b"x", language="zh")
    assert client.calls[0]["prompt"] == "custom"


def test_fake_stt_accepts_a_language():
    stt = FakeSTT(["你好"])
    assert stt.transcribe(b"x", "turn.webm", language="zh") == "你好"


def test_chinese_silence_hallucinations_are_recognized():
    # Whisper emits these on silence — famously "thanks for watching", learned
    # from YouTube subtitles. Feeding them to the agent would be noise.
    assert is_silence_hallucination("谢谢观看")
    assert is_silence_hallucination("谢谢观看。")
    assert is_silence_hallucination("字幕由Amara.org社区提供")


def test_real_chinese_speech_is_not_treated_as_silence():
    assert not is_silence_hallucination("我要预约洗牙")
    assert not is_silence_hallucination("好的")   # a real confirmation


def test_english_farewells_with_ascii_punctuation_are_not_silence():
    # Regression guard: ASCII trailing punctuation must not be stripped, or
    # "Thank you!" collides with the English silence artefact "thank you" and
    # a real caller's sign-off gets swallowed as silence.
    assert is_silence_hallucination("Thank you!") is False
    assert is_silence_hallucination("Bye!") is False


def test_traditional_chinese_silence_hallucination_is_recognized():
    # whisper-large-v3-turbo routinely emits Traditional characters for
    # Mandarin speech even when the transcription prompt is Simplified.
    assert is_silence_hallucination("謝謝觀看") is True
