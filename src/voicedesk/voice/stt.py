import os
from typing import Protocol

from voicedesk.lang import DEFAULT_LANG, normalize_lang

DEFAULT_STT_MODEL = "whisper-large-v3-turbo"

# Biases Whisper toward the vocabulary a receptionist call actually contains.
# Proper nouns and phone numbers are its weakest point without context.
TRANSCRIPTION_PROMPT = (
    "A phone call to BrightSmile Dental. The caller gives their full name, "
    "a phone number, a date and time, and a reason such as a cleaning, "
    "filling, crown, checkup, or whitening. Names are spelled normally, "
    "for example: Jane Doe, John Smith, Mary Lee."
)

TRANSCRIPTION_PROMPT_ZH = (
    "一通打给 BrightSmile 牙科诊所的电话。来电者会说出自己的姓名、"
    "电话号码、日期和时间，以及就诊原因，例如洗牙、补牙、牙冠、"
    "检查或牙齿美白。"
)

TRANSCRIPTION_PROMPTS = {
    "en": TRANSCRIPTION_PROMPT,
    "zh": TRANSCRIPTION_PROMPT_ZH,
}


class STTError(Exception):
    """Transcription failed (API error). The server degrades gracefully rather
    than crashing the call."""


# Whisper emits these instead of an empty string on silence or noise. The
# Chinese ones come from its YouTube subtitle training data.
SILENCE_HALLUCINATIONS = {
    "thank you.", "thank you", "you", "bye.", "bye",
    "thanks for watching!", ".", "so",
    "谢谢观看", "谢谢大家观看", "请不吝点赞", "明镜与点点栏目",
    "字幕由amara.org社区提供", "字幕志愿者", "小编",
    "謝謝觀看", "謝謝大家觀看", "請不吝點贊", "明鏡與點點欄目",
    "字幕由amara.org社區提供", "字幕志願者",
}

# Chinese sentences end in these; strip them before comparing. ASCII
# punctuation is deliberately NOT included — stripping "!" would make
# "Thank you!" collide with the English artefact "thank you" and swallow a
# real caller's words.
_TRAILING_PUNCT = " 。．，！？、…～"


def is_silence_hallucination(text: str) -> bool:
    """True when a transcript is one of Whisper's known silence artefacts."""
    return text.strip().strip(_TRAILING_PUNCT).strip().lower() in SILENCE_HALLUCINATIONS


class STTClient(Protocol):
    def transcribe(self, audio: bytes, filename: str = "audio.webm",
                   language: str = DEFAULT_LANG) -> str: ...


class FakeSTT:
    """Test double: returns scripted transcripts in order."""

    def __init__(self, scripted: list[str]):
        self._scripted = list(scripted)

    def transcribe(self, audio: bytes, filename: str = "audio.webm",
                   language: str = DEFAULT_LANG) -> str:
        return self._scripted.pop(0)


class GroqWhisper:
    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        client=None,
        prompt: str | None = None,
    ):
        # Whisper draws on a SEPARATE rate-limit pool from the chat model, so
        # transcription does not compete with the agent's LLM quota.
        self.model = model or os.environ.get("GROQ_STT_MODEL", DEFAULT_STT_MODEL)
        # None means "pick the prompt for the call's language"; an explicit
        # prompt overrides that for every language.
        self.prompt = prompt
        if client is not None:
            self.client = client  # injected (used by tests — no network/key)
        else:
            from groq import Groq  # imported lazily so tests don't need the package
            self.client = Groq(api_key=api_key or os.environ["GROQ_API_KEY"])

    def transcribe(self, audio: bytes, filename: str = "audio.webm",
                   language: str = DEFAULT_LANG) -> str:
        lang = normalize_lang(language)
        try:
            resp = self.client.audio.transcriptions.create(
                file=(filename, audio),
                model=self.model,
                language=lang,
                temperature=0,
                prompt=self.prompt or TRANSCRIPTION_PROMPTS[lang],
            )
        except Exception as e:  # noqa: BLE001 - translated to STTError
            raise STTError(str(e)) from e
        return (getattr(resp, "text", "") or "").strip()
