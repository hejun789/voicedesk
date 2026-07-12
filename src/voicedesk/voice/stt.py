import os
from typing import Protocol

DEFAULT_STT_MODEL = "whisper-large-v3-turbo"

# Biases Whisper toward the vocabulary a receptionist call actually contains.
# Proper nouns and phone numbers are its weakest point without context.
TRANSCRIPTION_PROMPT = (
    "A phone call to BrightSmile Dental. The caller gives their full name, "
    "a phone number, a date and time, and a reason such as a cleaning, "
    "filling, crown, checkup, or whitening. Names are spelled normally, "
    "for example: Jane Doe, John Smith, Mary Lee."
)


class STTError(Exception):
    """Transcription failed (API error). The server degrades gracefully rather
    than crashing the call."""


# Whisper emits these on silence/noise rather than an empty string.
SILENCE_HALLUCINATIONS = {"thank you.", "thank you", "you", "bye.", "bye",
                          "thanks for watching!", ".", "so"}


def is_silence_hallucination(text: str) -> bool:
    """True when a transcript is one of Whisper's known silence artefacts."""
    return text.strip().lower() in SILENCE_HALLUCINATIONS


class STTClient(Protocol):
    def transcribe(self, audio: bytes, filename: str = "audio.webm") -> str: ...


class FakeSTT:
    """Test double: returns scripted transcripts in order."""

    def __init__(self, scripted: list[str]):
        self._scripted = list(scripted)

    def transcribe(self, audio: bytes, filename: str = "audio.webm") -> str:
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
        self.prompt = prompt or TRANSCRIPTION_PROMPT
        if client is not None:
            self.client = client  # injected (used by tests — no network/key)
        else:
            from groq import Groq  # imported lazily so tests don't need the package
            self.client = Groq(api_key=api_key or os.environ["GROQ_API_KEY"])

    def transcribe(self, audio: bytes, filename: str = "audio.webm") -> str:
        try:
            resp = self.client.audio.transcriptions.create(
                file=(filename, audio),
                model=self.model,
                language="en",
                temperature=0,
                prompt=self.prompt,
            )
        except Exception as e:  # noqa: BLE001 - translated to STTError
            raise STTError(str(e)) from e
        return (getattr(resp, "text", "") or "").strip()
