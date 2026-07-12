import os
from typing import Protocol

DEFAULT_STT_MODEL = "whisper-large-v3-turbo"


class STTError(Exception):
    """Transcription failed (API error). The server degrades gracefully rather
    than crashing the call."""


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
    ):
        # Whisper draws on a SEPARATE rate-limit pool from the chat model, so
        # transcription does not compete with the agent's LLM quota.
        self.model = model or os.environ.get("GROQ_STT_MODEL", DEFAULT_STT_MODEL)
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
            )
        except Exception as e:  # noqa: BLE001 - translated to STTError
            raise STTError(str(e)) from e
        return (getattr(resp, "text", "") or "").strip()
