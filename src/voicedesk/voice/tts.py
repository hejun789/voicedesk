"""Text-to-speech via a self-hosted Piper voice (ONNX, runs on CPU, free).

Replaces the browser's built-in `window.speechSynthesis`: that API exposes no
handle on its own audio output, so the browser's echo cancellation can never
treat it as a reference signal. Piper instead returns plain WAV bytes that
app.js plays through the page's own Web Audio graph — a real, page-controlled
audio node the browser's AEC *can* reference.
"""
import io
import os
import wave
from pathlib import Path
from typing import Protocol

from voicedesk.lang import DEFAULT_LANG, normalize_lang

# rhasspy/piper-voices (Hugging Face) lays voices out as
# {lang}/{lang_region}/{name}/{quality}/{lang_region}-{name}-{quality}.{onnx,onnx.json}
_VOICE_IDS = {
    "en": "en_US-lessac-medium",
    "zh": "zh_CN-huayan-medium",
}


def _voice_repo_path(voice_id: str) -> str:
    lang_region, name, quality = voice_id.split("-")
    lang = lang_region.split("_")[0]
    return f"{lang}/{lang_region}/{name}/{quality}/{voice_id}"


class TTSError(Exception):
    """Speech synthesis failed. The server degrades gracefully rather than
    crashing the call — see the /tts route in server.py."""


class TTSClient(Protocol):
    def synthesize(self, text: str, language: str = DEFAULT_LANG) -> bytes: ...
    # Returns a complete WAV file as bytes.


class FakeTTS:
    """Test double: returns a fixed WAV payload, recording every call made."""

    def __init__(self, wav: bytes = b"RIFF....FAKEWAVE"):
        self._wav = wav
        self.calls: list[tuple[str, str]] = []

    def synthesize(self, text: str, language: str = DEFAULT_LANG) -> bytes:
        self.calls.append((text, normalize_lang(language)))
        return self._wav


class PiperTTS:
    """Real synthesis via the `piper-tts` package. Voice models are large
    (60-115MB) binaries fetched separately by download_voices.py — never at
    request time, never committed to git."""

    def __init__(self, voices_dir: str | Path | None = None):
        self.voices_dir = Path(voices_dir or os.environ.get(
            "PIPER_VOICES_DIR", "voices"))
        self._voices: dict[str, object] = {}  # lazily loaded PiperVoice per language

    def _voice_for(self, language: str):
        lang = normalize_lang(language)
        if lang not in self._voices:
            voice_id = _VOICE_IDS[lang]
            model_path = self.voices_dir / f"{_voice_repo_path(voice_id)}.onnx"
            if not model_path.exists():
                raise TTSError(
                    f"Piper voice model missing: {model_path}. "
                    "Run `python -m voicedesk.voice.download_voices` first.")
            from piper import PiperVoice  # imported lazily so unit tests need
                                           # neither onnxruntime nor the model
            self._voices[lang] = PiperVoice.load(
                str(model_path), config_path=str(model_path) + ".json")
        return self._voices[lang]

    def synthesize(self, text: str, language: str = DEFAULT_LANG) -> bytes:
        try:
            voice = self._voice_for(language)
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wav_file:
                voice.synthesize_wav(text, wav_file)
            return buf.getvalue()
        except TTSError:
            raise
        except Exception as e:  # noqa: BLE001 - translated to TTSError
            raise TTSError(str(e)) from e
