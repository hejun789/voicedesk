"""Text-to-speech via a self-hosted Piper voice (ONNX, runs on CPU, free).

Replaces the browser's built-in `window.speechSynthesis`: that API exposes no
handle on its own audio output, so the browser's echo cancellation can never
treat it as a reference signal. Piper instead returns plain WAV bytes that
app.js plays through the page's own Web Audio graph — a real, page-controlled
audio node the browser's AEC *can* reference.
"""
import io
import os
import re
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


# A Piper voice speaks exactly one language: zh_CN-huayan phonemizes through
# espeak's `cmn` and was trained only on Mandarin phonetics, so Latin text
# inside a Chinese reply is forced through a sound inventory that has no way
# to render it. Measured: it says "Springfield" in 0.45s where the English
# voice takes 0.74s -- it is compressing the syllables away, not pronouncing
# them. Brand and street names ("Delta Dental", "Market Street") are exactly
# what a receptionist has to say, and they cannot be translated away, so the
# reply is split by script and each run is spoken by the voice that owns it.
#
# A run must CONTAIN a letter to be foreign, but a digit welded to letters
# comes along with it. Both halves matter: "200 号 4 室" has no letters, so
# it stays Chinese and is read in Chinese; "Suite 4B" is a single token, and
# splitting the digit out would send "4" to the Chinese voice and "B" to the
# English one, tearing a room number across two speakers mid-word.
_WORD = r"[A-Za-z0-9]*[A-Za-z][A-Za-z0-9'’.\-]*"
_LATIN_RUN = re.compile(rf"{_WORD}(?:[ 	]+{_WORD})*")
_CJK_RUN = re.compile(r"[㐀-䶿一-鿿]+")
_FOREIGN_RUN = {"zh": _LATIN_RUN, "en": _CJK_RUN}


def _segment_by_script(text: str, primary: str) -> list[tuple[str, str]]:
    """Split `text` into (language, run) pairs so each run can be spoken by
    the voice that owns its script.

    Returns a single segment when the text is all one script -- the common
    case, and the one that must stay free: no second model load, no
    concatenation, no added latency.
    """
    if not text:
        return []
    pattern = _FOREIGN_RUN.get(primary)
    if pattern is None:
        return [(primary, text)]
    foreign = "en" if primary == "zh" else "zh"
    segments: list[tuple[str, str]] = []
    last = 0
    for match in pattern.finditer(text):
        if match.start() > last:
            segments.append((primary, text[last:match.start()]))
        segments.append((foreign, match.group()))
        last = match.end()
    if not segments:
        return [(primary, text)]
    if last < len(text):
        segments.append((primary, text[last:]))
    return segments


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

    def _synthesize_run(self, text: str, language: str):
        """Raw PCM frames for one single-language run, plus the wave params
        they were produced with.

        Piper emits no audio at all for input with nothing speakable in it --
        a lone "、" between two brand names is enough -- and `wave` then fails
        on close with the opaque "# channels not specified", because no
        header was ever written. That is a silent run, not a failure: report
        it as empty frames and let the caller decide whether the whole reply
        came to nothing.
        """
        voice = self._voice_for(language)
        buf = io.BytesIO()
        try:
            with wave.open(buf, "wb") as wav_file:
                voice.synthesize_wav(text, wav_file)
        except wave.Error:
            return b"", None
        with wave.open(io.BytesIO(buf.getvalue())) as f:
            return f.readframes(f.getnframes()), f.getparams()

    def synthesize(self, text: str, language: str = DEFAULT_LANG) -> bytes:
        try:
            lang = normalize_lang(language)
            frames: list[bytes] = []
            params = None
            for run_lang, run_text in _segment_by_script(text, lang):
                run_frames, run_params = self._synthesize_run(run_text, run_lang)
                if not run_frames:
                    continue
                if params is None:
                    params = run_params
                elif (run_params.framerate, run_params.sampwidth,
                      run_params.nchannels) != (params.framerate,
                                                params.sampwidth,
                                                params.nchannels):
                    # Concatenating raw PCM is only valid while every voice
                    # agrees on rate, width and channel count (all current
                    # Piper medium voices are 22050Hz mono 16-bit). Refuse
                    # loudly rather than emit audio that plays at the wrong
                    # speed, which is far harder to diagnose from a demo.
                    raise TTSError(
                        f"voice format mismatch: {run_params} vs {params}")
                frames.append(run_frames)
            if params is None:
                raise TTSError(f"no audio produced for {text!r}")
            out = io.BytesIO()
            with wave.open(out, "wb") as wav_file:
                wav_file.setnchannels(params.nchannels)
                wav_file.setsampwidth(params.sampwidth)
                wav_file.setframerate(params.framerate)
                wav_file.writeframes(b"".join(frames))
            return out.getvalue()
        except TTSError:
            raise
        except Exception as e:  # noqa: BLE001 - translated to TTSError
            raise TTSError(str(e)) from e
