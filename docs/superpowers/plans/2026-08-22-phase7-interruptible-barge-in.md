# Phase 7: Interruptible Barge-In Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make VoiceDesk's "可打断" (interruptible) claim true end-to-end: replace the browser's opaque `speechSynthesis` TTS with a self-hosted Piper voice that plays through the page's own Web Audio graph (so real echo cancellation has a reference signal to work with), and let the caller interrupt the agent during `THINKING`, not just `SPEAKING`.

**Architecture:** Two independent tracks that both land in `turn-state.js`/`app.js`:
1. **TTS swap (backend + frontend):** a new `voicedesk.voice.tts` module (mirrors the existing `voicedesk.voice.stt` pattern: `TTSClient` protocol, `FakeTTS`, `PiperTTS`) backs a new `/tts` endpoint; `app.js` fetches it and plays the WAV through an `AudioBufferSourceNode` instead of `window.speechSynthesis`.
2. **THINKING barge-in (frontend only):** `turn-state.js` currently refuses to leave `THINKING` on `SPEECH_START`/`PTT_DOWN` because "the server cannot cancel an in-flight agent call." That's still true — this plan does not change `/turn` — but it does not need to be cancelled: the existing `heard_chars` / `truncate_last_reply` machinery (already shipped for `SPEAKING`-barge-in) can retroactively delete a reply the caller never heard once it lands. Letting the caller leave `THINKING` immediately and forcing `heard_chars=0` on the next turn reuses that machinery instead of adding a cancellation path.

**Tech Stack:** Python 3.11, FastAPI, `piper-tts` (ONNX voices, CPU, MIT/GPL — free, self-hosted), `huggingface_hub` (voice model download), vanilla JS (no build step), Web Audio API, `node:test`.

**Spec:** No separate spec doc — this plan's Architecture section is the spec, reached via conversation with the project owner (VoiceDesk resume claim review → root-cause read of `vad.js`/`turn-state.js`/`app.js` → comparison against Deepgram Flux and FastRTC's `ReplyOnPause`/`_close_generator` designs → owner chose the free/self-hosted TTS route).

## Global Constraints

- No new paid dependencies — TTS must be free and self-hostable (owner's explicit choice).
- `/turn`'s request/response contract does not change — it stays a synchronous, lock-serialized call; no cancellation is added there.
- **`echoSafe` is removed entirely** — the variable, the toggle button, its CSS, and the mic-suppression branch in `tick()`. Barge-in is always available; there is no user-facing switch and no "safe mode" fallback. (Owner's explicit call: the toggle existed only to expose a tradeoff that Task 6 eliminates at the root, and a toggle for an internal audio-plumbing detail is not something a caller should ever see.) `vad.js`'s echo-floor heuristics stay exactly as they are and become the second line of defense behind the browser's now-functional AEC — do not touch `vad.js` in this plan.
- The UI is redesigned (Task 7). It must stay a single hand-written `index.html` with inline `<style>` and no build step, no framework, and no external asset or font requests — the app is served from a container with no CDN access and the project has deliberately never had a bundler.
- Existing `create_app(stt, sessions, lock=None, limiter=None)` call sites must keep working unchanged — `tts` is added as a new optional kwarg, not a required positional, matching how `limiter` was added in Phase 4.
- Two languages throughout: `en` → `en_US-lessac-medium`, `zh` → `zh_CN-huayan-medium` (both from the `rhasspy/piper-voices` Hugging Face repo).

---

## File Structure

- `src/voicedesk/voice/tts.py` (new) — `TTSClient` protocol, `FakeTTS`, `PiperTTS`. Mirrors `stt.py`.
- `src/voicedesk/voice/download_voices.py` (new) — one-shot fetch of the two `.onnx`/`.onnx.json` voice files from Hugging Face into `voices/`. Run at Docker build time, not at request time.
- `src/voicedesk/voice/server.py` (modify) — add `/tts` route and a `tts` param to `create_app`.
- `src/voicedesk/voice/__main__.py` (modify) — wire a real `PiperTTS()` into `create_app`.
- `requirements.txt`, `Dockerfile` (modify) — new deps, voice download at build time.
- `src/voicedesk/voice/static/turn-state.js` (modify) — `SPEECH_START`/`PTT_DOWN` during `THINKING` now interrupts it.
- `src/voicedesk/voice/static/app.js` (modify) — new `DISCARD_PENDING_REPLY` action; `speak()`/`cancelSpeech()` rewritten around `/tts` + Web Audio instead of `speechSynthesis`; `echoSafe` removed entirely.
- `src/voicedesk/voice/static/index.html` (modify) — echo-safe button removed (Task 6), then fully redesigned (Task 7).
- `tests/test_voice_tts.py` (new), `tests/test_voice_server_tts.py` (new), `tests/js/turn-state.test.mjs` (modify).

**Running the tests** (both commands are run from the repo root; the implementer briefs repeat them):
- Python: `PYTHONPATH=src .venv/Scripts/python.exe -m pytest -q` — the package lives under `src/`, so `PYTHONPATH=src` is required or every import fails.
- JS: `node --test tests/js/turn-state.test.mjs tests/js/vad.test.mjs` — pass the files explicitly.
- Verified baseline before any of this plan's work: **296 Python + 58 JS, 0 failures.**

---

## Task 1: `voicedesk.voice.tts` — TTSClient / FakeTTS / PiperTTS

**Files:**
- Create: `src/voicedesk/voice/tts.py`
- Test: `tests/test_voice_tts.py`

**Interfaces:**
- Produces: `TTSClient` (Protocol, method `synthesize(text: str, language: str = DEFAULT_LANG) -> bytes`, returning a complete WAV file), `FakeTTS(wav: bytes = ...)` with `.calls: list[tuple[str, str]]`, `PiperTTS(voices_dir: str | Path | None = None)`, `TTSError(Exception)`, and module-level `_VOICE_IDS: dict[str, str]` / `_voice_repo_path(voice_id: str) -> str` (Task 3's `download_voices.py` consumes both of these).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_voice_tts.py
from pathlib import Path

import pytest

from voicedesk.voice.tts import FakeTTS, PiperTTS, TTSError, _voice_repo_path


def test_fake_tts_returns_configured_wav_and_records_calls():
    tts = FakeTTS(wav=b"RIFF-fake-wav-bytes")
    assert tts.synthesize("hello", "en") == b"RIFF-fake-wav-bytes"
    assert tts.calls == [("hello", "en")]


def test_fake_tts_normalizes_language():
    tts = FakeTTS()
    tts.synthesize("你好", "zh-CN")
    assert tts.calls == [("你好", "zh")]


def test_voice_repo_path_matches_huggingface_layout():
    # rhasspy/piper-voices lays voices out as
    # {lang}/{lang_region}/{name}/{quality}/{lang_region}-{name}-{quality}
    assert _voice_repo_path("en_US-lessac-medium") == \
        "en/en_US/lessac/medium/en_US-lessac-medium"
    assert _voice_repo_path("zh_CN-huayan-medium") == \
        "zh/zh_CN/huayan/medium/zh_CN-huayan-medium"


def test_piper_tts_raises_clear_error_when_model_missing(tmp_path):
    # No network, no `piper` package needed for this path: the missing-file
    # check happens before piper is ever imported.
    tts = PiperTTS(voices_dir=tmp_path)
    with pytest.raises(TTSError, match="Piper voice model missing"):
        tts.synthesize("hello", "en")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_voice_tts.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'voicedesk.voice.tts'`

- [ ] **Step 3: Write the implementation**

```python
# src/voicedesk/voice/tts.py
"""Text-to-speech via a self-hosted Piper voice (ONNX, runs on CPU, free).

Replaces the browser's built-in `window.speechSynthesis`: that API exposes no
handle on its own audio output, so the browser's echo cancellation can never
treat it as a reference signal (see app.js's `echoSafe`). Piper instead
returns plain WAV bytes that app.js plays through the page's own Web Audio
graph — a real, page-controlled audio node the browser's AEC *can* reference.
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_voice_tts.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/voicedesk/voice/tts.py tests/test_voice_tts.py
git commit -m "feat(voice): add Piper TTS client (voicedesk.voice.tts)"
```

---

## Task 2: `/tts` endpoint

**Files:**
- Modify: `src/voicedesk/voice/server.py`
- Test: `tests/test_voice_server_tts.py`

**Interfaces:**
- Consumes: `TTSClient`, `TTSError` from `voicedesk.voice.tts` (Task 1); `FakeTTS` from the same module for tests.
- Produces: `create_app(stt, sessions, lock=None, limiter=None, tts=None) -> FastAPI` — same signature as today plus one new optional kwarg at the end, so every existing call site (`__main__.py`, `test_voice_server.py`, `test_voice_server_interrupt.py`, `test_voice_server_chinese.py`, `test_voice_static_cache.py`, `test_voice_server_limits.py`) needs **no changes**. New route: `POST /tts` (form fields `text`, `lang`) → `200` with `audio/wav` body, or `503` with JSON `{"detail": "tts_unavailable"}` / `{"detail": "tts_failed"}`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_voice_server_tts.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_voice_server_tts.py -v`
Expected: FAIL — `/tts` returns 404 (route does not exist yet)

- [ ] **Step 3: Add the route**

In `src/voicedesk/voice/server.py`, add to the imports:

```python
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response
```

(`HTTPException` and `Response` are new; keep the rest of the existing import line as-is.)

Add alongside the other `voicedesk.voice.stt` import:

```python
from voicedesk.voice.tts import TTSError
```

Change the signature and add the route (place it right after the `/turn` route, before `app.mount(...)`):

```python
def create_app(stt, sessions, lock=None, limiter=None, tts=None) -> FastAPI:
```

```python
    @app.post("/tts")
    async def synthesize(text: str = Form(...), lang: str = Form(DEFAULT_LANG)):
        if tts is None:
            raise HTTPException(status_code=503, detail="tts_unavailable")
        lang_norm = normalize_lang(lang)
        try:
            wav = await run_in_threadpool(tts.synthesize, text, lang_norm)
        except TTSError as e:
            print(f"[voice] TTS error: {e}", file=sys.stderr, flush=True)
            raise HTTPException(status_code=503, detail="tts_failed")
        return Response(content=wav, media_type="audio/wav")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_voice_server_tts.py -v`
Expected: PASS (4 tests)

Then run the full suite to confirm the new optional kwarg didn't disturb any existing route:

Run: `pytest tests/test_voice_server.py tests/test_voice_server_interrupt.py tests/test_voice_server_chinese.py tests/test_voice_static_cache.py tests/test_voice_server_limits.py -v`
Expected: PASS, unchanged

- [ ] **Step 5: Commit**

```bash
git add src/voicedesk/voice/server.py tests/test_voice_server_tts.py
git commit -m "feat(voice): add POST /tts endpoint"
```

---

## Task 3: Wire real Piper TTS into deployment

**Files:**
- Create: `src/voicedesk/voice/download_voices.py`
- Modify: `requirements.txt`, `src/voicedesk/voice/__main__.py`, `Dockerfile`, `.gitignore`, `src/voicedesk/voice/static/index.html` (one stale doc line)

**Interfaces:**
- Consumes: `_VOICE_IDS`, `_voice_repo_path` from `voicedesk.voice.tts` (Task 1).
- Produces: a `voices/` directory populated with both languages' `.onnx`/`.onnx.json` files, ready for `PiperTTS(voices_dir="voices")` to load.

- [ ] **Step 1: Add dependencies**

In `requirements.txt`, add two lines:

```
piper-tts==1.4.2
huggingface-hub==0.25.2
```

- [ ] **Step 2: Write the download script**

```python
# src/voicedesk/voice/download_voices.py
"""Fetches the Piper voice models PiperTTS needs.

Run once at container build time (see Dockerfile) — these are 60-115MB
binaries and do not belong in git.

    python -m voicedesk.voice.download_voices
"""
import os
from pathlib import Path

from voicedesk.voice.tts import _VOICE_IDS, _voice_repo_path

REPO_ID = "rhasspy/piper-voices"


def download_all(voices_dir: str | Path | None = None) -> None:
    from huggingface_hub import hf_hub_download  # build-time only dependency

    voices_dir = Path(voices_dir or os.environ.get("PIPER_VOICES_DIR", "voices"))
    voices_dir.mkdir(parents=True, exist_ok=True)
    for voice_id in _VOICE_IDS.values():
        repo_path = _voice_repo_path(voice_id)
        for suffix in (".onnx", ".onnx.json"):
            hf_hub_download(
                repo_id=REPO_ID,
                filename=f"{repo_path}{suffix}",
                local_dir=voices_dir,
            )


if __name__ == "__main__":
    download_all()
```

(`hf_hub_download(..., local_dir=voices_dir)` places the file at `voices_dir/{repo_path}{suffix}` — the exact same nested path `PiperTTS._voice_for` looks under, so no renaming step is needed.)

- [ ] **Step 3: Wire it into `__main__.py`**

In `src/voicedesk/voice/__main__.py`, add the import:

```python
from voicedesk.voice.tts import PiperTTS
```

Change:

```python
    app = create_app(GroqWhisper(), sessions, limiter=limiter)
```

to:

```python
    app = create_app(GroqWhisper(), sessions, limiter=limiter, tts=PiperTTS())
```

- [ ] **Step 4: Update the Dockerfile**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# espeak-ng backs Piper's phonemizer. Recent piper-tts wheels bundle their
# own copy, but installing the system package too is cheap insurance against
# a platform where the bundled one doesn't load.
RUN apt-get update && apt-get install -y --no-install-recommends espeak-ng \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
# answer_faq opens these by a path relative to the working directory, so they
# must sit at /app next to where the app runs.
COPY clinic_info.md clinic_info.zh.md ./

ENV PYTHONPATH=/app/src
ENV PORT=7860
# The public demo runs on 8b for its higher free-tier limits; override in the Space to raise quality.
ENV GROQ_MODEL=llama-3.1-8b-instant
ENV PIPER_VOICES_DIR=/app/voices

# Fetched at build time, not request time: these are ~150MB combined and
# would make the container's first reply take minutes instead of seconds.
RUN python -m voicedesk.voice.download_voices

EXPOSE 7860

CMD ["python", "-m", "voicedesk.voice"]
```

- [ ] **Step 5: Keep the voices directory out of git**

In `.gitignore`, add:

```
voices/
```

- [ ] **Step 6: Fix the now-stale Web Speech API line**

In `src/voicedesk/voice/static/index.html`, find the line (near the top, in the visitor-facing instructions):

```
Use Chrome or Edge (needs the microphone and the Web Speech API).
```

Replace with:

```
Use Chrome or Edge (needs microphone access).
```

- [ ] **Step 7: Manual verification (no automated test — this step is deploy wiring, not app logic)**

```bash
pip install -r requirements.txt
python -m voicedesk.voice.download_voices
python -m voicedesk.voice   # requires GROQ_API_KEY in .env, per existing README
```

In another terminal:

```bash
curl -s -X POST http://127.0.0.1:7860/tts -d "text=Hello there&lang=en" -o /tmp/out.wav
file /tmp/out.wav   # expect: "RIFF (little-endian) data, WAVE audio"
curl -s -X POST http://127.0.0.1:7860/tts -d "text=你好&lang=zh" -o /tmp/out_zh.wav
file /tmp/out_zh.wav
```

Play both files and confirm they're intelligible speech in the right language before moving on — Task 6 depends on this endpoint actually working.

- [ ] **Step 8: Commit**

```bash
git add requirements.txt src/voicedesk/voice/download_voices.py \
        src/voicedesk/voice/__main__.py Dockerfile .gitignore \
        src/voicedesk/voice/static/index.html
git commit -m "feat(voice): wire self-hosted Piper TTS into the running app"
```

---

## Task 4: `turn-state.js` — barge-in during THINKING

**Files:**
- Modify: `src/voicedesk/voice/static/turn-state.js`
- Test: `tests/js/turn-state.test.mjs`

**Interfaces:**
- Produces: a new action name `"DISCARD_PENDING_REPLY"`, returned by `next()` alongside `"START_RECORDING"` when `SPEECH_START`/`PTT_DOWN` arrives during `THINKING`. Task 5 (`app.js`) consumes this action name.

- [ ] **Step 1: Replace the outdated test and add a new one**

In `tests/js/turn-state.test.mjs`, replace this test (it asserts the old, no-longer-true behavior):

```js
test("speech during thinking is ignored — no barge-in while the LLM runs", () => {
  let s = next(initialState(), "ARM").state;
  s = next(s, "SPEECH_START").state;
  s = next(s, "SPEECH_END").state;
  assert.equal(s.name, THINKING);
  const { state, actions } = next(s, "SPEECH_START");
  assert.equal(state.name, THINKING);
  assert.deepEqual(actions, []);
});
```

with:

```js
test("speech during thinking interrupts it: the caller can talk over the LLM call", () => {
  // The server still cannot cancel the in-flight agent call (see /turn), so
  // it runs to completion regardless -- but the caller does not have to wait
  // for it. DISCARD_PENDING_REPLY tells app.js to report zero heard
  // characters on the next turn it sends, which the server's
  // truncate_last_reply() uses to drop the stale reply from history once it
  // lands (see the REPLY test below for the other half of this).
  let s = next(initialState(), "ARM").state;
  s = next(s, "SPEECH_START").state;
  s = next(s, "SPEECH_END").state;
  assert.equal(s.name, THINKING);
  const { state, actions } = next(s, "SPEECH_START");
  assert.equal(state.name, LISTENING);
  assert.equal(state.capturing, true);
  assert.deepEqual(actions, ["DISCARD_PENDING_REPLY", "START_RECORDING"]);
});

test("push-to-talk: pressing again during thinking interrupts it the same way", () => {
  let s = initialState("ptt");
  s = next(s, "PTT_DOWN").state;
  s = next(s, "PTT_UP").state;
  assert.equal(s.name, THINKING);
  const { state, actions } = next(s, "PTT_DOWN");
  assert.equal(state.name, LISTENING);
  assert.equal(state.capturing, true);
  assert.deepEqual(actions, ["DISCARD_PENDING_REPLY", "START_RECORDING"]);
});

test("a reply that lands after a thinking interrupt is not spoken", () => {
  // Mirrors "a late reply does not resurrect speaking after a barge-in"
  // above, but for a barge-in that happened during THINKING instead of
  // SPEAKING.
  let s = next(initialState(), "ARM").state;
  s = next(s, "SPEECH_START").state;
  s = next(s, "SPEECH_END").state;
  s = next(s, "SPEECH_START").state;   // interrupts THINKING
  assert.equal(s.name, LISTENING);
  const { state, actions } = next(s, "REPLY");
  assert.equal(state.name, LISTENING);
  assert.deepEqual(actions, []);
});
```

- [ ] **Step 2: Run tests to verify the new/changed ones fail**

Run: `node --test tests/js/turn-state.test.mjs`
Expected: FAIL on the three tests above — the interrupt still returns `THINKING`/`[]` and the third test's setup assertion (`s.name === LISTENING` after the second `SPEECH_START`) doesn't hold yet.

- [ ] **Step 3: Implement the reducer change**

In `src/voicedesk/voice/static/turn-state.js`, replace:

```js
    // Deliberately NOT allowed while THINKING: the server cannot cancel an
    // in-flight agent call, so interrupting there would desync history.
    if (name === LISTENING || name === IDLE) {
```

with:

```js
    // Barge-in during THINKING: the server still cannot cancel the in-flight
    // agent call, so it runs to completion and its reply lands in history as
    // normal — but the caller does not have to sit through it. The REPLY
    // handler below already drops a reply that arrives after the caller has
    // moved off THINKING; DISCARD_PENDING_REPLY additionally tells the next
    // turn's request to report zero heard characters, so the server's
    // truncate_last_reply() removes it from the agent's own history too —
    // otherwise the agent would believe it said something the caller never
    // heard and could reference it later.
    if (name === THINKING) {
      return {
        state: { ...state, name: LISTENING, capturing: true },
        actions: ["DISCARD_PENDING_REPLY", "START_RECORDING"],
      };
    }
    if (name === LISTENING || name === IDLE) {
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `node --test tests/js/turn-state.test.mjs`
Expected: PASS, all tests (existing + 3 new/changed)

- [ ] **Step 5: Commit**

```bash
git add src/voicedesk/voice/static/turn-state.js tests/js/turn-state.test.mjs
git commit -m "feat(voice-ui): allow barge-in during THINKING, not just SPEAKING"
```

---

## Task 5: `app.js` — consume `DISCARD_PENDING_REPLY`

**Files:**
- Modify: `src/voicedesk/voice/static/app.js`

**Interfaces:**
- Consumes: the `"DISCARD_PENDING_REPLY"` action from Task 4.

No automated test: `app.js` is browser-API glue with no unit test coverage in this codebase (only `vad.js` and `turn-state.js` are pure enough to run under `node:test` — see their file-header comments). Verified manually in Step 2.

- [ ] **Step 1: Add the action handler**

In `src/voicedesk/voice/static/app.js`, change:

```js
function runAction(action) {
  if (action === "START_RECORDING") startRecording();
  else if (action === "STOP_AND_SEND") stopRecordingAndSend();
  else if (action === "CANCEL_TTS") cancelSpeech();
  else if (action === "SPEAK") speak(pendingReply, pendingLang);
}
```

to:

```js
function runAction(action) {
  if (action === "START_RECORDING") startRecording();
  else if (action === "STOP_AND_SEND") stopRecordingAndSend();
  else if (action === "CANCEL_TTS") cancelSpeech();
  else if (action === "SPEAK") speak(pendingReply, pendingLang);
  else if (action === "DISCARD_PENDING_REPLY") heardChars = 0;
}
```

- [ ] **Step 2: Manual verification**

Run the app locally (`python -m voicedesk.voice`), open `http://127.0.0.1:7860/?debug=1`, hands-free mode:

1. Ask a question. While the button reads "Thinking…", start talking again immediately.
2. Confirm: the button switches straight to "Listening…"/"Recording…" — you are not forced to wait.
3. Confirm: when the first (interrupted) reply eventually comes back from the server, the UI never flashes "Speaking…" for it and it is not read aloud.
4. Open the Network tab: confirm the *next* `/turn` request's form data includes `heard_chars=0`.
5. Repeat in push-to-talk mode (press again while "Thinking…" is shown).

- [ ] **Step 3: Commit**

```bash
git add src/voicedesk/voice/static/app.js
git commit -m "feat(voice-ui): wire DISCARD_PENDING_REPLY into the action dispatcher"
```

---

## Task 6: `app.js` — replace `speechSynthesis` with Piper over Web Audio

**Files:**
- Modify: `src/voicedesk/voice/static/app.js`

**Interfaces:**
- Consumes: `POST /tts` (Task 2/3).

Depends on Task 2 for the endpoint to exist and Task 3 for it to return real, intelligible audio — do the manual verification in Step 3 only after Task 3's Step 7 has already confirmed `/tts` works standalone.

No automated test, same reasoning as Task 5.

- [ ] **Step 1: Replace the TTS state variables**

In `src/voicedesk/voice/static/app.js`, near the existing declaration:

```js
let heardChars = null;
let spokenText = "";
```

add:

```js
let ttsSource = null;      // the AudioBufferSourceNode currently playing, or null
let ttsStartedAt = 0;      // performance.now() when the current reply started playing
let ttsDurationMs = 0;     // total duration (ms) of the current reply's audio
```

- [ ] **Step 2: Rewrite `speak()` and `cancelSpeech()`**

Replace the entire `--- text to speech ---` section (both `speak()` and `cancelSpeech()`) with:

```js
// --- text to speech --------------------------------------------------------

async function speak(text, replyLang) {
  spokenText = text;
  heardChars = null;
  const ctx = audioCtx;   // captured now: a hang-up during the awaits below
                           // can null the global before this resumes
  let buffer;
  try {
    const res = await fetch("/tts", {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: new URLSearchParams({ text, lang: replyLang }),
    });
    if (!res.ok) throw new Error(`tts failed: ${res.status}`);
    const bytes = await res.arrayBuffer();
    if (!ctx || ctx.state === "closed") return;   // hung up mid-fetch
    buffer = await ctx.decodeAudioData(bytes);
  } catch (err) {
    // Synthesis, network, or decode failed -- nothing to speak. Move on
    // rather than leaving the call stuck in SPEAKING forever.
    spokenText = "";
    dispatch("TTS_END");
    return;
  }
  // The caller may have barged in (or hung up) while the round trip above
  // was in flight; do not start playback for a reply the state machine has
  // already moved past.
  if (state.name !== SPEAKING) return;
  ttsSource = ctx.createBufferSource();
  ttsSource.buffer = buffer;
  ttsSource.connect(ctx.destination);
  ttsDurationMs = buffer.duration * 1000;
  ttsStartedAt = performance.now();
  ttsSource.onended = () => {
    ttsSource = null;
    heardChars = null;
    spokenText = "";
    dispatch("TTS_END");
  };
  ttsSource.start();
}

function cancelSpeech() {
  if (!ttsSource) return;
  // Deterministic from elapsed playback time -- unlike the speechSynthesis
  // onboundary event this replaces, which Chrome's network-backed zh-CN
  // voice never fired, silently disabling heard_chars tracking for that
  // language.
  const elapsedMs = performance.now() - ttsStartedAt;
  const fraction = ttsDurationMs > 0 ? Math.min(1, elapsedMs / ttsDurationMs) : 0;
  heardChars = Math.round(spokenText.length * fraction);
  ttsSource.onended = null;   // suppress TTS_END: this is a barge-in, not natural completion
  ttsSource.stop();
  ttsSource = null;
  spokenText = "";
}
```

- [ ] **Step 3: Remove `echoSafe` entirely**

The toggle existed solely because `window.speechSynthesis` gave the browser's AEC nothing to reference. Step 2 removed that cause, so the toggle goes too — barge-in is now always available.

**3a.** Delete the `echoSafe` declaration and its whole comment block:

```js
// User toggle: while true, the mic is ignored entirely whenever the agent is
// speaking, so it can never hear (and react to) its own voice. Off by
// default -- it trades away barge-in for a guaranteed-correct fallback on
// unknown hardware where the heuristics in vad.js's echo floor may not hold.
// Default ON. Five rounds of magnitude-based echo rejection (dip tolerance,
// adaptive floor, grace window, peak-hold) each fixed a real live-mic failure
// and each was eventually beaten by another one -- because the Web Speech API
// exposes no handle on its own audio stream, so real echo cancellation is
// impossible here; every fix is a heuristic on a losing information-theoretic
// footing. For a public demo on hardware we don't control, "barge-in never
// works" (this toggle OFF) is worse than "barge-in usually doesn't happen"
// (this toggle ON): the former guarantees the agent never talks over itself,
// which is the failure mode that actually breaks a demo. Visitors on
// headphones, or who want to try interrupting it, can switch it off.
let echoSafe = true;
```

**3b.** Delete the `echoSafeBtn` element lookup near the top of the file:

```js
const echoSafeBtn = document.getElementById("echoSafeToggle");
```

**3c.** Delete the whole `echoSafeBtn.addEventListener("click", ...)` block (it is the block whose body starts `echoSafe = !echoSafe;` and ends with the long comment about `tick()` re-checking every frame).

**3d.** In `render()`, delete these two lines:

```js
  echoSafeBtn.textContent = L.echoSafe;
  echoSafeBtn.classList.toggle("active", echoSafe);
```

**3e.** Delete the `echoSafe` key from **both** language objects in `LABELS` (the `en` one and the `zh` one — they read `echoSafe: "Echo-safe" }` and `echoSafe: "防回声" }`; remove the key and leave the rest of each object intact).

**3f.** In `startVadLoop()`'s `tick()`, delete the entire suppression branch — from the comment `// Echo-safe mode: while the agent is speaking, ignore the mic` down to and including its closing `}` (the block containing `if (echoSafe && state.name === SPEAKING) { ... return; }` and the `discardRecording()`/`wasSuppressed = true;` bookkeeping inside it).

**3g.** Also in `tick()`, delete the now-unreachable `if (wasSuppressed) { ... }` recovery block that follows it (the one that recreates the VAD with `vad = createEnergyVAD();`). Nothing sets `wasSuppressed` to `true` any more.

**3h.** Delete the `wasSuppressed` declaration and its comment:

```js
// Tracks the previous frame's suppression state (see tick() below) so a
// falling edge -- suppression just ended -- can be detected and the
// detector's stale internal state cleared before real audio is fed to it
// again.
let wasSuppressed = false;
```

**3i.** In `stopVadLoop()`, delete the line `wasSuppressed = false;`.

**3j.** In `renderDebug()`'s array of overlay lines, delete the line reading:

```js
    `echoSafe   ${echoSafe}`,
```

**3k.** In `index.html`, delete the echo-safe button's markup:

```html
  <div id="echoSafeRow">
    <button id="echoSafeToggle" class="echo-safe">Echo-safe</button>
  </div>
```

and drop `.echo-safe` / `#echoSafeRow` from the two CSS rules that mention them, so they read:

```css
    #modes { display: flex; gap: .5rem; margin-bottom: 1rem; }
    .mode { flex: 1; padding: .5rem; border-radius: 8px; border: 1px solid #ccc;
            background: #fff; cursor: pointer; font-size: .85rem; }
    .mode.active { background: #475569; border-color: #475569; color: #fff; }
```

(Task 7 replaces this stylesheet wholesale; this keeps the page valid in between.)

**3l.** In `index.html`, replace the now-wrong hint paragraph:

```html
  <p class="hint">Click "Start call", then just start speaking — the receptionist
    listens for when you stop. "Echo-safe" is on by default so it never talks over
    itself; turn it off (e.g. on headphones) to interrupt it mid-sentence. Try:
    <em>"Book me Monday July 13th 2026 at 9am, Jane Doe, 5551234, for a cleaning."</em></p>
```

with:

```html
  <p class="hint">Click "Start call", then just start speaking — the receptionist
    listens for when you stop, and you can talk over it any time to interrupt. Try:
    <em>"Book me Monday July 13th 2026 at 9am, Jane Doe, 5551234, for a cleaning."</em></p>
```

**3m.** Verify nothing references the removed names — this must print no matches:

```bash
grep -n "echoSafe\|wasSuppressed" src/voicedesk/voice/static/app.js src/voicedesk/voice/static/index.html
```

- [ ] **Step 4: Manual verification**

With Task 3's `/tts` endpoint already confirmed working (its own Step 7):

1. Load `http://127.0.0.1:7860/?debug=1`, hands-free mode, English. Ask a question and confirm the reply is spoken aloud in Piper's voice (not the old system TTS voice — a clearly different voice is the signal this actually switched over).
2. Switch to 中文 and repeat — confirm Chinese speech plays.
3. Confirm the Echo-safe button is gone from the page and the debug overlay no longer shows an `echoSafe` line.
4. **The load-bearing test — barge-in on speakers, not headphones.** While the agent is speaking, say something. Confirm: `CANCEL_TTS` fires (audio stops immediately), and a plausible `heard_chars` value (roughly proportional to how far into the reply you interrupted) shows up in the next `/turn` request's form data — now driven by real elapsed playback time instead of a `speechSynthesis` boundary event, so this works for 中文 too, which it never did before.
5. **The failure mode this replaces — same test, agent must NOT interrupt itself.** Ask a question that gets a long reply and stay completely silent through it on speakers. Confirm the agent speaks the whole reply without cutting itself off. This is the exact self-echo failure the removed `echoSafe` toggle existed to prevent; it is now the browser's AEC plus `vad.js`'s echo floor holding the line instead. Watch the debug overlay's `floor`/`requirement` values while it speaks. **If the agent interrupts itself here, stop and report it — do not work around it by re-adding the toggle.**
6. Hang up mid-reply (click "tap to end call" while SPEAKING). Confirm no console errors from a stray `decodeAudioData` call on a closed `AudioContext`.

- [ ] **Step 5: Commit**

```bash
git add src/voicedesk/voice/static/app.js
git commit -m "feat(voice-ui): play TTS through Web Audio instead of speechSynthesis"
```

---

## Task 7: Redesign the UI

**Files:**
- Modify: `src/voicedesk/voice/static/index.html` (replaced wholesale)

**Interfaces:**
- Consumes: nothing new. This task is presentation only — **`app.js` must not be edited.** Every hook it reaches for has to survive the redesign unchanged.

### The contract with `app.js` (read before writing any markup)

`app.js` queries and mutates the DOM directly. Break any of these and the app silently stops working:

| Hook | Used by `app.js` as | Must stay |
|---|---|---|
| `#talk` | `talk.textContent = ...`, `talk.title = ...`, `talk.classList.add/remove("recording"\|"listening"\|"speaking")`, click + mouse/touch handlers | a `<button id="talk">` |
| `#transcript`, `#reply`, `#timings` | `.textContent = ...` | elements with those ids |
| `.lang[data-lang]` | `document.querySelectorAll(".lang")`, reads `btn.dataset.lang`, toggles `.active` | two buttons, `data-lang="en"` / `data-lang="zh"` |
| `.mode[data-mode]` | `document.querySelectorAll(".mode")`, reads `btn.dataset.mode`, toggles `.active` | two buttons, `data-mode="hands-free"` / `data-mode="ptt"` |

**The trap:** `app.js` assigns `talk.textContent`, `transcript.textContent`, `reply.textContent`, and `timings.textContent`. Assigning `textContent` **destroys every child node**. So these four elements must never contain decorative child elements (no `<span>` for an icon, no pulse-ring `<div>`) — anything of the sort is wiped the first time state changes. All visual effects on the call button must come from CSS on the button itself (`box-shadow`, `background`, pseudo-elements) or from an ancestor, never from a child element.

`render()` calls `talk.classList.remove("recording", "listening", "speaking")` and then re-adds one, so any *other* class on `#talk` survives — a base styling class is safe.

- [ ] **Step 1: Replace `index.html` with the redesign**

Write `src/voicedesk/voice/static/index.html` with exactly this content:

```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>VoiceDesk — BrightSmile Dental</title>
  <style>
    /* Light palette on :root, dark redefined under prefers-color-scheme, so
       the page follows the visitor's OS setting with no toggle and no JS. */
    :root {
      --bg: #f4f6f9;
      --surface: #ffffff;
      --surface-2: #f1f3f7;
      --text: #171a20;
      --muted: #6b7280;
      --border: #e3e7ee;
      --accent: #2563eb;
      --accent-soft: #e8efff;
      --idle: #2563eb;      --idle-bg: #eff5ff;
      --live: #16a34a;      --live-bg: #eefbf2;
      --rec: #dc2626;       --rec-bg: #fef1f1;
      --think: #7c3aed;     --think-bg: #f4efff;
      --speak: #d97706;     --speak-bg: #fff8ec;
      --shadow: 0 1px 2px rgba(16,24,40,.04), 0 8px 24px rgba(16,24,40,.06);
    }
    @media (prefers-color-scheme: dark) {
      :root {
        --bg: #0e1014;
        --surface: #171a21;
        --surface-2: #1e222b;
        --text: #e8eaee;
        --muted: #98a0ad;
        --border: #272c36;
        --accent: #6592ff;
        --accent-soft: #1b2540;
        --idle: #6592ff;    --idle-bg: #172038;
        --live: #4ade80;    --live-bg: #12261a;
        --rec: #f87171;     --rec-bg: #2a1618;
        --think: #a78bfa;   --think-bg: #221a33;
        --speak: #fbbf24;   --speak-bg: #2a2010;
        --shadow: 0 1px 2px rgba(0,0,0,.3), 0 8px 24px rgba(0,0,0,.35);
      }
    }

    * { box-sizing: border-box; }
    body {
      margin: 0; padding: 2.5rem 1rem 4rem;
      background: var(--bg); color: var(--text);
      font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
      line-height: 1.55;
      -webkit-font-smoothing: antialiased;
    }
    .shell { max-width: 34rem; margin: 0 auto; }

    header { margin-bottom: 1.5rem; }
    .brand { display: flex; align-items: center; gap: .625rem; }
    /* Decorative only, and deliberately NOT inside #talk or any element whose
       textContent app.js overwrites. */
    .brand .dot {
      width: .625rem; height: .625rem; border-radius: 50%;
      background: var(--accent); flex: none;
    }
    h1 { font-size: 1.15rem; font-weight: 650; margin: 0; letter-spacing: -.01em; }
    .sub { margin: .5rem 0 0; color: var(--muted); font-size: .9rem; }
    .sub em { color: var(--text); font-style: normal; font-weight: 500; }

    .card {
      background: var(--surface); border: 1px solid var(--border);
      border-radius: 18px; padding: 1.25rem; box-shadow: var(--shadow);
    }

    /* Segmented controls: language and mode are settings, not the main event,
       so they read as one quiet strip above the call button. */
    .controls { display: flex; gap: .5rem; margin-bottom: 1rem; flex-wrap: wrap; }
    .seg {
      display: flex; gap: .1875rem; flex: 1 1 12rem; min-width: 0;
      background: var(--surface-2); border: 1px solid var(--border);
      border-radius: 11px; padding: .1875rem;
    }
    .seg button {
      flex: 1; min-width: 0; padding: .4375rem .5rem;
      font: inherit; font-size: .8125rem; font-weight: 500;
      color: var(--muted); background: none; border: 0; border-radius: 8px;
      cursor: pointer; white-space: nowrap; overflow: hidden;
      text-overflow: ellipsis; transition: background .15s, color .15s;
    }
    .seg button:hover { color: var(--text); }
    .seg button.active {
      background: var(--surface); color: var(--text);
      box-shadow: 0 1px 2px rgba(16,24,40,.08);
    }

    /* The call button. All state feedback is background/border/shadow on the
       button itself -- never child elements, which app.js's textContent
       assignment would destroy. */
    #talk {
      display: block; width: 100%; min-height: 5.5rem;
      padding: 1.25rem 1rem; font: inherit; font-size: 1.05rem; font-weight: 600;
      border-radius: 14px; border: 1.5px solid var(--idle);
      background: var(--idle-bg); color: var(--idle);
      cursor: pointer; user-select: none; -webkit-user-select: none;
      transition: background .18s, border-color .18s, color .18s;
    }
    #talk:disabled { opacity: .5; cursor: default; }
    #talk.listening {
      border-color: var(--live); background: var(--live-bg); color: var(--live);
      animation: breathe 2.4s ease-in-out infinite;
    }
    #talk.recording {
      border-color: var(--rec); background: var(--rec-bg); color: var(--rec);
      animation: pulse 1.4s ease-out infinite;
    }
    #talk.speaking {
      border-color: var(--speak); background: var(--speak-bg); color: var(--speak);
    }
    /* Listening: a slow, calm breath -- the app is waiting, not urgent. */
    @keyframes breathe {
      0%, 100% { box-shadow: 0 0 0 0 rgba(22,163,74,.16); }
      50%      { box-shadow: 0 0 0 .5rem rgba(22,163,74,0); }
    }
    /* Recording: a faster ring so it is unmistakable that the mic is live. */
    @keyframes pulse {
      0%   { box-shadow: 0 0 0 0 rgba(220,38,38,.28); }
      100% { box-shadow: 0 0 0 .875rem rgba(220,38,38,0); }
    }
    @media (prefers-reduced-motion: reduce) {
      #talk.listening, #talk.recording { animation: none; }
    }

    /* Conversation. Two speakers, so they are visually opposed rather than
       stacked in identical grey slots. */
    .thread { margin-top: 1.25rem; display: grid; gap: 1rem; }
    .msg { display: grid; gap: .3125rem; }
    .who {
      font-size: .6875rem; font-weight: 600; text-transform: uppercase;
      letter-spacing: .07em; color: var(--muted);
    }
    .bubble {
      padding: .75rem .875rem; border-radius: 12px;
      background: var(--surface-2); font-size: .9375rem;
      overflow-wrap: anywhere;
    }
    .msg.you .bubble {
      background: var(--accent-soft); font-weight: 500;
    }
    #timings {
      margin-top: .25rem; font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: .6875rem; color: var(--muted); min-height: 1em;
    }
  </style>
</head>
<body>
  <div class="shell">
    <header>
      <div class="brand">
        <span class="dot"></span>
        <h1>VoiceDesk — BrightSmile Dental</h1>
      </div>
      <p class="sub">Start the call, then just speak — it listens for when you
        stop, and you can talk over it any time to interrupt.
        Try: <em>“Book me Monday July 13th 2026 at 9am, Jane Doe, 5551234,
        for a cleaning.”</em></p>
    </header>

    <div class="card">
      <div class="controls">
        <div class="seg" id="langs">
          <button class="lang active" data-lang="en">English</button>
          <button class="lang" data-lang="zh">中文</button>
        </div>
        <div class="seg" id="modes">
          <button class="mode active" data-mode="hands-free">Hands-free</button>
          <button class="mode" data-mode="ptt">Hold to talk</button>
        </div>
      </div>

      <button id="talk">Hold to talk</button>

      <div class="thread">
        <div class="msg you">
          <div class="who">You said</div>
          <div id="transcript" class="bubble">—</div>
        </div>
        <div class="msg">
          <div class="who">Receptionist</div>
          <div id="reply" class="bubble">—</div>
        </div>
      </div>
      <div id="timings"></div>
    </div>
  </div>

  <script type="module" src="/static/app.js"></script>
</body>
</html>
```

- [ ] **Step 2: Verify every `app.js` hook survived**

Each of these must print a match (run from the repo root):

```bash
grep -c 'id="talk"' src/voicedesk/voice/static/index.html
grep -c 'id="transcript"' src/voicedesk/voice/static/index.html
grep -c 'id="reply"' src/voicedesk/voice/static/index.html
grep -c 'id="timings"' src/voicedesk/voice/static/index.html
grep -c 'class="lang active" data-lang="en"' src/voicedesk/voice/static/index.html
grep -c 'data-lang="zh"' src/voicedesk/voice/static/index.html
grep -c 'data-mode="hands-free"' src/voicedesk/voice/static/index.html
grep -c 'data-mode="ptt"' src/voicedesk/voice/static/index.html
```

Expected: `1` for every one.

And this must print **no** matches — Task 6 removed the toggle, and it must not come back:

```bash
grep -n "echoSafe\|echo-safe" src/voicedesk/voice/static/index.html
```

- [ ] **Step 3: Confirm the static-file test still passes**

Run: `PYTHONPATH=src .venv/Scripts/python.exe -m pytest tests/test_voice_static_cache.py -q`
Expected: PASS (it serves `index.html` and asserts the no-store cache header; a malformed file would still pass this, which is why Step 2's grep checks exist).

- [ ] **Step 4: Manual verification**

Run the app (`PYTHONPATH=src .venv/Scripts/python.exe -m voicedesk.voice`) and load `http://127.0.0.1:7860/`:

1. Click through a full call. Confirm the button's colour changes across idle → listening (green, slow breath) → recording (red, faster pulse) → thinking → speaking (amber), and that its label text is correct in each state — a wiped label means a `textContent` child was introduced.
2. Switch to 中文 and confirm the labels translate and the segmented control's active pill moves.
3. Switch to "Hold to talk" and confirm press-and-hold still records.
4. Narrow the window to ~360px wide and confirm nothing overflows horizontally.
5. Flip the OS to dark mode and confirm the page follows with readable contrast.

- [ ] **Step 5: Commit**

```bash
git add src/voicedesk/voice/static/index.html
git commit -m "feat(voice-ui): redesign the call interface"
```

---

## Self-Review Notes

- **Spec coverage:** TTS swap → Tasks 1-3, 6. THINKING barge-in → Tasks 4-5. `echoSafe` removal → Task 6 Step 3. UI redesign → Task 7.
- **Tasks 5, 6, and 7 have no automated tests, by design.** This is deliberate and plan-mandated, not an oversight: `app.js` is browser-API glue (`MediaRecorder`, `AudioContext`, `fetch`, DOM) and `index.html` is markup — neither can run under `node:test`. The codebase's existing split reflects exactly this: only `vad.js` and `turn-state.js` have JS tests, and both carry file-header comments explaining that their purity is what makes them testable. Introducing a DOM-mocking harness to test these three tasks would be new infrastructure this plan does not call for. Each of these tasks instead ends with a concrete manual verification script.
- **Backend for THINKING barge-in:** no backend task exists for this on purpose — `test_heard_chars_zero_drops_the_previous_reply` (already in `tests/test_voice_server_interrupt.py`) already proves the server-side half of the mechanism Task 4/5 depend on; adding a redundant test would duplicate coverage rather than add any.
- **Type/name consistency checked:** `"DISCARD_PENDING_REPLY"` (Task 4 producer, Task 5 consumer), `create_app(..., tts=None)` (Task 2 producer, Task 3 consumer), `_VOICE_IDS`/`_voice_repo_path` (Task 1 producer, Task 3 consumer) all match across tasks.
