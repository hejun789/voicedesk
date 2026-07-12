# VoiceDesk Phase 3 — Voice Layer — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a caller speak into a browser, have the existing agent hear them, take real actions against the calendar, and speak its reply back — measuring per-turn latency.

**Architecture:** Push-to-talk, one HTTP POST per turn. The browser records audio and POSTs it; the server transcribes with Groq Whisper, calls the **unmodified** `Agent.respond(text) -> str`, and returns JSON; the browser speaks the reply with the Web Speech API. STT sits behind a protocol so the whole test suite stays offline (a `FakeSTT`, exactly like the existing `FakeLLM`). No WebSockets, no streaming.

**Tech Stack:** Python 3.11+, FastAPI, uvicorn, python-multipart, existing `groq` SDK (Whisper), stdlib, pytest + FastAPI `TestClient`. Browser: `MediaRecorder` + `speechSynthesis` (no frontend framework, no build step).

## Global Constraints

- **Cost $0.** Groq free tier only. No paid services, no Twilio.
- **The agent core is NOT modified.** `agent.py`, `tools.py`, `db.py`, `faq.py`, `registry.py`, `llm.py`, `groq_client.py`, and `evals/` are untouched by this phase. The voice layer wraps `Agent.respond(text) -> str`.
- **The test suite stays fully offline.** No test may require a network call, an API key, or a microphone. STT is injected; use `FakeSTT` + the existing `FakeLLM`.
- New runtime dependencies are limited to exactly: `fastapi`, `uvicorn`, `python-multipart`.
- Tests run as `PYTHONPATH=src python -m pytest ...` from the repo root (Bash/Git Bash). In PowerShell: `$env:PYTHONPATH="src"; python -m pytest`. If plain `python` lacks deps, use `./.venv/Scripts/python.exe`.
- Existing suite is **154 passing** before this phase. It must stay green.
- TDD throughout; commit after each green task.

---

### Task 1: Dependencies and the speech-to-text module

**Files:**
- Modify: `requirements.txt`
- Create: `src/voicedesk/voice/__init__.py` (empty)
- Create: `src/voicedesk/voice/stt.py`
- Test: `tests/test_voice_stt.py`

**Interfaces:**
- Produces:
  - `STTError(Exception)` — raised when transcription fails.
  - `STTClient` Protocol with `transcribe(audio: bytes, filename: str = "audio.webm") -> str`.
  - `FakeSTT(scripted: list[str])` — test double; pops one scripted transcript per `transcribe` call.
  - `GroqWhisper(model: str | None = None, api_key: str | None = None, client=None)` implementing `transcribe`. Uses Groq's audio transcription API. Imports `groq` lazily (so tests need neither the package nor a key), mirroring `GroqLLM`. Model defaults to `GROQ_STT_MODEL` env var, else `DEFAULT_STT_MODEL = "whisper-large-v3-turbo"`. Returns the transcript **stripped** of surrounding whitespace. Wraps any API exception in `STTError`.

- [ ] **Step 1: Add the dependencies** — `requirements.txt` becomes exactly:

```
groq==0.11.0
httpx<0.28          # groq 0.11.0 passes proxies= to httpx, removed in httpx 0.28
python-dotenv==1.0.1
pytest==8.3.2
fastapi==0.115.0
uvicorn==0.30.6
python-multipart==0.0.9
```

- [ ] **Step 2: Install them**

Run: `./.venv/Scripts/python.exe -m pip install -r requirements.txt`
Expected: fastapi, uvicorn, python-multipart installed; no errors. (If PyPI is flaky, retry with `-i https://mirrors.aliyun.com/pypi/simple`.)

- [ ] **Step 3: Create the package** — `src/voicedesk/voice/__init__.py` as an empty file.

- [ ] **Step 4: Write the failing test** — `tests/test_voice_stt.py`

```python
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
```

- [ ] **Step 5: Run test to verify it fails**

Run: `PYTHONPATH=src ./.venv/Scripts/python.exe -m pytest tests/test_voice_stt.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'voicedesk.voice'`

- [ ] **Step 6: Implement** — `src/voicedesk/voice/stt.py`

```python
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
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `PYTHONPATH=src ./.venv/Scripts/python.exe -m pytest tests/test_voice_stt.py -v`
Expected: PASS (5 passed)

- [ ] **Step 8: Commit**

```bash
git add requirements.txt src/voicedesk/voice tests/test_voice_stt.py
git commit -m "feat: Groq Whisper speech-to-text behind a swappable protocol"
```

---

### Task 2: Session store — one Agent per caller

**Files:**
- Create: `src/voicedesk/voice/session.py`
- Test: `tests/test_voice_session.py`

**Interfaces:**
- Consumes: `Agent(conn, llm)` from `voicedesk.agent` (only as the thing the factory returns — this module does not import it).
- Produces:
  - `DEFAULT_TTL_S = 1800` (30 minutes).
  - `SessionStore(agent_factory, ttl_s: float = DEFAULT_TTL_S, clock=time.monotonic)` where `agent_factory` is a zero-arg callable returning a fresh `Agent`.
  - `get_or_create(session_id: str) -> Agent` — returns the caller's existing Agent (so conversation history accumulates across turns) or creates one. Touching a session refreshes its last-used time.
  - `__len__()` — number of live sessions.
  - Idle sessions older than `ttl_s` are dropped on each `get_or_create` call, so the map cannot grow without bound.
  - The injectable `clock` exists so expiry is testable without sleeping.

- [ ] **Step 1: Write the failing test** — `tests/test_voice_session.py`

```python
from voicedesk.voice.session import SessionStore, DEFAULT_TTL_S


class _FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def _counter_factory():
    """Each call returns a distinct object, so we can prove identity/isolation."""
    made = []

    def factory():
        agent = object()
        made.append(agent)
        return agent

    return factory, made


def test_same_session_id_returns_the_same_agent():
    factory, made = _counter_factory()
    store = SessionStore(factory)
    a = store.get_or_create("s1")
    b = store.get_or_create("s1")
    assert a is b            # history must persist across turns
    assert len(made) == 1    # only one Agent was ever built


def test_different_session_ids_are_isolated():
    factory, made = _counter_factory()
    store = SessionStore(factory)
    assert store.get_or_create("s1") is not store.get_or_create("s2")
    assert len(made) == 2
    assert len(store) == 2


def test_idle_session_expires():
    factory, made = _counter_factory()
    clock = _FakeClock()
    store = SessionStore(factory, ttl_s=100, clock=clock)
    first = store.get_or_create("s1")
    clock.advance(101)
    second = store.get_or_create("s1")
    assert second is not first   # the old one expired, a new Agent was built
    assert len(made) == 2
    assert len(store) == 1       # the stale entry was dropped


def test_active_session_does_not_expire():
    factory, made = _counter_factory()
    clock = _FakeClock()
    store = SessionStore(factory, ttl_s=100, clock=clock)
    first = store.get_or_create("s1")
    clock.advance(60)
    again = store.get_or_create("s1")   # touching it refreshes last-used
    clock.advance(60)                    # 120s total, but only 60s idle
    assert store.get_or_create("s1") is first
    assert again is first
    assert len(made) == 1


def test_default_ttl_is_thirty_minutes():
    assert DEFAULT_TTL_S == 1800
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src ./.venv/Scripts/python.exe -m pytest tests/test_voice_session.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'voicedesk.voice.session'`

- [ ] **Step 3: Implement** — `src/voicedesk/voice/session.py`

```python
import time

DEFAULT_TTL_S = 1800  # 30 minutes


class SessionStore:
    """Maps a caller's session id to their Agent, so conversation history
    accumulates across turns (a caller can say "book me Monday" on one turn and
    give their name on the next). Idle sessions expire so the map cannot grow
    without bound.

    `agent_factory` is a zero-arg callable returning a fresh Agent. `clock` is
    injectable so expiry can be tested without sleeping.
    """

    def __init__(self, agent_factory, ttl_s: float = DEFAULT_TTL_S,
                 clock=time.monotonic):
        self._agent_factory = agent_factory
        self._ttl_s = ttl_s
        self._clock = clock
        self._sessions: dict[str, tuple] = {}  # id -> (agent, last_used_at)

    def get_or_create(self, session_id: str):
        self._expire()
        now = self._clock()
        entry = self._sessions.get(session_id)
        if entry is None:
            agent = self._agent_factory()
        else:
            agent = entry[0]
        self._sessions[session_id] = (agent, now)
        return agent

    def _expire(self) -> None:
        now = self._clock()
        stale = [
            sid for sid, (_, last_used) in self._sessions.items()
            if now - last_used > self._ttl_s
        ]
        for sid in stale:
            del self._sessions[sid]

    def __len__(self) -> int:
        return len(self._sessions)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src ./.venv/Scripts/python.exe -m pytest tests/test_voice_session.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add src/voicedesk/voice/session.py tests/test_voice_session.py
git commit -m "feat: per-caller session store with idle expiry"
```

---

### Task 3: The server — one HTTP POST per voice turn

**Files:**
- Create: `src/voicedesk/voice/server.py`
- Test: `tests/test_voice_server.py`

**Interfaces:**
- Consumes: `STTError` from `voicedesk.voice.stt`; a `SessionStore` (Task 2) exposing `get_or_create(session_id) -> Agent`; an `Agent` exposing `respond(text) -> str`.
- Produces:
  - `STATIC_DIR: Path` — `src/voicedesk/voice/static` (created in Task 4).
  - `DIDNT_CATCH: str` — the spoken reply when the transcript is empty.
  - `STT_FAILED: str` — the spoken reply when transcription errors.
  - `create_app(stt, sessions) -> FastAPI` — dependency-injected app factory. Routes:
    - `GET /` → serves `static/index.html`.
    - `POST /turn` (multipart: `session_id` form field, `audio` file) → JSON
      `{"transcript": str, "reply": str, "timings": {"stt_ms": int, "agent_ms": int, "total_ms": int}}`,
      plus an `"error"` key only when STT failed. Always HTTP 200 — a demo must never show a stack trace mid-call.
  - Empty/whitespace transcript → returns `DIDNT_CATCH` and **does not call the agent** (saves quota; keeps noise out of the conversation history).

- [ ] **Step 1: Write the failing test** — `tests/test_voice_server.py`

```python
import sqlite3
import pytest
from fastapi.testclient import TestClient

from voicedesk.db import init_db
from voicedesk.agent import Agent
from voicedesk.llm import FakeLLM, Message, ToolCall
from voicedesk.tools import lookup_appt
from voicedesk.voice.stt import FakeSTT, STTError
from voicedesk.voice.session import SessionStore
from voicedesk.voice.server import create_app, DIDNT_CATCH, STT_FAILED


class _RaisingSTT:
    def transcribe(self, audio, filename="audio.webm"):
        raise STTError("429 rate limit")


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:", check_same_thread=False)
    c.row_factory = sqlite3.Row
    init_db(c)
    yield c
    c.close()


def _client(conn, stt, llm):
    sessions = SessionStore(lambda: Agent(conn, llm))
    return TestClient(create_app(stt, sessions))


def _post(client, audio=b"fakeaudio", session_id="s1"):
    return client.post(
        "/turn",
        data={"session_id": session_id},
        files={"audio": ("turn.webm", audio, "audio/webm")},
    )


def test_voice_turn_books_a_real_appointment(conn):
    stt = FakeSTT(["Book me Monday July 13th 2026 at 9am, Jane Doe, 5551234, cleaning"])
    llm = FakeLLM([
        Message(content=None, tool_calls=[
            ToolCall(id="1", name="book", arguments={
                "patient_name": "Jane Doe", "phone": "5551234",
                "slot_iso": "2026-07-13T09:00", "reason": "cleaning"})]),
        Message(content="You're booked for Monday at 9am.", tool_calls=[]),
    ])
    r = _post(_client(conn, stt, llm))
    assert r.status_code == 200
    body = r.json()
    assert "Jane Doe" in body["transcript"]
    assert "booked" in body["reply"].lower()
    # the side effect really happened:
    assert lookup_appt(conn, phone="5551234")[0]["slot_iso"] == "2026-07-13T09:00"


def test_response_carries_latency_breakdown(conn):
    stt = FakeSTT(["what are your hours"])
    llm = FakeLLM([Message(content="Weekdays 9 to 5.", tool_calls=[])])
    body = _post(_client(conn, stt, llm)).json()
    t = body["timings"]
    assert set(t) == {"stt_ms", "agent_ms", "total_ms"}
    assert all(isinstance(v, int) and v >= 0 for v in t.values())


def test_empty_transcript_does_not_call_the_agent(conn):
    stt = FakeSTT([""])
    llm = FakeLLM([])  # any agent call would IndexError -> proves it wasn't called
    body = _post(_client(conn, stt, llm)).json()
    assert body["reply"] == DIDNT_CATCH
    assert body["transcript"] == ""


def test_whitespace_transcript_does_not_call_the_agent(conn):
    stt = FakeSTT(["   "])
    llm = FakeLLM([])
    body = _post(_client(conn, stt, llm)).json()
    assert body["reply"] == DIDNT_CATCH


def test_stt_failure_degrades_gracefully(conn):
    llm = FakeLLM([])
    r = _post(_client(conn, _RaisingSTT(), llm))
    assert r.status_code == 200          # never a stack trace mid-call
    body = r.json()
    assert body["reply"] == STT_FAILED
    assert "429" in body["error"]


def test_same_session_id_keeps_conversation_history(conn):
    stt = FakeSTT(["I'd like an appointment", "Jane Doe, 5551234"])
    llm = FakeLLM([
        Message(content="Sure — what's your name and number?", tool_calls=[]),
        Message(content="Thanks Jane, what day suits you?", tool_calls=[]),
    ])
    client = _client(conn, stt, llm)
    first = _post(client, session_id="s1").json()
    second = _post(client, session_id="s1").json()
    assert "name" in first["reply"]
    assert "Jane" in second["reply"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src ./.venv/Scripts/python.exe -m pytest tests/test_voice_server.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'voicedesk.voice.server'`

- [ ] **Step 3: Implement** — `src/voicedesk/voice/server.py`

```python
import time
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse

from voicedesk.voice.stt import STTError

STATIC_DIR = Path(__file__).parent / "static"

DIDNT_CATCH = "Sorry, I didn't catch that. Could you say that again?"
STT_FAILED = (
    "Sorry, I'm having trouble hearing you. "
    "Let me have a team member call you back."
)


def _ms_since(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)


def create_app(stt, sessions) -> FastAPI:
    """`stt` implements STTClient; `sessions` is a SessionStore. Both are
    injected so the whole app can be tested with no network and no microphone."""
    app = FastAPI(title="VoiceDesk")

    @app.get("/")
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    @app.post("/turn")
    async def turn(
        session_id: str = Form(...),
        audio: UploadFile = File(...),
    ):
        started = time.perf_counter()
        data = await audio.read()

        stt_started = time.perf_counter()
        try:
            transcript = stt.transcribe(data, audio.filename or "audio.webm")
        except STTError as e:
            # Never crash the call — speak an apology and report the error.
            return {
                "transcript": "",
                "reply": STT_FAILED,
                "timings": {"stt_ms": _ms_since(stt_started), "agent_ms": 0,
                            "total_ms": _ms_since(started)},
                "error": str(e),
            }
        stt_ms = _ms_since(stt_started)

        if not transcript.strip():
            # Don't spend an LLM call, and don't pollute the history with noise.
            return {
                "transcript": "",
                "reply": DIDNT_CATCH,
                "timings": {"stt_ms": stt_ms, "agent_ms": 0,
                            "total_ms": _ms_since(started)},
            }

        agent_started = time.perf_counter()
        agent = sessions.get_or_create(session_id)
        reply = agent.respond(transcript)
        agent_ms = _ms_since(agent_started)

        return {
            "transcript": transcript,
            "reply": reply,
            "timings": {"stt_ms": stt_ms, "agent_ms": agent_ms,
                        "total_ms": _ms_since(started)},
        }

    return app
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src ./.venv/Scripts/python.exe -m pytest tests/test_voice_server.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Run the FULL suite (nothing may regress)**

Run: `PYTHONPATH=src ./.venv/Scripts/python.exe -m pytest -q`
Expected: PASS — 154 pre-existing + the new voice tests, all green, no network.

- [ ] **Step 6: Commit**

```bash
git add src/voicedesk/voice/server.py tests/test_voice_server.py
git commit -m "feat: POST /turn - transcribe, run the agent, return reply + timings"
```

---

### Task 4: The browser page and the run entrypoint

**Files:**
- Create: `src/voicedesk/voice/static/index.html`
- Create: `src/voicedesk/voice/static/app.js`
- Create: `src/voicedesk/voice/__main__.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: `create_app(stt, sessions)`, `STATIC_DIR` from `voicedesk.voice.server`; `GroqWhisper` from `voicedesk.voice.stt`; `SessionStore` from `voicedesk.voice.session`; `Agent` from `voicedesk.agent`; `GroqLLM` from `voicedesk.groq_client`; `init_db` from `voicedesk.db`.
- Produces: `python -m voicedesk.voice` starts a uvicorn server on `http://127.0.0.1:8000` serving the page and the `/turn` endpoint against the real Groq Whisper + Groq LLM and the `voicedesk.db` SQLite file.

- [ ] **Step 1: Write `src/voicedesk/voice/static/index.html`**

```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>VoiceDesk — BrightSmile Dental</title>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 640px; margin: 3rem auto;
           padding: 0 1rem; color: #1a1a1a; }
    h1 { font-size: 1.4rem; }
    #talk { width: 100%; padding: 1.5rem; font-size: 1.1rem; border-radius: 12px;
            border: 2px solid #2563eb; background: #eff6ff; color: #1e40af;
            cursor: pointer; user-select: none; }
    #talk.recording { background: #dc2626; border-color: #dc2626; color: #fff; }
    #talk:disabled { opacity: .5; cursor: default; }
    .turn { margin-top: 1.5rem; padding: 1rem; border-radius: 10px; background: #f5f5f5; }
    .label { font-size: .75rem; text-transform: uppercase; letter-spacing: .05em;
             color: #666; }
    .you { font-weight: 600; }
    #timings { margin-top: 1rem; font-family: ui-monospace, monospace; font-size: .8rem;
               color: #555; }
    .hint { color: #666; font-size: .9rem; }
  </style>
</head>
<body>
  <h1>VoiceDesk — BrightSmile Dental</h1>
  <p class="hint">Hold the button, speak, then release. Try:
    <em>"Book me Monday July 13th 2026 at 9am, Jane Doe, 5551234, for a cleaning."</em></p>

  <button id="talk">Hold to talk</button>

  <div class="turn">
    <div class="label">You said</div>
    <div id="transcript" class="you">—</div>
    <div class="label" style="margin-top:.75rem">Receptionist</div>
    <div id="reply">—</div>
    <div id="timings"></div>
  </div>

  <script src="/static/app.js"></script>
</body>
</html>
```

- [ ] **Step 2: Write `src/voicedesk/voice/static/app.js`**

```javascript
const talk = document.getElementById("talk");
const transcriptEl = document.getElementById("transcript");
const replyEl = document.getElementById("reply");
const timingsEl = document.getElementById("timings");

// One session per page load, so the agent remembers this caller across turns.
const sessionId = crypto.randomUUID();

let recorder = null;
let chunks = [];

async function startRecording() {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  recorder = new MediaRecorder(stream);
  chunks = [];
  recorder.ondataavailable = (e) => chunks.push(e.data);
  recorder.onstop = () => {
    stream.getTracks().forEach((t) => t.stop());
    send(new Blob(chunks, { type: "audio/webm" }));
  };
  recorder.start();
  talk.classList.add("recording");
  talk.textContent = "Listening… release to send";
}

function stopRecording() {
  if (recorder && recorder.state === "recording") recorder.stop();
  talk.classList.remove("recording");
  talk.textContent = "Hold to talk";
}

async function send(blob) {
  talk.disabled = true;
  transcriptEl.textContent = "…";
  replyEl.textContent = "";
  timingsEl.textContent = "";

  const form = new FormData();
  form.append("session_id", sessionId);
  form.append("audio", blob, "turn.webm");

  try {
    const res = await fetch("/turn", { method: "POST", body: form });
    const data = await res.json();
    transcriptEl.textContent = data.transcript || "(didn't catch that)";
    replyEl.textContent = data.reply;
    const t = data.timings;
    timingsEl.textContent =
      `stt ${t.stt_ms}ms · agent ${t.agent_ms}ms · total ${t.total_ms}ms`;
    speak(data.reply);
  } catch (err) {
    replyEl.textContent = "Something went wrong. Please try again.";
  } finally {
    talk.disabled = false;
  }
}

function speak(text) {
  // Browser TTS: starts instantly, costs nothing, adds no network latency.
  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.rate = 1.05;
  window.speechSynthesis.speak(utterance);
}

talk.addEventListener("mousedown", startRecording);
talk.addEventListener("mouseup", stopRecording);
talk.addEventListener("mouseleave", stopRecording);
talk.addEventListener("touchstart", (e) => { e.preventDefault(); startRecording(); });
talk.addEventListener("touchend", (e) => { e.preventDefault(); stopRecording(); });
```

- [ ] **Step 3: Mount the static directory** — the page loads `/static/app.js`, so add the mount to `src/voicedesk/voice/server.py`. Add this import at the top of the file, alongside the existing imports:

```python
from fastapi.staticfiles import StaticFiles
```

and add this line inside `create_app`, immediately before the final `return app`:

```python
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
```

- [ ] **Step 4: Write the entrypoint** — `src/voicedesk/voice/__main__.py`

```python
import sqlite3

import uvicorn
from dotenv import load_dotenv

from voicedesk.agent import Agent
from voicedesk.db import init_db
from voicedesk.groq_client import GroqLLM
from voicedesk.voice.server import create_app
from voicedesk.voice.session import SessionStore
from voicedesk.voice.stt import GroqWhisper


def main() -> None:
    load_dotenv()
    # check_same_thread=False: FastAPI serves requests on a worker thread.
    conn = sqlite3.connect("voicedesk.db", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    init_db(conn)

    sessions = SessionStore(lambda: Agent(conn, GroqLLM()))
    app = create_app(GroqWhisper(), sessions)

    print("VoiceDesk is listening on http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Verify the app builds and serves the page offline**

Add this test to `tests/test_voice_server.py` (it needs no network — it uses the fakes):

```python
def test_index_page_is_served(conn):
    client = _client(conn, FakeSTT([]), FakeLLM([]))
    r = client.get("/")
    assert r.status_code == 200
    assert "Hold to talk" in r.text


def test_static_app_js_is_served(conn):
    client = _client(conn, FakeSTT([]), FakeLLM([]))
    r = client.get("/static/app.js")
    assert r.status_code == 200
    assert "speechSynthesis" in r.text
```

Run: `PYTHONPATH=src ./.venv/Scripts/python.exe -m pytest tests/test_voice_server.py -v`
Expected: PASS (8 passed)

- [ ] **Step 6: Run the FULL suite**

Run: `PYTHONPATH=src ./.venv/Scripts/python.exe -m pytest -q`
Expected: PASS — everything green, fully offline.

- [ ] **Step 7: Confirm the entrypoint imports cleanly** (does not start the server)

Run: `PYTHONPATH=src ./.venv/Scripts/python.exe -c "import voicedesk.voice.__main__; print('ok')"`
Expected: prints `ok` (no API key needed — `GroqWhisper`/`GroqLLM` import groq lazily inside `main()`).

- [ ] **Step 8: Update `README.md`** — insert this section immediately after the "Talk to the agent" block in the "Run it" section:

```markdown
### Talk to it by voice (Phase 3)
```powershell
$env:PYTHONPATH = "src"; python -m voicedesk.voice
```
Open <http://127.0.0.1:8000>, hold the button, and speak. The browser records your
voice, Groq Whisper transcribes it, the same agent takes the action, and the browser
speaks the reply back. Each turn shows its latency breakdown (stt / agent / total).

Use Chrome or Edge — it needs `MediaRecorder` and the Web Speech API.
```

- [ ] **Step 9: Commit**

```bash
git add src/voicedesk/voice tests/test_voice_server.py README.md
git commit -m "feat: browser voice UI (push-to-talk + speech synthesis) and run entrypoint"
```

- [ ] **Step 10: Manual smoke test (human, needs a Groq key and a microphone)**

Run: `$env:PYTHONPATH = "src"; python -m voicedesk.voice`
Open <http://127.0.0.1:8000> in Chrome, allow microphone access, hold the button and say:
"Book me Monday July 13th 2026 at 9am, Jane Doe, 5551234, for a cleaning."
Expected: the transcript appears, the agent speaks a confirmation, a latency breakdown is
shown, and `sqlite3 voicedesk.db "SELECT * FROM appointments;"` shows the booking.

---

## Phase 3 Definition of Done

- `python -m voicedesk.voice` serves a page where holding a button and speaking books a real appointment end to end, and the agent speaks its reply.
- Multi-turn works: details supplied across several turns complete a booking (same `session_id`).
- Every response carries a per-turn latency breakdown (`stt_ms`, `agent_ms`, `total_ms`), displayed live.
- The full test suite still runs **fully offline** (no API key, no network, no microphone) and passes.
- The agent core (`agent.py`, `tools.py`, `db.py`, `faq.py`, `registry.py`, `llm.py`, `groq_client.py`) and `evals/` are unchanged.

## What comes next (separate plan)

- **Phase 4 — deploy + polish:** host the demo, record the per-turn latency numbers in the README, and report cost per resolution.
