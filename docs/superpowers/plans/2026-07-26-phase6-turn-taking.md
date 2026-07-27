# Phase 6 — Conversational Turn-Taking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace push-to-talk with hands-free, interruptible conversation — the agent detects when the caller starts and stops speaking, and stops talking the moment the caller cuts in.

**Architecture:** Two new pure ES modules (`vad.js` for energy-based voice activity detection, `turn-state.js` for the conversation state machine) hold all decision logic and are unit-tested under Node with zero npm packages. `app.js` becomes a thin impure shell wiring them to `getUserMedia`, `MediaRecorder`, `fetch`, and `speechSynthesis`. A small server change lets the browser report how much of a reply the caller actually heard before interrupting, so `agent.messages` reflects reality.

**Tech Stack:** Vanilla ES modules + Web Audio API (browser), `node --test` (JS tests, no npm), FastAPI + pytest (server), GitHub Actions (CI).

## Global Constraints

- **Python must run via the project venv:** `./.venv/Scripts/python.exe`. The system Python lacks `fastapi` and will fail collection. Always `PYTHONPATH=src ./.venv/Scripts/python.exe -m pytest`.
- **Node is at `C:\Program Files\nodejs\node.exe` (v24.18.0).** Use `node` if it resolves on PATH; otherwise use the full path `"/c/Program Files/nodejs/node.exe"`.
- **Zero new dependencies.** No npm packages, no `package.json` required, no new Python packages. `node --test` is built in.
- **Git author is single-author.** Never add `Co-Authored-By` trailers to this repo.
- **Work on branch `phase6-turn-taking`**, not `main`.
- **All JS modules use ES module syntax** (`export` / `import`). `index.html` must load `app.js` with `type="module"`.
- **Barge-in is permitted only from the SPEAKING state**, never from THINKING (see spec decision 3).
- **Assistant messages carrying `tool_calls` are never truncated** — only the final plain-text spoken reply.
- Spec: `docs/superpowers/specs/2026-07-26-voicedesk-phase6-turn-taking-design.md`

---

## File Structure

| File | Responsibility |
|---|---|
| `src/voicedesk/agent.py` | *(modify)* gains `Agent.truncate_last_reply()` |
| `src/voicedesk/voice/server.py` | *(modify)* `/turn` accepts optional `heard_chars`, applies truncation before responding |
| `src/voicedesk/voice/static/vad.js` | *(create)* pure energy VAD — numbers in, speech events out |
| `src/voicedesk/voice/static/turn-state.js` | *(create)* pure conversation state machine |
| `src/voicedesk/voice/static/app.js` | *(rewrite)* impure browser shell; no decision logic |
| `src/voicedesk/voice/static/index.html` | *(modify)* `type="module"`, mode toggle, updated hint copy |
| `tests/test_agent_truncate.py` | *(create)* truncation unit tests |
| `tests/test_voice_server_interrupt.py` | *(create)* `/turn` + `heard_chars` integration tests |
| `tests/js/vad.test.mjs` | *(create)* VAD unit tests |
| `tests/js/turn-state.test.mjs` | *(create)* state machine unit tests |
| `.github/workflows/tests.yml` | *(modify)* add Node test step |
| `README.md` | *(modify)* describe hands-free + barge-in |

---

### Task 1: `Agent.truncate_last_reply`

Server-side history fidelity: when a caller interrupts, the stored assistant reply must shrink to what was actually spoken.

**Files:**
- Modify: `src/voicedesk/agent.py` (add method to `Agent`, after `respond`)
- Test: `tests/test_agent_truncate.py` (create)

**Interfaces:**
- Consumes: existing `Agent` class, `db` pytest fixture, `FakeLLM`, `Message` from `voicedesk.llm`
- Produces: `Agent.truncate_last_reply(heard_chars: int) -> None` — mutates `self.messages` in place, returns nothing. Used by Task 2.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_agent_truncate.py`:

```python
from voicedesk.agent import Agent
from voicedesk.llm import FakeLLM, Message


def _agent_with_reply(db, reply: str) -> Agent:
    agent = Agent(db, FakeLLM([Message(content=reply, tool_calls=[])]))
    agent.respond("hi")
    return agent


def test_truncates_reply_to_what_was_heard(db):
    agent = _agent_with_reply(db, "Booked for Monday at nine. Anything else?")
    agent.truncate_last_reply(26)   # through "…at nine."
    assert agent.messages[-1]["content"] == "Booked for Monday at nine. [interrupted]"


def test_marker_is_appended_when_cut_off(db):
    agent = _agent_with_reply(db, "Booked for Monday at nine. Anything else?")
    agent.truncate_last_reply(10)
    content = agent.messages[-1]["content"]
    assert content.endswith("[interrupted]")
    assert content.startswith("Booked for")
    assert "Anything else" not in content


def test_hearing_the_whole_reply_changes_nothing(db):
    full = "Booked for Monday."
    agent = _agent_with_reply(db, full)
    agent.truncate_last_reply(len(full))
    assert agent.messages[-1]["content"] == full


def test_heard_chars_beyond_length_is_clamped_and_changes_nothing(db):
    full = "Booked for Monday."
    agent = _agent_with_reply(db, full)
    agent.truncate_last_reply(9999)
    assert agent.messages[-1]["content"] == full


def test_hearing_nothing_removes_the_message_entirely(db):
    agent = _agent_with_reply(db, "Booked for Monday.")
    before = len(agent.messages)
    agent.truncate_last_reply(0)
    assert len(agent.messages) == before - 1
    assert agent.messages[-1]["role"] != "assistant"


def test_negative_heard_chars_clamps_to_zero(db):
    agent = _agent_with_reply(db, "Booked for Monday.")
    before = len(agent.messages)
    agent.truncate_last_reply(-5)
    assert len(agent.messages) == before - 1


def test_assistant_message_with_tool_calls_is_never_truncated(db):
    # A tool-call message is internal bookkeeping, never spoken aloud, so an
    # interruption must not corrupt it.
    agent = Agent(db, FakeLLM([]))
    agent.messages.append({
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": "1", "type": "function",
                        "function": {"name": "book", "arguments": "{}"}}],
    })
    before = list(agent.messages)
    agent.truncate_last_reply(0)
    assert agent.messages == before


def test_no_op_when_last_message_is_not_from_the_assistant(db):
    agent = Agent(db, FakeLLM([]))
    agent.messages.append({"role": "user", "content": "hello"})
    before = list(agent.messages)
    agent.truncate_last_reply(2)
    assert agent.messages == before
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONPATH=src ./.venv/Scripts/python.exe -m pytest tests/test_agent_truncate.py -v`
Expected: FAIL with `AttributeError: 'Agent' object has no attribute 'truncate_last_reply'`

- [ ] **Step 3: Write the minimal implementation**

In `src/voicedesk/agent.py`, add this method to `Agent`, directly after `respond`:

```python
    def truncate_last_reply(self, heard_chars: int) -> None:
        """Shrink the last spoken reply to the part the caller actually heard.

        When a caller barges in mid-sentence, the full reply is still sitting in
        history — so the agent believes it said things that were never spoken and
        may reference them later. The browser reports how far text-to-speech got;
        this trims history to match reality and marks it, so the model knows it was
        cut off and does not simply repeat itself.

        Only the final plain-text reply is eligible: messages carrying tool_calls
        are internal bookkeeping and are never spoken aloud.
        """
        if not self.messages:
            return
        msg = self.messages[-1]
        if msg.get("role") != "assistant" or msg.get("tool_calls"):
            return
        content = msg.get("content") or ""
        n = max(0, min(heard_chars, len(content)))
        if n >= len(content):
            return  # heard all of it — nothing was cut off
        heard = content[:n].rstrip()
        if not heard:
            del self.messages[-1]
        else:
            msg["content"] = heard + " [interrupted]"
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PYTHONPATH=src ./.venv/Scripts/python.exe -m pytest tests/test_agent_truncate.py -v`
Expected: PASS, 8 tests

Then run the full suite to confirm nothing regressed:
Run: `PYTHONPATH=src ./.venv/Scripts/python.exe -m pytest`
Expected: PASS, 287 tests (279 existing + 8 new)

- [ ] **Step 5: Commit**

```bash
git add src/voicedesk/agent.py tests/test_agent_truncate.py
git commit -m "feat: truncate interrupted replies in agent history"
```

---

### Task 2: `/turn` accepts `heard_chars`

Wire the truncation into the HTTP layer so the browser can report an interruption on its next turn.

**Files:**
- Modify: `src/voicedesk/voice/server.py:79-160` (the `turn` endpoint)
- Test: `tests/test_voice_server_interrupt.py` (create)

**Interfaces:**
- Consumes: `Agent.truncate_last_reply(heard_chars)` from Task 1; existing `create_app(stt, sessions, lock=None, limiter=None)`
- Produces: `/turn` accepts an optional `heard_chars` form field (integer). Task 5's `app.js` posts it.

- [ ] **Step 1: Write the failing tests**

First inspect `tests/test_voice_server.py` to copy its existing app-construction fixture style, then create `tests/test_voice_server_interrupt.py`:

```python
from fastapi.testclient import TestClient

from voicedesk.agent import Agent
from voicedesk.llm import FakeLLM, Message
from voicedesk.voice.server import create_app
from voicedesk.voice.session import SessionStore
from voicedesk.voice.stt import FakeSTT

AUDIO = b"x" * 2000  # over MIN_AUDIO_BYTES so it is not rejected as a stray tap


def _client(db, replies):
    """An app whose agent returns `replies` in order, sharing one SessionStore
    so a test can inspect conversation history after each turn."""
    sessions = SessionStore(
        lambda lang: Agent(db, FakeLLM([Message(content=r, tool_calls=[])
                                        for r in replies]))
    )
    app = create_app(FakeSTT(["first question", "second question"]), sessions)
    return TestClient(app), sessions


def _post(client, **extra):
    data = {"session_id": "s1", "lang": "en", **extra}
    return client.post("/turn", data=data,
                       files={"audio": ("turn.webm", AUDIO, "audio/webm")})


def test_heard_chars_truncates_the_previous_reply(db):
    client, sessions = _client(db, ["Booked for Monday at nine. Anything else?",
                                    "Sure."])
    _post(client)
    _post(client, heard_chars=26)
    agent = sessions.get_or_create("s1", "en")
    contents = [m.get("content") for m in agent.messages if m.get("role") == "assistant"]
    assert contents[0] == "Booked for Monday at nine. [interrupted]"


def test_omitting_heard_chars_leaves_history_untouched(db):
    # Regression guard: existing clients that never send the field must behave
    # exactly as they do today.
    full = "Booked for Monday at nine. Anything else?"
    client, sessions = _client(db, [full, "Sure."])
    _post(client)
    _post(client)
    agent = sessions.get_or_create("s1", "en")
    contents = [m.get("content") for m in agent.messages if m.get("role") == "assistant"]
    assert contents[0] == full


def test_heard_chars_zero_drops_the_previous_reply(db):
    client, sessions = _client(db, ["Booked for Monday.", "Sure."])
    _post(client)
    _post(client, heard_chars=0)
    agent = sessions.get_or_create("s1", "en")
    contents = [m.get("content") for m in agent.messages if m.get("role") == "assistant"]
    assert contents == ["Sure."]


def test_response_shape_is_unchanged_when_heard_chars_is_sent(db):
    client, _ = _client(db, ["Booked.", "Sure."])
    _post(client)
    res = _post(client, heard_chars=3)
    assert res.status_code == 200
    body = res.json()
    assert set(body) >= {"transcript", "reply", "timings", "lang"}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONPATH=src ./.venv/Scripts/python.exe -m pytest tests/test_voice_server_interrupt.py -v`
Expected: FAIL — `test_heard_chars_truncates_the_previous_reply` and `test_heard_chars_zero_drops_the_previous_reply` fail because the field is ignored, so the full reply is still in history.

- [ ] **Step 3: Write the minimal implementation**

In `src/voicedesk/voice/server.py`, add the parameter to the `turn` signature (after `lang`):

```python
        lang: str = Form(DEFAULT_LANG),
        heard_chars: int | None = Form(None),
```

Then change the `_run_agent` closure (currently at lines 142-145) to apply the truncation inside the lock, before responding:

```python
        def _run_agent() -> str:
            with lock:
                agent = sessions.get_or_create(session_id, lang)
                if heard_chars is not None:
                    # The caller cut the previous reply short; trim history to
                    # what they actually heard before adding this turn.
                    agent.truncate_last_reply(heard_chars)
                return agent.respond(transcript)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PYTHONPATH=src ./.venv/Scripts/python.exe -m pytest tests/test_voice_server_interrupt.py -v`
Expected: PASS, 4 tests

Run the full suite:
Run: `PYTHONPATH=src ./.venv/Scripts/python.exe -m pytest`
Expected: PASS, 291 tests

- [ ] **Step 5: Commit**

```bash
git add src/voicedesk/voice/server.py tests/test_voice_server_interrupt.py
git commit -m "feat: /turn accepts heard_chars to trim interrupted replies"
```

---

### Task 3: `vad.js` — energy voice activity detection

Pure decision function: loudness values in, speech boundary events out. No microphone, no DOM.

**Files:**
- Create: `src/voicedesk/voice/static/vad.js`
- Test: `tests/js/vad.test.mjs` (create)

**Interfaces:**
- Consumes: nothing
- Produces: `createEnergyVAD({ threshold, hangoverMs, minSpeechMs }) -> { process(level, nowMs) }` where `process` returns the string `"speech-start"`, `"speech-end"`, or `null`. Used by Task 5.

- [ ] **Step 1: Write the failing tests**

Create `tests/js/vad.test.mjs`:

```javascript
import { test } from "node:test";
import assert from "node:assert/strict";
import { createEnergyVAD } from "../../src/voicedesk/voice/static/vad.js";

const OPTS = { threshold: 0.02, hangoverMs: 800, minSpeechMs: 200 };
const LOUD = 0.5;
const QUIET = 0.001;

test("silence never fires an event", () => {
  const vad = createEnergyVAD(OPTS);
  for (let t = 0; t < 5000; t += 50) {
    assert.equal(vad.process(QUIET, t), null);
  }
});

test("sustained speech fires speech-start exactly once", () => {
  const vad = createEnergyVAD(OPTS);
  const events = [];
  for (let t = 0; t < 2000; t += 50) {
    const e = vad.process(LOUD, t);
    if (e) events.push(e);
  }
  assert.deepEqual(events, ["speech-start"]);
});

test("speech-start waits for minSpeechMs before firing", () => {
  const vad = createEnergyVAD(OPTS);
  assert.equal(vad.process(LOUD, 0), null);
  assert.equal(vad.process(LOUD, 100), null);   // still under 200ms
  assert.equal(vad.process(LOUD, 200), "speech-start");
});

test("a short blip below minSpeechMs never fires", () => {
  // A click, a cough, a door closing.
  const vad = createEnergyVAD(OPTS);
  assert.equal(vad.process(LOUD, 0), null);
  assert.equal(vad.process(LOUD, 100), null);
  for (let t = 150; t < 3000; t += 50) {
    assert.equal(vad.process(QUIET, t), null);
  }
});

test("a brief pause mid-sentence does NOT end the turn", () => {
  // The single most common way energy VAD gets this wrong: natural gaps
  // between words must not be mistaken for the end of a turn.
  const vad = createEnergyVAD(OPTS);
  assert.equal(vad.process(LOUD, 0), null);
  assert.equal(vad.process(LOUD, 200), "speech-start");
  for (let t = 250; t < 900; t += 50) {         // 600ms gap, under hangover
    assert.equal(vad.process(QUIET, t), null);
  }
  assert.equal(vad.process(LOUD, 950), null);   // speaking again, no event
  assert.equal(vad.process(LOUD, 1000), null);
});

test("silence for hangoverMs ends the turn exactly once", () => {
  const vad = createEnergyVAD(OPTS);
  vad.process(LOUD, 0);
  assert.equal(vad.process(LOUD, 200), "speech-start");
  let events = [];
  for (let t = 250; t < 2000; t += 50) {
    const e = vad.process(QUIET, t);
    if (e) events.push(e);
  }
  assert.deepEqual(events, ["speech-end"]);
});

test("a second utterance fires start and end again", () => {
  const vad = createEnergyVAD(OPTS);
  const events = [];
  const feed = (level, from, to) => {
    for (let t = from; t < to; t += 50) {
      const e = vad.process(level, t);
      if (e) events.push(e);
    }
  };
  feed(LOUD, 0, 500);
  feed(QUIET, 500, 1500);
  feed(LOUD, 1500, 2000);
  feed(QUIET, 2000, 3000);
  assert.deepEqual(events, ["speech-start", "speech-end", "speech-start", "speech-end"]);
});

test("defaults are supplied when no options are passed", () => {
  const vad = createEnergyVAD();
  assert.equal(typeof vad.process, "function");
  assert.equal(vad.process(0.0, 0), null);
});
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `node --test tests/js/vad.test.mjs`
Expected: FAIL — cannot find module `vad.js`

- [ ] **Step 3: Write the minimal implementation**

Create `src/voicedesk/voice/static/vad.js`:

```javascript
// Energy-based voice activity detection.
//
// Pure decision logic: it is handed a loudness value and a timestamp and
// answers whether a turn just started or just ended. It knows nothing about
// microphones or the DOM, which is what makes it unit-testable under Node.
//
// A better detector (Silero via onnxruntime-web is what production frameworks
// use) can replace this behind the same interface without touching app.js.

export function createEnergyVAD({
  threshold = 0.02,   // RMS above this counts as speech
  hangoverMs = 800,   // silence must persist this long to end a turn
  minSpeechMs = 200,  // sound must persist this long to start one
} = {}) {
  let speaking = false;
  let loudSince = null;
  let quietSince = null;

  return {
    process(level, nowMs) {
      const loud = level >= threshold;

      if (!speaking) {
        if (!loud) {
          loudSince = null;       // a dip cancels a nascent turn
          return null;
        }
        if (loudSince === null) loudSince = nowMs;
        if (nowMs - loudSince >= minSpeechMs) {
          speaking = true;
          loudSince = null;
          quietSince = null;
          return "speech-start";
        }
        return null;
      }

      // speaking
      if (loud) {
        quietSince = null;        // a pause between words is not the end
        return null;
      }
      if (quietSince === null) quietSince = nowMs;
      if (nowMs - quietSince >= hangoverMs) {
        speaking = false;
        quietSince = null;
        loudSince = null;
        return "speech-end";
      }
      return null;
    },
  };
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `node --test tests/js/vad.test.mjs`
Expected: PASS, 8 tests

- [ ] **Step 5: Commit**

```bash
git add src/voicedesk/voice/static/vad.js tests/js/vad.test.mjs
git commit -m "feat: energy-based voice activity detection"
```

---

### Task 4: `turn-state.js` — conversation state machine

Pure reducer describing what the call is doing and what should happen next.

**Files:**
- Create: `src/voicedesk/voice/static/turn-state.js`
- Test: `tests/js/turn-state.test.mjs` (create)

**Interfaces:**
- Consumes: nothing
- Produces:
  - Exported state name constants `IDLE`, `LISTENING`, `THINKING`, `SPEAKING` (string values equal to their names)
  - `initialState(mode = "hands-free") -> { name, capturing, mode }`
  - `next(state, event) -> { state, actions }` where `event` is one of `"ARM"`, `"DISARM"`, `"SPEECH_START"`, `"SPEECH_END"`, `"REPLY"`, `"TTS_END"`, `"TURN_ABORTED"`, `"PTT_DOWN"`, `"PTT_UP"`, and `actions` is an array drawn from `"START_RECORDING"`, `"STOP_AND_SEND"`, `"CANCEL_TTS"`, `"SPEAK"`.
  - Used by Task 5.

- [ ] **Step 1: Write the failing tests**

Create `tests/js/turn-state.test.mjs`:

```javascript
import { test } from "node:test";
import assert from "node:assert/strict";
import {
  IDLE, LISTENING, THINKING, SPEAKING, initialState, next,
} from "../../src/voicedesk/voice/static/turn-state.js";

test("a call starts idle and hands-free", () => {
  const s = initialState();
  assert.equal(s.name, IDLE);
  assert.equal(s.capturing, false);
  assert.equal(s.mode, "hands-free");
});

test("arming moves from idle to listening", () => {
  const { state, actions } = next(initialState(), "ARM");
  assert.equal(state.name, LISTENING);
  assert.deepEqual(actions, []);
});

test("speech while listening starts recording", () => {
  const armed = next(initialState(), "ARM").state;
  const { state, actions } = next(armed, "SPEECH_START");
  assert.equal(state.name, LISTENING);
  assert.equal(state.capturing, true);
  assert.deepEqual(actions, ["START_RECORDING"]);
});

test("end of speech sends the turn and moves to thinking", () => {
  let s = next(initialState(), "ARM").state;
  s = next(s, "SPEECH_START").state;
  const { state, actions } = next(s, "SPEECH_END");
  assert.equal(state.name, THINKING);
  assert.equal(state.capturing, false);
  assert.deepEqual(actions, ["STOP_AND_SEND"]);
});

test("end of speech without a capture in progress is ignored", () => {
  const armed = next(initialState(), "ARM").state;
  const { state, actions } = next(armed, "SPEECH_END");
  assert.equal(state.name, LISTENING);
  assert.deepEqual(actions, []);
});

test("a reply moves to speaking and speaks it", () => {
  let s = next(initialState(), "ARM").state;
  s = next(s, "SPEECH_START").state;
  s = next(s, "SPEECH_END").state;
  const { state, actions } = next(s, "REPLY");
  assert.equal(state.name, SPEAKING);
  assert.deepEqual(actions, ["SPEAK"]);
});

test("barge-in cancels speech and starts recording", () => {
  let s = next(initialState(), "ARM").state;
  s = next(s, "SPEECH_START").state;
  s = next(s, "SPEECH_END").state;
  s = next(s, "REPLY").state;
  const { state, actions } = next(s, "SPEECH_START");
  assert.equal(state.name, LISTENING);
  assert.equal(state.capturing, true);
  assert.deepEqual(actions, ["CANCEL_TTS", "START_RECORDING"]);
});

test("speech during thinking is ignored — no barge-in while the LLM runs", () => {
  let s = next(initialState(), "ARM").state;
  s = next(s, "SPEECH_START").state;
  s = next(s, "SPEECH_END").state;
  assert.equal(s.name, THINKING);
  const { state, actions } = next(s, "SPEECH_START");
  assert.equal(state.name, THINKING);
  assert.deepEqual(actions, []);
});

test("finishing speech returns to listening in hands-free mode", () => {
  let s = next(initialState("hands-free"), "ARM").state;
  s = next(s, "SPEECH_START").state;
  s = next(s, "SPEECH_END").state;
  s = next(s, "REPLY").state;
  const { state } = next(s, "TTS_END");
  assert.equal(state.name, LISTENING);
});

test("finishing speech returns to idle in push-to-talk mode", () => {
  let s = initialState("ptt");
  s = next(s, "PTT_DOWN").state;
  s = next(s, "PTT_UP").state;
  s = next(s, "REPLY").state;
  const { state } = next(s, "TTS_END");
  assert.equal(state.name, IDLE);
});

test("a late reply does not resurrect speaking after a barge-in", () => {
  // The user interrupted and is already talking again; the in-flight reply
  // that lands afterwards must not start speaking over them.
  let s = next(initialState(), "ARM").state;
  s = next(s, "SPEECH_START").state;
  s = next(s, "SPEECH_END").state;
  s = next(s, "REPLY").state;
  s = next(s, "SPEECH_START").state;   // barge-in
  assert.equal(s.name, LISTENING);
  const { state, actions } = next(s, "REPLY");
  assert.equal(state.name, LISTENING);
  assert.deepEqual(actions, []);
});

test("push-to-talk drives the same machine with button events", () => {
  let s = initialState("ptt");
  let r = next(s, "PTT_DOWN");
  assert.equal(r.state.capturing, true);
  assert.deepEqual(r.actions, ["START_RECORDING"]);
  r = next(r.state, "PTT_UP");
  assert.equal(r.state.name, THINKING);
  assert.deepEqual(r.actions, ["STOP_AND_SEND"]);
});

test("an aborted turn returns to listening instead of hanging in thinking", () => {
  // The upload was too small to transcribe, or the request failed. Without this
  // the machine would sit in THINKING forever and the call would be dead.
  let s = next(initialState(), "ARM").state;
  s = next(s, "SPEECH_START").state;
  s = next(s, "SPEECH_END").state;
  assert.equal(s.name, THINKING);
  const { state } = next(s, "TURN_ABORTED");
  assert.equal(state.name, LISTENING);
});

test("an aborted turn returns to idle in push-to-talk mode", () => {
  let s = initialState("ptt");
  s = next(s, "PTT_DOWN").state;
  s = next(s, "PTT_UP").state;
  const { state } = next(s, "TURN_ABORTED");
  assert.equal(state.name, IDLE);
});

test("disarming returns to idle and stops capturing", () => {
  let s = next(initialState(), "ARM").state;
  s = next(s, "SPEECH_START").state;
  const { state } = next(s, "DISARM");
  assert.equal(state.name, IDLE);
  assert.equal(state.capturing, false);
});

test("unknown events are ignored", () => {
  const s = next(initialState(), "ARM").state;
  const { state, actions } = next(s, "NONSENSE");
  assert.deepEqual(state, s);
  assert.deepEqual(actions, []);
});
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `node --test tests/js/turn-state.test.mjs`
Expected: FAIL — cannot find module `turn-state.js`

- [ ] **Step 3: Write the minimal implementation**

Create `src/voicedesk/voice/static/turn-state.js`:

```javascript
// The conversation state machine.
//
// A pure reducer: given the current state and an event, it returns the next
// state plus the side effects app.js should perform. It touches no browser API,
// so every transition — including the awkward ones like a reply arriving after
// the caller already interrupted — is unit-testable.
//
// Both input modes share this machine. Hands-free feeds it VAD events; push-to-
// talk feeds it button events. The transitions are otherwise identical.

export const IDLE = "IDLE";
export const LISTENING = "LISTENING";
export const THINKING = "THINKING";
export const SPEAKING = "SPEAKING";

export function initialState(mode = "hands-free") {
  return { name: IDLE, capturing: false, mode };
}

const stay = (state) => ({ state, actions: [] });

export function next(state, event) {
  const { name, capturing, mode } = state;

  if (event === "DISARM") {
    return { state: { ...state, name: IDLE, capturing: false }, actions: [] };
  }

  if (event === "ARM") {
    if (name !== IDLE) return stay(state);
    return { state: { ...state, name: LISTENING }, actions: [] };
  }

  if (event === "SPEECH_START" || event === "PTT_DOWN") {
    // Barge-in: the caller talks over the agent. Cut the audio immediately.
    if (name === SPEAKING) {
      return {
        state: { ...state, name: LISTENING, capturing: true },
        actions: ["CANCEL_TTS", "START_RECORDING"],
      };
    }
    // Deliberately NOT allowed while THINKING: the server cannot cancel an
    // in-flight agent call, so interrupting there would desync history.
    if (name === LISTENING || name === IDLE) {
      if (capturing) return stay(state);
      return {
        state: { ...state, name: LISTENING, capturing: true },
        actions: ["START_RECORDING"],
      };
    }
    return stay(state);
  }

  if (event === "SPEECH_END" || event === "PTT_UP") {
    if (!capturing) return stay(state);
    return {
      state: { ...state, name: THINKING, capturing: false },
      actions: ["STOP_AND_SEND"],
    };
  }

  if (event === "REPLY") {
    // A reply that lands after the caller already barged in must not speak
    // over them.
    if (name !== THINKING) return stay(state);
    return { state: { ...state, name: SPEAKING }, actions: ["SPEAK"] };
  }

  if (event === "TTS_END") {
    if (name !== SPEAKING) return stay(state);
    const back = mode === "hands-free" ? LISTENING : IDLE;
    return { state: { ...state, name: back }, actions: [] };
  }

  if (event === "TURN_ABORTED") {
    // The upload was unusable or the request failed. Without this the machine
    // would sit in THINKING forever and the call would be dead.
    if (name !== THINKING) return stay(state);
    const back = mode === "hands-free" ? LISTENING : IDLE;
    return { state: { ...state, name: back }, actions: [] };
  }

  return stay(state);
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `node --test tests/js/turn-state.test.mjs`
Expected: PASS, 16 tests

- [ ] **Step 5: Commit**

```bash
git add src/voicedesk/voice/static/turn-state.js tests/js/turn-state.test.mjs
git commit -m "feat: conversation state machine with barge-in"
```

---

### Task 5: Rewire the browser shell

Connect the two pure modules to real audio, and add the mode toggle. This is the impure layer — verified manually, not by unit tests.

**Files:**
- Rewrite: `src/voicedesk/voice/static/app.js`
- Modify: `src/voicedesk/voice/static/index.html`

**Interfaces:**
- Consumes: `createEnergyVAD` (Task 3), `initialState` / `next` / state constants (Task 4), `/turn` with `heard_chars` (Task 2)
- Produces: nothing consumed by later tasks

- [ ] **Step 1: Update `index.html`**

Change the script tag (line 48) to load a module:

```html
  <script type="module" src="/static/app.js"></script>
```

Change the hint paragraph (lines 30-31) to:

```html
  <p class="hint">Just start speaking — the receptionist listens, and you can
    interrupt it any time. Try:
    <em>"Book me Monday July 13th 2026 at 9am, Jane Doe, 5551234, for a cleaning."</em></p>
```

Add a mode toggle directly above the `#talk` button (after the `#langs` div, line 36):

```html
  <div id="modes">
    <button class="mode active" data-mode="hands-free">Hands-free</button>
    <button class="mode" data-mode="ptt">Hold to talk</button>
  </div>
```

Add styling for it inside the existing `<style>` block, after the `.lang.active` rule (line 25):

```css
    #modes { display: flex; gap: .5rem; margin-bottom: 1rem; }
    .mode { flex: 1; padding: .5rem; border-radius: 8px; border: 1px solid #ccc;
            background: #fff; cursor: pointer; font-size: .85rem; }
    .mode.active { background: #475569; border-color: #475569; color: #fff; }
    #talk.listening { background: #f0fdf4; border-color: #16a34a; color: #15803d; }
    #talk.speaking { background: #fefce8; border-color: #ca8a04; color: #a16207; }
```

- [ ] **Step 2: Rewrite `app.js`**

Replace the entire contents of `src/voicedesk/voice/static/app.js`:

```javascript
import { createEnergyVAD } from "./vad.js";
import {
  IDLE, LISTENING, THINKING, SPEAKING, initialState, next,
} from "./turn-state.js";

const talk = document.getElementById("talk");
const transcriptEl = document.getElementById("transcript");
const replyEl = document.getElementById("reply");
const timingsEl = document.getElementById("timings");

// One session per page load, so the agent remembers this caller across turns.
const sessionId = crypto.randomUUID();

let lang = "en";
const BCP47 = { en: "en-US", zh: "zh-CN" };

let state = initialState("hands-free");
let vad = null;
let audioCtx = null;
let analyser = null;
let micStream = null;
let recorder = null;
let chunks = [];
let rafId = null;

// How far text-to-speech got through the last reply before it was cut off.
// null means "nothing was interrupted"; the server then leaves history alone.
let heardChars = null;
let spokenText = "";

const LABELS = {
  en: { idle: "Start call", listening: "Listening…", thinking: "Thinking…",
        speaking: "Speaking… (interrupt any time)", ptt: "Hold to talk",
        recording: "Listening… release to send", blocked:
        "Microphone blocked — allow mic access and reload.",
        didnt: "(didn't catch that)" },
  zh: { idle: "开始通话", listening: "正在聆听…", thinking: "思考中…",
        speaking: "正在回答…（可随时打断）", ptt: "按住说话",
        recording: "正在聆听…松开发送", blocked: "麦克风被阻止，请允许后重新加载。",
        didnt: "（没有听清）" },
};

function render() {
  const L = LABELS[lang];
  document.documentElement.lang = lang === "zh" ? "zh-CN" : "en";
  talk.classList.remove("recording", "listening", "speaking");
  if (state.mode === "ptt") {
    talk.textContent = state.capturing ? L.recording : L.ptt;
    if (state.capturing) talk.classList.add("recording");
    return;
  }
  if (state.name === IDLE) talk.textContent = L.idle;
  else if (state.name === THINKING) talk.textContent = L.thinking;
  else if (state.name === SPEAKING) {
    talk.textContent = L.speaking;
    talk.classList.add("speaking");
  } else {
    talk.textContent = L.listening;
    talk.classList.add(state.capturing ? "recording" : "listening");
  }
}

function dispatch(event) {
  const result = next(state, event);
  state = result.state;
  for (const action of result.actions) runAction(action);
  render();
}

function runAction(action) {
  if (action === "START_RECORDING") startRecording();
  else if (action === "STOP_AND_SEND") stopRecordingAndSend();
  else if (action === "CANCEL_TTS") cancelSpeech();
  else if (action === "SPEAK") speak(pendingReply, pendingLang);
}

// --- microphone + VAD ------------------------------------------------------

async function openMic() {
  if (micStream) return true;
  try {
    micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (err) {
    replyEl.textContent = LABELS[lang].blocked;
    return false;
  }
  audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  analyser = audioCtx.createAnalyser();
  analyser.fftSize = 1024;
  audioCtx.createMediaStreamSource(micStream).connect(analyser);
  return true;
}

function startVadLoop() {
  const buf = new Float32Array(analyser.fftSize);
  vad = createEnergyVAD();
  const tick = () => {
    analyser.getFloatTimeDomainData(buf);
    let sum = 0;
    for (let i = 0; i < buf.length; i++) sum += buf[i] * buf[i];
    const rms = Math.sqrt(sum / buf.length);
    const event = vad.process(rms, performance.now());
    if (event === "speech-start") dispatch("SPEECH_START");
    else if (event === "speech-end") dispatch("SPEECH_END");
    rafId = requestAnimationFrame(tick);
  };
  rafId = requestAnimationFrame(tick);
}

function stopVadLoop() {
  if (rafId !== null) cancelAnimationFrame(rafId);
  rafId = null;
  vad = null;
}

function startRecording() {
  if (!micStream) return;
  recorder = new MediaRecorder(micStream);
  chunks = [];
  recorder.ondataavailable = (e) => chunks.push(e.data);
  recorder.onstop = () => send(new Blob(chunks, { type: "audio/webm" }));
  recorder.start();
}

function stopRecordingAndSend() {
  if (recorder && recorder.state === "recording") recorder.stop();
}

// --- server round trip -----------------------------------------------------

let pendingReply = "";
let pendingLang = "en";

async function send(blob) {
  if (!blob || blob.size < 1000) {
    transcriptEl.textContent = LABELS[lang].didnt;
    dispatch("TURN_ABORTED");   // nothing usable; get out of THINKING
    return;
  }
  transcriptEl.textContent = "…";
  replyEl.textContent = "";
  timingsEl.textContent = "";

  const form = new FormData();
  form.append("session_id", sessionId);
  form.append("lang", lang);
  form.append("audio", blob, "turn.webm");
  if (heardChars !== null) {
    form.append("heard_chars", String(heardChars));
    heardChars = null;
  }

  try {
    const res = await fetch("/turn", { method: "POST", body: form });
    const data = await res.json();
    transcriptEl.textContent = data.transcript || LABELS[lang].didnt;
    replyEl.textContent = data.reply;
    const t = data.timings;
    timingsEl.textContent =
      `stt ${t.stt_ms}ms · agent ${t.agent_ms}ms · total ${t.total_ms}ms`;
    pendingReply = data.reply;
    pendingLang = data.lang;
    dispatch("REPLY");
  } catch (err) {
    replyEl.textContent = "Something went wrong. Please try again.";
    dispatch("TURN_ABORTED");
  }
}

// --- text to speech --------------------------------------------------------

function speak(text, replyLang) {
  window.speechSynthesis.cancel();
  spokenText = text;
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.lang = BCP47[replyLang] || BCP47.en;
  utterance.rate = 1.05;
  // onboundary reports how far speech has progressed. If the caller interrupts,
  // this is how we know what they actually heard. Not every browser or voice
  // fires it; when it does not, heardChars stays null and the server leaves
  // history untouched.
  utterance.onboundary = (e) => {
    if (typeof e.charIndex === "number") heardChars = e.charIndex;
  };
  // Natural completion (not a barge-in): the caller heard the whole reply,
  // so clear the interruption-tracking state before the next turn starts.
  // Without this, a stale boundary index leaks into the next request and the
  // server truncates a reply the caller actually heard in full.
  utterance.onend = () => {
    heardChars = null;
    spokenText = "";
    dispatch("TTS_END");
  };
  utterance.onerror = (e) => {
    // cancel() fires 'error' with "interrupted"/"canceled" — that is our own
    // barge-in path, and cancelSpeech() has already recorded how much the
    // caller heard. Only a real synthesis failure should discard it.
    if (e.error !== "interrupted" && e.error !== "canceled") {
      heardChars = null;
      spokenText = "";
    }
    dispatch("TTS_END");
  };
  window.speechSynthesis.speak(utterance);
}

function cancelSpeech() {
  // Cancel fires no onend, so heardChars keeps whatever the last boundary was.
  // If no boundary ever fired, treat it as "heard nothing".
  if (heardChars === null && spokenText) heardChars = 0;
  window.speechSynthesis.cancel();
}

// --- controls --------------------------------------------------------------

document.querySelectorAll(".lang").forEach((btn) => {
  btn.addEventListener("click", () => {
    lang = btn.dataset.lang;
    document.querySelectorAll(".lang").forEach((b) =>
      b.classList.toggle("active", b === btn));
    render();
  });
});

document.querySelectorAll(".mode").forEach((btn) => {
  btn.addEventListener("click", async () => {
    const mode = btn.dataset.mode;
    document.querySelectorAll(".mode").forEach((b) =>
      b.classList.toggle("active", b === btn));
    cancelSpeech();
    if (recorder && recorder.state === "recording") {
      // Abandon this turn: detach onstop first so stopping the recorder does
      // NOT trigger send() for audio the caller no longer intends to submit.
      // DISARM emits no action to stop recording, so this must be explicit.
      recorder.onstop = null;
      recorder.stop();
    }
    stopVadLoop();
    dispatch("DISARM");
    state = initialState(mode);
    if (mode === "hands-free" && await openMic()) {
      startVadLoop();
      dispatch("ARM");
    }
    render();
  });
});

talk.addEventListener("click", async () => {
  if (state.mode !== "hands-free") return;
  if (state.name !== IDLE) return;
  if (await openMic()) {
    startVadLoop();
    dispatch("ARM");
  }
});

const pttDown = async (e) => {
  if (state.mode !== "ptt") return;
  e.preventDefault();
  if (await openMic()) dispatch("PTT_DOWN");
};
const pttUp = (e) => {
  if (state.mode !== "ptt") return;
  e.preventDefault();
  dispatch("PTT_UP");
};

talk.addEventListener("mousedown", pttDown);
talk.addEventListener("mouseup", pttUp);
talk.addEventListener("mouseleave", pttUp);
talk.addEventListener("touchstart", pttDown);
talk.addEventListener("touchend", pttUp);

render();
```

- [ ] **Step 3: Verify both automated suites still pass**

Run: `PYTHONPATH=src ./.venv/Scripts/python.exe -m pytest`
Expected: PASS, 291 tests

Run: `node --test tests/js/*.test.mjs`
Expected: PASS, 24 tests

- [ ] **Step 4: Manual browser verification — REQUIRED, do not skip**

Start the server: `PYTHONPATH=src ./.venv/Scripts/python.exe -m voicedesk.voice`
Open http://127.0.0.1:8000 in Chrome or Edge and confirm each of these:

1. **Hands-free happy path (English):** click "Start call", allow the mic, say *"What are your opening hours?"* without touching anything. It transcribes, replies, and speaks — then returns to Listening on its own.
2. **Natural pauses:** say a sentence with a deliberate ~0.5s pause in the middle. It must NOT cut the turn short at the pause.
3. **Barge-in:** ask something that produces a long reply, then start talking over it. Audio must stop within a fraction of a second and recording must begin.
4. **Interruption fidelity:** after barge-in, ask *"what did you just say?"* — the agent must not claim to have said the part you cut off.
5. **Push-to-talk toggle:** switch to "Hold to talk" and confirm hold/release still works exactly as before.
6. **Chinese:** switch to 中文, repeat steps 1 and 3.
7. **Mic denied:** block mic permission and reload — the page shows the "blocked" message and does not crash.

Record any defect found; fix it before committing.

- [ ] **Step 5: Commit**

```bash
git add src/voicedesk/voice/static/app.js src/voicedesk/voice/static/index.html
git commit -m "feat: hands-free conversation with barge-in in the browser"
```

---

### Task 6: CI and documentation

**Files:**
- Modify: `.github/workflows/tests.yml`
- Modify: `README.md`

**Interfaces:**
- Consumes: `tests/js/` from Tasks 3-4
- Produces: nothing

- [ ] **Step 1: Add the Node test step to CI**

In `.github/workflows/tests.yml`, add these two steps after the existing `- run: PYTHONPATH=src pytest` line:

```yaml
      - uses: actions/setup-node@v4
        with:
          node-version: "22"
      - run: node --test tests/js/*.test.mjs
```

- [ ] **Step 2: Verify the workflow file still parses**

Run: `./.venv/Scripts/python.exe -c "import yaml; yaml.safe_load(open('.github/workflows/tests.yml')); print('YAML OK')"`
Expected: `YAML OK`

- [ ] **Step 3: Update the README**

In `README.md`, find the "Live demo" section and change the instruction line so it no longer says to hold a button. Replace:

```
**🎙️ Try it: <https://voicedesk-ch1y.onrender.com>** — hold the button and speak, in English or
中文, to book an appointment by voice. Chrome or Edge (needs a microphone).
```

with:

```
**🎙️ Try it: <https://voicedesk-ch1y.onrender.com>** — start the call and just speak, in English
or 中文, to book an appointment by voice. It listens for when you stop talking, and you can
interrupt it mid-sentence. Chrome or Edge (needs a microphone).
```

- [ ] **Step 4: Run both suites one final time**

Run: `PYTHONPATH=src ./.venv/Scripts/python.exe -m pytest`
Expected: PASS, 291 tests

Run: `node --test tests/js/*.test.mjs`
Expected: PASS, 24 tests

- [ ] **Step 5: Commit**

```bash
git add .github/workflows/tests.yml README.md
git commit -m "ci: run the JS test suite; docs: describe hands-free mode"
```

---

## Completion

After Task 6, merge to `main` and push. Confirm the GitHub Actions run is green before considering the phase done, then let Render auto-deploy and re-verify the live site in both languages.

If the energy VAD proves unreliable in real use, the push-to-talk toggle is the immediate mitigation and reverting the merge is the full rollback — neither requires a code change under pressure.
