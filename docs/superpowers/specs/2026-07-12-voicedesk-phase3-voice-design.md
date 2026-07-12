# VoiceDesk Phase 3 — Voice Layer

**Design spec** · 2026-07-12 · Status: approved, ready for planning

## Purpose

Give the existing text agent a voice: a caller speaks into a browser, the agent hears them,
takes real actions against the calendar, and speaks back. This is the project's headline
differentiator — voice agents are rare in student portfolios and the difficulty is the point.

The agent core is **not modified**. `Agent.respond(text) -> str` is the seam the voice layer wraps.

## Foundational decisions

Three choices, made deliberately, that together simplify the architecture:

1. **Push-to-talk, turn-based.** The caller holds a button, speaks, releases. No always-on mic,
   no voice-activity detection, no barge-in. This isolates what actually matters — the agent
   taking real actions by voice — instead of sinking the phase into interruption logic.
2. **Speech-to-text via Groq Whisper** (`whisper-large-v3-turbo`). Fast, free, reuses the `groq`
   SDK already in the project, and — critically — it draws on a **separate rate-limit pool** from
   the chat model (measured in audio-seconds), so it does not compete with the LLM quota that has
   already caused trouble in this project.
3. **Text-to-speech in the browser** via the Web Speech API (`speechSynthesis`). Zero cost, zero
   dependencies, and **zero network latency** — speech begins the instant the reply arrives, which
   directly improves the headline turn-latency metric. The voice is somewhat robotic; that is an
   accepted trade.

**Consequence — no WebSockets.** The original Phase 1 spec assumed FastAPI + WebSocket audio
streaming. Because TTS now happens in the browser and turns are discrete, there is nothing to
stream: **one HTTP POST per turn is sufficient.** This removes streaming, partial transcripts, and
audio-chunk plumbing at no cost to the demo.

## Constraints

- **Cost $0.** Groq free tier only. No paid services, no Twilio.
- **Agent core unchanged.** `agent.py`, `tools.py`, `db.py`, `faq.py`, `registry.py`, `llm.py`,
  `groq_client.py` are not modified by this phase.
- **The test suite stays fully offline.** No test may require a network call, an API key, or a
  microphone. STT sits behind a protocol so tests inject a fake, exactly as `FakeLLM` does today.
- New runtime dependencies are limited to `fastapi`, `uvicorn`, and `python-multipart`.
- Python 3.11+.

## Flow

```
Browser                              FastAPI server
───────                              ──────────────
[hold to talk] record mic
  │  (MediaRecorder → webm blob)
  └── POST /turn (audio, session_id) ──►
                                       1. Groq Whisper   → transcript
                                       2. Agent.respond(transcript) → reply
                                       3. time each stage
  ◄── {transcript, reply, timings} ────┘
  │
  └── speechSynthesis.speak(reply)   🔊
```

One round trip per turn. The agent is invoked exactly as the CLI invokes it.

## Components

```
src/voicedesk/voice/__init__.py
src/voicedesk/voice/stt.py            # transcribe(audio: bytes) -> str, behind a protocol
src/voicedesk/voice/session.py        # session_id -> Agent, with idle expiry
src/voicedesk/voice/server.py         # FastAPI: POST /turn, GET / (serves the page)
src/voicedesk/voice/static/index.html # mic button, transcript, reply, live latency
src/voicedesk/voice/static/app.js     # MediaRecorder + fetch + speechSynthesis
```

Boundaries, each unit understandable and testable alone:

- **`stt.py`** — a `STTClient` protocol with `transcribe(audio: bytes, filename: str) -> str`;
  `GroqWhisper` implements it against the Groq audio API (lazy import, like `GroqLLM`); `FakeSTT`
  returns a scripted transcript for tests. The module knows nothing about HTTP or the agent.
- **`session.py`** — an in-memory `{session_id: (Agent, last_used_at)}` map. `get_or_create(id)`
  returns the caller's `Agent`, so conversation history accumulates across turns (the caller can
  say "book me Monday", then supply their name on the next turn). Sessions idle out after a
  timeout so the map cannot grow without bound. Knows nothing about HTTP or audio.
- **`server.py`** — wires STT → Agent → JSON and serves the static page. Contains no booking
  logic and no knowledge of tools.

## Latency

Voice is judged on turn latency, so it is measured and surfaced rather than hidden. Every response
carries a breakdown:

```json
{
  "transcript": "book me Monday at 9am",
  "reply": "Sure — can I take your name and phone number?",
  "timings": {"stt_ms": 380, "agent_ms": 1450, "total_ms": 1830}
}
```

The page displays this live. Because browser TTS starts speaking immediately, there is no hidden
TTS cost inflating the real number — the figure is honest and is the one to report in the README.

Note this is **per-turn** latency, which is the metric a voice product is actually judged on —
unlike the per-conversation figure the Phase 2 eval reports.

## Error handling

- **Empty or unintelligible audio** → Whisper returns an empty/near-empty transcript; the server
  responds with a spoken "Sorry, I didn't catch that." and does NOT call the agent (saves quota and
  avoids polluting the conversation history with noise).
- **STT failure** (API error) → HTTP 200 with a spoken apology and an `error` field, never a stack
  trace. The demo must never crash mid-call.
- **LLM failure** → already handled: `Agent.respond` degrades to its escalation reply.
- **Unknown session_id** → a new session is created rather than erroring.

## Testing strategy

- **`stt.py`** — unit-tested against a fake Groq client with no network, mirroring the existing
  `groq_client.py` tests.
- **`session.py`** — tested for: the same id returns the same `Agent` (history persists), different
  ids are isolated, and idle sessions expire.
- **`POST /turn`** — tested with FastAPI's `TestClient`, injecting `FakeSTT` + `FakeLLM`. This
  exercises a **complete voice turn end-to-end with zero network**: audio bytes in, transcript +
  reply + timings out, and a real appointment written to SQLite.
- **The Phase 2 eval harness is unchanged.** It drives the text interface, which is precisely what
  the voice layer wraps. Voice is a transport, not new agent behaviour, so there is nothing new to
  evaluate — and the existing 30 scenarios keep protecting the agent.

## Explicitly out of scope (YAGNI)

Barge-in, voice-activity detection, audio streaming, Twilio / a real phone number, authentication,
server-side audio persistence, and speaker identification.

## Success criteria

- A browser page where holding a button and speaking books a real appointment, end to end.
- The agent speaks its reply aloud.
- Multi-turn works: the caller can supply details across several turns and the booking completes.
- Per-turn latency is measured, displayed, and reportable.
- The full test suite still runs offline with no API key, and still passes.
- The agent core is byte-for-byte unchanged.
