# VoiceDesk Phase 6 — Conversational Turn-Taking

**Design spec** · 2026-07-26 · Status: approved, ready for planning

## Purpose

Replace push-to-talk with hands-free, interruptible conversation: the agent detects when the
caller starts and stops speaking, and stops talking the moment the caller cuts in.

This is the change that moves VoiceDesk from "voice-enabled chatbot" to "voice agent" as
practitioners use the term. Every production voice agent is full-duplex; VoiceDesk today is
half-duplex and turn-based — hold a button, release, wait. That single interaction choice is the
clearest "demo, not product" tell in the project.

## What is actually missing today

Phase 3 deliberately chose one HTTP POST per turn with browser-side TTS ("no WebSockets — TTS
runs in the browser so one HTTP POST per turn suffices"). That decision still holds for
transport, but it left three gaps:

1. **No endpointing.** The caller must hold and release a button. Nothing on a real phone call
   works this way.
2. **No barge-in.** `app.js` speaks a reply to completion. A caller cannot interrupt; there is no
   mechanism to stop mid-utterance.
3. **No conversational state.** There is no model of "the agent is currently speaking" versus
   "thinking" versus "listening", so there is nothing for an interruption to act against.

## Constraint discovered during design

**Groq Whisper cannot stream.** `stt.py` sends a complete audio file and receives one transcript
(`client.audio.transcriptions.create`); there is no partial-result API. Genuine streaming STT
would require swapping to the browser's `SpeechRecognition`, which would drop the Whisper
integration and restrict support to Chrome/Edge. That is explicitly **not** part of this phase.
Endpointing and barge-in do not require streaming STT — they require knowing *when* speech starts
and stops, which is a client-side signal.

## Foundational decisions

1. **Energy VAD behind a swappable interface, not a neural VAD.** ~50 lines of Web Audio API
   (loudness threshold plus a silence "hangover" timer) with zero dependencies, structured so a
   better detector can replace it. Silero-via-onnxruntime-web is what production frameworks use
   and is far more robust in noise, but costs every visitor a ~10MB WASM runtime plus a model
   download on a free-tier host. The interface boundary matters more than the first
   implementation, and mirrors the project's existing `LLMClient` / `STTClient` pattern.
2. **Push-to-talk survives as a toggle.** Hands-free is the default; hold-to-talk remains
   available. Energy VAD is the risky new component — in a noisy room or on a poor microphone it
   will misfire — and the live demo needs a proven fallback rather than being simply broken for
   that visitor. Cost: two input paths to maintain and test.
3. **Barge-in interrupts speaking, not thinking.** While an LLM request is in flight the mic is
   not armed for interruption. FastAPI does not cancel work already running in the threadpool, so
   aborting the fetch client-side would leave the browser and `agent.messages` disagreeing about
   what happened. "Interrupt while talking" is the standard definition of barge-in and avoids
   that entire class of bug.
4. **Interruption truncates server-side history, in Python.** See below. The logic is placed on
   the server specifically so it is covered by the existing pytest suite rather than living as
   untestable browser code.

## Architecture

```
static/vad.js         pure: (loudness, timestamp) -> "speech-start" | "speech-end" | null
static/turn-state.js  pure: next(state, event) -> { state, actions }
                      states: IDLE -> LISTENING -> THINKING -> SPEAKING
static/app.js         impure shell: getUserMedia, AnalyserNode -> RMS, MediaRecorder,
                      fetch, speechSynthesis, mode toggle. Wires the pure modules
                      to the browser and contains no decision logic of its own.
```

The two pure modules have no DOM and no Web Audio dependency, so they are unit-testable under
Node. `app.js` stays deliberately thin — the same separation the project already applies to LLM
and STT clients.

In hands-free mode the microphone stays open for the whole call: after `SPEAKING` finishes the
machine returns to `LISTENING`, not `IDLE`, so the caller can simply keep talking. `IDLE` is the
pre-call state and the resting state in push-to-talk mode. The toggle does not change the state
machine itself — it changes which events drive it, VAD signals versus button press and release —
so both input modes share one tested implementation.

Server changes are small: `/turn` gains one optional form field carrying how many characters of
the previous reply the caller actually heard, and `Agent` gains a method to truncate its last
assistant message to that prefix.

## Interruption fidelity

When the caller cuts the agent off at *"Your appointment is booked for Monday at nine—"*, the
server's history still records the full reply, including text the caller never heard. The agent
then believes it said things that were never spoken and may reference them later in the call.

`SpeechSynthesisUtterance` fires `onboundary` events carrying a `charIndex` while speaking. The
client tracks the furthest index reached; on barge-in that value is how much was actually heard.
It is sent with the next `/turn` POST, and the server truncates the last assistant message to
that prefix before appending the new user turn, followed by a short `[interrupted]` marker so the
model knows it was cut off and does not simply repeat itself.

**Degradation:** if `onboundary` never fires (unreliable on some browsers and voices) the client
sends nothing and the server leaves history untouched — today's behavior exactly. Out-of-range
indices are clamped. A reply truncated to nothing is dropped from history entirely. Assistant
messages carrying `tool_calls` are never touched; only the final spoken text reply is subject to
truncation.

## Testing

**JavaScript — `node --test`, no npm packages:**

- `vad.test.mjs` — silence stays silent; a burst above threshold fires `speech-start` exactly
  once rather than repeatedly; a brief dip mid-sentence does **not** end the turn (natural pauses
  between words are the most common way energy VAD gets this wrong); input shorter than
  `minSpeechMs` (a click or cough) is rejected.
- `turn-state.test.mjs` — legal transitions only; barge-in fires solely from `SPEAKING`; events
  arriving in the wrong state are no-ops; a reply landing after the user has already barged in
  does not resurrect `SPEAKING`.

Both modules take injected timestamps, so tests are deterministic with no fake timers and no
async — the same approach `RateLimiter` uses with its injectable clock.

**Python — existing pytest suite:** truncation behavior (clamping, dropping when empty, leaving
`tool_calls` messages untouched, no-op when no assistant message exists), plus `/turn` accepting
the new optional field, including a regression guard that its absence behaves exactly as today.

**CI:** `.github/workflows/tests.yml` gains a Node step so both suites run on every push.

**Known gap, not covered by automated tests:** the browser wiring in `app.js` — `getUserMedia`,
`AnalyserNode`, `MediaRecorder`, real TTS timing and `onboundary` behavior. Every prior phase's
live review caught runtime defects the offline fakes hid (Phase 3 found 5, Phase 5 found 6). A
manual browser pass in both English and Chinese is therefore a required step before merge, not an
optional one.

## Rollout

1. Branch `phase6-turn-taking` — the largest change since Phase 3, and it rewrites the live
   demo's interaction model.
2. Install Node (for `node --test`); add the Node step to CI.
3. Implement and verify both suites green locally.
4. Manual browser pass in both languages, against the local dev server where possible — manual
   voice testing consumes Groq quota, which the project has already hit its daily cap on once.
5. Merge to main, push, confirm the Actions run is green, let Render auto-deploy.
6. Mitigation if VAD misbehaves in the wild: the push-to-talk toggle is the immediate fallback;
   a commit revert is the full rollback.

## Out of scope

- Streaming LLM tokens and sentence-chunked TTS (the separate "streaming latency" subsystem —
  its own spec and plan, deliberately not tangled with this frontend rework)
- Streaming STT / replacing Groq Whisper with browser `SpeechRecognition`
- Barge-in during the thinking state (see decision 3)
- WebSocket transport; the existing one-POST-per-turn protocol is retained
- Telephony (Twilio or otherwise) — not free, explicitly ruled out by project constraints
