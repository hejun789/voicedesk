# VoiceDesk — AI Voice Receptionist for Clinics

**Design spec** · 2026-07-10 · Status: approved, ready for planning

## Purpose

A portfolio-grade, industry-level AI project that stands out for an AI-application-development
internship. It is a **voice agent that takes real actions**: a caller speaks to an AI receptionist
for a dental/medical clinic and it books, reschedules, or cancels appointments for real, answers
FAQs, and escalates to a human when unsure.

**Why this project:** voice agents are rare in student portfolios and a hot 2026 market, so the
difficulty itself is the differentiator. The project also demonstrates the skills companies hire
for in 2026 — agentic tool use, evals/observability, guardrails, and latency engineering — rather
than a commoditized RAG chatbot.

**Business problem it solves:** clinics miss 30%+ of inbound calls; each missed call can be a lost
patient. A 24/7 voice receptionist recovers that revenue and offloads front-desk staff.

## Constraints

- **Cost: $0.** Built entirely on free tiers / open-source. The only paid element (a real Twilio
  phone number) is explicitly deferred; the free build uses browser microphone input.
- **Scope: single clinic, simulated calendar.** No payments, no real EHR integration, no
  multi-tenant. (YAGNI — deliberately cut.)
- **Timeline:** ~2 weeks part-time, in 4 phases (see below).

## Free tech stack

| Layer | Choice | Rationale |
|---|---|---|
| LLM (brain) | Groq free tier (Llama 3.3 70B); Gemini free tier as backup | Free + very low latency, critical for natural voice turn-taking |
| Speech → Text | Groq Whisper (free tier) or local `faster-whisper` | Transcribe caller speech |
| Text → Speech | Web Speech API (browser) for MVP → Piper/Kokoro (local) for quality | The agent's voice, free |
| Telephony | Browser mic (WebRTC / MediaRecorder) | Free alternative to a paid phone line |
| Backend | FastAPI + WebSocket | Streams audio, orchestrates the agent loop |
| Data | SQLite | Appointments + open slots, zero setup |
| Hosting | Hugging Face Spaces or Render free tier | Free, shareable demo link |
| Evals | Custom Python harness | Free |

## Scope: capabilities

The agent can, for real:

1. **Book** an appointment — find open slots, confirm details, persist to DB.
2. **Reschedule / cancel** — look up an existing appointment by name/phone, modify it.
3. **Answer FAQs** — hours, location, insurance accepted — grounded via RAG over a small
   `clinic_info.md`.
4. **Escalate** — "a human will call you back" — triggered by a confidence gate when input is
   garbled, ambiguous, or no matching appointment is found.

Explicitly out of scope: payments, real EHR/calendar integration, multiple clinics, account auth.

## Architecture / data flow

```
Browser (mic + speaker)
   │   audio in  ▲ audio out
   ▼            │
FastAPI WebSocket server
   │ 1. STT (Whisper) → text
   │ 2. Agent loop (Groq LLM + tool calling)
   │       tools: find_slots, book, reschedule,
   │              cancel, lookup_appt, answer_faq, escalate
   │ 3. Tool executes against SQLite calendar
   │ 4. LLM composes reply → TTS → audio out
   ▼
SQLite (appointments, slots)  +  clinic_info.md (RAG for FAQs)
```

The **agent loop with tool calling** is the core: the LLM decides which tool to invoke, observes the
tool result, and composes the spoken reply. This is the 2026 "agent that takes actions" pattern.

### Component boundaries

- **STT module** — audio bytes in, transcript text out. Swappable (Groq API vs local Whisper).
- **Agent core** — transcript + conversation state in, tool call(s) + reply text out. No audio
  knowledge. This is what the eval harness drives directly (text in / text out).
- **Tools** — pure functions over the SQLite calendar (`find_slots`, `book`, `reschedule`,
  `cancel`, `lookup_appt`) plus `answer_faq` (RAG) and `escalate`. Each independently unit-testable.
- **TTS module** — reply text in, audio out. Swappable (Web Speech API vs Piper).
- **WebSocket server** — wires the four together and streams to the browser.

## What makes it industry-level (not a toy)

- **Eval harness** — ~30 scripted caller scenarios (clear bookings, reschedules, cancellations,
  ambiguous and adversarial inputs) scored automatically on: correct tool chosen? correct slot
  booked? escalated when it should? This is the primary résumé asset and doubles as the integration
  test suite (runs text-in/text-out, skipping audio).
- **Confidence gate / guardrails** — the agent escalates instead of guessing on low confidence,
  no-match lookups, or garbled input.
- **Latency budget** — measured and reported end-to-end response time (STT + LLM + TTS); the key
  voice-agent metric, surfaced in the README.
- **Barge-in handling** (stretch) — allow the caller to interrupt the agent mid-sentence.

## Testing strategy

- **Unit tests (TDD)** on each tool: booking logic, double-booking/slot-conflict prevention,
  lookup by name/phone, cancel/reschedule state transitions.
- **Eval suite as integration tests** — the ~30 scenarios exercise the agent core end-to-end in
  text mode.
- **Manual voice testing** for the live demo and latency measurement.

## Build phases (~2 weeks part-time)

1. **Text agent + tools + SQLite** — prove the brain works, text-only (no audio).
2. **Eval harness** — lock in quality before adding audio complexity.
3. **Add voice** — STT + TTS + WebSocket audio streaming in the browser.
4. **Deploy + polish** — shareable link, README written as a case study (problem → approach →
   metrics → tradeoffs), latency numbers, demo script.

## Success criteria

- A shareable live link where anyone can click "talk" and book a real appointment by voice.
- Eval suite runs with a reported pass rate and at least one measured improvement.
- README case study with latency numbers and an honest limitations section.
- Clean component boundaries so STT/TTS/LLM providers are swappable.
