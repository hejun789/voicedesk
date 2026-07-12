# VoiceDesk — AI Voice Receptionist for Clinics

An AI agent that answers the phone for a dental clinic: it **books, reschedules and
cancels real appointments**, answers questions about the practice, and hands off to a
human when it shouldn't be acting alone.

It is not a chatbot. It takes actions against a real calendar, and every action it
takes is measured by an evaluation harness.

**Headline result: fixing the bugs the eval harness found took the agent from a
53.3% to an 86.7% pass rate across 30 scripted caller scenarios.**

---

## The problem

Clinics miss 30%+ of inbound calls. Every missed call is a potentially lost patient,
and front-desk staff are interrupted constantly by routine scheduling. A receptionist
that works 24/7 and handles the routine 80% is a direct revenue and staffing win —
*provided it is trustworthy*.

"Trustworthy" is the hard part, and it's what this project is really about.

---

## What it does

| Capability | Detail |
|---|---|
| **Book** | Finds open slots and books a real appointment into a SQLite calendar |
| **Reschedule / cancel** | Looks the caller up by name or phone, then modifies their booking |
| **Answer FAQs** | Hours, location, insurance, services — grounded in a clinic document |
| **Escalate** | Hands off to a human on medical issues, billing disputes, unintelligible input, or anything out of scope |

Clinic hours are weekdays, hourly, 09:00–16:00. Double-booking is impossible by
construction (a partial unique index over *booked* rows).

---

## Architecture

```
CLI (text)  ──►  Agent core  ──►  Tool registry  ──►  Tools ──► SQLite calendar
                (LLM loop)         (7 tools)                └──► clinic_info.md (FAQ retrieval)
                     │
                LLMClient protocol
                     ├── GroqLLM   (production)
                     └── FakeLLM   (tests — no network)
```

Every boundary is deliberate:

- **Tools know nothing about the LLM.** They're pure functions over a SQLite
  connection, so they're unit-testable with an in-memory database.
- **The agent knows nothing about the provider.** It talks to an `LLMClient`
  protocol, so the entire 154-test suite runs offline against a `FakeLLM` — no
  network, no API key, no cost.
- **The agent knows nothing about audio**, which is what makes the planned voice
  layer (STT/TTS) an additive change rather than a rewrite.

---

## The evaluation harness — the actual point of this project

Anyone can demo an agent booking an appointment. The interesting question is: **how
often is it right, and how do you know?**

`evals/scenarios.json` holds 30 scripted caller scenarios — multi-turn conversations
replayed against the real LLM, scored on **objective outcomes**, not vibes:

```json
{
  "id": "cancel_by_phone",
  "seed": [{"patient_name": "Jane Doe", "phone": "5551234",
            "slot_iso": "2026-07-13T09:00", "reason": "cleaning"}],
  "turns": ["Hi, I need to cancel my appointment", "Jane Doe, 5551234"],
  "expect": {
    "tools_called": ["lookup_appt", "cancel"],
    "escalated": false,
    "appointments": [{"patient_name": "Jane Doe", "status": "cancelled"}]
  }
}
```

Design decisions worth calling out:

- **Outcome-based scoring, not an LLM judge.** Did the right tool get called? Is the
  appointment *actually in the database*, at the right time, in the right state? These
  are facts, not opinions — cheap to check and impossible to fudge.
- **Every run gets a fresh in-memory database**, so runs can't contaminate each other.
- **The harness observes the agent without modifying it** — tool calls are read back
  out of the agent's own message history, and an error-capturing proxy catches API
  failures the agent deliberately swallows.
- **Scenarios run N times and report PASS / FLAKY / FAIL.** LLMs are
  non-deterministic; a scenario that passes 2 out of 3 times is a *reliability finding*,
  not noise to hide.
- **API errors are counted separately from agent failures.** A run that dies on a rate
  limit is not evidence the agent is broken, and conflating the two produces confidently
  wrong numbers.
- **Escalation is deliberately over-weighted** (5 of 30 scenarios). Knowing when to
  *stop* is the safety property of the whole system.

---

## What the eval caught (and a demo never would)

Five real defects, every one invisible to a happy-path demo:

**1. A trailing `:00` was silently destroying bookings.**
The model emitted `"2026-07-17T16:00:00"`; `find_slots` produced `"2026-07-17T16:00"`.
`book()` compared them as strings, found no match, and rejected a *perfectly correct*
booking as "slot unavailable". The agent was right; the tool was brittle.
**Fix:** normalize model-supplied timestamps at the tool boundary — while still
rejecting a request for 09:30, because we must never silently move a caller to a
different time than they asked for.

**2. The agent was creating ghost appointments.**
It called `book()` with `patient_name: "unknown"` — and, on another run,
`patient_name: "patient_name"`, literally the parameter name. Because the tool schema
marked the field required, the model invented a value rather than asking the caller.
Real slots were consumed by patients who could never be looked up, called, or cancelled.
**Fix:** a hard guardrail in the tool itself that rejects placeholder values. The prompt
asks the model to behave; the tool boundary *enforces* it. Don't trust the model —
validate.

**3. The agent didn't know what day it was.**
No current date in the system prompt and no date tool, so "Monday" was unanchored. It
rescheduled an appointment to **2023-12-15**.
**Fix:** ground the system prompt in today's date and require absolute timestamps.

**4. It failed to escalate a medical emergency.**
A caller reporting an allergic reaction to anesthetic got a chatty reply instead of a
handoff to a human. This is the single most serious failure a receptionist agent can have.
**Fix (defense in depth):** when FAQ retrieval finds nothing, the *tool result itself*
now instructs the model to escalate. The safety path no longer depends on the model
remembering one line of a long system prompt.

**5. It retrieved FAQ answers and then ignored them.**
It correctly called `answer_faq("Cigna insurance")`, received the answer, and then
replied *"Do you have an appointment scheduled?"*.
**Fix:** prompt now requires it to relay what it retrieved.

---

## Results

30 scenarios, `llama-3.1-8b-instant`, identical scenarios before and after:

| Category | Before | After |
|---|---:|---:|
| Booking | 33% | **100%** |
| FAQ | 0% | **100%** |
| Cancel | 67% | **100%** |
| Reschedule | 33% | 67% |
| Escalation | 40% | 60% |
| Unavailable slots | 100% | 100% |
| Ambiguous input | 100% | 100% |
| **Overall** | **53.3%** | **86.7%** |

**Honest caveats** (these matter more than the number):

- These are single-run figures. A full 3× reliability pass was blocked by free-tier
  quota; the harness supports it (`--runs 3`) and reports FLAKY scenarios when run.
- The remaining failures are concentrated in **escalation**, which is the category I'd
  fix next precisely because it's the safety-critical one.
- The weekend/out-of-hours scenarios prove the *database* invariant (no appointment is
  created) but cannot yet detect a hallucinated verbal confirmation. That needs a
  judge-based check.
- I stopped tuning at 86.7% deliberately. Grinding a prompt until the eval goes green is
  optimizing the test, not the product.

---

## Operational engineering

The eval is a real workload (~250 API calls per pass, ~5 LLM calls per conversation),
and it surfaced provider problems that a toy project never sees:

- **Malformed tool calls.** Groq's Llama models intermittently emit a broken
  `<function=name{...}>` format that the API itself rejects. Non-deterministic — the same
  prompt succeeds on one run and fails on the next. The adapter detects and resamples.
- **Rate limiting done properly.** The client reads `x-ratelimit-remaining-tokens` from
  every response and **throttles before sending**, rather than provoking a 429 and
  reacting — because a rejected request still costs quota.
- **Quota exhaustion fails fast.** A `Retry-After` measured in minutes means a *daily*
  quota, which will never clear by retrying. The harness detects this, stops immediately
  with a clear message, and keeps partial results.
- **Graceful degradation.** An LLM failure never crashes the caller — the agent falls
  back to "let me have a team member call you back", which is the correct behavior for a
  receptionist.

---

## Run it

### Setup
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# Free API key at https://console.groq.com — copy .env.example to .env and paste it in
```
Put the real key in `.env` (gitignored), never in `.env.example`.

### Talk to the agent
```powershell
$env:PYTHONPATH = "src"; python -m voicedesk.cli
```
> *"Book me Monday July 13th 2026 at 9am, Jane Doe, 5551234, for a cleaning."*

### Talk to it by voice (Phase 3)
```powershell
$env:PYTHONPATH = "src"; python -m voicedesk.voice
```
Open <http://127.0.0.1:8000>, hold the button, and speak. The browser records your
voice, Groq Whisper transcribes it, the same agent takes the action, and the browser
speaks the reply back. Each turn shows its latency breakdown (stt / agent / total).

Use Chrome or Edge — it needs `MediaRecorder` and the Web Speech API.

### Run the tests (178, fully offline — no API key needed)
```powershell
$env:PYTHONPATH = "src"; python -m pytest -q
```

### Run the evals (needs an API key; run from the repo root)
```powershell
$env:PYTHONPATH = "src"
python -m voicedesk.evals --runs 1              # ~30 scenarios
python -m voicedesk.evals --runs 3              # + flakiness data
python -m voicedesk.evals --scenario faq_hours --runs 1   # one scenario, fast
```
Prints a console report and writes `reports/eval-<timestamp>.md`, stamped with the model
used. Set `GROQ_MODEL` in `.env` to compare models.

---

## Roadmap

- **Phase 1 — text agent** ✅ tool-calling loop, SQLite calendar, graceful escalation
- **Phase 2 — eval harness** ✅ 30 scenarios, outcome scoring, flakiness + latency
- **Phase 3 — voice** speech-to-text + text-to-speech over a WebSocket; the agent core
  and the eval harness carry over unchanged, because neither knows about audio
- **Phase 4 — deploy** hosted demo, per-turn latency budget, cost per resolution
