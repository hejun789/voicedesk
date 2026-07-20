# VoiceDesk — AI Voice Receptionist for Clinics

An AI agent that answers the phone for a dental clinic: it **books, reschedules and
cancels real appointments**, answers questions about the practice, and hands off to a
human when it shouldn't be acting alone.

It is not a chatbot. It takes actions against a real calendar, and every action it
takes is measured by an evaluation harness.

**Headline result: eval-driven debugging took the agent from a 53.3% to an 86.7% pass
rate across 30 scripted caller scenarios. The most rigorous measurement — 30 scenarios
× 3 runs, 90 runs total — puts it at 84.4% with per-scenario reliability data.
Voice turns complete in ~1.9s (p50).**

**But the part worth reading is [what using the product caught that the eval
passed](#what-using-the-product-caught-that-the-eval-passed), and [the two bugs in the
eval itself](#and-two-bugs-in-the-eval-itself).**

---

## Live demo

**🎙️ Try it: <https://voicedesk-ch1y.onrender.com>** — hold the button and speak, in English or
中文, to book an appointment by voice. Chrome or Edge (needs a microphone).

> It runs on a free tier: it sleeps when idle (≈30s to wake on the first hit) and is
> rate-limited, so it may reach its daily cap — in which case it says so politely.
> **Demo video:** _(add your 30-second recording here)_

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
Browser (mic + speech) ─┐
                        ├─►  Agent core  ──►  Tool registry  ──►  Tools ──► SQLite calendar
CLI (text) ─────────────┘   (LLM loop)         (7 tools)                └──► clinic_info.md (FAQ)
                              │
                         LLMClient protocol          Voice turn: POST /turn
                              ├── GroqLLM (prod)      browser audio → Groq Whisper (STT)
                              └── FakeLLM (tests)     → Agent.respond() → browser speaks reply
```

Every boundary is deliberate:

- **Tools know nothing about the LLM.** They're pure functions over a SQLite
  connection, so they're unit-testable with an in-memory database.
- **The agent knows nothing about the provider.** It talks to an `LLMClient`
  protocol, so the entire **256-test** suite runs offline against a `FakeLLM` — no
  network, no API key, no cost.
- **The agent knows nothing about audio.** That's why the voice layer (Groq Whisper
  in, browser speech-synthesis out) was an *additive* change that never touched the
  agent core — the same `Agent.respond(text) -> str` the CLI and the eval harness call.

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

## What using the product caught (that the eval passed)

Then I ran the voice demo myself, and it found five more — **including one the eval
was actively reporting as PASS.**

**6. The agent silently double-booked.**
I booked 9am, then said *"change it to 10am"*. It replied *"your appointment has been
booked for 10am"* — and created a **second** appointment. The 9am slot stayed occupied.
**The eval scenario for this passed**, because its assertion only checked that the new
appointment existed — never that the old one was gone.
**Fix:** `book()` now refuses a second same-day booking for the same patient
(`already_booked_that_day`), and the scenario now asserts `appointment_count: 1`, so it
can actually fail on the bug it was written for.

**7. It committed a phone number it mis-heard.**
Whisper transcribed `5551234` as `55512344`. The agent booked it instantly. That
appointment is attached to a number that doesn't exist — unreachable, uncancellable.
No STT is reliable on digit sequences; they carry no linguistic context.
**Fix:** the agent must now read the details back — phone **digit by digit** — and get an
explicit yes *before* calling `book()`. This is what every real phone system does, and it
is the only thing that actually works.

**8. It read the database ID out loud.** *"Your appointment ID is 1."* No receptionist
says that. **Fix:** internal identifiers are for tool calls, never for callers.

**9. Whisper mis-heard "Jane Doe" as "gym, dorm."** **Fix:** a transcription prompt
biasing Whisper toward the vocabulary a clinic call actually contains. Proper nouns are
its weakest point without context.

**10. The agent gave up mid-call** — *"let me have a team member call you back"* — on a
perfectly good request. Invisible, because the voice server logged nothing.
**Fix:** surfaced LLM retries and fallbacks in the server log, then found the cause: two
resamples was too few for Groq's flaky tool-calling. Raised the budget.

---

## And two bugs in the eval itself

Reading the *failures* (not the score) found two cases where the agent was right and the
**test** was wrong:

**11. A Unicode space defeated an assertion.** `faq_location` failed with
*"reply did not contain 'Market Street'"* — while the reply read
`located at␠200␠Market␠Street`. Those are U+202F **narrow no-break spaces**. The answer
was perfect; the substring check was brittle. **Fix:** `reply_contains` now normalizes
whitespace, so it tests the words rather than the typography.

**12. The eval punished the agent for being careful.** Every cancel/reschedule failure
looked like `lookup_appt(...)` → *stop*. The agent was asking *"shall I cancel your
Monday 9am?"* — and the scenario had no turn left to answer, so it never acted.
Confirming before a destructive action is **correct**; the scenario was under-specified.
**Fix:** made the confirm-before-destructive-action contract explicit in the prompt, and
gave those scenarios their confirmation turn. The agent wasn't made more reckless to
satisfy the test.

---

## Results

**Most rigorous measurement — 30 scenarios × 3 runs (90 runs), `openai/gpt-oss-120b`:**

| Category | Pass rate |
|---|---:|
| Unavailable slots | **100%** (12/12) |
| Lookup | **100%** (6/6) |
| Ambiguous input | **100%** (9/9) |
| Escalation | **93%** (14/15) |
| FAQ | 92% (11/12) |
| Booking | 89% (16/18) |
| Cancel | 56% (5/9) |
| Reschedule | 33% (3/9) |
| **Overall** | **84.4%** (76/90) |

**Per-turn voice latency: p50 1.86s** end-to-end (speech in → action taken → speech out).

**The improvement story** — same model, same scenarios, one variable changed (the bugs
above), single run each: **53.3% → 86.7%** on `llama-3.1-8b-instant`. Booking went
33% → 100%, FAQ 0% → 100%.

**Honest caveats** (these matter more than the number):

- **The 84.4% predates the two eval-bug fixes (#11, #12).** Most of the cancel/reschedule
  gap is the eval cutting the conversation off before the agent could act — not the agent
  failing. A partial re-run after the fixes showed FAQ at 100% and booking at 94% before
  free-tier quota stopped it; **the full post-fix number is not yet measured, so the
  honest headline stays at 84.4%.**
- **The 53.3% → 86.7% pair are single runs.** Single-run eval numbers are noisy — I watched
  the same scenario fail and then pass on consecutive runs. That is exactly why the harness
  runs each scenario 3× and reports FLAKY as a first-class result.
- **`llama-3.1-8b-instant` is not a viable production model here.** It invents
  `appointment_id`s rather than using the one `lookup_appt` returned — a small-model
  multi-step-tool-use failure. `openai/gpt-oss-120b` doesn't do this, and doesn't have the
  Llama family's malformed-`<function=...>` bug either. That choice was made from eval data,
  not vibes.
- The weekend/out-of-hours scenarios prove the *database* invariant (no appointment is
  created) but cannot detect a hallucinated verbal confirmation. That needs a judge.
- **I stopped tuning deliberately.** Grinding a prompt until the eval goes green optimizes
  the test, not the product.

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

### Run the tests (256, fully offline — no API key needed)
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

## Bilingual (English + 中文)

Verified end to end against the live API: a Mandarin caller books an appointment by voice,
the agent reads the phone number back digit by digit in Chinese (五五五一二三四), and the
browser speaks the reply in a Mandarin voice.

The agent takes calls in English or Chinese. Language is **explicit configuration, not
inference** — a toggle on the page sends `lang` with each turn, which selects Whisper's
language, the clinic document, the system prompt, and the browser's TTS voice. Whisper can
auto-detect, but it is unreliable on short utterances and would make every downstream
choice depend on a guess.

The interesting part was **FAQ retrieval**. It scored English by word overlap — and
`_tokens("你们的营业时间")` returns an **empty set**, because the regex was `[a-z]+`. Even a
Unicode-aware regex would not have helped: Chinese has no spaces between words, so there is
nothing to split on. The fix is character 2-grams (`营业时间` → `{营业, 业时, 时间}`), used
**only as a fallback when word tokenization yields nothing**. English keeps its measured
word path untouched; Chinese gets a path that works; no new dependencies.

Also Chinese-specific: Whisper hallucinates `谢谢观看` ("thanks for watching", learned from
YouTube subtitles) on silence, so the silence denylist needed Chinese entries — otherwise
noise would reach the booking tools.

All 30 scenarios are mirrored in Chinese with **identical expectations**, so the two
languages are directly comparable. Each suite runs independently, which matters on a free
tier that allows roughly one full 3× run per day:

```powershell
python -m voicedesk.evals --lang en --runs 3
python -m voicedesk.evals --lang zh --runs 3
```

---

## Roadmap

- **Phase 1 — text agent** ✅ tool-calling loop, SQLite calendar, graceful escalation
- **Phase 2 — eval harness** ✅ 30 scenarios, outcome scoring, flakiness + latency
- **Phase 3 — voice** ✅ push-to-talk in the browser: Groq Whisper (STT) → the same
  agent → browser speech-synthesis (TTS). One HTTP POST per turn — no WebSockets needed,
  because the browser speaks the reply itself. The agent core and eval harness carry over
  unchanged, because neither knows about audio. ~2s per turn end-to-end.
- **Phase 5 — Chinese** ✅ bilingual voice (EN/中文): explicit language config, character
  n-gram FAQ retrieval for Chinese, Chinese silence-artefact handling, and all 30 eval
  scenarios mirrored with identical expectations for a true per-language comparison
- **Phase 4 — deploy** hosted demo, per-turn latency budget, cost per resolution
