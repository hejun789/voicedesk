# VoiceDesk Phase 2 — Eval Harness

**Design spec** · 2026-07-10 · Status: approved, ready for planning

## Purpose

Prove — with numbers — how well the Phase 1 agent actually performs. Phase 1 demonstrates that
the agent *can* book an appointment; Phase 2 measures *how reliably* it does so across a spread of
realistic caller scenarios, including the ones where it should refuse and escalate.

This is the highest-value phase for the project's purpose (an AI-application internship portfolio):
it converts "I built an agent" into "I measured my agent and improved it," which is the distinction
hiring teams probe for. Industry data backs this: ~89% of teams run agent observability but only
~52% run evals, so doing evals well is still a differentiator.

## Foundational decision

**The eval runs against the real Groq LLM, not the `FakeLLM`.** An eval exists to measure how the
actual model behaves in the actual agent loop. A deterministic fake would measure nothing. The
consequence — runs are slower, non-deterministic, and need an API key — is accepted and is precisely
why the harness reports a *pass rate* rather than a binary green/red.

## Constraints

- **Cost $0:** Groq free tier only. No new paid services.
- **No new runtime dependencies:** scenarios are JSON (stdlib `json`), not YAML.
- **Phase 1 agent code is not modified.** The harness observes the agent from the outside.
  (The single exception is a small, additive extension of the Groq adapter's existing retry logic to
  also back off on rate limits — see "Reliability handling".)
- The existing fast offline test suite must stay fast and offline. Live eval is a separate,
  deliberately-invoked command.

## Scoring: outcome-based assertions

Runs are scored on objective, checkable facts, not on a judge's opinion of the prose. This is cheap,
deterministic, un-fudgeable, and impossible to accidentally grade on vibes. No LLM-as-judge in this
phase (explicitly cut — YAGNI).

Supported assertions in `expect` (deliberately small):

| Assertion | Meaning |
|---|---|
| `tools_called` | these tool names must all have been called (subset check, order-insensitive) |
| `tools_not_called` | none of these tool names may have been called |
| `escalated` | boolean — whether the `escalate` tool was called |
| `appointments` | expected final DB state: list of `{patient_name, slot_iso, status}` |
| `reply_contains` | optional case-insensitive substring check on the final reply |

A run PASSES only if every assertion present in `expect` holds.

## Scenario format

Scenarios live in `evals/scenarios.json` as data, not code, so they are trivial to extend.

```json
{
  "id": "cancel_by_phone",
  "category": "cancel",
  "seed": [
    {"patient_name": "Jane Doe", "phone": "5551234",
     "slot_iso": "2026-07-13T09:00", "reason": "cleaning"}
  ],
  "turns": [
    "Hi, I need to cancel my appointment",
    "Jane Doe, 5551234"
  ],
  "expect": {
    "tools_called": ["lookup_appt", "cancel"],
    "escalated": false,
    "appointments": [
      {"patient_name": "Jane Doe", "slot_iso": "2026-07-13T09:00",
       "status": "cancelled"}
    ]
  }
}
```

- **`seed`** pre-books appointments (via the existing `book` tool) so cancel/reschedule/lookup
  scenarios start from realistic state.
- **`turns`** are scripted user messages replayed in order into a single `Agent` instance, so
  conversation history accumulates exactly as in a real call. This supports both one-shot requests
  and realistic multi-turn back-and-forth (caller supplies name only after being asked).
- **`expect`** holds the assertions above.

Scenarios are fixed to the week of Monday **2026-07-13** so dates are deterministic.

## Isolation and observation

- **Every run gets a fresh in-memory SQLite database** (`sqlite3.connect(":memory:")` + `init_db`),
  so runs cannot contaminate one another. Seeds are applied per run.
- **Tool calls are read back out of `agent.messages`.** The agent already records assistant messages
  carrying `tool_calls` in OpenAI format; the harness parses those to learn which tools were called.
  This is why Phase 1 needs no changes: the harness observes without perturbing the system.

## Non-determinism: 3 runs per scenario

Each scenario runs **3 times**; the report distinguishes:

- **3/3 — reliable**
- **1/3 or 2/3 — flaky** (the most actionable signal; a single-run eval would hide this)
- **0/3 — broken**

This directly addresses the observed reality that the same prompt can succeed once and fail the next
time (Groq's `tool_use_failed` malformed-tool-call bug). Reliability becomes a measured metric rather
than an anecdote. ~30 scenarios × 3 = ~90 runs.

## Scenario coverage (~30)

| Category | ~n | Examples |
|---|---|---|
| Booking happy path | 6 | one-shot; multi-turn; time expressed loosely |
| FAQ | 4 | hours, location, insurance, services |
| Reschedule | 3 | move to a different day/time |
| Cancel | 3 | by phone, by name, no matching appointment |
| Lookup | 2 | "what do I have booked?" |
| Unavailable slot | 4 | double-book attempt, weekend, outside hours |
| **Escalation** | 5 | medical advice, gibberish, out-of-scope, unknown patient |
| Ambiguous | 3 | missing info, caller changes mind mid-call |

Escalation is deliberately over-weighted: knowing when to *stop and hand off* is the hardest and most
valuable behavior to prove, and it is the safety property of the whole system.

## Metrics reported

- **Overall pass rate** across runs and across scenarios (e.g. `84/90 runs (93.3%)`)
- **Per-scenario** result as `3/3` · `2/3` · `0/3` — separating reliable from flaky from broken
- **Per-category** breakdown, to show where the agent is weak
- **Latency** mean and p50 per run — the key metric for Phase 3 (voice needs low turn latency)
- **Failure detail** — expected vs actual for each failing run, so failures are debuggable

## Reliability handling

- Groq's free tier rate-limits, and ~300 LLM calls will hit **429s**. The Groq adapter's existing
  retry logic is extended to also back off and retry on rate-limit errors (additive change to
  `_is_tool_use_failed`'s sibling logic; the tool_use_failed behavior is unchanged).
- An `LLMError` surviving retries makes that run score **FAIL with a recorded reason** — it never
  crashes the harness. Flakiness must show up *in the metric*; that is the point of the exercise.
- CLI flags `--scenario <id>` and `--runs N` allow fast iteration on a single scenario.

## Files

```
evals/scenarios.json              # the ~30 scenarios (data, not code)
src/voicedesk/evals/__init__.py
src/voicedesk/evals/scoring.py    # pure assertion functions (unit-tested offline)
src/voicedesk/evals/runner.py     # loads scenarios, runs agent N×, collects results
src/voicedesk/evals/report.py     # console table + markdown report
reports/eval-<timestamp>.md       # the generated artifact
```

Component boundaries: `scoring.py` is pure (results in → pass/fail out) and knows nothing about the
network; `runner.py` owns orchestration and the live LLM; `report.py` only formats. Each is testable
alone.

## Testing strategy

- **`scoring.py` is unit-tested offline** against synthetic run records — no network, no API key.
- **`runner.py`'s scenario-loading, seeding, and tool-call extraction are unit-tested offline** using
  the existing `FakeLLM`, proving the harness plumbing works without touching Groq.
- The **live eval is a separate command**, run deliberately: `python -m voicedesk.evals.runner`.
  It is not part of the fast test suite, which must stay offline and fast.

## Success criteria

- `python -m voicedesk.evals.runner` produces a scored report over ~30 scenarios × 3 runs.
- The report names an overall pass rate, flags flaky scenarios, breaks results down by category, and
  reports latency.
- Failures are debuggable from the report alone (expected vs actual).
- A baseline pass rate is recorded, so a later improvement can be stated as a measured delta — the
  sentence the whole phase exists to earn.
