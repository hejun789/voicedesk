# VoiceDesk Phase 2 — Eval Harness — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an eval harness that runs ~30 scripted caller scenarios against the real Groq-powered agent, 3× each, scores them on objective outcomes, and reports a pass rate, flaky scenarios, per-category results, and latency.

**Architecture:** Scenarios are JSON data. `scoring.py` holds pure assertion functions (run record in → pass/fail out). `runner.py` orchestrates: for each run it builds a fresh in-memory DB, seeds it, drives an `Agent` through the scripted turns, and captures what happened. `report.py` formats results. The Phase 1 agent is **not modified** — the harness reads tool calls back out of `agent.messages` and wraps the LLM in an error-capturing proxy to observe failures without perturbing the system.

**Tech Stack:** Python 3.11+, stdlib `json`/`sqlite3`/`statistics`/`argparse`, existing `groq` + `python-dotenv`, `pytest`.

## Global Constraints

- **Cost $0:** Groq free tier only. No new paid services.
- **No new runtime dependencies.** Scenarios are JSON via stdlib `json` — not YAML.
- **Phase 1 agent code is NOT modified.** The only change to existing code is Task 1: an additive extension of the Groq adapter's retry logic to back off on rate limits.
- **The fast test suite stays offline and fast.** Every test in `tests/` runs with no network and no API key (use `FakeLLM` / synthetic records). The live eval is a separate, deliberately-invoked command.
- **Scenario dates are fixed to the week of Monday 2026-07-13** (Mon 07-13 … Fri 07-17) so expectations are deterministic. Clinic hours: weekdays, hourly 09:00–16:00 inclusive.
- Tests run as `PYTHONPATH=src python -m pytest ...` from the repo root (Bash/Git Bash). In PowerShell: `$env:PYTHONPATH="src"; python -m pytest`.
- **The live eval must be run from the repo root**, because `answer_faq` resolves `clinic_info.md` relative to the current working directory.
- TDD throughout; commit after each green task.

---

### Task 1: Back off and retry on Groq rate limits

The live eval makes ~300 LLM calls and will hit Groq free-tier 429s. The adapter already retries the malformed-tool-call case; extend it to also retry rate limits with exponential backoff. Everything else must still fail fast.

**Files:**
- Modify: `src/voicedesk/groq_client.py`
- Test: `tests/test_groq_client.py` (append)

**Interfaces:**
- Consumes: `LLMError`, `Message`, `ToolCall` from `voicedesk.llm`; the existing `_is_tool_use_failed(exc) -> bool`, `_to_message(choice) -> Message`, and `GroqLLM(model=None, api_key=None, client=None, max_retries=3)`.
- Produces:
  - `_is_rate_limited(exc: Exception) -> bool` — True when the exception is a rate-limit/429 error.
  - `GroqLLM.__init__` gains `backoff_base: float = 2.0` (seconds). Tests pass `backoff_base=0` so they never sleep.
  - `GroqLLM.complete` retries on rate limit with `backoff_base * (2 ** attempt)` seconds of sleep, retries on tool_use_failed with no sleep, and raises `LLMError` for everything else (and after retries are exhausted).

- [ ] **Step 1: Write the failing tests** — append to `tests/test_groq_client.py`

```python
def _rate_limit_error():
    e = Exception("429 Too Many Requests: rate limit exceeded")
    e.status_code = 429
    return e


def test_complete_retries_on_rate_limit_then_succeeds():
    good = SimpleNamespace(choices=[_fake_choice(content="ok")])
    client = _FakeGroqClient([_rate_limit_error(), good])
    llm = GroqLLM(client=client, max_retries=3, backoff_base=0)
    msg = llm.complete([], [])
    assert msg.content == "ok"
    assert client.calls == 2  # backed off and retried


def test_complete_raises_llmerror_after_persistent_rate_limit():
    client = _FakeGroqClient([_rate_limit_error() for _ in range(3)])
    llm = GroqLLM(client=client, max_retries=3, backoff_base=0)
    with pytest.raises(LLMError):
        llm.complete([], [])
    assert client.calls == 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src python -m pytest tests/test_groq_client.py -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'backoff_base'`

- [ ] **Step 3: Implement** — replace the whole of `src/voicedesk/groq_client.py` with:

```python
import json
import os
import time
from voicedesk.llm import Message, ToolCall, LLMError

DEFAULT_MODEL = "llama-3.3-70b-versatile"


def _to_message(choice) -> Message:
    m = choice.message
    calls = []
    for tc in (m.tool_calls or []):
        calls.append(ToolCall(
            id=tc.id,
            name=tc.function.name,
            arguments=json.loads(tc.function.arguments or "{}"),
        ))
    return Message(content=m.content, tool_calls=calls)


def _is_tool_use_failed(exc: Exception) -> bool:
    """True when Groq rejected a malformed tool call (code 'tool_use_failed').
    Detected without importing groq's exception types, so this stays testable
    and provider-agnostic. This failure is non-deterministic — a resample of
    the same request usually produces a valid structured tool call."""
    if getattr(exc, "code", None) == "tool_use_failed":
        return True
    return "tool_use_failed" in str(exc)


def _is_rate_limited(exc: Exception) -> bool:
    """True when Groq rejected the request for exceeding the free-tier rate
    limit. Worth waiting out — unlike auth or bad-request errors."""
    if getattr(exc, "status_code", None) == 429:
        return True
    if getattr(exc, "code", None) == "rate_limit_exceeded":
        return True
    text = str(exc).lower()
    return "rate limit" in text or "429" in text


class GroqLLM:
    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        client=None,
        max_retries: int = 3,
        backoff_base: float = 2.0,
    ):
        # Model is configurable via GROQ_MODEL so a different model can be tried
        # without code changes if tool-calling reliability is poor.
        self.model = model or os.environ.get("GROQ_MODEL", DEFAULT_MODEL)
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        if client is not None:
            self.client = client  # injected (used by tests — no network/key)
        else:
            from groq import Groq  # imported lazily so tests don't need the package
            self.client = Groq(api_key=api_key or os.environ["GROQ_API_KEY"])

    def complete(self, messages: list[dict], tools: list[dict]) -> Message:
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                )
                return _to_message(resp.choices[0])
            except Exception as e:  # noqa: BLE001 - translated to LLMError below
                last_exc = e
                if attempt < self.max_retries - 1:
                    # Rate limits are worth waiting out.
                    if _is_rate_limited(e):
                        time.sleep(self.backoff_base * (2 ** attempt))
                        continue
                    # Malformed tool calls are non-deterministic; resample at once.
                    if _is_tool_use_failed(e):
                        continue
                # Everything else (auth, bad request, ...) fails fast.
                raise LLMError(str(e)) from e
        raise LLMError(str(last_exc)) from last_exc
```

- [ ] **Step 4: Run the whole file's tests to verify they pass**

Run: `PYTHONPATH=src python -m pytest tests/test_groq_client.py -v`
Expected: PASS (7 passed — 2 `_to_message`, 3 pre-existing retry tests, 2 new rate-limit tests)

- [ ] **Step 5: Commit**

```bash
git add src/voicedesk/groq_client.py tests/test_groq_client.py
git commit -m "feat: back off and retry on Groq rate limits"
```

---

### Task 2: Scoring — run records and outcome assertions

**Files:**
- Create: `src/voicedesk/evals/__init__.py` (empty)
- Create: `src/voicedesk/evals/scoring.py`
- Test: `tests/test_evals_scoring.py`

**Interfaces:**
- Produces:
  - `@dataclass RunRecord(scenario_id: str, category: str, tools_called: list[str], escalated: bool, appointments: list[dict], final_reply: str, latency_s: float, error: str | None = None)` — everything observed during one run. Each dict in `appointments` has keys `patient_name`, `phone`, `slot_iso`, `reason`, `status`.
  - `@dataclass RunResult(record: RunRecord, passed: bool, failures: list[str])`
  - `score_run(record: RunRecord, expect: dict) -> RunResult` — applies the assertions present in `expect`. A run with `record.error` set fails immediately. Supported keys: `tools_called` (all must appear, order-insensitive), `tools_not_called` (none may appear), `escalated` (bool equality), `appointments` (each expected partial dict must match some actual row), `appointment_count` (exact row count), `reply_contains` (case-insensitive substring).

- [ ] **Step 1: Write the failing test** — `tests/test_evals_scoring.py`

```python
from voicedesk.evals.scoring import RunRecord, score_run


def _record(**over):
    base = dict(
        scenario_id="s1", category="booking", tools_called=["book"],
        escalated=False,
        appointments=[{"patient_name": "Jane Doe", "phone": "5551234",
                       "slot_iso": "2026-07-13T09:00", "reason": "cleaning",
                       "status": "booked"}],
        final_reply="You're booked for Monday at 9am.", latency_s=1.0, error=None,
    )
    base.update(over)
    return RunRecord(**base)


def test_passes_when_all_assertions_hold():
    res = score_run(_record(), {
        "tools_called": ["book"],
        "escalated": False,
        "appointments": [{"patient_name": "Jane Doe",
                          "slot_iso": "2026-07-13T09:00", "status": "booked"}],
        "appointment_count": 1,
        "reply_contains": "booked",
    })
    assert res.passed is True
    assert res.failures == []


def test_fails_on_missing_tool():
    res = score_run(_record(tools_called=["find_slots"]), {"tools_called": ["book"]})
    assert res.passed is False
    assert "book" in res.failures[0]


def test_fails_on_forbidden_tool():
    res = score_run(_record(tools_called=["book"]), {"tools_not_called": ["book"]})
    assert res.passed is False


def test_fails_on_escalation_mismatch():
    res = score_run(_record(escalated=False), {"escalated": True})
    assert res.passed is False


def test_fails_on_appointment_mismatch():
    res = score_run(_record(), {
        "appointments": [{"patient_name": "Jane Doe",
                          "slot_iso": "2026-07-13T11:00", "status": "booked"}]
    })
    assert res.passed is False


def test_fails_on_appointment_count():
    res = score_run(_record(), {"appointment_count": 0})
    assert res.passed is False


def test_fails_on_reply_contains():
    res = score_run(_record(), {"reply_contains": "cancelled"})
    assert res.passed is False


def test_llm_error_fails_immediately():
    res = score_run(_record(error="429 rate limit"), {"tools_called": ["book"]})
    assert res.passed is False
    assert res.failures == ["llm_error: 429 rate limit"]


def test_empty_expect_passes():
    assert score_run(_record(), {}).passed is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m pytest tests/test_evals_scoring.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'voicedesk.evals'`

- [ ] **Step 3: Create the package** — `src/voicedesk/evals/__init__.py` as an empty file.

- [ ] **Step 4: Implement** — `src/voicedesk/evals/scoring.py`

```python
from dataclasses import dataclass, field


@dataclass
class RunRecord:
    """Everything observed during a single run of one scenario."""
    scenario_id: str
    category: str
    tools_called: list[str]
    escalated: bool
    appointments: list[dict]
    final_reply: str
    latency_s: float
    error: str | None = None


@dataclass
class RunResult:
    record: RunRecord
    passed: bool
    failures: list[str] = field(default_factory=list)


def _matches(expected: dict, actual: dict) -> bool:
    """Partial match: every key in `expected` equals the same key in `actual`."""
    return all(actual.get(k) == v for k, v in expected.items())


def score_run(record: RunRecord, expect: dict) -> RunResult:
    if record.error:
        return RunResult(record=record, passed=False,
                         failures=[f"llm_error: {record.error}"])

    failures: list[str] = []
    called = set(record.tools_called)

    for name in expect.get("tools_called", []):
        if name not in called:
            failures.append(
                f"expected tool {name!r} to be called; called={sorted(called)}")

    for name in expect.get("tools_not_called", []):
        if name in called:
            failures.append(f"tool {name!r} should NOT have been called")

    if "escalated" in expect and record.escalated != expect["escalated"]:
        failures.append(
            f"expected escalated={expect['escalated']}, got {record.escalated}")

    for exp in expect.get("appointments", []):
        if not any(_matches(exp, a) for a in record.appointments):
            failures.append(
                f"no appointment matching {exp}; actual={record.appointments}")

    if "appointment_count" in expect:
        actual_n = len(record.appointments)
        if actual_n != expect["appointment_count"]:
            failures.append(
                f"expected {expect['appointment_count']} appointment(s), got {actual_n}")

    if "reply_contains" in expect:
        needle = expect["reply_contains"]
        if needle.lower() not in record.final_reply.lower():
            failures.append(
                f"reply did not contain {needle!r}; reply={record.final_reply!r}")

    return RunResult(record=record, passed=not failures, failures=failures)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `PYTHONPATH=src python -m pytest tests/test_evals_scoring.py -v`
Expected: PASS (9 passed)

- [ ] **Step 6: Commit**

```bash
git add src/voicedesk/evals tests/test_evals_scoring.py
git commit -m "feat: eval scoring with outcome assertions"
```

---

### Task 3: Runner helpers — scenario loading, DB isolation, seeding, tool-call extraction

**Files:**
- Create: `src/voicedesk/evals/runner.py`
- Test: `tests/test_evals_runner_helpers.py`

**Interfaces:**
- Consumes: `init_db(conn)` from `voicedesk.db`; `book(conn, patient_name, phone, slot_iso, reason) -> dict` from `voicedesk.tools`.
- Produces:
  - `load_scenarios(path: str = "evals/scenarios.json") -> list[dict]`
  - `tools_called_from(messages: list[dict]) -> list[str]` — pulls tool names out of the agent's OpenAI-format assistant messages (each has `tool_calls: [{"function": {"name": ...}}]`). Returns names in call order, including duplicates.
  - `fresh_db() -> sqlite3.Connection` — in-memory, `row_factory = sqlite3.Row`, schema applied.
  - `seed_db(conn, seed: list[dict]) -> None` — books each seed appointment via `tools.book`; raises `ValueError` if a seed booking fails (a bad seed must be loud, not silently skew results).
  - `all_appointments(conn) -> list[dict]` — every row as `{patient_name, phone, slot_iso, reason, status}`, ordered by `slot_iso`.

- [ ] **Step 1: Write the failing test** — `tests/test_evals_runner_helpers.py`

```python
import json
import pytest
from voicedesk.evals.runner import (
    load_scenarios, tools_called_from, fresh_db, seed_db, all_appointments,
)


def test_tools_called_from_extracts_names_in_order():
    messages = [
        {"role": "system", "content": "x"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "1", "type": "function",
             "function": {"name": "find_slots", "arguments": "{}"}},
            {"id": "2", "type": "function",
             "function": {"name": "book", "arguments": "{}"}},
        ]},
        {"role": "tool", "tool_call_id": "1", "content": "{}"},
        {"role": "assistant", "content": "done"},
    ]
    assert tools_called_from(messages) == ["find_slots", "book"]


def test_tools_called_from_empty_when_no_tool_calls():
    assert tools_called_from([{"role": "assistant", "content": "hello"}]) == []


def test_fresh_db_is_isolated():
    a, b = fresh_db(), fresh_db()
    seed_db(a, [{"patient_name": "Jane Doe", "phone": "5551234",
                 "slot_iso": "2026-07-13T09:00", "reason": "cleaning"}])
    assert len(all_appointments(a)) == 1
    assert all_appointments(b) == []  # separate DB, unaffected


def test_seed_db_then_all_appointments():
    conn = fresh_db()
    seed_db(conn, [{"patient_name": "Jane Doe", "phone": "5551234",
                    "slot_iso": "2026-07-13T09:00", "reason": "cleaning"}])
    appts = all_appointments(conn)
    assert appts == [{"patient_name": "Jane Doe", "phone": "5551234",
                      "slot_iso": "2026-07-13T09:00", "reason": "cleaning",
                      "status": "booked"}]


def test_seed_db_raises_on_impossible_slot():
    conn = fresh_db()
    with pytest.raises(ValueError):
        seed_db(conn, [{"patient_name": "X", "phone": "1",
                        "slot_iso": "2026-07-11T09:00",  # a Saturday
                        "reason": "cleaning"}])


def test_load_scenarios(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps([{"id": "a", "turns": ["hi"]}]), encoding="utf-8")
    assert load_scenarios(str(p))[0]["id"] == "a"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m pytest tests/test_evals_runner_helpers.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'voicedesk.evals.runner'`

- [ ] **Step 3: Implement** — `src/voicedesk/evals/runner.py`

```python
import json
import sqlite3

from voicedesk import tools
from voicedesk.db import init_db


def load_scenarios(path: str = "evals/scenarios.json") -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def tools_called_from(messages: list[dict]) -> list[str]:
    """Read which tools the agent called back out of its own message history.
    This is how the harness observes the agent without modifying it."""
    names: list[str] = []
    for m in messages:
        for tc in m.get("tool_calls") or []:
            names.append(tc["function"]["name"])
    return names


def fresh_db() -> sqlite3.Connection:
    """A new in-memory DB per run, so runs cannot contaminate each other."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    init_db(conn)
    return conn


def seed_db(conn: sqlite3.Connection, seed: list[dict]) -> None:
    for a in seed:
        res = tools.book(conn, a["patient_name"], a["phone"],
                         a["slot_iso"], a.get("reason", ""))
        if not res.get("ok"):
            raise ValueError(f"seed booking failed for {a}: {res}")


def all_appointments(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        "SELECT patient_name, phone, slot_iso, reason, status "
        "FROM appointments ORDER BY slot_iso"
    )
    return [
        {"patient_name": r[0], "phone": r[1], "slot_iso": r[2],
         "reason": r[3], "status": r[4]}
        for r in rows
    ]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src python -m pytest tests/test_evals_runner_helpers.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add src/voicedesk/evals/runner.py tests/test_evals_runner_helpers.py
git commit -m "feat: eval runner helpers (loading, isolation, seeding, observation)"
```

---

### Task 4: Runner orchestration — drive the agent through a scenario, N times

**Files:**
- Modify: `src/voicedesk/evals/runner.py` (append)
- Test: `tests/test_evals_runner.py`

**Interfaces:**
- Consumes: `Agent(conn, llm)` and `agent.respond(text) -> str` from `voicedesk.agent`; `LLMError` from `voicedesk.llm`; `RunRecord`, `RunResult`, `score_run` from `voicedesk.evals.scoring`; the Task 3 helpers.
- Produces:
  - `_ErrorCapturingLLM(inner)` — an `LLMClient` proxy that records the message of any `LLMError` raised by `inner.complete` (in `.error`) and re-raises. This is how the harness sees LLM failures even though `Agent` deliberately swallows them into its escalation fallback.
  - `run_scenario_once(scenario: dict, llm) -> RunRecord`
  - `run_scenario(scenario: dict, llm_factory, runs: int = 3) -> list[RunResult]` — `llm_factory` is a zero-arg callable returning a fresh `LLMClient` per run (needed so a scripted `FakeLLM` is fresh each run).
  - `run_all(scenarios: list[dict], llm_factory, runs: int = 3) -> list[RunResult]`

- [ ] **Step 1: Write the failing test** — `tests/test_evals_runner.py`

```python
from voicedesk.llm import FakeLLM, Message, ToolCall, LLMError
from voicedesk.evals.runner import run_scenario_once, run_scenario, run_all

BOOK_SCENARIO = {
    "id": "book_oneshot",
    "category": "booking",
    "turns": ["Book Monday July 13th 9am, Jane Doe, 5551234, cleaning"],
    "expect": {
        "tools_called": ["book"],
        "escalated": False,
        "appointments": [{"patient_name": "Jane Doe",
                          "slot_iso": "2026-07-13T09:00", "status": "booked"}],
    },
}


def _booking_llm():
    return FakeLLM([
        Message(content=None, tool_calls=[
            ToolCall(id="1", name="book", arguments={
                "patient_name": "Jane Doe", "phone": "5551234",
                "slot_iso": "2026-07-13T09:00", "reason": "cleaning"})]),
        Message(content="You're booked for Monday at 9am.", tool_calls=[]),
    ])


def test_run_scenario_once_records_what_happened():
    rec = run_scenario_once(BOOK_SCENARIO, _booking_llm())
    assert rec.scenario_id == "book_oneshot"
    assert rec.category == "booking"
    assert rec.tools_called == ["book"]
    assert rec.escalated is False
    assert rec.appointments[0]["slot_iso"] == "2026-07-13T09:00"
    assert "booked" in rec.final_reply.lower()
    assert rec.latency_s >= 0
    assert rec.error is None


def test_run_scenario_once_applies_seed():
    scenario = {
        "id": "cancel", "category": "cancel",
        "seed": [{"patient_name": "Jane Doe", "phone": "5551234",
                  "slot_iso": "2026-07-13T09:00", "reason": "cleaning"}],
        "turns": ["cancel it"],
        "expect": {},
    }
    llm = FakeLLM([Message(content="Which appointment?", tool_calls=[])])
    rec = run_scenario_once(scenario, llm)
    assert rec.appointments[0]["patient_name"] == "Jane Doe"


def test_run_scenario_once_captures_llm_error():
    class _Raising:
        def complete(self, messages, tools):
            raise LLMError("429 rate limit")

    rec = run_scenario_once(BOOK_SCENARIO, _Raising())
    assert rec.error == "429 rate limit"


def test_run_scenario_scores_each_run():
    results = run_scenario(BOOK_SCENARIO, _booking_llm, runs=3)
    assert len(results) == 3
    assert all(r.passed for r in results)


def test_run_scenario_detects_escalation():
    scenario = {
        "id": "esc", "category": "escalation",
        "turns": ["asdkjh qwe zxcv"],
        "expect": {"escalated": True},
    }

    def _llm():
        return FakeLLM([
            Message(content=None, tool_calls=[
                ToolCall(id="1", name="escalate",
                         arguments={"reason": "unintelligible"})]),
            Message(content="Let me have someone call you back.", tool_calls=[]),
        ])

    results = run_scenario(scenario, _llm, runs=1)
    assert results[0].passed is True
    assert results[0].record.escalated is True


def test_run_all_flattens_results():
    results = run_all([BOOK_SCENARIO, BOOK_SCENARIO], _booking_llm, runs=2)
    assert len(results) == 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m pytest tests/test_evals_runner.py -v`
Expected: FAIL — `ImportError: cannot import name 'run_scenario_once'`

- [ ] **Step 3: Implement** — append to `src/voicedesk/evals/runner.py`

Add these imports at the top of the file (alongside the existing ones):

```python
import time

from voicedesk.agent import Agent
from voicedesk.llm import LLMError
from voicedesk.evals.scoring import RunRecord, RunResult, score_run
```

Then append:

```python
ESCALATE_TOOL = "escalate"


class _ErrorCapturingLLM:
    """Wraps an LLMClient to record LLMErrors. The Agent deliberately swallows
    them into its escalation fallback, so without this the harness could not
    tell a genuine escalation apart from an API failure."""

    def __init__(self, inner):
        self.inner = inner
        self.error: str | None = None

    def complete(self, messages: list[dict], tools: list[dict]):
        try:
            return self.inner.complete(messages, tools)
        except LLMError as e:
            self.error = str(e)
            raise


def run_scenario_once(scenario: dict, llm) -> RunRecord:
    conn = fresh_db()
    seed_db(conn, scenario.get("seed", []))
    capturing = _ErrorCapturingLLM(llm)
    agent = Agent(conn, capturing)

    final_reply = ""
    start = time.perf_counter()
    for turn in scenario["turns"]:
        final_reply = agent.respond(turn)  # never raises; Agent degrades on LLMError
    latency_s = time.perf_counter() - start

    called = tools_called_from(agent.messages)
    return RunRecord(
        scenario_id=scenario["id"],
        category=scenario.get("category", ""),
        tools_called=called,
        escalated=ESCALATE_TOOL in called,
        appointments=all_appointments(conn),
        final_reply=final_reply,
        latency_s=latency_s,
        error=capturing.error,
    )


def run_scenario(scenario: dict, llm_factory, runs: int = 3) -> list[RunResult]:
    expect = scenario.get("expect", {})
    return [
        score_run(run_scenario_once(scenario, llm_factory()), expect)
        for _ in range(runs)
    ]


def run_all(scenarios: list[dict], llm_factory, runs: int = 3) -> list[RunResult]:
    results: list[RunResult] = []
    for scenario in scenarios:
        results.extend(run_scenario(scenario, llm_factory, runs=runs))
    return results
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src python -m pytest tests/test_evals_runner.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add src/voicedesk/evals/runner.py tests/test_evals_runner.py
git commit -m "feat: eval runner orchestration with error capture"
```

---

### Task 5: Report — summarize, console table, markdown artifact

**Files:**
- Create: `src/voicedesk/evals/report.py`
- Test: `tests/test_evals_report.py`

**Interfaces:**
- Consumes: `RunRecord`, `RunResult` from `voicedesk.evals.scoring`.
- Produces:
  - `summarize(results: list[RunResult]) -> dict` with keys: `total_runs`, `passed_runs`, `pass_rate` (0.0–1.0), `per_scenario` (`{id: {"passed": int, "total": int, "category": str}}`), `per_category` (`{cat: {"passed": int, "total": int}}`), `latency_mean`, `latency_p50`.
  - `status_of(passed: int, total: int) -> str` — `"PASS"` when all passed, `"FAIL"` when none passed, else `"FLAKY"`.
  - `format_console(results: list[RunResult]) -> str`
  - `format_markdown(results: list[RunResult]) -> str`

- [ ] **Step 1: Write the failing test** — `tests/test_evals_report.py`

```python
from voicedesk.evals.scoring import RunRecord, RunResult
from voicedesk.evals.report import (
    summarize, status_of, format_console, format_markdown,
)


def _result(sid, category, passed, latency=1.0, failures=None):
    rec = RunRecord(scenario_id=sid, category=category, tools_called=[],
                    escalated=False, appointments=[], final_reply="",
                    latency_s=latency)
    return RunResult(record=rec, passed=passed,
                     failures=failures or ([] if passed else ["boom"]))


def test_status_of():
    assert status_of(3, 3) == "PASS"
    assert status_of(0, 3) == "FAIL"
    assert status_of(2, 3) == "FLAKY"


def test_summarize_counts_and_rates():
    results = [
        _result("a", "booking", True, latency=1.0),
        _result("a", "booking", True, latency=3.0),
        _result("b", "escalation", False, latency=2.0),
    ]
    s = summarize(results)
    assert s["total_runs"] == 3
    assert s["passed_runs"] == 2
    assert round(s["pass_rate"], 3) == 0.667
    assert s["per_scenario"]["a"] == {"passed": 2, "total": 2, "category": "booking"}
    assert s["per_category"]["escalation"] == {"passed": 0, "total": 1}
    assert s["latency_mean"] == 2.0
    assert s["latency_p50"] == 2.0


def test_summarize_handles_empty():
    s = summarize([])
    assert s["total_runs"] == 0
    assert s["pass_rate"] == 0.0


def test_format_console_reports_rate_and_flaky():
    results = [
        _result("a", "booking", True),
        _result("a", "booking", False),
        _result("b", "faq", True),
    ]
    out = format_console(results)
    assert "2/3" in out          # overall passed runs
    assert "FLAKY" in out        # scenario a is 1/2
    assert "booking" in out      # category breakdown
    assert "boom" in out         # failure detail is shown


def test_format_markdown_is_markdown():
    out = format_markdown([_result("a", "booking", True)])
    assert out.startswith("# VoiceDesk Eval Report")
    assert "|" in out            # contains a markdown table
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m pytest tests/test_evals_report.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'voicedesk.evals.report'`

- [ ] **Step 3: Implement** — `src/voicedesk/evals/report.py`

```python
from datetime import datetime
from statistics import mean, median

from voicedesk.evals.scoring import RunResult


def status_of(passed: int, total: int) -> str:
    if passed == total:
        return "PASS"
    if passed == 0:
        return "FAIL"
    return "FLAKY"


def summarize(results: list[RunResult]) -> dict:
    total = len(results)
    passed = sum(1 for r in results if r.passed)

    per_scenario: dict[str, dict] = {}
    per_category: dict[str, dict] = {}
    for r in results:
        s = per_scenario.setdefault(
            r.record.scenario_id,
            {"passed": 0, "total": 0, "category": r.record.category},
        )
        s["total"] += 1
        s["passed"] += int(r.passed)

        c = per_category.setdefault(r.record.category, {"passed": 0, "total": 0})
        c["total"] += 1
        c["passed"] += int(r.passed)

    latencies = [r.record.latency_s for r in results]
    return {
        "total_runs": total,
        "passed_runs": passed,
        "pass_rate": (passed / total) if total else 0.0,
        "per_scenario": per_scenario,
        "per_category": per_category,
        "latency_mean": mean(latencies) if latencies else 0.0,
        "latency_p50": median(latencies) if latencies else 0.0,
    }


def _failure_lines(results: list[RunResult]) -> list[str]:
    lines = []
    for r in results:
        if not r.passed:
            for f in r.failures:
                lines.append(f"[{r.record.scenario_id}] {f}")
    return lines


def format_console(results: list[RunResult]) -> str:
    s = summarize(results)
    out = [
        "VoiceDesk Eval Report",
        "=====================",
        f"Overall: {s['passed_runs']}/{s['total_runs']} runs "
        f"({s['pass_rate'] * 100:.1f}%)",
        f"Latency: mean {s['latency_mean']:.2f}s, p50 {s['latency_p50']:.2f}s",
        "",
        f"{'SCENARIO':<28}{'RUNS':<8}{'STATUS'}",
    ]
    for sid, v in s["per_scenario"].items():
        runs = f"{v['passed']}/{v['total']}"
        out.append(f"{sid:<28}{runs:<8}{status_of(v['passed'], v['total'])}")

    out += ["", "BY CATEGORY"]
    for cat, v in s["per_category"].items():
        rate = (v["passed"] / v["total"] * 100) if v["total"] else 0.0
        out.append(f"{cat:<28}{v['passed']}/{v['total']} ({rate:.1f}%)")

    failures = _failure_lines(results)
    if failures:
        out += ["", "FAILURES"] + failures
    return "\n".join(out)


def format_markdown(results: list[RunResult]) -> str:
    s = summarize(results)
    out = [
        "# VoiceDesk Eval Report",
        "",
        f"_Generated {datetime.now():%Y-%m-%d %H:%M:%S}_",
        "",
        f"**Overall: {s['passed_runs']}/{s['total_runs']} runs "
        f"({s['pass_rate'] * 100:.1f}%)**",
        "",
        f"Latency: mean {s['latency_mean']:.2f}s · p50 {s['latency_p50']:.2f}s",
        "",
        "## Scenarios",
        "",
        "| Scenario | Category | Runs | Status |",
        "|---|---|---|---|",
    ]
    for sid, v in s["per_scenario"].items():
        out.append(
            f"| {sid} | {v['category']} | {v['passed']}/{v['total']} "
            f"| {status_of(v['passed'], v['total'])} |"
        )

    out += ["", "## By category", "", "| Category | Passed | Rate |", "|---|---|---|"]
    for cat, v in s["per_category"].items():
        rate = (v["passed"] / v["total"] * 100) if v["total"] else 0.0
        out.append(f"| {cat} | {v['passed']}/{v['total']} | {rate:.1f}% |")

    failures = _failure_lines(results)
    if failures:
        out += ["", "## Failures", ""]
        out += [f"- {line}" for line in failures]
    return "\n".join(out) + "\n"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src python -m pytest tests/test_evals_report.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add src/voicedesk/evals/report.py tests/test_evals_report.py
git commit -m "feat: eval report (summary, console table, markdown)"
```

---

### Task 6: The scenario suite (~30 scenarios)

Pure data. All dates are in the week of Monday **2026-07-13** (Mon 07-13, Tue 07-14, Wed 07-15, Thu 07-16, Fri 07-17). Saturday is 07-18, Sunday 07-19. Clinic hours: weekdays, hourly 09:00–16:00.

**Files:**
- Create: `evals/scenarios.json`
- Test: `tests/test_scenarios_file.py`

**Interfaces:**
- Consumes: `load_scenarios` from `voicedesk.evals.runner`.
- Produces: `evals/scenarios.json` — a JSON array of scenario objects, each with `id`, `category`, optional `seed`, `turns`, `expect`.

- [ ] **Step 1: Write the failing test** — `tests/test_scenarios_file.py`

This test guards the data file's integrity (valid ids, known categories, known assertion keys, seeds that are actually bookable). It does NOT call any LLM.

```python
import pytest
from voicedesk.evals.runner import load_scenarios, fresh_db, seed_db

VALID_EXPECT_KEYS = {
    "tools_called", "tools_not_called", "escalated",
    "appointments", "appointment_count", "reply_contains",
}
KNOWN_TOOLS = {
    "find_slots", "book", "reschedule", "cancel",
    "lookup_appt", "answer_faq", "escalate",
}


@pytest.fixture(scope="module")
def scenarios():
    return load_scenarios("evals/scenarios.json")


def test_has_about_thirty_scenarios(scenarios):
    assert len(scenarios) >= 30


def test_ids_are_unique(scenarios):
    ids = [s["id"] for s in scenarios]
    assert len(ids) == len(set(ids))


def test_every_scenario_is_well_formed(scenarios):
    for s in scenarios:
        assert s["id"] and s["category"]
        assert s["turns"] and all(isinstance(t, str) for t in s["turns"])
        assert set(s["expect"]).issubset(VALID_EXPECT_KEYS), s["id"]
        for key in ("tools_called", "tools_not_called"):
            assert set(s["expect"].get(key, [])).issubset(KNOWN_TOOLS), s["id"]


def test_every_seed_is_bookable(scenarios):
    # A seed that cannot be booked would silently skew results.
    for s in scenarios:
        if s.get("seed"):
            seed_db(fresh_db(), s["seed"])  # raises ValueError if unbookable


def test_escalation_category_is_well_represented(scenarios):
    escalation = [s for s in scenarios if s["category"] == "escalation"]
    assert len(escalation) >= 5
    assert all(s["expect"].get("escalated") is True for s in escalation)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m pytest tests/test_scenarios_file.py -v`
Expected: FAIL — `FileNotFoundError: evals/scenarios.json`

- [ ] **Step 3: Create `evals/scenarios.json`** with exactly this content

```json
[
  {
    "id": "book_oneshot",
    "category": "booking",
    "turns": ["Book me Monday July 13th at 9am. Jane Doe, 5551234, for a cleaning."],
    "expect": {
      "tools_called": ["book"],
      "escalated": false,
      "appointments": [{"patient_name": "Jane Doe", "slot_iso": "2026-07-13T09:00", "status": "booked"}]
    }
  },
  {
    "id": "book_multi_turn",
    "category": "booking",
    "turns": [
      "Hi, I'd like to make an appointment.",
      "Monday July 13th at 10am works.",
      "John Smith, 5559876.",
      "It's for a filling."
    ],
    "expect": {
      "tools_called": ["book"],
      "escalated": false,
      "appointments": [{"patient_name": "John Smith", "slot_iso": "2026-07-13T10:00", "status": "booked"}]
    }
  },
  {
    "id": "book_after_checking_availability",
    "category": "booking",
    "turns": [
      "What times do you have available on Tuesday July 14th?",
      "Book the 2pm one. Mary Lee, 5552222, for a checkup."
    ],
    "expect": {
      "tools_called": ["find_slots", "book"],
      "escalated": false,
      "appointments": [{"patient_name": "Mary Lee", "slot_iso": "2026-07-14T14:00", "status": "booked"}]
    }
  },
  {
    "id": "book_afternoon_slot",
    "category": "booking",
    "turns": ["I need an appointment Wednesday July 15th at 3pm. Tom Ray, 5553333, for a crown."],
    "expect": {
      "tools_called": ["book"],
      "escalated": false,
      "appointments": [{"patient_name": "Tom Ray", "slot_iso": "2026-07-15T15:00", "status": "booked"}]
    }
  },
  {
    "id": "book_earliest_available",
    "category": "booking",
    "turns": ["Book me the earliest slot you have on Thursday July 16th. Anna Kim, 5554444, cleaning."],
    "expect": {
      "tools_called": ["book"],
      "escalated": false,
      "appointments": [{"patient_name": "Anna Kim", "slot_iso": "2026-07-16T09:00", "status": "booked"}]
    }
  },
  {
    "id": "book_last_slot_of_day",
    "category": "booking",
    "turns": ["Friday July 17th at 4pm please. Sam Poe, 5555555, teeth whitening."],
    "expect": {
      "tools_called": ["book"],
      "escalated": false,
      "appointments": [{"patient_name": "Sam Poe", "slot_iso": "2026-07-17T16:00", "status": "booked"}]
    }
  },

  {
    "id": "faq_hours",
    "category": "faq",
    "turns": ["What are your opening hours?"],
    "expect": {
      "tools_called": ["answer_faq"],
      "escalated": false,
      "appointment_count": 0,
      "reply_contains": "Friday"
    }
  },
  {
    "id": "faq_location",
    "category": "faq",
    "turns": ["Where are you located?"],
    "expect": {
      "tools_called": ["answer_faq"],
      "escalated": false,
      "reply_contains": "Market Street"
    }
  },
  {
    "id": "faq_insurance",
    "category": "faq",
    "turns": ["Do you accept Cigna insurance?"],
    "expect": {
      "tools_called": ["answer_faq"],
      "escalated": false,
      "reply_contains": "Cigna"
    }
  },
  {
    "id": "faq_services",
    "category": "faq",
    "turns": ["Do you do teeth whitening?"],
    "expect": {
      "tools_called": ["answer_faq"],
      "escalated": false,
      "reply_contains": "whitening"
    }
  },

  {
    "id": "reschedule_same_day",
    "category": "reschedule",
    "seed": [{"patient_name": "Jane Doe", "phone": "5551234", "slot_iso": "2026-07-13T09:00", "reason": "cleaning"}],
    "turns": [
      "I need to move my appointment.",
      "Jane Doe, 5551234.",
      "Can we do 11am the same day instead?"
    ],
    "expect": {
      "tools_called": ["lookup_appt", "reschedule"],
      "escalated": false,
      "appointments": [{"patient_name": "Jane Doe", "slot_iso": "2026-07-13T11:00", "status": "booked"}]
    }
  },
  {
    "id": "reschedule_other_day",
    "category": "reschedule",
    "seed": [{"patient_name": "John Smith", "phone": "5559876", "slot_iso": "2026-07-13T10:00", "reason": "filling"}],
    "turns": ["Move my appointment to Tuesday July 14th at 9am. John Smith, 5559876."],
    "expect": {
      "tools_called": ["lookup_appt", "reschedule"],
      "escalated": false,
      "appointments": [{"patient_name": "John Smith", "slot_iso": "2026-07-14T09:00", "status": "booked"}]
    }
  },
  {
    "id": "reschedule_no_appointment",
    "category": "reschedule",
    "turns": ["I want to reschedule my appointment. Bob Nobody, 5550000."],
    "expect": {
      "tools_called": ["lookup_appt"],
      "tools_not_called": ["reschedule", "book"],
      "appointment_count": 0
    }
  },

  {
    "id": "cancel_by_phone",
    "category": "cancel",
    "seed": [{"patient_name": "Jane Doe", "phone": "5551234", "slot_iso": "2026-07-13T09:00", "reason": "cleaning"}],
    "turns": ["I need to cancel my appointment.", "Jane Doe, 5551234."],
    "expect": {
      "tools_called": ["lookup_appt", "cancel"],
      "escalated": false,
      "appointments": [{"patient_name": "Jane Doe", "slot_iso": "2026-07-13T09:00", "status": "cancelled"}]
    }
  },
  {
    "id": "cancel_by_name",
    "category": "cancel",
    "seed": [{"patient_name": "Mary Lee", "phone": "5552222", "slot_iso": "2026-07-14T14:00", "reason": "checkup"}],
    "turns": ["Please cancel the appointment for Mary Lee."],
    "expect": {
      "tools_called": ["lookup_appt", "cancel"],
      "escalated": false,
      "appointments": [{"patient_name": "Mary Lee", "slot_iso": "2026-07-14T14:00", "status": "cancelled"}]
    }
  },
  {
    "id": "cancel_no_appointment",
    "category": "cancel",
    "turns": ["Cancel my appointment. Ghost Person, 5559999."],
    "expect": {
      "tools_called": ["lookup_appt"],
      "tools_not_called": ["cancel"],
      "appointment_count": 0
    }
  },

  {
    "id": "lookup_by_phone",
    "category": "lookup",
    "seed": [{"patient_name": "Jane Doe", "phone": "5551234", "slot_iso": "2026-07-13T09:00", "reason": "cleaning"}],
    "turns": ["What appointments do I have? My number is 5551234."],
    "expect": {
      "tools_called": ["lookup_appt"],
      "tools_not_called": ["book", "cancel", "reschedule"],
      "escalated": false,
      "appointments": [{"patient_name": "Jane Doe", "slot_iso": "2026-07-13T09:00", "status": "booked"}]
    }
  },
  {
    "id": "lookup_by_name",
    "category": "lookup",
    "seed": [{"patient_name": "Tom Ray", "phone": "5553333", "slot_iso": "2026-07-15T15:00", "reason": "crown"}],
    "turns": ["Do I have anything booked? It's Tom Ray."],
    "expect": {
      "tools_called": ["lookup_appt"],
      "tools_not_called": ["book", "cancel", "reschedule"],
      "escalated": false
    }
  },

  {
    "id": "book_taken_slot",
    "category": "unavailable",
    "seed": [{"patient_name": "Jane Doe", "phone": "5551234", "slot_iso": "2026-07-13T09:00", "reason": "cleaning"}],
    "turns": ["Book me Monday July 13th at 9am. Carl New, 5556666, for a cleaning."],
    "expect": {
      "escalated": false,
      "appointments": [{"patient_name": "Jane Doe", "slot_iso": "2026-07-13T09:00", "status": "booked"}]
    }
  },
  {
    "id": "book_saturday_rejected",
    "category": "unavailable",
    "turns": ["Book me Saturday July 18th at 10am. Rita Sun, 5557777, for a cleaning."],
    "expect": {
      "appointment_count": 0
    }
  },
  {
    "id": "book_outside_hours_rejected",
    "category": "unavailable",
    "turns": ["Book me Monday July 13th at 8pm. Neil Late, 5558888, for a cleaning."],
    "expect": {
      "appointment_count": 0
    }
  },
  {
    "id": "book_sunday_rejected",
    "category": "unavailable",
    "turns": ["Can I come in on Sunday July 19th at noon? Paul Rest, 5551111, checkup."],
    "expect": {
      "appointment_count": 0
    }
  },

  {
    "id": "escalate_medical_advice",
    "category": "escalation",
    "turns": ["My tooth is bleeding a lot. What medication should I take right now?"],
    "expect": {
      "tools_called": ["escalate"],
      "escalated": true,
      "appointment_count": 0
    }
  },
  {
    "id": "escalate_gibberish",
    "category": "escalation",
    "turns": ["asdkjh qwe zxcv 12345 ??? mnbv"],
    "expect": {
      "tools_called": ["escalate"],
      "escalated": true,
      "appointment_count": 0
    }
  },
  {
    "id": "escalate_out_of_scope",
    "category": "escalation",
    "turns": ["Can you tell me what my insurance policy number is?"],
    "expect": {
      "escalated": true,
      "appointment_count": 0
    }
  },
  {
    "id": "escalate_billing_dispute",
    "category": "escalation",
    "turns": ["I was charged twice for my last visit and I want a refund immediately."],
    "expect": {
      "escalated": true,
      "appointment_count": 0
    }
  },
  {
    "id": "escalate_medical_emergency",
    "category": "escalation",
    "turns": ["I think I'm having an allergic reaction to the anesthetic from yesterday."],
    "expect": {
      "escalated": true,
      "appointment_count": 0
    }
  },

  {
    "id": "ambiguous_then_abandons",
    "category": "ambiguous",
    "turns": ["I want to book an appointment.", "Actually, never mind. Bye."],
    "expect": {
      "tools_not_called": ["book"],
      "appointment_count": 0
    }
  },
  {
    "id": "ambiguous_vague_time",
    "category": "ambiguous",
    "turns": ["I want to come in sometime next week."],
    "expect": {
      "tools_not_called": ["book"],
      "appointment_count": 0
    }
  },
  {
    "id": "changed_mind_mid_call",
    "category": "ambiguous",
    "turns": [
      "Book me Monday July 13th at 9am. Jane Doe, 5551234, for a cleaning.",
      "Actually, can you make it 11am instead?"
    ],
    "expect": {
      "escalated": false,
      "appointments": [{"patient_name": "Jane Doe", "slot_iso": "2026-07-13T11:00", "status": "booked"}]
    }
  }
]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src python -m pytest tests/test_scenarios_file.py -v`
Expected: PASS (5 passed) — 30 scenarios, unique ids, well-formed, all seeds bookable, 5 escalation scenarios.

- [ ] **Step 5: Commit**

```bash
git add evals/scenarios.json tests/test_scenarios_file.py
git commit -m "feat: 30-scenario eval suite weighted toward escalation"
```

---

### Task 7: CLI entrypoint and docs

**Files:**
- Create: `src/voicedesk/evals/__main__.py`
- Modify: `README.md`
- Modify: `.gitignore`
- Test: `tests/test_evals_cli.py`

**Interfaces:**
- Consumes: `load_scenarios`, `run_all` from `voicedesk.evals.runner`; `format_console`, `format_markdown` from `voicedesk.evals.report`; `GroqLLM` from `voicedesk.groq_client`.
- Produces:
  - `select_scenarios(scenarios: list[dict], scenario_id: str | None) -> list[dict]` — filters to one scenario by id; raises `SystemExit` with a message if the id is unknown. (Split out as a plain function so it is testable without invoking the CLI.)
  - `main() -> None` — parses `--scenarios PATH` (default `evals/scenarios.json`), `--scenario ID`, `--runs N` (default 3), `--out PATH` (default `reports/eval-<timestamp>.md`); loads and filters scenarios, runs them against a live `GroqLLM`, prints the console report, and writes the markdown report.
  - Invoked as `python -m voicedesk.evals`.

- [ ] **Step 1: Write the failing test** — `tests/test_evals_cli.py`

Only the pure, offline parts are tested. `main()` needs a live API key, so it is not called here; it is exercised by the manual run in Step 6.

```python
import pytest
from voicedesk.evals.__main__ import select_scenarios

SCENARIOS = [{"id": "a", "turns": ["x"]}, {"id": "b", "turns": ["y"]}]


def test_select_scenarios_returns_all_when_no_id():
    assert select_scenarios(SCENARIOS, None) == SCENARIOS


def test_select_scenarios_filters_by_id():
    assert select_scenarios(SCENARIOS, "b") == [{"id": "b", "turns": ["y"]}]


def test_select_scenarios_exits_on_unknown_id():
    with pytest.raises(SystemExit):
        select_scenarios(SCENARIOS, "nope")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m pytest tests/test_evals_cli.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'voicedesk.evals.__main__'`

- [ ] **Step 3: Implement** — `src/voicedesk/evals/__main__.py`

```python
import argparse
import os
import sys
from datetime import datetime

from dotenv import load_dotenv

from voicedesk.evals.report import format_console, format_markdown
from voicedesk.evals.runner import load_scenarios, run_all
from voicedesk.groq_client import GroqLLM


def select_scenarios(scenarios: list[dict], scenario_id: str | None) -> list[dict]:
    if scenario_id is None:
        return scenarios
    picked = [s for s in scenarios if s["id"] == scenario_id]
    if not picked:
        raise SystemExit(f"no scenario with id {scenario_id!r}")
    return picked


def main() -> None:
    load_dotenv()
    p = argparse.ArgumentParser(
        prog="voicedesk.evals",
        description="Run the VoiceDesk eval suite against the live Groq agent.",
    )
    p.add_argument("--scenarios", default="evals/scenarios.json")
    p.add_argument("--scenario", default=None, help="run only this scenario id")
    p.add_argument("--runs", type=int, default=3, help="runs per scenario")
    p.add_argument("--out", default=None, help="markdown report path")
    args = p.parse_args()

    scenarios = select_scenarios(load_scenarios(args.scenarios), args.scenario)
    print(f"Running {len(scenarios)} scenario(s) x {args.runs} run(s)...\n",
          file=sys.stderr)

    results = run_all(scenarios, lambda: GroqLLM(), runs=args.runs)

    print(format_console(results))

    out = args.out or f"reports/eval-{datetime.now():%Y%m%d-%H%M%S}.md"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(format_markdown(results))
    print(f"\nMarkdown report written to {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src python -m pytest tests/test_evals_cli.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Add the reports dir to `.gitignore`** — append this line, so ad-hoc eval runs don't clutter the repo (a chosen baseline report can still be committed with `git add -f`):

```
reports/
```

- [ ] **Step 6: Run the FULL offline suite, then a live smoke test**

Full offline suite (no network, no key needed):
Run: `PYTHONPATH=src python -m pytest -q`
Expected: PASS — all tests green (Phase 1's 40 plus the new eval tests).

Live smoke test of ONE scenario (needs `GROQ_API_KEY` in `.env`; must be run from the repo root):
Run: `PYTHONPATH=src python -m voicedesk.evals --scenario faq_hours --runs 1`
Expected: a console report showing `faq_hours 1/1 PASS` (or a legible FAIL with the reason), and a markdown report written under `reports/`.

- [ ] **Step 7: Update `README.md`** — add this section immediately after the `## Test` section:

```markdown
## Evals (Phase 2)

The eval harness runs ~30 scripted caller scenarios against the real Groq agent,
3x each, and scores them on objective outcomes (was the right tool called? is the
appointment actually in the DB? did it escalate when it should?).

Run from the repo root (needs `GROQ_API_KEY` in `.env`):

```powershell
# PowerShell
$env:PYTHONPATH = "src"
python -m voicedesk.evals                          # full suite, 3 runs each
python -m voicedesk.evals --scenario faq_hours --runs 1   # one scenario, fast
```

```bash
# Bash
PYTHONPATH=src python -m voicedesk.evals
```

It prints a console report and writes a markdown report to `reports/eval-<timestamp>.md`.
Each scenario is reported as PASS (3/3), FLAKY (1-2/3), or FAIL (0/3) — flakiness is a
first-class metric, because the underlying LLM is non-deterministic.
```

- [ ] **Step 8: Commit**

```bash
git add src/voicedesk/evals/__main__.py tests/test_evals_cli.py README.md .gitignore
git commit -m "feat: eval CLI entrypoint and docs"
```

---

## Phase 2 Definition of Done

- `PYTHONPATH=src python -m pytest` passes fully offline (no network, no API key).
- `python -m voicedesk.evals` runs 30 scenarios × 3 runs against the live agent and prints a report with an overall pass rate, per-scenario PASS/FLAKY/FAIL, per-category breakdown, latency, and debuggable failure detail.
- A markdown report artifact is written to `reports/`.
- Phase 1 agent code is unchanged apart from the Groq adapter's rate-limit backoff.
- A **baseline pass rate is recorded**, so any later improvement can be stated as a measured delta.

## What comes next (separate plans)

- **Phase 3 — Voice:** STT (Groq Whisper) + TTS + FastAPI WebSocket wrapping the same `Agent`. The eval harness carries over unchanged, since it drives the text interface.
- **Phase 4 — Deploy + polish:** hosting, latency measurement, README case study.
