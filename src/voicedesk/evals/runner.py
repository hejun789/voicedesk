import json
import sqlite3
import time

from voicedesk import tools
from voicedesk.agent import Agent
from voicedesk.db import init_db
from voicedesk.llm import LLMError
from voicedesk.evals.scoring import RunRecord, RunResult, score_run


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
