import re
from dataclasses import dataclass, field


_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_text(text: str) -> str:
    """Collapse every kind of whitespace (models emit U+00A0, U+202F, U+2009,
    newlines...) to single ASCII spaces, so a reply_contains check tests the
    words rather than the typography."""
    return _WHITESPACE_RE.sub(" ", text).strip()


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
    tool_calls: list[dict] = field(default_factory=list)


@dataclass
class RunResult:
    record: RunRecord
    passed: bool
    failures: list[str] = field(default_factory=list)


def _matches(expected: dict, actual: dict) -> bool:
    """Partial match: every key in `expected` equals the same key in `actual`."""
    return all(actual.get(k) == v for k, v in expected.items())


def score_run(record: RunRecord, expect: dict) -> RunResult:
    if record.error is not None:
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
        haystack = _normalize_text(record.final_reply).lower()
        if _normalize_text(needle).lower() not in haystack:
            failures.append(
                f"reply did not contain {needle!r}; reply={record.final_reply!r}")

    return RunResult(record=record, passed=not failures, failures=failures)
