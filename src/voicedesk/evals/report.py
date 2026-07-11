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

    error_runs = sum(1 for r in results if r.record.error is not None)
    latencies = [r.record.latency_s for r in results if r.record.error is None]
    return {
        "total_runs": total,
        "passed_runs": passed,
        "pass_rate": (passed / total) if total else 0.0,
        "per_scenario": per_scenario,
        "per_category": per_category,
        "latency_mean": mean(latencies) if latencies else 0.0,
        "latency_p50": median(latencies) if latencies else 0.0,
        "error_runs": error_runs,
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
    ]
    if s["error_runs"] > 0:
        out.append(
            f"Errors: {s['error_runs']} run(s) failed due to LLM/API errors "
            "(not agent defects)"
        )
    out += [
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
    ]
    if s["error_runs"] > 0:
        out += [
            "",
            f"Errors: {s['error_runs']} run(s) failed due to LLM/API errors "
            "(not agent defects)",
        ]
    out += [
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
