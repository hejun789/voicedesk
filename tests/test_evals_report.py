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
