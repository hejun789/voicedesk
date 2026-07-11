from voicedesk.evals.scoring import RunRecord, RunResult
from voicedesk.evals.report import (
    summarize, status_of, format_console, format_markdown,
)


def _result(sid, category, passed, latency=1.0, failures=None, error=None):
    rec = RunRecord(scenario_id=sid, category=category, tools_called=[],
                    escalated=False, appointments=[], final_reply="",
                    latency_s=latency, error=error)
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


def test_summarize_reports_error_runs():
    results = [
        _result("a", "booking", True),
        _result("b", "escalation", False, error="429 rate limit"),
        _result("c", "faq", False, error="timeout"),
    ]
    s = summarize(results)
    assert s["error_runs"] == 2


def test_summarize_error_runs_zero_when_none():
    s = summarize([_result("a", "booking", True)])
    assert s["error_runs"] == 0


def test_summarize_latency_excludes_errored_runs():
    results = [
        _result("a", "booking", True, latency=1.0),
        _result("a", "booking", True, latency=3.0),
        _result("b", "escalation", False, latency=999.0, error="429 rate limit"),
    ]
    s = summarize(results)
    assert s["latency_mean"] == 2.0
    assert s["latency_p50"] == 2.0


def test_summarize_latency_zero_when_all_errored():
    s = summarize([_result("a", "booking", False, latency=50.0, error="boom")])
    assert s["latency_mean"] == 0.0
    assert s["latency_p50"] == 0.0


def test_format_console_shows_error_line_when_errors_present():
    results = [_result("a", "booking", False, error="429 rate limit")]
    out = format_console(results)
    assert "Errors: 1 run(s) failed due to LLM/API errors (not agent defects)" in out


def test_format_console_omits_error_line_when_no_errors():
    out = format_console([_result("a", "booking", True)])
    assert "LLM/API errors" not in out


def test_format_markdown_shows_error_line_when_errors_present():
    results = [_result("a", "booking", False, error="429 rate limit")]
    out = format_markdown(results)
    assert "Errors: 1 run(s) failed due to LLM/API errors (not agent defects)" in out


def test_format_markdown_omits_error_line_when_no_errors():
    out = format_markdown([_result("a", "booking", True)])
    assert "LLM/API errors" not in out
