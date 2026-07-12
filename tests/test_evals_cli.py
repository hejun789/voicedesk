import pytest
from voicedesk.evals.__main__ import (
    select_scenarios, require_api_key, resolve_model, quota_exhausted,
)
from voicedesk.evals.scoring import RunRecord, RunResult
from voicedesk.groq_client import DEFAULT_MODEL

SCENARIOS = [{"id": "a", "turns": ["x"]}, {"id": "b", "turns": ["y"]}]


def test_select_scenarios_returns_all_when_no_id():
    assert select_scenarios(SCENARIOS, None) == SCENARIOS


def test_select_scenarios_filters_by_id():
    assert select_scenarios(SCENARIOS, "b") == [{"id": "b", "turns": ["y"]}]


def test_select_scenarios_exits_on_unknown_id():
    with pytest.raises(SystemExit):
        select_scenarios(SCENARIOS, "nope")


def test_require_api_key_exits_when_missing(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="GROQ_API_KEY not set"):
        require_api_key()


def test_require_api_key_exits_when_empty(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "")
    with pytest.raises(SystemExit, match="GROQ_API_KEY not set"):
        require_api_key()


def test_require_api_key_passes_when_set(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "sk-something")
    require_api_key()  # should not raise


def test_resolve_model_uses_env_var_when_set(monkeypatch):
    monkeypatch.setenv("GROQ_MODEL", "llama-3.1-8b-instant")
    assert resolve_model() == "llama-3.1-8b-instant"


def test_resolve_model_falls_back_to_default_when_unset(monkeypatch):
    monkeypatch.delenv("GROQ_MODEL", raising=False)
    assert resolve_model() == DEFAULT_MODEL


def _record(error=None):
    return RunRecord(
        scenario_id="s1", category="c", tools_called=[], escalated=False,
        appointments=[], final_reply="", latency_s=0.0, error=error,
    )


def test_quota_exhausted_true_when_any_result_reports_daily_quota():
    results = [
        RunResult(record=_record(), passed=True, failures=[]),
        RunResult(record=_record(error="Groq asked for a 300s wait — the daily "
                                        "quota for 'x' is exhausted."),
                   passed=False, failures=["llm_error: ..."]),
    ]
    assert quota_exhausted(results) is True


def test_quota_exhausted_false_for_normal_failure():
    results = [
        RunResult(record=_record(error="llm_error: 401 invalid api key"),
                   passed=False, failures=["llm_error: 401 invalid api key"]),
    ]
    assert quota_exhausted(results) is False


def test_quota_exhausted_false_for_passing_run():
    results = [RunResult(record=_record(), passed=True, failures=[])]
    assert quota_exhausted(results) is False
