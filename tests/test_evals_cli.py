import pytest
from voicedesk.evals.__main__ import select_scenarios, require_api_key

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
