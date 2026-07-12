import pytest
from voicedesk.evals.__main__ import select_scenarios, require_api_key, resolve_model
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
