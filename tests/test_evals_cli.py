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
