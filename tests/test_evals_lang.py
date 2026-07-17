import pytest
from voicedesk.llm import FakeLLM, Message
from voicedesk.evals.runner import run_scenario_once
from voicedesk.evals.__main__ import select_by_lang


class _CapturingLLM:
    """Captures the system prompt the agent was built with."""

    def __init__(self):
        self.system_prompt = None

    def complete(self, messages, tools):
        self.system_prompt = messages[0]["content"]
        return Message(content="ok", tool_calls=[])


def test_chinese_scenario_builds_a_chinese_agent():
    llm = _CapturingLLM()
    run_scenario_once(
        {"id": "zh_x", "category": "faq", "lang": "zh",
         "turns": ["你们的营业时间？"], "expect": {}},
        llm,
    )
    assert "简体中文" in llm.system_prompt


def test_scenario_without_lang_is_english():
    llm = _CapturingLLM()
    run_scenario_once(
        {"id": "x", "category": "faq", "turns": ["what are your hours?"],
         "expect": {}},
        llm,
    )
    assert "简体中文" not in llm.system_prompt


SCENARIOS = [
    {"id": "book_oneshot", "turns": ["x"]},
    {"id": "zh_book_oneshot", "lang": "zh", "turns": ["x"]},
]


def test_select_by_lang_filters_chinese():
    assert [s["id"] for s in select_by_lang(SCENARIOS, "zh")] == ["zh_book_oneshot"]


def test_select_by_lang_filters_english_including_unlabelled():
    # A scenario with no lang field is English.
    assert [s["id"] for s in select_by_lang(SCENARIOS, "en")] == ["book_oneshot"]


def test_select_by_lang_none_returns_everything():
    assert select_by_lang(SCENARIOS, None) == SCENARIOS
