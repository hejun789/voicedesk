from voicedesk.llm import FakeLLM, Message, ToolCall, LLMError
from voicedesk.evals.runner import run_scenario_once, run_scenario, run_all

BOOK_SCENARIO = {
    "id": "book_oneshot",
    "category": "booking",
    "turns": ["Book Monday July 13th 9am, Jane Doe, 5551234, cleaning"],
    "expect": {
        "tools_called": ["book"],
        "escalated": False,
        "appointments": [{"patient_name": "Jane Doe",
                          "slot_iso": "2026-07-13T09:00", "status": "booked"}],
    },
}


def _booking_llm():
    return FakeLLM([
        Message(content=None, tool_calls=[
            ToolCall(id="1", name="book", arguments={
                "patient_name": "Jane Doe", "phone": "5551234",
                "slot_iso": "2026-07-13T09:00", "reason": "cleaning"})]),
        Message(content="You're booked for Monday at 9am.", tool_calls=[]),
    ])


def test_run_scenario_once_records_what_happened():
    rec = run_scenario_once(BOOK_SCENARIO, _booking_llm())
    assert rec.scenario_id == "book_oneshot"
    assert rec.category == "booking"
    assert rec.tools_called == ["book"]
    assert rec.escalated is False
    assert rec.appointments[0]["slot_iso"] == "2026-07-13T09:00"
    assert "booked" in rec.final_reply.lower()
    assert rec.latency_s >= 0
    assert rec.error is None


def test_run_scenario_once_applies_seed():
    scenario = {
        "id": "cancel", "category": "cancel",
        "seed": [{"patient_name": "Jane Doe", "phone": "5551234",
                  "slot_iso": "2026-07-13T09:00", "reason": "cleaning"}],
        "turns": ["cancel it"],
        "expect": {},
    }
    llm = FakeLLM([Message(content="Which appointment?", tool_calls=[])])
    rec = run_scenario_once(scenario, llm)
    assert rec.appointments[0]["patient_name"] == "Jane Doe"


def test_run_scenario_once_captures_llm_error():
    class _Raising:
        def complete(self, messages, tools):
            raise LLMError("429 rate limit")

    rec = run_scenario_once(BOOK_SCENARIO, _Raising())
    assert rec.error == "429 rate limit"


def test_run_scenario_scores_each_run():
    results = run_scenario(BOOK_SCENARIO, _booking_llm, runs=3)
    assert len(results) == 3
    assert all(r.passed for r in results)


def test_run_scenario_detects_escalation():
    scenario = {
        "id": "esc", "category": "escalation",
        "turns": ["asdkjh qwe zxcv"],
        "expect": {"escalated": True},
    }

    def _llm():
        return FakeLLM([
            Message(content=None, tool_calls=[
                ToolCall(id="1", name="escalate",
                         arguments={"reason": "unintelligible"})]),
            Message(content="Let me have someone call you back.", tool_calls=[]),
        ])

    results = run_scenario(scenario, _llm, runs=1)
    assert results[0].passed is True
    assert results[0].record.escalated is True


def test_run_all_flattens_results():
    results = run_all([BOOK_SCENARIO, BOOK_SCENARIO], _booking_llm, runs=2)
    assert len(results) == 4
