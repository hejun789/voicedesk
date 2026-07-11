from datetime import date

from voicedesk.llm import FakeLLM, LLMError, Message, ToolCall
from voicedesk.agent import Agent, _FALLBACK, build_system_prompt


class _RaisingLLM:
    """LLMClient double whose complete() always raises LLMError."""

    def complete(self, messages, tools):
        raise LLMError("groq 400 tool_use_failed after retries")


def test_agent_returns_plain_text(db):
    llm = FakeLLM([Message(content="Hello! How can I help?", tool_calls=[])])
    agent = Agent(db, llm)
    assert agent.respond("hi") == "Hello! How can I help?"


def test_agent_escalates_on_llm_error(db):
    # A persistent LLM/API failure must degrade to the escalation reply,
    # never crash the caller (the REPL).
    agent = Agent(db, _RaisingLLM())
    assert agent.respond("book me monday 9am") == _FALLBACK


def test_agent_executes_tool_then_replies(db):
    llm = FakeLLM([
        Message(content=None, tool_calls=[
            ToolCall(id="1", name="book", arguments={
                "patient_name": "Jane", "phone": "5551234",
                "slot_iso": "2026-07-13T09:00", "reason": "cleaning"})]),
        Message(content="You're booked for Monday at 9am.", tool_calls=[]),
    ])
    agent = Agent(db, llm)
    reply = agent.respond("book me monday 9am, Jane, 5551234, cleaning")
    assert "booked" in reply.lower()
    # side effect really happened:
    from voicedesk.tools import lookup_appt
    assert lookup_appt(db, phone="5551234")[0]["slot_iso"] == "2026-07-13T09:00"


def test_agent_stops_at_iteration_cap(db):
    # LLM always asks for a tool, never returns text -> loop must terminate.
    looping = [
        Message(content=None, tool_calls=[
            ToolCall(id="x", name="find_slots", arguments={"day_iso": "2026-07-13"})])
        for _ in range(10)
    ]
    agent = Agent(db, FakeLLM(looping))
    reply = agent.respond("slots?")
    assert isinstance(reply, str) and len(reply) > 0  # returns a fallback, no crash


def test_agent_survives_tool_exception(db):
    llm = FakeLLM([
        Message(content=None, tool_calls=[
            ToolCall(id="1", name="book", arguments={
                "patient_name": "Jane", "phone": "5551234",
                "slot_iso": "2026-07-13T09:00"})]),  # missing required "reason" -> KeyError
        Message(content="Sorry, let me get a human.", tool_calls=[]),
    ])
    agent = Agent(db, llm)
    reply = agent.respond("book me monday 9am, Jane, 5551234")
    assert isinstance(reply, str)
    assert reply == "Sorry, let me get a human."


def test_build_system_prompt_grounds_today():
    prompt = build_system_prompt(date(2026, 7, 10))
    assert "Friday, 10 July 2026" in prompt


def test_build_system_prompt_forbids_placeholder_details():
    prompt = build_system_prompt(date(2026, 7, 10))
    assert "Never" in prompt
    assert "placeholder" in prompt


def test_build_system_prompt_requires_relaying_faq_answer():
    prompt = build_system_prompt(date(2026, 7, 10))
    assert "relay the retrieved answer" in prompt


def test_build_system_prompt_forbids_inventing_appointment_id():
    prompt = build_system_prompt(date(2026, 7, 10))
    assert "NEVER invent or guess an appointment_id" in prompt


def test_build_system_prompt_requires_escalation_for_medical_and_billing():
    prompt = build_system_prompt(date(2026, 7, 10))
    assert "medical" in prompt.lower()
    assert "billing dispute" in prompt.lower()


def test_build_system_prompt_requires_escalation_on_gibberish():
    prompt = build_system_prompt(date(2026, 7, 10))
    assert "gibberish" in prompt.lower()


def test_build_system_prompt_lookup_appt_name_or_phone_alone():
    prompt = build_system_prompt(date(2026, 7, 10))
    assert "name ALONE or a phone number ALONE" in prompt


def test_agent_default_system_prompt_has_current_year(db):
    agent = Agent(db, FakeLLM([]))
    assert str(date.today().year) in agent.messages[0]["content"]


def test_agent_explicit_system_prompt_used_verbatim(db):
    custom = "custom prompt text"
    agent = Agent(db, FakeLLM([]), system_prompt=custom)
    assert agent.messages[0]["content"] == custom
