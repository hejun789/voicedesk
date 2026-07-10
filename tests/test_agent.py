from voicedesk.llm import FakeLLM, Message, ToolCall
from voicedesk.agent import Agent


def test_agent_returns_plain_text(db):
    llm = FakeLLM([Message(content="Hello! How can I help?", tool_calls=[])])
    agent = Agent(db, llm)
    assert agent.respond("hi") == "Hello! How can I help?"


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
