from voicedesk.agent import Agent
from voicedesk.llm import FakeLLM, Message


def _agent_with_reply(db, reply: str) -> Agent:
    agent = Agent(db, FakeLLM([Message(content=reply, tool_calls=[])]))
    agent.respond("hi")
    return agent


def test_truncates_reply_to_what_was_heard(db):
    agent = _agent_with_reply(db, "Booked for Monday at nine. Anything else?")
    agent.truncate_last_reply(26)   # through "…at nine."
    assert agent.messages[-1]["content"] == "Booked for Monday at nine. [interrupted]"


def test_marker_is_appended_when_cut_off(db):
    agent = _agent_with_reply(db, "Booked for Monday at nine. Anything else?")
    agent.truncate_last_reply(10)
    content = agent.messages[-1]["content"]
    assert content.endswith("[interrupted]")
    assert content.startswith("Booked for")
    assert "Anything else" not in content


def test_hearing_the_whole_reply_changes_nothing(db):
    full = "Booked for Monday."
    agent = _agent_with_reply(db, full)
    agent.truncate_last_reply(len(full))
    assert agent.messages[-1]["content"] == full


def test_heard_chars_beyond_length_is_clamped_and_changes_nothing(db):
    full = "Booked for Monday."
    agent = _agent_with_reply(db, full)
    agent.truncate_last_reply(9999)
    assert agent.messages[-1]["content"] == full


def test_hearing_nothing_removes_the_message_entirely(db):
    agent = _agent_with_reply(db, "Booked for Monday.")
    before = len(agent.messages)
    agent.truncate_last_reply(0)
    assert len(agent.messages) == before - 1
    assert agent.messages[-1]["role"] != "assistant"


def test_negative_heard_chars_clamps_to_zero(db):
    agent = _agent_with_reply(db, "Booked for Monday.")
    before = len(agent.messages)
    agent.truncate_last_reply(-5)
    assert len(agent.messages) == before - 1


def test_assistant_message_with_tool_calls_is_never_truncated(db):
    # A tool-call message is internal bookkeeping, never spoken aloud, so an
    # interruption must not corrupt it.
    agent = Agent(db, FakeLLM([]))
    agent.messages.append({
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": "1", "type": "function",
                        "function": {"name": "book", "arguments": "{}"}}],
    })
    before = list(agent.messages)
    agent.truncate_last_reply(0)
    assert agent.messages == before


def test_no_op_when_last_message_is_not_from_the_assistant(db):
    agent = Agent(db, FakeLLM([]))
    agent.messages.append({"role": "user", "content": "hello"})
    before = list(agent.messages)
    agent.truncate_last_reply(2)
    assert agent.messages == before
