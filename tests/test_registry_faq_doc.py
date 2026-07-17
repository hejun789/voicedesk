from voicedesk.registry import dispatch


_doc_counter = [0]


def _doc(tmp_path, text):
    # Distinct filename per call: tests that create two documents in the same
    # tmp_path (e.g. a "caller" doc and a "model" doc) must not collide on
    # the same file, or the second write silently overwrites the first and
    # the test can no longer distinguish which path was actually read.
    _doc_counter[0] += 1
    p = tmp_path / f"doc{_doc_counter[0]}.md"
    p.write_text(text, encoding="utf-8")
    return str(p)


def test_faq_doc_path_from_the_caller_is_used(tmp_path):
    doc = _doc(tmp_path, "## Hours\nOpen Monday to Friday.\n")
    res = dispatch(None, "answer_faq", {"query": "hours"}, faq_doc_path=doc)
    assert "Monday" in res["answer"]


def test_caller_doc_path_beats_a_model_supplied_one(tmp_path):
    # The model must never choose which file gets read.
    caller = _doc(tmp_path, "## Hours\nCaller document wins.\n")
    model = _doc(tmp_path, "## Hours\nModel document loses.\n")
    res = dispatch(None, "answer_faq",
                   {"query": "hours", "doc_path": model},
                   faq_doc_path=caller)
    assert "Caller document wins" in res["answer"]


def test_dispatch_without_faq_doc_path_still_honours_args(tmp_path):
    # Backwards compatible: existing callers pass doc_path in args.
    doc = _doc(tmp_path, "## Hours\nOpen Monday to Friday.\n")
    res = dispatch(None, "answer_faq", {"query": "hours", "doc_path": doc})
    assert "Monday" in res["answer"]


def test_agent_passes_its_faq_doc_path_to_the_tool(db, tmp_path):
    from voicedesk.agent import Agent
    from voicedesk.llm import FakeLLM, Message, ToolCall

    doc = _doc(tmp_path, "## Hours\nWe open at dawn.\n")
    llm = FakeLLM([
        Message(content=None, tool_calls=[
            ToolCall(id="1", name="answer_faq", arguments={"query": "hours"})]),
        Message(content="We open at dawn.", tool_calls=[]),
    ])
    agent = Agent(db, llm, faq_doc_path=doc)
    assert "dawn" in agent.respond("what are your hours")
