from voicedesk.registry import TOOL_SCHEMAS, dispatch


def test_schemas_cover_expected_tools():
    names = {s["function"]["name"] for s in TOOL_SCHEMAS}
    assert names == {
        "find_slots", "book", "reschedule", "cancel",
        "lookup_appt", "answer_faq", "escalate",
    }


def test_dispatch_book(db):
    res = dispatch(db, "book", {
        "patient_name": "Jane", "phone": "5551234",
        "slot_iso": "2026-07-13T09:00", "reason": "cleaning",
    })
    assert res["ok"] is True


def test_dispatch_escalate():
    res = dispatch(None, "escalate", {"reason": "angry caller"})
    assert res == {"ok": True, "escalated": True, "reason": "angry caller"}


def test_dispatch_faq(tmp_path):
    doc = tmp_path / "info.md"
    doc.write_text("## Hours\nOpen Monday to Friday.\n")
    res = dispatch(None, "answer_faq", {"query": "hours", "doc_path": str(doc)})
    assert "Monday" in res["answer"]


def test_dispatch_unknown():
    assert dispatch(None, "nope", {}) == {"ok": False, "error": "unknown_tool"}


def test_dispatch_faq_no_match_steers_to_escalate(tmp_path):
    doc = tmp_path / "info.md"
    doc.write_text("## Hours\nOpen Monday to Friday.\n")
    res = dispatch(None, "answer_faq", {"query": "what is my insurance policy number", "doc_path": str(doc)})
    assert res["answer"] == "NO_MATCH"
    assert "escalate" in res["note"]


def test_dispatch_faq_match_returns_only_answer_key(tmp_path):
    doc = tmp_path / "info.md"
    doc.write_text("## Hours\nOpen Monday to Friday.\n")
    res = dispatch(None, "answer_faq", {"query": "hours", "doc_path": str(doc)})
    assert set(res) == {"answer"}
