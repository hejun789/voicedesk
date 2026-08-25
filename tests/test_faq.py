from voicedesk.faq import answer_faq

_CLINIC = """## Hours
Open Monday to Friday 9am to 5pm.

## Location
We are located at 200 Market Street.

## Insurance
We accept Delta Dental, Cigna and Aetna.

## Services
We offer cleanings and crowns.
"""


def _clinic(tmp_path):
    doc = tmp_path / "info.md"
    doc.write_text(_CLINIC, encoding="utf-8")
    return str(doc)


def test_faq_hours(tmp_path):
    doc = tmp_path / "info.md"
    doc.write_text(
        "## Hours\nOpen Monday to Friday 9am to 5pm.\n\n"
        "## Location\nWe are located at 200 Market Street.\n"
    )
    ans = answer_faq("what are your opening hours", str(doc))
    assert "Monday to Friday" in ans


def test_faq_location(tmp_path):
    doc = tmp_path / "info.md"
    doc.write_text(
        "## Hours\nOpen Monday to Friday 9am to 5pm.\n\n"
        "## Location\nWe are located at 200 Market Street.\n"
    )
    ans = answer_faq("where are you located", str(doc))
    assert "Market Street" in ans


def test_faq_no_match_returns_sentinel(tmp_path):
    doc = tmp_path / "info.md"
    doc.write_text("## Hours\nOpen Monday to Friday.\n")
    assert answer_faq("do you sell airplane tickets", str(doc)) == "NO_MATCH"


# A caller asks two things in one breath far more often than the tool's
# one-section-per-call shape assumed. Returning only the better-scoring
# section silently drops half the question; the model then has to re-query,
# and when that follow-up mis-scores it can loop until the agent's iteration
# cap, so the caller gets the generic fallback instead of an answer. Found
# live: "What are your opening hours? Do you take Cigna?" came back with
# insurance only, then burned four more answer_faq calls and fell back.

def test_two_topic_question_answers_both_topics(tmp_path):
    ans = answer_faq("What are your opening hours and do you accept Cigna?",
                     _clinic(tmp_path))
    assert "Monday to Friday" in ans
    assert "Cigna" in ans


def test_single_topic_question_does_not_drag_in_other_sections(tmp_path):
    # The other half of the contract: widening the match must not turn every
    # answer into the whole document.
    ans = answer_faq("what are your opening hours", _clinic(tmp_path))
    assert "Monday to Friday" in ans
    assert "Market Street" not in ans
    assert "Cigna" not in ans
