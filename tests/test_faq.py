from voicedesk.faq import answer_faq


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
