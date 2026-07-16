from voicedesk.evals.scoring import RunRecord, score_run


def _record(**over):
    base = dict(
        scenario_id="s1", category="booking", tools_called=["book"],
        escalated=False,
        appointments=[{"patient_name": "Jane Doe", "phone": "5551234",
                       "slot_iso": "2026-07-13T09:00", "reason": "cleaning",
                       "status": "booked"}],
        final_reply="You're booked for Monday at 9am.", latency_s=1.0, error=None,
    )
    base.update(over)
    return RunRecord(**base)


def test_passes_when_all_assertions_hold():
    res = score_run(_record(), {
        "tools_called": ["book"],
        "escalated": False,
        "appointments": [{"patient_name": "Jane Doe",
                          "slot_iso": "2026-07-13T09:00", "status": "booked"}],
        "appointment_count": 1,
        "reply_contains": "booked",
    })
    assert res.passed is True
    assert res.failures == []


def test_fails_on_missing_tool():
    res = score_run(_record(tools_called=["find_slots"]), {"tools_called": ["book"]})
    assert res.passed is False
    assert "book" in res.failures[0]


def test_fails_on_forbidden_tool():
    res = score_run(_record(tools_called=["book"]), {"tools_not_called": ["book"]})
    assert res.passed is False


def test_fails_on_escalation_mismatch():
    res = score_run(_record(escalated=False), {"escalated": True})
    assert res.passed is False


def test_fails_on_appointment_mismatch():
    res = score_run(_record(), {
        "appointments": [{"patient_name": "Jane Doe",
                          "slot_iso": "2026-07-13T11:00", "status": "booked"}]
    })
    assert res.passed is False


def test_fails_on_appointment_count():
    res = score_run(_record(), {"appointment_count": 0})
    assert res.passed is False


def test_fails_on_reply_contains():
    res = score_run(_record(), {"reply_contains": "cancelled"})
    assert res.passed is False


def test_reply_contains_passes_on_exact_substring():
    res = score_run(
        _record(final_reply="located at 200 Market Street"),
        {"reply_contains": "Market Street"},
    )
    assert res.passed is True


def test_reply_contains_ignores_unicode_whitespace_differences():
    # Model emitted U+202F (narrow no-break space) between "Market" and
    # "Street"; the words are still there and the check must still pass.
    res = score_run(
        _record(final_reply="200 Market Street, Suite 4"),
        {"reply_contains": "Market Street"},
    )
    assert res.passed is True


def test_reply_contains_still_fails_when_words_genuinely_absent():
    res = score_run(
        _record(final_reply="we are downtown"),
        {"reply_contains": "Market Street"},
    )
    assert res.passed is False


def test_reply_contains_still_case_insensitive():
    res = score_run(
        _record(final_reply="MARKET STREET is where we are"),
        {"reply_contains": "market street"},
    )
    assert res.passed is True


def test_llm_error_fails_immediately():
    res = score_run(_record(error="429 rate limit"), {"tools_called": ["book"]})
    assert res.passed is False
    assert res.failures == ["llm_error: 429 rate limit"]


def test_empty_expect_passes():
    assert score_run(_record(), {}).passed is True


def test_empty_string_error_still_fails_the_run():
    # An LLMError whose str() is empty ("") is falsy but still means the
    # API call failed; `if record.error:` would wrongly score assertions
    # against a successful-looking record.
    res = score_run(_record(error=""), {"tools_called": ["book"]})
    assert res.passed is False
    assert res.failures == ["llm_error: "]
