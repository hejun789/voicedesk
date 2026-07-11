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


def test_llm_error_fails_immediately():
    res = score_run(_record(error="429 rate limit"), {"tools_called": ["book"]})
    assert res.passed is False
    assert res.failures == ["llm_error: 429 rate limit"]


def test_empty_expect_passes():
    assert score_run(_record(), {}).passed is True
