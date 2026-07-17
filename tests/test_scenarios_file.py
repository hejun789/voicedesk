import pytest
from voicedesk.evals.runner import load_scenarios, fresh_db, seed_db

VALID_EXPECT_KEYS = {
    "tools_called", "tools_not_called", "escalated",
    "appointments", "appointment_count", "reply_contains",
}
KNOWN_TOOLS = {
    "find_slots", "book", "reschedule", "cancel",
    "lookup_appt", "answer_faq", "escalate",
}


@pytest.fixture(scope="module")
def scenarios():
    return load_scenarios("evals/scenarios.json")


def test_has_about_thirty_scenarios(scenarios):
    assert len(scenarios) >= 30


def test_has_sixty_scenarios(scenarios):
    assert len(scenarios) == 60


def test_every_english_scenario_has_a_chinese_mirror(scenarios):
    by_id = {s["id"]: s for s in scenarios}
    english = [s for s in scenarios if s.get("lang", "en") == "en"]
    assert len(english) == 30
    for s in english:
        mirror = by_id.get(f"zh_{s['id']}")
        assert mirror is not None, f"no Chinese mirror for {s['id']}"
        # The comparison is only meaningful if the expectations are identical.
        assert mirror["expect"] == s["expect"], s["id"]
        assert mirror.get("seed") == s.get("seed"), s["id"]
        assert mirror["category"] == s["category"], s["id"]


def test_chinese_scenarios_are_labelled_and_actually_chinese(scenarios):
    zh = [s for s in scenarios if s["id"].startswith("zh_")]
    assert len(zh) == 30
    for s in zh:
        assert s["lang"] == "zh", s["id"]
        # Every turn must contain CJK characters — a turn left in English would
        # silently measure the wrong thing.
        for turn in s["turns"]:
            assert any("一" <= ch <= "鿿" for ch in turn), s["id"]


def test_ids_are_unique(scenarios):
    ids = [s["id"] for s in scenarios]
    assert len(ids) == len(set(ids))


def test_every_scenario_is_well_formed(scenarios):
    for s in scenarios:
        assert s["id"] and s["category"]
        assert s["turns"] and all(isinstance(t, str) for t in s["turns"])
        assert set(s["expect"]).issubset(VALID_EXPECT_KEYS), s["id"]
        for key in ("tools_called", "tools_not_called"):
            assert set(s["expect"].get(key, [])).issubset(KNOWN_TOOLS), s["id"]


def test_every_seed_is_bookable(scenarios):
    # A seed that cannot be booked would silently skew results.
    for s in scenarios:
        if s.get("seed"):
            seed_db(fresh_db(), s["seed"])  # raises ValueError if unbookable


def test_escalation_category_is_well_represented(scenarios):
    escalation = [s for s in scenarios if s["category"] == "escalation"]
    assert len(escalation) >= 5
    assert all(s["expect"].get("escalated") is True for s in escalation)


VALID_APPOINTMENT_KEYS = {"patient_name", "phone", "slot_iso", "reason", "status"}


def test_expect_appointments_use_only_known_keys(scenarios):
    for s in scenarios:
        for appt in s["expect"].get("appointments", []):
            assert set(appt).issubset(VALID_APPOINTMENT_KEYS), s["id"]


def test_booking_scenarios_require_a_confirmation_turn(scenarios):
    # The agent must read details back and get an explicit yes before
    # calling book, so a booking can never complete in a single turn.
    for s in scenarios:
        books = "book" in s["expect"].get("tools_called", [])
        booked = any(
            appt.get("status") == "booked"
            for appt in s["expect"].get("appointments", [])
        )
        if books and booked:
            assert len(s["turns"]) >= 2, s["id"]


def test_destructive_scenarios_require_a_confirmation_turn(scenarios):
    # The agent must read the found appointment back and get an explicit yes
    # before cancelling or rescheduling it, so those actions can never
    # complete in a single turn either.
    for s in scenarios:
        tools_called = s["expect"].get("tools_called", [])
        if "cancel" in tools_called or "reschedule" in tools_called:
            assert len(s["turns"]) >= 2, s["id"]
