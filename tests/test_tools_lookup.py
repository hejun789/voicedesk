from voicedesk.tools import lookup_appt, book


def test_lookup_by_phone(db):
    book(db, "Jane Doe", "5551234", "2026-07-13T09:00", "cleaning")
    res = lookup_appt(db, phone="5551234")
    assert len(res) == 1
    assert res[0]["patient_name"] == "Jane Doe"
    assert res[0]["slot_iso"] == "2026-07-13T09:00"


def test_lookup_by_name_case_insensitive(db):
    book(db, "Jane Doe", "5551234", "2026-07-13T09:00", "cleaning")
    res = lookup_appt(db, name="jane")
    assert len(res) == 1


def test_lookup_no_criteria_returns_empty(db):
    book(db, "Jane Doe", "5551234", "2026-07-13T09:00", "cleaning")
    assert lookup_appt(db) == []


def test_lookup_no_match_returns_empty(db):
    book(db, "Jane Doe", "5551234", "2026-07-13T09:00", "cleaning")
    assert lookup_appt(db, phone="0000000") == []
