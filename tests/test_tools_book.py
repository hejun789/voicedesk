from voicedesk.tools import book, find_slots


def test_book_success(db):
    res = book(db, "Jane Doe", "5551234", "2026-07-13T09:00", "cleaning")
    assert res["ok"] is True
    assert isinstance(res["appointment_id"], int)
    assert "2026-07-13T09:00" not in find_slots(db, "2026-07-13")


def test_book_rejects_taken_slot(db):
    book(db, "Jane", "5551234", "2026-07-13T09:00", "cleaning")
    res = book(db, "John", "5559999", "2026-07-13T09:00", "cleaning")
    assert res == {"ok": False, "error": "slot_unavailable"}


def test_book_rejects_outside_hours(db):
    res = book(db, "Jane", "5551234", "2026-07-13T20:00", "cleaning")
    assert res == {"ok": False, "error": "slot_unavailable"}


def test_book_rejects_weekend(db):
    res = book(db, "Jane", "5551234", "2026-07-11T09:00", "cleaning")
    assert res == {"ok": False, "error": "slot_unavailable"}


def test_book_rejects_placeholder_name_unknown(db):
    res = book(db, "unknown", "5551234", "2026-07-13T09:00", "cleaning")
    assert res == {"ok": False, "error": "missing_patient_details"}
    assert "2026-07-13T09:00" in find_slots(db, "2026-07-13")


def test_book_rejects_placeholder_name_literal_param(db):
    res = book(db, "patient_name", "5551234", "2026-07-13T09:00", "cleaning")
    assert res == {"ok": False, "error": "missing_patient_details"}
    assert "2026-07-13T09:00" in find_slots(db, "2026-07-13")


def test_book_rejects_empty_name(db):
    res = book(db, "", "5551234", "2026-07-13T09:00", "cleaning")
    assert res == {"ok": False, "error": "missing_patient_details"}
    assert "2026-07-13T09:00" in find_slots(db, "2026-07-13")


def test_book_rejects_placeholder_phone(db):
    res = book(db, "Jane Doe", "unknown", "2026-07-13T09:00", "cleaning")
    assert res == {"ok": False, "error": "missing_patient_details"}
    assert "2026-07-13T09:00" in find_slots(db, "2026-07-13")


def test_book_rejects_phone_without_digits(db):
    res = book(db, "Jane Doe", "abc", "2026-07-13T09:00", "cleaning")
    assert res == {"ok": False, "error": "missing_patient_details"}
    assert "2026-07-13T09:00" in find_slots(db, "2026-07-13")


def test_book_accepts_real_values(db):
    res = book(db, "Jane Doe", "5551234", "2026-07-13T09:00", "cleaning")
    assert res["ok"] is True
    assert "2026-07-13T09:00" not in find_slots(db, "2026-07-13")
