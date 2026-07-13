from voicedesk.tools import book, cancel, find_slots, reschedule


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


def test_book_rejects_second_same_day_booking_same_phone(db):
    r1 = book(db, "Jane Doe", "5551234", "2026-07-13T09:00", "cleaning")
    assert r1["ok"] is True

    r2 = book(db, "Jane Doe", "5551234", "2026-07-13T10:00", "cleaning")
    assert r2 == {
        "ok": False,
        "error": "already_booked_that_day",
        "existing_slot_iso": "2026-07-13T09:00",
    }
    row = db.execute("SELECT COUNT(*) FROM appointments").fetchone()
    assert row[0] == 1


def test_book_allows_same_phone_on_different_day(db):
    r1 = book(db, "Jane Doe", "5551234", "2026-07-13T09:00", "cleaning")
    assert r1["ok"] is True
    r2 = book(db, "Jane Doe", "5551234", "2026-07-14T09:00", "cleaning")
    assert r2["ok"] is True
    row = db.execute("SELECT COUNT(*) FROM appointments").fetchone()
    assert row[0] == 2


def test_book_allows_different_phone_same_day(db):
    r1 = book(db, "Jane Doe", "5551234", "2026-07-13T09:00", "cleaning")
    assert r1["ok"] is True
    r2 = book(db, "John Smith", "5559876", "2026-07-13T10:00", "filling")
    assert r2["ok"] is True


def test_book_allows_same_phone_same_day_after_cancel(db):
    r1 = book(db, "Jane Doe", "5551234", "2026-07-13T09:00", "cleaning")
    assert r1["ok"] is True
    c = cancel(db, r1["appointment_id"])
    assert c["ok"] is True
    r2 = book(db, "Jane Doe", "5551234", "2026-07-13T10:00", "cleaning")
    assert r2["ok"] is True


def test_reschedule_unaffected_by_same_day_guard(db):
    r1 = book(db, "Jane Doe", "5551234", "2026-07-13T09:00", "cleaning")
    assert r1["ok"] is True
    res = reschedule(db, r1["appointment_id"], "2026-07-13T10:00")
    assert res == {"ok": True, "slot_iso": "2026-07-13T10:00"}
    row = db.execute(
        "SELECT COUNT(*) FROM appointments WHERE status = 'booked'"
    ).fetchone()
    assert row[0] == 1
    row2 = db.execute(
        "SELECT slot_iso FROM appointments WHERE status = 'booked'"
    ).fetchone()
    assert row2[0] == "2026-07-13T10:00"
