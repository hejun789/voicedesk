from voicedesk.tools import book, find_slots, lookup_appt, reschedule


def test_book_normalizes_seconds(db):
    res = book(db, "Jane Doe", "5551234", "2026-07-13T09:00:00", "cleaning")
    assert res["ok"] is True
    row = db.execute(
        "SELECT slot_iso FROM appointments WHERE id = ?", (res["appointment_id"],)
    ).fetchone()
    assert row[0] == "2026-07-13T09:00"


def test_book_normalizes_space_separator(db):
    res = book(db, "Jane Doe", "5551234", "2026-07-13 09:00", "cleaning")
    assert res["ok"] is True
    row = db.execute(
        "SELECT slot_iso FROM appointments WHERE id = ?", (res["appointment_id"],)
    ).fetchone()
    assert row[0] == "2026-07-13T09:00"


def test_book_rejects_non_hour_minutes_without_storing(db):
    res = book(db, "Jane Doe", "5551234", "2026-07-13T09:30", "cleaning")
    assert res == {"ok": False, "error": "slot_unavailable"}
    row = db.execute("SELECT COUNT(*) FROM appointments").fetchone()
    assert row[0] == 0


def test_book_rejects_unparseable_date_without_raising(db):
    res = book(db, "Jane Doe", "5551234", "not a date", "cleaning")
    assert res == {"ok": False, "error": "slot_unavailable"}


def test_find_slots_normalizes_full_timestamp(db):
    assert find_slots(db, "2026-07-13T10:00:00") == find_slots(db, "2026-07-13")


def test_find_slots_returns_empty_for_garbage(db):
    assert find_slots(db, "garbage") == []


def test_reschedule_normalizes_seconds(db):
    booked = book(db, "Jane Doe", "5551234", "2026-07-13T09:00", "cleaning")
    res = reschedule(db, booked["appointment_id"], "2026-07-13T11:00:00")
    assert res["ok"] is True
    row = db.execute(
        "SELECT slot_iso FROM appointments WHERE id = ?", (booked["appointment_id"],)
    ).fetchone()
    assert row[0] == "2026-07-13T11:00"


def test_lookup_appt_ignores_placeholders_for_both_fields(db):
    book(db, "Jane Doe", "5551234", "2026-07-13T09:00", "cleaning")
    assert lookup_appt(db, name="Your Name", phone="Your Phone Number") == []


def test_lookup_appt_ignores_placeholder_name_honors_real_phone(db):
    book(db, "Jane Doe", "5551234", "2026-07-13T09:00", "cleaning")
    results = lookup_appt(db, name="Your Name", phone="5551234")
    assert len(results) == 1
    assert results[0]["phone"] == "5551234"
