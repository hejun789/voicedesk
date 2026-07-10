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
