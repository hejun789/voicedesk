from voicedesk.tools import book, cancel, reschedule, find_slots, lookup_appt


def test_cancel_frees_slot(db):
    res = book(db, "Jane", "5551234", "2026-07-13T09:00", "cleaning")
    aid = res["appointment_id"]
    assert cancel(db, aid) == {"ok": True}
    assert "2026-07-13T09:00" in find_slots(db, "2026-07-13")


def test_cancel_unknown_id(db):
    assert cancel(db, 999) == {"ok": False, "error": "not_found"}


def test_reschedule_moves_appointment(db):
    aid = book(db, "Jane", "5551234", "2026-07-13T09:00", "cleaning")["appointment_id"]
    res = reschedule(db, aid, "2026-07-13T10:00")
    assert res == {"ok": True, "slot_iso": "2026-07-13T10:00"}
    slots = find_slots(db, "2026-07-13")
    assert "2026-07-13T09:00" in slots
    assert "2026-07-13T10:00" not in slots


def test_reschedule_unknown_id(db):
    assert reschedule(db, 999, "2026-07-13T10:00") == {"ok": False, "error": "not_found"}


def test_reschedule_to_taken_slot(db):
    a = book(db, "Jane", "5551234", "2026-07-13T09:00", "cleaning")["appointment_id"]
    book(db, "John", "5559999", "2026-07-13T10:00", "cleaning")
    assert reschedule(db, a, "2026-07-13T10:00") == {"ok": False, "error": "slot_unavailable"}
