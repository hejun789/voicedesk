from voicedesk.tools import book, cancel, reschedule, lookup_appt


def test_book_after_cancel_same_slot(db):
    r1 = book(db, "Jane", "5551234", "2026-07-13T09:00", "cleaning")
    assert r1["ok"] is True
    c = cancel(db, r1["appointment_id"])
    assert c["ok"] is True

    r2 = book(db, "John", "5559999", "2026-07-13T09:00", "checkup")
    assert r2 == {"ok": True, "appointment_id": r2["appointment_id"], "slot_iso": "2026-07-13T09:00"}

    found = lookup_appt(db, phone="5559999")
    assert found[0]["slot_iso"] == "2026-07-13T09:00"


def test_reschedule_into_cancelled_slot(db):
    a = book(db, "Patient A", "5551111", "2026-07-13T09:00", "cleaning")
    assert a["ok"] is True
    c = cancel(db, a["appointment_id"])
    assert c["ok"] is True

    b = book(db, "Patient B", "5552222", "2026-07-13T10:00", "checkup")
    assert b["ok"] is True

    r = reschedule(db, b["appointment_id"], "2026-07-13T09:00")
    assert r == {"ok": True, "slot_iso": "2026-07-13T09:00"}


def test_two_booked_same_slot_still_blocked(db):
    r1 = book(db, "Jane", "5551234", "2026-07-13T09:00", "cleaning")
    assert r1["ok"] is True

    r2 = book(db, "John", "5559999", "2026-07-13T09:00", "checkup")
    assert r2 == {"ok": False, "error": "slot_unavailable"}
