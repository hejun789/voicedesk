from voicedesk.tools import find_slots


def test_find_slots_weekday_all_open(db):
    # 2026-07-13 is a Monday
    slots = find_slots(db, "2026-07-13")
    assert slots[0] == "2026-07-13T09:00"
    assert slots[-1] == "2026-07-13T16:00"
    assert len(slots) == 8  # 09..16 inclusive


def test_find_slots_excludes_booked(db):
    db.execute(
        "INSERT INTO appointments (patient_name, phone, slot_iso, reason, status) "
        "VALUES ('A', '111', '2026-07-13T09:00', 'checkup', 'booked')"
    )
    db.commit()
    slots = find_slots(db, "2026-07-13")
    assert "2026-07-13T09:00" not in slots
    assert len(slots) == 7


def test_find_slots_weekend_closed(db):
    # 2026-07-11 is a Saturday
    assert find_slots(db, "2026-07-11") == []


def test_find_slots_ignores_cancelled(db):
    db.execute(
        "INSERT INTO appointments (patient_name, phone, slot_iso, reason, status) "
        "VALUES ('A', '111', '2026-07-13T09:00', 'checkup', 'cancelled')"
    )
    db.commit()
    slots = find_slots(db, "2026-07-13")
    assert "2026-07-13T09:00" in slots  # cancelled frees the slot
