import sqlite3
from voicedesk.db import init_db


def test_init_db_creates_appointments_table():
    conn = sqlite3.connect(":memory:")
    init_db(conn)
    cols = {row[1] for row in conn.execute("PRAGMA table_info(appointments)")}
    assert cols == {"id", "patient_name", "phone", "slot_iso", "reason", "status"}


def test_slot_iso_is_unique():
    conn = sqlite3.connect(":memory:")
    init_db(conn)
    conn.execute(
        "INSERT INTO appointments (patient_name, phone, slot_iso, reason, status) "
        "VALUES ('A', '111', '2026-07-13T09:00', 'checkup', 'booked')"
    )
    import pytest
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO appointments (patient_name, phone, slot_iso, reason, status) "
            "VALUES ('B', '222', '2026-07-13T09:00', 'checkup', 'booked')"
        )
