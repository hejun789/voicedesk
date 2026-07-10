import sqlite3

SCHEMA = """
CREATE TABLE IF NOT EXISTS appointments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_name TEXT NOT NULL,
    phone TEXT NOT NULL,
    slot_iso TEXT NOT NULL,
    reason TEXT,
    status TEXT NOT NULL DEFAULT 'booked'
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_unique_booked_slot
    ON appointments(slot_iso) WHERE status = 'booked';
"""


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()
