import sqlite3
from datetime import date, datetime

OPEN_HOURS = range(9, 17)  # 09:00 .. 16:00 inclusive


def _all_slots(day_iso: str) -> list[str]:
    d = date.fromisoformat(day_iso)
    if d.weekday() >= 5:  # Sat/Sun
        return []
    return [f"{day_iso}T{h:02d}:00" for h in OPEN_HOURS]


def find_slots(conn: sqlite3.Connection, day_iso: str) -> list[str]:
    candidates = _all_slots(day_iso)
    if not candidates:
        return []
    booked = {
        row[0]
        for row in conn.execute(
            "SELECT slot_iso FROM appointments "
            "WHERE status = 'booked' AND slot_iso LIKE ?",
            (f"{day_iso}T%",),
        )
    }
    return [s for s in candidates if s not in booked]
