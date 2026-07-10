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


def book(
    conn: sqlite3.Connection,
    patient_name: str,
    phone: str,
    slot_iso: str,
    reason: str,
) -> dict:
    day_iso = slot_iso.split("T")[0]
    if slot_iso not in find_slots(conn, day_iso):
        return {"ok": False, "error": "slot_unavailable"}
    cur = conn.execute(
        "INSERT INTO appointments (patient_name, phone, slot_iso, reason, status) "
        "VALUES (?, ?, ?, ?, 'booked')",
        (patient_name, phone, slot_iso, reason),
    )
    conn.commit()
    return {"ok": True, "appointment_id": cur.lastrowid, "slot_iso": slot_iso}


def lookup_appt(
    conn: sqlite3.Connection,
    name: str | None = None,
    phone: str | None = None,
) -> list[dict]:
    if not name and not phone:
        return []
    clauses = ["status = 'booked'"]
    params: list = []
    if name:
        clauses.append("LOWER(patient_name) LIKE ?")
        params.append(f"%{name.lower()}%")
    if phone:
        clauses.append("phone = ?")
        params.append(phone)
    sql = (
        "SELECT id, patient_name, phone, slot_iso, reason "
        "FROM appointments WHERE " + " AND ".join(clauses) + " ORDER BY slot_iso"
    )
    return [
        {
            "appointment_id": r[0],
            "patient_name": r[1],
            "phone": r[2],
            "slot_iso": r[3],
            "reason": r[4],
        }
        for r in conn.execute(sql, params)
    ]
