import sqlite3
from datetime import date, datetime

OPEN_HOURS = range(9, 17)  # 09:00 .. 16:00 inclusive

_PLACEHOLDER_VALUES = {
    "", "unknown", "n/a", "na", "none", "null", "nil", "string", "tbd",
    "patient_name", "phone", "reason", "name", "patient", "caller",
    "your name", "your phone number", "your phone",
    "未知", "无", "没有", "姓名", "电话", "电话号码",
}


def _is_placeholder(value: str) -> bool:
    """True when the model supplied a filler value instead of a real one."""
    return str(value).strip().lower() in _PLACEHOLDER_VALUES


def _normalize_slot(slot_iso: str) -> str | None:
    """Coerce a model-supplied timestamp to the canonical 'YYYY-MM-DDTHH:00'.

    Accepts seconds ('...T16:00:00'), a space separator, and stray
    whitespace. Refuses (returns None) when the minute component is not
    zero, since the clinic only has hourly slots and we must not silently
    move the caller to a different time than they asked for. Returns None
    when it cannot be parsed.
    """
    try:
        dt = datetime.fromisoformat(str(slot_iso).strip())
    except (ValueError, TypeError):
        return None
    if dt.minute != 0:
        return None
    return dt.strftime("%Y-%m-%dT%H:00")


def _normalize_day(day_iso: str) -> str | None:
    """Coerce a model-supplied date to canonical 'YYYY-MM-DD'.

    Accepts a full timestamp ('2026-07-11T10:00:00') by taking its date
    part. Returns None when it cannot be parsed.
    """
    value = str(day_iso).strip()
    try:
        return datetime.fromisoformat(value).date().isoformat()
    except (ValueError, TypeError):
        pass
    try:
        return date.fromisoformat(value).isoformat()
    except (ValueError, TypeError):
        return None


def _all_slots(day_iso: str) -> list[str]:
    d = date.fromisoformat(day_iso)
    if d.weekday() >= 5:  # Sat/Sun
        return []
    return [f"{day_iso}T{h:02d}:00" for h in OPEN_HOURS]


def find_slots(conn: sqlite3.Connection, day_iso: str) -> list[str]:
    normalized_day = _normalize_day(day_iso)
    if normalized_day is None:
        return []
    day_iso = normalized_day
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
    if _is_placeholder(patient_name) or _is_placeholder(phone):
        return {"ok": False, "error": "missing_patient_details"}
    if sum(c.isdigit() for c in str(phone)) < 3:
        return {"ok": False, "error": "missing_patient_details"}
    normalized_slot = _normalize_slot(slot_iso)
    if normalized_slot is None:
        return {"ok": False, "error": "slot_unavailable"}
    slot_iso = normalized_slot
    day_iso = slot_iso.split("T")[0]
    existing = conn.execute(
        "SELECT slot_iso FROM appointments "
        "WHERE phone = ? AND status = 'booked' AND slot_iso LIKE ?",
        (phone, f"{day_iso}T%"),
    ).fetchone()
    if existing is not None:
        return {"ok": False, "error": "already_booked_that_day",
                "existing_slot_iso": existing[0]}
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
    if name is not None and _is_placeholder(name):
        name = None
    if phone is not None and _is_placeholder(phone):
        phone = None
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


def cancel(conn: sqlite3.Connection, appointment_id: int) -> dict:
    cur = conn.execute(
        "UPDATE appointments SET status = 'cancelled' "
        "WHERE id = ? AND status = 'booked'",
        (appointment_id,),
    )
    conn.commit()
    if cur.rowcount == 0:
        return {"ok": False, "error": "not_found"}
    return {"ok": True}


def reschedule(
    conn: sqlite3.Connection, appointment_id: int, new_slot_iso: str
) -> dict:
    row = conn.execute(
        "SELECT patient_name, phone, reason FROM appointments "
        "WHERE id = ? AND status = 'booked'",
        (appointment_id,),
    ).fetchone()
    if row is None:
        return {"ok": False, "error": "not_found"}
    normalized_slot = _normalize_slot(new_slot_iso)
    if normalized_slot is None:
        return {"ok": False, "error": "slot_unavailable"}
    new_slot_iso = normalized_slot
    day_iso = new_slot_iso.split("T")[0]
    if new_slot_iso not in find_slots(conn, day_iso):
        return {"ok": False, "error": "slot_unavailable"}
    conn.execute(
        "UPDATE appointments SET slot_iso = ? WHERE id = ?",
        (new_slot_iso, appointment_id),
    )
    conn.commit()
    return {"ok": True, "slot_iso": new_slot_iso}
