import json
import sqlite3

from voicedesk import tools
from voicedesk.db import init_db


def load_scenarios(path: str = "evals/scenarios.json") -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def tools_called_from(messages: list[dict]) -> list[str]:
    """Read which tools the agent called back out of its own message history.
    This is how the harness observes the agent without modifying it."""
    names: list[str] = []
    for m in messages:
        for tc in m.get("tool_calls") or []:
            names.append(tc["function"]["name"])
    return names


def fresh_db() -> sqlite3.Connection:
    """A new in-memory DB per run, so runs cannot contaminate each other."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    init_db(conn)
    return conn


def seed_db(conn: sqlite3.Connection, seed: list[dict]) -> None:
    for a in seed:
        res = tools.book(conn, a["patient_name"], a["phone"],
                         a["slot_iso"], a.get("reason", ""))
        if not res.get("ok"):
            raise ValueError(f"seed booking failed for {a}: {res}")


def all_appointments(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        "SELECT patient_name, phone, slot_iso, reason, status "
        "FROM appointments ORDER BY slot_iso"
    )
    return [
        {"patient_name": r[0], "phone": r[1], "slot_iso": r[2],
         "reason": r[3], "status": r[4]}
        for r in rows
    ]
