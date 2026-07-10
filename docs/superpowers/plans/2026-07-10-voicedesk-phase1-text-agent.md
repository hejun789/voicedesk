# VoiceDesk Phase 1 — Text Agent + Tools + SQLite — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a working, text-only clinic booking agent — an LLM that uses tools to book, reschedule, cancel, and look up appointments against a SQLite calendar, answer FAQs, and escalate when unsure.

**Architecture:** Pure-function tools operate over an injectable SQLite connection so they are unit-testable with an in-memory DB. An agent core takes a transcript + conversation history, calls an LLM with tool schemas, executes the chosen tool, and returns a text reply. The LLM is accessed through a small `LLMClient` protocol so tests use a `FakeLLM` and production uses Groq. No audio in this phase — everything is text in / text out, which is exactly what the eval harness (Phase 2) will drive.

**Tech Stack:** Python 3.11+, `groq` SDK (OpenAI-compatible tool calling), `python-dotenv`, `pytest`, stdlib `sqlite3`.

## Global Constraints

- **Cost $0:** use only Groq free tier (LLM + Whisper later). No paid APIs. Copied from spec.
- **Single clinic, simulated calendar.** No payments, no real EHR, no multi-tenant, no auth. (YAGNI.)
- **Component boundaries stay swappable:** tools know nothing about the LLM; the agent core knows nothing about audio; the LLM is behind the `LLMClient` protocol.
- **Clinic hours (fixed):** weekdays only, hourly slots 09:00–16:00 inclusive (last slot 16:00), local naive time. Weekends closed.
- **TDD:** every behavior gets a failing test first. Commit after each green task.

---

### Task 1: Project scaffold, dependencies, and SQLite schema

**Files:**
- Create: `requirements.txt`
- Create: `.gitignore`
- Create: `src/voicedesk/__init__.py` (empty)
- Create: `src/voicedesk/db.py`
- Create: `tests/__init__.py` (empty)
- Create: `tests/conftest.py`
- Test: `tests/test_db.py`

**Interfaces:**
- Produces: `init_db(conn: sqlite3.Connection) -> None` — creates the `appointments` table.
  Schema: `appointments(id INTEGER PK, patient_name TEXT, phone TEXT, slot_iso TEXT UNIQUE, reason TEXT, status TEXT)` where `status` ∈ {`booked`, `cancelled`}. `slot_iso` is an ISO-8601 string like `"2026-07-13T09:00"`.
- Produces: `tests/conftest.py` fixture `db()` yielding an in-memory `sqlite3.Connection` with `init_db` already applied and `row_factory = sqlite3.Row`.

- [ ] **Step 1: Write `requirements.txt`**

```
groq==0.11.0
python-dotenv==1.0.1
pytest==8.3.2
```

- [ ] **Step 2: Write `.gitignore`**

```
__pycache__/
*.pyc
.env
.venv/
*.db
.pytest_cache/
```

- [ ] **Step 3: Create empty package files**

Create `src/voicedesk/__init__.py` and `tests/__init__.py` as empty files.

- [ ] **Step 4: Write the failing test** in `tests/test_db.py`

```python
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
```

- [ ] **Step 5: Run test to verify it fails**

Run: `cd /c/Users/12470/voicedesk && PYTHONPATH=src python -m pytest tests/test_db.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'voicedesk.db'`

- [ ] **Step 6: Write minimal implementation** in `src/voicedesk/db.py`

```python
import sqlite3

SCHEMA = """
CREATE TABLE IF NOT EXISTS appointments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_name TEXT NOT NULL,
    phone TEXT NOT NULL,
    slot_iso TEXT NOT NULL UNIQUE,
    reason TEXT,
    status TEXT NOT NULL DEFAULT 'booked'
);
"""


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()
```

- [ ] **Step 7: Write `tests/conftest.py`**

```python
import sqlite3
import pytest
from voicedesk.db import init_db


@pytest.fixture
def db():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    init_db(conn)
    yield conn
    conn.close()
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `PYTHONPATH=src python -m pytest tests/test_db.py -v`
Expected: PASS (2 passed)

- [ ] **Step 9: Commit**

```bash
git add requirements.txt .gitignore src tests
git commit -m "feat: project scaffold + appointments schema"
```

---

### Task 2: `find_slots` tool

**Files:**
- Create: `src/voicedesk/tools.py`
- Test: `tests/test_tools_find_slots.py`

**Interfaces:**
- Consumes: `db()` fixture from Task 1; `appointments` schema.
- Produces: `find_slots(conn, day_iso: str) -> list[str]` — returns open slot ISO strings for the given date (`"2026-07-13"`). Open = clinic hours (weekdays 09:00–16:00 hourly) minus slots with a `booked` appointment. Returns `[]` for weekends. Sorted ascending.

- [ ] **Step 1: Write the failing test** in `tests/test_tools_find_slots.py`

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m pytest tests/test_tools_find_slots.py -v`
Expected: FAIL with `ModuleNotFoundError` or `ImportError: cannot import name 'find_slots'`

- [ ] **Step 3: Write minimal implementation** in `src/voicedesk/tools.py`

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src python -m pytest tests/test_tools_find_slots.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/voicedesk/tools.py tests/test_tools_find_slots.py
git commit -m "feat: find_slots tool"
```

---

### Task 3: `book` tool with double-booking prevention

**Files:**
- Modify: `src/voicedesk/tools.py`
- Test: `tests/test_tools_book.py`

**Interfaces:**
- Consumes: `find_slots`, `appointments` schema.
- Produces: `book(conn, patient_name: str, phone: str, slot_iso: str, reason: str) -> dict`.
  Success: `{"ok": True, "appointment_id": int, "slot_iso": str}`.
  Failure (slot not open — taken or outside hours): `{"ok": False, "error": "slot_unavailable"}`.

- [ ] **Step 1: Write the failing test** in `tests/test_tools_book.py`

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m pytest tests/test_tools_book.py -v`
Expected: FAIL with `ImportError: cannot import name 'book'`

- [ ] **Step 3: Write minimal implementation** — append to `src/voicedesk/tools.py`

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src python -m pytest tests/test_tools_book.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/voicedesk/tools.py tests/test_tools_book.py
git commit -m "feat: book tool with double-booking prevention"
```

---

### Task 4: `lookup_appt` tool

**Files:**
- Modify: `src/voicedesk/tools.py`
- Test: `tests/test_tools_lookup.py`

**Interfaces:**
- Consumes: `book`, `appointments` schema.
- Produces: `lookup_appt(conn, name: str | None = None, phone: str | None = None) -> list[dict]`.
  Returns `booked` appointments matching name (case-insensitive substring) and/or phone (exact). Each dict: `{"appointment_id": int, "patient_name": str, "phone": str, "slot_iso": str, "reason": str}`. Empty list if no match or no criteria given.

- [ ] **Step 1: Write the failing test** in `tests/test_tools_lookup.py`

```python
from voicedesk.tools import lookup_appt, book


def test_lookup_by_phone(db):
    book(db, "Jane Doe", "5551234", "2026-07-13T09:00", "cleaning")
    res = lookup_appt(db, phone="5551234")
    assert len(res) == 1
    assert res[0]["patient_name"] == "Jane Doe"
    assert res[0]["slot_iso"] == "2026-07-13T09:00"


def test_lookup_by_name_case_insensitive(db):
    book(db, "Jane Doe", "5551234", "2026-07-13T09:00", "cleaning")
    res = lookup_appt(db, name="jane")
    assert len(res) == 1


def test_lookup_no_criteria_returns_empty(db):
    book(db, "Jane Doe", "5551234", "2026-07-13T09:00", "cleaning")
    assert lookup_appt(db) == []


def test_lookup_no_match_returns_empty(db):
    book(db, "Jane Doe", "5551234", "2026-07-13T09:00", "cleaning")
    assert lookup_appt(db, phone="0000000") == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m pytest tests/test_tools_lookup.py -v`
Expected: FAIL with `ImportError: cannot import name 'lookup_appt'`

- [ ] **Step 3: Write minimal implementation** — append to `src/voicedesk/tools.py`

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src python -m pytest tests/test_tools_lookup.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/voicedesk/tools.py tests/test_tools_lookup.py
git commit -m "feat: lookup_appt tool"
```

---

### Task 5: `cancel` and `reschedule` tools

**Files:**
- Modify: `src/voicedesk/tools.py`
- Test: `tests/test_tools_cancel_reschedule.py`

**Interfaces:**
- Consumes: `book`, `find_slots`, `appointments` schema.
- Produces:
  - `cancel(conn, appointment_id: int) -> dict` — sets status to `cancelled`. Success `{"ok": True}`; if no such booked appointment `{"ok": False, "error": "not_found"}`.
  - `reschedule(conn, appointment_id: int, new_slot_iso: str) -> dict` — moves a booked appointment to a new open slot. Success `{"ok": True, "slot_iso": new_slot_iso}`; `{"ok": False, "error": "not_found"}` if appointment missing; `{"ok": False, "error": "slot_unavailable"}` if new slot not open.

- [ ] **Step 1: Write the failing test** in `tests/test_tools_cancel_reschedule.py`

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m pytest tests/test_tools_cancel_reschedule.py -v`
Expected: FAIL with `ImportError: cannot import name 'cancel'`

- [ ] **Step 3: Write minimal implementation** — append to `src/voicedesk/tools.py`

```python
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
    day_iso = new_slot_iso.split("T")[0]
    if new_slot_iso not in find_slots(conn, day_iso):
        return {"ok": False, "error": "slot_unavailable"}
    conn.execute(
        "UPDATE appointments SET slot_iso = ? WHERE id = ?",
        (new_slot_iso, appointment_id),
    )
    conn.commit()
    return {"ok": True, "slot_iso": new_slot_iso}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src python -m pytest tests/test_tools_cancel_reschedule.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add src/voicedesk/tools.py tests/test_tools_cancel_reschedule.py
git commit -m "feat: cancel and reschedule tools"
```

---

### Task 6: `answer_faq` — lightweight keyword RAG

**Files:**
- Create: `clinic_info.md`
- Create: `src/voicedesk/faq.py`
- Test: `tests/test_faq.py`

**Interfaces:**
- Produces: `answer_faq(query: str, doc_path: str = "clinic_info.md") -> str`.
  Splits the doc into `## `-delimited sections, scores each by word-overlap with the query, returns the best-matching section body. Returns the literal string `"NO_MATCH"` when the top score is 0 (so the agent can escalate).

- [ ] **Step 1: Write `clinic_info.md`**

```markdown
## Hours
We are open Monday through Friday, 9am to 5pm. We are closed on weekends and public holidays.

## Location
BrightSmile Dental is at 200 Market Street, Suite 4, Springfield. Parking is free behind the building.

## Insurance
We accept Delta Dental, Cigna, MetLife, and Aetna. We also offer self-pay plans. Please bring your insurance card.

## Services
We offer cleanings, fillings, crowns, teeth whitening, and emergency dental care.
```

- [ ] **Step 2: Write the failing test** in `tests/test_faq.py`

```python
from voicedesk.faq import answer_faq


def test_faq_hours(tmp_path):
    doc = tmp_path / "info.md"
    doc.write_text(
        "## Hours\nOpen Monday to Friday 9am to 5pm.\n\n"
        "## Location\nWe are at 200 Market Street.\n"
    )
    ans = answer_faq("what are your opening hours", str(doc))
    assert "Monday to Friday" in ans


def test_faq_location(tmp_path):
    doc = tmp_path / "info.md"
    doc.write_text(
        "## Hours\nOpen Monday to Friday 9am to 5pm.\n\n"
        "## Location\nWe are at 200 Market Street.\n"
    )
    ans = answer_faq("where are you located", str(doc))
    assert "Market Street" in ans


def test_faq_no_match_returns_sentinel(tmp_path):
    doc = tmp_path / "info.md"
    doc.write_text("## Hours\nOpen Monday to Friday.\n")
    assert answer_faq("do you sell airplane tickets", str(doc)) == "NO_MATCH"
```

- [ ] **Step 3: Run test to verify it fails**

Run: `PYTHONPATH=src python -m pytest tests/test_faq.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'voicedesk.faq'`

- [ ] **Step 4: Write minimal implementation** in `src/voicedesk/faq.py`

```python
import re

_STOP = {"what", "are", "your", "the", "is", "do", "you", "a", "an", "to", "of", "we"}


def _tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]+", text.lower()) if w not in _STOP}


def _sections(doc: str) -> list[tuple[str, str]]:
    parts = re.split(r"^##\s+", doc, flags=re.MULTILINE)
    out = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        title, _, body = part.partition("\n")
        out.append((title.strip(), body.strip()))
    return out


def answer_faq(query: str, doc_path: str = "clinic_info.md") -> str:
    with open(doc_path, encoding="utf-8") as f:
        doc = f.read()
    q = _tokens(query)
    best_body, best_score = "NO_MATCH", 0
    for title, body in _sections(doc):
        score = len(q & _tokens(title + " " + body))
        if score > best_score:
            best_body, best_score = body, score
    return best_body if best_score > 0 else "NO_MATCH"
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `PYTHONPATH=src python -m pytest tests/test_faq.py -v`
Expected: PASS (3 passed)

- [ ] **Step 6: Commit**

```bash
git add clinic_info.md src/voicedesk/faq.py tests/test_faq.py
git commit -m "feat: answer_faq keyword retrieval with NO_MATCH sentinel"
```

---

### Task 7: Tool schemas + dispatcher for LLM tool-calling

**Files:**
- Create: `src/voicedesk/registry.py`
- Test: `tests/test_registry.py`

**Interfaces:**
- Consumes: all tools from `tools.py`, `answer_faq` from `faq.py`.
- Produces:
  - `TOOL_SCHEMAS: list[dict]` — OpenAI/Groq-format function schemas for: `find_slots`, `book`, `reschedule`, `cancel`, `lookup_appt`, `answer_faq`, `escalate`. Each has `{"type": "function", "function": {"name", "description", "parameters"}}`.
  - `dispatch(conn, name: str, args: dict) -> dict` — routes a tool call to the matching function and returns its result as a dict. `escalate` returns `{"ok": True, "escalated": True, "reason": args.get("reason", "")}`. `answer_faq` returns `{"answer": <str>}`. Unknown name returns `{"ok": False, "error": "unknown_tool"}`.

- [ ] **Step 1: Write the failing test** in `tests/test_registry.py`

```python
from voicedesk.registry import TOOL_SCHEMAS, dispatch


def test_schemas_cover_expected_tools():
    names = {s["function"]["name"] for s in TOOL_SCHEMAS}
    assert names == {
        "find_slots", "book", "reschedule", "cancel",
        "lookup_appt", "answer_faq", "escalate",
    }


def test_dispatch_book(db):
    res = dispatch(db, "book", {
        "patient_name": "Jane", "phone": "5551234",
        "slot_iso": "2026-07-13T09:00", "reason": "cleaning",
    })
    assert res["ok"] is True


def test_dispatch_escalate():
    res = dispatch(None, "escalate", {"reason": "angry caller"})
    assert res == {"ok": True, "escalated": True, "reason": "angry caller"}


def test_dispatch_faq(tmp_path):
    doc = tmp_path / "info.md"
    doc.write_text("## Hours\nOpen Monday to Friday.\n")
    res = dispatch(None, "answer_faq", {"query": "hours", "doc_path": str(doc)})
    assert "Monday" in res["answer"]


def test_dispatch_unknown():
    assert dispatch(None, "nope", {}) == {"ok": False, "error": "unknown_tool"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m pytest tests/test_registry.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'voicedesk.registry'`

- [ ] **Step 3: Write minimal implementation** in `src/voicedesk/registry.py`

```python
from voicedesk import tools
from voicedesk.faq import answer_faq


def _fn(name, description, properties, required):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


TOOL_SCHEMAS = [
    _fn("find_slots", "List open appointment slots for a date (YYYY-MM-DD).",
        {"day_iso": {"type": "string", "description": "Date as YYYY-MM-DD"}},
        ["day_iso"]),
    _fn("book", "Book an appointment in an open slot.",
        {"patient_name": {"type": "string"}, "phone": {"type": "string"},
         "slot_iso": {"type": "string", "description": "YYYY-MM-DDTHH:00"},
         "reason": {"type": "string"}},
        ["patient_name", "phone", "slot_iso", "reason"]),
    _fn("reschedule", "Move an existing appointment to a new open slot.",
        {"appointment_id": {"type": "integer"},
         "new_slot_iso": {"type": "string", "description": "YYYY-MM-DDTHH:00"}},
        ["appointment_id", "new_slot_iso"]),
    _fn("cancel", "Cancel an existing appointment by id.",
        {"appointment_id": {"type": "integer"}}, ["appointment_id"]),
    _fn("lookup_appt", "Find a patient's booked appointments by name and/or phone.",
        {"name": {"type": "string"}, "phone": {"type": "string"}}, []),
    _fn("answer_faq", "Answer a general clinic question (hours, location, insurance).",
        {"query": {"type": "string"}}, ["query"]),
    _fn("escalate", "Hand off to a human when unable to help confidently.",
        {"reason": {"type": "string"}}, ["reason"]),
]


def dispatch(conn, name: str, args: dict) -> dict:
    if name == "find_slots":
        return {"slots": tools.find_slots(conn, args["day_iso"])}
    if name == "book":
        return tools.book(conn, args["patient_name"], args["phone"],
                          args["slot_iso"], args["reason"])
    if name == "reschedule":
        return tools.reschedule(conn, args["appointment_id"], args["new_slot_iso"])
    if name == "cancel":
        return tools.cancel(conn, args["appointment_id"])
    if name == "lookup_appt":
        return {"results": tools.lookup_appt(conn, args.get("name"), args.get("phone"))}
    if name == "answer_faq":
        kwargs = {"doc_path": args["doc_path"]} if "doc_path" in args else {}
        return {"answer": answer_faq(args["query"], **kwargs)}
    if name == "escalate":
        return {"ok": True, "escalated": True, "reason": args.get("reason", "")}
    return {"ok": False, "error": "unknown_tool"}
```

Note: `find_slots` returns a bare list from `tools.py`; the test `test_dispatch_book` and others above don't assert on `find_slots`'s dispatch shape, but the agent (Task 8) relies on `dispatch` returning a dict, so it is wrapped as `{"slots": [...]}` here.

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src python -m pytest tests/test_registry.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add src/voicedesk/registry.py tests/test_registry.py
git commit -m "feat: tool schemas + dispatcher for LLM tool-calling"
```

---

### Task 8: Agent core (tool-calling loop + escalation), tested with a FakeLLM

**Files:**
- Create: `src/voicedesk/llm.py`
- Create: `src/voicedesk/agent.py`
- Test: `tests/test_agent.py`

**Interfaces:**
- Consumes: `TOOL_SCHEMAS`, `dispatch` from `registry.py`.
- Produces:
  - `llm.py`: `LLMClient` protocol with `complete(messages: list[dict], tools: list[dict]) -> Message`, where `Message` is a dataclass `Message(content: str | None, tool_calls: list[ToolCall])` and `ToolCall(id: str, name: str, arguments: dict)`. Also `FakeLLM(scripted: list[Message])` that pops one scripted `Message` per `complete` call.
  - `agent.py`: `Agent(conn, llm: LLMClient, system_prompt: str = DEFAULT_SYSTEM_PROMPT)` with `respond(user_text: str) -> str`. It appends the user turn, loops: call `llm.complete`; if the message has tool_calls, dispatch each, append a `role="tool"` result message, and loop again (max 5 iterations); if it has content, append and return it. Conversation history persists across `respond` calls on the same `Agent`.
  - `DEFAULT_SYSTEM_PROMPT: str` instructing the model to use tools, confirm details before booking, and call `escalate` when unsure.

- [ ] **Step 1: Write the failing test** in `tests/test_agent.py`

```python
from voicedesk.llm import FakeLLM, Message, ToolCall
from voicedesk.agent import Agent


def test_agent_returns_plain_text(db):
    llm = FakeLLM([Message(content="Hello! How can I help?", tool_calls=[])])
    agent = Agent(db, llm)
    assert agent.respond("hi") == "Hello! How can I help?"


def test_agent_executes_tool_then_replies(db):
    llm = FakeLLM([
        Message(content=None, tool_calls=[
            ToolCall(id="1", name="book", arguments={
                "patient_name": "Jane", "phone": "5551234",
                "slot_iso": "2026-07-13T09:00", "reason": "cleaning"})]),
        Message(content="You're booked for Monday at 9am.", tool_calls=[]),
    ])
    agent = Agent(db, llm)
    reply = agent.respond("book me monday 9am, Jane, 5551234, cleaning")
    assert "booked" in reply.lower()
    # side effect really happened:
    from voicedesk.tools import lookup_appt
    assert lookup_appt(db, phone="5551234")[0]["slot_iso"] == "2026-07-13T09:00"


def test_agent_stops_at_iteration_cap(db):
    # LLM always asks for a tool, never returns text -> loop must terminate.
    looping = [
        Message(content=None, tool_calls=[
            ToolCall(id="x", name="find_slots", arguments={"day_iso": "2026-07-13"})])
        for _ in range(10)
    ]
    agent = Agent(db, FakeLLM(looping))
    reply = agent.respond("slots?")
    assert isinstance(reply, str) and len(reply) > 0  # returns a fallback, no crash
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m pytest tests/test_agent.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'voicedesk.llm'`

- [ ] **Step 3: Write `src/voicedesk/llm.py`**

```python
from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class Message:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)


class LLMClient(Protocol):
    def complete(self, messages: list[dict], tools: list[dict]) -> Message: ...


class FakeLLM:
    """Test double: returns scripted messages in order."""

    def __init__(self, scripted: list[Message]):
        self._scripted = list(scripted)

    def complete(self, messages: list[dict], tools: list[dict]) -> Message:
        return self._scripted.pop(0)
```

- [ ] **Step 4: Write `src/voicedesk/agent.py`**

```python
import json
from voicedesk.llm import LLMClient, Message
from voicedesk.registry import TOOL_SCHEMAS, dispatch

DEFAULT_SYSTEM_PROMPT = (
    "You are the phone receptionist for BrightSmile Dental. "
    "Use the provided tools to find slots and to book, reschedule, cancel, or "
    "look up appointments, and to answer general questions. "
    "Always confirm the patient's name, phone, and desired time before booking. "
    "If a tool reports slot_unavailable, offer other open slots. "
    "If you cannot help confidently, or input is unclear or out of scope, call the "
    "escalate tool. Keep replies short and natural, as if speaking on a phone call."
)

MAX_ITERS = 5
_FALLBACK = (
    "I'm having trouble with that. Let me have a team member call you back."
)


class Agent:
    def __init__(self, conn, llm: LLMClient, system_prompt: str = DEFAULT_SYSTEM_PROMPT):
        self.conn = conn
        self.llm = llm
        self.messages: list[dict] = [{"role": "system", "content": system_prompt}]

    def respond(self, user_text: str) -> str:
        self.messages.append({"role": "user", "content": user_text})
        for _ in range(MAX_ITERS):
            msg: Message = self.llm.complete(self.messages, TOOL_SCHEMAS)
            if not msg.tool_calls:
                text = msg.content or _FALLBACK
                self.messages.append({"role": "assistant", "content": text})
                return text
            self.messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)}}
                    for tc in msg.tool_calls
                ],
            })
            for tc in msg.tool_calls:
                result = dispatch(self.conn, tc.name, tc.arguments)
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result),
                })
        self.messages.append({"role": "assistant", "content": _FALLBACK})
        return _FALLBACK
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `PYTHONPATH=src python -m pytest tests/test_agent.py -v`
Expected: PASS (3 passed)

- [ ] **Step 6: Commit**

```bash
git add src/voicedesk/llm.py src/voicedesk/agent.py tests/test_agent.py
git commit -m "feat: agent core tool-calling loop with iteration cap"
```

---

### Task 9: Groq LLM adapter + text REPL entrypoint

**Files:**
- Create: `src/voicedesk/groq_client.py`
- Create: `src/voicedesk/cli.py`
- Create: `.env.example`
- Create: `README.md`
- Test: `tests/test_groq_client.py`

**Interfaces:**
- Consumes: `Message`, `ToolCall` from `llm.py`; `TOOL_SCHEMAS`.
- Produces:
  - `groq_client.py`: `GroqLLM(model: str = "llama-3.3-70b-versatile", api_key: str | None = None)` implementing `complete`. It calls Groq chat completions with `tools=` and `tool_choice="auto"`, then maps the response into our `Message`/`ToolCall` dataclasses. A module function `_to_message(choice) -> Message` is unit-tested against a hand-built fake response object (no network).
  - `cli.py`: `main()` — loads `.env`, opens/creates `voicedesk.db`, seeds schema, builds `Agent(conn, GroqLLM())`, and runs a read-eval-print loop reading `you>` lines and printing `agent>` replies until `quit`.

- [ ] **Step 1: Write the failing test** in `tests/test_groq_client.py`

```python
from types import SimpleNamespace
from voicedesk.groq_client import _to_message


def _fake_choice(content=None, tool_calls=None):
    msg = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(message=msg)


def test_to_message_plain_text():
    msg = _to_message(_fake_choice(content="hi there"))
    assert msg.content == "hi there"
    assert msg.tool_calls == []


def test_to_message_with_tool_call():
    tc = SimpleNamespace(
        id="call_1",
        function=SimpleNamespace(name="cancel", arguments='{"appointment_id": 3}'),
    )
    msg = _to_message(_fake_choice(content=None, tool_calls=[tc]))
    assert msg.tool_calls[0].name == "cancel"
    assert msg.tool_calls[0].arguments == {"appointment_id": 3}
    assert msg.tool_calls[0].id == "call_1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m pytest tests/test_groq_client.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'voicedesk.groq_client'`

- [ ] **Step 3: Write `src/voicedesk/groq_client.py`**

```python
import json
import os
from voicedesk.llm import Message, ToolCall
from voicedesk.registry import TOOL_SCHEMAS


def _to_message(choice) -> Message:
    m = choice.message
    calls = []
    for tc in (m.tool_calls or []):
        calls.append(ToolCall(
            id=tc.id,
            name=tc.function.name,
            arguments=json.loads(tc.function.arguments or "{}"),
        ))
    return Message(content=m.content, tool_calls=calls)


class GroqLLM:
    def __init__(self, model: str = "llama-3.3-70b-versatile", api_key: str | None = None):
        from groq import Groq  # imported lazily so tests don't need the package
        self.model = model
        self.client = Groq(api_key=api_key or os.environ["GROQ_API_KEY"])

    def complete(self, messages: list[dict], tools: list[dict]) -> Message:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        return _to_message(resp.choices[0])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src python -m pytest tests/test_groq_client.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Write `src/voicedesk/cli.py`**

```python
import sqlite3
from dotenv import load_dotenv
from voicedesk.db import init_db
from voicedesk.agent import Agent
from voicedesk.groq_client import GroqLLM


def main() -> None:
    load_dotenv()
    conn = sqlite3.connect("voicedesk.db")
    conn.row_factory = sqlite3.Row
    init_db(conn)
    agent = Agent(conn, GroqLLM())
    print("VoiceDesk (text mode). Type 'quit' to exit.")
    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user.lower() in {"quit", "exit"}:
            break
        if not user:
            continue
        print("agent>", agent.respond(user))


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Write `.env.example`**

```
GROQ_API_KEY=your_free_groq_key_here
```

- [ ] **Step 7: Write `README.md`**

```markdown
# VoiceDesk — AI Voice Receptionist for Clinics

An AI agent that books, reschedules, and cancels clinic appointments and answers
FAQs. Phase 1 is text-only; voice (STT/TTS) and deployment come in later phases.

## Why
Clinics miss 30%+ of inbound calls — each is a potential lost patient. VoiceDesk is
a 24/7 receptionist that takes real booking actions, not just chat.

## Run (Phase 1, text mode)
1. `python -m venv .venv && source .venv/Scripts/activate` (Windows Git Bash)
2. `pip install -r requirements.txt`
3. Get a free API key at https://console.groq.com and copy `.env.example` to `.env`.
4. `PYTHONPATH=src python -m voicedesk.cli`

## Test
`PYTHONPATH=src python -m pytest -v`

## Architecture
Browser/CLI → agent core (LLM + tool calling) → tools over SQLite calendar.
Tools, agent, and LLM provider are cleanly separated so STT/TTS and Twilio can be
added without touching the booking logic.
```

- [ ] **Step 8: Run the full test suite**

Run: `PYTHONPATH=src python -m pytest -v`
Expected: PASS (all tests from Tasks 1–9 green)

- [ ] **Step 9: Manual smoke test (requires a free Groq key in `.env`)**

Run: `PYTHONPATH=src python -m voicedesk.cli`
Try: "What are your hours?" then "Book me for Monday July 13th at 9am, name Jane Doe, phone 5551234, for a cleaning." then "Actually cancel that."
Expected: natural replies; appointment appears then disappears (verify with a second run or a `sqlite3 voicedesk.db "SELECT * FROM appointments;"`).

- [ ] **Step 10: Commit**

```bash
git add src/voicedesk/groq_client.py src/voicedesk/cli.py .env.example README.md tests/test_groq_client.py
git commit -m "feat: Groq adapter + text REPL entrypoint + README"
```

---

## Phase 1 Definition of Done

- All unit tests pass (`PYTHONPATH=src python -m pytest -v`).
- `python -m voicedesk.cli` runs a real text conversation that books/reschedules/cancels/looks-up appointments and answers FAQs against SQLite.
- Tools, agent core, and LLM provider are cleanly separated (verified by tests using `FakeLLM` with no network).
- Escalation path exists (`escalate` tool + agent fallback on iteration cap).

## What comes next (separate plans)

- **Phase 2 — Eval harness:** ~30 scripted scenarios driving `Agent` with a real Groq LLM (or recorded transcripts), scored on correct tool / correct slot / correct escalation. Reuses the text-in/text-out `Agent.respond` interface unchanged.
- **Phase 3 — Voice:** STT (Groq Whisper) + TTS (Web Speech API / Piper) + FastAPI WebSocket wrapping the same `Agent`.
- **Phase 4 — Deploy + polish:** HF Spaces/Render, latency measurement, README case study.
