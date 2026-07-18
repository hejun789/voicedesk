# VoiceDesk Phase 4 — Public Deployment — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a public HTTPS link where anyone can book an appointment by voice (English or Chinese), with the Groq key kept server-side, the quota protected from strangers, and an isolated calendar per visitor.

**Architecture:** A `Dockerfile` runs the existing FastAPI app on Hugging Face Spaces (automatic HTTPS, key as a Space Secret). Each visitor session builds its own in-memory SQLite calendar, so the app persists nothing to disk. A stdlib rate limiter (per-IP + global daily caps) guards `POST /turn` and, when exceeded, returns a friendly bilingual "demo limit reached" reply instead of an error.

**Tech Stack:** Python 3.11+, existing FastAPI/uvicorn/Groq stack, Docker, Hugging Face Spaces (Docker SDK). No new Python dependencies.

## Global Constraints

- **Cost $0.** Groq free tier; HF Spaces free tier. **No new Python dependencies.**
- **The Groq key is a server-side secret** — never sent to the browser; `.env` is never in the image.
- **The deployed app persists nothing to disk** — each visitor session gets a fresh in-memory SQLite calendar (the pattern the eval harness already uses).
- **HF Docker Spaces serve on port 7860**; the app binds `0.0.0.0:$PORT` (default 7860).
- **The Space defaults to `GROQ_MODEL=llama-3.1-8b-instant`** for higher free-tier limits; the model stays an env var so switching to `gpt-oss-120b` is one Space setting.
- The rate limiter is pure stdlib with an **injectable clock**; limits come from env vars (`PER_IP_DAILY_LIMIT` default 8, `GLOBAL_DAILY_LIMIT` default 200).
- **The test suite stays fully offline** — no network, no API key, no microphone, no container.
- Tests run as `PYTHONPATH=src ./.venv/Scripts/python.exe -m pytest ...` from the repo root via Bash (Git Bash), NOT PowerShell.
- Existing suite is **256 passing** before this phase. It must stay green.
- This repo's commits are **single-author** — never add a `Co-Authored-By` trailer.
- TDD throughout; commit after each green task.

---

### Task 1: The rate limiter

**Files:**
- Create: `src/voicedesk/voice/limits.py`
- Test: `tests/test_voice_limits.py`

**Interfaces:**
- Produces:
  - `RateLimiter(per_ip_limit: int = 8, global_limit: int = 200, clock=<utc date>)` — an in-memory per-UTC-day limiter.
  - `RateLimiter.allow(ip: str) -> bool` — returns True and counts the turn when the IP is under its per-day cap AND the global per-day cap is not yet reached; returns False (without counting) when either cap is hit. Counters reset when the UTC date rolls over. `clock` is a zero-arg callable returning a `datetime.date`, injectable so day-rollover is testable without waiting.

- [ ] **Step 1: Write the failing test** — `tests/test_voice_limits.py`

```python
from datetime import date
from voicedesk.voice.limits import RateLimiter


class _Clock:
    def __init__(self, d=date(2026, 7, 16)):
        self.d = d

    def __call__(self):
        return self.d


def test_allows_up_to_the_per_ip_limit_then_blocks():
    rl = RateLimiter(per_ip_limit=3, global_limit=100, clock=_Clock())
    assert [rl.allow("1.1.1.1") for _ in range(3)] == [True, True, True]
    assert rl.allow("1.1.1.1") is False       # 4th turn from same IP blocked
    assert rl.allow("1.1.1.1") is False       # stays blocked


def test_different_ips_are_counted_separately():
    rl = RateLimiter(per_ip_limit=1, global_limit=100, clock=_Clock())
    assert rl.allow("1.1.1.1") is True
    assert rl.allow("1.1.1.1") is False
    assert rl.allow("2.2.2.2") is True         # a different IP is unaffected


def test_global_limit_stops_everyone():
    rl = RateLimiter(per_ip_limit=100, global_limit=2, clock=_Clock())
    assert rl.allow("a") is True
    assert rl.allow("b") is True
    assert rl.allow("c") is False              # global budget spent, new IP still blocked


def test_a_blocked_turn_does_not_consume_budget():
    rl = RateLimiter(per_ip_limit=1, global_limit=100, clock=_Clock())
    assert rl.allow("1.1.1.1") is True
    assert rl.allow("1.1.1.1") is False
    # the blocked attempts did not eat into the global budget:
    assert rl.allow("2.2.2.2") is True


def test_counters_reset_when_the_utc_day_rolls_over():
    clock = _Clock(date(2026, 7, 16))
    rl = RateLimiter(per_ip_limit=1, global_limit=1, clock=clock)
    assert rl.allow("1.1.1.1") is True
    assert rl.allow("1.1.1.1") is False        # spent for the day
    clock.d = date(2026, 7, 17)                # next UTC day
    assert rl.allow("1.1.1.1") is True         # fresh budget
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src ./.venv/Scripts/python.exe -m pytest tests/test_voice_limits.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'voicedesk.voice.limits'`

- [ ] **Step 3: Implement** — `src/voicedesk/voice/limits.py`

```python
from datetime import date, datetime, timezone


def _utc_today() -> date:
    return datetime.now(timezone.utc).date()


class RateLimiter:
    """Per-UTC-day turn limiter for the public demo: caps each visitor IP and
    the whole demo, so one person (or a bot) cannot drain the Groq quota.

    In-memory only — a restart resets it, which is fine for a demo. `clock`
    returns today's UTC date and is injectable so day-rollover is testable.
    """

    def __init__(self, per_ip_limit: int = 8, global_limit: int = 200,
                 clock=_utc_today):
        self.per_ip_limit = per_ip_limit
        self.global_limit = global_limit
        self._clock = clock
        self._day = clock()
        self._ip_counts: dict[str, int] = {}
        self._global = 0

    def _roll(self) -> None:
        today = self._clock()
        if today != self._day:
            self._day = today
            self._ip_counts = {}
            self._global = 0

    def allow(self, ip: str) -> bool:
        """True (and counts the turn) when under both caps; False otherwise."""
        self._roll()
        if self._global >= self.global_limit:
            return False
        if self._ip_counts.get(ip, 0) >= self.per_ip_limit:
            return False
        self._ip_counts[ip] = self._ip_counts.get(ip, 0) + 1
        self._global += 1
        return True
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src ./.venv/Scripts/python.exe -m pytest tests/test_voice_limits.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add src/voicedesk/voice/limits.py tests/test_voice_limits.py
git commit -m "feat: per-IP and global daily rate limiter for the public demo"
```

---

### Task 2: Wire the limiter into the server

**Files:**
- Modify: `src/voicedesk/voice/server.py`
- Test: `tests/test_voice_server_limits.py`

**Interfaces:**
- Consumes: `RateLimiter.allow(ip)` from `voicedesk.voice.limits`.
- Produces:
  - `create_app(stt, sessions, lock=None, limiter=None) -> FastAPI` — gains an optional `limiter`. When `None` (tests, local use without limiting) behaviour is unchanged.
  - `DEMO_LIMIT` / `DEMO_LIMIT_ZH` and the `_DEMO_LIMIT` per-language dict.
  - `POST /turn` checks `limiter.allow(client_ip)` immediately after normalizing `lang` and **before** reading the audio or calling STT/agent. Over the limit → HTTP 200 with the demo-limit reply, `"error": "rate_limited"`, and `"lang"`. The client IP is the first hop of `X-Forwarded-For` (set by HF), falling back to the socket peer.

- [ ] **Step 1: Write the failing test** — `tests/test_voice_server_limits.py`

```python
import sqlite3
import pytest
from fastapi.testclient import TestClient

from voicedesk.db import init_db
from voicedesk.agent import Agent
from voicedesk.llm import FakeLLM, Message
from voicedesk.voice.stt import FakeSTT
from voicedesk.voice.session import SessionStore
from voicedesk.voice.limits import RateLimiter
from voicedesk.voice.server import create_app, DEMO_LIMIT, DEMO_LIMIT_ZH


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:", check_same_thread=False)
    c.row_factory = sqlite3.Row
    init_db(c)
    yield c
    c.close()


def _client(conn, stt, llm, limiter):
    sessions = SessionStore(lambda lang: Agent(conn, llm))
    return TestClient(create_app(stt, sessions, limiter=limiter))


def _post(client, lang="en", xff=None):
    headers = {"X-Forwarded-For": xff} if xff else {}
    return client.post(
        "/turn",
        data={"session_id": "s1", "lang": lang},
        files={"audio": ("turn.webm", b"x" * 2000, "audio/webm")},
        headers=headers,
    )


def test_over_limit_returns_the_demo_message_without_calling_stt_or_agent(conn):
    # per_ip_limit=0 => every turn is over the limit. FakeSTT([]) / FakeLLM([])
    # would IndexError if reached, proving neither is called.
    limiter = RateLimiter(per_ip_limit=0, global_limit=100)
    body = _post(_client(conn, FakeSTT([]), FakeLLM([]), limiter)).json()
    assert body["reply"] == DEMO_LIMIT
    assert body["error"] == "rate_limited"
    assert body["lang"] == "en"


def test_over_limit_message_is_localized(conn):
    limiter = RateLimiter(per_ip_limit=0, global_limit=100)
    body = _post(_client(conn, FakeSTT([]), FakeLLM([]), limiter), lang="zh").json()
    assert body["reply"] == DEMO_LIMIT_ZH
    assert body["lang"] == "zh"


def test_under_the_limit_the_turn_is_processed_normally(conn):
    limiter = RateLimiter(per_ip_limit=5, global_limit=100)
    stt = FakeSTT(["what are your hours"])
    llm = FakeLLM([Message(content="Weekdays 9 to 5.", tool_calls=[])])
    body = _post(_client(conn, stt, llm, limiter)).json()
    assert body["reply"] == "Weekdays 9 to 5."


def test_per_ip_limit_blocks_the_next_turn_from_the_same_ip(conn):
    limiter = RateLimiter(per_ip_limit=1, global_limit=100)
    stt = FakeSTT(["hi"])                    # only the first (allowed) turn reaches STT
    llm = FakeLLM([Message(content="Hello!", tool_calls=[])])
    client = _client(conn, stt, llm, limiter)
    assert _post(client).json()["reply"] == "Hello!"
    assert _post(client).json()["reply"] == DEMO_LIMIT   # same IP, second turn blocked


def test_x_forwarded_for_identifies_the_visitor(conn):
    # Two different forwarded IPs get independent per-IP budgets.
    limiter = RateLimiter(per_ip_limit=1, global_limit=100)
    stt = FakeSTT(["a", "b"])
    llm = FakeLLM([Message(content="one", tool_calls=[]),
                   Message(content="two", tool_calls=[])])
    client = _client(conn, stt, llm, limiter)
    assert _post(client, xff="9.9.9.9").json()["reply"] == "one"
    assert _post(client, xff="9.9.9.9").json()["reply"] == DEMO_LIMIT   # IP1 spent
    assert _post(client, xff="8.8.8.8").json()["reply"] == "two"        # IP2 fresh
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src ./.venv/Scripts/python.exe -m pytest tests/test_voice_server_limits.py -v`
Expected: FAIL — `ImportError: cannot import name 'DEMO_LIMIT'`

- [ ] **Step 3: Implement** — edit `src/voicedesk/voice/server.py`.

Add `Request` to the FastAPI import (change the existing line):

```python
from fastapi import FastAPI, File, Form, Request, UploadFile
```

Add the demo-limit constants immediately after the existing `_STT_FAILED = {...}` line:

```python
DEMO_LIMIT = (
    "This free demo has reached its limit for today. "
    "Please clone the repository from GitHub to run it yourself."
)
DEMO_LIMIT_ZH = "这个免费体验今天已经达到上限。请从 GitHub 克隆代码库在本地运行。"
_DEMO_LIMIT = {"en": DEMO_LIMIT, "zh": DEMO_LIMIT_ZH}


def _client_ip(request: Request) -> str:
    """The visitor's IP: the first hop of X-Forwarded-For (set by the host's
    proxy), falling back to the socket peer when the header is absent."""
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else "unknown"
```

Change the `create_app` signature to add `limiter=None`:

```python
def create_app(stt, sessions, lock=None, limiter=None) -> FastAPI:
```

Change the `turn` handler to take the request and check the limiter first. Replace the handler's signature and the two lines after `started = ...` so it reads:

```python
    @app.post("/turn")
    async def turn(
        request: Request,
        session_id: str = Form(...),
        audio: UploadFile = File(...),
        lang: str = Form(DEFAULT_LANG),
    ):
        started = time.perf_counter()
        lang = normalize_lang(lang)

        if limiter is not None and not limiter.allow(_client_ip(request)):
            # Over the daily demo cap — don't spend STT or an LLM call.
            return {
                "transcript": "",
                "reply": _DEMO_LIMIT[lang],
                "timings": {"stt_ms": 0, "agent_ms": 0,
                            "total_ms": _ms_since(started)},
                "error": "rate_limited",
                "lang": lang,
            }

        data = await audio.read()
```

Leave the rest of the handler (the size guard, STT, agent, timings) exactly as it is.

- [ ] **Step 4: Run the new tests plus the existing server tests**

Run: `PYTHONPATH=src ./.venv/Scripts/python.exe -m pytest tests/test_voice_server_limits.py tests/test_voice_server.py tests/test_voice_server_chinese.py -v`
Expected: PASS — new limiter tests green and every pre-existing voice-server test unchanged (they pass `limiter=None`, so the new branch never runs).

- [ ] **Step 5: Commit**

```bash
git add src/voicedesk/voice/server.py tests/test_voice_server_limits.py
git commit -m "feat: rate-limit /turn with a friendly bilingual demo-limit reply"
```

---

### Task 3: A fresh calendar per session, and env-driven host/port/limits

**Files:**
- Modify: `src/voicedesk/voice/__main__.py`
- Test: `tests/test_voice_main.py`

**Interfaces:**
- Consumes: `Agent`, `build_system_prompt` from `voicedesk.agent`; `init_db` from `voicedesk.db`; `faq_doc_for` from `voicedesk.lang`; `SessionStore` from `voicedesk.voice.session`.
- Produces (all importable from `voicedesk.voice.__main__`, so they are testable offline without an API key):
  - `fresh_db() -> sqlite3.Connection` — a new in-memory SQLite calendar (`row_factory = sqlite3.Row`, `init_db` applied, `check_same_thread=False`).
  - `build_session_store(llm_factory, today=None) -> SessionStore` — a `SessionStore` whose factory builds, per session, `Agent(fresh_db(), llm_factory(), system_prompt=build_system_prompt(today or date.today(), lang), faq_doc_path=faq_doc_for(lang))`. `llm_factory` is a zero-arg callable returning an LLM, injected so tests use a `FakeLLM` and never touch the network.
  - `main()` now builds the store via `build_session_store`, builds a `RateLimiter` from env, requires `GROQ_API_KEY` at startup (fail-fast), and binds `0.0.0.0:$PORT` (default 7860).

- [ ] **Step 1: Write the failing test** — `tests/test_voice_main.py`

```python
from voicedesk.voice.__main__ import fresh_db, build_session_store
from voicedesk.llm import FakeLLM
from voicedesk.tools import book, lookup_appt


def test_fresh_db_is_an_initialized_isolated_calendar():
    a, b = fresh_db(), fresh_db()
    book(a, "Jane Doe", "5551234", "2026-07-13T09:00", "cleaning")
    assert lookup_appt(a, phone="5551234")          # a has the booking
    assert lookup_appt(b, phone="5551234") == []    # b is a separate calendar


def test_each_session_gets_its_own_calendar():
    store = build_session_store(lambda: FakeLLM([]))
    a = store.get_or_create("visitorA", "en")
    b = store.get_or_create("visitorB", "en")
    assert a.conn is not b.conn                      # isolated connections
    book(a.conn, "Jane Doe", "5551234", "2026-07-13T09:00", "cleaning")
    assert lookup_appt(a.conn, phone="5551234")
    assert lookup_appt(b.conn, phone="5551234") == []   # no cross-visitor leak


def test_same_session_reuses_its_calendar():
    store = build_session_store(lambda: FakeLLM([]))
    first = store.get_or_create("visitorA", "en")
    again = store.get_or_create("visitorA", "en")
    assert first is again                            # history + calendar persist


def test_chinese_session_gets_a_chinese_prompt():
    store = build_session_store(lambda: FakeLLM([]))
    agent = store.get_or_create("v", "zh")
    assert "简体中文" in agent.messages[0]["content"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src ./.venv/Scripts/python.exe -m pytest tests/test_voice_main.py -v`
Expected: FAIL — `ImportError: cannot import name 'fresh_db'`

- [ ] **Step 3: Implement** — replace the whole of `src/voicedesk/voice/__main__.py` with:

```python
import os
import sqlite3
import sys
from datetime import date

import uvicorn
from dotenv import load_dotenv

from voicedesk.agent import Agent, build_system_prompt
from voicedesk.db import init_db
from voicedesk.groq_client import GroqLLM
from voicedesk.lang import faq_doc_for
from voicedesk.voice.limits import RateLimiter
from voicedesk.voice.server import create_app
from voicedesk.voice.session import SessionStore
from voicedesk.voice.stt import GroqWhisper


def _log_retry(reason: str, wait_s: float, attempt: int) -> None:
    if reason == "rate_limited":
        print(f"[voice] rate limited — waiting {wait_s:.1f}s (retry {attempt})",
              file=sys.stderr, flush=True)
    elif reason == "throttle":
        print(f"[voice] approaching token limit — pausing {wait_s:.1f}s",
              file=sys.stderr, flush=True)
    else:
        print(f"[voice] malformed tool call — resampling (retry {attempt})",
              file=sys.stderr, flush=True)


def fresh_db() -> sqlite3.Connection:
    """A new in-memory calendar per visitor session, so visitors never collide
    on slots, no one sees another caller's data, and the app writes nothing to
    disk (which is what makes it deployable to an ephemeral container).
    check_same_thread=False because the blocking work runs in the threadpool."""
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    init_db(conn)
    return conn


def build_session_store(llm_factory, today=None) -> SessionStore:
    """A SessionStore that gives each (session, language) its own fresh calendar
    and a language-appropriate agent. `llm_factory` is injected so tests build a
    FakeLLM with no network."""
    day = today or date.today()
    return SessionStore(lambda lang: Agent(
        fresh_db(),
        llm_factory(),
        system_prompt=build_system_prompt(day, lang),
        faq_doc_path=faq_doc_for(lang),
    ))


def main() -> None:
    load_dotenv()
    if not os.environ.get("GROQ_API_KEY"):
        raise SystemExit(
            "GROQ_API_KEY is not set. Set it in .env locally, or as a Space "
            "Secret when deploying.")

    sessions = build_session_store(lambda: GroqLLM(on_retry=_log_retry))
    limiter = RateLimiter(
        per_ip_limit=int(os.environ.get("PER_IP_DAILY_LIMIT", "8")),
        global_limit=int(os.environ.get("GLOBAL_DAILY_LIMIT", "200")),
    )
    app = create_app(GroqWhisper(), sessions, limiter=limiter)

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "7860"))
    print(f"VoiceDesk is listening on http://{host}:{port}", flush=True)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the new tests plus a confirmation the module imports without a key**

Run: `PYTHONPATH=src ./.venv/Scripts/python.exe -m pytest tests/test_voice_main.py -v`
Expected: PASS (4 passed)

Run: `PYTHONPATH=src ./.venv/Scripts/python.exe -c "import voicedesk.voice.__main__; print('ok')"`
Expected: prints `ok` (importing the module must not require a key — only `main()` does, and it is not called on import).

- [ ] **Step 5: Run the FULL suite**

Run: `PYTHONPATH=src ./.venv/Scripts/python.exe -m pytest -q`
Expected: PASS — everything green, fully offline.

- [ ] **Step 6: Commit**

```bash
git add src/voicedesk/voice/__main__.py tests/test_voice_main.py
git commit -m "feat: fresh in-memory calendar per session; env host/port/limits; require key at startup"
```

---

### Task 4: The container, the Space config, and the docs

Infrastructure and documentation. The Docker build is validated by the human running it once (Step 7) — it is not an automated test.

**Files:**
- Create: `Dockerfile`
- Create: `.dockerignore`
- Create: `huggingface-space-README.md`
- Create: `docs/DEPLOY.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: `python -m voicedesk.voice` (Task 3's `main()`); `requirements.txt`; `clinic_info.md`, `clinic_info.zh.md`.
- Produces: a container that serves the app on `0.0.0.0:7860`, and the docs to deploy it.

- [ ] **Step 1: Write `Dockerfile`**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
# answer_faq opens these by a path relative to the working directory, so they
# must sit at /app next to where the app runs.
COPY clinic_info.md clinic_info.zh.md ./

ENV PYTHONPATH=/app/src
ENV PORT=7860
EXPOSE 7860

CMD ["python", "-m", "voicedesk.voice"]
```

- [ ] **Step 2: Write `.dockerignore`** — keep secrets, local state, and everything the runtime does not need out of the image:

```
.git/
.gitignore
.venv/
.env
.env.example
tests/
docs/
reports/
.superpowers/
*.db
__pycache__/
**/__pycache__/
.pytest_cache/
huggingface-space-README.md
README.md
```

- [ ] **Step 3: Write `huggingface-space-README.md`** — this becomes the Space's own README; the YAML header is how HF configures a Docker Space and its port:

```markdown
---
title: VoiceDesk
emoji: 🦷
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# VoiceDesk — AI Voice Receptionist

Hold the button and speak — in English or 中文 — to book a dental appointment by voice.
Use Chrome or Edge (needs the microphone and the Web Speech API).

This is a free-tier demo: usage is rate-limited, and each visitor gets a private,
self-resetting calendar. Source and full write-up:
https://github.com/hejun789/voicedesk
```

- [ ] **Step 4: Write `docs/DEPLOY.md`** — the manual steps (the parts a human must do):

````markdown
# Deploying the public demo (Hugging Face Spaces)

The app runs as a Docker Space. HF provides HTTPS automatically (required for the
microphone), and the Groq key is stored as a Space Secret — never in the image.

## One-time setup

1. Create a Space at https://huggingface.co/new-space
   - **SDK:** Docker
   - **Hardware:** CPU basic (free)
2. In the Space's **Settings → Variables and secrets**, add:
   - Secret `GROQ_API_KEY` = your Groq key
   - (optional) Variable `GROQ_MODEL` = `llama-3.1-8b-instant` (default; set
     `openai/gpt-oss-120b` for higher quality at the cost of a tighter daily cap)
   - (optional) Variables `PER_IP_DAILY_LIMIT`, `GLOBAL_DAILY_LIMIT` to tune the caps
3. Push this repo's contents to the Space, with `huggingface-space-README.md`
   renamed to `README.md` **in the Space** (its YAML header configures the Space).
   The Dockerfile at the repo root is what the Space builds.

## Verify the container locally before pushing

```bash
docker build -t voicedesk .
docker run --rm -p 7860:7860 -e GROQ_API_KEY=your_key voicedesk
# open http://localhost:7860, switch to English or 中文, hold the button and speak
```

## Record the demo video

Free-tier quota means the live link can hit its daily cap. Record a ~30s screen
capture of a successful booking (one English, one Mandarin turn) on
`openai/gpt-oss-120b`, and link it at the top of the README. The live link is the
bonus; the video guarantees an interviewer always sees it work.
````

- [ ] **Step 5: Update `README.md`** — add a live-demo block. Insert it immediately after the headline paragraph block (after the line that begins `**But the part worth reading is`):

```markdown

---

## Live demo

**🎙️ Try it: <https://hejun789-voicedesk.hf.space>** — hold the button and speak, in
English or 中文, to book an appointment by voice. Chrome or Edge (needs a microphone).

> It runs on a free tier, so it is rate-limited and may reach its daily cap — in which
> case it says so politely. **Demo video:** _(add your 30-second recording here)_

```

(Replace the URL with your actual Space URL once created — the pattern is
`https://<username>-<space-name>.hf.space`.)

- [ ] **Step 6: Run the full suite (nothing above changed code, but confirm green)**

Run: `PYTHONPATH=src ./.venv/Scripts/python.exe -m pytest -q`
Expected: PASS — all green.

- [ ] **Step 7: Build and run the container locally (human, needs Docker + a key)**

Run:
```bash
docker build -t voicedesk .
docker run --rm -p 7860:7860 -e GROQ_API_KEY=$GROQ_API_KEY voicedesk
```
Expected: the build succeeds; the container prints `VoiceDesk is listening on http://0.0.0.0:7860`; opening <http://localhost:7860> serves the page, and a spoken turn books an appointment. (If Docker is not installed, skip this step and validate on the Space itself after pushing.)

- [ ] **Step 8: Commit**

```bash
git add Dockerfile .dockerignore huggingface-space-README.md docs/DEPLOY.md README.md
git commit -m "feat: Dockerfile, HF Space config, and deployment docs for the public demo"
```

---

## Phase 4 Definition of Done

- `docker build` produces an image that serves the app on `0.0.0.0:7860` and books an appointment by voice.
- The Groq key is read only from the environment (Space Secret) — never in the image or the browser.
- Each visitor session has an isolated in-memory calendar; the app writes nothing to disk.
- Exceeding `PER_IP_DAILY_LIMIT` or `GLOBAL_DAILY_LIMIT` returns a friendly bilingual message, not an error, and does not call STT or the agent.
- The full test suite still runs offline with no API key.
- No new Python dependencies.

## Manual steps for the human (cannot be automated here)

1. **Install Docker** (if verifying locally) and run Task 4 Step 7.
2. **Create the HF Space**, set the `GROQ_API_KEY` secret, and push the repo (see `docs/DEPLOY.md`).
3. **Record the ~30s demo video** and link it in the README.
4. **Put the real Space URL** into the README's Live demo block.
