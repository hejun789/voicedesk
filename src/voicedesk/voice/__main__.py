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
