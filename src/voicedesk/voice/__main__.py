import os
import sqlite3
import sys
from datetime import date

import uvicorn
from dotenv import load_dotenv

from voicedesk.agent import Agent, build_system_prompt
from voicedesk.db import init_db
from voicedesk.groq_client import GroqLLM
from voicedesk.lang import LANGUAGES, faq_doc_for
from voicedesk.voice.limits import RateLimiter
from voicedesk.voice.server import create_app
from voicedesk.voice.session import SessionStore
from voicedesk.voice.stt import GroqWhisper
from voicedesk.voice.tts import PiperTTS


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


_WILDCARD_HOSTS = {"0.0.0.0", "::"}


def browsable_url(host: str, port: int) -> str:
    """A URL the reader can actually click.

    The server binds 0.0.0.0 so containers and hosted platforms can reach it,
    but that address means "every interface", not a destination — a browser
    handed it answers ERR_ADDRESS_INVALID. Print the loopback address instead
    so the startup line is a working link.
    """
    if host in _WILDCARD_HOSTS:
        host = "127.0.0.1"
    return f"http://{host}:{port}"


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
    # Deliberately generous relative to `limiter`: one turn makes exactly one
    # /tts call, so this only needs to guard against pathological reuse, not
    # normal traffic.
    tts_limiter = RateLimiter(
        per_ip_limit=int(os.environ.get("PER_IP_TTS_DAILY_LIMIT", "40")),
        global_limit=int(os.environ.get("GLOBAL_TTS_DAILY_LIMIT", "1000")),
    )
    tts = PiperTTS()
    app = create_app(GroqWhisper(), sessions, limiter=limiter, tts=tts,
                     tts_limiter=tts_limiter)

    # Piper's first call per language pays a 1.8-2.4s ONNX model load, which
    # would otherwise land inside the first reply's /tts call of every
    # container lifetime. Paying that cost here instead — a slower boot —
    # keeps it out of a caller's first turn.
    for warmup_lang in LANGUAGES:
        try:
            tts.synthesize("ok", warmup_lang)
        except Exception as e:  # noqa: BLE001 - warm-up must never block startup
            print(f"[voice] TTS warm-up failed for {warmup_lang!r}: {e}",
                  file=sys.stderr, flush=True)

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "7860"))
    print(f"VoiceDesk is listening on {browsable_url(host, port)}", flush=True)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
