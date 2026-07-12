import sqlite3
import sys

import uvicorn
from dotenv import load_dotenv

from voicedesk.agent import Agent
from voicedesk.db import init_db
from voicedesk.groq_client import GroqLLM
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


def main() -> None:
    load_dotenv()
    # check_same_thread=False: the blocking STT/agent work in /turn is
    # offloaded to the threadpool, so this connection is used from worker
    # threads, not the event-loop thread.
    conn = sqlite3.connect("voicedesk.db", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    init_db(conn)

    sessions = SessionStore(lambda: Agent(conn, GroqLLM(on_retry=_log_retry)))
    app = create_app(GroqWhisper(), sessions)

    print("VoiceDesk is listening on http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
