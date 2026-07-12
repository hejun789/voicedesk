import sqlite3

import uvicorn
from dotenv import load_dotenv

from voicedesk.agent import Agent
from voicedesk.db import init_db
from voicedesk.groq_client import GroqLLM
from voicedesk.voice.server import create_app
from voicedesk.voice.session import SessionStore
from voicedesk.voice.stt import GroqWhisper


def main() -> None:
    load_dotenv()
    # check_same_thread=False: FastAPI serves requests on a worker thread.
    conn = sqlite3.connect("voicedesk.db", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    init_db(conn)

    sessions = SessionStore(lambda: Agent(conn, GroqLLM()))
    app = create_app(GroqWhisper(), sessions)

    print("VoiceDesk is listening on http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
