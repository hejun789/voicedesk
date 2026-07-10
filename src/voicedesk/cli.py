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
