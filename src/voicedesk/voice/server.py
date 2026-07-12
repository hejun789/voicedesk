import time
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from voicedesk.voice.stt import STTError

STATIC_DIR = Path(__file__).parent / "static"

DIDNT_CATCH = "Sorry, I didn't catch that. Could you say that again?"
STT_FAILED = (
    "Sorry, I'm having trouble hearing you. "
    "Let me have a team member call you back."
)


def _ms_since(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)


def create_app(stt, sessions) -> FastAPI:
    """`stt` implements STTClient; `sessions` is a SessionStore. Both are
    injected so the whole app can be tested with no network and no microphone."""
    app = FastAPI(title="VoiceDesk")

    @app.get("/")
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    @app.post("/turn")
    async def turn(
        session_id: str = Form(...),
        audio: UploadFile = File(...),
    ):
        started = time.perf_counter()
        data = await audio.read()

        stt_started = time.perf_counter()
        try:
            transcript = stt.transcribe(data, audio.filename or "audio.webm")
        except STTError as e:
            # Never crash the call — speak an apology and report the error.
            return {
                "transcript": "",
                "reply": STT_FAILED,
                "timings": {"stt_ms": _ms_since(stt_started), "agent_ms": 0,
                            "total_ms": _ms_since(started)},
                "error": str(e),
            }
        stt_ms = _ms_since(stt_started)

        if not transcript.strip():
            # Don't spend an LLM call, and don't pollute the history with noise.
            return {
                "transcript": "",
                "reply": DIDNT_CATCH,
                "timings": {"stt_ms": stt_ms, "agent_ms": 0,
                            "total_ms": _ms_since(started)},
            }

        agent_started = time.perf_counter()
        agent = sessions.get_or_create(session_id)
        reply = agent.respond(transcript)
        agent_ms = _ms_since(agent_started)

        return {
            "transcript": transcript,
            "reply": reply,
            "timings": {"stt_ms": stt_ms, "agent_ms": agent_ms,
                        "total_ms": _ms_since(started)},
        }

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app
