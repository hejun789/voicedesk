import sys
import threading
import time
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from voicedesk.agent import _FALLBACK as AGENT_FALLBACK
from voicedesk.lang import DEFAULT_LANG, normalize_lang
from voicedesk.voice.stt import STTError, is_silence_hallucination
from voicedesk.voice.tts import TTSError

STATIC_DIR = Path(__file__).parent / "static"

# The front end is a handful of KB of hand-written JS with no build step and no
# content hash in its filename, so a cached copy is indistinguishable from a
# current one. Without this, a returning visitor keeps running the JavaScript
# from before the last deploy — and a developer testing a fix locally sees the
# OLD code while believing the fix failed. That cost a full round of
# false-negative manual testing on this project. Revalidating a few KB per load
# is the right trade.
_NO_STORE = "no-store, no-cache, must-revalidate"


class NoCacheStaticFiles(StaticFiles):
    """StaticFiles that forbids browser caching. See _NO_STORE above."""

    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = _NO_STORE
        return response

DIDNT_CATCH = "Sorry, I didn't catch that. Could you say that again?"
STT_FAILED = (
    "Sorry, I'm having trouble hearing you. "
    "Let me have a team member call you back."
)

DIDNT_CATCH_ZH = "抱歉，我没有听清，可以再说一遍吗？"
STT_FAILED_ZH = "抱歉，我听不清楚。我让同事回电给您。"

_DIDNT_CATCH = {"en": DIDNT_CATCH, "zh": DIDNT_CATCH_ZH}
_STT_FAILED = {"en": STT_FAILED, "zh": STT_FAILED_ZH}

DEMO_LIMIT = (
    "This free demo has reached its limit for today. "
    "Please clone the repository from GitHub to run it yourself."
)
DEMO_LIMIT_ZH = "这个免费体验今天已经达到上限。请从 GitHub 克隆代码库在本地运行。"
_DEMO_LIMIT = {"en": DEMO_LIMIT, "zh": DEMO_LIMIT_ZH}


def _client_ip(request: Request) -> str:
    """The visitor's IP: the first hop of X-Forwarded-For (set by the host's
    proxy), falling back to the socket peer when the header is absent.

    Note: X-Forwarded-For is client-controlled, so the per-IP cap is
    best-effort; the global daily cap is the real protection against quota
    drain."""
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


# A tap (rather than a hold) makes MediaRecorder emit an empty/near-empty
# blob. Treat anything under this as "didn't catch that" rather than
# spending an STT call on it.
MIN_AUDIO_BYTES = 1000

# Guard against absurdly large uploads.
MAX_AUDIO_BYTES = 10 * 1024 * 1024

# Piper allocates roughly 2.7KB of RAM per input character, fully buffered
# before the response is sent, and run_in_threadpool defaults to 40 workers —
# a handful of large requests can OOM a 2-vCPU Space and starve /turn behind
# them. 800 is far above any real agent reply, so this never clips a
# legitimate turn.
MAX_TTS_CHARS = 800


def _ms_since(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)


def create_app(stt, sessions, lock=None, limiter=None, tts=None,
               tts_limiter=None) -> FastAPI:
    """`stt` implements STTClient; `sessions` is a SessionStore. Both are
    injected so the whole app can be tested with no network and no microphone.

    `lock` serialises agent turns. Each session now has its own in-memory
    calendar, but a single global lock keeps concurrent visitors' agent calls
    from interleaving on shared process state and naturally throttles quota
    burn on the free tier. Defaults to a fresh threading.Lock().

    `tts` implements TTSClient and backs the /tts route; `tts_limiter` is a
    separate RateLimiter for that route (see MAX_TTS_CHARS above) — kept
    independent of `limiter` because every turn already makes exactly one
    /tts call, so sharing a counter would silently halve the caller's turn
    budget."""
    if lock is None:
        lock = threading.Lock()
    app = FastAPI(title="VoiceDesk")

    @app.get("/")
    def index():
        return FileResponse(STATIC_DIR / "index.html",
                            headers={"Cache-Control": _NO_STORE})

    @app.post("/turn")
    async def turn(
        request: Request,
        session_id: str = Form(...),
        audio: UploadFile = File(...),
        lang: str = Form(DEFAULT_LANG),
        heard_chars: int | None = Form(None),
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

        if len(data) < MIN_AUDIO_BYTES or len(data) > MAX_AUDIO_BYTES:
            # A stray tap or an oversized upload — don't spend an STT call,
            # don't touch the agent, just apologise.
            return {
                "transcript": "",
                "reply": _DIDNT_CATCH[lang],
                "timings": {"stt_ms": 0, "agent_ms": 0,
                            "total_ms": _ms_since(started)},
                "lang": lang,
            }

        stt_started = time.perf_counter()
        try:
            transcript = await run_in_threadpool(
                stt.transcribe, data, "turn.webm", lang)
        except STTError as e:
            # Never crash the call — speak an apology and report the error.
            print(f"[voice] STT error: {e}", file=sys.stderr, flush=True)
            return {
                "transcript": "",
                "reply": _STT_FAILED[lang],
                "timings": {"stt_ms": _ms_since(stt_started), "agent_ms": 0,
                            "total_ms": _ms_since(started)},
                "error": "stt_failed",
                "lang": lang,
            }
        stt_ms = _ms_since(stt_started)

        if not transcript.strip() or is_silence_hallucination(transcript):
            # Don't spend an LLM call, and don't pollute the history with noise.
            return {
                "transcript": "",
                "reply": _DIDNT_CATCH[lang],
                "timings": {"stt_ms": stt_ms, "agent_ms": 0,
                            "total_ms": _ms_since(started)},
                "lang": lang,
            }

        agent_started = time.perf_counter()

        def _run_agent() -> str:
            with lock:
                agent = sessions.get_or_create(session_id, lang)
                if heard_chars is not None:
                    # The caller cut the previous reply short; trim history to
                    # what they actually heard before adding this turn.
                    #
                    # Ordering note: a THINKING-interrupt's heard_chars=0 and
                    # the turn it interrupted are racing for this same lock --
                    # both are /turn requests, and only one truncates history
                    # that the other is still about to append to. Correctness
                    # relies on the interrupted turn reaching the lock first,
                    # which it reliably does: the interrupting turn's own
                    # /turn call cannot start until the caller finishes an
                    # utterance (min ~200ms), the VAD's hangover elapses
                    # (800ms), the recording uploads, and its own STT
                    # completes -- all strictly after the interrupted turn's
                    # /turn request was already sent.
                    agent.truncate_last_reply(heard_chars)
                return agent.respond(transcript)

        reply = await run_in_threadpool(_run_agent)
        agent_ms = _ms_since(agent_started)

        if reply == AGENT_FALLBACK:
            print(f"[voice] agent fell back (LLM error or iteration cap) on: "
                  f"{transcript!r}", file=sys.stderr, flush=True)

        return {
            "transcript": transcript,
            "reply": reply,
            "timings": {"stt_ms": stt_ms, "agent_ms": agent_ms,
                        "total_ms": _ms_since(started)},
            "lang": lang,
        }

    @app.post("/tts")
    async def synthesize(
        request: Request,
        text: str = Form(...),
        lang: str = Form(DEFAULT_LANG),
    ):
        if tts is None:
            raise HTTPException(status_code=503, detail="tts_unavailable")
        if len(text) > MAX_TTS_CHARS:
            raise HTTPException(status_code=413, detail="text_too_long")
        if tts_limiter is not None and not tts_limiter.allow(_client_ip(request)):
            raise HTTPException(status_code=429, detail="rate_limited")
        lang_norm = normalize_lang(lang)
        try:
            wav = await run_in_threadpool(tts.synthesize, text, lang_norm)
        except TTSError as e:
            print(f"[voice] TTS error: {e}", file=sys.stderr, flush=True)
            raise HTTPException(status_code=503, detail="tts_failed")
        return Response(content=wav, media_type="audio/wav")

    app.mount("/static", NoCacheStaticFiles(directory=STATIC_DIR), name="static")
    return app
