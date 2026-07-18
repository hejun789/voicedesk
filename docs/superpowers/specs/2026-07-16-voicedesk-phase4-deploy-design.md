# VoiceDesk Phase 4 — Public Demo Deployment

**Design spec** · 2026-07-16 · Status: approved, ready for planning

## Purpose

Give the project a **live link** an interviewer can click and talk to, instead of a
clone-and-run. The deployment must survive three realities a local demo hides: browsers
require **HTTPS** for microphone access, the Groq API key becomes a **server-side secret**
that strangers could drain, and a free host is **ephemeral** (it sleeps, restarts, and
writes nothing durable).

## Platform: Hugging Face Spaces (Docker SDK)

Chosen because the author already runs a Space (DocIntel), so the workflow is known, and it
solves all three constraints at once:

- **HTTPS is automatic** — the Space is served at `https://<user>-voicedesk.hf.space`, so
  `getUserMedia` (the microphone) works.
- **The key is a Space Secret** — `GROQ_API_KEY` is injected as an environment variable and
  never reaches the browser. `.env` stays gitignored and is never in the image.
- **The FastAPI app runs essentially unchanged** in a Dockerfile. HF Docker Spaces serve on
  **port 7860**, so the app binds a host/port read from the environment.

## Deployment model: the app already had no need for a database server

The single change that makes the app deployable to an ephemeral container is giving **each
visitor session its own in-memory SQLite calendar** — the same "fresh in-memory DB" pattern
the eval harness already uses per run.

Today `__main__.py` opens one shared `voicedesk.db` file. On a public link that is wrong:
the first visitor to book 09:00 blocks it for everyone, strangers see each other's bookings,
and the file fills with junk until the Space restarts. With a per-session in-memory calendar:

- visitors never collide on slots,
- nobody sees another caller's data,
- the calendar resets when the session expires,
- and **the app writes nothing to disk**, so no persistent volume is required.

The `Agent` already owns its `conn`, so the connection lives exactly as long as the session
holds the agent and is garbage-collected afterward. The session key is already
`(session_id, lang)`; the database simply moves into the session factory.

## Quota guard: `src/voicedesk/voice/limits.py`

A public link means anyone — including bots crawling Hugging Face — can spend the daily
token budget. An in-memory rate limiter, checked at the top of `POST /turn`, prevents one
visitor (or a burst) from draining it:

- **Per-IP:** at most `PER_IP_DAILY_LIMIT` turns per UTC day (default 8).
- **Global:** at most `GLOBAL_DAILY_LIMIT` turns per UTC day (default 200), a ceiling that
  keeps total usage within the model's budget with margin.
- Over either limit → **HTTP 200** with a friendly spoken reply, never a raw error:
  *"This free demo has reached its limit for today. Please clone the repository to run it
  yourself."* (with a Chinese variant, since the app is bilingual). The agent and STT are
  **not** called on a rejected turn — the rejection is cheap.

The limiter is pure stdlib: a dict keyed by IP plus a global counter, both reset when the
UTC date rolls over. It takes an injectable clock so expiry is testable without waiting —
the same design as `SessionStore`. Limits are environment variables so they can be tuned
after observing real traffic without a redeploy of code.

The caller's IP comes from the `X-Forwarded-For` header HF sets (the first hop), falling
back to the socket peer when absent.

## Model choice for the deployment

The Space defaults to **`llama-3.1-8b-instant`** (env `GROQ_MODEL`), not `gpt-oss-120b`.
Rationale: a public link's first duty is to *work when clicked*, and 8b's far higher
free-tier limits (14,400 requests/day) mean it almost always will, whereas gpt-oss-120b's
200k-tokens/day cap (~25–40 conversation turns total) would leave the demo showing "limit
reached" most of the day. Booking — the headline behavior — works well on 8b; its weakness
is multi-step tool use (cancel/reschedule), which the README video demonstrates on the
stronger model. Because `GROQ_MODEL` is an environment variable, switching the Space to
`gpt-oss-120b` is a one-setting change, with no code redeploy.

## The honest limitation, and the mitigation

Any free-tier public demo can hit its cap. Two things keep that from ever looking broken:

1. The graceful "demo limit reached" reply above.
2. A **short screen recording** (≈30s) of the agent booking an appointment by voice —
   ideally one English and one Mandarin turn on `gpt-oss-120b` — linked at the top of the
   README. The live link is the bonus; the video is the guarantee that an interviewer always
   sees it work. (Recording the video is a manual step for the author, noted in the plan.)

## Files

```
Dockerfile                          # new — python:3.11-slim, pip install -r requirements.txt,
                                    #        run uvicorn on 0.0.0.0:$PORT (default 7860)
.dockerignore                       # new — exclude .env, .venv, tests, docs, reports, .git
huggingface-space-README.md         # new — the Space's own README with the HF metadata
                                    #        header (title, sdk: docker, app_port: 7860)
src/voicedesk/voice/limits.py       # new — RateLimiter (per-IP + global, injectable clock)
src/voicedesk/voice/server.py       # + rate-limit check in /turn (before STT/agent);
                                    #   read client IP from X-Forwarded-For
src/voicedesk/voice/__main__.py     # fresh in-memory DB per session; bind host/port from env
README.md                           # + "Live demo" link and the video at the top
tests/test_voice_limits.py          # new — offline unit tests for the limiter
tests/test_voice_server_limits.py   # new — over-limit returns the friendly message offline
docs/DEPLOY.md                      # new — the manual HF Space steps (create Space, set the
                                    #        GROQ_API_KEY secret, connect the repo, record video)
```

## Error handling

- Over rate limit → HTTP 200, friendly bilingual message, no STT/agent call.
- Missing `GROQ_API_KEY` in the Space → the app still boots and serves the page; the failure
  surfaces only on the first `/turn`, where `GroqLLM` raises and the existing `LLMError`
  path degrades to the escalation reply rather than crashing. (The `require_api_key` check
  already used by the eval CLI is not applied to the server, so the page always loads.)
- A cold-started Space (first hit after sleep) simply takes longer on the first request; no
  code handling needed.

## Testing strategy

Everything stays **offline** — no network, no API key, no microphone, no container:

- `limits.py` — unit-tested with an injectable clock: per-IP cap, global cap, UTC-day reset,
  and that a normal visitor under both caps is allowed.
- `POST /turn` over-limit — tested with `FakeSTT`/`FakeLLM`, asserting the friendly message
  is returned and that neither STT nor the agent was called (a `FakeLLM([])` tripwire).
- The per-session in-memory DB is exercised by the existing voice-server tests (they already
  inject their own connection); a new test asserts two sessions get isolated calendars.
- The Docker image is validated by the author building and running it locally once before
  pushing to the Space — a manual step in `docs/DEPLOY.md`, not an automated test.

## Success criteria

- A public HTTPS URL where anyone can hold a button, speak, and book an appointment by
  voice, in English or Chinese.
- The Groq key is never exposed to the browser and is stored only as a Space Secret.
- One visitor cannot drain the daily quota; exceeding the cap shows a friendly message, not
  an error.
- Each visitor gets an isolated, self-resetting calendar; the app persists nothing to disk.
- The full test suite still runs offline with no API key.
- The README leads with a live link and a short demo video.
