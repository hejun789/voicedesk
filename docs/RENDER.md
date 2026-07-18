# Deploying the public demo (Render)

> Hugging Face now puts the Docker SDK behind a paid plan, so the public demo is hosted on
> **Render** instead. No application code changes are needed — the app already binds
> `0.0.0.0:$PORT` (which Render provides), persists nothing to disk (a fresh in-memory
> calendar per visitor), and keeps the Groq key server-side. Render gives free HTTPS
> (required for the microphone) and env-var secrets.

Render's free web services **sleep after ~15 minutes of inactivity**, so the first hit after
idle takes ~30s to wake. That is fine for a demo — and the same trade-off HF free tiers have.

## Steps

1. Sign in at <https://render.com> with your GitHub account.
2. **New + → Web Service**, and connect the `hejun789/voicedesk` repository.
3. Render detects the repo's `Dockerfile` and offers **Runtime: Docker** — choose it. The
   Dockerfile already pins `GROQ_MODEL=llama-3.1-8b-instant` and runs the app, so there is
   nothing to configure there.
   - **Instance type: Free.**
   - Leave the start command blank (the Dockerfile's `CMD` handles it).
4. Under **Environment**, add:
   - `GROQ_API_KEY` = your Groq key **(mark it secret)**. This is the whole security model —
     the key lives only here, never in the repo or the browser.
   - (optional) `PER_IP_DAILY_LIMIT`, `GLOBAL_DAILY_LIMIT` to tune the demo caps.
   - You do **not** need to set `PORT` — Render injects it and the app reads it.
5. **Create Web Service.** The first build takes a few minutes. When it's live you get a URL
   like `https://voicedesk.onrender.com`.
6. Open the URL in Chrome or Edge, switch to English or 中文, hold the button and speak.

## Alternative: native Python runtime (no Docker)

If you prefer not to use the Dockerfile, choose **Runtime: Python 3** and set:

- **Build command:** `pip install -r requirements.txt`
- **Start command:** `PYTHONPATH=src python -m voicedesk.voice`
- Env var `PYTHON_VERSION` = `3.11.9` (the code uses 3.11 syntax).
- Env var `GROQ_MODEL` = `llama-3.1-8b-instant` (the Docker runtime pins this automatically;
  the native runtime does not, and the code default is the larger `llama-3.3-70b-versatile`
  whose daily token budget the demo caps are **not** sized for — so set it explicitly).
- Plus `GROQ_API_KEY` as above.

`clinic_info.md` / `clinic_info.zh.md` sit at the repo root, which is the working directory
Render runs from, so FAQ retrieval finds them either way.

## Record the demo video

Render free instances sleep and the free-tier Groq quota can hit its daily cap, so the live
link is not guaranteed to be awake-and-available the moment an interviewer clicks it. Record
a ~30s screen capture of a successful booking (one English, one Mandarin turn) — ideally on
`openai/gpt-oss-120b` for quality — and link it at the top of the README. The live link is
the bonus; the video is the guarantee.
