# Deploying the public demo (Hugging Face Spaces)

> **Note:** Hugging Face now puts the Docker SDK behind a paid plan, so the recommended host
> is **Render** — see [RENDER.md](RENDER.md). This document is kept for reference in case HF
> restores a free Docker tier; the same repo `Dockerfile` works on both.

The app runs as a Docker Space. HF provides HTTPS automatically (required for the
microphone), and the Groq key is stored as a Space Secret — never in the image.

## One-time setup

1. Create a Space at https://huggingface.co/new-space
   - **SDK:** Docker
   - **Hardware:** CPU basic (free)
2. In the Space's **Settings → Variables and secrets**, add:
   - Secret `GROQ_API_KEY` = your Groq key
   - (optional) Variable `GROQ_MODEL` — the image already defaults this to
     `llama-3.1-8b-instant` (chosen for higher free-tier limits). Override it to
     `openai/gpt-oss-120b` for higher quality at the cost of a much tighter daily cap.
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
