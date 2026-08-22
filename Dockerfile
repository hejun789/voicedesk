FROM python:3.11-slim

WORKDIR /app

# espeak-ng backs Piper's phonemizer. Recent piper-tts wheels bundle their
# own copy, but installing the system package too is cheap insurance against
# a platform where the bundled one doesn't load.
RUN apt-get update && apt-get install -y --no-install-recommends espeak-ng \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
# answer_faq opens these by a path relative to the working directory, so they
# must sit at /app next to where the app runs.
COPY clinic_info.md clinic_info.zh.md ./

ENV PYTHONPATH=/app/src
ENV PORT=7860
# The public demo runs on 8b for its higher free-tier limits; override in the Space to raise quality.
ENV GROQ_MODEL=llama-3.1-8b-instant
ENV PIPER_VOICES_DIR=/app/voices

# Fetched at build time, not request time: these are ~150MB combined and
# would make the container's first reply take minutes instead of seconds.
#
# This RUN sits after COPY src/ on purpose: download_voices.py imports
# _VOICE_IDS/_voice_repo_path from voicedesk.voice.tts rather than
# hardcoding the voice list a second time, so the download path can never
# drift from the path PiperTTS actually reads. The cost is that ANY edit
# under src/ — including the static frontend files Tasks 4-7 touch
# repeatedly — busts this layer and every layer after it, forcing a fresh
# ~126MB Hugging Face re-download on every rebuild, not just changes that
# actually affect the voice models.
#
# Remedy, if build times become painful: split a narrow layer ahead of
# COPY src/ that copies only voicedesk/lang.py, voicedesk/voice/tts.py,
# voicedesk/voice/download_voices.py and their two __init__.py files, run
# this RUN against that layer, then COPY src/ (which will re-copy the same
# files harmlessly) afterwards. Unverified here — Docker isn't installed on
# this machine, so whoever applies it must actually build the image to
# confirm the app still starts before trusting it.
RUN python -m voicedesk.voice.download_voices

EXPOSE 7860

CMD ["python", "-m", "voicedesk.voice"]
