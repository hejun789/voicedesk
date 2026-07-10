# VoiceDesk — AI Voice Receptionist for Clinics

An AI agent that books, reschedules, and cancels clinic appointments and answers
FAQs. Phase 1 is text-only; voice (STT/TTS) and deployment come in later phases.

## Why
Clinics miss 30%+ of inbound calls — each is a potential lost patient. VoiceDesk is
a 24/7 receptionist that takes real booking actions, not just chat.

## Run (Phase 1, text mode)
1. `python -m venv .venv && source .venv/Scripts/activate` (Windows Git Bash)
2. `pip install -r requirements.txt`
3. Get a free API key at https://console.groq.com and copy `.env.example` to `.env`.
4. `PYTHONPATH=src python -m voicedesk.cli`

## Test
`PYTHONPATH=src python -m pytest -v`

## Architecture
Browser/CLI → agent core (LLM + tool calling) → tools over SQLite calendar.
Tools, agent, and LLM provider are cleanly separated so STT/TTS and Twilio can be
added without touching the booking logic.
