# VoiceDesk — AI Voice Receptionist for Clinics

An AI agent that books, reschedules, and cancels clinic appointments and answers
FAQs. Phase 1 is text-only; voice (STT/TTS) and deployment come in later phases.

## Why
Clinics miss 30%+ of inbound calls — each is a potential lost patient. VoiceDesk is
a 24/7 receptionist that takes real booking actions, not just chat.

## Run (Phase 1, text mode)

### Windows PowerShell
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # prompt should now show (.venv)
pip install -r requirements.txt       # in China: add  -i https://pypi.tuna.tsinghua.edu.cn/simple
# Get a free API key at https://console.groq.com,
# copy .env.example to .env, and paste your key into .env
$env:PYTHONPATH = "src"
python -m voicedesk.cli
```

### macOS / Linux / Git Bash
```bash
python -m venv .venv
source .venv/Scripts/activate          # or .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
cp .env.example .env                    # then paste your Groq key into .env
PYTHONPATH=src python -m voicedesk.cli
```

Note: put your real key in `.env` (gitignored), NOT in `.env.example` (the committed template).

## Test
```
# PowerShell:  $env:PYTHONPATH = "src"; python -m pytest -v
# Bash:        PYTHONPATH=src python -m pytest -v
```

## Architecture
Browser/CLI → agent core (LLM + tool calling) → tools over SQLite calendar.
Tools, agent, and LLM provider are cleanly separated so STT/TTS and Twilio can be
added without touching the booking logic.
