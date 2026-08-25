"""Fetches the Piper voice models PiperTTS needs.

Run once at container build time (see Dockerfile) — these are 60-115MB
binaries and do not belong in git.

    python -m voicedesk.voice.download_voices
"""
import os
from pathlib import Path

from voicedesk.voice.tts import _VOICE_IDS, _voice_repo_path

REPO_ID = "rhasspy/piper-voices"


def download_all(voices_dir: str | Path | None = None) -> None:
    from huggingface_hub import hf_hub_download  # build-time only dependency

    voices_dir = Path(voices_dir or os.environ.get("PIPER_VOICES_DIR", "voices"))
    voices_dir.mkdir(parents=True, exist_ok=True)
    for voice_id in _VOICE_IDS.values():
        repo_path = _voice_repo_path(voice_id)
        for suffix in (".onnx", ".onnx.json"):
            hf_hub_download(
                repo_id=REPO_ID,
                filename=f"{repo_path}{suffix}",
                local_dir=voices_dir,
            )


if __name__ == "__main__":
    download_all()
