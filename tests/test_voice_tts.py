from pathlib import Path

import pytest

from voicedesk.voice.tts import FakeTTS, PiperTTS, TTSError, _voice_repo_path


def test_fake_tts_returns_configured_wav_and_records_calls():
    tts = FakeTTS(wav=b"RIFF-fake-wav-bytes")
    assert tts.synthesize("hello", "en") == b"RIFF-fake-wav-bytes"
    assert tts.calls == [("hello", "en")]


def test_fake_tts_normalizes_language():
    tts = FakeTTS()
    tts.synthesize("你好", "zh-CN")
    assert tts.calls == [("你好", "zh")]


def test_voice_repo_path_matches_huggingface_layout():
    # rhasspy/piper-voices lays voices out as
    # {lang}/{lang_region}/{name}/{quality}/{lang_region}-{name}-{quality}
    assert _voice_repo_path("en_US-lessac-medium") == \
        "en/en_US/lessac/medium/en_US-lessac-medium"
    assert _voice_repo_path("zh_CN-huayan-medium") == \
        "zh/zh_CN/huayan/medium/zh_CN-huayan-medium"


def test_piper_tts_raises_clear_error_when_model_missing(tmp_path):
    # No network, no `piper` package needed for this path: the missing-file
    # check happens before piper is ever imported.
    tts = PiperTTS(voices_dir=tmp_path)
    with pytest.raises(TTSError, match="Piper voice model missing"):
        tts.synthesize("hello", "en")
