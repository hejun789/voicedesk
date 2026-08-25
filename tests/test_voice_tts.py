import io
import os
from pathlib import Path

import pytest

from voicedesk.voice.tts import (
    FakeTTS, PiperTTS, TTSError, _segment_by_script, _voice_repo_path,
)

_VOICES = Path(os.environ.get("PIPER_VOICES_DIR", "voices"))
needs_models = pytest.mark.skipif(
    not (_VOICES / "zh/zh_CN/huayan/medium/zh_CN-huayan-medium.onnx").exists(),
    reason="Piper voice models not downloaded (run download_voices.py)")


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


# --- mixed-script segmentation --------------------------------------------
#
# A Piper voice is single-language: zh_CN-huayan phonemizes through espeak's
# `cmn` and was trained only on Mandarin phonetics, so Latin text inside a
# Chinese reply comes out mangled. Measured: it renders "Springfield" in
# 0.45s against the English voice's 0.74s -- it is compressing the syllables,
# not saying them. Splitting by script and letting each voice speak its own
# is the fix.

def test_latin_runs_inside_chinese_are_tagged_english():
    assert _segment_by_script("我们接受 Delta Dental 保险", "zh") == [
        ("zh", "我们接受 "),
        ("en", "Delta Dental"),
        ("zh", " 保险"),
    ]


def test_cjk_runs_inside_english_are_tagged_chinese():
    assert _segment_by_script("Our address is 春田市 downtown", "en") == [
        ("en", "Our address is "),
        ("zh", "春田市"),
        ("en", " downtown"),
    ]


def test_single_script_text_is_one_segment():
    # The fast path: no second voice to load, no concatenation, no added
    # latency for the overwhelming majority of replies.
    assert _segment_by_script("我们周一至周五营业。", "zh") == [
        ("zh", "我们周一至周五营业。")]
    assert _segment_by_script("We are open weekdays.", "en") == [
        ("en", "We are open weekdays.")]


def test_digits_stay_with_the_primary_language():
    # "200 号 4 室" must be read in Chinese, not handed to the English voice
    # -- only letters mark a foreign run.
    assert _segment_by_script("地址是 200 号 4 室", "zh") == [
        ("zh", "地址是 200 号 4 室")]


def test_empty_text_is_no_segments():
    assert _segment_by_script("", "zh") == []


@needs_models
def test_mixed_text_is_spoken_by_both_voices():
    # The behavioural proof: routing the Latin run through the English voice
    # produces materially more audio than letting the Chinese voice swallow
    # it. Compares whole-reply synthesis against the same text with the
    # English stripped out, so it asserts on real rendered duration.
    import wave

    def seconds(wav: bytes) -> float:
        with wave.open(io.BytesIO(wav)) as f:
            return f.getnframes() / f.getframerate()

    tts = PiperTTS()
    mixed = tts.synthesize("我们接受 Delta Dental 保险", "zh")
    assert seconds(mixed) > seconds(tts.synthesize("我们接受 保险", "zh")) + 0.5


@needs_models
def test_mixed_output_is_a_single_playable_wav():
    # Concatenation must produce one valid file, not spliced headers --
    # app.js hands the bytes straight to decodeAudioData.
    import wave

    tts = PiperTTS()
    with wave.open(io.BytesIO(tts.synthesize("接受 Cigna 和 Aetna", "zh"))) as f:
        assert f.getnchannels() == 1
        assert f.getframerate() == 22050
        assert f.getsampwidth() == 2
        assert f.getnframes() > 0


@needs_models
def test_punctuation_only_segments_do_not_abort_the_reply():
    # Piper emits zero audio for punctuation-only input. Found live: the "、"
    # between two brand names made the whole reply fail with wave's opaque
    # "# channels not specified".
    tts = PiperTTS()
    assert len(tts.synthesize("接受 Cigna、MetLife 和 Aetna", "zh")) > 0


@needs_models
def test_text_with_no_speakable_content_raises_a_clear_error():
    tts = PiperTTS()
    with pytest.raises(TTSError, match="no audio"):
        tts.synthesize("，。、", "zh")
