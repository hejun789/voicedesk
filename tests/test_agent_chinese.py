from datetime import date
from voicedesk.agent import build_system_prompt


def test_english_prompt_is_unchanged_by_the_lang_parameter():
    # Regression guard: the English prompt is measured; adding a parameter must
    # not alter it.
    assert build_system_prompt(date(2026, 7, 10)) == \
           build_system_prompt(date(2026, 7, 10), "en")


def test_english_prompt_has_no_chinese_instruction():
    prompt = build_system_prompt(date(2026, 7, 10), "en")
    assert "简体中文" not in prompt


def test_chinese_prompt_requires_replying_in_chinese():
    prompt = build_system_prompt(date(2026, 7, 10), "zh")
    assert "简体中文" in prompt


def test_chinese_prompt_keeps_every_english_instruction():
    # The zh prompt is the en prompt PLUS a language instruction — none of the
    # safety rules may be dropped in translation.
    en = build_system_prompt(date(2026, 7, 10), "en")
    zh = build_system_prompt(date(2026, 7, 10), "zh")
    assert zh.startswith(en)
    assert len(zh) > len(en)


def test_chinese_prompt_requires_digit_by_digit_readback_in_chinese():
    prompt = build_system_prompt(date(2026, 7, 10), "zh")
    assert "逐个数字" in prompt


def test_unknown_lang_falls_back_to_english_prompt():
    assert build_system_prompt(date(2026, 7, 10), "klingon") == \
           build_system_prompt(date(2026, 7, 10), "en")
