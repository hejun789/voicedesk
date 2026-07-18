from voicedesk.lang import (
    LANGUAGES, DEFAULT_LANG, FAQ_DOCS, normalize_lang, faq_doc_for,
)


def test_languages_are_exactly_en_and_zh():
    assert LANGUAGES == ("en", "zh")
    assert DEFAULT_LANG == "en"


def test_normalize_lang_accepts_known_languages():
    assert normalize_lang("en") == "en"
    assert normalize_lang("zh") == "zh"


def test_normalize_lang_is_forgiving():
    assert normalize_lang(" ZH ") == "zh"       # case + whitespace
    assert normalize_lang("zh-CN") == "zh"      # region suffix
    assert normalize_lang("en-US") == "en"


def test_normalize_lang_falls_back_to_default():
    # An unknown language must never crash a call — it degrades to English.
    assert normalize_lang(None) == "en"
    assert normalize_lang("") == "en"
    assert normalize_lang("klingon") == "en"
    assert normalize_lang("fr") == "en"


def test_faq_doc_for_each_language():
    assert faq_doc_for("en") == "clinic_info.md"
    assert faq_doc_for("zh") == "clinic_info.zh.md"
    assert faq_doc_for("nonsense") == "clinic_info.md"
    assert set(FAQ_DOCS) == set(LANGUAGES)
