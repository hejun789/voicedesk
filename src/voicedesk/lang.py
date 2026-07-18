"""The one place that knows which languages exist.

Language is explicit configuration, never inferred: Whisper can auto-detect,
but it is unreliable on short utterances and would make every downstream
choice depend on a guess.
"""

LANGUAGES = ("en", "zh")
DEFAULT_LANG = "en"

FAQ_DOCS = {
    "en": "clinic_info.md",
    "zh": "clinic_info.zh.md",
}


def normalize_lang(value: str | None) -> str:
    """Coerce a caller-supplied language to a known one. Anything unknown
    degrades to English rather than raising — a bad value must never end a call."""
    if not value:
        return DEFAULT_LANG
    base = str(value).strip().lower().split("-")[0]
    return base if base in LANGUAGES else DEFAULT_LANG


def faq_doc_for(lang: str | None) -> str:
    """The clinic document to answer FAQs from, for a language."""
    return FAQ_DOCS[normalize_lang(lang)]
