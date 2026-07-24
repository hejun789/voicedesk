import re

_STOP = {"what", "are", "your", "the", "is", "do", "you", "a", "an", "to", "of", "we"}


def _tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]+", text.lower()) if w not in _STOP}


def _ngrams(text: str, n: int = 2) -> set[str]:
    """Overlapping character n-grams, with whitespace and punctuation removed.

    Chinese has no spaces between words, so word tokenization yields nothing and
    there is nothing to split on. Overlapping character sequences give a usable
    similarity signal without needing a segmenter or a new dependency.
    """
    cleaned = re.sub(r"[\s\W_]+", "", text.lower())
    return {cleaned[i:i + n] for i in range(len(cleaned) - n + 1)}


def _sections(doc: str) -> list[tuple[str, str]]:
    parts = re.split(r"^##\s+", doc, flags=re.MULTILINE)
    out = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        title, _, body = part.partition("\n")
        out.append((title.strip(), body.strip()))
    return out


# A section title is curated keywords; a body match can be incidental. Without
# this weight, the pronoun gram 们的 (from 我们的 in the hours body) ties 1-1
# with the genuine 地址 match for "你们的地址在哪里？", and the earlier section
# wins by accident. English does not need it — _STOP already drops pronouns.
_TITLE_WEIGHT = 3


def _score_words(q: set[str], title: str, body: str) -> int:
    return len(q & _tokens(title + " " + body))


def _score_ngrams(q: set[str], title: str, body: str) -> int:
    return _TITLE_WEIGHT * len(q & _ngrams(title)) + len(q & _ngrams(body))


# Bilingual bridge for the handful of topics the clinic doc actually covers.
# A query can be English even mid-Chinese-conversation (the model sometimes
# calls answer_faq("location") instead of a Chinese phrase), but the literal
# English word may never appear in a Chinese-only section body -- there is no
# lexical overlap for word- or n-gram-scoring to find. This maps a known
# English topic word straight to its section's Chinese anchor term.
_TOPIC_ALIASES: dict[str, str] = {
    "location": "地址", "address": "地址", "where": "地址",
    "hours": "营业时间", "time": "营业时间", "open": "营业时间",
    "insurance": "保险",
    "services": "服务项目", "service": "服务项目",
}


def answer_faq(query: str, doc_path: str = "clinic_info.md") -> str:
    with open(doc_path, encoding="utf-8") as f:
        doc = f.read()

    # Non-Latin script is the signal to fall back to n-grams, not "empty word
    # tokenization" -- an all-stopword English query (e.g. "what do you do")
    # also tokenizes to nothing and must still escalate, not fall through to
    # a scored n-gram match against the wrong section.
    q = _tokens(query)
    score = _score_words
    if not q and re.search(r"[^\x00-\x7F]", query):
        q, score = _ngrams(query), _score_ngrams
    if not q:
        return "NO_MATCH"

    best_body, best_score = "NO_MATCH", 0
    for title, body in _sections(doc):
        s = score(q, title, body)
        if s > best_score:
            best_body, best_score = body, s

    if best_score == 0 and score is _score_words:
        anchors = {_TOPIC_ALIASES[w] for w in q if w in _TOPIC_ALIASES}
        alias_q = {g for anchor in anchors for g in _ngrams(anchor)}
        if alias_q:
            for title, body in _sections(doc):
                s = _score_ngrams(alias_q, title, body)
                if s > best_score:
                    best_body, best_score = body, s

    return best_body if best_score > 0 else "NO_MATCH"
