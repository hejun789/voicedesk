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


def answer_faq(query: str, doc_path: str = "clinic_info.md") -> str:
    with open(doc_path, encoding="utf-8") as f:
        doc = f.read()

    # An empty word tokenization is the signal for a non-Latin script. English
    # keeps the word path unchanged; only then do we fall back to n-grams.
    q = _tokens(query)
    score = _score_words
    if not q:
        q = _ngrams(query)
        score = _score_ngrams
    if not q:
        return "NO_MATCH"

    best_body, best_score = "NO_MATCH", 0
    for title, body in _sections(doc):
        s = score(q, title, body)
        if s > best_score:
            best_body, best_score = body, s
    return best_body if best_score > 0 else "NO_MATCH"
