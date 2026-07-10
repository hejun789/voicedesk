import re

_STOP = {"what", "are", "your", "the", "is", "do", "you", "a", "an", "to", "of", "we"}


def _tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]+", text.lower()) if w not in _STOP}


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


def answer_faq(query: str, doc_path: str = "clinic_info.md") -> str:
    with open(doc_path, encoding="utf-8") as f:
        doc = f.read()
    q = _tokens(query)
    best_body, best_score = "NO_MATCH", 0
    for title, body in _sections(doc):
        score = len(q & _tokens(title + " " + body))
        if score > best_score:
            best_body, best_score = body, score
    return best_body if best_score > 0 else "NO_MATCH"
