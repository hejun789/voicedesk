# VoiceDesk Phase 5 — Chinese Support — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a caller speak Chinese to the same agent — book, reschedule, cancel, ask questions — and measure it with a mirrored Chinese eval suite.

**Architecture:** Language is explicit configuration (`"en"` | `"zh"`), never inferred. One new `lang.py` module owns the language table. A `lang` value flows from a browser toggle → the server → four decisions: Whisper's language, the FAQ document, the system prompt, and the TTS voice. English FAQ retrieval keeps its word-matching untouched; character 2-grams are a **fallback** used only when word tokenization yields nothing (the signal for a non-Latin script).

**Tech Stack:** Python 3.11+, stdlib `re` only (no new dependencies), existing FastAPI/Groq stack, pytest.

## Global Constraints

- **Cost $0.** Groq free tier only. **No new dependencies** — character n-grams are stdlib; do NOT add `jieba` or an embedding model.
- **English behavior must be measurably unchanged.** English FAQ retrieval measures 92–100% today; the n-gram path must never run for an English query. Every existing test must pass unweakened.
- **The test suite stays fully offline** — no network, no API key, no microphone.
- **The model never chooses files.** `dispatch` takes the FAQ document from its caller.
- Languages are exactly `("en", "zh")`. `DEFAULT_LANG = "en"`. Unknown/None normalizes to `"en"`.
- Scenario dates stay in the week of Monday **2026-07-13**; clinic hours weekdays 09:00–16:00.
- Tests run as `PYTHONPATH=src ./.venv/Scripts/python.exe -m pytest ...` from the repo root via Bash (Git Bash), NOT PowerShell.
- Existing suite is **208 passing** before this phase. It must stay green.
- This repo's commits are **single-author** — never add a `Co-Authored-By` trailer.
- TDD throughout; commit after each green task.

---

### Task 1: The language module and the Chinese clinic document

**Files:**
- Create: `src/voicedesk/lang.py`
- Create: `clinic_info.zh.md`
- Test: `tests/test_lang.py`

**Interfaces:**
- Produces:
  - `LANGUAGES: tuple[str, ...]` = `("en", "zh")`
  - `DEFAULT_LANG: str` = `"en"`
  - `FAQ_DOCS: dict[str, str]` = `{"en": "clinic_info.md", "zh": "clinic_info.zh.md"}`
  - `normalize_lang(value: str | None) -> str` — returns the value when it is a known language, else `DEFAULT_LANG`. Case-insensitive, tolerates surrounding whitespace and a region suffix (`"zh-CN"` → `"zh"`).
  - `faq_doc_for(lang: str | None) -> str` — the FAQ document path for a language, normalizing first.

- [ ] **Step 1: Write the failing test** — `tests/test_lang.py`

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src ./.venv/Scripts/python.exe -m pytest tests/test_lang.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'voicedesk.lang'`

- [ ] **Step 3: Implement** — `src/voicedesk/lang.py`

```python
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
```

- [ ] **Step 4: Write `clinic_info.zh.md`** — same four sections as the English document. Brand names stay in Latin script, which is how they are actually said on a call.

```markdown
## 营业时间
我们的营业时间是周一至周五，上午9点到下午5点。周末和公众假期休息。

## 地址
BrightSmile 牙科诊所位于 Springfield，Market Street 200号4室。大楼后方有免费停车位。

## 保险
我们接受 Delta Dental、Cigna、MetLife 和 Aetna 保险，也提供自费方案。就诊时请携带您的保险卡。

## 服务项目
我们提供洗牙、补牙、牙冠、牙齿美白，以及牙科急诊服务。
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `PYTHONPATH=src ./.venv/Scripts/python.exe -m pytest tests/test_lang.py -v`
Expected: PASS (5 passed)

- [ ] **Step 6: Commit**

```bash
git add src/voicedesk/lang.py clinic_info.zh.md tests/test_lang.py
git commit -m "feat: language table and Chinese clinic document"
```

---

### Task 2: FAQ retrieval that can see Chinese

The current tokenizer is `re.findall(r"[a-z]+", text.lower())` — ASCII only. `_tokens("你们的营业时间是什么时候")` returns an **empty set**, so every Chinese question scores 0 and returns `NO_MATCH`. Chinese also has no spaces between words, so a Unicode-aware regex would not yield words either.

**Files:**
- Modify: `src/voicedesk/faq.py`
- Test: `tests/test_faq_chinese.py`

**Interfaces:**
- Consumes: the existing `_tokens(text) -> set[str]`, `_sections(doc) -> list[tuple[str, str]]`, `answer_faq(query, doc_path="clinic_info.md") -> str`, and the `"NO_MATCH"` sentinel.
- Produces: `_ngrams(text: str, n: int = 2) -> set[str]` — overlapping character n-grams with whitespace and punctuation stripped. `answer_faq` keeps its signature; it selects the word path when `_tokens(query)` is non-empty (English — behavior unchanged), else the n-gram path (Chinese).

- [ ] **Step 1: Write the failing test** — `tests/test_faq_chinese.py`

```python
import pytest
from voicedesk.faq import answer_faq, _ngrams, _tokens

ZH_DOC = """## 营业时间
我们的营业时间是周一至周五，上午9点到下午5点。周末休息。

## 地址
BrightSmile 牙科诊所位于 Market Street 200号4室。

## 保险
我们接受 Delta Dental、Cigna、MetLife 和 Aetna 保险。
"""


@pytest.fixture
def zh_doc(tmp_path):
    p = tmp_path / "clinic_zh.md"
    p.write_text(ZH_DOC, encoding="utf-8")
    return str(p)


def test_ngrams_of_chinese():
    assert _ngrams("营业时间") == {"营业", "业时", "时间"}


def test_ngrams_strip_whitespace_and_punctuation():
    # Punctuation must not become part of a gram, or scoring gets noisy.
    assert _ngrams("营业，时间？") == {"营业", "业时", "时间"}


def test_ngrams_of_short_text_is_empty():
    assert _ngrams("好") == set()


def test_a_common_pronoun_does_not_outrank_a_real_title_match(zh_doc):
    # Regression guard for a real bug: "你们的地址在哪里？" shares the gram 们的
    # with 我们的 in the HOURS body, which ties 1-1 with the genuine 地址 match —
    # and the hours section comes first, so a flat score returns the wrong
    # section. Titles are curated keywords, so they must outweigh a body match.
    answer = answer_faq("你们的地址在哪里？", zh_doc)
    assert "Market Street" in answer
    assert "周一至周五" not in answer


def test_chinese_question_finds_hours(zh_doc):
    answer = answer_faq("你们的营业时间是什么时候？", zh_doc)
    assert "周一至周五" in answer


def test_chinese_question_finds_location(zh_doc):
    answer = answer_faq("你们的地址在哪里？", zh_doc)
    assert "Market Street" in answer


def test_chinese_question_finds_insurance(zh_doc):
    answer = answer_faq("你们接受保险吗？", zh_doc)
    assert "Cigna" in answer


def test_chinese_no_match_returns_sentinel(zh_doc):
    # Nothing in the document is about flights.
    assert answer_faq("你们卖飞机票吗", zh_doc) == "NO_MATCH"


def test_english_query_still_uses_the_word_path(tmp_path):
    # Regression guard: English retrieval is measured at 92-100% and must not
    # change. A query with word tokens must never fall through to n-grams.
    doc = tmp_path / "en.md"
    doc.write_text("## Hours\nOpen Monday to Friday 9am to 5pm.\n\n"
                   "## Location\nWe are located at 200 Market Street.\n",
                   encoding="utf-8")
    assert "Monday to Friday" in answer_faq("what are your opening hours", str(doc))
    assert "Market Street" in answer_faq("where are you located", str(doc))
    assert answer_faq("do you sell airplane tickets", str(doc)) == "NO_MATCH"
    # and the discriminator itself holds:
    assert _tokens("what are your opening hours")
    assert not _tokens("你们的营业时间")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src ./.venv/Scripts/python.exe -m pytest tests/test_faq_chinese.py -v`
Expected: FAIL — `ImportError: cannot import name '_ngrams'`

- [ ] **Step 3: Implement** — replace the whole of `src/voicedesk/faq.py` with:

```python
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
```

Note `_score_words` is exactly the old expression (`len(q & _tokens(title + " " + body))`), so the English path is unchanged — the regression test in Step 1 pins that.

- [ ] **Step 4: Run the new tests AND the existing FAQ tests**

Run: `PYTHONPATH=src ./.venv/Scripts/python.exe -m pytest tests/test_faq_chinese.py tests/test_faq.py -v`
Expected: PASS — the new Chinese tests plus every pre-existing English FAQ test, unchanged.

- [ ] **Step 5: Commit**

```bash
git add src/voicedesk/faq.py tests/test_faq_chinese.py
git commit -m "feat: character n-gram fallback so FAQ retrieval works in Chinese"
```

---

### Task 3: A language-aware prompt, and the caller chooses the FAQ document

**Files:**
- Modify: `src/voicedesk/agent.py`
- Modify: `src/voicedesk/registry.py`
- Modify: `src/voicedesk/tools.py`
- Test: `tests/test_agent_chinese.py`
- Test: `tests/test_registry_faq_doc.py`

**Interfaces:**
- Consumes: `DEFAULT_LANG`, `normalize_lang` from `voicedesk.lang`.
- Produces:
  - `build_system_prompt(today: date, lang: str = DEFAULT_LANG) -> str` — the existing English prompt, plus a language instruction appended when `lang == "zh"`. The English prompt (`lang="en"`) is **byte-for-byte unchanged**.
  - `Agent(conn, llm, system_prompt: str | None = None, faq_doc_path: str | None = None)` — stores `faq_doc_path` and passes it to `dispatch`.
  - `dispatch(conn, name, args, faq_doc_path: str | None = None) -> dict` — when `faq_doc_path` is given it takes precedence over any model-supplied `args["doc_path"]`.
  - `tools._PLACEHOLDER_VALUES` gains Chinese fillers.

- [ ] **Step 1: Write the failing tests** — `tests/test_agent_chinese.py`

```python
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
```

and `tests/test_registry_faq_doc.py`

```python
from voicedesk.registry import dispatch


def _doc(tmp_path, text):
    p = tmp_path / "doc.md"
    p.write_text(text, encoding="utf-8")
    return str(p)


def test_faq_doc_path_from_the_caller_is_used(tmp_path):
    doc = _doc(tmp_path, "## Hours\nOpen Monday to Friday.\n")
    res = dispatch(None, "answer_faq", {"query": "hours"}, faq_doc_path=doc)
    assert "Monday" in res["answer"]


def test_caller_doc_path_beats_a_model_supplied_one(tmp_path):
    # The model must never choose which file gets read.
    caller = _doc(tmp_path, "## Hours\nCaller document wins.\n")
    model = _doc(tmp_path, "## Hours\nModel document loses.\n")
    res = dispatch(None, "answer_faq",
                   {"query": "hours", "doc_path": model},
                   faq_doc_path=caller)
    assert "Caller document wins" in res["answer"]


def test_dispatch_without_faq_doc_path_still_honours_args(tmp_path):
    # Backwards compatible: existing callers pass doc_path in args.
    doc = _doc(tmp_path, "## Hours\nOpen Monday to Friday.\n")
    res = dispatch(None, "answer_faq", {"query": "hours", "doc_path": doc})
    assert "Monday" in res["answer"]


def test_agent_passes_its_faq_doc_path_to_the_tool(db, tmp_path):
    from voicedesk.agent import Agent
    from voicedesk.llm import FakeLLM, Message, ToolCall

    doc = _doc(tmp_path, "## Hours\nWe open at dawn.\n")
    llm = FakeLLM([
        Message(content=None, tool_calls=[
            ToolCall(id="1", name="answer_faq", arguments={"query": "hours"})]),
        Message(content="We open at dawn.", tool_calls=[]),
    ])
    agent = Agent(db, llm, faq_doc_path=doc)
    assert "dawn" in agent.respond("what are your hours")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src ./.venv/Scripts/python.exe -m pytest tests/test_agent_chinese.py tests/test_registry_faq_doc.py -v`
Expected: FAIL — `TypeError: build_system_prompt() takes 1 positional argument but 2 were given`, and `dispatch() got an unexpected keyword argument 'faq_doc_path'`

- [ ] **Step 3: Make `build_system_prompt` language-aware** — in `src/voicedesk/agent.py`, add this import at the top alongside the existing ones:

```python
from voicedesk.lang import DEFAULT_LANG, normalize_lang
```

Change the signature from `def build_system_prompt(today: date) -> str:` to:

```python
def build_system_prompt(today: date, lang: str = DEFAULT_LANG) -> str:
```

Then, at the very END of the function, replace the final `)` of the returned string expression so the English text is assigned and the language instruction is appended. Concretely, change the tail of the function from:

```python
        "Keep replies short and natural, as if speaking on a phone call."
    )
```

to:

```python
        "Keep replies short and natural, as if speaking on a phone call."
    )
    if normalize_lang(lang) == "zh":
        prompt += (
            " 来电者说中文。请全程用简体中文回复。"
            "向来电者复述电话号码时，请逐个数字用中文读出（例如“五五五一二三四”）。"
            "姓名如果是英文，按原样保留。"
        )
    return prompt
```

and change the `return (` that opens the string expression to `prompt = (` so the value is captured. (The English text between them is untouched — `build_system_prompt(today)` and `build_system_prompt(today, "en")` must return exactly what it returned before.)

- [ ] **Step 4: Give `Agent` a FAQ document** — in `src/voicedesk/agent.py`, change the constructor:

```python
class Agent:
    def __init__(self, conn, llm: LLMClient, system_prompt: str | None = None,
                 faq_doc_path: str | None = None):
        self.conn = conn
        self.llm = llm
        # Which clinic document answer_faq reads. The caller decides (it knows
        # the language); the model must never choose a file.
        self.faq_doc_path = faq_doc_path
        if system_prompt is None:
            system_prompt = build_system_prompt(date.today())
        self.messages: list[dict] = [{"role": "system", "content": system_prompt}]
```

and in `respond`, change the dispatch call from `dispatch(self.conn, tc.name, tc.arguments)` to:

```python
                    result = dispatch(self.conn, tc.name, tc.arguments,
                                      faq_doc_path=self.faq_doc_path)
```

- [ ] **Step 5: Let the caller supply the document** — in `src/voicedesk/registry.py`, change the signature from `def dispatch(conn, name: str, args: dict) -> dict:` to:

```python
def dispatch(conn, name: str, args: dict, faq_doc_path: str | None = None) -> dict:
```

and replace the `answer_faq` branch's first two lines. Change:

```python
    if name == "answer_faq":
        kwargs = {"doc_path": args["doc_path"]} if "doc_path" in args else {}
        answer = answer_faq(args["query"], **kwargs)
```

to:

```python
    if name == "answer_faq":
        # The caller chooses the document, never the model — otherwise a model
        # could name an arbitrary file in its tool arguments.
        doc_path = faq_doc_path or args.get("doc_path")
        kwargs = {"doc_path": doc_path} if doc_path else {}
        answer = answer_faq(args["query"], **kwargs)
```

Leave the rest of that branch (the `NO_MATCH` note, the plain `{"answer": ...}` return) exactly as it is.

- [ ] **Step 6: Add Chinese placeholder fillers** — in `src/voicedesk/tools.py`, add these entries to the existing `_PLACEHOLDER_VALUES` set (keep every existing entry):

```python
    "未知", "无", "没有", "姓名", "电话", "电话号码",
```

- [ ] **Step 7: Run the new tests plus every test that touches these modules**

Run: `PYTHONPATH=src ./.venv/Scripts/python.exe -m pytest tests/test_agent_chinese.py tests/test_registry_faq_doc.py tests/test_agent.py tests/test_registry.py tests/test_tools_book.py -v`
Expected: PASS — new tests green and all pre-existing agent/registry/tools tests unchanged.

- [ ] **Step 8: Commit**

```bash
git add src/voicedesk/agent.py src/voicedesk/registry.py src/voicedesk/tools.py tests/test_agent_chinese.py tests/test_registry_faq_doc.py
git commit -m "feat: language-aware system prompt, and the caller picks the FAQ document"
```

---

### Task 4: Chinese speech-to-text

**Files:**
- Modify: `src/voicedesk/voice/stt.py`
- Test: `tests/test_voice_stt_chinese.py`

**Interfaces:**
- Consumes: `DEFAULT_LANG`, `normalize_lang` from `voicedesk.lang`.
- Produces:
  - `TRANSCRIPTION_PROMPTS: dict[str, str]` — an English and a Chinese biasing prompt. `TRANSCRIPTION_PROMPT` stays as an alias of the English one (existing tests import it).
  - `STTClient.transcribe(audio: bytes, filename: str = "audio.webm", language: str = DEFAULT_LANG) -> str` — the protocol, `FakeSTT`, and `GroqWhisper` all gain the `language` parameter.
  - `GroqWhisper.transcribe` sends the normalized language and the matching prompt (an explicit constructor `prompt=` still overrides both languages).
  - `SILENCE_HALLUCINATIONS` gains Whisper's Chinese artifacts; `is_silence_hallucination` strips Chinese punctuation before comparing.

- [ ] **Step 1: Write the failing test** — `tests/test_voice_stt_chinese.py`

```python
from types import SimpleNamespace
from voicedesk.voice.stt import (
    FakeSTT, GroqWhisper, TRANSCRIPTION_PROMPTS, is_silence_hallucination,
)


class _FakeAudioClient:
    def __init__(self, text="好的"):
        self.calls = []
        outer = self
        self.audio = SimpleNamespace(
            transcriptions=SimpleNamespace(
                create=lambda **kw: outer._next(kw, text))
        )

    def _next(self, kwargs, text):
        self.calls.append(kwargs)
        return SimpleNamespace(text=text)


def test_transcribe_sends_chinese_language_and_prompt():
    client = _FakeAudioClient()
    GroqWhisper(client=client).transcribe(b"x", "turn.webm", language="zh")
    sent = client.calls[0]
    assert sent["language"] == "zh"
    assert sent["prompt"] == TRANSCRIPTION_PROMPTS["zh"]
    assert "牙科" in sent["prompt"]


def test_transcribe_defaults_to_english():
    client = _FakeAudioClient(text="hello")
    GroqWhisper(client=client).transcribe(b"x")
    assert client.calls[0]["language"] == "en"
    assert client.calls[0]["prompt"] == TRANSCRIPTION_PROMPTS["en"]


def test_transcribe_normalizes_an_unknown_language():
    client = _FakeAudioClient()
    GroqWhisper(client=client).transcribe(b"x", "turn.webm", language="klingon")
    assert client.calls[0]["language"] == "en"


def test_explicit_prompt_overrides_both_languages():
    client = _FakeAudioClient()
    GroqWhisper(client=client, prompt="custom").transcribe(b"x", language="zh")
    assert client.calls[0]["prompt"] == "custom"


def test_fake_stt_accepts_a_language():
    stt = FakeSTT(["你好"])
    assert stt.transcribe(b"x", "turn.webm", language="zh") == "你好"


def test_chinese_silence_hallucinations_are_recognized():
    # Whisper emits these on silence — famously "thanks for watching", learned
    # from YouTube subtitles. Feeding them to the agent would be noise.
    assert is_silence_hallucination("谢谢观看")
    assert is_silence_hallucination("谢谢观看。")
    assert is_silence_hallucination("字幕由Amara.org社区提供")


def test_real_chinese_speech_is_not_treated_as_silence():
    assert not is_silence_hallucination("我要预约洗牙")
    assert not is_silence_hallucination("好的")   # a real confirmation
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src ./.venv/Scripts/python.exe -m pytest tests/test_voice_stt_chinese.py -v`
Expected: FAIL — `ImportError: cannot import name 'TRANSCRIPTION_PROMPTS'`

- [ ] **Step 3: Add the language table and prompts** — in `src/voicedesk/voice/stt.py`, add this import at the top:

```python
from voicedesk.lang import DEFAULT_LANG, normalize_lang
```

Keep the existing `TRANSCRIPTION_PROMPT` exactly as it is (tests import it), and add below it:

```python
TRANSCRIPTION_PROMPT_ZH = (
    "一通打给 BrightSmile 牙科诊所的电话。来电者会说出自己的姓名、"
    "电话号码、日期和时间，以及就诊原因，例如洗牙、补牙、牙冠、"
    "检查或牙齿美白。"
)

TRANSCRIPTION_PROMPTS = {
    "en": TRANSCRIPTION_PROMPT,
    "zh": TRANSCRIPTION_PROMPT_ZH,
}
```

- [ ] **Step 4: Add the Chinese silence artifacts** — replace the existing `SILENCE_HALLUCINATIONS` set and `is_silence_hallucination` function with:

```python
# Whisper emits these instead of an empty string on silence or noise. The
# Chinese ones come from its YouTube subtitle training data.
SILENCE_HALLUCINATIONS = {
    "thank you.", "thank you", "you", "bye.", "bye",
    "thanks for watching!", ".", "so",
    "谢谢观看", "谢谢大家观看", "请不吝点赞", "明镜与点点栏目",
    "字幕由amara.org社区提供", "字幕志愿者", "小编",
}

# Chinese sentences end in these; strip them before comparing.
_TRAILING_PUNCT = " 。．，,！!？?、…～~"


def is_silence_hallucination(text: str) -> bool:
    """True when a transcript is one of Whisper's known silence artefacts."""
    return text.strip().strip(_TRAILING_PUNCT).strip().lower() in SILENCE_HALLUCINATIONS
```

- [ ] **Step 5: Thread the language through** — in `src/voicedesk/voice/stt.py`, update the protocol, the fake, and the real client.

Change the protocol:

```python
class STTClient(Protocol):
    def transcribe(self, audio: bytes, filename: str = "audio.webm",
                   language: str = DEFAULT_LANG) -> str: ...
```

Change `FakeSTT.transcribe`:

```python
    def transcribe(self, audio: bytes, filename: str = "audio.webm",
                   language: str = DEFAULT_LANG) -> str:
        return self._scripted.pop(0)
```

In `GroqWhisper.__init__`, change `self.prompt = prompt or TRANSCRIPTION_PROMPT` to:

```python
        # None means "pick the prompt for the call's language"; an explicit
        # prompt overrides that for every language.
        self.prompt = prompt
```

and replace `GroqWhisper.transcribe` with:

```python
    def transcribe(self, audio: bytes, filename: str = "audio.webm",
                   language: str = DEFAULT_LANG) -> str:
        lang = normalize_lang(language)
        try:
            resp = self.client.audio.transcriptions.create(
                file=(filename, audio),
                model=self.model,
                language=lang,
                temperature=0,
                prompt=self.prompt or TRANSCRIPTION_PROMPTS[lang],
            )
        except Exception as e:  # noqa: BLE001 - translated to STTError
            raise STTError(str(e)) from e
        return (getattr(resp, "text", "") or "").strip()
```

- [ ] **Step 6: Run the new tests plus the existing STT tests**

Run: `PYTHONPATH=src ./.venv/Scripts/python.exe -m pytest tests/test_voice_stt_chinese.py tests/test_voice_stt.py -v`
Expected: PASS — new tests green and every pre-existing STT test unchanged.

- [ ] **Step 7: Commit**

```bash
git add src/voicedesk/voice/stt.py tests/test_voice_stt_chinese.py
git commit -m "feat: Chinese speech-to-text with its own biasing prompt and silence artefacts"
```

---

### Task 5: The language reaches the browser

**Files:**
- Modify: `src/voicedesk/voice/session.py`
- Modify: `src/voicedesk/voice/server.py`
- Modify: `src/voicedesk/voice/__main__.py`
- Modify: `src/voicedesk/voice/static/index.html`
- Modify: `src/voicedesk/voice/static/app.js`
- Test: `tests/test_voice_server_chinese.py`
- Modify: `tests/test_voice_session.py`

**Interfaces:**
- Consumes: `DEFAULT_LANG`, `normalize_lang`, `faq_doc_for` from `voicedesk.lang`; `build_system_prompt(today, lang)`; `Agent(conn, llm, system_prompt=..., faq_doc_path=...)`; `stt.transcribe(audio, filename, language)`.
- Produces:
  - `SessionStore.get_or_create(session_id: str, lang: str = DEFAULT_LANG)` — the session key becomes `(session_id, lang)`, and `agent_factory` is now called as `agent_factory(lang)`. **Breaking change to the factory contract** — the existing tests' zero-arg factories must become one-arg.
  - `POST /turn` accepts a `lang` form field (normalized; defaults to `"en"`) and echoes `"lang"` in the response JSON.
  - `DIDNT_CATCH_ZH`, `STT_FAILED_ZH`; the existing `DIDNT_CATCH` / `STT_FAILED` remain the English strings (tests import them).

- [ ] **Step 1: Write the failing test** — `tests/test_voice_server_chinese.py`

```python
import sqlite3
import pytest
from fastapi.testclient import TestClient

from voicedesk.db import init_db
from voicedesk.agent import Agent, build_system_prompt
from voicedesk.lang import faq_doc_for
from voicedesk.llm import FakeLLM, Message
from voicedesk.voice.stt import FakeSTT
from voicedesk.voice.session import SessionStore
from voicedesk.voice.server import create_app, DIDNT_CATCH_ZH


class _RecordingSTT:
    """Records the language it was asked to transcribe in."""

    def __init__(self, text="我要预约"):
        self.text = text
        self.languages = []

    def transcribe(self, audio, filename="audio.webm", language="en"):
        self.languages.append(language)
        return self.text


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:", check_same_thread=False)
    c.row_factory = sqlite3.Row
    init_db(c)
    yield c
    c.close()


def _client(conn, stt, llm):
    from datetime import date
    sessions = SessionStore(lambda lang: Agent(
        conn, llm,
        system_prompt=build_system_prompt(date(2026, 7, 10), lang),
        faq_doc_path=faq_doc_for(lang),
    ))
    return TestClient(create_app(stt, sessions))


def _post(client, lang="en", session_id="s1"):
    data = {"session_id": session_id}
    if lang is not None:
        data["lang"] = lang
    return client.post("/turn", data=data,
                       files={"audio": ("turn.webm", b"x" * 2000, "audio/webm")})


def test_lang_reaches_the_transcriber_and_is_echoed_back(conn):
    stt = _RecordingSTT()
    llm = FakeLLM([Message(content="好的，请问您的姓名？", tool_calls=[])])
    body = _post(_client(conn, stt, llm), lang="zh").json()
    assert stt.languages == ["zh"]
    assert body["lang"] == "zh"
    assert body["reply"] == "好的，请问您的姓名？"


def test_lang_defaults_to_english_when_absent(conn):
    stt = _RecordingSTT(text="hello")
    llm = FakeLLM([Message(content="Hi!", tool_calls=[])])
    body = _post(_client(conn, stt, llm), lang=None).json()
    assert stt.languages == ["en"]
    assert body["lang"] == "en"


def test_unknown_lang_falls_back_to_english(conn):
    stt = _RecordingSTT(text="hello")
    llm = FakeLLM([Message(content="Hi!", tool_calls=[])])
    body = _post(_client(conn, stt, llm), lang="klingon").json()
    assert stt.languages == ["en"]
    assert body["lang"] == "en"


def test_didnt_catch_is_spoken_in_chinese(conn):
    # A stray tap must apologise in the caller's language.
    stt = FakeSTT([])   # never called — the audio is too small
    llm = FakeLLM([])
    client = _client(conn, stt, llm)
    r = client.post("/turn", data={"session_id": "s1", "lang": "zh"},
                    files={"audio": ("turn.webm", b"tiny", "audio/webm")})
    body = r.json()
    assert body["reply"] == DIDNT_CATCH_ZH
    assert body["lang"] == "zh"


def test_same_session_id_in_two_languages_is_two_conversations(conn):
    stt = _RecordingSTT()
    llm = FakeLLM([Message(content="a", tool_calls=[]),
                   Message(content="b", tool_calls=[])])
    client = _client(conn, stt, llm)
    _post(client, lang="en", session_id="s1")
    _post(client, lang="zh", session_id="s1")
    # Two separate agents were built — a language switch is a new context, not
    # a mixed-language history.
    assert stt.languages == ["en", "zh"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src ./.venv/Scripts/python.exe -m pytest tests/test_voice_server_chinese.py -v`
Expected: FAIL — `ImportError: cannot import name 'DIDNT_CATCH_ZH'`

- [ ] **Step 3: Key sessions by language** — in `src/voicedesk/voice/session.py`, add the import:

```python
from voicedesk.lang import DEFAULT_LANG, normalize_lang
```

and replace `get_or_create` with:

```python
    def get_or_create(self, session_id: str, lang: str = DEFAULT_LANG):
        """The caller's Agent for this language. A language switch is a new
        context, so it gets its own conversation rather than a mixed history."""
        self._expire()
        now = self._clock()
        lang = normalize_lang(lang)
        key = (session_id, lang)
        entry = self._sessions.get(key)
        if entry is None:
            agent = self._agent_factory(lang)
        else:
            agent = entry[0]
        self._sessions[key] = (agent, now)
        return agent
```

- [ ] **Step 4: Update BOTH existing call sites for the new factory contract** — the factory now receives a language, so every zero-arg factory must accept one. There are exactly two in the test suite; do not change any assertion in either file.

In `tests/test_voice_session.py`, the `_counter_factory` helper's inner function — change `def factory():` to:

```python
    def factory(lang="en"):
```

In `tests/test_voice_server.py:35`, the `_client` helper — change:

```python
    sessions = SessionStore(lambda: Agent(conn, llm))
```

to:

```python
    sessions = SessionStore(lambda lang: Agent(conn, llm))
```

(This helper backs all 8 existing voice-server tests; without this they fail with `TypeError: <lambda>() takes 0 positional arguments but 1 was given`.)

- [ ] **Step 5: Accept a language in the server** — in `src/voicedesk/voice/server.py`, add the import:

```python
from voicedesk.lang import DEFAULT_LANG, normalize_lang
```

Keep the existing `DIDNT_CATCH` and `STT_FAILED` strings and add below them:

```python
DIDNT_CATCH_ZH = "抱歉，我没有听清，可以再说一遍吗？"
STT_FAILED_ZH = "抱歉，我听不清楚。我让同事回电给您。"

_DIDNT_CATCH = {"en": DIDNT_CATCH, "zh": DIDNT_CATCH_ZH}
_STT_FAILED = {"en": STT_FAILED, "zh": STT_FAILED_ZH}
```

Change the handler signature to accept the field:

```python
    async def turn(
        session_id: str = Form(...),
        audio: UploadFile = File(...),
        lang: str = Form(DEFAULT_LANG),
    ):
        started = time.perf_counter()
        lang = normalize_lang(lang)
        data = await audio.read()
```

Then add `"lang": lang` to **every** returned dict in the handler, and swap the three message constants for their per-language lookups:
- the too-small/too-large branch → `"reply": _DIDNT_CATCH[lang]`
- the `except STTError` branch → `"reply": _STT_FAILED[lang]`
- the empty/hallucinated-transcript branch → `"reply": _DIDNT_CATCH[lang]`
- the normal branch → unchanged reply, plus `"lang": lang`

Pass the language to the transcriber:

```python
            transcript = await run_in_threadpool(
                stt.transcribe, data, "turn.webm", lang)
```

and to the session:

```python
        def _run_agent() -> str:
            with lock:
                agent = sessions.get_or_create(session_id, lang)
                return agent.respond(transcript)
```

- [ ] **Step 6: Build language-aware agents in the entrypoint** — in `src/voicedesk/voice/__main__.py`, add the imports:

```python
from datetime import date
from voicedesk.agent import Agent, build_system_prompt
from voicedesk.lang import faq_doc_for
```

(remove the plain `from voicedesk.agent import Agent` line if it is now duplicated) and change the `SessionStore` construction from `SessionStore(lambda: Agent(conn, GroqLLM(on_retry=_log_retry)))` to:

```python
    sessions = SessionStore(lambda lang: Agent(
        conn,
        GroqLLM(on_retry=_log_retry),
        system_prompt=build_system_prompt(date.today(), lang),
        faq_doc_path=faq_doc_for(lang),
    ))
```

- [ ] **Step 7: Add the toggle to the page** — in `src/voicedesk/voice/static/index.html`, add this CSS inside the existing `<style>` block:

```css
    #langs { display: flex; gap: .5rem; margin-bottom: 1rem; }
    .lang { flex: 1; padding: .6rem; border-radius: 8px; border: 1px solid #ccc;
            background: #fff; cursor: pointer; font-size: .95rem; }
    .lang.active { background: #1e40af; border-color: #1e40af; color: #fff; }
```

and insert this immediately before the `<button id="talk">` line:

```html
  <div id="langs">
    <button class="lang active" data-lang="en">English</button>
    <button class="lang" data-lang="zh">中文</button>
  </div>
```

- [ ] **Step 8: Send and speak the language** — in `src/voicedesk/voice/static/app.js`:

Add after the existing `const sessionId = crypto.randomUUID();` line:

```javascript
let lang = "en";
const BCP47 = { en: "en-US", zh: "zh-CN" };

document.querySelectorAll(".lang").forEach((btn) => {
  btn.addEventListener("click", () => {
    lang = btn.dataset.lang;
    document.querySelectorAll(".lang").forEach((b) =>
      b.classList.toggle("active", b === btn));
    talk.textContent = lang === "zh" ? "按住说话" : "Hold to talk";
  });
});
```

In `send(blob)`, add the language to the form — change:

```javascript
  form.append("session_id", sessionId);
```

to:

```javascript
  form.append("session_id", sessionId);
  form.append("lang", lang);
```

Change `speak(data.reply)` to `speak(data.reply, data.lang)`, and replace the `speak` function with:

```javascript
function speak(text, replyLang) {
  // Browser TTS: starts instantly, costs nothing, adds no network latency.
  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.lang = BCP47[replyLang] || BCP47.en;
  utterance.rate = 1.05;
  window.speechSynthesis.speak(utterance);
}
```

Finally, make the two hard-coded English UI strings language-aware. In `send(blob)`, change the tiny-blob bail-out block's two lines to:

```javascript
    transcriptEl.textContent = lang === "zh" ? "（没有听清）" : "(didn't catch that)";
    replyEl.textContent = lang === "zh"
      ? "抱歉，我没有听清，可以再说一遍吗？"
      : "Sorry, I didn't catch that. Could you say that again?";
```

- [ ] **Step 9: Run the new tests plus every voice test**

Run: `PYTHONPATH=src ./.venv/Scripts/python.exe -m pytest tests/test_voice_server_chinese.py tests/test_voice_server.py tests/test_voice_session.py -v`
Expected: PASS — new tests green and every pre-existing voice test unchanged.

- [ ] **Step 10: Confirm the entrypoint still imports cleanly (no API key)**

Run: `PYTHONPATH=src ./.venv/Scripts/python.exe -c "import voicedesk.voice.__main__; print('ok')"`
Expected: prints `ok`

- [ ] **Step 11: Commit**

```bash
git add src/voicedesk/voice tests/test_voice_server_chinese.py tests/test_voice_session.py
git commit -m "feat: language toggle from the browser through to Whisper and the TTS voice"
```

---

### Task 6: The eval speaks both languages

**Files:**
- Modify: `src/voicedesk/evals/runner.py`
- Modify: `src/voicedesk/evals/__main__.py`
- Test: `tests/test_evals_lang.py`

**Interfaces:**
- Consumes: `normalize_lang`, `faq_doc_for` from `voicedesk.lang`; `build_system_prompt(today, lang)`; `Agent(conn, llm, system_prompt=..., faq_doc_path=...)`.
- Produces:
  - `run_scenario_once` builds the agent with the scenario's language: `build_system_prompt(EVAL_TODAY, scenario.get("lang"))` and `faq_doc_for(scenario.get("lang"))`. A scenario without a `lang` is English — every existing scenario keeps working untouched.
  - `select_by_lang(scenarios: list[dict], lang: str | None) -> list[dict]` in `__main__.py` — filters to one language; `None` returns everything.
  - `--lang {en,zh}` CLI flag.

- [ ] **Step 1: Write the failing test** — `tests/test_evals_lang.py`

```python
import pytest
from voicedesk.llm import FakeLLM, Message
from voicedesk.evals.runner import run_scenario_once
from voicedesk.evals.__main__ import select_by_lang


class _CapturingLLM:
    """Captures the system prompt the agent was built with."""

    def __init__(self):
        self.system_prompt = None

    def complete(self, messages, tools):
        self.system_prompt = messages[0]["content"]
        return Message(content="ok", tool_calls=[])


def test_chinese_scenario_builds_a_chinese_agent():
    llm = _CapturingLLM()
    run_scenario_once(
        {"id": "zh_x", "category": "faq", "lang": "zh",
         "turns": ["你们的营业时间？"], "expect": {}},
        llm,
    )
    assert "简体中文" in llm.system_prompt


def test_scenario_without_lang_is_english():
    llm = _CapturingLLM()
    run_scenario_once(
        {"id": "x", "category": "faq", "turns": ["what are your hours?"],
         "expect": {}},
        llm,
    )
    assert "简体中文" not in llm.system_prompt


SCENARIOS = [
    {"id": "book_oneshot", "turns": ["x"]},
    {"id": "zh_book_oneshot", "lang": "zh", "turns": ["x"]},
]


def test_select_by_lang_filters_chinese():
    assert [s["id"] for s in select_by_lang(SCENARIOS, "zh")] == ["zh_book_oneshot"]


def test_select_by_lang_filters_english_including_unlabelled():
    # A scenario with no lang field is English.
    assert [s["id"] for s in select_by_lang(SCENARIOS, "en")] == ["book_oneshot"]


def test_select_by_lang_none_returns_everything():
    assert select_by_lang(SCENARIOS, None) == SCENARIOS
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src ./.venv/Scripts/python.exe -m pytest tests/test_evals_lang.py -v`
Expected: FAIL — `ImportError: cannot import name 'select_by_lang'`

- [ ] **Step 3: Build the agent in the scenario's language** — in `src/voicedesk/evals/runner.py`, add the import:

```python
from voicedesk.lang import faq_doc_for
```

and in `run_scenario_once`, change the agent construction from
`agent = Agent(conn, capturing, system_prompt=build_system_prompt(EVAL_TODAY))` to:

```python
    lang = scenario.get("lang")
    agent = Agent(
        conn, capturing,
        system_prompt=build_system_prompt(EVAL_TODAY, lang),
        faq_doc_path=faq_doc_for(lang),
    )
```

(`build_system_prompt` and `faq_doc_for` both normalize `None` to English, so scenarios without a `lang` field are unaffected.)

- [ ] **Step 4: Add the filter and the flag** — in `src/voicedesk/evals/__main__.py`, add after `select_scenarios`:

```python
def select_by_lang(scenarios: list[dict], lang: str | None) -> list[dict]:
    """Filter to one language. A scenario with no `lang` field is English."""
    if lang is None:
        return scenarios
    return [s for s in scenarios if s.get("lang", "en") == lang]
```

Add the argument next to the existing ones:

```python
    p.add_argument("--lang", default=None, choices=["en", "zh"],
                   help="run only this language's scenarios")
```

and change the scenario selection line from
`scenarios = select_scenarios(load_scenarios(args.scenarios), args.scenario)` to:

```python
    scenarios = select_by_lang(load_scenarios(args.scenarios), args.lang)
    scenarios = select_scenarios(scenarios, args.scenario)
```

- [ ] **Step 5: Run the new tests plus the existing eval tests**

Run: `PYTHONPATH=src ./.venv/Scripts/python.exe -m pytest tests/test_evals_lang.py tests/test_evals_runner.py tests/test_evals_cli.py -v`
Expected: PASS

- [ ] **Step 6: Confirm the CLI shows the new flag**

Run: `PYTHONPATH=src ./.venv/Scripts/python.exe -m voicedesk.evals --help`
Expected: exits 0 and lists `--lang {en,zh}`

- [ ] **Step 7: Commit**

```bash
git add src/voicedesk/evals tests/test_evals_lang.py
git commit -m "feat: per-language eval scenarios and a --lang filter"
```

---

### Task 7: The 30 Chinese scenarios

Pure data. Every Chinese scenario mirrors an English one: **same `category`, same `seed`, same `expect`, same dates** — only the `turns` are Chinese, and the id gains a `zh_` prefix. Identical expectations are what make the per-language comparison meaningful.

**Files:**
- Modify: `evals/scenarios.json`
- Modify: `tests/test_scenarios_file.py`

**Interfaces:**
- Consumes: `load_scenarios`, `fresh_db`, `seed_db` from `voicedesk.evals.runner`.
- Produces: `evals/scenarios.json` grows from 30 to **60** scenarios. The 30 new ones each carry `"lang": "zh"`.

- [ ] **Step 1: Write the failing test** — add to `tests/test_scenarios_file.py`

```python
def test_has_sixty_scenarios(scenarios):
    assert len(scenarios) == 60


def test_every_english_scenario_has_a_chinese_mirror(scenarios):
    by_id = {s["id"]: s for s in scenarios}
    english = [s for s in scenarios if s.get("lang", "en") == "en"]
    assert len(english) == 30
    for s in english:
        mirror = by_id.get(f"zh_{s['id']}")
        assert mirror is not None, f"no Chinese mirror for {s['id']}"
        # The comparison is only meaningful if the expectations are identical.
        assert mirror["expect"] == s["expect"], s["id"]
        assert mirror.get("seed") == s.get("seed"), s["id"]
        assert mirror["category"] == s["category"], s["id"]


def test_chinese_scenarios_are_labelled_and_actually_chinese(scenarios):
    zh = [s for s in scenarios if s["id"].startswith("zh_")]
    assert len(zh) == 30
    for s in zh:
        assert s["lang"] == "zh", s["id"]
        # Every turn must contain CJK characters — a turn left in English would
        # silently measure the wrong thing.
        for turn in s["turns"]:
            assert any("一" <= ch <= "鿿" for ch in turn), s["id"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src ./.venv/Scripts/python.exe -m pytest tests/test_scenarios_file.py -v`
Expected: FAIL — `assert 30 == 60`

- [ ] **Step 3: Generate the mirrors by COPYING the English scenarios**

Do NOT hand-write the `expect` blocks. The mirror test asserts each Chinese scenario's
`expect`, `seed` and `category` are **identical** to its English source, so derive them by
copying rather than retyping — retyping is how they drift.

Create `scripts/gen_zh_scenarios.py` with exactly this content, run it once, then delete it
(it is a one-off generator, not part of the product):

```python
"""One-off: mirror every English scenario into Chinese.

Copies category/seed/expect verbatim from the English source so the two
languages are directly comparable, and replaces only id, lang and turns.
"""
import copy
import json

ZH_TURNS = {
    "book_oneshot": [
        "帮我预约2026年7月13日星期一早上9点，我叫Jane Doe，电话5551234，洗牙。",
        "对，没错。",
    ],
    "book_multi_turn": [
        "你好，我想预约看牙。",
        "2026年7月13日星期一早上10点可以吗？",
        "我叫John Smith，电话5559876。",
        "补牙。",
        "对，没错。",
    ],
    "book_after_checking_availability": [
        "2026年7月14日星期二你们还有哪些时间？",
        "那就下午2点吧。我叫Mary Lee，电话5552222，做检查。",
        "对，没错。",
    ],
    "book_afternoon_slot": [
        "我要预约2026年7月15日星期三下午3点，我叫Tom Ray，电话5553333，做牙冠。",
        "对，没错。",
    ],
    "book_earliest_available": [
        "帮我约2026年7月16日星期四最早的时间，我叫Anna Kim，电话5554444，洗牙。",
        "对，没错。",
    ],
    "book_last_slot_of_day": [
        "2026年7月17日星期五下午4点，我叫Sam Poe，电话5555555，牙齿美白。",
        "对，没错。",
    ],
    "faq_hours": ["你们的营业时间是什么时候？"],
    "faq_location": ["你们诊所在哪里？"],
    "faq_insurance": ["你们接受Cigna保险吗？"],
    "faq_services": ["你们做牙齿美白吗？"],
    "reschedule_same_day": [
        "我想改一下我的预约时间。",
        "我叫Jane Doe，电话5551234。",
        "能不能改成同一天上午11点？",
        "对，麻烦改一下。",
    ],
    "reschedule_other_day": [
        "把我的预约改到2026年7月14日星期二早上9点。我叫John Smith，电话5559876。",
        "对，麻烦改一下。",
    ],
    "reschedule_no_appointment": ["我想改一下我的预约。我叫Bob Nobody，电话5550000。"],
    "cancel_by_phone": [
        "我要取消我的预约。",
        "我叫Jane Doe，电话5551234。",
        "对，麻烦取消。",
    ],
    "cancel_by_name": ["麻烦帮Mary Lee取消预约。", "对，麻烦取消。"],
    "cancel_no_appointment": ["取消我的预约。我叫Ghost Person，电话5559999。", "对，没错。"],
    "lookup_by_phone": ["我有预约吗？我的电话是5551234。"],
    "lookup_by_name": ["我有没有预约？我叫Tom Ray。"],
    "book_taken_slot": ["帮我约2026年7月13日星期一早上9点。我叫Carl New，电话5556666，洗牙。"],
    "book_saturday_rejected": ["帮我约2026年7月18日星期六早上10点。我叫Rita Sun，电话5557777，洗牙。"],
    "book_outside_hours_rejected": ["帮我约2026年7月13日星期一晚上8点。我叫Neil Late，电话5558888，洗牙。"],
    "book_sunday_rejected": ["2026年7月19日星期天中午可以来吗？我叫Paul Rest，电话5551111，做检查。"],
    "escalate_medical_advice": ["我的牙齿流血很厉害，现在应该吃什么药？"],
    "escalate_gibberish": ["啊啊啊 哦哦 呃呃呃 12345 ？？？ 嘟嘟嘟"],
    "escalate_out_of_scope": ["你能告诉我我的保险单号是多少吗？"],
    "escalate_billing_dispute": ["我上次看诊被重复扣款了，我要马上退款。"],
    "escalate_medical_emergency": ["我昨天打的麻药好像过敏了，现在很不舒服。"],
    "ambiguous_then_abandons": ["我想预约。", "算了，不用了，再见。"],
    "ambiguous_vague_time": ["我下周想找个时间过去。"],
    "changed_mind_mid_call": [
        "帮我约2026年7月13日星期一早上9点。我叫Jane Doe，电话5551234，洗牙。",
        "对，没错。",
        "等一下，能改成11点吗？",
        "对，麻烦改一下。",
    ],
}

PATH = "evals/scenarios.json"

with open(PATH, encoding="utf-8") as f:
    scenarios = json.load(f)

english = [s for s in scenarios if s.get("lang", "en") == "en"]
missing = {s["id"] for s in english} - set(ZH_TURNS)
extra = set(ZH_TURNS) - {s["id"] for s in english}
assert not missing, f"no Chinese turns for: {sorted(missing)}"
assert not extra, f"Chinese turns for unknown ids: {sorted(extra)}"

mirrors = []
for s in english:
    m = copy.deepcopy(s)          # category, seed and expect come along verbatim
    m["id"] = "zh_" + s["id"]
    m["lang"] = "zh"
    m["turns"] = ZH_TURNS[s["id"]]
    mirrors.append(m)

with open(PATH, "w", encoding="utf-8") as f:
    json.dump(scenarios + mirrors, f, ensure_ascii=False, indent=2)
    f.write("\n")

print(f"wrote {len(scenarios) + len(mirrors)} scenarios ({len(mirrors)} Chinese)")
```

Run it from the repo root:

Run: `./.venv/Scripts/python.exe scripts/gen_zh_scenarios.py`
Expected: `wrote 60 scenarios (30 Chinese)`

Then remove the generator — the JSON is the source of truth from here on:

Run: `rm -f scripts/gen_zh_scenarios.py && rmdir scripts 2>/dev/null; true`

Two notes on the generator, both load-bearing:
- `ensure_ascii=False` — without it every Chinese character is escaped to a `\u` sequence and
  the file becomes unreadable, which defeats the point of a native speaker reviewing it.
- The `missing` / `extra` assertions fail loudly if an English scenario has no Chinese turns
  (or vice versa), so the mirror cannot be silently incomplete.

- [ ] **Step 4: Verify the JSON parses and the mirror is complete**

Run: `./.venv/Scripts/python.exe -c "import json;d=json.load(open('evals/scenarios.json',encoding='utf-8'));print(len(d),'scenarios;',len([s for s in d if s.get('lang')=='zh']),'zh')"`
Expected: `60 scenarios; 30 zh`

- [ ] **Step 5: Run the scenario-file tests**

Run: `PYTHONPATH=src ./.venv/Scripts/python.exe -m pytest tests/test_scenarios_file.py -v`
Expected: PASS — 60 scenarios, every English scenario mirrored with identical `expect`/`seed`/`category`, every Chinese turn containing CJK characters.

- [ ] **Step 6: Run the FULL suite**

Run: `PYTHONPATH=src ./.venv/Scripts/python.exe -m pytest -q`
Expected: PASS — everything green, fully offline.

- [ ] **Step 7: Commit**

```bash
git add evals/scenarios.json tests/test_scenarios_file.py
git commit -m "feat: mirror all 30 eval scenarios in Chinese"
```

---

### Task 8: Documentation

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: nothing. Documentation only.

- [ ] **Step 1: Document the language support** — in `README.md`, insert this section immediately before the `## Roadmap` section:

```markdown
## Bilingual (English + 中文)

The agent takes calls in English or Chinese. Language is **explicit configuration, not
inference** — a toggle on the page sends `lang` with each turn, which selects Whisper's
language, the clinic document, the system prompt, and the browser's TTS voice. Whisper can
auto-detect, but it is unreliable on short utterances and would make every downstream
choice depend on a guess.

The interesting part was **FAQ retrieval**. It scored English by word overlap — and
`_tokens("你们的营业时间")` returns an **empty set**, because the regex was `[a-z]+`. Even a
Unicode-aware regex would not have helped: Chinese has no spaces between words, so there is
nothing to split on. The fix is character 2-grams (`营业时间` → `{营业, 业时, 时间}`), used
**only as a fallback when word tokenization yields nothing**. English keeps its measured
word path untouched; Chinese gets a path that works; no new dependencies.

Also Chinese-specific: Whisper hallucinates `谢谢观看` ("thanks for watching", learned from
YouTube subtitles) on silence, so the silence denylist needed Chinese entries — otherwise
noise would reach the booking tools.

All 30 scenarios are mirrored in Chinese with **identical expectations**, so the two
languages are directly comparable. Each suite runs independently, which matters on a free
tier that allows roughly one full 3× run per day:

```powershell
python -m voicedesk.evals --lang en --runs 3
python -m voicedesk.evals --lang zh --runs 3
```
```

- [ ] **Step 2: Mark the phase in the roadmap** — in `README.md`, add this line to the Roadmap list, immediately after the Phase 3 entry:

```markdown
- **Phase 5 — Chinese** ✅ bilingual voice (EN/中文): explicit language config, character
  n-gram FAQ retrieval for Chinese, Chinese silence-artefact handling, and all 30 eval
  scenarios mirrored with identical expectations for a true per-language comparison
```

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: bilingual support and why Chinese FAQ retrieval needed n-grams"
```

---

## Phase 5 Definition of Done

- Speaking Chinese into the browser books a real appointment; the agent replies in Chinese with a Chinese voice.
- A Chinese FAQ question retrieves the right section instead of escalating.
- `python -m voicedesk.evals --lang zh --runs 3` produces a scored report comparable to the English one.
- **English behavior is unchanged** — `build_system_prompt(today) == build_system_prompt(today, "en")`, English FAQ retrieval still uses the word path, and every pre-existing test passes unweakened.
- The full test suite runs offline with no API key.
- No new dependencies.

## Post-implementation, for the human

**The Chinese needs a native speaker's review.** Read `evals/scenarios.json`'s `zh_`
scenarios and `clinic_info.zh.md`. If a turn reads like a translation rather than something
a real caller would say, fix it — a scenario that does not sound like a real call measures
the wrong thing.

## What comes next (separate plan)

- **Phase 4 — deploy:** hosted demo (needs HTTPS for microphone access), cost per resolution.
