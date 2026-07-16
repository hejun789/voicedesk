# VoiceDesk Phase 5 — Chinese Language Support

**Design spec** · 2026-07-16 · Status: approved, ready for planning

## Purpose

Let a caller book, reschedule, cancel and ask questions **in Chinese**, by voice, with the
same agent and the same guarantees as English — and measure it with a mirrored Chinese eval
suite so the claim is backed by data rather than a demo.

This is also a test of the architecture's central claim: tools know nothing about the LLM,
the agent knows nothing about audio, and neither knows about language. If those boundaries
are real, Chinese support is additive.

## Why the current system is English-only

Not a config flag — four genuine blockers, one of them fundamental:

1. **FAQ retrieval is blind to Chinese.** `faq.py` tokenizes with `re.findall(r"[a-z]+")`,
   which matches ASCII only. Verified: `_tokens("你们的营业时间是什么时候")` returns an
   **empty set**, so every Chinese question scores 0 and returns `NO_MATCH`. Worse, Chinese
   has no spaces between words, so even a Unicode-aware regex would not yield words.
2. **`stt.py` hard-codes `language="en"`** and an English-only transcription prompt.
3. **`clinic_info.md` is English**, so there is nothing for Chinese retrieval to match.
4. **The browser never sets `utterance.lang`**, so speech synthesis uses the page default.

## Foundational decisions

1. **Language is explicit configuration, never inferred.** A `lang` value (`"en"` | `"zh"`)
   flows from a browser toggle to the server and drives four choices. Whisper *can*
   auto-detect, but it is unreliable on short utterances (a two-word reply like "好的" can
   misdetect), and it would make every downstream choice depend on a guess — untestable
   offline. Explicit is deterministic and lets an eval scenario pin the language.
2. **English keeps its existing word-matching; character n-grams are a *fallback*.**
   English FAQ retrieval measures 92–100% today. Replacing its tokenizer risks regressing a
   working, measured component. Instead: if word tokenization yields nothing (the signal for
   a non-Latin script), fall back to character 2-grams. English behavior is byte-for-byte
   unchanged; Chinese gets a path that works. Pure stdlib, no new dependencies.
3. **The full 30 English scenarios are mirrored in Chinese**, and the eval CLI gains a
   `--lang` filter so each language's suite runs independently. This matters: 60 scenarios ×
   3 runs = 180 runs ≈ 2 days of free-tier quota, but `--lang en --runs 3` and
   `--lang zh --runs 3` are ~90 runs each — the same cost as today's suite, one day apiece.
   The payoff is a true per-language comparison on identical scenarios.
4. **The Chinese is drafted here and corrected by the project's author**, who is a native
   speaker. A scenario that reads like a translation measures the wrong thing.

## Language flow

```
[ EN | 中文 ]  browser toggle
    │  POST /turn (audio, session_id, lang=zh)
    ▼
 1. Whisper        language="zh" + Chinese transcription prompt
 2. FAQ document   clinic_info.zh.md
 3. System prompt  built with lang="zh" → "reply in Chinese"
 4. Response       lang echoed back → browser TTS uses a zh-CN voice
```

## Components and changes

| File | Change |
|---|---|
| `src/voicedesk/lang.py` | **new** — `LANGUAGES = ("en", "zh")`, `DEFAULT_LANG = "en"`, `FAQ_DOCS = {"en": "clinic_info.md", "zh": "clinic_info.zh.md"}`, `normalize_lang(value) -> str` (unknown/None → `"en"`). One place that knows the languages. |
| `src/voicedesk/faq.py` | add `_ngrams(text, n=2)`; `answer_faq` uses the existing word path when `_tokens(query)` is non-empty, else the n-gram path. `NO_MATCH` semantics unchanged. |
| `clinic_info.zh.md` | **new** — Chinese clinic document, same four sections. |
| `src/voicedesk/agent.py` | `build_system_prompt(today, lang=DEFAULT_LANG)` — appends a reply-language instruction and Chinese digit read-back guidance for `zh`. `Agent(conn, llm, system_prompt=None, faq_doc_path=None)` — passes `faq_doc_path` to `dispatch`. |
| `src/voicedesk/registry.py` | `dispatch(conn, name, args, faq_doc_path=None)` — the **caller** supplies the FAQ document. The model must not choose files; today it could pass `doc_path` in its args, which is wrong on principle. When `faq_doc_path` is given it takes precedence. |
| `src/voicedesk/tools.py` | `_PLACEHOLDER_VALUES` gains Chinese equivalents (`未知`, `无`, `没有`). |
| `src/voicedesk/voice/stt.py` | `transcribe(audio, filename="audio.webm", language=DEFAULT_LANG)`; `TRANSCRIPTION_PROMPTS = {"en": ..., "zh": ...}`; `SILENCE_HALLUCINATIONS` gains the Chinese artifacts. |
| `src/voicedesk/voice/server.py` | `lang` form field (normalized, defaults to `en`); session key becomes `(session_id, lang)`; `DIDNT_CATCH` / `STT_FAILED` gain Chinese variants; `lang` echoed in the response. |
| `src/voicedesk/voice/static/` | EN / 中文 toggle; `lang` sent with each turn; `utterance.lang` set from the response. |
| `evals/scenarios.json` | each scenario gains an optional `"lang"` (default `"en"`); 30 new Chinese scenarios, ids prefixed `zh_`, mirroring the English ids/seeds/expectations. |
| `src/voicedesk/evals/runner.py` | builds the agent with the scenario's language (system prompt + FAQ doc). |
| `src/voicedesk/evals/__main__.py` | `--lang {en,zh}` filter. |

## Chinese-specific behavior

- **Whisper hallucinates Chinese artifacts on silence** — most famously `谢谢观看`
  ("thanks for watching", learned from YouTube subtitles), plus
  `字幕由Amara.org社区提供` and similar. The existing denylist is English-only and would
  not catch these, so silence would be fed to the booking tools.
- **Digit read-back in Chinese.** The confirm-before-booking contract still applies; the
  prompt instructs reading the phone number back digit by digit in Chinese.
- **Brand names stay in Latin script** in `clinic_info.zh.md` (Cigna, Delta Dental), which
  is how they are actually said on a call — and means an English word token in an otherwise
  Chinese query still matches.

## Session semantics

The session key becomes `(session_id, lang)`. Switching language mid-call starts a fresh
conversation rather than mixing languages in one history. This is deliberate: a language
switch is a new context, and it keeps the stored history coherent for the model.

## Eval

All 30 English scenarios are mirrored with `zh_` ids and `"lang": "zh"`, keeping the same
seeds, the same `expect` blocks, and the same dates (the week of Monday 2026-07-13). Only
the `turns` are Chinese. Identical expectations are what make the per-language comparison
meaningful.

Run them independently:
```
python -m voicedesk.evals --lang en --runs 3    # ~90 runs, ~1 day of quota
python -m voicedesk.evals --lang zh --runs 3    # ~90 runs, ~1 day of quota
```

**Known limitation, stated plainly:** roughly two-thirds of the mirrored scenarios exercise
language-independent tool logic (weekend refusal, the double-booking guard, placeholder
rejection) — the tools do not know what language they are in. Mirroring them still verifies
the *agent* behaves correctly in Chinese end to end, and an incomplete mirror would make the
comparison messy, but the marginal information per scenario is lower there than for the FAQ
and escalation cases.

## Testing strategy

Everything stays **offline** — no network, no API key, no microphone:

- `faq.py` — `_ngrams` unit tests; a Chinese question retrieves the right section from a
  Chinese fixture doc; **English retrieval is asserted unchanged** (regression guard).
- `lang.py` — `normalize_lang` maps unknown/None/garbage to `"en"`.
- `agent.py` — `build_system_prompt(today, "zh")` contains the reply-in-Chinese instruction;
  the English prompt is unchanged.
- `registry.py` — `dispatch(..., faq_doc_path=...)` reads the given document and takes
  precedence over a model-supplied `doc_path`.
- `stt.py` — the language and the matching transcription prompt reach the API (asserted on a
  fake client's recorded kwargs); Chinese silence artifacts are recognized.
- `server.py` — `lang=zh` reaches STT and is echoed back; an unknown lang falls back to
  `en`; the same `session_id` under two languages yields two separate agents.
- `scenarios.json` — 60 scenarios, unique ids, every `zh_` scenario has `"lang": "zh"` and a
  Chinese `turns` array; existing structural guards still apply.

## Explicitly out of scope (YAGNI)

Automatic language detection, a third language, mixed-language conversations within one
turn, Cantonese, translating `docs/`, and a Chinese voice-UI beyond the toggle and labels.

## Success criteria

- Speaking Chinese into the browser books a real appointment, and the agent replies in
  Chinese with a Chinese voice.
- A Chinese FAQ question retrieves the right section instead of escalating.
- `--lang zh --runs 3` produces a scored report comparable to the English one.
- English behavior is measurably unchanged — the English suite still passes as before.
- The full test suite still runs offline with no API key.
