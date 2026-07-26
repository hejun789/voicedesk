# VoiceDesk — Swap FAQ Retrieval to Local Embeddings

**Design spec** · 2026-07-26 · Status: approved, ready for planning

## Purpose

Replace `faq.py`'s keyword/n-gram FAQ matcher with semantic (embedding-based) retrieval, so
the system finds the right FAQ section by *meaning* rather than by literal token overlap —
free, with no external API or quota.

## Why the current system falls short

`faq.py` scores a query against a doc section by counting shared literal tokens: English words
(`_tokens`) or Chinese character 2-grams (`_ngrams`). This breaks down whenever a query and the
matching section share no literal characters — found via the live zh eval suite
(`zh_faq_location`, 0/3 reproducible failure): the model called `answer_faq("location")` in
English mid-Chinese-call, but the word "location" never appears anywhere in the Chinese-only
`地址` (address) section, so every section scored 0 and the caller was wrongly escalated instead
of answered.

The fix shipped for that (`bcb50c9`) was a hand-written `_TOPIC_ALIASES` dict mapping four known
English topic words to their Chinese section anchor. It closes the specific bug but not the
underlying class: it only covers four topics, requires a human to notice and hand-list every new
alias, and does nothing for a paraphrase nobody thought to list ("where are you" / "how do I get
there"). Embeddings solve the general problem: any phrasing that means the same thing lands near
the same vector, in either language, without a maintained list.

## Foundational decisions

1. **Full replacement, not a fallback layer.** The lexical matcher (`_tokens`, `_ngrams`,
   `_score_words`, `_score_ngrams`, `_TITLE_WEIGHT`, `_TOPIC_ALIASES`) is deleted outright.
   Keeping both as a two-path system would mean permanently maintaining the exact alias hack
   this change exists to remove. `_sections()` (splits the doc into title/body chunks) is kept —
   it's structural doc parsing, not lexical matching.
2. **Local ONNX model via `fastembed`, not `sentence-transformers`/PyTorch, not a hosted API.**
   The project's whole free-tier posture (Render free web service, ~512MB RAM ceiling) rules out
   a PyTorch dependency (large image, high memory) and rules out a hosted embedding API (a
   second external quota to run out of — the exact failure mode that just happened with Groq's
   daily cap). `fastembed` runs a small multilingual model via ONNX runtime: no API key, no rate
   limit, works fully offline once the model file is present.
3. **Deploy is conditional on a measured memory check, with an explicit rollback.** Before this
   ever reaches Render, it's built and run locally and the process's actual memory footprint
   with the model loaded is measured. If it doesn't comfortably fit the free tier, the change is
   not deployed and the current (already-live, working) keyword+alias system stays — this
   upgrade is a quality improvement, not worth risking the live demo going down.

## Architecture

```
answer_faq(query, doc_path)
    │
    ├─ 1. Load doc sections via _sections() (unchanged: title/body chunks)
    ├─ 2. Get-or-compute cached section embeddings for this doc_path
    │      (module-level cache: {doc_path: {title: vector}}, computed once per doc,
    │       not per call — a static 4-section doc never needs re-embedding)
    ├─ 3. Embed the query (same model, lazily loaded once at module level,
    │      not per call — model load is the expensive part)
    ├─ 4. Cosine similarity: query vector vs every cached section vector
    └─ 5. Return the closest section's body if similarity clears a threshold,
           else "NO_MATCH" (replaces the old `best_score > 0` check)
```

`answer_faq()`'s signature and callers (`registry.py`, `agent.py`) are unchanged — this is an
internal swap of the matching mechanism, not a redesign of the FAQ feature.

## Testing

Embeddings of fixed text are deterministic (no LLM sampling involved), so existing tests stay
exact-match, not flaky. `test_faq.py` and `test_faq_chinese.py` are rewritten against the new
matcher, keeping the same assertions (e.g. `"Market Street" in answer_faq(...)`,
`answer_faq(...) == "NO_MATCH"` for off-topic queries). Two new regression tests reproduce
today's real bug directly: `answer_faq("location", zh_doc)` and `answer_faq("clinic location",
zh_doc)` must both return the address section — now passing without any alias table. The
similarity threshold is set empirically: run known matches and known non-matches (e.g. "do you
sell airplane tickets") through the model locally and set the cutoff between the two score bands.

## Rollout

1. Add `fastembed` to `requirements.txt`; confirm the actual smallest multilingual model
   available via fastembed's supported-model list (not assumed from memory) — must support both
   English and Chinese.
2. Implement, run the full test suite locally, and measure real process memory with the model
   loaded.
3. Only if that fits comfortably under Render's free-tier ceiling: commit, push, verify the
   Actions CI run is green, and let Render redeploy.
4. If it doesn't fit: stop, do not deploy, keep the current system. Document the finding.

## Out of scope

- Re-running the full live eval suite against Groq to get a new headline pass-rate number
  (optional follow-up, not required to ship this)
- Any change to how `answer_faq` is called from `agent.py`/`registry.py`
- Multi-tenant or configurable FAQ docs beyond the existing two files
