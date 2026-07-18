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


def test_all_stopword_english_query_still_escalates(tmp_path):
    # Regression guard: "what do you do" is entirely composed of English stop
    # words, so _tokens(query) is empty -- the same signal used to detect
    # non-Latin script. The discriminator must be "non-Latin script", not
    # "empty word tokenization", or an all-stopword English caller utterance
    # gets answered with the wrong FAQ section instead of escalating.
    doc = tmp_path / "en.md"
    doc.write_text("## Hours\nOpen Monday to Friday 9am to 5pm.\n\n"
                   "## Location\nWe are located at 200 Market Street.\n",
                   encoding="utf-8")
    assert answer_faq("what do you do", str(doc)) == "NO_MATCH"
    assert answer_faq("are you the", str(doc)) == "NO_MATCH"


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
