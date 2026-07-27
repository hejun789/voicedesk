from voicedesk.voice.__main__ import fresh_db, build_session_store, browsable_url
from voicedesk.llm import FakeLLM
from voicedesk.tools import book, lookup_appt


def test_fresh_db_is_an_initialized_isolated_calendar():
    a, b = fresh_db(), fresh_db()
    book(a, "Jane Doe", "5551234", "2026-07-13T09:00", "cleaning")
    assert lookup_appt(a, phone="5551234")          # a has the booking
    assert lookup_appt(b, phone="5551234") == []    # b is a separate calendar


def test_each_session_gets_its_own_calendar():
    store = build_session_store(lambda: FakeLLM([]))
    a = store.get_or_create("visitorA", "en")
    b = store.get_or_create("visitorB", "en")
    assert a.conn is not b.conn                      # isolated connections
    book(a.conn, "Jane Doe", "5551234", "2026-07-13T09:00", "cleaning")
    assert lookup_appt(a.conn, phone="5551234")
    assert lookup_appt(b.conn, phone="5551234") == []   # no cross-visitor leak


def test_same_session_reuses_its_calendar():
    store = build_session_store(lambda: FakeLLM([]))
    first = store.get_or_create("visitorA", "en")
    again = store.get_or_create("visitorA", "en")
    assert first is again                            # history + calendar persist


def test_chinese_session_gets_a_chinese_prompt():
    store = build_session_store(lambda: FakeLLM([]))
    agent = store.get_or_create("v", "zh")
    assert "简体中文" in agent.messages[0]["content"]


def test_browsable_url_rewrites_the_wildcard_bind_address():
    # 0.0.0.0 means "listen on every interface"; it is not a destination a
    # browser can connect to (Chrome answers ERR_ADDRESS_INVALID). Printing it
    # verbatim sends the reader to a dead link, so show the loopback address
    # they can actually open.
    assert browsable_url("0.0.0.0", 7860) == "http://127.0.0.1:7860"


def test_browsable_url_rewrites_the_ipv6_wildcard_too():
    assert browsable_url("::", 7860) == "http://127.0.0.1:7860"


def test_browsable_url_leaves_a_real_host_alone():
    assert browsable_url("127.0.0.1", 8000) == "http://127.0.0.1:8000"
    assert browsable_url("example.com", 80) == "http://example.com:80"
