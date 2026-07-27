from fastapi.testclient import TestClient

from voicedesk.voice.server import create_app
from voicedesk.voice.session import SessionStore
from voicedesk.voice.stt import FakeSTT
from voicedesk.llm import FakeLLM
from voicedesk.agent import Agent


def _client(db):
    sessions = SessionStore(lambda lang: Agent(db, FakeLLM([])))
    return TestClient(create_app(FakeSTT([]), sessions))


def test_static_assets_are_served_with_no_store(db):
    # Browsers cache app.js aggressively. After a deploy (or a local edit) a
    # returning visitor silently runs OLD JavaScript, which is indistinguishable
    # from "the fix didn't work" -- this cost a full round of false-negative
    # manual testing. These files are a few KB; correctness beats the bytes.
    for asset in ("/static/app.js", "/static/vad.js", "/static/turn-state.js"):
        res = _client(db).get(asset)
        assert res.status_code == 200, asset
        assert "no-store" in res.headers.get("cache-control", ""), asset


def test_index_html_is_also_not_cached(db):
    res = _client(db).get("/")
    assert res.status_code == 200
    assert "no-store" in res.headers.get("cache-control", "")
