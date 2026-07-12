from voicedesk.voice.session import SessionStore, DEFAULT_TTL_S


class _FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def _counter_factory():
    """Each call returns a distinct object, so we can prove identity/isolation."""
    made = []

    def factory():
        agent = object()
        made.append(agent)
        return agent

    return factory, made


def test_same_session_id_returns_the_same_agent():
    factory, made = _counter_factory()
    store = SessionStore(factory)
    a = store.get_or_create("s1")
    b = store.get_or_create("s1")
    assert a is b            # history must persist across turns
    assert len(made) == 1    # only one Agent was ever built


def test_different_session_ids_are_isolated():
    factory, made = _counter_factory()
    store = SessionStore(factory)
    assert store.get_or_create("s1") is not store.get_or_create("s2")
    assert len(made) == 2
    assert len(store) == 2


def test_idle_session_expires():
    factory, made = _counter_factory()
    clock = _FakeClock()
    store = SessionStore(factory, ttl_s=100, clock=clock)
    first = store.get_or_create("s1")
    clock.advance(101)
    second = store.get_or_create("s1")
    assert second is not first   # the old one expired, a new Agent was built
    assert len(made) == 2
    assert len(store) == 1       # the stale entry was dropped


def test_active_session_does_not_expire():
    factory, made = _counter_factory()
    clock = _FakeClock()
    store = SessionStore(factory, ttl_s=100, clock=clock)
    first = store.get_or_create("s1")
    clock.advance(60)
    again = store.get_or_create("s1")   # touching it refreshes last-used
    clock.advance(60)                    # 120s total, but only 60s idle
    assert store.get_or_create("s1") is first
    assert again is first
    assert len(made) == 1


def test_default_ttl_is_thirty_minutes():
    assert DEFAULT_TTL_S == 1800
