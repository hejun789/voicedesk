import time

from voicedesk.lang import DEFAULT_LANG, normalize_lang

DEFAULT_TTL_S = 1800  # 30 minutes


class SessionStore:
    """Maps a caller's session id to their Agent, so conversation history
    accumulates across turns (a caller can say "book me Monday" on one turn and
    give their name on the next). Idle sessions expire so the map cannot grow
    without bound.

    `agent_factory` is a one-arg callable (lang) returning a fresh Agent.
    `clock` is injectable so expiry can be tested without sleeping.
    """

    def __init__(self, agent_factory, ttl_s: float = DEFAULT_TTL_S,
                 clock=time.monotonic):
        self._agent_factory = agent_factory
        self._ttl_s = ttl_s
        self._clock = clock
        self._sessions: dict[tuple, tuple] = {}  # (id, lang) -> (agent, last_used_at)

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

    def _expire(self) -> None:
        now = self._clock()
        stale = [
            sid for sid, (_, last_used) in self._sessions.items()
            if now - last_used > self._ttl_s
        ]
        for sid in stale:
            del self._sessions[sid]

    def __len__(self) -> int:
        return len(self._sessions)
