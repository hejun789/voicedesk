from datetime import date, datetime, timezone


def _utc_today() -> date:
    return datetime.now(timezone.utc).date()


class RateLimiter:
    """Per-UTC-day turn limiter for the public demo: caps each visitor IP and
    the whole demo, so one person (or a bot) cannot drain the Groq quota.

    In-memory only — a restart resets it, which is fine for a demo. `clock`
    returns today's UTC date and is injectable so day-rollover is testable.
    """

    def __init__(self, per_ip_limit: int = 8, global_limit: int = 200,
                 clock=_utc_today):
        self.per_ip_limit = per_ip_limit
        self.global_limit = global_limit
        self._clock = clock
        self._day = clock()
        self._ip_counts: dict[str, int] = {}
        self._global = 0

    def _roll(self) -> None:
        today = self._clock()
        if today != self._day:
            self._day = today
            self._ip_counts = {}
            self._global = 0

    def allow(self, ip: str) -> bool:
        """True (and counts the turn) when under both caps; False otherwise."""
        self._roll()
        if self._global >= self.global_limit:
            return False
        if self._ip_counts.get(ip, 0) >= self.per_ip_limit:
            return False
        self._ip_counts[ip] = self._ip_counts.get(ip, 0) + 1
        self._global += 1
        return True
