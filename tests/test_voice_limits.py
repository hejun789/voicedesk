from datetime import date
from voicedesk.voice.limits import RateLimiter


class _Clock:
    def __init__(self, d=date(2026, 7, 16)):
        self.d = d

    def __call__(self):
        return self.d


def test_allows_up_to_the_per_ip_limit_then_blocks():
    rl = RateLimiter(per_ip_limit=3, global_limit=100, clock=_Clock())
    assert [rl.allow("1.1.1.1") for _ in range(3)] == [True, True, True]
    assert rl.allow("1.1.1.1") is False       # 4th turn from same IP blocked
    assert rl.allow("1.1.1.1") is False       # stays blocked


def test_different_ips_are_counted_separately():
    rl = RateLimiter(per_ip_limit=1, global_limit=100, clock=_Clock())
    assert rl.allow("1.1.1.1") is True
    assert rl.allow("1.1.1.1") is False
    assert rl.allow("2.2.2.2") is True         # a different IP is unaffected


def test_global_limit_stops_everyone():
    rl = RateLimiter(per_ip_limit=100, global_limit=2, clock=_Clock())
    assert rl.allow("a") is True
    assert rl.allow("b") is True
    assert rl.allow("c") is False              # global budget spent, new IP still blocked


def test_a_blocked_turn_does_not_consume_budget():
    rl = RateLimiter(per_ip_limit=1, global_limit=100, clock=_Clock())
    assert rl.allow("1.1.1.1") is True
    assert rl.allow("1.1.1.1") is False
    # the blocked attempts did not eat into the global budget:
    assert rl.allow("2.2.2.2") is True


def test_counters_reset_when_the_utc_day_rolls_over():
    clock = _Clock(date(2026, 7, 16))
    rl = RateLimiter(per_ip_limit=1, global_limit=1, clock=clock)
    assert rl.allow("1.1.1.1") is True
    assert rl.allow("1.1.1.1") is False        # spent for the day
    clock.d = date(2026, 7, 17)                # next UTC day
    assert rl.allow("1.1.1.1") is True         # fresh budget
