"""Phase 6C acceptance tests — ASD tiebreaker orchestration.

Locks in the tiebreaker rule table:

  lightasd agrees with heuristic → boost heuristic confidence
  lightasd disagrees + VLM available → VLM wins
  lightasd disagrees + no VLM → heuristic wins
  lightasd unavailable + VLM available → VLM wins
  lightasd unavailable + no VLM → heuristic wins
"""

from backend.services.asd_tiebreaker import (
    AmbiguityWindow,
    DEFAULT_MARGIN_THRESHOLD,
    HEURISTIC_BOOST_ON_AGREEMENT,
    TIEBREAKER_ENV,
    TiebreakerDecision,
    find_ambiguous_windows,
    resolve_tiebreaker,
    tiebreaker_enabled,
)


# ── find_ambiguous_windows ─────────────────────────────────────────


def test_tiebreaker_fires_only_on_low_margin():
    """Stable high-margin timeline → no windows emitted."""
    timeline = [(i * 0.2, 0.5) for i in range(50)]  # 10s of 0.5 margin
    windows = find_ambiguous_windows(timeline)
    assert windows == []


def test_tiebreaker_finds_single_ambiguous_window():
    # 10s stable, then 2s ambiguous at t=10..12, then stable.
    timeline = [(i * 0.2, 0.5) for i in range(50)]
    timeline += [(10.2 + i * 0.2, 0.15) for i in range(10)]  # 10.2..12.0s
    timeline += [(12.2 + i * 0.2, 0.5) for i in range(20)]
    windows = find_ambiguous_windows(timeline)
    assert len(windows) == 1
    assert 10.0 <= windows[0].start <= 10.5
    assert 11.5 <= windows[0].end <= 12.5
    assert windows[0].duration >= 0.5
    assert windows[0].margin_min < DEFAULT_MARGIN_THRESHOLD
    assert len(windows[0].sample_timestamps) == 1


def test_tiebreaker_skips_short_ambiguity():
    """Windows shorter than min_duration are dropped."""
    timeline = [(i * 0.2, 0.5) for i in range(20)]
    # Only 0.2s of ambiguity — below min_duration of 0.5s.
    timeline += [(4.2, 0.1)]
    timeline += [(4.6 + i * 0.2, 0.5) for i in range(20)]
    windows = find_ambiguous_windows(timeline)
    assert windows == []


def test_tiebreaker_multiple_windows():
    timeline = []
    timeline += [(0.2 + i * 0.2, 0.5) for i in range(10)]   # stable
    timeline += [(2.2 + i * 0.2, 0.1) for i in range(6)]     # amb 1
    timeline += [(3.4 + i * 0.2, 0.5) for i in range(10)]   # stable
    timeline += [(5.4 + i * 0.2, 0.1) for i in range(6)]     # amb 2
    timeline += [(6.6 + i * 0.2, 0.5) for i in range(10)]   # stable
    windows = find_ambiguous_windows(timeline)
    assert len(windows) == 2


# ── resolve_tiebreaker rule table ──────────────────────────────────


def _mk_window() -> AmbiguityWindow:
    return AmbiguityWindow(
        start=10.0, end=12.0, margin_min=0.15,
        sample_timestamps=[11.0],
    )


def test_resolve_lightasd_agrees_boosts_heuristic():
    window = _mk_window()
    decision = resolve_tiebreaker(
        window, heuristic_choice=1, lightasd_choice=1,
        heuristic_confidence=0.4,
    )
    assert decision.source == "lightasd_agrees"
    assert decision.chosen_speaker == 1
    assert decision.confidence == 0.4 + HEURISTIC_BOOST_ON_AGREEMENT


def test_resolve_lightasd_agrees_confidence_capped_at_1():
    window = _mk_window()
    decision = resolve_tiebreaker(
        window, heuristic_choice=1, lightasd_choice=1,
        heuristic_confidence=0.9,
    )
    assert decision.confidence == 1.0


def test_resolve_lightasd_disagrees_vlm_wins():
    window = _mk_window()
    decision = resolve_tiebreaker(
        window, heuristic_choice=1, lightasd_choice=2, vlm_choice=2,
    )
    assert decision.source == "vlm_tiebreak"
    assert decision.chosen_speaker == 2


def test_resolve_lightasd_disagrees_no_vlm_keeps_heuristic():
    window = _mk_window()
    decision = resolve_tiebreaker(
        window, heuristic_choice=1, lightasd_choice=2, vlm_choice=None,
    )
    assert decision.source == "heuristic"
    assert decision.chosen_speaker == 1


def test_resolve_lightasd_unavailable_vlm_wins():
    window = _mk_window()
    decision = resolve_tiebreaker(
        window, heuristic_choice=1, lightasd_choice=None, vlm_choice=3,
    )
    assert decision.source == "vlm_tiebreak"
    assert decision.chosen_speaker == 3


def test_resolve_all_signals_missing_keeps_heuristic():
    window = _mk_window()
    decision = resolve_tiebreaker(
        window, heuristic_choice=1, lightasd_choice=None, vlm_choice=None,
    )
    assert decision.source == "heuristic"
    assert decision.chosen_speaker == 1


# ── Env flag ───────────────────────────────────────────────────────


def test_tiebreaker_default_enabled(monkeypatch):
    """Default is ON — the tiebreaker path is strictly safer than
    unmodified heuristic."""
    monkeypatch.delenv(TIEBREAKER_ENV, raising=False)
    assert tiebreaker_enabled()


def test_tiebreaker_explicit_disable(monkeypatch):
    for val in ("0", "false", "no", "off", "FALSE"):
        monkeypatch.setenv(TIEBREAKER_ENV, val)
        assert not tiebreaker_enabled(), val


def test_tiebreaker_explicit_enable(monkeypatch):
    for val in ("1", "true", "yes", "on"):
        monkeypatch.setenv(TIEBREAKER_ENV, val)
        assert tiebreaker_enabled(), val
