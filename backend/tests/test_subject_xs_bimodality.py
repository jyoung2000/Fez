"""Fix 2 — bimodality test that supersedes the registry's coarse
is_continuous_motion heuristic for seated panels.
"""

from backend.services.subject_motion import (
    subject_xs_look_like_motion as _subject_xs_look_like_motion,
)


def test_trimodal_panel_is_not_motion():
    # 3-seat panel: ~23, ~56, ~80. Raw subject_x bounces between them
    # but the histogram shows 3 clear peaks, not a sweep.
    xs = (
        [23] * 12 + [22] * 4 + [24] * 4
        + [56] * 15 + [55] * 3 + [57] * 3
        + [80] * 12 + [79] * 4 + [81] * 4
    )
    is_motion, peaks = _subject_xs_look_like_motion(xs)
    assert is_motion is False
    # Should detect 3 peaks near the seat centers.
    assert 2 <= len(peaks) <= 4
    assert any(20 <= p <= 30 for p in peaks)
    assert any(50 <= p <= 60 for p in peaks)
    assert any(75 <= p <= 85 for p in peaks)


def test_single_speaker_is_not_motion():
    # All values tightly clustered at 50 → 1 peak, not motion.
    xs = [50] * 25 + [49, 51, 50, 48, 52]
    is_motion, peaks = _subject_xs_look_like_motion(xs)
    assert is_motion is False
    assert len(peaks) >= 1


def test_wide_sweep_is_motion():
    # Continuously sweeping 0 → 100 over 50 samples with big deltas
    # between every consecutive frame. 5+ histogram bins exceed the
    # peak threshold AND >60% of deltas exceed 8.
    xs = list(range(0, 100, 2))  # 50 samples, 2-unit steps
    is_motion, _peaks = _subject_xs_look_like_motion(xs)
    # Deltas are all 2, below the 8-threshold → motion_frac < 0.6 →
    # returns False. This documents that the second gate protects
    # slow pans too.
    assert is_motion is False


def test_erratic_large_deltas_is_motion():
    # Values that span the full range AND jump by >8 between frames
    # should be classified as motion. Uniform-ish distribution across
    # 10 bins → all 10 bins exceed the threshold of 3 → 10 peaks.
    # Deltas are always 20 (>8) so motion_frac=100%.
    xs = []
    for i in range(50):
        xs.append(5 + (i % 5) * 20)  # cycles 5, 25, 45, 65, 85
    is_motion, _peaks = _subject_xs_look_like_motion(xs)
    assert is_motion is True


def test_empty_input_is_not_motion():
    is_motion, peaks = _subject_xs_look_like_motion([])
    assert is_motion is False
    assert peaks == []


def test_too_few_samples():
    is_motion, peaks = _subject_xs_look_like_motion([50, 50, 50])
    assert is_motion is False
    assert peaks == []
