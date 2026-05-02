"""Regression tests for ``choose_snap_centers`` (the helper that decides
between face_registry slot centers and bimodality peak centers).

The Tank vs Tyrese run had a clean 5-slot registry (24, 36, 53, 68, 79)
and a degraded 2-peak bimodality result (the VLM was rate-limited and
returned synthesized scenes whose ``subject_x`` clustered at center).
The previous logic threw the registry away because
``len(registry) > len(peaks) + 1``. The fix: bimodality peaks only win
when the registry is genuinely degenerate.
"""
from backend.services.subject_motion import choose_snap_centers


def test_sane_5_slot_registry_beats_2_peak_bimodality():
    """Five well-spaced registry slots should not be discarded for a
    2-peak bimodality result derived from degraded VLM scenes."""
    registry = [24.0, 36.0, 53.0, 68.0, 79.0]
    peaks = [55.0, 85.0]   # AI-attributed values clustered around center
    centers, used_peaks = choose_snap_centers(registry, peaks)
    assert centers == registry, (
        "A clean 5-slot registry must win over a 2-peak bimodality "
        f"derived from degraded VLM output; got {centers}"
    )
    assert used_peaks is False


def test_overfragmented_embedding_registry_yields_to_peaks():
    """11 close-packed embedding slots vs 2 clean bimodality peaks: peaks win."""
    registry = [14.0, 17.0, 18.0, 22.0, 35.0, 53.0, 67.0, 80.0, 82.0, 83.0, 85.0]
    peaks = [25.0, 75.0]
    centers, used_peaks = choose_snap_centers(registry, peaks)
    assert centers == peaks
    assert used_peaks is True


def test_empty_registry_uses_peaks():
    centers, used_peaks = choose_snap_centers([], [25.0, 75.0])
    assert centers == [25.0, 75.0]
    assert used_peaks is True


def test_empty_registry_and_no_peaks_returns_empty():
    centers, used_peaks = choose_snap_centers([], [])
    assert centers == []
    assert used_peaks is False


def test_modest_registry_kept_when_not_overfragmented():
    """Registry of 4 slots vs 2 peaks: 4 < 2*2, so registry is not
    over-fragmented — keep registry."""
    registry = [20.0, 40.0, 60.0, 80.0]
    peaks = [25.0, 75.0]
    centers, used_peaks = choose_snap_centers(registry, peaks)
    assert centers == registry
    assert used_peaks is False


def test_high_count_but_well_spaced_registry_kept():
    """6 well-spaced slots (no close pairs <8%) — even though 6 ≥ 2*3,
    the lack of close pairs means it's not the embedding-fragmentation
    signature, so registry wins."""
    registry = [10.0, 25.0, 40.0, 55.0, 70.0, 85.0]   # 15-pt spacing
    peaks = [20.0, 50.0, 80.0]
    centers, used_peaks = choose_snap_centers(registry, peaks)
    assert centers == registry
    assert used_peaks is False
