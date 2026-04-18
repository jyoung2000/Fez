"""Gap 5a — tests for the diarization fusion helper.

Coverage:
  - Backend selection (6 tests)
  - Device resolution (5 tests)
  - Cluster→slot mapping (4 tests)
  - Fusion branches (agree / disagree / override / passthrough) (8 tests)
  - End-to-end JBP crosstalk scenario (1 test)
  - Tuning constants sanity (1 test)

All 25 tests run in the sandbox without numpy / librosa / pyannote /
torch / sklearn — every heavy dependency is monkeypatched.
"""

from __future__ import annotations

import pytest

from backend.services import speaker_diarization as sd
from backend.services.active_speaker import SpeakerEvent
from backend.services.speaker_diarization import (
    AGREE_CONFIDENCE_BOOST,
    AGREE_CONFIDENCE_CAP,
    CLUSTER_TO_SLOT_THRESHOLD,
    DIARIZATION_OVERRIDE_CONFIDENCE,
    DISAGREE_CONFIDENCE_CAP,
    DiarizationSegment,
    FusionResult,
    LIP_NOISE_FLOOR,
    MIN_VRAM_MB_FOR_CUDA,
    fuse_lip_and_diarization,
    map_clusters_to_slots,
)


# ───────────── Shared helpers ─────────────


def _ev(start: float, end: float, slot: int, conf: float) -> SpeakerEvent:
    return SpeakerEvent(
        start=start, end=end, slot_id=slot, confidence=conf,
    )


# ───────────────────── Backend selection (6) ─────────────────────


def test_backend_off_when_use_diarization_false(monkeypatch):
    monkeypatch.setattr(sd, "USE_DIARIZATION", False)
    monkeypatch.setattr(sd, "DIARIZATION_BACKEND", "auto")
    assert sd._select_backend() == "off"


def test_backend_off_when_explicit_off(monkeypatch):
    monkeypatch.setattr(sd, "USE_DIARIZATION", True)
    monkeypatch.setattr(sd, "DIARIZATION_BACKEND", "off")
    assert sd._select_backend() == "off"


def test_backend_mfcc_when_explicit(monkeypatch):
    monkeypatch.setattr(sd, "USE_DIARIZATION", True)
    monkeypatch.setattr(sd, "DIARIZATION_BACKEND", "mfcc")
    assert sd._select_backend() == "mfcc"


def test_backend_pyannote_when_explicit_and_available(monkeypatch):
    monkeypatch.setattr(sd, "USE_DIARIZATION", True)
    monkeypatch.setattr(sd, "DIARIZATION_BACKEND", "pyannote")
    monkeypatch.setattr(sd, "_pyannote_available", lambda: True)
    assert sd._select_backend() == "pyannote"


def test_backend_falls_to_off_when_pyannote_requested_but_unavailable(monkeypatch):
    monkeypatch.setattr(sd, "USE_DIARIZATION", True)
    monkeypatch.setattr(sd, "DIARIZATION_BACKEND", "pyannote")
    monkeypatch.setattr(sd, "_pyannote_available", lambda: False)
    assert sd._select_backend() == "off"


def test_backend_auto_prefers_pyannote_then_mfcc(monkeypatch):
    monkeypatch.setattr(sd, "USE_DIARIZATION", True)
    monkeypatch.setattr(sd, "DIARIZATION_BACKEND", "auto")
    # Pyannote takes precedence when available
    monkeypatch.setattr(sd, "_pyannote_available", lambda: True)
    assert sd._select_backend() == "pyannote"
    # Otherwise falls to mfcc (numpy path) — numpy is NOT installed in
    # the sandbox, so auto will short-circuit to "off". This is the
    # correct degraded behavior in a sandbox. We exercise the
    # "numpy present → mfcc" branch by monkeypatching the import.
    monkeypatch.setattr(sd, "_pyannote_available", lambda: False)
    import sys
    if "numpy" not in sys.modules:
        fake_numpy = type(sys)("numpy")
        monkeypatch.setitem(sys.modules, "numpy", fake_numpy)
    assert sd._select_backend() == "mfcc"


# ───────────────────── Device resolution (5) ─────────────────────


def test_device_cpu_explicit(monkeypatch):
    monkeypatch.setattr(sd, "DIARIZATION_DEVICE", "cpu")
    assert sd._resolve_device() == "cpu"


def test_device_cuda_explicit(monkeypatch):
    monkeypatch.setattr(sd, "DIARIZATION_DEVICE", "cuda")
    assert sd._resolve_device() == "cuda"


def test_device_auto_no_cuda_probe_falls_to_cpu(monkeypatch):
    monkeypatch.setattr(sd, "DIARIZATION_DEVICE", "auto")
    monkeypatch.setattr(sd, "_cuda_free_mb", lambda: None)
    assert sd._resolve_device() == "cpu"


def test_device_auto_low_vram_falls_to_cpu(monkeypatch):
    monkeypatch.setattr(sd, "DIARIZATION_DEVICE", "auto")
    monkeypatch.setattr(
        sd, "_cuda_free_mb", lambda: MIN_VRAM_MB_FOR_CUDA - 1,
    )
    assert sd._resolve_device() == "cpu"


def test_device_auto_high_vram_picks_cuda(monkeypatch):
    monkeypatch.setattr(sd, "DIARIZATION_DEVICE", "auto")
    monkeypatch.setattr(
        sd, "_cuda_free_mb", lambda: MIN_VRAM_MB_FOR_CUDA + 1,
    )
    assert sd._resolve_device() == "cuda"


# ───────────────────── Cluster→slot mapping (4) ─────────────────────


def test_cluster_to_slot_empty_inputs():
    assert map_clusters_to_slots([], []) == {}
    assert map_clusters_to_slots([], [_ev(0, 1, 0, 0.5)]) == {}
    assert map_clusters_to_slots(
        [DiarizationSegment(0, 1, 0, 0.8)], [],
    ) == {}


def test_cluster_to_slot_clean_majority_maps():
    """Cluster 0 overlaps slot 0 for all 2.0 seconds → maps to 0."""
    diar = [DiarizationSegment(0.0, 2.0, cluster_id=0, confidence=0.85)]
    lips = [_ev(0.0, 2.0, slot=0, conf=0.7)]
    mapping = map_clusters_to_slots(diar, lips)
    assert mapping == {0: 0}


def test_cluster_to_slot_weak_majority_unmapped():
    """Cluster 0 overlaps slot 0 for only 40% → unmapped (-1)."""
    diar = [DiarizationSegment(0.0, 10.0, cluster_id=0, confidence=0.85)]
    # Slot 0 overlaps 4s out of 10s total cluster time = 40% < 60%
    lips = [_ev(0.0, 4.0, slot=0, conf=0.7)]
    mapping = map_clusters_to_slots(diar, lips)
    assert mapping == {0: -1}


def test_cluster_to_slot_multi_cluster_multi_slot():
    """Two distinct clusters each map to their own slot."""
    diar = [
        DiarizationSegment(0.0, 2.0, cluster_id=0, confidence=0.85),
        DiarizationSegment(2.0, 4.0, cluster_id=1, confidence=0.85),
    ]
    lips = [
        _ev(0.0, 2.0, slot=5, conf=0.7),
        _ev(2.0, 4.0, slot=7, conf=0.7),
    ]
    mapping = map_clusters_to_slots(diar, lips)
    assert mapping == {0: 5, 1: 7}


# ───────────────────── Fusion branches (8) ─────────────────────


def test_fusion_empty_lip_events_returns_empty():
    diar = [DiarizationSegment(0, 1, 0, 0.8)]
    result = fuse_lip_and_diarization([], diar)
    assert isinstance(result, FusionResult)
    assert result.events == []
    assert result.agree_count == 0


def test_fusion_empty_diarization_passes_lip_through():
    lips = [_ev(0.0, 1.0, slot=0, conf=0.7)]
    result = fuse_lip_and_diarization(lips, [])
    assert len(result.events) == 1
    assert result.events[0].slot_id == 0
    assert result.events[0].confidence == 0.7
    assert result.agree_count == 0
    assert result.disagree_count == 0


def test_fusion_all_clusters_unmapped_passes_through():
    """When every diar cluster is unmapped (all overlaps < threshold),
    the event list passes through unchanged. This short-circuit
    avoids walking every lip event when there's no useful signal."""
    # Diar cluster spans a long window; lip event covers only 20%.
    diar = [DiarizationSegment(0.0, 10.0, cluster_id=0, confidence=0.85)]
    lips = [_ev(0.0, 2.0, slot=3, conf=0.4)]
    result = fuse_lip_and_diarization(lips, diar)
    # Mapping is {0: -1} so the short-circuit fires and the events
    # pass through unchanged.
    assert result.events == lips
    assert result.agree_count == 0
    assert result.override_count == 0


def test_fusion_agree_boosts_confidence():
    """Lip slot matches diar cluster-mapped slot → confidence += 0.25,
    capped at 0.95."""
    diar = [DiarizationSegment(0.0, 1.0, cluster_id=0, confidence=0.85)]
    lips = [_ev(0.0, 1.0, slot=4, conf=0.5)]
    result = fuse_lip_and_diarization(lips, diar)
    assert result.agree_count == 1
    assert result.events[0].slot_id == 4
    assert result.events[0].confidence == pytest.approx(
        0.5 + AGREE_CONFIDENCE_BOOST,
    )


def test_fusion_agree_confidence_capped():
    """Confidence boost is clamped at AGREE_CONFIDENCE_CAP (0.95)."""
    diar = [DiarizationSegment(0.0, 1.0, cluster_id=0, confidence=0.85)]
    lips = [_ev(0.0, 1.0, slot=4, conf=0.9)]
    result = fuse_lip_and_diarization(lips, diar)
    assert result.agree_count == 1
    assert result.events[0].confidence == pytest.approx(AGREE_CONFIDENCE_CAP)


def test_fusion_override_when_lip_noisy():
    """Lip disagrees with diar AND lip confidence < LIP_NOISE_FLOOR
    AND diar confidence ≥ 0.6 → slot_id flips to diar's vote,
    confidence set to DIARIZATION_OVERRIDE_CONFIDENCE.

    Fixture construction: one long anchor event unambiguously maps
    cluster 1 → slot 9 (covers 100% of the cluster window), and
    a short noisy event on slot 3 sits inside that window and
    should get overridden.
    """
    diar = [
        DiarizationSegment(0.0, 1.0, cluster_id=0, confidence=0.85),
        DiarizationSegment(1.0, 5.0, cluster_id=1, confidence=0.85),
    ]
    lips = [
        _ev(0.0, 1.0, slot=4, conf=0.8),   # anchors cluster 0 → 4
        _ev(1.0, 5.0, slot=9, conf=0.8),   # anchors cluster 1 → 9 (4.0s)
        _ev(2.0, 3.0, slot=3, conf=0.2),   # noisy, disagrees (1.0s)
    ]
    result = fuse_lip_and_diarization(lips, diar)
    # Expect mapping {0: 4, 1: 9} — slot 9's 4.0s overlap dominates
    # slot 3's 1.0s overlap inside cluster 1.
    assert result.cluster_to_slot == {0: 4, 1: 9}
    # The noisy event (slot 3, conf 0.2) should be overridden to slot 9.
    noisy_fused = result.events[2]
    assert noisy_fused.slot_id == 9
    assert noisy_fused.confidence == pytest.approx(
        DIARIZATION_OVERRIDE_CONFIDENCE,
    )
    assert result.override_count >= 1


def test_fusion_disagree_keeps_lip_but_caps_confidence():
    """Lip disagrees with diar AND lip confidence ≥ LIP_NOISE_FLOOR
    → keep lip's slot, clamp confidence to DISAGREE_CONFIDENCE_CAP."""
    diar = [
        DiarizationSegment(0.0, 1.0, cluster_id=0, confidence=0.85),
        DiarizationSegment(1.0, 5.0, cluster_id=1, confidence=0.85),
    ]
    lips = [
        _ev(0.0, 1.0, slot=4, conf=0.9),   # anchors cluster 0 → 4
        _ev(1.0, 5.0, slot=7, conf=0.9),   # anchors cluster 1 → 7
        _ev(2.0, 3.0, slot=4, conf=0.9),   # confident lip disagreer
    ]
    result = fuse_lip_and_diarization(lips, diar)
    # Mapping unambiguous: {0: 4, 1: 7}
    assert result.cluster_to_slot == {0: 4, 1: 7}
    # The disagreer (slot 4, conf 0.9) keeps slot 4 but is capped.
    disagreer = result.events[2]
    assert disagreer.slot_id == 4
    assert disagreer.confidence == pytest.approx(DISAGREE_CONFIDENCE_CAP)
    assert result.disagree_count >= 1


def test_fusion_disagree_with_low_diar_confidence_does_not_override():
    """Even when lip confidence is below the noise floor, a low-
    confidence diarization vote (< 0.6) is not trusted enough to
    override — the event stays on the lip slot (still capped by
    DISAGREE_CONFIDENCE_CAP because slots differ)."""
    diar = [
        DiarizationSegment(0.0, 1.0, cluster_id=0, confidence=0.5),
        DiarizationSegment(1.0, 5.0, cluster_id=1, confidence=0.5),
    ]
    lips = [
        _ev(0.0, 1.0, slot=4, conf=0.9),   # anchors cluster 0 → 4
        _ev(1.0, 5.0, slot=7, conf=0.9),   # anchors cluster 1 → 7
        _ev(2.0, 3.0, slot=4, conf=0.3),   # disagrees with low lip conf
    ]
    result = fuse_lip_and_diarization(lips, diar)
    assert result.cluster_to_slot == {0: 4, 1: 7}
    noisy = result.events[2]
    # diar_conf=0.5 is below the 0.6 override gate → disagree path, NOT
    # override. slot stays as 4.
    assert noisy.slot_id == 4
    assert result.override_count == 0
    assert result.disagree_count >= 1


# ───────────────────── End-to-end JBP crosstalk (1) ─────────────────


def test_fusion_jbp_crosstalk_scenario():
    """Simulate the exact failure mode: 2 speakers, second speaker
    laughs while first speaker talks. Lip-motion incorrectly
    attributes the laughing reaction (slot 1) as the active speaker
    during a window where diarization correctly says slot 0 is
    speaking. Fusion should either override slot 1→0 (when lip
    confidence is low) or at minimum flag the disagreement."""
    # Two clusters: cluster 0 = Joe, cluster 1 = Mal. Joe speaks
    # the whole window; Mal's laugh overlaps Joe's sentence.
    diar = [
        DiarizationSegment(0.0, 3.0, cluster_id=0, confidence=0.85),  # Joe
        DiarizationSegment(3.0, 4.0, cluster_id=1, confidence=0.85),  # Mal
    ]
    lips = [
        _ev(0.0, 1.0, slot=0, conf=0.8),   # Joe speaking — agree
        _ev(1.0, 2.0, slot=1, conf=0.3),   # Mal laugh flagged — OVERRIDE
        _ev(2.0, 3.0, slot=0, conf=0.8),   # Joe — agree
        _ev(3.0, 4.0, slot=1, conf=0.8),   # Mal reply — agree
    ]
    result = fuse_lip_and_diarization(lips, diar)
    # 3 agrees (Joe, Joe, Mal) + 1 override (Mal laugh → Joe)
    assert result.agree_count == 3
    assert result.override_count == 1
    # The laugh frame should now point at Joe (slot 0).
    laugh_frame = result.events[1]
    assert laugh_frame.slot_id == 0
    assert laugh_frame.confidence == pytest.approx(
        DIARIZATION_OVERRIDE_CONFIDENCE,
    )


# ───────────────────── Tuning constants sanity (1) ─────────────────


def test_tuning_constants_are_in_expected_ranges():
    """Guard against accidental tuning-knob drift."""
    assert 0.0 < AGREE_CONFIDENCE_BOOST < 0.5
    assert 0.8 <= AGREE_CONFIDENCE_CAP <= 1.0
    assert 0.3 <= DISAGREE_CONFIDENCE_CAP <= 0.7
    assert 0.2 <= LIP_NOISE_FLOOR <= 0.6
    assert 0.4 <= DIARIZATION_OVERRIDE_CONFIDENCE <= 0.9
    assert 0.4 <= CLUSTER_TO_SLOT_THRESHOLD <= 0.9
    assert 100 <= MIN_VRAM_MB_FOR_CUDA <= 2000
