"""Gap 5b — tests for diarization agreement in the subject confidence
estimator.

Coverage (13 tests):
  - Backward compat: no diarization kwargs leaves legacy 0.20 slice intact
  - ``_has_diarization`` state resolution (3 branches)
  - ``_check_diarization_agreement`` (5 cases: no candidate, no data,
    majority agree, majority disagree, unmapped cluster)
  - ``evaluate_with_breakdown`` with diarization (4 branches: absent,
    present+agree, present+disagree, breakdown has new key only in
    diar mode)

All 13 tests run in the sandbox without numpy / cv2 / mediapipe.
"""

from __future__ import annotations

import pytest

from backend.services.active_speaker import SpeakerEvent
from backend.services.face_registry import FaceRegistry, FaceSlot
from backend.services.speaker_diarization import DiarizationSegment
from backend.services.subject_confidence import SubjectConfidenceEstimator


# ───────────── Fixture helpers ─────────────


class _StubFaceInfo:
    """Minimal ``FaceInfo``-shaped stub placed inside ``FrameFaces``
    stubs. The real estimator only reads a handful of attributes."""

    def __init__(self, nose_x=50.0, nose_y=50.0, width=10.0, height=10.0,
                 identity_id=0, confidence=0.9):
        self.nose_x = nose_x
        self.nose_y = nose_y
        self.x_center = nose_x
        self.y_center = nose_y
        self.width = width
        self.height = height
        self.identity_id = identity_id
        self.confidence = confidence
        self.lip_aperture = 0.0


class _StubFrameFaces:
    def __init__(self, timestamp, faces):
        self.timestamp = timestamp
        self.faces = faces


def _build_two_speaker_scene(active_slot: int = 0):
    """Build a minimal (registry, dense_faces, lip_events, transcript)
    quad for a 2-second 2-slot scene. Slot 0 sits on the left
    (nose_x=30), slot 1 on the right (nose_x=70)."""
    registry = FaceRegistry(
        slots=[
            FaceSlot(
                slot_id=0, x_center=30.0, x_min=28.0, x_max=32.0,
                frame_count=10, avg_width=10.0, avg_height=10.0,
            ),
            FaceSlot(
                slot_id=1, x_center=70.0, x_min=68.0, x_max=72.0,
                frame_count=10, avg_width=10.0, avg_height=10.0,
            ),
        ],
        total_frames=10,
        frames_with_faces=10,
    )
    dense: list = []
    # 5 frames over [0.0, 2.0] with both faces present
    for i in range(5):
        ts = 0.2 + 0.4 * i
        dense.append(_StubFrameFaces(ts, [
            _StubFaceInfo(nose_x=30.0, identity_id=0),
            _StubFaceInfo(nose_x=70.0, identity_id=1),
        ]))
    lip_events = [
        SpeakerEvent(
            start=0.0, end=2.0, slot_id=active_slot, confidence=0.9,
        ),
    ]
    transcript = []  # No transcript = no transcript contribution
    return registry, dense, lip_events, transcript


def _make_estimator(
    *,
    diarization_segments=None,
    cluster_to_slot=None,
    content_type: str = "unknown",
    active_slot: int = 0,
):
    registry, dense, lips, transcript = _build_two_speaker_scene(
        active_slot=active_slot,
    )
    est = SubjectConfidenceEstimator(
        face_registry=registry,
        dense_faces=dense,
        active_speaker_events=lips,
        transcript_segments=transcript,
        speaker_to_slot={},
        source_width=1920,
        source_height=1080,
        diarization_segments=diarization_segments,
        cluster_to_slot=cluster_to_slot,
        content_type=content_type,
    )
    # ── Stub out the legacy numpy-requiring helpers ──
    # This unit suite targets Phase B wiring (diarization as a
    # second vote in the confidence math). The legacy face-in-crop
    # and face-stability checks import numpy inline, which the
    # sandbox does not have. Replace them with deterministic stubs
    # so the tests exercise only the Phase B control flow.
    est._check_face_in_crop_window = lambda *a, **kw: (True, "stub_in_crop")
    est._check_face_stability = lambda *a, **kw: 1.0
    est._check_transcript_coverage = lambda *a, **kw: True
    return est


# ─────────── Backward compatibility ───────────


def test_no_diarization_kwargs_preserves_legacy_init():
    """Constructing the estimator with zero diarization kwargs must
    leave ``_has_diarization`` False, keep the full 0.20 lip budget,
    and the breakdown dict must still contain ``diarization_agree=0.0``
    (the new key is always present, just zero-valued in legacy
    mode) — the segmenter already initializes it to 0.0."""
    est = _make_estimator()
    assert est._has_diarization is False
    assert est._lip_weight == pytest.approx(0.20)
    assert est._diar_weight == pytest.approx(0.0)


# ─────────── _has_diarization state (3 branches) ───────────


def test_has_diarization_true_when_mapping_has_valid_slot():
    est = _make_estimator(
        diarization_segments=[
            DiarizationSegment(0.0, 2.0, cluster_id=0, confidence=0.85),
        ],
        cluster_to_slot={0: 0},
    )
    assert est._has_diarization is True


def test_has_diarization_false_when_all_clusters_unmapped():
    """All cluster ids mapped to -1 → treat as no diarization signal."""
    est = _make_estimator(
        diarization_segments=[
            DiarizationSegment(0.0, 2.0, cluster_id=0, confidence=0.85),
        ],
        cluster_to_slot={0: -1},
    )
    assert est._has_diarization is False
    assert est._lip_weight == pytest.approx(0.20)


def test_has_diarization_false_when_segments_empty():
    est = _make_estimator(
        diarization_segments=[],
        cluster_to_slot={0: 0},
    )
    assert est._has_diarization is False


# ─────────── _check_diarization_agreement (5 cases) ───────────


def test_diar_agreement_none_candidate_returns_false():
    est = _make_estimator(
        diarization_segments=[
            DiarizationSegment(0.0, 2.0, cluster_id=0, confidence=0.85),
        ],
        cluster_to_slot={0: 0},
    )
    assert est._check_diarization_agreement(0.0, 2.0, None) is False


def test_diar_agreement_no_data_returns_false():
    est = _make_estimator()
    assert est._check_diarization_agreement(0.0, 2.0, 0) is False


def test_diar_agreement_majority_agree_returns_true():
    """Cluster 0 → slot 0, diar covers 100% of segment → True."""
    est = _make_estimator(
        diarization_segments=[
            DiarizationSegment(0.0, 2.0, cluster_id=0, confidence=0.85),
        ],
        cluster_to_slot={0: 0},
    )
    assert est._check_diarization_agreement(0.0, 2.0, 0) is True


def test_diar_agreement_majority_disagree_returns_false():
    """Cluster 0 → slot 1, candidate = slot 0 → False."""
    est = _make_estimator(
        diarization_segments=[
            DiarizationSegment(0.0, 2.0, cluster_id=0, confidence=0.85),
        ],
        cluster_to_slot={0: 1},
    )
    assert est._check_diarization_agreement(0.0, 2.0, 0) is False


def test_diar_agreement_unmapped_cluster_overlap_returns_false():
    """Segment overlaps a diar cluster whose mapping is -1 → False
    (the overlap counts toward ``total_time`` but not ``agree_time``)."""
    est = _make_estimator(
        diarization_segments=[
            DiarizationSegment(0.0, 2.0, cluster_id=0, confidence=0.85),
            DiarizationSegment(0.0, 2.0, cluster_id=1, confidence=0.85),
        ],
        cluster_to_slot={0: 0, 1: -1},
    )
    # candidate slot 0: agree_time = 2.0 (cluster 0) out of
    # total_time = 4.0 (clusters 0 + 1). 2.0/4.0 = 0.5 → pass.
    assert est._check_diarization_agreement(0.0, 2.0, 0) is True
    # candidate slot 1: agree_time = 0 (cluster 1 → -1). 0/4 = 0 → fail.
    assert est._check_diarization_agreement(0.0, 2.0, 1) is False


# ─────────── evaluate_with_breakdown with diarization (4 tests) ───────────


def test_breakdown_has_diarization_key_even_in_legacy_mode():
    """The breakdown dict always contains ``diarization_agree``; in
    legacy mode it stays at 0.0 because ``_has_diarization=False``."""
    est = _make_estimator()
    conf, _reason, breakdown = est.evaluate_with_breakdown(
        seg_start=0.0, seg_end=2.0,
        candidate_slot=0, candidate_x=30,
    )
    assert "diarization_agree" in breakdown
    assert breakdown["diarization_agree"] == pytest.approx(0.0)
    # Speaker-agree uses the full 0.20 slice in legacy mode.
    assert breakdown["speaker_agree"] == pytest.approx(0.20)


def test_breakdown_agreement_contributes_diar_weight():
    """Lip AND diar both vote slot 0 → speaker_agree = 0.15,
    diarization_agree = 0.10, sum still ≤ the 0.25 "speaker" budget."""
    est = _make_estimator(
        diarization_segments=[
            DiarizationSegment(0.0, 2.0, cluster_id=0, confidence=0.85),
        ],
        cluster_to_slot={0: 0},
    )
    conf, _reason, breakdown = est.evaluate_with_breakdown(
        seg_start=0.0, seg_end=2.0,
        candidate_slot=0, candidate_x=30,
    )
    assert breakdown["speaker_agree"] == pytest.approx(0.15)
    assert breakdown["diarization_agree"] == pytest.approx(0.10)
    assert (
        breakdown["speaker_agree"] + breakdown["diarization_agree"]
        == pytest.approx(0.25)
    )


def test_breakdown_disagreement_logs_reason():
    """Diar says slot 1 but candidate is slot 0 → disagree reason in
    output string, no diar contribution."""
    est = _make_estimator(
        diarization_segments=[
            DiarizationSegment(0.0, 2.0, cluster_id=0, confidence=0.85),
        ],
        cluster_to_slot={0: 1},
    )
    conf, reason, breakdown = est.evaluate_with_breakdown(
        seg_start=0.0, seg_end=2.0,
        candidate_slot=0, candidate_x=30,
    )
    assert breakdown["diarization_agree"] == pytest.approx(0.0)
    assert "diarization_disagrees" in reason


def test_overall_confidence_never_exceeds_1():
    """Full budget: face_in_crop (0.40) + speaker_agree (0.15) +
    diar_agree (0.10) + stability (0.20 max) + transcript (0.20) =
    1.05 — but confidence must clamp at 1.0."""
    est = _make_estimator(
        diarization_segments=[
            DiarizationSegment(0.0, 2.0, cluster_id=0, confidence=0.85),
        ],
        cluster_to_slot={0: 0},
    )
    conf, _reason, _breakdown = est.evaluate_with_breakdown(
        seg_start=0.0, seg_end=2.0,
        candidate_slot=0, candidate_x=30,
    )
    assert 0.0 <= conf <= 1.0


# ─────────── Gap 5c per-content-type integration (3 tests) ───────────


def test_estimator_uses_panel_weights():
    """Panel content type → lip=0.12, diar=0.13 in breakdown."""
    est = _make_estimator(
        diarization_segments=[
            DiarizationSegment(0.0, 2.0, cluster_id=0, confidence=0.85),
        ],
        cluster_to_slot={0: 0},
        content_type="multi_speaker_panel",
    )
    conf, _reason, breakdown = est.evaluate_with_breakdown(
        seg_start=0.0, seg_end=2.0,
        candidate_slot=0, candidate_x=30,
    )
    assert breakdown["speaker_agree"] == pytest.approx(0.12)
    assert breakdown["diarization_agree"] == pytest.approx(0.13)


def test_estimator_uses_vlog_weights():
    """Vlog → lip=0.18, diar=0.07 (lip-favored)."""
    est = _make_estimator(
        diarization_segments=[
            DiarizationSegment(0.0, 2.0, cluster_id=0, confidence=0.85),
        ],
        cluster_to_slot={0: 0},
        content_type="vlog",
    )
    conf, _reason, breakdown = est.evaluate_with_breakdown(
        seg_start=0.0, seg_end=2.0,
        candidate_slot=0, candidate_x=30,
    )
    assert breakdown["speaker_agree"] == pytest.approx(0.18)
    assert breakdown["diarization_agree"] == pytest.approx(0.07)


def test_estimator_unknown_type_defaults_to_generic_split():
    """Unrecognized content type with diarization present → legacy
    0.15 / 0.10 generic split, not the no-diar (0.20, 0.0) path."""
    est = _make_estimator(
        diarization_segments=[
            DiarizationSegment(0.0, 2.0, cluster_id=0, confidence=0.85),
        ],
        cluster_to_slot={0: 0},
        content_type="totally_made_up",
    )
    conf, _reason, breakdown = est.evaluate_with_breakdown(
        seg_start=0.0, seg_end=2.0,
        candidate_slot=0, candidate_x=30,
    )
    assert breakdown["speaker_agree"] == pytest.approx(0.15)
    assert breakdown["diarization_agree"] == pytest.approx(0.10)
