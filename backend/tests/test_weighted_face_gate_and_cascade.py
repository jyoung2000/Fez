"""Accuracy fix tests — weighted face-in-crop gate + salient fallback
cascade + wide_master confidence-floor raise.

Targets the bug where a 3-seat panel had 12/62 faces inside the crop
window, failed the old 30% raw-count gate, and fell through to
hardcoded_center (x=960) producing empty-couch wide_master crops.
"""

from dataclasses import dataclass, field
from typing import Optional

from backend.services.face_registry import FaceRegistry, FaceSlot
from backend.services.subject_confidence import (
    SubjectConfidenceEstimator,
    _SalientFallbackResolver,
    resolve_fallback_center,
    CONFIDENCE_HIGH,
    CONFIDENCE_WIDE_MASTER_FLOOR,
    FALLBACK_CASCADE_BY_CONTENT,
    get_fallback_strategy,
)


# ── Test fixtures ──


@dataclass
class _Face:
    nose_x: float = 50.0
    nose_y: float = 50.0
    width: float = 10.0
    height: float = 12.0
    identity_id: int = -1
    pose_confidence: float = 1.0
    lip_aperture: float = 0.0

    # Expose alternate attrs some code paths look for.
    @property
    def x(self):
        return self.nose_x

    @property
    def y(self):
        return self.nose_y


@dataclass
class _FF:
    timestamp: float
    faces: list = field(default_factory=list)


@dataclass
class _SpeakerEvent:
    start: float
    end: float
    slot_id: int
    on_screen: bool = True


@dataclass
class _Shot:
    index: int
    start: float
    end: float


@dataclass
class _SR:
    """SaliencyRegion-compatible."""
    timestamp: float
    x: float
    y: float
    w: float
    h: float
    saliency_score: float = 0.7
    motion_score: float = 0.5
    spatial_score: float = 0.7


def _panel_registry():
    return FaceRegistry(
        slots=[
            FaceSlot(slot_id=0, x_center=23.0, x_min=20.0, x_max=26.0,
                     frame_count=100, avg_width=10.0, avg_height=12.0),
            FaceSlot(slot_id=1, x_center=56.0, x_min=53.0, x_max=59.0,
                     frame_count=100, avg_width=10.0, avg_height=12.0),
            FaceSlot(slot_id=2, x_center=80.0, x_min=77.0, x_max=83.0,
                     frame_count=100, avg_width=10.0, avg_height=12.0),
        ],
        total_frames=300,
        frames_with_faces=300,
    )


def _panel_dense_faces_with_crowd():
    """3 tracked speakers + 4 untracked crowd faces per frame.

    Simulates the Verzuz pathology: only 3/7 faces per frame belong
    to tracked identities. Raw-count gate fails (3/7 ≈ 43%), but
    weighted gate (tracked=1.0, crowd=0.2) gives ~80% weighted in-crop
    when the crop is centered on any of the three tracked slots.
    """
    out = []
    for i in range(60):
        ts = i * 0.5
        faces = [
            _Face(nose_x=23.0, identity_id=0),
            _Face(nose_x=56.0, identity_id=1),
            _Face(nose_x=80.0, identity_id=2),
            # 4 untracked crowd faces scattered across the frame.
            _Face(nose_x=5.0, identity_id=-1),
            _Face(nose_x=95.0, identity_id=-1),
            _Face(nose_x=15.0, identity_id=-1),
            _Face(nose_x=90.0, identity_id=-1),
        ]
        out.append(_FF(timestamp=ts, faces=faces))
    return out


# ── Fix 1: weighted face-in-crop gate ──


def test_weighted_gate_passes_when_tracked_speaker_in_crop():
    # Candidate crop centered on slot 1 (x=56). Tracked face at 56 is
    # inside the crop; tracked faces at 23 and 80 are OUTSIDE the
    # crop (crop_width_pct ≈ 31.6 → [40.2, 71.8]). Crowd faces at
    # 5, 95, 15, 90 are also outside. Raw count: 1/7 faces in crop
    # (~14%) — would have failed the old 30% gate. Weighted count:
    # tracked-in=1.0 * area, tracked-out=2*1.0*area, crowd-out=4*0.2*area.
    # Ratio ≈ 1.0/(1.0+2.0+0.8) ≈ 0.26 — still fails weighted 30%.
    # BUT the active-speaker short-circuit should fire because slot 1
    # is the active speaker and their face IS in the crop.
    estimator = SubjectConfidenceEstimator(
        face_registry=_panel_registry(),
        dense_faces=_panel_dense_faces_with_crowd(),
        active_speaker_events=[_SpeakerEvent(0.0, 30.0, slot_id=1)],
        transcript_segments=[],
        speaker_to_slot={},
        source_width=1920,
        source_height=1080,
    )
    in_crop, detail = estimator._check_face_in_crop_window(
        seg_start=0.0, seg_end=10.0,
        candidate_x=56,
        target_aspect_ratio=9 / 16,
        candidate_slot=1,
    )
    assert in_crop is True, f"gate failed: {detail}"
    assert "short-circuit" in detail


def test_weighted_gate_fails_when_no_tracked_face_in_crop():
    # Construct a pathological clip: no tracked slots at all, only
    # crowd (untracked) faces. Candidate crop contains only a single
    # crowd face out of many. Weighted ratio ≈ 1/N, well below 0.3.
    # No active-speaker short-circuit (candidate_slot=None). Gate
    # must fail.
    no_slot_registry = FaceRegistry(
        slots=[],  # empty — no tracked identities at all
        total_frames=60,
        frames_with_faces=60,
    )
    dense = [
        _FF(timestamp=0.0, faces=[
            _Face(nose_x=50.0, identity_id=-1),
            _Face(nose_x=5.0, identity_id=-1),
            _Face(nose_x=10.0, identity_id=-1),
            _Face(nose_x=15.0, identity_id=-1),
            _Face(nose_x=90.0, identity_id=-1),
            _Face(nose_x=95.0, identity_id=-1),
        ])
    ]
    estimator = SubjectConfidenceEstimator(
        face_registry=no_slot_registry,
        dense_faces=dense,
        active_speaker_events=[],
        transcript_segments=[],
        speaker_to_slot={},
        source_width=1920,
        source_height=1080,
    )
    # Candidate at x=50: crop ≈ [34.2, 65.8]. Only one face at 50.
    # Weighted ratio ≈ 1/6 (≈0.17) — below 0.30 threshold.
    in_crop, detail = estimator._check_face_in_crop_window(
        seg_start=0.0, seg_end=1.0,
        candidate_x=50,
        target_aspect_ratio=9 / 16,
        candidate_slot=None,
    )
    assert in_crop is False, f"expected gate fail, got pass: {detail}"
    assert "weighted=" in detail
    assert "<" in detail


def test_weighted_gate_counts_tracked_face_more_than_crowd():
    # Sanity check that the weighted gate values tracked > untracked:
    # a single tracked face in an otherwise empty crop should pass,
    # even against a much larger crowd outside.
    estimator = SubjectConfidenceEstimator(
        face_registry=_panel_registry(),
        dense_faces=[
            _FF(timestamp=0.0, faces=[
                _Face(nose_x=50.0, identity_id=0, width=20.0, height=25.0),
                # Untracked crowd — 10 tiny faces at x=5 (outside crop).
                *[_Face(nose_x=5.0, identity_id=-1, width=5.0, height=5.0)
                  for _ in range(10)],
            ]),
        ],
        active_speaker_events=[],
        transcript_segments=[],
        speaker_to_slot={},
        source_width=1920,
        source_height=1080,
    )
    in_crop, detail = estimator._check_face_in_crop_window(
        seg_start=0.0, seg_end=1.0,
        candidate_x=50,
        target_aspect_ratio=9 / 16,
        candidate_slot=0,
    )
    assert in_crop is True, f"weighted gate should pass: {detail}"


# ── Fix 2 + 3: salient fallback cascade ──


def test_cascade_prev_crop_continuity_wins_when_recent():
    class _LastSeg:
        end = 2.0
        subject_x = 60
        subject_y = 50
        confidence = 0.75

    estimator = SubjectConfidenceEstimator(
        face_registry=_panel_registry(),
        dense_faces=[],
        active_speaker_events=[],
        transcript_segments=[],
        speaker_to_slot={},
        source_width=1920,
        source_height=1080,
    )
    x, y, src = resolve_fallback_center(
        estimator, "multi_speaker_panel",
        seg_start=3.0, seg_end=5.0,
        last_seg=_LastSeg(),
        candidate_x=None,
    )
    assert src == "prev_crop_continuity"
    assert x == 60


def test_cascade_saliency_peak_when_no_prev():
    estimator = SubjectConfidenceEstimator(
        face_registry=_panel_registry(),
        dense_faces=[],
        active_speaker_events=[],
        transcript_segments=[],
        speaker_to_slot={},
        source_width=1920,
        source_height=1080,
        frame_saliency=[
            _SR(timestamp=1.0, x=72, y=45, w=15, h=20, saliency_score=0.7),
        ],
    )
    x, y, src = resolve_fallback_center(
        estimator, "anime",
        seg_start=0.0, seg_end=2.0,
        last_seg=None,
        candidate_x=None,
    )
    assert src == "saliency_peak"
    assert x == 72
    assert y == 45


def test_cascade_face_centroid_fallback():
    # No prev, no saliency, no active speaker — centroid of the
    # detected faces (including untracked) should win.
    estimator = SubjectConfidenceEstimator(
        face_registry=_panel_registry(),
        dense_faces=[
            _FF(timestamp=0.5, faces=[
                _Face(nose_x=40.0, identity_id=0, width=12, height=15),
                _Face(nose_x=60.0, identity_id=1, width=12, height=15),
            ]),
        ],
        active_speaker_events=[],
        transcript_segments=[],
        speaker_to_slot={},
        source_width=1920,
        source_height=1080,
    )
    x, y, src = resolve_fallback_center(
        estimator, "vlog",
        seg_start=0.0, seg_end=1.0,
        last_seg=None,
        candidate_x=None,
    )
    assert src == "face_centroid"
    # Mean of 40 and 60 weighted equally → 50
    assert 45 <= x <= 55


def test_cascade_never_returns_hardcoded_center_when_data_exists():
    estimator = SubjectConfidenceEstimator(
        face_registry=_panel_registry(),
        dense_faces=_panel_dense_faces_with_crowd(),
        active_speaker_events=[_SpeakerEvent(0.0, 30.0, slot_id=1)],
        transcript_segments=[],
        speaker_to_slot={},
        source_width=1920,
        source_height=1080,
    )
    x, y, src = resolve_fallback_center(
        estimator, "multi_speaker_panel",
        seg_start=5.0, seg_end=10.0,
        last_seg=None,
        candidate_x=None,
    )
    # active_speaker_slot should win for panel content.
    assert src == "active_speaker_slot"
    assert 50 <= x <= 62   # slot 1 center ≈ 56


def test_cascade_only_hardcoded_center_as_absolute_last_resort():
    estimator = SubjectConfidenceEstimator(
        face_registry=None,
        dense_faces=[],
        active_speaker_events=[],
        transcript_segments=[],
        speaker_to_slot={},
        source_width=1920,
        source_height=1080,
    )
    x, y, src = resolve_fallback_center(
        estimator, "unknown",
        seg_start=0.0, seg_end=1.0,
        last_seg=None,
        candidate_x=None,
    )
    assert src == "hardcoded_center"
    assert x == 50
    assert y == 50


# ── Fix 5: wide_master confidence floor raised ──


def test_confidence_floor_raised_to_15():
    # Confidence 0.20 used to produce wide_master. Now it should land
    # in the salient-fallback branch.
    assert CONFIDENCE_WIDE_MASTER_FLOOR == 0.15
    fb = get_fallback_strategy(
        confidence=0.20, content_type="multi_speaker_panel",
        last_confident_x=60, last_confident_slot=1,
        candidate_x=56, candidate_slot=1,
    )
    assert fb is not None
    strategy, layout, x, slot, reason = fb
    # Should fall through to salient-fallback stationary, not wide_master.
    assert strategy == "stationary"
    assert reason == "confidence_low_salient_fallback"


def test_confidence_very_low_also_prefers_salient_fallback():
    # Even at 0.10 (below the new floor), if a cascade center is
    # provided, we use it rather than letterboxing.
    fb = get_fallback_strategy(
        confidence=0.10, content_type="unknown",
        last_confident_x=None, last_confident_slot=None,
        candidate_x=56, candidate_slot=None,
    )
    assert fb is not None
    strategy, layout, x, slot, reason = fb
    assert strategy == "stationary"
    assert reason == "confidence_very_low_salient_fallback"


def test_confidence_very_low_falls_to_wide_master_when_no_cascade():
    fb = get_fallback_strategy(
        confidence=0.05, content_type="unknown",
        last_confident_x=None, last_confident_slot=None,
        candidate_x=None, candidate_slot=None,
    )
    assert fb is not None
    strategy, layout, _x, _slot, reason = fb
    assert strategy == "wide_master"
    assert "very_low" in reason


# ── per_frame_in_crop_pass_rate safeguard ──


def test_per_frame_pass_rate_reports_frame_coverage():
    # 5 frames, tracked face at x=60 in 4 of them and x=10 in 1 frame.
    # Candidate x=60 → 4/5 frames pass = 0.80.
    dense = []
    for i, face_x in enumerate([60, 60, 10, 60, 60]):
        dense.append(_FF(
            timestamp=i * 0.5,
            faces=[_Face(nose_x=face_x, identity_id=0)],
        ))
    estimator = SubjectConfidenceEstimator(
        face_registry=_panel_registry(),
        dense_faces=dense,
        active_speaker_events=[],
        transcript_segments=[],
        speaker_to_slot={},
        source_width=1920,
        source_height=1080,
    )
    rate = estimator.per_frame_in_crop_pass_rate(
        seg_start=0.0, seg_end=3.0,
        candidate_x=60,
        target_aspect_ratio=9 / 16,
    )
    assert 0.75 <= rate <= 0.85


def test_per_frame_pass_rate_handles_empty_window():
    estimator = SubjectConfidenceEstimator(
        face_registry=_panel_registry(),
        dense_faces=[],
        active_speaker_events=[],
        transcript_segments=[],
        speaker_to_slot={},
        source_width=1920,
        source_height=1080,
    )
    rate = estimator.per_frame_in_crop_pass_rate(
        seg_start=0.0, seg_end=1.0,
        candidate_x=50,
        target_aspect_ratio=9 / 16,
    )
    assert rate == 0.0
