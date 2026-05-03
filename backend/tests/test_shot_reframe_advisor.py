"""Phase 1 — tests for backend.services.shot_reframe_advisor.

Covers the spec's 13 required scenarios. All inputs are synthetic
``ShotAnalysis`` instances; no model calls or fixtures from disk.
"""

from __future__ import annotations

import pytest

from backend.services.content_classifier import ClipContentType
from backend.services.shot_reframe_advisor import (
    FaceSample,
    ReframeStrategy,
    SaliencyPeak,
    ShotAnalysis,
    advise_all_shots,
    recommend_strategy,
)


# ── Helpers ─────────────────────────────────────────────────────────


def _face(t, x, y=0.45, w=0.10, h=0.15, *, track=0, speaker=-1, speaking=False):
    return FaceSample(
        timestamp=t, x_center=x, y_center=y, width=w, height=h,
        track_id=track, speaker_id=speaker, is_speaking=speaking,
    )


def _shot(
    *, shot_type="MS", content_type=ClipContentType.GENERIC,
    duration=3.0, faces=None, peaks=None, text=None,
    has_facecam=False, has_screen=False, idx=0,
):
    return ShotAnalysis(
        shot_idx=idx,
        shot_type=shot_type,
        content_type=content_type,
        shot_duration_sec=duration,
        source_width=1920,
        source_height=1080,
        face_samples=list(faces or []),
        saliency_peaks=list(peaks or []),
        text_regions=list(text or []),
        has_facecam=has_facecam,
        has_screen_or_slides=has_screen,
    )


# ── 1. Single centered face → STATIC_CENTER ─────────────────────────


def test_single_centered_face():
    # Spec calls this a podcast / CU shot. ClipContentType.TALKING_HEAD
    # is the canonical podcast/interview key. Use shot_type=CU so the
    # talking_head genre override force-statics it (but base logic
    # already produces STATIC_CENTER).
    faces = [
        _face(t, x=0.45 + 0.005 * i, track=0, speaker=0, speaking=True)
        for i, t in enumerate([0.0, 0.5, 1.0, 1.5, 2.0])
    ]
    shot = _shot(
        shot_type="CU",
        content_type=ClipContentType.TALKING_HEAD,
        duration=2.0,
        faces=faces,
    )
    advice = recommend_strategy(shot)
    assert advice.strategy is ReframeStrategy.STATIC_CENTER
    assert advice.primary_subject_bbox is not None


# ── 2. Single moving face → SUBJECT_TRACKING ────────────────────────


def test_single_moving_face():
    # Face moves x = 0.20 → 0.70 across 60 frames at 30 fps (2 s).
    n = 60
    faces = [
        _face(
            t=i / 30.0,
            x=0.20 + (0.70 - 0.20) * (i / (n - 1)),
            track=0,
        )
        for i in range(n)
    ]
    shot = _shot(
        shot_type="MS",
        content_type=ClipContentType.GENERIC,
        duration=2.0,
        faces=faces,
    )
    advice = recommend_strategy(shot)
    assert advice.strategy is ReframeStrategy.SUBJECT_TRACKING


# ── 3. Two speakers alternating → SPEAKER_ALTERNATING ───────────────


def test_two_speakers_alternating():
    # 6-second shot. Speaker A talks 0-2s, B talks 2-4s, A talks 4-6s.
    faces: list = []
    for t in [0.0, 0.5, 1.0, 1.5]:
        faces.append(_face(t, x=0.30, track=0, speaker=0, speaking=True))
        faces.append(_face(t, x=0.70, track=1, speaker=1, speaking=False))
    for t in [2.0, 2.5, 3.0, 3.5]:
        faces.append(_face(t, x=0.30, track=0, speaker=0, speaking=False))
        faces.append(_face(t, x=0.70, track=1, speaker=1, speaking=True))
    for t in [4.0, 4.5, 5.0, 5.5]:
        faces.append(_face(t, x=0.30, track=0, speaker=0, speaking=True))
        faces.append(_face(t, x=0.70, track=1, speaker=1, speaking=False))
    shot = _shot(
        shot_type="MS",
        content_type=ClipContentType.MULTI_SPEAKER_PANEL,
        duration=6.0,
        faces=faces,
    )
    advice = recommend_strategy(shot)
    assert advice.strategy is ReframeStrategy.SPEAKER_ALTERNATING
    # The second speaker should land in secondary_subjects for layout.
    assert len(advice.secondary_subjects) >= 1


# ── 4. Two speakers overlapping → SUBJECT_TRACKING ──────────────────


def test_two_speakers_overlapping():
    # Both speak simultaneously at every timestamp; speaker 0 dominates
    # by virtue of having more total speaking samples (we give A more
    # speaking flags than B).
    faces: list = []
    for t in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]:
        faces.append(_face(t, x=0.30, track=0, speaker=0, speaking=True))
        faces.append(_face(t, x=0.70, track=1, speaker=1,
                           speaking=(t in (0.5, 1.5))))
    shot = _shot(
        shot_type="MS",
        content_type=ClipContentType.MULTI_SPEAKER_PANEL,
        duration=3.0,
        faces=faces,
    )
    advice = recommend_strategy(shot)
    assert advice.strategy is ReframeStrategy.SUBJECT_TRACKING


# ── 5. Extreme wide long → CONTEXTUAL_PAN ───────────────────────────


def test_extreme_wide_long():
    peaks = [SaliencyPeak(t, 0.5, 0.5, 1.0) for t in (0.0, 1.0, 2.0, 3.0, 4.0)]
    shot = _shot(
        shot_type="EWS",
        content_type=ClipContentType.GENERIC,
        duration=5.0,
        peaks=peaks,
    )
    advice = recommend_strategy(shot)
    assert advice.strategy is ReframeStrategy.CONTEXTUAL_PAN


# ── 6. Extreme wide short → BLUR_FILL_PRESERVE ──────────────────────


def test_extreme_wide_short():
    shot = _shot(
        shot_type="EWS",
        content_type=ClipContentType.GENERIC,
        duration=1.5,
    )
    advice = recommend_strategy(shot)
    assert advice.strategy is ReframeStrategy.BLUR_FILL_PRESERVE


# ── 7. Gameplay with facecam → MULTI_REGION ─────────────────────────


def test_gameplay_with_facecam():
    shot = _shot(
        shot_type="WS",
        content_type=ClipContentType.STREAM,
        duration=8.0,
        has_facecam=True,
    )
    advice = recommend_strategy(shot)
    assert advice.strategy is ReframeStrategy.MULTI_REGION


# ── 8. Anime dialogue → STATIC_CENTER (genre override) ──────────────


def test_anime_dialogue_static():
    # Single face on the right side (would normally be SUBJECT_TRACKING),
    # but ANIMATION + MCU triggers force_static_for_shot_types.
    faces = [_face(t, x=0.65, track=0) for t in (0.0, 0.5, 1.0, 1.5)]
    shot = _shot(
        shot_type="MCU",
        content_type=ClipContentType.ANIMATION,
        duration=2.0,
        faces=faces,
    )
    advice = recommend_strategy(shot)
    assert advice.strategy is ReframeStrategy.STATIC_CENTER
    assert advice.genre_override_applied is not None
    assert "force_static" in advice.genre_override_applied


# ── 9. Sports wide not blurred → SUBJECT_TRACKING (genre remap) ─────


def test_sports_wide_not_blurred():
    # WS isn't EWS, so base should yield SUBJECT_TRACKING already, but
    # the test guards the SPORTS contract: BLUR_FILL_PRESERVE must be
    # remapped if it ever occurs. Use a short EWS to *force*
    # BLUR_FILL_PRESERVE in base, then assert the SPORTS override
    # remaps it to SUBJECT_TRACKING. Per the test name we also check
    # the WS path doesn't return blur_fill.
    shot_ws = _shot(
        shot_type="WS",
        content_type=ClipContentType.SPORTS,
        duration=4.0,
        peaks=[SaliencyPeak(0.5, 0.5, 0.5, 1.0)],
    )
    advice_ws = recommend_strategy(shot_ws)
    assert advice_ws.strategy is not ReframeStrategy.BLUR_FILL_PRESERVE

    # And the EWS-short BLUR_FILL_PRESERVE remap path:
    shot_ews = _shot(
        shot_type="EWS",
        content_type=ClipContentType.SPORTS,
        duration=1.0,
    )
    advice_ews = recommend_strategy(shot_ews)
    assert advice_ews.strategy is ReframeStrategy.SUBJECT_TRACKING
    assert advice_ews.genre_override_applied is not None


# ── 10. Music video, two performers → SUBJECT_TRACKING ──────────────


def test_music_video_no_speaker_alt():
    # Two faces, both "speak" alternately — but MUSIC_VIDEO blocks
    # SPEAKER_ALTERNATING and remaps to SUBJECT_TRACKING.
    faces: list = []
    for t in [0.0, 0.5, 1.0, 1.5]:
        faces.append(_face(t, x=0.30, track=0, speaker=0, speaking=True))
        faces.append(_face(t, x=0.70, track=1, speaker=1, speaking=False))
    for t in [2.0, 2.5, 3.0, 3.5]:
        faces.append(_face(t, x=0.30, track=0, speaker=0, speaking=False))
        faces.append(_face(t, x=0.70, track=1, speaker=1, speaking=True))
    shot = _shot(
        shot_type="MS",
        content_type=ClipContentType.MUSIC_VIDEO,
        duration=4.0,
        faces=faces,
    )
    advice = recommend_strategy(shot)
    assert advice.strategy is ReframeStrategy.SUBJECT_TRACKING
    assert advice.genre_override_applied is not None
    assert "block" in advice.genre_override_applied


# ── 11. No faces concentrated saliency → STATIC_CENTER ──────────────


def test_no_faces_concentrated_saliency():
    # Std of x_center across peaks is ~0.05.
    peaks = [
        SaliencyPeak(t, 0.45, 0.50, 1.0) for t in (0.0, 1.0)
    ] + [
        SaliencyPeak(t, 0.55, 0.50, 1.0) for t in (2.0, 3.0)
    ]
    shot = _shot(
        shot_type="MS",
        content_type=ClipContentType.GENERIC,
        duration=4.0,
        peaks=peaks,
    )
    advice = recommend_strategy(shot)
    assert advice.strategy is ReframeStrategy.STATIC_CENTER


# ── 12. No faces spread saliency → CONTEXTUAL_PAN ───────────────────


def test_no_faces_spread_saliency():
    # Std ~0.30 across the shot.
    peaks = [
        SaliencyPeak(0.0, 0.10, 0.50, 1.0),
        SaliencyPeak(1.0, 0.30, 0.50, 0.8),
        SaliencyPeak(2.0, 0.70, 0.50, 0.9),
        SaliencyPeak(3.0, 0.90, 0.50, 1.0),
    ]
    shot = _shot(
        shot_type="MS",
        content_type=ClipContentType.GENERIC,
        duration=4.0,
        peaks=peaks,
    )
    advice = recommend_strategy(shot)
    assert advice.strategy is ReframeStrategy.CONTEXTUAL_PAN


# ── 13. Fallback always set ─────────────────────────────────────────


def test_fallback_always_set():
    """Every advice must declare a non-None fallback distinct from the
    primary strategy. Run advise_all_shots over a heterogeneous batch
    and verify the contract for every result."""
    # Build a tiny mock shot list with start/end attrs.
    class _Shot:
        def __init__(self, idx, start, end, st="MS"):
            self.shot_idx = idx
            self.start = start
            self.end = end
            self.shot_type = st

    shots = [
        _Shot(0, 0.0, 2.0, "CU"),
        _Shot(1, 2.0, 7.0, "EWS"),
        _Shot(2, 7.0, 8.5, "EWS"),
        _Shot(3, 8.5, 12.0, "MS"),
    ]

    class _Profile:
        content_type = ClipContentType.TALKING_HEAD
        has_facecam = False
        hud_regions = []

    advices = advise_all_shots(
        shots=shots,
        content_profile=_Profile(),
        face_tracks=[],
        speaker_events=[],
        saliency_data=[],
        text_regions=[],
        source_w=1920,
        source_h=1080,
    )
    assert len(advices) == 4
    for a in advices:
        assert a.fallback_strategy is not None
        assert isinstance(a.fallback_strategy, ReframeStrategy)
        assert a.fallback_strategy is not a.strategy
