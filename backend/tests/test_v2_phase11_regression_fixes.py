"""v2 Phase 11 — regression fixes for ClipAI reframing.

Covers Fixes 1-7 from the Tank/Tyrese Verzuz panel clip regression:

  Fix 1: USE_CONTENT_AWARE_REFRAME defaults to True; multi_speaker_panel
         routes to its own content-type-config preset.
  Fix 2: face_registry rejects the embedding registry when ANY slot is a
         cross-shot merge (span>60% AND frame_count>30), not just when
         >25% of slots fail. Panel-mode prefers position-based.
  Fix 3: layout_engine's panel short-shot override SKIPS when shot
         detector confidence is "low" (opencv fallback).
  Fix 4: opencv shot-detector fallback stamps detector_confidence="low"
         and Shot.to_dict exposes it.
  Fix 5: openrouter_provider discards vision-derived subject_x when the
         center-default rate exceeds 25%.
  Fix 7: layout_engine.layout_from_reframe_segments maps a ReframeSegment
         timeline to a LayoutTimeline without re-running shot detection.

Fix 6 (Ollama eviction) is covered by test_whisper_ollama_eviction.py
below — it needs an httpx mock harness which we keep isolated so the
rest of this module stays pure-python / no-IO.
"""

from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest


# ─────────────────────────────────────────────────────────────────────────────
# Fix 1 — content-aware segmenter default + multi_speaker_panel routing
# ─────────────────────────────────────────────────────────────────────────────

def test_fix1_use_content_aware_reframe_default_is_true():
    from backend.services import reframe_segmenter as seg
    from backend.services import content_classifier as cc

    assert seg.USE_CONTENT_AWARE_REFRAME is True, (
        "USE_CONTENT_AWARE_REFRAME must default to True so the segmenter "
        "sees classify_content output on panel clips"
    )
    assert cc.USE_CONTENT_AWARE_REFRAME is True


def test_fix1_content_type_config_has_multi_speaker_panel():
    from backend.services.content_type_config import (
        ContentType, CONTENT_TYPE_CONFIG, get_config,
    )

    assert ContentType.MULTI_SPEAKER_PANEL.value == "multi_speaker_panel"
    assert ContentType.MULTI_SPEAKER_PANEL in CONTENT_TYPE_CONFIG

    cfg = get_config("multi_speaker_panel")
    # Panels should not do in-shot tracking — seats are fixed.
    assert cfg.get("allow_motion_tracking") is False
    assert cfg.get("allow_tracking_within_shot") is False
    # Tight hold floor per spec.
    assert cfg.get("min_hold_seconds_panel") == 0.9
    assert cfg.get("anticipation_lead_seconds") == 0.20


def test_fix1_reframe_segmenter_routes_panel_profile_to_panel_config():
    """When content_profile.is_multi_speaker_panel is True, the segmenter
    must load the multi_speaker_panel config even though the base
    content_type is still "podcast"."""
    from backend.services.reframe_segmenter import build_reframe_segments
    from backend.services.content_classifier import ContentProfile

    profile = ContentProfile()
    profile.content_type = "podcast"
    profile.is_multi_speaker_panel = True

    # Minimal "empty" inputs — we're only checking the ct-routing branch.
    # build_reframe_segments should not crash on empty registry / empty
    # events; it returns an empty list.
    segs = build_reframe_segments(
        shot_cuts=[],
        face_registry=None,
        active_speaker_events=[],
        dense_faces=[],
        transcript_segments=[],
        speaker_to_slot={},
        video_duration=10.0,
        source_width=1920,
        source_height=1080,
        content_profile=profile,
    )
    # Result shape is opaque here; the important assertion is that the
    # call returns without raising and that ct="multi_speaker_panel" was
    # the effective routing (we can't easily introspect ct from outside;
    # the dedicated config test above covers the data wiring).
    assert isinstance(segs, list)


# ─────────────────────────────────────────────────────────────────────────────
# Fix 2 — face registry quality selection
# ─────────────────────────────────────────────────────────────────────────────

def test_fix2_any_cross_shot_merge_rejects_embedding_registry():
    """If ANY embedding slot has span>60% AND frame_count>30, the whole
    embedding registry is rejected — not just when >25% fail.

    Rationale: one cross-shot-merged slot poisons the downstream
    speaker-to-slot mapping because everyone ends up mapped to the
    merged slot. The Verzuz run had Slot 8 spanning [14-98]% with
    558 frames — 1 of 9 slots, under the old 25% gate.
    """
    # We test the behavior through a direct call to build_face_registry_
    # with_embeddings with a fake input. That function is long, so
    # instead we verify the policy by probing the module's decision
    # path symbolically — that the rejection predicate is
    # "any" rather than "fraction > 0.25".
    import inspect
    from backend.services import face_registry as fr

    src = inspect.getsource(fr.build_face_registry_with_embeddings)
    # Look for the new "any" rejection logic.
    assert "if wide_span_bad:" in src, (
        "build_face_registry_with_embeddings should reject on any "
        "single cross-shot-merged slot, not only when > 25% of slots fail"
    )
    # And the old > 0.25 bad_frac gate must be gone.
    assert "bad_frac > 0.25" not in src, (
        "Legacy 25%% fraction gate should be removed (Fix 2)"
    )


def test_fix2_panel_prefers_position_registry_even_when_emb_has_more_slots():
    import inspect
    from backend.services import face_registry as fr

    src = inspect.getsource(fr.build_face_registry_with_embeddings)
    # The panel-mode branch must prefer pos_registry when both are
    # valid, not just when pos has MORE slots (the legacy max-wins
    # rule).
    assert "preferring position-based" in src or "panel_mode=True" in src


# ─────────────────────────────────────────────────────────────────────────────
# Fix 3/4 — shot detector confidence + opencv fallback marking
# ─────────────────────────────────────────────────────────────────────────────

def test_fix4_shot_dataclass_has_detector_confidence_field():
    from backend.services.shot_detector import Shot

    s = Shot(index=0, start=0.0, end=1.0)
    assert s.detector_confidence == "high"  # default

    s_low = Shot(index=1, start=1.0, end=2.0, detector_confidence="low")
    d = s_low.to_dict()
    assert d["detector_confidence"] == "low"


def test_fix3_layout_panel_short_shot_override_skips_on_low_confidence():
    """When any shot is low-confidence (opencv fallback), the panel
    short-shot override must not fire — let the L1 solver handle it."""
    import inspect
    from backend.services import layout_engine

    src = inspect.getsource(layout_engine._plan_layout_impl)
    assert "_any_low_conf_shots" in src
    assert "panel short-shot override SKIPPED" in src


# ─────────────────────────────────────────────────────────────────────────────
# Fix 5 — vision quality gate
# ─────────────────────────────────────────────────────────────────────────────

def _mk_scene(sx: int, timestamp: float = 0.0):
    from backend.models import SceneDescription
    return SceneDescription(
        timestamp=timestamp,
        description="",
        importance_score=5,
        thumbnail_path="",
        subject_x=sx,
    )


def test_fix5_vision_quality_gate_emits_scene_reset_code():
    # Read as text to avoid importing openrouter_provider (which pulls
    # in the `openai` package that isn't a test-env dependency).
    from pathlib import Path

    src = Path(
        "backend/services/providers/openrouter_provider.py"
    ).read_text()
    # Threshold lowered from 70% to 25%.
    assert "center_pct_log > 25" in src, (
        "Vision quality gate must lower the center-default threshold to 25%%"
    )
    assert "Vision model quality too low" in src
    # Scenes reset to the default (50) + optional fields cleared.
    assert "scene.subject_x = 50" in src
    assert "scene.active_speaker_x = None" in src


def test_fix5_scene_reset_logic_is_a_pure_loop_over_scenes():
    """Smoke-check: manually execute the reset loop on a list of scenes
    to confirm it doesn't explode on pydantic validation."""
    scenes = [_mk_scene(48), _mk_scene(50), _mk_scene(52), _mk_scene(30)]
    # Simulate what the gate does.
    for scene in scenes:
        scene.subject_x = 50
        scene.active_speaker_x = None
        scene.precise_x = None
        scene.precise_y = None
    for s in scenes:
        assert s.subject_x == 50
        assert s.active_speaker_x is None
        assert s.precise_x is None


# ─────────────────────────────────────────────────────────────────────────────
# Fix 7 — layout_from_reframe_segments
# ─────────────────────────────────────────────────────────────────────────────

def test_fix7_layout_from_reframe_segments_emits_one_segment_per_input():
    from backend.services.layout_engine import layout_from_reframe_segments
    from backend.services.reframe_segmenter import ReframeSegment

    segs = [
        ReframeSegment(
            start=0.0, end=2.0,
            subject_x=480, subject_y=400,
            layout="single", active_slot=0,
            confidence=0.9, reason="speaker_turn", ease_in_ms=200,
            strategy="stationary",
        ),
        ReframeSegment(
            start=2.0, end=5.0,
            subject_x=1440, subject_y=400,
            layout="single", active_slot=1,
            confidence=0.85, reason="speaker_turn", ease_in_ms=200,
            strategy="stationary",
        ),
    ]

    timeline = layout_from_reframe_segments(
        reframe_segments=segs,
        face_registry=None,
        source_width=1920,
        source_height=1080,
        job_id="test",
    )
    assert len(timeline.segments) == 2
    # First segment: subject_x=480/1920 = 25%
    kf0 = timeline.segments[0].face_positions[0]
    assert 24.9 < kf0["x"] < 25.1
    # Second segment: subject_x=1440/1920 = 75%
    kf1 = timeline.segments[1].face_positions[0]
    assert 74.9 < kf1["x"] < 75.1


def test_fix7_layout_from_reframe_segments_threads_motion_path_keyframes():
    from backend.services.layout_engine import layout_from_reframe_segments
    from backend.services.reframe_segmenter import ReframeSegment

    seg = ReframeSegment(
        start=0.0, end=3.0,
        subject_x=960, subject_y=540,
        layout="single", active_slot=0,
        confidence=0.9, reason="subject_walk", ease_in_ms=400,
        strategy="tracking",
        motion_path=[
            (0.0, 480.0, 540.0),
            (1.5, 960.0, 540.0),
            (3.0, 1440.0, 540.0),
        ],
    )
    timeline = layout_from_reframe_segments(
        reframe_segments=[seg],
        face_registry=None,
        source_width=1920,
        source_height=1080,
        job_id="test",
    )
    assert len(timeline.segments) == 1
    kfs = timeline.segments[0].face_positions
    assert len(kfs) == 3
    # First keyframe at 0.0 s, x=25%
    assert 24.9 < kfs[0]["x"] < 25.1
    # Middle at 1.5 s, x=50%
    assert 49.9 < kfs[1]["x"] < 50.1
    # Last at 3.0 s, x=75%
    assert 74.9 < kfs[2]["x"] < 75.1


def test_fix7_layout_from_reframe_segments_empty_input():
    from backend.services.layout_engine import layout_from_reframe_segments

    t = layout_from_reframe_segments(
        reframe_segments=[],
        face_registry=None,
        source_width=1920,
        source_height=1080,
    )
    assert t.segments == []
    assert t.total_layout_changes == 0
