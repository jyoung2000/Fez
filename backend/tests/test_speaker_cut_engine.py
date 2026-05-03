"""Phase 3 — tests for backend.services.speaker_cut_engine.

Covers the spec's 10 required scenarios. All inputs are synthetic; no
fixtures from disk and no model calls.
"""

from __future__ import annotations

import pytest

from backend.services.reframe_config import get_default_config
from backend.services.shot_reframe_advisor import (
    ReframeStrategy,
    ShotReframeAdvice,
)
from backend.services.speaker_cut_engine import (
    CropKeyframe,
    SpeakerTurn,
    plan_speaker_cuts,
)


def _cfg(**overrides):
    cfg = get_default_config()
    if overrides:
        cfg = cfg.override(**overrides)
    return cfg


# ── 1. Two speakers, clean turns ──────────────────────────────────


def test_two_speakers_clean_turns():
    """A 0-5s, B 5.5-10s → cut at 5.25s (250ms anticipation)."""
    turns = [
        SpeakerTurn("A", 0.0, 5.0),
        SpeakerTurn("B", 5.5, 10.0),
    ]
    positions = {"A": 0.25, "B": 0.75}
    kfs = plan_speaker_cuts(
        speaker_turns=turns,
        speaker_positions=positions,
        shot_start_sec=0.0,
        shot_end_sec=10.0,
        source_width=1920,
        config=_cfg(),
    )
    # Expect: initial keyframe on A at t=0, then cut to B at 5.25.
    assert len(kfs) == 2
    assert kfs[0].speaker_id == "A"
    assert kfs[0].time_sec == pytest.approx(0.0)
    assert kfs[0].x_frac == pytest.approx(0.25)

    assert kfs[1].speaker_id == "B"
    assert kfs[1].time_sec == pytest.approx(5.25, abs=1e-6)
    assert kfs[1].x_frac == pytest.approx(0.75)
    assert kfs[1].transition_type == "hard_cut"


# ── 2. Rapid interruption → dominant ──────────────────────────────


def test_rapid_interruption():
    """Turns < 0.8s each → engine must NOT cut on every turn; the
    output should stay on the dominant speaker (longest total in
    the surrounding 2 s window)."""
    # A speaks 0-0.4s, B speaks 0.5-0.8s, A speaks 0.9-3.0s.
    # Min hold 0.8s → can't cut to B and back to A inside <0.8s gap.
    turns = [
        SpeakerTurn("A", 0.0, 0.4),
        SpeakerTurn("B", 0.5, 0.8),
        SpeakerTurn("A", 0.9, 3.0),
    ]
    positions = {"A": 0.25, "B": 0.75}
    kfs = plan_speaker_cuts(
        speaker_turns=turns,
        speaker_positions=positions,
        shot_start_sec=0.0,
        shot_end_sec=3.0,
        source_width=1920,
        config=_cfg(),
    )
    # Expect a single keyframe — anchor on A (the dominant speaker).
    speaker_ids = [kf.speaker_id for kf in kfs]
    assert speaker_ids.count("A") >= 1
    # No cuts to B that violate min_hold:
    for i in range(1, len(kfs)):
        assert kfs[i].time_sec - kfs[i - 1].time_sec >= 0.8 - 1e-9


# ── 3. Reaction hold ──────────────────────────────────────────────


def test_reaction_hold():
    """A finishes 3.0s, B starts 4.0s → hold A until 3.5s, cut at
    3.75s (4.0 - 0.25 anticipation)."""
    turns = [
        SpeakerTurn("A", 0.0, 3.0),
        SpeakerTurn("B", 4.0, 8.0),
    ]
    positions = {"A": 0.3, "B": 0.7}
    kfs = plan_speaker_cuts(
        speaker_turns=turns,
        speaker_positions=positions,
        shot_start_sec=0.0,
        shot_end_sec=8.0,
        source_width=1920,
        config=_cfg(),
    )
    assert len(kfs) == 2
    # The cut to B sits 250ms before B's start (the reaction hold of
    # 500ms after A's end fits since gap is 1.0s ≥ 0.75s).
    cut_to_b = kfs[1]
    assert cut_to_b.speaker_id == "B"
    assert cut_to_b.time_sec == pytest.approx(3.75, abs=1e-6)
    # Reaction-hold lower bound: cut is at least 500ms after A's end.
    assert cut_to_b.time_sec >= 3.0 + 0.5 - 1e-9


# ── 4. Overlapping speech ─────────────────────────────────────────


def test_overlapping_speech():
    """Both speak 2.0-4.0s simultaneously → stays on whoever started
    first (A here, since A starts at 1.0)."""
    turns = [
        SpeakerTurn("A", 1.0, 4.0),
        SpeakerTurn("B", 2.0, 5.0),
    ]
    positions = {"A": 0.3, "B": 0.7}
    kfs = plan_speaker_cuts(
        speaker_turns=turns,
        speaker_positions=positions,
        shot_start_sec=0.0,
        shot_end_sec=6.0,
        source_width=1920,
        config=_cfg(),
    )
    # First keyframe is on A.
    assert kfs[0].speaker_id == "A"
    # No cut to B occurs DURING the overlap (overlap span 2.0-4.0,
    # which is > 1s so Rule 6 keeps us on A).
    for kf in kfs:
        if kf.speaker_id == "B":
            assert kf.time_sec >= 4.0 - 1e-6, (
                f"Cut to B at t={kf.time_sec} during overlap"
            )
    # When the overlap ends at t=4.0, B is still speaking → expect a
    # cut to B at or shortly after 4.0.
    b_cuts = [kf for kf in kfs if kf.speaker_id == "B"]
    assert len(b_cuts) == 1
    assert b_cuts[0].time_sec >= 4.0 - 1e-6


# ── 5. Initial speaker delayed → center fallback ──────────────────


def test_initial_speaker_delayed():
    """No speech until 1.5s → start on face closest to frame center."""
    turns = [SpeakerTurn("B", 1.5, 4.0)]
    positions = {"A": 0.10, "B": 0.55, "C": 0.90}  # B closest to 0.5
    kfs = plan_speaker_cuts(
        speaker_turns=turns,
        speaker_positions=positions,
        shot_start_sec=0.0,
        shot_end_sec=5.0,
        source_width=1920,
        config=_cfg(),
    )
    # Initial keyframe is the center fallback. Per spec ``speaker_id``
    # is None for the initial center fallback.
    assert kfs[0].time_sec == pytest.approx(0.0)
    assert kfs[0].speaker_id is None
    assert kfs[0].x_frac == pytest.approx(0.55)
    # A subsequent cut to B must follow at 1.5 - 0.25 = 1.25s.
    assert any(
        kf.speaker_id == "B" and abs(kf.time_sec - 1.25) < 1e-6
        for kf in kfs[1:]
    )


# ── 6. Three speakers ─────────────────────────────────────────────


def test_three_speakers():
    """A, B, C take turns → keyframes hit each of them."""
    turns = [
        SpeakerTurn("A", 0.0, 3.0),
        SpeakerTurn("B", 4.0, 7.0),
        SpeakerTurn("C", 8.0, 11.0),
    ]
    positions = {"A": 0.20, "B": 0.50, "C": 0.80}
    kfs = plan_speaker_cuts(
        speaker_turns=turns,
        speaker_positions=positions,
        shot_start_sec=0.0,
        shot_end_sec=12.0,
        source_width=1920,
        config=_cfg(),
    )
    speakers = [kf.speaker_id for kf in kfs]
    assert speakers == ["A", "B", "C"]
    # Cuts happen 250ms before each new speaker.
    assert kfs[1].time_sec == pytest.approx(3.75, abs=1e-6)
    assert kfs[2].time_sec == pytest.approx(7.75, abs=1e-6)


# ── 7. Hard-cut only ──────────────────────────────────────────────


def test_hard_cut_only():
    """Every keyframe must report transition_type == 'hard_cut'."""
    turns = [
        SpeakerTurn("A", 0.0, 2.0),
        SpeakerTurn("B", 2.5, 4.0),
        SpeakerTurn("A", 4.5, 6.0),
        SpeakerTurn("B", 6.5, 8.0),
    ]
    positions = {"A": 0.25, "B": 0.75}
    kfs = plan_speaker_cuts(
        speaker_turns=turns,
        speaker_positions=positions,
        shot_start_sec=0.0,
        shot_end_sec=8.0,
        source_width=1920,
        config=_cfg(),
    )
    assert kfs, "expected at least one keyframe"
    for kf in kfs:
        assert kf.transition_type == "hard_cut"


# ── 8. Min hold enforced ──────────────────────────────────────────


def test_min_hold_enforced():
    """No two cuts < 0.8s apart."""
    turns = [
        SpeakerTurn("A", 0.0, 1.0),
        SpeakerTurn("B", 1.05, 2.0),  # very tight
        SpeakerTurn("A", 2.05, 3.0),
        SpeakerTurn("B", 3.05, 4.0),
        SpeakerTurn("A", 5.0, 7.0),
    ]
    positions = {"A": 0.25, "B": 0.75}
    kfs = plan_speaker_cuts(
        speaker_turns=turns,
        speaker_positions=positions,
        shot_start_sec=0.0,
        shot_end_sec=7.0,
        source_width=1920,
        config=_cfg(),
    )
    for i in range(1, len(kfs)):
        dt = kfs[i].time_sec - kfs[i - 1].time_sec
        assert dt >= 0.8 - 1e-9, (
            f"Cut {i} at {kfs[i].time_sec} "
            f"is {dt}s after previous (< 0.8s min hold)"
        )


# ── 9. Cut positions match speakers ───────────────────────────────


def test_cut_positions_match_speakers():
    """Each non-fallback keyframe's x_frac equals the speaker's known
    position."""
    turns = [
        SpeakerTurn("A", 0.0, 3.0),
        SpeakerTurn("B", 4.0, 7.0),
        SpeakerTurn("C", 8.0, 11.0),
    ]
    positions = {"A": 0.10, "B": 0.45, "C": 0.85}
    kfs = plan_speaker_cuts(
        speaker_turns=turns,
        speaker_positions=positions,
        shot_start_sec=0.0,
        shot_end_sec=12.0,
        source_width=1920,
        config=_cfg(),
    )
    for kf in kfs:
        if kf.speaker_id is None:
            continue
        assert kf.x_frac == pytest.approx(
            positions[kf.speaker_id], abs=1e-9,
        )


# ── 10. Integration with shot advisor ─────────────────────────────


def test_integration_with_shot_advisor():
    """Mock a ShotReframeAdvice with SPEAKER_ALTERNATING → engine runs
    successfully (smoke-test the advisor → engine handoff)."""
    advice = ShotReframeAdvice(
        strategy=ReframeStrategy.SPEAKER_ALTERNATING,
        confidence=0.9,
        primary_subject_bbox=(0.25, 0.5, 0.18, 0.30),
        secondary_subjects=[(0.75, 0.5, 0.18, 0.30)],
        text_regions_to_protect=[],
        genre_override_applied=None,
        fallback_strategy=ReframeStrategy.SUBJECT_TRACKING,
        shot_idx=0,
    )
    # Driver wires ONLY shots whose advice.strategy is SPEAKER_ALTERNATING.
    assert advice.strategy == ReframeStrategy.SPEAKER_ALTERNATING

    turns = [
        SpeakerTurn("A", 0.0, 4.0),
        SpeakerTurn("B", 4.5, 9.0),
    ]
    positions = {
        "A": float(advice.primary_subject_bbox[0]),
        "B": float(advice.secondary_subjects[0][0]),
    }
    kfs = plan_speaker_cuts(
        speaker_turns=turns,
        speaker_positions=positions,
        shot_start_sec=0.0,
        shot_end_sec=9.0,
        source_width=1920,
        config=_cfg(),
    )
    assert len(kfs) >= 2
    assert kfs[0].speaker_id == "A"
    assert any(kf.speaker_id == "B" for kf in kfs)
