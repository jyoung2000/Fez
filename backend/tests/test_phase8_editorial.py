"""Phase 8 — Editorial "camera language" prior tests.

Six test groups:

1. **Content-type gating** — ``applies_to_profile`` ON for
   narrative / podcast / vlog / cinematic_dialogue / talking_head /
   multi_speaker_panel / animation_dialogue; OFF for gaming /
   music_video / animation_action.

2. **J-cut detection + application** — boundary shifts to
   ``audio_start + lead_sec`` per the v2 spec.

3. **L-cut variant** — same shift mechanic, distinct kind label.

4. **Listener-hold detection** — sentence-end + other-speaker-
   silent → hold decision.

5. **Reaction-beat detection** — extreme audio spike + multi-
   face frame → slot swap.

6. **Apply decisions + Stage 9b AST guards** — verify the
   reframe segmenter wiring is structurally present.

Per the spec exit criterion: "synthetic dialogue fixture with
word-level timestamps shows crop arriving on speaker B 200 ms
*after* B's first word audible (J-cut). Listener-holds persist
400-800 ms on reaction beats. 6+ new tests." This file ships
~30 tests across the six groups.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pytest

from backend.services.editorial_prior import (
    DEFAULT_JL_CUT_LEAD_SEC,
    EDITORIAL_PRIOR_CONTENT_TYPES,
    LISTENER_HOLD_MAX_SEC,
    LISTENER_HOLD_MIN_SEC,
    REACTION_BEAT_DURATION_SEC,
    USE_EDITORIAL_PRIOR,
    EditorialApplyReport,
    EditorialDecision,
    apply_decisions,
    apply_editorial_prior,
    applies_to_profile,
    detect_j_cuts,
    detect_l_cuts,
    detect_listener_holds,
    detect_reaction_beats,
)


# ──────────────────── Stub dataclasses ────────────────────


@dataclass
class _Word:
    start: float
    end: float
    word: str = ""


@dataclass
class _TS:
    """TranscriptSegment-shaped stub."""
    start: float
    end: float
    text: str
    speaker: str
    words: list = field(default_factory=list)


@dataclass
class _Seg:
    """ReframeSegment-shaped stub."""
    start: float
    end: float
    active_slot: Optional[int]
    reason: str = ""


@dataclass
class _Slot:
    slot_id: int


@dataclass
class _Profile:
    content_type: str = "unknown"
    is_multi_speaker_panel: bool = False
    is_animated: bool = False
    anime_subtype: str = ""


# ──────────────────── Content-type gating ────────────────────


class TestAppliesToProfile:
    @pytest.mark.parametrize("ct", [
        "narrative", "podcast", "vlog",
        "cinematic_dialogue", "talking_head",
        "multi_speaker_panel", "animation_dialogue",
    ])
    def test_qualifying_types_pass(self, ct):
        assert applies_to_profile(_Profile(content_type=ct)) is True

    @pytest.mark.parametrize("ct", [
        "gameplay", "gameplay_moba", "gameplay_tps",
        "gameplay_racing", "stream", "music_video",
        "sports", "animation", "generic",
    ])
    def test_non_qualifying_types_excluded(self, ct):
        assert applies_to_profile(_Profile(content_type=ct)) is False

    def test_anime_action_explicit_exclusion(self):
        # Anime action content opts out even though it might
        # otherwise pass via animation_dialogue.
        p = _Profile(
            content_type="animation_dialogue",
            is_animated=True,
            anime_subtype="action",
        )
        assert applies_to_profile(p) is False

    def test_anime_dialogue_passes(self):
        p = _Profile(
            content_type="animation_dialogue",
            is_animated=True,
            anime_subtype="dialogue",
        )
        assert applies_to_profile(p) is True

    def test_none_excluded(self):
        assert applies_to_profile(None) is False

    def test_profile_with_no_content_type_excluded(self):
        @dataclass
        class _NoCT:
            content_type: Optional[str] = None
        assert applies_to_profile(_NoCT()) is False


# ──────────────────── J-cut detection ────────────────────


class TestDetectJCuts:
    def _ts_pair(self, audio_start_b: float = 9.9):
        return [
            _TS(start=8.0, end=9.95, text="Hello world.",
                speaker="Speaker 1",
                words=[
                    _Word(8.0, 8.5, "Hello"),
                    _Word(8.5, 9.95, "world."),
                ]),
            _TS(start=audio_start_b, end=12.0,
                text="Goodbye then.",
                speaker="Speaker 2",
                words=[
                    _Word(audio_start_b, audio_start_b + 0.6, "Goodbye"),
                    _Word(audio_start_b + 0.6, 12.0, "then."),
                ]),
        ]

    def _segs(self, switch_t: float = 10.0):
        return [
            _Seg(start=0.0, end=switch_t, active_slot=0),
            _Seg(start=switch_t, end=15.0, active_slot=1),
        ]

    def test_j_cut_detected_audio_leads_video(self):
        # Audio of speaker B starts at 9.9; video cut at 10.0.
        # New boundary should be 9.9 + 0.20 = 10.10 → delta +0.10
        decisions = detect_j_cuts(
            self._segs(10.0), self._ts_pair(9.9),
            speaker_to_slot={"Speaker 1": 0, "Speaker 2": 1},
        )
        assert len(decisions) == 1
        assert decisions[0].kind == "j_cut"
        assert decisions[0].delta_sec == pytest.approx(0.10, abs=1e-6)
        assert decisions[0].target_segment_idx == 1

    def test_j_cut_when_audio_lags_video(self):
        # Audio of B starts at 10.5 (after video cut at 10.0).
        # Spec: shift to audio_start + lead = 10.70 → delta +0.70
        decisions = detect_j_cuts(
            self._segs(10.0), self._ts_pair(10.5),
            speaker_to_slot={"Speaker 1": 0, "Speaker 2": 1},
        )
        assert len(decisions) == 1
        assert decisions[0].delta_sec == pytest.approx(0.70, abs=1e-6)

    def test_no_j_cut_when_no_speaker_change(self):
        # Both segments same active_slot → not a switch
        segs = [
            _Seg(start=0.0, end=10.0, active_slot=0),
            _Seg(start=10.0, end=15.0, active_slot=0),
        ]
        decisions = detect_j_cuts(segs, self._ts_pair(9.9))
        assert decisions == []

    def test_no_j_cut_when_only_one_segment(self):
        decisions = detect_j_cuts([_Seg(0, 10, 0)], self._ts_pair())
        assert decisions == []

    def test_no_j_cut_when_active_slot_none(self):
        segs = [
            _Seg(start=0.0, end=10.0, active_slot=0),
            _Seg(start=10.0, end=15.0, active_slot=None),
        ]
        assert detect_j_cuts(segs, self._ts_pair()) == []

    def test_no_j_cut_when_audio_outside_window(self):
        # Audio of B at t=15 — well past the boundary at 10.0
        ts = self._ts_pair(15.0)
        decisions = detect_j_cuts(
            self._segs(10.0), ts,
            speaker_to_slot={"Speaker 1": 0, "Speaker 2": 1},
        )
        # Outside the 1 s default window → no decision
        assert decisions == []

    def test_label_to_slot_fallback_without_explicit_mapping(self):
        # No speaker_to_slot — fall back to "Speaker N" → slot N-1
        decisions = detect_j_cuts(
            self._segs(10.0), self._ts_pair(9.9),
            speaker_to_slot=None,
        )
        assert len(decisions) == 1

    def test_no_shift_when_already_aligned(self):
        # Audio starts at 9.80, lead 0.20 → new boundary 10.00
        # which equals the existing seg.start → delta 0 → skip.
        decisions = detect_j_cuts(
            self._segs(10.0), self._ts_pair(9.80),
            speaker_to_slot={"Speaker 1": 0, "Speaker 2": 1},
        )
        assert decisions == []


# ──────────────────── L-cut variant ────────────────────


class TestDetectLCuts:
    def test_l_cut_kind_label(self):
        ts = [
            _TS(start=8.0, end=9.95, text="One.", speaker="Speaker 1",
                words=[_Word(8.0, 9.95, "One.")]),
            _TS(start=9.9, end=12.0, text="Two.", speaker="Speaker 2",
                words=[_Word(9.9, 12.0, "Two.")]),
        ]
        segs = [
            _Seg(start=0.0, end=10.0, active_slot=0),
            _Seg(start=10.0, end=15.0, active_slot=1),
        ]
        l_decisions = detect_l_cuts(
            segs, ts,
            speaker_to_slot={"Speaker 1": 0, "Speaker 2": 1},
        )
        # Same shift as J-cut, distinct kind
        assert len(l_decisions) == 1
        assert l_decisions[0].kind == "l_cut"
        assert l_decisions[0].delta_sec == pytest.approx(0.10, abs=1e-6)


# ──────────────────── Listener-hold detection ────────────────────


class TestDetectListenerHolds:
    def test_basic_listener_hold(self):
        # Speaker A finishes at t=5.0, Speaker B starts at t=5.7
        # → 0.7 s gap → hold of min(0.7, max=0.8) = 0.7
        ts = [
            _TS(start=4.0, end=5.0, text="Are you sure?",
                speaker="Speaker 1",
                words=[
                    _Word(4.0, 4.5, "Are"),
                    _Word(4.5, 4.8, "you"),
                    _Word(4.8, 5.0, "sure?"),
                ]),
            _TS(start=5.7, end=8.0, text="Yes I am.",
                speaker="Speaker 2",
                words=[
                    _Word(5.7, 6.0, "Yes"),
                    _Word(6.0, 6.3, "I"),
                    _Word(6.3, 8.0, "am."),
                ]),
        ]
        segs = [
            _Seg(start=0.0, end=5.7, active_slot=0),
            _Seg(start=5.7, end=10.0, active_slot=1),
        ]
        slots = [_Slot(0), _Slot(1)]
        decisions = detect_listener_holds(
            segs, ts, slots,
            speaker_to_slot={"Speaker 1": 0, "Speaker 2": 1},
        )
        assert len(decisions) == 1
        assert decisions[0].kind == "listener_hold"
        assert decisions[0].new_active_slot == 1
        # Hold = min(gap=0.7, max=0.8) = 0.7
        assert decisions[0].delta_sec == pytest.approx(0.70, abs=1e-6)

    def test_listener_hold_clamped_to_max(self):
        # 2 second gap → clamped to 0.8 max
        ts = [
            _TS(start=4.0, end=5.0, text="Hello.", speaker="Speaker 1",
                words=[_Word(4.0, 5.0, "Hello.")]),
            _TS(start=7.0, end=9.0, text="Hi.", speaker="Speaker 2",
                words=[_Word(7.0, 9.0, "Hi.")]),
        ]
        segs = [
            _Seg(start=0.0, end=7.0, active_slot=0),
            _Seg(start=7.0, end=10.0, active_slot=1),
        ]
        slots = [_Slot(0), _Slot(1)]
        decisions = detect_listener_holds(
            segs, ts, slots,
            speaker_to_slot={"Speaker 1": 0, "Speaker 2": 1},
        )
        assert len(decisions) == 1
        assert decisions[0].delta_sec == pytest.approx(LISTENER_HOLD_MAX_SEC, abs=1e-6)

    def test_no_listener_hold_when_gap_too_short(self):
        # 0.1 s gap → below min_hold_sec=0.4
        ts = [
            _TS(start=4.0, end=5.0, text="Yes.", speaker="Speaker 1",
                words=[_Word(4.0, 5.0, "Yes.")]),
            _TS(start=5.1, end=7.0, text="No.", speaker="Speaker 2",
                words=[_Word(5.1, 7.0, "No.")]),
        ]
        segs = [
            _Seg(start=0.0, end=5.1, active_slot=0),
            _Seg(start=5.1, end=10.0, active_slot=1),
        ]
        slots = [_Slot(0), _Slot(1)]
        decisions = detect_listener_holds(
            segs, ts, slots,
            speaker_to_slot={"Speaker 1": 0, "Speaker 2": 1},
        )
        assert decisions == []

    def test_no_listener_hold_without_punctuation(self):
        # No sentence-end → no opportunity
        ts = [
            _TS(start=4.0, end=5.0, text="and then he", speaker="Speaker 1",
                words=[_Word(4.0, 5.0, "he")]),
            _TS(start=5.7, end=7.0, text="said something", speaker="Speaker 2",
                words=[_Word(5.7, 7.0, "something")]),
        ]
        segs = [
            _Seg(start=0.0, end=5.7, active_slot=0),
            _Seg(start=5.7, end=10.0, active_slot=1),
        ]
        slots = [_Slot(0), _Slot(1)]
        decisions = detect_listener_holds(
            segs, ts, slots,
            speaker_to_slot={"Speaker 1": 0, "Speaker 2": 1},
        )
        assert decisions == []

    def test_no_listener_hold_with_single_face(self):
        # Only one face slot → no listener available
        ts = [
            _TS(start=4.0, end=5.0, text="Hello.", speaker="Speaker 1",
                words=[_Word(4.0, 5.0, "Hello.")]),
            _TS(start=5.7, end=7.0, text="Hi.", speaker="Speaker 2",
                words=[_Word(5.7, 7.0, "Hi.")]),
        ]
        segs = [_Seg(0.0, 10.0, 0)]
        slots = [_Slot(0)]
        decisions = detect_listener_holds(
            segs, ts, slots,
            speaker_to_slot={"Speaker 1": 0, "Speaker 2": 1},
        )
        assert decisions == []

    def test_listener_hold_recognizes_question(self):
        # Question mark counts as sentence end
        ts = [
            _TS(start=4.0, end=5.0, text="Really?", speaker="Speaker 1",
                words=[_Word(4.0, 5.0, "Really?")]),
            _TS(start=5.6, end=7.0, text="Yes.", speaker="Speaker 2",
                words=[_Word(5.6, 7.0, "Yes.")]),
        ]
        segs = [_Seg(0.0, 5.6, 0), _Seg(5.6, 10.0, 1)]
        slots = [_Slot(0), _Slot(1)]
        decisions = detect_listener_holds(
            segs, ts, slots,
            speaker_to_slot={"Speaker 1": 0, "Speaker 2": 1},
        )
        assert len(decisions) == 1
        assert decisions[0].delta_sec == pytest.approx(0.6, abs=1e-6)


# ──────────────────── Reaction-beat detection ────────────────────


class TestDetectReactionBeats:
    def test_reaction_beat_swaps_active_slot(self):
        # Extreme spike at t=5.5, segment [0, 10] with active slot 0
        # → swap to slot 1
        events = [{
            "timestamp": 5.5,
            "type": "extreme_spike",
            "loudness_db": -10,
            "delta_db": 18,
        }]
        segs = [_Seg(0.0, 10.0, 0)]
        slots = [_Slot(0), _Slot(1)]
        decisions = detect_reaction_beats(segs, events, slots)
        assert len(decisions) == 1
        assert decisions[0].kind == "reaction_beat"
        assert decisions[0].new_active_slot == 1
        assert decisions[0].target_segment_idx == 0

    def test_reaction_beat_silence_to_loud(self):
        events = [{"timestamp": 3.0, "type": "silence_to_loud"}]
        segs = [_Seg(0.0, 10.0, 0)]
        slots = [_Slot(0), _Slot(1)]
        decisions = detect_reaction_beats(segs, events, slots)
        assert len(decisions) == 1

    def test_no_reaction_for_volume_spike(self):
        # Plain volume_spike isn't in REACTION_AUDIO_TYPES
        events = [{"timestamp": 3.0, "type": "volume_spike"}]
        segs = [_Seg(0.0, 10.0, 0)]
        slots = [_Slot(0), _Slot(1)]
        assert detect_reaction_beats(segs, events, slots) == []

    def test_no_reaction_with_single_face(self):
        events = [{"timestamp": 3.0, "type": "extreme_spike"}]
        segs = [_Seg(0.0, 10.0, 0)]
        slots = [_Slot(0)]  # only one face
        assert detect_reaction_beats(segs, events, slots) == []

    def test_no_reaction_when_event_outside_segment(self):
        events = [{"timestamp": 99.0, "type": "extreme_spike"}]
        segs = [_Seg(0.0, 10.0, 0)]
        slots = [_Slot(0), _Slot(1)]
        assert detect_reaction_beats(segs, events, slots) == []

    def test_empty_inputs(self):
        assert detect_reaction_beats([], [], []) == []
        assert detect_reaction_beats(
            [_Seg(0, 10, 0)], [], [_Slot(0), _Slot(1)],
        ) == []


# ──────────────────── apply_decisions in-place mutation ────────────────────


class TestApplyDecisions:
    def test_j_cut_mutates_boundary(self):
        segs = [
            _Seg(start=0.0, end=10.0, active_slot=0),
            _Seg(start=10.0, end=15.0, active_slot=1),
        ]
        decisions = [EditorialDecision(
            kind="j_cut", target_segment_idx=1, delta_sec=0.10,
        )]
        n_j, _, _, _ = apply_decisions(segs, decisions)
        assert n_j == 1
        assert segs[0].end == pytest.approx(10.10, abs=1e-6)
        assert segs[1].start == pytest.approx(10.10, abs=1e-6)

    def test_j_cut_skipped_when_below_min_segment(self):
        # Tiny segment 9.95-10.05 — applying delta +0.10 would
        # shrink the second segment to length 0 → skip.
        segs = [
            _Seg(start=0.0, end=9.95, active_slot=0),
            _Seg(start=9.95, end=10.05, active_slot=1),
            _Seg(start=10.05, end=15.0, active_slot=0),
        ]
        decisions = [EditorialDecision(
            kind="j_cut", target_segment_idx=1, delta_sec=0.10,
        )]
        n_j, _, _, _ = apply_decisions(segs, decisions)
        # Below 0.30 s floor → skipped
        assert n_j == 0
        assert segs[1].start == 9.95

    def test_reaction_beat_mutates_active_slot(self):
        segs = [_Seg(0.0, 10.0, 0)]
        decisions = [EditorialDecision(
            kind="reaction_beat", target_segment_idx=0,
            new_active_slot=1, delta_sec=0.80,
        )]
        _, _, _, n_r = apply_decisions(segs, decisions)
        assert n_r == 1
        assert segs[0].active_slot == 1
        assert segs[0].reason == "editorial_reaction_beat"

    def test_listener_hold_counted_but_not_inserted(self):
        # Listener holds are counted in the report but the
        # actual segment insertion lives in the segmenter caller
        # for Phase 8 minimal.
        segs = [_Seg(0.0, 10.0, 0)]
        decisions = [EditorialDecision(
            kind="listener_hold", target_segment_idx=0,
            new_active_slot=1, delta_sec=0.6,
        )]
        _, _, n_l, _ = apply_decisions(segs, decisions)
        assert n_l == 1
        # No mutation to segment list
        assert len(segs) == 1
        assert segs[0].active_slot == 0

    def test_empty_inputs(self):
        assert apply_decisions([], []) == (0, 0, 0, 0)
        assert apply_decisions([_Seg(0, 10, 0)], []) == (0, 0, 0, 0)


# ──────────────────── Top-level entry ────────────────────


class TestApplyEditorialPriorTopLevel:
    def test_skipped_for_non_qualifying_content(self):
        report = apply_editorial_prior(
            [_Seg(0, 10, 0)], [], [], [],
            content_profile=_Profile(content_type="gameplay"),
        )
        assert report.skipped_reason  # non-empty
        assert report.n_j_cuts == 0

    def test_skipped_for_none_profile(self):
        report = apply_editorial_prior([], [], [], [], content_profile=None)
        assert report.skipped_reason

    def test_full_pass_with_qualifying_profile(self):
        ts = [
            _TS(start=8.0, end=9.95, text="Hello.", speaker="Speaker 1",
                words=[_Word(8.0, 9.95, "Hello.")]),
            _TS(start=9.9, end=12.0, text="Hi.", speaker="Speaker 2",
                words=[_Word(9.9, 12.0, "Hi.")]),
        ]
        segs = [
            _Seg(start=0.0, end=10.0, active_slot=0),
            _Seg(start=10.0, end=15.0, active_slot=1),
        ]
        slots = [_Slot(0), _Slot(1)]
        report = apply_editorial_prior(
            segs, ts, slots, [],
            content_profile=_Profile(content_type="podcast"),
            speaker_to_slot={"Speaker 1": 0, "Speaker 2": 1},
        )
        assert not report.skipped_reason
        assert report.n_j_cuts == 1


# ──────────────────── Feature flag default ────────────────────


class TestEditorialPriorFlagDefaultOff:
    def test_default_off(self):
        # Per the v2 ground rules, ships flag-off until in-docker
        # validation lands the post-Phase-8 numbers.
        assert USE_EDITORIAL_PRIOR is False


# ──────────────────── Stage 9b AST guards ────────────────────


class TestReframeSegmenterStage9bAST:
    SRC_PATH = (
        Path(__file__).resolve().parents[1] / "services" / "reframe_segmenter.py"
    )

    def test_stage_9b_block_present(self):
        src = self.SRC_PATH.read_text()
        assert "Stage 9b" in src
        assert "EditorialPrior" in src

    def test_imports_editorial_prior_lazily(self):
        src = self.SRC_PATH.read_text()
        assert "from backend.services.editorial_prior import" in src
        assert "USE_EDITORIAL_PRIOR" in src
        assert "apply_editorial_prior" in src

    def test_signature_has_audio_events_kwarg(self):
        src = self.SRC_PATH.read_text()
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "build_reframe_segments":
                kwarg_names = [a.arg for a in node.args.args] + [
                    a.arg for a in node.args.kwonlyargs
                ]
                assert "audio_events" in kwarg_names
                return
        pytest.fail("build_reframe_segments not found")

    def test_gate_uses_applies_to_profile(self):
        src = self.SRC_PATH.read_text()
        # The Stage 9b block calls applies_to_profile on the
        # content_profile before invoking the editorial state
        # machine.
        assert "_editorial_applies" in src or "applies_to_profile" in src

    def test_stage_9b_runs_before_stage_10(self):
        # Stage 9b must run BEFORE the L1 camera path solver
        # (Stage 10) so the smoothness pass sees the
        # editorial-adjusted boundaries. We assert by ordering
        # the source-file character offsets of the two markers.
        src = self.SRC_PATH.read_text()
        i_9b = src.find("Stage 9b")
        i_10 = src.find("Stage 10:")
        assert i_9b > 0
        assert i_10 > 0
        assert i_9b < i_10

    def test_stage_9b_wrapped_in_try_except(self):
        src = self.SRC_PATH.read_text()
        assert "Editorial prior failed" in src
