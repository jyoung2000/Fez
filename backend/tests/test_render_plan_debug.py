"""Unit tests for backend.services.render_plan_debug.

The Phase 10 debug payload is what lights up the
``ReframeDebugOverlay`` chip row and per-segment editorial tags. If
these fields silently stop emitting, the overlay goes blank and an
editor debugging a bad crop loses the most useful signal we have.

The tests here lock the contract between ``build_debug_payload`` and
the frontend overlay:

1. Header chips — ``content_type``, ``clip_content_type``,
   ``anime_subtype``, ``music_subtype``, ``gameplay_subtype``,
   ``game_type``, ``is_multi_speaker_panel``, ``is_animated``.
2. Per-segment arrays — ``confidence_per_segment``,
   ``reason_per_segment``, ``fallback_reasons``.
3. Editorial prior — ``editorial_prior_per_segment`` indexed by
   target segment, plus the ``editorial_summary`` counters.
4. Pacing sparklines from ``LocalPacingEstimator``.
5. Never-raise guarantee — pass garbage kwargs, get an empty dict.
"""

from __future__ import annotations

from types import SimpleNamespace

from backend.services.render_plan_debug import build_debug_payload


def _seg(**kw):
    """Build a duck-typed ReframeSegment for the tests."""
    return SimpleNamespace(
        start=kw.get("start", 0.0),
        end=kw.get("end", 1.0),
        subject_x=kw.get("subject_x", 960.0),
        subject_y=kw.get("subject_y", 540.0),
        confidence=kw.get("confidence", 1.0),
        reason=kw.get("reason", "hold"),
        fallback_reason=kw.get("fallback_reason", None),
        subject_source=kw.get("subject_source", ""),
        strategy=kw.get("strategy", "stationary"),
    )


def _job(**kw):
    return SimpleNamespace(
        content_type_override=kw.get("content_type_override", ""),
        anime_subtype=kw.get("anime_subtype", ""),
        music_subtype=kw.get("music_subtype", ""),
        game_type=kw.get("game_type", ""),
    )


# ────────────────────────────────────────────────────────────────
# Header chips — content routing
# ────────────────────────────────────────────────────────────────

class TestContentRouting:
    def test_empty_job_produces_minimal_debug(self):
        debug = build_debug_payload()
        assert isinstance(debug, dict)
        # No header fields should be set
        assert "content_type" not in debug
        assert "anime_subtype" not in debug

    def test_podcast_override_produces_content_type(self):
        job = _job(content_type_override="podcast")
        debug = build_debug_payload(job=job)
        assert debug["content_type"] == "podcast"

    def test_debate_override_sets_panel_flag(self):
        job = _job(content_type_override="debate")
        debug = build_debug_payload(job=job)
        assert debug["content_type"] == "podcast"
        assert debug["is_multi_speaker_panel"] is True

    def test_anime_override_sets_animated_flag_and_subtype(self):
        job = _job(content_type_override="anime", anime_subtype="action")
        debug = build_debug_payload(job=job)
        assert debug["content_type"] == "anime"
        assert debug["is_animated"] is True
        assert debug["anime_subtype"] == "action"

    def test_music_video_override_passes_subtype(self):
        job = _job(
            content_type_override="music_video",
            music_subtype="performance",
        )
        debug = build_debug_payload(job=job)
        assert debug["content_type"] == "music_video"
        assert debug["music_subtype"] == "performance"

    def test_gameplay_moba_override_sets_gameplay_subtype(self):
        job = _job(
            content_type_override="gameplay_moba",
            game_type="league_of_legends",
        )
        debug = build_debug_payload(job=job)
        assert debug["content_type"] == "gaming"
        assert debug["gameplay_subtype"] == "moba"
        assert debug["game_type"] == "league_of_legends"

    def test_stream_token_is_not_gameplay_fastpath_but_stream_subtype(self):
        job = _job(content_type_override="stream")
        debug = build_debug_payload(job=job)
        assert debug["content_type"] == "gaming"
        assert debug["gameplay_subtype"] == "stream"

    def test_auto_override_produces_no_content_type(self):
        job = _job(content_type_override="auto")
        debug = build_debug_payload(job=job)
        # Auto means "no override" — no chip should render
        assert "content_type" not in debug

    def test_content_profile_fallback_when_no_override(self):
        job = _job(content_type_override="")
        profile = SimpleNamespace(
            content_type="narrative",
            is_multi_speaker_panel=False,
            is_animated=False,
            gameplay_subtype=None,
        )
        debug = build_debug_payload(job=job, content_profile=profile)
        assert debug["content_type"] == "narrative"


# ────────────────────────────────────────────────────────────────
# Per-segment arrays
# ────────────────────────────────────────────────────────────────

class TestPerSegment:
    def test_confidence_per_segment_rounds_to_3dp(self):
        segs = [
            _seg(confidence=0.8767),  # 3dp = 0.877
            _seg(confidence=0.12345),  # 3dp = 0.123 (banker's rounding)
        ]
        debug = build_debug_payload(reframe_segments=segs)
        assert debug["confidence_per_segment"] == [0.877, 0.123]

    def test_reason_per_segment_prefers_fallback_reason(self):
        segs = [
            _seg(fallback_reason="wide_master_fallback"),
            _seg(fallback_reason=None, subject_source="face_registry"),
            _seg(fallback_reason=None, subject_source="anime_anchor_face"),
        ]
        debug = build_debug_payload(reframe_segments=segs)
        reasons = debug.get("reason_per_segment", [])
        assert reasons[0] == "wide_master_fallback"
        # face_registry is a mundane source so it's hidden
        assert reasons[1] is None
        # anime_anchor_face is interesting — should surface
        assert reasons[2] == "anime_anchor_face"

    def test_fallback_reasons_legacy_shape(self):
        segs = [
            _seg(reason="wide_master_fallback"),
            _seg(reason="hold"),
            _seg(reason="blur_fill"),
        ]
        debug = build_debug_payload(reframe_segments=segs)
        assert debug["fallback_reasons"][0] == "wide_master_fallback"
        assert debug["fallback_reasons"][1] is None
        assert debug["fallback_reasons"][2] == "blur_fill"

    def test_empty_segments_produces_no_per_segment_arrays(self):
        debug = build_debug_payload(reframe_segments=[])
        assert "confidence_per_segment" not in debug
        assert "reason_per_segment" not in debug


# ────────────────────────────────────────────────────────────────
# Editorial prior integration
# ────────────────────────────────────────────────────────────────

class TestEditorialPrior:
    def test_editorial_decisions_indexed_by_segment(self):
        segs = [_seg(), _seg(), _seg()]
        decisions = [
            SimpleNamespace(kind="j_cut", target_segment_idx=0),
            SimpleNamespace(kind="listener_hold", target_segment_idx=1),
            SimpleNamespace(kind="reaction_beat", target_segment_idx=1),
        ]
        report = SimpleNamespace(
            decisions=decisions,
            n_j_cuts=1,
            n_l_cuts=0,
            n_listener_holds=1,
            n_reaction_beats=1,
        )
        debug = build_debug_payload(
            reframe_segments=segs,
            editorial_report=report,
        )
        per_seg = debug["editorial_prior_per_segment"]
        assert per_seg[0] == ["j_cut"]
        assert set(per_seg[1]) == {"listener_hold", "reaction_beat"}
        assert per_seg[2] == []
        assert debug["editorial_summary"]["n_j_cuts"] == 1
        assert debug["editorial_summary"]["n_listener_holds"] == 1
        assert debug["editorial_summary"]["n_reaction_beats"] == 1

    def test_no_decisions_omits_per_segment_field(self):
        segs = [_seg()]
        report = SimpleNamespace(
            decisions=[],
            n_j_cuts=0,
            n_l_cuts=0,
            n_listener_holds=0,
            n_reaction_beats=0,
        )
        debug = build_debug_payload(
            reframe_segments=segs,
            editorial_report=report,
        )
        assert "editorial_prior_per_segment" not in debug

    def test_out_of_range_decisions_are_dropped(self):
        segs = [_seg(), _seg()]
        decisions = [
            SimpleNamespace(kind="j_cut", target_segment_idx=-1),
            SimpleNamespace(kind="l_cut", target_segment_idx=99),
            SimpleNamespace(kind="j_cut", target_segment_idx=1),
        ]
        report = SimpleNamespace(
            decisions=decisions,
            n_j_cuts=2,
            n_l_cuts=1,
            n_listener_holds=0,
            n_reaction_beats=0,
        )
        debug = build_debug_payload(
            reframe_segments=segs,
            editorial_report=report,
        )
        per_seg = debug["editorial_prior_per_segment"]
        assert per_seg[0] == []
        assert per_seg[1] == ["j_cut"]


# ────────────────────────────────────────────────────────────────
# Pacing sparklines
# ────────────────────────────────────────────────────────────────

class TestPacingSparklines:
    def test_pacing_and_min_hold_from_estimator(self):
        estimator = SimpleNamespace(
            pacing_per_sec=[0.1, 0.2, 0.3],
            min_hold_per_sec=[2.0, 1.5, 1.0],
        )
        debug = build_debug_payload(pacing_estimator=estimator)
        assert debug["pacing_per_sec"] == [0.1, 0.2, 0.3]
        assert debug["min_hold_per_sec"] == [2.0, 1.5, 1.0]

    def test_missing_estimator_omits_fields(self):
        debug = build_debug_payload()
        assert "pacing_per_sec" not in debug
        assert "min_hold_per_sec" not in debug


# ────────────────────────────────────────────────────────────────
# Robustness — must never raise
# ────────────────────────────────────────────────────────────────

class TestRobustness:
    def test_weird_job_does_not_raise(self):
        weird = SimpleNamespace()
        debug = build_debug_payload(job=weird)
        assert isinstance(debug, dict)

    def test_weird_segments_do_not_raise(self):
        weird_segs = [SimpleNamespace()]  # no .confidence, no .reason
        debug = build_debug_payload(reframe_segments=weird_segs)
        # confidence defaults to 1.0 when missing
        assert debug["confidence_per_segment"] == [1.0]

    def test_weird_editorial_report_does_not_raise(self):
        debug = build_debug_payload(
            reframe_segments=[_seg()],
            editorial_report=SimpleNamespace(),  # no .decisions
        )
        assert isinstance(debug, dict)
