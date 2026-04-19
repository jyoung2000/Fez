"""Fix 3.1: user override is a +2.0 bias, not a gate.

When geometry strongly disagrees with the hint, geometry wins. The
classifier still runs every heuristic signal; the hint only shifts
the scoring by +2.0 on the matching bucket. Geometry that disagrees
by >=2.5 dominates.
"""
from dataclasses import dataclass, field

from backend.services.content_classifier import classify_content


@dataclass
class _FaceSlot:
    slot_id: int
    x_center: float
    x_min: float = 0.0
    x_max: float = 100.0
    frame_count: int = 100
    avg_width: float = 10.0
    avg_height: float = 12.0


@dataclass
class _FaceRegistry:
    slots: list = field(default_factory=list)
    total_frames: int = 200
    frames_with_faces: int = 200

    @property
    def multi_speaker(self):
        return len(self.slots) >= 2

    @property
    def is_continuous_motion(self):
        return False


@dataclass
class _FaceInfo:
    identity_id: int
    nose_x: float
    nose_y: float = 50.0
    width: float = 10.0
    height: float = 12.0


@dataclass
class _FrameFaces:
    timestamp: float
    faces: list = field(default_factory=list)


def _anime_hint_profile():
    """Scenario per spec §3.1 acceptance:
    cut_rate=15, avg_faces=3, scene_desc_anime=0, hint='anime'.

    Geometry says podcast/narrative; hint says anime. Geometry must win.
    """
    # 240s clip, 60 cuts → cut_rate = 15/min.
    shot_cuts = [i * 4.0 for i in range(60)]
    # 2 stable seated slots → podcast geometry.
    registry = _FaceRegistry(slots=[
        _FaceSlot(0, 25, 23, 27, frame_count=200),
        _FaceSlot(1, 75, 73, 77, frame_count=200),
        _FaceSlot(2, 50, 48, 52, frame_count=150),
    ])
    # Each frame has 3 faces → avg_faces = 3.
    dense = [
        _FrameFaces(t, [
            _FaceInfo(0, 25.0), _FaceInfo(1, 75.0), _FaceInfo(2, 50.0),
        ])
        for t in range(0, 240)
    ]
    # No anime scene keywords.
    return classify_content(
        shot_cuts=shot_cuts,
        face_registry=registry,
        dense_faces=dense,
        scenes=[],
        video_duration=240.0,
        metadata={"content_type_override": "anime"},
    )


class TestOverrideIsHintNotGate:
    def test_anime_hint_does_not_win_over_podcast_geometry(self):
        """Anime hint with clear podcast/narrative geometry → geometry wins.

        Geometry dominates the +2.0 hint bias by at least 2.5 points.
        """
        profile = _anime_hint_profile()
        assert profile.content_type in ("narrative", "podcast"), (
            f"expected geometry winner, got {profile.content_type}"
        )
        assert profile.content_type != "anime"

    def test_hint_is_recorded_on_profile(self):
        """Even when the hint loses, the normalized hint is preserved
        on ``profile.user_hint`` so telemetry can show it."""
        profile = _anime_hint_profile()
        assert profile.user_hint == "anime"

    def test_hint_bias_recorded_in_signals(self):
        """``profile.signals['user_hint_bias']`` == 2.0 when the bias
        path was taken (i.e. non-degenerate inputs)."""
        profile = _anime_hint_profile()
        # Bias of 2.0 was applied; signal confirms the non-gate path
        # ran (as opposed to the degenerate-inputs short-circuit).
        assert profile.signals.get("user_hint_bias") == 2.0
        assert profile.signals.get("user_hint") == "anime"

    def test_confidence_is_not_sentinel_when_geometry_overrides(self):
        """Pre-3.1 confidence was hard-coded 1.0 on override. Post-3.1,
        it's driven by the score distribution — never hard-coded 1.0 on
        a non-degenerate hint."""
        profile = _anime_hint_profile()
        assert profile.confidence != 1.0


class TestHintStillHelpsWhenHeuristicsAreWeak:
    """A +2.0 bias DOES tip genuinely ambiguous cases — that's the point.
    Here the heuristic alone would bail to UNKNOWN / low-confidence;
    the hint gives it a decisive push toward the user's preference."""

    def test_weak_signals_plus_hint_routes_correctly(self):
        # A small amount of dense data, no scene descriptions, a single
        # face slot — talking-head-ish but not strongly so. User picks
        # "vlog" — the hint tips the balance toward vlog.
        registry = _FaceRegistry(slots=[
            _FaceSlot(0, 50, 45, 55, frame_count=100),
        ])
        dense = [
            _FrameFaces(t, [_FaceInfo(0, 50.0)])
            for t in range(0, 30)
        ]
        profile = classify_content(
            shot_cuts=[2.0, 10.0, 20.0],  # cut_rate = 6/min (> 2, < 10)
            face_registry=registry,
            dense_faces=dense,
            scenes=[],
            video_duration=30.0,
            metadata={"content_type_override": "vlog"},
        )
        # With vlog hint (+2.0) + cut_rate-band bonus (+1.5 vlog) +
        # single-slot bonus (+1.5 vlog if avg_range < 10), vlog wins.
        assert profile.content_type == "vlog"
        assert profile.user_hint == "vlog"


class TestGameplayHintStaysGate:
    """Gaming is explicitly out of scope for the 3.x overhaul — its
    pipeline is byte-for-byte untouched. A gameplay hint must still
    short-circuit to confidence=1.0 the way pre-3.1 did."""

    def test_gameplay_hint_short_circuits(self):
        profile = classify_content(
            shot_cuts=[i * 3.0 for i in range(80)],
            face_registry=_FaceRegistry(),
            dense_faces=[],
            scenes=[],
            video_duration=240.0,
            metadata={"content_type_override": "gameplay"},
        )
        assert profile.content_type == "gaming"
        assert profile.confidence == 1.0

    def test_stream_hint_short_circuits(self):
        profile = classify_content(
            shot_cuts=[],
            face_registry=_FaceRegistry(),
            dense_faces=[],
            scenes=[],
            video_duration=60.0,
            metadata={"content_type_override": "stream"},
        )
        assert profile.content_type == "gaming"
        assert profile.confidence == 1.0
