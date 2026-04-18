"""Phase 9 — Synthetic fixture set for AutoFlip parity measurement.

Mirrors the in-memory dataclass-stub pattern from
``backend.scripts.measure_reframe_lag`` so the fixtures don't need
ffmpeg / .mp4 binaries on disk. Every fixture is built from pure
Python lists + dataclass stubs; the runner script wires them through
``build_reframe_segments`` (which DOES need numpy at the
subject-confidence layer) only when scoring a real result.

Fixture set (matches the v2 spec):

  1. synthetic_2speaker_alternating  (existing — sub-second recall + overlap)
  2. synthetic_3speaker_panel        (debate routing + multi-region)
  3. synthetic_vlog_walk_and_talk    (vlog + thirds bias)
  4. synthetic_music_video_beat      (beat snap)
  5. synthetic_anime_hard_cuts       (anime + sub bar preservation)
  6. synthetic_tps_character_offset  (TPS action-center anchor)
  7. synthetic_stream_corner_facecam (stream + STACKED_GAMEPLAY)

Each ``FixtureSpec`` carries:

- ``name``                     — short ID used by the CLI
- ``description``              — human-readable summary
- ``content_type_override``    — UI dropdown token (Phase 1+2 normalizer)
- ``anime_subtype`` / ``music_subtype`` / ``game_type`` — Phase 2 sub-fields
- ``video_duration``           — seconds
- ``source_width`` / ``source_height`` — pixel dimensions (default 1920x1080)
- ``crop_aspect``              — output aspect (default 9/16)
- ``ground_truth``             — a ``GroundTruth`` object with the
  reference data the metrics need
- ``metrics``                  — which metrics to score on this fixture
- ``build``                    — callable returning the kwargs to pass
  into ``build_reframe_segments``

Registering a new fixture is a 30-line edit: add the ``FixtureSpec``
to ``FIXTURES`` and Phase 3-8 PRs immediately get the new metric row.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional


# ── Stub dataclasses (same shapes as measure_reframe_lag) ──────────

@dataclass
class _FaceSlot:
    slot_id: int
    x_center: float
    x_min: float = 0.0
    x_max: float = 100.0
    frame_count: int = 100
    avg_width: float = 10.0
    avg_height: float = 12.0
    y_center: float = 50.0


@dataclass
class _FaceRegistry:
    slots: list = field(default_factory=list)
    total_frames: int = 100
    frames_with_faces: int = 100
    # ``is_continuous_motion`` is read by classify_content's Signal 3
    # branch when the fixture has a single-slot vlog-style face. The
    # parity fixtures don't have actual motion data, so we default
    # to False (consistent with stationary panels / podcasts) and
    # let the vlog fixture override per its build callable.
    is_continuous_motion: bool = False

    @property
    def multi_speaker(self) -> bool:
        return len(self.slots) >= 2

    def slot_by_id(self, slot_id):
        for s in self.slots:
            if s.slot_id == slot_id:
                return s
        return None

    def nearest_slot(self, x: float):
        if not self.slots:
            return None
        return min(self.slots, key=lambda s: abs(s.x_center - x))


@dataclass
class _SpeakerEvent:
    start: float
    end: float
    slot_id: int
    confidence: float


@dataclass
class _FaceInfo:
    identity_id: int
    nose_x: float
    nose_y: float = 50.0
    width: float = 10.0
    height: float = 12.0
    lip_aperture: float = 0.05
    is_speaking: bool = True
    confidence: float = 0.95
    x_center: float = 0.0
    y_center: float = 50.0


@dataclass
class _FrameFaces:
    timestamp: float
    frame_path: str = ""
    faces: list = field(default_factory=list)
    primary_face_idx: int = -1


@dataclass
class _TranscriptSeg:
    start: float
    end: float
    text: str = ""
    speaker: str = "Speaker 1"
    confidence: Optional[float] = 0.95
    words: Optional[list] = None
    avg_logprob: Optional[float] = None
    no_speech_prob: Optional[float] = None


# ── Ground-truth container ─────────────────────────────────────────

@dataclass
class GroundTruth:
    """Reference data the metrics module compares actual output against."""

    # Speaker-change times in seconds. Used by sub_second_switch_recall.
    expected_switches: list[float] = field(default_factory=list)
    # Beat grid in seconds. Used by downbeat_snap_error (music video).
    beat_grid: list[float] = field(default_factory=list)
    # Per-frame required regions: list of [(left_pct, right_pct), ...].
    # Length must match the dense face frame count of the fixture.
    required_regions_per_frame: list = field(default_factory=list)
    # HUD bboxes in source-frame % space. Used by hud_preservation_rate.
    hud_zones: list = field(default_factory=list)
    # Source HUD-zone game key for documentation / debug.
    game_key: str = ""
    # Per-frame face centroid Y normalized to crop frame [0, 1]. Used
    # by face_centroid_in_thirds_rate. Phase 4 will populate this from
    # actual reframe output; for the baseline we leave it empty so the
    # metric scores 0.0 (no thirds compliance yet).
    face_y_in_crop_normalized: list[float] = field(default_factory=list)


# ── Top-level fixture spec ─────────────────────────────────────────

@dataclass
class FixtureSpec:
    name: str
    description: str
    content_type_override: str
    video_duration: float
    metrics: list[str]
    build: Callable[[], dict]
    ground_truth: GroundTruth = field(default_factory=GroundTruth)
    anime_subtype: str = ""
    music_subtype: str = ""
    game_type: str = ""
    source_width: int = 1920
    source_height: int = 1080
    crop_aspect: float = 9 / 16

    @property
    def crop_width_pct(self) -> float:
        """Crop width as % of source frame width.

        For 16:9 → 9:16, the crop window has width
        ``source_height * (9/16) = source_height * crop_aspect`` in
        pixels, which is ``crop_aspect * source_height / source_width``
        as a fraction of source width — and ``× 100`` for percent.
        """
        if self.source_width <= 0 or self.source_height <= 0:
            return 0.0
        crop_width_px = self.source_height * self.crop_aspect
        if crop_width_px > self.source_width:
            crop_width_px = self.source_width
        return (crop_width_px / self.source_width) * 100.0


# ── Helpers ─────────────────────────────────────────────────────────

def _make_dense(slot_id: int, x: float, start: float, end: float, step: float = 0.1) -> list:
    """Build a list of _FrameFaces with one face at ``x`` from start..end."""
    out = []
    t = start
    while t < end:
        out.append(_FrameFaces(
            timestamp=round(t, 4),
            faces=[_FaceInfo(identity_id=slot_id, nose_x=x, x_center=x)],
        ))
        t += step
    return out


def _make_dense_multi(positions_at_t: list[tuple[float, list[tuple[int, float]]]]) -> list:
    """Build dense frames with multiple faces per frame.

    ``positions_at_t`` is a list of ``(timestamp, [(slot_id, nose_x), ...])``.
    """
    out = []
    for t, faces in positions_at_t:
        out.append(_FrameFaces(
            timestamp=round(t, 4),
            faces=[
                _FaceInfo(identity_id=sid, nose_x=x, x_center=x)
                for sid, x in faces
            ],
        ))
    return out


# ── Fixture builders ───────────────────────────────────────────────

def _build_2speaker_alternating() -> dict:
    """Existing fixture from measure_reframe_lag (verbatim shape).

    Two speakers at x=25 and x=75, alternating every 2s for 20s.
    """
    registry = _FaceRegistry(slots=[
        _FaceSlot(slot_id=0, x_center=25.0),
        _FaceSlot(slot_id=1, x_center=75.0),
    ])
    speaker_to_slot = {"Speaker 1": 0, "Speaker 2": 1}

    transcript: list[_TranscriptSeg] = []
    as_events: list[_SpeakerEvent] = []
    dense: list[_FrameFaces] = []

    for i in range(10):
        t_start = i * 2.0
        t_end = (i + 1) * 2.0
        spk = "Speaker 1" if i % 2 == 0 else "Speaker 2"
        slot = 0 if i % 2 == 0 else 1
        x = 25.0 if slot == 0 else 75.0
        transcript.append(_TranscriptSeg(start=t_start, end=t_end, speaker=spk))
        as_events.append(_SpeakerEvent(start=t_start, end=t_end, slot_id=slot, confidence=0.95))
        dense.extend(_make_dense(slot, x, t_start, t_end, step=0.1))

    return dict(
        shot_cuts=[],
        face_registry=registry,
        active_speaker_events=as_events,
        dense_faces=dense,
        transcript_segments=transcript,
        speaker_to_slot=speaker_to_slot,
        video_duration=20.0,
    )


def _build_3speaker_panel() -> dict:
    """3-seat panel for debate routing.

    Speakers at x=20 / 50 / 80, rotating turns every 1.5s for 18s.
    Tests MULTI_SPEAKER_PANEL routing + multi-region containment.
    """
    registry = _FaceRegistry(slots=[
        _FaceSlot(slot_id=0, x_center=20.0),
        _FaceSlot(slot_id=1, x_center=50.0),
        _FaceSlot(slot_id=2, x_center=80.0),
    ])
    speaker_to_slot = {f"Speaker {i+1}": i for i in range(3)}

    transcript: list[_TranscriptSeg] = []
    as_events: list[_SpeakerEvent] = []
    dense: list[_FrameFaces] = []

    n_turns = 12
    for i in range(n_turns):
        t_start = i * 1.5
        t_end = (i + 1) * 1.5
        slot = i % 3
        spk = f"Speaker {slot + 1}"
        transcript.append(_TranscriptSeg(start=t_start, end=t_end, speaker=spk))
        as_events.append(_SpeakerEvent(start=t_start, end=t_end, slot_id=slot, confidence=0.92))
        # All three faces present every frame (it's a static panel),
        # but only the active speaker has lip aperture > 0.
        positions = []
        t = t_start
        while t < t_end:
            faces_at_t = [
                _FaceInfo(
                    identity_id=sid,
                    nose_x=x,
                    x_center=x,
                    lip_aperture=0.10 if sid == slot else 0.02,
                    is_speaking=(sid == slot),
                )
                for sid, x in [(0, 20.0), (1, 50.0), (2, 80.0)]
            ]
            positions.append(_FrameFaces(timestamp=round(t, 4), faces=faces_at_t))
            t += 0.1
        dense.extend(positions)

    return dict(
        shot_cuts=[],
        face_registry=registry,
        active_speaker_events=as_events,
        dense_faces=dense,
        transcript_segments=transcript,
        speaker_to_slot=speaker_to_slot,
        video_duration=18.0,
    )


def _build_vlog_walk_and_talk() -> dict:
    """Single subject, head moves 30% of frame width over 10s.

    Tests vlog routing + future thirds-bias logic. The face starts at
    x=35 and ends at x=65 in a smooth linear walk.
    """
    registry = _FaceRegistry(slots=[
        _FaceSlot(slot_id=0, x_center=50.0, x_min=35.0, x_max=65.0),
    ])
    speaker_to_slot = {"Speaker 1": 0}

    duration = 10.0
    fps = 10.0  # dense sample rate
    n = int(duration * fps)
    transcript = [_TranscriptSeg(start=0.0, end=duration, speaker="Speaker 1")]
    as_events = [_SpeakerEvent(start=0.0, end=duration, slot_id=0, confidence=0.95)]
    dense = []
    for i in range(n):
        t = i / fps
        # Linear walk 35 → 65
        x = 35.0 + (i / max(n - 1, 1)) * 30.0
        dense.append(_FrameFaces(
            timestamp=round(t, 4),
            faces=[_FaceInfo(identity_id=0, nose_x=x, x_center=x, nose_y=42.0)],
        ))

    return dict(
        shot_cuts=[],
        face_registry=registry,
        active_speaker_events=as_events,
        dense_faces=dense,
        transcript_segments=transcript,
        speaker_to_slot=speaker_to_slot,
        video_duration=duration,
    )


def _build_music_video_beat() -> dict:
    """Music video at 120 BPM (downbeat every 0.5s) with 4 alternating
    speakers across 12 seconds. The beat grid is the ground truth for
    downbeat_snap_error — Phase 5 must hit ±200 ms or better.
    """
    registry = _FaceRegistry(slots=[
        _FaceSlot(slot_id=0, x_center=20.0),
        _FaceSlot(slot_id=1, x_center=50.0),
        _FaceSlot(slot_id=2, x_center=80.0),
        _FaceSlot(slot_id=3, x_center=65.0),
    ])
    speaker_to_slot = {f"Speaker {i+1}": i for i in range(4)}

    transcript: list[_TranscriptSeg] = []
    as_events: list[_SpeakerEvent] = []
    dense: list[_FrameFaces] = []

    duration = 12.0
    bpm = 120.0
    beat_interval = 60.0 / bpm  # 0.5 s
    # One subject change every 4 beats = every 2 s.
    for i in range(int(duration / 2.0)):
        t_start = i * 2.0
        t_end = (i + 1) * 2.0
        slot = i % 4
        x = registry.slots[slot].x_center
        spk = f"Speaker {slot + 1}"
        transcript.append(_TranscriptSeg(start=t_start, end=t_end, speaker=spk))
        as_events.append(_SpeakerEvent(start=t_start, end=t_end, slot_id=slot, confidence=0.95))
        dense.extend(_make_dense(slot, x, t_start, t_end, step=0.05))

    return dict(
        shot_cuts=[],  # no shot cuts in this fixture; pulse cuts come from beat snap
        face_registry=registry,
        active_speaker_events=as_events,
        dense_faces=dense,
        transcript_segments=transcript,
        speaker_to_slot=speaker_to_slot,
        video_duration=duration,
    )


def _build_anime_hard_cuts() -> dict:
    """Anime-style fixture: hard cuts every 2 s, characters at fixed
    positions within each shot. Subtitle-bar at y=90% is documented as
    a required region so Phase 6's sub-preservation logic gets a
    fixture row to optimize against.
    """
    registry = _FaceRegistry(slots=[
        _FaceSlot(slot_id=0, x_center=30.0),
        _FaceSlot(slot_id=1, x_center=70.0),
    ])
    speaker_to_slot = {"Speaker 1": 0, "Speaker 2": 1}

    transcript: list[_TranscriptSeg] = []
    as_events: list[_SpeakerEvent] = []
    dense: list[_FrameFaces] = []
    shot_cuts: list[float] = []

    n_shots = 6
    for i in range(n_shots):
        t_start = i * 2.0
        t_end = (i + 1) * 2.0
        slot = i % 2
        x = 30.0 if slot == 0 else 70.0
        spk = f"Speaker {slot + 1}"
        transcript.append(_TranscriptSeg(start=t_start, end=t_end, speaker=spk))
        as_events.append(_SpeakerEvent(start=t_start, end=t_end, slot_id=slot, confidence=0.92))
        dense.extend(_make_dense(slot, x, t_start, t_end, step=0.1))
        if i > 0:
            shot_cuts.append(t_start)

    return dict(
        shot_cuts=shot_cuts,
        face_registry=registry,
        active_speaker_events=as_events,
        dense_faces=dense,
        transcript_segments=transcript,
        speaker_to_slot=speaker_to_slot,
        video_duration=12.0,
    )


def _build_anime_panning_close_up() -> dict:
    """Anime close-up shot whose face pans across the frame.

    Phase 6 — exercises the full anime reframe gap close: dense face
    track is populated (ratio = 1.0), the face nose_x pans 45 → 60 over
    8 seconds within a single shot. Phase 5's intra-shot motion override
    must flip the strategy from `stationary` to `tracking` (15 % motion
    is well above the 6 % threshold), and the Phase 4 per-frame anchor
    signal must keep the L1 path within ~4 % of the per-frame nose_x.

    Mirrors the production AoT job ``f218fad4-…`` where
    ``fetchJob: 614 scenes (0 dense face tracking, 614 AI)`` reduced
    the L1 solver to slot-center medians and faces drifted off-center
    in close-up shots.
    """
    registry = _FaceRegistry(
        slots=[_FaceSlot(slot_id=0, x_center=52.5, x_min=45.0, x_max=60.0)],
        is_continuous_motion=True,
    )
    speaker_to_slot = {"Speaker 1": 0}

    duration = 8.0
    fps = 2.0  # matches DENSE_FACE_SAMPLE_RATE = 0.5s
    n = int(duration * fps)
    dense: list[_FrameFaces] = []
    for i in range(n):
        t = i / fps
        # Linear pan from 45 → 60 percent over the shot
        x = 45.0 + (60.0 - 45.0) * (i / max(n - 1, 1))
        dense.append(_FrameFaces(
            timestamp=round(t, 4),
            faces=[_FaceInfo(
                identity_id=0, nose_x=x, x_center=x, y_center=50.0,
            )],
        ))

    transcript = [
        _TranscriptSeg(start=0.0, end=duration, speaker="Speaker 1"),
    ]
    as_events = [
        _SpeakerEvent(start=0.0, end=duration, slot_id=0, confidence=0.92),
    ]

    return dict(
        shot_cuts=[],
        face_registry=registry,
        active_speaker_events=as_events,
        dense_faces=dense,
        transcript_segments=transcript,
        speaker_to_slot=speaker_to_slot,
        video_duration=duration,
    )


def _gt_anime_panning_close_up() -> GroundTruth:
    """Per-frame ground truth for the panning close-up fixture.

    Required region per frame is the face bbox at the panning x.
    Face width is 12 % so the region tracks the face exactly.
    """
    duration = 8.0
    fps = 2.0
    n = int(duration * fps)
    face_w = 12.0
    half = face_w / 2.0
    regions: list[list[tuple[float, float]]] = []
    for i in range(n):
        x = 45.0 + (60.0 - 45.0) * (i / max(n - 1, 1))
        regions.append([(x - half, x + half)])
    return GroundTruth(
        expected_switches=[],
        required_regions_per_frame=regions,
    )


def _build_tps_character_offset() -> dict:
    """TPS gameplay: player character at x=40 / y=55, HUD in corners.

    The action_center_pct anchor for TPS is (50, 45) — Phase 7 must
    follow the character to its actual on-screen offset rather than
    locking to the hard-coded center. The baseline (Phase 1) just
    center-crops via the gameplay fast path and will report a high
    miss rate against the off-center character region.
    """
    registry = _FaceRegistry(slots=[])  # gameplay has no faces
    speaker_to_slot: dict = {}

    transcript: list[_TranscriptSeg] = []
    as_events: list[_SpeakerEvent] = []
    dense: list[_FrameFaces] = []

    duration = 10.0
    fps = 10.0
    n = int(duration * fps)
    for i in range(n):
        t = i / fps
        dense.append(_FrameFaces(timestamp=round(t, 4), faces=[]))

    return dict(
        shot_cuts=[],
        face_registry=registry,
        active_speaker_events=as_events,
        dense_faces=dense,
        transcript_segments=transcript,
        speaker_to_slot=speaker_to_slot,
        video_duration=duration,
    )


def _build_stream_corner_facecam() -> dict:
    """Stream: gameplay center + facecam in top-right corner.

    Tests Phase 7's STACKED_GAMEPLAY routing. The fixture has one face
    pinned at x=85 / y=15 (top-right corner) for the entire duration,
    with no speaker turns — the only "subject" is the facecam.
    """
    registry = _FaceRegistry(slots=[
        _FaceSlot(slot_id=0, x_center=85.0, y_center=15.0, x_min=82.0, x_max=88.0),
    ])
    speaker_to_slot = {"Speaker 1": 0}

    transcript = [_TranscriptSeg(start=0.0, end=10.0, speaker="Speaker 1")]
    as_events = [_SpeakerEvent(start=0.0, end=10.0, slot_id=0, confidence=0.9)]
    dense = []
    duration = 10.0
    fps = 10.0
    n = int(duration * fps)
    for i in range(n):
        t = i / fps
        dense.append(_FrameFaces(
            timestamp=round(t, 4),
            faces=[_FaceInfo(
                identity_id=0,
                nose_x=85.0,
                nose_y=15.0,
                x_center=85.0,
                y_center=15.0,
            )],
        ))

    return dict(
        shot_cuts=[],
        face_registry=registry,
        active_speaker_events=as_events,
        dense_faces=dense,
        transcript_segments=transcript,
        speaker_to_slot=speaker_to_slot,
        video_duration=duration,
    )


# ── Ground-truth helpers ───────────────────────────────────────────

def _gt_2speaker_alternating() -> GroundTruth:
    """Per-frame ground truth for the 2-speaker fixture.

    Phase 3 populates the per-frame required-region track so the
    ``required_region_miss_rate`` metric scores against actual face
    bbox positions instead of empty data. The two speakers sit at
    x=25 % (slot 0) and x=75 % (slot 1) with a 10 % bbox width.
    Only the ACTIVE speaker is required at each timestamp — the
    passive speaker is implicitly optional (Phase 4+ will track
    optionals separately if needed).
    """
    duration = 20.0
    fps = 10.0  # matches _build_2speaker_alternating's step=0.1
    n = int(duration * fps)
    regions: list[list[tuple[float, float]]] = []
    face_w = 10.0
    half = face_w / 2.0
    for i in range(n):
        t = i / fps
        # Speaker alternates every 2 s. Index 0 active for t<2, then
        # index 1, etc.
        slot_idx = int(t // 2.0) % 2
        cx = 25.0 if slot_idx == 0 else 75.0
        regions.append([(cx - half, cx + half)])
    return GroundTruth(
        expected_switches=[float(i) * 2.0 for i in range(1, 10)],
        required_regions_per_frame=regions,
    )


def _gt_3speaker_panel() -> GroundTruth:
    """Per-frame ground truth for the 3-seat panel fixture.

    Speaker rotates every 1.5 s across slots 0 / 1 / 2 (x = 20, 50, 80).
    The active speaker's bbox is the per-frame required region.
    """
    duration = 18.0
    fps = 10.0
    n = int(duration * fps)
    regions: list[list[tuple[float, float]]] = []
    face_w = 10.0
    half = face_w / 2.0
    centers_pct = [20.0, 50.0, 80.0]
    for i in range(n):
        t = i / fps
        slot_idx = int(t / 1.5) % 3
        cx = centers_pct[slot_idx]
        regions.append([(cx - half, cx + half)])
    return GroundTruth(
        expected_switches=[i * 1.5 for i in range(1, 12)],
        required_regions_per_frame=regions,
    )


def _gt_vlog_walk_and_talk() -> GroundTruth:
    # No speaker change — single subject. Ground truth has zero
    # expected switches; sub_second_recall scores 1.0 vacuously.
    # Required regions: the face bbox at every frame, ~10% wide,
    # walking from 35 → 65 over 100 frames.
    duration = 10.0
    fps = 10.0
    n = int(duration * fps)
    regions: list[list[tuple[float, float]]] = []
    face_w = 10.0
    for i in range(n):
        x = 35.0 + (i / max(n - 1, 1)) * 30.0
        regions.append([(x - face_w / 2, x + face_w / 2)])
    return GroundTruth(
        expected_switches=[],
        required_regions_per_frame=regions,
    )


def _gt_music_video_beat() -> GroundTruth:
    # 120 BPM downbeat grid: 0, 0.5, 1.0, ..., for 12 s.
    bpm = 120.0
    beat_interval = 60.0 / bpm
    duration = 12.0
    n_beats = int(duration / beat_interval)
    beat_grid = [round(i * beat_interval, 4) for i in range(n_beats + 1)]
    # Speaker change every 2 s (4 beats).
    expected = [float(i) * 2.0 for i in range(1, int(duration / 2.0))]
    return GroundTruth(expected_switches=expected, beat_grid=beat_grid)


def _gt_anime_hard_cuts() -> GroundTruth:
    """Per-frame ground truth for the anime hard-cuts fixture.

    Phase 6 dropped the sub-bar approach in favor of the
    "right moment" framing test. Subtitle bars in anime are
    typically wider than a 9:16 vertical crop and cannot be
    physically contained — chasing the bar would yank the crop
    away from the dramatic anchor (the face, the impact frame,
    the reaction shot). That's the wrong call for anime.

    The Phase 6 ground truth instead measures whether the
    segmenter lands the crop on the **active anime speaker /
    anchor** at every frame:

      - Speakers at slot 0 (x=30) and slot 1 (x=70), face
        bbox 10 % wide
      - Speaker rotates every 2 s (matches the 6 hard cuts)
      - Required region per frame = active speaker's face bbox

    With the existing single-subject path the segmenter already
    centers on each speaker → miss rate ≈ 0. The Phase 6 anime
    anchor adds VALUE for production cases where the dense face
    stream is unreliable (real anime detection has high miss
    rates, the fixture's synthetic data is clean), but on this
    fixture the anchor agrees with the speaker tracker so the
    metric is unchanged. The Phase 6 demonstration is in the
    unit tests + the AST guards on the Stage 7b wiring.
    """
    duration = 12.0
    fps = 10.0
    n = int(duration * fps)
    centers_pct = [30.0, 70.0]  # slot 0, slot 1
    face_w = 10.0
    half = face_w / 2.0
    regions: list[list[tuple[float, float]]] = []
    for i in range(n):
        t = i / fps
        slot_idx = int(t / 2.0) % 2
        cx = centers_pct[slot_idx]
        regions.append([(cx - half, cx + half)])
    return GroundTruth(
        expected_switches=[2.0, 4.0, 6.0, 8.0, 10.0],
        required_regions_per_frame=regions,
    )


def _gt_tps_character_offset() -> GroundTruth:
    # Character at x=40 % for the whole duration. HUD zones from
    # game_layouts.gta_v as a starting reference (Phase 7 will use
    # the actual lookup).
    duration = 10.0
    fps = 10.0
    n = int(duration * fps)
    char_w = 12.0
    char_x = 40.0
    char_region = [(char_x - char_w / 2, char_x + char_w / 2)]
    regions = [list(char_region) for _ in range(n)]
    hud_zones = [
        # GTA V HUD (% of source frame): minimap bottom-left,
        # weapon wheel bottom-right, health top-right.
        (0.0, 75.0, 18.0, 25.0),
        (80.0, 80.0, 20.0, 20.0),
        (70.0, 88.0, 28.0, 8.0),
    ]
    return GroundTruth(
        expected_switches=[],
        required_regions_per_frame=regions,
        hud_zones=hud_zones,
        game_key="gta_v",
    )


def _gt_stream_corner_facecam() -> GroundTruth:
    # Facecam region at x=82..88 (the corner face is ~6% wide).
    duration = 10.0
    fps = 10.0
    n = int(duration * fps)
    face_region = [(82.0, 88.0)]
    regions = [list(face_region) for _ in range(n)]
    # Generic FPS HUD as the "preserve me" zone for the gameplay
    # portion; Phase 7 will swap this for the configured game.
    hud_zones = [
        (70.0, 5.0, 28.0, 18.0),  # killfeed top-right (overlaps face region)
        (35.0, 88.0, 30.0, 10.0),  # health bar bottom-center
    ]
    return GroundTruth(
        expected_switches=[],
        required_regions_per_frame=regions,
        hud_zones=hud_zones,
        game_key="generic_fps",
    )


# ── Registry ────────────────────────────────────────────────────────

FIXTURES: dict[str, FixtureSpec] = {
    "2speaker_alternating": FixtureSpec(
        name="2speaker_alternating",
        description="Two speakers alternating every 2s for 20s (legacy fixture)",
        content_type_override="podcast",
        video_duration=20.0,
        metrics=[
            "sub_second_switch_recall",
            "overlap_count",
            "max_acceleration",
            "max_jerk",
            "required_region_miss_rate",
        ],
        ground_truth=_gt_2speaker_alternating(),
        build=_build_2speaker_alternating,
    ),
    "3speaker_panel": FixtureSpec(
        name="3speaker_panel",
        description="3-seat panel rotating every 1.5s for 18s — debate routing",
        content_type_override="debate",
        video_duration=18.0,
        metrics=[
            "sub_second_switch_recall",
            "overlap_count",
            "max_acceleration",
            "max_jerk",
            "required_region_miss_rate",
        ],
        ground_truth=_gt_3speaker_panel(),
        build=_build_3speaker_panel,
    ),
    "vlog_walk_and_talk": FixtureSpec(
        name="vlog_walk_and_talk",
        description="Single subject walks 35→65 % over 10s — vlog + thirds bias",
        content_type_override="vlog",
        video_duration=10.0,
        metrics=[
            "overlap_count",
            "max_acceleration",
            "max_jerk",
            "required_region_miss_rate",
            "face_centroid_in_thirds_rate",
        ],
        ground_truth=_gt_vlog_walk_and_talk(),
        build=_build_vlog_walk_and_talk,
    ),
    "music_video_beat": FixtureSpec(
        name="music_video_beat",
        description="120 BPM, 4 alternating speakers across 12s — beat snap",
        content_type_override="music_video",
        music_subtype="performance",
        video_duration=12.0,
        metrics=[
            "sub_second_switch_recall",
            "overlap_count",
            "downbeat_snap_error",
            "max_acceleration",
            "max_jerk",
        ],
        ground_truth=_gt_music_video_beat(),
        build=_build_music_video_beat,
    ),
    "anime_hard_cuts": FixtureSpec(
        name="anime_hard_cuts",
        description="6 hard cuts × 2s, alternating characters — anime + sub bar",
        content_type_override="anime",
        anime_subtype="dialogue",
        video_duration=12.0,
        metrics=[
            "sub_second_switch_recall",
            "overlap_count",
            "max_acceleration",
            "max_jerk",
            "required_region_miss_rate",
        ],
        ground_truth=_gt_anime_hard_cuts(),
        build=_build_anime_hard_cuts,
    ),
    "anime_panning_close_up": FixtureSpec(
        name="anime_panning_close_up",
        description=(
            "Anime close-up where the face pans 35→75 % over 8 s — "
            "exercises the Phase 5 intra-shot tracking override + "
            "Phase 4 per-frame face anchors"
        ),
        content_type_override="anime",
        anime_subtype="dialogue",
        video_duration=8.0,
        metrics=[
            "max_acceleration",
            "max_jerk",
            "required_region_miss_rate",
        ],
        ground_truth=_gt_anime_panning_close_up(),
        build=_build_anime_panning_close_up,
    ),
    "tps_character_offset": FixtureSpec(
        name="tps_character_offset",
        description="TPS character at x=40 %, HUD in corners — gameplay_tps",
        content_type_override="gameplay_tps",
        game_type="gta_v",
        video_duration=10.0,
        metrics=[
            "overlap_count",
            "max_acceleration",
            "max_jerk",
            "required_region_miss_rate",
            "hud_preservation_rate",
        ],
        ground_truth=_gt_tps_character_offset(),
        build=_build_tps_character_offset,
    ),
    "stream_corner_facecam": FixtureSpec(
        name="stream_corner_facecam",
        description="Facecam at x=85 corner + center gameplay — stream",
        content_type_override="stream",
        game_type="generic_fps",
        video_duration=10.0,
        metrics=[
            "overlap_count",
            "max_acceleration",
            "max_jerk",
            "required_region_miss_rate",
            "hud_preservation_rate",
        ],
        ground_truth=_gt_stream_corner_facecam(),
        build=_build_stream_corner_facecam,
    ),
}


def list_fixture_names() -> list[str]:
    return list(FIXTURES.keys())


def get_fixture(name: str) -> FixtureSpec:
    if name not in FIXTURES:
        raise KeyError(
            f"Unknown fixture {name!r}. Available: {', '.join(sorted(FIXTURES))}"
        )
    return FIXTURES[name]
