"""Layout confidence scoring — Blueprint v2 Phase 2.

Given a scene's Stage-3 signals (faces, active speaker timeline, HUD /
screen / webcam flags, saliency dispersion, object detections) and the
Stage-3 importance matrix row for the scene's content type, score each
candidate layout and return ``ConfidenceResult``:

    confidence = (top_score - second_score) / top_score

A value of 0.6 means the top candidate beats the runner-up by 60%+ —
the deterministic decision is safe. Below the threshold the layout
engine escalates to a VLM call (see ``layout_vlm_fallback.py``).

The scorer is pure / synchronous and deterministic — no network, no
frame extraction. It can be run in production alongside the existing
layout engine for diagnostics without any risk.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from backend.services.reframe_config import ReframeConfig, get_default_config


# Canonical set of layout ids the scorer + VLM fallback recognize.
# Kept in sync with ``backend.models.LayoutMode`` / the FFmpeg filter
# builder, plus ``ken_burns`` + ``object_tracker`` which don't live on
# the ``LayoutMode`` enum yet.
VALID_LAYOUTS: frozenset = frozenset({
    "single", "split", "triple", "pip",
    "screenshare", "gameplay",
    "ken_burns", "object_tracker",
})


@dataclass
class LayoutCandidate:
    layout: str
    score: float
    rationale: str = ""

    def to_dict(self) -> dict:
        return {
            "layout": str(self.layout),
            "score": float(self.score),
            "rationale": str(self.rationale),
        }


@dataclass
class ConfidenceResult:
    top: LayoutCandidate
    second: Optional[LayoutCandidate]
    confidence: float
    candidates: list = field(default_factory=list)

    def is_confident(self, threshold: float) -> bool:
        return self.confidence >= threshold

    def to_dict(self) -> dict:
        return {
            "top": self.top.to_dict() if self.top else None,
            "second": self.second.to_dict() if self.second else None,
            "confidence": float(self.confidence),
            "candidates": [c.to_dict() for c in self.candidates],
        }


def score_layout_candidates(
    *,
    scene_start: float,
    scene_end: float,
    dense_faces: list,
    active_speaker_events: list,
    has_hud: bool = False,
    has_screen_region: bool = False,
    has_webcam_overlay: bool = False,
    saliency_centroid_dispersion: float = 0.0,
    ball_detections: list = None,
    content_type: str = "generic",
    config: Optional[ReframeConfig] = None,
) -> ConfidenceResult:
    """Score each candidate layout for a scene and rank them.

    Inputs are intentionally primitive (counts, booleans, floats) so
    the scorer has no imports into the rest of the reframing stack
    beyond ``ReframeConfig``. The caller reduces FrameFaces /
    SpeakerEvent / ObjectDetection into these primitives.
    """
    config = (config or get_default_config()).for_content(content_type)
    w = config.importance
    duration = max(0.0, float(scene_end) - float(scene_start))

    face_count = _count_persistent_faces(dense_faces, scene_start, scene_end)
    turns_per_sec = _count_speaker_turns_per_sec(
        active_speaker_events, scene_start, scene_end,
    )
    ball_dets = ball_detections or []

    candidates: list[LayoutCandidate] = []

    # ── SINGLE — always a valid fallback.
    # ``0.5`` baseline makes SINGLE the default winner on anything
    # with a face but no better candidate. The ``1.5 * w.face`` lift
    # fires on face-bearing scenes; the ``0.3`` no-face floor lets
    # Ken Burns / object tracker win cleanly on subject-less scenes
    # without SINGLE's saliency term dominating.
    if face_count >= 1:
        single_score = 0.5 + 1.5 * w.face + 0.5 * w.saliency
    else:
        single_score = 0.3
    candidates.append(LayoutCandidate(
        "single", single_score,
        f"face_count={face_count}",
    ))

    # ── SPLIT — two persistent speakers with rapid turn exchange.
    # 2.5 * face beats SINGLE's ``1.5 * face`` lift so rapid-turn
    # podcasts flip to SPLIT instead of SINGLE on the dominant face.
    split_score = 0.0
    if face_count >= 2 and turns_per_sec > 0.5:
        split_score = 2.5 * w.face + 0.3 * w.saliency
    candidates.append(LayoutCandidate(
        "split", split_score,
        f"face_count={face_count},turns/s={turns_per_sec:.2f}",
    ))

    # ── TRIPLE — three persistent speakers with moderate turn rate.
    triple_score = 0.0
    if face_count >= 3 and turns_per_sec > 0.3:
        # Slightly below split because the 3-up layout is visually heavier.
        triple_score = 2.2 * w.face
    candidates.append(LayoutCandidate(
        "triple", triple_score,
        f"face_count={face_count},turns/s={turns_per_sec:.2f}",
    ))

    # ── PIP — secondary speaker overlay on primary speaker.
    # Narrower than SPLIT: two speakers but one dominates the
    # timeline (moderate turn rate, not rapid).
    pip_score = 0.0
    if face_count == 2 and 0.05 < turns_per_sec <= 0.5:
        pip_score = 2.0 * w.face + 0.2 * w.saliency
    candidates.append(LayoutCandidate(
        "pip", pip_score,
        f"face_count={face_count},turns/s={turns_per_sec:.2f}",
    ))

    # ── SCREENSHARE — top screen, bottom speaker.
    screenshare_score = 0.0
    if has_screen_region and face_count >= 1:
        screenshare_score = 1.3 * w.object + 0.8 * w.face + 0.4
    candidates.append(LayoutCandidate(
        "screenshare", screenshare_score,
        f"screen={has_screen_region}",
    ))

    # ── GAMEPLAY — 70% game / 30% webcam.
    gameplay_score = 0.0
    if has_hud and has_webcam_overlay:
        gameplay_score = 2.0 * w.object + 0.5 * w.face + 0.2
    candidates.append(LayoutCandidate(
        "gameplay", gameplay_score,
        f"hud={has_hud},webcam={has_webcam_overlay}",
    ))

    # ── KEN_BURNS — subject-less scene with diffuse saliency.
    # Only fires on content types where the layout_kenburns module
    # would actually emit a push. Other genres stay on SINGLE.
    kenburns_score = 0.0
    from backend.services.layout_kenburns import _content_type_allows_ken_burns
    if (face_count == 0
            and duration >= config.ken_burns_min_duration_sec
            and _content_type_allows_ken_burns(content_type)):
        kenburns_score = 0.8 * w.saliency + 0.3 * w.depth
        kenburns_score += float(saliency_centroid_dispersion) * 0.3
    candidates.append(LayoutCandidate(
        "ken_burns", kenburns_score,
        f"no_faces={face_count == 0},disp={saliency_centroid_dispersion:.2f}",
    ))

    # ── OBJECT_TRACKER — ball / car / tracked non-face subject.
    obj_score = 0.0
    if ball_dets:
        obj_score = 1.5 * w.object + 0.5 * w.motion + 0.3
    candidates.append(LayoutCandidate(
        "object_tracker", obj_score,
        f"ball_dets={len(ball_dets)}",
    ))

    candidates.sort(key=lambda c: c.score, reverse=True)
    top = candidates[0]
    second = candidates[1] if len(candidates) > 1 else None

    if top.score <= 0 or second is None or second.score <= 0:
        confidence = 1.0
    else:
        confidence = (top.score - second.score) / top.score

    return ConfidenceResult(
        top=top, second=second,
        confidence=float(max(0.0, min(1.0, confidence))),
        candidates=candidates,
    )


# ── Reducers from raw pipeline outputs → scorer inputs ───────────


def _extract_id(obj) -> Optional[int]:
    """Pull identity_id / slot_id from a face-like object, preserving 0."""
    for attr in ("identity_id", "slot_id"):
        if hasattr(obj, attr):
            v = getattr(obj, attr)
            if v is None:
                continue
            try:
                return int(v)
            except (TypeError, ValueError):
                continue
    return None


def _count_persistent_faces(dense_faces, start: float, end: float) -> int:
    """Number of distinct identity_ids present in >= 60% of scene samples.

    Accepts both:
      * ``FrameFaces`` objects (``.timestamp`` + ``.faces`` list with
        per-face ``identity_id``), and
      * bare face-like objects that already have ``timestamp`` +
        ``identity_id``/``slot_id`` (used by the test fixtures).
    """
    if not dense_faces:
        return 0
    dur = float(end) - float(start)
    if dur <= 0:
        return 0

    total_samples = 0
    per_id_count: dict[int, int] = {}

    for item in dense_faces:
        t = float(getattr(item, "timestamp", 0.0))
        if not (start <= t <= end):
            continue
        total_samples += 1
        sub_faces = getattr(item, "faces", None)
        if sub_faces:
            seen_this_frame: set[int] = set()
            for face in sub_faces:
                fid = _extract_id(face)
                if fid is None or fid < 0 or fid in seen_this_frame:
                    continue
                seen_this_frame.add(fid)
                per_id_count[fid] = per_id_count.get(fid, 0) + 1
        else:
            # Flat face-like object with its own identity_id / slot_id.
            fid = _extract_id(item)
            if fid is not None and fid >= 0:
                per_id_count[fid] = per_id_count.get(fid, 0) + 1

    if total_samples == 0:
        return 0
    threshold = total_samples * 0.6
    return sum(1 for count in per_id_count.values() if count >= threshold)


def _count_speaker_turns_per_sec(events, start: float, end: float) -> float:
    """Turns per second inside ``[start, end]``. A 'turn' is a change in
    ``slot_id`` between adjacent SpeakerEvents that fall inside the
    window."""
    dur = float(end) - float(start)
    if dur <= 0 or not events:
        return 0.0
    in_window = [
        ev for ev in events
        if start <= float(getattr(ev, "start", 0.0)) <= end
    ]
    if len(in_window) < 2:
        return 0.0
    in_window.sort(key=lambda e: float(getattr(e, "start", 0.0)))
    prev_slot = int(getattr(in_window[0], "slot_id", -1))
    turns = 0
    for ev in in_window[1:]:
        slot = int(getattr(ev, "slot_id", -1))
        if slot != prev_slot:
            turns += 1
        prev_slot = slot
    return turns / dur
