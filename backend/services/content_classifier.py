"""Content type classifier for the reframe pipeline.

Determines whether a video is narrative (movie/TV), podcast, gaming, vlog,
sports, music_video, or anime based on lightweight signals: cut rate, face
distribution, face position stability, and scene description hints.

Runs early in the pipeline (after frame extraction + face detection) so
the ReframeSegmenter can apply content-specific editorial strategies.

Also provides ClipContentType + classify_clip() for the camera solver's
per-content-type tuning path.
"""

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from backend.services.content_type_config import ContentType

logger = logging.getLogger(__name__)

USE_CONTENT_AWARE_REFRAME = os.environ.get(
    "USE_CONTENT_AWARE_REFRAME", "true"
).lower() in ("true", "1", "yes")


# ── Camera-solver content types (maps from existing ContentType) ──

class ClipContentType(str, Enum):
    """Simplified content types for camera solver tuning."""
    TALKING_HEAD = "talking_head"      # debates, podcasts, interviews
    CINEMATIC_DIALOGUE = "cinematic_dialogue"   # narrative w/ detected dialogue shots
    # Fix 3: multi-seat panel / interview where 2-5 people sit in
    # fixed positions. Distinct from TALKING_HEAD (which is 1-2
    # close-up talking heads) — panels need shot-cut-aware stationary
    # crops on each seat, not a single held crop on the loudest.
    MULTI_SPEAKER_PANEL = "multi_speaker_panel"
    ANIMATION = "animation"            # anime, cartoons
    ANIMATION_DIALOGUE = "animation_dialogue"   # talking anime characters
    MUSIC_VIDEO = "music_video"
    # ── Phase 2: gameplay sub-categories ──
    # GAMEPLAY remains the FPS / hero-shooter default (the legacy
    # behavior). The new variants allow Phase 7's gameplay subject
    # tracker to use the right per-genre action-center anchor:
    #   - MOBA / top-down: center-anchor with wider safe-zone
    #   - TPS: character is offset down+right of frame center
    #   - RACING: car is in lower-third, crop should anchor low
    GAMEPLAY = "gameplay"              # FPS / hero shooter (legacy default)
    GAMEPLAY_MOBA = "gameplay_moba"    # MOBA / top-down (LoL, Dota 2)
    GAMEPLAY_TPS = "gameplay_tps"      # third-person action (GTA, Elden Ring)
    GAMEPLAY_RACING = "gameplay_racing"  # racing / driving
    # ── v4: sports sub-categories ──
    # SPORTS is the generic fallback (field sports, boxing, etc). The
    # sub-types drive content-specific solver params and object-tracking
    # rules: basketball follows the ball, racing follows the lead car,
    # and both preserve the scoreboard region via cy floor.
    SPORTS = "sports"
    SPORTS_BASKETBALL = "sports_basketball"
    SPORTS_RACING = "sports_racing"
    STREAM = "stream"                  # facecam + gameplay
    GENERIC = "generic"


# Map from existing ContentType to ClipContentType
# Note: NARRATIVE is dynamically promoted to CINEMATIC_DIALOGUE inside
# classify_content() when dialogue-shot signals are present.
_CONTENT_TYPE_MAP = {
    ContentType.PODCAST.value: ClipContentType.TALKING_HEAD,
    ContentType.VLOG.value: ClipContentType.TALKING_HEAD,
    ContentType.ANIME.value: ClipContentType.ANIMATION,
    ContentType.MUSIC_VIDEO.value: ClipContentType.MUSIC_VIDEO,
    ContentType.GAMING.value: ClipContentType.GAMEPLAY,
    ContentType.NARRATIVE.value: ClipContentType.GENERIC,
    ContentType.SPORTS.value: ClipContentType.SPORTS,
    ContentType.UNKNOWN.value: ClipContentType.GENERIC,
}


@dataclass
class ContentProfile:
    content_type: str = "unknown"
    confidence: float = 0.0
    signals: dict = field(default_factory=dict)
    has_hud: bool = False
    has_facecam: bool = False
    facecam_region: Optional[tuple] = None  # (x, y, w, h) as 0-1 fractions
    hud_regions: list = field(default_factory=list)
    motion_profile: str = "low"  # "static" | "low" | "medium" | "high" | "chaotic"
    cut_rate_per_minute: float = 0.0
    # Set to True when narrative content also has dialogue-shot signals
    # (≥2 face slots and ≥30s of speech in the first 120s). Drives routing
    # to ClipContentType.CINEMATIC_DIALOGUE in classify_clip().
    is_cinematic_dialogue: bool = False
    # Set to True when the content is animated (anime / cartoon). Drives
    # ClipContentType.ANIMATION_DIALOGUE routing (narrative + animated) and
    # skips per-face human verification downstream.
    is_animated: bool = False
    # Fix 3: set to True when face signals match a seated multi-speaker
    # panel (2-5 stable seats, high multi-face-frame rate, per-slot
    # x-stdev small). Drives ClipContentType.MULTI_SPEAKER_PANEL routing
    # in classify_clip — overrides talking_head / vlog / generic.
    is_multi_speaker_panel: bool = False
    # ── Phase 2 sub-type fields (populated by user override) ──
    # anime_subtype: "action" | "dialogue" | "slice_of_life" | None.
    # Phase 6 reads this to pick the anime shot detector + lead-room
    # multiplier. None = heuristic fallback.
    anime_subtype: Optional[str] = None
    # music_subtype: "performance" | "narrative" | "lyric" | None.
    # Phase 5 reads this to set beat-snap aggressiveness. None =
    # heuristic fallback.
    music_subtype: Optional[str] = None
    # gameplay_subtype: "fps" | "moba" | "tps" | "racing" | "stream" | None.
    # Drives ClipContentType.GAMEPLAY_* routing in classify_clip. Phase
    # 7 uses this to pick per-genre action centers.
    gameplay_subtype: Optional[str] = None
    # sports_subtype: "basketball" | "racing" | None. Drives
    # ClipContentType.SPORTS_* routing for object-tracking rules
    # (follow the ball / car) and per-sport solver params.
    sports_subtype: Optional[str] = None
    # game_type: free-form game key from the existing game sub-dropdown
    # (e.g. "valorant", "league_of_legends"). Plumbed end-to-end so
    # Phase 7 can look up GAME_HUD_LAYOUTS without re-reading the job.
    game_type: str = ""
    # Fix 3.1: normalized user override (e.g. "anime", "podcast") when
    # the upload dropdown sent a hint. Unlike the pre-3.1 behavior, the
    # hint is a +2.0 score bias — not a gate — so geometry can still
    # override a wrong guess. None when no hint was provided.
    user_hint: Optional[str] = None


def _infer_sports_subtype_from_objects(
    frame_objects,
    *,
    ball_ratio_threshold: float = 0.15,
    vehicle_ratio_threshold: float = 0.10,
    vehicle_min_area_pct: float = 5.0,
) -> Optional[tuple[str, float, float]]:
    """Return ``(subtype, ball_ratio, vehicle_ratio)`` or ``None``.

    Pure function. Accepts either shape the pipeline might pass:

    1. **Flat list** of ``ObjectDetection``-shaped records (one entry
       per detection, each with ``.timestamp``, ``.class_name``,
       ``.w``, ``.h`` attributes). This is what ``_object_detections``
       is at call time in ``pipeline.py``. The helper groups by
       rounded timestamp internally.

    2. **Per-frame list** where each element has an ``.objects``
       attribute listing the per-frame detections (future-proof for
       a FrameObjects shape).

    Decision rule:
      - basketball wins when ``ball_ratio >= 0.15`` AND > ``vehicle_ratio``
      - racing wins when ``vehicle_ratio >= 0.10`` AND > ``ball_ratio``
      - otherwise no promotion (``None``)

    Vehicle detections must additionally span ``>= 5%`` of frame area
    to count — background cars in a wide running shot are dropped.
    ``width`` and ``height`` are assumed to be in percent of frame
    (0-100, matching the rest of the codebase).
    """
    if not frame_objects:
        return None

    # Shape detection: if the first element has ``.objects``, treat
    # it as a per-frame list; otherwise group the flat list by
    # rounded timestamp.
    _first = frame_objects[0]
    if hasattr(_first, "objects"):
        grouped = [list(getattr(fo, "objects", []) or []) for fo in frame_objects]
    else:
        by_ts: dict = {}
        for det in frame_objects:
            ts = round(float(getattr(det, "timestamp", 0.0) or 0.0), 2)
            by_ts.setdefault(ts, []).append(det)
        grouped = list(by_ts.values())

    n_frames = len(grouped)
    if n_frames == 0:
        return None

    ball_frames = 0
    vehicle_frames = 0
    area_gate = float(vehicle_min_area_pct) * 100.0  # 5% × 100 = 500

    for objs in grouped:
        if any(getattr(o, "class_name", "") == "sports ball" for o in objs):
            ball_frames += 1
        for o in objs:
            cn = getattr(o, "class_name", "")
            if cn not in ("car", "truck", "motorcycle"):
                continue
            w = float(getattr(o, "w", 0.0) or 0.0)
            h = float(getattr(o, "h", 0.0) or 0.0)
            if (w * h) >= area_gate:
                vehicle_frames += 1
                break  # one vehicle hit per frame is enough

    ball_ratio = ball_frames / n_frames
    vehicle_ratio = vehicle_frames / n_frames

    if ball_ratio >= ball_ratio_threshold and ball_ratio > vehicle_ratio:
        return ("basketball", ball_ratio, vehicle_ratio)
    if vehicle_ratio >= vehicle_ratio_threshold and vehicle_ratio > ball_ratio:
        return ("racing", ball_ratio, vehicle_ratio)
    return None


def _infer_music_subtype_from_formation_and_beat(
    dense_faces,
    beat_grid,
    *,
    formation_ratio_threshold: float = 0.08,
    beat_confidence_threshold: float = 0.6,
) -> Optional[tuple[str, float, float]]:
    """Return ``(subtype, formation_ratio, beat_confidence)`` or ``None``.

    Pure function. ``dense_faces`` is a list of ``FrameFaces`` from
    the dense detector; the helper runs ``_is_formation_frame`` on
    each (imported lazily to avoid a circular import at module
    load time). ``beat_grid`` is the ``BeatGrid`` object from the
    beat detector — its ``.confidence`` drives the promotion.

    Decision rule:
      - "performance" when formation_ratio >= 0.08 AND beat_conf >= 0.6
      - otherwise no promotion (``None``)
    """
    if not dense_faces:
        return None
    beat_conf = 0.0
    if beat_grid is not None:
        # Prefer effective_confidence() which zeros out empty grids;
        # fall back to a raw `confidence` attribute for stub shapes
        # the tests might pass.
        _eff = getattr(beat_grid, "effective_confidence", None)
        if callable(_eff):
            beat_conf = float(_eff())
        else:
            beat_conf = float(getattr(beat_grid, "confidence", 0.0) or 0.0)

    try:
        from backend.services.reframe_segmenter import _is_formation_frame
    except ImportError:
        return None

    n_frames = len(dense_faces)
    if n_frames == 0:
        return None
    formation_ratio = sum(
        1 for fr in dense_faces if _is_formation_frame(fr)
    ) / n_frames

    if (
        formation_ratio >= formation_ratio_threshold
        and beat_conf >= beat_confidence_threshold
    ):
        return ("performance", formation_ratio, beat_conf)
    return None


def classify_content(
    shot_cuts: list[float],
    face_registry,
    dense_faces: list,
    scenes: list,
    video_duration: float,
    metadata: dict = None,
    job_id: str = "",
    transcript_segments=None,
) -> ContentProfile:
    """Classify content type from available signals.

    Args:
        shot_cuts: Scene-cut timestamps.
        face_registry: FaceRegistry with face slots.
        dense_faces: Dense face detection results.
        scenes: AI scene descriptions.
        video_duration: Total duration in seconds.
        metadata: Job metadata dict (may contain user hints).
        job_id: For logging.

    Returns:
        ContentProfile with classified content type and signals.
    """
    _log = lambda msg, *a: logger.info("[%s] ContentClassifier: " + msg, job_id, *a)
    profile = ContentProfile()
    scores = {ct.value: 0.0 for ct in ContentType if ct not in (ContentType.UNKNOWN,)}
    signals = {}

    if video_duration <= 0:
        profile.content_type = ContentType.UNKNOWN.value
        return profile

    # ── Fix 3.1: user "hint" via metadata (bias, NOT a gate) ──
    #
    # The upload UI sends one of a fixed set of tokens via
    # ``JobResult.content_type_override``, which the pipeline injects
    # into ``metadata['content_type_override']`` before this call. We
    # normalize through ``content_type_strings.normalize_ui_content_type``
    # so legacy / phase-2 / invalid tokens all route consistently (see
    # that module for the token list).
    #
    # Pre-3.1 behavior: a present hint short-circuited to
    # ``confidence=1.0`` and skipped heuristic scoring entirely — if
    # the user mis-guessed (or sent a stale selection from an older
    # session), the wrong solver config ran for the full clip.
    #
    # 3.1 behavior: the hint still populates the panel / animated /
    # sub-type fields (so the secondary dropdowns keep working) but
    # only adds ``+2.0`` to the matching score. Heuristic signals
    # continue to run. Geometry that disagrees by ``>=2.5`` wins.
    #
    # Also still accepts the legacy ``content_type`` and ``reframe_style``
    # metadata keys for callers that haven't been updated.
    normalized = None
    if metadata:
        from backend.services.content_type_strings import (
            normalize_anime_subtype,
            normalize_music_subtype,
            normalize_ui_content_type,
        )

        user_type = (
            metadata.get("content_type_override")
            or metadata.get("content_type")
            or metadata.get("reframe_style")
        )
        normalized = normalize_ui_content_type(user_type) if user_type else None
        if normalized is not None:
            profile.user_hint = normalized.content_type.value
            profile.is_multi_speaker_panel = normalized.is_multi_speaker_panel
            profile.is_animated = normalized.is_animated
            # Phase 2: populate sub-type fields from the secondary UI
            # dropdowns. The classifier only honors the sub-type when
            # it matches the parent (e.g. anime_subtype is ignored if
            # the user picked a non-anime parent type) — otherwise a
            # stale sub-type from a previous selection would leak.
            if normalized.is_animated:
                profile.anime_subtype = normalize_anime_subtype(
                    metadata.get("anime_subtype")
                )
            if normalized.content_type == ContentType.MUSIC_VIDEO:
                profile.music_subtype = normalize_music_subtype(
                    metadata.get("music_subtype")
                )
            # Gameplay subtype is encoded directly in the parent token
            # (gameplay_moba → "moba"); always trust the normalized
            # value rather than a separate metadata field.
            profile.gameplay_subtype = normalized.gameplay_subtype
            # v4: sports subtype — either encoded in the UI token
            # (sports_basketball → "basketball") or supplied as a
            # separate metadata field for the "sports" parent token.
            if normalized.content_type == ContentType.SPORTS:
                profile.sports_subtype = (
                    normalized.sports_subtype
                    or (metadata.get("sports_subtype") or "").strip().lower()
                    or None
                )
            profile.game_type = (metadata.get("game_type") or "").strip().lower()
            signals["user_hint"] = normalized.content_type.value
            signals["user_hint_raw"] = normalized.raw
            # Gaming / stream stays as a gate: the gaming pipeline is
            # out of scope for the 3.x overhaul and its per-genre
            # behavior must remain byte-for-byte identical when the
            # user picks a gameplay token.
            _is_gameplay_hint = (
                normalized.content_type == ContentType.GAMING
                or (normalized.gameplay_subtype is not None)
            )
            if _is_gameplay_hint:
                profile.content_type = normalized.content_type.value
                profile.confidence = 1.0
                profile.signals = signals
                _log(
                    "gameplay hint %r → %s (gate, conf=1.00)",
                    normalized.raw, normalized.content_type.value,
                )
                return profile
            # Degenerate-inputs escape hatch: if there's no heuristic
            # input at all (no cuts, no faces, no scenes), we have
            # nothing to classify against — trust the hint. This keeps
            # the override-plumbing tests green for empty fixtures
            # while still letting geometry dominate on real clips.
            _degenerate = (
                not shot_cuts
                and not dense_faces
                and not scenes
                and (face_registry is None or not getattr(
                    face_registry, "slots", None))
            )
            if _degenerate:
                profile.content_type = normalized.content_type.value
                profile.confidence = 1.0
                profile.signals = signals
                _log(
                    "hint %r with degenerate inputs → %s (conf=1.00)",
                    normalized.raw, normalized.content_type.value,
                )
                return profile
            # Non-gameplay, non-degenerate: apply as a +2.0 bias to
            # the heuristic score for the matching bucket. Geometry
            # that disagrees by >=2.5 still wins.
            hint_key = normalized.content_type.value
            if hint_key in scores:
                scores[hint_key] += 2.0
                signals["user_hint_bias"] = 2.0
            _log(
                "user hint %r → +2.0 on %s (not a gate; heuristics continue)",
                normalized.raw, hint_key,
            )

    # ── Signal 1: Cut rate ──
    cut_rate = len(shot_cuts) / (video_duration / 60.0) if video_duration > 0 else 0
    profile.cut_rate_per_minute = cut_rate
    signals["cut_rate"] = round(cut_rate, 1)

    if cut_rate >= 10:
        # v4: drop narrative bonus from 3.0 → 2.0 and raise the sports
        # bonus from 1.0 → 2.5. High-cut-rate sports clips were landing
        # in narrative by margin of 2.0.
        scores["narrative"] += 2.0
        scores["sports"] += 2.5
    elif 2 < cut_rate < 10:
        scores["vlog"] += 1.5
        scores["narrative"] += 1.0
        scores["sports"] += 1.0
    elif cut_rate <= 2:
        scores["podcast"] += 3.0
        scores["gaming"] += 2.0

    # ── Signal 2: Face count distribution ──
    if dense_faces:
        face_counts = [len(df.faces) for df in dense_faces if df.faces]
        if face_counts:
            avg_faces = sum(face_counts) / len(face_counts)
            signals["avg_faces"] = round(avg_faces, 1)
            zero_face_pct = sum(1 for df in dense_faces if not df.faces) / len(dense_faces)
            signals["zero_face_pct"] = round(zero_face_pct, 2)

            if avg_faces >= 2.0:
                scores["podcast"] += 2.5
                scores["narrative"] += 1.0
            elif avg_faces >= 1.0:
                scores["vlog"] += 1.5
                scores["narrative"] += 0.5
            if zero_face_pct > 0.5:
                scores["gaming"] += 2.0
                scores["sports"] += 1.5

    # ── Signal 3: Face position stability ──
    if face_registry and face_registry.slots:
        total_frames = max(1, face_registry.total_frames)
        dominant_slot = max(face_registry.slots, key=lambda s: s.frame_count)
        dominant_pct = dominant_slot.frame_count / total_frames
        avg_range = sum(s.x_max - s.x_min for s in face_registry.slots) / len(face_registry.slots)
        signals["face_slots"] = len(face_registry.slots)
        signals["dominant_slot_pct"] = round(dominant_pct, 2)
        signals["avg_slot_x_range"] = round(avg_range, 1)

        if len(face_registry.slots) >= 2 and dominant_pct < 0.8 and avg_range < 15:
            scores["podcast"] += 3.0
        elif len(face_registry.slots) == 1 and avg_range < 10:
            scores["vlog"] += 1.5
        elif face_registry.is_continuous_motion:
            scores["narrative"] += 1.0
            scores["vlog"] += 1.0
            signals["continuous_motion"] = True

    # ── Signal 4: Letterbox detection (from metadata or scenes) ──
    if metadata:
        resolution = metadata.get("resolution", "")
        if resolution:
            parts = resolution.split("x")
            if len(parts) == 2:
                w, h = int(parts[0]), int(parts[1])
                if w > 0 and h > 0:
                    ratio = w / h
                    if ratio >= 2.2:  # 2.35:1 or 2.39:1 cinematic
                        scores["narrative"] += 3.0
                        signals["letterbox"] = True

    # ── Signal 5: Scene description hints ──
    if scenes:
        desc_lower = " ".join(
            (getattr(s, "description", "") or "").lower() for s in scenes[:50]
        )
        narrative_keywords = ["character", "scene", "dialogue", "close-up", "wide shot",
                              "reaction", "emotion", "dramatic", "cinematic"]
        gaming_keywords = ["gameplay", "hud", "health", "ammo", "minimap", "crosshair",
                           "score", "kill", "game", "level"]
        sports_keywords = ["field", "court", "goal", "score", "referee", "player",
                           "stadium", "match", "team"]

        narr_hits = sum(1 for kw in narrative_keywords if kw in desc_lower)
        game_hits = sum(1 for kw in gaming_keywords if kw in desc_lower)
        sport_hits = sum(1 for kw in sports_keywords if kw in desc_lower)

        if narr_hits >= 3:
            scores["narrative"] += 2.0
            signals["scene_desc_narrative"] = narr_hits
        if game_hits >= 2:
            scores["gaming"] += 3.0
            signals["scene_desc_gaming"] = game_hits
        if sport_hits >= 2:
            scores["sports"] += 2.5
            signals["scene_desc_sports"] = sport_hits

    # ── Signal 6: Motion profile from dense face variance ──
    if dense_faces and len(dense_faces) >= 10:
        # Compute variance of face x-positions across frames
        x_positions = []
        for df in dense_faces:
            if df.faces:
                avg_x = sum(f.nose_x for f in df.faces) / len(df.faces)
                x_positions.append(avg_x)
        if len(x_positions) >= 5:
            mean_x = sum(x_positions) / len(x_positions)
            variance = sum((x - mean_x) ** 2 for x in x_positions) / len(x_positions)
            stdev = variance ** 0.5
            signals["face_x_stdev"] = round(stdev, 1)

            if stdev < 3:
                profile.motion_profile = "static"
                scores["podcast"] += 1.0
            elif stdev < 8:
                profile.motion_profile = "low"
            elif stdev < 15:
                profile.motion_profile = "medium"
                scores["vlog"] += 0.5
            else:
                profile.motion_profile = "high"
                scores["sports"] += 0.5

    # ── Signal 7: Music video detection ──
    # Very high cut rate + low speech fraction signals music video
    if cut_rate >= 40:
        scores["music_video"] = scores.get("music_video", 0) + 3.0
        signals["high_cut_rate_music"] = True
    elif cut_rate >= 20:
        scores["music_video"] = scores.get("music_video", 0) + 1.5

    # Scene description hints for music_video and anime
    if scenes:
        desc_lower_all = " ".join(
            (getattr(s, "description", "") or "").lower() for s in scenes[:50]
        )
        music_keywords = ["music", "concert", "performance", "stage", "dancing",
                          "singer", "band", "microphone", "audience"]
        anime_keywords = ["anime", "animation", "animated", "cartoon", "manga",
                          "subtitles", "japanese"]

        music_hits = sum(1 for kw in music_keywords if kw in desc_lower_all)
        anime_hits = sum(1 for kw in anime_keywords if kw in desc_lower_all)

        if music_hits >= 2:
            scores["music_video"] = scores.get("music_video", 0) + 3.0
            signals["scene_desc_music_video"] = music_hits
        # Fix 4: tighten the anime text-signal gate. The Verzuz panel
        # got anime=3.0 with just 2 scene keyword hits and high avg
        # faces, which three-way-tied podcast/vlog/anime at conf=0.26.
        # Require BOTH a stronger scene-description signal (≥5 hits)
        # AND a "few faces" signature (avg_faces < 1.5) before the
        # text scores high. Otherwise give a small bump (+1.0) that
        # can be overridden by geometry. Tracks which rule fired.
        if anime_hits >= 2:
            signals["scene_desc_anime"] = anime_hits
            _avg_faces_for_anime = signals.get("avg_faces", 0)
            if anime_hits >= 5 and _avg_faces_for_anime < 1.5:
                # Strong anime signal: large bump (+5) so it dominates
                # geometry-driven scores. Exempt from the tiebreaker
                # below so it's respected even against high vlog/
                # narrative counts.
                scores["anime"] = scores.get("anime", 0) + 5.0
                signals["scene_desc_anime_strong"] = True
            else:
                # Weak text signal: small bump only. Prevents the
                # three-way tie with podcast/vlog on panels where
                # scene descriptions incidentally mention "animation"
                # / "subtitles" / etc.
                scores["anime"] = scores.get("anime", 0) + 1.0
                signals["scene_desc_anime_weak"] = True

    # ── Signal 8: Sports motion signature ──
    # Low-face + wide face-x spread + fast cuts is the unambiguous
    # broadcast-sports fingerprint: cameras pan between plays, the face
    # ratio is dominated by crowd/wide shots, and the cuts are quick.
    # Without this the classifier can't distinguish sports from
    # narrative when the scene description keywords are weak.
    if (
        signals.get("zero_face_pct", 0) > 0.40
        and signals.get("face_x_stdev", 0) > 12
        and cut_rate >= 8
    ):
        scores["sports"] += 3.0
        signals["sports_motion_signature"] = True
        _log("sports motion signature fired")

    # ── Pick winner ──
    # Fix 4: geometry-wins-over-text tiebreaker.
    # When scores are near-tied (top two within 1.0 AND both include a
    # text-driven category like anime/music_video), defer to the
    # geometry-based categories (podcast, vlog, narrative, gaming,
    # sports) which rely on face slot / cut-rate / avg_faces signals.
    # Text hits are noisy — 2 anime keyword matches out of 50 scenes
    # shouldn't beat stable face-registry geometry.
    best_type = max(scores, key=scores.get)
    best_score = scores[best_type]
    _TEXT_DRIVEN_TYPES = {"anime", "music_video"}
    _GEOMETRY_DRIVEN_TYPES = {"podcast", "vlog", "narrative", "gaming", "sports"}
    # Strong-anime detection (≥5 keyword hits + avg_faces<1.5) is
    # trusted and exempt from the tiebreaker. Weak anime hits (the
    # 2-keyword-hit noise that burned the Verzuz run) still go through.
    _strong_anime = signals.get("scene_desc_anime_strong") is True
    if best_type in _TEXT_DRIVEN_TYPES and not _strong_anime:
        # Find the top geometry-driven score for comparison.
        geom_contenders = [
            (t, s) for t, s in scores.items()
            if t in _GEOMETRY_DRIVEN_TYPES
        ]
        geom_contenders.sort(key=lambda x: -x[1])
        if geom_contenders and (best_score - geom_contenders[0][1]) < 1.0:
            best_type = geom_contenders[0][0]
            best_score = geom_contenders[0][1]
            signals["geometry_over_text_tiebreak"] = True

    total_score = sum(scores.values())
    confidence = best_score / total_score if total_score > 0 else 0.0

    # Require minimum confidence; fall back to unknown
    if confidence < 0.25 or best_score < 2.0:
        profile.content_type = ContentType.UNKNOWN.value
        profile.confidence = confidence
    else:
        profile.content_type = best_type
        profile.confidence = min(1.0, confidence)

    # ── Sports subtype auto-promotion (Week 2 Part D) ──
    # When the classifier landed on "sports" but the user didn't
    # provide a subtype via the upload dropdown, infer the subtype
    # from COCO object frequency. Basketball clips have a "sports ball"
    # object in ≥15% of frames; racing clips have a "car" / "truck" /
    # "motorcycle" at ≥10% of frames with ≥5% frame area. These
    # thresholds are tuned to fire on a half-court basketball clip
    # and a broadcaster-frame F1 clip but NOT on a generic running-
    # through-a-street shot where background vehicles are incidental.
    if (
        profile.content_type == ContentType.SPORTS.value
        and not profile.sports_subtype
    ):
        try:
            _sub = _infer_sports_subtype_from_objects(
                metadata.get("frame_objects") if metadata else None
            )
            if _sub is not None:
                sub_name, ball_ratio, vehicle_ratio = _sub
                profile.sports_subtype = sub_name
                signals["sports_subtype_autoprom"] = sub_name
                signals["sports_ball_frame_ratio"] = round(ball_ratio, 3)
                signals["sports_vehicle_frame_ratio"] = round(vehicle_ratio, 3)
                _log(
                    "sports → %s auto-promoted (ball=%.2f, vehicle=%.2f)",
                    sub_name, ball_ratio, vehicle_ratio,
                )
        except Exception as e:
            logger.warning(
                "[%s] Sports subtype auto-promotion failed: %s",
                job_id, e,
            )

    # ── Music video subtype auto-promotion (Week 2 Part E) ──
    # User picked "music_video" but no subtype: auto-promote to
    # "performance" when formation density + beat confidence are
    # both high. Formation frames come from dense_faces via
    # _is_formation_frame; beat confidence comes from the BeatGrid
    # populated upstream when librosa beat-tracking succeeded. If
    # the beat grid isn't available (music_beat_grid not in metadata),
    # we default to "narrative"-equivalent (leave subtype empty).
    if (
        profile.content_type == ContentType.MUSIC_VIDEO.value
        and not profile.music_subtype
    ):
        try:
            _music_sub = _infer_music_subtype_from_formation_and_beat(
                dense_faces=dense_faces,
                beat_grid=(
                    metadata.get("music_beat_grid") if metadata else None
                ),
            )
            if _music_sub is not None:
                sub_name, formation_ratio, beat_conf = _music_sub
                profile.music_subtype = sub_name
                signals["music_subtype_autoprom"] = sub_name
                signals["formation_ratio"] = round(formation_ratio, 3)
                signals["beat_confidence"] = round(beat_conf, 3)
                _log(
                    "music_video → %s auto-promoted "
                    "(formation=%.2f, beat_conf=%.2f)",
                    sub_name, formation_ratio, beat_conf,
                )
        except Exception as e:
            logger.warning(
                "[%s] Music subtype auto-promotion failed: %s",
                job_id, e,
            )

    # ── Cinematic dialogue detection ──
    # Promote NARRATIVE → CINEMATIC_DIALOGUE when we have ≥2 face slots AND
    # ≥30s of cumulative speech in the first 120s. This is the signal for
    # dialogue-driven film/TV where the pipeline should treat the active
    # speaker as load-bearing and background motion as noise.
    is_cinematic_dialogue = False
    num_slots_for_dialogue = (
        len(face_registry.slots) if face_registry and face_registry.slots else 0
    )
    cumulative_speech_seconds = 0.0
    if transcript_segments:
        for seg in transcript_segments:
            seg_start = seg.start if hasattr(seg, 'start') else seg.get('start', 0)
            seg_end = seg.end if hasattr(seg, 'end') else seg.get('end', 0)
            if seg_start >= 120.0:
                break
            clip_end_t = min(seg_end, 120.0)
            if clip_end_t > seg_start:
                cumulative_speech_seconds += (clip_end_t - seg_start)
    if (profile.content_type == ContentType.NARRATIVE.value
            and num_slots_for_dialogue >= 2
            and cumulative_speech_seconds >= 30.0):
        is_cinematic_dialogue = True
        signals["cinematic_dialogue"] = True
        signals["cumulative_speech_first_120s"] = round(cumulative_speech_seconds, 1)

    # ── Animation detection ──
    # Set is_animated=True when:
    #   (a) face_detector.ANIME_MODE_DETECTED is already True (the
    #       human-verifier rejection rate exceeded 50% — the most
    #       reliable anime signal, computed at detection time), OR
    #   (b) ContentType.ANIME was voted by the score heuristics, OR
    #   (c) scene descriptions contain ≥3 anime keywords.
    #
    # Inheriting the face-detector signal prevents the regression where
    # AnimeMode fires in face_detector but the classifier emits
    # `unknown` (low confidence / no scene_desc_anime votes) and
    # downstream saliency downranking stays disabled.
    is_animated = False
    anime_mode_inherited = False
    try:
        from backend.services.face_detector import ANIME_MODE_DETECTED
        if ANIME_MODE_DETECTED:
            is_animated = True
            anime_mode_inherited = True
            signals["anime_mode_from_detector"] = True
    except Exception:
        pass
    if not is_animated and profile.content_type == ContentType.ANIME.value:
        is_animated = True
    elif not is_animated and signals.get("scene_desc_anime", 0) >= 3:
        is_animated = True
        signals["animated_by_scene_desc"] = True

    # When AnimeMode was inherited from the detector but the content
    # type heuristic bailed to UNKNOWN (low confidence, no keyword
    # votes), promote UNKNOWN → NARRATIVE so downstream mapping takes
    # the narrative branch and can land on ANIMATION_DIALOGUE in
    # classify_clip(). We don't touch other classifications.
    if (anime_mode_inherited
            and profile.content_type == ContentType.UNKNOWN.value):
        profile.content_type = ContentType.NARRATIVE.value
        if profile.confidence < 0.25:
            profile.confidence = 0.25
        signals["anime_promoted_unknown_to_narrative"] = True
        logger.info(
            "[%s] ContentClassifier: anime_mode_inherited=True → NARRATIVE "
            "(promoted from UNKNOWN for ANIMATION_DIALOGUE routing)",
            job_id,
        )

    # ── Fix 3: multi-speaker panel detection ──
    # A seated interview / panel has:
    #   - 2-5 face slots (after Fix 1 teleport-merge)
    #   - ≥1.5 average face count per populated frame
    #   - ≥25% of populated frames show 2+ faces simultaneously
    #   - low per-slot x-stdev (each seat stays put; small x_range)
    #   - no single slot dominates >60% of frames
    #
    # When all hit, emit is_multi_speaker_panel so classify_clip routes
    # to ClipContentType.MULTI_SPEAKER_PANEL instead of talking_head /
    # vlog / generic (the wrong types the old classifier produced for
    # the Verzuz run).
    is_multi_speaker_panel = False
    multi_face_frame_pct = 0.0
    if dense_faces:
        populated = [df for df in dense_faces if df.faces]
        if populated:
            multi_face_frame_pct = sum(
                1 for df in populated if len(df.faces) >= 2
            ) / len(populated)
            signals["multi_face_frame_pct"] = round(multi_face_frame_pct, 2)

    if (face_registry and face_registry.slots
            and 2 <= len(face_registry.slots) <= 5
            and signals.get("avg_faces", 0) >= 1.5
            and multi_face_frame_pct >= 0.25):
        # Per-slot x-range (avg_slot_x_range) must be small enough that
        # each "seat" is actually stable, and no single slot dominates
        # >60% of the dense frames.
        total_frames_seen = sum(s.frame_count for s in face_registry.slots)
        if total_frames_seen > 0:
            avg_range = (
                sum(s.x_max - s.x_min for s in face_registry.slots)
                / len(face_registry.slots)
            )
            dom_slot = max(face_registry.slots, key=lambda s: s.frame_count)
            dom_pct = dom_slot.frame_count / total_frames_seen
            # Fix 4: loosen avg_range from 15 → 20 per the
            # accuracy-fix spec. The Verzuz run had avg_slot_x_range
            # ≈ 14.7 which passed the old gate, but the spec accepts
            # up to 20 as a "seated panel" signature since on-stage
            # characters can rock in-place. Still require dom_pct <
            # 0.6 (no single slot dominates the frame).
            if avg_range < 20 and dom_pct < 0.6:
                is_multi_speaker_panel = True
                signals["multi_speaker_panel"] = True
                signals["panel_avg_slot_range"] = round(avg_range, 1)
                signals["panel_dominant_pct"] = round(dom_pct, 2)
                # Add a score bump for PODCAST so the primary classifier
                # doesn't land on VLOG/ANIME when the panel signal fires.
                # Use a larger bump (+4.0) than the old +2.0 so the
                # geometry signal can outweigh text-driven anime hits
                # on ambiguous scene descriptions.
                scores["podcast"] = scores.get("podcast", 0) + 4.0
                # Re-pick winner with the bump.
                best_type = max(scores, key=scores.get)
                best_score = scores[best_type]
                total_score = sum(scores.values())
                confidence = best_score / total_score if total_score > 0 else 0.0
                if confidence >= 0.25 and best_score >= 2.0:
                    profile.content_type = best_type
                    profile.confidence = min(1.0, confidence)

    profile.signals = signals
    profile.is_cinematic_dialogue = is_cinematic_dialogue
    # Fix 3.1: OR with the user hint so a hint-driven is_animated /
    # is_multi_speaker_panel survives the heuristic reassignment here.
    profile.is_animated = is_animated or profile.is_animated
    profile.is_multi_speaker_panel = (
        is_multi_speaker_panel or profile.is_multi_speaker_panel
    )
    _log(
        "%s (conf=%.2f, signals=%s, scores=%s)",
        profile.content_type, profile.confidence,
        ", ".join(f"{k}={v}" for k, v in signals.items()),
        ", ".join(f"{k}={v:.1f}" for k, v in sorted(scores.items(), key=lambda x: -x[1])),
    )

    return profile


# ─────────────── Camera-solver content classification ────────────────────

def classify_clip(
    content_profile: Optional[ContentProfile] = None,
    persistent_regions=None,
    frame_faces: list = None,
    shot_count: int = 0,
    duration: float = 0,
) -> ClipContentType:
    """Classify clip for camera solver tuning.

    Uses the existing ContentProfile if available, then applies additional
    heuristics for STREAM detection (HUD + corner facecam).

    Args:
        content_profile: From classify_content() — already populated in pipeline
        persistent_regions: From persistent_region_detector — has_facecam, has_hud
        frame_faces: Existing FrameFaces list for face ratio computation
        shot_count: Number of detected shots
        duration: Clip duration in seconds

    Returns:
        ClipContentType for solver parameter routing.
    """
    # Start from existing classification if available
    base_type = ClipContentType.GENERIC
    if content_profile:
        base_type = _CONTENT_TYPE_MAP.get(
            content_profile.content_type, ClipContentType.GENERIC
        )
        # ── Phase 2: gameplay sub-type routing ──
        # When the user picked a specific gameplay variant (or stream)
        # the normalizer encodes it on profile.gameplay_subtype. Route
        # to the matching ClipContentType.GAMEPLAY_* before the
        # MULTI_SPEAKER_PANEL / CINEMATIC_DIALOGUE branches so a
        # gameplay clip never accidentally gets talking-head tuning.
        gameplay_sub = getattr(content_profile, "gameplay_subtype", None)
        if gameplay_sub == "moba":
            base_type = ClipContentType.GAMEPLAY_MOBA
        elif gameplay_sub == "tps":
            base_type = ClipContentType.GAMEPLAY_TPS
        elif gameplay_sub == "racing":
            base_type = ClipContentType.GAMEPLAY_RACING
        elif gameplay_sub == "stream":
            base_type = ClipContentType.STREAM
        elif gameplay_sub == "fps":
            base_type = ClipContentType.GAMEPLAY

        # ── v4: sports sub-type routing ──
        # When the user picked a specific sports variant (basketball,
        # racing) the normalizer encodes it on profile.sports_subtype.
        # Route to the matching ClipContentType.SPORTS_* so the camera
        # solver + required_regions builder can apply per-sport object
        # tracking (ball, car, player).
        sports_sub = getattr(content_profile, "sports_subtype", None)
        if base_type == ClipContentType.SPORTS:
            if sports_sub == "basketball":
                base_type = ClipContentType.SPORTS_BASKETBALL
            elif sports_sub == "racing":
                base_type = ClipContentType.SPORTS_RACING
        # Fix 3: MULTI_SPEAKER_PANEL promotion takes precedence over
        # CINEMATIC_DIALOGUE and the animation branch below — a seated
        # panel is a panel regardless of whether the raw classifier
        # guessed podcast/narrative/vlog/generic.
        if getattr(content_profile, "is_multi_speaker_panel", False):
            base_type = ClipContentType.MULTI_SPEAKER_PANEL
            logger.info(
                "[ContentClassifier] multi_speaker_panel signal fired "
                "→ MULTI_SPEAKER_PANEL (from base=%s)",
                content_profile.content_type,
            )
        # Narrative → CINEMATIC_DIALOGUE promotion (only if not already
        # routed to MULTI_SPEAKER_PANEL).
        elif getattr(content_profile, "is_cinematic_dialogue", False):
            base_type = ClipContentType.CINEMATIC_DIALOGUE
        # Animated narrative → ANIMATION_DIALOGUE (talking anime characters).
        # Applies the same saliency downrank + attention-anchor fallback
        # rules as CINEMATIC_DIALOGUE since the framing conventions (don't
        # drift onto background motion, hold on faces, bridge across gaps)
        # apply identically. We route to ANIMATION_DIALOGUE whenever
        # is_animated=True EXCEPT for legitimately non-narrative modes
        # (GAMEPLAY, STREAM, MUSIC_VIDEO) where the animation signal is
        # incidental and those layouts have their own rules.
        if getattr(content_profile, "is_animated", False):
            # Phase 2: anime_subtype overrides the heuristic. Action
            # anime stays on ANIMATION (Phase 6 will tune snappier intent
            # + bigger lead-room there); dialogue / slice-of-life route
            # to ANIMATION_DIALOGUE so the speaker-following framing
            # rules apply.
            anime_sub = getattr(content_profile, "anime_subtype", None)
            if anime_sub == "action":
                base_type = ClipContentType.ANIMATION
                logger.info(
                    "[ContentClassifier] anime_subtype=action → ANIMATION",
                )
            elif anime_sub in ("dialogue", "slice_of_life"):
                base_type = ClipContentType.ANIMATION_DIALOGUE
                logger.info(
                    "[ContentClassifier] anime_subtype=%s → ANIMATION_DIALOGUE",
                    anime_sub,
                )
            elif base_type in (
                # No explicit subtype — fall back to the legacy heuristic:
                # promote talking-head / cinematic-dialogue / generic
                # bases to ANIMATION_DIALOGUE; leave ANIMATION (anime
                # parent type's default mapping), MULTI_SPEAKER_PANEL,
                # MUSIC_VIDEO, and GAMEPLAY_* alone.
                ClipContentType.GENERIC,
                ClipContentType.CINEMATIC_DIALOGUE,
                ClipContentType.TALKING_HEAD,
            ):
                base_type = ClipContentType.ANIMATION_DIALOGUE
                logger.info(
                    "[ContentClassifier] anime_mode_inherited=True → ANIMATION_DIALOGUE",
                )

    # STREAM override: gameplay HUD + corner facecam
    if persistent_regions:
        has_facecam = getattr(persistent_regions, 'has_facecam', False)
        has_hud = getattr(persistent_regions, 'has_hud', False)

        if has_facecam and base_type == ClipContentType.GAMEPLAY:
            base_type = ClipContentType.STREAM
        elif has_facecam and frame_faces:
            # Check if face is consistently in a corner (stream pattern)
            corner_ratio = _fraction_of_faces_in_corner(frame_faces)
            if corner_ratio > 0.7:
                base_type = ClipContentType.STREAM

    # Emit a stable, grep-friendly signal at the end of classify_clip so the
    # v3 verifier can confirm content_type=animation_dialogue (or whichever
    # ClipContentType won) actually landed at the classifier layer. The
    # `[ContentClassifier]` tag mirrors the [Layout+Solver] tag emitted at
    # the solver layer so both signals can be correlated by job_id.
    logger.info(
        "[ContentClassifier] classify_clip: content_type=%s (from base=%s)",
        base_type.value,
        content_profile.content_type if content_profile else "none",
    )
    return base_type


def _fraction_of_faces_in_corner(frame_faces: list) -> float:
    """What fraction of detected faces appear in a screen corner?

    Corner = x < 20% or x > 80%, AND y < 25% or y > 75%.
    Used to detect facecam overlays in streams.
    """
    total = 0
    corner = 0
    for ff in frame_faces:
        for face in ff.faces:
            total += 1
            x = getattr(face, 'nose_x', 50)
            y = getattr(face, 'nose_y', 50)
            if (x < 20 or x > 80) and (y < 25 or y > 75):
                corner += 1
    return corner / max(total, 1)
