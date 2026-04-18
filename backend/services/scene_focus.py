"""Scene focus aggregator: collects required/non-required features per shot
and computes the minimum bounding rect and optimal crop center."""

import logging
from typing import Optional

from backend.services.focus_model import (
    FeatureKind,
    RequiredFeature,
    SceneFocusRegion,
)

logger = logging.getLogger(__name__)


def aggregate_scene_focus(
    shot_start: float,
    shot_end: float,
    dense_faces: list,
    saliency_keyframes: list,
    persistent_regions,
    face_registry,
    source_width: int,
    source_height: int,
    target_aspect: float = 9 / 16,
    saliency_regions: list = None,
    object_detections: list = None,
    subject_tracks: list = None,
    job_id: str = "",
    interpolated_timeline=None,
    frame_saliency: list = None,
) -> SceneFocusRegion:
    """Collect all required and non-required features in [shot_start, shot_end],
    compute the minimum bounding rect of required features, and decide whether
    that rect fits inside a target_aspect window within the source frame.
    """
    required = []
    optional = []

    # 1. Collect required features from dense faces
    #    Use half-open interval [shot_start, shot_end) so boundary frames
    #    belong to the next shot, not both.
    #    Skip non-human faces (figurines, posters, etc.) when verification is available.
    if dense_faces:
        for df in dense_faces:
            if df.timestamp < shot_start or df.timestamp >= shot_end:
                continue
            for f in df.faces:
                if not getattr(f, 'is_human', True):
                    continue
                sid = getattr(f, 'identity_id', -1)
                face_x = getattr(f, 'nose_x', getattr(f, 'x', 50))
                face_y = getattr(f, 'nose_y', getattr(f, 'y', 50))
                face_w = getattr(f, 'width', 10)
                face_h = getattr(f, 'height', face_w * 1.3)
                required.append(RequiredFeature(
                    t_start=df.timestamp,
                    t_end=df.timestamp,
                    x=float(face_x),
                    y=float(face_y),
                    w=float(face_w),
                    h=float(face_h),
                    kind=FeatureKind.FACE,
                    weight=1.0,
                    must_be_in_frame=True,
                    identity=int(sid) if sid >= 0 else None,
                ))

    # 1b. Supplement with interpolated timeline (denser per-frame positions)
    #     Only add non-anchor frames from the timeline — anchor frames already
    #     came from dense_faces above. This fills the gaps with tracker data.
    if interpolated_timeline is not None:
        for s in interpolated_timeline.samples:
            if s.timestamp < shot_start or s.timestamp >= shot_end:
                continue
            if s.is_anchor:
                continue  # already covered by dense_faces
            for slot_id, (cx, cy, w, h) in s.bboxes.items():
                conf = s.confidences.get(slot_id, 0.5)
                required.append(RequiredFeature(
                    t_start=s.timestamp,
                    t_end=s.timestamp,
                    x=float(cx),
                    y=float(cy),
                    w=float(w),
                    h=float(h),
                    kind=FeatureKind.FACE,
                    weight=float(conf),
                    must_be_in_frame=True,
                    identity=int(slot_id) if slot_id >= 0 else None,
                ))

    # 2. Collect persistent regions
    if persistent_regions:
        if getattr(persistent_regions, 'has_facecam', False) and persistent_regions.facecam_region:
            r = persistent_regions.facecam_region
            required.append(RequiredFeature(
                t_start=shot_start,
                t_end=shot_end,
                x=float((r.x + r.w / 2) * 100),
                y=float((r.y + r.h / 2) * 100),
                w=float(r.w * 100),
                h=float(r.h * 100),
                kind=FeatureKind.FACE,
                weight=1.0,
                must_be_in_frame=True,
            ))
        if getattr(persistent_regions, 'has_hud', False):
            for hr in getattr(persistent_regions, 'hud_regions', []):
                required.append(RequiredFeature(
                    t_start=shot_start,
                    t_end=shot_end,
                    x=float((hr.x + hr.w / 2) * 100),
                    y=float((hr.y + hr.h / 2) * 100),
                    w=float(hr.w * 100),
                    h=float(hr.h * 100),
                    kind=FeatureKind.HUD,
                    weight=0.9,
                    must_be_in_frame=True,
                    excluded_from_centroid=True,
                ))
        # Generic persistent regions as non-required
        for reg in getattr(persistent_regions, 'regions', []):
            if reg.region_type not in ('facecam', 'hud', 'scoreboard', 'lower_third'):
                optional.append(RequiredFeature(
                    t_start=shot_start,
                    t_end=shot_end,
                    x=float((reg.x + reg.w / 2) * 100),
                    y=float((reg.y + reg.h / 2) * 100),
                    w=float(reg.w * 100),
                    h=float(reg.h * 100),
                    kind=FeatureKind.OBJECT,
                    weight=float(reg.confidence) * 0.5,
                    must_be_in_frame=False,
                ))

    # 3. Collect saliency as non-required features.
    #    Prefer frame_saliency (SaliencyRegion list with full bboxes) over
    #    legacy saliency_keyframes (x-only tuples with hardcoded y/w/h).
    _sal_from_frame_saliency = 0
    if frame_saliency:
        for sr in frame_saliency:
            t = sr.timestamp
            if t < shot_start or t >= shot_end:
                continue
            optional.append(RequiredFeature(
                t_start=float(t),
                t_end=float(t),
                x=float(sr.x),
                y=float(sr.y),
                w=float(sr.w),
                h=float(sr.h),
                kind=FeatureKind.SALIENCY,
                weight=float(sr.saliency_score),
                must_be_in_frame=False,
            ))
            _sal_from_frame_saliency += 1
    elif saliency_keyframes:
        # Legacy fallback: x-only tuples — hardcoded y/w/h (deprecated path)
        for entry in saliency_keyframes:
            if len(entry) < 3:
                continue
            t, x, confidence = entry[0], entry[1], entry[2]
            if t < shot_start or t >= shot_end:
                continue
            optional.append(RequiredFeature(
                t_start=float(t),
                t_end=float(t),
                x=float(x),
                y=50.0,  # saliency tracker only gives x
                w=10.0,
                h=10.0,
                kind=FeatureKind.SALIENCY,
                weight=float(confidence),
                must_be_in_frame=False,
            ))

    if _sal_from_frame_saliency > 0:
        logger.info("[SaliencyParity] scene_focus: %d saliency features from frame_saliency (full bbox)",
                    _sal_from_frame_saliency)

    # 3.5. Object detections as required/non-required features
    if object_detections:
        from backend.services.object_detector import get_class_priority
        from collections import defaultdict as _defaultdict

        by_class = _defaultdict(list)
        for od in object_detections:
            if od.timestamp < shot_start or od.timestamp >= shot_end:
                continue
            by_class[od.class_name].append(od)

        for class_name, dets in by_class.items():
            if len(dets) < 2:
                is_moving = False
            else:
                xs = [d.x for d in dets]
                is_moving = (max(xs) - min(xs)) > 5.0

            must_be_in_frame, weight = get_class_priority(class_name, is_moving)

            for od in dets:
                feature = RequiredFeature(
                    t_start=od.timestamp,
                    t_end=od.timestamp,
                    x=od.x, y=od.y, w=od.w, h=od.h,
                    kind=FeatureKind.OBJECT,
                    weight=weight * od.confidence,
                    must_be_in_frame=must_be_in_frame,
                )
                if must_be_in_frame:
                    required.append(feature)
                else:
                    optional.append(feature)

    # 3.6. Spatiotemporal saliency regions as non-required features
    if saliency_regions:
        for sr in saliency_regions:
            if sr.timestamp < shot_start or sr.timestamp >= shot_end:
                continue
            optional.append(RequiredFeature(
                t_start=sr.timestamp,
                t_end=sr.timestamp,
                x=sr.x, y=sr.y, w=sr.w, h=sr.h,
                kind=FeatureKind.SALIENCY,
                weight=sr.saliency_score,
                must_be_in_frame=False,
            ))

    # 3.7. Subject tracks (face_confirmed and face_like_promoted)
    if subject_tracks:
        for track in subject_tracks:
            # Only include tracks whose trajectory overlaps this shot
            if track.t_end < shot_start or track.t_start >= shot_end:
                continue

            if track.source == "face_like_promoted":
                kind = FeatureKind.FACE_LIKE
                must_be_in_frame = True
            elif track.source == "face_confirmed":
                kind = FeatureKind.FACE
                must_be_in_frame = True
            else:
                continue

            for (t, x, y, w, h) in track.bbox_trajectory:
                if t < shot_start or t >= shot_end:
                    continue
                feature = RequiredFeature(
                    t_start=float(t),
                    t_end=float(t),
                    x=float(x), y=float(y),
                    w=float(w), h=float(h),
                    kind=kind,
                    weight=float(track.confidence),
                    must_be_in_frame=must_be_in_frame,
                    identity=int(track.persistent_id) if track.persistent_id >= 0 else None,
                )
                required.append(feature)

    # 4. Compute min bounding rect over required features
    if required:
        min_left = min(rf.left for rf in required)
        max_right = max(rf.right for rf in required)
        min_top = min(rf.top for rf in required)
        max_bottom = max(rf.bottom for rf in required)
        rect_w = max_right - min_left
        rect_h = max_bottom - min_top
        rect_cx = (min_left + max_right) / 2
        rect_cy = (min_top + max_bottom) / 2
        min_bounding_rect = (rect_cx, rect_cy, rect_w, rect_h)
    else:
        min_bounding_rect = (50, 50, 0, 0)

    # 5. Check fit
    src_aspect = source_width / source_height if source_height > 0 else 16 / 9
    if target_aspect < src_aspect:
        crop_width_pct = (target_aspect / src_aspect) * 100
    else:
        crop_width_pct = 100.0

    if not required:
        fits_target_aspect = True
    else:
        fits_target_aspect = (min_bounding_rect[2] <= crop_width_pct and
                              min_bounding_rect[3] <= 100)

    # 6. Optimal crop center
    #    Text/HUD features marked excluded_from_centroid are constraints (must
    #    stay in frame) but do NOT pull the crop center toward them. The crop
    #    center is derived only from human subjects.
    centroid_features = [rf for rf in required if not rf.excluded_from_centroid]

    if fits_target_aspect and centroid_features:
        # Use centroid of non-excluded features for crop center
        total_w = sum(rf.weight for rf in centroid_features)
        if total_w > 0:
            cx = sum(rf.x * rf.weight for rf in centroid_features) / total_w
            cy = sum(rf.y * rf.weight for rf in centroid_features) / total_w
        else:
            cx, cy = 50, 50
        # Clamp so crop window stays within [0, 100]
        half_cw = crop_width_pct / 2
        cx = max(half_cw, min(100 - half_cw, cx))
        cy = max(0, min(100, cy))
        optimal_crop_center = (cx, cy)
    elif centroid_features:
        # Weighted centroid of centroid-eligible features
        total_w = sum(rf.weight for rf in centroid_features)
        if total_w > 0:
            cx = sum(rf.x * rf.weight for rf in centroid_features) / total_w
            cy = sum(rf.y * rf.weight for rf in centroid_features) / total_w
        else:
            cx, cy = 50, 50
        optimal_crop_center = (cx, cy)
    elif required:
        # Only excluded features (text/HUD only, no human subjects) → center
        optimal_crop_center = (50, 50)
    else:
        optimal_crop_center = (50, 50)

    # 7. Per-frame targets
    per_frame_target = []
    # Collect all timestamps that have required features
    timestamps_seen = set()

    if dense_faces:
        for df in dense_faces:
            if df.timestamp < shot_start or df.timestamp >= shot_end:
                continue
            if df.timestamp in timestamps_seen:
                continue
            timestamps_seen.add(df.timestamp)

            visible = [rf for rf in required if rf.must_be_in_frame and
                       not rf.excluded_from_centroid and
                       abs(rf.t_start - df.timestamp) < 0.01]
            if visible:
                total_w = sum(rf.weight for rf in visible)
                if total_w > 0:
                    tx = sum(rf.x * rf.weight for rf in visible) / total_w
                    ty = sum(rf.y * rf.weight for rf in visible) / total_w
                else:
                    tx, ty = optimal_crop_center
                per_frame_target.append((df.timestamp, tx, ty))

    # Also add per-frame targets from subject tracks (for faceless content)
    if subject_tracks:
        for track in subject_tracks:
            for (t, x, y, w, h) in track.bbox_trajectory:
                if t < shot_start or t >= shot_end:
                    continue
                if t in timestamps_seen:
                    continue
                timestamps_seen.add(t)
                visible = [rf for rf in required if rf.must_be_in_frame and
                           not rf.excluded_from_centroid and
                           abs(rf.t_start - t) < 0.01]
                if visible:
                    total_w = sum(rf.weight for rf in visible)
                    if total_w > 0:
                        tx = sum(rf.x * rf.weight for rf in visible) / total_w
                        ty = sum(rf.y * rf.weight for rf in visible) / total_w
                    else:
                        tx, ty = optimal_crop_center
                    per_frame_target.append((t, tx, ty))
        per_frame_target.sort(key=lambda pft: pft[0])

    return SceneFocusRegion(
        shot_start=shot_start,
        shot_end=shot_end,
        required=required,
        optional=optional,
        min_bounding_rect=min_bounding_rect,
        fits_target_aspect=fits_target_aspect,
        optimal_crop_center=optimal_crop_center,
        per_frame_target=per_frame_target,
    )
