"""Subject confidence estimator for the confidence-gated fallback ladder.

Produces a per-segment confidence score combining:
  - Face-in-window check (is a face inside the proposed crop?)
  - Speaker agreement (active speaker matches face position)
  - Dense-face stability (face position consistency across segment)
  - Transcript coverage (identified speaker covers this time)

Used by the segmenter to gate the fallback ladder:
  >= 0.70: STATIONARY at face slot
  >= 0.50: inherit last known confident position
  >= 0.30: BLUR_FILL
  <  0.30: WIDE_MASTER
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ── Confidence thresholds for the fallback ladder ──
CONFIDENCE_HIGH = 0.70        # stationary at face slot
CONFIDENCE_MEDIUM = 0.50      # inherit last known position
CONFIDENCE_LOW = 0.30         # blur_fill
# Below CONFIDENCE_LOW → wide_master
# Fix 5: raise the wide_master floor. Previously, confidence < 0.30
# produced wide_master. That's too eager — confidence often drops
# momentarily on occlusion / profile frames even though the subject
# never left the crop. Only segments below 0.15 should be candidates
# for wide_master.
CONFIDENCE_WIDE_MASTER_FLOOR = 0.15

# Fix 1: weighted face-in-crop gate. Faces whose identity_id matches
# a tracked slot count with weight 1.0; untracked faces (crowd /
# background / false positives) count with weight 0.2. Weighting is
# further multiplied by face area as a percent of the frame so a large
# close-up face beats five tiny crowd faces.
UNTRACKED_FACE_WEIGHT = 0.2
WEIGHTED_FACE_RATIO_THRESHOLD = 0.30

# ── Fallback preferences per content type ──
# What to do when confidence is between LOW and MEDIUM
FALLBACK_PREFERENCE = {
    "narrative":    "wide_master",    # honor DP's composition
    "podcast":      "blur_fill",      # show the couch, not a letterbox
    "gaming":       "blur_fill",      # preserve HUD
    "vlog":         "blur_fill",      # motion reads through blur
    "sports":       "blur_fill",      # motion reads through blur
    "music_video":  "blur_fill",      # motion reads through blur
    "anime":        "blur_fill",      # motion reads through blur
    "multi_speaker_panel": "blur_fill",  # same as podcast — show the seats
    "unknown":      "blur_fill",      # safe default
}

# Fix 3: ordered fallback cascade per content type. Each tier name is
# resolved by _SalientFallbackResolver below. The cascade is tried in
# order; the first tier that returns a valid center wins.
FALLBACK_CASCADE_BY_CONTENT = {
    "multi_speaker_panel": [
        "active_speaker_slot",
        "prev_crop_continuity",
        "saliency_peak",
        "face_centroid",
        "prev_anywhere",
        "hardcoded_center",
    ],
    "podcast": [
        "active_speaker_slot",
        "prev_crop_continuity",
        "saliency_peak",
        "face_centroid",
        "prev_anywhere",
        "hardcoded_center",
    ],
    "talking_head": [
        "face_centroid",
        "prev_crop_continuity",
        "saliency_peak",
        "prev_anywhere",
        "hardcoded_center",
    ],
    "vlog": [
        "face_centroid",
        "saliency_peak",
        "motion_proxy",
        "prev_crop_continuity",
        "prev_anywhere",
        "hardcoded_center",
    ],
    "anime": [
        "face_centroid",
        "saliency_peak",
        "motion_proxy",
        "prev_crop_continuity",
        "prev_anywhere",
        "hardcoded_center",
    ],
    "narrative": [
        "face_centroid",
        "saliency_peak",
        "motion_proxy",
        "prev_crop_continuity",
        "prev_anywhere",
        "hardcoded_center",
    ],
    "music_video": [
        "face_centroid",
        "motion_proxy",
        "saliency_peak",
        "prev_crop_continuity",
        "prev_anywhere",
        "hardcoded_center",
    ],
    "sports": [
        "motion_proxy",
        "saliency_peak",
        "face_centroid",
        "prev_crop_continuity",
        "prev_anywhere",
        "hardcoded_center",
    ],
    "gaming": [
        "persistent_hud_region",
        "motion_proxy",
        "saliency_peak",
        "prev_crop_continuity",
        "prev_anywhere",
        "hardcoded_center",
    ],
    "unknown": [
        "prev_crop_continuity",
        "saliency_peak",
        "face_centroid",
        "motion_proxy",
        "prev_anywhere",
        "hardcoded_center",
    ],
}


class SubjectConfidenceEstimator:
    """Estimates confidence that a candidate crop window contains the intended subject.

    Usage:
        estimator = SubjectConfidenceEstimator(face_registry, dense_faces,
                                                active_speaker_events,
                                                transcript_segments,
                                                speaker_to_slot,
                                                source_width, source_height)
        conf, reason = estimator.evaluate(seg_start, seg_end,
                                          candidate_slot, candidate_x,
                                          target_aspect)
    """

    def __init__(
        self,
        face_registry,
        dense_faces: list,
        active_speaker_events: list,
        transcript_segments: list,
        speaker_to_slot: dict,
        source_width: int,
        source_height: int,
        frame_saliency: Optional[list] = None,
        persistent_regions=None,
        diarization_segments: Optional[list] = None,
        cluster_to_slot: Optional[dict] = None,
        content_type: str = "unknown",
    ):
        self.face_registry = face_registry
        self.dense_faces = dense_faces or []
        self.active_speaker_events = active_speaker_events or []
        self.transcript_segments = transcript_segments or []
        self.speaker_to_slot = speaker_to_slot or {}
        self.source_width = source_width
        self.source_height = source_height
        # Fix 1 + 2: saliency regions used by the weighted face gate's
        # short-circuit inspection AND by _SalientFallbackResolver when
        # the primary face tier fails.
        self.frame_saliency = frame_saliency or []
        self.persistent_regions = persistent_regions
        # Build a set of tracked slot_ids once for O(1) membership checks
        # in _check_face_in_crop_window's weighted loop.
        self._tracked_slot_ids: set = set()
        if face_registry and getattr(face_registry, "slots", None):
            for s in face_registry.slots:
                self._tracked_slot_ids.add(int(s.slot_id))

        # ── Gap 5b: diarization as a second vote ──
        # When a diarization pass ran in the pipeline AND at least one
        # cluster mapped to a known face slot, the estimator treats the
        # audio-side vote as an independent Check 2b alongside the lip-
        # motion vote. The weight split is 0.15 lip + 0.10 diar (the
        # 0.20 legacy lip slice, split); Phase C swaps this for a
        # per-content-type table lookup.
        self.diarization_segments = diarization_segments or []
        self.cluster_to_slot = cluster_to_slot or {}
        self._has_diarization = bool(
            self.diarization_segments
            and self.cluster_to_slot
            and any(v >= 0 for v in self.cluster_to_slot.values())
        )
        self.content_type = content_type or "unknown"
        # Gap 5c: per-content-type vote weight split. Looks up the
        # ``ACTIVE_SPEAKER_VOTE_WEIGHTS`` table and caches the pair
        # for the lifetime of this estimator instance. Legacy mode
        # (no diarization) returns (0.20, 0.0), preserving
        # bit-identical behavior with the pre-Gap-5 estimator.
        from backend.services.content_type_config import get_vote_weights
        self._lip_weight, self._diar_weight = get_vote_weights(
            self.content_type, self._has_diarization,
        )

    def evaluate(
        self,
        seg_start: float,
        seg_end: float,
        candidate_slot: Optional[int],
        candidate_x: int,
        target_aspect_ratio: float = 9 / 16,
    ) -> tuple:
        """Evaluate confidence that the candidate crop contains the subject.

        Args:
            seg_start: Segment start time (seconds)
            seg_end: Segment end time (seconds)
            candidate_slot: Proposed face registry slot (None = wide/unknown)
            candidate_x: Proposed subject_x (0-100)
            target_aspect_ratio: Width/height ratio of output (e.g., 9/16)

        Returns:
            (confidence: float, reason: str)
            confidence is [0.0, 1.0]
            reason explains the primary factor
        """
        confidence = 0.0
        reasons = []

        # ── Check 1: Face-in-window ──
        # Fix 1: weighted gate (tracked-slot faces count 1.0, untracked
        # count 0.2, scaled by face area) + active-speaker short-circuit
        # (if the active speaker is known AND their face is inside the
        # crop rect, the gate passes unconditionally).
        face_in_crop, face_check_detail = self._check_face_in_crop_window(
            seg_start, seg_end, candidate_x, target_aspect_ratio,
            candidate_slot=candidate_slot,
        )
        if face_in_crop:
            confidence += 0.40
        else:
            # No face in crop window → hard cap confidence at 0.30
            reasons.append(f"no_face_in_crop_window ({face_check_detail})")
            return min(0.30, confidence), "; ".join(reasons) if reasons else "no_face_in_crop"

        # ── Check 2: Speaker agreement (lip-motion vote) ──
        # Does the active-speaker slot match the candidate slot?
        speaker_agrees = self._check_speaker_agreement(
            seg_start, seg_end, candidate_slot,
        )
        if speaker_agrees:
            confidence += self._lip_weight
        else:
            reasons.append("speaker_disagrees")

        # ── Check 2b: Diarization agreement (second independent vote) ──
        # Gap 5b: when diarization produced cluster→slot mappings,
        # treat the audio-side vote as an independent signal. Lip
        # + diar each carry half of the 0.25 "speaker agree" budget
        # (tunable per-content-type in Phase C).
        if self._has_diarization:
            diar_agrees = self._check_diarization_agreement(
                seg_start, seg_end, candidate_slot,
            )
            if diar_agrees:
                confidence += self._diar_weight
            else:
                reasons.append("diarization_disagrees")

        # ── Check 3: Dense-face stability ──
        # How stable is the face position across the segment?
        stability = self._check_face_stability(seg_start, seg_end, candidate_slot)
        confidence += 0.20 * stability
        if stability < 0.5:
            reasons.append(f"unstable_face(stability={stability:.2f})")

        # ── Check 4: Transcript coverage ──
        # Does a transcript segment with an identified speaker cover this time?
        transcript_covers = self._check_transcript_coverage(seg_start, seg_end)
        if transcript_covers:
            confidence += 0.20
        else:
            reasons.append("no_transcript_coverage")

        confidence = max(0.0, min(1.0, confidence))
        reason = "; ".join(reasons) if reasons else "all_checks_passed"
        return confidence, reason

    def evaluate_with_breakdown(
        self,
        seg_start: float,
        seg_end: float,
        candidate_slot: Optional[int],
        candidate_x: int,
        target_aspect_ratio: float = 9 / 16,
    ) -> tuple:
        """Like evaluate(), but also returns a breakdown dict of the four input
        components for debugging.

        Returns:
            (confidence: float, reason: str, breakdown: dict)
            breakdown keys: face_in_crop, speaker_agree, stability, transcript
        """
        breakdown = {
            "face_in_crop": 0.0,
            "speaker_agree": 0.0,
            "diarization_agree": 0.0,
            "stability": 0.0,
            "transcript": 0.0,
        }
        confidence = 0.0
        reasons = []

        # Check 1: Face-in-window (Fix 1: weighted + short-circuit)
        face_in_crop, face_check_detail = self._check_face_in_crop_window(
            seg_start, seg_end, candidate_x, target_aspect_ratio,
            candidate_slot=candidate_slot,
        )
        if face_in_crop:
            confidence += 0.40
            breakdown["face_in_crop"] = 0.40
        else:
            reasons.append(f"no_face_in_crop_window ({face_check_detail})")
            confidence = min(0.30, confidence)
            breakdown["face_in_crop"] = confidence
            reason = "; ".join(reasons) if reasons else "no_face_in_crop"
            return confidence, reason, breakdown

        # Check 2: Speaker agreement (lip-motion vote)
        speaker_agrees = self._check_speaker_agreement(
            seg_start, seg_end, candidate_slot,
        )
        if speaker_agrees:
            confidence += self._lip_weight
            breakdown["speaker_agree"] = self._lip_weight
        else:
            reasons.append("speaker_disagrees")

        # Check 2b: Diarization agreement (Gap 5b second vote)
        if self._has_diarization:
            diar_agrees = self._check_diarization_agreement(
                seg_start, seg_end, candidate_slot,
            )
            if diar_agrees:
                confidence += self._diar_weight
                breakdown["diarization_agree"] = self._diar_weight
            else:
                reasons.append("diarization_disagrees")

        # Check 3: Dense-face stability
        stability = self._check_face_stability(seg_start, seg_end, candidate_slot)
        stability_contrib = 0.20 * stability
        confidence += stability_contrib
        breakdown["stability"] = round(stability_contrib, 4)
        if stability < 0.5:
            reasons.append(f"unstable_face(stability={stability:.2f})")

        # Check 4: Transcript coverage
        transcript_covers = self._check_transcript_coverage(seg_start, seg_end)
        if transcript_covers:
            confidence += 0.20
            breakdown["transcript"] = 0.20
        else:
            reasons.append("no_transcript_coverage")

        confidence = max(0.0, min(1.0, confidence))
        reason = "; ".join(reasons) if reasons else "all_checks_passed"
        return confidence, reason, breakdown

    def _check_face_in_crop_window(
        self,
        seg_start: float,
        seg_end: float,
        candidate_x: int,
        target_aspect_ratio: float,
        candidate_slot: Optional[int] = None,
    ) -> tuple:
        """Check whether the proposed crop rect contains the subject.

        Fix 1: the old check counted every detected face equally and
        gated at ≥30% face count inside the crop. On a panel with a
        reactive audience, 12/62 faces landed in the crop and the gate
        failed — despite all 3 panelists' faces being squarely inside.
        The gate now:

          (a) short-circuits to True when `candidate_slot` is the
              active speaker AND that speaker's face is inside the
              crop rect — the segment IS correct no matter how many
              background faces spill outside.
          (b) computes a WEIGHTED ratio where a tracked-slot face
              counts 1.0 and an untracked face counts 0.2, with both
              scaled by face area as a fraction of the frame (area
              scale ∈ [0.25, 2.0]).
          (c) passes when the weighted ratio ≥ WEIGHTED_FACE_RATIO_THRESHOLD.

        Returns (bool, detail_string).
        """
        if not self.dense_faces:
            return False, "no_dense_faces"

        # Compute crop window bounds in 0-100 coordinate space
        src_aspect = self.source_width / self.source_height if self.source_height > 0 else 16 / 9
        if target_aspect_ratio < src_aspect:
            # Vertical crop: window width as fraction of source
            crop_width_pct = (target_aspect_ratio / src_aspect) * 100
        else:
            crop_width_pct = 100.0

        crop_left = candidate_x - crop_width_pct / 2.0
        crop_right = candidate_x + crop_width_pct / 2.0
        # Clamp
        if crop_left < 0:
            crop_left = 0
            crop_right = crop_width_pct
        if crop_right > 100:
            crop_right = 100
            crop_left = 100 - crop_width_pct

        # ── (a) Active-speaker short-circuit ──
        # Build the set of slot_ids that are the active speaker at ANY
        # point in this segment. If candidate_slot matches one of
        # those, we only need to confirm that speaker's face actually
        # sits inside the crop rect — and if it does, we pass the gate
        # without looking at background faces at all.
        active_slots_in_segment: set = set()
        if self.active_speaker_events:
            for ev in self.active_speaker_events:
                ov_s = max(seg_start, ev.start)
                ov_e = min(seg_end, ev.end)
                if ov_e > ov_s and getattr(ev, "slot_id", -1) >= 0:
                    active_slots_in_segment.add(int(ev.slot_id))

        active_face_inside = False
        if (candidate_slot is not None
                and int(candidate_slot) in active_slots_in_segment):
            for df in self.dense_faces:
                if df.timestamp < seg_start or df.timestamp > seg_end:
                    continue
                for f in df.faces:
                    if getattr(f, "identity_id", -1) != int(candidate_slot):
                        continue
                    fx = getattr(f, "x", None)
                    if fx is None:
                        fx = getattr(f, "nose_x", 50)
                    if crop_left <= fx <= crop_right:
                        active_face_inside = True
                        break
                if active_face_inside:
                    break
            if active_face_inside:
                return True, (
                    f"active_speaker_slot{candidate_slot}_in_crop "
                    f"[{crop_left:.0f}-{crop_right:.0f}] (short-circuit)"
                )

        # ── (b) + (c) Weighted ratio gate ──
        weight_in = 0.0
        weight_total = 0.0
        raw_in = 0
        raw_total = 0
        for df in self.dense_faces:
            if df.timestamp < seg_start or df.timestamp > seg_end:
                continue
            for f in df.faces:
                face_x = getattr(f, "x", None)
                if face_x is None:
                    face_x = getattr(f, "nose_x", 50)

                sid = getattr(f, "identity_id", -1)
                tracked = int(sid) in self._tracked_slot_ids if sid is not None else False

                fw = float(getattr(f, "width", 10.0))
                fh = float(getattr(f, "height", 10.0))
                # face area as percent of frame * 0.01; clamp to
                # [0.25, 2.0] so tiny crowd faces still count a little
                # and single huge faces don't completely dominate.
                area_scale = max(0.25, min(2.0, (fw * fh) / 100.0))
                base = 1.0 if tracked else UNTRACKED_FACE_WEIGHT
                is_human_conf = float(getattr(f, "pose_confidence", 1.0) or 1.0)
                w = base * area_scale * max(0.3, is_human_conf)

                weight_total += w
                raw_total += 1
                if crop_left <= face_x <= crop_right:
                    weight_in += w
                    raw_in += 1

        if weight_total <= 0:
            return False, "no_faces_in_interval"

        ratio = weight_in / weight_total
        if ratio >= WEIGHTED_FACE_RATIO_THRESHOLD:
            return True, (
                f"{raw_in}/{raw_total} faces in crop, "
                f"weighted={ratio:.2f} ≥ {WEIGHTED_FACE_RATIO_THRESHOLD:.2f}"
            )
        return False, (
            f"only {raw_in}/{raw_total} faces in crop, "
            f"weighted={ratio:.2f} < {WEIGHTED_FACE_RATIO_THRESHOLD:.2f} "
            f"(need {WEIGHTED_FACE_RATIO_THRESHOLD:.0%})"
        )

    def per_frame_slot_visible_rate(
        self,
        seg_start: float,
        seg_end: float,
        slot_id: Optional[int],
    ) -> float:
        """Fraction of dense face frames in ``[seg_start, seg_end]`` that
        contain at least one face whose ``identity_id`` matches ``slot_id``.

        Stricter than :meth:`per_frame_in_crop_pass_rate`: that helper
        accepts ANY tracked slot whose face center sits inside the
        candidate crop, which lets adjacent boundary slots satisfy the
        gate even when the chosen slot is entirely off-camera. This
        helper resolves the off-screen-speaker class of bug by checking
        the specific slot only.
        """
        if not self.dense_faces or slot_id is None:
            return 0.0
        n_frames = 0
        n_pass = 0
        for df in self.dense_faces:
            ts = getattr(df, "timestamp", 0.0)
            if ts < seg_start or ts > seg_end:
                continue
            n_frames += 1
            for f in df.faces:
                sid = getattr(f, "identity_id", -1)
                if sid is not None and int(sid) == int(slot_id):
                    n_pass += 1
                    break
        if n_frames == 0:
            return 0.0
        return n_pass / n_frames

    def per_frame_in_crop_pass_rate(
        self,
        seg_start: float,
        seg_end: float,
        candidate_x: int,
        target_aspect_ratio: float = 9 / 16,
    ) -> float:
        """Fraction of dense face frames in [seg_start, seg_end] where
        AT LEAST ONE tracked-slot face center is inside the proposed crop.

        Fix 5: the confidence estimator above aggregates across the
        whole window, which means one bad mid-segment frame can pull
        the gate below the threshold for an entire 5 second segment.
        This per-frame robustness helper lets the segmenter promote a
        segment back out of wide_master when ≥40 %% of frames had a
        tracked face visibly inside the crop rect — the dropout is
        momentary, not systematic.

        Returns a 0.0-1.0 ratio; 0.0 if no frames fall in the window.
        """
        if not self.dense_faces:
            return 0.0
        src_aspect = (
            self.source_width / self.source_height
            if self.source_height > 0 else 16 / 9
        )
        if target_aspect_ratio < src_aspect:
            crop_width_pct = (target_aspect_ratio / src_aspect) * 100
        else:
            crop_width_pct = 100.0
        crop_left = candidate_x - crop_width_pct / 2.0
        crop_right = candidate_x + crop_width_pct / 2.0
        if crop_left < 0:
            crop_left = 0
            crop_right = crop_width_pct
        if crop_right > 100:
            crop_right = 100
            crop_left = 100 - crop_width_pct

        n_frames = 0
        n_pass = 0
        for df in self.dense_faces:
            if df.timestamp < seg_start or df.timestamp > seg_end:
                continue
            n_frames += 1
            for f in df.faces:
                sid = getattr(f, "identity_id", -1)
                if sid is None or int(sid) not in self._tracked_slot_ids:
                    continue
                fx = getattr(f, "x", None)
                if fx is None:
                    fx = getattr(f, "nose_x", 50)
                if crop_left <= fx <= crop_right:
                    n_pass += 1
                    break
        if n_frames == 0:
            return 0.0
        return n_pass / n_frames

    def _check_speaker_agreement(
        self,
        seg_start: float,
        seg_end: float,
        candidate_slot: Optional[int],
    ) -> bool:
        """Check if the active speaker slot matches the candidate slot."""
        if candidate_slot is None:
            return False

        if not self.active_speaker_events:
            return False

        total_time = 0.0
        agree_time = 0.0
        for ev in self.active_speaker_events:
            overlap_start = max(seg_start, ev.start)
            overlap_end = min(seg_end, ev.end)
            if overlap_start >= overlap_end:
                continue
            dur = overlap_end - overlap_start
            total_time += dur
            if ev.slot_id == candidate_slot:
                agree_time += dur

        if total_time <= 0:
            return False
        return (agree_time / total_time) >= 0.5

    def _check_diarization_agreement(
        self,
        seg_start: float,
        seg_end: float,
        candidate_slot: Optional[int],
    ) -> bool:
        """Gap 5b — Check 2b: does the diarization audio-side vote
        agree with the candidate slot over the majority of the
        segment window?

        Returns True when ≥ 50% of the diarization time inside
        ``[seg_start, seg_end)`` resolves (via ``cluster_to_slot``)
        to ``candidate_slot``. Returns False when diarization is
        absent, the candidate slot is unknown, no diarization
        clusters overlap the segment, or the majority is a
        different slot.
        """
        if candidate_slot is None:
            return False
        if not self.diarization_segments or not self.cluster_to_slot:
            return False

        total_time = 0.0
        agree_time = 0.0
        for diar in self.diarization_segments:
            lo = max(seg_start, diar.start)
            hi = min(seg_end, diar.end)
            if lo >= hi:
                continue
            dur = hi - lo
            total_time += dur
            resolved = self.cluster_to_slot.get(diar.cluster_id, -1)
            if resolved == candidate_slot:
                agree_time += dur

        if total_time <= 0:
            return False
        return (agree_time / total_time) >= 0.5

    def _check_face_stability(
        self,
        seg_start: float,
        seg_end: float,
        candidate_slot: Optional[int],
    ) -> float:
        """Check how stable the face position is across the segment.

        Returns 0.0-1.0 where 1.0 = perfectly stable.
        """
        if candidate_slot is None:
            return 0.0

        positions = []
        for df in self.dense_faces:
            if df.timestamp < seg_start or df.timestamp > seg_end:
                continue
            for f in df.faces:
                if getattr(f, 'identity_id', -1) == candidate_slot:
                    fx = getattr(f, 'x', None)
                    if fx is None:
                        fx = getattr(f, 'nose_x', 50)
                    positions.append(fx)

        if len(positions) < 2:
            return 0.5  # Neutral: not enough data

        import numpy as np
        std = float(np.std(positions))
        # Normalize: std of 0 = perfect stability (1.0); std of 30+ = very unstable (0.0)
        stability = max(0.0, 1.0 - std / 30.0)
        return stability

    def _check_transcript_coverage(
        self,
        seg_start: float,
        seg_end: float,
    ) -> bool:
        """Check if a transcript segment with an identified speaker covers this time."""
        if not self.transcript_segments or not self.speaker_to_slot:
            return False

        seg_dur = seg_end - seg_start
        if seg_dur <= 0:
            return False

        covered_time = 0.0
        for seg in self.transcript_segments:
            s_start = getattr(seg, 'start', 0)
            s_end = getattr(seg, 'end', 0)
            overlap_start = max(seg_start, s_start)
            overlap_end = min(seg_end, s_end)
            if overlap_start >= overlap_end:
                continue
            speaker = getattr(seg, 'speaker', None) or ''
            if speaker in self.speaker_to_slot:
                covered_time += overlap_end - overlap_start

        return (covered_time / seg_dur) >= 0.3

    def check_required_features_fit(
        self,
        seg_start: float,
        seg_end: float,
        crop_center_x: float,
        crop_center_y: float,
        crop_width_pct: float,
        crop_height_pct: float,
        required_features: list,
    ) -> tuple:
        """Returns (all_fit, detail). `all_fit` is True ONLY when every required
        feature's full bounding box is inside the crop rectangle, not just the
        center point. This is the AutoFlip hard constraint check.

        Args:
            seg_start, seg_end: Segment time range.
            crop_center_x, crop_center_y: Crop center in 0-100 space.
            crop_width_pct, crop_height_pct: Crop dimensions in 0-100 space.
            required_features: list[RequiredFeature] from focus_model.

        Returns:
            (all_fit: bool, detail: str)
        """
        crop_left = crop_center_x - crop_width_pct / 2
        crop_right = crop_center_x + crop_width_pct / 2
        crop_top = crop_center_y - crop_height_pct / 2
        crop_bottom = crop_center_y + crop_height_pct / 2

        # Clamp crop to frame bounds
        if crop_left < 0:
            crop_right -= crop_left
            crop_left = 0
        if crop_right > 100:
            crop_left -= (crop_right - 100)
            crop_right = 100

        for rf in required_features:
            if not rf.must_be_in_frame:
                continue
            # Check time overlap
            if rf.t_end < seg_start or rf.t_start > seg_end:
                continue
            # Check full bounding box containment
            if (rf.left < crop_left or rf.right > crop_right or
                    rf.top < crop_top or rf.bottom > crop_bottom):
                kind_str = rf.kind.value if hasattr(rf.kind, 'value') else str(rf.kind)
                return (False, f"clipped:{kind_str}:{rf.identity}")

        return (True, "all_required_in_frame")


def face_in_proposed_crop(seg, face_registry, dense_faces,
                         source_width: int = 1920, source_height: int = 1080,
                         target_aspect: float = 9 / 16) -> bool:
    """Returns True if at least one face from the registry has its center
    inside the segment's proposed crop rectangle."""
    if not face_registry or not dense_faces:
        return False

    src_aspect = source_width / source_height if source_height > 0 else 16 / 9
    if target_aspect < src_aspect:
        crop_w_pct = (target_aspect / src_aspect) * 100
    else:
        crop_w_pct = 100.0

    # subject_x may be in pixel space (>100) or legacy 0-100 space
    sx = seg.subject_x
    if isinstance(sx, float) and sx > 100.0:
        sx = sx / source_width * 100.0  # convert to 0-100

    crop_x_min = sx - crop_w_pct / 2
    crop_x_max = sx + crop_w_pct / 2
    if crop_x_min < 0:
        crop_x_min = 0
        crop_x_max = crop_w_pct
    if crop_x_max > 100:
        crop_x_max = 100
        crop_x_min = 100 - crop_w_pct

    seg_mid = (seg.start + seg.end) / 2.0
    for df in dense_faces:
        if abs(df.timestamp - seg_mid) > 0.5:
            continue
        for f in df.faces:
            sid = getattr(f, 'identity_id', -1)
            if sid < 0:
                continue
            face_x = getattr(f, 'nose_x', getattr(f, 'x', 50))
            if crop_x_min <= face_x <= crop_x_max:
                return True
    return False


class _SalientFallbackResolver:
    """Fix 2: multi-tier fallback cascade for the low-confidence case.

    Replaces the old `hardcoded_center (x=50)` fallback with an
    ordered list of strategies. Each tier returns (x, y, source_tag)
    in 0-100 percent coordinates; the first tier that returns a valid
    center wins.

    Tiers:
      - prev_crop_continuity: previous segment's (x,y) if it was
        confident (>= 0.5) AND its end is within 2s of this segment's
        start. Speakers don't teleport — continuity beats centering.
      - saliency_peak: dominant saliency region in the segment window,
        gated on mean_score >= 0.40.
      - face_centroid: weighted centroid of ALL faces (tracked and
        untracked) in the window. Serves as a body-proxy when face
        detection succeeded but the tracked slot gate failed.
      - motion_proxy: frame-to-frame face-position deltas. When a
        face is moving, the mean position of moving faces is the
        motion centroid — good for sports / dance / anime action.
      - active_speaker_slot: face_registry slot of whichever speaker
        is active during this segment, directly from the events.
      - persistent_hud_region: horizontal center of the game frame
        (NOT the HUD overlay) for gaming content.
      - prev_anywhere: previous segment's (x,y) regardless of age —
        last-ditch continuity before the hardcoded center.
      - hardcoded_center: true last resort at (50, 50).
    """

    def __init__(
        self,
        estimator: "SubjectConfidenceEstimator",
        content_type: str,
        last_seg,
        candidate_x: Optional[int],
        candidate_y: Optional[int] = 50,
    ):
        self.estimator = estimator
        self.content_type = content_type or "unknown"
        self.last_seg = last_seg  # expects .end, .subject_x, .subject_y, .confidence or None
        self.candidate_x = candidate_x
        self.candidate_y = candidate_y if candidate_y is not None else 50

    def resolve(self, seg_start: float, seg_end: float) -> tuple:
        cascade = FALLBACK_CASCADE_BY_CONTENT.get(
            self.content_type, FALLBACK_CASCADE_BY_CONTENT["unknown"],
        )
        for tier in cascade:
            result = self._try_tier(tier, seg_start, seg_end)
            if result is not None:
                x, y, source = result
                return (
                    int(round(max(0, min(100, x)))),
                    int(round(max(0, min(100, y)))),
                    source,
                )
        # Cascade is guaranteed to end in hardcoded_center, but just in
        # case it was misconfigured, return center here too.
        return 50, 50, "hardcoded_center_default"

    def _try_tier(self, tier: str, seg_start: float, seg_end: float):
        if tier == "prev_crop_continuity":
            ls = self.last_seg
            if ls is None:
                return None
            gap = float(seg_start) - float(getattr(ls, "end", -1.0))
            conf = float(getattr(ls, "confidence", 0.0) or 0.0)
            if gap <= 2.0 and gap >= 0.0 and conf >= 0.5:
                return (
                    _coerce_pct(getattr(ls, "subject_x_pct", None))
                    or _coerce_pct(getattr(ls, "subject_x", None), default=50),
                    _coerce_pct(getattr(ls, "subject_y_pct", None))
                    or _coerce_pct(getattr(ls, "subject_y", None), default=50),
                    "prev_crop_continuity",
                )
            return None

        if tier == "saliency_peak":
            peak = self._saliency_peak_in_window(seg_start, seg_end)
            if peak is not None:
                return (peak[0], peak[1], "saliency_peak")
            return None

        if tier == "face_centroid":
            centroid = self._face_centroid_in_window(seg_start, seg_end)
            if centroid is not None:
                return (centroid[0], centroid[1], "face_centroid")
            return None

        if tier == "motion_proxy":
            motion = self._motion_proxy_in_window(seg_start, seg_end)
            if motion is not None:
                return (motion[0], motion[1], "motion_proxy")
            return None

        if tier == "active_speaker_slot":
            active = self._active_speaker_slot_in_window(seg_start, seg_end)
            if active is not None:
                return (active[0], active[1], "active_speaker_slot")
            return None

        if tier == "persistent_hud_region":
            pr = self.estimator.persistent_regions
            if pr is not None:
                cx = getattr(pr, "game_frame_cx", None)
                cy = getattr(pr, "game_frame_cy", None)
                if cx is not None and cy is not None:
                    return (float(cx), float(cy), "persistent_hud_region")
            return None

        if tier == "prev_anywhere":
            ls = self.last_seg
            if ls is not None:
                return (
                    _coerce_pct(getattr(ls, "subject_x_pct", None))
                    or _coerce_pct(getattr(ls, "subject_x", None), default=50),
                    _coerce_pct(getattr(ls, "subject_y_pct", None))
                    or _coerce_pct(getattr(ls, "subject_y", None), default=50),
                    "prev_anywhere",
                )
            return None

        if tier == "hardcoded_center":
            return (50.0, 50.0, "hardcoded_center")

        return None

    # ── Tier helpers ──

    def _saliency_peak_in_window(self, seg_start: float, seg_end: float):
        regs = [
            r for r in self.estimator.frame_saliency
            if seg_start <= float(getattr(r, "timestamp", -1)) <= seg_end
        ]
        if not regs:
            return None
        # Gate on mean saliency strength; ignore weak noise blobs.
        scored = [
            r for r in regs
            if float(getattr(r, "saliency_score", 0.0)) >= 0.40
        ]
        if not scored:
            return None
        # Prefer the largest score·area product.
        best = max(
            scored,
            key=lambda r: float(r.saliency_score) * float(r.w) * float(r.h),
        )
        return (float(best.x), float(best.y))

    def _face_centroid_in_window(self, seg_start: float, seg_end: float):
        total_w = 0.0
        sum_x = 0.0
        sum_y = 0.0
        for df in self.estimator.dense_faces:
            if df.timestamp < seg_start or df.timestamp > seg_end:
                continue
            for f in df.faces:
                fx = getattr(f, "x", None)
                if fx is None:
                    fx = getattr(f, "nose_x", 50)
                fy = getattr(f, "y", None)
                if fy is None:
                    fy = getattr(f, "nose_y", 50)
                fw = float(getattr(f, "width", 10.0))
                fh = float(getattr(f, "height", 10.0))
                # Tracked identities get full weight; untracked faces
                # get the reduced weight so we don't centroid on
                # background crowds when a tracked speaker is also in
                # the frame but was missed by the primary gate.
                sid = getattr(f, "identity_id", -1)
                tracked = int(sid) in self.estimator._tracked_slot_ids if sid is not None else False
                base = 1.0 if tracked else UNTRACKED_FACE_WEIGHT
                w = max(0.1, base * (fw * fh) / 100.0)
                total_w += w
                sum_x += float(fx) * w
                sum_y += float(fy) * w
        if total_w <= 0:
            return None
        return (sum_x / total_w, sum_y / total_w)

    def _motion_proxy_in_window(self, seg_start: float, seg_end: float):
        # Face-position deltas across consecutive dense frames inside
        # the segment. The mean position of moving faces = motion
        # centroid proxy. Works even without full optical flow data.
        samples = []
        prev_map = {}
        for df in self.estimator.dense_faces:
            if df.timestamp < seg_start or df.timestamp > seg_end:
                continue
            curr_map = {}
            for f in df.faces:
                sid = getattr(f, "identity_id", -1)
                if sid is None or int(sid) < 0:
                    continue
                fx = getattr(f, "x", None)
                if fx is None:
                    fx = getattr(f, "nose_x", 50)
                fy = getattr(f, "y", None)
                if fy is None:
                    fy = getattr(f, "nose_y", 50)
                curr_map[int(sid)] = (float(fx), float(fy))
                if int(sid) in prev_map:
                    px, py = prev_map[int(sid)]
                    dx = float(fx) - px
                    dy = float(fy) - py
                    if (dx * dx + dy * dy) > 4.0:  # > 2% total move
                        samples.append((float(fx), float(fy)))
            prev_map = curr_map
        if not samples:
            return None
        sx = sum(p[0] for p in samples) / len(samples)
        sy = sum(p[1] for p in samples) / len(samples)
        return (sx, sy)

    def _active_speaker_slot_in_window(self, seg_start: float, seg_end: float):
        if not self.estimator.active_speaker_events:
            return None
        # Pick the slot with the largest coverage in the window.
        coverage: dict = {}
        for ev in self.estimator.active_speaker_events:
            ov_s = max(seg_start, ev.start)
            ov_e = min(seg_end, ev.end)
            if ov_e > ov_s and getattr(ev, "slot_id", -1) >= 0:
                coverage[int(ev.slot_id)] = (
                    coverage.get(int(ev.slot_id), 0.0) + (ov_e - ov_s)
                )
        if not coverage:
            return None
        best_slot = max(coverage, key=coverage.get)
        if self.estimator.face_registry is None:
            return None
        for s in getattr(self.estimator.face_registry, "slots", []) or []:
            if int(s.slot_id) == best_slot:
                return (float(s.x_center), 50.0)
        return None


def _coerce_pct(val, default=None):
    if val is None:
        return default
    try:
        v = float(val)
    except (TypeError, ValueError):
        return default
    # Values that look like pixel x/y get normalized to 0-100 when
    # caller stores them pre-scaled. Heuristic: if > 100, assume pixels
    # and caller should pass subject_x_pct explicitly; fall back to
    # default in that case to avoid invalid coordinates.
    if v < 0 or v > 100:
        return default
    return v


def resolve_fallback_center(
    estimator,
    content_type: str,
    seg_start: float,
    seg_end: float,
    last_seg,
    candidate_x: Optional[int],
    candidate_y: Optional[int] = 50,
) -> tuple:
    """Public entry point for Fix 2's salient fallback cascade.

    Args:
        estimator: SubjectConfidenceEstimator (holds dense_faces,
            face_registry, frame_saliency, active_speaker_events).
        content_type: Content type string; drives the cascade order.
        seg_start, seg_end: Segment time range in seconds.
        last_seg: Previous segment-like object (with .end, .subject_x,
            .subject_y, .confidence) or None for the first segment.
        candidate_x, candidate_y: Candidate 0-100 coordinates if any.

    Returns:
        (x_pct, y_pct, source_tag) — 0-100 integer coordinates + tag.
    """
    resolver = _SalientFallbackResolver(
        estimator=estimator,
        content_type=content_type,
        last_seg=last_seg,
        candidate_x=candidate_x,
        candidate_y=candidate_y,
    )
    return resolver.resolve(seg_start, seg_end)


def get_fallback_strategy(
    confidence: float,
    content_type: str,
    last_confident_x: Optional[int] = None,
    last_confident_slot: Optional[int] = None,
    candidate_x: Optional[int] = None,
    candidate_slot: Optional[int] = None,
) -> tuple:
    """Determine the fallback strategy based on confidence level.

    Returns (strategy, layout, subject_x, active_slot, reason).

    The fallback ladder:
      >= 0.70: no fallback needed (caller uses the candidate)
      >= 0.50: USE the current candidate position (face IS detected,
               just with medium confidence — don't discard it)
      >= 0.15: content-type-dependent (blur_fill or wide_master) but
               only after the salient-fallback cascade has tried and
               failed; caller passes the cascade result here via
               candidate_x. (Fix 5: raised wide_master floor from
               0.30 → 0.15 so momentary confidence dips don't trigger
               the letterbox.)
      <  0.15: wide_master (absolute last resort)
    """
    if confidence >= CONFIDENCE_HIGH:
        return None  # No fallback — use the candidate as-is

    if confidence >= CONFIDENCE_MEDIUM:
        # Medium confidence: a face IS detected but with lower certainty.
        # Use the CURRENT candidate position — not the stale last-known one.
        # The previous behavior of inheriting last_confident_x caused the
        # "subject at 51% but crop at 24%" bug when the face moved but
        # confidence dropped slightly.
        use_x = candidate_x if candidate_x is not None else last_confident_x
        use_slot = candidate_slot if candidate_slot is not None else last_confident_slot
        if use_x is not None:
            return (
                "stationary",
                "single",
                use_x,
                use_slot,
                "confidence_medium_current",
            )
        # No position at all — fall through to blur/wide
        pass

    preference = FALLBACK_PREFERENCE.get(content_type, "blur_fill")

    if confidence >= CONFIDENCE_WIDE_MASTER_FLOOR:
        # Fix 5: instead of defaulting to wide_master with hardcoded
        # center, the caller now passes in a salient-fallback center
        # (via candidate_x). Emit a stationary crop at that center.
        # Only when the cascade itself returned the hardcoded_center
        # tag (or candidate_x is None) do we fall to blur_fill.
        if candidate_x is not None:
            return (
                "stationary", "single",
                candidate_x,
                candidate_slot,
                "confidence_low_salient_fallback",
            )
        if preference == "wide_master":
            return ("wide_master", "wide_master", 50, None, "confidence_low_wide_master")
        else:
            return ("blur_fill", "blur_fill", 50, None, "confidence_low_blur_fill")

    # Absolute last resort — still try the cascade center before
    # letterboxing, but mark it so the pipeline knows the confidence
    # was very low.
    if candidate_x is not None:
        return (
            "stationary", "single",
            candidate_x,
            candidate_slot,
            "confidence_very_low_salient_fallback",
        )
    return ("wide_master", "wide_master", 50, None, "confidence_very_low_wide_master")
