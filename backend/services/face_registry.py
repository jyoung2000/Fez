"""Build a stable face registry from detection results.

Clusters face positions across all frames to identify consistent
face "slots" — positions where faces appear repeatedly. For a
2-speaker podcast, this produces slots like:
  Slot A: x=25% (left speaker, appears in 45 frames)
  Slot B: x=75% (right speaker, appears in 52 frames)

These slots are ground truth. The AI model's only job is to pick
which slot is the active speaker.
"""
import logging
import os
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ── Config flags for new behavior ──
REGISTRY_USE_HUMAN_WEIGHT = os.environ.get("REGISTRY_USE_HUMAN_WEIGHT", "true").lower() in ("true", "1", "yes")
REGISTRY_USE_COHESION_GATE = os.environ.get("REGISTRY_USE_COHESION_GATE", "true").lower() in ("true", "1", "yes")


# ── XC.1: face-registry burn-in ──────────────────────────────────────
# Real clips often open on B-roll, a single-host title card, or
# graphics before the panel actually assembles. Letting those frames
# influence slot discovery means slots get assigned wrong and stay
# wrong for the whole clip. The burn-in window says: "ignore the
# first N seconds when discovering identities; let the panel settle
# in before fixing slot positions." Frames inside the burn-in are
# still detected and tracked for downstream consumers — only the
# clustering input is filtered.
def _face_registry_burn_in_seconds() -> float:
    raw = os.environ.get("CLIPAI_FACE_REGISTRY_BURN_IN_S")
    if raw is None:
        return 5.0
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 5.0


def apply_face_registry_burn_in(face_results: list) -> list:
    """Return ``face_results`` with the first ``CLIPAI_FACE_REGISTRY_BURN_IN_S``
    seconds filtered out for slot-discovery purposes.

    Behavior:
      * burn-in 0 (or env-cleared) → returns ``face_results`` unchanged
      * clip shorter than burn-in window → returns ``face_results``
        unchanged (no point filtering everything away)
      * post-burn-in subset has fewer than 3 frames with faces → falls
        back to the unfiltered list so single-speaker / very-short
        clips still get a registry built
      * otherwise returns the post-burn-in list

    Callers pass the result to ``build_face_registry`` /
    ``build_face_registry_with_embeddings``. The full unfiltered list
    continues to flow into the active-speaker timeline / segmenter.
    """
    burn_in = _face_registry_burn_in_seconds()
    if burn_in <= 0 or not face_results:
        return list(face_results)
    timestamps = [
        float(getattr(fr, "timestamp", 0.0)) for fr in face_results
    ]
    if not timestamps:
        return list(face_results)
    span = max(timestamps) - min(timestamps)
    # Need at least burn_in + 0.5 s of additional content for the
    # filter to make sense. Otherwise we'd discard the whole clip.
    if span < burn_in + 0.5:
        return list(face_results)
    filtered = [
        fr for fr in face_results
        if float(getattr(fr, "timestamp", 0.0)) >= burn_in
    ]
    n_with_faces = sum(1 for fr in filtered if getattr(fr, "faces", None))
    if n_with_faces < 3:
        # Not enough post-burn-in data to cluster reliably. Leaving
        # the unfiltered list lets the legacy cohesion checks decide.
        return list(face_results)
    logger.info(
        "Face registry burn-in: dropped %d/%d frame results before t=%.1fs "
        "(%d frames retained for slot discovery)",
        len(face_results) - len(filtered), len(face_results), burn_in,
        len(filtered),
    )
    return filtered

# ── Cosine-similarity thresholds for face identity matching ──
# ArcFace 512-d embeddings have tighter within-class / looser between-class
# distributions than SFace 128-d, so the per-pair similarity thresholds shift.
# Read once at import time so the rest of the module can branch on simple
# constants without re-reading the env var on every call.
_FACE_EMB_BACKEND = os.environ.get("CLIPAI_FACE_EMBEDDING", "sface").lower()
# Same-identity floor: pair-level cosine similarity above this is treated as
# "same person." ArcFace separates classes better, so the threshold sits a hair
# higher than for SFace while keeping recall.
_SAME_ID_COSINE_THRESHOLD = 0.45 if _FACE_EMB_BACKEND == "arcface" else 0.40
# Different-identity floor: pairs below this are confidently NOT the same
# person. ArcFace's tighter class structure lets us be more permissive here
# (a lower threshold still flags "different").
_DIFF_ID_COSINE_THRESHOLD = 0.28 if _FACE_EMB_BACKEND == "arcface" else 0.35
# Teleport-merge: collapse two slots that share x-position AND look like the
# same identity. Held intentionally strict because false merges destroy the
# multi-speaker structure. ArcFace separates better, so the threshold can be
# slightly lower without losing precision.
_TELEPORT_MERGE_COSINE_SIM = 0.65 if _FACE_EMB_BACKEND == "arcface" else 0.70


@dataclass
class FaceSlot:
    """A stable face position identified across multiple frames."""
    slot_id: int          # 0-indexed
    x_center: float       # Average x-position (0-100)
    x_min: float          # Minimum observed x
    x_max: float          # Maximum observed x
    frame_count: int      # How many frames this face appears in
    avg_width: float      # Average face width
    avg_height: float     # Average face height

    def __post_init__(self):
        self.slot_id = int(self.slot_id)
        self.x_center = float(self.x_center)
        self.x_min = float(self.x_min)
        self.x_max = float(self.x_max)
        self.frame_count = int(self.frame_count)
        self.avg_width = float(self.avg_width)
        self.avg_height = float(self.avg_height)


@dataclass
class FaceRegistry:
    """Collection of stable face slots for a video."""
    slots: list[FaceSlot] = field(default_factory=list)
    total_frames: int = 0
    frames_with_faces: int = 0

    @property
    def multi_speaker(self) -> bool:
        return len(self.slots) >= 2

    @property
    def is_continuous_motion(self) -> bool:
        """Detect if face positions represent continuous motion rather than fixed speakers.

        Returns True when faces move freely across the frame (cartoons, sports,
        single-person vlogs) rather than sitting in fixed panel positions
        (podcasts, interviews).

        Heuristics:
        - Single slot with wide x_range (face moves within the "slot")
        - No slot has >80% of frames (no dominant fixed position)
        - Average slot x_range exceeds 15% (faces aren't stationary)
        """
        if not self.slots:
            return True  # No data → default to continuous (safer)
        if len(self.slots) == 1:
            slot = self.slots[0]
            # If the single slot spans a wide range, it's a moving subject
            return (slot.x_max - slot.x_min) > 15
        # Multiple slots: check if any single slot dominates AND has wide range
        total_frames = sum(s.frame_count for s in self.slots)
        if total_frames == 0:
            return True
        avg_range = sum(s.x_max - s.x_min for s in self.slots) / len(self.slots)
        dominant = max(self.slots, key=lambda s: s.frame_count)
        dominant_pct = dominant.frame_count / total_frames
        # If the dominant slot has wide range, it's moving even within its cluster
        if dominant_pct > 0.6 and (dominant.x_max - dominant.x_min) > 20:
            return True
        # If average slot range is high, faces aren't stationary
        if avg_range > 15:
            return True
        return False

    def nearest_slot(self, x: float) -> FaceSlot | None:
        """Find the slot closest to the given x position."""
        if not self.slots:
            return None
        return min(self.slots, key=lambda s: abs(s.x_center - x))

    def slot_by_id(self, slot_id: int) -> FaceSlot | None:
        for s in self.slots:
            if s.slot_id == slot_id:
                return s
        return None

    def to_dict(self) -> dict:
        """Serialize for API response to frontend."""
        return {
            "slots": [
                {"id": s.slot_id, "x": round(s.x_center), "frames": s.frame_count}
                for s in self.slots
            ],
            "total_frames": self.total_frames,
            "frames_with_faces": self.frames_with_faces,
            "multi_speaker": self.multi_speaker,
        }


def build_face_registry(
    face_results: list,  # list[FrameFaces]
    min_appearances: int = 3,
    cluster_gap: float = 15.0,
) -> FaceRegistry:
    """Build a face registry from face detection results.

    Groups detected faces by position across all frames. Faces that
    consistently appear near the same x-position are merged into a
    single "slot". Transient detections (< min_appearances) are
    discarded as noise.

    For multi-speaker panels (3+ faces/frame), automatically reduces
    the gap threshold to detect speakers sitting closer together.
    """
    # Collect all individual face positions — use nose_x (actual face center
    # from landmarks) rather than x_center (bbox center). nose_x is more
    # accurate: FaceMesh gives sub-pixel nose tip, YuNet gives nose keypoint.
    # The bbox center can be 10-20% off for side-profile faces.
    all_faces = []  # [(nose_x, width, height, frame_idx, weight)]
    non_human_kept = 0
    for fi, fr in enumerate(face_results):
        if not fr.faces:
            continue
        for face in fr.faces:
            # is_human: weight down rather than gate (Bug C fix)
            is_human = getattr(face, 'is_human', True)
            if REGISTRY_USE_HUMAN_WEIGHT:
                weight = 1.0 if is_human else 0.35
                if not is_human:
                    non_human_kept += 1
            else:
                if not is_human:
                    continue
                weight = 1.0
            all_faces.append((face.nose_x, face.width, face.height, fi, weight))
    if non_human_kept > 0:
        logger.info("is_human weight: kept %d non-human faces with weight=0.35", non_human_kept)

    if not all_faces:
        return FaceRegistry(
            total_frames=len(face_results),
            frames_with_faces=0,
        )

    # ── IQR outlier trimming ──
    # Remove extreme face positions that are noise (background people at frame
    # edges, logo detections, partial faces). Without this, a single face at
    # x=2% chains the entire [2-100] range into one garbage cluster.
    if len(all_faces) >= 20:
        x_sorted = sorted(f[0] for f in all_faces)
        q1_idx = len(x_sorted) // 4
        q3_idx = 3 * len(x_sorted) // 4
        q1 = x_sorted[q1_idx]
        q3 = x_sorted[q3_idx]
        iqr = q3 - q1
        # Use generous bounds (2.0 * IQR) to keep real edge speakers
        # but remove extreme outliers like x=2% or x=95%
        lower = max(5.0, q1 - 2.0 * iqr)
        upper = min(95.0, q3 + 2.0 * iqr)
        before_count = len(all_faces)
        all_faces = [f for f in all_faces if lower <= f[0] <= upper]
        trimmed = before_count - len(all_faces)
        if trimmed > 0:
            logger.info(
                "IQR outlier trimming: removed %d/%d faces outside [%.0f, %.0f] "
                "(Q1=%.0f, Q3=%.0f, IQR=%.0f)",
                trimmed, before_count, lower, upper, q1, q3, iqr,
            )
        if not all_faces:
            return FaceRegistry(
                total_frames=len(face_results),
                frames_with_faces=0,
            )

    # ── Adaptive gap threshold for multi-speaker panels ──
    # With 5 speakers across a frame, they're ~20% apart. The default
    # gap of 15 merges adjacent speakers. Scale down based on how many
    # faces are typically detected per frame.
    frame_face_counts = {}
    for f in all_faces:
        fi = f[3]  # frame_idx
        frame_face_counts[fi] = frame_face_counts.get(fi, 0) + 1
    max_faces_per_frame = max(frame_face_counts.values(), default=0)
    frames_with_3plus = sum(1 for c in frame_face_counts.values() if c >= 3)
    multi_speaker_ratio = frames_with_3plus / max(len(frame_face_counts), 1)

    if max_faces_per_frame >= 3 or multi_speaker_ratio > 0.05:
        # Multi-speaker: use per-frame position analysis to find the gap
        # that best separates speakers. Group faces within each frame by
        # x-position, find the median gap between adjacent faces in multi-face
        # frames, then use half that gap as the cluster threshold.
        per_frame_gaps = []
        for fi, count in frame_face_counts.items():
            if count < 2:
                continue
            frame_xs = sorted(f[0] for f in all_faces if f[3] == fi)
            for i in range(1, len(frame_xs)):
                per_frame_gaps.append(frame_xs[i] - frame_xs[i - 1])

        if per_frame_gaps:
            per_frame_gaps.sort()
            median_gap = per_frame_gaps[len(per_frame_gaps) // 2]
            # Use 50% of median inter-face gap as cluster threshold
            # (was 60%, tightened for 5+ speaker panels to prevent merge)
            gap_factor = 0.4 if max_faces_per_frame >= 5 else 0.5
            adaptive_gap = max(5.0, min(cluster_gap, median_gap * gap_factor))
            logger.info(
                "Multi-speaker panel (max %d faces/frame, %.0f%% with 3+): "
                "median inter-face gap=%.1f, cluster gap %.0f → %.0f",
                max_faces_per_frame, multi_speaker_ratio * 100,
                median_gap, cluster_gap, adaptive_gap,
            )
            cluster_gap = adaptive_gap

    # ── Two-pass clustering: first without midzone, then assign midzone ──
    # The mid-zone [40-60%] contains noise from merged detections, AI defaults,
    # and faces that are slightly off-center. If we cluster with these included,
    # they chain nearby real positions into one bloated cluster (e.g. [48-86%]).
    # Exception: if 3+ faces detected per frame, midzone faces are real center speakers.
    MIDZONE_LO, MIDZONE_HI = 40, 60
    if max_faces_per_frame >= 3:
        # 3+ faces in some frames — don't exclude midzone, center speaker is real
        outer_faces = all_faces
        midzone_faces = []
        logger.info("3+ faces in some frames — keeping midzone faces for clustering")
    else:
        outer_faces = [f for f in all_faces if f[0] < MIDZONE_LO or f[0] > MIDZONE_HI]
        midzone_faces = [f for f in all_faces if MIDZONE_LO <= f[0] <= MIDZONE_HI]

    # If no outer faces, fall back to using all faces — but check for bimodal
    # distribution within the midzone (two speakers both near center).
    if not outer_faces:
        # Check if midzone faces form two distinct groups (bimodal)
        if len(all_faces) >= 6:
            sorted_x = sorted(f[0] for f in all_faces)
            # Find the largest gap between consecutive face positions
            max_gap = 0
            max_gap_idx = 0
            for i in range(1, len(sorted_x)):
                gap = sorted_x[i] - sorted_x[i - 1]
                if gap > max_gap:
                    max_gap = gap
                    max_gap_idx = i
            # If there's a clear gap (>= 8%), split into two groups
            if max_gap >= 8:
                group_a = [f for f in all_faces if f[0] <= sorted_x[max_gap_idx - 1]]
                group_b = [f for f in all_faces if f[0] >= sorted_x[max_gap_idx]]
                if len(set(f[3] for f in group_a)) >= min_appearances and \
                   len(set(f[3] for f in group_b)) >= min_appearances:
                    logger.info(
                        "Bimodal midzone split: gap=%.1f%% at x=%.1f%%, "
                        "group_a=%d faces (mean=%.1f%%), group_b=%d faces (mean=%.1f%%)",
                        max_gap,
                        (sorted_x[max_gap_idx - 1] + sorted_x[max_gap_idx]) / 2,
                        len(group_a), sum(f[0] for f in group_a) / len(group_a),
                        len(group_b), sum(f[0] for f in group_b) / len(group_b),
                    )
                    outer_faces = all_faces
                    midzone_faces = []
                    # Use the detected gap as cluster_gap for this run
                    cluster_gap = max_gap * 0.8

        if not outer_faces:
            outer_faces = all_faces
            midzone_faces = []

    # Sort by x position and cluster by gap
    outer_faces.sort(key=lambda f: f[0])
    clusters: list[list[tuple]] = [[outer_faces[0]]]
    for face in outer_faces[1:]:
        if face[0] - clusters[-1][-1][0] > cluster_gap:
            clusters.append([face])
        else:
            clusters[-1].append(face)

    # Assign midzone faces to nearest cluster (if within 20% of cluster mean)
    for mf in midzone_faces:
        best_cluster = None
        best_dist = float('inf')
        for cl in clusters:
            cl_mean = sum(f[0] for f in cl) / len(cl)
            dist = abs(mf[0] - cl_mean)
            if dist < best_dist:
                best_dist = dist
                best_cluster = cl
        # Only assign if reasonably close (within 15% of a real cluster)
        if best_cluster is not None and best_dist <= 15:
            best_cluster.append(mf)

    # Build slots from clusters with enough appearances.
    # Use median (not mean) for x_center — robust to Haar cascade outliers
    # where the bbox extends asymmetrically into the background.
    # Also trim extreme outliers (outside IQR * 1.5) before computing.
    slots = []
    for cluster in clusters:
        # Use weighted frame count: sum of weights per unique frame
        frame_weights = {}
        for f in cluster:
            fi = f[3]
            w = f[4] if len(f) > 4 else 1.0
            frame_weights[fi] = max(frame_weights.get(fi, 0.0), w)
        weighted_frames = sum(frame_weights.values())
        unique_frames = len(frame_weights)
        # Threshold on weighted evidence, not raw count
        if weighted_frames < min_appearances:
            continue
        x_positions = sorted([f[0] for f in cluster])
        widths = [f[1] for f in cluster]
        heights = [f[2] for f in cluster]

        # IQR-based outlier trimming (AutoFlip-style temporal filtering)
        if len(x_positions) >= 5:
            q1_idx = len(x_positions) // 4
            q3_idx = 3 * len(x_positions) // 4
            q1 = x_positions[q1_idx]
            q3 = x_positions[q3_idx]
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            trimmed = [x for x in x_positions if lower <= x <= upper]
            if len(trimmed) >= min_appearances:
                x_positions = trimmed

        # Median for robustness
        mid = len(x_positions) // 2
        if len(x_positions) % 2 == 0 and len(x_positions) >= 2:
            median_x = (x_positions[mid - 1] + x_positions[mid]) / 2
        else:
            median_x = x_positions[mid]

        slots.append(FaceSlot(
            slot_id=len(slots),
            x_center=float(round(median_x, 1)),
            x_min=float(round(min(x_positions), 1)),
            x_max=float(round(max(x_positions), 1)),
            frame_count=unique_frames,
            avg_width=float(round(sum(widths) / len(widths), 1)),
            avg_height=float(round(sum(heights) / len(heights), 1)),
        ))

    # Sort by x position (left to right)
    slots.sort(key=lambda s: s.x_center)
    for i, s in enumerate(slots):
        s.slot_id = i

    # ── Histogram fallback for multi-speaker panels ──
    # If gap-based clustering produced only 1 slot from many faces with wide
    # x-range, the speakers are too close together for gap-based splitting.
    # Use histogram peak detection: bin positions into 5% buckets, find peaks.
    if len(slots) <= 1 and len(all_faces) >= 30 and max_faces_per_frame >= 2:
        x_range = max(f[0] for f in all_faces) - min(f[0] for f in all_faces)
        if x_range > 25:  # Only if faces span > 25% of frame (not all same speaker)
            logger.info(
                "Gap-based clustering found %d slot(s) from %d faces spanning %.0f%%. "
                "Trying histogram peak detection...",
                len(slots), len(all_faces), x_range,
            )
            # Histogram with 5% bins
            bin_size = 5.0
            n_bins = 20
            bins = [[] for _ in range(n_bins)]
            for f in all_faces:
                b = min(n_bins - 1, max(0, int(f[0] / bin_size)))
                bins[b].append(f)

            # Find peaks: bins with > 5 faces that are local maxima
            peak_slots = []
            for i in range(n_bins):
                count = len(bins[i])
                if count < max(5, len(all_faces) * 0.02):
                    continue
                left = len(bins[i - 1]) if i > 0 else 0
                right = len(bins[i + 1]) if i < n_bins - 1 else 0
                if count >= left and count >= right:
                    # Merge with adjacent bins for better statistics
                    peak_faces = list(bins[i])
                    if i > 0:
                        peak_faces.extend(bins[i - 1])
                    if i < n_bins - 1:
                        peak_faces.extend(bins[i + 1])

                    x_positions = sorted([f[0] for f in peak_faces])
                    # Weighted frame count for histogram path too
                    pf_weights = {}
                    for f in peak_faces:
                        fi = f[3]
                        w = f[4] if len(f) > 4 else 1.0
                        pf_weights[fi] = max(pf_weights.get(fi, 0.0), w)
                    unique_frames = len(pf_weights)
                    weighted_frames = sum(pf_weights.values())
                    if weighted_frames < min_appearances:
                        continue

                    mid = len(x_positions) // 2
                    median_x = x_positions[mid]
                    widths = [f[1] for f in peak_faces]
                    heights = [f[2] for f in peak_faces]

                    # Check it's not too close to an existing peak
                    too_close = any(abs(median_x - ps.x_center) < 10 for ps in peak_slots)
                    if too_close:
                        continue

                    peak_slots.append(FaceSlot(
                        slot_id=len(peak_slots),
                        x_center=float(round(median_x, 1)),
                        x_min=float(round(min(x_positions), 1)),
                        x_max=float(round(max(x_positions), 1)),
                        frame_count=unique_frames,
                        avg_width=float(round(sum(widths) / len(widths), 1)),
                        avg_height=float(round(sum(heights) / len(heights), 1)),
                    ))

            if len(peak_slots) >= 2:
                slots = peak_slots
                slots.sort(key=lambda s: s.x_center)
                for i, s in enumerate(slots):
                    s.slot_id = i
                logger.info(
                    "Histogram peak detection found %d speaker positions",
                    len(slots),
                )

    registry = FaceRegistry(
        slots=slots,
        total_frames=len(face_results),
        frames_with_faces=sum(1 for fr in face_results if fr.faces),
    )

    logger.info(
        "Face registry: %d slots from %d frames (%d with faces)",
        len(slots), registry.total_frames, registry.frames_with_faces,
    )
    for s in slots:
        logger.info(
            "  Slot %d: x=%.0f%% [%.0f-%.0f], %d frames, avg_size=%.0f%%x%.0f%%",
            s.slot_id, s.x_center, s.x_min, s.x_max,
            s.frame_count, s.avg_width, s.avg_height,
        )

    return registry


def _merge_teleporting_embedding_slots(
    slots: list,
    slot_face_indices: dict,
    embeddings_norm,
) -> list:
    """Collapse embedding slots that share a mean_x within ±8 %% AND whose
    face embeddings are cosine-similar above 0.70.

    Fix 1: seated-panel runs produced 11 embedding slots because SFace
    split the same person across pose / lighting variations. Several of
    those slots had x-ranges spanning 80%%+ of the frame width (e.g.
    [14-97], [15-98], [14-95]) — a real person doesn't teleport, and
    these are all the same identity. Merging them restores the actual
    seat count (5 position-cluster slots).

    Args:
        slots: list[FaceSlot] from embedding clustering (already sorted
            left-to-right with contiguous 0..N-1 slot_ids).
        slot_face_indices: {slot_id → list of face indices into
            embeddings_norm / all_faces} — the ORIGINAL embedding-group
            membership, NOT a re-assignment by nearest x. Using nearest-x
            reassignment breaks on stage/panel clips where multiple
            identities traverse the same region: their true groups are
            orthogonal in embedding space but re-assignment would lump
            them together and produce false merges.
        embeddings_norm: L2-normalized embedding matrix for cosine
            similarity computation.

    Returns:
        list[FaceSlot] — merged, re-sorted, re-numbered left-to-right.
    """
    import numpy as np

    if len(slots) < 2:
        return slots

    # Build per-slot centroid embeddings from the TRUE embedding-group
    # membership (passed in from the caller).
    slot_centroids: dict = {}
    for sid, face_idxs in slot_face_indices.items():
        if not face_idxs:
            continue
        embs = embeddings_norm[face_idxs]
        centroid = embs.mean(axis=0)
        c_norm = float(np.linalg.norm(centroid))
        if c_norm > 1e-8:
            centroid = centroid / c_norm
        slot_centroids[sid] = centroid

    # Union-find: merge any pair that satisfies BOTH conditions.
    parent = {s.slot_id: s.slot_id for s in slots}

    def _find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(a, b):
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[ra] = rb

    merges = 0
    for i in range(len(slots)):
        for j in range(i + 1, len(slots)):
            a, b = slots[i], slots[j]
            if abs(a.x_center - b.x_center) >= 8.0:
                continue
            ca = slot_centroids.get(a.slot_id)
            cb = slot_centroids.get(b.slot_id)
            if ca is None or cb is None:
                continue
            cos_sim = float(np.dot(ca, cb))
            # Threshold depends on CLIPAI_FACE_EMBEDDING backend
            if cos_sim > _TELEPORT_MERGE_COSINE_SIM:
                _union(a.slot_id, b.slot_id)
                merges += 1

    if merges == 0:
        return slots

    # Rebuild slots by merged group.
    group_members: dict = {}
    for s in slots:
        root = _find(s.slot_id)
        group_members.setdefault(root, []).append(s)

    merged: list = []
    for root, members in group_members.items():
        if len(members) == 1:
            merged.append(members[0])
            continue
        total_frames = sum(m.frame_count for m in members)
        # Weight each member's x_center by its frame_count.
        x_center_w = sum(m.x_center * m.frame_count for m in members) / max(total_frames, 1)
        x_min = min(m.x_min for m in members)
        x_max = max(m.x_max for m in members)
        avg_w = sum(m.avg_width * m.frame_count for m in members) / max(total_frames, 1)
        avg_h = sum(m.avg_height * m.frame_count for m in members) / max(total_frames, 1)
        merged.append(FaceSlot(
            slot_id=members[0].slot_id,  # placeholder — re-numbered below
            x_center=float(round(x_center_w, 1)),
            x_min=float(round(x_min, 1)),
            x_max=float(round(x_max, 1)),
            frame_count=int(total_frames),
            avg_width=float(round(avg_w, 1)),
            avg_height=float(round(avg_h, 1)),
        ))

    merged.sort(key=lambda s: s.x_center)
    for i, s in enumerate(merged):
        s.slot_id = i

    logger.info(
        "[FaceRegistry] teleport-merge: collapsed %d slot pair(s) "
        "→ %d slots (from %d)",
        merges, len(merged), len(slots),
    )
    return merged


def _detect_panel_mode(face_results: list) -> bool:
    """Heuristic: does this clip look like a seated multi-speaker panel?

    Used by build_face_registry_with_embeddings to decide whether
    position-based clustering should override embedding-based clustering.
    Panel signals:
      - ≥25% of frames have 2+ faces simultaneously (multi-face frames)
      - Average face count per populated frame ≥1.5

    This is the same panel signature content_classifier.py later keys
    on to emit content_type=multi_speaker_panel (Fix 3); we compute it
    here early because the registry decision must happen before the
    classifier runs.
    """
    if not face_results:
        return False
    populated = [fr for fr in face_results if fr.faces]
    if len(populated) < 10:
        return False
    total = len(populated)
    multi_face_frames = sum(1 for fr in populated if len(fr.faces) >= 2)
    multi_face_pct = multi_face_frames / total
    avg_faces = sum(len(fr.faces) for fr in populated) / total
    return multi_face_pct >= 0.25 and avg_faces >= 1.5


def build_face_registry_with_embeddings(
    face_results: list,
    min_appearances: int = 3,
    cosine_threshold: float = 0.20,
) -> "FaceRegistry":
    """Build face registry using identity embeddings for cross-frame matching.

    Uses cosine similarity of face embeddings to group faces across frames.
    Tight threshold (0.20) prevents chaining different people together.
    Falls back to position-based clustering if embeddings produce garbage
    clusters (span > 40% of frame width = different people merged).
    """
    import numpy as np

    # Collect all faces with embeddings (is_human=False weighted down, not gated)
    all_faces = []  # [(embedding, nose_x, width, height, frame_idx)]
    total_faces = 0
    non_human_kept = 0
    for fi, fr in enumerate(face_results):
        for face in fr.faces:
            is_human = getattr(face, 'is_human', True)
            if REGISTRY_USE_HUMAN_WEIGHT:
                if not is_human:
                    non_human_kept += 1
            else:
                if not is_human:
                    continue
            total_faces += 1
            if face.identity_embedding is not None:
                all_faces.append((
                    np.array(face.identity_embedding, dtype=np.float32),
                    face.nose_x, face.width, face.height, fi,
                ))

    # Fall back to position-based clustering if <50% of faces have embeddings
    if len(all_faces) < total_faces * 0.5 or len(all_faces) < 3:
        logger.info(
            "Embedding coverage too low (%d/%d faces) — falling back to position-based registry",
            len(all_faces), total_faces,
        )
        return build_face_registry(face_results, min_appearances)

    # Mixed-dimension guard: if the run mixes 128-d (SFace) and 512-d (ArcFace)
    # embeddings — e.g. mid-run env-var flip — np.stack would raise. Detect
    # the heterogeneity and fall back to position-based clustering instead of
    # crashing the pipeline.
    emb_dims = {len(f[0]) for f in all_faces}
    if len(emb_dims) > 1:
        logger.warning(
            "Mixed embedding dimensions detected (%s) — falling back to "
            "position-based registry. This usually means CLIPAI_FACE_EMBEDDING "
            "changed mid-run.",
            sorted(emb_dims),
        )
        return build_face_registry(face_results, min_appearances)

    # Build adjacency via cosine similarity
    n = len(all_faces)
    embeddings = np.stack([f[0] for f in all_faces])
    # Normalize for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    embeddings_norm = embeddings / norms

    # Union-Find for connected components
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # ── Co-occurrence guard ──
    # Two faces detected in the SAME frame at DIFFERENT positions cannot be
    # the same person. Build a "cannot-link" set of (i, j) pairs that must
    # never be unioned, preventing identity collapse when embeddings are
    # similar (same lighting, similar clothing, etc.)
    cannot_link = set()
    from collections import defaultdict as _defaultdict
    faces_by_frame = _defaultdict(list)
    for idx, (_, _, _, _, frame_idx) in enumerate(all_faces):
        faces_by_frame[frame_idx].append(idx)
    for frame_idx, indices in faces_by_frame.items():
        if len(indices) < 2:
            continue
        for ii in range(len(indices)):
            for jj in range(ii + 1, len(indices)):
                a, b = indices[ii], indices[jj]
                # Only block if they're spatially distinct (>5% apart)
                x_a = all_faces[a][1]
                x_b = all_faces[b][1]
                if abs(x_a - x_b) > 5.0:
                    pair = (min(a, b), max(a, b))
                    cannot_link.add(pair)

    # ── Grace period: require MIN_LIFETIME unique frames before merging ──
    # A face that has appeared in fewer than MIN_LIFETIME frames is "new" and
    # should not be merged into an existing identity — it may be a distinct
    # speaker who just appeared.
    MIN_LIFETIME_FRAMES = 10
    # Precompute unique frame count per face's eventual group
    face_frame_idx = [all_faces[i][4] for i in range(n)]

    # Compare all pairs — for typical video (< 500 faces), this is fast
    sim_matrix = embeddings_norm @ embeddings_norm.T
    for i in range(n):
        for j in range(i + 1, n):
            # SFace uses cosine distance; lower = more similar
            cosine_dist = 1.0 - sim_matrix[i, j]
            if cosine_dist < cosine_threshold:
                # Co-occurrence guard: skip if detected in same frame
                pair = (min(i, j), max(i, j))
                if pair in cannot_link:
                    continue
                # Check that merging won't link faces that co-occur
                # (transitivity: find all members of both groups)
                root_i = find(i)
                root_j = find(j)
                if root_i == root_j:
                    continue
                # Gather all members of both groups
                group_i = [k for k in range(n) if find(k) == root_i]
                group_j = [k for k in range(n) if find(k) == root_j]
                # Check if any pair across groups co-occurs
                co_occurs = False
                for gi in group_i:
                    for gj in group_j:
                        p = (min(gi, gj), max(gi, gj))
                        if p in cannot_link:
                            co_occurs = True
                            break
                    if co_occurs:
                        break
                if co_occurs:
                    continue
                union(i, j)

    # Group by connected component
    groups: dict[int, list[int]] = {}
    for i in range(n):
        root = find(i)
        groups.setdefault(root, []).append(i)

    # Build slots from groups. Track each slot's ORIGINAL embedding-group
    # membership so Fix 1's teleport-merge pass (below) can compute
    # per-slot centroids from the true embedding groups instead of
    # re-assigning faces by nearest x-center (which breaks on stage
    # clips where multiple identities traverse the same region).
    slots = []
    slot_to_group_indices: list = []  # parallel list; re-keyed after sort
    for group_indices in groups.values():
        unique_frames = len(set(all_faces[i][4] for i in group_indices))
        if unique_frames < min_appearances:
            continue

        x_positions = sorted([all_faces[i][1] for i in group_indices])
        widths = [all_faces[i][2] for i in group_indices]
        heights = [all_faces[i][3] for i in group_indices]

        # Median x for robustness
        mid = len(x_positions) // 2
        if len(x_positions) % 2 == 0 and len(x_positions) >= 2:
            median_x = (x_positions[mid - 1] + x_positions[mid]) / 2
        else:
            median_x = x_positions[mid]

        slot_to_group_indices.append(list(group_indices))
        slots.append(FaceSlot(
            slot_id=len(slots),
            x_center=float(round(median_x, 1)),
            x_min=float(round(min(x_positions), 1)),
            x_max=float(round(max(x_positions), 1)),
            frame_count=unique_frames,
            avg_width=float(round(sum(widths) / len(widths), 1)),
            avg_height=float(round(sum(heights) / len(heights), 1)),
        ))

    # Sort left to right. We also sort slot_to_group_indices in lockstep
    # so the parallel mapping survives — essential for the Fix 1
    # teleport-merge pass below.
    slot_order = sorted(range(len(slots)), key=lambda i: slots[i].x_center)
    slots = [slots[i] for i in slot_order]
    slot_to_group_indices = [slot_to_group_indices[i] for i in slot_order]
    for i, s in enumerate(slots):
        s.slot_id = i

    # Build a dict keyed by the final post-sort slot_id.
    slot_face_indices_by_id = {
        s.slot_id: slot_to_group_indices[i] for i, s in enumerate(slots)
    }

    registry = FaceRegistry(
        slots=slots,
        total_frames=len(face_results),
        frames_with_faces=sum(1 for fr in face_results if fr.faces),
    )

    # Validate clusters by embedding cohesion, not x-span.
    # A cluster with wide x-span (>40%) but high cosine cohesion (>0.55) is one
    # person moving across the stage. A cluster with low cohesion (<0.55) is
    # different people accidentally chained together. (Bug A fix)
    if REGISTRY_USE_COHESION_GATE:
        # Build slot → face indices map for cohesion check
        slot_face_indices = {}
        for slot_idx, group_indices in enumerate(
            [gi for gi in groups.values() if len(set(all_faces[i][4] for i in gi)) >= min_appearances]
        ):
            slot_face_indices[slot_idx] = group_indices

        incoherent_clusters = []
        for slot_idx, (slot, face_idxs) in enumerate(zip(slots, slot_face_indices.values())):
            span = slot.x_max - slot.x_min
            if len(face_idxs) < 3:
                continue  # not enough data to assess cohesion
            slot_embs = np.stack([all_faces[i][0] for i in face_idxs])
            slot_norms = np.linalg.norm(slot_embs, axis=1, keepdims=True)
            slot_norms = np.maximum(slot_norms, 1e-8)
            slot_embs_norm = slot_embs / slot_norms
            centroid = slot_embs_norm.mean(axis=0)
            c_norm = np.linalg.norm(centroid)
            if c_norm > 1e-8:
                centroid = centroid / c_norm
            sims = slot_embs_norm @ centroid
            cohesion = float(np.mean(sims))
            logger.info(
                "  Slot %d: span=%.0f%%, cohesion=%.3f (%d faces)",
                slot.slot_id, span, cohesion, len(face_idxs),
            )
            if cohesion < 0.55:
                incoherent_clusters.append(slot)

        if incoherent_clusters:
            logger.warning(
                "Embedding clustering: %d cluster(s) with low cohesion (<0.55): %s. "
                "Falling back to position-based registry for those.",
                len(incoherent_clusters),
                [(f"slot{s.slot_id}: span=[{s.x_min:.0f}-{s.x_max:.0f}]") for s in incoherent_clusters],
            )
            # Only fall back if ALL clusters are incoherent
            if len(incoherent_clusters) == len(slots):
                return build_face_registry(face_results, min_appearances)
            # Otherwise, remove only the incoherent slots
            incoherent_ids = {s.slot_id for s in incoherent_clusters}
            slots = [s for s in slots if s.slot_id not in incoherent_ids]
            for i, s in enumerate(slots):
                s.slot_id = i
    else:
        # Legacy span-based check (behind flag for rollback)
        garbage_clusters = [s for s in slots if (s.x_max - s.x_min) > 40]
        if garbage_clusters:
            logger.warning(
                "Embedding clustering produced %d garbage cluster(s) (span > 40%% width): %s. "
                "Falling back to position-based registry.",
                len(garbage_clusters),
                [(f"slot{s.slot_id}: [{s.x_min:.0f}-{s.x_max:.0f}]") for s in garbage_clusters],
            )
            return build_face_registry(face_results, min_appearances)

    # ── Cross-shot merge sanity check ──
    # Any embedding slot whose x-position spans >60% of the frame across
    # >30 detected frames is almost certainly a cross-shot merge — one
    # "identity" that's actually several different people or the same
    # person appearing at wildly different positions in unrelated shots.
    #
    # v2 Phase 11 (Fix 2): ANY single slot failing this check is enough
    # to poison the registry, because downstream speaker-to-slot mapping
    # collapses onto that one merged slot. The previous "> 25% of slots"
    # gate let the Verzuz/Tank-Tyrese clip through with Slot 8 spanning
    # [14-98]% (1 of 9 slots) — that single slot made the entire
    # embedding registry unusable. Lower the threshold to any-single-slot.
    if slots:
        wide_span_bad = [
            s for s in slots
            if (s.x_max - s.x_min) > 60 and s.frame_count > 30
        ]
        if wide_span_bad:
            pos_reg = build_face_registry(face_results, min_appearances)
            logger.warning(
                "[FaceRegistry] embedding registry rejected: %d of %d slot(s) "
                "had span>60%% AND frame_count>30 (cross-shot merge). Using "
                "position-based (%d slots) instead of embeddings (%d slots). "
                "Offending slots: %s",
                len(wide_span_bad), len(slots), len(pos_reg.slots), len(slots),
                [f"slot{s.slot_id}:[{s.x_min:.0f}-{s.x_max:.0f}],{s.frame_count}f"
                 for s in wide_span_bad],
            )
            return pos_reg

    # ── Fix 1: teleport-merge pass ──
    # Collapse embedding slots with similar mean_x + high cosine similarity.
    # SFace over-splits stable identities under pose / lighting variation,
    # producing slots that "teleport" across frame width. The merge pass
    # uses the TRUE embedding-group membership (slot_face_indices_by_id)
    # so per-slot centroids aren't contaminated by nearest-x reassignment
    # — critical on stage clips where multiple identities traverse the
    # same region and would otherwise get falsely merged.
    slots = _merge_teleporting_embedding_slots(
        slots, slot_face_indices_by_id, embeddings_norm,
    )
    # Re-apply to the registry object so downstream code sees the merged slots.
    registry.slots = slots

    # Take max(embedding_count, position_count): whichever finds more identities wins.
    # Embeddings are the primary source of truth, but position-based may catch
    # speakers that the embedding clusterer missed (e.g., no embeddings available).
    pos_registry = build_face_registry(face_results, min_appearances)

    # ── Fix 1: panel-aware arbitration ──
    # When both methods run, prefer POSITION clusters if we're on a seated
    # multi-speaker panel AND embedding clustering is fragmenting (producing
    # substantially more slots than position clustering). Max-wins is wrong
    # on panels because embedding splits happen on pose / lighting, not
    # identity, and inflating the slot count cascades into misclassification
    # (vlog vs multi_speaker_panel), mis-mapping speakers to phantom slots,
    # and slot-snap being skipped as "continuous motion."
    panel_mode = _detect_panel_mode(face_results)
    if (panel_mode
            and 2 <= len(pos_registry.slots) <= 6
            and len(slots) > 1.8 * len(pos_registry.slots)):
        logger.info(
            "[FaceRegistry] panel_mode=True, pos=%d, emb=%d (fragmented) "
            "— using position-based",
            len(pos_registry.slots), len(slots),
        )
        return pos_registry

    # v2 Phase 11 (Fix 2): for multi-speaker panels, prefer the position-
    # based registry whenever it produced a reasonable 2-6 slot layout,
    # even if the embedding registry found more slots. Embeddings split
    # on pose/lighting rather than identity on seated panel clips, so a
    # higher embedding-slot count usually means fragmentation, not
    # coverage. The legacy "max wins" rule is kept for non-panel content.
    if panel_mode and 2 <= len(pos_registry.slots) <= 6:
        if len(pos_registry.slots) <= len(slots):
            logger.info(
                "[FaceRegistry] panel_mode=True, pos=%d slots, emb=%d slots "
                "— preferring position-based (panel seats are fixed; embedding "
                "splits on pose/lighting, not identity)",
                len(pos_registry.slots), len(slots),
            )
            return pos_registry

    if len(pos_registry.slots) > len(slots):
        logger.info(
            "Embedding registry found %d slots but position-based found %d — using position-based (max wins)",
            len(slots), len(pos_registry.slots),
        )
        return pos_registry
    elif len(pos_registry.slots) < len(slots):
        logger.info(
            "Embedding registry found %d slots, position-based found %d — using embeddings (max wins)",
            len(slots), len(pos_registry.slots),
        )

    logger.info(
        "Face registry (embeddings): %d slots from %d faces (%d with embeddings)",
        len(slots), total_faces, len(all_faces),
    )
    for s in slots:
        logger.info(
            "  Slot %d: x=%.0f%% [%.0f-%.0f], %d frames, avg_size=%.0f%%x%.0f%%",
            s.slot_id, s.x_center, s.x_min, s.x_max,
            s.frame_count, s.avg_width, s.avg_height,
        )

    # Assign identity_id back to each face in the original results
    assign_identities(face_results, registry)

    return registry


def assign_identities(face_results: list, registry: "FaceRegistry") -> None:
    """Assign identity_id to each face.

    Uses embedding cosine similarity when available (matches even when
    speakers move positions). Falls back to nearest-slot-by-position
    for faces without embeddings.
    """
    if not registry.slots:
        return

    # Build reference embeddings per slot from faces already assigned
    slot_embeddings = {}
    for fr in face_results:
        for face in fr.faces:
            if face.identity_embedding is not None and face.identity_id >= 0:
                slot_embeddings.setdefault(face.identity_id, []).append(face.identity_embedding)

    # Compute centroid per slot
    slot_centroids = {}
    if slot_embeddings:
        try:
            import numpy as np
            for sid, embs in slot_embeddings.items():
                arr = np.array(embs, dtype=np.float32)
                centroid = arr.mean(axis=0)
                norm = np.linalg.norm(centroid)
                if norm > 1e-8:
                    centroid = centroid / norm
                slot_centroids[sid] = centroid
        except ImportError:
            pass

    for fr in face_results:
        for face in fr.faces:
            # Try embedding match first
            if face.identity_embedding is not None and slot_centroids:
                try:
                    import numpy as np
                    emb = np.array(face.identity_embedding, dtype=np.float32)
                    emb_norm = np.linalg.norm(emb)
                    if emb_norm > 1e-8:
                        emb = emb / emb_norm
                    best_sid = -1
                    best_sim = -1.0
                    for sid, centroid in slot_centroids.items():
                        sim = float(np.dot(emb, centroid))
                        if sim > best_sim:
                            best_sim = sim
                            best_sid = sid
                    # Threshold depends on CLIPAI_FACE_EMBEDDING backend
                    if best_sid >= 0 and best_sim > _SAME_ID_COSINE_THRESHOLD:
                        face.identity_id = best_sid
                        continue
                except (ImportError, Exception):
                    pass

            # Fallback: nearest slot by position
            slot = registry.nearest_slot(face.nose_x)
            if slot:
                face.identity_id = slot.slot_id
