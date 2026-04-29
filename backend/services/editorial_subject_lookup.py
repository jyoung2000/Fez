"""Per-frame subject position resolver for the editorial planner.

The Phase E editorial planner emits decisions like "shot N: cut to
``speaker_2``" or "shot M: react on ``speaker_3`` at t=4.2s". To turn
those labels into camera moves the LP solver needs ``(x, y)`` in source
coordinates. This module builds that resolver as a closure over the
extraction outputs (face registry, dense-face stream, speaker→slot map).

Resolution chain for ``(label, t)`` → position:
    1. Map ``label`` (e.g. ``"speaker_2"``) to a face-registry slot id
       via the ``speaker_to_slot`` mapping. If the label looks like
       ``"slot_<n>"`` it is read directly. Unknown labels return None.
    2. Walk the dense-face stream for the closest frame whose
       ``timestamp`` is within ``freshness_window_s`` of ``t`` AND that
       contains a face whose ``identity_id`` matches the slot. When
       found, return that face's ``(nose_x, nose_y)`` converted to
       pixels.
    3. Otherwise fall back to the slot's persistent ``x_center`` from
       the face registry. ``y`` is unknown at slot level so it falls
       back to the source mid-line (``source_height / 2``).
    4. If nothing resolves, return None — callers treat it as
       "no editorial override at this frame."

The returned positions are in PIXELS so callers can normalize to
whatever coordinate convention they need (the editorial planner
divides by ``source_width`` to convert to LP target-fractions; the
diagnostics renderer uses pixels directly).
"""

from __future__ import annotations

import logging
import re
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


_LABEL_SLOT_RE = re.compile(r"^slot[_\-]?(\d+)$", re.IGNORECASE)


def _parse_explicit_slot_label(label: str) -> Optional[int]:
    """Read ``slot_3`` / ``slot-3`` / ``Slot3`` styles → 3."""
    if not isinstance(label, str):
        return None
    m = _LABEL_SLOT_RE.match(label.strip())
    if m is None:
        return None
    try:
        return int(m.group(1))
    except (TypeError, ValueError):
        return None


def _infer_speaker_to_slot_from_dense_faces(dense_faces: list) -> dict[str, int]:
    """Synthesize a label→slot map when the caller didn't pass one.

    Walks the dense face stream, picks the most-frequent ``identity_id``
    for each ``is_speaking`` face flagged at any frame, and emits
    ``"speaker_<rank>"`` keys ranked by total speaking duration. This is
    a best-effort fallback — callers with a real Whisper-derived map
    should pass that through.
    """
    if not dense_faces:
        return {}
    slot_seconds: dict[int, float] = {}
    timestamps: list[float] = []
    for ff in dense_faces:
        ts = float(getattr(ff, "timestamp", 0.0))
        timestamps.append(ts)
    timestamps.sort()
    if not timestamps:
        return {}
    avg_dt = max(
        (timestamps[-1] - timestamps[0]) / max(len(timestamps) - 1, 1),
        0.04,
    )
    for ff in dense_faces:
        for face in getattr(ff, "faces", []) or []:
            sid = int(getattr(face, "identity_id", -1))
            if sid < 0:
                continue
            if not bool(getattr(face, "is_speaking", False)):
                continue
            slot_seconds[sid] = slot_seconds.get(sid, 0.0) + avg_dt
    if not slot_seconds:
        return {}
    ranked = sorted(slot_seconds.items(), key=lambda kv: -kv[1])
    return {f"speaker_{rank + 1}": sid for rank, (sid, _) in enumerate(ranked)}


def _slot_y_default(face_registry: Any, source_height: int) -> float:
    """Best-guess y-position for a slot when dense data is missing.

    Slots in the registry only store ``x_center`` / ``avg_height`` —
    no persistent y-center. Use the source mid-line as a neutral
    fallback so cross-y motion stays minimal under the editorial
    override.
    """
    return float(source_height) / 2.0


def build_subject_position_lookup(
    *,
    face_registry: Any,
    dense_faces: list,
    speaker_to_slot: Optional[dict[str, int]] = None,
    source_width: int = 1920,
    source_height: int = 1080,
    freshness_window_s: float = 0.2,
) -> Callable[[str, float], Optional[tuple[float, float]]]:
    """Build a closure resolving ``(label, t)`` → ``(x_px, y_px)`` or None.

    Args:
        face_registry: A :class:`FaceRegistry` (or duck-typed object
            with ``slots`` and ``slot_by_id``). May be None.
        dense_faces: Iterable of ``FrameFaces`` (or anything with
            ``timestamp`` and ``faces`` where each face exposes
            ``identity_id`` / ``nose_x`` / ``nose_y`` in 0-100 percent).
        speaker_to_slot: Optional explicit label→slot map from
            :func:`map_speakers_to_face_slots`. Inferred from
            ``dense_faces`` if absent.
        source_width: Source frame width in pixels (for percent → pixel).
        source_height: Source frame height in pixels.
        freshness_window_s: Maximum |t - frame_t| treated as "fresh"
            for dense-face resolution.

    Returns:
        A callable. Calling it with an unknown label or out-of-range
        timestamp returns ``None``.
    """
    s2s: dict[str, int] = dict(speaker_to_slot or {})
    if not s2s and dense_faces:
        s2s = _infer_speaker_to_slot_from_dense_faces(dense_faces)

    sorted_dense: list = []
    if dense_faces:
        try:
            sorted_dense = sorted(
                dense_faces, key=lambda ff: float(getattr(ff, "timestamp", 0.0)),
            )
        except Exception:
            sorted_dense = list(dense_faces)

    fallback_y = _slot_y_default(face_registry, source_height)
    sw = max(1, int(source_width))

    def _resolve_slot_id(label: Optional[str]) -> Optional[int]:
        if label is None:
            return None
        explicit = _parse_explicit_slot_label(label)
        if explicit is not None:
            return explicit
        if label in s2s:
            return int(s2s[label])
        return None

    def _slot_fallback_position(slot_id: int) -> Optional[tuple[float, float]]:
        if face_registry is None:
            return None
        slot = None
        if hasattr(face_registry, "slot_by_id"):
            try:
                slot = face_registry.slot_by_id(slot_id)
            except Exception:
                slot = None
        if slot is None:
            slots = getattr(face_registry, "slots", None) or []
            for s in slots:
                if int(getattr(s, "slot_id", -1)) == int(slot_id):
                    slot = s
                    break
        if slot is None:
            return None
        x_pct = float(getattr(slot, "x_center", 50.0))
        return (x_pct / 100.0 * sw, fallback_y)

    def _nearest_dense_face(slot_id: int, t: float) -> Optional[tuple[float, float]]:
        if not sorted_dense:
            return None
        best_dt = float("inf")
        best_face = None
        # Linear scan — dense_faces is typically a few hundred entries
        # at <=2 fps. The freshness window prunes most of them.
        for ff in sorted_dense:
            ts = float(getattr(ff, "timestamp", 0.0))
            dt = abs(ts - t)
            if dt > freshness_window_s:
                continue
            for face in getattr(ff, "faces", []) or []:
                sid = int(getattr(face, "identity_id", -1))
                if sid != slot_id:
                    continue
                if dt < best_dt:
                    best_dt = dt
                    best_face = face
        if best_face is None:
            return None
        x_pct = float(getattr(best_face, "nose_x", 50.0))
        y_pct = float(getattr(best_face, "nose_y", 50.0))
        return (x_pct / 100.0 * sw, y_pct / 100.0 * source_height)

    def lookup(label: str, t: float) -> Optional[tuple[float, float]]:
        slot_id = _resolve_slot_id(label)
        if slot_id is None or slot_id < 0:
            return None
        fresh = _nearest_dense_face(slot_id, float(t))
        if fresh is not None:
            return fresh
        return _slot_fallback_position(slot_id)

    return lookup
