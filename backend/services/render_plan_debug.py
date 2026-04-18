"""Phase 10 debug payload builder for RenderPlan JSON.

The ``ReframeDebugOverlay`` frontend component (Phase 10) renders a
compact timeline strip showing:

- **Content routing chips** — ``content_type``, ``clip_content_type``,
  ``anime_subtype``, ``music_subtype``, ``gameplay_subtype``,
  ``game_type``, ``is_multi_speaker_panel``, ``is_animated``.
- **Per-segment editorial decisions** — the kind tags emitted by
  ``backend.services.editorial_prior`` (``j_cut`` / ``l_cut`` /
  ``listener_hold`` / ``reaction_beat``) indexed by target segment.
- **Per-segment reasons** — the ``ReframeSegment.fallback_reason``
  string that explains why the solver chose this crop
  (``anime_anchor_face`` / ``gameplay_tracker_motion`` /
  ``multi_region_lp_fit`` / ``music_pulse_cut`` /
  ``editorial_reaction_beat`` / ``wide_master_fallback`` / …).

All of these were previously only available in in-memory pipeline
state. This module is the single place that marshals them into a
JSON-safe dict attached to ``render_plan["debug"]`` at cache time, so
the overlay can light up without any additional plumbing.

The shape mirrors the (pre-existing) legacy debug fields used by the
overlay (``pacing_per_sec``, ``min_hold_per_sec``,
``confidence_per_segment``, ``fallback_reasons``) and augments them
with the Phase 10 fields. Every field is optional — the overlay
gracefully hides missing entries.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)


def build_debug_payload(
    *,
    job: Any = None,
    content_profile: Any = None,
    reframe_segments: Optional[Iterable[Any]] = None,
    editorial_report: Any = None,
    pacing_estimator: Any = None,
    normalized_override: Any = None,
) -> dict:
    """Build the Phase 10 debug payload dict.

    All keyword arguments are optional so the caller can pass only
    what's available at the call site. Missing fields are omitted
    from the result rather than set to ``None``, so the frontend's
    ``chip && chip(...)`` pattern hides them automatically.

    Args:
        job: ``JobResult`` (or ``SimpleNamespace``-like) — read for
            ``content_type_override`` / ``anime_subtype`` /
            ``music_subtype`` / ``game_type``.
        content_profile: ``ContentProfile`` from
            ``backend.services.content_classifier``. Read for
            ``content_type`` and the sub-type flags.
        reframe_segments: The final ``list[ReframeSegment]`` after
            all Stage 7-11 passes have run. Read for per-segment
            ``fallback_reason`` and ``confidence`` arrays.
        editorial_report: The optional ``EditorialApplyReport`` from
            Stage 9b. Its ``decisions`` list is turned into a
            per-segment ``list[list[str]]`` keyed by
            ``target_segment_idx``.
        pacing_estimator: The ``LocalPacingEstimator`` when available
            — its ``pacing_per_sec`` + ``min_hold_per_sec`` arrays
            populate the sparkline charts.
        normalized_override: A pre-computed ``NormalizedContentType``
            (from ``normalize_ui_content_type``). Optional — if not
            passed we compute it from ``job.content_type_override``.

    Returns:
        A JSON-serializable dict with the keys the frontend overlay
        reads. Never raises — any per-field failure is caught and
        logged as a warning so the pipeline's RenderPlan caching
        path stays hot.
    """
    debug: dict[str, Any] = {}

    # ── Header chips (content routing) ────────────────────────────
    try:
        _fill_content_routing(debug, job, content_profile, normalized_override)
    except Exception as e:  # pragma: no cover — defensive
        logger.debug("render_plan_debug: content routing fill failed: %s", e)

    # ── Per-segment arrays ────────────────────────────────────────
    try:
        _fill_per_segment(debug, reframe_segments)
    except Exception as e:  # pragma: no cover — defensive
        logger.debug("render_plan_debug: per-segment fill failed: %s", e)

    # ── Editorial prior decisions ─────────────────────────────────
    try:
        _fill_editorial(debug, editorial_report, reframe_segments)
    except Exception as e:  # pragma: no cover — defensive
        logger.debug("render_plan_debug: editorial fill failed: %s", e)

    # ── Pacing + min-hold sparklines ──────────────────────────────
    try:
        _fill_pacing(debug, pacing_estimator)
    except Exception as e:  # pragma: no cover — defensive
        logger.debug("render_plan_debug: pacing fill failed: %s", e)

    return debug


def _fill_content_routing(debug, job, content_profile, normalized_override):
    """Populate header-chip fields (content_type, subtypes, flags)."""
    # Normalized override is the preferred source — it already
    # reconciles the UI dropdown token with the enum + flags.
    if normalized_override is None and job is not None:
        try:
            from backend.services.content_type_strings import (
                normalize_ui_content_type,
            )
            _override_token = getattr(job, "content_type_override", "") or ""
            if _override_token:
                normalized_override = normalize_ui_content_type(_override_token)
        except Exception:
            normalized_override = None

    # content_type + flags from normalized override, falling back to
    # the classifier's ContentProfile if the user picked "auto".
    if normalized_override is not None:
        try:
            debug["content_type"] = normalized_override.content_type.value
        except Exception:
            pass
        if getattr(normalized_override, "is_multi_speaker_panel", False):
            debug["is_multi_speaker_panel"] = True
        if getattr(normalized_override, "is_animated", False):
            debug["is_animated"] = True
        if getattr(normalized_override, "gameplay_subtype", None):
            debug["gameplay_subtype"] = normalized_override.gameplay_subtype
    elif content_profile is not None:
        _ct = getattr(content_profile, "content_type", None)
        if _ct:
            # content_type on ContentProfile may be a string or enum
            debug["content_type"] = getattr(_ct, "value", _ct)
        if getattr(content_profile, "is_multi_speaker_panel", False):
            debug["is_multi_speaker_panel"] = True
        if getattr(content_profile, "is_animated", False):
            debug["is_animated"] = True
        _gps = getattr(content_profile, "gameplay_subtype", None)
        if _gps:
            debug["gameplay_subtype"] = _gps

    # Sub-type tokens — always prefer the job's explicit override
    # values since the classifier may have dropped them.
    if job is not None:
        _anime = getattr(job, "anime_subtype", "") or ""
        _music = getattr(job, "music_subtype", "") or ""
        _game = getattr(job, "game_type", "") or ""
        if _anime:
            debug["anime_subtype"] = _anime
        if _music:
            debug["music_subtype"] = _music
        if _game:
            debug["game_type"] = _game

    # clip_content_type is populated by classify_clip downstream —
    # the best signal we can grab here is the content_type on the
    # ContentProfile (when available). Leave it blank otherwise; the
    # overlay falls back to the header content_type chip.
    if content_profile is not None:
        _clip = getattr(content_profile, "clip_content_type", None)
        if _clip:
            debug["clip_content_type"] = getattr(_clip, "value", _clip)


def _fill_per_segment(debug, reframe_segments):
    """Populate per-segment arrays from ``ReframeSegment`` objects."""
    if not reframe_segments:
        return

    segs = list(reframe_segments)
    if not segs:
        return

    confidences: list[float] = []
    reasons: list[Optional[str]] = []
    fallback_reasons_legacy: list[Optional[str]] = []

    for seg in segs:
        conf = getattr(seg, "confidence", 1.0)
        try:
            confidences.append(round(float(conf), 3))
        except (TypeError, ValueError):
            confidences.append(1.0)

        # ``reason_per_segment`` is the Phase 10 extension: any
        # non-empty explanation for why the solver chose this crop.
        # It's a superset of the legacy ``fallback_reasons`` field
        # (which only shows rescue/fallback tags in red).
        _fb = getattr(seg, "fallback_reason", None)
        _reason = getattr(seg, "reason", "") or ""
        _subj_src = getattr(seg, "subject_source", "") or ""

        # Prefer fallback_reason (most specific), then subject_source
        # (which the Phase 6-8 passes set to e.g. "anime_anchor_face"
        # / "gameplay_tracker_motion" / "editorial_reaction_beat"),
        # then fall back to the raw ``reason`` tag.
        if _fb:
            reasons.append(str(_fb))
        elif _subj_src and _subj_src not in {"active_speaker", "face_registry"}:
            reasons.append(str(_subj_src))
        else:
            reasons.append(None)

        # Legacy fallback_reasons array — kept for backwards compat
        # with the existing overlay code path. Populated when the
        # segment was routed to a fallback strategy.
        if _fb:
            fallback_reasons_legacy.append(str(_fb))
        elif any(tok in _reason for tok in ("fallback", "wide_master", "blur_fill")):
            fallback_reasons_legacy.append(_reason)
        else:
            fallback_reasons_legacy.append(None)

    debug["confidence_per_segment"] = confidences
    if any(r for r in reasons):
        debug["reason_per_segment"] = reasons
    debug["fallback_reasons"] = fallback_reasons_legacy


def _fill_editorial(debug, editorial_report, reframe_segments):
    """Populate ``editorial_prior_per_segment`` from the Phase 8 report."""
    if editorial_report is None:
        return

    decisions = getattr(editorial_report, "decisions", None) or []
    if not decisions:
        return

    if not reframe_segments:
        return

    n = len(list(reframe_segments))
    if n == 0:
        return

    per_segment: list[list[str]] = [[] for _ in range(n)]

    for dec in decisions:
        idx = getattr(dec, "target_segment_idx", -1)
        kind = getattr(dec, "kind", "") or ""
        if not kind:
            continue
        if 0 <= idx < n:
            per_segment[idx].append(kind)

    # Only emit if at least one segment carries a decision tag —
    # otherwise the overlay row would render empty slots.
    if any(per_segment):
        debug["editorial_prior_per_segment"] = per_segment

    # Summary counters — useful for the roll-up table in the docs.
    debug["editorial_summary"] = {
        "n_j_cuts": int(getattr(editorial_report, "n_j_cuts", 0)),
        "n_l_cuts": int(getattr(editorial_report, "n_l_cuts", 0)),
        "n_listener_holds": int(getattr(editorial_report, "n_listener_holds", 0)),
        "n_reaction_beats": int(getattr(editorial_report, "n_reaction_beats", 0)),
    }


def _fill_pacing(debug, pacing_estimator):
    """Populate pacing + min-hold sparklines from the estimator."""
    if pacing_estimator is None:
        return

    pacing = getattr(pacing_estimator, "pacing_per_sec", None)
    min_hold = getattr(pacing_estimator, "min_hold_per_sec", None)

    if pacing is not None:
        try:
            debug["pacing_per_sec"] = [round(float(v), 3) for v in pacing]
        except Exception:
            pass

    if min_hold is not None:
        try:
            debug["min_hold_per_sec"] = [round(float(v), 3) for v in min_hold]
        except Exception:
            pass
