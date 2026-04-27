# 2026 SOTA Reframing — Rollout ADR

**Status:** in-progress (Phase A–E shipped on `claude/clipai-sota-reframing-tOo41`).
**Date:** 2026-04-27
**Authors:** Claude Code (with Jalon)

---

## Context

Pre-Phase-A, the human-reframe pipeline was gated to **two content
types** (`multi_speaker_panel`, `talking_head`) by
`HUMAN_REFRAME_ALLOWED_CONTENT_TYPES`. Nine other content types
(podcast, vlog, narrative, music_video, sports, gaming, anime,
animation, tutorial) ran the legacy `reframe_segmenter`, which:

* Used OpenCV KCF / MOSSE / CSRT trackers (circa-2017, drift on
  occlusion).
* Had no global scene understanding — per-frame VLM batches at 13 s
  intervals, no temporal coherence.
* Had a 19-scalar MLP composition head that never sees pixels.
* Had no audio-visual saliency model — only a saliency *tracker*.
* Had no editorial planner — genre logic was scattered across 12
  helper modules.

The result: the human-quality path was unavailable for the majority
of content people upload. Tank vs. Tyrese-class panel content was
gated; everything else fell back to legacy.

## Decision

Implement five SOTA layers as independently shippable phases, each
behind a feature flag, and **remove the genre allowlist** once Phase E
is green:

| Phase | Module                          | Win                                          |
|-------|---------------------------------|----------------------------------------------|
| A     | `samurai_tracker`               | Pixel masks + occlusion-robust + distractor- |
|       |                                 | aware. SAM 2.1 Hiera-Tiny + motion-aware     |
|       |                                 | memory.                                      |
| B     | `cotracker3_dense`              | Dense per-pixel motion + camera-motion       |
|       |                                 | subtraction. Cuts jitter on panning sources. |
| C     | `av_saliency` + `ocr_regions`   | Gaze-aware framing + on-camera text          |
|       |                                 | protection. TASED-Net + PaddleOCR-lite.      |
| D     | `composition_head_clip`         | CLIP-conditioned framing prior. Replaces the |
|       |                                 | 19-scalar MLP. Drops face clipping ~50 %.    |
| E     | `editorial_planner`             | Single LLM call per clip emits a structured  |
|       |                                 | shot plan; LP solver consumes as soft prior. |
|       |                                 | Per-genre playbooks for talking_head,        |
|       |                                 | multi_speaker_panel, sports, gaming,         |
|       |                                 | music_video, narrative, anime, vlog,         |
|       |                                 | tutorial, default.                           |

Phase E is the keystone: with the editorial planner emitting a
per-genre playbook, the per-content-type code paths in
`genre_refinements.py` become the *implementation* of those playbooks
rather than the dispatch logic. The allowlist becomes redundant.

## Allowlist removal

```python
# Pre-Phase-E
HUMAN_REFRAME_ALLOWED_CONTENT_TYPES: frozenset[str] = frozenset({
    "multi_speaker_panel",
    "talking_head",
})

# Post-Phase-E
HUMAN_REFRAME_ALLOWED_CONTENT_TYPES: frozenset[str] = frozenset({
    "multi_speaker_panel", "talking_head",
    "podcast", "vlog", "interview",
    "narrative", "documentary", "cinematic_dialogue",
    "music_video", "concert", "performance",
    "sports", "sports_basketball", "sports_racing",
    "gaming", "gameplay",
    "anime", "animation", "animation_dialogue",
    "tutorial",
    "debate",
})
```

Rationale: the editorial planner's `GENRE_TO_TEMPLATE` now provides
the per-genre dispatch. Every entry in the new allowlist has a
matching prompt template in
`backend/services/editorial_planner_prompts/`.

## Config-flag changes

| Flag                                  | Pre-Phase-E | Post-Phase-E |
|---------------------------------------|-------------|--------------|
| `CLIPAI_TRACKER_BACKEND`              | n/a         | `auto`       |
| `CLIPAI_DENSE_POINT_TRACKING`         | n/a         | `True`       |
| `CLIPAI_SALIENCY_ENABLED`             | n/a         | `True`       |
| `CLIPAI_COMPOSITION_HEAD`             | n/a         | `clip`       |
| `CLIPAI_EDITORIAL_PLANNER`            | n/a         | `True`       |
| `CLIPAI_LEGACY_REFRAME` (rollback)    | n/a         | `False`      |

## Emergency rollback

If a regression is discovered post-deploy:

```bash
# Force the legacy reframe_segmenter path for one release.
export CLIPAI_LEGACY_REFRAME=1
docker compose down && docker compose up -d
```

Or for a per-phase rollback (e.g. trained head broken):

```bash
# Roll back to the legacy 19-scalar composition head.
export CLIPAI_COMPOSITION_HEAD=legacy
```

The flags are independent — you can keep SAMURAI tracking + the
editorial planner while reverting the CLIP head, etc.

## What's gone

* The pre-Phase-A genre allowlist (now widened).
* The OpenCV trackers as the *default* tracking backend (now the
  fallback path for no-GPU machines).
* The 19-scalar MLP composition head as the *default* prior (now the
  rollback option behind `CLIPAI_COMPOSITION_HEAD=legacy`).

## What's preserved

* Every existing module in `backend/services/` — none deleted.
  `composition_head.py`, `tracker_wrapper_opencv.py`,
  `reframe_segmenter.py` remain importable for the rollback path.
* Every existing parity metric. Phase A added four new ones
  (`identity_switch_count`, `saliency_in_crop_fraction`,
  `face_clipping_rate`, `text_region_clipping_rate`) — the existing
  eight (`sub_second_switch_recall`, `overlap_count`,
  `max_acceleration`, `max_jerk`, …) are unchanged and still scored.
* The bench harness (`compare_autoflip_vs_clipai.py`) — every Phase
  reports against the same fixture set.

## Validation gate

A Phase is shippable iff:

1. Its dedicated `tests/qa/test_phase_<x>_*.py` suite passes locally
   (every Claude Code Opus 4.7 worker runs this before pushing).
2. The homelab bench passes on every clip in
   `tests/real_content/manifest.json` — no clip regresses by more
   than 5 % on any pre-existing parity metric.
3. The Phase's primary new metric meets its target:
   * Phase A: `identity_switch_count` ≥ 30 % reduction on 4-speaker
     panel fixtures.
   * Phase B: `max_jerk` ≥ 40 % reduction on panning-source
     fixtures.
   * Phase C: `saliency_in_crop_fraction` ≥ 0.80 on dynamic content.
   * Phase D: `face_clipping_rate` ≥ 50 % reduction.
   * Phase E: aggregate quality eyeball-pass on every genre fixture.

## Done definition

After Phase E ships:

* A 10-minute Tank vs. Tyrese podcast completes in ~10 minutes wall
  time on the homelab GTX 1650.
* The 9:16 output tracks the active speaker tightly, reaction beats
  catch the laughing host on cue, headroom is appropriate, identity
  never switches mid-shot, wide-master moments feel motivated.
* The same pipeline produces human-quality output on basketball,
  gameplay, music video, anime, narrative, vlog, tutorial fixtures.
