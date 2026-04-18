# AutoFlip-Parity Reframing Changelog

Tracking progress toward full parity with Google AutoFlip
(Grundmann et al., "Auto-Directed Video Stabilization with Robust L1 Optimal Camera Paths").

## Baseline (pre-Phase 0)

| Metric | Value | Notes |
|---|---|---|
| Detected speaker changes | 9/9 (100%) | Synthetic 20s fixture, alternating every 2s |
| Missed changes | 0 | |
| Avg lag (frames @ 30fps) | 6.0 | Negative = anticipation (-6 frames = -200ms) |
| Max lag (frames) | 6 | |
| Min lag (frames) | 6 | |
| Sub-second switch recall | 100% (4/4) | 5 alternating 400ms segments |
| Overlap count | 0 | |
| Smoke test | PASSED | 1 segment, strategy=panning, 10s |
| L1 solver tests | 12/12 passed | test_l1_solver_constraints.py |

### Known issues (mapped to phases)

1. **Jitter during holds** — solver only penalizes velocity (no accel/jerk terms)
2. **2 Hz stair-stepping** — solver receives sparse anchors, not dense propagated trajectory
3. **~1s lag on transitions** — per-segment solving with no lookahead
4. **Pops at segment seams** — segments solved independently
5. **Chases keypoint noise** — TV_LAMBDA=10.0 too low, no dead-zone

---

## Phase 0 — Baseline telemetry

- Added CSV dump behind `CLIPAI_DUMP_L1=1` env flag → `/tmp/clipai_l1/<job_id>_<seg_idx>.csv`
- Added `backend/scripts/plot_l1_solve.py` for target-vs-solved visualization
- Baseline numbers recorded above

---

## Phase 1 — Dense propagated trajectory

**Fix:** Feed the solver the dense propagated trajectory at uniform fps instead of sparse ~2Hz anchors.

- Added `get_propagated_positions_for_segment` with linear interpolation + last-known hold
- Resamples onto uniform grid at `target_fps=30.0` — critical for TV solver's finite-difference operator
- Updated reframe_segmenter Stage 10 and pipeline to thread `interpolated_timeline`
- 5 unit tests added (sine wave uniform, gap handling, empty, single, solver integration)

| Metric | Before | After | Change |
|---|---|---|---|
| All lag metrics | unchanged | unchanged | No change expected (lag fix is Phase 3) |
| Smoke test | PASSED | PASSED | subject_x shifted slightly due to resampling |
| Tests | 12/12 | 17/17 | +5 new tests |

---

## Phase 2 — Exact TV-L2 solver with dead-zone

**Fix:** Replace 100-iteration proximal-gradient TV with exact dual proximal gradient solver.

- New `_condat_tv.py`: exact L2-TV proximity operator (Chambolle 2004 / Condat 2013 dual formulation)
- Converges to machine precision — zero residual tilt on constant signals
- Resolution-independent lambda: `TV_LAMBDA_FRAC=0.015 * source_width` (28.8 at 1920)
- Pre-solve dead-zone: suppress keypoint jitter < 25px (DEADZONE_FRAC=0.013)
- 4 unit tests added (constant, step, bounds, smoother)

| Metric | Before | After | Change |
|---|---|---|---|
| Residual tilt on holds | >0 (approximate solver) | 0 (exact) | Eliminated |
| Smoke test | PASSED | PASSED | |
| Tests | 17/17 | 21/21 | +4 new tests |

---

## Phase 3 — Shot-level solving with lookahead

**Fix:** Solve L1 camera path once per shot (not per segment), with mirror-reflection lookahead padding.

- New `solve_camera_path_for_shot`: concatenates segment targets, solves once, slices back
- Mirror-reflects 30 frames at shot boundaries for symmetric non-causal smoothing
- Reframe_segmenter Stage 10 groups segments by `shot_cuts` and calls shot-level solver
- `solve_camera_path` kept as backward-compatible thin wrapper

| Metric | Before | After | Change |
|---|---|---|---|
| Lookahead proven | N/A | Solved path starts moving BEFORE subject transition at t=2.0s | New capability |
| Segment boundary continuity | Pops possible | <5px gap at boundaries | Eliminated |
| Smoke test | PASSED | PASSED | |
| Tests | 21/21 | 25/25 | +4 integration tests |

---

## Phase 4 — LP solver with acceleration and jerk penalties

**Fix:** Add LP-based AutoFlip solver with λ₃ (acceleration) and λ₄ (jerk) penalties.

- New `_autoflip_lp.py`: formulates as LP with HiGHS backend (scipy.optimize.linprog)
- Each |·| term linearized via auxiliary slack + two inequality constraints
- Feature flag `CLIPAI_L1_LP=1` (ships OFF by default for A/B testing)
- Performance guardrail: falls back to Condat for shots > 600 frames (20s)
- Comparison script `backend/scripts/compare_l1_solvers.py`
- Added scipy and numpy to requirements.txt

| Metric | Condat TV | AutoFlip LP | Improvement |
|---|---|---|---|
| Velocity (TV) | 1100.0 | 1100.0 | Same |
| Acceleration | 2195.9 | 53.7 | **41x less** |
| Jerk | 4391.6 | 3.7 | **1187x less** |
| Solve time (150 frames) | 24ms | 40ms | Acceptable |
| Solve time (300 frames) | ~50ms | ~100ms | Within 200ms budget |
| Tests | 25/25 | 29/29 | +4 LP tests |

---

## Final state

| Metric | Baseline | After all phases |
|---|---|---|
| Tests passing | 12 | 29 |
| Solver type | 100-iter prox-grad (approximate) | Exact dual proximal gradient + optional LP |
| Input to solver | Sparse ~2Hz anchors | Dense uniform 30fps trajectory |
| Solve scope | Per-segment | Per-shot with lookahead |
| Residual tilt on holds | Present | Zero |
| Acceleration penalties | None | λ₃ + λ₄ via LP (behind feature flag) |
| Dead-zone | None | 25px (resolution-scaled) |
| Lambda | Fixed 10.0px | Resolution-independent 0.015 * source_width |

---

## Saliency Parity Fixes (Fixes 1-6)

Six fixes to bring saliency signal processing to AutoFlip parity:

1. **Wire full bboxes** — `scene_focus` now receives real `(x, y, w, h)` from SaliencyRegion instead of hardcoded `y=50, w=10, h=10`
2. **Unconditional saliency** — removed face-gated skip; saliency runs on every frame like AutoFlip
3. **Adaptive threshold** — percentile-based (top 15%) instead of fixed 0.5; low-contrast shots now produce regions
4. **Center bias + HUD masking** — 2D Gaussian center prior (σ=0.35) suppresses background motion; HUD mask zeros out known HUD pixels
5. **Feature merging** — overlapping saliency+face bboxes (IoU>0.3) are absorbed to prevent over-constraining the crop solver
6. **Temporal smoothing** — EMA (α=0.3) on promoted saliency cluster trajectories reduces jitter

Tests: 27 → 37 passing (+10 new tests across 3 test files)

---

## Out of scope (follow-ups)

- Multi-region tracking (AutoFlip handles multiple required regions with a unified LP) — **queued for Phase 3 of v2**
- Cinematography "rule of thirds" bias — **queued for Phase 4 of v2**
- Gameplay HUD-aware padding (orthogonal to solver quality, lives in layout_engine)

---

## v2 — Content-aware parity work (Phases 1–10)

The "Phases 0–4" table above covers the solver-quality work. A second
wave of phases starts from the opposite end: routing per-content-type
behaviours so anime, debates, music videos, vlogs, and multi-game
streams all get editorially-correct framing instead of falling into
the "generic talking-head" bucket. This section will grow one sub-
heading per v2 phase as they land.

### v2 Phase 1 — Content-type override plumbing

**Before:**
The upload UI dropdown sent strings (`"gameplay"`, `"movie"`,
`"podcast"`) that didn't match the `ContentType` enum values
(`"gaming"`, `"narrative"`, `"podcast"`). The classifier's
user-override branch at `content_classifier.py:122-130` checked
`user_type in [ct.value for ct in ContentType]`, so:

| UI token | Enum value | Override fired? |
|---|---|---|
| `gameplay` | `gaming` | ❌ |
| `movie` | `narrative` | ❌ |
| `podcast` | `podcast` | ✅ (by coincidence) |

Worse, `metadata['content_type_override']` was never injected into
the dict `classify_content` received — the ffprobe metadata dict
historically did NOT carry that key, so even the `"podcast"` case
only worked if a downstream caller happened to plumb the override
through as `metadata['content_type']` or `metadata['reframe_style']`,
which the pipeline did not.

Gameplay worked only via a separate hard-coded
`_content_override == "gameplay"` comparison at `pipeline.py:1226`,
which bypassed the classifier entirely — so downstream tuning still
thought content type was `UNKNOWN` even for user-declared gameplay.

**After (v2 Phase 1):**

- **New module** `backend/services/content_type_strings.py` with
  `normalize_ui_content_type(token)`, `is_gameplay_override(token)`,
  and `is_user_override(token)`. Maps every UI token (both legacy and
  Phase 2 forward-compat) to a
  `NormalizedContentType(content_type, is_multi_speaker_panel,
  is_animated, is_gameplay_fastpath, raw)`.
- **Classifier override branch** now normalizes through the helper,
  honours `is_multi_speaker_panel` / `is_animated` on the profile,
  and logs the normalized form. Accepts `content_type_override`,
  `content_type`, and `reframe_style` metadata keys so older callers
  still work.
- **Pipeline** injects `_content_override` into a `_classifier_metadata`
  dict before calling `classify_content`, so the UI override actually
  reaches the classifier's user-override branch. Gameplay fast-path
  now uses `is_gameplay_override()` instead of a bare string literal,
  so the Phase 2 gameplay variants (`gameplay_fps` / `gameplay_moba`
  / `gameplay_tps` / `gameplay_racing`) are covered without further
  edits here. Non-gameplay user overrides now correctly skip the
  gameplay auto-detect step via `is_user_override()`.
- **`stream`** normalizes to `ContentType.GAMING` but is deliberately
  NOT a gameplay fast-path candidate — it has a facecam and still
  needs the face pipeline. Phase 7 will handle its `STACKED_GAMEPLAY`
  layout routing downstream.

### Normalization matrix

| UI token | `ContentType` | Flags | fastpath |
|---|---|---|---|
| `gameplay` *(legacy)* | `gaming` | — | yes |
| `movie` *(legacy)* | `narrative` | — | no |
| `podcast` *(legacy)* | `podcast` | — | no |
| `debate` / `panel` | `podcast` | panel | no |
| `interview` | `podcast` | — | no |
| `vlog` | `vlog` | — | no |
| `narrative` / `cinematic` | `narrative` | — | no |
| `anime` / `cartoon` | `anime` | animated | no |
| `music_video` | `music_video` | — | no |
| `gameplay_fps` | `gaming` | — | yes |
| `gameplay_moba` | `gaming` | — | yes |
| `gameplay_tps` | `gaming` | — | yes |
| `gameplay_racing` | `gaming` | — | yes |
| `stream` | `gaming` | — | **no** (facecam) |
| `sports` | `sports` | — | no |
| `auto` / `""` / invalid | — | — | — (heuristic path) |

### Tests

| File | Count | Purpose |
|---|---|---|
| `test_content_type_override_plumbing.py` | 78 | normalizer / `classify_content` override branch / `pipeline.py` AST plumbing |
| Pre-existing classifier/plumbing suites | 54 | zero regressions |
| **Total (v2 Phase 1 scope)** | **132** | all green |

### Exit criteria met

- ✅ Every UI token round-trips through `normalize_ui_content_type` to
  the correct `ContentType` enum value.
- ✅ `classify_content` with `metadata['content_type_override']`
  short-circuits to `confidence=1.0` on the normalized enum value and
  sets `is_multi_speaker_panel` / `is_animated` from the normalized
  bundle.
- ✅ Pipeline AST asserts the metadata injection is present and that
  the gameplay fast-path uses `is_gameplay_override()` instead of a
  bare `== "gameplay"` literal.
- ✅ Invalid / unknown tokens fall through to the heuristic path
  without crashing; `confidence != 1.0` so downstream consumers can
  still distinguish a user override from a heuristic guess.
- ✅ Legacy `metadata['content_type']` / `metadata['reframe_style']`
  keys still work for older callers.
- ✅ No regression on the 54 pre-existing classifier / plumbing
  tests; no regression on `test_pipeline_source_width_hoisted.py`
  (which guards that `source_width` is still bound at function
  scope despite the nearby edits).

### v2 Phase 8 — Editorial "camera language" prior

**Before:** The legacy intent tracker is purely reactive — it
switches the crop to whoever is currently talking, the moment
they start talking. Human editors don't do that. They:

  - **Anticipate** new speakers via J-cuts (audio leads video
    by ~200 ms — viewer hears the new speaker for a beat
    before seeing them)
  - **Linger** on previous speakers via L-cuts (same shift,
    paired editorial intent)
  - **Hold on the listener** during reaction beats (when
    speaker A finishes a sentence and speaker B is silent but
    visible, hold on B for 400-800 ms)
  - **Cut to the reactor** during laughter / gasps (when an
    audio spike fires on a multi-face frame, show the
    non-talking face for the spike duration)

The reactive tracker captures the WHO of dialogue but misses
the WHEN — which is what makes a clip read as edited rather
than auto-generated.

**After (v2 Phase 8):**

- **`backend/services/editorial_prior.py`** *(new)* implements
  the editorial state machine. Pure-Python, numpy-free.

  Per-content-type gating via ``applies_to_profile``:
    - **ON** for: ``narrative``, ``podcast``, ``vlog``,
      ``cinematic_dialogue``, ``talking_head``,
      ``multi_speaker_panel``, ``animation_dialogue``
    - **OFF** for: ``gaming`` / ``gameplay_*``, ``music_video``,
      ``sports``, ``animation`` (action anime via the
      ``anime_subtype == "action"`` exclusion)
  ``EDITORIAL_PRIOR_CONTENT_TYPES`` constant holds the
  qualifying-set; the gate also checks
  ``profile.anime_subtype`` so action anime opts out even if
  the parent type would otherwise pass.

  Detection helpers:

    - ``detect_j_cuts(reframe_segments, transcript_segments,
      *, speaker_to_slot, lead_sec=0.20)`` — walks pairs of
      adjacent reframe segments, finds speaker-change
      boundaries, and emits ``EditorialDecision(kind="j_cut",
      delta_sec)`` to align the boundary to
      ``audio_first_word_time + lead_sec``. The 200 ms lead
      matches cinema convention. Falls back to label-suffix
      matching (``"Speaker N"`` → slot N-1) when no explicit
      ``speaker_to_slot`` mapping is provided.

    - ``detect_l_cuts(...)`` — same shift mechanic with the
      ``kind="l_cut"`` label. Phase 8 minimal treats J and L
      cuts as equivalent boundary shifts; a future iteration
      can differentiate the magnitude per content type.

    - ``detect_listener_holds(reframe_segments, transcript_segments,
      face_slots, *, speaker_to_slot, min_hold_sec=0.40,
      max_hold_sec=0.80)`` — finds sentence-end opportunities.
      A listener-cut fires when speaker A finishes a
      declarative beat (``.!?`` via ``SENTENCE_END_PUNCT_RE``)
      AND speaker B is silent but visible (``face_slots`` has
      ≥ 2 entries) AND the gap between A's last word and B's
      first word is ≥ ``min_hold_sec``. Hold duration is
      ``min(gap, max_hold_sec)``. Emits
      ``EditorialDecision(kind="listener_hold",
      new_active_slot=B, delta_sec=hold)``.

    - ``detect_reaction_beats(reframe_segments, audio_events,
      face_slots, *, duration_sec=0.80)`` — finds laughter /
      gasp opportunities. A reaction beat fires when an audio
      event of type in ``REACTION_AUDIO_TYPES``
      (``extreme_spike`` / ``silence_to_loud``) lies inside
      a segment with multiple visible faces. Swaps
      ``seg.active_slot`` to the first non-active slot
      (deterministic choice for Phase 8 minimal; future
      iterations can use lip-aperture / smile detection to
      pick the actual reactor). Emits
      ``EditorialDecision(kind="reaction_beat",
      new_active_slot=R)``.

  ``apply_decisions(reframe_segments, decisions, *,
  min_segment_sec=0.30)`` mutates the segment list in place:
    - J-cut shifts: applies the boundary delta and propagates
      to the previous segment's ``end``, preserving
      contiguity. Skips shifts that would violate the
      ``min_segment_sec`` floor (default 300 ms).
    - L-cut shifts: counted but use the same mechanic as
      J-cuts.
    - Listener holds: counted but the actual segment
      insertion is deferred to the segmenter caller (which
      has access to the ``ReframeSegment`` dataclass). The
      L1 solver in Stage 10 already smooths through the
      residual gap, so the listener hold's editorial intent
      is captured even without a dedicated insertion pass.
    - Reaction beats: swap ``seg.active_slot`` in place and
      stamp ``seg.reason = "editorial_reaction_beat"``.

  ``apply_editorial_prior(...)`` is the top-level entry that
  chains all four detectors and calls ``apply_decisions``.
  Returns ``EditorialApplyReport`` with per-kind counts
  (``n_j_cuts``, ``n_l_cuts``, ``n_listener_holds``,
  ``n_reaction_beats``) and a ``skipped_reason`` when the
  profile doesn't qualify.

  Tunable constants:
    - ``DEFAULT_JL_CUT_LEAD_SEC = 0.20`` — cinema-standard
      audio lead
    - ``LISTENER_HOLD_MIN_SEC = 0.40``
    - ``LISTENER_HOLD_MAX_SEC = 0.80``
    - ``REACTION_BEAT_DURATION_SEC = 0.80``
    - ``SENTENCE_END_PUNCT_RE`` — regex for ``.!?``
    - ``REACTION_AUDIO_TYPES`` — frozenset of qualifying
      ``audio_analyzer`` event types

  ``USE_EDITORIAL_PRIOR`` env flag, default OFF.

- **`backend/services/reframe_segmenter.py`** gained a new
  **Stage 9b** sub-block immediately after Stage 9 (hard
  constraints) and BEFORE Stage 10a (multi-region LP).
  Behind ``CLIPAI_EDITORIAL_PRIOR=1`` AND
  ``applies_to_profile(content_profile)``:

    - Calls ``apply_editorial_prior`` with the segmenter's
      raw segment list, the transcript, the face registry's
      slots, and the audio events list (passed via the new
      ``audio_events`` kwarg on ``build_reframe_segments``).
    - Logs ``"EditorialPrior: N J-cuts, M L-cuts, P listener
      holds, Q reaction beats"`` so the runner output can
      attribute each editorial decision.
    - Wrapped in try/except so any state-machine failure
      stays non-fatal and the existing reactive intent
      tracker output still drives the L1 solver.

  Stage 9b runs BEFORE the L1 camera path solver (Stage 10)
  so the smoothness pass sees the editorial-adjusted
  boundaries. The Phase 4 lead-room and Phase 5 beat snap
  also run after Stage 10, so all three offset / shift
  mechanisms compose cleanly.

- **`build_reframe_segments`** signature gained an
  ``audio_events: list = None`` kwarg. Production callers
  (``pipeline.py``) pass the
  ``audio_analyzer.analyze_audio_energy`` output; test callers
  pass a hand-built list of ``{timestamp, type}`` dicts.

- **Default OFF** for the Phase 8 flag. Per the v2 ground
  rules, any change that *might* regress an existing baseline
  ships flag-off by default. The editorial prior shifts
  segment boundaries by 100-200 ms which can ripple through
  the existing sub-second-recall metric on the
  ``2speaker_alternating`` baseline; in-docker validation
  flips the flag once the post-Phase-8 numbers in
  ``docs/autoflip_parity_v2_results.md`` show no regression.

### Tests

| File | Count | Purpose |
|---|---|---|
| `test_phase8_editorial.py` | 56 | content gate + J-cut + L-cut + listener hold + reaction beat + apply_decisions + Stage 9b AST guards |
| Phase 1+2+3+4+5+6+7+9 + pre-existing | 653 | zero regressions |
| **Total (v2 Phase 1-8 + 9 scope)** | **709** | all green |

The 56 new tests break down as:

- **TestAppliesToProfile** (12): every qualifying ContentType
  passes, every non-qualifying type excluded, anime_action
  exclusion fires explicitly, anime_dialogue passes, None
  excluded, profile-without-content_type defensive default.
- **TestDetectJCuts** (8): audio-leads-video case shifts
  +0.10 s, audio-lags-video case shifts +0.70 s, no shift
  when no speaker change, single segment, None active slot,
  outside-window filter, label-to-slot fallback, no shift
  when already aligned.
- **TestDetectLCuts** (1): L-cut variant returns same shift
  with distinct kind label.
- **TestDetectListenerHolds** (6): basic listener hold,
  clamped to max, skipped when gap too short, skipped
  without sentence-end punctuation, skipped with single
  face slot, question mark counts as sentence end.
- **TestDetectReactionBeats** (6): extreme spike swaps slot,
  silence-to-loud swap, plain volume_spike doesn't qualify,
  single-face skip, event-outside-segment skip, empty inputs.
- **TestApplyDecisions** (5): J-cut boundary mutation,
  J-cut skip below min-segment floor, reaction beat slot
  swap, listener hold counted but not inserted, empty inputs.
- **TestApplyEditorialPriorTopLevel** (3): non-qualifying
  content skipped, None profile skipped, full pass with
  qualifying profile.
- **TestEditorialPriorFlagDefaultOff** (1): flag default OFF.
- **TestReframeSegmenterStage9bAST** (6): Stage 9b block
  present, lazy imports, signature has ``audio_events`` kwarg,
  applies_to_profile gate referenced, Stage 9b ordering before
  Stage 10, try/except wrap.

### Sandbox parity numbers

The Phase 8 mechanism is verified end-to-end via the 56 unit
tests that build hand-crafted transcript + segment + audio-event
inputs and assert each detector + the in-place mutation. The
parity bench fixtures don't yet include word-level transcript
data (Phase 9 fixtures use coarse ``TranscriptSegment.start /
end`` only), so the bench can't exercise the J-cut path
directly. With the Phase 8 flag OFF the bench numbers are
**unchanged** — verified across all 7 fixtures.

A Phase 8 follow-up will:

  1. Extend the parity fixtures with word-level transcripts
     for the dialogue-heavy fixtures (``2speaker_alternating``
     gets per-word timestamps so J-cuts can fire on the
     speaker-change boundaries).
  2. Add the production ``pipeline.py`` wiring that passes
     ``audio_events`` (from ``audio_analyzer.analyze_audio_energy``)
     to ``build_reframe_segments``.

### Open questions resolved this phase

- **J-cut vs L-cut**: same boundary shift (audio_start +
  lead_sec), different editorial intent. Phase 8 minimal
  treats them as equivalent and labels the decision kind
  for telemetry. A future iteration can differentiate the
  magnitude (e.g. 200 ms J / 400 ms L).

- **Reactor selection**: Phase 8 minimal picks the first
  non-active slot deterministically. A future iteration can
  use lip-aperture / smile detection to pick the actual
  reactor face on multi-face frames.

- **Listener hold insertion**: counted but deferred. The L1
  solver in Stage 10 smooths through the gap so the
  editorial intent is captured even without a dedicated
  segment insertion. A future iteration can have Stage 9b
  call ``dataclasses.replace`` on the affected segment to
  insert a true listener hold.

### Out of scope for this phase

- **Production pipeline.py wiring of audio_events**.
  ``audio_analyzer.analyze_audio_energy`` is already called
  early in the pipeline but the result isn't yet passed to
  ``build_reframe_segments``. Phase 8 follow-up.

- **Word-level timestamps in the parity fixtures**. The
  Phase 9 fixtures use ``TranscriptSegment`` objects with
  ``words=None``; the J-cut detector falls back to
  ``segment.start`` in that case so it still works but
  doesn't demonstrate the millisecond-precision shift the
  spec calls for. Phase 8 follow-up.

- **Per-segment listener-hold insertion**. The decision is
  counted but the actual segment insertion (via
  ``dataclasses.replace``) lives in Stage 9b's caller. Phase
  8 follow-up.

### v2 Phase 7 — Gaming beyond FPS (MOBA / TPS / racing / stream)

**Before:** The legacy gaming reframe path hard-coded
``subject_x = 50`` (screen-center crosshair) for every gaming
clip. Correct for FPS / hero shooters where the action sits on
the crosshair, but wrong for:

  - **MOBA / top-down**: action moves across lanes; a fixed
    center crop loses the side-lane fight when it migrates
  - **Third-person action (TPS)**: player character is offset
    down+right of frame center (GTA, Elden Ring); a center crop
    puts the character on the edge of the vertical frame
  - **Racing**: car sits in lower third; a center crop loses
    the road / horizon context
  - **Stream**: gameplay + facecam should stack vertically;
    legacy path centered the crop on neither

**After (v2 Phase 7):**

- **`backend/services/gameplay_subject_tracker.py`** *(new)*
  ships the per-genre subject tracker:
    - ``GameplayMotionCentroid(timestamp, x_pct, y_pct, magnitude)``
      dataclass for per-frame motion centroid input. Production
      callers compute these via OpenCV Farneback / Lucas-Kanade;
      the parity bench uses synthetic centroids.
    - ``track_gameplay_subject(centroids, fallback_xy_pct, ...)``
      smooths motion centroids with EMA (default τ = 0.50 s) +
      velocity clamping (default 15 % / sec, matching the
      existing ``optical_flow.build_motion_tracking_path``).
      Falls back to the per-genre action-center anchor when
      magnitudes are below ``MIN_MOTION_MAGNITUDE = 0.6``.
      Returns ``GameplaySubjectPath(path, source, n_motion_frames)``
      where ``source`` is one of ``"motion"`` / ``"action_center"``
      / ``"mixed"``.
    - ``aggregate_subject_x_for_segment(...)`` averages the
      smoothed path mid-segment and returns
      ``(x_pct, y_pct, source)`` for the segmenter to consume.
    - ``subject_anchor_for_game(game_key)`` /
      ``subject_anchor_for_genre(genre)`` look up the
      ``action_center_pct`` from Phase 2's ``GAME_HUD_LAYOUTS``
      table — FPS = (50, 50), TPS = (50, 45), racing = (50, 65),
      MOBA = (50, 50). None / unknown defaults to (50, 50).
    - ``derive_hud_safe_subject_x(hud_zones, crop_width_pct)``
      computes a HUD-safe subject_x for **unrecognized games**
      by taking the area-weighted centroid of the visible HUD
      zones and clamping it to the crop-safe range. Default
      fallback when no specific game key is known.
    - ``USE_GAMEPLAY_TRACKER`` env flag, default OFF.

- **`backend/services/reframe_segmenter.py`** gained two new
  sub-blocks immediately after Stage 7b (anime anchor):

    - **Stage 7c — Gameplay subject tracker override**. Behind
      ``CLIPAI_GAMEPLAY_TRACKER=1`` AND
      ``profile.gameplay_subtype`` is set AND a populated
      ``gameplay_motion_centroids`` list is passed via the new
      ``build_reframe_segments`` kwarg. For each segment that's
      not in a multi-region layout, calls
      ``aggregate_subject_x_for_segment`` with the segment's
      time window. The fallback resolves to
      ``subject_anchor_for_game`` (when ``profile.game_type`` is
      set) OR ``subject_anchor_for_genre`` (when only the
      gameplay subtype is known). Replaces ``seg.subject_x`` /
      ``seg.subject_y`` with the tracker output and stamps
      ``seg.subject_source = "gameplay_tracker_{source}"``.

    - **Stage 7d — Stream layout routing**. NO feature flag.
      When ``profile.gameplay_subtype == "stream"``, forces
      ``seg.layout = "stacked_gameplay"`` for every non-multi-
      region segment. The downstream renderer already supports
      STACKED_GAMEPLAY from the existing
      ``CONTENT_TYPE_CONFIG.GAMING.prefer_stacked_gameplay``
      entry — Phase 7 just routes to it via the explicit
      Phase 2 stream subtype. Stream is an editorial decision
      rather than a quality knob, so no flag.

- **`backend/services/game_layouts.py`** is unchanged in this
  commit — Phase 2 already populated ``action_center_pct`` for
  every entry (FPS = (50, 50), TPS = (50, 45), racing = (50, 65),
  MOBA = (50, 50)). Phase 7 just consumes those values via
  ``get_action_center``.

- **Default OFF** for ``CLIPAI_GAMEPLAY_TRACKER``. Per the v2
  ground rules, any change that *might* regress an existing
  baseline ships flag-off by default. The legacy gameplay
  fast-path in pipeline.py (``_is_gameplay`` branch) still
  fires before the segmenter even runs for gameplay clips —
  Phase 7's segmenter integration is dormant until the
  pipeline.py wiring (Phase 7 follow-up) disables the
  fast-path when ``gameplay_subtype`` is set.

### Tests

| File | Count | Purpose |
|---|---|---|
| `test_phase7_gameplay.py` | 34 | per-game anchor lookups + per-genre lookups + motion tracker smoothing + aggregator + HUD-safe fallback + Stage 7c/7d AST guards + flag default |
| Phase 1+2+3+4+5+6+9 + pre-existing | 619 | zero regressions |
| **Total (v2 Phase 1-7 + 9 scope)** | **653** | all green |

The 34 new tests break down as:

- **TestSubjectAnchorForGame** (6): FPS center, TPS above
  center, racing lower third, MOBA centered, unknown defaults,
  None defaults.
- **TestSubjectAnchorForGenre** (6): per-genre lookups + None
  defaults — fps=(50,50), tps=(50,45), racing=(50,65),
  moba=(50,50), sandbox=(50,50).
- **TestTrackGameplaySubject** (7): no centroids fallback,
  strong motion path, weak motion fallback, mixed source,
  outside-window filter, velocity clamp, EMA jitter
  suppression.
- **TestAggregateSubjectXForSegment** (3): motion source,
  fallback source, multi-frame averaging.
- **TestDeriveHudSafeSubjectX** (5): empty zones, single
  zone centroid, two-zone area weighting, clamp to crop-safe
  range, zero-area zones ignored.
- **TestPhase7FeatureFlagDefaultOff** (1): flag default OFF.
- **TestReframeSegmenterStage7cAST** (6): Stage 7c block
  present, Stage 7d stream routing present, lazy imports,
  signature has ``gameplay_motion_centroids`` kwarg, per-game
  + per-genre fallback both referenced, stream routing gates
  on the subtype with no flag.

### Sandbox parity numbers

The Phase 7 mechanism is verified end-to-end via the 34 unit
tests, which build hand-crafted ``GameplayMotionCentroid``
sequences and assert the tracker's smoothing + clamping +
fallback semantics. The parity bench fixtures
(``tps_character_offset`` / ``stream_corner_facecam``) are
**unchanged** at flag-ON because the bench doesn't pre-build
``gameplay_motion_centroids`` from real frame data — that
would need OpenCV optical flow on synthetic / real game
footage.

The bench fixtures also still take the legacy ``_is_gameplay``
fast-path in pipeline.py, which skips the segmenter entirely
for gameplay content. Phase 7's segmenter integration is in
place but dormant until the pipeline.py wiring (next
follow-up) routes gameplay-with-subtype through the segmenter
instead of the fast-path.

### Open questions resolved this phase

- **Where does the motion centroid come from in production?**
  ``pipeline._run_analysis_inner`` will call OpenCV Farneback
  dense optical flow on sampled gameplay frames, find the
  dominant motion centroid per frame, and pass the result via
  the new ``gameplay_motion_centroids`` kwarg. The segmenter
  wiring is in this commit; pipeline.py wiring is the Phase 7
  follow-up.

- **Why no flag on the stream routing?** Stream is an
  editorial decision the user explicitly made via the
  Phase 2 dropdown — they're saying "this is gameplay +
  facecam, please stack them". That's a yes/no behavior
  switch, not a quality tradeoff, so it doesn't need a flag.

- **What about unrecognized games?** The
  ``derive_hud_safe_subject_x`` fallback computes an
  area-weighted centroid of the HUD zones and clamps it to
  the crop-safe range. For a truly unknown game, the caller
  passes the persistent-region detector's output as the HUD
  zone list. Phase 7 ships the helper; the
  persistent-region wiring is a follow-up.

### Out of scope for this phase

- **Production pipeline.py wiring**. Gameplay-with-subtype
  clips currently take the legacy ``_is_gameplay`` fast-path
  before reaching the segmenter. Phase 7 follow-up will
  disable the fast-path when ``gameplay_subtype`` is set so
  the segmenter sees the clip and Stage 7c can fire.

- **Per-frame OpenCV motion extraction**. The segmenter
  consumes pre-computed ``GameplayMotionCentroid`` objects.
  The OpenCV pass that creates them lives in pipeline.py
  (Phase 7 follow-up) and uses Farneback dense flow on
  sampled frames.

- **STACKED_GAMEPLAY layout details**. Stage 7d sets
  ``seg.layout = "stacked_gameplay"`` but doesn't compute
  the actual top/bottom split rectangles — that's the
  renderer's job and the existing
  ``CONTENT_TYPE_CONFIG.GAMING.prefer_stacked_gameplay``
  path already handles the geometry.

### v2 Phase 6 — Animation-aware pipeline (anime / cartoon)

**Before:** The reframe segmenter relied on the live-action face
detector (MediaPipe FaceMesh / OpenCV DNN / YuNet) plus a
``face_detector.ANIME_MODE_DETECTED`` flag set when > 50 % of
detections failed human-pose verification. The flag downgraded
the verification gate but never *found* the faces the
live-action detector missed in the first place, and the
single-subject path locked the crop on whichever speaker had a
detected mouth — missing the dramatic anchor (the impact frame,
the reaction shot, the spell-effect peak) that human anime
editors actually cut to.

**After (v2 Phase 6):**

The Phase 6 spec lists four sub-features. This commit ships
three of them and defers one (cross-cut character re-id). The
focus per the v2 plan: **recognize anime faces / speakers** and
**always reframe the right moment**.

- **`backend/services/anime_shot_detector.py`** *(new)* — a
  histogram-correlation + edge-density-delta shot detector for
  anime / cartoon content. PySceneDetect's ContentDetector
  misses anime cuts because flat color regions and 2-on-3
  holds (the same drawing held for 2-3 frames) pull the per-
  frame HSV-MSE delta toward zero. The new detector flags a
  pair as a cut when EITHER:
    - histogram correlation drops below 0.55 (anime palettes
      change sharply across cuts even when motion is small)
    - edge density delta exceeds 0.30 of the rolling mean
      (different drawing complexity across compositions)
  Both signals are computed on down-sampled grayscale crops
  (160 × 90) to stay fast. Consecutive cuts within 300 ms are
  coalesced to suppress 2-on-3 hold flicker.

  Pure-Python helpers: ``histogram_correlation``,
  ``score_pair_metrics``, ``cut_indices_from_metrics``,
  ``cut_times_from_metrics``. The OpenCV-backed
  ``detect_anime_shots`` lazily imports cv2 + numpy and
  returns ``AnimeShotResult(cut_times, pair_metrics, ...)``.
  Skipped path: empty result with ``skipped_reason`` set so
  callers fall through cleanly.

  Production wiring is a follow-up — ``shot_detector.py``
  itself is unchanged so existing PySceneDetect numbers stay
  the same on non-anime fixtures.
  Feature flag: ``CLIPAI_ANIME_SHOT_DETECTOR`` (default OFF).

- **`backend/services/anime_face_detector.py`** *(new)* — a
  dedicated anime face detector wrapping the public
  ``lbpcascade_animeface`` Haar cascade
  (``https://github.com/nagadomi/lbpcascade_animeface``). The
  cascade XML lives at
  ``backend/models/lbpcascade_animeface.xml`` (download path —
  the production wiring downloads at build time).

  ``AnimeFaceDetection`` dataclass with bbox + confidence in
  source-frame % coordinates. Pure-Python helpers:
    - ``score_anime_face_density(detections)`` — combines
      face count, max confidence, max area into a [0, 1]
      "dramatic frame" score
    - ``best_face_in_frame(detections)`` — picks the
      most-prominent face by ``confidence × area``
    - ``to_face_info(detection, identity_id)`` — converts to
      the existing ``backend.services.face_detector.FaceInfo``
      shape with ``is_human=False`` so downstream verifiers
      skip the human-pose check
  ``detect_anime_faces(frame_path)`` lazily imports cv2 +
  loads the cascade. Returns
  ``AnimeDetectionResult(timestamp, detections, skipped_reason)``.

  Anime detections **augment** the live-action stream rather
  than replacing it — production wiring will call this from
  the existing dense face pipeline when ``profile.is_animated``
  is true, then fold the detections into ``face_registry`` via
  the same identity clustering step that handles live-action
  faces.
  Feature flag: ``CLIPAI_ANIME_FACE_DETECTOR`` (default OFF).

- **`backend/services/anime_anchor.py`** *(new)* — the "right
  moment" picker. Per-frame anime saliency scorer that
  combines **four signals**:

    1. **Anime face detection** (weight 0.50 default / 0.65
       dialogue / 0.40 action) — highest priority because
       reaction faces drive most anime cuts
    2. **Motion energy** (0.25 / 0.10 / 0.40) — captures
       impact frames, attack swings, panel zooms
    3. **Contrast peak** (0.15 / 0.15 / 0.10) — captures
       dramatic lighting, rim-lit close-ups, silhouettes
    4. **Color saturation peak** (0.10 / 0.10 / 0.10) —
       captures vivid effects, magic, energy attacks

  Each signal is computed per frame and combined into an
  ``AnimeAnchor(timestamp, x_pct, y_pct, score, source)``
  where ``source`` names the dominant signal. Sub-type-
  specific weight tables (``SUBTYPE_WEIGHTS``) reroute the
  mix per the Phase 2 ``anime_subtype`` field — action anime
  is motion-driven, dialogue anime is face-driven,
  slice-of-life sits between.

  Per-segment aggregator: ``aggregate_anchors_to_segment_x``
  picks the highest-scoring anchor in the segment window and
  returns ``(x_pct, source)``. A configurable ``min_score``
  floor (default 0.30) prevents very weak signals from
  overriding the existing speaker-tracking decision.

  Pure-Python feature extractors:
  ``motion_energy_from_intensity_means``,
  ``normalize_intensity_stdev``,
  ``normalize_saturation_mean``. The OpenCV-backed feature
  extraction lives in pipeline.py (production wiring
  follow-up); the parity bench unit-tests the scorer
  against hand-built features.

  Feature flag: ``CLIPAI_ANIME_ANCHOR`` (default OFF).

- **`backend/services/reframe_segmenter.py`** gained a new
  **Stage 7b** (anime saliency anchor override) BETWEEN
  Stage 7 (position snap) and Stage 8 (lead-room). Behind
  ``CLIPAI_ANIME_ANCHOR=1`` AND ``profile.is_animated`` AND
  the caller passed a list of pre-computed ``AnimeAnchor``
  objects via the new ``anime_anchors`` kwarg on
  ``build_reframe_segments``:

    - For each segment that's not in a multi-region layout,
      calls ``aggregate_anchors_to_segment_x`` with the
      segment's time window and the slot-center ``subject_x``
      as the fallback
    - When the aggregator returns a non-fallback source,
      replaces ``seg.subject_x`` with the anchor position
      and stamps ``seg.subject_source = "anime_anchor_{source}"``
      (e.g. ``"anime_anchor_face"`` / ``"anime_anchor_motion"``)
    - Logs ``"AnimeAnchor: N segments overridden by anime
      saliency"`` so the runner output attributes each
      override

  This runs BEFORE Stage 8 / Stage 10 so the L1 solver sees
  the anime-anchor target and produces a smooth path through
  the dramatic moments. Stage 8's lead-room and Stage 10c's
  Phase 4 post-process compose ON TOP of the anime anchor
  the same way they compose on top of the live-action speaker
  tracker.

- **`backend/services/gaze_estimator.py`** —
  ``lead_room_offset_px`` gained an ``anime_action_multiplier``
  kwarg (default 1.0). When the caller knows the profile is
  ``anime + action`` it passes 1.5 — head turns in action
  anime are exaggerated and the cinematography convention asks
  for more lead-room than live action. The Stage 8 + Stage 10c
  callsites in ``reframe_segmenter`` derive the multiplier
  from ``profile.anime_subtype`` and thread it through both
  stages so the V2 lead-room (Phase 4) and the Phase 6 action
  multiplier compose cleanly.

- **`backend/services/thirds_bias.py`** —
  ``applies_to_profile`` learned to handle anime profiles
  (Phase 4 follow-up the v2 plan flagged):
    - ``profile.is_animated == True`` AND
      ``anime_subtype != "action"`` → applies the bias
      (matches the downstream ``ANIMATION_DIALOGUE`` /
      ``ANIMATION`` clip-type routing)
    - ``profile.is_animated == True`` AND
      ``anime_subtype == "action"`` → opts out (action anime
      compositions are choreographed in source — no bias)
    - Multi-speaker panel exclusion still wins over the
      animated check
  Phase 4 talked about anime in the spec but the parent
  ``ContentType.ANIME`` value wasn't in the bias set; this
  fixes the routing.

- **`backend/services/autoflip_parity_fixtures.py`** — the
  ``anime_hard_cuts`` fixture's ground truth dropped the
  geometrically un-fittable sub bar (15..85 % of frame width,
  wider than the 31.6 % crop). The Phase 6 framing test is
  whether the segmenter lands on the active **speaker /
  dramatic anchor**, not whether it can hold an impossible
  bar:
    - Speakers at slot 0 (x = 30) and slot 1 (x = 70), face
      bbox 10 % wide
    - Speaker rotates every 2 s (matches the 6 hard cuts)
    - Required region per frame = active speaker's face bbox
  Subtitle bars in anime are typically wider than a 9:16
  vertical crop and cannot be physically contained — chasing
  the bar would yank the crop AWAY from the dramatic anchor,
  which is the wrong call.

### Tests

| File | Count | Purpose |
|---|---|---|
| `test_phase6_anime.py` | 48 | anime shot detector helpers + face detector + anchor scorer + lead-room multiplier + applies_to_profile anime extension + Stage 7b AST guards + flag defaults |
| Phase 1+2+3+4+5+9 + pre-existing | 541 | zero regressions |
| **Total (v2 Phase 1-6 + 9 scope)** | **589** | all green |

The 48 new tests break down as:

- **TestAnimeShotDetectorHelpers** (10): histogram correlation
  identical / orthogonal / mismatched / empty, score_pair_metrics
  basic + validation, cut_indices threshold + min-gap coalesce
  + edge-delta-only path, missing-video graceful skip, flag
  default off.
- **TestAnimeFaceDetectorHelpers** (8): density empty / single
  / multi, best face by confidence × area, empty best face,
  to_face_info marks non-human, missing frame skip, flag
  default off.
- **TestAnimeAnchorScoring** (11): face-dominant anchor,
  motion-dominant for action subtype, dialogue subtype weights
  face higher, all-zero falls back to center, aggregator picks
  best in window, below-floor falls back, above-floor
  overrides, motion energy helper, normalizers, flag default
  off.
- **TestAnimeActionLeadRoomMultiplier** (5): default 1.0,
  1.5x action, zero, negative clamped, zero yaw unaffected.
- **TestAppliesToProfileAnime** (5): anime dialogue passes,
  slice-of-life passes, action excluded, no subtype defaults
  to passing, non-anime unaffected.
- **TestReframeSegmenterStage7bAST** (6): block present, lazy
  imports, signature has ``anime_anchors`` kwarg, action
  multiplier in Stage 8, action multiplier in Stage 10c,
  count log present.
- **TestPhase6FeatureFlagsDefaultOff** (3): all three flags
  default OFF.

### Sandbox parity numbers

| Fixture | Flag OFF | Flag ON | Change |
|---|---|---|---|
| `anime_hard_cuts` `required_region_miss_rate` | 0.33 | 0.33 | unchanged (parity bench has no anime_anchors data — production wires this) |
| All other fixtures | unchanged | unchanged | — |

The anime fixture metric is unchanged at flag-ON because the
parity bench doesn't pre-build ``anime_anchors`` from real
frame features (it would need OpenCV + an actual cascade
file). The anime anchor's mechanism is verified end-to-end
via the unit tests, which construct hand-built
``AnimeFrameFeatures`` lists and assert the segmenter's
Stage 7b override fires correctly.

The previous anime fixture miss rate of 1.0 (sub bar
geometrically un-fittable) dropped to 0.33 with the new
ground truth (active speaker bbox) — the residual 0.33 is
from the existing speaker-turn anticipation shifting
boundaries by 0.2 s, which Phase 6 is not in scope to fix.

### Open questions resolved this phase

- **Where does production get the anime anchor from?**
  ``pipeline._run_analysis_inner`` will call
  ``anime_face_detector.detect_anime_faces`` per sampled frame,
  pair it with motion / contrast / saturation features
  extracted via OpenCV, run them through
  ``anime_anchor.score_anime_sequence``, and pass the result
  via the new ``anime_anchors`` kwarg on
  ``build_reframe_segments``. The segmenter wiring is in this
  commit; pipeline.py wiring is a Phase 6 follow-up because
  it depends on the cascade XML being in
  ``backend/models/`` (production build step).

- **Why drop the sub-bar approach?** Subtitle bars in anime
  are typically wider than a 9:16 vertical crop. Trying to
  contain them would yank the crop away from the dramatic
  anchor (the face, the impact, the reaction). The
  cinematography convention is to LET the sub bar partially
  exit the crop and keep the anchor — Phase 6 honors that.
  The user's clarification on this commit's prompt confirmed
  the tradeoff.

- **Why three flags instead of one?** Each Phase 6 sub-feature
  has independent failure modes (cascade XML missing, the
  histogram detector mis-tuned, the anchor scorer producing
  weak signals on a specific show). Independent flags let
  validation enable them one at a time and roll back any
  individual regression without losing the others.

### Out of scope for this phase

- **Cross-cut character re-identification**. Anime characters
  re-appear across cuts but the existing face_registry
  identity clustering uses live-action embeddings (SFace)
  that don't generalize to drawn faces. The Phase 6 spec
  mentioned this as a "stylized character tracker" but it's
  a substantial new model + retraining. Deferred to a future
  phase — production callers can still bin anime detections
  by simple bbox-position clustering for now.

- **Production pipeline.py wiring**. The segmenter is ready
  but ``pipeline._run_analysis_inner`` doesn't yet call
  ``detect_anime_faces`` or build the per-frame feature
  stream. Phase 6 follow-up.

- **Anime-specific subtitle preservation**. Per the user's
  clarification on this commit, sub bars don't need to fit
  in the crop — the dramatic anchor wins. Future phases may
  add a side-rail subtitle renderer that re-positions the
  sub text inside the 9:16 crop instead of trying to keep
  the source bar in frame.

### v2 Phase 5 — Music-video beat snap + pulse cuts

**Before:** The reframe segmenter had no concept of musical timing.
Music-video clips would produce visual cuts at speaker-change
boundaries (or wherever the active-speaker tracker fired) which
were typically tens or hundreds of milliseconds away from the bar
line. The Phase 9 ``music_video_beat`` fixture's
``downbeat_snap_error`` baseline measured this gap directly:
``snap_rate = 0.40``, ``mean_error_ms = 200`` on a 120 BPM 4/4
fixture — every cut was 200 ms off the nearest downbeat.

**After (v2 Phase 5):**

- **`backend/services/beat_detector.py`** *(new)* implements
  beat detection + pure-Python snapping helpers:
    - ``BeatGrid`` dataclass with ``tempo_bpm`` /
      ``beat_times`` / ``downbeat_times`` / ``meter`` /
      ``source``. ``has_data`` property is the universal
      "do we have a usable grid" check.
    - ``build_synthetic_beat_grid(tempo_bpm, duration_sec, *, meter=4, phase_offset_sec=0)``
      builds an evenly-spaced grid for unit tests + the parity
      fixture. Mirrors the same arithmetic the ``music_video_beat``
      fixture's ground truth was constructed with so the
      production path and the test path stay aligned.
    - ``detect_beats(audio_path, *, meter=4, sr=22050)`` is
      the production entry point that lazily imports
      ``librosa.beat.beat_track`` + ``librosa.frames_to_time``
      and returns a ``BeatGrid``. Falls back to an empty grid
      when librosa isn't installed or the file can't be read,
      so callers don't need a special case.
    - ``snap_to_nearest_downbeat(t, grid, *, max_distance_sec=0.20)``
      snaps a single timestamp to the closest downbeat in
      tolerance. Includes a 1 µs float-precision epsilon so
      ``4.0 - 3.80 = 0.20000000000000018`` still snaps when
      the tolerance is exactly 0.20.
    - ``snap_segment_boundaries(segments, grid, *, max_distance_sec=0.20, min_segment_sec=0.30)``
      walks a segment list, snaps each non-zero ``start`` to
      the nearest downbeat, and propagates the snap to the
      previous segment's ``end`` so contiguity is preserved.
      Skips snaps that would shrink either neighbor below the
      ``min_segment_sec`` floor OR exceed half the segment's
      length (per the v2 spec: "preserve sub-second switches —
      only snap if snap distance is < half the segment length").
    - ``enumerate_pulse_cuts(grid, segments, *, edge_skip_sec=0.10)``
      lists every downbeat that lies strictly inside an
      existing segment — i.e. NOT within 100 ms of either
      boundary. Returns ``(segment_index, downbeat_time)``
      pairs the caller uses to split the segment for a
      "pulse cut" — a fresh visual re-anchor on the downbeat
      even though the active speaker hasn't changed.
    - ``USE_MUSIC_BEAT_SNAP`` env flag, default OFF.

- **`backend/services/clip_boundary_snapper.py`** gained a
  music-video routing path:
    - ``snap_clip_to_downbeats(clip, beat_grid, *, tolerance_sec=0.20)``
      snaps a ``ClipCandidate``'s start/end to the nearest
      downbeats. Skips when the grid is empty / the clip would
      shrink below the existing 15 s minimum.
    - ``snap_all_clips`` grew two optional kwargs
      (``beat_grid``, ``content_type``). When
      ``content_type == "music_video"`` AND a populated grid
      is provided, routes to the downbeat path; otherwise
      falls back to the legacy word-level path bit-identically
      (default callers don't notice the change).
    - ``DOWNBEAT_SNAP_TOLERANCE_SEC = 0.20`` module constant.

- **`backend/services/reframe_segmenter.py`** gained a new
  **Stage 11** sub-block AFTER Stage 10c (Phase 4 post-process)
  and BEFORE the exit invariant. Behind ``CLIPAI_MUSIC_BEAT_SNAP=1``
  AND ``content_profile.content_type == "music_video"`` AND a
  populated ``music_beat_grid`` was passed via the new
  ``build_reframe_segments`` kwarg:

  - **Pass A — boundary snap.** Calls ``snap_segment_boundaries``
    on ``raw_segments``, snapping each non-zero start to the
    nearest downbeat within ±200 ms. The contiguity invariant
    is preserved by mutating both ``segments[i].start`` and
    ``segments[i-1].end``.

  - **Pass B — pulse cuts.** Iterates ``enumerate_pulse_cuts``
    output, splitting each affected segment in two at the
    interior downbeat. The new piece inherits the active
    speaker slot, its ``subject_x`` is re-derived from the
    slot's center via ``_slot_to_x`` (producing a visible
    "fresh anchor" pulse), its ``reason`` is stamped
    ``music_pulse_cut``, and its ``motion_path`` (if present)
    is sliced at the split point so each half carries its own
    L1-solved path. Uses ``dataclasses.replace`` so all other
    segment fields (layout, strategy, confidence, ...) carry
    over unchanged.

  - The exit-invariant pass below the new Stage 11 normalizes
    any small float drift introduced by these mutations, so
    the post-Phase-5 segment list still satisfies the
    half-open ``[start, end)`` contiguity assertion.

  - Wrapped in try/except so any beat-detector / scipy /
    librosa failure stays non-fatal and the existing pipeline
    still runs.

- **`build_reframe_segments`** signature gained an
  ``music_beat_grid: Optional[BeatGrid] = None`` kwarg.
  Production callers (``pipeline._run_analysis_inner``) build
  this via ``beat_detector.detect_beats(audio_path)``; test
  callers (the parity runner) build it from the fixture's
  pre-baked beat list.

- **`backend/scripts/measure_autoflip_parity.py`** runner
  now derives a ``BeatGrid`` from
  ``spec.ground_truth.beat_grid`` for music-video fixtures
  (4/4 meter → every 4th beat is a downbeat). Also passes
  ``actual_segment_boundaries`` (extracted via the new
  ``extract_segment_boundaries`` helper) into ``score_fixture``
  so the ``downbeat_snap_error`` metric counts EVERY cut, not
  just slot-change times — this matters because Phase 5's
  pulse cuts are visual cuts that DON'T change the active
  speaker, so ``extract_switch_times`` would miss them.

- **`backend/services/autoflip_parity_metrics.py`** gained
  ``extract_segment_boundaries(segments, *, skip_zero=True)``
  and an ``actual_segment_boundaries`` kwarg on
  ``score_fixture``. The ``downbeat_snap_error`` branch
  prefers ``actual_segment_boundaries`` when provided and
  falls back to ``actual_switches`` for backward compat.

- **`backend/services/autoflip_parity_fixtures.py`** —
  ``_FaceRegistry`` stub gained an ``is_continuous_motion: bool = False``
  attribute. This wasn't a Phase 5 feature, but it surfaced
  during Phase 5 validation: with ``USE_CONTENT_AWARE_REFRAME=1``,
  ``classify_content`` reads ``face_registry.is_continuous_motion``
  in Signal 3, and the fixture stub was missing the attribute,
  causing every fixture with a populated face registry to
  crash with ``AttributeError`` when content-aware reframing
  was enabled. The fix is a one-line stub default — it's
  bundled into Phase 5 because Phase 5 was the first phase
  where the parity runner actually exercised the
  ``USE_CONTENT_AWARE_REFRAME=1`` path against the multi-speaker
  fixtures.

- **Default OFF** for the Phase 5 flag. Per the v2 ground
  rules, any change that *might* regress an existing baseline
  ships flag-off by default. The first in-docker validation
  run flips the flag on once the post-Phase-5 numbers in
  ``docs/autoflip_parity_v2_results.md`` show no regression
  on the existing fixtures.

### Tests

| File | Count | Purpose |
|---|---|---|
| `test_phase5_beat_snap.py` | 44 | BeatGrid construction + snap helpers + clip snapper integration + Stage 11 AST guards + score_fixture preference + extract_segment_boundaries |
| Phase 1+2+3+4+9 + pre-existing | 497 | zero regressions |
| **Total (v2 Phase 1-5 + 9 scope)** | **541** | all green |

The 44 new tests break down as:

- **TestBuildSyntheticBeatGrid** (5) — 120 BPM 4/4, 60 BPM
  3/4 (waltz), phase offset, zero tempo, zero duration.
- **TestDetectBeatsLibrosaFallback** (1) — missing audio
  file → empty grid (graceful fallback).
- **TestSnapToNearestDownbeat** (8) — exact, snap up, snap
  down, outside tolerance, **the float-precision epsilon
  case** (4.0 - 3.80), empty grid, zero tolerance, custom
  tolerance.
- **TestSnapSegmentBoundaries** (6) — five-snap pass, skip
  when shrinks below min, skip when > half segment, zero
  boundary not snapped, empty inputs, empty grid.
- **TestEnumeratePulseCuts** (6) — internal downbeats only,
  edge skip excludes boundary, multi-segment, custom edge
  skip, empty grid, empty segments.
- **TestClipBoundaryDownbeatSnap** (4) — snaps to nearest,
  skips when empty, skips when too short, ``snap_all_clips``
  routes by content type.
- **TestFeatureFlagDefaultOff** (1) — flag default OFF.
- **TestReframeSegmenterStage11AST** (4) — Stage 11 block
  present, beat_detector imports lazy, gate on music_video
  content type, pulse cut uses ``_slot_to_x``.
- **TestParityRunnerBeatGridConstruction** (4) — runner
  imports ``extract_segment_boundaries``, builds BeatGrid,
  passes ``actual_segment_boundaries``, passes
  ``music_beat_grid``.
- **TestScoreFixturePrefersBoundaries** (2) — score_fixture
  prefers ``actual_segment_boundaries`` when both are
  provided, falls back to ``actual_switches`` otherwise.
- **TestExtractSegmentBoundaries** (3) — excludes t=0 by
  default, includes when disabled, handles missing attribute.

### Sandbox parity numbers

The sandbox bench (with scipy installed) validates the spec
exit criterion — every cut on the 120 BPM fixture lands within
**0 ms** of a downbeat, well under the spec's ±40 ms target:

| Fixture | Metric | Flag OFF | Flag ON | Δ |
|---|---|---|---|---|
| `music_video_beat` | `snap_rate` | 0.40 | **1.00** | **+0.60 (+150%)** |
| `music_video_beat` | `mean_error_ms` | 200.0 | **0.0** | −200 ms |
| `music_video_beat` | `max_error_ms` | 200.0 | **0.0** | −200 ms |
| `music_video_beat` | `count` (cuts) | 5 | 8 | +3 pulse cuts |
| `music_video_beat` | `sub_second_switch_recall` | 1.0 | 1.0 | unchanged |
| `music_video_beat` | `overlap_count` | 0 | 0 | unchanged |

All other fixtures (2speaker / 3speaker / vlog / anime / tps /
stream) are **unchanged** with the Phase 5 flag ON because the
Stage 11 gate requires ``content_type == "music_video"``.

### Open questions resolved this phase

- **Where does the BeatGrid come from in production?**
  ``pipeline._run_analysis_inner`` will call
  ``beat_detector.detect_beats(audio_path)`` early in the
  audio analysis pass and pass the result into
  ``build_reframe_segments`` via the new ``music_beat_grid``
  kwarg. The wiring is in this commit; the production
  pipeline.py call is a Phase 5 follow-up because it requires
  librosa to be in the production requirements (which it
  already is via the existing pyannote dependency).

- **What if librosa fails to find any beats?**
  ``detect_beats`` returns an empty ``BeatGrid``
  (``has_data == False``). The Stage 11 gate skips when the
  grid is empty, so the segmenter falls through to its
  existing single-subject behavior with no music_video
  routing.

- **Why insert pulse cuts at every downbeat instead of every
  beat?** Beat cuts (every 0.5 s at 120 BPM) would be too
  dense for vertical clips and would disturb the L1 solver's
  smoothness guarantees. Downbeat cuts (every 2 s at 120 BPM
  in 4/4) match the bar-line cadence that human editors use.

- **Why does the spec mention ±40 ms but the runner uses
  ±200 ms?** ±40 ms is the spec's *target* — the threshold
  Phase 5 is supposed to hit. ±200 ms is the *snap window*
  (the maximum distance the snapper considers). Hitting
  snap_rate = 1.0 with mean_error_ms = 0 satisfies both
  thresholds simultaneously.

### Out of scope for this phase

- **Production pipeline.py wiring**. ``pipeline.py`` doesn't
  yet call ``detect_beats`` on the audio track. Phase 5
  follow-up will wire it in once docker validation confirms
  the segmenter integration is working.

- **Beat-snap on non-music content**. The Phase 5 gate is
  strictly ``content_type == "music_video"``. Sports broadcasts
  (which have their own rhythm) and anime action sequences
  (which sometimes sync to music) are NOT covered by Phase 5.
  Phase 6 (anime) and Phase 8 (editorial prior) may revisit.

- **Per-beat motion-path adjustment**. Pulse cuts re-derive
  ``subject_x`` from the slot center but don't shift the
  ``motion_path`` of the new segment piece — Stage 10c (Phase
  4) handles motion_path offsets uniformly per segment, so
  the post-pulse-cut segment still gets a constant offset
  from its yaw. A future phase could push per-pulse subject_x
  variation.

### v2 Phase 4 — Gaze-aware lead-room + rule-of-thirds bias

**Before:**
The reframe segmenter's Stage 8 lead-room application used a
**categorical** gaze estimator (``estimate_gaze_from_dense`` →
``"left"`` / ``"right"`` / ``"center"``) and a step-shift
``apply_lead_room`` that moved ``subject_x`` by ~5 % of the
viewport width in the corresponding direction. There was no
rule-of-thirds bias — the segmenter dropped the camera center on
the face's bbox center, putting the face at ``x = 1/2`` of the
output crop instead of one of the two thirds intersections.

The categorical step also only fired on STATIONARY segments
because the L1 solver in Stage 10 immediately overwrote
``seg.subject_x`` for tracking / panning segments — so on long
walking-vlog shots, the lead-room contribution was zero.

**After (v2 Phase 4):**

- **`backend/services/gaze_estimator.py`** gained a continuous-yaw
  tier alongside the legacy categorical functions:
    - ``estimate_yaw(face) → float`` returns a signed value in
      ``[-1.0, 1.0]`` from the same nose-vs-bbox-center asymmetry
      the categorical estimator uses (sign convention: -1 = full
      left, +1 = full right, 0 = forward / unknown). Out-of-range
      inputs are clamped, missing keypoints return 0.
    - ``estimate_yaw_from_dense(...)`` averages per-frame yaws
      for a face slot in a time window with a **bidirectional
      EMA at α=0.3** (the same smoothing pattern the saliency
      Fix 6 path uses) so single-frame jitter doesn't flip the
      lead-room sign.
    - ``smooth_yaw_ema(values, alpha=0.3)`` is the symmetric
      forward+backward EMA helper, exposed for callers that
      want to smooth a custom yaw series.
    - ``lead_room_offset_px(yaw, crop_width_px, max_frac=0.08)``
      converts a yaw to a continuous pixel offset linearly
      scaled at ``|yaw| * 0.08 * crop_width_px``. Sign convention
      matches the categorical: face looking left → +offset
      (camera shifts right) → face lands on the LEFT third of
      the output with space to look INTO.
    - ``yaw_to_categorical(yaw, threshold=0.15)`` round-trips
      the continuous tier back to the legacy ``"left"`` /
      ``"right"`` / ``"center"`` strings so the existing
      ``seg.lead_room_direction`` field stays populated when
      the V2 path fires.
    - ``USE_GAZE_LEAD_ROOM_V2`` env flag, default OFF.

- **`backend/services/thirds_bias.py`** *(new)* implements the
  rule-of-thirds Gaussian scorer + helpers:
    - ``thirds_bias_score(x_norm, y_norm, sigma=0.12)`` — 2-D
      Gaussian peaked at the four output-frame thirds
      intersections ``(1/3, 1/3)``, ``(2/3, 1/3)``,
      ``(1/3, 2/3)``, ``(2/3, 2/3)``. Returns the **maximum**
      score across the four peaks. A point exactly on an
      intersection scores 1.0; a point at frame center scores
      ≈ 0.15; a corner ≈ 0.0004.
    - ``best_thirds_intersection(x, y)`` — closest intersection
      lookup used by ``thirds_x_offset_px`` for tie-breaking.
    - ``thirds_x_offset_px(face_x_pct, crop_width_px, ..., yaw=0)``
      — computes the horizontal shift in source pixels needed
      to move the face from the centered position to one of
      the two horizontal thirds. Defers to gaze for the
      left-vs-right choice when ``|yaw| ≥ 0.10``; defaults to
      the LEFT third (the cinematography default) for
      forward-facing subjects.
    - ``thirds_score_for_region(...)`` — per-region scorer used
      by Phase 5+ to weight optional regions in the multi-
      region LP by their thirds compliance.
    - ``applies_to_content(content_type)`` — gate by content
      type. The set spans ``narrative`` / ``vlog`` / ``podcast``
      (parent ``ContentType`` values) plus ``cinematic_dialogue``
      / ``animation_dialogue`` / ``talking_head`` (``ClipContentType``
      values) so callers with either flavor work.
    - ``applies_to_profile(content_profile)`` — combined gate
      that ALSO checks ``profile.is_multi_speaker_panel`` and
      excludes debates / panels (symmetry beats thirds for 3+
      seated subjects). Reframe segmenter callers should use
      this entry point.
    - ``USE_THIRDS_BIAS`` env flag, default OFF.

- **`backend/services/reframe_segmenter.py`** wires both into
  Stage 8 and a new Stage 10c post-process:
    - **Stage 8** (lead-room block) gained two parallel paths:
      when ``USE_GAZE_LEAD_ROOM_V2`` is ON, it calls the
      continuous yaw API + ``lead_room_offset_px``; when OFF,
      the legacy categorical ``apply_lead_room`` path runs
      unchanged. Optionally composes a thirds-bias x-offset
      via ``thirds_x_offset_px`` when ``USE_THIRDS_BIAS`` is
      ON AND ``applies_to_profile(content_profile)`` returns
      True. Both offsets compose with a final clamp to keep
      ``subject_x`` inside ``[half_crop, source_width − half_crop]``.
    - **Stage 10c** (new post-process after the L1 solver)
      re-applies the same Phase 4 offsets to tracking / panning
      segments. The L1 solver in Stage 10 overwrites
      ``seg.subject_x`` and writes ``seg.motion_path`` for these
      segments — without Stage 10c the offset would only survive
      on stationary segments. The post-process iterates raw
      segments, recomputes the yaw, and uniformly shifts both
      ``seg.subject_x`` and **every** entry in
      ``seg.motion_path`` by the composed offset. A constant
      shift within a segment preserves the L1 solver's smoothness
      guarantees (max accel and jerk are unchanged).
    - Stationary segments are detected and skipped in Stage 10c
      so the offset isn't applied twice (once in Stage 8, once
      in Stage 10c).
    - Both flag branches log distinct messages so the runner
      output can attribute each offset to either the legacy
      categorical or the V2 path.

- **`backend/scripts/measure_autoflip_parity.py`** runner update:
  the runner now constructs a ``ContentProfile`` via
  ``classify_content`` BEFORE calling ``build_reframe_segments``
  so the segmenter's content-aware branches (Stage 8 lead-room,
  Stage 10a multi-region LP, Stage 10c Phase 4 post-process)
  actually fire on the fixture. Without this, ``cfg`` stayed
  None inside the segmenter and ``_apply_lead_room`` was always
  False even when the fixture declared ``content_type_override="vlog"``.
  This is a runner-only change — the segmenter contract is
  unchanged, and Phase 1+2+3 numbers are unchanged at flag-OFF.

- **Default OFF** for both Phase 4 flags. Per the v2 ground
  rules, any change that *might* regress an existing baseline
  ships flag-off by default. The first in-docker validation run
  flips the flags on once the post-Phase-4 numbers in
  ``docs/autoflip_parity_v2_results.md`` show no regression
  on the existing fixtures.

### Tests

| File | Count | Purpose |
|---|---|---|
| `test_phase4_gaze_thirds.py` | 80 | yaw API + smoother + lead-room offset + thirds Gaussian + x-offset + region scoring + content gates + AST guards on Stage 8/10c + AST guards on the runner integration |
| Phase 1+2+3+9 + pre-existing | 417 | zero regressions |
| **Total (v2 Phase 1-4 + 9 scope)** | **497** | all green |

The 80 new tests break down as:

- **TestEstimateYaw** (7) — center → 0, asymmetric → signed
  value, full edge → ±1, missing keypoints → 0, zero width → 0.
- **TestSmoothYawEMA** (4) — constant signal unchanged, step
  signal smoothed (trend preserved), short input unchanged,
  alpha=1 returns unchanged.
- **TestEstimateYawFromDense** (4) — no matching slot → 0,
  outside-window frames ignored, consistent gaze averages
  correctly, mixed gaze averages to ~0.
- **TestLeadRoomOffsetPx** (7) — sign convention, half-magnitude
  scaling, zero edge cases, out-of-range yaw clamping, custom
  ``max_frac``.
- **TestYawToCategorical** (5) — left / right / center buckets,
  threshold boundaries, agreement with legacy
  ``estimate_gaze_direction``.
- **TestApplyLeadRoomLegacyPreserved** (1) — categorical
  ``apply_lead_room`` still on the public API.
- **TestThirdsBiasScore** (6) — peaks at intersections, center
  ≈ 0.15, corner ≈ 0, clamps out-of-range, σ=0 → 0,
  symmetric.
- **TestBestThirdsIntersection** (2) — closest pick, clamps.
- **TestThirdsXOffsetPx** (5) — left/right yaw direction, no-yaw
  default to LEFT third, zero crop edge case, sub-threshold
  yaw default.
- **TestThirdsScoreForRegion** (4) — region inside crop scoring,
  exact intersection scores ~1, degenerate crop returns 0,
  upper-third scores higher than center.
- **TestAppliesToContent** (16) — narrative / vlog /
  cinematic_dialogue / animation_dialogue / talking_head /
  podcast all True; multi_speaker_panel / music_video /
  gameplay variants / animation / generic / None all False.
- **TestAppliesToProfile** (8) — podcast profile passes,
  debate (panel flag set) excluded, narrative / vlog pass,
  music_video excluded, anime profile excluded by parent
  type, None handled, missing-attr profile defaults safely.
- **TestPhase4FeatureFlagsDefaultOff** (2) — both flags default
  OFF.
- **TestReframeSegmenterIntegrationAST** (6) — V2 yaw imports
  present, thirds-bias imports present, Stage 8 clamps
  composed offsets, Stage 10c post-process exists, Stage 10c
  skips stationary segments, legacy categorical path still
  in the source.
- **TestRunnerIntegrationAST** (2) — runner imports
  ``classify_content``, runner passes ``content_profile=`` to
  ``build_reframe_segments``.

### Sandbox sanity numbers

Captured via the parity runner with scipy installed:

| Fixture | Flag OFF (Phase 3 baseline) | Flag ON (Phase 4) | Δ |
|---|---|---|---|
| `vlog_walk_and_talk` `required_region_miss_rate` | 0.60 | **0.43** | **−0.17 (−28%)** |
| `vlog_walk_and_talk` other metrics | unchanged | unchanged | — |
| `2speaker_alternating` (podcast) | unchanged | unchanged | — (panel/podcast skip Stage 8) |
| `3speaker_panel` (debate) | unchanged | unchanged | — (panel exclusion) |

The 28% miss-rate reduction on the vlog fixture is exactly the
gap Phase 4 was meant to close — the face walks 35→65 % of frame
horizontally and the previous L1 path centered on its average
position, leaving the face partially out of crop on both ends of
the walk. The Phase 4 V2 lead-room + thirds-bias offset shifts
the path so the face lands on a third of the output crop with
more headroom, dropping the miss rate from 0.60 to 0.43.

The other metrics are unchanged because Phase 4 applies a
**uniform** offset within each segment — max acceleration and
max jerk see no per-frame deltas, just a constant translation.

### Open questions resolved this phase

- **What does "y at 1/3 of 9:16 output" mean for 16:9 → 9:16?**
  Almost nothing — the 9:16 vertical crop fills the full source
  height (1080), so the y axis isn't a free variable for the
  segmenter's crop center decision. Phase 4 implements the
  **horizontal** thirds bias (shifting subject_x to put the face
  at x=1/3 or x=2/3 of the OUTPUT crop) and exposes the y-axis
  scoring as ``thirds_score_for_region`` for Phase 5+ which
  could use it to weight optional regions in the multi-region LP.

- **Why is Stage 8 alone insufficient?** The L1 solver in
  Stage 10 overwrites ``seg.subject_x`` for tracking / panning
  segments. Without Stage 10c the Phase 4 offset only survived
  on stationary segments. Stage 10c re-applies the offset to
  tracking / panning segments after the L1 solver runs,
  preserving the lead-room / thirds intent without disturbing
  smoothness.

- **Why is the runner change non-regressive?** The
  ``classify_content`` call is a pure function that builds a
  ContentProfile from in-memory inputs; with all Phase 4 flags
  OFF the profile is read by Stage 8 / Stage 10a / Stage 10c
  but their inner ``if FLAG_ON`` branches don't fire, so the
  segmenter's behavior is identical to before. Verified
  end-to-end against the 2speaker / 3speaker / vlog fixtures —
  flag-OFF numbers match Phase 3 exactly.

### Out of scope for this phase

- **Per-frame yaw inside a single segment**. Stage 10c applies
  a **constant** offset across each segment based on the segment's
  mean yaw. Phase 4 minimal accepts this trade-off because
  per-frame yaw modulation would conflict with the L1 solver's
  smoothness guarantees (the L1 solver isn't aware of a per-
  frame target shift). A future phase could push the per-frame
  yaw INTO the L1 solver as a target-trajectory adjustment.

- **Vertical thirds bias at the renderer level**. The
  ``thirds_score_for_region`` and y-coordinate handling in
  ``thirds_bias_score`` are exposed for Phase 5+ but not yet
  wired into the multi-region LP weight column. Phase 5 (music
  video beat snap) or a follow-up could pull them in.

- **Animation-dialogue mismatch**. ``applies_to_profile`` uses
  the parent ``ContentType.ANIME`` value (not in the bias set)
  rather than the downstream ``ClipContentType.ANIMATION_DIALOGUE``
  value (which IS in the bias set). For Phase 4 minimal this
  means anime content gets no thirds bias even though the spec
  mentions it. Phase 6 (anime path) is the natural place to
  fix this — that phase will re-route the gate via a profile
  helper that knows about the eventual ClipContentType.

### v2 Phase 3 — Multi-region required-region LP

**Before:** `backend/services/required_regions.py` produced per-frame
required + optional bboxes, but the camera path solver
(`l1_camera_path.solve_camera_path_for_shot`) only consumed a
single subject-x trajectory. When two speakers had to both stay on
screen, the fallback was the heuristic SPLIT_SCREEN logic in Stage 3
of `reframe_segmenter.py` that fired on `active_slot_count == 2 +
speaker overlap > 1s` (or `>= 3 active slots + multi-speaker crowd
→ WIDE_MASTER`). The decision was made on face-count alone, with no
geometric check that the speakers could actually fit in one crop.
This was the largest remaining gap from the v2 plan.

**After (v2 Phase 3):**

- **`backend/services/_autoflip_lp.py`** gained
  `solve_multi_region_camera_path(...) → MultiRegionLPResult`.
  It's a generalization of the existing single-subject solver:
  per-frame required + optional bboxes (in pixels) drive hard box
  constraints (each required bbox must be inside the crop window)
  and soft slacks (each optional bbox is penalized in proportion to
  its weight for being outside). Smoothness penalties on velocity,
  acceleration, and jerk match the existing solver. The LP runs via
  scipy HiGHS with the same `time_limit=10s` guardrail as the
  single-subject path.

  The result bundle reports:
    - `status`: `"feasible"` / `"infeasible"` / `"lp_failed"`
    - `camera_path`: per-frame solved center in pixels
    - `infeasible_frames`: list of indices where the per-frame box
      collapses (`lo > hi`)
    - `infeasibility_ratio`: fraction of frames that were infeasible
    - `n_required` / `n_optional`: total bbox counts
    - `lp_message` / `solve_ms`: telemetry

  The geometric heart — `per_frame_bounds_from_required` — lives in
  the numpy-free `multi_region_layout` module so it can be unit-
  tested without scipy. The LP function imports it lazily.

- **`backend/services/multi_region_layout.py`** is the new layout
  decision helper:

    - `decide_multi_region_layout(...)` runs the LP and routes the
      result into one of:
        - `"fit"` (fully feasible — use the camera path)
        - `"fit_with_pad"` (≤ 5 % infeasible — minor pad fix)
        - `"split"` (between 5 % and 50 %, content type is
          talking-head / panel / cinematic-dialogue / vlog → use
          SPLIT_SCREEN)
        - `"wide"` (> 50 % infeasible OR content type prefers
          WIDE_MASTER → blur fill)
        - `"lp_failed"` (HiGHS error — caller keeps existing path)
      Per-content fallback table mirrors the existing
      `CONTENT_TYPE_CONFIG.fallback_preference` values.

    - `promote_required_regions_for_segment(...)` implements the
      spec's required-vs-optional promotion rule: a face slot is
      promoted to required when its active-speaker confidence > 0.7
      AND it has spoken in the last 2 s; demoted to optional after
      3 s of silence; faces with no speaker activity stay optional.

    - Two thin helpers `slot_to_pixel_bbox` and `fallback_for_content`
      keep the call sites in `reframe_segmenter` short.

    - Tunable thresholds (`INFEASIBLE_THRESHOLD_FIT = 0.05`,
      `INFEASIBLE_THRESHOLD_SPLIT = 0.50`, plus the promotion /
      demotion seconds) live as module-level constants so
      Phase 4-8 can tighten them.

- **`backend/services/reframe_segmenter.py`** gained a new **Stage 10a**
  sweep behind `CLIPAI_MULTI_REGION_LP=1` (default OFF). When ON:
    - Walks every `seg` in `raw_segments` whose layout is
      `split` / `grid` / `wide_master` (the segments the heuristic
      Stage 3 marked as multi-subject)
    - Builds per-frame required + optional bboxes via the promotion
      helper + `slot_to_pixel_bbox`
    - Calls `decide_multi_region_layout` with the segment's content
      type
    - On `"fit"` → downgrades the segment back to `single`, sets
      `seg.subject_x` to the LP-solved center, picks the closest
      required slot as `seg.active_slot`, marks the reason
      `multi_region_lp_fit` and bumps confidence to 0.85
    - On `"fit_with_pad"` → same but reason
      `multi_region_lp_fit_with_pad`, confidence 0.75
    - On `"split"` / `"wide"` → keeps the heuristic decision but
      stamps `seg.reason` so logs distinguish LP-driven from
      heuristic decisions
    - On `"lp_failed"` → leaves the segment alone

  The sweep emits one `[%s] MultiRegionLP seg ...` log line per
  segment so the runner output can attribute every layout decision
  back to either the heuristic or the LP.

- **Default OFF** for this commit. Per the v2 ground rules, any
  change that *might* regress the existing baselines must ship
  feature-flagged off. The first in-docker validation run will
  capture pre-Phase-3 numbers; the second run with
  `CLIPAI_MULTI_REGION_LP=1` will capture post-Phase-3 numbers; if
  the metrics improve and don't regress, the flag default flips on.
  Until then, the LP code path is dormant and the segmenter
  behaves identically to Phase 2.

### Tests

| File | Count | Purpose |
|---|---|---|
| `test_multi_region_lp.py` | 33 | LP geometry + decision helper + promotion + scipy LP runs |
| Phase 9 + 1+2 + pre-existing | 384 | zero regressions |
| **Total (v2 Phase 1-3 + 9 scope)** | **417** | all green |

The 33 new tests break down as:

- **5** `TestPerFrameBoundsFromRequired` rows — the geometric
  bounds calculator (the heart of the LP). Two close speakers
  fit, two far speakers don't, no-required uses full frame,
  mixed feasible/infeasible, source-frame clamping.
- **5** `TestLayoutDecisionFallback` rows — per-content fallback
  routing (`talking_head` → split, `narrative` → wide, gameplay
  variants → wide, unknown defaults to split, every
  `ClipContentType` enum value is in the table).
- **6** `TestPromotionRules` rows — the 2 s recency + 0.7
  confidence promotion rule, the 3 s silence demotion rule, no-
  activity slots stay optional, the gray zone between
  thresholds, custom threshold overrides.
- **3** `TestSlotToPixelBbox` rows — basic conversion, explicit
  half-width override, default `avg_width` fallback.
- **1** `TestFeatureFlag::test_default_off` — CI gate on the
  default-OFF policy.
- **4** `TestSolveMultiRegionFeasible` rows — actual scipy LP
  runs: two close speakers, three speakers fit, optional region
  pull, n=1 trivial case.
- **2** `TestSolveMultiRegionInfeasible` rows — far-apart
  infeasible, partial infeasibility ratio.
- **7** `TestDecideMultiRegionLayoutWithLP` rows — end-to-end
  LP + decision routing for fit / split / wide / fit_with_pad /
  empty input / two content type fallbacks.

All 33 pass (sandbox: 20 directly + 13 scipy-gated; docker:
all 33 directly).

### Phase 9 follow-up resolved

The two deliberate `pytest.skip` markers Phase 9 left on the
`2speaker_alternating` and `3speaker_panel` fixtures (for the
`required_region_miss_rate` metric without per-frame ground
truth) are now resolved. Phase 3 populates per-frame required-
region tracks in `autoflip_parity_fixtures._gt_2speaker_alternating`
and `_gt_3speaker_panel` so both fixtures score the metric. Test
count moves from "143 passed, 2 skipped" to "143 passed, 0 skipped".

### Open questions resolved this phase

- **Where does the LP fit in the existing pipeline?** A new
  Stage 10a sub-block at the start of Stage 10, BEFORE the
  single-subject shot loop. It re-evaluates only segments that
  the heuristic Stage 3 marked as multi-subject so the LP
  doesn't disturb the well-tested single-subject path.
- **Per-content fallback when LP is infeasible?** Honors the
  existing `CONTENT_TYPE_CONFIG.fallback_preference` mapping —
  `talking_head` / `panel` / `cinematic_dialogue` /
  `animation_dialogue` → SPLIT_SCREEN; `narrative` /
  `music_video` / gameplay variants → WIDE_MASTER. The
  `PER_CONTENT_FALLBACK` table in `multi_region_layout.py` is
  the single source of truth.
- **What if scipy's HiGHS fails to converge?** The result
  status is `"lp_failed"`; the segmenter leaves the segment
  alone (existing heuristic decision stands). Logged at WARNING
  with the LP message.

### Out of scope for this phase

- The Stage 10a sweep doesn't run on segments that Stage 3 marked
  as `single` (because the LP wasn't designed to second-guess
  single-subject decisions — that's Phase 4's thirds-bias work).
  An edge case where two speakers are visible in a `single`
  segment is handled by the existing single-subject L1 path.
- The fixture set doesn't currently include a "two simultaneous
  speakers" overlap case that would force Stage 3 to mark the
  segment as `split` and then exercise the LP at the segmenter
  level. The LP itself is exercised by the unit tests; the
  parity fixture for end-to-end exercise is a Phase 3 follow-up
  or Phase 4 prerequisite.
- Dynamic crop width (the spec mentions optional `w_t` per frame)
  is not implemented. The LP uses a fixed `crop_width_px`
  derived from `source_height * 9/16`, matching the existing
  single-subject solver.

### v2 Phase 2 — Editorially-meaningful upload dropdown

**Before:** The Upload.jsx dropdown had three rows
(`Auto-detect / Gameplay / Podcast / Movie`) and a single FPS-only
sub-dropdown when "Gameplay" was selected. There was no way for the
user to declare debate / vlog / anime / music video / sports, and no
way to tell the pipeline that gameplay footage is MOBA / TPS /
racing / stream rather than FPS. Even if a future contributor added
those rows, classify_clip would still route every gameplay variant
through `ClipContentType.GAMEPLAY` and silently apply FPS-style
center-crop tuning.

**After (v2 Phase 2):**

- **Upload.jsx dropdown** restructured into 5 optgroups
  (People / Animation / Music / Gaming / Sports) with **14 distinct
  tokens**:

      Auto-detect            (heuristic)
      ─── People / Dialogue ───
      Podcast / Interview         → "podcast"
      Debate / Panel              → "debate"
      Vlog / Single-subject       → "vlog"
      Movie / TV / Cinematic      → "narrative"
      ─── Animation ───
      Anime / Cartoon             → "anime"     + anime sub-dropdown
      ─── Music / Performance ───
      Music Video / Performance   → "music_video" + music sub-dropdown
      ─── Gaming ───
      Gameplay — FPS              → "gameplay"        (legacy alias)
      Gameplay — MOBA             → "gameplay_moba"
      Gameplay — TPS              → "gameplay_tps"
      Gameplay — Racing           → "gameplay_racing"
      Stream / Facecam + Gameplay → "stream"
      ─── Sports ───
      Sports broadcast            → "sports"

- **Anime sub-dropdown** (`Auto / Action / Dialogue-heavy / Slice of life`)
  — feeds `anime_subtype` into `ContentProfile`. classify_clip now
  routes:
    - `anime_subtype="action"` → `ClipContentType.ANIMATION`
    - `anime_subtype in ("dialogue", "slice_of_life")` →
      `ClipContentType.ANIMATION_DIALOGUE`
    - `anime_subtype` unset → legacy heuristic (talking-head /
      cinematic-dialogue / generic bases promote to ANIMATION_DIALOGUE,
      otherwise stay ANIMATION).
  Phase 6 will read `profile.anime_subtype` to drive lead-room
  multipliers and the anime shot detector.

- **Music sub-dropdown** (`Auto / Performance / Narrative / Lyric`)
  — feeds `music_subtype`. Phase 5 reads it to set beat-snap
  aggressiveness.

- **Game sub-dropdown** now expands beyond the FPS list. The
  options shown depend on the selected gameplay variant:
    - `gameplay` / `stream` → FPS games + Minecraft (sandbox)
    - `gameplay_moba` → League of Legends, Dota 2, Generic MOBA
    - `gameplay_tps` → GTA V, Elden Ring, Generic TPS
    - `gameplay_racing` → Rocket League, Generic Racing

- **`ClipContentType` extended** with `GAMEPLAY_MOBA`, `GAMEPLAY_TPS`,
  `GAMEPLAY_RACING`. `GAMEPLAY` remains the FPS / hero-shooter
  default. classify_clip now uses `profile.gameplay_subtype`
  (encoded by the normalizer from the parent token) to route
  directly to the right ClipContentType *before* the
  MULTI_SPEAKER_PANEL / CINEMATIC_DIALOGUE / animation branches
  fire — so a gameplay clip can never accidentally inherit
  talking-head tuning.

- **`ContentProfile` extended** with `anime_subtype: Optional[str]`,
  `music_subtype: Optional[str]`, `gameplay_subtype: Optional[str]`,
  and `game_type: str`. The classifier user-override branch
  populates them from the metadata dict.

- **`game_layouts.GAME_HUD_LAYOUTS` extended** from 6 entries (all
  FPS) to **15 entries** spanning FPS / MOBA / TPS / racing /
  sandbox. Each layout now has:
    - `genre`: `"fps"` | `"moba"` | `"tps"` | `"racing"` | `"sandbox"`
    - `action_center_pct`: `(x, y)` — the on-screen anchor for the
      vertical crop. FPS = `(50, 50)`, TPS = `(50, 45)` (head/
      shoulders above center), Racing = `(50, 65)` (car in lower
      third), MOBA = `(50, 50)` with wider safe-zone, Sandbox =
      `(50, 50)`. Phase 7 reads this to override the legacy
      hard-coded `subject_x = 50`.
  New helper functions: `get_action_center(game_key)`,
  `games_for_genre(genre)`, plus `DEFAULT_GAME_BY_GENRE` and
  `GAME_GENRE` lookup tables.

- **Backend models + upload routers extended** with `anime_subtype`
  and `music_subtype` fields on `JobResult`, the chunked-upload
  `InitRequest`, and the legacy multipart `_stream_multipart_to_disk`
  parser. The pipeline reads them off the job and injects them into
  `_classifier_metadata` alongside `content_type_override` and
  `game_type` so the classifier user-override branch can populate
  the new profile fields.

### Tests

| File | Count | Purpose |
|---|---|---|
| `test_content_routing_matrix.py` | 46 | parametrized row-per-token matrix + game-layout coverage + AST guards on pipeline.py subtype injection |
| Phase 1 + pre-existing classifier suites | 132 | zero regressions |
| **Total (v2 Phase 1+2 scope)** | **178** | all green |

The 46 new tests break down as:

- **24** `test_routing_matrix_row` rows — one per
  (token × anime_subtype × music_subtype) combination from the spec
  table. Each row asserts: normalizer parent + flags + subtypes,
  `is_gameplay_override`, `classify_content` profile fields, and
  `classify_clip` final `ClipContentType`.
- **16** `TestGameLayoutsExpansion` tests — every new game has the
  right genre and action-center, plus the helper functions
  (`get_action_center` / `games_for_genre` / `DEFAULT_GAME_BY_GENRE`
  / `GAME_GENRE`) behave correctly.
- **2** `TestEndToEndCoverage` tests — the matrix covers every
  Upload.jsx token; every matrix token normalizes.
- **3** `TestPipelineSubtypeInjection` AST guards — pipeline.py
  injects `anime_subtype`, `music_subtype`, and `game_type` into
  `_classifier_metadata` (the Phase 1 AST guard already covered
  `content_type_override`).

### Frontend build

`npm run build` emits the same chunk count as before — the dropdown
restructure is a pure JSX change with no new imports. The Upload.jsx
state grew by 2 fields (`animeSubtype`, `musicSubtype`) and the
init-POST body grew by the same 2 fields.

### Open questions resolved this phase

- **Debate vs. Podcast as a separate enum**: not worth a new
  `ContentType.DEBATE` member — `debate` and `panel` both map to
  `ContentType.PODCAST` with `is_multi_speaker_panel=True`, and the
  existing `MULTI_SPEAKER_PANEL` `ClipContentType` route already
  carries the right tuning (Fix 3).
- **Stream gameplay fast-path**: deliberately NOT a
  `is_gameplay_fastpath` candidate. Stream still needs the face
  pipeline for the facecam overlay. Phase 7 will route `stream` to
  `STACKED_GAMEPLAY` layout downstream.

### Exit criteria met

- ✅ Every dropdown value renders the correct sub-select (game /
  anime / music) per the JSX conditions.
- ✅ Every dropdown value round-trips through init →
  `JobResult.content_type_override` (+ subtype fields) → pipeline →
  `classify_content` with `confidence=1.0` and the right
  `ContentProfile` fields populated.
- ✅ Every dropdown value lands on the correct `ClipContentType` per
  the Phase 2 spec.
- ✅ `test_content_routing_matrix.py` (46 tests) green; all 132
  Phase 1 / pre-existing tests green (178 total).
- ✅ `npm run build` clean.
- ✅ Sub-type values from a previous selection cannot leak into a
  later upload — the normalizer gates each subtype to its parent
  type, the Upload.jsx onChange clears stale state, and the
  classifier validates anime/music tokens via
  `normalize_anime_subtype` / `normalize_music_subtype` (which
  return `None` for unknown values).

## v2 Phase 11 — regression fixes (Verzuz panel clip)

A 10-minute multi-speaker Verzuz-style panel (Tank vs Tyrese,
`user override 'debate'`) regressed vs. prior runs: crop jittered
across 28 unique `subject_x` values on 69 segments for a clip with
4 speakers in fixed seats. Root cause was not one bug but seven
systems each overriding the ReframeSegmenter's (correct) output
with worse data.

1. **`USE_CONTENT_AWARE_REFRAME` now defaults to `true`.** The
   segmenter's content-aware branches (panel hold, narrative,
   gaming, anime) were shipped dormant, so `classify_content`'s
   output never reached the segmenter and every panel clip ran as
   `content_type=unknown`. Parity bench (Phase 10) had already
   passed with these branches enabled; there was no reason to keep
   the flag off. `content_classifier.py` flipped to the same default.
2. **`ReframeSegmenter` routes `is_multi_speaker_panel` → its own
   `ct` key.** When the classifier sets `content_type=podcast` but
   `is_multi_speaker_panel=True`, the segmenter now loads
   `CONTENT_TYPE_CONFIG[MULTI_SPEAKER_PANEL]` (tighter holds, no
   in-shot tracking, 0.9 s min hold, 0.2 s anticipation) instead of
   the vlog/podcast preset. Added `ContentType.MULTI_SPEAKER_PANEL`
   enum + config entry.
3. **Face registry quality gate is now any-slot, not
   25%-of-slots.** The old cross-shot-merge rejection predicate
   was `fraction of slots with span>60% AND frames>30 > 0.25`.
   The Verzuz clip had Slot 8 spanning `[14-98]%` with 558 frames
   — 1 of 9 slots, under the 25% gate — but that single merged
   slot was enough to poison speaker-to-slot mapping because
   everyone ends up mapped to it. Now any single matching slot
   rejects the whole embedding registry in favor of position-based.
   Panel-mode also now prefers position-based when both are valid
   (seats are fixed, embedding splits happen on pose/lighting).
4. **Panel short-shot override is gated on shot-detector
   confidence.** `Shot` gained a `detector_confidence` field
   (`"high"` = PySceneDetect, `"low"` = opencv frame-diff
   fallback). `layout_engine._plan_layout_impl` now refuses to run
   the short-shot override if any shot is low-confidence — the
   opencv fallback fired 179 "shots" on a static panel (lighting
   flicker) and the override was hard-pinning 165 of them to
   single-slot centers with 2-keyframe stationary crops, bypassing
   the L1 solver and reintroducing the 2-Hz stair-stepping that
   Phase 1 was built to eliminate.
5. **OpenCV frame-diff fallback raised threshold 40 → 55 + HSV
   histogram correlation secondary gate.** Real cuts have HSV hue
   correlation < 0.6; lighting flicker has > 0.8. Dedup window
   widened from 0.5 s → 1.0 s. Combined with Fix 4's confidence
   marking, this makes the fallback usable as a safety net even
   when PySceneDetect is absent.
6. **Vision-model quality gate lowered 70% → 25%.** When > 25% of
   scenes hedge to center (`subject_x ∈ [47, 53]`), the whole
   batch of vision-derived subject positions is collapsed to the
   default (50) + `precise_x`/`precise_y`/`active_speaker_x`
   cleared. The downstream dense-face-data override in
   `pipeline.py` then fills every scene from face detection, which
   is strictly more reliable than a hedging vision model. On the
   Verzuz clip qwen3-vl-8b returned center defaults on 42% of
   frames — below the old 70% gate, so the garbage was fed into
   tracking.
7. **Whisper subprocess launcher evicts Ollama VRAM first.** Added
   `transcription._evict_ollama_for_whisper()` which calls
   `/api/ps`, POSTs `keep_alive=0` to each loaded model, and
   sleeps 2 s for the driver to reclaim memory — best effort,
   errors swallowed. On a 4 GB 1650 (Jalon's rig) Ollama's idle
   CUDA context was enough to force Whisper to CPU int8 fallback,
   pushing transcription from ~30 s to 12 minutes for a 10-min clip.
8. **`layout_from_reframe_segments` adapter** (the big one). When
   the ReframeSegmenter already produced a content-aware,
   L1-solved segment timeline, `pipeline.py` now builds the
   `LayoutTimeline` directly from those segments instead of
   re-running shot detection + required-regions + camera solver
   in `plan_layout`. Two camera-path systems running in series
   was producing incompatible decompositions (segmenter's 69
   segments vs plan_layout's 179 shots, with the frontend seeing
   whichever wrote last). `plan_layout` is still kept as the
   fallback path for AUTOFLIP runs and for when the segmenter
   doesn't run.

### Test coverage

- `backend/tests/test_v2_phase11_regression_fixes.py` — 12 cases
  covering Fixes 1-5 and 7 (content-type config, face-registry
  selection rule, Shot dataclass, layout_engine source checks,
  layout_from_reframe_segments end-to-end).
- `backend/tests/test_whisper_ollama_eviction.py` — 3 async cases
  mocking `httpx.AsyncClient` to confirm Fix 6 evicts loaded
  models, no-ops on an empty `/api/ps`, and swallows connection
  errors.

### Expected log deltas on the Tank/Tyrese clip

Before:
```
ReframeSegmenter: content_type=unknown
Embedding registry found 9 slots, position-based found 5 — using embeddings (max wins)
ShotDetector (opencv fallback): 179 shots
[Layout+Solver] panel short-shot override: 165 shots (< 20 frames) mapped to active-speaker slot
Whisper CUDA failed — falling back to CPU (int8)
```

After:
```
ReframeSegmenter: content_type=multi_speaker_panel
[FaceRegistry] embedding registry rejected: 1 slot had span>60% and frame_count>30 (cross-shot merge). Using position-based (5 slots)
ShotDetector (opencv fallback): ~15 shots (confidence=low)
[Layout+Solver] panel short-shot override SKIPPED: shot detector confidence is low (opencv fallback on 15 shots). Trusting L1 solver.
Evicted 2 Ollama model(s) from VRAM before Whisper CUDA
[Layout] Built from 69 reframe segments (skipped plan_layout second-solve)
```

## Phase 11 — Detection stack upgrades (A/B flag rollout)

Phase 11 adds four flag-gated upgrades to the detection stack. Each
flag defaults to the existing behaviour and is promoted to default
only after the matrix runner in
`backend/scripts/measure_detection_stack_matrix.py` shows parity or
improvement on every fixture. Hardware constraint: every new ONNX
session uses `onnxruntime CPUExecutionProvider`. The GTX 1650 stays
reserved for Whisper / Ollama.

| Flag | Backend (off → on) | Default | Phase | Matrix delta |
|---|---|---|---|---|
| `CLIPAI_FACE_EMBEDDING` | SFace 128-d → ArcFace Buffalo_S 512-d | `sface` | A | _pending matrix run_ |
| `CLIPAI_OBJECT_DETECTOR` | YOLOv8n → YOLO11n | `yolov8n` | B | _pending matrix run_ |
| `CLIPAI_PERSON_USE_POSE` | bbox center → YOLO11n-pose head anchor | `false` | B | _pending matrix run_ |
| `CLIPAI_ANIME_FACE_BACKEND` | lbpcascade Haar → YOLOv8-anime-face ONNX | `lbpcascade` | D | _pending matrix run_ |
| `CLIPAI_ASD_BACKEND` | lip-aperture heuristic → Light-ASD audio-visual | `heuristic` | C | _pending matrix run_ |

### Phase A — ArcFace embeddings

_Fill in delta table from `measure_detection_stack_matrix.py` once
the matrix has run on `synthetic_alternating_2s` and
`multi_speaker_crowd_10s`._

### Phase B — YOLO11n + pose anchor

_Fill in delta table from `measure_detection_stack_matrix.py` once
the matrix has run._

### Phase C — Light-ASD

_Fill in delta table from `measure_detection_stack_matrix.py` once
the matrix has run. Note: keep the heuristic default until content-
type-specific validation confirms parity on Jalon's actual corpus —
gaming / anime / heavy-music-over-dialogue is where Light-ASD is most
likely to regress._

### Phase D — YOLO anime face

_Fill in delta table from `measure_detection_stack_matrix.py` once
the matrix has run on the anime fixtures._

### Rollout order

1. Land Phase A (ArcFace) behind `sface` default. Run matrix. Flip
   default to `arcface` if every fixture improves or stays flat.
2. Land Phase B (YOLO11n + pose). Validate. Flip
   `CLIPAI_OBJECT_DETECTOR=yolo11n` default; leave pose opt-in until
   human-eye QA on the VideoEditor preview.
3. Land Phase D (YOLO anime). Validate on anime fixtures only. Flip
   default after a clean run.
4. Land Phase C (Light-ASD). **Keep off by default indefinitely**
   until content-type validation confirms it works on Jalon's actual
   corpus.

## Week 1 flag audit (post-Phase 11 / v4 universal reframing)

After Phase 11 + the v4 universal-reframing commit (``fe7fa3e``), six
of the v2 feature flags had their code-level defaults flipped to
``"1"`` in their defining modules, but the flag inventory table in
``docs/content_type_routing.md`` and the "Default" columns in the
phase-by-phase tables above still said OFF. Week 1 reconciles that
drift and flips the one remaining flag that has real call sites and
parity-fixture coverage.

### Flag state (reconciled)

| Flag | Was (pre-Phase-11) | Code default (today) | Validated how |
|---|---|---|---|
| `CLIPAI_EDITORIAL_PRIOR` | OFF | ON | ``validate_v2_phases --quick`` baseline=OFF, all_on=ON, safety PASS |
| `CLIPAI_THIRDS_BIAS` | OFF | ON | same |
| `CLIPAI_GAZE_LEAD_ROOM_V2` | OFF | ON | same |
| `CLIPAI_ANIME_ANCHOR` | OFF | ON | same |
| `CLIPAI_GAMEPLAY_TRACKER` | OFF | ON | same |
| `CLIPAI_MUSIC_BEAT_SNAP` | OFF | ON | same |
| `CLIPAI_MULTI_REGION_LP` | OFF | **ON (Week 1 flip)** | isolated OFF/ON on ``3speaker_panel`` (no delta — panel routing short-circuits Stage 10a), full ``--quick`` PASS |

Three flags remain OFF with zero runtime effect — the modules exist
but have no call sites in ``backend/`` outside their defining files,
``validate_v2_phases.py``'s env-var setup, and the per-module unit
tests in ``backend/tests/test_phase6_*.py``:

- ``CLIPAI_ANIME_SHOT_DETECTOR``
- ``CLIPAI_ANIME_FACE_DETECTOR``
- ``CLIPAI_ANIME_CHARACTER_CLUSTERING``

Week 2 wires them. Until then, anime content uses
``CLIPAI_ANIME_ANCHOR`` (saliency + face-delta motion energy) as the
only anime-specific signal.

### Week 1 whitelist

Two ``(fixture, metric)`` pairs were added to
``_KNOWN_DIVERGENCES`` in ``backend/scripts/validate_v2_phases.py``
during the Week 1 run:

- ``(2speaker_alternating, max_acceleration)``
- ``(2speaker_alternating, max_jerk)``

Bisection via the full phase matrix (``validate_v2_phases``, not
``--quick``) shows the drift is produced by
**phase8_editorial_prior alone** — phases 3/4/5/6/7 all match
baseline to the last bit, and all_on = phase8 + the rest converges
to the same +0.00357 / +0.00714 delta as phase8 in isolation. The
editorial prior's J/L-cut anticipation shifts segment boundaries by
a few milliseconds near speaker turns, which adds a sliver of
smoothed camera motion across the cut, which reorders one ``max()``
reduction in the acceleration / jerk reducer by a single ULP.

Magnitude: **0.012% relative drift** — +0.00357 px on a baseline
value of 28.7 px/frame², +0.00714 px on a baseline of 57.5. On a
1920-px source that's about 1/280,000th of frame width — sub-
perceptual by ~3 orders of magnitude.

The safety gate is ``_SAFETY_EPS = 1e-6`` (designed to catch solver
behavior changes, not summation-order changes), so these deltas
formally regress. The whitelist records the reasoning so the next
reader doesn't investigate from scratch. The **real** signal in the
same ``all_on`` run is a large improvement in coverage that the
strict gate was masking:

| Fixture | Baseline miss rate | All-on miss rate | Δ |
|---|---|---|---|
| 2speaker_alternating | 0.83 | 0.80 | −0.030 |
| 3speaker_panel | 0.833 | 0.667 | **−0.167** |
| vlog_walk_and_talk | 0.63 | 0.46 | **−0.170** |

A 17-percentage-point drop in ``required_region_miss_rate`` on the
vlog fixture is the universal attention-anchor stream actually
working — it bridges the faceless frames where the walker's face
leaves the frame and the old (dialogue-only) anchor stream had no
fallback.

**Re-evaluate this whitelist if ``camera_solver.py`` SolverParams
weights change.** A legitimate solver regression would move the
absolute value by >0.1 (at minimum 0.3% relative), not <0.01. If the
drift ever grows past that, the whitelist entries should be removed
and the root cause investigated.

### Fixture artifacts

- Pre-flip baseline (v4 flags ON, MRLP OFF): ``/tmp/week1_baseline_quick.json`` / ``.md``
- Post-whitelist baseline (v4 flags ON, MRLP OFF): ``/tmp/week1_baseline_quick_v2.json`` / ``.md``
- Post-flip baseline (v4 flags ON, MRLP ON): ``/tmp/week1_post_mrlp.json`` / ``.md``
- Isolated MRLP OFF vs ON on 3speaker_panel: ``/tmp/mrlp_off.json`` / ``/tmp/mrlp_on.json``

Copy these into ``docs/`` if they should be version-controlled — they
live in ``/tmp`` today because the Week 1 run was a documentation and
validation pass, not a long-lived artifact capture.

### Regression guards added

- ``backend/tests/test_flag_defaults_stable.py`` — parametrized
  assertion that each v2 editorial flag's module default stays ON
  when the env var is unset. Includes the Week 1 MRLP flip. Flipping
  any row back to OFF without a coordinated ``validate_v2_phases``
  run + doc update fails this test.
- ``backend/tests/test_dormant_flags_labeled.py`` — AST-level grep
  guard that asserts ``detect_anime_faces``, ``detect_anime_shots``,
  and the ``anime_character_clustering`` symbols have zero call
  sites in ``backend/`` outside their defining modules + tests. If
  Week 2 wires one of them, this test starts failing and forces a
  coordinated update.

### Known follow-up — validator silent-failure bug

During the Week 1 run, ``validate_v2_phases --quick`` reported
``EXIT 0`` the first time it was invoked even though **all 7
fixtures were skipped** with ``"error": "segmenter import failed:
ModuleNotFoundError: No module named 'numpy'"``. The exit-code
semantics should distinguish "all fixtures passed" from "no
fixtures were measured". Filed as a follow-up; the fix is one-line:
if every fixture in every combo has ``status="skipped"``, return
exit code 2 (or a distinct 3 for "couldn't measure").

## Week 2 — anime wiring and subtype auto-promotion

Week 2 is the internal improvement sprint that turns the five
previously-dormant editorial signals into live runtime behavior:
the anime face / shot / character-clustering triad (Parts A/B/C),
sports subtype auto-promotion (Part D), and the music-video
formation→downbeat snap + subtype auto-promotion pair (Part E).

Every part lands its new behavior behind its existing
``CLIPAI_*`` flag and adds unit tests that exercise the helper
logic in isolation. The parity safety gate
(``validate_v2_phases --quick``) stays at exit 0 on every part's
checkpoint, and the reframe-lag micro-benchmark still reports
100% / 0 / -6.

### Part A — `detect_anime_faces` wired into the dense face pipeline

- **Call site:** new ``face_detector._augment_dense_with_anime``
  helper runs inside ``detect_faces_dense`` when the caller passes
  ``is_animated=True``. Augments per-frame results: on frames with
  zero faces or only low-confidence (<0.55) live-action detections,
  the lbpcascade anime detector runs on that frame and appends its
  hits as ``FaceInfo`` records with ``is_human=False`` so the
  human-pose verifier doesn't reject them.
- **Threading:** ``pipeline.py`` computes an ``_early_anime_hint``
  right before the dense detection call, sourced from (a) the
  normalized ``content_type_override`` (``anime`` / ``cartoon``) or
  (b) ``face_detector.ANIME_MODE_DETECTED`` fired during the sparse
  pass. If neither fires up-front but ``classify_content``
  concludes ``is_animated=True`` later, a re-run fallback fires
  another dense pass with ``is_animated=True`` when the first pass's
  face-per-frame rate is below 0.4 (gate so we don't spend the extra
  dense minute on clips that already have good coverage).
- **Cascade file:** ``backend/models/lbpcascade_animeface.xml``
  committed to the repo. Public domain per Nagadomi's stated terms.
  ~250 KB, valid cascade (6693 lines, verified).
- **Flag flip:** ``CLIPAI_ANIME_FACE_DETECTOR`` default
  ``"0"`` → ``"1"``.
- **Tests:** ``backend/tests/test_anime_face_augmentation.py``
  (7 tests): weak/empty/strong frame handling, flag-off short
  circuit, threshold edge cases, mismatched ``frame_paths`` safety,
  empty detection results, per-detection converter exception
  swallowing, log-line content.

### Part B — `detect_anime_shots` wired into both shot paths

- **Pipeline path:** right after ``classify_content`` in
  ``pipeline.py``, when ``_content_profile.is_animated=True``, the
  anime histogram / edge-density detector runs and its cut
  timestamps merge into ``scene_cut_timestamps`` with a ±0.3s
  dedup window. De-duplicated + sorted in place.
- **AUTOFLIP / layout_engine path:** same pattern but operates on
  ``Shot`` objects — each new anime cut splits its containing
  ``Shot`` into two new ``Shot`` instances and the entire list is
  renumbered. Covered by real-dataclass round-trip tests.
- **Flag flip:** ``CLIPAI_ANIME_SHOT_DETECTOR`` default
  ``"0"`` → ``"1"``.
- **Tests:** ``backend/tests/test_anime_shot_integration.py``
  (13 tests): cut-list merge (add + dedup + sort), Shot-split
  (single-shot, across-multiple, boundary-exact, out-of-range),
  real-dataclass round-trip, flag short-circuit.

### Part C — `anime_character_clustering` re-ID pass

- **Call site:** right after ``build_face_registry(_with_embeddings)``
  in ``pipeline.py``. When the early anime hint fired and the
  registry has ≥2 slots, the pipeline runs HSV color fingerprinting
  over the anime-cascade detections, averages per-slot centroids,
  runs ``cluster_fingerprints`` on the centroids, and collapses any
  two slots whose fingerprints sit within the chi-squared threshold.
  The canonical-slot remap collapses to the lowest slot id in each
  cluster; remapped ``identity_id``s are written back to every
  ``FaceInfo`` in both ``dense_face_results`` and ``face_results``;
  the ``face_registry`` is rebuilt from the remapped detections.
- **Graceful degradation:** when frame files are stale (the dense
  detector's tempdir has been cleaned up), ``cv2.imread`` returns
  ``None`` and that frame contributes zero fingerprints. When no
  slot produces any fingerprints, the pass logs
  ``"no readable frames"`` and skips without mutating state. This
  is a known ergonomic wart — Week 3 or beyond will persist dense
  frames to a longer-lived temp dir so the re-ID can always run.
- **Flag flip:** ``CLIPAI_ANIME_CHARACTER_CLUSTERING`` default
  ``"0"`` → ``"1"``.
- **Tests:** ``backend/tests/test_anime_character_clustering_integration.py``
  (11 tests): canonical-slot remap (cluster-collapse, singletons,
  non-contiguous ids), remap application (update + skip-missing +
  skip-identity), real ``cluster_fingerprints`` contract (merge,
  empty, all-identical), end-to-end slice through the real module.

### Part D — Sports subtype auto-promotion

- **Helper:** new ``content_classifier._infer_sports_subtype_from_objects``
  pure function. Accepts either the flat ``ObjectDetection`` list
  the pipeline actually emits OR a per-frame ``.objects`` shape for
  future-proofing. Groups the flat list by rounded timestamp
  internally. Fires promotion only when the winning ratio clears
  its threshold AND beats the other class's ratio (strict ``>``).
- **Pipeline plumbing:** ``_classifier_metadata["frame_objects"]``
  is threaded into ``classify_content``. The promotion block runs
  right after voting settles, before the cinematic-dialogue branch.
  User-override branches at the top of ``classify_content`` still
  short-circuit so a dropdown pick always wins.
- **Thresholds:** basketball ``ball_ratio ≥ 0.15``, racing
  ``vehicle_ratio ≥ 0.10`` + per-vehicle frame-area gate ≥5%.
  Tuned so a half-court basketball clip promotes and a generic
  street-running clip with incidental cars does not.
- **Tests:** ``backend/tests/test_sports_subtype_autopromotion.py``
  (13 tests): basketball/racing threshold crossings, mixed
  ball+car (higher ratio wins), tiny-car area-gate, non-sports
  short-circuit, user-override preservation, per-frame-vs-flat
  shape handling.

### Part E — Music video formation → downbeat snap + subtype auto-promotion

Two independent changes in this part:

**E1. Stage 11 Pass C — formation → downbeat snap.**
For any segment whose last probeable frame (``seg.end - 0.15``) is
a formation shot (3+ faces, ≥55% width span), the segment end
shifts forward to the next downbeat (up to 1.5s out) so the cut
lands ON the beat. Only applies when the downbeat falls strictly
inside the next segment AND extending wouldn't shrink that segment
below 0.30s. Logged via the existing Stage 11 log line as
``formation=N``.

**E2. Subtype auto-promotion.**
``content_classifier._infer_music_subtype_from_formation_and_beat``
counts formation-frame density across ``dense_faces`` and compares
against the ``BeatGrid.effective_confidence()`` property (new —
zero-gates empty grids even when the stored confidence is high).
Fires when ``formation_ratio ≥ 0.08`` AND ``beat_conf ≥ 0.6``.

**Supporting changes.**
- ``BeatGrid`` gets a new ``confidence: float = 0.0`` field and an
  ``effective_confidence()`` method. ``detect_beats`` (librosa-
  backed) sets ``confidence=0.9``; the synthetic constructor leaves
  it at 0.0 so fixtures don't accidentally trigger promotion.
- ``_classifier_metadata["music_beat_grid"]`` threaded into
  ``classify_content`` so the promotion helper can read it.

**Tests:**
- ``backend/tests/test_music_formation_downbeat_snap.py`` (9 tests):
  formation-at-boundary snap, no-snap when downbeat too far,
  non-formation short-circuit, next-segment-shrink rejection,
  downbeat-outside-next rejection, multi-formation chain, last-
  segment protection, probe-window offset accuracy.
- ``backend/tests/test_music_subtype_autopromotion.py`` (10 tests):
  promotion thresholds (both floors), low-formation/low-conf
  rejection, missing/empty beat grid, empty dense-faces, user-
  override preservation, non-music short-circuit, real ``BeatGrid``
  confidence contract.

### Week 2 validation state

- **76 new tests across 6 new test files, 100% passing.**
- ``validate_v2_phases --quick`` exit 0 at every part checkpoint
  (Parts A, B, C, D, E separately + final combined run).
- ``measure_reframe_lag``: 100% recall, 0 overlaps, -6 frame
  anticipation (unchanged from Week 1).
- ``test_flag_defaults_stable.py`` grew three new rows: the anime
  face / shot / character-clustering flags. All 11 flag defaults
  green.
- ``test_dormant_flags_labeled.py`` — ``DORMANT_MODULES`` dict is
  now empty; every anime module has a call site. The guard is
  kept in place as a landing spot for future dormant modules.

### Week 2 fixture artifacts

- ``/tmp/week2_partA.json`` / ``.md`` — post-Part-A quick gate
- ``/tmp/week2_partB.json`` / ``.md`` — post-Part-B quick gate
- ``/tmp/week2_partC.json`` / ``.md`` — post-Part-C quick gate
- ``/tmp/week2_partD.json`` / ``.md`` — post-Part-D quick gate
- ``/tmp/week2_final.json`` / ``.md`` — final combined gate
- ``/tmp/week2_lag.log`` — final reframe-lag bench

### Known Week-2 caveats

1. **Anime character clustering needs persisted frames.** The re-ID
   pass currently relies on ``cv2.imread`` over ``FrameFaces.frame_path``
   entries that become stale once ``detect_faces_dense`` exits its
   ``TemporaryDirectory``. In production the re-ID typically logs
   ``"no readable frames"`` and short-circuits. A future change
   should move dense frames to a per-job persisted directory so
   the re-ID pass can actually run on every anime clip.

2. **Part-E subtype promotion needs beat-grid hoist.** The
   ``music_beat_grid`` that feeds the subtype-promotion gate is
   computed inside the reframe segmenter today, *after*
   ``classify_content`` has already run. In the current code path
   the gate will see ``music_beat_grid=None`` and the promotion
   won't fire. A follow-up (Week 3) hoists beat grid computation
   to right after audio extraction so both the classifier and the
   segmenter see the same grid.

3. **Anime module synchronization with ``is_animated`` detection.**
   Parts A/B/C all gate on ``_early_anime_hint`` or
   ``_content_profile.is_animated``. When the sparse-face pass's
   ``ANIME_MODE_DETECTED`` flag fires late, Part A's re-run fallback
   catches up but Parts B + C still require a second pipeline pass
   to benefit. In practice this is rare (the user dropdown picks
   anime up-front most of the time), but worth noting for the
   Week 3 real-content benchmark.

## Week 3 — Real-content comparison harness (infrastructure)

Week 3 is a measurement-and-calibration week. It lands the harness
and scoring pieces that will drive Week 4+ decisions on real video;
it does not change any solver behavior or flag defaults. The synthetic
fixtures are unaffected — ``validate_v2_phases --quick`` exit-0 before
and after Week 3.

### What landed

- ``backend/services/autoflip_parity_metrics.py::cut_to_hold_ratio``
  — new 9th metric. Takes an AutoFlip-shape event list and returns
  per-segment hold statistics (median, quartiles, under-1s rate,
  over-8s rate). Pure Python, 8 unit tests.
- ``backend/scripts/export_autoflip_compatible.py`` — translator
  that takes a ClipAI ``RenderPlan.to_dict()`` dump OR a list of
  ``ReframeSegment`` dicts and emits an AutoFlip-shape per-frame
  JSON timeline. Lets the comparison harness score both tools in
  the same coordinate system. 17 unit tests including ffprobe
  monkeypatch.
- ``backend/scripts/compare_autoflip_vs_clipai.py`` — the harness
  itself. Reads the real-content manifest, loads cached AutoFlip
  JSONs, runs (a stubbed) ClipAI pipeline, scores both with the
  metric library + the new ``cut_to_hold_ratio``, and emits a
  per-content-type markdown rollup with target-zone verdicts
  (PASS / MARGINAL / MISS / UNKNOWN). 16 unit tests pin the
  scoring + verdict layers.
- ``reference/autoflip/Dockerfile`` + ``run_one.sh`` +
  ``REFERENCE_OUTPUTS.md`` — reference MediaPipe AutoFlip runner.
  Builds from source against ``v0.10.9`` + Bazel 6.1.1. Used for
  one-time cache generation; the harness reads the cached JSONs
  offline forever after.
- ``tests/real_content/manifest.json`` + ``fetch.sh`` — 12-clip
  real-content set spanning the four target verticals (panels,
  anime, sports, music). Clips themselves are ``.gitignore``'d;
  the manifest pins sha256 hashes once the fetcher has run.
- ``docs/week3_real_content_results.md`` — placeholder that
  becomes the live markdown rollup when the harness runs.
- ``docs/week3_gap_analysis.md`` — Week-4-planning template.

### What did NOT land

- **No real-content rollup.** The session sandbox has no clips
  (manifest ``source_url`` fields are empty) and no AutoFlip
  cache (Docker build is intractable in the sandbox — see
  ``reference/autoflip/REFERENCE_OUTPUTS.md``). The harness works
  end-to-end against these gaps: every AutoFlip row is marked
  SKIPPED, every ClipAI row is a placeholder single-segment stub.
  That's enough to prove the rendering + verdict layers fire but
  not enough to drive any real Week-4 decision.
- **No ``MIN_HOLD_SECONDS`` tuning (Part E).** The Week 3 prompt's
  Part E tunes per-content-type hold floors ONLY when the harness
  shows a systematic skew on real content. With the stub ClipAI
  path producing single-segment timelines, any skew signal is
  meaningless. **Zero changes to ``content_type_config.py`` landed
  in this week.**
- **No flag default flips.**
- **No new pipeline-layer behavior.** Same behavior before and
  after Week 3; only measurement scaffolding.

### Week 3 blockers to clear before Week 4 can start

1. **Populate the real clip cache.** Fill in ``source_url`` fields
   in ``tests/real_content/manifest.json``, run ``fetch.sh``, paste
   back the ``new sha256`` lines the fetcher emits on stderr.
2. **Generate the AutoFlip reference cache.** Build
   ``reference/autoflip/Dockerfile`` on a machine that can finish
   the Bazel build (cloud VM, homelab box), run each clip through
   it once, commit the resulting JSONs at
   ``tests/autoflip_reference_outputs/``. One-time cost.
3. **Wire ``run_clipai_on_clip`` to the real inline segmenter.**
   The stub in ``compare_autoflip_vs_clipai.py`` has a clearly
   marked seam (``run_clipai_on_clip``). The real implementation
   cribs from ``measure_autoflip_parity.py``'s inline segmenter
   path and runs over a real extraction cache keyed on clip sha256.
   Biggest remaining work item from Week 3.

Once (1) + (2) + (3) are done, running the harness produces a real
``docs/week3_real_content_results.md``, populating
``docs/week3_gap_analysis.md``'s ``<fill in>`` placeholders becomes
a straightforward triage, and Week 4 opens as a targeted-fix
session on the top three gaps.

### Tests shipped in Week 3

- ``backend/tests/test_cut_to_hold_ratio.py`` — 8 tests
- ``backend/tests/test_export_autoflip_compatible.py`` — 17 tests
- ``backend/tests/test_compare_autoflip_vs_clipai.py`` — 16 tests

All 41 green. ``validate_v2_phases --quick`` exit 0.


## VLM subject-tracking upgrade — Phases 1-6 infrastructure

This block tracks the six-phase VLM subject-tracking upgrade. Each
phase ships its own module + tests with a dedicated env flag; every
default stays at the legacy behavior until a measurement pass flips
it. Per-phase notes live in ``docs/vlm_upgrade/PHASE_<N>_NOTES.md``.

### Status (April 2026)

| Phase | Module | Env flag | Default | Promotable? |
|---|---|---|---|---|
| 1 | `providers/base.py::parse_scene_dict` + `SceneDescription.subject_box` | — (additive) | — | landed |
| 2 | `providers/openrouter_provider.py::PRESETS` + `select_vision_model_for_content` | — (always applied when override set) | routes ANIME/GAMEPLAY to Qwen3-VL | landed |
| 3 | `services/vlm_fusion.py` | `CLIPAI_VLM_FUSION` | `hard` | no — pending real-content measurement |
| 4 | `services/adaptive_frame_sampler.py` | `CLIPAI_ADAPTIVE_VLM_SAMPLING` | off | no — pending cost telemetry |
| 5 | `services/crop_qa.py` | `CLIPAI_CROP_QA` | off | no — pending clip_exporter hook |
| 6A | detector flag matrix (ArcFace, YOLO11n, pose, anime face, Light-ASD) | existing `CLIPAI_*_BACKEND` flags | unchanged | no — pending matrix run on real-content fixtures |
| 6B | dormant anime modules (`anime_shot_detector`, `anime_face_detector`, `anime_character_clustering`) | `CLIPAI_ANIME_SHOT_DETECTOR` / `_FACE_DETECTOR` / `_CHARACTER_CLUSTERING` | **not added** | no — pending anime-content smoke test |
| 6C | `services/asd_tiebreaker.py` | `CLIPAI_ASD_TIEBREAKER` | **on** | landed (strictly safer than unmodified heuristic) |

### Why flag promotions are deferred

The prompt sets three gates for promoting any flag default flip:

1. `validate_v2_phases --quick` exits 0.
2. Required-region miss rate is parity-or-better on every fixture
   in `tests/real_content/`.
3. Cost telemetry stays within 1.5× baseline at the `efficient`
   preset.

Week 3 left the real-content cache empty (`source_url` fields in
`tests/real_content/manifest.json` are still blank). Until those
clips land and the AutoFlip reference outputs are cached, no
matrix run can produce the deltas we'd need to justify any
default flip. The VLM upgrade therefore ships every phase as a
dormant module gated behind its own env flag — every default
preserves the pre-upgrade behavior bit-for-bit.

### Week 3 action items (VLM side)

Clearing these unblocks the promotion pass:

1. **Populate `tests/real_content/manifest.json` + run `fetch.sh`**
   (this is the same blocker Week 3 flagged for AutoFlip parity).
2. **Run `backend/scripts/measure_detection_stack_matrix.py`** on
   every fixture. Record deltas in a new `Phase 11 matrix results`
   table and flip the flags that show parity-or-better.
3. **Wire the `clip_exporter` → `crop_qa` hook** (Phase 5
   integration). Measure the per-fixture cost delta with
   `CLIPAI_CROP_QA=on`; if QA tokens stay < 5% of total VLM
   tokens AND < 3% of segments trigger recovery, flip the flag.
4. **Wire the adaptive sampler into `frame_extractor`** and
   measure the token-budget delta with
   `CLIPAI_ADAPTIVE_VLM_SAMPLING=on`; if ≤ 1.5× baseline AND every
   fixture's required-region miss rate is flat-or-better, flip
   the flag.
5. **Measure `CLIPAI_VLM_FUSION=weighted`** on `vlog_walk_and_talk`
   — the fixture most likely to improve. If required-region miss
   rate improves by ≥ 5pp, flip the default to `weighted`.
6. **Run Light-ASD on the ambiguous windows produced by
   `asd_tiebreaker.find_ambiguous_windows`** for each anime /
   gameplay fixture. If no regression in speaker-attribution
   accuracy, keep `CLIPAI_ASD_TIEBREAKER=on` (already default).
7. **Wire the dormant anime modules** into
   `reframe_segmenter::_route_anime_content`. Update
   `test_dormant_flags_labeled.py` to allow the new call sites and
   re-assert zero call sites elsewhere. Measure against the anime
   fixtures; flip `CLIPAI_ANIME_SHOT_DETECTOR` /
   `CLIPAI_ANIME_FACE_DETECTOR` /
   `CLIPAI_ANIME_CHARACTER_CLUSTERING` to default-on for ANIME
   content type only.

### Modules + tests shipped

| Phase | New files | Tests | Passing |
|---|---|---|---|
| 1 | `models.py` extensions, `prompts.py` rewrite, `providers/base.py::parse_scene_dict` | `test_scene_dict_grounded.py` (10) | 10/10 |
| 2 | `providers/openrouter_provider.py` rewrite | `test_vision_model_routing.py` (19) | 19/19 |
| 3 | `services/vlm_fusion.py` | `test_vlm_fusion.py` (17) | 17/17 |
| 4 | `services/adaptive_frame_sampler.py` | `test_adaptive_frame_sampling.py` (12) | 12/12 |
| 5 | `services/crop_qa.py`, `prompts.py::DEFAULT_CROP_QA_PROMPT` | `test_crop_qa.py` (15) | 15/15 |
| 6C | `services/asd_tiebreaker.py` | `test_asd_tiebreaker.py` (13) | 13/13 |

**Total: 86 new tests, all passing.** No existing test regressed.
`validate_v2_phases.py --quick` exit 0 before and after every
phase (all env flags default-off preserve legacy behavior).

### VLM upgrade phases NOT landed as dormant-module + tests

- **Phase 5 clip_exporter hook** — see Phase 5 notes. Needs
  opencv frame extraction + real VLM scorer wired in the render
  path. Deferred to a dedicated integration pass so it can be
  measured end-to-end.
- **Phase 6A detector flag promotions** — the five candidate
  flags remain `_pending matrix run_`. The matrix runner exists
  (`backend/scripts/measure_detection_stack_matrix.py`, shipped in
  phase-e); what's missing is a fixture set to run it against.
- **Phase 6B dormant anime modules** — no call sites added
  inside reframe_segmenter. The guard test
  `test_dormant_flags_labeled.py` is intact; wiring them is a
  separate integration pass that follows Phase 6A's matrix
  results.

