# Real-content human-reframe quality bench — 2026-04-23

First real-content human-reframe quality bench result.

Pipeline: `human_reframe.run_human_reframe` (NOT yet wired into production)
Source: 9 cached real clips, sha256-pinned, durations matching slug suffixes

## Synthetic fixtures

```
talking_head:    PASS  (chin_rate=0, head_rate=0, jitter=0, repair=0)
mixed_genre:     fail  (intentional adversarial fixture)
gappy_faces:     PASS
```

## Real fixtures (passing)

```
panel_breakfast_club_10s:
  chin=0.000  head=0.000  center_p95=0.082  jitter_p95=0.025
  repair=0.000  saccade_gap=12.1s
  PASS on every threshold.
```

## Real fixtures (failed on one axis)

```
panel_joebudden_10s:
  chin=0.000  head=0.000  center_p95=0.112  jitter_p95=0.109  ← over spec
  repair=0.000  saccade_gap=11.7s
  Single-axis tuning issue on multi-speaker content.
```

## Real fixtures (broken — bench bug, see issue 01)

```
panel_verzuz_tank_tyrese_10s
sports_nba_fastbreak_20s
sports_f1_onboard_20s
music_chris_brown_performance_15s
music_beyonce_solo_narrative_15s
```

Pattern: `chin=0, head=0, center_p95=0, jitter_p95=0, repair=0.95-1.0`.
Indicates the per-frame scoring loop received empty `dense_faces`.

Stderr included:

- `[DenseFaces] FFmpeg extraction failed (rc=183)`
- MediaPipe GL EGL init failures (`eglMakeCurrent() returned error 0x3008`)

## Real fixtures (partial — meaningful)

```
sports_boxing_8s:
  head_unreachable=0.22  repair=0.78

music_mj_thriller_formation_15s:
  head_unreachable=0.16  repair=0.84
```

These are real signal: the solver reached the lookahead's headroom limit
on motion-heavy content, and the repair pass took over for the remainder.

## What this means

The SOTA path is sound enough on talking-head / panel content to justify
**Phase C wiring** (the next PR) — `panel_breakfast_club_10s` passes every
axis, and `panel_joebudden_10s` only misses one. The Phase C scope can
safely include an opt-in flag for the multi-speaker panel content type
behind today's pass/fail evidence.

The 5 broken clips are NOT a model regression. The all-zeros fingerprint
plus the `[DenseFaces] FFmpeg extraction failed (rc=183)` stderr points at
an upstream face-detection / extraction failure inside the bench harness.
This blocks bench *coverage* but not Phase C *wiring*.

The 2 partial clips (boxing, MJ Thriller) are correct repair-rate
behavior on motion-heavy content where the camera can't lawfully reach
the desired headroom — exactly what the repair tier was designed for.

## Known issues

Drafts staged in `docs/issues_to_open/` for the user to paste into the
GitHub web UI:

- `01_face_detection_5_clips.md` — bench: face detection produces empty
  `dense_faces` on 5 of 9 real clips. Blocks coverage; does not block
  Phase C.
- `02_jitter_multi_speaker_panel.md` — bench: `panel_joebudden_10s` fails
  jitter only — multi-speaker tuning ticket. Blocks default-on
  graduation of `multi_speaker_panel`; does not block Phase C.
- `03_ground_truth_vertical_urls_are_channels.md` — `tests/real_content`:
  `ground_truth_vertical_url` is a channel handle for 5 slugs, blocking
  the human-parity bench (mae_cx, framing_macro_f1, cut_timing_deviation,
  lead_room_correlation). Does not block the quality bench.
- `04_anime_source_url_is_string_literal.md` — 3 anime slugs have
  `source_url="mp4"` instead of a `file://` path. Manifest cleanup,
  not user-blocking.
