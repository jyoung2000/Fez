# ClipAI Blueprint v2 — Phase 0 Migration

Phase 0 unlocks four features that were already built or half-built in
the codebase, behind env flags / content-type allowlists so existing
flows stay bit-identical until a feature is deliberately enabled.

## What Phase 0 changes

| Capability | Before | After |
|---|---|---|
| Multi-layout (SPLIT / TRIPLE / PIP / SCREENSHARE / GAMEPLAY) | Dead code — `ALLOW_MULTI_LAYOUT=false` gated everything off | Reachable for podcast / interview / panel / stream / gameplay / tutorial content by default; other types stay SINGLE |
| Post-render VLM quality gate | `extract_frames_for_critic()` defined, never called | One VLM call per export when `CLIPAI_POST_RENDER_CRITIC=1`; writes `post_render_report` onto the exported-clip record |
| Ken Burns / slow-push layout | None | `RenderOpKind.KEN_BURNS` + `_filter_ken_burns` + `layout_kenburns.py` detector. Fires on landscape / B-roll / establishing shots with no tracked subject |
| `deadband_frac` per genre | Single module-level `STATIONARY_THRESHOLD = 0.08` | `ReframeConfig.deadband_frac` with per-content entries (sports=0.08, racing=0.06, panel=0.12, documentary=0.15 …) threaded into `solve_camera_path` |

Zero new models. No external API changes. All existing tests continue
to pass bit-identical because the new code paths are gated.

## New env vars

| Var | Default | Effect |
|---|---|---|
| `ALLOW_MULTI_LAYOUT` | unset | When set to `1`/`true`, forces multi-layout ON for all content types (overrides the allowlist). When set to `0`/`false`, forces SINGLE everywhere. Unset = per-content allowlist. |
| `CLIPAI_POST_RENDER_CRITIC` | `0` | `1`/`true` enables the post-render VLM quality gate. Adds ~2-8s per clip (single multi-image VLM call). |
| `CLIPAI_DEADBAND_FRAC` | `0.10` | Global static-hold deadband default. Per-content entries in `_CONTENT_OVERRIDES` override. |
| `CLIPAI_KEN_BURNS_TRAVEL` | `0.08` | Max total Ken Burns travel as fraction of source. Set to `0.0` to disable travel (push stays centered). |
| `CLIPAI_KEN_BURNS_ZOOM` | `1.08` | Max zoom factor. Set to `1.0` to disable Ken Burns entirely. |
| `CLIPAI_KEN_BURNS_MIN_DUR` | `2.0` | Minimum scene duration (s) to qualify for Ken Burns. Set high (e.g. `999999`) to disable. |

## Multi-layout default content-type allowlist

By default these content types can emit multi-layout modes:

- `podcast`, `interview`, `multi_speaker_panel`
- `stream`, `gameplay` family (`fps`/`moba`/`tps`/`racing`)
- `tutorial`, `screen_share`

All other types (cinematic, narrative, talking_head, anime, etc.)
stay SINGLE-only. Split screens look wrong on drama / movie content;
talking-head / podcast one-person shots don't need them.

Override via `ReframeConfig.multi_layout_content_types` or the
`ALLOW_MULTI_LAYOUT` env var.

## Per-genre deadband defaults

| Content type | `deadband_frac` |
|---|---:|
| talking_head, sports, vlog | 0.08 |
| sports_racing | 0.06 |
| podcast, interview, gameplay family | 0.10 |
| multi_speaker_panel, cinematic_dialogue, music_video | 0.12 |
| documentary, landscape | 0.15 |

Larger values = stickier holds. Smaller = faster response.

## Post-render critic: what gets stored

When `CLIPAI_POST_RENDER_CRITIC=1`, each `exported_clips` entry gains
a `post_render_report` dict:

```json
{
  "ok": false,
  "issues": [
    {"t": 3.2, "issue": "speaker's face is half off the right edge",
     "severity": "high"}
  ],
  "sampled": 6,
  "latency_sec": 4.1
}
```

`ok=false` triggers a `warn:` entry on `job.pipeline_warnings` that
surfaces as a banner on the Analysis page. `ok` becomes `false` when
there are >= 2 high-severity issues.

Providers supported: Ollama (`/api/generate` with multi-image array),
OpenRouter (OpenAI-style multi-image messages). Anthropic / Gemini /
Groq providers skip post-render critique for now (fall through to the
next provider in the chain). Every failure mode is non-fatal — the
exporter never blocks on the critic.

## Rollback

| Feature | Rollback |
|---|---|
| Multi-layout | `ALLOW_MULTI_LAYOUT=false` env var forces SINGLE everywhere |
| Post-render critic | `CLIPAI_POST_RENDER_CRITIC=0` (default) |
| Ken Burns | `CLIPAI_KEN_BURNS_MIN_DUR=999999` or `CLIPAI_KEN_BURNS_ZOOM=1.0` |
| deadband_frac | Remove the per-content `deadband_frac` entries from `_CONTENT_OVERRIDES`; reverts to legacy `STATIONARY_THRESHOLD = 0.08` everywhere |

## Rollout checklist

1. Merge Phase 0 with post-render critic OFF by default (current
   state). Ken Burns and multi-layout default on for their applicable
   content types.
2. Week 1: Monitor render / bench runs for regressions. Ken Burns and
   multi-layout should never fire on unintended content types because
   they're gated on the ClipContentType the classifier picks.
3. Week 2: Enable `CLIPAI_POST_RENDER_CRITIC=1` for one class of users
   (e.g. admins) for observation. Expect ~2-8s of added latency per
   clip on the Ollama path.
4. Week 3: Promote post-render critic to default ON if the flagged-
   issue rate is < 25% and no user complains about export latency.

## Next phase

`clipai_blueprint_v2_phase1_importance_matrix.md` — consolidate the
scattered genre weight tables into a single importance matrix on
`reframe_config.py` so Phase 2 (hybrid layout) can read one table.
