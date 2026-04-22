# Tier 3 Outpainting Providers — ClipAI Blueprint v2 Phase 5

Tier 3 is generative video outpainting for the ~5 % of clips where
the Tier 1 blurred-pillar looks bad enough that the added cost +
latency of a diffusion pass is worth it. In 2026 the productized
options are:

| Provider | Pros | Cons | Price |
|---|---|---|---|
| **Luma Ray 2** | Hosted, productized, reliable; good at outpaint | API dependency | ~$0.04-0.10 per video-second |
| **ByteDance Seedance 2.0** | Multimodal, native video extension | API availability varies by region | Pricing not yet public |
| **MOTIA / GlobalPaint** | Open-source, self-host | Needs 24 GB+ VRAM; temporal-coherence gaps | GPU cost only |
| **Hierarchical Masked 3D Diffusion** | Temporally coherent | Academic, not productized | — |

## Decision

- **Primary:** Luma Ray 2 (via `outpaint_fill.py` backend
  `CLIPAI_OUTPAINT_PROVIDER=luma`, API key `LUMA_API_KEY`).
- **Self-hosted fallback:** generic HTTP service (`CLIPAI_OUTPAINT_PROVIDER=local`,
  `OUTPAINT_LOCAL_URL`) for shops that already run MOTIA or equivalent.
- **Safety-net:** Tier 1 `BLUR_FILL` on any failure. The exporter
  never fails a clip because outpainting failed.

### Cost model

Per-clip estimate at Luma Ray 2 pricing (~$0.07 / video-second,
mid-range):

| Clip length | Outpaint cost |
|---|---|
| 15 s | ~$1.05 |
| 30 s | ~$2.10 |
| 60 s | ~$4.20 |

Phase 5 caps outpainting at `CLIPAI_OUTPAINT_MAX_PER_CLIP` (default 3)
per clip so a multi-scene clip can't blow the budget. A clip-level
`outpaint_cost_usd` field (populated from the provider's returned
duration + rate) lands on the exported-clip record for UI display.

### Opt-in model

- Global gate: `CLIPAI_OUTPAINT_ENABLED=1` turns the path on at all.
- Per-clip trigger: a clip must either be flagged "hero" (virality
  score >= `CLIPAI_OUTPAINT_HERO_THRESHOLD`, default 8.5 on the 0-10
  scale, or 85 on the legacy 0-100 scale) **or** the user must opt
  in on a per-job basis via the settings page.
- Safety gates skip outpainting for content types where it is known
  to fail (anime, gameplay, screen-share) or where a face / HUD /
  fast motion sits at a crop edge.

### API credentials

| Provider | Env var(s) |
|---|---|
| Luma Ray 2 | `LUMA_API_KEY` (+ optional `LUMA_API_BASE`) |
| Seedance 2.0 | `SEEDANCE_API_KEY` + `SEEDANCE_API_BASE` |
| Local HTTP | `OUTPAINT_LOCAL_URL` (+ optional `OUTPAINT_LOCAL_TOKEN`) |

### Rollback

Flip `CLIPAI_OUTPAINT_ENABLED=0` and every line of Phase 5 becomes a
no-op. The user-facing toggle in Settings defaults to OFF for every
new account.

### Retirement path

None — outpainting is additive. Tier 1 (`BLUR_FILL`) remains the
baseline fill strategy and the fallback for the foreseeable future.
