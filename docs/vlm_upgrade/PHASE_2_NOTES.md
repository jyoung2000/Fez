# Phase 2 — PRESETS refresh + content-type vision routing

## What changed

### `backend/services/providers/openrouter_provider.py`

**PRESETS rewrite.**

| Tier | Old default | New default |
|---|---|---|
| free | `openrouter/free` (+ qwen2.5-vl-72b:free, gemma-3-27b, llama-3.2-11b, mistral-small, gemini-2.5-flash) | `openrouter/free` (+ **qwen3-vl-30b-a3b:free**, **qwen3-vl-8b-thinking:free**, qwen2.5-vl-72b:free, gemma, llama, gemini-2.5-flash) |
| efficient | `google/gemini-2.5-flash` | **`qwen/qwen3-vl-235b-a22b-instruct`** (+ qwen3-vl-30b, gemini-3.1-flash-lite, gemini-2.5-flash, gemini-2.5-flash-lite) |
| balanced | `google/gemini-2.5-flash` | **`google/gemini-3-flash-preview`** (+ qwen3-vl-235b, gemini-2.5-flash, gemini-2.5-flash-lite) |
| premium | `google/gemini-2.5-pro` | **`google/gemini-3-pro-preview`** (+ gemini-3.1-pro, gemini-3-flash, gemini-2.5-pro, gemini-2.5-flash) |

Every fallback chain ends with a stable Gemini 2.5 model so no path
can dead-end if a preview model is retired mid-job.

**New module-level helpers:**

- `_CONTENT_TYPE_VISION_OVERRIDES` — anime / animation / cartoon /
  gameplay / gameplay_fps / gameplay_moba / gameplay_tps /
  gameplay_racing / stream → `qwen/qwen3-vl-235b-a22b-instruct`.
  Every other content type falls through to preset default.
- `select_vision_model_for_content(content_type, preset_default)`
  — pure function, accepts a ClipContentType enum / its `.value`
  string / a raw string / None. Case-insensitive.

**New provider method:**

- `OpenRouterProvider.apply_vision_model_override(content_type)` —
  idempotent, mutates `self._vision_model` in place. Captures the
  original preset default in `self._base_vision_model` during
  `__init__` and prepends it to the fallback chain the first time
  an override fires, so a routed model that 401s always falls
  through to the user's configured preset default. Logs every
  decision with `Using vision model for {content_type}: {model_id}`.

### `backend/services/ai_orchestrator.py::analyze_frames`

- New optional `content_type` kwarg. When provided, calls
  `provider.apply_vision_model_override(content_type)` right before
  the VLM request. Wrapped in a try/except so non-OpenRouter
  providers (Anthropic, Gemini, Groq, Ollama) which do not implement
  this method are silently unaffected.

### `backend/services/pipeline.py:2233`

- Threads `_normalized_override` (the user's explicit
  `content_type_override` UI setting, already normalized via
  `content_type_strings.normalize_ui_content_type`) into
  `orchestrator.analyze_frames(..., content_type=...)`. Jobs
  without an explicit override route with `content_type=None`
  which is a pure no-op against the preset.

## What did NOT change

- **No cost telemetry plumbing yet.** The prompt spec calls for
  `vision_model_used` and `vision_model_cost_estimate_usd` in the
  job diagnostics block. That needs to land in a dedicated pass
  alongside Phase 4's adaptive-sampling cost measurements, where
  we can actually verify the 1.5× budget target on real fixtures.
  Phase 2 logs the decision via `logger.info` — grepping the logs
  is enough to confirm routing for now.
- **parse_scene_dict is not wired into analyze_frames yet.** The
  existing openrouter provider still uses its inline legacy parser.
  This is deliberate: swapping the parser touches 360 lines of
  fused face-detection / registry logic and is best done as a
  tightly scoped standalone change once Phase 3 fusion is in place.
- **Other providers (Anthropic, Gemini, Groq, Ollama)** are
  untouched. They don't ship the PRESETS system, and the prompt
  explicitly calls out "Do not touch OllamaProvider vision logic".

## Tests

`backend/tests/test_vision_model_routing.py` — 19 tests, all passing:

1-5: `PRESETS` sanity + new default IDs per tier.
6-10: `select_vision_model_for_content` returns Qwen3-VL for
anime / animation / cartoon / gameplay / gameplay_fps.
11-15: same function returns preset default for talking_head,
music_video, cinematic_dialogue, None, unknown strings.
16-17: enum-like object + case-insensitive.
18-19: `apply_vision_model_override` swaps / restores correctly and
logs the decision.

## Back-compat

- Jobs with no content_type_override thread `content_type=None`
  through `analyze_frames`. `apply_vision_model_override(None)`
  returns the base vision model and does not mutate state.
- Non-OpenRouter providers do not implement
  `apply_vision_model_override`; the orchestrator wraps the call
  in `try/except` so it silently no-ops.

## Cost notes

The override only fires on ANIME and GAMEPLAY content. Qwen3-VL-235B
via OpenRouter is price-competitive with Gemini 2.5 Flash
(~$0.50/M input, ~$3/M output as of April 2026) so the routed
content does not blow the 1.5× budget cap. Phase 4 will measure
against real fixtures.
