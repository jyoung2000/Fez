# Fez MCP Debug Server — Phase 1 Implementation Plan

**Status:** Draft. Do not start coding until the user confirms.

This plan reflects the realities documented in `MCP_IMPLEMENTATION_NOTES.md`.
Where this plan deviates from the original prompt, the deviation is called
out in the **Deviations from prompt** section at the end.

## 1. Module layout

New package at `backend/mcp_debug/` (sibling to existing `backend/mcp_tools.py`,
not a sibling top-level directory). Reasons:

- Fez is a Python monorepo with everything under `backend/`. A top-level
  `fez-mcp/` would force a duplicate dependency install.
- We need read-side imports of `backend.database`, `backend.config`,
  `backend.models`, `backend.services.render_plan`, etc. Living inside
  `backend/` keeps those imports clean and avoids `sys.path` hacks.
- The existing FastMCP mount in `backend/main.py` becomes a natural place
  to also mount the debug instance.

```
backend/mcp_debug/
  __init__.py
  server.py              # FastMCP construction + standalone uvicorn entry
  auth.py                # bearer-token middleware
  schemas.py             # Pydantic response models for tool outputs
  storage.py             # thin read-only adapter over backend.database
  tools/
    __init__.py
    renders.py           # get_recent_renders, get_render_details, compare_renders
    stages.py            # get_pipeline_stage
    render_plan.py       # get_render_plan, validate_render_plan
    logs.py              # get_logs (tail /data/logs/app.log)
    health.py            # get_circuit_breaker_state, get_pipeline_health
  resources.py           # fez://renders/{id}/transcript and friends
  tests/
    __init__.py
    test_storage.py
    test_tools_renders.py
    test_tools_stages.py
    test_tools_render_plan.py
    test_auth.py
```

`__main__.py` is intentionally not included — entry point is `server.py:main()`
runnable as `python -m backend.mcp_debug.server`. Matches the existing
`backend.mcp_server` pattern.

## 2. Tool surface — Phase 1 (read-only)

All tools return **structured JSON via Pydantic models**, not narrative
prose. All return `{"error": "...", "available_stages": [...]}` on missing
data rather than raising.

### `get_recent_renders(limit: int = 10, status: str | None = None, owner_username: str | None = None) -> RecentRendersResponse`

Lists the most recent renders, sorted by `updated_at` desc.

Returns: `{"renders": [RenderSummary], "total_seen": int}`

`RenderSummary` shape:
```python
class RenderSummary(BaseModel):
    job_id: str
    filename: str
    status: str                  # JobStatus value
    progress: int                # 0-100
    duration_sec: float
    resolution: str
    fps: float
    created_at: str
    updated_at: str
    owner_username: str
    content_type: str            # content_type_override or clip_content_type
    has_render_plan: bool
    has_transcript: bool
    has_scenes: bool
    clip_count: int
    error: str | None
    pipeline_warnings_count: int
    timings_total_sec: float | None
```

Implementation: calls `backend.database.list_jobs()`, sorts, filters,
projects.

> **Branch filter dropped.** The original prompt said
> `get_recent_renders(limit, branch?, status?)`. Fez has no notion of a
> "branch" attached to renders — that's a git concept, not a Fez data
> concept. Replaced with `owner_username` filter (which Fez does have).
> If the user wants per-git-branch render attribution, that's a Phase 3
> data-model change, not an MCP tool.

### `get_render_details(job_id: str) -> RenderDetailsResponse`

Returns the full pipeline trace for one render, including which stages
have output and which don't.

Returns: a flattened summary of every Phase-1-relevant JobResult field
(see schemas.py outline below). Excludes huge fields by default
(`face_registry_data`, full transcript text) and exposes them via
`get_pipeline_stage` instead.

```python
class RenderDetailsResponse(BaseModel):
    summary: RenderSummary
    timings: dict[str, float]
    pipeline_warnings: list[str]
    provider_used: dict[str, str]
    estimated_cost_usd: float | None
    stages_completed: list[str]      # ["ingestion","transcription",...]
    stages_missing: list[str]
    speaker_names: dict[str, str]
    layout_default: str
    tracking_mode: str
    content_type: str
    error: str | None
```

### `get_pipeline_stage(job_id: str, stage: str) -> StageResponse`

Single-stage drill-in. `stage` is one of:
`"ingestion" | "transcription" | "scene_detection" | "speaker_detection" | "vlm_analysis" | "render_plan" | "ffmpeg"`.

Returns shape varies by stage but is always a typed Pydantic model:

| stage | payload |
|---|---|
| `ingestion` | `{file_path, duration, resolution, fps, file_size_mb, source_sha256}` |
| `transcription` | `{segment_count, languages, speakers: list[str], coverage_sec, segments: list[TranscriptSegment]}` |
| `scene_detection` | `{scene_count, scene_cut_timestamps, scenes: list[SceneDescription]}` |
| `speaker_detection` | `{face_registry_summary, speaker_names, layout_timeline, dense_tracking_summary}` (omits raw face embeddings) |
| `vlm_analysis` | `{provider_used, scene_count, scenes: [{timestamp, importance_score, description, subject_x, subject_box, subject_confidence, active_slot, face_count}]}` |
| `render_plan` | delegates to `get_render_plan` |
| `ffmpeg` | `{exported_clips: list[ExportedClipMeta]}` from `JobResult.exported_clips` |

If the job is missing data for the requested stage, returns
`{"error": "stage X not yet computed", "available_stages": [...]}` rather
than raising.

### `get_render_plan(job_id: str) -> RenderPlanResponse`

Reconstructs the typed `RenderPlan` dataclass from `job.render_plan` dict
and returns:

```python
class RenderPlanResponse(BaseModel):
    source: dict             # {width, height, fps, total_duration_sec}
    target: dict             # {width, height}
    op_count: int
    ops: list[RenderOpView]  # flattened, JSON-clean
    validation_errors: list[str]   # from RenderPlan.validate()
    debug: dict | None             # render_plan["debug"] if present
```

`RenderOpView` flattens `RenderOp` to: `kind`, `start_sec`, `end_sec`,
`primary_rect`, `secondary_rect?`, `tertiary_rect?`, `quaternary_rect?`,
`motion_path` (compact), `ease_in_ms`, `strategy_label`, `content_type`,
`gaming_layout_mode`, `speaker_slot`, `speaker_label`.

### `get_logs(job_id: str | None = None, level: str = "INFO", limit: int = 200) -> LogsResponse`

Tails `/data/logs/app.log` (and rotated `.1`, `.2`, `.3`).

- `job_id`: substring match against the message body (Fez logs include
  `job_id=...` in messages — see notes).
- `level`: minimum log level to include.
- `limit`: cap on number of returned lines (default 200, max 2000).

Returns: `{"lines": list[LogLine], "log_files_searched": list[str], "truncated": bool}` where each `LogLine` is `{ts, level, logger, message}` parsed from
the format `"%(asctime)s [%(levelname)s] %(name)s: %(message)s"`.

> **Stage filter dropped from logs tool.** The original prompt said
> `get_logs(render_id, stage?, level?, limit?)`. Fez logs are not
> structured per-stage; the `stage` is implicit in the logger name and the
> message wording. A naive substring match would be misleading. We expose
> `level` and `job_id` filtering only; users can refine with the message
> body if they want.

### `get_circuit_breaker_state() -> CircuitBreakerStateResponse`

Returns:
```python
class CircuitBreakerStateResponse(BaseModel):
    available: bool                # False if MCP server is out-of-process
    providers: dict[str, ProviderState]
    note: str | None

class ProviderState(BaseModel):
    name: str
    is_degraded: bool
    degraded_until_iso: str | None
    failure_count_in_window: int
```

If the debug server is mounted **in-process** on the FastAPI app, it
imports the orchestrator from `backend.services.ai_orchestrator` (need a
small accessor) and returns live state. If it's running **out-of-process**
(Docker sidecar mode), it returns `{"available": false, "note": "circuit breaker is in-process state; run the MCP server in-process for live data"}`.

### `compare_renders(job_id_a: str, job_id_b: str) -> CompareResponse`

Returns a structured diff focused on the things that matter for debugging:

```python
class CompareResponse(BaseModel):
    a: RenderSummary
    b: RenderSummary
    timings_diff: dict[str, float]                # b - a per stage
    provider_used_diff: dict[str, tuple[str, str]]
    settings_snapshot_diff: dict[str, tuple]      # if we ever start snapshotting
    render_plan_diff: RenderPlanDiff
    pipeline_warnings_diff: dict[str, list[str]]  # {"a_only", "b_only"}
```

`RenderPlanDiff` shape: `{op_count_a, op_count_b, kind_histogram_a, kind_histogram_b, validation_errors_a, validation_errors_b, total_duration_a, total_duration_b}`.

> **Settings snapshot diff is best-effort only.** Fez does not currently
> snapshot the active `Settings` per-render. The diff for that field will
> be empty until/unless we add snapshotting (Phase 3).

## 3. Resources

```
fez://renders/{id}/transcript            -> JSON: list[TranscriptSegment]
fez://renders/{id}/render_plan.json      -> JSON: RenderPlan as dict
fez://renders/{id}/scene_timeline.json   -> JSON: scenes with full descriptions
fez://config/current                     -> JSON: redacted snapshot of Settings
```

`fez://config/current` masks the API keys (mirroring the existing
`backend.auth.mask_key` pattern in `mcp_tools.py:get_settings`).

## 4. Auth

`backend.mcp_debug.auth` — Starlette middleware that:

- Reads `MCP_AUTH_TOKEN` env var at startup. **If unset, the server
  refuses to start with a clear log message** (no implicit "open" mode).
- Checks `Authorization: Bearer <token>` on every request.
- Returns 401 with `{"error": "unauthorized"}` body on missing/wrong token.
- Constant-time comparison via `hmac.compare_digest`.
- Local-stdio mode (when launched via `mcp dev` style without HTTP) skips
  auth — auth only applies to the SSE/HTTP listener.

## 5. Transport

Two modes, both supported by FastMCP out of the box:

### Mode A — In-process (recommended, default)

Add a second FastMCP instance in `backend/main.py`:

```python
from backend.mcp_debug.server import build_debug_mcp
debug_mcp = build_debug_mcp()
app.mount("/mcp/debug", debug_mcp.streamable_http_app())
```

Auth middleware is wrapped around the mount. Transport is **streamable
HTTP** (matches existing `/mcp` mount and is what the official Python SDK
defaults to). Live circuit-breaker state is available because the
orchestrator lives in the same Python process.

### Mode B — Standalone SSE (for remote / Claude Code web)

`python -m backend.mcp_debug.server` runs uvicorn with the FastMCP
SSE app on `0.0.0.0:${MCP_PORT:-8765}`. Bearer-token auth enforced.

The user prompt explicitly asked for SSE on `0.0.0.0:8765`. We support it
via Mode B. Mode A is still the default for local development.

> **Note on transport choice.** The existing in-process mount at `/mcp`
> uses `streamable_http_app()`, not SSE. The SDK supports both. SSE is
> being deprecated upstream in favor of streamable HTTP. We'll expose
> SSE for Mode B because the prompt explicitly requested it, and document
> the streamable-HTTP alternative in the README. The container's port
> 8765 is transport-agnostic.

## 6. Storage adapter

`backend.mcp_debug.storage` is a thin async wrapper:

```python
async def list_recent(limit: int, status: str | None, owner_username: str | None) -> list[JobResult]: ...
async def load(job_id: str) -> JobResult | None: ...        # delegates to database.load_job
def parse_render_plan(raw: dict) -> RenderPlan | None: ...  # reconstruct dataclass
def stages_present(job: JobResult) -> tuple[list[str], list[str]]: ...
def tail_logs(level: str, job_id: str | None, limit: int) -> list[LogLine]: ...
```

This is the single seam tests mock — every tool calls into `storage`, so
unit tests can swap in a fake storage and verify shapes without needing
real `/data/uploads` content.

## 7. Tests (Phase 1 minimum bar)

- `test_storage.py` — round-trip JobResult through `database.save_job` /
  `load_job` using a temp `/data/uploads` dir; verify `list_recent` sort
  order and filters.
- `test_tools_renders.py` — exercise `get_recent_renders`,
  `get_render_details`, `compare_renders` against fixture jobs with
  varying completeness (full pipeline, partial, failed).
- `test_tools_stages.py` — verify each `get_pipeline_stage(stage=...)`
  returns the right schema, and missing stages return the structured
  `{"error", "available_stages"}` shape.
- `test_tools_render_plan.py` — feed a hand-built RenderPlan dict, verify
  `get_render_plan` reconstructs and validates it; feed a plan with a gap
  to verify validation errors surface.
- `test_auth.py` — bearer middleware: missing token → 401, wrong token →
  401, right token → 200. `MCP_AUTH_TOKEN` unset → server refuses to
  start.

Framework: pytest, matching the existing style under `tests/`. Will use
`pytest-asyncio` (already pulled in transitively by FastAPI). Tests will
**not** require running ffmpeg, torch, or a real video — fixtures use
hand-built `JobResult` instances.

## 8. Containerization

`Dockerfile.mcp` at repo root:

```dockerfile
FROM python:3.11-slim
WORKDIR /app
RUN useradd --create-home --uid 10001 mcp

COPY backend/requirements.mcp.txt ./requirements.mcp.txt
RUN pip install --no-cache-dir -r requirements.mcp.txt

COPY backend/ ./backend/

USER mcp
ENV MCP_PORT=8765
EXPOSE 8765

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"MCP_PORT\",\"8765\")}/healthz').read()" || exit 1

CMD ["python", "-m", "backend.mcp_debug.server"]
```

`backend/requirements.mcp.txt` (new file, slim subset — no torch, no
opencv, no ffmpeg deps, no whisper):

```
fastapi==0.111.0
uvicorn[standard]==0.30.0
pydantic>=2.7.2,<3.0.0
pydantic-settings>=2.5.2
aiofiles==23.2.1
mcp>=1.8.0
```

`docker-compose.mcp.yml`:

```yaml
services:
  fez-mcp:
    build:
      context: .
      dockerfile: Dockerfile.mcp
    container_name: fez-mcp
    restart: unless-stopped
    ports:
      - "8765:8765"
    volumes:
      - ${FEZ_RENDERS_DIR:-./data/uploads}:/data/uploads:ro
      - ${FEZ_LOGS_DIR:-./data/logs}:/data/logs:ro
    environment:
      MCP_AUTH_TOKEN: ${MCP_AUTH_TOKEN:?required}
      MCP_PORT: "8765"
      LOG_LEVEL: INFO
    user: "10001:10001"
```

The `${MCP_AUTH_TOKEN:?required}` syntax makes compose fail loudly if the
token isn't provided.

`/healthz` endpoint: a small Starlette route added next to the FastMCP
mount that returns `{"status": "ok"}` and does NOT require auth.

## 9. Documentation

`backend/mcp_debug/README.md` covers:

- Tool reference (one-line each, plus example response shape)
- Resource URIs
- How to run in-process (already mounted at `/mcp/debug` on the main app)
- How to run standalone (`python -m backend.mcp_debug.server`)
- Docker compose snippet for Unraid
- Claude Code MCP config — both stdio and SSE/HTTP examples
- Debugging workflow recipes:
  1. "Investigate why render X has the wrong priority slot" —
     `get_render_details` → `get_pipeline_stage(stage="speaker_detection")`
     → `get_render_plan` and look at `speaker_slot` per op.
  2. "Compare two renders of the same source on different config" —
     `get_recent_renders(limit=20)`, then `compare_renders(a, b)`.
  3. "Find recent speaker detection failures" —
     `get_recent_renders(status="failed")` →
     `get_pipeline_stage(stage="speaker_detection")` per render →
     filter to ones with `pipeline_warnings` matching speaker keywords.

## 10. Out of scope for Phase 1

- `trigger_reframe` action tool
- `replay_render` (no first-class replay/checkpoint API exists in Fez yet)
- Render-attached settings snapshot (would require pipeline change)
- Per-git-branch render attribution
- Per-stage log filtering beyond substring-by-job-id
- Modifying existing `/mcp` to add bearer auth (out of scope unless asked)

## 11. Commit plan

1. ✅ `MCP debug server: exploration notes (Step 1)` — done
2. (this commit) `MCP debug server: implementation plan (Step 2)`
3. After user confirms:
   - `MCP debug server: storage adapter + schemas + auth`
   - `MCP debug server: render & stage tools`
   - `MCP debug server: render plan & log tools`
   - `MCP debug server: circuit breaker, compare, resources`
   - `MCP debug server: in-process FastAPI mount + standalone server`
   - `MCP debug server: Dockerfile.mcp + compose snippet`
   - `MCP debug server: README + Claude Code config examples`

## 12. Deviations from the original prompt

1. **Module location**: package lives at `backend/mcp_debug/`, not a
   sibling `fez-mcp/` directory. Reason: see §1.
2. **`get_recent_renders` `branch?` param dropped.** Fez has no
   render-to-git-branch link. Replaced with `owner_username`.
3. **`get_logs` `stage?` param dropped.** Fez logs aren't structured per
   stage; substring matching would mislead. `level` and `job_id` filters
   kept.
4. **Transport defaults to streamable HTTP**, with SSE as a non-default
   option. SSE is being deprecated upstream; we document both.
5. **`get_circuit_breaker_state` returns `available: false` when the
   server runs out-of-process.** Live state is in-process only. This is a
   known limitation, surfaced in the response itself rather than silently
   returning stale data.
6. **Mounting on the main FastAPI app is the default.** A standalone
   sidecar container (`Dockerfile.mcp`) is supported for the user's
   stated remote-Claude-Code-web case.
7. **Settings-snapshot diff in `compare_renders` is empty in Phase 1.**
   Fez doesn't snapshot settings per render; adding that is a pipeline
   change.

---

**Ready to implement on confirmation.** No code will be written for the
server itself until the user reviews this plan.
