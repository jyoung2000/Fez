# Fez MCP Debug Server — Exploration Notes

Notes on the Fez/ClipAI codebase as it relates to building a read-only MCP
debug server for the reframing pipeline. All paths verified by reading the
actual files.

## Key terminology

- Fez is branded internally as **ClipAI**. The repo is `Fez` but every
  identifier in the code says `clipai`. The new MCP server should follow
  whichever is more recognizable to the user — defaulting to `clipai` for
  internal symbols (matches existing code) and `fez` for the user-facing
  server name and Docker image.
- A "render" in user vocabulary maps to a **`JobResult`** in code. There is
  no separate render entity. Each job represents one upload + one analysis +
  zero or more clip exports. Clip exports re-use the parent job's
  `RenderPlan` with a `clip_range`.

## 1. Storage layer

**Backend: JSON files on disk.** No SQL database.

- File: `/home/user/Fez/backend/database.py`
- Job dir: `/data/uploads/{job_id}/` (hardcoded — `_job_dir()` line 42)
- Job state: `/data/uploads/{job_id}/job.json` (single-file write per job)
- Atomic writes via `tempfile.mkstemp` + `os.replace` (line 60-64)
- Per-job `asyncio.Lock` cache at module level

**Read-side primitives (use these directly, do not reimplement):**

```python
from backend.database import (
    load_job,                # async (job_id) -> Optional[JobResult]
    list_jobs,               # async (owner_user_id?, owner_username?, include_unowned?) -> list[JobResult]
    save_job,                # do NOT call from MCP server — read-only
    delete_job,              # do NOT call
)
```

`list_jobs()` walks `/data/uploads/*/job.json` and returns parsed
`JobResult` instances. It does not sort. Caller must sort by
`job.created_at` or `job.updated_at` descending to get "recent renders".

**Other on-disk artifacts per job:**

```
/data/uploads/{job_id}/
  ├── job.json                # JobResult (the single source of truth)
  ├── <original_filename>     # source video (path stored in job.file_path)
  ├── frames/frame_NNN.jpg    # uniform-sampled VLM frames
  ├── audio.wav               # extracted audio for Whisper
  └── dense_frames/*.jpg      # 2 Hz dense face-tracking frames
```

`exported_clips` is a list[dict] field on the JobResult; the MP4 files
themselves live under `/data/outputs/{job_id}/` (see docker-compose volume
mount on docker-compose.yml line 12).

**Logs:**
- File: `/data/logs/app.log` with rotation (`.1`, `.2`, `.3`)
- Single global file — **NOT per-job**.
- Defined in `backend/main.py` lines 47-66.
- Logger format: `%(asctime)s [%(levelname)s] %(name)s: %(message)s`
- Logs are wiped on container start (lines 57-59).
- Per-job correlation: most pipeline log lines include `job_id=<...>` in
  the message body. There is no structured per-job log file or DB.

## 2. JobResult model — the core schema

File: `/home/user/Fez/backend/models.py` line 170. Selected fields the MCP
server will need to surface (full list ~70 fields):

```python
class JobResult(BaseModel):
    job_id: str
    filename: str
    file_path: str                           # absolute path to source video
    owner_user_id: str = ""
    owner_username: str = ""
    source_sha256: str = ""
    timings: dict = {}                       # {stage_name: wall_seconds}
    pipeline_warnings: list[str] = []        # soft warnings from pipeline
    duration: float
    resolution: str                          # "1920x1080"
    fps: float
    file_size_mb: float
    status: JobStatus                        # enum, see below
    progress: int                            # 0-100
    progress_message: str
    provider_used: dict                      # {"scenes": "openrouter:gemini-2.5-flash", ...}
    created_at: str                          # ISO timestamp string
    updated_at: str
    analysis_started_at: Optional[str]
    analysis_duration_seconds: Optional[float]
    summary: Optional[VideoSummary]
    scenes: list[SceneDescription]           # VLM-described keyframes
    transcript: list[TranscriptSegment]      # Whisper output w/ speakers
    translated_transcript: list[TranscriptSegment]
    clips: list[ClipCandidate]               # detected viral clip windows
    speaker_names: dict[str, str]            # {"Speaker 1": "Eric", ...}
    scene_cut_timestamps: list[float]        # detected shot cuts
    exported_clips: list[dict]               # rendered MP4 metadata
    error: Optional[str]
    estimated_cost_usd: Optional[float]
    face_registry_data: Optional[dict]       # serialized FaceRegistry
    layout_timeline: list[dict]              # [{start,end,layout_mode,face_positions}]
    default_layout_mode: str                 # "single"|"split"|"triple"|"grid"
    dense_tracking_summary: Optional[dict]
    subject_track: list[dict]                # 2 Hz [{t,x,y,conf,source}]
    tracking_mode: str                       # "continuous"|"multi_cluster"|"gameplay"
    content_type_override: str
    game_type: str
    anime_subtype: str
    music_subtype: str
    sports_subtype: str
    hot_zones: list[dict]
    chapters: list[dict]
    clip_content_type: str
    # The reframing IR — RenderPlan serialized via .to_dict():
    render_plan: dict                        # NOT a typed field, raw dict
```

**Note:** `render_plan` is stored as `dict`, not as a typed Pydantic model.
The actual typed version lives at
`backend/services/render_plan.py:RenderPlan` (a `@dataclass`, not Pydantic),
and is serialized via `RenderPlan.to_dict()` before being attached to the
job. The MCP server should re-parse it back to the dataclass when needed
for validation / pretty-printing.

`JobStatus` enum (models.py line 6): `QUEUED`, `EXTRACTING_FRAMES`,
`TRANSCRIBING`, `ANALYZING_SCENES`, `GENERATING_SUMMARY`,
`DETECTING_CLIPS`, `COMPLETE`, `FAILED`, `CANCELLED`.

## 3. RenderPlan IR

File: `/home/user/Fez/backend/services/render_plan.py`

```python
@dataclass
class RenderPlan:
    source_width: int
    source_height: int
    target_width: int                        # e.g. 1080
    target_height: int                       # e.g. 1920
    total_duration_sec: float
    fps: float
    ops: List[RenderOp]
    source_offset_sec: float = 0.0           # for clip exports

    def to_json(self) -> str: ...
    def to_dict(self) -> dict: ...
    def validate(self) -> List[str]: ...     # returns list of violations

@dataclass
class RenderOp:
    kind: RenderOpKind
    start_sec: float
    end_sec: float
    primary_rect: Rect                       # normalized 0-1
    secondary_rect: Optional[Rect] = None    # for split/stacked/grid
    tertiary_rect: Optional[Rect] = None
    quaternary_rect: Optional[Rect] = None
    motion_path: List[MotionKeypoint] = []   # for tracking ops
    ease_in_ms: int = 0
    strategy_label: str = ""
    content_type: str = "unknown"
    gaming_layout_mode: Optional[str] = None
    speaker_slot: Optional[int] = None
    speaker_label: Optional[str] = None

class RenderOpKind(str, Enum):
    CROP, TRACKING_CROP, WIDE_MASTER, BLUR_FILL,
    SPLIT_SCREEN, STACKED_GAMEPLAY, GRID_2X2,
    MOTIVATED_PUSH_IN, MOTIVATED_PULL_OUT
```

`RenderPlan.validate()` returns list[str] of structural violations
(non-contiguous ops, missing primary_rect, etc.) — directly useful to
expose as a `validate_render_plan` tool.

There's also a debug payload at
`backend/services/render_plan_debug.py:build_debug_payload()` that produces
`render_plan["debug"]` with content-routing, per-segment, editorial-prior,
and pacing diagnostics.

## 4. Pipeline stages — verified import paths

File: `/home/user/Fez/backend/services/pipeline.py` (~6000 lines)

| Stage | Module | Function |
|---|---|---|
| Orchestrator | `backend.services.pipeline` | `async run_analysis(job_id)` (line 596), inner `_run_analysis_inner` (line 738) |
| Cancel API | `backend.services.pipeline` | `request_cancel(job_id)`, `is_cancel_requested(job_id)` |
| Ingestion | `backend.services.ingest` | `ingest_video_from_path(...)` |
| Transcription | `backend.services.transcription` | `transcribe_audio(...)` |
| Scene VLM | `backend.services.ai_orchestrator` | `AIOrchestrator.analyze_frames(...)` |
| Face detection | `backend.services.face_detector` | `detect_faces_batch(...)` |
| Face registry | `backend.services.face_registry` | `build_face_registry(...)` |
| Active speaker | `backend.services.active_speaker` | `build_active_speaker_timeline_v3(...)` |
| Diarization | `backend.services.speaker_diarization` | `diarize_audio(...)` |
| Reframe segmenter | `backend.services.reframe_segmenter` | `build_reframe_segments(...)` |
| RenderPlan build | `backend.services.render_plan_builder` | `build_render_plan(...)` |
| RenderPlan debug | `backend.services.render_plan_debug` | `build_debug_payload(...)` |
| Human override | `backend.services.human_reframe_bridge` | `maybe_override_render_plan(...)` |
| FFmpeg export | `backend.services.clip_exporter` | `async export_clip(...)` |

## 5. Circuit breaker

File: `/home/user/Fez/backend/services/ai_orchestrator.py` line 26 (`class _CircuitBreaker`).

- **In-memory only.** No Redis, no file persistence. State lives on each
  `AIOrchestrator` instance.
- 3 failures within 600s (10 min) → degraded for 900s (15 min).
- API: `is_degraded(name)`, `record_failure(name)`, `record_success(name)`,
  `clear_degraded(name)`, `force_reset_all()`.
- Internal state: `_failures: dict[str, list[float]]` (monotonic timestamps),
  `_degraded_until: dict[str, float]`.

**Critical caveat for the MCP server:** the orchestrator is instantiated
inside `pipeline._run_analysis_inner`, not as a singleton. The "live"
circuit breaker state is the orchestrator's, but the MCP server runs in
the same Python process only if mounted alongside the FastAPI app. If the
MCP server is run as a separate process (per the user's request for SSE on
:8765 from a separate container/container-group), **it has no access to
the in-process circuit breaker state**.

This needs to be flagged in the plan: either (a) expose the orchestrator
state via a small in-process registry that gets persisted to a small JSON
file, (b) mount the MCP server in-process inside the FastAPI app (like the
existing `/mcp` mount), or (c) admit `get_circuit_breaker_state()` returns
"unavailable, server in separate process" and treat live VLM health as a
Phase 3 enhancement. **Recommendation: option (b) — in-process — for the
debug MCP server.** This also gives us free access to the same
JobResult cache, settings, and live config.

## 6. Configuration

File: `/home/user/Fez/backend/config.py`. `Settings` is a
`pydantic_settings.BaseSettings` subclass. Imported as
`from backend.config import settings`. Read-only access pattern is to call
`get_settings()` (cached via `@lru_cache`).

No dedicated render artifact path — everything is rooted at
`/data/uploads/{job_id}/`. Output MP4s land in `/data/outputs/{job_id}/`.

## 7. Existing MCP setup (this is important — do not duplicate)

Two existing files, both **workflow-focused** (upload, list, export, SEO):

- `backend/mcp_server.py` — stdio-only standalone server that calls the
  REST API over HTTP. Predates the in-process FastMCP mount.
- `backend/mcp_tools.py` — `register_tools(mcp)` for the FastMCP instance
  mounted at `/mcp` from `backend/main.py:541-559`.

The FastMCP instance is created with `stateless_http=True` and mounted via
`app.mount("/mcp", clipai_mcp.streamable_http_app())`. Transport is
**streamable HTTP**, not SSE. Bearer token auth is **not** present on this
mount.

The new debug MCP server is genuinely additive: zero overlap with the
existing tools, focused on pipeline-internal state (RenderPlan, circuit
breaker, dense subject track, layout timeline, scene cuts, timings, raw
warnings).

**Implementation note:** the simplest deployment is to add a second
FastMCP instance, register the new debug tools on it, and mount it at
`/mcp/debug` next to the existing `/mcp` mount — both run in-process. The
user's stated requirement (SSE on `0.0.0.0:8765` for remote Claude Code
web access) does require a separate listener; we can satisfy both by
exposing the same FastMCP via two transports if FastMCP supports it, or
by running a thin async wrapper. See plan.

## 8. Dependencies & Python

- Manager: **`requirements.txt`** (no Poetry, no `pyproject.toml`).
- File: `/home/user/Fez/backend/requirements.txt`.
- `mcp>=1.8.0` is already pinned (line 31). No new dependency needed.
- Python: 3.11 (from `Dockerfile` line 11: `FROM python:3.11-slim`).
- Existing deps that might matter for the MCP server: `fastapi==0.111.0`,
  `pydantic>=2.7.2`, `pydantic-settings`, `aiofiles==23.2.1`,
  `httpx==0.27.2`, `uvicorn[standard]==0.30.0`.

## 9. Tests

- Framework: pytest (no fixtures dir; tests are self-contained).
- Top-level: `/home/user/Fez/tests/` — 7 integration test modules.
- Service-level: `/home/user/Fez/backend/tests/` (referenced in folder
  listing). Will need to confirm during implementation.
- No conftest, no shared fixtures inferred from listing.
- Test command: standard `pytest` (no Makefile or `tox.ini`).

## 10. Docker / Unraid

- Two compose files: `docker-compose.yml` (GPU) and
  `docker-compose.gpu.yml` (also GPU? — confirm during implementation).
- Both mount `./data/uploads:/data/uploads` and `./data/outputs:/data/outputs`.
- Single existing port: `1353:1353`.
- Volumes confirmed at `/data/uploads`, `/data/outputs`, `/data/logs`,
  `/data/auth`, `/data/fonts`.

For the new MCP container, the right pattern is a **slim sidecar**: same
`python:3.11-slim` base, install only `mcp`, `pydantic`, `pydantic-settings`,
`aiofiles`, mount `/data/uploads` read-only, expose `8765`. No torch, no
opencv, no ffmpeg.

But: in-process mounting on the existing FastAPI app is simpler and gets
us live circuit breaker state for free. The plan should propose **both
modes** with in-process as the default, sidecar for users who want
isolation.

## 11. Programmatic reframe trigger

- `pipeline.run_analysis(job_id)` is the entry point. It expects the
  `JobResult` to already exist on disk (via `database.save_job()`).
- A user-flow trigger creates the job via
  `backend.routers.agent.upload_from_url` or similar, then `run_analysis`
  is fired as a background task.
- For the MCP server's Phase 1 (read-only), we don't need this. Phase 2
  (`trigger_reframe`, `replay_render`) would call `run_analysis` directly
  after constructing/reusing a JobResult.

## 12. Replay / checkpoint capability

**Does not exist as a first-class feature.** The pipeline persists
intermediate state on the JobResult after each major stage (transcript,
scenes, render_plan, etc.), and on re-analyze it skips re-extraction when
`source_sha256` matches and frames already exist on disk. But there is no
explicit "replay from stage X" API. This is correctly Phase 3 work and
should be flagged as such in the plan.

## 13. Real bugs / inconsistencies noticed (out of scope; not fixing)

- `backend/mcp_server.py` line 250 — `clipai_list_jobs` calls
  `/api/jobs-paginated` but `clipai_get_job` calls `/api/jobs/{id}`. These
  endpoint names should be confirmed; if `/api/jobs-paginated` doesn't
  exist the existing tool is broken. **Not investigating further per the
  scope rule.**
- `JobResult.render_plan` is typed as `dict` but the rest of the code
  treats it as `RenderPlan.to_dict()`. Round-tripping through Pydantic
  would lose the typed dataclass. The new MCP server should use
  `dataclasses` reflection to re-construct typed `RenderPlan` instances.
- The existing `mcp_tools.py` doesn't add bearer auth. If the new debug
  server adds auth and is mounted on the same FastAPI app, users may be
  surprised that `/mcp` (workflow) is unauthenticated while
  `/mcp/debug` (debug) requires a token. Worth a one-liner in the
  README, not a blocker.

## 14. Open questions for the user (none blocking)

None. The plan can be drafted with the recommendations above and confirmed
with the user before any code is written.
