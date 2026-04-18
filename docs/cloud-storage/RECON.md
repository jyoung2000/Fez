# Cloud Storage Integration — Phase 0 Reconnaissance

This document captures the state of the ClipAI codebase as of the start of the
cloud-storage-integration branch and the decisions that follow from it. It
exists so the rest of the phases can be executed without re-tracing the code
every time and so reviewers can see the assumptions the implementation is
built on.

## 1. Upload path & ingestion entry point

A local upload today goes through the chunked endpoints in
`backend/routers/chunked_upload.py`:

1. `POST /api/upload/init` — `init_upload()` at
   `backend/routers/chunked_upload.py:163`. Validates the extension against
   `ALLOWED_EXTENSIONS = {mp4, mov, avi, mkv, webm, m4v, 3gp}`, pre-creates
   `/data/uploads/.chunked_<upload_id>/video.assembling.<ext>` for the
   write-in-place path, and stores the session metadata (filename, size,
   language, subtitle language, content type overrides, game type, etc.) in
   the in-memory `_active_uploads` dict plus a disk-backed
   `upload_state` session so it survives restarts.
2. `POST /api/upload/chunk` — `upload_chunk()` at
   `backend/routers/chunked_upload.py:226`. Writes each chunk directly at its
   byte offset with `os.pwrite`-style `lseek+write`, verifies CRC32/SHA-256
   for each chunk, and updates the persistent session every 25%.
3. `POST /api/upload/complete` — `complete_upload()` at
   `backend/routers/chunked_upload.py:576`. Kicks off
   `_assemble_and_finalize()` as an `asyncio` task, returns immediately with
   `{job_id, poll: true}` to dodge upstream proxy timeouts.
4. `_assemble_and_finalize()` at
   `backend/routers/chunked_upload.py:393` is the real "ingest" worker. It:
   - Renames (or assembles) the upload into
     `/data/uploads/<job_id>/video.<ext>`.
   - Runs `validate_file_integrity` and `validate_video_header` from
     `backend/services/video_validation.py`.
   - Builds a `JobResult` with the UI metadata (language, subtitle language,
     `content_type_override`, `game_type`, `anime_subtype`, `music_subtype`,
     `sports_subtype`) and persists it via `database.save_job()`.
   - If `settings.AUTO_ANALYZE` is true, creates an `asyncio.Task` for
     `backend.services.pipeline.run_analysis(job_id)`.

The non-chunked single-shot endpoint at `backend/routers/upload.py:234`
(`upload_video()`) does the same thing inline: streams multipart directly to
`/data/uploads/<job_id>/video.<ext>`, validates the header, creates a
`JobResult`, calls `database.save_job`, and schedules `run_analysis`.

### Finding: there is no single `ingest_video_from_path()` today

The work between "file is on disk at its final path" and
"`run_analysis(job_id)` has been scheduled" is duplicated in two places
(`chunked_upload._assemble_and_finalize` and `upload.upload_video`). Both
end in an identical tail:

```python
job = JobResult(
    job_id=job_id,
    filename=filename,
    file_path=video_path,
    file_size_mb=round(total_bytes / (1024 * 1024), 2),
    language=lang,
    subtitle_language=sub_lang,
    content_type_override=ct_override,
    game_type=gt,
    anime_subtype=anime_sub,
    music_subtype=music_sub,
    sports_subtype=sports_sub,
    status=JobStatus.QUEUED,
    progress=0,
    progress_message="Uploaded, waiting for analysis",
    created_at=now,
    updated_at=now,
)
await database.save_job(job)
if settings.AUTO_ANALYZE:
    asyncio.create_task(run_analysis(job_id))
```

### Decision: extract a shared helper

We will introduce a small helper in a new module
`backend/services/ingest.py`:

```python
async def ingest_video_from_path(
    video_path: str,
    *,
    filename: str,
    file_size_bytes: int,
    metadata: IngestMetadata,
    job_id: str | None = None,
) -> str: ...
```

Responsibilities:

1. Validate the extension, then the header via
   `backend.services.video_validation`.
2. Move the file (if necessary) into `/data/uploads/<job_id>/video.<ext>` —
   the same layout everything downstream already expects.
3. Build the `JobResult` with the provided metadata.
4. `database.save_job(job)`.
5. If `settings.AUTO_ANALYZE`, schedule `run_analysis(job_id)`.
6. Return `job_id`.

`IngestMetadata` is a dataclass carrying the same language /
`content_type_override` / `game_type` / `*_subtype` fields that the chunked
upload already threads through. Both existing code paths
(`_assemble_and_finalize` and `upload_video`) should be refactored to call
this helper so the cloud-import path and the two local-upload paths are
byte-identical from the moment the file is on disk.

This is the smallest possible refactor that still satisfies the
"single ingest entry point" non-negotiable in the task spec. It does not
touch `pipeline.run_analysis` or anything downstream.

## 2. Auth / session model

`backend/auth.py` implements a single Bearer-token scheme that only protects
`/api/v1/*`. The frontend (Dashboard, Upload, Settings) does not send an
authentication header — it uses the relative `/api/*` paths directly. There
is no user model, no session cookie, no multi-tenant concept anywhere in the
codebase.

### Decision: hard-code `user_id = "local"`

For cloud OAuth token storage we still need a stable key so tokens can be
scoped per account, but since ClipAI on Unraid is single-user we will:

- Default `user_id` to the literal string `"local"` everywhere.
- Centralize it in `backend/app/cloud/auth.py::current_user_id()` so the
  day we add real auth, one function changes.

This keeps the migration story clean without pretending ClipAI has
multi-tenancy.

## 3. Config & secret storage

Secrets today live in two places:

1. **Environment variables** read by `backend/config.py::Settings`
   (pydantic-settings `BaseSettings` reading from `.env`). Examples:
   `OPENROUTER_API_KEY`, `HF_AUTH_TOKEN`, `ANTHROPIC_API_KEY`.
2. **Generated-and-persisted secrets** read/written by `backend/auth.py`
   under `/data/logs/api_key.json`. Used for the self-generated Bearer token.

There is no existing encryption for stored secrets.

### Decision

- **OAuth client credentials** (one-per-provider, operator-supplied) live in
  the same place every other third-party key lives: env vars loaded through
  `Settings`. They will be referenced in `.env.example` and the Unraid
  template as `GOOGLE_DRIVE_CLIENT_ID`, `GOOGLE_DRIVE_CLIENT_SECRET`,
  `GOOGLE_DRIVE_REDIRECT_URI`, `BOX_CLIENT_ID`, `BOX_CLIENT_SECRET`,
  `BOX_REDIRECT_URI`.
- **OAuth user tokens** (per-connected-account access + refresh tokens) are
  encrypted with Fernet using a new `CLIPAI_TOKEN_ENC_KEY` env var and
  stored per-account in the JSON "db" described in §4. If the key is
  missing, `backend/app/cloud/crypto.py` will generate one at startup and
  log a loud warning telling the operator to persist it in their Unraid
  template — rotating it invalidates all stored cloud sessions, and we
  never want that to happen silently.
- **Logging redaction**: a log filter under
  `backend/app/cloud/logging_filter.py` scrubs any occurrence of
  `access_token`, `refresh_token`, or an `Authorization:` header value in
  formatted log records, so even a rogue `logger.debug(response.json())`
  cannot leak a token.

## 4. Existing "DB" layer

There is **no SQLAlchemy and no Alembic**. `backend/database.py` is a
filesystem-backed JSON store: each job is `/data/uploads/<job_id>/job.json`,
written atomically via `tempfile.mkstemp` + `os.replace`, protected by an
in-memory `asyncio.Lock` keyed on `job_id`. `list_jobs()` iterates the
directory.

### Decision: no Alembic migration

Since there is no relational DB, the "`cloud_accounts` table" from the spec
becomes a small JSON store that mirrors the existing pattern:

- Directory: `/data/cloud_accounts/`
- File per account: `<account_id>.json` containing the schema described in
  the spec (id, user_id, provider, provider_user_id, display_name,
  encrypted access/refresh tokens, token_expires_at, scopes, timestamps).
- Atomic writes via `tempfile.mkstemp` + `os.replace`, same pattern as
  `database.save_job`.
- `backend/app/cloud/accounts.py` is the module that owns all reads/writes,
  so if we ever move to SQLite/Postgres the data-access layer changes in
  exactly one place.

The "unique (user_id, provider, provider_user_id)" constraint is enforced at
write time by scanning the directory and rejecting duplicates — acceptable
at single-user scale (O(n) where n ≤ 2 accounts is the realistic cap).

## 5. Open questions / confirmations

None of the above forces a larger refactor than the Phase 0 prompt
anticipated. The only non-trivial refactor is the extraction of
`ingest_video_from_path`, and it is a pure code move — no behavior change
for local uploads. Proceeding to Phase 1.
