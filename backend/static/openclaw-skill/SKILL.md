---
skill_name: ClipAI
version: "1.0.0"
description: >
  Video intelligence platform — upload videos, analyze content,
  detect viral-worthy clips, add animated subtitles with subject tracking,
  and export with SEO metadata.
base_url: "http://localhost:1353"
auth:
  type: bearer
  header: Authorization
  token_env: CLIPAI_API_KEY
  format: "Bearer <token>"
  obtain: "GET /api/v1/auth/current-key returns the current API key"
endpoints:
  health: "GET /api/v1/health"
  info: "GET /api/v1/info"
  skill: "GET /api/v1/skill"
  upload: "POST /api/v1/videos/upload"
  upload_url: "POST /api/v1/videos/upload-url"
  analyze: "POST /api/v1/videos/{job_id}/analyze"
  status: "GET /api/v1/videos/{job_id}/status"
  clips: "GET /api/v1/videos/{job_id}/clips"
  export: "POST /api/v1/videos/{job_id}/clips/{clip_id}/export"
  download: "GET /api/v1/exports/{export_id}/download"
  seo: "POST /api/v1/videos/{job_id}/clips/{clip_id}/seo/generate"
  workflow: "POST /api/v1/workflows/process-video"
mcp_endpoint: "/mcp"
---

# ClipAI — AI Agent Skill

ClipAI is a self-hosted video intelligence platform that transforms long-form videos into viral short-form clips with animated subtitles, subject tracking, and SEO metadata.

## Quick Start

```bash
# 1. Check health
curl http://localhost:1353/api/v1/health

# 2. Get your API key (auto-generated on first boot)
curl http://localhost:1353/api/v1/auth/current-key

# 3. Upload a video
curl -X POST http://localhost:1353/api/v1/videos/upload \
  -H "Authorization: Bearer <your-key>" \
  -F "file=@video.mp4"

# 4. Start analysis (returns immediately)
curl -X POST http://localhost:1353/api/v1/videos/{job_id}/analyze \
  -H "Authorization: Bearer <your-key>"

# 5. Poll status until complete
curl http://localhost:1353/api/v1/videos/{job_id}/status \
  -H "Authorization: Bearer <your-key>"

# 6. Get detected viral clips
curl http://localhost:1353/api/v1/videos/{job_id}/clips \
  -H "Authorization: Bearer <your-key>"

# 7. Export a clip
curl -X POST http://localhost:1353/api/v1/videos/{job_id}/clips/{clip_id}/export \
  -H "Authorization: Bearer <your-key>" \
  -H "Content-Type: application/json" \
  -d '{"aspect_ratio": "9:16"}'

# 8. Download when ready
curl http://localhost:1353/api/v1/exports/{export_id}/download \
  -H "Authorization: Bearer <your-key>" -o clip.mp4
```

## One-Shot Workflow

For agents that want to process a video end-to-end in a single call:

```bash
curl -X POST http://localhost:1353/api/v1/workflows/process-video \
  -H "Authorization: Bearer <your-key>" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://youtube.com/watch?v=example",
    "auto_export": true,
    "aspect_ratio": "9:16",
    "callback_url": "https://your-webhook.example.com/done"
  }'
```

This uploads, analyzes, detects clips, and optionally exports all clips automatically.

## Step-by-Step API Usage

### Authentication

All `/api/v1/*` endpoints (except `/health`, `/info`, `/skill`) require a Bearer token:

```
Authorization: Bearer clipai-xxxxx
```

The key is auto-generated on first boot. Retrieve it from the Settings page or via:
```
GET /api/v1/auth/current-key
```

### Response Format

Every response follows a consistent envelope:

```json
{
  "success": true,
  "data": { ... },
  "error": null,
  "meta": {
    "timestamp": "2025-01-01T00:00:00Z",
    "request_id": "req_abc123"
  }
}
```

On error:
```json
{
  "success": false,
  "data": null,
  "error": {
    "code": "INVALID_FILE",
    "message": "No file provided"
  },
  "meta": { ... }
}
```

### Video Upload

**File upload:**
```
POST /api/v1/videos/upload
Content-Type: multipart/form-data
Body: file=@video.mp4
```

**URL upload (YouTube, direct links):**
```
POST /api/v1/videos/upload-url
Content-Type: application/json
{
  "url": "https://youtube.com/watch?v=...",
  "language": "",
  "auto_analyze": true,
  "callback_url": ""
}
```

### Analysis

```
POST /api/v1/videos/{job_id}/analyze
```

Analysis is async. Poll the status endpoint or provide a `callback_url`:

```
GET /api/v1/videos/{job_id}/status
```

Status progression: `queued` → `transcribing` → `analyzing` → `finding_clips` → `complete`

### Retrieving Results

```
GET /api/v1/videos/{job_id}/summary      # AI-generated video summary
GET /api/v1/videos/{job_id}/transcript    # Full transcript with timestamps
GET /api/v1/videos/{job_id}/scenes        # Scene-by-scene breakdown
GET /api/v1/videos/{job_id}/clips         # Detected viral clips
```

### Clip Configuration

Before exporting, you can adjust clip settings:

```
PUT /api/v1/videos/{job_id}/clips/{clip_id}/settings
Content-Type: application/json
{
  "aspect_ratio": "9:16",
  "font_family": "DM Sans",
  "font_size": 9,
  "font_color": "#FFFFFF",
  "stroke_color": "#000000",
  "stroke_width": 3,
  "subtitle_position": 70,
  "animation": "word-highlight"
}
```

### Available Clip Settings

| Setting | Type | Default | Range / Options |
|---------|------|---------|-----------------|
| `aspect_ratio` | string | `"9:16"` | `"9:16"`, `"16:9"`, `"1:1"`, `"4:5"` |
| `font_family` | string | `"DM Sans"` | Any installed font |
| `font_size` | number | `9` | 1 – 20 |
| `font_color` | string | `"#FFFFFF"` | Hex color |
| `stroke_color` | string | `"#000000"` | Hex color |
| `stroke_width` | number | `3` | 0 – 10 |
| `highlight_color` | string | `"#FFD700"` | Hex color |
| `subtitle_position` | number | `70` | 0 – 100 (% from top) |
| `animation` | string | `"word-highlight"` | `"word-highlight"`, `"karaoke"`, `"fade-in"`, `"none"` |
| `words_per_group` | number | `3` | 1 – 8 |
| `max_words_per_line` | number | `5` | 1 – 10 |
| `subject_tracking` | boolean | `true` | Enable smart cropping |

### Export

```
POST /api/v1/videos/{job_id}/clips/{clip_id}/export
Content-Type: application/json
{
  "aspect_ratio": "9:16",
  "callback_url": ""
}
```

Returns an `export_id`. Poll or use webhook:

```
GET /api/v1/exports/{export_id}
```

When complete, download:
```
GET /api/v1/exports/{export_id}/download
```

### Batch Export

Export all clips at once:

```
POST /api/v1/videos/{job_id}/clips/batch-export
Content-Type: application/json
{
  "aspect_ratio": "9:16",
  "callback_url": ""
}
```

### SEO Generation

Generate titles, descriptions, hashtags, and hooks for a clip:

```
POST /api/v1/videos/{job_id}/clips/{clip_id}/seo/generate
```

Retrieve generated SEO:
```
GET /api/v1/videos/{job_id}/clips/{clip_id}/seo
```

### Settings

```
GET  /api/v1/settings              # Current system settings
PATCH /api/v1/settings             # Update settings
GET  /api/v1/settings/providers    # Provider status
```

## MCP Integration

ClipAI also exposes an MCP (Model Context Protocol) server at `/mcp` for compatible AI agents. The MCP server provides the same capabilities as the REST API through tool calls.

Connect with any MCP-compatible client:
```
Endpoint: http://localhost:1353/mcp
Transport: Streamable HTTP (stateless)
```

Available MCP tools: `upload_video`, `list_jobs`, `get_job_status`, `start_analysis`, `get_summary`, `get_transcript`, `get_scenes`, `get_viral_clips`, `configure_clip`, `export_clip`, `get_export_status`, `download_clip`, `generate_seo`, `get_seo_metadata`, `get_settings`, `update_settings`, `process_video_full`, `batch_export`.

## Error Handling

Common error codes:
- `JOB_NOT_FOUND` — Invalid job_id
- `CLIP_NOT_FOUND` — Invalid clip_id
- `NOT_ANALYZED` — Video hasn't been analyzed yet
- `ALREADY_ANALYZING` — Analysis already in progress
- `INVALID_FILE` — Upload issue
- `EXPORT_FAILED` — FFmpeg encoding error

Always check `response.success` before accessing `response.data`.

## Tips for Agents

1. **Use the one-shot workflow** (`POST /workflows/process-video`) for simple tasks — it handles upload through export.
2. **Poll with backoff** — status checks are cheap, but use 2-5 second intervals.
3. **Prefer URL upload** over file upload when possible — it handles YouTube and direct links.
4. **Check `/health` first** to verify the service is running.
5. **Use webhooks** (`callback_url`) instead of polling for production workflows.
6. **SEO generation requires an AI provider** — ensure at least one provider key is configured in Settings.
7. **Subject tracking** is enabled by default — it keeps the speaker centered in vertical crops.
8. **The MCP endpoint** at `/mcp` provides the same tools if your agent supports MCP.
