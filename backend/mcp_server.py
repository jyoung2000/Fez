"""MCP (Model Context Protocol) server for ClipAI.

Wraps the ClipAI REST API as MCP tools so that Claude Code, Cursor, Windsurf,
and other MCP-compatible AI tools can natively discover and use all ClipAI
capabilities through their tool-use interface.

Usage:
    python -m backend.mcp_server               # stdio transport (default)
    python -m backend.mcp_server --port 1354   # SSE transport on port 1354

Configure in your MCP client (e.g. claude_desktop_config.json):
    {
      "mcpServers": {
        "clipai": {
          "command": "python",
          "args": ["-m", "backend.mcp_server"],
          "env": {"CLIPAI_BASE_URL": "http://localhost:1353"}
        }
      }
    }
"""

import asyncio
import json
import logging
import os
import sys

import httpx

logger = logging.getLogger("clipai.mcp")

BASE_URL = os.environ.get("CLIPAI_BASE_URL", "http://localhost:1353")

TOOLS = [
    {
        "name": "clipai_health",
        "description": "Check if ClipAI is ready: Whisper model loaded, API keys configured, disk space available.",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "clipai_setup",
        "description": "Configure ClipAI API keys and default settings in one call.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "openrouter_api_key": {"type": "string", "description": "OpenRouter API key"},
                "anthropic_api_key": {"type": "string", "description": "Anthropic API key"},
                "gemini_api_key": {"type": "string", "description": "Gemini API key"},
                "groq_api_key": {"type": "string", "description": "Groq API key"},
                "preset": {"type": "string", "description": "AI preset: free, efficient, balanced, premium", "default": "balanced"},
            },
        },
    },
    {
        "name": "clipai_upload_video",
        "description": "Upload a video from URL for AI analysis and clip generation. Supports direct video URLs and YouTube/TikTok/Instagram URLs.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "Video URL to download"},
                "language": {"type": "string", "description": "ISO 639-1 language code (optional, empty=auto-detect)", "default": ""},
                "auto_analyze": {"type": "boolean", "description": "Start analysis immediately", "default": True},
            },
            "required": ["url"],
        },
    },
    {
        "name": "clipai_get_job",
        "description": "Get full details of a video analysis job including status, transcript, scenes, and detected clips.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "job_id": {"type": "string", "description": "Job ID"},
            },
            "required": ["job_id"],
        },
    },
    {
        "name": "clipai_list_jobs",
        "description": "List all video analysis jobs with pagination.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "description": "Max jobs to return", "default": 20},
                "offset": {"type": "integer", "description": "Number of jobs to skip", "default": 0},
                "status": {"type": "string", "description": "Filter by status (e.g. 'complete', 'running')"},
            },
        },
    },
    {
        "name": "clipai_export_clip",
        "description": "Export a video clip with optional subtitles, aspect ratio cropping, and quality settings.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "job_id": {"type": "string", "description": "Job ID"},
                "clip_id": {"type": "integer", "description": "Clip candidate ID"},
                "start": {"type": "number", "description": "Start time in seconds"},
                "end": {"type": "number", "description": "End time in seconds"},
                "aspect_ratio": {"type": "string", "description": "Aspect ratio: 16:9, 9:16, 1:1, 4:5", "default": "9:16"},
                "export_quality": {"type": "string", "description": "Quality: 720p, 1080p, 4k", "default": "1080p"},
                "subtitles_enabled": {"type": "boolean", "description": "Burn subtitles into video", "default": True},
                "clip_title": {"type": "string", "description": "Optional title for the exported file"},
            },
            "required": ["job_id", "clip_id", "start", "end"],
        },
    },
    {
        "name": "clipai_export_status",
        "description": "Check the status of a clip export. Returns download_url when complete.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "job_id": {"type": "string", "description": "Job ID"},
                "clip_id": {"type": "integer", "description": "Clip ID"},
            },
            "required": ["job_id", "clip_id"],
        },
    },
    {
        "name": "clipai_generate_seo",
        "description": "Generate SEO-optimized title, description, tags, and YouTube descriptions for a clip.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "job_id": {"type": "string", "description": "Job ID"},
                "clip_id": {"type": "integer", "description": "Clip ID"},
            },
            "required": ["job_id", "clip_id"],
        },
    },
    {
        "name": "clipai_generate_clips",
        "description": "Re-run viral clip detection with custom parameters (duration range, count, focus topic).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "job_id": {"type": "string", "description": "Job ID"},
                "min_duration": {"type": "number", "description": "Minimum clip duration in seconds", "default": 15},
                "max_duration": {"type": "number", "description": "Maximum clip duration in seconds", "default": 90},
                "clip_count": {"type": "integer", "description": "Number of clips to detect"},
                "clip_focus": {"type": "string", "description": "Focus topic (e.g. 'cooking tips', 'funny moments')"},
            },
            "required": ["job_id"],
        },
    },
    {
        "name": "clipai_pipeline",
        "description": "Full end-to-end pipeline: download video, analyze, detect clips, export, and generate SEO — all in one call.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "Video URL to process"},
                "language": {"type": "string", "description": "ISO language code (optional)", "default": ""},
                "clip_count": {"type": "integer", "description": "Number of clips to detect", "default": 5},
                "min_duration": {"type": "number", "description": "Min clip duration (seconds)", "default": 15},
                "max_duration": {"type": "number", "description": "Max clip duration (seconds)", "default": 90},
                "aspect_ratio": {"type": "string", "description": "Export aspect ratio", "default": "9:16"},
                "export_quality": {"type": "string", "description": "Export quality", "default": "1080p"},
                "subtitles_enabled": {"type": "boolean", "description": "Burn subtitles", "default": True},
                "auto_select": {"type": "string", "description": "'top' = export best clip, 'all' = export all", "default": "top"},
                "generate_seo": {"type": "boolean", "description": "Generate SEO metadata", "default": True},
            },
            "required": ["url"],
        },
    },
    {
        "name": "clipai_pipeline_status",
        "description": "Check the status of a running pipeline. Returns clips with download URLs when complete.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "pipeline_id": {"type": "string", "description": "Pipeline ID"},
            },
            "required": ["pipeline_id"],
        },
    },
    {
        "name": "clipai_batch_export",
        "description": "Export multiple clips in one call.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "job_id": {"type": "string", "description": "Job ID"},
                "clip_ids": {"type": "array", "items": {"type": "integer"}, "description": "Clip IDs to export (empty = all)"},
                "aspect_ratio": {"type": "string", "default": "9:16"},
                "export_quality": {"type": "string", "default": "1080p"},
                "subtitles_enabled": {"type": "boolean", "default": True},
                "generate_seo": {"type": "boolean", "default": False},
            },
            "required": ["job_id"],
        },
    },
    {
        "name": "clipai_download_all",
        "description": "Download all exported clips for a job as a ZIP file. Returns the download URL.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "job_id": {"type": "string", "description": "Job ID"},
            },
            "required": ["job_id"],
        },
    },
]


async def _api_call(method: str, path: str, json_body: dict = None, params: dict = None) -> dict:
    """Make an HTTP call to the ClipAI REST API."""
    url = f"{BASE_URL}{path}"
    async with httpx.AsyncClient(timeout=httpx.Timeout(connect=30, read=300, write=30, pool=30)) as client:
        if method == "GET":
            resp = await client.get(url, params=params)
        elif method == "POST":
            resp = await client.post(url, json=json_body)
        elif method == "PUT":
            resp = await client.put(url, json=json_body)
        elif method == "DELETE":
            resp = await client.delete(url)
        else:
            return {"error": f"Unknown method: {method}"}

        try:
            return resp.json()
        except Exception:
            return {"status_code": resp.status_code, "text": resp.text[:500]}


async def handle_tool_call(name: str, arguments: dict) -> str:
    """Route an MCP tool call to the appropriate REST API endpoint."""
    try:
        if name == "clipai_health":
            result = await _api_call("GET", "/api/health")

        elif name == "clipai_setup":
            result = await _api_call("POST", "/api/setup", json_body=arguments)

        elif name == "clipai_upload_video":
            result = await _api_call("POST", "/api/upload-url", json_body=arguments)

        elif name == "clipai_get_job":
            result = await _api_call("GET", f"/api/jobs/{arguments['job_id']}")

        elif name == "clipai_list_jobs":
            params = {k: v for k, v in arguments.items() if v is not None}
            result = await _api_call("GET", "/api/jobs-paginated", params=params)

        elif name == "clipai_export_clip":
            job_id = arguments.pop("job_id")
            result = await _api_call("POST", f"/api/jobs/{job_id}/export-clip", json_body=arguments)

        elif name == "clipai_export_status":
            result = await _api_call("GET", f"/api/jobs/{arguments['job_id']}/export-status/{arguments['clip_id']}")

        elif name == "clipai_generate_seo":
            result = await _api_call("POST", f"/api/jobs/{arguments['job_id']}/seo/{arguments['clip_id']}")

        elif name == "clipai_generate_clips":
            job_id = arguments.pop("job_id")
            result = await _api_call("POST", f"/api/jobs/{job_id}/generate-clips", json_body=arguments)

        elif name == "clipai_pipeline":
            result = await _api_call("POST", "/api/pipeline", json_body=arguments)

        elif name == "clipai_pipeline_status":
            result = await _api_call("GET", f"/api/pipeline/{arguments['pipeline_id']}")

        elif name == "clipai_batch_export":
            job_id = arguments.pop("job_id")
            result = await _api_call("POST", f"/api/jobs/{job_id}/export-batch", json_body=arguments)

        elif name == "clipai_download_all":
            job_id = arguments["job_id"]
            result = {"download_url": f"{BASE_URL}/api/jobs/{job_id}/download-all"}

        else:
            result = {"error": f"Unknown tool: {name}"}

        return json.dumps(result, indent=2, default=str)

    except Exception as e:
        return json.dumps({"error": str(e)})


async def run_stdio():
    """Run as MCP server over stdio (JSON-RPC 2.0)."""
    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    await asyncio.get_event_loop().connect_read_pipe(lambda: protocol, sys.stdin.buffer)

    writer_transport, writer_protocol = await asyncio.get_event_loop().connect_write_pipe(
        asyncio.streams.FlowControlMixin, sys.stdout.buffer
    )
    writer = asyncio.StreamWriter(writer_transport, writer_protocol, None, asyncio.get_event_loop())

    async def send_response(response: dict):
        data = json.dumps(response) + "\n"
        writer.write(data.encode())
        await writer.drain()

    while True:
        line = await reader.readline()
        if not line:
            break

        try:
            msg = json.loads(line.decode())
        except json.JSONDecodeError:
            continue

        method = msg.get("method", "")
        msg_id = msg.get("id")

        if method == "initialize":
            await send_response({
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "clipai", "version": "1.0.0"},
                },
            })

        elif method == "notifications/initialized":
            pass  # Client confirms initialization

        elif method == "tools/list":
            await send_response({
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {"tools": TOOLS},
            })

        elif method == "tools/call":
            tool_name = msg["params"]["name"]
            tool_args = msg["params"].get("arguments", {})
            result_text = await handle_tool_call(tool_name, tool_args)
            await send_response({
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [{"type": "text", "text": result_text}],
                },
            })

        elif msg_id is not None:
            await send_response({
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"},
            })


def main():
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    logger.info("Starting ClipAI MCP server (stdio transport)")
    logger.info("Connecting to ClipAI API at %s", BASE_URL)
    asyncio.run(run_stdio())


if __name__ == "__main__":
    main()
