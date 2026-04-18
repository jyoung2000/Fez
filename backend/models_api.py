"""Pydantic response models for all API endpoints.

These models are used as response_model= on FastAPI decorators so that the
OpenAPI spec (/openapi.json) includes full response schemas — critical for
AI agent frameworks that auto-generate tool definitions from the spec.
"""

from typing import Optional
from pydantic import BaseModel, Field

from backend.models import (
    ClipCandidate,
    ClipSEO,
    JobStatus,
    SceneDescription,
    SubtitleSettings,
    TranscriptSegment,
    VideoSummary,
)


# ── Generic ──────────────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    """Standardized error response."""
    error: str = Field(description="Short error type")
    detail: str = Field(description="Human-readable explanation")
    code: str = Field(description="Machine-readable error code, e.g. JOB_NOT_FOUND")
    retry_after: int = Field(default=0, description="Seconds to wait before retrying (for rate limits)")


# ── Upload ───────────────────────────────────────────────────────────

class UploadResponse(BaseModel):
    job_id: str
    status: str
    filename: str


class UploadUrlRequest(BaseModel):
    url: str = Field(description="Video URL to download (direct link or YouTube/social URL if yt-dlp installed)")
    language: str = Field(default="", description="ISO 639-1 language code for transcription (empty = auto-detect)")
    auto_analyze: bool = Field(default=True, description="Start analysis immediately after download")


class UploadUrlResponse(BaseModel):
    job_id: str
    status: str  # "downloading", "queued", "analyzing"
    filename: str
    deduplicated: bool = False


# ── Jobs ─────────────────────────────────────────────────────────────

class JobSummaryResponse(BaseModel):
    job_id: str
    filename: str
    duration: float
    status: JobStatus
    progress: int
    progress_message: str
    created_at: str
    clips_count: int
    provider_used: dict = {}
    file_size_mb: float = 0.0
    estimated_cost_usd: Optional[float] = None


class JobListResponse(BaseModel):
    jobs: list[JobSummaryResponse]
    total: int
    limit: int
    offset: int


class JobDetailResponse(BaseModel):
    job_id: str
    filename: str
    file_path: str
    language: str = ""
    duration: float = 0.0
    resolution: str = ""
    fps: float = 0.0
    file_size_mb: float = 0.0
    status: JobStatus
    progress: int = 0
    progress_message: str = ""
    provider_used: dict = {}
    created_at: str = ""
    updated_at: str = ""
    analysis_started_at: Optional[str] = None
    analysis_duration_seconds: Optional[float] = None
    summary: Optional[VideoSummary] = None
    scenes: list[SceneDescription] = []
    transcript: list[TranscriptSegment] = []
    clips: list[ClipCandidate] = []
    speaker_names: dict[str, str] = {}
    exported_clips: list[dict] = []
    error: Optional[str] = None
    estimated_cost_usd: Optional[float] = None


class JobStatusResponse(BaseModel):
    job_id: str
    status: str


class JobDeleteResponse(BaseModel):
    job_id: str
    deleted: bool


class AnalysisStartedResponse(BaseModel):
    job_id: str
    status: str = "analysis_started"


# ── Clips / Export ───────────────────────────────────────────────────

class ExportStartedResponse(BaseModel):
    export_id: str
    status: str = "exporting"


class ExportStatusResponse(BaseModel):
    status: str  # "encoding", "complete", "not_found", "failed"
    progress: int = 0
    download_url: Optional[str] = None
    filename: Optional[str] = None
    duration: Optional[float] = None
    export_quality: Optional[str] = None
    message: Optional[str] = None


class ExportCancelResponse(BaseModel):
    export_id: str
    status: str = "cancelled"


class ActiveExportItem(BaseModel):
    export_id: str
    job_id: str
    clip_id: Optional[str] = None
    status: str = "encoding"


class ClipTitleResponse(BaseModel):
    job_id: str
    clip_id: int
    title: str


class ClipTimesResponse(BaseModel):
    job_id: str
    clip_id: int
    start_time: float
    end_time: float
    duration: float


class ClipDeleteResponse(BaseModel):
    job_id: str
    clip_id: int
    deleted: bool = True
    remaining_clips: int


class BulkDeleteResponse(BaseModel):
    job_id: str
    deleted_count: int
    remaining_clips: int


class ClipGenerationResponse(BaseModel):
    status: str = "generating"
    message: str = "Generating clips..."


class ExportedClipItem(BaseModel):
    clip_id: Optional[int] = None
    filename: Optional[str] = None
    download_url: str
    start: Optional[float] = None
    end: Optional[float] = None


class SEOResponse(BaseModel):
    clip_id: int
    provider: str
    seo: ClipSEO


# ── Batch Export ─────────────────────────────────────────────────────

class BatchExportRequest(BaseModel):
    clip_ids: list[int] = Field(default=[], description="Clip IDs to export. Empty = export all clips.")
    aspect_ratio: str = Field(default="9:16", description="Aspect ratio: 16:9, 9:16, 1:1, 4:5")
    export_quality: str = Field(default="1080p", description="Quality: 720p, 1080p, 4k")
    subtitles_enabled: bool = True
    subtitle_settings: Optional[SubtitleSettings] = None
    generate_seo: bool = Field(default=False, description="Also generate SEO for each clip")
    callback_url: str = Field(default="", description="Webhook URL to POST results when batch completes")


class BatchExportResponse(BaseModel):
    batch_id: str
    job_id: str
    status: str = "running"
    clip_count: int
    message: str = ""


class BatchStatusResponse(BaseModel):
    batch_id: str
    job_id: str
    status: str  # "running", "complete", "failed"
    total_clips: int
    completed_clips: int
    failed_clips: int
    clips: list[dict] = []
    message: str = ""


# ── Transcript ───────────────────────────────────────────────────────

class SpeakerRenameResponse(BaseModel):
    job_id: str
    speaker_names: dict[str, str]
    transcript: list[dict]


class TranscriptSegmentResponse(BaseModel):
    job_id: str
    segment_index: int
    text: str
    speaker: str


class TranscriptDeleteResponse(BaseModel):
    job_id: str
    deleted_index: int
    remaining: int


class TranscriptInsertResponse(BaseModel):
    job_id: str
    inserted_index: int
    segment: dict


class WordRefreshResponse(BaseModel):
    job_id: str
    status: str
    segments_total: int = 0
    segments_with_words: int = 0


# ── Scenes ───────────────────────────────────────────────────────────

class SceneAddResponse(BaseModel):
    job_id: str
    scene_count: int
    scene: dict


class SceneUpdateResponse(BaseModel):
    job_id: str
    scene_index: int
    scene: dict


class SceneDeleteResponse(BaseModel):
    job_id: str
    scene_count: int


class RecenterResponse(BaseModel):
    job_id: str
    scenes_recentered: int
    subject_x: float = 50.0
    per_scene: bool = False
    message: str = ""


class ReanalyzeResponse(BaseModel):
    job_id: str
    status: str = "reanalyzing"
    scenes_count: int = 0


# ── Health ───────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = "ready"
    whisper_loaded: bool = False
    ollama_available: bool = False
    openrouter_key_set: bool = False
    gemini_key_set: bool = False
    groq_key_set: bool = False
    anthropic_key_set: bool = False
    disk_free_gb: float = 0.0
    active_jobs: int = 0
    version: str = "1.0.0"


# ── Pipeline ─────────────────────────────────────────────────────────

class PipelineRequest(BaseModel):
    url: str = Field(description="Video URL to process")
    language: str = Field(default="", description="ISO 639-1 language code (empty = auto-detect)")
    clip_count: int = Field(default=5, ge=1, le=50, description="Number of clips to detect")
    min_duration: float = Field(default=15, ge=1, description="Minimum clip duration in seconds")
    max_duration: float = Field(default=90, ge=1, description="Maximum clip duration in seconds")
    aspect_ratio: str = Field(default="9:16", description="Export aspect ratio: 16:9, 9:16, 1:1, 4:5")
    export_quality: str = Field(default="1080p", description="Export quality: 720p, 1080p, 4k")
    subtitles_enabled: bool = Field(default=True, description="Burn subtitles into exported clips")
    subtitle_settings: Optional[SubtitleSettings] = Field(default=None, description="Custom subtitle styling")
    auto_select: str = Field(default="top", description="'top' = export highest viral_score, 'all' = export all")
    generate_seo: bool = Field(default=True, description="Generate SEO metadata for each exported clip")
    callback_url: str = Field(default="", description="Webhook URL to POST results when pipeline completes")


class PipelineStartResponse(BaseModel):
    pipeline_id: str
    job_id: str
    status: str = "running"
    message: str = ""


class PipelineClipResult(BaseModel):
    clip_id: int
    title: str
    viral_score: int
    start_time: float
    end_time: float
    duration: float
    download_url: Optional[str] = None
    seo: Optional[ClipSEO] = None


class PipelineStatusResponse(BaseModel):
    pipeline_id: str
    job_id: str
    status: str  # "downloading", "analyzing", "detecting_clips", "exporting", "generating_seo", "complete", "failed"
    progress: int = 0
    progress_message: str = ""
    clips: list[PipelineClipResult] = []
    error: Optional[str] = None


# ── Setup ────────────────────────────────────────────────────────────

class SetupRequest(BaseModel):
    openrouter_api_key: str = Field(default="", description="OpenRouter API key")
    anthropic_api_key: str = Field(default="", description="Anthropic API key")
    gemini_api_key: str = Field(default="", description="Google Gemini API key")
    groq_api_key: str = Field(default="", description="Groq API key")
    preset: str = Field(default="balanced", description="AI preset: free, efficient, balanced, premium")
    default_export_quality: str = Field(default="1080p", description="Default export quality")
    default_aspect_ratio: str = Field(default="9:16", description="Default aspect ratio")
    auto_analyze: bool = Field(default=True, description="Auto-start analysis on upload")


class SetupResponse(BaseModel):
    status: str = "configured"
    providers_configured: list[str] = []
    message: str = ""
