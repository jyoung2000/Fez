from enum import Enum
from typing import Optional, Union
from pydantic import BaseModel, model_validator


class JobStatus(str, Enum):
    QUEUED = "queued"
    EXTRACTING_FRAMES = "extracting_frames"
    TRANSCRIBING = "transcribing"
    ANALYZING_SCENES = "analyzing_scenes"
    GENERATING_SUMMARY = "generating_summary"
    DETECTING_CLIPS = "detecting_clips"
    COMPLETE = "complete"
    FAILED = "failed"
    CANCELLED = "cancelled"


class LayoutMode(str, Enum):
    """Video layout modes for multi-speaker reframing."""
    SINGLE = "single"          # One speaker centered (current behavior)
    SPLIT = "split"            # Two speakers side-by-side (vertical split)
    TRIPLE = "triple"          # Three speakers in grid
    PICTURE_IN_PICTURE = "pip" # Main speaker large + secondary small overlay
    SCREENSHARE = "screenshare" # Screen content top, speaker bottom
    GAMEPLAY = "gameplay"      # Gameplay top 70%, speaker bottom 30%


class FrameData(BaseModel):
    timestamp: float
    path: str
    base64: Optional[str] = None
    face_data: Optional[object] = None  # FrameFaces from face_detector (set after extraction)
    face_registry: Optional[object] = None  # FaceRegistry from face_registry module

    class Config:
        arbitrary_types_allowed = True


class SceneDescription(BaseModel):
    timestamp: float
    description: str
    importance_score: int  # 1-10
    thumbnail_path: str
    subject_x: int = 50  # 0-100, horizontal subject position (0=left, 50=center, 100=right)
    active_speaker_x: Optional[int] = None  # 0-100, position of the person who is talking (if detectable)
    layout_mode: str = "single"           # Recommended layout for this timestamp
    face_count: int = 0                   # Number of faces detected at this timestamp
    face_positions: list[dict] = []       # [{slot_id, x, y, w, h, is_speaking, identity_id}]
    has_screen_content: bool = False       # Whether frame contains screen share / slides / text
    precise_x: Optional[float] = None      # Actual face nose_x (0-100), not slot-snapped
    precise_y: Optional[float] = None      # Actual face nose_y (0-100), for vertical centering
    primary_object_x: Optional[int] = None # Non-face object tracking position (0-100)
    primary_object_type: Optional[str] = None  # "ball", "product", "hand", "text", etc.
    # ── VLM upgrade Phase 1 — grounding-output fields ──
    # New VLM prompts ask for a normalized bbox instead of a 0-100 subject_x
    # estimate. ``subject_box`` is [x1, y1, x2, y2] in 0.0-1.0 image coords;
    # ``subject_x`` / ``precise_x`` are still derived so every legacy
    # consumer keeps working unchanged. See parse_scene_dict() in
    # backend.services.providers.base for the derivation rules.
    active_slot: Optional[int] = None  # slot_id of the active speaker at this timestamp, -1 or None if none
    subject_box: Optional[list[float]] = None       # [x1, y1, x2, y2] normalized 0-1, primary subject
    subject_confidence: float = 0.0                 # 0.0-1.0, merged confidence after Phase 3 fusion
    vlm_confidence: float = 0.0                     # 0.0-1.0, raw VLM-reported confidence (Phase 3 diagnostic)
    face_confidence: float = 0.0                    # 0.0-1.0, raw face-detector confidence (Phase 3 diagnostic)
    fusion_source: Optional[str] = None             # "face_high_conf" | "blended" | "vlm_only" | "fallback_center" | None
    secondary_subjects: list[dict] = []             # [{box, confidence, label}]
    no_subject_reason: Optional[str] = None         # "empty_frame"|"abstract"|"transition"|"occluded"|None
    description_source: Optional[str] = None
    # One of: "vlm_cloud", "vlm_local", "transcript_fallback",
    # "heuristic_fallback", "user_added", or None (legacy/unknown).
    # Surfaces in the UI as a small badge. Does NOT affect is_synthetic_scene.

    @model_validator(mode='before')
    @classmethod
    def sanitize_numpy_types(cls, data):
        """Convert numpy types to native Python types before validation."""
        if isinstance(data, dict):
            for key in ('subject_x', 'active_speaker_x', 'face_count',
                        'importance_score', 'primary_object_x', 'active_slot'):
                if key in data and data[key] is not None:
                    data[key] = int(data[key])
            if 'timestamp' in data:
                data['timestamp'] = float(data['timestamp'])
            if 'face_positions' in data and data['face_positions']:
                sanitized = []
                for fp in data['face_positions']:
                    sanitized.append({
                        k: (bool(v) if k == 'is_speaking'
                            else int(v) if isinstance(v, (int, float)) and not isinstance(v, bool)
                            else v)
                        for k, v in fp.items()
                    })
                data['face_positions'] = sanitized
        return data


class WordTimestamp(BaseModel):
    start: float
    end: float
    word: str


class TranscriptSegment(BaseModel):
    start: float
    end: float
    text: str
    speaker: str  # "Speaker 1", "Speaker 2", etc.
    words: Optional[list[WordTimestamp]] = None  # per-word timestamps from Whisper
    confidence: Optional[float] = None  # 0.0-1.0, derived from avg_logprob
    avg_logprob: Optional[float] = None  # Raw Whisper log probability
    no_speech_prob: Optional[float] = None  # Probability this is not speech


class ClipCandidate(BaseModel):
    id: int
    title: str
    start_time: float
    end_time: float
    duration: float
    viral_score: int  # 1-100 — composite computed from the four axes below
    # In focus mode we preserve the LLM-reported relevance in ``viral_score``
    # and stash the genre-weighted virality composite here so the UI can
    # show both numbers. In non-focus mode this field is unused.
    viral_score_composite: Optional[int] = None
    viral_score_reasoning: str
    clip_type: str
    platform: str  # tiktok | youtube_shorts | both
    suggested_caption: str
    hook_text: str
    why_this_works: str
    clip_focus: Optional[str] = None  # The focus topic used to generate this clip, if any
    focus_relevance: Optional[int] = None  # 1-100, how relevant to the focus query
    focus_tier: Optional[str] = None  # "strong", "moderate", "weak"
    # ── Phase 1 (OpusClip parity gap) — 4-axis decomposed scoring ──
    # Each axis is 0-100. The legacy ``viral_score`` is recomputed as a
    # genre-weighted composite of these four axes (see clip_scoring.py).
    # All four default to 0 so backward-compat code paths that only set
    # viral_score continue to round-trip; the composite helper detects
    # the all-zero case and falls back to the LLM's raw viral_score.
    hook_score: int = 0
    flow_score: int = 0
    value_score: int = 0
    trend_score: int = 0
    hook_reason: str = ""
    flow_reason: str = ""
    value_reason: str = ""
    trend_reason: str = ""
    # Diagnostics — populated by composite_score() so the job report can
    # show which weights produced the final viral_score and which trend
    # lexicon entries / sentiment tags fired. Free-form dict so we can
    # evolve the schema without migrations.
    score_diagnostics: Optional[dict] = None
    # Persisted SEO data (populated by generation endpoints)
    seo_title: Optional[str] = None
    seo_description: Optional[str] = None
    seo_tags: list[str] = []
    seo_platform_tips: Optional[str] = None
    shorts_description: Optional[str] = None
    longform_description: Optional[str] = None


class VideoSummary(BaseModel):
    overview: str
    key_topics: list[str]
    tone: str
    estimated_audience: str
    content_category: str


class JobResult(BaseModel):
    job_id: str
    filename: str
    file_path: str
    # Owner of this job (``User.id``). Empty string = legacy job
    # created before multi-user auth shipped; those are visible to any
    # admin but not to regular users.
    owner_user_id: str = ""
    # Username snapshot at upload time. Used as a fallback match when
    # ``owner_user_id`` no longer resolves to a live user — for example
    # on deployments whose ``/data/auth`` directory was wiped before it
    # was persisted as a docker volume and the admin's UUID was
    # regenerated on seed. Stable identifiers (like the head admin's
    # ``Jadmin``) survive that wipe, so matching on username restores
    # visibility into the user's previously-generated clips.
    owner_username: str = ""
    # SHA-256 of the source video bytes. Populated lazily by the
    # pipeline on the first analysis. Used by re-analyze runs to skip
    # frame + audio extraction when the source hasn't changed since
    # the previous run (frames/ + audio.wav already exist on disk).
    source_sha256: str = ""
    # Per-stage wall-clock timings filled in by ``_stage_timer``. Lets
    # the UI show "where time went" without a separate metrics store.
    # Shape: ``{"frame+audio extraction": 12.3, "transcription": 84.1, ...}``
    timings: dict = {}
    # Soft warnings the pipeline emitted but recovered from. Surfaced
    # to the UI alongside the result so users can see "saliency was
    # downsampled on this codec" style messages without grepping logs.
    pipeline_warnings: list[str] = []
    language: str = ""  # ISO 639-1 code, empty = auto-detect
    subtitle_language: str = ""  # ISO 639-1 target language for subtitles, empty = same as audio
    duration: float = 0.0
    resolution: str = ""
    fps: float = 0.0
    file_size_mb: float = 0.0
    status: JobStatus = JobStatus.QUEUED
    progress: int = 0  # 0-100
    progress_message: str = ""
    provider_used: dict = {}  # {task: model_name}
    created_at: str = ""
    updated_at: str = ""
    analysis_started_at: Optional[str] = None  # ISO timestamp when analysis began
    analysis_duration_seconds: Optional[float] = None  # total wall-clock time for analysis
    summary: Optional[VideoSummary] = None
    scenes: list[SceneDescription] = []
    transcript: list[TranscriptSegment] = []
    translated_transcript: list[TranscriptSegment] = []  # Translated subtitle segments
    clips: list[ClipCandidate] = []
    speaker_names: dict[str, str] = {}  # {"Speaker 1": "Eric", "Speaker 2": "Alice"}
    scene_cut_timestamps: list[float] = []  # Timestamps of camera cuts from frame extraction
    exported_clips: list[dict] = []
    subtitle_settings: Optional[dict] = None  # Canonical subtitle settings — server is source of truth
    error: Optional[str] = None
    estimated_cost_usd: Optional[float] = None
    face_registry_data: Optional[dict] = None   # Serialized FaceRegistry with embeddings
    layout_timeline: list[dict] = []            # [{start, end, layout_mode, face_positions}]
    default_layout_mode: str = "single"         # Overall recommended layout for the video
    dense_tracking_summary: Optional[dict] = None  # {total_frames, frames_with_faces, sample_rate, ...}
    # Dense per-frame subject position track. Sampled at ~2 Hz (see
    # ``DENSE_FACE_SAMPLE_RATE``) using a confidence-ordered cascade:
    # dense face → saliency region → YOLO object → nearest VLM scene →
    # forward-fill. Lets the 9:16 / 1:1 reframe honor the subject on
    # every frame instead of interpolating between sparse VLM scenes —
    # prevents the "empty 9:16 window for half a second" artifact the
    # user hits on anime and action content where the subject lives
    # off-center of a 16:9 frame.
    # Shape: [{"t": float_seconds, "x": 0..100, "y": 0..100,
    #          "conf": 0..1, "source": "face"|"saliency"|"object"|"scene"|"hold"}]
    subject_track: list[dict] = []
    tracking_mode: str = ""  # "continuous" | "multi_cluster" | "gameplay" — set by pipeline after face analysis
    # Phase 1 + 2 — content-type override pipeline.
    # ``content_type_override`` accepts every UI dropdown value: the
    # legacy ones (gameplay / podcast / movie) PLUS the Phase 2
    # additions (debate / panel / interview / vlog / narrative /
    # cinematic / anime / cartoon / music_video / gameplay_fps /
    # gameplay_moba / gameplay_tps / gameplay_racing / stream / sports).
    # Normalized via backend.services.content_type_strings.
    content_type_override: str = ""
    # Free-form game key from the Phase 2 game sub-dropdown. Now spans
    # FPS / MOBA / TPS / racing / sandbox — see backend.services.game_layouts.
    game_type: str = ""
    # Phase 2 anime sub-dropdown: "action" | "dialogue" | "slice_of_life" | "" (auto).
    # Drives classify_clip ANIMATION vs ANIMATION_DIALOGUE routing,
    # and Phase 6's anime shot detector + lead-room multiplier.
    anime_subtype: str = ""
    # Phase 2 music-video sub-dropdown: "performance" | "narrative" | "lyric" | "" (auto).
    # Phase 5 reads this to set beat-snap aggressiveness.
    music_subtype: str = ""
    # v4 sports sub-dropdown: "basketball" | "racing" | "" (auto / generic).
    # Drives ClipContentType.SPORTS_* routing and per-sport object-tracking
    # rules (follow the ball / lead car / scoreboard preservation).
    sports_subtype: str = ""
    filler_events: list[dict] = []  # [{start, end, type, text}] from filler word detection
    emphasis_keywords: list[str] = []  # Words to highlight in captions
    thumbnail_path: Optional[str] = None  # Absolute path to OG preview thumbnail JPG
    # Legacy transcription coverage audit report from the retired
    # Silero-VAD gap-fill feature. Kept on the model so historical job
    # JSONs load without a migration; no longer produced or read by
    # the app (retranscribe replaces the previous gap-fill flow).
    coverage_report: Optional[dict] = None
    # ── Clip-detection context (persisted so the regenerate route can
    #    reuse the same inputs the initial run used). See
    #    backend/services/pipeline.py where these are populated. All
    #    default to empty so legacy jobs load unchanged.
    #
    #    hot_zones: list of HotZone-shaped dicts — {start,end,composite_score,
    #      audio_score,transcript_score,scene_score,speaker_score,signals}.
    #    chapters: list of Chapter-shaped dicts — {start,end,title,topic_keywords,
    #      speaker_turns,scene_cuts}.
    #    trend_context: pre-formatted prompt block.
    #    sentiment_timeline: pre-formatted prompt block.
    #    clip_content_type: ClipContentType.value used at detection time.
    hot_zones: list[dict] = []
    chapters: list[dict] = []
    trend_context: Optional[str] = None
    sentiment_timeline: Optional[str] = None
    clip_content_type: str = ""


class ClipSEO(BaseModel):
    title: str
    description: str
    tags: list[str] = []
    platform_tips: str = ""


class SubtitleSettings(BaseModel):
    font: str = "DM Sans"
    size: Union[str, int, float] = "medium"  # "small" | "medium" | "large" | numeric px (12-72)
    font_weight: Union[str, int] = "bold"  # "normal"|"bold" or numeric 100-900
    font_color: str = "#FFFFFF"  # default subtitle text color (hex)
    position: str = "bottom"  # "top" | "center" | "bottom"
    speaker_colors: dict[str, str] = {}  # {"Speaker 1": "#00D9FF"}
    use_speaker_colors: bool = True  # True = per-speaker colors, False = uniform font_color
    background_enabled: bool = False
    background_color: str = "#000000"
    background_opacity: int = 75  # 0-100
    background_radius: int = 0  # 0-20, border radius in px for subtitle background
    outline_color: str = "#000000"  # text outline color (hex)
    outline_opacity: int = 100  # 0-100, text outline opacity
    outline_width: int = 2  # 0-10, text outline thickness in reference pixels
    show_speaker_labels: bool = False  # show "Speaker:" prefix in subtitle text
    max_width: int = 90  # 20-100, max subtitle width as % of video width
    offset_v: int = 4  # 0-100, vertical offset from edge as % of video height
    max_words: int = 0  # 0 = disabled, 1-20 = max words per subtitle event
    active_word_enabled: bool = False  # highlight the currently spoken word
    active_word_color: str = "#FFD700"  # text color of the active word (gold)
    active_word_outline_color: str = "#000000"  # outline/stroke color of the active word
    active_word_bg_color: str = "#000000"  # background color behind the active word
    active_word_bg_opacity: int = 0  # 0-100, background opacity (0 = no background)
    active_word_bg_radius: int = 4  # 0-20, border radius of the active word background box


class SegmentSettings(BaseModel):
    start: float          # Absolute start time in seconds
    end: float            # Absolute end time in seconds
    volume: float = 1.0   # 0.0 to 2.0 gain
    muted: bool = False
    subtitles_enabled: bool = True
    subject_tracking_enabled: bool = True  # Per-segment subject tracking toggle
    speed: float = 1.0    # 0.25 to 4.0 playback speed


class VideoEffects(BaseModel):
    """Video effects applied via FFmpeg eq/hue/boxblur filters to match preview."""
    brightness: float = 0.0    # -100 to 100 (maps to FFmpeg eq brightness)
    contrast: float = 0.0      # -100 to 100 (maps to FFmpeg eq contrast)
    saturation: float = 0.0    # -100 to 100 (maps to FFmpeg eq saturation)
    blur: float = 0.0          # 0 to 20 (maps to FFmpeg boxblur)
    hue_rotate: float = 0.0    # 0 to 360 degrees (maps to FFmpeg hue)
    sepia: float = 0.0         # 0 to 100 (maps to FFmpeg colorchannelmixer)
    opacity: float = 1.0       # 0 to 1 (maps to FFmpeg colorchannelmixer alpha)
    # Video transform — position/size/rotation from Properties panel
    position_x: float = 50.0   # 0-100% (50 = centered)
    position_y: float = 50.0   # 0-100% (50 = centered)
    width: float = 100.0       # 0-200% of frame (100 = original)
    height: float = 100.0      # 0-200% of frame (100 = original)
    rotation: float = 0.0      # degrees
    fade_in: float = 0.0       # seconds
    fade_out: float = 0.0      # seconds


class TextOverlay(BaseModel):
    """Text overlay for FFmpeg drawtext filter."""
    item_id: str = ""          # timeline item ID for compositing order lookup
    text: str = ""
    x: float = 50              # position % (0-100)
    y: float = 50              # position % (0-100)
    font_size: float = 48
    font_color: str = "#FFFFFF"
    font_family: str = "sans-serif"
    font_weight: float = 400
    background_color: Optional[str] = None
    outline_width: float = 0
    outline_color: str = "#000000"
    start_time: float = 0.0    # relative to clip start
    end_time: float = 0.0
    rotation: float = 0.0
    opacity: float = 1.0
    fade_in: float = 0.0       # seconds
    fade_out: float = 0.0      # seconds
    animation: str = "none"
    text_align: str = "center"
    shadow_blur: float = 0
    shadow_color: str = "rgba(0,0,0,0.5)"
    shadow_offset_x: float = 0
    shadow_offset_y: float = 0
    bg_opacity: float = 0
    bg_padding: float = 8
    bg_radius: float = 4


class ImageOverlay(BaseModel):
    """Image overlay for FFmpeg overlay filter."""
    item_id: str = ""          # timeline item ID for compositing order lookup
    src: str = ""              # path or URL to image
    x: float = 50              # position % (0-100)
    y: float = 50              # position % (0-100)
    width: float = 30          # size % (0-100)
    height: float = 30         # size % (0-100)
    start_time: float = 0.0
    end_time: float = 0.0
    opacity: float = 1.0
    fade_in: float = 0.0       # seconds
    fade_out: float = 0.0      # seconds
    rotation: float = 0.0      # degrees (-360 to 360)
    # Visual effects from multi-track editor properties panel
    brightness: float = 0.0    # -100 to 100
    contrast: float = 0.0      # -100 to 100
    saturation: float = 0.0    # -100 to 100
    blur: float = 0.0          # 0 to 20 (px)
    hue_rotate: float = 0.0    # 0 to 360 (degrees)
    sepia: float = 0.0         # 0 to 100


class ShapeOverlay(BaseModel):
    """Shape overlay rendered as a temporary PNG and composited via FFmpeg overlay."""
    item_id: str = ""          # timeline item ID for compositing order lookup
    shape_type: str = "rectangle"  # rectangle | circle | ellipse | line | arrow
    x: float = 50              # center position % (0-100)
    y: float = 50              # center position % (0-100)
    width: float = 20          # size % (0-100)
    height: float = 20         # size % (0-100)
    fill_color: str = "#FF3B30"
    stroke_color: str = "#FFFFFF"
    stroke_width: float = 2
    corner_radius: float = 0   # for rectangles
    start_time: float = 0.0
    end_time: float = 0.0
    rotation: float = 0.0
    opacity: float = 1.0
    fade_in: float = 0.0      # seconds
    fade_out: float = 0.0     # seconds


class AudioOverlay(BaseModel):
    """Additional audio item from multi-track editor (background music, SFX)."""
    src: str = ""              # path or URL to audio file
    start_time: float = 0.0   # start on the timeline (seconds)
    end_time: float = 0.0     # end on the timeline (seconds)
    volume: float = 1.0       # 0.0 to 2.0 gain
    speed: float = 1.0        # 0.25 to 4.0 playback speed (atempo / asetrate)
    fade_in: float = 0.0      # seconds
    fade_out: float = 0.0     # seconds
    # Track-level mute / solo flag from the multi-track editor. When
    # the user mutes a track in the timeline, the export drops its
    # samples instead of mixing them in. ``False`` (default) keeps
    # legacy behavior where every audio overlay is mixed.
    audio_muted: bool = False


class ExportRequest(BaseModel):
    start: float
    end: float
    clip_id: int
    clip_title: Optional[str] = None  # optional title — used as export filename
    aspect_ratio: Optional[str] = None  # "16:9" | "9:16" | "1:1" | "4:5" | None=source
    subtitles_enabled: bool = False    # True if ANY subtitles needed (global or per-segment)
    global_subtitles_enabled: Optional[bool] = None  # Original global toggle (before segment overrides)
    subtitle_settings: Optional[SubtitleSettings] = None
    export_quality: str = "1080p"  # "720p" | "1080p" | "4k"
    # VideoEditor params — applied during FFmpeg export
    volume: float = 1.0              # 0.0 to 2.0 gain
    speed: float = 1.0               # 0.25 to 4.0 playback speed
    trim_start_offset: float = 0.0   # Seconds trimmed from clip start
    trim_end_offset: float = 0.0     # Seconds trimmed from clip end
    segments: list[SegmentSettings] = []  # Per-segment volume/subtitle overrides
    # Multi-track editor effects — applied via FFmpeg filters to match preview
    video_effects: Optional[VideoEffects] = None
    text_overlays: list[TextOverlay] = []
    image_overlays: list[ImageOverlay] = []
    shape_overlays: list[ShapeOverlay] = []
    audio_overlays: list[AudioOverlay] = []  # Additional audio items (music, SFX)
    overlay_compositing_order: list[dict] = []  # Global render order for cross-type compositing
    edited_subtitle_segments: Optional[list[TranscriptSegment]] = None  # User-edited subtitle timing from timeline
    layout_mode: str = "auto"  # "auto" | "single" | "split" | "triple" | "pip" | "screenshare"
    pip_position: str = "bottom_right"  # For PIP mode
    pip_size_pct: float = 25.0          # For PIP mode
    hook_text: str = ""  # Text overlay for the opening frame
    subject_keyframes: Optional[list[dict]] = None  # Frontend-computed [{time, x}] for export crop parity


class FullVideoExportRequest(BaseModel):
    aspect_ratio: Optional[str] = None  # "16:9" | "9:16" | "1:1" | "4:5" | None=source
    subtitles_enabled: bool = False    # True if ANY subtitles needed (global or per-segment)
    global_subtitles_enabled: Optional[bool] = None  # Original global toggle (before segment overrides)
    subtitle_settings: Optional[SubtitleSettings] = None
    export_quality: str = "1080p"  # "720p" | "1080p" | "4k"
    # VideoEditor params — applied during FFmpeg export
    volume: float = 1.0
    speed: float = 1.0
    trim_start_offset: float = 0.0
    trim_end_offset: float = 0.0
    segments: list[SegmentSettings] = []  # Per-segment volume/subtitle overrides
    # Multi-track editor effects — applied via FFmpeg filters to match preview.
    video_effects: Optional[VideoEffects] = None
    text_overlays: list[TextOverlay] = []
    image_overlays: list[ImageOverlay] = []
    shape_overlays: list[ShapeOverlay] = []
    # Multi-track audio + global compositing order — previously missing
    # from the full-video schema, which silently dropped background
    # music / SFX layers and the cross-type render order from the
    # frontend payload because Pydantic strips undefined fields and
    # ``hasattr(req, "audio_overlays")`` then returned False.
    audio_overlays: list[AudioOverlay] = []
    overlay_compositing_order: list[dict] = []
    edited_subtitle_segments: Optional[list[TranscriptSegment]] = None  # User-edited subtitle timing from timeline
    # Layout / hook fields from the editor — also missing previously,
    # so a user choosing PIP / split-screen on the full video got the
    # auto-detected layout instead of their explicit pick.
    layout_mode: str = "auto"
    pip_position: str = "bottom_right"
    pip_size_pct: float = 25.0
    hook_text: str = ""
    subject_keyframes: Optional[list[dict]] = None  # Frontend-computed [{time, x}] for export crop parity


class UpdateClipTitleRequest(BaseModel):
    title: str


class UpdateClipTimesRequest(BaseModel):
    start_time: Optional[float] = None
    end_time: Optional[float] = None


class GenerateClipsRequest(BaseModel):
    min_duration: float = 15
    max_duration: float = 600
    clip_count: Optional[int] = None  # None = use settings.MAX_CLIP_CANDIDATES
    clip_focus: Optional[str] = None  # Optional focus topic (e.g. "fighting", "cooking tips")
    viral_score_min: int = 0  # Minimum viral score (0-100), clips below are filtered out
    viral_score_max: int = 100  # Maximum viral score (0-100), clips above are filtered out
    # Focus-mode only: keep clips whose LLM-reported focus_relevance (or
    # viral_score in focus mode) is at or above this floor. 0 = disabled.
    min_relevance: int = 0
    # When true, append new clips to the existing set; when false, replace
    # all existing clips. Default True preserves the historical regenerate
    # behavior (the original /generate-clips route always merged). The
    # per-clip "Find more like this" action sends append=True; a
    # "regenerate from scratch" action can pass False.
    append: bool = True
    # If set, narrow the detection window to this [start, end] range in
    # seconds. Used by the "Find more like this" per-clip action.
    scope_start: Optional[float] = None
    scope_end: Optional[float] = None


class SettingsUpdate(BaseModel):
    ai_provider: Optional[str] = None
    openrouter_preset: Optional[str] = None
    openrouter_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    groq_api_key: Optional[str] = None
    whisper_model: Optional[str] = None
    frame_sample_rate: Optional[int] = None
    max_clip_candidates: Optional[int] = None
    fallback_chain: Optional[str] = None


class ClipPreset(BaseModel):
    id: str
    name: str
    settings: dict
    created_at: str


class SavePresetRequest(BaseModel):
    name: str
    settings: dict


class RenamePresetRequest(BaseModel):
    name: str


class TranslateRequest(BaseModel):
    source_language: str = ""  # Auto-detect from job if empty
    target_language: str       # ISO 639-1 code


# ── Multi-track Timeline Models ─────────────────────────────────────────────

class TimelineItem(BaseModel):
    id: str
    track_id: str
    type: str  # 'video', 'audio', 'image', 'subtitle'
    media_ref: Optional[str] = None
    start: float
    end: float
    trim_start: float = 0
    trim_end: Optional[float] = None
    volume: float = 1.0
    speed: float = 1.0
    opacity: float = 1.0
    position: Optional[dict] = None  # {x, y} percentages
    size: Optional[dict] = None      # {w, h} percentages
    fade_in: float = 0
    fade_out: float = 0
    subtitle_text: Optional[str] = None
    subtitle_style: Optional[dict] = None


class TimelineExportRequest(BaseModel):
    tracks: list[dict] = []
    items: list[TimelineItem] = []
    duration: float = 0
    resolution: str = "1080p"  # "720p", "1080p", "4K"
    quality: int = 23          # CRF value
    format: str = "mp4"


class EditorStateRequest(BaseModel):
    tracks: list[dict] = []
    items: list[dict] = []
    mediaLibrary: list[dict] = []
    duration: float = 0
