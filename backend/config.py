from dataclasses import dataclass, replace
from pydantic_settings import BaseSettings
from typing import Optional
from functools import lru_cache


class Settings(BaseSettings):
    # Primary provider
    AI_PROVIDER: str = "openrouter"

    # OpenRouter (uses OpenAI-compatible API — set your key from openrouter.ai)
    OPENROUTER_API_KEY: str = ""
    OPENROUTER_PRESET: str = "balanced"  # free | efficient | balanced | premium | custom
    OPENROUTER_VISION_MODEL: str = "google/gemini-2.5-flash"
    OPENROUTER_TEXT_MODEL: str = "google/gemini-2.5-pro"
    OPENROUTER_SUMMARY_MODEL: str = "google/gemini-2.5-flash"

    # Direct provider keys
    ANTHROPIC_API_KEY: str = ""
    # Per-provider model picks — populated by the per-user settings
    # overlay (``/api/auth/me/settings``) so each user can run a
    # different model on the same backend without restarting.
    ANTHROPIC_MODEL: str = ""             # blank → provider's MODEL constant default
    GEMINI_API_KEY: str = ""
    GEMINI_TEXT_MODEL: str = ""           # blank → "gemini-2.0-flash"
    GEMINI_VIDEO_MODEL: str = ""          # blank → "gemini-2.5-flash"
    GEMINI_USE_NATIVE_VIDEO: bool = False
    GROQ_API_KEY: str = ""
    GROQ_TEXT_MODEL: str = ""             # blank → "llama-3.3-70b-versatile"

    # Ollama — defaults sized for 4GB VRAM (GTX 1650 class).
    # moondream:1.8b (~1GB) fits entirely in 4GB VRAM for fast GPU vision inference.
    # qwen2.5:3b-instruct (~1.8GB) fits in VRAM alongside moondream for GPU text inference.
    # Both together = ~3.1GB < 4GB VRAM, so both get full GPU acceleration.
    # Users with 8GB+ VRAM can switch to llava:7b + llama3.1:8b in Settings.
    OLLAMA_HOST: str = "http://ollama:11434"
    OLLAMA_VISION_MODEL: str = "moondream:1.8b"
    OLLAMA_TEXT_MODEL: str = "qwen2.5:3b-instruct"
    OLLAMA_TRANSLATION_MODEL: str = "qwen2.5:3b"  # Dedicated model for subtitle translation (multilingual)

    # Fallback chain (ollama excluded by default — user can enable it in Settings)
    AI_FALLBACK_CHAIN: str = "openrouter,gemini,groq"

    # Analysis settings
    WHISPER_MODEL: str = "small"  # Auto-upgraded to large-v3-turbo when GPU detected
    WHISPER_MODEL_USER_SET: bool = False  # True when user explicitly chose a model in UI
    WHISPER_BEAM_SIZE: int = 5    # beam search for better accuracy (was 1/greedy)
    WHISPER_VAD_FILTER: bool = True    # skip silence — major speedup
    # ── Phase 1 coverage boosters ──
    # These default to ON because they improve first-pass transcript
    # coverage on ~every content type (podcasts, interviews, vlogs,
    # cinematic, music videos, anime) with negligible quality cost.
    # All four are per-user overridable via the settings overlay so
    # users who want the legacy behavior can disable them.
    #
    # Apply FFmpeg ``highpass=80, afftdn=-25, loudnorm=-18 LUFS`` before
    # Whisper sees the audio. Adds ~3-6s to extraction time on a
    # typical clip but recovers 2-5 percentage points of coverage on
    # mixed / quiet / noisy source material — this is the single
    # biggest reason Whisper drops faint speech.
    WHISPER_AUDIO_PRECONDITION: bool = True
    # Silero VAD onset sensitivity. 0.15 (faster-whisper default) is
    # conservative and drops whispered / soft speech on realistic
    # content. 0.10 matches silero's published default and catches the
    # faint speech the 0.15 threshold was rejecting. CJK / anime paths
    # still override this to 0.08 for even more sensitivity.
    WHISPER_VAD_ONSET: float = 0.10
    # ``no_speech_threshold`` — Whisper's self-reported confidence that
    # a chunk is NOT speech. 0.8 (previous hard-coded default) threw
    # out a lot of valid quiet speech; 0.5 catches it while the worker
    # hallucination filter cleans up any false positives.
    WHISPER_NO_SPEECH_THRESHOLD: float = 0.5
    # ── Phase 2 coverage booster: recall-pass gap filling ──
    # After the main Whisper pass, compare transcript coverage against
    # an independent webrtcvad speech-presence mask. For any window >2s
    # where VAD says "speech here" but the transcript is empty, re-run
    # Whisper on just that slice with recall-first parameters
    # (vad_filter=False, no_speech_threshold=0.2, no temperature
    # fallback, no previous-text conditioning). Recovers speech that
    # Whisper's internal gates and the main-pass VAD rejected.
    # Bounded by WHISPER_GAP_FILL_MAX_GAPS and WHISPER_GAP_FILL_MAX_AUDIO_SEC.
    WHISPER_GAP_FILL_ENABLED: bool = True
    WHISPER_GAP_FILL_MIN_GAP_SEC: float = 2.0
    WHISPER_GAP_FILL_MAX_GAPS: int = 40
    WHISPER_GAP_FILL_MAX_AUDIO_SEC: float = 600.0
    # Auto-upgrade the Whisper model tier when the detected GPU has
    # spare VRAM. Existing logic already jumped ``small → large-v3-turbo``
    # on ≥6 GB cards; this flag extends the ladder so mid-tier GPUs
    # get ``small → medium`` on ≥2.5 GB as well.
    WHISPER_AUTO_UPGRADE: bool = True
    FRAME_SAMPLE_RATE: int = 10        # seconds between frames (lower=more detail, slower)
    MAX_CLIP_CANDIDATES: int = 12
    # Adaptive frame extraction
    MIN_FRAMES: int = 30               # minimum for any video
    FRAMES_PER_MINUTE: float = 6       # target density (first 30 min; diminishes for longer videos)
    CONCURRENT_ANALYSES: int = 2
    AUTO_ANALYZE: bool = True
    SUBJECT_TRACKING_ENABLED: bool = True
    DENSE_FACE_SAMPLE_RATE: float = 0.5  # seconds between dense face detection frames (0.5 = 2fps)

    # ── Scene-analysis VLM budget reallocation ─────────────────
    # When True, reorder VLM batches so semantically novel frames are
    # analyzed first. If the vision provider hits rate limits mid-job,
    # the most distinctive frames still get real descriptions.
    SCENE_NOVELTY_REWEIGHT_ENABLED: bool = False
    SCENE_NOVELTY_DISTANCE_THRESHOLD: float = 0.35

    # FFmpeg encoding settings
    FFMPEG_PRESET: str = "fast"       # ultrafast|superfast|veryfast|faster|fast|medium|slow
    FFMPEG_CRF: int = 23             # 0-51, lower=better quality, 23=default
    FFMPEG_THREADS: int = 4          # Limit threads to control memory usage (0=auto risks OOM)
    FFMPEG_FASTSTART: bool = True    # -movflags +faststart for web streaming

    # GPU Hardware Acceleration — user toggle persisted to user_settings.json
    # Auto-enabled at startup when NVIDIA GPU is detected (see main.py)
    GPU_ACCELERATION_ENABLED: bool = False   # Toggle in Settings > Advanced
    GPU_VENDOR_OVERRIDE: str = ""            # Empty = auto-detect; "nvidia", "intel", "amd", "apple" to force
    GPU_HWDECODE_ENABLED: bool = True        # Use GPU for video decoding (NVDEC/DXVA2/VAAPI/VideoToolbox)
    GPU_HEVC_FOR_4K: bool = True             # Use HEVC encoder for 4K exports when available
    GPU_DEVICE_INDEX: str = ""               # GPU device index for FFmpeg (e.g. "0", "1") — set by client GPU report

    # AI transcript post-correction
    AI_TRANSCRIPT_CORRECTION: bool = True  # Use LLM to fix proper nouns, punctuation, fillers

    # Speaker diarization (pyannote)
    DIARIZATION_ENABLED: bool = True  # Use pyannote for real speaker diarization
    DIARIZATION_MIN_SPEAKERS: int = 1
    DIARIZATION_MAX_SPEAKERS: int = 0  # 0 = unlimited (pyannote auto-detects)
    HF_AUTH_TOKEN: str = ""  # HuggingFace token for pyannote model access

    # Camera solver (per-shot AutoFlip-style crop planning)
    CLIPAI_CAMERA_SOLVER: str = "on"  # "on" | "off" — env CLIPAI_CAMERA_SOLVER

    # Content-type routing for solver tuning (off by default until tested)
    CLIPAI_CONTENT_ROUTING: str = "off"  # "on" | "off" — env CLIPAI_CONTENT_ROUTING
    CLIPAI_CONTENT_TYPES_ENABLED: str = ""  # comma-separated: "talking_head,stream" — empty = all

    # ── Cloud storage integration (Google Drive + Box) ───────────────────────
    # Feature flag — default on, flip to "false" in the Unraid template to
    # kill the entire feature if something goes wrong in production.
    CLIPAI_CLOUD_STORAGE_ENABLED: bool = True

    # Fernet key (32 url-safe base64 bytes) used to encrypt stored OAuth
    # tokens at rest. If missing, a loud warning is logged and an ephemeral
    # key is generated (which means every container restart drops stored
    # cloud sessions — set this in your Unraid template).
    CLIPAI_TOKEN_ENC_KEY: str = ""

    # Google Drive OAuth client credentials.
    GOOGLE_DRIVE_CLIENT_ID: str = ""
    GOOGLE_DRIVE_CLIENT_SECRET: str = ""
    GOOGLE_DRIVE_REDIRECT_URI: str = "http://localhost:8000/api/cloud/google_drive/callback"

    # Box OAuth client credentials.
    BOX_CLIENT_ID: str = ""
    BOX_CLIENT_SECRET: str = ""
    BOX_REDIRECT_URI: str = "http://localhost:8000/api/cloud/box/callback"

    @property
    def active_provider_chain(self) -> list[str]:
        return [p.strip() for p in self.AI_FALLBACK_CHAIN.split(",") if p.strip()]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()


# ── Duration Tier System ─────────────────────────────────────────────────


@dataclass
class VideoDurationTier:
    """Pipeline configuration that auto-adjusts based on video duration."""
    name: str
    max_minutes: float
    frame_sample_rate: int           # seconds between frame samples
    summary_strategy: str            # "single" | "map_reduce"
    summary_chunk_minutes: int       # chunk size for map_reduce (0 = N/A)
    window_duration: float           # clip detection window size in seconds
    window_overlap: float            # overlap between windows in seconds
    max_clip_candidates: int         # max clips to return
    max_gaps_pass2: int              # max coverage gaps to scan in pass 2
    hot_zone_top_n: int              # how many hot zones to send to AI
    per_call_timeout_base: int       # base timeout per AI call in seconds
    vision_batch_concurrency: int    # concurrent vision API batches


DURATION_TIERS = [
    VideoDurationTier("short",     15,   8, "single",      0,    0,    0, 12,  4, 15, 120, 2),
    VideoDurationTier("medium",    60,  12, "single",      0,  480,   90, 20,  6, 20, 180, 2),
    VideoDurationTier("long",     180,  20, "map_reduce", 10,  600,  120, 40,  8, 25, 240, 3),
    VideoDurationTier("marathon", 9999, 35, "map_reduce", 15,  600,  120, 60, 12, 30, 300, 3),
]


def get_duration_tier(duration_seconds: float) -> VideoDurationTier:
    """Return the appropriate tier configuration for a given video duration."""
    minutes = duration_seconds / 60
    for tier in DURATION_TIERS:
        if minutes <= tier.max_minutes:
            return tier
    return DURATION_TIERS[-1]


@dataclass
class OllamaTierOverrides:
    """Overrides applied when Ollama is the active provider.

    Ollama (local AI) has much smaller context windows (4K-8K tokens),
    slower inference (~12 tok/s on GTX 1650), and limited VRAM (4GB).
    Windows must be smaller, processing must be sequential, and timeouts
    must be much longer than cloud providers.
    """
    window_duration: float           # smaller windows (4-5 min) to fit context
    window_overlap: float
    per_call_timeout_base: int       # 300-480s for local 7B inference
    summary_strategy: str            # always map_reduce (context too small for single-pass)
    summary_chunk_minutes: int       # smaller chunks (5-10 min) for tiny context
    vision_batch_concurrency: int    # 1 for 4GB VRAM (never concurrent)
    max_transcript_chars: int        # per-window transcript budget (chars)
    max_scene_chars: int             # per-window scene budget (chars)
    sequential_windows: bool         # True = one window at a time (VRAM safety)


OLLAMA_TIER_OVERRIDES = {
    #                           window  overlap  timeout  summary   chunk  conc  tx_ch  sc_ch  seq
    "short":    OllamaTierOverrides(180,  20, 240, "single",       0, 1, 2500,  800, True),
    "medium":   OllamaTierOverrides(180,  30, 300, "map_reduce",   5, 1, 2000,  600, True),
    "long":     OllamaTierOverrides(180,  30, 360, "map_reduce",   5, 1, 1800,  500, True),
    "marathon": OllamaTierOverrides(180,  30, 420, "map_reduce",   5, 1, 1500,  400, True),
}

# Model recommendations based on available VRAM
OLLAMA_VRAM_PROFILES = {
    "4gb": {
        "vision": "moondream:1.8b",
        "text": "qwen2.5:3b-instruct",
        "translation": "qwen2.5:3b",
        "notes": "GTX 1650 / 4GB — fastest models that fit in VRAM",
    },
    "6gb": {
        "vision": "llava:7b-v1.6-q4_0",
        "text": "qwen2.5:7b-instruct-q4_0",
        "translation": "qwen2.5:3b",
        "notes": "RTX 2060 / 6GB — good balance of speed and quality",
    },
    "8gb+": {
        "vision": "llava:13b-v1.6-q4_0",
        "text": "llama3.1:8b-instruct-q4_0",
        "translation": "qwen2.5:7b",
        "notes": "RTX 3060+ / 8GB+ — highest quality local models",
    },
}


def apply_ollama_overrides(tier: VideoDurationTier, is_ollama: bool) -> VideoDurationTier:
    """Return a modified tier with Ollama-appropriate settings if Ollama is active."""
    if not is_ollama:
        return tier
    overrides = OLLAMA_TIER_OVERRIDES.get(tier.name, OLLAMA_TIER_OVERRIDES["medium"])
    return replace(
        tier,
        window_duration=overrides.window_duration,
        window_overlap=overrides.window_overlap,
        per_call_timeout_base=overrides.per_call_timeout_base,
        summary_strategy=overrides.summary_strategy,
        summary_chunk_minutes=overrides.summary_chunk_minutes if overrides.summary_chunk_minutes else tier.summary_chunk_minutes,
        vision_batch_concurrency=overrides.vision_batch_concurrency,
    )
