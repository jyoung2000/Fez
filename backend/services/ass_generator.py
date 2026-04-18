"""Generate ASS (Advanced SubStation Alpha) subtitle files from transcript segments.

ASS is used instead of SRT for FFmpeg subtitle burning because it supports:
- Per-speaker color styling
- Custom fonts and sizes
- Precise positioning (top/center/bottom)
- Margin control to prevent text from going off-screen
- Word wrap mode to prevent mid-word breaks
"""

import logging
import subprocess

from backend.models import TranscriptSegment

logger = logging.getLogger(__name__)

# ── Pillow-to-libass width calibration ──────────────────────────────
# Pillow's ImageFont.getlength() consistently overestimates text advance
# width compared to libass/FreeType rendering.  Cross-font testing across
# DM Sans, Montserrat, Inter, Roboto, Poppins, Open Sans at 20-60px shows
# Pillow overestimates by 8-15%.  This factor is multiplied into Pillow
# widths when computing BGDRAW/AWDRAW rectangle positions so that boxes
# tightly wrap the actual rendered text (matching the CSS preview).
#
# If boxes still appear too wide/narrow for a specific font, adjust this.
# Lower values = tighter boxes, higher = looser.  Valid range: 0.80-1.00.
_PILLOW_TO_LIBASS_WIDTH_RATIO = 0.92


def _resolve_font_path(font_name: str, bold: bool = False) -> str | None:
    """Resolve a font family name to a file path using fc-match."""
    try:
        style = ":style=Bold" if bold else ""
        result = subprocess.run(
            ["fc-match", f"{font_name}{style}", "--format=%{file}"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            path = result.stdout.strip()
            import os
            if os.path.isfile(path):
                return path
    except Exception:
        pass
    return None


def _measure_text(text: str, font_path: str, size_px: int) -> tuple[float, float, float, float, float] | None:
    """Measure text dimensions using Pillow.

    Returns (advance_width, font_ascent, font_descent, visual_top, visual_height)
    or None.

    - advance_width: how far the cursor moves (for text positioning/centering)
    - font_ascent/descent: font-level metrics (for Y positioning to match libass)
    - visual_top: top of visible ink relative to baseline (from getbbox)
    - visual_height: height of visible ink (tight bbox, for drawing backgrounds)
    """
    try:
        from PIL import ImageFont
        font = ImageFont.truetype(font_path, size_px)
        advance_width = font.getlength(text)
        font_ascent, font_descent = font.getmetrics()
        # getbbox() returns the tight visual bounding box of rendered glyphs.
        # This is much tighter than getmetrics() which includes space for
        # accents and diacritics that may not be present in the actual text.
        bbox = font.getbbox(text)
        if bbox:
            visual_top = bbox[1]       # top of ink relative to origin
            visual_height = bbox[3] - bbox[1]  # tight visual height
        else:
            visual_top = 0
            visual_height = font_ascent + font_descent
        return (advance_width, font_ascent, font_descent, visual_top, visual_height)
    except Exception:
        return None


def _measure_reference_height(font_path: str, size_px: int) -> tuple[float, float] | None:
    """Measure reference visual metrics using 'Hg' — a string that contains
    both a tall ascender (H) and a deep descender (g).

    Returns (ref_vis_top, ref_vis_h) or None.

    Using a fixed reference string ensures:
    - Consistent background height across all subtitle segments
    - No visual "jumping" between lines with different character sets
    - Tighter fit than font_ascent + font_descent (which includes
      space for accents and extreme descenders not present in most text)
    """
    try:
        from PIL import ImageFont
        font = ImageFont.truetype(font_path, size_px)
        bbox = font.getbbox("Hg")
        if bbox:
            return (bbox[1], bbox[3] - bbox[1])
        # Fallback: use font metrics
        asc, desc = font.getmetrics()
        return (0, asc + desc)
    except Exception:
        return None


def _ass_rounded_rect(w: int, h: int, r: int) -> str:
    """Generate ASS drawing commands for a rounded rectangle.

    Coordinates are relative to the drawing origin (top-left).
    Uses cubic bezier curves to approximate quarter circles at corners.
    """
    r = min(r, min(w, h) // 2)  # clamp radius
    if r <= 0:
        # Simple rectangle
        return f"m 0 0 l {w} 0 {w} {h} 0 {h}"
    # Bezier approximation constant for quarter circles
    k = 0.5523
    kr = round(r * k)
    return (
        f"m {r} 0 "
        f"l {w - r} 0 "
        f"b {w - r + kr} 0 {w} {r - kr} {w} {r} "
        f"l {w} {h - r} "
        f"b {w} {h - r + kr} {w - r + kr} {h} {w - r} {h} "
        f"l {r} {h} "
        f"b {r - kr} {h} 0 {h - r + kr} 0 {h - r} "
        f"l 0 {r} "
        f"b 0 {r - kr} {r - kr} 0 {r} 0"
    )


FONT_SIZE_MAP = {
    "small": 22,
    "medium": 30,
    "large": 40,
}

FONT_WEIGHT_MAP = {
    "normal": 0,
    "bold": -1,  # ASS v4+ uses -1 for bold (1 is interpreted as weight=1, ultra-thin)
}


def _normalize_font_weight(weight) -> tuple[int, int]:
    """Normalize font weight to (numeric_weight, ass_bold_flag).

    Accepts string ("normal", "bold") or numeric (100-900) weights.
    Returns (numeric_weight, ass_bold) where ass_bold is -1 (bold) or 0 (normal).
    """
    if isinstance(weight, (int, float)):
        w = int(weight)
        return w, -1 if w >= 600 else 0
    w_str = str(weight).lower().strip()
    if w_str == "bold":
        return 700, -1
    if w_str == "black":
        return 900, -1
    return 400, 0

# ASS alignment: 2=bottom-center, 5=middle-center, 8=top-center
POSITION_ALIGNMENT = {
    "bottom": 2,
    "center": 5,
    "top": 8,
}

# Reference resolution — all sizes are designed for 1920x1080
REF_W = 1920
REF_H = 1080

DEFAULT_MARGIN_H = 80
DEFAULT_MARGIN_V = {
    "bottom": 40,
    "center": 0,
    "top": 40,
}

# Safe-area limits: total margin (base + inset) per side must not consume
# more than this fraction of the frame, guaranteeing a minimum text area.
MIN_TEXT_AREA_W = 0.20  # subtitle text area >= 20% of output width
MIN_TEXT_AREA_H = 0.60  # subtitle text area >= 60% of output height

DEFAULT_SPEAKER_PALETTE = [
    "#00D9FF", "#F59E0B", "#10B981", "#A78BFA", "#EF4444", "#EC4899",
    "#06B6D4", "#8B5CF6", "#F97316", "#14B8A6", "#E879F9", "#84CC16",
    "#FB7185", "#38BDF8", "#FBBF24", "#34D399", "#C084FC", "#F472B6",
    "#22D3EE", "#A3E635", "#FB923C", "#2DD4BF", "#818CF8", "#F87171",
]


def _hex_to_ass_color(hex_color: str) -> str:
    """Convert CSS hex color (#RRGGBB) to ASS color format (&H00BBGGRR&)."""
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        return "&H00FFFFFF&"
    r = hex_color[0:2]
    g = hex_color[2:4]
    b = hex_color[4:6]
    return f"&H00{b}{g}{r}&".upper()


def _hex_to_ass_color_with_alpha(hex_color: str, opacity_pct: int) -> str:
    """Convert CSS hex color + opacity percentage to ASS color with alpha.

    ASS alpha: 00 = fully opaque, FF = fully transparent.
    opacity_pct: 0 = fully transparent, 100 = fully opaque.
    """
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        hex_color = "000000"
    r = hex_color[0:2]
    g = hex_color[2:4]
    b = hex_color[4:6]
    alpha = 255 - max(0, min(255, int(opacity_pct / 100 * 255)))
    return f"&H{alpha:02X}{b}{g}{r}&".upper()


def _format_ass_time(seconds: float) -> str:
    """Convert seconds to ASS timestamp format: H:MM:SS.cc"""
    if seconds < 0:
        seconds = 0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def _sanitize_style_name(name: str) -> str:
    """Make a speaker name safe for use as an ASS style name."""
    return name.replace(",", "_").replace("\\", "_").replace("{", "_").replace("}", "_").strip()


def _align_word_timestamps(seg_word_ts, words, clip_start, clip_end):
    """Best-effort alignment of word timestamps to (possibly edited) text.

    Returns a list of (start, end, word) tuples matching `words`, or None
    if alignment is impossible (e.g., text was completely rewritten).
    """
    if len(seg_word_ts) == len(words):
        return seg_word_ts  # Exact match — use as-is

    # If counts differ by more than 3, too divergent for positional alignment
    if abs(len(seg_word_ts) - len(words)) > 3:
        return None

    # More timestamps than words (user deleted words): sample evenly
    if len(seg_word_ts) > len(words):
        step = len(seg_word_ts) / len(words)
        return [seg_word_ts[min(int(i * step), len(seg_word_ts) - 1)] for i in range(len(words))]

    # Fewer timestamps than words (user added words): split longest-duration entries
    result = list(seg_word_ts)
    while len(result) < len(words):
        max_dur_idx = max(range(len(result)), key=lambda i: result[i][1] - result[i][0])
        ts = result[max_dur_idx]
        mid = (ts[0] + ts[1]) / 2
        result[max_dur_idx] = (ts[0], mid, ts[2])
        result.insert(max_dur_idx + 1, (mid, ts[1], ''))

    return result[:len(words)]


def split_segments_by_max_words(
    segments: list[tuple],
    max_words: int,
) -> list[tuple]:
    """Split subtitle segments so no segment exceeds max_words.

    Each input tuple is (start, end, text, speaker, words) where words is optional.
    Time is distributed proportionally by word count.
    Segments with <= max_words words are returned unchanged.
    """
    if max_words <= 0:
        return segments

    result = []
    for seg in segments:
        start, end, text, speaker = seg[0], seg[1], seg[2], seg[3]
        seg_words = seg[4] if len(seg) > 4 else None
        words = text.split()
        if len(words) <= max_words:
            result.append(seg)
            continue

        total_words = len(words)
        duration = end - start
        has_word_ts = seg_words and len(seg_words) == total_words
        current_time = start

        for i in range(0, total_words, max_words):
            chunk_words = words[i : i + max_words]
            chunk_count = len(chunk_words)
            chunk_word_ts = None

            if has_word_ts:
                # Use actual word timestamps for accurate chunk boundaries
                chunk_word_ts = seg_words[i : i + max_words]
                if chunk_word_ts:
                    last_word_end = chunk_word_ts[-1][1]  # end timestamp
                    chunk_end = last_word_end + 0.02  # 20ms padding
                else:
                    chunk_word_ts = None
                    chunk_end = current_time + duration * (chunk_count / total_words)
            else:
                # Fallback: proportional splitting
                chunk_end = current_time + duration * (chunk_count / total_words)
                if seg_words:
                    chunk_word_ts = seg_words[i : i + max_words]
                    if not chunk_word_ts:
                        chunk_word_ts = None

            # Ensure the last chunk ends exactly at the segment end
            if i + max_words >= total_words:
                chunk_end = end
            # Never exceed segment end
            chunk_end = min(chunk_end, end)

            # Skip chunks that would be too short (< 0.1s)
            if chunk_end - current_time >= 0.1:
                result.append((
                    current_time,
                    chunk_end,
                    " ".join(chunk_words),
                    speaker,
                    chunk_word_ts,
                ))
            current_time = chunk_end

    return result


def generate_ass(
    segments: list[TranscriptSegment],
    start_time: float,
    end_time: float,
    font: str = "DM Sans",
    font_size: str | int | float = "medium",
    font_weight: str | int = "bold",
    font_color: str = "#FFFFFF",
    position: str = "bottom",
    speaker_colors: dict[str, str] | None = None,
    use_speaker_colors: bool = True,
    video_width: int = 1920,
    video_height: int = 1080,
    background_enabled: bool = False,
    background_color: str = "#000000",
    background_opacity: int = 75,
    background_radius: int = 0,
    outline_color: str = "#000000",
    outline_opacity: int = 100,
    outline_width: int = 2,
    content_inset_v: int = 0,
    content_inset_h: int = 0,
    show_speaker_labels: bool = False,
    max_width_pct: int = 90,
    offset_v_pct: int = 4,
    max_words: int = 0,
    active_word_enabled: bool = False,
    active_word_color: str = "#FFD700",
    active_word_outline_color: str = "#000000",
    active_word_bg_color: str = "#000000",
    active_word_bg_opacity: int = 0,
    active_word_bg_radius: int = 4,
    hook_text: str = "",
) -> str:
    """Generate an ASS subtitle string from transcript segments within a time range.

    Segments are filtered to the clip range and timestamps are offset to start at 0.
    Each speaker gets a distinct style with their assigned color.

    content_inset_v: extra vertical margin (px) to push subtitles into the actual
    video content area when blur-background padding is present.  Only applied
    for top/bottom positions — center stays centered.

    content_inset_h: extra horizontal margin (px) to keep subtitles within the
    actual video content area when blur-background pillarboxing is present.
    Applied for all positions including center.
    """
    speaker_colors = speaker_colors or {}
    if isinstance(font_size, (int, float)):
        size_px = int(font_size)
    else:
        size_px = FONT_SIZE_MAP.get(font_size, 30)
    # alignment is set below after margin_v computation (absolute vertical positioning)
    _numeric_weight, bold_flag = _normalize_font_weight(font_weight)

    # Clamp user-configurable values
    max_width_pct = max(20, min(100, max_width_pct))
    offset_v_pct = max(0, min(100, offset_v_pct))

    # Clamp outline values
    outline_width = max(0, min(10, outline_width))
    outline_opacity = max(0, min(100, outline_opacity))

    # Coerce settings that may arrive as strings from JSON
    background_enabled = bool(background_enabled)

    # Scale proportionally to the output resolution.
    # Use min-dimension ratio so text stays at designed size for all standard
    # aspect ratios (all have min dim = 1080) and scales correctly for
    # non-standard resolutions.
    font_scale = min(video_width, video_height) / min(REF_W, REF_H)

    # Font size in the ASS file must match the frontend preview exactly.
    # The frontend computes: max(16, Math.round(basePx * fontScale)), so
    # the backend must use round() (not int/truncate) for parity.
    size_px = max(16, round(size_px * font_scale))

    # Scale outline width proportionally with font size.
    # CSS `-webkit-text-stroke: Wpx` with `paint-order: stroke fill` produces
    # W/2 visible pixels of border on each side (the fill covers the inner
    # half).  The frontend uses `scaledOlWidth * 2` as the total CSS stroke,
    # so visible per-side = scaledOlWidth = round(olWidth * fontScale).
    # ASS `\bord` specifies the border expanding outward from the glyph —
    # it IS the per-side width — so it should equal the same 1x value.
    scaled_outline_width = max(0, round(outline_width * font_scale)) if outline_width > 0 else 0

    # Horizontal margin from max_width_pct: (100% - max_width%) / 2 of output width
    margin_h = max(20, int(video_width * (100 - max_width_pct) / 100 / 2))

    # Vertical alignment + margin.
    #
    # Honor the user's ``position`` choice (top / center / bottom) so
    # the export matches what the preview shows. Previously this was
    # hard-coded to bottom-center which silently overrode every
    # top/center setting on every clip.
    #
    # ASS \an alignment quadrants:
    #     7 8 9   (top:    left | center | right)
    #     4 5 6   (middle: left | center | right)
    #     1 2 3   (bottom: left | center | right)
    # MarginV is interpreted relative to the alignment edge, so:
    #     bottom alignment → margin from bottom up
    #     top alignment    → margin from top   down
    #     middle alignment → margin is ignored (centered)
    pos = (position or "bottom").strip().lower()
    if pos == "top":
        alignment = 8
        margin_v = int(video_height * (100 - offset_v_pct) / 100)
    elif pos == "center":
        alignment = 5
        margin_v = 0
    else:  # "bottom" or unknown
        alignment = 2
        margin_v = int(video_height * offset_v_pct / 100)

    # When blur-background mode is active, push subtitles inward so they
    # sit on the actual video content rather than floating in the blur zone.
    if content_inset_h > 0:
        margin_h += content_inset_h
    if content_inset_v > 0:
        margin_v += content_inset_v

    # Safe-area cap for horizontal margin only
    max_margin_h = int(video_width * (1 - MIN_TEXT_AREA_W) / 2)
    margin_h = min(margin_h, max_margin_h)

    # Filter segments to clip range
    clip_segments = []
    for seg in segments:
        if seg.end <= start_time or seg.start >= end_time:
            continue
        clip_start = max(seg.start, start_time) - start_time
        clip_end = min(seg.end, end_time) - start_time
        if clip_end - clip_start < 0.1:
            continue
        # Carry per-word timestamps (offset to clip-relative time) for active word timing
        seg_words = None
        if hasattr(seg, "words") and seg.words:
            seg_words = [
                (w.start - start_time, w.end - start_time, w.word)
                for w in seg.words
                if w.end > (max(seg.start, start_time)) and w.start < (min(seg.end, end_time))
            ]
            if not seg_words:
                seg_words = None
        # Extend segment end to cover last word if word timestamps exceed it
        if seg_words:
            last_word_end = max(w[1] for w in seg_words)
            if last_word_end > clip_end:
                clip_end = min(last_word_end + 0.05, end_time - start_time)
        clip_segments.append((clip_start, clip_end, seg.text.strip(), seg.speaker, seg_words))

    # Apply max_words splitting.
    # ``max_words = 0`` is the explicit "no limit" sentinel the
    # frontend uses (``ClipSettingsPanel`` toggle Off), so we honor
    # it here for parity with the preview. Previously we silently
    # fell back to 14, which broke long single-line captions on
    # export. We still apply 14 as a soft safety cap for ``None``
    # / missing values so a corrupted job doesn't render an
    # entire 5-minute monologue as one ASS line.
    if max_words is None:
        effective_max_words = 14
    elif max_words > 0:
        effective_max_words = max_words
    else:
        effective_max_words = 0
    if effective_max_words > 0:
        clip_segments = split_segments_by_max_words(clip_segments, effective_max_words)

    # --- Eliminate inter-segment temporal overlap ---
    # Transcript segments (especially from Whisper) often have overlapping
    # timestamps.  When two ASS Dialogue events overlap in time, libass
    # renders BOTH simultaneously and stacks them vertically, causing the
    # visible "bouncing up and down" subtitle effect.
    #
    # Fix: sort by start time and clamp each segment's end so it never
    # exceeds the next segment's start.  This mirrors the frontend, which
    # only renders one active segment at any given playback time.
    if len(clip_segments) > 1:
        clip_segments.sort(key=lambda s: s[0])
        clamped = []
        for i, seg in enumerate(clip_segments):
            cs, ce, txt, sp = seg[0], seg[1], seg[2], seg[3]
            sw = seg[4] if len(seg) > 4 else None
            if i < len(clip_segments) - 1:
                next_start = clip_segments[i + 1][0]
                if ce > next_start:
                    ce = next_start
            if ce - cs >= 0.05:
                clamped.append((cs, ce, txt, sp, sw))
        clip_segments = clamped

    if not clip_segments:
        return ""

    # Per-speaker speech rate (words per second) for active word timing.
    # Must match the frontend computeSpeakerRates() algorithm exactly.
    speaker_rates: dict[str, float] = {}
    if active_word_enabled:
        _sp_stats: dict[str, dict] = {}
        for seg in clip_segments:
            cs, ce, tx, sp = seg[0], seg[1], seg[2], seg[3]
            wc = len(tx.split())
            dur = ce - cs
            if dur <= 0 or wc == 0:
                continue
            if sp not in _sp_stats:
                _sp_stats[sp] = {"words": 0, "time": 0.0}
            _sp_stats[sp]["words"] += wc
            _sp_stats[sp]["time"] += dur
        for sp, s in _sp_stats.items():
            speaker_rates[sp] = s["words"] / s["time"] if s["time"] > 0 else 3.0

    # Collect unique speakers and assign colors
    speakers_seen = []
    for seg in clip_segments:
        sp = seg[3]
        if sp not in speakers_seen:
            speakers_seen.append(sp)

    speaker_color_map = {}
    # When speaker colors are enabled, they override the font color picker.
    for i, sp in enumerate(speakers_seen):
        if use_speaker_colors:
            if sp in speaker_colors:
                speaker_color_map[sp] = speaker_colors[sp]
            else:
                speaker_color_map[sp] = DEFAULT_SPEAKER_PALETTE[i % len(DEFAULT_SPEAKER_PALETTE)]
        else:
            speaker_color_map[sp] = font_color or "#FFFFFF"

    # Build ASS header
    lines = [
        "[Script Info]",
        "Title: ClipAI Subtitles",
        "ScriptType: v4.00+",
        f"PlayResX: {video_width}",
        f"PlayResY: {video_height}",
        "WrapStyle: 1",
        "ScaledBorderAndShadow: no",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, "
        "Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
    ]

    # Rounded-corner background: when background_radius > 0, use ASS drawing
    # commands (\p1) to render a rounded rectangle BELOW the text instead of
    # BorderStyle=3 (which only renders sharp rectangles).
    _bg_split = bool(background_enabled and background_radius > 0)
    _bg_font_path = None
    _bg_draw_pad_h = 0
    _bg_draw_pad_v = 0
    _bg_draw_radius = 0
    _bg_ref_vis_top = 0
    _bg_ref_vis_h = 0
    if _bg_split:
        _bg_font_path = _resolve_font_path(font, bold=(bold_flag == -1))
        if not _bg_font_path:
            logger.warning("Cannot resolve font path for '%s' — "
                           "falling back to sharp background box", font)
            _bg_split = False
        else:
            # Measure reference visual height once for consistent backgrounds
            ref = _measure_reference_height(_bg_font_path, size_px)
            if ref:
                _bg_ref_vis_top, _bg_ref_vis_h = ref
            else:
                # Fallback: use font metrics (less tight but still works)
                fallback_m = _measure_text("Hg", _bg_font_path, size_px)
                if fallback_m:
                    _bg_ref_vis_top = fallback_m[3]
                    _bg_ref_vis_h = fallback_m[4]
                else:
                    _bg_split = False

            if _bg_split:
                # Match frontend CSS padding: max(1, max(floor(4 * backendFontScale), 2) * subtitleScale)px
                # At output resolution (subtitleScale=1, backendFontScale=font_scale):
                # padding = max(1, max(floor(4 * font_scale), 2)) — UNIFORM in CSS
                _css_pad = max(1, max(int(4 * font_scale), 2))
                _bg_draw_pad_h = _css_pad
                _bg_draw_pad_v = max(1, round(_css_pad * 0.6))  # Slightly tighter vertical for inline feel
                _bg_draw_radius = max(1, round(background_radius * font_scale))

    # Compute outline/background style settings (same for all speakers)
    if background_enabled and not _bg_split:
        # BorderStyle=3: OutlineColour = box color, BackColour = shadow behind box.
        # With Shadow=0, BackColour should be invisible but some libass builds
        # still composite it at offset (0,0), doubling the box opacity.
        # Fix: keep BackColour fully transparent so only OutlineColour renders.
        style_outline_color = _hex_to_ass_color_with_alpha(background_color, background_opacity)
        back_color_ass = "&HFF000000&"  # fully transparent — no shadow needed
        border_style = 3
        # Match frontend CSS uniform padding: max(1, max(floor(4 * fontScale), 2))
        ol_width = max(2, int(4 * font_scale))  # BorderStyle=3 padding
        shadow_depth = 0
    elif _bg_split:
        # Background via drawing commands: base style has no box.
        # OutlineColour = transparent, BackColour = transparent.
        style_outline_color = "&HFF000000&"  # fully transparent
        back_color_ass = "&HFF000000&"  # fully transparent
        border_style = 1
        ol_width = 0
        shadow_depth = 0
    else:
        style_outline_color = _hex_to_ass_color_with_alpha(outline_color, outline_opacity)
        back_color_ass = "&H80000000&"  # shadow color (semi-transparent black)
        border_style = 1
        ol_width = scaled_outline_width
        # Shadow depth matches frontend: proportional to outline width
        # (backendOlWidth in ClipPreview.jsx:622).
        shadow_depth = max(1, min(4, round(scaled_outline_width * 0.75))) if scaled_outline_width > 0 else 0

    # With the two-layer architecture for active word mode, shadow is rendered
    # correctly on Layer 0 (uniform color, no \c overrides → continuous shadow).
    # No need to suppress shadow in the Style definition.
    style_shadow = shadow_depth

    # Create a style per speaker
    for sp in speakers_seen:
        color_hex = speaker_color_map[sp]
        ass_color = _hex_to_ass_color(color_hex)
        style_name = _sanitize_style_name(sp)

        lines.append(
            f"Style: {style_name},{font},{size_px},{ass_color},&H000000FF&,{style_outline_color},{back_color_ass},"
            f"{bold_flag},0,0,0,100,100,0,0,{border_style},{ol_width},{style_shadow},{alignment},{margin_h},{margin_h},{margin_v},1"
        )

    # When background is enabled AND outline_width > 0, create an outline
    # overlay style (_OL) that renders text outline on top of the box.
    # ASS BorderStyle=3 hijacks OutlineColour for box fill, so a separate
    # BorderStyle=1 layer is the only way to show both box AND text outline.
    bg_has_outline = background_enabled and scaled_outline_width > 0
    if bg_has_outline:
        ol_ass_color = _hex_to_ass_color_with_alpha(outline_color, outline_opacity)
        for sp in speakers_seen:
            color_hex = speaker_color_map[sp]
            ass_color = _hex_to_ass_color(color_hex)
            sn = _sanitize_style_name(sp)
            # _OL style: BorderStyle=1 for text outline, transparent background
            lines.append(
                f"Style: {sn}_OL,{font},{size_px},{ass_color},&H000000FF&,{ol_ass_color},&HFF000000&,"
                f"{bold_flag},0,0,0,100,100,0,0,1,{scaled_outline_width},0,{alignment},{margin_h},{margin_h},{margin_v},1"
            )

    # When background (BorderStyle=3) + active word highlighting are both
    # enabled, per-word \c color overrides cause libass to segment the
    # background box at every tag boundary — each word gets its own box,
    # and adjacent boxes overlap creating visible black bar borders.
    #
    # Fix: create a separate ASS style for the color overlay layer (Layer 1)
    # that uses BorderStyle=1 with Outline=0 and Shadow=0.  This style
    # renders ONLY the colored text fill with zero box/border/shadow.
    # BorderStyle is a STYLE-LEVEL setting that CANNOT be overridden by
    # inline tags, so a separate style is the only robust solution.
    if active_word_enabled:
        for sp in speakers_seen:
            color_hex = speaker_color_map[sp]
            ass_color = _hex_to_ass_color(color_hex)
            sn = _sanitize_style_name(sp)
            # _AW style: identical layout (font, size, bold, alignment, margins)
            # but BorderStyle=1, Outline=0, Shadow=0, transparent OutlineColour
            # and BackColour.  Renders text fill only — no box, no border.
            lines.append(
                f"Style: {sn}_AW,{font},{size_px},{ass_color},&H000000FF&,&HFF000000&,&HFF000000&,"
                f"{bold_flag},0,0,0,100,100,0,0,1,0,0,{alignment},{margin_h},{margin_h},{margin_v},1"
            )

    # Active word background uses BorderStyle=3 (ASS native box) which
    # guarantees pixel-perfect alignment because the same engine (libass)
    # renders both the text and the box.
    # Match frontend CSS: padding = max(1, 2*subtitleScale)px horizontal, max(1, 1*subtitleScale)px vertical
    # BorderStyle=3 Outline is uniform padding, so use the larger (horizontal) value
    # Frontend uses 2px base * subtitleScale; backend equivalent is 2 * font_scale
    # Match frontend CSS: padding 1px vertical / 2px horizontal.
    # ASS BorderStyle=3 Outline is uniform, so use 2px (closer to the
    # larger horizontal value — the vertical will be slightly thicker
    # than CSS but prevents the box from clipping glyph strokes).
    _aw_bg_box_padding = max(1, round(2 * font_scale))

    # Rounded corners: ASS BorderStyle=3 only renders sharp rectangles.
    # For rounded corners we use ASS drawing commands (\p1) to render a
    # rounded rectangle on a layer BELOW the text.  This requires Pillow
    # font measurements to position the drawing at the word's location.
    # If the font can't be resolved, we fall back to sharp BorderStyle=3.
    _aw_bg_split = False  # Always use BorderStyle=3 for pixel-perfect active word boxes

    if active_word_enabled and active_word_bg_opacity > 0:
        # OutlineColour = active word bg color (used as box fill in BorderStyle=3)
        # Default to transparent so non-active words show no box
        aw_bg_style_ol = "&HFF000000&"  # fully transparent by default (per-word \3c\3a overrides)
        for sp in speakers_seen:
            color_hex = speaker_color_map[sp]
            ass_color = _hex_to_ass_color(color_hex)
            sn = _sanitize_style_name(sp)
            lines.append(
                f"Style: {sn}_AWBG,{font},{size_px},{ass_color},&H000000FF&,{aw_bg_style_ol},&HFF000000&,"
                f"{bold_flag},0,0,0,100,100,0,0,3,{_aw_bg_box_padding},0,{alignment},{margin_h},{margin_h},{margin_v},1"
            )

    # BGDRAW style for rounded-rect background drawing events.
    # Same approach as AWDRAW: font_size=64 → 1:1 pixel mapping, alignment=7.
    if _bg_split:
        lines.append(
            "Style: BGDRAW,Arial,64,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,"
            "0,0,0,0,100,100,0,0,1,0,0,7,0,0,0,1"
        )

    # ── Hook text overlay style ──
    if hook_text and len(hook_text) > 3:
        hook_size = int(size_px * 1.4)  # Larger than subtitles
        lines.append(
            f"Style: Hook,{font},{hook_size},&H00FFFFFF,&H000000FF,&H00000000,&H80000000,"
            f"1,0,0,0,100,100,0,0,1,3,0,5,30,30,30,1"
        )

    lines.append("")
    lines.append("[Events]")
    lines.append("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text")

    # ── Hook text event (first 2.5 seconds) ──
    if hook_text and len(hook_text) > 3:
        hook_escaped = hook_text.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")
        lines.append(
            f"Dialogue: 0,0:00:00.00,0:00:02.50,Hook,,0,0,0,,"
            f"{{\\an5\\fad(300,500)}}{hook_escaped}"
        )

    # Pre-compute active word ASS colors (defaults for when active word is off)
    aw_bg = None
    aw_bg_color = None
    aw_bg_alpha = None
    if active_word_enabled:
        aw_color = _hex_to_ass_color(active_word_color)
        # Honor the user's per-spoken-word outline color when set —
        # falls back to the line outline color so existing exports
        # without an explicit override stay unchanged.
        aw_outline_hex = (active_word_outline_color or "").strip()
        if aw_outline_hex:
            aw_outline = _hex_to_ass_color(aw_outline_hex)
        else:
            aw_outline = style_outline_color
        aw_bg = None
        aw_bg_color = None  # ASS color for \3c (box fill in BorderStyle=3)
        aw_bg_alpha = None  # ASS alpha for \3a (box opacity in BorderStyle=3)
        if active_word_bg_opacity > 0:
            aw_bg = _hex_to_ass_color_with_alpha(
                active_word_bg_color, active_word_bg_opacity
            )
            # Separate color (&H00BBGGRR&) and alpha (&HAA&) for \3c and \3a tags
            aw_bg_color = _hex_to_ass_color(active_word_bg_color)
            alpha_byte = 255 - max(0, min(255, int(active_word_bg_opacity / 100 * 255)))
            aw_bg_alpha = f"&H{alpha_byte:02X}&"

    # Explicit border override tags for every Dialogue event.
    # Even though the Style already sets Outline, some libass builds and
    # FFmpeg versions lose the outline through style caching / fallback.
    # Prepending explicit \bord + \3c + \shad tags per-event guarantees
    # the outline is always rendered in the exported video.
    # Always emit even when ol_width==0 (\bord0\shad0) to prevent libass
    # from inheriting unexpected border state from the Style definition.
    if not background_enabled:
        bord_tag = f"\\bord{ol_width}\\shad{shadow_depth}\\3c{style_outline_color}"
        # Layer 1 tag for active word two-layer architecture:
        # No border, no shadow, transparent outline — renders ONLY text fill.
        # \3a&HFF& makes OutlineColour fully transparent so no outline bleeds.
        aw_nobord_tag = "\\bord0\\shad0\\3a&HFF&"
    else:
        bord_tag = ""
        # Background mode: Layer 1 uses the _AW style (BorderStyle=1) so
        # the box is disabled at the style level.  These inline tags are
        # belt-and-suspenders: \bord0 = no padding, \shad0 = no shadow,
        # \3a&HFF& = transparent outline, \4a&HFF& = transparent back color.
        aw_nobord_tag = "\\bord0\\shad0\\3a&HFF&\\4a&HFF&"

    # When active word background is enabled, Layer 1 uses _AWBG
    # (BorderStyle=3) where \3a controls box visibility per word.
    if aw_bg:
        # \shad0 = no shadow, \3a&HFF& = transparent box by default
        aw_nobord_tag = "\\shad0\\3a&HFF&"

    # Style suffix for Layer 1 events:
    #   _AWBG: active word background (BorderStyle=3 per-word box)
    #   _AW:   background mode (text fill only, no box)
    #   "":    outline mode without aw_bg (same base style)
    if aw_bg:
        aw_style_suffix = "_AWBG"
    elif background_enabled and active_word_enabled:
        aw_style_suffix = "_AW"
    else:
        aw_style_suffix = ""

    # Add dialogue events
    # Two-layer architecture for active-word mode:
    #   Layer 0 = base text for each segment (always visible, full duration)
    #   Layer 1 = per-word highlight events overlaid on top
    # This guarantees subtitles never disappear between words or during gaps.
    #
    # For standard mode: single-layer events on Layer 0 as before.
    pending_word_events: list[tuple[float, float, str, str]] = []
    pending_box_events: list[tuple[float, float, str, str]] = []
    pending_bg_draw_events: list[tuple[float, float, str, str]] = []  # BGDRAW events
    base_text_events: list[tuple[float, float, str, str]] = []

    for seg in clip_segments:
        clip_start, clip_end, text, speaker = seg[0], seg[1], seg[2], seg[3]
        seg_word_ts = seg[4] if len(seg) > 4 else None
        style_name = _sanitize_style_name(speaker)
        safe_text = text.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")

        if active_word_enabled:
            prefix = f"{speaker}: " if show_speaker_labels and speaker else ""

            # ── Two-layer architecture for active word highlighting ──
            #
            # The root cause of "black bars" is that ASS/libass segments
            # border rendering at every \c (PrimaryColour) override tag.
            # When two bordered text runs meet, their border rectangles
            # overlap at the junction, doubling the border opacity and
            # creating visible dark seams.
            #
            # The fix uses two ASS layers that composite together:
            #
            #   Layer 0 (border layer): Full text with uniform color
            #     (no \c overrides) + \bord + \shad.  Because there are
            #     no color changes, libass treats the entire line as ONE
            #     text run with a single continuous border/shadow — no
            #     seams possible.
            #
            #   Layer 1 (color layer): Full text with per-word \c color
            #     overrides + \bord0\shad0\3a&HFF& (no border, no shadow,
            #     transparent outline).  This renders ONLY the colored
            #     text fill on top of Layer 0's border.
            #
            # This mirrors how CSS renders subtitles in the browser
            # preview: -webkit-text-stroke is a single layer behind all
            # text, and inline <span style="color:gold"> only changes
            # the fill color on top.

            words = safe_text.split()
            if len(words) <= 1:
                # Single word — no color transitions, so no segmentation
                # issue.  Just use a single layer with active word color.
                if background_enabled:
                    # Background mode: Layer 0 = box, Layer 1 = colored text
                    base_text_events.append((clip_start, clip_end, style_name, f"{prefix}{safe_text}"))
                    nobord_prefix = "{" + aw_nobord_tag + "}" if aw_nobord_tag else ""
                    aw_tags = f"\\c{aw_color}"
                    if aw_bg_color:
                        aw_tags += f"\\3c{aw_bg_color}\\3a{aw_bg_alpha}"
                    event_text = f"{nobord_prefix}{prefix}{{{aw_tags}}}{safe_text}"
                    pending_word_events.append((clip_start, clip_end, style_name + aw_style_suffix, event_text))
                else:
                    # Layer 0: border layer with uniform color
                    bord_prefix = f"{{{bord_tag}}}" if bord_tag else ""
                    border_event_text = f"{bord_prefix}{prefix}{safe_text}"
                    base_text_events.append((clip_start, clip_end, style_name, border_event_text))
                    # Layer 1: color layer with \bord0 (no duplicate borders)
                    nobord_prefix = "{" + aw_nobord_tag + "}" if aw_nobord_tag else ""
                    aw_tags = f"\\c{aw_color}"
                    if aw_bg_color:
                        aw_tags += f"\\3c{aw_bg_color}\\3a{aw_bg_alpha}"
                    event_text = f"{nobord_prefix}{prefix}{{{aw_tags}}}{safe_text}"
                    pending_word_events.append((clip_start, clip_end, style_name + aw_style_suffix, event_text))
            elif seg_word_ts and len(seg_word_ts) > 0 and len(words) > 0 and _align_word_timestamps(seg_word_ts, words, clip_start, clip_end) is not None:
                # Real per-word timestamps from Whisper (possibly aligned
                # after minor user edits to subtitle text).
                seg_word_ts = _align_word_timestamps(seg_word_ts, words, clip_start, clip_end)
                # Match the preview's per-speaker rate-scaled anticipation
                # so the highlight crossfade lands at the same audio
                # sample whether you watch the preview, the server-rendered
                # MP4, or the (eventual) browser export. Preview formula
                # in SubtitleOverlay.getCurrentWordIndex (word-ts branch):
                #     adjusted = baseT + (anticipation * rateScale) - _AUDIO_BUFFER_S
                # where ``rateScale = clamp(0.6, 1.6, 3.0 / speakerWps)``
                # and ``anticipation = _ANTICIPATION_S``.
                speaker_wps = speaker_rates.get(speaker, 3.0) if speaker_rates else 3.0
                try:
                    rate_scale = max(0.6, min(1.6, 3.0 / float(speaker_wps)))
                except (TypeError, ValueError, ZeroDivisionError):
                    rate_scale = 1.0
                # Constants must mirror the preview's
                # ``SubtitleOverlay.getCurrentWordIndex`` (lines ~58-65)
                # so we don't ship a different lead/lag in export.
                _ANTICIPATION_LOCAL = 0.10
                _AUDIO_BUFFER_LOCAL = 0.12
                _WORD_ANTICIPATION_S = max(
                    0.0, _ANTICIPATION_LOCAL * rate_scale - _AUDIO_BUFFER_LOCAL,
                )
                base_color = _hex_to_ass_color(speaker_color_map[speaker])

                # Background mode: ONE Layer 0 event per segment = one
                # continuous background box.  Per-word Layer 0 events are
                # only needed for outline mode (BorderStyle=1) where each
                # per-word event carries its own \bord tags.
                # Compute extended end to cover last word's full duration
                last_w = seg_word_ts[-1] if seg_word_ts else None
                extended_seg_end = clip_end
                if last_w:
                    natural_last = last_w[1] - _WORD_ANTICIPATION_S
                    extended_seg_end = min(max(clip_end, natural_last + 0.3), clip_end + 0.5)
                if background_enabled:
                    base_text_events.append((clip_start, extended_seg_end, style_name, f"{prefix}{safe_text}"))

                for word_idx in range(len(words)):
                    w_start, w_end, _ = seg_word_ts[word_idx]
                    w_start = max(w_start - _WORD_ANTICIPATION_S, clip_start)
                    w_end = max(w_end - _WORD_ANTICIPATION_S, w_start + 0.01)
                    if word_idx == len(words) - 1:
                        # Last word: use natural end + 0.3s grace, allow up to
                        # 0.5s past clip_end so highlighting isn't cut short
                        natural_end = seg_word_ts[word_idx][1] - _WORD_ANTICIPATION_S
                        w_end = min(max(w_end, natural_end) + 0.3, clip_end + 0.5)
                    else:
                        w_end = min(w_end, clip_end)
                    if w_end - w_start < 0.01:
                        continue

                    # Layer 0: Border layer — full text, uniform color,
                    # WITH border/shadow.  No \c overrides → seamless border.
                    # Skipped for background mode (single per-segment event above).
                    if not background_enabled:
                        border_prefix = f"{{{bord_tag}}}" if bord_tag else ""
                        border_event_text = f"{border_prefix}{prefix}{safe_text}"
                        base_text_events.append((w_start, w_end, style_name, border_event_text))

                    # Layer 1: Color layer — full text, per-word colors,
                    # NO border/box.  In outline mode: \bord0\shad0\3a&HFF&.
                    # In background mode: _AW style (BorderStyle=1, no box).
                    before = " ".join(words[:word_idx])
                    active = words[word_idx]
                    after = " ".join(words[word_idx + 1:])
                    nobord_prefix = "{" + aw_nobord_tag + "}" if aw_nobord_tag else ""
                    # base_tag: non-active word color + transparent box
                    if aw_bg_color:
                        base_tag = "{" + f"\\c{base_color}\\3a&HFF&" + "}"
                    else:
                        base_tag = "{" + f"\\c{base_color}" + "}"
                    # aw_tag: active word color + visible box. When the
                    # user supplied a per-word outline override AND no
                    # background box is active, override the outline
                    # color + alpha for just this run so the active
                    # word renders with its own border on top of the
                    # uniform Layer-0 stroke.
                    aw_extra = f"\\c{aw_color}"
                    if aw_bg_color:
                        aw_extra += f"\\3c{aw_bg_color}\\3a{aw_bg_alpha}"
                    elif aw_outline and aw_outline != style_outline_color:
                        aw_extra += f"\\3c{aw_outline}\\3a&H00&\\bord{ol_width}"
                    aw_tag = "{" + aw_extra + "}"
                    # reset_tag: after the active word, hide the box again
                    if aw_bg_color:
                        reset_tag = "{" + f"\\c{base_color}\\3a&HFF&" + "}"
                    else:
                        reset_tag = base_tag
                    parts = []
                    if before:
                        parts.append(f"{base_tag}{before} ")
                    parts.append(f"{aw_tag}{active}")
                    if after:
                        parts.append(f"{reset_tag} {after}")
                    color_event_text = nobord_prefix + prefix + "".join(parts)
                    pending_word_events.append((w_start, w_end, style_name + aw_style_suffix, color_event_text))
            else:
                # Fallback: character-proportional estimation with natural
                # speech rhythm.  Uses punctuation-aware, speaker-rate-scaled
                # timing that matches the frontend getCurrentWordIndex().
                _BASE_OVERHEAD_S = 0.04
                # Match frontend timing exactly so export word highlights
                # align with what the user previewed.  The frontend uses
                # anticipation=0.10 and subtracts audio_buffer=0.12 for a
                # net -0.02s offset.  Since FFmpeg has no audio buffer lag,
                # we apply the net offset when shifting word boundaries below.
                _ANTICIPATION_S = 0.10
                _AUDIO_BUFFER_S = 0.12
                _PUNCT_PAUSE = {
                    ",": 0.15, ";": 0.16, ":": 0.12,
                    ".": 0.22, "!": 0.22, "?": 0.24,
                    "\u2014": 0.12, "\u2013": 0.10,
                }
                _FAST_WORDS = frozenset({
                    "the", "a", "an", "to", "in", "on", "at", "of", "for",
                    "and", "but", "or", "is", "was", "are", "were", "it",
                    "its", "this", "that",
                })
                base_color = _hex_to_ass_color(speaker_color_map[speaker])
                total_chars = sum(len(w) for w in words)
                if total_chars == 0:
                    total_chars = 1
                duration = clip_end - clip_start

                # Per-speaker speech rate scaling
                speaker_wps = speaker_rates.get(speaker, 3.0) if speaker_rates else 3.0
                rate_scale = max(0.6, min(1.6, 3.0 / speaker_wps))
                anticipation = _ANTICIPATION_S * rate_scale

                # Punctuation pauses per word
                punct_pauses = []
                for w in words:
                    last_char = w[-1] if w else ""
                    punct_pauses.append(_PUNCT_PAUSE.get(last_char, 0.0) * rate_scale)
                total_punct = sum(punct_pauses)

                # Base overhead + punctuation
                base_overhead = _BASE_OVERHEAD_S * rate_scale * len(words)
                total_pause = base_overhead + total_punct

                # Remaining time is character-proportional
                char_time = max(duration - total_pause, duration * 0.45)
                pause_scale = (duration - char_time) / max(total_pause, 0.01)

                # Compute raw durations with natural-speech adjustments,
                # then normalize to fit exactly within the segment.
                raw_durations = []
                for word_idx in range(len(words)):
                    char_dur = char_time * (len(words[word_idx]) / total_chars)
                    pause = (_BASE_OVERHEAD_S * rate_scale + punct_pauses[word_idx]) * pause_scale
                    word_dur = char_dur + pause
                    stripped = words[word_idx].lower().rstrip(".,!?;:\u2014\u2013")
                    if stripped in _FAST_WORDS:
                        word_dur *= 0.75
                    if word_idx == 0:
                        word_dur *= 1.15
                    elif word_idx == len(words) - 1:
                        word_dur *= 1.10
                    raw_durations.append(word_dur)

                # Normalize so total exactly matches segment duration
                total_raw = sum(raw_durations)
                if total_raw > 0:
                    norm = duration / total_raw
                    raw_durations = [d * norm for d in raw_durations]

                # Background mode: ONE Layer 0 event per segment (continuous box)
                if background_enabled:
                    base_text_events.append((clip_start, clip_end, style_name, f"{prefix}{safe_text}"))

                # Apply net timing offset to match frontend's perception model.
                # Frontend: elapsed = (t - start) + anticipation*rateScale - 0.12
                # This shifts word boundaries so the highlight at any given
                # playback time matches the preview.
                _net_offset = anticipation - _AUDIO_BUFFER_S
                current_time = clip_start + _net_offset
                for word_idx in range(len(words)):
                    word_dur = raw_durations[word_idx]
                    word_end = current_time + word_dur
                    if word_idx == len(words) - 1:
                        word_end = min(current_time + word_dur + 0.3, clip_end)
                    if word_end - current_time < 0.01:
                        current_time = word_end
                        continue

                    shifted_start = max(clip_start, current_time - anticipation)

                    # Layer 0: Border layer — uniform color, continuous outline
                    # Skipped for background mode (single per-segment event above).
                    if not background_enabled:
                        border_prefix = f"{{{bord_tag}}}" if bord_tag else ""
                        border_event_text = f"{border_prefix}{prefix}{safe_text}"
                        base_text_events.append((shifted_start, word_end, style_name, border_event_text))

                    # Layer 1: Color layer — per-word coloring, no border/box.
                    before = " ".join(words[:word_idx])
                    active = words[word_idx]
                    after = " ".join(words[word_idx + 1:])
                    nobord_prefix = "{" + aw_nobord_tag + "}" if aw_nobord_tag else ""
                    # base_tag: non-active word color + transparent box
                    if aw_bg_color:
                        base_tag = "{" + f"\\c{base_color}\\3a&HFF&" + "}"
                    else:
                        base_tag = "{" + f"\\c{base_color}" + "}"
                    # aw_tag: active word color + visible box. When the
                    # user supplied a per-word outline override AND no
                    # background box is active, override the outline
                    # color + alpha for just this run so the active
                    # word renders with its own border on top of the
                    # uniform Layer-0 stroke.
                    aw_extra = f"\\c{aw_color}"
                    if aw_bg_color:
                        aw_extra += f"\\3c{aw_bg_color}\\3a{aw_bg_alpha}"
                    elif aw_outline and aw_outline != style_outline_color:
                        aw_extra += f"\\3c{aw_outline}\\3a&H00&\\bord{ol_width}"
                    aw_tag = "{" + aw_extra + "}"
                    # reset_tag: after the active word, hide the box again
                    if aw_bg_color:
                        reset_tag = "{" + f"\\c{base_color}\\3a&HFF&" + "}"
                    else:
                        reset_tag = base_tag
                    parts = []
                    if before:
                        parts.append(f"{base_tag}{before} ")
                    parts.append(f"{aw_tag}{active}")
                    if after:
                        parts.append(f"{reset_tag} {after}")
                    color_event_text = nobord_prefix + prefix + "".join(parts)
                    pending_word_events.append((shifted_start, word_end, style_name + aw_style_suffix, color_event_text))
                    current_time = word_end
        else:
            # Standard: single event with plain text + explicit outline override
            if show_speaker_labels and speaker:
                safe_text = f"{speaker}: {safe_text}"
            bord_override = f"{{{bord_tag}}}" if bord_tag else ""
            base_text_events.append((clip_start, clip_end, style_name, f"{bord_override}{safe_text}"))


    # --- Layer 0: base text events ---
    # Fill small inter-segment gaps so text never disappears briefly.
    if base_text_events:
        base_text_events.sort(key=lambda e: e[0])
        for i in range(len(base_text_events) - 1):
            ev_start, ev_end, ev_style, ev_text = base_text_events[i]
            next_start = base_text_events[i + 1][0]
            if ev_end > next_start:
                # Overlap: clamp to eliminate bouncing on Layer 0
                base_text_events[i] = (ev_start, next_start, ev_style, ev_text)
            elif next_start - ev_end < 0.15:
                # Small gap: extend to fill — keeps text visible between segments
                base_text_events[i] = (ev_start, next_start, ev_style, ev_text)

    # --- Per-word color events (active word mode, Layer 1) ---
    # Eliminate temporal overlap AND fill small gaps between word events.
    # Overlap: when multiple word events overlap in time, libass renders
    # them all simultaneously and stacks them vertically ("bouncing").
    # Gaps: short gaps should be filled to keep the last highlighted word
    # visible until the next word starts. But long gaps (> 0.5s, e.g.
    # between segments during pauses) should NOT be filled — letting the
    # subtitle disappear during natural pauses looks more human-edited
    # than keeping a stale word highlighted for seconds.
    if pending_word_events:
        pending_word_events.sort(key=lambda e: e[0])
        _MAX_GAP_FILL_S = 0.5
        for i in range(len(pending_word_events) - 1):
            ev_start, ev_end, ev_style, ev_text = pending_word_events[i]
            next_start = pending_word_events[i + 1][0]
            if ev_end >= next_start:
                # Overlap: clamp to eliminate bouncing
                pending_word_events[i] = (ev_start, next_start, ev_style, ev_text)
            elif next_start - ev_end <= _MAX_GAP_FILL_S:
                # Small gap: extend to fill
                pending_word_events[i] = (ev_start, next_start, ev_style, ev_text)
            # Long gaps (> 0.5s): let subtitle disappear during pauses

    # ── Generate rounded-rect background drawing events (BGDRAW) ──
    # When _bg_split is active, each base text event gets a corresponding
    # BGDRAW event with a rounded rectangle behind the text.
    if _bg_split and _bg_font_path and base_text_events:
        import re as _re
        _bg_color = _hex_to_ass_color(background_color)
        _bg_alpha_byte = 255 - max(0, min(255, int(background_opacity / 100 * 255)))
        _bg_alpha = f"&H{_bg_alpha_byte:02X}&"

        for ev_start, ev_end, ev_style, ev_text in base_text_events:
            # Strip ASS override tags to get plain text for measurement
            plain = _re.sub(r"\{[^}]*\}", "", ev_text)
            if not plain.strip():
                continue
            m = _measure_text(plain, _bg_font_path, size_px)
            if not m:
                continue
            text_w = m[0] * _PILLOW_TO_LIBASS_WIDTH_RATIO
            text_asc = m[1]
            text_desc = m[2]
            line_h = text_asc + text_desc  # Full font cell (for libass Y positioning)

            # Horizontal: center text within margin-bounded area
            text_area_w = video_width - 2 * margin_h
            line_left_x = margin_h + (text_area_w - text_w) / 2

            # Vertical: compute where libass places the font cell top
            if alignment == 2:
                text_top_y = video_height - margin_v - line_h
            elif alignment == 5:
                text_top_y = (video_height - line_h) / 2
            elif alignment == 8:
                text_top_y = margin_v
            else:
                continue

            # ── Tight background using reference visual height ──
            # The reference "Hg" measurement gives us a stable visual height
            # that includes both ascenders and descenders. This is much tighter
            # than ascent+descent (which reserves space for accents like Ä).
            #
            # Position the background rect centered on the reference visual
            # midpoint within the font cell:
            #   ref_center_y = text_top_y + _bg_ref_vis_top + _bg_ref_vis_h / 2
            #   draw_y = ref_center_y - _bg_ref_vis_h / 2 - pad_v
            #          = text_top_y + _bg_ref_vis_top - pad_v
            draw_x = round(line_left_x - _bg_draw_pad_h)
            draw_y = round(text_top_y + _bg_ref_vis_top - _bg_draw_pad_v)
            draw_w = round(text_w + 2 * _bg_draw_pad_h)
            draw_h = round(_bg_ref_vis_h + 2 * _bg_draw_pad_v)

            drawing = _ass_rounded_rect(draw_w, draw_h, _bg_draw_radius)
            event_text = (
                f"{{\\an7\\pos({draw_x},{draw_y})"
                f"\\p1\\c{_bg_color}\\1a{_bg_alpha}"
                f"\\bord0\\shad0}}{drawing}"
            )
            pending_bg_draw_events.append((ev_start, ev_end, "BGDRAW", event_text))

        logger.info(
            "BGDRAW: generated %d rounded-rect drawing events "
            "(radius=%dpx, pad=%dx%d, ref_vis_h=%d, font=%s)",
            len(pending_bg_draw_events), _bg_draw_radius,
            _bg_draw_pad_h, _bg_draw_pad_v, _bg_ref_vis_h, _bg_font_path,
        )

    # ═══════════════════════════════════════════════════════════════════
    # FINAL OVERLAP ELIMINATION — unified pass at centisecond precision
    # ═══════════════════════════════════════════════════════════════════
    # All upstream clamping operates at float precision.  ASS timestamps
    # have centisecond resolution (H:MM:SS.cc), so float values that are
    # < 0.005 apart can round to the same centisecond — or worse, round
    # so that event N's formatted end > event N+1's formatted start,
    # creating a 1-centisecond overlap that makes libass stack them.
    #
    # This final pass operates at the OUTPUT precision (centiseconds)
    # to guarantee no two same-layer events overlap at the resolution
    # that libass actually sees.
    def _to_cs(t: float) -> int:
        """Convert seconds to centiseconds matching _format_ass_time output.

        _format_ass_time uses f'{s:05.2f}' which rounds to 2 decimal
        places using the C printf convention (round half away from zero).
        Python's round() uses banker's rounding (round half to even),
        which can produce different results at .5 boundaries:
          round(2.5) → 2  (banker's)  vs  f-string '2.50' → 2  (same here)
          round(3.5) → 4  (banker's)  vs  f-string '3.50' → 4  (same here)
        But at centisecond boundaries like 1.005s:
          round(1.005 * 100) = round(100.5) → 100  (banker's rounds to even)
          f'{1.005:05.2f}' → '01.01'  (printf rounds 0.5 up → 101 cs)

        To match _format_ass_time exactly, we replicate the printf-style
        rounding by using the Decimal module or by formatting and parsing.
        Simpler approach: format the seconds part the same way _format_ass_time
        does and convert back.
        """
        if t < 0:
            t = 0
        s = t % 60
        # Format with :.2f (same as _format_ass_time) and parse back
        formatted = f"{s:.2f}"
        cs_from_seconds = int(round(float(formatted) * 100))
        # Add the minutes/hours contribution
        total_minutes = int(t // 60)
        return total_minutes * 6000 + cs_from_seconds

    # When background + outline are both enabled, create outline overlay events
    # on a separate layer using the _OL style (BorderStyle=1 with text outline).
    outline_overlay_events: list[tuple[float, float, str, str]] = []
    if bg_has_outline:
        for ev in base_text_events:
            # Replace style with _OL variant for outline rendering
            ol_style = ev[2] + "_OL"
            outline_overlay_events.append((ev[0], ev[1], ol_style, ev[3]))

    # Collect all events as (layer, start, end, style, text) tuples.
    # Layer ordering (lower = renders first / behind):
    #   Layer 0: BGDRAW events (rounded-rect subtitle background)
    #   Layer N: base text events (border layer, uniform color)
    #   Layer N+1: outline overlay events (when bg + outline both enabled)
    #   Layer N+2: active word box events (BorderStyle=3 via _AWBG style)
    #   Layer N+3: active word color events (topmost, sharp text)
    all_events: list[tuple[int, float, float, str, str]] = []
    next_layer = 0
    if pending_bg_draw_events:
        for ev in pending_bg_draw_events:
            all_events.append((next_layer, ev[0], ev[1], ev[2], ev[3]))
        next_layer += 1
    for ev in base_text_events:
        all_events.append((next_layer, ev[0], ev[1], ev[2], ev[3]))
    next_layer += 1
    for ev in outline_overlay_events:
        all_events.append((next_layer, ev[0], ev[1], ev[2], ev[3]))
    if outline_overlay_events:
        next_layer += 1
    if pending_box_events:
        for ev in pending_box_events:
            all_events.append((next_layer, ev[0], ev[1], ev[2], ev[3]))
        next_layer += 1
    for ev in pending_word_events:
        all_events.append((next_layer, ev[0], ev[1], ev[2], ev[3]))

    # Process each layer independently.
    _max_layer = max((e[0] for e in all_events), default=0)
    for layer in range(0, _max_layer + 1):
        layer_evs = [e for e in all_events if e[0] == layer]
        if not layer_evs:
            continue
        layer_evs.sort(key=lambda e: e[1])

        # Clamp any residual overlaps at centisecond precision.
        # Use _format_ass_time round-trip to compare at the exact precision
        # that libass will see — this eliminates any float→string rounding
        # edge cases that _to_cs might not perfectly capture.
        for i in range(len(layer_evs) - 1):
            _, s, e, st, tx = layer_evs[i]
            _, ns, _, _, _ = layer_evs[i + 1]
            e_cs = _to_cs(e)
            ns_cs = _to_cs(ns)
            if e_cs >= ns_cs:
                # Overlap or touching at display precision — clamp end
                # to next start so libass never renders both events.
                # When e_cs == ns_cs, the formatted timestamps are
                # identical, meaning libass would display both events
                # at that centisecond — clamp to eliminate.
                layer_evs[i] = (layer, s, ns, st, tx)

        # Merge sub-centisecond events into their predecessor instead of
        # silently dropping them — ensures every word gets highlighted.
        final_layer_evs = []
        for ev in layer_evs:
            layer_num, ev_s, ev_e, ev_st, ev_tx = ev
            dur_cs = _to_cs(ev_e) - _to_cs(ev_s)
            if dur_cs >= 1:
                final_layer_evs.append(ev)
            elif final_layer_evs:
                # Merge into previous event: extend its end and use this
                # event's text (which has the current word highlighted).
                prev = final_layer_evs[-1]
                final_layer_evs[-1] = (prev[0], prev[1], ev_e, prev[3], ev_tx)

        # Emit events to ASS output
        for _, ev_s, ev_e, ev_st, ev_tx in final_layer_evs:
            if _to_cs(ev_e) - _to_cs(ev_s) >= 1:  # at least 1 centisecond
                lines.append(
                    f"Dialogue: {layer},{_format_ass_time(ev_s)},"
                    f"{_format_ass_time(ev_e)},{ev_st},,0,0,0,,{ev_tx}"
                )

    logger.info(
        "ASS generated: font=%s size=%s(%dpx) weight=%s color=%s pos=%s "
        "bg=%s(%s) bg_radius=%d bg_split=%s bg_ref_vis_h=%d outline=%dpx speakers=%d segments=%d "
        "active_word=%s aw_bg_opacity=%d aw_bg_padding=%d "
        "bg_draw_events=%d word_events=%d max_words=%d res=%dx%d",
        font, font_size, size_px, font_weight, font_color, position,
        "yes" if background_enabled else "no",
        f"{background_color}@{background_opacity}%" if background_enabled else "n/a",
        background_radius, _bg_split, _bg_ref_vis_h if _bg_split else 0,
        ol_width, len(speakers_seen), len(clip_segments),
        "yes" if active_word_enabled else "no",
        active_word_bg_opacity, _aw_bg_box_padding,
        len(pending_bg_draw_events), len(pending_word_events),
        max_words, video_width, video_height,
    )

    return "\n".join(lines) + "\n"


def append_text_overlays_to_ass(
    ass_content: str,
    text_overlays: list[dict],
    clip_start: float = 0,
    video_out_w: int = 1920,
    video_out_h: int = 1080,
) -> str:
    """Append text overlay items as ASS dialogue events.

    Used when FFmpeg's drawtext filter is unavailable. Converts each
    text overlay to an ASS Dialogue line with positioning, font,
    color, and timing that matches the preview.
    """
    if not text_overlays:
        return ass_content

    _PREVIEW_REF = 960
    res_scale = max(video_out_w, video_out_h) / _PREVIEW_REF

    new_styles = []
    new_events = []

    for i, overlay in enumerate(text_overlays):
        text = overlay.get("text", "").strip()
        if not text:
            continue

        start_t = max(0, overlay.get("start_time", 0) - clip_start)
        end_t = max(0, overlay.get("end_time", 0) - clip_start)
        if end_t <= start_t:
            continue

        pos_x = int((overlay.get("x", 50) / 100.0) * video_out_w)
        pos_y = int((overlay.get("y", 50) / 100.0) * video_out_h)

        font_size = max(8, int(round(overlay.get("font_size", 48) * res_scale)))
        font_family = overlay.get("font_family", "Arial")
        font_color = overlay.get("font_color", "#FFFFFF")
        font_weight = overlay.get("font_weight", 400)
        if isinstance(font_weight, str):
            font_weight = 700 if font_weight.lower() == "bold" else 400
        is_bold = 1 if int(font_weight) >= 600 else 0

        primary_color = _hex_to_ass_color(font_color)
        outline_width = max(0, int(round(overlay.get("outline_width", 0) * res_scale / 2)))
        outline_color = _hex_to_ass_color(overlay.get("outline_color", "#000000"))

        shadow_x = int(round(overlay.get("shadow_offset_x", 0) * res_scale))
        shadow_y = int(round(overlay.get("shadow_offset_y", 0) * res_scale))
        shadow_depth = max(abs(shadow_x), abs(shadow_y))

        opacity = overlay.get("opacity", 1.0)
        alpha = max(0, min(255, int((1 - opacity) * 255)))
        alpha_tag = f"\\1a&H{alpha:02X}&" if alpha > 0 else ""

        style_name = f"TextOverlay{i + 1}"
        align = 5  # Middle-center, positioned with \pos

        style_line = (
            f"Style: {style_name},{font_family},{font_size},"
            f"{primary_color},&H000000FF&,{outline_color},&H00000000&,"
            f"{is_bold},0,0,0,100,100,0,0,1,{outline_width},{shadow_depth},"
            f"{align},10,10,10,1"
        )
        new_styles.append(style_line)

        start_ts = _format_ass_time(start_t)
        end_ts = _format_ass_time(end_t)
        text_escaped = text.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")
        text_escaped = text_escaped.replace("\n", "\\N")
        override = f"{{\\pos({pos_x},{pos_y}){alpha_tag}}}"

        event_line = f"Dialogue: 10,{start_ts},{end_ts},{style_name},,0,0,0,,{override}{text_escaped}"
        new_events.append(event_line)

    if not new_events:
        return ass_content

    lines = ass_content.split("\n")

    # Insert new styles after the last existing Style: line
    last_style_idx = -1
    for idx, line in enumerate(lines):
        if line.startswith("Style:"):
            last_style_idx = idx
    if last_style_idx >= 0:
        for j, style in enumerate(new_styles):
            lines.insert(last_style_idx + 1 + j, style)

    # Append dialogue events at the end
    lines.extend(new_events)

    return "\n".join(lines)
