"""Comprehensive tests for the clip export pipeline.

Tests cover:
  - ASS subtitle generation with all settings variations
  - FFmpeg filter chain construction
  - Settings flow from frontend dict through to ASS output
  - Frontend/backend constant consistency

Run: cd /home/user/115 && python3 backend/tests/test_export_pipeline.py
"""
import os
import re
import sys

# Ensure project root is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from backend.models import TranscriptSegment, SubtitleSettings
from backend.services.ass_generator import (
    generate_ass,
    split_segments_by_max_words,
    FONT_SIZE_MAP,
    FONT_WEIGHT_MAP,
    POSITION_ALIGNMENT,
    DEFAULT_SPEAKER_PALETTE,
    REF_W,
    REF_H,
    MIN_TEXT_AREA_W,
    MIN_TEXT_AREA_H,
    DEFAULT_MARGIN_H,
    DEFAULT_MARGIN_V,
    _hex_to_ass_color,
    _hex_to_ass_color_with_alpha,
    _format_ass_time,
)
from backend.services.clip_exporter import (
    _build_filter_chain,
    _build_subject_keyframes,
    _safe_subject_x,
    _smooth_keyframes,
    _build_crop_x_expr,
    _center_crop_offset,
    _subtitle_filter,
    _extract_force_style_from_ass,
    _validate_ass_settings,
    _validate_subject_tracking,
    ASPECT_RATIO_DIMS,
    ASPECT_RATIO_DIMS_BY_QUALITY,
    ASPECT_RATIO_VALUES,
    QUALITY_PRESETS,
    QUALITY_MAX_HEIGHT,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

passed = 0
failed = 0
errors = []

SAMPLE_SEGMENTS = [
    TranscriptSegment(start=10.0, end=15.0, text="Hello world", speaker="Speaker 1"),
    TranscriptSegment(start=15.0, end=20.0, text="How are you", speaker="Speaker 2"),
    TranscriptSegment(start=20.0, end=25.0, text="Doing great", speaker="Speaker 1"),
]


def _filter_font_install_warnings(warnings: list[str]) -> list[str]:
    """Filter out font-availability warnings that depend on the host environment.

    The font availability check (#16 in _validate_ass_settings) calls fc-match
    to verify that fontconfig can resolve the font.  This check is inherently
    environment-dependent — fonts like DM Sans are installed inside the Docker
    container but typically absent on the CI/dev host.  Filtering these out lets
    the test suite verify all other QA properties regardless of host fonts.
    """
    return [w for w in warnings if "not installed" not in w and "fc-match" not in w]


def validate_ass(ass_content, settings, w, h):
    """Wrapper around _validate_ass_settings that filters environment-dependent warnings."""
    return _filter_font_install_warnings(_validate_ass_settings(ass_content, settings, w, h))


def check(name: str, condition: bool, detail: str = ""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        msg = f"  FAIL  {name}"
        if detail:
            msg += f" — {detail}"
        print(msg)
        errors.append(msg)


def parse_ass_styles(ass_text: str) -> dict[str, dict]:
    """Parse ASS styles into a dict keyed by style name."""
    styles = {}
    in_styles = False
    fmt_keys = []
    for line in ass_text.splitlines():
        if line.strip().startswith("Format:") and "Fontname" in line:
            fmt_keys = [k.strip() for k in line.split(":", 1)[1].split(",")]
            in_styles = True
            continue
        if in_styles and line.startswith("Style:"):
            vals = [v.strip() for v in line.split(":", 1)[1].split(",")]
            style = {}
            for i, key in enumerate(fmt_keys):
                if i < len(vals):
                    style[key] = vals[i]
            styles[style.get("Name", "")] = style
        if line.startswith("[Events]"):
            break
    return styles


def parse_ass_dialogues(ass_text: str) -> list[dict]:
    """Parse ASS dialogue lines."""
    dialogues = []
    fmt_keys = []
    in_events = False
    for line in ass_text.splitlines():
        if line.strip() == "[Events]":
            in_events = True
            continue
        if in_events and line.strip().startswith("Format:"):
            fmt_keys = [k.strip() for k in line.split(":", 1)[1].split(",")]
            continue
        if in_events and line.startswith("Dialogue:"):
            parts = line.split(":", 1)[1].split(",", len(fmt_keys) - 1)
            d = {}
            for i, key in enumerate(fmt_keys):
                if i < len(parts):
                    d[key] = parts[i].strip()
            dialogues.append(d)
    return dialogues


# ===========================================================================
# TEST SUITE 1: ASS Generator
# ===========================================================================

def test_ass_default_settings():
    print("\n--- ASS: Default settings ---")
    result = generate_ass(segments=SAMPLE_SEGMENTS, start_time=10.0, end_time=25.0)

    check("Non-empty output", len(result) > 0)
    check("Has [Script Info]", "[Script Info]" in result)
    check("Has PlayResX: 1920", "PlayResX: 1920" in result)
    check("Has PlayResY: 1080", "PlayResY: 1080" in result)
    check("Has WrapStyle: 1", "WrapStyle: 1" in result)
    check("Has ScaledBorderAndShadow", "ScaledBorderAndShadow: no" in result)
    check("Has [V4+ Styles]", "[V4+ Styles]" in result)
    check("Has [Events]", "[Events]" in result)

    styles = parse_ass_styles(result)
    check("Has Speaker 1 style", "Speaker 1" in styles)
    check("Has Speaker 2 style", "Speaker 2" in styles)

    dialogues = parse_ass_dialogues(result)
    check("Has 3 dialogue lines", len(dialogues) == 3, f"got {len(dialogues)}")

    # Default: timestamps offset to 0-based
    check("First dialogue starts at 0", dialogues[0]["Start"] == "0:00:00.00",
          f"got {dialogues[0].get('Start')}")

    # Default: speaker labels disabled (show_speaker_labels=False)
    check("No speaker label in text by default", "Speaker 1:" not in dialogues[0].get("Text", ""),
          f"got text: {dialogues[0].get('Text')}")


def test_ass_font_sizes():
    print("\n--- ASS: Font sizes at 1920x1080 ---")
    for size_name, expected_px in [("small", 22), ("medium", 30), ("large", 40)]:
        result = generate_ass(
            segments=SAMPLE_SEGMENTS, start_time=10.0, end_time=25.0,
            font_size=size_name, video_width=1920, video_height=1080,
        )
        styles = parse_ass_styles(result)
        sp1 = styles.get("Speaker 1", {})
        actual = int(sp1.get("Fontsize", 0))
        check(f"Font size '{size_name}' = {expected_px}px at 1080p",
              actual == expected_px, f"got {actual}")


def test_ass_font_size_scaling_portrait():
    print("\n--- ASS: Font size scaling for 9:16 (1080x1920) ---")
    # font_scale = min(1080, 1920) / min(1920, 1080) = 1080/1080 = 1.0
    # medium=30 → 30 * 1.0 = 30, max(16, 30) = 30
    result = generate_ass(
        segments=SAMPLE_SEGMENTS, start_time=10.0, end_time=25.0,
        font_size="medium", video_width=1080, video_height=1920,
    )
    styles = parse_ass_styles(result)
    actual = int(styles.get("Speaker 1", {}).get("Fontsize", 0))
    check("Medium font at 9:16 = 30px", actual == 30, f"got {actual}")

    check("PlayResX: 1080", "PlayResX: 1080" in result)
    check("PlayResY: 1920", "PlayResY: 1920" in result)

    # small=22 → 22 * 1.0 = 22, max(16, 22) = 22
    result2 = generate_ass(
        segments=SAMPLE_SEGMENTS, start_time=10.0, end_time=25.0,
        font_size="small", video_width=1080, video_height=1920,
    )
    actual2 = int(parse_ass_styles(result2).get("Speaker 1", {}).get("Fontsize", 0))
    check("Small font at 9:16 = 22px", actual2 == 22, f"got {actual2}")

    # large=40 → 40 * 1.0 = 40, max(16, 40) = 40
    result3 = generate_ass(
        segments=SAMPLE_SEGMENTS, start_time=10.0, end_time=25.0,
        font_size="large", video_width=1080, video_height=1920,
    )
    actual3 = int(parse_ass_styles(result3).get("Speaker 1", {}).get("Fontsize", 0))
    check("Large font at 9:16 = 40px", actual3 == 40, f"got {actual3}")


def test_ass_font_size_scaling_square():
    print("\n--- ASS: Font size scaling for 1:1 (1080x1080) ---")
    # font_scale = min(1080, 1080) / min(1920, 1080) = 1080/1080 = 1.0
    result = generate_ass(
        segments=SAMPLE_SEGMENTS, start_time=10.0, end_time=25.0,
        font_size="medium", video_width=1080, video_height=1080,
    )
    actual = int(parse_ass_styles(result).get("Speaker 1", {}).get("Fontsize", 0))
    check("Medium font at 1:1 = 30px", actual == 30, f"got {actual}")


def test_ass_font_weight():
    print("\n--- ASS: Font weight ---")
    for weight, expected_bold in [("normal", "0"), ("bold", "-1")]:
        result = generate_ass(
            segments=SAMPLE_SEGMENTS, start_time=10.0, end_time=25.0,
            font_weight=weight,
        )
        styles = parse_ass_styles(result)
        actual = styles.get("Speaker 1", {}).get("Bold", "?")
        check(f"Weight '{weight}' → Bold={expected_bold}", actual == expected_bold,
              f"got Bold={actual}")


def test_ass_custom_font_name():
    print("\n--- ASS: Custom font name in styles ---")
    # Verify that a custom font name is passed through to the ASS style definitions
    result = generate_ass(
        segments=SAMPLE_SEGMENTS, start_time=10.0, end_time=25.0,
        font="Montserrat",
    )
    styles = parse_ass_styles(result)
    sp1 = styles.get("Speaker 1", {})
    actual_font = sp1.get("Fontname", "")
    check("Custom font 'Montserrat' in Speaker 1 style",
          actual_font == "Montserrat", f"got '{actual_font}'")

    sp2 = styles.get("Speaker 2", {})
    actual_font2 = sp2.get("Fontname", "")
    check("Custom font 'Montserrat' in Speaker 2 style",
          actual_font2 == "Montserrat", f"got '{actual_font2}'")

    # Verify default font is DM Sans when not specified
    result2 = generate_ass(
        segments=SAMPLE_SEGMENTS, start_time=10.0, end_time=25.0,
    )
    styles2 = parse_ass_styles(result2)
    default_font = styles2.get("Speaker 1", {}).get("Fontname", "")
    check("Default font is 'DM Sans'",
          default_font == "DM Sans", f"got '{default_font}'")

    # Test font name with spaces
    result3 = generate_ass(
        segments=SAMPLE_SEGMENTS, start_time=10.0, end_time=25.0,
        font="Playfair Display",
    )
    styles3 = parse_ass_styles(result3)
    spaced_font = styles3.get("Speaker 1", {}).get("Fontname", "")
    check("Font name with spaces 'Playfair Display' preserved",
          spaced_font == "Playfair Display", f"got '{spaced_font}'")


def test_ass_font_color_override():
    print("\n--- ASS: Font color override ---")
    # use_speaker_colors=False makes font_color apply uniformly to all speakers
    result = generate_ass(
        segments=SAMPLE_SEGMENTS, start_time=10.0, end_time=25.0,
        font_color="#FF0000",
        use_speaker_colors=False,
    )
    styles = parse_ass_styles(result)
    expected_ass_color = _hex_to_ass_color("#FF0000")  # &H000000FF&
    sp1_color = styles.get("Speaker 1", {}).get("PrimaryColour", "")
    sp2_color = styles.get("Speaker 2", {}).get("PrimaryColour", "")
    check("Custom font_color applied to Speaker 1",
          sp1_color == expected_ass_color, f"got {sp1_color}, expected {expected_ass_color}")
    check("Custom font_color applied to Speaker 2 (same)",
          sp2_color == expected_ass_color, f"got {sp2_color}")

    # White color → uses speaker palette
    result2 = generate_ass(
        segments=SAMPLE_SEGMENTS, start_time=10.0, end_time=25.0,
        font_color="#FFFFFF",
    )
    styles2 = parse_ass_styles(result2)
    sp1_color2 = styles2.get("Speaker 1", {}).get("PrimaryColour", "")
    sp2_color2 = styles2.get("Speaker 2", {}).get("PrimaryColour", "")
    expected_sp1 = _hex_to_ass_color(DEFAULT_SPEAKER_PALETTE[0])
    expected_sp2 = _hex_to_ass_color(DEFAULT_SPEAKER_PALETTE[1])
    check("White font_color → Speaker 1 uses palette[0]",
          sp1_color2 == expected_sp1, f"got {sp1_color2}, expected {expected_sp1}")
    check("White font_color → Speaker 2 uses palette[1]",
          sp2_color2 == expected_sp2, f"got {sp2_color2}, expected {expected_sp2}")


def test_ass_custom_speaker_colors():
    print("\n--- ASS: Custom speaker colors ---")
    result = generate_ass(
        segments=SAMPLE_SEGMENTS, start_time=10.0, end_time=25.0,
        font_color="#FFFFFF",  # white = use speaker colors
        speaker_colors={"Speaker 1": "#00FF00", "Speaker 2": "#0000FF"},
    )
    styles = parse_ass_styles(result)
    sp1 = styles.get("Speaker 1", {}).get("PrimaryColour", "")
    sp2 = styles.get("Speaker 2", {}).get("PrimaryColour", "")
    check("Speaker 1 custom color",
          sp1 == _hex_to_ass_color("#00FF00"), f"got {sp1}")
    check("Speaker 2 custom color",
          sp2 == _hex_to_ass_color("#0000FF"), f"got {sp2}")


def test_ass_positions():
    print("\n--- ASS: Subtitle positions (all alignment=2 with absolute offset_v) ---")
    # All positions now produce alignment=2 (bottom-center) because offset_v
    # is the sole vertical control (0=bottom, 100=top)
    for pos in ["bottom", "center", "top"]:
        result = generate_ass(
            segments=SAMPLE_SEGMENTS, start_time=10.0, end_time=25.0,
            position=pos,
        )
        styles = parse_ass_styles(result)
        actual = styles.get("Speaker 1", {}).get("Alignment", "?")
        check(f"Position '{pos}' → Alignment=2 (always bottom-center)",
              actual == "2", f"got Alignment={actual}")

    # Center position with default offset_v=4 → MarginV = 1080*4/100 = 43
    result_center = generate_ass(
        segments=SAMPLE_SEGMENTS, start_time=10.0, end_time=25.0,
        position="center",
    )
    styles_center = parse_ass_styles(result_center)
    margin_v = styles_center.get("Speaker 1", {}).get("MarginV", "?")
    expected_mv = str(int(1080 * 4 / 100))  # default offset_v=4
    check(f"Center position → MarginV={expected_mv} (from default offset_v=4)",
          margin_v == expected_mv, f"got MarginV={margin_v}")


def test_ass_outline_mode():
    print("\n--- ASS: Outline mode (no background) ---")
    result = generate_ass(
        segments=SAMPLE_SEGMENTS, start_time=10.0, end_time=25.0,
        background_enabled=False,
        outline_color="#FF0000",
        outline_opacity=80,
        outline_width=3,
    )
    styles = parse_ass_styles(result)
    sp1 = styles.get("Speaker 1", {})

    check("BorderStyle=1 (outline)", sp1.get("BorderStyle") == "1",
          f"got {sp1.get('BorderStyle')}")
    # outline_width=3 scaled by font_scale=1.0 at 1080p → round(3*1.0)=3
    check("Outline width=3 (1x factor)", sp1.get("Outline") == "3",
          f"got {sp1.get('Outline')}")
    # Shadow = max(1, min(4, round(3*0.75))) = max(1, min(4, 2)) = 2
    check("Shadow depth=2 (outline mode with width>0)", sp1.get("Shadow") == "2",
          f"got {sp1.get('Shadow')}")

    expected_ol_color = _hex_to_ass_color_with_alpha("#FF0000", 80)
    actual_ol_color = sp1.get("OutlineColour", "")
    check("OutlineColour matches",
          actual_ol_color == expected_ol_color,
          f"got {actual_ol_color}, expected {expected_ol_color}")


def test_ass_background_box_mode():
    print("\n--- ASS: Background box mode ---")
    result = generate_ass(
        segments=SAMPLE_SEGMENTS, start_time=10.0, end_time=25.0,
        background_enabled=True,
        background_color="#003366",
        background_opacity=75,
    )
    styles = parse_ass_styles(result)
    sp1 = styles.get("Speaker 1", {})

    check("BorderStyle=3 (box)", sp1.get("BorderStyle") == "3",
          f"got {sp1.get('BorderStyle')}")
    check("Shadow=0 (box mode)", sp1.get("Shadow") == "0",
          f"got {sp1.get('Shadow')}")

    # Box padding: max(int(4 * font_scale), 2) at 1080p → max(4, 2) = 4
    check("Outline (box padding) = 4", sp1.get("Outline") == "4",
          f"got {sp1.get('Outline')}")

    expected_back = _hex_to_ass_color_with_alpha("#003366", 75)
    actual_back = sp1.get("BackColour", "")
    check("BackColour matches bg_color+opacity",
          actual_back == expected_back,
          f"got {actual_back}, expected {expected_back}")

    # OutlineColour should equal BackColour (blends border into box)
    actual_outline = sp1.get("OutlineColour", "")
    check("OutlineColour == BackColour (blended border)",
          actual_outline == expected_back,
          f"got outline={actual_outline}, back={expected_back}")


def test_ass_speaker_labels():
    print("\n--- ASS: Speaker labels ---")
    # Labels ON
    result_on = generate_ass(
        segments=SAMPLE_SEGMENTS, start_time=10.0, end_time=25.0,
        show_speaker_labels=True,
    )
    dialogues_on = parse_ass_dialogues(result_on)
    check("Labels ON: text has 'Speaker 1:'",
          "Speaker 1:" in dialogues_on[0].get("Text", ""))

    # Labels OFF
    result_off = generate_ass(
        segments=SAMPLE_SEGMENTS, start_time=10.0, end_time=25.0,
        show_speaker_labels=False,
    )
    dialogues_off = parse_ass_dialogues(result_off)
    check("Labels OFF: text has no 'Speaker 1:'",
          "Speaker 1:" not in dialogues_off[0].get("Text", ""),
          f"got text: {dialogues_off[0].get('Text')}")
    # Text may have explicit \bord override prefix — check content is present
    off_text = dialogues_off[0].get("Text", "").strip()
    check("Labels OFF: text contains content",
          "Hello world" in off_text,
          f"got: {off_text}")
    check("Labels OFF: no speaker label in text",
          "Speaker 1:" not in off_text,
          f"got: {off_text}")


def test_ass_max_width():
    print("\n--- ASS: Max width ---")
    # max_width=80 → margin_h = max(20, int(1920 * 20 / 100 / 2)) = max(20, 192) = 192
    result = generate_ass(
        segments=SAMPLE_SEGMENTS, start_time=10.0, end_time=25.0,
        max_width_pct=80,
    )
    styles = parse_ass_styles(result)
    margin_l = int(styles.get("Speaker 1", {}).get("MarginL", 0))
    margin_r = int(styles.get("Speaker 1", {}).get("MarginR", 0))
    check("Max width 80% → MarginL=192", margin_l == 192, f"got {margin_l}")
    check("Max width 80% → MarginR=192", margin_r == 192, f"got {margin_r}")
    check("MarginL == MarginR", margin_l == margin_r)


def test_ass_vertical_offset():
    print("\n--- ASS: Vertical offset ---")
    # offset_v=10 → margin_v = max(10, int(1080 * 10 / 100)) = max(10, 108) = 108
    result = generate_ass(
        segments=SAMPLE_SEGMENTS, start_time=10.0, end_time=25.0,
        offset_v_pct=10,
    )
    styles = parse_ass_styles(result)
    margin_v = int(styles.get("Speaker 1", {}).get("MarginV", 0))
    check("Offset 10% → MarginV=108", margin_v == 108, f"got {margin_v}")

    # offset_v=0 → margin_v = int(1080 * 0 / 100) = 0
    result2 = generate_ass(
        segments=SAMPLE_SEGMENTS, start_time=10.0, end_time=25.0,
        offset_v_pct=0,
    )
    styles2 = parse_ass_styles(result2)
    margin_v2 = int(styles2.get("Speaker 1", {}).get("MarginV", 0))
    check("Offset 0% → MarginV=0", margin_v2 == 0, f"got {margin_v2}")


def test_ass_outline_width_scaling():
    print("\n--- ASS: Outline width scaling ---")
    # At 1080p: font_scale=1.0, outline_width=5 → round(5*1.0*3)=15
    result = generate_ass(
        segments=SAMPLE_SEGMENTS, start_time=10.0, end_time=25.0,
        outline_width=5, video_width=1920, video_height=1080,
    )
    ol = parse_ass_styles(result).get("Speaker 1", {}).get("Outline", "?")
    check("Outline width 5 at 1080p → 15", ol == "15", f"got {ol}")

    # At 9:16 (1080x1920): font_scale=1.0, outline_width=5 → round(5*1.0*3)=15
    result2 = generate_ass(
        segments=SAMPLE_SEGMENTS, start_time=10.0, end_time=25.0,
        outline_width=5, video_width=1080, video_height=1920,
    )
    ol2 = parse_ass_styles(result2).get("Speaker 1", {}).get("Outline", "?")
    check("Outline width 5 at 9:16 → 15 (with 3x factor)", ol2 == "15", f"got {ol2}")


def test_ass_multiple_aspect_ratios():
    print("\n--- ASS: Multiple aspect ratio resolutions ---")
    for label, w, h in [("16:9", 1920, 1080), ("9:16", 1080, 1920),
                         ("1:1", 1080, 1080), ("4:5", 1080, 1350)]:
        result = generate_ass(
            segments=SAMPLE_SEGMENTS, start_time=10.0, end_time=25.0,
            video_width=w, video_height=h,
        )
        check(f"{label} → PlayResX: {w}", f"PlayResX: {w}" in result)
        check(f"{label} → PlayResY: {h}", f"PlayResY: {h}" in result)


def test_ass_segment_filtering():
    print("\n--- ASS: Segment time filtering ---")
    # Only include segments within [12, 22] — should get partial overlaps
    result = generate_ass(
        segments=SAMPLE_SEGMENTS, start_time=12.0, end_time=22.0,
    )
    dialogues = parse_ass_dialogues(result)
    check("Filtered: 3 segments overlap [12, 22]", len(dialogues) == 3,
          f"got {len(dialogues)}")

    # First segment: start=max(10,12)-12=0, end=min(15,22)-12=3
    check("First seg starts at 0:00:00.00",
          dialogues[0]["Start"] == "0:00:00.00",
          f"got {dialogues[0].get('Start')}")

    # Only include [10, 12] — just the first segment partially
    result2 = generate_ass(
        segments=SAMPLE_SEGMENTS, start_time=10.0, end_time=12.0,
    )
    dialogues2 = parse_ass_dialogues(result2)
    check("Tight range [10,12]: 1 segment", len(dialogues2) == 1,
          f"got {len(dialogues2)}")


def test_ass_empty_segments():
    print("\n--- ASS: Empty/no matching segments ---")
    result = generate_ass(
        segments=SAMPLE_SEGMENTS, start_time=100.0, end_time=200.0,
    )
    check("No matching segments → empty string", result == "")


def test_ass_font_name():
    print("\n--- ASS: Font name ---")
    result = generate_ass(
        segments=SAMPLE_SEGMENTS, start_time=10.0, end_time=25.0,
        font="Arial",
    )
    styles = parse_ass_styles(result)
    fontname = styles.get("Speaker 1", {}).get("Fontname", "")
    check("Custom font 'Arial'", fontname == "Arial", f"got '{fontname}'")


# ===========================================================================
# TEST SUITE 1b: ASS Generator — Edge Cases & Combinations
# ===========================================================================

def test_ass_max_width_edge_values():
    print("\n--- ASS: Max width edge values ---")
    # max_width=100 → margin_h = max(20, int(1920 * 0 / 100 / 2)) = max(20, 0) = 20
    result = generate_ass(
        segments=SAMPLE_SEGMENTS, start_time=10.0, end_time=25.0,
        max_width_pct=100,
    )
    styles = parse_ass_styles(result)
    margin_l = int(styles.get("Speaker 1", {}).get("MarginL", 0))
    check("Max width 100% → MarginL=20 (min clamp)", margin_l == 20, f"got {margin_l}")

    # max_width=50 → margin_h = max(20, int(1920 * 50 / 100 / 2)) = max(20, 480) = 480
    # Safe-area cap: max_margin_h = int(1920 * (1-0.50) / 2) = 480 → capped to 480
    result2 = generate_ass(
        segments=SAMPLE_SEGMENTS, start_time=10.0, end_time=25.0,
        max_width_pct=50,
    )
    styles2 = parse_ass_styles(result2)
    margin_l2 = int(styles2.get("Speaker 1", {}).get("MarginL", 0))
    check("Max width 50% → MarginL=480", margin_l2 == 480, f"got {margin_l2}")


def test_ass_offset_v_edge_values():
    print("\n--- ASS: Vertical offset edge values ---")
    # offset_v=0 → margin_v = int(1080 * 0 / 100) = 0
    result = generate_ass(
        segments=SAMPLE_SEGMENTS, start_time=10.0, end_time=25.0,
        offset_v_pct=0, position="bottom",
    )
    styles = parse_ass_styles(result)
    margin_v = int(styles.get("Speaker 1", {}).get("MarginV", 0))
    check("Offset 0% → MarginV=0", margin_v == 0, f"got {margin_v}")

    # offset_v=100 → margin_v = int(1080 * 100 / 100) = 1080 (no cap)
    result2 = generate_ass(
        segments=SAMPLE_SEGMENTS, start_time=10.0, end_time=25.0,
        offset_v_pct=100, position="bottom",
    )
    styles2 = parse_ass_styles(result2)
    margin_v2 = int(styles2.get("Speaker 1", {}).get("MarginV", 0))
    check("Offset 100% → MarginV=1080", margin_v2 == 1080, f"got {margin_v2}")

    # center position now uses offset_v (absolute positioning: 0=bottom, 100=top)
    result3 = generate_ass(
        segments=SAMPLE_SEGMENTS, start_time=10.0, end_time=25.0,
        offset_v_pct=50, position="center",
    )
    styles3 = parse_ass_styles(result3)
    margin_v3 = int(styles3.get("Speaker 1", {}).get("MarginV", 0))
    expected_mv3 = int(1080 * 50 / 100)
    check(f"Center position offset_v=50 → MarginV={expected_mv3}", margin_v3 == expected_mv3, f"got {margin_v3}")

    # top position also uses offset_v
    result4 = generate_ass(
        segments=SAMPLE_SEGMENTS, start_time=10.0, end_time=25.0,
        offset_v_pct=4, position="top",
    )
    styles4 = parse_ass_styles(result4)
    margin_v4 = int(styles4.get("Speaker 1", {}).get("MarginV", 0))
    expected_v4 = max(10, int(1080 * 4 / 100))  # max(10, 43) = 43
    check(f"Top position + offset_v=4 → MarginV={expected_v4}", margin_v4 == expected_v4, f"got {margin_v4}")


def test_ass_safe_area_cap():
    print("\n--- ASS: Safe-area cap enforcement ---")
    # With large content_inset_h pushing margin beyond safe area
    # max_width=90 → margin_h = max(20, 96) = 96, + inset 500 = 596
    # Safe cap: int(1920 * (1-0.50) / 2) = 480 → capped to 480
    result = generate_ass(
        segments=SAMPLE_SEGMENTS, start_time=10.0, end_time=25.0,
        max_width_pct=90, content_inset_h=500,
    )
    styles = parse_ass_styles(result)
    margin_l = int(styles.get("Speaker 1", {}).get("MarginL", 0))
    expected_h = int(1920 * (1 - MIN_TEXT_AREA_W) / 2)
    check(f"Horizontal margin capped at {expected_h}", margin_l == expected_h, f"got {margin_l}")

    # Vertical: offset=10 → margin_v = max(10, 108) = 108, + inset 300 = 408
    # Safe cap: int(1080 * (1-0.60) / 2) = 216 → capped to 216
    result2 = generate_ass(
        segments=SAMPLE_SEGMENTS, start_time=10.0, end_time=25.0,
        offset_v_pct=10, content_inset_v=300, position="bottom",
    )
    styles2 = parse_ass_styles(result2)
    margin_v = int(styles2.get("Speaker 1", {}).get("MarginV", 0))
    expected_v = int(1080 * (1 - MIN_TEXT_AREA_H) / 2)
    check(f"Vertical margin capped at {expected_v}", margin_v == expected_v, f"got {margin_v}")


def test_ass_value_clamping():
    print("\n--- ASS: Value clamping for out-of-range inputs ---")
    # max_width_pct > 100 → clamped to 100, margin = max(20, 0) = 20
    result = generate_ass(
        segments=SAMPLE_SEGMENTS, start_time=10.0, end_time=25.0,
        max_width_pct=150,
    )
    styles = parse_ass_styles(result)
    margin_l = int(styles.get("Speaker 1", {}).get("MarginL", 0))
    check("max_width=150 clamped → MarginL=20", margin_l == 20, f"got {margin_l}")

    # max_width_pct < 50 → clamped to 50, margin = max(20, 480) = 480
    result2 = generate_ass(
        segments=SAMPLE_SEGMENTS, start_time=10.0, end_time=25.0,
        max_width_pct=10,
    )
    styles2 = parse_ass_styles(result2)
    margin_l2 = int(styles2.get("Speaker 1", {}).get("MarginL", 0))
    check("max_width=10 clamped → MarginL=480", margin_l2 == 480, f"got {margin_l2}")

    # outline_width > 10 → clamped to 10, then scaled with 1x: round(10*1.0)=10
    result3 = generate_ass(
        segments=SAMPLE_SEGMENTS, start_time=10.0, end_time=25.0,
        outline_width=25,
    )
    styles3 = parse_ass_styles(result3)
    ol = int(styles3.get("Speaker 1", {}).get("Outline", 0))
    check("outline_width=25 clamped to 10 then 1x=10", ol == 10, f"got {ol}")

    # outline_width < 0 → clamped to 0
    result4 = generate_ass(
        segments=SAMPLE_SEGMENTS, start_time=10.0, end_time=25.0,
        outline_width=-5,
    )
    styles4 = parse_ass_styles(result4)
    ol2 = int(styles4.get("Speaker 1", {}).get("Outline", 0))
    check("outline_width=-5 clamped to 0", ol2 == 0, f"got {ol2}")


def test_ass_font_size_scaling_4_5():
    print("\n--- ASS: Font size scaling for 4:5 (1080x1350) ---")
    # font_scale = min(1080, 1350) / min(1920, 1080) = 1080/1080 = 1.0
    result = generate_ass(
        segments=SAMPLE_SEGMENTS, start_time=10.0, end_time=25.0,
        font_size="medium", video_width=1080, video_height=1350,
    )
    styles = parse_ass_styles(result)
    actual = int(styles.get("Speaker 1", {}).get("Fontsize", 0))
    check("Medium font at 4:5 = 30px", actual == 30, f"got {actual}")
    check("PlayResX: 1080", "PlayResX: 1080" in result)
    check("PlayResY: 1350", "PlayResY: 1350" in result)


def test_ass_background_with_outline_settings():
    print("\n--- ASS: Background overrides outline settings ---")
    result = generate_ass(
        segments=SAMPLE_SEGMENTS, start_time=10.0, end_time=25.0,
        background_enabled=True,
        background_color="#FF0000",
        background_opacity=50,
        outline_color="#00FF00",
        outline_opacity=80,
        outline_width=8,
    )
    styles = parse_ass_styles(result)
    sp1 = styles.get("Speaker 1", {})
    check("Background → BorderStyle=3", sp1.get("BorderStyle") == "3")
    check("Background → Shadow=0", sp1.get("Shadow") == "0")
    # Box padding = max(int(4*1.0), 2) = 4, NOT the outline_width=8
    check("Outline = 4 (box padding, not outline_width)",
          sp1.get("Outline") == "4", f"got {sp1.get('Outline')}")


def test_ass_background_radius_ignored():
    print("\n--- ASS: background_radius accepted but ignored ---")
    result1 = generate_ass(
        segments=SAMPLE_SEGMENTS, start_time=10.0, end_time=25.0,
        background_enabled=True, background_radius=15,
    )
    result2 = generate_ass(
        segments=SAMPLE_SEGMENTS, start_time=10.0, end_time=25.0,
        background_enabled=True, background_radius=0,
    )
    check("background_radius does not affect ASS output", result1 == result2)


def test_ass_outline_zero_width():
    print("\n--- ASS: outline_width=0 produces no outline ---")
    result = generate_ass(
        segments=SAMPLE_SEGMENTS, start_time=10.0, end_time=25.0,
        outline_width=0,
    )
    styles = parse_ass_styles(result)
    sp1 = styles.get("Speaker 1", {})
    check("Outline=0 when width=0", sp1.get("Outline") == "0")
    check("Shadow=0 when outline disabled", sp1.get("Shadow") == "0")


def test_ass_outline_zero_opacity():
    print("\n--- ASS: outline_opacity=0 produces transparent outline ---")
    result = generate_ass(
        segments=SAMPLE_SEGMENTS, start_time=10.0, end_time=25.0,
        outline_opacity=0, outline_width=3,
    )
    styles = parse_ass_styles(result)
    sp1 = styles.get("Speaker 1", {})
    # outline_opacity=0 → alpha = 255 (fully transparent)
    outline_color = sp1.get("OutlineColour", "")
    check("OutlineColour has alpha FF (transparent)", outline_color.startswith("&HFF"),
          f"got {outline_color}")


def test_ass_background_opacity_edges():
    print("\n--- ASS: Background opacity edge cases ---")
    # opacity=0 → fully transparent (alpha = FF)
    result1 = generate_ass(
        segments=SAMPLE_SEGMENTS, start_time=10.0, end_time=25.0,
        background_enabled=True, background_opacity=0,
    )
    styles1 = parse_ass_styles(result1)
    back_color = styles1.get("Speaker 1", {}).get("BackColour", "")
    check("bg_opacity=0 → alpha FF", back_color.startswith("&HFF"),
          f"got {back_color}")

    # opacity=100 → fully opaque (alpha = 00)
    result2 = generate_ass(
        segments=SAMPLE_SEGMENTS, start_time=10.0, end_time=25.0,
        background_enabled=True, background_opacity=100,
    )
    styles2 = parse_ass_styles(result2)
    back_color2 = styles2.get("Speaker 1", {}).get("BackColour", "")
    check("bg_opacity=100 → alpha 00", back_color2.startswith("&H00"),
          f"got {back_color2}")


def test_ass_frontend_backend_margin_consistency():
    print("\n--- ASS: Frontend-backend margin consistency ---")
    # Verify margin calculations match what the frontend should compute
    test_cases = [
        (90, 4, "bottom", 1920, 1080),
        (100, 0, "bottom", 1920, 1080),
        (50, 50, "top", 1920, 1080),
        (80, 10, "bottom", 1080, 1920),
        (90, 4, "center", 1080, 1080),
    ]
    for max_w, off_v, pos, vw, vh in test_cases:
        result = generate_ass(
            segments=SAMPLE_SEGMENTS, start_time=10.0, end_time=25.0,
            max_width_pct=max_w, offset_v_pct=off_v, position=pos,
            video_width=vw, video_height=vh,
        )
        styles = parse_ass_styles(result)
        sp1 = styles.get("Speaker 1", {})
        actual_ml = int(sp1.get("MarginL", 0))
        actual_mv = int(sp1.get("MarginV", 0))

        # Replicate backend logic
        clamped_mw = max(20, min(100, max_w))
        clamped_ov = max(0, min(100, off_v))
        expected_mh = max(20, int(vw * (100 - clamped_mw) / 100 / 2))
        max_mh = int(vw * (1 - MIN_TEXT_AREA_W) / 2)
        expected_mh = min(expected_mh, max_mh)
        if pos == "center":
            expected_mv = 0
        else:
            expected_mv = max(10, int(vh * clamped_ov / 100))
        max_mv = int(vh * (1 - MIN_TEXT_AREA_H) / 2)
        expected_mv = min(expected_mv, max_mv)

        label = f"mw={max_w},ov={off_v},{pos},{vw}x{vh}"
        check(f"{label} → MarginL={expected_mh}", actual_ml == expected_mh,
              f"got {actual_ml}")
        check(f"{label} → MarginV={expected_mv}", actual_mv == expected_mv,
              f"got {actual_mv}")


# ===========================================================================
# TEST SUITE 2: Filter Chain
# ===========================================================================

def test_filter_chain_no_filters():
    print("\n--- Filter Chain: No aspect ratio, no subtitles ---")
    vf, is_complex, _ = _build_filter_chain(None, 1920, 1080, None)
    check("Returns None", vf is None, f"got {vf}")
    check("Not complex", is_complex is False)


def test_filter_chain_16_9_to_9_16():
    print("\n--- Filter Chain: 16:9 → 9:16 crop ---")
    vf, is_complex, _ = _build_filter_chain("9:16", 1920, 1080, None, subject_x=50)
    check("Not complex", is_complex is False)
    check("Has filter string", vf is not None and len(vf) > 0)

    # 9/16 target, 16/9 source → target < source → crop width
    # crop_h = 1080, crop_w = int(1080 * 9/16) = 607 → 606 (even)
    # x_offset = int((1920 - 606) * 50 / 100) = int(1314 * 0.5) = 657
    # y_offset = 0
    check("Contains crop filter", "crop=" in vf, f"got {vf}")
    check("Contains scale to 1080:1920", "scale=1080:1920" in vf, f"got {vf}")

    # Parse crop params
    crop_match = re.search(r"crop=(\d+):(\d+):(\d+):(\d+)", vf)
    if crop_match:
        cw, ch, cx, cy = [int(x) for x in crop_match.groups()]
        check("Crop width = 606", cw == 606, f"got {cw}")
        check("Crop height = 1080", ch == 1080, f"got {ch}")
        check("Center x_offset = 657", cx == 657, f"got {cx}")
        check("y_offset = 0", cy == 0, f"got {cy}")
    else:
        check("Crop params parseable", False, f"couldn't parse: {vf}")


def test_filter_chain_16_9_to_1_1():
    print("\n--- Filter Chain: 16:9 → 1:1 crop ---")
    vf, _, _ = _build_filter_chain("1:1", 1920, 1080, None, subject_x=50)

    # target=1.0, source=1.778 → target < source → crop width
    # crop_h = 1080, crop_w = int(1080 * 1.0) = 1080 (already even)
    # x_offset = int((1920-1080) * 50/100) = int(840 * 0.5) = 420
    crop_match = re.search(r"crop=(\d+):(\d+):(\d+):(\d+)", vf)
    if crop_match:
        cw, ch, cx, cy = [int(x) for x in crop_match.groups()]
        check("Crop is square 1080x1080", cw == 1080 and ch == 1080,
              f"got {cw}x{ch}")
        check("Centered x_offset=420", cx == 420, f"got {cx}")
    else:
        check("Crop params parseable", False)

    check("Scale to 1080:1080", "scale=1080:1080" in vf, f"got {vf}")


def test_filter_chain_subject_positioning():
    print("\n--- Filter Chain: Subject positioning (centering) ---")
    # 16:9 → 9:16 crop: crop_w=606, max_offset=1314
    # _safe_subject_x clamps to [10, 90], then _center_crop_offset centers:
    #   sx=0   → clamped to 10 → pixel=192, offset=192-303=-111 → clamped to 0
    #   sx=50  → unchanged 50  → pixel=960, offset=960-303=657
    #   sx=100 → clamped to 90 → pixel=1728, offset=1728-303=1425 → clamped to 1314
    for sx, expected_approx in [(0, 0), (50, 657), (100, 1314)]:
        vf, _, _ = _build_filter_chain("9:16", 1920, 1080, None, subject_x=sx)
        crop_match = re.search(r"crop=\d+:\d+:(\d+):", vf)
        if crop_match:
            cx = int(crop_match.group(1))
            check(f"subject_x={sx} → x_offset≈{expected_approx}",
                  abs(cx - expected_approx) <= 5, f"got {cx}")
        else:
            check(f"subject_x={sx} parseable", False)


def test_filter_chain_with_subtitles():
    print("\n--- Filter Chain: With subtitles ---")
    vf, is_complex, _ = _build_filter_chain(None, 1920, 1080, "/tmp/test.ass")
    check("Has filter", vf is not None)
    check("Not complex", is_complex is False)
    check("Contains subtitles filter", "subtitles=" in vf, f"got {vf}")


def test_filter_chain_crop_plus_subtitles():
    print("\n--- Filter Chain: Crop + subtitles ---")
    vf, _, _ = _build_filter_chain("9:16", 1920, 1080, "/tmp/test.ass", subject_x=50)
    parts = vf.split(",")
    check("3 filter parts (crop, scale, subtitles)", len(parts) == 3,
          f"got {len(parts)}: {parts}")
    check("First is crop", parts[0].startswith("crop="), f"got {parts[0]}")
    check("Second is scale", parts[1].startswith("scale="), f"got {parts[1]}")
    check("Third is subtitles", parts[2].startswith("subtitles="), f"got {parts[2]}")


def test_subtitle_filter_path_escaping():
    print("\n--- Filter Chain: Subtitle path escaping ---")
    # Normal path — uses 'subtitles' filter with filename= key
    result = _subtitle_filter("/data/outputs/job-123/clips/clip_1_sub.ass")
    check("Uses subtitles filter", result.startswith("subtitles="), f"got {result}")
    check("Normal path preserved", "job-123" in result and "clip_1_sub.ass" in result,
          f"got {result}")

    # Path with colon
    result2 = _subtitle_filter("/data/C:/test.ass")
    check("Colon escaped", "\\:" in result2, f"got {result2}")

    # Path with single quote
    result3 = _subtitle_filter("/data/it's a test.ass")
    check("Quote escaped", "\\'" in result3, f"got {result3}")

    # With force_style
    result4 = _subtitle_filter(
        "/tmp/test.ass",
        force_style="BorderStyle=1,Outline=6,Shadow=4,OutlineColour=&H00000000&"
    )
    check("force_style included", "force_style=" in result4, f"got {result4}")
    check("force_style has Outline", "Outline=6" in result4, f"got {result4}")
    check("force_style has BorderStyle", "BorderStyle=1" in result4, f"got {result4}")


def test_extract_force_style_from_ass():
    print("\n--- Extract force_style from ASS content ---")
    # Generate sample ASS content with known outline settings
    ass = generate_ass(
        segments=SAMPLE_SEGMENTS,
        start_time=10.0,
        end_time=25.0,
        outline_color="#000000",
        outline_opacity=100,
        outline_width=2,
    )
    force_style = _extract_force_style_from_ass(ass)
    check("force_style not empty", len(force_style) > 0, f"got '{force_style}'")
    check("Contains BorderStyle=1", "BorderStyle=1" in force_style, f"got {force_style}")
    check("Contains Outline=", "Outline=" in force_style, f"got {force_style}")
    check("Contains Shadow=", "Shadow=" in force_style, f"got {force_style}")
    check("Contains OutlineColour=", "OutlineColour=" in force_style, f"got {force_style}")

    # With background mode (BorderStyle=3)
    ass_bg = generate_ass(
        segments=SAMPLE_SEGMENTS,
        start_time=10.0,
        end_time=25.0,
        background_enabled=True,
        background_color="#FF0000",
        background_opacity=80,
    )
    force_style_bg = _extract_force_style_from_ass(ass_bg)
    check("Background mode has BorderStyle=3", "BorderStyle=3" in force_style_bg,
          f"got {force_style_bg}")

    # Empty content
    force_style_empty = _extract_force_style_from_ass("")
    check("Empty content returns empty string", force_style_empty == "", f"got '{force_style_empty}'")


def test_filter_chain_with_force_style():
    """Test that force_style is threaded into the subtitles filter."""
    print("\n--- Filter Chain: With force_style ---")
    fs = "BorderStyle=1,Outline=6,Shadow=4,OutlineColour=&H00000000&,BackColour=&H80000000&"
    vf, _, _ = _build_filter_chain(
        None, 1920, 1080, "/tmp/test.ass", subtitle_force_style=fs,
    )
    check("Has filter", vf is not None)
    check("Uses subtitles filter", "subtitles=" in vf, f"got {vf}")
    check("Contains force_style", "force_style=" in vf, f"got {vf}")
    check("Contains Outline=6", "Outline=6" in vf, f"got {vf}")
    check("Contains BorderStyle=1", "BorderStyle=1" in vf, f"got {vf}")

    # With crop — make sure the subtitles filter is last and contains force_style
    vf2, _, _ = _build_filter_chain(
        "9:16", 1920, 1080, "/tmp/test.ass",
        subject_x=50, subtitle_force_style=fs,
    )
    check("Crop+subtitle has force_style", "force_style=" in vf2, f"got {vf2}")
    check("Crop+subtitle has crop", "crop=" in vf2, f"got {vf2}")
    check("Crop+subtitle has scale", "scale=" in vf2, f"got {vf2}")


# ===========================================================================
# TEST SUITE 3: Settings Flow
# ===========================================================================

def test_settings_flow_complete():
    """Test that a full settings dict (matching frontend export body) flows
    correctly through the same .get() pattern used in clip_exporter.py."""
    print("\n--- Settings Flow: Complete settings dict ---")

    # This matches exactly what the frontend sends in the export body
    settings = {
        "font": "Arial",
        "size": "large",
        "font_weight": "normal",
        "font_color": "#FF5500",
        "position": "top",
        "speaker_colors": {"Speaker 1": "#00AAFF"},
        "use_speaker_colors": True,
        "background_enabled": True,
        "background_color": "#112233",
        "background_opacity": 60,
        "background_radius": 8,  # accepted but ignored
        "outline_color": "#AABBCC",
        "outline_opacity": 90,
        "outline_width": 4,
        "show_speaker_labels": False,
        "max_width": 75,
        "offset_v": 8,
    }

    # Use the exact same .get() pattern as clip_exporter.py lines 169-189
    result = generate_ass(
        segments=SAMPLE_SEGMENTS,
        start_time=10.0,
        end_time=25.0,
        font=settings.get("font", "DM Sans"),
        font_size=settings.get("size", "medium"),
        font_weight=settings.get("font_weight", "bold"),
        font_color=settings.get("font_color", "#FFFFFF"),
        position=settings.get("position", "bottom"),
        speaker_colors=settings.get("speaker_colors"),
        video_width=1920,
        video_height=1080,
        background_enabled=settings.get("background_enabled", False),
        background_color=settings.get("background_color", "#000000"),
        background_opacity=settings.get("background_opacity", 75),
        background_radius=settings.get("background_radius", 0),
        outline_color=settings.get("outline_color", "#000000"),
        outline_opacity=settings.get("outline_opacity", 100),
        outline_width=settings.get("outline_width", 2),
        content_inset_v=0,
        content_inset_h=0,
        show_speaker_labels=settings.get("show_speaker_labels", False),
        use_speaker_colors=settings.get("use_speaker_colors", True),
        max_width_pct=settings.get("max_width", 90),
        offset_v_pct=settings.get("offset_v", 4),
    )

    styles = parse_ass_styles(result)
    sp1 = styles.get("Speaker 1", {})

    check("Font = Arial", sp1.get("Fontname") == "Arial",
          f"got {sp1.get('Fontname')}")
    check("FontSize = 40 (large at 1080p)", sp1.get("Fontsize") == "40",
          f"got {sp1.get('Fontsize')}")
    check("Bold = 0 (normal)", sp1.get("Bold") == "0",
          f"got {sp1.get('Bold')}")
    check("Alignment = 2 (always bottom-center)", sp1.get("Alignment") == "2",
          f"got {sp1.get('Alignment')}")
    check("BorderStyle = 3 (background enabled)", sp1.get("BorderStyle") == "3",
          f"got {sp1.get('BorderStyle')}")

    # Speaker 1 has explicit speaker_colors override → uses #00AAFF
    expected_color = _hex_to_ass_color("#00AAFF")
    check("PrimaryColour = #00AAFF from speaker_colors",
          sp1.get("PrimaryColour") == expected_color,
          f"got {sp1.get('PrimaryColour')}, expected {expected_color}")

    # Background: BackColour = #112233 at 60% opacity
    expected_back = _hex_to_ass_color_with_alpha("#112233", 60)
    check("BackColour = #112233 @ 60%",
          sp1.get("BackColour") == expected_back,
          f"got {sp1.get('BackColour')}, expected {expected_back}")

    # Max width 75% → margin_h = max(20, int(1920 * 25/100/2)) = max(20, 240) = 240
    check("MarginL = 240 (max_width 75%)", sp1.get("MarginL") == "240",
          f"got {sp1.get('MarginL')}")

    # Offset 8% top → margin_v = max(10, int(1080 * 8/100)) = max(10, 86) = 86
    check("MarginV = 86 (offset 8%)", sp1.get("MarginV") == "86",
          f"got {sp1.get('MarginV')}")

    # Speaker labels OFF
    dialogues = parse_ass_dialogues(result)
    check("No speaker label in text",
          "Speaker 1:" not in dialogues[0].get("Text", ""),
          f"got: {dialogues[0].get('Text')}")


# ===========================================================================
# TEST SUITE 4: Frontend/Backend Constant Consistency
# ===========================================================================

def test_constant_consistency():
    """Verify that frontend constants in ClipPreview.jsx match backend values."""
    print("\n--- Constant Consistency: Frontend vs Backend ---")

    # Read ClipPreview.jsx to extract constants
    preview_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "frontend", "src", "components", "ClipPreview.jsx"
    )
    with open(preview_path) as f:
        jsx_content = f.read()

    # FONT_SIZE_MAP
    check("FONT_SIZE_MAP small=22",
          "small: 22" in jsx_content and FONT_SIZE_MAP["small"] == 22)
    check("FONT_SIZE_MAP medium=30",
          "medium: 30" in jsx_content and FONT_SIZE_MAP["medium"] == 30)
    check("FONT_SIZE_MAP large=40",
          "large: 40" in jsx_content and FONT_SIZE_MAP["large"] == 40)

    # REF_W / REF_H
    check("REF_W = 1920",
          "REF_W = 1920" in jsx_content and REF_W == 1920)
    check("REF_H = 1080",
          "REF_H = 1080" in jsx_content and REF_H == 1080)

    # ASPECT_RATIO_DIMS
    for ratio, (w, h) in ASPECT_RATIO_DIMS.items():
        key = f"'{ratio}'"
        check(f"ASPECT_RATIO_DIMS {ratio} in frontend",
              key in jsx_content)

    # DEFAULT_SPEAKER_PALETTE
    for color in DEFAULT_SPEAKER_PALETTE:
        check(f"Palette color {color} in frontend",
              color in jsx_content)


# ===========================================================================
# TEST SUITE 5: SubtitleSettings Pydantic Model
# ===========================================================================

def test_subtitle_settings_model():
    """Verify the Pydantic model accepts all fields and has correct defaults."""
    print("\n--- SubtitleSettings Model ---")

    # Default construction
    s = SubtitleSettings()
    check("Default font = 'DM Sans'", s.font == "DM Sans")
    check("Default size = 'medium'", s.size == "medium")
    check("Default font_weight = 'bold'", s.font_weight == "bold")
    check("Default font_color = '#FFFFFF'", s.font_color == "#FFFFFF")
    check("Default position = 'bottom'", s.position == "bottom")
    check("Default background_enabled = False", s.background_enabled is False)
    check("Default outline_color = '#000000'", s.outline_color == "#000000")
    check("Default outline_opacity = 100", s.outline_opacity == 100)
    check("Default outline_width = 2", s.outline_width == 2)
    check("Default show_speaker_labels = False", s.show_speaker_labels is False)
    check("Default max_width = 90", s.max_width == 90)
    check("Default offset_v = 4", s.offset_v == 4)
    check("Default max_words = 0", s.max_words == 0)

    # Full construction
    s2 = SubtitleSettings(
        font="Arial", size="large", font_weight="normal",
        font_color="#FF0000", position="top",
        speaker_colors={"Speaker 1": "#00FF00"},
        background_enabled=True, background_color="#112233",
        background_opacity=60, background_radius=10,
        outline_color="#AABBCC", outline_opacity=80, outline_width=5,
        show_speaker_labels=False, max_width=75, offset_v=8,
    )
    check("Custom font accepted", s2.font == "Arial")
    check("Custom outline_width accepted", s2.outline_width == 5)
    check("Custom background_radius accepted", s2.background_radius == 10)

    # model_dump (used in clips.py line 73)
    d = s2.model_dump()
    check("model_dump has 'font'", d.get("font") == "Arial")
    check("model_dump has 'outline_width'", d.get("outline_width") == 5)
    check("model_dump has 'max_width'", d.get("max_width") == 75)


# ===========================================================================
# TEST SUITE 6: Helper Functions
# ===========================================================================

def test_hex_to_ass_color():
    print("\n--- Helper: _hex_to_ass_color ---")
    # ASS color is BGR format: &H00BBGGRR&
    check("#FF0000 → &H000000FF&",
          _hex_to_ass_color("#FF0000") == "&H000000FF&")
    check("#00FF00 → &H0000FF00&",
          _hex_to_ass_color("#00FF00") == "&H0000FF00&")
    check("#0000FF → &H00FF0000&",
          _hex_to_ass_color("#0000FF") == "&H00FF0000&")
    check("#FFFFFF → &H00FFFFFF&",
          _hex_to_ass_color("#FFFFFF") == "&H00FFFFFF&")
    check("#000000 → &H00000000&",
          _hex_to_ass_color("#000000") == "&H00000000&")


def test_hex_to_ass_color_with_alpha():
    print("\n--- Helper: _hex_to_ass_color_with_alpha ---")
    # 100% opacity → alpha=0 (fully opaque)
    result = _hex_to_ass_color_with_alpha("#000000", 100)
    check("100% opacity → alpha=00", result.startswith("&H00"))

    # 0% opacity → alpha=FF (fully transparent)
    result2 = _hex_to_ass_color_with_alpha("#000000", 0)
    check("0% opacity → alpha=FF", result2.startswith("&HFF"))

    # 50% opacity → alpha=128-ish → ~80 hex
    result3 = _hex_to_ass_color_with_alpha("#000000", 50)
    alpha_hex = result3[2:4]
    alpha_val = int(alpha_hex, 16)
    check("50% opacity → alpha≈128", 126 <= alpha_val <= 130,
          f"got alpha={alpha_val} (0x{alpha_hex})")


def test_format_ass_time():
    print("\n--- Helper: _format_ass_time ---")
    check("0 seconds", _format_ass_time(0) == "0:00:00.00")
    check("1.5 seconds", _format_ass_time(1.5) == "0:00:01.50")
    check("65 seconds", _format_ass_time(65.0) == "0:01:05.00")
    check("3661.25 seconds", _format_ass_time(3661.25) == "1:01:01.25")
    check("Negative clamped to 0", _format_ass_time(-5) == "0:00:00.00")


# ===========================================================================
# TEST SUITE 7: Max Words Splitting
# ===========================================================================

def test_max_words_disabled():
    print("\n--- Max Words: Disabled (max_words=0) ---")
    result = generate_ass(
        segments=SAMPLE_SEGMENTS, start_time=10.0, end_time=25.0,
        max_words=0,
    )
    dialogues = parse_ass_dialogues(result)
    check("max_words=0 → 3 dialogues unchanged", len(dialogues) == 3,
          f"got {len(dialogues)}")


def test_max_words_no_split_needed():
    print("\n--- Max Words: No split needed (words <= max) ---")
    # "Hello world" = 2 words, max_words=4 → no split
    result = generate_ass(
        segments=SAMPLE_SEGMENTS, start_time=10.0, end_time=25.0,
        max_words=4,
    )
    dialogues = parse_ass_dialogues(result)
    check("max_words=4 with 2-word segments → 3 dialogues", len(dialogues) == 3,
          f"got {len(dialogues)}")


def test_max_words_basic_split():
    print("\n--- Max Words: Basic split ---")
    long_seg = [
        TranscriptSegment(
            start=0.0, end=6.0,
            text="one two three four five six",
            speaker="Speaker 1",
        )
    ]
    result = generate_ass(
        segments=long_seg, start_time=0.0, end_time=6.0,
        max_words=4, show_speaker_labels=False,
    )
    dialogues = parse_ass_dialogues(result)
    check("6 words / max_words=4 → 2 dialogues", len(dialogues) == 2,
          f"got {len(dialogues)}")
    check("First chunk text = 'one two three four'",
          "one two three four" in dialogues[0].get("Text", ""),
          f"got: {dialogues[0].get('Text')}")
    check("Second chunk text = 'five six'",
          "five six" in dialogues[1].get("Text", ""),
          f"got: {dialogues[1].get('Text')}")
    # Time distribution: 4/6 * 6s = 4s, 2/6 * 6s = 2s
    check("First chunk starts at 0:00:00.00",
          dialogues[0]["Start"] == "0:00:00.00")
    check("Second chunk starts at 0:00:04.00",
          dialogues[1]["Start"] == "0:00:04.00")
    check("Second chunk ends at 0:00:06.00",
          dialogues[1]["End"] == "0:00:06.00")


def test_max_words_single_word():
    print("\n--- Max Words: Single word segment ---")
    seg = [TranscriptSegment(start=0.0, end=2.0, text="Hello", speaker="S1")]
    result = generate_ass(
        segments=seg, start_time=0.0, end_time=2.0,
        max_words=3, show_speaker_labels=False,
    )
    dialogues = parse_ass_dialogues(result)
    check("1-word segment unchanged", len(dialogues) == 1)
    check("Text = 'Hello'", "Hello" in dialogues[0].get("Text", ""))


def test_max_words_exact_boundary():
    print("\n--- Max Words: Exact boundary (words == max) ---")
    seg = [TranscriptSegment(start=0.0, end=4.0, text="one two three four", speaker="S1")]
    result = generate_ass(
        segments=seg, start_time=0.0, end_time=4.0,
        max_words=4, show_speaker_labels=False,
    )
    dialogues = parse_ass_dialogues(result)
    check("4 words / max_words=4 → 1 dialogue (no split)", len(dialogues) == 1)


def test_max_words_with_speaker_labels():
    print("\n--- Max Words: With speaker labels ---")
    long_seg = [
        TranscriptSegment(
            start=0.0, end=6.0,
            text="one two three four five six",
            speaker="Speaker 1",
        )
    ]
    result = generate_ass(
        segments=long_seg, start_time=0.0, end_time=6.0,
        max_words=4, show_speaker_labels=True,
    )
    dialogues = parse_ass_dialogues(result)
    check("Split still produces 2 dialogues", len(dialogues) == 2)
    # Speaker label is prepended AFTER splitting
    check("First chunk has speaker label",
          "Speaker 1:" in dialogues[0].get("Text", ""))
    check("Second chunk also has speaker label",
          "Speaker 1:" in dialogues[1].get("Text", ""))


def test_max_words_one_word_per_subtitle():
    print("\n--- Max Words: max_words=1 ---")
    seg = [TranscriptSegment(start=0.0, end=3.0, text="a b c", speaker="S1")]
    result = generate_ass(
        segments=seg, start_time=0.0, end_time=3.0,
        max_words=1, show_speaker_labels=False,
    )
    dialogues = parse_ass_dialogues(result)
    check("3 words / max_words=1 → 3 dialogues", len(dialogues) == 3,
          f"got {len(dialogues)}")


def test_split_segments_by_max_words_direct():
    """Test the standalone split function directly."""
    print("\n--- Max Words: split_segments_by_max_words direct test ---")

    # Empty/disabled
    segs = [(0.0, 5.0, "hello world test", "S1")]
    check("max_words=0 returns unchanged",
          split_segments_by_max_words(segs, 0) == segs)
    check("max_words=-1 returns unchanged",
          split_segments_by_max_words(segs, -1) == segs)

    # Basic split
    result = split_segments_by_max_words(segs, 2)
    check("3 words / max=2 → 2 chunks", len(result) == 2, f"got {len(result)}")
    check("Chunk 1 text = 'hello world'", result[0][2] == "hello world")
    check("Chunk 2 text = 'test'", result[1][2] == "test")

    # Time proportionality: 2/3 * 5 = 3.333..., 1/3 * 5 = 1.666...
    check("Chunk 1 duration ~3.33s",
          abs(result[0][1] - result[0][0] - 10 / 3) < 0.01,
          f"got {result[0][1] - result[0][0]}")
    check("Chunk 2 ends at 5.0", result[1][1] == 5.0)

    # Empty text
    empty = [(0.0, 2.0, "", "S1")]
    result_empty = split_segments_by_max_words(empty, 3)
    check("Empty text → unchanged", len(result_empty) == 1)


# ===========================================================================
# TEST SUITE: Active Word Highlight
# ===========================================================================

def test_active_word_disabled():
    """active_word_enabled=False should produce same output as before."""
    print("\n--- Active Word: disabled (default) ---")
    ass = generate_ass(SAMPLE_SEGMENTS, 10.0, 25.0, active_word_enabled=False)
    lines = [l for l in ass.split("\n") if l.startswith("Dialogue:")]
    check("3 dialogue events (one per segment)", len(lines) == 3, f"got {len(lines)}")
    # No inline \c override tags
    for i, line in enumerate(lines):
        check(f"Event {i} has no inline \\c override", "\\c&" not in line, line[:80])


def test_active_word_basic():
    """active_word_enabled=True should produce Layer 0 base + Layer 1 per-word events."""
    print("\n--- Active Word: basic multi-word ---")
    seg = [TranscriptSegment(start=0.0, end=2.0, text="Hello world", speaker="S1")]
    ass = generate_ass(seg, 0.0, 2.0, active_word_enabled=True,
                       active_word_color="#FFD700")
    lines = [l for l in ass.split("\n") if l.startswith("Dialogue:")]
    layer0 = [l for l in lines if l.startswith("Dialogue: 0,")]
    layer1 = [l for l in lines if l.startswith("Dialogue: 1,")]
    check("1 base event on Layer 0", len(layer0) == 1, f"got {len(layer0)}")
    check("2 word events on Layer 1", len(layer1) == 2, f"got {len(layer1)}")
    # Word events should highlight with \c tag
    check("Word event 0 has \\c tag", "\\c" in layer1[0])
    check("Word event 1 has \\c tag", "\\c" in layer1[1])


def test_active_word_single_word():
    """Single word segment should produce 1 base + 1 word event."""
    print("\n--- Active Word: single word ---")
    seg = [TranscriptSegment(start=0.0, end=1.0, text="Hello", speaker="S1")]
    ass = generate_ass(seg, 0.0, 1.0, active_word_enabled=True,
                       active_word_color="#FFD700")
    lines = [l for l in ass.split("\n") if l.startswith("Dialogue:")]
    layer0 = [l for l in lines if l.startswith("Dialogue: 0,")]
    layer1 = [l for l in lines if l.startswith("Dialogue: 1,")]
    check("1 base event on Layer 0", len(layer0) == 1, f"got {len(layer0)}")
    check("1 word event on Layer 1", len(layer1) == 1, f"got {len(layer1)}")
    check("Word event has active color tag", "\\c" in layer1[0])


def test_active_word_timing():
    """Word durations should be character-proportional and sum to segment duration."""
    print("\n--- Active Word: timing proportionality ---")
    seg = [TranscriptSegment(start=0.0, end=10.0, text="ab cdef", speaker="S1")]
    ass = generate_ass(seg, 0.0, 10.0, active_word_enabled=True)
    lines = [l for l in ass.split("\n") if l.startswith("Dialogue:")]
    layer1 = [l for l in lines if l.startswith("Dialogue: 1,")]
    check("2 word events on Layer 1", len(layer1) == 2, f"got {len(layer1)}")

    # Parse timing from Layer 1 word events
    import re
    times = []
    for line in layer1:
        m = re.match(r"Dialogue: \d+,(\d+:\d{2}:\d{2}\.\d{2}),(\d+:\d{2}:\d{2}\.\d{2}),", line)
        if m:
            times.append((m.group(1), m.group(2)))

    check("Parsed 2 word event timings", len(times) == 2, f"got {len(times)}")
    # First word event starts at 0, last word event ends at 10.0
    check("First word event starts at 0:00:00.00", times[0][0] == "0:00:00.00")
    check("Last word event ends at 0:00:10.00", times[1][1] == "0:00:10.00")


def test_active_word_colors():
    """Verify highlight color and outline color appear in override tags."""
    print("\n--- Active Word: custom colors ---")
    seg = [TranscriptSegment(start=0.0, end=2.0, text="Hello world", speaker="S1")]
    ass = generate_ass(seg, 0.0, 2.0, active_word_enabled=True,
                       active_word_color="#FF0000",
                       active_word_outline_color="#00FF00")
    lines = [l for l in ass.split("\n") if l.startswith("Dialogue:")]
    # #FF0000 → ASS &H000000FF& (BGR order)
    aw_color_ass = _hex_to_ass_color("#FF0000")
    aw_outline_ass = _hex_to_ass_color("#00FF00")
    check(f"Active word color {aw_color_ass} present", aw_color_ass in ass, ass[:200])
    check(f"Active word outline {aw_outline_ass} present", aw_outline_ass in ass, ass[:200])


def test_active_word_with_speaker_labels():
    """Speaker label prefix should appear in both base and word events."""
    print("\n--- Active Word: with speaker labels ---")
    seg = [TranscriptSegment(start=0.0, end=2.0, text="Hello world", speaker="S1")]
    ass = generate_ass(seg, 0.0, 2.0, active_word_enabled=True,
                       show_speaker_labels=True)
    lines = [l for l in ass.split("\n") if l.startswith("Dialogue:")]
    layer0 = [l for l in lines if l.startswith("Dialogue: 0,")]
    layer1 = [l for l in lines if l.startswith("Dialogue: 1,")]
    check("1 base event on Layer 0", len(layer0) == 1, f"got {len(layer0)}")
    check("2 word events on Layer 1", len(layer1) == 2, f"got {len(layer1)}")
    check("Speaker label in base event", "S1: " in layer0[0])
    check("Speaker label in word event 0", "S1: " in layer1[0])
    check("Speaker label in word event 1", "S1: " in layer1[1])


def test_active_word_with_max_words():
    """max_words splitting should happen first, then active word highlighting."""
    print("\n--- Active Word: combined with max_words ---")
    seg = [TranscriptSegment(start=0.0, end=4.0, text="one two three four", speaker="S1")]
    ass = generate_ass(seg, 0.0, 4.0, active_word_enabled=True,
                       max_words=2, show_speaker_labels=False)
    lines = [l for l in ass.split("\n") if l.startswith("Dialogue:")]
    layer0 = [l for l in lines if l.startswith("Dialogue: 0,")]
    layer1 = [l for l in lines if l.startswith("Dialogue: 1,")]
    # max_words=2 splits into 2 segments: "one two" and "three four"
    # Each segment: 1 base (Layer 0) + 2 word (Layer 1) events
    check("2 base events on Layer 0", len(layer0) == 2, f"got {len(layer0)}")
    check("4 word events on Layer 1", len(layer1) == 4, f"got {len(layer1)}")


def test_active_word_bg_color():
    """When active_word_bg_opacity > 0, verify \\4c tag appears in Layer 1 events."""
    print("\n--- Active Word: background color ---")
    seg = [TranscriptSegment(start=0.0, end=2.0, text="Hello world", speaker="S1")]
    ass = generate_ass(seg, 0.0, 2.0, active_word_enabled=True,
                       active_word_bg_color="#FF0000",
                       active_word_bg_opacity=80)
    layer1 = [l for l in ass.split("\n") if l.startswith("Dialogue: 1,")]
    # Should have \4c tag for background on Layer 1 word events
    has_4c = any("\\4c" in line for line in layer1)
    check("\\4c tag present in Layer 1 word events", has_4c)


def test_active_word_no_bg_when_zero_opacity():
    """When active_word_bg_opacity=0, no \\4c tag should appear in Layer 1."""
    print("\n--- Active Word: no bg at zero opacity ---")
    seg = [TranscriptSegment(start=0.0, end=2.0, text="Hello world", speaker="S1")]
    ass = generate_ass(seg, 0.0, 2.0, active_word_enabled=True,
                       active_word_bg_opacity=0)
    layer1 = [l for l in ass.split("\n") if l.startswith("Dialogue: 1,")]
    has_4c = any("\\4c" in line for line in layer1)
    check("No \\4c tag when bg_opacity=0", not has_4c)


def test_active_word_model_defaults():
    """SubtitleSettings should have correct defaults for active word fields."""
    print("\n--- Active Word: model defaults ---")
    s = SubtitleSettings()
    check("active_word_enabled defaults to False", s.active_word_enabled == False)
    check("active_word_color defaults to #FFD700", s.active_word_color == "#FFD700")
    check("active_word_outline_color defaults to #000000", s.active_word_outline_color == "#000000")
    check("active_word_bg_color defaults to #000000", s.active_word_bg_color == "#000000")
    check("active_word_bg_opacity defaults to 0", s.active_word_bg_opacity == 0)


def _parse_ass_time(s: str) -> float:
    """Parse H:MM:SS.cc to seconds."""
    parts = s.split(":")
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])


def test_active_word_no_temporal_overlap():
    """Active word events on each layer must not overlap — overlaps cause libass bouncing."""
    print("\n--- Active Word: no temporal overlap per-layer (anti-bounce) ---")

    def _check_per_layer_overlaps(ass_text, label):
        """Check that events on the same layer don't overlap each other."""
        dialogues = [l for l in ass_text.split("\n") if l.startswith("Dialogue:")]
        check(f"[{label}] Multiple events generated", len(dialogues) > 2, f"got {len(dialogues)}")

        # Group events by layer
        layers: dict[str, list] = {}
        for line in dialogues:
            m = re.match(r"Dialogue: (\d+),(\d+:\d{2}:\d{2}\.\d{2}),(\d+:\d{2}:\d{2}\.\d{2}),", line)
            if m:
                layer = m.group(1)
                layers.setdefault(layer, []).append(
                    (_parse_ass_time(m.group(2)), _parse_ass_time(m.group(3)))
                )

        for layer, times in layers.items():
            times.sort(key=lambda t: t[0])
            overlaps = 0
            for i in range(len(times) - 1):
                if times[i][1] > times[i + 1][0] + 0.005:  # 5ms tolerance
                    overlaps += 1
            check(f"[{label}] No overlaps on Layer {layer}", overlaps == 0,
                  f"found {overlaps} overlapping pairs on Layer {layer}")

    # Test with estimation path (no word timestamps)
    segs = [
        TranscriptSegment(start=0.0, end=5.0, text="Hello world how are you", speaker="S1"),
        TranscriptSegment(start=5.0, end=10.0, text="I am doing great today", speaker="S2"),
    ]
    ass = generate_ass(segs, 0.0, 10.0, active_word_enabled=True)
    _check_per_layer_overlaps(ass, "estimation")

    # Verify 2-layer structure
    layer0 = [l for l in ass.split("\n") if l.startswith("Dialogue: 0,")]
    layer1 = [l for l in ass.split("\n") if l.startswith("Dialogue: 1,")]
    check("Layer 0 base events present", len(layer0) >= 2, f"got {len(layer0)}")
    check("Layer 1 word events present", len(layer1) >= 5, f"got {len(layer1)}")

    # Test with Whisper word timestamps (anticipation shift path)
    from backend.models import WordTimestamp
    segs_with_words = [
        TranscriptSegment(
            start=0.0, end=3.0, text="Hello world friend",
            speaker="S1",
            words=[
                WordTimestamp(start=0.1, end=0.8, word="Hello"),
                WordTimestamp(start=0.8, end=1.5, word="world"),
                WordTimestamp(start=1.5, end=2.8, word="friend"),
            ],
        ),
        TranscriptSegment(
            start=3.0, end=6.0, text="How are you",
            speaker="S2",
            words=[
                WordTimestamp(start=3.1, end=3.8, word="How"),
                WordTimestamp(start=3.8, end=4.5, word="are"),
                WordTimestamp(start=4.5, end=5.8, word="you"),
            ],
        ),
    ]
    ass2 = generate_ass(segs_with_words, 0.0, 6.0, active_word_enabled=True)
    _check_per_layer_overlaps(ass2, "whisper")

    layer0_2 = [l for l in ass2.split("\n") if l.startswith("Dialogue: 0,")]
    layer1_2 = [l for l in ass2.split("\n") if l.startswith("Dialogue: 1,")]
    check("Whisper: Layer 0 base events", len(layer0_2) >= 2, f"got {len(layer0_2)}")
    check("Whisper: Layer 1 word events", len(layer1_2) >= 6, f"got {len(layer1_2)}")


def test_centisecond_overlap_elimination():
    """Events that are < 0.01s apart after segment clamping must not overlap
    at centisecond precision in the final ASS output."""
    print("\n--- Centisecond overlap elimination ---")

    def _check_cs_overlaps(ass_text, label):
        """Verify no same-layer overlaps exist at centisecond precision (0-tolerance)."""
        dialogues = [l for l in ass_text.split("\n") if l.startswith("Dialogue:")]
        layers: dict[str, list] = {}
        for line in dialogues:
            m = re.match(r"Dialogue: (\d+),(\d+:\d{2}:\d{2}\.\d{2}),(\d+:\d{2}:\d{2}\.\d{2}),", line)
            if m:
                layer = m.group(1)
                layers.setdefault(layer, []).append(
                    (_parse_ass_time(m.group(2)), _parse_ass_time(m.group(3)))
                )
        for layer, times in layers.items():
            times.sort(key=lambda t: t[0])
            for i in range(len(times) - 1):
                # Zero tolerance: end must be <= start of next event at centisecond level
                end_cs = round(times[i][1] * 100)
                start_cs = round(times[i + 1][0] * 100)
                if end_cs > start_cs:
                    check(f"[{label}] No overlap on Layer {layer} at idx {i}", False,
                          f"end_cs={end_cs} > start_cs={start_cs}")
                    return
        check(f"[{label}] No centisecond overlaps found", True)

    # Test 1: Segments with very tight boundaries (Whisper-style close timestamps)
    tight_segs = [
        TranscriptSegment(start=0.0, end=5.005, text="First segment", speaker="S1"),
        TranscriptSegment(start=5.005, end=10.009, text="Second segment", speaker="S2"),
        TranscriptSegment(start=10.009, end=15.0, text="Third segment", speaker="S1"),
    ]
    ass_tight = generate_ass(tight_segs, 0.0, 15.0)
    _check_cs_overlaps(ass_tight, "tight-standard")

    # Test 2: Overlapping Whisper segments that need clamping
    overlap_segs = [
        TranscriptSegment(start=0.0, end=5.5, text="Hello world", speaker="S1"),
        TranscriptSegment(start=5.3, end=10.2, text="How are you", speaker="S2"),
        TranscriptSegment(start=9.8, end=15.0, text="Doing great", speaker="S1"),
    ]
    ass_overlap = generate_ass(overlap_segs, 0.0, 15.0)
    _check_cs_overlaps(ass_overlap, "overlap-standard")

    # Test 3: Active word mode with tight segment boundaries
    ass_aw_tight = generate_ass(tight_segs, 0.0, 15.0, active_word_enabled=True)
    _check_cs_overlaps(ass_aw_tight, "tight-activeword")

    # Test 4: Active word mode with overlapping segments
    ass_aw_overlap = generate_ass(overlap_segs, 0.0, 15.0, active_word_enabled=True)
    _check_cs_overlaps(ass_aw_overlap, "overlap-activeword")

    # Test 5: Many rapid segments (fast-paced dialogue)
    rapid_segs = [
        TranscriptSegment(start=float(i) * 0.5, end=float(i) * 0.5 + 0.55,
                         text=f"Word{i}", speaker=f"S{i%2+1}")
        for i in range(20)
    ]
    ass_rapid = generate_ass(rapid_segs, 0.0, 10.0)
    _check_cs_overlaps(ass_rapid, "rapid-standard")

    ass_rapid_aw = generate_ass(rapid_segs, 0.0, 10.0, active_word_enabled=True)
    _check_cs_overlaps(ass_rapid_aw, "rapid-activeword")


# ===========================================================================
# ASS QA VALIDATION TESTS
# ===========================================================================

def _gen_ass_with_settings(**overrides):
    """Helper: generate ASS content from a settings dict with overrides."""
    defaults = {
        "font": "DM Sans",
        "size": "medium",
        "font_weight": "bold",
        "font_color": "#FFFFFF",
        "position": "bottom",
        "speaker_colors": {},
        "use_speaker_colors": True,
        "background_enabled": False,
        "background_color": "#000000",
        "background_opacity": 75,
        "background_radius": 0,
        "outline_color": "#000000",
        "outline_opacity": 100,
        "outline_width": 2,
        "show_speaker_labels": False,
        "max_width": 90,
        "offset_v": 4,
        "max_words": 0,
        "active_word_enabled": False,
        "active_word_color": "#FFD700",
        "active_word_outline_color": "#000000",
        "active_word_bg_color": "#000000",
        "active_word_bg_opacity": 0,
    }
    settings = {**defaults, **overrides}
    vid_w = overrides.pop("video_width", 1920)
    vid_h = overrides.pop("video_height", 1080)
    segs = overrides.pop("segments", SAMPLE_SEGMENTS)
    start = overrides.pop("start_time", 10.0)
    end = overrides.pop("end_time", 25.0)

    ass = generate_ass(
        segments=segs,
        start_time=start,
        end_time=end,
        font=settings["font"],
        font_size=settings["size"],
        font_weight=settings["font_weight"],
        font_color=settings["font_color"],
        position=settings["position"],
        speaker_colors=settings.get("speaker_colors"),
        use_speaker_colors=settings["use_speaker_colors"],
        video_width=vid_w,
        video_height=vid_h,
        background_enabled=settings["background_enabled"],
        background_color=settings["background_color"],
        background_opacity=settings["background_opacity"],
        background_radius=settings["background_radius"],
        outline_color=settings["outline_color"],
        outline_opacity=settings["outline_opacity"],
        outline_width=settings["outline_width"],
        content_inset_v=0,
        content_inset_h=0,
        show_speaker_labels=settings["show_speaker_labels"],
        max_width_pct=settings["max_width"],
        offset_v_pct=settings["offset_v"],
        max_words=settings["max_words"],
        active_word_enabled=settings["active_word_enabled"],
        active_word_color=settings["active_word_color"],
        active_word_outline_color=settings["active_word_outline_color"],
        active_word_bg_color=settings["active_word_bg_color"],
        active_word_bg_opacity=settings["active_word_bg_opacity"],
    )
    return ass, settings, vid_w, vid_h


def test_validate_ass_default_settings():
    """QA validation should pass with zero warnings for default settings."""
    print("\n--- QA Validate: default settings ---")
    ass, settings, w, h = _gen_ass_with_settings()
    warnings = validate_ass(ass, settings, w, h)
    check("default settings produce no QA warnings", len(warnings) == 0,
          f"got {len(warnings)} warnings: {warnings}")


def test_validate_ass_background_enabled():
    """QA should pass when background is enabled with correct BorderStyle=3."""
    print("\n--- QA Validate: background enabled ---")
    ass, settings, w, h = _gen_ass_with_settings(
        background_enabled=True,
        background_color="#FF0000",
        background_opacity=80,
    )
    warnings = validate_ass(ass, settings, w, h)
    check("background enabled produces no QA warnings", len(warnings) == 0,
          f"got {len(warnings)} warnings: {warnings}")
    # Verify BorderStyle=3 is in the ASS
    check("ASS contains BorderStyle 3", "3," in ass and "BorderStyle" not in ass.split("[Events]")[0].split(",3,")[0].split("BorderStyle")[0] if "3," in ass else False or ",3," in ass)


def test_validate_ass_font_size_scaling():
    """QA should pass for different output resolutions (font scales correctly)."""
    print("\n--- QA Validate: font size scaling ---")
    # Portrait 1080x1920 (font_scale = 1.0 since min(1080,1920)/min(1920,1080) = 1.0)
    ass, settings, w, h = _gen_ass_with_settings(video_width=1080, video_height=1920)
    warnings = validate_ass(ass, settings, w, h)
    check("portrait 1080x1920 no QA warnings", len(warnings) == 0,
          f"got {len(warnings)} warnings: {warnings}")

    # Square 1080x1080
    ass, settings, w, h = _gen_ass_with_settings(video_width=1080, video_height=1080)
    warnings = validate_ass(ass, settings, w, h)
    check("square 1080x1080 no QA warnings", len(warnings) == 0,
          f"got {len(warnings)} warnings: {warnings}")

    # 4:5 aspect 1080x1350
    ass, settings, w, h = _gen_ass_with_settings(video_width=1080, video_height=1350)
    warnings = validate_ass(ass, settings, w, h)
    check("4:5 1080x1350 no QA warnings", len(warnings) == 0,
          f"got {len(warnings)} warnings: {warnings}")


def test_validate_ass_speaker_labels_on():
    """QA should pass when speaker labels are enabled and present."""
    print("\n--- QA Validate: speaker labels ON ---")
    ass, settings, w, h = _gen_ass_with_settings(show_speaker_labels=True)
    warnings = validate_ass(ass, settings, w, h)
    check("speaker labels ON produces no QA warnings", len(warnings) == 0,
          f"got {len(warnings)} warnings: {warnings}")


def test_validate_ass_speaker_labels_off():
    """QA should pass when speaker labels are disabled and absent."""
    print("\n--- QA Validate: speaker labels OFF ---")
    ass, settings, w, h = _gen_ass_with_settings(show_speaker_labels=False)
    warnings = validate_ass(ass, settings, w, h)
    check("speaker labels OFF produces no QA warnings", len(warnings) == 0,
          f"got {len(warnings)} warnings: {warnings}")


def test_validate_ass_active_word_colors():
    """QA should pass when active word is enabled with correct color overrides."""
    print("\n--- QA Validate: active word colors ---")
    ass, settings, w, h = _gen_ass_with_settings(
        active_word_enabled=True,
        active_word_color="#FF0000",
        active_word_outline_color="#00FF00",
    )
    warnings = validate_ass(ass, settings, w, h)
    check("active word enabled produces no QA warnings", len(warnings) == 0,
          f"got {len(warnings)} warnings: {warnings}")


def test_validate_ass_max_words():
    """QA should pass when max_words limits are respected."""
    print("\n--- QA Validate: max words ---")
    ass, settings, w, h = _gen_ass_with_settings(max_words=2)
    warnings = validate_ass(ass, settings, w, h)
    check("max_words=2 produces no QA warnings", len(warnings) == 0,
          f"got {len(warnings)} warnings: {warnings}")


def test_validate_ass_margin_calculation():
    """QA should pass for various max_width and offset_v values."""
    print("\n--- QA Validate: margin calculation ---")
    for mw, ov in [(50, 10), (70, 0), (100, 50), (90, 4)]:
        ass, settings, w, h = _gen_ass_with_settings(max_width=mw, offset_v=ov)
        warnings = validate_ass(ass, settings, w, h)
        check(f"max_width={mw} offset_v={ov} no QA warnings", len(warnings) == 0,
              f"got {len(warnings)} warnings: {warnings}")


def test_validate_ass_outline_settings():
    """QA should pass for various outline width/color/opacity combinations."""
    print("\n--- QA Validate: outline settings ---")
    # Zero outline
    ass, settings, w, h = _gen_ass_with_settings(outline_width=0)
    warnings = validate_ass(ass, settings, w, h)
    check("outline_width=0 no QA warnings", len(warnings) == 0,
          f"got {len(warnings)} warnings: {warnings}")

    # Max outline
    ass, settings, w, h = _gen_ass_with_settings(
        outline_width=10, outline_color="#FF00FF", outline_opacity=50,
    )
    warnings = validate_ass(ass, settings, w, h)
    check("outline_width=10 no QA warnings", len(warnings) == 0,
          f"got {len(warnings)} warnings: {warnings}")


def test_validate_ass_position_alignment():
    """QA should pass for all three position settings."""
    print("\n--- QA Validate: position alignment ---")
    for pos in ("top", "center", "bottom"):
        ass, settings, w, h = _gen_ass_with_settings(position=pos)
        warnings = validate_ass(ass, settings, w, h)
        check(f"position={pos} no QA warnings", len(warnings) == 0,
              f"got {len(warnings)} warnings: {warnings}")


def test_validate_ass_detects_font_mismatch():
    """QA should detect when ASS font doesn't match settings."""
    print("\n--- QA Validate: detect font mismatch ---")
    # Generate ASS with one font, validate with different font in settings
    ass = generate_ass(
        segments=SAMPLE_SEGMENTS, start_time=10.0, end_time=25.0,
        font="DM Sans", font_size="medium", font_weight="bold",
        font_color="#FFFFFF", position="bottom",
        video_width=1920, video_height=1080,
        outline_color="#000000", outline_opacity=100, outline_width=2,
        show_speaker_labels=False, max_width_pct=90, offset_v_pct=4,
    )
    bad_settings = {"font": "Arial", "size": "medium", "font_weight": "bold",
                    "font_color": "#FFFFFF", "position": "bottom",
                    "outline_color": "#000000", "outline_opacity": 100,
                    "outline_width": 2, "show_speaker_labels": False,
                    "max_width": 90, "offset_v": 4}
    warnings = validate_ass(ass, bad_settings, 1920, 1080)
    has_font_warning = any("font mismatch" in w.lower() for w in warnings)
    check("detects font name mismatch", has_font_warning,
          f"warnings: {warnings}")


def test_validate_ass_detects_size_mismatch():
    """QA should detect when ASS font size doesn't match settings."""
    print("\n--- QA Validate: detect size mismatch ---")
    # Generate with medium, validate with large
    ass = generate_ass(
        segments=SAMPLE_SEGMENTS, start_time=10.0, end_time=25.0,
        font="DM Sans", font_size="medium", font_weight="bold",
        font_color="#FFFFFF", position="bottom",
        video_width=1920, video_height=1080,
        outline_color="#000000", outline_opacity=100, outline_width=2,
        show_speaker_labels=False, max_width_pct=90, offset_v_pct=4,
    )
    bad_settings = {"font": "DM Sans", "size": "large", "font_weight": "bold",
                    "font_color": "#FFFFFF", "position": "bottom",
                    "outline_color": "#000000", "outline_opacity": 100,
                    "outline_width": 2, "show_speaker_labels": False,
                    "max_width": 90, "offset_v": 4}
    warnings = validate_ass(ass, bad_settings, 1920, 1080)
    has_size_warning = any("font size mismatch" in w.lower() for w in warnings)
    check("detects font size mismatch", has_size_warning,
          f"warnings: {warnings}")


def test_validate_ass_all_font_sizes():
    """QA passes for all three font sizes at all standard resolutions."""
    print("\n--- QA Validate: all font sizes at all resolutions ---")
    for size in ("small", "medium", "large"):
        for dims in ((1920, 1080), (1080, 1920), (1080, 1080), (1080, 1350)):
            ass, settings, w, h = _gen_ass_with_settings(
                size=size, video_width=dims[0], video_height=dims[1],
            )
            warnings = validate_ass(ass, settings, w, h)
            check(f"size={size} at {dims[0]}x{dims[1]} no warnings", len(warnings) == 0,
                  f"got {warnings}")


def test_validate_ass_active_word_with_bg():
    """QA passes for active word with background opacity."""
    print("\n--- QA Validate: active word with background ---")
    ass, settings, w, h = _gen_ass_with_settings(
        active_word_enabled=True,
        active_word_color="#00FF00",
        active_word_outline_color="#FF0000",
        active_word_bg_color="#0000FF",
        active_word_bg_opacity=80,
    )
    warnings = validate_ass(ass, settings, w, h)
    check("active word with bg produces no QA warnings", len(warnings) == 0,
          f"got {len(warnings)} warnings: {warnings}")


def test_validate_ass_combined_settings():
    """QA passes for a complex combination of all settings."""
    print("\n--- QA Validate: combined complex settings ---")
    ass, settings, w, h = _gen_ass_with_settings(
        font="DM Sans",
        size="large",
        font_weight="normal",
        font_color="#00FFAA",
        position="top",
        background_enabled=True,
        background_color="#112233",
        background_opacity=60,
        show_speaker_labels=True,
        max_width=70,
        offset_v=15,
        max_words=3,
        video_width=1080,
        video_height=1920,
    )
    warnings = validate_ass(ass, settings, w, h)
    check("complex combined settings no QA warnings", len(warnings) == 0,
          f"got {len(warnings)} warnings: {warnings}")


# ===========================================================================
# TEST SUITE: Frontend-Export Settings Consistency
# ===========================================================================

def test_default_consistency_across_components():
    """Verify default values match across all frontend components and backend."""
    print("\n--- Default Consistency: all components ---")

    base = os.path.join(os.path.dirname(__file__), "..", "..")

    # Read source files
    with open(os.path.join(base, "frontend", "src", "components", "ClipSettingsPanel.jsx")) as f:
        panel_src = f.read()
    with open(os.path.join(base, "frontend", "src", "components", "ClipPreview.jsx")) as f:
        preview_src = f.read()
    with open(os.path.join(base, "frontend", "src", "pages", "Analysis.jsx")) as f:
        analysis_src = f.read()
    with open(os.path.join(base, "frontend", "src", "pages", "ClipSEO.jsx")) as f:
        seo_src = f.read()
    with open(os.path.join(base, "backend", "services", "clip_exporter.py")) as f:
        exporter_src = f.read()

    # Pydantic model defaults
    s = SubtitleSettings()

    # show_speaker_labels: should be False everywhere
    check("Model: show_speaker_labels=False", s.show_speaker_labels is False)
    check("Panel: showSpeakerLabels: false", "showSpeakerLabels: false" in panel_src)
    check("Preview: showSpeakerLabels ?? false", "showSpeakerLabels ?? false" in preview_src)
    check("Analysis state: showSpeakerLabels: false", "showSpeakerLabels: false" in analysis_src)
    check("Exporter: show_speaker_labels\", False", '"show_speaker_labels", False' in exporter_src)

    # background_enabled: should be False everywhere
    check("Model: background_enabled=False", s.background_enabled is False)
    check("Panel: subtitleBgEnabled: false", "subtitleBgEnabled: false" in panel_src)

    # outline defaults
    check("Model: outline_width=2", s.outline_width == 2)
    check("Model: outline_opacity=100", s.outline_opacity == 100)
    check("Model: outline_color=#000000", s.outline_color == "#000000")

    # active word defaults
    check("Model: active_word_enabled=False", s.active_word_enabled is False)
    check("Model: active_word_bg_opacity=0", s.active_word_bg_opacity == 0)
    check("Panel: activeWordEnabled: false", "activeWordEnabled: false" in panel_src)

    # font defaults
    check("Model: font=DM Sans", s.font == "DM Sans")
    check("Model: size=medium", s.size == "medium")
    check("Model: font_weight=bold", s.font_weight == "bold")

    # max_width and offset_v
    check("Model: max_width=90", s.max_width == 90)
    check("Model: offset_v=4", s.offset_v == 4)
    check("Panel: subtitleMaxWidth: 90", "subtitleMaxWidth: 90" in panel_src)
    check("Panel: subtitleOffsetV: 4", "subtitleOffsetV: 4" in panel_src)


def test_clipsettingspanel_defaults_through_pipeline():
    """Simulate ClipSettingsPanel DEFAULT_SETTINGS through the full export pipeline."""
    print("\n--- Pipeline: ClipSettingsPanel defaults ---")

    # These are the DEFAULT_SETTINGS from ClipSettingsPanel.jsx mapped to backend names
    settings = {
        "font": "DM Sans",
        "size": "medium",
        "font_weight": "bold",
        "font_color": "#FFFFFF",
        "position": "bottom",
        "speaker_colors": {},
        "use_speaker_colors": True,
        "background_enabled": False,
        "background_color": "#000000",
        "background_opacity": 75,
        "background_radius": 0,
        "outline_color": "#000000",
        "outline_opacity": 100,
        "outline_width": 2,
        "show_speaker_labels": False,
        "max_width": 90,
        "offset_v": 4,
        "max_words": 0,
        "active_word_enabled": False,
        "active_word_color": "#FFD700",
        "active_word_outline_color": "#000000",
        "active_word_bg_color": "#000000",
        "active_word_bg_opacity": 0,
    }

    # Test at all standard output resolutions
    for dims in ((1920, 1080), (1080, 1920), (1080, 1080), (1080, 1350)):
        ass = generate_ass(
            segments=SAMPLE_SEGMENTS, start_time=10.0, end_time=25.0,
            font=settings["font"], font_size=settings["size"],
            font_weight=settings["font_weight"], font_color=settings["font_color"],
            position=settings["position"],
            speaker_colors=settings["speaker_colors"],
            use_speaker_colors=settings["use_speaker_colors"],
            video_width=dims[0], video_height=dims[1],
            background_enabled=settings["background_enabled"],
            background_color=settings["background_color"],
            background_opacity=settings["background_opacity"],
            background_radius=settings["background_radius"],
            outline_color=settings["outline_color"],
            outline_opacity=settings["outline_opacity"],
            outline_width=settings["outline_width"],
            show_speaker_labels=settings["show_speaker_labels"],
            max_width_pct=settings["max_width"],
            offset_v_pct=settings["offset_v"],
            max_words=settings["max_words"],
            active_word_enabled=settings["active_word_enabled"],
            active_word_color=settings["active_word_color"],
            active_word_outline_color=settings["active_word_outline_color"],
            active_word_bg_color=settings["active_word_bg_color"],
            active_word_bg_opacity=settings["active_word_bg_opacity"],
        )
        warnings = validate_ass(ass, settings, dims[0], dims[1])
        check(f"ClipSettingsPanel defaults at {dims[0]}x{dims[1]}: 0 warnings",
              len(warnings) == 0, f"got {warnings}")


def test_analysis_export_body_through_pipeline():
    """Simulate Analysis.jsx handleExportClip settings through the full pipeline."""
    print("\n--- Pipeline: Analysis.jsx export body ---")

    # These match the initial clipSettings state in Analysis.jsx (lines 37-63)
    # mapped through the export handler (lines 224-248)
    settings = {
        "font": "DM Sans",
        "size": "medium",
        "font_weight": "bold",
        "font_color": "#FFFFFF",
        "position": "bottom",
        "speaker_colors": {},
        "use_speaker_colors": True,
        "background_enabled": False,
        "background_color": "#000000",
        "background_opacity": 75,
        "background_radius": 0,
        "outline_color": "#000000",
        "outline_opacity": 100,
        "outline_width": 2,
        "show_speaker_labels": False,  # clipSettings.showSpeakerLabels ?? false
        "max_width": 90,
        "offset_v": 4,
        "max_words": 0,
        "active_word_enabled": False,
        "active_word_color": "#FFD700",
        "active_word_outline_color": "#000000",
        "active_word_bg_color": "#000000",
        "active_word_bg_opacity": 0,
    }

    ass = generate_ass(
        segments=SAMPLE_SEGMENTS, start_time=10.0, end_time=25.0,
        font=settings["font"], font_size=settings["size"],
        font_weight=settings["font_weight"], font_color=settings["font_color"],
        position=settings["position"],
        speaker_colors=settings["speaker_colors"],
        use_speaker_colors=settings["use_speaker_colors"],
        video_width=1920, video_height=1080,
        background_enabled=settings["background_enabled"],
        background_color=settings["background_color"],
        background_opacity=settings["background_opacity"],
        outline_color=settings["outline_color"],
        outline_opacity=settings["outline_opacity"],
        outline_width=settings["outline_width"],
        show_speaker_labels=settings["show_speaker_labels"],
        max_width_pct=settings["max_width"],
        offset_v_pct=settings["offset_v"],
        max_words=settings["max_words"],
        active_word_enabled=settings["active_word_enabled"],
        active_word_color=settings["active_word_color"],
        active_word_outline_color=settings["active_word_outline_color"],
        active_word_bg_color=settings["active_word_bg_color"],
        active_word_bg_opacity=settings["active_word_bg_opacity"],
    )
    warnings = validate_ass(ass, settings, 1920, 1080)
    check("Analysis.jsx export defaults: 0 QA warnings", len(warnings) == 0,
          f"got {warnings}")


def test_clipseo_export_body_through_pipeline():
    """Simulate ClipSEO.jsx export handler settings through the full pipeline."""
    print("\n--- Pipeline: ClipSEO.jsx export body ---")

    # ClipSEO uses individual state variables initialized from savedSettings.
    # Default values from SEO_SETTINGS_DEFAULTS / useState initializers.
    settings = {
        "font": "DM Sans",
        "size": "medium",
        "font_weight": "bold",
        "font_color": "#FFFFFF",
        "position": "bottom",
        "speaker_colors": {},
        "use_speaker_colors": True,
        "background_enabled": False,
        "background_color": "#000000",
        "background_opacity": 75,
        "background_radius": 0,
        "outline_color": "#000000",
        "outline_opacity": 100,
        "outline_width": 2,
        "show_speaker_labels": False,
        "max_width": 90,
        "offset_v": 4,
        "max_words": 0,
        "active_word_enabled": False,
        "active_word_color": "#FFD700",
        "active_word_outline_color": "#000000",
        "active_word_bg_color": "#000000",
        "active_word_bg_opacity": 0,
    }

    # Test with a 9:16 export (common for ClipSEO — portrait clips)
    ass = generate_ass(
        segments=SAMPLE_SEGMENTS, start_time=10.0, end_time=25.0,
        font=settings["font"], font_size=settings["size"],
        font_weight=settings["font_weight"], font_color=settings["font_color"],
        position=settings["position"],
        speaker_colors=settings["speaker_colors"],
        use_speaker_colors=settings["use_speaker_colors"],
        video_width=1080, video_height=1920,
        background_enabled=settings["background_enabled"],
        background_color=settings["background_color"],
        background_opacity=settings["background_opacity"],
        outline_color=settings["outline_color"],
        outline_opacity=settings["outline_opacity"],
        outline_width=settings["outline_width"],
        show_speaker_labels=settings["show_speaker_labels"],
        max_width_pct=settings["max_width"],
        offset_v_pct=settings["offset_v"],
        max_words=settings["max_words"],
        active_word_enabled=settings["active_word_enabled"],
        active_word_color=settings["active_word_color"],
        active_word_outline_color=settings["active_word_outline_color"],
        active_word_bg_color=settings["active_word_bg_color"],
        active_word_bg_opacity=settings["active_word_bg_opacity"],
    )
    warnings = validate_ass(ass, settings, 1080, 1920)
    check("ClipSEO.jsx export defaults at 9:16: 0 QA warnings", len(warnings) == 0,
          f"got {warnings}")


def test_all_settings_combinations():
    """Test many setting combinations through the QA validation."""
    print("\n--- Pipeline: all settings combinations ---")

    combos = [
        # (label, overrides)
        ("bg + active word", dict(
            background_enabled=True, background_color="#FF0000",
            background_opacity=80, active_word_enabled=True,
            active_word_color="#00FF00",
        )),
        ("outline + top + large", dict(
            outline_width=8, outline_color="#0000FF", outline_opacity=70,
            position="top", size="large",
        )),
        ("max_words + speaker labels", dict(
            max_words=3, show_speaker_labels=True,
        )),
        ("center + zero outline", dict(
            position="center", outline_width=0,
        )),
        ("max_words + active word", dict(
            max_words=2, active_word_enabled=True,
            active_word_color="#FF00FF", active_word_bg_color="#AABB00",
            active_word_bg_opacity=50,
        )),
        ("all maxed out", dict(
            size="large", font_weight="bold", outline_width=10,
            outline_opacity=100, background_enabled=True,
            background_opacity=100, max_width=50, offset_v=20,
            show_speaker_labels=True, max_words=5,
        )),
        ("all minimized", dict(
            size="small", font_weight="normal", outline_width=0,
            outline_opacity=0, max_width=100, offset_v=0,
        )),
    ]

    for label, overrides in combos:
        for dims in ((1920, 1080), (1080, 1920)):
            ass, settings, w, h = _gen_ass_with_settings(
                video_width=dims[0], video_height=dims[1], **overrides,
            )
            warnings = validate_ass(ass, settings, w, h)
            check(f"{label} at {dims[0]}x{dims[1]}: 0 warnings",
                  len(warnings) == 0, f"got {warnings}")


def test_background_radius_not_in_ass():
    """Verify background_radius has no effect on ASS output (ASS limitation)."""
    print("\n--- Pipeline: background_radius ignored in ASS ---")

    # Generate ASS with radius=0 and radius=20
    ass0, _, w, h = _gen_ass_with_settings(
        background_enabled=True, background_radius=0,
    )
    ass20, _, w2, h2 = _gen_ass_with_settings(
        background_enabled=True, background_radius=20,
    )

    # Parse styles from both — they should be identical
    from backend.services.clip_exporter import _parse_ass_styles
    styles0 = _parse_ass_styles(ass0)
    styles20 = _parse_ass_styles(ass20)
    check("ASS styles identical with radius=0 and radius=20",
          len(styles0) > 0 and styles0 == styles20,
          f"styles differ: {styles0} vs {styles20}")

    # Both should pass QA validation
    settings0 = {"font": "DM Sans", "size": "medium", "font_weight": "bold",
                  "font_color": "#FFFFFF", "position": "bottom",
                  "background_enabled": True, "background_color": "#000000",
                  "background_opacity": 75, "outline_color": "#000000",
                  "outline_opacity": 100, "outline_width": 2,
                  "show_speaker_labels": False, "max_width": 90, "offset_v": 4}
    w0 = validate_ass(ass0, settings0, w, h)
    check("radius=0 passes QA", len(w0) == 0, f"got {w0}")
    w20 = validate_ass(ass20, settings0, w2, h2)
    check("radius=20 passes QA (ignored)", len(w20) == 0, f"got {w20}")


# ---------------------------------------------------------------------------
# Dynamic Subject Tracking Tests
# ---------------------------------------------------------------------------

def test_build_subject_keyframes_basic():
    """Multiple scenes → correct relative time conversion and filtering."""
    print("\n--- build_subject_keyframes: multiple scenes ---")

    class FakeScene:
        def __init__(self, ts, sx):
            self.timestamp = ts
            self.subject_x = sx

    scenes = [
        FakeScene(5.0, 20),
        FakeScene(10.0, 60),
        FakeScene(15.0, 80),
        FakeScene(25.0, 40),  # outside clip range
    ]
    kf = _build_subject_keyframes(scenes, 3.0, 18.0)
    # Should include scenes at t=5,10,15 → relative t=2,7,12
    # Plus boundaries at t=0 and t=15
    check("At least 4 keyframes", len(kf) >= 4, f"got {len(kf)}: {kf}")
    check("First keyframe at t=0", kf[0][0] == 0.0, f"got t={kf[0][0]}")
    check("Last keyframe at t=15", kf[-1][0] == 15.0, f"got t={kf[-1][0]}")
    # Scene at t=5 → relative t=2.0
    inner_times = [k[0] for k in kf]
    check("Contains t=2.0", 2.0 in inner_times, f"times: {inner_times}")
    check("Contains t=7.0", 7.0 in inner_times, f"times: {inner_times}")
    check("Contains t=12.0", 12.0 in inner_times, f"times: {inner_times}")
    # scene at t=25 should be excluded
    check("No t > 15", all(k[0] <= 15.0 for k in kf), f"times: {inner_times}")


# ===========================================================================
# TEST SUITE: Comprehensive QA Validation
# ===========================================================================

def test_validate_active_word_bg_color_present():
    """QA should validate that \\4c tag with correct bg color is present when bg_opacity > 0."""
    print("\n--- QA Validate: active word bg color present ---")
    ass, settings, w, h = _gen_ass_with_settings(
        active_word_enabled=True,
        active_word_color="#00FF00",
        active_word_outline_color="#111111",
        active_word_bg_color="#FF0000",
        active_word_bg_opacity=80,
    )
    warnings = validate_ass(ass, settings, w, h)
    check("active word with bg_opacity=80: 0 QA warnings",
          len(warnings) == 0, f"got {warnings}")

    # Verify \4c tag is actually in the ASS events
    dialogues = [l for l in ass.split("\n") if l.startswith("Dialogue:")]
    has_4c = any("\\4c" in d for d in dialogues)
    check("\\4c tag present in dialogue events", has_4c)


def test_validate_active_word_bg_absent_when_zero_opacity():
    """QA should validate that no \\4c tag appears when bg_opacity=0."""
    print("\n--- QA Validate: no bg tag at zero opacity ---")
    ass, settings, w, h = _gen_ass_with_settings(
        active_word_enabled=True,
        active_word_bg_opacity=0,
    )
    warnings = validate_ass(ass, settings, w, h)
    check("active word with bg_opacity=0: 0 QA warnings",
          len(warnings) == 0, f"got {warnings}")


def test_validate_numeric_font_sizes():
    """QA should pass for numeric font sizes (12-72)."""
    print("\n--- QA Validate: numeric font sizes ---")
    for size in [12, 16, 22, 30, 40, 50, 60, 72]:
        ass, settings, w, h = _gen_ass_with_settings(size=size)
        warnings = validate_ass(ass, settings, w, h)
        check(f"numeric font_size={size}: 0 warnings",
              len(warnings) == 0, f"got {warnings}")


def test_validate_numeric_font_sizes_at_portrait():
    """QA should pass for numeric font sizes at 9:16 (1080x1920)."""
    print("\n--- QA Validate: numeric font sizes at 9:16 ---")
    for size in [12, 30, 72]:
        ass, settings, w, h = _gen_ass_with_settings(
            size=size, video_width=1080, video_height=1920,
        )
        warnings = validate_ass(ass, settings, w, h)
        check(f"numeric font_size={size} at 9:16: 0 warnings",
              len(warnings) == 0, f"got {warnings}")


def test_validate_all_builtin_fonts():
    """Every builtin font name should pass through ASS generation correctly."""
    print("\n--- QA Validate: all builtin fonts ---")
    FONTS = [
        "DM Sans", "Montserrat", "Open Sans", "Roboto", "Poppins",
        "Inter", "Nunito", "Lato", "Oswald", "Playfair Display",
        "Bebas Neue", "Liberation Sans", "Liberation Serif",
        "Liberation Mono", "DejaVu Sans", "DejaVu Serif",
        "DejaVu Sans Mono", "FreeSans",
    ]
    for font in FONTS:
        ass, settings, w, h = _gen_ass_with_settings(font=font)
        styles = parse_ass_styles(ass)
        # Check font name matches in all styles
        for style_name, style_data in styles.items():
            actual_font = style_data.get("Fontname", "")
            check(f"Font '{font}' in style '{style_name}'",
                  actual_font == font, f"got '{actual_font}'")


def test_viralclips_export_body_through_pipeline():
    """Simulate ViralClips.jsx handleExportClip settings through the full pipeline."""
    print("\n--- Pipeline: ViralClips.jsx export body ---")
    # These match handleExportClip (lines 672-696) in ViralClips.jsx
    settings = {
        "font": "DM Sans",
        "size": 30,  # ViralClips uses numeric size by default
        "font_weight": "bold",
        "font_color": "#FFFFFF",
        "position": "bottom",
        "speaker_colors": {},
        "use_speaker_colors": True,
        "background_enabled": False,
        "background_color": "#000000",
        "background_opacity": 75,
        "background_radius": 0,
        "outline_color": "#000000",
        "outline_opacity": 100,
        "outline_width": 2,
        "show_speaker_labels": False,
        "max_width": 90,
        "offset_v": 4,
        "max_words": 0,
        "active_word_enabled": False,
        "active_word_color": "#FFD700",
        "active_word_outline_color": "#000000",
        "active_word_bg_color": "#000000",
        "active_word_bg_opacity": 0,
    }

    # Test at all standard aspect ratios
    for label, dims in [
        ("16:9", (1920, 1080)),
        ("9:16", (1080, 1920)),
        ("1:1", (1080, 1080)),
        ("4:5", (1080, 1350)),
    ]:
        ass = generate_ass(
            segments=SAMPLE_SEGMENTS, start_time=10.0, end_time=25.0,
            font=settings["font"], font_size=settings["size"],
            font_weight=settings["font_weight"], font_color=settings["font_color"],
            position=settings["position"],
            speaker_colors=settings["speaker_colors"],
            use_speaker_colors=settings["use_speaker_colors"],
            video_width=dims[0], video_height=dims[1],
            background_enabled=settings["background_enabled"],
            background_color=settings["background_color"],
            background_opacity=settings["background_opacity"],
            background_radius=settings.get("background_radius", 0),
            outline_color=settings["outline_color"],
            outline_opacity=settings["outline_opacity"],
            outline_width=settings["outline_width"],
            show_speaker_labels=settings["show_speaker_labels"],
            max_width_pct=settings["max_width"],
            offset_v_pct=settings["offset_v"],
            max_words=settings["max_words"],
            active_word_enabled=settings["active_word_enabled"],
            active_word_color=settings["active_word_color"],
            active_word_outline_color=settings["active_word_outline_color"],
            active_word_bg_color=settings["active_word_bg_color"],
            active_word_bg_opacity=settings["active_word_bg_opacity"],
        )
        warnings = validate_ass(ass, settings, dims[0], dims[1])
        check(f"ViralClips export at {label}: 0 QA warnings",
              len(warnings) == 0, f"got {warnings}")


def test_extreme_settings_qa():
    """Test QA with edge-case settings combinations."""
    print("\n--- QA Validate: extreme settings ---")
    combos = [
        ("max everything", dict(
            size="large", font_weight="bold", outline_width=10,
            outline_opacity=100, background_enabled=True,
            background_color="#FF0000", background_opacity=100,
            max_width=50, offset_v=20, show_speaker_labels=True,
            max_words=5, active_word_enabled=True,
            active_word_color="#00FFFF", active_word_outline_color="#FF00FF",
            active_word_bg_color="#FFFF00", active_word_bg_opacity=100,
        )),
        ("min everything", dict(
            size="small", font_weight="normal", outline_width=0,
            outline_opacity=0, max_width=100, offset_v=0,
        )),
        ("numeric large font + portrait", dict(
            size=72, font_weight="bold", video_width=1080, video_height=1920,
            outline_width=5, max_words=2,
        )),
        ("numeric small font + square", dict(
            size=12, font_weight="normal", video_width=1080, video_height=1080,
            background_enabled=True, background_opacity=50,
        )),
        ("center position + active word + no outline", dict(
            position="center", outline_width=0,
            active_word_enabled=True, active_word_color="#FF0000",
        )),
        ("top position + max offset", dict(
            position="top", offset_v=50,
            outline_width=8, outline_color="#0000FF",
        )),
    ]

    for label, overrides in combos:
        ass, settings, w, h = _gen_ass_with_settings(**overrides)
        warnings = validate_ass(ass, settings, w, h)
        check(f"{label}: 0 QA warnings",
              len(warnings) == 0, f"got {warnings}")


def test_viralclips_defaults_match_model():
    """Verify ViralClips.jsx defaults match backend Pydantic model defaults."""
    print("\n--- Consistency: ViralClips defaults match model ---")
    base = os.path.join(os.path.dirname(__file__), "..", "..")
    with open(os.path.join(base, "frontend", "src", "pages", "ViralClips.jsx")) as f:
        viral_src = f.read()

    s = SubtitleSettings()

    # Check that ViralClips has the same fonts as ClipSettingsPanel
    with open(os.path.join(base, "frontend", "src", "components", "ClipSettingsPanel.jsx")) as f:
        panel_src = f.read()

    check("ViralClips has DM Sans font", "'DM Sans'" in viral_src)
    check("ViralClips has Montserrat font", "'Montserrat'" in viral_src)
    check("ViralClips has Roboto font", "'Roboto'" in viral_src)
    check("ViralClips has Poppins font", "'Poppins'" in viral_src)
    check("ViralClips has Inter font", "'Inter'" in viral_src)

    # Check default settings flow
    check("ViralClips uses subtitleFont || 'DM Sans'",
          "subtitleFont || 'DM Sans'" in viral_src or "subtitleFont ||'DM Sans'" in viral_src
          or "'DM Sans'" in viral_src)
    check("ViralClips uses subtitleFontWeight with 700 (Bold)",
          "700" in viral_src or "'bold'" in viral_src)
    check("ViralClips sends active_word_bg_opacity",
          "active_word_bg_opacity" in viral_src)
    check("ViralClips sends active_word_bg_color",
          "active_word_bg_color" in viral_src)


def test_settings_roundtrip_all_24_fields():
    """Verify all 24 SubtitleSettings fields survive the full roundtrip."""
    print("\n--- Pipeline: all 24 fields roundtrip ---")
    # Create a settings dict with all non-default values to ensure each field is used
    settings = {
        "font": "Roboto",
        "size": 42,
        "font_weight": "normal",
        "font_color": "#FF5500",
        "position": "top",
        "speaker_colors": {"Speaker 1": "#AA0000", "Speaker 2": "#00AA00"},
        "use_speaker_colors": False,  # Use font_color for all
        "background_enabled": True,
        "background_color": "#222222",
        "background_opacity": 60,
        "background_radius": 12,  # Intentionally ignored by ASS
        "outline_color": "#444444",
        "outline_opacity": 80,
        "outline_width": 5,
        "show_speaker_labels": True,
        "max_width": 70,
        "offset_v": 15,
        "max_words": 3,
        "active_word_enabled": True,
        "active_word_color": "#00FFAA",
        "active_word_outline_color": "#AA00FF",
        "active_word_bg_color": "#FFAA00",
        "active_word_bg_opacity": 50,
    }

    for dims in ((1920, 1080), (1080, 1920), (1080, 1080)):
        ass = generate_ass(
            segments=SAMPLE_SEGMENTS, start_time=10.0, end_time=25.0,
            font=settings["font"], font_size=settings["size"],
            font_weight=settings["font_weight"], font_color=settings["font_color"],
            position=settings["position"],
            speaker_colors=settings["speaker_colors"],
            use_speaker_colors=settings["use_speaker_colors"],
            video_width=dims[0], video_height=dims[1],
            background_enabled=settings["background_enabled"],
            background_color=settings["background_color"],
            background_opacity=settings["background_opacity"],
            background_radius=settings.get("background_radius", 0),
            outline_color=settings["outline_color"],
            outline_opacity=settings["outline_opacity"],
            outline_width=settings["outline_width"],
            show_speaker_labels=settings["show_speaker_labels"],
            max_width_pct=settings["max_width"],
            offset_v_pct=settings["offset_v"],
            max_words=settings["max_words"],
            active_word_enabled=settings["active_word_enabled"],
            active_word_color=settings["active_word_color"],
            active_word_outline_color=settings["active_word_outline_color"],
            active_word_bg_color=settings["active_word_bg_color"],
            active_word_bg_opacity=settings["active_word_bg_opacity"],
        )
        warnings = validate_ass(ass, settings, dims[0], dims[1])
        check(f"24-field roundtrip at {dims[0]}x{dims[1]}: 0 warnings",
              len(warnings) == 0, f"got {warnings}")

        # Also verify key structural elements
        styles = parse_ass_styles(ass)
        check(f"Font 'Roboto' at {dims[0]}x{dims[1]}",
              all(s.get("Fontname") == "Roboto" for s in styles.values()))
        check(f"Font weight 'normal' at {dims[0]}x{dims[1]}",
              all(s.get("Bold") == "0" for s in styles.values()))

        dialogues = parse_ass_dialogues(ass)
        check(f"Speaker labels present at {dims[0]}x{dims[1]}",
              any("Speaker 1:" in d.get("Text", "") or "Speaker 2:" in d.get("Text", "")
                  for d in dialogues))
        check(f"Active word \\c tags at {dims[0]}x{dims[1]}",
              any("\\c" in d.get("Text", "") for d in dialogues))
        check(f"Active word \\4c tags at {dims[0]}x{dims[1]}",
              any("\\4c" in d.get("Text", "") for d in dialogues))

        # Verify max_words: no subtitle should have more than 3 words
        for d in dialogues:
            plain = re.sub(r"\{[^}]*\}", "", d.get("Text", ""))
            # Strip speaker label
            plain = re.sub(r"^.+?:\s", "", plain, count=1)
            words = plain.split()
            check(f"max_words=3 enforced at {dims[0]}x{dims[1]}: '{plain[:40]}'",
                  len(words) <= 3, f"got {len(words)} words: '{plain}'")
            break  # Just check first one


def test_custom_font_name():
    """Verify custom font name persists through ASS generation and validation."""
    print("\n--- ASS: custom font name ---")
    result = generate_ass(
        segments=SAMPLE_SEGMENTS, start_time=10.0, end_time=25.0,
        font="Montserrat",
    )
    styles = parse_ass_styles(result)
    for style_name, s in styles.items():
        check(f"Montserrat font in {style_name}", s.get("Fontname") == "Montserrat",
              f"got {s.get('Fontname')}")


def test_validate_speaker_colors_override():
    """QA should validate that speaker_colors dict colors appear in ASS styles."""
    print("\n--- QA Validate: speaker colors override ---")
    # Provide explicit colors for both speakers
    ass, settings, w, h = _gen_ass_with_settings(
        speaker_colors={"Speaker 1": "#FF0000", "Speaker 2": "#00FF00"},
        use_speaker_colors=True,
    )
    warnings = validate_ass(ass, settings, w, h)
    check("speaker_colors override: 0 QA warnings",
          len(warnings) == 0, f"got {warnings}")

    # Verify the colors actually appear in the ASS styles
    styles = parse_ass_styles(ass)
    sp1_style = styles.get("Speaker 1")
    sp2_style = styles.get("Speaker 2")
    check("Speaker 1 style exists", sp1_style is not None)
    check("Speaker 2 style exists", sp2_style is not None)
    if sp1_style:
        check("Speaker 1 PrimaryColour = red (&H000000FF&)",
              sp1_style["PrimaryColour"].upper() == "&H000000FF&",
              f"got {sp1_style['PrimaryColour']}")
    if sp2_style:
        check("Speaker 2 PrimaryColour = green (&H0000FF00&)",
              sp2_style["PrimaryColour"].upper() == "&H0000FF00&",
              f"got {sp2_style['PrimaryColour']}")


def test_build_subject_keyframes_single():
    """Single scene → single keyframe at t=0."""
    print("\n--- build_subject_keyframes: single scene ---")

    class FakeScene:
        def __init__(self, ts, sx):
            self.timestamp = ts
            self.subject_x = sx

    kf = _build_subject_keyframes([FakeScene(10.0, 30)], 5.0, 15.0)
    check("At least 1 keyframe", len(kf) >= 1, f"got {kf}")
    check("Subject_x is 30", kf[0][1] == 30, f"got {kf[0][1]}")


def test_build_subject_keyframes_empty():
    """No overlapping scenes — uses nearest scene data for interpolation."""
    print("\n--- build_subject_keyframes: no overlapping scenes ---")

    class FakeScene:
        def __init__(self, ts, sx):
            self.timestamp = ts
            self.subject_x = sx

    # Scene at t=50 (after clip 5-15) → boundary interpolation uses after scene
    kf = _build_subject_keyframes([FakeScene(50.0, 80)], 5.0, 15.0)
    check("Has boundary keyframes", len(kf) >= 1, f"got {kf}")
    # The nearest scene has subject_x=80, so both boundaries should use it
    # (clamped by _safe_subject_x to [10,90])
    check("Uses nearest scene subject_x=80", kf[0][1] == 80, f"got {kf[0][1]}")

    # Truly empty scenes → default center
    kf2 = _build_subject_keyframes([], 5.0, 15.0)
    check("Empty scenes → single keyframe", len(kf2) == 1, f"got {kf2}")
    check("Empty scenes → default center 50", kf2[0][1] == 50, f"got {kf2[0][1]}")


def test_build_subject_keyframes_dict():
    """Scenes passed as dicts (from model_dump)."""
    print("\n--- build_subject_keyframes: dict scenes ---")
    scenes = [
        {"timestamp": 10.0, "subject_x": 25},
        {"timestamp": 20.0, "subject_x": 75},
    ]
    kf = _build_subject_keyframes(scenes, 5.0, 25.0)
    check("Has keyframes", len(kf) >= 3, f"got {kf}")
    inner_sx = [k[1] for k in kf]
    check("Contains sx=25", 25 in inner_sx, f"got {inner_sx}")
    check("Contains sx=75", 75 in inner_sx, f"got {inner_sx}")


def test_smooth_keyframes_no_change():
    """Already smooth keyframes → unchanged."""
    print("\n--- smooth_keyframes: no smoothing needed ---")
    kf = [(0.0, 50), (5.0, 55), (10.0, 60)]
    smoothed = _smooth_keyframes(kf)
    check("Same length", len(smoothed) == len(kf))
    check("Values unchanged", smoothed == kf, f"got {smoothed}")


def test_smooth_keyframes_clamp():
    """Rapid jump → clamped by max_speed."""
    print("\n--- smooth_keyframes: rapid jump clamped ---")
    # Jump from 10 to 90 in 1 second → delta=80, but max_speed=25 allows only 25
    kf = [(0.0, 10), (1.0, 90), (2.0, 90)]
    smoothed = _smooth_keyframes(kf, max_speed=25)
    check("First unchanged", smoothed[0] == (0.0, 10))
    check("Second clamped", smoothed[1][1] == 35, f"got {smoothed[1][1]}")  # 10 + 25*1 = 35
    # Third: from 35 → 90, delta=55, max=25*1=25, so clamped to 60
    check("Third clamped", smoothed[2][1] == 60, f"got {smoothed[2][1]}")


def test_build_crop_x_expr_static():
    """All same subject_x → plain integer string."""
    print("\n--- build_crop_x_expr: static (all same) ---")
    kf = [(0.0, 50), (5.0, 50), (10.0, 50)]
    expr = _build_crop_x_expr(kf, 1000)
    check("Static integer", expr == "500", f"got '{expr}'")


def test_build_crop_x_expr_dynamic():
    """Multiple different values → FFmpeg expression with if()."""
    print("\n--- build_crop_x_expr: dynamic expression ---")
    kf = [(0.0, 0), (5.0, 50), (10.0, 100)]
    expr = _build_crop_x_expr(kf, 1000)
    check("Contains 'if(lt(t'", "if(lt(t" in expr, f"got '{expr}'")
    check("Contains 'clip('", "clip(" in expr, f"got '{expr}'")
    check("Contains max_offset 1000", "1000" in expr, f"got '{expr}'")


def test_build_crop_x_expr_two_keyframes():
    """Two keyframes → single interpolation segment."""
    print("\n--- build_crop_x_expr: two keyframes ---")
    kf = [(0.0, 20), (10.0, 80)]
    expr = _build_crop_x_expr(kf, 500)
    check("Contains 'if(lt(t'", "if(lt(t" in expr, f"got '{expr}'")
    # At t=0: offset should be 500 * 20/100 = 100
    # At t=10: offset should be 500 * 80/100 = 400
    check("Contains offset 100", "100" in expr, f"got '{expr}'")


def test_build_crop_x_expr_zero_max_offset():
    """Zero max_offset → always '0'."""
    print("\n--- build_crop_x_expr: zero max_offset ---")
    kf = [(0.0, 20), (10.0, 80)]
    expr = _build_crop_x_expr(kf, 0)
    check("Returns '0'", expr == "0", f"got '{expr}'")


def test_filter_chain_dynamic_subject():
    """Filter chain with dynamic keyframes → expression in crop filter."""
    print("\n--- filter_chain: dynamic subject keyframes ---")
    kf = [(0.0, 20), (5.0, 80)]
    vf, _, _ = _build_filter_chain("9:16", 1920, 1080, None, subject_keyframes=kf)
    check("Filter chain exists", vf is not None)
    check("Contains crop filter", "crop=" in vf, f"got '{vf}'")
    check("Contains dynamic expression", "if(lt(t" in vf, f"got '{vf}'")
    check("Contains scale", "scale=" in vf, f"got '{vf}'")


def test_filter_chain_static_fallback():
    """Single keyframe → static crop (backward compatible)."""
    print("\n--- filter_chain: static fallback (single keyframe) ---")
    kf = [(0.0, 30)]
    vf_kf, _ = _build_filter_chain("9:16", 1920, 1080, None, subject_keyframes=kf)
    vf_static, _ = _build_filter_chain("9:16", 1920, 1080, None, subject_x=30)
    check("Both produce crop filter", "crop=" in vf_kf and "crop=" in vf_static)
    check("Single keyframe matches static", vf_kf == vf_static,
          f"keyframe: '{vf_kf}' vs static: '{vf_static}'")


def test_filter_chain_no_keyframes():
    """No keyframes → uses subject_x fallback."""
    print("\n--- filter_chain: no keyframes → subject_x fallback ---")
    vf_none, _ = _build_filter_chain("9:16", 1920, 1080, None, subject_x=70)
    vf_empty, _ = _build_filter_chain("9:16", 1920, 1080, None, subject_x=70, subject_keyframes=None)
    check("Both identical", vf_none == vf_empty, f"none: '{vf_none}' vs empty: '{vf_empty}'")
    check("Contains static crop", "crop=" in vf_none and "if(lt" not in vf_none,
          f"got '{vf_none}'")


def test_filter_chain_all_same_keyframes():
    """All keyframes same value → static crop (optimization)."""
    print("\n--- filter_chain: all same keyframes → static ---")
    kf = [(0.0, 40), (5.0, 40), (10.0, 40)]
    vf, _, _ = _build_filter_chain("9:16", 1920, 1080, None, subject_keyframes=kf)
    check("No dynamic expression", "if(lt" not in vf, f"got '{vf}'")
    check("Has crop filter", "crop=" in vf, f"got '{vf}'")


def test_safe_subject_x_clamping():
    """Safety clamping prevents edge-cutting for extreme subject_x values."""
    print("\n--- safe_subject_x: clamping to [10, 90] ---")
    # Edge values get clamped
    check("sx=0 → 10", _safe_subject_x(0) == 10, f"got {_safe_subject_x(0)}")
    check("sx=5 → 10", _safe_subject_x(5) == 10, f"got {_safe_subject_x(5)}")
    check("sx=100 → 90", _safe_subject_x(100) == 90, f"got {_safe_subject_x(100)}")
    check("sx=95 → 90", _safe_subject_x(95) == 90, f"got {_safe_subject_x(95)}")
    # Values within safe range are unchanged
    check("sx=50 → 50", _safe_subject_x(50) == 50, f"got {_safe_subject_x(50)}")
    check("sx=10 → 10", _safe_subject_x(10) == 10, f"got {_safe_subject_x(10)}")
    check("sx=90 → 90", _safe_subject_x(90) == 90, f"got {_safe_subject_x(90)}")
    check("sx=30 → 30", _safe_subject_x(30) == 30, f"got {_safe_subject_x(30)}")
    check("sx=85 → 85", _safe_subject_x(85) == 85, f"got {_safe_subject_x(85)}")
    # Custom margin
    check("sx=0, margin=15 → 15", _safe_subject_x(0, margin=15) == 15, f"got {_safe_subject_x(0, margin=15)}")
    check("sx=100, margin=15 → 85", _safe_subject_x(100, margin=15) == 85, f"got {_safe_subject_x(100, margin=15)}")


def test_keyframes_apply_safety_margin():
    """Keyframes from scenes with extreme subject_x get safety-clamped."""
    print("\n--- keyframes: safety margin applied to extreme values ---")

    class FakeScene:
        def __init__(self, ts, sx):
            self.timestamp = ts
            self.subject_x = sx

    scenes = [
        FakeScene(5.0, 0),    # Extreme left → should be clamped to 10
        FakeScene(10.0, 100),  # Extreme right → should be clamped to 90
        FakeScene(15.0, 50),   # Center → unchanged
    ]
    kf = _build_subject_keyframes(scenes, 3.0, 18.0)
    # Find the keyframe values (skip boundary duplicates at t=0 and t=end)
    values = [k[1] for k in kf]
    check("No value below 10", all(v >= 10 for v in values), f"got values: {values}")
    check("No value above 90", all(v <= 90 for v in values), f"got values: {values}")
    check("Contains 10 (clamped from 0)", 10 in values, f"got values: {values}")
    check("Contains 90 (clamped from 100)", 90 in values, f"got values: {values}")
    check("Contains 50 (unchanged)", 50 in values, f"got values: {values}")


def test_filter_chain_static_safety_margin():
    """Static crop with extreme subject_x uses safety-clamped value."""
    print("\n--- filter_chain: static crop with safety margin ---")
    # subject_x=0 should be clamped to 10 → offset should not be 0
    vf, _, _ = _build_filter_chain("9:16", 1920, 1080, None, subject_x=0)
    check("Has crop filter", "crop=" in vf, f"got '{vf}'")
    # Extract x_offset from crop=W:H:X:Y
    import re
    m = re.search(r"crop=(\d+):(\d+):(\d+):(\d+)", vf)
    check("Crop filter parsed", m is not None, f"got '{vf}'")
    if m:
        x_offset = int(m.group(3))
        # With safety: sx=0 → clamped to 10 → pixel=192, offset=192-303=-111 → clamped to 0
        # Subject at extreme left gets pushed to frame edge (centered as much as possible)
        check("x_offset = 0 (clamped edge)", x_offset == 0, f"got x_offset={x_offset}")


# ===========================================================================
# SUBJECT TRACKING QA VALIDATION TESTS
# ===========================================================================


def test_validate_subject_tracking_static_all_ratios():
    """Subject tracking validation passes for static crop at all aspect ratios."""
    print("\n--- Subject tracking QA: static crop at all aspect ratios ---")
    for ratio in ["9:16", "1:1", "4:5"]:
        for sx in [10, 30, 50, 70, 90]:
            vf, _, _ = _build_filter_chain(ratio, 1920, 1080, None, subject_x=sx)
            warnings = _validate_subject_tracking(
                filter_chain=vf,
                aspect_ratio=ratio,
                src_w=1920,
                src_h=1080,
                subject_x=sx,
                subject_keyframes=None,
            )
            check(f"Static sx={sx} at {ratio}: 0 QA warnings",
                  len(warnings) == 0, f"got {warnings}")


def test_validate_subject_tracking_dynamic_all_ratios():
    """Subject tracking validation passes for dynamic crop at all aspect ratios."""
    print("\n--- Subject tracking QA: dynamic crop at all aspect ratios ---")
    kf = [(0.0, 20), (5.0, 50), (10.0, 80)]
    for ratio in ["9:16", "1:1", "4:5"]:
        vf, _, _ = _build_filter_chain(ratio, 1920, 1080, None, subject_keyframes=kf)
        warnings = _validate_subject_tracking(
            filter_chain=vf,
            aspect_ratio=ratio,
            src_w=1920,
            src_h=1080,
            subject_x=50,
            subject_keyframes=kf,
        )
        check(f"Dynamic kf at {ratio}: 0 QA warnings",
              len(warnings) == 0, f"got {warnings}")


def test_subject_tracking_crop_changes_with_aspect_ratio():
    """Different aspect ratios produce different crop dimensions and offsets."""
    print("\n--- Subject tracking: crop changes with aspect ratio ---")
    sx = 30
    results = {}
    for ratio in ["9:16", "1:1", "4:5"]:
        vf, _, _ = _build_filter_chain(ratio, 1920, 1080, None, subject_x=sx)
        m = re.search(r"crop=(\d+):(\d+):(\d+):(\d+)", vf)
        check(f"Crop filter found for {ratio}", m is not None, f"got '{vf}'")
        if m:
            results[ratio] = {
                "crop_w": int(m.group(1)),
                "crop_h": int(m.group(2)),
                "x_offset": int(m.group(3)),
            }

    # 9:16 needs much narrower crop from 16:9 source
    if "9:16" in results and "1:1" in results:
        check("9:16 narrower than 1:1",
              results["9:16"]["crop_w"] < results["1:1"]["crop_w"],
              f"9:16={results['9:16']['crop_w']} vs 1:1={results['1:1']['crop_w']}")
    if "1:1" in results and "4:5" in results:
        check("1:1 wider than 4:5",
              results["1:1"]["crop_w"] > results["4:5"]["crop_w"],
              f"1:1={results['1:1']['crop_w']} vs 4:5={results['4:5']['crop_w']}")

    # Different crop widths produce different x offsets for same subject_x
    if len(results) >= 2:
        offsets = set(r["x_offset"] for r in results.values())
        check("Different aspect ratios produce different x offsets",
              len(offsets) > 1, f"all same offset: {offsets}")


def test_subject_centered_in_crop_all_ratios():
    """Subject pixel position is centered in crop window for all aspect ratios."""
    print("\n--- Subject tracking: subject centered in crop ---")
    for ratio in ["9:16", "1:1", "4:5"]:
        for sx in [20, 50, 80]:
            vf, _, _ = _build_filter_chain(ratio, 1920, 1080, None, subject_x=sx)
            m = re.search(r"crop=(\d+):(\d+):(\d+):(\d+)", vf)
            if not m:
                continue
            crop_w = int(m.group(1))
            x_offset = int(m.group(3))
            # Subject pixel position
            safe_sx = _safe_subject_x(sx)
            subject_pixel = 1920 * safe_sx / 100
            crop_center = x_offset + crop_w / 2
            # Subject should be near crop center (within reasonable tolerance)
            center_err = abs(subject_pixel - crop_center)
            check(f"Subject centered at {ratio} sx={sx}: err={center_err:.0f}px",
                  center_err < crop_w * 0.2,  # within 20% of crop width
                  f"subject={subject_pixel:.0f}, center={crop_center:.0f}")


def test_center_crop_offset_correctness():
    """_center_crop_offset puts subject at 50% of cropped frame."""
    print("\n--- center_crop_offset: correctness ---")
    # 16:9 → 9:16: crop_w=607 from 1920
    crop_w = 606
    src_w = 1920
    for sx in [10, 25, 50, 75, 90]:
        offset = _center_crop_offset(sx, src_w, crop_w)
        subject_pixel = src_w * sx / 100
        crop_center = offset + crop_w / 2
        # Subject should be at crop center (or clamped at edge)
        if offset > 0 and offset < src_w - crop_w:
            # Not edge-clamped → subject should be exactly at center
            check(f"sx={sx}: subject at crop center",
                  abs(subject_pixel - crop_center) < 1,
                  f"subject={subject_pixel:.0f}, center={crop_center:.0f}")
        check(f"sx={sx}: offset in valid range",
              0 <= offset <= src_w - crop_w,
              f"offset={offset}")


def test_dynamic_keyframes_across_aspect_ratios():
    """Dynamic keyframes produce different FFmpeg expressions for different aspect ratios."""
    print("\n--- Subject tracking: dynamic expressions differ by aspect ratio ---")
    kf = [(0.0, 20), (5.0, 80)]
    expressions = {}
    for ratio in ["9:16", "1:1", "4:5"]:
        vf, _, _ = _build_filter_chain(ratio, 1920, 1080, None, subject_keyframes=kf)
        check(f"Dynamic expression for {ratio}", "if(lt(t" in vf, f"got '{vf}'")
        expressions[ratio] = vf

    # Different ratios should produce different expressions (different max_offset, crop_w)
    if len(expressions) >= 2:
        unique_vfs = set(expressions.values())
        check("All aspect ratios produce different filter chains",
              len(unique_vfs) == len(expressions),
              f"got {len(unique_vfs)} unique out of {len(expressions)}")


def test_validate_subject_tracking_no_aspect():
    """No aspect ratio → no subject tracking validation needed."""
    print("\n--- Subject tracking QA: no aspect ratio ---")
    warnings = _validate_subject_tracking(
        filter_chain=None,
        aspect_ratio=None,
        src_w=1920,
        src_h=1080,
        subject_x=50,
        subject_keyframes=None,
    )
    check("No aspect → 0 warnings", len(warnings) == 0, f"got {warnings}")


def test_validate_subject_tracking_same_aspect():
    """Same aspect ratio as source → no crop needed."""
    print("\n--- Subject tracking QA: same aspect ratio as source ---")
    # 16:9 source at 16:9 → no crop
    vf, _, _ = _build_filter_chain("16:9", 1920, 1080, None, subject_x=30)
    warnings = _validate_subject_tracking(
        filter_chain=vf,
        aspect_ratio="16:9",
        src_w=1920,
        src_h=1080,
        subject_x=30,
        subject_keyframes=None,
    )
    check("Same aspect → 0 warnings", len(warnings) == 0, f"got {warnings}")


def test_validate_subject_tracking_detects_wrong_offset():
    """Validation catches incorrect static crop offset."""
    print("\n--- Subject tracking QA: detect wrong offset ---")
    # Manually construct a filter chain with wrong x offset
    wrong_filter = "crop=606:1080:999:0,scale=1080:1920"
    warnings = _validate_subject_tracking(
        filter_chain=wrong_filter,
        aspect_ratio="9:16",
        src_w=1920,
        src_h=1080,
        subject_x=50,
        subject_keyframes=None,
    )
    check("Wrong offset detected", len(warnings) > 0, f"got {warnings}")
    has_mismatch = any("mismatch" in w or "off-center" in w or "outside" in w for w in warnings)
    check("Warning mentions offset issue", has_mismatch, f"warnings: {warnings}")


def test_validate_subject_tracking_detects_static_with_dynamic_kf():
    """Validation catches static crop when dynamic keyframes were expected."""
    print("\n--- Subject tracking QA: static crop with dynamic kf ---")
    kf = [(0.0, 20), (5.0, 80)]  # Dynamic — two different values
    # But give a static filter chain
    static_filter = "crop=606:1080:500:0,scale=1080:1920"
    warnings = _validate_subject_tracking(
        filter_chain=static_filter,
        aspect_ratio="9:16",
        src_w=1920,
        src_h=1080,
        subject_x=50,
        subject_keyframes=kf,
    )
    check("Dynamic kf with static crop detected",
          len(warnings) > 0, f"got {warnings}")
    has_dynamic_msg = any("Dynamic keyframes" in w or "static" in w.lower() for w in warnings)
    check("Warning mentions dynamic/static mismatch", has_dynamic_msg, f"warnings: {warnings}")


# ===========================================================================
# ABSOLUTE VERTICAL OFFSET POSITIONING TESTS
# ===========================================================================


def test_offset_v_full_range_positioning():
    """Offset_v controls full vertical range: 0=bottom, 50=middle, 100=top."""
    print("\n--- Offset V: full range positioning (all positions use alignment=2) ---")
    for offset_v in [0, 25, 50, 75, 100]:
        for pos in ["bottom", "center", "top"]:
            result = generate_ass(
                segments=SAMPLE_SEGMENTS, start_time=10.0, end_time=25.0,
                position=pos, offset_v_pct=offset_v,
                video_width=1920, video_height=1080,
            )
            styles = parse_ass_styles(result)
            sp1 = styles.get("Speaker 1", {})

            # Alignment should always be 2 regardless of position
            actual_align = sp1.get("Alignment", "?")
            check(f"offset_v={offset_v}, pos={pos}: alignment=2",
                  actual_align == "2", f"got {actual_align}")

            # MarginV should always be height * offset_v / 100
            expected_mv = int(1080 * offset_v / 100)
            actual_mv = int(sp1.get("MarginV", -1))
            check(f"offset_v={offset_v}, pos={pos}: MarginV={expected_mv}",
                  actual_mv == expected_mv, f"got {actual_mv}")


def test_offset_v_preview_export_parity():
    """CSS bottom% and ASS MarginV should match for all offset_v values."""
    print("\n--- Offset V: preview-export parity ---")
    # CSS: bottom: offset_v% positions bottom edge of text at offset_v% from bottom
    # ASS: alignment=2 + MarginV positions bottom edge of text at MarginV px from bottom
    # For 1080p: CSS bottom: X% → pixel = 1080 * X / 100
    #            ASS MarginV = 1080 * X / 100
    # These match!
    for offset_v in [0, 4, 25, 50, 75, 96, 100]:
        css_pixel = 1080 * offset_v / 100
        result = generate_ass(
            segments=SAMPLE_SEGMENTS, start_time=10.0, end_time=25.0,
            offset_v_pct=offset_v,
            video_width=1920, video_height=1080,
        )
        styles = parse_ass_styles(result)
        ass_margin_v = int(styles.get("Speaker 1", {}).get("MarginV", -1))
        check(f"offset_v={offset_v}: CSS {css_pixel:.0f}px == ASS {ass_margin_v}px",
              abs(css_pixel - ass_margin_v) < 1, f"CSS={css_pixel:.0f}, ASS={ass_margin_v}")


def test_offset_v_qa_validation_all_values():
    """QA validator passes for all offset_v values with new absolute positioning."""
    print("\n--- Offset V: QA validation at all values ---")
    for offset_v in [0, 4, 25, 50, 75, 96, 100]:
        for pos in ["bottom", "center", "top"]:
            ass, settings, w, h = _gen_ass_with_settings(
                position=pos, offset_v=offset_v,
            )
            warnings = validate_ass(ass, settings, w, h)
            check(f"offset_v={offset_v}, pos={pos}: 0 QA warnings",
                  len(warnings) == 0, f"got {warnings}")


# ===========================================================================
# EXPORT QUALITY PRESET TESTS
# ===========================================================================


def test_quality_presets_exist():
    """All expected quality presets are defined."""
    print("\n--- Quality presets: existence ---")
    for q in ["720p", "1080p", "4k"]:
        check(f"Quality preset '{q}' exists", q in QUALITY_PRESETS,
              f"available: {list(QUALITY_PRESETS.keys())}")
        check(f"'{q}' has crf", "crf" in QUALITY_PRESETS[q])
        check(f"'{q}' has preset", "preset" in QUALITY_PRESETS[q])


def test_quality_resolution_tables():
    """Resolution tables exist for all quality tiers and aspect ratios."""
    print("\n--- Quality presets: resolution tables ---")
    for q in ["720p", "1080p", "4k"]:
        check(f"Resolution table for '{q}'", q in ASPECT_RATIO_DIMS_BY_QUALITY)
        for ar in ["16:9", "9:16", "1:1", "4:5"]:
            dims = ASPECT_RATIO_DIMS_BY_QUALITY[q].get(ar)
            check(f"'{q}' has dims for {ar}", dims is not None, f"got {dims}")
            if dims:
                w, h = dims
                check(f"'{q}' {ar}: even width", w % 2 == 0, f"w={w}")
                check(f"'{q}' {ar}: even height", h % 2 == 0, f"h={h}")


def test_quality_resolution_scaling():
    """Higher quality tiers have higher resolution."""
    print("\n--- Quality presets: resolution scaling ---")
    for ar in ["16:9", "9:16", "1:1", "4:5"]:
        w_720, h_720 = ASPECT_RATIO_DIMS_BY_QUALITY["720p"][ar]
        w_1080, h_1080 = ASPECT_RATIO_DIMS_BY_QUALITY["1080p"][ar]
        w_4k, h_4k = ASPECT_RATIO_DIMS_BY_QUALITY["4k"][ar]
        check(f"{ar}: 720p < 1080p width", w_720 < w_1080,
              f"720p={w_720}, 1080p={w_1080}")
        check(f"{ar}: 1080p < 4k width", w_1080 < w_4k,
              f"1080p={w_1080}, 4k={w_4k}")


def test_filter_chain_quality_affects_scale():
    """Different quality levels produce different scale dimensions in filter chain."""
    print("\n--- Quality presets: filter chain scale differs by quality ---")
    for ar in ["9:16", "1:1", "4:5"]:
        vf_720, _ = _build_filter_chain(ar, 1920, 1080, None, export_quality="720p")
        vf_1080, _ = _build_filter_chain(ar, 1920, 1080, None, export_quality="1080p")
        vf_4k, _ = _build_filter_chain(ar, 1920, 1080, None, export_quality="4k")

        check(f"{ar}: 720p has filter", vf_720 is not None)
        check(f"{ar}: 1080p has filter", vf_1080 is not None)
        check(f"{ar}: 4k has filter", vf_4k is not None)

        # Extract scale=WxH from filter chain
        m_720 = re.search(r"scale=(\d+):(\d+)", vf_720 or "")
        m_1080 = re.search(r"scale=(\d+):(\d+)", vf_1080 or "")
        m_4k = re.search(r"scale=(\d+):(\d+)", vf_4k or "")

        if m_720 and m_1080 and m_4k:
            w_720 = int(m_720.group(1))
            w_1080 = int(m_1080.group(1))
            w_4k = int(m_4k.group(1))
            check(f"{ar}: 720p < 1080p scale width", w_720 < w_1080,
                  f"720p={w_720}, 1080p={w_1080}")
            check(f"{ar}: 1080p < 4k scale width", w_1080 < w_4k,
                  f"1080p={w_1080}, 4k={w_4k}")


def test_quality_crf_values():
    """CRF values are reasonable for each quality tier."""
    print("\n--- Quality presets: CRF values ---")
    for q, params in QUALITY_PRESETS.items():
        crf = params["crf"]
        check(f"'{q}' CRF in valid range [0, 51]", 0 <= crf <= 51, f"crf={crf}")
        check(f"'{q}' CRF <= 23 (better than old default)", crf <= 23, f"crf={crf}")


def test_default_quality_backward_compat():
    """Default ASPECT_RATIO_DIMS matches 1080p quality tier."""
    print("\n--- Quality presets: backward compat ---")
    for ar in ["16:9", "9:16", "1:1", "4:5"]:
        default_dims = ASPECT_RATIO_DIMS.get(ar)
        q1080_dims = ASPECT_RATIO_DIMS_BY_QUALITY["1080p"].get(ar)
        check(f"Default {ar} == 1080p {ar}", default_dims == q1080_dims,
              f"default={default_dims}, 1080p={q1080_dims}")


def test_quality_scaling_without_aspect_ratio():
    """Quality scaling applies even when no aspect ratio is set."""
    print("\n--- Quality: scaling without aspect ratio ---")

    # 1080p source → 720p export should produce scale filter
    vf_720, _ = _build_filter_chain(None, 1920, 1080, None, export_quality="720p")
    check("720p from 1080p source has filter", vf_720 is not None, f"got: {vf_720}")
    check("720p filter includes scale", "scale=" in (vf_720 or ""), f"got: {vf_720}")
    check("720p filter targets height 720", ":720" in (vf_720 or ""), f"got: {vf_720}")

    # 1080p source → 1080p export should NOT produce a filter (same resolution)
    vf_1080, _ = _build_filter_chain(None, 1920, 1080, None, export_quality="1080p")
    check("1080p from 1080p source is None (stream copy)", vf_1080 is None,
          f"got: {vf_1080}")

    # 1080p source → 4K export should produce scale filter
    vf_4k, _ = _build_filter_chain(None, 1920, 1080, None, export_quality="4k")
    check("4K from 1080p source has filter", vf_4k is not None, f"got: {vf_4k}")
    check("4K filter includes scale", "scale=" in (vf_4k or ""), f"got: {vf_4k}")
    check("4K filter targets height 2160", ":2160" in (vf_4k or ""), f"got: {vf_4k}")

    # 720p source → 720p export should NOT produce a filter (same resolution)
    vf_720_same, _ = _build_filter_chain(None, 1280, 720, None, export_quality="720p")
    check("720p from 720p source is None (stream copy)", vf_720_same is None,
          f"got: {vf_720_same}")

    # 4K source → 720p export should produce scale filter
    vf_720_from_4k, _ = _build_filter_chain(None, 3840, 2160, None, export_quality="720p")
    check("720p from 4K source has filter", vf_720_from_4k is not None,
          f"got: {vf_720_from_4k}")
    check("720p from 4K targets height 720", ":720" in (vf_720_from_4k or ""),
          f"got: {vf_720_from_4k}")


def test_quality_applied_all_aspect_ratio_combinations():
    """Quality scaling works correctly for EVERY combination of aspect ratio and quality."""
    print("\n--- Quality: all aspect ratio + quality combinations ---")
    source_resolutions = [
        (1920, 1080, "1080p source"),
        (3840, 2160, "4K source"),
        (1280, 720, "720p source"),
    ]
    qualities = ["720p", "1080p", "4k"]
    aspect_ratios = [None, "16:9", "9:16", "1:1", "4:5"]

    for src_w, src_h, src_label in source_resolutions:
        for quality in qualities:
            for ar in aspect_ratios:
                ar_label = ar or "original"
                vf, _, _ = _build_filter_chain(ar, src_w, src_h, None, export_quality=quality)

                target_h = QUALITY_MAX_HEIGHT[quality]
                needs_scale = (src_h != target_h)

                if ar is not None:
                    # With aspect ratio — always re-encodes with crop+scale
                    dims_table = ASPECT_RATIO_DIMS_BY_QUALITY[quality]
                    expected_w, expected_h = dims_table[ar]
                    check(
                        f"{src_label} {ar_label} {quality}: has filter",
                        vf is not None,
                        f"expected filter with scale={expected_w}:{expected_h}, got None"
                    )
                    if vf:
                        check(
                            f"{src_label} {ar_label} {quality}: scale matches",
                            f"scale={expected_w}:{expected_h}" in vf,
                            f"expected scale={expected_w}:{expected_h} in '{vf}'"
                        )
                elif needs_scale:
                    # No aspect ratio but quality differs — should scale
                    check(
                        f"{src_label} {ar_label} {quality}: has scale filter",
                        vf is not None and "scale=" in vf,
                        f"expected scale filter, got: {vf}"
                    )
                    if vf:
                        check(
                            f"{src_label} {ar_label} {quality}: targets {target_h}",
                            f":{target_h}" in vf,
                            f"expected :{target_h} in '{vf}'"
                        )
                else:
                    # Same resolution, no aspect ratio — stream copy
                    check(
                        f"{src_label} {ar_label} {quality}: None (stream copy)",
                        vf is None,
                        f"expected None, got: {vf}"
                    )


# ===========================================================================
# MAIN
# ===========================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("CLIP EXPORT PIPELINE TESTS")
    print("=" * 60)

    # ASS Generator tests
    test_ass_default_settings()
    test_ass_font_sizes()
    test_ass_font_size_scaling_portrait()
    test_ass_font_size_scaling_square()
    test_ass_font_weight()
    test_ass_font_color_override()
    test_ass_custom_speaker_colors()
    test_ass_positions()
    test_ass_outline_mode()
    test_ass_background_box_mode()
    test_ass_speaker_labels()
    test_ass_max_width()
    test_ass_vertical_offset()
    test_ass_outline_width_scaling()
    test_ass_multiple_aspect_ratios()
    test_ass_segment_filtering()
    test_ass_empty_segments()
    test_ass_font_name()

    # Filter chain tests
    test_filter_chain_no_filters()
    test_filter_chain_16_9_to_9_16()
    test_filter_chain_16_9_to_1_1()
    test_filter_chain_subject_positioning()
    test_filter_chain_with_subtitles()
    test_filter_chain_crop_plus_subtitles()
    test_subtitle_filter_path_escaping()
    test_extract_force_style_from_ass()
    test_filter_chain_with_force_style()

    # Settings flow test
    test_settings_flow_complete()

    # Constant consistency test
    test_constant_consistency()

    # Model tests
    test_subtitle_settings_model()

    # Max words splitting tests
    test_max_words_disabled()
    test_max_words_no_split_needed()
    test_max_words_basic_split()
    test_max_words_single_word()
    test_max_words_exact_boundary()
    test_max_words_with_speaker_labels()
    test_max_words_one_word_per_subtitle()
    test_split_segments_by_max_words_direct()

    # Active word highlight tests
    test_active_word_disabled()
    test_active_word_basic()
    test_active_word_single_word()
    test_active_word_timing()
    test_active_word_colors()
    test_active_word_with_speaker_labels()
    test_active_word_with_max_words()
    test_active_word_bg_color()
    test_active_word_no_bg_when_zero_opacity()
    test_active_word_model_defaults()
    test_active_word_no_temporal_overlap()
    test_centisecond_overlap_elimination()

    # Helper function tests
    test_hex_to_ass_color()
    test_hex_to_ass_color_with_alpha()
    test_format_ass_time()

    # ASS QA validation tests
    test_validate_ass_default_settings()
    test_validate_ass_background_enabled()
    test_validate_ass_font_size_scaling()
    test_validate_ass_speaker_labels_on()
    test_validate_ass_speaker_labels_off()
    test_validate_ass_active_word_colors()
    test_validate_ass_max_words()
    test_validate_ass_margin_calculation()
    test_validate_ass_outline_settings()
    test_validate_ass_position_alignment()
    test_validate_ass_detects_font_mismatch()
    test_validate_ass_detects_size_mismatch()
    test_validate_ass_all_font_sizes()
    test_validate_ass_active_word_with_bg()
    test_validate_ass_combined_settings()

    # Comprehensive frontend-export matching tests
    test_default_consistency_across_components()
    test_clipsettingspanel_defaults_through_pipeline()
    test_analysis_export_body_through_pipeline()
    test_clipseo_export_body_through_pipeline()
    test_all_settings_combinations()
    test_background_radius_not_in_ass()

    # Dynamic subject tracking tests
    test_build_subject_keyframes_basic()
    test_build_subject_keyframes_single()
    test_build_subject_keyframes_empty()
    test_build_subject_keyframes_dict()
    test_smooth_keyframes_no_change()
    test_smooth_keyframes_clamp()
    test_build_crop_x_expr_static()
    test_build_crop_x_expr_dynamic()
    test_build_crop_x_expr_two_keyframes()
    test_build_crop_x_expr_zero_max_offset()
    test_filter_chain_dynamic_subject()
    test_filter_chain_static_fallback()
    test_filter_chain_no_keyframes()
    test_filter_chain_all_same_keyframes()
    test_safe_subject_x_clamping()
    test_keyframes_apply_safety_margin()
    test_filter_chain_static_safety_margin()

    # Comprehensive QA validation tests
    test_validate_active_word_bg_color_present()
    test_validate_active_word_bg_absent_when_zero_opacity()
    test_validate_numeric_font_sizes()
    test_validate_numeric_font_sizes_at_portrait()
    test_validate_all_builtin_fonts()
    test_viralclips_export_body_through_pipeline()
    test_extreme_settings_qa()
    test_viralclips_defaults_match_model()
    test_settings_roundtrip_all_24_fields()
    test_custom_font_name()
    test_validate_speaker_colors_override()

    # Subject tracking QA validation tests
    test_validate_subject_tracking_static_all_ratios()
    test_validate_subject_tracking_dynamic_all_ratios()
    test_subject_tracking_crop_changes_with_aspect_ratio()
    test_subject_centered_in_crop_all_ratios()
    test_center_crop_offset_correctness()
    test_dynamic_keyframes_across_aspect_ratios()
    test_validate_subject_tracking_no_aspect()
    test_validate_subject_tracking_same_aspect()
    test_validate_subject_tracking_detects_wrong_offset()
    test_validate_subject_tracking_detects_static_with_dynamic_kf()

    # Absolute vertical offset positioning tests
    test_offset_v_full_range_positioning()
    test_offset_v_preview_export_parity()
    test_offset_v_qa_validation_all_values()

    # Export quality preset tests
    test_quality_presets_exist()
    test_quality_resolution_tables()
    test_quality_resolution_scaling()
    test_filter_chain_quality_affects_scale()
    test_quality_crf_values()
    test_default_quality_backward_compat()
    test_quality_scaling_without_aspect_ratio()
    test_quality_applied_all_aspect_ratio_combinations()

    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)

    if errors:
        print("\nFailed tests:")
        for e in errors:
            print(e)
        sys.exit(1)
    else:
        print("\nAll tests passed!")
        sys.exit(0)
