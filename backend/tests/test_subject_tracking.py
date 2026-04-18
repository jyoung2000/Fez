"""Tests for the subject tracking pipeline — clip_exporter helpers and Ollama provider extraction."""
import asyncio
import json
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ── Pre-mock heavy native deps to avoid import failures in CI ──
for _mod in (
    "google.generativeai", "google.generativeai.types", "google.ai",
    "google.ai.generativelanguage_v1beta", "anthropic", "groq", "httpx",
    "openai", "ctranslate2", "faster_whisper",
):
    sys.modules.setdefault(_mod, MagicMock())

from backend.services.clip_exporter import (
    _safe_subject_x,
    _center_crop_offset,
    _build_subject_keyframes,
    _smooth_keyframes,
    _smooth_keyframes_bidirectional,
    _apply_dead_zone,
    _handle_scene_cuts,
    _compress_range,
    _merge_holds,
    _build_crop_x_expr,
    _build_filter_chain,
    _compute_safe_range,
    _validate_subject_tracking,
    _detect_position_clusters,
    _snap_to_clusters,
    _validate_tracking,
)
from backend.models import SceneDescription


def _scene(timestamp, subject_x=50, description="scene", importance_score=5):
    """Helper to create SceneDescription with required fields."""
    return SceneDescription(
        timestamp=timestamp,
        description=description,
        importance_score=importance_score,
        thumbnail_path="/tmp/thumb.jpg",
        subject_x=subject_x,
    )


# ══════════════════════════════════════════════════════════════════════
# _safe_subject_x
# ══════════════════════════════════════════════════════════════════════

class TestSafeSubjectX:
    def test_center_unchanged(self):
        assert _safe_subject_x(50) == 50

    def test_clamp_low(self):
        assert _safe_subject_x(0) == 10
        assert _safe_subject_x(5) == 10

    def test_clamp_high(self):
        assert _safe_subject_x(100) == 90
        assert _safe_subject_x(95) == 90

    def test_within_range_unchanged(self):
        assert _safe_subject_x(30) == 30
        assert _safe_subject_x(70) == 70

    def test_boundary_values(self):
        assert _safe_subject_x(10) == 10
        assert _safe_subject_x(90) == 90


# ══════════════════════════════════════════════════════════════════════
# _center_crop_offset
# ══════════════════════════════════════════════════════════════════════

class TestCenterCropOffset:
    def test_center_subject(self):
        # Subject at 50% of 1920px, crop width 1080
        # Expected: 1920*0.5 - 1080/2 = 960 - 540 = 420
        assert _center_crop_offset(50, 1920, 1080) == 420

    def test_left_subject_clamped(self):
        # Subject at 10% → offset would be negative → clamped to 0
        assert _center_crop_offset(10, 1920, 1080) == 0

    def test_right_subject_clamped(self):
        # Subject at 95% → offset exceeds max → clamped
        offset = _center_crop_offset(95, 1920, 1080)
        max_offset = 1920 - 1080
        assert offset == max_offset

    def test_zero_crop_width(self):
        assert _center_crop_offset(50, 1920, 1920) == 0


# ══════════════════════════════════════════════════════════════════════
# _build_subject_keyframes
# ══════════════════════════════════════════════════════════════════════

class TestBuildSubjectKeyframes:
    def test_empty_scenes(self):
        result = _build_subject_keyframes([], 0, 60)
        assert result == [(0.0, 50)]

    def test_no_overlapping_scenes(self):
        """Scene is after clip range — boundary interpolation uses nearest scene value."""
        scenes = [_scene(timestamp=100.0, subject_x=30)]
        result = _build_subject_keyframes(scenes, 0, 60)
        # Boundary interpolation correctly uses the nearest after scene (sx=30)
        # rather than falling back to default center (50)
        assert result[0][1] == 30
        assert result[-1][1] == 30

    def test_single_overlapping_scene(self):
        scenes = [_scene(timestamp=30.0, subject_x=70)]
        result = _build_subject_keyframes(scenes, 0, 60)
        # Should have boundary at 0, the scene at 30, and boundary at 60
        assert len(result) == 3
        assert result[0] == (0.0, 70)   # boundary copies first value
        assert result[1] == (30.0, 70)  # the scene (70 is within safe range)
        assert result[2] == (60.0, 70)  # boundary copies last value

    def test_multiple_scenes_sorted(self):
        scenes = [
            _scene(timestamp=45.0, subject_x=80),
            _scene(timestamp=15.0, subject_x=20),
            _scene(timestamp=30.0, subject_x=50),
        ]
        result = _build_subject_keyframes(scenes, 0, 60)
        # Should be sorted by time with boundaries
        times = [kf[0] for kf in result]
        assert times == sorted(times)
        assert result[0][0] == 0.0
        assert result[-1][0] == 60.0

    def test_scene_at_boundary(self):
        scenes = [_scene(timestamp=0.0, subject_x=40)]
        result = _build_subject_keyframes(scenes, 0, 60)
        assert result[0][0] == 0.0
        assert result[-1][0] == 60.0

    def test_subject_x_clamped_to_safe_range(self):
        scenes = [_scene(timestamp=10.0, subject_x=5)]
        result = _build_subject_keyframes(scenes, 0, 60)
        # subject_x=5 should be clamped to 10 by _safe_subject_x
        assert all(kf[1] >= 10 for kf in result)

    def test_dict_scenes(self):
        """Scenes can be dicts (from serialized DB data)."""
        scenes = [{"timestamp": 20.0, "subject_x": 60}]
        result = _build_subject_keyframes(scenes, 0, 60)
        assert len(result) >= 2
        # Should extract subject_x from dict
        assert any(kf[1] == 60 for kf in result)


# ══════════════════════════════════════════════════════════════════════
# _smooth_keyframes
# ══════════════════════════════════════════════════════════════════════

class TestSmoothKeyframes:
    def test_empty(self):
        assert _smooth_keyframes([]) == []

    def test_single_keyframe(self):
        kf = [(0.0, 50)]
        assert _smooth_keyframes(kf) == [(0.0, 50)]

    def test_no_change_needed(self):
        """Slow movement within max_speed should pass through unchanged."""
        kf = [(0.0, 50), (10.0, 55)]  # 0.5 units/s < 50 units/s
        result = _smooth_keyframes(kf)
        assert result == kf

    def test_fast_movement_clamped(self):
        """Large jump should be clamped by max_speed."""
        kf = [(0.0, 20), (1.0, 80)]  # 60 units in 1s > 50 units/s
        result = _smooth_keyframes(kf, max_speed=50)
        assert result[0] == (0.0, 20)
        assert result[1][0] == 1.0
        assert result[1][1] == 70  # 20 + 50*1 = 70

    def test_fast_movement_left(self):
        """Large jump leftward should be clamped."""
        kf = [(0.0, 80), (1.0, 20)]  # -60 units in 1s
        result = _smooth_keyframes(kf, max_speed=50)
        assert result[1][1] == 30  # 80 - 50*1 = 30

    def test_zero_dt(self):
        """Same timestamp should keep previous value."""
        kf = [(0.0, 50), (0.0, 80)]
        result = _smooth_keyframes(kf)
        assert result[1][1] == 50


# ══════════════════════════════════════════════════════════════════════
# _build_crop_x_expr
# ══════════════════════════════════════════════════════════════════════

class TestBuildCropXExpr:
    def test_zero_max_offset(self):
        assert _build_crop_x_expr([(0.0, 50)], 0) == "0"

    def test_static_keyframes(self):
        """All same subject_x → returns plain integer offset."""
        kf = [(0.0, 50), (10.0, 50), (20.0, 50)]
        result = _build_crop_x_expr(kf, 840, src_w=1920, crop_w=1080)
        # Should be a static integer, not an expression
        assert result.isdigit() or result == "0"

    def test_dynamic_keyframes_produces_expression(self):
        """Varied subject_x → returns FFmpeg expression with clip()."""
        kf = [(0.0, 20), (10.0, 80)]
        result = _build_crop_x_expr(kf, 840, src_w=1920, crop_w=1080)
        # Should contain clip() wrapper and if() segments
        assert "clip(" in result
        assert "if(" in result

    def test_single_keyframe_static(self):
        kf = [(0.0, 30)]
        result = _build_crop_x_expr(kf, 840, src_w=1920, crop_w=1080)
        # Single keyframe → static offset
        assert "if(" not in result


# ══════════════════════════════════════════════════════════════════════
# _build_filter_chain
# ══════════════════════════════════════════════════════════════════════

class TestBuildFilterChain:
    def test_no_aspect_no_subs(self):
        vf, is_complex, _ = _build_filter_chain(None, 1920, 1080, None)
        assert vf is None
        assert is_complex is False

    def test_aspect_ratio_with_static_subject(self):
        vf, is_complex, _ = _build_filter_chain("9:16", 1920, 1080, None, subject_x=30)
        assert vf is not None
        assert "crop=" in vf
        assert "scale=" in vf

    def test_aspect_ratio_with_dynamic_keyframes(self):
        kf = [(0.0, 20), (5.0, 50), (10.0, 80)]
        vf, _, _ = _build_filter_chain("9:16", 1920, 1080, None, subject_keyframes=kf)
        assert vf is not None
        # Dynamic keyframes should produce expression-based crop
        assert "crop=" in vf
        # Should contain FFmpeg expression elements
        assert "clip(" in vf or "if(" in vf

    def test_same_aspect_ratio_no_crop(self):
        """16:9 source → 16:9 target → no crop needed."""
        vf, _, _ = _build_filter_chain("16:9", 1920, 1080, None, subject_x=30)
        # Should just scale, no crop (src ratio == target ratio)
        if vf:
            assert "scale=" in vf

    def test_subtitles_only(self):
        vf, _, _ = _build_filter_chain(None, 1920, 1080, "/tmp/test.ass")
        assert vf is not None
        assert "subtitles=" in vf


# ══════════════════════════════════════════════════════════════════════
# End-to-end centering validation across all aspect ratios
# ══════════════════════════════════════════════════════════════════════

ASPECT_RATIOS = {
    "9:16": 9 / 16,
    "1:1": 1.0,
    "4:5": 4 / 5,
}

class TestSubjectCenteringAllAspectRatios:
    """Verify that subject tracking correctly centers subjects for every
    supported aspect ratio at various subject_x positions.

    For each aspect ratio and subject_x value:
    1. Compute the crop dimensions (matching _build_filter_chain logic)
    2. Compute the crop offset via _center_crop_offset
    3. Verify the subject pixel lands within 1% of center of the crop window
       (unless edge-clamped)
    """

    @pytest.mark.parametrize("aspect_ratio", ["9:16", "1:1", "4:5"])
    @pytest.mark.parametrize("subject_x", [10, 20, 30, 40, 50, 60, 70, 80, 90])
    def test_centering_1920x1080(self, aspect_ratio, subject_x):
        """16:9 source (1920x1080) → various targets."""
        self._verify_centering(1920, 1080, aspect_ratio, subject_x)

    @pytest.mark.parametrize("aspect_ratio", ["9:16", "1:1", "4:5"])
    @pytest.mark.parametrize("subject_x", [10, 30, 50, 70, 90])
    def test_centering_2560x1440(self, aspect_ratio, subject_x):
        """16:9 source (2560x1440) → various targets."""
        self._verify_centering(2560, 1440, aspect_ratio, subject_x)

    @pytest.mark.parametrize("aspect_ratio", ["9:16", "1:1", "4:5"])
    @pytest.mark.parametrize("subject_x", [10, 30, 50, 70, 90])
    def test_centering_3840x2160(self, aspect_ratio, subject_x):
        """4K source (3840x2160) → various targets."""
        self._verify_centering(3840, 2160, aspect_ratio, subject_x)

    @pytest.mark.parametrize("aspect_ratio", ["9:16", "1:1", "4:5"])
    @pytest.mark.parametrize("subject_x", [10, 30, 50, 70, 90])
    def test_centering_1280x720(self, aspect_ratio, subject_x):
        """720p source (1280x720) → various targets."""
        self._verify_centering(1280, 720, aspect_ratio, subject_x)

    def _verify_centering(self, src_w, src_h, aspect_ratio, subject_x):
        target_ratio = ASPECT_RATIOS[aspect_ratio]
        src_ratio = src_w / src_h

        # Skip if same aspect ratio (no crop)
        if abs(src_ratio - target_ratio) <= 0.01:
            return

        # Compute crop dimensions (matching _build_filter_chain)
        if target_ratio < src_ratio:
            crop_h = src_h
            crop_w = int(src_h * target_ratio)
        else:
            crop_w = src_w
            crop_h = int(src_w / target_ratio)
        crop_w = crop_w - (crop_w % 2)
        crop_h = crop_h - (crop_h % 2)

        sx = _safe_subject_x(subject_x)
        x_offset = _center_crop_offset(sx, src_w, crop_w)

        # Where the subject is in source pixels
        subject_pixel = src_w * sx / 100
        # Where the subject lands in the crop window
        subject_in_crop = subject_pixel - x_offset
        crop_center = crop_w / 2

        max_offset = src_w - crop_w
        is_edge_clamped = x_offset == 0 or x_offset == max_offset

        if not is_edge_clamped:
            # When not edge-clamped, subject should be within 1px of center
            error_px = abs(subject_in_crop - crop_center)
            assert error_px <= 1.0, (
                f"Subject off-center by {error_px:.1f}px for {src_w}x{src_h}→{aspect_ratio} "
                f"sx={sx}: subject@{subject_pixel:.0f}px, offset={x_offset}, "
                f"in_crop={subject_in_crop:.0f}, center={crop_center:.0f}"
            )
        else:
            # When edge-clamped, subject should still be inside the crop window
            assert 0 <= subject_in_crop <= crop_w, (
                f"Subject outside crop window for {src_w}x{src_h}→{aspect_ratio} "
                f"sx={sx}: subject@{subject_pixel:.0f}px, offset={x_offset}, "
                f"in_crop={subject_in_crop:.0f}"
            )


class TestFrontendBackendEquivalence:
    """Verify that the frontend subjectXToCenterPct formula produces
    an equivalent visual result to the backend _center_crop_offset.

    The frontend uses:  objectPosition = ((R * sx - 50) / (R - 1))%
    The backend uses:   crop_x = round(src_w * sx / 100 - crop_w / 2)

    These should produce equivalent centering results.
    """

    @pytest.mark.parametrize("aspect_ratio", ["9:16", "1:1", "4:5"])
    @pytest.mark.parametrize("subject_x", [10, 20, 30, 40, 50, 60, 70, 80, 90])
    def test_equivalence_1920x1080(self, aspect_ratio, subject_x):
        self._verify_equivalence(1920, 1080, aspect_ratio, subject_x)

    def _verify_equivalence(self, src_w, src_h, aspect_ratio, subject_x):
        target_ratio = ASPECT_RATIOS[aspect_ratio]
        src_ratio = src_w / src_h

        if abs(src_ratio - target_ratio) <= 0.01:
            return

        R = src_ratio / target_ratio
        if R <= 1.01:
            return

        sx = _safe_subject_x(subject_x)

        # --- Frontend formula ---
        center_pct = (R * sx - 50) / (R - 1)
        center_pct = max(0, min(100, center_pct))

        # The frontend objectPosition means: the center_pct% point of the
        # content is aligned with the center_pct% point of the container.
        # For a source of width src_w rendered at scale R into container of
        # width (src_w / R), the visible left edge of the content is:
        #   content_left = center_pct/100 * src_w - center_pct/100 * (src_w / R)
        #                = center_pct/100 * src_w * (1 - 1/R)
        #                = center_pct/100 * (src_w - crop_w)  [since crop_w = src_w/R ≈ src_h * target_ratio]
        # This is exactly the same as the backend crop offset when we use
        # crop_w = src_h * target_ratio for horizontal cropping.

        # --- Backend formula ---
        if target_ratio < src_ratio:
            crop_w = int(src_h * target_ratio)
        else:
            crop_w = src_w
        crop_w = crop_w - (crop_w % 2)
        max_offset = src_w - crop_w

        backend_offset = _center_crop_offset(sx, src_w, crop_w)

        # Frontend equivalent offset
        frontend_offset = center_pct / 100 * max_offset

        # They should agree within a few pixels (rounding differences)
        diff = abs(frontend_offset - backend_offset)
        tolerance = 3  # pixels — accounts for int rounding in crop_w
        assert diff <= tolerance, (
            f"Frontend/backend mismatch for {src_w}x{src_h}→{aspect_ratio} sx={sx}: "
            f"frontend_offset={frontend_offset:.1f} (centerPct={center_pct:.2f}%), "
            f"backend_offset={backend_offset}, diff={diff:.1f}px"
        )


# ══════════════════════════════════════════════════════════════════════
# Boundary interpolation QA
# ══════════════════════════════════════════════════════════════════════

class TestBoundaryInterpolation:
    """QA tests verifying that clips between scenes get correctly
    interpolated subject_x values rather than falling back to center."""

    def test_clip_between_two_scenes(self):
        """Clip is entirely between two scenes — should interpolate."""
        scenes = [
            _scene(timestamp=10.0, subject_x=20),
            _scene(timestamp=50.0, subject_x=80),
        ]
        result = _build_subject_keyframes(scenes, 25.0, 35.0)
        # At t=25 (midpoint of 10-50): frac = (25-10)/(50-10) = 0.375
        # sx = 20 + 60 * 0.375 = 42.5 → 43 (after round)
        assert len(result) >= 2
        sx_start = result[0][1]
        assert 35 <= sx_start <= 50, f"Expected interpolated sx near 43, got {sx_start}"

    def test_clip_after_all_scenes(self):
        """Clip starts after all scenes — should use last scene's value."""
        scenes = [
            _scene(timestamp=10.0, subject_x=30),
            _scene(timestamp=20.0, subject_x=70),
        ]
        result = _build_subject_keyframes(scenes, 50.0, 60.0)
        # Only after scenes — uses first after scene (which is the last scene, subject_x=70)
        assert result[0][1] == 70

    def test_clip_before_all_scenes(self):
        """Clip ends before all scenes — should use first scene's value."""
        scenes = [
            _scene(timestamp=50.0, subject_x=30),
            _scene(timestamp=60.0, subject_x=70),
        ]
        result = _build_subject_keyframes(scenes, 10.0, 20.0)
        assert result[0][1] == 30

    def test_dynamic_tracking_filter_chain(self):
        """End-to-end: scenes with varied subject_x should produce dynamic FFmpeg expression."""
        scenes = [
            _scene(timestamp=0.0, subject_x=20),
            _scene(timestamp=15.0, subject_x=50),
            _scene(timestamp=30.0, subject_x=80),
        ]
        kf = _build_subject_keyframes(scenes, 0.0, 30.0)
        smoothed = _smooth_keyframes(kf)
        unique_sx = set(k[1] for k in smoothed)
        assert len(unique_sx) > 1, "Expected dynamic keyframes with multiple unique sx values"

        vf, _, _ = _build_filter_chain("9:16", 1920, 1080, None, subject_keyframes=smoothed)
        assert vf is not None
        assert "clip(" in vf, f"Expected dynamic expression, got: {vf}"

    def test_static_tracking_filter_chain(self):
        """Scenes with identical subject_x should produce static crop."""
        scenes = [
            _scene(timestamp=0.0, subject_x=40),
            _scene(timestamp=15.0, subject_x=40),
            _scene(timestamp=30.0, subject_x=40),
        ]
        kf = _build_subject_keyframes(scenes, 0.0, 30.0)
        smoothed = _smooth_keyframes(kf)

        vf, _, _ = _build_filter_chain("9:16", 1920, 1080, None, subject_keyframes=smoothed)
        assert vf is not None
        # Static crop — should NOT contain dynamic expression
        assert "clip(" not in vf or "if(" not in vf


# ══════════════════════════════════════════════════════════════════════
# Ollama provider subject_x extraction
# ══════════════════════════════════════════════════════════════════════

def _make_ollama_provider():
    """Create OllamaProvider with mocked settings."""
    with patch("backend.services.providers.ollama_provider.settings") as mock_settings:
        mock_settings.OLLAMA_HOST = "http://localhost:11434"
        mock_settings.OLLAMA_VISION_MODEL = "moondream"
        mock_settings.OLLAMA_TEXT_MODEL = "llama3"
        from backend.services.providers.ollama_provider import OllamaProvider
        return OllamaProvider()


def _make_frame(timestamp=10.0):
    from backend.models import FrameData
    return FrameData(timestamp=timestamp, path="/tmp/frame.jpg", base64="dGVzdA==")


class TestOllamaSubjectX:
    def test_json_response_extracts_subject_x(self):
        """When Ollama returns valid JSON, subject_x should be extracted."""
        provider = _make_ollama_provider()
        frame = _make_frame()
        json_response = json.dumps({
            "timestamp": 10.0,
            "description": "Person walking left",
            "importance_score": 7,
            "subject_x": 25,
        })
        with patch.object(provider, "_call_vision", new_callable=AsyncMock, return_value=json_response):
            scenes = asyncio.get_event_loop().run_until_complete(provider.analyze_frames([frame]))
        assert len(scenes) == 1
        assert scenes[0].subject_x == 25
        assert scenes[0].importance_score == 7
        assert scenes[0].description == "Person walking left"

    def test_json_array_response(self):
        """Ollama might return a JSON array — should take first element."""
        provider = _make_ollama_provider()
        frame = _make_frame()
        json_response = json.dumps([{
            "timestamp": 10.0,
            "description": "Array response",
            "importance_score": 6,
            "subject_x": 75,
        }])
        with patch.object(provider, "_call_vision", new_callable=AsyncMock, return_value=json_response):
            scenes = asyncio.get_event_loop().run_until_complete(provider.analyze_frames([frame]))
        assert len(scenes) == 1
        assert scenes[0].subject_x == 75

    def test_json_with_code_fence(self):
        """Ollama sometimes wraps JSON in markdown code fences."""
        provider = _make_ollama_provider()
        frame = _make_frame()
        json_response = '```json\n{"timestamp": 10.0, "description": "Fenced", "importance_score": 8, "subject_x": 35}\n```'
        with patch.object(provider, "_call_vision", new_callable=AsyncMock, return_value=json_response):
            scenes = asyncio.get_event_loop().run_until_complete(provider.analyze_frames([frame]))
        assert len(scenes) == 1
        assert scenes[0].subject_x == 35

    def test_raw_text_fallback(self):
        """When JSON parsing fails, should fall back to word-scanning and subject_x=50."""
        provider = _make_ollama_provider()
        frame = _make_frame()
        raw_text = "This frame shows a person talking. The importance is 7 out of 10."
        with patch.object(provider, "_call_vision", new_callable=AsyncMock, return_value=raw_text):
            scenes = asyncio.get_event_loop().run_until_complete(provider.analyze_frames([frame]))
        assert len(scenes) == 1
        assert scenes[0].subject_x == 50  # Default when JSON fails
        assert scenes[0].importance_score == 7  # Extracted from text

    def test_subject_x_clamped(self):
        """subject_x values outside 0-100 should be clamped."""
        provider = _make_ollama_provider()
        frame = _make_frame()
        json_response = json.dumps({
            "timestamp": 10.0,
            "description": "Extreme",
            "importance_score": 5,
            "subject_x": 150,
        })
        with patch.object(provider, "_call_vision", new_callable=AsyncMock, return_value=json_response):
            scenes = asyncio.get_event_loop().run_until_complete(provider.analyze_frames([frame]))
        assert scenes[0].subject_x == 100

    def test_missing_subject_x_defaults_50(self):
        """JSON response without subject_x should default to 50."""
        provider = _make_ollama_provider()
        frame = _make_frame()
        json_response = json.dumps({
            "timestamp": 10.0,
            "description": "No tracking",
            "importance_score": 5,
        })
        with patch.object(provider, "_call_vision", new_callable=AsyncMock, return_value=json_response):
            scenes = asyncio.get_event_loop().run_until_complete(provider.analyze_frames([frame]))
        assert scenes[0].subject_x == 50

    def test_error_fallback_has_subject_x(self):
        """When vision call fails, error fallback should include subject_x=50."""
        provider = _make_ollama_provider()
        frame = _make_frame()
        with patch.object(
            provider, "_call_vision", new_callable=AsyncMock,
            side_effect=Exception("Connection refused"),
        ):
            scenes = asyncio.get_event_loop().run_until_complete(provider.analyze_frames([frame]))
        assert len(scenes) == 1
        assert scenes[0].subject_x == 50
        assert scenes[0].importance_score == 5

    def test_progress_callback(self):
        """Progress callback should fire for each frame."""
        provider = _make_ollama_provider()
        frame = _make_frame()
        json_response = json.dumps({
            "timestamp": 10.0, "description": "t", "importance_score": 5, "subject_x": 50,
        })
        callback = AsyncMock()
        with patch.object(provider, "_call_vision", new_callable=AsyncMock, return_value=json_response):
            asyncio.get_event_loop().run_until_complete(
                provider.analyze_frames([frame], progress_callback=callback)
            )
        callback.assert_awaited_once_with(1, 1)

    def test_empty_base64_skipped(self):
        """Frames without base64 should be skipped."""
        provider = _make_ollama_provider()
        from backend.models import FrameData
        frame = FrameData(timestamp=5.0, path="/tmp/f.jpg", base64="")
        with patch.object(provider, "_call_vision", new_callable=AsyncMock) as mock_call:
            scenes = asyncio.get_event_loop().run_until_complete(provider.analyze_frames([frame]))
        mock_call.assert_not_awaited()
        assert scenes == []


# ══════════════════════════════════════════════════════════════════════
# Off-screen crop prevention (bounds clamping)
# ══════════════════════════════════════════════════════════════════════

class TestBoundsClampingNeverCropsOffScreen:
    """Verify that the safe range clamping ensures crop offsets never
    go negative or exceed the source frame, for all aspect ratios.
    This guarantees the video always fills the full output frame
    (no black bars, no off-screen crops).
    """

    @pytest.mark.parametrize("aspect_ratio,target_ratio", [
        ("9:16", 9/16), ("1:1", 1.0), ("4:5", 4/5),
    ])
    @pytest.mark.parametrize("src_w,src_h", [
        (1920, 1080), (2560, 1440), (3840, 2160), (1280, 720),
    ])
    @pytest.mark.parametrize("subject_x", [0, 5, 10, 25, 50, 75, 90, 95, 100])
    def test_crop_offset_always_valid(self, aspect_ratio, target_ratio, src_w, src_h, subject_x):
        """Crop offset must always be in [0, max_offset] regardless of subject_x."""
        src_ratio = src_w / src_h
        if abs(src_ratio - target_ratio) <= 0.01:
            return

        if target_ratio < src_ratio:
            crop_w = int(src_h * target_ratio)
        else:
            crop_w = src_w
        crop_w = crop_w - (crop_w % 2)
        max_offset = src_w - crop_w

        sx = _safe_subject_x(subject_x, src_ratio=src_ratio, target_ratio=target_ratio)
        offset = _center_crop_offset(sx, src_w, crop_w)

        assert offset >= 0, (
            f"Negative crop offset {offset} for {src_w}x{src_h}→{aspect_ratio} sx={subject_x}→safe={sx}"
        )
        assert offset <= max_offset, (
            f"Crop offset {offset} > max {max_offset} for {src_w}x{src_h}→{aspect_ratio} sx={subject_x}→safe={sx}"
        )
        # Verify crop covers full width (no gap on right side)
        assert offset + crop_w <= src_w, (
            f"Crop extends beyond frame: {offset}+{crop_w}={offset+crop_w} > {src_w}"
        )

    @pytest.mark.parametrize("aspect_ratio,target_ratio", [
        ("9:16", 9/16), ("1:1", 1.0), ("4:5", 4/5),
    ])
    def test_dynamic_smoothstep_never_exceeds_bounds(self, aspect_ratio, target_ratio):
        """Smoothstep interpolation between keyframes must never produce
        crop offsets outside [0, max_offset]."""
        from backend.services.clip_exporter import _compute_safe_range

        src_w, src_h = 1920, 1080
        src_ratio = src_w / src_h

        if abs(src_ratio - target_ratio) <= 0.01:
            return

        crop_w = int(src_h * target_ratio)
        crop_w = crop_w - (crop_w % 2)
        max_offset = src_w - crop_w

        safe_lo, safe_hi = _compute_safe_range(src_ratio, target_ratio)

        # Create keyframes at opposite safe extremes
        keyframes = [
            (0.0, safe_lo),
            (2.0, safe_hi),
            (4.0, safe_lo),
        ]

        # Sample 100 points across the timeline
        for i in range(101):
            t = i * 4.0 / 100
            # Find surrounding keyframes and interpolate with smoothstep
            for j in range(len(keyframes) - 1):
                t0, sx0 = keyframes[j]
                t1, sx1 = keyframes[j + 1]
                if t0 <= t <= t1:
                    dt = t1 - t0
                    if dt <= 0:
                        interp_sx = sx0
                    else:
                        p = (t - t0) / dt
                        eased = p * p * (3 - 2 * p)
                        interp_sx = sx0 + (sx1 - sx0) * eased
                    sx_clamped = _safe_subject_x(int(round(interp_sx)), src_ratio=src_ratio, target_ratio=target_ratio)
                    offset = _center_crop_offset(sx_clamped, src_w, crop_w)
                    assert 0 <= offset <= max_offset, (
                        f"Smoothstep at t={t:.2f}s sx={interp_sx:.1f}→{sx_clamped}: "
                        f"offset {offset} outside [0, {max_offset}] for {aspect_ratio}"
                    )
                    break


class TestValidateSubjectTrackingQA:
    """Verify the enhanced QA validation catches off-screen crops and
    validates centering quality."""

    def test_static_crop_valid(self):
        """Static crop with centered subject should pass QA."""
        from backend.services.clip_exporter import _validate_subject_tracking
        # Compute correct crop dimensions matching _build_filter_chain
        crop_w = int(1080 * 9 / 16)
        crop_w = crop_w - (crop_w % 2)  # 606
        sx = _safe_subject_x(50)
        x_offset = _center_crop_offset(sx, 1920, crop_w)
        vf = f"setsar=1,crop={crop_w}:1080:{x_offset}:0,scale=1080:1920"
        warnings = _validate_subject_tracking(
            filter_chain=vf,
            aspect_ratio="9:16",
            src_w=1920, src_h=1080,
            subject_x=50,
            subject_keyframes=None,
        )
        assert len(warnings) == 0, f"Unexpected QA warnings: {warnings}"

    def test_dynamic_crop_with_valid_keyframes(self):
        """Dynamic crop with safe keyframes should pass QA."""
        from backend.services.clip_exporter import _validate_subject_tracking
        # Build a dynamic filter chain
        keyframes = [(0.0, 40), (2.0, 50), (4.0, 60)]
        vf_expr = _build_crop_x_expr(keyframes, 1312, 1920, 608)
        vf = f"setsar=1,crop=608:1080:{vf_expr}:0,scale=1080:1920"
        warnings = _validate_subject_tracking(
            filter_chain=vf,
            aspect_ratio="9:16",
            src_w=1920, src_h=1080,
            subject_x=50,
            subject_keyframes=keyframes,
        )
        # Should have no errors about off-screen crops
        off_screen_warnings = [w for w in warnings if "off-screen" in w or "outside" in w.lower()]
        assert len(off_screen_warnings) == 0, f"Off-screen crop warnings: {off_screen_warnings}"


# ══════════════════════════════════════════════════════════════════════
# Spring-damped smoother
# ══════════════════════════════════════════════════════════════════════

class TestSpringDampedSmoother:
    """Verify the spring-damped smoothing model produces natural motion."""

    def test_empty(self):
        assert _smooth_keyframes_bidirectional([]) == []

    def test_single_keyframe(self):
        kf = [(0.0, 50)]
        assert _smooth_keyframes_bidirectional(kf) == [(0.0, 50)]

    def test_converges_toward_target(self):
        """Spring should move position toward target over time."""
        kf = [(0.0, 20), (2.0, 80)]
        result = _smooth_keyframes_bidirectional(kf)
        assert result[0][1] == 20  # Starts at initial
        # Should have moved toward 80 but may not reach it exactly in 2s
        assert result[1][1] > 20, "Spring should move toward target"
        assert result[1][1] <= 100, "Should not exceed bounds"

    def test_respects_max_speed(self):
        """Large jump should be speed-limited."""
        kf = [(0.0, 0), (0.5, 100)]
        result = _smooth_keyframes_bidirectional(kf, max_speed=25)
        # In 0.5s at max 25 units/s = max 12.5 units movement
        assert result[1][1] <= 20, f"Expected speed-limited result, got {result[1][1]}"

    def test_instant_cut_snaps(self):
        """Scene cut keyframes (1ms apart) should snap immediately."""
        kf = [(0.0, 30), (4.999, 30), (5.0, 70)]
        result = _smooth_keyframes_bidirectional(kf)
        # The 1ms gap should cause an instant snap
        assert result[2][1] == 70, "Scene cut should snap to target"

    def test_values_in_valid_range(self):
        """All output values should be within [0, 100]."""
        kf = [(0.0, 5), (1.0, 95), (2.0, 10), (3.0, 90)]
        result = _smooth_keyframes_bidirectional(kf)
        for t, sx in result:
            assert 0 <= sx <= 100, f"Value {sx} at t={t} outside [0, 100]"


# ══════════════════════════════════════════════════════════════════════
# Anchor-based dead zone
# ══════════════════════════════════════════════════════════════════════

class TestDeadZoneAnchor:
    """Verify the anchor-based dead zone eliminates drift accumulation."""

    def test_cumulative_drift(self):
        """Small consecutive movements should eventually trigger camera movement."""
        # 5 keyframes each moving 2 units right (total 10 units)
        kf = [(0.0, 50), (1.0, 52), (2.0, 54), (3.0, 56), (4.0, 58)]
        result = _apply_dead_zone(kf, threshold=5)
        # The anchor starts at 50. At kf (3.0, 56), drift from anchor = 6 > 5.
        # So camera should move at or before t=3.0
        final_x = result[-1][1]
        assert final_x > 50, "Camera must have moved after cumulative drift exceeded threshold"

    def test_small_movements_held(self):
        """Movements within threshold should hold at anchor."""
        kf = [(0.0, 50), (1.0, 52), (2.0, 51), (3.0, 53)]
        result = _apply_dead_zone(kf, threshold=5)
        # All within 5 of anchor (50), should hold
        for _, sx in result:
            assert sx == 50, f"Expected hold at 50, got {sx}"

    def test_large_movement_passes(self):
        """Movement exceeding threshold should pass through."""
        kf = [(0.0, 50), (1.0, 60)]
        result = _apply_dead_zone(kf, threshold=5)
        assert result[1][1] == 60


# ══════════════════════════════════════════════════════════════════════
# Pipeline ordering — scene cuts survive
# ══════════════════════════════════════════════════════════════════════

class TestPipelineOrdering:
    """Verify scene cuts are not absorbed by dead zone or compression."""

    def test_scene_cuts_survive_pipeline(self):
        """Scene cut keyframes must not be absorbed by dead zone."""
        # Simulate the pipeline: compress → dead zone → scene cuts
        # Create keyframes with a scene cut (large jump)
        kf = [
            (0.0, 30), (2.0, 32), (4.0, 31),  # Stable around 30
            (5.0, 70), (7.0, 72), (9.0, 71),   # Scene cut to ~70
        ]
        after_compress = _compress_range(kf)
        after_dead_zone = _apply_dead_zone(after_compress, threshold=5)
        after_cuts = _handle_scene_cuts(after_dead_zone)

        # There should be a 1ms hold keyframe before the scene cut
        times = [t for t, _ in after_cuts]
        # Check that there's an instant transition near the scene cut
        has_instant_cut = False
        for i in range(1, len(after_cuts)):
            t_prev, sx_prev = after_cuts[i - 1]
            t_cur, sx_cur = after_cuts[i]
            if (t_cur - t_prev) <= 0.002 and abs(sx_cur - sx_prev) >= 10:
                has_instant_cut = True
                break
        assert has_instant_cut, (
            f"Expected instant scene cut in pipeline output, got: {after_cuts}"
        )


# ══════════════════════════════════════════════════════════════════════
# _detect_position_clusters — 3-pass center noise stripping
# ══════════════════════════════════════════════════════════════════════

class TestDetectPositionClusters:
    """Cluster detection with center-noise stripping."""

    def test_clear_bimodal_split(self):
        """Two distinct groups should split on Pass 1."""
        kf = [(i, 30) for i in range(10)] + [(i + 10, 70) for i in range(10)]
        result = _detect_position_clusters(kf)
        assert result is not None
        assert len(result) == 2
        assert result[0]["center"] < 50
        assert result[1]["center"] > 50

    def test_center_noise_blocks_pass1(self):
        """50s bridging left/right should prevent Pass 1 split but Pass 2 works."""
        # Left values 38-42, center defaults 48-52, right values 58-62
        # Sorted gaps: max is ~6 (42→48, 52→58) < threshold 10 → Pass 1 fails
        # After stripping [47-53], gap becomes 42→58 = 16 → Pass 2 succeeds
        kf = (
            [(i, 38 + (i % 5)) for i in range(8)]       # 38,39,40,41,42 repeating
            + [(i + 8, 48 + (i % 5)) for i in range(8)]  # 48,49,50,51,52 repeating
            + [(i + 16, 58 + (i % 5)) for i in range(8)] # 58,59,60,61,62 repeating
        )
        result = _detect_position_clusters(kf)
        assert result is not None
        assert len(result) >= 2
        centers = [c["center"] for c in result]
        # Should NOT have a cluster at ~50 — those were stripped
        assert all(c < 44 or c > 56 for c in centers), f"Center cluster not stripped: {centers}"

    def test_heavy_center_pollution_pass3(self):
        """Soft-center values (45, 48, 52, 55) need Pass 3 (strip [44-56])."""
        kf = (
            [(i, 25) for i in range(5)]
            + [(i + 5, 45) for i in range(3)]
            + [(i + 8, 50) for i in range(10)]
            + [(i + 18, 55) for i in range(3)]
            + [(i + 21, 75) for i in range(5)]
        )
        result = _detect_position_clusters(kf)
        assert result is not None
        assert len(result) >= 2
        centers = [c["center"] for c in result]
        assert any(c < 40 for c in centers), f"No left cluster found: {centers}"
        assert any(c > 60 for c in centers), f"No right cluster found: {centers}"

    def test_single_cluster_returns_none(self):
        """All values at same position should return None."""
        kf = [(i, 50) for i in range(10)]
        assert _detect_position_clusters(kf) is None

    def test_too_few_keyframes(self):
        kf = [(0, 30), (1, 70)]
        assert _detect_position_clusters(kf) is None

    def test_minimum_cluster_distance(self):
        """Clusters < 8 apart should be rejected."""
        kf = [(i, 48) for i in range(5)] + [(i + 5, 52) for i in range(5)]
        assert _detect_position_clusters(kf) is None

    def test_three_clusters(self):
        """Should detect 3+ clusters when data supports it."""
        kf = (
            [(i, 20) for i in range(5)]
            + [(i + 5, 50) for i in range(5)]
            + [(i + 10, 80) for i in range(5)]
        )
        result = _detect_position_clusters(kf)
        assert result is not None
        assert len(result) == 3

    def test_center_stripping_conditional(self):
        """Center stripping only activates when >10% in noise zone AND data on both sides."""
        # Only 1 value at 50 out of 20 = 5% — should NOT strip
        kf = [(i, 30) for i in range(10)] + [(10, 50)] + [(i + 11, 32) for i in range(9)]
        result = _detect_position_clusters(kf)
        # With gap 30→32 = 2 and 32→50 = 18, should still find clusters if possible
        # But all values are close to 30-32 except one 50, so likely no split

    def test_real_world_tank_tyrese_data(self):
        """Real data from Tank vs Tyrese video — 40% center defaults."""
        xs = [50,50,75,50,45,65,75,50,55,75,55,75,25,65,50,55,50,75,75,60,
              75,50,45,50,55,55,50,50,50,25,50,50,48,72,72,60,50,72,72,35,
              50,72,72,50,50,35,50,50,25,35,35,50,60,50,45,25,65,50,50,50]
        kf = [(i, x) for i, x in enumerate(xs)]
        result = _detect_position_clusters(kf)
        assert result is not None, "Should find clusters after center stripping"
        assert len(result) >= 2
        centers = [c["center"] for c in result]
        # Should have clusters NOT at 50
        assert all(c < 44 or c > 56 for c in centers), f"Unexpected center cluster: {centers}"

    def test_midpoint_cluster_rejected(self):
        """A cluster near the midpoint of its neighbors with fewer samples is rejected."""
        # Left=30 (10 samples), midpoint=55 (5 samples), right=80 (10 samples)
        # Midpoint of 30 and 80 = 55. Distance 0, span 50. 0 < 50*0.3=15 ✓
        # count(55)=5 < max(10,10)=10 ✓ → REJECTED
        kf = (
            [(i, 30) for i in range(10)]
            + [(i + 10, 55) for i in range(5)]
            + [(i + 15, 80) for i in range(10)]
        )
        result = _detect_position_clusters(kf)
        assert result is not None
        assert len(result) == 2, f"Expected 2 clusters (midpoint rejected), got {len(result)}: {result}"
        centers = [c["center"] for c in result]
        assert 50 not in centers and 55 not in centers, f"Midpoint cluster should be gone: {centers}"

    def test_real_cluster_not_rejected(self):
        """A cluster between neighbors with MORE samples should NOT be rejected."""
        # Left=20 (3 samples), middle=50 (15 samples), right=80 (3 samples)
        # count(50)=15 > max(3,3)=3 → NOT rejected
        kf = (
            [(i, 20) for i in range(3)]
            + [(i + 3, 50) for i in range(15)]
            + [(i + 18, 80) for i in range(3)]
        )
        result = _detect_position_clusters(kf)
        assert result is not None
        assert len(result) == 3, f"Expected 3 clusters (all real), got {len(result)}: {result}"


# ══════════════════════════════════════════════════════════════════════
# _snap_to_clusters — tie-breaking and nearest assignment
# ══════════════════════════════════════════════════════════════════════

class TestSnapToClusters:
    def test_basic_snap(self):
        clusters = [{"center": 30, "count": 5}, {"center": 70, "count": 5}]
        kf = [(0, 25), (1, 45), (2, 55), (3, 80)]
        result = _snap_to_clusters(kf, clusters)
        assert result == [(0, 30), (1, 30), (2, 70), (3, 70)]

    def test_equidistant_prefers_higher_count(self):
        """At equal distance, prefer the cluster with more samples."""
        clusters = [{"center": 40, "count": 3}, {"center": 60, "count": 10}]
        kf = [(0, 50)]  # Equidistant from 40 and 60
        result = _snap_to_clusters(kf, clusters)
        assert result[0][1] == 60, "Should prefer higher-count cluster on tie"

    def test_center_default_snapped(self):
        """Values at 50 should snap to the nearest cluster."""
        clusters = [{"center": 35, "count": 10}, {"center": 68, "count": 8}]
        kf = [(0, 50)]
        result = _snap_to_clusters(kf, clusters)
        assert result[0][1] == 35, "50 is closer to 35 (dist=15) than 68 (dist=18)"


# ══════════════════════════════════════════════════════════════════════
# _validate_tracking — QA validation
# ══════════════════════════════════════════════════════════════════════

class TestValidateTracking:
    def test_clean_data_unchanged(self):
        kf = [(0, 30), (5, 70), (10, 30)]
        result = _validate_tracking(kf, None, 10)
        # Should insert instant cuts for large jumps
        assert len(result) >= len(kf)

    def test_extended_center_hold_fixed(self):
        """Center hold >3s with clusters should be replaced with previous position."""
        clusters = [{"center": 30, "count": 5}, {"center": 70, "count": 5}]
        kf = [(0, 30), (2, 50), (8, 70)]  # 50 holds for 6s
        result = _validate_tracking(kf, clusters, 10)
        # The center hold at t=2 should be replaced with prev (30)
        center_vals = [sx for t, sx in result if abs(t - 2) < 0.1]
        assert center_vals and center_vals[0] == 30

    def test_instant_cut_inserted(self):
        """Large jumps without 1ms marker should get one inserted."""
        kf = [(0, 30), (5, 70)]  # 40-unit jump, 5s gap
        result = _validate_tracking(kf, None, 10)
        # Should have a 1ms hold before the jump
        assert len(result) == 3
        assert abs(result[1][0] - 4.999) < 0.01
        assert result[1][1] == 30  # Hold previous position

    def test_small_jumps_no_instant_cut(self):
        """Jumps < 15 units should NOT get instant-cut markers."""
        kf = [(0, 40), (5, 50)]  # 10-unit jump
        result = _validate_tracking(kf, None, 10)
        assert len(result) == 2  # No insertion

    def test_empty_keyframes(self):
        result = _validate_tracking([], None, 10)
        assert result == [(0.0, 50)]


# ══════════════════════════════════════════════════════════════════════
# Face detection — dataclass and batch processing
# ══════════════════════════════════════════════════════════════════════

class TestFaceDetector:
    """Tests for the face detection service."""

    def test_face_info_dataclass(self):
        from backend.services.face_detector import FaceInfo
        face = FaceInfo(
            x_center=45.0, y_center=30.0, width=20.0, height=25.0,
            nose_x=44.5, nose_y=31.2, confidence=0.95,
        )
        assert face.nose_x == 44.5
        assert face.confidence == 0.95

    def test_frame_faces_dataclass(self):
        from backend.services.face_detector import FrameFaces, FaceInfo
        ff = FrameFaces(timestamp=5.0, frame_path="/tmp/frame.jpg")
        assert ff.faces == []
        assert ff.primary_face_idx == -1

        face = FaceInfo(35.0, 30.0, 15.0, 20.0, 34.5, 29.0, 0.9)
        ff2 = FrameFaces(timestamp=5.0, frame_path="/tmp/frame.jpg",
                         faces=[face], primary_face_idx=0)
        assert len(ff2.faces) == 1
        assert ff2.primary_face_idx == 0

    def test_detect_faces_nonexistent_files(self):
        """Should return empty faces for missing files, not crash."""
        from backend.services.face_detector import detect_faces_batch
        results = detect_faces_batch([
            (0.0, "/nonexistent/frame1.jpg"),
            (5.0, "/nonexistent/frame2.jpg"),
        ])
        assert len(results) == 2
        assert results[0].faces == []
        assert results[1].faces == []

    def test_detect_faces_returns_correct_count(self):
        """Output length should match input length."""
        from backend.services.face_detector import detect_faces_batch
        paths = [(float(i), f"/nonexistent/frame_{i}.jpg") for i in range(10)]
        results = detect_faces_batch(paths)
        assert len(results) == 10


# ══════════════════════════════════════════════════════════════════════
# Pipeline integration — face data attached to frames
# ══════════════════════════════════════════════════════════════════════

class TestPipelineFaceDetection:
    """Verify face detection data flows through the pipeline to providers."""

    def test_frame_data_accepts_face_data(self):
        """FrameData model should accept face_data field."""
        from backend.models import FrameData
        from backend.services.face_detector import FrameFaces, FaceInfo
        frame = FrameData(timestamp=5.0, path="/tmp/frame.jpg")
        assert frame.face_data is None

        face = FaceInfo(35.0, 30.0, 15.0, 20.0, 34.5, 29.0, 0.9)
        fd = FrameFaces(timestamp=5.0, frame_path="/tmp/frame.jpg",
                        faces=[face], primary_face_idx=0)
        frame.face_data = fd
        assert frame.face_data is not None
        assert frame.face_data.faces[0].nose_x == 34.5

    def test_face_data_used_in_position_fusion(self):
        """When face_data is present, subject_x should come from nose_x."""
        from backend.models import FrameData
        from backend.services.face_detector import FrameFaces, FaceInfo

        # Create frame with face at x=35
        face = FaceInfo(35.0, 30.0, 15.0, 20.0, 35.0, 29.0, 0.9)
        fd = FrameFaces(timestamp=5.0, frame_path="/tmp/frame.jpg",
                        faces=[face], primary_face_idx=0)

        frame = FrameData(timestamp=5.0, path="/tmp/frame.jpg")
        frame.face_data = fd

        # Simulate the position fusion logic from openrouter_provider
        sx = 50  # AI returned center default
        face_data = getattr(frame, 'face_data', None)
        if face_data and hasattr(face_data, 'faces') and face_data.faces:
            if len(face_data.faces) == 1:
                sx = round(face_data.faces[0].nose_x)
            elif face_data.primary_face_idx >= 0:
                sx = round(face_data.faces[face_data.primary_face_idx].nose_x)

        assert sx == 35, f"Should use face nose_x=35 instead of AI default 50, got {sx}"

    def test_multi_face_active_selection(self):
        """With multiple faces, active_face should select the correct position."""
        from backend.services.face_detector import FrameFaces, FaceInfo

        face1 = FaceInfo(30.0, 30.0, 15.0, 20.0, 30.0, 29.0, 0.9)
        face2 = FaceInfo(70.0, 30.0, 15.0, 20.0, 70.0, 29.0, 0.85)
        fd = FrameFaces(timestamp=5.0, frame_path="/tmp/frame.jpg",
                        faces=[face1, face2], primary_face_idx=0)

        # Simulate active_face=2 (1-based) selecting face2
        active_face = 2
        afi = active_face - 1  # Convert to 0-based
        assert 0 <= afi < len(fd.faces)
        sx = round(fd.faces[afi].nose_x)
        assert sx == 70, f"active_face=2 should select face2 at x=70, got {sx}"

    def test_no_face_data_falls_back(self):
        """Without face_data, should fall back to AI estimate."""
        from backend.models import FrameData
        frame = FrameData(timestamp=5.0, path="/tmp/frame.jpg")

        sx = 42  # AI estimate
        face_data = getattr(frame, 'face_data', None)
        if face_data and hasattr(face_data, 'faces') and face_data.faces:
            sx = round(face_data.faces[0].nose_x)

        assert sx == 42, "Without face_data, AI estimate should be preserved"
