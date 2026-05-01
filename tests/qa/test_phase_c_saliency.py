"""Phase C QA — TASED-Net AV saliency, OCR text exclusion, LP saliency cost.

Validates Phase C contributions WITHOUT requiring torch / paddle / GPU.
The math helpers (peak extraction, audio energy, OCR bbox conversion,
saliency nudge, soft-promotion) run on real numpy. The TASED-Net /
PaddleOCR adapters are mocked.

Run:  pytest tests/qa/test_phase_c_saliency.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


# ──────────────────────────────────────────────────────────────────
# 1. AV saliency — peak extraction + audio energy + normalization
# ──────────────────────────────────────────────────────────────────


class TestPeakExtraction:
    def test_single_peak_recovered(self):
        from backend.services.av_saliency import extract_peaks
        h = np.zeros((144, 256), dtype=np.float32)
        h[70, 130] = 1.0
        peaks = extract_peaks(h, top_k=3)
        assert len(peaks) == 1
        x_pct, y_pct, score = peaks[0]
        assert 50.0 <= x_pct <= 52.0    # px=130 of 256 → ~50.9 %
        assert 48.0 <= y_pct <= 50.0    # py=70 of 144 → ~49.0 %
        assert score == pytest.approx(1.0)

    def test_top_k_with_nms(self):
        from backend.services.av_saliency import extract_peaks
        h = np.zeros((144, 256), dtype=np.float32)
        # Two peaks far apart + one very close to the first (suppressed)
        h[70, 50] = 0.9
        h[70, 52] = 0.85          # within NMS radius — should be suppressed
        h[100, 200] = 0.7
        peaks = extract_peaks(h, top_k=3, neighbourhood=8)
        assert len(peaks) == 2
        # Sorted by descending score in their order of discovery.
        assert peaks[0][2] == pytest.approx(0.9)
        assert peaks[1][2] == pytest.approx(0.7)

    def test_empty_heatmap(self):
        from backend.services.av_saliency import extract_peaks
        assert extract_peaks(np.zeros((10, 10))) == []
        assert extract_peaks(np.zeros((0, 0))) == []


class TestNormalization:
    def test_normalize_to_unit_range(self):
        from backend.services.av_saliency import normalize_heatmap
        raw = np.array([[5.0, 7.0], [3.0, 9.0]], dtype=np.float32)
        n = normalize_heatmap(raw)
        assert n.min() == pytest.approx(0.0)
        assert n.max() == pytest.approx(1.0)

    def test_constant_heatmap_returns_zeros(self):
        from backend.services.av_saliency import normalize_heatmap
        n = normalize_heatmap(np.full((4, 4), 5.0, dtype=np.float32))
        assert (n == 0).all()


class TestAudioEnergy:
    def test_silent_audio_zero_energy(self):
        from backend.services.av_saliency import audio_energy_per_second
        e = audio_energy_per_second(np.zeros(48000, dtype=np.float32), 16000)
        assert (e == 0).all()

    def test_loud_then_quiet_audio_emits_peak(self):
        from backend.services.av_saliency import audio_energy_per_second
        sr = 16000
        sig = np.zeros(sr * 3, dtype=np.float32)
        sig[:sr] = 0.5    # 1 second of loud audio
        e = audio_energy_per_second(sig, sr)
        assert len(e) == 3
        # First second should be the loudest (after normalization).
        assert e[0] > e[1]
        assert e[0] > e[2]
        assert e[0] == pytest.approx(1.0)


# ──────────────────────────────────────────────────────────────────
# 2. AvSaliency orchestration
# ──────────────────────────────────────────────────────────────────


class _FakeTasedAdapter:
    def __init__(self, *, device="cuda"):
        self.device = device

    def load(self):
        pass

    def predict_clip(self, frames, audio_energy):
        T = max(1, frames.shape[0] // 30)
        H, W = 144, 256
        rng = np.random.default_rng(0)
        out = rng.uniform(0, 1, (T, H, W)).astype(np.float32)
        # Plant a strong peak so the post-norm peaks list is non-empty.
        out[:, 70, 130] = 5.0
        return out


class TestAvSaliencyOrchestration:
    @mock.patch("backend.services.av_saliency._is_gpu_visible", return_value=False)
    def test_construction_downgrades_to_cpu_without_gpu(self, _gpu):
        """Documented contract (av_saliency.py:391-393): CUDA-requested
        with no GPU visible silently downgrades to CPU rather than
        raising. The fallback chain ensures a working adapter is loaded
        on CPU. Production-critical: on a 4 GB GTX 1650 the GPU is
        shared with Whisper / Ollama / Parakeet, and intermittent
        unavailability must not abort the entire reframing pipeline.
        """
        from backend.services.av_saliency import AvSaliency
        sal = AvSaliency()
        assert sal.device == "cpu"
        # Fallback chain produced *some* adapter — TASED-Net (if its
        # weights are present) or the always-available spectral
        # residual.
        assert sal._adapter is not None
        assert getattr(sal._adapter, "backend_name", None) is not None

    @mock.patch("backend.services.av_saliency._is_gpu_visible", return_value=True)
    def test_predict_returns_normalized_heatmaps_and_peaks(self, _gpu):
        from backend.services.av_saliency import AvSaliency
        T_frames = 60
        frames = np.zeros((T_frames, 720, 1280, 3), dtype=np.uint8)
        sal = AvSaliency(_adapter_cls=_FakeTasedAdapter)
        result = sal.predict(frames, np.array([0.0, 1.0]))
        assert result.heatmaps.shape[0] == 2
        assert result.heatmaps.min() >= 0.0
        assert result.heatmaps.max() <= 1.0
        # Each second has 1+ peak from the planted maximum.
        assert all(len(p) >= 1 for p in result.peaks)


# ──────────────────────────────────────────────────────────────────
# 3. OCR module
# ──────────────────────────────────────────────────────────────────


class TestOcrHelpers:
    def test_bbox_from_quadrilateral(self):
        from backend.services.ocr_regions import _bbox_from_quadrilateral
        quad = [(10, 20), (50, 22), (52, 60), (8, 58)]
        x, y, w, h = _bbox_from_quadrilateral(quad)
        assert x == 8
        assert y == 20
        assert w == 44
        assert h == 40

    def test_to_pct_converts_correctly(self):
        from backend.services.ocr_regions import _to_pct
        bbox = (320, 540, 640, 540)   # right half of 1280x1080
        x_p, y_p, w_p, h_p = _to_pct(bbox, 1080, 1280)
        assert x_p == pytest.approx(25.0)
        assert y_p == pytest.approx(50.0)
        assert w_p == pytest.approx(50.0)
        assert h_p == pytest.approx(50.0)

    def test_filter_drops_low_confidence(self):
        from backend.services.ocr_regions import filter_regions, TextRegion
        regs = [
            TextRegion(0, 0, 30, 10, "high", 0.95),
            TextRegion(0, 0, 30, 10, "low", 0.20),
        ]
        kept = filter_regions(regs, min_confidence=0.5)
        assert len(kept) == 1
        assert kept[0].text == "high"

    def test_filter_drops_tiny_regions(self):
        from backend.services.ocr_regions import filter_regions, TextRegion
        regs = [
            TextRegion(0, 0, 0.1, 0.1, "tiny", 0.95),    # 0.01 % area
            TextRegion(0, 0, 30, 10, "big", 0.95),       # 300 % * 100 → big
        ]
        kept = filter_regions(regs, min_area_pct=0.05)
        assert len(kept) == 1
        assert kept[0].text == "big"


class _FakePaddleAdapter:
    def __init__(self, *, lang="en", min_confidence=0.6):
        self.lang = lang
        self.min_confidence = min_confidence

    def load(self):
        pass

    def detect(self, frame_bgr):
        return [[
            [
                [(100, 50), (300, 50), (300, 100), (100, 100)],
                ("FINAL SCORE: 89-86", 0.92),
            ],
            [
                [(50, 600), (200, 600), (200, 640), (50, 640)],
                ("@johndoe", 0.88),
            ],
        ]]


class TestOcrDetection:
    def test_detect_runs_on_synthetic_frame(self):
        from backend.services.ocr_regions import OcrRegionDetector
        det = OcrRegionDetector(_adapter_cls=_FakePaddleAdapter)
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        regions = det.detect(frame)
        assert len(regions) == 2
        # Coordinates should be in % of source frame.
        for r in regions:
            assert 0.0 <= r.x_pct <= 100.0
            assert 0.0 <= r.y_pct <= 100.0
            assert 0.0 < r.w_pct <= 100.0
            assert 0.0 < r.h_pct <= 100.0

    def test_empty_frame_returns_empty(self):
        from backend.services.ocr_regions import OcrRegionDetector
        det = OcrRegionDetector(_adapter_cls=_FakePaddleAdapter)
        # Empty array
        assert det.detect(np.zeros((0, 0, 3), dtype=np.uint8)) == []


# ──────────────────────────────────────────────────────────────────
# 4. Saliency LP nudge
# ──────────────────────────────────────────────────────────────────


class TestSaliencyNudge:
    def test_nudge_pulls_target_toward_far_peak(self):
        from backend.services.saliency_lp_term import apply_saliency_nudge
        # Per-frame target sitting at 30 %, peak at 70 %.
        tx = [30.0] * 5
        ts = [0.0, 0.5, 1.0, 1.5, 2.0]
        peaks = [
            [(70.0, 50.0, 1.0)],  # second 0
            [(70.0, 50.0, 1.0)],  # second 1
            [(70.0, 50.0, 1.0)],  # second 2
        ]
        nudged = apply_saliency_nudge(
            tx, ts, peaks, lambda_saliency=0.5, peak_radius_pct=10.0,
        )
        # Each nudged target should be PAST 30 toward 70 (but not at 70).
        for v in nudged:
            assert 30.0 < v < 70.0

    def test_no_nudge_when_target_already_on_peak(self):
        from backend.services.saliency_lp_term import apply_saliency_nudge
        tx = [70.0, 70.0, 70.0]
        ts = [0.0, 1.0, 2.0]
        peaks = [
            [(70.0, 50.0, 1.0)],
            [(70.0, 50.0, 1.0)],
            [(70.0, 50.0, 1.0)],
        ]
        nudged = apply_saliency_nudge(
            tx, ts, peaks, lambda_saliency=0.5, peak_radius_pct=10.0,
        )
        # Within peak radius — no nudge applied.
        for v in nudged:
            assert v == pytest.approx(70.0)

    def test_nudge_respects_bounds(self):
        from backend.services.saliency_lp_term import apply_saliency_nudge
        tx = [50.0]
        ts = [0.0]
        peaks = [[(95.0, 50.0, 1.0)]]
        nudged = apply_saliency_nudge(
            tx, ts, peaks, lambda_saliency=1.0, hi_bounds=[60.0],
            peak_radius_pct=5.0,
        )
        # Nudge would push to ~95 but high bound clamps to 60.
        assert nudged[0] <= 60.0

    def test_no_peaks_returns_input(self):
        from backend.services.saliency_lp_term import apply_saliency_nudge
        tx = [10.0, 20.0, 30.0]
        out = apply_saliency_nudge(tx, [0, 1, 2], [])
        assert out == tx


class TestSoftPromote:
    def test_promotes_peaks_to_preferred_regions(self):
        from backend.services.saliency_lp_term import soft_promote_peaks_to_required
        peaks = [
            [(50.0, 25.0, 0.9), (80.0, 60.0, 0.7)],
            [(40.0, 30.0, 0.85)],
        ]
        regions = soft_promote_peaks_to_required(peaks)
        assert len(regions) == 3
        for r in regions:
            tier = getattr(r, "tier", None) or r["tier"]
            assert tier == "preferred"


# ──────────────────────────────────────────────────────────────────
# 5. Parity metrics: saliency_in_crop_fraction + text_region_clipping_rate
# ──────────────────────────────────────────────────────────────────


class TestSaliencyInCropMetric:
    def test_perfect_score_when_crop_contains_peaks(self):
        from backend.services.autoflip_parity_metrics import saliency_in_crop_fraction
        # crop centered at 50, width 40 → covers [30, 70]
        crop_centers = [50.0, 50.0, 50.0]
        peaks = [
            [(45.0, 50.0, 1.0)],
            [(55.0, 50.0, 1.0)],
            [(60.0, 50.0, 1.0)],
        ]
        f = saliency_in_crop_fraction(crop_centers, 40.0, peaks)
        assert f == pytest.approx(1.0)

    def test_zero_when_crop_misses_all_peaks(self):
        from backend.services.autoflip_parity_metrics import saliency_in_crop_fraction
        crop_centers = [50.0, 50.0]
        peaks = [
            [(10.0, 50.0, 1.0)],
            [(95.0, 50.0, 1.0)],
        ]
        f = saliency_in_crop_fraction(crop_centers, 20.0, peaks)
        assert f == pytest.approx(0.0)


class TestTextClipRateMetric:
    def test_no_clip_when_text_inside_crop(self):
        from backend.services.autoflip_parity_metrics import text_region_clipping_rate
        crops = [50.0]
        regions = [[(40.0, 50.0, 20.0, 5.0)]]   # text from 40 to 60
        r = text_region_clipping_rate(crops, 40.0, regions)   # crop covers [30, 70]
        assert r == pytest.approx(0.0)

    def test_clip_when_text_extends_past_crop(self):
        from backend.services.autoflip_parity_metrics import text_region_clipping_rate
        crops = [50.0]
        regions = [[(20.0, 50.0, 60.0, 5.0)]]   # text from 20 to 80
        r = text_region_clipping_rate(crops, 40.0, regions)   # crop covers [30, 70]
        assert r == pytest.approx(1.0)


# ──────────────────────────────────────────────────────────────────
# 6. Config flag
# ──────────────────────────────────────────────────────────────────


class TestConfigFlag:
    def test_saliency_enabled_default_on_after_phase_e(self):
        """Phase C shipped CLIPAI_SALIENCY_ENABLED default-OFF.
        Phase E flipped it ON; this test now checks the shipped default.
        """
        try:
            from backend.config import Settings
        except ModuleNotFoundError as exc:
            pytest.skip(f"backend.config deps missing: {exc}")
        s = Settings()
        assert s.CLIPAI_SALIENCY_ENABLED is True


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
