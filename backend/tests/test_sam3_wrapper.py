"""Blueprint v2 Phase 4 — SAM 3 wrapper + adapters + safety contracts."""

import asyncio
from typing import Optional

import pytest

from backend.services import sam3_wrapper as sam3
from backend.services.sam3_genre_prompts import (
    PROMPTS_BY_GENRE,
    prompts_for,
)
from backend.services.sam3_wrapper import (
    SAM3Detection,
    SAM3Result,
    _instance_to_slot,
    _parse_detections_payload,
    get_telemetry_snapshot,
    sam3_to_dense_faces,
    sam3_to_required_regions,
    segment_and_track,
)


# ── Genre prompt table ────────────────────────────────────────────


def test_prompts_for_known_genre():
    assert prompts_for("talking_head") == ["visible face", "speaking person"]
    assert "basketball" in prompts_for("sports_basketball")


def test_prompts_for_unknown_falls_back_to_generic():
    assert prompts_for("some-future-genre") == PROMPTS_BY_GENRE["generic"]


def test_prompts_for_enum_input_is_unwrapped():
    class _CT:
        def __init__(self, v):
            self.value = v

    assert prompts_for(_CT("animation")) == PROMPTS_BY_GENRE["animation"]


def test_prompts_for_returns_a_copy():
    """Mutations by the caller must not leak into the table."""
    lst = prompts_for("talking_head")
    lst.append("mutated")
    assert "mutated" not in PROMPTS_BY_GENRE["talking_head"]


def test_prompts_for_landscape_is_empty():
    """Ken Burns handles landscape — nothing to track."""
    assert prompts_for("landscape") == []


# ── _instance_to_slot ─────────────────────────────────────────────


def test_instance_to_slot_parses_trailing_int():
    assert _instance_to_slot("face:0") == 0
    assert _instance_to_slot("face:27") == 27
    assert _instance_to_slot("ball_12") == 12


def test_instance_to_slot_hashes_non_numeric():
    """Slot must be >= 0 and stable across calls."""
    a = _instance_to_slot("nova")
    b = _instance_to_slot("nova")
    c = _instance_to_slot("lucas")
    assert a >= 0 and b >= 0
    assert a == b
    assert a != c  # different instances land in different slots


def test_instance_to_slot_empty_is_minus_one():
    assert _instance_to_slot("") == -1


# ── _parse_detections_payload ─────────────────────────────────────


def _det_dict(i: int, t: float, inst: str, prompt: str, bbox=(0, 0, 10, 10)):
    return {
        "frame": i,
        "t": t,
        "id": inst,
        "prompt": prompt,
        "bbox": list(bbox),
        "score": 0.9,
    }


def test_parse_dict_payload():
    payload = {
        "detections": [
            _det_dict(0, 0.0, "face:0", "visible face"),
            _det_dict(1, 0.1, "face:0", "visible face"),
        ],
        "frame_count": 2,
        "latency_sec": 3.4,
    }
    result = _parse_detections_payload(payload, "local_service")
    assert result is not None
    assert result.backend == "local_service"
    assert len(result.detections) == 2
    assert result.frame_count == 2
    assert result.latency_sec == pytest.approx(3.4)
    assert result.instances_by_prompt["visible face"]["face:0"] == 2


def test_parse_bare_list_payload():
    payload = [
        _det_dict(0, 0.0, "face:0", "visible face"),
    ]
    result = _parse_detections_payload(payload, "api_replicate")
    assert result is not None
    assert result.backend == "api_replicate"
    # frame_count is inferred from the distinct (frame_idx, timestamp) set.
    assert result.frame_count == 1


def test_parse_handles_none_and_garbage():
    assert _parse_detections_payload(None, "x") is None
    assert _parse_detections_payload(42, "x") is None
    assert _parse_detections_payload({"detections": "garbage"}, "x") is None


def test_parse_drops_invalid_detection_entries():
    payload = [
        _det_dict(0, 0.0, "face:0", "visible face"),
        {"frame": 1, "prompt": "face"},  # missing bbox → dropped
        "garbage",                          # not a dict → dropped
    ]
    result = _parse_detections_payload(payload, "x")
    assert len(result.detections) == 1


# ── Master flag / dispatch ────────────────────────────────────────


def test_segment_and_track_disabled_by_master_flag(monkeypatch):
    """CLIPAI_SAM3_ENABLED unset → always None regardless of backend."""
    monkeypatch.delenv("CLIPAI_SAM3_ENABLED", raising=False)
    monkeypatch.setenv("SAM3_BACKEND", "local_service")
    out = asyncio.run(segment_and_track("v.mp4", ["face"]))
    assert out is None


def test_segment_and_track_disabled_backend_returns_none(monkeypatch):
    monkeypatch.setenv("CLIPAI_SAM3_ENABLED", "1")
    monkeypatch.setenv("SAM3_BACKEND", "disabled")
    out = asyncio.run(segment_and_track("v.mp4", ["face"]))
    assert out is None


def test_segment_and_track_unknown_backend_returns_none(monkeypatch):
    monkeypatch.setenv("CLIPAI_SAM3_ENABLED", "1")
    monkeypatch.setenv("SAM3_BACKEND", "magic")
    out = asyncio.run(segment_and_track("v.mp4", ["face"]))
    assert out is None


def test_segment_and_track_empty_prompts_returns_none(monkeypatch):
    monkeypatch.setenv("CLIPAI_SAM3_ENABLED", "1")
    monkeypatch.setenv("SAM3_BACKEND", "local_service")
    out = asyncio.run(segment_and_track("v.mp4", []))
    assert out is None


def test_segment_and_track_uses_stubbed_backend(monkeypatch):
    """Mocked dispatcher returns canned result + telemetry is recorded."""
    monkeypatch.setenv("CLIPAI_SAM3_ENABLED", "1")
    monkeypatch.setenv("SAM3_BACKEND", "local_service")

    async def _stub(**kwargs):
        return _parse_detections_payload(
            {
                "detections": [
                    _det_dict(0, 0.0, "face:0", "visible face"),
                    _det_dict(1, 0.2, "face:0", "visible face"),
                ],
                "frame_count": 2,
                "latency_sec": 1.2,
            },
            "local_service",
        )

    monkeypatch.setattr(sam3, "_call_local_service", _stub)

    before = len(get_telemetry_snapshot(limit=256))
    out = asyncio.run(segment_and_track("v.mp4", ["visible face"]))
    assert out is not None
    assert out.backend == "local_service"
    assert len(out.detections) == 2
    # Telemetry grew by exactly 1.
    after = get_telemetry_snapshot(limit=256)
    assert len(after) == before + 1
    assert after[-1]["ok"] is True


def test_segment_and_track_backend_failure_returns_none_and_logs_telemetry(monkeypatch):
    monkeypatch.setenv("CLIPAI_SAM3_ENABLED", "1")
    monkeypatch.setenv("SAM3_BACKEND", "local_service")

    async def _stub(**kwargs):
        raise RuntimeError("service unreachable")

    monkeypatch.setattr(sam3, "_call_local_service", _stub)

    before = len(get_telemetry_snapshot(limit=256))
    out = asyncio.run(segment_and_track("v.mp4", ["visible face"]))
    assert out is None
    after = get_telemetry_snapshot(limit=256)
    assert len(after) == before + 1
    assert after[-1]["ok"] is False
    assert "service unreachable" in after[-1]["error"]


# ── sam3_to_dense_faces ───────────────────────────────────────────


def _mk_result(detections: list, backend: str = "local_service") -> SAM3Result:
    return SAM3Result(
        detections=detections,
        backend=backend,
        frame_count=len({(d.frame_idx, d.timestamp) for d in detections}),
    )


def test_dense_faces_empty_on_none_or_empty():
    assert sam3_to_dense_faces(None, source_width=1920, source_height=1080) == []
    empty = _mk_result([])
    assert sam3_to_dense_faces(empty, source_width=1920, source_height=1080) == []


def test_dense_faces_converts_pixels_to_percent():
    dets = [
        SAM3Detection(
            frame_idx=0, timestamp=0.0,
            instance_id="face:0", prompt="visible face",
            bbox=(480, 270, 960, 810), score=0.9,
        ),
    ]
    result = _mk_result(dets)
    ffs = sam3_to_dense_faces(result, source_width=1920, source_height=1080)
    assert len(ffs) == 1
    ff = ffs[0]
    assert ff.timestamp == pytest.approx(0.0)
    assert ff.source == "sam3"
    assert len(ff.faces) == 1
    face = ff.faces[0]
    # Center of (480, 270, 960, 810) → (720, 540) px → (37.5%, 50%).
    assert face.nose_x == pytest.approx(37.5)
    assert face.nose_y == pytest.approx(50.0)
    # Width 480px / 1920 * 100 = 25%.
    assert face.width == pytest.approx(25.0)
    assert face.height == pytest.approx(50.0)
    assert face.identity_id == 0  # face:0 → slot 0


def test_dense_faces_skips_non_face_prompts():
    dets = [
        SAM3Detection(0, 0.0, "face:0", "visible face", (0, 0, 10, 10), 0.9),
        SAM3Detection(0, 0.0, "ball:0", "the ball", (0, 0, 10, 10), 0.9),
    ]
    ffs = sam3_to_dense_faces(_mk_result(dets), source_width=100, source_height=100)
    assert len(ffs) == 1 and len(ffs[0].faces) == 1


def test_dense_faces_anime_face_flagged_nonhuman():
    dets = [
        SAM3Detection(0, 0.0, "char:0", "anime face", (0, 0, 50, 50), 0.8),
    ]
    ffs = sam3_to_dense_faces(_mk_result(dets), source_width=100, source_height=100)
    assert ffs[0].faces[0].is_human is False


def test_dense_faces_instance_ids_stable_across_frames():
    """Identity persistence: the same instance_id across frames maps
    to the same ClipAI slot every time."""
    dets = [
        SAM3Detection(i, float(i) / 10, "face:7", "visible face",
                      (0, 0, 100, 100), 0.9)
        for i in range(5)
    ]
    ffs = sam3_to_dense_faces(_mk_result(dets), source_width=1920, source_height=1080)
    slots = {ff.faces[0].identity_id for ff in ffs if ff.faces}
    assert slots == {7}


# ── sam3_to_required_regions ──────────────────────────────────────


def test_required_regions_promotes_hud_to_required_tier():
    dets = [
        SAM3Detection(0, 0.0, "hud:0", "HUD elements",
                      (0, 0, 400, 100), 0.95),
    ]
    regs = sam3_to_required_regions(
        _mk_result(dets), source_width=1920, source_height=1080,
    )
    assert len(regs) == 1
    assert regs[0].tier == "required"
    assert regs[0].source == "sam3"
    # Required-gain should be large (matrix default 1000).
    assert regs[0].weight >= 100.0


def test_required_regions_promotes_ball_to_preferred_tier():
    from backend.services.reframe_config import get_default_config

    dets = [
        SAM3Detection(0, 0.0, "ball:0", "the ball",
                      (900, 500, 1000, 600), 0.88),
    ]
    cfg = get_default_config().for_content("sports_basketball")
    regs = sam3_to_required_regions(
        _mk_result(dets),
        source_width=1920, source_height=1080,
        config=cfg,
    )
    assert len(regs) == 1
    assert regs[0].tier == "preferred"
    # Ball weight scales with the genre's object importance (1.0 for
    # basketball per Phase 1 matrix).
    assert regs[0].weight == pytest.approx(cfg.importance.object)


def test_required_regions_skips_unlisted_prompts():
    dets = [
        SAM3Detection(0, 0.0, "x:0", "speaking person",
                      (0, 0, 100, 100), 0.9),
        SAM3Detection(0, 0.0, "x:1", "visible face",
                      (0, 0, 100, 100), 0.9),
    ]
    regs = sam3_to_required_regions(
        _mk_result(dets), source_width=1920, source_height=1080,
    )
    # Face / speaking-person are handled by dense_faces, not
    # required_regions; this adapter must skip them.
    assert regs == []


def test_required_regions_none_result_yields_empty_list():
    assert sam3_to_required_regions(None, source_width=100, source_height=100) == []
