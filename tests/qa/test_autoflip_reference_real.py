"""Task 5A — real MediaPipe AutoFlip reference generator.

Mirrors the test contract of ``test_naive_baseline_reference.py`` but
exercises the real-AutoFlip path: docker invocation is mocked, but
the metadata parser, manifest iteration, and shape contract are real.

Run:  pytest tests/qa/test_autoflip_reference_real.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest import mock

import pytest

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


_FIXTURE_PBTXT = (
    _REPO / "tests" / "fixtures" / "autoflip" / "sample_metadata.pbtxt"
)


def _stub_probe():
    return {
        "duration": 3.0, "width": 1920, "height": 1080, "fps": 30.0,
    }


def _make_runner(metadata_text: str = None):
    """Return a docker_runner stub that writes a metadata pbtxt file
    to ``output_dir/<slug>_metadata.pbtxt`` and returns 0.

    When ``metadata_text`` is None the fixture pbtxt is used.
    """
    text = metadata_text if metadata_text is not None else _FIXTURE_PBTXT.read_text()

    def _runner(*, image, source_dir, output_dir, slug, ext, aspect_ratio):
        (Path(output_dir) / f"{slug}_metadata.pbtxt").write_text(text)
        return 0

    return _runner


# ── parser-level coverage ─────────────────────────────────────────────


def test_protobuf_parsing_translates_to_event_shape():
    """Given a known AutoFlip-shape pbtxt, the parser produces events
    with crop_cx/crop_cy/crop_w/crop_h/scene_change in 0-1 range."""
    from backend.scripts.run_autoflip_reference import (
        parse_autoflip_metadata,
    )
    text = _FIXTURE_PBTXT.read_text()
    events = parse_autoflip_metadata(text, fps=30.0, duration=3.0)
    # Three render_frame blocks → at least three events; the parser
    # may pad to fps × duration so >= 3 is the contract.
    assert len(events) >= 3
    for e in events:
        assert {"t", "crop_cx", "crop_cy", "crop_w", "crop_h", "scene_change"} <= set(e)
        assert 0.0 <= e["crop_cx"] <= 1.0
        assert 0.0 <= e["crop_cy"] <= 1.0
        assert 0.0 < e["crop_w"] <= 1.0
        assert 0.0 < e["crop_h"] <= 1.0
        assert isinstance(e["scene_change"], bool)
    # First frame's crop should drift right relative to the third's
    # input x (0.20 → 0.40), so crop_cx should grow.
    assert events[0]["crop_cx"] < events[2]["crop_cx"]


def test_parser_uses_timestamp_us_when_present():
    from backend.scripts.run_autoflip_reference import (
        parse_autoflip_metadata,
    )
    events = parse_autoflip_metadata(
        _FIXTURE_PBTXT.read_text(), fps=30.0, duration=3.0,
    )
    # First three events come straight from timestamp_us=0/33333/66666
    # → t=0.0, ~0.0333, ~0.0666.
    assert events[0]["t"] == pytest.approx(0.0, abs=1e-3)
    assert events[1]["t"] == pytest.approx(0.0333, abs=1e-3)
    assert events[2]["t"] == pytest.approx(0.0666, abs=1e-3)


def test_parser_falls_back_to_uniform_spacing_without_timestamps():
    """If the pbtxt has no timestamp_us, events are spaced 1/fps."""
    from backend.scripts.run_autoflip_reference import (
        parse_autoflip_metadata,
    )
    text = (
        "render_frame { rect { x: 0.1 y: 0.0 width: 0.5 height: 1.0 } }\n"
        "render_frame { rect { x: 0.2 y: 0.0 width: 0.5 height: 1.0 } }\n"
    )
    events = parse_autoflip_metadata(text, fps=10.0, duration=0.2)
    assert events[0]["t"] == pytest.approx(0.0)
    assert events[1]["t"] == pytest.approx(0.1)


def test_parser_returns_empty_on_unparseable_text():
    from backend.scripts.run_autoflip_reference import (
        parse_autoflip_metadata,
    )
    assert parse_autoflip_metadata("garbage", fps=30.0, duration=1.0) == []


# ── per-clip and manifest-level coverage ──────────────────────────────


def test_aspect_ratio_default_is_9_16(tmp_path):
    """The docker runner is invoked with --aspect_ratio=9:16 by default."""
    from backend.scripts import run_autoflip_reference

    captured = {}

    def _runner(**kwargs):
        captured.update(kwargs)
        (tmp_path / f"{kwargs['slug']}_metadata.pbtxt").write_text(
            _FIXTURE_PBTXT.read_text(),
        )
        return 0

    fake_video = tmp_path / "stub.mp4"
    fake_video.write_bytes(b"\x00")
    with mock.patch.object(
        run_autoflip_reference, "_probe_metadata",
        return_value=_stub_probe(),
    ):
        run_autoflip_reference.autoflip_events_for_clip(
            str(fake_video), tmp_path,
            slug="stub", ext="mp4", docker_runner=_runner,
        )
    assert captured["aspect_ratio"] == "9:16"


def test_tool_label_is_autoflip_real(tmp_path):
    """Default tool_label is 'autoflip_real' so the bench can
    distinguish real AutoFlip from naive_baseline."""
    from backend.scripts import run_autoflip_reference

    fake_video = tmp_path / "stub.mp4"
    fake_video.write_bytes(b"\x00")
    with mock.patch.object(
        run_autoflip_reference, "_probe_metadata",
        return_value=_stub_probe(),
    ):
        ref = run_autoflip_reference.autoflip_events_for_clip(
            str(fake_video), tmp_path,
            slug="stub", ext="mp4",
            docker_runner=_make_runner(),
        )
    assert ref["tool"] == "autoflip_real"
    assert ref["metadata"]["tool"] == "autoflip_real"
    assert ref["events"], "expected at least one parsed event"


def test_tool_label_can_be_autoflip_prebuilt(tmp_path):
    """Path A2 prebuilt-binary builds emit 'autoflip_prebuilt'."""
    from backend.scripts import run_autoflip_reference

    fake_video = tmp_path / "stub.mp4"
    fake_video.write_bytes(b"\x00")
    with mock.patch.object(
        run_autoflip_reference, "_probe_metadata",
        return_value=_stub_probe(),
    ):
        ref = run_autoflip_reference.autoflip_events_for_clip(
            str(fake_video), tmp_path,
            slug="stub", ext="mp4",
            tool_label="autoflip_prebuilt",
            docker_runner=_make_runner(),
        )
    assert ref["tool"] == "autoflip_prebuilt"


def test_run_for_manifest_writes_output_per_clip(tmp_path):
    from backend.scripts import run_autoflip_reference
    from backend.scripts.run_autoflip_reference import run_for_manifest

    real_dir = tmp_path / "real"
    real_dir.mkdir()
    (real_dir / "clip_a.mp4").write_bytes(b"\x00")
    (real_dir / "clip_b.mp4").write_bytes(b"\x00")

    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "clips": [
            {"slug": "clip_a", "ext": "mp4"},
            {"slug": "clip_b", "ext": "mp4"},
        ],
    }))
    out_dir = tmp_path / "out"

    with mock.patch.object(
        run_autoflip_reference, "_probe_metadata",
        return_value=_stub_probe(),
    ):
        summary = run_for_manifest(
            manifest, out_dir,
            real_content_dir=real_dir,
            docker_runner=_make_runner(),
        )

    assert sorted(summary["generated"]) == ["clip_a", "clip_b"]
    assert summary["skipped"] == []
    for slug in ("clip_a", "clip_b"):
        payload = json.loads((out_dir / f"{slug}.json").read_text())
        assert payload["tool"] == "autoflip_real"
        assert payload["events"], slug


def test_skips_when_docker_image_missing(tmp_path):
    """When the docker runner returns non-zero, the slug is skipped
    and the script does not crash."""
    from backend.scripts import run_autoflip_reference
    from backend.scripts.run_autoflip_reference import run_for_manifest

    real_dir = tmp_path / "real"
    real_dir.mkdir()
    (real_dir / "clip_a.mp4").write_bytes(b"\x00")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"clips": [{"slug": "clip_a", "ext": "mp4"}]}))

    def _failing_runner(**kwargs):
        return 125  # docker daemon unreachable

    with mock.patch.object(
        run_autoflip_reference, "_probe_metadata",
        return_value=_stub_probe(),
    ):
        summary = run_for_manifest(
            manifest, tmp_path / "out",
            real_content_dir=real_dir,
            docker_runner=_failing_runner,
        )
    assert summary["generated"] == []
    assert len(summary["skipped"]) == 1
    assert summary["skipped"][0][0] == "clip_a"
    assert "exited with code 125" in summary["skipped"][0][1]


def test_skips_when_source_video_missing(tmp_path):
    """Same back-compat path as run_naive_baseline."""
    from backend.scripts.run_autoflip_reference import run_for_manifest

    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "clips": [{"slug": "missing_clip", "ext": "mp4"}],
    }))
    summary = run_for_manifest(
        manifest, tmp_path / "out",
        real_content_dir=tmp_path / "videos_dont_exist",
    )
    assert summary["generated"] == []
    assert len(summary["skipped"]) == 1
    assert "video missing" in summary["skipped"][0][1]


# ── bench rollup label propagation ────────────────────────────────────


def test_loader_propagates_tool_label(tmp_path):
    """load_autoflip_timeline returns the tool label when asked."""
    from backend.scripts.compare_autoflip_vs_clipai import (
        load_autoflip_timeline,
    )
    payload = {
        "tool": "autoflip_real",
        "metadata": {"duration": 1.0, "width": 1920, "height": 1080, "fps": 10.0},
        "events": [
            {"t": 0.0, "crop_cx": 0.5, "crop_cy": 0.5,
             "crop_w": 0.5625, "crop_h": 1.0, "scene_change": False},
        ],
    }
    (tmp_path / "panel_clip.json").write_text(json.dumps(payload))
    events, tool = load_autoflip_timeline(
        {"slug": "panel_clip"}, tmp_path, return_tool=True,
    )
    assert tool == "autoflip_real"
    assert events is not None and len(events) == 1


def test_loader_legacy_signature_still_returns_events_only(tmp_path):
    """Without return_tool=True, the legacy single-value return shape
    is preserved so older callers keep working."""
    from backend.scripts.compare_autoflip_vs_clipai import (
        load_autoflip_timeline,
    )
    (tmp_path / "panel_clip.json").write_text(json.dumps({
        "tool": "naive_baseline",
        "events": [{"t": 0.0, "crop_cx": 0.5, "crop_cy": 0.5,
                    "crop_w": 0.5, "crop_h": 1.0, "scene_change": False}],
    }))
    events = load_autoflip_timeline(
        {"slug": "panel_clip"}, tmp_path,
    )
    assert isinstance(events, list)
    assert len(events) == 1


def test_rollup_shows_autoflip_real_when_present():
    """The markdown rollup labels the AutoFlip column 'autoflip
    (real)' when the tool was autoflip_real."""
    from backend.scripts.compare_autoflip_vs_clipai import (
        ClipResult, render_markdown,
    )
    r = ClipResult(
        slug="panel_clip",
        content_type="multi_speaker_panel",
        target_clipcontenttype="multi_speaker_panel",
        subtype=None,
        autoflip={"median_hold_sec": 2.5, "max_acceleration": 1.0,
                  "max_jerk": 0.5},
        autoflip_tool="autoflip_real",
    )
    md = render_markdown([r], autoflip_outputs_dir=Path("."), clipai_invocation="x")
    assert "autoflip (real)" in md
    # No "Warning — every AutoFlip column ... naive baseline" line
    assert "every AutoFlip column on this run was" not in md


def test_rollup_warns_on_naive_baseline_strawman():
    """When only naive_baseline is present, the rollup includes a
    visible warning so SOTA claims aren't accidentally made."""
    from backend.scripts.compare_autoflip_vs_clipai import (
        ClipResult, render_markdown,
    )
    r = ClipResult(
        slug="panel_clip",
        content_type="multi_speaker_panel",
        target_clipcontenttype="multi_speaker_panel",
        subtype=None,
        autoflip={"median_hold_sec": 2.5, "max_acceleration": 1.0,
                  "max_jerk": 0.5},
        autoflip_tool="naive_baseline",
    )
    md = render_markdown([r], autoflip_outputs_dir=Path("."), clipai_invocation="x")
    assert "autoflip (naive)" in md
    assert "every AutoFlip column on this run was" in md


def test_main_loop_records_autoflip_tool_on_clipresult(tmp_path):
    """End-to-end: when load_autoflip_timeline returns a tool label,
    ClipResult.autoflip_tool is populated so render_markdown can use
    it. We verify by writing a JSON ref to disk and calling the
    loader directly (the main loop's flow is one line of glue)."""
    from backend.scripts.compare_autoflip_vs_clipai import (
        ClipResult, load_autoflip_timeline,
    )
    payload = {
        "tool": "autoflip_real",
        "events": [{"t": 0.0, "crop_cx": 0.5, "crop_cy": 0.5,
                    "crop_w": 0.5625, "crop_h": 1.0,
                    "scene_change": False}],
    }
    (tmp_path / "x.json").write_text(json.dumps(payload))
    events, tool = load_autoflip_timeline(
        {"slug": "x"}, tmp_path, return_tool=True,
    )
    r = ClipResult(slug="x", content_type="", target_clipcontenttype="",
                   subtype=None, autoflip_tool=tool)
    assert r.autoflip_tool == "autoflip_real"
    assert events is not None
