"""Task 7 — perceptual A/B harness 3-up renderer.

Validates ``backend.scripts.render_ab_preview.render_ab_preview`` and
its building blocks. The full ffmpeg path is mocked so the suite stays
fast and runs without ffmpeg installed.

Run:  pytest tests/qa/test_ab_preview.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import pytest

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def test_render_ab_preview_missing_source(tmp_path):
    from backend.scripts.render_ab_preview import render_ab_preview
    out_path = str(tmp_path / "ab.mp4")
    summary = render_ab_preview(
        str(tmp_path / "no_such.mp4"),
        output_path=out_path,
    )
    assert summary["ok"] is False
    assert "missing source" in summary["logs"][0]


def test_render_ab_preview_three_variants_then_stitch(tmp_path):
    """All three variant renders + the stitch must run, in order, and
    each variant must use a distinct crop_cx."""
    from backend.scripts import render_ab_preview as mod

    src = tmp_path / "source.mp4"
    src.write_bytes(b"\x00")
    out_path = str(tmp_path / "ab.mp4")

    crop_cxs: list[float] = []
    labels: list[str] = []

    def fake_render(source_path, *, output_path, label, crop_cx, **kw):
        crop_cxs.append(crop_cx)
        labels.append(label)
        Path(output_path).write_bytes(b"\x00")
        return True, ""

    def fake_stitch(variant_paths, *, audio_source, output_path):
        Path(output_path).write_bytes(b"\x00")
        return True, ""

    with mock.patch.object(mod, "_render_variant", side_effect=fake_render), \
         mock.patch.object(mod, "_stitch_three_up", side_effect=fake_stitch):
        summary = mod.render_ab_preview(
            str(src), output_path=out_path,
        )

    assert summary["ok"] is True
    assert len(crop_cxs) == 3
    # Three distinct crops drive three distinct variants.
    assert len(set(crop_cxs)) == 3
    # Default labels in the right order.
    assert labels[0] == "ClipAI 2026"
    assert "ClipAI" in labels[1]
    assert "baseline" in labels[2].lower()


def test_render_ab_preview_aborts_on_variant_failure(tmp_path):
    """A failed variant render → overall failure, stitch not attempted."""
    from backend.scripts import render_ab_preview as mod

    src = tmp_path / "source.mp4"
    src.write_bytes(b"\x00")
    out_path = str(tmp_path / "ab.mp4")

    def fake_render(source_path, *, output_path, label, **kw):
        return False, "synthetic ffmpeg failure"

    stitch_called = mock.Mock()
    with mock.patch.object(mod, "_render_variant", side_effect=fake_render), \
         mock.patch.object(mod, "_stitch_three_up", side_effect=stitch_called):
        summary = mod.render_ab_preview(
            str(src), output_path=out_path,
        )
    assert summary["ok"] is False
    stitch_called.assert_not_called()


def test_render_ab_preview_propagates_stitch_failure(tmp_path):
    from backend.scripts import render_ab_preview as mod

    src = tmp_path / "source.mp4"
    src.write_bytes(b"\x00")
    out_path = str(tmp_path / "ab.mp4")

    with mock.patch.object(
        mod, "_render_variant",
        side_effect=lambda *a, **k: (True, ""),
    ), mock.patch.object(
        mod, "_stitch_three_up",
        side_effect=lambda *a, **k: (False, "stitch ffmpeg crash"),
    ):
        summary = mod.render_ab_preview(
            str(src), output_path=out_path,
        )
    assert summary["ok"] is False
    assert any("stitch" in line for line in summary["logs"])


def test_label_overrides_propagate(tmp_path):
    from backend.scripts import render_ab_preview as mod

    src = tmp_path / "source.mp4"
    src.write_bytes(b"\x00")
    out_path = str(tmp_path / "ab.mp4")

    seen_labels: list[str] = []

    def fake_render(source_path, *, output_path, label, **kw):
        seen_labels.append(label)
        Path(output_path).write_bytes(b"\x00")
        return True, ""

    with mock.patch.object(mod, "_render_variant", side_effect=fake_render), \
         mock.patch.object(
             mod, "_stitch_three_up",
             side_effect=lambda *a, **k: (True, ""),
         ):
        mod.render_ab_preview(
            str(src), output_path=out_path,
            label_a="A!", label_b="B!", label_c="C!",
        )
    assert seen_labels == ["A!", "B!", "C!"]


def test_run_returncode_handles_no_ffmpeg(tmp_path):
    """When ffmpeg isn't on PATH, the variant render reports the error
    cleanly instead of raising."""
    from backend.scripts import render_ab_preview as mod
    src = tmp_path / "source.mp4"
    src.write_bytes(b"\x00")
    with mock.patch("backend.scripts.render_ab_preview.shutil.which",
                    return_value=None):
        ok, log = mod._render_variant(
            str(src), output_path=str(tmp_path / "v.mp4"), label="x",
        )
    assert ok is False
    assert "ffmpeg" in log


def test_endpoint_routes_registered_in_source():
    """The /sota-clip-ab and /sota-clip-ab-preview routes are wired in
    diagnostics.py. Verified via source string match — importing the
    real router pulls heavy deps (httpx, fastapi) we don't need here.
    """
    src = (Path(_REPO) / "backend/routers/diagnostics.py").read_text()
    assert '@router.post("/sota-clip-ab")' in src
    assert '@router.get("/sota-clip-ab-preview/{token}.mp4")' in src
    # Phase event the UI subscribes to.
    assert '"ab_render"' in src
    # The endpoint shells out to render_ab_preview — verify the call.
    assert "render_ab_preview" in src
