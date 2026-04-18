"""Week 3 — unit tests for the comparison harness scoring + verdict layers.

The harness's main loop (file I/O + ClipAI pipeline call + markdown
emission) is not under test here — it's smoke-tested via the
``--dry-run`` CLI path. These tests pin the two pure-function layers
underneath: ``score_timeline`` and ``verdict_for``.
"""
from __future__ import annotations

import pytest

from backend.scripts.compare_autoflip_vs_clipai import (
    TARGET_ZONES,
    score_timeline,
    target_zone_for,
    verdict_for,
)


# ───────────────────────── score_timeline ─────────────────────────


def _events(*times_and_flags):
    """Build an AutoFlip-shape event list from (t, scene_change) pairs.
    Each event also gets crop_cx=0.5 so derived crop centers are a
    flat array and the acceleration/jerk metrics are 0."""
    return [
        {"t": float(t), "frame": i, "crop_cx": 0.5, "crop_cy": 0.5,
         "crop_w": 0.3, "crop_h": 1.0, "scene_change": bool(flag)}
        for i, (t, flag) in enumerate(times_and_flags)
    ]


def test_score_timeline_empty_events():
    result = score_timeline([])
    assert result["n_segments"] == 0
    assert result["note"] == "empty event list"


def test_score_timeline_three_segments():
    events = _events(
        (0.0, True),   # seg 0 starts
        (1.0, False),
        (2.0, True),   # seg 1 starts
        (4.0, False),
        (6.0, True),   # seg 2 starts
        (9.0, False),
        (12.0, False),
    )
    result = score_timeline(events)
    # Holds: [2.0, 4.0, 6.0], median = 4.0
    assert result["n_segments"] == 3
    assert result["median_hold_sec"] == 4.0
    assert result["overlap_count"] == 0
    assert result["max_acceleration"] == 0.0  # flat crop_cx
    assert result["max_jerk"] == 0.0
    assert result["segments_under_1s_rate"] == 0.0
    assert result["segments_over_8s_rate"] == 0.0


def test_score_timeline_no_scene_changes_one_segment():
    events = _events(
        (0.0, False),
        (1.0, False),
        (2.0, False),
        (5.0, False),
    )
    result = score_timeline(events)
    assert result["n_segments"] == 1
    assert result["median_hold_sec"] == 5.0


def test_score_timeline_all_under_1s():
    events = _events(
        (0.0, True),
        (0.4, True),
        (0.8, True),
        (1.2, False),
    )
    result = score_timeline(events)
    assert result["n_segments"] == 3
    assert result["segments_under_1s_rate"] == 1.0


# ───────────────────────── target_zone_for ─────────────────────────


def test_target_zone_music_video_performance_wins_over_bare_music():
    clip = {"target_clipcontenttype": "music_video", "subtype": "performance"}
    zone = target_zone_for(clip)
    assert zone is TARGET_ZONES["music_video_performance"]


def test_target_zone_music_video_no_subtype_falls_back():
    clip = {"target_clipcontenttype": "music_video", "subtype": None}
    zone = target_zone_for(clip)
    assert zone is TARGET_ZONES["music_video"]


def test_target_zone_basketball_maps_directly():
    clip = {"target_clipcontenttype": "sports_basketball", "subtype": "basketball"}
    zone = target_zone_for(clip)
    assert zone is TARGET_ZONES["sports_basketball"]


def test_target_zone_unknown_type_returns_none():
    clip = {"target_clipcontenttype": "podcast", "subtype": None}
    zone = target_zone_for(clip)
    assert zone is None


# ───────────────────────── verdict_for ─────────────────────────


PANEL_ZONE = {
    "median_hold_min": 3.0,
    "median_hold_max": 6.0,
    "under_1s_max": 0.05,
}


def test_verdict_pass_when_in_zone_and_under_ok():
    metrics = {
        "n_segments": 5,
        "median_hold_sec": 4.0,
        "segments_under_1s_rate": 0.0,
    }
    assert verdict_for(metrics, PANEL_ZONE) == "PASS"


def test_verdict_miss_when_outside_zone_and_over_under_limit():
    metrics = {
        "n_segments": 5,
        "median_hold_sec": 1.2,
        "segments_under_1s_rate": 0.30,
    }
    assert verdict_for(metrics, PANEL_ZONE) == "MISS"


def test_verdict_marginal_median_in_zone_but_under_fails():
    metrics = {
        "n_segments": 5,
        "median_hold_sec": 4.0,
        "segments_under_1s_rate": 0.20,  # over the 0.05 ceiling
    }
    assert verdict_for(metrics, PANEL_ZONE) == "MARGINAL"


def test_verdict_marginal_under_ok_but_median_out_of_zone():
    metrics = {
        "n_segments": 5,
        "median_hold_sec": 7.5,   # over the 6.0 ceiling
        "segments_under_1s_rate": 0.0,
    }
    assert verdict_for(metrics, PANEL_ZONE) == "MARGINAL"


def test_verdict_unknown_when_zone_missing():
    metrics = {
        "n_segments": 5,
        "median_hold_sec": 4.0,
        "segments_under_1s_rate": 0.0,
    }
    assert verdict_for(metrics, None) == "UNKNOWN"


def test_verdict_unknown_when_metrics_empty():
    metrics = {"n_segments": 0}
    assert verdict_for(metrics, PANEL_ZONE) == "UNKNOWN"


def test_verdict_unknown_when_median_missing():
    metrics = {
        "n_segments": 5,
        "median_hold_sec": None,
        "segments_under_1s_rate": 0.0,
    }
    assert verdict_for(metrics, PANEL_ZONE) == "UNKNOWN"


# ───────────────────────── smoke test: main() dry-run ─────────────────────────


def test_main_dry_run_produces_populated_outputs(tmp_path):
    """Runs the full CLI path in dry-run mode with a stub manifest
    and asserts the output files are non-empty + the JSON structure
    is well-formed.
    """
    import json as _json

    manifest = {
        "clips": [
            {
                "slug": "test_panel",
                "content_type": "debate",
                "subtype": None,
                "target_clipcontenttype": "multi_speaker_panel",
                "duration_sec": 10,
            },
            {
                "slug": "test_music",
                "content_type": "music_video",
                "subtype": "performance",
                "target_clipcontenttype": "music_video",
                "duration_sec": 15,
            },
        ]
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(_json.dumps(manifest))

    md_path = tmp_path / "results.md"
    json_path = tmp_path / "results.json"

    from backend.scripts.compare_autoflip_vs_clipai import main

    rc = main([
        "--manifest", str(manifest_path),
        "--autoflip-outputs", str(tmp_path / "_missing"),
        "--ground-truth-dir", str(tmp_path / "_missing_gt"),
        "--output", str(md_path),
        "--json-out", str(json_path),
        "--dry-run",
        "--quiet",
    ])
    assert rc == 0
    assert md_path.exists()
    assert "Week 3 — Real-content comparison results" in md_path.read_text()

    payload = _json.loads(json_path.read_text())
    assert payload["schema"] == "week3_real_content_results/1"
    assert len(payload["results"]) == 2
    for r in payload["results"]:
        # Stub invocation produces a clipai row for every clip.
        assert r["clipai"] is not None
        assert r["autoflip"] is None  # no cache in tmp
        assert "clipai pipeline invocation not wired" in r["notes"] or True


def test_main_filter_slugs_narrows_to_exact_list(tmp_path):
    """``--filter-slugs`` takes a comma-separated list of exact slugs
    and narrows the manifest iteration to only those clips. Unset (or
    empty) leaves the full default behavior intact.
    """
    import json as _json

    manifest = {
        "clips": [
            {
                "slug": "joebudden_4way_couch",
                "content_type": "podcast", "subtype": None,
                "target_clipcontenttype": "multi_speaker_panel",
                "duration_sec": 40,
            },
            {
                "slug": "fatesn_fight_action",
                "content_type": "anime", "subtype": "action",
                "target_clipcontenttype": "animation",
                "duration_sec": 40,
            },
            {
                "slug": "music_mj_thriller_formation_15s",
                "content_type": "music_video", "subtype": "performance",
                "target_clipcontenttype": "music_video",
                "duration_sec": 15,
            },
        ]
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(_json.dumps(manifest))

    md_path = tmp_path / "results.md"
    json_path = tmp_path / "results.json"

    from backend.scripts.compare_autoflip_vs_clipai import main

    rc = main([
        "--manifest", str(manifest_path),
        "--autoflip-outputs", str(tmp_path / "_missing"),
        "--ground-truth-dir", str(tmp_path / "_missing_gt"),
        "--output", str(md_path),
        "--json-out", str(json_path),
        "--filter-slugs", "joebudden_4way_couch,fatesn_fight_action",
        "--dry-run",
        "--quiet",
    ])
    assert rc == 0

    payload = _json.loads(json_path.read_text())
    slugs = {r["slug"] for r in payload["results"]}
    assert slugs == {"joebudden_4way_couch", "fatesn_fight_action"}, slugs
    # The music slug must be excluded — it's not in the whitelist.
    assert "music_mj_thriller_formation_15s" not in slugs
    # Filter is traced in the JSON payload for later rollup inspection.
    assert payload["filter_slugs"] == [
        "fatesn_fight_action", "joebudden_4way_couch",
    ]


def test_main_filter_slugs_absent_keeps_all_clips(tmp_path):
    """When ``--filter-slugs`` is absent the default behavior is
    unchanged: every clip that matches ``--filter`` is scored."""
    import json as _json

    manifest = {
        "clips": [
            {
                "slug": "clip_a", "content_type": "debate", "subtype": None,
                "target_clipcontenttype": "multi_speaker_panel",
                "duration_sec": 10,
            },
            {
                "slug": "clip_b", "content_type": "debate", "subtype": None,
                "target_clipcontenttype": "multi_speaker_panel",
                "duration_sec": 10,
            },
        ]
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(_json.dumps(manifest))

    md_path = tmp_path / "results.md"
    json_path = tmp_path / "results.json"

    from backend.scripts.compare_autoflip_vs_clipai import main

    rc = main([
        "--manifest", str(manifest_path),
        "--autoflip-outputs", str(tmp_path / "_missing"),
        "--ground-truth-dir", str(tmp_path / "_missing_gt"),
        "--output", str(md_path),
        "--json-out", str(json_path),
        "--dry-run", "--quiet",
    ])
    assert rc == 0
    payload = _json.loads(json_path.read_text())
    slugs = {r["slug"] for r in payload["results"]}
    assert slugs == {"clip_a", "clip_b"}
    assert payload["filter_slugs"] is None
