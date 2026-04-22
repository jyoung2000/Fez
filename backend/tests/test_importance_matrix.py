"""Blueprint v2 Phase 1 — Stage 3 importance matrix.

Single source of truth for how the reframing stack mixes Stage 3
signals (face, saliency, motion, depth, object). Before Phase 1
these weights were scattered across required_regions.py,
genre_refinements.py, content_type_config.py, and per-genre files.
"""

import pytest

from backend.services.reframe_config import (
    ImportanceWeights,
    _IMPORTANCE_MATRIX,
    get_default_config,
    importance_for,
    importance_matrix_as_dict,
)


# ── Per-genre weights match blueprint ─────────────────────────────


def test_talking_head_face_dominant():
    cfg = get_default_config().for_content("talking_head")
    assert cfg.importance.face == 1.0
    assert cfg.importance.face > cfg.importance.saliency * 3


def test_sports_racing_motion_dominant():
    cfg = get_default_config().for_content("sports_racing")
    assert cfg.importance.motion > cfg.importance.face
    assert cfg.importance.object == 1.0


def test_landscape_face_zero():
    cfg = get_default_config().for_content("landscape")
    assert cfg.importance.face == 0.0
    assert cfg.importance.saliency == 1.0


def test_gameplay_moba_saliency_dominant():
    cfg = get_default_config().for_content("gameplay_moba")
    assert cfg.importance.saliency == 0.8
    assert cfg.importance.face == 0.2


def test_required_gain_dominates_soft_weights():
    for key, w in _IMPORTANCE_MATRIX.items():
        soft_sum = w.face + w.saliency + w.motion + w.depth + w.object
        # Blueprint invariant: H = 1000 must dominate the soft sum.
        assert w.required_region_gain >= soft_sum * 100, (
            f"{key}: required_region_gain {w.required_region_gain} not "
            f">= 100x soft sum {soft_sum}"
        )


# ── Fallbacks ─────────────────────────────────────────────────────


def test_unknown_falls_back_to_generic():
    a = get_default_config().for_content("unknown").importance
    b = get_default_config().for_content("generic").importance
    assert a == b


def test_missing_key_falls_back_to_generic():
    a = importance_for("not-a-real-content-type").face
    b = importance_for("generic").face
    assert a == b


def test_none_content_type_falls_back_to_generic():
    a = importance_for(None).face
    b = importance_for("generic").face
    assert a == b


def test_enum_like_input_is_unwrapped():
    class _FakeCT:
        def __init__(self, v):
            self.value = v
    assert importance_for(_FakeCT("talking_head")).face == 1.0


# ── Every known ClipContentType has an entry ──────────────────────


def test_all_known_content_types_have_entry():
    """Guards against silently missing genres — every value in
    ClipContentType must have an explicit matrix entry so QA can
    tune it without a code change. (Entries may coincidentally
    match the generic defaults — the check is presence, not
    distinctness.)"""
    from backend.services.content_classifier import ClipContentType
    for t in ClipContentType:
        w = importance_for(t.value)
        assert w.face is not None
        assert t.value in _IMPORTANCE_MATRIX, (
            f"ClipContentType.{t.name} ({t.value!r}) missing from matrix"
        )


# ── Dict serialization ─────────────────────────────────────────────


def test_importance_matrix_as_dict_is_json_safe():
    import json

    data = importance_matrix_as_dict()
    json.dumps(data)
    assert data["talking_head"]["face"] == 1.0
    assert data["landscape"]["saliency"] == 1.0
    assert data["generic"]["face"] == 1.0


# ── ReframeConfig plumbing ────────────────────────────────────────


def test_for_content_overlays_matrix_row_automatically():
    # Phase 1: ``for_content(ct)`` applies the matrix row even when the
    # ``content_overrides`` table has no entry for that key.
    cfg = get_default_config()
    # Default config has generic importance.
    assert cfg.importance.face == 1.0
    panel = cfg.for_content("multi_speaker_panel")
    assert panel.importance.face == 1.0  # still 1.0 per matrix
    panel_v = cfg.for_content("landscape")
    assert panel_v.importance.face == 0.0  # matrix row applied


def test_for_content_explicit_override_wins_over_matrix():
    custom = ImportanceWeights(face=0.42, saliency=0.42, motion=0.42)
    cfg = get_default_config()
    overrides = dict(cfg.content_overrides)
    overrides["sports"] = {**overrides.get("sports", {}), "importance": custom}
    cfg_custom = cfg.override(content_overrides=overrides)
    assert cfg_custom.for_content("sports").importance == custom


# ── content_type_config shim parity ──────────────────────────────


@pytest.mark.parametrize("ct,field_name,expected", [
    ("talking_head", "face", 1.0),
    ("landscape", "face", 0.0),
    ("sports_racing", "object", 1.0),
    ("gameplay_moba", "saliency", 0.8),
    ("music_video", "motion", 0.4),
])
def test_content_type_config_shim_matches_matrix(ct, field_name, expected):
    from backend.services.content_type_config import (
        get_depth_weight,
        get_face_weight,
        get_motion_weight,
        get_object_weight,
        get_saliency_weight,
    )
    dispatch = {
        "face": get_face_weight,
        "saliency": get_saliency_weight,
        "motion": get_motion_weight,
        "object": get_object_weight,
        "depth": get_depth_weight,
    }
    assert dispatch[field_name](ct) == pytest.approx(expected)
