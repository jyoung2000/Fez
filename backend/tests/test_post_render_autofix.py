"""Blueprint v2 Phase 3 — auto-fix strategy table + splice + confidence."""

import pytest

from backend.services.post_render_critic import (
    AUTO_FIX_KINDS,
    COSMETIC_KINDS,
    IssueKind,
    PostRenderIssue,
    PostRenderReport,
    STRUCTURAL_KINDS,
    _parse_response,
)
from backend.services.post_render_autofix import (
    FixStrategy,
    _STRATEGIES,
    classify_clip_confidence,
    snap_to_shot_boundaries,
    strategy_for,
)


# ── Taxonomy ──────────────────────────────────────────────────────


def test_auto_fix_kinds_disjoint_from_structural():
    assert AUTO_FIX_KINDS.isdisjoint(STRUCTURAL_KINDS)
    assert AUTO_FIX_KINDS.isdisjoint(COSMETIC_KINDS)
    assert STRUCTURAL_KINDS.isdisjoint(COSMETIC_KINDS)


def test_all_auto_fix_kinds_have_strategy():
    """Every kind in AUTO_FIX_KINDS must have a strategy so the
    dispatcher never returns None for a kind we claim to handle."""
    for kind in AUTO_FIX_KINDS:
        strat = _STRATEGIES.get(kind)
        assert strat is not None, f"Missing strategy for {kind!r}"
        assert strat.issue_kind == kind


def test_parse_response_preserves_kind():
    raw = (
        '[{"t": 3.2, "kind": "tight_framing", "severity": "high", '
        '"issue": "edge clip"}]'
    )
    issues = _parse_response(raw)
    assert len(issues) == 1
    assert issues[0].kind == IssueKind.TIGHT_FRAMING


def test_parse_response_unknown_kind_becomes_other():
    raw = '[{"t": 1, "kind": "quantum_catastrophe", "issue": "x"}]'
    issues = _parse_response(raw)
    assert issues[0].kind == IssueKind.OTHER


def test_parse_response_missing_kind_becomes_other():
    """Back-compat: Phase 0 VLMs that don't know the taxonomy still
    produce parseable issues."""
    raw = '[{"t": 1, "issue": "something broken", "severity": "medium"}]'
    issues = _parse_response(raw)
    assert issues[0].kind == IssueKind.OTHER


# ── Strategy dispatcher ───────────────────────────────────────────


def test_strategy_for_auto_fixable_returns_strategy():
    issue = PostRenderIssue(t=1.0, issue="x", kind=IssueKind.JITTER)
    strat = strategy_for(issue)
    assert strat is not None
    assert strat.issue_kind == IssueKind.JITTER
    assert strat.config_overrides["lp_lambda_v"] == 40.0


def test_strategy_for_structural_returns_none():
    issue = PostRenderIssue(t=1.0, issue="x", kind=IssueKind.SUBJECT_LEFT_FRAME)
    assert strategy_for(issue) is None


def test_strategy_for_cosmetic_returns_none():
    issue = PostRenderIssue(t=1.0, issue="x", kind=IssueKind.SINGLE_FRAME_GLITCH)
    assert strategy_for(issue) is None


def test_strategy_for_unknown_kind_returns_none():
    issue = PostRenderIssue(t=1.0, issue="x", kind=IssueKind.OTHER)
    assert strategy_for(issue) is None


# ── Per-strategy parity guards ────────────────────────────────────


@pytest.mark.parametrize("kind,field_name,expected", [
    (IssueKind.TIGHT_FRAMING, "headroom_min", 0.08),
    (IssueKind.TIGHT_FRAMING, "headroom_max", 0.18),
    (IssueKind.HEAD_OR_CHIN_CLIP, "headroom_min", 0.10),
    (IssueKind.JITTER, "lp_lambda_v", 40.0),
    (IssueKind.JITTER, "deadband_frac", 0.14),
    (IssueKind.PAN_ACROSS_CUT, "ease_shot_cut_ms", 0),
    (IssueKind.PAN_ACROSS_CUT, "match_cut_threshold", 0.05),
])
def test_strategy_overrides_pinned(kind, field_name, expected):
    strat = _STRATEGIES[kind]
    assert strat.config_overrides.get(field_name) == expected


def test_widen_crop_frac_set_for_framing_issues():
    assert _STRATEGIES[IssueKind.TIGHT_FRAMING].widen_crop_frac > 0
    assert _STRATEGIES[IssueKind.HEAD_OR_CHIN_CLIP].widen_crop_frac > 0
    assert _STRATEGIES[IssueKind.TEXT_CUT_OFF].widen_crop_frac > 0


# ── Shot-boundary snapping ────────────────────────────────────────


def test_snap_to_shot_boundaries_snaps_nearby():
    start, end = snap_to_shot_boundaries(
        start=0.3, end=5.4, shot_boundaries=[0.0, 5.0, 10.0], tol=0.5,
    )
    assert start == 0.0
    assert end == 5.0


def test_snap_to_shot_boundaries_leaves_far_boundaries_alone():
    start, end = snap_to_shot_boundaries(
        start=2.0, end=7.0, shot_boundaries=[0.0, 20.0], tol=0.5,
    )
    assert start == 2.0
    assert end == 7.0


def test_snap_to_shot_boundaries_invalid_window_falls_back():
    # A snap that would collapse the window returns the original bounds.
    start, end = snap_to_shot_boundaries(
        start=3.0, end=3.1, shot_boundaries=[3.05, 3.05], tol=0.5,
    )
    assert start == 3.0
    assert end == 3.1


# ── Confidence classification ────────────────────────────────────


def _report(kinds):
    return PostRenderReport(
        ok=(len(kinds) == 0),
        issues=[
            PostRenderIssue(t=float(i), issue="x", kind=k)
            for i, k in enumerate(kinds)
        ],
    )


def test_confidence_high_when_no_issues():
    assert classify_clip_confidence(_report([])) == "high"


def test_confidence_low_on_any_structural_issue():
    assert classify_clip_confidence(
        _report([IssueKind.JITTER, IssueKind.SUBJECT_LEFT_FRAME])
    ) == "low"
    assert classify_clip_confidence(
        _report([IssueKind.HUD_ADJACENT])
    ) == "low"
    assert classify_clip_confidence(
        _report([IssueKind.OUTPAINT_HALLUC])
    ) == "low"


def test_confidence_medium_for_fixable_only():
    assert classify_clip_confidence(
        _report([IssueKind.TIGHT_FRAMING, IssueKind.JITTER])
    ) == "medium"


def test_confidence_medium_for_cosmetic_only():
    assert classify_clip_confidence(
        _report([IssueKind.SINGLE_FRAME_GLITCH])
    ) == "medium"


def test_confidence_none_report_returns_high():
    assert classify_clip_confidence(None) == "high"


# ── Auto-fixable / structural / cosmetic report helpers ──────────


def test_report_structural_and_fixable_helpers():
    r = PostRenderReport(
        ok=False,
        issues=[
            PostRenderIssue(t=1, issue="a", kind=IssueKind.TIGHT_FRAMING),
            PostRenderIssue(t=2, issue="b", kind=IssueKind.SUBJECT_LEFT_FRAME),
            PostRenderIssue(t=3, issue="c", kind=IssueKind.SINGLE_FRAME_GLITCH),
        ],
    )
    assert len(r.structural_issues()) == 1
    assert len(r.auto_fixable_issues()) == 1
    assert r.structural_issues()[0].kind == IssueKind.SUBJECT_LEFT_FRAME
