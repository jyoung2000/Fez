"""Phase 5 acceptance tests — editorial crop QA scoring + recovery.

Exercises ``score_crop_quality`` with an injected fake VLM scorer so
we can drive the aggregate-and-decide path without any OpenRouter
round-trip.
"""

from backend.services.crop_qa import (
    CROP_QA_ENV,
    CropQualityReport,
    CropQualitySample,
    QA_ACCEPTABLE_SCORE,
    QA_FATAL_SCORE,
    apply_recovery_params,
    crop_qa_enabled,
    decide_recovery,
    score_crop_quality,
)


def _scorer(by_ts: dict) -> callable:
    """Build a fake scorer that returns the given responses keyed by ts."""
    def _fn(ts):
        return by_ts.get(round(ts, 2), {
            "head_in_frame": True,
            "awkward_crop": False,
            "subject_partially_off_frame": False,
            "dead_space_dominant": False,
            "quality_score": 9,
        })
    return _fn


# ── Aggregate / acceptance ─────────────────────────────────────────


def test_qa_perfect_crop_no_recovery():
    scorer = _scorer({
        1.0: {"quality_score": 9},
        2.0: {"quality_score": 10},
        3.0: {"quality_score": 8},
    })
    report = score_crop_quality(
        segment_index=0,
        sample_timestamps=[1.0, 2.0, 3.0],
        vlm_scorer=scorer,
    )
    assert report.aggregate_score == 9.0
    assert report.triggered_recovery is None
    assert len(report.samples) == 3


def test_qa_subject_off_frame_triggers_loose_dead_zone():
    """Score 6.0 + any subject_partially_off_frame → loose_dead_zone."""
    scorer = _scorer({
        1.0: {
            "quality_score": 6,
            "subject_partially_off_frame": True,
        },
        2.0: {"quality_score": 6},
        3.0: {"quality_score": 6},
    })
    report = score_crop_quality(0, [1.0, 2.0, 3.0], scorer)
    assert report.aggregate_score == 6.0
    assert report.triggered_recovery == "loose_dead_zone"


def test_qa_awkward_crop_triggers_padding():
    scorer = _scorer({
        1.0: {"quality_score": 6, "awkward_crop": True},
        2.0: {"quality_score": 6},
        3.0: {"quality_score": 6},
    })
    report = score_crop_quality(0, [1.0, 2.0, 3.0], scorer)
    assert report.triggered_recovery == "padding"


def test_qa_severe_failure_falls_to_safety_center():
    scorer = _scorer({
        1.0: {
            "quality_score": 2,
            "subject_partially_off_frame": True,
            "head_in_frame": False,
        },
        2.0: {"quality_score": 3},
        3.0: {"quality_score": 4},
    })
    report = score_crop_quality(0, [1.0, 2.0, 3.0], scorer)
    assert report.aggregate_score == 3.0
    assert report.triggered_recovery == "safety_center"


def test_qa_borderline_7_is_accepted():
    """A 7.0 aggregate is exactly at the acceptable threshold."""
    scorer = _scorer({
        1.0: {"quality_score": 7},
        2.0: {"quality_score": 7},
        3.0: {"quality_score": 7},
    })
    report = score_crop_quality(0, [1.0, 2.0, 3.0], scorer)
    assert report.aggregate_score == 7.0
    assert report.triggered_recovery is None


def test_qa_weak_without_specific_failure_defaults_to_loose_dead_zone():
    """Aggregate 6 but no explicit failure flag → loose_dead_zone default."""
    scorer = _scorer({
        1.0: {"quality_score": 6},
        2.0: {"quality_score": 6},
        3.0: {"quality_score": 6},
    })
    report = score_crop_quality(0, [1.0, 2.0, 3.0], scorer)
    assert report.triggered_recovery == "loose_dead_zone"


# ── decide_recovery rule table ─────────────────────────────────────


def _mk_report(score: float, off=False, awkward=False) -> CropQualityReport:
    return CropQualityReport(
        segment_index=0,
        samples=[CropQualitySample(
            timestamp=0.0,
            head_in_frame=True,
            awkward_crop=awkward,
            subject_partially_off_frame=off,
            dead_space_dominant=False,
            quality_score=score,
        )],
        aggregate_score=score,
    )


def test_decide_recovery_threshold_edges():
    assert decide_recovery(_mk_report(9.0)) is None
    assert decide_recovery(_mk_report(7.0)) is None
    assert decide_recovery(_mk_report(6.99, off=True)) == "loose_dead_zone"
    assert decide_recovery(_mk_report(5.0, off=True)) == "loose_dead_zone"
    assert decide_recovery(_mk_report(4.99, off=True)) == "safety_center"
    assert decide_recovery(_mk_report(0.0)) == "safety_center"


# ── apply_recovery_params ──────────────────────────────────────────


def test_apply_recovery_params_loose_dead_zone():
    dz, l2, pad = apply_recovery_params(
        "loose_dead_zone",
        dead_zone_px=100.0, lambda2=1.0, crop_padding_frac=0.1,
    )
    assert dz == 150.0
    assert l2 == 0.5
    assert pad == 0.1


def test_apply_recovery_params_padding():
    dz, l2, pad = apply_recovery_params(
        "padding",
        dead_zone_px=100.0, lambda2=1.0, crop_padding_frac=0.1,
    )
    assert dz == 100.0
    assert l2 == 1.0
    assert abs(pad - 0.15) < 1e-9


def test_apply_recovery_params_safety_center_passthrough():
    out = apply_recovery_params(
        "safety_center", 100.0, 1.0, 0.1,
    )
    assert out == (100.0, 1.0, 0.1)


def test_apply_recovery_params_unknown_returns_inputs():
    out = apply_recovery_params("no_op", 100.0, 1.0, 0.1)
    assert out == (100.0, 1.0, 0.1)


# ── Env flag ───────────────────────────────────────────────────────


def test_qa_default_flag_off(monkeypatch):
    monkeypatch.delenv(CROP_QA_ENV, raising=False)
    assert not crop_qa_enabled()


def test_qa_flag_on(monkeypatch):
    for val in ("1", "true", "on", "Yes"):
        monkeypatch.setenv(CROP_QA_ENV, val)
        assert crop_qa_enabled(), val


def test_qa_disabled_no_calls(monkeypatch):
    """When the flag is off, score_crop_quality is never invoked from
    the pipeline — but the module itself is still importable and
    callable for test/debug purposes."""
    monkeypatch.delenv(CROP_QA_ENV, raising=False)
    # Directly calling the function still works; the gate is the
    # caller's responsibility.
    scorer_calls = []

    def never_called(ts):
        scorer_calls.append(ts)
        return {"quality_score": 10}

    # Empty timestamp list → no calls.
    report = score_crop_quality(0, [], never_called)
    assert scorer_calls == []
    assert report.aggregate_score == 0.0
    assert report.triggered_recovery is None


# ── Threshold constants locked in ──────────────────────────────────


def test_threshold_constants_match_spec():
    assert QA_ACCEPTABLE_SCORE == 7.0
    assert QA_FATAL_SCORE == 5.0
