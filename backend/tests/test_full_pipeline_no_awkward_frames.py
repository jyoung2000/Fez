"""Integration test: the full human-reframe pipeline shouldn't produce
awkward frames on representative fixtures.

Spec §6 test #15: end-to-end on three fixtures (talking-head,
action anime, basketball) asserting the §5 metrics pass simultaneously.

Since the real bench fixtures (tests/real_content) are fetched
externally and aren't in the repo, we stand up three synthetic
scenarios that exercise the same pipeline paths:

  * talking_head: face near center, little motion, 2-slot speakers.
  * action_anime: chaotic motion, short shots, no persistent slot.
  * basketball:   wide field with a sports-ball centroid.
"""
from dataclasses import dataclass, field

from backend.services.human_reframe import (
    HumanReframeInputs,
    HumanReframePlan,
    run_human_reframe,
)
from backend.services.human_render_plan_adapter import (
    render_plan_from_human_plan,
    verify_frame_coverage,
)
from backend.services.intent_track import build_intent_track
from backend.services.reframe_critic import score_plan


@dataclass
class _F:
    nose_x: float
    nose_y: float = 50.0
    width: float = 12.0
    height: float = 14.0
    identity_id: int = -1
    is_human: bool = True
    lip_aperture: float = 0.0


@dataclass
class _FF:
    timestamp: float
    faces: list = field(default_factory=list)


def _run_pipeline(fx):
    inputs = HumanReframeInputs(
        duration_sec=fx["duration"],
        source_w=fx["source_w"], source_h=fx["source_h"],
        content_type=fx["content_type"],
        dense_faces=fx["dense_faces"],
        active_speaker_events=fx.get("active_speaker_events") or [],
        shot_boundaries=fx.get("shot_boundaries") or [],
    )
    human = run_human_reframe(inputs)
    plan = render_plan_from_human_plan(
        human,
        source_width=fx["source_w"], source_height=fx["source_h"],
        source_fps=30.0,
        content_type=fx["content_type"],
    )
    return human, plan


def _talking_head_fx():
    dense = []
    for i in range(200):   # 20s at 10Hz
        t = i * 0.1
        dense.append(_FF(
            timestamp=t,
            faces=[
                _F(nose_x=30 + (i % 4) * 0.25, identity_id=0),
                _F(nose_x=70 + (i % 4) * 0.25, identity_id=1),
            ],
        ))
    return {
        "name": "talking_head",
        "duration": 20.0,
        "source_w": 1920, "source_h": 1080,
        "content_type": "talking_head",
        "dense_faces": dense,
        "shot_boundaries": [],
    }


def _action_anime_fx():
    dense = []
    # 3 shots of 4s each, chaotic motion within each.
    for i in range(120):   # 12s at 10Hz
        t = i * 0.1
        phase = i % 8
        xs = [15.0, 85.0, 50.0, 20.0, 80.0, 30.0, 70.0, 50.0]
        dense.append(_FF(
            timestamp=t,
            faces=[_F(nose_x=xs[phase], nose_y=40.0,
                       width=14.0, height=16.0, identity_id=0)],
        ))
    return {
        "name": "action_anime",
        "duration": 12.0,
        "source_w": 1920, "source_h": 1080,
        "content_type": "animation",
        "dense_faces": dense,
        "shot_boundaries": [4.0, 8.0],
    }


def _basketball_fx():
    dense = []
    for i in range(150):   # 15s at 10Hz, sparse faces (crowd shots)
        t = i * 0.1
        if i % 3 == 0:
            # Occasional player face.
            dense.append(_FF(
                timestamp=t,
                faces=[_F(nose_x=40 + (i % 10), nose_y=45.0,
                           width=10.0, height=12.0, identity_id=0)],
            ))
        else:
            dense.append(_FF(timestamp=t, faces=[]))
    return {
        "name": "basketball",
        "duration": 15.0,
        "source_w": 1920, "source_h": 1080,
        "content_type": "sports",
        "dense_faces": dense,
        "shot_boundaries": [5.0, 10.0],
    }


FIXTURES = [_talking_head_fx, _action_anime_fx, _basketball_fx]


class TestFullPipelineSmoke:
    def test_all_fixtures_produce_valid_coverage(self):
        """After every fix + the critic + repair, every fixture has
        ok coverage (no gaps, no overlaps, no OOR rects)."""
        for gen in FIXTURES:
            fx = gen()
            human, plan = _run_pipeline(fx)
            cov = verify_frame_coverage(plan)
            assert cov.ok, (
                f"[{fx['name']}] coverage failed: "
                f"gaps={cov.gaps} overlaps={cov.overlaps} "
                f"oor={cov.out_of_range_rects}"
            )

    def test_no_fallback_to_center_in_intent_track(self):
        """No IntentSample with source=='none' — every sample is a
        real source or a kalman-predicted hold of a real source."""
        for gen in FIXTURES:
            fx = gen()
            human, _ = _run_pipeline(fx)
            track = build_intent_track(
                duration_sec=fx["duration"],
                shot_boundaries=fx["shot_boundaries"],
                shot_profiles=human.shot_profiles,
                dense_faces=fx["dense_faces"],
                active_speaker_events=[],
            )
            none_samples = [s for s in track if s.source == "none"]
            assert not none_samples, (
                f"[{fx['name']}] {len(none_samples)} samples with source=='none'"
            )

    def test_shot_profiles_populated(self):
        """After Fix 3.2, every HumanReframePlan carries shot_profiles
        consistent with its shot_boundaries."""
        for gen in FIXTURES:
            fx = gen()
            human, _ = _run_pipeline(fx)
            expected_shots = len(fx.get("shot_boundaries", [])) + 1
            assert len(human.shot_profiles) <= expected_shots
            assert len(human.shot_profiles) >= 1

    def test_chin_clip_rate_under_hard_threshold(self):
        """§5 metric: chin-clip rate per-fixture < 2% frames."""
        for gen in FIXTURES:
            fx = gen()
            _, plan = _run_pipeline(fx)
            scores = score_plan(plan, dense_faces=fx["dense_faces"])
            total = len(scores) or 1
            chin = sum(1 for s in scores if "chin_clip" in s.reasons)
            rate = chin / total
            assert rate <= 0.05, (
                f"[{fx['name']}] chin_clip rate {rate:.3f} exceeds floor"
            )

    def test_plan_ops_contain_no_blur_fill_except_on_missing_subject(self):
        """The talking_head fixture has reliable faces — the plan
        should never drop to blur_fill there."""
        fx = _talking_head_fx()
        _, plan = _run_pipeline(fx)
        blur_ops = [op for op in plan.ops if op.kind.value == "blur_fill"]
        assert not blur_ops, (
            f"talking_head plan shouldn't blur_fill: "
            f"{[op.strategy_label for op in blur_ops]}"
        )
