"""Phase E QA — editorial planner + LP soft prior + genre allowlist removal.

Validates the keystone phase: per-genre prompt dispatch, robust JSON
parsing (strict / lenient / regex-extract), cache hit/miss, LP-target
integration, and the ``HUMAN_REFRAME_ALLOWED_CONTENT_TYPES`` widening.

Run:  pytest tests/qa/test_phase_e_editorial.py -v
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from pathlib import Path
from unittest import mock

import pytest

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


# ──────────────────────────────────────────────────────────────────
# 1. Genre dispatch (10 templates + default)
# ──────────────────────────────────────────────────────────────────


class TestGenreDispatch:
    def test_all_ten_templates_present(self):
        from backend.services.editorial_planner import GENRE_TO_TEMPLATE
        # Should cover every promised genre.
        promised = {
            "talking_head", "multi_speaker_panel", "sports", "gaming",
            "music_video", "narrative", "anime", "tutorial", "vlog",
        }
        assert promised.issubset(set(GENRE_TO_TEMPLATE.keys()))

    def test_template_files_exist(self):
        from backend.services.editorial_planner import (
            DEFAULT_PROMPT_DIR, GENRE_TO_TEMPLATE,
        )
        for ct, fname in GENRE_TO_TEMPLATE.items():
            path = DEFAULT_PROMPT_DIR / fname
            assert path.exists(), f"prompt template missing: {fname} for {ct}"

    def test_default_template_exists(self):
        from backend.services.editorial_planner import (
            DEFAULT_PROMPT_DIR, DEFAULT_TEMPLATE,
        )
        assert (DEFAULT_PROMPT_DIR / DEFAULT_TEMPLATE).exists()

    def test_unknown_genre_falls_back_to_default(self):
        from backend.services.editorial_planner import _load_template
        # No real "scifi_horror" template — should use default.
        text = _load_template("scifi_horror")
        # Just verify we got non-empty text from the default file.
        assert len(text) > 100

    def test_each_template_mentions_inputs_placeholder(self):
        """Every template MUST contain {inputs} so format_inputs survives."""
        from backend.services.editorial_planner import (
            DEFAULT_PROMPT_DIR, GENRE_TO_TEMPLATE, DEFAULT_TEMPLATE,
        )
        all_files = set(GENRE_TO_TEMPLATE.values()) | {DEFAULT_TEMPLATE}
        for fname in all_files:
            text = (DEFAULT_PROMPT_DIR / fname).read_text(encoding="utf-8")
            assert "{inputs}" in text, f"{fname} missing {{inputs}} placeholder"


# ──────────────────────────────────────────────────────────────────
# 2. Robust JSON parsing
# ──────────────────────────────────────────────────────────────────


class TestRobustJson:
    def test_strict_json_round_trip(self):
        from backend.services.editorial_planner import parse_plan_response
        payload = json.dumps({
            "genre": "talking_head",
            "shots": [
                {"start": 0, "end": 5, "intent": "establish",
                 "framing": "medium", "subject": "speaker"},
            ],
        })
        plan = parse_plan_response(payload, content_type="talking_head")
        assert len(plan.shots) == 1
        assert plan.shots[0].intent == "establish"

    def test_extract_json_from_markdown_fenced_response(self):
        from backend.services.editorial_planner import parse_plan_response
        text = (
            "Sure — here's the plan:\n```json\n"
            '{"genre": "vlog", "shots": [{"start": 0, "end": 3, '
            '"framing": "medium"}]}\n```\nLet me know.'
        )
        plan = parse_plan_response(text, content_type="vlog")
        assert len(plan.shots) == 1

    def test_lenient_recovery_from_trailing_comma(self):
        from backend.services.editorial_planner import parse_plan_response
        # Trailing comma — strict json.loads rejects, regex extracts
        # the largest balanced block which still has the comma but
        # the regex extract yields a json that's still parseable IF
        # json5 is installed; we just expect the parser to gracefully
        # return either a parsed plan or empty (never crash).
        plan = parse_plan_response(
            '{"genre": "narrative", "shots": [{"start": 0, "end": 5,}],}',
            content_type="narrative",
        )
        # Either parsed or empty — shouldn't crash.
        assert plan.genre == "narrative"

    def test_garbage_returns_empty_plan(self):
        from backend.services.editorial_planner import parse_plan_response
        plan = parse_plan_response("not json at all", content_type="anime")
        assert plan.is_empty
        assert plan.genre == "anime"

    def test_empty_string_returns_empty_plan(self):
        from backend.services.editorial_planner import parse_plan_response
        plan = parse_plan_response("", content_type="default")
        assert plan.is_empty


# ──────────────────────────────────────────────────────────────────
# 3. Cache hit / miss
# ──────────────────────────────────────────────────────────────────


class _StubOrchestrator:
    def __init__(self, *, returns: str = "{}"):
        self._returns = returns
        self.calls = 0

    async def text_completion(
        self, prompt, max_tokens=4096, timeout=60, job_id="",
    ):
        self.calls += 1
        return self._returns


class TestCache:
    def test_cache_miss_then_hit(self):
        from backend.services.editorial_planner import (
            EditorialPlannerInputs, plan_clip,
        )
        inputs = EditorialPlannerInputs(
            content_type="talking_head",
            scene_descriptions=[{"timestamp": 0, "description": "speaker"}],
            duration_sec=10.0,
        )
        payload = json.dumps({
            "genre": "talking_head",
            "shots": [{"start": 0, "end": 10, "intent": "hold",
                       "framing": "medium", "subject": "speaker"}],
        })
        orch = _StubOrchestrator(returns=payload)
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            plan1 = asyncio.run(plan_clip(
                inputs, orchestrator=orch, cache_dir=cache_dir,
            ))
            plan2 = asyncio.run(plan_clip(
                inputs, orchestrator=orch, cache_dir=cache_dir,
            ))
        assert orch.calls == 1
        assert len(plan1.shots) == 1
        assert len(plan2.shots) == 1


# ──────────────────────────────────────────────────────────────────
# 4. LLM failure → empty plan, no crash
# ──────────────────────────────────────────────────────────────────


class _CrashingOrchestrator:
    async def text_completion(self, prompt, **kwargs):
        raise RuntimeError("provider chain exhausted")


class TestRobustness:
    def test_provider_failure_returns_empty_plan(self):
        from backend.services.editorial_planner import (
            EditorialPlannerInputs, plan_clip,
        )
        inputs = EditorialPlannerInputs(content_type="vlog", duration_sec=5.0)
        with tempfile.TemporaryDirectory() as tmp:
            plan = asyncio.run(plan_clip(
                inputs, orchestrator=_CrashingOrchestrator(),
                cache_dir=Path(tmp),
            ))
        assert plan.is_empty
        assert plan.genre == "vlog"


# ──────────────────────────────────────────────────────────────────
# 5. LP soft-prior integration
# ──────────────────────────────────────────────────────────────────


class TestLpSoftPrior:
    def test_apply_plan_emits_zoom_hints(self):
        from backend.services.editorial_planner import (
            EditorialPlan, EditorialShot, apply_plan_to_lp_targets,
        )
        plan = EditorialPlan(shots=[
            EditorialShot(start=0.0, end=2.0, framing="tight"),
            EditorialShot(start=2.0, end=4.0, framing="wide"),
        ])
        timestamps = [0.0, 1.0, 2.0, 3.0]
        tx = [0.5, 0.5, 0.5, 0.5]
        ty = [0.5, 0.5, 0.5, 0.5]
        _, _, zoom = apply_plan_to_lp_targets(plan, timestamps, tx, ty)
        # Frames 0-1 should map to "tight" zoom (~0.7), 2-3 to "wide" (~1.15).
        assert zoom[0] == pytest.approx(0.70)
        assert zoom[1] == pytest.approx(0.70)
        assert zoom[2] == pytest.approx(1.15)
        assert zoom[3] == pytest.approx(1.15)

    def test_empty_plan_returns_default_zoom(self):
        from backend.services.editorial_planner import (
            EditorialPlan, apply_plan_to_lp_targets,
        )
        plan = EditorialPlan()
        _, _, zoom = apply_plan_to_lp_targets(
            plan, [0.0, 1.0, 2.0], [0.5] * 3, [0.5] * 3,
        )
        assert all(z == 1.0 for z in zoom)


# ──────────────────────────────────────────────────────────────────
# 6. Allowlist removal — every promised genre now allowed
# ──────────────────────────────────────────────────────────────────


class TestAllowlistRemoval:
    def test_every_promised_genre_now_allowed(self):
        from backend.services.human_reframe_pipeline import (
            HUMAN_REFRAME_ALLOWED_CONTENT_TYPES,
        )
        promised = {
            "multi_speaker_panel", "talking_head",
            "podcast", "vlog", "interview",
            "narrative", "documentary",
            "music_video", "concert", "performance",
            "sports", "sports_basketball", "sports_racing",
            "gaming", "gameplay",
            "anime", "animation", "animation_dialogue",
            "tutorial",
        }
        missing = promised - HUMAN_REFRAME_ALLOWED_CONTENT_TYPES
        assert not missing, f"missing from allowlist: {missing}"

    def test_allowlist_at_least_doubled(self):
        from backend.services.human_reframe_pipeline import (
            HUMAN_REFRAME_ALLOWED_CONTENT_TYPES,
        )
        # Pre-Phase-E it was just 2 entries; Phase E expands to many more.
        assert len(HUMAN_REFRAME_ALLOWED_CONTENT_TYPES) >= 15


# ──────────────────────────────────────────────────────────────────
# 7. Cache key stability
# ──────────────────────────────────────────────────────────────────


class TestCacheKey:
    def test_identical_inputs_same_key(self):
        from backend.services.editorial_planner import (
            EditorialPlannerInputs, _cache_key,
        )
        a = EditorialPlannerInputs(
            content_type="vlog", duration_sec=5.0,
            scene_descriptions=[{"description": "test"}],
        )
        b = EditorialPlannerInputs(
            content_type="vlog", duration_sec=5.0,
            scene_descriptions=[{"description": "test"}],
        )
        assert _cache_key(a) == _cache_key(b)

    def test_different_content_type_different_key(self):
        from backend.services.editorial_planner import (
            EditorialPlannerInputs, _cache_key,
        )
        a = EditorialPlannerInputs(content_type="vlog", duration_sec=5.0)
        b = EditorialPlannerInputs(content_type="anime", duration_sec=5.0)
        assert _cache_key(a) != _cache_key(b)


# ──────────────────────────────────────────────────────────────────
# 8. Phase E config defaults
# ──────────────────────────────────────────────────────────────────


class TestConfigDefaultsAfterPhaseE:
    def test_all_phase_flags_on_by_default(self):
        try:
            from backend.config import Settings
        except ModuleNotFoundError as exc:
            pytest.skip(f"backend.config deps missing: {exc}")
        s = Settings()
        # Phase E flips defaults to ON.
        assert s.CLIPAI_DENSE_POINT_TRACKING is True
        assert s.CLIPAI_SALIENCY_ENABLED is True
        assert s.CLIPAI_COMPOSITION_HEAD == "clip"
        assert s.CLIPAI_EDITORIAL_PLANNER is True
        # Emergency rollback OFF by default.
        assert s.CLIPAI_LEGACY_REFRAME is False

    def test_config_source_documents_phase_e_rollout(self):
        config_src = Path(_REPO / "backend" / "config.py").read_text()
        assert "Phase E" in config_src
        assert "rollback" in config_src.lower()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
