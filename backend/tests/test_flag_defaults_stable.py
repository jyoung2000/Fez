"""Regression guard: v2 editorial flags stay ON by default.

Week 1 of the AutoFlip-parity editorial work validated that these
flags ship ON without safety-metric regression on the synthetic
fixtures (modulo the two whitelisted ``phase8-editorial-prior-fp-drift``
entries — see ``docs/reframing_autoflip_parity.md`` Week 1 section).

Flipping any of these back to ``"0"`` without a corresponding
``validate_v2_phases --quick`` run + matching doc update in
``docs/content_type_routing.md`` / ``docs/reframing_autoflip_parity.md``
is a drift regression. This test catches the drift at the module-default
level before it lands in ``main``.

Each row is a ``(module_path, attribute_name, env_var_name)`` triple.
The test deletes the env var, reloads the module, and asserts the
attribute is ``True`` — i.e. the in-code default is ON.
"""
from __future__ import annotations

import importlib

import pytest


# (module_path, attribute_name, env_var_name)
#
# Keep this list sorted by module_path so new entries merge cleanly.
# Adding a row here requires a matching row in
# docs/content_type_routing.md's feature-flag inventory table.
ON_BY_DEFAULT = [
    (
        "backend.services.anime_anchor",
        "USE_ANIME_ANCHOR",
        "CLIPAI_ANIME_ANCHOR",
    ),
    # Week 2 Part A flip:
    (
        "backend.services.anime_face_detector",
        "USE_ANIME_FACE_DETECTOR",
        "CLIPAI_ANIME_FACE_DETECTOR",
    ),
    # Week 2 Part B flip:
    (
        "backend.services.anime_shot_detector",
        "USE_ANIME_SHOT_DETECTOR",
        "CLIPAI_ANIME_SHOT_DETECTOR",
    ),
    # Week 2 Part C flip:
    (
        "backend.services.anime_character_clustering",
        "USE_ANIME_CHARACTER_CLUSTERING",
        "CLIPAI_ANIME_CHARACTER_CLUSTERING",
    ),
    (
        "backend.services.beat_detector",
        "USE_MUSIC_BEAT_SNAP",
        "CLIPAI_MUSIC_BEAT_SNAP",
    ),
    (
        "backend.services.editorial_prior",
        "USE_EDITORIAL_PRIOR",
        "CLIPAI_EDITORIAL_PRIOR",
    ),
    (
        "backend.services.gameplay_subject_tracker",
        "USE_GAMEPLAY_TRACKER",
        "CLIPAI_GAMEPLAY_TRACKER",
    ),
    (
        "backend.services.gaze_estimator",
        "USE_GAZE_LEAD_ROOM_V2",
        "CLIPAI_GAZE_LEAD_ROOM_V2",
    ),
    # Week 1 flip: MRLP was OFF pre-Week-1 and flipped to ON after
    # the isolated OFF/ON comparison on 3speaker_panel showed no
    # delta (panel routing short-circuits Stage 10a).
    (
        "backend.services.multi_region_layout",
        "USE_MULTI_REGION_LP",
        "CLIPAI_MULTI_REGION_LP",
    ),
    (
        "backend.services.thirds_bias",
        "USE_THIRDS_BIAS",
        "CLIPAI_THIRDS_BIAS",
    ),
]


@pytest.mark.parametrize(
    "module_path,attr,env_var",
    ON_BY_DEFAULT,
    ids=[row[2] for row in ON_BY_DEFAULT],
)
def test_flag_on_by_default_when_env_unset(
    module_path: str,
    attr: str,
    env_var: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The module-level default must be ``True`` when the env var is unset.

    If this fails for a flag, one of the following is true:
      1. Someone flipped the default back to ``"0"`` — update the
         docs/content_type_routing.md flag inventory table AND re-run
         ``python -m backend.scripts.validate_v2_phases --quick`` to
         confirm non-regression before removing the row from this
         ``ON_BY_DEFAULT`` list.
      2. The attribute name or module path drifted — update the row.
      3. The env var name drifted — update the row.
    """
    monkeypatch.delenv(env_var, raising=False)
    mod = importlib.import_module(module_path)
    mod = importlib.reload(mod)
    value = getattr(mod, attr)
    assert value is True, (
        f"{env_var} default regressed to OFF. If this is intentional, "
        f"update ON_BY_DEFAULT in this test AND re-run "
        f"`python -m backend.scripts.validate_v2_phases --quick` "
        f"to confirm no safety-metric regression. Then update "
        f"docs/content_type_routing.md."
    )


def test_use_content_aware_reframe_on_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Umbrella flag ``USE_CONTENT_AWARE_REFRAME`` must stay on.

    Lives in a separate test because (a) the env var matches the
    attribute name, unlike the ``CLIPAI_*`` flags, and (b) there are
    two modules that read it — ``reframe_segmenter.py`` and
    ``content_classifier.py``. Both must agree.
    """
    monkeypatch.delenv("USE_CONTENT_AWARE_REFRAME", raising=False)

    for module_path in (
        "backend.services.reframe_segmenter",
        "backend.services.content_classifier",
    ):
        mod = importlib.import_module(module_path)
        mod = importlib.reload(mod)
        assert getattr(mod, "USE_CONTENT_AWARE_REFRAME") is True, (
            f"USE_CONTENT_AWARE_REFRAME default regressed to OFF in "
            f"{module_path}. This is the umbrella gate for per-content-type "
            f"tuning; flipping it OFF silently disables Phase 3-8 behavior."
        )
