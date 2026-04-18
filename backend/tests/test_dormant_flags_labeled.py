"""Regression guard: dormant v2 flags are still dormant.

Three anime-specific modules ship in ``backend/services/`` but have
zero production call sites as of Week 1 — only the defining file,
the per-module unit tests in ``backend/tests/test_phase6_*.py``, and
``backend/scripts/validate_v2_phases.py``'s env-var setup reference
them. Their ``CLIPAI_*`` flags are gates on code paths that never run.

Week 2 plans to wire all three:
  - ``detect_anime_faces`` → ``face_detector.py`` dense pipeline
  - ``detect_anime_shots`` → ``shot_detector.py`` cut detection
  - ``cluster_anime_characters`` / ``anime_character_clustering``
    → ``face_registry.py`` cross-cut re-ID

When Week 2 lands, this test starts failing for the module that got
wired — that is the correct signal to:

  1. Flip the module's flag default from ``"0"`` to ``"1"`` in the
     defining file.
  2. Add a row to ``backend/tests/test_flag_defaults_stable.py``'s
     ``ON_BY_DEFAULT`` list.
  3. Update the ``CLIPAI_ANIME_*`` rows in
     ``docs/content_type_routing.md`` (Code default → ON, Effective? →
     Yes (live)).
  4. Remove the corresponding entry from ``DORMANT_MODULES`` in this
     test.
  5. Re-run ``python -m backend.scripts.validate_v2_phases --quick``
     to confirm non-regression.

The test uses an AST-level walk instead of a plain ``grep`` so
docstrings and comments that mention the symbol by name don't
produce false positives (the defining modules' docstrings reference
their own public symbols).
"""
from __future__ import annotations

import ast
import pathlib

# Repo root = two parents up from this file
# (backend/tests/test_dormant_flags_labeled.py → backend/ → repo)
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent


# module_stem → set of public symbol names that would be called
# when the module is wired into the pipeline. If any of these
# appear in another backend/ file (excluding tests + scripts), the
# module is no longer dormant.
DORMANT_MODULES: dict[str, set[str]] = {
    # All three anime modules wired in Week 2:
    #   Part A — anime_face_detector:
    #     backend/services/face_detector.py::_augment_dense_with_anime
    #   Part B — anime_shot_detector:
    #     backend/services/pipeline.py (post-classify cut merge) +
    #     backend/services/layout_engine.py (Shot-object split)
    #   Part C — anime_character_clustering:
    #     backend/services/pipeline.py (post-face-registry re-ID)
    # All three removed from this guard on 2026-04-14; flags flipped
    # to "1" in the same change. The guard is kept in place (even
    # though empty) so future dormant modules have a landing spot.
}


def _find_callers(symbol: str, skip_files: set[pathlib.Path]) -> list[pathlib.Path]:
    """Return the backend/ files that reference ``symbol`` as a name,
    attribute, or import.

    Excluded:
      - ``skip_files`` (the defining module(s))
      - Anything under ``backend/tests/`` (per-module unit tests
        are allowed to reference the dormant symbols directly)
      - Anything under ``backend/scripts/`` (``validate_v2_phases.py``
        sets env vars for these modules via string literals, never
        imports them)

    The search is AST-based: comments and docstrings that mention
    the symbol name do NOT count. Only real Python references do.
    """
    hits: list[pathlib.Path] = []
    backend = REPO_ROOT / "backend"
    if not backend.exists():
        return hits

    for py in backend.rglob("*.py"):
        if py in skip_files:
            continue
        # Skip tests — they're allowed to exercise the dormant modules.
        if "tests" in py.parts:
            continue
        # Skip scripts — validate_v2_phases.py references flag names
        # via string literals in its _BASELINE env dict, but never
        # imports the modules themselves.
        if "scripts" in py.parts:
            continue

        try:
            src = py.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        # Cheap pre-filter: if the literal symbol name isn't in the
        # file at all, skip the AST parse.
        if symbol not in src:
            continue

        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            # Name reference: `x = detect_anime_faces(...)`
            if isinstance(node, ast.Name) and node.id == symbol:
                hits.append(py)
                break
            # Attribute access: `mod.detect_anime_faces(...)`
            if isinstance(node, ast.Attribute) and node.attr == symbol:
                hits.append(py)
                break
            # Import: `from backend.services.anime_face_detector import detect_anime_faces`
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name == symbol:
                        hits.append(py)
                        break
                else:
                    continue
                break
    return hits


def test_dormant_modules_have_no_call_sites() -> None:
    """Every symbol in ``DORMANT_MODULES`` has zero prod call sites."""
    services_dir = REPO_ROOT / "backend" / "services"
    failures: list[str] = []

    for module_stem, symbols in DORMANT_MODULES.items():
        defining = services_dir / f"{module_stem}.py"
        skip = {defining}
        for sym in symbols:
            callers = _find_callers(sym, skip_files=skip)
            if callers:
                failures.append(
                    f"{sym} (from {module_stem}) has callers: "
                    f"{[str(p.relative_to(REPO_ROOT)) for p in callers]}"
                )

    assert not failures, (
        "Dormant anime modules are no longer dormant:\n  "
        + "\n  ".join(failures)
        + "\n\n"
        "If Week 2 wired one of them, flip the module's flag default "
        "from '0' to '1', add a row to test_flag_defaults_stable.py's "
        "ON_BY_DEFAULT list, update docs/content_type_routing.md to "
        "mark it 'Yes (live)', and remove the corresponding entry "
        "from DORMANT_MODULES in this test. Then re-run "
        "validate_v2_phases --quick to confirm non-regression."
    )


def test_dormant_module_flags_default_off() -> None:
    """Double-check that the dormant flags are still OFF by default.

    If someone flips one to ON without wiring the module, the
    ``CLIPAI_*`` flag becomes a lie — env=1 does nothing because no
    code path reads it. Catch that here so the error is explicit.
    """
    import importlib
    cases: list[tuple[str, str, str]] = [
        # All three anime flags moved to test_flag_defaults_stable.py
        # in Week 2 Parts A + B + C. If a new dormant module appears
        # in the future, add its (module_path, attr, env_var) tuple
        # here.
    ]

    import os
    for module_path, attr, env_var in cases:
        # Clear any ambient env var so we measure the pure default.
        prior = os.environ.pop(env_var, None)
        try:
            mod = importlib.import_module(module_path)
            mod = importlib.reload(mod)
            value = getattr(mod, attr)
            assert value is False, (
                f"{env_var} default is ON but {module_path} is still "
                f"dormant (per test_dormant_modules_have_no_call_sites). "
                f"A dormant flag flipped ON is a lie — env=1 does "
                f"nothing because the module has no call sites. "
                f"Either wire the module (Week 2 plan) or revert the "
                f"flag default to '0'."
            )
        finally:
            if prior is not None:
                os.environ[env_var] = prior
