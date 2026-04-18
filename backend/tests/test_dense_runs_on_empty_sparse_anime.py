"""Phase 1 — dense face detection must run on anime jobs even when
the sparse FaceMesh / YuNet pass returned zero faces.

Before the fix, ``backend/services/pipeline.py`` gated the dense block
on ``face_results`` from the sparse pass being non-empty. On pure
anime the sparse pass returns ~0 faces (the human-trained models
reject stylized characters), so the dense block — including its
``is_animated=True`` lbpcascade augmentation path — got skipped and
the L1 solver had no per-second face anchors to work with.

The fix loosens the gate to::

    if settings.SUBJECT_TRACKING_ENABLED and (face_results or _early_anime_hint):

and hoists the ``_early_anime_hint`` assignment ABOVE the gate so it
can participate in the condition.

This test is a static AST check on the function body — running the
real ``_run_analysis_inner`` requires ffmpeg / cv2 / mediapipe / a
real video file, which is overkill for verifying a one-line gate.
"""

from __future__ import annotations

import ast


def _find_function(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    return None


def _read_pipeline_source() -> str:
    with open("backend/services/pipeline.py") as f:
        return f.read()


def test_early_anime_hint_assigned_before_dense_gate():
    """``_early_anime_hint`` must be assigned BEFORE the dense-pass gate
    that uses it, otherwise the gate references an undefined name on
    the first sparse-empty anime job.
    """
    src = _read_pipeline_source()
    tree = ast.parse(src)
    fn = _find_function(tree, "_run_analysis_inner")
    assert fn is not None, "_run_analysis_inner not found in pipeline.py"

    first_assign_line = None
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == "_early_anime_hint":
                    if first_assign_line is None or node.lineno < first_assign_line:
                        first_assign_line = node.lineno

    assert first_assign_line is not None, (
        "_early_anime_hint is never assigned at _run_analysis_inner scope"
    )

    # Find the FIRST Load reference to _early_anime_hint in the same scope —
    # that's the dense-pass gate. It must come AFTER the assignment.
    first_load_line = None
    for node in ast.walk(fn):
        if isinstance(node, ast.Name) and node.id == "_early_anime_hint":
            if isinstance(node.ctx, ast.Load):
                if first_load_line is None or node.lineno < first_load_line:
                    first_load_line = node.lineno

    assert first_load_line is not None, (
        "_early_anime_hint is assigned but never read — gate is missing"
    )
    assert first_load_line >= first_assign_line, (
        f"_early_anime_hint is read at line {first_load_line} BEFORE "
        f"its assignment at line {first_assign_line}"
    )


def test_dense_gate_includes_anime_hint():
    """Locate the dense-face block and verify the gate uses
    ``_early_anime_hint`` (so anime jobs with empty sparse results
    still enter the dense pass).
    """
    src = _read_pipeline_source()
    # Find the comment marker for the dense-face block; the gate
    # condition is the next ``if`` statement after it.
    marker = "# ── Dense face detection (1fps, CPU-only) ──"
    assert marker in src, "dense-face block marker missing from pipeline.py"
    tail = src.split(marker, 1)[1]
    # Take the first ~30 lines after the marker so we don't accidentally
    # match a later branch.
    head = "\n".join(tail.splitlines()[:30])
    assert "_early_anime_hint" in head, (
        "Dense face gate does NOT reference _early_anime_hint — "
        "anime jobs with empty sparse face_results will skip the dense "
        "pass entirely. Loosen the gate to:\n"
        "    if settings.SUBJECT_TRACKING_ENABLED and (face_results or _early_anime_hint):"
    )
    assert "face_results" in head, (
        "Dense face gate must still reference face_results for the "
        "live-action path"
    )


def test_anime_signal_propagates_into_dense_call():
    """``detect_faces_dense(...)`` must be called with
    ``is_animated=_early_anime_hint`` so the cascade augmentation
    actually runs on the dense frames."""
    src = _read_pipeline_source()
    # Quick string check — the test_pipeline_source_width_hoisted.py
    # convention is to do regex-style content checks for hot spots in
    # the pipeline rather than walk the AST for kwargs.
    assert "is_animated=_early_anime_hint" in src, (
        "detect_faces_dense must be called with is_animated=_early_anime_hint"
    )
