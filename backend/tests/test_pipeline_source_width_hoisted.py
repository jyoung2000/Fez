"""v4.1 Fix 1 regression: pipeline source_width must be in scope for both
autoflip and reframe_segmenter post-processing loops.

Before the hotfix, `source_width` was used on lines 2668 and 2764 of
pipeline.py without being defined at the function scope — only inside
build_reframe_segments as a kwarg. When the reframe_segmenter returned
segments, the pipeline's subject_x → 0-100 conversion raised NameError,
which was caught by the broad except and logged as:

    ReframeSegmenter failed (falling back to per-second):
    name 'source_width' is not defined

...after 9.5 minutes of L1 solving that was then THROWN AWAY.

This test reads pipeline.py as source and asserts that source_width is
bound at the function scope of `_run_analysis_inner` before any of the
bare references use it. We don't run the full pipeline (too heavy for
a unit test); static AST inspection is enough to lock in the fix.
"""

import ast


def _find_function(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    return None


def test_source_width_hoisted_to_pipeline_function_scope():
    with open("backend/services/pipeline.py") as f:
        src = f.read()
    tree = ast.parse(src)

    fn = _find_function(tree, "_run_analysis_inner")
    assert fn is not None, "_run_analysis_inner not found in pipeline.py"

    # Walk statements in order: find the first assignment to `source_width`
    # and record its line number. Then find all bare Name("source_width")
    # loads and assert each appears after the assignment.
    first_assign_line = None
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == "source_width":
                    if first_assign_line is None or node.lineno < first_assign_line:
                        first_assign_line = node.lineno

    assert first_assign_line is not None, (
        "source_width is never assigned at _run_analysis_inner scope — "
        "the v4.1 hotfix regressed. Hoist `source_width = metadata.get("
        "'width', 1920)` near the top of the function."
    )

    # Now gather all bare references (Name loads) to source_width and
    # assert each is after the assignment. Keyword arguments like
    # `source_width=metadata.get(...)` are Keyword nodes, not Name loads,
    # so they don't appear here.
    bad_refs = []
    for node in ast.walk(fn):
        if isinstance(node, ast.Name) and node.id == "source_width":
            if isinstance(node.ctx, ast.Load) and node.lineno < first_assign_line:
                bad_refs.append(node.lineno)

    assert not bad_refs, (
        f"bare source_width read before assignment at lines {bad_refs} "
        f"(assignment at line {first_assign_line})"
    )


def test_source_height_also_hoisted():
    """Same check for source_height so the AUTOFLIP aspect ratio math
    can read it without NameError when USE_AUTOFLIP_REFRAME=true."""
    with open("backend/services/pipeline.py") as f:
        src = f.read()
    tree = ast.parse(src)

    fn = _find_function(tree, "_run_analysis_inner")
    assert fn is not None

    first_assign_line = None
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == "source_height":
                    if first_assign_line is None or node.lineno < first_assign_line:
                        first_assign_line = node.lineno

    assert first_assign_line is not None, (
        "source_height is never assigned at _run_analysis_inner scope"
    )
