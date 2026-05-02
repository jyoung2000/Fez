"""Static-analysis test that ``transcribe_audio_subprocess`` is never
re-bound inside ``_branch_transcription``.

The bug we're guarding against is subtle: a local import of a name that
is also imported at module level silently turns every earlier read of
that name into ``UnboundLocalError`` for the whole function body. AST
inspection catches this before the next regression slips in.
"""
import ast
import pathlib


def _branch_transcription_node():
    src = pathlib.Path("backend/services/pipeline.py").read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "_branch_transcription":
            return node
    raise AssertionError("_branch_transcription not found in pipeline.py")


def test_transcribe_audio_subprocess_not_locally_rebound():
    fn = _branch_transcription_node()
    locally_imported_names: list[str] = []
    for sub in ast.walk(fn):
        if isinstance(sub, ast.ImportFrom):
            for alias in sub.names:
                locally_imported_names.append(alias.asname or alias.name)
    assert "transcribe_audio_subprocess" not in locally_imported_names, (
        "transcribe_audio_subprocess is imported at module level "
        "(pipeline.py line 30); a local re-bind inside "
        "_branch_transcription causes UnboundLocalError on the first "
        "read of the name. Use the module-level import instead."
    )
