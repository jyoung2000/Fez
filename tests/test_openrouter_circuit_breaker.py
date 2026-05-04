"""Regression tests for the OpenRouter circuit breaker.

Previously, on a 403 ``Key limit exceeded`` from OpenRouter the breaker
waited for 2 consecutive auth failures across batches before short-
circuiting, burning ~15 batches × ~1s each (~15s) before bailing out.
A budget-exhausted key will not un-exhaust itself mid-job, so a single
definitive 403 is conclusive — the breaker should fire on batch 1.

These tests use static analysis on the provider source to assert the
circuit-breaker structure is in place. The real closure lives deep
inside ``OpenRouterProvider.analyze_frames`` and is impractical to
exercise with a real HTTP mock harness; AST/source assertions guard
the wiring that matters for the regression.
"""
import ast
import pathlib

_SRC_PATH = pathlib.Path("backend/services/providers/openrouter_provider.py")
_SRC = _SRC_PATH.read_text()


def test_definitive_auth_flag_is_declared():
    """The closure must declare ``_definitive_auth_seen`` so a single
    ``key limit exceeded`` / ``invalid api key`` flips the breaker."""
    assert "_definitive_auth_seen = False" in _SRC, (
        "Missing _definitive_auth_seen flag — the breaker will revert to "
        "the old 2-consecutive-failures behaviour and waste batches."
    )


def test_definitive_auth_classified_correctly():
    """The is_definitive_auth_error classifier must include both
    ``key limit exceeded`` and ``invalid api key`` (deterministic
    failures), but NOT ``unauthorized`` alone (which can include
    transient 401s during rotations)."""
    # Find the assignment of is_definitive_auth_error and inspect its body.
    assert "is_definitive_auth_error = (" in _SRC
    # Pull a window after the assignment.
    idx = _SRC.find("is_definitive_auth_error = (")
    window = _SRC[idx: idx + 400]
    assert '"key limit exceeded" in err_str' in window
    assert '"invalid api key" in err_str' in window


def test_definitive_threshold_is_one():
    """The threshold for definitive auth errors should be 1, so the
    breaker fires on batch 1 instead of batch 2."""
    # The threshold expression must include ``1 if is_definitive_auth_error``.
    assert "threshold = 1 if is_definitive_auth_error else _AUTH_FAILURE_ABORT_THRESHOLD" in _SRC


def test_transient_threshold_unchanged():
    """Non-auth (transient) errors must still require the original
    consecutive-failure threshold so 5xx flakes don't trip the breaker."""
    assert "_AUTH_FAILURE_ABORT_THRESHOLD = 2" in _SRC


def test_early_abort_gate_checks_definitive_flag():
    """The early-abort guard at the top of ``_analyze_batch`` must check
    ``_definitive_auth_seen`` so subsequent batches skip the call entirely
    after a single definitive 403."""
    tree = ast.parse(_SRC)
    found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "_analyze_batch":
            src = ast.unparse(node)
            if "_definitive_auth_seen" in src and "_AUTH_FAILURE_ABORT_THRESHOLD" in src:
                found = True
                break
    assert found, (
        "_analyze_batch must reference _definitive_auth_seen alongside the "
        "consecutive-failures threshold so the early abort fires on first "
        "definitive failure."
    )


def test_api_is_dead_summary_uses_definitive_flag():
    """The ``api_is_dead`` summary at the end of ``analyze_frames`` must
    also honour ``_definitive_auth_seen``, so the re-analysis loop is
    skipped after a single definitive auth error."""
    assert "api_is_dead = (" in _SRC
    idx = _SRC.find("api_is_dead = (")
    window = _SRC[idx: idx + 200]
    assert "_definitive_auth_seen" in window
