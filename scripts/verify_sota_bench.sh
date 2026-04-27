#!/usr/bin/env bash
# Verify the 2026 SOTA reframing validation works end-to-end BEFORE
# pushing or trusting the GUI button.
#
# Run this in TWO contexts:
#   1. Locally (in the repo root, on your dev box):
#        bash scripts/verify_sota_bench.sh local
#      Validates that the QA harness passes against the current
#      working tree.
#
#   2. Inside the running container (after `docker compose up -d`):
#        docker compose exec backend bash scripts/verify_sota_bench.sh container
#      Validates that the harness is baked into the production image
#      AND that the diagnostics endpoint actually works.
#
# Exits 0 if every check passes, non-zero on the first failure with a
# clear message about what to fix.

set -euo pipefail

mode="${1:-local}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

ok()   { printf "\033[32m  PASS\033[0m  %s\n" "$1"; }
warn() { printf "\033[33m  WARN\033[0m  %s\n" "$1"; }
fail() { printf "\033[31m  FAIL\033[0m  %s\n" "$1"; exit 1; }

echo "================================================================"
echo "  SOTA Reframing Validation - Pre-Push Verification (${mode})"
echo "================================================================"
echo

# ── Check 1: harness files present ──────────────────────────────────
echo "[1/5] Harness file layout"
[[ -f tests/qa/run_all_phases.py ]] || fail "tests/qa/run_all_phases.py missing"
ok "tests/qa/run_all_phases.py present"
[[ -f tests/qa/__init__.py ]] || fail "tests/qa/__init__.py missing"
ok "tests/qa/__init__.py present"
n_suites=$(ls tests/qa/test_phase_*.py 2>/dev/null | wc -l)
[[ "$n_suites" -ge 5 ]] || fail "expected >=5 test_phase_*.py suites, got $n_suites"
ok "${n_suites} phase suites present"

# ── Check 2: Python module import ────────────────────────────────────
echo
echo "[2/5] Module-import smoke test"
python -c "import importlib; importlib.import_module('tests.qa.run_all_phases'); print('imported')" \
  > /dev/null 2>&1 \
  || fail "python -c 'import tests.qa.run_all_phases' failed (CWD must be repo root; PYTHONPATH must include it)"
ok "tests.qa.run_all_phases imports clean"

# ── Check 3: harness runs to completion ──────────────────────────────
echo
echo "[3/5] Harness runs to completion"
log_file="$(mktemp)"
trap 'rm -f "$log_file"' EXIT
if python -u -m tests.qa.run_all_phases > "$log_file" 2>&1; then
  ok "harness exit 0"
else
  echo "----- harness output -----"
  cat "$log_file"
  echo "--------------------------"
  fail "harness exited non-zero"
fi
if grep -q "OVERALL: PASS" "$log_file"; then
  ok "harness reports OVERALL PASS"
else
  echo "----- harness output -----"
  cat "$log_file"
  echo "--------------------------"
  fail "harness output missing 'OVERALL: PASS'"
fi
n_pass=$(grep -c "\[PASS\]" "$log_file" || true)
[[ "$n_pass" -ge 5 ]] || fail "expected >=5 [PASS] lines, got $n_pass"
ok "${n_pass} phase suites green"

# ── Check 4: diagnostics endpoint (container only) ───────────────────
if [[ "$mode" == "container" ]]; then
  echo
  echo "[4/5] /api/diagnostics/sota-bench-status reachable"
  if command -v curl >/dev/null 2>&1; then
    status_json=$(curl -fsS http://127.0.0.1:1353/api/diagnostics/sota-bench-status \
      || fail "curl to /api/diagnostics/sota-bench-status failed")
    echo "$status_json" | python -m json.tool > /dev/null \
      || fail "endpoint returned non-JSON: $status_json"
    ok=$(echo "$status_json" | python -c "import json,sys; print(json.load(sys.stdin).get('ok'))")
    if [[ "$ok" == "True" ]]; then
      ok "endpoint reports ok=true"
    else
      echo "----- endpoint payload -----"
      echo "$status_json" | python -m json.tool
      echo "----------------------------"
      fail "endpoint reports ok=false"
    fi
  else
    warn "curl not available in container; skipping endpoint check"
  fi
else
  echo
  echo "[4/5] (skipped - 'container' mode only)"
fi

# ── Check 5: synchronous bench-qa endpoint (container only) ──────────
if [[ "$mode" == "container" ]]; then
  echo
  echo "[5/5] /api/diagnostics/sota-bench-qa runs"
  if command -v curl >/dev/null 2>&1; then
    qa_json=$(curl -fsS -X POST -H 'Content-Type: application/json' \
      -d '{}' http://127.0.0.1:1353/api/diagnostics/sota-bench-qa \
      || fail "curl POST to /api/diagnostics/sota-bench-qa failed")
    qa_ok=$(echo "$qa_json" | python -c "import json,sys; print(json.load(sys.stdin).get('ok'))")
    if [[ "$qa_ok" == "True" ]]; then
      ok "synchronous QA endpoint returns ok=true"
    else
      echo "----- /sota-bench-qa payload (truncated) -----"
      echo "$qa_json" | python -m json.tool 2>/dev/null | head -40 || echo "$qa_json" | head -40
      echo "-----------------------------------------------"
      fail "/sota-bench-qa returned ok=false"
    fi
  fi
else
  echo
  echo "[5/5] (skipped - 'container' mode only)"
fi

echo
echo "================================================================"
echo "  ALL CHECKS PASSED - safe to click the GUI button."
echo "================================================================"
