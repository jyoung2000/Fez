"""ClipAI 2026 SOTA reframing — phase QA suites.

Each ``test_phase_<X>_<name>.py`` module is the local validation a
Claude Code Opus 4.7 worker runs to gate a phase BEFORE pushing.

These suites are deliberately mock-heavy so they pass on a CPU-only
sandbox (``torch`` / ``sam2`` / ``cotracker`` / ``CLIP`` not installed).
The real-content bench (``backend.scripts.compare_autoflip_vs_clipai``)
is the homelab-side gate and runs separately — see
``tests/qa/MASTER_QA_HARNESS.md``.
"""
