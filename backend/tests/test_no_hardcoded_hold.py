"""Tests verifying no hardcoded hold thresholds remain in production code."""

import re
import pytest

from backend.services.content_type_config import TuningConfig, CONTENT_TYPE_CONFIG, ContentType


class TestNoHardcodedHold:
    def test_no_bare_1_5_in_segmenter_files(self):
        """Segmenter files should not contain a bare '= 1.5' hold threshold.
        The only allowed location is TuningConfig._DEFAULTS."""
        import os
        services_dir = os.path.join(os.path.dirname(__file__), '..', 'services')
        services_dir = os.path.normpath(services_dir)

        # Only check files involved in segmentation / hold logic
        target_files = [
            'reframe_segmenter.py',
            'autoflip_segmenter.py',
            'intent_tracker.py',
        ]

        violations = []
        for fname in target_files:
            fpath = os.path.join(services_dir, fname)
            if not os.path.exists(fpath):
                continue
            with open(fpath) as f:
                for lineno, line in enumerate(f, 1):
                    stripped = line.strip()
                    if stripped.startswith('#'):
                        continue
                    if re.search(r'(?:else|=)\s*1\.5\b', line):
                        # Allow TuningConfig _DEFAULTS and fallback references
                        if 'intent_min_hold_fallback' in line:
                            continue
                        violations.append(f"{fname}:{lineno}: {stripped}")

        assert violations == [], (
            f"Found hardcoded 1.5 hold threshold in segmenter code:\n" +
            "\n".join(violations)
        )

    def test_no_bare_min_hold_assignments_in_segmenters(self):
        """Segmenter files should not contain bare numeric min_hold assignments.
        All min_hold values should flow from LocalPacingEstimator or TuningConfig."""
        import os
        services_dir = os.path.join(os.path.dirname(__file__), '..', 'services')
        services_dir = os.path.normpath(services_dir)

        target_files = [
            'reframe_segmenter.py',
            'autoflip_segmenter.py',
            'intent_tracker.py',
        ]

        allowed_patterns = [
            'min_hold_at(',           # LocalPacingEstimator method call
            'MIN_HOLD_SECONDS',       # Module-level default (legacy path)
            'intent_min_hold_fallback',  # TuningConfig field
            'local_min_hold',         # Variable (derived from pacing or tuning)
            '_min_hold',              # Internal variable
            '_log(',                  # Log messages
        ]

        violations = []
        for fname in target_files:
            fpath = os.path.join(services_dir, fname)
            if not os.path.exists(fpath):
                continue
            with open(fpath) as f:
                for lineno, line in enumerate(f, 1):
                    stripped = line.strip()
                    if stripped.startswith('#') or stripped.startswith('"') or stripped.startswith("'"):
                        continue
                    # Look for bare numeric min_hold assignments like `min_hold = 1.5`
                    if re.search(r'min_hold\s*=\s*\d+\.?\d*\b', stripped):
                        if any(p in stripped for p in allowed_patterns):
                            continue
                        violations.append(f"{fname}:{lineno}: {stripped}")

        assert violations == [], (
            f"Found bare min_hold numeric assignments:\n" +
            "\n".join(violations)
        )

    def test_tuning_config_has_fallback_field(self):
        """TuningConfig should provide intent_min_hold_fallback with default 1.5."""
        tuning = TuningConfig({})
        assert tuning.intent_min_hold_fallback == 1.5
