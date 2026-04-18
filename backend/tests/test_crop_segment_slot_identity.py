"""Tests for multi-speaker crop-track slot identity (Phase D).

Validates that:
- keyframesToCropSegmentsWithSlots produces one segment per slot change
- Speaker labels use speakerNames when available
- Legacy geometric path still works when no slot data
- extractSlotTimelineFromRenderPlan extracts slots correctly
"""
import pytest
import json
import subprocess
import os
import sys


# We test the JS functions by running them through Node.js
# since the frontend code is JavaScript.


def _node_available():
    """Check if node is available for JS tests."""
    try:
        result = subprocess.run(
            ["node", "--version"],
            capture_output=True, text=True, timeout=5,
        )
        return result.returncode == 0
    except Exception:
        return False


@pytest.mark.skipif(not _node_available(), reason="Node.js not available")
class TestCropSegmentSlotIdentity:
    """E4: Multi-speaker crop-track preserves backend slot identity."""

    def _run_js(self, script):
        """Run a JS script and return the parsed JSON output."""
        # Resolve path to subjectTracking.js
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        js_path = os.path.join(repo_root, "frontend", "src", "utils", "subjectTracking.js")

        # Build a CommonJS-compatible test wrapper
        full_script = f"""
        // Stub console.log to suppress noise
        const _logs = [];
        console.log = (...args) => _logs.push(args.join(' '));

        // We need to handle ES module exports - extract the functions we need
        const fs = require('fs');
        const src = fs.readFileSync('{js_path}', 'utf8');

        // Remove export keywords and import statements for eval
        let cjs = src
            .replace(/^export (function|const|class|let|var|async)/gm, '$1')
            .replace(/^export \\{{ .* \\}};?$/gm, '')
            .replace(/^export default /gm, '')
            .replace(/^import .* from .*/gm, '');

        // Eval the source
        eval(cjs);

        // Run the test
        {script}
        """

        result = subprocess.run(
            ["node", "-e", full_script],
            capture_output=True, text=True, timeout=10,
            cwd=repo_root,
        )
        if result.returncode != 0:
            pytest.fail(f"JS execution failed: {result.stderr}")
        return result.stdout.strip()

    def test_slot_segments_3_speakers(self):
        """3-speaker scene cycling [0, 1, 2, 1, 0, 2] produces 6 segments."""
        output = self._run_js("""
        const keyframes = [
            {t: 0, x: 20}, {t: 5, x: 50}, {t: 10, x: 80},
            {t: 15, x: 50}, {t: 20, x: 20}, {t: 25, x: 80}, {t: 30, x: 80},
        ];
        const slotTimeline = [
            {t: 0, slot: 0}, {t: 5, slot: 1}, {t: 10, slot: 2},
            {t: 15, slot: 1}, {t: 20, slot: 0}, {t: 25, slot: 2},
        ];
        const result = keyframesToCropSegmentsWithSlots(keyframes, 30, slotTimeline, {});
        const labels = result.map(s => s.label);
        process.stdout.write(JSON.stringify({count: result.length, labels}));
        """)
        data = json.loads(output)
        assert data["count"] == 6, f"Expected 6 segments, got {data['count']}"
        assert data["labels"] == [
            "Speaker 1", "Speaker 2", "Speaker 3",
            "Speaker 2", "Speaker 1", "Speaker 3",
        ]

    def test_slot_segments_with_speaker_names(self):
        """speakerNames overrides default labels."""
        output = self._run_js("""
        const keyframes = [
            {t: 0, x: 20}, {t: 5, x: 50}, {t: 10, x: 80},
            {t: 15, x: 50}, {t: 20, x: 20}, {t: 25, x: 80}, {t: 30, x: 80},
        ];
        const slotTimeline = [
            {t: 0, slot: 0}, {t: 5, slot: 1}, {t: 10, slot: 2},
            {t: 15, slot: 1}, {t: 20, slot: 0}, {t: 25, slot: 2},
        ];
        const names = {0: "Alice", 2: "Charlie"};
        const result = keyframesToCropSegmentsWithSlots(keyframes, 30, slotTimeline, names);
        const labels = result.map(s => s.label);
        process.stdout.write(JSON.stringify({labels}));
        """)
        data = json.loads(output)
        assert data["labels"] == [
            "Alice", "Speaker 2", "Charlie",
            "Speaker 2", "Alice", "Charlie",
        ]

    def test_extract_slot_timeline_from_render_plan(self):
        """extractSlotTimelineFromRenderPlan extracts speaker_slot from ops."""
        output = self._run_js("""
        const plan = {
            ops: [
                {start_sec: 0, end_sec: 5, speaker_slot: 0},
                {start_sec: 5, end_sec: 10, speaker_slot: 1},
                {start_sec: 10, end_sec: 15, speaker_slot: null},
                {start_sec: 15, end_sec: 20, speaker_slot: 2},
            ],
        };
        const timeline = extractSlotTimelineFromRenderPlan(plan);
        process.stdout.write(JSON.stringify(timeline));
        """)
        data = json.loads(output)
        # Should have 3 entries (slot null is skipped)
        assert len(data) == 3
        assert data[0] == {"t": 0, "slot": 0}
        assert data[1] == {"t": 5, "slot": 1}
        assert data[2] == {"t": 15, "slot": 2}

    def test_geometric_fallback_when_no_slots(self):
        """keyframesToCropSegments still works for legacy jobs."""
        output = self._run_js("""
        const keyframes = [
            {t: 0, x: 20}, {t: 5, x: 70}, {t: 10, x: 20}, {t: 15, x: 70},
            {t: 20, x: 20},
        ];
        const clusters = [{center: 20, count: 3}, {center: 70, count: 2}];
        const result = keyframesToCropSegments(keyframes, 20, clusters);
        process.stdout.write(JSON.stringify({
            count: result.length,
            hasLabels: result.every(s => s.label && s.label.startsWith('Speaker')),
        }));
        """)
        data = json.loads(output)
        assert data["count"] >= 2, f"Expected at least 2 segments"
        assert data["hasLabels"] is True


class TestPhaseDBbackend:
    """Backend-side Phase D checks."""

    def test_scene_description_has_active_slot(self):
        """SceneDescription model should accept active_slot."""
        from backend.models import SceneDescription

        scene = SceneDescription(
            timestamp=1.0,
            description="test",
            importance_score=5,
            thumbnail_path="",
            active_slot=2,
        )
        assert scene.active_slot == 2

    def test_render_op_has_speaker_fields(self):
        """RenderOp should have speaker_slot and speaker_label."""
        from backend.services.render_plan import RenderOp, RenderOpKind, Rect

        op = RenderOp(
            kind=RenderOpKind.CROP,
            start_sec=0.0,
            end_sec=5.0,
            primary_rect=Rect(x=0.0, y=0.0, w=0.5, h=1.0),
            speaker_slot=1,
            speaker_label="Alice",
        )
        assert op.speaker_slot == 1
        assert op.speaker_label == "Alice"

    def test_render_op_defaults_none(self):
        """speaker_slot/speaker_label default to None (backward compat)."""
        from backend.services.render_plan import RenderOp, RenderOpKind, Rect

        op = RenderOp(
            kind=RenderOpKind.CROP,
            start_sec=0.0,
            end_sec=5.0,
            primary_rect=Rect(x=0.0, y=0.0, w=0.5, h=1.0),
        )
        assert op.speaker_slot is None
        assert op.speaker_label is None

    def test_render_plan_validates_with_speaker_fields(self):
        """RenderPlan with speaker fields should still validate."""
        from backend.services.render_plan import RenderPlan, RenderOp, RenderOpKind, Rect

        plan = RenderPlan(
            source_width=1920,
            source_height=1080,
            target_width=1080,
            target_height=1920,
            total_duration_sec=5.0,
            fps=30.0,
            ops=[RenderOp(
                kind=RenderOpKind.CROP,
                start_sec=0.0,
                end_sec=5.0,
                primary_rect=Rect(x=0.1, y=0.0, w=0.3, h=1.0),
                speaker_slot=0,
                speaker_label="Speaker 1",
            )],
        )
        violations = plan.validate()
        assert len(violations) == 0, f"Validation failed: {violations}"

    def test_render_plan_serializes_speaker_fields(self):
        """speaker_slot/speaker_label should survive JSON round-trip."""
        from backend.services.render_plan import RenderPlan, RenderOp, RenderOpKind, Rect
        import json

        plan = RenderPlan(
            source_width=1920,
            source_height=1080,
            target_width=1080,
            target_height=1920,
            total_duration_sec=5.0,
            fps=30.0,
            ops=[RenderOp(
                kind=RenderOpKind.CROP,
                start_sec=0.0,
                end_sec=5.0,
                primary_rect=Rect(x=0.1, y=0.0, w=0.3, h=1.0),
                speaker_slot=2,
                speaker_label="Charlie",
            )],
        )
        json_str = plan.to_json()
        data = json.loads(json_str)
        assert data["ops"][0]["speaker_slot"] == 2
        assert data["ops"][0]["speaker_label"] == "Charlie"
