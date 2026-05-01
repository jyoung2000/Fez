"""TACT pipeline wire-up integration tests.

These tests verify the structural wiring of TACT phases 2-5 inside
the pipeline. They drive the ledger + ladder + reconciler + (when
enabled) translation track against the same code paths the
production pipeline uses, but with mocked subprocess rungs so the
tests can run in any environment without real Whisper / wav2vec2 /
PANNs / NeMo.

The single acceptance criterion from the wire-up prompt §4 is
``coverage_report["coverage_ratio"] == 1.0`` on a real video. This
file's tests exercise the same invariant on synthetic fixtures with
mocked rungs — the actual real-video smoke test is documented in
``backend/tests/fixtures/coverage/README.md`` as the user's homelab
acceptance step.

What's verified here:
  * Ledger built from primary Whisper output reaches 1.0 coverage
    after the ladder runs.
  * Quarantined hallucinations are claimed correctly and
    re-presented to the ladder.
  * Reconciler stats are readable via the module-level accessor.
  * Translation track produces a paired report when enabled.
  * Per-fixture: every committed fixture file ends with
    coverage_ratio == 1.0 against a no-op-rung-1 + Rung-6 ladder.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.models import TranscriptSegment, WordTimestamp
from backend.services.escalation_ladder import (
    EscalationContext,
    LedgerSpan,
    RungResult,
    _rung_mark_unintelligible,
    escalate_uncovered_intervals_sync,
)
from backend.services.transcription_ledger import CoverageLedger
from backend.services.transcription_reconciler import (
    get_last_reconciliation_stats,
    reconcile_n_passes,
)


FIXTURES_DIR = Path(__file__).parent / "fixtures" / "coverage"


def _segments(*ranges_with_text: tuple[float, float, str]) -> list[TranscriptSegment]:
    """Build TranscriptSegment list from (start, end, text) tuples."""
    return [
        TranscriptSegment(
            start=s, end=e, text=t, speaker="Speaker 1",
            words=[WordTimestamp(start=s, end=e, word=t)],
            confidence=0.9,
        )
        for s, e, t in ranges_with_text
    ]


def run_tact_harness(
    audio_path: str,
    audio_duration_sec: float,
    fake_whisper_segments: list[TranscriptSegment],
    *,
    fake_quarantined: list[dict] = None,
    fake_offset_segments: list[TranscriptSegment] = None,
    fake_parakeet_segments: list[TranscriptSegment] = None,
    rungs=None,
    language: str = "en",
):
    """Drive the TACT block's logic against synthetic input.

    Mirrors the production pipeline's flow:
      1. Optionally reconcile primary + offset (+ parakeet) passes.
      2. Build the ledger from the reconciled segments.
      3. Claim quarantined regions.
      4. Run the ladder.
      5. Build coverage_report.

    Returns ``(final_segments, coverage_report, ladder_stats)``.
    """
    audio_duration_ms = int(audio_duration_sec * 1000)
    primary = list(fake_whisper_segments)

    # Reconciliation step (optional).
    passes = [("primary", primary)]
    chunk_grid_offsets = {"primary": 0.0}
    if fake_offset_segments is not None:
        passes.append(("offset_15s", list(fake_offset_segments)))
        chunk_grid_offsets["offset_15s"] = 15.0
    if fake_parakeet_segments is not None:
        passes.append(("parakeet", list(fake_parakeet_segments)))
        chunk_grid_offsets["parakeet"] = 0.0
    if len(passes) >= 2:
        reconciled, _ = reconcile_n_passes(
            passes, chunk_grid_offsets=chunk_grid_offsets,
        )
        primary = reconciled

    # Build the ledger.
    ledger = CoverageLedger(audio_duration_ms, bin_ms=20)
    ledger.from_segments(primary, source_pass="whisper_main")
    if fake_quarantined:
        ledger.from_segments(
            fake_quarantined,
            source_pass="quarantined_hallucinations",
            status="quarantined",
            flag_key="quarantine_reason",
        )

    ctx = EscalationContext(
        audio_path=audio_path,
        audio_duration_ms=audio_duration_ms,
        language=language,
        task="transcribe",
    )
    if rungs is None:
        # Default: only Rung 6 (terminator). Real Rungs 1-5 need
        # subprocess + models that aren't available in unit-test env.
        rungs = [("rung_6_unintelligible", _rung_mark_unintelligible)]
    stats = escalate_uncovered_intervals_sync(ledger, ctx, rungs=rungs)

    coverage_report = ledger.to_report_dict()
    coverage_report["ladder_stats"] = stats.as_dict()
    recon = get_last_reconciliation_stats()
    if recon is not None:
        coverage_report["reconciliation"] = recon.as_dict()
    return ledger.to_segments(), coverage_report, stats


# ──────────────────────────────────────────────────────────────────────────
# The acceptance criterion, applied to every fixture
# ──────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("fixture", [
    "silent_mouthing",
    "applause_only",
    "music_only",
    "quiet_speech_60db",
    "sentence_across_chunk_boundary",
    "speech_over_music",
])
def test_coverage_invariant_per_fixture(fixture):
    """For every fixture, after the harness runs, coverage_ratio == 1.0.

    The fixtures are synthetic waveforms. Real Whisper would produce
    very different segment lists; here we pass an empty primary
    (simulating "Whisper found nothing") so the ladder is responsible
    for covering the whole duration via Rung 6. This proves the
    coverage invariant: even when every primary pass + every higher
    rung produces nothing, Rung 6 catches every interval and
    coverage_ratio reaches exactly 1.000.
    """
    audio_path = str(FIXTURES_DIR / f"{fixture}.wav")
    truth = json.loads((FIXTURES_DIR / f"{fixture}.ground_truth.json").read_text())
    duration = float(truth["duration_sec"])

    _segs, coverage_report, stats = run_tact_harness(
        audio_path, duration,
        fake_whisper_segments=[],  # simulate Whisper produced nothing
    )
    # The headline acceptance criterion.
    assert coverage_report["coverage_ratio"] >= truth["min_overall_coverage_ratio"]
    assert coverage_report["status_distribution_ms"]["uncovered"] == 0


def test_status_distribution_sums_to_audio_duration():
    """coverage_report['status_distribution_ms'] sums to
    audio_duration_ms exactly. Acceptance criterion §4 part 2."""
    _, coverage_report, _ = run_tact_harness(
        "/dev/null", 10.0, fake_whisper_segments=[],
    )
    audio_duration_ms = coverage_report["audio_duration_ms"]
    total = sum(coverage_report["status_distribution_ms"].values())
    assert total == audio_duration_ms


def test_ladder_stats_show_non_zero_rung_activity():
    """coverage_report['ladder_stats'] shows non-zero rung activity.
    Acceptance criterion §4 part 3."""
    _, coverage_report, _ = run_tact_harness(
        "/dev/null", 10.0, fake_whisper_segments=[],
    )
    ladder_stats = coverage_report["ladder_stats"]
    # With an empty primary pass, the entire duration is one
    # uncovered interval — Rung 6 catches it.
    assert ladder_stats["intervals_resolved"] >= 1
    assert ladder_stats["rung_resolutions"].get("rung_6_unintelligible", 0) >= 1


# ──────────────────────────────────────────────────────────────────────────
# Quarantine round-trip
# ──────────────────────────────────────────────────────────────────────────

def test_quarantined_segments_re_presented_to_ladder():
    """Quarantined regions appear as ladder intervals (the ladder
    iterates over uncovered ∪ quarantined). Production deployment
    relies on the higher rungs (especially Rungs 5/6 with real
    PANNs / wav2vec2) to replace quarantined claims with correct
    event tags or [unintelligible]; here we just verify the
    visitor pattern sees the quarantined interval."""
    audio_path = str(FIXTURES_DIR / "music_only.wav")
    quarantined = [
        {"start": 2.0, "end": 4.0, "text": "thank you for watching",
         "quarantine_reason": "boilerplate",
         # Real Whisper hallucinations carry avg_logprob; mirror that
         # so the quarantined claim's confidence isn't an artificial 1.0.
         "avg_logprob": -2.5,
         "confidence": 0.05},
    ]
    primary = _segments((4.0, 6.0, "kept_speech"))
    intervals_seen = []

    def replacer(audio_path, start_ms, end_ms, ctx):
        """Pretends to be Rung 5 — replaces the quarantined region
        with an event tag at higher confidence than the quarantined
        incumbent."""
        intervals_seen.append((start_ms, end_ms))
        if 2_000 <= start_ms < 4_000:
            return RungResult(
                rung_name="replacer",
                spans=[LedgerSpan(
                    start_ms=start_ms, end_ms=end_ms,
                    status="covered_event", content="[music]",
                    content_type="event", source_pass="replacer",
                    confidence=0.7,
                )],
            )
        return RungResult(rung_name="replacer")

    rungs = [
        ("replacer", replacer),
        ("rung_6_unintelligible", _rung_mark_unintelligible),
    ]
    _, coverage_report, _ = run_tact_harness(
        audio_path, 10.0,
        fake_whisper_segments=primary,
        fake_quarantined=quarantined,
        rungs=rungs,
    )
    # Visitor sees three intervals: [0, 2000], [2000, 4000]
    # (the quarantined region), and [6000, 10000].
    assert (2_000, 4_000) in intervals_seen
    # After the replacer + Rung 6, no quarantined or uncovered ms.
    dist = coverage_report["status_distribution_ms"]
    assert dist["uncovered"] == 0
    assert dist["quarantined"] == 0
    assert coverage_report["coverage_ratio"] == 1.0


# ──────────────────────────────────────────────────────────────────────────
# Reconciliation stats accessor wired through harness
# ──────────────────────────────────────────────────────────────────────────

def test_harness_records_reconciliation_stats():
    """When ≥ 2 passes are reconciled, the harness reads the
    module-level accessor so coverage_report carries the
    reconciliation block."""
    primary = _segments((0.0, 1.0, "alpha"), (2.0, 3.0, "beta"))
    offset = _segments((0.0, 1.0, "alpha"), (2.0, 3.0, "beta"))
    _, coverage_report, _ = run_tact_harness(
        "/dev/null", 10.0,
        fake_whisper_segments=primary,
        fake_offset_segments=offset,
    )
    # reconciliation block is present and shows agreed words.
    assert "reconciliation" in coverage_report
    assert coverage_report["reconciliation"]["total_words"] >= 2
    assert coverage_report["reconciliation"]["agreed"] >= 2


def test_three_way_reconciliation_with_parakeet():
    """Adding a Parakeet pass produces a 3-pass reconciliation that
    still hits coverage 1.0."""
    primary = _segments((0.0, 1.0, "hello"))
    offset = _segments((0.0, 1.0, "hello"))
    parakeet = _segments((0.0, 1.0, "hello"))
    _, coverage_report, _ = run_tact_harness(
        "/dev/null", 10.0,
        fake_whisper_segments=primary,
        fake_offset_segments=offset,
        fake_parakeet_segments=parakeet,
    )
    assert coverage_report["coverage_ratio"] == 1.0
    assert coverage_report["reconciliation"]["passes"] == [
        "primary", "offset_15s", "parakeet",
    ]


# ──────────────────────────────────────────────────────────────────────────
# Translation track (optional path)
# ──────────────────────────────────────────────────────────────────────────

def test_translation_track_produces_paired_report():
    """When translation runs, ledger.to_paired_report_dict produces a
    superset of the source report with a `translation` block."""
    from backend.services.translation_track import (
        TranslationCandidate,
        _BACKENDS,
        register_backend,
        run_translation_track,
    )

    @register_backend("test_translate")
    def _t(src, tgt):
        return TranslationCandidate(
            backend_name="test_translate",
            text=f"FR:{src}",
            confidence=0.9,
        )
    try:
        # Build a covered ledger first, then translate.
        ledger = CoverageLedger(audio_duration_ms=10_000, bin_ms=20)
        ledger.from_segments(
            _segments((0.0, 5.0, "hello world")),
            source_pass="whisper_main",
        )
        # Cover the rest with a stub event so coverage hits 1.0.
        ledger.claim(LedgerSpan(
            start_ms=5_000, end_ms=10_000, status="covered_silence",
            content="[silence]", content_type="silence",
            source_pass="rung_5", confidence=0.8,
        ))
        translation_ledger, t_stats = run_translation_track(
            ledger, target_language="fr", backend="test_translate",
        )
        report = ledger.to_paired_report_dict(translation_ledger)
        assert "source" in report
        assert "translation" in report
        assert report["translation"]["target_language"] == "fr"
        assert report["translation"]["coverage_ratio"] > 0.0
        assert report["translation"]["untranslated_ratio"] == 0.0
        assert t_stats.spans_translated >= 1
    finally:
        _BACKENDS.pop("test_translate", None)


# ──────────────────────────────────────────────────────────────────────────
# Fixture file integrity
# ──────────────────────────────────────────────────────────────────────────

def test_every_fixture_has_a_ground_truth():
    wavs = sorted(FIXTURES_DIR.glob("*.wav"))
    assert wavs, "no fixture wavs committed"
    for wav in wavs:
        gt = wav.with_suffix(".ground_truth.json")
        assert gt.exists(), f"missing ground truth for {wav.name}"
        truth = json.loads(gt.read_text())
        assert "duration_sec" in truth
        assert "min_overall_coverage_ratio" in truth


# ──────────────────────────────────────────────────────────────────────────
# No-regression smoke: existing pre-TACT tests still importable
# ──────────────────────────────────────────────────────────────────────────

def test_no_regression_in_prior_phase_tests():
    """Smoke import — confirms no circular import surfaced from the
    pipeline rewire."""
    import importlib
    for module_name in [
        "backend.tests.test_transcription_gap_filler",
        "backend.tests.test_transcription_ledger",
        "backend.tests.test_hallucination_quarantine",
        "backend.tests.test_escalation_ladder",
        "backend.tests.test_transcription_reconciler",
        "backend.tests.test_translation_track",
        "backend.tests.test_parakeet_transcriber",
        "backend.tests.test_orchestrator_neighbor_text",
        "backend.tests.test_rung_2_alt_checkpoint",
        "backend.tests.test_rung_4_forced_alignment",
        "backend.tests.test_rung_5_event_classifier",
    ]:
        importlib.import_module(module_name)
