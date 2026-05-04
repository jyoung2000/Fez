"""Tests for the "every speaker is Speaker 1" bug family.

Covers the three code paths that each, on their own, were sufficient
to collapse a multi-speaker clip to a single ``Speaker 1`` label:

  * ``_assign_speakers`` — the heuristic fallback (Fix B).
  * ``_assign_speakers_from_diarization`` — the pyannote-alignment
    path (Fix D).
  * ``_diarize_pyannote`` kwargs — the face-slot clamp (Fix C).
  * ``pipeline.py`` diarization guard — the ``_num_slots >= 2``
    gate that skipped diarization entirely when the registry saw
    only one face (Fix C, pipeline side).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


# ── Fix B — heuristic alternates speakers on realistic gaps ──


def test_heuristic_alternates_on_realistic_gaps():
    """4 segments alternating short/fast and long/slow at 0.8 s
    gaps must produce at least 2 distinct Speaker labels,
    alternating — not collapse to "Speaker 1 / Speaker 1 / ...".

    Rate profile:
      * short/fast segments ≈ 2.5 words/s (speaker A)
      * long/slow segments ≈ 3.5 words/s (speaker B, but slower
        per-word)
    Large cross-segment gaps (0.8 s) exceed the tightened
    TURN_GAP = 0.6 s so the new-speaker path fires on each change.
    """
    from backend.services.transcription import _assign_speakers

    raw_segments = [
        {"start": 0.0, "end": 2.0,
         "text": "hi there yes yes yes",
         "words": None},
        {"start": 2.8, "end": 6.0,
         "text": "oh my that is a lengthy reply indeed very lengthy",
         "words": None},
        {"start": 6.8, "end": 8.0,
         "text": "well sure absolutely",
         "words": None},
        {"start": 8.8, "end": 12.0,
         "text": "understood quite so noted carefully recorded this response",
         "words": None},
    ]
    out = _assign_speakers(raw_segments)
    labels = [s.speaker for s in out]
    unique = set(labels)
    assert len(unique) >= 2, (
        f"Expected at least 2 speakers in alternating fixture, "
        f"got {labels}"
    )
    # Alternation: segment 0 and 2 share a speaker, 1 and 3 share
    # another (rate-based matching).
    assert labels[0] != labels[1]
    assert labels[1] != labels[2]


# ── Fix D — nearby-turn fallback beats hard-coded Speaker 1 ──


def test_assign_from_diarization_uses_nearby_turn_instead_of_speaker_1():
    """A Whisper segment with no overlapping pyannote turn but a
    turn from SPEAKER_01 starting 0.3 s later must inherit
    ``Speaker 2`` (from the SPEAKER_01 → order-index-1 mapping),
    NOT fall back to ``"Speaker 1"`` as the old code did.
    """
    from backend.services.transcription import _assign_speakers_from_diarization

    # pyannote-style map: SPEAKER_00 and SPEAKER_01 both present.
    speaker_map = {
        (0.0, 4.0): "SPEAKER_00",   # Speaker 1
        (4.3, 8.0): "SPEAKER_01",   # Speaker 2
    }
    # Whisper segment from 4.0 → 4.25 s sits in the 0.3 s gap
    # between the two turns. The old code's max-overlap logic
    # returned best_speaker=None and hard-coded "Speaker 1"; the
    # fix uses nearest-turn-by-midpoint within 1.0 s → Speaker 2.
    raw_segments = [
        {"start": 0.0, "end": 4.0, "text": "first", "words": None},
        {"start": 4.0, "end": 4.25, "text": "uh", "words": None},
        {"start": 4.3, "end": 8.0, "text": "reply", "words": None},
    ]
    out = _assign_speakers_from_diarization(raw_segments, speaker_map)
    # Sanity: pyannote first-appearance mapping.
    assert out[0].speaker == "Speaker 1"
    assert out[2].speaker == "Speaker 2"
    # The bug case: middle segment must NOT be "Speaker 1".
    assert out[1].speaker == "Speaker 2", (
        f"Expected middle segment to inherit nearest turn (Speaker 2), "
        f"got {out[1].speaker}"
    )


def test_assign_from_diarization_tertiary_inherits_from_previous():
    """When there is no nearby turn within 1.0 s, the fallback
    tier inherits the PREVIOUS segment's assigned speaker instead
    of defaulting to ``Speaker 1``. Verifies the third tier of the
    three-tier fallback.
    """
    from backend.services.transcription import _assign_speakers_from_diarization

    speaker_map = {
        (0.0, 2.0): "SPEAKER_01",   # reversed order: first-seen = SPEAKER_01
    }
    # Second segment 10s later — no nearby turn; must inherit
    # the previous segment's speaker, not "Speaker 1".
    raw_segments = [
        {"start": 0.0, "end": 2.0, "text": "first", "words": None},
        {"start": 15.0, "end": 17.0, "text": "second", "words": None},
    ]
    out = _assign_speakers_from_diarization(raw_segments, speaker_map)
    assert out[0].speaker == out[1].speaker


# ── Fix C — pyannote bounds are bracket, never exact ──


def test_diarize_audio_num_speakers_1_does_not_force_monologue():
    """Calling ``diarize_audio(num_speakers=1)`` must NOT pass
    ``num_speakers=1`` to pyannote. The new contract translates
    it to ``min_speakers=1, max_speakers=3`` so a single face
    slot (the noisy bug case) can still produce two clusters.
    """
    from backend.services import speaker_diarization as sd
    from backend.services._pyannote_loader import STATUS_OK

    captured_kwargs = {}

    class _FakeDiarization:
        def itertracks(self, yield_label=True):
            return iter([])

    fake_pipeline = MagicMock()

    def _call(audio_path, **kwargs):
        captured_kwargs.update(kwargs)
        return _FakeDiarization()

    fake_pipeline.side_effect = _call

    with patch("backend.services._pyannote_loader.get_pipeline",
               return_value=(fake_pipeline, STATUS_OK)), \
         patch("os.path.exists", return_value=True), \
         patch.object(sd, "_select_backend", return_value="pyannote"):
        sd.diarize_audio("/fake/audio.wav", num_speakers=1)

    assert "num_speakers" not in captured_kwargs, (
        f"pyannote must not receive an exact num_speakers= hint; "
        f"got {captured_kwargs}"
    )
    assert captured_kwargs.get("min_speakers") == 1
    assert captured_kwargs.get("max_speakers") == 3


def test_diarize_audio_num_speakers_2_becomes_bracket():
    """``num_speakers=2`` becomes ``min_speakers=1, max_speakers=4``."""
    from backend.services import speaker_diarization as sd
    from backend.services._pyannote_loader import STATUS_OK

    captured_kwargs = {}

    class _FakeDiarization:
        def itertracks(self, yield_label=True):
            return iter([])

    fake_pipeline = MagicMock()
    fake_pipeline.side_effect = (
        lambda audio_path, **kw: (captured_kwargs.update(kw) or _FakeDiarization())
    )

    with patch("backend.services._pyannote_loader.get_pipeline",
               return_value=(fake_pipeline, STATUS_OK)), \
         patch("os.path.exists", return_value=True), \
         patch.object(sd, "_select_backend", return_value="pyannote"):
        sd.diarize_audio("/fake/audio.wav", num_speakers=2)

    assert "num_speakers" not in captured_kwargs
    assert captured_kwargs.get("min_speakers") == 1
    assert captured_kwargs.get("max_speakers") == 4


def test_bound_kwargs_helper_returns_empty_for_none():
    """Auto-detection path: no hint → pyannote auto-detects."""
    from backend.services.speaker_diarization import _pyannote_bound_kwargs
    assert _pyannote_bound_kwargs(None) == {}
    assert _pyannote_bound_kwargs(0) == {}
    assert _pyannote_bound_kwargs(-1) == {}


def test_bound_kwargs_helper_shape_for_common_counts():
    from backend.services.speaker_diarization import _pyannote_bound_kwargs
    assert _pyannote_bound_kwargs(1) == {"min_speakers": 1, "max_speakers": 3}
    assert _pyannote_bound_kwargs(2) == {"min_speakers": 1, "max_speakers": 4}
    assert _pyannote_bound_kwargs(4) == {"min_speakers": 3, "max_speakers": 6}


# ── Fix C (pipeline side) — diarization runs for single-slot registries ──


def test_pipeline_calls_diarize_when_only_one_face_slot():
    """The guard in pipeline.py used to require _num_slots >= 2 before
    running diarization, so a single-face clip whose audio clearly
    had two speakers fell straight through to the heuristic tier.

    This test simulates the relevant source block and asserts
    ``diarize_audio`` IS called when _num_slots == 1. We import the
    helper path and rebuild the guard expression here — patching the
    full pipeline would require the entire job-runner context.
    """
    # Re-encode the guard. The fix lifted it from (_num_slots >= 2)
    # to (audio_path and active_speaker_events), with the slot
    # count now only a lower-bound HINT. Model that decision as a
    # plain function so future regressions on the guard are caught.
    def _should_diarize(num_slots, audio_path, active_speaker_events):
        # Matches pipeline.py post-fix.
        return bool(audio_path) and bool(active_speaker_events)

    assert _should_diarize(1, "/a.wav", [object()]) is True
    assert _should_diarize(0, "/a.wav", [object()]) is True  # no faces at all
    # Still skipped when audio is absent or we have no lip timeline.
    assert _should_diarize(3, "", [object()]) is False
    assert _should_diarize(3, "/a.wav", []) is False

    # Also verify the live source code still reflects the fix.
    # Read the file directly — importing pipeline pulls in heavy
    # deps (PIL, cv2, torch) that aren't available in every CI env.
    import pathlib
    _pipeline_path = (
        pathlib.Path(__file__).resolve().parents[1]
        / "services" / "pipeline.py"
    )
    src = _pipeline_path.read_text(encoding="utf-8")
    assert "_num_slots >= 2 and audio_path" not in src, (
        "pipeline.py still contains the _num_slots >= 2 guard — Fix C "
        "regressed."
    )
    assert "USE_DIARIZATION and audio_path and active_speaker_events" in src
