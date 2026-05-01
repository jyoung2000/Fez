# TACT coverage fixtures

Audio fixtures for the TACT integration tests. Each fixture targets
one of the coverage failure modes catalogued in
`docs/transcription_coverage.md`.

## Files

| File | Failure mode | Synthesis |
|------|--------------|-----------|
| `silent_mouthing.wav` | §5 lip-without-voice | Pure silence at < -50 dBFS, 10 s |
| `applause_only.wav` | non-speech audio | White-noise burst at 0 dBFS, 10 s |
| `music_only.wav` | §3 music bed | 440 Hz + 660 Hz sine sum, 10 s |
| `quiet_speech_60db.wav` | §1 noise gate | Sine tone at -40 dBFS, 10 s (placeholder) |
| `sentence_across_chunk_boundary.wav` | §8 chunk seam | 35 s synthesized waveform spanning the 30 s boundary |
| `speech_over_music.wav` | §3 music bed | Mixed sine + harmonic noise, 10 s |

These are **synthetic waveforms**, not real speech. They exist so the
integration tests can verify the TACT pipeline's structural behavior
(ledger bootstrap, ladder dispatch, coverage invariant) against
predictable inputs without requiring real Whisper inference or large
binary fixtures committed to the repo.

The real-audio acceptance test is a homelab smoke step using a
30-minute test video. Run it with:

```bash
python -m backend.cli transcribe --input some_30min_video.mp4 --job-id smoke_test
python -c "
import json
from pathlib import Path
job = json.loads(Path('jobs/smoke_test/job.json').read_text())
cr = job['coverage_report']
print('coverage_ratio:', cr.get('coverage_ratio') or cr.get('source', {}).get('coverage_ratio'))
print('ladder_stats:', cr.get('ladder_stats'))
"
```

The acceptance criterion is `coverage_ratio == 1.0` with non-zero
rung activity in `ladder_stats`.

## Ground truth

Each `*.wav` ships with a `*.ground_truth.json` documenting expected
coverage outcomes. Format:

```json
{
  "duration_sec": 10.0,
  "min_overall_coverage_ratio": 1.0,
  "expected_event_tags": ["[silence]"],
  "notes": "Pure silence below -50 dBFS"
}
```

The integration tests check `coverage_ratio >= min_overall_coverage_ratio`
and (when `expected_event_tags` is set) that those tags appear in the
final transcript.
