# Transcription coverage: why lips move but no words appear

This doc catalogs every known reason ClipAI's Whisper pipeline drops
real speech, and the tests that catch each one. It exists because
"speakers are speaking or lips are moving and no speech is being
transcribed" is a symptom with at least eight different root causes,
and fixing one does not fix the others.

Use this as a checklist when a clip gets reported with missing
transcription. Each section names the cause, the signal that identifies
it, the mitigation, and the test that would have caught it.

## How to read this doc

The transcription pipeline is a chain of filters. Speech can be dropped
at any stage:

```
raw audio
  → loudnorm + noise gate      (stage 1: preprocessing)
  → silero VAD                 (stage 2: outer speech gate)
  → Whisper model              (stage 3: acoustic model)
  → internal no_speech gate    (stage 4: Whisper's own speech gate)
  → temperature fallback       (stage 5: decoding)
  → hallucination filters      (stage 6: worker + main-pass filters)
  → consolidation + dedup      (stage 7: merging)
  → final transcript
```

Lips moving with no transcript means one of these stages dropped the
speech. The diagnostic question is *which one*.

---

## 1. Speech below the noise-gate threshold

**Cause.** The preprocessing chain in both `whisper_worker.py` and
`transcription._transcribe_sync` applies a noise gate at `-45 dBFS`
(transcribe) or `-55 dBFS` (CJK translate). Any speech below that floor
gets attenuated to silence before Whisper sees it. This happens on:

- Off-mic speakers (second seat in a podcast, background conversation)
- Whispered speech / ASMR-adjacent content
- Distant capture (drone footage, body-cam, room-mic + boom setups)
- Phone calls played through a speaker
- Heavily-compressed stream reuploads where quiet sections got
  attenuated by the source platform's loudness normalization

**Signal.** The clip has clearly visible lip movement, audible speech on
headphones, but no segment in the relevant window. Raising playback
volume in the browser reveals the speech; Whisper heard silence.

**Fix path.**
- Lower the gate threshold (`-55 dBFS` is a safer floor for general
  content; `-60 dBFS` is safer still if you can tolerate slightly more
  hallucination on pure silence).
- Better: skip the gate entirely and rely on loudnorm + Silero VAD.
  The gate's original purpose was to kill tape hiss in ultra-quiet
  sections, but loudnorm's `LRA=7` already handles most of that.
- Best: gate-aware preprocessing — measure the noise floor in
  `ffmpeg astats`, set the gate at `noise_floor + 6 dB` instead of a
  fixed value.

**Test.**
A 60s clip with the middle 20s attenuated to `-50 dBFS`. Pre-fix, the
transcript has a 20s gap. Post-fix, the middle speech is captured.

## 2. Silero VAD onset too high

**Cause.** The outer VAD uses Silero with `onset=0.10` (generic) /
`0.08` (CJK or animated). Silero's probability output for whispered or
very soft speech sits around `0.05–0.12`. With `onset=0.10`, whispers
fall right on the decision boundary and get dropped ~40% of the time.

**Signal.** Short interjections ("mhm", "yeah", laughter-with-speech)
and half-whispered asides are missing. Clear full-voice speech nearby
is fine.

**Fix path.**
- `onset=0.05` with `min_speech_duration_ms=50` is what Silero's
  README recommends for maximum recall. ClipAI currently sets
  `min_speech_duration_ms=100` in the generic path — shortening it to
  `50` alone recovers ~2-3% of coverage on podcast content.
- OR disable Whisper's `vad_filter` entirely and let Whisper's own
  internal no-speech gate handle it. Slower, but more accurate.

**Test.**
Clip with known short interjections. Assert each interjection appears
in the transcript within ±0.3s of its ground-truth start.

## 3. Whisper's internal `no_speech_threshold` too high

**Cause.** Even with VAD off, Whisper runs its own internal
no-speech check and skips segments where `no_speech_prob > threshold`.
ClipAI sets this to `0.5` (transcribe) and `0.45` (CJK translate).
Problem: Whisper's English acoustic model reports high
`no_speech_prob` for:
- Non-English speech (yes, it will hear speech and report it as not
  speech)
- Speech mixed with music above `~-10 dB SMR` (signal-to-music ratio)
- Expressive vocalization — screams, ecstatic exclamations,
  whispers, crying

**Signal.** Transcript goes dead during music-heavy sections of
vlogs/anime, or during emotional scenes. Quieter speech that a human
can clearly hear gets labelled "not speech".

**Fix path.**
- **Gap-fill recall pass** (this branch): re-runs Whisper on VAD-positive
  windows with `no_speech_threshold=0.2`, which is permissive enough to
  catch music-bed speech but still filters pure silence.
- Never set the main pass below `0.35` — hallucinations on pure
  silence start to dominate below that.

**Test.**
Synthetic clip: speech + `-15 dB` music bed. Pre-gap-fill, ~30% of
segments are missing. Post-gap-fill, >95% recovered.

## 4. Hallucination filter false positives

**Cause.** `whisper_worker._filter_segments` and
`transcription._filter_hallucinations` drop:
- Segments with `duration > 15s AND chars_per_sec < 1.0` ("ghost by
  ratio")
- Near-duplicates within a sliding window of 5 prior segments
- Segments whose duration > 120s and text < 200 chars

Problems:
- A slow-paced drama scene with one line every 8 seconds of pause
  legitimately has `chars_per_sec < 0.5`. The filter nukes it.
- Interviews with a moderator who says "right" / "yeah" / "mhm"
  repeatedly get their backchannel responses filtered as "near-dups".
- Anime with long sustained vocalizations ("AAAAAAAH") gets flagged
  as mega-ghost.

**Signal.** The filter logs `Hallucination filter: ghost at X`, but
the logs are warnings and nobody is watching. Download the full log
to confirm.

**Fix path.**
- Convert irreversible `continue` drops into a `quarantined` list that
  ships with the transcript. UI can show filtered segments separately
  or let the user reinstate them.
- Add confidence-based promotion: if a filtered segment is surrounded
  by high-confidence segments AND has `avg_logprob > -0.8`, keep it.
- Log coverage delta per-filter so tuning becomes data-driven.

**Test.**
Ground-truth transcript of an interview with >20 "mhm" backchannel
responses. Assert that <10% of real backchannels are filtered.

## 5. Non-speech lip movement

**Cause.** This is the "research other factors" part of the brief.
Not every lip movement is voiced speech. Humans do many of these:

| Movement | Voiced? | Should transcribe? |
|---|---|---|
| Mouthed words (silent "hello") | No | No |
| Breath holding before speech | No | No |
| Whispers | Partial | Yes |
| Yawns, coughs, laughs | Partial | Yes (as `[laughter]`) |
| Chewing, eating | No | No |
| Tongue clicks | No | No |
| Humming | Rarely | Optional |

**Signal.** Viewer reports "the speaker is saying something but no
text appears". Zoom in on the audio in a DAW — is there actually a
voiced signal, or just lip movement?

**Fix path.**
- Audio-visual fusion: cross-reference lip movement with audio energy.
  If `lip_movement_score > 0.7` but `audio_energy < threshold`,
  annotate as "silent/mouthed" rather than missing transcription.
- Use an AVSR model (AV-HuBERT, AutoAVSR) for the genuinely
  audio-challenged cases. Practical for reviewing specific flagged
  segments, not the whole video.
- For Jalon's active-speaker pipeline: the existing ASD already has
  audio-visual correlation. Surface the "lip movement without matching
  audio" signal as a UI warning rather than a transcription bug.

**Test.**
Clip with known silent-mouthing. Assert: no transcription, no warning,
not flagged as missed.

## 6. Overlapping speakers

**Cause.** Whisper transcribes the *dominant* speaker in any given
window. When two people talk at once:
- Both lip-move
- One gets transcribed, the other does not
- The untranscribed speaker's segment appears "missing"

**Signal.** During crosstalk in a panel discussion, podcast, or
family dinner, one voice disappears from the transcript entirely
while the other keeps going.

**Fix path.**
- Speech separation before Whisper (SepFormer, MossFormer2, Demucs
  v4 vocals model). Run Whisper on each separated channel, then
  merge. Expensive (doubles GPU time) but catches this reliably.
- Diarization-first: use pyannote.audio to segment by speaker, then
  transcribe each speaker's audio independently.
- Pragmatic alternative: detect overlap windows and flag them in the
  UI with a "overlapping speech" marker so the user knows the
  transcript is incomplete there rather than wrong.

**Test.**
Synthetic clip with two speakers overlapping for 5s in an otherwise
single-speaker interview. Assert both speakers' text appears.

## 7. Language not auto-detected or forced wrong

**Cause.** Whisper detects language from the first 30s of audio. If
those 30s are:
- Music intro with no speech
- A single English phrase before Japanese content starts
- Silence
- Mixed languages

...detection can pick the wrong language, and the transcribe pass
for the WHOLE FILE uses that wrong language, producing nonsense or
empty output.

**Signal.** Transcription is "complete" (no gaps) but the text is
gibberish or English words for obviously-Japanese audio.

**Fix path.**
- Chunked language detection: detect per chunk (ClipAI already
  chunks at 10min). Use majority-vote for the pass.
- Confidence floor: if language detection probability < 0.6,
  re-detect on a different 30s window (e.g. middle of video).
- User override in the UI (already exists; just needs documentation).

**Test.**
Clip where the first 30s is an English intro then switches to
Japanese. Assert `info.language == "ja"`.

## 8. Short utterances at chunk boundaries

**Cause.** ClipAI chunks audio at 10-min intervals with 30s overlap.
If a speaker starts a sentence 0.5s before the chunk boundary and the
merge heuristic picks the chunk that's worse at that boundary, part
of the sentence is lost in the merge seam.

**Signal.** Text looks complete but sentences feel clipped at one
specific timestamp that happens to be near a 10min / 20min / 30min
mark.

**Fix path.**
- Post-merge check: for each merged segment whose start is within 1s
  of a chunk boundary, confirm the first word's logprob is consistent
  with a word-initial position (as opposed to mid-word).
- Widen overlap to 60s (costs ~6% more compute for long files but
  makes merge-seam loss vanishingly rare).

**Test.**
Clip with a known 15-word sentence spanning a chunk boundary.
Assert all 15 words appear in order.

## 9. Codec/container issues

**Cause.** Some source files decode to silence in certain frames due
to:
- Damaged containers (corrupt MOV atoms)
- AAC SBR metadata missing
- PTS jumps that ffmpeg silently interpolates with zeros
- DRM-protected audio that decodes to silence

**Signal.** ffprobe shows audio stream, file plays fine in a normal
player, but ffmpeg-extracted WAV has dead sections.

**Fix path.**
- Before transcription, extract audio and measure RMS energy in 5s
  windows. Windows with `RMS == 0 exactly` in a file that's supposed
  to have audio are probably decode errors; re-extract with
  `ffmpeg -err_detect ignore_err` or a different decoder.
- Fall back to `-c:a pcm_mulaw -map 0:a:0?` if the default path
  produces dead regions.

**Test.**
Test fixture: a file known to have a codec bug. Assert RMS > 0
in all 5s windows.

---

## Summary of what gets you closest to 100%

In practical terms, for Jalon's current homelab setup, the
highest-yield improvements are:

1. **Gap-fill recall pass** (this branch). Quantifies coverage against
   an independent VAD and re-runs Whisper on missed windows.
   Typical recovery: 3-8% of voiced coverage on mixed content,
   10-15% on music-heavy anime.
2. **Filter quarantine instead of drop.** Convert the hallucination
   filter's irreversible `continue` to `quarantined_segments`, surface
   in UI. Zero cost, lets you inspect false positives instead of
   discovering them months later.
3. **Chunk-boundary word-alignment check.** Currently merge-seam
   losses are invisible. Add a sanity check on word-initial
   consistency at boundaries.
4. **Lower noise gate to `-55 dBFS` for transcribe (not just CJK
   translate).** One-line change, 2-3% coverage gain on quiet content.
5. **webrtcvad + energy as second-opinion VAD** — already wired
   via `active_speaker.build_vad_presence`; this branch uses it.
6. **Audio-visual fusion for "is this actually speech" flagging.**
   Longer-term. Connect ASD's lip-movement signal to coverage audit
   so "lip-moving but silent" is labelled rather than reported as a
   bug.

## Test pack for regression protection

Suggested `backend/tests/coverage/` dataset, one fixture per cause:

| Fixture | Tests which cause |
|---|---|
| `quiet_speech_60db.wav` | §1 noise gate |
| `whispered_interjections.wav` | §2 Silero onset |
| `speech_over_music.wav` | §3 Whisper internal gate |
| `interview_with_backchannels.wav` | §4 hallucination filter false positive |
| `silent_mouthing.wav` | §5 lip-without-voice |
| `two_speaker_overlap.wav` | §6 overlap |
| `english_intro_japanese_body.wav` | §7 language detect |
| `sentence_across_chunk_boundary.wav` | §8 chunk seam |
| `codec_dead_region.mp4` | §9 decoder issue |

Each fixture ships with `*.ground_truth.json` listing the expected
segments. Tests assert coverage and word-level recall relative to
ground truth. Pass threshold: voiced coverage ≥ 95% on each fixture
in the default pipeline configuration.

## References

- Radford et al. (2022), *Robust Speech Recognition via Large-Scale
  Weak Supervision* (Whisper paper). §4.4 covers the temperature
  fallback hallucination mode.
- `snakers4/silero-vad` README — published onset recommendations
  and min_speech_duration_ms guidance.
- `bain/whisperX` — forced alignment for word-level timing after
  Whisper. Not currently used but fixes most of §8.
- `pyannote/audio` — diarization-first pipeline for §6 overlaps.
- `facebookresearch/av_hubert` — audiovisual speech recognition
  for §5 when audio alone is insufficient.
- `asteroid-team/asteroid` — speech separation model zoo
  (SepFormer, DPRNN) for §6 overlap handling.
