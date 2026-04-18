// Ad-hoc smoke test for frontend/src/utils/subtitleTiming.js.
//
// No frontend test harness exists, so this file is intentionally a
// framework-free Node script. Run it with:
//
//     node scripts/test_subtitleTiming.mjs
//
// It exits non-zero on any failed assertion so it's CI-friendly if you
// decide to wire it up later.

import {
  spokenWindow,
  isSpokenAt,
  WORD_HEAD_S,
  WORD_TAIL_S,
} from '../frontend/src/utils/subtitleTiming.js';

let failures = 0;
function check(name, cond, details) {
  if (cond) {
    console.log(`  ok  ${name}`);
  } else {
    failures += 1;
    console.log(`  FAIL ${name}`);
    if (details !== undefined) console.log('       ', details);
  }
}
function approx(a, b, eps = 1e-9) {
  return Math.abs(a - b) < eps;
}

console.log('spokenWindow — segment with no words');
{
  const seg = { start: 1.0, end: 2.5 };
  const w = spokenWindow(seg);
  check('returns segment-level start', approx(w.start, 1.0), w);
  // Fallback path now extends the end by WORD_TAIL_S so back-to-back
  // non-word-timestamped segments overlap and first-match produces
  // the late-bias transition.
  check('extends end by WORD_TAIL_S in fallback', approx(w.end, 2.5 + WORD_TAIL_S), w);
}

console.log('spokenWindow — segment with valid words');
{
  const seg = {
    start: 1.0,
    end: 2.5,
    words: [
      { start: 1.10, end: 1.40 },
      { start: 1.45, end: 1.80 },
      { start: 1.85, end: 2.30 },
    ],
  };
  const w = spokenWindow(seg);
  // start = max(segStart - 0.02, ws - WORD_HEAD_S)
  //       = max(0.98, 1.10 - 0.04) = max(0.98, 1.06) = 1.06
  check('start uses words[0].start - WORD_HEAD_S', approx(w.start, 1.10 - WORD_HEAD_S), w);
  // end = max(segEnd, we) + WORD_TAIL_S
  //     = max(2.5, 2.30) + 0.20 = 2.5 + 0.20 = 2.70
  check('end = max(segEnd, we) + WORD_TAIL_S (segEnd wins)', approx(w.end, 2.5 + WORD_TAIL_S), w);
}

console.log('spokenWindow — words extend past segEnd (we wins)');
{
  // Whisper sometimes places the final word's end past the segment
  // boundary; the ``max`` branch should honor it.
  const seg = {
    start: 1.0,
    end: 2.0,
    words: [
      { start: 1.10, end: 2.30 },
    ],
  };
  const w = spokenWindow(seg);
  // end = max(2.0, 2.30) + 0.20 = 2.30 + 0.20 = 2.50
  check('end uses we when we > segEnd', approx(w.end, 2.30 + WORD_TAIL_S), w);
}

console.log('spokenWindow — segment with inverted/broken words');
{
  const seg1 = { start: 1.0, end: 2.0, words: [{ start: 0, end: 0 }] };
  const w1 = spokenWindow(seg1);
  // Broken words fall back to seg-level with the WORD_TAIL_S extension.
  check(
    'zero-duration word falls back to seg + tail',
    approx(w1.start, 1.0) && approx(w1.end, 2.0 + WORD_TAIL_S),
    w1,
  );

  const seg2 = { start: 1.0, end: 2.0, words: [{ start: 1.5, end: 1.2 }] };
  const w2 = spokenWindow(seg2);
  check(
    'inverted word (end <= start) falls back to seg + tail',
    approx(w2.start, 1.0) && approx(w2.end, 2.0 + WORD_TAIL_S),
    w2,
  );

  const seg3 = { start: 1.0, end: 2.0, words: [{ start: 'bad', end: 1.5 }] };
  const w3 = spokenWindow(seg3);
  check(
    'non-numeric word start falls back to seg + tail',
    approx(w3.start, 1.0) && approx(w3.end, 2.0 + WORD_TAIL_S),
    w3,
  );

  const seg4 = { start: 1.0, end: 2.0, words: [] };
  const w4 = spokenWindow(seg4);
  check(
    'empty word array falls back to seg + tail',
    approx(w4.start, 1.0) && approx(w4.end, 2.0 + WORD_TAIL_S),
    w4,
  );
}

console.log('spokenWindow — clamp to segStart on the start side');
{
  // Words earlier than segStart should be clamped up to
  // ``segStart - 0.02`` so we can't overlap the previous segment.
  const seg = {
    start: 5.0,
    end: 6.0,
    words: [
      { start: 1.00, end: 1.20 }, // pathological early
      { start: 1.30, end: 1.50 },
    ],
  };
  const w = spokenWindow(seg);
  check('start clamped to segStart - 0.02', approx(w.start, 4.98), w);
}

console.log('isSpokenAt — boundary conditions');
{
  const seg = {
    start: 1.0,
    end: 2.0,
    words: [
      { start: 1.02, end: 1.50 },
      { start: 1.55, end: 1.95 },
    ],
  };
  const w = spokenWindow(seg);
  // w.start = max(0.98, 1.02 - 0.04) = max(0.98, 0.98) = 0.98
  // w.end = max(2.0, 1.95) + 0.20 = 2.0 + 0.20 = 2.20
  check('start = 0.98', approx(w.start, 0.98), w);
  check('end = 2.20', approx(w.end, 2.20), w);
  check('inclusive at start', isSpokenAt(seg, w.start) === true, w);
  check('exclusive at end', isSpokenAt(seg, w.end) === false, w);
  check('1 ms inside is active', isSpokenAt(seg, w.start + 0.001) === true, w);
  check('1 ms after end is not active', isSpokenAt(seg, w.end + 0.001) === false, w);
  check('1 ms before start is not active', isSpokenAt(seg, w.start - 0.001) === false, w);
}

console.log('Integration — TranscriptViewer gap-hold math');
{
  // Minimal re-implementation of the TranscriptViewer memo body so we
  // can unit-test the gap-hold behavior without a React harness.
  const activeIdx = (transcript, currentTime, GAP_HOLD_SEC = 0.75) => {
    if (currentTime == null || !transcript.length) return -1;
    let lastEndedIdx = -1;
    let lastEndedTime = -Infinity;
    for (let i = 0; i < transcript.length; i++) {
      const w = spokenWindow(transcript[i]);
      if (w.start <= currentTime && currentTime < w.end) return i;
      if (w.end <= currentTime && w.end > lastEndedTime) {
        lastEndedTime = w.end;
        lastEndedIdx = i;
      }
    }
    if (lastEndedIdx >= 0 && currentTime - lastEndedTime <= GAP_HOLD_SEC) {
      return lastEndedIdx;
    }
    return -1;
  };

  // Segments with no word timestamps — spokenWindow returns
  // [seg.start, seg.end + WORD_TAIL_S].
  //   seg0: 0.0–1.0 → window 0.0–1.20
  //   (short gap: 0.3 s)
  //   seg1: 1.3–2.0 → window 1.3–2.20
  //   (long gap: 2.0 s)
  //   seg2: 4.0–5.0 → window 4.0–5.20
  const tr = [
    { start: 0.0, end: 1.0 },
    { start: 1.3, end: 2.0 },
    { start: 4.0, end: 5.0 },
  ];
  check('t=0.5 → seg0', activeIdx(tr, 0.5) === 0);
  // With the tail extension, seg0 still has window [0, 1.20] so the
  // small 0.3s gap is absorbed by the window, not by gap-hold.
  check('t=1.1 (inside seg0 window) → seg0', activeIdx(tr, 1.1) === 0);
  check('t=1.5 → seg1', activeIdx(tr, 1.5) === 1);
  // Right after seg1's window ends (2.20), gap-hold keeps seg1
  // active for up to 0.75 s.
  check('t=2.3 (just past seg1 end, inside 0.75s hold) → seg1', activeIdx(tr, 2.3) === 1);
  check('t=2.9 (0.70s past seg1 end, still inside hold) → seg1', activeIdx(tr, 2.9) === 1);
  check('t=3.0 (exactly 0.80s past, past hold) → -1', activeIdx(tr, 3.0) === -1);
  check('t=3.5 (far past hold) → -1', activeIdx(tr, 3.5) === -1);
  check('t=4.1 → seg2', activeIdx(tr, 4.1) === 2);
  check('t=99 (past everything, past hold) → -1', activeIdx(tr, 99) === -1);
}

console.log('Regression — diarization split, back-to-back lines');
{
  // Scenario from the ClipSEO screenshot:
  //   row A = "Please, before I die, I want to know what my father did..."
  //           seg: [883, 890], words end at 889.5 (Whisper trimmed the
  //           trailing silence out of the word timestamps).
  //   row B = "Huh? I thought you'd know." seg: [890, 893], words at 890.3.
  //
  // Old code: A's window = min(seg.end+0.15, we+0.06) = min(890.15, 889.56)
  //           = 889.56. At t=890 (when B's direct hit begins), the
  //           loop matches B, highlight jumps to B even though the
  //           audio is still saying A.
  //
  // New code: A's window end = max(seg.end, we) + WORD_TAIL_S
  //           = max(890, 889.5) + 0.20 = 890.20. A stays active until
  //           890.20. B's start = max(890 - 0.02, 890.3 - 0.04) = 890.26.
  //           So at t=890.10 (inside A's tail), direct match on A wins.
  //           At t=890.25, neither A nor B direct-matches, gap-hold
  //           keeps A. At t=890.30 B wins via direct match.
  const activeIdx = (transcript, currentTime, GAP_HOLD_SEC = 0.75) => {
    let lastEndedIdx = -1;
    let lastEndedTime = -Infinity;
    for (let i = 0; i < transcript.length; i++) {
      const w = spokenWindow(transcript[i]);
      if (w.start <= currentTime && currentTime < w.end) return i;
      if (w.end <= currentTime && w.end > lastEndedTime) {
        lastEndedTime = w.end;
        lastEndedIdx = i;
      }
    }
    if (lastEndedIdx >= 0 && currentTime - lastEndedTime <= GAP_HOLD_SEC) {
      return lastEndedIdx;
    }
    return -1;
  };
  const tr = [
    {
      start: 883.0,
      end: 890.0,
      words: [
        { start: 883.1, end: 884.5 },
        { start: 884.6, end: 889.5 },
      ],
    },
    {
      start: 890.0,
      end: 893.0,
      words: [
        { start: 890.3, end: 893.0 },
      ],
    },
  ];
  check('t=886.0 → row A (mid-line)', activeIdx(tr, 886.0) === 0);
  check('t=890.0 (exactly seg boundary) → row A still active', activeIdx(tr, 890.0) === 0);
  check('t=890.10 (inside A tail) → row A', activeIdx(tr, 890.10) === 0);
  check('t=890.19 (still inside A tail) → row A', activeIdx(tr, 890.19) === 0);
  check('t=890.25 (past A tail, before B start, in gap-hold) → row A', activeIdx(tr, 890.25) === 0);
  check('t=890.30 (B direct hit starts) → row B', activeIdx(tr, 890.30) === 1);
  check('t=892.0 → row B', activeIdx(tr, 892.0) === 1);
}

console.log('');
if (failures > 0) {
  console.log(`FAILED: ${failures} assertion(s)`);
  process.exit(1);
} else {
  console.log('All subtitleTiming smoke tests passed.');
}
