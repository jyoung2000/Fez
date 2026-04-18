/**
 * Subject tracking utilities for dynamic crop positioning.
 *
 * These functions build keyframes from scene analysis data and interpolate
 * the subject's horizontal position at any point in time, enabling the
 * preview player and FFmpeg export to follow the subject smoothly.
 *
 * Processing pipeline (applied in order):
 *   1. buildSubjectKeyframes()        — raw (time, subject_x) pairs with DYNAMIC safe margin
 *   2. compressRange()                — limit total sx swing per clip toward median (aspect-ratio-aware)
 *   3. applyDeadZone()                — anchor-based hold zone (eliminate drift, not movements)
 *   4. handleSceneCuts()              — insert 1ms instant-jump keyframes at hard cuts
 *   5. smoothKeyframesBidirectional() — damped-lerp with hold-then-move (maxSpeed=15)
 *   6. mergeHolds()                   — merge similar consecutive values into rests (tolerance=3)
 *   7. interpolateSubjectX()          — smoothstep ease-in/ease-out interpolation
 */

/** Build version for debugging — check browser console for this string */
export const SUBJECT_TRACKING_VERSION = 'v3.1-dense-parity-2026-04-04';
console.log(`[SubjectTracking] Module loaded: ${SUBJECT_TRACKING_VERSION}`);

/** Legacy safety margin — used when no aspect ratio info is provided. */
const SAFE_MARGIN = 10;

/**
 * Compute the safe subject_x range for a given aspect ratio conversion.
 * Ensures that any subject_x within this range will produce a non-clamped
 * objectPosition value — meaning the subject can actually be centered.
 *
 * @param {number} srcRatio - Source video aspect ratio (e.g., 16/9)
 * @param {number} targetRatio - Target crop aspect ratio (e.g., 9/16)
 * @param {number} edgeBuffer - Buffer from objectPosition 0%/100% (default 8)
 * @returns {{ min: number, max: number }} Safe subject_x range
 */
export function computeSafeRange(srcRatio, targetRatio, edgeBuffer = 3) {
  const R = srcRatio / targetRatio;
  if (R <= 1.01) {
    // No horizontal overflow — any subject_x is fine
    return { min: 5, max: 95 };
  }
  // Scale edgeBuffer with magnification ratio so that at high R values
  // (e.g. 16:9→9:16, R≈3.16) the crop can't push a face to the frame edge.
  // A face is ~10% of frame width. At R=3.16 the crop window is ~31% of
  // source width, so 10% of source = ~32% of crop. We need the crop center
  // to stay at least half a face width from the crop edge.
  // edgeBuffer=8 works for R<2, but at R≈3 we need ~14-15.
  // Use gentler scaling (2x instead of 3x) to avoid over-clamping edge
  // speakers in 2-person podcast layouts (e.g. faces at x=20% and x=80%).
  const scaledBuffer = Math.min(15, Math.round(edgeBuffer + (R - 1) * 1.5));
  // Invert the centerPct formula: sx = (pct * (R - 1) + 50) / R
  const sxAtMin = (scaledBuffer * (R - 1) + 50) / R;
  const sxAtMax = ((100 - scaledBuffer) * (R - 1) + 50) / R;
  return {
    min: Math.ceil(Math.max(5, sxAtMin)),
    max: Math.floor(Math.min(95, sxAtMax)),
  };
}

/**
 * Clamp subject_x to the safe range for a given aspect ratio.
 * Falls back to static margin if no aspect ratio info provided.
 * Matches backend _safe_subject_x() exactly for preview-export parity.
 *
 * @param {number} sx - Raw subject_x value (0-100)
 * @param {number|null} srcRatio - Source video aspect ratio (optional)
 * @param {number|null} targetRatio - Target crop aspect ratio (optional)
 * @returns {number} Clamped subject_x
 */
export function safeSubjectX(sx, srcRatio = null, targetRatio = null) {
  if (srcRatio && targetRatio) {
    const range = computeSafeRange(srcRatio, targetRatio);
    return Math.max(range.min, Math.min(range.max, Math.round(sx)));
  }
  // Fallback: static margin (legacy behavior)
  return Math.max(SAFE_MARGIN, Math.min(100 - SAFE_MARGIN, Math.round(sx)));
}

/**
 * Build a map of speaker → average subject_x position by correlating
 * scene analysis data with transcript speaker labels.
 *
 * @param {Array} scenes - Scene objects with {timestamp, subject_x}
 * @param {Array} transcript - Transcript segments with {start, end, speaker}
 * @returns {Object} Map of speaker name → average subject_x
 */
export function buildSpeakerPositionMap(scenes, transcript) {
  if (!scenes?.length || !transcript?.length) return {};

  const speakerXValues = {};

  for (const scene of scenes) {
    const ts = scene.timestamp;
    const sx = scene.subject_x ?? 50;

    const activeSeg = transcript.find(seg => {
      const segStart = seg.start ?? seg.start_time ?? 0;
      const segEnd = seg.end ?? seg.end_time ?? 0;
      return ts >= segStart - 0.5 && ts <= segEnd + 0.5;
    });

    if (activeSeg?.speaker) {
      if (!speakerXValues[activeSeg.speaker]) {
        speakerXValues[activeSeg.speaker] = [];
      }
      speakerXValues[activeSeg.speaker].push(sx);
    }
  }

  const speakerMap = {};
  for (const [speaker, values] of Object.entries(speakerXValues)) {
    if (values.length > 0) {
      speakerMap[speaker] = Math.round(values.reduce((a, b) => a + b, 0) / values.length);
    }
  }
  return speakerMap;
}

/**
 * Build dense keyframes at every speaker change using the speaker-position map.
 *
 * @param {Array} transcript - Transcript segments with {start, end, speaker}
 * @param {Object} speakerMap - Speaker → subject_x map
 * @param {number} clipStart - Clip start time in seconds
 * @param {number} clipEnd - Clip end time in seconds
 * @param {number|null} srcRatio - Source video aspect ratio
 * @param {number|null} targetRatio - Target crop aspect ratio
 * @returns {Array<{t: number, x: number}>|null} Dense keyframes, or null if insufficient data
 */
export function buildSpeakerKeyframes(transcript, speakerMap, clipStart, clipEnd, srcRatio = null, targetRatio = null) {
  if (!transcript?.length || !speakerMap || Object.keys(speakerMap).length < 2) return null;

  const clipDur = clipEnd - clipStart;
  if (clipDur <= 0) return null;

  const overlapping = transcript
    .filter(seg => {
      const segStart = seg.start ?? seg.start_time ?? 0;
      const segEnd = seg.end ?? seg.end_time ?? 0;
      return segEnd > clipStart && segStart < clipEnd;
    })
    .sort((a, b) => (a.start ?? a.start_time ?? 0) - (b.start ?? b.start_time ?? 0));

  if (overlapping.length === 0) return null;

  const keyframes = [];
  let lastSpeaker = null;

  for (const seg of overlapping) {
    const segStart = Math.max(clipStart, seg.start ?? seg.start_time ?? 0);
    const speaker = seg.speaker;
    if (!speaker || !speakerMap[speaker]) continue;
    if (speaker === lastSpeaker) continue;

    const sx = safeSubjectX(speakerMap[speaker], srcRatio, targetRatio);
    keyframes.push({ t: Math.max(0, segStart - clipStart), x: sx });
    lastSpeaker = speaker;
  }

  if (keyframes.length === 0) return null;
  if (keyframes[0].t > 0) keyframes.unshift({ t: 0, x: keyframes[0].x });
  if (keyframes[keyframes.length - 1].t < clipDur) {
    keyframes.push({ t: clipDur, x: keyframes[keyframes.length - 1].x });
  }
  return keyframes.length >= 2 ? keyframes : null;
}

/**
 * Detect N distinct position clusters in subject_x values.
 * Uses recursive largest-gap splitting to find 2-6 natural groupings.
 *
 * Works WITHOUT audio diarization — catches multi-speaker scenarios
 * even when Whisper only detects 1 speaker, by finding position clusters
 * in the scene analysis subject_x values.
 *
 * For 2 speakers: [{center:35, count:12}, {center:65, count:15}]
 * For 4 speakers: [{center:15, count:5}, {center:35, count:12}, {center:65, count:15}, {center:85, count:8}]
 *
 * Matches backend _detect_position_clusters() exactly for preview-export parity.
 *
 * @param {Array<{t: number, x: number}>} keyframes
 * @param {number} gapThreshold - Minimum gap to split a cluster (default 10)
 * @param {number} minClusterSize - Min samples per cluster (default 2)
 * @param {number} maxClusters - Maximum clusters to detect (default 6)
 * @returns {Array<{center: number, count: number}>|null} Sorted clusters or null
 */
export function detectPositionClusters(keyframes, gapThreshold = 10, minClusterSize = 2, skipCenterStrip = false) {
  if (!keyframes || keyframes.length < 4) return null;

  const xs = keyframes.map(k => k.x);

  // Recursive gap-based splitting — no cap, finds as many clusters as exist
  function splitCluster(values) {
    if (values.length < minClusterSize * 2) return [values];
    const sorted = [...values].sort((a, b) => a - b);
    let maxGap = 0;
    let splitIdx = -1;
    for (let i = 1; i < sorted.length; i++) {
      const gap = sorted[i] - sorted[i - 1];
      if (gap > maxGap) {
        maxGap = gap;
        splitIdx = i;
      }
    }
    if (maxGap < gapThreshold || splitIdx < 0) return [values];
    const left = sorted.slice(0, splitIdx);
    const right = sorted.slice(splitIdx);
    if (left.length < minClusterSize || right.length < minClusterSize) return [values];
    return [...splitCluster(left), ...splitCluster(right)];
  }

  // Compute center (median) and count for each cluster
  const median = (arr) => {
    const s = [...arr].sort((a, b) => a - b);
    const mid = Math.floor(s.length / 2);
    return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
  };

  function buildResult(clusters) {
    if (clusters.length < 2) return null;
    const result = clusters
      .map(values => {
        const sorted = [...values].sort((a, b) => a - b);
        const trim = Math.max(1, Math.floor(sorted.length * 0.1));
        const trimmed = sorted.length > 2 ? sorted.slice(trim, sorted.length - trim) : sorted;
        const center = trimmed.length > 0
          ? Math.round(trimmed.reduce((s, v) => s + v, 0) / trimmed.length)
          : Math.round(median(sorted));
        return { center, count: values.length };
      })
      .sort((a, b) => a.center - b.center);
    for (let i = 1; i < result.length; i++) {
      if (result[i].center - result[i - 1].center < 8) return null;
    }
    // ── Midpoint cluster rejection ──
    // A cluster near the midpoint between its neighbors with fewer samples
    // is likely an averaging artifact (e.g., merged face detection spanning
    // both speakers). Remove it.
    if (result.length >= 3) {
      const totalCount = result.reduce((sum, c) => sum + c.count, 0);
      for (let i = result.length - 2; i >= 1; i--) {
        const mid = (result[i - 1].center + result[i + 1].center) / 2;
        const span = result[i + 1].center - result[i - 1].center;
        // Never reject a cluster with 20%+ of total keyframes (e.g., center speaker)
        if (Math.abs(result[i].center - mid) < span * 0.3 &&
            result[i].count < Math.max(result[i - 1].count, result[i + 1].count) &&
            result[i].count < totalCount * 0.2) {
          result.splice(i, 1);
        }
      }
      if (result.length < 2) return null;
    }
    return result;
  }

  // ── Run ALL passes and pick the best result ──
  // Pass 1 might find a midpoint phantom cluster that has MORE samples
  // than real speaker clusters (from Haar cascade merged detections).
  // Center stripping in Pass 2/3 removes those values and often produces
  // a cleaner 2-cluster result that should be preferred.

  const result1 = buildResult(splitCluster(xs));

  // Pass 2 & 3: Strip center noise zone — ONLY for sparse AI data.
  // Dense face tracking (per-second) uses actual face positions from slot centers,
  // not AI vision defaults. Stripping center values from dense data deletes real
  // speaker positions (e.g., a center speaker at 53% in a multi-person panel).
  let result2 = null;
  let result3 = null;
  if (!skipCenterStrip) {
    // Pass 2: Strip center noise zone [47, 53]
    const CENTER_LO = 47, CENTER_HI = 53;
    const nonCenter = xs.filter(x => x < CENTER_LO || x > CENTER_HI);
    const centerCount = xs.length - nonCenter.length;
    if (centerCount > xs.length * 0.10 && nonCenter.length >= minClusterSize * 2) {
      const hasLeft = nonCenter.some(x => x < CENTER_LO);
      const hasRight = nonCenter.some(x => x > CENTER_HI);
      if (hasLeft && hasRight) {
        result2 = buildResult(splitCluster(nonCenter));
      }
    }

    // Pass 3: Aggressive strip [44, 56]
    const WIDE_LO = 44, WIDE_HI = 56;
    const farFromCenter = xs.filter(x => x < WIDE_LO || x > WIDE_HI);
    const wideCount = xs.length - farFromCenter.length;
    if (wideCount > xs.length * 0.15 && farFromCenter.length >= minClusterSize * 2) {
      const hasLeft = farFromCenter.some(x => x < WIDE_LO);
      const hasRight = farFromCenter.some(x => x > WIDE_HI);
      if (hasLeft && hasRight) {
        result3 = buildResult(splitCluster(farFromCenter));
      }
    }
  }

  // Pick the best result: prefer fewer clusters (cleaner tracking),
  // fall back to more clusters if that's all we have.
  const candidates = [result1, result2, result3].filter(Boolean);
  if (candidates.length === 0) return null;

  candidates.sort((a, b) => {
    if (a.length !== b.length) return a.length - b.length;
    const order = [result1, result2, result3];
    return order.indexOf(a) - order.indexOf(b);
  });

  return candidates[0];
}

// Backward compat — old name delegates to new function
export function detectBimodalClusters(keyframes, gapThreshold = 12, minClusterSize = 3) {
  const clusters = detectPositionClusters(keyframes, gapThreshold, minClusterSize, 6);
  if (!clusters || clusters.length < 2) return null;
  // Return old format for any callers still using the bimodal shape
  return { left: clusters[0].center, right: clusters[clusters.length - 1].center,
           split: (clusters[0].center + clusters[clusters.length - 1].center) / 2 };
}

/**
 * Snap each keyframe to its nearest cluster center.
 * Works with any number of clusters (2, 3, 4, ...).
 *
 * Matches backend _snap_to_clusters() exactly for preview-export parity.
 *
 * @param {Array<{t: number, x: number}>} keyframes
 * @param {Array<{center: number}>} clusters - Cluster objects with center field
 * @returns {Array<{t: number, x: number}>}
 */
export function snapToClusters(keyframes, clusters) {
  return keyframes.map(kf => {
    let nearest = clusters[0].center;
    let nearestCount = clusters[0].count;
    let minDist = Math.abs(kf.x - nearest);
    for (let i = 1; i < clusters.length; i++) {
      const dist = Math.abs(kf.x - clusters[i].center);
      if (dist < minDist || (dist === minDist && clusters[i].count > nearestCount)) {
        minDist = dist;
        nearest = clusters[i].center;
        nearestCount = clusters[i].count;
      }
    }
    return { t: kf.t, x: nearest };
  });
}

/**
 * Compute AutoFlip-style bbox-aware subject_x for close-up shots.
 * When a face bbox is wider than the crop window, slot-average X is unreliable.
 * Instead, use the per-frame face center from face_positions data.
 *
 * @param {Object} scene - Scene with face_positions
 * @param {number} fallbackX - Slot-based subject_x to fall back to
 * @param {number|null} srcRatio - Source aspect ratio (e.g. 16/9)
 * @param {number|null} targetRatio - Target aspect ratio (e.g. 9/16)
 * @returns {number} Best crop center X (0-100)
 */
function bboxAwareSubjectX(scene, fallbackX, srcRatio, targetRatio) {
  if (!scene.face_positions?.length || !srcRatio || !targetRatio) return fallbackX;

  // Crop window width as % of source width
  const R = srcRatio / targetRatio;
  const cropWidthPct = R <= 1 ? 100 : (100 / R);

  // Find the speaking (or largest) face
  const speaking = scene.face_positions.find(f => f.is_speaking);
  const face = speaking || scene.face_positions.reduce((best, f) =>
    (f.w ?? 0) > (best.w ?? 0) ? f : best
  , scene.face_positions[0]);

  if (!face || typeof face.x !== 'number') return fallbackX;

  const faceCenterX = face.x; // already 0-100
  const faceWidthPct = face.w ?? 0; // 0-100

  // Only override when this is a close-up (face > 25% of source width)
  // — panel shots where face is ~8% wide should keep slot-snapped X.
  if (faceWidthPct < 25) return fallbackX;

  // Center the crop on the face, clamped so crop stays in frame
  const halfCrop = cropWidthPct / 2;
  return Math.max(halfCrop, Math.min(100 - halfCrop, faceCenterX));
}

/**
 * Build sorted keyframes from scenes for a clip range.
 *
 * Uses scenes both within and outside the clip range.  Scenes outside the
 * clip boundaries are used to interpolate accurate subject_x values at the
 * clip start/end, preventing a fallback to center (50) when no scenes fall
 * strictly within range.  Matches backend _build_subject_keyframes() logic.
 *
 * @param {Array} scenes - Scene objects with {timestamp, subject_x}
 * @param {number} clipStart - Clip start time in seconds
 * @param {number} clipEnd - Clip end time in seconds
 * @param {number|null} srcRatio - Source video aspect ratio (optional)
 * @param {number|null} targetRatio - Target crop aspect ratio (optional)
 * @returns {Array<{t: number, x: number}>} Sorted keyframes (t = seconds from clip start)
 */
export function buildSubjectKeyframes(scenes, clipStart, clipEnd, srcRatio = null, targetRatio = null) {
  if (!scenes?.length) return [{ t: 0, x: 50 }];

  const sorted = [...scenes].sort((a, b) => a.timestamp - b.timestamp);
  const before = sorted.filter((s) => s.timestamp < clipStart);
  const within = sorted.filter((s) => clipStart <= s.timestamp && s.timestamp <= clipEnd);
  const after = sorted.filter((s) => s.timestamp > clipEnd);

  // Prefer active_speaker_x (when AI detected who is talking) over generic subject_x
  const _sx = (s) => s.active_speaker_x ?? s.subject_x ?? 50;
  // precise_x: actual face nose_x from detection, not slot-snapped
  const _px = (s) => s.precise_x ?? _sx(s);

  const raw = within.map((s) => {
    const slotX = safeSubjectX(_sx(s), srcRatio, targetRatio);
    const closeupX = bboxAwareSubjectX(s, slotX, srcRatio, targetRatio);
    return {
      t: s.timestamp - clipStart,
      x: safeSubjectX(closeupX, srcRatio, targetRatio),
      px: safeSubjectX(_px(s), srcRatio, targetRatio),
    };
  });

  const interp = (tAbs, s1, s2) => {
    const dt = s2.timestamp - s1.timestamp;
    if (dt <= 0) return safeSubjectX(_sx(s1), srcRatio, targetRatio);
    const frac = Math.min(1, Math.max(0, (tAbs - s1.timestamp) / dt));
    return safeSubjectX(Math.round(
      _sx(s1) + (_sx(s2) - _sx(s1)) * frac
    ), srcRatio, targetRatio);
  };

  const clipDur = clipEnd - clipStart;

  // Compute accurate boundary value at t=0 (clipStart)
  if (raw.length === 0 || raw[0].t > 0) {
    let sx0;
    if (before.length && within.length) {
      sx0 = interp(clipStart, before[before.length - 1], within[0]);
    } else if (before.length && after.length && !within.length) {
      sx0 = interp(clipStart, before[before.length - 1], after[0]);
    } else if (before.length) {
      sx0 = safeSubjectX(_sx(before[before.length - 1]), srcRatio, targetRatio);
    } else if (within.length) {
      sx0 = safeSubjectX(_sx(within[0]), srcRatio, targetRatio);
    } else if (after.length) {
      sx0 = safeSubjectX(_sx(after[0]), srcRatio, targetRatio);
    } else {
      sx0 = 50;
    }
    raw.unshift({ t: 0, x: sx0 });
  }

  // Compute accurate boundary value at t=clipDur (clipEnd)
  if (clipDur > 0 && (raw.length === 0 || raw[raw.length - 1].t < clipDur)) {
    let sxEnd;
    if (after.length && within.length) {
      sxEnd = interp(clipEnd, within[within.length - 1], after[0]);
    } else if (before.length && after.length && !within.length) {
      sxEnd = interp(clipEnd, before[before.length - 1], after[0]);
    } else if (after.length) {
      sxEnd = safeSubjectX(_sx(after[0]), srcRatio, targetRatio);
    } else if (within.length) {
      sxEnd = safeSubjectX(_sx(within[within.length - 1]), srcRatio, targetRatio);
    } else if (before.length) {
      sxEnd = safeSubjectX(_sx(before[before.length - 1]), srcRatio, targetRatio);
    } else {
      sxEnd = 50;
    }
    raw.push({ t: clipDur, x: sxEnd });
  }

  return raw.length ? raw : [{ t: 0, x: 50 }];
}

/**
 * Detect large subject_x jumps between consecutive keyframes and insert
 * instant-jump keyframes at likely scene cuts.
 *
 * When subject_x changes by more than jumpThreshold between consecutive
 * keyframes, this is likely a scene cut — the subject didn't physically
 * move, the camera cut to a new shot.  Human editors cut-to instantly,
 * they never pan across a scene cut.
 *
 * Inserts a keyframe 1ms before the cut with the OLD position, so the
 * transition is truly instant — below one frame at any display rate.
 *
 * Matches backend _handle_scene_cuts() exactly for preview-export parity.
 *
 * @param {Array<{t: number, x: number}>} keyframes - Sorted keyframes
 * @param {number} jumpThreshold - Minimum subject_x delta to treat as a cut (default 15)
 * @returns {Array<{t: number, x: number}>} Keyframes with instant-cut transitions
 */
export function handleSceneCuts(keyframes, jumpThreshold = 15) {
  if (!keyframes || keyframes.length <= 1) return keyframes ? [...keyframes] : [];

  const result = [keyframes[0]];
  for (let i = 1; i < keyframes.length; i++) {
    const prev = result[result.length - 1];
    const cur = keyframes[i];
    const delta = Math.abs(cur.x - prev.x);

    if (delta >= jumpThreshold && (cur.t - prev.t) > 0.1) {
      // Large jump detected — insert instant cut
      // 1ms gap: below one frame at any frame rate, so smoothstep can't catch it
      const cutTime = Math.round((cur.t - 0.001) * 1000) / 1000;
      if (cutTime > prev.t) {
        result.push({ t: cutTime, x: prev.x }); // Hold old position until cut
      }
    }

    result.push({ t: cur.t, x: cur.x });
  }

  return result;
}

/**
 * Force instant cuts at camera shot boundaries.
 * Scene cuts (camera angle changes) should ALWAYS trigger instant reframe,
 * regardless of how much subject_x changed.
 */
export function injectShotBoundaryCuts(keyframes, sceneCuts, clipStart, clipEnd) {
  if (!sceneCuts || !sceneCuts.length || !keyframes || keyframes.length <= 1) {
    return keyframes ? [...keyframes] : [];
  }

  const clipDur = clipEnd - clipStart;
  // Convert to clip-relative time, filter to within clip bounds
  const clipCutTimes = sceneCuts
    .map(t => t - clipStart)
    .filter(t => t > 0.1 && t < clipDur - 0.1)
    .sort((a, b) => a - b);

  if (!clipCutTimes.length) return [...keyframes];

  const result = [...keyframes];
  let inserted = 0;

  for (const cutTime of clipCutTimes) {
    let insertIdx = result.length;
    for (let i = 0; i < result.length; i++) {
      if (result[i].t >= cutTime) { insertIdx = i; break; }
    }

    const beforeX = insertIdx > 0 ? result[insertIdx - 1].x : result[0].x;
    const afterX = insertIdx < result.length ? result[insertIdx].x : beforeX;

    if (Math.abs(afterX - beforeX) > 3) {
      const holdTime = Math.round((cutTime - 0.001) * 1000) / 1000;
      const prevT = insertIdx > 0 ? result[insertIdx - 1].t : 0;
      if (holdTime > prevT) {
        result.splice(insertIdx, 0,
          { t: holdTime, x: beforeX },
          { t: cutTime, x: afterX },
        );
        inserted += 2;
      }
    }
  }

  return result;
}

/**
 * Compress the range of subject_x values to prevent erratic swinging.
 *
 * If the full range of sx values exceeds maxRange, compress toward the
 * median so total motion stays within bounds. Preserves relative timing
 * and direction of motion — just reduces amplitude.
 *
 * When aspect ratios are provided, the maxRange is scaled up proportionally
 * to R (the magnification factor). At R=3.16 (16:9→9:16), the visible
 * crop window is only ~32% of the source width, so the subject can
 * legitimately span a wider sx range while still appearing within frame.
 *
 * Matches backend _compress_range() exactly for preview-export parity.
 *
 * @param {Array<{t: number, x: number}>} keyframes
 * @param {number} maxRange - Maximum allowed range of sx values (default 30)
 * @param {number|null} srcRatio - Source video aspect ratio (optional)
 * @param {number|null} targetRatio - Target crop aspect ratio (optional)
 * @returns {Array<{t: number, x: number}>}
 */
export function compressRange(keyframes, maxRange = 30, srcRatio = null, targetRatio = null) {
  if (!keyframes || keyframes.length <= 1) return keyframes ? [...keyframes] : [];

  // Scale maxRange based on aspect ratio magnification. When cropping to a
  // narrower aspect ratio (e.g. 16:9→9:16), the visible window is much
  // smaller than the source, so larger sx movement is needed to keep the
  // subject centered. Without this, compressRange squashes tracking to a
  // tiny band and the subject drifts out of frame.
  let effectiveMaxRange = maxRange;
  if (srcRatio && targetRatio) {
    const R = srcRatio / targetRatio;
    if (R > 1.01) {
      // Scale up: allow the full safe range as max motion
      const safeRange = computeSafeRange(srcRatio, targetRatio);
      effectiveMaxRange = Math.max(maxRange, safeRange.max - safeRange.min);
    }
  }

  const xs = keyframes.map(k => k.x);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const currentRange = maxX - minX;

  if (currentRange <= effectiveMaxRange) return [...keyframes.map(k => ({ ...k }))];

  // Compress toward median
  const sorted = xs.slice().sort((a, b) => a - b);
  const median = sorted[Math.floor(sorted.length / 2)];
  const scale = effectiveMaxRange / currentRange;

  return keyframes.map(k => ({
    t: k.t,
    x: Math.max(0, Math.min(100, median + (k.x - median) * scale)),
  }));
}

/**
 * Eliminate jittery micro-movements by snapping small changes to previous value.
 *
 * When aspect ratio info is provided, the threshold is computed dynamically
 * so it operates on VISIBLE crop movement (~3% of crop width) rather than
 * raw source-frame movement. This prevents the dead zone from being too
 * permissive at high R values (e.g. 16:9→9:16 where R=3.16).
 *
 * Matches backend _apply_dead_zone() exactly for preview-export parity.
 *
 * @param {Array<{t: number, x: number}>} keyframes - Sorted keyframes
 * @param {number} threshold - Minimum delta to allow movement (default 5)
 * @param {number|null} srcRatio - Source video aspect ratio (optional)
 * @param {number|null} targetRatio - Target crop aspect ratio (optional)
 * @returns {Array<{t: number, x: number}>} Keyframes with micro-movements removed
 */
export function applyDeadZone(keyframes, threshold = 5, srcRatio = null, targetRatio = null) {
  if (!keyframes || keyframes.length <= 1) return keyframes ? [...keyframes] : [];

  // We want about 3-4% of VISIBLE crop width as the dead zone
  const VISIBLE_THRESHOLD = 6;
  let effectiveThreshold = threshold;
  if (srcRatio && targetRatio) {
    const R = srcRatio / targetRatio;
    if (R > 1.01) {
      effectiveThreshold = Math.max(3, Math.round(VISIBLE_THRESHOLD * (R - 1) / R));
    }
  }

  // Two-threshold hysteresis: must exceed threshold to START tracking,
  // must drop below 40% to STOP tracking. This eliminates stutter.
  const engageThreshold = effectiveThreshold;
  const disengageThreshold = Math.max(1, Math.round(effectiveThreshold * 0.4));

  const result = [{ ...keyframes[0] }];
  let anchor = keyframes[0].x;
  let isTracking = false;

  for (let i = 1; i < keyframes.length; i++) {
    const cur = keyframes[i];
    const driftFromAnchor = Math.abs(cur.x - anchor);

    if (!isTracking) {
      if (driftFromAnchor >= engageThreshold) {
        isTracking = true;
        result.push({ t: cur.t, x: cur.x });
        anchor = cur.x;
      } else {
        result.push({ t: cur.t, x: anchor });
      }
    } else {
      if (driftFromAnchor <= disengageThreshold) {
        isTracking = false;
        result.push({ t: cur.t, x: anchor });
      } else {
        result.push({ t: cur.t, x: cur.x });
        anchor = cur.x;
      }
    }
  }

  return result;
}

/**
 * Insert synthetic keyframes to convert long interpolations into hold-then-snap.
 * Matches backend _insert_snap_transitions() exactly for preview-export parity.
 *
 * Without this, two keyframes at t=0,x=30 and t=10,x=70 produce a
 * 10-second slow pan. With this they become a ~150ms snap at the midpoint.
 */
export function insertSnapTransitions(keyframes, jumpThreshold = 15, snapDuration = 0.15) {
  if (!keyframes || keyframes.length <= 1) return keyframes ? [...keyframes] : [];

  const result = [keyframes[0]];
  for (let i = 1; i < keyframes.length; i++) {
    const { t: t0, x: x0 } = result[result.length - 1];
    const { t: t1, x: x1 } = keyframes[i];
    const dt = t1 - t0;
    const jump = Math.abs(x1 - x0);

    if (jump >= jumpThreshold && dt > snapDuration * 4) {
      const midT = (t0 + t1) / 2;
      const halfSnap = snapDuration / 2;
      result.push({ t: midT - halfSnap, x: x0 });
      result.push({ t: midT + halfSnap, x: x1 });
    }
    result.push(keyframes[i]);
  }
  return result;
}


/**
 * Hold-then-snap smoother for human-edited camera feel.
 *
 * Instead of continuously drifting toward the target (damped-lerp),
 * this holds the camera COMPLETELY STILL until the subject drifts far
 * enough to warrant a reframe, then snaps FAST (120-450ms) to the new
 * position with an ease-out curve (fast start, gentle landing).
 *
 * Pattern: HOLD → SNAP → HOLD → SNAP (never continuous drift)
 *
 * Matches backend _smooth_keyframes_bidirectional() exactly for preview-export parity.
 *
 * @param {Array<{t: number, x: number}>} keyframes - Sorted keyframes
 * @param {number} maxSpeed - Maximum subject_x units per second (default 22)
 * @param {number|null} srcRatio - Source video aspect ratio (optional)
 * @param {number|null} targetRatio - Target crop aspect ratio (optional)
 * @returns {Array<{t: number, x: number}>} Smoothed keyframes
 */
export function smoothKeyframesBidirectional(keyframes, maxSpeed = 22, srcRatio = null, targetRatio = null) {
  if (!keyframes || keyframes.length <= 1) return keyframes ? [...keyframes] : [];

  // Scale parameters for aspect ratio magnification
  let reframeThreshold = 4;   // Respond quickly — 4% drift triggers reframe
  let reframeDuration = 0.15; // Fast snap (seconds) — human-like decisive movement
  let effectiveMaxSpeed = maxSpeed;

  if (srcRatio && targetRatio) {
    const R = srcRatio / targetRatio;
    if (R > 1.5) {
      // Narrower crop = smaller movements are more visible, react even faster
      reframeThreshold = Math.max(2, Math.round(4 / Math.sqrt(R)));
      reframeDuration = Math.max(0.08, 0.15 / Math.sqrt(R));
      effectiveMaxSpeed = Math.min(100, maxSpeed * Math.sqrt(R));
    }
  }

  const dt_step = 0.016;
  const result = [{ t: keyframes[0].t, x: keyframes[0].x }];
  let holdPos = keyframes[0].x;       // Where the camera is holding
  let pos = keyframes[0].x;            // Current actual position
  let reframing = false;                // Are we mid-reframe?
  let reframeTarget = keyframes[0].x;
  let reframeProgress = 0;             // 0 to 1 progress through current reframe
  let reframeStartPos = keyframes[0].x;
  let lastDirection = 0;

  for (let i = 1; i < keyframes.length; i++) {
    const target = keyframes[i].x;
    const segDt = keyframes[i].t - keyframes[i - 1].t;

    if (segDt <= 0.002) {
      // Scene cut — instant snap, no smoothing
      pos = target;
      holdPos = target;
      reframing = false;
      reframeProgress = 0;
      lastDirection = 0;
      result.push({ t: keyframes[i].t, x: pos });
      continue;
    }

    // Track direction for hold-on-reversal logic
    const newDirection = target > holdPos ? 1 : (target < holdPos ? -1 : 0);

    // Decide whether to start a reframe
    const driftFromHold = Math.abs(target - holdPos);
    if (!reframing && driftFromHold >= reframeThreshold) {
      // Subject has drifted far enough — commit to a reframe
      reframing = true;
      reframeTarget = target;
      reframeStartPos = pos;
      reframeProgress = 0;
      // Adaptive pan speed: scale duration based on distance — stay fast
      const distance = Math.abs(target - pos);
      reframeDuration = Math.max(0.08, Math.min(0.25, 0.06 + distance * 0.004));
    } else if (reframing) {
      // Already reframing — update target if same direction, else keep current
      if (newDirection !== 0 && lastDirection !== 0 && newDirection !== lastDirection) {
        // Direction reversed mid-reframe — finish current reframe, don't chase
      } else {
        reframeTarget = target;
      }
    }

    if (newDirection !== 0) lastDirection = newDirection;

    // Simulate movement over this segment
    let simTime = 0;
    while (simTime < segDt) {
      const step = Math.min(dt_step, segDt - simTime);

      if (reframing) {
        // Move DECISIVELY toward target using ease-out (fast start, gentle end)
        reframeProgress += step / reframeDuration;

        if (reframeProgress >= 1.0) {
          // Reframe complete — lock into new hold
          pos = reframeTarget;
          holdPos = reframeTarget;
          reframing = false;
          reframeProgress = 0;
        } else {
          // Ease-out curve: 1 - (1-t)^3 — fast start, smooth deceleration
          const eased = 1 - Math.pow(1 - Math.min(1, reframeProgress), 3);
          pos = reframeStartPos + (reframeTarget - reframeStartPos) * eased;
        }
      }
      // When NOT reframing, pos stays at holdPos — camera is locked still

      simTime += step;
    }

    pos = Math.max(0, Math.min(100, pos));
    result.push({ t: keyframes[i].t, x: pos });
  }

  return result;
}

/**
 * Merge consecutive keyframes with similar values into holds.
 *
 * If several consecutive keyframes are within tolerance of each other,
 * snap them all to the first value — creating a visible 'rest' period
 * where the crop holds steady.
 *
 * Matches backend _merge_holds() exactly for preview-export parity.
 *
 * @param {Array<{t: number, x: number}>} keyframes - Sorted keyframes
 * @param {number} tolerance - Maximum delta to merge (default 3)
 * @returns {Array<{t: number, x: number}>} Keyframes with holds merged
 */
export function mergeHolds(keyframes, tolerance = 3) {
  if (!keyframes || keyframes.length <= 1) return keyframes ? [...keyframes] : [];

  const result = [keyframes[0]];
  for (let i = 1; i < keyframes.length; i++) {
    const cur = keyframes[i];
    const prevX = result[result.length - 1].x;
    if (Math.abs(cur.x - prevX) <= tolerance) {
      result.push({ t: cur.t, x: prevX }); // Hold at previous position
    } else {
      result.push({ t: cur.t, x: cur.x });
    }
  }

  return result;
}

/**
 * Full keyframe processing pipeline:
 *   build → compress range → dead zone → scene cuts → spring smooth → merge holds
 *
 * Matches the backend export_clip() pipeline exactly for preview-export parity.
 *
 * @param {Array} scenes - Scene objects with {timestamp, subject_x}
 * @param {number} clipStart - Clip start time in seconds
 * @param {number} clipEnd - Clip end time in seconds
 * @param {number|null} srcRatio - Source video aspect ratio (optional)
 * @param {number|null} targetRatio - Target crop aspect ratio (optional)
 * @returns {Array<{t: number, x: number}>} Fully processed keyframes
 */
export function processKeyframes(scenes, clipStart, clipEnd, srcRatio = null, targetRatio = null, transcript = null, sceneCuts = null, trackingMode = null) {
  // Gameplay mode: static center crop — crosshair is always at frame center in FPS games.
  // Skip all face-based tracking, clustering, and smoothing.
  if (trackingMode === 'gameplay') {
    console.log('[SubjectTracking] Gameplay mode — static center crop (subject_x=50)');
    return [{ t: 0, x: 50 }];
  }

  // ── Reframe-segment mode: backend produced motivated editorial segments ──
  // If the backend returned reframe-segment data (descriptions start with "[reframe:"),
  // skip ALL frontend smoothing/clustering — honor the segments as-is.
  if (scenes?.length > 0) {
    const reframeScenes = scenes.filter(s =>
      s.description && s.description.startsWith('[reframe:')
    );
    if (reframeScenes.length > 0) {
      const clipDur = clipEnd - clipStart;
      const sorted = [...reframeScenes]
        .sort((a, b) => a.timestamp - b.timestamp);

      // Build raw keyframes from reframe scenes, collapsing end-markers
      // Each segment emits a start scene at seg.start and an end-marker at
      // seg.end - 0.001. We only need the start scenes — the step-function
      // interpolation holds each value until the next keyframe naturally.
      const seen = new Set();
      const keyframes = [];
      for (const s of sorted) {
        const t = s.timestamp - clipStart;
        if (t < -0.5 || t > clipDur + 0.5) continue;
        // Collapse near-duplicate timestamps (end-markers at t-0.001)
        const tRounded = Math.round(t * 100) / 100;
        if (seen.has(tRounded)) continue;
        seen.add(tRounded);

        const sx = s.active_speaker_x ?? s.subject_x ?? 50;
        const safeSx = safeSubjectX(sx, srcRatio, targetRatio);
        // Parse reason, ease_in_ms, strategy, confidence, subject_source from description
        // Format: [reframe:reason:easeMs:strategy:confidence:subject_source]
        const match = s.description.match(/\[reframe:(\w+):?(\d+)?:?(\w+)?:?([\d.]+)?:?(\w+)?\]/);
        const reason = match?.[1] || 'hold';
        const easeMs = match?.[2] ? parseInt(match[2], 10) : 0;
        const strategy = match?.[3] || 'stationary';
        const confidence = match?.[4] ? parseFloat(match[4]) : null;
        const subjectSource = match?.[5] || null;
        keyframes.push({ t: Math.max(0, Math.min(clipDur, t)), x: safeSx, reason, easeMs, strategy, layoutMode: s.layout_mode, confidence, subjectSource });
      }

      // Deduplicate consecutive same-x entries. When removing, preserve the
      // transition keyframe (the one with easeMs > 0 or a new reason).
      const deduped = keyframes.length > 0 ? [keyframes[0]] : [];
      for (let i = 1; i < keyframes.length; i++) {
        if (keyframes[i].x !== deduped[deduped.length - 1].x) {
          deduped.push(keyframes[i]);
        }
      }
      // Ensure we have at least start and end keyframes
      if (deduped.length > 0 && deduped[0].t > 0) {
        deduped.unshift({ t: 0, x: deduped[0].x, easeMs: 0 });
      }
      if (deduped.length > 0 && deduped[deduped.length - 1].t < clipDur) {
        deduped.push({ t: clipDur, x: deduped[deduped.length - 1].x, easeMs: 0 });
      }
      const uniqueX = new Set(deduped.map(k => k.x));
      const strategyCounts = {};
      deduped.forEach(k => { strategyCounts[k.strategy || 'stationary'] = (strategyCounts[k.strategy || 'stationary'] || 0) + 1; });
      const strategyStr = Object.entries(strategyCounts).map(([k, v]) => `${k}=${v}`).join(', ');
      console.log(
        `[SubjectTracking] PHASE 1: reframe-segment mode, ${reframeScenes.length} segments, unique_x=${uniqueX.size}`,
        `(values: ${[...uniqueX].join(', ')}), strategies: {${strategyStr}}`
      );
      return deduped.length > 0 ? deduped : [{ t: 0, x: 50 }];
    }
  }

  // ── PHASE 0: Build raw keyframes ──
  const raw = buildSubjectKeyframes(scenes, clipStart, clipEnd, srcRatio, targetRatio);
  if (!raw || raw.length === 0) return [{ t: 0, x: 50 }];
  if (raw.length === 1) return raw;

  // ── PHASE 1: Detect position clusters (N speakers from visual data) ──
  // This works WITHOUT audio diarization — catches multi-speaker scenarios
  // even when Whisper only detects 1 speaker, by finding position clusters
  // in the scene analysis subject_x values.
  const isDenseData = raw.length >= 100;  // Per-second dense face detection
  // For dense data, skip center-stripping — values come from actual face positions,
  // not noisy AI defaults. A speaker at 53% is real, not center noise.
  const clusters = detectPositionClusters(raw, 10, 2, isDenseData);

  // Detect continuous motion: if dense data has many unique x values (>10),
  // the backend classified this as continuous (not slot-snapped). Skip clustering
  // to preserve the raw face positions for smooth tracking.
  const uniqueX = new Set(raw.map(k => k.x));
  const isContinuousMotion = isDenseData && uniqueX.size > 10;

  console.log(
    `[SubjectTracking] PHASE 1: raw=${raw.length} keyframes, clusters=${clusters ? clusters.length : 'null'}, isDense=${isDenseData}, continuous=${isContinuousMotion}, uniqueX=${uniqueX.size}`,
    clusters ? clusters.map(c => `center=${c.center} count=${c.count}`).join(', ') : 'none',
  );

  if (clusters && clusters.length >= 2 && !isContinuousMotion) {
    // Clamp cluster centers to the safe range for the target aspect ratio
    // so that snapToClusters produces values already within bounds.
    // Without this, a cluster center at 24 might be snapped to, but then
    // bounds enforcement shifts it to 21 — misaligning with the face.
    if (srcRatio && targetRatio) {
      const safeRange = computeSafeRange(srcRatio, targetRatio);
      for (const c of clusters) {
        c.center = Math.max(safeRange.min, Math.min(safeRange.max, c.center));
      }
    }

    // Multi-position mode: snap to cluster centers, then use scene cuts for instant jumps.
    // NO smoothing — speaker/position changes must be instant snaps, not pans.
    // Dense data from backend slot-center-snapping benefits from cluster snap too:
    // reinforces stability and ensures the Phase 1 instant-snap path is used
    // instead of Phase 3 smoothing which creates off-center intermediate values.
    const snapped = snapToClusters(raw, clusters);
    // Carry precise_x from the original raw keyframe
    for (let i = 0; i < snapped.length; i++) {
      snapped[i].px = raw[i]?.px ?? snapped[i].x;
    }

    // Remove consecutive duplicates (same speaker holding) to clean up
    const deduped = [snapped[0]];
    for (let i = 1; i < snapped.length; i++) {
      if (snapped[i].x !== deduped[deduped.length - 1].x) {
        deduped.push(snapped[i]);
      } else if (i === snapped.length - 1) {
        deduped.push({ t: snapped[i].t, x: deduped[deduped.length - 1].x });
      }
    }

    // ── Anti-jitter: filter detection noise ──
    // For dense data: only remove isolated single-sample blips (≤1s) where
    // both neighbors have the same position (face detection artifact, not a
    // real speaker change). Real speaker changes — even brief 1s ones — are
    // preserved for fluid responsive reframing.
    // For sparse AI data: use minimum hold to suppress noisy scene descriptions.
    if (isDenseData) {
      if (deduped.length >= 3) {
        let noiseRemoved = 0;
        let ni = 1;
        while (ni < deduped.length - 1) {
          const holdDuration = deduped[ni + 1].t - deduped[ni].t;
          const isIsolatedBlip = (
            deduped[ni - 1].x === deduped[ni + 1].x &&  // neighbors agree
            deduped[ni].x !== deduped[ni - 1].x &&       // this one disagrees
            holdDuration <= 1.0                            // single sample (≤1s)
          );
          if (isIsolatedBlip) {
            deduped.splice(ni, 1);
            noiseRemoved++;
          } else {
            ni++;
          }
        }
        if (noiseRemoved > 0) {
          console.log(`[SubjectTracking] Noise filter: removed ${noiseRemoved} isolated detection blips (≤1s)`);
        }
      }
    } else {
      // Sparse AI data: use minimum hold duration to suppress noise
      const MIN_HOLD_SECONDS = 2.0;
      if (deduped.length >= 3) {
        let i = 1;
        while (i < deduped.length - 1) {
          const nextT = deduped[i + 1].t;
          const holdDuration = nextT - deduped[i].t;
          if (holdDuration < MIN_HOLD_SECONDS && deduped[i - 1].x === deduped[i + 1].x) {
            deduped.splice(i, 1);
          } else {
            i++;
          }
        }
      }
      if (deduped.length >= 3) {
        let i = 1;
        while (i < deduped.length - 1) {
          const nextT = deduped[i + 1].t;
          const holdDuration = nextT - deduped[i].t;
          if (holdDuration < MIN_HOLD_SECONDS) {
            deduped[i].x = deduped[i - 1].x;
            if (deduped[i].x === deduped[i - 1].x) {
              deduped.splice(i, 1);
            } else {
              i++;
            }
          } else {
            i++;
          }
        }
      }
    }

    // ── Camera mode selection (Google AutoFlip architecture) ──
    // For static multi-speaker content (podcasts, interviews), use STATIONARY mode
    // with extended hold times. Only use TRACKING if faces actually move.
    const clusterVariances = clusters.map(c => {
      const memberKfs = raw.filter(kf => {
        const nearest = clusters.reduce((best, cl) =>
          Math.abs(kf.x - cl.center) < Math.abs(kf.x - best.center) ? cl : best
        );
        return nearest === c;
      });
      if (memberKfs.length < 2) return 0;
      return Math.sqrt(
        memberKfs.reduce((sum, kf) => sum + (kf.x - c.center) ** 2, 0) / memberKfs.length
      );
    });
    const isStaticContent = Math.max(...clusterVariances) < 5;
    // STATIONARY MODE: for sparse AI data only, extend minimum hold for stability.
    // Dense data tracks every speaker change responsively — no suppression.
    if (!isDenseData && isStaticContent && deduped.length >= 3) {
      const STATIC_MIN_HOLD = 3.0;
      let si = 1;
      while (si < deduped.length - 1) {
        const holdDuration = deduped[si + 1].t - deduped[si].t;
        if (holdDuration < STATIC_MIN_HOLD) {
          deduped[si] = { t: deduped[si].t, x: deduped[si - 1].x };
          if (deduped[si].x === deduped[si - 1].x) {
            deduped.splice(si, 1);
          } else {
            si++;
          }
        } else {
          si++;
        }
      }
    }

    // Ensure start and end keyframes
    if (deduped[0].t > 0) {
      deduped.unshift({ t: 0, x: deduped[0].x });
    }
    const clipDur = clipEnd - clipStart;
    if (deduped[deduped.length - 1].t < clipDur) {
      deduped.push({ t: clipDur, x: deduped[deduped.length - 1].x });
    }

    // ── Fix initial snap: don't start at center default ──
    // If the first keyframe is in the center noise zone (44-56), it's likely
    // a title card or default. Snap to the first non-center value's cluster.
    // SKIP for dense data — center values are real speaker positions, not noise.
    if (!isDenseData && deduped.length > 0 && deduped[0].x >= 44 && deduped[0].x <= 56) {
      const firstReal = raw.find(kf => kf.x < 44 || kf.x > 56);
      if (firstReal) {
        let nearestCenter = clusters[0].center;
        let minDist = Math.abs(firstReal.x - nearestCenter);
        for (let c = 1; c < clusters.length; c++) {
          const dist = Math.abs(firstReal.x - clusters[c].center);
          if (dist < minDist) { minDist = dist; nearestCenter = clusters[c].center; }
        }
        deduped[0].x = nearestCenter;
      }
    }

    // handleSceneCuts inserts 1ms instant-jump transitions at speaker changes
    // (delta between clusters is always > 15, so every change triggers an instant cut)
    let afterCuts = handleSceneCuts(deduped);
    afterCuts = injectShotBoundaryCuts(afterCuts, sceneCuts, clipStart, clipEnd);

    // Final bounds enforcement
    let result;
    if (srcRatio && targetRatio) {
      const range = computeSafeRange(srcRatio, targetRatio);
      result = afterCuts.map(kf => ({
        t: kf.t,
        x: Math.max(range.min, Math.min(range.max, Math.round(kf.x))),
      }));
    } else {
      result = afterCuts.map(kf => ({
        t: kf.t,
        x: Math.max(0, Math.min(100, Math.round(kf.x))),
      }));
    }

    // ── Post-bounds blip filter (sparse AI data only) ──
    // For sparse data, remove sub-threshold holds that create visible micro-jumps.
    // For dense data, all speaker changes are preserved — noise was already
    // filtered by the detection-noise filter above.
    if (!isDenseData && result.length >= 3) {
      const POST_MIN_HOLD = 2.0;
      let beforeCount = result.length;
      let ri = 1;
      while (ri < result.length - 1) {
        const hold = result[ri + 1].t - result[ri].t;
        if (hold > 0.01 && hold < POST_MIN_HOLD && result[ri - 1].x === result[ri + 1].x) {
          result.splice(ri, 1);
        } else {
          ri++;
        }
      }
      ri = 1;
      while (ri < result.length - 1) {
        const hold = result[ri + 1].t - result[ri].t;
        if (hold > 0.01 && hold < POST_MIN_HOLD) {
          result[ri].x = result[ri - 1].x;
          if (result[ri].x === result[ri - 1].x) {
            result.splice(ri, 1);
          } else {
            ri++;
          }
        } else {
          ri++;
        }
      }
      if (result.length < beforeCount) {
        console.log(`[SubjectTracking] Post-bounds filter: ${beforeCount} → ${result.length} keyframes (removed ${beforeCount - result.length} blips < ${POST_MIN_HOLD}s)`);
      }
    }

    // ── Fix leading center keyframes after bounds enforcement ──
    // SKIP for dense data — center values are real speaker positions, not noise.
    if (!isDenseData && result.length > 0) {
      const firstRealKf = raw.find(kf => kf.x < 44 || kf.x > 56);
      if (firstRealKf) {
        let bestCenter = clusters[0].center;
        let bestDist = Math.abs(firstRealKf.x - bestCenter);
        for (const c of clusters) {
          const d = Math.abs(firstRealKf.x - c.center);
          if (d < bestDist) { bestDist = d; bestCenter = c.center; }
        }
        const safeCenter = srcRatio && targetRatio
          ? Math.max(computeSafeRange(srcRatio, targetRatio).min,
                     Math.min(computeSafeRange(srcRatio, targetRatio).max, bestCenter))
          : bestCenter;
        for (let j = 0; j < result.length; j++) {
          if (result[j].x >= 44 && result[j].x <= 56) {
            result[j].x = safeCenter;
          } else {
            break;
          }
        }
      }
    }

    // ── Per-cluster precise face centering ──
    // For dense data: the backend already slot-center-snaps to stable positions
    // computed from hundreds of face detections. No frontend override needed.
    // The px values from raw keyframes include noisy sparse AI scenes that
    // pull medians away from correct positions (e.g., cluster 24 → median 15).
    // For sparse data: could apply per-cluster median, but not needed since
    // sparse data uses Phase 3 smoothing instead of Phase 1 clustering.
    if (isDenseData && clusters) {
      // Log cluster positions for debugging (no override applied)
      console.log('[SubjectTracking] Dense data: using cluster centers directly (no px override)',
        clusters.map(c => c.center));
    }

    // ── Snap keyframe timestamps to dense sample boundaries ──
    // Dense face data has ~1s intervals. Align transition keyframes to the
    // nearest raw sample time so the crop changes at the exact moment the
    // speaker changes, not 1-2 frames before or after.
    if (isDenseData && result.length >= 2) {
      for (let ki = 1; ki < result.length; ki++) {
        // Skip the first keyframe (t=0) and 1ms transition pairs
        if (result[ki].t <= 0.01) continue;
        if (ki > 0 && result[ki].t - result[ki - 1].t <= 0.01) continue;
        // Find nearest raw keyframe timestamp
        let bestRaw = raw[0];
        let bestDist = Math.abs(result[ki].t - raw[0].t);
        for (const r of raw) {
          const d = Math.abs(result[ki].t - r.t);
          if (d < bestDist) { bestDist = d; bestRaw = r; }
        }
        // Only snap if within 1s of a raw sample (don't move far)
        if (bestDist > 0 && bestDist <= 1.0) {
          result[ki].t = bestRaw.t;
        }
      }
    }

    // ── QA validation: fix extended center holds and missing instant cuts ──
    const qa = validateTracking(result, clusters, clipEnd - clipStart, isDenseData);
    if (!qa.passed) {
      qa.warnings.forEach(w => console.log(`[SubjectTracking] ${w}`));
      return qa.fixedKeyframes;
    }

    return result;
  }

  // ── PHASE 2: Try speaker-aware tracking (requires 2+ speakers in transcript) ──
  if (transcript?.length && scenes?.length) {
    const speakerMap = buildSpeakerPositionMap(scenes, transcript);
    if (Object.keys(speakerMap).length >= 2) {
      const speakerKf = buildSpeakerKeyframes(transcript, speakerMap, clipStart, clipEnd, srcRatio, targetRatio);
      if (speakerKf && speakerKf.length >= 2) {
        const afterCuts = handleSceneCuts(speakerKf);
        let result;
        if (srcRatio && targetRatio) {
          const range = computeSafeRange(srcRatio, targetRatio);
          result = afterCuts.map(kf => ({ t: kf.t, x: Math.max(range.min, Math.min(range.max, Math.round(kf.x))) }));
        } else {
          result = afterCuts.map(kf => ({ t: kf.t, x: Math.max(0, Math.min(100, Math.round(kf.x))) }));
        }
        if (result.length > 1) {
          const xs = result.map(kf => kf.x);
          if (Math.max(...xs) - Math.min(...xs) >= 5) return result;
        }
      }
    }
  }

  // ── PHASE 3: Single-subject tracking (original pipeline) ──
  // Sparse data detection: when we have very few keyframes (≤ 4),
  // relax pipeline thresholds so the little tracking data we have
  // doesn't get killed by dead zones and convergence checks.
  // Dense data (100+ scenes = per-second tracking) gets special handling:
  // - NO compression (positions are already speaker-accurate from face detection)
  // - Minimal dead zone (just filter sub-pixel jitter)
  // - Higher smooth speed (allow fast speaker transitions)
  const isSparse = raw.length <= 4;
  const isDense = raw.length >= 100;  // Per-second synthetic scenes from dense face detection
  const deadZoneThreshold = isSparse ? 3 : (isDense ? 2 : 5);
  const compressMaxRange = isSparse ? 60 : (isDense ? 100 : 30);
  const smoothMaxSpeed = isSparse ? 30 : (isDense ? 80 : 22);
  const holdTolerance = isSparse ? 2 : (isDense ? 2 : 3);

  // For dense data, skip compression entirely — the face positions are pixel-accurate
  // from backend speaker-aware detection. Compressing toward median pulls all positions
  // toward the dominant speaker, causing off-center framing for other speakers.
  const afterCompress = isDenseData ? [...raw.map(k => ({ ...k }))] : compressRange(raw, compressMaxRange, srcRatio, targetRatio);
  const afterDeadZone = applyDeadZone(afterCompress, deadZoneThreshold, srcRatio, targetRatio);
  let afterCuts3 = handleSceneCuts(afterDeadZone);
  afterCuts3 = injectShotBoundaryCuts(afterCuts3, sceneCuts, clipStart, clipEnd);
  // Insert hold-then-snap transitions for large jumps before smoothing
  const afterSnaps = insertSnapTransitions(afterCuts3);
  const afterSmooth = smoothKeyframesBidirectional(afterSnaps, smoothMaxSpeed, srcRatio, targetRatio);
  const afterHolds = mergeHolds(afterSmooth, holdTolerance);

  // Final bounds enforcement — ensure every keyframe x is clamped to [0, 100]
  // and within the safe range for the aspect ratio. This prevents any pipeline
  // stage from producing values that would push the crop off-screen.
  let result;
  if (srcRatio && targetRatio) {
    const range = computeSafeRange(srcRatio, targetRatio);
    result = afterHolds.map(kf => ({
      t: kf.t,
      x: Math.max(range.min, Math.min(range.max, Math.round(kf.x))),
    }));
  } else {
    result = afterHolds.map(kf => ({
      t: kf.t,
      x: Math.max(0, Math.min(100, Math.round(kf.x))),
    }));
  }

  // Check for near-convergence: collapse to static to avoid jitter.
  // Scale threshold by aspect ratio and data density.
  if (result.length > 1) {
    const finalXs = result.map(kf => kf.x);
    const minX = Math.min(...finalXs);
    const maxX = Math.max(...finalXs);
    const R = (srcRatio && targetRatio) ? srcRatio / targetRatio : 1;
    let convergenceThreshold = R > 1.5 ? Math.max(2, Math.round(5 / R)) : 5;
    // With sparse data, even small differences are meaningful — lower threshold
    if (isSparse) convergenceThreshold = Math.max(1, Math.round(convergenceThreshold * 0.6));
    if (maxX - minX < convergenceThreshold) {
      let staticX;
      if (result.length <= 1) {
        staticX = result[0].x;
      } else {
        const clipMid = result[Math.floor(result.length / 2)].t;
        let bestIdx = 0;
        let bestDist = Infinity;
        for (let j = 0; j < result.length; j++) {
          const dist = Math.abs(result[j].t - clipMid);
          if (dist < bestDist) { bestDist = dist; bestIdx = j; }
        }
        staticX = result[bestIdx].x;
      }
      return [{ t: 0, x: staticX }];
    }
  }
  return result;
}

/**
 * Compute a static subject_x value for a clip using interpolation from
 * nearby scenes.  Used as the fallback when dynamic keyframes aren't
 * available.  Matches backend clip_subject_x computation.
 *
 * @param {Array} scenes - All scene objects with {timestamp, subject_x}
 * @param {number} clipStart - Clip start time in seconds
 * @param {number} clipEnd - Clip end time in seconds
 * @returns {number} Subject x position (0-100)
 */
export function computeClipSubjectX(scenes, clipStart, clipEnd) {
  if (!scenes?.length) return 50;

  const sorted = [...scenes].sort((a, b) => a.timestamp - b.timestamp);
  const inRange = sorted.filter((s) => clipStart <= s.timestamp && s.timestamp <= clipEnd);

  if (inRange.length > 0) {
    return safeSubjectX(
      inRange.reduce((sum, s) => sum + (s.subject_x ?? 50), 0) / inRange.length
    );
  }

  // No in-range scenes — interpolate from nearest boundary scenes
  const before = sorted.filter((s) => s.timestamp < clipStart);
  const after = sorted.filter((s) => s.timestamp > clipEnd);
  const nb = before.length ? before[before.length - 1] : null;
  const na = after.length ? after[0] : null;

  if (nb && na) {
    const mid = (clipStart + clipEnd) / 2;
    const dt = na.timestamp - nb.timestamp;
    if (dt > 0) {
      const frac = (mid - nb.timestamp) / dt;
      return safeSubjectX((nb.subject_x ?? 50) + ((na.subject_x ?? 50) - (nb.subject_x ?? 50)) * frac);
    }
    return safeSubjectX(nb.subject_x ?? 50);
  }
  if (nb) return safeSubjectX(nb.subject_x ?? 50);
  if (na) return safeSubjectX(na.subject_x ?? 50);
  return 50;
}

/**
 * Interpolate subject_x at a given time using keyframes.
 *
 * Smoothstep (cubic Hermite: 3t² - 2t³) easing between adjacent
 * keyframes so the virtual camera pans like a human operator — no
 * linear ramping, no jitter, no dead holds on sparse data. Scene cuts
 * are preserved as instant snaps through two mechanisms:
 *
 *   1. ``handleSceneCuts`` / ``injectShotBoundaryCuts`` insert 1 ms
 *      transition pairs ``(t-0.001, old_x) → (t, new_x)`` around hard
 *      cuts. Smoothstep over a 1 ms window is effectively a step, so
 *      cuts remain instantaneous.
 *   2. A keyframe can carry ``snap: true`` to force an explicit step at
 *      that boundary regardless of its time delta to the next keyframe.
 *      The dense backend subject_track uses this at shot boundaries.
 *
 * Matches backend ``_build_crop_x_expr`` smoothstep exactly for
 * preview-export parity.
 *
 * @param {Array<{t: number, x: number, snap?: boolean}>} keyframes - Sorted keyframes
 * @param {number} t - Time in seconds (relative to clip start)
 * @returns {number} Interpolated subject_x (0-100)
 */
export function interpolateSubjectX(keyframes, t) {
  if (!keyframes?.length) return 50;
  if (keyframes.length === 1) return keyframes[0].x;

  // Clamp before first / after last
  if (t <= keyframes[0].t) return keyframes[0].x;
  if (t >= keyframes[keyframes.length - 1].t) return keyframes[keyframes.length - 1].x;

  // Find the bracketing keyframes [i, i+1] such that kf[i].t <= t < kf[i+1].t.
  // Linear scan backward — typical keyframe counts are small (tens to low
  // hundreds) and this runs every video frame, so a cache-friendly scan
  // beats bisect overhead.
  let left = null;
  let right = null;
  for (let i = keyframes.length - 1; i >= 0; i--) {
    if (keyframes[i].t <= t) {
      left = keyframes[i];
      right = keyframes[i + 1] || left;
      break;
    }
  }
  if (!left) return keyframes[0].x;
  if (!right || right === left) return left.x;

  const span = right.t - left.t;
  if (span <= 0) return right.x;
  // Snap keyframes: caller explicitly marked this boundary as a hard
  // cut — hold the left value and only jump when the playhead crosses
  // the right keyframe.
  if (right.snap) return left.x;

  const u = Math.max(0, Math.min(1, (t - left.t) / span));
  const e = u * u * (3 - 2 * u); // smoothstep (cubic Hermite)
  return left.x + (right.x - left.x) * e;
}

/**
 * Build preview keyframes directly from the backend's dense subject
 * track (``JobResult.subject_track``). The track is already at ~2 Hz
 * with the face → saliency → object → scene fallback cascade applied
 * and light EMA smoothing, so the only frontend work is:
 *
 *   1. Clamp into the clip range ``[clipStart, clipEnd]``.
 *   2. Apply the same aspect-ratio safety margin (``safeSubjectX``) the
 *      sparse path uses, so the crop window never exposes pillarbox bars.
 *   3. Drop samples that round to the same x as the previous sample —
 *      smoothstep handles "no motion" naturally and this shrinks the
 *      payload that gets shipped to the export endpoint.
 *   4. Mark large deltas as ``snap`` so hard cuts stay crisp.
 *
 * @param {Array<{t: number, x: number, y?: number, conf?: number, source?: string}>} track
 * @param {number} clipStart - Seconds (wall-clock) at which the clip begins
 * @param {number} clipEnd - Seconds (wall-clock) at which the clip ends
 * @param {number|null} srcRatio - Source aspect ratio (w/h), used for clamping
 * @param {number|null} targetRatio - Target aspect ratio (w/h), used for clamping
 * @returns {Array<{t: number, x: number, snap?: boolean, source?: string}>}
 */
export function buildKeyframesFromSubjectTrack(track, clipStart, clipEnd, srcRatio = null, targetRatio = null) {
  if (!Array.isArray(track) || track.length === 0) return null;
  const clipDur = Math.max(0, (clipEnd ?? 0) - (clipStart ?? 0));
  if (clipDur <= 0) return null;

  // Select samples inside the clip window. The track is stored in
  // wall-clock seconds; keyframes are clip-relative.
  const scoped = track
    .filter(p => p && typeof p.t === 'number' && typeof p.x === 'number')
    .map(p => ({
      t: p.t - clipStart,
      x: safeSubjectX(p.x, srcRatio, targetRatio),
      source: p.source,
      conf: p.conf ?? 0.5,
    }))
    .filter(p => p.t >= -0.25 && p.t <= clipDur + 0.25)
    .sort((a, b) => a.t - b.t);

  if (scoped.length === 0) return null;

  // Clamp edges to [0, clipDur] so the preview's interpolateSubjectX
  // doesn't hold a sample that's "before the start" of the clip.
  scoped[0].t = Math.max(0, scoped[0].t);
  scoped[scoped.length - 1].t = Math.min(clipDur, scoped[scoped.length - 1].t);

  // Drop samples whose x is within 1 pct of the previous and whose
  // timing is close — smoothstep handles the rest. Large jumps (>20
  // pct) are flagged as snaps so cuts stay instant regardless of the
  // time gap to the next keyframe.
  const out = [scoped[0]];
  for (let i = 1; i < scoped.length; i++) {
    const prev = out[out.length - 1];
    const cur = scoped[i];
    const dx = Math.abs(cur.x - prev.x);
    const dt = cur.t - prev.t;
    if (dx < 1.0 && dt < 1.5) continue; // redundant sample
    if (dx > 20) cur.snap = true;       // hard cut — preserve the step
    out.push(cur);
  }

  // Guarantee a keyframe at t=0 and t=clipDur so interpolateSubjectX
  // never falls through to a stale value at the boundaries.
  if (out[0].t > 0.01) out.unshift({ t: 0, x: out[0].x, source: 'hold_back' });
  if (out[out.length - 1].t < clipDur - 0.01) {
    out.push({ t: clipDur, x: out[out.length - 1].x, source: 'hold_fwd' });
  }

  return out;
}

// Keep old smoothKeyframes export for backward compatibility (unused but safe)
export { smoothKeyframesBidirectional as smoothKeyframes };

/**
 * Convert subject_x (0-100) to a CSS objectPosition percentage that
 * centers the subject in the cropped frame.
 *
 * With objectFit: cover, objectPosition X% aligns the X% point of the
 * content with the X% point of the container.
 *
 * R = srcRatio / targetRatio
 * centerPct = (R * sx - 50) / (R - 1)
 *
 * Hard-clamped to [EDGE_GUARD, 100-EDGE_GUARD] to prevent exposing
 * baked-in pillarboxing from the source video. The backend uses
 * FFmpeg cropdetect to strip bars during export, but the preview
 * plays the raw source. Full-frame coverage > subject centering.
 *
 * @param {number} sx - Subject x position (0-100)
 * @param {number} srcRatio - Source video aspect ratio
 * @param {number} targetRatio - Target crop aspect ratio
 * @returns {number} CSS objectPosition percentage
 */
export function subjectXToCenterPct(sx, srcRatio, targetRatio) {
  const R = srcRatio / targetRatio;
  if (R <= 1.01) return Math.max(0, Math.min(100, sx));
  const pct = (R * sx - 50) / (R - 1);
  // Clamp to [0, 100] only — no edge guard. The upstream safeSubjectX()
  // pipeline already constrains sx to the safe range for the aspect ratio.
  // Adding an edge guard here breaks preview-export parity because the
  // backend _center_crop_offset() clamps to [0, max_offset] with no
  // equivalent edge guard.
  return Math.max(0, Math.min(100, pct));
}

/**
 * Check if keyframes represent dynamic motion (more than one unique x value).
 *
 * @param {Array<{t: number, x: number}>} keyframes
 * @returns {boolean}
 */
export function isDynamic(keyframes) {
  if (!keyframes || keyframes.length <= 1) return false;
  const first = keyframes[0].x;
  return keyframes.some((kf) => kf.x !== first);
}

/**
 * Compute average face Y center from scenes within a clip range.
 * Uses face_positions[].y data from dense face detection when available.
 *
 * @param {Array} scenes - Scene objects with face_positions
 * @param {number} clipStart - Clip start time
 * @param {number} clipEnd - Clip end time
 * @returns {number} Average face Y position (0-100), or 50 if no data
 */
export function computeFaceYCenter(scenes, clipStart, clipEnd) {
  if (!scenes?.length) return 50;
  const ys = [];
  for (const s of scenes) {
    const t = s.timestamp;
    if (t < clipStart || t > clipEnd) continue;
    if (s.face_positions?.length) {
      // Prefer speaking face, else largest face
      const speaking = s.face_positions.find(f => f.is_speaking);
      const face = speaking || s.face_positions[0];
      if (face && typeof face.y === 'number') {
        ys.push(face.y);
      }
    }
  }
  if (ys.length === 0) return 50;
  return ys.reduce((a, b) => a + b, 0) / ys.length;
}

/**
 * Convert face Y center percentage to CSS objectPosition Y percentage.
 * Matches backend _compute_face_y_offset() logic: places face at 38% of
 * crop height (rule of thirds / broadcast framing).
 *
 * For horizontal crops (e.g. 16:9→9:16), vertical magnification R_v > 1
 * means the crop is taller than the source visible area, so face Y
 * positioning matters. For vertical crops of landscape video, R_v < 1
 * (crop is shorter than source), which is when Y repositioning helps most.
 *
 * @param {number} faceY - Face Y center as 0-100 percentage of source height
 * @param {number} srcRatio - Source aspect ratio (e.g. 16/9)
 * @param {number} targetRatio - Target aspect ratio (e.g. 9/16)
 * @param {number} targetFacePosition - Where face should be in crop (0.38 = upper third)
 * @returns {number} CSS objectPosition Y percentage (0-100)
 */
export function faceYToCenterPct(faceY, srcRatio, targetRatio, targetFacePosition = 0.38) {
  // Vertical magnification: how much taller the crop is relative to source
  const R_v = (targetRatio > 0 && srcRatio > 0)
    ? (srcRatio / targetRatio)  // For 16:9→9:16: R=3.16 (crop is much narrower but same height)
    : 1;

  // When target is taller than source (portrait from landscape), there's no vertical
  // overflow — objectFit:cover handles it. When target is wider (landscape from portrait),
  // there IS vertical overflow and we need to position Y.
  // Actually: objectFit:cover scales to fill, so for 16:9→9:16:
  //   - Source 16:9 scaled to fill 9:16 container: width matches, height overflows
  //   - The video is scaled up so its width fills the container
  //   - Height overflows: (sourceH * scale) > containerH
  //   - So vertical Y positioning DOES matter
  // R_v for vertical overflow = sourceH_scaled / containerH
  // When crop is narrower: the video is scaled by containerW/sourceW
  // Scaled height = sourceH * (containerW/sourceW) = containerW / srcRatio
  // Container height = containerW / targetRatio
  // R_v = (containerW/srcRatio) / (containerW/targetRatio) = targetRatio/srcRatio
  const verticalR = targetRatio / srcRatio;

  if (verticalR <= 1.01) {
    // No vertical overflow — any Y position is fine, default to center
    return 50;
  }

  // Convert faceY to objectPosition Y percentage
  // We want the face at targetFacePosition (38%) of the visible crop
  // objectPosition Y = percentage of the overflow region
  // Similar to horizontal: pct = (verticalR * faceY - targetFacePosition*100) / (verticalR - 1)
  // But we want face at 38% of crop, not 50%, so adjust:
  const targetPct = targetFacePosition * 100; // 38
  const pct = (verticalR * faceY - targetPct) / (verticalR - 1);
  return Math.max(0, Math.min(100, pct));
}


/**
 * QA validation for subject tracking output.
 * Checks that:
 *  - No extended sequences stuck at center (50 ± 3) when clusters exist
 *  - All large position changes have instant-cut markers (no pans through dead space)
 *
 * @param {Array<{t: number, x: number}>} keyframes
 * @param {Array<{center: number, count: number}>|null} clusters
 * @param {number} clipDuration
 * @param {boolean} isDenseData - When true, center positions are trustworthy (from face detection, not AI defaults)
 * @returns {{ passed: boolean, warnings: string[], fixedKeyframes: Array }}
 */
export function validateTracking(keyframes, clusters, clipDuration, isDenseData = false) {
  const warnings = [];
  const fixed = keyframes.map(kf => ({ ...kf }));

  if (!keyframes || keyframes.length === 0) {
    return { passed: false, warnings: ['No keyframes'], fixedKeyframes: [{ t: 0, x: 50 }] };
  }

  // Check 1: No extended center holds when multi-position data exists
  // SKIP for dense data: when face detection provides the positions, a center value
  // IS the real face position (e.g. slot at 53%), not a vision model default.
  if (clusters && clusters.length >= 2 && !isDenseData) {
    for (let i = 0; i < fixed.length - 1; i++) {
      const hold = fixed[i + 1].t - fixed[i].t;
      if (fixed[i].x >= 47 && fixed[i].x <= 53 && hold > 3.0) {
        const prev = i > 0 ? fixed[i - 1].x : null;
        if (prev !== null && clusters.some(c => c.center === prev)) {
          warnings.push(`QA: center hold at t=${fixed[i].t.toFixed(1)}s (${hold.toFixed(1)}s) → holding previous at ${prev}%`);
          fixed[i].x = prev;
        }
      }
    }
  }

  // Check 2: Large position changes must have instant-cut markers
  for (let i = 0; i < fixed.length - 1; i++) {
    const delta = Math.abs(fixed[i + 1].x - fixed[i].x);
    const dt = fixed[i + 1].t - fixed[i].t;
    if (delta > 15 && dt > 0.01) {
      const cutTime = Math.round((fixed[i + 1].t - 0.001) * 1000) / 1000;
      if (cutTime > fixed[i].t) {
        fixed.splice(i + 1, 0, { t: cutTime, x: fixed[i].x });
        warnings.push(`QA: inserted instant cut at t=${cutTime.toFixed(3)}s (delta=${delta})`);
        i++;
      }
    }
  }

  return { passed: warnings.length === 0, warnings, fixedKeyframes: fixed };
}


/**
 * Diagnostic: validate that each keyframe position would center the subject
 * within the crop window for a given aspect ratio conversion.
 *
 * @param {Array<{t:number, x:number}>} keyframes
 * @param {number} srcRatio - Source aspect ratio (e.g. 16/9)
 * @param {number} targetRatio - Target aspect ratio (e.g. 9/16)
 * @returns {{ keyframes: Array, corrections: number }}
 */
export function validateCentering(keyframes, srcRatio, targetRatio) {
  if (!srcRatio || !targetRatio) return { keyframes, corrections: 0 };
  const R = srcRatio / targetRatio;
  if (R <= 1.01) return { keyframes, corrections: 0 };

  const cropWidth = 1 / R;
  let corrections = 0;

  const fixed = keyframes.map(kf => {
    const objPos = (R * kf.x - 50) / (R - 1);
    const cropStart = (objPos / 100) * (1 - cropWidth);
    const faceInCrop = (kf.x / 100 - cropStart) / cropWidth;

    if (faceInCrop < 0.2 || faceInCrop > 0.8) {
      corrections++;
      console.log(
        `[CenteringQA] t=${kf.t.toFixed(1)}s: face at ${Math.round(faceInCrop * 100)}% of crop (should be ~50%)`
      );
    }
    return { ...kf };
  });

  return { keyframes: fixed, corrections };
}


/**
 * Compute layout-aware crop regions for a given timestamp.
 *
 * @param {number} t - Current playback time (relative to clip start)
 * @param {Array} layoutTimeline - [{start, end, layout_mode, face_positions, left_face_slot, right_face_slot}]
 * @param {number} srcW - Source video width
 * @param {number} srcH - Source video height
 * @param {string} targetAspect - Target aspect ratio string ("9:16", etc.)
 * @param {object} faceRegistry - {slots: [{id, x, frames}], multi_speaker}
 * @returns {Object} Layout render info:
 *   For SINGLE: { mode: "single" }
 *   For SPLIT:  { mode: "split", topX: number, bottomX: number }
 *   For PIP:    { mode: "pip", mainX: number, pipX: number, pipPosition: string, pipSize: number }
 *   For TRIPLE: { mode: "triple", positions: [x1, x2, x3] }
 *   For SCREENSHARE: { mode: "screenshare", speakerX: number }
 */
export function computeLayoutAtTime(t, layoutTimeline, srcW, srcH, targetAspect, faceRegistry) {
  if (!layoutTimeline || !layoutTimeline.length) {
    return { mode: 'single' };
  }

  // Find active layout segment
  const seg = layoutTimeline.find(s => t >= s.start && t < s.end);
  if (!seg || seg.layout_mode === 'single') {
    return { mode: 'single' };
  }

  const slots = faceRegistry?.slots || [];
  const slotMap = {};
  for (const s of slots) {
    slotMap[s.id] = s.x;
  }

  if (seg.layout_mode === 'split') {
    const leftX = slotMap[seg.left_face_slot] ?? 25;
    const rightX = slotMap[seg.right_face_slot] ?? 75;
    return { mode: 'split', topX: leftX, bottomX: rightX };
  }

  if (seg.layout_mode === 'pip') {
    const mainX = slotMap[seg.primary_face_slot] ?? 50;
    const pipX = slotMap[seg.pip_face_slot] ?? 50;
    return {
      mode: 'pip',
      mainX,
      pipX,
      pipPosition: seg.pip_position || 'bottom_right',
      pipSize: seg.pip_size_pct || 25,
    };
  }

  if (seg.layout_mode === 'triple') {
    const positions = slots.slice(0, 3).map(s => s.x);
    while (positions.length < 3) positions.push(50);
    return { mode: 'triple', positions };
  }

  if (seg.layout_mode === 'screenshare') {
    const speakerX = slots.length > 0 ? slots[0].x : 50;
    return { mode: 'screenshare', speakerX };
  }

  return { mode: 'single' };
}


/**
 * Convert subject tracking keyframes into crop segments for the timeline.
 *
 * Each segment represents a continuous period at a fixed crop X position.
 * Consecutive keyframes with the same x value are merged into one segment.
 *
 * @param {Array<{t: number, x: number}>} keyframes - Processed keyframes
 * @param {number} duration - Total clip duration in seconds
 * @param {Array<{center: number, count: number}>|null} clusters - Detected speaker clusters
 * @returns {Array<{id: string, startTime: number, endTime: number, cropX: number, clusterId: number, isManualOverride: boolean, label: string, originalCropX: number}>}
 */
export function keyframesToCropSegments(keyframes, duration, clusters) {
  if (!keyframes?.length || !duration) return [];

  // Build a lookup: x → closest cluster index
  const clusterLookup = (x) => {
    if (!clusters?.length) return -1;
    let bestIdx = 0;
    let bestDist = Math.abs(x - clusters[0].center);
    for (let i = 1; i < clusters.length; i++) {
      const d = Math.abs(x - clusters[i].center);
      if (d < bestDist) { bestDist = d; bestIdx = i; }
    }
    return bestDist <= 15 ? bestIdx : -1;
  };

  // Build segments by cluster assignment, not exact position.
  // This prevents the LP solver's smooth ease curves from creating
  // dozens of micro-segments during transitions.  A segment break
  // occurs only when the keyframe's cluster changes (i.e., the camera
  // has arrived at a different speaker position).
  const segments = [];
  let segStart = keyframes[0].t;
  let segX = keyframes[0].x;
  let segCluster = clusterLookup(segX);

  for (let i = 1; i < keyframes.length; i++) {
    const kf = keyframes[i];
    const kfCluster = clusterLookup(kf.x);

    // Break on cluster change — smooth transitions between clusters
    // are absorbed into the preceding segment (the camera is "in transit")
    if (kfCluster !== segCluster && kfCluster !== -1) {
      segments.push({ startTime: segStart, endTime: kf.t, cropX: segX });
      segStart = kf.t;
      segX = kf.x;
      segCluster = kfCluster;
    }
  }
  // Final segment to end of clip
  segments.push({ startTime: segStart, endTime: duration, cropX: segX });

  // Merge adjacent segments with the same cluster (can happen if the
  // solver briefly leaves and re-enters the same cluster)
  const merged = [segments[0]];
  for (let i = 1; i < segments.length; i++) {
    const prev = merged[merged.length - 1];
    const curr = segments[i];
    if (clusterLookup(prev.cropX) === clusterLookup(curr.cropX)) {
      // Merge: extend previous segment
      prev.endTime = curr.endTime;
    } else {
      merged.push(curr);
    }
  }

  // Enforce strict contiguity: each segment's endTime === next segment's startTime
  for (let i = 0; i < merged.length - 1; i++) {
    merged[i].endTime = merged[i + 1].startTime;
  }

  // Annotate with IDs, cluster info, and labels
  return merged.map((seg, i) => {
    const cId = clusterLookup(seg.cropX);
    return {
      id: `crop-${i}`,
      startTime: seg.startTime,
      endTime: seg.endTime,
      cropX: seg.cropX,
      originalCropX: seg.cropX,
      clusterId: cId,
      isManualOverride: false,
      label: cId >= 0 ? `Speaker ${cId + 1}` : `${seg.cropX}%`,
    };
  });
}

/**
 * Convert subject tracking keyframes into crop segments using backend slot identity.
 * Preferred over keyframesToCropSegments when slot data is available from the RenderPlan.
 *
 * @param {Array<{t: number, x: number}>} keyframes - Processed keyframes
 * @param {number} duration - Total clip duration in seconds
 * @param {Array<{t: number, slot: number}>} slotTimeline - Sparse slot assignments from backend
 * @param {Object} speakerNames - Map of slot_id → display name (from user renames)
 * @returns {Array<{id: string, startTime: number, endTime: number, cropX: number, clusterId: number, isManualOverride: boolean, label: string, originalCropX: number, speakerSlot: number}>}
 */
export function keyframesToCropSegmentsWithSlots(keyframes, duration, slotTimeline, speakerNames = {}) {
  if (!keyframes?.length || !duration || !slotTimeline?.length) return [];

  // Build a function to look up the active slot at a given time
  const slotAtTime = (t) => {
    let best = slotTimeline[0];
    for (const entry of slotTimeline) {
      if (entry.t <= t) best = entry;
      else break;
    }
    return best?.slot ?? -1;
  };

  // Build segments by slot changes
  const segments = [];
  let segStart = 0;
  let segSlot = slotAtTime(0);
  let segX = keyframes[0]?.x ?? 50;

  // Walk through keyframes and break on slot changes
  for (let i = 1; i < keyframes.length; i++) {
    const kf = keyframes[i];
    const slot = slotAtTime(kf.t);

    if (slot !== segSlot && slot >= 0) {
      segments.push({ startTime: segStart, endTime: kf.t, cropX: segX, slot: segSlot });
      segStart = kf.t;
      segX = kf.x;
      segSlot = slot;
    }
  }
  // Final segment
  segments.push({ startTime: segStart, endTime: duration, cropX: segX, slot: segSlot });

  // Merge adjacent segments with same slot
  const merged = [segments[0]];
  for (let i = 1; i < segments.length; i++) {
    const prev = merged[merged.length - 1];
    if (prev.slot === segments[i].slot) {
      prev.endTime = segments[i].endTime;
    } else {
      merged.push(segments[i]);
    }
  }

  // Enforce contiguity
  for (let i = 0; i < merged.length - 1; i++) {
    merged[i].endTime = merged[i + 1].startTime;
  }

  console.log(`[CropTrack] segments_from_slots=${segments.length}`);

  // Label resolution: speakerNames[slot] > "Speaker ${slot+1}" > geometric fallback
  return merged.map((seg, i) => {
    const slot = seg.slot;
    const label = speakerNames[slot] || (slot >= 0 ? `Speaker ${slot + 1}` : `${seg.cropX}%`);
    return {
      id: `crop-${i}`,
      startTime: seg.startTime,
      endTime: seg.endTime,
      cropX: seg.cropX,
      originalCropX: seg.cropX,
      clusterId: slot,
      isManualOverride: false,
      label,
      speakerSlot: slot,
    };
  });
}

/**
 * Extract a slot timeline from RenderPlan ops.
 * Each op may have speaker_slot set by the backend.
 *
 * @param {Object} renderPlan - RenderPlan with ops array
 * @returns {Array<{t: number, slot: number}>|null}
 */
export function extractSlotTimelineFromRenderPlan(renderPlan) {
  if (!renderPlan?.ops?.length) return null;
  const timeline = [];
  for (const op of renderPlan.ops) {
    if (op.speaker_slot != null && op.speaker_slot >= 0) {
      timeline.push({ t: op.start_sec, slot: op.speaker_slot });
    }
  }
  return timeline.length > 0 ? timeline : null;
}


// ── RenderPlan integration ──────────────────────────────────────────────
// When USE_RENDER_PLAN is enabled on the backend, the preview can fetch
// a RenderPlan JSON instead of building keyframes from raw scenes.
// This ensures the preview shows exactly what FFmpeg will export.

/**
 * Fetch the RenderPlan for a job from the backend API.
 *
 * @param {string} jobId - The job ID
 * @param {Object} options
 * @param {string} options.mode - "full" or "clip"
 * @param {number} [options.clipIndex] - Clip index (required for mode="clip")
 * @param {string} [options.aspectRatio] - Target aspect ratio (e.g., "9:16")
 * @returns {Promise<Object|null>} The RenderPlan JSON, or null if unavailable
 */
export async function fetchRenderPlan(jobId, { mode = 'full', clipIndex, aspectRatio = '9:16' } = {}) {
  if (!jobId) return null;
  try {
    const params = new URLSearchParams({ mode, aspect_ratio: aspectRatio });
    if (clipIndex != null) params.set('clip_index', String(clipIndex));
    const res = await fetch(`/api/jobs/${jobId}/render_plan?${params}`);
    if (!res.ok) return null;
    const plan = await res.json();
    if (plan && plan.ops?.length > 0) {
      console.log(`[SubjectTracking] RenderPlan loaded: ${plan.ops.length} ops, ${plan.total_duration_sec?.toFixed(1)}s`);
      return plan;
    }
    return null;
  } catch (err) {
    console.log('[SubjectTracking] RenderPlan not available:', err.message);
    return null;
  }
}
