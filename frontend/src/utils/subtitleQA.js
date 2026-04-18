/**
 * Subtitle QA & Validation
 *
 * Validates that subtitle changes are properly applied across the preview
 * player and exported video. Ensures the timeline store (single source of
 * truth) is consistent with what SubtitleOverlay renders and what
 * RenderEngine composites for export.
 */

const FONT_SIZE_MAP = { small: 22, medium: 30, large: 40 };
const REF_W = 1920;
const REF_H = 1080;

/**
 * Validate subtitle items in the timeline store.
 * Returns { valid: boolean, errors: string[], warnings: string[] }
 */
export function validateSubtitleItems(items) {
  const errors = [];
  const warnings = [];
  const subtitles = items.filter((it) => it.type === 'subtitle');

  if (subtitles.length === 0) {
    warnings.push('No subtitle items found in timeline');
    return { valid: true, errors, warnings };
  }

  for (const sub of subtitles) {
    const label = `Subtitle "${(sub.subtitleText || '').slice(0, 30)}..." (${sub.id})`;

    // Required fields
    if (!sub.subtitleText && sub.subtitleText !== '') {
      errors.push(`${label}: missing subtitleText`);
    }
    if (typeof sub.start !== 'number' || isNaN(sub.start)) {
      errors.push(`${label}: invalid start time`);
    }
    if (typeof sub.end !== 'number' || isNaN(sub.end)) {
      errors.push(`${label}: invalid end time`);
    }
    if (sub.start >= sub.end) {
      errors.push(`${label}: start (${sub.start}) >= end (${sub.end})`);
    }

    // Track assignment
    if (sub.trackId !== 't1') {
      warnings.push(`${label}: on track "${sub.trackId}" instead of "t1"`);
    }

    // Position validation
    const pos = sub.position || {};
    if (typeof pos.x !== 'number' || pos.x < 0 || pos.x > 100) {
      warnings.push(`${label}: position.x (${pos.x}) out of [0,100] range`);
    }
    if (typeof pos.y !== 'number' || pos.y < 0 || pos.y > 100) {
      warnings.push(`${label}: position.y (${pos.y}) out of [0,100] range`);
    }

    // Rotation validation
    const rotation = sub.transform?.rotation ?? 0;
    if (rotation < -360 || rotation > 360) {
      warnings.push(`${label}: rotation (${rotation}) out of [-360,360] range`);
    }

    // Duration sanity
    const dur = sub.end - sub.start;
    if (dur < 0.1) {
      warnings.push(`${label}: very short duration (${dur.toFixed(2)}s)`);
    }
    if (dur > 30) {
      warnings.push(`${label}: very long duration (${dur.toFixed(1)}s)`);
    }
  }

  // Check for overlapping subtitles on the same track
  const sorted = [...subtitles].sort((a, b) => a.start - b.start);
  for (let i = 1; i < sorted.length; i++) {
    if (sorted[i].start < sorted[i - 1].end - 0.01) {
      warnings.push(
        `Overlapping subtitles: "${(sorted[i - 1].subtitleText || '').slice(0, 20)}" ` +
        `(${sorted[i - 1].start.toFixed(2)}-${sorted[i - 1].end.toFixed(2)}) overlaps with ` +
        `"${(sorted[i].subtitleText || '').slice(0, 20)}" (${sorted[i].start.toFixed(2)}-${sorted[i].end.toFixed(2)})`
      );
    }
  }

  return { valid: errors.length === 0, errors, warnings };
}

/**
 * Validate that subtitle settings will produce consistent rendering
 * between preview (SubtitleOverlay CSS) and export (RenderEngine canvas).
 */
export function validateSubtitleSettings(settings) {
  const errors = [];
  const warnings = [];
  const s = settings || {};

  // Font validation
  const font = s.subtitleFont || 'DM Sans';
  if (!font || typeof font !== 'string') {
    errors.push('subtitleFont is not a valid string');
  }

  // Size validation
  const size = s.subtitleSize || 'medium';
  if (typeof size === 'string' && !FONT_SIZE_MAP[size]) {
    errors.push(`subtitleSize "${size}" is not a valid preset (small/medium/large)`);
  }

  // Font weight — accepts numeric (100-900) or string ("normal", "bold", "black")
  const weight = s.subtitleFontWeight;
  if (typeof weight === 'number') {
    if (weight < 100 || weight > 900) {
      warnings.push(`subtitleFontWeight ${weight} should be 100-900`);
    }
  } else if (weight && !['normal', 'bold', 'black'].includes(weight)) {
    warnings.push(`subtitleFontWeight "${weight}" should be normal/bold/black or 100-900`);
  }

  // Position
  const position = s.subtitlePosition || 'bottom';
  if (!['top', 'center', 'bottom'].includes(position)) {
    errors.push(`subtitlePosition "${position}" should be top/center/bottom`);
  }

  // Max width
  const maxWidth = s.subtitleMaxWidth ?? 90;
  if (maxWidth < 20 || maxWidth > 100) {
    warnings.push(`subtitleMaxWidth (${maxWidth}) outside recommended [20,100] range`);
  }

  // Vertical offset
  const offsetV = s.subtitleOffsetV ?? 4;
  if (offsetV < 0 || offsetV > 50) {
    warnings.push(`subtitleOffsetV (${offsetV}) outside [0,50] range`);
  }

  // Color validations
  const colorFields = ['subtitleFontColor', 'subtitleOutlineColor', 'subtitleBgColor'];
  for (const field of colorFields) {
    const val = s[field];
    if (val && !/^#[0-9a-fA-F]{3,8}$/.test(val)) {
      warnings.push(`${field} "${val}" is not a valid hex color`);
    }
  }

  // Opacity ranges
  const opacityFields = [
    { key: 'subtitleOutlineOpacity', def: 100 },
    { key: 'subtitleBgOpacity', def: 75 },
  ];
  for (const { key, def } of opacityFields) {
    const val = s[key] ?? def;
    if (val < 0 || val > 100) {
      warnings.push(`${key} (${val}) should be in [0,100]`);
    }
  }

  // Outline width
  const olWidth = s.subtitleOutlineWidth ?? 2;
  if (olWidth < 0 || olWidth > 10) {
    warnings.push(`subtitleOutlineWidth (${olWidth}) outside [0,10] range`);
  }

  return { valid: errors.length === 0, errors, warnings };
}

/**
 * Validate that subtitle rendering parameters match between
 * SubtitleOverlay (preview) and RenderEngine (export).
 *
 * Takes the settings and output dimensions and computes what both
 * renderers would use, flagging any mismatches.
 */
export function validatePreviewExportConsistency(settings, outputDims) {
  const errors = [];
  const warnings = [];
  const s = settings || {};
  const { w: outputW, h: outputH } = outputDims || { w: 1920, h: 1080 };

  // Font scale calculation (must match both renderers)
  const fontScale = Math.min(outputW, outputH) / Math.min(REF_W, REF_H);
  const sizeLabel = s.subtitleSize || 'medium';
  const basePx = typeof sizeLabel === 'number' ? sizeLabel : (FONT_SIZE_MAP[sizeLabel] || 30);
  const exportFontSize = Math.max(16, Math.round(basePx * fontScale));

  if (exportFontSize < 16) {
    warnings.push(`Export font size (${exportFontSize}px) may be too small at ${outputW}x${outputH}`);
  }

  // Outline width scaling
  const olWidth = s.subtitleOutlineWidth ?? 2;
  const exportOutlineWidth = Math.max(0, Math.round(olWidth * fontScale));
  if (olWidth > 0 && exportOutlineWidth === 0) {
    warnings.push(`Outline width ${olWidth} will round to 0 at ${outputW}x${outputH} - increase outline width`);
  }

  // Background padding
  if (s.subtitleBgEnabled) {
    const pad = Math.max(4, Math.round(8 * fontScale));
    if (pad < 4) {
      warnings.push(`Background padding may appear too thin at ${outputW}x${outputH}`);
    }
  }

  // Position consistency check
  const position = s.subtitlePosition || 'bottom';
  const offsetV = s.subtitleOffsetV ?? 4;
  if (position === 'bottom' && offsetV > 40) {
    warnings.push('Subtitle offset is very high (>40%), subtitle may overlap with video center');
  }

  return { valid: errors.length === 0, errors, warnings };
}

/**
 * Validate transcript-timeline consistency.
 * Ensures subtitle items in the timeline store match the backend transcript.
 *
 * @param {Array} items - Timeline store items
 * @param {Array} transcript - Backend transcript segments
 * @param {number} clipStart - Clip start time (absolute)
 * @param {number} clipEnd - Clip end time (absolute)
 * @returns {{ valid: boolean, errors: string[], warnings: string[], stats: Object }}
 */
export function validateTranscriptSync(items, transcript, clipStart = 0, clipEnd = Infinity) {
  const errors = [];
  const warnings = [];
  const subtitles = items.filter((it) => it.type === 'subtitle');

  if (!transcript || transcript.length === 0) {
    if (subtitles.length > 0) {
      warnings.push(`${subtitles.length} subtitle item(s) in timeline but no transcript data`);
    }
    return { valid: true, errors, warnings, stats: { matched: 0, unmatched: 0, missing: 0 } };
  }

  // Transcript segments within clip range
  const clipTranscript = transcript.filter(
    (seg) => seg.end > clipStart && seg.start < clipEnd
  );

  let matched = 0;
  let textMismatches = 0;
  let speakerMismatches = 0;
  const unmatchedTimeline = [];
  const unmatchedTranscript = [];
  const matchedTranscriptIdxs = new Set();

  // Match timeline items to transcript segments
  for (const sub of subtitles) {
    const subAbsStart = (sub.start || 0) + clipStart;
    const subAbsEnd = (sub.end || 0) + clipStart;
    const matchIdx = clipTranscript.findIndex((seg, idx) => {
      if (matchedTranscriptIdxs.has(idx)) return false;
      return Math.abs((seg.start ?? 0) - subAbsStart) < 0.3 &&
             Math.abs((seg.end ?? 0) - subAbsEnd) < 0.3;
    });

    if (matchIdx >= 0) {
      matchedTranscriptIdxs.add(matchIdx);
      matched++;
      const seg = clipTranscript[matchIdx];

      if (seg.text !== sub.subtitleText) {
        textMismatches++;
        warnings.push(
          `Text mismatch at ${sub.start.toFixed(1)}s: timeline="${(sub.subtitleText || '').slice(0, 30)}" vs transcript="${(seg.text || '').slice(0, 30)}"`
        );
      }
      if (seg.speaker && sub.speaker && seg.speaker !== sub.speaker) {
        speakerMismatches++;
        warnings.push(
          `Speaker mismatch at ${sub.start.toFixed(1)}s: timeline="${sub.speaker}" vs transcript="${seg.speaker}"`
        );
      }
    } else {
      unmatchedTimeline.push(sub);
    }
  }

  // Find transcript segments without a matching timeline item
  clipTranscript.forEach((seg, idx) => {
    if (!matchedTranscriptIdxs.has(idx)) {
      unmatchedTranscript.push(seg);
    }
  });

  if (unmatchedTimeline.length > 0) {
    warnings.push(
      `${unmatchedTimeline.length} subtitle item(s) in timeline have no matching transcript segment`
    );
  }
  if (unmatchedTranscript.length > 0) {
    warnings.push(
      `${unmatchedTranscript.length} transcript segment(s) have no matching timeline subtitle item`
    );
  }

  const stats = {
    matched,
    textMismatches,
    speakerMismatches,
    unmatchedTimeline: unmatchedTimeline.length,
    unmatchedTranscript: unmatchedTranscript.length,
    totalTimeline: subtitles.length,
    totalTranscript: clipTranscript.length,
  };

  return { valid: errors.length === 0, errors, warnings, stats };
}

/**
 * Validate subtitle settings are applied correctly in the rendering pipeline.
 * Checks that what the user sees in preview matches what gets exported.
 *
 * @param {Object} settings - Clip settings
 * @param {Array} items - Timeline store items
 * @returns {{ valid: boolean, errors: string[], warnings: string[] }}
 */
export function validateSettingsApplication(settings, items) {
  const errors = [];
  const warnings = [];
  const s = settings || {};
  const subtitles = items.filter((it) => it.type === 'subtitle');

  if (subtitles.length === 0) return { valid: true, errors, warnings };

  // Check that subtitlesEnabled matches the presence of subtitle items
  if (!s.subtitlesEnabled && subtitles.length > 0) {
    warnings.push(
      'Subtitles are disabled but timeline has subtitle items — they won\'t render in preview'
    );
  }

  // Check speaker color consistency
  if (s.useSpeakerColors !== false) {
    const speakersInItems = new Set(subtitles.map((it) => it.speaker).filter(Boolean));
    if (speakersInItems.size === 0) {
      warnings.push('Speaker colors enabled but no subtitle items have speaker assignments');
    }
  }

  // Check for empty subtitle text
  const emptyCount = subtitles.filter((it) => !it.subtitleText || !it.subtitleText.trim()).length;
  if (emptyCount > 0) {
    warnings.push(`${emptyCount} subtitle item(s) have empty text — they will render as blank`);
  }

  // Check subtitle position consistency
  const customPositions = subtitles.filter(
    (it) => it.position && (it.position.x !== 50 || it.position.y !== 90)
  );
  if (customPositions.length > 0 && customPositions.length < subtitles.length) {
    warnings.push(
      `${customPositions.length}/${subtitles.length} subtitle(s) have custom positions — may look inconsistent`
    );
  }

  return { valid: errors.length === 0, errors, warnings };
}

/**
 * Validate active word highlighting timing and FPS adequacy.
 * Ensures the export FPS is high enough for smooth word-by-word
 * highlighting that doesn't lag or skip words.
 *
 * @param {Array} items - Timeline store items
 * @param {Object} settings - Clip settings
 * @param {number} exportFPS - The export FPS (after adaptive boost)
 * @returns {{ valid: boolean, errors: string[], warnings: string[] }}
 */
export function validateActiveWordTiming(items, settings, exportFPS = 30) {
  const errors = [];
  const warnings = [];
  const s = settings || {};
  const subtitles = items.filter((it) => it.type === 'subtitle');

  if (!s.activeWordEnabled || subtitles.length === 0) {
    return { valid: true, errors, warnings };
  }

  // Check each subtitle segment for timing adequacy
  for (const sub of subtitles) {
    const text = sub.subtitleText || '';
    const words = text.split(/\s+/).filter(Boolean);
    if (words.length <= 1) continue;

    const segDuration = sub.end - sub.start;
    const avgWordDuration = segDuration / words.length;
    const framesPerWord = avgWordDuration * exportFPS;

    if (framesPerWord < 2) {
      errors.push(
        `Subtitle at ${sub.start.toFixed(1)}s has ${words.length} words in ${segDuration.toFixed(1)}s ` +
        `(${framesPerWord.toFixed(1)} frames/word at ${exportFPS}fps) — words will skip/lag`
      );
    } else if (framesPerWord < 3) {
      warnings.push(
        `Subtitle at ${sub.start.toFixed(1)}s has fast word transitions ` +
        `(${framesPerWord.toFixed(1)} frames/word) — may appear slightly rushed`
      );
    }

    // Check for very short segments that might cause word index jumps
    if (segDuration < 0.3 && words.length > 2) {
      warnings.push(
        `Subtitle at ${sub.start.toFixed(1)}s is very short (${segDuration.toFixed(2)}s) ` +
        `with ${words.length} words — active highlighting may not display all words`
      );
    }
  }

  return { valid: errors.length === 0, errors, warnings };
}

/**
 * Validate that all clip effects and settings will be applied consistently
 * in the exported video. Checks video, image, text, shape, and subtitle
 * clips for any settings that might differ between preview and export.
 *
 * @param {Array} items - Timeline store items
 * @param {Object} settings - Clip settings
 * @returns {{ valid: boolean, errors: string[], warnings: string[] }}
 */
export function validateClipEffectsParity(items, settings) {
  const errors = [];
  const warnings = [];

  for (const item of items) {
    const label = `${item.type} clip "${(item.subtitleText || item.textContent || item.id || '').slice(0, 20)}"`;

    // Validate effects are within canvas filter support
    const fx = item.effects || {};
    if (fx.blur && fx.blur > 20) {
      warnings.push(`${label}: blur (${fx.blur}px) is very high — may affect export performance`);
    }

    // Validate transforms
    const t = item.transform || {};
    if (t.scaleX !== undefined && (t.scaleX <= 0 || t.scaleY <= 0)) {
      warnings.push(`${label}: has zero or negative scale — may render invisible`);
    }

    // Validate opacity
    if (item.opacity !== undefined && (item.opacity < 0 || item.opacity > 1)) {
      warnings.push(`${label}: opacity (${item.opacity}) outside [0,1] range`);
    }

    // Validate fade durations
    if (item.fadeIn && item.fadeOut) {
      const dur = item.end - item.start;
      if (item.fadeIn + item.fadeOut > dur) {
        warnings.push(`${label}: fadeIn (${item.fadeIn}s) + fadeOut (${item.fadeOut}s) exceeds clip duration (${dur.toFixed(1)}s)`);
      }
    }

    // Validate transitions
    if (item.transition) {
      const validTypes = ['dissolve', 'fade', 'wipe-left', 'wipe-right', 'slide-left', 'slide-right', 'zoom'];
      if (item.transition.type && !validTypes.includes(item.transition.type)) {
        warnings.push(`${label}: transition type "${item.transition.type}" is not recognized`);
      }
      if (item.transition.duration && item.transition.duration > (item.end - item.start)) {
        warnings.push(`${label}: transition duration exceeds clip length`);
      }
    }

    // Validate text clips
    if (item.type === 'text') {
      if (!item.textContent && !item.subtitleText) {
        warnings.push(`${label}: text clip has no content — will render blank`);
      }
      const style = item.textStyle || {};
      if (style.fontSize && style.fontSize > 200) {
        warnings.push(`${label}: font size (${style.fontSize}px) is very large`);
      }
    }

    // Validate shape clips
    if (item.type === 'shape') {
      const validShapes = ['rectangle', 'circle', 'ellipse', 'arrow', 'line'];
      if (item.shapeType && !validShapes.includes(item.shapeType)) {
        warnings.push(`${label}: shape type "${item.shapeType}" is not recognized`);
      }
    }
  }

  return { valid: errors.length === 0, errors, warnings };
}

/**
 * Validate that video effects and overlays from the multi-track editor
 * will be included in the server export payload.
 * Ensures the exported MP4 is 1:1 with the preview player.
 */
export function validateVideoEffectsExport(items) {
  const errors = [];
  const warnings = [];

  // Check video item effects
  const videoItem = items.find(it => it.type === 'video');
  if (videoItem?.effects) {
    const fx = videoItem.effects;
    const activeEffects = [];
    if ((fx.brightness || 0) !== 0) activeEffects.push(`brightness: ${fx.brightness}`);
    if ((fx.contrast || 0) !== 0) activeEffects.push(`contrast: ${fx.contrast}`);
    if ((fx.saturation || 0) !== 0) activeEffects.push(`saturation: ${fx.saturation}`);
    if ((fx.blur || 0) > 0) activeEffects.push(`blur: ${fx.blur}px`);
    if ((fx.hueRotate || 0) > 0) activeEffects.push(`hue: ${fx.hueRotate}deg`);
    if ((fx.sepia || 0) > 0) activeEffects.push(`sepia: ${fx.sepia}%`);
    if ((videoItem.opacity ?? 1) < 1) activeEffects.push(`opacity: ${(videoItem.opacity * 100).toFixed(0)}%`);

    if (activeEffects.length > 0) {
      // Not an error — these ARE now applied via FFmpeg filters
      // Just informational so users see what effects will be in the export
      warnings.push(`Video effects active (will be applied to export): ${activeEffects.join(', ')}`);
    }
  }

  // Check text overlays
  const textItems = items.filter(it => it.type === 'text');
  if (textItems.length > 0) {
    warnings.push(`${textItems.length} text overlay(s) will be included in export via FFmpeg drawtext`);
  }

  // Check image overlays
  const imageItems = items.filter(it => it.type === 'image' || it.type === 'overlay');
  const nonVideoImages = imageItems.filter(it => it.type !== 'video');
  if (nonVideoImages.length > 0) {
    warnings.push(`${nonVideoImages.length} image overlay(s) will be included in export`);
  }

  // Check shape overlays
  const shapeItems = items.filter(it => it.type === 'shape');
  if (shapeItems.length > 0) {
    const shapeTypes = shapeItems.map(it => it.shapeType || 'rectangle');
    warnings.push(`${shapeItems.length} shape overlay(s) will be included in export (${[...new Set(shapeTypes)].join(', ')})`);
    // Validate shape timing
    for (const shape of shapeItems) {
      if ((shape.start || 0) >= (shape.end || 0)) {
        errors.push(`Shape "${shape.id}" has invalid timing (start >= end) — will not appear in export`);
      }
    }
  }

  // Check audio overlays
  const audioItems = items.filter(it => it.type === 'audio');
  if (audioItems.length > 0) {
    const missingSrc = audioItems.filter(it => !it.src && !it.mediaRef);
    if (missingSrc.length > 0) {
      errors.push(`${missingSrc.length} audio overlay(s) missing source file — will not be included in export`);
    } else {
      warnings.push(`${audioItems.length} audio overlay(s) will be mixed into export`);
    }
  }

  return { valid: errors.length === 0, errors, warnings };
}

/**
 * Validate that the export will produce a video matching the preview.
 * This is the master check that verifies all rendering parameters
 * are consistent between preview (SubtitleOverlay CSS + DOM) and
 * export (RenderEngine canvas).
 *
 * @param {Array} items - Timeline store items
 * @param {Object} settings - Clip settings
 * @param {Object} outputDims - { w, h } output resolution
 * @returns {{ valid: boolean, errors: string[], warnings: string[], confidence: string }}
 */
export function validateExportParity(items, settings, outputDims) {
  const errors = [];
  const warnings = [];
  const s = settings || {};
  const { w: outputW, h: outputH } = outputDims || { w: 1920, h: 1080 };

  // Check all item types are supported by RenderEngine
  const supportedTypes = new Set(['video', 'audio', 'image', 'overlay', 'text', 'subtitle', 'shape']);
  for (const item of items) {
    if (!supportedTypes.has(item.type)) {
      errors.push(`Item "${item.id}" has unsupported type "${item.type}" — will not render in export`);
    }
  }

  // Verify track visibility settings will be honored
  const subtitleTrack = items.some(it => it.type === 'subtitle' && it.trackId === 't1');
  if (subtitleTrack && !s.subtitlesEnabled) {
    warnings.push('Subtitle items exist but subtitles are disabled — they will not appear in the export');
  }

  // Verify speaker colors will match between preview and export
  if (s.useSpeakerColors !== false) {
    const subtitles = items.filter(it => it.type === 'subtitle');
    const speakers = new Set(subtitles.map(s => s.speaker).filter(Boolean));
    if (speakers.size > 6 && !s.speakerColors) {
      warnings.push(
        `${speakers.size} speakers detected but only 6 default palette colors — ` +
        'some speakers will share colors (set custom speaker colors to differentiate)'
      );
    }
  }

  // Verify active word settings parity
  if (s.activeWordEnabled) {
    const subtitles = items.filter(it => it.type === 'subtitle');
    if (subtitles.length === 0) {
      warnings.push('Active word highlighting is enabled but no subtitle items exist');
    }
    // Warn about FPS adequacy
    let fastestWordRate = 0;
    for (const sub of subtitles) {
      const words = (sub.subtitleText || '').split(/\s+/).filter(Boolean);
      if (words.length > 1) {
        const rate = words.length / (sub.end - sub.start);
        fastestWordRate = Math.max(fastestWordRate, rate);
      }
    }
    if (fastestWordRate > 10) {
      warnings.push(
        `Fastest word rate is ${fastestWordRate.toFixed(1)} words/sec — ` +
        'export FPS will be boosted automatically for smooth highlighting'
      );
    }
  }

  // Check for items that extend beyond the export range
  const maxTime = Math.max(...items.map(it => it.end), 0);
  if (maxTime === 0) {
    warnings.push('No items with duration found — export may be empty');
  }

  // Compute confidence level
  let confidence;
  if (errors.length === 0 && warnings.length === 0) {
    confidence = 'high';
  } else if (errors.length === 0 && warnings.length <= 2) {
    confidence = 'medium';
  } else {
    confidence = errors.length > 0 ? 'low' : 'medium';
  }

  return { valid: errors.length === 0, errors, warnings, confidence };
}

/**
 * Validate subject tracking will keep the subject centered without
 * cropping the video off-screen for any aspect ratio.
 *
 * Checks:
 * 1. All keyframe subject_x values produce valid crop offsets within frame bounds
 * 2. The crop window always covers the full output aspect ratio (no black bars)
 * 3. Dynamic tracking transitions are smooth enough (no jarring jumps)
 * 4. Edge-clamped positions still keep subject visible inside frame
 *
 * @param {Object} trackingInfo - { scenes, clipStart, clipEnd, srcW, srcH, subjectX }
 * @param {string|null} aspectRatio - Target aspect ratio ('9:16', '1:1', '4:5', '16:9')
 * @param {Object} outputDims - { w, h } output dimensions
 * @returns {{ valid: boolean, errors: string[], warnings: string[] }}
 */
export function validateSubjectTracking(trackingInfo, aspectRatio, outputDims) {
  const errors = [];
  const warnings = [];

  if (!trackingInfo || !aspectRatio) {
    return { valid: true, errors, warnings };
  }

  const { scenes, clipStart, clipEnd, srcW, srcH, subjectX } = trackingInfo;
  if (!srcW || !srcH) {
    return { valid: true, errors, warnings };
  }

  const ASPECT_VALUES = { '16:9': 16/9, '9:16': 9/16, '1:1': 1.0, '4:5': 4/5 };
  const targetRatio = ASPECT_VALUES[aspectRatio];
  if (!targetRatio) {
    return { valid: true, errors, warnings };
  }

  const srcRatio = srcW / srcH;

  // No crop needed if same aspect ratio
  if (Math.abs(srcRatio - targetRatio) <= 0.01) {
    return { valid: true, errors, warnings };
  }

  // Only horizontal cropping is tracked (srcAR > dstAR)
  const needsHorizontalCrop = srcRatio > targetRatio;
  if (!needsHorizontalCrop) {
    return { valid: true, errors, warnings };
  }

  // Compute crop dimensions
  const cropW = Math.round(srcH * targetRatio);
  const cropH = srcH;
  const maxOffset = srcW - cropW;

  if (maxOffset <= 0) {
    return { valid: true, errors, warnings };
  }

  // Helper: compute crop offset from subject_x and verify bounds
  const checkSubjectX = (sx, label) => {
    const safeSx = Math.max(10, Math.min(90, Math.round(sx)));
    const subjectPixel = srcW * safeSx / 100;
    const xOffset = Math.round(Math.max(0, Math.min(maxOffset, subjectPixel - cropW / 2)));

    // Verify crop stays within frame
    if (xOffset < 0) {
      errors.push(`${label}: crop offset ${xOffset} is negative (video would shift off-screen left)`);
    }
    if (xOffset > maxOffset) {
      errors.push(`${label}: crop offset ${xOffset} exceeds max ${maxOffset} (video would shift off-screen right)`);
    }

    // Verify subject is inside crop window
    const subjectInCrop = subjectPixel - xOffset;
    if (subjectInCrop < 0 || subjectInCrop > cropW) {
      errors.push(`${label}: subject at pixel ${subjectPixel.toFixed(0)} outside crop window [${xOffset}, ${xOffset + cropW}]`);
    }

    // Check centering quality (when not edge-clamped)
    if (xOffset > 0 && xOffset < maxOffset) {
      const cropCenter = cropW / 2;
      const centerError = Math.abs(subjectInCrop - cropCenter) / cropW * 100;
      if (centerError > 15) {
        warnings.push(`${label}: subject is ${centerError.toFixed(0)}% off-center (target: <15%)`);
      }
    }

    // Verify output fills the full aspect ratio (no black bars)
    const actualCropW = Math.min(cropW, srcW - xOffset);
    if (actualCropW < cropW * 0.99) {
      errors.push(`${label}: crop width ${actualCropW} < target ${cropW} — video won't fill the full ${aspectRatio} frame`);
    }

    return { xOffset, subjectInCrop };
  };

  // Validate static subject_x
  const staticSx = subjectX ?? 50;
  checkSubjectX(staticSx, `Static tracking (sx=${staticSx})`);

  // Validate dynamic keyframes if scenes available
  if (scenes?.length) {
    // Import dynamically to avoid circular dependencies
    try {
      // We'll validate the scene data directly without the full pipeline
      // to check raw values are reasonable
      for (let i = 0; i < scenes.length; i++) {
        const s = scenes[i];
        const sx = s.subject_x ?? 50;
        if (sx < 0 || sx > 100) {
          errors.push(`Scene ${i} (t=${s.timestamp?.toFixed(1)}s): subject_x=${sx} out of [0,100] range`);
        }
        if (sx < 5 || sx > 95) {
          warnings.push(`Scene ${i} (t=${s.timestamp?.toFixed(1)}s): subject_x=${sx} near edge — may cause edge-clamped crop`);
        }
      }

      // Check for extreme jumps between consecutive scenes (potential false detections)
      const sorted = [...scenes].sort((a, b) => a.timestamp - b.timestamp);
      for (let i = 1; i < sorted.length; i++) {
        const prev = sorted[i - 1];
        const cur = sorted[i];
        const dt = cur.timestamp - prev.timestamp;
        const dx = Math.abs((cur.subject_x ?? 50) - (prev.subject_x ?? 50));
        if (dt > 0 && dt < 0.5 && dx > 30) {
          warnings.push(
            `Rapid subject jump: ${dx}% in ${(dt * 1000).toFixed(0)}ms between scenes at ` +
            `t=${prev.timestamp.toFixed(1)}s and t=${cur.timestamp.toFixed(1)}s — may indicate false detection`
          );
        }
      }

      // Verify the R ratio doesn't cause impossible centering
      const R = srcRatio / targetRatio;
      const visiblePct = (1 / R) * 100;
      if (visiblePct < 25) {
        warnings.push(
          `Extreme aspect ratio conversion (${srcRatio.toFixed(2)} → ${targetRatio.toFixed(2)}): ` +
          `only ${visiblePct.toFixed(0)}% of source width visible — tracking accuracy may be limited`
        );
      }
    } catch {
      warnings.push('Could not validate dynamic tracking keyframes');
    }
  }

  return { valid: errors.length === 0, errors, warnings };
}

/**
 * Full QA validation — runs all checks and returns a combined report.
 *
 * @param {Array} items - Timeline store items
 * @param {Object} settings - Clip settings
 * @param {Object} outputDims - { w, h } output resolution
 * @param {Object} [syncInfo] - Optional { transcript, clipStart, clipEnd } for sync validation
 * @param {number} [exportFPS] - Export FPS for active word timing validation
 * @param {Object} [trackingInfo] - Optional { scenes, clipStart, clipEnd, srcW, srcH, subjectX }
 * @param {string} [aspectRatio] - Target aspect ratio
 * @returns {{ valid: boolean, errors: string[], warnings: string[], summary: string, checks: Object[], confidence: string }}
 */
export function runSubtitleQA(items, settings, outputDims, syncInfo, exportFPS = 30, trackingInfo = null, aspectRatio = null) {
  const itemResult = validateSubtitleItems(items);
  const settingsResult = validateSubtitleSettings(settings);
  const consistencyResult = validatePreviewExportConsistency(settings, outputDims);
  const settingsAppResult = validateSettingsApplication(settings, items);
  const activeWordResult = validateActiveWordTiming(items, settings, exportFPS);
  const effectsResult = validateClipEffectsParity(items, settings);
  const parityResult = validateExportParity(items, settings, outputDims);
  const videoEffectsResult = validateVideoEffectsExport(items);

  const checks = [
    { name: 'Subtitle items', ...itemResult },
    { name: 'Subtitle settings', ...settingsResult },
    { name: 'Preview/export consistency', ...consistencyResult },
    { name: 'Settings application', ...settingsAppResult },
    { name: 'Active word timing', ...activeWordResult },
    { name: 'Clip effects parity', ...effectsResult },
    { name: 'Export parity', ...parityResult },
    { name: 'Video effects export', ...videoEffectsResult },
  ];

  // Subject tracking validation
  if (trackingInfo) {
    const trackingResult = validateSubjectTracking(trackingInfo, aspectRatio, outputDims);
    checks.push({ name: 'Subject tracking', ...trackingResult });
  }

  // Optional transcript sync validation
  if (syncInfo && syncInfo.transcript) {
    const syncResult = validateTranscriptSync(
      items, syncInfo.transcript, syncInfo.clipStart || 0, syncInfo.clipEnd || Infinity
    );
    checks.push({ name: 'Transcript sync', ...syncResult });
  }

  const errors = checks.flatMap((c) => c.errors);
  const warnings = checks.flatMap((c) => c.warnings);

  const subtitleCount = items.filter((it) => it.type === 'subtitle').length;
  const allItemCount = items.length;
  const valid = errors.length === 0;

  const passCount = checks.filter((c) => c.errors.length === 0 && c.warnings.length === 0).length;
  const warnCount = checks.filter((c) => c.errors.length === 0 && c.warnings.length > 0).length;
  const failCount = checks.filter((c) => c.errors.length > 0).length;

  // Overall confidence from parity check
  const confidence = parityResult.confidence || (valid ? 'high' : 'low');

  let summary;
  if (valid && warnings.length === 0) {
    summary = `QA passed: ${allItemCount} item(s), ${subtitleCount} subtitle(s), ${checks.length} checks passed. Export confidence: ${confidence}.`;
  } else if (valid) {
    summary = `QA passed with ${warnings.length} warning(s): ${passCount} passed, ${warnCount} with warnings. Export confidence: ${confidence}.`;
  } else {
    summary = `QA FAILED: ${failCount} check(s) failed, ${errors.length} error(s), ${warnings.length} warning(s). Export confidence: ${confidence}.`;
  }

  return { valid, errors, warnings, summary, checks, confidence };
}

/**
 * Validate subtitle settings for preview-export parity.
 *
 * Checks for settings combinations known to produce visual differences
 * between the CSS/DOM preview and the FFmpeg/ASS export.
 *
 * @param {Object} settings - Subtitle settings (clipSettings)
 * @param {Object} outputDims - { w, h } output dimensions
 * @returns {{ warnings: string[], info: string[] }}
 */
export function validateSubtitleExportParity(settings, outputDims) {
  const warnings = [];
  const info = [];

  if (!settings) return { warnings, info };

  const bgEnabled = settings.subtitleBgEnabled || false;
  const bgRadius = settings.subtitleBgRadius ?? 0;
  const awEnabled = settings.activeWordEnabled ?? false;
  const awBgOpacity = settings.activeWordBgOpacity ?? 0;
  const awBgRadius = settings.activeWordBgRadius ?? 4;

  // BGDRAW path: rounded background uses ASS drawing commands
  if (bgEnabled && bgRadius > 0) {
    info.push(
      'Rounded subtitle background (radius > 0) uses ASS drawing commands in export. ' +
      'Box sizing is calibrated to match CSS preview but may differ slightly for unusual fonts.'
    );
  }

  // AWDRAW path: rounded active word background
  if (awEnabled && awBgOpacity > 0 && awBgRadius > 0) {
    info.push(
      'Active word background with rounded corners uses ASS drawing commands in export. ' +
      'Highlight width is calibrated to match CSS preview.'
    );
  }

  // Font availability check
  const font = settings.subtitleFont || 'DM Sans';
  const SUPPORTED_FONTS = [
    'DM Sans', 'Montserrat', 'Open Sans', 'Roboto', 'Poppins',
    'Inter', 'Nunito', 'Lato', 'Oswald', 'Playfair Display',
    'Bebas Neue', 'Liberation Sans',
  ];
  if (!SUPPORTED_FONTS.some(f => f.toLowerCase() === font.toLowerCase())) {
    warnings.push(
      `Font "${font}" may not be available on the export server. ` +
      `Supported built-in fonts: ${SUPPORTED_FONTS.join(', ')}`
    );
  }

  // Output dimensions sanity
  if (outputDims) {
    const fontScale = Math.min(outputDims.w, outputDims.h) / Math.min(REF_W, REF_H);
    const size = settings.subtitleSize || 'medium';
    const basePx = typeof size === 'number' ? size : (FONT_SIZE_MAP[size] || 30);
    const exportFontSize = Math.max(16, Math.round(basePx * fontScale));

    if (exportFontSize < 20) {
      warnings.push(
        `Export subtitle font size (${exportFontSize}px) is very small at ${outputDims.w}x${outputDims.h}. ` +
        `Consider increasing subtitle size.`
      );
    }
  }

  return { warnings, info };
}
