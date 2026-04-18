/**
 * RenderEngine — Canvas-based compositor for preview and export.
 *
 * THE #1 RULE: the same renderFrame() function is called for both
 * real-time preview (via rAF) and stepped export (via ExportEngine).
 * This guarantees pixel-perfect WYSIWYG — what the user sees in
 * preview is exactly what appears in the exported MP4.
 */

// Compositing combines TYPE priority (subtitle > overlay > video >
// audio) with track-array index. Imported lazily so this module
// remains testable in isolation without a Zustand store.
import { getCompositingOrder as _compositingOrder } from '../stores/timelineStore';

// ── Builtin font URL map (mirrors SubtitleOverlay / ClipSettingsPanel) ────
const BUILTIN_FONT_FILES = {
  'DM Sans': '/api/fonts/builtin/DMSans.ttf',
  'Montserrat': '/api/fonts/builtin/Montserrat.ttf',
  'Open Sans': '/api/fonts/builtin/OpenSans.ttf',
  'Roboto': '/api/fonts/builtin/Roboto.ttf',
  'Poppins': '/api/fonts/builtin/Poppins-Regular.ttf',
  'Inter': '/api/fonts/builtin/Inter.ttf',
  'Nunito': '/api/fonts/builtin/Nunito.ttf',
  'Lato': '/api/fonts/builtin/Lato-Regular.ttf',
  'Oswald': '/api/fonts/builtin/Oswald.ttf',
  'Playfair Display': '/api/fonts/builtin/PlayfairDisplay.ttf',
  'Bebas Neue': '/api/fonts/builtin/BebasNeue-Regular.ttf',
  'Liberation Sans': '/api/fonts/builtin/LiberationSans-Regular.ttf',
};

// Bold-specific font files for static-weight fonts that need separate bold files.
// Variable-weight fonts (DM Sans, Montserrat, Inter, etc.) don't need entries here
// because a single file covers all weights when loaded with the correct descriptor.
const BUILTIN_FONT_BOLD_FILES = {
  'Poppins': '/api/fonts/builtin/Poppins-Bold.ttf',
  'Lato': '/api/fonts/builtin/Lato-Bold.ttf',
  'Liberation Sans': '/api/fonts/builtin/LiberationSans-Bold.ttf',
};

// Variable-weight fonts — these contain all weights in a single file.
// Set detected by URL pattern (encoded brackets in Google Fonts URLs).
const VARIABLE_WEIGHT_FONTS = new Set([
  'DM Sans', 'Montserrat', 'Open Sans', 'Roboto', 'Inter',
  'Nunito', 'Oswald', 'Playfair Display',
]);

const DEFAULT_SPEAKER_PALETTE = [
  '#00D9FF', '#F59E0B', '#10B981', '#A78BFA', '#EF4444', '#EC4899',
  '#06B6D4', '#8B5CF6', '#F97316', '#14B8A6', '#E879F9', '#84CC16',
  '#FB7185', '#38BDF8', '#FBBF24', '#34D399', '#C084FC', '#F472B6',
  '#22D3EE', '#A3E635', '#FB923C', '#2DD4BF', '#818CF8', '#F87171',
];

const FONT_SIZE_MAP = { small: 22, medium: 30, large: 40 };
const REF_W = 1920;
const REF_H = 1080;

// ── Active word timing constants (MUST match SubtitleOverlay exactly) ────
const _BASE_OVERHEAD_S = 0.04;
const _ANTICIPATION_S  = 0.10;
const _AUDIO_BUFFER_S  = 0.12;
const _PUNCT_PAUSE = { ',': 0.15, ';': 0.16, ':': 0.12, '.': 0.22, '!': 0.22, '?': 0.24, '\u2014': 0.12, '\u2013': 0.10 };
const _FAST_WORDS = new Set([
  'the', 'a', 'an', 'to', 'in', 'on', 'at', 'of', 'for',
  'and', 'but', 'or', 'is', 'was', 'are', 'were', 'it',
  'its', 'this', 'that',
]);

/**
 * Compute the active word index for a subtitle segment at a given time.
 * This is the SAME algorithm used by SubtitleOverlay's getCurrentWordIndex
 * to guarantee 1:1 parity between preview and export.
 */
function computeActiveWordIndex(segment, relativeTime, speakerRates) {
  const text = segment?.subtitleText || segment?.text || '';
  if (!text) return -1;
  const words = text.split(/\s+/).filter(Boolean);
  if (words.length <= 1) return words.length === 1 ? 0 : -1;

  // Use word-level timestamps if available.
  //
  // Coordinate-agnostic compare: word timestamps may be **timeline-
  // absolute** (the convention Whisper produces) or **clip-relative**
  // (some legacy paths offset them). We auto-detect by checking
  // whether the first word's start lies inside the segment's
  // [start, end] window. The active-word HIGHLIGHT then uses the
  // same convention as the segment, eliminating the
  // ``preview ≠ export`` drift the audit flagged.
  if (segment.words && segment.words.length === words.length) {
    const w0 = segment.words[0];
    const wordsAreClipRelative = w0.start < segment.start - 0.01;
    const baseT = wordsAreClipRelative
      ? (relativeTime - segment.start)
      : relativeTime;
    // Match the preview's per-speaker rate-scaled anticipation
    // (SubtitleOverlay.getCurrentWordIndex word-ts branch). Without
    // this, the canvas export highlights at fixed 0.10s anticipation
    // while the preview scales by speaker rate — slow speakers got
    // an early highlight in preview but a late one in export.
    const speakerWpsW = (speakerRates && speakerRates[segment.speaker]) || 3.0;
    const rateScaleW = Math.max(0.6, Math.min(1.6, 3.0 / speakerWpsW));
    const anticipationW = _ANTICIPATION_S * rateScaleW;
    const adjusted = baseT + anticipationW - _AUDIO_BUFFER_S;
    if (adjusted < w0.start) return -1;
    for (let i = 0; i < segment.words.length; i++) {
      if (adjusted < segment.words[i].end) return i;
    }
    return segment.words.length - 1;
  }

  // Proportional timing fallback (matches SubtitleOverlay exactly)
  const totalChars = words.reduce((sum, w) => sum + w.length, 0);
  if (totalChars === 0) return -1;
  const segDuration = segment.end - segment.start;
  const speakerWps = (speakerRates && speakerRates[segment.speaker]) || 3.0;
  const rateScale = Math.max(0.6, Math.min(1.6, 3.0 / speakerWps));
  const anticipation = _ANTICIPATION_S * rateScale;
  const elapsed = (relativeTime - segment.start) + anticipation - _AUDIO_BUFFER_S;
  if (elapsed < 0) return -1;

  const punctPauses = words.map((w) => {
    const last = w[w.length - 1];
    return (_PUNCT_PAUSE[last] || 0) * rateScale;
  });
  const totalPunct = punctPauses.reduce((a, b) => a + b, 0);
  const baseOverhead = _BASE_OVERHEAD_S * rateScale * words.length;
  const totalPause = baseOverhead + totalPunct;
  const charTime = Math.max(segDuration - totalPause, segDuration * 0.45);
  const pauseScale = (segDuration - charTime) / Math.max(totalPause, 0.01);

  let t = 0;
  for (let i = 0; i < words.length; i++) {
    const charDur = charTime * (words[i].length / totalChars);
    const pause = (_BASE_OVERHEAD_S * rateScale + punctPauses[i]) * pauseScale;
    let wordDur = charDur + pause;
    const stripped = words[i].toLowerCase().replace(/[.,!?;:\u2014\u2013]+$/, '');
    if (_FAST_WORDS.has(stripped)) wordDur *= 0.75;
    if (i === 0) wordDur *= 1.15;
    else if (i === words.length - 1) wordDur *= 1.10;
    if (elapsed < t + wordDur) return i;
    t += wordDur;
  }
  return words.length - 1;
}

/**
 * Compute per-speaker word rates from subtitle segments.
 * Matches SubtitleOverlay's computeSpeakerRates exactly.
 */
function computeSpeakerRates(segments) {
  const stats = {};
  for (const seg of segments) {
    const wc = (seg.text || seg.subtitleText || '').split(/\s+/).filter(Boolean).length;
    const dur = seg.end - seg.start;
    if (dur <= 0 || wc === 0) continue;
    if (!stats[seg.speaker]) stats[seg.speaker] = { words: 0, time: 0 };
    stats[seg.speaker].words += wc;
    stats[seg.speaker].time += dur;
  }
  const rates = {};
  for (const [sp, s] of Object.entries(stats)) {
    rates[sp] = s.time > 0 ? s.words / s.time : 3.0;
  }
  return rates;
}

/**
 * Get speaker color matching SubtitleOverlay's getSpeakerColor logic.
 */
function getSpeakerColor(speaker, speakersOrdered, settings) {
  const fontColor = settings?.subtitleFontColor;
  const useSpeaker = settings?.useSpeakerColors ?? true;
  // When speaker colors are enabled, they override the font color picker
  if (useSpeaker) {
    const speakerColors = settings?.speakerColors || {};
    if (speakerColors[speaker]) return speakerColors[speaker];
    const idx = speakersOrdered.indexOf(speaker);
    return DEFAULT_SPEAKER_PALETTE[(idx >= 0 ? idx : 0) % DEFAULT_SPEAKER_PALETTE.length];
  }
  // Speaker colors off — use explicit font color or default white
  return fontColor || '#FFFFFF';
}

// ── Transition renderers ──────────────────────────────────────────────────
const TRANSITIONS = {
  dissolve(ctx, outCanvas, inCanvas, progress, w, h) {
    ctx.globalAlpha = 1 - progress;
    ctx.drawImage(outCanvas, 0, 0, w, h);
    ctx.globalAlpha = progress;
    ctx.drawImage(inCanvas, 0, 0, w, h);
    ctx.globalAlpha = 1;
  },
  fade(ctx, outCanvas, inCanvas, progress, w, h) {
    if (progress < 0.5) {
      ctx.globalAlpha = 1 - progress * 2;
      ctx.drawImage(outCanvas, 0, 0, w, h);
    } else {
      ctx.globalAlpha = (progress - 0.5) * 2;
      ctx.drawImage(inCanvas, 0, 0, w, h);
    }
    ctx.globalAlpha = 1;
  },
  'wipe-left'(ctx, outCanvas, inCanvas, progress, w, h) {
    const splitX = w * progress;
    ctx.drawImage(outCanvas, 0, 0, w, h);
    ctx.save();
    ctx.beginPath();
    ctx.rect(0, 0, splitX, h);
    ctx.clip();
    ctx.drawImage(inCanvas, 0, 0, w, h);
    ctx.restore();
  },
  'wipe-right'(ctx, outCanvas, inCanvas, progress, w, h) {
    const splitX = w * (1 - progress);
    ctx.drawImage(outCanvas, 0, 0, w, h);
    ctx.save();
    ctx.beginPath();
    ctx.rect(splitX, 0, w - splitX, h);
    ctx.clip();
    ctx.drawImage(inCanvas, 0, 0, w, h);
    ctx.restore();
  },
  'slide-left'(ctx, outCanvas, inCanvas, progress, w, h) {
    const offset = w * progress;
    ctx.drawImage(outCanvas, -offset, 0, w, h);
    ctx.drawImage(inCanvas, w - offset, 0, w, h);
  },
  'slide-right'(ctx, outCanvas, inCanvas, progress, w, h) {
    const offset = w * progress;
    ctx.drawImage(outCanvas, offset, 0, w, h);
    ctx.drawImage(inCanvas, -w + offset, 0, w, h);
  },
  zoom(ctx, outCanvas, inCanvas, progress, w, h) {
    const scale = 1 + progress * 0.3;
    ctx.globalAlpha = 1 - progress;
    ctx.save();
    ctx.translate(w / 2, h / 2);
    ctx.scale(scale, scale);
    ctx.translate(-w / 2, -h / 2);
    ctx.drawImage(outCanvas, 0, 0, w, h);
    ctx.restore();
    ctx.globalAlpha = progress;
    ctx.drawImage(inCanvas, 0, 0, w, h);
    ctx.globalAlpha = 1;
  },
};

export default class RenderEngine {
  constructor(canvas, options = {}) {
    this.canvas = canvas;
    // Request GPU-accelerated canvas — willReadFrequently: false lets Chrome
    // keep the canvas on the GPU (hardware composited) instead of forcing
    // software readback on every frame.
    this.ctx = canvas.getContext('2d', { willReadFrequently: false }) || canvas.getContext('2d');
    this.width = options.width || 1920;
    this.height = options.height || 1080;
    this._fontCache = new Set();
    this._transitionBuffer1 = null;
    this._transitionBuffer2 = null;

    // Cached speaker data for active word highlighting (computed once per export)
    this._speakerRates = null;
    this._speakersOrdered = null;
    this._subtitleSegments = null;
  }

  /**
   * Pre-compute speaker data for a set of subtitle clips.
   * Call this once before export to enable accurate active word rendering.
   */
  prepareSubtitleContext(allClips) {
    const subtitles = (allClips || []).filter(c => c.type === 'subtitle');
    this._subtitleSegments = subtitles;
    this._speakerRates = computeSpeakerRates(subtitles);
    const seen = [];
    for (const seg of subtitles) {
      if (seg.speaker && !seen.includes(seg.speaker)) seen.push(seg.speaker);
    }
    this._speakersOrdered = seen;
  }

  /**
   * Set output resolution. Called when aspect ratio changes.
   */
  setResolution(width, height) {
    this.width = width;
    this.height = height;
    this.canvas.width = width;
    this.canvas.height = height;
    this._transitionBuffer1 = null;
    this._transitionBuffer2 = null;
  }

  /**
   * Pre-load a font via FontFace API so canvas text rendering works.
   * Resolves custom fonts from /api/fonts if not in the builtin map.
   *
   * @param {string} fontName - Font family name
   * @param {string} [url] - Optional explicit URL
   * @param {number} [weight] - Font weight (400, 700, etc.)
   */
  async loadFont(fontName, url, weight) {
    const isBold = weight && weight >= 600;
    const cacheKey = isBold ? `${fontName}__bold` : fontName;
    if (this._fontCache.has(cacheKey)) return;
    try {
      let fontUrl = url;
      if (!fontUrl) {
        // For bold weights, check if a separate bold file exists (static fonts)
        if (isBold && BUILTIN_FONT_BOLD_FILES[fontName]) {
          fontUrl = BUILTIN_FONT_BOLD_FILES[fontName];
        } else {
          fontUrl = BUILTIN_FONT_FILES[fontName];
        }
      }
      // For custom fonts not in the builtin map, look up via /api/fonts
      if (!fontUrl) {
        if (!this._customFontMap) {
          try {
            const res = await fetch('/api/fonts');
            const fonts = res.ok ? await res.json() : [];
            this._customFontMap = {};
            for (const f of fonts) {
              if (f.name && f.url) this._customFontMap[f.name] = f.url;
            }
          } catch {
            this._customFontMap = {};
          }
        }
        fontUrl = this._customFontMap[fontName];
      }
      if (!fontUrl) return;

      // Set FontFace weight descriptor for proper canvas font matching
      const descriptors = {};
      if (VARIABLE_WEIGHT_FONTS.has(fontName)) {
        // Variable fonts cover all weights in a single file
        descriptors.weight = '1 1000';
      } else if (isBold) {
        descriptors.weight = 'bold';
      }

      const face = new FontFace(fontName, `url(${fontUrl})`, descriptors);
      await face.load();
      document.fonts.add(face);
      this._fontCache.add(cacheKey);
      this._fontCache.add(fontName); // Also cache base name for fire-and-forget lookups
    } catch (err) {
      console.warn(`[RenderEngine] Font load failed: ${fontName} (weight=${weight})`, err);
    }
  }

  /**
   * Pre-load all fonts needed by a set of clips.
   * Call before export to ensure all fonts are ready for canvas rendering.
   */
  async ensureFontsLoaded(clips, settings) {
    const needed = new Map(); // key: "family|weight" → {family, weight}
    for (const clip of clips) {
      if (clip.type === 'text' && clip.textStyle?.fontFamily) {
        const fw = clip.textStyle.fontWeight || 400;
        const key = `${clip.textStyle.fontFamily}|${fw}`;
        if (!needed.has(key)) {
          needed.set(key, { family: clip.textStyle.fontFamily, weight: fw });
        }
      }
      if (clip.type === 'subtitle') {
        const subS = settings?.subtitle || settings || {};
        const fn = subS.subtitleFont || 'DM Sans';
        const fw = typeof subS.subtitleFontWeight === 'number' ? subS.subtitleFontWeight :
                   subS.subtitleFontWeight === 'bold' ? 700 :
                   subS.subtitleFontWeight === 'black' ? 900 : 400;
        const key = `${fn}|${fw}`;
        if (!needed.has(key)) {
          needed.set(key, { family: fn, weight: fw });
        }
      }
    }
    for (const { family, weight } of needed.values()) {
      await this.loadFont(family, null, weight);
    }
    await document.fonts.ready;
  }

  /**
   * Core render function — composites all visible items at the given time.
   *
   * @param {number} currentTime - Playback time in seconds
   * @param {Array} tracks - Track definitions from store
   * @param {Array} clips - All clip/item objects from store
   * @param {Object} settings - Project settings (backgroundColor, subtitleSettings, etc.)
   * @param {Map} mediaElements - Map of assetId → HTMLVideoElement | HTMLImageElement
   */
  renderFrame(currentTime, tracks, clips, settings, mediaElements) {
    const { ctx, width, height } = this;

    // Clear and fill background
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = settings.backgroundColor || '#000000';
    ctx.fillRect(0, 0, width, height);

    // Ensure fonts for text/subtitle clips are loaded (fire-and-forget — the
    // font will be available on the next frame if not yet cached).
    for (const clip of clips) {
      if (clip.type === 'text') {
        const fn = clip.textStyle?.fontFamily;
        if (fn && !this._fontCache.has(fn)) this.loadFont(fn);
      } else if (clip.type === 'subtitle') {
        const subS = settings?.subtitle || settings || {};
        const fn = subS.subtitleFont || 'DM Sans';
        if (!this._fontCache.has(fn)) this.loadFont(fn);
      }
    }

    // Collect visible clips at currentTime, sorted by track position (bottom-to-top).
    // Track position in the timeline UI is the single source of truth for z-ordering:
    // tracks at the TOP of the UI (lower array index) render ON TOP in the preview
    // and export. This matches the mental model: what you see stacked higher in
    // the timeline appears in front.
    //
    // In export mode, track.visible is ignored — all tracks are included in the
    // exported video. Track visibility is a preview-only feature (like solo/mute
    // in DAWs vs NLEs: DaVinci Resolve, Premiere Pro treat visibility as preview-only).
    // Visibility vs audio: ``track.muted`` historically suppressed the
    // entire layer (video pixels AND audio). That breaks the "mute
    // audio without hiding the picture" workflow every NLE supports.
    // We now prefer the explicit pair ``track.audioMuted`` + ``track.
    // videoVisible``; legacy ``track.muted`` falls back to ``audioMuted``
    // only so old projects keep working without going black.
    const visibleClips = [];
    for (const clip of clips) {
      if (currentTime >= clip.start && currentTime < clip.end) {
        const track = tracks.find(t => t.id === clip.trackId);
        if (!track) continue;
        // Resolve the new fields with legacy fallback.
        const videoVisible = track.videoVisible !== undefined
          ? track.videoVisible
          : (track.visible !== false);
        // ``audioMuted`` is the new explicit field. The legacy
        // ``muted`` flag now only mutes audio (no longer hides video).
        // Note: actual audio routing happens in ExportEngine /
        // PlaybackEngine; here we just need to keep the pixels visible.
        if (this._exportMode || videoVisible) {
          // Two-axis ordering via getCompositingOrder: TYPE priority
          // first (subtitle > overlay > video > audio) then UI index.
          // Falls back to a pure-index order if the helper isn't
          // present so this module stays usable in isolation.
          let order = 0;
          if (typeof _compositingOrder === 'function') {
            order = _compositingOrder(track, tracks);
          } else {
            const trackIdx = tracks.findIndex(t => t.id === track.id);
            order = trackIdx >= 0 ? tracks.length - trackIdx : 0;
          }
          visibleClips.push({ clip, track, order });
        }
      }
    }
    visibleClips.sort((a, b) => a.order - b.order);

    // Render each visible clip bottom-to-top
    for (const { clip, track } of visibleClips) {
      try {
        this._renderClip(ctx, clip, track, currentTime, settings, mediaElements, clips);
      } catch {
        // Skip failed clips gracefully
      }
    }
  }

  _renderClip(ctx, clip, track, currentTime, settings, mediaElements, allClips) {
    const { width, height } = this;

    // Check for transition on this clip's leading edge
    if (clip.transition && clip.transition.duration > 0) {
      const transStart = clip.start;
      const transDur = clip.transition.duration;
      const transEnd = transStart + transDur;

      if (currentTime >= transStart && currentTime < transEnd) {
        const progress = (currentTime - transStart) / transDur;
        this._renderTransition(ctx, clip, track, currentTime, progress, settings, mediaElements, allClips);
        return;
      }
    }

    ctx.save();

    // Apply clip opacity with fadeIn/fadeOut
    let opacity = clip.opacity ?? 1;
    const clipDur = clip.end - clip.start;
    const elapsed = currentTime - clip.start;
    if (clip.fadeIn > 0 && elapsed < clip.fadeIn) {
      opacity *= elapsed / clip.fadeIn;
    }
    if (clip.fadeOut > 0 && (clipDur - elapsed) < clip.fadeOut) {
      opacity *= (clipDur - elapsed) / clip.fadeOut;
    }
    if (opacity < 1) {
      ctx.globalAlpha = opacity;
    }

    // Build filter string from clip effects
    const filterStr = this._buildFilterString(clip.effects);
    if (filterStr) {
      ctx.filter = filterStr;
    }

    switch (clip.type) {
      case 'video':
        this._renderVideo(ctx, clip, currentTime, settings, mediaElements);
        break;
      case 'audio':
        // Audio clips don't render visually
        break;
      case 'image':
      case 'overlay':
        this._renderImage(ctx, clip, mediaElements);
        break;
      case 'text':
        this._renderText(ctx, clip, currentTime);
        break;
      case 'subtitle':
        this._renderSubtitle(ctx, clip, currentTime, settings);
        break;
      case 'shape':
        this._renderShape(ctx, clip);
        break;
    }

    ctx.filter = 'none';
    ctx.globalAlpha = 1;
    ctx.restore();
  }

  /**
   * Prepare subject tracking keyframes for dynamic crop positioning.
   * Call this once before export (or per-clip) so _renderVideo can
   * interpolate the subject position at each frame time.
   *
   * @param {Array} scenes - Scene objects with {timestamp, subject_x}
   * @param {number} clipStart - Clip start time (absolute seconds)
   * @param {number} clipEnd - Clip end time (absolute seconds)
   * @param {number|null} srcRatio - Source video aspect ratio
   * @param {number|null} targetRatio - Target crop aspect ratio
   */
  prepareSubjectTracking(scenes, clipStart, clipEnd, srcRatio, targetRatio) {
    // Lazy-import to avoid circular deps — these are the same functions
    // used by VideoPlayer and backend for parity
    if (!this._stImported) {
      try {
        // Import will be set up by the caller via setSubjectTrackingFns
        this._stImported = true;
      } catch { /* noop */ }
    }
    if (this._processKeyframes && scenes?.length) {
      this._subjectKeyframes = this._processKeyframes(scenes, clipStart, clipEnd, srcRatio, targetRatio);
      this._subjectClipStart = clipStart;
    } else {
      this._subjectKeyframes = null;
      this._subjectClipStart = 0;
    }
  }

  /**
   * Inject subject tracking functions so RenderEngine can perform
   * dynamic keyframe interpolation matching the preview player exactly.
   *
   * @param {Function} processKeyframes - Full pipeline: build→cuts→compress→deadzone→smooth→holds
   * @param {Function} interpolateSubjectX - Smoothstep interpolation at time t
   */
  setSubjectTrackingFns(processKeyframes, interpolateSubjectX) {
    this._processKeyframes = processKeyframes;
    this._interpolateSubjectX = interpolateSubjectX;
    this._stImported = true;
  }

  /**
   * Set layout data for multi-speaker compositing.
   * @param {Array|null} layoutTimeline - [{start, end, layout_mode, ...}]
   * @param {object|null} faceRegistry - {slots: [{id, x, frames}]}
   * @param {string} defaultMode - "single", "split", etc.
   */
  setLayoutData(layoutTimeline, faceRegistry, defaultMode) {
    this._layoutTimeline = layoutTimeline;
    this._faceRegistry = faceRegistry;
    this._defaultLayoutMode = defaultMode || 'single';
  }

  _renderVideo(ctx, clip, currentTime, settings, mediaElements) {
    const mediaEl = mediaElements?.get(clip.mediaRef || clip.id);
    if (!mediaEl || !(mediaEl instanceof HTMLVideoElement)) return;

    const { width, height } = this;
    const vw = mediaEl.videoWidth || width;
    const vh = mediaEl.videoHeight || height;

    // Layout-aware rendering: split mode draws two halves from same video
    if (this._defaultLayoutMode === 'split' && this._faceRegistry?.slots?.length >= 2) {
      const sortedSlots = [...this._faceRegistry.slots].sort((a, b) => a.x - b.x);
      const separatorPx = 3;
      const halfH = Math.floor((height - separatorPx) / 2);
      const srcAR = vw / vh;
      const halfAR = width / halfH;

      // Each half: crop from source centered on speaker
      for (let i = 0; i < 2; i++) {
        const speakerX = sortedSlots[i]?.x ?? 50;
        let csx = 0, csy = 0, csw = vw, csh = vh;
        if (srcAR > halfAR) {
          csw = Math.round(vh * halfAR);
          const subjectPx = vw * speakerX / 100;
          csx = Math.round(Math.max(0, Math.min(vw - csw, subjectPx - csw / 2)));
        } else {
          csh = Math.round(vw / halfAR);
          csy = Math.round((vh - csh) / 2);
        }
        const dy = i === 0 ? 0 : halfH + separatorPx;
        ctx.drawImage(mediaEl, csx, csy, csw, csh, 0, dy, width, halfH);
      }

      // Draw separator line
      ctx.fillStyle = 'rgba(0,0,0,0.3)';
      ctx.fillRect(0, halfH, width, separatorPx);
      return;
    }

    // ── Gaming layout modes (Phase 5) ──
    // When a render plan op carries ``gaming_layout_mode`` (the
    // chooser populated it for gameplay clips), draw using the
    // matching mode. The pixel math mirrors
    // ``backend/services/gameplay_geometry.py`` so the FFmpeg
    // export and this Canvas preview agree to the pixel.
    const gamingMode = clip.gamingLayoutMode || clip.gaming_layout_mode;
    if (gamingMode === 'blurfill' || gamingMode === 'wide_zoom' || gamingMode === 'fullscreen') {
      const camXPct = (clip.subjectX ?? settings.subjectX ?? 50);
      this._renderGamingLayout(ctx, mediaEl, vw, vh, gamingMode, camXPct);
      return;
    }

    // Calculate crop region for aspect ratio
    const srcAR = vw / vh;
    const dstAR = width / height;

    let sx = 0, sy = 0, sw = vw, sh = vh;

    if (Math.abs(srcAR - dstAR) > 0.01) {
      // Need to crop source to match destination AR
      if (srcAR > dstAR) {
        // Source is wider — crop horizontally
        sw = Math.round(vh * dstAR);

        // Compute subject position: use dynamic keyframe interpolation
        // if available (matches VideoPlayer/backend exactly), otherwise
        // fall back to static clip.subjectX.
        let subjectPct = clip.subjectX ?? settings.subjectX ?? 50;
        if (this._subjectKeyframes && this._interpolateSubjectX) {
          const relTime = currentTime - (this._subjectClipStart || 0);
          subjectPct = this._interpolateSubjectX(this._subjectKeyframes, relTime);
        }

        // Use centering formula matching backend _center_crop_offset():
        // Place crop window so subject pixel ends up at crop center.
        // subject_pixel = vw * subjectPct / 100
        // x_offset = subject_pixel - crop_width / 2, clamped to [0, vw - sw]
        const subjectPixel = vw * subjectPct / 100;
        const maxOffset = vw - sw;
        sx = Math.round(Math.max(0, Math.min(maxOffset, subjectPixel - sw / 2)));
      } else {
        // Source is taller — crop vertically
        sh = Math.round(vw / dstAR);
        sy = Math.round((vh - sh) / 2);
      }
    }

    // Apply transform (position, size, rotation)
    const transform = clip.transform || {};
    const pos = clip.position || { x: 50, y: 50 };
    const size = clip.size || { w: 100, h: 100 };
    const rotation = transform.rotation ?? 0;
    const scaleX = transform.scaleX ?? 1;
    const scaleY = transform.scaleY ?? 1;

    // Check if user has custom position/size (non-default)
    const effectivePos = (pos.x === 0 && pos.y === 0) ? { x: 50, y: 50 } : pos;
    const hasCustomLayout = effectivePos.x !== 50 || effectivePos.y !== 50 ||
      size.w !== 100 || size.h !== 100 || rotation !== 0 ||
      scaleX !== 1 || scaleY !== 1;

    if (hasCustomLayout) {
      // Custom position/size/rotation: draw video at specified position and size
      const drawW = (size.w / 100) * width;
      const drawH = (size.h / 100) * height;
      const cx = (effectivePos.x / 100) * width;
      const cy = (effectivePos.y / 100) * height;

      ctx.save();
      ctx.translate(cx, cy);
      if (rotation !== 0) ctx.rotate((rotation * Math.PI) / 180);
      ctx.scale(scaleX, scaleY);
      ctx.drawImage(mediaEl, sx, sy, sw, sh, -drawW / 2, -drawH / 2, drawW, drawH);
      ctx.restore();
    } else {
      // Default: fill entire canvas
      const dx = (transform.x || 0) * width / 100;
      const dy = (transform.y || 0) * height / 100;
      if (dx !== 0 || dy !== 0) {
        ctx.save();
        ctx.translate(width / 2 + dx, height / 2 + dy);
        ctx.drawImage(mediaEl, sx, sy, sw, sh, -width / 2, -height / 2, width, height);
        ctx.restore();
      } else {
        ctx.drawImage(mediaEl, sx, sy, sw, sh, 0, 0, width, height);
      }
    }
  }

  /**
   * Phase 5 — gaming layout renderer.
   *
   * Mirrors the rect math in
   * ``backend/services/gameplay_geometry.py`` so this Canvas
   * preview and the FFmpeg export stay within ±2 px of each
   * other for every gaming layout mode.
   *
   * Modes:
   *   - "fullscreen": center crop with cam_x_pct
   *   - "blurfill":   sharp 16:9 band over a blurred fill
   *   - "wide_zoom":  wider crop showing more horizontal context
   */
  _renderGamingLayout(ctx, mediaEl, vw, vh, mode, camXPct) {
    const { width: tw, height: th } = this;

    const evenRound = (n) => {
      const r = Math.round(n);
      return r - (r % 2);
    };

    if (mode === 'blurfill') {
      // Band is 16:9 source scaled to target_w
      const bandW = tw;
      const bandH = evenRound(tw * vh / Math.max(vw, 1));
      let bandY = Math.floor((th - bandH) / 2);
      bandY -= bandY % 2;

      // Blur layer: scale source to FILL target_h, then crop center
      // and apply CSS blur via offscreen canvas.
      ctx.save();
      const off = document.createElement('canvas');
      off.width = tw;
      off.height = th;
      const offCtx = off.getContext('2d');
      const fillScale = Math.max(tw / vw, th / vh);
      const fillW = vw * fillScale;
      const fillH = vh * fillScale;
      const fillX = (tw - fillW) / 2;
      const fillY = (th - fillH) / 2;
      offCtx.drawImage(mediaEl, fillX, fillY, fillW, fillH);
      ctx.filter = 'blur(20px)';
      ctx.drawImage(off, 0, 0);
      ctx.filter = 'none';
      ctx.restore();

      // Sharp band on top
      ctx.drawImage(mediaEl, 0, 0, vw, vh, 0, bandY, bandW, bandH);
      return;
    }

    // fullscreen + wide_zoom share the cropped-source path
    const targetAspect = tw / Math.max(th, 1);
    const cropScale = mode === 'wide_zoom' ? 1.25 : 1.0;
    let cropW = evenRound(vh * targetAspect * cropScale);
    cropW = Math.min(cropW, vw);
    const camPx = vw * camXPct / 100;
    let cropX = Math.max(0, Math.min(vw - cropW, Math.round(camPx - cropW / 2)));
    ctx.drawImage(mediaEl, cropX, 0, cropW, vh, 0, 0, tw, th);
  }

  _renderImage(ctx, clip, mediaElements) {
    const mediaEl = mediaElements?.get(clip.mediaRef || clip.id);
    if (!mediaEl) return;

    const { width, height } = this;
    const transform = clip.transform || {};
    // pos is center-based (DOM uses left/top + translate(-50%, -50%))
    const pos = clip.position || { x: 50, y: 50 };
    const size = clip.size || { w: 30, h: 30 };

    const cx = (pos.x / 100) * width;
    const cy = (pos.y / 100) * height;
    const dw = (size.w / 100) * width;
    const dh = (size.h / 100) * height;
    const rotation = transform.rotation ?? 0;

    ctx.save();
    ctx.translate(cx, cy);
    if (rotation !== 0) ctx.rotate((rotation * Math.PI) / 180);
    ctx.drawImage(mediaEl, -dw / 2, -dh / 2, dw, dh);
    ctx.restore();
  }

  _renderText(ctx, clip, currentTime) {
    const { width, height } = this;
    const text = clip.textContent || clip.subtitleText || '';
    if (!text) return;

    const style = clip.textStyle || {};
    const fontSize = style.fontSize || 48;
    const fontFamily = style.fontFamily || 'DM Sans';
    const fontWeight = style.fontWeight || 400;
    const color = style.color || '#FFFFFF';
    const align = style.textAlign || 'center';
    const pos = clip.position || { x: 50, y: 50 };
    const size = clip.size || { w: 80, h: 20 };
    const rotation = (clip.transform?.rotation) ?? 0;

    ctx.save();

    // Animation
    let animAlpha = 1;
    let animOffsetY = 0;
    let animScale = 1;
    const anim = style.animation || 'none';
    const clipDur = clip.end - clip.start;
    const elapsed = currentTime - clip.start;

    if (anim === 'fade-in' && elapsed < 0.5) {
      animAlpha = elapsed / 0.5;
    } else if (anim === 'slide-up' && elapsed < 0.5) {
      animOffsetY = (1 - elapsed / 0.5) * 30;
      animAlpha = elapsed / 0.5;
    } else if (anim === 'pop' && elapsed < 0.3) {
      const t = elapsed / 0.3;
      animScale = 0.5 + 0.5 * (t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2);
    } else if (anim === 'bounce' && elapsed < 0.6) {
      // Match the DOM preview in TimelineOverlay.jsx: a damped sine
      // wave that overshoots then settles. Without this branch, the
      // export would render plain static text while the preview
      // bounced — the most-reported "preview ≠ export" bug.
      const t = Math.min(1, elapsed / 0.6);
      // 3 oscillations, decaying amplitude
      const amp = 18 * (1 - t);
      animOffsetY = -Math.abs(Math.sin(t * Math.PI * 3)) * amp;
      // Subtle scale pulse so the bounce reads on still text
      animScale = 1 + Math.sin(t * Math.PI * 3) * 0.05 * (1 - t);
    }

    ctx.globalAlpha *= animAlpha;

    // Font
    ctx.font = `${fontWeight} ${fontSize}px "${fontFamily}", sans-serif`;
    ctx.textAlign = align;
    ctx.textBaseline = 'middle';

    const x = (pos.x / 100) * width;
    const y = (pos.y / 100) * height + animOffsetY;

    // Apply rotation and pop scale around center
    ctx.translate(x, y);
    if (rotation !== 0) ctx.rotate((rotation * Math.PI) / 180);
    if (animScale !== 1) ctx.scale(animScale, animScale);
    // Now draw relative to origin (which is the text center)
    const drawX = 0;
    const drawY = 0;

    // Adjust textAlign anchor for centered drawing
    const textAlignX = align === 'center' ? 0 : align === 'right' ? 0 : 0;

    // Background box
    if (style.bgColor && style.bgOpacity > 0) {
      const metrics = ctx.measureText(text);
      const pad = style.bgPadding || 8;
      const boxW = metrics.width + pad * 2;
      const boxH = fontSize * 1.4 + pad * 2;
      ctx.fillStyle = this._hexToRgba(style.bgColor, (style.bgOpacity || 75) / 100);
      const bx = align === 'center' ? -boxW / 2 : align === 'right' ? -boxW : 0;
      ctx.beginPath();
      ctx.roundRect(bx, -boxH / 2, boxW, boxH, style.bgRadius || 4);
      ctx.fill();
    }

    // Text outline/stroke
    if (style.outlineWidth > 0) {
      ctx.strokeStyle = style.outlineColor || '#000000';
      ctx.lineWidth = style.outlineWidth * 2;
      ctx.lineJoin = 'round';
      ctx.strokeText(text, drawX, drawY);
    }

    // Text shadow
    if (style.shadowBlur > 0 || style.shadowOffsetX || style.shadowOffsetY) {
      ctx.shadowColor = style.shadowColor || 'rgba(0,0,0,0.5)';
      ctx.shadowBlur = style.shadowBlur || 2;
      ctx.shadowOffsetX = style.shadowOffsetX || 1;
      ctx.shadowOffsetY = style.shadowOffsetY || 1;
    }

    // Fill text
    ctx.fillStyle = color;

    // Use item's width for text wrapping (matches DOM rendering)
    const maxWidth = (size.w / 100) * width;

    if (anim === 'typewriter') {
      const charsToShow = Math.floor((elapsed / clipDur) * text.length);
      ctx.fillText(text.slice(0, charsToShow), drawX, drawY);
    } else {
      this._drawWrappedText(ctx, text, drawX, drawY, maxWidth, fontSize * 1.4);
    }

    ctx.restore();
  }

  _renderSubtitle(ctx, clip, currentTime, settings) {
    const { width, height } = this;
    const text = clip.subtitleText || '';
    if (!text) return;

    const subSettings = settings.subtitle || settings || {};
    const fontName = subSettings.subtitleFont || 'DM Sans';
    const sizeLabel = subSettings.subtitleSize || 'medium';
    const basePx = typeof sizeLabel === 'number' ? sizeLabel : (FONT_SIZE_MAP[sizeLabel] || 30);
    const fontScale = Math.min(width, height) / Math.min(REF_W, REF_H);
    const fontSize = Math.max(16, Math.round(basePx * fontScale));
    const fontWeight = typeof subSettings.subtitleFontWeight === 'number' ? subSettings.subtitleFontWeight : subSettings.subtitleFontWeight === 'bold' ? 700 : subSettings.subtitleFontWeight === 'black' ? 900 : 400;
    const settingsPosition = subSettings.subtitlePosition || 'bottom';
    const maxWidthPct = subSettings.subtitleMaxWidth ?? 90;
    const offsetVPct = subSettings.subtitleOffsetV ?? 4;
    const bgEnabled = subSettings.subtitleBgEnabled || false;
    const bgColor = subSettings.subtitleBgColor || '#000000';
    const bgOpacity = subSettings.subtitleBgOpacity ?? 75;
    const outlineColor = subSettings.subtitleOutlineColor || '#000000';
    const outlineOpacity = (subSettings.subtitleOutlineOpacity ?? 100) / 100;
    const outlineWidth = Math.max(0, Math.round((subSettings.subtitleOutlineWidth ?? 2) * fontScale));
    const activeWordEnabled = subSettings.activeWordEnabled || false;
    const activeWordColor = subSettings.activeWordColor || '#FFD700';
    const activeWordOutlineColor = subSettings.activeWordOutlineColor || '#000000';
    const activeWordBgColor = subSettings.activeWordBgColor || '#000000';
    const activeWordBgOpacity = subSettings.activeWordBgOpacity ?? 0;

    // Resolve font color using speaker colors (matches SubtitleOverlay exactly)
    const speakersOrdered = this._speakersOrdered || [];
    const fontColor = getSpeakerColor(clip.speaker, speakersOrdered, subSettings);

    // Per-item position and rotation from timeline store
    const itemPos = clip.position || { x: 50, y: 90 };
    const itemRotation = (clip.transform?.rotation) ?? 0;
    const hasCustomPosition = itemPos.x !== 50 || itemPos.y !== 90;

    ctx.save();

    // Position: use per-item position if set, otherwise use settings-based position
    const maxWidth = (maxWidthPct / 100) * width;
    let x, y;
    if (hasCustomPosition) {
      x = (itemPos.x / 100) * width;
      y = (itemPos.y / 100) * height;
    } else {
      x = width / 2;
      if (settingsPosition === 'top') {
        y = (offsetVPct / 100) * height + fontSize;
      } else if (settingsPosition === 'center') {
        y = height / 2;
      } else {
        y = height - (offsetVPct / 100) * height - fontSize * 0.4;
      }
    }

    // Apply rotation around the subtitle's position
    if (itemRotation !== 0) {
      ctx.translate(x, y);
      ctx.rotate((itemRotation * Math.PI) / 180);
      ctx.translate(-x, -y);
    }

    ctx.font = `${fontWeight} ${fontSize}px "${fontName}", sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';

    // Measure for background
    const lines = this._wrapText(ctx, text, maxWidth);
    const lineHeight = fontSize * 1.4;
    const totalHeight = lines.length * lineHeight;

    // Background box
    if (bgEnabled) {
      let maxLineW = 0;
      for (const line of lines) {
        maxLineW = Math.max(maxLineW, ctx.measureText(line).width);
      }
      const pad = Math.max(4, Math.round(8 * fontScale));
      ctx.fillStyle = this._hexToRgba(bgColor, bgOpacity / 100);
      ctx.beginPath();
      ctx.roundRect(
        x - maxLineW / 2 - pad,
        y - totalHeight / 2 - pad,
        maxLineW + pad * 2,
        totalHeight + pad * 2,
        4
      );
      ctx.fill();
    }

    // Compute active word index internally (same algorithm as SubtitleOverlay)
    // This ensures 1:1 parity — no reliance on external _activeWordIndex property.
    let globalActiveWordIdx = -1;
    if (activeWordEnabled) {
      const relTime = currentTime; // currentTime is already clip-relative
      globalActiveWordIdx = computeActiveWordIndex(clip, relTime, this._speakerRates);
    }

    // Draw each line with proper per-line word index tracking
    const startY = y - (totalHeight - lineHeight) / 2;
    // Build a flat word list to map global word index to per-line positions
    const allWords = text.split(/\s+/).filter(Boolean);
    let globalWordOffset = 0;

    for (let i = 0; i < lines.length; i++) {
      const lineY = startY + i * lineHeight;
      const lineWords = lines[i].split(/\s+/).filter(Boolean);

      // Outline for the full line
      if (outlineWidth > 0 && !bgEnabled) {
        ctx.strokeStyle = this._hexToRgba(outlineColor, outlineOpacity);
        ctx.lineWidth = outlineWidth * 2;
        ctx.lineJoin = 'round';
        ctx.strokeText(lines[i], x, lineY);
      }

      // Active word highlighting with correct per-line index tracking
      if (activeWordEnabled && globalActiveWordIdx >= 0) {
        const lineWidth = ctx.measureText(lines[i]).width;
        let wordX = x - lineWidth / 2;
        ctx.textAlign = 'left';

        for (let w = 0; w < lineWords.length; w++) {
          const word = lineWords[w];
          const isLastWord = w === lineWords.length - 1;
          const wordWithSpace = isLastWord ? word : word + ' ';
          const ww = ctx.measureText(wordWithSpace).width;
          const globalIdx = globalWordOffset + w;
          const isActive = globalIdx === globalActiveWordIdx;

          // Active word outline (matches SubtitleOverlay's per-word outline)
          if (isActive && outlineWidth > 0 && !bgEnabled) {
            const awOlColor = this._hexToRgba(activeWordOutlineColor, outlineOpacity);
            ctx.strokeStyle = awOlColor;
            ctx.lineWidth = outlineWidth * 2;
            ctx.lineJoin = 'round';
            ctx.strokeText(word, wordX, lineY);
          }

          // Active word background highlight — rounded corners match
          // the preview's CSS ``border-radius`` so the box shape is
          // consistent across DOM preview and canvas export. Falls
          // back to a sharp fillRect when ``roundRect`` isn't
          // available (Safari < 16.4 polyfill in Timeline already
          // exists; here we just guard).
          if (isActive && activeWordBgOpacity > 0) {
            const wordWidth = ctx.measureText(word).width;
            const padX = Math.max(1, 2 * fontScale);
            const bgX = wordX - padX;
            const bgY = lineY - fontSize * 0.6;
            const bgW = wordWidth + padX * 2;
            const bgH = fontSize * 1.2;
            const rawR = (clip.activeWordBgRadius != null
              ? clip.activeWordBgRadius
              : (clip.subtitleSettings?.activeWordBgRadius ?? 4));
            const r = Math.max(
              0,
              Math.min(rawR * fontScale, Math.min(bgW, bgH) / 2),
            );
            ctx.fillStyle = this._hexToRgba(activeWordBgColor, activeWordBgOpacity / 100);
            if (r > 0 && typeof ctx.roundRect === 'function') {
              ctx.beginPath();
              ctx.roundRect(bgX, bgY, bgW, bgH, r);
              ctx.fill();
            } else {
              ctx.fillRect(bgX, bgY, bgW, bgH);
            }
          }

          ctx.fillStyle = isActive ? activeWordColor : fontColor;
          ctx.fillText(word, wordX, lineY);
          wordX += ww;
        }
        ctx.textAlign = 'center';
      } else {
        ctx.fillStyle = fontColor;
        ctx.fillText(lines[i], x, lineY);
      }

      globalWordOffset += lineWords.length;
    }

    ctx.restore();
  }

  _renderShape(ctx, clip) {
    const { width, height } = this;
    // pos is center-based (DOM uses left/top + translate(-50%, -50%))
    const pos = clip.position || { x: 10, y: 10 };
    const size = clip.size || { w: 20, h: 20 };
    const style = clip.shapeStyle || {};
    const rotation = (clip.transform?.rotation) ?? 0;

    const cx = (pos.x / 100) * width;
    const cy = (pos.y / 100) * height;
    const dw = (size.w / 100) * width;
    const dh = (size.h / 100) * height;

    // Shape properties (strokeWidth, cornerRadius) are authored in CSS pixels
    // at the DOM preview resolution (~50vh ≈ 540px height on a 1080p display).
    // Scale them to the current canvas resolution so exports match the preview.
    const SHAPE_PROP_REF_H = 540;
    const shapeScale = height / SHAPE_PROP_REF_H;

    ctx.save();
    // Translate to center, apply rotation, then draw relative to center
    ctx.translate(cx, cy);
    if (rotation !== 0) ctx.rotate((rotation * Math.PI) / 180);

    ctx.fillStyle = style.fillColor || '#FF3B30';
    ctx.strokeStyle = style.strokeColor || '#FFFFFF';
    ctx.lineWidth = (style.strokeWidth || 2) * shapeScale;

    const shapeType = clip.shapeType || 'rectangle';

    switch (shapeType) {
      case 'rectangle':
        ctx.beginPath();
        ctx.roundRect(-dw / 2, -dh / 2, dw, dh, (style.cornerRadius || 0) * shapeScale);
        if (style.fillColor) ctx.fill();
        if (style.strokeWidth > 0) ctx.stroke();
        break;
      case 'circle':
      case 'ellipse':
        ctx.beginPath();
        ctx.ellipse(0, 0, dw / 2, dh / 2, 0, 0, Math.PI * 2);
        if (style.fillColor) ctx.fill();
        if (style.strokeWidth > 0) ctx.stroke();
        break;
      case 'arrow':
        ctx.beginPath();
        ctx.moveTo(-dw / 2, 0);
        ctx.lineTo(dw * 0.2, 0);
        ctx.lineTo(dw * 0.2, -dh / 2);
        ctx.lineTo(dw / 2, 0);
        ctx.lineTo(dw * 0.2, dh / 2);
        ctx.lineTo(dw * 0.2, 0);
        ctx.closePath();
        if (style.fillColor) ctx.fill();
        if (style.strokeWidth > 0) ctx.stroke();
        break;
      case 'line':
        ctx.beginPath();
        ctx.moveTo(-dw / 2, -dh / 2);
        ctx.lineTo(dw / 2, dh / 2);
        ctx.stroke();
        break;
    }

    ctx.restore();
  }

  _renderTransition(ctx, incomingClip, track, currentTime, progress, settings, mediaElements, allClips) {
    // Find the outgoing clip (the clip that ends where this one starts)
    const outgoingClip = allClips.find(c =>
      c.trackId === track.id &&
      c.id !== incomingClip.id &&
      Math.abs(c.end - incomingClip.start) < 0.05
    );

    if (!outgoingClip) {
      // No outgoing clip — just render incoming with fade
      ctx.globalAlpha = progress;
      this._renderClipDirect(ctx, incomingClip, currentTime, settings, mediaElements);
      ctx.globalAlpha = 1;
      return;
    }

    // Get or create transition buffers
    if (!this._transitionBuffer1 || this._transitionBuffer1.width !== this.width) {
      this._transitionBuffer1 = new OffscreenCanvas(this.width, this.height);
      this._transitionBuffer2 = new OffscreenCanvas(this.width, this.height);
    }

    // Render outgoing to buffer 1
    const ctx1 = this._transitionBuffer1.getContext('2d', { willReadFrequently: false });
    ctx1.clearRect(0, 0, this.width, this.height);
    this._renderClipToCtx(ctx1, outgoingClip, currentTime, settings, mediaElements);

    // Render incoming to buffer 2
    const ctx2 = this._transitionBuffer2.getContext('2d', { willReadFrequently: false });
    ctx2.clearRect(0, 0, this.width, this.height);
    this._renderClipToCtx(ctx2, incomingClip, currentTime, settings, mediaElements);

    // Apply transition
    const transType = incomingClip.transition.type || 'dissolve';
    const renderer = TRANSITIONS[transType] || TRANSITIONS.dissolve;
    renderer(ctx, this._transitionBuffer1, this._transitionBuffer2, progress, this.width, this.height);
  }

  _renderClipDirect(ctx, clip, currentTime, settings, mediaElements) {
    const filterStr = this._buildFilterString(clip.effects);
    if (filterStr) ctx.filter = filterStr;
    if (clip.type === 'video') this._renderVideo(ctx, clip, currentTime, settings, mediaElements);
    else if (clip.type === 'image' || clip.type === 'overlay') this._renderImage(ctx, clip, mediaElements);
    ctx.filter = 'none';
  }

  _renderClipToCtx(targetCtx, clip, currentTime, settings, mediaElements) {
    // Render through the same per-type renderers as the live preview
    // path so transitions don't drift from the regular composite.
    //
    // Previously this drew video clips as a full-frame ``drawImage``
    // ignoring crop / subject_x / layout mode / aspect-ratio
    // letterboxing, which made every transition snap to a different
    // composition than the surrounding frames.
    const filterStr = this._buildFilterString(clip.effects);
    if (filterStr) targetCtx.filter = filterStr;
    try {
      if (clip.type === 'video') {
        this._renderVideo(targetCtx, clip, currentTime, settings, mediaElements);
      } else if (clip.type === 'image' || clip.type === 'overlay') {
        this._renderImage(targetCtx, clip, mediaElements);
      } else if (clip.type === 'text') {
        this._renderText(targetCtx, clip, currentTime);
      } else if (clip.type === 'shape') {
        this._renderShape(targetCtx, clip);
      }
    } finally {
      targetCtx.filter = 'none';
    }
  }

  _buildFilterString(effects) {
    if (!effects || typeof effects !== 'object' || Array.isArray(effects)) return null;
    const parts = [];
    if (effects.brightness != null && effects.brightness !== 0) {
      parts.push(`brightness(${1 + effects.brightness / 100})`);
    }
    if (effects.contrast != null && effects.contrast !== 0) {
      parts.push(`contrast(${1 + effects.contrast / 100})`);
    }
    if (effects.saturation != null && effects.saturation !== 0) {
      parts.push(`saturate(${1 + effects.saturation / 100})`);
    }
    if (effects.blur > 0) {
      parts.push(`blur(${effects.blur}px)`);
    }
    if (effects.hueRotate > 0) {
      parts.push(`hue-rotate(${effects.hueRotate}deg)`);
    }
    if (effects.sepia > 0) {
      parts.push(`sepia(${effects.sepia / 100})`);
    }
    return parts.length > 0 ? parts.join(' ') : null;
  }

  _wrapText(ctx, text, maxWidth) {
    const words = text.split(/\s+/);
    const lines = [];
    let currentLine = '';

    for (const word of words) {
      const testLine = currentLine ? `${currentLine} ${word}` : word;
      if (ctx.measureText(testLine).width > maxWidth && currentLine) {
        lines.push(currentLine);
        currentLine = word;
      } else {
        currentLine = testLine;
      }
    }
    if (currentLine) lines.push(currentLine);
    return lines.length > 0 ? lines : [''];
  }

  _drawWrappedText(ctx, text, x, y, maxWidth, lineHeight) {
    const lines = this._wrapText(ctx, text, maxWidth);
    const startY = y - ((lines.length - 1) * lineHeight) / 2;
    for (let i = 0; i < lines.length; i++) {
      ctx.fillText(lines[i], x, startY + i * lineHeight);
    }
  }

  _hexToRgba(hex, alpha) {
    hex = (hex || '#000000').replace('#', '');
    if (hex.length === 3) hex = hex[0] + hex[0] + hex[1] + hex[1] + hex[2] + hex[2];
    const r = parseInt(hex.substring(0, 2), 16);
    const g = parseInt(hex.substring(2, 4), 16);
    const b = parseInt(hex.substring(4, 6), 16);
    return `rgba(${r},${g},${b},${alpha})`;
  }

  /**
   * Clean up resources
   */
  destroy() {
    this._transitionBuffer1 = null;
    this._transitionBuffer2 = null;
    this._fontCache.clear();
    this._speakerRates = null;
    this._speakersOrdered = null;
    this._subtitleSegments = null;
    this._subjectKeyframes = null;
    this._processKeyframes = null;
    this._interpolateSubjectX = null;
  }
}
