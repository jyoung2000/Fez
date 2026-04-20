import React, { useRef, useState, useEffect, useMemo, useCallback } from 'react';
import { processKeyframes, interpolateSubjectX, isDynamic, safeSubjectX, subjectXToCenterPct, computeLayoutAtTime, computeFaceYCenter, faceYToCenterPct } from '../utils/subjectTracking';
import { RenderPlanRenderer } from '../utils/renderPlanRenderer';
import ReframeDebugOverlay from './ReframeDebugOverlay';
import useTimelineStore from '../stores/timelineStore';
import { outlineTextShadow } from '../utils/textOutline';
import useResponsive from '../hooks/useResponsive';
import useVideoLoadRetry from '../hooks/useVideoLoadRetry';
import { SPEED_OPTIONS as SHARED_SPEED_OPTIONS } from '../utils/defaultSettings';

// --- Constants replicated from backend ---
// clip_exporter.py:12-17
const ASPECT_RATIO_DIMS = {
  '16:9': [1920, 1080],
  '9:16': [1080, 1920],
  '1:1': [1080, 1080],
  '4:5': [1080, 1350],
};

// ── Builtin font URL map (mirrors ClipSettingsPanel) ─────────────────────
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
  'Liberation Serif': '/api/fonts/builtin/LiberationSerif-Regular.ttf',
  'Liberation Mono': '/api/fonts/builtin/LiberationMono-Regular.ttf',
  'DejaVu Sans': '/api/fonts/builtin/DejaVuSans.ttf',
  'DejaVu Serif': '/api/fonts/builtin/DejaVuSerif.ttf',
  'DejaVu Sans Mono': '/api/fonts/builtin/DejaVuSansMono.ttf',
  'FreeSans': '/api/fonts/builtin/FreeSans.ttf',
};

function registerFontFace(fontName, url) {
  const existingId = `custom-font-${fontName.replace(/\s+/g, '-')}`;
  if (document.getElementById(existingId)) return;
  const style = document.createElement('style');
  style.id = existingId;
  style.textContent = `@font-face { font-family: '${fontName}'; src: url('${url}'); font-weight: 100 900; font-display: swap; }`;
  document.head.appendChild(style);
}
// clip_exporter.py:19-24
const ASPECT_RATIO_VALUES = {
  '16:9': 16 / 9,
  '9:16': 9 / 16,
  '1:1': 1.0,
  '4:5': 4 / 5,
};

// ass_generator.py:14-18
const FONT_SIZE_MAP = { small: 22, medium: 30, large: 40 };

// ass_generator.py:32-34 — reference resolution for scaling
const REF_W = 1920;
const REF_H = 1080;

// ass_generator.py:48-55
const DEFAULT_SPEAKER_PALETTE = [
  '#00D9FF', '#F59E0B', '#10B981', '#A78BFA', '#EF4444', '#EC4899',
  '#06B6D4', '#8B5CF6', '#F97316', '#14B8A6', '#E879F9', '#84CC16',
  '#FB7185', '#38BDF8', '#FBBF24', '#34D399', '#C084FC', '#F472B6',
  '#22D3EE', '#A3E635', '#FB923C', '#2DD4BF', '#818CF8', '#F87171',
];

// --- Backend logic replicated in JS ---

// Determine the aspect ratio rendering mode — matches _build_filter_chain logic
// Always crops to fill (no blur-background mode).
function getFrameMode(aspectRatio, sourceWidth, sourceHeight) {
  if (!aspectRatio || !ASPECT_RATIO_VALUES[aspectRatio]) {
    return { mode: 'original', targetRatio: sourceWidth / sourceHeight };
  }
  const targetRatio = ASPECT_RATIO_VALUES[aspectRatio];
  const srcRatio = sourceWidth / sourceHeight;
  if (Math.abs(srcRatio - targetRatio) < 0.01) {
    return { mode: 'original', targetRatio };
  }
  return { mode: 'crop', targetRatio };
}

// subjectXToCenterPct is imported from subjectTracking.js (shared with VideoEditor)

function formatTime(seconds) {
  if (!seconds || isNaN(seconds)) return '0:00';
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, '0')}`;
}

// Compute the speaker color for a segment
function getSpeakerColor(speaker, speakersOrdered, subtitleSettings) {
  const useSpeaker = subtitleSettings?.useSpeakerColors ?? true;
  if (!useSpeaker) return subtitleSettings?.subtitleFontColor || '#FFFFFF';

  const speakerColors = subtitleSettings?.speakerColors || {};
  if (speakerColors[speaker]) return speakerColors[speaker];

  const idx = speakersOrdered.indexOf(speaker);
  return DEFAULT_SPEAKER_PALETTE[(idx >= 0 ? idx : 0) % DEFAULT_SPEAKER_PALETTE.length];
}

// Parse hex color to rgba
function hexToRgba(hex, opacity) {
  hex = (hex || '#000000').replace('#', '');
  if (hex.length === 3) hex = hex[0] + hex[0] + hex[1] + hex[1] + hex[2] + hex[2];
  const r = parseInt(hex.substring(0, 2), 16);
  const g = parseInt(hex.substring(2, 4), 16);
  const b = parseInt(hex.substring(4, 6), 16);
  return `rgba(${r},${g},${b},${opacity})`;
}

// ── Active word timing ───────────────────────────────────────────────
// Dynamically times each word within a segment to match the speaker's
// natural cadence.  Uses:
//  1. Per-speaker speech rate (words/sec) computed from all their segments
//  2. Punctuation-aware pauses (commas, periods, etc.)
//  3. Character-proportional duration with natural speech weighting
//  4. Anticipation offset so highlight leads audio for perceptual sync
const _BASE_OVERHEAD_S = 0.04;   // minimum gap between words
const _ANTICIPATION_S  = 0.10;   // perceptual lead — highlight leads audio
const _AUDIO_BUFFER_S  = 0.12;   // compensate for browser audio output lag

// Extra pause added AFTER a word that ends with punctuation
const _PUNCT_PAUSE = { ',': 0.15, ';': 0.16, ':': 0.12, '.': 0.22, '!': 0.22, '?': 0.24, '\u2014': 0.12, '\u2013': 0.10 };

// Function words are spoken ~25% faster in natural speech
const _FAST_WORDS = new Set([
  'the', 'a', 'an', 'to', 'in', 'on', 'at', 'of', 'for',
  'and', 'but', 'or', 'is', 'was', 'are', 'were', 'it',
  'its', 'this', 'that',
]);

// Compute per-speaker words-per-second from the clip's transcript segments.
// Returns a Map<speaker, wps>.  Called once per clip, not per frame.
function computeSpeakerRates(clipSegments) {
  const stats = {};
  for (const seg of clipSegments) {
    const wc = (seg.text || '').split(/\s+/).filter(Boolean).length;
    const dur = seg.end - seg.start;
    if (dur <= 0 || wc === 0) continue;
    if (!stats[seg.speaker]) stats[seg.speaker] = { words: 0, time: 0 };
    stats[seg.speaker].words += wc;
    stats[seg.speaker].time += dur;
  }
  const rates = {};
  for (const [sp, s] of Object.entries(stats)) {
    rates[sp] = s.time > 0 ? s.words / s.time : 3.0; // default ~3 wps
  }
  return rates;
}

function getCurrentWordIndex(segment, relativeTime, speakerRates) {
  if (!segment || !segment.text) return -1;
  const words = segment.text.split(/\s+/).filter(Boolean);
  if (words.length <= 1) return words.length === 1 ? 0 : -1;

  // Use real per-word timestamps from Whisper when available and word count
  // matches the display text (user may have edited the transcript text).
  // Words are already in clip-relative time (offset during clipSegments construction).
  // Apply the same audio buffer compensation as the fallback path — browsers
  // report video.currentTime ~120ms before the user actually hears the audio
  // (output pipeline latency).  Combined with the 100ms anticipation lead, the
  // net perceptual effect is: highlight appears ~100ms before the word is heard,
  // which feels like natural "reading ahead" sync.
  if (segment.words && segment.words.length === words.length) {
    const anticipation = 0.10; // 100ms lead for perceptual sync
    const adjusted = relativeTime + anticipation - _AUDIO_BUFFER_S;
    if (adjusted < segment.words[0].start) return -1;
    for (let i = 0; i < segment.words.length; i++) {
      if (adjusted < segment.words[i].end) return i;
    }
    return segment.words.length - 1;
  }

  // Fallback: character-proportional estimation for segments without word data
  const totalChars = words.reduce((sum, w) => sum + w.length, 0);
  if (totalChars === 0) return -1;
  const segDuration = segment.end - segment.start;

  // Per-speaker speech rate scaling: faster speakers → less overhead
  const speakerWps = (speakerRates && speakerRates[segment.speaker]) || 3.0;
  const rateScale = Math.max(0.6, Math.min(1.6, 3.0 / speakerWps));

  // Anticipation scales with speech rate; audio buffer is constant browser latency
  const anticipation = _ANTICIPATION_S * rateScale;
  const elapsed = (relativeTime - segment.start) + anticipation - _AUDIO_BUFFER_S;
  if (elapsed < 0) return -1;

  // Compute punctuation pauses for each word
  const punctPauses = words.map((w) => {
    const last = w[w.length - 1];
    return (_PUNCT_PAUSE[last] || 0) * rateScale;
  });
  const totalPunct = punctPauses.reduce((a, b) => a + b, 0);

  // Base overhead budget (gaps between words)
  const baseOverhead = _BASE_OVERHEAD_S * rateScale * words.length;
  const totalPause = baseOverhead + totalPunct;

  // Remaining time is distributed proportionally by character count
  const charTime = Math.max(segDuration - totalPause, segDuration * 0.45);
  const pauseScale = (segDuration - charTime) / Math.max(totalPause, 0.01);

  let t = 0;
  for (let i = 0; i < words.length; i++) {
    const charDur = charTime * (words[i].length / totalChars);
    const pause = (_BASE_OVERHEAD_S * rateScale + punctPauses[i]) * pauseScale;
    let wordDur = charDur + pause;
    // Function words are spoken faster
    const stripped = words[i].toLowerCase().replace(/[.,!?;:\u2014\u2013]+$/, '');
    if (_FAST_WORDS.has(stripped)) wordDur *= 0.75;
    // First word emphasis (slightly longer hold)
    if (i === 0) wordDur *= 1.15;
    // Last word trailing emphasis
    else if (i === words.length - 1) wordDur *= 1.10;
    if (elapsed < t + wordDur) return i;
    t += wordDur;
  }
  return words.length - 1;
}

/**
 * Split subtitle segments so no segment exceeds maxWords.
 * Time is distributed proportionally by word count.
 */
function splitSegmentsByMaxWords(segments, maxWords) {
  if (!maxWords || maxWords <= 0) return segments;
  const result = [];
  for (const seg of segments) {
    const words = seg.text.split(/\s+/).filter(Boolean);
    if (words.length <= maxWords) {
      result.push(seg);
      continue;
    }
    const totalWords = words.length;
    const duration = seg.end - seg.start;
    let currentTime = seg.start;
    for (let i = 0; i < totalWords; i += maxWords) {
      const chunkWords = words.slice(i, i + maxWords);
      const chunkDuration = duration * (chunkWords.length / totalWords);
      let chunkEnd = currentTime + chunkDuration;
      if (i + maxWords >= totalWords) chunkEnd = seg.end;
      if (chunkEnd - currentTime >= 0.1) {
        result.push({ start: currentTime, end: chunkEnd, text: chunkWords.join(' '), speaker: seg.speaker });
      }
      currentTime = chunkEnd;
    }
  }
  return result;
}

export default function ClipPreview({
  src,
  clipStart = 0,
  clipEnd = 0,
  aspectRatio,
  sourceWidth = 1920,
  sourceHeight = 1080,
  subjectX = 50,
  scenes,
  sceneCuts = null,
  subtitlesEnabled = false,
  subtitleSettings,
  transcript = [],
  onClose,
  title,
  inline = false,
  initialVolume,
  initialSpeed,
  layoutTimeline = null,
  faceRegistry = null,
  defaultLayoutMode = 'single',
  trackingMode = null,
  jobId = null,
  clipIndex = null,
}) {
  const { isMobile } = useResponsive();
  const fgVideoRef = useRef(null);
  const splitBottomVideoRef = useRef(null);
  const containerRef = useRef(null);
  const fullscreenRef = useRef(null);
  const canvasRef = useRef(null);
  const renderPlanRendererRef = useRef(null);
  const [renderPlanData, setRenderPlanData] = useState(null);

  // ── RenderPlan fetch: get the plan from the backend when jobId is available ──
  useEffect(() => {
    if (!jobId || !aspectRatio) return;
    let cancelled = false;
    const mode = clipIndex != null ? 'clip' : 'full';
    const params = new URLSearchParams({ mode, aspect_ratio: aspectRatio });
    if (clipIndex != null) params.set('clip_index', String(clipIndex));

    fetch(`/api/jobs/${jobId}/render_plan?${params}`)
      .then(r => r.ok ? r.json() : null)
      .then(plan => {
        if (!cancelled && plan && plan.ops?.length > 0) {
          setRenderPlanData(plan);
          console.log(`[RenderPlan] Loaded plan: ${plan.ops.length} ops, ${plan.total_duration_sec?.toFixed(1)}s`);
        }
      })
      .catch(err => {
        console.log('[RenderPlan] Not available, using legacy preview:', err.message);
      });
    return () => { cancelled = true; };
  }, [jobId, clipIndex, aspectRatio]);

  // ── RenderPlan Canvas rendering via rAF ──
  useEffect(() => {
    if (!renderPlanData) return;
    const canvas = canvasRef.current;
    const video = fgVideoRef.current;
    if (!canvas || !video) return;

    const renderer = new RenderPlanRenderer(canvas, video, renderPlanData);
    renderPlanRendererRef.current = renderer;

    let animId;
    const tick = () => {
      const relTime = video.currentTime - clipStart;
      renderer.draw(Math.max(0, relTime));
      animId = requestAnimationFrame(tick);
    };
    animId = requestAnimationFrame(tick);

    return () => {
      cancelAnimationFrame(animId);
      renderer.dispose();
      renderPlanRendererRef.current = null;
    };
  }, [renderPlanData, clipStart]);

  // Dynamic subject tracking keyframes
  const subjectKeyframes = useMemo(
    () => {
      if (!scenes?.length) {
        console.log('[SubjectTracking] ClipPreview: no scenes available — using static subject_x');
        return null;
      }
      // Compute aspect ratios for dynamic safe margin in pipeline
      const _srcRatio = sourceWidth / sourceHeight;
      const _targetRatio = (aspectRatio && ASPECT_RATIO_VALUES[aspectRatio]) ? ASPECT_RATIO_VALUES[aspectRatio] : _srcRatio;
      const _isCrop = Math.abs(_srcRatio - _targetRatio) > 0.01;
      const processed = processKeyframes(scenes, clipStart, clipEnd, _isCrop ? _srcRatio : null, _isCrop ? _targetRatio : null, transcript || null, sceneCuts || null, trackingMode);
      const dynamic = processed && isDynamic(processed);
      console.log(
        `[SubjectTracking] ClipPreview: ${processed?.length || 0} keyframes (pipeline: build→compress→deadzone→cuts→smooth→holds) ` +
        `(${clipStart.toFixed(1)}s-${clipEnd.toFixed(1)}s), ` +
        `mode=${trackingMode === 'gameplay' ? 'GAMEPLAY' : (dynamic ? 'DYNAMIC' : 'STATIC')}, ` +
        `sx range: [${Math.min(...(processed || []).map(k=>k.x))}-${Math.max(...(processed || []).map(k=>k.x))}], ` +
        `keyframes: ${JSON.stringify(processed?.map(k => ({t: +k.t.toFixed(2), x: k.x})))}`
      );
      return processed;
    },
    [scenes, clipStart, clipEnd, aspectRatio, sourceWidth, sourceHeight, transcript, sceneCuts, trackingMode],
  );
  const hasDynamicSubject = useMemo(
    () => subjectKeyframes && isDynamic(subjectKeyframes),
    [subjectKeyframes],
  );

  // Compute face Y center from scene data for vertical positioning
  // Matches backend _compute_face_y_offset() for preview-export parity
  const faceYCenter = useMemo(() => {
    if (!scenes?.length) return 50;
    const y = computeFaceYCenter(scenes, clipStart, clipEnd);
    console.log(`[SubjectTracking] faceYCenter=${y.toFixed(1)}% (from scenes ${clipStart.toFixed(1)}s-${clipEnd.toFixed(1)}s)`);
    return y;
  }, [scenes, clipStart, clipEnd]);

  // Compute Y objectPosition percentage (for vertical crop offset)
  const yPositionPct = useMemo(() => {
    const _srcRatio = sourceWidth / sourceHeight;
    const _targetRatio = (aspectRatio && ASPECT_RATIO_VALUES[aspectRatio]) ? ASPECT_RATIO_VALUES[aspectRatio] : _srcRatio;
    const pct = faceYToCenterPct(faceYCenter, _srcRatio, _targetRatio);
    if (pct !== 50) {
      console.log(`[SubjectTracking] yPositionPct=${pct.toFixed(1)}% (faceY=${faceYCenter.toFixed(1)}%, R_v=${(_targetRatio/_srcRatio).toFixed(2)})`);
    }
    return pct;
  }, [faceYCenter, sourceWidth, sourceHeight, aspectRatio]);

  // Frame mode + crop flag must be computed BEFORE trackingStatus's
  // useMemo factory references ``isCrop``. Previously these were defined
  // ~50 lines below, causing a temporal-dead-zone ReferenceError the
  // first time the memo body actually evaluated.
  const frameMode = useMemo(
    () => getFrameMode(aspectRatio, sourceWidth, sourceHeight),
    [aspectRatio, sourceWidth, sourceHeight],
  );
  const isCrop = frameMode.mode === 'crop';
  const targetRatio = frameMode.targetRatio;

  // Tracking status for user feedback
  const trackingStatus = useMemo(() => {
    if (!isCrop) return null;
    if (!scenes?.length) return { mode: 'no-data', label: 'No AI data', color: '#f59e0b' };
    if (!subjectKeyframes?.length) return { mode: 'error', label: 'Tracking failed', color: '#ef4444' };
    const first = subjectKeyframes[0];
    const sx = first.x;
    const conf = first.confidence;
    const strategy = first.strategy || 'stationary';
    const strategyChip = strategy === 'stationary' ? 'CROP'
      : strategy === 'split_screen' ? 'SPLIT'
      : strategy === 'blur_fill' ? 'BLUR'
      : strategy === 'wide_master' ? 'WIDE'
      : strategy === 'grid' ? 'GRID'
      : strategy === 'tracking' ? 'TRACK'
      : strategy.toUpperCase();
    const confLabel = conf != null ? ` \u00b7 Conf: ${Math.round(conf * 100)}%` : '';

    if (hasDynamicSubject) {
      return {
        mode: 'dynamic',
        label: `Subject x: ${sx}%${confLabel} \u00b7 ${strategyChip}`,
        color: '#10b981',
      };
    }
    if (Math.abs(sx - 50) < 3 && strategy === 'stationary') {
      return { mode: 'center', label: `Centered${confLabel} \u00b7 ${strategyChip}`, color: '#6b7280' };
    }
    return { mode: 'static', label: `Subject x: ${sx}%${confLabel} \u00b7 ${strategyChip}`, color: '#10b981' };
  }, [isCrop, scenes, subjectKeyframes, hasDynamicSubject]);

  // Sub-set of the centralized SPEED_OPTIONS — preview chrome only
  // exposes the most common speeds. Full range (0.25-4×) is reachable
  // from the Properties panel.
  const SPEED_OPTIONS = SHARED_SPEED_OPTIONS.filter((s) => s >= 0.5 && s <= 2.0);

  const [playing, setPlaying] = useState(false);
  const [videoReady, setVideoReady] = useState(false);
  // Use a ref for the raw video time to avoid re-rendering on every timeupdate.
  // Only the display time (throttled) triggers re-renders.
  const currentTimeRef = useRef(clipStart);
  const [displayTime, setDisplayTime] = useState(clipStart);
  const [volume, setVolume] = useState(initialVolume != null ? initialVolume / 100 : 1);
  const [speed, setSpeed] = useState(initialSpeed != null && initialSpeed > 0 ? initialSpeed : 1.0);
  const [hovered, setHovered] = useState(false);
  const [containerSize, setContainerSize] = useState({ w: 0, h: 0 });
  const [currentSubtitle, setCurrentSubtitle] = useState(null);
  const [currentWordIdx, setCurrentWordIdx] = useState(-1);
  const [isFullscreen, setIsFullscreen] = useState(false);

  const clipDur = clipEnd - clipStart;
  const elapsed = Math.max(0, Math.min(clipDur, displayTime - clipStart));
  const progress = clipDur > 0 ? (elapsed / clipDur) * 100 : 0;
  // ``frameMode``, ``isCrop``, ``targetRatio`` are defined above so
  // the ``trackingStatus`` useMemo body can reference them safely.

  // Pre-filter transcript segments to clip range (ass_generator.py:176-183)
  const clipSegments = useMemo(() => {
    if (!subtitlesEnabled || !transcript?.length) return [];
    const filtered = transcript
      .filter((seg) => seg.end > clipStart && seg.start < clipEnd)
      .map((seg) => {
        const segStart = Math.max(seg.start, clipStart);
        const segEnd = Math.min(seg.end, clipEnd);
        // Carry through per-word timestamps, offset to clip-relative time
        let words = null;
        if (seg.words && seg.words.length > 0) {
          words = seg.words
            .filter((w) => w.end > segStart && w.start < segEnd)
            .map((w) => ({
              start: w.start - clipStart,
              end: w.end - clipStart,
              word: w.word,
            }));
          if (words.length === 0) words = null;
        }
        return {
          start: segStart - clipStart,
          end: segEnd - clipStart,
          text: (seg.text || '').trim(),
          speaker: seg.speaker || '',
          words,
        };
      })
      .filter((seg) => seg.end - seg.start >= 0.1);
    const maxWords = subtitleSettings?.subtitleMaxWords || 0;
    return maxWords > 0 ? splitSegmentsByMaxWords(filtered, maxWords) : filtered;
  }, [subtitlesEnabled, transcript, clipStart, clipEnd, subtitleSettings?.subtitleMaxWords]);

  // Ordered speakers list for color assignment (ass_generator.py:189-191)
  const speakersOrdered = useMemo(() => {
    const seen = [];
    for (const seg of clipSegments) {
      if (seg.speaker && !seen.includes(seg.speaker)) seen.push(seg.speaker);
    }
    return seen;
  }, [clipSegments]);

  // Per-speaker speech rate for active word timing accuracy
  const speakerRates = useMemo(
    () => computeSpeakerRates(clipSegments),
    [clipSegments],
  );

  // --- Escape key ---
  useEffect(() => {
    if (!onClose) return;
    const onKey = (e) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);

  // --- Track container size for font scaling ---
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      for (const entry of entries) {
        setContainerSize({ w: entry.contentRect.width, h: entry.contentRect.height });
      }
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // --- Seek to start and auto-play ---
  // Use a ref-based timeupdate handler to avoid re-rendering the entire
  // component ~4x/sec. The display time is updated via the rAF subtitle
  // loop (throttled to ~250ms) which batches the state update with
  // subtitle changes to minimize React re-renders.
  const displayTimeRafRef = useRef(0);
  useEffect(() => {
    const video = fgVideoRef.current;
    if (!video) return;

    const onMetadata = () => {
      video.currentTime = clipStart;
    };

    const onCanPlay = () => {
      setVideoReady(true);
      video.play().then(() => setPlaying(true)).catch(() => {});
    };

    const onTimeUpdate = () => {
      currentTimeRef.current = video.currentTime;
      if (clipEnd && video.currentTime >= clipEnd) {
        video.pause();
        setPlaying(false);
      }
    };

    // NOTE: the ``error`` event is intentionally NOT handled here.
    // useVideoLoadRetry (attached below) owns ``error`` for the
    // foreground video and retries with exponential backoff while
    // the backend browser-preview transcode is in flight (503 +
    // Retry-After). Attaching a second one-shot listener here would
    // race with that retry loop.
    video.addEventListener('loadedmetadata', onMetadata);
    video.addEventListener('canplay', onCanPlay);
    video.addEventListener('timeupdate', onTimeUpdate);
    if (video.readyState >= 3) {
      setVideoReady(true);
      onCanPlay();
    } else if (video.readyState >= 1) {
      onMetadata();
    }

    return () => {
      video.removeEventListener('loadedmetadata', onMetadata);
      video.removeEventListener('canplay', onCanPlay);
      video.removeEventListener('timeupdate', onTimeUpdate);
    };
  }, [clipStart, clipEnd]);

  // Retry loader for the 503 + Retry-After window while the backend
  // browser-preview transcode is in flight. Without this, a user who
  // opens a clip before transcode finishes (typical for large 4K / VP9
  // / MKV uploads) sees a dead <video>. The split-view bottom <video>
  // and the fullscreen single <video> share the same src so retrying
  // the foreground element re-drives the backend preview build for
  // both.
  const {
    preparing: fgPreparing,
    error: fgLoadError,
    reset: retryFgLoad,
  } = useVideoLoadRetry(fgVideoRef, src);

  // --- Sync volume from settings ---
  useEffect(() => {
    if (initialVolume == null) return;
    const v = Math.max(0, Math.min(1, initialVolume / 100));
    setVolume(v);
    if (fgVideoRef.current) fgVideoRef.current.volume = v;
  }, [initialVolume]);

  // --- Sync speed from settings ---
  useEffect(() => {
    if (initialSpeed == null || initialSpeed <= 0) return;
    setSpeed(initialSpeed);
    if (fgVideoRef.current) fgVideoRef.current.playbackRate = initialSpeed;
  }, [initialSpeed]);

  // --- Apply speed to video element ---
  useEffect(() => {
    if (fgVideoRef.current) fgVideoRef.current.playbackRate = speed;
  }, [speed]);

  // --- Register @font-face and preload the selected subtitle font so the
  //     browser downloads it before the subtitle text first renders.
  useEffect(() => {
    const font = subtitleSettings?.subtitleFont;
    if (!font || typeof document === 'undefined') return;

    // Register builtin font @font-face if known
    if (BUILTIN_FONT_FILES[font]) {
      registerFontFace(font, BUILTIN_FONT_FILES[font]);
    } else {
      // Custom font — look up URL from /api/fonts
      fetch('/api/fonts')
        .then((r) => r.ok ? r.json() : [])
        .then((fonts) => {
          const match = fonts.find((f) => f.name === font);
          if (match) {
            registerFontFace(match.name, match.url);
            document.fonts.load(`400 16px "${font}"`).catch(() => {});
            document.fonts.load(`700 16px "${font}"`).catch(() => {});
          }
        })
        .catch(() => {});
    }

    // Trigger download for both normal and bold weights
    document.fonts.load(`400 16px "${font}"`).catch(() => {});
    document.fonts.load(`700 16px "${font}"`).catch(() => {});
  }, [subtitleSettings?.subtitleFont]);

  // --- Subtitle sync + display time update via requestAnimationFrame ---
  // This rAF loop handles two things:
  // 1. Subtitle segment matching (only sets state when subtitle changes)
  // 2. Display time updates for the seek bar (throttled to ~250ms)
  // By batching both into one loop, we minimize React re-renders during playback.
  const activeWordEnabled = subtitleSettings?.activeWordEnabled || false;
  useEffect(() => {
    const video = fgVideoRef.current;
    if (!video) return;

    // If subtitles are disabled, still run the display time updater
    const hasSubtitles = subtitlesEnabled && clipSegments.length > 0;
    if (!hasSubtitles) {
      setCurrentSubtitle(null);
      setCurrentWordIdx(-1);
    }

    let animId;
    let prevSubRef = null;     // Track previous subtitle by reference
    let prevWordIdx = -1;
    let lastDisplayUpdate = 0; // Timestamp of last display time setState
    // Use binary-search-friendly index hint for segment lookup
    let lastSegIdx = 0;

    const tick = () => {
      const now = video.currentTime;
      currentTimeRef.current = now;
      const relTime = now - clipStart;

      // --- Subtitle matching (only when subtitles enabled) ---
      if (hasSubtitles) {
        // Optimized segment lookup: start from last known index and scan
        // forward/backward (segments are sorted by time). Falls back to
        // linear scan if the hint is stale.
        let active = null;
        const segs = clipSegments;
        const len = segs.length;
        // Check last known segment first (common case: same segment)
        if (lastSegIdx < len && segs[lastSegIdx].start <= relTime && relTime < segs[lastSegIdx].end) {
          active = segs[lastSegIdx];
        } else {
          // Scan forward from hint
          for (let i = lastSegIdx + 1; i < len; i++) {
            if (segs[i].start <= relTime && relTime < segs[i].end) {
              active = segs[i];
              lastSegIdx = i;
              break;
            }
            if (segs[i].start > relTime) break; // Past current time
          }
          // If not found forward, scan backward
          if (!active) {
            for (let i = Math.min(lastSegIdx, len - 1); i >= 0; i--) {
              if (segs[i].start <= relTime && relTime < segs[i].end) {
                active = segs[i];
                lastSegIdx = i;
                break;
              }
              if (segs[i].end <= relTime) break; // Before current time
            }
          }
        }

        // Only update React state when the active subtitle actually changes
        if (active !== prevSubRef) {
          prevSubRef = active;
          setCurrentSubtitle(active || null);
        }

        if (active && activeWordEnabled) {
          const idx = getCurrentWordIndex(active, relTime, speakerRates);
          if (idx !== prevWordIdx) {
            prevWordIdx = idx;
            setCurrentWordIdx(idx);
          }
        } else if (prevWordIdx !== -1) {
          prevWordIdx = -1;
          setCurrentWordIdx(-1);
        }
      }

      // --- Throttled display time update for seek bar (~4 updates/sec) ---
      const nowMs = performance.now();
      if (nowMs - lastDisplayUpdate > 250) {
        lastDisplayUpdate = nowMs;
        setDisplayTime(now);
      }

      animId = requestAnimationFrame(tick);
    };
    animId = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(animId);
  }, [subtitlesEnabled, clipSegments, clipStart, activeWordEnabled, speakerRates]);

  // --- Dynamic subject tracking: update objectPosition via rAF for instant ~60fps snaps ---
  const srcRatio = sourceWidth / sourceHeight;
  const lastAppliedPctRef = useRef(null);
  // Detect reframe-segment mode: keyframes from ReframeSegmenter have easeMs field
  const isReframeSegmentMode = useMemo(
    () => subjectKeyframes?.some(k => k.easeMs !== undefined),
    [subjectKeyframes],
  );
  useEffect(() => {
    if (!hasDynamicSubject) return;
    const video = fgVideoRef.current;
    if (!video) return;
    const R = srcRatio / targetRatio;
    let logCount = 0;
    let lastPct = null;
    console.log(
      `[SubjectTracking] DYNAMIC mode active (rAF): R=${R.toFixed(3)} (src=${srcRatio.toFixed(3)}, target=${targetRatio.toFixed(3)}), ` +
      `${subjectKeyframes.length} keyframes` +
      (isReframeSegmentMode ? ' [reframe-segment mode]' : '')
    );

    // Look up cropX from editable crop segments (user may have adjusted).
    // Falls back to interpolateSubjectX from the original keyframes.
    const getCropXAtTime = (relTime) => {
      const { cropSegments } = useTimelineStore.getState();
      if (cropSegments?.length > 0) {
        const seg = cropSegments.find(s => relTime >= s.startTime && relTime < s.endTime);
        if (seg) return seg.cropX;
        const last = cropSegments[cropSegments.length - 1];
        if (relTime >= last.endTime) return last.cropX;
      }
      return interpolateSubjectX(subjectKeyframes, relTime);
    };

    // ── Reframe-segment easing state ──
    // Instead of CSS transitions (which fight with rAF writes), we implement
    // easing in JS: when a position change is detected, we animate from
    // old→new over easeMs using cubic-bezier in the rAF loop itself.
    let easeState = null; // { fromPct, toPct, startTime, durationMs }

    // Ease-out curve: fast start, smooth deceleration — mimics a human
    // camera operator snapping to the subject then settling gently.
    // 1 - (1-t)^3 reaches 87.5% in the first half of the duration.
    const easeCurve = (t) => {
      if (t <= 0) return 0;
      if (t >= 1) return 1;
      return 1 - Math.pow(1 - t, 3);
    };

    // Find the easeMs for the keyframe that starts at or just before relTime
    const getEaseMsForTransition = (relTime) => {
      if (!isReframeSegmentMode || !subjectKeyframes) return 0;
      let active = subjectKeyframes[0];
      for (let i = 1; i < subjectKeyframes.length; i++) {
        if (subjectKeyframes[i].t <= relTime) {
          active = subjectKeyframes[i];
        } else {
          break;
        }
      }
      return active?.easeMs || 0;
    };

    // Apply initial position synchronously to eliminate 1-2 frame gap
    // between effect cleanup and first rAF tick
    {
      const initRel = video.currentTime - clipStart;
      const initSx = getCropXAtTime(initRel);
      const initPct = subjectXToCenterPct(Math.max(0, Math.min(100, initSx)), srcRatio, targetRatio);
      video.style.objectPosition = `${initPct}% ${yPositionPct}%`;
      lastAppliedPctRef.current = initPct;
    }
    let animId;
    const tick = () => {
      const relTime = video.currentTime - clipStart;
      const sx = getCropXAtTime(relTime);
      const targetPct = subjectXToCenterPct(Math.max(0, Math.min(100, sx)), srcRatio, targetRatio);
      const targetRounded = Math.round(targetPct * 10000) / 10000;

      // Determine what to render this frame
      let renderPct = targetPct;

      if (isReframeSegmentMode && targetRounded !== lastPct && lastPct !== null) {
        // Position just changed — start an ease if the transition has easeMs > 0
        const easeMs = getEaseMsForTransition(relTime);
        if (easeMs > 0) {
          const fromPct = lastAppliedPctRef.current ?? targetPct;
          easeState = {
            fromPct,
            toPct: targetPct,
            startTime: performance.now(),
            durationMs: easeMs,
          };
        } else {
          easeState = null; // snap
        }
      }

      // If we're in an active ease, compute the interpolated position
      if (easeState) {
        const elapsed = performance.now() - easeState.startTime;
        if (elapsed >= easeState.durationMs) {
          // Ease complete
          renderPct = easeState.toPct;
          easeState = null;
        } else {
          const t = elapsed / easeState.durationMs;
          renderPct = easeState.fromPct + (easeState.toPct - easeState.fromPct) * easeCurve(t);
        }
      }

      const renderRounded = Math.round(renderPct * 10000) / 10000;
      const renderedPctRef = lastAppliedPctRef.current != null
        ? Math.round(lastAppliedPctRef.current * 10000) / 10000
        : null;

      if (renderRounded !== renderedPctRef) {
        video.style.objectPosition = `${renderPct}% ${yPositionPct}%`;
        lastAppliedPctRef.current = renderPct;
        // Log first 5 updates and then every 30th for debugging
        if (logCount < 5 || logCount % 30 === 0) {
          console.log(
            `[SubjectTracking] t=${relTime.toFixed(2)}s: sx=${sx.toFixed(1)} → objectPosition=${renderPct.toFixed(2)}% ${yPositionPct.toFixed(1)}%` +
            (easeState ? ` [easing]` : '')
          );
        }
        logCount++;
      }

      // Update lastPct to track the *target* (not the eased render), so we
      // detect the next position change correctly
      lastPct = targetRounded;

      animId = requestAnimationFrame(tick);
    };
    animId = requestAnimationFrame(tick);
    return () => {
      cancelAnimationFrame(animId);
      // Do NOT clear video.style.objectPosition here — the cleanup runs
      // after React's DOM commit, so clearing would overwrite the correct
      // static objectPosition that React just applied.
    };
  }, [hasDynamicSubject, subjectKeyframes, clipStart, srcRatio, targetRatio, yPositionPct, isReframeSegmentMode]);

  // When switching from dynamic to static mode (e.g. after "Reset Subject to
  // Center"), ensure the video's objectPosition is set to the correct static
  // value.  This runs after the dynamic effect's cleanup, guaranteeing the
  // final DOM state reflects the centered position.
  useEffect(() => {
    if (hasDynamicSubject || !isCrop) return;
    const video = fgVideoRef.current;
    if (!video) return;
    // Use processed keyframe value if available — it interpolates from nearby
    // scenes and is more accurate than the subjectX prop (which may be 50)
    const effectiveSx = (subjectKeyframes?.length >= 1)
      ? subjectKeyframes[0].x
      : subjectX;
    const sx = safeSubjectX(effectiveSx, srcRatio, targetRatio);
    const centerPct = subjectXToCenterPct(
      Math.max(0, Math.min(100, sx)), srcRatio, targetRatio,
    );
    video.style.objectPosition = `${centerPct}% ${yPositionPct}%`;
  }, [hasDynamicSubject, isCrop, subjectX, subjectKeyframes, srcRatio, targetRatio, yPositionPct]);

  // --- Controls ---
  const togglePlay = useCallback(() => {
    const video = fgVideoRef.current;
    if (!video) return;
    if (video.paused) {
      if (clipEnd && video.currentTime >= clipEnd) {
        video.currentTime = clipStart;
        currentTimeRef.current = clipStart;
      }
      video.play().then(() => setPlaying(true)).catch(() => {});
    } else {
      video.pause();
      setPlaying(false);
      // Update display time immediately on pause so seek bar is accurate
      setDisplayTime(video.currentTime);
    }
  }, [clipStart, clipEnd]);

  const seekBarRef = useRef(null);
  const draggingRef = useRef(false);

  const seekToX = useCallback((clientX) => {
    const video = fgVideoRef.current;
    const bar = seekBarRef.current;
    if (!video || !bar || clipDur <= 0) return;
    const rect = bar.getBoundingClientRect();
    const pct = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
    const newTime = clipStart + pct * clipDur;
    video.currentTime = newTime;
    currentTimeRef.current = newTime;
    setDisplayTime(newTime); // Immediate feedback while scrubbing
  }, [clipStart, clipDur]);

  const onSeekPointerDown = useCallback((e) => {
    draggingRef.current = true;
    seekToX(e.clientX);
    const onMove = (ev) => seekToX(ev.clientX);
    const onUp = () => {
      draggingRef.current = false;
      window.removeEventListener('pointermove', onMove);
      window.removeEventListener('pointerup', onUp);
    };
    window.addEventListener('pointermove', onMove);
    window.addEventListener('pointerup', onUp);
  }, [seekToX]);

  const toggleFullscreen = useCallback(() => {
    const el = fullscreenRef.current;
    if (!el) return;
    if (document.fullscreenElement) {
      document.exitFullscreen().catch(() => {});
    } else {
      el.requestFullscreen().catch(() => {});
    }
  }, []);

  useEffect(() => {
    const onFsChange = () => setIsFullscreen(!!document.fullscreenElement);
    document.addEventListener('fullscreenchange', onFsChange);
    return () => document.removeEventListener('fullscreenchange', onFsChange);
  }, []);

  // --- Subtitle font size (ass_generator.py:146-150) ---
  // Use the OUTPUT resolution for scaling (matching what ASS generator does)
  const outputDims = useMemo(() => {
    if (aspectRatio && ASPECT_RATIO_DIMS[aspectRatio]) {
      return { w: ASPECT_RATIO_DIMS[aspectRatio][0], h: ASPECT_RATIO_DIMS[aspectRatio][1] };
    }
    return { w: sourceWidth, h: sourceHeight };
  }, [aspectRatio, sourceWidth, sourceHeight]);

  const subtitleScale = useMemo(() => {
    if (containerSize.w === 0) return 0;
    // Scale relative to the output resolution, same as backend
    const scaleW = containerSize.w / outputDims.w;
    const scaleH = containerSize.h / outputDims.h;
    return Math.min(scaleW, scaleH);
  }, [containerSize, outputDims]);

  // Two-step scaling matching backend (ass_generator.py:146-158):
  // Step 1: scale from reference (1920x1080) to output resolution
  // Step 2: scale from output resolution to container pixels
  const backendFontScale = useMemo(
    () => Math.min(outputDims.w, outputDims.h) / Math.min(REF_W, REF_H),
    [outputDims],
  );

  const subtitleFontSize = useMemo(() => {
    if (!subtitlesEnabled || subtitleScale === 0) return 14;
    const size = subtitleSettings?.subtitleSize || 'medium';
    const basePx = typeof size === 'number' ? size : (FONT_SIZE_MAP[size] || 30);
    // Step 1: backend scales font to output resolution (integer — matches ASS Fontsize)
    const backendPx = Math.max(16, Math.round(basePx * backendFontScale));
    // Step 2: scale from output resolution to container — use fractional CSS
    // pixels so the preview renders at the exact proportional size as the
    // export.  Rounding to integers at small container sizes introduces
    // visible error (e.g. 30 * 0.37 = 11.1 → round to 11 → 29.7px equiv).
    return Math.max(8, backendPx * subtitleScale);
  }, [subtitlesEnabled, subtitleSettings?.subtitleSize, subtitleScale, backendFontScale]);

  // --- Render subtitle overlay ---
  const renderSubtitles = () => {
    if (!subtitlesEnabled || !currentSubtitle) return null;

    const position = subtitleSettings?.subtitlePosition || 'bottom';
    const maxWidthPct = subtitleSettings?.subtitleMaxWidth ?? 90;
    const offsetVPct = subtitleSettings?.subtitleOffsetV ?? 4;
    const bgEnabled = subtitleSettings?.subtitleBgEnabled || false;
    const bgColor = subtitleSettings?.subtitleBgColor || '#000000';
    const bgOpacity = subtitleSettings?.subtitleBgOpacity ?? 75;
    const fontWeight = typeof subtitleSettings?.subtitleFontWeight === 'number' ? subtitleSettings.subtitleFontWeight : subtitleSettings?.subtitleFontWeight === 'bold' ? 700 : 400;
    const rawFont = subtitleSettings?.subtitleFont || 'DM Sans';
    // Wrap in quotes for multi-word names and add generic fallback so the
    // browser never falls back to the inherited UI font stack when the
    // @font-face file is still loading or fails to download.
    const fontFamily = `"${rawFont}", sans-serif`;
    const showLabels = subtitleSettings?.showSpeakerLabels ?? false;
    const color = getSpeakerColor(currentSubtitle.speaker, speakersOrdered, subtitleSettings);

    // Outline settings — clamp values to match backend (ass_generator.py:200-202)
    const olColorHex = (subtitleSettings?.subtitleOutlineColor || '#000000').replace('#', '');
    const olR = parseInt(olColorHex.substring(0, 2), 16) || 0;
    const olG = parseInt(olColorHex.substring(2, 4), 16) || 0;
    const olB = parseInt(olColorHex.substring(4, 6), 16) || 0;
    const olOpacity = Math.max(0, Math.min(100, subtitleSettings?.subtitleOutlineOpacity ?? 100)) / 100;
    const olWidth = Math.max(0, Math.min(10, subtitleSettings?.subtitleOutlineWidth ?? 2));
    // Two-step scaling matching backend reference resolution
    const backendOlWidth = Math.max(0, Math.round(olWidth * backendFontScale));
    // Use fractional CSS pixels for container scaling — integer rounding at
    // small container sizes causes massive proportional error for thin values
    // (e.g. 2 * 0.37 = 0.74 → round to 1 → 2.7px equiv instead of 2px).
    const scaledOlWidth = backendOlWidth * subtitleScale;

    // Build outline style matching ASS rendering.
    // ASS uses BorderStyle=1 (outline) with a hard drop shadow whose depth
    // scales proportionally with the outline width: max(1, min(4, round(olWidth * 0.75))).
    // We replicate this with CSS text-stroke + text-shadow.
    let outlineStyle;
    if (bgEnabled) {
      // Background box mode — no visible outline (blends into box in ASS)
      outlineStyle = {};
    } else if (scaledOlWidth > 0) {
      // Outline mode — use text-stroke to match ASS BorderStyle=1.
      // Shadow depth matches ASS: proportional to outline width.
      const shadowDepth = Math.max(1, Math.min(4, Math.round(backendOlWidth * 0.75)));
      const scaledShadow = shadowDepth * subtitleScale;
      const olColorStr = `rgba(${olR},${olG},${olB},${olOpacity})`;
      const dropShadow = `${scaledShadow}px ${scaledShadow}px 0px rgba(0,0,0,0.5)`;
      outlineStyle = {
        WebkitTextStroke: `${scaledOlWidth * 2}px ${olColorStr}`,
        paintOrder: 'stroke fill',
        textShadow: outlineTextShadow(scaledOlWidth, olColorStr, dropShadow),
      };
    } else {
      // No outline, no background — minimal shadow for readability
      outlineStyle = {
        textShadow: '1px 1px 2px rgba(0,0,0,0.8)',
      };
    }

    // Margin calculations matching backend (ass_generator.py:196-235)
    const clampedMaxWidth = Math.max(20, Math.min(100, maxWidthPct));
    const clampedOffsetV = Math.max(0, Math.min(100, offsetVPct));

    // Horizontal: backend uses max(20, ...) minimum + safe-area cap
    const marginH_px = Math.max(20, Math.floor(outputDims.w * (100 - clampedMaxWidth) / 100 / 2));
    const maxMarginH = Math.floor(outputDims.w * 0.40); // (1 - MIN_TEXT_AREA_W=0.20) / 2
    const effectiveMarginH = Math.min(marginH_px, maxMarginH) / outputDims.w * 100;

    // Vertical positioning based on position setting
    let positionStyle;
    if (position === 'top') {
      positionStyle = { top: `${clampedOffsetV}%` };
    } else if (position === 'center') {
      positionStyle = { top: '50%', transform: 'translateY(-50%)' };
    } else {
      positionStyle = { bottom: `${clampedOffsetV}%` };
    }

    const text = showLabels && currentSubtitle.speaker
      ? `${currentSubtitle.speaker}: ${currentSubtitle.text}`
      : currentSubtitle.text;

    // Active word highlight settings
    const awEnabled = activeWordEnabled;
    const awColor = subtitleSettings?.activeWordColor || '#FFD700';
    const awOutlineColor = subtitleSettings?.activeWordOutlineColor || '#000000';
    const awBgColor = subtitleSettings?.activeWordBgColor || '#000000';
    const awBgOpacity = subtitleSettings?.activeWordBgOpacity ?? 0;

    // Build per-word content when active word highlighting is enabled
    let textContent;
    if (awEnabled && currentWordIdx >= 0) {
      const words = currentSubtitle.text.split(/\s+/).filter(Boolean);
      const prefix = showLabels && currentSubtitle.speaker ? `${currentSubtitle.speaker}: ` : '';

      // Parse active word outline color
      const awOlHex = awOutlineColor.replace('#', '');
      const awOlR = parseInt(awOlHex.substring(0, 2), 16) || 0;
      const awOlG = parseInt(awOlHex.substring(2, 4), 16) || 0;
      const awOlB = parseInt(awOlHex.substring(4, 6), 16) || 0;

      textContent = (
        <>
          {prefix}
          {words.map((word, idx) => {
            const isActive = idx === currentWordIdx;
            const wordStyle = isActive
              ? {
                  color: awColor,
                  ...(scaledOlWidth > 0
                    ? {
                        WebkitTextStroke: `${scaledOlWidth * 2}px rgba(${awOlR},${awOlG},${awOlB},${olOpacity})`,
                        paintOrder: 'stroke fill',
                        textShadow: outlineTextShadow(scaledOlWidth, `rgba(${awOlR},${awOlG},${awOlB},${olOpacity})`),
                      }
                    : {}),
                  ...(awBgOpacity > 0
                    ? {
                        backgroundColor: hexToRgba(awBgColor, awBgOpacity / 100),
                        // ASS \4c tag renders a rectangular highlight — no
                        // border-radius support.  Omit borderRadius to match export.
                        padding: `0 ${Math.max(1, 2 * subtitleScale)}px`,
                      }
                    : {}),
                }
              : {};
            return (
              <span key={idx} style={wordStyle}>
                {word}{idx < words.length - 1 ? ' ' : ''}
              </span>
            );
          })}
        </>
      );
    } else {
      textContent = text;
    }

    return (
      <div
        style={{
          position: 'absolute',
          left: `${effectiveMarginH}%`,
          right: `${effectiveMarginH}%`,
          textAlign: 'center',
          pointerEvents: 'none',
          zIndex: 5,
          ...positionStyle,
        }}
      >
        <span
          style={{
            display: 'inline-block',
            fontFamily,
            fontSize: subtitleFontSize,
            fontWeight,
            color,
            lineHeight: 1.4,
            wordWrap: 'break-word',
            overflowWrap: 'break-word',
            whiteSpace: 'pre-wrap',
            ...outlineStyle,
            ...(bgEnabled
              ? {
                  background: hexToRgba(bgColor, bgOpacity / 100),
                  // ASS BorderStyle=3 uses Outline width as uniform padding on
                  // all 4 sides.  Two-step: compute backend value then scale.
                  // Backend: max(int(4 * font_scale), 2) — use Math.floor to
                  // match Python int() truncation.
                  padding: `${Math.max(1, Math.max(Math.floor(4 * backendFontScale), 2) * subtitleScale)}px`,
                }
              : {}),
          }}
        >
          {textContent}
        </span>
      </div>
    );
  };

  // Determine active layout mode for current frame
  const activeLayoutMode = useMemo(() => {
    if (!layoutTimeline?.length || defaultLayoutMode === 'single') return 'single';
    return defaultLayoutMode;
  }, [layoutTimeline, defaultLayoutMode]);

  // Sync split-mode bottom video with top video
  useEffect(() => {
    if (activeLayoutMode !== 'split') return;
    const topVideo = fgVideoRef.current;
    const bottomVideo = splitBottomVideoRef.current;
    if (!topVideo || !bottomVideo) return;

    const syncTime = () => {
      if (Math.abs(bottomVideo.currentTime - topVideo.currentTime) > 0.1) {
        bottomVideo.currentTime = topVideo.currentTime;
      }
    };
    const syncPlay = () => { bottomVideo.play().catch(() => {}); syncTime(); };
    const syncPause = () => { bottomVideo.pause(); syncTime(); };

    topVideo.addEventListener('play', syncPlay);
    topVideo.addEventListener('pause', syncPause);
    topVideo.addEventListener('seeked', syncTime);
    topVideo.addEventListener('timeupdate', syncTime);

    // Initial sync
    bottomVideo.currentTime = topVideo.currentTime;
    if (!topVideo.paused) bottomVideo.play().catch(() => {});

    return () => {
      topVideo.removeEventListener('play', syncPlay);
      topVideo.removeEventListener('pause', syncPause);
      topVideo.removeEventListener('seeked', syncTime);
      topVideo.removeEventListener('timeupdate', syncTime);
    };
  }, [activeLayoutMode]);

  // --- Video area ---
  // Single wrapper div (no key="crop"/"original") to preserve the <video> element
  // across crop/non-crop transitions. Different keys would destroy and recreate
  // the video element, causing flash and loss of playback position.
  const renderVideoArea = () => {
    const srcRatioLocal = sourceWidth / sourceHeight;

    // Shared overlay for the 503 + Retry-After preview-transcode window
    // (see useVideoLoadRetry + backend/services/browser_preview.py).
    // Injected into both split and single layout containers — both
    // layouts hang <video> off fgVideoRef, so one overlay covers both.
    const previewLoadOverlay = (fgPreparing || fgLoadError) ? (
      <div style={{
        position: 'absolute',
        inset: 0,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        flexDirection: 'column',
        gap: 10,
        background: fgLoadError ? 'rgba(0,0,0,0.7)' : 'rgba(0,0,0,0.55)',
        color: '#fff',
        fontSize: 13,
        fontFamily: 'var(--font-mono, monospace)',
        zIndex: 16,
        pointerEvents: fgLoadError ? 'auto' : 'none',
      }}>
        {fgLoadError ? (
          <>
            <div>Failed to load video</div>
            <button
              onClick={retryFgLoad}
              style={{
                padding: '4px 12px',
                background: 'var(--accent-amber, #FF9F0A)',
                color: '#000',
                border: 0,
                borderRadius: 4,
                cursor: 'pointer',
                fontSize: 12,
              }}
            >
              Retry
            </button>
          </>
        ) : (
          <>
            <div style={{
              width: 18, height: 18,
              border: '2px solid rgba(255,255,255,0.3)',
              borderTopColor: '#fff',
              borderRadius: '50%',
              animation: 'cp-preview-spin 0.9s linear infinite',
            }} />
            <div>Preparing preview…</div>
            <style>{`@keyframes cp-preview-spin { to { transform: rotate(360deg); } }`}</style>
          </>
        )}
      </div>
    ) : null;

    // Layout-aware rendering: SPLIT mode shows two video instances stacked
    if (activeLayoutMode === 'split' && isCrop && faceRegistry?.slots?.length >= 2) {
      const sortedSlots = [...(faceRegistry.slots || [])].sort((a, b) => a.x - b.x);
      const topSpeakerX = sortedSlots[0]?.x ?? 25;
      const bottomSpeakerX = sortedSlots[1]?.x ?? 75;
      const topPct = subjectXToCenterPct(topSpeakerX, srcRatioLocal, targetRatio);
      const bottomPct = subjectXToCenterPct(bottomSpeakerX, srcRatioLocal, targetRatio);

      return (
        <div style={{ position: 'relative', width: '100%', height: '100%' }}>
          {previewLoadOverlay}
          {trackingStatus && (
            <div style={{
              position: 'absolute', top: 8, right: 8, zIndex: 15,
              display: 'flex', alignItems: 'center', gap: 5,
              padding: '3px 8px', borderRadius: 6,
              background: 'rgba(0,0,0,0.6)', backdropFilter: 'blur(4px)',
              fontSize: 11, color: '#e5e7eb', pointerEvents: 'none',
            }}>
              <span style={{
                width: 6, height: 6, borderRadius: '50%',
                background: '#22c55e',
              }} />
              Split view
            </div>
          )}
          <div style={{ display: 'flex', flexDirection: 'column', width: '100%', height: '100%' }}>
            <div style={{ flex: 1, overflow: 'hidden', position: 'relative' }}>
              <video
                ref={fgVideoRef}
                src={src}
                preload="auto"
                playsInline
                style={{
                  width: '100%',
                  height: '100%',
                  display: 'block',
                  objectFit: 'cover',
                  objectPosition: `${topPct}% ${yPositionPct}%`,
                }}
                onClick={togglePlay}
              />
            </div>
            <div style={{ height: '3px', background: 'rgba(0,0,0,0.3)', flexShrink: 0 }} />
            <div style={{ flex: 1, overflow: 'hidden', position: 'relative' }}>
              <video
                ref={splitBottomVideoRef}
                src={src}
                preload="auto"
                playsInline
                muted
                style={{
                  width: '100%',
                  height: '100%',
                  display: 'block',
                  objectFit: 'cover',
                  objectPosition: `${bottomPct}% ${yPositionPct}%`,
                }}
                onClick={togglePlay}
              />
            </div>
          </div>
        </div>
      );
    }

    const videoStyle = {
      width: '100%',
      height: '100%',
      display: 'block',
    };

    if (isCrop) {
      const initialSx = hasDynamicSubject
        ? subjectKeyframes[0].x
        : (subjectKeyframes?.length >= 1 ? subjectKeyframes[0].x : safeSubjectX(subjectX, srcRatioLocal, targetRatio));
      const centerPct = subjectXToCenterPct(Math.max(0, Math.min(100, initialSx)), srcRatioLocal, targetRatio);
      Object.assign(videoStyle, {
        objectFit: 'cover',
        objectPosition: `${centerPct}% ${yPositionPct}%`,
      });
    } else {
      Object.assign(videoStyle, {
        objectFit: 'contain',
      });
    }

    return (
      <div style={{ position: 'relative', width: '100%', height: '100%' }}>
        {previewLoadOverlay}
        {/* Subject tracking status indicator */}
        {trackingStatus && (
          <div style={{
            position: 'absolute', top: 8, right: 8, zIndex: 15,
            display: 'flex', alignItems: 'center', gap: 5,
            padding: '3px 8px', borderRadius: 6,
            background: 'rgba(0,0,0,0.6)', backdropFilter: 'blur(4px)',
            fontSize: 11, color: '#e5e7eb', pointerEvents: 'none',
          }}>
            <span style={{
              width: 6, height: 6, borderRadius: '50%',
              background: trackingStatus.color,
              boxShadow: trackingStatus.mode === 'dynamic' ? `0 0 4px ${trackingStatus.color}` : 'none',
            }} />
            {trackingStatus.label}
          </div>
        )}
        {/* Subject tracking overlay — shows crop center and tracking status */}
        {isCrop && (
          <>
            {/* Center crosshair — shows where the subject is being centered */}
            {hasDynamicSubject && (
              <>
                {/* Vertical center line */}
                <div style={{
                  position: 'absolute',
                  top: '20%', bottom: '20%', left: '50%',
                  width: 1, marginLeft: -0.5,
                  background: 'rgba(16, 185, 129, 0.5)',
                  zIndex: 14, pointerEvents: 'none',
                }} />
                {/* Horizontal center line */}
                <div style={{
                  position: 'absolute',
                  left: '20%', right: '20%',
                  top: yPositionPct !== 50 ? `${Math.max(10, Math.min(90, 100 - yPositionPct))}%` : '50%',
                  height: 1, marginTop: -0.5,
                  background: 'rgba(16, 185, 129, 0.35)',
                  zIndex: 14, pointerEvents: 'none',
                }} />
                {/* Center target dot */}
                <div style={{
                  position: 'absolute',
                  left: '50%',
                  top: yPositionPct !== 50 ? `${Math.max(10, Math.min(90, 100 - yPositionPct))}%` : '50%',
                  width: 8, height: 8, marginLeft: -4, marginTop: -4,
                  borderRadius: '50%',
                  border: '1.5px solid rgba(16, 185, 129, 0.7)',
                  background: 'rgba(16, 185, 129, 0.15)',
                  zIndex: 14, pointerEvents: 'none',
                }} />
              </>
            )}
            {/* Crop boundary — subtle border showing the active crop region */}
            <div style={{
              position: 'absolute', inset: 0,
              border: '1px solid rgba(255, 255, 255, 0.15)',
              borderRadius: 2,
              zIndex: 13, pointerEvents: 'none',
            }} />
          </>
        )}
        <video
          ref={fgVideoRef}
          src={src}
          preload="auto"
          playsInline
          style={renderPlanData ? { ...videoStyle, display: 'none' } : videoStyle}
          onClick={togglePlay}
        />
        {/* RenderPlan Canvas: shown when render plan is available for pixel-perfect preview */}
        {renderPlanData && (
          <canvas
            ref={canvasRef}
            onClick={togglePlay}
            style={{
              width: '100%',
              height: '100%',
              display: 'block',
              objectFit: 'contain',
            }}
          />
        )}
      </div>
    );
  };

  // --- Compute container constraints ---
  const maxH = isFullscreen ? 'calc(100vh - 50px)' : (inline ? '50vh' : '65vh');
  const maxHpx = (typeof window !== 'undefined' ? window.innerHeight : 900)
    * (inline ? 0.50 : 0.65);

  // Compute max width: for portrait ratios, constrain width to maintain aspect ratio
  // within the maxHeight budget.  Always use pixel values for consistency.
  const contentMaxWidth = useMemo(() => {
    if (targetRatio < 1) {
      // Portrait: width = height * ratio
      const constrained = Math.round(maxHpx * targetRatio);
      return inline ? constrained : Math.min(constrained, 800);
    }
    // Landscape or square
    return inline ? 9999 : 800;
  }, [targetRatio, maxHpx, inline]);

  const btnStyle = {
    background: 'none', border: 'none', color: 'var(--video-controls-text)',
    fontSize: isMobile ? 20 : 16, padding: isMobile ? '6px 10px' : '2px 6px',
    cursor: 'pointer', lineHeight: 1, minHeight: isMobile ? 44 : undefined,
  };

  const content = (
    <div
      ref={fullscreenRef}
      style={isFullscreen ? {
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        background: 'var(--video-bg)',
        width: '100vw',
        height: '100vh',
      } : {
        position: 'relative',
        maxWidth: contentMaxWidth,
        width: '100%',
        margin: '0 auto',
        background: 'var(--bg-panel)',
        border: inline ? 'none' : '1px solid var(--border)',
        borderRadius: inline ? 0 : (isMobile ? 0 : 'var(--radius-md)'),
        overflow: 'hidden',
      }}
    >
      {/* Header — hidden in fullscreen */}
      {!inline && !isFullscreen && (
        <div style={{
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          padding: '10px 16px',
          borderBottom: '1px solid var(--border)',
        }}>
          <div>
            <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-primary)' }}>
              {title || 'Clip Preview'}
            </div>
            <div style={{ fontSize: 11, fontFamily: 'var(--font-mono)', color: 'var(--text-muted)', marginTop: 2 }}>
              {formatTime(clipStart)} &rarr; {formatTime(clipEnd)}
              <span style={{ marginLeft: 8, color: 'var(--text-secondary)' }}>({formatTime(clipDur / speed)})</span>
              {speed !== 1.0 && (
                <span style={{ marginLeft: 8, color: '#FFD60A', fontSize: 10 }}>
                  {speed}x speed
                </span>
              )}
              {aspectRatio && (
                <span style={{ marginLeft: 8, color: 'var(--accent-amber)', fontSize: 10 }}>
                  {aspectRatio}{isCrop ? ' (crop)' : ''}
                </span>
              )}
              {subtitlesEnabled && (
                <span style={{ marginLeft: 8, color: 'var(--accent-cyan)', fontSize: 10 }}>Subtitles</span>
              )}
            </div>
          </div>
          {onClose && (
            <button
              onClick={onClose}
              style={{
                background: 'none', border: 'none',
                color: 'var(--text-muted)', fontSize: 20,
                cursor: 'pointer', padding: '0 4px', lineHeight: 1,
              }}
            >
              &times;
            </button>
          )}
        </div>
      )}

      {/* Video frame — always enforce the target aspect ratio */}
      <div
        ref={containerRef}
        style={{
          position: 'relative',
          background: 'var(--video-bg)',
          cursor: 'pointer',
          overflow: 'hidden',
          aspectRatio: `${targetRatio}`,
          ...(isFullscreen ? {
            height: '100vh',
            maxWidth: '100vw',
            width: 'auto',
          } : {
            width: '100%',
            maxHeight: maxH,
          }),
        }}
        onClick={togglePlay}
        onMouseEnter={() => setHovered(true)}
        onMouseLeave={() => setHovered(false)}
      >
        {renderVideoArea()}
        {renderSubtitles()}

        {/* Reframe debug overlay (dev mode) */}
        {renderPlanData && renderPlanData.debug && (
          <ReframeDebugOverlay
            renderPlan={renderPlanData}
            currentTime={displayTime - clipStart}
            clipDuration={clipEnd - clipStart}
          />
        )}

        {/* Loading overlay */}
        {!videoReady && (
          <div style={{
            position: 'absolute', inset: 0, zIndex: 20,
            display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
            background: 'var(--video-bg, #000)',
            gap: 10,
          }}>
            <div style={{
              width: 24, height: 24,
              border: '2px solid rgba(255,255,255,0.15)',
              borderTopColor: 'var(--accent-cyan, #0A84FF)',
              borderRadius: '50%',
              animation: 'spin 0.8s linear infinite',
            }} />
            <span style={{ fontSize: 11, color: 'rgba(255,255,255,0.5)' }}>Loading...</span>
            <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
          </div>
        )}

        {/* Play/pause overlay */}
        {videoReady && !playing && (
          <div style={{
            position: 'absolute', inset: 0, zIndex: 6,
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            background: 'var(--overlay-light)', pointerEvents: 'none',
          }}>
            <div style={{
              width: 56, height: 56, borderRadius: '50%',
              background: 'var(--overlay-heavy)', border: '2px solid var(--video-controls-text)',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              fontSize: 22, color: 'var(--video-controls-text)', paddingLeft: 3,
            }}>
              &#9654;
            </div>
          </div>
        )}
      </div>

      {/* Controls — hidden until video is ready */}
      <div style={{
        background: 'var(--bg-elevated)',
        padding: isMobile ? '8px 12px' : '6px 12px',
        ...(videoReady ? {} : { opacity: 0.3, pointerEvents: 'none' }),
      }}>
        {/* Draggable seek bar */}
        <div
          ref={seekBarRef}
          onPointerDown={onSeekPointerDown}
          style={{
            height: isMobile ? 8 : 6, background: 'var(--border)', cursor: 'pointer',
            position: 'relative', borderRadius: 4, marginBottom: 8, touchAction: 'none',
          }}
        >
          <div style={{
            height: '100%', width: `${progress}%`,
            background: 'var(--accent-cyan)', borderRadius: 3,
            position: 'relative',
          }}>
            <div style={{
              position: 'absolute', right: -5, top: -3,
              width: 12, height: 12, borderRadius: '50%',
              background: 'var(--accent-cyan)',
            }} />
          </div>
        </div>

        {/* Buttons row: play/pause, volume, fullscreen */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <button onClick={togglePlay} style={{ ...btnStyle, fontSize: 18 }}>
            {playing ? '\u23F8' : '\u25B6'}
          </button>

          <span style={{
            fontFamily: 'var(--font-mono)', fontSize: 11,
            color: 'var(--text-secondary)', marginLeft: 2,
          }}>
            {formatTime(elapsed / speed)} / {formatTime(clipDur / speed)}
          </span>

          <div style={{ flex: 1 }} />

          <input
            type="range" min="0" max="1" step="0.1"
            value={volume}
            onChange={(e) => {
              const v = parseFloat(e.target.value);
              setVolume(v);
              if (fgVideoRef.current) fgVideoRef.current.volume = v;
            }}
            style={{ width: isMobile ? 40 : 50, accentColor: 'var(--accent-cyan)' }}
          />

          <button
            onClick={() => {
              const idx = SPEED_OPTIONS.indexOf(speed);
              const next = SPEED_OPTIONS[(idx + 1) % SPEED_OPTIONS.length];
              setSpeed(next);
            }}
            style={{
              background: speed !== 1.0 ? 'rgba(255, 214, 10, 0.12)' : 'none',
              border: speed !== 1.0 ? '1px solid rgba(255, 214, 10, 0.25)' : '1px solid rgba(255,255,255,0.1)',
              color: speed !== 1.0 ? '#FFD60A' : 'var(--text-secondary)',
              fontSize: 11,
              fontFamily: 'var(--font-mono)',
              fontWeight: 500,
              cursor: 'pointer',
              padding: '2px 6px',
              borderRadius: 4,
              lineHeight: 1.2,
            }}
            title="Playback speed (click to cycle)"
          >
            {speed}x
          </button>

          <button
            onClick={toggleFullscreen}
            style={{
              background: 'none', border: 'none',
              color: 'var(--text-secondary)', fontSize: 16,
              cursor: 'pointer', padding: '0 4px', lineHeight: 1,
            }}
            title={isFullscreen ? 'Exit fullscreen' : 'Fullscreen'}
          >
            {isFullscreen ? '\u2715' : '\u26F6'}
          </button>
        </div>
      </div>
    </div>
  );

  // Inline mode: render directly, no lightbox overlay
  if (inline) {
    return content;
  }

  // Lightbox mode
  return (
    <div
      onClick={onClose}
      style={{
        position: 'fixed', inset: 0, zIndex: 9999,
        background: 'var(--lightbox-bg)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        padding: isMobile ? 0 : 24,
      }}
    >
      {/* Stop clicks inside the player from closing the lightbox */}
      <div onClick={(e) => e.stopPropagation()} style={{ maxWidth: contentMaxWidth, width: '100%' }}>
        {content}
      </div>
    </div>
  );
}
