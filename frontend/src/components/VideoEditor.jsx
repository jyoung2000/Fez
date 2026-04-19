import React, { useRef, useState, useEffect, useMemo, useCallback } from 'react';
import { processKeyframes, interpolateSubjectX, isDynamic, safeSubjectX, subjectXToCenterPct, detectPositionClusters, buildSubjectKeyframes, keyframesToCropSegments, keyframesToCropSegmentsWithSlots, extractSlotTimelineFromRenderPlan, buildKeyframesFromSubjectTrack } from '../utils/subjectTracking';
import useTimelineStore from '../stores/timelineStore';
import useResponsive from '../hooks/useResponsive';
import useTimelinePersistence from '../hooks/useTimelinePersistence';
import useKeyboardShortcuts from '../hooks/useKeyboardShortcuts';
import useEncodingManager from '../hooks/useEncodingManager';
import { usePlayer } from '../contexts/PlayerContext';
import EditorErrorBoundary from './EditorErrorBoundary';
import Timeline from './Timeline';
import TimelineOverlay from './TimelineOverlay';
import MediaUploader from './MediaUploader';
import PropertiesPanel from './PropertiesPanel';
import ToolBar from './ToolBar';
import EffectsPanel from './EffectsPanel';
import TransitionPicker from './TransitionPicker';
import ExportDialog from './ExportDialog';
import InteractiveOverlay from './InteractiveOverlay';
import { hexToRgbString } from '../utils/colorUtils';
import { runEditorQA, autoFixTrackCompatibility } from '../utils/editorQA';
import './VideoEditor.css';

// ── Segment Color Palette ────────────────────────────────────────────────────
const SEGMENT_COLORS = [
  '#0A84FF', // Blue (Apple system blue)
  '#30D158', // Green
  '#FF9F0A', // Orange
  '#AF52DE', // Purple
  '#FF375F', // Pink
  '#64D2FF', // Cyan
  '#FFD60A', // Yellow
  '#AC8E68', // Tan/Brown
];

// Speaker color palette (must match SubtitleOverlay / ClipSettingsPanel)
const DEFAULT_SPEAKER_PALETTE = [
  '#00D9FF', '#F59E0B', '#10B981', '#A78BFA', '#EF4444', '#EC4899',
  '#06B6D4', '#8B5CF6', '#F97316', '#14B8A6', '#E879F9', '#84CC16',
  '#FB7185', '#38BDF8', '#FBBF24', '#34D399', '#C084FC', '#F472B6',
  '#22D3EE', '#A3E635', '#FB923C', '#2DD4BF', '#818CF8', '#F87171',
];

// ── Constants ────────────────────────────────────────────────────────────────
const ASPECT_RATIO_VALUES = {
  '16:9': 16 / 9,
  '9:16': 9 / 16,
  '1:1': 1.0,
  '4:5': 4 / 5,
};

const SPEED_PRESETS = [
  { value: 0.25, label: '' },
  { value: 0.5, label: '' },
  { value: 0.75, label: '' },
  { value: 1.0, label: 'Normal' },
  { value: 1.25, label: '' },
  { value: 1.5, label: '' },
  { value: 2.0, label: '' },
  { value: 4.0, label: '' },
];

// ── SVG Icons (Apple SF Symbols aesthetic) ────────────────────────────────────
const Icon = {
  Play: () => (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor" stroke="none">
      <path d="M6.5 4.1c-.9-.5-2 .1-2 1.2v13.4c0 1.1 1.1 1.7 2 1.2l11.6-6.7c.9-.5.9-1.8 0-2.4L6.5 4.1z" />
    </svg>
  ),
  Pause: () => (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor" stroke="none">
      <rect x="5" y="3" width="5" height="18" rx="1.5" />
      <rect x="14" y="3" width="5" height="18" rx="1.5" />
    </svg>
  ),
  SkipBack: () => (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <polygon points="11 19 2 12 11 5 11 19" fill="currentColor" stroke="none" />
      <line x1="22" y1="5" x2="22" y2="19" />
      <text x="18" y="22" fontSize="8" fill="currentColor" stroke="none" fontFamily="var(--font-mono, monospace)">5</text>
    </svg>
  ),
  SkipForward: () => (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <polygon points="13 19 22 12 13 5 13 19" fill="currentColor" stroke="none" />
      <line x1="2" y1="5" x2="2" y2="19" />
      <text x="3" y="22" fontSize="8" fill="currentColor" stroke="none" fontFamily="var(--font-mono, monospace)">5</text>
    </svg>
  ),
  VolumeMuted: () => (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M11 5L6 9H2v6h4l5 4V5z" fill="currentColor" opacity="0.5" stroke="none" />
      <line x1="23" y1="9" x2="17" y2="15" />
      <line x1="17" y1="9" x2="23" y2="15" />
    </svg>
  ),
  VolumeLow: () => (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M11 5L6 9H2v6h4l5 4V5z" fill="currentColor" opacity="0.5" stroke="none" />
      <path d="M15.54 8.46a5 5 0 010 7.07" />
    </svg>
  ),
  VolumeMed: () => (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M11 5L6 9H2v6h4l5 4V5z" fill="currentColor" opacity="0.5" stroke="none" />
      <path d="M15.54 8.46a5 5 0 010 7.07" />
      <path d="M19.07 4.93a10 10 0 010 14.14" />
    </svg>
  ),
  VolumeHigh: () => (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M11 5L6 9H2v6h4l5 4V5z" fill="currentColor" opacity="0.5" stroke="none" />
      <path d="M15.54 8.46a5 5 0 010 7.07" />
      <path d="M19.07 4.93a10 10 0 010 14.14" />
    </svg>
  ),
  Fullscreen: () => (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M8 3H5a2 2 0 00-2 2v3" />
      <path d="M21 8V5a2 2 0 00-2-2h-3" />
      <path d="M3 16v3a2 2 0 002 2h3" />
      <path d="M16 21h3a2 2 0 002-2v-3" />
    </svg>
  ),
  ExitFullscreen: () => (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M4 14h4v4" />
      <path d="M20 10h-4V6" />
      <path d="M14 10l7-7" />
      <path d="M3 21l7-7" />
    </svg>
  ),
  Close: () => (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <line x1="18" y1="6" x2="6" y2="18" />
      <line x1="6" y1="6" x2="18" y2="18" />
    </svg>
  ),
};

// ── Helpers ──────────────────────────────────────────────────────────────────
// subjectXToCenterPct is imported from subjectTracking.js (shared with ClipPreview)

function formatTimecode(seconds) {
  if (!seconds || isNaN(seconds) || seconds < 0) return '0:00.00';
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  const ms = Math.floor((seconds % 1) * 100);
  return `${m}:${s.toString().padStart(2, '0')}.${ms.toString().padStart(2, '0')}`;
}

function formatTimeShort(seconds) {
  if (!seconds || isNaN(seconds) || seconds < 0) return '0:00';
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, '0')}`;
}

/** Parse user-typed timecodes: "3:20", "3:20.5", "200" (raw seconds), "1:05:30" */
function parseTimecodeInput(str) {
  if (!str || !str.trim()) return null;
  const s = str.trim();
  // h:mm:ss or h:mm:ss.cc
  const hms = s.match(/^(\d+):(\d{1,2}):(\d{1,2})(?:\.(\d+))?$/);
  if (hms) {
    return parseInt(hms[1]) * 3600 + parseInt(hms[2]) * 60 + parseInt(hms[3]) + (hms[4] ? parseFloat('0.' + hms[4]) : 0);
  }
  // m:ss or m:ss.cc
  const ms = s.match(/^(\d+):(\d{1,2})(?:\.(\d+))?$/);
  if (ms) {
    return parseInt(ms[1]) * 60 + parseInt(ms[2]) + (ms[3] ? parseFloat('0.' + ms[3]) : 0);
  }
  // Raw number (seconds)
  const num = parseFloat(s);
  if (!isNaN(num) && num >= 0) return num;
  return null;
}

// ── Component ────────────────────────────────────────────────────────────────
const ASPECT_RATIO_OPTIONS = [
  { value: null, label: 'Original', icon: null },
  { value: '16:9', label: '16:9', icon: 'landscape' },
  { value: '9:16', label: '9:16', icon: 'portrait' },
  { value: '1:1', label: '1:1', icon: 'square' },
  { value: '4:5', label: '4:5', icon: 'portrait' },
];

export default function VideoEditor({
  src,
  clipStart = 0,
  clipEnd = 0,
  onTimeUpdate,
  onTrimChange,
  onApplyTrim,
  onVolumeChange,
  onSpeedChange,
  onSegmentsChange,
  onAspectRatioChange,
  aspectRatio,
  sourceWidth = 1920,
  sourceHeight = 1080,
  subjectX = 50,
  scenes,
  sceneCuts = null,
  // Dense per-frame subject track from the backend pipeline. When
  // present this is the authoritative source for the 9:16 / 1:1 reframe
  // window — the sparse scene-based ``processKeyframes`` path is only
  // a fallback for legacy jobs that were analyzed before the dense
  // track landed.
  subjectTrack = null,
  layoutTimeline = null,
  faceRegistry = null,
  defaultLayoutMode = 'single',
  initialTime,
  initialVolume,
  initialSpeed,
  initialSegments,
  title,
  onClose,
  subtitleOverlay,
  compact = false,
  // Settings-panel integration
  settings,
  speakers,
  speakerNames,
  onSettingsChange,
  // Multi-track editor props
  jobId,
  clipId,
  transcript,
  onTranscriptUpdated,
  // Public share integration. When provided, the export POST is
  // redirected to the token-gated ``/api/share/public/{token}/...``
  // endpoints so share-link recipients can export without signing in.
  // The share variant enforces scope on the backend (clip-scoped links
  // can only export their own clip) — the frontend doesn't need to
  // guard separately.
  shareToken = null,
  // Analysis state
  isProcessing = false,
  // Expose internal video element to parent via callback
  onVideoRef,
  // Expose computed subject keyframes to parent for export parity
  onSubjectKeyframes,
  // Optional RenderPlan for backend slot identity (Phase D)
  renderPlan = null,
}) {
  const { isMobile } = useResponsive();

  // ── Multi-track editor state ──────────────────────
  // Default to open on desktop so multi-track edits are immediately visible
  const [showMultiTrack, setShowMultiTrack] = useState(() => {
    const w = typeof window !== 'undefined' ? window.innerWidth : 1024;
    // Open by default on tablet+ (≥768px), collapsed on phone (<768px).
    // Phone users can toggle open via the Multi-Track Editor button.
    return w >= 768;
  });
  const [showMediaLibrary, setShowMediaLibrary] = useState(false);
  const [showProperties, setShowProperties] = useState(false);
  const [showExportDialog, setShowExportDialog] = useState(false);
  const [showEffectsPanel, setShowEffectsPanel] = useState(false);
  const [showTransitions, setShowTransitions] = useState(false);
  const initFromClip = useTimelineStore((s) => s.initFromClip);
  const setSceneCutsInStore = useTimelineStore((s) => s.setSceneCuts);
  const timelineStoreItems = useTimelineStore((s) => s.items);

  // Push detected scene cuts into the timeline store so Timeline's
  // findSnapTarget can use them for trim-handle snapping.
  useEffect(() => {
    setSceneCutsInStore(Array.isArray(sceneCuts) ? sceneCuts : []);
  }, [sceneCuts, setSceneCutsInStore]);
  const setSelectedItemId = useTimelineStore((s) => s.setSelectedItemId);
  const storeSelectedItemId = useTimelineStore((s) => s.selectedItemId);
  const updateTimelineItem = useTimelineStore((s) => s.updateItem);
  const timelineTracks = useTimelineStore((s) => s.tracks);
  const timelineMediaLibrary = useTimelineStore((s) => s.mediaLibrary);
  const { recovered } = useTimelinePersistence(jobId, clipId);
  const encoding = useEncodingManager();
  const isEncoding = useMemo(() => {
    if (!jobId || !clipId) return false;
    const exportId = `${jobId}_${parseInt(clipId)}`;
    const task = encoding.tasks[exportId];
    return task?.status === 'encoding';
  }, [jobId, clipId, encoding.tasks]);

  const handleServerExport = useCallback((payload) => {
    if (!jobId || !clipId) return;
    // Share recipients don't have an auth session — route them
    // through the token-gated public export endpoint. ``startExport``
    // accepts ``options.endpoint`` to override the POST target, and
    // ``options.downloadUrlRewriter`` to rewrite the server-emitted
    // download URL so the auto-download fetch doesn't 401 on the
    // normal ``/api/files/...`` path.
    let options;
    if (shareToken) {
      const tokenEnc = encodeURIComponent(shareToken);
      options = {
        endpoint: `/api/share/public/${tokenEnc}/export-clip`,
        downloadUrlRewriter: (url) => {
          // Export broadcast emits ``/api/files/{job_id}/clips/{file}``.
          // Swap to the token-gated download endpoint, preserving any
          // cache-bust query string the backend appended.
          const m = /^\/api\/files\/[^/]+\/(?:clips\/)?([^?]+)(\?.*)?$/.exec(url || '');
          if (!m) return url;
          return `/api/share/public/${tokenEnc}/download/${m[1]}${m[2] || ''}`;
        },
      };
    }
    encoding.startExport(
      jobId,
      parseInt(clipId),
      title || `Clip ${clipId}`,
      payload,
      options,
    );
  }, [jobId, clipId, title, encoding, shareToken]);

  // ── Live QA: validate editor state and auto-fix track compatibility issues ──
  // Runs when tracks/items/selection change (NOT on every playhead frame update)
  useEffect(() => {
    if (!timelineTracks.length || !timelineStoreItems.length) return;
    const storeSegments = useTimelineStore.getState().segments;
    const qa = runEditorQA(timelineTracks, timelineStoreItems, {
      selectedItemId: storeSelectedItemId,
      settings,
      segments: storeSegments,
    });
    if (qa.errors.length > 0) {
      // Auto-fix track compatibility violations
      const fixes = autoFixTrackCompatibility(timelineTracks, timelineStoreItems);
      if (fixes.length > 0) {
        for (const fix of fixes) {
          updateTimelineItem(fix.itemId, { trackId: fix.toTrackId });
        }
        if (process.env.NODE_ENV === 'development') {
          console.warn('[EditorQA] Auto-fixed track violations:', fixes);
        }
      }
    }
    if (process.env.NODE_ENV === 'development' && qa.violations.length > 0) {
      console.warn('[EditorQA]', qa.summary, qa.violations);
    }
  }, [timelineTracks, timelineStoreItems, updateTimelineItem, storeSelectedItemId, settings]);

  // Initialize timeline store when clip data changes
  const addItem = useTimelineStore((s) => s.addItem);
  const removeItem = useTimelineStore((s) => s.removeItem);
  const multiTrackInitialized = useRef(false);
  const lastInitClipEnd = useRef(0);
  useEffect(() => {
    if (!src) return;
    const effectiveEnd = clipEnd > clipStart ? clipEnd : 0;
    // Skip if clipEnd is not yet available (still processing)
    if (effectiveEnd <= 0) return;

    // Re-init if clipEnd becomes available for the first time, or changes significantly
    const needsInit = !multiTrackInitialized.current || (lastInitClipEnd.current === 0 && effectiveEnd > 0);

    // Also re-init if the store's video item doesn't match our clip range
    // (can happen when useTimelinePersistence recovers stale state)
    const videoItem = timelineStoreItems.find(it => it.type === 'video');
    const storeClipMismatch = videoItem &&
        (Math.abs((videoItem.trimStart || 0) - clipStart) > 0.5 ||
         Math.abs((videoItem.trimEnd || 0) - effectiveEnd) > 0.5);

    if (!needsInit && !storeClipMismatch) return;

    if (!recovered || timelineStoreItems.length === 0 || lastInitClipEnd.current === 0 || storeClipMismatch) {
      // Fresh init — populate with transcript subtitles
      initFromClip({ src, clipStart, clipEnd: effectiveEnd, subtitleSegments: transcript || [] });
    } else if (recovered && transcript && transcript.length > 0) {
      // Recovered state but no subtitle items — backfill from transcript
      const hasSubtitles = timelineStoreItems.some((it) => it.type === 'subtitle');
      if (!hasSubtitles) {
        transcript.forEach((seg) => {
          if (seg.end > clipStart && seg.start < effectiveEnd) {
            const s = Math.max(seg.start, clipStart);
            const e = Math.min(seg.end, effectiveEnd);
            addItem({
              trackId: 't1',
              type: 'subtitle',
              mediaRef: null,
              start: s - clipStart,
              end: e - clipStart,
              trimStart: 0,
              trimEnd: null,
              volume: 1.0,
              speed: 1.0,
              opacity: 1.0,
              position: { x: 50, y: 90 },
              size: { w: 100, h: 100 },
              effects: {},
              fadeIn: 0,
              fadeOut: 0,
              subtitleText: seg.text,
              subtitleStyle: null,
              speaker: seg.speaker || null,
            });
          }
        });
      }
    }
    multiTrackInitialized.current = true;
    lastInitClipEnd.current = effectiveEnd;
  }, [src, clipStart, clipEnd, initFromClip, recovered, timelineStoreItems.length, transcript, addItem]);

  // ── Sync: settings.subtitlesEnabled → timeline track visibility ──
  // The settings toggle is the PRIMARY control for subtitle visibility.
  // The timeline track eye icon follows it. This ensures the user always
  // sees consistent behavior regardless of which control they use.
  const toggleTrackVisibility = useTimelineStore((s) => s.toggleTrackVisibility);
  useEffect(() => {
    const subTrack = timelineTracks.find(t => t.type === 'subtitle');
    if (!subTrack) return;

    const trackVisible = subTrack.visible !== false;
    const settingsEnabled = settings?.subtitlesEnabled ?? true;

    // Sync: if settings says ON but track is hidden, show the track
    // If settings says OFF but track is visible, hide the track
    if (settingsEnabled && !trackVisible) {
      toggleTrackVisibility(subTrack.id);
    } else if (!settingsEnabled && trackVisible) {
      toggleTrackVisibility(subTrack.id);
    }
  }, [settings?.subtitlesEnabled, timelineTracks, toggleTrackVisibility]);

  // ── Subtitle sync refs (shared by forward and reverse sync effects) ──
  const subtitleSyncTimerRef = useRef(null);
  const lastSyncedSubtitlesRef = useRef(new Map());

  // ── Reverse sync: when transcript prop changes (e.g. from TranscriptViewer edits),
  //    update matching subtitle items in the timeline store.
  //    Handles: text changes, speaker changes, added segments, deleted segments ──
  const prevTranscriptRef = useRef(transcript);
  useEffect(() => {
    if (!transcript || !multiTrackInitialized.current) return;
    if (prevTranscriptRef.current === transcript) return;
    prevTranscriptRef.current = transcript;

    const effectiveEnd = clipEnd || clipStart;
    const subtitleItems = timelineStoreItems.filter((it) => it.type === 'subtitle');

    // Build map of transcript segments that fall within the clip range
    const clipTranscript = transcript
      .map((seg, idx) => ({ ...seg, _origIdx: idx }))
      .filter((seg) => seg.end > clipStart && seg.start < effectiveEnd);

    // Match existing subtitle items to transcript segments.
    // Prefer stable transcriptIndex; fall back to time proximity.
    const matched = new Set(); // transcript _origIdx values that matched
    subtitleItems.forEach((subItem) => {
      const subAbsStart = (subItem.start || 0) + clipStart;
      const subAbsEnd = (subItem.end || 0) + clipStart;
      let match = null;
      // Try transcriptIndex first
      if (subItem.transcriptIndex != null) {
        match = clipTranscript.find((seg) => seg._origIdx === subItem.transcriptIndex && !matched.has(seg._origIdx));
      }
      // Fall back to time proximity
      if (!match) {
        match = clipTranscript.find((seg) => {
          if (matched.has(seg._origIdx)) return false;
          return Math.abs((seg.start ?? 0) - subAbsStart) < 0.15 &&
                 Math.abs((seg.end ?? 0) - subAbsEnd) < 0.15;
        });
      }
      if (match) {
        matched.add(match._origIdx);
        const updates = {};
        if (match.text !== subItem.subtitleText) updates.subtitleText = match.text;
        if (match.speaker !== subItem.speaker) updates.speaker = match.speaker;
        // Update transcriptIndex if not set
        if (subItem.transcriptIndex == null) updates.transcriptIndex = match._origIdx;
        if (Object.keys(updates).length > 0) {
          updateTimelineItem(subItem.id, updates);
          lastSyncedSubtitlesRef.current.set(subItem.id, match.text);
        }
      }
    });

    // Handle deleted segments: remove timeline items that no longer have a transcript match
    subtitleItems.forEach((subItem) => {
      const subAbsStart = (subItem.start || 0) + clipStart;
      const subAbsEnd = (subItem.end || 0) + clipStart;
      // Check by transcriptIndex first, then time proximity
      let hasMatch = false;
      if (subItem.transcriptIndex != null) {
        hasMatch = clipTranscript.some((seg) => seg._origIdx === subItem.transcriptIndex);
      }
      if (!hasMatch) {
        hasMatch = clipTranscript.some((seg) =>
          Math.abs((seg.start ?? 0) - subAbsStart) < 0.3 &&
          Math.abs((seg.end ?? 0) - subAbsEnd) < 0.3
        );
      }
      if (!hasMatch) {
        removeItem(subItem.id);
      }
    });

    // Handle added segments: create timeline items for transcript segments without a match
    clipTranscript.forEach((seg) => {
      if (matched.has(seg._origIdx)) return;
      // Check if any existing subtitle item matches this segment
      const alreadyExists = subtitleItems.some((subItem) => {
        if (subItem.transcriptIndex === seg._origIdx) return true;
        const subAbsStart = (subItem.start || 0) + clipStart;
        const subAbsEnd = (subItem.end || 0) + clipStart;
        return Math.abs((seg.start ?? 0) - subAbsStart) < 0.3 &&
               Math.abs((seg.end ?? 0) - subAbsEnd) < 0.3;
      });
      if (!alreadyExists) {
        const s = Math.max(seg.start, clipStart);
        const e = Math.min(seg.end, effectiveEnd);
        addItem({
          trackId: 't1',
          type: 'subtitle',
          start: s - clipStart,
          end: e - clipStart,
          position: { x: 50, y: 90 },
          size: { w: 100, h: 100 },
          subtitleText: seg.text,
          speaker: seg.speaker || null,
          transcriptIndex: seg._origIdx,
        });
      }
    });
  }, [transcript, timelineStoreItems, clipStart, clipEnd, updateTimelineItem, removeItem, addItem]);

  const videoRef = useRef(null);
  const containerRef = useRef(null);
  const viewportRef = useRef(null);
  // Stage = the aspect-ratio-correct visible area inside the viewport. In
  // normal mode it perfectly fills the viewport; in lightbox mode the
  // viewport becomes a full-screen black backdrop and the stage is sized
  // to the selected aspect ratio (letterboxed). All overlays/subtitles/
  // interactive handles position themselves relative to this stage so they
  // always sit on the actual video frame, never the letterbox bars.
  const stageRef = useRef(null);
  const viewportClickRef = useRef({ downTime: 0, moved: false });
  const timelineRef = useRef(null);
  const [overlayInteracting, setOverlayInteracting] = useState(false);
  const arrowHoldRef = useRef({ key: null, interval: null });
  const skipTimeRef = useRef(null);
  const waveformCanvasRef = useRef(null);
  const waveformDataRef = useRef(null);
  const audioCtxRef = useRef(null);
  const gainNodeRef = useRef(null);
  const sourceNodeRef = useRef(null);
  const animFrameRef = useRef(null);
  const thumbnailCanvasRef = useRef(null);
  const thumbnailsRef = useRef([]);
  const audioOverlayRefs = useRef({}); // { [itemId]: HTMLAudioElement }

  // ── State ──────────────────────────────────────────
  const [playing, setPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(clipStart);

  // Sync local playing state → timeline store so the Timeline canvas
  // can run its requestAnimationFrame redraw loop during playback.
  useEffect(() => {
    useTimelineStore.getState().setIsPlaying(playing);
  }, [playing]);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [showTimecodeRemaining, setShowTimecodeRemaining] = useState(false);
  const [editingTimecode, setEditingTimecode] = useState(false);
  const [timecodeInput, setTimecodeInput] = useState('');
  const timecodeInputRef = useRef(null);
  const [videoReady, setVideoReady] = useState(false);
  const [videoError, setVideoError] = useState(false);
  // While the backend is still transcoding the browser preview
  // (typical for large 4K / VP9 / MKV uploads) the ``<video>``
  // element fires ``error`` on every retry until the preview is
  // ready. We show a friendlier "Preparing preview..." card during
  // that window instead of the hard "Failed to load video" error,
  // which is reserved for the case where we've exhausted the whole
  // retry budget.
  const [videoPreparing, setVideoPreparing] = useState(false);

  // Volume: 0-200 (percentage)
  const [volume, setVolume] = useState(100);
  const [prevVolume, setPrevVolume] = useState(100);
  const [isMuted, setIsMuted] = useState(false);

  // Speed
  const [speed, setSpeed] = useState(1.0);
  const [showSpeedMenu, setShowSpeedMenu] = useState(false);

  // Trim: offsets from clipStart/clipEnd
  const [trimStartOffset, setTrimStartOffset] = useState(0);
  const [trimEndOffset, setTrimEndOffset] = useState(0);
  const [draggingHandle, setDraggingHandle] = useState(null); // 'left' | 'right' | null
  const [draggingPlayhead, setDraggingPlayhead] = useState(false);
  const [trimApplied, setTrimApplied] = useState(false);
  const [videoDuration, setVideoDuration] = useState(0);

  // Segments: per-region settings overrides
  // Each segment: { id, start, end, volume, muted, subtitlesEnabled, speed }
  // start/end are absolute times (same coordinate system as clipStart/clipEnd)
  const [segments, setSegments] = useState(initialSegments || []);
  const segmentIdRef = useRef(1);
  const [selectedSegmentId, setSelectedSegmentId] = useState(null);

  // J-K-L shuttle control
  const [shuttleSpeed, setShuttleSpeed] = useState(0);

  // Hover time preview
  const [hoverTime, setHoverTime] = useState(null);
  const [hoverX, setHoverX] = useState(0);

  // Segment time editing state — which segment field is being edited
  const [editingSegTime, setEditingSegTime] = useState(null); // { segId, field: 'start'|'end' }
  const [segTimeInput, setSegTimeInput] = useState('');
  const segTimeInputRef = useRef(null);

  // Segment drag state for edge-resize and move
  const [segDrag, setSegDrag] = useState(null); // { segId, mode: 'start'|'end'|'move', startPointerX, origStart, origEnd }
  const [segHoverEdge, setSegHoverEdge] = useState(null); // { segId, edge: 'start'|'end'|'center' }
  const segDragTooltipRef = useRef(null);

  // Segment entry indicator (briefly shows segment info on viewport when entering)
  const [segEntryIndicator, setSegEntryIndicator] = useState(null); // { label, volume, speed, muted }
  const segEntryTimerRef = useRef(null);

  // Segment label editing
  const [editingSegLabel, setEditingSegLabel] = useState(null); // segId
  const [segLabelInput, setSegLabelInput] = useState('');
  const segLabelInputRef = useRef(null);

  // Segment delete confirmation
  const [deleteConfirmId, setDeleteConfirmId] = useState(null);

  // Video item properties from multi-track timeline (opacity, effects, position, size, rotation)
  const [videoItemOpacity, setVideoItemOpacity] = useState(1);
  const [videoItemFilter, setVideoItemFilter] = useState('');
  const [videoItemPosition, setVideoItemPosition] = useState({ x: 50, y: 50 });
  const [videoItemSize, setVideoItemSize] = useState({ w: 100, h: 100 });
  const [videoItemRotation, setVideoItemRotation] = useState(0);

  // Check if the video track is hidden (for preview visibility)
  const videoTrackHidden = useMemo(() => {
    const videoTrack = timelineTracks.find((t) => t.type === 'video');
    return videoTrack?.visible === false;
  }, [timelineTracks]);

  // Check if audio tracks are muted (for preview audio)
  const audioTrackMuted = useMemo(() => {
    const audioTracks = timelineTracks.filter((t) => t.type === 'audio');
    return audioTracks.length > 0 && audioTracks.every((t) => t.muted === true);
  }, [timelineTracks]);

  // Sync video timeline item properties → actual video element
  // Find the video item at the current playhead (not just the first one)
  // so split segments with different effects render correctly.
  const relativePlayhead = currentTime - clipStart;
  const videoTimelineItem = useMemo(() => {
    const atPlayhead = timelineStoreItems.find(
      (it) => it.type === 'video' && relativePlayhead >= it.start && relativePlayhead < it.end
    );
    return atPlayhead || timelineStoreItems.find((it) => it.type === 'video') || null;
  }, [timelineStoreItems, relativePlayhead]);
  useEffect(() => {
    if (!videoTimelineItem) return;
    // Volume
    if (videoTimelineItem.volume != null) {
      const vol = Math.round(Math.max(0, Math.min(2, videoTimelineItem.volume)) * 100);
      if (Math.abs(vol - volume) > 1) {
        setVolume(vol);
        if (vol === 0) setIsMuted(true);
        else if (isMuted && vol > 0) setIsMuted(false);
      }
    }
    // Speed
    if (videoTimelineItem.speed != null && Math.abs(videoTimelineItem.speed - speed) > 0.001) {
      setSpeed(videoTimelineItem.speed);
      if (videoRef.current) videoRef.current.playbackRate = videoTimelineItem.speed;
    }
    // Opacity
    setVideoItemOpacity(videoTimelineItem.opacity ?? 1);
    // Position, Size, Rotation
    const vPos = videoTimelineItem.position || { x: 50, y: 50 };
    // Treat legacy {x:0,y:0} as centered
    setVideoItemPosition(vPos.x === 0 && vPos.y === 0 ? { x: 50, y: 50 } : vPos);
    setVideoItemSize(videoTimelineItem.size || { w: 100, h: 100 });
    setVideoItemRotation(videoTimelineItem.transform?.rotation || 0);
    // Effects → CSS filter
    const eff = videoTimelineItem.effects || {};
    const filters = [];
    if (eff.brightness) filters.push(`brightness(${1 + eff.brightness / 100})`);
    if (eff.contrast) filters.push(`contrast(${1 + eff.contrast / 100})`);
    if (eff.saturation) filters.push(`saturate(${1 + eff.saturation / 100})`);
    if (eff.blur) filters.push(`blur(${eff.blur}px)`);
    if (eff.hueRotate) filters.push(`hue-rotate(${eff.hueRotate}deg)`);
    if (eff.sepia) filters.push(`sepia(${eff.sepia / 100})`);
    setVideoItemFilter(filters.length ? filters.join(' ') : '');
  }, [videoTimelineItem]);

  // ── Sync a1 audio item → global volume/speed ────────
  // The a1 audio track item controls the same audio stream as the video.
  // When the user edits volume/speed on the a1 audio item via the Properties
  // panel, those changes must override the global volume/speed controls.
  const a1AudioItem = useMemo(() => {
    return timelineStoreItems.find(it => it.type === 'audio' && it.trackId === 'a1') || null;
  }, [timelineStoreItems]);

  useEffect(() => {
    if (!a1AudioItem) return;
    // a1 audio volume (0-2 scale) → global volume (0-200 scale)
    if (a1AudioItem.volume != null) {
      const vol = Math.round(Math.max(0, Math.min(2, a1AudioItem.volume)) * 100);
      if (Math.abs(vol - volume) > 1) {
        setVolume(vol);
        if (vol === 0) setIsMuted(true);
        else if (isMuted && vol > 0) setIsMuted(false);
      }
    }
    // a1 audio speed → global speed
    if (a1AudioItem.speed != null && Math.abs(a1AudioItem.speed - speed) > 0.001) {
      setSpeed(a1AudioItem.speed);
      if (videoRef.current) videoRef.current.playbackRate = a1AudioItem.speed;
    }
  }, [a1AudioItem]);

  // Reverse sync: write VideoEditor volume/speed back to the video timeline item
  // AND the a1 audio item so the PropertiesPanel stays in sync with the playback controls.
  useEffect(() => {
    if (!videoTimelineItem) return;
    const itemVol = videoTimelineItem.volume ?? 1;
    const editorVol = volume / 100;
    if (Math.abs(itemVol - editorVol) > 0.02) {
      updateTimelineItem(videoTimelineItem.id, { volume: editorVol });
    }
    // Also sync to a1 audio item
    if (a1AudioItem && a1AudioItem.id !== videoTimelineItem.id) {
      const a1Vol = a1AudioItem.volume ?? 1;
      if (Math.abs(a1Vol - editorVol) > 0.02) {
        updateTimelineItem(a1AudioItem.id, { volume: editorVol });
      }
    }
  }, [volume, videoTimelineItem?.id, a1AudioItem?.id]);

  useEffect(() => {
    if (!videoTimelineItem) return;
    const itemSpd = videoTimelineItem.speed ?? 1;
    if (Math.abs(itemSpd - speed) > 0.001) {
      updateTimelineItem(videoTimelineItem.id, { speed });
    }
    // Also sync to a1 audio item
    if (a1AudioItem && a1AudioItem.id !== videoTimelineItem.id) {
      const a1Spd = a1AudioItem.speed ?? 1;
      if (Math.abs(a1Spd - speed) > 0.001) {
        updateTimelineItem(a1AudioItem.id, { speed });
      }
    }
  }, [speed, videoTimelineItem?.id, a1AudioItem?.id]);

  // Auto-open properties panel when an item is selected (via viewport or timeline)
  useEffect(() => {
    if (storeSelectedItemId && showMultiTrack) {
      setShowProperties(true);
    }
  }, [storeSelectedItemId, showMultiTrack]);

  // ── Forward sync: subtitle edits in timeline store → backend transcript API ──
  // Handles text, speaker, and timing changes. Debounced to 800ms.
  // Uses transcriptIndex for stable matching (survives timing changes).
  // Runs regardless of showMultiTrack (SubtitleOverlay edits also sync).
  useEffect(() => {
    if (!jobId || !transcript) return;

    const subtitleItems = timelineStoreItems.filter((it) => it.type === 'subtitle');
    if (subtitleItems.length === 0) return;

    // Find subtitle items whose text, speaker, or timing differs from transcript
    const pendingUpdates = [];
    subtitleItems.forEach((subItem) => {
      const subAbsStart = (subItem.start || 0) + clipStart;
      const subAbsEnd = (subItem.end || 0) + clipStart;

      // Prefer stable transcriptIndex for matching; fall back to time proximity
      let matchIdx = -1;
      if (subItem.transcriptIndex != null && subItem.transcriptIndex < transcript.length) {
        matchIdx = subItem.transcriptIndex;
      } else {
        matchIdx = transcript.findIndex((seg) => {
          const tStart = seg.start ?? 0;
          const tEnd = seg.end ?? 0;
          return Math.abs(tStart - subAbsStart) < 0.15 && Math.abs(tEnd - subAbsEnd) < 0.15;
        });
      }
      if (matchIdx < 0) return;

      const seg = transcript[matchIdx];
      const timelineText = subItem.subtitleText || '';
      const lastSynced = lastSyncedSubtitlesRef.current.get(subItem.id);
      const body = {};

      // Text change
      if (timelineText !== (seg.text || '') && timelineText !== lastSynced) {
        body.text = timelineText;
      }
      // Speaker change
      if (subItem.speaker && subItem.speaker !== seg.speaker) {
        body.speaker = subItem.speaker;
      }
      // Timing change — convert clip-relative back to absolute
      if (Math.abs(subAbsStart - (seg.start ?? 0)) > 0.05) {
        body.start = subAbsStart;
      }
      if (Math.abs(subAbsEnd - (seg.end ?? 0)) > 0.05) {
        body.end = subAbsEnd;
      }

      if (Object.keys(body).length > 0) {
        pendingUpdates.push({ itemId: subItem.id, segmentIndex: matchIdx, body });
      }
    });

    if (pendingUpdates.length === 0) return;

    if (subtitleSyncTimerRef.current) clearTimeout(subtitleSyncTimerRef.current);
    subtitleSyncTimerRef.current = setTimeout(async () => {
      for (const upd of pendingUpdates) {
        try {
          const res = await fetch(`/api/jobs/${jobId}/transcript/${upd.segmentIndex}`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(upd.body),
          });
          if (res.ok) {
            if (upd.body.text != null) {
              lastSyncedSubtitlesRef.current.set(upd.itemId, upd.body.text);
            }
          }
        } catch (err) {
          console.error('Subtitle sync to transcript failed:', err);
        }
      }
      if (onTranscriptUpdated) onTranscriptUpdated();
    }, 800);

    return () => {
      if (subtitleSyncTimerRef.current) clearTimeout(subtitleSyncTimerRef.current);
    };
  }, [timelineStoreItems, jobId, transcript, clipStart, onTranscriptUpdated]);

  // Derived: the currently selected segment object (or null)
  const selectedSegment = useMemo(
    () => segments.find(s => s.id === selectedSegmentId) || null,
    [segments, selectedSegmentId],
  );

  // Active segment: the segment the playhead is currently inside (for live UI feedback)
  const [activeSegmentId, setActiveSegmentId] = useState(null);
  const activeSegment = useMemo(
    () => segments.find(s => s.id === activeSegmentId) || null,
    [segments, activeSegmentId],
  );

  // controlTargetSegment — The segment that controls (volume, speed, mute)
  // modify.  Falls back to the segment the playhead is currently inside so
  // that changes match what the user sees in the control bar.  When no
  // segment is selected AND the playhead is outside all segments, this is
  // null and changes go to the global clip settings.
  const controlTargetSegment = selectedSegment || activeSegment;

  // displaySegment — What the controls DISPLAY.
  // Shows selected segment's values if selected, else active segment's values.
  const displaySegment = controlTargetSegment;

  // Backward compat alias — used in controls bar class
  const effectiveSegment = displaySegment;

  // Derived — use video's natural duration as fallback when clipEnd is 0
  const effectiveClipEnd = clipEnd > clipStart ? clipEnd : (videoDuration || clipEnd);
  const clipDur = effectiveClipEnd - clipStart;
  const trimmedStart = clipStart + trimStartOffset;
  const trimmedEnd = effectiveClipEnd - trimEndOffset;
  const trimmedDur = trimmedEnd - trimmedStart;
  const elapsed = Math.max(0, Math.min(trimmedDur, currentTime - trimmedStart));

  // ── Speed-adjusted duration: accounts for per-segment speed overrides ──
  // Builds a timeline of (start, end, speed) covering the trimmed range,
  // then sums each region's wall-clock duration = media_duration / speed.
  const { effectiveDuration, mediaToWallClock } = useMemo(() => {
    // Collect segments that overlap the trimmed range, sorted by start
    const segs = segments
      .filter(s => s.end > trimmedStart && s.start < trimmedEnd)
      .sort((a, b) => a.start - b.start);

    // Build timeline entries: [{start, end, speed}] in trimmed-relative coords
    const timeline = [];
    let pos = 0; // clip-relative position
    for (const seg of segs) {
      const segStart = Math.max(0, seg.start - trimmedStart);
      const segEnd = Math.min(trimmedDur, seg.end - trimmedStart);
      if (segEnd <= segStart) continue;
      // Gap before segment
      if (segStart > pos + 0.001) {
        timeline.push({ start: pos, end: segStart, speed });
      }
      timeline.push({ start: segStart, end: segEnd, speed: seg.speed || 1.0 });
      pos = segEnd;
    }
    // Gap after last segment
    if (pos < trimmedDur - 0.001) {
      timeline.push({ start: pos, end: trimmedDur, speed });
    }
    // If no segments, the whole range uses global speed
    if (timeline.length === 0) {
      timeline.push({ start: 0, end: trimmedDur, speed });
    }

    // Total effective (wall-clock) duration
    let total = 0;
    for (const t of timeline) {
      total += (t.end - t.start) / t.speed;
    }

    // Map media elapsed time → wall-clock elapsed time
    const mapFn = (mediaElapsed) => {
      let wallClock = 0;
      for (const t of timeline) {
        if (mediaElapsed <= t.start) break;
        const consumed = Math.min(mediaElapsed, t.end) - t.start;
        wallClock += consumed / t.speed;
        if (mediaElapsed <= t.end) break;
      }
      return wallClock;
    };

    return { effectiveDuration: total, mediaToWallClock: mapFn };
  }, [segments, trimmedStart, trimmedEnd, trimmedDur, speed]);

  // ── Aspect ratio & subject tracking ─────────────────
  const srcRatio = sourceWidth / sourceHeight;
  const targetRatio = useMemo(() => {
    if (aspectRatio && ASPECT_RATIO_VALUES[aspectRatio]) return ASPECT_RATIO_VALUES[aspectRatio];
    return srcRatio;
  }, [aspectRatio, srcRatio]);

  const isCrop = useMemo(() => {
    if (!aspectRatio || !ASPECT_RATIO_VALUES[aspectRatio]) return false;
    return Math.abs(srcRatio - ASPECT_RATIO_VALUES[aspectRatio]) > 0.01;
  }, [aspectRatio, srcRatio]);

  const subjectKeyframes = useMemo(() => {
    if (clipStart == null || clipEnd == null) return null;

    // PRIMARY: dense per-frame subject_track from the pipeline. Covers
    // every frame at ~2 Hz via the face → saliency → object → scene
    // cascade, so even anime / action sequences with off-center
    // subjects follow the action correctly. This is the fix for the
    // "9:16 window points at empty space for half a second" artifact.
    if (Array.isArray(subjectTrack) && subjectTrack.length > 0) {
      const dense = buildKeyframesFromSubjectTrack(
        subjectTrack, clipStart, clipEnd,
        isCrop ? srcRatio : null, isCrop ? targetRatio : null,
      );
      if (dense && dense.length > 0) {
        const dynamic = isDynamic(dense);
        const sourceCounts = {};
        subjectTrack.forEach(p => { sourceCounts[p.source || '?'] = (sourceCounts[p.source || '?'] || 0) + 1; });
        console.log(
          `[SubjectTracking] VideoEditor(dense): ${dense.length} keyframes from ${subjectTrack.length} track samples, ` +
          `isCrop=${isCrop}, dynamic=${dynamic}, sources=${JSON.stringify(sourceCounts)}, aspect=${aspectRatio}`
        );
        return dense;
      }
    }

    // FALLBACK: sparse VLM-scene keyframes for legacy jobs analyzed
    // before the dense subject_track landed on the pipeline.
    if (!scenes?.length) return null;
    const result = processKeyframes(scenes, clipStart, clipEnd, isCrop ? srcRatio : null, isCrop ? targetRatio : null, transcript || null, sceneCuts || null);
    const dynamic = result && isDynamic(result);
    console.log(
      `[SubjectTracking] VideoEditor(sparse): ${result?.length || 0} keyframes, ` +
      `isCrop=${isCrop}, dynamic=${dynamic}, ` +
      `unique_x=[${result ? [...new Set(result.map(k=>k.x))].sort((a,b)=>a-b).join(',') : ''}], ` +
      `scenes=${scenes.length}, aspect=${aspectRatio}`
    );
    return result;
  }, [subjectTrack, scenes, clipStart, clipEnd, isCrop, srcRatio, targetRatio, transcript, sceneCuts, aspectRatio]);

  // Expose keyframes to parent for export parity
  useEffect(() => {
    if (onSubjectKeyframes) onSubjectKeyframes(subjectKeyframes);
  }, [subjectKeyframes]); // eslint-disable-line react-hooks/exhaustive-deps

  // Populate crop segments on timeline when keyframes change
  useEffect(() => {
    console.log('[CropTrack]', {
      isCrop, aspectRatio, srcRatio, targetRatio,
      hasKeyframes: !!subjectKeyframes?.length,
      kfCount: subjectKeyframes?.length || 0,
      dur: clipEnd - clipStart,
    });
    const setCropSegments = useTimelineStore.getState().setCropSegments;
    if (!isCrop) {
      setCropSegments([]);
      return;
    }
    const dur = clipEnd - clipStart;
    if (!subjectKeyframes?.length) {
      // No scene data yet — generate a default center-crop segment so the crop
      // track isn't empty while the user waits for subject tracking to complete.
      setCropSegments([{
        id: 'crop-default',
        startTime: 0,
        endTime: dur,
        cropX: 50,
        originalCropX: 50,
        clusterId: 0,
        isManualOverride: false,
        label: 'Center',
      }]);
      return;
    }
    const clusters = detectPositionClusters(subjectKeyframes);
    // Prefer backend slot identity when available from render plan
    const slotTimeline = renderPlan ? extractSlotTimelineFromRenderPlan(renderPlan) : null;
    const segments = slotTimeline?.length
      ? keyframesToCropSegmentsWithSlots(subjectKeyframes, dur, slotTimeline, speakerNames || {})
      : keyframesToCropSegments(subjectKeyframes, dur, clusters);
    setCropSegments(segments);
  }, [subjectKeyframes, isCrop, clipStart, clipEnd, renderPlan, speakerNames]);

  const hasDynamicSubject = useMemo(
    () => isCrop && subjectKeyframes && isDynamic(subjectKeyframes),
    [isCrop, subjectKeyframes],
  );

  // Tracking status for user feedback
  const trackingStatus = useMemo(() => {
    if (!isCrop) return null;
    if (!scenes?.length) return { mode: 'no-data', label: 'No AI data', color: '#f59e0b' };
    if (!subjectKeyframes?.length) return { mode: 'error', label: 'Tracking failed', color: '#ef4444' };
    if (hasDynamicSubject) {
      const raw = buildSubjectKeyframes(scenes, clipStart, clipEnd,
        isCrop ? srcRatio : null, isCrop ? targetRatio : null);
      const clusters = detectPositionClusters(raw);
      if (clusters && clusters.length >= 2) {
        const positions = clusters.map(c => `${c.center}%`).join(' \u2194 ');
        return {
          mode: 'multi',
          label: `Face tracked \u00b7 ${clusters.length} speakers (${positions})`,
          color: '#10b981',
        };
      }
      return {
        mode: 'dynamic',
        label: `Face tracked \u00b7 ${subjectKeyframes.length} keyframes`,
        color: '#10b981',
      };
    }
    const sx = subjectKeyframes[0].x;
    if (Math.abs(sx - 50) < 3) {
      return { mode: 'center', label: 'Centered', color: '#6b7280' };
    }
    return { mode: 'static', label: `Face tracked at ${sx}%`, color: '#10b981' };
  }, [isCrop, scenes, subjectKeyframes, hasDynamicSubject, clipStart, clipEnd, srcRatio, targetRatio]);

  // Live face position — updates during playback via requestAnimationFrame
  const [livePosition, setLivePosition] = useState(null);
  useEffect(() => {
    if (!hasDynamicSubject || !isCrop) { setLivePosition(null); return; }
    const video = videoRef.current;
    if (!video) return;
    let animId;
    let lastPos = null;
    const tick = () => {
      if (!video.paused) {
        const relTime = video.currentTime - (clipStart || 0);
        const sx = interpolateSubjectX(subjectKeyframes, relTime);
        const rounded = Math.round(sx);
        if (rounded !== lastPos) { lastPos = rounded; setLivePosition(rounded); }
      }
      animId = requestAnimationFrame(tick);
    };
    animId = requestAnimationFrame(tick);
    return () => { cancelAnimationFrame(animId); setLivePosition(null); };
  }, [hasDynamicSubject, isCrop, subjectKeyframes, clipStart]);

  // ── Segment helpers ────────────────────────────────
  const getActiveSegment = useCallback((t) => {
    return segments.find(s => t >= s.start && t < s.end) || null;
  }, [segments]);

  const addSegment = useCallback(() => {
    const t = videoRef.current?.currentTime ?? currentTime;
    const halfDur = 2; // +/- 2 seconds around playhead
    const segStart = Math.max(trimmedStart, t - halfDur);
    const segEnd = Math.min(trimmedEnd, t + halfDur);
    if (segEnd - segStart < 0.5) return; // too short

    // Check for overlap with existing segments
    const overlaps = segments.some(s => segStart < s.end && segEnd > s.start);
    if (overlaps) return;

    const newSeg = {
      id: `seg_${segmentIdRef.current++}`,
      start: segStart,
      end: segEnd,
      volume: 100,
      muted: false,
      subtitlesEnabled: true,
      subjectTrackingEnabled: true,
      speed: 1.0,
      color: SEGMENT_COLORS[segments.length % SEGMENT_COLORS.length],
      label: `Segment ${segments.length + 1}`,
    };
    const next = [...segments, newSeg].sort((a, b) => a.start - b.start);
    setSegments(next);
    onSegmentsChange?.(next);
    setSelectedSegmentId(newSeg.id);
  }, [currentTime, trimmedStart, trimmedEnd, segments, onSegmentsChange]);

  const removeSegment = useCallback((segId) => {
    const next = segments.filter(s => s.id !== segId);
    setSegments(next);
    onSegmentsChange?.(next);
  }, [segments, onSegmentsChange]);

  const toggleSegmentMute = useCallback((segId) => {
    const next = segments.map(s => s.id === segId ? { ...s, muted: !s.muted, volume: s.muted ? 100 : 0 } : s);
    setSegments(next);
    onSegmentsChange?.(next);
  }, [segments, onSegmentsChange]);

  const toggleSegmentSubs = useCallback((segId) => {
    const next = segments.map(s => s.id === segId ? { ...s, subtitlesEnabled: !s.subtitlesEnabled } : s);
    setSegments(next);
    onSegmentsChange?.(next);
  }, [segments, onSegmentsChange]);

  const toggleSegmentTracking = useCallback((segId) => {
    const next = segments.map(s => s.id === segId ? { ...s, subjectTrackingEnabled: !(s.subjectTrackingEnabled !== false) } : s);
    setSegments(next);
    onSegmentsChange?.(next);
  }, [segments, onSegmentsChange]);

  const updateSegment = useCallback((segId, updates) => {
    const next = segments.map(s => s.id === segId ? { ...s, ...updates } : s);
    setSegments(next);
    onSegmentsChange?.(next);
  }, [segments, onSegmentsChange]);

  const deselectSegment = useCallback(() => {
    setSelectedSegmentId(null);
  }, []);

  // Commit an inline segment time edit
  const commitSegTimeEdit = useCallback(() => {
    if (!editingSegTime) return;
    const parsed = parseTimecodeInput(segTimeInput);
    if (parsed == null) { setEditingSegTime(null); return; }
    const absTime = clipStart + parsed; // user types relative time
    const seg = segments.find(s => s.id === editingSegTime.segId);
    if (!seg) { setEditingSegTime(null); return; }
    const field = editingSegTime.field;
    let newStart = field === 'start' ? absTime : seg.start;
    let newEnd = field === 'end' ? absTime : seg.end;
    // Validate: clamp to clip range, ensure min 0.5s duration
    newStart = Math.max(clipStart, Math.min(effectiveClipEnd - 0.5, newStart));
    newEnd = Math.max(clipStart + 0.5, Math.min(effectiveClipEnd, newEnd));
    if (newEnd - newStart < 0.5) {
      if (field === 'start') newStart = newEnd - 0.5;
      else newEnd = newStart + 0.5;
    }
    updateSegment(editingSegTime.segId, { start: newStart, end: newEnd });
    setEditingSegTime(null);
  }, [editingSegTime, segTimeInput, clipStart, effectiveClipEnd, segments, updateSegment]);

  // ── Reset on new clip ──────────────────────────────
  useEffect(() => {
    const vol = (initialVolume != null && initialVolume >= 0) ? initialVolume : 100;
    const spd = (initialSpeed != null && initialSpeed > 0) ? initialSpeed : 1.0;
    setTrimStartOffset(0);
    setTrimEndOffset(0);
    setSegments(initialSegments || []);
    setVolume(vol);
    setPrevVolume(vol);
    setIsMuted(false);
    setSpeed(spd);
    setShowSpeedMenu(false);
    setVideoError(false);
    if (videoRef.current) {
      videoRef.current.playbackRate = spd;
    }
    onTrimChange?.({ trimStart: 0, trimEnd: 0 });
    onVolumeChange?.(vol / 100);
    onSpeedChange?.(spd);
  }, [clipStart, clipEnd, src]);

  // ── Sync segments from parent (e.g. loaded from localStorage after mount) ──
  useEffect(() => {
    if (initialSegments && initialSegments.length > 0) {
      // Ensure backward compatibility — add color/label if missing
      const migrated = initialSegments.map((s, i) => ({
        ...s,
        color: s.color || SEGMENT_COLORS[i % SEGMENT_COLORS.length],
        label: s.label || `Segment ${i + 1}`,
      }));
      setSegments(migrated);
      // Ensure segment IDs don't collide with new segments
      const maxId = migrated.reduce((mx, s) => {
        const num = parseInt(String(s.id).replace(/\D/g, ''), 10);
        return isNaN(num) ? mx : Math.max(mx, num);
      }, 0);
      segmentIdRef.current = maxId + 1;
    }
  }, [initialSegments]);

  useEffect(() => {
    useTimelineStore.getState().setSegments(
      segments.map((s) => ({
        ...s,
        start: s.start != null ? s.start - clipStart : 0,
        end: s.end != null ? s.end - clipStart : 0,
      }))
    );
  }, [segments, clipStart]);

  // ── Sync volume/speed from settings panel ──────────
  // These ONLY apply to global state — never route to segments.
  // Segment settings should only change via direct user interaction.
  useEffect(() => {
    if (initialVolume == null || initialVolume < 0) return;
    setVolume(initialVolume);
    setPrevVolume(initialVolume);
    if (initialVolume === 0) setIsMuted(true);
    else setIsMuted(false);
  }, [initialVolume]);

  useEffect(() => {
    if (initialSpeed == null || initialSpeed <= 0) return;
    setSpeed(initialSpeed);
    if (videoRef.current) {
      videoRef.current.playbackRate = initialSpeed;
    }
  }, [initialSpeed]);

  // ── Expose video element to parent via callback ────
  useEffect(() => {
    if (onVideoRef) onVideoRef(videoRef.current);
    return () => { if (onVideoRef) onVideoRef(null); };
  }, [onVideoRef]);

  // ── Video metadata & error ─────────────────────────
  // Retry state lives outside the effect so the retry attempt count
  // survives the canplay → error ping-pong that happens while the
  // backend is still transcoding a browser-friendly preview of the
  // source (big/odd-codec uploads take 30-60s to transcode, and the
  // analysis page typically opens *during* that window).
  const retryCountRef = useRef(0);
  const retryTimerRef = useRef(null);
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    // Reset retry bookkeeping whenever the src changes — a new src
    // deserves its own fresh retry budget.
    retryCountRef.current = 0;
    if (retryTimerRef.current) {
      clearTimeout(retryTimerRef.current);
      retryTimerRef.current = null;
    }
    setVideoPreparing(false);
    const onMetadata = () => {
      if (video.duration && isFinite(video.duration)) {
        setVideoDuration(video.duration);
      }
      // Seek to clipStart so the first frame is visible immediately
      if (video.currentTime === 0 && clipStart > 0) {
        video.currentTime = clipStart;
      }
    };
    const onCanPlay = () => {
      setVideoReady(true);
      setVideoPreparing(false);
      // A successful canplay means the source is good now — zero
      // out the retry counter so a later transient hiccup gets the
      // full retry budget again.
      retryCountRef.current = 0;
      if (video.duration && isFinite(video.duration)) {
        setVideoDuration(video.duration);
      }
    };
    const onError = () => {
      // The browser-preview transcode on the backend is asynchronous
      // and can take several minutes on large 4K / VP9 / MKV sources.
      // While it's in flight, ``/api/files/.../video.*`` returns 503
      // (Retry-After: 5) so the ``<video>`` element fires ``error``
      // on every attempt until the preview lands. We retry with
      // exponential backoff for long enough to cover a realistic
      // transcode of a multi-hundred-MB source — ~10 minutes total —
      // and surface a friendly "Preparing preview..." card during
      // that window instead of the hard error.
      //
      // Schedule: 0.5s, 1s, 2s, 4s, 8s, then 10s steady. 60 tries ≈
      // 10 minutes. After that the hard "Failed to load video" card
      // shows and the user can hit Retry manually.
      const attempt = retryCountRef.current;
      const MAX_ATTEMPTS = 60;
      if (attempt >= MAX_ATTEMPTS) {
        setVideoError(true);
        setVideoPreparing(false);
        return;
      }
      // After the first couple of instant retries, show the
      // "Preparing preview..." indicator so the user knows it's
      // still working rather than staring at a blank viewport.
      if (attempt >= 2) setVideoPreparing(true);
      const delays = [500, 1000, 2000, 4000, 8000];
      const delay = attempt < delays.length ? delays[attempt] : 10000;
      retryCountRef.current = attempt + 1;
      if (retryTimerRef.current) clearTimeout(retryTimerRef.current);
      retryTimerRef.current = setTimeout(() => {
        retryTimerRef.current = null;
        if (videoRef.current) {
          try { videoRef.current.load(); } catch { /* element gone */ }
        }
      }, delay);
    };
    const onDuration = () => {
      if (video.duration && isFinite(video.duration)) {
        setVideoDuration(video.duration);
      }
    };
    video.addEventListener('loadedmetadata', onMetadata);
    video.addEventListener('canplay', onCanPlay);
    video.addEventListener('durationchange', onDuration);
    video.addEventListener('error', onError);
    // If already past canplay when effect runs (e.g. cached)
    if (video.readyState >= 3) {
      setVideoReady(true);
      if (video.duration && isFinite(video.duration)) {
        setVideoDuration(video.duration);
      }
    } else if (video.readyState >= 1) {
      // Have metadata but not yet playable
      if (video.duration && isFinite(video.duration)) {
        setVideoDuration(video.duration);
      }
    }
    return () => {
      video.removeEventListener('loadedmetadata', onMetadata);
      video.removeEventListener('canplay', onCanPlay);
      video.removeEventListener('durationchange', onDuration);
      video.removeEventListener('error', onError);
      if (retryTimerRef.current) {
        clearTimeout(retryTimerRef.current);
        retryTimerRef.current = null;
      }
    };
  }, [src, clipStart]);

  // ── Auto-seek and auto-play on clip change ─────────
  useEffect(() => {
    const video = videoRef.current;
    if (!video || clipStart === undefined || clipStart === null) return;
    video.currentTime = clipStart;
    setCurrentTime(clipStart);
    // Only auto-play if video is ready (has enough data to begin playback)
    if (video.readyState >= 3) {
      video.play().then(() => setPlaying(true)).catch(() => {});
    }
  }, [clipStart, clipEnd]);

  // ── Lightbox (fullscreen) escape + body scroll lock ───
  // The fullscreen button opens a custom in-app lightbox instead of using the
  // browser's Fullscreen API. This lets us pin the video to the *selected*
  // aspect ratio (with letterboxing) and guarantee no editor chrome covers
  // the frame — the browser API would leave the entire ve-container visible
  // including the toolbar and timeline.
  useEffect(() => {
    if (!isFullscreen) return;
    const onKey = (e) => {
      if (e.key === 'Escape') {
        e.preventDefault();
        setIsFullscreen(false);
      }
    };
    window.addEventListener('keydown', onKey);
    const prevOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    return () => {
      window.removeEventListener('keydown', onKey);
      document.body.style.overflow = prevOverflow;
    };
  }, [isFullscreen]);

  // ── Waveform generation ────────────────────────────
  useEffect(() => {
    if (!src) return;
    let cancelled = false;
    const generateWaveform = async () => {
      try {
        // For large files, fetching the entire video into memory fails.
        // Use a Range request to fetch only the first 10MB — enough for
        // the audio codec headers and representative samples.
        const MAX_BYTES = 10 * 1024 * 1024;
        const response = await fetch(src, {
          headers: { Range: `bytes=0-${MAX_BYTES - 1}` },
        });
        if (cancelled) return;
        const arrayBuffer = await response.arrayBuffer();
        if (cancelled) return;
        const offlineCtx = new (window.OfflineAudioContext || window.webkitOfflineAudioContext)(1, 1, 44100);
        const audioBuffer = await offlineCtx.decodeAudioData(arrayBuffer);
        if (cancelled) return;
        const rawData = audioBuffer.getChannelData(0);
        // Downsample to ~200 bars
        const barCount = 200;
        const samplesPerBar = Math.floor(rawData.length / barCount);
        const bars = [];
        for (let i = 0; i < barCount; i++) {
          let sum = 0;
          const start = i * samplesPerBar;
          for (let j = start; j < start + samplesPerBar && j < rawData.length; j++) {
            sum += Math.abs(rawData[j]);
          }
          bars.push(sum / samplesPerBar);
        }
        // Normalize
        const max = Math.max(...bars, 0.01);
        waveformDataRef.current = bars.map(v => v / max);
        drawWaveform();
      } catch {
        // Range request or decode failed — generate a flat placeholder waveform
        // so the track isn't just a black bar.
        if (!cancelled) {
          const barCount = 200;
          const bars = [];
          for (let i = 0; i < barCount; i++) {
            // Create a subtle random pattern so it looks like a waveform
            bars.push(0.15 + Math.random() * 0.25);
          }
          waveformDataRef.current = bars;
          drawWaveform();
        }
      }
    };
    generateWaveform();
    return () => { cancelled = true; };
  }, [src]);

  // Live ref so ``drawWaveform`` can read the current playhead WITHOUT
  // listing ``currentTime`` in its dep array. Previously the waveform
  // useCallback rebuilt every frame, which made the ``[drawWaveform]``
  // dep effect fire every frame, repainting the entire canvas at
  // ~60 Hz during playback.
  const currentTimeRef = useRef(currentTime);
  currentTimeRef.current = currentTime;

  // Backing-store sizing is only valid for one DPR + size combo; track
  // it so we don't reallocate the canvas every paint.
  const lastSizeRef = useRef({ w: 0, h: 0, dpr: 0 });

  // Draw waveform on canvas
  const drawWaveform = useCallback(() => {
    const canvas = waveformCanvasRef.current;
    const data = waveformDataRef.current;
    if (!canvas || !data) return;
    const ctx = canvas.getContext && canvas.getContext('2d');
    if (!ctx) return; // Guard against null context (private mode / unusual envs).
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    const sizeChanged =
      lastSizeRef.current.w !== rect.width ||
      lastSizeRef.current.h !== rect.height ||
      lastSizeRef.current.dpr !== dpr;
    if (sizeChanged) {
      canvas.width = rect.width * dpr;
      canvas.height = rect.height * dpr;
      ctx.scale(dpr, dpr);
      lastSizeRef.current = { w: rect.width, h: rect.height, dpr };
    }
    ctx.clearRect(0, 0, rect.width, rect.height);

    const isDark = document.documentElement.dataset.theme === 'dark';
    const barWidth = rect.width / data.length;
    const midY = rect.height / 2;

    const ct = currentTimeRef.current;
    const leftPct = trimStartOffset / (clipDur || 1);
    const rightPct = 1 - trimEndOffset / (clipDur || 1);
    const playPct = clipDur > 0 ? (ct - clipStart) / clipDur : 0;

    for (let i = 0; i < data.length; i++) {
      const x = i * barWidth;
      const barH = Math.max(1, data[i] * midY * 0.9);
      const pct = i / data.length;

      if (pct < leftPct || pct > rightPct) {
        ctx.fillStyle = isDark ? 'rgba(255, 255, 255, 0.12)' : 'rgba(0, 0, 0, 0.10)';
      } else if (pct <= playPct) {
        ctx.fillStyle = 'rgba(10, 132, 255, 0.9)';
      } else {
        ctx.fillStyle = isDark ? 'rgba(255, 255, 255, 0.5)' : 'rgba(0, 0, 0, 0.4)';
      }
      ctx.fillRect(x, midY - barH, barWidth - 0.5, barH * 2);
    }
  }, [trimStartOffset, trimEndOffset, clipDur, clipStart]);

  // Redraw waveform on state changes that actually invalidate the
  // pixels we just drew (trim handles, clip range). currentTime is
  // intentionally absent — playhead progress is repainted by the
  // throttled effect below to avoid full-canvas redraws at 60 Hz.
  useEffect(() => {
    drawWaveform();
  }, [drawWaveform]);

  // Throttled playhead repaint: only ~12 Hz (every ~80 ms). Painting
  // a ~1500-bar canvas at 60 Hz was a major frame-time hit during
  // playback. ``playing`` toggles the rAF loop on/off so paused
  // editors do zero work.
  useEffect(() => {
    if (!playing) {
      drawWaveform();
      return undefined;
    }
    let rafId;
    let lastPaint = 0;
    const PAINT_INTERVAL_MS = 80;
    const tick = (now) => {
      if (now - lastPaint >= PAINT_INTERVAL_MS) {
        drawWaveform();
        lastPaint = now;
      }
      rafId = requestAnimationFrame(tick);
    };
    rafId = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafId);
  }, [playing, drawWaveform]);

  // Redraw waveform on resize
  useEffect(() => {
    const canvas = waveformCanvasRef.current;
    if (!canvas) return;
    const ro = new ResizeObserver(() => drawWaveform());
    ro.observe(canvas);
    return () => ro.disconnect();
  }, [drawWaveform]);

  // ── Keyframe thumbnail generation ─────────────────
  const drawThumbnails = useCallback(() => {
    const canvas = thumbnailCanvasRef.current;
    const thumbs = thumbnailsRef.current;
    if (!canvas || !thumbs.length) return;
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) return;
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    const ctx = canvas.getContext && canvas.getContext('2d');
    if (!ctx) return; // Guard against null context.
    ctx.scale(dpr, dpr);
    const sw = rect.width / thumbs.length;
    thumbs.forEach((thumb, i) => {
      if (thumb) {
        try { ctx.drawImage(thumb, i * sw, 0, sw, rect.height); } catch {}
      }
    });
  }, []);

  useEffect(() => {
    if (!src || !videoReady || clipDur <= 0) return;
    let cancelled = false;
    const generate = async () => {
      try {
        const tv = document.createElement('video');
        tv.muted = true;
        tv.preload = 'auto';
        tv.src = src;
        await new Promise((resolve, reject) => {
          tv.onloadeddata = resolve;
          tv.onerror = () => reject();
          setTimeout(() => reject(), 15000);
        });
        if (cancelled) { tv.src = ''; return; }
        const NUM = 15;
        const vw = tv.videoWidth || 320;
        const vh = tv.videoHeight || 180;
        const tH = 60;
        const tW = Math.round(tH * (vw / vh));
        const canvases = [];
        for (let i = 0; i < NUM; i++) {
          if (cancelled) break;
          const time = clipStart + ((i + 0.5) / NUM) * clipDur;
          tv.currentTime = Math.min(time, (tv.duration || time) - 0.05);
          await new Promise(r => { tv.onseeked = r; setTimeout(r, 3000); });
          if (cancelled) break;
          try {
            const c = document.createElement('canvas');
            c.width = tW; c.height = tH;
            const tctx = c.getContext && c.getContext('2d');
            if (!tctx) { canvases.push(null); continue; }
            tctx.drawImage(tv, 0, 0, tW, tH);
            canvases.push(c);
          } catch { canvases.push(null); }
        }
        tv.src = ''; tv.load();
        if (!cancelled && canvases.some(Boolean)) {
          thumbnailsRef.current = canvases;
          drawThumbnails();
        }
      } catch { /* thumbnails are optional */ }
    };
    generate();
    return () => { cancelled = true; thumbnailsRef.current = []; };
  }, [src, videoReady, clipStart, clipDur, drawThumbnails]);

  // Redraw thumbnails on resize
  useEffect(() => {
    const canvas = thumbnailCanvasRef.current;
    if (!canvas) return;
    const ro = new ResizeObserver(() => drawThumbnails());
    ro.observe(canvas);
    return () => ro.disconnect();
  }, [drawThumbnails]);

  // ── Playback time tracking via rAF + native events ──
  // rAF provides smooth visual updates; native events ensure we never
  // miss a seek or time change on any device (mobile can throttle rAF).
  const syncTime = useCallback((t) => {
    setCurrentTime(t);
    onTimeUpdate?.(t);
    // Clamp to non-negative: before the video seeks to clipStart,
    // t can be 0 while clipStart > 0, producing a negative playhead.
    //
    // Skip the timeline-store write while the user is actively
    // scrubbing. During a scrub the mouse is the source of truth for
    // the playhead position; the video element's ``seeked`` /
    // ``timeupdate`` feedback snaps ``currentTime`` to the nearest
    // decoded keyframe, and propagating that back onto the timeline
    // store would visually yank the playhead back out from under the
    // user's cursor at every seek. Timeline sets ``isScrubbing`` on
    // pointerdown and clears it on pointerup, so video-driven
    // playhead sync resumes the instant the user releases.
    const { setPlayhead, isScrubbing } = useTimelineStore.getState();
    if (!isScrubbing) {
      setPlayhead(Math.max(0, t - clipStart));
    }
  }, [onTimeUpdate, clipStart]);

  // Track which segment is active for volume override — store id + a hash
  // of the segment's settings so we re-apply when properties change.
  const activeSegRef = useRef({ id: null, hash: null });

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    let rafId;
    // Tracks the last ``video.currentTime`` value we actually propagated
    // out via ``syncTime``. Used to short-circuit the rAF tick so a
    // paused (or off-screen) ``VideoEditor`` doesn't hammer the parent's
    // time state at ~60 Hz with the same value over and over.
    //
    // This matters on the Analysis page's Transcript tab: the sticky
    // ``VideoEditor`` there is hidden behind ``display: none`` but still
    // mounted, so its tick keeps running. Without this guard it spams
    // ``onTimeUpdate(0)`` 60× per second and stomps on the visible
    // transcript-tab ``VideoPlayer``'s real time updates — the active
    // transcript line ends up pinned to segment 0 no matter where the
    // visible player actually is.
    let lastReportedT = -1;

    // Simple hash of segment settings to detect property changes
    const segHash = (seg) => seg ? `${seg.id}_${seg.muted}_${seg.volume}_${seg.speed}` : null;

    const tick = () => {
      // Stop the rAF loop entirely while the video is paused. The
      // ``play``/``pause`` listeners below restart the loop on play.
      // Previously this scheduled itself every frame even when the
      // editor was hidden behind ``display: none`` (Transcript tab),
      // burning CPU + battery for nothing.
      if (video.paused) {
        rafId = null;
        return;
      }
      const t = video.currentTime;
      // Short-circuit when time hasn't advanced: nothing downstream in
      // this tick cares about an unchanged ``t`` (segment lookups,
      // auto-stop-at-trim-end, volume/speed application are all pure
      // functions of ``t``), so we can skip straight to scheduling the
      // next frame.
      if (t === lastReportedT) {
        rafId = requestAnimationFrame(tick);
        return;
      }
      lastReportedT = t;
      syncTime(t);
      // Auto-stop at trimmed end
      if (trimmedEnd && t >= trimmedEnd) {
        video.pause();
        setPlaying(false);
      }
      // Apply per-segment volume + speed overrides during playback.
      // Re-apply whenever the active segment changes OR its properties change
      // (e.g. user adjusts volume slider while playhead is inside the segment).
      const seg = segments.length > 0 ? segments.find(s => t >= s.start && t < s.end) : null;
      const curId = seg ? seg.id : null;
      const curHash = segHash(seg);
      const prev = activeSegRef.current;

      // Update active segment ID for UI display (only on change to avoid re-renders)
      if (curId !== prev.id) {
        setActiveSegmentId(curId);
        // Show segment entry indicator on viewport
        if (seg && curId) {
          setSegEntryIndicator({
            label: seg.label || 'Segment',
            volume: seg.volume,
            speed: seg.speed || 1.0,
            muted: seg.muted,
            color: seg.color || SEGMENT_COLORS[0],
          });
          if (segEntryTimerRef.current) clearTimeout(segEntryTimerRef.current);
          segEntryTimerRef.current = setTimeout(() => setSegEntryIndicator(null), 1500);
        }
      }
      if (curId !== prev.id || curHash !== prev.hash) {
        activeSegRef.current = { id: curId, hash: curHash };
        if (seg) {
          // Apply segment volume with smooth ramp
          const segVol = seg.muted ? 0 : seg.volume;
          if (gainNodeRef.current && audioCtxRef.current) {
            const now = audioCtxRef.current.currentTime;
            gainNodeRef.current.gain.cancelScheduledValues(now);
            gainNodeRef.current.gain.setValueAtTime(gainNodeRef.current.gain.value, now);
            gainNodeRef.current.gain.linearRampToValueAtTime(segVol / 100, now + 0.05);
          } else {
            video.volume = Math.min(1, segVol / 100);
          }
          // Apply segment speed
          const segSpeed = seg.speed || speed;
          if (Math.abs(video.playbackRate - segSpeed) > 0.001) {
            video.playbackRate = segSpeed;
          }
        } else {
          // Restore global volume with smooth ramp
          const effectiveVol = isMuted ? 0 : volume;
          if (gainNodeRef.current && audioCtxRef.current) {
            const now = audioCtxRef.current.currentTime;
            gainNodeRef.current.gain.cancelScheduledValues(now);
            gainNodeRef.current.gain.setValueAtTime(gainNodeRef.current.gain.value, now);
            gainNodeRef.current.gain.linearRampToValueAtTime(effectiveVol / 100, now + 0.05);
          } else {
            video.volume = Math.min(1, effectiveVol / 100);
          }
          // Restore global speed
          if (Math.abs(video.playbackRate - speed) > 0.001) {
            video.playbackRate = speed;
          }
        }
      }
      rafId = requestAnimationFrame(tick);
    };
    // Restart the rAF loop on play; the loop self-stops on pause.
    const startTick = () => {
      if (rafId == null) {
        rafId = requestAnimationFrame(tick);
      }
    };
    if (!video.paused) startTick();
    video.addEventListener('play', startTick);
    video.addEventListener('playing', startTick);

    // Fallback: native timeupdate + seeked ensure accuracy on mobile
    // where rAF may be throttled or skipped, AND deliver a final
    // ``syncTime`` call right after seeks while the loop is paused.
    // Shares ``lastReportedT`` with the rAF tick so both paths dedupe
    // against each other.
    const onNativeTime = () => {
      const t = video.currentTime;
      if (t === lastReportedT) return;
      lastReportedT = t;
      syncTime(t);
    };
    video.addEventListener('timeupdate', onNativeTime);
    video.addEventListener('seeked', onNativeTime);

    return () => {
      if (rafId != null) cancelAnimationFrame(rafId);
      rafId = null;
      activeSegRef.current = { id: null, hash: null };
      video.removeEventListener('play', startTick);
      video.removeEventListener('playing', startTick);
      video.removeEventListener('timeupdate', onNativeTime);
      video.removeEventListener('seeked', onNativeTime);
    };
  }, [trimmedEnd, syncTime, segments, volume, isMuted, speed]);

  // ── Dynamic subject tracking via rAF — instant snaps ───────────────
  const lastAppliedPctRef = useRef(null);
  useEffect(() => {
    if (!hasDynamicSubject) return;
    const video = videoRef.current;
    if (!video) return;

    // Helper: look up cropX from editable crop segments at a given relative time.
    // Falls back to interpolateSubjectX from the original keyframes.
    const getCropXAtTime = (relTime) => {
      const { cropSegments } = useTimelineStore.getState();
      if (cropSegments?.length > 0) {
        const seg = cropSegments.find(s => relTime >= s.startTime && relTime < s.endTime);
        if (seg) return seg.cropX;
        // Past end — use last segment
        const last = cropSegments[cropSegments.length - 1];
        if (relTime >= last.endTime) return last.cropX;
      }
      return interpolateSubjectX(subjectKeyframes, relTime);
    };

    // Detect reframe-segment mode (keyframes carry easeMs metadata)
    const isReframeMode = subjectKeyframes?.some(k => k.easeMs !== undefined);

    // Ease-out: fast snap to target, smooth deceleration — like a human operator
    const easeCurve = (t) => {
      if (t <= 0) return 0;
      if (t >= 1) return 1;
      return 1 - Math.pow(1 - t, 3);
    };

    const getEaseMsForTransition = (relTime) => {
      if (!isReframeMode || !subjectKeyframes) return 0;
      let active = subjectKeyframes[0];
      for (let i = 1; i < subjectKeyframes.length; i++) {
        if (subjectKeyframes[i].t <= relTime) active = subjectKeyframes[i];
        else break;
      }
      return active?.easeMs || 0;
    };

    // Apply initial position synchronously to eliminate 1-2 frame gap
    {
      const initRel = video.currentTime - (clipStart || 0);
      const initSx = getCropXAtTime(initRel);
      const initPct = subjectXToCenterPct(Math.max(0, Math.min(100, initSx)), srcRatio, targetRatio);
      video.style.objectPosition = `${initPct}% 50%`;
      lastAppliedPctRef.current = initPct;
    }
    let animId;
    let lastPct = null;
    let easeState = null; // { fromPct, toPct, startTime, durationMs }
    const tick = () => {
      const absTime = video.currentTime;
      const relTime = absTime - (clipStart || 0);
      // Check if the current segment has tracking disabled
      const activeSeg = segments.find(s => absTime >= s.start && absTime < s.end);
      const trackingOn = !activeSeg || activeSeg.subjectTrackingEnabled !== false;
      const sx = trackingOn
        ? getCropXAtTime(relTime)
        : (safeSubjectX ? safeSubjectX(subjectX, srcRatio, targetRatio) : subjectX);
      const targetPct = subjectXToCenterPct(Math.max(0, Math.min(100, sx)), srcRatio, targetRatio);
      const targetRounded = Math.round(targetPct * 10000) / 10000;

      let renderPct = targetPct;

      // Start easing on position change (reframe-segment mode)
      if (isReframeMode && targetRounded !== lastPct && lastPct !== null) {
        const easeMs = getEaseMsForTransition(relTime);
        if (easeMs > 0) {
          easeState = {
            fromPct: lastAppliedPctRef.current ?? targetPct,
            toPct: targetPct,
            startTime: performance.now(),
            durationMs: easeMs,
          };
        } else {
          easeState = null;
        }
      }

      if (easeState) {
        const elapsed = performance.now() - easeState.startTime;
        if (elapsed >= easeState.durationMs) {
          renderPct = easeState.toPct;
          easeState = null;
        } else {
          const t = elapsed / easeState.durationMs;
          renderPct = easeState.fromPct + (easeState.toPct - easeState.fromPct) * easeCurve(t);
        }
      }

      const renderRounded = Math.round(renderPct * 10000) / 10000;
      const appliedRounded = lastAppliedPctRef.current != null
        ? Math.round(lastAppliedPctRef.current * 10000) / 10000 : null;
      if (renderRounded !== appliedRounded) {
        video.style.objectPosition = `${renderPct}% 50%`;
        lastAppliedPctRef.current = renderPct;
      }
      lastPct = targetRounded;
      animId = requestAnimationFrame(tick);
    };
    animId = requestAnimationFrame(tick);
    return () => {
      cancelAnimationFrame(animId);
      // Do NOT clear video.style.objectPosition here — the cleanup runs
      // after React's DOM commit, so clearing would overwrite the correct
      // static objectPosition that React just applied. This matches
      // ClipPreview.jsx behavior.
    };
  }, [hasDynamicSubject, subjectKeyframes, clipStart, srcRatio, targetRatio, segments, subjectX]);

  // Static subject tracking fallback
  useEffect(() => {
    if (hasDynamicSubject || !isCrop) return;
    const video = videoRef.current;
    if (!video) return;
    // Use processed keyframe value if available — it interpolates from nearby
    // scenes and is more accurate than the subjectX prop (which may be 50)
    const effectiveSx = (subjectKeyframes?.length >= 1)
      ? subjectKeyframes[0].x
      : subjectX;
    const sx = safeSubjectX ? safeSubjectX(effectiveSx, srcRatio, targetRatio) : effectiveSx;
    const centerPct = subjectXToCenterPct(Math.max(0, Math.min(100, sx)), srcRatio, targetRatio);
    video.style.objectPosition = `${centerPct}% 50%`;
  }, [hasDynamicSubject, isCrop, subjectX, subjectKeyframes, srcRatio, targetRatio]);

  // ── Web Audio API for volume > 100% ────────────────
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    // Only set up Web Audio if not already done
    if (!audioCtxRef.current) {
      try {
        const ctx = new (window.AudioContext || window.webkitAudioContext)();
        const source = ctx.createMediaElementSource(video);
        const gain = ctx.createGain();
        source.connect(gain);
        gain.connect(ctx.destination);
        audioCtxRef.current = ctx;
        sourceNodeRef.current = source;
        gainNodeRef.current = gain;
      } catch (e) {
        // Fallback: no Web Audio, cap at 100%
        console.warn('[VideoEditor] Web Audio API not available:', e);
      }
    }

    return () => {
      // Don't disconnect on every render; only on true unmount handled below
    };
  }, [src]);

  // Cleanup Web Audio on unmount
  useEffect(() => {
    return () => {
      if (audioCtxRef.current) {
        audioCtxRef.current.close().catch(() => {});
        audioCtxRef.current = null;
        sourceNodeRef.current = null;
        gainNodeRef.current = null;
      }
    };
  }, []);

  // ── Apply volume changes ───────────────────────────
  // Skip direct video manipulation when playhead is inside a segment —
  // the rAF loop handles per-segment volume to avoid conflicts.
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    // Check if playhead is inside a segment — if so, rAF handles it
    const t = video.currentTime;
    const inSegment = segments.length > 0 && segments.some(s => t >= s.start && t < s.end);
    if (!inSegment) {
      const effectiveVol = (isMuted || audioTrackMuted) ? 0 : volume;
      if (gainNodeRef.current) {
        // Web Audio path: set GainNode, video.volume = 1
        video.volume = 1;
        gainNodeRef.current.gain.value = effectiveVol / 100;
        // Resume AudioContext if suspended (autoplay policy)
        if (audioCtxRef.current?.state === 'suspended') {
          audioCtxRef.current.resume().catch(() => {});
        }
      } else {
        // Fallback path: native volume 0-1
        video.volume = Math.min(1, effectiveVol / 100);
      }
    }
    // ONLY fire global callback when NOT editing a segment
    if (!selectedSegmentId && !activeSegmentId) {
      onVolumeChange?.(isMuted ? 0 : volume / 100);
    }
  }, [volume, isMuted, audioTrackMuted, onVolumeChange, segments, selectedSegmentId, activeSegmentId]);

  // ── Apply speed changes ────────────────────────────
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    // Skip when playhead is inside a segment — rAF handles per-segment speed
    const t = video.currentTime;
    const inSegment = segments.length > 0 && segments.some(s => t >= s.start && t < s.end);
    if (!inSegment) {
      video.playbackRate = speed;
    }
    // ONLY fire global callback when NOT editing a segment
    if (!selectedSegmentId && !activeSegmentId) {
      onSpeedChange?.(speed);
    }
  }, [speed, onSpeedChange, segments, selectedSegmentId, activeSegmentId]);

  // ── Audio overlay preview playback ──────────────────
  // Manage HTMLAudioElement instances for audio/music overlay tracks
  // so volume and speed changes are audible in the preview player.
  const audioOverlayItems = useMemo(() => {
    return timelineStoreItems.filter(it => it.type === 'audio' && it.trackId !== 'a1');
  }, [timelineStoreItems]);

  // Create/destroy Audio elements when overlay items change
  useEffect(() => {
    const current = audioOverlayRefs.current;
    const activeIds = new Set(audioOverlayItems.map(it => it.id));

    // Remove stale audio elements
    for (const id of Object.keys(current)) {
      if (!activeIds.has(id)) {
        current[id].pause();
        current[id].src = '';
        delete current[id];
      }
    }

    // Create new audio elements
    for (const item of audioOverlayItems) {
      if (current[item.id]) continue;
      let url = item.src || '';
      if (!url && item.mediaRef) {
        const media = timelineMediaLibrary.find(m => m.id === item.mediaRef);
        url = media?.url || '';
      }
      if (!url) continue;
      const audio = new Audio(url);
      audio.preload = 'auto';
      audio.volume = Math.min(1, Math.max(0, item.volume ?? 1));
      audio.playbackRate = item.speed ?? 1;
      audio.addEventListener('error', () => {
        console.warn(`[AudioOverlay] Failed to load audio for item ${item.id}: ${url}`);
        // If blob URL was revoked (page reload), try mediaRef backend URL
        if (url.startsWith('blob:') && item.mediaRef) {
          const media = timelineMediaLibrary.find(m => m.id === item.mediaRef);
          const backendUrl = media?.url;
          if (backendUrl && !backendUrl.startsWith('blob:')) {
            console.log(`[AudioOverlay] Retrying with backend URL for ${item.id}`);
            audio.src = backendUrl;
            audio.load();
          }
        }
      });
      current[item.id] = audio;
    }

    return () => {
      // Cleanup all on unmount
      for (const id of Object.keys(audioOverlayRefs.current)) {
        audioOverlayRefs.current[id].pause();
        audioOverlayRefs.current[id].src = '';
      }
      audioOverlayRefs.current = {};
    };
  }, [audioOverlayItems]); // eslint-disable-line react-hooks/exhaustive-deps

  // Sync volume and speed when properties change
  useEffect(() => {
    const current = audioOverlayRefs.current;
    for (const item of audioOverlayItems) {
      const audio = current[item.id];
      if (!audio) continue;
      const vol = Math.min(1, Math.max(0, item.volume ?? 1));
      if (Math.abs(audio.volume - vol) > 0.001) audio.volume = vol;
      const spd = item.speed ?? 1;
      if (Math.abs(audio.playbackRate - spd) > 0.001) audio.playbackRate = spd;
    }
  }, [audioOverlayItems]);

  // Sync audio overlay play/pause/seek with main video
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    const current = audioOverlayRefs.current;

    const syncAudioOverlays = () => {
      const absTime = video.currentTime;
      const relTime = absTime - clipStart;
      for (const item of audioOverlayItems) {
        const audio = current[item.id];
        if (!audio) continue;
        const itemStart = item.start || 0;
        const itemEnd = item.end || 0;
        const inRange = relTime >= itemStart && relTime < itemEnd;
        if (inRange && !video.paused) {
          const audioTime = relTime - itemStart + (item.trimStart || 0);
          if (Math.abs(audio.currentTime - audioTime) > 0.3) {
            audio.currentTime = audioTime;
          }
          if (audio.paused) audio.play().catch(() => {});
        } else {
          if (!audio.paused) audio.pause();
        }
      }
    };

    const onPlay = () => syncAudioOverlays();
    const onPause = () => {
      for (const item of audioOverlayItems) {
        const audio = current[item.id];
        if (audio && !audio.paused) audio.pause();
      }
    };
    const onSeeked = () => syncAudioOverlays();

    video.addEventListener('play', onPlay);
    video.addEventListener('pause', onPause);
    video.addEventListener('seeked', onSeeked);

    // Also sync during playback via timeupdate
    video.addEventListener('timeupdate', syncAudioOverlays);

    return () => {
      video.removeEventListener('play', onPlay);
      video.removeEventListener('pause', onPause);
      video.removeEventListener('seeked', onSeeked);
      video.removeEventListener('timeupdate', syncAudioOverlays);
    };
  }, [audioOverlayItems, clipStart]);

  // ── Trim change callback ───────────────────────────
  useEffect(() => {
    onTrimChange?.({ trimStart: trimStartOffset, trimEnd: trimEndOffset });
  }, [trimStartOffset, trimEndOffset, onTrimChange]);

  // ── Player controls ────────────────────────────────
  const togglePlay = useCallback(() => {
    const video = videoRef.current;
    if (!video) return;
    // Resume AudioContext if needed
    if (audioCtxRef.current?.state === 'suspended') {
      audioCtxRef.current.resume().catch(() => {});
    }
    if (video.paused) {
      if (video.currentTime >= trimmedEnd) video.currentTime = trimmedStart;
      video.play().then(() => setPlaying(true)).catch(() => {});
    } else {
      video.pause();
      setPlaying(false);
    }
  }, [trimmedStart, trimmedEnd]);

  // seekTo clamps to full clip range, NOT trim region.
  // Trim boundaries only constrain playback auto-stop, not manual seeking.
  const seekTo = useCallback((time) => {
    const video = videoRef.current;
    if (!video) return;
    const clamped = Math.max(clipStart, Math.min(effectiveClipEnd, time));
    video.currentTime = clamped;
    setCurrentTime(clamped);
    useTimelineStore.getState().setPlayhead(clamped - clipStart);
  }, [clipStart, effectiveClipEnd]);

  // Register this editor as the active player. PlayerContext also
  // maintains the legacy ``window.__clipai_*`` globals so any caller
  // that hasn't migrated still works.
  const player = usePlayer();
  useEffect(() => {
    const api = {
      seek: seekTo,
      pause: () => {
        const video = videoRef.current;
        if (video && !video.paused) {
          video.pause();
          setPlaying(false);
        }
      },
      play: () => {
        const video = videoRef.current;
        if (video && video.paused) {
          video.play().then(() => setPlaying(true)).catch(() => {});
        }
      },
      getTime: () => videoRef.current?.currentTime ?? 0,
    };
    player.register(api);
    return () => player.unregister();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [seekTo]);

  const skipTime = useCallback((delta) => {
    const video = videoRef.current;
    if (!video) return;
    seekTo(video.currentTime + delta);
  }, [seekTo]);

  // ── Stable Timeline callbacks ─────────────────────────────
  // These used to be inline arrow functions in the JSX below, which
  // meant ``Timeline`` got a fresh ``onSeek`` / ``onItemSelect`` /
  // ``onSubtitleVisibilityChange`` identity on every VideoEditor
  // re-render — and VideoEditor re-renders on every rAF of the preview
  // playhead. Timeline's drag ``useEffect`` lists those handlers in its
  // dependency array, so the effect was tearing down and re-mounting
  // continuously. Result: the rAF-coalesced playhead scrub was being
  // cancelled before it could flush, leaving the playhead "stationary"
  // while the user dragged it.
  const handleTimelineSeek = useCallback((time) => {
    const video = videoRef.current;
    if (!video) return;
    const absTime = clipStart + time;
    // ── Frame-accurate scrub preview ──
    // Pause the playback FIRST so the browser commits the seek as a
    // hard frame swap. While playing, browsers coalesce rapid
    // currentTime writes and skip intermediate frames — that's why
    // scrubbing felt like the preview "stuck" until release.
    if (!video.paused) {
      try { video.pause(); } catch { /* noop */ }
      try { setPlaying(false); } catch { /* noop */ }
    }
    // Prefer ``fastSeek`` (Firefox + Safari) — it jumps to the nearest
    // keyframe without decoding every intermediate frame, the right
    // tradeoff during scrub. Chrome/Edge fall through to
    // ``currentTime`` which is accurate but slower.
    try {
      if (typeof video.fastSeek === 'function') {
        video.fastSeek(absTime);
      } else {
        video.currentTime = absTime;
      }
    } catch {
      try { video.currentTime = absTime; } catch { /* noop */ }
    }
    setCurrentTime(absTime);
    onTimeUpdate?.(absTime);
  }, [clipStart, onTimeUpdate]);

  const handleTimelineItemSelect = useCallback(() => {
    setShowProperties(true);
  }, []);

  const handleTimelineSubtitleVisibility = useCallback(
    (visible) => {
      if (onSettingsChange) {
        onSettingsChange({ ...settings, subtitlesEnabled: visible });
      }
    },
    [onSettingsChange, settings],
  );

  const toggleMute = useCallback(() => {
    if (controlTargetSegment) {
      updateSegment(controlTargetSegment.id, {
        muted: !controlTargetSegment.muted,
        volume: controlTargetSegment.muted ? 100 : 0,
      });
      return;
    }
    if (isMuted) {
      setIsMuted(false);
      setVolume(prevVolume || 100);
    } else {
      setPrevVolume(volume);
      setIsMuted(true);
    }
  }, [isMuted, volume, prevVolume, controlTargetSegment, updateSegment]);

  const toggleFullscreen = useCallback(() => {
    setIsFullscreen((prev) => !prev);
  }, []);

  // ── NLE keyboard shortcuts (only active in multi-track mode) ──
  const handleShuttleSpeed = useCallback((dir) => {
    if (dir === 'stop') {
      setShuttleSpeed(0);
      if (videoRef.current) { videoRef.current.pause(); setPlaying(false); }
    } else if (dir === 'reverse') {
      setShuttleSpeed(prev => {
        if (prev > 0) return 0;
        const steps = [0, -1, -2, -4];
        const idx = steps.indexOf(prev);
        return steps[Math.min(idx + 1, steps.length - 1)] ?? -1;
      });
    } else if (dir === 'forward') {
      setShuttleSpeed(prev => {
        if (prev < 0) return 0;
        const steps = [0, 1, 2, 4];
        const idx = steps.indexOf(prev);
        return steps[Math.min(idx + 1, steps.length - 1)] ?? 1;
      });
    }
  }, []);
  useKeyboardShortcuts({
    enabled: showMultiTrack,
    onTogglePlay: togglePlay,
    onSeek: seekTo,
    onSkipTime: skipTime,
    onToggleMute: toggleMute,
    onShuttleSpeed: handleShuttleSpeed,
    // ``seekTo`` expects wall-clock time. Without this, End jumped to
    // ``timelineStore.duration`` (timeline-relative) which clamped to
    // ``clipStart`` for any clip with a non-zero start.
    clipRange: { start: clipStart, end: clipEnd },
  });

  // ── Apply trim ────────────────────────────────────
  const handleApplyTrim = useCallback(() => {
    setTrimApplied(true);
    onTrimChange?.({ trimStart: trimStartOffset, trimEnd: trimEndOffset });
    // Notify parent to update clip boundaries so the trimmed range becomes the full video
    if (onApplyTrim) {
      // Adjust segment coordinates to fit the new clip range
      const newStart = trimmedStart;
      const newEnd = trimmedEnd;
      const adjusted = segments
        .map(seg => {
          const clampedStart = Math.max(newStart, seg.start);
          const clampedEnd = Math.min(newEnd, seg.end);
          if (clampedEnd - clampedStart < 0.5) return null; // segment falls outside
          return { ...seg, start: clampedStart, end: clampedEnd };
        })
        .filter(Boolean);
      if (adjusted.length !== segments.length || adjusted.some((s, i) => s.start !== segments[i]?.start || s.end !== segments[i]?.end)) {
        setSegments(adjusted);
        onSegmentsChange?.(adjusted);
      }
      onApplyTrim({ start: newStart, end: newEnd });
      // Reset trim offsets since the clip boundaries are now narrower
      setTrimStartOffset(0);
      setTrimEndOffset(0);
    }
  }, [trimStartOffset, trimEndOffset, trimmedStart, trimmedEnd, onTrimChange, onApplyTrim, segments, onSegmentsChange]);

  // Reset applied state when trim handles change
  useEffect(() => {
    setTrimApplied(false);
  }, [trimStartOffset, trimEndOffset]);

  // ── Timeline pointer handling ──────────────────────
  const getTimeFromPointer = useCallback((clientX) => {
    const track = timelineRef.current;
    if (!track || clipDur <= 0) return clipStart;
    const rect = track.getBoundingClientRect();
    const pct = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
    return clipStart + pct * clipDur;
  }, [clipStart, clipDur]);

  const startDragTracking = useCallback((e, handle) => {
    const onMove = (ev) => {
      const time = getTimeFromPointer(ev.clientX);
      const offset = time - clipStart;

      if (handle === 'left') {
        const maxOffset = clipDur - trimEndOffset - 1; // minimum 1s gap
        const newOffset = Math.max(0, Math.min(maxOffset, offset));
        setTrimStartOffset(newOffset);
        // Show frame at handle position
        const video = videoRef.current;
        if (video) video.currentTime = clipStart + newOffset;
      } else {
        const maxOffset = clipDur - trimStartOffset - 1;
        const fromEnd = effectiveClipEnd - time;
        const newOffset = Math.max(0, Math.min(maxOffset, fromEnd));
        setTrimEndOffset(newOffset);
        const video = videoRef.current;
        if (video) video.currentTime = effectiveClipEnd - newOffset;
      }
    };
    const onUp = () => {
      setDraggingHandle(null);
      window.removeEventListener('pointermove', onMove);
      window.removeEventListener('pointerup', onUp);
    };
    window.addEventListener('pointermove', onMove);
    window.addEventListener('pointerup', onUp);
  }, [clipStart, effectiveClipEnd, clipDur, trimStartOffset, trimEndOffset, getTimeFromPointer]);

  // ── Segment drag-to-resize and drag-to-move ────────
  const EDGE_THRESHOLD_PX = 8;
  const PLAYHEAD_SNAP_PX = 5;

  const onSegmentPointerMove = useCallback((e) => {
    if (segDrag) return;
    const track = timelineRef.current;
    if (!track || clipDur <= 0) return;
    const rect = track.getBoundingClientRect();
    const pointerX = e.clientX - rect.left;
    const trackWidth = rect.width;

    for (const seg of segments) {
      const segLeftPx = ((seg.start - clipStart) / clipDur) * trackWidth;
      const segRightPx = ((seg.end - clipStart) / clipDur) * trackWidth;

      if (Math.abs(pointerX - segLeftPx) < EDGE_THRESHOLD_PX) {
        setSegHoverEdge({ segId: seg.id, edge: 'start' });
        return;
      }
      if (Math.abs(pointerX - segRightPx) < EDGE_THRESHOLD_PX) {
        setSegHoverEdge({ segId: seg.id, edge: 'end' });
        return;
      }
      if (pointerX > segLeftPx + EDGE_THRESHOLD_PX && pointerX < segRightPx - EDGE_THRESHOLD_PX) {
        setSegHoverEdge({ segId: seg.id, edge: 'center' });
        return;
      }
    }
    setSegHoverEdge(null);
  }, [segments, clipStart, clipDur, segDrag]);

  const startSegDrag = useCallback((e, segId, mode) => {
    e.preventDefault();
    e.stopPropagation();
    const seg = segments.find(s => s.id === segId);
    if (!seg) return;
    setSelectedSegmentId(segId);

    const track = timelineRef.current;
    if (!track) return;
    const trackRect = track.getBoundingClientRect();

    const state = {
      segId,
      mode,
      startPointerX: e.clientX,
      origStart: seg.start,
      origEnd: seg.end,
      trackLeft: trackRect.left,
      trackWidth: trackRect.width,
    };
    setSegDrag(state);

    const onMove = (ev) => {
      const pxDelta = ev.clientX - state.startPointerX;
      const timeDelta = (pxDelta / state.trackWidth) * clipDur;
      const playheadTime = videoRef.current?.currentTime ?? currentTime;
      const playheadPx = ((playheadTime - clipStart) / clipDur) * state.trackWidth;
      const pointerPx = ev.clientX - state.trackLeft;

      if (mode === 'start') {
        let newStart = state.origStart + timeDelta;
        if (Math.abs(pointerPx - playheadPx) < PLAYHEAD_SNAP_PX) newStart = playheadTime;
        newStart = Math.max(clipStart, Math.min(state.origEnd - 0.5, newStart));
        const prevSeg = segments.filter(s => s.id !== segId && s.end <= state.origStart).sort((a, b) => b.end - a.end)[0];
        if (prevSeg) newStart = Math.max(prevSeg.end, newStart);
        updateSegment(segId, { start: newStart });
      } else if (mode === 'end') {
        let newEnd = state.origEnd + timeDelta;
        if (Math.abs(pointerPx - playheadPx) < PLAYHEAD_SNAP_PX) newEnd = playheadTime;
        newEnd = Math.max(state.origStart + 0.5, Math.min(effectiveClipEnd, newEnd));
        const nextSeg = segments.filter(s => s.id !== segId && s.start >= state.origEnd).sort((a, b) => a.start - b.start)[0];
        if (nextSeg) newEnd = Math.min(nextSeg.start, newEnd);
        updateSegment(segId, { end: newEnd });
      } else if (mode === 'move') {
        const dur = state.origEnd - state.origStart;
        let newStart = state.origStart + timeDelta;
        let newEnd = newStart + dur;
        if (newStart < clipStart) { newStart = clipStart; newEnd = newStart + dur; }
        if (newEnd > effectiveClipEnd) { newEnd = effectiveClipEnd; newStart = newEnd - dur; }
        const prevSeg = segments.filter(s => s.id !== segId && s.end <= state.origStart).sort((a, b) => b.end - a.end)[0];
        const nextSeg = segments.filter(s => s.id !== segId && s.start >= state.origEnd).sort((a, b) => a.start - b.start)[0];
        if (prevSeg && newStart < prevSeg.end) { newStart = prevSeg.end; newEnd = newStart + dur; }
        if (nextSeg && newEnd > nextSeg.start) { newEnd = nextSeg.start; newStart = newEnd - dur; }
        updateSegment(segId, { start: newStart, end: newEnd });
      }
    };

    const onUp = () => {
      setSegDrag(null);
      window.removeEventListener('pointermove', onMove);
      window.removeEventListener('pointerup', onUp);
    };
    window.addEventListener('pointermove', onMove);
    window.addEventListener('pointerup', onUp);
  }, [segments, clipStart, clipDur, effectiveClipEnd, currentTime, updateSegment]);

  const onTimelinePointerDown = useCallback((e) => {
    e.preventDefault();
    const track = timelineRef.current;
    if (!track) return;
    const rect = track.getBoundingClientRect();
    const pct = (e.clientX - rect.left) / rect.width;
    const time = clipStart + pct * clipDur;

    // Check if clicking near trim handles
    const leftHandlePct = trimStartOffset / clipDur;
    const rightHandlePct = 1 - trimEndOffset / clipDur;
    const handleThresholdPct = 24 / rect.width; // 24px hit area

    if (Math.abs(pct - leftHandlePct) < handleThresholdPct) {
      setDraggingHandle('left');
      startDragTracking(e, 'left');
      return;
    }
    if (Math.abs(pct - rightHandlePct) < handleThresholdPct) {
      setDraggingHandle('right');
      startDragTracking(e, 'right');
      return;
    }

    // Check if clicking near a segment edge (for resize) or center (for move)
    if (segments.length > 0) {
      const pointerX = e.clientX - rect.left;
      const trackWidth = rect.width;
      for (const seg of segments) {
        const segLeftPx = ((seg.start - clipStart) / clipDur) * trackWidth;
        const segRightPx = ((seg.end - clipStart) / clipDur) * trackWidth;
        if (Math.abs(pointerX - segLeftPx) < EDGE_THRESHOLD_PX) {
          startSegDrag(e, seg.id, 'start');
          return;
        }
        if (Math.abs(pointerX - segRightPx) < EDGE_THRESHOLD_PX) {
          startSegDrag(e, seg.id, 'end');
          return;
        }
        if (pointerX > segLeftPx + EDGE_THRESHOLD_PX && pointerX < segRightPx - EDGE_THRESHOLD_PX) {
          startSegDrag(e, seg.id, 'move');
          return;
        }
      }
    }

    // Click-to-seek anywhere on timeline (Premiere-style)
    seekTo(time);

    // Check if click is inside a segment — if so, select it; otherwise deselect
    const clickedSeg = segments.find(s => time >= s.start && time < s.end);
    if (clickedSeg) {
      setSelectedSegmentId(prev => prev === clickedSeg.id ? null : clickedSeg.id);
    } else {
      setSelectedSegmentId(null);
    }

    setDraggingPlayhead(true);

    // Remember play state to restore after scrub
    const wasPlaying = !videoRef.current?.paused;
    if (wasPlaying) {
      videoRef.current.pause();
      setPlaying(false);
    }

    let rafPending = null;
    const onMove = (ev) => {
      const t = getTimeFromPointer(ev.clientX);
      // Throttle via rAF for smooth scrubbing — seekTo handles clamping to clip range
      if (rafPending === null) {
        rafPending = requestAnimationFrame(() => {
          seekTo(t);
          rafPending = null;
        });
      }
    };
    const onUp = () => {
      setDraggingPlayhead(false);
      if (rafPending !== null) cancelAnimationFrame(rafPending);
      // Resume playback if it was playing before scrub
      if (wasPlaying && videoRef.current) {
        videoRef.current.play().then(() => setPlaying(true)).catch(() => {});
      }
      window.removeEventListener('pointermove', onMove);
      window.removeEventListener('pointerup', onUp);
    };
    window.addEventListener('pointermove', onMove);
    window.addEventListener('pointerup', onUp);
  }, [clipStart, clipDur, trimStartOffset, trimEndOffset, seekTo, getTimeFromPointer, startDragTracking, segments, startSegDrag]);

  // Trim handle direct pointer down
  const onTrimHandlePointerDown = useCallback((e, handle) => {
    e.preventDefault();
    e.stopPropagation();
    setDraggingHandle(handle);
    startDragTracking(e, handle);
  }, [startDragTracking]);

  // ── Speed menu ─────────────────────────────────────
  const selectSpeed = useCallback((val) => {
    if (controlTargetSegment) {
      updateSegment(controlTargetSegment.id, { speed: val });
      setShowSpeedMenu(false);
      return;
    }
    setSpeed(val);
    setShowSpeedMenu(false);
  }, [controlTargetSegment, updateSegment]);

  // Close speed menu on outside click
  useEffect(() => {
    if (!showSpeedMenu) return;
    const onClick = () => setShowSpeedMenu(false);
    window.addEventListener('click', onClick);
    return () => window.removeEventListener('click', onClick);
  }, [showSpeedMenu]);

  // Keep refs in sync for keyboard effect (avoids stale closures / effect re-runs)
  skipTimeRef.current = skipTime;
  // ``currentTimeRef`` is declared above for the waveform redraw guard;
  // we only need to keep it in sync with state here for the keyboard
  // path that reads it.
  currentTimeRef.current = currentTime;
  const segmentsRef = useRef(segments);
  segmentsRef.current = segments;
  const selectedSegmentIdRef = useRef(selectedSegmentId);
  selectedSegmentIdRef.current = selectedSegmentId;
  const selectedSegmentRef = useRef(selectedSegment);
  selectedSegmentRef.current = selectedSegment;

  // Stable refs for keyboard handler — prevents effect re-registration from killing arrow hold intervals
  const togglePlayRef = useRef(togglePlay);
  togglePlayRef.current = togglePlay;
  const seekToRef_kb = useRef(seekTo);
  seekToRef_kb.current = seekTo;
  const toggleMuteRef = useRef(toggleMute);
  toggleMuteRef.current = toggleMute;
  const trimmedStartRef = useRef(trimmedStart);
  trimmedStartRef.current = trimmedStart;
  const trimmedEndRef = useRef(trimmedEnd);
  trimmedEndRef.current = trimmedEnd;
  const onSegmentsChangeRef = useRef(onSegmentsChange);
  onSegmentsChangeRef.current = onSegmentsChange;
  const showMultiTrackRef = useRef(showMultiTrack);
  showMultiTrackRef.current = showMultiTrack;

  // ── Keyboard shortcuts (J-K-L shuttle control) ─────
  useEffect(() => {
    const onKeyDown = (e) => {
      // Don't capture keys when typing in inputs or contenteditable elements
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.tagName === 'SELECT') return;
      if (e.target.contentEditable === 'true' || e.target.closest('[contenteditable="true"]')) return;
      // Prevent native video element keyboard handling
      if (e.target.tagName === 'VIDEO') e.target.blur();

      // When multi-track is active, useKeyboardShortcuts handles transport controls
      // (Space, Arrows, J/K/L, Home/End, M). Skip them here to avoid double-firing.
      if (showMultiTrackRef.current) {
        const mtKeys = ['Space', 'ArrowLeft', 'ArrowRight', 'KeyJ', 'KeyK', 'KeyL', 'Home', 'End', 'KeyM'];
        if (mtKeys.includes(e.code)) {
          e.preventDefault();
          return;
        }
      }

      switch (e.code) {
        case 'Space':
          e.preventDefault();
          setShuttleSpeed(0);
          togglePlayRef.current();
          break;
        case 'ArrowLeft':
        case 'ArrowRight': {
          e.preventDefault();
          const arrowDir = e.code === 'ArrowLeft' ? -1 : 1;
          const arrowDelta = e.shiftKey ? arrowDir : arrowDir / 30;

          const hold = arrowHoldRef.current;
          if (e.repeat) {
            // If our interval is still running, let it handle stepping
            if (hold.interval) break;
            // Otherwise effect cleanup killed the interval; restart below
          } else {
            // First press: immediate single step
            skipTimeRef.current(arrowDelta);
          }
          // Start (or restart) hold-to-repeat interval
          if (hold.interval) clearInterval(hold.interval);
          hold.key = e.code;
          hold.interval = setInterval(() => {
            skipTimeRef.current?.(arrowDelta);
          }, 1000 / 15); // 15 steps per second while held
          break;
        }
        case 'KeyJ':
          e.preventDefault();
          setShuttleSpeed(prev => {
            if (prev > 0) return 0; // If going forward, stop first
            const reverseSteps = [0, -1, -2, -4];
            const curIdx = reverseSteps.indexOf(prev);
            return reverseSteps[Math.min(curIdx + 1, reverseSteps.length - 1)] ?? -1;
          });
          break;
        case 'KeyK':
          e.preventDefault();
          setShuttleSpeed(0);
          if (videoRef.current) {
            videoRef.current.pause();
            setPlaying(false);
          }
          break;
        case 'KeyL':
          e.preventDefault();
          setShuttleSpeed(prev => {
            if (prev < 0) return 0; // If going reverse, stop first
            const forwardSteps = [0, 1, 2, 4];
            const curIdx = forwardSteps.indexOf(prev);
            return forwardSteps[Math.min(curIdx + 1, forwardSteps.length - 1)] ?? 1;
          });
          break;
        case 'Home':
          e.preventDefault();
          seekToRef_kb.current(trimmedStartRef.current);
          break;
        case 'End':
          e.preventDefault();
          seekToRef_kb.current(trimmedEndRef.current);
          break;
        case 'KeyM':
          e.preventDefault();
          toggleMuteRef.current();
          break;
        // ── Segment shortcuts (only in simple mode, not multi-track) ──
        case 'KeyS': {
          if (showMultiTrackRef.current) break;
          e.preventDefault();
          const splitTime = videoRef.current?.currentTime ?? currentTimeRef.current;
          // Split existing segment at playhead, or create a new one
          const existingSeg = segmentsRef.current.find(s => splitTime > s.start + 0.5 && splitTime < s.end - 0.5);
          if (existingSeg) {
            const seg1 = { ...existingSeg, end: splitTime };
            const newId = `seg_${segmentIdRef.current++}`;
            const segs = segmentsRef.current;
            const seg2 = {
              ...existingSeg,
              id: newId,
              start: splitTime,
              label: `Segment ${segs.length + 1}`,
              color: SEGMENT_COLORS[segs.length % SEGMENT_COLORS.length],
            };
            const next = segs.map(s => s.id === existingSeg.id ? seg1 : s);
            next.push(seg2);
            next.sort((a, b) => a.start - b.start);
            setSegments(next);
            onSegmentsChangeRef.current?.(next);
            setSelectedSegmentId(newId);
          } else {
            const halfDur = 2;
            const segs = segmentsRef.current;
            const newSeg = {
              id: `seg_${segmentIdRef.current++}`,
              start: Math.max(trimmedStartRef.current, splitTime - halfDur),
              end: Math.min(trimmedEndRef.current, splitTime + halfDur),
              volume: 100,
              muted: false,
              subtitlesEnabled: true,
              speed: 1.0,
              color: SEGMENT_COLORS[segs.length % SEGMENT_COLORS.length],
              label: `Segment ${segs.length + 1}`,
            };
            const next = [...segs, newSeg].sort((a, b) => a.start - b.start);
            setSegments(next);
            onSegmentsChangeRef.current?.(next);
            setSelectedSegmentId(newSeg.id);
          }
          break;
        }
        case 'Delete':
        case 'Backspace': {
          if (showMultiTrackRef.current) break;
          const delSegId = selectedSegmentIdRef.current;
          if (delSegId) {
            e.preventDefault();
            const next = segmentsRef.current.filter(s => s.id !== delSegId);
            setSegments(next);
            onSegmentsChangeRef.current?.(next);
            setSelectedSegmentId(null);
          }
          break;
        }
        case 'Escape':
          if (showMultiTrackRef.current) break;
          if (selectedSegmentIdRef.current) {
            e.preventDefault();
            setSelectedSegmentId(null);
          }
          break;
        case 'Tab': {
          if (showMultiTrackRef.current) break;
          const segsTab = segmentsRef.current;
          if (segsTab.length > 0) {
            e.preventDefault();
            const curIdx = segsTab.findIndex(s => s.id === selectedSegmentIdRef.current);
            if (e.shiftKey) {
              const prevIdx = curIdx <= 0 ? segsTab.length - 1 : curIdx - 1;
              setSelectedSegmentId(segsTab[prevIdx].id);
            } else {
              const nextIdx = curIdx < 0 || curIdx >= segsTab.length - 1 ? 0 : curIdx + 1;
              setSelectedSegmentId(segsTab[nextIdx].id);
            }
          }
          break;
        }
        case 'BracketLeft':
          if (showMultiTrackRef.current) break;
          if (selectedSegmentRef.current) {
            e.preventDefault();
            seekToRef_kb.current(selectedSegmentRef.current.start);
          }
          break;
        case 'BracketRight':
          if (showMultiTrackRef.current) break;
          if (selectedSegmentRef.current) {
            e.preventDefault();
            seekToRef_kb.current(selectedSegmentRef.current.end);
          }
          break;
      }
    };
    const onKeyUp = (e) => {
      if (e.code === 'ArrowLeft' || e.code === 'ArrowRight') {
        const hold = arrowHoldRef.current;
        if (hold.key === e.code && hold.interval) {
          clearInterval(hold.interval);
          hold.interval = null;
          hold.key = null;
        }
      }
    };
    window.addEventListener('keydown', onKeyDown);
    window.addEventListener('keyup', onKeyUp);
    return () => {
      window.removeEventListener('keydown', onKeyDown);
      window.removeEventListener('keyup', onKeyUp);
      // Clean up any lingering interval
      const hold = arrowHoldRef.current;
      if (hold.interval) {
        clearInterval(hold.interval);
        hold.interval = null;
        hold.key = null;
      }
    };
  }, []); // Stable: all dependencies accessed via refs

  // ── J-K-L shuttle speed effect ──────────────────────
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    if (shuttleSpeed === 0) {
      // Restore original playback rate and pause when shuttle stops
      video.playbackRate = speed;
      video.pause();
      setPlaying(false);
      return;
    }

    if (shuttleSpeed > 0) {
      video.playbackRate = shuttleSpeed;
      video.play().then(() => setPlaying(true)).catch(() => {});
    } else {
      // Reverse playback via interval-based frame stepping
      video.pause();
      setPlaying(true); // Show as "playing" for UI
      const interval = setInterval(() => {
        const step = (1 / 30) * Math.abs(shuttleSpeed);
        const newTime = Math.max(trimmedStart, video.currentTime - step);
        video.currentTime = newTime;
        setCurrentTime(newTime);
        if (newTime <= trimmedStart) {
          setShuttleSpeed(0);
          setPlaying(false);
        }
      }, 1000 / 30);
      return () => clearInterval(interval);
    }
  }, [shuttleSpeed, trimmedStart, speed]);

  // ── Volume icon selector ───────────────────────────
  const VolumeIcon = useMemo(() => {
    if (isMuted || volume === 0) return Icon.VolumeMuted;
    if (volume < 50) return Icon.VolumeLow;
    if (volume <= 100) return Icon.VolumeMed;
    return Icon.VolumeHigh;
  }, [isMuted, volume]);

  // ── Timeline position calculations ─────────────────
  const leftTrimPct = clipDur > 0 ? (trimStartOffset / clipDur) * 100 : 0;
  const rightTrimPct = clipDur > 0 ? (trimEndOffset / clipDur) * 100 : 0;
  const progressPct = clipDur > 0 ? ((currentTime - clipStart) / clipDur) * 100 : 0;
  const playheadPct = Math.max(0, Math.min(100, progressPct));

  // ── Ruler marks ────────────────────────────────────
  const rulerMarks = useMemo(() => {
    if (clipDur <= 0) return [];
    const count = compact ? 3 : 5;
    const marks = [];
    for (let i = 0; i < count; i++) {
      const t = (i / (count - 1)) * clipDur;
      marks.push(formatTimeShort(t));
    }
    return marks;
  }, [clipDur, compact]);

  // Trim active check
  const hasTrim = trimStartOffset > 0.01 || trimEndOffset > 0.01;

  // ── Error state ────────────────────────────────────
  if (videoError) {
    return (
      <div className="ve-error">
        <span>Failed to load video</span>
        <button className="ve-error__retry" onClick={() => {
          setVideoError(false);
          setVideoPreparing(false);
          // Reset auto-retry budget so a manual Retry gets another
          // full round of backoff attempts, not just a single shot.
          retryCountRef.current = 0;
          if (retryTimerRef.current) {
            clearTimeout(retryTimerRef.current);
            retryTimerRef.current = null;
          }
          videoRef.current?.load();
        }}>
          Retry
        </button>
      </div>
    );
  }

  // ── Render ─────────────────────────────────────────
  // Lightbox overlays the viewport only (not the whole container) so the
  // editor chrome stays where it is — the lightbox just paints over it.
  const containerClass = [
    've-container',
    compact && 've-container--compact',
  ].filter(Boolean).join(' ');

  const initialObjectPosition = isCrop
    ? `${subjectXToCenterPct(
        (subjectKeyframes?.length >= 1)
          ? subjectKeyframes[0].x
          : (safeSubjectX ? safeSubjectX(subjectX, srcRatio, targetRatio) : subjectX),
        srcRatio,
        targetRatio,
      )}% 50%`
    : undefined;

  return (
    <div
      ref={containerRef}
      className={containerClass}
    >
      {/* ── Header ── */}
      {(title || onClose) && !isFullscreen && (
        <div className="ve-header">
          <span className="ve-header__title">{title || 'Clip Preview'}</span>
          {onClose && (
            <button className="ve-header__close" onClick={onClose} title="Close">
              <Icon.Close />
            </button>
          )}
        </div>
      )}

      {/* ── Viewport ── */}
      <div
        ref={viewportRef}
        className={`ve-viewport${isFullscreen ? ' ve-viewport--fullscreen' : ''}`}
        style={isFullscreen ? {} : {
          aspectRatio: `${targetRatio}`,
          maxHeight: compact ? '55vh' : '50vh',
          maxWidth: compact ? undefined : `calc(50vh * ${targetRatio})`,
          width: '100%',
          margin: '0 auto',
        }}
        onPointerDown={() => { viewportClickRef.current = { downTime: Date.now(), moved: false }; }}
        onPointerMove={() => { if (viewportClickRef.current.downTime) viewportClickRef.current.moved = true; }}
        onClick={(e) => {
          if (overlayInteracting) return;
          if (viewportClickRef.current.moved) return;
          if (Date.now() - viewportClickRef.current.downTime > 300) return;
          // Lightbox: clicking the black letterbox backdrop (i.e. the viewport
          // itself, not the stage or its descendants) dismisses the lightbox.
          // Clicks on the stage/video still toggle play.
          if (isFullscreen && e.target === e.currentTarget) {
            setIsFullscreen(false);
            return;
          }
          // In multi-track mode on desktop: clicking the viewport selects the
          // video item (so the user can drag/resize/rotate it). If the video is
          // already selected, deselect and toggle play instead.
          // On mobile: always toggle play directly — users rely on tapping the
          // viewport to play/pause, and can select items via the timeline.
          if (showMultiTrack && videoTimelineItem && !isMobile && !isFullscreen) {
            if (storeSelectedItemId === videoTimelineItem.id) {
              setSelectedItemId(null);
              togglePlay();
            } else {
              setSelectedItemId(videoTimelineItem.id);
            }
          } else {
            setSelectedItemId(null);
            togglePlay();
          }
        }}
      >
        {/* Lightbox close button — only visible in fullscreen */}
        {isFullscreen && (
          <button
            type="button"
            className="ve-lightbox-close"
            onClick={(e) => { e.stopPropagation(); setIsFullscreen(false); }}
            title="Exit fullscreen (Esc)"
            aria-label="Exit fullscreen"
          >
            <Icon.Close />
          </button>
        )}
        <div
          ref={stageRef}
          className={`ve-stage${isFullscreen ? ' ve-stage--fullscreen' : ''}`}
          style={isFullscreen ? { '--ve-target-ratio': targetRatio } : undefined}
        >
        {/* Preview-transcode overlay. The backend returns 503 while
            the browser-playable preview is still being built (common
            for large 4K / VP9 / MKV uploads). The ``<video>`` element
            fires ``error`` on every retry in that window; rather than
            showing a hard "Failed to load video" card (which would
            unmount the player and stall the retry loop), we overlay a
            friendly status message and let the retry timer keep
            ticking in the background. */}
        {videoPreparing && (
          <div className="ve-preparing-overlay">
            <span className="ve-preparing-overlay__title">
              Preparing video preview…
            </span>
            <span className="ve-preparing-overlay__hint">
              Large or high-resolution uploads can take a few minutes
              to transcode for in-browser playback.
            </span>
          </div>
        )}
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
              boxShadow: (trackingStatus.mode === 'dynamic' || trackingStatus.mode === 'multi') ? `0 0 4px ${trackingStatus.color}` : 'none',
            }} />
            {livePosition !== null ? `Face tracked at ${livePosition}%` : trackingStatus.label}
          </div>
        )}
        <video
          ref={videoRef}
          src={src}
          tabIndex={-1}
          playsInline
          preload="auto"
          style={(() => {
            const hasCustomTransform = (
              videoItemPosition.x !== 50 || videoItemPosition.y !== 50 ||
              videoItemSize.w !== 100 || videoItemSize.h !== 100 ||
              videoItemRotation !== 0
            );
            if (hasCustomTransform) {
              // When user has adjusted position/size/rotation, render video as
              // a positioned element within the viewport (like other overlay items)
              return {
                position: 'absolute',
                left: `${videoItemPosition.x}%`,
                top: `${videoItemPosition.y}%`,
                width: `${videoItemSize.w}%`,
                height: `${videoItemSize.h}%`,
                objectFit: isCrop ? 'cover' : 'contain',
                objectPosition: initialObjectPosition,
                transform: `translate(-50%, -50%)${videoItemRotation ? ` rotate(${videoItemRotation}deg)` : ''}`,
                opacity: videoItemOpacity,
                filter: videoItemFilter || undefined,
                transition: 'opacity 0.1s, filter 0.1s',
                // Hide video visually when track is hidden (use visibility so audio still plays)
                visibility: videoTrackHidden ? 'hidden' : undefined,
              };
            }
            // Default: fill viewport
            return {
              objectFit: isCrop ? 'cover' : 'contain',
              objectPosition: initialObjectPosition,
              opacity: videoItemOpacity,
              filter: videoItemFilter || undefined,
              transition: 'opacity 0.1s, filter 0.1s',
              visibility: videoTrackHidden ? 'hidden' : undefined,
            };
          })()}
        />

        {/* Multi-track timeline overlay: text, shapes, images — always rendered
            so edits remain visible even when the multi-track editor panel is closed */}
        <TimelineOverlay
          currentTime={currentTime - clipStart}
          clipStart={clipStart}
        />

        {/* Subtitle overlay renders AFTER timeline overlays so subtitles
            (which default to the topmost track) appear on top of shapes,
            images, and text overlays — matching the track stacking order. */}
        {subtitleOverlay}

        {/* Interactive overlay: click-to-select, drag, resize, rotate on preview.
            Disabled in lightbox/fullscreen — that view is preview-focused, not
            for editing — so taps don't accidentally drag/select items. */}
        {showMultiTrack && !isFullscreen && (
          <InteractiveOverlay
            currentTime={currentTime - clipStart}
            clipStart={clipStart}
            containerRef={stageRef}
            onInteraction={setOverlayInteracting}
          />
        )}

        {/* Segment entry indicator */}
        {segEntryIndicator && (
          <div className="ve-segment-enter-indicator" style={{ '--seg-color': segEntryIndicator.color }}>
            <span className="ve-segment-enter-indicator__dot" style={{ background: segEntryIndicator.color }} />
            <span>{String(segEntryIndicator.label || '')}</span>
            {segEntryIndicator.muted ? (
              <span style={{ color: '#FF3B30' }}>Muted</span>
            ) : (
              <span>{segEntryIndicator.volume}%</span>
            )}
            {Math.abs(segEntryIndicator.speed - 1.0) > 0.001 && (
              <span>{segEntryIndicator.speed}x</span>
            )}
          </div>
        )}

        {aspectRatio && (
          <div className="ve-aspect-badge">{String(aspectRatio || '')}</div>
        )}

        {!videoReady && (
          <div style={{
            position: 'absolute', inset: 0, zIndex: 20,
            display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
            background: 'var(--bg-elevated, #1a1a1a)',
            gap: 12,
          }}>
            <div className="ve-loading__spinner" />
            <span style={{ fontSize: 12, color: 'var(--text-muted, #888)', letterSpacing: '0.02em' }}>
              Loading video...
            </span>
          </div>
        )}

        {videoReady && !playing && (
          <div className="ve-viewport__play-overlay" style={{ pointerEvents: 'none' }}>
            <div className="ve-viewport__play-icon">
              <svg width="22" height="22" viewBox="0 0 24 24" fill="white" stroke="none">
                <path d="M6.5 4.1c-.9-.5-2 .1-2 1.2v13.4c0 1.1 1.1 1.7 2 1.2l11.6-6.7c.9-.5.9-1.8 0-2.4L6.5 4.1z" />
              </svg>
            </div>
          </div>
        )}
        </div> {/* /ve-stage */}
      </div>

      {/* ── Aspect Ratio Picker ── */}
      {onAspectRatioChange && (
        <div style={{
          display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 4,
          padding: '5px 8px', margin: '0 auto',
          maxWidth: compact ? undefined : `calc(50vh * ${targetRatio})`,
          width: '100%',
        }}>
          <span style={{
            fontSize: 10, fontWeight: 600, color: 'var(--text-muted, #888)',
            marginRight: 4, whiteSpace: 'nowrap', textTransform: 'uppercase',
            letterSpacing: '0.05em',
          }}>
            Ratio
          </span>
          {ASPECT_RATIO_OPTIONS.map((opt) => {
            const isActive = aspectRatio === opt.value;
            return (
              <button
                key={opt.label}
                onClick={(e) => { e.stopPropagation(); onAspectRatioChange(opt.value); }}
                style={{
                  display: 'flex', alignItems: 'center', gap: 3,
                  padding: '3px 8px', fontSize: 10, fontWeight: isActive ? 700 : 500,
                  background: isActive ? 'var(--accent-cyan, #0A84FF)' : 'var(--bg-elevated, rgba(0,0,0,0.04))',
                  color: isActive ? '#fff' : 'var(--text-secondary, #666)',
                  border: isActive ? '1px solid var(--accent-cyan, #0A84FF)' : '1px solid var(--border-dim, rgba(0,0,0,0.08))',
                  borderRadius: 'var(--radius-xs, 4px)', cursor: 'pointer',
                  transition: 'all 0.15s ease',
                  whiteSpace: 'nowrap',
                }}
                title={opt.value ? `${opt.value} crop` : 'Original aspect ratio'}
              >
                {opt.value && (
                  <svg width="12" height="12" viewBox="0 0 16 16" fill="none" style={{ opacity: 0.7 }}>
                    {opt.value === '16:9' && <rect x="1" y="3.5" width="14" height="9" rx="1.5" stroke="currentColor" strokeWidth="1.3" />}
                    {opt.value === '9:16' && <rect x="3.5" y="1" width="9" height="14" rx="1.5" stroke="currentColor" strokeWidth="1.3" />}
                    {opt.value === '1:1' && <rect x="2" y="2" width="12" height="12" rx="1.5" stroke="currentColor" strokeWidth="1.3" />}
                    {opt.value === '4:5' && <rect x="2.5" y="1.5" width="11" height="13" rx="1.5" stroke="currentColor" strokeWidth="1.3" />}
                  </svg>
                )}
                {opt.label}
              </button>
            );
          })}
        </div>
      )}

      {/* ── Active Settings Indicator ── */}
      {(segments.length > 0 || (aspectRatio && aspectRatio !== null) || Math.abs(speed - 1.0) > 0.001 || Math.abs(volume - 100) > 0.5 || isMuted) && (
        <div style={{
          display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 5,
          padding: '3px 8px', flexWrap: 'wrap',
        }}>
          {aspectRatio && (
            <span style={{
              display: 'inline-flex', alignItems: 'center', gap: 3,
              padding: '2px 6px', fontSize: 9, fontWeight: 600,
              background: 'rgba(10, 132, 255, 0.08)', color: 'var(--accent-cyan, #0A84FF)',
              border: '1px solid rgba(10, 132, 255, 0.2)', borderRadius: 3,
            }}>
              <svg width="10" height="10" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
                <rect x="1" y="3" width="14" height="10" rx="1.5" />
              </svg>
              {String(aspectRatio)}
            </span>
          )}
          {isMuted && (
            <span style={{
              display: 'inline-flex', alignItems: 'center', gap: 3,
              padding: '2px 6px', fontSize: 9, fontWeight: 600,
              background: 'rgba(255, 59, 48, 0.08)', color: '#FF3B30',
              border: '1px solid rgba(255, 59, 48, 0.2)', borderRadius: 3,
            }}>
              <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
                <path d="M11 5L6 9H2v6h4l5 4V5z" /><line x1="23" y1="9" x2="17" y2="15" /><line x1="17" y1="9" x2="23" y2="15" />
              </svg>
              Muted
            </span>
          )}
          {!isMuted && Math.abs(volume - 100) > 0.5 && (
            <span style={{
              display: 'inline-flex', alignItems: 'center', gap: 3,
              padding: '2px 6px', fontSize: 9, fontWeight: 600,
              background: volume > 100 ? 'rgba(255, 159, 10, 0.08)' : 'rgba(10, 132, 255, 0.08)',
              color: volume > 100 ? 'var(--accent-amber, #FF9F0A)' : 'var(--accent-cyan, #0A84FF)',
              border: `1px solid ${volume > 100 ? 'rgba(255, 159, 10, 0.2)' : 'rgba(10, 132, 255, 0.2)'}`,
              borderRadius: 3,
            }}>
              Vol {volume}%
            </span>
          )}
          {Math.abs(speed - 1.0) > 0.001 && (
            <span style={{
              display: 'inline-flex', alignItems: 'center', gap: 3,
              padding: '2px 6px', fontSize: 9, fontWeight: 600,
              background: 'rgba(175, 82, 222, 0.08)', color: '#AF52DE',
              border: '1px solid rgba(175, 82, 222, 0.2)', borderRadius: 3,
            }}>
              {speed}x Speed
            </span>
          )}
          {segments.length > 0 && (
            <span style={{
              display: 'inline-flex', alignItems: 'center', gap: 3,
              padding: '2px 6px', fontSize: 9, fontWeight: 600,
              background: 'rgba(48, 209, 88, 0.08)', color: '#30D158',
              border: '1px solid rgba(48, 209, 88, 0.2)', borderRadius: 3,
            }}>
              {segments.length} Segment{segments.length > 1 ? 's' : ''}
            </span>
          )}
        </div>
      )}

      {/* ── Timeline ── */}
      <div className="ve-timeline" style={videoReady ? undefined : { opacity: 0.3, pointerEvents: 'none' }}>
        {/* Ruler — clickable for Premiere-style seek */}
        <div
          className="ve-timeline__ruler"
          onPointerDown={(e) => {
            e.preventDefault();
            const track = timelineRef.current;
            if (!track || clipDur <= 0) return;
            const rect = track.getBoundingClientRect();
            const pct = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
            const time = clipStart + pct * clipDur;
            seekTo(time);
            // Enable drag-to-scrub from ruler
            const onMove = (ev) => {
              const t = getTimeFromPointer(ev.clientX);
              seekTo(t);
            };
            const onUp = () => {
              window.removeEventListener('pointermove', onMove);
              window.removeEventListener('pointerup', onUp);
            };
            window.addEventListener('pointermove', onMove);
            window.addEventListener('pointerup', onUp);
          }}
        >
          {rulerMarks.map((mark, i) => (
            <span key={i}>{mark}</span>
          ))}
        </div>

        <div style={{ position: 'relative' }}>
          <span className="ve-timeline__track-label">VIDEO</span>
        <div
          ref={timelineRef}
          className="ve-timeline__track"
          onPointerDown={onTimelinePointerDown}
          onPointerMove={(e) => {
            if (draggingHandle || draggingPlayhead) return;
            const track = timelineRef.current;
            if (!track || clipDur <= 0) return;
            const rect = track.getBoundingClientRect();
            const pct = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
            setHoverTime(pct * clipDur);
            setHoverX(e.clientX - rect.left);
            // Segment edge detection for cursor change
            onSegmentPointerMove(e);
          }}
          onPointerLeave={() => { setHoverTime(null); setSegHoverEdge(null); }}
          style={{
            cursor: segDrag
              ? (segDrag.mode === 'move' ? 'grabbing' : 'col-resize')
              : segHoverEdge
                ? (segHoverEdge.edge === 'center' ? 'grab' : 'col-resize')
                : 'pointer',
          }}
          onDoubleClick={(e) => {
            e.stopPropagation();
            const track = timelineRef.current;
            if (!track || clipDur <= 0) return;
            const rect = track.getBoundingClientRect();
            const pct = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
            const splitTime = clipStart + pct * clipDur;

            // If inside an existing segment, split it
            const existingSeg = segments.find(s => splitTime > s.start + 0.5 && splitTime < s.end - 0.5);
            if (existingSeg) {
              const seg1 = { ...existingSeg, end: splitTime };
              const newId = `seg_${segmentIdRef.current++}`;
              const seg2 = {
                ...existingSeg,
                id: newId,
                start: splitTime,
                label: `Segment ${segments.length + 1}`,
                color: SEGMENT_COLORS[segments.length % SEGMENT_COLORS.length],
              };
              const next = segments.map(s => s.id === existingSeg.id ? seg1 : s);
              next.push(seg2);
              next.sort((a, b) => a.start - b.start);
              setSegments(next);
              onSegmentsChange?.(next);
              setSelectedSegmentId(newId);
              return;
            }

            // Otherwise, create a new segment around the split point (±2s default)
            const halfDur = 2;
            const newSeg = {
              id: `seg_${segmentIdRef.current++}`,
              start: Math.max(trimmedStart, splitTime - halfDur),
              end: Math.min(trimmedEnd, splitTime + halfDur),
              volume: 100,
              muted: false,
              subtitlesEnabled: true,
              speed: 1.0,
              color: SEGMENT_COLORS[segments.length % SEGMENT_COLORS.length],
              label: `Segment ${segments.length + 1}`,
            };
            const next = [...segments, newSeg].sort((a, b) => a.start - b.start);
            setSegments(next);
            onSegmentsChange?.(next);
            setSelectedSegmentId(newSeg.id);
          }}
        >
          {/* Keyframe thumbnails */}
          <canvas ref={thumbnailCanvasRef} className="ve-timeline__thumbnails" />

          {/* Dimmed regions */}
          {leftTrimPct > 0 && (
            <div className="ve-timeline__dimmed-left" style={{ width: `${leftTrimPct}%` }} />
          )}
          {rightTrimPct > 0 && (
            <div className="ve-timeline__dimmed-right" style={{ width: `${rightTrimPct}%` }} />
          )}

          {/* Active trim region */}
          <div
            className="ve-timeline__trim-region"
            style={{
              left: `${leftTrimPct}%`,
              width: `${100 - leftTrimPct - rightTrimPct}%`,
            }}
          />

          {/* Progress fill */}
          <div
            className="ve-timeline__progress"
            style={{
              left: `${leftTrimPct}%`,
              width: `${Math.max(0, playheadPct - leftTrimPct)}%`,
            }}
          />

          {/* Hover time indicator */}
          {hoverTime !== null && !draggingHandle && !draggingPlayhead && (
            <>
              <div className="ve-timeline__hover-line" style={{ left: `${(hoverTime / clipDur) * 100}%` }} />
              <div className="ve-timeline__hover-tooltip" style={{ left: hoverX }}>
                {formatTimecode(hoverTime)}
              </div>
            </>
          )}

          {/* Playhead */}
          <div className="ve-timeline__playhead" style={{ left: `${playheadPct}%` }} />

          {/* Left trim handle */}
          <div
            className={`ve-trim-handle ve-trim-handle--left${draggingHandle === 'left' ? ' ve-trim-handle--active' : ''}`}
            style={{ left: `${leftTrimPct}%` }}
            onPointerDown={(e) => onTrimHandlePointerDown(e, 'left')}
          >
            <div className="ve-trim-handle__grip">
              <span /><span /><span />
            </div>
            {draggingHandle === 'left' && (
              <div className="ve-trim-tooltip">
                {formatTimecode(trimStartOffset)}
              </div>
            )}
          </div>

          {/* Right trim handle */}
          <div
            className={`ve-trim-handle ve-trim-handle--right${draggingHandle === 'right' ? ' ve-trim-handle--active' : ''}`}
            style={{ left: `calc(${100 - rightTrimPct}% - 10px)` }}
            onPointerDown={(e) => onTrimHandlePointerDown(e, 'right')}
          >
            <div className="ve-trim-handle__grip">
              <span /><span /><span />
            </div>
            {draggingHandle === 'right' && (
              <div className="ve-trim-tooltip">
                {formatTimecode(trimEndOffset)}
              </div>
            )}
          </div>

          {/* ── Segment overlays on timeline ── */}
          {segments.map(seg => {
          const segLeftPct = clipDur > 0 ? ((seg.start - clipStart) / clipDur) * 100 : 0;
          const segWidthPct = clipDur > 0 ? ((seg.end - seg.start) / clipDur) * 100 : 0;
          const isSegSelected = selectedSegmentId === seg.id;
          const isSegActive = activeSegmentId === seg.id;
          const segHexColor = seg.color || SEGMENT_COLORS[0];
          const segRgb = hexToRgbString(segHexColor);
          return (
            <div
              key={seg.id}
              className={`ve-timeline__segment${isSegSelected ? ' ve-timeline__segment--selected' : ''}${isSegActive ? ' ve-timeline__segment--active' : ''}`}
              style={{
                position: 'absolute',
                left: `${segLeftPct}%`,
                width: `${segWidthPct}%`,
                top: 0, bottom: 0,
                background: isSegSelected ? `rgba(${segRgb}, 0.3)` : isSegActive ? `rgba(${segRgb}, 0.2)` : `rgba(${segRgb}, 0.12)`,
                borderLeft: `2px solid rgba(${segRgb}, ${isSegSelected ? '1' : '0.5'})`,
                borderRight: `2px solid rgba(${segRgb}, ${isSegSelected ? '1' : '0.5'})`,
                pointerEvents: 'none',
                zIndex: 3,
              }}
            >
              {/* Segment label */}
              <span className="ve-timeline__segment-label" style={{
                position: 'absolute', top: 1, left: 3,
                display: 'flex', alignItems: 'center', gap: 2,
                fontSize: 7, fontWeight: 700, letterSpacing: '0.04em',
                color: segHexColor,
                textTransform: 'uppercase', lineHeight: 1, pointerEvents: 'none',
                whiteSpace: 'nowrap', opacity: isSegSelected ? 1 : 0.8,
              }}>
                <span style={{
                  display: 'inline-block', width: 5, height: 5,
                  borderRadius: '50%', background: segHexColor,
                }} />
                {String(seg.label || 'Segment')}
                {seg.muted ? ' (muted)' : ''}
                {!seg.subtitlesEnabled ? ' (no subs)' : ''}
                {seg.subjectTrackingEnabled === false ? ' (no tracking)' : ''}
              </span>
              {/* Bottom border indicator bar */}
              <div style={{
                position: 'absolute', bottom: 0, left: 0, right: 0, height: 2,
                background: segHexColor, opacity: isSegSelected ? 0.9 : 0.5,
              }} />
            </div>
          );
        })}
        </div>
        </div>{/* close VIDEO track-label wrapper */}

        {/* ── Waveform audio track (separate row) ── */}
        <div style={{ position: 'relative' }}>
          <span className="ve-timeline__track-label">AUDIO</span>
        <div className="ve-timeline__waveform-track" onPointerDown={onTimelinePointerDown}>
          <canvas ref={waveformCanvasRef} className="ve-timeline__waveform" />
          {/* Segment overlays on waveform track */}
          {segments.map(seg => {
            const segLeftPct = clipDur > 0 ? ((seg.start - clipStart) / clipDur) * 100 : 0;
            const segWidthPct = clipDur > 0 ? ((seg.end - seg.start) / clipDur) * 100 : 0;
            const segHexColor = seg.color || SEGMENT_COLORS[0];
            const hexToRgb = (hex) => { const r = parseInt(hex.slice(1,3),16); const g = parseInt(hex.slice(3,5),16); const b = parseInt(hex.slice(5,7),16); return `${r},${g},${b}`; };
            const segRgb = hexToRgb(segHexColor);
            return (
              <div key={`wf-${seg.id}`} style={{
                position: 'absolute', left: `${segLeftPct}%`, width: `${segWidthPct}%`,
                top: 0, bottom: 0, zIndex: 2, pointerEvents: 'none',
                background: seg.muted ? `rgba(${segRgb},0.15)` : `rgba(${segRgb},0.1)`,
                borderLeft: `1px solid rgba(${segRgb},0.4)`,
                borderRight: `1px solid rgba(${segRgb},0.4)`,
              }} />
            );
          })}
          <div className="ve-timeline__playhead" style={{ left: `${playheadPct}%` }} />
        </div>
        </div>{/* close AUDIO track-label wrapper */}

        {/* ── Segment chips row ── */}
        {(segments.length > 0 || hasTrim) && (
          <div style={{
            display: 'flex', alignItems: 'center', gap: 4, padding: '4px 0',
            flexWrap: 'wrap', fontSize: 10,
          }}>
            {hasTrim && (
              <button
                onClick={(e) => { e.stopPropagation(); addSegment(); }}
                className="ve-segment-chip ve-segment-chip--add"
                title="Save current selection as a segment"
              >
                + Add Segment
              </button>
            )}
            {segments.map(seg => {
              const isSelected = selectedSegmentId === seg.id;
              const segHexColor = seg.color || SEGMENT_COLORS[0];
              return (
                <div
                  key={seg.id}
                  className={`ve-segment-chip${isSelected ? ' ve-segment-chip--selected' : ''}`}
                  onClick={(e) => { e.stopPropagation(); setSelectedSegmentId(isSelected ? null : seg.id); }}
                  style={{
                    '--seg-color': segHexColor,
                    borderColor: isSelected ? segHexColor : undefined,
                  }}
                >
                  <span className="ve-segment-chip__dot" style={{ background: segHexColor }} />
                  <span className="ve-segment-chip__label">{String(seg.label || 'Segment')}</span>
                  <span className="ve-segment-chip__time">
                    {formatTimeShort(seg.start - clipStart)} – {formatTimeShort(seg.end - clipStart)}
                  </span>
                  {!isSelected && (
                    <span className="ve-segment-chip__summary">
                      {seg.muted ? 'muted' : `${seg.volume}%`}
                      {seg.speed && Math.abs(seg.speed - 1.0) > 0.001 ? ` · ${seg.speed}x` : ''}
                      {!seg.subtitlesEnabled ? ' · no subs' : ''}
                      {seg.subjectTrackingEnabled === false ? ' · no tracking' : ''}
                    </span>
                  )}
                </div>
              );
            })}
            {/* Quick action: Split at playhead */}
            {segments.length > 0 && (
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  // Simulate S key
                  const splitTime = videoRef.current?.currentTime ?? currentTime;
                  const existingSeg = segments.find(s => splitTime > s.start + 0.5 && splitTime < s.end - 0.5);
                  if (existingSeg) {
                    const seg1 = { ...existingSeg, end: splitTime };
                    const newId = `seg_${segmentIdRef.current++}`;
                    const seg2 = { ...existingSeg, id: newId, start: splitTime, label: `Segment ${segments.length + 1}`, color: SEGMENT_COLORS[segments.length % SEGMENT_COLORS.length] };
                    const next = segments.map(s => s.id === existingSeg.id ? seg1 : s);
                    next.push(seg2);
                    next.sort((a, b) => a.start - b.start);
                    setSegments(next);
                    onSegmentsChange?.(next);
                    setSelectedSegmentId(newId);
                  }
                }}
                className="ve-segment-chip ve-segment-chip--action"
                title="Split segment at playhead (S)"
              >
                Split at Playhead
              </button>
            )}
          </div>
        )}

        {/* ── Segment Inspector Panel ── */}
        {selectedSegment && (() => {
          const seg = selectedSegment;
          const segHexColor = seg.color || SEGMENT_COLORS[0];
          const isEditingStart = editingSegTime?.segId === seg.id && editingSegTime?.field === 'start';
          const isEditingEnd = editingSegTime?.segId === seg.id && editingSegTime?.field === 'end';
          const segDurSec = seg.end - seg.start;
          const isEditingLabel = editingSegLabel === seg.id;
          return (
            <div className="ve-segment-inspector" style={{ '--seg-color': segHexColor }} onClick={(e) => e.stopPropagation()}>
              {/* Header row */}
              <div className="ve-segment-inspector__header">
                <span className="ve-segment-inspector__color-dot" style={{ background: segHexColor }} />
                {isEditingLabel ? (
                  <input
                    ref={segLabelInputRef}
                    className="ve-segment-inspector__label-input"
                    type="text"
                    value={segLabelInput}
                    onChange={(e) => setSegLabelInput(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') {
                        if (segLabelInput.trim()) updateSegment(seg.id, { label: segLabelInput.trim() });
                        setEditingSegLabel(null);
                      } else if (e.key === 'Escape') {
                        setEditingSegLabel(null);
                      }
                      e.stopPropagation();
                    }}
                    onBlur={() => {
                      if (segLabelInput.trim()) updateSegment(seg.id, { label: segLabelInput.trim() });
                      setEditingSegLabel(null);
                    }}
                    autoFocus
                  />
                ) : (
                  <span
                    className="ve-segment-inspector__label"
                    onClick={(e) => {
                      e.stopPropagation();
                      setSegLabelInput(seg.label || '');
                      setEditingSegLabel(seg.id);
                      setTimeout(() => segLabelInputRef.current?.select(), 0);
                    }}
                    title="Click to rename"
                  >
                    {String(seg.label || 'Segment')}
                  </span>
                )}
                {/* Time range */}
                <div className="ve-segment-inspector__time-range">
                  {isEditingStart ? (
                    <input
                      ref={segTimeInputRef}
                      type="text" value={segTimeInput}
                      onChange={(e) => setSegTimeInput(e.target.value)}
                      onKeyDown={(e) => { if (e.key === 'Enter') commitSegTimeEdit(); else if (e.key === 'Escape') setEditingSegTime(null); e.stopPropagation(); }}
                      onBlur={commitSegTimeEdit}
                      className="ve-segment-inspector__time-input"
                      autoFocus
                    />
                  ) : (
                    <span
                      className="ve-segment-inspector__time-editable"
                      onClick={(e) => {
                        e.stopPropagation();
                        setSegTimeInput(formatTimeShort(seg.start - clipStart));
                        setEditingSegTime({ segId: seg.id, field: 'start' });
                        setTimeout(() => segTimeInputRef.current?.select(), 0);
                      }}
                    >
                      {formatTimeShort(seg.start - clipStart)}
                    </span>
                  )}
                  <span className="ve-segment-inspector__time-sep">—</span>
                  {isEditingEnd ? (
                    <input
                      ref={segTimeInputRef}
                      type="text" value={segTimeInput}
                      onChange={(e) => setSegTimeInput(e.target.value)}
                      onKeyDown={(e) => { if (e.key === 'Enter') commitSegTimeEdit(); else if (e.key === 'Escape') setEditingSegTime(null); e.stopPropagation(); }}
                      onBlur={commitSegTimeEdit}
                      className="ve-segment-inspector__time-input"
                      autoFocus
                    />
                  ) : (
                    <span
                      className="ve-segment-inspector__time-editable"
                      onClick={(e) => {
                        e.stopPropagation();
                        setSegTimeInput(formatTimeShort(seg.end - clipStart));
                        setEditingSegTime({ segId: seg.id, field: 'end' });
                        setTimeout(() => segTimeInputRef.current?.select(), 0);
                      }}
                    >
                      {formatTimeShort(seg.end - clipStart)}
                    </span>
                  )}
                  <span className="ve-segment-inspector__time-dur">({formatTimeShort(segDurSec)})</span>
                </div>
                {/* Delete button */}
                <button
                  className="ve-segment-inspector__delete"
                  onClick={(e) => {
                    e.stopPropagation();
                    if (deleteConfirmId === seg.id) {
                      removeSegment(seg.id);
                      setSelectedSegmentId(null);
                      setDeleteConfirmId(null);
                    } else {
                      setDeleteConfirmId(seg.id);
                      setTimeout(() => setDeleteConfirmId(null), 2000);
                    }
                  }}
                  title={deleteConfirmId === seg.id ? 'Click again to confirm' : 'Remove segment'}
                >
                  {deleteConfirmId === seg.id ? 'Remove?' : '×'}
                </button>
              </div>
              {/* Controls row */}
              <div className="ve-segment-inspector__controls">
                {/* Volume */}
                <div className="ve-segment-inspector__volume-row">
                  <span className="ve-segment-inspector__control-label">Volume</span>
                  <input
                    type="range" min="0" max="200" step="1"
                    value={seg.muted ? 0 : seg.volume}
                    onChange={(e) => {
                      const v = parseInt(e.target.value);
                      updateSegment(seg.id, { volume: v, muted: v === 0 });
                    }}
                    className="ve-segment-inspector__volume-slider"
                    style={{
                      background: `linear-gradient(to right, ${segHexColor} ${(seg.muted ? 0 : seg.volume) / 2}%, var(--ve-slider-track, rgba(0,0,0,0.12)) ${(seg.muted ? 0 : seg.volume) / 2}%)`,
                    }}
                  />
                  <span className="ve-segment-inspector__volume-value">{seg.muted ? 0 : seg.volume}%</span>
                </div>
                {/* Speed pills */}
                <div className="ve-segment-inspector__speed-row">
                  <span className="ve-segment-inspector__control-label">Speed</span>
                  <div className="ve-segment-inspector__speed-pills">
                    {[0.5, 0.75, 1.0, 1.25, 1.5, 2.0].map(spd => (
                      <button
                        key={spd}
                        className={`ve-segment-inspector__speed-pill${Math.abs((seg.speed || 1) - spd) < 0.001 ? ' ve-segment-inspector__speed-pill--active' : ''}`}
                        onClick={(e) => { e.stopPropagation(); updateSegment(seg.id, { speed: spd }); }}
                        style={Math.abs((seg.speed || 1) - spd) < 0.001 ? { background: segHexColor, color: '#fff', borderColor: segHexColor } : {}}
                      >
                        {String(spd)}x
                      </button>
                    ))}
                  </div>
                </div>
                {/* Toggles */}
                <div className="ve-segment-inspector__toggles">
                  <label className="ve-segment-inspector__toggle" onClick={(e) => e.stopPropagation()}>
                    <input
                      type="checkbox" checked={seg.subtitlesEnabled !== false}
                      onChange={() => toggleSegmentSubs(seg.id)}
                    />
                    <span>Subtitles</span>
                  </label>
                  <label className="ve-segment-inspector__toggle" onClick={(e) => e.stopPropagation()}>
                    <input
                      type="checkbox" checked={seg.muted}
                      onChange={() => toggleSegmentMute(seg.id)}
                    />
                    <span>Muted</span>
                  </label>
                  <label className="ve-segment-inspector__toggle" onClick={(e) => e.stopPropagation()}>
                    <input
                      type="checkbox" checked={seg.subjectTrackingEnabled !== false}
                      onChange={() => toggleSegmentTracking(seg.id)}
                    />
                    <span>Subject Tracking</span>
                  </label>
                </div>
              </div>
              {/* Footer */}
              <div className="ve-segment-inspector__actions">
                <button
                  className="ve-segment-inspector__done"
                  onClick={(e) => { e.stopPropagation(); deselectSegment(); }}
                >
                  Done
                </button>
              </div>
            </div>
          );
        })()}
      </div>

      {/* ── Speaker Color Editor (below timeline, above controls) ── */}
      {!isMobile && speakers && speakers.length > 0 && settings?.subtitlesEnabled && (
        <div style={{
          display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap',
          padding: '8px 14px',
          background: 'var(--ve-chrome-bg)',
          borderTop: '1px solid var(--ve-chrome-border)',
          borderBottom: '1px solid var(--ve-chrome-border)',
        }}>
          <span style={{
            fontSize: 11, fontWeight: 700, letterSpacing: '0.04em',
            textTransform: 'uppercase', color: 'var(--ve-text)',
            marginRight: 4,
          }}>
            Speaker Colors
          </span>
          {speakers.map((spk, spkIdx) => {
            const color = settings?.speakerColors?.[spk] || DEFAULT_SPEAKER_PALETTE[spkIdx % DEFAULT_SPEAKER_PALETTE.length];
            const displayName = String(speakerNames?.[spk] || spk || '');
            return (
              <label key={spk} style={{
                display: 'inline-flex', alignItems: 'center', gap: 6,
                cursor: 'pointer',
                padding: '4px 8px',
                borderRadius: 6,
                background: 'var(--ve-track-bg, rgba(0,0,0,0.04))',
                border: '1px solid var(--ve-chrome-border)',
                transition: 'all 0.15s ease',
              }}>
                <input
                  type="color"
                  value={color}
                  onChange={(e) => {
                    e.stopPropagation();
                    if (onSettingsChange) {
                      const newColors = { ...(settings?.speakerColors || {}), [spk]: e.target.value };
                      onSettingsChange({ ...settings, speakerColors: newColors });
                    }
                  }}
                  onClick={(e) => e.stopPropagation()}
                  style={{
                    width: 24, height: 24, padding: 0,
                    border: '2px solid var(--border, #ddd)',
                    borderRadius: 6, cursor: 'pointer', background: 'none',
                  }}
                  title={`Color for ${displayName}`}
                />
                <span style={{
                  color: 'var(--ve-text)', fontSize: 12, fontWeight: 500,
                  maxWidth: 120, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                }}>
                  {displayName}
                </span>
              </label>
            );
          })}
        </div>
      )}

      {/* ── Controls bar ── */}
      <div className={`ve-controls${compact ? ' ve-controls--compact' : ''}${effectiveSegment ? ' ve-controls--segment-mode' : ''}`}
        style={videoReady ? undefined : { opacity: 0.4, pointerEvents: 'none' }}
      >
        {/* Transport row: centered on all devices */}
        <div className="ve-controls__transport">
          <button className="ve-btn" onClick={(e) => { e.stopPropagation(); skipTime(-5); }} title="Back 5s (J)" disabled={!videoReady}>
            <Icon.SkipBack />
          </button>

          <button className="ve-btn ve-btn--play" onClick={(e) => { e.stopPropagation(); togglePlay(); }} title="Play/Pause (Space)" disabled={!videoReady}>
            {playing ? <Icon.Pause /> : <Icon.Play />}
          </button>

          <button className="ve-btn" onClick={(e) => { e.stopPropagation(); skipTime(5); }} title="Forward 5s (L)" disabled={!videoReady}>
            <Icon.SkipForward />
          </button>

          {editingTimecode ? (
            <input
              ref={timecodeInputRef}
              className="ve-timecode ve-timecode--input"
              type="text"
              inputMode="numeric"
              value={timecodeInput}
              onChange={(e) => setTimecodeInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') {
                  const parsed = parseTimecodeInput(timecodeInput);
                  if (parsed != null) seekTo(trimmedStart + parsed);
                  setEditingTimecode(false);
                } else if (e.key === 'Escape') {
                  setEditingTimecode(false);
                }
                e.stopPropagation();
              }}
              onBlur={() => {
                const parsed = parseTimecodeInput(timecodeInput);
                if (parsed != null) seekTo(trimmedStart + parsed);
                setEditingTimecode(false);
              }}
              onClick={(e) => e.stopPropagation()}
              placeholder="0:00"
              autoFocus
            />
          ) : (
            <span
              className="ve-timecode"
              onClick={(e) => {
                e.stopPropagation();
                setTimecodeInput(formatTimecode(elapsed).replace(/\..*$/, ''));
                setEditingTimecode(true);
                setTimeout(() => timecodeInputRef.current?.select(), 0);
              }}
              title="Click to type a time"
            >
              {showTimecodeRemaining
                ? `-${formatTimecode(Math.max(0, effectiveDuration - mediaToWallClock(elapsed)))}`
                : formatTimecode(mediaToWallClock(elapsed))}
              {' / '}
              {formatTimecode(effectiveDuration)}
            </span>
          )}
        </div>

        {/* Center: trim apply */}
        <div className="ve-controls__center">
          {hasTrim && (
            trimApplied ? (
              <span className="ve-trim-applied" title="Trim applied to preview and export">
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
                  <polyline points="20 6 9 17 4 12" />
                </svg>
                Trim Applied · {formatTimeShort(effectiveDuration)}
              </span>
            ) : (
              <button className="ve-apply-trim" onClick={(e) => { e.stopPropagation(); handleApplyTrim(); }} title="Apply trim to preview and export">
                Apply Trim · {formatTimeShort(effectiveDuration)}
              </button>
            )
          )}
        </div>

        {/* Right: volume + speed + fullscreen */}
        <div className="ve-controls__right">
          {/* Volume — segment-aware */}
          <div className={`ve-volume${isMobile ? '' : ''}`} onClick={(e) => e.stopPropagation()}>
            <button className="ve-btn" onClick={toggleMute} title={effectiveSegment ? `${effectiveSegment.muted ? 'Unmute' : 'Mute'} segment` : isMuted ? 'Unmute (M)' : 'Mute (M)'}>
              <VolumeIcon />
            </button>
            <div className="ve-volume__slider-wrap">
              {(() => {
                const dispVol = displaySegment ? (displaySegment.muted ? 0 : displaySegment.volume) : (isMuted ? 0 : volume);
                const dispMax = displaySegment ? displaySegment.volume : volume;
                return (
                  <>
                    <input
                      type="range"
                      className="ve-volume__slider"
                      min="0"
                      max="200"
                      step="1"
                      value={dispVol}
                      onChange={(e) => {
                        const v = parseInt(e.target.value);
                        if (controlTargetSegment) {
                          updateSegment(controlTargetSegment.id, { volume: v, muted: v === 0 });
                        } else {
                          setVolume(v);
                          if (isMuted && v > 0) setIsMuted(false);
                        }
                      }}
                      style={{
                        background: `linear-gradient(to right, ${
                          dispMax > 100 ? 'var(--accent-amber, #FF9F0A)' : 'var(--accent-cyan, #0A84FF)'
                        } ${dispVol / 2}%, var(--ve-slider-track, rgba(0,0,0,0.12)) ${dispVol / 2}%)`,
                      }}
                    />
                    <span className={`ve-volume__label${dispMax > 150 ? ' ve-volume__label--warn' : ''}`}>
                      {String(dispVol)}%
                    </span>
                  </>
                );
              })()}
            </div>
          </div>

          {/* Speed — segment-aware */}
          {(() => {
            const dispSpeed = displaySegment ? (displaySegment.speed || 1.0) : speed;
            return (
            <div className="ve-speed" onClick={(e) => e.stopPropagation()}>
              <button
                className={`ve-speed__btn${dispSpeed !== 1.0 ? ' ve-speed__btn--active' : ''}`}
                onClick={() => setShowSpeedMenu((v) => !v)}
                title={displaySegment ? 'Segment playback speed' : 'Playback speed'}
              >
                {dispSpeed}x
              </button>
              {showSpeedMenu && (
                <div className="ve-speed__dropdown" onClick={(e) => e.stopPropagation()}>
                  {SPEED_PRESETS.map((p) => (
                    <button
                      key={p.value}
                      className={`ve-speed__option${dispSpeed === p.value ? ' ve-speed__option--current' : ''}`}
                      onClick={() => selectSpeed(p.value)}
                    >
                      {p.value}x
                      {p.label && <span className="ve-speed__option-label">{p.label}</span>}
                    </button>
                  ))}
                </div>
              )}
            </div>
            );
          })()}

          {/* Active segment indicator */}
          {displaySegment && (
            <span style={{
              fontSize: 9, fontFamily: 'var(--font-mono, monospace)', padding: '2px 6px',
              borderRadius: 4,
              background: `${displaySegment.color || SEGMENT_COLORS[0]}18`,
              color: displaySegment.color || SEGMENT_COLORS[0],
              border: `1px solid ${displaySegment.color || SEGMENT_COLORS[0]}40`,
              whiteSpace: 'nowrap', lineHeight: 1.3,
            }}>
              <span style={{ display: 'inline-block', width: 5, height: 5, borderRadius: '50%', background: displaySegment.color || SEGMENT_COLORS[0], marginRight: 3, verticalAlign: 'middle' }} />
              {displaySegment.label || 'Segment'} · {displaySegment.muted ? 'muted' : `${displaySegment.volume}%`}
              {displaySegment.speed && Math.abs(displaySegment.speed - 1.0) > 0.001 ? ` · ${displaySegment.speed}x` : ''}
            </span>
          )}

          {/* Fullscreen */}
          <button className="ve-btn" onClick={(e) => { e.stopPropagation(); toggleFullscreen(); }} title="Fullscreen">
            {isFullscreen ? <Icon.ExitFullscreen /> : <Icon.Fullscreen />}
          </button>
        </div>
      </div>

      {/* ── Multi-Track Editor Toggle ── */}
      {!compact && (
        <div className="ve-multitrack-toggle">
          <button
            className={`ve-multitrack-toggle__btn${showMultiTrack ? ' ve-multitrack-toggle__btn--active' : ''}${isProcessing ? ' ve-multitrack-toggle__btn--disabled' : ''}`}
            onClick={(e) => {
              e.stopPropagation();
              if (isProcessing) return;
              setShowMultiTrack(v => {
                if (!v) setShowProperties(true); // auto-show properties when opening
                return !v;
              });
            }}
            title={isProcessing ? 'Available after analysis completes' : 'Toggle multi-track timeline editor'}
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <rect x="1" y="3" width="22" height="4" rx="1" />
              <rect x="1" y="10" width="22" height="4" rx="1" />
              <rect x="1" y="17" width="22" height="4" rx="1" />
            </svg>
            {showMultiTrack ? 'Hide Multi-Track' : 'Multi-Track Editor'}
          </button>
          {showMultiTrack && (
            <>
              <button
                className={`ve-multitrack-toggle__btn ve-multitrack-toggle__btn--sub${showMediaLibrary ? ' ve-multitrack-toggle__btn--active' : ''}`}
                onClick={(e) => { e.stopPropagation(); setShowMediaLibrary(v => !v); }}
              >
                {showMediaLibrary ? 'Hide Media' : 'Media Library'}
              </button>
              <button
                className={`ve-multitrack-toggle__btn ve-multitrack-toggle__btn--sub${showProperties ? ' ve-multitrack-toggle__btn--active' : ''}`}
                onClick={(e) => { e.stopPropagation(); setShowProperties(v => !v); }}
              >
                {showProperties ? 'Hide Properties' : 'Properties'}
              </button>
            </>
          )}
        </div>
      )}

      {/* ── Multi-Track Editor Panels ── */}
      {showMultiTrack && (
        <div className="ve-multitrack" style={isEncoding ? { position: 'relative' } : undefined}>
          {isEncoding && (
            <div className="ve-multitrack__encoding-overlay">
              <div className="ve-multitrack__encoding-label">
                Encoding in progress...
              </div>
            </div>
          )}
          {/* ToolBar */}
          <div className="ve-multitrack__toolbar">
            <EditorErrorBoundary name="ToolBar" compact>
              <ToolBar />
            </EditorErrorBoundary>
            <div className="ve-multitrack__toolbar-actions">
            </div>
          </div>

          {/* Main content area */}
          <div className="ve-multitrack__content">
            {/* Media Library Sidebar */}
            {showMediaLibrary && (
              <div className="ve-multitrack__sidebar ve-multitrack__sidebar--left">
                <div className="ve-multitrack__sidebar-header">
                  <span>Media Library</span>
                  <button
                    className="ve-btn"
                    onClick={() => setShowMediaLibrary(false)}
                    style={{ minWidth: 24, minHeight: 24, fontSize: 12 }}
                  >
                    ✕
                  </button>
                </div>
                <MediaUploader jobId={jobId} />
              </div>
            )}

            {/* Center area: timeline always fills available space */}
            <div className="ve-multitrack__center">
              {/* Timeline */}
              <div className="ve-multitrack__timeline">
                <EditorErrorBoundary name="Timeline" compact>
                  <Timeline
                    compact={compact}
                    onSeek={handleTimelineSeek}
                    onItemSelect={handleTimelineItemSelect}
                    onSubtitleVisibilityChange={handleTimelineSubtitleVisibility}
                  />
                </EditorErrorBoundary>
              </div>
            </div>

            {/* Right Sidebar: Properties + Effects + Transitions stacked */}
            {showProperties && (
              <div className="ve-multitrack__sidebar ve-multitrack__sidebar--right">
                <div className="ve-multitrack__sidebar-header">
                  <span>Properties</span>
                  <button
                    className="ve-btn"
                    onClick={() => setShowProperties(false)}
                    style={{ minWidth: 24, minHeight: 24, fontSize: 12 }}
                  >
                    ✕
                  </button>
                </div>
                <EditorErrorBoundary name="Properties" compact>
                  <PropertiesPanel compact={compact} settings={settings} onSettingsChange={onSettingsChange} />
                </EditorErrorBoundary>

                {/* Effects Section (collapsible, inside sidebar) */}
                {showEffectsPanel && (
                  <div className="ve-multitrack__sidebar-section">
                    <div className="ve-multitrack__sidebar-header">
                      <span>Effects</span>
                      <button className="ve-btn" onClick={() => setShowEffectsPanel(false)} style={{ minWidth: 24, minHeight: 24, fontSize: 12 }}>✕</button>
                    </div>
                    <EditorErrorBoundary name="Effects" compact>
                      <EffectsPanel />
                    </EditorErrorBoundary>
                  </div>
                )}

                {/* Transitions Section (collapsible, inside sidebar) */}
                {showTransitions && (
                  <div className="ve-multitrack__sidebar-section">
                    <div className="ve-multitrack__sidebar-header">
                      <span>Transitions</span>
                      <button className="ve-btn" onClick={() => setShowTransitions(false)} style={{ minWidth: 24, minHeight: 24, fontSize: 12 }}>✕</button>
                    </div>
                    <TransitionPicker />
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      )}

      {/* ── Export Dialog ── */}
      {showExportDialog && (
        <ExportDialog
          onClose={() => setShowExportDialog(false)}
          jobId={jobId}
          clipId={clipId}
          clipTitle={title}
          settings={settings}
          startTime={clipStart}
          endTime={effectiveClipEnd}
          aspectRatio={aspectRatio}
          transcript={transcript}
          scenes={scenes}
          sourceWidth={sourceWidth}
          sourceHeight={sourceHeight}
          subjectX={subjectX}
          onServerExport={handleServerExport}
        />
      )}

      {/* ── Keyboard shortcuts hint ── */}
      {!compact && (
        <div className="ve-shortcuts">
          <span><kbd>Space</kbd> Play/Pause</span>
          <span><kbd>J</kbd>/<kbd>K</kbd>/<kbd>L</kbd> Shuttle</span>
          <span><kbd>{'\u2190'}</kbd>/<kbd>{'\u2192'}</kbd> Frame</span>
          <span><kbd>M</kbd> Mute</span>
          <span><kbd>S</kbd> Split/Add Segment</span>
          <span><kbd>Del</kbd> Remove Segment</span>
          <span><kbd>Esc</kbd> Deselect</span>
          <span><kbd>Tab</kbd> Cycle Segments</span>
          {showMultiTrack && <span><kbd>Ctrl+Z</kbd>/<kbd>Ctrl+Shift+Z</kbd> Undo/Redo</span>}
        </div>
      )}
    </div>
  );
}
