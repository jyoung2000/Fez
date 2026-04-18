import React, { useRef, useEffect, useCallback, useMemo, useState } from 'react';
import useTimelineStore, { getMaxItemDuration, hashGroupId } from '../stores/timelineStore';
import {
  computeThumbStops,
  ensureThumbnail,
  getCachedThumbnail,
} from '../utils/filmstrip';

// ── Constants ────────────────────────────────────────────────────────────────
const TRACK_HEIGHT = 64;
const TRACK_GAP = 0;
const LABEL_WIDTH = 120;
const HANDLE_WIDTH = 6;
const HANDLE_HIT_AREA = 12;
const RULER_HEIGHT = 28;
const PLAYHEAD_GRAB_WIDTH = 16; // px on each side of playhead for grab detection

const TRACK_COLORS = {
  video: '#3B82F6',
  overlay: '#F59E0B',
  audio: '#10B981',
  subtitle: '#8B5CF6',
  text: '#EC4899',
  shape: '#F97316',
  crop: '#06B6D4',
};

const TRACK_ICONS = {
  video: '\uD83C\uDFAC',
  overlay: '\uD83D\uDDBC',
  audio: '\uD83C\uDFB5',
  subtitle: '\uD83D\uDCAC',
  text: 'T',
  shape: '\u25A1',
  crop: '\u2702',
};

// Crop segment cluster colors
const CROP_CLUSTER_COLORS = [
  '#3B82F6', // blue — speaker 0
  '#10B981', // green — speaker 1
  '#F59E0B', // amber — speaker 2
  '#EC4899', // pink — speaker 3
  '#8B5CF6', // purple — manual override / unknown
];

function formatTime(s) {
  if (!s || isNaN(s) || s < 0) return '0:00';
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return `${m}:${sec.toString().padStart(2, '0')}`;
}

function formatTimeMs(s) {
  if (!s || isNaN(s) || s < 0) return '0:00.00';
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  const ms = Math.floor((s % 1) * 100);
  return `${m}:${sec.toString().padStart(2, '0')}.${ms.toString().padStart(2, '0')}`;
}

/**
 * Find the nearest snap target for a given time value.
 * Returns { snappedTime, snapTarget } or null if no snap found.
 */
function findSnapTarget(candidateTime, items, excludeItemId, playhead, duration, pps, sceneCuts = null) {
  // Adaptive threshold: larger zone when zoomed out, smaller when zoomed in
  const threshold = Math.max(5, Math.min(12, 600 / pps));

  const targets = new Set();
  targets.add(0);
  targets.add(playhead);
  if (duration > 0) targets.add(duration);

  for (const item of items) {
    if (item.id === excludeItemId) continue;
    targets.add(item.start);
    targets.add(item.end);
  }

  // Snap to scene cuts when the timeline knows about them. This makes
  // trim-handle dragging "click" into the cut detected by the analysis
  // pipeline, matching the expectation that a user would tend to cut
  // **on** a shot boundary rather than 200 ms inside it.
  if (Array.isArray(sceneCuts)) {
    for (const t of sceneCuts) {
      if (Number.isFinite(t)) targets.add(t);
    }
  }

  let bestDist = Infinity;
  let bestTarget = null;

  for (const target of targets) {
    const distPx = Math.abs((candidateTime - target) * pps);
    if (distPx < threshold && distPx < bestDist) {
      bestDist = distPx;
      bestTarget = target;
    }
  }

  if (bestTarget !== null) {
    return { snappedTime: bestTarget, snapTarget: bestTarget };
  }
  return null;
}

export default function Timeline({ compact = false, onSeek, onItemSelect, onSubtitleVisibilityChange }) {
  const canvasRef = useRef(null);
  const containerRef = useRef(null);

  // Stable ref so async filmstrip generations can request a repaint
  // through the latest ``draw`` callback identity.
  const requestRedrawRef = useRef(null);

  // Subscribe per-slice for STATE that needs to trigger re-renders.
  // ACTIONS are pulled via ``_store.getState()`` below — they are
  // stable references for the lifetime of the store, so subscribing
  // to them was 25+ useless ``useSyncExternalStore`` registrations
  // each waking the component on every state change.
  const tracks = useTimelineStore((s) => s.tracks);
  const items = useTimelineStore((s) => s.items);
  const cropSegments = useTimelineStore((s) => s.cropSegments);
  const selectedCropSegmentId = useTimelineStore((s) => s.selectedCropSegmentId);
  const playhead = useTimelineStore((s) => s.playhead);
  const duration = useTimelineStore((s) => s.duration);
  const zoom = useTimelineStore((s) => s.zoom);
  const scrollX = useTimelineStore((s) => s.scrollX);
  const snapEnabled = useTimelineStore((s) => s.snapEnabled);
  // Subscribed so a snap-guide change triggers a redraw via the
  // ``[draw]`` effect; ``draw`` reads the live value below.
  const snapLine = useTimelineStore((s) => s.snapLine);
  const sceneCuts = useTimelineStore((s) => s.sceneCuts);
  const selectedItemId = useTimelineStore((s) => s.selectedItemId);
  const selectedItemIds = useTimelineStore((s) => s.selectedItemIds);
  const activeTool = useTimelineStore((s) => s.activeTool);
  const isPlaying = useTimelineStore((s) => s.isPlaying);
  const segments = useTimelineStore((s) => s.segments);
  const hasOriginalSubtitles = useTimelineStore(
    (s) => (s._originalSubtitles || []).length > 0,
  );

  // Action references — stable across renders, so we read them off the
  // store reference once and avoid the per-action selector subscriptions.
  const _store = useTimelineStore;
  const selectCropSegment = _store.getState().selectCropSegment;
  const setPlayhead = _store.getState().setPlayhead;
  const setIsScrubbing = _store.getState().setIsScrubbing;
  const setZoom = _store.getState().setZoom;
  const setScrollX = _store.getState().setScrollX;
  const setSelectedItemId = _store.getState().setSelectedItemId;
  const setSelectedItemIds = _store.getState().setSelectedItemIds;
  const toggleSelectedItem = _store.getState().toggleSelectedItem;
  const updateItem = _store.getState().updateItem;
  const addItem = _store.getState().addItem;
  const splitItem = _store.getState().splitItem;
  const removeItem = _store.getState().removeItem;
  const toggleSnap = _store.getState().toggleSnap;
  const addTrack = _store.getState().addTrack;
  const toggleTrackVisibility = _store.getState().toggleTrackVisibility;
  const toggleTrackMute = _store.getState().toggleTrackMute;
  const toggleTrackLock = _store.getState().toggleTrackLock;
  const updateTrack = _store.getState().updateTrack;
  const reorderTracks = _store.getState().reorderTracks;
  const resetSubtitleTimings = _store.getState().resetSubtitleTimings;
  const groupItems = _store.getState().groupItems;
  const ungroupItems = _store.getState().ungroupItems;

  // Track drag-to-reorder state
  const [dragTrackIdx, setDragTrackIdx] = useState(null);
  const [dragOverTrackIdx, setDragOverTrackIdx] = useState(null);
  const [renamingTrackId, setRenamingTrackId] = useState(null);

  const [isDragging, setIsDragging] = useState(false);
  const [dragInfo, setDragInfo] = useState(null);
  const [hoverTime, setHoverTime] = useState(null);
  const [contextMenu, setContextMenu] = useState(null);
  const [showAddTrack, setShowAddTrack] = useState(false);
  const [spaceHeld, setSpaceHeld] = useState(false);

  // Spacebar hold for pan mode
  useEffect(() => {
    const onKeyDown = (e) => { if (e.code === 'Space' && !e.repeat) setSpaceHeld(true); };
    const onKeyUp = (e) => { if (e.code === 'Space') setSpaceHeld(false); };
    window.addEventListener('keydown', onKeyDown);
    window.addEventListener('keyup', onKeyUp);
    return () => { window.removeEventListener('keydown', onKeyDown); window.removeEventListener('keyup', onKeyUp); };
  }, []);

  const basePPS = compact ? 40 : 60;
  const pps = basePPS * zoom;

  // Ref for playhead pixel position — avoids stale closures in event handlers
  // without recreating callbacks on every playhead update (which happens every frame).
  const playheadRef = useRef(playhead);
  playheadRef.current = playhead;
  const ppsRef = useRef(pps);
  ppsRef.current = pps;

  // ── Scrub-flush state lives on refs, not effect-local variables ──
  // The playhead drag effect previously stored ``_scrubRaf`` and
  // ``_scrubPending`` as ``let`` bindings inside the effect body. That
  // worked fine when the effect never re-ran mid-drag — but the parent
  // (VideoEditor) passes ``onSeek`` as a fresh inline arrow on every
  // render, which bumps Timeline's drag effect deps every frame and
  // tears the effect down. Each teardown drops the pending rAF handle
  // and the buffered scrub time with it → ``setPlayhead`` was never
  // flushed → playhead looked frozen during scrub.
  //
  // Refs survive effect rebuilds. The new effect body picks up the
  // same ``_scrubPending`` the old body queued, the same rAF handle
  // gets cancelled in cleanup, and the flush either fires or is
  // deterministically aborted on pointerup.
  const _scrubRafRef = useRef(0);
  const _scrubPendingRef = useRef(null);
  // Also ref the parent callbacks so the effect can read the latest
  // ``onSeek`` / ``setPlayhead`` without listing them as deps. Reading
  // through a ref is a cheap way to say "use the freshest value, but
  // don't trigger re-mounts when the value changes".
  const onSeekRef = useRef(onSeek);
  onSeekRef.current = onSeek;
  const setPlayheadRef = useRef(setPlayhead);
  setPlayheadRef.current = setPlayhead;

  // ── Canvas rendering ──────────────────────────────────────────────────────
  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext && canvas.getContext('2d');
    // ``getContext`` can return null in restricted contexts (private
    // mode quirks, very old browsers, hardware-acceleration disabled).
    // Without this guard ``ctx.scale`` throws and the timeline goes
    // completely blank with no error boundary catch.
    if (!ctx) return;
    // ``roundRect`` polyfill — Safari < 16.4 and older Chromium don't
    // implement it, and this canvas calls it 7+ times per draw. Without
    // the polyfill the FIRST clip body draw throws ``TypeError:
    // ctx.roundRect is not a function`` and the entire multi-track view
    // blanks out (the parent error boundary then shows the failure UI).
    if (typeof ctx.roundRect !== 'function') {
      // eslint-disable-next-line no-param-reassign
      ctx.roundRect = function (x, y, w, h, r) {
        const radius = typeof r === 'number' ? Math.max(0, Math.min(r, Math.min(w, h) / 2)) : 0;
        if (radius <= 0) {
          this.rect(x, y, w, h);
          return this;
        }
        this.moveTo(x + radius, y);
        this.lineTo(x + w - radius, y);
        this.quadraticCurveTo(x + w, y, x + w, y + radius);
        this.lineTo(x + w, y + h - radius);
        this.quadraticCurveTo(x + w, y + h, x + w - radius, y + h);
        this.lineTo(x + radius, y + h);
        this.quadraticCurveTo(x, y + h, x, y + h - radius);
        this.lineTo(x, y + radius);
        this.quadraticCurveTo(x, y, x + radius, y);
        return this;
      };
    }
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, rect.width, rect.height);

    const canvasW = rect.width;
    const canvasH = rect.height;
    const isDark = document.documentElement.dataset?.theme === 'dark';
    const contentLeft = LABEL_WIDTH;
    const contentWidth = canvasW - LABEL_WIDTH;
    const sx = scrollX;

    // ── Ruler ──
    ctx.fillStyle = isDark ? 'rgba(255,255,255,0.04)' : 'rgba(0,0,0,0.03)';
    ctx.fillRect(0, 0, canvasW, RULER_HEIGHT);

    ctx.fillStyle = isDark ? 'rgba(255,255,255,0.5)' : 'rgba(0,0,0,0.5)';
    ctx.font = '10px "SF Mono", "Menlo", monospace';
    ctx.textAlign = 'center';

    let interval = 1;
    if (pps < 15) interval = 10;
    else if (pps < 30) interval = 5;
    else if (pps < 60) interval = 2;
    else if (pps > 120) interval = 0.5;

    const maxItemEnd = items.length > 0 ? Math.max(...items.map(it => it.end || 0)) : 0;
    const maxTime = Math.max(duration || 0, maxItemEnd, 30) * 1.05;
    for (let t = 0; t <= maxTime; t += interval) {
      const x = contentLeft + t * pps - sx;
      if (x < contentLeft - 10 || x > canvasW + 10) continue;
      ctx.fillText(formatTime(t), x, 16);

      // Tick marks
      ctx.strokeStyle = isDark ? 'rgba(255,255,255,0.08)' : 'rgba(0,0,0,0.06)';
      ctx.beginPath();
      ctx.moveTo(x, RULER_HEIGHT - 4);
      ctx.lineTo(x, RULER_HEIGHT);
      ctx.stroke();

      // Grid lines
      ctx.strokeStyle = isDark ? 'rgba(255,255,255,0.04)' : 'rgba(0,0,0,0.04)';
      ctx.beginPath();
      ctx.moveTo(x, RULER_HEIGHT);
      ctx.lineTo(x, canvasH);
      ctx.stroke();
    }

    // ── Track lanes (ALL tracks always visible in timeline) ──
    tracks.forEach((track, idx) => {
      const y = RULER_HEIGHT + idx * (TRACK_HEIGHT + TRACK_GAP);
      const isHidden = track.visible === false;

      // Track label background
      ctx.fillStyle = isDark ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.03)';
      ctx.fillRect(0, y, LABEL_WIDTH - 1, TRACK_HEIGHT);

      // Track lane background
      ctx.fillStyle = isDark ? 'rgba(255,255,255,0.02)' : 'rgba(0,0,0,0.015)';
      ctx.fillRect(contentLeft, y, contentWidth, TRACK_HEIGHT);

      // Track border
      ctx.strokeStyle = isDark ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.05)';
      ctx.strokeRect(contentLeft, y, contentWidth, TRACK_HEIGHT);

      // Muted overlay
      if (track.muted) {
        ctx.fillStyle = isDark ? 'rgba(255,59,48,0.06)' : 'rgba(255,59,48,0.04)';
        ctx.fillRect(contentLeft, y, contentWidth, TRACK_HEIGHT);
      }

      // Hidden track overlay — dimmed with diagonal stripes pattern
      if (isHidden) {
        ctx.fillStyle = isDark ? 'rgba(0,0,0,0.35)' : 'rgba(128,128,128,0.15)';
        ctx.fillRect(contentLeft, y, contentWidth, TRACK_HEIGHT);
        ctx.fillRect(0, y, LABEL_WIDTH - 1, TRACK_HEIGHT);
      }

      // Track separator line — drawn in the gap between this track and the next
      if (idx < tracks.length - 1) {
        const sepY = y + TRACK_HEIGHT + Math.floor(TRACK_GAP / 2);
        ctx.strokeStyle = isDark ? 'rgba(255,255,255,0.08)' : 'rgba(0,0,0,0.08)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(0, sepY);
        ctx.lineTo(canvasW, sepY);
        ctx.stroke();
      }
    });

    // ── Items (clips) ──
    // Precompute each item's next sibling start (on same track) so we can
    // cap the minimum render width without visually overlapping neighbors.
    const nextSiblingStartByItemId = {};
    {
      const byTrackSorted = {};
      for (const it of items) {
        (byTrackSorted[it.trackId] || (byTrackSorted[it.trackId] = [])).push(it);
      }
      for (const trackItems of Object.values(byTrackSorted)) {
        trackItems.sort((a, b) => a.start - b.start);
        for (let i = 0; i < trackItems.length; i++) {
          nextSiblingStartByItemId[trackItems[i].id] = i + 1 < trackItems.length
            ? trackItems[i + 1].start
            : Infinity;
        }
      }
    }

    items.forEach((item) => {
      const trackIdx = tracks.findIndex((t) => t.id === item.trackId);
      if (trackIdx < 0) return;
      const y = RULER_HEIGHT + trackIdx * (TRACK_HEIGHT + TRACK_GAP);
      const x1 = contentLeft + item.start * pps - sx;
      const x2 = contentLeft + item.end * pps - sx;
      const w = x2 - x1;

      // Clip rendering to track bounds — prevents visual bleed across the gap
      ctx.save();
      ctx.beginPath();
      ctx.rect(0, y, canvasW, TRACK_HEIGHT);
      ctx.clip();

      if (x2 < contentLeft || x1 > canvasW) { ctx.restore(); return; }

      const color = TRACK_COLORS[item.type] || TRACK_COLORS.video;
      const isSelected = item.id === selectedItemId;
      const isMultiSelected = selectedItemIds.includes(item.id);

      // Clip body — enforce minimum visual width of 4px for visibility, but
      // cap against the next sibling's start position so short adjacent
      // items never visually overlap each other.
      ctx.fillStyle = (isSelected || isMultiSelected) ? color + 'DD' : color + '77';
      const rr = 4;
      const clipX = Math.max(x1, contentLeft);
      const nextSiblingStart = nextSiblingStartByItemId[item.id];
      const maxRightX = nextSiblingStart !== undefined && nextSiblingStart !== Infinity
        ? contentLeft + nextSiblingStart * pps - sx
        : canvasW;
      const availableW = Math.max(0, maxRightX - clipX);
      const actualW = Math.min(w, canvasW - clipX);
      const clipW = Math.min(Math.max(actualW, 4), availableW || actualW);
      ctx.beginPath();
      ctx.roundRect(clipX, y + 2, clipW, TRACK_HEIGHT - 4, rr);
      ctx.fill();

      // ── Filmstrip thumbnails (long-zoom only) ──
      // Skip when:
      //   * the track isn't a video carrier (audio / text / shape items
      //     don't have meaningful thumbnails),
      //   * the clip is too narrow for thumbs to be useful,
      //   * we don't have a media URL on the item.
      // Throttled by computeThumbStops which targets ~thumbW pixels per
      // thumb so the count scales with zoom.
      const thumbCarrier = item.type === 'video' || item.type === 'image';
      const innerH = TRACK_HEIGHT - 8;
      const minClipPxForThumbs = 80;
      const mediaSrc = item.mediaSrc || item.src || null;
      if (thumbCarrier && clipW >= minClipPxForThumbs && innerH > 24 && mediaSrc) {
        const thumbW = Math.round(innerH * (16 / 9));
        const stops = computeThumbStops(item.start, item.end, pps, thumbW);
        let scheduledRedraw = false;
        for (const t of stops) {
          const tx = contentLeft + t * pps - sx;
          // Skip thumbs entirely outside the clip's visible portion.
          if (tx + thumbW < clipX || tx > clipX + clipW) continue;
          const cached = getCachedThumbnail(mediaSrc, t, thumbW, innerH);
          if (cached) {
            ctx.save();
            ctx.beginPath();
            ctx.roundRect(clipX, y + 4, clipW, innerH, rr - 1);
            ctx.clip();
            ctx.globalAlpha = 0.85;
            ctx.drawImage(cached, tx - thumbW / 2, y + 4, thumbW, innerH);
            ctx.globalAlpha = 1;
            ctx.restore();
          } else if (!scheduledRedraw) {
            // Fire one async generation per draw — they share a cache
            // so subsequent draws hit. We schedule a redraw via the
            // shared invalidate ref so the latest ``draw`` fn is used.
            scheduledRedraw = true;
            ensureThumbnail(mediaSrc, t, thumbW, innerH)
              .then(() => requestRedrawRef.current && requestRedrawRef.current())
              .catch(() => {});
          }
        }
      }

      // Selected border (solid white for primary, dashed cyan for multi-select)
      if (isSelected) {
        ctx.strokeStyle = '#FFFFFF';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.roundRect(clipX, y + 2, clipW, TRACK_HEIGHT - 4, rr);
        ctx.stroke();
        ctx.lineWidth = 1;
      } else if (isMultiSelected) {
        ctx.strokeStyle = '#00D4FF';
        ctx.lineWidth = 2;
        ctx.setLineDash([4, 3]);
        ctx.beginPath();
        ctx.roundRect(clipX, y + 2, clipW, TRACK_HEIGHT - 4, rr);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.lineWidth = 1;
      }

      // Group indicator: colored bottom bar
      if (item.groupId && clipW > 8) {
        const hue = hashGroupId(item.groupId) % 360;
        ctx.fillStyle = `hsl(${hue}, 70%, 50%)`;
        ctx.fillRect(clipX + 2, y + TRACK_HEIGHT - 6, clipW - 4, 4);
      }

      // Transition indicator
      if (item.transition) {
        const transDur = item.transition.duration || 0.5;
        const transW = transDur * pps;
        ctx.fillStyle = 'rgba(255,255,255,0.25)';
        ctx.beginPath();
        ctx.moveTo(x1, y + 2);
        ctx.lineTo(x1 + transW, y + 2);
        ctx.lineTo(x1, y + TRACK_HEIGHT - 2);
        ctx.closePath();
        ctx.fill();
      }

      // Clip label
      if (w > 35) {
        ctx.fillStyle = '#fff';
        ctx.font = '10px -apple-system, BlinkMacSystemFont, sans-serif';
        ctx.textAlign = 'left';
        const label = item.textContent
          ? item.textContent.slice(0, 25)
          : item.subtitleText
            ? item.subtitleText.slice(0, 25)
            : item.type;
        ctx.fillText(label, Math.max(x1 + 8, contentLeft + 4), y + TRACK_HEIGHT / 2 + 4, w - 16);
      }

      // Trim handles (visual, for selected items)
      if (isSelected && w > 20) {
        ctx.fillStyle = '#FFFFFF';
        ctx.globalAlpha = 0.8;
        ctx.fillRect(x1, y + 4, HANDLE_WIDTH, TRACK_HEIGHT - 8);
        ctx.fillRect(x2 - HANDLE_WIDTH, y + 4, HANDLE_WIDTH, TRACK_HEIGHT - 8);
        ctx.globalAlpha = 1;
      }

      // Effects indicator dot
      const effects = item.effects;
      if (effects && typeof effects === 'object' && !Array.isArray(effects)) {
        const hasEffects = Object.entries(effects).some(([k, v]) => v !== 0 && v !== undefined && v !== null);
        if (hasEffects) {
          ctx.fillStyle = '#FFD700';
          ctx.beginPath();
          ctx.arc(x2 - 12, y + 8, 3, 0, Math.PI * 2);
          ctx.fill();
        }
      }
      ctx.restore(); // End per-item track clip
    });

    // ── Crop segments (on the crop track) ──
    if (cropSegments && cropSegments.length > 0) {
      const cropTrackIdx = tracks.findIndex((t) => t.type === 'crop');
      if (cropTrackIdx >= 0) {
        const cropTrack = tracks[cropTrackIdx];
        if (cropTrack.visible !== false) {
          const cy = RULER_HEIGHT + cropTrackIdx * (TRACK_HEIGHT + TRACK_GAP);

          // Clip crop segment rendering to crop track bounds
          ctx.save();
          ctx.beginPath();
          ctx.rect(0, cy, canvasW, TRACK_HEIGHT);
          ctx.clip();

          // Precompute next-segment start times so min-width clamping doesn't
          // cause adjacent crop segments to visually overlap each other.
          const cropsSorted = [...cropSegments].sort((a, b) => a.startTime - b.startTime);
          const nextCropStartById = {};
          for (let i = 0; i < cropsSorted.length; i++) {
            nextCropStartById[cropsSorted[i].id] = i + 1 < cropsSorted.length
              ? cropsSorted[i + 1].startTime
              : Infinity;
          }

          cropSegments.forEach((seg) => {
            const cx1 = contentLeft + seg.startTime * pps - sx;
            const cx2 = contentLeft + seg.endTime * pps - sx;
            const cw = cx2 - cx1;
            if (cx2 < contentLeft || cx1 > canvasW) return;
            const clipCX = Math.max(cx1, contentLeft);
            const nextStart = nextCropStartById[seg.id];
            const maxRightCX = nextStart !== undefined && nextStart !== Infinity
              ? contentLeft + nextStart * pps - sx
              : canvasW;
            const availableCW = Math.max(0, maxRightCX - clipCX);
            const actualCW = Math.min(cw, canvasW - clipCX);
            const clipCW = Math.min(Math.max(actualCW, 4), availableCW || actualCW);

            // Color by cluster or manual override
            const clrIdx = seg.isManualOverride ? 4 : Math.max(0, seg.clusterId);
            const baseColor = CROP_CLUSTER_COLORS[clrIdx % CROP_CLUSTER_COLORS.length];
            const isSelCrop = seg.id === selectedCropSegmentId;
            ctx.fillStyle = isSelCrop ? baseColor + 'DD' : baseColor + '88';
            ctx.beginPath();
            ctx.roundRect(clipCX, cy + 3, clipCW, TRACK_HEIGHT - 6, 3);
            ctx.fill();

            // Selection border
            if (isSelCrop) {
              ctx.strokeStyle = '#FFFFFF';
              ctx.lineWidth = 2;
              ctx.beginPath();
              ctx.roundRect(clipCX, cy + 3, clipCW, TRACK_HEIGHT - 6, 3);
              ctx.stroke();
              ctx.lineWidth = 1;
            }

            // Label
            if (cw > 30) {
              ctx.fillStyle = '#fff';
              ctx.font = '10px -apple-system, BlinkMacSystemFont, sans-serif';
              ctx.textAlign = 'left';
              const lbl = seg.label || `${seg.cropX}%`;
              ctx.fillText(lbl, Math.max(cx1 + 6, contentLeft + 4), cy + TRACK_HEIGHT / 2 + 3, cw - 12);
            }
          });
          ctx.restore(); // End crop track clip
        }
      }
    }

    // ── Segment boundaries ──
    if (segments && segments.length > 0) {
      segments.forEach((seg) => {
        const segStart = seg.start != null ? seg.start : 0;
        const segEnd = seg.end != null ? seg.end : 0;
        for (const edge of [segStart, segEnd]) {
          const ex = contentLeft + edge * pps - sx;
          if (ex < contentLeft || ex > canvasW) continue;
          ctx.strokeStyle = seg.color || '#FF9F0A';
          ctx.lineWidth = 1.5;
          ctx.setLineDash([4, 3]);
          ctx.beginPath();
          ctx.moveTo(ex, RULER_HEIGHT);
          ctx.lineTo(ex, canvasH);
          ctx.stroke();
          ctx.setLineDash([]);
          ctx.lineWidth = 1;
        }
        const sx1 = contentLeft + segStart * pps - sx;
        const sx2 = contentLeft + segEnd * pps - sx;
        if (sx2 > contentLeft && sx1 < canvasW && seg.label) {
          ctx.fillStyle = (seg.color || '#FF9F0A') + '18';
          ctx.fillRect(Math.max(sx1, contentLeft), RULER_HEIGHT, Math.min(sx2, canvasW) - Math.max(sx1, contentLeft), canvasH - RULER_HEIGHT);
          ctx.fillStyle = seg.color || '#FF9F0A';
          ctx.font = '9px -apple-system, BlinkMacSystemFont, sans-serif';
          ctx.textAlign = 'left';
          ctx.fillText(seg.label, Math.max(sx1 + 4, contentLeft + 4), RULER_HEIGHT + 10);
        }
      });
    }

    // ── Playhead ──
    // Read from ref for smooth rAF-driven updates (avoids stale closure)
    const phX = contentLeft + playheadRef.current * pps - sx;
    if (phX >= contentLeft && phX <= canvasW) {
      // Playhead line with subtle glow
      ctx.save();
      ctx.shadowColor = 'rgba(255, 59, 48, 0.5)';
      ctx.shadowBlur = 6;
      ctx.strokeStyle = '#FF3B30';
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.moveTo(phX, 0);
      ctx.lineTo(phX, canvasH);
      ctx.stroke();
      ctx.restore();
      ctx.lineWidth = 1;

      // Playhead handle — large inverted triangle for easy grabbing
      ctx.fillStyle = '#FF3B30';
      ctx.beginPath();
      ctx.moveTo(phX - 12, 0);
      ctx.lineTo(phX + 12, 0);
      ctx.lineTo(phX, 18);
      ctx.closePath();
      ctx.fill();

      // White inner triangle for visibility
      ctx.fillStyle = 'rgba(255, 255, 255, 0.5)';
      ctx.beginPath();
      ctx.moveTo(phX - 6, 1);
      ctx.lineTo(phX + 6, 1);
      ctx.lineTo(phX, 10);
      ctx.closePath();
      ctx.fill();
    }

    // ── Snap guide line ──
    // ``snapLine`` is subscribed at the top of the component so it
    // shows up in the ``draw`` useCallback's dep array; reading the
    // outer binding keeps the closure consistent with that dep.
    if (snapLine && isDragging) {
      const snapX = contentLeft + snapLine.time * pps - sx;
      if (snapX >= contentLeft && snapX <= canvasW) {
        ctx.save();
        ctx.strokeStyle = '#00D4FF';
        ctx.lineWidth = 1.5;
        ctx.setLineDash([]);
        ctx.globalAlpha = 0.9;
        ctx.beginPath();
        ctx.moveTo(snapX, RULER_HEIGHT);
        ctx.lineTo(snapX, canvasH);
        ctx.stroke();

        // Small diamond indicator at top
        ctx.fillStyle = '#00D4FF';
        ctx.beginPath();
        ctx.moveTo(snapX, RULER_HEIGHT - 1);
        ctx.lineTo(snapX - 4, RULER_HEIGHT + 5);
        ctx.lineTo(snapX, RULER_HEIGHT + 11);
        ctx.lineTo(snapX + 4, RULER_HEIGHT + 5);
        ctx.closePath();
        ctx.fill();

        ctx.restore();
      }
    }

    // ── Hover indicator ──
    if (hoverTime !== null) {
      const hx = contentLeft + hoverTime * pps - sx;
      if (hx >= contentLeft && hx <= canvasW) {
        ctx.strokeStyle = isDark ? 'rgba(255,255,255,0.2)' : 'rgba(0,0,0,0.2)';
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        ctx.moveTo(hx, RULER_HEIGHT);
        ctx.lineTo(hx, canvasH);
        ctx.stroke();
        ctx.setLineDash([]);

        // Tooltip
        ctx.fillStyle = 'rgba(0,0,0,0.85)';
        const tooltipText = formatTimeMs(hoverTime);
        const tw = ctx.measureText(tooltipText).width + 10;
        ctx.beginPath();
        ctx.roundRect(Math.min(hx - tw / 2, canvasW - tw - 4), 3, tw, 16, 4);
        ctx.fill();
        ctx.fillStyle = '#fff';
        ctx.font = '10px "SF Mono", monospace';
        ctx.textAlign = 'center';
        ctx.fillText(tooltipText, Math.min(hx, canvasW - tw / 2 - 4), 14);
      }
    }

    // ── Razor cursor indicator ──
    if (activeTool === 'razor' && hoverTime !== null) {
      const rx = contentLeft + hoverTime * pps - sx;
      if (rx >= contentLeft && rx <= canvasW) {
        ctx.strokeStyle = '#FF9500';
        ctx.lineWidth = 2;
        ctx.setLineDash([6, 3]);
        ctx.beginPath();
        ctx.moveTo(rx, RULER_HEIGHT);
        ctx.lineTo(rx, canvasH);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.lineWidth = 1;
      }
    }
  }, [tracks, items, duration, zoom, scrollX, selectedItemId, selectedItemIds, hoverTime, pps, compact, activeTool, segments, cropSegments, selectedCropSegmentId, snapLine]);
  // ``playhead`` is intentionally absent: every play tick was both
  // recreating ``draw`` (causing the ``[draw]`` effect below to re-fire)
  // AND running the rAF loop. Two redraw mechanisms stacked.
  // The rAF loop below is now the SINGLE source of repaint during
  // playback; ``playhead`` is read off ``playheadRef`` inside ``draw``.

  // Continuous redraw during playback. Throttled to ~30 Hz so a long
  // timeline doesn't repaint at full display refresh.
  useEffect(() => {
    if (!isPlaying) return;
    let rafId;
    let lastPaint = 0;
    const PAINT_INTERVAL_MS = 33;   // ~30 fps redraw is plenty for the
                                    // playhead bar; the canvas is static
                                    // otherwise.
    const tick = (now) => {
      if (now - lastPaint >= PAINT_INTERVAL_MS) {
        draw();
        lastPaint = now;
      }
      rafId = requestAnimationFrame(tick);
    };
    rafId = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafId);
  }, [isPlaying, draw]);

  // Auto-scroll timeline to keep playhead visible during playback
  useEffect(() => {
    if (!isPlaying) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const visibleWidth = canvas.getBoundingClientRect().width - LABEL_WIDTH;
    if (visibleWidth <= 0) return;
    const playheadPx = playhead * pps;
    const viewStart = scrollX;
    const viewEnd = scrollX + visibleWidth;
    // When playhead moves past 80% of the visible area, scroll to keep it at 20%
    if (playheadPx > viewEnd - visibleWidth * 0.2) {
      setScrollX(Math.max(0, playheadPx - visibleWidth * 0.2));
    } else if (playheadPx < viewStart) {
      setScrollX(Math.max(0, playheadPx - visibleWidth * 0.1));
    }
  }, [isPlaying, playhead, pps, scrollX, setScrollX]);

  // Redraw on state changes
  useEffect(() => { draw(); }, [draw]);

  // Redraw when the playhead moves while the video is paused. The
  // primary ``draw`` callback intentionally omits ``playhead`` from
  // its dependency array (playback already owns its own rAF repaint
  // loop, and adding ``playhead`` there would stack two redraw
  // mechanisms at 60 Hz). That leaves paused-state playhead updates
  // — e.g. scrubbing the multi-track timeline, arrow-key nudges,
  // ``setPlayhead`` calls from the video's ``seeked`` event — without
  // any automatic redraw trigger, which is what made the playhead
  // appear to freeze under the cursor during scrub. Piggy-backing on
  // ``requestRedrawRef`` uses the latest ``draw`` closure without
  // pulling it into this effect's dep list (which would re-fire on
  // every state change the draw callback tracks).
  useEffect(() => {
    if (isPlaying) return;
    requestRedrawRef.current?.();
  }, [playhead, isPlaying]);

  // Track the latest ``draw`` so async filmstrip generations don't
  // hold a stale closure. We assign the ref in an effect so React
  // sees the canonical render order.
  useEffect(() => {
    requestRedrawRef.current = draw;
    return () => { requestRedrawRef.current = null; };
  }, [draw]);

  // Resize observer
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ro = new ResizeObserver(() => draw());
    ro.observe(canvas);
    return () => ro.disconnect();
  }, [draw]);

  // ── Pointer helpers ────────────────────────────────────────────────────────
  const getTimeFromX = useCallback((clientX) => {
    const canvas = canvasRef.current;
    if (!canvas) return 0;
    const rect = canvas.getBoundingClientRect();
    const x = clientX - rect.left - LABEL_WIDTH + scrollX;
    return Math.max(0, x / pps);
  }, [pps, scrollX]);

  const getTrackFromY = useCallback((clientY) => {
    const canvas = canvasRef.current;
    if (!canvas) return null;
    const rect = canvas.getBoundingClientRect();
    const y = clientY - rect.top - RULER_HEIGHT;
    const idx = Math.floor(y / (TRACK_HEIGHT + TRACK_GAP));
    // Read latest tracks from store to avoid stale closures
    const currentTracks = useTimelineStore.getState().tracks;
    if (idx < 0 || idx >= currentTracks.length) return null;
    return currentTracks[idx] || null;
  }, []);

  const hitTestItem = useCallback((clientX, clientY) => {
    // Read ALL values from getState() to avoid stale closures
    const { items: currentItems, scrollX: currentScrollX, selectedItemId: selectedId, tracks: currentTracks } = useTimelineStore.getState();
    const currentPps = ppsRef.current;

    const canvas = canvasRef.current;
    if (!canvas) return null;
    const rect = canvas.getBoundingClientRect();
    const px = clientX - rect.left;
    const x = px - LABEL_WIDTH + currentScrollX;
    const time = Math.max(0, x / currentPps);

    // Inline track-from-Y to use currentTracks from getState()
    const mouseY = clientY - rect.top;
    const trackIdx = Math.floor((mouseY - RULER_HEIGHT) / (TRACK_HEIGHT + TRACK_GAP));
    if (trackIdx < 0 || trackIdx >= currentTracks.length) return null;
    const track = currentTracks[trackIdx];
    if (!track) return null;

    // Collect ALL matching items, then pick the best one
    const matches = [];
    for (const item of currentItems) {
      if (item.trackId !== track.id) continue;
      if (time < item.start || time > item.end) continue;

      const x1 = LABEL_WIDTH + item.start * currentPps - currentScrollX;
      const x2 = LABEL_WIDTH + item.end * currentPps - currentScrollX;

      let edge = 'body';
      if (Math.abs(px - x1) < HANDLE_HIT_AREA) edge = 'left';
      else if (Math.abs(px - x2) < HANDLE_HIT_AREA) edge = 'right';

      matches.push({ item, edge, x1, x2 });
    }

    if (matches.length === 0) return null;

    // Priority 1: currently selected item (for boundary clicks between adjacent items)
    const selected = matches.find(m => m.item.id === selectedId);
    if (selected) return selected;

    // Priority 2: narrowest (most specific) clip at the click point, then newest (last in array)
    matches.sort((a, b) => {
      const widthDiff = (a.x2 - a.x1) - (b.x2 - b.x1);
      if (Math.abs(widthDiff) > 0.5) return widthDiff;
      // Same width: prefer the one later in the items array (newest)
      return currentItems.indexOf(b.item) - currentItems.indexOf(a.item);
    });
    return matches[0];
  }, []);

  // Hit-test a crop segment on the crop track. Returns { seg, edge } or null.
  const hitTestCropSegment = useCallback((clientX, clientY) => {
    const { cropSegments: segs, scrollX: currentScrollX, tracks: currentTracks } = useTimelineStore.getState();
    if (!segs?.length) return null;

    const cropTrackIdx = currentTracks.findIndex((t) => t.type === 'crop');
    if (cropTrackIdx < 0) return null;
    const cropTrack = currentTracks[cropTrackIdx];
    if (!cropTrack || cropTrack.visible === false || cropTrack.locked) return null;

    const canvas = canvasRef.current;
    if (!canvas) return null;
    const rect = canvas.getBoundingClientRect();
    const px = clientX - rect.left;
    const mouseY = clientY - rect.top;
    const currentPps = ppsRef.current;

    // Verify Y is on the crop track
    const trackIdx = Math.floor((mouseY - RULER_HEIGHT) / (TRACK_HEIGHT + TRACK_GAP));
    if (trackIdx !== cropTrackIdx) return null;

    const x = px - LABEL_WIDTH + currentScrollX;
    const time = Math.max(0, x / currentPps);

    for (const seg of segs) {
      if (time < seg.startTime || time > seg.endTime) continue;
      const x1 = LABEL_WIDTH + seg.startTime * currentPps - currentScrollX;
      const x2 = LABEL_WIDTH + seg.endTime * currentPps - currentScrollX;
      let edge = 'body';
      if (Math.abs(px - x1) < HANDLE_HIT_AREA) edge = 'left';
      else if (Math.abs(px - x2) < HANDLE_HIT_AREA) edge = 'right';
      return { seg, edge, x1, x2 };
    }
    return null;
  }, []);

  // ── Pointer events ─────────────────────────────────────────────────────────
  const onPointerDown = useCallback((e) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();

    // Right-click context menu
    if (e.button === 2) {
      e.preventDefault();
      const time = getTimeFromX(e.clientX);
      const hit = hitTestItem(e.clientX, e.clientY);
      setContextMenu({
        x: e.clientX - rect.left,
        y: e.clientY - rect.top,
        time,
        item: hit?.item || null,
      });
      return;
    }

    setContextMenu(null);

    // Middle-click or space+left-click: pan/scroll
    if (e.button === 1 || (e.button === 0 && spaceHeld)) {
      e.preventDefault();
      setIsDragging(true);
      setDragInfo({ type: 'pan', startX: e.clientX, origScrollX: scrollX });
      return;
    }

    // Razor tool: split on click (blocked on locked tracks)
    if (activeTool === 'razor') {
      const time = getTimeFromX(e.clientX);
      const hit = hitTestItem(e.clientX, e.clientY);
      if (hit?.item) {
        const currentTracks = useTimelineStore.getState().tracks;
        const itemTrack = currentTracks.find((t) => t.id === hit.item.trackId);
        if (itemTrack?.locked) return; // Cannot split items on locked tracks
        splitItem(hit.item.id, time);
      }
      return;
    }

    // Playhead grab: check if click is near the playhead line or handle.
    // Only grab the playhead directly from the ruler area (triangle handle).
    // Clicks on items near the playhead should select the item, not grab the playhead.
    const currentScrollX = useTimelineStore.getState().scrollX;
    const playheadPixelX = rect.left + LABEL_WIDTH + playheadRef.current * ppsRef.current - currentScrollX;
    const mouseY = e.clientY - rect.top;
    const distToPlayhead = Math.abs(e.clientX - playheadPixelX);
    const isInRuler = mouseY <= RULER_HEIGHT + 8;
    if (isInRuler && distToPlayhead <= PLAYHEAD_GRAB_WIDTH) {
      // Grab the playhead directly from the ruler
      const time = getTimeFromX(e.clientX);
      setIsScrubbing(true);
      setPlayhead(time);
      onSeek?.(time);
      setIsDragging(true);
      setDragInfo({ type: 'scrub', startX: e.clientX });
      return;
    }

    const hit = hitTestItem(e.clientX, e.clientY);
    if (hit) {
      // Ctrl/Cmd+Click: toggle multi-select (no drag)
      if (e.ctrlKey || e.metaKey) {
        toggleSelectedItem(hit.item.id);
        onItemSelect?.(hit.item);
        return;
      }

      // Alt+Click: select all group members (group-aware selection)
      if (e.altKey && hit.item.groupId) {
        const currentItems = useTimelineStore.getState().items;
        const groupMembers = currentItems.filter(it => it.groupId === hit.item.groupId).map(it => it.id);
        setSelectedItemIds(groupMembers);
        useTimelineStore.setState({ selectedItemId: hit.item.id });
        onItemSelect?.(hit.item);
        // Continue to drag logic below
      } else {
        // Plain click: always single-select the clicked item
        setSelectedItemId(hit.item.id);
        onItemSelect?.(hit.item);
      }

      // Check if item is on a locked track — allow selection but block drag/trim
      const { tracks: latestTracks, items: latestItems } = useTimelineStore.getState();
      const itemTrack = latestTracks.find((t) => t.id === hit.item.trackId);
      if (itemTrack?.locked) {
        // Selection allowed, but no drag/trim
      } else if (hit.edge === 'left' || hit.edge === 'right') {
        // Pause undo history during drag so intermediate frames don't flood it
        useTimelineStore.temporal.getState().pause();
        setIsDragging(true);
        setDragInfo({
          type: 'trim',
          itemId: hit.item.id,
          edge: hit.edge,
          origStart: hit.item.start,
          origEnd: hit.item.end,
          origTrimStart: hit.item.trimStart,
          origTrimEnd: hit.item.trimEnd,
          startX: e.clientX,
        });
      } else {
        // Multi-item drag: build snapshots for all selected items
        const currentSelectedIds = useTimelineStore.getState().selectedItemIds;
        const dragIds = currentSelectedIds.includes(hit.item.id)
          ? currentSelectedIds
          : [hit.item.id];
        const snapshots = dragIds.map(id => {
          const it = latestItems.find(i => i.id === id);
          return it ? { id, origStart: it.start, origEnd: it.end, origTrackId: it.trackId } : null;
        }).filter(Boolean);

        // Pause undo history during drag so intermediate frames don't flood it
        useTimelineStore.temporal.getState().pause();
        setIsDragging(true);
        setDragInfo({
          type: 'move',
          itemId: hit.item.id,
          snapshots,
          origStart: hit.item.start,
          origEnd: hit.item.end,
          origTrackId: hit.item.trackId,
          startX: e.clientX,
          startY: e.clientY,
        });
      }
    } else {
      // No item hit — check crop track click, then playhead, then seek
      const time = getTimeFromX(e.clientX);

      // Check if click is on a crop segment (with edge detection for trim/move)
      const cropHit = hitTestCropSegment(e.clientX, e.clientY);
      if (cropHit) {
        selectCropSegment(cropHit.seg.id);
        setSelectedItemId(null);
        onItemSelect?.(null); // Open properties panel for crop segment

        // Pause undo history during drag so intermediate frames don't flood it
        useTimelineStore.temporal.getState().pause();
        setIsDragging(true);
        if (cropHit.edge === 'left' || cropHit.edge === 'right') {
          setDragInfo({
            type: 'trim-crop',
            segId: cropHit.seg.id,
            edge: cropHit.edge,
            origStart: cropHit.seg.startTime,
            origEnd: cropHit.seg.endTime,
            startX: e.clientX,
          });
        } else {
          setDragInfo({
            type: 'move-crop',
            segId: cropHit.seg.id,
            origStart: cropHit.seg.startTime,
            origEnd: cropHit.seg.endTime,
            startX: e.clientX,
          });
        }
        return;
      }

      if (!isInRuler && distToPlayhead <= Math.max(PLAYHEAD_GRAB_WIDTH / 2, 8)) {
        // Grab the playhead line directly
        setIsScrubbing(true);
        setPlayhead(time);
        onSeek?.(time);
        setIsDragging(true);
        setDragInfo({ type: 'scrub', startX: e.clientX });
      } else {
        // Click on empty area: seek + deselect
        setIsScrubbing(true);
        setPlayhead(time);
        setSelectedItemId(null);
        selectCropSegment(null);
        onSeek?.(time);
        setIsDragging(true);
        setDragInfo({ type: 'scrub', startX: e.clientX });
      }
    }
  }, [hitTestItem, hitTestCropSegment, getTimeFromX, setPlayhead, setSelectedItemId, setSelectedItemIds, toggleSelectedItem, selectCropSegment, onSeek, activeTool, splitItem, spaceHeld, scrollX, onItemSelect]);

  useEffect(() => {
    if (!isDragging || !dragInfo) return;

    // ── rAF-coalesced scrub flush ──
    // Pointer events fire at >100Hz on modern devices. Calling
    // ``setPlayhead`` + ``onSeek`` on every event triggers a Zustand
    // store update (re-renders Timeline + every other subscriber) AND
    // a video-element ``currentTime`` write (which has to seek to a
    // keyframe + decode) per frame — the result is the choppy scrub
    // the user reported. Coalesce both writes to one per animation
    // frame so the video gets at most ~60 seeks per second instead of
    // hundreds. The scrub thumb itself updates at full pointer-event
    // rate visually because the canvas redraw is decoupled.
    // Pending rAF handle + buffered time live on refs so they
    // survive an effect re-mount caused by a dep identity change
    // mid-drag. Without this, a fresh ``onSeek`` identity from the
    // parent tore down the effect every frame, cancelled the rAF, and
    // dropped the buffered time — making the playhead look frozen.
    const _flushScrub = () => {
      _scrubRafRef.current = 0;
      const t = _scrubPendingRef.current;
      if (t == null) return;
      _scrubPendingRef.current = null;
      try { setPlayheadRef.current?.(t); } catch { /* noop */ }
      try { onSeekRef.current?.(t); } catch { /* noop */ }
      // ``playhead`` is intentionally NOT in ``draw``'s dep list (so
      // playback doesn't recreate the callback 60×/sec), which means
      // a plain ``setPlayhead`` during scrub won't automatically
      // re-run the ``useEffect(() => draw(), [draw])``. Request a
      // redraw directly so the playhead line visually tracks the
      // cursor at rAF rate — without this call the playhead appeared
      // frozen under the mouse for paused scrubs.
      try { requestRedrawRef.current?.(); } catch { /* noop */ }
    };

    const onMove = (e) => {
      if (dragInfo.type === 'pan') {
        const dx = e.clientX - dragInfo.startX;
        setScrollX(Math.max(0, dragInfo.origScrollX - dx));
        return;
      }

      const time = getTimeFromX(e.clientX);

      if (dragInfo.type === 'scrub') {
        // Buffer the latest time; the rAF flush picks the freshest
        // value when the browser is ready to paint.
        _scrubPendingRef.current = time;
        if (_scrubRafRef.current === 0) {
          _scrubRafRef.current = requestAnimationFrame(_flushScrub);
        }
        return;
      } else if (dragInfo.type === 'trim') {
        const item = items.find((i) => i.id === dragInfo.itemId);
        if (!item) return;
        const mediaLib = useTimelineStore.getState().mediaLibrary;
        const maxDur = getMaxItemDuration(item, mediaLib);
        const setSnapLine = useTimelineStore.getState().setSnapLine;

        if (dragInfo.edge === 'left') {
          let newStart = Math.max(0, Math.min(dragInfo.origEnd - 0.1, time));
          if (maxDur < Infinity) {
            const minStart = dragInfo.origEnd - maxDur;
            newStart = Math.max(newStart, minStart);
          }
          if (snapEnabled) {
            const snap = findSnapTarget(newStart, items, dragInfo.itemId, playheadRef.current, duration, pps, sceneCuts);
            if (snap) {
              newStart = Math.max(maxDur < Infinity ? dragInfo.origEnd - maxDur : 0, snap.snappedTime);
              setSnapLine({ time: snap.snapTarget });
            } else {
              setSnapLine(null);
            }
          }
          updateItem(dragInfo.itemId, { start: newStart });
        } else {
          let newEnd = Math.max(dragInfo.origStart + 0.1, time);
          if (maxDur < Infinity) {
            newEnd = Math.min(newEnd, item.start + maxDur);
          }
          if (snapEnabled) {
            const snap = findSnapTarget(newEnd, items, dragInfo.itemId, playheadRef.current, duration, pps, sceneCuts);
            if (snap) {
              newEnd = maxDur < Infinity
                ? Math.min(snap.snappedTime, item.start + maxDur)
                : snap.snappedTime;
              setSnapLine({ time: snap.snapTarget });
            } else {
              setSnapLine(null);
            }
          }
          updateItem(dragInfo.itemId, { end: newEnd });
        }
      } else if (dragInfo.type === 'move') {
        const dx = (e.clientX - dragInfo.startX) / pps;
        const snapshots = dragInfo.snapshots || [{ id: dragInfo.itemId, origStart: dragInfo.origStart, origEnd: dragInfo.origEnd, origTrackId: dragInfo.origTrackId }];
        const primarySnap = snapshots[0];
        let newStart = Math.max(0, primarySnap.origStart + dx);
        const dur = primarySnap.origEnd - primarySnap.origStart;

        // Snap (both edges of the primary moving item)
        const setSnapLine = useTimelineStore.getState().setSnapLine;
        if (snapEnabled) {
          const snapStart = findSnapTarget(newStart, items, dragInfo.itemId, playheadRef.current, duration, pps, sceneCuts);
          const snapEnd = findSnapTarget(newStart + dur, items, dragInfo.itemId, playheadRef.current, duration, pps, sceneCuts);

          if (snapStart && (!snapEnd || Math.abs(snapStart.snappedTime - newStart) * pps <= Math.abs(snapEnd.snappedTime - (newStart + dur)) * pps)) {
            newStart = snapStart.snappedTime;
            setSnapLine({ time: snapStart.snapTarget });
          } else if (snapEnd) {
            newStart = snapEnd.snappedTime - dur;
            setSnapLine({ time: snapEnd.snapTarget });
          } else {
            setSnapLine(null);
          }
        } else {
          setSnapLine(null);
        }

        // Compute actual delta from snapped primary position
        const actualDelta = newStart - primarySnap.origStart;

        // Track change (only for single-item drag on the primary item)
        const track = getTrackFromY(e.clientY);
        let primaryTrackId = dragInfo.origTrackId;
        if (track && snapshots.length === 1) {
          const draggedItem = useTimelineStore.getState().items.find((i) => i.id === dragInfo.itemId);
          const itemType = draggedItem?.type;
          const trackType = track.type;
          const compatible =
            (itemType === 'video' && trackType === 'video') ||
            (itemType === 'audio' && trackType === 'audio') ||
            ((itemType === 'text' || itemType === 'shape' || itemType === 'image' || itemType === 'overlay') && trackType === 'overlay') ||
            (itemType === 'subtitle' && trackType === 'subtitle') ||
            (itemType === 'crop' && trackType === 'crop');
          if (compatible && !track.locked) primaryTrackId = track.id;
        }

        // Apply to all items in the drag group
        for (const snap of snapshots) {
          const itemNewStart = Math.max(0, snap.origStart + actualDelta);
          const itemDur = snap.origEnd - snap.origStart;
          const itemTrackId = snap.id === dragInfo.itemId ? primaryTrackId : snap.origTrackId;
          updateItem(snap.id, { start: itemNewStart, end: itemNewStart + itemDur, trackId: itemTrackId });
        }
      } else if (dragInfo.type === 'trim-crop') {
        // Trim the left or right edge of a crop segment
        const { cropSegments: segs, updateCropSegment } = useTimelineStore.getState();
        const seg = segs.find((s) => s.id === dragInfo.segId);
        if (!seg) return;
        const idx = segs.findIndex((s) => s.id === dragInfo.segId);
        const prev = idx > 0 ? segs[idx - 1] : null;
        const next = idx < segs.length - 1 ? segs[idx + 1] : null;
        const MIN_DUR = 0.05; // minimum 50ms so handles don't collapse

        if (dragInfo.edge === 'left') {
          let newStart = Math.max(0, time);
          // Clamp against previous neighbor's end
          if (prev) newStart = Math.max(newStart, prev.endTime);
          // Keep at least MIN_DUR
          newStart = Math.min(newStart, dragInfo.origEnd - MIN_DUR);
          updateCropSegment({ ...seg, startTime: newStart, endTime: dragInfo.origEnd });
        } else {
          let newEnd = Math.max(dragInfo.origStart + MIN_DUR, time);
          // Clamp against next neighbor's start
          if (next) newEnd = Math.min(newEnd, next.startTime);
          // Clamp against clip duration (if known)
          if (duration > 0) newEnd = Math.min(newEnd, duration);
          updateCropSegment({ ...seg, startTime: dragInfo.origStart, endTime: newEnd });
        }
      } else if (dragInfo.type === 'move-crop') {
        // Move (slide) a crop segment along the timeline
        const { cropSegments: segs, updateCropSegment } = useTimelineStore.getState();
        const seg = segs.find((s) => s.id === dragInfo.segId);
        if (!seg) return;
        const idx = segs.findIndex((s) => s.id === dragInfo.segId);
        const prev = idx > 0 ? segs[idx - 1] : null;
        const next = idx < segs.length - 1 ? segs[idx + 1] : null;
        const dur = dragInfo.origEnd - dragInfo.origStart;
        const dx = (e.clientX - dragInfo.startX) / pps;
        let newStart = Math.max(0, dragInfo.origStart + dx);
        // Clamp within neighbor boundaries
        if (prev) newStart = Math.max(newStart, prev.endTime);
        if (next) newStart = Math.min(newStart, next.startTime - dur);
        if (duration > 0) newStart = Math.min(newStart, Math.max(0, duration - dur));
        updateCropSegment({ ...seg, startTime: newStart, endTime: newStart + dur });
      }
    };

    const onUp = () => {
      // Resume undo history so the final drag state is recorded as one snapshot
      if (dragInfo.type === 'move' || dragInfo.type === 'trim' ||
          dragInfo.type === 'trim-crop' || dragInfo.type === 'move-crop') {
        useTimelineStore.temporal.getState().resume();
      }
      // Make sure the FINAL scrub position lands on the video before we
      // tear down — otherwise releasing mid-frame can leave the
      // playhead one rAF behind the cursor.
      if (_scrubRafRef.current !== 0) {
        cancelAnimationFrame(_scrubRafRef.current);
        _scrubRafRef.current = 0;
      }
      if (_scrubPendingRef.current != null) {
        const finalT = _scrubPendingRef.current;
        _scrubPendingRef.current = null;
        try { setPlayheadRef.current?.(finalT); } catch { /* noop */ }
        try { onSeekRef.current?.(finalT); } catch { /* noop */ }
      }
      // Clear the scrub flag so VideoEditor's ``syncTime`` /
      // ``seeked`` handlers resume driving the playhead from
      // ``video.currentTime``. Doing this AFTER the final flush
      // guarantees the video has the freshest timestamp before the
      // timeupdate feedback loop wakes back up.
      if (dragInfo.type === 'scrub') {
        useTimelineStore.getState().setIsScrubbing(false);
      }
      useTimelineStore.getState().setSnapLine(null);
      setIsDragging(false);
      setDragInfo(null);
    };

    window.addEventListener('pointermove', onMove);
    window.addEventListener('pointerup', onUp);
    return () => {
      window.removeEventListener('pointermove', onMove);
      window.removeEventListener('pointerup', onUp);
      // Cancel any outstanding scrub flush so an unmount mid-drag
      // doesn't fire ``setPlayhead`` after the component is gone.
      // NOTE: this intentionally does NOT clear ``_scrubPendingRef`` —
      // if the effect is re-mounting (dep churn mid-drag, not a real
      // unmount), the freshly queued buffered time must survive. The
      // new effect body either schedules a new rAF on the next pointer
      // move or the ``onUp`` handler flushes it on release.
      if (_scrubRafRef.current !== 0) {
        cancelAnimationFrame(_scrubRafRef.current);
        _scrubRafRef.current = 0;
      }
      // Safety: if component unmounts during a drag of any kind, resume
      // undo history. The previous version only handled ``move`` /
      // ``trim`` and would leave the temporal store paused after a
      // crop-segment drag mid-unmount, breaking undo for the rest of
      // the session.
      if (
        dragInfo.type === 'move' ||
        dragInfo.type === 'trim' ||
        dragInfo.type === 'trim-crop' ||
        dragInfo.type === 'move-crop'
      ) {
        useTimelineStore.temporal.getState().resume();
      }
      // Safety: clear the scrub flag so the video's timeupdate loop
      // stays wired up if this unmount was a teardown rather than a
      // dep-driven re-mount mid-drag.
      if (dragInfo.type === 'scrub') {
        useTimelineStore.getState().setIsScrubbing(false);
      }
    };
  // ``playhead`` / ``onSeek`` / ``setPlayhead`` are intentionally
  // absent from this dep list. They're read through refs
  // (``playheadRef``, ``onSeekRef``, ``setPlayheadRef``) so a parent
  // re-render with a new ``onSeek`` identity doesn't tear down the
  // drag listeners mid-gesture. Listing them here was the root cause
  // of the "playhead stays stationary" bug.
  //
  // ``sceneCuts`` is consumed by ``findSnapTarget`` inside ``onMove`` —
  // omitting it here meant new scene cuts (arriving asynchronously
  // while the user kept the drag handle held) wouldn't snap.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isDragging, dragInfo, items, pps, scrollX, snapEnabled, sceneCuts, getTimeFromX, getTrackFromY, updateItem, setScrollX]);

  // ── Hover ──────────────────────────────────────────────────────────────────
  const onPointerMove = useCallback((e) => {
    if (isDragging) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    if (x < LABEL_WIDTH) { setHoverTime(null); canvas.style.cursor = 'default'; return; }

    setHoverTime(getTimeFromX(e.clientX));

    if (spaceHeld) {
      canvas.style.cursor = 'grab';
      return;
    }

    if (activeTool === 'razor') {
      canvas.style.cursor = 'crosshair';
      return;
    }

    // Check if hovering near the playhead handle in the ruler area for grab cursor
    const phPixelX = rect.left + LABEL_WIDTH + playhead * pps - scrollX;
    const mouseY = e.clientY - rect.top;
    const isInRulerArea = mouseY <= RULER_HEIGHT + 8;
    if (isInRulerArea && Math.abs(e.clientX - phPixelX) <= PLAYHEAD_GRAB_WIDTH) {
      canvas.style.cursor = 'col-resize';
      return;
    }

    const hit = hitTestItem(e.clientX, e.clientY);
    if (hit) {
      canvas.style.cursor = hit.edge === 'left' || hit.edge === 'right' ? 'col-resize' : 'grab';
      return;
    }
    const cropHit = hitTestCropSegment(e.clientX, e.clientY);
    if (cropHit) {
      canvas.style.cursor = cropHit.edge === 'left' || cropHit.edge === 'right' ? 'col-resize' : 'grab';
      return;
    }
    if (!isInRulerArea && Math.abs(e.clientX - phPixelX) <= Math.max(PLAYHEAD_GRAB_WIDTH / 2, 8)) {
      // Show col-resize cursor on the playhead line outside ruler when no item is under cursor
      canvas.style.cursor = 'col-resize';
    } else {
      canvas.style.cursor = 'pointer';
    }
  }, [isDragging, getTimeFromX, hitTestItem, hitTestCropSegment, activeTool, spaceHeld, playhead, pps, scrollX]);

  const onPointerLeave = useCallback(() => setHoverTime(null), []);

  // ── Zoom via Ctrl+Wheel ────────────────────────────────────────────────────
  const onWheel = useCallback((e) => {
    if (e.ctrlKey || e.metaKey) {
      e.preventDefault();
      const delta = e.deltaY > 0 ? -0.1 : 0.1;
      setZoom(zoom + delta);
    } else {
      setScrollX(scrollX + e.deltaX + (e.shiftKey ? e.deltaY : 0));
    }
  }, [zoom, scrollX, setZoom, setScrollX]);

  // ── Drop from media library ────────────────────────────────────────────────
  const onDrop = useCallback((e) => {
    e.preventDefault();
    const data = e.dataTransfer.getData('application/x-clipai-media');
    if (!data) return;
    try {
      const media = JSON.parse(data);
      const time = getTimeFromX(e.clientX);
      const dropTrack = getTrackFromY(e.clientY);
      if (!dropTrack) return;
      if (dropTrack.locked) return; // Cannot drop items onto locked tracks

      // addItem auto-routes to the correct compatible track if the target is incompatible
      addItem({
        trackId: dropTrack.id,
        type: media.type,
        mediaRef: media.id,
        start: time,
        end: time + (media.duration || 5),
      });
    } catch { /* invalid data */ }
  }, [getTimeFromX, getTrackFromY, addItem]);

  const onDragOver = useCallback((e) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'copy';
  }, []);

  // ── Context menu ───────────────────────────────────────────────────────────
  const onContextMenu = useCallback((e) => e.preventDefault(), []);

  const handleContextAction = useCallback((action) => {
    if (!contextMenu) return;
    const { time, item } = contextMenu;
    // Block destructive actions on locked tracks
    if (item) {
      const itemTrack = tracks.find((t) => t.id === item.trackId);
      if (itemTrack?.locked && (action === 'split' || action === 'delete' || action === 'duplicate')) {
        setContextMenu(null);
        return;
      }
    }
    switch (action) {
      case 'split':
        if (item) splitItem(item.id, time);
        break;
      case 'delete':
        if (item) removeItem(item.id);
        break;
      case 'duplicate':
        if (item) useTimelineStore.getState().duplicateItem(item.id);
        break;
      case 'group': {
        const ids = useTimelineStore.getState().selectedItemIds;
        if (ids.length >= 2) groupItems(ids);
        break;
      }
      case 'ungroup': {
        const ids = useTimelineStore.getState().selectedItemIds;
        if (ids.length > 0) ungroupItems(ids);
        break;
      }
    }
    setContextMenu(null);
  }, [contextMenu, splitItem, removeItem, tracks, groupItems, ungroupItems]);

  useEffect(() => {
    if (!contextMenu) return;
    const close = () => setContextMenu(null);
    window.addEventListener('click', close);
    return () => window.removeEventListener('click', close);
  }, [contextMenu]);

  // ── Compute canvas height ──────────────────────────────────────────────────
  const canvasHeight = RULER_HEIGHT + tracks.length * (TRACK_HEIGHT + TRACK_GAP) + 12;

  return (
    <div ref={containerRef} className="ve-multi-timeline" style={{ position: 'relative', height: '100%' }}>
      {/* Toolbar row */}
      <div className="ve-multi-timeline__toolbar">
        <button
          className="ve-btn"
          onClick={() => setZoom(Math.max(0.01, zoom - 0.2))}
          title="Zoom out"
          style={{ fontSize: 12, padding: '2px 6px', minWidth: 24, minHeight: 24 }}
        >
          -
        </button>
        <input
          type="range"
          min="0.01"
          max="10"
          step="0.01"
          value={zoom}
          onChange={(e) => setZoom(parseFloat(e.target.value))}
          className="ve-multi-timeline__zoom-slider"
        />
        <button
          className="ve-btn"
          onClick={() => setZoom(Math.min(10, zoom + 0.2))}
          title="Zoom in"
          style={{ fontSize: 12, padding: '2px 6px', minWidth: 24, minHeight: 24 }}
        >
          +
        </button>
        <button
          className="ve-btn"
          onClick={() => {
            // Fit entire content extent in view
            const canvas = canvasRef.current;
            if (canvas) {
              const maxEnd = items.length > 0
                ? Math.max(...items.map(it => it.end || 0))
                : duration || 30;
              const fitDuration = maxEnd * 1.05 || 30;
              const availableWidth = canvas.getBoundingClientRect().width - LABEL_WIDTH;
              const fitZoom = Math.max(0.01, availableWidth / (fitDuration * basePPS));
              setZoom(fitZoom);
              setScrollX(0);
            }
          }}
          title="Fit entire video in view"
          style={{ fontSize: 10, padding: '2px 8px', minWidth: 'auto', minHeight: 24, fontWeight: 600 }}
        >
          Fit
        </button>
        <button
          className={`ve-btn${snapEnabled ? ' ve-btn--active-snap' : ''}`}
          onClick={toggleSnap}
          title={`Snap: ${snapEnabled ? 'ON' : 'OFF'} (N)`}
          style={{ fontSize: 10, padding: '2px 6px', minWidth: 'auto', minHeight: 24 }}
        >
          Snap {snapEnabled ? 'ON' : 'OFF'}
        </button>
        <div style={{ flex: 1 }} />
        <div style={{ position: 'relative' }}>
          <button
            className="ve-btn"
            onClick={() => setShowAddTrack(!showAddTrack)}
            style={{ fontSize: 10, padding: '2px 8px', minHeight: 24 }}
          >
            + Track
          </button>
          {showAddTrack && (
            <div className="ve-multi-timeline__add-track-dropdown">
              {['video', 'audio', 'overlay', 'subtitle'].map(type => (
                <button
                  key={type}
                  onClick={() => { addTrack(type); setShowAddTrack(false); }}
                  className="ve-multi-timeline__add-track-option"
                >
                  {TRACK_ICONS[type]} {type.charAt(0).toUpperCase() + type.slice(1)}
                </button>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Canvas area with track header overlay */}
      <div style={{ position: 'relative' }}>
        {/* Track header controls — overlays the canvas label area */}
        {/* Styled like DaVinci Resolve / Premiere Pro: eye (visibility), mute, lock per track */}
        <div
          className="ve-multi-timeline__track-headers"
          style={{
            position: 'absolute',
            left: 0,
            top: 0,
            width: LABEL_WIDTH - 1,
            zIndex: 5,
            pointerEvents: 'none',
          }}
        >
          {/* Spacer for ruler */}
          <div style={{ height: RULER_HEIGHT }} />
          {tracks.map((track, trackIdx) => {
            const isHidden = track.visible === false;
            const isDragOver = dragOverTrackIdx === trackIdx && dragTrackIdx !== trackIdx;
            return (
              <div
                key={track.id}
                className="ve-multi-timeline__track-header"
                draggable
                onDragStart={(e) => {
                  setDragTrackIdx(trackIdx);
                  e.dataTransfer.effectAllowed = 'move';
                  e.dataTransfer.setData('text/plain', String(trackIdx));
                }}
                onDragOver={(e) => {
                  e.preventDefault();
                  e.dataTransfer.dropEffect = 'move';
                  setDragOverTrackIdx(trackIdx);
                }}
                onDragLeave={() => { if (dragOverTrackIdx === trackIdx) setDragOverTrackIdx(null); }}
                onDrop={(e) => {
                  e.preventDefault();
                  if (dragTrackIdx != null && dragTrackIdx !== trackIdx) {
                    reorderTracks(dragTrackIdx, trackIdx);
                  }
                  setDragTrackIdx(null);
                  setDragOverTrackIdx(null);
                }}
                onDragEnd={() => { setDragTrackIdx(null); setDragOverTrackIdx(null); }}
                style={{
                  height: TRACK_HEIGHT,
                  marginBottom: TRACK_GAP,
                  display: 'flex',
                  flexDirection: 'column',
                  justifyContent: 'center',
                  gap: 2,
                  padding: '2px 4px',
                  pointerEvents: 'auto',
                  opacity: isHidden ? 0.5 : (dragTrackIdx === trackIdx ? 0.4 : 1),
                  cursor: 'grab',
                  borderTop: isDragOver ? '2px solid var(--accent, #0A84FF)' : '2px solid transparent',
                  transition: 'opacity 0.15s, border-color 0.15s',
                }}
              >
                {/* Drag handle + track name */}
                <span style={{
                  fontSize: 10,
                  fontWeight: 500,
                  color: 'var(--ve-text, #ccc)',
                  whiteSpace: 'nowrap',
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  display: 'flex',
                  alignItems: 'center',
                  gap: 3,
                  minWidth: 0,
                  opacity: isHidden ? 0.5 : 0.8,
                }}>
                  <svg width="8" height="10" viewBox="0 0 8 10" fill="currentColor" style={{ opacity: 0.35, flexShrink: 0 }}>
                    <circle cx="2" cy="2" r="1" /><circle cx="6" cy="2" r="1" />
                    <circle cx="2" cy="5" r="1" /><circle cx="6" cy="5" r="1" />
                    <circle cx="2" cy="8" r="1" /><circle cx="6" cy="8" r="1" />
                  </svg>
                  {TRACK_ICONS[track.type] || ''}{' '}
                  {renamingTrackId === track.id ? (
                    <input
                      autoFocus
                      defaultValue={track.name}
                      onBlur={(e) => {
                        const val = e.target.value.trim();
                        if (val && val !== track.name) updateTrack(track.id, { name: val });
                        // Defer input removal so pointer events resolve their target
                        // before the DOM mutates (input → span swap)
                        requestAnimationFrame(() => setRenamingTrackId(null));
                      }}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter') { e.target.blur(); }
                        else if (e.key === 'Escape') { setRenamingTrackId(null); }
                        e.stopPropagation();
                      }}
                      onClick={(e) => e.stopPropagation()}
                      onMouseDown={(e) => e.stopPropagation()}
                      onPointerDown={(e) => e.stopPropagation()}
                      style={{
                        fontSize: 10, fontWeight: 500, width: '100%',
                        background: 'var(--ve-surface, #222)', color: 'var(--ve-text, #ccc)',
                        border: '1px solid var(--accent, #0A84FF)', borderRadius: 2,
                        padding: '0 2px', outline: 'none', minWidth: 0,
                      }}
                    />
                  ) : (
                    <span
                      onDoubleClick={(e) => { e.stopPropagation(); setRenamingTrackId(track.id); }}
                      style={{ cursor: 'text', minWidth: 0, overflow: 'hidden', textOverflow: 'ellipsis' }}
                      title="Double-click to rename"
                    >
                      {track.name}
                    </span>
                  )}
                  {track.type === 'subtitle' && isHidden && (
                    <span style={{
                      fontSize: 9,
                      color: 'var(--ve-text-muted, #999)',
                      marginLeft: 4,
                      opacity: 0.6,
                    }}>
                      (hidden)
                    </span>
                  )}
                </span>
                {/* Controls row */}
                <div style={{ display: 'flex', gap: 1 }}>
                  {/* Visibility toggle (eye icon) — preview only */}
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      toggleTrackVisibility(track.id);
                      // Sync subtitle track visibility to settings
                      if (track.type === 'subtitle' && onSubtitleVisibilityChange) {
                        onSubtitleVisibilityChange(!(track.visible !== false));
                      }
                    }}
                    title={isHidden ? `Show ${track.name} in preview` : `Hide ${track.name} from preview (still in export)`}
                    className="ve-multi-timeline__track-ctrl"
                    style={{
                      background: 'none', border: 'none', cursor: 'pointer',
                      padding: '2px', lineHeight: 1, fontSize: 12,
                      opacity: isHidden ? 0.4 : 0.7,
                      color: isHidden ? 'var(--ve-text-muted, #999)' : 'var(--ve-text, #666)',
                    }}
                  >
                    {isHidden ? (
                      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M17.94 17.94A10.07 10.07 0 0112 20c-7 0-11-8-11-8a18.45 18.45 0 015.06-5.94" />
                        <path d="M9.9 4.24A9.12 9.12 0 0112 4c7 0 11 8 11 8a18.5 18.5 0 01-2.16 3.19" />
                        <line x1="1" y1="1" x2="23" y2="23" />
                      </svg>
                    ) : (
                      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" />
                        <circle cx="12" cy="12" r="3" />
                      </svg>
                    )}
                  </button>
                  {/* Mute toggle */}
                  <button
                    onClick={(e) => { e.stopPropagation(); toggleTrackMute(track.id); }}
                    title={track.muted ? `Unmute ${track.name}` : `Mute ${track.name}`}
                    className="ve-multi-timeline__track-ctrl"
                    style={{
                      background: 'none', border: 'none', cursor: 'pointer',
                      padding: '2px', lineHeight: 1, fontSize: 9, fontWeight: 700,
                      opacity: track.muted ? 1 : 0.35,
                      color: track.muted ? 'var(--danger, #ef4444)' : 'var(--ve-text, #666)',
                    }}
                  >
                    M
                  </button>
                  {/* Lock toggle */}
                  <button
                    onClick={(e) => { e.stopPropagation(); toggleTrackLock(track.id); }}
                    title={track.locked ? `Unlock ${track.name}` : `Lock ${track.name}`}
                    className="ve-multi-timeline__track-ctrl"
                    style={{
                      background: 'none', border: 'none', cursor: 'pointer',
                      padding: '2px', lineHeight: 1, fontSize: 10,
                      opacity: track.locked ? 0.8 : 0.35,
                      color: 'var(--ve-text, #999)',
                    }}
                  >
                    {track.locked ? (
                      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <rect x="3" y="11" width="18" height="11" rx="2" ry="2" />
                        <path d="M7 11V7a5 5 0 0110 0v4" />
                      </svg>
                    ) : (
                      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <rect x="3" y="11" width="18" height="11" rx="2" ry="2" />
                        <path d="M7 11V7a5 5 0 019.9-1" />
                      </svg>
                    )}
                  </button>
                  {/* Reset subtitle timings button — only on subtitle tracks */}
                  {track.type === 'subtitle' && hasOriginalSubtitles && (
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        resetSubtitleTimings();
                      }}
                      title="Reset all subtitles to original timing from transcript"
                      className="ve-multi-timeline__track-ctrl"
                      style={{
                        background: 'none', border: 'none', cursor: 'pointer',
                        padding: '2px', lineHeight: 1, fontSize: 10,
                        opacity: 0.7,
                        color: 'var(--ve-text, #666)',
                      }}
                    >
                      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <polyline points="1 4 1 10 7 10" />
                        <path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10" />
                      </svg>
                    </button>
                  )}
                </div>
              </div>
            );
          })}
        </div>

        {/* Canvas */}
        <canvas
          ref={canvasRef}
          className="ve-multi-timeline__canvas"
          style={{ width: '100%', height: Math.max(canvasHeight, 280) }}
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
        onPointerLeave={onPointerLeave}
        onWheel={onWheel}
        onDrop={onDrop}
        onDragOver={onDragOver}
        onContextMenu={onContextMenu}
        />
      </div>

      {/* Context menu */}
      {contextMenu && (
        <div
          className="ve-multi-timeline__context-menu"
          style={{ left: contextMenu.x, top: contextMenu.y }}
        >
          {contextMenu.item ? (
            <>
              <button onClick={() => handleContextAction('split')}>Split at cursor</button>
              <button onClick={() => handleContextAction('delete')}>Delete</button>
              <button onClick={() => handleContextAction('duplicate')}>Duplicate</button>
              {selectedItemIds.length >= 2 && (
                <button onClick={() => handleContextAction('group')}>Group Selected</button>
              )}
              {contextMenu.item.groupId && (
                <button onClick={() => handleContextAction('ungroup')}>Ungroup</button>
              )}
            </>
          ) : (
            <div style={{ padding: '4px 8px', fontSize: 10, color: 'var(--ve-text-muted)' }}>
              No item selected
            </div>
          )}
        </div>
      )}
    </div>
  );
}
