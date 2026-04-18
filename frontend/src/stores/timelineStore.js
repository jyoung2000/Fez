import { create } from 'zustand';
import { immer } from 'zustand/middleware/immer';
import { temporal } from 'zundo';

// ── Default track setup ─────────────────────────────────────────────────────
const createDefaultTracks = () => [
  { id: 't1', type: 'subtitle', name: 'Subtitles', muted: false, locked: false, visible: true },
  { id: 'v2', type: 'overlay', name: 'Overlay', muted: false, locked: false, visible: true },
  { id: 'c1', type: 'crop', name: 'Crop', muted: false, locked: false, visible: true },
  { id: 'v1', type: 'video', name: 'Video', muted: false, locked: false, visible: true },
  { id: 'a2', type: 'audio', name: 'Music', muted: false, locked: false, visible: true },
  { id: 'a1', type: 'audio', name: 'Audio', muted: false, locked: false, visible: true },
];

// Track-item type compatibility map
// Defines which item types are allowed on each track type
export const TRACK_ALLOWED_TYPES = {
  video: ['video'],
  overlay: ['text', 'shape', 'image', 'overlay'],
  crop: ['crop'],
  audio: ['audio'],
  subtitle: ['subtitle'],
};

// Fixed compositing priority by track type.
// Higher values render on top. This ensures overlays and subtitles
// are ALWAYS rendered above the video track, regardless of the visual
// arrangement in the timeline UI.
export const TRACK_COMPOSITING_PRIORITY = {
  audio: 0,
  video: 1,
  overlay: 2,
  subtitle: 3,
};

/**
 * Get the compositing order for a track.
 *
 * Two-axis sort: TYPE PRIORITY first (subtitles must always sit above
 * overlay which must sit above video which sits above audio), then
 * INDEX inside the same type (so dragging two video tracks reorders
 * them visually). The previous implementation only used array index,
 * which let a user accidentally hide subtitles by reordering tracks.
 *
 * @param {Object} track - The track object
 * @param {Array} tracks - The full tracks array (needed to find position)
 * @returns {number} Compositing order (higher = rendered later = on top)
 */
export function getCompositingOrder(track, tracks) {
  if (!track || !tracks) return 0;
  const idx = tracks.findIndex(t => t.id === track.id);
  if (idx < 0) return 0;
  // Pack type-priority into the high bits and the inverted UI index
  // into the low bits. tracks.length is bounded well below 1000 in
  // practice so the multiply is safe.
  const typePriority = TRACK_COMPOSITING_PRIORITY[track.type] ?? 1;
  const localOrder = tracks.length - idx;
  return typePriority * 1000 + localOrder;
}

// Check if an item type is compatible with a track type
function isTrackCompatible(itemType, trackType) {
  const allowed = TRACK_ALLOWED_TYPES[trackType];
  return allowed ? allowed.includes(itemType) : false;
}

// Find the first compatible track for an item type
function findCompatibleTrack(tracks, itemType) {
  return tracks.find((t) => isTrackCompatible(itemType, t.type)) || null;
}

/**
 * Compute the maximum allowed duration for a video/audio item based on its
 * source media duration and current trimStart. Non-media items return Infinity.
 */
export function getMaxItemDuration(item, mediaLibrary) {
  if (item.type !== 'video' && item.type !== 'audio') return Infinity;
  if (!item.mediaRef) return Infinity;
  const media = mediaLibrary.find(m => m.id === item.mediaRef);
  if (!media || !media.duration || media.duration <= 0) return Infinity;
  const trimStart = item.trimStart || 0;
  const maxDur = media.duration - trimStart;
  return maxDur > 0 ? maxDur : Infinity;
}

/**
 * Recompute timeline duration from the furthest item end.
 */
function computeTimelineDuration(items) {
  if (items.length === 0) return 0;
  return Math.max(...items.map(it => it.end || 0));
}

// ── Overlap prevention helpers ───────────────────────────────────────────────

/**
 * Get items on the same track, excluding specific IDs, sorted by start time.
 */
function getTrackSiblings(items, trackId, excludeIds) {
  const exclude = new Set(excludeIds);
  return items
    .filter(it => it.trackId === trackId && !exclude.has(it.id))
    .sort((a, b) => a.start - b.start);
}

/**
 * Clamp a move so the item doesn't overlap siblings on the same track.
 * Returns the clamped start time.
 */
function clampMoveToTrack(duration, newStart, siblings) {
  if (siblings.length === 0) return Math.max(0, newStart);

  // Build gaps: [0, first.start], [sib.end, nextSib.start], [last.end, Infinity]
  const gaps = [];
  gaps.push({ start: 0, end: siblings[0].start });
  for (let i = 0; i < siblings.length - 1; i++) {
    gaps.push({ start: siblings[i].end, end: siblings[i + 1].start });
  }
  gaps.push({ start: siblings[siblings.length - 1].end, end: Infinity });

  // Filter to gaps the item can actually fit in
  const fittingGaps = gaps.filter(g => (g.end - g.start) >= duration - 0.001);

  if (fittingGaps.length === 0) {
    // No gap large enough — find the gap the item would land in and clamp to boundary
    let bestGap = gaps[0];
    let bestDist = Infinity;
    for (const g of gaps) {
      const mid = g.end === Infinity ? g.start : (g.start + g.end) / 2;
      const dist = Math.abs(newStart + duration / 2 - mid);
      if (dist < bestDist) { bestDist = dist; bestGap = g; }
    }
    return Math.max(bestGap.start, Math.min(newStart, bestGap.end - duration));
  }

  // Find the fitting gap closest to the desired newStart
  let bestGap = fittingGaps[0];
  let bestDist = Infinity;
  for (const g of fittingGaps) {
    const clampedInGap = Math.max(g.start, Math.min(newStart, g.end - duration));
    const dist = Math.abs(clampedInGap - newStart);
    if (dist < bestDist) { bestDist = dist; bestGap = g; }
  }

  return Math.max(bestGap.start, Math.min(newStart, bestGap.end - duration));
}

/**
 * Clamp a left-edge trim so it doesn't overlap the previous sibling's end.
 */
function clampTrimLeft(newStart, siblings, itemEnd) {
  // Find the closest sibling that ends before our current end
  let maxStart = 0;
  for (const sib of siblings) {
    if (sib.end <= itemEnd && sib.end > maxStart) {
      maxStart = sib.end;
    }
  }
  return Math.max(newStart, maxStart);
}

/**
 * Clamp a right-edge trim so it doesn't overlap the next sibling's start.
 */
function clampTrimRight(newEnd, siblings, itemStart) {
  // Find the closest sibling that starts after our current start
  let minEnd = Infinity;
  for (const sib of siblings) {
    if (sib.start >= itemStart && sib.start < minEnd) {
      minEnd = sib.start;
    }
  }
  return Math.min(newEnd, minEnd);
}

/**
 * Resolve all overlaps on each track by pushing overlapping items forward.
 * Items are processed left-to-right; if an item overlaps a previous one,
 * its start is pushed to the previous item's end.
 */
function resolveAllOverlaps(items) {
  // Group items by track
  const byTrack = {};
  for (const item of items) {
    (byTrack[item.trackId] || (byTrack[item.trackId] = [])).push(item);
  }
  for (const trackId of Object.keys(byTrack)) {
    const trackItems = byTrack[trackId].sort((a, b) => a.start - b.start);
    for (let i = 1; i < trackItems.length; i++) {
      const prev = trackItems[i - 1];
      const curr = trackItems[i];
      if (curr.start < prev.end) {
        const dur = curr.end - curr.start;
        curr.start = prev.end;
        curr.end = curr.start + dur;
      }
    }
  }
}

/**
 * Resolve overlaps in crop segments by clamping endTime to next segment's startTime.
 * Modifies the array in place. Drops zero/negative-duration segments.
 * Returns the cleaned array.
 */
function resolveAllCropOverlaps(cropSegments) {
  if (!cropSegments || cropSegments.length <= 1) return cropSegments;
  cropSegments.sort((a, b) => a.startTime - b.startTime);
  for (let i = 0; i < cropSegments.length - 1; i++) {
    if (cropSegments[i].endTime > cropSegments[i + 1].startTime) {
      cropSegments[i].endTime = cropSegments[i + 1].startTime;
    }
  }
  // Remove degenerate segments
  for (let i = cropSegments.length - 1; i >= 0; i--) {
    if (cropSegments[i].endTime <= cropSegments[i].startTime) {
      cropSegments.splice(i, 1);
    }
  }
  return cropSegments;
}

/**
 * Centralized overlap invariant enforcement.
 *
 * This is the SINGLE safety net that guarantees no two elements on the same
 * track ever overlap, regardless of which action mutated the state.
 * Runs as a Zustand middleware after every set() call that touches items
 * or cropSegments.
 *
 * - For items: calls resolveAllOverlaps (push-forward strategy)
 * - For cropSegments: calls resolveAllCropOverlaps (clamp-endTime strategy)
 * - In development, logs a warning when overlaps are auto-fixed so bugs
 *   in individual action handlers are surfaced early.
 */
function enforceNoOverlapsMiddleware(config) {
  return (set, get, api) =>
    config(
      (...args) => {
        set(...args);

        // After every state update, check for and fix overlaps.
        // We read the current state and produce a new immutable update
        // via a normal set() call (no replace flag) so immer and Zustand
        // subscribers both see the change.
        const state = get();

        // Check items for overlaps
        let itemsNeedFix = false;
        if (state.items && state.items.length > 1) {
          const byTrack = {};
          for (const item of state.items) {
            (byTrack[item.trackId] || (byTrack[item.trackId] = [])).push(item);
          }
          for (const trackItems of Object.values(byTrack)) {
            if (trackItems.length < 2) continue;
            trackItems.sort((a, b) => a.start - b.start);
            for (let i = 1; i < trackItems.length; i++) {
              if (trackItems[i].start < trackItems[i - 1].end - 0.001) {
                itemsNeedFix = true;
                break;
              }
            }
            if (itemsNeedFix) break;
          }
        }

        // Check crop segments for overlaps
        let cropsNeedFix = false;
        if (state.cropSegments && state.cropSegments.length > 1) {
          const sorted = [...state.cropSegments].sort((a, b) => a.startTime - b.startTime);
          for (let i = 1; i < sorted.length; i++) {
            if (sorted[i].startTime < sorted[i - 1].endTime - 0.001) {
              cropsNeedFix = true;
              break;
            }
          }
        }

        if (itemsNeedFix || cropsNeedFix) {
          if (process.env.NODE_ENV === 'development') {
            if (itemsNeedFix) console.warn('[TimelineStore] Overlap detected in items — auto-fixing');
            if (cropsNeedFix) console.warn('[TimelineStore] Overlap detected in cropSegments — auto-fixing');
          }
          // IMPORTANT: `set` here is the outer (temporal) set, NOT the immer
          // producer. A function that mutates in place and returns undefined
          // is a no-op from Zustand's point of view and won't trigger
          // re-renders. We must build NEW arrays and return them as a
          // partial state update so Zustand detects the change.
          const update = {};
          if (itemsNeedFix) {
            const newItems = state.items.map((it) => ({ ...it }));
            resolveAllOverlaps(newItems);
            update.items = newItems;
          }
          if (cropsNeedFix) {
            const newCrops = state.cropSegments.map((c) => ({ ...c }));
            resolveAllCropOverlaps(newCrops);
            update.cropSegments = newCrops;
          }
          set(update);
        }
      },
      get,
      api,
    );
}

// ── Group ID helpers ────────────────────────────────────────────────────────
let _groupIdCounter = 1;
const nextGroupId = () => `g${_groupIdCounter++}`;

function hashGroupId(groupId) {
  let hash = 0;
  for (let i = 0; i < groupId.length; i++) {
    hash = ((hash << 5) - hash) + groupId.charCodeAt(i);
    hash |= 0;
  }
  return Math.abs(hash);
}

let _itemIdCounter = 1;
const nextItemId = () => `item-${_itemIdCounter++}`;
const nextMediaId = () => `media-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`;

// ── Store with Immer + Zundo ────────────────────────────────────────────────
const useTimelineStore = create(
  temporal(
    enforceNoOverlapsMiddleware(
    immer((set, get) => ({
      // ── Project settings ──
      project: {
        name: 'Untitled',
        resolution: { w: 1920, h: 1080 },
        fps: 30,
        aspectRatio: null,
        backgroundColor: '#000000',
      },

      // ── Timeline data ──
      tracks: createDefaultTracks(),
      items: [],
      mediaLibrary: [],

      // ── Original subtitle timings (snapshot from initFromClip for reset) ──
      _originalSubtitles: [],

      // ── Crop segments (subject tracking keyframes as editable segments) ──
      cropSegments: [],          // [{id, startTime, endTime, cropX, clusterId, isManualOverride, label}]
      selectedCropSegmentId: null,

      // ── Playback state (not tracked by undo) ──
      playhead: 0,
      duration: 0,
      zoom: 1.0,
      scrollX: 0,
      snapEnabled: true,
      snapLine: null, // { time: number } | null — active snap guide position
      // Scene-cut timestamps (seconds, timeline-absolute) populated by
      // VideoEditor from the analysis job. Used by Timeline.findSnapTarget
      // so trim handles "click" onto detected cuts. Empty when unknown.
      sceneCuts: [],
      // Loop in/out playback range. ``start`` and ``end`` are seconds
      // on the timeline; ``enabled`` toggles enforcement. The keyboard
      // shortcuts (I / O / Shift+L) drive these; PlaybackEngine reads
      // them to bounce playback back to ``start`` whenever it crosses
      // ``end`` while looping is on.
      loopRange: { start: null, end: null, enabled: false },
      selectedItemId: null,
      selectedItemIds: [],
      isPlaying: false,
      // True while the user is actively scrubbing the playhead by
      // dragging on the multi-track timeline. During a scrub the MOUSE
      // is the source of truth for the playhead position — the video
      // element's ``seeked`` / ``timeupdate`` events (which otherwise
      // snap ``setPlayhead`` to the nearest decoded keyframe via
      // ``syncTime``) must be suppressed, otherwise the playhead
      // visually lags or jumps backwards while the cursor is still
      // moving forward. Timeline sets this true on scrub pointerdown
      // and clears it on pointerup; VideoEditor reads it in its
      // timeupdate/seeked handlers to decide whether to propagate the
      // video position onto the timeline store.
      isScrubbing: false,

      // ── Selection & tools ──
      activeTool: 'select', // 'select' | 'razor' | 'text' | 'shape'
      activePanel: 'properties', // 'properties' | 'effects' | 'media'

      // ── Segments (backward compat with VideoEditor segment-based editing) ──
      segments: [],
      selectedSegmentId: null,

      // ── Clip settings (merged from VideoEditor useState) ──
      volume: 100,
      speed: 1.0,
      isMuted: false,
      trimStartOffset: 0,
      trimEndOffset: 0,

      // ── Subtitle settings (from ClipSettingsPanel) ──
      subtitleSettings: {},

      // ── Actions ──

      // Project
      setProject: (updates) => set((state) => {
        Object.assign(state.project, updates);
      }),

      setResolution: (w, h) => set((state) => {
        state.project.resolution = { w, h };
      }),

      setAspectRatio: (ar) => set((state) => {
        state.project.aspectRatio = ar;
      }),

      // Playback (not tracked by undo)
      setPlayhead: (t) => set({ playhead: t }),
      setIsPlaying: (v) => set({ isPlaying: v }),
      setIsScrubbing: (v) => set({ isScrubbing: !!v }),
      setDuration: (d) => set({ duration: d }),
      setZoom: (z) => set({ zoom: Math.max(0.01, Math.min(10, z)) }),
      setScrollX: (x) => set({ scrollX: Math.max(0, x) }),
      toggleSnap: () => set((state) => { state.snapEnabled = !state.snapEnabled; }),
      setSnapLine: (line) => set({ snapLine: line }),
      // Replace the scene-cut snap targets. Pass [] (or undefined) to
      // clear. Coerces numbers and drops invalid entries defensively.
      setSceneCuts: (cuts) => set({
        sceneCuts: Array.isArray(cuts)
          ? cuts.map((t) => Number(t)).filter((t) => Number.isFinite(t)).sort((a, b) => a - b)
          : [],
      }),
      // Update one or both endpoints of the loop range. Pass ``null``
      // for either to clear that endpoint. Auto-enables loop when both
      // ends become valid; auto-disables when an endpoint is cleared.
      setLoopRange: (patch) => set((state) => {
        const next = { ...state.loopRange, ...(patch || {}) };
        const startNum = typeof next.start === 'number' ? next.start : null;
        const endNum = typeof next.end === 'number' ? next.end : null;
        // Normalize: start must be < end. If user set them out of
        // order, swap so the loop region is always positive duration.
        if (startNum !== null && endNum !== null && startNum > endNum) {
          state.loopRange = { start: endNum, end: startNum, enabled: true };
        } else {
          state.loopRange = {
            start: startNum,
            end: endNum,
            enabled: startNum !== null && endNum !== null,
          };
        }
      }),
      toggleLoop: () => set((state) => {
        // Only flip enabled when both endpoints exist; otherwise no-op.
        if (state.loopRange.start === null || state.loopRange.end === null) return;
        state.loopRange.enabled = !state.loopRange.enabled;
      }),
      clearLoop: () => set({ loopRange: { start: null, end: null, enabled: false } }),
      setSelectedItemId: (id) => set({ selectedItemId: id, selectedItemIds: id ? [id] : [] }),
      setSelectedItemIds: (ids) => set({ selectedItemIds: ids, selectedItemId: ids[0] || null }),
      toggleSelectedItem: (itemId) => set((state) => {
        const idx = state.selectedItemIds.indexOf(itemId);
        if (idx >= 0) {
          state.selectedItemIds.splice(idx, 1);
          state.selectedItemId = state.selectedItemIds[0] || null;
        } else {
          state.selectedItemIds.push(itemId);
          state.selectedItemId = itemId;
        }
      }),
      setActiveTool: (tool) => set({ activeTool: tool }),
      setActivePanel: (panel) => set({ activePanel: panel }),

      // Segment state (backward compat)
      setSegments: (segs) => set({ segments: segs }),
      setSelectedSegmentId: (id) => set({ selectedSegmentId: id }),
      setVolume: (v) => set({ volume: v }),
      setSpeed: (s) => set({ speed: s }),
      setIsMuted: (m) => set({ isMuted: m }),
      setTrimStartOffset: (o) => set({ trimStartOffset: o }),
      setTrimEndOffset: (o) => set({ trimEndOffset: o }),
      setSubtitleSettings: (s) => set({ subtitleSettings: s }),

      // Track operations
      addTrack: (type, name) => set((state) => {
        const id = `${type.charAt(0)}${state.tracks.length + 1}-${Date.now().toString(36)}`;
        state.tracks.push({
          id, type,
          name: name || `${type} ${state.tracks.length + 1}`,
          muted: false, locked: false, visible: true,
        });
      }),

      removeTrack: (trackId) => set((state) => {
        state.tracks = state.tracks.filter(t => t.id !== trackId);
        state.items = state.items.filter(i => i.trackId !== trackId);
      }),

      updateTrack: (trackId, updates) => set((state) => {
        const track = state.tracks.find(t => t.id === trackId);
        if (track) Object.assign(track, updates);
      }),

      // Audio mute. Keeps ``muted`` in sync with the new explicit
      // ``audioMuted`` field — RenderEngine reads ``audioMuted`` first
      // and the legacy ``muted`` only as a fallback.
      toggleTrackMute: (trackId) => set((state) => {
        const track = state.tracks.find(t => t.id === trackId);
        if (!track) return;
        const next = !(track.audioMuted !== undefined ? track.audioMuted : track.muted);
        track.audioMuted = next;
        track.muted = next;
      }),

      toggleTrackLock: (trackId) => set((state) => {
        const track = state.tracks.find(t => t.id === trackId);
        if (track) track.locked = !track.locked;
      }),

      // Video pixel visibility. Keeps the legacy ``visible`` and the
      // new ``videoVisible`` aligned. RenderEngine prefers
      // ``videoVisible`` so a future schema bump can drop ``visible``.
      toggleTrackVisibility: (trackId) => set((state) => {
        const track = state.tracks.find(t => t.id === trackId);
        if (!track) return;
        const next = !(track.videoVisible !== undefined ? track.videoVisible : track.visible);
        track.videoVisible = next;
        track.visible = next;
      }),

      // Reorder tracks: move track at fromIndex to toIndex.
      // Compositing order is determined by array position — index 0 (top of UI)
      // renders on top. No separate order field needed.
      reorderTracks: (fromIndex, toIndex) => set((state) => {
        if (fromIndex === toIndex) return;
        if (fromIndex < 0 || fromIndex >= state.tracks.length) return;
        if (toIndex < 0 || toIndex >= state.tracks.length) return;
        const [moved] = state.tracks.splice(fromIndex, 1);
        state.tracks.splice(toIndex, 0, moved);
      }),

      // Item operations
      addItem: (item) => {
        const id = item.id || nextItemId();
        const itemType = item.type || 'video';

        set((state) => {
          // Auto-route to the correct track based on item type
          let trackId = item.trackId || 'v1';
          const track = state.tracks.find((t) => t.id === trackId);
          if (track) {
            if (track.locked || !isTrackCompatible(itemType, track.type)) {
              const correctTrack = findCompatibleTrack(state.tracks, itemType);
              if (correctTrack && !correctTrack.locked) trackId = correctTrack.id;
              else if (correctTrack?.locked) return; // All compatible tracks are locked
            }
          }
          const newItem = {
            id,
            trackId,
            type: itemType,
            mediaRef: item.mediaRef || null,
            start: item.start || 0,
            end: item.end || 0,
            trimStart: item.trimStart || 0,
            trimEnd: item.trimEnd || null,
            volume: item.volume ?? 1.0,
            speed: item.speed ?? 1.0,
            opacity: item.opacity ?? 1.0,
            position: item.position || { x: 50, y: 50 },
            size: item.size || (itemType === 'image' || itemType === 'overlay' ? { w: 30, h: 30 } : { w: 100, h: 100 }),
            transform: item.transform || { x: 0, y: 0, scaleX: 1, scaleY: 1, rotation: 0, opacity: 1 },
            effects: item.effects || { brightness: 0, contrast: 0, saturation: 0, blur: 0, hueRotate: 0, sepia: 0 },
            fadeIn: item.fadeIn || 0,
            fadeOut: item.fadeOut || 0,
            transition: item.transition || null,
            subtitleText: item.subtitleText || null,
            subtitleStyle: item.subtitleStyle || null,
            speaker: item.speaker || null,
            words: item.words || null,
            transcriptIndex: item.transcriptIndex ?? null,
            textContent: item.textContent || null,
            textStyle: item.textStyle || null,
            shapeType: item.shapeType || null,
            shapeStyle: item.shapeStyle || null,
            subjectX: item.subjectX ?? 50,
            clipSettings: item.clipSettings || null,
            groupId: item.groupId || null,
          };
          // Overlap prevention: clamp inside the Immer draft so concurrent
          // addItem calls each see the latest state
          const siblings = getTrackSiblings(state.items, trackId, [id]);
          if (siblings.length > 0) {
            const dur = newItem.end - newItem.start;
            const clampedStart = clampMoveToTrack(dur, newItem.start, siblings);
            newItem.start = clampedStart;
            newItem.end = clampedStart + dur;
          }
          state.items.push(newItem);
          resolveAllOverlaps(state.items);
          state.duration = Math.max(state.duration, newItem.end);
        });
        return id;
      },

      removeItem: (itemId) => set((state) => {
        const item = state.items.find(i => i.id === itemId);
        if (item) {
          const track = state.tracks.find(t => t.id === item.trackId);
          if (track?.locked) return; // Cannot remove items from locked tracks
        }
        state.items = state.items.filter(i => i.id !== itemId);
        resolveAllOverlaps(state.items);
        state.duration = computeTimelineDuration(state.items);
        if (state.selectedItemId === itemId) {
          state.selectedItemId = null;
          state.selectedItemIds = [];
        }
      }),

      // Batch-remove multiple items in a single undo snapshot
      removeItems: (itemIds) => set((state) => {
        const lockedTrackIds = new Set(state.tracks.filter(t => t.locked).map(t => t.id));
        const idsToRemove = new Set(
          itemIds.filter(id => {
            const item = state.items.find(i => i.id === id);
            return item && !lockedTrackIds.has(item.trackId);
          })
        );
        if (idsToRemove.size === 0) return;
        state.items = state.items.filter(i => !idsToRemove.has(i.id));
        resolveAllOverlaps(state.items);
        state.duration = computeTimelineDuration(state.items);
        if (idsToRemove.has(state.selectedItemId)) {
          state.selectedItemId = null;
        }
        state.selectedItemIds = state.selectedItemIds.filter(id => !idsToRemove.has(id));
      }),

      updateItem: (itemId, updates) => set((state) => {
        const item = state.items.find(i => i.id === itemId);
        if (!item) return;
        // Prevent modifications to items on locked tracks
        const currentTrack = state.tracks.find(t => t.id === item.trackId);
        if (currentTrack?.locked) return;
        // Validate track change: prevent moving items to incompatible tracks
        if (updates.trackId && updates.trackId !== item.trackId) {
          const targetTrack = state.tracks.find(t => t.id === updates.trackId);
          if (targetTrack && !isTrackCompatible(item.type, targetTrack.type)) {
            // Silently reject the track change — keep item on current track
            const { trackId, ...rest } = updates;
            Object.assign(item, rest);
            return;
          }
        }
        Object.assign(item, updates);
        // Overlap prevention: clamp start/end if they changed or item moved to new track
        if (updates.start !== undefined || updates.end !== undefined || updates.trackId !== undefined) {
          const targetTrackId = item.trackId;
          const siblings = getTrackSiblings(state.items, targetTrackId, [item.id]);
          if (siblings.length > 0) {
            const dur = item.end - item.start;
            if (updates.start !== undefined && updates.end !== undefined) {
              // Move: clamp both
              const clamped = clampMoveToTrack(dur, item.start, siblings);
              item.start = clamped;
              item.end = clamped + dur;
            } else if (updates.start !== undefined && updates.end === undefined) {
              // Left trim
              item.start = clampTrimLeft(item.start, siblings, item.end);
            } else if (updates.end !== undefined && updates.start === undefined) {
              // Right trim
              item.end = clampTrimRight(item.end, siblings, item.start);
            } else {
              // Track-only change (no explicit start/end): clamp to avoid overlaps on new track
              const clamped = clampMoveToTrack(dur, item.start, siblings);
              item.start = clamped;
              item.end = clamped + dur;
            }
          }
        }
        // Clamp end so video/audio items never exceed source media duration
        const maxDur = getMaxItemDuration(item, state.mediaLibrary);
        if (maxDur < Infinity) {
          const maxEnd = item.start + maxDur;
          if (item.end > maxEnd) {
            item.end = maxEnd;
          }
        }
        resolveAllOverlaps(state.items);
        state.duration = computeTimelineDuration(state.items);
      }),

      updateItemWithSnapshot: (itemId, updates) => set((state) => {
        const item = state.items.find(i => i.id === itemId);
        if (!item) return;
        Object.assign(item, updates);
        // Overlap prevention when timing or track changes
        if (updates.start !== undefined || updates.end !== undefined || updates.trackId !== undefined) {
          const siblings = getTrackSiblings(state.items, item.trackId, [item.id]);
          if (siblings.length > 0) {
            const dur = item.end - item.start;
            const clamped = clampMoveToTrack(dur, item.start, siblings);
            item.start = clamped;
            item.end = clamped + dur;
          }
        }
        resolveAllOverlaps(state.items);
        state.duration = computeTimelineDuration(state.items);
      }),

      // Reset all subtitle items to their original timing/position from initFromClip
      resetSubtitleTimings: () => set((state) => {
        const originals = state._originalSubtitles;
        if (!originals || originals.length === 0) return;

        // Remove all current subtitle items
        state.items = state.items.filter((it) => it.type !== 'subtitle');

        // Re-create subtitles from the original snapshot, resolving overlaps
        let lastEnd = 0;
        const sortedOrig = [...originals].sort((a, b) => a.start - b.start);
        sortedOrig.forEach((orig) => {
          const adjStart = Math.max(orig.start, lastEnd);
          if (adjStart >= orig.end) return; // Skip degenerate
          lastEnd = orig.end;
          state.items.push({
            id: nextItemId(),
            trackId: 't1',
            type: 'subtitle',
            mediaRef: null,
            start: adjStart,
            end: orig.end,
            trimStart: 0,
            trimEnd: null,
            volume: 1.0,
            speed: 1.0,
            opacity: 1.0,
            position: orig.position ? { ...orig.position } : { x: 50, y: 90 },
            size: { w: 100, h: 100 },
            transform: { x: 0, y: 0, scaleX: 1, scaleY: 1, rotation: 0, opacity: 1 },
            effects: {},
            fadeIn: 0,
            fadeOut: 0,
            subtitleText: orig.subtitleText,
            subtitleStyle: null,
            speaker: orig.speaker || null,
            transition: null,
            words: orig.words || null,
            transcriptIndex: orig.transcriptIndex ?? null,
          });
        });

        // Deselect if the selected item was a subtitle (it no longer exists)
        if (state.selectedItemId) {
          const stillExists = state.items.find((it) => it.id === state.selectedItemId);
          if (!stillExists) {
            state.selectedItemId = null;
            state.selectedItemIds = [];
          }
        }
      }),

      splitItem: (itemId, time) => set((state) => {
        const idx = state.items.findIndex(i => i.id === itemId);
        if (idx < 0) return;
        const item = state.items[idx];
        // Prevent splitting items on locked tracks
        const track = state.tracks.find(t => t.id === item.trackId);
        if (track?.locked) return;
        if (time <= item.start || time >= item.end) return;

        const newId = nextItemId();
        const offset = time - item.start;
        const item2 = {
          ...JSON.parse(JSON.stringify(item)),
          id: newId,
          start: time,
          trimStart: (item.trimStart || 0) + offset,
        };

        state.items[idx].end = time;
        state.items.push(item2);
        resolveAllOverlaps(state.items);
        state.duration = computeTimelineDuration(state.items);
        state.selectedItemId = newId;
        state.selectedItemIds = [newId];
      }),

      duplicateItem: (itemId) => set((state) => {
        const item = state.items.find(i => i.id === itemId);
        if (!item) return;
        const dur = item.end - item.start;
        const newId = nextItemId();
        const clone = {
          ...JSON.parse(JSON.stringify(item)),
          id: newId,
          start: item.end,
          end: item.end + dur,
        };
        // Clamp duplicated video/audio items to source media duration
        const maxDur = getMaxItemDuration(clone, state.mediaLibrary);
        if (maxDur < Infinity) {
          const maxEnd = clone.start + maxDur;
          if (clone.end > maxEnd) {
            clone.end = maxEnd;
          }
        }
        // Overlap prevention: clamp duplicate to a non-overlapping position
        const siblings = getTrackSiblings(state.items, clone.trackId, [newId]);
        if (siblings.length > 0) {
          const cloneDur = clone.end - clone.start;
          const clamped = clampMoveToTrack(cloneDur, clone.start, siblings);
          clone.start = clamped;
          clone.end = clamped + cloneDur;
        }
        state.items.push(clone);
        resolveAllOverlaps(state.items);
        state.duration = computeTimelineDuration(state.items);
        state.selectedItemId = newId;
        state.selectedItemIds = [newId];
      }),

      // Batch move items (for multi-select drag)
      moveItems: (itemIds, deltaTime, targetTrackId) => set((state) => {
        const idSet = new Set(itemIds);
        for (const id of itemIds) {
          const item = state.items.find(i => i.id === id);
          if (!item) continue;
          const dur = item.end - item.start;
          let newStart = Math.max(0, item.start + deltaTime);
          if (targetTrackId) item.trackId = targetTrackId;
          // Overlap prevention: exclude all items being moved together
          const siblings = getTrackSiblings(state.items, item.trackId, [...idSet]);
          if (siblings.length > 0) {
            newStart = clampMoveToTrack(dur, newStart, siblings);
          }
          item.start = newStart;
          item.end = newStart + dur;
        }
        resolveAllOverlaps(state.items);
        state.duration = computeTimelineDuration(state.items);
      }),

      // Group operations
      groupItems: (itemIds) => set((state) => {
        if (!itemIds || itemIds.length < 2) return;
        const gid = nextGroupId();
        for (const id of itemIds) {
          const item = state.items.find(i => i.id === id);
          if (item) item.groupId = gid;
        }
      }),

      ungroupItems: (itemIds) => set((state) => {
        if (!itemIds || itemIds.length === 0) return;
        for (const id of itemIds) {
          const item = state.items.find(i => i.id === id);
          if (item) item.groupId = null;
        }
      }),

      // Media library
      addMedia: (media) => {
        const id = media.id || nextMediaId();
        const entry = {
          id,
          type: media.type || 'video',
          filename: media.filename || 'untitled',
          duration: media.duration || 0,
          url: media.url || '',
          thumbnailUrl: media.thumbnailUrl || '',
          waveformData: media.waveformData || [],
        };
        set((state) => {
          state.mediaLibrary.push(entry);
        });
        return id;
      },

      updateMedia: (mediaId, updates) => set((state) => {
        const item = state.mediaLibrary.find(m => m.id === mediaId);
        if (item) {
          // Never mutate the `id` field — it would break mediaRef lookups
          // from timeline items that still reference the original local ID.
          const { id: _ignoredId, ...safeUpdates } = updates;
          Object.assign(item, safeUpdates);
        }
      }),

      // Replace a media entry's ID and update all timeline items that reference it.
      // Use this when the backend returns a server-assigned ID after upload.
      replaceMediaId: (oldId, newId, updates) => set((state) => {
        const entry = state.mediaLibrary.find(m => m.id === oldId);
        if (entry) {
          Object.assign(entry, updates || {}, { id: newId });
          for (const item of state.items) {
            if (item.mediaRef === oldId) {
              item.mediaRef = newId;
            }
          }
        }
      }),

      removeMedia: (mediaId) => set((state) => {
        state.mediaLibrary = state.mediaLibrary.filter(m => m.id !== mediaId);
        state.items = state.items.filter(i => i.mediaRef !== mediaId);
      }),

      removeMediaBatch: (mediaIds) => set((state) => {
        const idSet = new Set(mediaIds);
        state.mediaLibrary = state.mediaLibrary.filter(m => !idSet.has(m.id));
        state.items = state.items.filter(i => !idSet.has(i.mediaRef));
      }),

      // Initialize timeline with clip data (backward compat)
      initFromClip: (clipData) => {
        const { src, clipStart, clipEnd, subtitleSegments, hookText } = clipData;
        const duration = clipEnd - clipStart;

        const baseMediaId = nextMediaId();
        const mediaLibrary = [
          { id: baseMediaId, type: 'video', filename: 'Source Video', duration, url: src, thumbnailUrl: '', waveformData: [] },
        ];

        const items = [
          {
            id: nextItemId(),
            trackId: 'v1',
            type: 'video',
            mediaRef: baseMediaId,
            start: 0,
            end: duration,
            trimStart: clipStart,
            trimEnd: clipEnd,
            volume: 1.0,
            speed: 1.0,
            opacity: 1.0,
            position: { x: 50, y: 50 },
            size: { w: 100, h: 100 },
            transform: { x: 0, y: 0, scaleX: 1, scaleY: 1, rotation: 0, opacity: 1 },
            effects: { brightness: 0, contrast: 0, saturation: 0, blur: 0, hueRotate: 0, sepia: 0 },
            fadeIn: 0,
            fadeOut: 0,
            transition: null,
          },
          {
            id: nextItemId(),
            trackId: 'a1',
            type: 'audio',
            mediaRef: baseMediaId,
            start: 0,
            end: duration,
            trimStart: clipStart,
            trimEnd: clipEnd,
            volume: 1.0,
            speed: 1.0,
            opacity: 1.0,
            position: { x: 0, y: 0 },
            size: { w: 100, h: 100 },
            transform: { x: 0, y: 0, scaleX: 1, scaleY: 1, rotation: 0, opacity: 1 },
            effects: {},
            fadeIn: 0,
            fadeOut: 0,
            transition: null,
          },
        ];

        // Add subtitle items if provided — resolve overlaps from server data
        let lastSubEnd = 0;
        if (Array.isArray(subtitleSegments)) {
          // Sort by start time and pre-compute relative times for overlap prevention
          const sorted = subtitleSegments
            .filter(seg => seg.end > clipStart && seg.start < clipEnd)
            .sort((a, b) => a.start - b.start);

          sorted.forEach((seg, segIdx) => {
            const clampedStart = Math.max(seg.start, clipStart);
            const clampedEnd = Math.min(seg.end, clipEnd);
            // Ensure no overlap with the previous subtitle on the same track
            const relStart = clampedStart - clipStart;
            const relEnd = clampedEnd - clipStart;
            const adjStart = Math.max(relStart, lastSubEnd);
            if (adjStart >= relEnd) return; // Skip degenerate segments
            // Preserve word-level timestamps for accurate active word highlighting
            let segWords = null;
            if (seg.words && Array.isArray(seg.words)) {
              segWords = seg.words
                .filter(w => w.end > clipStart && w.start < clipEnd)
                .map(w => ({
                  ...w,
                  start: w.start - clipStart,
                  end: w.end - clipStart,
                }));
            }
            // Extend segment end to cover last word if word timestamps exceed it,
            // but NEVER extend past the next segment's start (prevents overlap)
            let effectiveEnd = relEnd;
            if (segWords && segWords.length > 0) {
              const lastWordEnd = Math.max(...segWords.map(w => w.end));
              if (lastWordEnd > effectiveEnd) {
                effectiveEnd = Math.min(lastWordEnd + 0.05, duration);
              }
            }
            // Cap at next segment's start to prevent overlap
            const nextSeg = sorted[segIdx + 1];
            if (nextSeg) {
              const nextRelStart = Math.max(nextSeg.start, clipStart) - clipStart;
              effectiveEnd = Math.min(effectiveEnd, nextRelStart);
            }
            if (adjStart >= effectiveEnd) return; // Skip if capping made it degenerate
            lastSubEnd = effectiveEnd;
            items.push({
              id: nextItemId(),
              trackId: 't1',
              type: 'subtitle',
              mediaRef: null,
              start: adjStart,
              end: effectiveEnd,
              trimStart: 0,
              trimEnd: null,
              volume: 1.0,
              speed: 1.0,
              opacity: 1.0,
              position: { x: 50, y: 90 },
              size: { w: 100, h: 100 },
              transform: { x: 0, y: 0, scaleX: 1, scaleY: 1, rotation: 0, opacity: 1 },
              effects: {},
              fadeIn: 0,
              fadeOut: 0,
              subtitleText: seg.text,
              subtitleStyle: null,
              speaker: seg.speaker || null,
              transition: null,
              words: segWords,
              transcriptIndex: segIdx,
            });
          });
        }

        // Add CTA/hook text overlay item if provided
        if (hookText && hookText.length > 3) {
          items.push({
            id: nextItemId(),
            trackId: 'v2',
            type: 'text',
            mediaRef: null,
            start: 0,
            end: Math.min(3.0, duration),
            trimStart: 0,
            trimEnd: null,
            volume: 1.0,
            speed: 1.0,
            opacity: 1.0,
            position: { x: 50, y: 40 },
            size: { w: 80, h: 20 },
            transform: { x: 0, y: 0, scaleX: 1, scaleY: 1, rotation: 0, opacity: 1 },
            effects: {},
            fadeIn: 0.3,
            fadeOut: 0.5,
            transition: null,
            textContent: hookText,
            textStyle: {
              fontSize: 28,
              fontWeight: 700,
              fontFamily: 'sans-serif',
              color: '#FFFFFF',
              textAlign: 'center',
              backgroundColor: 'rgba(0,0,0,0.7)',
              padding: 8,
              borderRadius: 8,
              outlineWidth: 0,
              outlineColor: '#000000',
            },
          });
        }

        // Resolve any overlaps from source data before snapshotting
        resolveAllOverlaps(items);

        // Snapshot original subtitle timings so user can reset after accidental moves
        const originalSubtitles = items
          .filter((it) => it.type === 'subtitle')
          .map((it) => ({ id: it.id, start: it.start, end: it.end, subtitleText: it.subtitleText, speaker: it.speaker, words: it.words, position: { ...it.position }, transcriptIndex: it.transcriptIndex }));

        set({
          tracks: createDefaultTracks(),
          items,
          mediaLibrary,
          _originalSubtitles: originalSubtitles,
          cropSegments: [],
          selectedCropSegmentId: null,
          playhead: 0,
          duration,
          zoom: 1.0,
          scrollX: 0,
          selectedItemId: null,
          selectedItemIds: [],
          isPlaying: false,
          isScrubbing: false,
          activeTool: 'select',
        });
      },

      // ── Crop segment actions ──
      setCropSegments: (segments) => {
        // Enforce no-overlap: sort by startTime, clamp each segment's endTime
        // to the next segment's startTime.
        if (!segments || segments.length === 0) {
          set({ cropSegments: [], selectedCropSegmentId: null });
          return;
        }
        const sorted = [...segments].sort((a, b) => a.startTime - b.startTime);
        for (let i = 0; i < sorted.length - 1; i++) {
          if (sorted[i].endTime > sorted[i + 1].startTime) {
            sorted[i] = { ...sorted[i], endTime: sorted[i + 1].startTime };
          }
        }
        // Drop degenerate segments (zero or negative duration)
        const valid = sorted.filter(s => s.endTime > s.startTime);
        set({ cropSegments: valid, selectedCropSegmentId: null });
      },
      selectCropSegment: (id) => set({ selectedCropSegmentId: id }),
      updateCropSegment: (updated) => set((state) => {
        const idx = state.cropSegments.findIndex(s => s.id === updated.id);
        if (idx < 0) return;
        const merged = { ...updated, isManualOverride: true };
        // Clamp timing against neighbors to prevent overlap
        if (idx > 0) {
          merged.startTime = Math.max(merged.startTime, state.cropSegments[idx - 1].startTime);
          if (merged.startTime < state.cropSegments[idx - 1].endTime) {
            // Push start to prev's end (can't overlap)
            merged.startTime = state.cropSegments[idx - 1].endTime;
          }
        }
        if (idx < state.cropSegments.length - 1) {
          const next = state.cropSegments[idx + 1];
          if (merged.endTime > next.startTime) {
            merged.endTime = next.startTime;
          }
        }
        if (merged.endTime <= merged.startTime) return; // Degenerate
        state.cropSegments[idx] = merged;
      }),
      splitCropSegment: (segmentId, splitTime) => set((state) => {
        const seg = state.cropSegments.find(s => s.id === segmentId);
        if (!seg || splitTime <= seg.startTime || splitTime >= seg.endTime) return {};
        return {
          cropSegments: state.cropSegments.flatMap(s => {
            if (s.id !== segmentId) return [s];
            return [
              { ...s, id: `${s.id}-a`, endTime: splitTime },
              { ...s, id: `${s.id}-b`, startTime: splitTime },
            ];
          }),
        };
      }),
      mergeCropWithNext: (segmentId) => set((state) => {
        const idx = state.cropSegments.findIndex(s => s.id === segmentId);
        if (idx < 0 || idx >= state.cropSegments.length - 1) return {};
        const merged = { ...state.cropSegments[idx], endTime: state.cropSegments[idx + 1].endTime };
        const segs = [...state.cropSegments];
        segs.splice(idx, 2, merged);
        return { cropSegments: segs };
      }),
      resetCropSegment: (segmentId, originalX) => set((state) => ({
        cropSegments: state.cropSegments.map(s =>
          s.id === segmentId ? { ...s, cropX: originalX, isManualOverride: false } : s
        ),
      })),

      // Reset to defaults
      reset: () => {
        set({
          project: {
            name: 'Untitled',
            resolution: { w: 1920, h: 1080 },
            fps: 30,
            aspectRatio: null,
            backgroundColor: '#000000',
          },
          tracks: createDefaultTracks(),
          items: [],
          mediaLibrary: [],
          playhead: 0,
          duration: 0,
          zoom: 1.0,
          scrollX: 0,
          snapEnabled: true,
          selectedItemId: null,
          selectedItemIds: [],
          isPlaying: false,
          isScrubbing: false,
          activeTool: 'select',
          segments: [],
          selectedSegmentId: null,
          volume: 100,
          speed: 1.0,
          isMuted: false,
          trimStartOffset: 0,
          trimEndOffset: 0,
          subtitleSettings: {},
          _originalSubtitles: [],
        });
      },

      // Get items visible at a specific time
      getVisibleItems: (time) => {
        return get().items.filter((item) => time >= item.start && time < item.end);
      },

      // Get items on a specific track
      getTrackItems: (trackId) => {
        return get().items.filter((item) => item.trackId === trackId);
      },

      // Export state for persistence
      exportState: () => {
        const { tracks, items, mediaLibrary, duration, project, segments, subtitleSettings } = get();
        return { tracks, items, mediaLibrary, duration, project, segments, subtitleSettings };
      },

      // Import state from persistence — preserves saved track order so that
      // user reordering and added tracks survive container restarts.
      importState: (state) => {
        if (state && state.tracks && state.items) {
          // Use saved track order as the source of truth. The array order
          // IS the compositing order (index 0 = top of UI = renders on top).
          // Only add missing default tracks if they were removed from saved state.
          const defaults = createDefaultTracks();
          const savedIds = new Set(state.tracks.map(t => t.id));

          // Start with all saved tracks in their persisted order
          const merged = state.tracks.map(t => ({ ...t }));

          // Append any default tracks missing from saved state (safety net)
          for (const def of defaults) {
            if (!savedIds.has(def.id)) {
              merged.push({ ...def });
            }
          }

          // Resolve any overlaps in persisted data
          const rehydratedItems = [...state.items];
          resolveAllOverlaps(rehydratedItems);

          set({
            tracks: merged,
            items: rehydratedItems,
            mediaLibrary: state.mediaLibrary || [],
            duration: state.duration || 0,
            project: state.project || { name: 'Untitled', resolution: { w: 1920, h: 1080 }, fps: 30, aspectRatio: null, backgroundColor: '#000000' },
            segments: state.segments || [],
            subtitleSettings: state.subtitleSettings || {},
          });
        }
      },
    }))),
    {
      // Zundo temporal config
      limit: 100,
      // Track undo-able state (exclude playback position, UI transient state).
      // duration is included so undo/redo properly restores the timeline extent.
      partialize: (state) => ({
        tracks: state.tracks,
        items: state.items,
        duration: state.duration,
        project: state.project,
        segments: state.segments,
      }),
      // Only create undo snapshots when tracked properties actually change.
      // Immer's structural sharing keeps the same reference for unchanged
      // sub-trees, so reference equality per field is both correct and fast.
      // Without this, every set() call (e.g. setPlayhead at 30fps) creates a
      // snapshot, flooding the undo stack with no-op entries and pushing out
      // real changes like element deletions.
      equality: (pastState, currentState) =>
        pastState.tracks === currentState.tracks &&
        pastState.items === currentState.items &&
        pastState.duration === currentState.duration &&
        pastState.project === currentState.project &&
        pastState.segments === currentState.segments,
    }
  )
);

export { hashGroupId };
export default useTimelineStore;
