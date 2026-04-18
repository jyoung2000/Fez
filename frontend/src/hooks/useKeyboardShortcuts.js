import { useEffect, useCallback, useRef } from 'react';
import useTimelineStore from '../stores/timelineStore';

/**
 * Centralized keyboard shortcut handling for the video editor.
 * Covers tool selection, transport controls, undo/redo, and clip manipulation.
 */
export default function useKeyboardShortcuts({
  enabled = true,
  onTogglePlay,
  onSeek,
  onSkipTime,
  onToggleMute,
  onShuttleSpeed,
  // Wall-clock seek bounds for Home / End. When the editor is showing
  // a clip slice (clipStart > 0), Home / End must seek to the **wall
  // clock** boundaries of the clip, not the timeline-relative 0 /
  // duration. Defaulting to ``null`` preserves the legacy behavior.
  clipRange = null,
} = {}) {
  const setActiveTool = useTimelineStore((s) => s.setActiveTool);
  const removeItem = useTimelineStore((s) => s.removeItem);
  const removeItems = useTimelineStore((s) => s.removeItems);
  const splitItem = useTimelineStore((s) => s.splitItem);

  // Stable refs for all callback props — prevents effect re-registration
  // from killing arrow hold intervals when parent re-renders
  const arrowHoldRef = useRef({ key: null, interval: null });
  const onSkipTimeRef = useRef(onSkipTime);
  onSkipTimeRef.current = onSkipTime;
  const onTogglePlayRef = useRef(onTogglePlay);
  onTogglePlayRef.current = onTogglePlay;
  const onSeekRef = useRef(onSeek);
  onSeekRef.current = onSeek;
  const onToggleMuteRef = useRef(onToggleMute);
  onToggleMuteRef.current = onToggleMute;
  const onShuttleSpeedRef = useRef(onShuttleSpeed);
  onShuttleSpeedRef.current = onShuttleSpeed;
  const clipRangeRef = useRef(clipRange);
  clipRangeRef.current = clipRange;

  const handleKeyDown = useCallback((e) => {
    if (!enabled) return;

    // Don't capture when typing in inputs/textareas/contenteditable
    const tag = e.target.tagName;
    if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;
    if (e.target.contentEditable === 'true') return;

    // Ctrl/Cmd shortcuts
    if (e.ctrlKey || e.metaKey) {
      switch (e.code) {
        case 'KeyZ':
          e.preventDefault();
          if (e.shiftKey) {
            useTimelineStore.temporal.getState().redo();
          } else {
            useTimelineStore.temporal.getState().undo();
          }
          return;
        case 'KeyY':
          e.preventDefault();
          useTimelineStore.temporal.getState().redo();
          return;
        case 'KeyS':
          e.preventDefault();
          // TODO: Project save
          return;
        case 'KeyA':
          e.preventDefault();
          // Read items from store at call time (not closure)
          const allIds = useTimelineStore.getState().items.map(i => i.id);
          useTimelineStore.getState().setSelectedItemIds(allIds);
          return;
        case 'KeyG': {
          e.preventDefault();
          const state = useTimelineStore.getState();
          const selIds = state.selectedItemIds;
          if (e.shiftKey) {
            // Ctrl+Shift+G: Ungroup selected items
            if (selIds.length > 0) {
              state.ungroupItems(selIds);
            }
          } else {
            // Ctrl+G: Toggle — ungroup if all selected share the same group, otherwise group
            if (selIds.length >= 2) {
              const selItems = selIds.map(id => state.items.find(i => i.id === id)).filter(Boolean);
              const firstGroupId = selItems[0]?.groupId;
              const allSameGroup = firstGroupId && selItems.every(i => i.groupId === firstGroupId);
              if (allSameGroup) {
                state.ungroupItems(selIds);
              } else {
                state.groupItems(selIds);
              }
            }
          }
          return;
        }
      }
    }

    switch (e.code) {
      // Transport
      case 'Space':
        e.preventDefault();
        onTogglePlayRef.current?.();
        break;
      case 'ArrowLeft':
      case 'ArrowRight': {
        e.preventDefault();
        const dir = e.code === 'ArrowLeft' ? -1 : 1;
        const delta = e.shiftKey ? dir : dir / 30;

        const hold = arrowHoldRef.current;
        if (e.repeat) {
          // If our interval is still running, let it handle stepping
          if (hold.interval) return;
          // Otherwise effect cleanup killed the interval; restart below
        } else {
          // First press: immediate single step
          onSkipTimeRef.current?.(delta);
        }
        // Start (or restart) hold-to-repeat interval
        if (hold.interval) clearInterval(hold.interval);
        hold.key = e.code;
        hold.interval = setInterval(() => {
          onSkipTimeRef.current?.(delta);
        }, 1000 / 15); // 15 steps per second while held
        break;
      }

      // J/K/L shuttle
      case 'KeyJ':
        e.preventDefault();
        onShuttleSpeedRef.current?.('reverse');
        break;
      case 'KeyK':
        e.preventDefault();
        onShuttleSpeedRef.current?.('stop');
        break;
      case 'KeyL':
        e.preventDefault();
        // Shift+L = toggle loop playback; plain L = J/K/L shuttle forward.
        if (e.shiftKey) {
          const st = useTimelineStore.getState();
          if (typeof st.toggleLoop === 'function') st.toggleLoop();
        } else {
          onShuttleSpeedRef.current?.('forward');
        }
        break;

      // Home/End — wall-clock when a clipRange is supplied, else
      // timeline-relative for the multi-track editor's full timeline.
      case 'Home':
        e.preventDefault();
        if (clipRangeRef.current && Number.isFinite(clipRangeRef.current.start)) {
          onSeekRef.current?.(clipRangeRef.current.start);
        } else {
          onSeekRef.current?.(0);
        }
        break;
      case 'End':
        e.preventDefault();
        if (clipRangeRef.current && Number.isFinite(clipRangeRef.current.end)) {
          onSeekRef.current?.(clipRangeRef.current.end);
        } else {
          onSeekRef.current?.(useTimelineStore.getState().duration);
        }
        break;

      // Mute
      case 'KeyM':
        e.preventDefault();
        onToggleMuteRef.current?.();
        break;

      // Tool selection
      case 'KeyV':
        e.preventDefault();
        setActiveTool('select');
        break;
      case 'KeyC':
        e.preventDefault();
        setActiveTool('razor');
        break;
      case 'KeyT':
        e.preventDefault();
        setActiveTool('text');
        break;
      case 'KeyR':
        if (!e.ctrlKey && !e.metaKey) {
          e.preventDefault();
          setActiveTool('shape');
        }
        break;

      // Snap toggle
      case 'KeyN':
        e.preventDefault();
        useTimelineStore.getState().toggleSnap();
        break;

      // Delete selected (supports multi-select) — single undo snapshot
      case 'Delete':
      case 'Backspace':
        if (!e.target.closest('[contenteditable]')) {
          const state = useTimelineStore.getState();
          const idsToDelete = state.selectedItemIds.length > 0
            ? state.selectedItemIds
            : (state.selectedItemId ? [state.selectedItemId] : []);
          if (idsToDelete.length > 0) {
            e.preventDefault();
            if (idsToDelete.length === 1) {
              removeItem(idsToDelete[0]);
            } else {
              removeItems(idsToDelete);
            }
          }
        }
        break;

      // Escape — deselect
      case 'Escape':
        e.preventDefault();
        useTimelineStore.getState().setSelectedItemId(null);
        break;

      // Split at playhead — read playhead and items from store at call time
      case 'KeyS':
        if (!e.ctrlKey && !e.metaKey) {
          e.preventDefault();
          const st = useTimelineStore.getState();
          const itemAtPlayhead = st.items.find(i =>
            st.playhead > i.start + 0.1 && st.playhead < i.end - 0.1 &&
            (i.type === 'video' || i.type === 'audio')
          );
          if (itemAtPlayhead) {
            splitItem(itemAtPlayhead.id, st.playhead);
          }
        }
        break;

      // ── Frame step (NLE convention: `,` = back one frame, `.` = forward) ──
      // Uses the project FPS from the store; falls back to 30. Shift
      // multiplier is *5 frames* so power users can scrub larger jumps
      // without leaving the keyboard.
      case 'Comma':
      case 'Period': {
        e.preventDefault();
        const st = useTimelineStore.getState();
        const fps = (st.project && st.project.fps) || 30;
        const dir = e.code === 'Comma' ? -1 : 1;
        const steps = e.shiftKey ? 5 : 1;
        onSkipTimeRef.current?.((dir * steps) / fps);
        break;
      }

      // ── Loop in / out points ──
      // I = set loop in, O = set loop out, Shift+L = toggle loop on/off,
      // Alt+I/O = clear that point. The store handles the actual loop
      // playback enforcement; the keys here just shape the in/out range.
      case 'KeyI':
      case 'KeyO': {
        e.preventDefault();
        const st = useTimelineStore.getState();
        if (typeof st.setLoopRange !== 'function') break;
        const t = st.playhead || 0;
        if (e.altKey) {
          // Alt+I or Alt+O clears that endpoint.
          if (e.code === 'KeyI') st.setLoopRange({ start: null });
          else st.setLoopRange({ end: null });
          break;
        }
        if (e.code === 'KeyI') st.setLoopRange({ start: t });
        else st.setLoopRange({ end: t });
        break;
      }
    }
  }, [enabled, setActiveTool, removeItem, removeItems, splitItem]);

  const handleKeyUp = useCallback((e) => {
    if (e.code === 'ArrowLeft' || e.code === 'ArrowRight') {
      const hold = arrowHoldRef.current;
      if (hold.key === e.code && hold.interval) {
        clearInterval(hold.interval);
        hold.interval = null;
        hold.key = null;
      }
    }
  }, []);

  useEffect(() => {
    if (!enabled) return;
    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('keyup', handleKeyUp);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('keyup', handleKeyUp);
      const hold = arrowHoldRef.current;
      if (hold.interval) {
        clearInterval(hold.interval);
        hold.interval = null;
        hold.key = null;
      }
    };
  }, [enabled, handleKeyDown, handleKeyUp]);
}
