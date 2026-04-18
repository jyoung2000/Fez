/**
 * Undo coalescing helpers.
 *
 * Continuous-input controls (range sliders, color pickers, text inputs)
 * fire `onChange` on every tick. When each tick goes straight into the
 * timeline store via `updateItem`, every micro-change pushes a new
 * snapshot onto the zundo undo stack. The result: pressing `Ctrl+Z`
 * only reverts a single slider tick, so the user has to mash undo
 * dozens of times to get back to where they started — and it *feels*
 * like undo doesn't work for property changes at all.
 *
 * The fix mirrors what `Timeline` and `InteractiveOverlay` already do
 * for canvas/timeline drags: pause the temporal middleware while a
 * gesture is in flight and resume on release. zundo coalesces all the
 * intermediate state into a single before/after snapshot, so one
 * `Ctrl+Z` rewinds the entire interaction.
 */

import useTimelineStore from '../stores/timelineStore';

/**
 * Pointer-event handlers that pause the undo stack on press and
 * resume on release. Spread the result onto any `<input>`, `<button>`,
 * or other interactive element that fires continuous changes.
 *
 *     <input type="range" {...undoCoalesceHandlers()} ... />
 *
 * Pointer events are a superset of mouse/touch/pen, so a single set
 * of listeners covers every input modality. `pointerup` and
 * `pointercancel` both flush — the latter handles touch interruption
 * (scrolling, OS gestures) so the temporal middleware never gets
 * left in the paused state.
 */
export function undoCoalesceHandlers() {
  return {
    onPointerDown: () => {
      try {
        useTimelineStore.temporal.getState().pause();
      } catch {
        // Temporal middleware not attached (e.g. tests) — no-op.
      }
    },
    onPointerUp: () => {
      try {
        useTimelineStore.temporal.getState().resume();
      } catch {
        /* no-op */
      }
    },
    onPointerCancel: () => {
      try {
        useTimelineStore.temporal.getState().resume();
      } catch {
        /* no-op */
      }
    },
    // Keyboard sliders (arrow keys with focus) don't fire pointer
    // events. The browser still fires a ``change`` event when the
    // user releases focus or after each keystroke; we don't need to
    // pause for those because they're already discrete steps.
  };
}

/**
 * Manual pause/resume — handy for places that have their own
 * gesture lifecycle (e.g. a custom drag handler).
 */
export function pauseUndo() {
  try {
    useTimelineStore.temporal.getState().pause();
  } catch {
    /* no-op */
  }
}

export function resumeUndo() {
  try {
    useTimelineStore.temporal.getState().resume();
  } catch {
    /* no-op */
  }
}
