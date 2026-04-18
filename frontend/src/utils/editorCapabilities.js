/**
 * Editor capability self-check.
 *
 * Runs once when the editor mounts and surfaces a single, friendly
 * banner if the visitor's browser is missing something the editor
 * actually needs (Canvas2D, Web Audio, pointer events, etc.). Optional
 * features (WebCodecs for client-side export, OffscreenCanvas for
 * filmstrip thumbs) are reported as INFO so the user knows which
 * fast-paths are available.
 *
 * Lives outside the React tree so it can be called from any panel
 * without dragging in a context.
 */

let _cached = null;

export function checkEditorCapabilities() {
  if (_cached) return _cached;

  const issues = [];     // hard requirements that are missing
  const downgrades = []; // optional fast-paths that are missing
  const ok = [];         // what's available — useful for the diag panel

  // ── Canvas 2D ─────────────────────────────────────────────
  try {
    const probe = document.createElement('canvas');
    const ctx = probe.getContext && probe.getContext('2d');
    if (!ctx) {
      issues.push('Canvas 2D rendering is not available — the timeline + preview cannot draw.');
    } else {
      ok.push('Canvas 2D');
      // Subfeature probe: ``roundRect`` (Safari < 16.4 / very old browsers).
      if (typeof ctx.roundRect !== 'function') {
        downgrades.push(
          'Canvas2D.roundRect is not available — clip cards on the timeline render with sharp corners.',
        );
      }
    }
  } catch (e) {
    issues.push(`Canvas probe threw: ${e?.message || e}`);
  }

  // ── Pointer events ─────────────────────────────────────────
  if (typeof window !== 'undefined' && !('PointerEvent' in window)) {
    issues.push(
      'PointerEvent is not supported — drag/resize/rotate gestures will not work on this browser.',
    );
  } else {
    ok.push('PointerEvent');
  }

  // ── Web Audio (volume gain + waveform analysis) ────────────
  const AudioCtx = (typeof window !== 'undefined') &&
    (window.AudioContext || window.webkitAudioContext);
  if (!AudioCtx) {
    downgrades.push(
      'Web Audio API not available — volume boost above 100 % and waveform display are disabled.',
    );
  } else {
    ok.push('Web Audio');
  }

  // ── ResizeObserver (used by overlays to reflow) ────────────
  if (typeof window !== 'undefined' && typeof window.ResizeObserver !== 'function') {
    downgrades.push('ResizeObserver missing — overlays may not reflow when the player resizes.');
  } else {
    ok.push('ResizeObserver');
  }

  // ── OffscreenCanvas (filmstrip thumbnails) ─────────────────
  if (typeof OffscreenCanvas !== 'function') {
    downgrades.push(
      'OffscreenCanvas missing — timeline filmstrip thumbnails will be skipped at high zoom.',
    );
  } else {
    ok.push('OffscreenCanvas');
  }

  // ── createImageBitmap (filmstrip + scrubbing thumbs) ───────
  if (typeof window !== 'undefined' && typeof window.createImageBitmap !== 'function') {
    downgrades.push('createImageBitmap missing — thumbnail decode falls back to slower paths.');
  } else {
    ok.push('createImageBitmap');
  }

  // ── WebCodecs (client-side export) ─────────────────────────
  if (typeof window === 'undefined' || typeof window.VideoEncoder !== 'function') {
    downgrades.push(
      'WebCodecs (VideoEncoder) is not available — Browser Export is disabled. Server export still works.',
    );
  } else {
    ok.push('WebCodecs');
  }

  // ── HTMLMediaElement.captureStream (browser export tap) ────
  // Only relevant when WebCodecs is also present.
  try {
    if (typeof HTMLMediaElement !== 'undefined' && !HTMLMediaElement.prototype.captureStream) {
      downgrades.push('HTMLMediaElement.captureStream missing — browser export may fall back.');
    }
  } catch { /* noop */ }

  // ── document.fonts (subtitle font loading) ─────────────────
  if (typeof document !== 'undefined' && (!document.fonts || typeof document.fonts.load !== 'function')) {
    downgrades.push('document.fonts missing — custom subtitle fonts may render with a fallback face.');
  } else {
    ok.push('FontFaceSet');
  }

  _cached = { issues, downgrades, ok };
  return _cached;
}

/** Reset the capability cache (mainly for tests). */
export function _resetEditorCapabilitiesCache() {
  _cached = null;
}
