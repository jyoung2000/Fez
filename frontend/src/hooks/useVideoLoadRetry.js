import { useEffect, useRef, useState } from 'react';

/**
 * useVideoLoadRetry — keep an <video> element trying to load its ``src``
 * while the backend browser-preview transcode is still in flight.
 *
 * Why this exists
 * ---------------
 * ``/api/files/.../video.*`` and the share route return ``HTTP 503`` +
 * ``Retry-After: 5`` while ``backend/services/browser_preview`` is
 * still transcoding the source into a browser-playable derivative.
 * That transcode can take several minutes on large (4K / VP9 / MKV
 * with AC3) uploads. During the window, every ``<video>.load()`` fires
 * a fresh ``error`` event and naive one-shot retry loops give up
 * almost immediately and leave the element looking broken.
 *
 * This hook:
 *   * swallows ``error`` events and schedules a ``video.load()`` retry
 *     with exponential backoff (0.5s → 1s → 2s → 4s → 8s → 10s steady),
 *   * caps at ``maxAttempts`` (default 60 ≈ 10 minutes) before
 *     surfacing a hard failure via ``error=true`` in the return,
 *   * exposes ``preparing=true`` after the first couple of retries so
 *     the component can render a "Preparing preview…" overlay instead
 *     of a dead element,
 *   * resets all bookkeeping whenever ``src`` changes OR a successful
 *     ``canplay`` fires (so a later transient hiccup gets a fresh
 *     retry budget).
 *
 * Returns ``{ ready, preparing, error, reset }``:
 *   * ``ready``      — canplay has fired at least once for the current src.
 *   * ``preparing``  — we're in the retry loop and the user should see
 *                      "Preparing preview…".
 *   * ``error``      — retry budget exhausted; show a manual-Retry UI.
 *   * ``reset()``    — zero out retry bookkeeping and re-load immediately
 *                      (wire this to the manual-Retry button).
 *
 * The hook mirrors the logic baked into ``VideoEditor.jsx`` (commit
 * 321ff2b) so ``ClipPreview`` and ``VideoPlayer`` behave the same way
 * when the source is still transcoding.
 */
export default function useVideoLoadRetry(videoRef, src, options = {}) {
  const {
    maxAttempts = 60,
    // Delay schedule in ms. After the table is exhausted the last
    // value is used for every subsequent retry. Total ≈ 10 min with
    // the default table + default maxAttempts.
    delays = [500, 1000, 2000, 4000, 8000],
    steadyDelay = 10000,
    // After this many attempts, flip ``preparing`` to true so the
    // component can show the spinner. Kept at 2 so a real transient
    // network hiccup (single error burst) doesn't flash the overlay.
    preparingAfterAttempt = 2,
  } = options;

  const [ready, setReady] = useState(false);
  const [preparing, setPreparing] = useState(false);
  const [error, setError] = useState(false);

  const attemptRef = useRef(0);
  const timerRef = useRef(null);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return undefined;

    // Fresh src deserves a fresh budget.
    attemptRef.current = 0;
    if (timerRef.current) {
      clearTimeout(timerRef.current);
      timerRef.current = null;
    }
    setReady(false);
    setPreparing(false);
    setError(false);

    const onCanPlay = () => {
      setReady(true);
      setPreparing(false);
      setError(false);
      // A successful canplay means the source is good now — zero out
      // the retry counter so a later hiccup gets the full budget again.
      attemptRef.current = 0;
    };

    const onError = () => {
      const attempt = attemptRef.current;
      if (attempt >= maxAttempts) {
        setError(true);
        setPreparing(false);
        return;
      }
      if (attempt >= preparingAfterAttempt) setPreparing(true);
      const delay = attempt < delays.length ? delays[attempt] : steadyDelay;
      attemptRef.current = attempt + 1;
      if (timerRef.current) clearTimeout(timerRef.current);
      timerRef.current = setTimeout(() => {
        timerRef.current = null;
        // The video element might have unmounted (strict-mode double
        // effect or parent re-render). Guard every access.
        if (videoRef.current) {
          try {
            videoRef.current.load();
          } catch {
            /* element gone — ignore */
          }
        }
      }, delay);
    };

    video.addEventListener('canplay', onCanPlay);
    video.addEventListener('error', onError);

    // If we attach after the element is already past canplay (e.g.
    // because the browser cached the response), synthesise the
    // ready-state transition so downstream consumers don't wait
    // forever for an event that already fired.
    if (video.readyState >= 3) {
      setReady(true);
    }

    return () => {
      video.removeEventListener('canplay', onCanPlay);
      video.removeEventListener('error', onError);
      if (timerRef.current) {
        clearTimeout(timerRef.current);
        timerRef.current = null;
      }
    };
    // videoRef is a ref object — stable identity; including it would
    // be a lint-noise-only change. src drives the real dependency.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [src, maxAttempts, steadyDelay, preparingAfterAttempt]);

  const reset = () => {
    attemptRef.current = 0;
    if (timerRef.current) {
      clearTimeout(timerRef.current);
      timerRef.current = null;
    }
    setError(false);
    setPreparing(false);
    if (videoRef.current) {
      try {
        videoRef.current.load();
      } catch {
        /* element gone — ignore */
      }
    }
  };

  return { ready, preparing, error, reset };
}
