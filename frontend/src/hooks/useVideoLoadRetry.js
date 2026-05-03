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
 * with AC3) uploads. During the window, a naive one-shot retry loop
 * gives up almost immediately and leaves the element looking broken —
 * the root cause is that Chromium frequently surfaces a 503 on a
 * range request as a ``stalled`` / ``suspend`` event instead of
 * ``error``, so code that only listens to ``error`` waits forever
 * for an event that never fires.
 *
 * This hook:
 *   * HEAD-probes ``src`` on mount and every reset. A 503 puts the
 *     hook into ``preparing`` immediately (no need to wait on the
 *     ``<video>`` element firing anything). A 401 flips a dedicated
 *     ``authError`` flag and does NOT retry — retrying a broken
 *     session forever is worse than failing loudly.
 *   * Listens to ``error``, ``stalled``, and ``suspend``, debounced
 *     against ``progress`` so normal buffering doesn't count.
 *   * Uses ``Retry-After`` from the HEAD response for the first
 *     retry delay when present.
 *   * Runs a tight exponential backoff (0.2s → 0.4s → 0.8s → 1.5s →
 *     3s → 5s steady) up to ``maxAttempts`` (default 60 ≈ 5 minutes).
 *     The backend's faster (HW / ultrafast) preview encoder makes
 *     most encodes finish inside the first few retries, so the
 *     player picks the file up almost as soon as it's written
 *     instead of sleeping for 5-10 s after the encoder finishes.
 *
 * Returns ``{ ready, preparing, error, authError, reset }``:
 *   * ``ready``      — canplay has fired at least once for the current src.
 *   * ``preparing``  — we're in the retry loop ("Preparing preview…").
 *   * ``error``      — retry budget exhausted; show a manual-Retry UI.
 *   * ``authError``  — fetch probe returned 401; parent should
 *                      prompt the user to re-login.
 *   * ``reset()``    — zero out retry bookkeeping and re-probe.
 */
export default function useVideoLoadRetry(videoRef, src, options = {}) {
  const {
    maxAttempts = 60,
    // Tight initial backoff so we re-probe quickly while the
    // preview encoder is finishing — the HW / ultrafast pipeline
    // typically lands the final preview within the first handful
    // of attempts, and a 500 ms first delay would waste half the
    // time-to-play budget on a sleep.
    delays = [200, 400, 800, 1500, 3000],
    steadyDelay = 5000,
    // After this many attempts, flip ``preparing`` to true so the
    // component can show the spinner. Kept at 2 so a real transient
    // network hiccup (single error burst) doesn't flash the overlay.
    preparingAfterAttempt = 2,
    // A stall event only counts as a retry trigger if no progress
    // event fires within this window (ms). Chromium emits stall /
    // suspend during normal buffering too; debouncing against
    // progress avoids counting those.
    stallDebounceMs = 3000,
  } = options;

  const [ready, setReady] = useState(false);
  const [preparing, setPreparing] = useState(false);
  const [error, setError] = useState(false);
  const [authError, setAuthError] = useState(false);

  const attemptRef = useRef(0);
  const timerRef = useRef(null);
  const stallTimerRef = useRef(null);
  const firstDelayOverrideRef = useRef(null);
  const abortRef = useRef(null);
  // Flag set by the periodic HEAD probe when the backend keeps
  // returning 503. If the transcode is genuinely still in flight we
  // refuse to give up — the 5-minute attempt budget is calibrated for
  // a typical encode, not a 4K libx264 ultrafast on a tiny CPU box,
  // which can legitimately run longer. See ``_scheduleRetry``.
  const backendStillWorkingRef = useRef(false);
  // Cap the number of budget resets so a backend that crashes mid-
  // transcode (last 503 was minutes ago, now nothing responds) still
  // surfaces an error eventually instead of spinning forever. Three
  // full budgets ≈ 15 minutes — long enough for any sane encode, short
  // enough that a hung backend doesn't strand the user.
  const budgetResetsRef = useRef(0);
  const MAX_BUDGET_RESETS = 3;

  useEffect(() => {
    const video = videoRef.current;
    if (!video || !src) return undefined;

    // Fresh src deserves a fresh budget.
    attemptRef.current = 0;
    if (timerRef.current) { clearTimeout(timerRef.current); timerRef.current = null; }
    if (stallTimerRef.current) { clearTimeout(stallTimerRef.current); stallTimerRef.current = null; }
    firstDelayOverrideRef.current = null;
    setReady(false);
    setPreparing(false);
    setError(false);
    setAuthError(false);

    // ── HEAD probe ──────────────────────────────────────────────
    //
    // Chromium doesn't always surface a 503 on a Range request as
    // an HTMLMediaElement ``error`` event — it often reports it as
    // a ``stalled`` / ``suspend`` instead. Issuing our own HEAD
    // tells us the server's state directly so we can start the
    // retry loop without waiting on an event that may never fire.
    if (abortRef.current) abortRef.current.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    backendStillWorkingRef.current = false;
    budgetResetsRef.current = 0;
    // ``ready2xx`` flag — true when HEAD said the file is finally
    // playable. Used by the budget-exhausted re-probe to ``video.load()``
    // and pick the new content up immediately, instead of leaving the
    // already-errored element idle.
    const probe = async ({ kickLoadOn2xx = false } = {}) => {
      try {
        const resp = await fetch(src, { method: 'HEAD', signal: ctrl.signal });
        if (ctrl.signal.aborted) return;
        if (resp.status === 503) {
          // Transcode still running — enter the preparing state
          // immediately, use Retry-After for the first delay if
          // present.
          setPreparing(true);
          backendStillWorkingRef.current = true;
          const retryAfter = parseInt(resp.headers.get('Retry-After') || '', 10);
          if (Number.isFinite(retryAfter) && retryAfter > 0) {
            firstDelayOverrideRef.current = retryAfter * 1000;
          }
          _scheduleRetry();
        } else if (resp.status === 401) {
          setAuthError(true);
          setPreparing(false);
        } else if (resp.status >= 200 && resp.status < 400) {
          backendStillWorkingRef.current = false;
          if (kickLoadOn2xx && videoRef.current) {
            try { videoRef.current.load(); } catch { /* element gone */ }
          }
        } else {
          // 4xx/5xx other than 401/503 — fall into retry; the next
          // HEAD will tell us whether it's transient or permanent.
          backendStillWorkingRef.current = false;
          _scheduleRetry();
        }
      } catch (e) {
        if (ctrl.signal.aborted) return;
        // Network error on the probe — fall into the retry loop.
        _scheduleRetry();
      }
    };
    probe();

    function _scheduleRetry() {
      const attempt = attemptRef.current;
      if (attempt >= maxAttempts) {
        // Don't surface a hard failure while the backend is still
        // actively transcoding — that's the "preview never loads"
        // bug we exist to prevent. Re-probe the server: if it still
        // says 503, reset the budget and keep going. Only flip to
        // ``error`` when the backend has stopped reporting progress.
        if (
          backendStillWorkingRef.current
          && budgetResetsRef.current < MAX_BUDGET_RESETS
        ) {
          attemptRef.current = 0;
          backendStillWorkingRef.current = false;
          budgetResetsRef.current += 1;
          probe({ kickLoadOn2xx: true });
          return;
        }
        setError(true);
        setPreparing(false);
        return;
      }
      if (attempt >= preparingAfterAttempt) setPreparing(true);
      let delay;
      if (firstDelayOverrideRef.current != null && attempt === 0) {
        delay = firstDelayOverrideRef.current;
        firstDelayOverrideRef.current = null;
      } else {
        delay = attempt < delays.length ? delays[attempt] : steadyDelay;
      }
      attemptRef.current = attempt + 1;
      if (timerRef.current) clearTimeout(timerRef.current);
      timerRef.current = setTimeout(() => {
        timerRef.current = null;
        if (videoRef.current) {
          try { videoRef.current.load(); } catch { /* element gone */ }
        }
      }, delay);
    }

    const onCanPlay = () => {
      setReady(true);
      setPreparing(false);
      setError(false);
      setAuthError(false);
      attemptRef.current = 0;
    };

    const onError = () => {
      // Element ``error`` — always a hard signal to retry.
      _scheduleRetry();
    };

    const onStallOrSuspend = () => {
      // Debounce: a subsequent ``progress`` within ``stallDebounceMs``
      // means the video is just buffering. Only if we're still stuck
      // after the window do we kick off a retry.
      if (stallTimerRef.current) clearTimeout(stallTimerRef.current);
      stallTimerRef.current = setTimeout(() => {
        stallTimerRef.current = null;
        // Only fire if the element is still not playing anything.
        const v = videoRef.current;
        if (!v) return;
        if (v.readyState >= 3) return;
        _scheduleRetry();
      }, stallDebounceMs);
    };

    const onProgress = () => {
      if (stallTimerRef.current) {
        clearTimeout(stallTimerRef.current);
        stallTimerRef.current = null;
      }
    };

    video.addEventListener('canplay', onCanPlay);
    video.addEventListener('error', onError);
    video.addEventListener('stalled', onStallOrSuspend);
    video.addEventListener('suspend', onStallOrSuspend);
    video.addEventListener('progress', onProgress);

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
      video.removeEventListener('stalled', onStallOrSuspend);
      video.removeEventListener('suspend', onStallOrSuspend);
      video.removeEventListener('progress', onProgress);
      if (timerRef.current) { clearTimeout(timerRef.current); timerRef.current = null; }
      if (stallTimerRef.current) { clearTimeout(stallTimerRef.current); stallTimerRef.current = null; }
      if (abortRef.current) { abortRef.current.abort(); abortRef.current = null; }
    };
    // videoRef is a ref object — stable identity; src drives the
    // real dependency.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [src, maxAttempts, steadyDelay, preparingAfterAttempt, stallDebounceMs]);

  const reset = () => {
    attemptRef.current = 0;
    budgetResetsRef.current = 0;
    backendStillWorkingRef.current = false;
    if (timerRef.current) { clearTimeout(timerRef.current); timerRef.current = null; }
    if (stallTimerRef.current) { clearTimeout(stallTimerRef.current); stallTimerRef.current = null; }
    firstDelayOverrideRef.current = null;
    setError(false);
    setPreparing(false);
    setAuthError(false);
    // Re-probe via the effect: bumping src dep isn't possible from
    // here, so we re-run the probe inline.
    if (videoRef.current && src) {
      if (abortRef.current) abortRef.current.abort();
      const ctrl = new AbortController();
      abortRef.current = ctrl;
      (async () => {
        try {
          const resp = await fetch(src, { method: 'HEAD', signal: ctrl.signal });
          if (ctrl.signal.aborted) return;
          if (resp.status === 503) {
            setPreparing(true);
            const retryAfter = parseInt(resp.headers.get('Retry-After') || '', 10);
            if (Number.isFinite(retryAfter) && retryAfter > 0) {
              firstDelayOverrideRef.current = retryAfter * 1000;
            }
          } else if (resp.status === 401) {
            setAuthError(true);
            return;
          }
        } catch {
          /* network error — proceed to load() */
        }
        try {
          if (videoRef.current) videoRef.current.load();
        } catch { /* element gone */ }
      })();
    }
  };

  return { ready, preparing, error, authError, reset };
}
