import { useEffect, useRef, useState, useCallback } from 'react';

/**
 * Mint and cache short-lived signed URLs for ``/api/files/<job>/<path>``.
 *
 * Why this exists
 * ---------------
 * The AuthMiddleware fingerprint check (and intermittent reverse-proxy
 * cookie weirdness) used to kill playback of large MKV / VP9 / 4K
 * previews because a single mismatched header would blow away the
 * session cookie mid-range-request. Signed URLs uncouple media fetch
 * auth from the session cookie: the URL itself carries a short-lived
 * HMAC signature, minted on demand via ``POST /api/media/sign``.
 *
 * The cache lives in module scope, shared across components. If two
 * components ask for the same (jobId, path) simultaneously they share
 * one in-flight fetch. The entry is refreshed ~60s before ``expiry``
 * so a slow preview never 401s mid-playback because of an expired URL.
 */

const _urlCache = new Map(); // key -> { url, expiry }
const _inFlight = new Map(); // key -> Promise<{url, expiry}>
const _EXPIRY_GRACE_SECONDS = 60;

function _keyFor(jobId, path) {
  return `${jobId}/${path}`;
}

function _nowSeconds() {
  return Math.floor(Date.now() / 1000);
}

function _isFresh(entry) {
  return entry && typeof entry.expiry === 'number'
    && entry.expiry - _EXPIRY_GRACE_SECONDS > _nowSeconds();
}

/**
 * Return a signed ``/api/files/...`` URL, minting one if necessary.
 *
 * Uses a module-scope cache keyed on ``{jobId}/{path}`` so two
 * concurrent callers share one network round-trip and no redundant
 * signatures hit the backend.
 *
 * @param {string} jobId
 * @param {string} path  path relative to the job root (no leading slash)
 * @param {object} [opts]
 * @param {AbortSignal} [opts.signal]
 * @returns {Promise<string>} fully-formed URL including ``?exp=&sig=&u=``
 */
export async function getSignedMediaUrl(jobId, path, opts = {}) {
  if (!jobId || !path) {
    throw new Error('getSignedMediaUrl: jobId and path are required');
  }
  const normPath = String(path).replace(/^\/+/, '');
  const key = _keyFor(jobId, normPath);
  const cached = _urlCache.get(key);
  if (_isFresh(cached)) return cached.url;

  let inflight = _inFlight.get(key);
  if (!inflight) {
    inflight = (async () => {
      const res = await fetch('/api/media/sign', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({ job_id: jobId, path: normPath }),
        signal: opts.signal,
      });
      if (!res.ok) {
        throw new Error(`sign failed: HTTP ${res.status}`);
      }
      const body = await res.json();
      const entry = { url: String(body.url || ''), expiry: Number(body.expires_at || 0) };
      if (!entry.url) throw new Error('sign returned empty url');
      _urlCache.set(key, entry);
      return entry;
    })().finally(() => { _inFlight.delete(key); });
    _inFlight.set(key, inflight);
  }
  const entry = await inflight;
  return entry.url;
}

/**
 * Drop a cached entry. Components should call this when a fetch to
 * the signed URL returns 401 so the next attempt re-signs.
 */
export function invalidateSignedMediaUrl(jobId, path) {
  const key = _keyFor(jobId, String(path).replace(/^\/+/, ''));
  _urlCache.delete(key);
}

export function _resetCacheForTests() {
  _urlCache.clear();
  _inFlight.clear();
}

/**
 * React hook wrapping :func:`getSignedMediaUrl` with re-signing on
 * expiry and abort-on-unmount. Returns ``{ url, loading, error,
 * refetch }``.
 *
 *   * ``url`` is ``null`` until the first mint succeeds.
 *   * ``loading`` is true while a sign call is in flight.
 *   * ``error`` carries the last fetch error (or null).
 *   * ``refetch()`` forces a fresh mint, useful when a 401 leaks
 *     through from a stale URL.
 */
export function useSignedMediaUrl(jobId, path) {
  const [url, setUrl] = useState(null);
  const [loading, setLoading] = useState(Boolean(jobId && path));
  const [error, setError] = useState(null);
  const abortRef = useRef(null);
  const timerRef = useRef(null);

  const load = useCallback(async (force = false) => {
    if (!jobId || !path) {
      setUrl(null); setLoading(false); setError(null);
      return;
    }
    if (force) invalidateSignedMediaUrl(jobId, path);
    if (abortRef.current) abortRef.current.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    setLoading(true);
    setError(null);
    try {
      const u = await getSignedMediaUrl(jobId, path, { signal: ctrl.signal });
      if (ctrl.signal.aborted) return;
      setUrl(u);
      // Schedule auto-refresh shortly before expiry so playback
      // never racing the signature cut-off. Parse ``exp`` from the
      // URL — avoids another round-trip / state plumbing.
      try {
        const m = /[?&]exp=(\d+)/.exec(u);
        const expiry = m ? parseInt(m[1], 10) : 0;
        if (expiry > 0) {
          const msUntilRefresh = Math.max(
            5000,
            (expiry - _EXPIRY_GRACE_SECONDS - _nowSeconds()) * 1000,
          );
          if (timerRef.current) clearTimeout(timerRef.current);
          timerRef.current = setTimeout(() => { load(true); }, msUntilRefresh);
        }
      } catch {
        /* exp parse failed — no auto-refresh scheduled */
      }
    } catch (e) {
      if (ctrl.signal.aborted || e?.name === 'AbortError') return;
      setError(e);
    } finally {
      if (!ctrl.signal.aborted) setLoading(false);
    }
  }, [jobId, path]);

  useEffect(() => {
    load(false);
    return () => {
      if (abortRef.current) abortRef.current.abort();
      if (timerRef.current) { clearTimeout(timerRef.current); timerRef.current = null; }
    };
  }, [load]);

  const refetch = useCallback(() => load(true), [load]);
  return { url, loading, error, refetch };
}
