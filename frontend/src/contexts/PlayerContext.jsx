import React, { createContext, useCallback, useContext, useMemo, useRef } from 'react';

/**
 * PlayerContext — typed replacement for the legacy
 * ``window.__clipai_seekTo`` / ``__clipai_pausePlayer`` globals.
 *
 * The previous design let every mounted player register the same
 * ``window.__clipai_*`` functions; the last writer always won, which
 * broke flows where the Analysis page mounted both ``VideoPlayer`` and
 * ``VideoEditor`` (e.g. clip preview). This context provides a single
 * registration channel scoped to the React tree:
 *
 *   * The active player calls ``register({ seek, pause, getTime })``
 *     and ``unregister()`` on mount / unmount.
 *   * Anything that needs to drive the player (transcript clicks,
 *     keyboard shortcuts, AI agent actions, share-link landing pages)
 *     pulls the API via ``usePlayer()``. Always reflects the
 *     currently-active player even when several mount/unmount during
 *     navigation.
 *   * For backward compatibility, ``register`` mirrors the API onto
 *     ``window.__clipai_*`` so existing callers keep working until
 *     they migrate.
 */
const PlayerContext = createContext(null);

export function PlayerProvider({ children }) {
  const apiRef = useRef(null);

  const register = useCallback((api) => {
    apiRef.current = api;
    // Backward-compat aliases so legacy code (e.g. the AI agent's
    // ``executeFunction`` switch in agent.py-driven UIs) still works.
    if (typeof window !== 'undefined') {
      if (api?.seek) window.__clipai_seekTo = api.seek;
      if (api?.pause) window.__clipai_pausePlayer = api.pause;
      if (api?.getTime) window.__clipai_getPlayerTime = api.getTime;
    }
  }, []);

  const unregister = useCallback(() => {
    apiRef.current = null;
    if (typeof window !== 'undefined') {
      delete window.__clipai_seekTo;
      delete window.__clipai_pausePlayer;
      delete window.__clipai_getPlayerTime;
    }
  }, []);

  // Stable callable surface — consumers read these instead of poking at
  // the ref directly so future implementations can swap in queueing,
  // logging, or cancellation without breaking call sites.
  const seek = useCallback((t) => apiRef.current?.seek?.(t), []);
  const pause = useCallback(() => apiRef.current?.pause?.(), []);
  const play = useCallback(() => apiRef.current?.play?.(), []);
  const getTime = useCallback(() => apiRef.current?.getTime?.() ?? 0, []);

  const value = useMemo(() => ({
    register, unregister, seek, pause, play, getTime,
  }), [register, unregister, seek, pause, play, getTime]);

  return <PlayerContext.Provider value={value}>{children}</PlayerContext.Provider>;
}

export function usePlayer() {
  const ctx = useContext(PlayerContext);
  if (ctx) return ctx;
  // Pre-migration fallback: if no provider is mounted yet, return a
  // shim that proxies to the legacy window globals so callers don't
  // crash. Once App.jsx wraps everything in PlayerProvider this can
  // be tightened to throw.
  return {
    register: () => {},
    unregister: () => {},
    seek: (t) => (typeof window !== 'undefined' && typeof window.__clipai_seekTo === 'function')
      ? window.__clipai_seekTo(t) : undefined,
    pause: () => (typeof window !== 'undefined' && typeof window.__clipai_pausePlayer === 'function')
      ? window.__clipai_pausePlayer() : undefined,
    play: () => undefined,
    getTime: () => (typeof window !== 'undefined' && typeof window.__clipai_getPlayerTime === 'function')
      ? window.__clipai_getPlayerTime() : 0,
  };
}
