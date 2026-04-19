import React, { createContext, useCallback, useContext, useEffect, useRef, useState } from 'react';

// Shape of the auth context we expose:
//   user:       { id, username, role, head_admin, created_at } | null
//   status:     'loading' | 'authenticated' | 'unauthenticated'
//   login(username, password)          -> throws on error
//   logout()                           -> always resolves
//   refresh()                          -> re-fetches /api/auth/me
//   changePassword(current, next)      -> throws on error
//   isAdmin                            -> shortcut
//
// Every API call goes through the built-in `fetch` with credentials:'include'
// so the HttpOnly session cookie rides along. A 401 from any call triggers
// the AuthGate to redirect to /login.

const AuthContext = createContext(null);

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) {
    throw new Error('useAuth() called outside <AuthProvider>');
  }
  return ctx;
}

async function apiFetch(url, { method = 'GET', body, headers } = {}) {
  const init = {
    method,
    credentials: 'include',
    headers: { 'Accept': 'application/json', ...(headers || {}) },
  };
  if (body !== undefined) {
    init.headers['Content-Type'] = 'application/json';
    init.body = typeof body === 'string' ? body : JSON.stringify(body);
  }
  const resp = await fetch(url, init);
  if (resp.status === 204) return null;
  let data = null;
  try {
    data = await resp.json();
  } catch {
    data = null;
  }
  if (!resp.ok) {
    const err = new Error((data && data.detail) || resp.statusText || 'request failed');
    err.status = resp.status;
    err.data = data;
    throw err;
  }
  return data;
}

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [status, setStatus] = useState('loading'); // 'loading' | 'authenticated' | 'unauthenticated'
  const inflightRef = useRef(null);

  const refresh = useCallback(async () => {
    if (inflightRef.current) return inflightRef.current;
    const p = (async () => {
      try {
        const me = await apiFetch('/api/auth/me');
        setUser(me?.user || null);
        setStatus(me?.user ? 'authenticated' : 'unauthenticated');
      } catch (e) {
        if (e.status === 401) {
          setUser(null);
          setStatus('unauthenticated');
        } else {
          // Unexpected error: keep "loading" so we retry later rather
          // than showing the user a false "logged out" state.
          console.error('auth refresh error', e);
          setUser(null);
          setStatus('unauthenticated');
        }
      } finally {
        inflightRef.current = null;
      }
    })();
    inflightRef.current = p;
    return p;
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  // Listen for 401 responses from anywhere in the app and bounce to
  // login. We patch the global fetch so any component (legacy pages,
  // dropzones, etc.) benefits automatically.
  //
  // Defensive design: a single 401 does NOT immediately yank the
  // user to /login. Some endpoints (e.g. anything Settings.jsx polls
  // on mount) can 401 for reasons unrelated to the session being
  // dead — a misconfigured route, a per-resource permission, a race.
  // If we redirect on every 401, those can trigger a loop with the
  // Login page (which navigates back as soon as it sees ``user``
  // populated from a still-valid cookie). So when a 401 fires, we
  // re-verify against ``/api/auth/me`` and only force the redirect
  // when the session truly is gone.
  useEffect(() => {
    const originalFetch = window.fetch.bind(window);
    let verifyPromise = null;
    let lastRedirectAt = 0;
    window.fetch = async (input, init) => {
      const merged = init ? { credentials: 'include', ...init } : { credentials: 'include' };
      const resp = await originalFetch(input, merged);
      try {
        const url = typeof input === 'string' ? input : input?.url || '';
        const isOurApi = url.includes('/api/');
        // Skip auth endpoints themselves: ``/api/auth/login`` 401s on
        // bad credentials (handled by the form), and ``/api/auth/me``
        // is the verify probe — recursing would loop forever.
        const isAuthEndpoint = url.includes('/api/auth/login')
          || url.includes('/api/auth/me');
        if (resp.status === 401 && isOurApi && !isAuthEndpoint) {
          // Coalesce concurrent 401s onto a single verify probe so a
          // page that fans out 5+ requests doesn't fire 5+ /api/auth/me
          // calls (and 5+ navigations) when the session really did
          // expire.
          if (!verifyPromise) {
            verifyPromise = (async () => {
              try {
                const probe = await originalFetch('/api/auth/me', {
                  credentials: 'include',
                });
                return probe.status === 401;
              } catch {
                // Network error on the probe — don't bounce; the
                // original failure will surface to the caller.
                return false;
              } finally {
                // Clear after a beat so subsequent 401s re-probe
                // (the user might have just signed out in another tab).
                setTimeout(() => { verifyPromise = null; }, 500);
              }
            })();
          }
          const sessionDead = await verifyPromise;
          if (sessionDead) {
            setUser(null);
            setStatus('unauthenticated');
            // Cooldown: even after we decide the session is gone,
            // don't fire ``window.location.href`` more than once
            // per second. Multiple awaiters of the same probe
            // could otherwise stack navigations.
            const now = Date.now();
            if (
              now - lastRedirectAt > 1000
              && !window.location.pathname.startsWith('/login')
            ) {
              lastRedirectAt = now;
              const here = window.location.pathname + window.location.search;
              window.location.href = `/login?next=${encodeURIComponent(here)}`;
            }
          } else if (typeof console !== 'undefined') {
            // Session is still valid but a specific endpoint 401'd —
            // log so a developer can track down the offending route
            // instead of silently dropping the user to /login.
            console.warn(
              `[auth] 401 from ${url} but /api/auth/me still authenticated; not redirecting.`,
            );
          }
        }
      } catch {}
      return resp;
    };
    return () => {
      window.fetch = originalFetch;
    };
  }, []);

  /**
   * Sign in with username + password.
   * ``remember`` controls cookie persistence:
   *   - true  → 30-day rolling cookie, stay signed in across browser restarts
   *   - false → session cookie, browser drops on quit, re-prompt next launch
   * Server enforces the same flag so it can't be flipped client-side after
   * the cookie is issued.
   */
  const login = useCallback(async (username, password, remember = true) => {
    const data = await apiFetch('/api/auth/login', {
      method: 'POST',
      body: { username, password, remember: !!remember },
    });
    setUser(data?.user || null);
    setStatus(data?.user ? 'authenticated' : 'unauthenticated');
    return data?.user;
  }, []);

  const logout = useCallback(async () => {
    // Server invalidates the session token AND clears the cookie.
    // After this call the previous auth token can no longer be used
    // to authenticate; the user has to sign in again.
    try {
      await apiFetch('/api/auth/logout', { method: 'POST' });
    } catch {}
    setUser(null);
    setStatus('unauthenticated');
  }, []);

  const changePassword = useCallback(async (current_password, new_password) => {
    await apiFetch('/api/auth/me/password', {
      method: 'PATCH',
      body: { current_password, new_password },
    });
  }, []);

  const value = {
    user,
    status,
    isAdmin: user?.role === 'admin',
    login,
    logout,
    refresh,
    changePassword,
    apiFetch,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function RequireAuth({ children }) {
  const { user, status } = useAuth();
  if (status === 'loading') return null;
  if (status === 'unauthenticated' || !user) {
    const here = window.location.pathname + window.location.search;
    window.location.href = `/login?next=${encodeURIComponent(here)}`;
    return null;
  }
  return children;
}

export function RequireAdmin({ children }) {
  const { user, status } = useAuth();
  if (status === 'loading') return null;
  if (!user) {
    window.location.href = '/login';
    return null;
  }
  if (user.role !== 'admin') {
    return (
      <div style={{ padding: 48, textAlign: 'center' }}>
        <h2 style={{ color: 'var(--danger, #ef4444)', marginBottom: 12 }}>Admin only</h2>
        <p style={{ color: 'var(--text-muted, #888)' }}>
          You don't have permission to view this page.
        </p>
      </div>
    );
  }
  return children;
}
