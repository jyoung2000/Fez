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
  useEffect(() => {
    const originalFetch = window.fetch.bind(window);
    window.fetch = async (input, init) => {
      const merged = init ? { credentials: 'include', ...init } : { credentials: 'include' };
      const resp = await originalFetch(input, merged);
      try {
        const url = typeof input === 'string' ? input : input?.url || '';
        // Only react to 401 from our own API; don't hijack external fetches.
        if (resp.status === 401 && url.includes('/api/') && !url.includes('/api/auth/login')) {
          setUser(null);
          setStatus('unauthenticated');
          // Only redirect if we're not already on the login page.
          if (!window.location.pathname.startsWith('/login')) {
            const here = window.location.pathname + window.location.search;
            window.location.href = `/login?next=${encodeURIComponent(here)}`;
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
