import React, { useEffect, useRef, useState } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { useAuth } from '../auth/AuthContext';

/**
 * Apple-style sign-in screen.
 *
 * Design notes:
 *   * Pure-white system sans (SF Pro on Apple, system-ui on everything
 *     else). Generous letter spacing, oversized headline, no all-caps
 *     labels — just like the macOS / iCloud login surface.
 *   * Big 56-px tall text fields with subtle inset shadow + cyan focus
 *     ring. Fonts inside the inputs are 17 px — the iOS minimum to
 *     prevent mobile-Safari from auto-zooming on focus.
 *   * Background is a soft radial wash so the card floats; works in
 *     both light + dark themes via CSS vars with sensible fallbacks.
 */
// localStorage key the login page uses to remember the user's last
// "Remember me" choice. We never store the password — only the
// preference for the checkbox. Pre-checking what they last picked
// is a UX nicety: if they always tick it, they don't have to think
// about it again.
const REMEMBER_PREF_KEY = 'clipai_login_remember';

export default function Login() {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState('');
  const [bootstrap, setBootstrap] = useState(null);
  const [focused, setFocused] = useState(null);
  const [remember, setRemember] = useState(() => {
    try {
      const v = localStorage.getItem(REMEMBER_PREF_KEY);
      // Default to TRUE so first-time visitors stay signed in (matches
      // the "stay signed in for 30 days" footer copy). Users on a
      // shared device can untick once and the choice sticks.
      return v === null ? true : v === 'true';
    } catch {
      return true;
    }
  });
  const usernameRef = useRef(null);
  const { login, user } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();

  useEffect(() => {
    fetch('/api/auth/bootstrap', { credentials: 'include' })
      .then((r) => r.json())
      .then(setBootstrap)
      .catch(() => setBootstrap({ has_users: true }));
  }, []);

  // Already logged in? Bounce to the ``next`` redirect target (or home).
  //
  // Safety net for redirect loops: if a protected page (e.g. /settings)
  // makes a fetch that 401s for a reason unrelated to the session
  // being dead, the global fetch interceptor force-navigates to
  // /login?next=that-page. This effect would then auto-bounce back,
  // the page would 401 again, and the user would see /login and the
  // target flash back and forth indefinitely. Track recent bounces
  // per ``next`` value in sessionStorage and stop auto-navigating
  // once we've bounced 3+ times within 5 seconds — surface a
  // friendly error instead so the user isn't trapped.
  useEffect(() => {
    if (!user) return;
    const params = new URLSearchParams(location.search);
    const next = params.get('next') || '/';

    let blocked = false;
    try {
      const key = `__clipai_loop_${next}`;
      const now = Date.now();
      let count = 0;
      let firstAt = now;
      const stored = sessionStorage.getItem(key);
      if (stored) {
        try {
          const parsed = JSON.parse(stored);
          if (now - parsed.firstAt < 5000) {
            count = parsed.count;
            firstAt = parsed.firstAt;
          }
        } catch { /* corrupt entry — ignore */ }
      }
      count += 1;
      sessionStorage.setItem(key, JSON.stringify({ count, firstAt }));
      if (count >= 3) {
        blocked = true;
        console.warn(
          `[auth] redirect loop to ${next} detected (${count} bounces in `
          + `${now - firstAt}ms); not auto-navigating. Look for a /api/* `
          + `endpoint that returns 401 while you are signed in.`,
        );
      }
    } catch { /* sessionStorage unavailable — fall through and navigate */ }

    if (blocked) {
      setError(
        `Couldn't open ${next} — it kept redirecting back to the sign-in `
        + `page. Try a different page from the menu, or sign in again.`,
      );
      return;
    }
    navigate(next, { replace: true });
  }, [user, location.search, navigate]);

  // Clear the loop counter for this ``next`` once the user submits a
  // fresh login — they've explicitly chosen to retry, give them a
  // clean slate.
  useEffect(() => {
    if (!submitting) return;
    try {
      const params = new URLSearchParams(location.search);
      const next = params.get('next') || '/';
      sessionStorage.removeItem(`__clipai_loop_${next}`);
    } catch { /* sessionStorage unavailable */ }
  }, [submitting, location.search]);

  async function handleSubmit(e) {
    e.preventDefault();
    setError('');
    setSubmitting(true);
    // Persist the user's last choice so the checkbox starts in the
    // same state next visit. Storing only the boolean — never the
    // password — is safe on shared devices.
    try { localStorage.setItem(REMEMBER_PREF_KEY, String(remember)); }
    catch { /* private mode */ }
    try {
      await login(username.trim(), password, remember);
      const params = new URLSearchParams(location.search);
      const next = params.get('next') || '/';
      navigate(next, { replace: true });
    } catch (err) {
      setError(err?.message || 'Login failed');
      // Re-focus username so the user can correct quickly without
      // grabbing the mouse.
      try { usernameRef.current?.focus(); } catch { /* noop */ }
    } finally {
      setSubmitting(false);
    }
  }

  const fontStack =
    '-apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text", ' +
    '"Segoe UI", Roboto, "Helvetica Neue", Arial, system-ui, sans-serif';

  return (
    <div
      style={{
        minHeight: '100vh',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        // Soft dual-radial wash: cyan glow top-left, accent dot bottom-
        // right, on the page bg. Falls back to a flat fill in light
        // theme via the CSS var.
        background:
          'radial-gradient(circle at 18% 20%, rgba(10,132,255,0.18), transparent 55%), ' +
          'radial-gradient(circle at 82% 80%, rgba(255,159,10,0.10), transparent 55%), ' +
          'var(--bg-base, #0b0b10)',
        color: 'var(--text-primary, #f5f5f7)',
        fontFamily: fontStack,
        padding: 24,
        WebkitFontSmoothing: 'antialiased',
        MozOsxFontSmoothing: 'grayscale',
      }}
    >
      <div
        style={{
          width: '100%',
          maxWidth: 440,
          padding: '40px 36px 32px',
          background: 'var(--bg-elevated, rgba(28,28,40,0.78))',
          backdropFilter: 'blur(40px) saturate(180%)',
          WebkitBackdropFilter: 'blur(40px) saturate(180%)',
          border: '1px solid var(--border, rgba(255,255,255,0.08))',
          borderRadius: 22,
          boxShadow:
            '0 24px 80px rgba(0,0,0,0.45), ' +
            '0 1px 0 rgba(255,255,255,0.04) inset, ' +
            '0 -1px 0 rgba(0,0,0,0.25) inset',
        }}
      >
        {/* App mark — tinted disc with a play glyph. Pure CSS so we
            don't depend on any image asset. */}
        <div
          style={{
            width: 56, height: 56,
            margin: '0 auto 18px',
            borderRadius: 14,
            background:
              'linear-gradient(135deg, var(--accent-cyan, #0A84FF) 0%, #5E5CE6 100%)',
            boxShadow:
              '0 10px 30px rgba(10,132,255,0.35), ' +
              '0 1px 0 rgba(255,255,255,0.18) inset',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}
        >
          <svg width="22" height="22" viewBox="0 0 24 24" fill="#fff" aria-hidden>
            <path d="M8 5v14l11-7z" />
          </svg>
        </div>

        <h1
          style={{
            margin: 0,
            fontSize: 30,
            fontWeight: 600,
            textAlign: 'center',
            letterSpacing: '-0.02em',
            color: 'var(--text-primary, #f5f5f7)',
          }}
        >
          Sign in to ClipAI
        </h1>
        <p
          style={{
            margin: '10px 0 28px',
            fontSize: 15,
            fontWeight: 400,
            textAlign: 'center',
            color: 'var(--text-secondary, rgba(235,235,245,0.65))',
            lineHeight: 1.4,
          }}
        >
          Use your ClipAI username and password.
        </p>

        <form onSubmit={handleSubmit} noValidate>
          <FloatingField
            label="Username"
            id="login-username"
            type="text"
            value={username}
            onChange={setUsername}
            disabled={submitting}
            inputRef={usernameRef}
            autoComplete="username"
            autoFocus
            focused={focused === 'username'}
            onFocus={() => setFocused('username')}
            onBlur={() => setFocused(null)}
          />

          <div style={{ height: 14 }} />

          <FloatingField
            label="Password"
            id="login-password"
            type="password"
            value={password}
            onChange={setPassword}
            disabled={submitting}
            autoComplete="current-password"
            focused={focused === 'password'}
            onFocus={() => setFocused('password')}
            onBlur={() => setFocused(null)}
          />

          {/* Remember me — controls whether the cookie persists past
              the browser tab. Persistent cookies last 30 days; the
              session-scoped variant is dropped when the browser quits.
              Defaults to ON for first-time visitors and remembers the
              user's last choice via localStorage. */}
          <label
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 10,
              marginTop: 16,
              fontSize: 14,
              color: 'var(--text-secondary, rgba(235,235,245,0.78))',
              cursor: submitting ? 'not-allowed' : 'pointer',
              userSelect: 'none',
            }}
          >
            <span
              role="checkbox"
              aria-checked={remember}
              tabIndex={0}
              onClick={() => !submitting && setRemember((v) => !v)}
              onKeyDown={(e) => {
                if (submitting) return;
                if (e.key === ' ' || e.key === 'Enter') {
                  e.preventDefault();
                  setRemember((v) => !v);
                }
              }}
              style={{
                width: 22,
                height: 22,
                borderRadius: 6,
                background: remember
                  ? 'var(--accent-cyan, #0A84FF)'
                  : 'transparent',
                border: `1.5px solid ${remember
                  ? 'var(--accent-cyan, #0A84FF)'
                  : 'var(--border, rgba(255,255,255,0.20))'}`,
                display: 'inline-flex',
                alignItems: 'center',
                justifyContent: 'center',
                transition: 'background 0.15s ease, border-color 0.15s ease',
                flexShrink: 0,
              }}
            >
              {remember && (
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
                  <polyline points="20 6 9 17 4 12" />
                </svg>
              )}
            </span>
            <span
              onClick={() => !submitting && setRemember((v) => !v)}
              style={{ flex: 1, lineHeight: 1.3 }}
            >
              <span style={{ fontWeight: 500 }}>Remember me on this device</span>
              <span style={{
                display: 'block',
                fontSize: 12,
                color: 'var(--text-muted, rgba(235,235,245,0.50))',
                marginTop: 2,
              }}>
                {remember
                  ? 'Stay signed in for up to 30 days.'
                  : 'You\u2019ll need to sign in again the next time you open ClipAI.'}
              </span>
            </span>
          </label>

          {error && (
            <div
              role="alert"
              style={{
                marginTop: 16,
                padding: '12px 14px',
                borderRadius: 12,
                background: 'rgba(255, 69, 58, 0.10)',
                border: '1px solid rgba(255, 69, 58, 0.35)',
                color: '#ff8a80',
                fontSize: 14,
                fontWeight: 500,
                lineHeight: 1.4,
              }}
            >
              {error}
            </div>
          )}

          <button
            type="submit"
            disabled={submitting || !username || !password}
            style={{
              marginTop: 24,
              width: '100%',
              minHeight: 50,
              padding: '0 16px',
              background:
                submitting || !username || !password
                  ? 'rgba(10,132,255,0.45)'
                  : 'var(--accent-cyan, #0A84FF)',
              color: '#fff',
              border: 'none',
              borderRadius: 14,
              fontSize: 17,
              fontWeight: 600,
              fontFamily: fontStack,
              letterSpacing: '-0.01em',
              cursor:
                submitting || !username || !password ? 'not-allowed' : 'pointer',
              transition: 'background 0.15s ease, transform 0.1s ease',
              boxShadow:
                submitting || !username || !password
                  ? 'none'
                  : '0 8px 24px rgba(10,132,255,0.35)',
            }}
            onMouseDown={(e) => {
              if (!submitting) e.currentTarget.style.transform = 'scale(0.985)';
            }}
            onMouseUp={(e) => { e.currentTarget.style.transform = 'scale(1)'; }}
            onMouseLeave={(e) => { e.currentTarget.style.transform = 'scale(1)'; }}
          >
            {submitting ? 'Signing in…' : 'Sign in'}
          </button>
        </form>

        {bootstrap && !bootstrap.has_users && (
          <div
            style={{
              marginTop: 24,
              padding: '14px 16px',
              borderRadius: 12,
              background: 'rgba(10,132,255,0.08)',
              border: '1px solid rgba(10,132,255,0.25)',
              fontSize: 13,
              lineHeight: 1.5,
              color: 'var(--text-secondary, rgba(235,235,245,0.65))',
            }}
          >
            <b style={{ color: 'var(--text-primary, #f5f5f7)' }}>
              First run.
            </b>{' '}
            The default admin is <Mono>Jadmin</Mono>. Check the server console
            for the initial password, or set one via{' '}
            <Mono>CLIPAI_ADMIN_PASSWORD</Mono> before starting.
          </div>
        )}

        <p
          style={{
            marginTop: 28,
            marginBottom: 0,
            fontSize: 12,
            textAlign: 'center',
            color: 'var(--text-muted, rgba(235,235,245,0.40))',
            letterSpacing: '-0.005em',
          }}
        >
          ClipAI
        </p>
      </div>
    </div>
  );
}

/**
 * Apple-style floating-label text field.
 *
 * 56 px tall, 17 px input font (avoids mobile-Safari focus zoom),
 * label that animates up + scales down when the input is focused or
 * has a value. Cyan focus ring. Disabled state dims to 50 %.
 */
function FloatingField({
  label, id, type, value, onChange, disabled, focused,
  onFocus, onBlur, inputRef, autoComplete, autoFocus,
}) {
  const isLifted = focused || (value && value.length > 0);
  return (
    <div
      style={{
        position: 'relative',
        height: 56,
        background: 'var(--bg-base, rgba(20,20,28,0.6))',
        border: `1px solid ${
          focused
            ? 'var(--accent-cyan, #0A84FF)'
            : 'var(--border, rgba(255,255,255,0.10))'
        }`,
        borderRadius: 14,
        boxShadow: focused
          ? '0 0 0 4px rgba(10,132,255,0.18)'
          : 'none',
        transition: 'border-color 0.15s ease, box-shadow 0.15s ease',
        opacity: disabled ? 0.5 : 1,
      }}
    >
      <label
        htmlFor={id}
        style={{
          position: 'absolute',
          left: 16,
          top: '50%',
          transform: isLifted
            ? 'translateY(-22px) scale(0.78)'
            : 'translateY(-50%) scale(1)',
          transformOrigin: 'left center',
          color: focused
            ? 'var(--accent-cyan, #0A84FF)'
            : 'var(--text-muted, rgba(235,235,245,0.55))',
          fontSize: 16,
          fontWeight: 500,
          letterSpacing: '-0.01em',
          pointerEvents: 'none',
          transition: 'transform 0.18s ease, color 0.15s ease',
          background: 'transparent',
        }}
      >
        {label}
      </label>
      <input
        ref={inputRef}
        id={id}
        type={type}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onFocus={onFocus}
        onBlur={onBlur}
        autoComplete={autoComplete}
        autoFocus={autoFocus}
        disabled={disabled}
        spellCheck={false}
        autoCapitalize="off"
        autoCorrect="off"
        style={{
          position: 'absolute',
          inset: 0,
          width: '100%',
          height: '100%',
          padding: '22px 16px 8px',
          background: 'transparent',
          border: 'none',
          outline: 'none',
          color: 'var(--text-primary, #f5f5f7)',
          fontSize: 17,
          fontWeight: 500,
          letterSpacing: '-0.01em',
          fontFamily: 'inherit',
          // Hide the browser's password-reveal eye to keep the field clean
          // on Edge / Safari.
          WebkitTextSecurity: undefined,
          boxSizing: 'border-box',
        }}
      />
    </div>
  );
}

function Mono({ children }) {
  return (
    <code
      style={{
        fontFamily:
          'ui-monospace, "SF Mono", Menlo, Monaco, Consolas, monospace',
        fontSize: 12,
        padding: '1px 6px',
        borderRadius: 6,
        background: 'rgba(255,255,255,0.07)',
        color: 'var(--text-primary, #f5f5f7)',
      }}
    >
      {children}
    </code>
  );
}
