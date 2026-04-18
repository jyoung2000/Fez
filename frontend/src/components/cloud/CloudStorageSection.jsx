/**
 * Settings page section: Cloud Storage.
 *
 * Shows one card per provider (Google Drive + Box). For each provider:
 *
 *   - If the server has no client id/secret configured, we show an
 *     inline credentials form plus a link to the setup doc so the
 *     user can paste their OAuth client id/secret/redirect URI
 *     without touching .env files or container vars.
 *   - If configured but not connected, we show a Connect button that
 *     pops open the OAuth URL in a new window. When that window
 *     redirects back through /api/cloud/{provider}/callback it lands
 *     on /settings?connected=... — we detect that query string on mount
 *     and refresh the provider list.
 *   - If connected, we show the display name (email), scopes, and a
 *     Disconnect button.
 *
 * The credentials form also exposes a "Test credentials" button that
 * calls /api/cloud/config/test/{provider} and renders a per-check
 * pass/fail list so the user can see exactly which field is wrong.
 */
import React, { useCallback, useEffect, useState } from 'react';
import useCloudProviders from '../../hooks/useCloudProviders';

const PROVIDER_LABELS = {
  google_drive: {
    title: 'Google Drive',
    description:
      'Browse videos from your Google Drive and analyze them without a local download.',
    icon: 'GD',
    clientIdPlaceholder: '1234567890-abc.apps.googleusercontent.com',
    redirectHint: '/api/cloud/google_drive/callback',
  },
  box: {
    title: 'Box',
    description:
      'Browse videos from your Box account and analyze them without a local download.',
    icon: 'BX',
    clientIdPlaceholder: 'abc123def456',
    redirectHint: '/api/cloud/box/callback',
  },
};

function cardStyle() {
  return {
    background: 'var(--bg-panel)',
    border: '1px solid var(--border)',
    borderRadius: 'var(--radius-md)',
    padding: 16,
    marginBottom: 12,
  };
}

function buttonStyle(variant = 'primary') {
  if (variant === 'danger') {
    return {
      padding: '8px 16px',
      fontSize: 13,
      background: 'transparent',
      color: 'var(--text-secondary)',
      border: '1px solid var(--border)',
      borderRadius: 'var(--radius-sm)',
      cursor: 'pointer',
    };
  }
  if (variant === 'secondary') {
    return {
      padding: '8px 16px',
      fontSize: 13,
      background: 'transparent',
      color: 'var(--accent-cyan)',
      border: '1px solid var(--accent-cyan)',
      borderRadius: 'var(--radius-sm)',
      cursor: 'pointer',
      fontWeight: 500,
    };
  }
  return {
    padding: '8px 16px',
    fontSize: 13,
    background: 'var(--accent-cyan)',
    color: '#000',
    border: 'none',
    borderRadius: 'var(--radius-sm)',
    cursor: 'pointer',
    fontWeight: 600,
  };
}

function inputStyle() {
  return {
    width: '100%',
    padding: '8px 10px',
    fontSize: 13,
    background: 'var(--bg-panel)',
    color: 'var(--text-primary)',
    border: '1px solid var(--border)',
    borderRadius: 'var(--radius-sm)',
    fontFamily: 'var(--font-mono, ui-monospace, monospace)',
    boxSizing: 'border-box',
  };
}

function labelStyle() {
  return {
    display: 'block',
    fontSize: 11,
    color: 'var(--text-muted)',
    textTransform: 'uppercase',
    letterSpacing: '0.04em',
    marginBottom: 4,
    marginTop: 10,
  };
}

function buildDefaultRedirectUri(provider) {
  if (typeof window === 'undefined') return '';
  const { protocol, host } = window.location;
  return `${protocol}//${host}/api/cloud/${provider}/callback`;
}

function CredentialsForm({ provider, onSaved }) {
  const label = PROVIDER_LABELS[provider] || {};
  const [clientId, setClientId] = useState('');
  const [clientSecret, setClientSecret] = useState('');
  const [redirectUri, setRedirectUri] = useState('');
  const [secretAlreadySet, setSecretAlreadySet] = useState(false);
  const [secretMasked, setSecretMasked] = useState('');
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState(null); // { ok, message }
  const [testResult, setTestResult] = useState(null);

  // Prefill the form from whatever the server currently has.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const resp = await fetch('/api/cloud/config');
        if (!resp.ok) return;
        const data = await resp.json();
        const row = data.providers?.[provider];
        if (!row || cancelled) return;
        setClientId(row.client_id || '');
        setRedirectUri(row.redirect_uri || buildDefaultRedirectUri(provider));
        setSecretAlreadySet(!!row.client_secret_set);
        setSecretMasked(row.client_secret_masked || '');
      } catch {
        // non-fatal — the form still works for a fresh install
        if (!cancelled) {
          setRedirectUri(buildDefaultRedirectUri(provider));
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [provider]);

  const handleSave = useCallback(
    async (e) => {
      e?.preventDefault?.();
      setBusy(true);
      setStatus(null);
      setTestResult(null);
      try {
        const resp = await fetch('/api/cloud/config', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            provider,
            client_id: clientId.trim(),
            // Empty client_secret = "keep existing" on the server side.
            client_secret: clientSecret.trim(),
            redirect_uri: redirectUri.trim(),
          }),
        });
        const data = await resp.json().catch(() => ({}));
        if (!resp.ok) {
          throw new Error(data.detail || `HTTP ${resp.status}`);
        }
        setStatus({
          ok: true,
          message: data.configured
            ? 'Saved. Click Connect to authorize your account.'
            : 'Saved — but the provider still reports as not configured. Check the fields.',
        });
        // If the caller gave us a refresh callback, use it so the card
        // can swap from "enter credentials" to "Connect".
        if (onSaved) await onSaved();
      } catch (err) {
        setStatus({ ok: false, message: err.message || 'Save failed' });
      } finally {
        setBusy(false);
      }
    },
    [provider, clientId, clientSecret, redirectUri, onSaved]
  );

  const handleTest = useCallback(async () => {
    setBusy(true);
    setTestResult(null);
    try {
      const resp = await fetch(`/api/cloud/config/test/${provider}`, {
        method: 'POST',
      });
      const data = await resp.json().catch(() => ({}));
      setTestResult({
        status: data.status || (resp.ok ? 'ok' : 'error'),
        checks: Array.isArray(data.checks) ? data.checks : [],
        httpStatus: resp.status,
      });
    } catch (err) {
      setTestResult({
        status: 'error',
        checks: [{ label: 'network', ok: false, detail: err.message }],
      });
    } finally {
      setBusy(false);
    }
  }, [provider]);

  return (
    <form onSubmit={handleSave} style={{ marginTop: 12 }}>
      <div
        style={{
          fontSize: 12,
          color: 'var(--text-muted)',
          marginBottom: 8,
        }}
      >
        Paste your OAuth client credentials from the provider's developer
        console. Need help?{' '}
        <a
          href="/docs/cloud-storage/SETUP.md"
          target="_blank"
          rel="noopener noreferrer"
          style={{ color: 'var(--accent-cyan)' }}
        >
          Open the setup guide →
        </a>
      </div>

      <label style={labelStyle()}>Client ID</label>
      <input
        type="text"
        autoComplete="off"
        spellCheck={false}
        placeholder={label.clientIdPlaceholder}
        value={clientId}
        onChange={(e) => setClientId(e.target.value)}
        style={inputStyle()}
      />

      <label style={labelStyle()}>
        Client Secret
        {secretAlreadySet && (
          <span style={{ marginLeft: 8, textTransform: 'none', color: 'var(--text-secondary)' }}>
            (saved: {secretMasked}) — leave blank to keep
          </span>
        )}
      </label>
      <input
        type="password"
        autoComplete="off"
        spellCheck={false}
        placeholder={secretAlreadySet ? '•••••••••• (leave blank to keep existing)' : 'Paste client secret'}
        value={clientSecret}
        onChange={(e) => setClientSecret(e.target.value)}
        style={inputStyle()}
      />

      <label style={labelStyle()}>
        Redirect URI
        <span style={{ marginLeft: 8, textTransform: 'none', color: 'var(--text-muted)' }}>
          (must match the value you set in the provider's console)
        </span>
      </label>
      <input
        type="text"
        autoComplete="off"
        spellCheck={false}
        placeholder={`${typeof window !== 'undefined' ? window.location.origin : 'http://host'}${label.redirectHint || ''}`}
        value={redirectUri}
        onChange={(e) => setRedirectUri(e.target.value)}
        style={inputStyle()}
      />

      <div style={{ display: 'flex', gap: 8, marginTop: 14, flexWrap: 'wrap' }}>
        <button type="submit" disabled={busy} style={buttonStyle('primary')}>
          {busy ? 'Working...' : 'Save credentials'}
        </button>
        <button
          type="button"
          onClick={handleTest}
          disabled={busy}
          style={buttonStyle('secondary')}
        >
          Test credentials
        </button>
      </div>

      {status && (
        <div
          style={{
            marginTop: 12,
            padding: 10,
            fontSize: 12,
            border: '1px solid',
            borderColor: status.ok ? 'var(--accent-green, #4caf50)' : 'var(--accent-red, #f66)',
            borderRadius: 'var(--radius-sm)',
            color: status.ok ? 'var(--accent-green, #4caf50)' : 'var(--accent-red, #f66)',
          }}
        >
          {status.message}
        </div>
      )}

      {testResult && (
        <div
          style={{
            marginTop: 12,
            padding: 12,
            border: '1px solid var(--border)',
            borderRadius: 'var(--radius-sm)',
            background: 'var(--bg-hover, #0e1116)',
          }}
        >
          <div
            style={{
              fontSize: 12,
              fontWeight: 600,
              marginBottom: 8,
              color:
                testResult.status === 'ok'
                  ? 'var(--accent-green, #4caf50)'
                  : 'var(--accent-red, #f66)',
            }}
          >
            {testResult.status === 'ok' ? 'All checks passed' : 'Some checks failed'}
          </div>
          <ul style={{ margin: 0, padding: 0, listStyle: 'none' }}>
            {testResult.checks.map((c, idx) => (
              <li
                key={idx}
                style={{
                  fontSize: 12,
                  padding: '3px 0',
                  color: c.ok ? 'var(--text-secondary)' : 'var(--accent-red, #f66)',
                }}
              >
                <span style={{ marginRight: 8 }}>{c.ok ? '✓' : '✗'}</span>
                {c.label}
                {c.detail ? (
                  <span style={{ marginLeft: 8, color: 'var(--text-muted)' }}>— {c.detail}</span>
                ) : null}
              </li>
            ))}
          </ul>
        </div>
      )}
    </form>
  );
}

export default function CloudStorageSection() {
  const { loading, enabled, providers, error, refresh } = useCloudProviders();
  const [busy, setBusy] = useState('');
  const [expanded, setExpanded] = useState(() => new Set());

  // If a callback-driven redirect lands us here with ?connected=<provider>,
  // refresh the provider list and clean up the URL.
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const connected = params.get('connected');
    if (connected) {
      refresh();
      params.delete('connected');
      const newQs = params.toString();
      const newPath =
        window.location.pathname + (newQs ? `?${newQs}` : '') + window.location.hash;
      window.history.replaceState({}, '', newPath);
    }
  }, [refresh]);

  // Listen for postMessage from the OAuth popup (if a child window flow
  // is used instead of a full top-level redirect).
  useEffect(() => {
    const onMessage = (event) => {
      if (!event?.data || typeof event.data !== 'object') return;
      if (event.data.type === 'clipai:cloud-connected') {
        refresh();
      }
    };
    window.addEventListener('message', onMessage);
    return () => window.removeEventListener('message', onMessage);
  }, [refresh]);

  const handleConnect = useCallback(
    async (providerName) => {
      setBusy(providerName);
      try {
        const resp = await fetch(`/api/cloud/${providerName}/authorize`);
        if (!resp.ok) {
          const body = await resp.json().catch(() => ({}));
          throw new Error(body.detail || `HTTP ${resp.status}`);
        }
        const { authorize_url } = await resp.json();
        // Open in a new tab — the callback redirect handler will bounce
        // users back to /settings?connected=<provider>. This is the most
        // compatible approach across browsers and iframe sandboxes.
        window.open(authorize_url, '_blank', 'noopener');
      } catch (err) {
        alert(`Failed to start OAuth for ${providerName}: ${err.message}`);
      } finally {
        setBusy('');
      }
    },
    []
  );

  const handleDisconnect = useCallback(
    async (accountId, providerName) => {
      if (!window.confirm(`Disconnect ${PROVIDER_LABELS[providerName]?.title || providerName}?`)) {
        return;
      }
      setBusy(providerName);
      try {
        const resp = await fetch(`/api/cloud/accounts/${accountId}`, { method: 'DELETE' });
        if (!resp.ok) {
          const body = await resp.json().catch(() => ({}));
          throw new Error(body.detail || `HTTP ${resp.status}`);
        }
        await refresh();
      } catch (err) {
        alert(`Failed to disconnect: ${err.message}`);
      } finally {
        setBusy('');
      }
    },
    [refresh]
  );

  const toggleExpanded = useCallback((providerName) => {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(providerName)) {
        next.delete(providerName);
      } else {
        next.add(providerName);
      }
      return next;
    });
  }, []);

  if (loading) {
    return (
      <div style={{ padding: 16, color: 'var(--text-secondary)' }}>
        Loading cloud storage providers...
      </div>
    );
  }

  if (!enabled) {
    return (
      <div style={{ padding: 16, color: 'var(--text-secondary)' }}>
        Cloud storage integration is disabled. Set
        <code style={{ margin: '0 6px' }}>CLIPAI_CLOUD_STORAGE_ENABLED=true</code>
        to enable it.
      </div>
    );
  }

  if (error) {
    return (
      <div style={{ padding: 16, color: 'var(--accent-red, #f66)' }}>
        Failed to load cloud providers: {error}
      </div>
    );
  }

  return (
    <div>
      <p
        style={{
          fontSize: 13,
          color: 'var(--text-secondary)',
          marginTop: 0,
          marginBottom: 12,
        }}
      >
        Connect a cloud account to browse and import videos directly from Google Drive or
        Box. Files are streamed server-side &mdash; no tokens touch your browser and nothing
        is copied to your local machine. Cloud storage is optional; you can always
        upload local files from the Upload page without connecting a provider.
      </p>

      {providers.map((p) => {
        const label = PROVIDER_LABELS[p.name] || { title: p.name, description: '', icon: '?' };
        const isExpanded = expanded.has(p.name);
        return (
          <div key={p.name} style={cardStyle()}>
            <div style={{ display: 'flex', alignItems: 'flex-start', gap: 12 }}>
              <div
                style={{
                  width: 40,
                  height: 40,
                  borderRadius: 'var(--radius-sm)',
                  background: 'var(--bg-hover, #1a1a1a)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  fontWeight: 700,
                  color: 'var(--accent-cyan)',
                  fontFamily: 'var(--font-mono)',
                }}
              >
                {label.icon}
              </div>
              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{ fontWeight: 600, marginBottom: 4 }}>{label.title}</div>
                <div
                  style={{
                    fontSize: 13,
                    color: 'var(--text-secondary)',
                    marginBottom: 8,
                  }}
                >
                  {label.description}
                </div>

                {!p.configured && (
                  <>
                    <div
                      style={{
                        fontSize: 12,
                        color: 'var(--text-muted)',
                        marginBottom: 8,
                      }}
                    >
                      Not configured yet. Paste your OAuth client credentials below to
                      enable the {label.title} tab on the Upload page.
                    </div>
                    <button
                      type="button"
                      onClick={() => toggleExpanded(p.name)}
                      style={buttonStyle('secondary')}
                    >
                      {isExpanded ? 'Hide form' : 'Enter credentials'}
                    </button>
                    {isExpanded && (
                      <CredentialsForm provider={p.name} onSaved={refresh} />
                    )}
                  </>
                )}

                {p.configured && !p.connected && (
                  <div>
                    <button
                      type="button"
                      onClick={() => handleConnect(p.name)}
                      disabled={busy === p.name}
                      style={buttonStyle('primary')}
                    >
                      {busy === p.name ? 'Opening...' : 'Connect'}
                    </button>
                    <button
                      type="button"
                      onClick={() => toggleExpanded(p.name)}
                      style={{ ...buttonStyle('secondary'), marginLeft: 8 }}
                    >
                      {isExpanded ? 'Hide credentials' : 'Edit credentials'}
                    </button>
                    {isExpanded && (
                      <CredentialsForm provider={p.name} onSaved={refresh} />
                    )}
                  </div>
                )}

                {p.configured && p.connected && p.account && (
                  <div>
                    <div
                      style={{
                        fontSize: 13,
                        color: 'var(--text-primary)',
                        marginBottom: 6,
                      }}
                    >
                      Connected as{' '}
                      <strong>{p.account.display_name}</strong>
                    </div>
                    {p.account.token_expires_at && (
                      <div
                        style={{
                          fontSize: 11,
                          color: 'var(--text-muted)',
                          marginBottom: 8,
                        }}
                      >
                        Token expires: {new Date(p.account.token_expires_at).toLocaleString()}
                      </div>
                    )}
                    <button
                      type="button"
                      onClick={() => handleDisconnect(p.account.id, p.name)}
                      disabled={busy === p.name}
                      style={buttonStyle('danger')}
                    >
                      {busy === p.name ? 'Working...' : 'Disconnect'}
                    </button>
                    <button
                      type="button"
                      onClick={() => toggleExpanded(p.name)}
                      style={{ ...buttonStyle('secondary'), marginLeft: 8 }}
                    >
                      {isExpanded ? 'Hide credentials' : 'Edit credentials'}
                    </button>
                    {isExpanded && (
                      <CredentialsForm provider={p.name} onSaved={refresh} />
                    )}
                  </div>
                )}
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}
