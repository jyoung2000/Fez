import React, { useCallback, useEffect, useState } from 'react';
import { useAuth } from '../auth/AuthContext';
import { showToast } from './Toast';

/**
 * Self-contained share dialog. Pass a ``jobId`` (and optionally a
 * ``clipId``) — the dialog issues a signed share-link via
 * ``POST /api/share/links``, displays the public URL, copies on
 * click, and lets the owner revoke any of their existing links to
 * the same job. Recipients of the URL can view the linked content
 * without signing in.
 *
 * The button that triggers this dialog is owned by the parent page
 * (Analysis / ClipSEO) so it can sit naturally in the page header.
 */
export default function ShareDialog({ open, onClose, jobId, clipId, title }) {
  const { apiFetch } = useAuth();
  const [links, setLinks] = useState([]);
  const [loading, setLoading] = useState(false);
  const [creating, setCreating] = useState(false);
  const [ttlDays, setTtlDays] = useState(365);
  const [scope, setScope] = useState(clipId != null ? 'clip' : 'job');
  const [note, setNote] = useState('');
  const [err, setErr] = useState('');
  const [copyState, setCopyState] = useState({});

  const refresh = useCallback(async () => {
    if (!jobId) return;
    setLoading(true);
    setErr('');
    try {
      const data = await apiFetch(`/api/share/links?job_id=${encodeURIComponent(jobId)}`);
      setLinks(data?.links || []);
    } catch (e) {
      setErr(e?.message || 'Failed to load links');
    } finally {
      setLoading(false);
    }
  }, [apiFetch, jobId]);

  useEffect(() => { if (open) refresh(); }, [open, refresh]);
  useEffect(() => { setScope(clipId != null ? 'clip' : 'job'); }, [clipId]);

  async function createLink() {
    setCreating(true);
    setErr('');
    try {
      const body = {
        job_id: jobId,
        scope,
        ttl_days: Math.max(1, Math.min(3650, Number(ttlDays) || 365)),
        note,
      };
      if (scope === 'clip') body.clip_id = clipId;
      await apiFetch('/api/share/links', { method: 'POST', body });
      await refresh();
    } catch (e) {
      setErr(e?.message || 'Failed to create link');
    } finally {
      setCreating(false);
    }
  }

  async function revoke(token) {
    if (!window.confirm('Revoke this share link? Anyone using it will lose access.')) return;
    try {
      await apiFetch(`/api/share/links/${encodeURIComponent(token)}`, { method: 'DELETE' });
      await refresh();
    } catch (e) {
      setErr(e?.message || 'Revoke failed');
    }
  }

  function publicUrl(link) {
    const base = window.location.origin;
    if (link.scope === 'clip' && link.clip_id != null) {
      return `${base}/share/${link.token}/clip/${link.clip_id}`;
    }
    return `${base}/share/${link.token}`;
  }

  async function copy(token, link) {
    const url = publicUrl(link);
    try {
      await navigator.clipboard.writeText(url);
      setCopyState((s) => ({ ...s, [token]: 'copied' }));
      setTimeout(() => setCopyState((s) => ({ ...s, [token]: '' })), 1500);
    } catch {
      setCopyState((s) => ({ ...s, [token]: 'failed' }));
    }
  }

  if (!open) return null;

  return (
    <div style={overlay} onClick={onClose}>
      <div onClick={(e) => e.stopPropagation()} style={card}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8 }}>
          <h3 style={{ margin: 0, fontSize: 16 }}>Share {scope === 'clip' ? 'clip' : 'analysis'}</h3>
          <button style={ghostBtn} onClick={onClose}>Close</button>
        </div>
        <p style={{ margin: '0 0 16px', fontSize: 12, color: 'var(--text-muted)' }}>
          Anyone with the link can view {scope === 'clip' ? 'this clip' : 'this analysis'} without signing in.
          Revoke a link anytime to cut access.
        </p>

        {err && <div style={errBox}>{err}</div>}

        {/* Create form */}
        <div style={section}>
          <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 8 }}>Create a new link</div>
          <div style={{ display: 'grid', gridTemplateColumns: clipId != null ? 'auto 1fr 100px' : '1fr 100px', gap: 8, alignItems: 'end' }}>
            {clipId != null && (
              <select value={scope} onChange={(e) => setScope(e.target.value)} style={inp}>
                <option value="clip">This clip only</option>
                <option value="job">Whole analysis</option>
              </select>
            )}
            <input
              placeholder="Note (optional, e.g. 'Sent to client')"
              value={note}
              onChange={(e) => setNote(e.target.value)}
              style={inp}
              maxLength={200}
            />
            <input
              type="number"
              min={1}
              max={3650}
              value={ttlDays}
              onChange={(e) => setTtlDays(e.target.value)}
              style={inp}
              title="Days until link expires"
            />
          </div>
          <div style={{ marginTop: 10, display: 'flex', justifyContent: 'flex-end' }}>
            <button onClick={createLink} disabled={creating} style={primaryBtn}>
              {creating ? 'Creating…' : 'Create link'}
            </button>
          </div>
        </div>

        {/* Existing links */}
        <div style={section}>
          <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 8 }}>
            Existing links {loading && <span style={{ color: 'var(--text-muted)' }}>· loading…</span>}
          </div>
          {(!loading && links.length === 0) && (
            <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>No share links yet.</div>
          )}
          <ul style={{ listStyle: 'none', margin: 0, padding: 0 }}>
            {links.map((l) => (
              <li key={l.token} style={linkRow}>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                    <span style={badge(l.scope === 'clip' ? '#a855f7' : '#00D9FF')}>
                      {l.scope === 'clip' ? `clip #${l.clip_id}` : 'whole job'}
                    </span>
                    {l.note && <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>{l.note}</span>}
                  </div>
                  <div style={{ fontFamily: 'monospace', fontSize: 11, color: 'var(--text-secondary)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', marginTop: 4 }}>
                    {publicUrl(l)}
                  </div>
                  <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 2 }}>
                    Expires {l.expires_at?.split('T')[0]}
                  </div>
                </div>
                <div style={{ display: 'flex', gap: 6 }}>
                  <button onClick={() => copy(l.token, l)} style={ghostBtn}>
                    {copyState[l.token] === 'copied' ? 'Copied ✓' : copyState[l.token] === 'failed' ? 'Copy failed' : 'Copy URL'}
                  </button>
                  <button onClick={() => revoke(l.token)} style={{ ...ghostBtn, color: '#ffb4b4', borderColor: 'rgba(239,68,68,0.35)' }}>
                    Revoke
                  </button>
                </div>
              </li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
}

const overlay = {
  position: 'fixed', inset: 0, zIndex: 100,
  background: 'rgba(0,0,0,0.6)',
  display: 'flex', alignItems: 'center', justifyContent: 'center',
  padding: 24,
};
const card = {
  width: '100%', maxWidth: 560,
  background: 'var(--bg-elevated, #151526)',
  border: '1px solid var(--border, #26263a)',
  borderRadius: 10,
  padding: 24,
  maxHeight: '85vh',
  overflowY: 'auto',
};
const section = {
  marginTop: 12,
  padding: 12,
  background: 'var(--bg-panel, #1a1a2e)',
  border: '1px solid var(--border, #26263a)',
  borderRadius: 8,
};
const inp = {
  width: '100%',
  padding: '8px 10px',
  background: 'var(--bg-base, #0a0a0f)',
  border: '1px solid var(--border, #26263a)',
  color: 'var(--text-primary, #eee)',
  borderRadius: 6,
  fontSize: 13,
  boxSizing: 'border-box',
};
const ghostBtn = {
  padding: '6px 12px',
  background: 'transparent',
  border: '1px solid var(--border, #26263a)',
  color: 'var(--text-primary, #eee)',
  borderRadius: 6,
  cursor: 'pointer',
  fontSize: 12,
};
const primaryBtn = {
  padding: '8px 16px',
  background: 'var(--accent-cyan, #00D9FF)',
  color: '#000',
  border: 'none',
  borderRadius: 6,
  cursor: 'pointer',
  fontSize: 12,
  fontWeight: 600,
};
const errBox = {
  padding: '8px 12px',
  marginBottom: 12,
  borderRadius: 6,
  background: 'rgba(239, 68, 68, 0.12)',
  border: '1px solid rgba(239, 68, 68, 0.35)',
  color: '#ffb4b4',
  fontSize: 12,
};
const linkRow = {
  padding: '10px 0',
  display: 'flex',
  gap: 12,
  alignItems: 'center',
  borderTop: '1px solid var(--border, #26263a)',
};
const badge = (color) => ({
  padding: '2px 6px',
  borderRadius: 4,
  background: color + '22',
  border: `1px solid ${color}66`,
  color,
  fontSize: 10,
  fontWeight: 600,
  textTransform: 'uppercase',
});

/**
 * One-click share button.
 *
 * Behavior:
 *   1. Click → POST /api/share/links to mint a public, unauthenticated
 *      share token (re-uses the most recent existing token for the
 *      same job + scope when one exists, so repeat clicks don't pile
 *      up tokens).
 *   2. Copy the resulting ``/share/<token>[/clip/<id>]`` URL to the
 *      clipboard.
 *   3. Flash "Copied!" via toast + button label for 2s.
 *
 * No popup, no dialog, no menu. The full revoke-management flow lives
 * in Settings → Users (admins) for the rare case anyone needs to kill
 * a token; the per-page share button optimises for the common case
 * (grab a link and paste it).
 */
export function ShareButton({ jobId, clipId, label = 'Share', style }) {
  const { apiFetch } = useAuth();
  const [busy, setBusy] = useState(false);
  const [copied, setCopied] = useState(false);

  const buildPublicUrl = useCallback((link) => {
    const base = window.location.origin;
    if (link.scope === 'clip' && link.clip_id != null) {
      return `${base}/share/${link.token}/clip/${link.clip_id}`;
    }
    return `${base}/share/${link.token}`;
  }, []);

  const writeClipboard = useCallback(async (text) => {
    // navigator.clipboard.writeText requires HTTPS or localhost; fall
    // back to the legacy textarea+execCommand path for plain-HTTP dev.
    if (navigator.clipboard && window.isSecureContext) {
      try { await navigator.clipboard.writeText(text); return true; }
      catch { /* fall through */ }
    }
    try {
      const ta = document.createElement('textarea');
      ta.value = text;
      ta.style.position = 'fixed';
      ta.style.left = '-9999px';
      ta.style.opacity = '0';
      document.body.appendChild(ta);
      ta.focus();
      ta.select();
      const ok = document.execCommand('copy');
      document.body.removeChild(ta);
      return ok;
    } catch { return false; }
  }, []);

  const handleClick = useCallback(async () => {
    if (!jobId || busy) return;
    setBusy(true);
    try {
      const wantClipScope = clipId != null;
      const scope = wantClipScope ? 'clip' : 'job';
      // Re-use an existing matching token when one is available so
      // we don't mint a new token on every click.
      let link = null;
      try {
        const data = await apiFetch(
          `/api/share/links?job_id=${encodeURIComponent(jobId)}`,
        );
        const links = (data && data.links) || [];
        link = links.find((l) => {
          if (l.scope !== scope) return false;
          if (wantClipScope && Number(l.clip_id) !== Number(clipId)) return false;
          // Skip expired tokens.
          if (l.expires_at && Date.parse(l.expires_at) < Date.now()) return false;
          return true;
        }) || null;
      } catch { /* will fall through to create */ }

      if (!link) {
        const body = { job_id: jobId, scope, ttl_days: 365 };
        if (wantClipScope) body.clip_id = clipId;
        const created = await apiFetch('/api/share/links', {
          method: 'POST', body,
        });
        link = (created && (created.link || created)) || null;
      }
      if (!link || !link.token) {
        throw new Error('Server did not return a share token');
      }

      const url = buildPublicUrl(link);
      const ok = await writeClipboard(url);
      if (ok) {
        showToast('Share link copied!', 'success');
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
      } else {
        showToast(`Share link: ${url}`, 'info');
      }
    } catch (e) {
      showToast(`Share failed: ${e?.message || e}`, 'error');
    } finally {
      setBusy(false);
    }
  }, [jobId, clipId, busy, apiFetch, buildPublicUrl, writeClipboard]);

  return (
    <button
      onClick={handleClick}
      disabled={busy || !jobId}
      title={busy ? 'Creating share link…' : 'Copy a public link to this page'}
      style={{
        padding: '6px 14px',
        background: copied
          ? 'var(--success, #34C759)'
          : 'var(--accent-cyan, #0A84FF)',
        // White label + icon on every page (Analysis, ClipSEO, etc.)
        // — guaranteed contrast against the cyan / green fill
        // regardless of theme.
        color: '#fff',
        border: 'none',
        borderRadius: 6,
        cursor: busy ? 'wait' : 'pointer',
        opacity: busy ? 0.85 : 1,
        fontSize: 12,
        fontWeight: 600,
        display: 'inline-flex',
        alignItems: 'center',
        gap: 6,
        transition: 'background 0.15s ease',
        ...style,
      }}
    >
      {copied ? (
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
          <polyline points="20 6 9 17 4 12" />
        </svg>
      ) : (
        <svg
          width="14"
          height="14"
          viewBox="0 0 24 24"
          fill="none"
          stroke="#fff"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <circle cx="18" cy="5" r="3"></circle>
          <circle cx="6" cy="12" r="3"></circle>
          <circle cx="18" cy="19" r="3"></circle>
          <line x1="8.59" y1="13.51" x2="15.42" y2="17.49"></line>
          <line x1="15.41" y1="6.51" x2="8.59" y2="10.49"></line>
        </svg>
      )}
      {busy ? 'Copying…' : (copied ? 'Copied!' : label)}
    </button>
  );
}
