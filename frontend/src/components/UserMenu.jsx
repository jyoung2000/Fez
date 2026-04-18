import React, { useState } from 'react';
import { useAuth } from '../auth/AuthContext';

/**
 * Pinned to the sidebar footer. Shows the current username + role and
 * a logout button. Clicking the username reveals a small menu with
 * "Change password" and "Sign out".
 */
export default function UserMenu({ collapsed }) {
  const { user, logout, changePassword, isAdmin } = useAuth();
  const [open, setOpen] = useState(false);
  const [showPw, setShowPw] = useState(false);

  if (!user) return null;

  const initials = (user.username || '?').slice(0, 2).toUpperCase();

  return (
    <div
      style={{
        marginTop: 'auto',
        padding: '10px 12px',
        borderTop: '1px solid var(--border, #26263a)',
        background: 'var(--bg-elevated, #151526)',
        position: 'relative',
      }}
    >
      <button
        onClick={() => setOpen((v) => !v)}
        title={`${user.username} (${user.role})`}
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 10,
          width: '100%',
          padding: '6px 8px',
          borderRadius: 6,
          background: open ? 'var(--accent-cyan-dim, rgba(10,132,255,0.12))' : 'transparent',
          color: 'var(--text-primary, #eee)',
          border: '1px solid transparent',
          cursor: 'pointer',
          textAlign: 'left',
          transition: 'background 0.15s ease',
        }}
        onMouseEnter={(e) => (e.currentTarget.style.background = 'var(--accent-cyan-dim, rgba(10,132,255,0.12))')}
        onMouseLeave={(e) => (e.currentTarget.style.background = open ? 'var(--accent-cyan-dim, rgba(10,132,255,0.12))' : 'transparent')}
      >
        <div style={{
          width: 28, height: 28, borderRadius: '50%',
          background: isAdmin ? 'var(--accent-cyan, #00D9FF)' : 'var(--accent-purple, #a855f7)',
          color: '#000', fontWeight: 700,
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          fontSize: 11,
        }}>{initials}</div>
        {!collapsed && (
          <div style={{ minWidth: 0, flex: 1 }}>
            <div style={{ fontSize: 13, fontWeight: 600, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
              {user.username}
            </div>
            <div style={{ fontSize: 11, color: 'var(--text-muted, #888)' }}>
              {isAdmin ? 'Admin' : 'User'}
              {user.head_admin ? ' · head' : ''}
            </div>
          </div>
        )}
      </button>

      {open && (
        <div
          role="menu"
          style={{
            position: 'absolute',
            left: 10, right: 10, bottom: 64,
            background: 'var(--bg-panel, #1f1f32)',
            border: '1px solid var(--border, #26263a)',
            borderRadius: 8,
            padding: 6,
            boxShadow: '0 12px 32px rgba(0, 0, 0, 0.4)',
            zIndex: 50,
          }}
        >
          <button
            style={menuItemStyle}
            onClick={() => { setShowPw(true); setOpen(false); }}
            onMouseEnter={(e) => (e.currentTarget.style.background = 'var(--accent-cyan-dim, rgba(10,132,255,0.12))')}
            onMouseLeave={(e) => (e.currentTarget.style.background = 'transparent')}
          >
            Change password
          </button>
          <button
            style={{ ...menuItemStyle, color: '#ffb4b4' }}
            onClick={() => { setOpen(false); logout(); }}
            onMouseEnter={(e) => (e.currentTarget.style.background = 'var(--accent-cyan-dim, rgba(10,132,255,0.12))')}
            onMouseLeave={(e) => (e.currentTarget.style.background = 'transparent')}
          >
            Sign out
          </button>
        </div>
      )}

      {showPw && <ChangePasswordDialog onClose={() => setShowPw(false)} changePassword={changePassword} />}
    </div>
  );
}

function ChangePasswordDialog({ onClose, changePassword }) {
  const [current, setCurrent] = useState('');
  const [next, setNext] = useState('');
  const [confirm, setConfirm] = useState('');
  const [status, setStatus] = useState('');
  const [busy, setBusy] = useState(false);

  async function submit(e) {
    e.preventDefault();
    setStatus('');
    if (next !== confirm) { setStatus('New passwords do not match'); return; }
    if (next.length < 6) { setStatus('Password must be at least 6 characters'); return; }
    setBusy(true);
    try {
      await changePassword(current, next);
      setStatus('Password changed. You will be signed out of other devices.');
      setTimeout(onClose, 1500);
    } catch (err) {
      setStatus(err?.message || 'Change failed');
    } finally {
      setBusy(false);
    }
  }

  return (
    <div style={{
      position: 'fixed', inset: 0, zIndex: 100,
      background: 'rgba(0,0,0,0.6)',
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      padding: 24,
    }} onClick={onClose}>
      <div
        onClick={(e) => e.stopPropagation()}
        style={{
          width: '100%', maxWidth: 360,
          background: 'var(--bg-elevated, #151526)',
          border: '1px solid var(--border, #26263a)',
          borderRadius: 10,
          padding: 24,
        }}
      >
        <h3 style={{ margin: '0 0 16px', fontSize: 16 }}>Change password</h3>
        <form onSubmit={submit}>
          <label style={lbl}>Current password</label>
          <input type="password" value={current} onChange={(e) => setCurrent(e.target.value)} style={inp} />
          <label style={{ ...lbl, marginTop: 12 }}>New password</label>
          <input type="password" value={next} onChange={(e) => setNext(e.target.value)} style={inp} />
          <label style={{ ...lbl, marginTop: 12 }}>Confirm new password</label>
          <input type="password" value={confirm} onChange={(e) => setConfirm(e.target.value)} style={inp} />

          {status && (
            <div style={{
              marginTop: 12, padding: 10, borderRadius: 6,
              background: status.startsWith('Password changed')
                ? 'rgba(16, 185, 129, 0.12)' : 'rgba(239, 68, 68, 0.12)',
              border: `1px solid ${status.startsWith('Password changed') ? 'rgba(16, 185, 129, 0.35)' : 'rgba(239, 68, 68, 0.35)'}`,
              fontSize: 12,
            }}>{status}</div>
          )}

          <div style={{ display: 'flex', gap: 8, marginTop: 16 }}>
            <button type="button" onClick={onClose} disabled={busy} style={btnGhost}>Cancel</button>
            <button type="submit" disabled={busy} style={btnPrimary}>{busy ? 'Saving…' : 'Save'}</button>
          </div>
        </form>
      </div>
    </div>
  );
}

const menuItemStyle = {
  display: 'block',
  width: '100%',
  padding: '8px 10px',
  background: 'transparent',
  color: 'var(--text-primary, #eee)',
  border: 'none',
  borderRadius: 4,
  fontSize: 13,
  textAlign: 'left',
  cursor: 'pointer',
  transition: 'background 0.15s ease',
};

const lbl = { display: 'block', fontSize: 12, color: 'var(--text-muted, #888)', marginBottom: 4 };
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
const btnGhost = {
  flex: 1,
  padding: '8px 14px',
  background: 'transparent',
  border: '1px solid var(--border, #26263a)',
  color: 'var(--text-primary, #eee)',
  borderRadius: 6,
  cursor: 'pointer',
  fontSize: 13,
};
const btnPrimary = {
  flex: 1,
  padding: '8px 14px',
  background: 'var(--accent-cyan, #00D9FF)',
  color: '#000',
  border: 'none',
  borderRadius: 6,
  cursor: 'pointer',
  fontSize: 13,
  fontWeight: 600,
};
