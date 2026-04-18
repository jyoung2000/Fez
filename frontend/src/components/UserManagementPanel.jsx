import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { useAuth } from '../auth/AuthContext';

/**
 * User management tab. Admin-only. Exposes:
 *   - list users (username, role, active, head_admin, created_at)
 *   - create user (username + password + role)
 *   - change role (user ↔ admin)
 *   - activate / deactivate
 *   - reset password
 *   - delete user
 *   - list + revoke active sessions
 *
 * The head admin (``Jadmin``) is protected: cannot be deleted or
 * demoted. The current admin cannot delete or demote themselves.
 */
export default function UserManagementPanel() {
  const { user: me, isAdmin, apiFetch } = useAuth();
  const [users, setUsers] = useState([]);
  const [sessions, setSessions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState('');
  const [createBusy, setCreateBusy] = useState(false);

  const [newUsername, setNewUsername] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [newRole, setNewRole] = useState('user');

  const [resetTarget, setResetTarget] = useState(null); // {id, username}
  const [resetValue, setResetValue] = useState('');

  const refresh = useCallback(async () => {
    setLoading(true);
    setErr('');
    try {
      const [u, s] = await Promise.all([
        apiFetch('/api/auth/admin/users'),
        apiFetch('/api/auth/admin/sessions'),
      ]);
      setUsers(u?.users || []);
      setSessions(s?.sessions || []);
    } catch (e) {
      setErr(e?.message || 'Failed to load users');
    } finally {
      setLoading(false);
    }
  }, [apiFetch]);

  useEffect(() => { refresh(); }, [refresh]);

  const sessionsByUser = useMemo(() => {
    const m = {};
    for (const s of sessions) {
      (m[s.user_id] = m[s.user_id] || []).push(s);
    }
    return m;
  }, [sessions]);

  async function createUser(e) {
    e.preventDefault();
    setErr('');
    setCreateBusy(true);
    try {
      await apiFetch('/api/auth/admin/users', {
        method: 'POST',
        body: { username: newUsername.trim(), password: newPassword, role: newRole },
      });
      setNewUsername(''); setNewPassword(''); setNewRole('user');
      await refresh();
    } catch (e) {
      setErr(e?.message || 'Create failed');
    } finally {
      setCreateBusy(false);
    }
  }

  async function patchUser(id, patch) {
    setErr('');
    try {
      await apiFetch(`/api/auth/admin/users/${id}`, { method: 'PATCH', body: patch });
      await refresh();
    } catch (e) {
      setErr(e?.message || 'Update failed');
    }
  }

  async function deleteUser(id, username) {
    if (!window.confirm(`Delete user "${username}"? Their jobs will remain but be admin-only visible.`)) return;
    setErr('');
    try {
      await apiFetch(`/api/auth/admin/users/${id}`, { method: 'DELETE' });
      await refresh();
    } catch (e) {
      setErr(e?.message || 'Delete failed');
    }
  }

  async function resetPassword() {
    if (!resetTarget) return;
    setErr('');
    try {
      await apiFetch(`/api/auth/admin/users/${resetTarget.id}/reset_password`, {
        method: 'POST',
        body: { new_password: resetValue },
      });
      setResetTarget(null); setResetValue('');
      await refresh();
    } catch (e) {
      setErr(e?.message || 'Reset failed');
    }
  }

  async function revokeSession(token) {
    try {
      await apiFetch(`/api/auth/admin/sessions/${encodeURIComponent(token)}`, { method: 'DELETE' });
      await refresh();
    } catch (e) {
      setErr(e?.message || 'Revoke failed');
    }
  }

  if (!isAdmin) {
    return (
      <div style={{ padding: 24, color: 'var(--text-muted)' }}>
        Admin access required.
      </div>
    );
  }

  return (
    <div style={{ padding: '16px 24px', maxWidth: 960 }}>
      <h2 style={{ marginTop: 0, fontSize: 18, display: 'flex', alignItems: 'center', gap: 12 }}>
        Users
        <span style={pill}>{users.length}</span>
        <button
          style={ghostBtn}
          onClick={refresh}
          disabled={loading}
        >
          {loading ? 'Loading…' : 'Refresh'}
        </button>
      </h2>

      {err && (
        <div style={errBox}>{err}</div>
      )}

      {/* ── Create user ── */}
      <section style={card}>
        <h3 style={h3}>Create a new user</h3>
        <form onSubmit={createUser} style={{ display: 'flex', flexWrap: 'wrap', gap: 10, alignItems: 'flex-end' }}>
          <div style={{ flex: '1 1 180px' }}>
            <label style={lbl}>Username</label>
            <input value={newUsername} onChange={(e) => setNewUsername(e.target.value)} style={inp} required minLength={2} />
          </div>
          <div style={{ flex: '1 1 180px' }}>
            <label style={lbl}>Password</label>
            <input type="password" value={newPassword} onChange={(e) => setNewPassword(e.target.value)} style={inp} required minLength={6} />
          </div>
          <div style={{ flex: '0 0 130px' }}>
            <label style={lbl}>Role</label>
            <select value={newRole} onChange={(e) => setNewRole(e.target.value)} style={inp}>
              <option value="user">User</option>
              <option value="admin">Admin</option>
            </select>
          </div>
          <button type="submit" disabled={createBusy || !newUsername || !newPassword} style={primaryBtn}>
            {createBusy ? 'Creating…' : 'Create user'}
          </button>
        </form>
        <p style={hint}>
          Only two roles are supported: <b>user</b> (their own dashboard + library) and
          <b> admin</b> (manage users and see every job).
        </p>
      </section>

      {/* ── User list ── */}
      <section style={card}>
        <h3 style={h3}>Users</h3>
        <table style={tableStyle}>
          <thead>
            <tr>
              <th style={th}>Username</th>
              <th style={th}>Role</th>
              <th style={th}>Active</th>
              <th style={th}>Sessions</th>
              <th style={th}>Created</th>
              <th style={th}></th>
            </tr>
          </thead>
          <tbody>
            {users.map((u) => {
              const isSelf = me?.id === u.id;
              const locked = u.head_admin;
              return (
                <tr key={u.id} style={{ borderTop: '1px solid var(--border, #26263a)' }}>
                  <td style={td}>
                    <div style={{ fontWeight: 600 }}>{u.username}</div>
                    {locked && <div style={{ fontSize: 11, color: 'var(--accent-cyan, #00D9FF)' }}>head admin</div>}
                    {isSelf && <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>you</div>}
                  </td>
                  <td style={td}>
                    <select
                      value={u.role}
                      disabled={locked || isSelf}
                      onChange={(e) => patchUser(u.id, { role: e.target.value })}
                      style={{ ...inp, padding: '4px 8px', fontSize: 12 }}
                    >
                      <option value="user">user</option>
                      <option value="admin">admin</option>
                    </select>
                  </td>
                  <td style={td}>
                    <label style={{ display: 'inline-flex', gap: 6, alignItems: 'center', cursor: (locked || isSelf) ? 'not-allowed' : 'pointer' }}>
                      <input
                        type="checkbox"
                        checked={!!u.active}
                        disabled={locked || isSelf}
                        onChange={(e) => patchUser(u.id, { active: e.target.checked })}
                      />
                      <span style={{ fontSize: 12 }}>{u.active ? 'Active' : 'Disabled'}</span>
                    </label>
                  </td>
                  <td style={td}>
                    <div style={{ fontSize: 12 }}>
                      {(sessionsByUser[u.id] || []).length} open
                    </div>
                  </td>
                  <td style={td}>
                    <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>{u.created_at?.split('T')[0]}</div>
                  </td>
                  <td style={td}>
                    <div style={{ display: 'flex', gap: 6, justifyContent: 'flex-end' }}>
                      <button style={ghostBtn} onClick={() => { setResetTarget({ id: u.id, username: u.username }); setResetValue(''); }}>
                        Reset password
                      </button>
                      <button
                        style={{ ...ghostBtn, color: '#ffb4b4', borderColor: 'rgba(239,68,68,0.35)' }}
                        onClick={() => deleteUser(u.id, u.username)}
                        disabled={locked || isSelf}
                      >
                        Delete
                      </button>
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </section>

      {/* ── Sessions ── */}
      <section style={card}>
        <h3 style={h3}>Active sessions <span style={pill}>{sessions.length}</span></h3>
        <table style={tableStyle}>
          <thead>
            <tr>
              <th style={th}>User</th>
              <th style={th}>IP</th>
              <th style={th}>Created</th>
              <th style={th}>Last seen</th>
              <th style={th}></th>
            </tr>
          </thead>
          <tbody>
            {sessions.map((s) => {
              const u = users.find((x) => x.id === s.user_id);
              return (
                <tr key={s.token} style={{ borderTop: '1px solid var(--border, #26263a)' }}>
                  <td style={td}>{u?.username || s.user_id?.slice(0, 8)}</td>
                  <td style={{ ...td, fontFamily: 'monospace', fontSize: 11 }}>{s.ip || '—'}</td>
                  <td style={{ ...td, fontSize: 11 }}>{s.created_at}</td>
                  <td style={{ ...td, fontSize: 11 }}>{s.last_seen}</td>
                  <td style={td}>
                    <button style={ghostBtn} onClick={() => revokeSession(s.token)}>Revoke</button>
                  </td>
                </tr>
              );
            })}
            {sessions.length === 0 && (
              <tr><td style={{ ...td, color: 'var(--text-muted)' }} colSpan={5}>No active sessions.</td></tr>
            )}
          </tbody>
        </table>
      </section>

      {/* ── Reset password modal ── */}
      {resetTarget && (
        <div style={modalBack} onClick={() => setResetTarget(null)}>
          <div onClick={(e) => e.stopPropagation()} style={modalCard}>
            <h3 style={{ marginTop: 0 }}>Reset password for {resetTarget.username}</h3>
            <label style={lbl}>New password</label>
            <input type="password" value={resetValue} onChange={(e) => setResetValue(e.target.value)} style={inp} minLength={6} />
            <p style={hint}>
              Active sessions for this user will be revoked, so they'll be forced to log in again.
            </p>
            <div style={{ display: 'flex', gap: 8, marginTop: 12 }}>
              <button style={ghostBtn} onClick={() => setResetTarget(null)}>Cancel</button>
              <button style={primaryBtn} disabled={resetValue.length < 6} onClick={resetPassword}>Reset</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

const card = {
  background: 'var(--bg-panel, #1a1a2e)',
  border: '1px solid var(--border, #26263a)',
  borderRadius: 8,
  padding: 16,
  marginBottom: 16,
};
const h3 = { margin: '0 0 12px', fontSize: 14, color: 'var(--text-primary, #eee)' };
const lbl = { display: 'block', fontSize: 11, color: 'var(--text-muted, #888)', marginBottom: 4, textTransform: 'uppercase', letterSpacing: 0.5 };
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
const hint = { fontSize: 11, color: 'var(--text-muted, #888)', marginTop: 8 };
const tableStyle = { width: '100%', borderCollapse: 'collapse', fontSize: 13 };
const th = { textAlign: 'left', padding: '8px 10px', fontSize: 11, color: 'var(--text-muted, #888)', textTransform: 'uppercase', letterSpacing: 0.5 };
const td = { padding: '8px 10px', verticalAlign: 'middle' };
const pill = {
  display: 'inline-block',
  padding: '2px 8px',
  borderRadius: 999,
  // White on black so the active-user / active-session counts read
  // clearly in both light and dark themes.
  background: '#000',
  color: '#fff',
  fontSize: 11,
  fontWeight: 600,
};
const ghostBtn = {
  padding: '5px 10px',
  background: 'transparent',
  border: '1px solid var(--border, #26263a)',
  color: 'var(--text-primary, #eee)',
  borderRadius: 6,
  cursor: 'pointer',
  fontSize: 12,
};
const primaryBtn = {
  padding: '8px 14px',
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
  marginBottom: 16,
  borderRadius: 6,
  background: 'rgba(239, 68, 68, 0.12)',
  border: '1px solid rgba(239, 68, 68, 0.35)',
  color: '#ffb4b4',
  fontSize: 13,
};
const modalBack = {
  position: 'fixed', inset: 0, zIndex: 100,
  background: 'rgba(0,0,0,0.6)',
  display: 'flex', alignItems: 'center', justifyContent: 'center',
  padding: 24,
};
const modalCard = {
  width: '100%', maxWidth: 360,
  background: 'var(--bg-elevated, #151526)',
  border: '1px solid var(--border, #26263a)',
  borderRadius: 10,
  padding: 24,
};
