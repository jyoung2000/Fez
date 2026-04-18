/**
 * Cloud sync for localStorage — persists UI state to the server so
 * settings stay consistent across browsers.
 *
 * How it works:
 * 1. On app start, pullFromCloud() fetches server state → writes to localStorage
 * 2. localStorage.setItem / removeItem are monkey-patched so that changes
 *    to synced keys automatically trigger a debounced push to the server
 * 3. The server is the source of truth — pull overwrites local values
 */

/** Keys that are synced verbatim */
const SYNCED_KEYS = [
  'clipai_clip_settings',
  'clipai-theme',
  'clipai_seo_layout',
  'clipai_viral_clip_overrides',
];

/** Prefix for per-clip segment keys: clipai_segments_{jobId}_{clipId} */
const SEGMENT_PREFIX = 'clipai_segments_';

/** Activity logs are large and ephemeral — skip syncing */
const SKIP_KEYS = ['clipai_activity_logs'];

function isSyncedKey(key) {
  if (SKIP_KEYS.includes(key)) return false;
  return SYNCED_KEYS.includes(key) || key.startsWith(SEGMENT_PREFIX);
}

// ── Debounced push ──────────────────────────────────────────

let _pushTimer = null;
const PUSH_DEBOUNCE_MS = 1500;

function gatherSyncedState() {
  const state = {};
  for (let i = 0; i < localStorage.length; i++) {
    const key = localStorage.key(i);
    if (!isSyncedKey(key)) continue;
    const raw = _originalGetItem(key);
    if (raw == null) continue;
    try {
      state[key] = JSON.parse(raw);
    } catch {
      state[key] = raw;
    }
  }
  return state;
}

function schedulePush() {
  clearTimeout(_pushTimer);
  _pushTimer = setTimeout(async () => {
    try {
      const state = gatherSyncedState();
      await fetch('/api/ui-state', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(state),
      });
    } catch (e) {
      console.warn('[cloudSync] push failed:', e);
    }
  }, PUSH_DEBOUNCE_MS);
}

// ── Pull from server ────────────────────────────────────────

/** Fetch server state and write into localStorage. Returns true on success. */
export async function pullFromCloud() {
  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 5000);
    const res = await fetch('/api/ui-state', { signal: controller.signal });
    clearTimeout(timeout);
    if (!res.ok) return false;
    const state = await res.json();
    if (!state || typeof state !== 'object') return false;

    for (const [key, value] of Object.entries(state)) {
      if (!isSyncedKey(key)) continue;
      const serialized = typeof value === 'string' ? value : JSON.stringify(value);
      _originalSetItem(key, serialized);
    }

    // Apply theme immediately to prevent flash
    const theme = state['clipai-theme'];
    if (theme === 'dark') {
      document.documentElement.dataset.theme = 'dark';
    } else if (theme === 'light' || theme === '') {
      delete document.documentElement.dataset.theme;
    }

    return true;
  } catch (e) {
    console.warn('[cloudSync] pull failed:', e);
    return false;
  }
}

// ── Monkey-patch localStorage ───────────────────────────────

const _originalSetItem = localStorage.setItem.bind(localStorage);
const _originalGetItem = localStorage.getItem.bind(localStorage);
const _originalRemoveItem = localStorage.removeItem.bind(localStorage);

localStorage.setItem = (key, value) => {
  _originalSetItem(key, value);
  if (isSyncedKey(key)) schedulePush();
};

localStorage.removeItem = (key) => {
  _originalRemoveItem(key);
  if (isSyncedKey(key)) schedulePush();
};
