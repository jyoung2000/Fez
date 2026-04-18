import { useEffect, useRef, useCallback, useState } from 'react';
import { openDB } from 'idb';
import useTimelineStore from '../stores/timelineStore';

const DB_NAME = 'clipai-editor';
const DB_VERSION = 1;
const STORE_NAME = 'projects';
const AUTOSAVE_DEBOUNCE_MS = 500;
const SERVER_SYNC_INTERVAL_MS = 60000;

async function getDB() {
  return openDB(DB_NAME, DB_VERSION, {
    upgrade(db) {
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME, { keyPath: 'key' });
      }
    },
  });
}

/**
 * Reconcile media library entries with the backend.
 *
 * After IndexedDB recovery or page reload, some media entries may have stale
 * blob: URLs (which are invalidated when the page unloads). This function
 * fetches the authoritative media list from the backend and:
 * 1. Replaces any stale blob: URLs with the backend URL
 * 2. Adds any backend media that is missing from the local store
 * 3. Leaves entries with valid backend URLs untouched
 *
 * This ensures uploaded media persists across container restarts and browser
 * cache clears — only explicit user deletion removes media.
 */
async function reconcileMediaLibrary(jobId) {
  const updateMedia = useTimelineStore.getState().updateMedia;
  const addMedia = useTimelineStore.getState().addMedia;

  // Fetch both job-specific and global library media
  const urls = ['/api/media/list'];
  if (jobId && jobId !== '_library') {
    urls.push(`/api/media/list?job_id=${jobId}`);
  }

  for (const url of urls) {
    try {
      const res = await fetch(url);
      if (!res.ok) continue;
      const data = await res.json();
      const backendItems = data.items || [];

      const currentLibrary = useTimelineStore.getState().mediaLibrary;
      const currentById = new Map(currentLibrary.map(m => [m.id, m]));

      for (const backendItem of backendItems) {
        const existing = currentById.get(backendItem.id);
        if (existing) {
          // Entry exists locally — update URL if it's a stale blob: URL
          // or if the backend URL has changed
          if (
            existing.url.startsWith('blob:') ||
            (!existing.url.startsWith('/api/') && backendItem.url)
          ) {
            updateMedia(existing.id, { url: backendItem.url });
          }
        } else {
          // Entry missing locally — add from backend
          addMedia({
            id: backendItem.id,
            type: backendItem.type,
            filename: backendItem.filename,
            url: backendItem.url,
            thumbnailUrl: backendItem.type === 'image' ? backendItem.url : '',
            duration: 0,
          });
        }
      }
    } catch {
      // Backend unavailable — skip reconciliation
    }
  }
}

export default function useTimelinePersistence(jobId, clipId) {
  const exportState = useTimelineStore((s) => s.exportState);
  const importState = useTimelineStore((s) => s.importState);
  const tracks = useTimelineStore((s) => s.tracks);
  const items = useTimelineStore((s) => s.items);
  const mediaLibrary = useTimelineStore((s) => s.mediaLibrary);
  const [recovered, setRecovered] = useState(false);

  const saveTimerRef = useRef(null);
  const serverTimerRef = useRef(null);
  const reconcileRef = useRef(false);
  const key = `${jobId || 'unknown'}_${clipId || 'default'}`;

  // ── Load from IndexedDB on mount ──────────────────────────────────────────
  useEffect(() => {
    if (!jobId) return;
    let cancelled = false;
    (async () => {
      try {
        const db = await getDB();
        const saved = await db.get(STORE_NAME, key);
        if (saved && saved.state && !cancelled) {
          const state = saved.state;
          if (Array.isArray(state.items) && state.items.length > 0) {
            importState(state);
            setRecovered(true);
          }
        }
      } catch {
        // IndexedDB unavailable — no recovery
      }

      // Reconcile media URLs with backend after recovery (or on fresh load)
      if (!cancelled && !reconcileRef.current) {
        reconcileRef.current = true;
        await reconcileMediaLibrary(jobId);
      }
    })();
    return () => { cancelled = true; };
  }, [jobId, clipId, key]);

  // ── Auto-save to IndexedDB (debounced) ────────────────────────────────────
  // Triggers on tracks, items, OR mediaLibrary changes so new uploads are
  // persisted immediately (not just on the 60s server sync interval).
  useEffect(() => {
    if (!jobId) return;
    if (saveTimerRef.current) clearTimeout(saveTimerRef.current);
    saveTimerRef.current = setTimeout(async () => {
      try {
        const state = exportState();
        const db = await getDB();
        await db.put(STORE_NAME, {
          key,
          state,
          lastModified: new Date().toISOString(),
        });
      } catch {
        // Silently fail
      }
    }, AUTOSAVE_DEBOUNCE_MS);

    return () => {
      if (saveTimerRef.current) clearTimeout(saveTimerRef.current);
    };
  }, [tracks, items, mediaLibrary, jobId, clipId, key, exportState]);

  // ── Server sync every 60s ─────────────────────────────────────────────────
  const syncToServer = useCallback(async () => {
    if (!jobId || clipId == null) return;
    try {
      const state = exportState();
      await fetch(`/api/jobs/${jobId}/clips/${clipId}/editor-state`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(state),
      });
    } catch {
      // Server sync is best-effort
    }
  }, [jobId, clipId, exportState]);

  useEffect(() => {
    if (!jobId) return;
    serverTimerRef.current = setInterval(syncToServer, SERVER_SYNC_INTERVAL_MS);
    return () => {
      if (serverTimerRef.current) clearInterval(serverTimerRef.current);
    };
  }, [syncToServer, jobId]);

  return { recovered };
}
