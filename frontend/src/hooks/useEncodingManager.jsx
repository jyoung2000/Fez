import React, { createContext, useContext, useState, useCallback, useRef, useEffect } from 'react';
import { sendNotification } from '../utils/notifications';

const EncodingContext = createContext(null);

const LOGS_STORAGE_KEY = 'clipai_activity_logs';
const MAX_LOGS = 2000;

function loadPersistedLogs() {
  try {
    const saved = localStorage.getItem(LOGS_STORAGE_KEY);
    if (saved) {
      const parsed = JSON.parse(saved);
      if (Array.isArray(parsed)) return parsed.slice(0, MAX_LOGS);
    }
  } catch {}
  return [];
}

function persistLogs(logs) {
  try {
    localStorage.setItem(LOGS_STORAGE_KEY, JSON.stringify(logs.slice(0, MAX_LOGS)));
  } catch {}
}

/**
 * Global encoding state manager.
 * Tracks all active clip exports across pages so encoding persists
 * even when the user navigates away from the clips tab.
 *
 * Also maintains a persistent activity log (localStorage-backed) that
 * captures all processing events: exports, analysis progress, errors,
 * cancellations, clip generation, etc.
 */
export function EncodingProvider({ children }) {
  const [tasks, setTasks] = useState({});
  const [logs, setLogs] = useState(loadPersistedLogs);
  const logIdRef = useRef(
    // Resume log IDs from persisted data to avoid collisions
    logs.length > 0 ? Math.max(...logs.map((l) => l.id || 0)) + 1 : 1
  );
  const wsRefs = useRef({});
  // Track which job WebSockets we've opened for global monitoring
  const monitorWsRefs = useRef({});
  // Dedup: track recently downloaded URLs to prevent duplicate browser downloads
  // (multiple WS connections can receive the same export_complete broadcast)
  const recentDownloadsRef = useRef(new Set());
  // Guard: track in-flight export IDs to prevent duplicate export requests
  // (e.g., double-click before React state update disables the button)
  const activeExportsRef = useRef(new Set());

  // Persist logs whenever they change
  useEffect(() => {
    persistLogs(logs);
  }, [logs]);

  const pushLog = useCallback((level, message, extra) => {
    const id = ++logIdRef.current;
    const safeMsg = typeof message === 'string' ? message : String(message ?? '');
    const entry = { id, timestamp: Date.now(), level, message: safeMsg };
    if (extra) entry.extra = extra;
    setLogs((prev) => [entry, ...prev].slice(0, MAX_LOGS));
    return entry;
  }, []);

  const startExport = useCallback((jobId, clipId, clipTitle, exportBody, options) => {
    const exportId = `${jobId}_${clipId}`;

    // Prevent duplicate exports: if this exact export is already in-flight,
    // ignore the second request (protects against double-clicks and race
    // conditions before React state updates disable the button).
    if (activeExportsRef.current.has(exportId)) {
      return exportId;
    }
    activeExportsRef.current.add(exportId);

    const quality = exportBody?.export_quality || '1080p';
    const apiEndpoint = (options && options.endpoint) || `/api/jobs/${jobId}/export-clip`;
    // ``options.downloadUrlRewriter`` lets callers transform the server-
    // reported ``/api/files/...`` URL into a different path before the
    // client attempts to fetch it. Used by public share pages to route
    // the download through the token-gated ``/api/share/public/...``
    // endpoint instead of the auth-gated ``/api/files/...`` route the
    // export task emits by default.
    const downloadUrlRewriter =
      (options && typeof options.downloadUrlRewriter === 'function')
        ? options.downloadUrlRewriter
        : null;

    setTasks((prev) => ({
      ...prev,
      [exportId]: {
        jobId,
        clipId,
        clipTitle: clipTitle || `Clip ${clipId}`,
        status: 'encoding',
        message: 'Starting export...',
        startedAt: Date.now(),
        downloadUrl: null,
        exportQuality: quality,
        downloadUrlRewriter,
      },
    }));

    pushLog('info', `Export started: ${clipTitle || `Clip ${clipId}`} [${quality}]`);

    // Open WebSocket BEFORE sending POST so we don't miss any messages
    // the backend broadcasts immediately after starting the export task.
    _openWs(jobId, clipId, clipTitle);

    console.log('[EncodingManager] Export payload keys:', Object.keys(exportBody || {}));
    console.log('[EncodingManager] Export payload:', JSON.stringify(exportBody, null, 2)?.slice(0, 2000));
    fetch(apiEndpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(exportBody, (_, v) =>
        typeof v === 'number' && !Number.isFinite(v) ? null : v
      ),
    })
      .then(async (res) => {
        if (!res.ok) {
          let detail = '';
          try {
            const body = await res.json();
            detail = JSON.stringify(body.detail || body);
          } catch (_) { /* ignore parse errors */ }
          throw new Error(`Export request failed: ${res.status}${detail ? ` — ${detail}` : ''}`);
        }
      })
      .catch((err) => {
        activeExportsRef.current.delete(exportId);
        setTasks((prev) => ({
          ...prev,
          [exportId]: { ...prev[exportId], status: 'error', message: err.message },
        }));
        pushLog('error', `Export failed for ${clipTitle || `Clip ${clipId}`}: ${err.message}`);
      });

    return exportId;
  }, [pushLog]);

  const _openWs = useCallback((jobId, clipId, clipTitle) => {
    if (wsRefs.current[jobId]) {
      return;
    }

    const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const ws = new WebSocket(`${proto}//${window.location.host}/ws/jobs/${jobId}`);
    wsRefs.current[jobId] = ws;

    ws.onmessage = (evt) => {
      try {
        const msg = JSON.parse(evt.data);
        // Coerce message to string to prevent React error #310
        if (msg.message != null && typeof msg.message !== 'string') msg.message = String(msg.message);
        if (msg.type === 'status' && msg.status === 'exporting') {
          // Update ALL encoding tasks for this job (not just one clip)
          setTasks((prev) => {
            const next = { ...prev };
            let changed = false;
            for (const key of Object.keys(next)) {
              if (key.startsWith(`${jobId}_`) && next[key].status === 'encoding') {
                next[key] = {
                  ...next[key],
                  message: msg.message || 'Encoding...',
                  progress: msg.progress ?? next[key].progress ?? 0,
                };
                changed = true;
              }
            }
            return changed ? next : prev;
          });
          pushLog('info', msg.message || 'Encoding...');
        } else if (msg.type === 'export_complete') {
          const completedExportId = `${jobId}_${msg.clip_id}`;
          activeExportsRef.current.delete(completedExportId);
          setTasks((prev) => {
            const task = prev[completedExportId];
            if (!task) return prev;
            return {
              ...prev,
              [completedExportId]: {
                ...task,
                status: 'complete',
                message: msg.message || 'Export complete',
                // Apply any caller-supplied URL rewriter (e.g. share
                // pages rewriting ``/api/files/...`` → ``/api/share/public/...``).
                // Without this, an unauth'd share recipient's browser
                // would get a 401 when the auto-download fetch fires.
                downloadUrl: (() => {
                  if (!msg.download_url) return null;
                  const rw = prev[completedExportId]?.downloadUrlRewriter;
                  try {
                    return rw ? rw(msg.download_url) : msg.download_url;
                  } catch {
                    return msg.download_url;
                  }
                })(),
              },
            };
          });
          pushLog('success', msg.message || `Clip exported successfully`);
          // ``msg.download_url`` is the raw, server-emitted URL. The
          // auto-download path below uses it directly, so apply the
          // same rewrite there too — otherwise the blob fetch would
          // 401 on public share pages.
          const rewriteForFetch = (url) => {
            if (!url) return url;
            let rw = null;
            setTasks((prev) => {
              rw = prev[completedExportId]?.downloadUrlRewriter || null;
              return prev;
            });
            try {
              return rw ? rw(url) : url;
            } catch {
              return url;
            }
          };
          msg.download_url = rewriteForFetch(msg.download_url);
          // Use the task's stored clipTitle (more reliable than closure-captured value
          // since the WS may have been opened for a different clip on the same job)
          const _completedTitle = (() => {
            // Access tasks via setTasks to get current value
            let t = null;
            setTasks((prev) => { t = prev[completedExportId]; return prev; });
            return t?.clipTitle || clipTitle || `Clip ${msg.clip_id}`;
          })();
          sendNotification('Export Complete', {
            body: `${_completedTitle} is ready for download.`,
            tag: `export-${jobId}-${msg.clip_id}`,
          });
          if (msg.download_url && !recentDownloadsRef.current.has(msg.download_url)) {
            // Dedup: mark this URL so duplicate WS messages don't trigger
            // multiple browser downloads for the same file.
            recentDownloadsRef.current.add(msg.download_url);
            setTimeout(() => recentDownloadsRef.current.delete(msg.download_url), 10000);
            // Auto-download: use a blob fetch so the browser saves
            // immediately without a "Save As" dialog or title prompt.
            // The download attribute with an explicit filename bypasses
            // Content-Disposition: attachment which can trigger save dialogs.
            const title = _completedTitle;
            // Retrieve quality from the task object (stored at startExport time)
            let exportQuality = '1080p';
            setTasks((prev) => {
              const task = prev[completedExportId];
              if (task?.exportQuality) exportQuality = task.exportQuality;
              return prev; // no mutation
            });
            const qualityTag = exportQuality.toUpperCase();
            // Allow brackets in the regex so the [QUALITY] prefix survives
            const safeName = title.replace(/[^a-zA-Z0-9_\-\s().\[\]]/g, '').trim() || 'clip';
            const ext = (msg.download_url.split('.').pop() || 'mp4').split('?')[0];
            const downloadName = `[${qualityTag}] ${safeName}.${ext}`;
            fetch(msg.download_url)
              .then((res) => {
                if (!res.ok) throw new Error(`HTTP ${res.status}`);
                return res.blob();
              })
              .then((blob) => {
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = downloadName;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                setTimeout(() => URL.revokeObjectURL(url), 5000);
                pushLog('info', `Auto-download started: ${downloadName}`);
              })
              .catch((err) => {
                console.warn('[EncodingManager] Blob download failed, using direct link:', err);
                pushLog('warning', `Blob download failed (${err.message}), trying direct link...`);
                // Fallback: direct link click if blob fetch fails
                const a = document.createElement('a');
                a.href = msg.download_url;
                a.download = downloadName;
                a.target = '_blank';
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
              });
          }
          _maybeCloseWs(jobId);
        } else if (msg.type === 'error') {
          // Clear active export guards for all exports on this job
          for (const key of activeExportsRef.current) {
            if (key.startsWith(`${jobId}_`)) activeExportsRef.current.delete(key);
          }
          setTasks((prev) => {
            const next = { ...prev };
            for (const key of Object.keys(next)) {
              if (key.startsWith(`${jobId}_`) && next[key].status === 'encoding') {
                next[key] = { ...next[key], status: 'error', message: msg.message || 'Export failed' };
              }
            }
            return next;
          });
          pushLog('error', msg.message || 'Export error');
          _maybeCloseWs(jobId);
        } else if (msg.type === 'subject_tracking') {
          pushLog(msg.enabled ? 'info' : 'warning', msg.message);
        }
      } catch {}
    };

    ws.onerror = () => {
      pushLog('error', `WebSocket error for job ${jobId}`);
    };

    ws.onclose = () => {
      delete wsRefs.current[jobId];
    };
  }, [pushLog]);

  const _maybeCloseWs = useCallback((jobId) => {
    setTimeout(() => {
      setTasks((prev) => {
        const hasActive = Object.keys(prev).some(
          (k) => k.startsWith(`${jobId}_`) && prev[k].status === 'encoding'
        );
        if (!hasActive && wsRefs.current[jobId]) {
          wsRefs.current[jobId].close();
          delete wsRefs.current[jobId];
        }
        return prev;
      });
    }, 1000);
  }, []);

  // --- Global job monitor ---
  // Connects to active/running jobs to capture all processing logs
  // (analysis, transcription, scene detection, clip detection, etc.)
  const monitorJob = useCallback((jobId, jobFilename) => {
    if (monitorWsRefs.current[jobId]) return;

    const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const ws = new WebSocket(`${proto}//${window.location.host}/ws/jobs/${jobId}`);
    monitorWsRefs.current[jobId] = ws;

    const label = jobFilename || jobId.slice(0, 8);

    ws.onmessage = (evt) => {
      try {
        const msg = JSON.parse(evt.data);
        if (msg.message != null && typeof msg.message !== 'string') msg.message = String(msg.message);
        if (msg.type === 'status') {
          const level = msg.status === 'failed' ? 'error' : 'status';
          pushLog(level, `[${label}] ${msg.message || `Status: ${msg.status}`}`, {
            jobId, status: msg.status, progress: msg.progress,
          });
        } else if (msg.type === 'complete') {
          pushLog('success', `[${label}] ${msg.message || 'Processing complete'}`, { jobId });
          // Close after a short delay
          setTimeout(() => {
            if (monitorWsRefs.current[jobId]) {
              monitorWsRefs.current[jobId].close();
              delete monitorWsRefs.current[jobId];
            }
          }, 2000);
        } else if (msg.type === 'fallback') {
          pushLog('warning', `[${label}] Provider fallback: ${msg.from_provider} -> ${msg.to_provider} (${msg.reason})`, { jobId });
        } else if (msg.type === 'clips_generated') {
          pushLog('success', `[${label}] ${msg.message || `Generated ${msg.count} clips`}`, { jobId });
        } else if (msg.type === 'cancelled') {
          pushLog('warning', `[${label}] Job cancelled`, { jobId });
        } else if (msg.type === 'error') {
          pushLog('error', `[${label}] ${msg.message || 'Error'}`, { jobId });
        } else if (msg.type === 'export_complete') {
          pushLog('success', `[${label}] ${msg.message || 'Export complete'}`, { jobId });
        } else if (msg.type === 'subject_tracking') {
          pushLog(msg.enabled ? 'info' : 'warning', `[${label}] ${msg.message}`, { jobId });
        }
      } catch {}
    };

    ws.onclose = () => {
      delete monitorWsRefs.current[jobId];
    };
  }, [pushLog]);

  const stopMonitorJob = useCallback((jobId) => {
    const ws = monitorWsRefs.current[jobId];
    if (ws) {
      ws.close();
      delete monitorWsRefs.current[jobId];
    }
  }, []);

  const cancelExport = useCallback(async (exportId) => {
    const task = tasks[exportId];
    if (!task || task.status !== 'encoding') return;

    try {
      const res = await fetch(`/api/jobs/${task.jobId}/cancel-export/${task.clipId}`, {
        method: 'POST',
      });
      if (res.ok) {
        activeExportsRef.current.delete(exportId);
        setTasks((prev) => ({
          ...prev,
          [exportId]: { ...prev[exportId], status: 'cancelled', message: 'Cancelled by user' },
        }));
        pushLog('warning', `Export cancelled: ${task.clipTitle}`);
      }
    } catch (err) {
      pushLog('error', `Failed to cancel: ${err.message}`);
    }
  }, [tasks, pushLog]);

  const clearTask = useCallback((exportId) => {
    setTasks((prev) => {
      const next = { ...prev };
      delete next[exportId];
      return next;
    });
  }, []);

  const clearLogs = useCallback(() => {
    setLogs([]);
    persistLogs([]);
  }, []);

  const exportLogsAsFile = useCallback(() => {
    const lines = logs.slice().reverse().map((entry) => {
      const d = new Date(entry.timestamp);
      const ts = d.toLocaleString();
      const level = (entry.level || 'info').toUpperCase().padEnd(7);
      return `[${ts}] ${level} ${entry.message}`;
    });
    const content = lines.join('\n');
    const blob = new Blob([content], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `clipai-logs-${new Date().toISOString().slice(0, 10)}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }, [logs]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      Object.values(wsRefs.current).forEach((ws) => {
        try { ws.close(); } catch {}
      });
      Object.values(monitorWsRefs.current).forEach((ws) => {
        try { ws.close(); } catch {}
      });
    };
  }, []);

  const activeCount = Object.values(tasks).filter((t) => t.status === 'encoding').length;

  // Latest activity message for the global status indicator
  const latestActivity = logs.length > 0 ? logs[0] : null;

  // Count of monitored (processing) jobs
  const monitoredJobCount = Object.keys(monitorWsRefs.current).length;

  return (
    <EncodingContext.Provider value={{
      tasks,
      logs,
      activeCount,
      latestActivity,
      monitoredJobCount,
      startExport,
      cancelExport,
      clearTask,
      clearLogs,
      pushLog,
      exportLogsAsFile,
      monitorJob,
      stopMonitorJob,
    }}>
      {children}
    </EncodingContext.Provider>
  );
}

export default function useEncodingManager() {
  const ctx = useContext(EncodingContext);
  if (!ctx) throw new Error('useEncodingManager must be used within EncodingProvider');
  return ctx;
}
