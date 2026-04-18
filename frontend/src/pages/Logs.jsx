import React, { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import useEncodingManager from '../hooks/useEncodingManager';
import useResponsive from '../hooks/useResponsive';
import ProgressBar from '../components/ProgressBar';
import { useAuth } from '../auth/AuthContext';

function formatTime(ts) {
  if (!ts) return '';
  const d = new Date(ts);
  return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

function formatDateTime(ts) {
  if (!ts) return '';
  const d = new Date(ts);
  return d.toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
}

function formatDuration(seconds) {
  if (!seconds && seconds !== 0) return '-';
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, '0')}`;
}

function elapsed(startedAt) {
  if (!startedAt) return '';
  const secs = Math.floor((Date.now() - startedAt) / 1000);
  if (secs < 60) return `${secs}s`;
  return `${Math.floor(secs / 60)}m ${secs % 60}s`;
}

const LOG_LEVEL_COLORS = {
  info: 'var(--accent-cyan)',
  success: 'var(--success)',
  warning: 'var(--accent-amber)',
  error: 'var(--danger)',
  status: 'var(--text-secondary)',
};

const LOG_LEVEL_LABELS = {
  info: 'INFO',
  success: 'OK',
  warning: 'WARN',
  error: 'ERR',
  status: 'STATUS',
};

const STATUS_LABELS = {
  encoding: { label: 'Encoding', color: 'var(--accent-amber)' },
  complete: { label: 'Complete', color: 'var(--success)' },
  error: { label: 'Failed', color: 'var(--danger)' },
  cancelled: { label: 'Cancelled', color: 'var(--text-muted)' },
};

const selectStyle = {
  padding: '6px 10px', borderRadius: 'var(--radius-sm)', fontSize: 12,
  background: 'var(--bg-elevated)', border: '1px solid var(--border)',
  color: 'var(--text-primary)',
};

export default function Logs() {
  const { isMobile } = useResponsive();
  const { user } = useAuth();
  const {
    tasks, logs, activeCount, cancelExport, clearTask, clearLogs,
    pushLog, exportLogsAsFile, monitorJob, startExport,
  } = useEncodingManager();

  const [exportedClips, setExportedClips] = useState([]);
  const [loadingExports, setLoadingExports] = useState(true);
  const [exportSearch, setExportSearch] = useState('');
  const [exportSourceFilter, setExportSourceFilter] = useState('all');
  const [exportSort, setExportSort] = useState('recent'); // recent | oldest | duration_long | duration_short | title_az | title_za
  const [logsTab, setLogsTab] = useState('encoding');
  const [logFilter, setLogFilter] = useState('all'); // all | info | success | warning | error | status
  const [logSearch, setLogSearch] = useState('');
  const [elapsedTick, setElapsedTick] = useState(0);
  const [qualityMenuOpen, setQualityMenuOpen] = useState(null); // index of open quality menu
  const [allocation, setAllocation] = useState({ active_jobs: [], resources: {} });
  const [allocationLoading, setAllocationLoading] = useState(false);
  const [forceActioning, setForceActioning] = useState({});
  const [editingTitle, setEditingTitle] = useState(null); // index of clip being edited
  const [editTitleValue, setEditTitleValue] = useState('');
  const titleInputRef = useRef(null);
  // Backend-tracked exports that aren't in the frontend tasks state
  // (e.g. after a page refresh or when navigating from another tab)
  const [backendExports, setBackendExports] = useState([]);

  const fetchAllocation = useCallback(async () => {
    setAllocationLoading(true);
    try {
      const res = await fetch('/api/allocation');
      if (res.ok) setAllocation(await res.json());
    } catch {}
    setAllocationLoading(false);
  }, []);

  const handleForceStop = async (job) => {
    const key = job.export_key || job.job_id;
    setForceActioning((p) => ({ ...p, [key]: 'stopping' }));
    try {
      if (job.type === 'export' && job.clip_id != null) {
        await fetch(`/api/jobs/${job.job_id}/cancel-export/${job.clip_id}`, { method: 'POST' });
      } else {
        await fetch(`/api/jobs/${job.job_id}/cancel`, { method: 'POST' });
      }
      await fetchAllocation();
    } catch {}
    setForceActioning((p) => ({ ...p, [key]: null }));
  };

  const handleForceFail = async (job) => {
    const key = job.export_key || job.job_id;
    setForceActioning((p) => ({ ...p, [key]: 'failing' }));
    try {
      await fetch(`/api/jobs/${job.job_id}/force-fail`, { method: 'POST' });
      await fetchAllocation();
    } catch {}
    setForceActioning((p) => ({ ...p, [key]: null }));
  };

  // Tick to update elapsed time displays
  useEffect(() => {
    const interval = setInterval(() => setElapsedTick((t) => t + 1), 1000);
    return () => clearInterval(interval);
  }, []);

  // Recover active exports from backend that aren't in frontend state.
  // This covers page refreshes and cross-tab navigation.
  useEffect(() => {
    let cancelled = false;
    const fetchBackendExports = async () => {
      try {
        const res = await fetch('/api/allocation');
        if (!res.ok || cancelled) return;
        const data = await res.json();
        const activeExports = (data.active_jobs || []).filter((j) => j.type === 'export' && j.status === 'encoding');
        // Only keep exports that the frontend doesn't already track
        const untracked = activeExports.filter((exp) => {
          const key = `${exp.job_id}_${exp.clip_id || exp.filename?.replace('Clip ', '')}`;
          return !tasks[key];
        });
        if (!cancelled) setBackendExports(untracked);
      } catch {}
    };
    fetchBackendExports();
    const interval = setInterval(fetchBackendExports, 5000);
    return () => { cancelled = true; clearInterval(interval); };
  }, [tasks]);

  // Fetch all exported clips from all jobs the current user can see.
  // Filter the merged list down to clips THIS USER actually exported
  // (matched on ``exported_by_user_id`` stamped at export time). Admins
  // see every export across every job so they can audit / clean up
  // storage. Legacy exports without an ``exported_by_user_id`` field
  // stay visible to the job owner only; and exports whose stamped UUID
  // no longer resolves (e.g. the auth store was wiped before ``/data/auth``
  // was persisted as a docker volume) fall back to matching on
  // ``exported_by_username`` so a user signing in with the same username
  // still sees their history.
  const fetchExportedClips = useCallback(async () => {
    try {
      const res = await fetch('/api/jobs');
      if (!res.ok) return;
      const jobList = await res.json();
      const completed = jobList.filter((j) => j.status === 'complete');

      const allExported = [];
      const myUserId = user?.id || '';
      const myUsername = (user?.username || '').toLowerCase();
      const isAdmin = user?.role === 'admin';
      for (const j of completed) {
        const detail = await fetch(`/api/jobs/${j.job_id}`).then((r) => r.ok ? r.json() : null).catch(() => null);
        if (!detail?.exported_clips?.length) continue;
        const isJobOwner = !!myUserId && detail.owner_user_id === myUserId;
        for (const ec of detail.exported_clips) {
          const exportedBy = ec.exported_by_user_id || '';
          const exportedByName = (ec.exported_by_username || '').toLowerCase();
          // Show this export when:
          //   - The current user actually exported it (uid match), OR
          //   - Their username matches the export stamp (survives a
          //     container restart that regenerated the UUID because
          //     ``/data/auth`` wasn't persisted before the fix), OR
          //   - The current user is an admin (can audit every export), OR
          //   - It was exported before per-user attribution existed
          //     (legacy entry, no ``exported_by_user_id``) AND the
          //     current user owns the parent job — that user is the
          //     only sensible default audience.
          if (
            (exportedBy && exportedBy === myUserId) ||
            (exportedByName && exportedByName === myUsername) ||
            isAdmin ||
            (!exportedBy && isJobOwner)
          ) {
            allExported.push({
              ...ec,
              jobId: detail.job_id,
              sourceFilename: detail.filename,
            });
          }
        }
      }
      setExportedClips(allExported);
    } catch {
    } finally {
      setLoadingExports(false);
    }
  }, [user?.id, user?.username, user?.role]);

  useEffect(() => { fetchExportedClips(); }, [fetchExportedClips]);

  // Re-export an existing clip at a different quality
  const handleReExport = (ec, quality) => {
    setQualityMenuOpen(null);
    const body = {
      start: ec.start,
      end: ec.end,
      clip_id: ec.clip_id,
      clip_title: ec.title || `Clip ${ec.clip_id}`,
      export_quality: quality,
    };
    if (ec.aspect_ratio) body.aspect_ratio = ec.aspect_ratio;
    body.subtitles_enabled = ec.subtitles_enabled || false;
    if (ec.subtitles_enabled && ec.subtitle_settings) {
      body.subtitle_settings = ec.subtitle_settings;
    }
    startExport(ec.jobId, ec.clip_id, ec.title || `Clip ${ec.clip_id}`, body);
    pushLog('info', `Re-exporting "${ec.title || `Clip ${ec.clip_id}`}" at ${quality}`);
  };

  // Save edited clip title to backend
  const handleSaveTitle = async (ec) => {
    const newTitle = editTitleValue.trim();
    if (!newTitle || newTitle === (ec.title || ec.filename)) {
      setEditingTitle(null);
      return;
    }
    try {
      const res = await fetch(`/api/jobs/${ec.jobId}/clips/${ec.clip_id}/title`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title: newTitle }),
      });
      if (res.ok) {
        setExportedClips((prev) =>
          prev.map((clip) =>
            clip.jobId === ec.jobId && clip.clip_id === ec.clip_id
              ? { ...clip, title: newTitle }
              : clip
          )
        );
        pushLog('success', `Title updated to "${newTitle}"`);
      } else {
        pushLog('error', 'Failed to update title');
      }
    } catch (err) {
      pushLog('error', `Error updating title: ${err.message}`);
    }
    setEditingTitle(null);
  };

  // Monitor active/running jobs for comprehensive log capture
  useEffect(() => {
    let cancelled = false;
    async function monitorActiveJobs() {
      try {
        const res = await fetch('/api/jobs');
        if (!res.ok || cancelled) return;
        const jobList = await res.json();
        const running = jobList.filter((j) =>
          j.status !== 'complete' && j.status !== 'failed' && j.status !== 'cancelled' && j.status !== 'queued'
        );
        running.forEach((j) => monitorJob(j.job_id, j.filename));
      } catch {}
    }
    monitorActiveJobs();
    return () => { cancelled = true; };
  }, [monitorJob]);

  // Re-fetch exports when a task completes
  const completedCount = Object.values(tasks).filter((t) => t.status === 'complete').length;
  useEffect(() => {
    if (completedCount > 0) fetchExportedClips();
  }, [completedCount, fetchExportedClips]);

  // Derive unique source filenames for filter
  const sourceFiles = useMemo(() => {
    return [...new Set(exportedClips.map((ec) => ec.sourceFilename).filter(Boolean))].sort();
  }, [exportedClips]);

  // Filter, search, and sort exported clips
  const filteredExports = useMemo(() => {
    let result = [...exportedClips];

    // Source filter
    if (exportSourceFilter !== 'all') {
      result = result.filter((ec) => ec.sourceFilename === exportSourceFilter);
    }

    // Text search
    if (exportSearch.trim()) {
      const q = exportSearch.trim().toLowerCase();
      result = result.filter((ec) =>
        (ec.filename || '').toLowerCase().includes(q) ||
        (ec.title || '').toLowerCase().includes(q) ||
        (ec.sourceFilename || '').toLowerCase().includes(q)
      );
    }

    // Sort
    result.sort((a, b) => {
      switch (exportSort) {
        case 'recent':
          return (b.exported_at || '').localeCompare(a.exported_at || '');
        case 'oldest':
          return (a.exported_at || '').localeCompare(b.exported_at || '');
        case 'duration_long': {
          const durA = a.duration ?? (a.end - a.start) ?? 0;
          const durB = b.duration ?? (b.end - b.start) ?? 0;
          return durB - durA;
        }
        case 'duration_short': {
          const durA2 = a.duration ?? (a.end - a.start) ?? 0;
          const durB2 = b.duration ?? (b.end - b.start) ?? 0;
          return durA2 - durB2;
        }
        case 'title_az':
          return (a.title || a.filename || '').localeCompare(b.title || b.filename || '');
        case 'title_za':
          return (b.title || b.filename || '').localeCompare(a.title || a.filename || '');
        case 'source_az':
          return (a.sourceFilename || '').localeCompare(b.sourceFilename || '');
        default:
          return 0;
      }
    });

    return result;
  }, [exportedClips, exportSourceFilter, exportSearch, exportSort]);

  // Filter and search activity logs
  const filteredLogs = useMemo(() => {
    let result = [...logs];
    if (logFilter !== 'all') {
      result = result.filter((entry) => entry.level === logFilter);
    }
    if (logSearch.trim()) {
      const q = logSearch.trim().toLowerCase();
      result = result.filter((entry) => (entry.message || '').toLowerCase().includes(q));
    }
    return result;
  }, [logs, logFilter, logSearch]);

  const taskList = Object.entries(tasks);
  const activeTaskList = taskList.filter(([, t]) => t.status === 'encoding');
  const doneTaskList = taskList.filter(([, t]) => t.status !== 'encoding');
  // Total active count includes backend-recovered exports
  const totalActiveCount = activeTaskList.length + backendExports.length;

  const tabStyle = (active) => ({
    padding: '8px 16px',
    fontSize: 13,
    fontWeight: active ? 600 : 400,
    color: active ? 'var(--accent-cyan)' : 'var(--text-secondary)',
    background: active ? 'var(--accent-cyan-dim)' : 'transparent',
    border: `1px solid ${active ? 'var(--accent-cyan)' : 'var(--border)'}`,
    borderRadius: 'var(--radius-sm)',
    cursor: 'pointer',
  });

  return (
    <div>
      <h2 style={{ fontSize: 20, marginBottom: 16 }}>Logs & Exports</h2>

      {/* Sub-tabs */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 20, flexWrap: 'wrap' }}>
        <button onClick={() => setLogsTab('encoding')} style={tabStyle(logsTab === 'encoding')}>
          Encoding
          {totalActiveCount > 0 && (
            <span style={{
              marginLeft: 6, fontSize: 10, fontFamily: 'var(--font-mono)',
              background: 'var(--accent-amber)', color: 'var(--bg-base)',
              padding: '1px 6px', borderRadius: 8, fontWeight: 700,
            }}>
              {totalActiveCount}
            </span>
          )}
        </button>
        <button onClick={() => setLogsTab('exports')} style={tabStyle(logsTab === 'exports')}>
          Exported Clips
          {exportedClips.length > 0 && (
            <span style={{
              marginLeft: 6, fontSize: 10, fontFamily: 'var(--font-mono)',
              color: 'var(--text-muted)',
            }}>
              ({exportedClips.length})
            </span>
          )}
        </button>
        <button onClick={() => setLogsTab('logs')} style={tabStyle(logsTab === 'logs')}>
          Activity Log
          {logs.length > 0 && (
            <span style={{
              marginLeft: 6, fontSize: 10, fontFamily: 'var(--font-mono)',
              color: 'var(--text-muted)',
            }}>
              ({logs.length})
            </span>
          )}
        </button>
        <button onClick={() => { setLogsTab('allocation'); fetchAllocation(); }} style={tabStyle(logsTab === 'allocation')}>
          Allocation
          {allocation.active_jobs?.length > 0 && (
            <span style={{
              marginLeft: 6, fontSize: 10, fontFamily: 'var(--font-mono)',
              background: 'var(--danger)', color: 'var(--bg-base)',
              padding: '1px 6px', borderRadius: 8, fontWeight: 700,
            }}>
              {allocation.active_jobs.length}
            </span>
          )}
        </button>
      </div>

      {/* ═══ Encoding Tab ═══ */}
      {logsTab === 'encoding' && (
        <div>
          {/* Active Exports */}
          {totalActiveCount > 0 && (
            <div style={{ marginBottom: 24 }}>
              <h3 style={{ fontSize: 13, fontFamily: 'var(--font-mono)', color: 'var(--accent-amber)', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 12 }}>
                Currently Encoding ({totalActiveCount})
              </h3>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                {/* Frontend-tracked active exports (full progress info) */}
                {activeTaskList.map(([id, task]) => (
                  <div
                    key={id}
                    style={{
                      background: 'var(--bg-panel)',
                      border: '1px solid var(--accent-amber)',
                      borderRadius: 'var(--radius-md)',
                      padding: 14,
                      display: 'flex',
                      alignItems: 'center',
                      gap: 12,
                      flexWrap: 'wrap',
                    }}
                  >
                    <div style={{
                      width: 10, height: 10, borderRadius: '50%',
                      background: 'var(--accent-amber)',
                      animation: 'pulse 1.5s ease-in-out infinite',
                      flexShrink: 0,
                    }} />
                    <div style={{ flex: 1, minWidth: 0 }}>
                      <div style={{ fontSize: 14, fontWeight: 600, marginBottom: 2, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', display: 'flex', alignItems: 'center', gap: 6 }}>
                        {task.clipTitle}
                        {task.exportQuality && (
                          <span style={{
                            fontSize: 9, fontWeight: 700, padding: '1px 5px',
                            background: 'var(--accent-amber)', color: 'var(--bg-base)',
                            borderRadius: 3, fontFamily: 'var(--font-mono)',
                          }}>
                            {task.exportQuality.toUpperCase()}
                          </span>
                        )}
                      </div>
                      <div style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
                        {typeof task.message === 'string' ? task.message : String(task.message ?? '')}
                      </div>
                      {task.progress != null && task.status === 'encoding' && (
                        <div style={{ marginTop: 4 }}>
                          <ProgressBar progress={task.progress} variant="amber" />
                        </div>
                      )}
                    </div>
                    <div style={{ fontSize: 11, fontFamily: 'var(--font-mono)', color: 'var(--text-muted)', flexShrink: 0 }}>
                      {elapsed(task.startedAt)}
                    </div>
                    <button
                      onClick={() => cancelExport(id)}
                      style={{
                        padding: '6px 14px',
                        background: 'var(--danger)',
                        color: '#fff',
                        border: 'none',
                        borderRadius: 'var(--radius-sm)',
                        fontSize: 11,
                        fontWeight: 600,
                        cursor: 'pointer',
                        flexShrink: 0,
                      }}
                    >
                      Cancel
                    </button>
                  </div>
                ))}
                {/* Backend-recovered exports (not tracked by frontend state) */}
                {backendExports.map((exp) => (
                  <div
                    key={`backend-${exp.job_id}-${exp.filename}`}
                    style={{
                      background: 'var(--bg-panel)',
                      border: '1px solid var(--accent-amber)',
                      borderRadius: 'var(--radius-md)',
                      padding: 14,
                      display: 'flex',
                      alignItems: 'center',
                      gap: 12,
                      flexWrap: 'wrap',
                    }}
                  >
                    <div style={{
                      width: 10, height: 10, borderRadius: '50%',
                      background: 'var(--accent-amber)',
                      animation: 'pulse 1.5s ease-in-out infinite',
                      flexShrink: 0,
                    }} />
                    <div style={{ flex: 1, minWidth: 0 }}>
                      <div style={{ fontSize: 14, fontWeight: 600, marginBottom: 2, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                        {exp.filename || `Export (${exp.job_id.slice(0, 8)})`}
                      </div>
                      <div style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
                        {exp.progress_message || 'Encoding...'}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Completed/Failed Tasks */}
          {doneTaskList.length > 0 && (
            <div>
              <h3 style={{ fontSize: 13, fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 12 }}>
                Recent Tasks
              </h3>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                {doneTaskList.map(([id, task]) => {
                  const st = STATUS_LABELS[task.status] || STATUS_LABELS.error;
                  return (
                    <div
                      key={id}
                      style={{
                        background: 'var(--bg-panel)',
                        border: '1px solid var(--border)',
                        borderRadius: 'var(--radius-md)',
                        padding: '10px 14px',
                        display: 'flex',
                        alignItems: 'center',
                        gap: 12,
                        flexWrap: 'wrap',
                      }}
                    >
                      <span style={{
                        fontSize: 9, fontFamily: 'var(--font-mono)', textTransform: 'uppercase',
                        letterSpacing: '0.05em', color: st.color, fontWeight: 700,
                        background: `${st.color}15`, padding: '2px 8px', borderRadius: 3,
                      }}>
                        {st.label}
                      </span>
                      <div style={{ flex: 1, minWidth: 0 }}>
                        <span style={{ fontSize: 13, fontWeight: 500 }}>{task.clipTitle}</span>
                        {task.exportQuality && (
                          <span style={{
                            fontSize: 9, fontWeight: 700, padding: '1px 5px', marginLeft: 6,
                            background: 'var(--accent-cyan-dim)', color: 'var(--accent-cyan)',
                            borderRadius: 3, fontFamily: 'var(--font-mono)',
                          }}>
                            {task.exportQuality.toUpperCase()}
                          </span>
                        )}
                        <span style={{ fontSize: 11, color: 'var(--text-muted)', marginLeft: 8 }}>{typeof task.message === 'string' ? task.message : String(task.message ?? '')}</span>
                        {task.progress != null && task.status === 'encoding' && (
                          <div style={{ marginTop: 4 }}>
                            <ProgressBar progress={task.progress} variant="amber" />
                          </div>
                        )}
                      </div>
                      {task.downloadUrl && (
                        <a
                          href={task.downloadUrl}
                          download
                          style={{
                            padding: '4px 12px',
                            background: 'var(--accent-cyan-dim)',
                            color: 'var(--accent-cyan)',
                            border: '1px solid var(--accent-cyan)',
                            borderRadius: 'var(--radius-sm)',
                            fontSize: 11,
                            fontWeight: 600,
                            textDecoration: 'none',
                            flexShrink: 0,
                          }}
                        >
                          Download
                        </a>
                      )}
                      <button
                        onClick={() => clearTask(id)}
                        style={{
                          background: 'none', border: 'none',
                          color: 'var(--text-muted)', cursor: 'pointer',
                          fontSize: 14, padding: 4,
                        }}
                        title="Dismiss"
                      >
                        &times;
                      </button>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {taskList.length === 0 && backendExports.length === 0 && (
            <div style={{ textAlign: 'center', padding: '60px 24px', color: 'var(--text-muted)' }}>
              <div style={{ fontSize: 36, marginBottom: 12, opacity: 0.3 }}>&#x1F3AC;</div>
              <p>No encoding tasks. Export a clip from the Clips tab to see progress here.</p>
            </div>
          )}
        </div>
      )}

      {/* ═══ Exported Clips Tab ═══ */}
      {logsTab === 'exports' && (
        <div>
          {/* Search, Filter & Sort Bar */}
          <div style={{
            display: 'flex', gap: 10, marginBottom: 16, flexWrap: 'wrap', alignItems: 'center',
            padding: '10px 14px', background: 'var(--bg-panel)', border: '1px solid var(--border)',
            borderRadius: 'var(--radius-md)',
          }}>
            {/* Search */}
            <div style={{ position: 'relative', flex: 1, minWidth: 180 }}>
              <div style={{
                position: 'absolute', left: 10, top: '50%', transform: 'translateY(-50%)',
                color: 'var(--text-muted)', fontSize: 14, pointerEvents: 'none',
              }}>
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <circle cx="11" cy="11" r="8" /><line x1="21" y1="21" x2="16.65" y2="16.65" />
                </svg>
              </div>
              <input
                type="text"
                value={exportSearch}
                onChange={(e) => setExportSearch(e.target.value)}
                placeholder="Search by title, filename, source..."
                style={{
                  width: '100%',
                  padding: '6px 10px 6px 30px',
                  fontSize: 12,
                  background: 'var(--bg-elevated)',
                  border: '1px solid var(--border)',
                  borderRadius: 'var(--radius-sm)',
                  color: 'var(--text-primary)',
                  outline: 'none',
                }}
              />
              {exportSearch && (
                <button
                  onClick={() => setExportSearch('')}
                  style={{
                    position: 'absolute', right: 6, top: '50%', transform: 'translateY(-50%)',
                    background: 'var(--bg-elevated)', border: 'none', borderRadius: '50%',
                    width: 16, height: 16, display: 'flex', alignItems: 'center', justifyContent: 'center',
                    color: 'var(--text-muted)', fontSize: 10, cursor: 'pointer', lineHeight: 1,
                  }}
                >
                  &times;
                </button>
              )}
            </div>

            {/* Source filter */}
            {sourceFiles.length > 1 && (
              <select
                value={exportSourceFilter}
                onChange={(e) => setExportSourceFilter(e.target.value)}
                style={{ ...selectStyle, maxWidth: 180 }}
              >
                <option value="all">All Videos</option>
                {sourceFiles.map((f) => (
                  <option key={f} value={f}>{f.length > 25 ? f.slice(0, 22) + '...' : f}</option>
                ))}
              </select>
            )}

            {/* Sort */}
            <select
              value={exportSort}
              onChange={(e) => setExportSort(e.target.value)}
              style={selectStyle}
            >
              <option value="recent">Newest First</option>
              <option value="oldest">Oldest First</option>
              <option value="duration_long">Duration: Longest</option>
              <option value="duration_short">Duration: Shortest</option>
              <option value="title_az">Title: A-Z</option>
              <option value="title_za">Title: Z-A</option>
              <option value="source_az">Source: A-Z</option>
            </select>

            {/* Count */}
            <span style={{ fontSize: 11, fontFamily: 'var(--font-mono)', color: 'var(--text-muted)', whiteSpace: 'nowrap' }}>
              {filteredExports.length} of {exportedClips.length}
            </span>
          </div>

          {/* Exported clips list */}
          {loadingExports ? (
            <div style={{ textAlign: 'center', padding: 48, color: 'var(--text-secondary)' }}>
              <div style={{
                width: 24, height: 24, border: '2px solid var(--border)', borderTopColor: 'var(--accent-cyan)',
                borderRadius: '50%', animation: 'spin 0.8s linear infinite',
                margin: '0 auto 12px',
              }} />
              Loading exported clips...
            </div>
          ) : filteredExports.length > 0 ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              {filteredExports.map((ec, i) => {
                const dur = ec.duration ?? (ec.end != null && ec.start != null ? ec.end - ec.start : null);
                return (
                  <div
                    key={`${ec.jobId}-${ec.clip_id}-${i}`}
                    style={{
                      background: 'var(--bg-panel)',
                      border: '1px solid var(--border)',
                      borderRadius: 'var(--radius-md)',
                      padding: '12px 16px',
                      display: 'flex',
                      alignItems: 'center',
                      gap: 12,
                      flexWrap: 'wrap',
                    }}
                  >
                    <div style={{
                      width: 36, height: 36, borderRadius: 'var(--radius-sm)',
                      background: 'var(--accent-cyan-dim)',
                      display: 'flex', alignItems: 'center', justifyContent: 'center',
                      color: 'var(--accent-cyan)', fontSize: 11, fontWeight: 700,
                      fontFamily: 'var(--font-mono)', flexShrink: 0,
                    }}>
                      {ec.is_full_video ? 'FULL' : 'MP4'}
                    </div>
                    <div style={{ flex: 1, minWidth: 0 }}>
                      {editingTitle === i ? (
                        <div style={{ display: 'flex', gap: 6, alignItems: 'center', marginBottom: 2 }}>
                          <input
                            ref={titleInputRef}
                            type="text"
                            value={editTitleValue}
                            onChange={(e) => setEditTitleValue(e.target.value)}
                            onKeyDown={(e) => {
                              if (e.key === 'Enter') handleSaveTitle(ec);
                              if (e.key === 'Escape') setEditingTitle(null);
                            }}
                            autoFocus
                            style={{
                              flex: 1, padding: '3px 6px', fontSize: 13, fontWeight: 600,
                              background: 'var(--bg-elevated)', color: 'var(--text-primary)',
                              border: '1px solid var(--accent-cyan)', borderRadius: 'var(--radius-sm)',
                              outline: 'none', lineHeight: 1.3, minWidth: 0,
                            }}
                          />
                          <button
                            onClick={() => handleSaveTitle(ec)}
                            style={{
                              padding: '3px 8px', fontSize: 11, fontWeight: 600,
                              background: 'var(--accent-cyan)', color: 'var(--bg-base)',
                              border: 'none', borderRadius: 'var(--radius-sm)', cursor: 'pointer',
                              whiteSpace: 'nowrap',
                            }}
                          >
                            Save
                          </button>
                          <button
                            onClick={() => setEditingTitle(null)}
                            style={{
                              padding: '3px 8px', fontSize: 11,
                              background: 'none', color: 'var(--text-muted)',
                              border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)', cursor: 'pointer',
                            }}
                          >
                            Cancel
                          </button>
                        </div>
                      ) : (
                        <div
                          onClick={() => { setEditingTitle(i); setEditTitleValue(ec.title || ec.filename || ''); }}
                          title="Click to edit title"
                          style={{
                            fontSize: 13, fontWeight: 600, overflow: 'hidden', textOverflow: 'ellipsis',
                            whiteSpace: 'nowrap', marginBottom: 2, cursor: 'pointer',
                            borderBottom: '1px dashed transparent', transition: 'border-color 0.15s',
                          }}
                          onMouseEnter={(e) => e.currentTarget.style.borderBottomColor = 'var(--accent-cyan)'}
                          onMouseLeave={(e) => e.currentTarget.style.borderBottomColor = 'transparent'}
                        >
                          {ec.title || ec.filename}
                          <span style={{ fontSize: 10, color: 'var(--text-muted)', marginLeft: 6, opacity: 0.6 }}>&#x270E;</span>
                        </div>
                      )}
                      <div style={{ fontSize: 11, color: 'var(--text-muted)', display: 'flex', gap: 10, flexWrap: 'wrap', alignItems: 'center' }}>
                        <span style={{ maxWidth: 160, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                          {ec.sourceFilename}
                        </span>
                        {dur != null && (
                          <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)' }}>
                            {formatDuration(dur)}
                          </span>
                        )}
                        {ec.start != null && ec.end != null && (
                          <span style={{ fontFamily: 'var(--font-mono)' }}>
                            {formatDuration(ec.start)} &rarr; {formatDuration(ec.end)}
                          </span>
                        )}
                        {ec.exported_at && (
                          <span style={{ fontSize: 10 }}>
                            {formatDateTime(ec.exported_at)}
                          </span>
                        )}
                      </div>
                    </div>
                    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', flexShrink: 0, position: 'relative', gap: 3 }}>
                      <button
                        onClick={() => setQualityMenuOpen(qualityMenuOpen === i ? null : i)}
                        style={{
                          fontSize: 10, fontWeight: 700, padding: '1px 6px',
                          background: 'var(--accent-cyan-dim)', color: 'var(--accent-cyan)',
                          border: '1px solid var(--accent-cyan)', borderRadius: 3,
                          fontFamily: 'var(--font-mono)', cursor: 'pointer',
                          display: 'flex', alignItems: 'center', gap: 3,
                        }}
                      >
                        {(ec.export_quality || '1080p').toUpperCase()} ▾
                      </button>
                      <a
                        href={`/api/files/${ec.jobId}/clips/${ec.filename}`}
                        download
                        style={{
                          padding: '6px 14px',
                          background: 'var(--accent-cyan)',
                          color: 'var(--bg-base)',
                          border: 'none',
                          borderRadius: 'var(--radius-sm)',
                          fontSize: 12,
                          fontWeight: 600,
                          textDecoration: 'none',
                          whiteSpace: 'nowrap',
                        }}
                      >
                        Export MP4
                      </a>
                      {qualityMenuOpen === i && (
                        <div style={{
                          position: 'absolute', bottom: '100%', right: 0,
                          marginBottom: 2, background: 'var(--bg-elevated)',
                          border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)',
                          zIndex: 20, overflow: 'hidden', minWidth: 150,
                          boxShadow: 'var(--shadow-sm)',
                        }}>
                          <div style={{ padding: '6px 10px', fontSize: 10, color: 'var(--text-muted)', borderBottom: '1px solid var(--border)', textTransform: 'uppercase', letterSpacing: 0.5 }}>
                            Re-export at
                          </div>
                          {['720p', '1080p', '4k'].map((q) => (
                            <button
                              key={q}
                              onClick={() => handleReExport(ec, q)}
                              style={{
                                display: 'block', width: '100%', padding: '6px 10px',
                                background: q === (ec.export_quality || '1080p') ? 'var(--accent-cyan)' : 'transparent',
                                color: q === (ec.export_quality || '1080p') ? 'var(--bg-base)' : 'var(--text-primary)',
                                border: 'none', fontSize: 12, textAlign: 'left',
                                cursor: 'pointer',
                              }}
                            >
                              {q === '720p' ? '720p (Smaller)' : q === '1080p' ? '1080p (Default)' : '4K (Best)'}
                            </button>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          ) : (
            <div style={{ textAlign: 'center', padding: '60px 24px', color: 'var(--text-muted)' }}>
              <div style={{ fontSize: 36, marginBottom: 12, opacity: 0.3 }}>&#x1F4E6;</div>
              <p>{exportSearch.trim()
                ? `No exported clips match "${exportSearch.trim()}"`
                : exportSourceFilter !== 'all'
                  ? 'No exported clips from this source.'
                  : 'No exported clips yet. Export clips from the Clips tab.'}</p>
            </div>
          )}
        </div>
      )}

      {/* ═══ Activity Log Tab ═══ */}
      {logsTab === 'logs' && (
        <div>
          {/* Log toolbar */}
          <div style={{
            display: 'flex', gap: 10, marginBottom: 12, flexWrap: 'wrap', alignItems: 'center',
            padding: '8px 12px', background: 'var(--bg-panel)', border: '1px solid var(--border)',
            borderRadius: 'var(--radius-md)',
          }}>
            {/* Search */}
            <div style={{ position: 'relative', flex: 1, minWidth: 160 }}>
              <input
                type="text"
                value={logSearch}
                onChange={(e) => setLogSearch(e.target.value)}
                placeholder="Search logs..."
                style={{
                  width: '100%',
                  padding: '5px 8px',
                  fontSize: 12,
                  background: 'var(--bg-elevated)',
                  border: '1px solid var(--border)',
                  borderRadius: 'var(--radius-sm)',
                  color: 'var(--text-primary)',
                  outline: 'none',
                  fontFamily: 'var(--font-mono)',
                }}
              />
            </div>

            {/* Level filter */}
            <select
              value={logFilter}
              onChange={(e) => setLogFilter(e.target.value)}
              style={{ ...selectStyle, fontFamily: 'var(--font-mono)' }}
            >
              <option value="all">All Levels</option>
              <option value="info">Info</option>
              <option value="success">Success</option>
              <option value="warning">Warning</option>
              <option value="error">Error</option>
              <option value="status">Status</option>
            </select>

            <span style={{ fontSize: 11, fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>
              {filteredLogs.length}{filteredLogs.length !== logs.length ? ` of ${logs.length}` : ''} entries
            </span>

            <div style={{ marginLeft: 'auto', display: 'flex', gap: 6 }}>
              {logs.length > 0 && (
                <button
                  onClick={exportLogsAsFile}
                  style={{
                    padding: '4px 10px',
                    background: 'var(--accent-cyan-dim)',
                    border: '1px solid var(--accent-cyan)',
                    borderRadius: 'var(--radius-sm)',
                    fontSize: 11,
                    fontWeight: 600,
                    color: 'var(--accent-cyan)',
                    cursor: 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    gap: 4,
                  }}
                  title="Download activity logs as text file"
                >
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                    <polyline points="7 10 12 15 17 10" />
                    <line x1="12" y1="15" x2="12" y2="3" />
                  </svg>
                  Export Activity
                </button>
              )}
              <a
                href="/api/logs/export"
                download
                style={{
                  padding: '4px 10px',
                  background: 'var(--bg-elevated)',
                  border: '1px solid var(--border)',
                  borderRadius: 'var(--radius-sm)',
                  fontSize: 11,
                  fontWeight: 600,
                  color: 'var(--text-secondary)',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: 4,
                  textDecoration: 'none',
                }}
                title="Download all server logs since container started"
              >
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <rect x="2" y="3" width="20" height="14" rx="2" ry="2" />
                  <line x1="8" y1="21" x2="16" y2="21" />
                  <line x1="12" y1="17" x2="12" y2="21" />
                </svg>
                Full Server Logs
              </a>
              {logs.length > 0 && (
                <button
                  onClick={clearLogs}
                  style={{
                    padding: '4px 10px',
                    background: 'none',
                    border: '1px solid var(--border)',
                    borderRadius: 'var(--radius-sm)',
                    fontSize: 11,
                    color: 'var(--text-muted)',
                    cursor: 'pointer',
                  }}
                >
                  Clear
                </button>
              )}
            </div>
          </div>

          {/* Log entries */}
          <div style={{
            background: 'var(--bg-panel)',
            border: '1px solid var(--border)',
            borderRadius: 'var(--radius-md)',
            maxHeight: 600,
            overflow: 'auto',
            fontFamily: 'var(--font-mono)',
            fontSize: 12,
          }}>
            {filteredLogs.length > 0 ? (
              filteredLogs.map((entry) => (
                <div
                  key={entry.id}
                  style={{
                    padding: '6px 12px',
                    borderBottom: '1px solid var(--border)',
                    display: 'flex',
                    gap: 8,
                    alignItems: 'baseline',
                  }}
                >
                  <span style={{ color: 'var(--text-muted)', fontSize: 10, flexShrink: 0, minWidth: 68 }}>
                    {formatTime(entry.timestamp)}
                  </span>
                  <span style={{
                    fontSize: 9, fontWeight: 700, flexShrink: 0, minWidth: 42,
                    color: LOG_LEVEL_COLORS[entry.level] || 'var(--text-muted)',
                    textTransform: 'uppercase',
                  }}>
                    {LOG_LEVEL_LABELS[entry.level] || entry.level}
                  </span>
                  <span style={{
                    color: LOG_LEVEL_COLORS[entry.level] || 'var(--text-secondary)',
                    wordBreak: 'break-word',
                  }}>
                    {typeof entry.message === 'string' ? entry.message : String(entry.message ?? '')}
                  </span>
                </div>
              ))
            ) : (
              <div style={{ padding: 24, textAlign: 'center', color: 'var(--text-muted)' }}>
                {logs.length === 0
                  ? 'No log entries yet. Processing events will appear here as videos are analyzed and clips are exported.'
                  : 'No logs match current filters.'}
              </div>
            )}
          </div>
        </div>
      )}

      {/* ═══ Allocation Tab ═══ */}
      {logsTab === 'allocation' && (
        <div>
          {/* Resource usage overview */}
          {allocation.resources?.cpu_percent !== undefined && (
            <div style={{
              display: 'grid', gridTemplateColumns: isMobile ? '1fr' : '1fr 1fr 1fr', gap: 12,
              marginBottom: 20,
            }}>
              {[
                { label: 'CPU', value: `${allocation.resources.cpu_percent}%`, color: allocation.resources.cpu_percent > 80 ? 'var(--danger)' : 'var(--accent-cyan)' },
                { label: 'Memory', value: `${allocation.resources.memory_used_mb}MB / ${allocation.resources.memory_total_mb}MB (${allocation.resources.memory_percent}%)`, color: allocation.resources.memory_percent > 85 ? 'var(--danger)' : 'var(--accent-cyan)' },
                { label: 'Disk', value: `${allocation.resources.disk_used_gb}GB / ${allocation.resources.disk_total_gb}GB (${allocation.resources.disk_percent}%)`, color: allocation.resources.disk_percent > 90 ? 'var(--danger)' : 'var(--accent-cyan)' },
              ].map(({ label, value, color }) => (
                <div key={label} style={{
                  background: 'var(--bg-panel)', border: '1px solid var(--border)',
                  borderRadius: 'var(--radius-md)', padding: '12px 16px',
                }}>
                  <div style={{ fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 4 }}>{label}</div>
                  <div style={{ fontSize: 13, fontFamily: 'var(--font-mono)', color, fontWeight: 600 }}>{value}</div>
                </div>
              ))}
            </div>
          )}

          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
            <h3 style={{ fontSize: 13, fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.06em', margin: 0 }}>
              Active Processes ({allocation.active_jobs?.length || 0})
            </h3>
            <button
              onClick={fetchAllocation}
              disabled={allocationLoading}
              style={{
                padding: '5px 12px', background: 'var(--bg-elevated)',
                color: 'var(--accent-cyan)', border: '1px solid var(--border)',
                borderRadius: 'var(--radius-sm)', fontSize: 11, fontWeight: 500,
                opacity: allocationLoading ? 0.6 : 1,
              }}
            >
              {allocationLoading ? 'Refreshing...' : 'Refresh'}
            </button>
          </div>

          {allocation.active_jobs?.length > 0 ? (
            <div style={{ display: 'grid', gap: 10 }}>
              {allocation.active_jobs.map((job) => (
                <div key={`${job.job_id}-${job.type}`} style={{
                  background: 'var(--bg-panel)', border: '1px solid var(--border)',
                  borderRadius: 'var(--radius-md)', padding: '12px 16px',
                }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 12, marginBottom: 8 }}>
                    <div style={{ flex: 1, minWidth: 0 }}>
                      <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-primary)', marginBottom: 2, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                        {job.filename || job.job_id}
                      </div>
                      <div style={{ fontSize: 11, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                        {job.type === 'analysis' ? 'Video Analysis' : job.type === 'export' ? 'Clip Export' : 'Clip Detection'}
                        {job.file_size_mb ? ` — ${job.file_size_mb}MB` : ''}
                        {job.duration ? ` — ${formatDuration(job.duration)}` : ''}
                        {job.elapsed_seconds ? ` — running ${job.elapsed_seconds > 60 ? Math.floor(job.elapsed_seconds / 60) + 'm' : job.elapsed_seconds + 's'}` : ''}
                      </div>
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexShrink: 0 }}>
                      <span style={{
                        fontSize: 10, fontFamily: 'var(--font-mono)', fontWeight: 700,
                        padding: '2px 8px', borderRadius: 8,
                        background: job.status === 'running' ? 'var(--accent-amber)' : job.status === 'encoding' ? 'var(--accent-cyan)' : 'var(--text-muted)',
                        color: 'var(--bg-base)',
                      }}>
                        {job.status?.toUpperCase()}
                      </span>
                    </div>
                  </div>
                  {job.progress_message && (
                    <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 8 }}>
                      {job.progress != null && <span style={{ color: 'var(--accent-cyan)', fontWeight: 600 }}>{job.progress}% </span>}
                      {job.progress_message}
                    </div>
                  )}
                  {job.progress != null && (
                    <div style={{ height: 3, background: 'var(--border)', borderRadius: 2, marginBottom: 10, overflow: 'hidden' }}>
                      <div style={{ height: '100%', background: 'var(--accent-cyan)', width: `${job.progress}%`, transition: 'width 0.3s ease' }} />
                    </div>
                  )}
                  <div style={{ display: 'flex', gap: 8 }}>
                    <button
                      onClick={() => handleForceStop(job)}
                      disabled={!!forceActioning[job.export_key || job.job_id]}
                      style={{
                        padding: '5px 12px', fontSize: 11, fontWeight: 600,
                        background: 'var(--amber-dim)', border: '1px solid var(--accent-amber)',
                        color: 'var(--accent-amber)', borderRadius: 'var(--radius-sm)',
                        opacity: forceActioning[job.export_key || job.job_id] ? 0.5 : 1,
                      }}
                    >
                      {forceActioning[job.export_key || job.job_id] === 'stopping' ? 'Stopping...' : 'Force Stop'}
                    </button>
                    <button
                      onClick={() => handleForceFail(job)}
                      disabled={!!forceActioning[job.export_key || job.job_id]}
                      style={{
                        padding: '5px 12px', fontSize: 11, fontWeight: 600,
                        background: 'var(--danger-dim)', border: '1px solid var(--danger)',
                        color: 'var(--danger)', borderRadius: 'var(--radius-sm)',
                        opacity: forceActioning[job.export_key || job.job_id] ? 0.5 : 1,
                      }}
                    >
                      {forceActioning[job.export_key || job.job_id] === 'failing' ? 'Failing...' : 'Force Fail'}
                    </button>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div style={{ padding: 32, textAlign: 'center', color: 'var(--text-muted)', fontSize: 13 }}>
              No active processes. The container is idle.
            </div>
          )}
        </div>
      )}

      <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.4; }
        }
      `}</style>
    </div>
  );
}
