import React, { useState, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import useResponsive from '../hooks/useResponsive';

function formatDuration(seconds) {
  if (!seconds) return '-';
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, '0')}`;
}

function formatDate(iso) {
  if (!iso) return '';
  try {
    return new Date(iso).toLocaleDateString(undefined, {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  } catch {
    return iso;
  }
}

const STATUS_STYLES = {
  queued: { className: 'badge-gray', label: 'Queued' },
  extracting_frames: { className: 'badge-cyan pulse', label: 'Processing' },
  transcribing: { className: 'badge-cyan pulse', label: 'Processing' },
  analyzing_scenes: { className: 'badge-cyan pulse', label: 'Processing' },
  generating_summary: { className: 'badge-cyan pulse', label: 'Processing' },
  detecting_clips: { className: 'badge-cyan pulse', label: 'Processing' },
  complete: { className: 'badge-green', label: 'Complete' },
  failed: { className: 'badge-red', label: 'Failed' },
  cancelled: { className: 'badge-amber', label: 'Cancelled' },
};

const CANCELLABLE = [
  'queued', 'extracting_frames', 'transcribing',
  'analyzing_scenes', 'generating_summary', 'detecting_clips',
];

export default function Dashboard() {
  const [jobs, setJobs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [cancelling, setCancelling] = useState({});
  const [searchQuery, setSearchQuery] = useState('');
  const [statusFilter, setStatusFilter] = useState('all');
  const [sortBy, setSortBy] = useState('newest');
  const navigate = useNavigate();
  const { isMobile } = useResponsive();

  // Track cancelled job IDs so the 5-second poll never brings them back
  const cancelledIdsRef = React.useRef(new Set());

  const handleCancel = async (e, jobId) => {
    e.stopPropagation();
    if (cancelling[jobId]) return; // debounce
    setCancelling((prev) => ({ ...prev, [jobId]: true }));

    // Mark this ID as cancelled so polls will filter it out
    cancelledIdsRef.current.add(jobId);

    // Optimistically remove from UI immediately
    setJobs((prev) => prev.filter((j) => j.job_id !== jobId));

    try {
      await fetch(`/api/jobs/${jobId}/cancel`, { method: 'POST' });
      // Wait for status to transition then delete the job files
      _waitAndDelete(jobId);
    } catch {
      // If cancel fails, still try to clean up
      _waitAndDelete(jobId);
    }
  };

  // Poll for cancelled status then auto-delete job files
  const _waitAndDelete = async (jobId) => {
    for (let i = 0; i < 15; i++) {
      await new Promise((r) => setTimeout(r, 1000));
      try {
        const res = await fetch(`/api/jobs/${jobId}`);
        if (!res.ok) {
          // Already gone — clean up tracking
          cancelledIdsRef.current.delete(jobId);
          setCancelling((prev) => { const n = { ...prev }; delete n[jobId]; return n; });
          return;
        }
        const job = await res.json();
        if (job.status === 'cancelled' || job.status === 'failed') {
          await fetch(`/api/jobs/${jobId}`, { method: 'DELETE' });
          cancelledIdsRef.current.delete(jobId);
          setCancelling((prev) => { const n = { ...prev }; delete n[jobId]; return n; });
          return;
        }
      } catch {
        cancelledIdsRef.current.delete(jobId);
        setCancelling((prev) => { const n = { ...prev }; delete n[jobId]; return n; });
        return;
      }
    }
    // Timed out — try deleting anyway
    try { await fetch(`/api/jobs/${jobId}`, { method: 'DELETE' }); } catch {}
    cancelledIdsRef.current.delete(jobId);
    setCancelling((prev) => { const n = { ...prev }; delete n[jobId]; return n; });
  };

  const handleDelete = async (e, jobId) => {
    e.stopPropagation();
    try {
      const res = await fetch(`/api/jobs/${jobId}`, { method: 'DELETE' });
      if (res.ok) {
        setJobs((prev) => prev.filter((j) => j.job_id !== jobId));
      }
    } catch {}
  };

  const [fetchError, setFetchError] = useState(false);
  const [lastFetchTime, setLastFetchTime] = useState(null);

  useEffect(() => {
    const fetchJobs = async () => {
      try {
        const res = await fetch('/api/jobs');
        if (res.ok) {
          const data = await res.json();
          // Filter out jobs that are being cancelled — prevents them from reappearing
          const filtered = data.filter((j) => !cancelledIdsRef.current.has(j.job_id));
          setJobs(filtered);
          setFetchError(false);
          setLastFetchTime(Date.now());
        } else {
          setFetchError(true);
        }
      } catch {
        setFetchError(true);
      } finally {
        setLoading(false);
      }
    };
    fetchJobs();
    const interval = setInterval(fetchJobs, 5000);
    return () => clearInterval(interval);
  }, []);

  // Filter and sort jobs
  let filteredJobs = jobs;
  if (searchQuery.trim()) {
    const q = searchQuery.trim().toLowerCase();
    filteredJobs = filteredJobs.filter((j) =>
      (j.filename || '').toLowerCase().includes(q) ||
      (j.progress_message || '').toLowerCase().includes(q) ||
      (j.job_id || '').toLowerCase().includes(q)
    );
  }
  if (statusFilter !== 'all') {
    if (statusFilter === 'processing') {
      filteredJobs = filteredJobs.filter((j) => !['complete', 'failed', 'queued', 'cancelled'].includes(j.status));
    } else {
      filteredJobs = filteredJobs.filter((j) => j.status === statusFilter);
    }
  }
  filteredJobs = [...filteredJobs].sort((a, b) => {
    if (sortBy === 'oldest') return (a.created_at || '').localeCompare(b.created_at || '');
    if (sortBy === 'name') return (a.filename || '').localeCompare(b.filename || '');
    if (sortBy === 'clips') return (b.clips_count || 0) - (a.clips_count || 0);
    return (b.created_at || '').localeCompare(a.created_at || ''); // newest
  });

  const totalClips = jobs.reduce((sum, j) => sum + (j.clips_count || 0), 0);
  const processingCount = jobs.filter((j) => !['complete', 'failed', 'queued', 'cancelled'].includes(j.status)).length;

  if (loading) {
    return (
      <div style={{ textAlign: 'center', padding: 48, color: 'var(--text-secondary)' }}>
        <div style={{
          width: 24, height: 24, border: '2px solid var(--border)', borderTopColor: 'var(--accent-cyan)',
          borderRadius: '50%', animation: 'spin 0.8s linear infinite',
          margin: '0 auto 12px',
        }} />
        Connecting to container...
      </div>
    );
  }

  return (
    <div>
      {/* Connection error banner */}
      {fetchError && (
        <div style={{
          padding: '10px 16px', marginBottom: 16,
          background: 'var(--amber-dim)', border: '1px solid var(--accent-amber)',
          borderRadius: 'var(--radius-sm)', fontSize: 13, color: 'var(--accent-amber)',
          display: 'flex', alignItems: 'center', gap: 8,
        }}>
          <span style={{ fontSize: 10, animation: 'pulse 1.5s ease-in-out infinite' }}>{'\u25CF'}</span>
          Unable to reach the backend — retrying automatically...
        </div>
      )}

      {/* Stats bar */}
      <div style={{ display: 'flex', gap: isMobile ? 10 : 16, marginBottom: isMobile ? 20 : 24, flexWrap: 'wrap' }}>
        {[
          { label: 'Total Videos', value: jobs.length, color: 'var(--accent-cyan)' },
          { label: 'Clips Extracted', value: totalClips, color: 'var(--accent-amber)' },
          { label: 'Processing', value: processingCount, color: processingCount > 0 ? 'var(--accent-amber)' : 'var(--text-primary)' },
        ].map(({ label, value, color }) => (
          <div key={label} style={{
            background: 'var(--bg-panel)', border: '1px solid var(--border)',
            borderRadius: 'var(--radius-md)', padding: isMobile ? '10px 14px' : '12px 20px',
            flex: 1, minWidth: isMobile ? 0 : 140, boxShadow: 'var(--shadow-sm)',
          }}>
            <div style={{ fontSize: 10, fontFamily: 'var(--font-mono)', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.1em' }}>
              {label}
            </div>
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: isMobile ? 22 : 28, fontWeight: 700, color }}>
              {value}
            </div>
          </div>
        ))}
      </div>

      {/* Search & Filter Bar */}
      {jobs.length > 0 && (
        <div style={{ marginBottom: 16, display: 'flex', gap: 10, flexWrap: 'wrap', alignItems: 'stretch' }}>
          {/* Search input */}
          <div style={{ position: 'relative', flex: '1 1 220px', minWidth: 0 }}>
            <div style={{
              position: 'absolute', left: 14, top: '50%', transform: 'translateY(-50%)',
              color: 'var(--text-muted)', fontSize: 15, pointerEvents: 'none', lineHeight: 1,
            }}>
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <circle cx="11" cy="11" r="8" /><line x1="21" y1="21" x2="16.65" y2="16.65" />
              </svg>
            </div>
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search videos by name..."
              style={{
                width: '100%',
                padding: '10px 14px 10px 40px',
                fontSize: 14,
                background: 'var(--bg-panel)',
                border: '1px solid var(--border)',
                borderRadius: 'var(--radius-md)',
                color: 'var(--text-primary)',
                outline: 'none',
                transition: 'border-color 0.2s ease, box-shadow 0.2s ease',
              }}
              onFocus={(e) => {
                e.target.style.borderColor = 'var(--accent-cyan)';
                e.target.style.boxShadow = '0 0 0 3px var(--accent-cyan-dim)';
              }}
              onBlur={(e) => {
                e.target.style.borderColor = 'var(--border)';
                e.target.style.boxShadow = 'none';
              }}
            />
            {searchQuery && (
              <button
                onClick={() => setSearchQuery('')}
                style={{
                  position: 'absolute', right: 10, top: '50%', transform: 'translateY(-50%)',
                  background: 'var(--bg-elevated)', border: 'none', borderRadius: '50%',
                  width: 20, height: 20, display: 'flex', alignItems: 'center', justifyContent: 'center',
                  color: 'var(--text-muted)', fontSize: 12, cursor: 'pointer', lineHeight: 1,
                }}
              >
                &times;
              </button>
            )}
          </div>

          {/* Status filter */}
          <select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
            style={{
              padding: '10px 12px', borderRadius: 'var(--radius-md)', fontSize: 13,
              background: 'var(--bg-panel)', border: '1px solid var(--border)',
              color: 'var(--text-primary)', outline: 'none', cursor: 'pointer',
            }}
          >
            <option value="all">All Status</option>
            <option value="processing">Processing</option>
            <option value="complete">Complete</option>
            <option value="failed">Failed</option>
            <option value="queued">Queued</option>
          </select>

          {/* Sort */}
          <select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value)}
            style={{
              padding: '10px 12px', borderRadius: 'var(--radius-md)', fontSize: 13,
              background: 'var(--bg-panel)', border: '1px solid var(--border)',
              color: 'var(--text-primary)', outline: 'none', cursor: 'pointer',
            }}
          >
            <option value="newest">Newest First</option>
            <option value="oldest">Oldest First</option>
            <option value="name">By Name</option>
            <option value="clips">Most Clips</option>
          </select>
        </div>
      )}

      {/* Job list or empty state */}
      {jobs.length === 0 ? (
        <div
          style={{
            textAlign: 'center',
            padding: isMobile ? '48px 20px' : '80px 24px',
            border: '2px dashed var(--border)',
            borderRadius: 'var(--radius-lg)',
          }}
        >
          <div style={{ fontSize: 48, marginBottom: 16, opacity: 0.3 }}>&#x1F3AC;</div>
          <h3 style={{ fontSize: 18, marginBottom: 8, color: 'var(--text-secondary)' }}>
            No videos yet
          </h3>
          <p style={{ color: 'var(--text-muted)', marginBottom: 24 }}>
            Drop your first video to get started
          </p>
          <Link
            to="/upload"
            style={{
              display: 'inline-block',
              padding: '10px 24px',
              background: 'var(--accent-cyan)',
              color: 'var(--bg-base)',
              fontWeight: 600,
              borderRadius: 'var(--radius-sm)',
              textDecoration: 'none',
            }}
          >
            Upload Video
          </Link>
        </div>
      ) : filteredJobs.length === 0 ? (
        <div style={{
          textAlign: 'center', padding: '48px 24px',
          color: 'var(--text-muted)', fontSize: 14,
        }}>
          {searchQuery.trim()
            ? `No videos match "${searchQuery.trim()}".`
            : 'No videos match the selected filter.'}
        </div>
      ) : (
        <div
          className="responsive-grid"
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fill, minmax(min(320px, 100%), 1fr))',
            gap: isMobile ? 12 : 16,
          }}
        >
          {filteredJobs.map((job, i) => {
            const statusInfo = STATUS_STYLES[job.status] || STATUS_STYLES.queued;
            return (
              <div
                key={job.job_id}
                className="card-hover slide-in"
                onClick={() => navigate(`/analysis/${job.job_id}`)}
                style={{
                  background: 'var(--bg-panel)',
                  border: '1px solid var(--border)',
                  borderRadius: 'var(--radius-md)',
                  boxShadow: 'var(--shadow-sm)',
                  cursor: 'pointer',
                  animationDelay: `${i * 50}ms`,
                }}
              >
                {/* Progress bar for active jobs */}
                {job.progress > 0 && job.progress < 100 && (
                  <div style={{ height: 2, background: 'var(--bg-elevated)' }}>
                    <div
                      className="shimmer"
                      style={{ height: '100%', width: `${job.progress}%`, background: 'var(--accent-cyan)' }}
                    />
                  </div>
                )}

                <div style={{ padding: 16 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 12 }}>
                    <div>
                      <h4 style={{ fontSize: 14, marginBottom: 4, wordBreak: 'break-word' }}>
                        {String(job.filename || '')}
                      </h4>
                      <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                        {formatDate(job.created_at)}
                      </span>
                    </div>
                    <span className={`badge ${statusInfo.className}`}>
                      {statusInfo.label}
                    </span>
                  </div>

                  <div style={{ display: 'flex', gap: 16, fontSize: 12, color: 'var(--text-secondary)' }}>
                    <span style={{ fontFamily: 'var(--font-mono)' }}>
                      {formatDuration(job.duration)}
                    </span>
                    {job.file_size_mb > 0 && (
                      <span style={{ fontFamily: 'var(--font-mono)' }}>
                        {job.file_size_mb.toFixed(1)}MB
                      </span>
                    )}
                    {job.clips_count > 0 && (
                      <span style={{ color: 'var(--accent-amber)', fontFamily: 'var(--font-mono)' }}>
                        {job.clips_count} clips
                      </span>
                    )}
                  </div>

                  {job.progress_message && job.status !== 'complete' && (
                    <div style={{ marginTop: 8, fontSize: 11, color: 'var(--accent-cyan)' }}>
                      {String(job.progress_message || '')}
                    </div>
                  )}

                  {/* Action buttons */}
                  <div style={{ display: 'flex', gap: 8, marginTop: 10 }}>
                    {CANCELLABLE.includes(job.status) && (
                      <button
                        onClick={(e) => handleCancel(e, job.job_id)}
                        disabled={cancelling[job.job_id]}
                        style={{
                          padding: '6px 14px',
                          background: 'var(--amber-dim)',
                          border: '1px solid var(--accent-amber)',
                          color: 'var(--accent-amber)',
                          fontSize: 12,
                          fontWeight: 600,
                          cursor: cancelling[job.job_id] ? 'default' : 'pointer',
                          borderRadius: 'var(--radius-sm)',
                          opacity: cancelling[job.job_id] ? 0.6 : 1,
                        }}
                      >
                        {cancelling[job.job_id] ? 'Cancelling...' : 'Cancel'}
                      </button>
                    )}
                    {(job.status === 'failed' || job.status === 'cancelled') && (
                      <button
                        onClick={(e) => handleDelete(e, job.job_id)}
                        style={{
                          padding: '6px 14px',
                          background: 'var(--danger-dim)',
                          border: '1px solid var(--danger)',
                          color: 'var(--danger)',
                          fontSize: 12,
                          fontWeight: 600,
                          cursor: 'pointer',
                          borderRadius: 'var(--radius-sm)',
                        }}
                      >
                        Remove
                      </button>
                    )}
                    {job.status === 'complete' && (
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          if (window.confirm(`Delete "${job.filename}" and all its clips? This cannot be undone.`)) {
                            handleDelete(e, job.job_id);
                          }
                        }}
                        style={{
                          padding: '6px 14px',
                          background: 'var(--danger-dim)',
                          border: '1px solid var(--danger)',
                          color: 'var(--danger)',
                          fontSize: 12,
                          fontWeight: 600,
                          cursor: 'pointer',
                          borderRadius: 'var(--radius-sm)',
                        }}
                      >
                        Delete
                      </button>
                    )}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
