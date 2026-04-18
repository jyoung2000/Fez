import React, { useState, useEffect } from 'react';
import useResponsive from '../hooks/useResponsive';

export default function CostTracker() {
  const { isMobile } = useResponsive();
  const [jobs, setJobs] = useState([]);

  useEffect(() => {
    fetch('/api/jobs')
      .then((r) => r.json())
      .then(setJobs)
      .catch(() => {});
  }, []);

  const totalCost = jobs.reduce((sum, j) => sum + (j.estimated_cost_usd || 0), 0);
  const completedJobs = jobs.filter((j) => j.status === 'complete').length;
  const hasFreeTier = jobs.some((j) => {
    const providers = Object.values(j.provider_used || {});
    return providers.some((p) => p && (p.includes('free') || p === 'ollama'));
  });

  const download = () => {
    const header = 'Job ID,Filename,Provider,Status,Cost USD\n';
    const rows = jobs
      .map(
        (j) =>
          `${j.job_id},${j.filename},${JSON.stringify(j.provider_used || {})},${j.status},${j.estimated_cost_usd || 0}`
      )
      .join('\n');
    const blob = new Blob([header + rows], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'clipai_usage.csv';
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div>
      <div
        style={{
          display: 'flex',
          gap: isMobile ? 12 : 24,
          marginBottom: 24,
          flexWrap: 'wrap',
        }}
      >
        <div
          style={{
            background: 'var(--bg-panel)',
            border: '1px solid var(--border)',
            padding: isMobile ? 12 : 16,
            borderRadius: 'var(--radius-md)',
            boxShadow: 'var(--shadow-sm)',
            flex: 1,
            minWidth: isMobile ? '100%' : 150,
          }}
        >
          <div style={{ fontSize: 11, color: 'var(--text-secondary)', textTransform: 'uppercase', marginBottom: 4 }}>
            Total Videos
          </div>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: 24, color: 'var(--accent-cyan)' }}>
            {jobs.length}
          </div>
        </div>
        <div
          style={{
            background: 'var(--bg-panel)',
            border: '1px solid var(--border)',
            padding: isMobile ? 12 : 16,
            borderRadius: 'var(--radius-md)',
            boxShadow: 'var(--shadow-sm)',
            flex: 1,
            minWidth: isMobile ? '100%' : 150,
          }}
        >
          <div style={{ fontSize: 11, color: 'var(--text-secondary)', textTransform: 'uppercase', marginBottom: 4 }}>
            Est. Total Cost
          </div>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: 24, color: 'var(--accent-amber)' }}>
            ${totalCost.toFixed(4)}
          </div>
          {totalCost === 0 && completedJobs > 0 && (
            <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 4 }}>
              {hasFreeTier ? 'Free tier / local models' : 'Cost tracked on next analysis'}
            </div>
          )}
        </div>
        <div
          style={{
            background: 'var(--bg-panel)',
            border: '1px solid var(--border)',
            padding: isMobile ? 12 : 16,
            borderRadius: 'var(--radius-md)',
            boxShadow: 'var(--shadow-sm)',
            flex: 1,
            minWidth: isMobile ? '100%' : 150,
          }}
        >
          <div style={{ fontSize: 11, color: 'var(--text-secondary)', textTransform: 'uppercase', marginBottom: 4 }}>
            Completed
          </div>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: 24, color: 'var(--success)' }}>
            {completedJobs}
          </div>
        </div>
      </div>

      {/* Job table */}
      <div style={{ overflow: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <thead>
            <tr style={{ borderBottom: '1px solid var(--border)' }}>
              <th style={{ textAlign: 'left', padding: '8px', color: 'var(--text-secondary)' }}>Filename</th>
              <th style={{ textAlign: 'left', padding: '8px', color: 'var(--text-secondary)' }}>Provider</th>
              <th style={{ textAlign: 'left', padding: '8px', color: 'var(--text-secondary)' }}>Status</th>
              <th style={{ textAlign: 'right', padding: '8px', color: 'var(--text-secondary)' }}>Cost</th>
            </tr>
          </thead>
          <tbody>
            {jobs.map((j) => (
              <tr key={j.job_id} style={{ borderBottom: '1px solid var(--border)' }}>
                <td style={{ padding: '8px', color: 'var(--text-primary)' }}>{j.filename}</td>
                <td style={{ padding: '8px', color: 'var(--text-secondary)', fontFamily: 'var(--font-mono)', fontSize: 11 }}>
                  {Object.values(j.provider_used || {}).join(', ') || '-'}
                </td>
                <td style={{ padding: '8px' }}>
                  <span className={`badge ${j.status === 'complete' ? 'badge-green' : j.status === 'failed' ? 'badge-red' : 'badge-cyan'}`}>
                    {j.status}
                  </span>
                </td>
                <td style={{ padding: '8px', textAlign: 'right', fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)' }}>
                  {j.estimated_cost_usd ? `$${j.estimated_cost_usd.toFixed(4)}` : j.status === 'complete' ? 'Free' : '-'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div style={{ marginTop: 16 }}>
        <button
          onClick={download}
          style={{
            padding: '8px 16px',
            background: 'var(--bg-elevated)',
            color: 'var(--text-secondary)',
            border: '1px solid var(--border)',
            borderRadius: 'var(--radius-sm)',
            fontSize: 12,
          }}
        >
          Export as CSV
        </button>
      </div>
    </div>
  );
}
