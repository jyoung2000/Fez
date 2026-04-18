import React, { useRef } from 'react';

export default function ProgressBar({ progress, message, variant = 'cyan' }) {
  const colorMap = { amber: 'var(--accent-amber)', green: 'var(--success)', cyan: 'var(--accent-cyan)' };
  const color = colorMap[variant] || colorMap.cyan;

  // Track peak progress to ensure monotonically increasing display.
  // This prevents the bar from jumping backward on transient state updates.
  const peakRef = useRef(0);
  const safeProgress = Math.min(100, Math.max(0, progress));
  if (safeProgress >= peakRef.current) {
    peakRef.current = safeProgress;
  }
  // Reset peak when progress drops to 0 (new task started)
  if (safeProgress === 0) {
    peakRef.current = 0;
  }
  const displayProgress = peakRef.current;

  return (
    <div style={{ width: '100%' }}>
      {message && (
        <div
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            marginBottom: 6,
          }}
        >
          <span style={{ fontSize: 12, color: 'var(--text-secondary)' }}>{typeof message === 'string' ? message : String(message ?? '')}</span>
          <span style={{ fontSize: 12, fontWeight: 600, fontFamily: 'var(--font-mono)', color }}>{displayProgress}%</span>
        </div>
      )}
      <div
        style={{
          height: 10,
          background: 'var(--bg-elevated)',
          position: 'relative',
          overflow: 'hidden',
          borderRadius: 5,
          border: '1px solid var(--border)',
        }}
      >
        <div
          className={displayProgress < 100 ? 'shimmer' : ''}
          style={{
            height: '100%',
            width: `${displayProgress}%`,
            background: color,
            transition: 'width 0.3s ease',
            borderRadius: 5,
          }}
        />
      </div>
    </div>
  );
}
