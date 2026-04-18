import React from 'react';

/**
 * Persistent connection status banner.
 * Shows only when there's a problem (reconnecting/disconnected).
 * When connected, renders nothing (zero footprint).
 */
export default function ConnectionStatus({ status, latency }) {
  if (status === 'connected') return null;

  const configs = {
    connecting: {
      bg: 'var(--accent-cyan-dim)',
      border: 'var(--accent-cyan)',
      color: 'var(--accent-cyan)',
      icon: '\u25CF',
      message: 'Connecting to container...',
      pulse: true,
    },
    reconnecting: {
      bg: 'var(--amber-dim)',
      border: 'var(--accent-amber)',
      color: 'var(--accent-amber)',
      icon: '\u25CF',
      message: 'Reconnecting to container...',
      pulse: true,
    },
    disconnected: {
      bg: 'var(--danger-dim)',
      border: 'var(--danger)',
      color: 'var(--danger)',
      icon: '\u25CF',
      message: 'Connection lost \u2014 the container may be stopped or unreachable',
      pulse: false,
    },
  };

  const c = configs[status] || configs.disconnected;

  return (
    <div
      style={{
        padding: '8px 16px',
        background: c.bg,
        borderBottom: `1px solid ${c.border}`,
        display: 'flex',
        alignItems: 'center',
        gap: 8,
        fontSize: 13,
        fontWeight: 500,
        color: c.color,
        zIndex: 100,
      }}
    >
      <span
        style={{
          fontSize: 10,
          animation: c.pulse ? 'pulse 1.5s ease-in-out infinite' : undefined,
          flexShrink: 0,
        }}
      >
        {c.icon}
      </span>
      <span style={{ flex: 1 }}>{c.message}</span>
      {status === 'reconnecting' && (
        <span
          style={{
            fontSize: 11,
            fontFamily: 'var(--font-mono)',
            opacity: 0.7,
          }}
        >
          retrying...
        </span>
      )}
    </div>
  );
}
