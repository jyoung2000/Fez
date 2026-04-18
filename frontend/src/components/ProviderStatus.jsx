import React, { useState, useEffect } from 'react';

const STATUS_COLORS = {
  connected: 'var(--success)',
  configured: 'var(--accent-cyan)',
  loading: 'var(--accent-amber)',
  offline: 'var(--danger)',
  not_configured: 'var(--text-muted)',
};

// Shorten model ID for display: "meta-llama/llama-4-scout:free" → "llama-4-scout"
function shortModel(modelId) {
  if (!modelId) return '';
  let name = modelId;
  // Remove provider prefix
  if (name.includes('/')) name = name.split('/').pop();
  // Remove :free suffix
  name = name.replace(/:free$/, '');
  return name;
}

export default function ProviderStatus({ collapsed, onActiveChange }) {
  const [statuses, setStatuses] = useState({});

  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const res = await fetch('/api/providers/status');
        if (res.ok) {
          const data = await res.json();
          setStatuses(data);
          if (onActiveChange && data._active) {
            onActiveChange(data._active);
          }
        }
      } catch {
        // silent
      }
    };
    fetchStatus();
    const interval = setInterval(fetchStatus, 30000);
    return () => clearInterval(interval);
  }, []);

  const providers = ['ollama', 'openrouter', 'anthropic', 'gemini', 'groq'];
  const active = statuses._active || {};

  return (
    <div
      style={{
        padding: collapsed ? '8px' : '12px 16px',
        borderTop: '1px solid var(--border)',
      }}
    >
      {collapsed ? (
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 6 }}>
          {/* Active model indicator */}
          {active.provider && active.provider !== 'none' && (
            <div
              title={`T: ${active.transcript_model || 'whisper-base'} | V: ${shortModel(active.vision_model)} | Tx: ${shortModel(active.text_model)}`}
              style={{
                width: 20,
                height: 20,
                borderRadius: 'var(--radius-sm)',
                background: 'var(--accent-cyan)',
                color: 'var(--bg-base)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: 9,
                fontWeight: 700,
                fontFamily: 'var(--font-mono)',
                marginBottom: 4,
              }}
            >
              {active.provider.charAt(0).toUpperCase()}
            </div>
          )}
          {providers.map((name) => {
            const status = statuses[name]?.status || 'not_configured';
            return (
              <div
                key={name}
                title={`${name}: ${status}`}
                style={{
                  width: 8,
                  height: 8,
                  borderRadius: '50%',
                  background: STATUS_COLORS[status] || STATUS_COLORS.not_configured,
                  animation: status === 'connected' ? 'pulse 2s infinite' : undefined,
                }}
              />
            );
          })}
        </div>
      ) : (
        <div>
          {/* Active Models Section */}
          {active.provider && active.provider !== 'none' && (
            <div style={{ marginBottom: 12 }}>
              <div
                style={{
                  fontSize: 10,
                  fontFamily: 'var(--font-mono)',
                  color: 'var(--text-muted)',
                  textTransform: 'uppercase',
                  letterSpacing: '0.1em',
                  marginBottom: 6,
                }}
              >
                Active Models
              </div>
              <div
                style={{
                  background: 'var(--bg-elevated)',
                  border: '1px solid var(--border)',
                  borderRadius: 'var(--radius-sm)',
                  padding: '8px 10px',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: 4,
                }}
              >
                {/* Transcript AI */}
                <div style={{ fontSize: 10, fontFamily: 'var(--font-mono)', lineHeight: 1.5 }}>
                  <span style={{ color: 'var(--accent-amber)', fontWeight: 600 }}>transcript:</span>{' '}
                  <span style={{ color: 'var(--text-secondary)' }}>
                    {active.transcript_model || 'whisper-base'}
                  </span>
                </div>
                {/* Vision AI */}
                {active.vision_model && (
                  <div style={{ fontSize: 10, fontFamily: 'var(--font-mono)', lineHeight: 1.5 }}>
                    <span style={{ color: 'var(--accent-cyan)', fontWeight: 600 }}>vision:</span>{' '}
                    <span style={{ color: 'var(--text-secondary)' }}>{shortModel(active.vision_model)}</span>
                  </div>
                )}
                {/* Text AI (clip detection) */}
                {active.text_model && (
                  <div style={{ fontSize: 10, fontFamily: 'var(--font-mono)', lineHeight: 1.5 }}>
                    <span style={{ color: 'var(--success)', fontWeight: 600 }}>text:</span>{' '}
                    <span style={{ color: 'var(--text-secondary)' }}>{shortModel(active.text_model)}</span>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Provider list */}
          <div
            style={{
              fontSize: 10,
              fontFamily: 'var(--font-mono)',
              color: 'var(--text-muted)',
              textTransform: 'uppercase',
              letterSpacing: '0.1em',
              marginBottom: 8,
            }}
          >
            Providers
          </div>
          {providers.map((name) => {
            const info = statuses[name] || {};
            const status = info.status || 'not_configured';
            const isActive = name === active.provider;
            return (
              <div
                key={name}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 8,
                  padding: '4px 0',
                  fontSize: 12,
                }}
              >
                <div
                  style={{
                    width: 8,
                    height: 8,
                    borderRadius: '50%',
                    background: STATUS_COLORS[status] || STATUS_COLORS.not_configured,
                    flexShrink: 0,
                    animation: status === 'connected' ? 'pulse 2s infinite' : undefined,
                  }}
                />
                <span style={{
                  color: isActive ? 'var(--accent-cyan)' : 'var(--text-secondary)',
                  textTransform: 'capitalize',
                  fontWeight: isActive ? 600 : 400,
                }}>
                  {name}
                </span>
                {isActive && (
                  <span style={{ fontSize: 9, color: 'var(--accent-cyan)', fontFamily: 'var(--font-mono)', marginLeft: 'auto' }}>
                    ACTIVE
                  </span>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
