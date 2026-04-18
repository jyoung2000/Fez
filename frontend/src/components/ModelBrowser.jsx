import React, { useState, useEffect, useMemo } from 'react';

export default function ModelBrowser({ onSelect, type = 'text' }) {
  const [models, setModels] = useState({ vision_models: [], text_models: [] });
  const [search, setSearch] = useState('');
  const [freeOnly, setFreeOnly] = useState(false);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  const loadModels = () => {
    setLoading(true);
    fetch('/api/providers/models')
      .then((r) => r.json())
      .then(setModels)
      .catch(() => {})
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    loadModels();
  }, []);

  const handleRefresh = async () => {
    setRefreshing(true);
    try {
      await fetch('/api/providers/models/refresh', { method: 'POST' });
      loadModels();
    } catch {} finally {
      setRefreshing(false);
    }
  };

  const list = type === 'vision' ? models.vision_models : models.text_models;

  const filtered = useMemo(() => {
    let result = list || [];
    // For vision models, filter out those with context too small for subject tracking
    if (type === 'vision') {
      result = result.filter((m) => {
        const ctx = m.context_length || 0;
        // Allow models with unknown context (Ollama, auto-routers) or >= 16K
        return ctx === 0 || ctx >= 16000;
      });
    }
    if (search.trim()) {
      const q = search.toLowerCase();
      result = result.filter((m) => m.id.toLowerCase().includes(q) || m.name.toLowerCase().includes(q));
    }
    if (freeOnly) {
      result = result.filter((m) => m.id.includes(':free') || (m.pricing?.prompt === '0' && m.pricing?.completion === '0'));
    }
    return result.slice(0, 50);
  }, [list, search, freeOnly, type]);

  if (loading) {
    return <div style={{ color: 'var(--text-secondary)', fontSize: 13 }}>Loading models...</div>;
  }

  return (
    <div>
      <div style={{ display: 'flex', gap: 8, marginBottom: 12, flexWrap: 'wrap' }}>
        <input
          type="text"
          placeholder="Search models..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          style={{ flex: 1, padding: '6px 10px', borderRadius: 'var(--radius-sm)', fontSize: 13 }}
        />
        <label style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: 12, color: 'var(--text-secondary)' }}>
          <input
            type="checkbox"
            checked={freeOnly}
            onChange={(e) => setFreeOnly(e.target.checked)}
            style={{ accentColor: 'var(--accent-cyan)' }}
          />
          Free only
        </label>
        <button
          onClick={handleRefresh}
          disabled={refreshing}
          title="Refresh model list from OpenRouter"
          style={{
            padding: '4px 10px',
            background: 'var(--bg-elevated)',
            color: 'var(--accent-cyan)',
            border: '1px solid var(--border)',
            borderRadius: 'var(--radius-sm)',
            fontSize: 12,
            opacity: refreshing ? 0.5 : 1,
          }}
        >
          {refreshing ? '...' : '\u21bb'}
        </button>
      </div>

      <div style={{ maxHeight: 300, overflow: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <thead>
            <tr style={{ borderBottom: '1px solid var(--border)' }}>
              <th style={{ textAlign: 'left', padding: '6px 8px', color: 'var(--text-secondary)', fontWeight: 600 }}>Model</th>
              <th style={{ textAlign: 'right', padding: '6px 8px', color: 'var(--text-secondary)', fontWeight: 600 }}>Context</th>
              <th style={{ textAlign: 'center', padding: '6px 8px', width: 60 }}></th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((m) => (
              <tr
                key={m.id}
                style={{ borderBottom: '1px solid var(--border)' }}
              >
                <td style={{ padding: '6px 8px' }}>
                  <div style={{ color: 'var(--text-primary)' }}>{m.name}</div>
                  <div style={{ color: 'var(--text-muted)', fontSize: 10 }}>{m.id}</div>
                </td>
                <td style={{ textAlign: 'right', padding: '6px 8px', fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)' }}>
                  {m.context_length ? `${(m.context_length / 1000).toFixed(0)}k` : '-'}
                </td>
                <td style={{ textAlign: 'center', padding: '6px 8px' }}>
                  <button
                    onClick={() => onSelect?.(m.id)}
                    style={{
                      padding: '3px 8px',
                      background: 'var(--accent-cyan-dim)',
                      color: 'var(--accent-cyan)',
                      border: '1px solid var(--accent-cyan)',
                      borderRadius: 'var(--radius-sm)',
                      fontSize: 10,
                      fontWeight: 600,
                    }}
                  >
                    Select
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        {filtered.length === 0 && (
          <div style={{ textAlign: 'center', padding: 16, color: 'var(--text-muted)', fontSize: 12 }}>
            No models found.
          </div>
        )}
      </div>
    </div>
  );
}
