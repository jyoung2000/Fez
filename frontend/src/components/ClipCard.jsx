import React, { useState } from 'react';

function formatDuration(seconds) {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, '0')}`;
}

function parseDuration(str) {
  const trimmed = (str || '').trim();
  if (!trimmed) return null;
  const parts = trimmed.split(':');
  if (parts.length === 2) {
    const m = parseInt(parts[0], 10);
    const s = parseInt(parts[1], 10);
    if (!isNaN(m) && !isNaN(s) && m >= 0 && s >= 0 && s < 60) return m * 60 + s;
    return null;
  }
  const num = parseFloat(trimmed);
  if (!isNaN(num) && num >= 0) return num;
  return null;
}

export default function ClipCard({ clip, jobId, isBest, onPreview, onExport, onDelete, onTimesChanged, onFindMoreLikeThis, selected, onSelect, exportQuality = '1080p', scenes }) {
  const [exporting, setExporting] = useState(false);
  const [qualityMenuOpen, setQualityMenuOpen] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [editingTimes, setEditingTimes] = useState(false);
  const [startText, setStartText] = useState('');
  const [endText, setEndText] = useState('');
  const [savingTimes, setSavingTimes] = useState(false);
  const [editingTitle, setEditingTitle] = useState(false);
  const [titleText, setTitleText] = useState('');
  const [savingTitle, setSavingTitle] = useState(false);
  // Score Breakdown panel (Enhancement 1) — collapsed by default.
  const [breakdownOpen, setBreakdownOpen] = useState(false);

  // In focus mode the LLM puts relevance in viral_score and the
  // backend moves the composite virality to viral_score_composite.
  // The big number rendered in the score pill follows this rule so the
  // UI matches the mode's contract (see Bug 3 in the clip-focus audit).
  const isFocusClip = !!clip.clip_focus;
  const primaryScore = isFocusClip
    ? (clip.focus_relevance != null ? clip.focus_relevance : clip.viral_score)
    : clip.viral_score;
  const compositeScore = clip.viral_score_composite != null ? clip.viral_score_composite : clip.viral_score;
  const diag = clip.score_diagnostics || {};
  const axisScores = diag.axis_scores || {
    hook: clip.hook_score || 0,
    flow: clip.flow_score || 0,
    value: clip.value_score || 0,
    trend: clip.trend_score || 0,
  };
  const axisReasons = diag.axis_reasons || {
    hook: clip.hook_reason,
    flow: clip.flow_reason,
    value: clip.value_reason,
    trend: clip.trend_reason,
  };
  const weights = diag.weights || { hook: 0.3, flow: 0.25, value: 0.3, trend: 0.15 };
  const contentType = diag.content_type || null;
  const legacyFill = !!diag.legacy_fill;
  const hasAnyAxis = Object.values(axisScores).some((v) => (v || 0) > 0);

  // Focus tier color (ported from ViralClips.jsx 1576-1584 per Bug 5)
  const tier = clip.focus_tier;
  const tierColor = tier === 'strong' ? 'var(--success)'
    : tier === 'moderate' ? 'var(--accent-amber)'
    : 'var(--text-secondary)';
  const tierBg = tier === 'strong' ? 'rgba(52,199,89,0.15)'
    : tier === 'moderate' ? 'rgba(255,214,0,0.15)'
    : 'rgba(142,142,147,0.15)';

  // Find the most important scene description within this clip's time range
  const clipSubject = React.useMemo(() => {
    if (!scenes?.length) return null;
    const inRange = scenes.filter(
      (s) => s.timestamp >= clip.start_time && s.timestamp <= clip.end_time
    );
    if (!inRange.length) return null;
    // Pick the highest importance scene as the primary subject
    const best = inRange.reduce((a, b) => (b.importance_score > a.importance_score ? b : a), inRange[0]);
    return best.description;
  }, [scenes, clip.start_time, clip.end_time]);

  const scoreColor = primaryScore >= 80
    ? 'var(--accent-amber)'
    : primaryScore >= 50
      ? 'var(--accent-cyan)'
      : 'var(--text-secondary)';

  const handleExport = async (qualityOverride) => {
    setExporting(true);
    setQualityMenuOpen(false);
    try {
      await onExport(clip, qualityOverride);
    } finally {
      setExporting(false);
    }
  };

  const openTimeEdit = () => {
    setStartText(formatDuration(clip.start_time));
    setEndText(formatDuration(clip.end_time));
    setEditingTimes(true);
  };

  const cancelTimeEdit = () => {
    setEditingTimes(false);
  };

  const saveTimeEdit = async () => {
    const newStart = parseDuration(startText);
    const newEnd = parseDuration(endText);
    if (newStart === null || newEnd === null || newStart >= newEnd || newStart < 0) return;
    setSavingTimes(true);
    try {
      const res = await fetch(`/api/jobs/${jobId}/clips/${clip.id}/times`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ start_time: newStart, end_time: newEnd }),
      });
      if (res.ok) {
        setEditingTimes(false);
        if (onTimesChanged) onTimesChanged();
      }
    } catch {
      // ignore
    } finally {
      setSavingTimes(false);
    }
  };

  const openTitleEdit = () => {
    setTitleText(clip.title || '');
    setEditingTitle(true);
  };

  const saveTitleEdit = async () => {
    const trimmed = titleText.trim();
    if (!trimmed || trimmed === clip.title) { setEditingTitle(false); return; }
    setSavingTitle(true);
    try {
      const res = await fetch(`/api/jobs/${jobId}/clips/${clip.id}/title`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title: trimmed }),
      });
      if (res.ok) {
        setEditingTitle(false);
        if (onTimesChanged) onTimesChanged();
      }
    } catch {
      // ignore
    } finally {
      setSavingTitle(false);
    }
  };

  return (
    <div
      className="card-hover slide-in"
      style={{
        background: 'var(--bg-panel)',
        border: `1px solid ${isBest ? 'var(--accent-amber)' : 'var(--border)'}`,
        borderRadius: 'var(--radius-md)',
        boxShadow: 'var(--shadow-sm)',
        padding: 16,
        animationDelay: `${clip.id * 50}ms`,
        // Grid/flex children default to ``min-width: auto`` which
        // lets long unbreakable content (captions, hooks, focus
        // topics) push the card past its column width. Forcing
        // ``min-width: 0`` + clipping overflow keeps every child —
        // badges, captions, the score pill, action buttons — inside
        // the rounded border.
        minWidth: 0,
        overflow: 'hidden',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 12 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          {onSelect && (
            <input
              type="checkbox"
              checked={selected || false}
              onChange={(e) => onSelect(clip.id, e.target.checked)}
              style={{ accentColor: 'var(--accent-cyan)' }}
            />
          )}
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: 12, color: 'var(--text-secondary)' }}>
            CLIP {clip.id}
          </span>
          {isBest && (
            <span className="badge badge-amber">BEST</span>
          )}
        </div>
        <div style={{ display: 'flex', alignItems: 'flex-start', gap: 8 }}>
          <div style={{ textAlign: 'right' }}>
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: 20, fontWeight: 700, color: isFocusClip ? 'var(--success)' : scoreColor }}>
              {typeof primaryScore === 'number' ? primaryScore : String(primaryScore ?? '')}
            </div>
            <div style={{ fontSize: 10, color: isFocusClip ? 'var(--success)' : 'var(--text-secondary)', textTransform: 'uppercase' }}>
              {isFocusClip ? (tier ? `${tier} MATCH` : 'RELEVANCE') : '/100'}
            </div>
            {isFocusClip && compositeScore != null && compositeScore !== primaryScore && (
              <div
                title="Composite virality score (genre-weighted 4-axis)"
                style={{ fontSize: 9, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', marginTop: 2 }}
              >
                virality: {compositeScore}
              </div>
            )}
          </div>
          {onDelete && (
            <button
              className="clip-delete-btn"
              onClick={async (e) => {
                e.stopPropagation();
                if (!window.confirm(`Delete clip ${clip.id} "${clip.title}"?`)) return;
                setDeleting(true);
                try { await onDelete(clip); } finally { setDeleting(false); }
              }}
              disabled={deleting}
              title="Delete clip"
              style={{
                padding: '4px 6px',
                background: 'transparent',
                color: 'var(--danger, #ff3b30)',
                border: '1px solid transparent',
                borderRadius: 'var(--radius-sm)',
                fontSize: 13,
                lineHeight: 1,
                cursor: deleting ? 'default' : 'pointer',
                opacity: 0,
                flexShrink: 0,
                transition: 'opacity 0.15s, border-color 0.15s',
              }}
            >
              {deleting ? '...' : '\u{1F5D1}'}
            </button>
          )}
        </div>
      </div>

      {editingTitle ? (
        <div style={{ display: 'flex', gap: 6, alignItems: 'center', marginBottom: 8 }}>
          <input
            type="text"
            value={titleText}
            onChange={(e) => setTitleText(e.target.value)}
            onKeyDown={(e) => { if (e.key === 'Enter') saveTitleEdit(); if (e.key === 'Escape') setEditingTitle(false); }}
            autoFocus
            style={{
              flex: 1, padding: '4px 8px', fontSize: 14, fontWeight: 600,
              background: 'var(--bg-base)', color: 'var(--text-primary)',
              border: '1px solid var(--accent-cyan)', borderRadius: 'var(--radius-sm)',
              outline: 'none', lineHeight: 1.3, minWidth: 0,
            }}
          />
          <button
            onClick={saveTitleEdit}
            disabled={savingTitle}
            style={{
              padding: '3px 8px', fontSize: 10, fontWeight: 600,
              background: 'var(--accent-cyan)', color: 'var(--bg-base)',
              border: 'none', borderRadius: 'var(--radius-sm)', cursor: 'pointer',
              whiteSpace: 'nowrap',
            }}
          >
            {savingTitle ? '...' : 'Save'}
          </button>
          <button
            onClick={() => setEditingTitle(false)}
            style={{
              padding: '3px 8px', fontSize: 10,
              background: 'none', color: 'var(--text-muted)',
              border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)', cursor: 'pointer',
            }}
          >
            Cancel
          </button>
        </div>
      ) : (
        <h4
          onClick={jobId && onTimesChanged ? openTitleEdit : undefined}
          title={jobId && onTimesChanged ? 'Click to edit title' : String(clip.title || '')}
          style={{
            fontSize: 14, marginBottom: 8, lineHeight: 1.3,
            cursor: jobId && onTimesChanged ? 'pointer' : 'default',
            borderBottom: '1px dashed transparent',
            transition: 'border-color 0.15s',
            maxWidth: '100%',
            overflowWrap: 'anywhere',
            wordBreak: 'break-word',
          }}
          onMouseEnter={(e) => { if (jobId && onTimesChanged) e.currentTarget.style.borderBottomColor = 'var(--accent-cyan)'; }}
          onMouseLeave={(e) => { e.currentTarget.style.borderBottomColor = 'transparent'; }}
        >
          {String(clip.title || '')}
        </h4>
      )}

      {/* Time display / editor */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 12, flexWrap: 'wrap', alignItems: 'center' }}>
        {editingTimes ? (
          <>
            <input
              type="text"
              value={startText}
              onChange={(e) => setStartText(e.target.value)}
              onKeyDown={(e) => { if (e.key === 'Enter') saveTimeEdit(); if (e.key === 'Escape') cancelTimeEdit(); }}
              placeholder="0:00"
              style={{
                width: 52, padding: '3px 6px', fontSize: 12,
                fontFamily: 'var(--font-mono)', background: 'var(--bg-base)',
                border: '1px solid var(--accent-cyan)', borderRadius: 'var(--radius-sm)',
                color: 'var(--accent-cyan)', textAlign: 'center', outline: 'none',
              }}
            />
            <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>&rarr;</span>
            <input
              type="text"
              value={endText}
              onChange={(e) => setEndText(e.target.value)}
              onKeyDown={(e) => { if (e.key === 'Enter') saveTimeEdit(); if (e.key === 'Escape') cancelTimeEdit(); }}
              placeholder="0:00"
              style={{
                width: 52, padding: '3px 6px', fontSize: 12,
                fontFamily: 'var(--font-mono)', background: 'var(--bg-base)',
                border: '1px solid var(--accent-cyan)', borderRadius: 'var(--radius-sm)',
                color: 'var(--accent-cyan)', textAlign: 'center', outline: 'none',
              }}
            />
            <button
              onClick={saveTimeEdit}
              disabled={savingTimes}
              style={{
                padding: '2px 8px', fontSize: 10, fontWeight: 600,
                background: 'var(--accent-cyan)', color: 'var(--bg-base)',
                border: 'none', borderRadius: 'var(--radius-sm)', cursor: 'pointer',
              }}
            >
              {savingTimes ? '...' : 'Save'}
            </button>
            <button
              onClick={cancelTimeEdit}
              style={{
                padding: '2px 8px', fontSize: 10,
                background: 'none', color: 'var(--text-muted)',
                border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)', cursor: 'pointer',
              }}
            >
              Cancel
            </button>
          </>
        ) : (
          <>
            <span
              onClick={jobId && onTimesChanged ? openTimeEdit : undefined}
              title={jobId && onTimesChanged ? 'Click to edit clip times' : undefined}
              style={{
                fontFamily: 'var(--font-mono)', fontSize: 12, color: 'var(--accent-cyan)',
                cursor: jobId && onTimesChanged ? 'pointer' : 'default',
                padding: '1px 4px', borderRadius: 'var(--radius-sm)',
                transition: 'background 0.15s',
              }}
              onMouseEnter={(e) => { if (jobId && onTimesChanged) e.currentTarget.style.background = 'var(--bg-elevated)'; }}
              onMouseLeave={(e) => { e.currentTarget.style.background = 'transparent'; }}
            >
              {formatDuration(clip.start_time)} &rarr; {formatDuration(clip.end_time)}
            </span>
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: 12, color: 'var(--text-secondary)' }}>
              ({formatDuration(clip.duration)})
            </span>
          </>
        )}
        {!editingTimes && (
          <>
            <span className={`badge ${clip.platform === 'tiktok' ? 'badge-cyan' : clip.platform === 'youtube_shorts' ? 'badge-red' : 'badge-gray'}`}>
              {String(clip.platform || '').replace('_', ' ')}
            </span>
            <span className="badge badge-gray">{String(clip.clip_type || '')}</span>
            {clip.clip_focus && (
              <span
                className="badge"
                title={`Focus: ${String(clip.clip_focus || '')}`}
                style={{
                  background: 'var(--success-dim, rgba(52,199,89,0.12))',
                  color: 'var(--success)',
                  border: '1px solid var(--success)',
                }}
              >
                Focus: {String(clip.clip_focus || '')}
              </span>
            )}
            {/* Focus tier badge — ported from ViralClips.jsx per Bug 5 */}
            {clip.focus_relevance != null && (
              <span className="badge" style={{
                background: tierBg,
                color: tierColor,
                border: `1px solid ${tierColor}`,
              }}>
                Relevance: {clip.focus_relevance}/100
              </span>
            )}
            {/* Task 6 — visual verification badge. The orchestrator
                writes score_diagnostics.visual_verification with
                {pre_score, post_score, delta, note} after running
                apply_visual_verification. */}
            {(() => {
              const vv = diag.visual_verification;
              if (!vv || vv.note === 'skipped') return null;
              const isUp = (vv.delta || 0) >= 0;
              const color = isUp ? 'var(--success, #34c759)' : 'var(--warning, #ff9f0a)';
              const symbol = isUp ? '✓' : '↓';
              const label = isUp ? 'Verified' : 'Verified';
              const tip = (
                `Visual verifier: ${vv.pre_score} → ${vv.post_score} (${vv.delta >= 0 ? '+' : ''}${vv.delta})`
              );
              return (
                <span
                  className="badge"
                  title={tip}
                  style={{
                    background: 'transparent',
                    color,
                    border: `1px solid ${color}`,
                  }}
                >
                  {symbol} {label}
                </span>
              );
            })()}
          </>
        )}
      </div>

      {clipSubject && (
        <div style={{
          fontSize: 11, color: 'var(--text-muted)', marginBottom: 10,
          padding: '6px 8px', background: 'var(--bg-elevated)',
          borderRadius: 'var(--radius-sm)', borderLeft: '2px solid var(--accent-cyan)',
          lineHeight: 1.4,
        }}>
          <span style={{ fontWeight: 600, color: 'var(--text-secondary)', fontSize: 10, textTransform: 'uppercase', letterSpacing: '0.03em' }}>Detected subject: </span>
          {clipSubject.length > 120 ? clipSubject.slice(0, 120) + '...' : clipSubject}
        </div>
      )}

      <div
        style={{
          fontSize: 12, color: 'var(--text-secondary)', marginBottom: 8,
          overflowWrap: 'anywhere', wordBreak: 'break-word',
        }}
      >
        <div style={{ marginBottom: 4 }}>
          <strong style={{ color: 'var(--text-primary)' }}>Caption:</strong> {String(clip.suggested_caption || '')}
        </div>
        <div style={{ marginBottom: 4 }}>
          <strong style={{ color: 'var(--text-primary)' }}>Hook:</strong> {String(clip.hook_text || '')}
        </div>
        <div style={{ marginBottom: 4 }}>
          <strong style={{ color: 'var(--text-primary)' }}>Why it works:</strong> {String(clip.why_this_works || '')}
        </div>
        {clip.viral_score_reasoning && (
          <div>
            <strong style={{ color: 'var(--text-primary)' }}>Score reasoning:</strong> {String(clip.viral_score_reasoning || '')}
          </div>
        )}
      </div>

      {/* Score Breakdown panel (Enhancement 1 + Bug 5 in the clip-focus audit) */}
      {hasAnyAxis && (
        <div style={{ marginTop: 8, marginBottom: 8 }}>
          <button
            type="button"
            onClick={() => setBreakdownOpen((v) => !v)}
            aria-expanded={breakdownOpen}
            aria-controls={`clip-breakdown-${clip.id}`}
            style={{
              fontSize: 10, color: 'var(--text-muted)',
              background: 'transparent', border: 'none', padding: 0,
              cursor: 'pointer', textTransform: 'uppercase', letterSpacing: '0.05em',
              fontFamily: 'var(--font-mono)',
            }}
          >
            {breakdownOpen ? '▼' : '▶'} Score Breakdown
            {legacyFill && (
              <span style={{ marginLeft: 6, color: 'var(--accent-amber)' }} title="Axis scores estimated from legacy composite">
                (legacy)
              </span>
            )}
          </button>
          {breakdownOpen && (
            <div id={`clip-breakdown-${clip.id}`} style={{
              marginTop: 8, padding: 10,
              background: 'var(--bg-elevated)',
              borderRadius: 'var(--radius-sm)',
              border: '1px solid var(--border)',
            }}>
              {isFocusClip && clip.focus_relevance != null && (
                <div style={{ marginBottom: 10 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, marginBottom: 3, color: 'var(--success)' }}>
                    <span style={{ fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em' }}>Relevance</span>
                    <span style={{ fontFamily: 'var(--font-mono)' }}>{clip.focus_relevance}/100</span>
                  </div>
                  <div style={{ height: 6, background: 'rgba(52,199,89,0.15)', borderRadius: 3, overflow: 'hidden' }}>
                    <div style={{
                      width: `${Math.max(0, Math.min(100, clip.focus_relevance))}%`,
                      height: '100%', background: 'var(--success)',
                    }} />
                  </div>
                </div>
              )}
              {['hook', 'flow', 'value', 'trend'].map((axis) => {
                const score = axisScores[axis] || 0;
                const weight = weights[axis] || 0;
                const contribution = Math.round(score * weight * 10) / 10;
                const reason = axisReasons[axis];
                return (
                  <div key={axis} style={{ marginBottom: 8, opacity: legacyFill ? 0.55 : 1 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, marginBottom: 3, color: 'var(--text-secondary)' }}>
                      <span style={{ fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                        {axis}
                        <span style={{ color: 'var(--text-muted)', marginLeft: 4, fontWeight: 400 }}>
                          ({Math.round(weight * 100)}%)
                        </span>
                      </span>
                      <span style={{ fontFamily: 'var(--font-mono)' }}>
                        {score} <span style={{ color: 'var(--text-muted)' }}>= {contribution}</span>
                      </span>
                    </div>
                    <div style={{
                      height: 6,
                      background: 'var(--bg-panel)',
                      borderRadius: 3,
                      overflow: 'hidden',
                      backgroundImage: legacyFill ? 'repeating-linear-gradient(45deg, var(--bg-panel), var(--bg-panel) 4px, var(--border) 4px, var(--border) 8px)' : undefined,
                    }}>
                      <div style={{
                        width: `${Math.max(0, Math.min(100, score))}%`,
                        height: '100%',
                        background: 'var(--accent-cyan)',
                      }} />
                    </div>
                    {reason && (
                      <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 3, lineHeight: 1.4 }}>
                        {String(reason).slice(0, 140)}
                      </div>
                    )}
                  </div>
                );
              })}
              <div style={{ fontSize: 9, color: 'var(--text-muted)', marginTop: 6, fontFamily: 'var(--font-mono)' }}>
                Composite via {contentType || 'generic'} weights
                {legacyFill && ' (legacy fill)'}
                {!isFocusClip && compositeScore != null && ` = ${compositeScore}`}
                {isFocusClip && compositeScore != null && ` (virality: ${compositeScore})`}
              </div>
            </div>
          )}
        </div>
      )}

      <div style={{ display: 'flex', gap: 6, alignItems: 'stretch', marginTop: 12 }}>
        <button
          onClick={() => onPreview(clip)}
          style={{
            flex: '1 1 0%',
            padding: '8px 8px',
            background: 'var(--accent-cyan-dim)',
            color: 'var(--accent-cyan)',
            border: '1px solid var(--accent-cyan)',
            borderRadius: 'var(--radius-sm)',
            fontSize: 12,
            fontWeight: 600,
            whiteSpace: 'nowrap',
            minWidth: 0,
            overflow: 'hidden',
            textOverflow: 'ellipsis',
          }}
        >
          Preview
        </button>
        <div style={{ flex: '1 1 0%', position: 'relative', minWidth: 0, display: 'flex' }}>
          <div style={{ display: 'flex', flex: 1, minWidth: 0 }}>
            <button
              onClick={() => handleExport()}
              disabled={exporting}
              style={{
                flex: 1,
                padding: '8px 8px',
                background: exporting ? 'var(--bg-elevated)' : 'var(--accent-cyan)',
                color: exporting ? 'var(--text-secondary)' : 'var(--bg-base)',
                border: 'none',
                borderRadius: 'var(--radius-sm) 0 0 var(--radius-sm)',
                fontSize: 12,
                fontWeight: 600,
                whiteSpace: 'nowrap',
                minWidth: 0,
                overflow: 'hidden',
                textOverflow: 'ellipsis',
              }}
            >
              {exporting ? 'Exporting...' : `Export ${exportQuality.toUpperCase()}`}
            </button>
            <button
              onClick={() => setQualityMenuOpen(!qualityMenuOpen)}
              disabled={exporting}
              style={{
                padding: '8px 6px',
                background: exporting ? 'var(--bg-elevated)' : 'var(--accent-cyan)',
                color: exporting ? 'var(--text-secondary)' : 'var(--bg-base)',
                border: 'none',
                borderLeft: '1px solid rgba(0,0,0,0.15)',
                borderRadius: '0 var(--radius-sm) var(--radius-sm) 0',
                fontSize: 10,
                cursor: exporting ? 'default' : 'pointer',
                flexShrink: 0,
              }}
            >
              ▼
            </button>
          </div>
          {qualityMenuOpen && (
            <div style={{
              position: 'absolute', bottom: '100%', left: 0, right: 0,
              marginBottom: 2, background: 'var(--bg-elevated)',
              border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)',
              zIndex: 20, overflow: 'hidden',
              boxShadow: 'var(--shadow-sm)',
            }}>
              {['720p', '1080p', '4k'].map((q) => (
                <button
                  key={q}
                  onClick={() => handleExport(q)}
                  style={{
                    display: 'block', width: '100%', padding: '6px 10px',
                    background: q === exportQuality ? 'var(--accent-cyan)' : 'transparent',
                    color: q === exportQuality ? 'var(--bg-base)' : 'var(--text-primary)',
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
      {/* "Find more like this" per-clip action (Enhancement 7) */}
      {onFindMoreLikeThis && (
        <div style={{ marginTop: 6 }}>
          <button
            onClick={() => onFindMoreLikeThis(clip)}
            title="Run a new focus search using this clip's subject/focus as the query"
            style={{
              padding: '4px 10px', fontSize: 10, fontWeight: 600,
              background: 'transparent',
              color: 'var(--text-secondary)',
              border: '1px dashed var(--border)',
              borderRadius: 'var(--radius-sm)', cursor: 'pointer',
            }}
          >
            ✨ Find more like this
          </button>
        </div>
      )}
    </div>
  );
}
