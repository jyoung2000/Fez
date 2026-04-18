import React, { useMemo } from 'react';

/**
 * ReframeDebugOverlay — shows pacing, min_hold, confidence, and fallback
 * reasons as a timeline strip below the preview player.
 *
 * Enabled via dev flag or keyboard shortcut. Reuses the RenderPlan debug
 * data — no new API needed.
 *
 * v2 Phase 10: surfaces content routing (content_type, clip_content_type,
 * anime_subtype, music_subtype, game_type, gameplay_subtype) in a header
 * chip row plus per-segment editorial-prior decision tags
 * (j_cut / l_cut / listener_hold / reaction_beat) under the strategy row.
 * These hint at which Phase 3-8 tuning paths fired on this clip so an
 * editor debugging a bad crop can attribute it to the right layer.
 *
 * Props:
 *   renderPlan: RenderPlan JSON from /api/jobs/{id}/render_plan?debug=1
 *   currentTime: current playback position (seconds, clip-relative)
 *   clipDuration: total clip duration in seconds
 */
export default function ReframeDebugOverlay({ renderPlan, currentTime = 0, clipDuration = 0 }) {
  const debug = renderPlan?.debug;
  const ops = renderPlan?.ops;

  if (!debug && !ops) return null;

  const duration = clipDuration || renderPlan?.total_duration_sec || 1;

  // v2 Phase 10: content routing identifiers surfaced via RenderPlan.debug
  // (populated by the parity harness + pipeline.py when the v2 flags are
  // on). Gracefully handle legacy RenderPlans that don't carry these
  // fields — the header chips just render nothing.
  const contentType = debug?.content_type;
  const clipContentType = debug?.clip_content_type;
  const animeSubtype = debug?.anime_subtype;
  const musicSubtype = debug?.music_subtype;
  const gameplaySubtype = debug?.gameplay_subtype;
  const gameType = debug?.game_type;
  const isMultiSpeakerPanel = debug?.is_multi_speaker_panel;
  const isAnimated = debug?.is_animated;
  const perSegmentEditorial = debug?.editorial_prior_per_segment;
  const perSegmentReason = debug?.reason_per_segment;

  // Strategy color map
  const strategyColors = {
    crop: '#3b82f6',           // blue
    tracking_crop: '#8b5cf6',  // purple
    wide_master: '#f59e0b',    // amber
    blur_fill: '#6366f1',      // indigo
    split_screen: '#10b981',   // emerald
    stacked_gameplay: '#14b8a6', // teal
    grid_2x2: '#06b6d4',      // cyan
  };

  // Per-content-type chip color (v2 Phase 10)
  const contentTypeColors = {
    talking_head: '#60a5fa',
    multi_speaker_panel: '#34d399',
    cinematic_dialogue: '#a78bfa',
    animation: '#f472b6',
    animation_dialogue: '#f472b6',
    music_video: '#fb7185',
    gameplay: '#fbbf24',
    gameplay_moba: '#fbbf24',
    gameplay_tps: '#fbbf24',
    gameplay_racing: '#fbbf24',
    stream: '#14b8a6',
    generic: '#9ca3af',
  };

  // Editorial decision tag color (v2 Phase 10)
  const editorialColors = {
    j_cut: '#60a5fa',           // blue
    l_cut: '#a78bfa',           // purple
    listener_hold: '#34d399',   // green
    reaction_beat: '#fb7185',   // rose
  };

  // Confidence color
  const confColor = (conf) => {
    if (conf >= 0.7) return '#22c55e';  // green
    if (conf >= 0.5) return '#f59e0b';  // amber
    if (conf >= 0.3) return '#ef4444';  // red
    return '#991b1b';                   // dark red
  };

  // Pacing sparkline
  const pacingSvg = useMemo(() => {
    const pacing = debug?.pacing_per_sec;
    if (!pacing || pacing.length === 0) return null;

    const w = 600;
    const h = 30;
    const points = pacing.map((v, i) => {
      const x = (i / Math.max(1, pacing.length - 1)) * w;
      const y = h - v * h;
      return `${x},${y}`;
    }).join(' ');

    return (
      <svg width="100%" height={h} viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none"
           style={{ display: 'block' }}>
        <polyline points={points} fill="none" stroke="#60a5fa" strokeWidth="1.5" />
        {/* Playhead */}
        <line x1={(currentTime / duration) * w} y1={0}
              x2={(currentTime / duration) * w} y2={h}
              stroke="white" strokeWidth="1" opacity="0.6" />
      </svg>
    );
  }, [debug?.pacing_per_sec, currentTime, duration]);

  // MinHold sparkline
  const minHoldSvg = useMemo(() => {
    const minHold = debug?.min_hold_per_sec;
    if (!minHold || minHold.length === 0) return null;

    const w = 600;
    const h = 20;
    const maxHold = 3.0;
    const points = minHold.map((v, i) => {
      const x = (i / Math.max(1, minHold.length - 1)) * w;
      const y = h - (v / maxHold) * h;
      return `${x},${y}`;
    }).join(' ');

    return (
      <svg width="100%" height={h} viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none"
           style={{ display: 'block' }}>
        <polyline points={points} fill="none" stroke="#fbbf24" strokeWidth="1.5" />
      </svg>
    );
  }, [debug?.min_hold_per_sec]);

  // v2 Phase 10: content-routing chip row. Renders a single-line
  // summary of which tuning layer picked the crop — useful when an
  // editor is debugging why a clip ends up in wide_master or split.
  const hasContentRouting = (
    contentType || clipContentType || animeSubtype || musicSubtype
    || gameplaySubtype || gameType
  );

  const chip = (label, value, color) => (
    <span style={{
      padding: '1px 5px',
      borderRadius: 3,
      background: `${color}22`,
      color: color,
      fontSize: 9,
      fontWeight: 600,
      border: `1px solid ${color}55`,
      whiteSpace: 'nowrap',
    }}>
      {label}:{' '}
      <span style={{ color: '#e5e7eb', fontWeight: 400 }}>{value}</span>
    </span>
  );

  return (
    <div style={{
      width: '100%',
      background: 'rgba(0, 0, 0, 0.85)',
      borderTop: '1px solid rgba(255, 255, 255, 0.1)',
      padding: '6px 8px',
      fontSize: 10,
      color: '#9ca3af',
      fontFamily: 'monospace',
    }}>
      {/* Content routing chips (v2 Phase 10) */}
      {hasContentRouting && (
        <div style={{
          display: 'flex',
          gap: 4,
          flexWrap: 'wrap',
          marginBottom: 5,
          alignItems: 'center',
        }}>
          {contentType && chip(
            'content',
            contentType,
            contentTypeColors[contentType] || '#9ca3af',
          )}
          {clipContentType && clipContentType !== contentType && chip(
            'clip',
            clipContentType,
            contentTypeColors[clipContentType] || '#9ca3af',
          )}
          {animeSubtype && chip('anime', animeSubtype, '#f472b6')}
          {musicSubtype && chip('music', musicSubtype, '#fb7185')}
          {gameplaySubtype && chip('genre', gameplaySubtype, '#fbbf24')}
          {gameType && chip('game', gameType, '#fbbf24')}
          {isMultiSpeakerPanel && chip('panel', 'yes', '#34d399')}
          {isAnimated && chip('animated', 'yes', '#f472b6')}
        </div>
      )}

      {/* Pacing score line chart */}
      {pacingSvg && (
        <div style={{ marginBottom: 4 }}>
          <span style={{ color: '#60a5fa', marginRight: 8 }}>pacing</span>
          {pacingSvg}
        </div>
      )}

      {/* MinHold line chart */}
      {minHoldSvg && (
        <div style={{ marginBottom: 4 }}>
          <span style={{ color: '#fbbf24', marginRight: 8 }}>min_hold</span>
          {minHoldSvg}
        </div>
      )}

      {/* Strategy chips */}
      {ops && ops.length > 0 && (
        <div style={{ display: 'flex', gap: 1, marginBottom: 3, height: 14 }}>
          {ops.map((op, i) => {
            const widthPct = ((op.end_sec - op.start_sec) / duration) * 100;
            return (
              <div key={i} style={{
                width: `${widthPct}%`,
                minWidth: 2,
                height: '100%',
                background: strategyColors[op.kind] || '#4b5563',
                borderRadius: 2,
                overflow: 'hidden',
                fontSize: 8,
                color: 'white',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                whiteSpace: 'nowrap',
              }}>
                {widthPct > 5 ? op.kind.replace('_', ' ') : ''}
              </div>
            );
          })}
        </div>
      )}

      {/* Per-segment editorial decision tags (v2 Phase 10) */}
      {perSegmentEditorial && perSegmentEditorial.length > 0 && ops && (
        <div style={{
          display: 'flex',
          gap: 1,
          marginBottom: 3,
          height: 12,
        }}>
          {ops.map((op, i) => {
            const widthPct = ((op.end_sec - op.start_sec) / duration) * 100;
            const kinds = perSegmentEditorial[i] || [];
            if (kinds.length === 0) {
              return (
                <div key={i} style={{
                  width: `${widthPct}%`,
                  minWidth: 2,
                  height: '100%',
                }} />
              );
            }
            // Show the first kind as the background; tooltip shows all
            const primary = kinds[0];
            return (
              <div key={i} style={{
                width: `${widthPct}%`,
                minWidth: 2,
                height: '100%',
                background: editorialColors[primary] || '#6b7280',
                borderRadius: 1,
                fontSize: 7,
                color: 'white',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                whiteSpace: 'nowrap',
                opacity: 0.85,
              }}
              title={`#${i}: ${kinds.join(', ')}`}>
                {widthPct > 4 ? primary.replace('_', ' ').slice(0, 8) : ''}
              </div>
            );
          })}
        </div>
      )}

      {/* Confidence bars */}
      {debug?.confidence_per_segment && debug.confidence_per_segment.length > 0 && ops && (
        <div style={{ display: 'flex', gap: 1, marginBottom: 3, height: 10 }}>
          {ops.map((op, i) => {
            const widthPct = ((op.end_sec - op.start_sec) / duration) * 100;
            const conf = debug.confidence_per_segment[i] ?? 1.0;
            return (
              <div key={i} style={{
                width: `${widthPct}%`,
                minWidth: 2,
                height: '100%',
                background: confColor(conf),
                borderRadius: 1,
                opacity: 0.8,
              }}
              title={`confidence: ${conf.toFixed(2)}`}
              />
            );
          })}
        </div>
      )}

      {/* Fallback reason tags */}
      {debug?.fallback_reasons && debug.fallback_reasons.some(r => r) && (
        <div style={{ display: 'flex', gap: 2, flexWrap: 'wrap', marginTop: 2 }}>
          {debug.fallback_reasons.map((reason, i) =>
            reason ? (
              <span key={i} style={{
                padding: '1px 4px',
                borderRadius: 3,
                background: 'rgba(239, 68, 68, 0.2)',
                color: '#fca5a5',
                fontSize: 9,
              }}>
                #{i}: {reason}
              </span>
            ) : null
          )}
        </div>
      )}

      {/* Per-segment reason tags (v2 Phase 10) —
          shows anime_anchor_face / gameplay_tracker_motion /
          multi_region_lp_fit / music_pulse_cut / editorial_reaction_beat */}
      {perSegmentReason && perSegmentReason.some(r => r) && (
        <div style={{
          display: 'flex',
          gap: 2,
          flexWrap: 'wrap',
          marginTop: 2,
          fontSize: 8,
        }}>
          {perSegmentReason.map((reason, i) =>
            reason ? (
              <span key={`r${i}`} style={{
                padding: '1px 4px',
                borderRadius: 3,
                background: 'rgba(96, 165, 250, 0.15)',
                color: '#93c5fd',
                fontSize: 8,
                border: '1px solid rgba(96, 165, 250, 0.25)',
              }}
              title={`segment #${i} reason`}>
                #{i}: {reason}
              </span>
            ) : null
          )}
        </div>
      )}
    </div>
  );
}
