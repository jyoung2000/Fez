/**
 * RetentionCurve — Mini audience retention chart for clip cards.
 *
 * Displays a predicted viewer retention curve as a small SVG sparkline
 * with overall score and factor breakdown on hover.
 */
import { useState } from 'react';

const CHART_WIDTH = 180;
const CHART_HEIGHT = 48;
const PADDING = 2;

function RetentionCurve({ retentionData }) {
  const [showDetails, setShowDetails] = useState(false);

  if (!retentionData || !retentionData.retention_curve || retentionData.retention_curve.length === 0) {
    return null;
  }

  const { retention_curve, overall_score, factors } = retentionData;

  // Build SVG path from retention curve data
  const points = retention_curve.map((point, i) => {
    const x = PADDING + (point.time_pct / 100) * (CHART_WIDTH - 2 * PADDING);
    const y = PADDING + ((100 - point.retention_pct) / 100) * (CHART_HEIGHT - 2 * PADDING);
    return { x, y };
  });

  const pathData = points
    .map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x.toFixed(1)} ${p.y.toFixed(1)}`)
    .join(' ');

  // Fill area under curve
  const fillData = pathData +
    ` L ${points[points.length - 1].x.toFixed(1)} ${CHART_HEIGHT - PADDING}` +
    ` L ${PADDING} ${CHART_HEIGHT - PADDING} Z`;

  // Color based on overall score
  const getColor = (score) => {
    if (score >= 75) return '#22c55e'; // green
    if (score >= 50) return '#eab308'; // yellow
    return '#ef4444'; // red
  };

  const color = getColor(overall_score);

  const factorLabels = {
    hook_strength: 'Hook',
    pacing_variance: 'Pacing',
    visual_dynamism: 'Visual',
    emotional_arc: 'Arc',
    standalone_comprehension: 'Standalone',
  };

  return (
    <div
      style={{
        position: 'relative',
        display: 'inline-block',
        cursor: 'pointer',
      }}
      onMouseEnter={() => setShowDetails(true)}
      onMouseLeave={() => setShowDetails(false)}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
        <svg
          width={CHART_WIDTH}
          height={CHART_HEIGHT}
          viewBox={`0 0 ${CHART_WIDTH} ${CHART_HEIGHT}`}
          style={{
            borderRadius: '4px',
            background: 'rgba(0,0,0,0.2)',
          }}
        >
          {/* Fill under curve */}
          <path
            d={fillData}
            fill={color}
            fillOpacity={0.15}
          />
          {/* Curve line */}
          <path
            d={pathData}
            fill="none"
            stroke={color}
            strokeWidth={1.5}
            strokeLinecap="round"
            strokeLinejoin="round"
          />
          {/* 50% reference line */}
          <line
            x1={PADDING}
            y1={CHART_HEIGHT / 2}
            x2={CHART_WIDTH - PADDING}
            y2={CHART_HEIGHT / 2}
            stroke="rgba(255,255,255,0.1)"
            strokeWidth={0.5}
            strokeDasharray="2,2"
          />
        </svg>
        <span
          style={{
            fontSize: '12px',
            fontWeight: 600,
            color: color,
            minWidth: '28px',
          }}
        >
          {overall_score}%
        </span>
      </div>

      {/* Tooltip with factor breakdown */}
      {showDetails && factors && (
        <div
          style={{
            position: 'absolute',
            bottom: '100%',
            left: '50%',
            transform: 'translateX(-50%)',
            marginBottom: '8px',
            background: '#1a1a2e',
            border: '1px solid rgba(255,255,255,0.15)',
            borderRadius: '8px',
            padding: '10px 12px',
            zIndex: 100,
            minWidth: '180px',
            boxShadow: '0 4px 12px rgba(0,0,0,0.4)',
          }}
        >
          <div style={{ fontSize: '11px', fontWeight: 600, marginBottom: '6px', color: '#fff' }}>
            Retention Prediction
          </div>
          {Object.entries(factors).map(([key, value]) => (
            <div
              key={key}
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                fontSize: '10px',
                color: 'rgba(255,255,255,0.7)',
                marginBottom: '3px',
              }}
            >
              <span>{factorLabels[key] || key}</span>
              <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                <div
                  style={{
                    width: '40px',
                    height: '4px',
                    background: 'rgba(255,255,255,0.1)',
                    borderRadius: '2px',
                    overflow: 'hidden',
                  }}
                >
                  <div
                    style={{
                      width: `${value}%`,
                      height: '100%',
                      background: getColor(value),
                      borderRadius: '2px',
                    }}
                  />
                </div>
                <span style={{ minWidth: '24px', textAlign: 'right' }}>{value}</span>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default RetentionCurve;
