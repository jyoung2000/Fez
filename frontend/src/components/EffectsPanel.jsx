import React, { useCallback } from 'react';
import useTimelineStore from '../stores/timelineStore';
import { undoCoalesceHandlers } from '../utils/undoCoalesce';

const EFFECT_CONTROLS = [
  { key: 'brightness', label: 'Brightness', min: -100, max: 100, step: 1, unit: '' },
  { key: 'contrast', label: 'Contrast', min: -100, max: 100, step: 1, unit: '' },
  { key: 'saturation', label: 'Saturation', min: -100, max: 100, step: 1, unit: '' },
  { key: 'blur', label: 'Blur', min: 0, max: 20, step: 0.5, unit: 'px' },
  { key: 'hueRotate', label: 'Hue Rotate', min: 0, max: 360, step: 1, unit: '°' },
  { key: 'sepia', label: 'Sepia', min: 0, max: 100, step: 1, unit: '%' },
];

export default function EffectsPanel({ compact = false }) {
  const selectedItemId = useTimelineStore((s) => s.selectedItemId);
  const items = useTimelineStore((s) => s.items);
  const updateItem = useTimelineStore((s) => s.updateItem);

  const item = items.find((i) => i.id === selectedItemId) || null;

  const updateEffect = useCallback((key, value) => {
    if (!item) return;
    const effects = { ...(item.effects || {}), [key]: value };
    updateItem(item.id, { effects });
  }, [item, updateItem]);

  const resetAll = useCallback(() => {
    if (!item) return;
    updateItem(item.id, {
      effects: { brightness: 0, contrast: 0, saturation: 0, blur: 0, hueRotate: 0, sepia: 0 },
    });
  }, [item, updateItem]);

  if (!item || (item.type !== 'video' && item.type !== 'image' && item.type !== 'overlay')) {
    return (
      <div className="ve-effects ve-effects--empty">
        <span className="ve-effects__placeholder">
          Select a video or image clip to adjust effects
        </span>
      </div>
    );
  }

  const effects = item.effects || {};

  return (
    <div className={`ve-effects${compact ? ' ve-effects--compact' : ''}`}>
      <div className="ve-effects__header">
        <span className="ve-effects__title">Effects</span>
        <button
          className="ve-effects__reset"
          onClick={resetAll}
          title="Reset all effects"
        >
          Reset All
        </button>
      </div>

      {EFFECT_CONTROLS.map((ctrl) => {
        const value = effects[ctrl.key] ?? (ctrl.key === 'brightness' || ctrl.key === 'contrast' || ctrl.key === 'saturation' ? 0 : 0);
        return (
          <div key={ctrl.key} className="ve-effects__control">
            <div className="ve-effects__control-header">
              <label className="ve-effects__label">{ctrl.label}</label>
              <span className="ve-effects__value">
                {value}{ctrl.unit}
              </span>
            </div>
            <input
              type="range"
              min={ctrl.min}
              max={ctrl.max}
              step={ctrl.step}
              value={value}
              onChange={(e) => updateEffect(ctrl.key, parseFloat(e.target.value))}
              {...undoCoalesceHandlers()}
              className="ve-effects__slider"
              style={{
                background: `linear-gradient(to right, var(--ve-accent, #0A84FF) ${
                  ((value - ctrl.min) / (ctrl.max - ctrl.min)) * 100
                }%, var(--ve-slider-track, rgba(0,0,0,0.12)) ${
                  ((value - ctrl.min) / (ctrl.max - ctrl.min)) * 100
                }%)`,
              }}
            />
          </div>
        );
      })}

      {/* Opacity (applies to all visual items) */}
      <div className="ve-effects__control">
        <div className="ve-effects__control-header">
          <label className="ve-effects__label">Opacity</label>
          <span className="ve-effects__value">{Math.round((item.opacity ?? 1) * 100)}%</span>
        </div>
        <input
          type="range"
          min="0"
          max="1"
          step="0.01"
          value={item.opacity ?? 1}
          onChange={(e) => updateItem(item.id, { opacity: parseFloat(e.target.value) })}
          {...undoCoalesceHandlers()}
          className="ve-effects__slider"
          style={{
            background: `linear-gradient(to right, var(--ve-accent, #0A84FF) ${
              (item.opacity ?? 1) * 100
            }%, var(--ve-slider-track, rgba(0,0,0,0.12)) ${
              (item.opacity ?? 1) * 100
            }%)`,
          }}
        />
      </div>
    </div>
  );
}
