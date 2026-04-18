import React, { useCallback } from 'react';
import useTimelineStore from '../stores/timelineStore';
import { TRANSITION_RANGES } from '../utils/defaultSettings';

const TRANSITION_TYPES = [
  { id: 'none', label: 'None', icon: '—' },
  { id: 'dissolve', label: 'Dissolve', icon: '◐' },
  { id: 'fade', label: 'Fade', icon: '◑' },
  { id: 'wipe-left', label: 'Wipe Left', icon: '◧' },
  { id: 'wipe-right', label: 'Wipe Right', icon: '◨' },
  { id: 'slide-left', label: 'Slide Left', icon: '⇠' },
  { id: 'slide-right', label: 'Slide Right', icon: '⇢' },
  { id: 'zoom', label: 'Zoom', icon: '⊕' },
];

// Pulled from utils/defaultSettings so the picker, the properties
// panel, and any future bulk-edit dialog all expose the same range.
const DURATION_PRESETS = TRANSITION_RANGES.durationPresets;

export default function TransitionPicker({ itemId, onClose }) {
  const items = useTimelineStore((s) => s.items);
  const updateItem = useTimelineStore((s) => s.updateItem);

  const item = items.find((i) => i.id === itemId) || null;

  const setTransition = useCallback((type) => {
    if (!item) return;
    if (type === 'none') {
      updateItem(item.id, { transition: null });
    } else {
      const current = item.transition || {};
      updateItem(item.id, {
        transition: { type, duration: current.duration || 0.5 },
      });
    }
  }, [item, updateItem]);

  const setDuration = useCallback((dur) => {
    if (!item) return;
    const current = item.transition || { type: 'dissolve' };
    updateItem(item.id, {
      transition: { ...current, duration: dur },
    });
  }, [item, updateItem]);

  if (!item) return null;

  const currentType = item.transition?.type || 'none';
  const currentDuration = item.transition?.duration || 0.5;

  return (
    <div className="ve-transition-picker" onClick={(e) => e.stopPropagation()}>
      <div className="ve-transition-picker__header">
        <span className="ve-transition-picker__title">Transition</span>
        {onClose && (
          <button className="ve-transition-picker__close" onClick={onClose}>
            ✕
          </button>
        )}
      </div>

      <div className="ve-transition-picker__grid">
        {TRANSITION_TYPES.map((t) => (
          <button
            key={t.id}
            className={`ve-transition-picker__item${currentType === t.id ? ' ve-transition-picker__item--active' : ''}`}
            onClick={() => setTransition(t.id)}
            title={t.label}
          >
            <span className="ve-transition-picker__icon">{t.icon}</span>
            <span className="ve-transition-picker__label">{t.label}</span>
          </button>
        ))}
      </div>

      {currentType !== 'none' && (
        <div className="ve-transition-picker__duration">
          <span className="ve-transition-picker__dur-label">Duration</span>
          <div className="ve-transition-picker__dur-pills">
            {DURATION_PRESETS.map((d) => (
              <button
                key={d}
                className={`ve-transition-picker__dur-pill${
                  Math.abs(currentDuration - d) < 0.01 ? ' ve-transition-picker__dur-pill--active' : ''
                }`}
                onClick={() => setDuration(d)}
              >
                {d}s
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
