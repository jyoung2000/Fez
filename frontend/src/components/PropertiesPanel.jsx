import React, { useCallback, useMemo, useState, useRef, useEffect } from 'react';
import useTimelineStore, { getMaxItemDuration } from '../stores/timelineStore';
import {
  SUBTITLE_RANGES,
  TRANSITION_RANGES,
  offsetVFromPosition,
  positionFromOffsetV,
} from '../utils/defaultSettings';
import {
  loadPresetsForType,
  savePreset,
  applyPreset,
  TYPE_LABELS,
} from '../utils/trackPresets';
import { undoCoalesceHandlers } from '../utils/undoCoalesce';

const SPEED_PRESETS = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 4.0];
const FONT_OPTIONS = [
  'DM Sans', 'Montserrat', 'Open Sans', 'Roboto', 'Poppins', 'Inter',
  'Nunito', 'Lato', 'Oswald', 'Playfair Display', 'Bebas Neue',
  'Liberation Sans', 'DejaVu Sans',
];

// Available font weights per font family.  Variable fonts support a full range;
// static fonts only have the weights for which a file exists on the system.
const ALL_WEIGHT_OPTIONS = [
  { value: 100, label: 'Thin' },
  { value: 200, label: 'ExtraLight' },
  { value: 300, label: 'Light' },
  { value: 400, label: 'Regular' },
  { value: 500, label: 'Medium' },
  { value: 600, label: 'SemiBold' },
  { value: 700, label: 'Bold' },
  { value: 800, label: 'ExtraBold' },
  { value: 900, label: 'Black' },
];
// Fonts that ship as variable fonts (support full 100-900 range)
const VARIABLE_FONTS = new Set([
  'DM Sans', 'Montserrat', 'Open Sans', 'Roboto', 'Inter', 'Nunito',
  'Oswald', 'Playfair Display', 'Lato',
]);
// Fonts that only have regular (400) and bold (700) static files
const STATIC_FONT_WEIGHTS = {
  'Poppins': [400, 700],
  'Bebas Neue': [400],
  'Liberation Sans': [400, 700],
  'Liberation Serif': [400, 700],
  'Liberation Mono': [400, 700],
  'DejaVu Sans': [400, 700],
  'DejaVu Serif': [400, 700],
  'DejaVu Sans Mono': [400, 700],
  'FreeSans': [400, 700],
};
function getWeightOptionsForFont(fontFamily) {
  if (VARIABLE_FONTS.has(fontFamily)) return ALL_WEIGHT_OPTIONS;
  const weights = STATIC_FONT_WEIGHTS[fontFamily];
  if (weights) return ALL_WEIGHT_OPTIONS.filter(w => weights.includes(w.value));
  // Custom/unknown fonts: show common weights (browser will synthesize)
  return ALL_WEIGHT_OPTIONS.filter(w => [400, 700].includes(w.value));
}
const TEXT_ANIMATIONS = [
  { id: 'none', label: 'None' },
  { id: 'fade-in', label: 'Fade In' },
  { id: 'typewriter', label: 'Typewriter' },
  { id: 'slide-up', label: 'Slide Up' },
  { id: 'pop', label: 'Pop' },
  { id: 'bounce', label: 'Bounce' },
];
const SHAPE_TYPES = [
  { id: 'rectangle', label: 'Rectangle', icon: '▬' },
  { id: 'circle', label: 'Circle', icon: '●' },
  { id: 'ellipse', label: 'Ellipse', icon: '⬮' },
  { id: 'arrow', label: 'Arrow', icon: '➜' },
  { id: 'line', label: 'Line', icon: '╱' },
];

function formatTime(s) {
  if (!s || isNaN(s) || s < 0) return '0:00.00';
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  const ms = Math.floor((s % 1) * 100);
  return `${m}:${sec.toString().padStart(2, '0')}.${ms.toString().padStart(2, '0')}`;
}

/* Reusable slider row with editable number input */
function SliderRow({ label, value, min, max, step = 1, unit = '', onChange }) {
  const [editing, setEditing] = useState(false);
  const [editVal, setEditVal] = useState('');
  const displayVal = typeof value === 'number' ? (Number.isInteger(value) ? String(value) : value.toFixed(1)) : String(value);

  const commitEdit = () => {
    setEditing(false);
    const parsed = parseFloat(editVal);
    if (!isNaN(parsed)) {
      onChange(Math.max(min, Math.min(max, parsed)));
    }
  };

  return (
    <div className="ve-properties__slider-row">
      <span className="ve-properties__field-label" style={{ minWidth: 55 }}>{label}</span>
      <input
        type="range" min={min} max={max} step={step} value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        {...undoCoalesceHandlers()}
        className="ve-properties__slider"
      />
      {editing ? (
        <input
          type="number"
          min={min} max={max} step={step}
          value={editVal}
          onChange={(e) => setEditVal(e.target.value)}
          onBlur={commitEdit}
          onKeyDown={(e) => { if (e.key === 'Enter') commitEdit(); if (e.key === 'Escape') setEditing(false); }}
          autoFocus
          className="ve-properties__slider-value ve-properties__slider-value--editing"
        />
      ) : (
        <span
          className="ve-properties__slider-value ve-properties__slider-value--clickable"
          onClick={() => { setEditVal(displayVal); setEditing(true); }}
          title="Click to type a value"
        >
          {displayVal}{unit}
        </span>
      )}
    </div>
  );
}

/* Reusable number field */
function NumField({ label, value, min, max, step = 1, onChange }) {
  return (
    <div className="ve-properties__field">
      <span className="ve-properties__field-label">{label}</span>
      <input
        type="number" min={min} max={max} step={step} value={value}
        onChange={(e) => onChange(parseFloat(e.target.value) || 0)}
        className="ve-properties__input"
      />
    </div>
  );
}

/* Reusable color field */
function ColorField({ label, value, onChange }) {
  return (
    <div className="ve-properties__field">
      <span className="ve-properties__field-label">{label}</span>
      <input
        type="color" value={value || '#FFFFFF'}
        onChange={(e) => onChange(e.target.value)}
        {...undoCoalesceHandlers()}
        className="ve-properties__color"
      />
    </div>
  );
}

/* ── SubtitleProperties — edits global SubtitleOverlay (Subs On) settings ── */
function SubtitleProperties({ item, update, settings, onSettingsChange, customFonts = [] }) {
  const s = settings || {};
  const set = (key, val) => onSettingsChange && onSettingsChange({ ...s, [key]: val });

  return (
    <>
      {/* Subtitle Text (per-item) */}
      <div className="ve-properties__section">
        <label className="ve-properties__label">Subtitle Text</label>
        <textarea
          value={item.subtitleText || ''}
          onChange={(e) => update('subtitleText', e.target.value)}
          className="ve-properties__textarea"
          rows={2}
          placeholder="Subtitle text..."
        />
        {item.speaker && (
          <div style={{ fontSize: 10, color: 'var(--ve-text-muted, #888)', marginTop: 4 }}>
            Speaker: {item.speaker}
          </div>
        )}
      </div>

      {/* Font Settings */}
      <div className="ve-properties__section">
        <label className="ve-properties__label">Subtitle Font</label>
        <div className="ve-properties__row">
          <div className="ve-properties__field" style={{ flex: 2 }}>
            <span className="ve-properties__field-label">Family</span>
            <select
              value={s.subtitleFont || 'DM Sans'}
              onChange={(e) => {
                const newFont = e.target.value;
                set('subtitleFont', newFont);
                const curWeight = typeof s.subtitleFontWeight === 'number' ? s.subtitleFontWeight : (s.subtitleFontWeight === 'bold' ? 700 : 400);
                const available = getWeightOptionsForFont(newFont).map(w => w.value);
                if (!available.includes(curWeight)) {
                  const closest = available.reduce((a, b) => Math.abs(b - curWeight) < Math.abs(a - curWeight) ? b : a);
                  set('subtitleFontWeight', closest);
                }
              }}
              className="ve-properties__select"
            >
              {FONT_OPTIONS.map(f => <option key={f} value={f}>{f}</option>)}
                  {customFonts.length > 0 && (
                    <optgroup label="Custom Fonts">
                      {customFonts.filter(f => f.name && !FONT_OPTIONS.includes(f.name)).map(f => (
                        <option key={f.name} value={f.name}>{f.name}</option>
                      ))}
                    </optgroup>
                  )}
            </select>
          </div>
          <div className="ve-properties__field">
            <span className="ve-properties__field-label">Weight</span>
            <select
              value={typeof s.subtitleFontWeight === 'number' ? s.subtitleFontWeight : (s.subtitleFontWeight === 'bold' ? 700 : s.subtitleFontWeight === 'black' ? 900 : 400)}
              onChange={(e) => set('subtitleFontWeight', parseInt(e.target.value))}
              className="ve-properties__select"
            >
              {getWeightOptionsForFont(s.subtitleFont || 'DM Sans').map(w => (
                <option key={w.value} value={w.value}>{w.label}</option>
              ))}
            </select>
          </div>
        </div>
        <div className="ve-properties__row">
          <div className="ve-properties__field">
            <span className="ve-properties__field-label">Size</span>
            <select
              value={s.subtitleSize || 'medium'}
              onChange={(e) => set('subtitleSize', e.target.value)}
              className="ve-properties__select"
            >
              <option value="small">Small</option>
              <option value="medium">Medium</option>
              <option value="large">Large</option>
            </select>
          </div>
          <ColorField label="Color" value={s.subtitleFontColor || '#FFFFFF'} onChange={(v) => set('subtitleFontColor', v)} />
        </div>
      </div>

      {/* Position — pills + numeric offset bind through the same
          source-of-truth helpers so they can never disagree. */}
      <div className="ve-properties__section">
        <label className="ve-properties__label">Position</label>
        <div className="ve-properties__speed-pills">
          {['top', 'center', 'bottom'].map(p => {
            // Drive the active pill from the canonical numeric offset
            // so a job loaded with only ``subtitleOffsetV`` set still
            // highlights the right pill.
            const activePill = positionFromOffsetV(s.subtitleOffsetV ?? offsetVFromPosition(s.subtitlePosition));
            return (
              <button
                key={p}
                className={`ve-properties__speed-pill${activePill === p ? ' ve-properties__speed-pill--active' : ''}`}
                onClick={() => {
                  // Update both fields atomically so save / reload
                  // round-trips don't flip the pill.
                  set('subtitlePosition', p);
                  set('subtitleOffsetV', offsetVFromPosition(p));
                }}
                style={{ fontSize: 10, textTransform: 'capitalize' }}
              >
                {p}
              </button>
            );
          })}
        </div>
        <SliderRow
          label="Max W"
          value={s.subtitleMaxWidth ?? 90}
          min={SUBTITLE_RANGES.maxWidth.min} max={SUBTITLE_RANGES.maxWidth.max} step={SUBTITLE_RANGES.maxWidth.step}
          unit="%"
          onChange={(v) => set('subtitleMaxWidth', v)}
        />
        <SliderRow
          label="Offset"
          value={s.subtitleOffsetV ?? 4}
          min={SUBTITLE_RANGES.offsetV.min} max={SUBTITLE_RANGES.offsetV.max} step={SUBTITLE_RANGES.offsetV.step}
          unit="%"
          onChange={(v) => {
            // Update both fields so the categorical pill recomputes
            // from the same numeric source.
            set('subtitleOffsetV', v);
            set('subtitlePosition', positionFromOffsetV(v));
          }}
        />
      </div>

      {/* Max Words Per Subtitle */}
      <div className="ve-properties__section">
        <label className="ve-properties__label">Max Words Per Subtitle</label>
        <div className="ve-properties__slider-row">
          <span className="ve-properties__field-label">Enabled</span>
          <button
            className={`ve-properties__speed-pill${(s.subtitleMaxWords || 0) > 0 ? ' ve-properties__speed-pill--active' : ''}`}
            onClick={() => set('subtitleMaxWords', (s.subtitleMaxWords || 0) > 0 ? 0 : 4)}
          >
            {(s.subtitleMaxWords || 0) > 0 ? 'On' : 'Off'}
          </button>
        </div>
        {(s.subtitleMaxWords || 0) > 0 && (
          <SliderRow
            label="Words"
            value={s.subtitleMaxWords}
            min={Math.max(1, SUBTITLE_RANGES.maxWords.min)}
            max={SUBTITLE_RANGES.maxWords.max}
            step={SUBTITLE_RANGES.maxWords.step}
            onChange={(v) => set('subtitleMaxWords', v)}
          />
        )}
      </div>

      {/* Active Word Highlight */}
      <div className="ve-properties__section">
        <label className="ve-properties__label">Active Word Highlight</label>
        <div className="ve-properties__slider-row">
          <span className="ve-properties__field-label">Enabled</span>
          <button
            className={`ve-properties__speed-pill${s.activeWordEnabled ? ' ve-properties__speed-pill--active' : ''}`}
            onClick={() => set('activeWordEnabled', !s.activeWordEnabled)}
          >
            {s.activeWordEnabled ? 'On' : 'Off'}
          </button>
        </div>
        {s.activeWordEnabled && (
          <>
            <div className="ve-properties__row">
              <ColorField label="Word Color" value={s.activeWordColor || '#FFD700'} onChange={(v) => set('activeWordColor', v)} />
              <ColorField label="Stroke" value={s.activeWordOutlineColor || '#000000'} onChange={(v) => set('activeWordOutlineColor', v)} />
            </div>
            <div className="ve-properties__row">
              <ColorField label="BG Color" value={s.activeWordBgColor || '#000000'} onChange={(v) => set('activeWordBgColor', v)} />
              <NumField label="BG Opacity" value={s.activeWordBgOpacity ?? 0} min={0} max={100} onChange={(v) => set('activeWordBgOpacity', v)} />
            </div>
          </>
        )}
      </div>

      {/* Outline */}
      <div className="ve-properties__section">
        <label className="ve-properties__label">Outline</label>
        <div className="ve-properties__row">
          <ColorField label="Color" value={s.subtitleOutlineColor || '#000000'} onChange={(v) => set('subtitleOutlineColor', v)} />
          <NumField
            label="Width"
            value={s.subtitleOutlineWidth ?? 2}
            min={SUBTITLE_RANGES.outlineWidth.min}
            max={SUBTITLE_RANGES.outlineWidth.max}
            step={SUBTITLE_RANGES.outlineWidth.step}
            onChange={(v) => set('subtitleOutlineWidth', v)}
          />
        </div>
        <SliderRow
          label="Opacity"
          value={s.subtitleOutlineOpacity ?? 100}
          min={SUBTITLE_RANGES.outlineOpacity.min}
          max={SUBTITLE_RANGES.outlineOpacity.max}
          step={SUBTITLE_RANGES.outlineOpacity.step}
          unit="%"
          onChange={(v) => set('subtitleOutlineOpacity', v)}
        />
      </div>

      {/* Background */}
      <div className="ve-properties__section">
        <label className="ve-properties__label">Background</label>
        <div className="ve-properties__slider-row">
          <span className="ve-properties__field-label">Enabled</span>
          <button
            className={`ve-properties__speed-pill${s.subtitleBgEnabled ? ' ve-properties__speed-pill--active' : ''}`}
            onClick={() => set('subtitleBgEnabled', !s.subtitleBgEnabled)}
          >
            {s.subtitleBgEnabled ? 'On' : 'Off'}
          </button>
        </div>
        {s.subtitleBgEnabled && (
          <>
            <div className="ve-properties__row">
              <ColorField label="BG Color" value={s.subtitleBgColor || '#000000'} onChange={(v) => set('subtitleBgColor', v)} />
              <NumField label="Opacity" value={s.subtitleBgOpacity ?? 75} min={0} max={100} onChange={(v) => set('subtitleBgOpacity', v)} />
            </div>
          </>
        )}
      </div>

      {/* Speaker Colors */}
      <div className="ve-properties__section">
        <label className="ve-properties__label">Speaker Colors</label>
        <div className="ve-properties__slider-row">
          <span className="ve-properties__field-label">Enabled</span>
          <button
            className={`ve-properties__speed-pill${(s.useSpeakerColors ?? true) ? ' ve-properties__speed-pill--active' : ''}`}
            onClick={() => set('useSpeakerColors', !(s.useSpeakerColors ?? true))}
          >
            {(s.useSpeakerColors ?? true) ? 'On' : 'Off'}
          </button>
        </div>
        <div className="ve-properties__slider-row">
          <span className="ve-properties__field-label">Labels</span>
          <button
            className={`ve-properties__speed-pill${s.showSpeakerLabels ? ' ve-properties__speed-pill--active' : ''}`}
            onClick={() => set('showSpeakerLabels', !s.showSpeakerLabels)}
          >
            {s.showSpeakerLabels ? 'On' : 'Off'}
          </button>
        </div>
      </div>
    </>
  );
}

/* ── PresetBar — dropdown + save + load for track-type presets ── */
function PresetBar({ item, updateItem, settings, onSettingsChange }) {
  const [presets, setPresets] = useState([]);
  const [showSave, setShowSave] = useState(false);
  const [saveName, setSaveName] = useState('');
  const [message, setMessage] = useState(null);
  const saveInputRef = useRef(null);
  const type = item?.type;

  // Refresh presets when type changes or after save
  const refreshPresets = useCallback(() => {
    if (!type) return;
    setPresets(loadPresetsForType(type));
  }, [type]);

  // Load presets on mount and when type changes. ``useEffect`` instead
  // of ``useMemo`` because this is a side effect (state update), not a
  // memoized value — running it during render triggered React 18
  // strict-mode warnings and could double-fire under concurrent
  // rendering.
  useEffect(() => { refreshPresets(); }, [refreshPresets]);

  const handleSave = useCallback(() => {
    const name = saveName.trim();
    if (!name || !item) return;
    try {
      savePreset(name, type, item, settings);
      setShowSave(false);
      setSaveName('');
      refreshPresets();
      setMessage('Preset saved');
      setTimeout(() => setMessage(null), 2000);
    } catch (err) {
      setMessage(err.message);
      setTimeout(() => setMessage(null), 3000);
    }
  }, [saveName, item, type, settings, refreshPresets]);

  const handleLoad = useCallback((presetId) => {
    const preset = presets.find(p => p.id === presetId);
    if (!preset || !item) return;
    try {
      const { itemUpdates, subtitleSettings: subSettings } = applyPreset(preset, type);
      updateItem(item.id, itemUpdates);
      if (subSettings && onSettingsChange) {
        onSettingsChange({ ...settings, ...subSettings });
      }
      setMessage('Preset applied');
      setTimeout(() => setMessage(null), 2000);
    } catch (err) {
      setMessage(err.message);
      setTimeout(() => setMessage(null), 3000);
    }
  }, [presets, item, type, updateItem, settings, onSettingsChange]);

  if (!type) return null;

  return (
    <div className="ve-properties__section" style={{ paddingBottom: 6 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
        <select
          className="ve-properties__select"
          style={{ flex: 1, fontSize: 11 }}
          value=""
          onChange={(e) => {
            if (e.target.value) handleLoad(e.target.value);
          }}
        >
          <option value="">
            {presets.length === 0
              ? `No ${TYPE_LABELS[type] || type} presets`
              : `Load ${TYPE_LABELS[type] || type} preset...`}
          </option>
          {presets.map(p => (
            <option key={p.id} value={p.id}>{p.name}</option>
          ))}
        </select>
        <button
          className="ve-properties__speed-pill"
          style={{ fontSize: 10, padding: '4px 8px', whiteSpace: 'nowrap' }}
          onClick={() => {
            setShowSave(!showSave);
            setSaveName(`${TYPE_LABELS[type] || type} Preset`);
            setTimeout(() => saveInputRef.current?.focus(), 50);
          }}
          title="Save current properties as a preset"
        >
          Save
        </button>
      </div>
      {showSave && (
        <div style={{ display: 'flex', gap: 4, marginTop: 6 }}>
          <input
            ref={saveInputRef}
            type="text"
            value={saveName}
            onChange={(e) => setSaveName(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') handleSave();
              if (e.key === 'Escape') setShowSave(false);
            }}
            placeholder="Preset name..."
            className="ve-properties__input"
            style={{ flex: 1, fontSize: 11 }}
          />
          <button
            className="ve-properties__speed-pill ve-properties__speed-pill--active"
            style={{ fontSize: 10, padding: '4px 8px' }}
            onClick={handleSave}
            disabled={!saveName.trim()}
          >
            Save
          </button>
          <button
            className="ve-properties__speed-pill"
            style={{ fontSize: 10, padding: '4px 6px' }}
            onClick={() => setShowSave(false)}
          >
            ✕
          </button>
        </div>
      )}
      {message && (
        <div style={{
          fontSize: 10, marginTop: 4, padding: '2px 6px', borderRadius: 3,
          color: message.includes('error') || message.includes('Cannot') ? '#FF375F' : '#30D158',
          background: message.includes('error') || message.includes('Cannot')
            ? 'rgba(255,55,95,0.1)' : 'rgba(48,209,88,0.1)',
        }}>
          {message}
        </div>
      )}
    </div>
  );
}

const DEFAULTS = {
  position: { x: 50, y: 50 },
  size: { video: { w: 100, h: 100 }, text: { w: 35, h: 12 }, shape: { w: 40, h: 30 }, other: { w: 30, h: 30 } },
  rotation: 0,
  fadeIn: 0,
  fadeOut: 0,
  volume: 1,
  speed: 1.0,
  opacity: 1,
  effects: { brightness: 0, contrast: 0, saturation: 0, blur: 0, hueRotate: 0, sepia: 0 },
  textStyle: { fontSize: 48, fontFamily: 'DM Sans', fontWeight: 700, color: '#FFFFFF', textAlign: 'center', outlineWidth: 2, outlineColor: '#000000', bgOpacity: 0, bgPadding: 8, bgRadius: 4, shadowBlur: 0, shadowOffsetX: 0, shadowOffsetY: 0, animation: 'none' },
};

function hasChanged(current, def) {
  if (current == null && def == null) return false;
  if (typeof def === 'object' && def !== null) {
    return Object.keys(def).some(k => {
      const cv = current?.[k];
      const dv = def[k];
      return cv !== undefined && cv !== null && cv !== dv;
    });
  }
  return current !== def;
}

function ResetButton({ onClick, label = 'Reset' }) {
  return (
    <button
      className="ve-properties__speed-pill"
      style={{ fontSize: 9, padding: '2px 6px', marginLeft: 'auto', opacity: 0.7 }}
      onClick={onClick}
      title={`Reset ${label} to default`}
    >
      {label}
    </button>
  );
}

export default function PropertiesPanel({ compact = false, settings = null, onSettingsChange = null }) {
  const selectedItemId = useTimelineStore((s) => s.selectedItemId);
  const items = useTimelineStore((s) => s.items);
  const tracks = useTimelineStore((s) => s.tracks);
  const updateItem = useTimelineStore((s) => s.updateItem);
  const removeItem = useTimelineStore((s) => s.removeItem);
  const mediaLibrary = useTimelineStore((s) => s.mediaLibrary);
  const [expandedSections, setExpandedSections] = useState({});

  const item = useMemo(
    () => items.find((i) => i.id === selectedItemId) || null,
    [items, selectedItemId],
  );

  // Check if the selected item is on a locked track
  const isLocked = useMemo(() => {
    if (!item) return false;
    const track = tracks.find((t) => t.id === item.trackId);
    return track?.locked === true;
  }, [item, tracks]);

  const update = useCallback(
    (field, value) => {
      if (!item || isLocked) return;
      updateItem(item.id, { [field]: value });
    },
    [item, updateItem, isLocked],
  );

  const updateNested = useCallback(
    (field, subField, value) => {
      if (!item || isLocked) return;
      const current = item[field] || {};
      updateItem(item.id, { [field]: { ...current, [subField]: value } });
    },
    [item, updateItem, isLocked],
  );

  const toggleSection = (name) => setExpandedSections(prev => ({ ...prev, [name]: !prev[name] }));

  // Fetch custom uploaded fonts from the backend and inject @font-face rules
  // so both the dropdown and the canvas preview can use them.
  const [customFonts, setCustomFonts] = useState([]);
  useEffect(() => {
    let cancelled = false;
    fetch('/api/fonts')
      .then(r => r.ok ? r.json() : [])
      .then(fonts => {
        if (cancelled || !Array.isArray(fonts)) return;
        setCustomFonts(fonts);
        // Inject @font-face rules for each custom font so the browser
        // can render them in the canvas preview and the dropdown.
        for (const f of fonts) {
          const ruleId = `custom-font-${f.filename}`;
          if (document.getElementById(ruleId)) continue;
          const style = document.createElement('style');
          style.id = ruleId;
          style.textContent = `@font-face { font-family: '${f.name}'; src: url('${f.url}'); font-display: swap; }`;
          document.head.appendChild(style);
        }
      })
      .catch(() => {});
    return () => { cancelled = true; };
  }, []);


  // ── Live QA: validate item type matches track and properties are correct ──
  // NOTE: This useMemo MUST be before any early returns to satisfy Rules of Hooks.
  const itemTrack = item ? tracks.find((t) => t.id === item.trackId) : null;
  const trackTypeMismatch = useMemo(() => {
    if (!item || !itemTrack) return null;
    const allowed = {
      video: ['video'], overlay: ['text', 'shape', 'image', 'overlay'],
      audio: ['audio'], subtitle: ['subtitle'],
    };
    const ok = allowed[itemTrack.type];
    if (ok && !ok.includes(item.type)) {
      return `Item type "${item.type}" is on incompatible track "${itemTrack.name}" (${itemTrack.type})`;
    }
    return null;
  }, [item, itemTrack]);

  // ── Crop segment properties ──
  const cropSegments = useTimelineStore((s) => s.cropSegments);
  const selectedCropSegmentId = useTimelineStore((s) => s.selectedCropSegmentId);
  const updateCropSegment = useTimelineStore((s) => s.updateCropSegment);
  const splitCropSegment = useTimelineStore((s) => s.splitCropSegment);
  const mergeCropWithNext = useTimelineStore((s) => s.mergeCropWithNext);
  const resetCropSegment = useTimelineStore((s) => s.resetCropSegment);
  const playhead = useTimelineStore((s) => s.playhead);

  const selectedCropSeg = useMemo(
    () => selectedCropSegmentId ? cropSegments.find(s => s.id === selectedCropSegmentId) : null,
    [cropSegments, selectedCropSegmentId],
  );

  if (!item) {
    if (selectedCropSeg) {
      return (
        <div className={`ve-properties${compact ? ' ve-properties--compact' : ''}`}>
          <div className="ve-properties__header">
            <span className="ve-properties__type-badge" data-type="crop" style={{ background: '#06B6D4', color: '#fff', padding: '2px 8px', borderRadius: 4, fontSize: 11, fontWeight: 600 }}>
              Crop
            </span>
          </div>

          {/* Position slider */}
          {/* Defensive numeric coercion — without these guards a
              corrupted crop segment (cropX/startTime/endTime missing
              or NaN, e.g. from a partially-imported project file)
              would crash the entire properties panel via ``.toFixed``
              on undefined. */}
          <div style={{ padding: '8px 12px' }}>
            <label style={{ fontSize: 11, color: 'var(--text-muted)', display: 'block', marginBottom: 4 }}>Horizontal Position</label>
            <input
              type="range"
              min={0} max={100} step={1}
              value={Number.isFinite(selectedCropSeg.cropX) ? selectedCropSeg.cropX : 50}
              onChange={e => updateCropSegment({ ...selectedCropSeg, cropX: Number(e.target.value) })}
              {...undoCoalesceHandlers()}
              style={{ width: '100%', accentColor: '#06B6D4' }}
            />
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, color: 'var(--text-muted)' }}>
              <span>Left</span>
              <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-primary)' }}>
                {Number.isFinite(selectedCropSeg.cropX) ? selectedCropSeg.cropX : 50}%
              </span>
              <span>Right</span>
            </div>
          </div>

          {/* Timing */}
          <div style={{ padding: '4px 12px', display: 'flex', gap: 8 }}>
            <div style={{ flex: 1 }}>
              <label style={{ fontSize: 10, color: 'var(--text-muted)', display: 'block' }}>Start</label>
              <span style={{ fontSize: 12, fontFamily: 'var(--font-mono)' }}>
                {(Number.isFinite(selectedCropSeg.startTime) ? selectedCropSeg.startTime : 0).toFixed(1)}s
              </span>
            </div>
            <div style={{ flex: 1 }}>
              <label style={{ fontSize: 10, color: 'var(--text-muted)', display: 'block' }}>End</label>
              <span style={{ fontSize: 12, fontFamily: 'var(--font-mono)' }}>
                {(Number.isFinite(selectedCropSeg.endTime) ? selectedCropSeg.endTime : 0).toFixed(1)}s
              </span>
            </div>
            <div style={{ flex: 1 }}>
              <label style={{ fontSize: 10, color: 'var(--text-muted)', display: 'block' }}>Duration</label>
              <span style={{ fontSize: 12, fontFamily: 'var(--font-mono)' }}>
                {Math.max(0,
                  (Number.isFinite(selectedCropSeg.endTime) ? selectedCropSeg.endTime : 0)
                  - (Number.isFinite(selectedCropSeg.startTime) ? selectedCropSeg.startTime : 0)
                ).toFixed(1)}s
              </span>
            </div>
          </div>

          {/* Label */}
          <div style={{ padding: '4px 12px' }}>
            <label style={{ fontSize: 10, color: 'var(--text-muted)', display: 'block' }}>Segment</label>
            <span style={{ fontSize: 12 }}>
              {selectedCropSeg.label}
              {selectedCropSeg.isManualOverride && <span style={{ fontSize: 10, color: '#8B5CF6', marginLeft: 6 }}>(manual)</span>}
            </span>
          </div>

          {/* Actions */}
          <div style={{ padding: '8px 12px', display: 'flex', gap: 6, flexWrap: 'wrap', borderTop: '1px solid var(--border)' }}>
            <button
              onClick={() => splitCropSegment(selectedCropSeg.id, playhead)}
              style={{ padding: '4px 10px', fontSize: 11, background: 'var(--bg-elevated)', color: 'var(--text-secondary)', border: '1px solid var(--border)', borderRadius: 4, cursor: 'pointer' }}
            >
              Split at Playhead
            </button>
            <button
              onClick={() => mergeCropWithNext(selectedCropSeg.id)}
              style={{ padding: '4px 10px', fontSize: 11, background: 'var(--bg-elevated)', color: 'var(--text-secondary)', border: '1px solid var(--border)', borderRadius: 4, cursor: 'pointer' }}
            >
              Merge with Next
            </button>
            {selectedCropSeg.isManualOverride && (
              <button
                onClick={() => resetCropSegment(selectedCropSeg.id, selectedCropSeg.originalCropX)}
                style={{ padding: '4px 10px', fontSize: 11, background: 'var(--bg-elevated)', color: '#06B6D4', border: '1px solid var(--border)', borderRadius: 4, cursor: 'pointer' }}
              >
                Reset to Auto
              </button>
            )}
          </div>
        </div>
      );
    }

    return (
      <div className="ve-properties ve-properties--empty">
        <span className="ve-properties__placeholder">
          Select a timeline item to edit properties
        </span>
      </div>
    );
  }

  const duration = item.end - item.start;
  const isVisual = item.type !== 'audio';
  const isMediaClip = item.type === 'video' || item.type === 'audio';
  const isOverlay = item.type === 'video' || item.type === 'text' || item.type === 'shape' || item.type === 'image' || item.type === 'overlay' || item.type === 'subtitle';

  // Determine the human-readable type label for the badge
  const TYPE_LABELS_MAP = {
    video: 'Video', audio: 'Audio', text: 'Text', image: 'Image',
    overlay: 'Overlay', shape: 'Shape', subtitle: 'Subtitle',
  };

  return (
    <div className={`ve-properties${compact ? ' ve-properties--compact' : ''}`}>
      {/* QA warning if track/type mismatch detected */}
      {trackTypeMismatch && (
        <div style={{
          padding: '4px 8px', fontSize: 10, background: 'rgba(255,59,48,0.1)',
          color: '#FF375F', borderRadius: 4, marginBottom: 4,
        }}>
          QA: {trackTypeMismatch}
        </div>
      )}
      {/* Header */}
      <div className="ve-properties__header">
        <span className="ve-properties__type-badge" data-type={item.type}>
          {TYPE_LABELS_MAP[item.type] || item.type}
        </span>
        {isLocked && (
          <span style={{ fontSize: 11, color: 'var(--danger, #ef4444)', marginLeft: 4, display: 'flex', alignItems: 'center', gap: 3 }}>
            🔒 Locked
          </span>
        )}
        <button
          className="ve-properties__delete"
          onClick={() => { if (!isLocked) removeItem(item.id); }}
          title={isLocked ? 'Cannot delete — track is locked' : 'Delete item'}
          disabled={isLocked}
          style={isLocked ? { opacity: 0.3, cursor: 'not-allowed' } : undefined}
        >
          ✕
        </button>
      </div>

      {/* Preset Bar */}
      <PresetBar item={item} updateItem={updateItem} settings={settings} onSettingsChange={onSettingsChange} />

      {/* ── Timing ── */}
      <div className="ve-properties__section">
        <label className="ve-properties__label">Timing</label>
        <div className="ve-properties__row">
          <NumField label="Start" value={parseFloat(item.start.toFixed(2))} min={0} max={item.end - 0.1} step={0.01} onChange={(v) => update('start', v)} />
          <NumField label="End" value={parseFloat(item.end.toFixed(2))} min={item.start + 0.1} max={item.start + getMaxItemDuration(item, mediaLibrary)} step={0.01} onChange={(v) => update('end', v)} />
          <div className="ve-properties__field">
            <span className="ve-properties__field-label">Duration</span>
            <span className="ve-properties__field-value">{formatTime(duration)}</span>
          </div>
        </div>
      </div>

      {/* ── Position & Size (all visual items including video) ── */}
      {isOverlay && (
        <div className="ve-properties__section">
          <div style={{ display: 'flex', alignItems: 'center' }}>
            <label className="ve-properties__label" style={{ marginBottom: 0 }}>Position & Size</label>
            {(hasChanged(item.position, DEFAULTS.position) ||
              hasChanged(item.size, DEFAULTS.size[item.type] || DEFAULTS.size.other) ||
              (item.transform?.rotation || 0) !== DEFAULTS.rotation) && (
              <ResetButton label="Reset" onClick={() => {
                const defSize = DEFAULTS.size[item.type] || DEFAULTS.size.other;
                update('position', { ...DEFAULTS.position });
                update('size', { ...defSize });
                updateNested('transform', 'rotation', 0);
              }} />
            )}
          </div>
          <SliderRow label="X" value={item.position?.x ?? 50} min={0} max={100} step={0.5} unit="%" onChange={(v) => update('position', { ...item.position, x: v })} />
          <SliderRow label="Y" value={item.position?.y ?? 50} min={0} max={100} step={0.5} unit="%" onChange={(v) => update('position', { ...item.position, y: v })} />
          <SliderRow label="Width" value={item.size?.w ?? 50} min={1} max={200} step={0.5} unit="%" onChange={(v) => update('size', { ...item.size, w: v })} />
          <SliderRow label="Height" value={item.size?.h ?? 50} min={1} max={200} step={0.5} unit="%" onChange={(v) => update('size', { ...item.size, h: v })} />
          <SliderRow label="Rotation" value={item.transform?.rotation ?? 0} min={-360} max={360} step={1} unit="°" onChange={(v) => updateNested('transform', 'rotation', v)} />
          {item.type === 'video' && (
            <button
              className="ve-properties__speed-pill"
              style={{ marginTop: 4, fontSize: 10, padding: '3px 8px' }}
              onClick={() => {
                update('position', { x: 50, y: 50 });
                update('size', { w: 100, h: 100 });
                updateNested('transform', 'rotation', 0);
              }}
              title="Reset video to fill viewport"
            >
              Reset to Fill
            </button>
          )}
        </div>
      )}

      {/* ── Fades ── */}
      <div className="ve-properties__section">
        <div style={{ display: 'flex', alignItems: 'center' }}>
          <label className="ve-properties__label" style={{ marginBottom: 0 }}>Fades</label>
          {((item.fadeIn || 0) !== 0 || (item.fadeOut || 0) !== 0) && (
            <ResetButton label="Reset" onClick={() => { update('fadeIn', 0); update('fadeOut', 0); }} />
          )}
        </div>
        <div className="ve-properties__row">
          <NumField label="Fade In" value={item.fadeIn || 0} min={0} max={duration} step={0.1} onChange={(v) => update('fadeIn', v)} />
          <NumField label="Fade Out" value={item.fadeOut || 0} min={0} max={duration} step={0.1} onChange={(v) => update('fadeOut', v)} />
        </div>
      </div>

      {/* ── Volume (video & audio) ── */}
      {isMediaClip && (
        <div className="ve-properties__section">
          <div style={{ display: 'flex', alignItems: 'center' }}>
            <label className="ve-properties__label" style={{ marginBottom: 0 }}>Audio</label>
            {(item.volume ?? 1) !== 1 && (
              <ResetButton label="Reset" onClick={() => update('volume', 1)} />
            )}
          </div>
          <SliderRow label="Volume" value={item.volume ?? 1} min={0} max={2} step={0.01} onChange={(v) => update('volume', v)} />
        </div>
      )}

      {/* ── Speed (video & audio) ── */}
      {isMediaClip && (
        <div className="ve-properties__section">
          <div style={{ display: 'flex', alignItems: 'center' }}>
            <label className="ve-properties__label" style={{ marginBottom: 0 }}>Speed</label>
            {item.speed !== 1.0 && (
              <ResetButton label="Reset" onClick={() => update('speed', 1.0)} />
            )}
          </div>
          <div className="ve-properties__speed-pills">
            {SPEED_PRESETS.map((s) => (
              <button
                key={s}
                className={`ve-properties__speed-pill${item.speed === s ? ' ve-properties__speed-pill--active' : ''}`}
                onClick={() => update('speed', s)}
              >
                {s}x
              </button>
            ))}
          </div>
        </div>
      )}

      {/* ── Opacity (all visual items) ── */}
      {isVisual && (
        <div className="ve-properties__section">
          <div style={{ display: 'flex', alignItems: 'center' }}>
            <label className="ve-properties__label" style={{ marginBottom: 0 }}>Opacity</label>
            {(item.opacity ?? 1) !== 1 && (
              <ResetButton label="Reset" onClick={() => update('opacity', 1)} />
            )}
          </div>
          <SliderRow label="" value={item.opacity ?? 1} min={0} max={1} step={0.01} onChange={(v) => update('opacity', v)} />
        </div>
      )}

      {/* ── Effects (all visual items) ── */}
      {(item.type === 'video' || item.type === 'text' || item.type === 'shape' || item.type === 'image' || item.type === 'overlay') && (
        <div className="ve-properties__section">
          <div style={{ display: 'flex', alignItems: 'center' }}>
            <label className="ve-properties__label" onClick={() => toggleSection('effects')} style={{ cursor: 'pointer', marginBottom: 0 }}>
              Effects {expandedSections.effects === false ? '▸' : '▾'}
            </label>
            {hasChanged(item.effects, DEFAULTS.effects) && (
              <ResetButton label="Reset" onClick={() => update('effects', { ...DEFAULTS.effects })} />
            )}
          </div>
          {expandedSections.effects !== false && (
            <>
              {[
                { key: 'brightness', label: 'Brightness', min: -100, max: 100 },
                { key: 'contrast', label: 'Contrast', min: -100, max: 100 },
                { key: 'saturation', label: 'Saturation', min: -100, max: 100 },
                { key: 'blur', label: 'Blur', min: 0, max: 20, step: 0.5 },
                { key: 'hueRotate', label: 'Hue', min: 0, max: 360 },
                { key: 'sepia', label: 'Sepia', min: 0, max: 100 },
              ].map(ctrl => (
                <SliderRow
                  key={ctrl.key}
                  label={ctrl.label}
                  value={(item.effects || {})[ctrl.key] ?? 0}
                  min={ctrl.min}
                  max={ctrl.max}
                  step={ctrl.step || 1}
                  onChange={(v) => update('effects', { ...(item.effects || {}), [ctrl.key]: v })}
                />
              ))}
            </>
          )}
        </div>
      )}

      {/* ══════════════════════════════════════════════
          TEXT STYLING — comprehensive controls
         ══════════════════════════════════════════════ */}
      {item.type === 'text' && (
        <>
          {/* Text Content */}
          <div className="ve-properties__section">
            <label className="ve-properties__label">Text Content</label>
            <textarea
              value={item.textContent || ''}
              onChange={(e) => update('textContent', e.target.value)}
              className="ve-properties__textarea"
              rows={3}
              placeholder="Enter text..."
            />
          </div>

          {/* Font */}
          <div className="ve-properties__section">
            <label className="ve-properties__label">Font</label>
            <div className="ve-properties__row">
              <div className="ve-properties__field" style={{ flex: 2 }}>
                <span className="ve-properties__field-label">Family</span>
                <select
                  value={item.textStyle?.fontFamily || 'DM Sans'}
                  onChange={(e) => {
                    const newFont = e.target.value;
                    updateNested('textStyle', 'fontFamily', newFont);
                    // Snap weight to closest available if current weight isn't supported
                    const curWeight = item.textStyle?.fontWeight || 700;
                    const available = getWeightOptionsForFont(newFont).map(w => w.value);
                    if (!available.includes(curWeight)) {
                      const closest = available.reduce((a, b) => Math.abs(b - curWeight) < Math.abs(a - curWeight) ? b : a);
                      updateNested('textStyle', 'fontWeight', closest);
                    }
                  }}
                  className="ve-properties__select"
                >
                  {FONT_OPTIONS.map(f => <option key={f} value={f}>{f}</option>)}
                  {customFonts.length > 0 && (
                    <optgroup label="Custom Fonts">
                      {customFonts.filter(f => f.name && !FONT_OPTIONS.includes(f.name)).map(f => (
                        <option key={f.name} value={f.name}>{f.name}</option>
                      ))}
                    </optgroup>
                  )}
                </select>
              </div>
              <div className="ve-properties__field">
                <span className="ve-properties__field-label">Weight</span>
                <select
                  value={item.textStyle?.fontWeight || 700}
                  onChange={(e) => updateNested('textStyle', 'fontWeight', parseInt(e.target.value))}
                  className="ve-properties__select"
                >
                  {getWeightOptionsForFont(item.textStyle?.fontFamily || 'DM Sans').map(w => (
                    <option key={w.value} value={w.value}>{w.label}</option>
                  ))}
                </select>
              </div>
            </div>
            <SliderRow label="Size" value={item.textStyle?.fontSize || 48} min={8} max={200} step={1} unit="px" onChange={(v) => updateNested('textStyle', 'fontSize', v)} />
            <div className="ve-properties__row">
              <ColorField label="Color" value={item.textStyle?.color || '#FFFFFF'} onChange={(v) => updateNested('textStyle', 'color', v)} />
              <div className="ve-properties__field">
                <span className="ve-properties__field-label">Align</span>
                <div style={{ display: 'flex', gap: 2 }}>
                  {['left', 'center', 'right'].map(a => (
                    <button
                      key={a}
                      className={`ve-properties__speed-pill${(item.textStyle?.textAlign || 'center') === a ? ' ve-properties__speed-pill--active' : ''}`}
                      onClick={() => updateNested('textStyle', 'textAlign', a)}
                      style={{ padding: '3px 7px', fontSize: 9, textTransform: 'capitalize' }}
                    >
                      {a === 'left' ? '◀' : a === 'right' ? '▶' : '◆'}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Outline & Shadow */}
          <div className="ve-properties__section">
            <label className="ve-properties__label" onClick={() => toggleSection('textOutline')} style={{ cursor: 'pointer' }}>
              Outline & Shadow {expandedSections.textOutline === false ? '▸' : '▾'}
            </label>
            {expandedSections.textOutline !== false && (
              <>
                <div className="ve-properties__row">
                  <NumField label="Outline" value={item.textStyle?.outlineWidth || 0} min={0} max={10} step={0.5} onChange={(v) => updateNested('textStyle', 'outlineWidth', v)} />
                  <ColorField label="Stroke" value={item.textStyle?.outlineColor || '#000000'} onChange={(v) => updateNested('textStyle', 'outlineColor', v)} />
                </div>
                <SliderRow label="Shadow" value={item.textStyle?.shadowBlur || 0} min={0} max={20} step={0.5} unit="px" onChange={(v) => updateNested('textStyle', 'shadowBlur', v)} />
                <div className="ve-properties__row">
                  <NumField label="Shd X" value={item.textStyle?.shadowOffsetX || 0} min={-20} max={20} onChange={(v) => updateNested('textStyle', 'shadowOffsetX', v)} />
                  <NumField label="Shd Y" value={item.textStyle?.shadowOffsetY || 0} min={-20} max={20} onChange={(v) => updateNested('textStyle', 'shadowOffsetY', v)} />
                </div>
                <div className="ve-properties__row">
                  <ColorField label="Shadow" value={item.textStyle?.shadowColor || 'rgba(0,0,0,0.5)'} onChange={(v) => updateNested('textStyle', 'shadowColor', v)} />
                </div>
              </>
            )}
          </div>

          {/* Background Box */}
          <div className="ve-properties__section">
            <label className="ve-properties__label" onClick={() => toggleSection('textBg')} style={{ cursor: 'pointer' }}>
              Background Box {expandedSections.textBg === false ? '▸' : '▾'}
            </label>
            {expandedSections.textBg !== false && (
              <>
                <div className="ve-properties__row">
                  <ColorField label="BG Color" value={item.textStyle?.bgColor || '#000000'} onChange={(v) => updateNested('textStyle', 'bgColor', v)} />
                  <NumField label="Opacity" value={item.textStyle?.bgOpacity || 0} min={0} max={100} onChange={(v) => updateNested('textStyle', 'bgOpacity', v)} />
                </div>
                <div className="ve-properties__row">
                  <NumField label="Padding" value={item.textStyle?.bgPadding || 8} min={0} max={40} onChange={(v) => updateNested('textStyle', 'bgPadding', v)} />
                  <NumField label="Radius" value={item.textStyle?.bgRadius || 4} min={0} max={40} onChange={(v) => updateNested('textStyle', 'bgRadius', v)} />
                </div>
              </>
            )}
          </div>

          {/* Animation */}
          <div className="ve-properties__section">
            <label className="ve-properties__label">Animation</label>
            <div className="ve-properties__speed-pills">
              {TEXT_ANIMATIONS.map(a => (
                <button
                  key={a.id}
                  className={`ve-properties__speed-pill${(item.textStyle?.animation || 'none') === a.id ? ' ve-properties__speed-pill--active' : ''}`}
                  onClick={() => updateNested('textStyle', 'animation', a.id)}
                  style={{ fontSize: 9 }}
                >
                  {a.label}
                </button>
              ))}
            </div>
          </div>
        </>
      )}

      {/* ══════════════════════════════════════════════
          SHAPE STYLING — comprehensive controls
         ══════════════════════════════════════════════ */}
      {item.type === 'shape' && (
        <>
          <div className="ve-properties__section">
            <label className="ve-properties__label">Shape Type</label>
            <div className="ve-properties__speed-pills">
              {SHAPE_TYPES.map(s => (
                <button
                  key={s.id}
                  className={`ve-properties__speed-pill${(item.shapeType || 'rectangle') === s.id ? ' ve-properties__speed-pill--active' : ''}`}
                  onClick={() => update('shapeType', s.id)}
                  title={s.label}
                  style={{ fontSize: 12, padding: '4px 8px' }}
                >
                  {s.icon}
                </button>
              ))}
            </div>
          </div>

          <div className="ve-properties__section">
            <label className="ve-properties__label">Colors</label>
            <div className="ve-properties__row">
              <ColorField label="Fill" value={item.shapeStyle?.fillColor || '#FF3B30'} onChange={(v) => updateNested('shapeStyle', 'fillColor', v)} />
              <ColorField label="Stroke" value={item.shapeStyle?.strokeColor || '#FFFFFF'} onChange={(v) => updateNested('shapeStyle', 'strokeColor', v)} />
            </div>
            <SliderRow label="Stroke W" value={item.shapeStyle?.strokeWidth ?? 2} min={0} max={20} step={0.5} unit="px" onChange={(v) => updateNested('shapeStyle', 'strokeWidth', v)} />
            {(item.shapeType === 'rectangle' || !item.shapeType) && (
              <SliderRow label="Radius" value={item.shapeStyle?.cornerRadius ?? 0} min={0} max={100} step={1} unit="px" onChange={(v) => updateNested('shapeStyle', 'cornerRadius', v)} />
            )}
          </div>
        </>
      )}

      {/* ══════════════════════════════════════════════
          IMAGE/OVERLAY — media controls
         ══════════════════════════════════════════════ */}
      {(item.type === 'image' || item.type === 'overlay') && (
        <div className="ve-properties__section">
          <label className="ve-properties__label">Media</label>
          {item.mediaSrc || item.mediaRef ? (
            <div style={{ fontSize: 10, color: 'var(--ve-text-muted)', wordBreak: 'break-all', marginBottom: 6 }}>
              {(item.mediaSrc || item.mediaRef || '').split('/').pop() || 'Media file'}
            </div>
          ) : (
            <div style={{ fontSize: 10, color: 'var(--ve-text-muted)', marginBottom: 6, fontStyle: 'italic' }}>
              No media source — drag from Media Library
            </div>
          )}
        </div>
      )}

      {/* ── Subtitle — edits the global SubtitleOverlay (Subs On) settings ── */}
      {item.type === 'subtitle' && (
        <SubtitleProperties item={item} update={update} settings={settings} onSettingsChange={onSettingsChange} customFonts={customFonts} />
      )}

      {/* ── Transition ── */}
      {(item.type === 'video' || item.type === 'image') && (
        <div className="ve-properties__section">
          <label className="ve-properties__label">Transition</label>
          <div className="ve-properties__row">
            <div className="ve-properties__field">
              <span className="ve-properties__field-label">Type</span>
              <select
                value={item.transition?.type || 'none'}
                onChange={(e) => {
                  if (e.target.value === 'none') {
                    update('transition', null);
                  } else {
                    update('transition', {
                      type: e.target.value,
                      duration: item.transition?.duration || 0.5,
                    });
                  }
                }}
                className="ve-properties__select"
              >
                <option value="none">None</option>
                <option value="dissolve">Dissolve</option>
                <option value="fade">Fade</option>
                <option value="wipe-left">Wipe Left</option>
                <option value="wipe-right">Wipe Right</option>
                <option value="slide-left">Slide Left</option>
                <option value="slide-right">Slide Right</option>
                <option value="zoom">Zoom</option>
              </select>
            </div>
            {item.transition && (
              <NumField
                label="Duration"
                value={item.transition?.duration || 0.5}
                min={TRANSITION_RANGES.duration.min}
                max={TRANSITION_RANGES.duration.max}
                step={TRANSITION_RANGES.duration.step}
                onChange={(v) => update('transition', { ...item.transition, duration: v })}
              />
            )}
          </div>
        </div>
      )}

      {/* ── Clip Settings (apply global settings to this timeline item) ── */}
      {(item.type === 'video') && settings && (
        <div className="ve-properties__section">
          <label className="ve-properties__label">Clip Settings</label>
          <p style={{ fontSize: 10, color: 'var(--ve-text-muted, #888)', margin: '0 0 8px', lineHeight: 1.4 }}>
            Apply your subtitle and export settings from the main editor to this clip.
          </p>
          <div className="ve-properties__row" style={{ flexDirection: 'column', gap: 6 }}>
            <button
              className="ve-properties__preset-btn"
              onClick={() => {
                updateItem(item.id, {
                  clipSettings: { ...settings },
                });
              }}
            >
              Apply Current Settings
            </button>
            {item.clipSettings && (
              <span style={{ fontSize: 9, color: 'var(--ve-accent, #0A84FF)', fontFamily: 'var(--ve-font-mono, monospace)' }}>
                Settings applied ({item.clipSettings.exportQuality || '1080p'}, subs {item.clipSettings.subtitlesEnabled ? 'on' : 'off'})
              </span>
            )}
          </div>
          {/* Export quality override */}
          <div className="ve-properties__row" style={{ marginTop: 8 }}>
            <div className="ve-properties__field">
              <span className="ve-properties__field-label">Quality</span>
              <select
                value={item.clipSettings?.exportQuality || settings?.exportQuality || '1080p'}
                onChange={(e) => {
                  const cs = { ...(item.clipSettings || settings || {}), exportQuality: e.target.value };
                  updateItem(item.id, { clipSettings: cs });
                }}
                className="ve-properties__select"
              >
                <option value="720p">720p</option>
                <option value="1080p">1080p</option>
                <option value="4k">4K</option>
              </select>
            </div>
          </div>
          {/* Subtitles toggle */}
          <div className="ve-properties__slider-row" style={{ marginTop: 6 }}>
            <span className="ve-properties__field-label">Subtitles</span>
            <button
              className={`ve-properties__speed-pill${(item.clipSettings?.subtitlesEnabled ?? settings?.subtitlesEnabled) ? ' ve-properties__speed-pill--active' : ''}`}
              onClick={() => {
                const newVal = !(item.clipSettings?.subtitlesEnabled ?? settings?.subtitlesEnabled);
                const cs = { ...(item.clipSettings || settings || {}), subtitlesEnabled: newVal };
                updateItem(item.id, { clipSettings: cs });
                if (onSettingsChange) {
                  onSettingsChange({ ...settings, subtitlesEnabled: newVal });
                }
              }}
            >
              {(item.clipSettings?.subtitlesEnabled ?? settings?.subtitlesEnabled) ? 'On' : 'Off'}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
