import React, { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { Link } from 'react-router-dom';
import { showToast } from '../components/Toast';
import ClipPreview from '../components/ClipPreview';
import ClipSettingsPanel from '../components/ClipSettingsPanel';
import EditorErrorBoundary from '../components/EditorErrorBoundary';
import useResponsive from '../hooks/useResponsive';
import useEncodingManager from '../hooks/useEncodingManager';
import { useAuth } from '../auth/AuthContext';
import { computeClipSubjectX } from '../utils/subjectTracking';
import sanitizeJob, { sanitizeSubtitleSettings } from '../utils/sanitizeJob';
import useTimelineStore from '../stores/timelineStore';
import { buildOverlayPayload, buildVideoEffectsPayload, mapSubtitleSettings } from '../utils/buildExportPayload';
import { DEFAULT_CLIP_SETTINGS } from '../utils/defaultSettings';

function formatDuration(seconds) {
  if (!seconds) return '-';
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, '0')}`;
}

const ASPECT_RATIOS = [
  { value: null, label: 'Original' },
  { value: '16:9', label: '16:9' },
  { value: '9:16', label: '9:16' },
  { value: '1:1', label: '1:1' },
  { value: '4:5', label: '4:5' },
];

const BUILTIN_FONTS = [
  'DM Sans', 'Montserrat', 'Open Sans', 'Roboto', 'Poppins', 'Inter',
  'Nunito', 'Lato', 'Oswald', 'Playfair Display', 'Bebas Neue',
  'Liberation Sans', 'Liberation Serif', 'Liberation Mono',
  'DejaVu Sans', 'DejaVu Serif', 'DejaVu Sans Mono', 'FreeSans',
];

// Builtin fonts that need @font-face registration so browser preview matches export
const BUILTIN_FONT_FILES = {
  'DM Sans': '/api/fonts/builtin/DMSans.ttf',
  'Montserrat': '/api/fonts/builtin/Montserrat.ttf',
  'Open Sans': '/api/fonts/builtin/OpenSans.ttf',
  'Roboto': '/api/fonts/builtin/Roboto.ttf',
  'Poppins': '/api/fonts/builtin/Poppins-Regular.ttf',
  'Inter': '/api/fonts/builtin/Inter.ttf',
  'Nunito': '/api/fonts/builtin/Nunito.ttf',
  'Lato': '/api/fonts/builtin/Lato-Regular.ttf',
  'Oswald': '/api/fonts/builtin/Oswald.ttf',
  'Playfair Display': '/api/fonts/builtin/PlayfairDisplay.ttf',
  'Bebas Neue': '/api/fonts/builtin/BebasNeue-Regular.ttf',
  'Liberation Sans': '/api/fonts/builtin/LiberationSans-Regular.ttf',
  'Liberation Serif': '/api/fonts/builtin/LiberationSerif-Regular.ttf',
  'Liberation Mono': '/api/fonts/builtin/LiberationMono-Regular.ttf',
  'DejaVu Sans': '/api/fonts/builtin/DejaVuSans.ttf',
  'DejaVu Serif': '/api/fonts/builtin/DejaVuSerif.ttf',
  'DejaVu Sans Mono': '/api/fonts/builtin/DejaVuSansMono.ttf',
  'FreeSans': '/api/fonts/builtin/FreeSans.ttf',
};

const SIZES = [
  { value: 22, label: 'S' },
  { value: 30, label: 'M' },
  { value: 40, label: 'L' },
];
const POSITIONS = ['top', 'center', 'bottom'];
const DEFAULT_PALETTE = ['#00D9FF', '#F59E0B', '#10B981', '#A78BFA', '#EF4444', '#EC4899'];

// Register a custom font via @font-face so the browser can render it in the preview
function registerFontFace(fontName, url) {
  const id = `custom-font-${fontName.replace(/\s+/g, '-')}`;
  if (document.getElementById(id)) return;
  const style = document.createElement('style');
  style.id = id;
  style.textContent = `@font-face { font-family: '${fontName}'; src: url('${url}'); font-weight: 100 900; font-display: swap; }`;
  document.head.appendChild(style);
}

const SETTINGS_KEY = 'clipai_clip_settings';
const OVERRIDES_KEY = 'clipai_viral_clip_overrides';

const FONT_WEIGHTS = [
  { value: 400, label: 'Regular' },
  { value: 700, label: 'Bold' },
];

const DEFAULT_SETTINGS = DEFAULT_CLIP_SETTINGS;

function loadExportSettings() {
  try {
    const saved = localStorage.getItem(SETTINGS_KEY);
    if (saved) return { ...DEFAULT_SETTINGS, ...sanitizeSubtitleSettings(JSON.parse(saved)) };
  } catch {}
  return { ...DEFAULT_SETTINGS };
}

function loadClipOverrides() {
  try {
    const saved = localStorage.getItem(OVERRIDES_KEY);
    if (saved) return JSON.parse(saved);
  } catch {}
  return {};
}

const radioStyle = (active) => ({
  padding: '4px 10px',
  fontSize: 11,
  border: `1px solid ${active ? 'var(--accent-cyan)' : 'var(--border)'}`,
  background: active ? 'var(--accent-cyan-dim)' : 'var(--bg-elevated)',
  color: active ? 'var(--accent-cyan)' : 'var(--text-secondary)',
  cursor: 'pointer',
  borderRadius: 'var(--radius-sm)',
  fontWeight: active ? 600 : 400,
});

const labelStyle = {
  fontSize: 13, color: 'var(--text-secondary)', marginBottom: 6,
  textTransform: 'uppercase', letterSpacing: '0.04em',
};

// Toggle switch component
function Toggle({ value, onChange }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
      <div
        onClick={() => onChange(!value)}
        style={{
          width: 34, height: 18, borderRadius: 9,
          background: value ? 'var(--accent-cyan)' : 'var(--bg-elevated)',
          border: `1px solid ${value ? 'var(--accent-cyan)' : 'var(--border)'}`,
          position: 'relative', cursor: 'pointer', transition: 'background 0.2s',
        }}
      >
        <div style={{
          width: 14, height: 14, borderRadius: '50%',
          background: value ? 'var(--nav-active-icon-text)' : 'var(--text-muted)',
          position: 'absolute', top: 1,
          left: value ? 17 : 1,
          transition: 'left 0.2s',
        }} />
      </div>
      <span style={{ fontSize: 11, color: 'var(--text-secondary)' }}>
        {value ? 'ON' : 'OFF'}
      </span>
    </div>
  );
}

// Reusable subtitle settings editor
function SubtitleSettingsEditor({ values, onChange, speakerList, customFonts, compact }) {
  const set = (key, val) => onChange(key, val);
  const gap = compact ? 10 : 16;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap }}>
      {/* Aspect Ratio */}
      <div>
        <div style={labelStyle}>Aspect Ratio</div>
        <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap' }}>
          {ASPECT_RATIOS.map((r) => (
            <button key={r.value || 'original'} onClick={() => set('aspectRatio', r.value)} style={radioStyle(values.aspectRatio === r.value)}>
              {r.label}
            </button>
          ))}
        </div>
      </div>

      {/* Subtitles Toggle */}
      <div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <div style={labelStyle}>Subtitles</div>
          <Toggle value={values.subtitlesEnabled} onChange={(v) => set('subtitlesEnabled', v)} />
        </div>
      </div>

      {/* Subtitle Detail Settings */}
      {values.subtitlesEnabled && (
        <div style={{ display: 'flex', flexDirection: 'column', gap, paddingLeft: 8, borderLeft: '2px solid var(--border)' }}>
          {/* Font */}
          <div>
            <div style={labelStyle}>Font</div>
            <select
              value={values.subtitleFont}
              onChange={(e) => set('subtitleFont', e.target.value)}
              style={{
                padding: '4px 8px', borderRadius: 'var(--radius-sm)', fontSize: 12,
                background: 'var(--bg-elevated)', color: 'var(--text-primary)',
                border: '1px solid var(--border)', width: '100%', maxWidth: 240,
              }}
            >
              <optgroup label="Built-in">
                {BUILTIN_FONTS.map((f) => <option key={f} value={f}>{f}</option>)}
              </optgroup>
              {customFonts.length > 0 && (
                <optgroup label="Custom">
                  {customFonts.map((f) => <option key={f.name} value={f.name}>{f.name}</option>)}
                </optgroup>
              )}
            </select>
          </div>

          {/* Size — slider + number input matching ClipSettingsPanel (12-72) */}
          <div>
            <div style={labelStyle}>
              Size: <span style={{ color: 'var(--accent-cyan)', fontFamily: 'var(--font-mono)' }}>{typeof values.subtitleSize === 'number' ? values.subtitleSize : 30}px</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 6 }}>
              <input
                type="range" min="12" max="72"
                value={Math.min(72, Math.max(12, typeof values.subtitleSize === 'number' ? values.subtitleSize : 30))}
                onChange={(e) => set('subtitleSize', parseInt(e.target.value))}
                style={{ flex: 1, accentColor: 'var(--accent-cyan)' }}
              />
              <input
                type="number"
                value={typeof values.subtitleSize === 'number' ? values.subtitleSize : 30}
                onChange={(e) => { const v = parseInt(e.target.value); if (!isNaN(v) && v > 0) set('subtitleSize', v); }}
                style={{
                  width: 48, padding: '3px 4px', fontSize: 12, textAlign: 'center',
                  background: 'var(--bg-elevated)', color: 'var(--text-primary)',
                  border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)',
                }}
              />
            </div>
            <div style={{ display: 'flex', gap: 4 }}>
              {SIZES.map((s) => (
                <button key={s.value} onClick={() => set('subtitleSize', s.value)} style={radioStyle(values.subtitleSize === s.value)}>
                  {s.label}
                </button>
              ))}
            </div>
          </div>

          {/* Font Weight */}
          <div>
            <div style={labelStyle}>Weight</div>
            <div style={{ display: 'flex', gap: 4 }}>
              {FONT_WEIGHTS.map((w) => (
                <button key={w.value} onClick={() => set('subtitleFontWeight', w.value)} style={radioStyle((typeof values.subtitleFontWeight === 'number' ? values.subtitleFontWeight : (values.subtitleFontWeight === 'bold' ? 700 : 400)) === w.value)}>
                  {w.label}
                </button>
              ))}
            </div>
          </div>

          {/* Font Color */}
          <div>
            <div style={labelStyle}>Font Color</div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <input
                type="color"
                value={values.subtitleFontColor || '#FFFFFF'}
                onChange={(e) => set('subtitleFontColor', e.target.value)}
                style={{ width: 22, height: 22, border: 'none', cursor: 'pointer', background: 'none' }}
              />
              <span style={{ fontSize: 11, fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)' }}>
                {values.subtitleFontColor || '#FFFFFF'}
              </span>
            </div>
          </div>

          {/* Font Outline */}
          <div>
            <div style={labelStyle}>Font Outline</div>
            <div style={{ display: 'flex', gap: 10, alignItems: 'center', flexWrap: 'wrap' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>Color</span>
                <input
                  type="color"
                  value={values.subtitleOutlineColor || '#000000'}
                  onChange={(e) => set('subtitleOutlineColor', e.target.value)}
                  style={{ width: 22, height: 22, border: 'none', cursor: 'pointer', background: 'none' }}
                />
                <span style={{ fontSize: 11, fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)' }}>
                  {values.subtitleOutlineColor || '#000000'}
                </span>
              </div>
            </div>
            <div style={{ display: 'flex', gap: 10, marginTop: 6, flexWrap: 'wrap' }}>
              <div style={{ flex: 1, minWidth: 100 }}>
                <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 4 }}>
                  Opacity: <span style={{ color: 'var(--accent-cyan)', fontFamily: 'var(--font-mono)' }}>{values.subtitleOutlineOpacity ?? 100}%</span>
                </div>
                <input
                  type="range" min="0" max="100" step="5"
                  value={values.subtitleOutlineOpacity ?? 100}
                  onChange={(e) => set('subtitleOutlineOpacity', parseInt(e.target.value))}
                  style={{ width: '100%', accentColor: 'var(--accent-cyan)' }}
                />
              </div>
              <div style={{ flex: 1, minWidth: 100 }}>
                <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 4 }}>
                  Thickness: <span style={{ color: 'var(--accent-cyan)', fontFamily: 'var(--font-mono)' }}>{values.subtitleOutlineWidth ?? 2}px</span>
                </div>
                <input
                  type="range" min="0" max="10" step="1"
                  value={values.subtitleOutlineWidth ?? 2}
                  onChange={(e) => set('subtitleOutlineWidth', parseInt(e.target.value))}
                  style={{ width: '100%', accentColor: 'var(--accent-cyan)' }}
                />
              </div>
            </div>
          </div>

          {/* Position */}
          <div>
            <div style={labelStyle}>Position</div>
            <div style={{ display: 'flex', gap: 4 }}>
              {POSITIONS.map((p) => (
                <button key={p} onClick={() => {
                  set('subtitlePosition', p);
                  set('subtitleOffsetV', p === 'top' ? 96 : p === 'center' ? 50 : 4);
                }} style={radioStyle(
                  p === 'top' ? (values.subtitleOffsetV ?? 4) > 66
                  : p === 'center' ? (values.subtitleOffsetV ?? 4) >= 34 && (values.subtitleOffsetV ?? 4) <= 66
                  : (values.subtitleOffsetV ?? 4) < 34
                )}>
                  {p.charAt(0).toUpperCase() + p.slice(1)}
                </button>
              ))}
            </div>
          </div>

          {/* Show Speaker Labels */}
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
              <div style={labelStyle}>Speaker Labels</div>
              <Toggle value={values.showSpeakerLabels ?? false} onChange={(v) => set('showSpeakerLabels', v)} />
            </div>
          </div>

          {/* Max Width */}
          <div>
            <div style={{ ...labelStyle, marginBottom: 6 }}>
              Max Width: <span style={{ color: 'var(--accent-cyan)', fontFamily: 'var(--font-mono)' }}>{values.subtitleMaxWidth ?? 90}%</span>
            </div>
            <input
              type="range" min="50" max="100" step="5"
              value={values.subtitleMaxWidth ?? 90}
              onChange={(e) => set('subtitleMaxWidth', parseInt(e.target.value))}
              style={{ width: '100%', maxWidth: 240, accentColor: 'var(--accent-cyan)' }}
            />
          </div>

          {/* Vertical Offset */}
          <div>
            <div style={{ ...labelStyle, marginBottom: 6 }}>
              Vertical Offset: <span style={{ color: 'var(--accent-cyan)', fontFamily: 'var(--font-mono)' }}>{values.subtitleOffsetV ?? 4}%</span>
            </div>
            <input
              type="range" min="0" max="100" step="1"
              value={values.subtitleOffsetV ?? 4}
              onChange={(e) => set('subtitleOffsetV', parseInt(e.target.value))}
              style={{ width: '100%', maxWidth: 240, accentColor: 'var(--accent-cyan)' }}
            />
          </div>

          {/* Max Words Per Subtitle */}
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: (values.subtitleMaxWords ?? 0) > 0 ? 4 : 0 }}>
              <div style={labelStyle}>Max Words</div>
              <Toggle value={(values.subtitleMaxWords ?? 0) > 0} onChange={(v) => set('subtitleMaxWords', v ? 4 : 0)} />
            </div>
            {(values.subtitleMaxWords ?? 0) > 0 && (
              <div>
                <div style={{ ...labelStyle, marginBottom: 6 }}>
                  Words: <span style={{ color: 'var(--accent-cyan)', fontFamily: 'var(--font-mono)' }}>{values.subtitleMaxWords}</span>
                </div>
                <input
                  type="range" min="1" max="20" step="1"
                  value={values.subtitleMaxWords}
                  onChange={(e) => set('subtitleMaxWords', parseInt(e.target.value))}
                  style={{ width: '100%', maxWidth: 240, accentColor: 'var(--accent-cyan)' }}
                />
              </div>
            )}
          </div>

          {/* Per-Speaker Colors toggle + palette */}
          {speakerList.length > 0 && (
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
                <div style={labelStyle}>Speaker Colors</div>
                <Toggle value={values.useSpeakerColors ?? true} onChange={(v) => set('useSpeakerColors', v)} />
              </div>
              {(values.useSpeakerColors ?? true) && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                  {speakerList.map((sp, idx) => (
                    <div key={sp} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                      <input
                        type="color"
                        value={(values.speakerColors || {})[sp] || DEFAULT_PALETTE[idx % DEFAULT_PALETTE.length]}
                        onChange={(e) => set('speakerColors', { ...values.speakerColors, [sp]: e.target.value })}
                        style={{ width: 22, height: 22, border: 'none', cursor: 'pointer', background: 'none' }}
                      />
                      <span style={{ fontSize: 13, color: 'var(--text-secondary)' }}>{sp}</span>
                    </div>
                  ))}
                </div>
              )}
              {!(values.useSpeakerColors ?? true) && (
                <div style={{ fontSize: 11, color: 'var(--text-muted)', fontStyle: 'italic' }}>
                  All speakers use font color
                </div>
              )}
            </div>
          )}

          {/* Background */}
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
              <div style={labelStyle}>Background Box</div>
              <Toggle value={values.subtitleBgEnabled} onChange={(v) => set('subtitleBgEnabled', v)} />
            </div>
            {values.subtitleBgEnabled && (
              <div style={{ display: 'flex', gap: 10, alignItems: 'center', flexWrap: 'wrap' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                  <span style={{ fontSize: 13, color: 'var(--text-secondary)' }}>Color</span>
                  <input
                    type="color"
                    value={values.subtitleBgColor}
                    onChange={(e) => set('subtitleBgColor', e.target.value)}
                    style={{ width: 22, height: 22, border: 'none', cursor: 'pointer', background: 'none' }}
                  />
                </div>
                <div style={{ flex: 1, minWidth: 100 }}>
                  <div style={{ fontSize: 13, color: 'var(--text-secondary)', marginBottom: 4 }}>
                    Opacity: <span style={{ color: 'var(--accent-cyan)', fontFamily: 'var(--font-mono)' }}>{values.subtitleBgOpacity}%</span>
                  </div>
                  <input
                    type="range" min="10" max="100" step="5"
                    value={values.subtitleBgOpacity}
                    onChange={(e) => set('subtitleBgOpacity', parseInt(e.target.value))}
                    style={{ width: '100%', accentColor: 'var(--accent-cyan)' }}
                  />
                </div>
                {/* Border radius removed — ASS subtitles render rectangular boxes */}
              </div>
            )}
          </div>

          {/* Active Word Highlight */}
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: values.activeWordEnabled ? 8 : 0 }}>
              <div style={labelStyle}>Active Word</div>
              <Toggle value={values.activeWordEnabled || false} onChange={(v) => set('activeWordEnabled', v)} />
            </div>
            {values.activeWordEnabled && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                <div style={{ display: 'flex', gap: 10, alignItems: 'center', flexWrap: 'wrap' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                    <span style={{ fontSize: 13, color: 'var(--text-secondary)' }}>Color</span>
                    <input type="color" value={values.activeWordColor || '#FFD700'}
                      onChange={(e) => set('activeWordColor', e.target.value)}
                      style={{ width: 22, height: 22, border: 'none', cursor: 'pointer', background: 'none' }} />
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                    <span style={{ fontSize: 13, color: 'var(--text-secondary)' }}>Stroke</span>
                    <input type="color" value={values.activeWordOutlineColor || '#000000'}
                      onChange={(e) => set('activeWordOutlineColor', e.target.value)}
                      style={{ width: 22, height: 22, border: 'none', cursor: 'pointer', background: 'none' }} />
                  </div>
                </div>
                <div style={{ display: 'flex', gap: 10, alignItems: 'center', flexWrap: 'wrap' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                    <span style={{ fontSize: 13, color: 'var(--text-secondary)' }}>BG</span>
                    <input type="color" value={values.activeWordBgColor || '#000000'}
                      onChange={(e) => set('activeWordBgColor', e.target.value)}
                      style={{ width: 22, height: 22, border: 'none', cursor: 'pointer', background: 'none' }} />
                  </div>
                  <div style={{ flex: 1, minWidth: 100 }}>
                    <div style={{ fontSize: 13, color: 'var(--text-secondary)', marginBottom: 4 }}>
                      Opacity: <span style={{ color: 'var(--accent-cyan)', fontFamily: 'var(--font-mono)' }}>{values.activeWordBgOpacity ?? 0}%</span>
                    </div>
                    <input type="range" min="0" max="100" step="5"
                      value={values.activeWordBgOpacity ?? 0}
                      onChange={(e) => set('activeWordBgOpacity', parseInt(e.target.value))}
                      style={{ width: '100%', accentColor: 'var(--accent-cyan)' }} />
                  </div>
                  <div style={{ flex: 1, minWidth: 100 }}>
                    <div style={{ fontSize: 13, color: 'var(--text-secondary)', marginBottom: 4 }}>
                      Radius: <span style={{ color: 'var(--accent-cyan)', fontFamily: 'var(--font-mono)' }}>{values.activeWordBgRadius ?? 4}px</span>
                    </div>
                    <input type="range" min="0" max="20" step="1"
                      value={values.activeWordBgRadius ?? 4}
                      onChange={(e) => set('activeWordBgRadius', parseInt(e.target.value))}
                      style={{ width: '100%', accentColor: 'var(--accent-cyan)' }} />
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}


export default function ViralClips() {
  const { isMobile } = useResponsive();
  const { user } = useAuth();
  const encoding = useEncodingManager();
  // Multi-track editor timeline state (global zustand store, populated if user edited in VideoEditor)
  const timelineItems = useTimelineStore((s) => s.items);
  const timelineMediaLibrary = useTimelineStore((s) => s.mediaLibrary);
  const [jobs, setJobs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [filters, setFilters] = useState({ minScore: 0, platform: 'all', sort: 'viral_score', clipType: 'all', source: 'all' });
  const [settings, setSettings] = useState(loadExportSettings);
  const [customFonts, setCustomFonts] = useState([]);
  const [clipOverrides, setClipOverrides] = useState(loadClipOverrides);
  const [editingClip, setEditingClip] = useState(null);
  const [previewClip, setPreviewClip] = useState(null);
  const [previewKey, setPreviewKey] = useState(0); // Incremented to force ClipPreview remount
  const [centerSubjectState, setCenterSubjectState] = useState('idle'); // idle | centering | done
  const [trackingApplied, setTrackingApplied] = useState(false); // flash when tracking updates
  // Staged aspect ratio: pendingAR is what user selected, activeAR is what
  // ClipPreview is currently rendering. They differ only while waiting for
  // subject tracking data from the backend.
  const [trackingLoading, setTrackingLoading] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [settingsAppliedFlash, setSettingsAppliedFlash] = useState(false);
  const settingsAppliedTimerRef = useRef(null);
  const [selectedClips, setSelectedClips] = useState(new Set());
  const [clipSettingsOpen, setClipSettingsOpen] = useState(false);
  // Editable title state: key = clipKey, value = edited title (null = not editing)
  const [editingTitle, setEditingTitle] = useState(null);
  const [editTitleValue, setEditTitleValue] = useState('');
  const [qualityMenuOpen, setQualityMenuOpen] = useState(null); // clip id or null
  const titleInputRef = useRef(null);

  // Persist per-clip overrides
  useEffect(() => {
    try { localStorage.setItem(OVERRIDES_KEY, JSON.stringify(clipOverrides)); } catch {}
  }, [clipOverrides]);

  // Register @font-face for builtin fonts so browser preview matches export
  useEffect(() => {
    Object.entries(BUILTIN_FONT_FILES).forEach(([name, url]) => {
      registerFontFace(name, url);
    });
  }, []);

  // Load custom fonts and register @font-face for preview rendering
  useEffect(() => {
    fetch('/api/fonts')
      .then((r) => r.ok ? r.json() : [])
      .then((fonts) => {
        setCustomFonts(fonts);
        fonts.forEach((f) => registerFontFace(f.name, f.url));
      })
      .catch(() => {});
  }, []);

  const fetchJobs = useCallback(async () => {
    try {
      const res = await fetch('/api/jobs');
      if (!res.ok) return;
      const list = await res.json();
      // Per-user filter for the Clips tab.
      //
      // Regular users already get only their own jobs from ``/api/jobs``
      // (the backend filters by ``owner_user_id`` OR ``owner_username``),
      // but admins see EVERY job. This surface is a "my clips" view — it
      // should show the same completed-with-clips jobs the user sees on
      // the Analysis page.
      //
      // UUID-only matching breaks on deployments whose ``/data/auth``
      // directory was wiped before it was persisted as a docker volume:
      // the admin's UUID was regenerated on seed, so jobs stamped with
      // the old UUID no longer resolve and the Clips tab went empty
      // even though the Analysis page (which loads a specific job by
      // id) kept working. We now fall back to matching on the stored
      // ``owner_username`` (stable for the head admin ``Jadmin``),
      // include admins unconditionally so they have parity with
      // /api/jobs, and still keep legacy pre-auth jobs visible.
      const myUserId = user?.id || '';
      const myUsername = (user?.username || '').trim().toLowerCase();
      const isAdmin = user?.role === 'admin';
      const ownedByMe = (j) => {
        if (isAdmin) return true;
        const owner = j.owner_user_id || '';
        const ownerName = (j.owner_username || '').trim().toLowerCase();
        if (myUserId && owner === myUserId) return true;
        if (myUsername && ownerName && ownerName === myUsername) return true;
        // Legacy pre-auth upload with no owner recorded: only admins
        // see those (the backend already hides them from regular
        // users), but keep the fallback here so the filter stays
        // consistent with /api/jobs' contract.
        return !owner;
      };
      const withClips = list.filter(
        (j) => j.status === 'complete' && j.clips_count > 0 && ownedByMe(j),
      );
      const details = await Promise.all(
        withClips.map((j) =>
          fetch(`/api/jobs/${j.job_id}`)
            .then((r) => r.ok ? r.json() : null)
            .then((d) => d ? sanitizeJob(d) : null)
            .catch(() => null)
        )
      );
      setJobs(details.filter(Boolean));
    } catch {
    } finally {
      setLoading(false);
    }
  }, [user?.id, user?.username, user?.role]);

  useEffect(() => { fetchJobs(); }, [fetchJobs]);

  // ── Server-first subtitle settings persistence ──
  // Load settings from the most recently updated job that has subtitle_settings
  const settingsLoadedFromServer = useRef(false);
  const skipNextServerSave = useRef(false);
  useEffect(() => {
    if (!jobs || jobs.length === 0) return;
    // Find the most recently updated job with subtitle_settings
    const withSettings = jobs
      .filter((j) => j.subtitle_settings && Object.keys(j.subtitle_settings).length > 0)
      .sort((a, b) => (b.updated_at || '').localeCompare(a.updated_at || ''));
    if (withSettings.length > 0 && !settingsLoadedFromServer.current) {
      const serverSettings = sanitizeSubtitleSettings(withSettings[0].subtitle_settings);
      skipNextServerSave.current = true;
      setSettings((prev) => ({ ...prev, ...serverSettings }));
      settingsLoadedFromServer.current = true;
    }
  }, [jobs]);

  // Debounced save settings to all loaded jobs (server is source of truth)
  const saveSettingsTimerRef = useRef(null);
  useEffect(() => {
    if (skipNextServerSave.current) {
      skipNextServerSave.current = false;
      return;
    }
    if (!jobs || jobs.length === 0) return;
    if (saveSettingsTimerRef.current) clearTimeout(saveSettingsTimerRef.current);
    saveSettingsTimerRef.current = setTimeout(() => {
      // Save to all loaded jobs so settings are consistent everywhere
      jobs.forEach((j) => {
        fetch(`/api/jobs/${j.job_id}/subtitle-settings`, {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(settings),
        }).catch(() => {});
      });
    }, 800);
    return () => { if (saveSettingsTimerRef.current) clearTimeout(saveSettingsTimerRef.current); };
  }, [settings, jobs]);

  // --- Per-clip override helpers ---
  // Key must be unique per clip — include start_time as a guard against
  // duplicate IDs from AI providers (which would cause React to drop cards).
  const getClipKey = (clip) => `${clip.jobId}_${clip.id}_${clip.start_time}`;
  const hasCustomSettings = (clip) => {
    const ovr = clipOverrides[getClipKey(clip)];
    return ovr && Object.keys(ovr).length > 0;
  };
  const getClipSettings = (clip) => {
    const ovr = clipOverrides[getClipKey(clip)];
    if (!ovr || Object.keys(ovr).length === 0) return settings;
    return { ...settings, ...ovr };
  };
  // updateClipOverride removed — per-clip edits now accumulate
  // individual key-value changes directly in the SubtitleSettingsEditor
  // onChange handler to prevent full-snapshot overrides.
  const resetClipOverride = (clip) => {
    const key = getClipKey(clip);
    setClipOverrides((prev) => {
      const next = { ...prev };
      delete next[key];
      return next;
    });
  };

  // --- Preview data: settings, transcript, source dims for previewed clip ---
  const previewClipSettings = useMemo(() => {
    if (!previewClip) return settings;
    return getClipSettings(previewClip);
  }, [previewClip, settings, clipOverrides]);

  const previewTranscript = useMemo(() => {
    if (!previewClip) return [];
    const job = jobs.find((j) => j.job_id === previewClip.jobId);
    return job?.translated_transcript?.length ? job.translated_transcript : (job?.transcript || []);
  }, [previewClip, jobs]);

  const previewSourceDims = useMemo(() => {
    if (!previewClip) return { w: 1920, h: 1080 };
    const job = jobs.find((j) => j.job_id === previewClip.jobId);
    if (job?.resolution) {
      const parts = job.resolution.split('x').map(Number);
      if (parts.length === 2 && parts[0] > 0 && parts[1] > 0) {
        return { w: parts[0], h: parts[1] };
      }
    }
    return { w: 1920, h: 1080 };
  }, [previewClip, jobs]);

  const previewSubjectX = useMemo(() => {
    if (!previewClip) return 50;
    const job = jobs.find((j) => j.job_id === previewClip.jobId);
    if (job?.scenes?.length) {
      return computeClipSubjectX(job.scenes, previewClip.start_time, previewClip.end_time);
    }
    return 50;
  }, [previewClip, jobs]);

  const previewScenes = useMemo(() => {
    if (!previewClip) return [];
    const job = jobs.find((j) => j.job_id === previewClip.jobId);
    return job?.scenes || [];
  }, [previewClip, jobs]);

  // --- Per-clip subject tracking: auto-apply when clip or aspect ratio changes ---
  // Uses a staged AR pattern: pendingAR (previewClipSettings.aspectRatio) vs
  // activeAspectRatio. The preview only switches AR once tracking data is ready.
  const prevTrackingRef = useRef({ clipId: null, jobId: null, ar: null });
  // The AR that ClipPreview is actually rendering — lags behind
  // previewClipSettings.aspectRatio while tracking data loads.
  const [activeAspectRatio, setActiveAspectRatio] = useState(
    () => previewClipSettings?.aspectRatio || null
  );

  // Sync activeAR when clip changes (new clip should start at its settings' AR)
  useEffect(() => {
    if (!previewClip) {
      setActiveAspectRatio(null);
      setTrackingLoading(false);
      return;
    }
    // When a new clip opens, check if it has AI data immediately
    const job = jobs.find((j) => j.job_id === previewClip.jobId);
    const scenes = job?.scenes || [];
    const hasAiData = scenes.some((s) => {
      const sx = typeof s === 'object' ? (s.subject_x ?? 50) : 50;
      return sx !== 50;
    });
    if (hasAiData) {
      setActiveAspectRatio(previewClipSettings.aspectRatio || null);
    }
  }, [previewClip?.id, previewClip?.jobId]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    const requestedAr = previewClipSettings.aspectRatio;
    const prev = prevTrackingRef.current;
    const clipId = previewClip?.id ?? null;
    const jobId = previewClip?.jobId ?? null;

    const clipChanged = clipId !== prev.clipId || jobId !== prev.jobId;
    const arChanged = requestedAr !== prev.ar;
    prevTrackingRef.current = { clipId, jobId, ar: requestedAr };

    if (!previewClip || !requestedAr) {
      setActiveAspectRatio(requestedAr || null);
      setTrackingLoading(false);
      return;
    }
    if (!clipChanged && !arChanged) return;

    const job = jobs.find((j) => j.job_id === previewClip.jobId);
    const scenes = job?.scenes || [];
    const hasAiData = scenes.some((s) => {
      const sx = typeof s === 'object' ? (s.subject_x ?? 50) : 50;
      return sx !== 50;
    });

    if (hasAiData) {
      // CASE A: Data exists — processKeyframes() runs synchronously inside
      // ClipPreview's useMemo. Apply the new AR immediately.
      setActiveAspectRatio(requestedAr);
      setTrackingLoading(false);
      if (clipChanged) {
        setPreviewKey((k) => k + 1);
      }
      setTrackingApplied(true);
      setTimeout(() => setTrackingApplied(false), 2000);
    } else {
      // CASE B: No AI data — keep preview at OLD AR while backend analyzes.
      setTrackingLoading(true);
      setCenterSubjectState('centering');

      fetch(`/api/jobs/${previewClip.jobId}/recenter-subject`, { method: 'POST' })
        .then((res) => {
          if (!res.ok) throw new Error('recenter failed');
          return res.json();
        })
        .then((data) => {
          if (data.status === 'reanalyzing') {
            showToast(`Analyzing subject position for ${requestedAr} crop...`, 'info');
            const poll = setInterval(async () => {
              try {
                const jr = await fetch(`/api/jobs/${previewClip.jobId}`, { cache: 'no-store' });
                if (!jr.ok) return;
                const jd = await jr.json();
                const sx = (jd.scenes || []).map((s) => s.subject_x);
                if (sx.some((v) => v !== 50)) {
                  clearInterval(poll);
                  setJobs((prev) => prev.map((j) =>
                    j.job_id === previewClip.jobId ? sanitizeJob(jd) : j
                  ));
                  // NOW apply the new AR — data is ready
                  setActiveAspectRatio(requestedAr);
                  setPreviewKey((k) => k + 1);
                  setTrackingLoading(false);
                  setCenterSubjectState('done');
                  setTrackingApplied(true);
                  showToast('Subject tracking ready — preview updated', 'success');
                  setTimeout(() => {
                    setCenterSubjectState('idle');
                    setTrackingApplied(false);
                  }, 2200);
                }
              } catch { /* ignore polling errors */ }
            }, 3000);
            // Timeout: if tracking doesn't complete in 2 minutes, apply AR anyway
            setTimeout(() => {
              clearInterval(poll);
              setActiveAspectRatio(requestedAr);
              setTrackingLoading(false);
              setCenterSubjectState((s) => s === 'centering' ? 'idle' : s);
            }, 120000);
          } else {
            // Backend says data already exists — refresh and apply
            fetch(`/api/jobs/${previewClip.jobId}`, { cache: 'no-store' })
              .then((jr) => jr.ok ? jr.json() : null)
              .then((jd) => {
                if (jd) setJobs((prev) => prev.map((j) =>
                  j.job_id === previewClip.jobId ? sanitizeJob(jd) : j
                ));
              })
              .catch(() => {});
            setActiveAspectRatio(requestedAr);
            setPreviewKey((k) => k + 1);
            setTrackingLoading(false);
            setCenterSubjectState('done');
            setTrackingApplied(true);
            showToast('Subject tracking ready — preview updated', 'success');
            setTimeout(() => {
              setCenterSubjectState('idle');
              setTrackingApplied(false);
            }, 2200);
          }
        })
        .catch(() => {
          // Network error — apply AR anyway as fallback
          setActiveAspectRatio(requestedAr);
          setTrackingLoading(false);
          setCenterSubjectState('idle');
        });
    }
  }, [previewClip?.id, previewClip?.jobId, previewClipSettings.aspectRatio]); // eslint-disable-line react-hooks/exhaustive-deps

  // --- Auto-refresh word timestamps for jobs that lack them ---
  // When active word highlighting is enabled and the current jobs' transcripts
  // don't have per-word timestamps (pre-existing transcripts), automatically
  // trigger a backend refresh so word highlighting uses real Whisper timing.
  const wordRefreshTriggeredRef = useRef(new Set());
  useEffect(() => {
    // Determine if active word is enabled in global settings or any clip override
    const awGlobal = settings.activeWordEnabled;
    const awAnyOverride = Object.values(clipOverrides).some((ov) => ov.activeWordEnabled);
    if (!awGlobal && !awAnyOverride) return;

    for (const job of jobs) {
      if (!job.transcript || job.transcript.length === 0) continue;
      // Already triggered for this job
      if (wordRefreshTriggeredRef.current.has(job.job_id)) continue;
      // Check if any segment already has word timestamps
      const hasWords = job.transcript.some((seg) => seg.words && seg.words.length > 0);
      if (hasWords) continue;

      // This job needs word timestamps — trigger refresh
      wordRefreshTriggeredRef.current.add(job.job_id);
      fetch(`/api/jobs/${job.job_id}/refresh-word-timestamps`, { method: 'POST' })
        .then((r) => r.json())
        .then((data) => {
          if (data.status === 'started') {
            showToast('Extracting word timestamps for accurate highlighting...', 'info');
            // Poll until complete, then re-fetch job data
            const poll = setInterval(async () => {
              try {
                const jr = await fetch(`/api/jobs/${job.job_id}`);
                if (!jr.ok) return;
                const jd = await jr.json();
                const nowHasWords = (jd.transcript || []).some((s) => s.words && s.words.length > 0);
                if (nowHasWords) {
                  clearInterval(poll);
                  setJobs((prev) => prev.map((j) => j.job_id === job.job_id ? sanitizeJob(jd) : j));
                  showToast('Word timestamps ready — active word highlighting is now accurate', 'success');
                }
              } catch { /* ignore */ }
            }, 3000);
            // Safety: stop polling after 5 minutes
            setTimeout(() => clearInterval(poll), 300000);
          }
        })
        .catch(() => {});
    }
  }, [settings.activeWordEnabled, clipOverrides, jobs]);

  // Auto-apply flash indicator: deep compare, excluding transient keys that
  // get initialized asynchronously (speakerColors, speakerNames)
  const prevSettingsJsonRef = useRef('');
  const settingsToCompareJson = useCallback((s) => {
    if (!s) return '';
    const { speakerColors, speakerNames, ...rest } = s;
    return JSON.stringify(rest);
  }, []);
  useEffect(() => {
    const json = settingsToCompareJson(previewClipSettings);
    if (!previewClip) { prevSettingsJsonRef.current = json; return; }
    if (prevSettingsJsonRef.current && prevSettingsJsonRef.current !== json) {
      setSettingsAppliedFlash(true);
      if (settingsAppliedTimerRef.current) clearTimeout(settingsAppliedTimerRef.current);
      settingsAppliedTimerRef.current = setTimeout(() => setSettingsAppliedFlash(false), 1800);
    }
    prevSettingsJsonRef.current = json;
  }, [previewClipSettings, previewClip, settingsToCompareJson]);
  useEffect(() => () => { if (settingsAppliedTimerRef.current) clearTimeout(settingsAppliedTimerRef.current); }, []);

  const handleApplySettings = useCallback((applied) => {
    setSettings(applied);
    setSettingsAppliedFlash(true);
    if (settingsAppliedTimerRef.current) clearTimeout(settingsAppliedTimerRef.current);
    settingsAppliedTimerRef.current = setTimeout(() => setSettingsAppliedFlash(false), 1800);
    showToast('Settings applied to preview & export', 'success');
  }, []);

  const handleExportClip = (jobId, clip, qualityOverride) => {
    const cs = getClipSettings(clip);
    const quality = qualityOverride || cs.exportQuality || '1080p';
    const body = {
      start: clip.start_time,
      end: clip.end_time,
      clip_id: clip.id,
      clip_title: clip.title || `Clip ${clip.id}`,
      export_quality: quality,
    };
    if (cs.aspectRatio) body.aspect_ratio = cs.aspectRatio;
    const globalSubsOn = cs.subtitlesEnabled || false;
    const anySegmentSubsOn = useTimelineStore.getState().segments?.some(s => s.subtitlesEnabled !== false) || false;
    const needsSubs = globalSubsOn || anySegmentSubsOn;
    body.subtitles_enabled = needsSubs;
    body.global_subtitles_enabled = globalSubsOn;
    if (needsSubs) {
      body.subtitle_settings = mapSubtitleSettings(cs);
    }
    // Include playback volume/speed if non-default
    if (cs.playbackVolume != null && cs.playbackVolume !== 100) body.volume = cs.playbackVolume / 100;
    if (cs.playbackSpeed != null && cs.playbackSpeed !== 1.0) body.speed = cs.playbackSpeed;

    // Include multi-track editor video effects + transform so export matches preview
    const videoEffects = buildVideoEffectsPayload(timelineItems);
    if (videoEffects) body.video_effects = videoEffects;

    // Build overlay arrays via shared utility (consistent filtering + validation)
    const overlays = buildOverlayPayload({
      timelineItems,
      mediaLibrary: timelineMediaLibrary,
      clipStart: clip.start_time,
      tracks: useTimelineStore.getState().tracks,
    });
    if (overlays.textOverlays.length > 0) body.text_overlays = overlays.textOverlays;
    if (overlays.imageOverlays.length > 0) body.image_overlays = overlays.imageOverlays;
    if (overlays.shapeOverlays.length > 0) body.shape_overlays = overlays.shapeOverlays;
    if (overlays.audioOverlays.length > 0) body.audio_overlays = overlays.audioOverlays;
    if (overlays.compositingOrder?.length > 0) {
      body.overlay_compositing_order = overlays.compositingOrder;
    }
    if (overlays.warnings.length > 0) {
      for (const w of overlays.warnings) console.warn(`[Export] ${w}`);
    }

    // Diagnostic logging: full export payload for debugging overlay/settings issues
    console.log('[ViralClips Export] clip:', clip.id, 'payload:', JSON.stringify({
      aspect_ratio: body.aspect_ratio,
      subtitles_enabled: body.subtitles_enabled,
      subtitle_settings: body.subtitle_settings ? 'YES' : 'NO',
      video_effects: body.video_effects ? 'YES' : 'NO',
      text_overlays: body.text_overlays?.length || 0,
      image_overlays: body.image_overlays?.length || 0,
      shape_overlays: body.shape_overlays?.length || 0,
      audio_overlays: body.audio_overlays?.length || 0,
    }));

    encoding.startExport(jobId, clip.id, clip.title || `Clip ${clip.id}`, body);
    showToast(`Exporting "${clip.title || `Clip ${clip.id}`}" at ${quality}...`, 'info');
  };

  // Save edited clip title to backend
  const handleSaveTitle = async (clip) => {
    const newTitle = editTitleValue.trim();
    if (!newTitle || newTitle === clip.title) {
      setEditingTitle(null);
      return;
    }
    try {
      const res = await fetch(`/api/jobs/${clip.jobId}/clips/${clip.id}/title`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title: newTitle }),
      });
      if (res.ok) {
        // Update the local jobs data to reflect the new title
        setJobs((prev) => prev.map((job) => {
          if (job.job_id !== clip.jobId) return job;
          return {
            ...job,
            clips: job.clips.map((c) =>
              c.id === clip.id ? { ...c, title: newTitle } : c
            ),
          };
        }));
        showToast(`Title updated to "${newTitle}"`, 'success');
      } else {
        showToast('Failed to update title', 'error');
      }
    } catch (err) {
      showToast(`Error: ${err.message}`, 'error');
    }
    setEditingTitle(null);
  };

  const handleDeleteClip = async (clip) => {
    try {
      const res = await fetch(`/api/jobs/${clip.jobId}/clips/${clip.id}`, { method: 'DELETE' });
      if (res.ok) {
        setJobs((prev) => prev.map((job) => {
          if (job.job_id !== clip.jobId) return job;
          return {
            ...job,
            clips: job.clips.filter((c) => c.id !== clip.id),
            exported_clips: (job.exported_clips || []).filter((ec) => ec.clip_id !== clip.id),
          };
        }));
        const clipKey = getClipKey(clip);
        setSelectedClips((prev) => {
          const next = new Set(prev);
          next.delete(clipKey);
          return next;
        });
        showToast(`Clip ${clip.id} deleted`, 'info');
      } else {
        const err = await res.json().catch(() => ({}));
        showToast(err.detail || 'Failed to delete clip', 'error');
      }
    } catch {
      showToast('Failed to delete clip', 'error');
    }
  };

  // Collect all clips across jobs
  const allClips = [];
  const allSpeakers = new Set();
  jobs.forEach((job) => {
    const fileExt = job.file_path?.split('.').pop() || 'mp4';
    (job.clips || []).forEach((clip) => {
      allClips.push({ ...clip, jobId: job.job_id, filename: job.filename, fileExt, _scenes: job.scenes || [] });
    });
    (job.transcript || []).forEach((seg) => {
      if (seg.speaker) allSpeakers.add(seg.speaker);
    });
  });
  const speakerList = [...allSpeakers];

  // Derive unique clip types and source filenames for filter dropdowns
  const clipTypes = [...new Set(allClips.map((c) => c.clip_type).filter(Boolean))].sort();
  const sourceFiles = [...new Set(allClips.map((c) => c.filename).filter(Boolean))].sort();

  // Apply filters
  let filtered = [...allClips];
  if (filters.minScore > 0) filtered = filtered.filter((c) => c.viral_score >= filters.minScore);
  if (filters.platform !== 'all') filtered = filtered.filter((c) => c.platform === filters.platform || c.platform === 'both');
  if (filters.clipType !== 'all') filtered = filtered.filter((c) => c.clip_type === filters.clipType);
  if (filters.source !== 'all') filtered = filtered.filter((c) => c.filename === filters.source);

  // Apply text search filter — covers all visible text on the clip card
  if (searchQuery.trim()) {
    const q = searchQuery.trim().toLowerCase();
    filtered = filtered.filter((c) =>
      (c.title || '').toLowerCase().includes(q) ||
      (c.suggested_caption || '').toLowerCase().includes(q) ||
      (c.hook_text || '').toLowerCase().includes(q) ||
      (c.why_this_works || '').toLowerCase().includes(q) ||
      (c.clip_type || '').toLowerCase().includes(q) ||
      (c.platform || '').toLowerCase().includes(q) ||
      (c.filename || '').toLowerCase().includes(q) ||
      (c.viral_score_reasoning || '').toLowerCase().includes(q) ||
      (c.clip_focus || '').toLowerCase().includes(q) ||
      String(c.id).includes(q) ||
      String(c.viral_score).includes(q) ||
      (c.suggested_hashtags || []).some((h) => h.toLowerCase().includes(q))
    );
  }

  filtered.sort((a, b) => {
    if (filters.sort === 'duration') return b.duration - a.duration;
    if (filters.sort === 'duration_short') return a.duration - b.duration;
    if (filters.sort === 'timestamp') return a.start_time - b.start_time;
    if (filters.sort === 'score_low') return a.viral_score - b.viral_score;
    if (filters.sort === 'title_az') return (a.title || '').localeCompare(b.title || '');
    return b.viral_score - a.viral_score;
  });

  const handleDeleteSelected = async () => {
    if (selectedClips.size === 0) return;
    if (!window.confirm(`Delete ${selectedClips.size} selected clip(s)? This cannot be undone.`)) return;

    // Group by jobId for bulk deletion
    const byJob = {};
    for (const clipKey of selectedClips) {
      const clip = filtered.find((c) => getClipKey(c) === clipKey);
      if (clip) {
        if (!byJob[clip.jobId]) byJob[clip.jobId] = [];
        byJob[clip.jobId].push(clip.id);
      }
    }

    let totalDeleted = 0;
    for (const [jobId, clipIds] of Object.entries(byJob)) {
      try {
        const res = await fetch(`/api/jobs/${jobId}/delete-clips`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ clip_ids: clipIds }),
        });
        if (res.ok) {
          const data = await res.json();
          totalDeleted += data.deleted_count;
          const idsSet = new Set(clipIds);
          setJobs((prev) => prev.map((job) => {
            if (job.job_id !== jobId) return job;
            return {
              ...job,
              clips: job.clips.filter((c) => !idsSet.has(c.id)),
              exported_clips: (job.exported_clips || []).filter((ec) => !idsSet.has(ec.clip_id)),
            };
          }));
        }
      } catch {
        // continue with other jobs
      }
    }
    setSelectedClips(new Set());
    if (totalDeleted > 0) {
      showToast(`Deleted ${totalDeleted} clip(s)`, 'info');
    }
  };

  const totalClips = allClips.length;

  if (loading) {
    return (
      <div style={{ textAlign: 'center', padding: 48, color: 'var(--text-secondary)' }}>
        <div style={{
          width: 24, height: 24, border: '2px solid var(--border)', borderTopColor: 'var(--accent-cyan)',
          borderRadius: '50%', animation: 'spin 0.8s linear infinite',
          margin: '0 auto 12px',
        }} />
        Loading clips...
      </div>
    );
  }

  return (
    <div>
      <h2 style={{ fontSize: 20, marginBottom: 24 }}>Viral Clips</h2>

      {/* Stats */}
      <div style={{ display: 'flex', gap: 16, marginBottom: 24, flexWrap: 'wrap' }}>
        <div style={{ background: 'var(--bg-panel)', border: '1px solid var(--border)', padding: '12px 20px', flex: 1, minWidth: 140 }}>
          <div style={{ fontSize: 10, fontFamily: 'var(--font-mono)', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.1em' }}>
            Total Clips
          </div>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: 28, fontWeight: 700, color: 'var(--accent-amber)' }}>
            {totalClips}
          </div>
        </div>
        <div style={{ background: 'var(--bg-panel)', border: '1px solid var(--border)', padding: '12px 20px', flex: 1, minWidth: 140 }}>
          <div style={{ fontSize: 10, fontFamily: 'var(--font-mono)', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.1em' }}>
            Videos with Clips
          </div>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: 28, fontWeight: 700, color: 'var(--accent-cyan)' }}>
            {jobs.length}
          </div>
        </div>
        <div style={{ background: 'var(--bg-panel)', border: '1px solid var(--border)', padding: '12px 20px', flex: 1, minWidth: 140 }}>
          <div style={{ fontSize: 10, fontFamily: 'var(--font-mono)', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.1em' }}>
            Avg Score
          </div>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: 28, fontWeight: 700, color: 'var(--text-primary)' }}>
            {totalClips > 0 ? Math.round(allClips.reduce((sum, c) => sum + c.viral_score, 0) / totalClips) : '-'}
          </div>
        </div>
      </div>

      {/* Clip Settings Panel — Collapsible */}
      {totalClips > 0 && (
        <div style={{ marginBottom: 20 }}>
          <button
            onClick={() => setClipSettingsOpen((v) => !v)}
            style={{
              width: '100%',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              padding: '10px 14px',
              background: 'var(--bg-panel)',
              border: '1px solid var(--border)',
              borderRadius: clipSettingsOpen ? 'var(--radius-md) var(--radius-md) 0 0' : 'var(--radius-md)',
              cursor: 'pointer',
              color: 'var(--text-primary)',
              fontSize: 13,
              fontWeight: 600,
              letterSpacing: '0.02em',
              transition: 'border-radius 0.2s ease',
            }}
          >
            <span style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--accent-cyan)', textTransform: 'uppercase', letterSpacing: '0.06em' }}>
                Clip Settings
              </span>
              {!clipSettingsOpen && settings.subtitlesEnabled && (
                <span style={{ fontSize: 9, fontFamily: 'var(--font-mono)', color: 'var(--accent-cyan)', background: 'var(--accent-cyan-dim)', padding: '2px 6px', borderRadius: 3 }}>SUBS</span>
              )}
              {!clipSettingsOpen && settings.aspectRatio && (
                <span style={{ fontSize: 9, fontFamily: 'var(--font-mono)', color: 'var(--accent-amber)', background: 'rgba(245,158,11,0.1)', padding: '2px 6px', borderRadius: 3 }}>{String(settings.aspectRatio || '')}</span>
              )}
            </span>
            <span style={{ fontSize: 14, color: 'var(--text-muted)', transition: 'transform 0.2s ease', transform: clipSettingsOpen ? 'rotate(180deg)' : 'rotate(0deg)' }}>
              &#x25BC;
            </span>
          </button>
          {clipSettingsOpen && (
            <div style={{
              border: '1px solid var(--border)',
              borderTop: 'none',
              borderRadius: '0 0 var(--radius-md) var(--radius-md)',
              overflow: 'hidden',
            }}>
              <EditorErrorBoundary name="Clip settings" compact>
                <ClipSettingsPanel
                  speakers={speakerList}
                  onSettingsChange={setSettings}
                  onApplySettings={handleApplySettings}
                />
              </EditorErrorBoundary>
            </div>
          )}
        </div>
      )}

      {/* Search Bar */}
      {totalClips > 0 && (
        <div style={{ position: 'relative', marginBottom: 16 }}>
          <div style={{
            position: 'absolute', left: 14, top: '50%', transform: 'translateY(-50%)',
            color: 'var(--text-muted)', fontSize: 15, pointerEvents: 'none', lineHeight: 1,
          }}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="11" cy="11" r="8" /><line x1="21" y1="21" x2="16.65" y2="16.65" />
            </svg>
          </div>
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search clips by title, caption, hook, type..."
            style={{
              width: '100%',
              padding: '10px 14px 10px 40px',
              fontSize: 14,
              background: 'var(--bg-panel)',
              border: '1px solid var(--border)',
              borderRadius: 'var(--radius-md)',
              color: 'var(--text-primary)',
              outline: 'none',
              transition: 'border-color 0.2s ease, box-shadow 0.2s ease',
            }}
            onFocus={(e) => {
              e.target.style.borderColor = 'var(--accent-cyan)';
              e.target.style.boxShadow = '0 0 0 3px var(--accent-cyan-dim)';
            }}
            onBlur={(e) => {
              e.target.style.borderColor = 'var(--border)';
              e.target.style.boxShadow = 'none';
            }}
          />
          {searchQuery && (
            <button
              onClick={() => setSearchQuery('')}
              style={{
                position: 'absolute', right: 10, top: '50%', transform: 'translateY(-50%)',
                background: 'var(--bg-elevated)', border: 'none', borderRadius: '50%',
                width: 20, height: 20, display: 'flex', alignItems: 'center', justifyContent: 'center',
                color: 'var(--text-muted)', fontSize: 12, cursor: 'pointer', lineHeight: 1,
              }}
            >
              &times;
            </button>
          )}
        </div>
      )}

      {/* Filters & Sort */}
      {totalClips > 0 && (
        <div style={{
          display: 'flex', gap: 10, marginBottom: 16, flexWrap: 'wrap', alignItems: 'center',
          padding: '10px 14px', background: 'var(--bg-panel)', border: '1px solid var(--border)',
          borderRadius: 'var(--radius-md)',
        }}>
          <label style={{ fontSize: 12, color: 'var(--text-secondary)', display: 'flex', alignItems: 'center', gap: 6 }}>
            Score
            <input
              type="range"
              min="0"
              max="100"
              value={filters.minScore}
              onChange={(e) => setFilters((f) => ({ ...f, minScore: parseInt(e.target.value) }))}
              style={{ width: 80, accentColor: 'var(--accent-cyan)' }}
            />
            <span style={{ fontFamily: 'var(--font-mono)', width: 28, fontSize: 12, color: 'var(--accent-cyan)' }}>{filters.minScore}</span>
          </label>
          <select
            value={filters.platform}
            onChange={(e) => setFilters((f) => ({ ...f, platform: e.target.value }))}
            style={{ padding: '6px 10px', borderRadius: 'var(--radius-sm)', fontSize: 12, background: 'var(--bg-elevated)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
          >
            <option value="all">All Platforms</option>
            <option value="tiktok">TikTok</option>
            <option value="youtube_shorts">YouTube Shorts</option>
            <option value="both">Both</option>
          </select>
          {clipTypes.length > 1 && (
            <select
              value={filters.clipType}
              onChange={(e) => setFilters((f) => ({ ...f, clipType: e.target.value }))}
              style={{ padding: '6px 10px', borderRadius: 'var(--radius-sm)', fontSize: 12, background: 'var(--bg-elevated)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
            >
              <option value="all">All Types</option>
              {clipTypes.map((t) => (
                <option key={t} value={t}>{t.charAt(0).toUpperCase() + t.slice(1)}</option>
              ))}
            </select>
          )}
          {sourceFiles.length > 1 && (
            <select
              value={filters.source}
              onChange={(e) => setFilters((f) => ({ ...f, source: e.target.value }))}
              style={{ padding: '6px 10px', borderRadius: 'var(--radius-sm)', fontSize: 12, background: 'var(--bg-elevated)', border: '1px solid var(--border)', color: 'var(--text-primary)', maxWidth: 180 }}
            >
              <option value="all">All Videos</option>
              {sourceFiles.map((f) => (
                <option key={f} value={f}>{f.length > 25 ? f.slice(0, 22) + '...' : f}</option>
              ))}
            </select>
          )}
          <select
            value={filters.sort}
            onChange={(e) => setFilters((f) => ({ ...f, sort: e.target.value }))}
            style={{ padding: '6px 10px', borderRadius: 'var(--radius-sm)', fontSize: 12, background: 'var(--bg-elevated)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
          >
            <option value="viral_score">Score: High to Low</option>
            <option value="score_low">Score: Low to High</option>
            <option value="duration">Duration: Longest</option>
            <option value="duration_short">Duration: Shortest</option>
            <option value="timestamp">Timestamp</option>
            <option value="title_az">Title: A-Z</option>
          </select>

          {selectedClips.size > 0 && (
            <button
              onClick={() => {
                for (const clipKey of selectedClips) {
                  const clip = filtered.find((c) => getClipKey(c) === clipKey);
                  if (clip) handleExportClip(clip.jobId, clip);
                }
              }}
              style={{
                padding: '6px 16px',
                background: 'var(--accent-cyan)',
                color: 'var(--bg-base)',
                border: 'none',
                borderRadius: 'var(--radius-sm)',
                fontSize: 12,
                fontWeight: 600,
                cursor: 'pointer',
                whiteSpace: 'nowrap',
              }}
            >
              Export Selected ({selectedClips.size})
            </button>
          )}
          {selectedClips.size > 0 && (
            <button
              onClick={handleDeleteSelected}
              style={{
                padding: '6px 16px',
                background: 'var(--danger-dim, rgba(255,59,48,0.08))',
                color: 'var(--danger, #ff3b30)',
                border: '1px solid var(--danger, #ff3b30)',
                borderRadius: 'var(--radius-sm)',
                fontSize: 12,
                fontWeight: 600,
                cursor: 'pointer',
                whiteSpace: 'nowrap',
              }}
            >
              Delete Selected ({selectedClips.size})
            </button>
          )}

          <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 10 }}>
            <button
              onClick={() => {
                const allKeys = filtered.map((c) => getClipKey(c));
                const allSelected = allKeys.every((k) => selectedClips.has(k));
                if (allSelected) {
                  setSelectedClips(new Set());
                } else {
                  setSelectedClips(new Set(allKeys));
                }
              }}
              style={{
                padding: '4px 10px',
                background: 'none',
                border: '1px solid var(--border)',
                borderRadius: 'var(--radius-sm)',
                fontSize: 11,
                color: 'var(--text-secondary)',
                cursor: 'pointer',
                whiteSpace: 'nowrap',
              }}
            >
              {filtered.length > 0 && filtered.every((c) => selectedClips.has(getClipKey(c)))
                ? 'Deselect All'
                : 'Select All'}
            </button>
            <span style={{
              fontSize: 11,
              fontFamily: 'var(--font-mono)',
              color: 'var(--text-muted)',
              textTransform: 'uppercase',
              letterSpacing: '0.05em',
            }}>
              {filtered.length} of {totalClips}
            </span>
          </div>
        </div>
      )}

      {/* Clip grid */}
      {filtered.length > 0 ? (
        <div className="responsive-grid" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(min(340px, 100%), 1fr))', gap: isMobile ? 12 : 16 }}>
          {filtered.map((clip) => {
            const clipKey = getClipKey(clip);
            const isEditing = editingClip === clipKey;
            const isCustom = hasCustomSettings(clip);
            const clipSettings = getClipSettings(clip);
            const scoreColor = clip.viral_score >= 80
              ? 'var(--accent-amber)'
              : clip.viral_score >= 50
                ? 'var(--accent-cyan)'
                : 'var(--text-secondary)';
            return (
              <div
                key={clipKey}
                className="card-hover slide-in"
                style={{
                  background: 'var(--bg-panel)',
                  border: `1px solid ${isCustom ? 'var(--accent-cyan)' : 'var(--border)'}`,
                  padding: 16,
                  // Grid items default to ``min-width: auto`` which lets long
                  // unbreakable content (captions, focus topics, filenames)
                  // push the card past its grid column. Constrain to the
                  // cell and clip anything that still overflows so badges,
                  // titles, and action buttons stay inside the rounded
                  // border of the card.
                  minWidth: 0,
                  overflow: 'hidden',
                }}
              >
                {/* Source video label */}
                <div style={{ marginBottom: 8, display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 8, minWidth: 0 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, minWidth: 0, flex: '1 1 0%' }}>
                    <input
                      type="checkbox"
                      checked={selectedClips.has(clipKey)}
                      onChange={(e) => {
                        setSelectedClips((prev) => {
                          const next = new Set(prev);
                          if (e.target.checked) next.add(clipKey);
                          else next.delete(clipKey);
                          return next;
                        });
                      }}
                      style={{ accentColor: 'var(--accent-cyan)', cursor: 'pointer', width: 15, height: 15, flexShrink: 0 }}
                    />
                    <Link
                      to={`/analysis/${clip.jobId}`}
                      title={clip.filename}
                      style={{
                        fontSize: 10,
                        fontFamily: 'var(--font-mono)',
                        color: 'var(--accent-cyan)',
                        textDecoration: 'none',
                        textTransform: 'uppercase',
                        letterSpacing: '0.05em',
                        minWidth: 0,
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                        whiteSpace: 'nowrap',
                      }}
                    >
                      {clip.filename}
                    </Link>
                  </div>
                  {isCustom && (
                    <span style={{
                      fontSize: 9,
                      fontFamily: 'var(--font-mono)',
                      color: 'var(--accent-cyan)',
                      background: 'var(--accent-cyan-dim)',
                      padding: '2px 6px',
                      borderRadius: 3,
                      textTransform: 'uppercase',
                      letterSpacing: '0.05em',
                    }}>
                      Custom
                    </span>
                  )}
                </div>

                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 12 }}>
                  <span style={{ fontFamily: 'var(--font-mono)', fontSize: 12, color: 'var(--text-secondary)' }}>
                    CLIP {clip.id}
                  </span>
                  <div style={{ textAlign: 'right' }}>
                    <div style={{ fontFamily: 'var(--font-mono)', fontSize: 20, fontWeight: 700, color: clip.clip_focus ? 'var(--success)' : scoreColor }}>
                      {String(clip.focus_relevance || clip.viral_score || '')}
                    </div>
                    <div style={{ fontSize: 10, color: clip.clip_focus ? 'var(--success)' : 'var(--text-secondary)', textTransform: 'uppercase' }}>
                      {clip.clip_focus ? String(clip.focus_tier || 'FOCUS') : '/100'}
                    </div>
                  </div>
                </div>

                {/* Editable Title */}
                {editingTitle === clipKey ? (
                  <div style={{ display: 'flex', gap: 6, alignItems: 'center', marginBottom: 8 }}>
                    <input
                      ref={titleInputRef}
                      type="text"
                      value={editTitleValue}
                      onChange={(e) => setEditTitleValue(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter') handleSaveTitle(clip);
                        if (e.key === 'Escape') setEditingTitle(null);
                      }}
                      autoFocus
                      style={{
                        flex: 1, padding: '4px 8px', fontSize: 14, fontWeight: 600,
                        background: 'var(--bg-elevated)', color: 'var(--text-primary)',
                        border: '1px solid var(--accent-cyan)', borderRadius: 'var(--radius-sm)',
                        outline: 'none', lineHeight: 1.3,
                      }}
                    />
                    <button
                      onClick={() => handleSaveTitle(clip)}
                      style={{
                        padding: '4px 8px', fontSize: 11, fontWeight: 600,
                        background: 'var(--accent-cyan)', color: 'var(--bg-base)',
                        border: 'none', borderRadius: 'var(--radius-sm)', cursor: 'pointer',
                      }}
                    >
                      Save
                    </button>
                    <button
                      onClick={() => setEditingTitle(null)}
                      style={{
                        padding: '4px 8px', fontSize: 11,
                        background: 'none', color: 'var(--text-muted)',
                        border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)', cursor: 'pointer',
                      }}
                    >
                      Cancel
                    </button>
                  </div>
                ) : (
                  <h4
                    onClick={() => { setEditingTitle(clipKey); setEditTitleValue(clip.title || ''); }}
                    title={String(clip.title || '')}
                    style={{
                      fontSize: 14, marginBottom: 8, lineHeight: 1.3, cursor: 'pointer',
                      borderBottom: '1px dashed transparent',
                      transition: 'border-color 0.15s',
                      overflowWrap: 'anywhere',
                      wordBreak: 'break-word',
                    }}
                    onMouseEnter={(e) => e.target.style.borderBottomColor = 'var(--accent-cyan)'}
                    onMouseLeave={(e) => e.target.style.borderBottomColor = 'transparent'}
                  >
                    {String(clip.title || '')}
                    <span style={{ fontSize: 10, color: 'var(--text-muted)', marginLeft: 6, opacity: 0.6 }}>&#x270E;</span>
                  </h4>
                )}

                <div style={{ display: 'flex', gap: 8, marginBottom: 12, flexWrap: 'wrap' }}>
                  <span style={{ fontFamily: 'var(--font-mono)', fontSize: 12, color: 'var(--accent-cyan)' }}>
                    {formatDuration(clip.start_time)} &rarr; {formatDuration(clip.end_time)}
                  </span>
                  <span style={{ fontFamily: 'var(--font-mono)', fontSize: 12, color: 'var(--text-secondary)' }}>
                    ({formatDuration(clip.duration)})
                  </span>
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
                  {clip.focus_relevance != null && (
                    <span className="badge" style={{
                      background: clip.focus_tier === 'strong' ? 'rgba(52,199,89,0.15)' : clip.focus_tier === 'moderate' ? 'rgba(255,214,0,0.15)' : 'rgba(142,142,147,0.15)',
                      color: clip.focus_tier === 'strong' ? 'var(--success)' : clip.focus_tier === 'moderate' ? 'var(--accent-amber)' : 'var(--text-secondary)',
                      border: `1px solid ${clip.focus_tier === 'strong' ? 'var(--success)' : clip.focus_tier === 'moderate' ? 'var(--accent-amber)' : 'var(--text-secondary)'}`,
                    }}>
                      Relevance: {clip.focus_relevance}/100
                    </span>
                  )}
                </div>

                {/* Detected subject from scene analysis */}
                {clip._scenes?.length > 0 && (() => {
                  const inRange = clip._scenes.filter(
                    (s) => s.timestamp >= clip.start_time && s.timestamp <= clip.end_time
                  );
                  if (!inRange.length) return null;
                  const best = inRange.reduce((a, b) => (b.importance_score > a.importance_score ? b : a), inRange[0]);
                  return (
                    <div style={{
                      fontSize: 11, color: 'var(--text-muted)', marginBottom: 10,
                      padding: '6px 8px', background: 'var(--bg-elevated)',
                      borderRadius: 'var(--radius-sm)', borderLeft: '2px solid var(--accent-cyan)',
                      lineHeight: 1.4,
                    }}>
                      <span style={{ fontWeight: 600, color: 'var(--text-secondary)', fontSize: 10, textTransform: 'uppercase', letterSpacing: '0.03em' }}>Detected subject: </span>
                      {String(best.description || '').length > 120 ? String(best.description).slice(0, 120) + '...' : String(best.description || '')}
                    </div>
                  );
                })()}

                <div
                  style={{
                    fontSize: 12, color: 'var(--text-secondary)', marginBottom: 8,
                    overflowWrap: 'anywhere', wordBreak: 'break-word',
                  }}
                >
                  {clip.suggested_caption && (
                    <div style={{ marginBottom: 4 }}>
                      <strong style={{ color: 'var(--text-primary)' }}>Caption:</strong> {String(clip.suggested_caption || '')}
                    </div>
                  )}
                  {clip.hook_text && (
                    <div style={{ marginBottom: 4 }}>
                      <strong style={{ color: 'var(--text-primary)' }}>Hook:</strong> {String(clip.hook_text || '')}
                    </div>
                  )}
                  {clip.why_this_works && (
                    <div>
                      <strong style={{ color: 'var(--text-primary)' }}>Why it works:</strong> {String(clip.why_this_works || '')}
                    </div>
                  )}
                </div>

                {/* Per-clip settings toggle */}
                <button
                  onClick={() => setEditingClip(isEditing ? null : clipKey)}
                  style={{
                    width: '100%',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    padding: '6px 0',
                    background: 'none',
                    border: 'none',
                    borderTop: '1px solid var(--border)',
                    cursor: 'pointer',
                    color: isCustom ? 'var(--accent-cyan)' : 'var(--text-muted)',
                    fontSize: 11,
                    fontWeight: 500,
                  }}
                >
                  <span>
                    Clip Settings
                    {isCustom && clipSettings.subtitlesEnabled && (
                      <span style={{ marginLeft: 6, fontSize: 9, color: 'var(--accent-cyan)' }}>SUBS</span>
                    )}
                    {isCustom && clipSettings.aspectRatio && (
                      <span style={{ marginLeft: 6, fontSize: 9, color: 'var(--accent-amber)' }}>{String(clipSettings.aspectRatio || '')}</span>
                    )}
                  </span>
                  <span style={{ fontSize: 12 }}>{isEditing ? '\u25B4' : '\u25BE'}</span>
                </button>

                {/* Inline per-clip settings editor */}
                {isEditing && (
                  <div style={{
                    padding: '10px 0 6px',
                    borderTop: '1px solid var(--border)',
                  }}>
                    <SubtitleSettingsEditor
                      values={clipSettings}
                      onChange={(key, val) => {
                        const ck = getClipKey(clip);
                        setClipOverrides((prev) => ({
                          ...prev,
                          [ck]: { ...(prev[ck] || {}), [key]: val },
                        }));
                      }}
                      speakerList={speakerList}
                      customFonts={customFonts}
                      compact
                    />
                    {isCustom && (
                      <button
                        onClick={() => resetClipOverride(clip)}
                        style={{
                          marginTop: 10,
                          padding: '4px 10px',
                          fontSize: 10,
                          color: 'var(--text-muted)',
                          background: 'var(--bg-elevated)',
                          border: '1px solid var(--border)',
                          borderRadius: 'var(--radius-sm)',
                          cursor: 'pointer',
                        }}
                      >
                        Reset to defaults
                      </button>
                    )}
                  </div>
                )}

                <div style={{ display: 'flex', gap: 6, marginTop: 8, alignItems: 'stretch', flexWrap: 'wrap' }}>
                  <button
                    onClick={() => setPreviewClip(clip)}
                    style={{
                      flex: '1 1 60px',
                      padding: '8px 8px',
                      background: 'var(--accent-cyan-dim)',
                      color: 'var(--accent-cyan)',
                      border: '1px solid var(--accent-cyan)',
                      borderRadius: 'var(--radius-sm)',
                      fontSize: 12,
                      fontWeight: 600,
                      cursor: 'pointer',
                      whiteSpace: 'nowrap',
                      minWidth: 60,
                    }}
                  >
                    Preview
                  </button>
                  <Link
                    to={`/seo/${clip.jobId}/${clip.id}`}
                    style={{
                      flex: '1 1 40px',
                      padding: '8px 8px',
                      background: 'var(--bg-elevated)',
                      color: 'var(--accent-amber)',
                      border: '1px solid var(--accent-amber)',
                      borderRadius: 'var(--radius-sm)',
                      fontSize: 12,
                      fontWeight: 600,
                      cursor: 'pointer',
                      textDecoration: 'none',
                      textAlign: 'center',
                      whiteSpace: 'nowrap',
                      minWidth: 40,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                    }}
                  >
                    SEO
                  </Link>
                  {(() => {
                    const exportId = `${clip.jobId}_${clip.id}`;
                    const isExporting = encoding.tasks[exportId]?.status === 'encoding';
                    const cs = getClipSettings(clip);
                    const defaultQuality = cs.exportQuality || '1080p';
                    const isMenuOpen = qualityMenuOpen === clip.id;
                    return (
                      <div style={{ flex: '1 1 90px', position: 'relative', minWidth: 90, display: 'flex' }}>
                        <div style={{ display: 'flex', flex: 1, minWidth: 0 }}>
                          <button
                            onClick={() => handleExportClip(clip.jobId, clip)}
                            disabled={isExporting}
                            style={{
                              flex: 1,
                              padding: '8px 8px',
                              background: isExporting ? 'var(--bg-elevated)' : 'var(--accent-cyan)',
                              color: isExporting ? 'var(--accent-cyan)' : 'var(--bg-base)',
                              border: isExporting ? '1px solid var(--accent-cyan)' : 'none',
                              borderRadius: 'var(--radius-sm) 0 0 var(--radius-sm)',
                              fontSize: 12,
                              fontWeight: 600,
                              cursor: isExporting ? 'wait' : 'pointer',
                              opacity: isExporting ? 0.8 : 1,
                              whiteSpace: 'nowrap',
                              minWidth: 0,
                              overflow: 'hidden',
                              textOverflow: 'ellipsis',
                            }}
                          >
                            {isExporting ? 'Exporting...' : `Export ${defaultQuality.toUpperCase()}`}
                          </button>
                          <button
                            onClick={() => setQualityMenuOpen(isMenuOpen ? null : clip.id)}
                            disabled={isExporting}
                            style={{
                              padding: '8px 6px',
                              background: isExporting ? 'var(--bg-elevated)' : 'var(--accent-cyan)',
                              color: isExporting ? 'var(--accent-cyan)' : 'var(--bg-base)',
                              border: 'none',
                              borderLeft: '1px solid rgba(0,0,0,0.15)',
                              borderRadius: '0 var(--radius-sm) var(--radius-sm) 0',
                              fontSize: 10,
                              cursor: isExporting ? 'wait' : 'pointer',
                              opacity: isExporting ? 0.8 : 1,
                              flexShrink: 0,
                            }}
                          >
                            ▼
                          </button>
                        </div>
                        {isMenuOpen && (
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
                                onClick={() => {
                                  setQualityMenuOpen(null);
                                  handleExportClip(clip.jobId, clip, q);
                                }}
                                style={{
                                  display: 'block', width: '100%', padding: '6px 10px',
                                  background: q === defaultQuality ? 'var(--accent-cyan)' : 'transparent',
                                  color: q === defaultQuality ? 'var(--bg-base)' : 'var(--text-primary)',
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
                    );
                  })()}
                  <button
                    onClick={() => {
                      if (window.confirm(`Delete clip ${clip.id} "${clip.title}"?`)) {
                        handleDeleteClip(clip);
                      }
                    }}
                    title="Delete clip"
                    style={{
                      padding: '5px 8px',
                      background: 'transparent',
                      color: 'var(--danger, #ff3b30)',
                      border: '1px solid var(--danger, #ff3b30)',
                      borderRadius: 'var(--radius-sm)',
                      fontSize: 14,
                      lineHeight: '1.2',
                      cursor: 'pointer',
                      flexShrink: 0,
                      minWidth: 30,
                      minHeight: 28,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      opacity: 0.6,
                      transition: 'opacity 0.15s',
                      overflow: 'visible',
                    }}
                    onMouseEnter={(e) => { e.currentTarget.style.opacity = '1'; }}
                    onMouseLeave={(e) => { e.currentTarget.style.opacity = '0.6'; }}
                  >
                    &#x2715;
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      ) : totalClips === 0 ? (
        <div style={{
          textAlign: 'center',
          padding: '80px 24px',
          border: '2px dashed var(--border)',
        }}>
          <div style={{ fontSize: 48, marginBottom: 16, opacity: 0.3 }}>&#x1F3AC;</div>
          <h3 style={{ fontSize: 18, marginBottom: 8, color: 'var(--text-secondary)' }}>
            No viral clips yet
          </h3>
          <p style={{ color: 'var(--text-muted)', marginBottom: 24 }}>
            Upload a video and run analysis to detect viral clip candidates
          </p>
          <Link
            to="/upload"
            style={{
              display: 'inline-block',
              padding: '10px 24px',
              background: 'var(--accent-cyan)',
              color: 'var(--bg-base)',
              fontWeight: 600,
              borderRadius: 'var(--radius-sm)',
              textDecoration: 'none',
            }}
          >
            Upload Video
          </Link>
        </div>
      ) : (
        <div style={{ textAlign: 'center', padding: 48, color: 'var(--text-muted)' }}>
          {searchQuery.trim()
            ? `No clips match "${searchQuery.trim()}".`
            : 'No clips match current filters.'}
        </div>
      )}

      {/* Loading overlay while subject tracking computes for new AR */}
      {trackingLoading && (
        <div style={{
          position: 'fixed', inset: 0, zIndex: 10002,
          display: 'flex', flexDirection: 'column',
          alignItems: 'center', justifyContent: 'center',
          background: 'rgba(0,0,0,0.6)',
          backdropFilter: 'blur(4px)',
          pointerEvents: 'all',
        }}>
          <div style={{
            background: 'var(--bg-panel)',
            borderRadius: 'var(--radius-lg)',
            padding: '24px 36px',
            textAlign: 'center',
            boxShadow: '0 8px 32px rgba(0,0,0,0.3)',
            maxWidth: 340,
          }}>
            <div style={{
              width: 32, height: 32, margin: '0 auto 12px',
              border: '3px solid var(--border)',
              borderTopColor: 'var(--accent-cyan)',
              borderRadius: '50%',
              animation: 'spin 0.8s linear infinite',
            }} />
            <div style={{ fontSize: 14, fontWeight: 600, color: 'var(--text-primary)', marginBottom: 4 }}>
              Applying subject tracking
            </div>
            <div style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
              Analyzing subject position for {previewClipSettings.aspectRatio} crop...
            </div>
          </div>
        </div>
      )}

      {/* Clip preview lightbox */}
      {previewClip && (
        <>
          <EditorErrorBoundary
            name="Clip preview"
            onReset={() => { setPreviewClip(null); setCenterSubjectState('idle'); setTrackingApplied(false); }}
            fallbackHint="Close and re-open the preview, or try a different clip."
          >
          <ClipPreview
            key={`preview-${previewClip.jobId}-${previewClip.id}-${previewKey}`}
            src={`/api/files/${previewClip.jobId}/video.${previewClip.fileExt || 'mp4'}`}
            clipStart={previewClip.start_time}
            clipEnd={previewClip.end_time}
            title={previewClip.title || `Clip ${previewClip.id}`}
            aspectRatio={activeAspectRatio || null}
            sourceWidth={previewSourceDims.w}
            sourceHeight={previewSourceDims.h}
            subjectX={previewSubjectX}
            scenes={previewScenes}
            sceneCuts={jobs.find(j => j.job_id === previewClip?.jobId)?.scene_cut_timestamps || null}
            subjectTrack={jobs.find(j => j.job_id === previewClip?.jobId)?.subject_track || null}
            subtitlesEnabled={previewClipSettings.subtitlesEnabled || false}
            subtitleSettings={previewClipSettings}
            transcript={previewTranscript}
            initialVolume={previewClipSettings.playbackVolume}
            initialSpeed={previewClipSettings.playbackSpeed}
            layoutTimeline={jobs.find(j => j.job_id === previewClip?.jobId)?.layout_timeline || null}
            faceRegistry={jobs.find(j => j.job_id === previewClip?.jobId)?.face_registry_data || null}
            defaultLayoutMode={
              previewClipSettings.layoutMode ||
              jobs.find(j => j.job_id === previewClip?.jobId)?.default_layout_mode ||
              'single'
            }
            trackingMode={jobs.find(j => j.job_id === previewClip?.jobId)?.tracking_mode || null}
            onClose={() => { setPreviewClip(null); setCenterSubjectState('idle'); setTrackingApplied(false); }}
          />
          </EditorErrorBoundary>
          {/* Tracking-applied indicator — flashes when subject tracking updates */}
          {trackingApplied && (
            <div style={{
              position: 'fixed', top: 80, left: '50%', transform: 'translateX(-50%)',
              zIndex: 10001,
              background: 'rgba(16,185,129,0.9)',
              color: '#fff',
              padding: '8px 18px',
              borderRadius: 'var(--radius-sm)',
              fontSize: 13, fontWeight: 600,
              backdropFilter: 'blur(8px)',
              boxShadow: '0 4px 20px rgba(16,185,129,0.35)',
              animation: 'slideIn 0.3s ease forwards',
              display: 'flex', alignItems: 'center', gap: 8,
            }}>
              <span style={{ fontSize: 16 }}>&#10003;</span>
              Subject tracking applied to {previewClip?.title || `Clip ${previewClip?.id}`} &mdash; preview &amp; export updated
            </div>
          )}
          {/* Subject tracking controls — visible when crop is active */}
          {previewClipSettings.aspectRatio && (
            <div style={{
              position: 'fixed', bottom: 70, left: '50%', transform: 'translateX(-50%)',
              zIndex: 10000,
              display: 'flex', gap: 6,
            }}>
              <button
                disabled={centerSubjectState !== 'idle'}
                onClick={async () => {
                  if (centerSubjectState !== 'idle') return;
                  setCenterSubjectState('centering');
                  const finishSuccess = (msg) => {
                    setCenterSubjectState('done');
                    setTrackingApplied(true);
                    showToast(msg, 'success');
                    setTimeout(() => { setCenterSubjectState('idle'); setTrackingApplied(false); }, 2200);
                  };
                  const finishError = (msg) => {
                    setCenterSubjectState('idle');
                    showToast(msg, 'error');
                  };
                  try {
                    const res = await fetch(`/api/jobs/${previewClip.jobId}/recenter-subject`, { method: 'POST' });
                    if (!res.ok) { finishError('Failed to center subject'); return; }
                    const data = await res.json();
                    if (data.status === 'reanalyzing') {
                      showToast('Analyzing subject position with AI...', 'info');
                      const poll = setInterval(async () => {
                        try {
                          const jr = await fetch(`/api/jobs/${previewClip.jobId}`, { cache: 'no-store' });
                          if (!jr.ok) return;
                          const jd = await jr.json();
                          const sx = (jd.scenes || []).map((s) => s.subject_x);
                          if (sx.some((v) => v !== 50)) {
                            clearInterval(poll);
                            setJobs((prev) => prev.map((j) => j.job_id === previewClip.jobId ? sanitizeJob(jd) : j));
                            setPreviewKey((k) => k + 1);
                            finishSuccess('Subject centered');
                          }
                        } catch { /* ignore */ }
                      }, 3000);
                      setTimeout(() => { clearInterval(poll); setCenterSubjectState((s) => s === 'centering' ? 'idle' : s); }, 120000);
                    } else {
                      const jr = await fetch(`/api/jobs/${previewClip.jobId}`, { cache: 'no-store' });
                      if (jr.ok) {
                        const jd = await jr.json();
                        setJobs((prev) => prev.map((j) => j.job_id === previewClip.jobId ? sanitizeJob(jd) : j));
                        setPreviewKey((k) => k + 1);
                      }
                      finishSuccess(data.per_scene
                        ? 'Subject tracking applied — dynamic framing active'
                        : `Subject centered at ${data.subject_x ?? 50}%`);
                    }
                  } catch { finishError('Failed to center subject'); }
                }}
                style={{
                  padding: '6px 12px', fontSize: 11, fontWeight: 600,
                  background: centerSubjectState === 'done'
                    ? 'rgba(16,185,129,0.85)'
                    : centerSubjectState === 'centering'
                      ? 'rgba(10,132,255,0.25)'
                      : 'rgba(0,0,0,0.7)',
                  color: centerSubjectState === 'done' ? '#fff' : 'var(--accent-cyan)',
                  border: `1px solid ${centerSubjectState === 'done' ? '#10B981' : 'var(--accent-cyan)'}`,
                  borderRadius: 'var(--radius-sm)',
                  cursor: centerSubjectState !== 'idle' ? 'default' : 'pointer',
                  backdropFilter: 'blur(8px)',
                  transition: 'background 0.3s, color 0.3s, border-color 0.3s, transform 0.15s',
                  opacity: centerSubjectState === 'centering' ? 0.9 : 1,
                  transform: centerSubjectState === 'done' ? 'scale(1.05)' : 'scale(1)',
                  pointerEvents: centerSubjectState !== 'idle' ? 'none' : 'auto',
                }}
              >
                {centerSubjectState === 'centering' && (
                  <span style={{
                    display: 'inline-block', width: 12, height: 12, marginRight: 5,
                    border: '2px solid var(--accent-cyan)', borderTopColor: 'transparent',
                    borderRadius: '50%', animation: 'spin 0.8s linear infinite',
                    verticalAlign: 'middle',
                  }} />
                )}
                {centerSubjectState === 'done' && (
                  <span style={{ marginRight: 4, verticalAlign: 'middle' }}>&#10003;</span>
                )}
                {centerSubjectState === 'centering'
                  ? 'Centering...'
                  : centerSubjectState === 'done'
                    ? 'Centered'
                    : 'Center Subject'}
              </button>
              <button
                onClick={async () => {
                  try {
                    const res = await fetch(`/api/jobs/${previewClip.jobId}/reanalyze-subject`, { method: 'POST' });
                    if (res.ok) {
                      showToast('Re-analyzing subject positions with AI...', 'info');
                      const poll = setInterval(async () => {
                        try {
                          const jr = await fetch(`/api/jobs/${previewClip.jobId}`, { cache: 'no-store' });
                          if (!jr.ok) return;
                          const jd = await jr.json();
                          const sx = (jd.scenes || []).map((s) => s.subject_x);
                          const nonDefault = sx.some((v) => v !== 50);
                          if (nonDefault) {
                            clearInterval(poll);
                            setJobs((prev) => prev.map((j) => j.job_id === previewClip.jobId ? sanitizeJob(jd) : j));
                            setPreviewKey((k) => k + 1);
                            setTrackingApplied(true);
                            showToast('Subject tracking updated', 'success');
                            setTimeout(() => setTrackingApplied(false), 2200);
                          }
                        } catch { /* ignore */ }
                      }, 3000);
                      setTimeout(() => clearInterval(poll), 120000);
                    }
                  } catch { showToast('Failed to re-analyze', 'error'); }
                }}
                style={{
                  padding: '6px 12px', fontSize: 11, fontWeight: 600,
                  background: 'rgba(0,0,0,0.7)', color: 'var(--accent-amber)',
                  border: '1px solid var(--accent-amber)', borderRadius: 'var(--radius-sm)',
                  cursor: 'pointer', backdropFilter: 'blur(8px)',
                }}
              >
                Re-analyze Subject
              </button>
            </div>
          )}
          {/* Auto-applied settings indicator */}
          {settingsAppliedFlash && (
            <div style={{
              position: 'fixed', bottom: 32, left: '50%', transform: 'translateX(-50%)',
              zIndex: 10000,
              display: 'flex', alignItems: 'center', gap: 8,
              padding: '8px 18px',
              background: 'var(--success)', color: '#fff',
              borderRadius: 'var(--radius-xl)',
              fontSize: 13, fontWeight: 600,
              boxShadow: '0 4px 20px rgba(0,0,0,0.2)',
              animation: 'slideIn 0.25s ease forwards',
              pointerEvents: 'none',
            }}>
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <polyline points="20 6 9 17 4 12" />
              </svg>
              Settings applied to preview
            </div>
          )}
        </>
      )}
    </div>
  );
}
