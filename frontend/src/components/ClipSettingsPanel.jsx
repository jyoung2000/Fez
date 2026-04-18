import React, { useState, useEffect, useMemo, useRef } from 'react';
import { outlineTextShadow } from '../utils/textOutline';
import { sanitizeSubtitleSettings } from '../utils/sanitizeJob';
import useResponsive from '../hooks/useResponsive';
import {
  DEFAULT_CLIP_SETTINGS,
  SUBTITLE_RANGES,
  SPEED_OPTIONS,
  offsetVFromPosition,
  positionFromOffsetV,
} from '../utils/defaultSettings';

const STORAGE_KEY = 'clipai_clip_settings';

const DEFAULT_SETTINGS = DEFAULT_CLIP_SETTINGS;

const ASPECT_RATIOS = [
  { value: null, label: 'Original', desc: 'Source aspect ratio' },
  { value: '16:9', label: '16:9', desc: 'Landscape (YouTube)' },
  { value: '9:16', label: '9:16', desc: 'Portrait (TikTok / Reels)' },
  { value: '1:1', label: '1:1', desc: 'Square (Instagram)' },
  { value: '4:5', label: '4:5', desc: 'Portrait (Instagram Feed)' },
];

const BUILTIN_FONTS = [
  'DM Sans',
  'Montserrat',
  'Open Sans',
  'Roboto',
  'Poppins',
  'Inter',
  'Nunito',
  'Lato',
  'Oswald',
  'Playfair Display',
  'Bebas Neue',
  'Liberation Sans',
  'Liberation Serif',
  'Liberation Mono',
  'DejaVu Sans',
  'DejaVu Serif',
  'DejaVu Sans Mono',
  'FreeSans',
];

// Builtin fonts that need @font-face registration (not standard web fonts).
// Maps font family name → backend serving path for the regular weight.
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

const FONT_ACCEPT = '.ttf,.otf,.woff,.woff2,.eot,.TTF,.OTF,.WOFF,.WOFF2,.EOT';

// Backend-matching constants for font size scaling (ass_generator.py)
const FONT_SIZE_MAP = { small: 22, medium: 30, large: 40 };
const REF_W = 1920;
const REF_H = 1080;
const ASPECT_RATIO_DIMS = {
  '16:9': [1920, 1080],
  '9:16': [1080, 1920],
  '1:1': [1080, 1080],
  '4:5': [1080, 1350],
};

const SIZES = [
  { value: 22, label: 'S' },
  { value: 30, label: 'M' },
  { value: 40, label: 'L' },
];

// Available font weights per font family
const ALL_FONT_WEIGHTS = [
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
const VARIABLE_FONTS = new Set([
  'DM Sans', 'Montserrat', 'Open Sans', 'Roboto', 'Inter', 'Nunito',
  'Oswald', 'Playfair Display', 'Lato',
]);
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
  if (VARIABLE_FONTS.has(fontFamily)) return ALL_FONT_WEIGHTS;
  const weights = STATIC_FONT_WEIGHTS[fontFamily];
  if (weights) return ALL_FONT_WEIGHTS.filter(w => weights.includes(w.value));
  return ALL_FONT_WEIGHTS.filter(w => [400, 700].includes(w.value));
}
// Convert legacy string weights to numeric
function normalizeWeight(w) {
  if (typeof w === 'number') return w;
  if (w === 'bold') return 700;
  if (w === 'black') return 900;
  return 400; // 'normal' and any other string
}

const POSITIONS = [
  { value: 'top', label: 'Top' },
  { value: 'center', label: 'Center' },
  { value: 'bottom', label: 'Bottom' },
];

const DEFAULT_PALETTE = ['#00D9FF', '#F59E0B', '#10B981', '#A78BFA', '#EF4444', '#EC4899'];

// Register a custom font via @font-face so the browser can render it in the preview
function registerFontFace(fontName, url) {
  const existingId = `custom-font-${fontName.replace(/\s+/g, '-')}`;
  if (document.getElementById(existingId)) return;
  const style = document.createElement('style');
  style.id = existingId;
  style.textContent = `@font-face { font-family: '${fontName}'; src: url('${url}'); font-weight: 100 900; font-display: swap; }`;
  document.head.appendChild(style);
}

function getAspectDimensions(ratio) {
  if (!ratio) return null;
  const parts = ratio.split(':').map(Number);
  if (parts.length === 2 && parts[0] > 0 && parts[1] > 0) {
    return { w: parts[0], h: parts[1] };
  }
  return null;
}

function loadSettings() {
  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved) {
      const parsed = JSON.parse(saved);
      // Migration: old default was subtitlesEnabled=false, new default is true.
      // Only override if the stored value was the old default AND no subtitle
      // settings were customized (meaning user never touched subtitle settings).
      if (parsed.subtitlesEnabled === false && !parsed.subtitleFont) {
        parsed.subtitlesEnabled = true;
      }
      return { ...DEFAULT_SETTINGS, ...sanitizeSubtitleSettings(parsed) };
    }
  } catch {}
  return { ...DEFAULT_SETTINGS };
}

function saveSettings(settings) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
  } catch {}
}

export default function ClipSettingsPanel({ speakers, speakerNames, videoResolution, onSettingsChange, onApplySettings, onPresetsLoaded, serverSettings, parentSettings }) {
  const { isMobile } = useResponsive();
  const [settings, setSettings] = useState(() => {
    // Prefer server-provided settings over localStorage (server is source of truth)
    if (serverSettings && Object.keys(serverSettings).length > 0) {
      return { ...DEFAULT_SETTINGS, ...sanitizeSubtitleSettings(serverSettings) };
    }
    const loaded = loadSettings();
    return loaded;
  });
  const serverSettingsApplied = useRef(false);

  // When serverSettings prop arrives (async from job fetch), apply it once
  useEffect(() => {
    if (serverSettings && Object.keys(serverSettings).length > 0 && !serverSettingsApplied.current) {
      serverSettingsApplied.current = true;
      setSettings(prev => ({ ...DEFAULT_SETTINGS, ...sanitizeSubtitleSettings(serverSettings) }));
    }
  }, [serverSettings]);
  const [customFonts, setCustomFonts] = useState([]);
  const [fontUploading, setFontUploading] = useState(false);
  const fontInputRef = useRef(null);
  const [presets, setPresets] = useState([]);
  const [presetName, setPresetName] = useState('');
  const [showSaveInput, setShowSaveInput] = useState(false);
  const [activePresetName, setActivePresetName] = useState('');

  // Register @font-face for builtin fonts so the browser preview matches
  // the actual fonts installed in the Docker image (used by FFmpeg/libass)
  useEffect(() => {
    Object.entries(BUILTIN_FONT_FILES).forEach(([name, url]) => {
      registerFontFace(name, url);
    });
  }, []);

  // Load custom fonts from backend on mount.
  // ``AbortController`` ensures we don't ``setState`` after the panel
  // unmounts (which produces React warnings AND can dirty the global
  // FontFaceSet with no consumer).
  useEffect(() => {
    const ctrl = new AbortController();
    fetch('/api/fonts', { signal: ctrl.signal })
      .then((r) => (r.ok ? r.json() : []))
      .then((fonts) => {
        if (ctrl.signal.aborted) return;
        setCustomFonts(fonts);
        fonts.forEach((f) => registerFontFace(f.name, f.url));
      })
      .catch((err) => {
        if (err && err.name !== 'AbortError') { /* swallow per legacy */ }
      });
    return () => ctrl.abort();
  }, []);

  // Load saved presets from backend on mount.
  useEffect(() => {
    const ctrl = new AbortController();
    fetch('/api/clip-presets', { signal: ctrl.signal })
      .then((r) => (r.ok ? r.json() : []))
      .then((loaded) => {
        if (ctrl.signal.aborted) return;
        setPresets(loaded);
        onPresetsLoaded?.(loaded);
      })
      .catch((err) => {
        if (err && err.name !== 'AbortError') { /* swallow per legacy */ }
      });
    return () => ctrl.abort();
  }, []);

  // Notify parent when presets change (save/delete)
  useEffect(() => {
    onPresetsLoaded?.(presets);
  }, [presets, onPresetsLoaded]);

  // Initialize speaker colors from speakers list
  useEffect(() => {
    if (speakers && speakers.length > 0) {
      setSettings((prev) => {
        const colors = { ...prev.speakerColors };
        let changed = false;
        speakers.forEach((sp, i) => {
          const displayName = speakerNames?.[sp] || sp;
          if (!colors[sp]) {
            colors[sp] = DEFAULT_PALETTE[i % DEFAULT_PALETTE.length];
            changed = true;
          }
        });
        if (!changed) return prev;
        return { ...prev, speakerColors: colors };
      });
    }
  }, [speakers, speakerNames]);

  // Notify parent of settings changes and persist
  useEffect(() => {
    onSettingsChange?.(settings);
    saveSettings(settings);
  }, [settings, onSettingsChange]);

  // Eagerly preload the selected font so the browser downloads it before
  // any preview element references it (avoids stuck fallback / FOUT).
  useEffect(() => {
    const font = settings.subtitleFont;
    if (!font) return;
    // ``document.fonts`` is missing on some headless browsers and very
    // old WebViews. Without this guard the panel mount would throw
    // ``TypeError: Cannot read properties of undefined (reading 'load')``
    // and the entire ClipSettings panel would fail to render.
    if (typeof document === 'undefined' || !document.fonts || typeof document.fonts.load !== 'function') {
      return;
    }
    document.fonts.load(`400 16px "${font}"`).catch(() => {});
    document.fonts.load(`700 16px "${font}"`).catch(() => {});
  }, [settings.subtitleFont]);

  // Sync subtitlesEnabled when toggled externally (e.g. Analysis toolbar button)
  useEffect(() => {
    if (parentSettings && parentSettings.subtitlesEnabled !== undefined) {
      setSettings(prev => {
        if (prev.subtitlesEnabled !== parentSettings.subtitlesEnabled) {
          return { ...prev, subtitlesEnabled: parentSettings.subtitlesEnabled };
        }
        return prev;
      });
    }
  }, [parentSettings?.subtitlesEnabled]);

  const update = (key, value) => {
    setSettings((prev) => ({ ...prev, [key]: value }));
  };

  const handleReset = () => {
    const colors = {};
    (speakers || []).forEach((sp, i) => {
      colors[sp] = DEFAULT_PALETTE[i % DEFAULT_PALETTE.length];
    });
    setSettings({ ...DEFAULT_SETTINGS, speakerColors: colors });
    setActivePresetName('');
  };

  const handleSavePreset = async () => {
    const name = presetName.trim();
    if (!name) return;
    // Exclude speakerColors since they are video-specific
    const { speakerColors, ...settingsToSave } = settings;
    try {
      const res = await fetch('/api/clip-presets', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, settings: settingsToSave }),
      });
      if (res.ok) {
        const preset = await res.json();
        setPresets((prev) => [...prev, preset]);
        setPresetName('');
        setShowSaveInput(false);
      }
    } catch {}
  };

  const handleLoadPreset = (presetId) => {
    if (!presetId) return;
    const preset = presets.find((p) => p.id === presetId);
    if (!preset) return;
    // Merge with defaults to handle missing keys from older presets,
    // preserve current speakerColors since they are video-specific
    const merged = { ...DEFAULT_SETTINGS, ...preset.settings, speakerColors: settings.speakerColors };
    setSettings(merged);
    setActivePresetName(preset.name);
  };

  const handleDeletePreset = async (presetId) => {
    try {
      const preset = presets.find((p) => p.id === presetId);
      const res = await fetch(`/api/clip-presets/${presetId}`, { method: 'DELETE' });
      if (res.ok) {
        setPresets((prev) => prev.filter((p) => p.id !== presetId));
        // Clear active name if the deleted preset was active
        if (preset && preset.name === activePresetName) {
          setActivePresetName('');
        }
      }
    } catch {}
  };

  // Parse source dimensions
  const sourceDims = useMemo(() => {
    if (videoResolution) {
      const parts = videoResolution.split('x').map(Number);
      if (parts.length === 2 && parts[0] > 0 && parts[1] > 0) {
        return { w: parts[0], h: parts[1] };
      }
    }
    return { w: 1920, h: 1080 };
  }, [videoResolution]);

  // Preview dimensions
  const previewWidth = 220;
  const previewAspect = settings.aspectRatio
    ? getAspectDimensions(settings.aspectRatio)
    : { w: sourceDims.w, h: sourceDims.h };
  const previewHeight = previewAspect
    ? Math.round(previewWidth * (previewAspect.h / previewAspect.w))
    : Math.round(previewWidth * (sourceDims.h / sourceDims.w));

  // Two-step font scaling matching backend (ass_generator.py:204-210)
  const outputW = settings.aspectRatio && ASPECT_RATIO_DIMS[settings.aspectRatio]
    ? ASPECT_RATIO_DIMS[settings.aspectRatio][0] : sourceDims.w;
  const outputH = settings.aspectRatio && ASPECT_RATIO_DIMS[settings.aspectRatio]
    ? ASPECT_RATIO_DIMS[settings.aspectRatio][1] : sourceDims.h;
  const panelFontScale = Math.min(outputW, outputH) / Math.min(REF_W, REF_H);
  const panelBasePx = typeof settings.subtitleSize === 'number' ? settings.subtitleSize : (FONT_SIZE_MAP[settings.subtitleSize] || 30);
  const panelBackendPx = Math.max(16, Math.round(panelBasePx * panelFontScale));
  const panelPreviewScale = Math.min(previewWidth / outputW, previewHeight / outputH);
  const fontSizePx = Math.max(8, Math.round(panelBackendPx * panelPreviewScale));

  // Sample subtitle text for preview
  const sampleSpeaker = speakers?.[0] || 'Speaker 1';
  const sampleSpeaker2 = speakers?.[1] || (speakers?.length > 0 ? speakers[0] : 'Speaker 2');
  const sampleColor1 = settings.speakerColors[sampleSpeaker] || DEFAULT_PALETTE[0];
  const sampleColor2 = settings.speakerColors[sampleSpeaker2] || DEFAULT_PALETTE[1];
  const sampleName1 = speakerNames?.[sampleSpeaker] || sampleSpeaker;
  const sampleName2 = speakerNames?.[sampleSpeaker2] || sampleSpeaker2;

  const sectionStyle = {
    padding: '12px 0',
    borderBottom: '1px solid var(--border)',
  };

  const labelStyle = {
    fontSize: 13,
    fontFamily: 'var(--font-mono)',
    color: 'var(--text-primary)',
    textTransform: 'uppercase',
    letterSpacing: '0.05em',
    marginBottom: 8,
    display: 'block',
  };

  const radioGroupStyle = {
    display: 'flex',
    gap: 6,
    flexWrap: 'wrap',
  };

  const numInputStyle = {
    width: 48,
    padding: '2px 4px',
    fontSize: 12,
    fontFamily: 'var(--font-mono)',
    background: 'var(--bg-elevated)',
    color: 'var(--accent-cyan)',
    border: '1px solid var(--border)',
    borderRadius: 'var(--radius-sm)',
    outline: 'none',
    textAlign: 'center',
  };

  const radioBtnStyle = (active) => ({
    padding: '5px 12px',
    fontSize: 12,
    background: active ? 'var(--accent-cyan-dim)' : 'var(--bg-elevated)',
    color: active ? 'var(--accent-cyan)' : 'var(--text-secondary)',
    border: `1px solid ${active ? 'var(--accent-cyan)' : 'var(--border)'}`,
    borderRadius: 'var(--radius-sm)',
    cursor: 'pointer',
    fontWeight: active ? 600 : 400,
  });

  return (
    <div style={{
      background: 'var(--bg-panel)',
      border: '1px solid var(--border)',
      borderRadius: 'var(--radius-sm)',
      overflow: 'hidden',
    }}>
      {/* Header */}
      <div style={{
        padding: '10px 16px',
        borderBottom: '1px solid var(--border)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
      }}>
        <span style={{
          fontFamily: 'var(--font-mono)',
          fontWeight: 700,
          fontSize: 13,
          color: 'var(--text-primary)',
        }}>
          Clip Settings
        </span>
        {(settings.aspectRatio || settings.subtitlesEnabled) && (
          <span style={{ fontSize: 10, color: 'var(--accent-cyan)', fontWeight: 400 }}>
            {[
              settings.aspectRatio,
              settings.subtitlesEnabled ? 'subtitles' : null,
            ].filter(Boolean).join(' + ')}
          </span>
        )}
      </div>

      <div style={{ padding: '0 16px 0' }}>
        <div style={{ display: 'flex', flexDirection: 'column' }}>
          {/* Presets */}
          <div style={sectionStyle}>
            <span style={labelStyle}>Presets</span>
            <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
              <select
                value=""
                onChange={(e) => handleLoadPreset(e.target.value)}
                style={{
                  flex: 1,
                  padding: '6px 8px',
                  fontSize: 12,
                  borderRadius: 'var(--radius-sm)',
                  background: 'var(--bg-elevated)',
                  color: 'var(--text-primary)',
                  border: '1px solid var(--border)',
                }}
              >
                <option value="">Load a preset...</option>
                {presets.map((p) => (
                  <option key={p.id} value={p.id}>{String(p.name ?? '')}</option>
                ))}
              </select>
              <button
                onClick={() => setShowSaveInput((v) => !v)}
                title="Save current settings as preset"
                style={{
                  padding: '5px 10px',
                  fontSize: 11,
                  background: showSaveInput ? 'var(--accent-cyan-dim)' : 'var(--bg-elevated)',
                  color: showSaveInput ? 'var(--accent-cyan)' : 'var(--text-secondary)',
                  border: `1px solid ${showSaveInput ? 'var(--accent-cyan)' : 'var(--border)'}`,
                  borderRadius: 'var(--radius-sm)',
                  cursor: 'pointer',
                  whiteSpace: 'nowrap',
                }}
              >
                + Save
              </button>
            </div>
            {showSaveInput && (
              <div style={{ display: 'flex', gap: 6, marginTop: 8 }}>
                <input
                  type="text"
                  placeholder="Preset name..."
                  value={presetName}
                  onChange={(e) => setPresetName(e.target.value)}
                  onKeyDown={(e) => { if (e.key === 'Enter') handleSavePreset(); }}
                  autoFocus
                  style={{
                    flex: 1,
                    padding: '6px 8px',
                    fontSize: 12,
                    fontFamily: 'var(--font-mono)',
                    background: 'var(--bg-elevated)',
                    color: 'var(--accent-cyan)',
                    border: '1px solid var(--border)',
                    borderRadius: 'var(--radius-sm)',
                    outline: 'none',
                  }}
                />
                <button
                  onClick={handleSavePreset}
                  disabled={!presetName.trim()}
                  style={{
                    padding: '5px 12px',
                    fontSize: 11,
                    background: presetName.trim() ? 'var(--accent-cyan)' : 'var(--bg-elevated)',
                    color: presetName.trim() ? 'var(--bg-base)' : 'var(--text-muted)',
                    border: 'none',
                    borderRadius: 'var(--radius-sm)',
                    cursor: presetName.trim() ? 'pointer' : 'default',
                    fontWeight: 600,
                  }}
                >
                  Save
                </button>
              </div>
            )}
            {activePresetName && (
              <div style={{
                marginTop: 8,
                padding: '6px 10px',
                fontSize: 11,
                fontFamily: 'var(--font-mono)',
                background: 'var(--accent-cyan-dim)',
                border: '1px solid var(--accent-cyan)',
                borderRadius: 'var(--radius-sm)',
                color: 'var(--accent-cyan)',
                letterSpacing: '0.02em',
              }}>
                Active: {activePresetName}
              </div>
            )}
          </div>

          {/* Settings */}
          <div>
              {/* Aspect Ratio */}
              <div style={sectionStyle}>
                <span style={labelStyle}>Aspect Ratio</span>
                <div style={radioGroupStyle}>
                  {ASPECT_RATIOS.map((ar) => (
                    <button
                      key={ar.label}
                      onClick={() => update('aspectRatio', ar.value)}
                      style={radioBtnStyle(settings.aspectRatio === ar.value)}
                      title={ar.desc}
                    >
                      {ar.label}
                    </button>
                  ))}
                </div>
              </div>

              {/* Layout Mode */}
              <div style={sectionStyle}>
                <span style={labelStyle}>Layout Mode</span>
                <div style={radioGroupStyle}>
                  {[
                    { value: 'auto', label: 'Auto', desc: 'Automatically detect optimal layout' },
                    { value: 'single', label: 'Single', desc: 'One speaker centered (default)' },
                    { value: 'split', label: 'Split', desc: 'Two speakers side-by-side' },
                    { value: 'pip', label: 'PiP', desc: 'Picture-in-picture overlay' },
                  ].map((lm) => (
                    <button
                      key={lm.value}
                      onClick={() => update('layoutMode', lm.value)}
                      style={radioBtnStyle(
                        (settings.layoutMode || 'auto') === lm.value
                      )}
                      title={lm.desc}
                      disabled={
                        (lm.value === 'split' || lm.value === 'pip') &&
                        !(settings.faceRegistry?.slots?.length >= 2)
                      }
                    >
                      {lm.label}
                    </button>
                  ))}
                </div>
              </div>

              {/* Export Quality */}
              <div style={sectionStyle}>
                <span style={labelStyle}>Default Export Quality</span>
                <div style={radioGroupStyle}>
                  {[
                    { value: '720p', label: '720p' },
                    { value: '1080p', label: '1080p' },
                    { value: '4k', label: '4K' },
                  ].map((q) => (
                    <button
                      key={q.value}
                      onClick={() => update('exportQuality', q.value)}
                      style={radioBtnStyle(settings.exportQuality === q.value)}
                    >
                      {q.label}
                    </button>
                  ))}
                </div>
              </div>

              {/* Playback Volume */}
              <div style={sectionStyle}>
                <span style={labelStyle}>Playback Volume</span>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <button
                    onClick={() => update('playbackVolume', settings.playbackVolume > 0 ? 0 : 100)}
                    style={{
                      display: 'flex', alignItems: 'center', justifyContent: 'center',
                      width: 32, height: 32, border: 'none', borderRadius: 'var(--radius-sm)',
                      background: 'var(--bg-elevated)', color: 'var(--text-secondary)',
                      cursor: 'pointer',
                    }}
                    title={settings.playbackVolume > 0 ? 'Mute' : 'Unmute'}
                  >
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M11 5L6 9H2v6h4l5 4V5z" fill="currentColor" opacity="0.5" stroke="none" />
                      {settings.playbackVolume === 0 ? (
                        <>
                          <line x1="23" y1="9" x2="17" y2="15" />
                          <line x1="17" y1="9" x2="23" y2="15" />
                        </>
                      ) : settings.playbackVolume <= 100 ? (
                        <path d="M15.54 8.46a5 5 0 010 7.07" />
                      ) : (
                        <>
                          <path d="M15.54 8.46a5 5 0 010 7.07" />
                          <path d="M19.07 4.93a10 10 0 010 14.14" />
                        </>
                      )}
                    </svg>
                  </button>
                  <input
                    type="range"
                    min="0"
                    max="200"
                    step="1"
                    value={settings.playbackVolume ?? 100}
                    onChange={(e) => update('playbackVolume', parseInt(e.target.value))}
                    style={{
                      flex: 1,
                      accentColor: (settings.playbackVolume ?? 100) > 100 ? '#FFD60A' : 'var(--accent-cyan)',
                    }}
                  />
                  <span style={{
                    fontSize: 11,
                    fontFamily: 'var(--font-mono)',
                    color: (settings.playbackVolume ?? 100) > 150 ? '#FFD60A' : 'var(--text-secondary)',
                    minWidth: 36,
                    textAlign: 'right',
                  }}>
                    {settings.playbackVolume ?? 100}%
                  </span>
                </div>
              </div>

              {/* Playback Speed */}
              <div style={sectionStyle}>
                <span style={labelStyle}>Playback Speed</span>
                <div style={radioGroupStyle}>
                  {[
                    { value: 0.25, label: '0.25x' },
                    { value: 0.5, label: '0.5x' },
                    { value: 0.75, label: '0.75x' },
                    { value: 1.0, label: '1x' },
                    { value: 1.25, label: '1.25x' },
                    { value: 1.5, label: '1.5x' },
                    { value: 2.0, label: '2x' },
                    { value: 4.0, label: '4x' },
                  ].map((s) => (
                    <button
                      key={s.value}
                      onClick={() => update('playbackSpeed', s.value)}
                      style={radioBtnStyle((settings.playbackSpeed ?? 1.0) === s.value)}
                    >
                      {s.label}
                    </button>
                  ))}
                </div>
              </div>

              {/* Subtitles */}
              <div style={sectionStyle}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: settings.subtitlesEnabled ? 12 : 0 }}>
                  <span style={labelStyle}>Subtitles</span>
                  <label style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer' }}>
                    <span style={{ fontSize: 12, color: settings.subtitlesEnabled ? 'var(--accent-cyan)' : 'var(--text-secondary)' }}>
                      {settings.subtitlesEnabled ? 'ON' : 'OFF'}
                    </span>
                    <div
                      onClick={() => update('subtitlesEnabled', !settings.subtitlesEnabled)}
                      style={{
                        width: 36,
                        height: 20,
                        borderRadius: 10,
                        background: settings.subtitlesEnabled ? 'var(--accent-cyan)' : 'var(--bg-elevated)',
                        border: `1px solid ${settings.subtitlesEnabled ? 'var(--accent-cyan)' : 'var(--border)'}`,
                        position: 'relative',
                        cursor: 'pointer',
                        transition: 'background 0.2s',
                      }}
                    >
                      <div style={{
                        width: 14,
                        height: 14,
                        borderRadius: '50%',
                        background: settings.subtitlesEnabled ? 'var(--bg-base)' : 'var(--text-secondary)',
                        position: 'absolute',
                        top: 2,
                        left: settings.subtitlesEnabled ? 19 : 2,
                        transition: 'left 0.2s',
                      }} />
                    </div>
                  </label>
                </div>

                {settings.subtitlesEnabled && (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
                    {/* Font */}
                    <div>
                      <div style={{ fontSize: 13, color: 'var(--text-primary)', marginBottom: 6 }}>Font</div>
                      <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
                        <select
                          value={settings.subtitleFont}
                          onChange={(e) => {
                            const newFont = e.target.value;
                            update('subtitleFont', newFont);
                            // Snap weight to closest available if current weight isn't supported
                            const curWeight = normalizeWeight(settings.subtitleFontWeight);
                            const available = getWeightOptionsForFont(newFont).map(w => w.value);
                            if (!available.includes(curWeight)) {
                              const closest = available.reduce((a, b) => Math.abs(b - curWeight) < Math.abs(a - curWeight) ? b : a);
                              update('subtitleFontWeight', closest);
                            }
                          }}
                          style={{ flex: 1, padding: '6px 8px', borderRadius: 'var(--radius-sm)', fontSize: 12 }}
                        >
                          {BUILTIN_FONTS.map((f) => (
                            <option key={f} value={f} style={{ fontFamily: f }}>{f}</option>
                          ))}
                          {customFonts.length > 0 && (
                            <optgroup label="Custom Fonts">
                              {customFonts.map((f) => (
                                <option key={f.name} value={f.name} style={{ fontFamily: f.name }}>{String(f.name ?? '')}</option>
                              ))}
                            </optgroup>
                          )}
                        </select>
                        <input
                          ref={fontInputRef}
                          type="file"
                          accept={FONT_ACCEPT}
                          multiple
                          style={{ display: 'none' }}
                          onChange={async (e) => {
                            const files = Array.from(e.target.files || []);
                            if (!files.length) return;
                            setFontUploading(true);
                            try {
                              for (const file of files) {
                                const form = new FormData();
                                form.append('file', file);
                                const res = await fetch('/api/fonts/upload', { method: 'POST', body: form });
                                if (!res.ok) continue;
                                const font = await res.json();
                                registerFontFace(font.name, font.url);
                                setCustomFonts((prev) => {
                                  if (prev.some((f) => f.name === font.name)) return prev;
                                  return [...prev, font];
                                });
                              }
                            } catch {}
                            setFontUploading(false);
                            e.target.value = '';
                          }}
                        />
                        <button
                          onClick={() => fontInputRef.current?.click()}
                          disabled={fontUploading}
                          title="Upload custom font (.ttf, .otf, .woff, .woff2, .eot)"
                          style={{
                            padding: '5px 10px',
                            fontSize: 11,
                            background: 'var(--bg-elevated)',
                            color: 'var(--text-secondary)',
                            border: '1px solid var(--border)',
                            borderRadius: 'var(--radius-sm)',
                            cursor: fontUploading ? 'default' : 'pointer',
                            whiteSpace: 'nowrap',
                            opacity: fontUploading ? 0.6 : 1,
                          }}
                        >
                          {fontUploading ? 'Uploading...' : '+ Font'}
                        </button>
                      </div>
                    </div>

                    {/* Size */}
                    <div>
                      <div style={{ fontSize: 13, color: 'var(--text-primary)', marginBottom: 6 }}>Font Size</div>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
                        <input
                          type="range"
                          min="12"
                          max="72"
                          value={Math.min(72, Math.max(12, typeof settings.subtitleSize === 'number' ? settings.subtitleSize : (FONT_SIZE_MAP[settings.subtitleSize] || 30)))}
                          onChange={(e) => update('subtitleSize', parseInt(e.target.value))}
                          style={{ flex: 1, accentColor: 'var(--accent-cyan)' }}
                        />
                        <input
                          type="number"
                          value={typeof settings.subtitleSize === 'number' ? settings.subtitleSize : (FONT_SIZE_MAP[settings.subtitleSize] || 30)}
                          onChange={(e) => { const v = parseInt(e.target.value); if (!isNaN(v) && v > 0) update('subtitleSize', v); }}
                          style={numInputStyle}
                        />
                      </div>
                      <div style={radioGroupStyle}>
                        {SIZES.map((s) => (
                          <button
                            key={s.value}
                            onClick={() => update('subtitleSize', s.value)}
                            style={radioBtnStyle(settings.subtitleSize === s.value)}
                          >
                            {s.label}
                          </button>
                        ))}
                      </div>
                    </div>

                    {/* Font Weight */}
                    <div>
                      <div style={{ fontSize: 13, color: 'var(--text-primary)', marginBottom: 6 }}>Weight</div>
                      <div style={radioGroupStyle}>
                        {getWeightOptionsForFont(settings.subtitleFont).map((w) => (
                          <button
                            key={w.value}
                            onClick={() => update('subtitleFontWeight', w.value)}
                            style={radioBtnStyle(normalizeWeight(settings.subtitleFontWeight) === w.value)}
                          >
                            {w.label}
                          </button>
                        ))}
                      </div>
                    </div>

                    {/* Font Color */}
                    <div>
                      <div style={{ fontSize: 13, color: 'var(--text-primary)', marginBottom: 6 }}>Font Color</div>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                        <input
                          type="color"
                          value={settings.subtitleFontColor}
                          onChange={(e) => update('subtitleFontColor', e.target.value)}
                          style={{
                            width: 28,
                            height: 28,
                            border: '1px solid var(--border)',
                            borderRadius: 'var(--radius-sm)',
                            padding: 1,
                            cursor: 'pointer',
                            background: 'var(--bg-elevated)',
                          }}
                        />
                        <span style={{ fontSize: 11, fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)' }}>
                            {String(settings.subtitleFontColor || '')}
                        </span>
                      </div>
                    </div>

                    {/* Font Outline */}
                    <div>
                      <div style={{ fontSize: 13, color: 'var(--text-primary)', marginBottom: 6 }}>Font Outline</div>
                      <div style={{ display: 'flex', gap: 12, alignItems: 'center', flexWrap: 'wrap' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                          <span style={{ fontSize: 12, color: 'var(--text-secondary)' }}>Color</span>
                          <input
                            type="color"
                            value={settings.subtitleOutlineColor}
                            onChange={(e) => update('subtitleOutlineColor', e.target.value)}
                            style={{
                              width: 28,
                              height: 28,
                              border: '1px solid var(--border)',
                              borderRadius: 'var(--radius-sm)',
                              padding: 1,
                              cursor: 'pointer',
                              background: 'var(--bg-elevated)',
                            }}
                          />
                          <span style={{ fontSize: 11, fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)' }}>
                            {String(settings.subtitleOutlineColor || '')}
                          </span>
                        </div>
                      </div>
                      <div style={{ display: 'flex', flexDirection: 'column', gap: 8, marginTop: 8 }}>
                        <div>
                          <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginBottom: 4 }}>Opacity</div>
                          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                            <input
                              type="range" min="0" max="100" step="5"
                              value={Math.min(100, Math.max(0, settings.subtitleOutlineOpacity))}
                              onChange={(e) => update('subtitleOutlineOpacity', parseInt(e.target.value))}
                              style={{ flex: 1, accentColor: 'var(--accent-cyan)' }}
                            />
                            <input
                              type="number"
                              value={settings.subtitleOutlineOpacity}
                              onChange={(e) => { const v = parseInt(e.target.value); if (!isNaN(v)) update('subtitleOutlineOpacity', v); }}
                              style={numInputStyle}
                            />
                          </div>
                        </div>
                        <div>
                          <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginBottom: 4 }}>Thickness</div>
                          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                            <input
                              type="range" min="0" max="10" step="1"
                              value={Math.min(10, Math.max(0, settings.subtitleOutlineWidth))}
                              onChange={(e) => update('subtitleOutlineWidth', parseInt(e.target.value))}
                              style={{ flex: 1, accentColor: 'var(--accent-cyan)' }}
                            />
                            <input
                              type="number"
                              value={settings.subtitleOutlineWidth}
                              onChange={(e) => { const v = parseInt(e.target.value); if (!isNaN(v)) update('subtitleOutlineWidth', v); }}
                              style={numInputStyle}
                            />
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Position — pills + numeric offset bind through
                        the same source-of-truth helpers in
                        defaultSettings.js so they can never disagree. */}
                    <div>
                      <div style={{ fontSize: 13, color: 'var(--text-primary)', marginBottom: 6 }}>Position</div>
                      <div style={radioGroupStyle}>
                        {POSITIONS.map((p) => {
                          const activePos = positionFromOffsetV(settings.subtitleOffsetV);
                          return (
                            <button
                              key={p.value}
                              onClick={() => {
                                update('subtitlePosition', p.value);
                                update('subtitleOffsetV', offsetVFromPosition(p.value));
                              }}
                              style={radioBtnStyle(activePos === p.value)}
                            >
                              {p.label}
                            </button>
                          );
                        })}
                      </div>
                    </div>

                    {/* Show Speaker Labels */}
                    <div>
                      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                        <div style={{ fontSize: 13, color: 'var(--text-primary)' }}>Show Speaker Labels</div>
                        <div
                          onClick={() => update('showSpeakerLabels', !settings.showSpeakerLabels)}
                          style={{
                            width: 36,
                            height: 20,
                            borderRadius: 10,
                            background: settings.showSpeakerLabels ? 'var(--accent-cyan)' : 'var(--bg-elevated)',
                            border: `1px solid ${settings.showSpeakerLabels ? 'var(--accent-cyan)' : 'var(--border)'}`,
                            position: 'relative',
                            cursor: 'pointer',
                            transition: 'background 0.2s',
                          }}
                        >
                          <div style={{
                            width: 14,
                            height: 14,
                            borderRadius: '50%',
                            background: settings.showSpeakerLabels ? 'var(--bg-base)' : 'var(--text-secondary)',
                            position: 'absolute',
                            top: 2,
                            left: settings.showSpeakerLabels ? 19 : 2,
                            transition: 'left 0.2s',
                          }} />
                        </div>
                      </div>
                    </div>

                    {/* Max Width */}
                    <div>
                      <div style={{ fontSize: 13, color: 'var(--text-primary)', marginBottom: 6 }}>Max Width</div>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                        <input
                          type="range" min="20" max="100" step="5"
                          value={Math.min(100, Math.max(20, settings.subtitleMaxWidth))}
                          onChange={(e) => update('subtitleMaxWidth', parseInt(e.target.value))}
                          style={{ flex: 1, accentColor: 'var(--accent-cyan)' }}
                        />
                        <input
                          type="number"
                          value={settings.subtitleMaxWidth}
                          onChange={(e) => { const v = parseInt(e.target.value); if (!isNaN(v) && v > 0) update('subtitleMaxWidth', v); }}
                          style={numInputStyle}
                        />
                      </div>
                    </div>

                    {/* Vertical Offset — clamped to centralized range
                        and kept in sync with the categorical Position
                        pills via positionFromOffsetV. */}
                    <div>
                      <div style={{ fontSize: 13, color: 'var(--text-primary)', marginBottom: 6 }}>Vertical Offset</div>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                        <input
                          type="range"
                          min={SUBTITLE_RANGES.offsetV.min}
                          max={SUBTITLE_RANGES.offsetV.max}
                          step={SUBTITLE_RANGES.offsetV.step}
                          value={Math.min(SUBTITLE_RANGES.offsetV.max, Math.max(SUBTITLE_RANGES.offsetV.min, settings.subtitleOffsetV))}
                          onChange={(e) => {
                            const v = parseInt(e.target.value);
                            update('subtitleOffsetV', v);
                            update('subtitlePosition', positionFromOffsetV(v));
                          }}
                          style={{ flex: 1, accentColor: 'var(--accent-cyan)' }}
                        />
                        <input
                          type="number"
                          min={SUBTITLE_RANGES.offsetV.min}
                          max={SUBTITLE_RANGES.offsetV.max}
                          value={settings.subtitleOffsetV}
                          onChange={(e) => {
                            const v = parseInt(e.target.value);
                            if (Number.isNaN(v)) return;
                            const clamped = Math.min(SUBTITLE_RANGES.offsetV.max, Math.max(SUBTITLE_RANGES.offsetV.min, v));
                            update('subtitleOffsetV', clamped);
                            update('subtitlePosition', positionFromOffsetV(clamped));
                          }}
                          style={numInputStyle}
                        />
                      </div>
                    </div>

                    {/* Max Words Per Subtitle */}
                    <div>
                      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: settings.subtitleMaxWords > 0 ? 6 : 0 }}>
                        <div style={{ fontSize: 13, color: 'var(--text-primary)' }}>Max Words Per Subtitle</div>
                        <div
                          onClick={() => update('subtitleMaxWords', settings.subtitleMaxWords > 0 ? 0 : 4)}
                          style={{
                            width: 36,
                            height: 20,
                            borderRadius: 10,
                            background: settings.subtitleMaxWords > 0 ? 'var(--accent-cyan)' : 'var(--bg-elevated)',
                            border: `1px solid ${settings.subtitleMaxWords > 0 ? 'var(--accent-cyan)' : 'var(--border)'}`,
                            position: 'relative',
                            cursor: 'pointer',
                            transition: 'background 0.2s',
                          }}
                        >
                          <div style={{
                            width: 14,
                            height: 14,
                            borderRadius: '50%',
                            background: settings.subtitleMaxWords > 0 ? 'var(--bg-base)' : 'var(--text-secondary)',
                            position: 'absolute',
                            top: 2,
                            left: settings.subtitleMaxWords > 0 ? 19 : 2,
                            transition: 'left 0.2s',
                          }} />
                        </div>
                      </div>
                      {settings.subtitleMaxWords > 0 && (
                        <div>
                          <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginBottom: 4 }}>Words</div>
                          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                            <input
                              type="range"
                              min={Math.max(1, SUBTITLE_RANGES.maxWords.min)}
                              max={SUBTITLE_RANGES.maxWords.max}
                              step={SUBTITLE_RANGES.maxWords.step}
                              value={Math.min(SUBTITLE_RANGES.maxWords.max, Math.max(1, settings.subtitleMaxWords))}
                              onChange={(e) => update('subtitleMaxWords', parseInt(e.target.value))}
                              style={{ flex: 1, accentColor: 'var(--accent-cyan)' }}
                            />
                            <input
                              type="number"
                              min={Math.max(1, SUBTITLE_RANGES.maxWords.min)}
                              max={SUBTITLE_RANGES.maxWords.max}
                              value={settings.subtitleMaxWords}
                              onChange={(e) => {
                                const v = parseInt(e.target.value);
                                if (Number.isNaN(v)) return;
                                const clamped = Math.min(SUBTITLE_RANGES.maxWords.max, Math.max(1, v));
                                update('subtitleMaxWords', clamped);
                              }}
                              style={numInputStyle}
                            />
                          </div>
                        </div>
                      )}
                    </div>

                    {/* Active Word Highlight */}
                    <div>
                      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: settings.activeWordEnabled ? 12 : 0 }}>
                        <div style={{ fontSize: 13, color: 'var(--text-primary)' }}>Active Word Highlight</div>
                        <div
                          onClick={() => update('activeWordEnabled', !settings.activeWordEnabled)}
                          style={{
                            width: 36,
                            height: 20,
                            borderRadius: 10,
                            background: settings.activeWordEnabled ? 'var(--accent-cyan)' : 'var(--bg-elevated)',
                            border: `1px solid ${settings.activeWordEnabled ? 'var(--accent-cyan)' : 'var(--border)'}`,
                            position: 'relative',
                            cursor: 'pointer',
                            transition: 'background 0.2s',
                          }}
                        >
                          <div style={{
                            width: 14,
                            height: 14,
                            borderRadius: '50%',
                            background: settings.activeWordEnabled ? 'var(--bg-base)' : 'var(--text-secondary)',
                            position: 'absolute',
                            top: 2,
                            left: settings.activeWordEnabled ? 19 : 2,
                            transition: 'left 0.2s',
                          }} />
                        </div>
                      </div>
                      {settings.activeWordEnabled && (
                        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                          {/* Word Color */}
                          <div>
                            <div style={{ fontSize: 13, color: 'var(--text-primary)', marginBottom: 6 }}>Word Color</div>
                            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                              <input
                                type="color"
                                value={settings.activeWordColor}
                                onChange={(e) => update('activeWordColor', e.target.value)}
                                style={{
                                  width: 28, height: 28,
                                  border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)',
                                  padding: 1, cursor: 'pointer', background: 'var(--bg-elevated)',
                                }}
                              />
                              <span style={{ fontSize: 11, fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)' }}>
                                {String(settings.activeWordColor || '')}
                              </span>
                            </div>
                          </div>
                          {/* Word Stroke Color */}
                          <div>
                            <div style={{ fontSize: 13, color: 'var(--text-primary)', marginBottom: 6 }}>Word Stroke Color</div>
                            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                              <input
                                type="color"
                                value={settings.activeWordOutlineColor}
                                onChange={(e) => update('activeWordOutlineColor', e.target.value)}
                                style={{
                                  width: 28, height: 28,
                                  border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)',
                                  padding: 1, cursor: 'pointer', background: 'var(--bg-elevated)',
                                }}
                              />
                              <span style={{ fontSize: 11, fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)' }}>
                                {String(settings.activeWordOutlineColor || '')}
                              </span>
                            </div>
                          </div>
                          {/* Word Background */}
                          <div>
                            <div style={{ fontSize: 13, color: 'var(--text-primary)', marginBottom: 6 }}>Word Background</div>
                            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                                <span style={{ fontSize: 12, color: 'var(--text-secondary)' }}>Color</span>
                                <input
                                  type="color"
                                  value={settings.activeWordBgColor}
                                  onChange={(e) => update('activeWordBgColor', e.target.value)}
                                  style={{
                                    width: 28, height: 28,
                                    border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)',
                                    padding: 1, cursor: 'pointer', background: 'var(--bg-elevated)',
                                  }}
                                />
                              </div>
                              <div>
                                <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginBottom: 4 }}>Opacity</div>
                                <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                                  <input
                                    type="range" min="0" max="100" step="5"
                                    value={Math.min(100, Math.max(0, settings.activeWordBgOpacity))}
                                    onChange={(e) => update('activeWordBgOpacity', parseInt(e.target.value))}
                                    style={{ flex: 1, accentColor: 'var(--accent-cyan)' }}
                                  />
                                  <input
                                    type="number"
                                    value={settings.activeWordBgOpacity}
                                    onChange={(e) => { const v = parseInt(e.target.value); if (!isNaN(v)) update('activeWordBgOpacity', v); }}
                                    style={numInputStyle}
                                  />
                                </div>
                              </div>
                              <div>
                                <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginBottom: 4 }}>Radius</div>
                                <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                                  <input
                                    type="range" min="0" max="20" step="1"
                                    value={Math.min(20, Math.max(0, settings.activeWordBgRadius ?? 4))}
                                    onChange={(e) => update('activeWordBgRadius', parseInt(e.target.value))}
                                    style={{ flex: 1, accentColor: 'var(--accent-cyan)' }}
                                  />
                                  <input
                                    type="number"
                                    value={settings.activeWordBgRadius ?? 4}
                                    onChange={(e) => { const v = parseInt(e.target.value); if (!isNaN(v)) update('activeWordBgRadius', v); }}
                                    style={numInputStyle}
                                  />
                                </div>
                              </div>
                            </div>
                          </div>
                        </div>
                      )}
                    </div>

                    {/* Speaker Colors — toggle between per-speaker colors and uniform font color */}
                    {speakers && speakers.length > 0 && (
                      <div>
                        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: (settings.useSpeakerColors ?? true) ? 8 : 0 }}>
                          <div style={{ fontSize: 13, color: 'var(--text-primary)' }}>Speaker Colors</div>
                          <div
                            onClick={() => update('useSpeakerColors', !(settings.useSpeakerColors ?? true))}
                            style={{
                              width: 36, height: 20, borderRadius: 10,
                              background: (settings.useSpeakerColors ?? true) ? 'var(--accent-cyan)' : 'var(--bg-elevated)',
                              border: `1px solid ${(settings.useSpeakerColors ?? true) ? 'var(--accent-cyan)' : 'var(--border)'}`,
                              position: 'relative', cursor: 'pointer', transition: 'background 0.2s',
                              flexShrink: 0,
                            }}
                          >
                            <div style={{
                              width: 16, height: 16, borderRadius: '50%',
                              background: (settings.useSpeakerColors ?? true) ? '#fff' : 'var(--text-muted)',
                              position: 'absolute', top: 1,
                              left: (settings.useSpeakerColors ?? true) ? 17 : 1,
                              transition: 'left 0.2s',
                            }} />
                          </div>
                        </div>
                        {(settings.useSpeakerColors ?? true) ? (
                          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                            {speakers.map((sp, i) => {
                              const displayName = speakerNames?.[sp] || sp;
                              const color = settings.speakerColors[sp] || DEFAULT_PALETTE[i % DEFAULT_PALETTE.length];
                              return (
                                <div key={sp} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                                  <input
                                    type="color"
                                    value={color}
                                    onChange={(e) => {
                                      update('speakerColors', {
                                        ...settings.speakerColors,
                                        [sp]: e.target.value,
                                      });
                                    }}
                                    style={{
                                      width: 28,
                                      height: 28,
                                      border: '1px solid var(--border)',
                                      borderRadius: 'var(--radius-sm)',
                                      padding: 1,
                                      cursor: 'pointer',
                                      background: 'var(--bg-elevated)',
                                    }}
                                  />
                                  <span style={{ fontSize: 12, color }}>{String(displayName || '')}</span>
                                </div>
                              );
                            })}
                          </div>
                        ) : (
                          <div style={{ fontSize: 11, color: 'var(--text-muted)', fontStyle: 'italic', marginTop: 4 }}>
                            All speakers use the font color ({settings.subtitleFontColor || '#FFFFFF'})
                          </div>
                        )}
                      </div>
                    )}

                    {/* Subtitle Background */}
                    <div>
                      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: settings.subtitleBgEnabled ? 8 : 0 }}>
                        <div style={{ fontSize: 13, color: 'var(--text-primary)' }}>Background Box</div>
                        <div
                          onClick={() => update('subtitleBgEnabled', !settings.subtitleBgEnabled)}
                          style={{
                            width: 36,
                            height: 20,
                            borderRadius: 10,
                            background: settings.subtitleBgEnabled ? 'var(--accent-cyan)' : 'var(--bg-elevated)',
                            border: `1px solid ${settings.subtitleBgEnabled ? 'var(--accent-cyan)' : 'var(--border)'}`,
                            position: 'relative',
                            cursor: 'pointer',
                            transition: 'background 0.2s',
                          }}
                        >
                          <div style={{
                            width: 14,
                            height: 14,
                            borderRadius: '50%',
                            background: settings.subtitleBgEnabled ? 'var(--bg-base)' : 'var(--text-secondary)',
                            position: 'absolute',
                            top: 2,
                            left: settings.subtitleBgEnabled ? 19 : 2,
                            transition: 'left 0.2s',
                          }} />
                        </div>
                      </div>

                      {settings.subtitleBgEnabled && (
                        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                            <span style={{ fontSize: 13, color: 'var(--text-primary)' }}>Color</span>
                            <input
                              type="color"
                              value={settings.subtitleBgColor}
                              onChange={(e) => update('subtitleBgColor', e.target.value)}
                              style={{
                                width: 28,
                                height: 28,
                                border: '1px solid var(--border)',
                                borderRadius: 'var(--radius-sm)',
                                padding: 1,
                                cursor: 'pointer',
                                background: 'var(--bg-elevated)',
                              }}
                            />
                          </div>
                          <div>
                            <div style={{ fontSize: 13, color: 'var(--text-primary)', marginBottom: 6 }}>Opacity</div>
                            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                              <input
                                type="range" min="0" max="100" step="5"
                                value={Math.min(100, Math.max(0, settings.subtitleBgOpacity))}
                                onChange={(e) => update('subtitleBgOpacity', parseInt(e.target.value))}
                                style={{ flex: 1, accentColor: 'var(--accent-cyan)' }}
                              />
                              <input
                                type="number"
                                value={settings.subtitleBgOpacity}
                                onChange={(e) => { const v = parseInt(e.target.value); if (!isNaN(v)) update('subtitleBgOpacity', v); }}
                                style={numInputStyle}
                              />
                            </div>
                          </div>
                          <div>
                            <div style={{ fontSize: 13, color: 'var(--text-primary)', marginBottom: 6 }}>
                              Radius <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>(not supported in export)</span>
                            </div>
                            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                              <input
                                type="range" min="0" max="20" step="1"
                                value={Math.min(20, Math.max(0, settings.subtitleBgRadius))}
                                onChange={(e) => update('subtitleBgRadius', parseInt(e.target.value))}
                                style={{ flex: 1, accentColor: 'var(--accent-cyan)' }}
                              />
                              <input
                                type="number"
                                value={settings.subtitleBgRadius}
                                onChange={(e) => { const v = parseInt(e.target.value); if (!isNaN(v)) update('subtitleBgRadius', v); }}
                                style={numInputStyle}
                              />
                            </div>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>

            </div>

            {/* Preview */}
            <div style={{ borderTop: '1px solid var(--border)', paddingTop: 12, marginTop: 4 }}>
              <span style={labelStyle}>Preview</span>
              <div style={{
                background: 'var(--bg-base)',
                border: '1px solid var(--border)',
                borderRadius: 'var(--radius-sm)',
                padding: 16,
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
              }}>
                {/* Aspect ratio preview */}
                <div style={{
                  width: previewWidth,
                  height: previewHeight,
                  background: 'var(--bg-elevated)',
                  border: '1px solid var(--border)',
                  borderRadius: 2,
                  position: 'relative',
                  overflow: 'hidden',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                }}>
                  {/* Simulated video content */}
                  <div style={{
                    width: '60%',
                    height: '70%',
                    background: 'var(--bg-elevated)',
                    borderRadius: 8,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                  }}>
                    <div style={{
                      width: 40,
                      height: 40,
                      borderRadius: '50%',
                      background: 'var(--accent-cyan-dim)',
                      border: '2px solid var(--glow-color)',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      fontSize: 16,
                      color: 'var(--accent-cyan)',
                    }}>
                      &#9654;
                    </div>
                  </div>

                  {/* Crop guide overlay when aspect ratio differs from source */}
                  {settings.aspectRatio && (
                    <div style={{
                      position: 'absolute',
                      top: 0,
                      left: 0,
                      right: 0,
                      bottom: 0,
                      border: '2px dashed var(--accent-amber)',
                      borderRadius: 2,
                      pointerEvents: 'none',
                    }}>
                      <div style={{
                        position: 'absolute',
                        top: 4,
                        right: 4,
                        fontSize: 9,
                        fontFamily: 'var(--font-mono)',
                        color: 'var(--accent-amber)',
                        background: 'var(--overlay-heavy)',
                        padding: '1px 4px',
                        borderRadius: 2,
                      }}>
                        {String(settings.aspectRatio || '')}
                      </div>
                    </div>
                  )}

                  {/* Subtitle preview */}
                  {settings.subtitlesEnabled && (() => {
                    const offsetPx = Math.round(previewHeight * settings.subtitleOffsetV / 100);
                    // Scale outline for the small preview (220px wide vs 1920 ref)
                    const previewScale = Math.min(previewWidth / 1920, previewHeight / 1080);
                    const olWidth = settings.subtitleOutlineWidth ?? 2;
                    const olOpacity = (settings.subtitleOutlineOpacity ?? 100) / 100;
                    const olColor = settings.subtitleOutlineColor || '#000000';
                    const olHex = olColor.replace('#', '');
                    const olR = parseInt(olHex.substring(0, 2), 16) || 0;
                    const olG = parseInt(olHex.substring(2, 4), 16) || 0;
                    const olB = parseInt(olHex.substring(4, 6), 16) || 0;
                    const scaledOlWidth = Math.max(0, Math.round(olWidth * previewScale));
                    const olColorStr = `rgba(${olR},${olG},${olB},${olOpacity})`;
                    const outlineStyle = !settings.subtitleBgEnabled && scaledOlWidth > 0
                      ? {
                          WebkitTextStroke: `${scaledOlWidth * 2}px ${olColorStr}`,
                          paintOrder: 'stroke fill',
                          textShadow: outlineTextShadow(scaledOlWidth, olColorStr, '1px 1px 2px rgba(0,0,0,0.5)'),
                        }
                      : (!settings.subtitleBgEnabled
                          ? { textShadow: '1px 1px 2px rgba(0,0,0,0.7)' }
                          : {});
                    return (
                    <div style={{
                      position: 'absolute',
                      left: 12,
                      right: 12,
                      ...(settings.subtitlePosition === 'top' ? { top: Math.max(4, offsetPx) } : {}),
                      ...(settings.subtitlePosition === 'center' ? { top: '50%', transform: 'translateY(-50%)' } : {}),
                      ...(settings.subtitlePosition === 'bottom' ? { bottom: Math.max(4, offsetPx) } : {}),
                      textAlign: 'center',
                      pointerEvents: 'none',
                    }}>
                      <div style={{
                        display: 'inline-block',
                        background: settings.subtitleBgEnabled
                          ? `${settings.subtitleBgColor}${Math.round(settings.subtitleBgOpacity / 100 * 255).toString(16).padStart(2, '0')}`
                          : 'transparent',
                        borderRadius: settings.subtitleBgEnabled ? `${settings.subtitleBgRadius || 0}px` : undefined,
                        padding: settings.subtitleBgEnabled ? '4px 10px' : '2px 6px',
                        maxWidth: `${settings.subtitleMaxWidth}%`,
                      }}>
                        <div style={{
                          fontFamily: `"${settings.subtitleFont}", sans-serif`,
                          fontSize: fontSizePx,
                          fontWeight: normalizeWeight(settings.subtitleFontWeight),
                          color: (settings.useSpeakerColors ?? true) ? sampleColor1 : (settings.subtitleFontColor || '#FFFFFF'),
                          lineHeight: 1.4,
                          wordWrap: 'break-word',
                          overflowWrap: 'break-word',
                          whiteSpace: 'pre-wrap',
                          ...outlineStyle,
                        }}>
                          {settings.showSpeakerLabels ? `${sampleName1}: ` : ''}{(() => {
                            const sampleText = settings.subtitleMaxWords > 0 ? 'Welcome to the show, everyone'.split(' ').slice(0, settings.subtitleMaxWords).join(' ') : 'Welcome to the show, everyone';
                            if (settings.activeWordEnabled) {
                              return sampleText.split(' ').map((w, i) => (
                                <span key={i} style={i === 2 ? { color: settings.activeWordColor, ...(settings.activeWordBgOpacity > 0 ? { backgroundColor: `${settings.activeWordBgColor}${Math.round(settings.activeWordBgOpacity / 100 * 255).toString(16).padStart(2, '0')}`, borderRadius: 2, padding: '0 2px' } : {}) } : {}}>
                                  {w}{i < sampleText.split(' ').length - 1 ? ' ' : ''}
                                </span>
                              ));
                            }
                            return sampleText;
                          })()}
                        </div>
                        {speakers && speakers.length > 1 && (
                          <div style={{
                            fontFamily: `"${settings.subtitleFont}", sans-serif`,
                            fontSize: fontSizePx,
                            fontWeight: normalizeWeight(settings.subtitleFontWeight),
                            color: (settings.useSpeakerColors ?? true) ? sampleColor2 : (settings.subtitleFontColor || '#FFFFFF'),
                            lineHeight: 1.4,
                            marginTop: 2,
                            wordWrap: 'break-word',
                            overflowWrap: 'break-word',
                            whiteSpace: 'pre-wrap',
                            ...outlineStyle,
                          }}>
                            {settings.showSpeakerLabels ? `${sampleName2}: ` : ''}{settings.subtitleMaxWords > 0 ? 'Thanks for having me!'.split(' ').slice(0, settings.subtitleMaxWords).join(' ') : 'Thanks for having me!'}
                          </div>
                        )}
                      </div>
                    </div>
                    );
                  })()}
                </div>

                {/* Info labels below preview */}
                <div style={{
                  marginTop: 10,
                  display: 'flex',
                  gap: 12,
                  fontSize: 10,
                  fontFamily: 'var(--font-mono)',
                  color: 'var(--text-secondary)',
                }}>
                  <span>
                    {settings.aspectRatio || `${sourceDims.w}:${sourceDims.h}`}
                  </span>
                  {settings.subtitlesEnabled && (
                    <span style={{ color: 'var(--accent-cyan)' }}>
                      Subtitles ON
                    </span>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>

      {/* Actions footer */}
      <div style={{
        display: 'flex',
        gap: 8,
        padding: '12px 16px',
        borderTop: '1px solid var(--border)',
        background: 'var(--bg-panel)',
      }}>
        {onApplySettings && (
          <button
            onClick={() => onApplySettings(settings)}
            style={{
              flex: 1,
              padding: '8px 12px',
              background: 'var(--accent-cyan)',
              color: 'var(--bg-base)',
              border: 'none',
              borderRadius: 'var(--radius-sm)',
              fontSize: 12,
              fontWeight: 700,
              cursor: 'pointer',
            }}
          >
            Apply Settings
          </button>
        )}
        <button
          onClick={handleReset}
          style={{
            padding: '8px 12px',
            background: 'var(--bg-elevated)',
            color: 'var(--text-secondary)',
            border: '1px solid var(--border)',
            borderRadius: 'var(--radius-sm)',
            fontSize: 11,
          }}
        >
          Reset
        </button>
      </div>
    </div>
  );
}
