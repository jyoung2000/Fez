import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { ShareButton } from '../components/ShareDialog';
import VideoPlayer from '../components/VideoPlayer';
import ClipPreview from '../components/ClipPreview';
import VideoEditor from '../components/VideoEditor';
import SubtitleOverlay from '../components/SubtitleOverlay';
import ProgressBar from '../components/ProgressBar';
import SceneCard from '../components/SceneCard';
import TranscriptViewer from '../components/TranscriptViewer';
import ClipCard from '../components/ClipCard';
import sanitizeJob, { sanitizeSubtitleSettings } from '../utils/sanitizeJob';
import { sendNotification, requestNotificationPermission } from '../utils/notifications';
import ClipSettingsPanel from '../components/ClipSettingsPanel';
import EditorErrorBoundary from '../components/EditorErrorBoundary';
import { showToast } from '../components/Toast';
import useResponsive from '../hooks/useResponsive';
import useEncodingManager from '../hooks/useEncodingManager';
import { computeClipSubjectX } from '../utils/subjectTracking';
import useTimelineStore from '../stores/timelineStore';
import { buildOverlayPayload, buildVideoEffectsPayload, mapSubtitleSettings } from '../utils/buildExportPayload';
import { DEFAULT_CLIP_SETTINGS, DEFAULT_GEN_SETTINGS } from '../utils/defaultSettings';

// Speaker color palette (must match SubtitleOverlay / ClipSettingsPanel / VideoEditor)
const DEFAULT_SPEAKER_PALETTE = [
  '#00D9FF', '#F59E0B', '#10B981', '#A78BFA', '#EF4444', '#EC4899',
  '#06B6D4', '#8B5CF6', '#F97316', '#14B8A6', '#E879F9', '#84CC16',
  '#FB7185', '#38BDF8', '#FBBF24', '#34D399', '#C084FC', '#F472B6',
  '#22D3EE', '#A3E635', '#FB923C', '#2DD4BF', '#818CF8', '#F87171',
];

function formatDuration(seconds) {
  if (!seconds) return '-';
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, '0')}`;
}

function formatDurationInput(seconds) {
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
    if (!isNaN(m) && !isNaN(s) && m >= 0 && s >= 0 && s < 60) {
      return m * 60 + s;
    }
    return null;
  }
  const num = parseInt(trimmed, 10);
  if (!isNaN(num) && num >= 0) return num;
  return null;
}

const GEN_STORAGE_KEY = 'clipai_generation_settings';

function loadGenSettings() {
  try {
    const saved = JSON.parse(localStorage.getItem(GEN_STORAGE_KEY));
    return { ...DEFAULT_GEN_SETTINGS, ...saved };
  } catch { return { ...DEFAULT_GEN_SETTINGS }; }
}

// Example chips shown below the focus textarea when it's empty.
// Clicking one fills the textarea (see ClipFocus panel below).
const FOCUS_EXAMPLE_CHIPS = [
  'funny moments',
  'emotional reveals',
  'arguments',
  'advice and insights',
  'biggest mistakes',
  'hot takes',
];

function pushFocusHistory(current, query) {
  const q = (query || '').trim();
  if (!q) return current || [];
  const existing = (current || []).filter((h) => h !== q);
  return [q, ...existing].slice(0, 10);
}

const TABS = ['Summary', 'Key Scenes', 'Transcript', 'Viral Clips'];

// sanitizeJob imported from ../utils/sanitizeJob

/**
 * Error boundary that isolates VideoEditor crashes so they don't take down
 * the entire Analysis page.  Shows a retry button on failure.
 */
class VideoEditorBoundary extends React.Component {
  constructor(props) { super(props); this.state = { hasError: false, error: null }; }
  static getDerivedStateFromError(error) { return { hasError: true, error }; }
  componentDidCatch(error, info) {
    console.error('[VideoEditorBoundary]', error, info?.componentStack?.slice(0, 500));
  }
  render() {
    if (this.state.hasError) {
      return (
        <div style={{ padding: 24, textAlign: 'center', color: 'var(--danger, #ef4444)' }}>
          <div style={{ fontSize: 14, fontWeight: 600, marginBottom: 8 }}>Video preview failed to render</div>
          <div style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 12 }}>
            {String(this.state.error?.message || 'Unknown error')}
          </div>
          <button
            onClick={() => this.setState({ hasError: false, error: null })}
            style={{ padding: '6px 16px', fontSize: 12, background: 'var(--accent-cyan)', color: '#000', border: 'none', borderRadius: 4, cursor: 'pointer' }}
          >
            Retry
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}

function AddSceneForm({ jobId, duration, onAdded }) {
  const [open, setOpen] = React.useState(false);
  const [timestamp, setTimestamp] = React.useState('');
  const [description, setDescription] = React.useState('');
  const [score, setScore] = React.useState(7);
  const [saving, setSaving] = React.useState(false);

  const handleAdd = async () => {
    const ts = parseDuration(timestamp);
    if (ts === null || ts < 0) return;
    if (duration && ts > duration) return;
    if (!description.trim()) return;
    setSaving(true);
    try {
      const res = await fetch(`/api/jobs/${jobId}/scenes`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          timestamp: ts,
          description: description.trim(),
          importance_score: score,
        }),
      });
      if (res.ok) {
        setTimestamp('');
        setDescription('');
        setScore(7);
        setOpen(false);
        if (onAdded) onAdded();
      }
    } catch {
    } finally {
      setSaving(false);
    }
  };

  if (!open) {
    return (
      <div style={{ marginBottom: 16 }}>
        <button
          onClick={() => setOpen(true)}
          style={{
            padding: '8px 18px',
            fontSize: 12,
            fontWeight: 600,
            background: 'var(--accent-cyan-dim)',
            color: 'var(--accent-cyan)',
            border: '1px solid var(--accent-cyan)',
            borderRadius: 'var(--radius-sm)',
            cursor: 'pointer',
          }}
        >
          + Add Key Scene
        </button>
      </div>
    );
  }

  return (
    <div style={{
      marginBottom: 16,
      padding: 16,
      background: 'var(--bg-panel)',
      border: '1px solid var(--accent-cyan)',
      borderRadius: 'var(--radius-md)',
    }}>
      <div style={{ fontSize: 12, fontFamily: 'var(--font-mono)', color: 'var(--accent-cyan)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: 12 }}>
        Add Key Scene
      </div>
      <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 10 }}>
        <div style={{ flex: '0 0 auto' }}>
          <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 4 }}>Timestamp (M:SS)</div>
          <input
            type="text"
            value={timestamp}
            onChange={(e) => setTimestamp(e.target.value)}
            placeholder="1:30"
            style={{
              width: 80,
              padding: '6px 8px',
              fontSize: 12,
              fontFamily: 'var(--font-mono)',
              background: 'var(--bg-elevated)',
              color: 'var(--text-primary)',
              border: '1px solid var(--border)',
              borderRadius: 'var(--radius-sm)',
              outline: 'none',
            }}
          />
        </div>
        <div style={{ flex: '0 0 auto' }}>
          <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 4 }}>
            Importance: <span style={{ color: 'var(--accent-cyan)', fontFamily: 'var(--font-mono)' }}>{score}/10</span>
          </div>
          <input
            type="range"
            min="1"
            max="10"
            value={score}
            onChange={(e) => setScore(parseInt(e.target.value))}
            style={{ width: 120, accentColor: 'var(--accent-cyan)' }}
          />
        </div>
      </div>
      <div style={{ marginBottom: 10 }}>
        <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 4 }}>Description</div>
        <textarea
          value={description}
          onChange={(e) => setDescription(e.target.value)}
          placeholder="Describe what happens at this moment..."
          rows={2}
          style={{
            width: '100%',
            fontSize: 12,
            lineHeight: 1.4,
            color: 'var(--text-primary)',
            background: 'var(--bg-elevated)',
            border: '1px solid var(--border)',
            borderRadius: 'var(--radius-sm)',
            padding: '6px 8px',
            resize: 'vertical',
            outline: 'none',
            fontFamily: 'inherit',
          }}
        />
      </div>
      <div style={{ display: 'flex', gap: 6 }}>
        <button
          onClick={handleAdd}
          disabled={saving || !description.trim() || parseDuration(timestamp) === null}
          style={{
            padding: '6px 16px',
            fontSize: 12,
            fontWeight: 600,
            background: 'var(--accent-cyan)',
            color: 'var(--bg-base)',
            border: 'none',
            borderRadius: 'var(--radius-sm)',
            cursor: saving ? 'default' : 'pointer',
            opacity: saving || !description.trim() || parseDuration(timestamp) === null ? 0.5 : 1,
          }}
        >
          {saving ? 'Adding...' : 'Add Scene'}
        </button>
        <button
          onClick={() => setOpen(false)}
          style={{
            padding: '6px 16px',
            fontSize: 12,
            background: 'none',
            color: 'var(--text-muted)',
            border: '1px solid var(--border)',
            borderRadius: 'var(--radius-sm)',
            cursor: 'pointer',
          }}
        >
          Cancel
        </button>
      </div>
    </div>
  );
}

export default function Analysis() {
  const { jobId } = useParams();
  const navigate = useNavigate();
  const [job, setJob] = useState(null);
  const [tab, setTab] = useState(0);
  const prevTabRef = useRef(0);
  const [transcriptInitTime, setTranscriptInitTime] = useState(null);
  const stickyPlayerRef = useRef(null);
  const [loading, setLoading] = useState(true);
  const [clipPreview, setClipPreview] = useState(null);
  const [videoCurrentTime, setVideoCurrentTime] = useState(0);
  const [cancellingJob, setCancellingJob] = useState(false);
  const [selectedClips, setSelectedClips] = useState(new Set());
  const [filters, setFilters] = useState({ minScore: 0, platform: 'all', type: 'all', sort: 'viral_score' });
  const wsRef = useRef(null);
  const [activityLog, setActivityLog] = useState([]);
  const [logExpanded, setLogExpanded] = useState(true);
  const logEndRef = useRef(null);
  // Subtitle/clip settings — server is the source of truth.
  // On mount we start with defaults; once the job loads, server-stored
  // settings replace them (see the effect below).
  const CLIP_SETTINGS_DEFAULTS = React.useMemo(() => ({ ...DEFAULT_CLIP_SETTINGS }), []);
  const [clipSettings, setClipSettings] = useState(CLIP_SETTINGS_DEFAULTS);
  const clipSettingsLoadedFromServer = useRef(false);
  const skipNextServerSave = useRef(false);

  // ── Multi-track editor timeline items (for export) ──
  const timelineItems = useTimelineStore((s) => s.items);
  const timelineTracks = useTimelineStore((s) => s.tracks);
  const timelineMediaLibrary = useTimelineStore((s) => s.mediaLibrary);

  // ── Server is source of truth for subtitle settings ──
  // On initial load this is handled in fetchJob (same batch as setJob) to avoid
  // a flash. This effect only handles subsequent changes (e.g., refreshes).
  const initialSettingsApplied = useRef(false);
  useEffect(() => {
    if (!job) return;
    if (!initialSettingsApplied.current) {
      initialSettingsApplied.current = true;
      return;
    }
    if (job.subtitle_settings && Object.keys(job.subtitle_settings).length > 0) {
      // Server has canonical settings — use them, merged over defaults
      skipNextServerSave.current = true; // Don't echo back to server
      setClipSettings({ ...CLIP_SETTINGS_DEFAULTS, ...sanitizeSubtitleSettings(job.subtitle_settings) });
      clipSettingsLoadedFromServer.current = true;
    } else if (!clipSettingsLoadedFromServer.current) {
      // No server settings yet — fall back to localStorage for first-time migration
      try {
        const saved = localStorage.getItem('clipai_clip_settings');
        if (saved) {
          const parsed = JSON.parse(saved);
          setClipSettings({ ...CLIP_SETTINGS_DEFAULTS, ...sanitizeSubtitleSettings(parsed) });
        }
      } catch {}
    }
  }, [job?.subtitle_settings, CLIP_SETTINGS_DEFAULTS]);

  // ── Persist settings to server whenever they change (debounced) ──
  // This replaces the old localStorage-only persistence.  The server is the
  // source of truth; localStorage is only kept as a temporary migration path.
  const saveSettingsTimerRef = useRef(null);
  useEffect(() => {
    // Also keep localStorage in sync as a fallback
    try { localStorage.setItem('clipai_clip_settings', JSON.stringify(clipSettings)); } catch {}
    // Skip the echo-back when settings were just loaded from server
    if (skipNextServerSave.current) {
      skipNextServerSave.current = false;
      return;
    }
    // Debounced save to server
    if (!jobId) return;
    if (saveSettingsTimerRef.current) clearTimeout(saveSettingsTimerRef.current);
    saveSettingsTimerRef.current = setTimeout(() => {
      fetch(`/api/jobs/${jobId}/subtitle-settings`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(clipSettings),
      }).catch(() => {});
    }, 800);
    return () => { if (saveSettingsTimerRef.current) clearTimeout(saveSettingsTimerRef.current); };
  }, [clipSettings, jobId]);

  // VideoEditor state for export params
  const [editorTrim, setEditorTrim] = useState({ trimStart: 0, trimEnd: 0 });
  const [editorSubjectKeyframes, setEditorSubjectKeyframes] = useState(null);
  const [editorVolume, setEditorVolume] = useState(1.0);
  const [editorSpeed, setEditorSpeed] = useState(1.0);
  const [editorSegments, setEditorSegments] = useState([]);
  // Persist segments per clip ID so switching clips doesn't lose segments
  const clipSegmentsMapRef = useRef({});

  // ── localStorage helpers for segment persistence across refresh ──
  const segStorageKey = useCallback((cId) => `clipai_segments_${jobId}_${cId}`, [jobId]);
  const saveSegmentsToStorage = useCallback((cId, segs) => {
    try {
      if (segs && segs.length > 0) {
        localStorage.setItem(segStorageKey(cId), JSON.stringify(segs));
      } else {
        localStorage.removeItem(segStorageKey(cId));
      }
    } catch {}
  }, [segStorageKey]);
  const loadSegmentsFromStorage = useCallback((cId) => {
    try {
      const raw = localStorage.getItem(segStorageKey(cId));
      return raw ? JSON.parse(raw) : [];
    } catch { return []; }
  }, [segStorageKey]);
  // Restore segments from localStorage on mount (full video uses key 'full')
  useEffect(() => {
    if (!jobId) return;
    const saved = loadSegmentsFromStorage('full');
    if (saved.length > 0) setEditorSegments(saved);
  }, [jobId, loadSegmentsFromStorage]);

  // ── Editor state persistence to server (shared with ClipSEO page) ──
  const editorStateSaveTimerRef = useRef(null);
  useEffect(() => () => { if (editorStateSaveTimerRef.current) clearTimeout(editorStateSaveTimerRef.current); }, []);

  const saveEditorStateToServer = useCallback((cId, state) => {
    if (!jobId || !cId) return;
    fetch(`/api/jobs/${jobId}/clips/${cId}/editor-state`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(state),
    }).catch(() => {});
  }, [jobId]);

  const loadEditorStateFromServer = useCallback(async (cId) => {
    if (!jobId || !cId) return null;
    try {
      const res = await fetch(`/api/jobs/${jobId}/clips/${cId}/editor-state`);
      if (res.ok) {
        const data = await res.json();
        return data?.state || null;
      }
    } catch {}
    return null;
  }, [jobId]);

  // Debounced save of clip editor state to server when editing a clip preview
  useEffect(() => {
    if (!clipPreview?.id || !jobId) return;
    if (editorStateSaveTimerRef.current) clearTimeout(editorStateSaveTimerRef.current);
    editorStateSaveTimerRef.current = setTimeout(() => {
      saveEditorStateToServer(clipPreview.id, {
        trim: editorTrim,
        volume: editorVolume,
        speed: editorSpeed,
        segments: editorSegments,
      });
    }, 1500);
  }, [editorTrim, editorVolume, editorSpeed, editorSegments, clipPreview?.id, jobId, saveEditorStateToServer]);

  // Track applied trim ranges so the trimmed region becomes the full video
  const [fullVideoRange, setFullVideoRange] = useState(null); // { start, end } for full video editor
  const [showInlineSubSettings, setShowInlineSubSettings] = useState(false);

  // ── Mark Key Scene inline state ──
  const [showMarkScene, setShowMarkScene] = useState(false);
  const [markSceneDesc, setMarkSceneDesc] = useState('');
  const [markSceneScore, setMarkSceneScore] = useState(8);
  const [markSceneSaving, setMarkSceneSaving] = useState(false);
  const [inlineCustomFonts, setInlineCustomFonts] = useState([]);

  // Fetch custom fonts for the inline subtitle settings panel and register
  // @font-face so they render in the dropdown and subtitle preview.
  useEffect(() => {
    // Register builtin fonts via @font-face (same mapping as ClipSettingsPanel)
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
    Object.entries(BUILTIN_FONT_FILES).forEach(([name, url]) => {
      const id = `custom-font-${name.replace(/\s+/g, '-')}`;
      if (!document.getElementById(id)) {
        const style = document.createElement('style');
        style.id = id;
        style.textContent = `@font-face { font-family: '${name}'; src: url('${url}'); font-weight: 100 900; font-display: swap; }`;
        document.head.appendChild(style);
      }
    });
    // Fetch user-uploaded custom fonts
    fetch('/api/fonts')
      .then((r) => r.ok ? r.json() : [])
      .then((fonts) => {
        setInlineCustomFonts(fonts);
        // Register @font-face for each custom font
        fonts.forEach((f) => {
          const id = `custom-font-${f.name.replace(/\s+/g, '-')}`;
          if (!document.getElementById(id)) {
            const style = document.createElement('style');
            style.id = id;
            style.textContent = `@font-face { font-family: '${f.name}'; src: url('${f.url}'); font-weight: 100 900; font-display: swap; }`;
            document.head.appendChild(style);
          }
        });
      })
      .catch(() => {});
  }, []);
  const [isGeneratingClips, setIsGeneratingClips] = useState(false);
  const [stuckSeconds, setStuckSeconds] = useState(0);
  const lastProgressRef = useRef({ message: '', time: Date.now() });
  const [genSettings, setGenSettings] = useState(loadGenSettings);
  const [clipFocusEnabled, setClipFocusEnabled] = useState(() => !!loadGenSettings().clipFocusEnabled);
  const [clipFocusText, setClipFocusText] = useState(() => loadGenSettings().clipFocusText || '');
  // Sync focus state to genSettings persistence (Bug 13) so a page
  // refresh mid-search doesn't lose the user's query.
  React.useEffect(() => {
    setGenSettings((prev) => {
      if (prev.clipFocusEnabled === clipFocusEnabled && prev.clipFocusText === clipFocusText) return prev;
      return { ...prev, clipFocusEnabled, clipFocusText };
    });
  }, [clipFocusEnabled, clipFocusText]);
  // Recent focus query history + dropdown state
  const [focusHistoryOpen, setFocusHistoryOpen] = useState(false);
  // "Why these clips?" diagnostics drawer
  const [diagOpen, setDiagOpen] = useState(false);
  const [diagData, setDiagData] = useState(null);
  const [diagLoading, setDiagLoading] = useState(false);
  const [minText, setMinText] = useState(() => formatDurationInput(loadGenSettings().minDuration));
  const [maxText, setMaxText] = useState(() => formatDurationInput(loadGenSettings().maxDuration));
  const { isMobile } = useResponsive();
  const encoding = useEncodingManager();
  const [clipSearchQuery, setClipSearchQuery] = useState('');
  const [settingsAppliedFlash, setSettingsAppliedFlash] = useState(false);
  const settingsAppliedTimerRef = useRef(null);
  const prevClipSettingsRef = useRef(clipSettings);
  const [clipPresets, setClipPresets] = useState([]);
  const [selectedPresetId, setSelectedPresetId] = useState('');
  const [qaResult, setQaResult] = useState(null);
  const [qaLoading, setQaLoading] = useState(false);
  const [inlinePresetSaveOpen, setInlinePresetSaveOpen] = useState(false);
  const [inlinePresetName, setInlinePresetName] = useState('');
  const [inlineActivePreset, setInlineActivePreset] = useState('');
  const fullVideoExporting = encoding.tasks[`${jobId}_0`]?.status === 'encoding';
  const [exportQualityMenuOpen, setExportQualityMenuOpen] = useState(false);
  const exportQualityRef = React.useRef(null);
  React.useEffect(() => {
    if (!exportQualityMenuOpen) return;
    const handler = (e) => {
      if (exportQualityRef.current && !exportQualityRef.current.contains(e.target)) {
        setExportQualityMenuOpen(false);
      }
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, [exportQualityMenuOpen]);

  // Staged aspect ratio: gate preview on tracking readiness
  const [trackingLoading, setTrackingLoading] = useState(false);
  const [activeAspectRatio, setActiveAspectRatio] = useState(
    () => clipSettings?.aspectRatio || null
  );

  // Speaker detection (post-processing diarization)
  const [diarizeNumSpeakers, setDiarizeNumSpeakers] = useState(0);
  const [diarizeLoading, setDiarizeLoading] = useState(false);
  const [diarizeResult, setDiarizeResult] = useState('');
  const [retranscribing, setRetranscribing] = useState(false);

  // Fetch presets on mount so the preset bar is available from the start
  useEffect(() => {
    fetch('/api/clip-presets')
      .then(r => r.ok ? r.json() : [])
      .then(data => { if (Array.isArray(data)) setClipPresets(data); })
      .catch(() => {});
  }, []);

  const handleRunDiarization = useCallback(async () => {
    setDiarizeLoading(true);
    setDiarizeResult('');
    try {
      const res = await fetch(`/api/jobs/${jobId}/diarize`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ num_speakers: diarizeNumSpeakers }),
      });
      if (res.ok) {
        const data = await res.json();
        setDiarizeResult(`Detected ${data.speakers_detected} speaker${data.speakers_detected !== 1 ? 's' : ''}`);
        fetchJob();
      } else {
        const err = await res.json().catch(() => ({}));
        showToast(err.detail || 'Diarization failed', 'error');
      }
    } catch (e) {
      showToast('Diarization request failed', 'error');
    } finally {
      setDiarizeLoading(false);
    }
  }, [jobId, diarizeNumSpeakers]);

  const handleRetranscribe = useCallback(async () => {
    if (retranscribing) return;
    const ok = window.confirm(
      'Re-run Whisper on the full audio using the language and subtitle ' +
      'settings you chose at upload?\n\n' +
      'This replaces the current transcript and takes a few minutes. ' +
      'Scene analysis, clips, and exported videos are not affected.'
    );
    if (!ok) return;

    setRetranscribing(true);
    try {
      const res = await fetch(`/api/jobs/${jobId}/retranscribe`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        showToast(data?.detail || 'Retranscribe failed', 'error');
        return;
      }
      // Backend now returns 202 immediately and streams progress via the
      // existing job-status polling + WebSocket. Don't keep the button
      // disabled forever — the polling/WS updates will flip it back to
      // disabled while ``status === 'transcribing'``, and re-enable it
      // automatically once Whisper finishes.
      showToast(
        data?.message ||
          'Retranscribe started — the transcript will refresh when Whisper finishes.',
        'success',
      );
      await fetchJob();
    } catch (e) {
      showToast('Retranscribe request failed', 'error');
    } finally {
      setRetranscribing(false);
    }
  }, [jobId, retranscribing]);

  const fetchJobRetryRef = useRef(0);
  const fetchJob = useCallback(async () => {
    try {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 15000);
      const res = await fetch(`/api/jobs/${jobId}`, { signal: controller.signal });
      clearTimeout(timeout);
      if (res.ok) {
        const data = sanitizeJob(await res.json());
        // Log scene count on every fetch for tracking debug
        if (data.scenes?.length) {
          const denseCount = data.scenes.filter(s => s.description === '[dense face tracking]').length;
          console.log(`[Analysis] fetchJob: ${data.scenes.length} scenes (${denseCount} dense face tracking, ${data.scenes.length - denseCount} AI), status=${data.status}`);
        }
        setJob(data);
        fetchJobRetryRef.current = 0;
        // Sync generating state from job status (handles page refresh mid-generation)
        if (data.status === 'detecting_clips') {
          setIsGeneratingClips((prev) => prev || true);
        }
        // ── Apply server subtitle settings in the same batch as setJob ──
        // Prevents flash where component renders with default settings
        // (subtitles off, no speaker colors) before the settings effect fires.
        if (!clipSettingsLoadedFromServer.current && data.subtitle_settings && Object.keys(data.subtitle_settings).length > 0) {
          skipNextServerSave.current = true;
          setClipSettings(prev => ({ ...prev, ...sanitizeSubtitleSettings(data.subtitle_settings) }));
          clipSettingsLoadedFromServer.current = true;
        }
      } else if (res.status === 404 && fetchJobRetryRef.current < 10) {
        // Job may still be initializing (pipeline writes job.json async).
        // Retry a few times before showing "not found" to avoid flash during
        // large file uploads where there's a delay between upload complete
        // and job.json being written.
        fetchJobRetryRef.current += 1;
        setTimeout(() => fetchJob(), 2000);
        return; // Don't clear loading yet
      }
    } catch (err) {
      // Only log — if setJob was never called, job stays null and the "not found" guard handles it.
      console.warn('[Analysis] fetchJob error:', err?.message || err);
    } finally {
      setLoading(false);
    }
  }, [jobId]);

  const handleMarkKeyScene = useCallback(async () => {
    if (!markSceneDesc.trim() || markSceneSaving) return;
    setMarkSceneSaving(true);
    try {
      const res = await fetch(`/api/jobs/${jobId}/scenes`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          timestamp: videoCurrentTime,
          description: markSceneDesc.trim(),
          importance_score: markSceneScore,
        }),
      });
      if (res.ok) {
        showToast(`Key scene marked at ${formatDuration(videoCurrentTime)}`, 'success');
        setMarkSceneDesc('');
        setMarkSceneScore(8);
        setShowMarkScene(false);
        fetchJob();
      } else {
        showToast('Failed to mark key scene', 'error');
      }
    } catch {
      showToast('Failed to mark key scene', 'error');
    } finally {
      setMarkSceneSaving(false);
    }
  }, [jobId, videoCurrentTime, markSceneDesc, markSceneScore, markSceneSaving, fetchJob]);

  const pushLog = useCallback((type, message, extra) => {
    const ts = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    // Ensure message is always a string to prevent React error #310
    const safeMsg = typeof message === 'string' ? message : String(message ?? '');
    // Spread extra FIRST so ts/type/message always win (prevents overwrite).
    // Also sanitize extra values — any objects would cause #310 if rendered.
    const safeExtra = {};
    if (extra && typeof extra === 'object') {
      for (const [k, v] of Object.entries(extra)) {
        if (k === 'ts' || k === 'type' || k === 'message') continue; // never overwrite core fields
        safeExtra[k] = (v != null && typeof v === 'object') ? JSON.stringify(v) : v;
      }
    }
    setActivityLog((prev) => [...prev, { ...safeExtra, ts, type, message: safeMsg }]);
  }, []);

  // Auto-scroll log to bottom (within its own scroll container, not the page)
  useEffect(() => {
    const el = logEndRef.current?.parentElement;
    if (el) el.scrollTop = el.scrollHeight;
  }, [activityLog.length]);

  // Persist generation settings
  useEffect(() => {
    try { localStorage.setItem(GEN_STORAGE_KEY, JSON.stringify(genSettings)); } catch {}
  }, [genSettings]);

  // Initial fetch + WebSocket with auto-reconnection
  useEffect(() => {
    fetchJob();
    let reconnectDelay = 1000;
    let unmounted = false;

    function connectWs() {
      if (unmounted) return;
      const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const ws = new WebSocket(`${proto}//${window.location.host}/ws/jobs/${jobId}`);
      wsRef.current = ws;

      ws.onopen = () => {
        reconnectDelay = 1000; // reset backoff on successful connect
        pushLog('info', 'Connected to live updates');
        requestNotificationPermission();
      };

      ws.onmessage = (evt) => {
        try {
          const msg = JSON.parse(evt.data);
          // Coerce message to string — backend may send objects in some edge cases
          if (msg.message != null && typeof msg.message !== 'string') {
            msg.message = typeof msg.message === 'object' ? JSON.stringify(msg.message) : String(msg.message);
          }
          // Also coerce status
          if (msg.status != null && typeof msg.status !== 'string') {
            msg.status = String(msg.status);
          }
          if (msg.type === 'status' || msg.type === 'complete') {
            // Ignore export-related status messages — encoding progress is
            // handled by useEncodingManager.  Updating job.status to
            // 'exporting' would cause isProcessing to toggle and the
            // analysis progress bar to blink in and out.
            const isExportStatus = msg.status === 'exporting' || msg.status === 'generating_seo';
            if (!isExportStatus) {
              setJob((prev) => {
                if (!prev) return prev;
                // Coerce all values to safe primitives — WS messages are not sanitized
                const nextStatus = typeof msg.status === 'string' ? msg.status : String(msg.status || prev.status);
                const nextProgress = typeof msg.progress === 'number' ? msg.progress : (prev.progress ?? 0);
                const nextMessage = typeof msg.message === 'string'
                  ? msg.message
                  : (msg.message != null && typeof msg.message === 'object')
                    ? JSON.stringify(msg.message)
                    : String(msg.message ?? prev.progress_message ?? '');
                // Skip update if nothing actually changed — prevents cascading re-renders
                // during rapid WS messages (large file processing can send many per second)
                if (prev.status === nextStatus && prev.progress === nextProgress && prev.progress_message === nextMessage) {
                  return prev;
                }
                return { ...prev, status: nextStatus, progress: nextProgress, progress_message: nextMessage };
              });
            }
            // Track clip generation state from status messages — use functional
            // updater to avoid re-render when value hasn't changed
            if (msg.status === 'detecting_clips') {
              setIsGeneratingClips((prev) => prev || true);
            } else if (msg.type === 'complete') {
              setIsGeneratingClips((prev) => prev ? false : prev);
            }
            pushLog(
              msg.type === 'complete' ? 'success' : 'status',
              msg.message || `Status: ${msg.status}`,
              { progress: msg.progress },
            );
            if (msg.type === 'complete') {
              sendNotification('Analysis Complete', {
                body: msg.message || 'Your video analysis has finished.',
                tag: `analysis-${jobId}`,
              });
              fetchJob();
            }
          } else if (msg.type === 'fallback') {
            pushLog('warning', `Provider fallback: ${String(msg.from_provider || '?')} → ${String(msg.to_provider || '?')} (${String(msg.reason || 'unknown')})`);
            showToast(`Fallback: ${String(msg.from_provider || '?')} -> ${String(msg.to_provider || '?')}: ${String(msg.reason || '')}`, 'warning');
          } else if (msg.type === 'export_complete') {
            const label = msg.clip_id === 0 ? 'Full video' : `Clip ${msg.clip_id}`;
            pushLog('success', `${label} exported`);
            showToast(`${label} exported!`, 'success');
            fetchJob();
            // Download is handled by useEncodingManager (global) — do NOT
            // trigger a second download here to avoid duplicate file saves.
          } else if (msg.type === 'clips_generated') {
            pushLog('success', String(msg.message || `Generated ${msg.count} clips`));
            // Bug 7 — when the filter chain wipes everything out, show a
            // targeted toast with remediation instead of a generic
            // "0 clips" success message. The backend now sends a
            // filter_state block so we know exactly what to suggest.
            const fs = msg.filter_state || {};
            if (msg.count === 0 && (fs.focus_enabled || fs.viral_score_min > 0 || fs.viral_score_max < 100 || fs.min_relevance > 0)) {
              const parts = [];
              if (fs.focus_enabled && fs.focus_query) parts.push(`matching "${fs.focus_query}"`);
              const rangeActive = fs.viral_score_min > 0 || fs.viral_score_max < 100;
              if (rangeActive) parts.push(`within viral score ${fs.viral_score_min}-${fs.viral_score_max}`);
              if (fs.min_relevance > 0) parts.push(`with relevance >= ${fs.min_relevance}`);
              const suffix = parts.length ? ` ${parts.join(' ')}` : '';
              showToast(
                `Found 0 clips${suffix}. Try lowering thresholds, broadening your query, or clicking "Relax filters".`,
                'warning',
              );
            } else {
              showToast(String(msg.message || `Found ${msg.count} clip candidates`), 'success');
            }
            setIsGeneratingClips((prev) => prev ? false : prev);
            fetchJob();
            // Re-run QA validation after new clips are generated
            setQaResult(null);
            setTimeout(() => runQaValidation(), 1500);
          } else if (msg.type === 'cancelled') {
            pushLog('warning', 'Job cancelled by user');
            showToast('Job cancelled', 'info');
            fetchJob();
          } else if (msg.type === 'subject_tracking') {
            pushLog(
              msg.enabled ? 'info' : 'warning',
              String(msg.message || ''),
              { tracked_scenes: msg.tracked_scenes, total_scenes: msg.total_scenes },
            );
          } else if (msg.type === 'error') {
            pushLog('error', String(msg.message || 'Unknown error'));
            showToast(String(msg.message || 'Unknown error'), 'error');
            setIsGeneratingClips((prev) => prev ? false : prev);
            fetchJob();
          } else if (msg.type === 'background_task') {
            // Background post-processing (transcript polishing, translation)
            const taskName = String(msg.task || 'background');
            const taskStatus = String(msg.status || 'running');
            const taskMsg = String(msg.message || `${taskName}: ${taskStatus}`);
            pushLog(
              taskStatus === 'complete' ? 'success' : taskStatus === 'failed' ? 'warning' : 'info',
              taskMsg,
            );
            // Refresh job data when background task completes (e.g., polished transcript)
            if (taskStatus === 'complete') {
              fetchJob();
            }
          } else if (msg.type === 'heartbeat') {
            // Pipeline heartbeat — shows the pipeline is still alive during
            // long-running stages.  Log it and reset the stuck timer.
            pushLog('info', String(msg.message || 'Still processing...'));
          } else {
            // Unknown message type — log but don't crash.  Coerce all fields.
            const safeType = String(msg.type || 'unknown');
            const safeMsg = String(msg.message || msg.status || `Received: ${safeType}`);
            pushLog('info', safeMsg);
          }
        } catch {
        }
      };

      ws.onclose = () => {
        if (unmounted) return;
        pushLog('info', `Connection lost — reconnecting in ${reconnectDelay / 1000}s...`);
        fetchJob();
        setTimeout(() => {
          if (!unmounted) connectWs();
        }, reconnectDelay);
        reconnectDelay = Math.min(reconnectDelay * 2, 16000); // exponential backoff, cap at 16s
      };
    }

    connectWs();

    return () => {
      unmounted = true;
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, [jobId, fetchJob, pushLog]);

  // Periodic refresh for active jobs
  useEffect(() => {
    if (!job || ['complete', 'failed', 'cancelled'].includes(job.status)) return;
    const interval = setInterval(fetchJob, 5000);
    return () => clearInterval(interval);
  }, [job?.status, fetchJob]);

  // Stuck detection: track when progress_message last changed
  useEffect(() => {
    if (!job || ['complete', 'failed', 'cancelled'].includes(job.status)) {
      setStuckSeconds((prev) => prev === 0 ? prev : 0);
      return;
    }
    const currentMsg = job.progress_message || job.status;
    if (currentMsg !== lastProgressRef.current.message) {
      lastProgressRef.current = { message: currentMsg, time: Date.now() };
      setStuckSeconds((prev) => prev === 0 ? prev : 0);
    }
    const interval = setInterval(() => {
      const elapsed = Math.floor((Date.now() - lastProgressRef.current.time) / 1000);
      setStuckSeconds((prev) => prev === elapsed ? prev : elapsed);
    }, 5000);
    return () => clearInterval(interval);
  }, [job?.progress_message, job?.status]);

  const handleSeek = (time) => {
    if (window.__clipai_seekTo) window.__clipai_seekTo(time);
  };

  // Sync players when switching to/from the Transcript tab (tab 2).
  // Entering tab 2: pause the sticky player and capture its time so the
  // transcript tab's inline player can pick up where it left off.
  // Leaving tab 2: seek the sticky player to where the transcript player was.
  useEffect(() => {
    const prev = prevTabRef.current;
    prevTabRef.current = tab;

    if (tab === 2 && prev !== 2) {
      // Entering transcript tab — pause sticky player and capture time
      const stickyVideo = stickyPlayerRef.current?.querySelector('video');
      const stickyTime = stickyVideo ? stickyVideo.currentTime : videoCurrentTime;
      if (stickyVideo && !stickyVideo.paused) {
        stickyVideo.pause();
      }
      setTranscriptInitTime(stickyTime);
    } else if (prev === 2 && tab !== 2) {
      // Leaving transcript tab — sync sticky player to transcript's current time
      const stickyVideo = stickyPlayerRef.current?.querySelector('video');
      if (stickyVideo) {
        stickyVideo.currentTime = videoCurrentTime;
      }
      setTranscriptInitTime(null);
    }
  }, [tab]);

  const handleClipPreview = (clip) => {
    // Save current clip's editor state before switching
    if (clipPreview) {
      clipSegmentsMapRef.current[clipPreview.id] = editorSegments;
      saveSegmentsToStorage(clipPreview.id, editorSegments);
      saveEditorStateToServer(clipPreview.id, {
        trim: editorTrim,
        volume: editorVolume,
        speed: editorSpeed,
        segments: editorSegments,
      });
    }
    // Restore segments for the new clip: in-memory cache first, then localStorage
    const savedSegments = clipSegmentsMapRef.current[clip.id] || loadSegmentsFromStorage(clip.id);
    clipSegmentsMapRef.current[clip.id] = savedSegments;
    setEditorSegments(savedSegments);
    // Reset editor state defaults — server state loaded async below
    setEditorTrim({ trimStart: 0, trimEnd: 0 });
    setEditorVolume(1.0);
    setEditorSpeed(1.0);
    // Load persisted editor state from server (async)
    loadEditorStateFromServer(clip.id).then(state => {
      if (state) {
        if (state.trim) setEditorTrim(state.trim);
        if (state.volume != null) setEditorVolume(state.volume);
        if (state.speed != null) setEditorSpeed(state.speed);
        if (state.segments?.length > 0 && !clipSegmentsMapRef.current[clip.id]?.length) {
          setEditorSegments(state.segments);
          clipSegmentsMapRef.current[clip.id] = state.segments;
        }
      }
    });

    // ── Eagerly populate timeline store with clip-filtered subtitles ──
    // The timeline store is a global singleton. Before React re-renders,
    // it still has subtitle items from the previous editor (full-video or
    // another clip). SubtitleOverlay reads from this store during render.
    // By calling initFromClip synchronously here, the store has the correct
    // clip-contextual items before the first render of the new VideoEditor.
    const fullTranscript = job.translated_transcript?.length
      ? job.translated_transcript
      : (job.transcript || []);
    useTimelineStore.getState().initFromClip({
      src: `/api/files/${jobId}/video.${job.file_path?.split('.').pop() || 'mp4'}`,
      clipStart: clip.start_time,
      clipEnd: clip.end_time,
      subtitleSegments: fullTranscript,
      hookText: clip.hook_text || clip.title || '',
    });

    setClipPreview(clip);
    // NOTE: handleSeek removed — VideoEditor auto-seeks to clipStart on remount
    // via key={`clip-${clipPreview.id}`} triggering fresh mount with auto-seek effect.
    setTab(3);
  };

  const handleExportClip = (clip, settingsOverrideOrQuality) => {
    // Accept either a settings override object or a quality string
    let cs, qualityOverride;
    if (typeof settingsOverrideOrQuality === 'string') {
      cs = clipSettings;
      qualityOverride = settingsOverrideOrQuality;
    } else {
      cs = settingsOverrideOrQuality || clipSettings;
      qualityOverride = null;
    }
    const quality = qualityOverride || (cs && cs.exportQuality) || '1080p';
    const exportBody = {
      start: clip.start_time,
      end: clip.end_time,
      clip_id: clip.id,
      clip_title: clip.title || `Clip ${clip.id}`,
      export_quality: quality,
    };
    if (cs) {
      if (cs.aspectRatio) {
        exportBody.aspect_ratio = cs.aspectRatio;
      }
      // Enable subtitles if global is on OR any segment has subtitles enabled
      const globalSubsOn = cs.subtitlesEnabled || false;
      const anySegmentSubsOn = editorSegments.some(s => s.subtitlesEnabled !== false);
      const needsSubtitles = globalSubsOn || anySegmentSubsOn;
      exportBody.subtitles_enabled = needsSubtitles;
      exportBody.global_subtitles_enabled = globalSubsOn;
      if (needsSubtitles) {
        exportBody.subtitle_settings = mapSubtitleSettings(cs);
      }
    }
    // Include VideoEditor trim/volume/speed/segments params
    if (editorTrim.trimStart > 0) exportBody.trim_start_offset = editorTrim.trimStart;
    if (editorTrim.trimEnd > 0) exportBody.trim_end_offset = editorTrim.trimEnd;
    if (editorVolume !== 1.0) exportBody.volume = editorVolume;
    if (editorSpeed !== 1.0) exportBody.speed = editorSpeed;
    if (editorSegments.length > 0) {
      exportBody.segments = editorSegments.map(s => ({
        start: s.start, end: s.end,
        volume: (s.muted ? 0 : s.volume) / 100, // Convert to 0-2.0 gain
        muted: s.muted,
        subtitles_enabled: s.subtitlesEnabled,
        subject_tracking_enabled: s.subjectTrackingEnabled !== false,
        speed: s.speed || 1.0,
      }));
    }
    // Include multi-track editor video effects + transform so export matches preview
    const videoEffects = buildVideoEffectsPayload(timelineItems);
    if (videoEffects) exportBody.video_effects = videoEffects;

    // Build overlay arrays via shared utility (consistent filtering + validation)
    const overlays = buildOverlayPayload({
      timelineItems,
      mediaLibrary: timelineMediaLibrary,
      clipStart: clip.start_time,
      tracks: useTimelineStore.getState().tracks,
    });
    if (overlays.textOverlays.length > 0) exportBody.text_overlays = overlays.textOverlays;
    if (overlays.imageOverlays.length > 0) exportBody.image_overlays = overlays.imageOverlays;
    if (overlays.shapeOverlays.length > 0) exportBody.shape_overlays = overlays.shapeOverlays;
    if (overlays.audioOverlays.length > 0) exportBody.audio_overlays = overlays.audioOverlays;
    if (overlays.compositingOrder?.length > 0) {
      exportBody.overlay_compositing_order = overlays.compositingOrder;
    }
    if (overlays.warnings.length > 0) {
      for (const w of overlays.warnings) console.warn(`[Export] ${w}`);
    }

    // Include user-edited subtitle timing/text from the timeline store so the
    // export uses actual item durations/text (may have been resized or edited).
    // Mirrors ExportDialog.jsx logic.
    const subtitleItemsForExport = timelineItems
      .filter(it => it.type === 'subtitle')
      .sort((a, b) => a.start - b.start)
      .map(it => ({
        start: (it.start || 0) + clip.start_time,
        end: (it.end || 0) + clip.start_time,
        text: it.subtitleText || '',
        speaker: it.speaker || '',
        words: it.words ? it.words.map(w => ({
          start: (w.start || 0) + clip.start_time,
          end: (w.end || 0) + clip.start_time,
          word: w.text || w.word || '',
        })) : null,
      }));
    if (subtitleItemsForExport.length > 0) {
      exportBody.edited_subtitle_segments = subtitleItemsForExport;
    }

    // Diagnostic logging: full export payload for debugging overlay/settings issues
    console.log('[Analysis Export] clip:', clip.id, 'payload:', JSON.stringify({
      aspect_ratio: exportBody.aspect_ratio,
      subtitles_enabled: exportBody.subtitles_enabled,
      subtitle_settings: exportBody.subtitle_settings ? 'YES' : 'NO',
      video_effects: exportBody.video_effects ? 'YES' : 'NO',
      volume: exportBody.volume,
      speed: exportBody.speed,
      trim: [exportBody.trim_start_offset || 0, exportBody.trim_end_offset || 0],
      segments: exportBody.segments?.length || 0,
      text_overlays: exportBody.text_overlays?.length || 0,
      image_overlays: exportBody.image_overlays?.length || 0,
      shape_overlays: exportBody.shape_overlays?.length || 0,
      audio_overlays: exportBody.audio_overlays?.length || 0,
      timelineItems_total: timelineItems.length,
      timelineItems_types: [...new Set(timelineItems.map(it => it.type))],
    }));

    // Include subject tracking keyframes for export parity.
    // Prefer crop segments from timeline (may have user edits) over raw keyframes.
    if (exportBody.aspect_ratio) {
      const { cropSegments } = useTimelineStore.getState();
      if (cropSegments?.length > 0) {
        exportBody.subject_keyframes = cropSegments.map(seg => ({
          time: +seg.startTime.toFixed(3),
          x: seg.cropX,
        }));
      } else if (editorSubjectKeyframes?.length > 0) {
        exportBody.subject_keyframes = editorSubjectKeyframes.map(kf => ({
          time: +kf.t.toFixed(3),
          x: kf.x,
        }));
      }
    }

    encoding.startExport(jobId, clip.id, clip.title || `Clip ${clip.id}`, exportBody);
    showToast(`Exporting "${clip.title || `Clip ${clip.id}`}" at ${quality}...`, 'info');
  };

  const handleExportFullVideo = () => {
    if (!job?.duration) return;
    const cs = clipSettings;
    const quality = cs?.exportQuality || '1080p';
    const body = { export_quality: quality };
    if (cs?.aspectRatio) body.aspect_ratio = cs.aspectRatio;
    const fvGlobalSubsOn = cs?.subtitlesEnabled || false;
    const fvAnySegmentSubsOn = editorSegments.some(s => s.subtitlesEnabled !== false);
    const fvNeedsSubtitles = fvGlobalSubsOn || fvAnySegmentSubsOn;
    body.subtitles_enabled = fvNeedsSubtitles;
    body.global_subtitles_enabled = fvGlobalSubsOn;
    if (fvNeedsSubtitles) {
      body.subtitle_settings = mapSubtitleSettings(cs);
    }
    // Include trim/volume/speed from VideoEditor
    if (fullVideoRange) {
      body.trim_start_offset = fullVideoRange.start;
      body.trim_end_offset = (job.duration || 0) - fullVideoRange.end;
    }
    if (editorTrim.trimStart > 0) body.trim_start_offset = (body.trim_start_offset || 0) + editorTrim.trimStart;
    if (editorTrim.trimEnd > 0) body.trim_end_offset = (body.trim_end_offset || 0) + editorTrim.trimEnd;
    if (editorVolume !== 1.0) body.volume = editorVolume;
    if (editorSpeed !== 1.0) body.speed = editorSpeed;
    if (editorSegments.length > 0) {
      body.segments = editorSegments.map(s => ({
        start: s.start, end: s.end,
        volume: (s.muted ? 0 : s.volume) / 100,
        muted: s.muted,
        subtitles_enabled: s.subtitlesEnabled,
        subject_tracking_enabled: s.subjectTrackingEnabled !== false,
        speed: s.speed || 1.0,
      }));
    }
    // Video effects from multi-track editor
    const fvVideoEffects = buildVideoEffectsPayload(timelineItems);
    if (fvVideoEffects) body.video_effects = fvVideoEffects;

    // Build overlay arrays via shared utility
    // For full-video export, clipStart is 0 (timeline items are already video-relative)
    const fvOverlays = buildOverlayPayload({
      timelineItems,
      mediaLibrary: timelineMediaLibrary,
      clipStart: 0,
      tracks: useTimelineStore.getState().tracks,
    });
    if (fvOverlays.textOverlays.length > 0) body.text_overlays = fvOverlays.textOverlays;
    if (fvOverlays.imageOverlays.length > 0) body.image_overlays = fvOverlays.imageOverlays;
    if (fvOverlays.shapeOverlays.length > 0) body.shape_overlays = fvOverlays.shapeOverlays;
    if (fvOverlays.audioOverlays.length > 0) body.audio_overlays = fvOverlays.audioOverlays;
    if (fvOverlays.compositingOrder?.length > 0) {
      body.overlay_compositing_order = fvOverlays.compositingOrder;
    }
    if (fvOverlays.warnings.length > 0) {
      for (const w of fvOverlays.warnings) console.warn(`[Export] ${w}`);
    }

    // Include edited subtitle segments for full video export
    const fvSubtitleItems = timelineItems
      .filter(it => it.type === 'subtitle')
      .sort((a, b) => a.start - b.start)
      .map(it => ({
        start: (it.start || 0) + (fullVideoRange ? fullVideoRange.start : 0),
        end: (it.end || 0) + (fullVideoRange ? fullVideoRange.start : 0),
        text: it.subtitleText || '',
        speaker: it.speaker || '',
        words: it.words ? it.words.map(w => ({
          start: (w.start || 0) + (fullVideoRange ? fullVideoRange.start : 0),
          end: (w.end || 0) + (fullVideoRange ? fullVideoRange.start : 0),
          word: w.text || w.word || '',
        })) : null,
      }));
    if (fvSubtitleItems.length > 0) {
      body.edited_subtitle_segments = fvSubtitleItems;
    }

    // Include subject tracking keyframes — prefer crop segments (may have user edits)
    if (body.aspect_ratio) {
      const { cropSegments } = useTimelineStore.getState();
      if (cropSegments?.length > 0) {
        body.subject_keyframes = cropSegments.map(seg => ({
          time: +seg.startTime.toFixed(3),
          x: seg.cropX,
        }));
      } else if (editorSubjectKeyframes?.length > 0) {
        body.subject_keyframes = editorSubjectKeyframes.map(kf => ({
          time: +kf.t.toFixed(3),
          x: kf.x,
        }));
      }
    }

    encoding.startExport(jobId, 0, job.filename || 'Full Video', body, {
      endpoint: `/api/jobs/${jobId}/export-full-video`,
    });
    showToast(`Exporting full video at ${quality}... This may take a while for long videos.`, 'info');
  };

  const handleSelectClip = (clipId, checked) => {
    setSelectedClips((prev) => {
      const next = new Set(prev);
      if (checked) next.add(clipId);
      else next.delete(clipId);
      return next;
    });
  };

  const handleDeleteClip = async (clip) => {
    try {
      const res = await fetch(`/api/jobs/${jobId}/clips/${clip.id}`, { method: 'DELETE' });
      if (res.ok) {
        setJob((prev) => ({
          ...prev,
          clips: prev.clips.filter((c) => c.id !== clip.id),
          exported_clips: (prev.exported_clips || []).filter((ec) => ec.clip_id !== clip.id),
        }));
        selectedClips.delete(clip.id);
        setSelectedClips(new Set(selectedClips));
        showToast(`Clip ${clip.id} deleted`, 'info');
      } else {
        const err = await res.json().catch(() => ({}));
        showToast(err.detail || 'Failed to delete clip', 'error');
      }
    } catch {
      showToast('Failed to delete clip', 'error');
    }
  };

  const runQaValidation = useCallback(async () => {
    if (!jobId) return;
    setQaLoading(true);
    try {
      const res = await fetch(`/api/jobs/${jobId}/qa-validate`);
      if (res.ok) {
        const data = await res.json();
        setQaResult(data);
      }
    } catch {
    } finally {
      setQaLoading(false);
    }
  }, [jobId]);

  // Auto-run QA when analysis completes
  useEffect(() => {
    if (job?.status === 'complete' && job?.clips?.length > 0 && !qaResult) {
      runQaValidation();
    }
  }, [job?.status, job?.clips?.length, runQaValidation, qaResult]);

  const handleGenerateClips = async (opts = {}) => {
    setIsGeneratingClips(true);
    try {
      const body = {
        min_duration: genSettings.minDuration,
        max_duration: genSettings.maxDuration,
        clip_count: genSettings.clipCount || null,
        append: opts.append !== undefined ? opts.append : true,
      };
      // Bug 9 — allow focus mode AND viral score range to BOTH apply
      const focusActive = clipFocusEnabled && clipFocusText.trim();
      if (focusActive) {
        body.clip_focus = clipFocusText.trim();
        if (genSettings.minRelevance > 0) body.min_relevance = genSettings.minRelevance;
        // Push the query into history (Enhancement 3)
        setGenSettings((prev) => ({
          ...prev,
          focusHistory: pushFocusHistory(prev.focusHistory, clipFocusText.trim()),
        }));
      }
      if (genSettings.viralScoreMin > 0) body.viral_score_min = genSettings.viralScoreMin;
      if (genSettings.viralScoreMax < 100) body.viral_score_max = genSettings.viralScoreMax;
      if (opts.scope_start != null && opts.scope_end != null) {
        body.scope_start = opts.scope_start;
        body.scope_end = opts.scope_end;
      }
      if (opts.clip_focus) {
        body.clip_focus = opts.clip_focus;
      }
      const res = await fetch(`/api/jobs/${jobId}/generate-clips`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (res.ok) {
        if (body.clip_focus) {
          showToast(`AI is searching for "${body.clip_focus}" in your video...`, 'info');
        } else {
          showToast('AI is analyzing your video for viral moments...', 'info');
        }
      } else {
        const data = await res.json().catch(() => ({}));
        showToast(data.detail || 'Failed to generate clips', 'error');
        setIsGeneratingClips(false);
      }
    } catch (err) {
      showToast(`Failed to generate clips: ${err.message}`, 'error');
      setIsGeneratingClips(false);
    }
  };

  // Cancel button — POSTs to the cancel endpoint added in phase 2.
  // The cooperative cancel_event on the server turns the running task
  // into a no-op within a few seconds.
  const handleCancelGenerate = async () => {
    try {
      const res = await fetch(`/api/jobs/${jobId}/cancel-generate`, { method: 'POST' });
      if (res.ok) {
        showToast('Cancelling clip generation...', 'info');
        // Optimistically clear local spinner after a short delay so the
        // Cancel button goes away even if the cancelled event is slow.
        setTimeout(() => setIsGeneratingClips(false), 1500);
      }
    } catch { /* ignore */ }
  };

  // "Why these clips?" diagnostics drawer loader
  const loadDiagnostics = useCallback(async () => {
    setDiagLoading(true);
    try {
      const res = await fetch(`/api/jobs/${jobId}/clip-diagnostics`);
      if (res.ok) {
        setDiagData(await res.json());
      }
    } catch { /* ignore */ } finally {
      setDiagLoading(false);
    }
  }, [jobId]);

  // Expand Search — re-runs generate with relaxed filters
  const handleRelaxFilters = useCallback(() => {
    setGenSettings((prev) => ({
      ...prev,
      viralScoreMin: 0,
      viralScoreMax: 100,
      minRelevance: Math.max(20, Math.min(30, prev.minRelevance || 50)),
    }));
    setTimeout(() => handleGenerateClips(), 100);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const updateGen = (key, value) => {
    setGenSettings((prev) => ({ ...prev, [key]: value }));
  };

  // Auto-apply flash indicator: deep compare, excluding transient keys that
  // get initialized asynchronously (speakerColors, speakerNames)
  const prevClipSettingsJsonRef = useRef('');
  const settingsToCompareJson = useCallback((s) => {
    if (!s) return '';
    const { speakerColors, speakerNames, ...rest } = s;
    return JSON.stringify(rest);
  }, []);
  useEffect(() => {
    const json = settingsToCompareJson(clipSettings);
    if (!clipPreview) { prevClipSettingsJsonRef.current = json; return; }
    if (prevClipSettingsJsonRef.current && prevClipSettingsJsonRef.current !== json) {
      setSettingsAppliedFlash(true);
      if (settingsAppliedTimerRef.current) clearTimeout(settingsAppliedTimerRef.current);
      settingsAppliedTimerRef.current = setTimeout(() => setSettingsAppliedFlash(false), 1800);
    }
    prevClipSettingsJsonRef.current = json;
  }, [clipSettings, clipPreview, settingsToCompareJson]);
  useEffect(() => () => { if (settingsAppliedTimerRef.current) clearTimeout(settingsAppliedTimerRef.current); }, []);

  // Determine if playhead is inside a segment — used for segment-aware subs toggle.
  // MUST be before early returns to satisfy Rules of Hooks.
  const activeSegment = useMemo(() => {
    if (!editorSegments || editorSegments.length === 0) return null;
    return editorSegments.find(s => videoCurrentTime >= s.start && videoCurrentTime < s.end) || null;
  }, [editorSegments, videoCurrentTime]);

  // Effective subs state: active segment's subtitlesEnabled takes precedence over global
  const effectiveSubsEnabled = activeSegment ? (activeSegment.subtitlesEnabled !== false) : clipSettings.subtitlesEnabled;

  const handleSubsToggle = useCallback(() => {
    if (activeSegment) {
      // Toggle the active segment's subtitlesEnabled
      const newVal = activeSegment.subtitlesEnabled === false; // flip: false→true, true/undefined→false
      setEditorSegments(prev =>
        prev.map(s => s.id === activeSegment.id ? { ...s, subtitlesEnabled: newVal } : s)
      );
    } else {
      setClipSettings(prev => ({ ...prev, subtitlesEnabled: !prev.subtitlesEnabled }));
    }
  }, [activeSegment]);

  const handleApplyClipSettings = (settings) => {
    setClipSettings(settings);
    showToast('Clip settings applied to preview & export', 'info');
  };

  // Derive unique speakers from transcript — MUST be before early returns
  // because the useEffect below is a hook and hooks cannot be skipped.
  const speakers = useMemo(() => {
    const sp = [];
    (job?.transcript || []).forEach((seg) => {
      if (!sp.includes(seg.speaker)) sp.push(seg.speaker);
    });
    return sp;
  }, [job?.transcript]);

  // Auto-initialize speaker colors from palette when speakers are detected
  // This ensures each speaker gets a unique color even before ClipSettingsPanel mounts
  // MUST be before early returns to satisfy Rules of Hooks (error #310).
  useEffect(() => {
    if (speakers.length === 0) return;
    setClipSettings((prev) => {
      const currentColors = prev.speakerColors || {};
      const needsInit = speakers.some((sp) => !currentColors[sp]);
      if (!needsInit) return prev;
      const newColors = { ...currentColors };
      speakers.forEach((sp, i) => {
        if (!newColors[sp]) {
          newColors[sp] = DEFAULT_SPEAKER_PALETTE[i % DEFAULT_SPEAKER_PALETTE.length];
        }
      });
      return { ...prev, speakerColors: newColors };
    });
  }, [speakers]);

  // Handle speaker color change from transcript viewer (keeps subtitle colors in sync)
  const handleSpeakerColorChanged = useCallback((newName, color, oldName) => {
    setClipSettings(prev => {
      const colors = { ...prev.speakerColors };
      if (oldName && colors[oldName]) delete colors[oldName];
      colors[newName] = color;
      return { ...prev, speakerColors: colors };
    });
  }, []);

  // Handle new speaker added from transcript viewer
  const handleSpeakerAdded = useCallback((name, color) => {
    setClipSettings(prev => {
      const colors = { ...prev.speakerColors };
      colors[name] = color;
      return { ...prev, speakerColors: colors };
    });
    // Also register the speaker name on the server
    if (jobId) {
      fetch(`/api/jobs/${jobId}/speakers`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ speaker_names: { [name]: name } }),
      }).catch(() => {});
    }
  }, [jobId]);

  // --- Auto-trigger subject tracking when clip or aspect ratio changes ---
  // --- Per-clip subject tracking: staged AR pattern ---
  // Uses pendingAR (clipSettings.aspectRatio) vs activeAspectRatio.
  // The preview only switches AR once tracking data is ready.
  const prevAnalysisTrackingRef = useRef({ clipId: null, ar: null });
  const subjectTrackingPollRef = useRef(null);

  // Sync activeAR when clip changes or closes
  useEffect(() => {
    if (!job) return;

    // Full-video editor mode (no clip selected) — promote pending AR directly.
    // The VideoEditor's own subject-keyframe effect will populate cropSegments
    // from job.scenes / dense keyframes as soon as isCrop flips true.
    if (!clipPreview) {
      setActiveAspectRatio(clipSettings?.aspectRatio || null);
      setTrackingLoading(false);
      return;
    }

    // Per-clip mode: only promote once tracking data is ready.
    const scenes = job.scenes || [];
    const hasSparseAiData = scenes.some((s) => {
      const sx = typeof s === 'object' ? (s.subject_x ?? 50) : 50;
      return sx !== 50;
    });
    // Dense pipeline: real positions live in precise_x / face_positions,
    // and dense_tracking_summary is set when the pipeline ran.
    const hasDenseData = !!job.dense_tracking_summary ||
      scenes.some((s) => s.precise_x != null && s.precise_x !== 50);

    if (hasSparseAiData || hasDenseData) {
      setActiveAspectRatio(clipSettings?.aspectRatio || null);
    }
  }, [clipPreview?.id, clipSettings?.aspectRatio, job?.id]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    // Clean up any previous polling interval
    if (subjectTrackingPollRef.current) {
      clearInterval(subjectTrackingPollRef.current);
      subjectTrackingPollRef.current = null;
    }

    // Guard: job not loaded yet — skip tracking logic
    if (!job) return;

    const requestedAr = clipSettings?.aspectRatio;
    const prev = prevAnalysisTrackingRef.current;
    const clipId = clipPreview?.id ?? null;

    const clipChanged = clipId !== prev.clipId;
    const arChanged = requestedAr !== prev.ar;
    prevAnalysisTrackingRef.current = { clipId, ar: requestedAr };

    if (!clipPreview || !requestedAr) {
      setActiveAspectRatio(requestedAr || null);
      setTrackingLoading(false);
      return;
    }
    if (!clipChanged && !arChanged) return;

    // Check if this job already has AI-detected per-scene subject positions
    const scenes = job.scenes || [];
    const hasSparseData = scenes.some((s) => {
      const sx = typeof s === 'object' ? (s.subject_x ?? 50) : 50;
      return sx !== 50;
    });
    const hasDenseData = !!job.dense_tracking_summary ||
      scenes.some((s) => s.precise_x != null && s.precise_x !== 50);
    const hasAiData = hasSparseData || hasDenseData;

    let cancelled = false;
    if (hasAiData) {
      // CASE A: Data exists — apply immediately
      setActiveAspectRatio(requestedAr);
      setTrackingLoading(false);
    } else if (jobId) {
      // CASE B: No AI data — keep preview at OLD AR while backend analyzes.
      setTrackingLoading(true);
      fetch(`/api/jobs/${jobId}/recenter-subject`, { method: 'POST' })
        .then((res) => {
          if (cancelled || !res.ok) throw new Error('recenter failed');
          return res.json();
        })
        .then((data) => {
          if (cancelled) return;
          if (data.status === 'reanalyzing') {
            showToast(`Analyzing subject position for ${requestedAr} crop...`, 'info');
            const poll = setInterval(async () => {
              if (cancelled) { clearInterval(poll); return; }
              try {
                const jr = await fetch(`/api/jobs/${jobId}`, { cache: 'no-store' });
                if (!jr.ok) return;
                const jd = await jr.json();
                const sxVals = (jd.scenes || []).map((s) => s.subject_x);
                if (sxVals.some((v) => v !== 50)) {
                  clearInterval(poll);
                  subjectTrackingPollRef.current = null;
                  await fetchJob();
                  // NOW apply the new AR — data is ready
                  setActiveAspectRatio(requestedAr);
                  setTrackingLoading(false);
                  showToast('Subject tracking ready — preview updated', 'success');
                }
              } catch { /* ignore polling errors */ }
            }, 3000);
            subjectTrackingPollRef.current = poll;
            // Timeout: apply AR anyway after 2 minutes
            setTimeout(() => {
              clearInterval(poll);
              subjectTrackingPollRef.current = null;
              setActiveAspectRatio(requestedAr);
              setTrackingLoading(false);
            }, 120000);
          } else {
            // Data already exists — refresh and apply
            fetchJob();
            setActiveAspectRatio(requestedAr);
            setTrackingLoading(false);
            showToast('Subject tracking ready — preview updated', 'success');
          }
        })
        .catch((err) => {
          if (!cancelled) {
            console.warn('[Analysis] Subject tracking auto-trigger failed:', err);
            // Network error — apply AR anyway as fallback
            setActiveAspectRatio(requestedAr);
            setTrackingLoading(false);
          }
        });
    }

    return () => {
      cancelled = true;
      if (subjectTrackingPollRef.current) {
        clearInterval(subjectTrackingPollRef.current);
        subjectTrackingPollRef.current = null;
      }
    };
  }, [clipPreview?.id, clipSettings?.aspectRatio, job?.scenes]); // eslint-disable-line react-hooks/exhaustive-deps

  // ══════════════════════════════════════════════════════════════════
  // !! ALL React hooks (useState, useEffect, useRef, useMemo,
  // !! useCallback) MUST be declared ABOVE this line.
  // !! Moving hooks below causes React error #310 on page load.
  // ══════════════════════════════════════════════════════════════════
  if (loading) {
    return (
      <div style={{ textAlign: 'center', padding: 48, color: 'var(--text-secondary)' }}>
        <div style={{
          width: 24, height: 24, border: '2px solid var(--border)', borderTopColor: 'var(--accent-cyan)',
          borderRadius: '50%', animation: 'spin 0.8s linear infinite',
          margin: '0 auto 12px',
        }} />
        <div style={{ fontSize: 14, marginBottom: 4 }}>Connecting to analysis...</div>
        <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>
          Your video has been uploaded. The analysis pipeline is starting up.
        </div>
      </div>
    );
  }

  if (!job) {
    return <div style={{ textAlign: 'center', padding: 48, color: 'var(--danger)' }}>Job not found</div>;
  }

  const isProcessing = !['complete', 'failed', 'cancelled'].includes(job.status);

  // Parse source video dimensions (plain variable — not a hook, so safe after early returns)
  let sourceDims = { w: 1920, h: 1080 };
  if (job.resolution) {
    const parts = job.resolution.split('x').map(Number);
    if (parts.length === 2 && parts[0] > 0 && parts[1] > 0) {
      sourceDims = { w: parts[0], h: parts[1] };
    }
  }
  const videoSrc = `/api/files/${jobId}/video.${job.file_path?.split('.').pop() || 'mp4'}`;
  const bestClipId = job.clips?.length ? job.clips.reduce((best, c) => c.viral_score > best.viral_score ? c : best, job.clips[0])?.id : null;

  // Compute subject_x from scenes (with boundary interpolation for clips between scene timestamps)
  const clipSubjectX = clipPreview && job.scenes?.length
    ? computeClipSubjectX(job.scenes, clipPreview.start_time, clipPreview.end_time)
    : 50;

  // Always use ClipPreview when a clip is selected so it responds to
  // aspect ratio and subtitle settings changes in real time.
  const showExportPreview = !!clipPreview;

  // Derive unique clip types for filter dropdown
  const analysisClipTypes = [...new Set((job.clips || []).map((c) => c.clip_type).filter(Boolean))].sort();

  // Filter and sort clips
  let filteredClips = [...(job.clips || [])];
  if (filters.minScore > 0) filteredClips = filteredClips.filter((c) => c.viral_score >= filters.minScore);
  if (filters.platform !== 'all') filteredClips = filteredClips.filter((c) => c.platform === filters.platform || c.platform === 'both');
  if (filters.type !== 'all') filteredClips = filteredClips.filter((c) => c.clip_type === filters.type);

  // Apply text search filter — covers all visible text on the clip card
  if (clipSearchQuery.trim()) {
    const q = clipSearchQuery.trim().toLowerCase();
    filteredClips = filteredClips.filter((c) =>
      (c.title || '').toLowerCase().includes(q) ||
      (c.suggested_caption || '').toLowerCase().includes(q) ||
      (c.hook_text || '').toLowerCase().includes(q) ||
      (c.why_this_works || '').toLowerCase().includes(q) ||
      (c.clip_type || '').toLowerCase().includes(q) ||
      (c.platform || '').toLowerCase().includes(q) ||
      (c.viral_score_reasoning || '').toLowerCase().includes(q) ||
      (c.clip_focus || '').toLowerCase().includes(q) ||
      String(c.id).includes(q) ||
      String(c.viral_score).includes(q) ||
      (c.suggested_hashtags || []).some((h) => h.toLowerCase().includes(q))
    );
  }

  filteredClips.sort((a, b) => {
    if (filters.sort === 'newest') return b.id - a.id;
    if (filters.sort === 'duration') return b.duration - a.duration;
    if (filters.sort === 'duration_short') return a.duration - b.duration;
    if (filters.sort === 'timestamp') return a.start_time - b.start_time;
    if (filters.sort === 'score_low') return a.viral_score - b.viral_score;
    if (filters.sort === 'title_az') return (a.title || '').localeCompare(b.title || '');
    return b.viral_score - a.viral_score;
  });

  // ── Inline subtitle settings toolbar + panel (shared by both editors) ──
  const builtinFonts = [
    'DM Sans', 'Montserrat', 'Open Sans', 'Roboto', 'Poppins', 'Inter',
    'Nunito', 'Lato', 'Oswald', 'Playfair Display', 'Bebas Neue',
    'Liberation Sans', 'Liberation Serif', 'Liberation Mono',
    'DejaVu Sans', 'DejaVu Serif', 'DejaVu Sans Mono', 'FreeSans',
  ];
  const inlineLabelStyle = { fontSize: 10, fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.04em' };
  const inlineFieldStyle = { display: 'flex', flexDirection: 'column', gap: 3, minWidth: 80 };
  const inlineChipStyle = (active) => ({
    padding: '4px 8px', fontSize: 10, fontWeight: 600,
    background: active ? 'var(--accent-cyan)' : 'var(--bg-elevated)',
    color: active ? '#fff' : 'var(--text-secondary)',
    border: '1px solid var(--border)', borderRadius: 'var(--radius-xs)', cursor: 'pointer',
  });
  const updateCS = (key, val) => setClipSettings(prev => ({ ...prev, [key]: val }));

  // ── Preset helpers for inline bar ──
  const handleInlineSavePreset = async () => {
    const name = inlinePresetName.trim();
    if (!name) return;
    const { speakerColors, ...settingsToSave } = clipSettings;
    try {
      const res = await fetch('/api/clip-presets', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, settings: settingsToSave }),
      });
      if (res.ok) {
        const preset = await res.json();
        setClipPresets(prev => [...prev, preset]);
        setInlinePresetName('');
        setInlinePresetSaveOpen(false);
        setInlineActivePreset(preset.name);
        showToast(`Preset "${name}" saved`, 'success');
      }
    } catch { showToast('Failed to save preset', 'error'); }
  };

  const handleInlineLoadPreset = (presetId) => {
    if (!presetId) return;
    const preset = clipPresets.find(p => p.id === presetId);
    if (!preset) return;
    const merged = { ...clipSettings, ...sanitizeSubtitleSettings(preset.settings), speakerColors: clipSettings.speakerColors };
    setClipSettings(merged);
    setInlineActivePreset(preset.name);
    showToast(`Preset "${preset.name}" loaded — speed: ${merged.playbackSpeed || 1}x, volume: ${merged.playbackVolume ?? 100}%`, 'info');
  };

  const handleApplyPresetToSegment = (presetId) => {
    if (!activeSegment) {
      showToast('No segment selected — place the playhead inside a timeline segment', 'warning');
      return;
    }
    const preset = presetId ? clipPresets.find(p => p.id === presetId) : null;
    const settingsToApply = preset ? { ...clipSettings, ...sanitizeSubtitleSettings(preset.settings), speakerColors: clipSettings.speakerColors } : clipSettings;
    setEditorSegments(prev =>
      prev.map(s => s.id === activeSegment.id ? {
        ...s,
        subtitlesEnabled: settingsToApply.subtitlesEnabled,
        playbackSpeed: settingsToApply.playbackSpeed,
        playbackVolume: settingsToApply.playbackVolume,
        aspectRatio: settingsToApply.aspectRatio,
        exportQuality: settingsToApply.exportQuality,
        subtitleSettings: settingsToApply,
      } : s)
    );
    if (preset) setClipSettings(settingsToApply);
    showToast(`Settings applied to segment at ${formatDuration(activeSegment.start)}`, 'success');
  };

  const handleInlineDeletePreset = async (presetId) => {
    const preset = clipPresets.find(p => p.id === presetId);
    try {
      const res = await fetch(`/api/clip-presets/${presetId}`, { method: 'DELETE' });
      if (res.ok) {
        setClipPresets(prev => prev.filter(p => p.id !== presetId));
        if (preset && preset.name === inlineActivePreset) setInlineActivePreset('');
        showToast(`Preset deleted`, 'success');
      }
    } catch {}
  };

  // ── Preset Bar (rendered below subtitle toolbar) ──
  const renderPresetBar = () => (
    <div style={{
      display: 'flex', alignItems: 'center', gap: 8, padding: '6px 12px',
      background: 'var(--bg-panel)', borderTop: '1px solid var(--border)',
      flexWrap: isMobile ? 'nowrap' : 'wrap', marginTop: -1,
      overflowX: isMobile ? 'auto' : undefined,
      WebkitOverflowScrolling: isMobile ? 'touch' : undefined,
      scrollbarWidth: isMobile ? 'none' : undefined,
      opacity: isProcessing ? 0.4 : 1,
      pointerEvents: isProcessing ? 'none' : 'auto',
    }}>
      {/* Label */}
      <span style={{
        fontSize: 10, fontWeight: 600, color: 'var(--text-muted)',
        textTransform: 'uppercase', letterSpacing: '0.05em', whiteSpace: 'nowrap',
      }}>Presets</span>

      {/* Load dropdown */}
      <select
        value=""
        onChange={(e) => handleInlineLoadPreset(e.target.value)}
        style={{
          padding: '4px 8px', fontSize: 11, minWidth: 140,
          borderRadius: 'var(--radius-sm)',
          background: 'var(--bg-elevated)', color: 'var(--text-primary)',
          border: '1px solid var(--border)',
        }}
      >
        <option value="">Load preset...</option>
        {clipPresets.map(p => (
          <option key={p.id} value={p.id}>{String(p.name || '')}</option>
        ))}
      </select>

      {/* Save button */}
      <button
        onClick={() => setInlinePresetSaveOpen(v => !v)}
        title="Save current settings as a preset"
        style={{
          padding: '4px 10px', fontSize: 10, fontWeight: 600,
          background: inlinePresetSaveOpen ? 'var(--accent-cyan-dim)' : 'var(--bg-elevated)',
          color: inlinePresetSaveOpen ? 'var(--accent-cyan)' : 'var(--text-secondary)',
          border: `1px solid ${inlinePresetSaveOpen ? 'var(--accent-cyan)' : 'var(--border)'}`,
          borderRadius: 'var(--radius-sm)', cursor: 'pointer', whiteSpace: 'nowrap',
        }}
      >
        + Save
      </button>

      {/* Apply to segment button */}
      <button
        onClick={() => handleApplyPresetToSegment(null)}
        title={activeSegment ? `Apply current settings to segment at ${formatDuration(activeSegment.start)}` : 'Place playhead inside a segment first'}
        style={{
          padding: '4px 10px', fontSize: 10, fontWeight: 600,
          background: activeSegment ? 'var(--accent-cyan)' : 'var(--bg-elevated)',
          color: activeSegment ? '#fff' : 'var(--text-muted)',
          border: 'none',
          borderRadius: 'var(--radius-sm)',
          cursor: activeSegment ? 'pointer' : 'default',
          opacity: activeSegment ? 1 : 0.5,
          whiteSpace: 'nowrap',
        }}
      >
        Apply to Segment
      </button>

      {/* Active preset indicator */}
      {inlineActivePreset && (
        <span style={{
          fontSize: 10, fontFamily: 'var(--font-mono)',
          color: 'var(--accent-cyan)', whiteSpace: 'nowrap',
        }}>
          Active: {String(inlineActivePreset || '')}
        </span>
      )}

      {/* Delete dropdown — only when presets exist */}
      {clipPresets.length > 0 && (
        <select
          value=""
          onChange={(e) => { if (e.target.value) handleInlineDeletePreset(e.target.value); }}
          title="Delete a preset"
          style={{
            marginLeft: isMobile ? undefined : 'auto',
            padding: '4px 6px', fontSize: 10,
            borderRadius: 'var(--radius-sm)',
            background: 'var(--bg-elevated)', color: 'var(--text-muted)',
            border: '1px solid var(--border)', maxWidth: 100,
          }}
        >
          <option value="">Delete...</option>
          {clipPresets.map(p => (
            <option key={p.id} value={p.id}>{String(p.name || '')}</option>
          ))}
        </select>
      )}

      {/* Save input row (conditionally shown) */}
      {inlinePresetSaveOpen && (
        <div style={{ width: '100%', display: 'flex', gap: 6, marginTop: 4 }}>
          <input
            type="text"
            placeholder="Preset name..."
            value={inlinePresetName}
            onChange={(e) => setInlinePresetName(e.target.value)}
            onKeyDown={(e) => { if (e.key === 'Enter') handleInlineSavePreset(); }}
            autoFocus
            style={{
              flex: 1, padding: '5px 8px', fontSize: 11,
              fontFamily: 'var(--font-mono)',
              background: 'var(--bg-elevated)', color: 'var(--accent-cyan)',
              border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)',
              outline: 'none',
            }}
          />
          <button
            onClick={handleInlineSavePreset}
            disabled={!inlinePresetName.trim()}
            style={{
              padding: '5px 12px', fontSize: 10, fontWeight: 600,
              background: inlinePresetName.trim() ? 'var(--accent-cyan)' : 'var(--bg-elevated)',
              color: inlinePresetName.trim() ? '#fff' : 'var(--text-muted)',
              border: 'none', borderRadius: 'var(--radius-sm)',
              cursor: inlinePresetName.trim() ? 'pointer' : 'default',
            }}
          >
            Save
          </button>
        </div>
      )}
    </div>
  );

  const renderInlineSubToolbar = (extraLeft) => (
    <div style={{
      display: 'flex', alignItems: 'center', gap: 8, padding: '8px 12px',
      background: 'var(--bg-panel)', borderRadius: '0 0 var(--radius-md) var(--radius-md)',
      borderTop: '1px solid var(--border)', flexWrap: 'wrap', marginTop: -1,
      justifyContent: isMobile ? 'center' : undefined,
      opacity: isProcessing ? 0.4 : 1,
      pointerEvents: isProcessing ? 'none' : 'auto',
    }}>
      {extraLeft}
      <button
        onClick={handleSubsToggle}
        style={{
          display: 'flex', alignItems: 'center', gap: 4,
          padding: '5px 10px', fontSize: 11, fontWeight: 600,
          background: effectiveSubsEnabled ? 'var(--accent-cyan-dim)' : 'var(--bg-elevated)',
          color: effectiveSubsEnabled ? 'var(--accent-cyan)' : 'var(--text-secondary)',
          border: `1px solid ${effectiveSubsEnabled ? 'var(--accent-cyan)' : 'var(--border)'}`,
          borderRadius: 'var(--radius-sm)', cursor: 'pointer', whiteSpace: 'nowrap',
        }}>
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <rect x="1" y="4" width="22" height="16" rx="2" /><line x1="1" y1="14" x2="23" y2="14" />
        </svg>
        {activeSegment ? 'Segment ' : ''}Subs {effectiveSubsEnabled ? 'On' : 'Off'}
      </button>
      <button
        onClick={() => !isProcessing && setShowInlineSubSettings(v => !v)}
        disabled={isProcessing}
        style={{
          display: 'flex', alignItems: 'center', gap: 3,
          padding: '5px 8px', fontSize: 10, fontWeight: 500,
          background: showInlineSubSettings ? 'var(--accent-cyan-dim)' : 'transparent',
          color: 'var(--text-muted)',
          border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)',
          cursor: isProcessing ? 'not-allowed' : 'pointer', whiteSpace: 'nowrap',
          opacity: isProcessing ? 0.4 : 1,
        }}>
        <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <circle cx="12" cy="12" r="3" /><path d="M19.4 15a1.65 1.65 0 00.33 1.82l.06.06a2 2 0 010 2.83 2 2 0 01-2.83 0l-.06-.06a1.65 1.65 0 00-1.82-.33 1.65 1.65 0 00-1 1.51V21a2 2 0 01-4 0v-.09A1.65 1.65 0 009 19.4a1.65 1.65 0 00-1.82.33l-.06.06a2 2 0 01-2.83-2.83l.06-.06A1.65 1.65 0 004.68 15a1.65 1.65 0 00-1.51-1H3a2 2 0 010-4h.09A1.65 1.65 0 004.6 9a1.65 1.65 0 00-.33-1.82l-.06-.06a2 2 0 012.83-2.83l.06.06A1.65 1.65 0 009 4.68a1.65 1.65 0 001-1.51V3a2 2 0 014 0v.09a1.65 1.65 0 001 1.51 1.65 1.65 0 001.82-.33l.06-.06a2 2 0 012.83 2.83l-.06.06A1.65 1.65 0 0019.4 9a1.65 1.65 0 001.51 1H21a2 2 0 010 4h-.09a1.65 1.65 0 00-1.51 1z" />
        </svg>
        {showInlineSubSettings ? 'Hide Settings' : 'Subtitle Settings'}
      </button>
      {/* Mark Key Scene — adds current playhead position as a keyscene for AI clip generation */}
      <button
        onClick={() => !isProcessing && setShowMarkScene(v => !v)}
        disabled={isProcessing}
        title={isProcessing ? 'Available after analysis completes' : 'Mark current moment as a key scene for AI clip generation'}
        style={{
          display: 'flex', alignItems: 'center', gap: 4,
          padding: '5px 10px', fontSize: 11, fontWeight: 600,
          background: showMarkScene ? 'var(--accent-amber-dim, rgba(255,159,10,0.12))' : 'var(--bg-elevated)',
          color: showMarkScene ? 'var(--accent-amber, #FF9F0A)' : 'var(--text-secondary)',
          border: `1px solid ${showMarkScene ? 'var(--accent-amber, #FF9F0A)' : 'var(--border)'}`,
          borderRadius: 'var(--radius-sm)', cursor: isProcessing ? 'not-allowed' : 'pointer', whiteSpace: 'nowrap',
          opacity: isProcessing ? 0.4 : 1,
        }}>
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2" />
        </svg>
        Mark Key Scene
      </button>
      {/* Apply Settings — confirms to user that current settings will be used for export */}
      <button
        onClick={() => {
          setSettingsAppliedFlash(true);
          if (settingsAppliedTimerRef.current) clearTimeout(settingsAppliedTimerRef.current);
          settingsAppliedTimerRef.current = setTimeout(() => setSettingsAppliedFlash(false), 2500);
          showToast('Settings applied — your exported video will use these subtitle settings', 'info');
        }}
        style={{
          marginLeft: isMobile ? undefined : 'auto',
          display: 'flex', alignItems: 'center', gap: 4,
          padding: '5px 12px', fontSize: 11, fontWeight: 600,
          background: settingsAppliedFlash ? 'var(--success)' : 'var(--bg-elevated)',
          color: settingsAppliedFlash ? '#fff' : 'var(--text-secondary)',
          border: `1px solid ${settingsAppliedFlash ? 'var(--success)' : 'var(--border)'}`,
          borderRadius: 'var(--radius-sm)', cursor: 'pointer', whiteSpace: 'nowrap',
          transition: 'all 0.2s ease',
        }}>
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
          {settingsAppliedFlash
            ? <polyline points="20 6 9 17 4 12" />
            : <><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4" /><polyline points="7 10 12 15 17 10" /><line x1="12" y1="15" x2="12" y2="3" /></>
          }
        </svg>
        {settingsAppliedFlash ? 'Settings Applied' : 'Apply Settings'}
      </button>
    </div>
  );

  const renderMarkScenePanel = () => {
    if (!showMarkScene) return null;
    return (
      <div style={{
        display: 'flex', alignItems: 'center', gap: 8, padding: '8px 12px',
        background: 'var(--bg-panel)', borderTop: '1px solid var(--border)',
        flexWrap: 'wrap', marginTop: -1,
      }}>
        <span style={{
          fontSize: 10, fontFamily: 'var(--font-mono)', color: 'var(--accent-amber, #FF9F0A)',
          textTransform: 'uppercase', letterSpacing: '0.05em', whiteSpace: 'nowrap',
        }}>
          {formatDuration(videoCurrentTime)}
        </span>
        <input
          type="text"
          value={markSceneDesc}
          onChange={(e) => setMarkSceneDesc(e.target.value)}
          onKeyDown={(e) => { if (e.key === 'Enter') handleMarkKeyScene(); }}
          placeholder="Describe this moment (e.g. 'dramatic reveal', 'emotional reaction')..."
          style={{
            flex: 1, minWidth: 180, padding: '5px 10px', fontSize: 12,
            background: 'var(--bg-elevated)', color: 'var(--text-primary)',
            border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)',
            outline: 'none',
          }}
          autoFocus
        />
        <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
          <span style={{ fontSize: 10, color: 'var(--text-muted)', whiteSpace: 'nowrap' }}>Score:</span>
          <select
            value={markSceneScore}
            onChange={(e) => setMarkSceneScore(parseInt(e.target.value))}
            style={{
              padding: '4px 6px', fontSize: 11, background: 'var(--bg-elevated)',
              color: 'var(--text-primary)', border: '1px solid var(--border)',
              borderRadius: 'var(--radius-sm)', cursor: 'pointer',
            }}
          >
            {[10, 9, 8, 7, 6, 5, 4, 3, 2, 1].map(v => (
              <option key={v} value={v}>{v}{v >= 9 ? ' - viral' : v >= 7 ? ' - compelling' : v >= 4 ? ' - interesting' : ''}</option>
            ))}
          </select>
        </div>
        <button
          onClick={handleMarkKeyScene}
          disabled={markSceneSaving || !markSceneDesc.trim()}
          style={{
            display: 'flex', alignItems: 'center', gap: 4,
            padding: '5px 12px', fontSize: 11, fontWeight: 600,
            background: markSceneDesc.trim() ? 'var(--accent-amber, #FF9F0A)' : 'var(--bg-elevated)',
            color: markSceneDesc.trim() ? '#fff' : 'var(--text-muted)',
            border: 'none', borderRadius: 'var(--radius-sm)',
            cursor: markSceneDesc.trim() ? 'pointer' : 'not-allowed',
            whiteSpace: 'nowrap', opacity: markSceneSaving ? 0.6 : 1,
          }}
        >
          {markSceneSaving ? 'Saving...' : 'Add'}
        </button>
        <button
          onClick={() => { setShowMarkScene(false); setMarkSceneDesc(''); }}
          style={{
            padding: '5px 8px', fontSize: 11, background: 'transparent',
            color: 'var(--text-muted)', border: '1px solid var(--border)',
            borderRadius: 'var(--radius-sm)', cursor: 'pointer',
          }}
        >
          Cancel
        </button>
      </div>
    );
  };

  const renderInlineSubPanel = () => {
    if (!showInlineSubSettings) return null;
    return (
      <div style={{
        padding: isMobile ? '8px 10px' : '12px 16px', background: 'var(--bg-panel)',
        border: '1px solid var(--border)', borderTop: 'none',
        borderRadius: '0 0 var(--radius-md) var(--radius-md)',
      }}>
        {/* Row 1: Font, Size, Weight, Color */}
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: isMobile ? 8 : 12, alignItems: 'flex-end', marginBottom: 10 }}>
          <div style={{ ...inlineFieldStyle, minWidth: isMobile ? 90 : 120 }}>
            <label style={inlineLabelStyle}>Font</label>
            <select value={clipSettings.subtitleFont || 'DM Sans'} onChange={e => updateCS('subtitleFont', e.target.value)}
              style={{ padding: '5px 8px', fontSize: 12, borderRadius: 'var(--radius-xs)', border: '1px solid var(--border)', background: 'var(--bg-elevated)', color: 'var(--text-primary)' }}>
              {builtinFonts.map(f => <option key={f} value={f}>{f}</option>)}
              {inlineCustomFonts.length > 0 && (
                <optgroup label="Custom Fonts">
                  {inlineCustomFonts.map(f => <option key={f.name} value={f.name}>{String(f.name ?? '')}</option>)}
                </optgroup>
              )}
            </select>
          </div>
          <div style={inlineFieldStyle}>
            <label style={inlineLabelStyle}>Size</label>
            <div style={{ display: 'flex', gap: 2 }}>
              {[{ l: 'S', v: 22 }, { l: 'M', v: 30 }, { l: 'L', v: 40 }].map(s => (
                <button key={s.v} onClick={() => updateCS('subtitleSize', s.v)} style={inlineChipStyle((clipSettings.subtitleSize || 30) === s.v)}>{s.l}</button>
              ))}
            </div>
          </div>
          <div style={inlineFieldStyle}>
            <label style={inlineLabelStyle}>Weight</label>
            <div style={{ display: 'flex', gap: 2 }}>
              {[{ v: 400, l: 'Regular' }, { v: 700, l: 'Bold' }].map(w => {
                const cur = typeof clipSettings.subtitleFontWeight === 'number' ? clipSettings.subtitleFontWeight : (clipSettings.subtitleFontWeight === 'bold' ? 700 : 400);
                return (
                  <button key={w.v} onClick={() => updateCS('subtitleFontWeight', w.v)}
                    style={{ ...inlineChipStyle(cur === w.v), fontWeight: w.v }}>{w.l}</button>
                );
              })}
            </div>
          </div>
          <div style={inlineFieldStyle}>
            <label style={inlineLabelStyle}>Color</label>
            <input type="color" value={clipSettings.subtitleFontColor || '#FFFFFF'} onChange={e => updateCS('subtitleFontColor', e.target.value)}
              style={{ width: 32, height: 28, border: '1px solid var(--border)', borderRadius: 'var(--radius-xs)', cursor: 'pointer', padding: 1 }} />
          </div>
          <div style={inlineFieldStyle}>
            <label style={inlineLabelStyle}>Position</label>
            <div style={{ display: 'flex', gap: 2 }}>
              {['top', 'center', 'bottom'].map(p => (
                <button key={p} onClick={() => updateCS('subtitlePosition', p)} style={{ ...inlineChipStyle((clipSettings.subtitlePosition || 'bottom') === p), textTransform: 'capitalize' }}>{p}</button>
              ))}
            </div>
          </div>
        </div>

        {/* Row 2: Max Width, Offset, Outline, Max Words */}
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: isMobile ? 8 : 12, alignItems: 'flex-end', marginBottom: 10 }}>
          <div style={{ ...inlineFieldStyle, minWidth: isMobile ? 90 : 120 }}>
            <label style={inlineLabelStyle}>Max Width</label>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <input type="range" min="20" max="100" step="5" value={Math.min(100, Math.max(20, clipSettings.subtitleMaxWidth))}
                onChange={e => updateCS('subtitleMaxWidth', parseInt(e.target.value))}
                style={{ flex: 1, accentColor: 'var(--accent-cyan)', minWidth: 60 }} />
              <span style={{ fontSize: 10, fontFamily: 'var(--font-mono)', color: 'var(--text-muted)', minWidth: 28, textAlign: 'right' }}>{String(clipSettings.subtitleMaxWidth ?? '')}%</span>
            </div>
          </div>
          <div style={{ ...inlineFieldStyle, minWidth: 100 }}>
            <label style={inlineLabelStyle}>Vertical Offset</label>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <input type="range" min="0" max="40" step="1" value={clipSettings.subtitleOffsetV ?? 4}
                onChange={e => updateCS('subtitleOffsetV', parseInt(e.target.value))}
                style={{ flex: 1, accentColor: 'var(--accent-cyan)', minWidth: 60 }} />
              <span style={{ fontSize: 10, fontFamily: 'var(--font-mono)', color: 'var(--text-muted)', minWidth: 28, textAlign: 'right' }}>{String(clipSettings.subtitleOffsetV ?? 4)}%</span>
            </div>
          </div>
          <div style={{ ...inlineFieldStyle, minWidth: 100 }}>
            <label style={inlineLabelStyle}>Outline</label>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <input type="range" min="0" max="10" step="1" value={clipSettings.subtitleOutlineWidth ?? 2}
                onChange={e => updateCS('subtitleOutlineWidth', parseInt(e.target.value))}
                style={{ flex: 1, accentColor: 'var(--accent-cyan)', minWidth: 60 }} />
              <input type="color" value={clipSettings.subtitleOutlineColor || '#000000'} onChange={e => updateCS('subtitleOutlineColor', e.target.value)}
                style={{ width: 24, height: 22, border: '1px solid var(--border)', borderRadius: 'var(--radius-xs)', cursor: 'pointer', padding: 1 }} />
            </div>
          </div>
          <div style={{ ...inlineFieldStyle, minWidth: 80 }}>
            <label style={inlineLabelStyle}>Max Words</label>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <input type="range" min="0" max="12" step="1" value={clipSettings.subtitleMaxWords ?? 0}
                onChange={e => updateCS('subtitleMaxWords', parseInt(e.target.value))}
                style={{ flex: 1, accentColor: 'var(--accent-cyan)', minWidth: 50 }} />
              <span style={{ fontSize: 10, fontFamily: 'var(--font-mono)', color: 'var(--text-muted)', minWidth: 20, textAlign: 'right' }}>{typeof clipSettings.subtitleMaxWords === 'number' ? (clipSettings.subtitleMaxWords || 'Off') : 'Off'}</span>
            </div>
          </div>
        </div>

        {/* Row 3: Background, Active Word, Speaker Labels */}
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: isMobile ? 8 : 12, alignItems: 'flex-end' }}>
          <div style={inlineFieldStyle}>
            <label style={inlineLabelStyle}>Background</label>
            <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
              <button onClick={() => updateCS('subtitleBgEnabled', !clipSettings.subtitleBgEnabled)}
                style={inlineChipStyle(clipSettings.subtitleBgEnabled)}>
                {clipSettings.subtitleBgEnabled ? 'On' : 'Off'}
              </button>
              {clipSettings.subtitleBgEnabled && (
                <>
                  <input type="color" value={clipSettings.subtitleBgColor || '#000000'} onChange={e => updateCS('subtitleBgColor', e.target.value)}
                    style={{ width: 24, height: 22, border: '1px solid var(--border)', borderRadius: 'var(--radius-xs)', cursor: 'pointer', padding: 1 }} />
                  <input type="range" min="0" max="100" step="5" value={clipSettings.subtitleBgOpacity ?? 75}
                    onChange={e => updateCS('subtitleBgOpacity', parseInt(e.target.value))}
                    style={{ width: 50, accentColor: 'var(--accent-cyan)' }} />
                </>
              )}
            </div>
          </div>
          <div style={inlineFieldStyle}>
            <label style={inlineLabelStyle}>Active Word</label>
            <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
              <button onClick={() => updateCS('activeWordEnabled', !clipSettings.activeWordEnabled)}
                style={inlineChipStyle(clipSettings.activeWordEnabled)}>
                {clipSettings.activeWordEnabled ? 'On' : 'Off'}
              </button>
              {clipSettings.activeWordEnabled && (
                <input type="color" value={clipSettings.activeWordColor || '#FFD700'} onChange={e => updateCS('activeWordColor', e.target.value)}
                  style={{ width: 24, height: 22, border: '1px solid var(--border)', borderRadius: 'var(--radius-xs)', cursor: 'pointer', padding: 1 }} />
              )}
            </div>
          </div>
          <div style={inlineFieldStyle}>
            <label style={inlineLabelStyle}>Speaker Labels</label>
            <button onClick={() => updateCS('showSpeakerLabels', !(clipSettings.showSpeakerLabels ?? false))}
              style={inlineChipStyle(clipSettings.showSpeakerLabels ?? false)}>
              {(clipSettings.showSpeakerLabels ?? false) ? 'On' : 'Off'}
            </button>
          </div>
          <div style={inlineFieldStyle}>
            <label style={inlineLabelStyle}>Speaker Colors</label>
            <button onClick={() => updateCS('useSpeakerColors', !(clipSettings.useSpeakerColors ?? true))}
              style={inlineChipStyle(clipSettings.useSpeakerColors ?? true)}>
              {(clipSettings.useSpeakerColors ?? true) ? 'On' : 'Off'}
            </button>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div>
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
              Analyzing subject position for {clipSettings.aspectRatio} crop...
            </div>
          </div>
        </div>
      )}

      {/* Page header — surfaces a Share button so the owner can hand a
          public link to anyone, without requiring the recipient to sign
          in. Renders the clip-scoped variant when an export preview is
          open so the link is automatically narrowed to that clip. */}
      {jobId && (
        <div style={{
          display: 'flex', alignItems: 'center', justifyContent: 'flex-end',
          gap: 8,
          padding: '8px var(--page-pad, 16px)',
          borderBottom: '1px solid var(--border, #26263a)',
        }}>
          <ShareButton
            jobId={jobId}
            clipId={showExportPreview && clipPreview ? clipPreview.id : undefined}
            label={showExportPreview && clipPreview ? 'Share clip' : 'Share analysis'}
          />
        </div>
      )}

      {/* Pipeline diagnostics banner removed per user request —
          Whisper / heuristic / pipeline info lines were cluttering the
          area above the preview player. The backend still records
          ``pipeline_warnings`` on the job (visible in the Logs tab),
          and ``checkEditorCapabilities()`` still runs internally so
          other panels can surface capability issues inline. */}
      

      {/* Video Player (sticky) — hidden on Transcript tab where we show side-by-side layout */}
      <div ref={stickyPlayerRef} style={{ position: 'sticky', top: 0, zIndex: 10, background: 'var(--bg-base)', display: (tab === 2 && !showExportPreview) ? 'none' : 'block' }}>
        {showExportPreview ? (
          <div style={{ position: 'relative', width: isMobile ? '100%' : '85vw', maxWidth: '1600px', margin: '0 auto' }}>
            <VideoEditorBoundary>
            <VideoEditor
              key={`clip-${clipPreview.id}`}
              src={videoSrc}
              clipStart={clipPreview.start_time}
              clipEnd={clipPreview.end_time}
              title={String(clipPreview.title || `Clip ${clipPreview.id}`)}
              aspectRatio={activeAspectRatio || null}
              sourceWidth={sourceDims.w}
              sourceHeight={sourceDims.h}
              subjectX={clipSubjectX}
              scenes={job.scenes || []}
              sceneCuts={job.scene_cut_timestamps || null}
              subjectTrack={job.subject_track || null}
              layoutTimeline={job.layout_timeline || null}
              faceRegistry={job.face_registry_data || null}
              defaultLayoutMode={clipSettings.layoutMode || job.default_layout_mode || 'single'}
              initialVolume={clipSettings.playbackVolume}
              initialSpeed={clipSettings.playbackSpeed}
              onTimeUpdate={setVideoCurrentTime}
              onTrimChange={setEditorTrim}
              onApplyTrim={({ start, end }) => {
                setClipPreview((prev) => prev ? { ...prev, start_time: start, end_time: end } : prev);
                setEditorTrim({ trimStart: 0, trimEnd: 0 });
                showToast('Trim applied — clip range updated', 'info');
              }}
              onAspectRatioChange={(ar) => setClipSettings((prev) => ({ ...prev, aspectRatio: ar }))}
              onVolumeChange={setEditorVolume}
              onSpeedChange={setEditorSpeed}
              onSegmentsChange={(segs) => {
                setEditorSegments(segs);
                if (clipPreview) {
                  clipSegmentsMapRef.current[clipPreview.id] = segs;
                  saveSegmentsToStorage(clipPreview.id, segs);
                }
              }}
              initialSegments={editorSegments}
              settings={clipSettings}
              speakers={speakers}
              speakerNames={job.speaker_names}
              onSettingsChange={setClipSettings}
              jobId={jobId}
              clipId={clipPreview.id}
              transcript={job.translated_transcript?.length ? job.translated_transcript : (job.transcript || [])}
              onTranscriptUpdated={fetchJob}
              isProcessing={isProcessing}
              onSubjectKeyframes={setEditorSubjectKeyframes}
              onClose={() => {
                if (clipPreview) {
                  clipSegmentsMapRef.current[clipPreview.id] = editorSegments;
                  saveSegmentsToStorage(clipPreview.id, editorSegments);
                }
                // Eagerly populate store with full-video subtitles before re-rendering
                const fvTranscript = job.translated_transcript?.length
                  ? job.translated_transcript
                  : (job.transcript || []);
                const fvStart = fullVideoRange ? fullVideoRange.start : 0;
                const fvEnd = fullVideoRange ? fullVideoRange.end : (job.duration || 0);
                useTimelineStore.getState().initFromClip({
                  src: `/api/files/${jobId}/video.${job.file_path?.split('.').pop() || 'mp4'}`,
                  clipStart: fvStart,
                  clipEnd: fvEnd,
                  subtitleSegments: fvTranscript,
                });
                setClipPreview(null);
              }}
              subtitleOverlay={
                <SubtitleOverlay
                  currentTime={videoCurrentTime}
                  transcript={job.translated_transcript?.length ? job.translated_transcript : (job.transcript || [])}
                  clipStart={clipPreview.start_time}
                  clipEnd={clipPreview.end_time}
                  settings={clipSettings}
                  aspectRatio={activeAspectRatio || null}
                  sourceWidth={sourceDims.w}
                  sourceHeight={sourceDims.h}
                  segments={editorSegments}
                />
              }
            />
            </VideoEditorBoundary>
            {/* Auto-applied settings indicator */}
            {settingsAppliedFlash && (
              <div style={{
                position: 'absolute', bottom: 52, left: '50%', transform: 'translateX(-50%)',
                zIndex: 20,
                display: 'flex', alignItems: 'center', gap: 6,
                padding: '6px 14px',
                background: 'var(--success)', color: '#fff',
                borderRadius: 'var(--radius-xl)',
                fontSize: 12, fontWeight: 600,
                boxShadow: '0 4px 20px rgba(0,0,0,0.2)',
                animation: 'slideIn 0.25s ease forwards',
                pointerEvents: 'none',
                whiteSpace: 'nowrap',
              }}>
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                  <polyline points="20 6 9 17 4 12" />
                </svg>
                Settings applied to preview
              </div>
            )}
            {/* Inline subtitle settings for clip preview */}
            {renderInlineSubToolbar(null)}
            {renderPresetBar()}
            {renderMarkScenePanel()}
            {renderInlineSubPanel()}
          </div>
        ) : (
          <div style={{ width: isMobile ? '100%' : '85vw', maxWidth: '1600px', margin: '0 auto' }}>
            <VideoEditorBoundary>
            <VideoEditor
              src={videoSrc}
              clipStart={fullVideoRange ? fullVideoRange.start : 0}
              clipEnd={fullVideoRange ? fullVideoRange.end : (job.duration || 0)}
              title={String(job.filename || 'Full Video')}
              aspectRatio={activeAspectRatio || null}
              sourceWidth={sourceDims.w}
              sourceHeight={sourceDims.h}
              subjectX={50}
              scenes={job.scenes || []}
              sceneCuts={job.scene_cut_timestamps || null}
              subjectTrack={job.subject_track || null}
              layoutTimeline={job.layout_timeline || null}
              faceRegistry={job.face_registry_data || null}
              defaultLayoutMode={clipSettings.layoutMode || job.default_layout_mode || 'single'}
              initialVolume={clipSettings.playbackVolume}
              initialSpeed={clipSettings.playbackSpeed}
              onTimeUpdate={setVideoCurrentTime}
              onTrimChange={setEditorTrim}
              onApplyTrim={({ start, end }) => {
                setFullVideoRange({ start, end });
                setEditorTrim({ trimStart: 0, trimEnd: 0 });
                showToast('Trim applied — video range updated', 'info');
              }}
              onAspectRatioChange={(ar) => setClipSettings((prev) => ({ ...prev, aspectRatio: ar }))}
              onVolumeChange={setEditorVolume}
              onSpeedChange={setEditorSpeed}
              onSegmentsChange={(segs) => {
                setEditorSegments(segs);
                saveSegmentsToStorage('full', segs);
              }}
              initialSegments={editorSegments}
              settings={clipSettings}
              speakers={speakers}
              speakerNames={job.speaker_names}
              onSettingsChange={setClipSettings}
              jobId={jobId}
              transcript={job.translated_transcript?.length ? job.translated_transcript : (job.transcript || [])}
              onTranscriptUpdated={fetchJob}
              isProcessing={isProcessing}
              onSubjectKeyframes={setEditorSubjectKeyframes}
              subtitleOverlay={
                <SubtitleOverlay
                  currentTime={videoCurrentTime}
                  transcript={job.translated_transcript?.length ? job.translated_transcript : (job.transcript || [])}
                  clipStart={fullVideoRange ? fullVideoRange.start : 0}
                  clipEnd={fullVideoRange ? fullVideoRange.end : (job.duration || 0)}
                  settings={clipSettings}
                  aspectRatio={activeAspectRatio || null}
                  sourceWidth={sourceDims.w}
                  sourceHeight={sourceDims.h}
                  segments={editorSegments}
                />
              }
            />
            </VideoEditorBoundary>

            {/* ── Inline Editor Toolbar: Export + Subtitle Settings ── */}
            {renderInlineSubToolbar(
              <>
                {fullVideoExporting ? (
                  <span style={{ fontSize: 11, color: 'var(--accent-cyan)', fontFamily: 'var(--font-mono)' }}>Exporting...</span>
                ) : (
                  <div ref={exportQualityRef} style={{ position: 'relative', display: 'inline-flex' }}>
                    <button onClick={handleExportFullVideo} style={{
                      display: 'flex', alignItems: 'center', gap: 5,
                      padding: '6px 14px', fontSize: 11, fontWeight: 700,
                      background: 'var(--accent-cyan)', color: '#fff',
                      border: 'none', borderRadius: 'var(--radius-sm) 0 0 var(--radius-sm)', cursor: 'pointer', whiteSpace: 'nowrap',
                    }}>
                      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4" /><polyline points="7 10 12 15 17 10" /><line x1="12" y1="15" x2="12" y2="3" />
                      </svg>
                      Export Full Video ({clipSettings?.exportQuality || '1080p'})
                    </button>
                    <button
                      onClick={() => setExportQualityMenuOpen(v => !v)}
                      style={{
                        display: 'flex', alignItems: 'center', padding: '6px 6px',
                        background: 'var(--accent-cyan)', color: '#fff',
                        border: 'none', borderLeft: '1px solid rgba(255,255,255,0.25)',
                        borderRadius: '0 var(--radius-sm) var(--radius-sm) 0', cursor: 'pointer',
                      }}
                    >
                      <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
                        <polyline points="6 9 12 15 18 9" />
                      </svg>
                    </button>
                    {exportQualityMenuOpen && (
                      <div style={{
                        position: 'absolute', top: '100%', right: 0, marginTop: 2,
                        background: 'var(--bg-elevated)', border: '1px solid var(--border)',
                        borderRadius: 'var(--radius-sm)', zIndex: 50, overflow: 'hidden',
                        minWidth: 150, boxShadow: 'var(--shadow-sm)',
                      }}>
                        <div style={{ padding: '6px 10px', fontSize: 10, color: 'var(--text-muted)', borderBottom: '1px solid var(--border)', textTransform: 'uppercase', letterSpacing: 0.5 }}>
                          Export Quality
                        </div>
                        {['720p', '1080p', '4k'].map((q) => (
                          <button
                            key={q}
                            onClick={() => {
                              setClipSettings(prev => ({ ...prev, exportQuality: q }));
                              setExportQualityMenuOpen(false);
                            }}
                            style={{
                              display: 'block', width: '100%', padding: '6px 10px',
                              background: q === (clipSettings?.exportQuality || '1080p') ? 'var(--accent-cyan)' : 'transparent',
                              color: q === (clipSettings?.exportQuality || '1080p') ? '#fff' : 'var(--text-primary)',
                              border: 'none', fontSize: 12, textAlign: 'left', cursor: 'pointer',
                            }}
                            onMouseEnter={e => { if (q !== (clipSettings?.exportQuality || '1080p')) e.target.style.background = 'var(--bg-hover)'; }}
                            onMouseLeave={e => { if (q !== (clipSettings?.exportQuality || '1080p')) e.target.style.background = 'transparent'; }}
                          >
                            {q === '720p' ? '720p (Smaller)' : q === '1080p' ? '1080p (Default)' : '4K (Best)'}
                          </button>
                        ))}
                      </div>
                    )}
                  </div>
                )}
                <div style={{ width: 1, height: 20, background: 'var(--border)', margin: '0 4px' }} />
              </>
            )}
            {renderPresetBar()}
            {renderMarkScenePanel()}
            {renderInlineSubPanel()}
          </div>
        )}
        {/* Export button — visible when a clip is loaded in the preview player */}
        {showExportPreview && clipPreview && (
          <div style={{ display: 'flex', justifyContent: 'center', padding: '8px 0 4px' }}>
            <button
              onClick={() => handleExportClip(clipPreview)}
              disabled={encoding.tasks[`${jobId}_${clipPreview.id}`]?.status === 'encoding'}
              style={{
                display: 'flex', alignItems: 'center', gap: 6,
                padding: '8px 24px',
                fontSize: 12,
                fontWeight: 600,
                background: encoding.tasks[`${jobId}_${clipPreview.id}`]?.status === 'encoding'
                  ? 'var(--bg-elevated)' : 'var(--accent-cyan)',
                color: encoding.tasks[`${jobId}_${clipPreview.id}`]?.status === 'encoding'
                  ? 'var(--text-muted)' : 'var(--bg-base)',
                border: 'none',
                borderRadius: 'var(--radius-sm)',
                cursor: encoding.tasks[`${jobId}_${clipPreview.id}`]?.status === 'encoding'
                  ? 'default' : 'pointer',
                fontFamily: 'var(--font-mono)',
                textTransform: 'uppercase',
                letterSpacing: '0.03em',
              }}
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4" />
                <polyline points="7 10 12 15 17 10" />
                <line x1="12" y1="15" x2="12" y2="3" />
              </svg>
              {encoding.tasks[`${jobId}_${clipPreview.id}`]?.status === 'encoding'
                ? 'Exporting...'
                : `Export Clip (${clipSettings.exportQuality || '1080p'})`}
            </button>
          </div>
        )}
        {/* Current detected subject for previewed clip */}
        {clipPreview && job.scenes?.length > 0 && (() => {
          const inRange = (job.scenes || []).filter(
            (s) => s.timestamp >= clipPreview.start_time && s.timestamp <= clipPreview.end_time
          );
          if (!inRange.length) return null;
          const best = inRange.reduce((a, b) => (b.importance_score > a.importance_score ? b : a), inRange[0]);
          return (
            <div style={{
              padding: '6px 10px', fontSize: 11, color: 'var(--text-muted)',
              background: 'var(--bg-elevated)', borderRadius: 'var(--radius-sm)',
              borderLeft: '2px solid var(--accent-cyan)', lineHeight: 1.4,
              marginTop: 2,
            }}>
              <span style={{ fontWeight: 600, color: 'var(--text-secondary)', fontSize: 10, textTransform: 'uppercase', letterSpacing: '0.03em' }}>
                Current subject (Clip {clipPreview.id}):
              </span>{' '}
              {String(best.description || '').length > 150 ? String(best.description || '').slice(0, 150) + '...' : String(best.description || '')}
              <span style={{ marginLeft: 6, fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--accent-cyan)' }}>
                x={typeof best.subject_x === 'number' ? best.subject_x : 50}%
              </span>
            </div>
          );
        })()}
      </div>

      {/* Progress for active jobs */}
      {isProcessing && (
        <div style={{ padding: '12px 0' }}>
          {/* Queued waiting banner */}
          {(job.status === 'queued' && (job.progress || 0) <= 1) && (
            <div style={{
              marginBottom: 10, padding: '10px 14px',
              background: 'var(--accent-cyan-dim)', border: '1px solid var(--accent-cyan)',
              borderRadius: 'var(--radius-sm)', fontSize: 12, color: 'var(--accent-cyan)',
              display: 'flex', alignItems: 'center', gap: 8,
            }}>
              <span style={{ width: 8, height: 8, borderRadius: '50%', background: 'var(--accent-cyan)', animation: 'pulse 1.5s ease-in-out infinite', flexShrink: 0 }} />
              Upload complete — your video is queued for analysis. The pipeline will start shortly.
            </div>
          )}
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <div style={{ flex: 1 }}>
              <ProgressBar progress={job.progress || 0} message={String(job.progress_message || 'Preparing analysis pipeline...')} />
            </div>
            <button
              disabled={cancellingJob}
              onClick={async () => {
                if (cancellingJob) return;
                setCancellingJob(true);
                showToast('Cancelling...', 'info');
                try {
                  await fetch(`/api/jobs/${jobId}/cancel`, { method: 'POST' });
                  navigate('/');
                } catch {
                  setCancellingJob(false);
                }
              }}
              style={{
                padding: '6px 16px',
                background: 'var(--amber-dim)',
                border: '1px solid var(--accent-amber)',
                color: 'var(--accent-amber)',
                fontSize: 12,
                fontWeight: 600,
                cursor: cancellingJob ? 'default' : 'pointer',
                borderRadius: 'var(--radius-sm)',
                whiteSpace: 'nowrap',
                flexShrink: 0,
                opacity: cancellingJob ? 0.6 : 1,
              }}
            >
              {cancellingJob ? 'Cancelling...' : 'Cancel'}
            </button>
          </div>
          {/* Stuck/slow warning */}
          {stuckSeconds >= 60 && (
            <div style={{
              marginTop: 8, padding: '8px 12px',
              background: stuckSeconds >= 300 ? 'var(--danger-dim, rgba(255,59,48,0.08))' : 'var(--amber-dim)',
              border: `1px solid ${stuckSeconds >= 300 ? 'var(--danger, #ff3b30)' : 'var(--accent-amber)'}`,
              borderRadius: 'var(--radius-sm)', fontSize: 12,
              color: stuckSeconds >= 300 ? 'var(--danger, #ff3b30)' : 'var(--accent-amber)',
              display: 'flex', alignItems: 'center', gap: 8,
            }}>
              <span style={{ animation: 'pulse 1.5s ease-in-out infinite', flexShrink: 0 }}>{'\u25CF'}</span>
              <span>
                {stuckSeconds >= 300
                  ? `No progress update for ${Math.floor(stuckSeconds / 60)}m — the pipeline may be stuck. You can cancel and retry, or check the activity log for details.`
                  : stuckSeconds >= 180
                    ? `No progress update for ${Math.floor(stuckSeconds / 60)}m — the container may be processing a large file. Check the activity log below for details.`
                    : `Still working... no update for ${stuckSeconds}s — large files can take a while to process.`
                }
              </span>
            </div>
          )}
        </div>
      )}

      {/* Encoding progress bars — shown for any active exports on this job */}
      {Object.entries(encoding.tasks).filter(([key, t]) => key.startsWith(`${jobId}_`) && t.status === 'encoding').length > 0 && (
        <div style={{
          padding: '10px 0',
          display: 'flex', flexDirection: 'column', gap: 8,
        }}>
          {Object.entries(encoding.tasks)
            .filter(([key, t]) => key.startsWith(`${jobId}_`) && t.status === 'encoding')
            .map(([key, t]) => (
              <div key={key} style={{
                padding: '10px 14px', background: 'var(--bg-panel)',
                border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)',
              }}>
                <ProgressBar
                  progress={t.progress || 0}
                  message={t.message || `Encoding ${t.clipTitle || 'clip'}...`}
                  variant="cyan"
                />
              </div>
            ))}
        </div>
      )}

      {/* Activity Log */}
      {activityLog.length > 0 && (isProcessing || job.status === 'complete' || job.status === 'failed' || job.status === 'cancelled') && (
        <div style={{
          background: 'var(--bg-panel)', border: '1px solid var(--border)',
          borderRadius: 'var(--radius-sm)', marginBottom: 16, overflow: 'hidden',
        }}>
          <button
            onClick={() => setLogExpanded((p) => !p)}
            style={{
              width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'space-between',
              padding: '8px 12px', background: 'none', border: 'none', cursor: 'pointer',
              color: 'var(--text-secondary)', fontSize: 11, fontFamily: 'var(--font-mono)',
              textTransform: 'uppercase', letterSpacing: '0.05em',
            }}
          >
            <span>
              Processing Log
              <span style={{ color: 'var(--text-muted)', marginLeft: 8, textTransform: 'none', letterSpacing: 0 }}>
                ({activityLog.length} events)
              </span>
            </span>
            <span style={{ fontSize: 14 }}>{logExpanded ? '\u25B4' : '\u25BE'}</span>
          </button>
          {logExpanded && (
            <div style={{
              maxHeight: 200, overflowY: 'auto', padding: isMobile ? '8px 12px' : '12px 16px',
              fontFamily: 'var(--font-mono)', fontSize: 11, lineHeight: 1.7,
            }}>
              {activityLog.map((entry, i) => {
                const colors = {
                  status: 'var(--text-secondary)',
                  success: 'var(--success)',
                  warning: 'var(--accent-amber)',
                  error: 'var(--danger)',
                  info: 'var(--text-muted)',
                };
                return (
                  <div key={i} style={{ display: 'flex', gap: 8 }}>
                    <span style={{ color: 'var(--text-muted)', flexShrink: 0 }}>{String(entry.ts || '')}</span>
                    {entry.progress !== undefined && (
                      <span style={{ color: 'var(--accent-cyan)', flexShrink: 0, minWidth: 30, textAlign: 'right' }}>
                        {typeof entry.progress === 'number' ? entry.progress : String(entry.progress ?? '')}%
                      </span>
                    )}
                    <span style={{ color: colors[entry.type] || colors.status }}>
                      {typeof entry.message === 'string' ? entry.message : String(entry.message ?? '')}
                    </span>
                  </div>
                );
              })}
              <div ref={logEndRef} />
            </div>
          )}
        </div>
      )}

      {/* Cancelled banner */}
      {job.status === 'cancelled' && (
        <div style={{ padding: '12px 16px', background: 'var(--amber-dim)', border: '1px solid var(--accent-amber)', color: 'var(--accent-amber)', fontSize: 13, margin: '12px 0', borderRadius: 'var(--radius-sm)' }}>
          Job was cancelled. Partial results may be available below.
        </div>
      )}

      {/* Error */}
      {job.status === 'failed' && job.error && (
        <div style={{ padding: '12px 16px', background: 'var(--danger-dim)', border: '1px solid var(--danger)', color: 'var(--danger)', fontSize: 13, margin: '12px 0', borderRadius: 'var(--radius-sm)' }}>
          {typeof job.error === 'string' ? job.error : JSON.stringify(job.error)}
        </div>
      )}

      {/* Metadata bar */}
      <div style={{ display: 'flex', gap: 16, padding: '12px 0', flexWrap: 'wrap', fontSize: 12, color: 'var(--text-secondary)', borderBottom: '1px solid var(--border)', marginBottom: 16 }}>
        {job.duration > 0 && <span style={{ fontFamily: 'var(--font-mono)' }}>Duration: {formatDuration(job.duration)}</span>}
        {job.resolution && <span style={{ fontFamily: 'var(--font-mono)' }}>{String(job.resolution)}</span>}
        {job.fps > 0 && <span style={{ fontFamily: 'var(--font-mono)' }}>{job.fps} FPS</span>}
        {job.file_size_mb > 0 && <span style={{ fontFamily: 'var(--font-mono)' }}>{job.file_size_mb.toFixed(1)} MB</span>}
        {Object.keys(job.provider_used || {}).length > 0 && (
          <span style={{ lineHeight: 1.5 }}>
            {Object.entries(job.provider_used).map(([task, model]) => {
              const modelStr = typeof model === 'object' ? JSON.stringify(model) : String(model || '');
              const isFallback = modelStr.includes('(partial)') || modelStr === 'fallback' || modelStr === 'none';
              return (
                <span key={task} style={{
                  display: 'inline-block',
                  marginRight: 8,
                  padding: '1px 6px',
                  borderRadius: 4,
                  background: isFallback ? 'rgba(245,158,11,0.15)' : 'rgba(16,185,129,0.1)',
                  color: isFallback ? 'var(--accent-amber)' : 'var(--text-secondary)',
                  fontSize: 11,
                }}>
                  <strong>{String(task)}</strong>: {modelStr}
                </span>
              );
            })}
          </span>
        )}
        {job.analysis_duration_seconds > 0 && (
          <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--success)' }}>
            Analyzed in {job.analysis_duration_seconds < 60
              ? `${Math.round(job.analysis_duration_seconds)}s`
              : `${Math.floor(job.analysis_duration_seconds / 60)}m ${Math.round(job.analysis_duration_seconds % 60)}s`}
          </span>
        )}
        {/* The Share button lives in the page header (top-right);
            duplicating it inside the metadata bar generated invalid
            ``/share/analysis/<jobId>`` URLs (the public share routes
            are token-based: ``/share/<token>``). Keep the metadata
            bar focused on diagnostics only. */}
      </div>

      {/* Tabs — iOS segmented control on mobile, standard tabs on desktop */}
      {isMobile ? (
        <div style={{
          display: 'flex', gap: 2,
          margin: '0 0 16px',
          padding: 3,
          background: 'var(--bg-elevated)',
          borderRadius: 10,
          overflow: 'hidden',
        }}>
          {TABS.map((t, i) => (
            <button
              key={t}
              onClick={() => setTab(i)}
              style={{
                flex: 1,
                padding: '8px 4px',
                background: tab === i ? 'var(--bg-panel)' : 'transparent',
                border: 'none',
                borderRadius: 8,
                color: tab === i ? 'var(--text-primary)' : 'var(--text-muted)',
                fontSize: 11,
                fontWeight: tab === i ? 600 : 400,
                whiteSpace: 'nowrap',
                transition: 'all 0.2s ease',
                boxShadow: tab === i ? '0 1px 3px rgba(0,0,0,0.08)' : 'none',
                position: 'relative',
              }}
            >
              {i === 3 ? 'Clips' : t}
              {i === 3 && job.clips?.length > 0 && (
                <span style={{
                  marginLeft: 3, fontSize: 9, fontWeight: 700,
                  background: tab === i ? 'var(--accent-cyan)' : 'var(--accent-amber)',
                  color: 'var(--bg-base)',
                  padding: '1px 4px', borderRadius: 6,
                }}>
                  {job.clips.length}
                </span>
              )}
            </button>
          ))}
        </div>
      ) : (
        <div className="responsive-tabs" style={{ display: 'flex', gap: 0, borderBottom: '1px solid var(--border)', marginBottom: 24 }}>
          {TABS.map((t, i) => (
            <button
              key={t}
              onClick={() => setTab(i)}
              style={{
                padding: '10px 20px',
                background: 'none',
                border: 'none',
                borderBottom: tab === i ? '2px solid var(--accent-cyan)' : '2px solid transparent',
                color: tab === i ? 'var(--accent-cyan)' : 'var(--text-secondary)',
                fontSize: 13,
                fontWeight: tab === i ? 600 : 400,
                fontFamily: 'var(--font-mono)',
                whiteSpace: 'nowrap',
                flexShrink: 0,
              }}
            >
              {t}
              {i === 3 && job.clips?.length > 0 && (
                <span style={{ marginLeft: 6, fontSize: 10, background: 'var(--accent-amber)', color: 'var(--bg-base)', padding: '1px 5px', borderRadius: 8, fontWeight: 700 }}>
                  {job.clips.length}
                </span>
              )}
            </button>
          ))}
        </div>
      )}

      {/* Tab Content */}
      {tab === 0 && (
        <div>
          {job.summary ? (
            <div className="slide-in">
              <div style={{ background: 'var(--bg-panel)', border: '1px solid var(--border)', borderRadius: 'var(--radius-md)', padding: 20, marginBottom: 16, boxShadow: 'var(--shadow-sm)' }}>
                <h3 style={{ fontSize: 14, marginBottom: 12, color: 'var(--accent-cyan)' }}>Overview</h3>
                <p style={{ fontSize: 14, lineHeight: 1.6, color: 'var(--text-primary)' }}>{String(job.summary.overview || '')}</p>
              </div>

              <div style={{ display: 'flex', flexDirection: isMobile ? 'column' : 'row', gap: 16, flexWrap: 'wrap', marginBottom: 16 }}>
                <div style={{ background: 'var(--bg-panel)', border: '1px solid var(--border)', borderRadius: 'var(--radius-md)', padding: 16, flex: 1, minWidth: 200, boxShadow: 'var(--shadow-sm)' }}>
                  <h4 style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 8, textTransform: 'uppercase', letterSpacing: '0.06em' }}>Key Topics</h4>
                  <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                    {(job.summary.key_topics || []).map((t, i) => (
                      <span key={i} className="badge badge-cyan">{typeof t === 'string' ? t : String(t)}</span>
                    ))}
                  </div>
                </div>
                <div style={{ background: 'var(--bg-panel)', border: '1px solid var(--border)', borderRadius: 'var(--radius-md)', padding: 16, flex: 1, minWidth: 200, boxShadow: 'var(--shadow-sm)' }}>
                  <h4 style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 8, textTransform: 'uppercase', letterSpacing: '0.06em' }}>Details</h4>
                  <div style={{ fontSize: 13, color: 'var(--text-secondary)' }}>
                    <div style={{ marginBottom: 4 }}>Tone: <span className="badge badge-gray">{String(job.summary.tone || '')}</span></div>
                    <div style={{ marginBottom: 4 }}>Audience: {String(job.summary.estimated_audience || '')}</div>
                    <div>Category: <span className="badge badge-amber">{String(job.summary.content_category || '')}</span></div>
                  </div>
                </div>
              </div>
            </div>
          ) : (
            <div style={{ textAlign: 'center', padding: 48, color: 'var(--text-muted)' }}>
              {isProcessing ? 'Generating summary...' : 'No summary available'}
            </div>
          )}
        </div>
      )}

      {tab === 1 && (
        <div>
          {/* AI coverage strip */}
          {job.scene_coverage && (
            <div style={{ marginBottom: 16, padding: '10px 12px', background: 'var(--bg-panel)', border: '1px solid var(--border)', borderRadius: 'var(--radius-md)' }}>
              <div style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between', gap: 12, flexWrap: 'wrap' }}>
                <div style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
                  <strong style={{ color: 'var(--accent-cyan)' }}>{job.scene_coverage.ai_coverage_percent}%</strong> of video understood by AI
                  <span style={{ color: 'var(--text-muted)', margin: '0 6px' }}>·</span>
                  {job.scene_coverage.real_scenes} AI-described
                  {job.scene_coverage.synthetic_scenes > 0 && (
                    <>
                      <span style={{ color: 'var(--text-muted)', margin: '0 6px' }}>·</span>
                      {job.scene_coverage.synthetic_scenes} transcript-derived
                    </>
                  )}
                </div>
                <div style={{ fontSize: 10, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                  {job.scene_coverage.frames_per_minute} fpm
                </div>
              </div>
              <div style={{ height: 3, background: 'var(--bg-elevated)', marginTop: 6, borderRadius: 2, overflow: 'hidden' }}>
                <div style={{
                  width: `${Math.min(100, job.scene_coverage.ai_coverage_percent)}%`,
                  height: '100%',
                  background: job.scene_coverage.ai_coverage_percent > 70
                    ? 'var(--accent-cyan)'
                    : job.scene_coverage.ai_coverage_percent > 40
                      ? 'var(--accent-amber)'
                      : 'var(--text-muted)',
                  transition: 'width 0.3s',
                }} />
              </div>
            </div>
          )}

          {/* Add Scene form */}
          <AddSceneForm jobId={jobId} duration={job.duration} onAdded={fetchJob} />

          {/* Management hint */}
          {job.scenes?.length > 0 && (
            <div style={{ fontSize: 10, color: 'var(--text-muted)', marginBottom: 12, display: 'flex', alignItems: 'center', gap: 6 }}>
              <span style={{ color: 'var(--accent-cyan)' }}>{'\u270E'}</span> Hover over a scene card to edit or delete it
            </div>
          )}

          {/* Timeline bar */}
          {job.scenes?.length > 0 && job.duration > 0 && (
            <div style={{ marginBottom: 24 }}>
              <div style={{ display: 'flex', height: 40, gap: 1, alignItems: 'flex-end' }}>
                {job.scenes.map((scene, i) => (
                  <div
                    key={i}
                    onClick={() => handleSeek(scene.timestamp)}
                    style={{
                      flex: 1,
                      height: `${scene.importance_score * 10}%`,
                      background: scene.importance_score >= 8 ? 'var(--accent-amber)' : scene.importance_score >= 5 ? 'var(--accent-cyan)' : 'var(--border)',
                      cursor: 'pointer',
                      minWidth: 4,
                      transition: 'opacity 0.2s',
                    }}
                    title={`${formatDuration(scene.timestamp)} - Score: ${Number(scene.importance_score) || 0}/10`}
                  />
                ))}
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, color: 'var(--text-muted)', marginTop: 4 }}>
                <span>0:00</span>
                <span>{formatDuration(job.duration)}</span>
              </div>
            </div>
          )}

          <div className="responsive-grid" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(min(240px, 100%), 1fr))', gap: 16 }}>
            {(job.scenes || []).map((scene, i) => (
              <SceneCard key={i} scene={scene} sceneIndex={i} jobId={jobId} onClick={handleSeek} onUpdated={fetchJob} />
            ))}
          </div>
          {(!job.scenes || job.scenes.length === 0) && (
            <div style={{ textAlign: 'center', padding: 48, color: 'var(--text-muted)' }}>
              {isProcessing ? 'Analyzing scenes...' : 'No scenes analyzed'}
            </div>
          )}
        </div>
      )}

      {tab === 2 && (
        <div>
          {job.transcript?.length > 0 ? (
            <div style={{
              display: 'flex',
              gap: 20,
              flexDirection: isMobile ? 'column' : 'row',
              alignItems: 'flex-start',
            }}>
              {/* Left: Video player (sticky on desktop) */}
              <div style={{
                width: isMobile ? '100%' : '45%',
                maxWidth: isMobile ? '100%' : 560,
                flexShrink: 0,
                position: isMobile ? 'static' : 'sticky',
                top: 12,
                alignSelf: 'flex-start',
              }}>
                <VideoPlayer
                  src={videoSrc}
                  onTimeUpdate={setVideoCurrentTime}
                  sourceWidth={sourceDims.w}
                  sourceHeight={sourceDims.h}
                  scenes={job.scenes || []}
                  initialTime={transcriptInitTime}
                />
              </div>

              {/* Right: Transcript actions + editable transcript */}
              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{ display: 'flex', gap: 8, marginBottom: 16, flexWrap: 'wrap', alignItems: 'center' }}>
                  <button
                    onClick={async () => {
                      await fetchJob();
                      showToast('Transcript refreshed — subtitles updated for preview & export', 'success');
                    }}
                    style={{
                      padding: '6px 14px',
                      background: 'var(--accent-cyan-dim)',
                      color: 'var(--accent-cyan)',
                      border: '1px solid var(--accent-cyan)',
                      borderRadius: 'var(--radius-sm)',
                      fontSize: 12,
                      fontWeight: 600,
                      cursor: 'pointer',
                    }}
                  >
                    Update Subtitles
                  </button>
                  <a
                    href={`/api/jobs/${jobId}/transcript.srt`}
                    download
                    style={{
                      padding: '6px 14px',
                      background: 'var(--bg-elevated)',
                      color: 'var(--accent-cyan)',
                      border: '1px solid var(--border)',
                      borderRadius: 'var(--radius-sm)',
                      fontSize: 12,
                      fontWeight: 600,
                      textDecoration: 'none',
                      fontFamily: 'var(--font-mono)',
                    }}
                  >
                    &#x2B07; Download SRT (with speakers)
                  </a>
                  <a
                    href={`/api/jobs/${jobId}/transcript.srt?speakers=false`}
                    download
                    style={{
                      padding: '6px 14px',
                      background: 'var(--bg-elevated)',
                      color: 'var(--text-secondary)',
                      border: '1px solid var(--border)',
                      borderRadius: 'var(--radius-sm)',
                      fontSize: 12,
                      textDecoration: 'none',
                      fontFamily: 'var(--font-mono)',
                    }}
                  >
                    &#x2B07; SRT (no speakers)
                  </a>
                  {(() => {
                    const retranscribeRunning = retranscribing || job?.status === 'transcribing';
                    const pct = Math.max(0, Math.min(100, Number(job?.progress || 0)));
                    return (
                      <button
                        onClick={handleRetranscribe}
                        disabled={retranscribeRunning || job?.status === 'processing'}
                        title={
                          retranscribeRunning
                            ? (job?.progress_message || 'Running Whisper on the full audio…')
                            : 'Re-run Whisper on the full audio using the language and subtitle settings you picked at upload'
                        }
                        style={{
                          padding: '6px 14px',
                          marginLeft: 'auto',
                          background: retranscribeRunning
                            ? `linear-gradient(90deg, var(--amber-dim) 0%, var(--amber-dim) ${pct}%, var(--bg-elevated) ${pct}%, var(--bg-elevated) 100%)`
                            : 'var(--bg-elevated)',
                          color: 'var(--accent-amber)',
                          border: '1px solid var(--accent-amber)',
                          borderRadius: 'var(--radius-sm)',
                          fontSize: 12,
                          fontWeight: 600,
                          cursor: retranscribeRunning ? 'progress' : 'pointer',
                          opacity: retranscribeRunning ? 0.85 : 1,
                          minWidth: 160,
                        }}
                      >
                        {retranscribeRunning
                          ? `Retranscribing… ${pct}%`
                          : 'Retranscribe video'}
                      </button>
                    );
                  })()}
                </div>
                {/* Speaker Detection (post-processing diarization) */}
                <div style={{
                  padding: '12px 16px', background: 'var(--bg-panel)',
                  border: '1px solid var(--border)', borderRadius: 'var(--radius-md)',
                  marginBottom: 16,
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
                    <span style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-primary)' }}>
                      Speaker Detection
                    </span>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                      <label style={{ fontSize: 11, color: 'var(--text-muted)' }}>Speakers:</label>
                      <select
                        value={diarizeNumSpeakers}
                        onChange={(e) => setDiarizeNumSpeakers(parseInt(e.target.value))}
                        style={{
                          padding: '4px 8px', fontSize: 12, background: 'var(--bg-base)',
                          border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)',
                          color: 'var(--text-primary)',
                        }}
                      >
                        <option value={0}>Auto-detect</option>
                        {[1,2,3,4,5,6,7,8,9,10].map(n => (
                          <option key={n} value={n}>{n} speaker{n > 1 ? 's' : ''}</option>
                        ))}
                      </select>
                    </div>
                    <button
                      onClick={handleRunDiarization}
                      disabled={diarizeLoading}
                      style={{
                        padding: '6px 16px', fontSize: 12, fontWeight: 600,
                        background: 'var(--accent-cyan)', color: 'var(--bg-base)',
                        border: 'none', borderRadius: 'var(--radius-sm)',
                        opacity: diarizeLoading ? 0.5 : 1, cursor: diarizeLoading ? 'default' : 'pointer',
                      }}
                    >
                      {diarizeLoading ? 'Detecting...' : 'Run Speaker Detection'}
                    </button>
                    {diarizeResult && (
                      <span style={{ fontSize: 11, color: 'var(--success)' }}>
                        {diarizeResult}
                      </span>
                    )}
                  </div>
                  <p style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 6, marginBottom: 0 }}>
                    Analyzes the audio to identify who is speaking. Specify the exact number of speakers for best accuracy.
                  </p>
                </div>

                <TranscriptViewer
                  transcript={job.translated_transcript?.length ? job.translated_transcript : (job.transcript || [])}
                  currentTime={videoCurrentTime}
                  speakerColors={clipSettings?.speakerColors}
                  onSpeakerColorChanged={handleSpeakerColorChanged}
                  onSpeakerAdded={handleSpeakerAdded}
                  onSeek={handleSeek}
                  jobId={jobId}
                  onSpeakerRenamed={fetchJob}
                  onTranscriptUpdated={fetchJob}
                />
              </div>
            </div>
          ) : (
            <div style={{ textAlign: 'center', padding: 48, color: 'var(--text-muted)' }}>
              {isProcessing ? 'Transcribing audio...' : 'No transcript available'}
            </div>
          )}
        </div>
      )}

      {tab === 3 && (
        <div>
          {/* Local AI banner */}
          {Object.values(job.provider_used || {}).some((v) => v === 'ollama') && (
            <div style={{ padding: '10px 16px', background: 'var(--amber-dim)', border: '1px solid var(--accent-amber)', marginBottom: 16, fontSize: 13, color: 'var(--accent-amber)', borderRadius: 'var(--radius-sm)' }}>
              Local AI - analysis is free and private. Results may vary from cloud quality.
            </div>
          )}

          {/* Clip Discovery bar — same relaxed gate as the sidebar
              below so users with empty scenes still see the
              "Generate Clips" controls. The clip-detector falls back
              to transcript-only when no scenes are present. */}
          {(job.transcript?.length > 0 || job.summary || job.duration > 0) && (
            <div style={{
              display: 'flex',
              gap: isMobile ? 8 : 16,
              marginBottom: 16,
              padding: isMobile ? '12px' : '14px 18px',
              background: 'var(--bg-panel)',
              border: '1px solid var(--border)',
              borderRadius: isMobile ? 14 : 'var(--radius-md)',
              alignItems: 'flex-end',
              flexWrap: 'wrap',
            }}>
              <div style={{ flex: isMobile ? '1 1 100%' : '0 0 auto', opacity: isGeneratingClips ? 0.5 : 1, pointerEvents: isGeneratingClips ? 'none' : 'auto', transition: 'opacity 0.3s' }}>
                {!isMobile && (
                  <div style={{
                    fontSize: 11,
                    fontFamily: 'var(--font-mono)',
                    color: 'var(--text-primary)',
                    textTransform: 'uppercase',
                    letterSpacing: '0.05em',
                    marginBottom: 8,
                  }}>
                    Find Viral Moments
                  </div>
                )}
                <div style={{ display: 'flex', gap: 16, alignItems: 'center', flexWrap: 'wrap' }}>
                  <label style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 12, color: 'var(--text-secondary)' }}>
                    Clips
                    <input
                      type="range"
                      min="1"
                      max="100"
                      value={Math.min(100, genSettings.clipCount || 12)}
                      onChange={(e) => updateGen('clipCount', parseInt(e.target.value))}
                      disabled={isGeneratingClips}
                      style={{ width: 80, accentColor: 'var(--accent-cyan)' }}
                    />
                    <input
                      type="number"
                      min="1"
                      max="200"
                      value={genSettings.clipCount || 12}
                      onChange={(e) => {
                        const val = parseInt(e.target.value);
                        if (!isNaN(val) && val >= 1 && val <= 200) updateGen('clipCount', val);
                      }}
                      disabled={isGeneratingClips}
                      style={{
                        width: 48, padding: '3px 4px', fontSize: 13,
                        fontFamily: 'var(--font-mono)', textAlign: 'center',
                        background: 'var(--bg-elevated)', color: isGeneratingClips ? 'var(--text-muted)' : 'var(--accent-cyan)',
                        border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)',
                        outline: 'none',
                      }}
                    />
                  </label>
                  <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12, color: 'var(--text-secondary)' }}>
                    Min
                    <input
                      type="text"
                      value={minText}
                      onChange={(e) => setMinText(e.target.value)}
                      onBlur={() => {
                        const val = parseDuration(minText);
                        if (val !== null && val >= 1 && val <= 7200) {
                          updateGen('minDuration', val);
                          if (val >= genSettings.maxDuration) {
                            const newMax = Math.min(7200, val + 30);
                            updateGen('maxDuration', newMax);
                            setMaxText(formatDurationInput(newMax));
                          }
                          setMinText(formatDurationInput(val));
                        } else {
                          setMinText(formatDurationInput(genSettings.minDuration));
                        }
                      }}
                      onKeyDown={(e) => { if (e.key === 'Enter') e.target.blur(); }}
                      disabled={isGeneratingClips}
                      style={{
                        width: 56,
                        padding: '4px 6px',
                        fontSize: 12,
                        fontFamily: 'var(--font-mono)',
                        background: 'var(--bg-elevated)',
                        color: isGeneratingClips ? 'var(--text-muted)' : 'var(--accent-cyan)',
                        border: '1px solid var(--border)',
                        borderRadius: 'var(--radius-sm)',
                        outline: 'none',
                        textAlign: 'center',
                      }}
                    />
                  </label>
                  <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12, color: 'var(--text-secondary)' }}>
                    Max
                    <input
                      type="text"
                      value={maxText}
                      onChange={(e) => setMaxText(e.target.value)}
                      onBlur={() => {
                        const val = parseDuration(maxText);
                        if (val !== null && val >= 1 && val <= 7200) {
                          updateGen('maxDuration', val);
                          if (val <= genSettings.minDuration) {
                            const newMin = Math.max(1, val - 30);
                            updateGen('minDuration', newMin);
                            setMinText(formatDurationInput(newMin));
                          }
                          setMaxText(formatDurationInput(val));
                        } else {
                          setMaxText(formatDurationInput(genSettings.maxDuration));
                        }
                      }}
                      onKeyDown={(e) => { if (e.key === 'Enter') e.target.blur(); }}
                      disabled={isGeneratingClips}
                      style={{
                        width: 56,
                        padding: '4px 6px',
                        fontSize: 12,
                        fontFamily: 'var(--font-mono)',
                        background: 'var(--bg-elevated)',
                        color: isGeneratingClips ? 'var(--text-muted)' : 'var(--accent-cyan)',
                        border: '1px solid var(--border)',
                        borderRadius: 'var(--radius-sm)',
                        outline: 'none',
                        textAlign: 'center',
                      }}
                    />
                  </label>
                </div>
                {/* Viral Score Range — Bug 1 + 9 in the clip-focus audit: always
                    visible, even in focus mode. The label swaps to "Virality"
                    when focus is on so the user understands this is the
                    virality score (not relevance). */}
                {true && (
                  <div style={{ display: 'flex', gap: 16, alignItems: 'center', flexWrap: 'wrap', marginTop: 8 }}
                       role="group" aria-label="Viral score range filter">
                    <span style={{ fontSize: 12, color: 'var(--text-secondary)', whiteSpace: 'nowrap' }}>
                      {clipFocusEnabled ? 'Virality Score' : 'Viral Score'}
                    </span>
                    <label style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: 12, color: 'var(--text-secondary)' }}>
                      Min
                      <input
                        type="number"
                        min="0"
                        max="100"
                        value={genSettings.viralScoreMin ?? 0}
                        onChange={(e) => {
                          const val = parseInt(e.target.value);
                          if (!isNaN(val) && val >= 0 && val <= 100) {
                            updateGen('viralScoreMin', val);
                            if (val > (genSettings.viralScoreMax ?? 100)) updateGen('viralScoreMax', val);
                          }
                        }}
                        disabled={isGeneratingClips}
                        style={{
                          width: 44, padding: '3px 4px', fontSize: 12,
                          fontFamily: 'var(--font-mono)', textAlign: 'center',
                          background: 'var(--bg-elevated)', color: isGeneratingClips ? 'var(--text-muted)' : 'var(--accent-cyan)',
                          border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)',
                          outline: 'none',
                        }}
                      />
                    </label>
                    <input
                      type="range"
                      min="0"
                      max="100"
                      value={genSettings.viralScoreMin ?? 0}
                      onChange={(e) => {
                        const val = parseInt(e.target.value);
                        updateGen('viralScoreMin', val);
                        if (val > (genSettings.viralScoreMax ?? 100)) updateGen('viralScoreMax', val);
                      }}
                      disabled={isGeneratingClips}
                      style={{ width: 80, accentColor: 'var(--accent-cyan)' }}
                    />
                    <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>&ndash;</span>
                    <input
                      type="range"
                      min="0"
                      max="100"
                      value={genSettings.viralScoreMax ?? 100}
                      onChange={(e) => {
                        const val = parseInt(e.target.value);
                        updateGen('viralScoreMax', val);
                        if (val < (genSettings.viralScoreMin ?? 0)) updateGen('viralScoreMin', val);
                      }}
                      disabled={isGeneratingClips}
                      style={{ width: 80, accentColor: 'var(--accent-cyan)' }}
                    />
                    <label style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: 12, color: 'var(--text-secondary)' }}>
                      Max
                      <input
                        type="number"
                        min="0"
                        max="100"
                        value={genSettings.viralScoreMax ?? 100}
                        onChange={(e) => {
                          const val = parseInt(e.target.value);
                          if (!isNaN(val) && val >= 0 && val <= 100) {
                            updateGen('viralScoreMax', val);
                            if (val < (genSettings.viralScoreMin ?? 0)) updateGen('viralScoreMin', val);
                          }
                        }}
                        disabled={isGeneratingClips}
                        style={{
                          width: 44, padding: '3px 4px', fontSize: 12,
                          fontFamily: 'var(--font-mono)', textAlign: 'center',
                          background: 'var(--bg-elevated)', color: isGeneratingClips ? 'var(--text-muted)' : 'var(--accent-cyan)',
                          border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)',
                          outline: 'none',
                        }}
                      />
                    </label>
                    {(genSettings.viralScoreMin > 0 || genSettings.viralScoreMax < 100) && (
                      <span style={{ fontSize: 10, color: 'var(--accent-amber)' }}>
                        Only clips scoring {genSettings.viralScoreMin}-{genSettings.viralScoreMax} will be kept
                      </span>
                    )}
                    {/* Score-range presets (Enhancement 8) */}
                    <div style={{ display: 'flex', gap: 4, marginLeft: 'auto' }}>
                      {[
                        { label: 'Top 10%', min: 90, max: 100 },
                        { label: 'Strong', min: 70, max: 100 },
                        { label: 'All', min: 0, max: 100 },
                      ].map((p) => {
                        const active = genSettings.viralScoreMin === p.min && genSettings.viralScoreMax === p.max;
                        return (
                          <button
                            key={p.label}
                            onClick={() => setGenSettings((prev) => ({ ...prev, viralScoreMin: p.min, viralScoreMax: p.max }))}
                            disabled={isGeneratingClips}
                            style={{
                              padding: '2px 8px', fontSize: 10, fontWeight: 600,
                              background: active ? 'var(--accent-cyan)' : 'transparent',
                              color: active ? 'var(--bg-base)' : 'var(--text-secondary)',
                              border: `1px solid ${active ? 'var(--accent-cyan)' : 'var(--border)'}`,
                              borderRadius: 'var(--radius-sm)',
                              cursor: isGeneratingClips ? 'not-allowed' : 'pointer',
                            }}
                          >
                            {p.label}
                          </button>
                        );
                      })}
                    </div>
                  </div>
                )}
                {/* Clip Focus toggle — Enhancement 12: role=switch a11y */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: 6, marginTop: 10, width: '100%' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <label
                      htmlFor="clip-focus-switch"
                      style={{ fontSize: 12, color: 'var(--text-secondary)' }}
                    >
                      Clip Focus
                    </label>
                    <button
                      id="clip-focus-switch"
                      role="switch"
                      aria-checked={clipFocusEnabled}
                      aria-label="Enable clip focus mode"
                      onClick={() => { if (!isGeneratingClips) setClipFocusEnabled(!clipFocusEnabled); }}
                      disabled={isGeneratingClips}
                      style={{
                        width: 42, height: 26, borderRadius: 13, border: 'none',
                        cursor: isGeneratingClips ? 'not-allowed' : 'pointer',
                        background: clipFocusEnabled ? 'var(--success)' : 'var(--border)',
                        position: 'relative', transition: 'background 0.25s ease', flexShrink: 0,
                        opacity: isGeneratingClips ? 0.5 : 1,
                      }}
                    >
                      <div style={{
                        width: 20, height: 20, borderRadius: '50%', background: 'white',
                        position: 'absolute', top: 3,
                        left: clipFocusEnabled ? 19 : 3,
                        transition: 'left 0.25s cubic-bezier(0.4, 0, 0.2, 1)',
                        boxShadow: '0 1px 3px rgba(0,0,0,0.2)',
                      }} />
                    </button>
                  </div>
                  {clipFocusEnabled && (
                    <div style={{ position: 'relative', width: '100%' }}>
                      <textarea
                        placeholder={isMobile ? 'e.g. "funny cooking moments"' : "Try compound queries for best results:\n• \"funny cooking moments\"\n• \"emotional reveals\"\n• comma or newline-separated multi-query: \"origin story, biggest mistake\""}
                        value={clipFocusText}
                        onChange={(e) => setClipFocusText(e.target.value)}
                        onFocus={() => { if ((genSettings.focusHistory || []).length > 0 && !clipFocusText) setFocusHistoryOpen(true); }}
                        onBlur={() => setTimeout(() => setFocusHistoryOpen(false), 150)}
                        onKeyDown={(e) => {
                          // Enhancement 3 — Cmd/Ctrl+Enter submits
                          if ((e.metaKey || e.ctrlKey) && e.key === 'Enter' && clipFocusText.trim() && !isGeneratingClips) {
                            e.preventDefault();
                            handleGenerateClips();
                          }
                        }}
                        rows={isMobile ? 2 : 3}
                        disabled={isGeneratingClips}
                        aria-label="Clip focus query"
                        aria-describedby="clip-focus-helper"
                        maxLength={300}
                        style={{
                          width: '100%', padding: '8px 10px', fontSize: 13,
                          background: 'var(--bg-elevated)',
                          color: isGeneratingClips ? 'var(--text-muted)' : 'var(--text-primary)',
                          border: `1px solid ${isGeneratingClips ? 'var(--border)' : 'var(--success)'}`,
                          borderRadius: 'var(--radius-md)', outline: 'none',
                          resize: 'none', minHeight: isMobile ? 44 : 60, lineHeight: 1.4,
                          opacity: isGeneratingClips ? 0.5 : 1,
                        }}
                      />
                      {/* Character counter */}
                      {clipFocusText.length > 0 && (
                        <span style={{
                          position: 'absolute', right: 8, bottom: 4,
                          fontSize: 9, color: clipFocusText.length > 250 ? 'var(--accent-amber)' : 'var(--text-muted)',
                          pointerEvents: 'none', fontFamily: 'var(--font-mono)',
                        }}>
                          {clipFocusText.length}/300
                        </span>
                      )}
                      {/* History dropdown (Enhancement 3) */}
                      {focusHistoryOpen && (genSettings.focusHistory || []).length > 0 && (
                        <div style={{
                          position: 'absolute', top: '100%', left: 0, right: 0, marginTop: 4,
                          background: 'var(--bg-elevated)',
                          border: '1px solid var(--border)',
                          borderRadius: 'var(--radius-md)',
                          boxShadow: 'var(--shadow-md)',
                          zIndex: 40, maxHeight: 200, overflowY: 'auto',
                        }}>
                          <div style={{ padding: '4px 10px', fontSize: 9, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Recent queries</div>
                          {(genSettings.focusHistory || []).map((h, i) => (
                            <div
                              key={i}
                              onMouseDown={(e) => { e.preventDefault(); setClipFocusText(h); setFocusHistoryOpen(false); }}
                              style={{
                                padding: '6px 10px', fontSize: 12, cursor: 'pointer',
                                borderTop: '1px solid var(--border-subtle, rgba(255,255,255,0.04))',
                                color: 'var(--text-primary)',
                              }}
                              onMouseEnter={(e) => { e.currentTarget.style.background = 'var(--bg-panel)'; }}
                              onMouseLeave={(e) => { e.currentTarget.style.background = 'transparent'; }}
                            >
                              {h}
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  )}
                  {/* Example chips — shown when the textarea is empty (Enhancement 3) */}
                  {clipFocusEnabled && !clipFocusText.trim() && (
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                      {FOCUS_EXAMPLE_CHIPS.map((chip) => (
                        <button
                          key={chip}
                          onClick={() => setClipFocusText(chip)}
                          disabled={isGeneratingClips}
                          style={{
                            padding: '3px 10px', fontSize: 10,
                            background: 'var(--bg-elevated)',
                            color: 'var(--text-secondary)',
                            border: '1px solid var(--border)',
                            borderRadius: 999,
                            cursor: isGeneratingClips ? 'not-allowed' : 'pointer',
                          }}
                        >
                          {chip}
                        </button>
                      ))}
                    </div>
                  )}
                  {/* Min Relevance slider + presets (Bug 1 + Enhancement 8) */}
                  {clipFocusEnabled && (
                    <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap', marginTop: 4 }}
                         role="group" aria-label="Minimum focus relevance filter">
                      <span style={{ fontSize: 11, color: 'var(--text-secondary)', whiteSpace: 'nowrap' }}>Min Relevance</span>
                      <input
                        type="range"
                        min="0"
                        max="100"
                        value={genSettings.minRelevance ?? 50}
                        aria-valuemin={0}
                        aria-valuemax={100}
                        aria-valuenow={genSettings.minRelevance ?? 50}
                        aria-valuetext={`minimum relevance ${genSettings.minRelevance ?? 50}`}
                        onChange={(e) => setGenSettings((prev) => ({ ...prev, minRelevance: parseInt(e.target.value) }))}
                        disabled={isGeneratingClips}
                        style={{ width: 120, accentColor: 'var(--success)' }}
                      />
                      <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--success)', minWidth: 28, textAlign: 'center' }}>
                        {genSettings.minRelevance ?? 50}
                      </span>
                      <div style={{ display: 'flex', gap: 4 }}>
                        {[
                          { label: 'Strong', v: 80 },
                          { label: 'Strong+Mod', v: 50 },
                          { label: 'All', v: 0 },
                        ].map((p) => {
                          const active = (genSettings.minRelevance ?? 50) === p.v;
                          return (
                            <button
                              key={p.label}
                              onClick={() => setGenSettings((prev) => ({ ...prev, minRelevance: p.v }))}
                              disabled={isGeneratingClips}
                              style={{
                                padding: '2px 8px', fontSize: 10, fontWeight: 600,
                                background: active ? 'var(--success)' : 'transparent',
                                color: active ? 'var(--bg-base)' : 'var(--text-secondary)',
                                border: `1px solid ${active ? 'var(--success)' : 'var(--border)'}`,
                                borderRadius: 'var(--radius-sm)',
                                cursor: isGeneratingClips ? 'not-allowed' : 'pointer',
                              }}
                            >
                              {p.label}
                            </button>
                          );
                        })}
                      </div>
                    </div>
                  )}
                  {clipFocusEnabled && !isMobile && (
                    <span id="clip-focus-helper" style={{ fontSize: 10, color: 'var(--text-muted)', lineHeight: 1.3 }}>
                      AI finds clips matching your topic with semantic expansion and relevance scoring.
                      Use commas or newlines for multi-query OR mode. <kbd style={{ fontFamily: 'var(--font-mono)' }}>Cmd/Ctrl+Enter</kbd> to submit.
                    </span>
                  )}
                </div>
                {/* Generate / Cancel row + live filter preview (Enhancement 2) */}
                <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginTop: 12, width: '100%', flexWrap: 'wrap' }}>
                  <button
                    onClick={() => handleGenerateClips()}
                    disabled={isGeneratingClips || (clipFocusEnabled && !clipFocusText.trim())}
                    style={{
                      padding: '8px 20px',
                      background: isGeneratingClips ? 'var(--accent-amber)' : clipFocusEnabled ? 'var(--success)' : 'var(--accent-cyan)',
                      color: 'var(--bg-base)',
                      border: 'none',
                      borderRadius: 'var(--radius-sm)',
                      fontSize: 13,
                      fontWeight: 700,
                      cursor: isGeneratingClips ? 'wait' : 'pointer',
                      whiteSpace: 'nowrap',
                      opacity: (clipFocusEnabled && !clipFocusText.trim()) ? 0.5 : 1,
                    }}
                  >
                    {isGeneratingClips ? 'Generating...' : clipFocusEnabled ? 'Find Focused Clips' : 'Generate Clips'}
                  </button>
                  {isGeneratingClips && (
                    <button
                      onClick={handleCancelGenerate}
                      aria-label="Cancel clip generation"
                      style={{
                        padding: '8px 16px',
                        background: 'transparent',
                        color: 'var(--danger, #ff3b30)',
                        border: '1px solid var(--danger, #ff3b30)',
                        borderRadius: 'var(--radius-sm)',
                        fontSize: 12, fontWeight: 600, cursor: 'pointer',
                      }}
                    >
                      Cancel
                    </button>
                  )}
                  {/* Live filter preview (Enhancement 2) — recompute from existing job.clips */}
                  {!isGeneratingClips && job?.clips && job.clips.length > 0 && (() => {
                    const srcClips = job.clips || [];
                    const minR = clipFocusEnabled ? (genSettings.minRelevance ?? 0) : 0;
                    const vmin = genSettings.viralScoreMin ?? 0;
                    const vmax = genSettings.viralScoreMax ?? 100;
                    const matching = srcClips.filter((c) => {
                      const virality = (clipFocusEnabled && c.viral_score_composite != null) ? c.viral_score_composite : c.viral_score;
                      const relevance = c.focus_relevance != null ? c.focus_relevance : c.viral_score;
                      if (virality < vmin || virality > vmax) return false;
                      if (clipFocusEnabled && minR > 0 && relevance < minR) return false;
                      return true;
                    });
                    const changed = matching.length !== srcClips.length;
                    if (!changed) return null;
                    return (
                      <span style={{ fontSize: 10, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                        Of {srcClips.length} existing: {matching.length} would pass
                      </span>
                    );
                  })()}
                  <span
                    onClick={() => navigate('/settings?tab=prompts&section=viral-algorithm')}
                    style={{
                      fontSize: 10,
                      color: 'var(--text-muted)',
                      cursor: 'pointer',
                      textDecoration: 'underline',
                      textDecorationStyle: 'dashed',
                      marginLeft: 'auto',
                    }}
                    title="Customize the AI prompt that controls how viral clips are detected and scored"
                  >
                    Edit Viral Algorithm
                  </span>
                </div>
              </div>
            </div>
          )}

          {/* Sidebar + Clips layout */}
          <div className="clip-panel-layout" style={{ display: 'flex', gap: 20, alignItems: 'flex-start' }}>

            {/* Left: Settings Sidebar (sticky).
                Render whenever the job has SOMETHING the user can act on
                — even if scene analysis returned empty. Previously the
                whole sidebar (ClipSettings + Export Full Video) was
                gated on ``scenes.length > 0`` AND ``summary``, which
                meant any pipeline that returned no scenes (vision
                provider failure, OOM, wrong model id) silently hid the
                clip settings entirely.
                The sidebar still relies on ``job`` being loaded; we
                show as soon as transcription finished OR a summary was
                generated. */}
            {(job.transcript?.length > 0 || job.summary || job.duration > 0) && (
              <div
                className="clip-settings-sidebar"
                style={{
                  width: isMobile ? '100%' : 320,
                  flexShrink: 0,
                  position: isMobile ? 'static' : 'sticky',
                  top: 16,
                  maxHeight: 'calc(100vh - 120px)',
                  overflowY: 'auto',
                }}
              >
                <EditorErrorBoundary name="Clip settings" compact>
                  <ClipSettingsPanel
                    speakers={speakers}
                    speakerNames={job.speaker_names}
                    videoResolution={job.resolution}
                    onSettingsChange={setClipSettings}
                    onApplySettings={handleApplyClipSettings}
                    onPresetsLoaded={setClipPresets}
                    serverSettings={job.subtitle_settings}
                    parentSettings={clipSettings}
                  />
                </EditorErrorBoundary>

                {/* Export Full Video */}
                <div style={{
                  marginTop: 16,
                  padding: 16,
                  background: 'var(--bg-panel)',
                  border: '1px solid var(--border)',
                  borderRadius: 'var(--radius-md)',
                  boxShadow: 'var(--shadow-sm)',
                }}>
                  <h4 style={{
                    fontSize: 12,
                    color: 'var(--text-muted)',
                    marginBottom: 10,
                    textTransform: 'uppercase',
                    letterSpacing: '0.06em',
                    fontFamily: 'var(--font-mono)',
                  }}>
                    Export Full Video
                  </h4>
                  <p style={{ fontSize: 11, color: 'var(--text-secondary)', marginBottom: 12, lineHeight: 1.4 }}>
                    Export the entire video with the clip settings above applied — aspect ratio, subtitles{job.scenes?.length > 0 ? ', and Intelligent Dynamic Subject Tracking' : ''}.
                  </p>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 2 }}>
                      <span style={{ fontSize: 11, color: 'var(--text-muted)', minWidth: 50 }}>Quality</span>
                      <div style={{ display: 'flex', gap: 0, borderRadius: 'var(--radius-sm)', overflow: 'hidden', border: '1px solid var(--border)' }}>
                        {[
                          { value: '720p', label: '720p' },
                          { value: '1080p', label: '1080p' },
                          { value: '4k', label: '4K' },
                        ].map((q) => (
                          <button
                            key={q.value}
                            onClick={() => setClipSettings(prev => ({ ...prev, exportQuality: q.value }))}
                            style={{
                              padding: '4px 12px', fontSize: 11, fontWeight: 600,
                              background: (clipSettings?.exportQuality || '1080p') === q.value ? 'var(--accent-amber)' : 'transparent',
                              color: (clipSettings?.exportQuality || '1080p') === q.value ? 'var(--bg-base)' : 'var(--text-secondary)',
                              border: 'none', cursor: 'pointer',
                              borderRight: q.value !== '4k' ? '1px solid var(--border)' : 'none',
                              transition: 'background 0.15s, color 0.15s',
                            }}
                          >
                            {q.label}
                          </button>
                        ))}
                      </div>
                    </div>
                    <div style={{ display: 'flex', gap: 8 }}>
                      <button
                        onClick={() => {
                          if (!job?.duration) return;
                          handleClipPreview({
                            id: 0,
                            start_time: 0,
                            end_time: job.duration,
                            title: job.filename || 'Full Video',
                          });
                        }}
                        style={{
                          flex: 1,
                          padding: '10px 16px',
                          background: 'transparent',
                          color: 'var(--accent-amber)',
                          border: '1px solid var(--accent-amber)',
                          borderRadius: 'var(--radius-sm)',
                          fontSize: 13,
                          fontWeight: 700,
                          cursor: 'pointer',
                          transition: 'background 0.2s, color 0.2s',
                        }}
                      >
                        Preview
                      </button>
                      <button
                        onClick={handleExportFullVideo}
                        disabled={fullVideoExporting}
                        style={{
                          flex: 2,
                          padding: '10px 16px',
                          background: fullVideoExporting ? 'var(--bg-elevated)' : 'var(--accent-amber)',
                          color: fullVideoExporting ? 'var(--text-muted)' : 'var(--bg-base)',
                          border: 'none',
                          borderRadius: 'var(--radius-sm)',
                          fontSize: 13,
                          fontWeight: 700,
                          cursor: fullVideoExporting ? 'wait' : 'pointer',
                          transition: 'background 0.2s',
                        }}
                      >
                        {fullVideoExporting ? 'Starting Export...' : `Export (${clipSettings?.exportQuality || '1080p'})`}
                      </button>
                    </div>
                    <div style={{ fontSize: 10, color: 'var(--text-muted)', lineHeight: 1.4 }}>
                      {clipSettings?.aspectRatio ? (
                        <span>Aspect ratio: <strong style={{ color: 'var(--text-secondary)' }}>{String(clipSettings.aspectRatio || '')}</strong></span>
                      ) : (
                        <span>Original aspect ratio</span>
                      )}
                      {clipSettings?.subtitlesEnabled && (
                        <span> &bull; Subtitles enabled</span>
                      )}
                      {job.scenes?.length > 0 && clipSettings?.aspectRatio && (
                        <span> &bull; Subject tracking ({job.scenes.length} scenes)</span>
                      )}
                    </div>
                  </div>
                </div>

                {/* QA Validation Panel */}
                <div style={{
                  marginTop: 16,
                  padding: 16,
                  background: 'var(--bg-panel)',
                  border: `1px solid ${qaResult?.overall === 'pass' ? 'var(--success)' : qaResult?.overall === 'warn' ? 'var(--accent-amber)' : qaResult?.overall === 'fail' ? 'var(--danger)' : 'var(--border)'}`,
                  borderRadius: 'var(--radius-md)',
                  boxShadow: 'var(--shadow-sm)',
                }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
                    <h4 style={{
                      fontSize: 12,
                      color: 'var(--text-muted)',
                      textTransform: 'uppercase',
                      letterSpacing: '0.06em',
                      fontFamily: 'var(--font-mono)',
                      margin: 0,
                    }}>
                      Pipeline QA
                    </h4>
                    <button
                      onClick={() => { setQaResult(null); runQaValidation(); }}
                      disabled={qaLoading}
                      style={{
                        padding: '4px 10px',
                        fontSize: 10,
                        fontWeight: 600,
                        background: 'var(--bg-elevated)',
                        color: 'var(--text-secondary)',
                        border: '1px solid var(--border)',
                        borderRadius: 'var(--radius-sm)',
                        cursor: qaLoading ? 'wait' : 'pointer',
                      }}
                    >
                      {qaLoading ? 'Checking...' : 'Re-run'}
                    </button>
                  </div>

                  {qaResult ? (
                    <div>
                      <div style={{
                        display: 'flex', alignItems: 'center', gap: 8, marginBottom: 10,
                        padding: '6px 10px',
                        background: qaResult.overall === 'pass' ? 'rgba(16, 185, 129, 0.1)' : qaResult.overall === 'warn' ? 'rgba(245, 158, 11, 0.1)' : 'rgba(239, 68, 68, 0.1)',
                        borderRadius: 'var(--radius-sm)',
                      }}>
                        <span style={{
                          width: 8, height: 8, borderRadius: '50%',
                          background: qaResult.overall === 'pass' ? 'var(--success)' : qaResult.overall === 'warn' ? 'var(--accent-amber)' : 'var(--danger)',
                        }} />
                        <span style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-primary)' }}>
                          {qaResult.overall === 'pass' ? 'All Checks Passed' : qaResult.overall === 'warn' ? 'Passed with Warnings' : 'Issues Detected'}
                        </span>
                        <span style={{ fontSize: 10, color: 'var(--text-muted)', marginLeft: 'auto' }}>
                          {String(qaResult.summary || '')}
                        </span>
                      </div>
                      <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                        {qaResult.checks?.map((check, i) => (
                          <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 11 }}>
                            <span style={{
                              width: 6, height: 6, borderRadius: '50%', flexShrink: 0,
                              background: check.status === 'pass' ? 'var(--success)' : check.status === 'warn' ? 'var(--accent-amber)' : check.status === 'fail' ? 'var(--danger)' : 'var(--text-muted)',
                            }} />
                            <span style={{ color: 'var(--text-secondary)', fontWeight: 600, minWidth: 100 }}>{String(check.name || '')}</span>
                            <span style={{ color: 'var(--text-muted)', flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{String(check.detail || '')}</span>
                          </div>
                        ))}
                      </div>
                      {qaResult.errors?.length > 0 && (
                        <div style={{ marginTop: 8, padding: '6px 8px', background: 'rgba(239, 68, 68, 0.08)', borderRadius: 'var(--radius-sm)' }}>
                          {qaResult.errors.map((e, i) => (
                            <div key={i} style={{ fontSize: 10, color: 'var(--danger)', lineHeight: 1.4 }}>{String(e || '')}</div>
                          ))}
                        </div>
                      )}
                      {qaResult.warnings?.length > 0 && (
                        <div style={{ marginTop: 6, padding: '6px 8px', background: 'rgba(245, 158, 11, 0.08)', borderRadius: 'var(--radius-sm)' }}>
                          {qaResult.warnings.slice(0, 5).map((w, i) => (
                            <div key={i} style={{ fontSize: 10, color: 'var(--accent-amber)', lineHeight: 1.4 }}>{String(w || '')}</div>
                          ))}
                          {qaResult.warnings.length > 5 && (
                            <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>+{qaResult.warnings.length - 5} more warnings</div>
                          )}
                        </div>
                      )}
                    </div>
                  ) : (
                    <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                      {qaLoading ? 'Running validation checks...' : 'QA validation runs automatically when analysis completes'}
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Right: Clips area */}
            <div style={{ flex: 1, minWidth: 0 }}>
              {/* Search Bar */}
              {job.clips?.length > 0 && (
                <div style={{ position: 'relative', marginBottom: 12 }}>
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
                    value={clipSearchQuery}
                    onChange={(e) => setClipSearchQuery(e.target.value)}
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
                  {clipSearchQuery && (
                    <button
                      onClick={() => setClipSearchQuery('')}
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

              {/* Filters */}
              {job.clips?.length > 0 && (
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
                  {analysisClipTypes.length > 1 && (
                    <select
                      value={filters.type}
                      onChange={(e) => setFilters((f) => ({ ...f, type: e.target.value }))}
                      style={{ padding: '6px 10px', borderRadius: 'var(--radius-sm)', fontSize: 12, background: 'var(--bg-elevated)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
                    >
                      <option value="all">All Types</option>
                      {analysisClipTypes.map((t) => (
                        <option key={t} value={t}>{t.charAt(0).toUpperCase() + t.slice(1)}</option>
                      ))}
                    </select>
                  )}
                  <select
                    value={filters.sort}
                    onChange={(e) => setFilters((f) => ({ ...f, sort: e.target.value }))}
                    style={{ padding: '6px 10px', borderRadius: 'var(--radius-sm)', fontSize: 12, background: 'var(--bg-elevated)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
                  >
                    <option value="newest">Newest First</option>
                    <option value="viral_score">Score: High to Low</option>
                    <option value="score_low">Score: Low to High</option>
                    <option value="duration">Duration: Longest</option>
                    <option value="duration_short">Duration: Shortest</option>
                    <option value="timestamp">Timestamp</option>
                    <option value="title_az">Title: A-Z</option>
                  </select>

                  {/* Select All / Deselect All */}
                  <button
                    onClick={() => {
                      const allIds = filteredClips.map((c) => c.id);
                      const allSelected = allIds.length > 0 && allIds.every((id) => selectedClips.has(id));
                      if (allSelected) {
                        setSelectedClips(new Set());
                      } else {
                        setSelectedClips(new Set(allIds));
                      }
                    }}
                    style={{
                      padding: '6px 12px',
                      background: 'var(--bg-elevated)',
                      color: 'var(--text-secondary)',
                      border: '1px solid var(--border)',
                      borderRadius: 'var(--radius-sm)',
                      fontSize: 11,
                      fontWeight: 600,
                      cursor: 'pointer',
                      whiteSpace: 'nowrap',
                    }}
                  >
                    {filteredClips.length > 0 && filteredClips.every((c) => selectedClips.has(c.id))
                      ? 'Deselect All'
                      : 'Select All'}
                  </button>

                  {selectedClips.size > 0 && (
                    <button
                      onClick={async () => {
                        for (const clipId of selectedClips) {
                          const clip = job.clips.find((c) => c.id === clipId);
                          if (clip) await handleExportClip(clip);
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

                  {/* Apply Preset to Selected */}
                  {selectedClips.size > 0 && clipPresets.length > 0 && (
                    <div style={{ display: 'flex', gap: 6, alignItems: 'center', marginLeft: 'auto' }}>
                      <select
                        value={selectedPresetId}
                        onChange={(e) => setSelectedPresetId(e.target.value)}
                        style={{
                          padding: '6px 10px',
                          borderRadius: 'var(--radius-sm)',
                          fontSize: 11,
                          background: 'var(--bg-elevated)',
                          border: '1px solid var(--border)',
                          color: 'var(--text-primary)',
                        }}
                      >
                        <option value="">Pick preset...</option>
                        {clipPresets.map((p) => (
                          <option key={p.id} value={p.id}>{String(p.name || '')}</option>
                        ))}
                      </select>
                      <button
                        disabled={!selectedPresetId}
                        onClick={async () => {
                          const preset = clipPresets.find((p) => p.id === selectedPresetId);
                          if (!preset) return;
                          // Merge preset settings with current (preserve speaker colors)
                          const merged = {
                            ...clipSettings,
                            ...sanitizeSubtitleSettings(preset.settings),
                            speakerColors: clipSettings.speakerColors,
                          };
                          // Update UI settings panel to reflect the preset
                          setClipSettings(merged);
                          // Export selected clips using the merged settings directly
                          // (avoids React state batching delay)
                          for (const clipId of selectedClips) {
                            const clip = job.clips.find((c) => c.id === clipId);
                            if (clip) await handleExportClip(clip, merged);
                          }
                          showToast(`Applied "${preset.name}" and exporting ${selectedClips.size} clip(s)`, 'info');
                        }}
                        style={{
                          padding: '6px 12px',
                          background: selectedPresetId ? 'var(--accent-amber)' : 'var(--bg-elevated)',
                          color: selectedPresetId ? 'var(--bg-base)' : 'var(--text-muted)',
                          border: 'none',
                          borderRadius: 'var(--radius-sm)',
                          fontSize: 11,
                          fontWeight: 600,
                          cursor: selectedPresetId ? 'pointer' : 'default',
                          whiteSpace: 'nowrap',
                        }}
                      >
                        Apply &amp; Export
                      </button>
                    </div>
                  )}
                </div>
              )}

              {/* Clip count */}
              {job.clips?.length > 0 && (
                <div style={{
                  fontSize: 11,
                  fontFamily: 'var(--font-mono)',
                  color: 'var(--text-muted)',
                  marginBottom: 12,
                  textTransform: 'uppercase',
                  letterSpacing: '0.05em',
                }}>
                  Showing {filteredClips.length} of {job.clips.length} clips
                </div>
              )}

              {/* Clip cards */}
              <div className="responsive-grid" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(min(300px, 100%), 1fr))', gap: 16 }}>
                {filteredClips.map((clip) => (
                  <ClipCard
                    key={`${clip.id}_${clip.start_time}`}
                    clip={clip}
                    jobId={jobId}
                    isBest={clip.id === bestClipId}
                    onPreview={handleClipPreview}
                    onExport={handleExportClip}
                    onDelete={handleDeleteClip}
                    onTimesChanged={fetchJob}
                    onFindMoreLikeThis={(c) => {
                      // Re-run focus mode with this clip's focus/title as the query,
                      // scoped to a ±60s window around the source clip. Enhancement 7.
                      const query = (c.clip_focus || c.title || '').trim();
                      if (!query) { showToast('No focus query available on this clip', 'warning'); return; }
                      setClipFocusEnabled(true);
                      setClipFocusText(query);
                      const pad = 60;
                      handleGenerateClips({
                        clip_focus: query,
                        scope_start: Math.max(0, c.start_time - pad),
                        scope_end: Math.min(job?.duration || c.end_time + pad, c.end_time + pad),
                        append: true,
                      });
                    }}
                    selected={selectedClips.has(clip.id)}
                    onSelect={handleSelectClip}
                    exportQuality={clipSettings?.exportQuality || '1080p'}
                    scenes={job.scenes}
                  />
                ))}
              </div>
              {(!job.clips || job.clips.length === 0) && (
                <div style={{ textAlign: 'center', padding: 48, color: 'var(--text-muted)' }}>
                  {isProcessing ? 'Detecting viral moments...' : 'No clips detected. Adjust the settings above and click Generate Clips.'}
                  {!isProcessing && (clipFocusEnabled || (genSettings.viralScoreMin > 0 || genSettings.viralScoreMax < 100)) && (
                    <div style={{ marginTop: 16 }}>
                      <button
                        onClick={handleRelaxFilters}
                        style={{
                          padding: '8px 16px', fontSize: 12, fontWeight: 600,
                          background: 'var(--accent-cyan)', color: 'var(--bg-base)',
                          border: 'none', borderRadius: 'var(--radius-sm)', cursor: 'pointer',
                        }}
                      >
                        Relax filters &amp; retry
                      </button>
                    </div>
                  )}
                </div>
              )}
              {/* "Why these clips?" diagnostics drawer (Enhancement 6) */}
              {job.clips?.length > 0 && (
                <div style={{ marginTop: 16 }}>
                  <button
                    type="button"
                    onClick={() => {
                      setDiagOpen((v) => {
                        const next = !v;
                        if (next && !diagData) loadDiagnostics();
                        return next;
                      });
                    }}
                    aria-expanded={diagOpen}
                    style={{
                      fontSize: 11, color: 'var(--text-muted)',
                      background: 'transparent', border: 'none', cursor: 'pointer',
                      fontFamily: 'var(--font-mono)', textTransform: 'uppercase',
                      letterSpacing: '0.05em', padding: 0,
                    }}
                  >
                    {diagOpen ? '▼' : '▶'} Why these clips?
                  </button>
                  {diagOpen && (
                    <div style={{
                      marginTop: 8, padding: 12,
                      background: 'var(--bg-panel)',
                      border: '1px solid var(--border)',
                      borderRadius: 'var(--radius-md)',
                      fontSize: 11, color: 'var(--text-secondary)',
                    }}>
                      {diagLoading && <div>Loading diagnostics...</div>}
                      {!diagLoading && diagData && (
                        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                          <div>
                            <strong>Provider:</strong>{' '}
                            <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--accent-cyan)' }}>
                              {String(diagData.provider_used?.clips || 'unknown')}
                            </span>
                          </div>
                          <div>
                            <strong>Content type:</strong>{' '}
                            <span style={{ fontFamily: 'var(--font-mono)' }}>
                              {String(diagData.clip_content_type || 'generic')}
                            </span>
                          </div>
                          <div>
                            <strong>4-axis scoring:</strong>{' '}
                            {diagData.four_axis_enabled ? 'enabled' : 'disabled'}
                            {diagData.legacy_fill_count > 0 && (
                              <span style={{ color: 'var(--accent-amber)', marginLeft: 6 }}>
                                ({diagData.legacy_fill_count} clips using legacy fill)
                              </span>
                            )}
                          </div>
                          <div>
                            <strong>Axis population</strong> (out of {diagData.axis_population?.total ?? 0}):{' '}
                            <span style={{ fontFamily: 'var(--font-mono)' }}>
                              hook {diagData.axis_population?.hook ?? 0} ·
                              flow {diagData.axis_population?.flow ?? 0} ·
                              value {diagData.axis_population?.value ?? 0} ·
                              trend {diagData.axis_population?.trend ?? 0}
                            </span>
                          </div>
                          <div>
                            <strong>Hot zones:</strong>{' '}
                            {(diagData.hot_zones || []).length} persisted
                          </div>
                          <div>
                            <strong>Chapters:</strong>{' '}
                            {(diagData.chapters || []).length} persisted
                          </div>
                          <div>
                            <strong>Trend context:</strong>{' '}
                            {diagData.trend_context ? `${String(diagData.trend_context).length} chars` : 'none'}
                          </div>
                          <div>
                            <strong>Sentiment timeline:</strong>{' '}
                            {diagData.sentiment_timeline ? `${String(diagData.sentiment_timeline).length} chars` : 'none'}
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )}
              {job.clips?.length > 0 && filteredClips.length === 0 && (
                <div style={{ textAlign: 'center', padding: 48, color: 'var(--text-muted)' }}>
                  {clipSearchQuery.trim()
                    ? `No clips match "${clipSearchQuery.trim()}".`
                    : 'No clips match current filters.'}
                </div>
              )}

              {/* Exported clips */}
              {job.exported_clips?.length > 0 && (
                <div style={{ marginTop: 24 }}>
                  <h3 style={{ fontSize: 14, marginBottom: 12, color: 'var(--accent-cyan)' }}>Exported Clips</h3>
                  <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                    {job.exported_clips.map((ec, i) => (
                      <a
                        key={i}
                        href={`/api/files/${jobId}/clips/${ec.filename}`}
                        download
                        style={{
                          padding: '8px 16px',
                          background: 'var(--bg-elevated)',
                          border: '1px solid var(--border)',
                          borderRadius: 'var(--radius-sm)',
                          fontSize: 12,
                          color: 'var(--accent-cyan)',
                        }}
                      >
                        {String(ec.filename || '')}
                      </a>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
