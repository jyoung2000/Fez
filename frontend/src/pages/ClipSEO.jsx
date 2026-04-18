import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { useParams, Link } from 'react-router-dom';
import { ShareButton } from '../components/ShareDialog';
import { showToast } from '../components/Toast';
import { processKeyframes, interpolateSubjectX, isDynamic, computeClipSubjectX, fetchRenderPlan } from '../utils/subjectTracking';
import ClipSettingsPanel from '../components/ClipSettingsPanel';
import TranscriptViewer from '../components/TranscriptViewer';
import VideoEditor from '../components/VideoEditor';
import SubtitleOverlay from '../components/SubtitleOverlay';
import EditorErrorBoundary from '../components/EditorErrorBoundary';
import useResponsive from '../hooks/useResponsive';
import useEncodingManager from '../hooks/useEncodingManager';
import sanitizeJob from '../utils/sanitizeJob';
import useTimelineStore from '../stores/timelineStore';
import { buildOverlayPayload, buildVideoEffectsPayload, mapSubtitleSettings } from '../utils/buildExportPayload';

function formatDuration(seconds) {
  if (!seconds && seconds !== 0) return '0:00';
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

const DEFAULT_PALETTE = ['#00D9FF', '#F59E0B', '#10B981', '#A78BFA', '#EF4444', '#EC4899'];

// Backend-matching constants (ass_generator.py)
const FONT_SIZE_MAP = { small: 22, medium: 30, large: 40 };
const REF_W = 1920;
const REF_H = 1080;
const ASPECT_RATIO_DIMS = {
  '16:9': [1920, 1080],
  '9:16': [1080, 1920],
  '1:1': [1080, 1080],
  '4:5': [1080, 1350],
};
const ASPECT_RATIO_VALUES = {
  '16:9': 16 / 9,
  '9:16': 9 / 16,
  '1:1': 1.0,
  '4:5': 4 / 5,
};


function splitSegmentsByMaxWords(segments, maxWords) {
  if (!maxWords || maxWords <= 0) return segments;
  const result = [];
  for (const seg of segments) {
    const words = (seg.text || '').split(/\s+/).filter(Boolean);
    if (words.length <= maxWords) { result.push(seg); continue; }
    const totalWords = words.length;
    const duration = seg.end - seg.start;
    let currentTime = seg.start;
    for (let i = 0; i < totalWords; i += maxWords) {
      const chunkWords = words.slice(i, i + maxWords);
      const chunkDuration = duration * (chunkWords.length / totalWords);
      let chunkEnd = currentTime + chunkDuration;
      if (i + maxWords >= totalWords) chunkEnd = seg.end;
      if (chunkEnd - currentTime >= 0.1) {
        result.push({ ...seg, start: currentTime, end: chunkEnd, text: chunkWords.join(' ') });
      }
      currentTime = chunkEnd;
    }
  }
  return result;
}

function getCurrentSubtitle(transcript, currentTime, clipStart, clipEnd) {
  if (!transcript || !transcript.length) return [];
  return transcript.filter((seg) =>
    seg.start <= currentTime && seg.end > currentTime &&
    seg.start < clipEnd && seg.end > clipStart
  );
}

const _WORD_OVERHEAD_S = 0.06;
const _ANTICIPATION_S = 0.0;      // perceptual lead (0 = neutral)
const _AUDIO_BUFFER_S = 0.12;     // compensate for browser audio output lag
function getCurrentWordIndex(segment, relativeTime) {
  if (!segment || !segment.text) return -1;
  const words = segment.text.split(/\s+/).filter(Boolean);
  if (words.length <= 1) return words.length === 1 ? 0 : -1;
  const totalChars = words.reduce((sum, w) => sum + w.length, 0);
  if (totalChars === 0) return -1;
  const segDuration = segment.end - segment.start;
  const elapsed = (relativeTime - segment.start) + _ANTICIPATION_S - _AUDIO_BUFFER_S;
  if (elapsed < 0) return -1;
  const totalOverhead = _WORD_OVERHEAD_S * words.length;
  const charTime = Math.max(segDuration - totalOverhead, segDuration * 0.5);
  const overheadPer = (segDuration - charTime) / words.length;
  let t = 0;
  for (let i = 0; i < words.length; i++) {
    const wordDur = charTime * (words[i].length / totalChars) + overheadPer;
    if (elapsed < t + wordDur) return i;
    t += wordDur;
  }
  return words.length - 1;
}

export default function ClipSEO() {
  const { jobId, clipId } = useParams();
  // Multi-track editor timeline state (global zustand store)
  const timelineItems = useTimelineStore((s) => s.items);
  const timelineMediaLibrary = useTimelineStore((s) => s.mediaLibrary);
  const [job, setJob] = useState(null);
  const [clip, setClip] = useState(null);
  const [seo, setSeo] = useState(null);
  const [loading, setLoading] = useState(true);
  const [generating, setGenerating] = useState(false);
  const [genStatus, setGenStatus] = useState('');
  const [genElapsed, setGenElapsed] = useState(0);
  const [copied, setCopied] = useState(null);
  const [shortsDesc, setShortsDesc] = useState('');
  const [longFormDesc, setLongFormDesc] = useState('');
  const [generatingShorts, setGeneratingShorts] = useState(false);
  const [generatingLongForm, setGeneratingLongForm] = useState(false);
  const [shortsElapsed, setShortsElapsed] = useState(0);
  const [longFormElapsed, setLongFormElapsed] = useState(0);

  // Custom uploaded fonts
  const [customFonts, setCustomFonts] = useState([]);
  useEffect(() => {
    let cancelled = false;
    fetch('/api/fonts')
      .then(r => r.ok ? r.json() : [])
      .then(fonts => {
        if (cancelled || !Array.isArray(fonts)) return;
        setCustomFonts(fonts);
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

  // Debounced save for user edits to SEO data
  const saveTimerRef = useRef(null);
  const saveClipSeo = useCallback((fields) => {
    if (saveTimerRef.current) clearTimeout(saveTimerRef.current);
    saveTimerRef.current = setTimeout(() => {
      fetch(`/api/jobs/${jobId}/clips/${clipId}/seo`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(fields),
      }).catch(() => {});
    }, 800);
  }, [jobId, clipId]);
  useEffect(() => () => { if (saveTimerRef.current) clearTimeout(saveTimerRef.current); }, []);

  const videoRef = useRef(null);
  const handleVideoRef = useCallback((el) => { videoRef.current = el; }, []);
  const videoContainerRef = useRef(null);
  const wsRef = useRef(null);
  const [playing, setPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [videoContainerSize, setVideoContainerSize] = useState({ w: 0, h: 0 });

  // Clip trim state
  const [startTime, setStartTime] = useState(null);
  const [endTime, setEndTime] = useState(null);
  const [startText, setStartText] = useState('');
  const [endText, setEndText] = useState('');

  // Clip settings — managed by ClipSettingsPanel, received via onSettingsChange.
  // Load from localStorage so settings persist across page reloads immediately.
  const CLIP_SETTINGS_DEFAULTS = React.useMemo(() => ({
    aspectRatio: null,
    subtitlesEnabled: false,
    subtitleFont: 'DM Sans',
    subtitleSize: 30,
    subtitleFontWeight: 700,
    subtitleFontColor: '#FFFFFF',
    subtitlePosition: 'bottom',
    speakerColors: {},
    subtitleBgEnabled: false,
    subtitleBgColor: '#000000',
    subtitleBgOpacity: 75,
    subtitleBgRadius: 0,
    subtitleOutlineColor: '#000000',
    subtitleOutlineOpacity: 100,
    subtitleOutlineWidth: 2,
    showSpeakerLabels: false,
    subtitleMaxWidth: 90,
    subtitleOffsetV: 4,
    subtitleMaxWords: 0,
    activeWordEnabled: false,
    activeWordColor: '#FFD700',
    activeWordOutlineColor: '#000000',
    activeWordBgColor: '#000000',
    activeWordBgOpacity: 0,
    activeWordBgRadius: 4,
    useSpeakerColors: true,
    playbackVolume: 100,
    playbackSpeed: 1.0,
  }), []);
  const [clipSettings, setClipSettings] = useState(CLIP_SETTINGS_DEFAULTS);
  const clipSettingsLoadedFromServer = useRef(false);
  const skipNextServerSave = useRef(false);

  // ── Server is source of truth for subtitle settings ──
  // On initial load this is handled in the fetch callback (same batch as setJob)
  // to avoid a flash. This effect only handles subsequent changes (e.g., fetchJob
  // refreshes after transcript edit, or hot-reload during dev).
  const initialSettingsApplied = useRef(false);
  useEffect(() => {
    if (!job) return;
    // Skip the first run — settings were already applied in the fetch callback
    if (!initialSettingsApplied.current) {
      initialSettingsApplied.current = true;
      return;
    }
    if (job.subtitle_settings && Object.keys(job.subtitle_settings).length > 0) {
      skipNextServerSave.current = true;
      setClipSettings({ ...CLIP_SETTINGS_DEFAULTS, ...job.subtitle_settings });
      clipSettingsLoadedFromServer.current = true;
    } else if (!clipSettingsLoadedFromServer.current) {
      try {
        const saved = localStorage.getItem('clipai_clip_settings');
        if (saved) setClipSettings({ ...CLIP_SETTINGS_DEFAULTS, ...JSON.parse(saved) });
      } catch {}
    }
  }, [job?.subtitle_settings, CLIP_SETTINGS_DEFAULTS]);

  // Persist to server (debounced) + localStorage fallback
  const saveSettingsTimerRef = useRef(null);
  useEffect(() => {
    if (clipSettings && Object.keys(clipSettings).length > 0) {
      try { localStorage.setItem('clipai_clip_settings', JSON.stringify(clipSettings)); } catch {}
    }
    if (skipNextServerSave.current) {
      skipNextServerSave.current = false;
      return;
    }
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

  const [currentWordIdx, setCurrentWordIdx] = useState(-1);
  const [settingsAppliedFlash, setSettingsAppliedFlash] = useState(false);
  const settingsFlashTimerRef = useRef(null);
  useEffect(() => () => { if (settingsFlashTimerRef.current) clearTimeout(settingsFlashTimerRef.current); }, []);

  // VideoEditor state for export params
  const [editorTrim, setEditorTrim] = useState({ trimStart: 0, trimEnd: 0 });
  const [editorVolume, setEditorVolume] = useState(1.0);
  const [editorSpeed, setEditorSpeed] = useState(1.0);

  // ── Segment persistence via localStorage + server ──
  const segStorageKey = `clipai_segments_${jobId}_${clipId}`;
  const [editorSegments, setEditorSegments] = useState(() => {
    try {
      const raw = localStorage.getItem(segStorageKey);
      return raw ? JSON.parse(raw) : [];
    } catch { return []; }
  });
  const saveSegments = useCallback((segs) => {
    setEditorSegments(segs);
    try {
      if (segs && segs.length > 0) localStorage.setItem(segStorageKey, JSON.stringify(segs));
      else localStorage.removeItem(segStorageKey);
    } catch {}
  }, [segStorageKey]);

  // ── Editor state persistence to server (shared with Analysis page) ──
  const editorStateSaveTimerRef = useRef(null);
  const editorStateLoadedRef = useRef(false);
  useEffect(() => () => { if (editorStateSaveTimerRef.current) clearTimeout(editorStateSaveTimerRef.current); }, []);

  // Load editor state from server on mount
  useEffect(() => {
    if (!jobId || !clipId) return;
    fetch(`/api/jobs/${jobId}/clips/${clipId}/editor-state`)
      .then(r => r.ok ? r.json() : null)
      .then(data => {
        if (data?.state) {
          const s = data.state;
          if (s.trim) setEditorTrim(s.trim);
          if (s.volume != null) setEditorVolume(s.volume);
          if (s.speed != null) setEditorSpeed(s.speed);
          if (s.segments?.length > 0 && editorSegments.length === 0) {
            setEditorSegments(s.segments);
            try { localStorage.setItem(segStorageKey, JSON.stringify(s.segments)); } catch {}
          }
        }
        editorStateLoadedRef.current = true;
      })
      .catch(() => { editorStateLoadedRef.current = true; });
  }, [jobId, clipId]); // eslint-disable-line react-hooks/exhaustive-deps

  // Save editor state to server (debounced) when trim/volume/speed/segments change
  useEffect(() => {
    if (!editorStateLoadedRef.current || !jobId || !clipId) return;
    if (editorStateSaveTimerRef.current) clearTimeout(editorStateSaveTimerRef.current);
    editorStateSaveTimerRef.current = setTimeout(() => {
      fetch(`/api/jobs/${jobId}/clips/${clipId}/editor-state`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          trim: editorTrim,
          volume: editorVolume,
          speed: editorSpeed,
          segments: editorSegments,
        }),
      }).catch(() => {});
    }, 1000);
  }, [editorTrim, editorVolume, editorSpeed, editorSegments, jobId, clipId]);
  const [showInlineSubSettings, setShowInlineSubSettings] = useState(false);

  // Layout mode: 'editor' = full-width NLE above, 'sidebyside' = player left + transcript right
  const [layoutMode, setLayoutMode] = useState(() => {
    try { return localStorage.getItem('clipai_seo_layout') || 'editor'; } catch { return 'editor'; }
  });
  useEffect(() => {
    try { localStorage.setItem('clipai_seo_layout', layoutMode); } catch {}
  }, [layoutMode]);

  // Fullscreen state
  const fullscreenRef = useRef(null);
  const [isFullscreen, setIsFullscreen] = useState(false);

  // Speaker name editing
  const [editingSpeaker, setEditingSpeaker] = useState(null);
  const [editSpeakerValue, setEditSpeakerValue] = useState('');

  // Persistent encoding manager (survives page navigation)
  const encoding = useEncodingManager();
  const exportId = `${jobId}_${clipId}`;
  const exportTask = encoding.tasks[exportId];
  const exporting = exportTask?.status === 'encoding';
  const exportProgress = exporting ? (exportTask?.message || 'Exporting...') : '';
  const downloadUrl = exportTask?.status === 'complete' ? exportTask.downloadUrl : null;

  // Responsive
  const { isMobile } = useResponsive();

  // Derive speakers from job transcript
  const speakers = [];
  (job?.transcript || []).forEach((seg) => {
    if (seg.speaker && !speakers.includes(seg.speaker)) speakers.push(seg.speaker);
  });

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

  // Destructure clipSettings so existing references work seamlessly
  const {
    aspectRatio = null,
    exportQuality = '1080p',
    subtitlesEnabled = false,
    subtitleFont = 'DM Sans',
    subtitleSize = 30,
    subtitleFontWeight = 700,
    subtitleFontColor = '#FFFFFF',
    subtitlePosition = 'bottom',
    useSpeakerColors = true,
    speakerColors = {},
    subtitleBgEnabled = false,
    subtitleBgColor = '#000000',
    subtitleBgOpacity = 75,
    subtitleBgRadius = 0,
    subtitleOutlineColor = '#000000',
    subtitleOutlineOpacity = 100,
    subtitleOutlineWidth = 2,
    showSpeakerLabels = false,
    subtitleMaxWidth = 90,
    subtitleOffsetV = 4,
    subtitleMaxWords = 0,
    activeWordEnabled = false,
    activeWordColor = '#FFD700',
    activeWordOutlineColor = '#000000',
    activeWordBgColor = '#000000',
    activeWordBgOpacity = 0,
    activeWordBgRadius = 4,
    playbackVolume = 100,
    playbackSpeed = 1.0,
  } = clipSettings;

  // Load job data
  useEffect(() => {
    fetch(`/api/jobs/${jobId}`)
      .then((r) => r.ok ? r.json() : null)
      .then((raw) => raw ? sanitizeJob(raw) : null)
      .then((data) => {
        if (data) {
          setJob(data);
          const found = (data.clips || []).find((c) => c.id === parseInt(clipId));
          setClip(found || null);
          if (found) {
            setStartTime(found.start_time);
            setEndTime(found.end_time);
            setStartText(formatDuration(found.start_time));
            setEndText(formatDuration(found.end_time));
            // Restore persisted SEO data
            if (found.seo_title) {
              setSeo({
                title: typeof found.seo_title === 'string' ? found.seo_title : String(found.seo_title),
                description: typeof found.seo_description === 'string' ? found.seo_description : String(found.seo_description ?? ''),
                tags: Array.isArray(found.seo_tags) ? found.seo_tags.map((t) => typeof t === 'string' ? t : String(t)) : [],
                platform_tips: typeof found.seo_platform_tips === 'string' ? found.seo_platform_tips : String(found.seo_platform_tips ?? ''),
              });
            }
            if (found.shorts_description) setShortsDesc(found.shorts_description);
            if (found.longform_description) setLongFormDesc(found.longform_description);
          }
          // ── Apply server subtitle settings in the same batch as setJob ──
          // This prevents a flash where the component renders with default
          // settings (subtitles off, no speaker colors) before the useEffect
          // for job.subtitle_settings fires on the next tick.
          if (data.subtitle_settings && Object.keys(data.subtitle_settings).length > 0) {
            skipNextServerSave.current = true;
            setClipSettings(prev => ({ ...prev, ...data.subtitle_settings }));
            clipSettingsLoadedFromServer.current = true;
          } else if (!clipSettingsLoadedFromServer.current) {
            try {
              const saved = localStorage.getItem('clipai_clip_settings');
              if (saved) setClipSettings(prev => ({ ...prev, ...JSON.parse(saved) }));
            } catch {}
          }
        }
      })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, [jobId, clipId]);

  const fetchJob = useCallback(async () => {
    try {
      const res = await fetch(`/api/jobs/${jobId}`);
      if (res.ok) {
        const data = sanitizeJob(await res.json());
        setJob(data);
        const found = (data.clips || []).find((c) => c.id === parseInt(clipId));
        if (found) setClip(found);
      }
    } catch {}
  }, [jobId, clipId]);

  // Track fullscreen changes
  useEffect(() => {
    const onFsChange = () => setIsFullscreen(!!document.fullscreenElement);
    document.addEventListener('fullscreenchange', onFsChange);
    return () => document.removeEventListener('fullscreenchange', onFsChange);
  }, []);

  // Track video container size for subtitle font scaling
  useEffect(() => {
    const el = videoContainerRef.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      for (const entry of entries) {
        setVideoContainerSize({ w: entry.contentRect.width, h: entry.contentRect.height });
      }
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // Parse source dimensions from job resolution
  const sourceDims = useMemo(() => {
    if (job?.resolution) {
      const parts = job.resolution.split('x').map(Number);
      if (parts.length === 2 && parts[0] > 0 && parts[1] > 0) {
        return { w: parts[0], h: parts[1] };
      }
    }
    return { w: 1920, h: 1080 };
  }, [job?.resolution]);

  // Stable references for VideoEditor props — prevents re-mount loops
  // caused by new array/object identity on every render.
  const stableScenes = useMemo(() => job?.scenes || [], [job?.scenes]);
  const stableTranscript = useMemo(
    () => job?.translated_transcript?.length ? job.translated_transcript : (job?.transcript || []),
    [job?.translated_transcript, job?.transcript],
  );
  const stableSpeakerNames = useMemo(() => job?.speaker_names || {}, [job?.speaker_names]);
  const stableSceneCuts = useMemo(() => job?.scene_cut_timestamps || null, [job?.scene_cut_timestamps]);
  const stableSubjectTrack = useMemo(() => job?.subject_track || null, [job?.subject_track]);

  // Fetch RenderPlan for backend slot identity (Phase D)
  const [renderPlan, setRenderPlan] = useState(null);
  useEffect(() => {
    if (!jobId || !clipId) return;
    let cancelled = false;
    fetchRenderPlan(jobId, { mode: 'clip', clipIndex: parseInt(clipId), aspectRatio: aspectRatio || '9:16' })
      .then(plan => { if (!cancelled) setRenderPlan(plan); })
      .catch(() => {});
    return () => { cancelled = true; };
  }, [jobId, clipId, aspectRatio]);

  // Output dimensions based on aspect ratio (matches clip_exporter.py)
  const outputDims = useMemo(() => {
    if (aspectRatio && ASPECT_RATIO_DIMS[aspectRatio]) {
      return { w: ASPECT_RATIO_DIMS[aspectRatio][0], h: ASPECT_RATIO_DIMS[aspectRatio][1] };
    }
    return sourceDims;
  }, [aspectRatio, sourceDims]);

  // Two-step font scaling matching backend (ass_generator.py:146-158)
  const backendFontScale = useMemo(
    () => Math.min(outputDims.w, outputDims.h) / Math.min(REF_W, REF_H),
    [outputDims],
  );
  const containerScale = useMemo(() => {
    if (videoContainerSize.w === 0) return 0;
    const scaleW = videoContainerSize.w / outputDims.w;
    const scaleH = videoContainerSize.h / outputDims.h;
    return Math.min(scaleW, scaleH);
  }, [videoContainerSize, outputDims]);

  // Compute subject_x with boundary interpolation (matches backend logic)
  const clipSubjectX = useMemo(() => {
    if (!job?.scenes?.length || startTime === null || endTime === null) return 50;
    return computeClipSubjectX(job.scenes, startTime, endTime);
  }, [job?.scenes, startTime, endTime]);

  // Dynamic subject tracking keyframes (full pipeline matching backend)
  const subjectKeyframes = useMemo(
    () => {
      if (!job?.scenes?.length || startTime === null || endTime === null) return null;
      const _srcRatio = sourceDims.w / sourceDims.h;
      const _targetRatio = (aspectRatio && ASPECT_RATIO_VALUES[aspectRatio])
        ? ASPECT_RATIO_VALUES[aspectRatio]
        : _srcRatio;
      const _isCrop = Math.abs(_srcRatio - _targetRatio) > 0.01;
      return processKeyframes(job.scenes, startTime, endTime, _isCrop ? _srcRatio : null, _isCrop ? _targetRatio : null, null, job.scene_cut_timestamps || null);
    },
    [job?.scenes, startTime, endTime, aspectRatio, sourceDims],
  );

  const clipTimeRange = useMemo(() => {
    if (startTime === null || endTime === null) return null;
    return { start: startTime, end: endTime };
  }, [startTime, endTime]);

  // Determine if cropping is needed for the selected aspect ratio
  const isCrop = useMemo(() => {
    if (!aspectRatio || !ASPECT_RATIO_VALUES[aspectRatio]) return false;
    const srcRatio = sourceDims.w / sourceDims.h;
    return Math.abs(srcRatio - ASPECT_RATIO_VALUES[aspectRatio]) > 0.01;
  }, [aspectRatio, sourceDims]);

  const targetRatio = aspectRatio && ASPECT_RATIO_VALUES[aspectRatio]
    ? ASPECT_RATIO_VALUES[aspectRatio]
    : sourceDims.w / sourceDims.h;

  // Center portrait video by constraining width
  const videoMaxWidth = useMemo(() => {
    if (targetRatio < 1) {
      const maxHpx = (typeof window !== 'undefined' ? window.innerHeight : 900) * 0.45;
      return Math.min(Math.round(maxHpx * targetRatio), 400);
    }
    return undefined;
  }, [targetRatio]);

  // Pre-split transcript segments by max words for subtitle preview
  const subtitleTranscript = job?.translated_transcript?.length ? job.translated_transcript : (job?.transcript || []);
  const splitTranscript = useMemo(() => {
    if (!subtitleMaxWords || !subtitleTranscript?.length) return subtitleTranscript;
    return splitSegmentsByMaxWords(subtitleTranscript, subtitleMaxWords);
  }, [subtitleTranscript, subtitleMaxWords]);

  // WebSocket for SEO generation progress only;
  // export progress is handled by the global useEncodingManager hook.
  useEffect(() => {
    const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const ws = new WebSocket(`${proto}//${window.location.host}/ws/jobs/${jobId}`);
    wsRef.current = ws;

    ws.onmessage = (evt) => {
      try {
        const msg = JSON.parse(evt.data);
        if (msg.type === 'status' && msg.status === 'generating_seo') {
          setGenStatus(typeof msg.message === 'string' ? msg.message : String(msg.message ?? 'Generating...'));
        }
      } catch {}
    };

    return () => {
      ws.close();
      wsRef.current = null;
    };
  }, [jobId]);

  // Video controls for clip preview
  useEffect(() => {
    const video = videoRef.current;
    if (!video || startTime === null) return;

    const onLoaded = () => { video.currentTime = startTime; };
    const onTimeUpdate = () => {
      setCurrentTime(video.currentTime);
      if (endTime && video.currentTime >= endTime) {
        video.pause();
        setPlaying(false);
      }
    };

    video.addEventListener('loadedmetadata', onLoaded);
    video.addEventListener('timeupdate', onTimeUpdate);
    if (video.readyState >= 1) onLoaded();

    return () => {
      video.removeEventListener('loadedmetadata', onLoaded);
      video.removeEventListener('timeupdate', onTimeUpdate);
    };
  }, [startTime, endTime]);

  // Dynamic subject tracking: update objectPosition during playback
  const hasDynamicSubject = useMemo(
    () => isCrop && subjectKeyframes && isDynamic(subjectKeyframes),
    [isCrop, subjectKeyframes],
  );
  useEffect(() => {
    if (!hasDynamicSubject) return;
    const video = videoRef.current;
    if (!video) return;
    const onTime = () => {
      const relTime = video.currentTime - (startTime || 0);
      const sx = interpolateSubjectX(subjectKeyframes, relTime);
      video.style.objectPosition = `${Math.max(0, Math.min(100, Math.round(sx)))}% 50%`;
    };
    video.addEventListener('timeupdate', onTime);
    onTime();
    return () => video.removeEventListener('timeupdate', onTime);
  }, [hasDynamicSubject, subjectKeyframes, startTime]);

  // rAF loop for smooth active word tracking
  useEffect(() => {
    if (!activeWordEnabled || !subtitlesEnabled) { setCurrentWordIdx(-1); return; }
    const video = videoRef.current;
    if (!video) return;
    let animId;
    let prevIdx = -1;
    const tick = () => {
      const active = getCurrentSubtitle(splitTranscript, video.currentTime, startTime || 0, endTime || Infinity);
      if (active.length > 0) {
        const idx = getCurrentWordIndex(active[0], video.currentTime);
        if (idx !== prevIdx) { prevIdx = idx; setCurrentWordIdx(idx); }
      } else if (prevIdx !== -1) {
        prevIdx = -1;
        setCurrentWordIdx(-1);
      }
      animId = requestAnimationFrame(tick);
    };
    animId = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(animId);
  }, [activeWordEnabled, subtitlesEnabled, splitTranscript, startTime, endTime]);

  const togglePlay = () => {
    const video = videoRef.current;
    if (!video || startTime === null) return;
    if (video.paused) {
      if (endTime && video.currentTime >= endTime) video.currentTime = startTime;
      video.play().then(() => setPlaying(true)).catch(() => {});
    } else {
      video.pause();
      setPlaying(false);
    }
  };

  const toggleFullscreen = () => {
    const el = fullscreenRef.current;
    if (!el) return;
    if (document.fullscreenElement) {
      document.exitFullscreen().catch(() => {});
    } else {
      el.requestFullscreen().catch(() => {});
    }
  };

  const handleRenameSpeaker = async (oldName, newName) => {
    const trimmed = newName.trim();
    if (!trimmed || trimmed === oldName) {
      setEditingSpeaker(null);
      return;
    }
    try {
      const res = await fetch(`/api/jobs/${jobId}/speakers`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ speaker_names: { [oldName]: trimmed } }),
      });
      if (res.ok) {
        await fetchJob();
        showToast(`Renamed "${oldName}" to "${trimmed}"`, 'success');
      }
    } catch {
      showToast('Speaker rename failed', 'error');
    }
    setEditingSpeaker(null);
  };

  const generateSEO = useCallback(async () => {
    setGenerating(true);
    setGenElapsed(0);
    setGenStatus('Connecting to AI provider...');

    const startMs = Date.now();
    const timerInterval = setInterval(() => {
      setGenElapsed(Math.floor((Date.now() - startMs) / 1000));
    }, 1000);

    const statusInterval = setInterval(() => {
      setGenStatus((prev) => {
        const msgs = [
          'Connecting to AI provider...',
          'Analyzing clip transcript...',
          'Generating SEO metadata...',
          'Crafting title and caption...',
          'Picking tags...',
          'Almost done...',
        ];
        const idx = msgs.indexOf(prev);
        return msgs[Math.min(idx + 1, msgs.length - 1)];
      });
    }, 4000);

    try {
      const res = await fetch(`/api/jobs/${jobId}/seo/${clipId}`, { method: 'POST' });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || 'SEO generation failed');
      }
      const data = await res.json();
      const rawSeo = data.seo || {};
      setSeo({
        title: typeof rawSeo.title === 'string' ? rawSeo.title : String(rawSeo.title ?? ''),
        description: typeof rawSeo.description === 'string' ? rawSeo.description : String(rawSeo.description ?? ''),
        tags: Array.isArray(rawSeo.tags) ? rawSeo.tags.map((t) => typeof t === 'string' ? t : String(t)) : [],
        platform_tips: typeof rawSeo.platform_tips === 'string' ? rawSeo.platform_tips : String(rawSeo.platform_tips ?? ''),
      });
      showToast(`SEO generated via ${data.provider}`, 'success');
    } catch (err) {
      showToast(err.message, 'error');
    } finally {
      clearInterval(statusInterval);
      clearInterval(timerInterval);
      setGenerating(false);
      setGenStatus('');
      setGenElapsed(0);
    }
  }, [jobId, clipId]);

  const generateDescription = useCallback(async (descType) => {
    const isShorts = descType === 'shorts';
    const setter = isShorts ? setShortsDesc : setLongFormDesc;
    const setLoading = isShorts ? setGeneratingShorts : setGeneratingLongForm;
    const setElapsed = isShorts ? setShortsElapsed : setLongFormElapsed;
    setLoading(true);
    setElapsed(0);

    const startMs = Date.now();
    const timerInterval = setInterval(() => {
      setElapsed(Math.floor((Date.now() - startMs) / 1000));
    }, 1000);

    try {
      const res = await fetch(`/api/jobs/${jobId}/generate-description/${clipId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ description_type: descType }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || 'Description generation failed');
      }
      const data = await res.json();
      setter(typeof data.description === 'string' ? data.description : String(data.description ?? ''));
      showToast(`${isShorts ? 'Shorts' : 'YouTube'} description generated via ${data.provider}`, 'success');
    } catch (err) {
      showToast(err.message, 'error');
    } finally {
      clearInterval(timerInterval);
      setLoading(false);
      setElapsed(0);
    }
  }, [jobId, clipId]);

  // Derive a status message from elapsed seconds for description generation
  const descStatusMessage = (elapsed) => {
    if (elapsed < 3) return 'Connecting to AI provider...';
    if (elapsed < 8) return 'Analyzing transcript and key scenes...';
    if (elapsed < 16) return 'Writing description with context...';
    if (elapsed < 25) return 'Generating keywords and hashtags...';
    if (elapsed < 40) return 'Refining output...';
    return 'Almost done — large descriptions take longer...';
  };

  const handleExport = useCallback(() => {
    if (startTime === null || endTime === null) return;
    // Enable subtitles if global is on OR any segment has subtitles enabled
    const globalSubsOn = subtitlesEnabled || false;
    const anySegmentSubsOn = editorSegments.some(s => s.subtitlesEnabled !== false);
    const needsSubtitles = globalSubsOn || anySegmentSubsOn;
    const body = {
      start: startTime,
      end: endTime,
      clip_id: parseInt(clipId),
      clip_title: clip?.title || `Clip ${clipId}`,
      aspect_ratio: aspectRatio,
      export_quality: exportQuality || '1080p',
      subtitles_enabled: needsSubtitles,
      global_subtitles_enabled: globalSubsOn,
    };
    if (needsSubtitles) {
      body.subtitle_settings = mapSubtitleSettings(clipSettings);
    }
    // Include VideoEditor trim/volume/speed/segments params
    if (editorTrim.trimStart > 0) body.trim_start_offset = editorTrim.trimStart;
    if (editorTrim.trimEnd > 0) body.trim_end_offset = editorTrim.trimEnd;
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
    // Include multi-track editor video effects + transform so export matches preview
    const videoEffects = buildVideoEffectsPayload(timelineItems);
    if (videoEffects) body.video_effects = videoEffects;

    // Build overlay arrays via shared utility (consistent filtering + validation)
    const overlays = buildOverlayPayload({
      timelineItems,
      mediaLibrary: timelineMediaLibrary,
      clipStart: startTime,
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
    encoding.startExport(jobId, parseInt(clipId), clip?.title || `Clip ${clipId}`, body);
    showToast(`Exporting "${clip?.title || `Clip ${clipId}`}"...`, 'info');
  }, [jobId, clipId, clip, startTime, endTime, clipSettings, encoding, editorTrim, editorVolume, editorSpeed, editorSegments, timelineItems, timelineMediaLibrary]);

  const handleApplySettings = useCallback((applied) => {
    setClipSettings(applied);
    setSettingsAppliedFlash(true);
    if (settingsFlashTimerRef.current) clearTimeout(settingsFlashTimerRef.current);
    settingsFlashTimerRef.current = setTimeout(() => setSettingsAppliedFlash(false), 1800);
    showToast('Settings applied to preview & export', 'success');
  }, []);

  // Determine if playhead is inside a segment — used for segment-aware subs toggle.
  // MUST be before early returns to satisfy Rules of Hooks.
  const activeSegment = useMemo(() => {
    if (!editorSegments || editorSegments.length === 0) return null;
    return editorSegments.find(s => currentTime >= s.start && currentTime < s.end) || null;
  }, [editorSegments, currentTime]);

  const effectiveSubsEnabled = activeSegment ? (activeSegment.subtitlesEnabled !== false) : subtitlesEnabled;

  const handleSubsToggle = useCallback(() => {
    if (activeSegment) {
      const newVal = activeSegment.subtitlesEnabled === false;
      setEditorSegments(prev =>
        prev.map(s => s.id === activeSegment.id ? { ...s, subtitlesEnabled: newVal } : s)
      );
    } else {
      setClipSettings(prev => ({ ...prev, subtitlesEnabled: !prev.subtitlesEnabled }));
    }
  }, [activeSegment]);

  const copyToClipboard = (text, label) => {
    navigator.clipboard.writeText(text).then(() => {
      setCopied(label);
      setTimeout(() => setCopied(null), 2000);
    });
  };

  if (loading) {
    return (
      <div style={{ textAlign: 'center', padding: 48, color: 'var(--text-secondary)' }}>
        <div style={{
          width: 24, height: 24, border: '2px solid var(--border)', borderTopColor: 'var(--accent-cyan)',
          borderRadius: '50%', animation: 'spin 0.8s linear infinite',
          margin: '0 auto 12px',
        }} />
        Loading clip editor...
      </div>
    );
  }

  if (!job || !clip) {
    return (
      <div style={{ textAlign: 'center', padding: 48 }}>
        <div style={{ color: 'var(--danger)', marginBottom: 16 }}>Clip not found</div>
        <Link to="/clips" style={{ color: 'var(--accent-cyan)' }}>Back to Viral Clips</Link>
      </div>
    );
  }

  const videoExt = job.file_path?.split('.').pop() || 'mp4';
  const videoSrc = `/api/files/${jobId}/video.${videoExt}`;
  const clipDur = (endTime ?? clip.end_time) - (startTime ?? clip.start_time);
  const elapsed = Math.max(0, Math.min(clipDur, currentTime - (startTime ?? clip.start_time)));
  const progress = clipDur > 0 ? (elapsed / clipDur) * 100 : 0;
  const scoreColor = clip.viral_score >= 80 ? 'var(--accent-amber)' : clip.viral_score >= 50 ? 'var(--accent-cyan)' : 'var(--text-secondary)';

  const sectionStyle = {
    background: 'var(--bg-panel)',
    border: '1px solid var(--border)',
    borderRadius: 'var(--radius-md)',
    padding: 16,
    marginBottom: 16,
  };

  const copyBtnStyle = (label) => ({
    padding: '4px 10px',
    fontSize: 10,
    background: copied === label ? 'var(--accent-cyan)' : 'var(--bg-elevated)',
    color: copied === label ? 'var(--bg-base)' : 'var(--text-muted)',
    border: '1px solid var(--border)',
    borderRadius: 'var(--radius-sm)',
    cursor: 'pointer',
    fontFamily: 'var(--font-mono)',
  });

  const timeInput = {
    width: '100%',
    padding: '5px 8px',
    fontSize: 13,
    fontFamily: 'var(--font-mono)',
    background: 'var(--bg-elevated)',
    color: 'var(--accent-cyan)',
    border: '1px solid var(--border)',
    borderRadius: 'var(--radius-sm)',
    outline: 'none',
  };

  return (
    <div>
      {/* Shared keyframes */}
      <style>{`
        @keyframes seo-spin { to { transform: rotate(360deg); } }
        @keyframes seo-pulse { 0%,100% { opacity: 0.5; } 50% { opacity: 1; } }
      `}</style>

      {/* Breadcrumb + Share + Layout Toggle */}
      <div style={{ marginBottom: 16, display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 8 }}>
        <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>
          <Link to="/clips" style={{ color: 'var(--accent-cyan)', textDecoration: 'none' }}>Viral Clips</Link>
          {' / '}
          <Link to={`/analysis/${jobId}`} style={{ color: 'var(--accent-cyan)', textDecoration: 'none' }}>{String(job.filename || '')}</Link>
          {' / '}
          <span style={{ color: 'var(--text-primary)' }}>SEO — Clip {clipId}</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          {/* Share button: hands the recipient a public URL for THIS
              clip. Recipients view the clip without needing an account. */}
          <ShareButton jobId={jobId} clipId={Number(clipId)} label="Share clip" />
        </div>
        {!isMobile && (
          <div style={{ display: 'flex', gap: 2, background: 'var(--bg-elevated)', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border)', padding: 2 }}>
            <button
              onClick={() => setLayoutMode('editor')}
              title="Full editor — focus on trimming, speed, and volume controls"
              style={{
                display: 'flex', alignItems: 'center', gap: 5,
                padding: '5px 12px', fontSize: 11, fontWeight: 500,
                background: layoutMode === 'editor' ? 'var(--accent-cyan-dim)' : 'transparent',
                color: layoutMode === 'editor' ? 'var(--accent-cyan)' : 'var(--text-muted)',
                border: layoutMode === 'editor' ? '1px solid var(--accent-cyan)' : '1px solid transparent',
                borderRadius: 'var(--radius-sm)', cursor: 'pointer',
                transition: 'all 0.15s ease',
              }}
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                <rect x="2" y="3" width="20" height="14" rx="2" ry="2" />
                <line x1="2" y1="20" x2="22" y2="20" />
                <line x1="6" y1="20" x2="6" y2="17" />
                <line x1="18" y1="20" x2="18" y2="17" />
              </svg>
              Editor
            </button>
            <button
              onClick={() => setLayoutMode('sidebyside')}
              title="Side-by-side — watch clip while editing transcript"
              style={{
                display: 'flex', alignItems: 'center', gap: 5,
                padding: '5px 12px', fontSize: 11, fontWeight: 500,
                background: layoutMode === 'sidebyside' ? 'var(--accent-cyan-dim)' : 'transparent',
                color: layoutMode === 'sidebyside' ? 'var(--accent-cyan)' : 'var(--text-muted)',
                border: layoutMode === 'sidebyside' ? '1px solid var(--accent-cyan)' : '1px solid transparent',
                borderRadius: 'var(--radius-sm)', cursor: 'pointer',
                transition: 'all 0.15s ease',
              }}
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                <rect x="2" y="3" width="20" height="18" rx="2" ry="2" />
                <line x1="12" y1="3" x2="12" y2="21" />
              </svg>
              Side by Side
            </button>
          </div>
        )}
      </div>

      {/* ── VIDEO EDITOR: Single instance, always mounted — layout changes via CSS only ── */}
      {/* Wrapping flex container for sidebyside mode */}
      <div style={{
        display: !isMobile && layoutMode === 'sidebyside' ? 'flex' : 'block',
        gap: !isMobile && layoutMode === 'sidebyside' ? 16 : undefined,
        marginBottom: 20,
        alignItems: !isMobile && layoutMode === 'sidebyside' ? 'flex-start' : undefined,
      }}>
        <div style={{
          width: isMobile ? '100%' : layoutMode === 'editor' ? '90vw' : undefined,
          flex: !isMobile && layoutMode === 'sidebyside' ? '1 1 50%' : undefined,
          minWidth: !isMobile && layoutMode === 'sidebyside' ? 0 : undefined,
          maxWidth: !isMobile && layoutMode === 'sidebyside' ? '60%' : undefined,
          position: !isMobile && layoutMode === 'sidebyside' ? 'sticky' : 'relative',
          top: !isMobile && layoutMode === 'sidebyside' ? 12 : undefined,
          margin: isMobile || layoutMode === 'sidebyside' ? undefined : '0 auto',
        }}>
          <EditorErrorBoundary
            name="Video editor"
            fallbackHint="If this keeps failing, refresh the page or pick a different clip from the Viral Clips tab."
          >
          <VideoEditor
            src={videoSrc}
            clipStart={startTime ?? clip.start_time}
            clipEnd={endTime ?? clip.end_time}
            title={clip.title || `Clip ${clipId}`}
            aspectRatio={aspectRatio}
            sourceWidth={sourceDims.w}
            sourceHeight={sourceDims.h}
            subjectX={clipSubjectX}
            scenes={stableScenes}
            sceneCuts={stableSceneCuts}
            subjectTrack={stableSubjectTrack}
            initialVolume={playbackVolume}
            initialSpeed={playbackSpeed}
            onTimeUpdate={setCurrentTime}
            onTrimChange={setEditorTrim}
            onVolumeChange={setEditorVolume}
            onSpeedChange={setEditorSpeed}
            onSegmentsChange={saveSegments}
            initialSegments={editorSegments}
            settings={clipSettings}
            speakers={speakers}
            speakerNames={stableSpeakerNames}
            onSettingsChange={setClipSettings}
            jobId={jobId}
            clipId={clipId}
            transcript={stableTranscript}
            onTranscriptUpdated={fetchJob}
            onVideoRef={handleVideoRef}
            renderPlan={renderPlan}
            onAspectRatioChange={(ar) => setClipSettings((prev) => ({ ...prev, aspectRatio: ar }))}
            subtitleOverlay={
              <SubtitleOverlay
                currentTime={currentTime}
                transcript={stableTranscript}
                clipStart={startTime ?? clip.start_time}
                clipEnd={endTime ?? clip.end_time}
                settings={clipSettings}
                aspectRatio={aspectRatio}
                sourceWidth={sourceDims.w}
                sourceHeight={sourceDims.h}
                segments={editorSegments}
              />
            }
            compact={!isMobile && layoutMode === 'sidebyside'}
          />
          </EditorErrorBoundary>

            {/* ── Editor mode toolbar: full export + subtitle settings ── */}
            <div style={{
              display: (isMobile || layoutMode === 'editor') ? 'flex' : 'none',
              alignItems: 'center', gap: 8, padding: '8px 12px',
              background: 'var(--bg-panel)', borderRadius: '0 0 var(--radius-md) var(--radius-md)',
              borderTop: '1px solid var(--border)', flexWrap: 'wrap',
              marginTop: -1,
            }}>
              {/* Export */}
              {exporting ? (
                <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                  <div style={{
                    width: 14, height: 14,
                    border: '2px solid var(--border)', borderTop: '2px solid var(--accent-cyan)',
                    borderRadius: '50%', animation: 'seo-spin 1s linear infinite',
                  }} />
                  <span style={{ fontSize: 11, color: 'var(--accent-cyan)', fontFamily: 'var(--font-mono)' }}>
                    {exportProgress || 'Exporting...'}
                  </span>
                </div>
              ) : (
                <button
                  onClick={handleExport}
                  style={{
                    display: 'flex', alignItems: 'center', gap: 5,
                    padding: '6px 14px', fontSize: 11, fontWeight: 700,
                    background: 'var(--accent-cyan)', color: '#fff',
                    border: 'none', borderRadius: 'var(--radius-sm)',
                    cursor: 'pointer', whiteSpace: 'nowrap',
                  }}
                >
                  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4" /><polyline points="7 10 12 15 17 10" /><line x1="12" y1="15" x2="12" y2="3" />
                  </svg>
                  Export {exportQuality || '1080p'}
                </button>
              )}
              {downloadUrl && (
                <a href={downloadUrl} download style={{
                  padding: '6px 12px', fontSize: 11, fontWeight: 600,
                  background: 'var(--bg-elevated)', color: 'var(--accent-cyan)',
                  border: '1px solid var(--accent-cyan)', borderRadius: 'var(--radius-sm)',
                  textDecoration: 'none', whiteSpace: 'nowrap',
                }}>
                  Download
                </a>
              )}

              <div style={{ width: 1, height: 20, background: 'var(--border)', margin: '0 4px' }} />

              {/* Subtitles toggle — segment-aware */}
              <button
                onClick={handleSubsToggle}
                style={{
                  display: 'flex', alignItems: 'center', gap: 4,
                  padding: '5px 10px', fontSize: 11, fontWeight: 600,
                  background: effectiveSubsEnabled ? 'var(--accent-cyan-dim)' : 'var(--bg-elevated)',
                  color: effectiveSubsEnabled ? 'var(--accent-cyan)' : 'var(--text-secondary)',
                  border: `1px solid ${effectiveSubsEnabled ? 'var(--accent-cyan)' : 'var(--border)'}`,
                  borderRadius: 'var(--radius-sm)', cursor: 'pointer', whiteSpace: 'nowrap',
                }}
              >
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <rect x="1" y="4" width="22" height="16" rx="2" /><line x1="1" y1="14" x2="23" y2="14" />
                </svg>
                {activeSegment ? 'Segment ' : ''}Subs {effectiveSubsEnabled ? 'On' : 'Off'}
              </button>

              {/* Subtitle settings expand */}
              <button
                onClick={() => setShowInlineSubSettings(v => !v)}
                style={{
                  display: 'flex', alignItems: 'center', gap: 3,
                  padding: '5px 8px', fontSize: 10, fontWeight: 500,
                  background: showInlineSubSettings ? 'var(--accent-cyan-dim)' : 'transparent',
                  color: 'var(--text-muted)',
                  border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)',
                  cursor: 'pointer', whiteSpace: 'nowrap',
                }}
              >
                <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <circle cx="12" cy="12" r="3" /><path d="M19.4 15a1.65 1.65 0 00.33 1.82l.06.06a2 2 0 010 2.83 2 2 0 01-2.83 0l-.06-.06a1.65 1.65 0 00-1.82-.33 1.65 1.65 0 00-1 1.51V21a2 2 0 01-4 0v-.09A1.65 1.65 0 009 19.4a1.65 1.65 0 00-1.82.33l-.06.06a2 2 0 01-2.83-2.83l.06-.06A1.65 1.65 0 004.68 15a1.65 1.65 0 00-1.51-1H3a2 2 0 010-4h.09A1.65 1.65 0 004.6 9a1.65 1.65 0 00-.33-1.82l-.06-.06a2 2 0 012.83-2.83l.06.06A1.65 1.65 0 009 4.68a1.65 1.65 0 001-1.51V3a2 2 0 014 0v.09a1.65 1.65 0 001 1.51 1.65 1.65 0 001.82-.33l.06-.06a2 2 0 012.83 2.83l-.06.06A1.65 1.65 0 0019.4 9a1.65 1.65 0 001.51 1H21a2 2 0 010 4h-.09a1.65 1.65 0 00-1.51 1z" />
                </svg>
                {showInlineSubSettings ? 'Hide' : 'Settings'}
              </button>

              {/* Meta info */}
              <span style={{ marginLeft: 'auto', fontSize: 10, fontFamily: 'var(--font-mono)', color: 'var(--text-muted)', whiteSpace: 'nowrap' }}>
                {formatDuration(startTime)} - {formatDuration(endTime)} &middot; {aspectRatio || 'original'}
              </span>
            </div>

            {/* ── Collapsible inline subtitle settings (editor mode only) ── */}
            {(isMobile || layoutMode === 'editor') && showInlineSubSettings && (
              <div style={{
                padding: '12px 16px', background: 'var(--bg-panel)',
                border: '1px solid var(--border)', borderTop: 'none',
                borderRadius: '0 0 var(--radius-md) var(--radius-md)',
                display: 'flex', flexWrap: 'wrap', gap: 12, alignItems: 'flex-end',
              }}>
                {/* Font */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: 3, minWidth: 120 }}>
                  <label style={{ fontSize: 10, fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>Font</label>
                  <select
                    value={subtitleFont}
                    onChange={e => setClipSettings(prev => ({ ...prev, subtitleFont: e.target.value }))}
                    style={{ padding: '5px 8px', fontSize: 12, borderRadius: 'var(--radius-xs)', border: '1px solid var(--border)', background: 'var(--bg-elevated)', color: 'var(--text-primary)' }}
                  >
                    {['DM Sans', 'Montserrat', 'Open Sans', 'Roboto', 'Poppins', 'Inter', 'Nunito', 'Lato', 'Oswald', 'Playfair Display', 'Bebas Neue'].map(f => (
                      <option key={f} value={f}>{f}</option>
                    ))}
                    {customFonts.length > 0 && (
                      <optgroup label="Custom Fonts">
                        {customFonts.map(f => <option key={f.name} value={f.name}>{f.name}</option>)}
                      </optgroup>
                    )}
                  </select>
                </div>
                {/* Size */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
                  <label style={{ fontSize: 10, fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>Size</label>
                  <div style={{ display: 'flex', gap: 2 }}>
                    {[{ l: 'S', v: 22 }, { l: 'M', v: 30 }, { l: 'L', v: 40 }].map(s => (
                      <button key={s.v} onClick={() => setClipSettings(prev => ({ ...prev, subtitleSize: s.v }))}
                        style={{
                          padding: '4px 10px', fontSize: 11, fontWeight: 600,
                          background: subtitleSize === s.v ? 'var(--accent-cyan)' : 'var(--bg-elevated)',
                          color: subtitleSize === s.v ? '#fff' : 'var(--text-secondary)',
                          border: '1px solid var(--border)', borderRadius: 'var(--radius-xs)', cursor: 'pointer',
                        }}>{s.l}</button>
                    ))}
                  </div>
                </div>
                {/* Weight */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
                  <label style={{ fontSize: 10, fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>Weight</label>
                  <div style={{ display: 'flex', gap: 2 }}>
                    {[{ v: 400, l: 'Regular' }, { v: 700, l: 'Bold' }, { v: 900, l: 'Black' }].map(w => (
                      <button key={w.v} onClick={() => setClipSettings(prev => ({ ...prev, subtitleFontWeight: w.v }))}
                        style={{
                          padding: '4px 8px', fontSize: 10, fontWeight: w.v,
                          background: (typeof subtitleFontWeight === 'number' ? subtitleFontWeight : (subtitleFontWeight === 'bold' ? 700 : subtitleFontWeight === 'black' ? 900 : 400)) === w.v ? 'var(--accent-cyan)' : 'var(--bg-elevated)',
                          color: (typeof subtitleFontWeight === 'number' ? subtitleFontWeight : (subtitleFontWeight === 'bold' ? 700 : subtitleFontWeight === 'black' ? 900 : 400)) === w.v ? '#fff' : 'var(--text-secondary)',
                          border: '1px solid var(--border)', borderRadius: 'var(--radius-xs)', cursor: 'pointer',
                        }}>{w.l}</button>
                    ))}
                  </div>
                </div>
                {/* Color */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
                  <label style={{ fontSize: 10, fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>Color</label>
                  <input type="color" value={subtitleFontColor} onChange={e => setClipSettings(prev => ({ ...prev, subtitleFontColor: e.target.value }))}
                    style={{ width: 32, height: 28, border: '1px solid var(--border)', borderRadius: 'var(--radius-xs)', cursor: 'pointer', padding: 1 }} />
                </div>
                {/* Position */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
                  <label style={{ fontSize: 10, fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>Position</label>
                  <div style={{ display: 'flex', gap: 2 }}>
                    {['top', 'center', 'bottom'].map(p => (
                      <button key={p} onClick={() => setClipSettings(prev => ({ ...prev, subtitlePosition: p }))}
                        style={{
                          padding: '4px 8px', fontSize: 10, fontWeight: 600,
                          background: subtitlePosition === p ? 'var(--accent-cyan)' : 'var(--bg-elevated)',
                          color: subtitlePosition === p ? '#fff' : 'var(--text-secondary)',
                          border: '1px solid var(--border)', borderRadius: 'var(--radius-xs)', cursor: 'pointer', textTransform: 'capitalize',
                        }}>{p}</button>
                    ))}
                  </div>
                </div>
                {/* Outline width */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: 3, minWidth: 80 }}>
                  <label style={{ fontSize: 10, fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>Outline</label>
                  <input type="range" min="0" max="10" step="1" value={subtitleOutlineWidth}
                    onChange={e => setClipSettings(prev => ({ ...prev, subtitleOutlineWidth: parseInt(e.target.value) }))}
                    style={{ width: 80, accentColor: 'var(--accent-cyan)' }} />
                </div>
              </div>
            )}

            {/* ── Compact toolbar (side-by-side mode only) ── */}
            <div style={{
              display: !isMobile && layoutMode === 'sidebyside' ? 'flex' : 'none',
              alignItems: 'center', gap: 6, padding: '6px 10px',
              background: 'var(--bg-panel)', borderRadius: '0 0 var(--radius-sm) var(--radius-sm)',
              borderTop: '1px solid var(--border)', flexWrap: 'wrap', marginTop: -1,
            }}>
              {exporting ? (
                <span style={{ fontSize: 10, color: 'var(--accent-cyan)', fontFamily: 'var(--font-mono)' }}>Exporting...</span>
              ) : (
                <button onClick={handleExport} style={{
                  padding: '4px 10px', fontSize: 10, fontWeight: 700,
                  background: 'var(--accent-cyan)', color: '#fff',
                  border: 'none', borderRadius: 'var(--radius-xs)', cursor: 'pointer',
                }}>Export {exportQuality || '1080p'}</button>
              )}
              <button
                onClick={handleSubsToggle}
                style={{
                  padding: '4px 8px', fontSize: 10, fontWeight: 600,
                  background: effectiveSubsEnabled ? 'var(--accent-cyan-dim)' : 'var(--bg-elevated)',
                  color: effectiveSubsEnabled ? 'var(--accent-cyan)' : 'var(--text-muted)',
                  border: `1px solid ${effectiveSubsEnabled ? 'var(--accent-cyan)' : 'var(--border)'}`,
                  borderRadius: 'var(--radius-xs)', cursor: 'pointer',
                }}>{activeSegment ? 'Seg ' : ''}Subs {effectiveSubsEnabled ? 'On' : 'Off'}</button>
            </div>
          </div>
          {/* ── Side-by-side transcript panel ── */}
          {!isMobile && layoutMode === 'sidebyside' && (
            <div style={{ flex: '1 1 50%', minWidth: 0 }}>
              {job?.transcript?.length > 0 && clipTimeRange ? (
                <div>
                  <div style={{ fontSize: 11, fontFamily: 'var(--font-mono)', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: 8 }}>
                    Clip Transcript
                  </div>
                  <div style={{ ...sectionStyle, padding: '8px 10px' }}>
                    <TranscriptViewer
                      transcript={job.transcript}
                      timeRange={clipTimeRange}
                      currentTime={currentTime}
                      maxHeight={600}
                      speakerColors={speakerColors}
                      onSpeakerColorChanged={handleSpeakerColorChanged}
                      onSpeakerAdded={handleSpeakerAdded}
                      onSeek={(time) => {
                        const video = videoRef.current;
                        if (video) {
                          video.currentTime = time;
                          setCurrentTime(time);
                        }
                      }}
                      jobId={jobId}
                      onSpeakerRenamed={fetchJob}
                      onTranscriptUpdated={fetchJob}
                    />
                  </div>
                </div>
              ) : (
                <div style={{ padding: '40px 20px', textAlign: 'center', color: 'var(--text-muted)', fontSize: 13 }}>
                  No transcript available for this clip
                </div>
              )}
            </div>
          )}
        </div>

      <div className="clip-panel-layout" style={{ display: 'flex', gap: 20, alignItems: 'flex-start', flexWrap: isMobile ? 'wrap' : 'nowrap', flexDirection: isMobile ? 'column' : 'row' }}>
        {/* Left column: clip info + export settings */}
        <div className="clip-settings-sidebar" style={{ width: isMobile ? '100%' : 380, position: isMobile ? 'static' : 'sticky', top: 20, alignSelf: 'flex-start', maxHeight: isMobile ? 'none' : 'calc(100vh - 40px)', overflowY: isMobile ? 'visible' : 'auto' }}>

          {/* Clip info */}
          <div style={sectionStyle}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
              <h3 style={{ fontSize: 16, margin: 0 }}>{String(clip.title || '')}</h3>
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: 22, fontWeight: 700, color: scoreColor }}>
                {clip.viral_score}<span style={{ fontSize: 10, color: 'var(--text-muted)' }}>/100</span>
              </div>
            </div>
            <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginBottom: 8 }}>
              <span className={`badge ${clip.platform === 'tiktok' ? 'badge-cyan' : clip.platform === 'youtube_shorts' ? 'badge-red' : 'badge-gray'}`}>
                {String(clip.platform || '').replace('_', ' ')}
              </span>
              <span className="badge badge-gray">{String(clip.clip_type || '')}</span>
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--text-muted)' }}>
                {formatDuration(clipDur)}
              </span>
            </div>
            {clip.hook_text && (
              <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginBottom: 4 }}>
                <strong style={{ color: 'var(--text-primary)' }}>Hook:</strong> {String(clip.hook_text || '')}
              </div>
            )}
            {clip.why_this_works && (
              <div style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
                <strong style={{ color: 'var(--text-primary)' }}>Why it works:</strong> {String(clip.why_this_works || '')}
              </div>
            )}
          </div>

          {/* Trim Controls */}
          <div style={sectionStyle}>
            <div style={{ fontSize: 13, fontFamily: 'var(--font-mono)', fontWeight: 700, color: 'var(--text-primary)', marginBottom: 10, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
              Trim
            </div>
            <div style={{ display: 'flex', gap: 12 }}>
              <div style={{ flex: 1 }}>
                <div style={{ fontSize: 10, color: 'var(--text-muted)', marginBottom: 3 }}>Start</div>
                <input
                  type="text"
                  value={startText}
                  onChange={(e) => setStartText(e.target.value)}
                  onBlur={() => {
                    const val = parseDuration(startText);
                    if (val !== null && val >= 0 && val < (endTime ?? clip.end_time)) {
                      setStartTime(val);
                      setStartText(formatDuration(val));
                      if (videoRef.current) videoRef.current.currentTime = val;
                    } else {
                      setStartText(formatDuration(startTime));
                    }
                  }}
                  onKeyDown={(e) => { if (e.key === 'Enter') e.target.blur(); }}
                  style={timeInput}
                />
              </div>
              <div style={{ flex: 1 }}>
                <div style={{ fontSize: 10, color: 'var(--text-muted)', marginBottom: 3 }}>End</div>
                <input
                  type="text"
                  value={endText}
                  onChange={(e) => setEndText(e.target.value)}
                  onBlur={() => {
                    const val = parseDuration(endText);
                    if (val !== null && val > (startTime ?? 0)) {
                      setEndTime(val);
                      setEndText(formatDuration(val));
                    } else {
                      setEndText(formatDuration(endTime));
                    }
                  }}
                  onKeyDown={(e) => { if (e.key === 'Enter') e.target.blur(); }}
                  style={timeInput}
                />
              </div>
              <div style={{ display: 'flex', alignItems: 'flex-end' }}>
                <button
                  onClick={() => {
                    setStartTime(clip.start_time);
                    setEndTime(clip.end_time);
                    setStartText(formatDuration(clip.start_time));
                    setEndText(formatDuration(clip.end_time));
                    if (videoRef.current) videoRef.current.currentTime = clip.start_time;
                  }}
                  title="Reset to original clip times"
                  style={{
                    padding: '5px 8px', fontSize: 10,
                    background: 'var(--bg-elevated)', color: 'var(--text-muted)',
                    border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)',
                    cursor: 'pointer', whiteSpace: 'nowrap',
                  }}
                >
                  Reset
                </button>
              </div>
            </div>
          </div>


          {/* Clip Settings Panel */}
          <div style={{ marginBottom: 16 }}>
            <EditorErrorBoundary name="Clip settings" compact>
              <ClipSettingsPanel
                speakers={speakers}
                speakerNames={job.speaker_names}
                videoResolution={job.resolution}
                onSettingsChange={setClipSettings}
                onApplySettings={handleApplySettings}
                serverSettings={job?.subtitle_settings}
                parentSettings={clipSettings}
              />
            </EditorErrorBoundary>
          </div>

          {/* Speaker Renaming */}
          {speakers.length > 0 && (
            <div style={sectionStyle}>
              <div style={{ fontSize: 13, fontFamily: 'var(--font-mono)', fontWeight: 700, color: 'var(--text-primary)', marginBottom: 10, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                Rename Speakers
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                {speakers.map((sp) => {
                  const isEditing = editingSpeaker === sp;
                  return (
                    <div key={sp} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                      {isEditing ? (
                        <input
                          type="text"
                          value={editSpeakerValue}
                          onChange={(e) => setEditSpeakerValue(e.target.value)}
                          onBlur={() => handleRenameSpeaker(sp, editSpeakerValue)}
                          onKeyDown={(e) => {
                            if (e.key === 'Enter') handleRenameSpeaker(sp, editSpeakerValue);
                            if (e.key === 'Escape') setEditingSpeaker(null);
                          }}
                          autoFocus
                          style={{
                            flex: 1, fontSize: 13, color: 'var(--text-primary)',
                            background: 'var(--bg-elevated)', border: '1px solid var(--border)',
                            borderRadius: 'var(--radius-sm)', padding: '4px 8px', outline: 'none',
                            fontFamily: 'var(--font-mono)',
                          }}
                        />
                      ) : (
                        <span
                          onClick={() => { setEditingSpeaker(sp); setEditSpeakerValue(sp); }}
                          style={{ fontSize: 13, color: 'var(--text-primary)', cursor: 'pointer', fontFamily: 'var(--font-mono)' }}
                          title="Click to rename"
                        >
                          {sp}
                        </span>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Export Button */}
          <div style={sectionStyle}>
            {exporting ? (
              <div style={{ textAlign: 'center', padding: '8px 0' }}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 10 }}>
                  <div style={{
                    width: 20, height: 20,
                    border: '2px solid var(--border)',
                    borderTop: '2px solid var(--accent-cyan)',
                    borderRadius: '50%',
                    animation: 'seo-spin 1s linear infinite',
                  }} />
                  <span style={{ fontSize: 12, color: 'var(--accent-cyan)', fontFamily: 'var(--font-mono)' }}>
                    {exportProgress || 'Exporting...'}
                  </span>
                </div>
              </div>
            ) : (
              <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                <button
                  onClick={handleExport}
                  style={{
                    flex: 1, padding: '10px 16px',
                    background: 'var(--accent-cyan)', color: 'var(--bg-base)',
                    border: 'none', borderRadius: 'var(--radius-sm)',
                    fontSize: 13, fontWeight: 700, cursor: 'pointer',
                  }}
                >
                  Export MP4
                </button>
                {downloadUrl && (
                  <a
                    href={downloadUrl}
                    download
                    style={{
                      padding: '10px 16px',
                      background: 'var(--bg-elevated)', color: 'var(--accent-cyan)',
                      border: '1px solid var(--accent-cyan)', borderRadius: 'var(--radius-sm)',
                      fontSize: 12, fontWeight: 600, textDecoration: 'none',
                      display: 'flex', alignItems: 'center',
                    }}
                  >
                    Download Again
                  </a>
                )}
              </div>
            )}

            <div style={{ marginTop: 8, fontSize: 10, fontFamily: 'var(--font-mono)', color: 'var(--text-muted)', display: 'flex', gap: 10, flexWrap: 'wrap', alignItems: 'center' }}>
              <span>{formatDuration(startTime)} - {formatDuration(endTime)}</span>
              <span>{aspectRatio || 'original'}</span>
              {subtitlesEnabled && <span style={{ color: 'var(--accent-cyan)' }}>subs: {subtitleFont} / {subtitleSize} / {subtitleFontWeight} / {subtitlePosition} / {subtitleMaxWidth}%w{subtitleMaxWords > 0 ? ` / ${subtitleMaxWords}w` : ''}</span>}
              {settingsAppliedFlash && (
                <span style={{
                  color: '#10B981',
                  fontWeight: 600,
                  animation: 'seo-pulse 1.5s ease-in-out',
                }}>
                  Settings applied
                </span>
              )}
            </div>
          </div>

        </div>

        {/* Right column: SEO content */}
        <div style={{ flex: '1 1 auto', minWidth: 0, overflowY: 'auto', maxHeight: isMobile ? 'none' : 'calc(100vh - 100px)' }}>
          {/* ── Clip Transcript — hidden in sidebyside mode where it's shown above ─── */}
          {layoutMode !== 'sidebyside' && job?.transcript?.length > 0 && clipTimeRange && (
            <div style={{ marginBottom: 16 }}>
              <div style={{ fontSize: 11, fontFamily: 'var(--font-mono)', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: 8 }}>
                Clip Transcript
              </div>
              <div style={{ ...sectionStyle, padding: '8px 10px' }}>
                <TranscriptViewer
                  transcript={job.transcript}
                  timeRange={clipTimeRange}
                  currentTime={currentTime}
                  maxHeight={400}
                  speakerColors={speakerColors}
                  onSpeakerColorChanged={handleSpeakerColorChanged}
                  onSpeakerAdded={handleSpeakerAdded}
                  onSeek={(time) => {
                    const video = videoRef.current;
                    if (video) {
                      video.currentTime = time;
                      setCurrentTime(time);
                    }
                  }}
                  jobId={jobId}
                  onSpeakerRenamed={fetchJob}
                  onTranscriptUpdated={fetchJob}
                />
              </div>
            </div>
          )}
          {/* Share button lives in the page header (top-right) and
              copies a token-based public URL on click. The previous
              "Copy Share Link" button here generated the legacy
              ``/share/clip/<jobId>/<clipId>`` route which the public
              SharedView never knew how to load — duplicating that
              button confused users into copying broken URLs. */}
          {!seo ? (
            <div style={{ ...sectionStyle, textAlign: 'center', padding: '48px 24px' }}>
              {generating ? (
                <>
                  <div style={{
                    width: 48, height: 48, margin: '0 auto 16px',
                    border: '3px solid var(--border)',
                    borderTop: '3px solid var(--accent-cyan)',
                    borderRadius: '50%',
                    animation: 'seo-spin 1s linear infinite',
                  }} />
                  <h3 style={{ fontSize: 16, marginBottom: 8, color: 'var(--text-primary)' }}>
                    Generating SEO...
                  </h3>
                  <p style={{
                    fontSize: 12, color: 'var(--accent-cyan)',
                    animation: 'seo-pulse 2s ease-in-out infinite',
                    fontFamily: 'var(--font-mono)',
                  }}>
                    {String(genStatus ?? '')}
                  </p>
                  <p style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 12, fontFamily: 'var(--font-mono)' }}>
                    {genElapsed}s elapsed
                  </p>
                </>
              ) : (
                <>
                  <div style={{ fontSize: 36, marginBottom: 16, opacity: 0.3 }}>&#128269;</div>
                  <h3 style={{ fontSize: 16, marginBottom: 8, color: 'var(--text-secondary)' }}>
                    Generate SEO Metadata
                  </h3>
                  <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 20, maxWidth: 360, margin: '0 auto 20px' }}>
                    Use AI to generate a title, caption, and tags for this clip — written like a real person would post it
                  </p>
                  <button
                    onClick={generateSEO}
                    style={{
                      padding: '10px 24px',
                      background: 'var(--accent-cyan)', color: 'var(--bg-base)',
                      border: 'none', borderRadius: 'var(--radius-sm)',
                      fontSize: 13, fontWeight: 600, cursor: 'pointer',
                    }}
                  >
                    Generate SEO
                  </button>
                </>
              )}
            </div>
          ) : (
            <>
              {/* Copy All — ready to paste */}
              <div style={{ ...sectionStyle, background: 'var(--accent-cyan-dim)', borderColor: 'var(--accent-cyan)' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--accent-cyan)' }}>
                    Ready to post
                  </div>
                  <button
                    onClick={() => copyToClipboard(
                      `${seo.title}\n\n${seo.description}\n\n${(seo.tags || []).join(' ')}`,
                      'all'
                    )}
                    style={{
                      padding: '6px 14px', fontSize: 11, fontWeight: 600,
                      background: copied === 'all' ? 'var(--accent-cyan)' : 'var(--bg-panel)',
                      color: copied === 'all' ? 'var(--bg-base)' : 'var(--accent-cyan)',
                      border: '1px solid var(--accent-cyan)', borderRadius: 'var(--radius-sm)', cursor: 'pointer',
                    }}
                  >
                    {copied === 'all' ? 'Copied!' : 'Copy All'}
                  </button>
                </div>
              </div>

              {/* Title */}
              <div style={sectionStyle}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                  <div style={{ fontSize: 10, fontFamily: 'var(--font-mono)', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.1em' }}>
                    Title
                  </div>
                  <button onClick={() => copyToClipboard(seo.title, 'title')} style={copyBtnStyle('title')}>
                    {copied === 'title' ? 'Copied' : 'Copy'}
                  </button>
                </div>
                <div style={{ fontSize: 16, fontWeight: 600, color: 'var(--text-primary)', lineHeight: 1.4 }}>
                  {String(seo.title ?? '')}
                </div>
              </div>

              {/* Caption */}
              <div style={sectionStyle}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                  <div style={{ fontSize: 10, fontFamily: 'var(--font-mono)', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.1em' }}>
                    Caption
                  </div>
                  <button onClick={() => copyToClipboard(seo.description, 'desc')} style={copyBtnStyle('desc')}>
                    {copied === 'desc' ? 'Copied' : 'Copy'}
                  </button>
                </div>
                <div style={{ fontSize: 13, color: 'var(--text-secondary)', lineHeight: 1.5 }}>
                  {String(seo.description ?? '')}
                </div>
              </div>

              {/* Tags */}
              <div style={sectionStyle}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                  <div style={{ fontSize: 10, fontFamily: 'var(--font-mono)', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.1em' }}>
                    Tags
                  </div>
                  <button onClick={() => copyToClipboard((seo.tags || []).join(' '), 'tags')} style={copyBtnStyle('tags')}>
                    {copied === 'tags' ? 'Copied' : 'Copy All'}
                  </button>
                </div>
                <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                  {(seo.tags || []).map((tag, i) => (
                    <span
                      key={i}
                      onClick={() => copyToClipboard(tag, `tag-${i}`)}
                      style={{
                        padding: '4px 10px',
                        background: 'var(--bg-elevated)',
                        border: '1px solid var(--border)',
                        borderRadius: 12, fontSize: 12,
                        color: 'var(--accent-cyan)', cursor: 'pointer',
                      }}
                    >
                      {String(tag ?? '')}
                    </span>
                  ))}
                </div>
              </div>

              {/* Platform Tips */}
              {seo.platform_tips && (
                <div style={{ ...sectionStyle, background: 'var(--bg-elevated)', borderStyle: 'dashed' }}>
                  <div style={{ fontSize: 12, color: 'var(--text-secondary)', lineHeight: 1.5, fontStyle: 'italic' }}>
                    {String(seo.platform_tips ?? '')}
                  </div>
                </div>
              )}

              {/* Regenerate with inline status */}
              {generating ? (
                <div style={{
                  ...sectionStyle,
                  borderColor: 'var(--accent-cyan)',
                  display: 'flex', alignItems: 'center', gap: 12,
                  padding: '12px 16px',
                }}>
                  <div style={{
                    width: 18, height: 18, flexShrink: 0,
                    border: '2px solid var(--border)',
                    borderTop: '2px solid var(--accent-cyan)',
                    borderRadius: '50%',
                    animation: 'seo-spin 1s linear infinite',
                  }} />
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{
                      fontSize: 12, color: 'var(--accent-cyan)',
                      fontFamily: 'var(--font-mono)',
                      animation: 'seo-pulse 2s ease-in-out infinite',
                      whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
                    }}>
                      {String(genStatus ?? '')}
                    </div>
                  </div>
                  <span style={{ fontSize: 10, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', flexShrink: 0 }}>
                    {genElapsed}s
                  </span>
                </div>
              ) : (
                <button
                  onClick={generateSEO}
                  style={{
                    padding: '8px 16px',
                    background: 'var(--bg-elevated)', color: 'var(--text-secondary)',
                    border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)',
                    fontSize: 12, cursor: 'pointer',
                  }}
                >
                  Regenerate
                </button>
              )}
            </>
          )}

          {/* ── YouTube Description Generators ─────────────────────── */}
          <div style={{ marginTop: 24 }} id="youtube-descriptions">
            <div style={{ fontSize: 12, fontFamily: 'var(--font-mono)', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: 12 }}>
              YouTube Description Generators
            </div>

            {/* YouTube Shorts Description */}
            <div style={sectionStyle}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-primary)' }}>
                  YouTube Shorts Description
                </div>
                <div style={{ display: 'flex', gap: 6 }}>
                  {shortsDesc && (
                    <button
                      onClick={() => copyToClipboard(shortsDesc, 'shorts')}
                      style={{
                        padding: '4px 10px', fontSize: 11, fontWeight: 600,
                        background: copied === 'shorts' ? 'var(--accent-cyan)' : 'var(--bg-panel)',
                        color: copied === 'shorts' ? 'var(--bg-base)' : 'var(--accent-cyan)',
                        border: '1px solid var(--accent-cyan)', borderRadius: 'var(--radius-sm)', cursor: 'pointer',
                      }}
                    >
                      {copied === 'shorts' ? 'Copied!' : 'Copy'}
                    </button>
                  )}
                  <button
                    onClick={() => generateDescription('shorts')}
                    disabled={generatingShorts}
                    style={{
                      padding: '4px 10px', fontSize: 11, fontWeight: 600,
                      background: generatingShorts ? 'var(--bg-elevated)' : 'var(--accent-cyan)',
                      color: generatingShorts ? 'var(--text-muted)' : 'var(--bg-base)',
                      border: 'none', borderRadius: 'var(--radius-sm)',
                      cursor: generatingShorts ? 'default' : 'pointer',
                    }}
                  >
                    {generatingShorts ? `${shortsElapsed}s...` : (shortsDesc ? 'Regenerate' : 'Generate')}
                  </button>
                </div>
              </div>
              <p style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 8 }}>
                Optimized for YouTube Shorts SEO — hook line, keywords, hashtags, CTA. Under 500 characters.
              </p>
              {generatingShorts ? (
                <div style={{
                  padding: '16px', borderRadius: 'var(--radius-sm)',
                  background: 'var(--bg-elevated)', border: '1px solid var(--accent-cyan)',
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8 }}>
                    <div style={{
                      width: 18, height: 18, flexShrink: 0,
                      border: '2px solid var(--border)',
                      borderTop: '2px solid var(--accent-cyan)',
                      borderRadius: '50%',
                      animation: 'seo-spin 1s linear infinite',
                    }} />
                    <span style={{
                      fontSize: 12, color: 'var(--accent-cyan)',
                      fontFamily: 'var(--font-mono)',
                      animation: 'seo-pulse 2s ease-in-out infinite',
                    }}>
                      {descStatusMessage(shortsElapsed)}
                    </span>
                    <span style={{ fontSize: 10, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', marginLeft: 'auto', flexShrink: 0 }}>
                      {shortsElapsed}s
                    </span>
                  </div>
                  <div style={{
                    height: 3, borderRadius: 2, background: 'var(--border)', overflow: 'hidden',
                  }}>
                    <div style={{
                      height: '100%', borderRadius: 2,
                      background: 'var(--accent-cyan)',
                      width: `${Math.min(95, shortsElapsed * 2.5)}%`,
                      transition: 'width 1s linear',
                    }} />
                  </div>
                </div>
              ) : shortsDesc ? (
                <textarea
                  value={shortsDesc}
                  onChange={(e) => { setShortsDesc(e.target.value); saveClipSeo({ shorts_description: e.target.value }); }}
                  style={{
                    width: '100%', minHeight: 120, padding: 10,
                    background: 'var(--bg-elevated)', color: 'var(--text-primary)',
                    border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)',
                    fontSize: 13, lineHeight: 1.5, resize: 'vertical',
                    fontFamily: 'var(--font-sans)',
                  }}
                />
              ) : (
                <div style={{ padding: '20px 0', textAlign: 'center', color: 'var(--text-muted)', fontSize: 12 }}>
                  Click "Generate" to create a Shorts-optimized description
                </div>
              )}
            </div>

            {/* YouTube Long-Form Description */}
            <div style={sectionStyle}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-primary)' }}>
                  YouTube Description
                </div>
                <div style={{ display: 'flex', gap: 6 }}>
                  {longFormDesc && (
                    <button
                      onClick={() => copyToClipboard(longFormDesc, 'longform')}
                      style={{
                        padding: '4px 10px', fontSize: 11, fontWeight: 600,
                        background: copied === 'longform' ? 'var(--accent-cyan)' : 'var(--bg-panel)',
                        color: copied === 'longform' ? 'var(--bg-base)' : 'var(--accent-cyan)',
                        border: '1px solid var(--accent-cyan)', borderRadius: 'var(--radius-sm)', cursor: 'pointer',
                      }}
                    >
                      {copied === 'longform' ? 'Copied!' : 'Copy'}
                    </button>
                  )}
                  <button
                    onClick={() => generateDescription('long_form')}
                    disabled={generatingLongForm}
                    style={{
                      padding: '4px 10px', fontSize: 11, fontWeight: 600,
                      background: generatingLongForm ? 'var(--bg-elevated)' : 'var(--accent-cyan)',
                      color: generatingLongForm ? 'var(--text-muted)' : 'var(--bg-base)',
                      border: 'none', borderRadius: 'var(--radius-sm)',
                      cursor: generatingLongForm ? 'default' : 'pointer',
                    }}
                  >
                    {generatingLongForm ? `${longFormElapsed}s...` : (longFormDesc ? 'Regenerate' : 'Generate')}
                  </button>
                </div>
              </div>
              <p style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 8 }}>
                Full YouTube description with keywords, timestamps, hashtags, and social links. 500-2000 characters.
              </p>
              {generatingLongForm ? (
                <div style={{
                  padding: '16px', borderRadius: 'var(--radius-sm)',
                  background: 'var(--bg-elevated)', border: '1px solid var(--accent-cyan)',
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8 }}>
                    <div style={{
                      width: 18, height: 18, flexShrink: 0,
                      border: '2px solid var(--border)',
                      borderTop: '2px solid var(--accent-cyan)',
                      borderRadius: '50%',
                      animation: 'seo-spin 1s linear infinite',
                    }} />
                    <span style={{
                      fontSize: 12, color: 'var(--accent-cyan)',
                      fontFamily: 'var(--font-mono)',
                      animation: 'seo-pulse 2s ease-in-out infinite',
                    }}>
                      {descStatusMessage(longFormElapsed)}
                    </span>
                    <span style={{ fontSize: 10, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', marginLeft: 'auto', flexShrink: 0 }}>
                      {longFormElapsed}s
                    </span>
                  </div>
                  <div style={{
                    height: 3, borderRadius: 2, background: 'var(--border)', overflow: 'hidden',
                  }}>
                    <div style={{
                      height: '100%', borderRadius: 2,
                      background: 'var(--accent-cyan)',
                      width: `${Math.min(95, longFormElapsed * 2.5)}%`,
                      transition: 'width 1s linear',
                    }} />
                  </div>
                </div>
              ) : longFormDesc ? (
                <textarea
                  value={longFormDesc}
                  onChange={(e) => { setLongFormDesc(e.target.value); saveClipSeo({ longform_description: e.target.value }); }}
                  style={{
                    width: '100%', minHeight: 200, padding: 10,
                    background: 'var(--bg-elevated)', color: 'var(--text-primary)',
                    border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)',
                    fontSize: 13, lineHeight: 1.5, resize: 'vertical',
                    fontFamily: 'var(--font-sans)',
                  }}
                />
              ) : (
                <div style={{ padding: '20px 0', textAlign: 'center', color: 'var(--text-muted)', fontSize: 12 }}>
                  Click "Generate" to create a full YouTube description
                </div>
              )}
            </div>
          </div>

        </div>
      </div>
    </div>
  );
}
