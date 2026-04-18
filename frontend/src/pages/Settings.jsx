import React, { useState, useEffect, useRef } from 'react';
import { useSearchParams } from 'react-router-dom';
import ModelBrowser from '../components/ModelBrowser';
import PipelineDiagnostics from '../components/PipelineDiagnostics';
import CostTracker from '../components/CostTracker';
import CloudStorageSection from '../components/cloud/CloudStorageSection';
import UserManagementPanel from '../components/UserManagementPanel';
import { useAuth } from '../auth/AuthContext';
import { showToast } from '../components/Toast';
import useResponsive from '../hooks/useResponsive';
import { loadAllPresets, deletePreset as deleteTrackPreset, renamePreset as renameTrackPreset } from '../utils/trackPresets';

const FONT_ACCEPT = '.ttf,.otf,.woff,.woff2,.eot,.TTF,.OTF,.WOFF,.WOFF2,.EOT';

const PROVIDER_KEYS = [
  { name: 'openrouter', label: 'OpenRouter', placeholder: 'sk-or-v1-...', helpUrl: 'https://openrouter.ai/keys', helpText: '300+ AI models through one key — free tier available' },
  { name: 'anthropic', label: 'Anthropic', placeholder: 'sk-ant-...', helpUrl: 'https://console.anthropic.com/settings/keys', helpText: 'Claude models — best for complex reasoning' },
  { name: 'gemini', label: 'Google Gemini', placeholder: 'AIza...', helpUrl: 'https://aistudio.google.com/apikey', helpText: 'Gemini models — great vision + long context' },
];

// Shared styles
const dropdownStyle = {
  width: '100%', padding: '8px 12px', background: 'var(--bg-panel)',
  border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)', color: 'var(--text-primary)',
  fontSize: 12, fontFamily: 'var(--font-mono)', boxSizing: 'border-box',
  appearance: 'none', cursor: 'pointer',
  backgroundImage: 'url("data:image/svg+xml,%3csvg xmlns=\'http://www.w3.org/2000/svg\' fill=\'none\' viewBox=\'0 0 20 20\'%3e%3cpath stroke=\'%236b7280\' stroke-linecap=\'round\' stroke-linejoin=\'round\' stroke-width=\'1.5\' d=\'M6 8l4 4 4-4\'/%3e%3c/svg%3e")',
  backgroundPosition: 'right 8px center', backgroundRepeat: 'no-repeat', backgroundSize: '16px',
  paddingRight: 28,
};

const TAB_NAME_TO_INDEX = { 'ai-provider': 0, prompts: 1, fonts: 2, presets: 3, advanced: 4, 'usage-costs': 5, 'api-access': 6, 'about': 7, 'users': 8 };

export default function Settings() {
  const { isMobile } = useResponsive();
  const [searchParams] = useSearchParams();
  const [settingsTab, setSettingsTab] = useState(() => {
    const tab = searchParams.get('tab');
    return tab && TAB_NAME_TO_INDEX[tab] !== undefined ? TAB_NAME_TO_INDEX[tab] : 0;
  });
  const [statuses, setStatuses] = useState({});
  const viralAlgorithmRef = useRef(null);

  // Auto-scroll to section when navigated with ?section=viral-algorithm
  useEffect(() => {
    const section = searchParams.get('section');
    if (section === 'viral-algorithm' && settingsTab === 1) {
      // Small delay to let the tab content render
      const timer = setTimeout(() => {
        viralAlgorithmRef.current?.scrollIntoView({ behavior: 'smooth', block: 'center' });
      }, 150);
      return () => clearTimeout(timer);
    }
  }, [settingsTab, searchParams]);

  // Per-provider API key state
  const [providerKeys, setProviderKeys] = useState({});
  const [providerSaving, setProviderSaving] = useState({});
  const [providerTesting, setProviderTesting] = useState({});
  const [providerResults, setProviderResults] = useState({});

  // Per-task model selection
  const [availableModels, setAvailableModels] = useState({ transcript: [], vision: [], text: [] });
  const [currentModels, setCurrentModels] = useState({ transcript_model: '', vision_model: '', text_model: '' });
  const [pendingModels, setPendingModels] = useState({ transcript_model: '', vision_model: '', text_model: '' });
  const [modelsSaving, setModelsSaving] = useState(false);
  const [modelsLoading, setModelsLoading] = useState(false);
  const [refreshing, setRefreshing] = useState(false);

  // Transcription speed settings
  const [transSettings, setTransSettings] = useState({ beam_size: 1, vad_filter: true, frame_sample_rate: 10 });
  const [transSaved, setTransSaved] = useState({ beam_size: 1, vad_filter: true, frame_sample_rate: 10 });
  const [transSaving, setTransSaving] = useState(false);

  // NOTE: Whisper testing is now in PipelineDiagnostics component

  // FFmpeg encoding settings
  const [ffmpegThreads, setFfmpegThreads] = useState(4);
  const [ffmpegThreadsSaved, setFfmpegThreadsSaved] = useState(4);
  const [ffmpegThreadsSaving, setFfmpegThreadsSaving] = useState(false);

  // Prompt customization state
  const [prompts, setPrompts] = useState({ frame_analysis: '', viral_clip_detection: '', subject_tracking: '', summary: '', seo: '' });
  const [promptDefaults, setPromptDefaults] = useState({ frame_analysis: '', viral_clip_detection: '', subject_tracking: '', summary: '', seo: '' });
  const [promptsSaving, setPromptsSaving] = useState(false);
  const [promptsLoaded, setPromptsLoaded] = useState(false);

  // Subject tracking toggle
  const [subjectTrackingEnabled, setSubjectTrackingEnabled] = useState(true);
  const [subjectTrackingSaving, setSubjectTrackingSaving] = useState(false);
  const [trackingTest, setTrackingTest] = useState(null);
  const [trackingTestRunning, setTrackingTestRunning] = useState(false);

  // Font management state
  const [customFonts, setCustomFonts] = useState([]);
  const [fontUploading, setFontUploading] = useState(false);
  const fontInputRef = useRef(null);

  // Preset management state
  const [presets, setPresets] = useState([]);
  const [editingPresetId, setEditingPresetId] = useState(null);
  const [editPresetName, setEditPresetName] = useState('');

  // Track preset management state
  const [trackPresets, setTrackPresets] = useState([]);
  const [editingTrackPresetId, setEditingTrackPresetId] = useState(null);
  const [editTrackPresetName, setEditTrackPresetName] = useState('');

  // API Access state
  const [apiKey, setApiKey] = useState('');
  const [apiKeyMasked, setApiKeyMasked] = useState('');
  const [apiKeyRevealed, setApiKeyRevealed] = useState(false);
  const [apiKeyLoading, setApiKeyLoading] = useState(false);
  const [apiKeyRegenerating, setApiKeyRegenerating] = useState(false);
  const [apiTestResult, setApiTestResult] = useState(null);
  const [apiTesting, setApiTesting] = useState(false);

  // Site customisation state
  const [siteTitle, setSiteTitle] = useState('');
  const [siteFavicon, setSiteFavicon] = useState(null); // filename from server
  const [siteLogo, setSiteLogo] = useState(null);       // filename from server
  const [siteSaving, setSiteSaving] = useState(false);
  const faviconInputRef = useRef(null);
  const logoInputRef = useRef(null);

  // GPU Hardware Acceleration toggle
  const [gpuEnabled, setGpuEnabled] = useState(false);
  const [gpuSaving, setGpuSaving] = useState(false);
  const [gpuInfo, setGpuInfo] = useState(null);
  const [gpuLoading, setGpuLoading] = useState(false);

  // Client-side GPU (browser) state
  const [clientGpuEnabled, setClientGpuEnabled] = useState(false);
  const [clientGpuInfo, setClientGpuInfo] = useState(null);
  const [clientGpuScanning, setClientGpuScanning] = useState(false);
  const [selectedClientGpuId, setSelectedClientGpuId] = useState('');
  const [clientWhisperEnabled, setClientWhisperEnabled] = useState(true);
  const [clientEncodingEnabled, setClientEncodingEnabled] = useState(true);

  // Load provider statuses
  useEffect(() => {
    fetch('/api/providers/status')
      .then((r) => r.json())
      .then(setStatuses)
      .catch(() => {});
  }, []);

  // Load GPU acceleration state
  useEffect(() => {
    fetch('/api/gpu-acceleration')
      .then((r) => r.json())
      .then((data) => {
        setGpuEnabled(data.enabled);
        setGpuInfo(data.detected);
      })
      .catch(() => {});
  }, []);

  // Load client GPU preferences and scan browser GPUs
  useEffect(() => {
    const saved = localStorage.getItem('clipai_client_gpu');
    let savedGpuId = '';
    if (saved) {
      try {
        const parsed = JSON.parse(saved);
        setClientGpuEnabled(parsed.enabled || false);
        savedGpuId = parsed.selectedGpuId || '';
        setSelectedClientGpuId(savedGpuId);
        setClientWhisperEnabled(parsed.whisperEnabled ?? true);
        setClientEncodingEnabled(parsed.encodingEnabled ?? true);
      } catch { /* ignore corrupt data */ }
    }

    (async () => {
      setClientGpuScanning(true);
      try {
        const { scanClientGPU } = await import('../utils/clientGpu.js');
        const info = await scanClientGPU();
        setClientGpuInfo(info);
        if (!savedGpuId && info.recommended) {
          setSelectedClientGpuId(info.recommended.id);
        }
      } catch (e) {
        console.warn('Client GPU scan failed:', e);
        // CRITICAL: Set a fallback so the toggle isn't permanently disabled.
        // Without this, clientGpuInfo stays null and the toggle is greyed out
        // with no error message, leaving users unable to enable GPU processing.
        setClientGpuInfo({
          webgpuSupported: !!(typeof navigator !== 'undefined' && navigator?.gpu),
          webcodecSupported: typeof VideoEncoder !== 'undefined',
          gpus: [],
          webcodecs: { h264HardwareEncode: false, hevcHardwareEncode: false, h264Decode: false, hevcDecode: false },
          recommended: null,
          scanErrors: [e?.message || 'GPU scan failed — check browser console for details'],
        });
      }
      setClientGpuScanning(false);
    })();
  }, []);

  // Load site customisation
  useEffect(() => {
    fetch('/api/site-config')
      .then((r) => r.json())
      .then((cfg) => {
        if (cfg.title) setSiteTitle(cfg.title);
        if (cfg.favicon) setSiteFavicon(cfg.favicon);
        if (cfg.logo) setSiteLogo(cfg.logo);
      })
      .catch(() => {});
  }, []);

  // Load presets
  useEffect(() => {
    fetch('/api/clip-presets')
      .then((r) => r.ok ? r.json() : [])
      .then(setPresets)
      .catch(() => {});
  }, []);

  // Load track presets from localStorage
  useEffect(() => {
    setTrackPresets(loadAllPresets());
  }, []);

  // Load transcription settings
  useEffect(() => {
    fetch('/api/transcription/settings')
      .then((r) => r.json())
      .then((data) => {
        const s = { beam_size: data.beam_size ?? 1, vad_filter: data.vad_filter ?? true, frame_sample_rate: data.frame_sample_rate ?? 10 };
        setTransSettings(s);
        setTransSaved(s);
      })
      .catch(() => {});
  }, []);

  // Load FFmpeg encoding settings
  useEffect(() => {
    fetch('/api/encoding/settings')
      .then((r) => r.json())
      .then((data) => {
        const t = data.threads ?? 4;
        setFfmpegThreads(t);
        setFfmpegThreadsSaved(t);
      })
      .catch(() => {});
  }, []);

  // Load prompts
  useEffect(() => {
    fetch('/api/prompts')
      .then((r) => r.json())
      .then((data) => {
        setPrompts(data.current);
        setPromptDefaults(data.defaults);
        setPromptsLoaded(true);
      })
      .catch(() => {});
  }, []);

  // Load subject tracking toggle
  useEffect(() => {
    fetch('/api/subject-tracking')
      .then((r) => r.json())
      .then((data) => setSubjectTrackingEnabled(data.enabled))
      .catch(() => {});
  }, []);

  // Load custom fonts
  useEffect(() => {
    fetch('/api/fonts')
      .then((r) => r.ok ? r.json() : [])
      .then(setCustomFonts)
      .catch(() => {});
  }, []);

  // Load API key when API Access tab is selected
  useEffect(() => {
    if (settingsTab === 6 && !apiKey) {
      setApiKeyLoading(true);
      fetch('/api/v1/auth/current-key')
        .then((r) => r.json())
        .then((envelope) => {
          if (envelope.success && envelope.data) {
            setApiKey(envelope.data.key);
            setApiKeyMasked(envelope.data.masked);
          }
        })
        .catch(() => showToast('Failed to load API key', 'error'))
        .finally(() => setApiKeyLoading(false));
    }
  }, [settingsTab]);

  const handleRegenerateApiKey = async () => {
    if (!confirm('Regenerate API key? The current key will be permanently invalidated.')) return;
    setApiKeyRegenerating(true);
    try {
      const res = await fetch('/api/v1/auth/regenerate-key', { method: 'POST' });
      const envelope = await res.json();
      if (envelope.success && envelope.data) {
        setApiKey(envelope.data.key);
        setApiKeyMasked(envelope.data.masked);
        setApiKeyRevealed(false);
        showToast('API key regenerated', 'success');
      } else {
        showToast('Failed to regenerate key', 'error');
      }
    } catch { showToast('Failed to regenerate key', 'error'); }
    setApiKeyRegenerating(false);
  };

  const handleTestApiConnection = async () => {
    setApiTesting(true);
    setApiTestResult(null);
    try {
      const res = await fetch('/api/v1/health', {
        headers: { 'Authorization': `Bearer ${apiKey}` },
      });
      const data = await res.json();
      if (data.success) {
        setApiTestResult({ status: 'success', message: 'Connection successful' });
        showToast('API connection test passed', 'success');
      } else {
        setApiTestResult({ status: 'error', message: 'Unexpected response' });
      }
    } catch {
      setApiTestResult({ status: 'error', message: 'Connection failed' });
      showToast('API connection test failed', 'error');
    }
    setApiTesting(false);
  };

  const handleFontUpload = async (e) => {
    const files = Array.from(e.target.files || []);
    if (!files.length) return;
    setFontUploading(true);
    try {
      for (const file of files) {
        const form = new FormData();
        form.append('file', file);
        const res = await fetch('/api/fonts/upload', { method: 'POST', body: form });
        if (!res.ok) { showToast(`Failed to upload ${file.name}`, 'error'); continue; }
        const font = await res.json();
        setCustomFonts((prev) => prev.some((f) => f.name === font.name) ? prev : [...prev, font]);
        showToast(`Font "${font.name}" uploaded`, 'success');
      }
    } catch { showToast('Font upload failed', 'error'); }
    setFontUploading(false);
    e.target.value = '';
  };

  const handleDeleteFont = async (font) => {
    try {
      await fetch(`/api/fonts/${encodeURIComponent(font.filename)}`, { method: 'DELETE' });
      setCustomFonts((prev) => prev.filter((f) => f.name !== font.name));
      showToast(`Font "${font.name}" removed`, 'info');
    } catch { showToast('Failed to remove font', 'error'); }
  };

  // Load available models when any provider is configured
  const loadAvailableModels = async () => {
    setModelsLoading(true);
    try {
      const res = await fetch('/api/providers/models/available');
      if (res.ok) {
        const data = await res.json();
        setAvailableModels({ transcript: data.transcript || [], vision: data.vision || [], text: data.text || [] });
        if (data.current) {
          setCurrentModels(data.current);
          setPendingModels(data.current);
        }
      }
    } catch {} finally {
      setModelsLoading(false);
    }
  };

  useEffect(() => {
    loadAvailableModels();
  }, []);

  // Reload models when provider status changes (including Ollama toggle)
  useEffect(() => {
    const hasProvider = statuses.openrouter?.status === 'configured' || statuses.openrouter?.status === 'connected'
      || statuses.anthropic?.status === 'configured' || statuses.gemini?.status === 'configured'
      || statuses._active?.ollama_enabled;
    if (hasProvider) loadAvailableModels();
  }, [statuses.openrouter?.status, statuses.anthropic?.status, statuses.gemini?.status, statuses._active?.ollama_enabled]);

  // Map our short provider name (the dropdown id) to the backend
  // env-var name expected by the per-user settings overlay.
  const PROVIDER_TO_ENV = {
    openrouter: 'OPENROUTER_API_KEY',
    anthropic:  'ANTHROPIC_API_KEY',
    gemini:     'GEMINI_API_KEY',
    groq:       'GROQ_API_KEY',
    huggingface: 'HF_AUTH_TOKEN',
  };

  // Save & test a provider key. The key is persisted twice when a
  // user is signed in:
  //   1. The legacy ``/api/providers/key`` route stores it in the
  //      install-wide config (so headless / single-user installs
  //      still work).
  //   2. ``/api/auth/me/settings`` stores it under the user's record;
  //      ``settings_overlay`` then makes the user's own key win for
  //      jobs they start.
  const handleSaveKey = async (providerName) => {
    const key = (providerKeys[providerName] || '').trim();
    if (!key) { showToast('Enter an API key first', 'warning'); return; }
    setProviderSaving((p) => ({ ...p, [providerName]: true }));
    try {
      const res = await fetch('/api/providers/key', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ provider: providerName, key }),
      });
      if (res.ok) {
        // Persist per-user too. Silently no-ops for anonymous visitors.
        const envName = PROVIDER_TO_ENV[providerName];
        if (envName) await savePerUserSettings({ [envName]: key });
        showToast('Key saved! Testing...', 'info');
        await handleTestProvider(providerName);
        loadAvailableModels();
      } else {
        showToast('Failed to save key', 'error');
      }
    } catch {
      showToast('Failed to save key', 'error');
    } finally {
      setProviderSaving((p) => ({ ...p, [providerName]: false }));
    }
  };

  const handleTestProvider = async (providerName) => {
    setProviderTesting((p) => ({ ...p, [providerName]: true }));
    setProviderResults((p) => ({ ...p, [providerName]: null }));
    try {
      const res = await fetch(`/api/providers/test/${providerName}`, { method: 'POST' });
      if (res.ok) {
        const result = await res.json();
        setProviderResults((p) => ({ ...p, [providerName]: result }));
        setStatuses((prev) => ({
          ...prev,
          [providerName]: { ...prev[providerName], status: result.status === 'connected' ? 'connected' : result.status === 'invalid_key' ? 'invalid_key' : prev[providerName]?.status || 'configured' },
        }));
        if (result.status === 'connected') {
          showToast(`${providerName}: Connected!`, 'success');
        } else if (result.status === 'invalid_key') {
          showToast(`${providerName}: Invalid key`, 'error');
        } else {
          showToast(result.message || `${providerName}: ${result.status}`, 'warning');
        }
      }
    } catch {
      showToast(`Failed to test ${providerName}`, 'error');
    } finally {
      setProviderTesting((p) => ({ ...p, [providerName]: false }));
    }
  };

  // Buffer a model selection (does NOT save yet)
  const handleSelectModel = (task, modelId) => {
    const key = task + '_model';
    setPendingModels((prev) => ({ ...prev, [key]: modelId }));
  };

  // Check if any model selection has changed from the saved state
  const modelsHaveChanges =
    pendingModels.transcript_model !== currentModels.transcript_model ||
    pendingModels.vision_model !== currentModels.vision_model ||
    pendingModels.text_model !== currentModels.text_model;

  // Build the per-user env-var patch for a model selection. Picks
  // the right OPENROUTER_*_MODEL vs OLLAMA_*_MODEL key by inspecting
  // the model's ``provider`` field returned by /api/providers/models/available.
  const _perUserModelPatch = (task, modelId) => {
    if (!modelId) return null;
    if (task === 'transcript') return { WHISPER_MODEL: modelId };
    const list = availableModels[task] || [];
    const m = list.find((x) => x.id === modelId);
    const provider = (m && m.provider) || '';
    const upTask = task.toUpperCase();
    if (provider === 'ollama' || provider === 'local') {
      return { [`OLLAMA_${upTask}_MODEL`]: modelId };
    }
    // Default to OpenRouter for any cloud-hosted pick. This matches
    // the existing behavior: cloud models route through OpenRouter
    // unless a user has an explicit Anthropic / Gemini direct path.
    return { [`OPENROUTER_${upTask}_MODEL`]: modelId };
  };

  // Save all pending model changes at once
  const handleSaveAllModels = async () => {
    setModelsSaving(true);
    const body = {};
    if (pendingModels.transcript_model !== currentModels.transcript_model)
      body.transcript_model = pendingModels.transcript_model;
    if (pendingModels.vision_model !== currentModels.vision_model)
      body.vision_model = pendingModels.vision_model;
    if (pendingModels.text_model !== currentModels.text_model)
      body.text_model = pendingModels.text_model;

    try {
      const res = await fetch('/api/providers/models/save', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (res.ok) {
        const data = await res.json();
        // Persist per-user picks alongside the global save.
        const userPatch = {
          ..._perUserModelPatch('transcript', body.transcript_model),
          ..._perUserModelPatch('vision', body.vision_model),
          ..._perUserModelPatch('text', body.text_model),
        };
        await savePerUserSettings(userPatch);

        // Refresh status, then reload models so the UI reflects the saved state.
        // Do this sequentially to avoid the status-change useEffect from racing
        // and overwriting the just-saved models with stale data.
        try {
          const statusRes = await fetch('/api/providers/status');
          if (statusRes.ok) setStatuses(await statusRes.json());
        } catch {}
        // Now reload available models — the server now knows Ollama is primary
        await loadAvailableModels();
        showToast('Models saved successfully', 'success');
      }
    } catch {
      showToast('Failed to save models', 'error');
    } finally {
      setModelsSaving(false);
    }
  };

  // Legacy single-task save for Advanced tab ModelBrowser
  const handleSaveModel = async (task, modelId) => {
    const body = {};
    if (task === 'transcript') body.transcript_model = modelId;
    if (task === 'vision') body.vision_model = modelId;
    if (task === 'text') body.text_model = modelId;
    try {
      const res = await fetch('/api/providers/models/save', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (res.ok) {
        // Mirror the pick into the user's per-user settings.
        const userPatch = _perUserModelPatch(task, modelId);
        if (userPatch) await savePerUserSettings(userPatch);
        // Refresh status first, then reload models to avoid race condition
        try {
          const statusRes = await fetch('/api/providers/status');
          if (statusRes.ok) setStatuses(await statusRes.json());
        } catch {}
        await loadAvailableModels();
        showToast('Model saved', 'success');
      }
    } catch {
      showToast('Failed to save model', 'error');
    }
  };

  // Refresh models from OpenRouter
  const handleRefreshModels = async () => {
    setRefreshing(true);
    try {
      const res = await fetch('/api/providers/models/refresh', { method: 'POST' });
      if (res.ok) {
        const data = await res.json();
        showToast(data.message || 'Models refreshed', 'success');
        await loadAvailableModels();
      }
    } catch {
      showToast('Failed to refresh models', 'error');
    } finally {
      setRefreshing(false);
    }
  };

  // Transcription settings handlers
  const transHasChanges = transSettings.beam_size !== transSaved.beam_size || transSettings.vad_filter !== transSaved.vad_filter || transSettings.frame_sample_rate !== transSaved.frame_sample_rate;

  const handleSaveTransSettings = async () => {
    setTransSaving(true);
    try {
      const res = await fetch('/api/transcription/settings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(transSettings),
      });
      if (res.ok) {
        const data = await res.json();
        const saved = { beam_size: data.beam_size, vad_filter: data.vad_filter, frame_sample_rate: data.frame_sample_rate };
        setTransSettings(saved);
        setTransSaved(saved);
        // Persist Whisper bits per-user too. ``frame_sample_rate``
        // isn't currently in the per-user overlay allow-list, so we
        // only mirror beam_size + vad_filter.
        await savePerUserSettings({
          WHISPER_BEAM_SIZE: saved.beam_size,
          WHISPER_VAD_FILTER: saved.vad_filter,
        });
        fetch('/api/providers/status').then((r) => r.json()).then(setStatuses).catch(() => {});
        showToast('Transcription settings saved', 'success');
      }
    } catch { showToast('Failed to save transcription settings', 'error'); }
    finally { setTransSaving(false); }
  };

  const handleSaveFfmpegThreads = async (val) => {
    setFfmpegThreadsSaving(true);
    try {
      const res = await fetch('/api/encoding/settings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ threads: val }),
      });
      if (res.ok) {
        const data = await res.json();
        setFfmpegThreads(data.threads);
        setFfmpegThreadsSaved(data.threads);
        showToast(`FFmpeg threads set to ${data.threads === 0 ? 'auto' : data.threads}`, 'success');
      }
    } catch { showToast('Failed to save FFmpeg thread setting', 'error'); }
    finally { setFfmpegThreadsSaving(false); }
  };

  // Prompt handlers
  const handleSavePrompts = async () => {
    setPromptsSaving(true);
    try {
      const res = await fetch('/api/prompts', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(prompts),
      });
      if (res.ok) {
        const data = await res.json();
        if (data.status === 'error') showToast(data.message, 'error');
        else { setPrompts(data.prompts); showToast('Prompts saved', 'success'); }
      }
    } catch { showToast('Failed to save prompts', 'error'); }
    finally { setPromptsSaving(false); }
  };

  const handleResetAllPrompts = async () => {
    setPromptsSaving(true);
    try {
      const res = await fetch('/api/prompts/reset', { method: 'POST' });
      if (res.ok) { const data = await res.json(); setPrompts(data.prompts); showToast('Prompts reset', 'success'); }
    } catch { showToast('Failed to reset', 'error'); }
    finally { setPromptsSaving(false); }
  };

  const handleToggleSubjectTracking = async (enabled) => {
    setSubjectTrackingSaving(true);
    try {
      const res = await fetch('/api/subject-tracking', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enabled }),
      });
      if (res.ok) {
        setSubjectTrackingEnabled(enabled);
        showToast(`Subject tracking ${enabled ? 'enabled' : 'disabled'}`, 'success');
      }
    } catch { showToast('Failed to update subject tracking', 'error'); }
    finally { setSubjectTrackingSaving(false); }
  };

  const handleTestSubjectTracking = async () => {
    setTrackingTestRunning(true);
    setTrackingTest(null);
    try {
      const res = await fetch('/api/diagnostics/test-subject-tracking', { method: 'POST' });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setTrackingTest(data);
    } catch (e) {
      setTrackingTest({
        overall_status: 'fail',
        summary: `Test failed: ${e.message}`,
        results: [],
      });
    }
    setTrackingTestRunning(false);
  };

  const handleToggleGpu = async (enabled) => {
    setGpuSaving(true);
    setGpuLoading(true);
    try {
      const res = await fetch('/api/gpu-acceleration', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enabled }),
      });
      if (res.ok) {
        const data = await res.json();
        setGpuEnabled(data.enabled);
        setGpuInfo(data.detected);
        if (enabled && data.detected?.vendor !== 'none') {
          showToast(`GPU acceleration enabled — ${data.detected.gpu_name} (${data.detected.encoder})`, 'success');
        } else if (enabled && data.detected?.vendor === 'none') {
          showToast('GPU acceleration enabled but no compatible GPU detected — using CPU', 'warning');
        } else {
          showToast('GPU acceleration disabled — using CPU encoding', 'success');
        }
      }
    } catch { showToast('Failed to update GPU acceleration', 'error'); }
    finally { setGpuSaving(false); setGpuLoading(false); }
  };

  const handleSaveClientGpu = (overrides = {}) => {
    const state = {
      enabled: overrides.enabled ?? clientGpuEnabled,
      selectedGpuId: overrides.selectedGpuId ?? selectedClientGpuId,
      whisperEnabled: overrides.whisperEnabled ?? clientWhisperEnabled,
      encodingEnabled: overrides.encodingEnabled ?? clientEncodingEnabled,
    };
    // Include GPU name so other components (Layout top bar) can display it
    const gpuId = state.selectedGpuId;
    const selectedGpu = clientGpuInfo?.gpus?.find(g => g.id === gpuId);
    state.selectedGpuName = selectedGpu?.name || '';
    localStorage.setItem('clipai_client_gpu', JSON.stringify(state));
    window.dispatchEvent(new Event('clientgpu-changed'));

    // Report selected GPU to the server so FFmpeg can use the right device
    if (selectedGpu) {
      fetch('/api/client-gpu-report', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          webgpu_supported: clientGpuInfo?.webgpuSupported || false,
          webcodec_supported: clientGpuInfo?.webcodecSupported || false,
          gpu_name: selectedGpu.name || '',
          gpu_vendor: selectedGpu.vendor || '',
          estimated_vram_mb: selectedGpu.estimatedVRAM_MB || 0,
          whisper_capable: selectedGpu.whisperCapable || false,
          h264_hardware_encode: clientGpuInfo?.webcodecs?.h264HardwareEncode || false,
          hevc_hardware_encode: clientGpuInfo?.webcodecs?.hevcHardwareEncode || false,
          gpu_index: selectedGpu.serverIndex || '0',
          gpu_backend: selectedGpu.backend || '',
        }),
      }).catch(() => {});
    }
  };

  const handleToggleClientGpu = (enabled) => {
    setClientGpuEnabled(enabled);
    handleSaveClientGpu({ enabled });
    if (enabled && clientGpuInfo?.gpus?.length > 0) {
      showToast(`Client GPU enabled — ${clientGpuInfo.recommended?.name || 'GPU detected'}`, 'success');
    } else if (enabled) {
      showToast('Client GPU enabled but no WebGPU-compatible GPU found', 'warning');
    } else {
      showToast('Client GPU disabled — processing will use the server', 'success');
    }
  };

  const handleRenamePreset = async (presetId) => {
    const name = editPresetName.trim();
    if (!name) return;
    try {
      const res = await fetch(`/api/clip-presets/${presetId}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name }),
      });
      if (res.ok) {
        const updated = await res.json();
        setPresets((prev) => prev.map((p) => p.id === presetId ? updated : p));
        setEditingPresetId(null);
        setEditPresetName('');
        showToast('Preset renamed', 'success');
      }
    } catch { showToast('Rename failed', 'error'); }
  };

  const handleDeletePresetSettings = async (presetId) => {
    try {
      const res = await fetch(`/api/clip-presets/${presetId}`, { method: 'DELETE' });
      if (res.ok) {
        setPresets((prev) => prev.filter((p) => p.id !== presetId));
        showToast('Preset deleted', 'success');
      }
    } catch { showToast('Delete failed', 'error'); }
  };

  const handleExportPreset = (preset) => {
    const blob = new Blob([JSON.stringify(preset, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `preset-${preset.name.replace(/\s+/g, '-')}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const { isAdmin, user, apiFetch } = useAuth();

  // Track which keys (if any) the signed-in user has saved per-user.
  // Used purely for UI hints: a "saved" badge next to API key fields,
  // and a "your value" indicator next to model dropdowns.
  const [perUserSettings, setPerUserSettings] = useState({});

  // Persist a patch of env-var keys to the signed-in user's per-user
  // settings. Silently no-ops when the visitor isn't authenticated
  // (so anonymous flows still work for local single-user installs).
  // Returns the merged server response so callers can refresh state.
  const savePerUserSettings = async (patch) => {
    if (!user || !patch) return null;
    const data = {};
    for (const [k, v] of Object.entries(patch)) {
      if (v === undefined || v === null || v === '') continue;
      // Coerce booleans + numbers to string — backend accepts strings
      // and coerces back to the right type via _coerce in
      // settings_overlay.py.
      data[k] = typeof v === 'boolean' || typeof v === 'number' ? String(v) : v;
    }
    if (Object.keys(data).length === 0) return null;
    try {
      const resp = await apiFetch('/api/auth/me/settings', {
        method: 'PUT',
        body: { data },
      });
      if (resp?.settings) setPerUserSettings(resp.settings);
      return resp;
    } catch (e) {
      // Don't surface as an error — global save already succeeded.
      console.warn('Per-user settings save failed:', e);
      return null;
    }
  };

  // Load the signed-in user's per-user settings on mount so the
  // existing Settings UI fields can be pre-filled with the user's
  // own values (overriding the system defaults that would otherwise
  // be loaded from /api/providers/* and /api/transcription/settings).
  useEffect(() => {
    if (!user) return;
    let alive = true;
    (async () => {
      try {
        const resp = await apiFetch('/api/auth/me/settings');
        if (!alive) return;
        const s = resp?.settings || {};
        setPerUserSettings(s);

        // ── Overlay API keys ──
        // Server returns "<set>" for each saved secret; show a
        // placeholder only — we never round-trip the real value.
        setProviderKeys((prev) => {
          const next = { ...prev };
          if (s.OPENROUTER_API_KEY === '<set>') next.openrouter = next.openrouter || '';
          if (s.ANTHROPIC_API_KEY === '<set>')  next.anthropic  = next.anthropic  || '';
          if (s.GEMINI_API_KEY === '<set>')     next.gemini     = next.gemini     || '';
          return next;
        });

        // ── Overlay model picks (vision / text / whisper) ──
        // Pick whichever variant the user actually saved. Falls back
        // to whatever the global config returned via loadAvailableModels.
        const visionPick = s.OPENROUTER_VISION_MODEL || s.OLLAMA_VISION_MODEL || '';
        const textPick   = s.OPENROUTER_TEXT_MODEL   || s.OLLAMA_TEXT_MODEL   || '';
        const transPick  = s.WHISPER_MODEL || '';
        if (visionPick || textPick || transPick) {
          setPendingModels((prev) => ({
            transcript_model: transPick || prev.transcript_model,
            vision_model:     visionPick || prev.vision_model,
            text_model:       textPick   || prev.text_model,
          }));
          setCurrentModels((prev) => ({
            transcript_model: transPick || prev.transcript_model,
            vision_model:     visionPick || prev.vision_model,
            text_model:       textPick   || prev.text_model,
          }));
        }

        // ── Overlay Whisper decoding bits ──
        const beam = s.WHISPER_BEAM_SIZE != null && s.WHISPER_BEAM_SIZE !== ''
          ? parseInt(s.WHISPER_BEAM_SIZE, 10)
          : null;
        const vad = s.WHISPER_VAD_FILTER != null && s.WHISPER_VAD_FILTER !== ''
          ? (String(s.WHISPER_VAD_FILTER).toLowerCase() === 'true' || s.WHISPER_VAD_FILTER === '1')
          : null;
        if (Number.isFinite(beam) || vad !== null) {
          setTransSettings((prev) => ({
            ...prev,
            ...(Number.isFinite(beam) ? { beam_size: beam } : {}),
            ...(vad !== null ? { vad_filter: vad } : {}),
          }));
          setTransSaved((prev) => ({
            ...prev,
            ...(Number.isFinite(beam) ? { beam_size: beam } : {}),
            ...(vad !== null ? { vad_filter: vad } : {}),
          }));
        }
      } catch (e) {
        // Anonymous / not signed in or the per-user store hasn't been
        // initialized yet — no overlay, the existing system defaults
        // remain in effect.
        console.debug('Per-user settings load skipped:', e?.message || e);
      }
    })();
    return () => { alive = false; };
    // ``apiFetch`` is stable (memoized in AuthContext); user identity
    // change retriggers the load.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [user?.id]);
  const SETTINGS_TABS = [
    'AI Provider', 'Prompts', 'Fonts', 'Presets', 'Advanced',
    'Usage & Costs', 'API Access', 'About',
    ...(isAdmin ? ['Users'] : []),
  ];
  const active = statuses._active || {};

  // Model dropdown renderer
  // Format Unix timestamp to "Mon YYYY"
  const formatRelease = (ts) => {
    if (!ts) return '';
    const d = new Date(ts * 1000);
    const months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
    return `${months[d.getUTCMonth()]} ${d.getUTCFullYear()}`;
  };

  const speedBadge = (m) => {
    if (!m.speed) return '';
    const label = m.speed === 'fast' ? 'FAST' : m.speed === 'slow' ? 'SLOW' : 'MED';
    return `[${label}${m.est_time_display ? ' ' + m.est_time_display : ''}]`;
  };

  const qualityStars = (score) => {
    if (!score || score < 1) return '';
    const filled = Math.min(5, Math.max(1, score));
    return '\u2605'.repeat(filled) + '\u2606'.repeat(5 - filled);
  };

  const ModelDropdown = ({ task, models, pendingValue, savedValue, label, desc }) => {
    const isChanged = pendingValue !== savedValue;
    const selectedModel = models.find((m) => m.id === pendingValue);
    return (
      <div style={{ marginBottom: 20 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
          <h4 style={{ fontSize: 13, margin: 0, color: 'var(--text-primary)' }}>
            {label}
            {isChanged && (
              <span style={{ fontSize: 10, fontWeight: 400, color: 'var(--accent-amber)', marginLeft: 8 }}>
                unsaved
              </span>
            )}
          </h4>
          {savedValue && (
            <span style={{ fontSize: 10, fontFamily: 'var(--font-mono)', color: isChanged ? 'var(--text-muted)' : 'var(--accent-cyan)' }}>
              {savedValue}
            </span>
          )}
        </div>
        <p style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 8, lineHeight: 1.5 }}>{desc}</p>
        <div style={{ position: 'relative' }}>
          <select
            value={pendingValue || ''}
            onChange={(e) => handleSelectModel(task, e.target.value)}
            style={{
              ...dropdownStyle,
              borderColor: isChanged ? 'var(--accent-amber)' : undefined,
            }}
          >
            <option value="">-- Select a model --</option>
            {models.map((m) => {
              const price = m.is_free ? '[FREE]' : m.cost_per_hour > 0 ? `[$${m.cost_per_hour.toFixed(3)}/hr]` : '';
              const released = m.created ? `[${formatRelease(m.created)}]` : '';
              const provider = m.provider !== 'local' ? ` (${m.provider})` : '';
              const speed = speedBadge(m);
              const stars = m.quality_score ? qualityStars(m.quality_score) : '';
              const tracking = task === 'vision' && m.tracking_score
                ? m.tracking_score >= 4 ? '[TRACK:\u2605\u2605]' : m.tracking_score >= 2 ? '[TRACK:\u2605]' : '[TRACK:\u26A0]'
                : '';
              return (
                <option key={m.id} value={m.id}>
                  {[stars, speed, tracking, price, released, m.name + provider].filter(Boolean).join(' ')}
                </option>
              );
            })}
          </select>
        </div>
        {/* Speed / quality info for selected model */}
        {selectedModel && (selectedModel.speed || selectedModel.quality_score) && (
          <div style={{
            marginTop: 6, padding: '6px 10px', borderRadius: 'var(--radius-sm)',
            background: 'var(--bg-elevated)', border: '1px solid var(--border)',
            display: 'flex', gap: 12, alignItems: 'center', fontSize: 11, flexWrap: 'wrap',
          }}>
            {selectedModel.speed && (
              <span style={{
                padding: '2px 8px', borderRadius: 8, fontWeight: 700, fontSize: 10,
                fontFamily: 'var(--font-mono)',
                background: selectedModel.speed === 'fast' ? 'var(--success)' : selectedModel.speed === 'slow' ? 'var(--accent-amber)' : 'var(--accent-cyan)',
                color: 'var(--bg-base)',
              }}>
                {selectedModel.speed === 'fast' ? 'SPEED' : selectedModel.speed === 'slow' ? 'QUALITY' : 'BALANCED'}
              </span>
            )}
            {selectedModel.speed && (
              <span style={{ color: 'var(--text-secondary)', fontFamily: 'var(--font-mono)' }}>
                10 min video: {selectedModel.est_time_display || '~2min'}
              </span>
            )}
            {selectedModel.quality_score && (
              <span style={{
                display: 'inline-flex', alignItems: 'center', gap: 4,
                color: selectedModel.quality_score >= 4 ? 'var(--accent-amber)' : selectedModel.quality_score <= 2 ? 'var(--text-muted)' : 'var(--accent-cyan)',
              }}>
                <span style={{ fontSize: 12, letterSpacing: 1 }}>
                  {qualityStars(selectedModel.quality_score)}
                </span>
                <span style={{ fontSize: 10, textTransform: 'uppercase', fontFamily: 'var(--font-mono)' }}>
                  {selectedModel.quality || `${selectedModel.quality_score}/5`}
                </span>
              </span>
            )}
          </div>
        )}
      </div>
    );
  };

  return (
    <div>
      <h2 style={{ fontSize: isMobile ? 18 : 20, marginBottom: isMobile ? 20 : 24 }}>Settings</h2>

      {/* Settings tabs */}
      <div className="responsive-tabs" style={{ display: 'flex', gap: 0, borderBottom: '1px solid var(--border)', marginBottom: isMobile ? 20 : 24 }}>
        {SETTINGS_TABS.map((t, i) => (
          <button
            key={t}
            onClick={() => setSettingsTab(i)}
            style={{
              padding: isMobile ? '10px 14px' : '10px 20px', background: 'none', border: 'none',
              borderBottom: settingsTab === i ? '2px solid var(--accent-cyan)' : '2px solid transparent',
              color: settingsTab === i ? 'var(--accent-cyan)' : 'var(--text-secondary)',
              fontSize: 13, fontWeight: settingsTab === i ? 600 : 400, fontFamily: 'var(--font-mono)',
              whiteSpace: 'nowrap', flexShrink: 0,
            }}
          >
            {t}
          </button>
        ))}
      </div>

      {/* ═══════ Tab 0: AI Provider ═══════ */}
      {settingsTab === 0 && (
        <div style={{ maxWidth: isMobile ? '100%' : 640 }}>

          {/* The AI Provider tab below IS the per-user settings UI when
              the visitor is signed in. ``handleSaveKey``,
              ``handleSaveAllModels`` and ``handleSaveTransSettings``
              persist their values both to the system config (legacy
              shared default) AND to ``/api/auth/me/settings`` so the
              currently-authenticated user's jobs use that user's keys
              and model picks via the backend's settings overlay. */}

          {/* ── Active Models Banner ── */}
          <div style={{
            background: 'var(--bg-elevated)', border: '1px solid var(--border)',
            borderRadius: 'var(--radius-md)', padding: isMobile ? '12px 14px' : '14px 18px', marginBottom: 24,
            boxShadow: 'var(--shadow-sm)',
          }}>
            <div style={{ fontSize: 11, fontFamily: 'var(--font-mono)', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 8 }}>
              Active Models
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: isMobile ? '1fr' : '1fr 1fr 1fr', gap: 12 }}>
              {[
                { label: 'Transcript', model: active.transcript_model || currentModels.transcript_model || 'base', color: 'var(--accent-amber)' },
                { label: 'Vision', model: active.vision_model || currentModels.vision_model, color: 'var(--accent-cyan)' },
                { label: 'Text', model: active.text_model || currentModels.text_model, color: 'var(--success)' },
              ].map(({ label, model, color }) => (
                <div key={label}>
                  <div style={{ fontSize: 10, color: 'var(--text-muted)', marginBottom: 2 }}>{label}</div>
                  <div style={{ fontSize: 12, fontFamily: 'var(--font-mono)', color, fontWeight: 600, wordBreak: 'break-all' }}>
                    {model ? model.replace(/^.*\//, '') : 'Not set'}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* ── API Keys Section ── */}
          <h3 style={{ fontSize: 14, marginBottom: 12, color: 'var(--text-secondary)' }}>
            API Keys
          </h3>
          <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 16, lineHeight: 1.6 }}>
            Add API keys for the providers you want to use. Each key is tested automatically after saving.
          </p>

          <div style={{ display: 'grid', gap: 12, marginBottom: 32 }}>
            {PROVIDER_KEYS.map((prov) => {
              const st = statuses[prov.name]?.status || 'not_configured';
              const isUp = st === 'connected' || st === 'configured';
              const isBusy = providerSaving[prov.name] || providerTesting[prov.name];
              const result = providerResults[prov.name];
              return (
                <div key={prov.name} style={{
                  background: 'var(--bg-panel)', border: '1px solid var(--border)',
                  borderRadius: 'var(--radius-md)', padding: isMobile ? '12px' : '12px 16px',
                  boxShadow: 'var(--shadow-sm)',
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
                    <div style={{
                      width: 8, height: 8, borderRadius: '50%', flexShrink: 0,
                      background: st === 'connected' ? 'var(--success)' : isUp ? 'var(--accent-cyan)' : st === 'invalid_key' ? 'var(--danger)' : 'var(--text-muted)',
                    }} />
                    <span style={{ fontSize: 13, fontWeight: 600, flex: 1 }}>{prov.label}</span>
                    <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>
                      {isUp ? 'Connected' : st === 'invalid_key' ? 'Invalid' : 'Not configured'}
                    </span>
                  </div>
                  <p style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 8, lineHeight: 1.4 }}>
                    {prov.helpText} —{' '}
                    <a href={prov.helpUrl} target="_blank" rel="noopener noreferrer"
                      style={{ color: 'var(--accent-cyan)', textDecoration: 'none' }}>Get key</a>
                  </p>
                  <div style={{ display: 'flex', gap: 6, flexDirection: isMobile ? 'column' : 'row' }}>
                    <input
                      type="password"
                      placeholder={prov.placeholder}
                      value={providerKeys[prov.name] || ''}
                      onChange={(e) => setProviderKeys((p) => ({ ...p, [prov.name]: e.target.value }))}
                      onKeyDown={(e) => e.key === 'Enter' && handleSaveKey(prov.name)}
                      style={{
                        flex: 1, padding: '7px 10px', background: 'var(--bg-base)',
                        border: `1px solid ${isUp ? 'var(--success)' : 'var(--border)'}`,
                        borderRadius: 'var(--radius-sm)', color: 'var(--text-primary)', fontSize: 12, fontFamily: 'var(--font-mono)',
                        boxSizing: 'border-box',
                      }}
                    />
                    <button
                      onClick={() => handleSaveKey(prov.name)}
                      disabled={isBusy || !(providerKeys[prov.name] || '').trim()}
                      style={{
                        padding: '7px 14px', background: 'var(--accent-cyan)', color: 'var(--bg-base)',
                        border: 'none', borderRadius: 'var(--radius-sm)', fontSize: 11, fontWeight: 600,
                        opacity: isBusy || !(providerKeys[prov.name] || '').trim() ? 0.5 : 1, whiteSpace: 'nowrap',
                      }}
                    >
                      {providerSaving[prov.name] ? 'Saving...' : 'Save & Test'}
                    </button>
                    {isUp && (
                      <button
                        onClick={() => handleTestProvider(prov.name)}
                        disabled={isBusy}
                        style={{
                          padding: '7px 10px', background: 'var(--bg-elevated)', color: 'var(--text-secondary)',
                          border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)', fontSize: 11,
                          opacity: isBusy ? 0.5 : 1, whiteSpace: 'nowrap',
                        }}
                      >
                        {providerTesting[prov.name] ? '...' : 'Test'}
                      </button>
                    )}
                  </div>
                  {result && (
                    <div style={{
                      marginTop: 8, padding: '6px 10px', borderRadius: 'var(--radius-sm)', fontSize: 11, lineHeight: 1.5,
                      background: result.status === 'connected' ? 'var(--success-dim)' : result.status === 'invalid_key' ? 'var(--danger-dim)' : 'var(--amber-dim)',
                      color: result.status === 'connected' ? 'var(--success)' : result.status === 'invalid_key' ? 'var(--danger)' : 'var(--accent-amber)',
                    }}>
                      {result.message}
                      {result.usage_usd !== undefined && (
                        <span style={{ marginLeft: 8, color: 'var(--text-muted)' }}>
                          Usage: ${Number(result.usage_usd).toFixed(4)}
                        </span>
                      )}
                    </div>
                  )}
                </div>
              );
            })}

            {/* Ollama (local) — toggle to enable/disable */}
            <div style={{
              background: 'var(--bg-panel)', border: '1px solid var(--border)',
              borderRadius: 'var(--radius-md)', padding: '12px 16px',
              opacity: statuses._active?.ollama_enabled ? 1 : 0.6,
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
                <div style={{
                  width: 8, height: 8, borderRadius: '50%', flexShrink: 0,
                  background: statuses.ollama?.status === 'connected' ? 'var(--success)' : 'var(--text-muted)',
                }} />
                <span style={{ fontSize: 13, fontWeight: 600, flex: 1 }}>Ollama (Local)</span>
                <label style={{ display: 'flex', alignItems: 'center', gap: 6, cursor: 'pointer', fontSize: 11, color: 'var(--text-secondary)' }}>
                  {statuses._active?.ollama_enabled ? 'Enabled' : 'Disabled'}
                  <input
                    type="checkbox"
                    checked={!!statuses._active?.ollama_enabled}
                    onChange={async (e) => {
                      const enabled = e.target.checked;
                      try {
                        const res = await fetch('/api/providers/ollama/toggle', {
                          method: 'POST',
                          headers: { 'Content-Type': 'application/json' },
                          body: JSON.stringify({ enabled }),
                        });
                        if (res.ok) {
                          const data = await res.json();
                          showToast(enabled ? 'Ollama enabled in fallback chain' : 'Ollama disabled', 'success');
                          setStatuses(prev => ({
                            ...prev,
                            _active: { ...prev._active, ollama_enabled: enabled, fallback_chain: data.chain },
                          }));
                          loadAvailableModels();
                        }
                      } catch {
                        showToast('Failed to toggle Ollama', 'error');
                      }
                    }}
                    style={{ accentColor: 'var(--accent)' }}
                  />
                </label>
                {statuses._active?.ollama_enabled && (
                  <button
                    onClick={() => handleTestProvider('ollama')}
                    disabled={providerTesting.ollama}
                    style={{
                      padding: '4px 10px', background: 'var(--bg-elevated)', color: 'var(--text-secondary)',
                      border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)', fontSize: 11,
                      opacity: providerTesting.ollama ? 0.5 : 1,
                    }}
                  >
                    {providerTesting.ollama ? '...' : 'Test'}
                  </button>
                )}
              </div>
              <p style={{ fontSize: 11, color: 'var(--text-muted)', lineHeight: 1.4, margin: 0 }}>
                No API key needed — runs models locally on your GPU.
                {!statuses._active?.ollama_enabled && ' Toggle on to use Ollama as your primary AI provider.'}
                {statuses._active?.ollama_enabled && (
                  <>
                    <span style={{ display: 'block', color: 'var(--success)', fontWeight: 600, marginTop: 4 }}>
                      Primary provider — Ollama models will be tried first.
                    </span>
                    {statuses.ollama?.models_loaded?.length > 0 && (
                      <span style={{ display: 'block', fontFamily: 'var(--font-mono)', fontSize: 10, marginTop: 4 }}>
                        Available: {statuses.ollama.models_loaded.join(', ')}
                      </span>
                    )}
                  </>
                )}
              </p>
            </div>

            {/* Speaker Detection — HuggingFace token for pyannote neural diarization */}
            {(() => {
              const hfName = 'huggingface';
              const hfSt = statuses[hfName]?.status || 'not_configured';
              const hfUp = hfSt === 'connected' || hfSt === 'configured';
              const hfBusy = providerSaving[hfName] || providerTesting[hfName];
              const hfResult = providerResults[hfName];
              return (
                <div style={{
                  background: 'var(--bg-panel)', border: '1px solid var(--border)',
                  borderRadius: 'var(--radius-md)', padding: isMobile ? '12px' : '12px 16px',
                  boxShadow: 'var(--shadow-sm)',
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
                    <div style={{
                      width: 8, height: 8, borderRadius: '50%', flexShrink: 0,
                      background: hfSt === 'connected' ? 'var(--success)' : hfUp ? 'var(--accent-cyan)' : hfSt === 'invalid_key' ? 'var(--danger)' : 'var(--text-muted)',
                    }} />
                    <span style={{ fontSize: 13, fontWeight: 600, flex: 1 }}>Speaker Detection</span>
                    <span style={{ fontSize: 10, color: hfUp ? 'var(--success)' : 'var(--text-muted)', fontWeight: 600 }}>
                      {hfUp ? 'Neural diarization active' : 'Heuristic mode'}
                    </span>
                  </div>
                  <p style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 8, lineHeight: 1.5 }}>
                    Without a token, speaker detection uses pause-based heuristics.
                    Add a HuggingFace token to enable <strong style={{ color: 'var(--text-secondary)' }}>pyannote neural diarization</strong> —
                    accurately identifies who is speaking with unlimited speakers.
                  </p>
                  <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 10, lineHeight: 1.6, paddingLeft: 8, borderLeft: '2px solid var(--border)' }}>
                    <div>1. Create a free account at <a href="https://huggingface.co/join" target="_blank" rel="noopener noreferrer" style={{ color: 'var(--accent-cyan)', textDecoration: 'none' }}>huggingface.co</a></div>
                    <div>2. Get a token at <a href="https://huggingface.co/settings/tokens" target="_blank" rel="noopener noreferrer" style={{ color: 'var(--accent-cyan)', textDecoration: 'none' }}>Settings &gt; Access Tokens</a></div>
                    <div>3. Accept model terms at <a href="https://huggingface.co/pyannote/speaker-diarization-3.1" target="_blank" rel="noopener noreferrer" style={{ color: 'var(--accent-cyan)', textDecoration: 'none' }}>pyannote/speaker-diarization-3.1</a></div>
                  </div>
                  <div style={{ display: 'flex', gap: 6, flexDirection: isMobile ? 'column' : 'row' }}>
                    <input
                      type="password"
                      placeholder="hf_..."
                      value={providerKeys[hfName] || ''}
                      onChange={(e) => setProviderKeys((p) => ({ ...p, [hfName]: e.target.value }))}
                      onKeyDown={(e) => e.key === 'Enter' && handleSaveKey(hfName)}
                      style={{
                        flex: 1, padding: '7px 10px', background: 'var(--bg-base)',
                        border: `1px solid ${hfUp ? 'var(--success)' : 'var(--border)'}`,
                        borderRadius: 'var(--radius-sm)', color: 'var(--text-primary)', fontSize: 12, fontFamily: 'var(--font-mono)',
                        boxSizing: 'border-box',
                      }}
                    />
                    <button
                      onClick={() => handleSaveKey(hfName)}
                      disabled={hfBusy || !(providerKeys[hfName] || '').trim()}
                      style={{
                        padding: '7px 14px', background: 'var(--accent-cyan)', color: 'var(--bg-base)',
                        border: 'none', borderRadius: 'var(--radius-sm)', fontSize: 11, fontWeight: 600,
                        opacity: hfBusy || !(providerKeys[hfName] || '').trim() ? 0.5 : 1, whiteSpace: 'nowrap',
                      }}
                    >
                      {providerSaving[hfName] ? 'Saving...' : 'Save & Test'}
                    </button>
                    {hfUp && (
                      <button
                        onClick={() => handleTestProvider(hfName)}
                        disabled={hfBusy}
                        style={{
                          padding: '7px 10px', background: 'var(--bg-elevated)', color: 'var(--text-secondary)',
                          border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)', fontSize: 11,
                          opacity: hfBusy ? 0.5 : 1, whiteSpace: 'nowrap',
                        }}
                      >
                        {providerTesting[hfName] ? '...' : 'Test'}
                      </button>
                    )}
                  </div>
                  {hfResult && (
                    <div style={{
                      marginTop: 8, padding: '6px 10px', borderRadius: 'var(--radius-sm)', fontSize: 11, lineHeight: 1.5,
                      background: hfResult.status === 'connected' ? 'var(--success-dim)' : hfResult.status === 'invalid_key' ? 'var(--danger-dim)' : 'var(--amber-dim)',
                      color: hfResult.status === 'connected' ? 'var(--success)' : hfResult.status === 'invalid_key' ? 'var(--danger)' : 'var(--accent-amber)',
                    }}>
                      {hfResult.message}
                    </div>
                  )}
                </div>
              );
            })()}
          </div>

          {/* ── Model Selection Section ── */}
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
            <h3 style={{ fontSize: 14, margin: 0, color: 'var(--text-secondary)' }}>
              Model Selection
            </h3>
            <button
              onClick={handleRefreshModels}
              disabled={refreshing}
              style={{
                padding: '5px 12px', background: 'var(--bg-elevated)',
                color: refreshing ? 'var(--text-muted)' : 'var(--accent-cyan)',
                border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)', fontSize: 11, fontWeight: 500,
                display: 'flex', alignItems: 'center', gap: 6, opacity: refreshing ? 0.6 : 1,
              }}
            >
              <span style={{ display: 'inline-block', transition: 'transform 0.3s', transform: refreshing ? 'rotate(180deg)' : 'none' }}>
                &#x21bb;
              </span>
              {refreshing ? 'Refreshing...' : 'Refresh Models'}
            </button>
          </div>
          <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 20, lineHeight: 1.6 }}>
            Pick which AI model handles each task. Free models are listed first.
            Costs shown are estimates per 1-hour video.
          </p>

          {modelsLoading ? (
            <div style={{ padding: 16, color: 'var(--text-muted)', fontSize: 12 }}>Loading models...</div>
          ) : (
            <>
              <ModelDropdown
                task="transcript"
                models={availableModels.transcript}
                pendingValue={pendingModels.transcript_model}
                savedValue={currentModels.transcript_model}
                label="Transcript AI (Local Whisper)"
                desc="Speech-to-text runs locally using OpenAI Whisper — always free, no API key needed. Cloud providers like OpenRouter only offer text/vision LLMs, not transcription. Larger models are more accurate but need more RAM/GPU."
              />

              {/* ── Transcription Speed/Quality ── */}
              <div style={{
                background: 'var(--bg-panel)', border: '1px solid var(--border)',
                borderRadius: 'var(--radius-md)', padding: '14px 16px', marginBottom: 20,
              }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
                  <h4 style={{ fontSize: 13, margin: 0, color: 'var(--text-primary)' }}>
                    Transcription Speed
                    {transHasChanges && (
                      <span style={{ fontSize: 10, fontWeight: 400, color: 'var(--accent-amber)', marginLeft: 8 }}>
                        unsaved
                      </span>
                    )}
                  </h4>
                </div>
                <p style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 12, lineHeight: 1.5 }}>
                  Control the speed vs accuracy tradeoff. VAD filter skips silence for a major speedup.
                  Lower beam size is faster but less accurate.
                </p>

                {/* VAD Filter toggle */}
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10, padding: '6px 0' }}>
                  <div>
                    <span style={{ fontSize: 12, color: 'var(--text-primary)' }}>Skip Silence (VAD Filter)</span>
                    <span style={{ fontSize: 10, color: 'var(--text-muted)', display: 'block' }}>
                      Skips non-speech sections — 2-3x faster
                    </span>
                  </div>
                  <button
                    onClick={() => setTransSettings((p) => ({ ...p, vad_filter: !p.vad_filter }))}
                    style={{
                      width: 44, height: 24, borderRadius: 12, border: 'none', cursor: 'pointer',
                      background: transSettings.vad_filter ? 'var(--accent-cyan)' : 'var(--border)',
                      position: 'relative', transition: 'background 0.2s',
                    }}
                  >
                    <div style={{
                      width: 18, height: 18, borderRadius: '50%', background: 'white',
                      position: 'absolute', top: 3,
                      left: transSettings.vad_filter ? 23 : 3,
                      transition: 'left 0.2s',
                    }} />
                  </button>
                </div>

                {/* Beam Size slider */}
                <div style={{ padding: '6px 0' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
                    <span style={{ fontSize: 12, color: 'var(--text-primary)' }}>Beam Size</span>
                    <span style={{ fontSize: 11, fontFamily: 'var(--font-mono)', color: 'var(--accent-cyan)' }}>
                      {transSettings.beam_size === 1 ? '1 (Fast)' : transSettings.beam_size >= 5 ? '5 (Accurate)' : transSettings.beam_size}
                    </span>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <span style={{ fontSize: 10, color: 'var(--text-muted)', flexShrink: 0 }}>Fast</span>
                    <input
                      type="range"
                      min="1"
                      max="5"
                      value={transSettings.beam_size}
                      onChange={(e) => setTransSettings((p) => ({ ...p, beam_size: parseInt(e.target.value) }))}
                      style={{ flex: 1, accentColor: 'var(--accent-cyan)' }}
                    />
                    <span style={{ fontSize: 10, color: 'var(--text-muted)', flexShrink: 0 }}>Accurate</span>
                  </div>
                  <span style={{ fontSize: 10, color: 'var(--text-muted)', display: 'block', marginTop: 4 }}>
                    1 = greedy (fastest), 5 = beam search (most accurate)
                  </span>
                </div>

                {/* Frame Sample Rate slider */}
                <div style={{ padding: '6px 0' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
                    <span style={{ fontSize: 12, color: 'var(--text-primary)' }}>Frame Sample Rate</span>
                    <span style={{ fontSize: 11, fontFamily: 'var(--font-mono)', color: 'var(--accent-cyan)' }}>
                      Every {transSettings.frame_sample_rate}s
                    </span>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <span style={{ fontSize: 10, color: 'var(--text-muted)', flexShrink: 0 }}>5s</span>
                    <input
                      type="range"
                      min="5"
                      max="30"
                      step="5"
                      value={transSettings.frame_sample_rate}
                      onChange={(e) => setTransSettings((p) => ({ ...p, frame_sample_rate: parseInt(e.target.value) }))}
                      style={{ flex: 1, accentColor: 'var(--accent-cyan)' }}
                    />
                    <span style={{ fontSize: 10, color: 'var(--text-muted)', flexShrink: 0 }}>30s</span>
                  </div>
                  <span style={{ fontSize: 10, color: 'var(--text-muted)', display: 'block', marginTop: 4 }}>
                    Lower = more visual detail but slower processing. Higher = faster but less detail.
                  </span>
                </div>

                {/* Save button */}
                {transHasChanges && (
                  <button
                    onClick={handleSaveTransSettings}
                    disabled={transSaving}
                    style={{
                      marginTop: 10, padding: '7px 20px',
                      background: 'var(--accent-cyan)', color: 'var(--bg-base)',
                      border: 'none', borderRadius: 'var(--radius-sm)', fontSize: 12, fontWeight: 600,
                      opacity: transSaving ? 0.5 : 1, width: '100%',
                    }}
                  >
                    {transSaving ? 'Saving...' : 'Save Transcription Settings'}
                  </button>
                )}
              </div>

              <ModelDropdown
                task="vision"
                models={availableModels.vision}
                pendingValue={pendingModels.vision_model}
                savedValue={currentModels.vision_model}
                label="Vision AI"
                desc="Analyzes video frames for visual content, importance, and social media potential. Requires a vision-capable model."
              />
              <ModelDropdown
                task="text"
                models={availableModels.text}
                pendingValue={pendingModels.text_model}
                savedValue={currentModels.text_model}
                label="Text AI"
                desc="Generates content summaries and detects viral clip candidates. Any text model works — smarter models find better clips."
              />

              {/* ── Save Button ── */}
              <div style={{
                display: 'flex', alignItems: 'center', gap: 12,
                padding: '14px 0', borderTop: '1px solid var(--border)', marginTop: 4,
              }}>
                <button
                  onClick={handleSaveAllModels}
                  disabled={!modelsHaveChanges || modelsSaving}
                  style={{
                    padding: '10px 28px',
                    background: modelsHaveChanges ? 'var(--accent-cyan)' : 'var(--bg-elevated)',
                    color: modelsHaveChanges ? 'var(--bg-base)' : 'var(--text-muted)',
                    border: modelsHaveChanges ? 'none' : '1px solid var(--border)',
                    borderRadius: 'var(--radius-sm)', fontSize: 13, fontWeight: 600,
                    opacity: (!modelsHaveChanges || modelsSaving) ? 0.5 : 1,
                    transition: 'all 0.15s ease',
                  }}
                >
                  {modelsSaving ? 'Saving...' : 'Save Models'}
                </button>
                {modelsHaveChanges && !modelsSaving && (
                  <span style={{ fontSize: 11, color: 'var(--accent-amber)' }}>
                    You have unsaved changes
                  </span>
                )}
                {!modelsHaveChanges && !modelsSaving && currentModels.vision_model && (
                  <span style={{ fontSize: 11, color: 'var(--success)' }}>
                    All models saved
                  </span>
                )}
              </div>
            </>
          )}
        </div>
      )}

      {/* ═══════ Tab 1: Prompts ═══════ */}
      {settingsTab === 1 && (
        <div style={{ maxWidth: isMobile ? '100%' : 640 }}>
          <h3 style={{ fontSize: 14, marginBottom: 8, color: 'var(--text-secondary)' }}>
            AI Analysis Prompts
          </h3>
          <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 20, lineHeight: 1.6 }}>
            Customize the instructions sent to the AI when analyzing your videos.
            The JSON output format and clip duration constraints are enforced separately.
          </p>

          {/* Subject Tracking Toggle */}
          <div style={{
            background: 'var(--bg-panel)', border: '1px solid var(--border)',
            borderRadius: 'var(--radius-md)', padding: '14px 18px', marginBottom: 24,
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div>
                <h4 style={{ fontSize: 13, margin: 0, color: 'var(--text-primary)' }}>
                  Intelligent Dynamic Subject Tracking
                </h4>
                <p style={{ fontSize: 11, color: 'var(--text-muted)', margin: '4px 0 0', lineHeight: 1.5 }}>
                  AI tracks the main subject across frames to keep it centered when cropping to different aspect ratios.
                </p>
              </div>
              <button
                onClick={() => handleToggleSubjectTracking(!subjectTrackingEnabled)}
                disabled={subjectTrackingSaving}
                style={{
                  position: 'relative', width: 44, height: 24, borderRadius: 12, border: 'none',
                  background: subjectTrackingEnabled ? 'var(--accent-cyan)' : 'var(--bg-elevated)',
                  cursor: subjectTrackingSaving ? 'default' : 'pointer', flexShrink: 0, marginLeft: 16,
                  transition: 'background 0.2s',
                  opacity: subjectTrackingSaving ? 0.5 : 1,
                }}
              >
                <div style={{
                  position: 'absolute', top: 3, left: subjectTrackingEnabled ? 23 : 3,
                  width: 18, height: 18, borderRadius: '50%', background: 'var(--nav-active-icon-text)',
                  transition: 'left 0.2s', boxShadow: '0 1px 3px rgba(0,0,0,0.2)',
                }} />
              </button>
            </div>
            <div style={{
              marginTop: 8, fontSize: 10, fontFamily: 'var(--font-mono)',
              color: subjectTrackingEnabled ? 'var(--accent-cyan)' : 'var(--text-muted)',
            }}>
              {subjectTrackingEnabled ? 'Enabled — subjects will be tracked and centered during crop' : 'Disabled — crops will use center of frame'}
            </div>
          </div>

          {!promptsLoaded ? (
            <div style={{ padding: 16, color: 'var(--text-muted)', fontSize: 12 }}>Loading prompts...</div>
          ) : (
            <>
              {/* Frame Analysis Prompt */}
              <div style={{ marginBottom: 28 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
                  <h4 style={{ fontSize: 13, margin: 0, color: 'var(--text-primary)' }}>Scene / Frame Analysis</h4>
                  <button
                    onClick={() => setPrompts((prev) => ({ ...prev, frame_analysis: promptDefaults.frame_analysis }))}
                    disabled={prompts.frame_analysis === promptDefaults.frame_analysis}
                    style={{
                      padding: '3px 10px', background: 'var(--bg-elevated)', color: 'var(--text-secondary)',
                      border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)', fontSize: 11,
                      opacity: prompts.frame_analysis === promptDefaults.frame_analysis ? 0.4 : 1,
                    }}
                  >
                    Reset to Default
                  </button>
                </div>
                <p style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 8, lineHeight: 1.5 }}>
                  Tells the AI what to look for in each video frame.
                </p>
                <textarea
                  value={prompts.frame_analysis}
                  onChange={(e) => setPrompts((prev) => ({ ...prev, frame_analysis: e.target.value }))}
                  rows={8}
                  style={{
                    width: '100%', minHeight: 120, maxHeight: 400, padding: '10px 12px', resize: 'vertical',
                    background: 'var(--bg-panel)', border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)',
                    color: 'var(--text-primary)', fontSize: 12, fontFamily: 'var(--font-mono)', lineHeight: 1.6, boxSizing: 'border-box',
                  }}
                />
                <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 4 }}>
                  <span style={{ fontSize: 10, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                    {prompts.frame_analysis.length.toLocaleString()} / 10,000
                  </span>
                  {prompts.frame_analysis !== promptDefaults.frame_analysis && (
                    <span style={{ fontSize: 10, color: 'var(--accent-amber)' }}>Modified</span>
                  )}
                </div>
              </div>

              {/* Video Summary Prompt */}
              <div style={{ marginBottom: 28 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
                  <h4 style={{ fontSize: 13, margin: 0, color: 'var(--text-primary)' }}>Video Summary</h4>
                  <button
                    onClick={() => setPrompts((prev) => ({ ...prev, summary: promptDefaults.summary }))}
                    disabled={prompts.summary === promptDefaults.summary}
                    style={{
                      padding: '3px 10px', background: 'var(--bg-elevated)', color: 'var(--text-secondary)',
                      border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)', fontSize: 11,
                      opacity: prompts.summary === promptDefaults.summary ? 0.4 : 1,
                    }}
                  >
                    Reset to Default
                  </button>
                </div>
                <p style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 8, lineHeight: 1.5 }}>
                  Controls how the AI summarizes your video — the overview, topics, tone, audience, and category.
                  Write in a conversational style to get natural, human-readable summaries.
                </p>
                <textarea
                  value={prompts.summary}
                  onChange={(e) => setPrompts((prev) => ({ ...prev, summary: e.target.value }))}
                  rows={8}
                  style={{
                    width: '100%', minHeight: 120, maxHeight: 400, padding: '10px 12px', resize: 'vertical',
                    background: 'var(--bg-panel)', border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)',
                    color: 'var(--text-primary)', fontSize: 12, fontFamily: 'var(--font-mono)', lineHeight: 1.6, boxSizing: 'border-box',
                  }}
                />
                <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 4 }}>
                  <span style={{ fontSize: 10, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                    {prompts.summary.length.toLocaleString()} / 10,000
                  </span>
                  {prompts.summary !== promptDefaults.summary && (
                    <span style={{ fontSize: 10, color: 'var(--accent-amber)' }}>Modified</span>
                  )}
                </div>
              </div>

              {/* SEO / Tags Prompt */}
              <div style={{ marginBottom: 28 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
                  <h4 style={{ fontSize: 13, margin: 0, color: 'var(--text-primary)' }}>SEO / Titles &amp; Tags</h4>
                  <button
                    onClick={() => setPrompts((prev) => ({ ...prev, seo: promptDefaults.seo }))}
                    disabled={prompts.seo === promptDefaults.seo}
                    style={{
                      padding: '3px 10px', background: 'var(--bg-elevated)', color: 'var(--text-secondary)',
                      border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)', fontSize: 11,
                      opacity: prompts.seo === promptDefaults.seo ? 0.4 : 1,
                    }}
                  >
                    Reset to Default
                  </button>
                </div>
                <p style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 8, lineHeight: 1.5 }}>
                  Controls how the AI generates clip titles, descriptions, hashtags, and platform tips.
                  Write in a casual, authentic style to get natural social media captions instead of robotic marketing copy.
                </p>
                <textarea
                  value={prompts.seo}
                  onChange={(e) => setPrompts((prev) => ({ ...prev, seo: e.target.value }))}
                  rows={8}
                  style={{
                    width: '100%', minHeight: 120, maxHeight: 400, padding: '10px 12px', resize: 'vertical',
                    background: 'var(--bg-panel)', border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)',
                    color: 'var(--text-primary)', fontSize: 12, fontFamily: 'var(--font-mono)', lineHeight: 1.6, boxSizing: 'border-box',
                  }}
                />
                <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 4 }}>
                  <span style={{ fontSize: 10, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                    {prompts.seo.length.toLocaleString()} / 10,000
                  </span>
                  {prompts.seo !== promptDefaults.seo && (
                    <span style={{ fontSize: 10, color: 'var(--accent-amber)' }}>Modified</span>
                  )}
                </div>
              </div>

              {/* Viral Algorithm Section */}
              <div ref={viralAlgorithmRef} style={{
                background: 'var(--bg-panel)', border: '1px solid var(--border)',
                borderRadius: 'var(--radius-md)', padding: '14px 18px', marginBottom: 24,
              }}>
                <h4 style={{ fontSize: 13, margin: '0 0 8px', color: 'var(--text-primary)' }}>
                  Viral Algorithm
                </h4>
                <p style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 12, lineHeight: 1.5 }}>
                  This is the core prompt that controls how the AI identifies viral-worthy clips.
                  Edit the strategy below to change what the AI looks for — hooks, engagement patterns,
                  platform targeting, scoring criteria. Changes are saved and persist across container restarts.
                </p>
                <div style={{ display: 'flex', justifyContent: 'flex-end', marginBottom: 6 }}>
                  <button
                    onClick={() => setPrompts((prev) => ({ ...prev, viral_clip_detection: promptDefaults.viral_clip_detection }))}
                    disabled={prompts.viral_clip_detection === promptDefaults.viral_clip_detection}
                    style={{
                      padding: '3px 10px', background: 'var(--bg-elevated)', color: 'var(--text-secondary)',
                      border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)', fontSize: 11,
                      opacity: prompts.viral_clip_detection === promptDefaults.viral_clip_detection ? 0.4 : 1,
                    }}
                  >
                    Reset to Default
                  </button>
                </div>
                <textarea
                  value={prompts.viral_clip_detection}
                  onChange={(e) => setPrompts((prev) => ({ ...prev, viral_clip_detection: e.target.value }))}
                  rows={12}
                  style={{
                    width: '100%', minHeight: 160, maxHeight: 500, padding: '10px 12px', resize: 'vertical',
                    background: 'var(--bg-base)', border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)',
                    color: 'var(--text-primary)', fontSize: 12, fontFamily: 'var(--font-mono)', lineHeight: 1.6, boxSizing: 'border-box',
                  }}
                />
                <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 4 }}>
                  <span style={{ fontSize: 10, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                    {prompts.viral_clip_detection.length.toLocaleString()} / 10,000
                  </span>
                  {prompts.viral_clip_detection !== promptDefaults.viral_clip_detection && (
                    <span style={{ fontSize: 10, color: 'var(--accent-amber)' }}>Modified — will be used for all future clip generation</span>
                  )}
                </div>
              </div>

              {/* Subject Tracking Prompt */}
              <div style={{ marginBottom: 28, opacity: subjectTrackingEnabled ? 1 : 0.4, pointerEvents: subjectTrackingEnabled ? 'auto' : 'none' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
                  <h4 style={{ fontSize: 13, margin: 0, color: 'var(--text-primary)' }}>Subject Tracking</h4>
                  <button
                    onClick={() => setPrompts((prev) => ({ ...prev, subject_tracking: promptDefaults.subject_tracking }))}
                    disabled={prompts.subject_tracking === promptDefaults.subject_tracking}
                    style={{
                      padding: '3px 10px', background: 'var(--bg-elevated)', color: 'var(--text-secondary)',
                      border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)', fontSize: 11,
                      opacity: prompts.subject_tracking === promptDefaults.subject_tracking ? 0.4 : 1,
                    }}
                  >
                    Reset to Default
                  </button>
                </div>
                <p style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 8, lineHeight: 1.5 }}>
                  Instructions for estimating subject position in each frame. Used for smart cropping across aspect ratios.
                </p>
                <textarea
                  value={prompts.subject_tracking}
                  onChange={(e) => setPrompts((prev) => ({ ...prev, subject_tracking: e.target.value }))}
                  rows={6}
                  style={{
                    width: '100%', minHeight: 100, maxHeight: 400, padding: '10px 12px', resize: 'vertical',
                    background: 'var(--bg-panel)', border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)',
                    color: 'var(--text-primary)', fontSize: 12, fontFamily: 'var(--font-mono)', lineHeight: 1.6, boxSizing: 'border-box',
                  }}
                />
                <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 4 }}>
                  <span style={{ fontSize: 10, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                    {prompts.subject_tracking.length.toLocaleString()} / 10,000
                  </span>
                  {prompts.subject_tracking !== promptDefaults.subject_tracking && (
                    <span style={{ fontSize: 10, color: 'var(--accent-amber)' }}>Modified</span>
                  )}
                </div>
              </div>

              {/* Action buttons */}
              <div style={{ display: 'flex', gap: 8 }}>
                <button
                  onClick={handleSavePrompts}
                  disabled={promptsSaving}
                  style={{
                    padding: '8px 20px', background: 'var(--accent-cyan)', color: 'var(--bg-base)',
                    border: 'none', borderRadius: 'var(--radius-sm)', fontSize: 12, fontWeight: 600, opacity: promptsSaving ? 0.5 : 1,
                  }}
                >
                  {promptsSaving ? 'Saving...' : 'Save Prompts'}
                </button>
                <button
                  onClick={handleResetAllPrompts}
                  disabled={promptsSaving || (prompts.frame_analysis === promptDefaults.frame_analysis && prompts.viral_clip_detection === promptDefaults.viral_clip_detection && prompts.subject_tracking === promptDefaults.subject_tracking && prompts.summary === promptDefaults.summary && prompts.seo === promptDefaults.seo)}
                  style={{
                    padding: '8px 16px', background: 'var(--bg-elevated)', color: 'var(--text-secondary)',
                    border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)', fontSize: 12,
                    opacity: (promptsSaving || (prompts.frame_analysis === promptDefaults.frame_analysis && prompts.viral_clip_detection === promptDefaults.viral_clip_detection && prompts.subject_tracking === promptDefaults.subject_tracking && prompts.summary === promptDefaults.summary && prompts.seo === promptDefaults.seo)) ? 0.4 : 1,
                  }}
                >
                  Reset All to Defaults
                </button>
              </div>
            </>
          )}
        </div>
      )}

      {/* ═══════ Tab 2: Fonts ═══════ */}
      {settingsTab === 2 && (
        <div style={{ maxWidth: isMobile ? '100%' : 640 }}>
          <h3 style={{ fontSize: 14, marginBottom: 8, color: 'var(--text-secondary)' }}>
            Custom Fonts
          </h3>
          <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 20, lineHeight: 1.6 }}>
            Upload custom fonts for use in clip subtitles. Supported formats: TTF, OTF, WOFF, WOFF2, EOT (max 20MB each).
          </p>

          {/* Upload area */}
          <div style={{
            border: '2px dashed var(--border)',
            borderRadius: 'var(--radius-md)',
            padding: '24px',
            textAlign: 'center',
            marginBottom: 24,
            background: 'var(--bg-panel)',
            cursor: fontUploading ? 'default' : 'pointer',
            opacity: fontUploading ? 0.6 : 1,
          }}
            onClick={() => !fontUploading && fontInputRef.current?.click()}
          >
            <input
              ref={fontInputRef}
              type="file"
              accept={FONT_ACCEPT}
              multiple
              style={{ display: 'none' }}
              onChange={handleFontUpload}
            />
            <div style={{ fontSize: 32, marginBottom: 8, opacity: 0.4 }}>Aa</div>
            <div style={{ fontSize: 13, color: 'var(--text-secondary)', marginBottom: 4 }}>
              {fontUploading ? 'Uploading...' : 'Click to upload fonts'}
            </div>
            <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>
              .ttf, .otf, .woff, .woff2, .eot
            </div>
          </div>

          {/* Font list */}
          {customFonts.length > 0 ? (
            <div style={{ display: 'grid', gap: 8 }}>
              {customFonts.map((font) => (
                <div key={font.name} style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  padding: '10px 16px',
                  background: 'var(--bg-panel)',
                  border: '1px solid var(--border)',
                  borderRadius: 'var(--radius-md)',
                }}>
                  <div>
                    <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-primary)' }}>
                      {font.name}
                    </div>
                    <div style={{ fontSize: 10, fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>
                      {font.filename}
                    </div>
                  </div>
                  <button
                    onClick={() => handleDeleteFont(font)}
                    style={{
                      padding: '4px 12px',
                      background: 'var(--danger-dim)',
                      color: 'var(--danger)',
                      border: '1px solid var(--danger)',
                      borderRadius: 'var(--radius-sm)',
                      fontSize: 11,
                      cursor: 'pointer',
                    }}
                  >
                    Remove
                  </button>
                </div>
              ))}
            </div>
          ) : (
            <div style={{
              textAlign: 'center',
              padding: '32px 16px',
              color: 'var(--text-muted)',
              fontSize: 12,
              border: '1px solid var(--border)',
              borderRadius: 'var(--radius-md)',
              background: 'var(--bg-panel)',
            }}>
              No custom fonts uploaded yet. Upload fonts above to use them in clip subtitles.
            </div>
          )}
        </div>
      )}

      {/* ═══════ Tab 3: Presets ═══════ */}
      {settingsTab === 3 && (
        <div style={{ maxWidth: isMobile ? '100%' : 640 }}>
          <h3 style={{ fontSize: 14, marginBottom: 8, color: 'var(--text-secondary)' }}>Clip Setting Presets</h3>
          <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 20, lineHeight: 1.6 }}>
            Manage your saved clip export presets. These are shared across the Analysis and SEO pages.
          </p>
          {presets.length === 0 ? (
            <div style={{
              textAlign: 'center', padding: '32px 16px', color: 'var(--text-muted)',
              fontSize: 12, border: '1px solid var(--border)', borderRadius: 'var(--radius-md)',
              background: 'var(--bg-panel)',
            }}>
              No presets saved yet. Save presets from the Clip Settings panel or the SEO page.
            </div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              {presets.map((p) => (
                <div key={p.id} style={{
                  display: 'flex', alignItems: 'center', gap: 12,
                  padding: '10px 14px', background: 'var(--bg-panel)',
                  border: '1px solid var(--border)', borderRadius: 'var(--radius-md)',
                  flexWrap: 'wrap',
                }}>
                  {editingPresetId === p.id ? (
                    <div style={{ flex: 1, minWidth: 200, display: 'flex', gap: 6, alignItems: 'center' }}>
                      <input
                        type="text" value={editPresetName}
                        onChange={(e) => setEditPresetName(e.target.value)}
                        onKeyDown={(e) => {
                          if (e.key === 'Enter') handleRenamePreset(p.id);
                          if (e.key === 'Escape') setEditingPresetId(null);
                        }}
                        autoFocus
                        style={{
                          flex: 1, padding: '4px 8px', fontSize: 13,
                          background: 'var(--bg-elevated)', color: 'var(--accent-cyan)',
                          border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)',
                          fontFamily: 'var(--font-mono)', outline: 'none',
                        }}
                      />
                      <button
                        onClick={() => handleRenamePreset(p.id)}
                        style={{
                          padding: '4px 10px', fontSize: 11,
                          background: 'var(--accent-cyan)', color: 'var(--bg-base)',
                          border: 'none', borderRadius: 'var(--radius-sm)', cursor: 'pointer',
                        }}
                      >
                        Save
                      </button>
                    </div>
                  ) : (
                    <div style={{ flex: 1, minWidth: 0 }}>
                      <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-primary)' }}>{p.name}</div>
                      <div style={{ fontSize: 10, fontFamily: 'var(--font-mono)', color: 'var(--text-muted)', marginTop: 2 }}>
                        {p.created_at ? new Date(p.created_at).toLocaleDateString() : ''}
                        {p.settings?.aspectRatio ? ` | ${p.settings.aspectRatio}` : ''}
                        {p.settings?.subtitlesEnabled ? ' | subs' : ''}
                      </div>
                    </div>
                  )}
                  <div style={{ display: 'flex', gap: 6 }}>
                    <button
                      onClick={() => { setEditingPresetId(p.id); setEditPresetName(p.name); }}
                      style={{
                        padding: '4px 10px', fontSize: 11, background: 'var(--bg-elevated)',
                        color: 'var(--text-secondary)', border: '1px solid var(--border)',
                        borderRadius: 'var(--radius-sm)', cursor: 'pointer',
                      }}
                    >
                      Rename
                    </button>
                    <button
                      onClick={() => handleExportPreset(p)}
                      style={{
                        padding: '4px 10px', fontSize: 11, background: 'var(--bg-elevated)',
                        color: 'var(--accent-cyan)', border: '1px solid var(--border)',
                        borderRadius: 'var(--radius-sm)', cursor: 'pointer',
                      }}
                    >
                      Export
                    </button>
                    <button
                      onClick={() => handleDeletePresetSettings(p.id)}
                      style={{
                        padding: '4px 10px', fontSize: 11, background: 'var(--danger-dim)',
                        color: 'var(--danger)', border: '1px solid var(--danger)',
                        borderRadius: 'var(--radius-sm)', cursor: 'pointer',
                      }}
                    >
                      Delete
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* ── Track Property Presets ── */}
          <h3 style={{ fontSize: 14, marginBottom: 8, marginTop: 32, color: 'var(--text-secondary)' }}>Track Property Presets</h3>
          <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 20, lineHeight: 1.6 }}>
            Manage saved track property presets. These are saved locally and can be applied to matching track types in the video editor.
          </p>
          {trackPresets.length === 0 ? (
            <div style={{
              textAlign: 'center', padding: '32px 16px', color: 'var(--text-muted)',
              fontSize: 12, border: '1px solid var(--border)', borderRadius: 'var(--radius-md)',
              background: 'var(--bg-panel)',
            }}>
              No track presets saved yet. Save presets from the Properties panel in the video editor.
            </div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              {trackPresets.map((p) => (
                <div key={p.id} style={{
                  display: 'flex', alignItems: 'center', gap: 12,
                  padding: '10px 14px', background: 'var(--bg-panel)',
                  border: '1px solid var(--border)', borderRadius: 'var(--radius-md)',
                  flexWrap: 'wrap',
                }}>
                  {editingTrackPresetId === p.id ? (
                    <div style={{ flex: 1, minWidth: 200, display: 'flex', gap: 6, alignItems: 'center' }}>
                      <input
                        type="text" value={editTrackPresetName}
                        onChange={(e) => setEditTrackPresetName(e.target.value)}
                        onKeyDown={(e) => {
                          if (e.key === 'Enter') {
                            renameTrackPreset(p.id, editTrackPresetName);
                            setTrackPresets(loadAllPresets());
                            setEditingTrackPresetId(null);
                            showToast('Preset renamed');
                          }
                          if (e.key === 'Escape') setEditingTrackPresetId(null);
                        }}
                        autoFocus
                        style={{
                          flex: 1, padding: '4px 8px', fontSize: 13,
                          background: 'var(--bg-elevated)', color: 'var(--accent-cyan)',
                          border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)',
                          fontFamily: 'var(--font-mono)', outline: 'none',
                        }}
                      />
                      <button
                        onClick={() => {
                          renameTrackPreset(p.id, editTrackPresetName);
                          setTrackPresets(loadAllPresets());
                          setEditingTrackPresetId(null);
                          showToast('Preset renamed');
                        }}
                        style={{
                          padding: '4px 10px', fontSize: 11,
                          background: 'var(--accent-cyan)', color: 'var(--bg-base)',
                          border: 'none', borderRadius: 'var(--radius-sm)', cursor: 'pointer',
                        }}
                      >
                        Save
                      </button>
                    </div>
                  ) : (
                    <div style={{ flex: 1, minWidth: 0 }}>
                      <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-primary)' }}>{p.name}</div>
                      <div style={{ fontSize: 10, fontFamily: 'var(--font-mono)', color: 'var(--text-muted)', marginTop: 2 }}>
                        {p.type} | {p.createdAt ? new Date(p.createdAt).toLocaleDateString() : ''}
                      </div>
                    </div>
                  )}
                  <div style={{ display: 'flex', gap: 6 }}>
                    <button
                      onClick={() => { setEditingTrackPresetId(p.id); setEditTrackPresetName(p.name); }}
                      style={{
                        padding: '4px 10px', fontSize: 11, background: 'var(--bg-elevated)',
                        color: 'var(--text-secondary)', border: '1px solid var(--border)',
                        borderRadius: 'var(--radius-sm)', cursor: 'pointer',
                      }}
                    >
                      Rename
                    </button>
                    <button
                      onClick={() => {
                        deleteTrackPreset(p.id);
                        setTrackPresets(loadAllPresets());
                        showToast('Preset deleted');
                      }}
                      style={{
                        padding: '4px 10px', fontSize: 11, background: 'var(--danger-dim)',
                        color: 'var(--danger)', border: '1px solid var(--danger)',
                        borderRadius: 'var(--radius-sm)', cursor: 'pointer',
                      }}
                    >
                      Delete
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* ═══════ Tab 4: Advanced ═══════ */}
      {settingsTab === 4 && (
        <div>
          <div style={{ maxWidth: isMobile ? '100%' : 480 }}>

            {/* ── Cloud Storage ── */}
            <div style={{ marginBottom: 32 }}>
              <h3 style={{ fontSize: 14, marginBottom: 16, color: 'var(--text-secondary)' }}>Cloud Storage</h3>
              <CloudStorageSection />
            </div>

            {/* ── GPU Hardware Acceleration ── */}
            <div style={{ marginBottom: 32 }}>
              <h3 style={{ fontSize: 14, marginBottom: 16, color: 'var(--text-secondary)' }}>GPU Hardware Acceleration</h3>

              {/* Toggle row */}
              <div style={{
                display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                padding: '12px 16px', background: 'var(--bg-panel)',
                border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)',
              }}>
                <div style={{ flex: 1 }}>
                  <span style={{ fontSize: 13, color: 'var(--text-primary)' }}>Enable GPU Acceleration</span>
                  <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 4 }}>
                    Uses your GPU for faster video encoding, decoding, and transcription.
                    Requires a compatible NVIDIA, AMD, or Intel GPU passed through to the container.
                  </div>
                </div>
                <button
                  onClick={() => handleToggleGpu(!gpuEnabled)}
                  disabled={gpuSaving}
                  style={{
                    width: 44, height: 24, borderRadius: 12, border: 'none', position: 'relative',
                    background: gpuEnabled ? 'var(--accent-cyan)' : 'var(--bg-elevated)',
                    cursor: gpuSaving ? 'default' : 'pointer', flexShrink: 0, marginLeft: 16,
                    transition: 'background 0.2s', opacity: gpuSaving ? 0.5 : 1,
                  }}
                >
                  <div style={{
                    position: 'absolute', top: 3, left: gpuEnabled ? 23 : 3,
                    width: 18, height: 18, borderRadius: '50%', background: 'var(--nav-active-icon-text)',
                    transition: 'left 0.2s', boxShadow: '0 1px 3px rgba(0,0,0,0.2)',
                  }} />
                </button>
              </div>

              {/* Status text */}
              <div style={{
                marginTop: 8, fontSize: 10, fontFamily: 'var(--font-mono)',
                color: gpuEnabled ? 'var(--accent-cyan)' : 'var(--text-muted)',
              }}>
                {gpuSaving ? 'Detecting GPU...' :
                 gpuEnabled ? 'Enabled — GPU will be used for encoding, decoding, and transcription' :
                 'Disabled — using CPU for all processing'}
              </div>

              {/* GPU Info Card — shown when enabled */}
              {gpuEnabled && gpuInfo && (
                <div style={{
                  marginTop: 12, padding: '12px 16px',
                  background: gpuInfo.vendor !== 'none' ? 'rgba(0, 217, 255, 0.05)' : 'rgba(255, 165, 0, 0.05)',
                  border: `1px solid ${gpuInfo.vendor !== 'none' ? 'rgba(0, 217, 255, 0.2)' : 'rgba(255, 165, 0, 0.2)'}`,
                  borderRadius: 'var(--radius-sm)',
                }}>
                  {gpuLoading ? (
                    <div style={{ fontSize: 12, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                      Scanning for GPU hardware...
                    </div>
                  ) : gpuInfo.vendor !== 'none' ? (
                    /* GPU detected — show details */
                    <div style={{ display: 'grid', gap: 6 }}>
                      {/* All detected GPUs */}
                      {gpuInfo.gpus?.length > 0 ? (
                        <>
                          <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 2 }}>
                            Detected GPUs ({gpuInfo.gpus.length})
                          </div>
                          {gpuInfo.gpus.map((gpu, i) => (
                            <div key={i} style={{
                              display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                              fontSize: 12, padding: '4px 8px',
                              background: 'rgba(255, 255, 255, 0.03)', borderRadius: 4,
                            }}>
                              <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-primary)' }}>{String(gpu.name || '')}</span>
                              <span style={{
                                fontSize: 10, padding: '1px 6px', borderRadius: 3,
                                fontFamily: 'var(--font-mono)',
                                background: gpu.vendor === 'nvidia' ? 'rgba(118, 185, 0, 0.15)' : 'rgba(0, 114, 198, 0.15)',
                                color: gpu.vendor === 'nvidia' ? '#76b900' : '#0072c6',
                              }}>
                                {String(gpu.vendor || '').toUpperCase()} · {gpu.type === 'discrete' ? 'Discrete' : gpu.type === 'integrated' ? 'Integrated' : 'GPU'}
                                {gpu.vram_mb > 0 ? ` · ${gpu.vram_mb} MB` : ''}
                              </span>
                            </div>
                          ))}
                          {/* GPU device selector for FFmpeg — shown when multiple GPUs available */}
                          {gpuInfo.gpus.length > 1 && (
                            <div style={{ marginTop: 8 }}>
                              <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 4 }}>Select GPU for Encoding</div>
                              <select
                                defaultValue={gpuInfo.gpus[0]?.index || '0'}
                                onChange={async (e) => {
                                  try {
                                    await fetch('/api/client-gpu-report', {
                                      method: 'POST',
                                      headers: { 'Content-Type': 'application/json' },
                                      body: JSON.stringify({
                                        gpu_index: e.target.value,
                                        gpu_name: gpuInfo.gpus.find(g => g.index === e.target.value)?.name || '',
                                        gpu_vendor: gpuInfo.gpus.find(g => g.index === e.target.value)?.vendor || '',
                                      }),
                                    });
                                    showToast(`GPU ${e.target.value} selected for encoding`, 'success');
                                  } catch { showToast('Failed to update GPU selection', 'error'); }
                                }}
                                style={{
                                  width: '100%', padding: '6px 10px', fontSize: 12,
                                  fontFamily: 'var(--font-mono)',
                                  background: 'var(--bg-base)', color: 'var(--text-primary)',
                                  border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)',
                                }}
                              >
                                {gpuInfo.gpus.map((gpu) => (
                                  <option key={gpu.index} value={gpu.index}>
                                    GPU {gpu.index}: {gpu.name} ({gpu.vendor.toUpperCase()}{gpu.vram_mb > 0 ? ` · ${gpu.vram_mb} MB` : ''})
                                  </option>
                                ))}
                              </select>
                            </div>
                          )}
                        </>
                      ) : (
                        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12 }}>
                          <span style={{ color: 'var(--text-muted)' }}>GPU</span>
                          <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-primary)' }}>{String(gpuInfo.gpu_name || '')}</span>
                        </div>
                      )}
                      {/* Active encoder/decoder info */}
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, marginTop: 4 }}>
                        <span style={{ color: 'var(--text-muted)' }}>Video Encoder</span>
                        <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--success)' }}>{String(gpuInfo.encoder || '')}</span>
                      </div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12 }}>
                        <span style={{ color: 'var(--text-muted)' }}>Video Decoder</span>
                        <span style={{ fontFamily: 'var(--font-mono)', color: gpuInfo.decoder ? 'var(--success)' : 'var(--text-muted)' }}>
                          {String(gpuInfo.decoder || 'N/A')}
                        </span>
                      </div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12 }}>
                        <span style={{ color: 'var(--text-muted)' }}>Whisper (Transcription)</span>
                        <span style={{ fontFamily: 'var(--font-mono)', color: (gpuInfo.cuda_available || gpuInfo.vendor === 'apple') ? 'var(--success)' : 'var(--text-muted)' }}>
                          {gpuInfo.cuda_available ? 'CUDA (GPU)' : gpuInfo.vendor === 'apple' ? 'Core ML (GPU)' : 'CPU'}
                        </span>
                      </div>
                      {gpuInfo.driver_version && (
                        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12 }}>
                          <span style={{ color: 'var(--text-muted)' }}>Driver</span>
                          <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)' }}>{String(gpuInfo.driver_version || '')}</span>
                        </div>
                      )}
                      {/* GPU Issues / Setup Guidance */}
                      {gpuInfo.gpu_issues && gpuInfo.gpu_issues.length > 0 && (
                        <div style={{
                          marginTop: 10, padding: '8px 10px',
                          background: 'rgba(255, 159, 10, 0.08)',
                          border: '1px solid rgba(255, 159, 10, 0.2)',
                          borderRadius: 'var(--radius-sm)',
                        }}>
                          <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--warning)', marginBottom: 4 }}>
                            GPU detected but not fully available for encoding
                          </div>
                          {gpuInfo.gpu_issues.map((issue, i) => (
                            <div key={i} style={{ fontSize: 10, color: 'var(--text-muted)', lineHeight: 1.5, marginBottom: 2 }}>
                              {'• '}{issue}
                            </div>
                          ))}
                          <div style={{ fontSize: 10, color: 'var(--text-muted)', lineHeight: 1.5, marginTop: 6, borderTop: '1px solid rgba(255,159,10,0.1)', paddingTop: 6 }}>
                            <strong>Docker GPU Passthrough Setup:</strong><br/>
                            {'1. '}Install nvidia-container-toolkit on host:<br/>
                            <code style={{ fontSize: 9, background: 'var(--bg-elevated)', padding: '1px 4px', borderRadius: 3, display: 'inline-block', marginLeft: 12 }}>
                              apt install nvidia-container-toolkit && systemctl restart docker
                            </code><br/>
                            {'2. '}Run with GPU access:<br/>
                            <code style={{ fontSize: 9, background: 'var(--bg-elevated)', padding: '1px 4px', borderRadius: 3, display: 'inline-block', marginLeft: 12 }}>
                              docker run --gpus all --runtime=nvidia ...
                            </code><br/>
                            {'3. '}Use a CUDA-enabled image with FFmpeg NVENC support
                          </div>
                        </div>
                      )}
                    </div>
                  ) : (
                    /* No GPU detected — show setup instructions */
                    <div>
                      <div style={{ fontSize: 12, color: 'var(--warning)', marginBottom: 6, fontWeight: 600 }}>
                        No compatible GPU detected
                      </div>
                      <div style={{ fontSize: 11, color: 'var(--text-muted)', lineHeight: 1.5 }}>
                        To use GPU acceleration:<br/>
                        {'• '}Pass your GPU to the container via <code style={{ fontSize: 10, background: 'var(--bg-elevated)', padding: '1px 4px', borderRadius: 3 }}>docker-compose.gpu.yml</code><br/>
                        {'• '}NVIDIA: Install nvidia-container-toolkit on the host<br/>
                        {'• '}Intel/AMD: Pass <code style={{ fontSize: 10, background: 'var(--bg-elevated)', padding: '1px 4px', borderRadius: 3 }}>/dev/dri</code> device to the container<br/>
                        {'• '}macOS (Apple Silicon): Run natively (not Docker) — VideoToolbox requires direct macOS access<br/>
                        {'• '}Rebuild with <code style={{ fontSize: 10, background: 'var(--bg-elevated)', padding: '1px 4px', borderRadius: 3 }}>Dockerfile.gpu</code> for hardware encoder support (Linux/NVIDIA)
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* ── Pipeline Diagnostics ── */}
            <PipelineDiagnostics />

            {/* ── Subject Tracking Validation ── */}
            {subjectTrackingEnabled && (
              <div style={{
                background: 'var(--bg-panel)', border: '1px solid var(--border)',
                borderRadius: 'var(--radius-md)', padding: '14px 18px', marginBottom: 32,
              }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                  <div>
                    <h4 style={{ fontSize: 13, margin: 0, color: 'var(--text-primary)' }}>
                      Subject Tracking Validation
                    </h4>
                    <p style={{ fontSize: 11, color: 'var(--text-muted)', margin: '4px 0 0', lineHeight: 1.5 }}>
                      Tests whether your vision AI can detect a moving subject across 5 positions (far left to far right) using synthetic images.
                    </p>
                  </div>
                  <button
                    onClick={handleTestSubjectTracking}
                    disabled={trackingTestRunning}
                    style={{
                      padding: '6px 14px', fontSize: 11, fontWeight: 500,
                      background: trackingTestRunning ? 'var(--bg-elevated)' : 'var(--accent-cyan)',
                      color: trackingTestRunning ? 'var(--text-muted)' : '#fff',
                      border: 'none', borderRadius: 'var(--radius-sm)',
                      cursor: trackingTestRunning ? 'default' : 'pointer',
                      opacity: trackingTestRunning ? 0.6 : 1,
                      flexShrink: 0, marginLeft: 16,
                    }}
                  >
                    {trackingTestRunning ? 'Testing...' : 'Run Test'}
                  </button>
                </div>

                {trackingTestRunning && (
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '12px 0' }}>
                    <div style={{
                      width: 16, height: 16,
                      border: '2px solid var(--border)', borderTopColor: 'var(--accent-cyan)',
                      borderRadius: '50%', animation: 'spin 0.8s linear infinite',
                    }} />
                    <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                      Sending 5 test images to vision model... (may take 20-60s)
                    </span>
                  </div>
                )}

                {trackingTest && !trackingTestRunning && (
                  <div>
                    <div style={{
                      display: 'flex', alignItems: 'center', gap: 8, padding: '8px 12px',
                      background: trackingTest.overall_status === 'pass' ? 'rgba(16,185,129,0.08)'
                        : trackingTest.overall_status === 'warn' ? 'rgba(245,158,11,0.08)'
                        : 'rgba(239,68,68,0.08)',
                      borderRadius: 'var(--radius-sm)', marginBottom: 10,
                    }}>
                      <span style={{ fontSize: 14 }}>
                        {trackingTest.overall_status === 'pass' ? '\u2705'
                          : trackingTest.overall_status === 'warn' ? '\u26a0\ufe0f' : '\u274c'}
                      </span>
                      <span style={{
                        fontSize: 12, fontWeight: 500,
                        color: trackingTest.overall_status === 'pass' ? 'var(--success)'
                          : trackingTest.overall_status === 'warn' ? '#f59e0b' : '#ef4444',
                      }}>
                        {trackingTest.summary}
                      </span>
                    </div>

                    <div style={{ display: 'flex', gap: 16, marginBottom: 10, flexWrap: 'wrap' }}>
                      <div style={{ fontSize: 10, fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>
                        Model: <span style={{ color: 'var(--text-secondary)' }}>{trackingTest.model || '?'}</span>
                      </div>
                      <div style={{ fontSize: 10, fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>
                        JSON: <span style={{ color: 'var(--text-secondary)' }}>{trackingTest.json_compliance || '?'}</span>
                      </div>
                      <div style={{ fontSize: 10, fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>
                        Avg error: <span style={{ color: 'var(--text-secondary)' }}>
                          {trackingTest.avg_error != null ? `${trackingTest.avg_error}%` : 'N/A'}
                        </span>
                      </div>
                      <div style={{ fontSize: 10, fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>
                        Speed: <span style={{ color: 'var(--text-secondary)' }}>
                          {trackingTest.avg_speed_ms ? `${(trackingTest.avg_speed_ms / 1000).toFixed(1)}s/frame` : 'N/A'}
                        </span>
                      </div>
                      {trackingTest.format_json_used && (
                        <div style={{ fontSize: 10, fontFamily: 'var(--font-mono)', color: 'var(--accent-cyan)' }}>
                          format:json active
                        </div>
                      )}
                    </div>

                    {trackingTest.results?.map((r, i) => (
                      <div key={i} style={{
                        display: 'flex', alignItems: 'center', gap: 8,
                        padding: '5px 0', borderTop: i > 0 ? '1px solid rgba(128,128,128,0.1)' : 'none',
                      }}>
                        <span style={{ fontSize: 12, width: 18, textAlign: 'center', flexShrink: 0 }}>
                          {r.status === 'pass' ? '\u2705' : r.status === 'warn' ? '\u26a0\ufe0f' : '\u274c'}
                        </span>
                        <span style={{
                          fontSize: 11, fontWeight: 500, width: 50, flexShrink: 0,
                          color: 'var(--text-primary)', textTransform: 'capitalize',
                        }}>
                          {r.label}
                        </span>
                        <span style={{ fontSize: 10, fontFamily: 'var(--font-mono)', color: 'var(--text-muted)', flex: 1 }}>
                          {r.message}
                        </span>
                        <span style={{ fontSize: 10, fontFamily: 'var(--font-mono)', color: 'var(--text-muted)', flexShrink: 0 }}>
                          {r.duration_ms}ms
                        </span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}

            {/* ── Client GPU (Browser) ── */}
            <div style={{ marginBottom: 32 }}>
              <h3 style={{ fontSize: 14, marginBottom: 4, color: 'var(--text-secondary)' }}>Client GPU (Browser)</h3>
              <p style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 16, lineHeight: 1.5 }}>
                Use your desktop PC's GPU directly through Chrome for transcription and video encoding.
                This uses WebGPU and WebCodecs APIs — processing happens in your browser, not on the server.
              </p>

              {/* Main toggle */}
              <div style={{
                display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                padding: '12px 16px', background: 'var(--bg-panel)',
                border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)',
              }}>
                <div style={{ flex: 1 }}>
                  <span style={{ fontSize: 13, color: 'var(--text-primary)' }}>Enable Client-Side GPU Processing</span>
                  <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 4 }}>
                    Leverages your local GPU via Chrome WebGPU for Whisper transcription
                    and WebCodecs for hardware video encoding (NVENC/VideoToolbox/VAAPI).
                  </div>
                </div>
                <button
                  onClick={() => handleToggleClientGpu(!clientGpuEnabled)}
                  disabled={clientGpuScanning || (!clientGpuInfo && !clientGpuEnabled)}
                  style={{
                    width: 44, height: 24, borderRadius: 12, border: 'none', position: 'relative',
                    background: clientGpuEnabled ? 'var(--accent-cyan)' : 'var(--bg-elevated)',
                    cursor: (clientGpuScanning || (!clientGpuInfo && !clientGpuEnabled)) ? 'default' : 'pointer',
                    flexShrink: 0, marginLeft: 16,
                    transition: 'background 0.2s',
                    opacity: (clientGpuScanning || (!clientGpuInfo && !clientGpuEnabled)) ? 0.5 : 1,
                  }}
                >
                  <div style={{
                    position: 'absolute', top: 3, left: clientGpuEnabled ? 23 : 3,
                    width: 18, height: 18, borderRadius: '50%', background: 'var(--nav-active-icon-text)',
                    transition: 'left 0.2s', boxShadow: '0 1px 3px rgba(0,0,0,0.2)',
                  }} />
                </button>
              </div>

              {/* Status line */}
              <div style={{
                marginTop: 8, fontSize: 10, fontFamily: 'var(--font-mono)',
                color: clientGpuScanning ? 'var(--text-muted)' :
                       clientGpuEnabled ? 'var(--accent-cyan)' :
                       clientGpuInfo?.gpus?.length > 0 ? 'var(--success)' :
                       'var(--text-muted)',
              }}>
                {clientGpuScanning ? 'Scanning browser GPU capabilities...' :
                 clientGpuEnabled ? `Enabled — using ${clientGpuInfo?.gpus?.find(g => g.id === selectedClientGpuId)?.name || 'detected GPU'}` :
                 clientGpuInfo?.gpus?.length > 0 ? `${clientGpuInfo.gpus.length} GPU(s) detected — toggle on to enable` :
                 'Disabled — all processing runs on the server'}
              </div>

              {/* Detected GPUs summary — shown before toggle enabled so user can see what was found */}
              {!clientGpuScanning && !clientGpuEnabled && clientGpuInfo?.gpus?.length > 0 && (
                <div style={{
                  marginTop: 12, padding: '12px 16px',
                  background: 'rgba(0, 217, 255, 0.05)',
                  border: '1px solid rgba(0, 217, 255, 0.2)',
                  borderRadius: 'var(--radius-sm)',
                }}>
                  <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 8 }}>Detected GPUs</div>
                  <div style={{ display: 'grid', gap: 6 }}>
                    {clientGpuInfo.gpus.map(gpu => (
                      <div key={gpu.id} style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12 }}>
                        <span style={{ color: 'var(--text-primary)', fontFamily: 'var(--font-mono)' }}>{String(gpu.name || '')}</span>
                        <span style={{
                          fontSize: 10, padding: '1px 6px', borderRadius: 3,
                          background: gpu.backend === 'webgpu' ? 'rgba(0, 217, 255, 0.15)' :
                                     gpu.backend === 'server-detected' ? 'rgba(118, 185, 0, 0.15)' : 'rgba(255, 255, 255, 0.08)',
                          color: gpu.backend === 'webgpu' ? 'var(--accent-cyan)' :
                                 gpu.backend === 'server-detected' ? '#76b900' : 'var(--text-muted)',
                          fontFamily: 'var(--font-mono)',
                        }}>
                          {gpu.backend === 'webgpu' ? 'WebGPU' : gpu.backend === 'webgpu-software' ? 'Software' : gpu.backend === 'server-detected' ? 'Server' : 'WebGL'}
                          {' · '}{gpu.type === 'discrete' ? 'Discrete' : gpu.type === 'integrated' ? 'Integrated' : gpu.type === 'software' ? 'Software' : 'GPU'}
                          {gpu.estimatedVRAM_MB > 0 ? ` · ${gpu.estimatedVRAM_MB} MB` : ''}
                        </span>
                      </div>
                    ))}
                  </div>
                  {clientGpuInfo.webcodecs?.h264HardwareEncode && (
                    <div style={{ marginTop: 8, fontSize: 10, color: 'var(--success)', fontFamily: 'var(--font-mono)' }}>
                      H.264 hardware encoding available via WebCodecs
                    </div>
                  )}
                  {!clientGpuInfo.webgpuSupported && (
                    <div style={{ marginTop: 8, fontSize: 10, color: 'var(--text-muted)', lineHeight: 1.5 }}>
                      WebGPU not available — Whisper transcription will use CPU. Video encoding can still use hardware via WebCodecs.
                    </div>
                  )}
                </div>
              )}

              {/* WebGPU Not Supported — browser setup instructions (only when no GPUs detected at all) */}
              {!clientGpuScanning && clientGpuInfo && !clientGpuInfo.webgpuSupported && clientGpuInfo.gpus?.length === 0 && (
                <div style={{
                  marginTop: 12, padding: '14px 16px',
                  background: 'rgba(255, 165, 0, 0.06)',
                  border: '1px solid rgba(255, 165, 0, 0.25)',
                  borderRadius: 'var(--radius-sm)',
                }}>
                  <div style={{ fontSize: 13, color: 'var(--warning)', marginBottom: 10, fontWeight: 600 }}>
                    WebGPU is not enabled in your browser
                  </div>
                  <div style={{ fontSize: 11, color: 'var(--text-secondary)', lineHeight: 1.7, marginBottom: 12 }}>
                    Client-side GPU processing requires WebGPU, a modern browser API for GPU compute.
                    Follow the steps below for your browser to enable it.
                  </div>

                  {/* Chrome */}
                  <div style={{ marginBottom: 14 }}>
                    <div style={{ fontSize: 12, color: 'var(--text-primary)', fontWeight: 600, marginBottom: 4 }}>
                      Google Chrome (Recommended)
                    </div>
                    <div style={{ fontSize: 11, color: 'var(--text-muted)', lineHeight: 1.7 }}>
                      {'1. '}Update to <strong style={{ color: 'var(--text-secondary)' }}>Chrome 113+</strong> (WebGPU is enabled by default on Windows and macOS)<br/>
                      {'2. '}Visit <code style={{ fontSize: 10, background: 'var(--bg-elevated)', padding: '2px 6px', borderRadius: 3, color: 'var(--accent-cyan)' }}>chrome://gpu</code> and
                      look for <strong style={{ color: 'var(--text-secondary)' }}>"WebGPU: Hardware accelerated"</strong><br/>
                      {'3. '}If it says "Disabled" or "Software only", go to <code style={{ fontSize: 10, background: 'var(--bg-elevated)', padding: '2px 6px', borderRadius: 3, color: 'var(--accent-cyan)' }}>chrome://flags/#enable-unsafe-webgpu</code> and set it to <strong style={{ color: 'var(--text-secondary)' }}>Enabled</strong><br/>
                      {'4. '}Relaunch Chrome and reload this page
                    </div>
                  </div>

                  {/* Chrome on Linux */}
                  <div style={{ marginBottom: 14 }}>
                    <div style={{ fontSize: 12, color: 'var(--text-primary)', fontWeight: 600, marginBottom: 4 }}>
                      Chrome on Linux
                    </div>
                    <div style={{ fontSize: 11, color: 'var(--text-muted)', lineHeight: 1.7 }}>
                      WebGPU on Linux requires extra flags:<br/>
                      {'1. '}Enable <code style={{ fontSize: 10, background: 'var(--bg-elevated)', padding: '2px 6px', borderRadius: 3, color: 'var(--accent-cyan)' }}>chrome://flags/#enable-unsafe-webgpu</code><br/>
                      {'2. '}Enable <code style={{ fontSize: 10, background: 'var(--bg-elevated)', padding: '2px 6px', borderRadius: 3, color: 'var(--accent-cyan)' }}>chrome://flags/#enable-vulkan</code> (WebGPU uses Vulkan on Linux)<br/>
                      {'3. '}Ensure your GPU drivers support Vulkan (NVIDIA 470+, Mesa 21.0+ for AMD/Intel)<br/>
                      {'4. '}Relaunch Chrome with: <code style={{ fontSize: 10, background: 'var(--bg-elevated)', padding: '2px 6px', borderRadius: 3, color: 'var(--text-secondary)' }}>google-chrome --enable-features=Vulkan,UseSkiaRenderer</code>
                    </div>
                  </div>

                  {/* Edge */}
                  <div style={{ marginBottom: 14 }}>
                    <div style={{ fontSize: 12, color: 'var(--text-primary)', fontWeight: 600, marginBottom: 4 }}>
                      Microsoft Edge
                    </div>
                    <div style={{ fontSize: 11, color: 'var(--text-muted)', lineHeight: 1.7 }}>
                      {'1. '}Update to <strong style={{ color: 'var(--text-secondary)' }}>Edge 113+</strong><br/>
                      {'2. '}Visit <code style={{ fontSize: 10, background: 'var(--bg-elevated)', padding: '2px 6px', borderRadius: 3, color: 'var(--accent-cyan)' }}>edge://flags/#enable-unsafe-webgpu</code> and set to <strong style={{ color: 'var(--text-secondary)' }}>Enabled</strong><br/>
                      {'3. '}Relaunch Edge and reload this page
                    </div>
                  </div>

                  {/* Firefox */}
                  <div style={{ marginBottom: 14 }}>
                    <div style={{ fontSize: 12, color: 'var(--text-primary)', fontWeight: 600, marginBottom: 4 }}>
                      Firefox
                    </div>
                    <div style={{ fontSize: 11, color: 'var(--text-muted)', lineHeight: 1.7 }}>
                      {'1. '}Update to <strong style={{ color: 'var(--text-secondary)' }}>Firefox 141+</strong> (Nightly has the best support)<br/>
                      {'2. '}Go to <code style={{ fontSize: 10, background: 'var(--bg-elevated)', padding: '2px 6px', borderRadius: 3, color: 'var(--accent-cyan)' }}>about:config</code><br/>
                      {'3. '}Set <code style={{ fontSize: 10, background: 'var(--bg-elevated)', padding: '2px 6px', borderRadius: 3, color: 'var(--text-secondary)' }}>dom.webgpu.enabled</code> to <strong style={{ color: 'var(--text-secondary)' }}>true</strong><br/>
                      {'4. '}Restart Firefox and reload this page<br/>
                      <span style={{ fontSize: 10, color: 'var(--text-muted)', fontStyle: 'italic' }}>
                        Note: Firefox WebGPU support is experimental and may have limited compatibility.
                      </span>
                    </div>
                  </div>

                  {/* Safari */}
                  <div style={{ marginBottom: 14 }}>
                    <div style={{ fontSize: 12, color: 'var(--text-primary)', fontWeight: 600, marginBottom: 4 }}>
                      Safari (macOS)
                    </div>
                    <div style={{ fontSize: 11, color: 'var(--text-muted)', lineHeight: 1.7 }}>
                      {'1. '}Update to <strong style={{ color: 'var(--text-secondary)' }}>Safari 18+</strong> (macOS Sequoia) — WebGPU is enabled by default<br/>
                      {'2. '}On older versions: Safari {'>'} Settings {'>'} Advanced {'>'} check "Show features for web developers"<br/>
                      {'3. '}Then: Develop menu {'>'} Feature Flags {'>'} enable <strong style={{ color: 'var(--text-secondary)' }}>WebGPU</strong><br/>
                      <span style={{ fontSize: 10, color: 'var(--text-muted)', fontStyle: 'italic' }}>
                        Note: Chrome is recommended for best WebGPU + WebCodecs support.
                      </span>
                    </div>
                  </div>

                  {/* General troubleshooting */}
                  <div style={{
                    padding: '8px 12px', background: 'var(--bg-elevated)',
                    borderRadius: 'var(--radius-sm)', fontSize: 11, color: 'var(--text-muted)', lineHeight: 1.6,
                  }}>
                    <strong style={{ color: 'var(--text-secondary)' }}>Troubleshooting:</strong><br/>
                    {'• '}Make sure your GPU drivers are up to date (NVIDIA GeForce Experience / AMD Adrenalin / Intel Arc Control)<br/>
                    {'• '}Verify hardware acceleration is on: Chrome Settings {'>'} System {'>'} "Use hardware acceleration when available"<br/>
                    {'• '}In <code style={{ fontSize: 10, background: 'var(--bg-base)', padding: '1px 4px', borderRadius: 3 }}>chrome://gpu</code>, check that "Graphics Feature Status" shows "Hardware accelerated" for most features<br/>
                    {'• '}Virtual machines, remote desktop, and some enterprise policies may block WebGPU<br/>
                    {'• '}After enabling flags, you must fully relaunch the browser (not just refresh)
                  </div>

                  {/* Rescan button */}
                  <button
                    onClick={async () => {
                      setClientGpuScanning(true);
                      try {
                        const { scanClientGPU } = await import('../utils/clientGpu.js');
                        const info = await scanClientGPU();
                        setClientGpuInfo(info);
                        if (info.gpus.length > 0) {
                          showToast(`Found ${info.gpus.length} GPU(s): ${info.gpus.map(g => g.name).join(', ')}`, 'success');
                        } else if (info.webgpuSupported) {
                          showToast(`WebGPU available but no adapters returned${info.scanErrors ? ' — ' + info.scanErrors[0] : ''}`, 'warning');
                        } else {
                          showToast('WebGPU still not available — check the instructions above', 'warning');
                        }
                      } catch (e) {
                        showToast(`GPU scan failed: ${e?.message || 'unknown error'}`, 'error');
                        setClientGpuInfo({
                          webgpuSupported: !!navigator?.gpu,
                          gpus: [],
                          webcodecs: { h264HardwareEncode: false, hevcHardwareEncode: false, h264Decode: false, hevcDecode: false },
                          recommended: null,
                          scanErrors: [e?.message || 'Scan failed'],
                        });
                      }
                      setClientGpuScanning(false);
                    }}
                    disabled={clientGpuScanning}
                    style={{
                      marginTop: 12, padding: '6px 16px', fontSize: 11,
                      fontFamily: 'var(--font-mono)',
                      background: 'var(--bg-panel)', color: 'var(--accent-cyan)',
                      border: '1px solid var(--accent-cyan)', borderRadius: 'var(--radius-sm)',
                      cursor: 'pointer', opacity: clientGpuScanning ? 0.5 : 1,
                    }}
                  >
                    {clientGpuScanning ? 'Scanning...' : 'Rescan after enabling WebGPU'}
                  </button>
                </div>
              )}

              {/* GPU Details + Selection (when enabled) */}
              {clientGpuEnabled && clientGpuInfo && (
                <div style={{
                  marginTop: 12, padding: '12px 16px',
                  background: clientGpuInfo.gpus.length > 0 ? 'rgba(0, 217, 255, 0.05)' : 'rgba(255, 165, 0, 0.05)',
                  border: `1px solid ${clientGpuInfo.gpus.length > 0 ? 'rgba(0, 217, 255, 0.2)' : 'rgba(255, 165, 0, 0.2)'}`,
                  borderRadius: 'var(--radius-sm)',
                }}>
                  {clientGpuScanning ? (
                    <div style={{ fontSize: 12, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                      Scanning GPUs...
                    </div>
                  ) : clientGpuInfo.gpus.length > 0 ? (
                    <div style={{ display: 'grid', gap: 10 }}>
                      {/* GPU selector — always shown so user can pick their preferred GPU */}
                      <div>
                        <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 6 }}>
                          Select GPU for Passthrough ({clientGpuInfo.gpus.length} detected)
                        </div>
                        <select
                          value={selectedClientGpuId}
                          onChange={(e) => {
                            setSelectedClientGpuId(e.target.value);
                            handleSaveClientGpu({ selectedGpuId: e.target.value });
                          }}
                          style={{
                            width: '100%', padding: '6px 10px', fontSize: 12,
                            fontFamily: 'var(--font-mono)',
                            background: 'var(--bg-base)', color: 'var(--text-primary)',
                            border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)',
                          }}
                        >
                          {clientGpuInfo.gpus.map(gpu => (
                            <option key={gpu.id} value={gpu.id}>
                              {gpu.name} ({gpu.type === 'discrete' ? 'Discrete' : gpu.type === 'integrated' ? 'Integrated' : gpu.type === 'software' ? 'Software' : 'GPU'}
                              {gpu.backend === 'server-detected' ? ' · Server' : ''}
                              {gpu.estimatedVRAM_MB > 0 ? ` · ${gpu.estimatedVRAM_MB} MB` : ''})
                            </option>
                          ))}
                        </select>
                      </div>

                      {/* Selected GPU info */}
                      {(() => {
                        const gpu = clientGpuInfo.gpus.find(g => g.id === selectedClientGpuId) || clientGpuInfo.gpus[0];
                        if (!gpu) return null;
                        return (
                          <div style={{ display: 'grid', gap: 6 }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12 }}>
                              <span style={{ color: 'var(--text-muted)' }}>GPU</span>
                              <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-primary)' }}>{String(gpu.name || '')}</span>
                            </div>
                            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12 }}>
                              <span style={{ color: 'var(--text-muted)' }}>Type</span>
                              <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-primary)' }}>
                                {gpu.type === 'discrete' ? 'Discrete GPU' : gpu.type === 'integrated' ? 'Integrated GPU' : 'GPU'}
                              </span>
                            </div>
                            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12 }}>
                              <span style={{ color: 'var(--text-muted)' }}>Backend</span>
                              <span style={{ fontFamily: 'var(--font-mono)', color: gpu.backend === 'webgpu' ? 'var(--success)' : 'var(--text-muted)' }}>
                                {gpu.backend === 'webgpu' ? 'WebGPU' : 'WebGL only'}
                              </span>
                            </div>
                            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12 }}>
                              <span style={{ color: 'var(--text-muted)' }}>FP16 Shaders</span>
                              <span style={{ fontFamily: 'var(--font-mono)', color: gpu.hasFP16 ? 'var(--success)' : 'var(--text-muted)' }}>
                                {gpu.hasFP16 ? 'Supported (2-3x ML speedup)' : 'Not available'}
                              </span>
                            </div>
                            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12 }}>
                              <span style={{ color: 'var(--text-muted)' }}>Whisper Ready</span>
                              <span style={{ fontFamily: 'var(--font-mono)', color: gpu.whisperCapable ? 'var(--success)' : 'var(--warning)' }}>
                                {gpu.whisperCapable ? 'Yes' : gpu.backend === 'webgl-only' ? 'No (requires WebGPU)' : 'No (insufficient buffer size)'}
                              </span>
                            </div>
                          </div>
                        );
                      })()}

                      {/* WebCodecs capabilities */}
                      <div style={{ borderTop: '1px solid var(--border)', paddingTop: 8, marginTop: 4 }}>
                        <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 6 }}>WebCodecs (Video Encoding)</div>
                        <div style={{ display: 'grid', gap: 4 }}>
                          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12 }}>
                            <span style={{ color: 'var(--text-muted)' }}>H.264 Hardware Encode</span>
                            <span style={{ fontFamily: 'var(--font-mono)', color: clientGpuInfo.webcodecs.h264HardwareEncode ? 'var(--success)' : 'var(--text-muted)' }}>
                              {clientGpuInfo.webcodecs.h264HardwareEncode ? 'Hardware Accelerated' : 'Software only'}
                            </span>
                          </div>
                          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12 }}>
                            <span style={{ color: 'var(--text-muted)' }}>H.265/HEVC Hardware Encode</span>
                            <span style={{ fontFamily: 'var(--font-mono)', color: clientGpuInfo.webcodecs.hevcHardwareEncode ? 'var(--success)' : 'var(--text-muted)' }}>
                              {clientGpuInfo.webcodecs.hevcHardwareEncode ? 'Supported' : 'Not available'}
                            </span>
                          </div>
                        </div>
                      </div>

                      {/* Task toggles */}
                      <div style={{ borderTop: '1px solid var(--border)', paddingTop: 8, marginTop: 4 }}>
                        <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 8 }}>Client-Side Tasks</div>

                        {/* Whisper toggle */}
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                          <div>
                            <span style={{ fontSize: 12, color: 'var(--text-primary)' }}>Transcription (Whisper WebGPU)</span>
                            <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>
                              Runs Whisper speech-to-text in your browser via Transformers.js
                            </div>
                          </div>
                          <button
                            onClick={() => {
                              const val = !clientWhisperEnabled;
                              setClientWhisperEnabled(val);
                              handleSaveClientGpu({ whisperEnabled: val });
                            }}
                            style={{
                              width: 36, height: 20, borderRadius: 10, border: 'none', position: 'relative',
                              background: clientWhisperEnabled ? 'var(--accent-cyan)' : 'var(--bg-elevated)',
                              cursor: 'pointer', flexShrink: 0, marginLeft: 12, transition: 'background 0.2s',
                            }}
                          >
                            <div style={{
                              position: 'absolute', top: 2, left: clientWhisperEnabled ? 18 : 2,
                              width: 16, height: 16, borderRadius: '50%', background: 'var(--nav-active-icon-text)',
                              transition: 'left 0.2s', boxShadow: '0 1px 2px rgba(0,0,0,0.2)',
                            }} />
                          </button>
                        </div>

                        {/* Encoding toggle */}
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                          <div>
                            <span style={{ fontSize: 12, color: 'var(--text-primary)' }}>Video Encoding (WebCodecs)</span>
                            <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>
                              Uses H.264 hardware encoding in the browser for clip export (NVENC/VideoToolbox/VAAPI)
                            </div>
                          </div>
                          <button
                            onClick={() => {
                              const val = !clientEncodingEnabled;
                              setClientEncodingEnabled(val);
                              handleSaveClientGpu({ encodingEnabled: val });
                            }}
                            style={{
                              width: 36, height: 20, borderRadius: 10, border: 'none', position: 'relative',
                              background: clientEncodingEnabled ? 'var(--accent-cyan)' : 'var(--bg-elevated)',
                              cursor: 'pointer', flexShrink: 0, marginLeft: 12, transition: 'background 0.2s',
                            }}
                          >
                            <div style={{
                              position: 'absolute', top: 2, left: clientEncodingEnabled ? 18 : 2,
                              width: 16, height: 16, borderRadius: '50%', background: 'var(--nav-active-icon-text)',
                              transition: 'left 0.2s', boxShadow: '0 1px 2px rgba(0,0,0,0.2)',
                            }} />
                          </button>
                        </div>
                      </div>

                      {/* Rescan button */}
                      <button
                        onClick={async () => {
                          setClientGpuScanning(true);
                          try {
                            const { scanClientGPU } = await import('../utils/clientGpu.js');
                            const info = await scanClientGPU();
                            setClientGpuInfo(info);
                            showToast(`Found ${info.gpus.length} GPU(s): ${info.gpus.map(g => g.name).join(', ')}`, 'success');
                          } catch (e) { showToast(`GPU scan failed: ${e?.message || 'unknown error'}`, 'error'); }
                          setClientGpuScanning(false);
                        }}
                        disabled={clientGpuScanning}
                        style={{
                          marginTop: 4, padding: '4px 12px', fontSize: 10,
                          fontFamily: 'var(--font-mono)',
                          background: 'var(--bg-elevated)', color: 'var(--text-secondary)',
                          border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)',
                          cursor: 'pointer',
                        }}
                      >
                        {clientGpuScanning ? 'Scanning...' : 'Rescan GPUs'}
                      </button>
                    </div>
                  ) : (
                    /* WebGPU available but no GPU adapter returned */
                    <div>
                      <div style={{ fontSize: 12, color: 'var(--warning)', marginBottom: 8, fontWeight: 600 }}>
                        No WebGPU-compatible GPU adapter detected
                      </div>
                      {clientGpuInfo?.scanErrors?.length > 0 && (
                        <div style={{
                          marginBottom: 10, padding: '8px 10px',
                          background: 'rgba(255, 100, 100, 0.08)',
                          border: '1px solid rgba(255, 100, 100, 0.2)',
                          borderRadius: 'var(--radius-sm)',
                        }}>
                          <div style={{ fontSize: 10, color: 'var(--text-muted)', marginBottom: 4, fontWeight: 600 }}>Scan errors:</div>
                          {clientGpuInfo.scanErrors.map((err, i) => (
                            <div key={i} style={{ fontSize: 10, fontFamily: 'var(--font-mono)', color: 'var(--warning)', lineHeight: 1.5 }}>
                              {err}
                            </div>
                          ))}
                        </div>
                      )}
                      <div style={{ fontSize: 11, color: 'var(--text-muted)', lineHeight: 1.7 }}>
                        WebGPU is supported by your browser but no GPU adapter was returned. This usually means:<br/><br/>
                        <strong style={{ color: 'var(--text-secondary)' }}>Driver Issues</strong><br/>
                        {'• '}Update your GPU drivers to the latest version (NVIDIA GeForce Experience, AMD Adrenalin, or Intel Arc Control)<br/>
                        {'• '}NVIDIA: requires driver version 470+ for Vulkan/WebGPU support<br/>
                        {'• '}AMD: requires Mesa 21.0+ on Linux or latest Adrenalin on Windows<br/><br/>
                        <strong style={{ color: 'var(--text-secondary)' }}>Hardware Acceleration Disabled</strong><br/>
                        {'• '}Check Chrome Settings {'>'} System {'>'} ensure "Use hardware acceleration when available" is ON<br/>
                        {'• '}Visit <code style={{ fontSize: 10, background: 'var(--bg-elevated)', padding: '2px 6px', borderRadius: 3, color: 'var(--accent-cyan)' }}>chrome://gpu</code> and
                        verify "WebGPU" shows "Hardware accelerated" (not "Disabled" or "Software only")<br/><br/>
                        <strong style={{ color: 'var(--text-secondary)' }}>Environment Issues</strong><br/>
                        {'• '}Remote desktop sessions (RDP, VNC, Parsec) may not expose the GPU to the browser<br/>
                        {'• '}Virtual machines need GPU passthrough configured<br/>
                        {'• '}Some enterprise group policies disable GPU access in the browser<br/>
                        {'• '}On Linux, ensure Vulkan is working: run <code style={{ fontSize: 10, background: 'var(--bg-elevated)', padding: '2px 6px', borderRadius: 3, color: 'var(--text-secondary)' }}>vulkaninfo</code> in a terminal<br/><br/>
                        <strong style={{ color: 'var(--text-secondary)' }}>Debugging</strong><br/>
                        {'• '}Open browser DevTools (F12) {'>'} Console tab to see detailed GPU detection logs<br/>
                        {'• '}Try running in the Console: <code style={{ fontSize: 10, background: 'var(--bg-elevated)', padding: '2px 6px', borderRadius: 3, color: 'var(--text-secondary)' }}>await navigator.gpu.requestAdapter()</code> to test WebGPU directly
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* ── FFmpeg Threads ── */}
            <div style={{ marginBottom: 32 }}>
              <h3 style={{ fontSize: 14, marginBottom: 4, color: 'var(--text-secondary)' }}>FFmpeg Threads</h3>
              <p style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 16, lineHeight: 1.5 }}>
                Controls how many CPU threads FFmpeg uses during video export.
                Lower values use less memory (safer for containers), higher values export faster.
              </p>

              <div style={{
                padding: '12px 16px', background: 'var(--bg-panel)',
                border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)',
              }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                  <span style={{ fontSize: 13, color: 'var(--text-primary)' }}>Thread Count</span>
                  <span style={{ fontSize: 12, fontFamily: 'var(--font-mono)', color: 'var(--accent-cyan)' }}>
                    {ffmpegThreads === 0 ? 'Auto (all cores)' : ffmpegThreads}
                  </span>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <span style={{ fontSize: 10, color: 'var(--text-muted)', flexShrink: 0 }}>0</span>
                  <input
                    type="range"
                    min="0"
                    max="16"
                    step="1"
                    value={ffmpegThreads}
                    onChange={(e) => setFfmpegThreads(parseInt(e.target.value))}
                    style={{ flex: 1, accentColor: 'var(--accent-cyan)' }}
                  />
                  <span style={{ fontSize: 10, color: 'var(--text-muted)', flexShrink: 0 }}>16</span>
                </div>
                <span style={{ fontSize: 10, color: 'var(--text-muted)', display: 'block', marginTop: 4 }}>
                  0 = auto (uses all cores — may cause out-of-memory in containers). Recommended: 2–4 for Docker.
                </span>
                {ffmpegThreads !== ffmpegThreadsSaved && (
                  <button
                    onClick={() => handleSaveFfmpegThreads(ffmpegThreads)}
                    disabled={ffmpegThreadsSaving}
                    style={{
                      marginTop: 8, padding: '4px 14px', background: 'var(--accent-cyan)',
                      color: 'var(--bg-base)', border: 'none', borderRadius: 'var(--radius-sm)',
                      fontSize: 11, fontWeight: 600, cursor: 'pointer',
                    }}
                  >
                    {ffmpegThreadsSaving ? 'Saving...' : 'Save'}
                  </button>
                )}
              </div>
            </div>

            <h3 style={{ fontSize: 14, marginBottom: 16, color: 'var(--text-secondary)' }}>Model Override</h3>
            <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 16 }}>
              Browse all OpenRouter models and select custom overrides.
            </p>

            <div style={{ marginBottom: 24 }}>
              <h4 style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 8 }}>Vision Model</h4>
              <ModelBrowser type="vision" onSelect={(id) => { handleSaveModel('vision', id); }} />
            </div>

            <div style={{ marginBottom: 24 }}>
              <h4 style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 8 }}>Text Model</h4>
              <ModelBrowser type="text" onSelect={(id) => { handleSaveModel('text', id); }} />
            </div>

            <h3 style={{ fontSize: 14, marginBottom: 16, color: 'var(--text-secondary)' }}>Analysis Settings</h3>

            <div style={{ display: 'grid', gap: 12 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '8px 12px', background: 'var(--bg-panel)', border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)' }}>
                <span style={{ fontSize: 13 }}>Whisper Model</span>
                <span style={{ fontSize: 12, fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)' }}>{String(currentModels.transcript_model || 'base')}</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '8px 12px', background: 'var(--bg-panel)', border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)' }}>
                <span style={{ fontSize: 13 }}>Beam Size</span>
                <span style={{ fontSize: 12, fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)' }}>
                  {transSaved.beam_size}{transSaved.beam_size === 1 ? ' (fast)' : transSaved.beam_size >= 5 ? ' (accurate)' : ''}
                </span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '8px 12px', background: 'var(--bg-panel)', border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)' }}>
                <span style={{ fontSize: 13 }}>VAD Filter (Skip Silence)</span>
                <span style={{ fontSize: 12, fontFamily: 'var(--font-mono)', color: transSaved.vad_filter ? 'var(--success)' : 'var(--text-muted)' }}>
                  {transSaved.vad_filter ? 'On' : 'Off'}
                </span>
              </div>

              {/* Frame Sample Rate — editable slider */}
              <div style={{ padding: '10px 12px', background: 'var(--bg-panel)', border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                  <span style={{ fontSize: 13 }}>Frame Sample Rate</span>
                  <span style={{ fontSize: 12, fontFamily: 'var(--font-mono)', color: 'var(--accent-cyan)' }}>
                    Every {transSettings.frame_sample_rate}s
                  </span>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <span style={{ fontSize: 10, color: 'var(--text-muted)', flexShrink: 0 }}>5s</span>
                  <input
                    type="range"
                    min="5"
                    max="30"
                    step="5"
                    value={transSettings.frame_sample_rate}
                    onChange={(e) => setTransSettings((p) => ({ ...p, frame_sample_rate: parseInt(e.target.value) }))}
                    style={{ flex: 1, accentColor: 'var(--accent-cyan)' }}
                  />
                  <span style={{ fontSize: 10, color: 'var(--text-muted)', flexShrink: 0 }}>30s</span>
                </div>
                <span style={{ fontSize: 10, color: 'var(--text-muted)', display: 'block', marginTop: 4 }}>
                  Lower = more visual detail but slower. Higher = faster but less detail for clip detection.
                </span>
                {transSettings.frame_sample_rate !== transSaved.frame_sample_rate && (
                  <button
                    onClick={handleSaveTransSettings}
                    disabled={transSaving}
                    style={{
                      marginTop: 8, padding: '4px 14px', background: 'var(--accent-cyan)',
                      color: 'var(--bg-base)', border: 'none', borderRadius: 'var(--radius-sm)',
                      fontSize: 11, fontWeight: 600, cursor: 'pointer',
                    }}
                  >
                    {transSaving ? 'Saving...' : 'Save'}
                  </button>
                )}
              </div>

              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '8px 12px', background: 'var(--bg-panel)', border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)' }}>
                <span style={{ fontSize: 13 }}>Max Clip Candidates</span>
                <span style={{ fontSize: 12, fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)' }}>12</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '8px 12px', background: 'var(--bg-panel)', border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)' }}>
                <span style={{ fontSize: 13 }}>Fallback Chain</span>
                <span style={{ fontSize: 12, fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)' }}>{(statuses._active?.fallback_chain || ['openrouter', 'gemini', 'groq']).join(' \u2192 ')}</span>
              </div>
            </div>

          </div>
        </div>
      )}

      {/* ═══════ Tab 5: Usage & Costs ═══════ */}
      {settingsTab === 5 && <CostTracker />}

      {/* ═══════ Tab 6: API Access ═══════ */}
      {settingsTab === 6 && (
        <div style={{ maxWidth: isMobile ? '100%' : 640 }}>

          {/* API Key Section */}
          <div style={{ marginBottom: 24 }}>
            <h3 style={{ fontSize: 15, marginBottom: 12 }}>API Key</h3>
            <p style={{ fontSize: 12, color: 'var(--text-secondary)', marginBottom: 12 }}>
              Use this key to authenticate requests to the REST API (<code style={{ fontSize: 11, background: 'var(--bg-elevated)', padding: '1px 4px', borderRadius: 3 }}>/api/v1/*</code>).
            </p>
            <div style={{
              display: 'flex', alignItems: 'center', gap: 8,
              padding: '10px 12px', background: 'var(--bg-panel)',
              border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)',
            }}>
              {apiKeyLoading ? (
                <span style={{ fontSize: 12, color: 'var(--text-secondary)', fontFamily: 'var(--font-mono)' }}>Loading...</span>
              ) : (
                <code style={{
                  flex: 1, fontSize: 12, fontFamily: 'var(--font-mono)',
                  color: 'var(--accent-cyan)', wordBreak: 'break-all',
                  userSelect: apiKeyRevealed ? 'all' : 'none',
                }}>
                  {apiKeyRevealed ? apiKey : apiKeyMasked}
                </code>
              )}
              <button
                onClick={() => setApiKeyRevealed(!apiKeyRevealed)}
                style={{
                  padding: '4px 10px', fontSize: 11, background: 'var(--bg-elevated)',
                  border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)',
                  color: 'var(--text-secondary)', cursor: 'pointer', whiteSpace: 'nowrap',
                }}
              >
                {apiKeyRevealed ? 'Hide' : 'Reveal'}
              </button>
              <button
                onClick={() => { navigator.clipboard.writeText(apiKey); showToast('API key copied', 'success'); }}
                style={{
                  padding: '4px 10px', fontSize: 11, background: 'var(--bg-elevated)',
                  border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)',
                  color: 'var(--text-secondary)', cursor: 'pointer', whiteSpace: 'nowrap',
                }}
              >
                Copy
              </button>
            </div>
            <div style={{ display: 'flex', gap: 8, marginTop: 8 }}>
              <button
                onClick={handleRegenerateApiKey}
                disabled={apiKeyRegenerating}
                style={{
                  padding: '6px 14px', fontSize: 12, fontWeight: 500,
                  background: 'transparent', border: '1px solid var(--accent-amber)',
                  borderRadius: 'var(--radius-sm)', color: 'var(--accent-amber)',
                  cursor: apiKeyRegenerating ? 'not-allowed' : 'pointer', opacity: apiKeyRegenerating ? 0.5 : 1,
                }}
              >
                {apiKeyRegenerating ? 'Regenerating...' : 'Regenerate Key'}
              </button>
            </div>
          </div>

          {/* Endpoints Section */}
          <div style={{ marginBottom: 24 }}>
            <h3 style={{ fontSize: 15, marginBottom: 12 }}>Endpoints</h3>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
              {[
                { label: 'REST API', value: `${window.location.origin}/api/v1` },
                { label: 'MCP Server', value: `${window.location.origin}/mcp` },
                { label: 'Health Check', value: `${window.location.origin}/api/v1/health` },
                { label: 'Agent Skill', value: `${window.location.origin}/api/v1/skill` },
              ].map(({ label, value }) => (
                <div key={label} style={{
                  display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                  padding: '8px 12px', background: 'var(--bg-panel)',
                  border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)',
                }}>
                  <span style={{ fontSize: 13 }}>{label}</span>
                  <code style={{ fontSize: 11, fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)', wordBreak: 'break-all' }}>{value}</code>
                </div>
              ))}
            </div>
          </div>

          {/* Connection Test */}
          <div style={{ marginBottom: 24 }}>
            <h3 style={{ fontSize: 15, marginBottom: 12 }}>Connection Test</h3>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <button
                onClick={handleTestApiConnection}
                disabled={apiTesting}
                style={{
                  padding: '6px 14px', fontSize: 12, fontWeight: 500,
                  background: 'var(--accent-cyan)', border: 'none',
                  borderRadius: 'var(--radius-sm)', color: 'var(--bg-base)',
                  cursor: apiTesting ? 'not-allowed' : 'pointer', opacity: apiTesting ? 0.5 : 1,
                }}
              >
                {apiTesting ? 'Testing...' : 'Test Connection'}
              </button>
              {apiTestResult && (
                <span style={{
                  fontSize: 12,
                  color: apiTestResult.status === 'success' ? 'var(--success)' : 'var(--error)',
                }}>
                  {apiTestResult.message}
                </span>
              )}
            </div>
          </div>

          {/* Quick Start Snippets */}
          <div style={{ marginBottom: 24 }}>
            <h3 style={{ fontSize: 15, marginBottom: 12 }}>Quick Start</h3>

            {/* cURL Example */}
            <div style={{ marginBottom: 12 }}>
              <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginBottom: 4 }}>cURL</div>
              <div style={{
                position: 'relative', background: 'var(--bg-panel)',
                border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)',
                padding: '10px 12px', paddingRight: 60,
              }}>
                <pre style={{
                  fontSize: 11, fontFamily: 'var(--font-mono)', color: 'var(--text-primary)',
                  whiteSpace: 'pre-wrap', wordBreak: 'break-all', margin: 0,
                }}>{`curl ${window.location.origin}/api/v1/health \\
  -H "Authorization: Bearer ${apiKeyRevealed ? apiKey : '<your-key>'}"`.trim()}</pre>
                <button
                  onClick={() => {
                    navigator.clipboard.writeText(`curl ${window.location.origin}/api/v1/health -H "Authorization: Bearer ${apiKey}"`);
                    showToast('Copied', 'success');
                  }}
                  style={{
                    position: 'absolute', top: 6, right: 6,
                    padding: '2px 8px', fontSize: 10, background: 'var(--bg-elevated)',
                    border: '1px solid var(--border)', borderRadius: 3,
                    color: 'var(--text-secondary)', cursor: 'pointer',
                  }}
                >
                  Copy
                </button>
              </div>
            </div>

            {/* Python Example */}
            <div style={{ marginBottom: 12 }}>
              <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginBottom: 4 }}>Python</div>
              <div style={{
                position: 'relative', background: 'var(--bg-panel)',
                border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)',
                padding: '10px 12px', paddingRight: 60,
              }}>
                <pre style={{
                  fontSize: 11, fontFamily: 'var(--font-mono)', color: 'var(--text-primary)',
                  whiteSpace: 'pre-wrap', wordBreak: 'break-all', margin: 0,
                }}>{`import httpx

client = httpx.Client(
    base_url="${window.location.origin}/api/v1",
    headers={"Authorization": "Bearer ${apiKeyRevealed ? apiKey : '<your-key>'}"}
)

# Check health
resp = client.get("/health")
print(resp.json())`.trim()}</pre>
                <button
                  onClick={() => {
                    navigator.clipboard.writeText(`import httpx\n\nclient = httpx.Client(\n    base_url="${window.location.origin}/api/v1",\n    headers={"Authorization": "Bearer ${apiKey}"}\n)\n\nresp = client.get("/health")\nprint(resp.json())`);
                    showToast('Copied', 'success');
                  }}
                  style={{
                    position: 'absolute', top: 6, right: 6,
                    padding: '2px 8px', fontSize: 10, background: 'var(--bg-elevated)',
                    border: '1px solid var(--border)', borderRadius: 3,
                    color: 'var(--text-secondary)', cursor: 'pointer',
                  }}
                >
                  Copy
                </button>
              </div>
            </div>

            {/* MCP Config Example */}
            <div>
              <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginBottom: 4 }}>MCP Client Config</div>
              <div style={{
                position: 'relative', background: 'var(--bg-panel)',
                border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)',
                padding: '10px 12px', paddingRight: 60,
              }}>
                <pre style={{
                  fontSize: 11, fontFamily: 'var(--font-mono)', color: 'var(--text-primary)',
                  whiteSpace: 'pre-wrap', wordBreak: 'break-all', margin: 0,
                }}>{`{
  "mcpServers": {
    "clipai": {
      "type": "streamable-http",
      "url": "${window.location.origin}/mcp"
    }
  }
}`.trim()}</pre>
                <button
                  onClick={() => {
                    navigator.clipboard.writeText(JSON.stringify({ mcpServers: { clipai: { type: 'streamable-http', url: `${window.location.origin}/mcp` } } }, null, 2));
                    showToast('Copied', 'success');
                  }}
                  style={{
                    position: 'absolute', top: 6, right: 6,
                    padding: '2px 8px', fontSize: 10, background: 'var(--bg-elevated)',
                    border: '1px solid var(--border)', borderRadius: 3,
                    color: 'var(--text-secondary)', cursor: 'pointer',
                  }}
                >
                  Copy
                </button>
              </div>
            </div>
          </div>

        </div>
      )}

      {/* ═══════ Tab 7: About ═══════ */}
      {settingsTab === 7 && (
        <div style={{ maxWidth: isMobile ? '100%' : 640 }}>

          {/* Developer credit */}
          <div style={{
            padding: 24, borderRadius: 'var(--radius-md)', marginBottom: 28,
            background: 'var(--bg-panel)', border: '1px solid var(--border)',
            textAlign: 'center',
          }}>
            <div style={{ fontSize: 15, fontWeight: 600, color: 'var(--text-primary)', marginBottom: 6 }}>
              Developed by Jalon Young
            </div>
            <p style={{ fontSize: 12, color: 'var(--text-secondary)', margin: '0 0 14px', lineHeight: 1.6 }}>
              Full-stack engineer &amp; creator of ClipAI.
            </p>
            <a
              href="https://jalonyoung.com"
              target="_blank"
              rel="noopener noreferrer"
              style={{
                display: 'inline-block', padding: '8px 20px', fontSize: 12, fontWeight: 600,
                fontFamily: 'var(--font-mono)', color: '#fff', background: 'var(--accent-cyan)',
                borderRadius: 'var(--radius-sm)', textDecoration: 'none',
              }}
            >
              jalonyoung.com
            </a>
          </div>

          {/* Site Customisation */}
          <h3 style={{ fontSize: 14, fontWeight: 600, marginBottom: 16, color: 'var(--text-primary)' }}>
            Site Customisation
          </h3>

          {/* Title */}
          <div style={{ marginBottom: 20 }}>
            <label style={{ display: 'block', fontSize: 12, fontWeight: 600, marginBottom: 6, color: 'var(--text-primary)' }}>
              Site Title
            </label>
            <input
              type="text"
              value={siteTitle}
              onChange={(e) => setSiteTitle(e.target.value)}
              placeholder="ClipAI — Video Intelligence"
              style={{
                width: '100%', padding: '8px 12px', fontSize: 12, fontFamily: 'var(--font-mono)',
                background: 'var(--bg-panel)', border: '1px solid var(--border)',
                borderRadius: 'var(--radius-sm)', color: 'var(--text-primary)', boxSizing: 'border-box',
              }}
            />
          </div>

          {/* Favicon */}
          <div style={{ marginBottom: 20 }}>
            <label style={{ display: 'block', fontSize: 12, fontWeight: 600, marginBottom: 6, color: 'var(--text-primary)' }}>
              Favicon
            </label>
            <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
              {siteFavicon && (
                <img
                  src={`/api/site-uploads/${siteFavicon}`}
                  alt="favicon"
                  style={{ width: 28, height: 28, objectFit: 'contain', borderRadius: 4, border: '1px solid var(--border)' }}
                />
              )}
              <input
                ref={faviconInputRef}
                type="file"
                accept=".ico,.png,.svg,.jpg,.jpeg,.gif,.webp"
                style={{ display: 'none' }}
                onChange={(e) => {
                  const file = e.target.files?.[0];
                  if (!file) return;
                  setSiteSaving(true);
                  const fd = new FormData();
                  fd.append('favicon', file);
                  fetch('/api/site-config', { method: 'POST', body: fd })
                    .then((r) => r.json())
                    .then((res) => {
                      if (res.favicon) {
                        setSiteFavicon(res.favicon);
                        let link = document.querySelector("link[rel~='icon']");
                        if (!link) { link = document.createElement('link'); link.rel = 'icon'; document.head.appendChild(link); }
                        link.href = `/api/site-uploads/${res.favicon}`;
                      }
                      showToast('Favicon updated', 'success');
                    })
                    .catch(() => showToast('Upload failed', 'error'))
                    .finally(() => setSiteSaving(false));
                  e.target.value = '';
                }}
              />
              <button
                onClick={() => faviconInputRef.current?.click()}
                disabled={siteSaving}
                style={{
                  padding: '6px 14px', fontSize: 11, fontWeight: 600, fontFamily: 'var(--font-mono)',
                  background: 'var(--bg-elevated)', border: '1px solid var(--border)',
                  borderRadius: 'var(--radius-sm)', color: 'var(--text-primary)', cursor: 'pointer',
                }}
              >
                {siteFavicon ? 'Replace' : 'Upload'}
              </button>
              {siteFavicon && (
                <button
                  onClick={() => {
                    setSiteSaving(true);
                    const fd = new FormData();
                    fd.append('remove_favicon', 'true');
                    fetch('/api/site-config', { method: 'POST', body: fd })
                      .then((r) => r.json())
                      .then(() => {
                        setSiteFavicon(null);
                        const link = document.querySelector("link[rel~='icon']");
                        if (link) link.href = "data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'><text y='28' font-size='28'>C</text></svg>";
                        showToast('Favicon removed', 'success');
                      })
                      .catch(() => showToast('Failed', 'error'))
                      .finally(() => setSiteSaving(false));
                  }}
                  disabled={siteSaving}
                  style={{
                    padding: '6px 14px', fontSize: 11, fontFamily: 'var(--font-mono)',
                    background: 'none', border: '1px solid var(--danger, #ff453a)',
                    borderRadius: 'var(--radius-sm)', color: 'var(--danger, #ff453a)', cursor: 'pointer',
                  }}
                >
                  Remove
                </button>
              )}
            </div>
          </div>

          {/* Logo */}
          <div style={{ marginBottom: 28 }}>
            <label style={{ display: 'block', fontSize: 12, fontWeight: 600, marginBottom: 6, color: 'var(--text-primary)' }}>
              Logo
            </label>
            <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
              {siteLogo && (
                <img
                  src={`/api/site-uploads/${siteLogo}`}
                  alt="logo"
                  style={{ height: 36, maxWidth: 120, objectFit: 'contain', borderRadius: 4, border: '1px solid var(--border)' }}
                />
              )}
              <input
                ref={logoInputRef}
                type="file"
                accept=".png,.svg,.jpg,.jpeg,.gif,.webp"
                style={{ display: 'none' }}
                onChange={(e) => {
                  const file = e.target.files?.[0];
                  if (!file) return;
                  setSiteSaving(true);
                  const fd = new FormData();
                  fd.append('logo', file);
                  fetch('/api/site-config', { method: 'POST', body: fd })
                    .then((r) => r.json())
                    .then((res) => {
                      if (res.logo) setSiteLogo(res.logo);
                      showToast('Logo updated', 'success');
                    })
                    .catch(() => showToast('Upload failed', 'error'))
                    .finally(() => setSiteSaving(false));
                  e.target.value = '';
                }}
              />
              <button
                onClick={() => logoInputRef.current?.click()}
                disabled={siteSaving}
                style={{
                  padding: '6px 14px', fontSize: 11, fontWeight: 600, fontFamily: 'var(--font-mono)',
                  background: 'var(--bg-elevated)', border: '1px solid var(--border)',
                  borderRadius: 'var(--radius-sm)', color: 'var(--text-primary)', cursor: 'pointer',
                }}
              >
                {siteLogo ? 'Replace' : 'Upload'}
              </button>
              {siteLogo && (
                <button
                  onClick={() => {
                    setSiteSaving(true);
                    const fd = new FormData();
                    fd.append('remove_logo', 'true');
                    fetch('/api/site-config', { method: 'POST', body: fd })
                      .then((r) => r.json())
                      .then(() => { setSiteLogo(null); showToast('Logo removed', 'success'); })
                      .catch(() => showToast('Failed', 'error'))
                      .finally(() => setSiteSaving(false));
                  }}
                  disabled={siteSaving}
                  style={{
                    padding: '6px 14px', fontSize: 11, fontFamily: 'var(--font-mono)',
                    background: 'none', border: '1px solid var(--danger, #ff453a)',
                    borderRadius: 'var(--radius-sm)', color: 'var(--danger, #ff453a)', cursor: 'pointer',
                  }}
                >
                  Remove
                </button>
              )}
            </div>
          </div>

          {/* Save title button */}
          <button
            onClick={() => {
              setSiteSaving(true);
              const fd = new FormData();
              fd.append('title', siteTitle);
              fetch('/api/site-config', { method: 'POST', body: fd })
                .then((r) => r.json())
                .then((res) => {
                  if (res.title !== undefined) document.title = res.title || 'ClipAI — Video Intelligence';
                  showToast('Site settings saved', 'success');
                })
                .catch(() => showToast('Save failed', 'error'))
                .finally(() => setSiteSaving(false));
            }}
            disabled={siteSaving}
            style={{
              padding: '10px 28px', fontSize: 13, fontWeight: 600, fontFamily: 'var(--font-mono)',
              background: 'var(--accent-cyan)', color: '#fff', border: 'none',
              borderRadius: 'var(--radius-sm)', cursor: siteSaving ? 'wait' : 'pointer',
              opacity: siteSaving ? 0.6 : 1,
            }}
          >
            {siteSaving ? 'Saving...' : 'Save Title'}
          </button>
        </div>
      )}

      {/* ── Users tab (admin-only) ── */}
      {settingsTab === 8 && isAdmin && (
        <UserManagementPanel />
      )}
    </div>
  );
}
