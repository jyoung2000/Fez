import React, { useState, useEffect, useCallback } from 'react';

// ── Colors ──────────────────────────────────────────────────────────────
const STATUS_ICONS = { pass: '\u2705', warn: '\u26a0\ufe0f', fail: '\u274c', running: null };
const MODEL_COLORS = { vision: '#8b5cf6', text: '#3b82f6' };

function formatBytes(bytes) {
  if (!bytes) return '0 MB';
  const gb = bytes / (1024 * 1024 * 1024);
  if (gb >= 0.95) return `${gb.toFixed(1)} GB`;
  return `${(bytes / (1024 * 1024)).toFixed(0)} MB`;
}

// ── Spinner ─────────────────────────────────────────────────────────────
function Spinner() {
  return (
    <span style={{
      display: 'inline-block', width: 14, height: 14,
      border: '2px solid var(--border)', borderTopColor: '#3b82f6',
      borderRadius: '50%', animation: 'spin 0.8s linear infinite',
    }} />
  );
}

// ── VRAM Gauge ──────────────────────────────────────────────────────────
function VramGauge({ gpu, loadedModels, ollamaAvailable, torchGpu, onUnload, onReleaseGpu, onRestart }) {
  // Ollama offline
  if (ollamaAvailable === false) {
    return (
      <div style={cardStyle}>
        <div style={{ fontSize: 12, color: 'var(--text-muted)', textAlign: 'center', padding: '8px 0' }}>
          Ollama offline — reconnecting...
        </div>
      </div>
    );
  }

  // No GPU hardware at all
  if (gpu && !gpu.gpu_available && !gpu.cuda_available) {
    return (
      <div style={cardStyle}>
        <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>
          No NVIDIA GPU detected — Ollama running on CPU only
        </div>
        {loadedModels.length > 0 && (
          <div style={{ marginTop: 8 }}>
            {loadedModels.map((m, i) => (
              <div key={i} style={{ fontSize: 10, fontFamily: 'var(--font-mono)', color: 'var(--text-muted)', marginTop: 2 }}>
                <span style={{ display: 'inline-block', width: 8, height: 8, borderRadius: 2, marginRight: 6, background: '#f59e0b' }} />
                {m.name} — CPU — {formatBytes(m.size_bytes)}
              </div>
            ))}
            <button onClick={onUnload} style={smallBtnStyle}>Unload All Models</button>
          </div>
        )}
      </div>
    );
  }

  // Still loading
  if (!gpu) {
    return (
      <div style={cardStyle}>
        <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>Loading GPU info...</div>
      </div>
    );
  }

  // GPU exists — render the full gauge
  const totalBytes = gpu.vram_total_bytes || 1;
  const usedBytes = gpu.vram_used_bytes || 0;
  const usedPct = Math.min(100, (usedBytes / totalBytes) * 100);
  const barColor = usedPct > 85 ? '#ef4444' : usedPct > 60 ? '#f59e0b' : '#22c55e';
  const isPoisoned = gpu.gpu_poisoned;

  // Build model segments for VRAM bar
  const segments = loadedModels.map((m) => {
    const pct = totalBytes > 0 ? Math.min(100, (m.vram_bytes / totalBytes) * 100) : 0;
    const isVision = /moondream|llava|vision/i.test(m.name);
    return { name: m.name, pct, color: isVision ? MODEL_COLORS.vision : MODEL_COLORS.text, vram: m.vram_bytes };
  });

  // Torch reserved memory segment (orange — the hidden VRAM hog)
  const torchReserved = torchGpu?.reserved_bytes || 0;
  if (torchReserved > 50 * 1024 * 1024) { // Only show if > 50MB
    const torchPct = totalBytes > 0 ? Math.min(100, (torchReserved / totalBytes) * 100) : 0;
    segments.push({ name: 'Torch/Whisper', pct: torchPct, color: '#f59e0b', vram: torchReserved });
  }

  return (
    <div style={cardStyle}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
        <span style={{ fontSize: 11, fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)' }}>
          GPU: {gpu.gpu_name || 'Unknown'}
        </span>
        <span style={{ fontSize: 10, fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>
          <span style={{
            display: 'inline-block', width: 6, height: 6, borderRadius: '50%', marginRight: 4,
            background: isPoisoned ? '#f59e0b' : gpu.gpu_in_use ? '#22c55e' : 'var(--text-muted)',
          }} />
          Live (2s)
        </span>
      </div>

      {/* GPU Poisoned Warning */}
      {isPoisoned && (
        <div style={{
          padding: '8px 12px', marginBottom: 10,
          background: 'rgba(245, 158, 11, 0.08)',
          border: '1px solid rgba(245, 158, 11, 0.25)',
          borderRadius: 'var(--radius-sm)',
          fontSize: 11, color: '#f59e0b', lineHeight: 1.5,
        }}>
          GPU detected but Ollama is not using it. This usually means the GPU scheduler
          was poisoned by a prior CUDA OOM crash. Restart the Ollama container to fix.
          <button onClick={onRestart} style={{
            marginLeft: 8, padding: '2px 10px', fontSize: 10, fontFamily: 'var(--font-mono)',
            border: '1px solid #f59e0b', borderRadius: 'var(--radius-sm)',
            background: 'transparent', color: '#f59e0b', cursor: 'pointer',
          }}>
            Restart Ollama
          </button>
        </div>
      )}

      {/* VRAM Bar */}
      <div style={{
        height: 28, borderRadius: 6, background: '#1f2937',
        overflow: 'hidden', position: 'relative', marginBottom: 6,
      }}>
        {/* Colored segments per model */}
        <div style={{ display: 'flex', height: '100%' }}>
          {segments.map((seg, i) => (
            <div key={i} title={`${seg.name}: ${formatBytes(seg.vram)}`} style={{
              width: `${seg.pct}%`, height: '100%', background: seg.color,
              transition: 'width 0.5s ease-out', minWidth: seg.pct > 0 ? 4 : 0,
            }} />
          ))}
        </div>
        {/* Model name labels overlaid */}
        {segments.some(s => s.pct > 8) && (
          <div style={{
            position: 'absolute', top: 0, left: 0, right: 0, bottom: 0,
            display: 'flex', alignItems: 'center', paddingLeft: 10, gap: 8,
          }}>
            {segments.filter(s => s.pct > 8).map((s, i) => (
              <span key={i} style={{
                fontSize: 10, fontWeight: 500, color: '#fff',
                textShadow: '0 0 4px rgba(0,0,0,0.7)', whiteSpace: 'nowrap',
              }}>
                {s.name}
              </span>
            ))}
          </div>
        )}
      </div>

      {/* Usage label */}
      <div style={{
        display: 'flex', justifyContent: 'space-between',
        fontSize: 11, color: 'var(--text-muted)', marginBottom: 10,
      }}>
        <span>VRAM usage</span>
        <span style={{ fontWeight: 500, color: barColor }}>
          {formatBytes(usedBytes)} / {formatBytes(totalBytes)}
        </span>
      </div>

      {/* Model list */}
      {loadedModels.length > 0 && (
        <div style={{ borderTop: '1px solid var(--border)', paddingTop: 8 }}>
          {loadedModels.map((m, i) => {
            const isGpu = m.vram_bytes > 0;
            const gpuPct = m.size_bytes > 0 ? Math.round((m.vram_bytes / m.size_bytes) * 100) : 0;
            const isVision = /moondream|llava|vision/i.test(m.name);
            return (
              <div key={i} style={{
                display: 'flex', alignItems: 'center', gap: 8,
                fontSize: 11, padding: '3px 0',
              }}>
                <span style={{
                  width: 8, height: 8, borderRadius: 2, flexShrink: 0,
                  background: isVision ? MODEL_COLORS.vision : MODEL_COLORS.text,
                }} />
                <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-primary)' }}>
                  {m.name}
                </span>
                <span style={{ fontFamily: 'var(--font-mono)', color: isGpu ? '#22c55e' : '#f59e0b', fontSize: 10 }}>
                  {isGpu ? `GPU ${gpuPct}%` : 'CPU'}
                </span>
                {m.vram_bytes > 0 && (
                  <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-muted)', fontSize: 10 }}>
                    {formatBytes(m.vram_bytes)} VRAM
                  </span>
                )}
              </div>
            );
          })}
        </div>
      )}
      {/* Torch reserved memory warning */}
      {torchReserved > 100 * 1024 * 1024 && (
        <div style={{
          display: 'flex', alignItems: 'center', gap: 8,
          fontSize: 11, padding: '3px 0',
          borderTop: loadedModels.length > 0 ? 'none' : '1px solid var(--border)',
          paddingTop: loadedModels.length > 0 ? 0 : 8,
        }}>
          <span style={{ width: 8, height: 8, borderRadius: 2, flexShrink: 0, background: '#f59e0b' }} />
          <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-primary)' }}>Torch/Whisper</span>
          <span style={{ fontFamily: 'var(--font-mono)', color: '#f59e0b', fontSize: 10 }}>
            {formatBytes(torchReserved)} reserved
          </span>
        </div>
      )}
      {loadedModels.length === 0 && torchReserved <= 100 * 1024 * 1024 && (
        <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>No models loaded</div>
      )}

      <div style={{ display: 'flex', gap: 8 }}>
        <button onClick={onUnload} style={smallBtnStyle}>Unload Ollama Models</button>
        {torchReserved > 100 * 1024 * 1024 && (
          <button onClick={onReleaseGpu} style={{ ...smallBtnStyle, color: '#f59e0b', borderColor: '#f59e0b' }}>
            Release Torch VRAM
          </button>
        )}
      </div>
    </div>
  );
}

// ── Phase Result Row ────────────────────────────────────────────────────
function PhaseRow({ phase }) {
  const icon = phase.status === 'running' ? <Spinner /> : STATUS_ICONS[phase.status] || '';
  return (
    <div style={{
      display: 'flex', alignItems: 'flex-start', gap: 8, padding: '6px 0',
      borderBottom: '1px solid rgba(128,128,128,0.1)',
    }}>
      <span style={{ fontSize: 14, width: 20, textAlign: 'center', flexShrink: 0 }}>{icon}</span>
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ fontSize: 12, color: 'var(--text-primary)' }}>
          {phase.label}
          {phase.duration_ms != null && phase.status !== 'running' && (
            <span style={{ fontSize: 10, color: 'var(--text-muted)', marginLeft: 6 }}>
              {phase.duration_ms}ms
            </span>
          )}
        </div>
        {phase.message && phase.status !== 'running' && (
          <div style={{
            fontSize: 10, fontFamily: 'var(--font-mono)', marginTop: 2,
            color: phase.status === 'fail' ? '#ef4444' : phase.status === 'warn' ? '#f59e0b' : 'var(--text-muted)',
          }}>
            {phase.message}
          </div>
        )}
        {phase.gpu_status && phase.status !== 'running' && (
          <div style={{ fontSize: 10, fontFamily: 'var(--font-mono)', color: 'var(--text-muted)', marginTop: 1 }}>
            GPU: {phase.gpu_status}
          </div>
        )}
        {phase.sample_output && (
          <div style={{
            fontSize: 10, fontStyle: 'italic', color: 'var(--text-muted)',
            marginTop: 2, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
          }}>
            &quot;{phase.sample_output}&quot;
          </div>
        )}
      </div>
    </div>
  );
}

// ── Main Component ──────────────────────────────────────────────────────
export default function PipelineDiagnostics() {
  const [gpuStatus, setGpuStatus] = useState(null);
  const [loadedModels, setLoadedModels] = useState([]);
  const [torchGpu, setTorchGpu] = useState(null);
  const [ollamaAvailable, setOllamaAvailable] = useState(null);
  const [testRunning, setTestRunning] = useState(false);
  const [testPhases, setTestPhases] = useState([]);
  const [testOverall, setTestOverall] = useState(null);
  const [restartMsg, setRestartMsg] = useState(null);
  const [testIncludeWhisper, setTestIncludeWhisper] = useState(true);
  const [testTranslation, setTestTranslation] = useState(false);

  // Poll GPU status every 2s
  useEffect(() => {
    let active = true;
    const poll = async () => {
      try {
        const resp = await fetch('/api/diagnostics/gpu-status');
        if (!resp.ok) {
          if (active) setOllamaAvailable(false);
          return;
        }
        const data = await resp.json();
        if (!active) return;
        setGpuStatus(data.gpu);
        setLoadedModels(data.loaded_models || []);
        setTorchGpu(data.torch_gpu || null);
        setOllamaAvailable(data.ollama_available);
      } catch {
        if (active) setOllamaAvailable(false);
      }
    };
    poll();
    const id = setInterval(poll, 2000);
    return () => { active = false; clearInterval(id); };
  }, []);

  const handleUnload = useCallback(async () => {
    try { await fetch('/api/diagnostics/unload-models', { method: 'POST' }); } catch { /* */ }
  }, []);

  const handleReleaseGpu = useCallback(async () => {
    try { await fetch('/api/diagnostics/release-gpu', { method: 'POST' }); } catch { /* */ }
  }, []);

  const handleRestart = useCallback(async () => {
    setRestartMsg(null);
    try {
      const resp = await fetch('/api/diagnostics/restart-ollama', { method: 'POST' });
      const data = await resp.json();
      setRestartMsg(data.message);
      if (data.status === 'ok') {
        // Briefly show offline then let polling re-detect
        setOllamaAvailable(false);
      }
    } catch (e) {
      setRestartMsg(`Failed: ${e.message}. Run manually: docker restart clipai-ollama`);
    }
  }, []);

  const runTest = useCallback(async () => {
    setTestRunning(true);
    setTestPhases([]);
    setTestOverall(null);
    try {
      const resp = await fetch('/api/diagnostics/test-pipeline', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          include_whisper: testIncludeWhisper,
          test_translation: testTranslation,
        }),
      });
      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';
        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          try {
            const evt = JSON.parse(line.slice(6));
            if (evt.type === 'phase_start') {
              setTestPhases((prev) => [...prev, { phase: evt.data.phase, label: evt.data.label, status: 'running' }]);
            } else if (evt.type === 'phase_result') {
              setTestPhases((prev) => prev.map((p) => p.phase === evt.data.phase ? { ...p, ...evt.data } : p));
            } else if (evt.type === 'complete') {
              setTestOverall(evt.data);
            }
          } catch { /* skip malformed */ }
        }
      }
    } catch (e) {
      setTestOverall({ overall_status: 'fail', error: e.message });
    } finally {
      setTestRunning(false);
    }
  }, [testIncludeWhisper, testTranslation]);

  // ── 2026 SOTA Reframing - 1-click QA + bench runner ───────────────
  const [sotaRunning, setSotaRunning] = useState(false);
  const [sotaSkipBench, setSotaSkipBench] = useState(false);
  const [sotaPhases, setSotaPhases] = useState([]);
  const [sotaLogs, setSotaLogs] = useState([]);
  const [sotaResult, setSotaResult] = useState(null);
  const sotaLogRef = React.useRef(null);

  // ── SOTA clip-test (upload + run on user-supplied MP4) ───────────
  const [sotaClipFile, setSotaClipFile] = useState(null);
  const [sotaClipContentType, setSotaClipContentType] = useState('default');
  const [sotaClipUploadProgress, setSotaClipUploadProgress] = useState(null); // 0..100 or null
  const sotaClipFileInputRef = React.useRef(null);

  // Task A — cache state surfaced in SSE phase_result for "cache_check"
  // so the operator sees a stale-cache warning + a "Force re-extract"
  // button when the deployed cache predates the running container.
  const [cacheCheck, setCacheCheck] = useState(null);
  const [forceReextract, setForceReextract] = useState(false);

  // Task B — AutoFlip references badge state. Polled from
  // /api/diagnostics/refs-state on mount + after a build.
  // (Build action removed in Layer 1 PR; the badge is now read-only.)
  const [refsState, setRefsState] = useState({ expected: 0, real: 0, naive: 0 });

  const fetchRefsState = useCallback(async () => {
    try {
      const resp = await fetch('/api/diagnostics/refs-state');
      if (resp.ok) {
        const data = await resp.json();
        setRefsState({
          expected: data.expected || 0,
          real: data.real || 0,
          naive: data.naive || 0,
        });
      }
    } catch { /* badge is best-effort */ }
  }, []);

  React.useEffect(() => { fetchRefsState(); }, [fetchRefsState]);

  // Layer 1 critic engine replaces what the AutoFlip baseline was
  // supposed to provide. The "Build AutoFlip image" button + handler
  // were removed in the L1 PR — AutoFlip is deprecated upstream
  // (Google sunset, March 2023) and incompatible with our current
  // OpenCV. Real-content reframing quality is judged by the critic
  // engine layers (saliency / DOVER / VLM / engagement / brain) now.

  const SOTA_CONTENT_TYPES = [
    { value: 'default', label: 'Auto / default' },
    { value: 'multi_speaker_panel', label: 'Multi-speaker panel' },
    { value: 'talking_head', label: 'Talking head' },
    { value: 'vlog', label: 'Vlog' },
    { value: 'narrative', label: 'Narrative / cinema' },
    { value: 'music_video', label: 'Music video / concert' },
    { value: 'sports', label: 'Sports' },
    { value: 'gaming', label: 'Gaming / gameplay' },
    { value: 'anime', label: 'Anime / animation' },
    { value: 'tutorial', label: 'Tutorial / how-to' },
    { value: 'podcast', label: 'Podcast' },
    { value: 'interview', label: 'Interview' },
  ];

  React.useEffect(() => {
    if (sotaLogRef.current) {
      sotaLogRef.current.scrollTop = sotaLogRef.current.scrollHeight;
    }
  }, [sotaLogs]);

  const runSotaBench = useCallback(async () => {
    setSotaRunning(true);
    setSotaPhases([]);
    setSotaLogs([]);
    setSotaResult(null);

    // Step 0: probe the bench-status endpoint so the operator sees
    // exactly what the server thinks of its environment before any
    // streaming starts. This is fast (<200 ms) and never tears the
    // chunked encoding because it is a regular JSON response.
    try {
      const probe = await fetch('/api/diagnostics/sota-bench-status');
      if (probe.ok) {
        const data = await probe.json();
        setSotaLogs((prev) => [...prev, {
          kind: 'meta',
          text: `[probe] qa_runner_exists=${data.qa_runner_exists} ` +
                `subprocess=${data.subprocess_plumbing?.ok} ` +
                `module_import=${data.module_import?.ok} ` +
                `suites=${(data.suite_files || []).length}`,
        }]);
        if (!data.ok) {
          setSotaResult({
            ok: false,
            stage: 'preflight',
            message:
              !data.qa_runner_exists
                ? `QA harness not in image at ${data.qa_runner_path}. ` +
                  `Rebuild the container with: ` +
                  `docker compose down && docker compose build --no-cache && docker compose up -d`
                : !data.subprocess_plumbing?.ok
                  ? `subprocess plumbing broken: ${data.subprocess_plumbing?.stderr || 'unknown'}`
                : !data.module_import?.ok
                  ? `harness import failed: ${data.module_import?.error}`
                : `unknown preflight failure (suites=${data.suite_files?.length})`,
          });
          setSotaRunning(false);
          return;
        }
      }
    } catch (e) {
      setSotaLogs((prev) => [...prev, {
        kind: 'meta', text: `[probe] failed: ${e.message} (continuing anyway)`,
      }]);
    }

    // Step 1: QA-only mode uses the synchronous JSON endpoint - no
    // SSE, no streaming surface. This is BY FAR the most reliable
    // path across container networks / reverse proxies.
    if (sotaSkipBench) {
      setSotaPhases([{
        phase: 'qa_harness',
        label: 'Running tests/qa/run_all_phases (sync; ~5 s)...',
        status: 'running',
      }]);
      try {
        const resp = await fetch('/api/diagnostics/sota-bench-qa', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: '{}',
        });
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const data = await resp.json();
        const lines = (data.stdout || '').split('\n').filter(Boolean);
        setSotaLogs((prev) => [
          ...prev,
          ...lines.map((text) => ({ kind: 'log', text })),
          ...(data.stderr ? data.stderr.split('\n')
              .filter(Boolean)
              .map((text) => ({ kind: 'meta', text: `stderr: ${text}` })) : []),
          { kind: 'meta', text: `(exit code ${data.exit_code})` },
        ]);
        setSotaPhases([{
          phase: 'qa_harness',
          label: 'Running tests/qa/run_all_phases (sync)',
          status: data.ok ? 'pass' : 'fail',
        }]);
        setSotaResult({
          ok: data.ok,
          stage: 'qa_harness',
          message: data.ok
            ? 'QA harness PASSED. Bench skipped per "QA only" checkbox.'
            : `QA harness failed: ${data.error || `exit ${data.exit_code}`}`,
        });
      } catch (e) {
        setSotaResult({ ok: false, message: `Network error: ${e.message}` });
      } finally {
        setSotaRunning(false);
      }
      return;
    }

    // Step 2: Full bench - SSE streaming. Long-running, needs live updates.
    try {
      const resp = await fetch('/api/diagnostics/sota-bench', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ skip_bench: false }),
      });
      if (!resp.ok || !resp.body) {
        throw new Error(`HTTP ${resp.status}`);
      }
      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';
        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          try {
            const evt = JSON.parse(line.slice(6));
            if (evt.type === 'phase_start') {
              setSotaPhases((prev) => [
                ...prev,
                { phase: evt.data.phase, label: evt.data.label, status: 'running' },
              ]);
            } else if (evt.type === 'phase_result') {
              setSotaPhases((prev) =>
                prev.map((p) =>
                  p.phase === evt.data.phase ? { ...p, status: evt.data.status } : p,
                ),
              );
            } else if (evt.type === 'log') {
              setSotaLogs((prev) => [...prev, { kind: 'log', text: evt.data.line }]);
            } else if (evt.type === 'exit_code') {
              setSotaLogs((prev) => [
                ...prev,
                { kind: 'meta', text: `(process exited with code ${evt.data.code})` },
              ]);
            } else if (evt.type === 'heartbeat') {
              // Keep alive; do not surface.
            } else if (evt.type === 'complete') {
              setSotaResult(evt.data);
            }
          } catch {
            /* skip malformed */
          }
        }
      }
    } catch (e) {
      setSotaResult({
        ok: false,
        message: `Streaming failed: ${e.message}. Try "QA only" mode (synchronous).`,
      });
    } finally {
      setSotaRunning(false);
    }
  }, [sotaSkipBench]);

  // ── Upload + run SOTA pipeline on a user-supplied MP4 ─────────────
  const runSotaClipTest = useCallback(async () => {
    if (!sotaClipFile) return;
    setSotaRunning(true);
    setSotaPhases([]);
    setSotaLogs([]);
    setSotaResult(null);
    setCacheCheck(null);
    setSotaClipUploadProgress(0);

    try {
      // Step 1: upload the file. Use XMLHttpRequest so we can track
      // progress (fetch + ReadableStream upload progress is still
      // patchy across browsers).
      setSotaLogs((prev) => [...prev, {
        kind: 'meta',
        text: `[upload] sending ${sotaClipFile.name} (${(sotaClipFile.size / 1_048_576).toFixed(1)} MB) ...`,
      }]);

      const uploadJson = await new Promise((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        const formData = new FormData();
        formData.append('file', sotaClipFile);
        formData.append('content_type', sotaClipContentType || 'default');
        xhr.open('POST', '/api/diagnostics/sota-clip-upload');
        xhr.upload.onprogress = (ev) => {
          if (ev.lengthComputable) {
            setSotaClipUploadProgress(
              Math.round((ev.loaded / ev.total) * 100),
            );
          }
        };
        xhr.onload = () => {
          if (xhr.status >= 200 && xhr.status < 300) {
            try { resolve(JSON.parse(xhr.responseText)); }
            catch (e) { reject(new Error(`bad upload response: ${e.message}`)); }
          } else {
            reject(new Error(`upload HTTP ${xhr.status}: ${xhr.responseText}`));
          }
        };
        xhr.onerror = () => reject(new Error('upload network error'));
        xhr.send(formData);
      });

      setSotaClipUploadProgress(null);
      if (!uploadJson.ok) {
        throw new Error(uploadJson.error || 'upload rejected');
      }
      setSotaLogs((prev) => [...prev, {
        kind: 'meta',
        text: `[upload] done. token=${uploadJson.token?.slice(0, 8)}... slug=${uploadJson.slug} size=${uploadJson.size_mb} MB`,
      }]);

      // Step 2: SSE-stream the bench against the uploaded clip.
      // ``force`` triggers --force-reextract on the bench script
      // when the cache-state badge from a prior run was stale.
      const benchResp = await fetch('/api/diagnostics/sota-clip-bench', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          token: uploadJson.token,
          force: forceReextract,
        }),
      });
      // Reset the toggle so the next run defaults back to cache-fresh.
      setForceReextract(false);
      if (!benchResp.ok || !benchResp.body) {
        throw new Error(`bench HTTP ${benchResp.status}`);
      }
      const reader = benchResp.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';
        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          try {
            const evt = JSON.parse(line.slice(6));
            if (evt.type === 'phase_start') {
              setSotaPhases((prev) => [
                ...prev,
                { phase: evt.data.phase, label: evt.data.label, status: 'running' },
              ]);
            } else if (evt.type === 'phase_result') {
              setSotaPhases((prev) =>
                prev.map((p) => p.phase === evt.data.phase ? { ...p, status: evt.data.status } : p),
              );
              // Task A: capture the full cache_check payload so the
              // sticky badge below can render age / version / warning.
              if (evt.data.phase === 'cache_check') {
                setCacheCheck(evt.data);
              }
            } else if (evt.type === 'log') {
              setSotaLogs((prev) => [...prev, { kind: 'log', text: evt.data.line }]);
            } else if (evt.type === 'exit_code') {
              setSotaLogs((prev) => [...prev, {
                kind: 'meta', text: `(process exited with code ${evt.data.code})`,
              }]);
            } else if (evt.type === 'heartbeat') {
              // silent
            } else if (evt.type === 'complete') {
              setSotaResult(evt.data);
            }
          } catch { /* malformed - skip */ }
        }
      }
    } catch (e) {
      setSotaResult({
        ok: false,
        message: `Clip test failed: ${e.message}`,
      });
    } finally {
      setSotaClipUploadProgress(null);
      setSotaRunning(false);
    }
  }, [sotaClipFile, sotaClipContentType]);

  return (
    <div style={{ marginBottom: 32 }}>
      <h3 style={{ fontSize: 14, marginBottom: 16, color: 'var(--text-secondary)' }}>
        Pipeline Diagnostics
      </h3>

      {/* Live VRAM Gauge — always rendered */}
      <VramGauge
        gpu={gpuStatus}
        loadedModels={loadedModels}
        torchGpu={torchGpu}
        ollamaAvailable={ollamaAvailable}
        onUnload={handleUnload}
        onReleaseGpu={handleReleaseGpu}
        onRestart={handleRestart}
      />

      {/* Restart message */}
      {restartMsg && (
        <div style={{
          marginTop: 8, padding: '8px 12px', fontSize: 11, fontFamily: 'var(--font-mono)',
          color: 'var(--text-muted)', background: 'var(--bg-panel)',
          border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)',
        }}>
          {restartMsg}
        </div>
      )}

      {/* Pipeline Test Runner */}
      <div style={{ ...cardStyle, marginTop: 12 }}>
        <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 10 }}>
          Simulates the full video analysis pipeline — same order, same VRAM management.
          Catches OOM errors, model load failures, and GPU handoff issues before processing a real video.
        </div>

        {/* Test options */}
        <div style={{ display: 'flex', gap: 16, marginBottom: 12, flexWrap: 'wrap' }}>
          <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 11, color: 'var(--text-secondary)', cursor: 'pointer' }}>
            <input
              type="checkbox"
              checked={testIncludeWhisper}
              onChange={(e) => setTestIncludeWhisper(e.target.checked)}
              disabled={testRunning}
              style={{ accentColor: 'var(--accent-cyan)' }}
            />
            Test Whisper transcription
          </label>
          <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 11, color: testIncludeWhisper ? 'var(--text-secondary)' : 'var(--text-muted)', cursor: testIncludeWhisper ? 'pointer' : 'default' }}>
            <input
              type="checkbox"
              checked={testTranslation}
              onChange={(e) => setTestTranslation(e.target.checked)}
              disabled={testRunning || !testIncludeWhisper}
              style={{ accentColor: 'var(--accent-cyan)' }}
            />
            Test translation (ja→en)
          </label>
        </div>

        <button
          onClick={runTest}
          disabled={testRunning}
          style={{
            ...smallBtnStyle,
            background: testRunning ? 'var(--bg-elevated)' : 'var(--accent-cyan)',
            color: testRunning ? 'var(--text-muted)' : '#fff',
            cursor: testRunning ? 'default' : 'pointer',
            marginBottom: 12, padding: '6px 16px',
          }}
        >
          {testRunning ? 'Running...' : 'Run Pipeline Test'}
        </button>

        {testPhases.length > 0 && (
          <div style={{
            padding: '8px 12px', background: 'var(--bg-base)',
            border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)',
          }}>
            {testPhases.map((p) => <PhaseRow key={p.phase} phase={p} />)}
            {testOverall && (
              <div style={{
                marginTop: 8, padding: '8px 0', fontSize: 12, fontWeight: 600,
                color: testOverall.overall_status === 'pass' ? '#22c55e' : '#ef4444',
                textAlign: 'center',
              }}>
                {testOverall.overall_status === 'pass'
                  ? `\u2705 Pipeline ready \u2014 ${testOverall.summary?.whisper_tested ? 'Whisper + ' : ''}vision + text verified`
                  : '\u274c Pipeline has issues \u2014 check results above'}
              </div>
            )}
          </div>
        )}
      </div>

      {/* SOTA Reframing 2026 \u2014 1-click QA + bench runner */}
      <div style={{ ...cardStyle, marginTop: 12 }}>
        <div style={{
          fontSize: 12, fontWeight: 600, color: 'var(--text-primary)',
          marginBottom: 4,
        }}>
          2026 SOTA Reframing - Validate
        </div>
        <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 10 }}>
          Runs the local Phase A-E QA harness (101 mocked unit tests) and then the
          real-content fixture bench with every SOTA flag (SAMURAI tracking,
          CoTracker3 dense, AV saliency, CLIP composition head, editorial planner)
          turned ON. The QA stage is fast (~5 s); the bench stage needs the GPU
          and the fixture cache and can take several minutes.
        </div>

        <div style={{ display: 'flex', gap: 16, marginBottom: 12, flexWrap: 'wrap' }}>
          <label style={{
            display: 'flex', alignItems: 'center', gap: 6, fontSize: 11,
            color: 'var(--text-secondary)',
            cursor: sotaRunning ? 'default' : 'pointer',
          }}>
            <input
              type="checkbox"
              checked={sotaSkipBench}
              onChange={(e) => setSotaSkipBench(e.target.checked)}
              disabled={sotaRunning}
              style={{ accentColor: 'var(--accent-cyan)' }}
            />
            QA only (skip fixture bench)
          </label>
        </div>

        <button
          onClick={runSotaBench}
          disabled={sotaRunning}
          style={{
            ...smallBtnStyle,
            background: sotaRunning ? 'var(--bg-elevated)' : 'var(--accent-cyan)',
            color: sotaRunning ? 'var(--text-muted)' : '#fff',
            cursor: sotaRunning ? 'default' : 'pointer',
            marginBottom: 12, padding: '6px 16px',
          }}
        >
          {sotaRunning
            ? (sotaSkipBench ? 'Running QA...' : 'Running QA + bench...')
            : (sotaSkipBench ? 'Run QA harness' : 'Run QA + SOTA bench')}
        </button>

        {/* ── Clip-test: upload an MP4 and run the SOTA pipeline on it ── */}
        <div style={{
          marginTop: 4, marginBottom: 12, paddingTop: 12,
          borderTop: '1px dashed var(--border)',
        }}>
          <div style={{
            fontSize: 12, fontWeight: 600, color: 'var(--text-primary)',
            marginBottom: 4,
          }}>
            Test on your own MP4
          </div>
          <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 10 }}>
            Drop in an MP4 / MOV / WebM / MKV and the SOTA reframing pipeline
            runs on just that clip with all Phase A-E flags ON. Skips the
            fixture cache entirely. Auth required (you are logged in).
          </div>

          {/* Task B: AutoFlip references status + on-demand rebuild */}
          {refsState.expected > 0 && (
            <div style={{
              display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: 8,
              fontSize: 11, fontFamily: 'var(--font-mono)',
              color: 'var(--text-secondary)', marginBottom: 10,
            }}>
              <span>AutoFlip references:</span>
              <span style={{
                padding: '1px 6px',
                background: refsState.real > 0 ? 'rgba(34,197,94,0.12)' : 'var(--bg-base)',
                color: refsState.real > 0 ? '#22c55e' : 'var(--text-muted)',
                border: '1px solid var(--border)', borderRadius: 4,
              }}>
                {refsState.real}/{refsState.expected} real
              </span>
              <span style={{
                padding: '1px 6px',
                background: refsState.naive > 0 ? 'rgba(59,130,246,0.12)' : 'var(--bg-base)',
                color: refsState.naive > 0 ? '#3b82f6' : 'var(--text-muted)',
                border: '1px solid var(--border)', borderRadius: 4,
              }}>
                {refsState.naive}/{refsState.expected} naive
              </span>
              {/* AutoFlip Build button removed in Layer 1 PR. The
                  critic engine (saliency / quality / VLM / engagement)
                  is the supported quality bar now. */}
            </div>
          )}

          {/* Hidden native file input + visible "Choose..." trigger */}
          <input
            ref={sotaClipFileInputRef}
            type="file"
            accept="video/mp4,video/quicktime,video/webm,video/x-matroska,.mp4,.mov,.webm,.mkv"
            onChange={(e) => {
              const f = e.target.files?.[0] || null;
              setSotaClipFile(f);
              if (!f) setSotaClipUploadProgress(null);
            }}
            disabled={sotaRunning}
            style={{ display: 'none' }}
          />

          <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 10, flexWrap: 'wrap' }}>
            <button
              onClick={() => sotaClipFileInputRef.current?.click()}
              disabled={sotaRunning}
              style={{
                ...smallBtnStyle,
                padding: '4px 12px',
                cursor: sotaRunning ? 'default' : 'pointer',
              }}
            >
              {sotaClipFile ? 'Change file...' : 'Choose MP4...'}
            </button>
            {sotaClipFile && (
              <span style={{
                fontSize: 11, fontFamily: 'var(--font-mono)',
                color: 'var(--text-secondary)',
              }}>
                {sotaClipFile.name} ({(sotaClipFile.size / 1_048_576).toFixed(1)} MB)
              </span>
            )}
            {sotaClipFile && !sotaRunning && (
              <button
                onClick={() => {
                  setSotaClipFile(null);
                  if (sotaClipFileInputRef.current) {
                    sotaClipFileInputRef.current.value = '';
                  }
                }}
                style={{
                  ...smallBtnStyle, padding: '2px 8px', fontSize: 10,
                }}
                title="Clear selected file"
              >
                Clear
              </button>
            )}
          </div>

          {sotaClipFile && (
            <div style={{ marginBottom: 10 }}>
              <label style={{
                display: 'block', fontSize: 11,
                color: 'var(--text-secondary)', marginBottom: 4,
              }}>
                Content type (drives the editorial planner's per-genre playbook):
              </label>
              <select
                value={sotaClipContentType}
                onChange={(e) => setSotaClipContentType(e.target.value)}
                disabled={sotaRunning}
                style={{
                  width: '100%', maxWidth: 360, padding: '6px 10px',
                  fontSize: 11, fontFamily: 'var(--font-mono)',
                  background: 'var(--bg-elevated)',
                  color: 'var(--text-primary)',
                  border: '1px solid var(--border)',
                  borderRadius: 'var(--radius-sm)',
                  cursor: sotaRunning ? 'default' : 'pointer',
                }}
              >
                {SOTA_CONTENT_TYPES.map(({ value, label }) => (
                  <option key={value} value={value}>{label}</option>
                ))}
              </select>
            </div>
          )}

          <button
            onClick={runSotaClipTest}
            disabled={sotaRunning || !sotaClipFile}
            style={{
              ...smallBtnStyle,
              background: (sotaRunning || !sotaClipFile)
                ? 'var(--bg-elevated)' : 'var(--accent-cyan)',
              color: (sotaRunning || !sotaClipFile)
                ? 'var(--text-muted)' : '#fff',
              cursor: (sotaRunning || !sotaClipFile) ? 'default' : 'pointer',
              padding: '6px 16px',
            }}
          >
            {sotaRunning
              ? (sotaClipUploadProgress !== null
                  ? `Uploading ${sotaClipUploadProgress}%...`
                  : 'Running SOTA pipeline...')
              : 'Upload + Run SOTA pipeline on this MP4'}
          </button>

          {sotaClipUploadProgress !== null && (
            <div style={{
              marginTop: 8, height: 6,
              background: 'var(--bg-base)',
              borderRadius: 'var(--radius-sm)',
              overflow: 'hidden',
            }}>
              <div style={{
                height: '100%',
                width: `${sotaClipUploadProgress}%`,
                background: 'var(--accent-cyan)',
                transition: 'width 0.2s ease',
              }} />
            </div>
          )}
        </div>

        {sotaPhases.length > 0 && (
          <div style={{
            padding: '8px 12px', background: 'var(--bg-base)',
            border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)',
            marginBottom: 8,
          }}>
            {sotaPhases.map((p) => (
              <div key={p.phase} style={{
                display: 'flex', alignItems: 'center', gap: 8,
                padding: '4px 0', fontSize: 11, fontFamily: 'var(--font-mono)',
                color: 'var(--text-secondary)',
              }}>
                <span style={{ width: 18 }}>
                  {p.status === 'pass' && '\u2705'}
                  {p.status === 'fail' && '\u274c'}
                  {p.status === 'running' && <Spinner />}
                </span>
                <span>{p.label}</span>
              </div>
            ))}
          </div>
        )}

        {sotaLogs.length > 0 && (
          <div
            ref={sotaLogRef}
            style={{
              maxHeight: 280, overflowY: 'auto',
              padding: '8px 12px', background: 'var(--bg-base)',
              border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)',
              fontSize: 10, fontFamily: 'var(--font-mono)',
              color: 'var(--text-muted)', whiteSpace: 'pre-wrap',
              lineHeight: 1.45, marginBottom: 8,
            }}
          >
            {sotaLogs.map((l, i) => (
              <div key={i} style={{
                color: l.kind === 'phase' ? 'var(--accent-cyan)'
                     : l.kind === 'meta'  ? 'var(--text-secondary)'
                     : 'var(--text-muted)',
              }}>
                {l.text}
              </div>
            ))}
          </div>
        )}

        {sotaResult && (
          <div style={{
            padding: '8px 12px', fontSize: 12, fontWeight: 600,
            color: sotaResult.ok ? '#22c55e' : '#ef4444',
            background: sotaResult.ok
              ? 'rgba(34, 197, 94, 0.08)'
              : 'rgba(239, 68, 68, 0.08)',
            border: `1px solid ${sotaResult.ok ? '#22c55e' : '#ef4444'}`,
            borderRadius: 'var(--radius-sm)',
          }}>
            {sotaResult.ok ? '\u2705 ' : '\u274c '}{sotaResult.message}
          </div>
        )}

        {/* Layer 1+ critic engine summary. Renders when the bench
            populated ``result.critic`` for at least one clip. The
            backend SSE stream attaches the per-run aggregate to
            sotaResult.critic so this row lights up without a JSON
            re-fetch. */}
        {sotaResult?.critic?.saliency && (() => {
          const sal = sotaResult.critic.saliency;
          const inCrop = (
            sal.in_crop_mean ?? sal.in_crop_fraction
          );
          const salColor = (v) => {
            if (v == null) return 'var(--text-muted)';
            if (v >= 0.75) return '#22c55e';
            if (v >= 0.50) return '#ff9f0a';
            return '#ef4444';
          };
          return (
            <div style={{
              marginTop: 8,
              padding: '8px 12px',
              border: '1px solid var(--border)',
              borderRadius: 'var(--radius-sm)',
              background: 'var(--bg-base)',
            }}>
              <div style={{
                fontSize: 11, fontWeight: 600,
                color: 'var(--text-secondary)', marginBottom: 6,
              }}>
                Critic engine
              </div>
              <div style={{
                display: 'flex', gap: 12, flexWrap: 'wrap',
                fontSize: 12, alignItems: 'center',
              }}>
                <span style={{ minWidth: 110, color: 'var(--text-muted)' }}>
                  Saliency (L1):
                </span>
                <span style={{ color: salColor(inCrop), fontWeight: 600 }}>
                  {inCrop != null
                    ? `${Math.round(inCrop * 100)}% in-crop`
                    : 'not measured'}
                </span>
                {sal.windows_flagged > 0 && (
                  <span style={{ color: '#ff9f0a', fontSize: 11 }}>
                    {sal.windows_flagged} window{sal.windows_flagged === 1 ? '' : 's'} flagged
                  </span>
                )}
                {sal.windows_fixed > 0 && (
                  <span style={{ color: '#3b82f6', fontSize: 11 }}>
                    +{sal.windows_fixed} fixed
                  </span>
                )}
                {sal.backend && sal.backend !== 'tased_net' && (
                  <span style={{
                    color: 'var(--text-muted)', fontSize: 10,
                    padding: '1px 6px',
                    border: '1px solid var(--border)',
                    borderRadius: 4,
                  }}>
                    {sal.backend === 'spectral_residual'
                      ? '(spectral fallback)'
                      : `(${sal.backend})`}
                  </span>
                )}
              </div>
            </div>
          );
        })()}

        {/* Task A: cache state row \u2014 surfaces stale-cache warning so
            the operator knows when their numbers came from a cache
            written before the running container's code landed. */}
        {cacheCheck && cacheCheck.exists !== false && (
          <div style={{
            marginTop: 6,
            padding: '6px 10px', fontSize: 11,
            fontFamily: 'var(--font-mono)',
            color: cacheCheck.status === 'stale_warning' ? '#ff9f0a' : 'var(--text-muted)',
            background: cacheCheck.status === 'stale_warning'
              ? 'rgba(255, 159, 10, 0.06)' : 'transparent',
            border: cacheCheck.status === 'stale_warning'
              ? '1px solid rgba(255, 159, 10, 0.4)' : '1px solid transparent',
            borderRadius: 'var(--radius-sm)',
            display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: 8,
          }}>
            <span>
              {cacheCheck.status === 'stale_warning' ? '\u26a0 ' : '\u2705 '}
              Cache: {cacheCheck.status === 'stale_warning' ? 'STALE' : 'fresh'}
            </span>
            {cacheCheck.cache_age_seconds !== null && (
              <span>
                {' \u2022 '}
                {cacheCheck.cache_age_seconds < 60
                  ? `${cacheCheck.cache_age_seconds}s old`
                  : `${Math.round(cacheCheck.cache_age_seconds / 60)} min old`}
              </span>
            )}
            <span>
              {' \u2022 v'}{cacheCheck.cache_version ?? '?'}
              {cacheCheck.expected_version != null
                && cacheCheck.cache_version != null
                && cacheCheck.cache_version !== cacheCheck.expected_version
                && ` (code expects v${cacheCheck.expected_version})`}
            </span>
            {cacheCheck.status === 'stale_warning' && (
              <button
                onClick={() => {
                  setForceReextract(true);
                  runSotaClipTest();
                }}
                disabled={sotaRunning}
                style={{
                  ...smallBtnStyle,
                  padding: '2px 10px', fontSize: 10,
                  marginLeft: 'auto',
                  cursor: sotaRunning ? 'default' : 'pointer',
                  background: 'rgba(255, 159, 10, 0.16)',
                  color: '#ff9f0a',
                  borderColor: 'rgba(255, 159, 10, 0.6)',
                }}
                title="Re-run with --force-reextract to rebuild the cache from current code"
              >
                Force re-extract
              </button>
            )}
            {cacheCheck.warning && (
              <div style={{ width: '100%', fontStyle: 'italic', marginTop: 2 }}>
                {cacheCheck.warning}
              </div>
            )}
          </div>
        )}

        {/* Results + 9:16 preview - the bundle that proves the run worked */}
        {(sotaResult?.preview_url || sotaResult?.results_md_url) && (
          <div style={{
            marginTop: 12, padding: 12,
            background: 'var(--bg-base)',
            border: '1px solid var(--border)',
            borderRadius: 'var(--radius-sm)',
          }}>
            <div style={{
              fontSize: 12, fontWeight: 600,
              color: 'var(--text-primary)', marginBottom: 8,
            }}>
              Run artifacts
            </div>
            <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 12 }}>
              The bench scored framing decisions; the rendered MP4 lets you
              eyeball the actual output. Watch for: subject stays in frame,
              no jitter on speaker turns, no faces clipped at the edges,
              A/B cuts on speaker changes (panel content), tight on speaker
              / wide on dialogue silence.
            </div>

            {/* Big, prominent download buttons */}
            <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 12 }}>
              {sotaResult.preview_url && (
                <a
                  href={sotaResult.preview_url}
                  download
                  style={{
                    ...smallBtnStyle,
                    display: 'inline-flex', alignItems: 'center', gap: 6,
                    background: 'var(--accent-cyan)', color: '#fff',
                    border: '1px solid var(--accent-cyan)',
                    textDecoration: 'none',
                    padding: '8px 16px', fontSize: 12, fontWeight: 600,
                  }}
                  title="Download the SOTA-rendered 9:16 MP4"
                >
                  ⬇ Download 9:16 video
                  {sotaResult.preview_size_mb
                    ? ` (${sotaResult.preview_size_mb} MB)` : ''}
                </a>
              )}
              {sotaResult.results_md_url && (
                <a
                  href={sotaResult.results_md_url}
                  download
                  style={{
                    ...smallBtnStyle,
                    display: 'inline-flex', alignItems: 'center', gap: 6,
                    textDecoration: 'none',
                    padding: '8px 16px', fontSize: 12, fontWeight: 600,
                  }}
                  title="Download the bench's markdown rollup"
                >
                  ⬇ Download results.md
                </a>
              )}
              {sotaResult.results_json_url && (
                <a
                  href={sotaResult.results_json_url}
                  download
                  style={{
                    ...smallBtnStyle,
                    display: 'inline-flex', alignItems: 'center', gap: 6,
                    textDecoration: 'none',
                    padding: '8px 16px', fontSize: 12, fontWeight: 600,
                  }}
                  title="Download the bench's machine-readable JSON metrics"
                >
                  ⬇ Download results.json
                </a>
              )}
              {sotaResult.results_md_url && (
                <a
                  href={sotaResult.results_md_url}
                  target="_blank"
                  rel="noopener noreferrer"
                  style={{
                    ...smallBtnStyle,
                    display: 'inline-flex', alignItems: 'center', gap: 6,
                    textDecoration: 'none',
                    padding: '8px 16px', fontSize: 12, fontWeight: 600,
                  }}
                  title="Open the markdown in a new tab without downloading"
                >
                  ↗ View results.md
                </a>
              )}
            </div>

            {/* Inline player */}
            {sotaResult.preview_url && (
              <>
                <div style={{
                  fontSize: 11, fontWeight: 600,
                  color: 'var(--text-secondary)', marginBottom: 6,
                }}>
                  Watch the 9:16 preview
                </div>
                <video
                  key={sotaResult.preview_url}
                  src={sotaResult.preview_url}
                  controls
                  playsInline
                  preload="metadata"
                  style={{
                    width: '100%', maxWidth: 360, maxHeight: 640,
                    background: '#000', borderRadius: 'var(--radius-sm)',
                    display: 'block',
                  }}
                />
              </>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

// ── Shared Styles ───────────────────────────────────────────────────────
const cardStyle = {
  padding: '12px 16px', background: 'var(--bg-panel)',
  border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)',
};

const smallBtnStyle = {
  marginTop: 8, padding: '4px 12px', fontSize: 10, fontFamily: 'var(--font-mono)',
  background: 'var(--bg-elevated)', color: 'var(--text-secondary)',
  border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)',
  cursor: 'pointer',
};
