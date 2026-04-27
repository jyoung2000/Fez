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

  // ── 2026 SOTA Reframing — 1-click QA + bench runner ───────────────
  const [sotaRunning, setSotaRunning] = useState(false);
  const [sotaSkipBench, setSotaSkipBench] = useState(false);
  const [sotaPhases, setSotaPhases] = useState([]);
  const [sotaLogs, setSotaLogs] = useState([]);
  const [sotaResult, setSotaResult] = useState(null);
  const sotaLogRef = React.useRef(null);

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
    try {
      const resp = await fetch('/api/diagnostics/sota-bench', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ skip_bench: sotaSkipBench }),
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
              // Keep the connection alive; do not surface to user.
            } else if (evt.type === 'complete') {
              setSotaResult(evt.data);
            }
          } catch {
            /* skip malformed */
          }
        }
      }
    } catch (e) {
      setSotaResult({ ok: false, message: `Network error: ${e.message}` });
    } finally {
      setSotaRunning(false);
    }
  }, [sotaSkipBench]);

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
