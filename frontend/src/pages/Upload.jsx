import React, { useState, useRef, useCallback, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import ProgressBar from '../components/ProgressBar';
import useResponsive from '../hooks/useResponsive';
import CloudSourceTabs from '../components/cloud/CloudSourceTabs';

const ACCEPTED = 'video/*,.mp4,.mov,.avi,.mkv,.webm,.m4v,.3gp';
const ACCEPTED_DISPLAY = 'MP4 \u00B7 MOV \u00B7 AVI \u00B7 MKV \u00B7 WEBM \u00B7 M4V \u00B7 3GP';
const VALID_EXTENSIONS = ['mp4', 'mov', 'avi', 'mkv', 'webm', 'm4v', '3gp', 'qt'];

const IS_MOBILE = /Android|iPhone|iPad|iPod/i.test(navigator.userAgent);
const IS_ANDROID = /Android/i.test(navigator.userAgent);
const MOBILE_CHUNK = 4 * 1024 * 1024;           // 4 MB — safe for iOS Safari memory
const DESKTOP_CHUNK_DEFAULT = 25 * 1024 * 1024;  // 25 MB — fewer HTTP round-trips
const MIN_CHUNK = 5 * 1024 * 1024;    // 5 MB
const MAX_CHUNK = 100 * 1024 * 1024;  // 100 MB

// Adaptive chunk sizing: mobile always 4MB; desktop uses stored throughput
function getAdaptiveChunkSize() {
  if (IS_MOBILE) return MOBILE_CHUNK;
  try {
    const stored = localStorage.getItem('clipai_chunk_speed');
    if (stored) {
      const bytesPerSec = parseFloat(stored);
      // Target ~3 seconds per chunk — larger chunks = fewer round-trips
      const ideal = Math.round(bytesPerSec * 3);
      // Floor at 15MB to ensure we don't regress to tiny chunks from old cache
      return Math.max(15 * 1024 * 1024, Math.min(ideal, MAX_CHUNK));
    }
  } catch { /* ignore */ }
  return DESKTOP_CHUNK_DEFAULT;
}

const CHUNK_SIZE = getAdaptiveChunkSize();
const MAX_RETRIES = 4;
const RETRY_DELAYS = [2000, 4000, 8000, 16000]; // exponential backoff

// Quick client-side check: read first 12 bytes to catch obviously corrupt files
function validateFileHeader(file) {
  return new Promise((resolve) => {
    const slice = file.slice(0, 12);
    const reader = new FileReader();
    reader.onload = () => {
      const bytes = new Uint8Array(reader.result);
      if (bytes.length < 8) {
        resolve('File is too small to be a valid video');
        return;
      }
      if (bytes.slice(0, 8).every((b) => b === 0)) {
        resolve(
          'This file appears to be corrupt or an incomplete download \u2014 ' +
          'the first bytes are all zeros. Try re-downloading or re-exporting the video.'
        );
        return;
      }
      resolve(null);
    };
    reader.onerror = () => resolve(null);
    reader.readAsArrayBuffer(slice);
  });
}

// CRC32 lookup table — generated once at module load (~10ms per 25MB chunk)
const CRC32_TABLE = new Uint32Array(256);
(function buildCRC32Table() {
  for (let i = 0; i < 256; i++) {
    let c = i;
    for (let j = 0; j < 8; j++) {
      c = (c & 1) ? (0xEDB88320 ^ (c >>> 1)) : (c >>> 1);
    }
    CRC32_TABLE[i] = c;
  }
})();

function crc32(buffer) {
  const view = new Uint8Array(buffer);
  let crc = 0xFFFFFFFF;
  for (let i = 0; i < view.length; i++) {
    crc = CRC32_TABLE[(crc ^ view[i]) & 0xFF] ^ (crc >>> 8);
  }
  return (crc ^ 0xFFFFFFFF) >>> 0;
}

// Streaming CRC32: reads blob in 256KB windows, never holds more than 256KB.
// Prevents OOM on mobile where blob.arrayBuffer() on a 4MB+ chunk can double-allocate.
const CRC_WINDOW = 256 * 1024;
async function crc32Streaming(blob) {
  let crc = 0xFFFFFFFF;
  let offset = 0;
  while (offset < blob.size) {
    const end = Math.min(offset + CRC_WINDOW, blob.size);
    const buf = await blob.slice(offset, end).arrayBuffer();
    const view = new Uint8Array(buf);
    for (let i = 0; i < view.length; i++) {
      crc = CRC32_TABLE[(crc ^ view[i]) & 0xFF] ^ (crc >>> 8);
    }
    offset = end;
  }
  return (crc ^ 0xFFFFFFFF) >>> 0;
}

// Upload a chunk via XHR with real-time byte-level progress
function uploadChunkXHR(formData, onProgress, timeoutMs = 300000) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    let aborted = false;
    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable && onProgress) onProgress(event.loaded, event.total);
    };
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try { resolve(JSON.parse(xhr.responseText)); }
        catch { resolve({ ok: true }); }
      } else {
        let errMsg = `HTTP ${xhr.status}`;
        try {
          const body = JSON.parse(xhr.responseText);
          if (body.detail) errMsg = typeof body.detail === 'string' ? body.detail : errMsg;
        } catch {}
        reject(new Error(errMsg));
      }
    };
    xhr.onerror = () => { if (!aborted) reject(new Error('Network error during chunk upload')); };
    xhr.ontimeout = () => reject(new Error('Chunk upload timed out'));
    xhr.timeout = timeoutMs;
    xhr.open('POST', '/api/upload/chunk');
    xhr.send(formData);
  });
}

// Compute SHA-256 hash of the full file using streaming reads
async function computeFileHash(file) {
  try {
    const SLICE = 2 * 1024 * 1024; // 2MB slices
    // Web Crypto doesn't support incremental hashing, so read full file
    // For very large files (>2GB) this may fail — fall back gracefully
    const buffer = await file.arrayBuffer();
    const hashBuffer = await crypto.subtle.digest('SHA-256', buffer);
    return Array.from(new Uint8Array(hashBuffer))
      .map((b) => b.toString(16).padStart(2, '0'))
      .join('');
  } catch {
    return '';
  }
}

const LANGUAGES = [
  { code: '', label: 'Auto-detect' },
  { code: 'en', label: 'English' },
  { code: 'es', label: 'Spanish' },
  { code: 'fr', label: 'French' },
  { code: 'de', label: 'German' },
  { code: 'it', label: 'Italian' },
  { code: 'pt', label: 'Portuguese' },
  { code: 'ru', label: 'Russian' },
  { code: 'ja', label: 'Japanese' },
  { code: 'ko', label: 'Korean' },
  { code: 'zh', label: 'Chinese' },
  { code: 'ar', label: 'Arabic' },
  { code: 'hi', label: 'Hindi' },
  { code: 'nl', label: 'Dutch' },
  { code: 'pl', label: 'Polish' },
  { code: 'tr', label: 'Turkish' },
  { code: 'vi', label: 'Vietnamese' },
  { code: 'th', label: 'Thai' },
  { code: 'uk', label: 'Ukrainian' },
  { code: 'sv', label: 'Swedish' },
];

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

function formatSpeed(bytesPerSec) {
  if (bytesPerSec < 1024 * 1024) return `${(bytesPerSec / 1024).toFixed(0)} KB/s`;
  return `${(bytesPerSec / (1024 * 1024)).toFixed(1)} MB/s`;
}

function formatETA(seconds) {
  if (!seconds || !isFinite(seconds) || seconds <= 0) return '--';
  if (seconds < 60) return `${Math.ceil(seconds)}s`;
  const m = Math.floor(seconds / 60);
  const s = Math.ceil(seconds % 60);
  return `${m}m ${s}s`;
}

// ── QA Check Display ────────────────────────────────────────────────────────

function QAReport({ qa, label }) {
  if (!qa || Object.keys(qa).length === 0) return null;
  const checks = Object.entries(qa).map(([key, val]) => {
    if (key === 'integrity' && typeof val === 'object') {
      return Object.entries(val).map(([subKey, subVal]) => ({
        name: `integrity.${subKey}`,
        ...subVal,
      }));
    }
    return [{ name: key, ...val }];
  }).flat();

  const allPassed = checks.every((c) => c.pass !== false);
  const bgColor = allPassed ? 'rgba(48, 209, 88, 0.1)' : 'rgba(255, 55, 95, 0.1)';
  const borderColor = allPassed ? 'rgba(48, 209, 88, 0.3)' : 'rgba(255, 55, 95, 0.3)';

  return (
    <div style={{
      marginTop: 12,
      padding: '10px 14px',
      background: bgColor,
      border: `1px solid ${borderColor}`,
      borderRadius: 'var(--radius-sm)',
      fontSize: 12,
    }}>
      <div style={{ fontWeight: 600, marginBottom: 6, display: 'flex', alignItems: 'center', gap: 6 }}>
        <span style={{ fontSize: 14 }}>{allPassed ? '\u2713' : '\u2717'}</span>
        {label || 'Upload QA Report'}
      </div>
      {checks.map((check, i) => {
        const passed = check.pass !== false;
        const icon = passed ? '\u2713' : '\u2717';
        const color = passed ? '#30D158' : '#FF375F';
        const detail = check.error
          || (check.expected !== undefined && !passed
            ? `expected ${check.expected}, got ${check.actual}`
            : check.note || '');
        return (
          <div key={i} style={{
            display: 'flex', alignItems: 'flex-start', gap: 6, padding: '2px 0',
            color: 'var(--text-secondary)',
          }}>
            <span style={{ color, flexShrink: 0, fontFamily: 'var(--font-mono)', fontSize: 11 }}>{icon}</span>
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11 }}>
              {check.name}{detail ? ` — ${detail}` : ''}
            </span>
          </div>
        );
      })}
    </div>
  );
}

// ── Chunk Progress Grid ─────────────────────────────────────────────────────

function ChunkGrid({ totalChunks, chunkStates }) {
  if (totalChunks <= 0) return null;
  // Only show grid for files with multiple chunks
  if (totalChunks <= 1) return null;

  // Collapse into a compact bar for very large chunk counts
  const maxVisible = 200;
  const showCompact = totalChunks > maxVisible;

  if (showCompact) {
    const done = Object.values(chunkStates).filter((s) => s === 'done').length;
    const failed = Object.values(chunkStates).filter((s) => s === 'error').length;
    const uploading = Object.values(chunkStates).filter((s) => s === 'uploading').length;
    return (
      <div style={{ marginTop: 8, fontSize: 11, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
        Chunks: {done}/{totalChunks} complete
        {uploading > 0 && <span style={{ color: 'var(--accent-cyan)' }}> | {uploading} uploading</span>}
        {failed > 0 && <span style={{ color: '#FF375F' }}> | {failed} failed</span>}
      </div>
    );
  }

  return (
    <div style={{
      marginTop: 8,
      display: 'flex',
      flexWrap: 'wrap',
      gap: 2,
    }}>
      {Array.from({ length: totalChunks }, (_, i) => {
        const state = chunkStates[i] || 'pending';
        const colors = {
          pending: 'var(--bg-elevated)',
          uploading: 'var(--accent-cyan)',
          done: '#30D158',
          error: '#FF375F',
          retrying: '#FF9F0A',
        };
        return (
          <div
            key={i}
            title={`Chunk ${i + 1}: ${state}`}
            style={{
              width: totalChunks > 100 ? 4 : totalChunks > 50 ? 6 : 8,
              height: totalChunks > 100 ? 4 : totalChunks > 50 ? 6 : 8,
              borderRadius: 1,
              background: colors[state] || colors.pending,
              transition: 'background 0.2s',
            }}
          />
        );
      })}
    </div>
  );
}

// ── Main Component ──────────────────────────────────────────────────────────

export default function Upload() {
  const [dragOver, setDragOver] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [language, setLanguage] = useState('');
  const [subtitleLanguage, setSubtitleLanguage] = useState('');
  const [contentTypeOverride, setContentTypeOverride] = useState('');
  const [gameType, setGameType] = useState('');
  // Phase 2 sub-dropdown values. Empty string = "auto" (let the
  // heuristic classifier decide). The normalizer rejects unknown
  // values server-side via normalize_anime_subtype / normalize_music_subtype.
  const [animeSubtype, setAnimeSubtype] = useState('');
  const [musicSubtype, setMusicSubtype] = useState('');
  const [sportsSubtype, setSportsSubtype] = useState('');
  const [uploading, setUploading] = useState(false);
  const [uploadDone, setUploadDone] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState(null);
  const [uploadPhase, setUploadPhase] = useState(''); // 'chunking', 'assembling', 'validating', 'complete'
  const [chunkStates, setChunkStates] = useState({});
  const [totalChunks, setTotalChunks] = useState(0);
  const [speed, setSpeed] = useState(0);
  const [eta, setEta] = useState(0);
  const [qaReport, setQaReport] = useState(null);
  const [retryCount, setRetryCount] = useState(0);
  const [uploadLog, setUploadLog] = useState([]);
  const fileRef = useRef(null);
  const cameraRef = useRef(null);
  const abortRef = useRef(false);
  const uploadIdRef = useRef(null);
  const wakeLockRef = useRef(null);
  const hiddenSinceRef = useRef(null);
  const bgFetchActiveRef = useRef(false);
  const [bgFetchActive, setBgFetchActive] = useState(false);
  const navigate = useNavigate();
  const { isMobile } = useResponsive();

  const addLog = useCallback((msg, level = 'info') => {
    const ts = new Date().toLocaleTimeString();
    setUploadLog((prev) => [...prev.slice(-49), { ts, msg, level }]);
  }, []);

  // Warn user before leaving during upload
  useEffect(() => {
    if (!uploading) return;
    const handler = (e) => {
      e.preventDefault();
      e.returnValue = '';
    };
    window.addEventListener('beforeunload', handler);
    return () => window.removeEventListener('beforeunload', handler);
  }, [uploading]);

  const handleFile = useCallback(async (file) => {
    if (!file) return;
    const ext = file.name.split('.').pop()?.toLowerCase() || '';
    const mimeOk = file.type && file.type.startsWith('video/');
    const extOk = VALID_EXTENSIONS.includes(ext);
    if (!mimeOk && !extOk) {
      setError(`Unsupported format — type="${file.type || 'unknown'}", extension=".${ext}". Accepted: video files (${VALID_EXTENSIONS.join(', ')})`);
      return;
    }
    const headerErr = await validateFileHeader(file);
    if (headerErr) {
      setError(headerErr);
      setSelectedFile(null);
      return;
    }
    setSelectedFile(file);
    setError(null);
    setUploadDone(false);
    setQaReport(null);
    setUploadLog([]);
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer?.files?.[0];
    handleFile(file);
  }, [handleFile]);

  const uploadChunkWithRetry = useCallback(async (uploadId, chunkIndex, blob, onChunkProgress) => {
    for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
      try {
        // Streaming CRC32: reads blob in 256KB windows, never double-allocates
        const checksum = await crc32Streaming(blob);

        const formData = new FormData();
        formData.append('upload_id', uploadId);
        formData.append('chunk_index', chunkIndex.toString());
        formData.append('chunk_crc32', checksum.toString());
        // Pass original blob directly — no new Blob([buffer]) copy
        formData.append('file', blob, `chunk_${chunkIndex}`);

        if (attempt > 0) {
          setChunkStates((prev) => ({ ...prev, [chunkIndex]: 'retrying' }));
          addLog(`Retrying chunk ${chunkIndex + 1} (attempt ${attempt + 1})`, 'warn');
        }

        // Use XHR for real-time upload progress within each chunk
        const result = await uploadChunkXHR(formData, onChunkProgress);
        return result;
      } catch (err) {
        if (abortRef.current) throw err;
        if (attempt < MAX_RETRIES) {
          const delay = RETRY_DELAYS[attempt] || 16000;
          addLog(`Chunk ${chunkIndex + 1} error: ${err.message}. Retrying in ${delay / 1000}s...`, 'warn');
          await new Promise((r) => setTimeout(r, delay));
        } else {
          throw err;
        }
      }
    }
  }, [addLog]);

  const handleUpload = async () => {
    if (!selectedFile) return;
    setUploading(true);
    setUploadDone(false);
    setProgress(0);
    setError(null);
    setUploadPhase('chunking');
    setChunkStates({});
    setQaReport(null);
    setRetryCount(0);
    setUploadLog([]);
    abortRef.current = false;
    bgFetchActiveRef.current = false;
    setBgFetchActive(false);
    hiddenSinceRef.current = null;

    // Acquire Wake Lock to keep screen alive during upload (iOS + Android)
    if ('wakeLock' in navigator) {
      try {
        wakeLockRef.current = await navigator.wakeLock.request('screen');
      } catch { /* Wake lock failure must not block upload */ }
    }

    // Visibility change handler: track hidden/visible for resume, re-acquire wake lock
    const onVisibilityChange = async () => {
      if (document.visibilityState === 'hidden') {
        hiddenSinceRef.current = Date.now();
      } else {
        // Re-acquire wake lock if it was released when page was hidden
        if ('wakeLock' in navigator && wakeLockRef.current === null) {
          try { wakeLockRef.current = await navigator.wakeLock.request('screen'); } catch {}
        }
      }
    };
    document.addEventListener('visibilitychange', onVisibilityChange);

    const file = selectedFile;
    const numChunks = Math.ceil(file.size / CHUNK_SIZE);
    setTotalChunks(numChunks);

    addLog(`Starting chunked upload: ${file.name} (${formatBytes(file.size)}, ${numChunks} chunks${IS_MOBILE ? `, ${formatBytes(CHUNK_SIZE)} mobile chunks` : ''})`);

    // Step 1: Try to resume a previous upload, or initialize a new session
    let uploadId, serverChunkSize, serverTotalChunks;
    // Resume key: sessionStorage keyed by file identity so page reload can offer to resume
    const resumeKey = `clipai_upload_${file.name}_${file.size}_${file.lastModified}`;
    let resumedChunks = new Set();

    try {
      // Check for a resumable upload (check sessionStorage first, fall back to localStorage)
      const savedUploadId = sessionStorage.getItem(resumeKey) || localStorage.getItem(resumeKey);
      if (savedUploadId) {
        try {
          const resumeResp = await fetch(`/api/upload/resume/${savedUploadId}`);
          if (resumeResp.ok) {
            const resumeData = await resumeResp.json();
            if (resumeData.state === 'uploading') {
              uploadId = savedUploadId;
              serverChunkSize = resumeData.chunk_size;
              serverTotalChunks = resumeData.total_chunks;
              resumedChunks = new Set(resumeData.chunks_received);
              setTotalChunks(serverTotalChunks);
              addLog(`Resuming upload ${uploadId.slice(0, 8)}... (${resumedChunks.size}/${serverTotalChunks} chunks already done)`);
              // Mark resumed chunks as done in the grid
              const doneStates = {};
              for (const idx of resumedChunks) doneStates[idx] = 'done';
              setChunkStates(doneStates);
            }
          }
        } catch { /* Fall through to fresh upload */ }
      }

      // If resume didn't work, init a new session
      if (!uploadId) {
        addLog('Initializing upload session...');
        const initResp = await fetch('/api/upload/init', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            filename: file.name,
            file_size: file.size,
            language,
            subtitle_language: subtitleLanguage,
            content_type_override: contentTypeOverride,
            game_type: gameType,
            anime_subtype: animeSubtype,
            music_subtype: musicSubtype,
            sports_subtype: sportsSubtype,
            chunk_size: CHUNK_SIZE,
          }),
        });

        if (!initResp.ok) {
          let msg = 'Failed to initialize upload';
          try {
            const body = await initResp.json();
            if (body.detail) msg = body.detail;
          } catch {}
          throw new Error(msg);
        }

        const initData = await initResp.json();
        uploadId = initData.upload_id;
        serverChunkSize = initData.chunk_size;
        serverTotalChunks = initData.total_chunks;
        setTotalChunks(serverTotalChunks);
        addLog(`Session created: ${uploadId.slice(0, 8)}... (${serverTotalChunks} chunks of ${formatBytes(serverChunkSize)})`);
      }

      uploadIdRef.current = uploadId;
      sessionStorage.setItem(resumeKey, uploadId);

      // ── Android Background Fetch path ──
      // No auth header needed: chunked upload endpoints have no auth dependency.
      if ('serviceWorker' in navigator && 'BackgroundFetchManager' in self) {
        try {
          const reg = await navigator.serviceWorker.ready;
          if (reg.backgroundFetch) {
            const chunkSz = serverChunkSize || CHUNK_SIZE;
            const nChunks = serverTotalChunks || numChunks;
            const requests = [];
            for (let ci = 0; ci < nChunks; ci++) {
              if (resumedChunks.has(ci)) continue;
              const s = ci * chunkSz;
              const e = Math.min(s + chunkSz, file.size);
              const fd = new FormData();
              fd.append('upload_id', uploadId);
              fd.append('chunk_index', ci.toString());
              fd.append('chunk_crc32', ''); // skip CRC for BG fetch — server validates size
              fd.append('file', file.slice(s, e), `chunk_${ci}`);
              requests.push(new Request('/api/upload/chunk', { method: 'POST', body: fd }));
            }
            const bgFetch = await reg.backgroundFetch.fetch(
              `clipai-upload-${uploadId}`,
              requests,
              { title: `Uploading ${file.name}`, icons: [], downloadTotal: file.size },
            );
            bgFetchActiveRef.current = true;
            setBgFetchActive(true);
            addLog(`Background Fetch started — upload will continue even if you close this tab`, 'success');

            // Mirror progress into the UI when tab is foregrounded
            bgFetch.addEventListener('progress', () => {
              if (bgFetch.downloaded > 0) {
                const pct = Math.min(90, Math.round((bgFetch.downloaded / file.size) * 90));
                setProgress(pct);
              }
            });

            // Wait for completion (tab still open)
            await new Promise((resolve, reject) => {
              const checkResult = (result) => {
                if (result === '') resolve(); // success
                else reject(new Error(`Background Fetch ended: ${result}`));
              };
              bgFetch.addEventListener('progress', () => {
                if (bgFetch.result) checkResult(bgFetch.result);
              });
              // Also check immediately in case already done
              if (bgFetch.result) checkResult(bgFetch.result);
            });

            // Background Fetch completed — now call /complete
            setProgress(92);
            setUploadPhase('assembling');
            addLog('Background upload complete. Finalizing...');
            const completeForm = new FormData();
            completeForm.append('upload_id', uploadId);
            completeForm.append('file_hash', '');
            const completeResp = await fetch('/api/upload/complete', { method: 'POST', body: completeForm });
            if (!completeResp.ok) throw new Error('Assembly failed after Background Fetch');
            const completeData = await completeResp.json();

            // Poll for assembly completion (same as foreground path below)
            if (completeData.poll) {
              const pollStart = Date.now();
              while (Date.now() - pollStart < 30 * 60 * 1000) {
                await new Promise(r => setTimeout(r, 2000));
                const statusResp = await fetch(`/api/upload/status/${uploadId}`);
                if (!statusResp.ok) continue;
                const status = await statusResp.json();
                if (status.state === 'complete') {
                  setUploadPhase('complete');
                  setProgress(100);
                  setUploadDone(true);
                  setQaReport(status.qa);
                  sessionStorage.removeItem(resumeKey);
                  addLog(`Upload complete! Job ID: ${status.job_id || completeData.job_id}`, 'success');
                  document.removeEventListener('visibilitychange', onVisibilityChange);
                  try { wakeLockRef.current?.release(); wakeLockRef.current = null; } catch {}
                  setTimeout(() => navigate(`/analysis/${status.job_id || completeData.job_id}`), 800);
                  return;
                }
                if (status.state === 'error') throw new Error(status.error || 'Assembly failed');
              }
              throw new Error('Assembly timed out');
            }

            // Direct response (no polling)
            setUploadPhase('complete');
            setProgress(100);
            setUploadDone(true);
            sessionStorage.removeItem(resumeKey);
            document.removeEventListener('visibilitychange', onVisibilityChange);
            try { wakeLockRef.current?.release(); wakeLockRef.current = null; } catch {}
            setTimeout(() => navigate(`/analysis/${completeData.job_id}`), 800);
            return;
          }
        } catch (bgErr) {
          // Background Fetch failed (quota, permission, unsupported) — fall through to foreground
          addLog(`Background Fetch unavailable (${bgErr.message}), using foreground upload`, 'warn');
          bgFetchActiveRef.current = false;
          setBgFetchActive(false);
        }
      }
    } catch (err) {
      setError(err.message);
      setUploading(false);
      addLog(`Init failed: ${err.message}`, 'error');
      document.removeEventListener('visibilitychange', onVisibilityChange);
      try { wakeLockRef.current?.release(); wakeLockRef.current = null; } catch {}
      return;
    }

    // Step 2: Upload chunks in parallel (6 concurrent) with progress tracking
    const CONCURRENCY = 6;
    const chunkSize = serverChunkSize || CHUNK_SIZE;
    const totalChunksActual = serverTotalChunks || numChunks;
    // Account for already-resumed chunks
    let bytesUploaded = 0;
    let completedCount = resumedChunks.size;
    for (const idx of resumedChunks) {
      const s = idx * chunkSize;
      const e = Math.min(s + chunkSize, file.size);
      bytesUploaded += (e - s);
    }
    const startTime = Date.now();
    let lastSpeedCalcTime = startTime;
    let lastSpeedCalcBytes = bytesUploaded;
    // Only queue chunks that haven't been uploaded yet
    const queue = Array.from({ length: totalChunksActual }, (_, i) => i).filter((i) => !resumedChunks.has(i));
    let uploadError = null;
    let lastLogPct = 0;  // Track last logged percentage for milestone logging

    addLog(`Uploading ${queue.length} chunks (${formatBytes(file.size - bytesUploaded)} remaining)...`);

    async function uploadNext() {
      while (queue.length > 0 && !abortRef.current && !uploadError) {
        const i = queue.shift();
        const start = i * chunkSize;
        const end = Math.min(start + chunkSize, file.size);
        const blob = file.slice(start, end);

        // Visibility-aware resume: if page was hidden, check server for completed chunks
        if (hiddenSinceRef.current !== null) {
          hiddenSinceRef.current = null;
          try {
            const resumeResp = await fetch(`/api/upload/resume/${uploadId}`);
            if (resumeResp.ok) {
              const resumeData = await resumeResp.json();
              const serverDone = new Set(resumeData.chunks_received || []);
              if (serverDone.has(i)) {
                // This chunk was already received — mark done and skip
                setChunkStates((prev) => ({ ...prev, [i]: 'done' }));
                const skippedBytes = end - start;
                bytesUploaded += skippedBytes;
                completedCount++;
                addLog(`Resumed: chunk ${i + 1} already on server, skipping`, 'info');
                continue;
              }
              // Also drain any other completed chunks from the queue
              for (let qi = queue.length - 1; qi >= 0; qi--) {
                if (serverDone.has(queue[qi])) {
                  const ci = queue.splice(qi, 1)[0];
                  const cs = ci * chunkSize;
                  const ce = Math.min(cs + chunkSize, file.size);
                  setChunkStates((prev) => ({ ...prev, [ci]: 'done' }));
                  bytesUploaded += (ce - cs);
                  completedCount++;
                }
              }
            }
          } catch { /* Resume check failed — just continue uploading */ }
        }

        setChunkStates((prev) => ({ ...prev, [i]: 'uploading' }));
        const chunkBytes = end - start;
        const chunkStartTime = Date.now();

        // Intra-chunk progress: updates progress bar smoothly during each chunk
        const onChunkProgress = (loaded, total) => {
          const partialBytes = (loaded / total) * chunkBytes;
          const totalSoFar = bytesUploaded + partialBytes;
          setProgress(Math.min(90, Math.round((totalSoFar / file.size) * 90)));
          const now = Date.now();
          if (now - lastSpeedCalcTime >= 500) {
            const elapsedSec = (now - chunkStartTime) / 1000;
            if (elapsedSec > 0) {
              const currentSpeed = partialBytes / elapsedSec;
              setSpeed(currentSpeed);
              setEta(currentSpeed > 0 ? (file.size - bytesUploaded - partialBytes) / currentSpeed : 0);
              lastSpeedCalcTime = now;
            }
          }
        };

        try {
          await uploadChunkWithRetry(uploadId, i, blob, onChunkProgress);
          setChunkStates((prev) => ({ ...prev, [i]: 'done' }));
          bytesUploaded += chunkBytes;
          completedCount++;

          // Store throughput for adaptive chunk sizing
          const chunkDuration = (Date.now() - chunkStartTime) / 1000;
          if (chunkDuration > 0) {
            try { localStorage.setItem('clipai_chunk_speed', (chunkBytes / chunkDuration).toString()); } catch {}
          }

          const pct = Math.round((completedCount / totalChunksActual) * 100);
          setProgress(Math.round((completedCount / totalChunksActual) * 90));

          // Log milestones at 10%, 25%, 50%, 75%, 90%, and every 25 chunks for large uploads
          const milestones = [10, 25, 50, 75, 90];
          const hitMilestone = milestones.find(m => pct >= m && lastLogPct < m);
          const chunkMilestone = totalChunksActual > 30 && completedCount % 25 === 0 && !hitMilestone;
          if (hitMilestone || chunkMilestone || completedCount === totalChunksActual) {
            lastLogPct = pct;
            const elapsedSec = (Date.now() - startTime) / 1000;
            const avgSpeed = bytesUploaded / Math.max(elapsedSec, 0.1);
            const etaStr = avgSpeed > 0 ? formatETA((file.size - bytesUploaded) / avgSpeed) : '';
            if (completedCount === totalChunksActual) {
              addLog(`All ${totalChunksActual} chunks uploaded (${formatBytes(file.size)}) in ${Math.round(elapsedSec)}s`);
            } else {
              addLog(`Upload ${pct}% — ${formatBytes(bytesUploaded)} / ${formatBytes(file.size)} at ${formatSpeed(avgSpeed)}${etaStr ? ` — ETA ${etaStr}` : ''}`);
            }
          }
        } catch (err) {
          setChunkStates((prev) => ({ ...prev, [i]: 'error' }));
          uploadError = err;
          throw err;
        }
      }
    }

    try {
      await Promise.all(Array.from({ length: CONCURRENCY }, () => uploadNext()));
    } catch (err) {
      if (abortRef.current) {
        addLog('Upload cancelled by user', 'warn');
        setError('Upload cancelled');
      } else {
        setError(`Upload failed: ${err.message}`);
        addLog(`Fatal error: ${err.message}`, 'error');
      }
      setUploading(false);
      document.removeEventListener('visibilitychange', onVisibilityChange);
      try { wakeLockRef.current?.release(); wakeLockRef.current = null; } catch {}
      return;
    }

    if (abortRef.current) {
      addLog('Upload cancelled by user', 'warn');
      setError('Upload cancelled');
      setUploading(false);
      document.removeEventListener('visibilitychange', onVisibilityChange);
      try { wakeLockRef.current?.release(); wakeLockRef.current = null; } catch {}
      return;
    }

    // Step 3: Complete — kick off assembly (non-blocking)
    // Skip whole-file SHA-256 — reading 1.6GB into browser memory takes 10-30s
    // and can crash on large files. Server validates integrity via size match.
    setUploadPhase('assembling');
    setProgress(92);
    addLog('All chunks uploaded. Finalizing upload...');

    try {
      const completeForm = new FormData();
      completeForm.append('upload_id', uploadId);
      completeForm.append('file_hash', '');

      // This returns immediately — assembly runs in background on server
      const completeResp = await fetch('/api/upload/complete', {
        method: 'POST',
        body: completeForm,
      });

      if (!completeResp.ok) {
        let msg = `Assembly/validation failed (HTTP ${completeResp.status})`;
        try {
          const text = await completeResp.text();
          try {
            const body = JSON.parse(text);
            if (body.detail) {
              if (typeof body.detail === 'object') {
                msg = body.detail.message || msg;
                if (body.detail.qa) setQaReport(body.detail.qa);
              } else {
                msg = String(body.detail);
              }
            }
          } catch {
            if (text && text.length < 500) msg += `: ${text}`;
          }
        } catch {}
        throw new Error(msg);
      }

      const completeData = await completeResp.json();

      // If server returned poll=true, poll /status until assembly is done
      if (completeData.poll) {
        const fileSizeMB = file.size / (1024 * 1024);
        // With write-in-place, finalization is just a rename + validation (~5-10s)
        const estimatedSeconds = Math.max(5, Math.round(fileSizeMB / 50));
        addLog('Finalizing upload — validating file integrity...');
        const pollStart = Date.now();
        // Scale poll timeout with file size: min 10 min, max 30 min
        const pollTimeout = Math.min(30, Math.max(10, Math.round(fileSizeMB / 100))) * 60 * 1000;
        let lastLoggedState = '';

        while (Date.now() - pollStart < pollTimeout) {
          if (abortRef.current) {
            throw new Error('Upload cancelled during assembly');
          }

          await new Promise(r => setTimeout(r, 2000)); // Poll every 2s

          const elapsedSec = Math.round((Date.now() - pollStart) / 1000);
          const remainingSec = Math.max(0, estimatedSeconds - elapsedSec);
          // Progress: 92% → 98% over the estimated duration
          const assemblyPct = Math.min(1, elapsedSec / Math.max(estimatedSeconds, 1));
          setProgress(Math.min(98, Math.round(92 + assemblyPct * 6)));

          try {
            const statusResp = await fetch(`/api/upload/status/${uploadId}`);
            if (!statusResp.ok) continue;
            const status = await statusResp.json();

            if (status.state === 'assembling' && lastLoggedState !== 'assembling-update') {
              if (elapsedSec > 0 && elapsedSec % 10 < 3) {
                addLog(`Finalizing: ${elapsedSec}s elapsed`);
                setProgress(Math.min(96, 92 + Math.min(4, elapsedSec / 2)));
                lastLoggedState = 'assembling-update';
                setTimeout(() => { lastLoggedState = ''; }, 8000);
              }
            } else if (status.state === 'validating') {
              if (lastLoggedState !== 'validating') {
                addLog('Validating file integrity...');
                lastLoggedState = 'validating';
                setProgress(97);
              } else if (elapsedSec > 0 && elapsedSec % 30 < 3 && lastLoggedState === 'validating') {
                // Log periodic updates during slow disk sync
                addLog(`Writing to disk: ${elapsedSec}s elapsed — this can take a few minutes on parity storage`);
              }
            } else if (status.state === 'complete') {
              setUploadPhase('complete');
              setProgress(100);
              setUploadDone(true);
              setQaReport(status.qa);
              sessionStorage.removeItem(resumeKey);
              const jobId = status.job_id || completeData.job_id;
              addLog(`Upload complete! Job ID: ${jobId}, QA: ${status.qa?.overall?.pass ? 'PASSED' : 'ISSUES FOUND'}`, 'success');
              document.removeEventListener('visibilitychange', onVisibilityChange);
              try { wakeLockRef.current?.release(); wakeLockRef.current = null; } catch {}
              setTimeout(() => {
                navigate(`/analysis/${jobId}`);
              }, 800);
              return; // Done!
            } else if (status.state === 'error') {
              const errMsg = status.error || 'Assembly failed on server';
              if (status.qa) setQaReport(status.qa);
              throw new Error(errMsg);
            }
            // state === 'assembling' — keep polling
          } catch (pollErr) {
            if (pollErr.message && !pollErr.message.includes('fetch')) {
              throw pollErr; // Re-throw assembly errors
            }
            // Network error during poll — retry
          }
        }
        throw new Error('Assembly timed out after 10 minutes');
      }

      // Legacy path: server returned result directly (no polling needed)
      setUploadPhase('complete');
      setProgress(100);
      setUploadDone(true);
      setQaReport(completeData.qa);
      sessionStorage.removeItem(resumeKey);
      addLog(`Upload complete! Job ID: ${completeData.job_id}, QA: ${completeData.qa?.overall?.pass ? 'PASSED' : 'ISSUES FOUND'}`, 'success');
      document.removeEventListener('visibilitychange', onVisibilityChange);
      try { wakeLockRef.current?.release(); wakeLockRef.current = null; } catch {}

      setTimeout(() => {
        navigate(`/analysis/${completeData.job_id}`);
      }, 800);
    } catch (err) {
      setError(err.message);
      setUploading(false);
      setUploadPhase('');
      addLog(`Completion failed: ${err.message}`, 'error');
      document.removeEventListener('visibilitychange', onVisibilityChange);
      try { wakeLockRef.current?.release(); wakeLockRef.current = null; } catch {}
    }
  };

  const cancelUpload = useCallback(async () => {
    abortRef.current = true;
    try { wakeLockRef.current?.release(); wakeLockRef.current = null; } catch {}
    if (uploadIdRef.current) {
      try {
        await fetch(`/api/upload/${uploadIdRef.current}`, { method: 'DELETE' });
      } catch { /* Best-effort cleanup */ }
      uploadIdRef.current = null;
    }
  }, []);

  // Post-upload countdown
  const [postUploadSeconds, setPostUploadSeconds] = useState(0);
  useEffect(() => {
    if (!uploadDone) { setPostUploadSeconds(0); return; }
    const interval = setInterval(() => setPostUploadSeconds((s) => s + 1), 1000);
    return () => clearInterval(interval);
  }, [uploadDone]);

  const uploadMessage = uploadDone
    ? 'Upload complete \u2014 preparing analysis pipeline...'
    : uploadPhase === 'assembling'
      ? 'Assembling file on server...'
      : uploadPhase === 'validating'
        ? 'Validating file integrity...'
        : progress >= 90
          ? 'Finishing upload...'
          : 'Uploading...';

  // Cloud import hook-in: when the user picks a file from Google Drive or
  // Box, the import endpoint returns a job_id immediately and the
  // existing /analysis/<job_id> route takes over (it opens the same
  // websocket channel the local upload progress UI uses).
  const handleCloudJobStart = useCallback(
    (jobId) => {
      if (!jobId) return;
      navigate(`/analysis/${jobId}`);
    },
    [navigate]
  );

  return (
    <div style={{ maxWidth: isMobile ? '100%' : 640, margin: '0 auto' }}>
      <h2 style={{ fontSize: isMobile ? 18 : 20, marginBottom: isMobile ? 20 : 24 }}>Upload Video</h2>

      {/* Cloud storage tabs — hidden entirely when the feature flag is off
          or the operator hasn't configured any cloud provider. */}
      {!uploading && (
        <CloudSourceTabs
          uploadMetadata={{
            language,
            subtitle_language: subtitleLanguage,
            content_type_override: contentTypeOverride,
            game_type: gameType,
            anime_subtype: animeSubtype,
            music_subtype: musicSubtype,
            sports_subtype: sportsSubtype,
          }}
          onJobStart={handleCloudJobStart}
        />
      )}

      {/* Drop zone */}
      <div
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
        onClick={() => !uploading && fileRef.current?.click()}
        style={{
          border: `2px dashed ${dragOver ? 'var(--accent-cyan)' : 'var(--border)'}`,
          background: dragOver ? 'var(--accent-cyan-dim)' : 'var(--bg-panel)',
          borderRadius: 'var(--radius-lg)',
          padding: isMobile ? '36px 16px' : '48px 24px',
          textAlign: 'center',
          cursor: uploading ? 'default' : 'pointer',
          transition: 'all 0.2s ease',
        }}
      >
        {/* No capture attribute — on iOS, capture forces camera-only and hides the library */}
        <input
          ref={fileRef}
          type="file"
          accept={ACCEPTED}
          onChange={(e) => handleFile(e.target.files?.[0])}
          style={{ display: 'none' }}
        />

        {!selectedFile ? (
          <>
            <div style={{ fontSize: 40, marginBottom: 12, opacity: 0.4 }}>&#x2B06;</div>
            <p style={{ color: 'var(--text-secondary)', marginBottom: 8 }}>
              {IS_MOBILE ? 'Tap to choose a video' : 'Drag and drop your video here, or click to browse'}
            </p>
            <p style={{ color: 'var(--text-muted)', fontSize: 12 }}>
              {ACCEPTED_DISPLAY}
            </p>
          </>
        ) : (
          <div>
            <div style={{ fontSize: 14, color: 'var(--text-primary)', marginBottom: 8, fontWeight: 600 }}>
              {selectedFile.name}
            </div>
            <div style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
              {formatBytes(selectedFile.size)}
            </div>
          </div>
        )}
      </div>

      {/* Record new video button — mobile only, uses capture="environment" for camera */}
      {IS_MOBILE && !uploading && (
        <div style={{ marginTop: 8, textAlign: 'center' }}>
          <input
            ref={cameraRef}
            type="file"
            accept="video/*"
            capture="environment"
            onChange={(e) => handleFile(e.target.files?.[0])}
            style={{ display: 'none' }}
          />
          <button
            onClick={() => cameraRef.current?.click()}
            style={{
              padding: '8px 16px',
              fontSize: 13,
              background: 'transparent',
              color: 'var(--text-secondary)',
              border: '1px solid var(--border)',
              borderRadius: 'var(--radius-sm)',
              cursor: 'pointer',
            }}
          >
            Record new video
          </button>
        </div>
      )}

      {/* Language selector */}
      {selectedFile && !uploading && (
        <div style={{ marginTop: 16 }}>
          <label
            htmlFor="lang-select"
            style={{ display: 'block', fontSize: 13, color: 'var(--text-secondary)', marginBottom: 6 }}
          >
            Video language (helps transcription accuracy)
          </label>
          <select
            id="lang-select"
            value={language}
            onChange={(e) => {
              setLanguage(e.target.value);
              if (!e.target.value || e.target.value === 'en') {
                setSubtitleLanguage('');
              }
            }}
            style={{
              width: '100%',
              padding: '10px 12px',
              background: 'var(--bg-panel)',
              color: 'var(--text-primary)',
              border: '1px solid var(--border)',
              fontSize: 14,
              borderRadius: 'var(--radius-sm)',
              outline: 'none',
            }}
          >
            {LANGUAGES.map(({ code, label }) => (
              <option key={code} value={code}>{label}</option>
            ))}
          </select>
        </div>
      )}

      {/* Subtitle translation selector */}
      {selectedFile && !uploading && (
        <div style={{ marginTop: 12 }}>
          <label
            htmlFor="subtitle-lang-select"
            style={{ display: 'block', fontSize: 13, color: 'var(--text-secondary)', marginBottom: 6 }}
          >
            Translate subtitles to (optional)
          </label>
          <select
            id="subtitle-lang-select"
            value={subtitleLanguage}
            onChange={(e) => setSubtitleLanguage(e.target.value)}
            style={{
              width: '100%',
              padding: '10px 12px',
              background: 'var(--bg-panel)',
              color: 'var(--text-primary)',
              border: '1px solid var(--border)',
              fontSize: 14,
              borderRadius: 'var(--radius-sm)',
              outline: 'none',
            }}
          >
            <option value="">No translation (keep original language)</option>
            {LANGUAGES.filter(l => l.code).map(({ code, label }) => (
              <option key={code} value={code}>{label}</option>
            ))}
          </select>
          {subtitleLanguage && (
            <p style={{ fontSize: 11, color: 'var(--text-tertiary)', marginTop: 4 }}>
              Subtitles will be automatically translated to {LANGUAGES.find(l => l.code === subtitleLanguage)?.label || subtitleLanguage} after transcription
            </p>
          )}
        </div>
      )}

      {/* Content type selector */}
      {selectedFile && !uploading && (
        <div style={{ marginTop: 12 }}>
          <label
            htmlFor="content-type-select"
            style={{ display: 'block', fontSize: 13, color: 'var(--text-secondary)', marginBottom: 6 }}
          >
            Content type (helps tracking accuracy)
          </label>
          <select
            id="content-type-select"
            value={contentTypeOverride}
            onChange={(e) => {
              const next = e.target.value;
              setContentTypeOverride(next);
              // Clear sub-dropdown state when leaving its parent so a
              // stale selection from a previous content-type pick can't
              // accidentally ride along to the upload init payload.
              if (!next.startsWith('gameplay') && next !== 'stream') setGameType('');
              if (next !== 'anime') setAnimeSubtype('');
              if (next !== 'music_video') setMusicSubtype('');
              if (!next.startsWith('sports')) setSportsSubtype('');
            }}
            style={{
              width: '100%',
              padding: '10px 12px',
              background: 'var(--bg-panel)',
              color: 'var(--text-primary)',
              border: '1px solid var(--border)',
              fontSize: 14,
              borderRadius: 'var(--radius-sm)',
              outline: 'none',
            }}
          >
            <option value="">Auto-detect</option>
            <optgroup label="People / Dialogue">
              <option value="podcast">Podcast / Interview (2+ seated)</option>
              <option value="debate">Debate / Panel (3+ seats)</option>
              <option value="vlog">Vlog / Single-subject handheld</option>
              <option value="narrative">Movie / TV / Cinematic dialogue</option>
            </optgroup>
            <optgroup label="Animation">
              <option value="anime">Anime / Cartoon</option>
            </optgroup>
            <optgroup label="Music / Performance">
              <option value="music_video">Music Video / Performance</option>
            </optgroup>
            <optgroup label="Gaming">
              <option value="gameplay">Gameplay — FPS / Hero Shooter</option>
              <option value="gameplay_moba">Gameplay — MOBA / Top-down</option>
              <option value="gameplay_tps">Gameplay — Third-person / Action</option>
              <option value="gameplay_racing">Gameplay — Racing / Driving</option>
              <option value="stream">Stream / Facecam + Gameplay</option>
            </optgroup>
            <optgroup label="Sports">
              <option value="sports">Sports (Auto-detect)</option>
              <option value="sports_basketball">Basketball</option>
              <option value="sports_racing">Racing / Motorsport</option>
            </optgroup>
          </select>
        </div>
      )}

      {/* Anime sub-type selector */}
      {selectedFile && !uploading && contentTypeOverride === 'anime' && (
        <div style={{ marginTop: 12 }}>
          <label
            htmlFor="anime-subtype-select"
            style={{ display: 'block', fontSize: 13, color: 'var(--text-secondary)', marginBottom: 6 }}
          >
            Anime sub-style
          </label>
          <select
            id="anime-subtype-select"
            value={animeSubtype}
            onChange={(e) => setAnimeSubtype(e.target.value)}
            style={{
              width: '100%',
              padding: '10px 12px',
              background: 'var(--bg-panel)',
              color: 'var(--text-primary)',
              border: '1px solid var(--border)',
              fontSize: 14,
              borderRadius: 'var(--radius-sm)',
              outline: 'none',
            }}
          >
            <option value="">Auto</option>
            <option value="action">Action / Fight</option>
            <option value="dialogue">Dialogue-heavy</option>
            <option value="slice_of_life">Slice of life</option>
          </select>
          <p style={{ fontSize: 11, color: 'var(--text-tertiary)', marginTop: 4 }}>
            Action picks faster intent + bigger lead-room; Dialogue follows speakers like cinematic dialogue.
          </p>
        </div>
      )}

      {/* Music-video sub-type selector */}
      {selectedFile && !uploading && contentTypeOverride === 'music_video' && (
        <div style={{ marginTop: 12 }}>
          <label
            htmlFor="music-subtype-select"
            style={{ display: 'block', fontSize: 13, color: 'var(--text-secondary)', marginBottom: 6 }}
          >
            Music-video sub-style
          </label>
          <select
            id="music-subtype-select"
            value={musicSubtype}
            onChange={(e) => setMusicSubtype(e.target.value)}
            style={{
              width: '100%',
              padding: '10px 12px',
              background: 'var(--bg-panel)',
              color: 'var(--text-primary)',
              border: '1px solid var(--border)',
              fontSize: 14,
              borderRadius: 'var(--radius-sm)',
              outline: 'none',
            }}
          >
            <option value="">Auto</option>
            <option value="performance">Performance / On-stage</option>
            <option value="narrative">Narrative / Story</option>
            <option value="lyric">Lyric / Static</option>
          </select>
          <p style={{ fontSize: 11, color: 'var(--text-tertiary)', marginTop: 4 }}>
            Performance and narrative use beat-snapped pulse cuts; Lyric stays static.
          </p>
        </div>
      )}

      {/* Sports sub-type selector (shown when the bare "sports" parent is picked) */}
      {selectedFile && !uploading && contentTypeOverride === 'sports' && (
        <div style={{ marginTop: 12 }}>
          <label
            htmlFor="sports-subtype-select"
            style={{ display: 'block', fontSize: 13, color: 'var(--text-secondary)', marginBottom: 6 }}
          >
            Sports sub-style
          </label>
          <select
            id="sports-subtype-select"
            value={sportsSubtype}
            onChange={(e) => setSportsSubtype(e.target.value)}
            style={{
              width: '100%',
              padding: '10px 12px',
              background: 'var(--bg-panel)',
              color: 'var(--text-primary)',
              border: '1px solid var(--border)',
              fontSize: 14,
              borderRadius: 'var(--radius-sm)',
              outline: 'none',
            }}
          >
            <option value="">Auto / Generic</option>
            <option value="basketball">Basketball (follow the ball)</option>
            <option value="racing">Racing / Motorsport (follow the lead car)</option>
          </select>
          <p style={{ fontSize: 11, color: 'var(--text-tertiary)', marginTop: 4 }}>
            Basketball tracks the ball; Racing biases toward the lower-third car crop.
          </p>
        </div>
      )}

      {/* Game selector (shown for any gameplay variant or stream) */}
      {selectedFile && !uploading && (
        contentTypeOverride.startsWith('gameplay') || contentTypeOverride === 'stream'
      ) && (
        <div style={{ marginTop: 12 }}>
          <label
            htmlFor="game-type-select"
            style={{ display: 'block', fontSize: 13, color: 'var(--text-secondary)', marginBottom: 6 }}
          >
            Game (for HUD layout)
          </label>
          <select
            id="game-type-select"
            value={gameType}
            onChange={(e) => setGameType(e.target.value)}
            style={{
              width: '100%',
              padding: '10px 12px',
              background: 'var(--bg-panel)',
              color: 'var(--text-primary)',
              border: '1px solid var(--border)',
              fontSize: 14,
              borderRadius: 'var(--radius-sm)',
              outline: 'none',
            }}
          >
            <option value="">Auto / Generic for this genre</option>
            {/* FPS / hero shooters */}
            {(contentTypeOverride === 'gameplay'
              || contentTypeOverride === 'stream') && (
              <optgroup label="FPS / Hero Shooter">
                <option value="overwatch">Overwatch / Overwatch 2</option>
                <option value="marvel_rivals">Marvel Rivals</option>
                <option value="valorant">Valorant</option>
                <option value="apex_legends">Apex Legends</option>
                <option value="fortnite">Fortnite</option>
                <option value="generic_fps">Other FPS</option>
              </optgroup>
            )}
            {/* MOBA */}
            {contentTypeOverride === 'gameplay_moba' && (
              <optgroup label="MOBA / Top-down">
                <option value="league_of_legends">League of Legends</option>
                <option value="dota2">Dota 2</option>
                <option value="generic_moba">Other MOBA</option>
              </optgroup>
            )}
            {/* TPS / action */}
            {contentTypeOverride === 'gameplay_tps' && (
              <optgroup label="Third-person Action">
                <option value="gta_v">Grand Theft Auto V</option>
                <option value="elden_ring">Elden Ring</option>
                <option value="generic_tps">Other Third-Person Action</option>
              </optgroup>
            )}
            {/* Racing */}
            {contentTypeOverride === 'gameplay_racing' && (
              <optgroup label="Racing / Driving">
                <option value="rocket_league">Rocket League</option>
                <option value="generic_racing">Other Racing / Driving</option>
              </optgroup>
            )}
            {/* Always-available sandbox option (Minecraft is the dominant case) */}
            {(contentTypeOverride === 'gameplay'
              || contentTypeOverride === 'stream') && (
              <optgroup label="Sandbox">
                <option value="minecraft">Minecraft</option>
              </optgroup>
            )}
          </select>
          <p style={{ fontSize: 11, color: 'var(--text-tertiary)', marginTop: 4 }}>
            Selecting the game enables HUD-aware vertical cropping (killfeed, health, abilities preserved in 9:16)
          </p>
        </div>
      )}

      {/* Upload complete banner */}
      {uploadDone && (
        <div style={{
          marginTop: 16,
          padding: '16px 20px',
          background: 'var(--success-dim)',
          border: '1px solid var(--success-border)',
          borderRadius: 'var(--radius-sm)',
          textAlign: 'center',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8, marginBottom: 8 }}>
            <div style={{
              width: 16, height: 16, border: '2px solid var(--success)',
              borderTopColor: 'transparent', borderRadius: '50%',
              animation: 'spin 0.8s linear infinite',
            }} />
            <span style={{ color: 'var(--success)', fontSize: 14, fontWeight: 600 }}>
              Upload complete — starting analysis pipeline...
            </span>
          </div>
          <p style={{ fontSize: 12, color: 'var(--text-muted)', margin: '0 0 12px', lineHeight: 1.5 }}>
            Redirecting to analysis page{postUploadSeconds > 0 ? ` (${postUploadSeconds}s)` : ''}...
            You'll see live progress for each step on the analysis page.
          </p>
          <div style={{ textAlign: 'left', margin: '0 auto', maxWidth: 340 }}>
            {[
              { label: 'Frame extraction', est: '~30s' },
              { label: 'Audio transcription', est: '~1-2 min' },
              { label: 'Scene analysis', est: '~1-2 min' },
              { label: 'Viral clip detection', est: '~30s' },
            ].map((step, i) => (
              <div key={i} style={{
                display: 'flex', alignItems: 'center', gap: 8,
                padding: '4px 0', fontSize: 12, color: 'var(--text-secondary)',
              }}>
                <span style={{
                  width: 6, height: 6, borderRadius: '50%',
                  background: 'var(--text-muted)', flexShrink: 0,
                }} />
                <span style={{ flex: 1 }}>{step.label}</span>
                <span style={{ fontSize: 11, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>{step.est}</span>
              </div>
            ))}
          </div>
          <p style={{ fontSize: 11, color: 'var(--text-muted)', margin: '10px 0 0', lineHeight: 1.4, fontStyle: 'italic' }}>
            Total estimated time: 3-5 minutes depending on video length
          </p>
          <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
        </div>
      )}

      {/* Progress */}
      {uploading && (
        <div style={{ marginTop: 16 }}>
          <ProgressBar
            progress={progress}
            message={uploadMessage}
            variant={uploadDone ? 'green' : 'cyan'}
          />

          {/* Speed / ETA / Chunk info */}
          {!uploadDone && uploadPhase === 'chunking' && (
            <div style={{
              display: 'flex',
              justifyContent: 'space-between',
              fontSize: 11,
              color: 'var(--text-muted)',
              fontFamily: 'var(--font-mono)',
              marginTop: 6,
              padding: '0 2px',
            }}>
              <span>{formatBytes(Math.round(progress / 90 * selectedFile.size))} / {formatBytes(selectedFile.size)}</span>
              <span>{speed > 0 ? formatSpeed(speed) : '--'}</span>
              <span>ETA: {speed > 0 ? formatETA(eta) : '--'}</span>
            </div>
          )}

          {/* Chunk progress grid */}
          <ChunkGrid totalChunks={totalChunks} chunkStates={chunkStates} />

          {/* Debug: chunk size info */}
          <div style={{ marginTop: 4, fontSize: 10, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', opacity: 0.6 }}>
            Chunk size: {formatBytes(CHUNK_SIZE)}{IS_MOBILE ? ' (mobile)' : ' (desktop)'}
          </div>
        </div>
      )}

      {/* Android Background Fetch banner */}
      {uploading && !uploadDone && bgFetchActive && IS_ANDROID && (
        <div style={{
          marginTop: 12,
          padding: '10px 14px',
          background: 'rgba(48, 209, 88, 0.08)',
          border: '1px solid rgba(48, 209, 88, 0.3)',
          borderRadius: 'var(--radius-sm)',
          fontSize: 12,
          color: '#30D158',
          lineHeight: 1.4,
        }}>
          Upload running in background — you can close this tab.
        </div>
      )}

      {/* iOS foreground upload warning */}
      {uploading && !uploadDone && IS_MOBILE && !bgFetchActive && !IS_ANDROID && (
        <div style={{
          marginTop: 12,
          padding: '10px 14px',
          background: 'rgba(245, 158, 11, 0.08)',
          border: '1px solid rgba(245, 158, 11, 0.3)',
          borderRadius: 'var(--radius-sm)',
          fontSize: 12,
          color: 'var(--accent-amber)',
          lineHeight: 1.4,
        }}>
          Keep this tab visible — iOS limits background uploads. If you switch apps, the upload will resume when you return.
        </div>
      )}

      {/* Upload in-progress warning + cancel */}
      {uploading && !uploadDone && (
        <div style={{
          marginTop: 12,
          padding: '10px 14px',
          background: 'rgba(245, 158, 11, 0.08)',
          border: '1px solid rgba(245, 158, 11, 0.3)',
          borderRadius: 'var(--radius-sm)',
          display: 'flex',
          flexDirection: 'column',
          gap: 6,
        }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <span style={{ fontSize: 16, flexShrink: 0 }}>&#9888;</span>
              <span style={{ fontSize: 12, color: 'var(--accent-amber)', lineHeight: 1.4 }}>
                Upload in progress — do not close this tab or navigate away until the upload is complete.
              </span>
            </div>
            <button
              onClick={cancelUpload}
              style={{
                padding: '4px 12px',
                fontSize: 11,
                background: 'rgba(255, 55, 95, 0.15)',
                color: '#FF375F',
                border: '1px solid rgba(255, 55, 95, 0.3)',
                borderRadius: 'var(--radius-sm)',
                cursor: 'pointer',
                flexShrink: 0,
              }}
            >
              Cancel
            </button>
          </div>
          <span style={{ fontSize: 11, color: 'var(--text-muted)', lineHeight: 1.4, paddingLeft: 24 }}>
            Using chunked upload ({formatBytes(CHUNK_SIZE)} chunks) with automatic retry for reliability.
            {selectedFile && selectedFile.size > 100 * 1024 * 1024
              ? ' Large file detected — chunked upload ensures the transfer won\'t stall.'
              : ''}
          </span>
        </div>
      )}

      {/* QA Report */}
      {qaReport && <QAReport qa={qaReport} label="Upload QA Validation" />}

      {/* Upload Log */}
      {uploadLog.length > 0 && (
        <details style={{ marginTop: 12 }}>
          <summary style={{
            fontSize: 11,
            color: 'var(--text-muted)',
            cursor: 'pointer',
            userSelect: 'none',
          }}>
            Upload Log ({uploadLog.length} entries)
          </summary>
          <div style={{
            marginTop: 4,
            padding: '8px 10px',
            background: 'var(--bg-panel)',
            border: '1px solid var(--border)',
            borderRadius: 'var(--radius-sm)',
            maxHeight: 200,
            overflowY: 'auto',
            fontSize: 10,
            fontFamily: 'var(--font-mono)',
            lineHeight: 1.6,
          }}>
            {uploadLog.map((entry, i) => {
              const colors = { info: 'var(--text-muted)', warn: '#FF9F0A', error: '#FF375F', success: '#30D158' };
              return (
                <div key={i} style={{ color: colors[entry.level] || colors.info }}>
                  <span style={{ opacity: 0.5 }}>{entry.ts}</span> {entry.msg}
                </div>
              );
            })}
          </div>
        </details>
      )}

      {/* Error */}
      {error && (
        <div style={{ marginTop: 16, padding: '10px 16px', background: 'var(--danger-dim)', border: '1px solid var(--danger)', color: 'var(--danger)', fontSize: 13, borderRadius: 'var(--radius-sm)' }}>
          {error}
        </div>
      )}

      {/* Upload button */}
      {selectedFile && !uploading && (
        <button
          onClick={handleUpload}
          style={{
            marginTop: 16,
            width: '100%',
            padding: '12px',
            background: 'var(--accent-cyan)',
            color: 'var(--bg-base)',
            border: 'none',
            fontSize: 14,
            fontWeight: 600,
            borderRadius: 'var(--radius-sm)',
          }}
        >
          Upload & Analyze
        </button>
      )}
    </div>
  );
}
