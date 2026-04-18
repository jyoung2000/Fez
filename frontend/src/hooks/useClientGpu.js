import { useState, useEffect, useRef } from 'react';

/**
 * Hook for reading the client GPU preferences from localStorage.
 * Settings.jsx is the producer — this hook is the consumer.
 */
export function useClientGpuPreferences() {
  const [prefs, setPrefs] = useState({
    enabled: false,
    selectedGpuId: '',
    selectedGpuName: '',
    whisperEnabled: true,
    encodingEnabled: true,
  });

  useEffect(() => {
    const load = () => {
      try {
        const saved = localStorage.getItem('clipai_client_gpu');
        if (saved) {
          const parsed = JSON.parse(saved);
          // Ensure all values are primitives (never objects) to prevent React error #310
          setPrefs({
            enabled: !!parsed.enabled,
            selectedGpuId: String(parsed.selectedGpuId || ''),
            selectedGpuName: String(parsed.selectedGpuName || ''),
            whisperEnabled: parsed.whisperEnabled !== false,
            encodingEnabled: parsed.encodingEnabled !== false,
          });
        }
      } catch { /* ignore */ }
    };
    load();

    // Listen for changes from other tabs and same-tab updates
    window.addEventListener('storage', load);
    window.addEventListener('clientgpu-changed', load);
    return () => {
      window.removeEventListener('storage', load);
      window.removeEventListener('clientgpu-changed', load);
    };
  }, []);

  const shouldUseClientWhisper = prefs.enabled && prefs.whisperEnabled;
  const shouldUseClientEncoding = prefs.enabled && prefs.encodingEnabled;

  return { ...prefs, shouldUseClientWhisper, shouldUseClientEncoding };
}

/**
 * Auto-detect and enable client GPU on first visit when WebGPU is available.
 * Runs once per browser — if already configured, does nothing.
 * Call this from a top-level component (e.g. Layout).
 */
export function useAutoDetectGpu() {
  const ranRef = useRef(false);

  useEffect(() => {
    if (ranRef.current) return;
    ranRef.current = true;

    // Skip if already configured
    const existing = localStorage.getItem('clipai_client_gpu');
    if (existing) {
      try {
        const parsed = JSON.parse(existing);
        // If already configured (enabled or explicitly disabled), respect that
        if (parsed.selectedGpuId) return;
      } catch { /* continue to auto-detect */ }
    }

    // Only auto-detect if WebGPU is available
    if (typeof navigator === 'undefined' || !navigator.gpu) return;

    // Run scan asynchronously
    (async () => {
      try {
        const { scanClientGPU } = await import('../utils/clientGpu.js');
        const info = await scanClientGPU();

        if (!info.gpus || info.gpus.length === 0) return;

        const gpu = info.recommended || info.gpus[0];
        if (!gpu) return;

        const state = {
          enabled: true,
          selectedGpuId: String(gpu.id || ''),
          selectedGpuName: String(gpu.name || ''),
          whisperEnabled: !!gpu.whisperCapable,
          encodingEnabled: !!info.webcodecSupported,
        };
        localStorage.setItem('clipai_client_gpu', JSON.stringify(state));
        window.dispatchEvent(new Event('clientgpu-changed'));

        // Report to server
        fetch('/api/client-gpu-report', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            webgpu_supported: info.webgpuSupported || false,
            webcodec_supported: info.webcodecSupported || false,
            gpu_name: gpu.name || '',
            gpu_vendor: gpu.vendor || '',
            estimated_vram_mb: gpu.estimatedVRAM_MB || 0,
            whisper_capable: gpu.whisperCapable || false,
            h264_hardware_encode: info.webcodecs?.h264HardwareEncode || false,
            hevc_hardware_encode: info.webcodecs?.hevcHardwareEncode || false,
            gpu_index: gpu.serverIndex || '0',
            gpu_backend: gpu.backend || '',
          }),
        }).catch(() => {});

        console.info('[GPU] Auto-detected and enabled:', gpu.name);
      } catch (e) {
        console.warn('[GPU] Auto-detection failed:', e);
      }
    })();
  }, []);
}
