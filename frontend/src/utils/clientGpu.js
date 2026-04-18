/**
 * Client-side GPU detection via WebGPU, WebGL, and WebCodecs.
 * Detects all available GPUs in the user's browser and checks
 * hardware encoding capabilities.
 *
 * Supports: NVIDIA (D3D12/Vulkan), Intel (D3D12), AMD (D3D12/Vulkan),
 * Apple Silicon (Metal), and software renderers.
 */

export async function detectClientGPUs() {
  const gpus = [];
  const errors = [];

  // ── WebGPU Detection ──
  // Try multiple adapter request strategies to find all GPUs:
  //   1. 'high-performance' — usually the discrete GPU (NVIDIA/AMD)
  //   2. 'low-power' — usually the integrated GPU (Intel/Apple)
  //   3. no preference — browser's default choice (may differ from above)
  //   4. forceFallbackAdapter — software renderer (e.g. Microsoft Basic Render)
  if (typeof navigator !== 'undefined' && navigator.gpu) {
    const adapterRequests = [
      { powerPreference: 'high-performance' },
      { powerPreference: 'low-power' },
      {},                                    // default — may return a different adapter
      { forceFallbackAdapter: true },        // software renderer
    ];

    for (const reqOpts of adapterRequests) {
      try {
        const adapter = await navigator.gpu.requestAdapter(reqOpts);
        if (!adapter) continue;

        // adapter.info is a synchronous property in Chrome 121+.
        // Fall back to requestAdapterInfo() for older browsers.
        let info;
        try {
          info = adapter.info;
          // Some browsers return a GPUAdapterInfo object — ensure we can read it
          if (info && typeof info.vendor === 'undefined' && typeof adapter.requestAdapterInfo === 'function') {
            info = await adapter.requestAdapterInfo();
          }
        } catch {
          // Fallback for browsers where adapter.info throws or doesn't exist
          if (typeof adapter.requestAdapterInfo === 'function') {
            try { info = await adapter.requestAdapterInfo(); } catch { /* give up */ }
          }
        }
        info = info || {};

        const limits = adapter.limits || {};

        // adapter.features is a GPUSupportedFeatures (Set-like) — use Array.from for safety
        let features = [];
        try {
          features = Array.from(adapter.features || []);
        } catch {
          // Some implementations don't support iteration
          features = [];
        }

        const isFallback = adapter.isFallbackAdapter || reqOpts.forceFallbackAdapter;
        const powerPref = reqOpts.powerPreference || (isFallback ? 'fallback' : 'default');

        // Build a stable ID to deduplicate (same physical GPU across request strategies)
        const gpuId = `${info.vendor || 'unknown'}-${info.architecture || 'unknown'}-${info.device || powerPref}`;
        if (gpus.some(g => g.id === gpuId)) continue;

        const guessedType = isFallback ? 'software' :
          powerPref === 'high-performance' ? 'discrete' :
          powerPref === 'low-power' ? 'integrated' :
          _guessGPUTypeFromInfo(info);

        gpus.push({
          id: gpuId,
          name: _formatGPUName(info),
          vendor: info.vendor || 'unknown',
          architecture: info.architecture || '',
          device: info.device || '',
          description: info.description || '',
          type: guessedType,
          backend: isFallback ? 'webgpu-software' : 'webgpu',
          hasFP16: features.includes('shader-f16'),
          hasTimestampQuery: features.includes('timestamp-query'),
          maxBufferSize: limits.maxBufferSize || 0,
          maxStorageBufferBindingSize: limits.maxStorageBufferBindingSize || 0,
          estimatedVRAM_MB: Math.round((limits.maxBufferSize || 0) / (1024 * 1024)),
          features,
          powerPreference: powerPref,
          isFallbackAdapter: !!isFallback,
          whisperCapable: !isFallback && features.includes('shader-f16') && (limits.maxStorageBufferBindingSize || 0) >= 128 * 1024 * 1024,
        });
      } catch (e) {
        const label = reqOpts.forceFallbackAdapter ? 'fallback' : (reqOpts.powerPreference || 'default');
        console.warn(`WebGPU adapter request (${label}) failed:`, e);
        errors.push(`WebGPU ${label}: ${e?.message || e}`);
      }
    }
  }

  // ── WebGL Fallback (for GPU name identification + fallback detection) ──
  try {
    const canvas = document.createElement('canvas');
    const gl = canvas.getContext('webgl2') || canvas.getContext('webgl');
    if (gl) {
      const debugInfo = gl.getExtension('WEBGL_debug_renderer_info');
      if (debugInfo) {
        const renderer = gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL) || '';
        const vendor = gl.getParameter(debugInfo.UNMASKED_VENDOR_WEBGL) || '';
        const cleanName = _extractGPUNameFromANGLE(renderer);

        // Enrich existing WebGPU entries with the more readable WebGL renderer name
        for (const gpu of gpus) {
          if (!gpu.webglRenderer) {
            gpu.webglRenderer = renderer;
            gpu.webglVendor = vendor;
            // Use WebGL name if it's more descriptive than the WebGPU adapter name
            if (cleanName && _isGenericName(gpu.name)) {
              gpu.name = cleanName;
            }
          }
        }

        // If no WebGPU adapters found, add a WebGL-only entry so the user at
        // least sees their GPU — they can still use WebCodecs for encoding.
        if (gpus.length === 0 && renderer) {
          gpus.push({
            id: `webgl-${vendor}-${renderer}`.slice(0, 200),
            name: cleanName || renderer,
            vendor: vendor || 'unknown',
            architecture: '',
            device: '',
            type: _guessGPUType(renderer),
            backend: 'webgl-only',
            hasFP16: false,
            hasTimestampQuery: false,
            maxBufferSize: 0,
            maxStorageBufferBindingSize: 0,
            estimatedVRAM_MB: 0,
            features: [],
            powerPreference: null,
            whisperCapable: false,
            webglRenderer: renderer,
            webglVendor: vendor,
          });
        }
      }
    }
  } catch (e) {
    console.warn('WebGL detection failed:', e);
    errors.push(`WebGL: ${e?.message || e}`);
  }

  // Attach errors for diagnostics
  gpus._errors = errors;
  return gpus;
}

/**
 * Extract a clean GPU model name from ANGLE renderer strings.
 *
 * Examples:
 *   "ANGLE (Intel, Intel(R) UHD Graphics 630 (0x00003E9B) Direct3D11 ...)" → "Intel UHD Graphics 630"
 *   "ANGLE (NVIDIA, NVIDIA GeForce GTX 1650 Direct3D11 ...)" → "NVIDIA GeForce GTX 1650"
 *   "ANGLE (Apple, ANGLE Metal Renderer: Apple M3 Pro, ...)" → "Apple M3 Pro"
 *   "ANGLE (AMD, AMD Radeon RX 6800 XT Direct3D11 ...)" → "AMD Radeon RX 6800 XT"
 */
function _extractGPUNameFromANGLE(renderer) {
  if (!renderer) return '';

  // Apple Metal: "ANGLE (Apple, ANGLE Metal Renderer: Apple M3 Pro, Version ...)"
  const appleMatch = renderer.match(/Apple (M\d[\w\s]*?)(?:,|$)/);
  if (appleMatch) return `Apple ${appleMatch[1].trim()}`;

  // D3D11/D3D12 pattern: "ANGLE (Vendor, Full GPU Name (0xHEXID) Direct3D1x ...)"
  const d3dMatch = renderer.match(/ANGLE \([^,]+,\s*(.+?)\s*(?:\(0x[0-9A-Fa-f]+\)|Direct3D|,)/);
  if (d3dMatch) {
    let name = d3dMatch[1].trim();
    // Remove (R) and (TM) markers for cleaner display
    name = name.replace(/\(R\)/gi, '').replace(/\(TM\)/gi, '').replace(/\s{2,}/g, ' ').trim();
    return name;
  }

  // Vulkan pattern: "ANGLE (Vendor, GPU Name Vulkan ...)"
  const vkMatch = renderer.match(/ANGLE \([^,]+,\s*(.+?)\s*(?:Vulkan|,)/);
  if (vkMatch) return vkMatch[1].trim();

  // If it contains recognizable GPU names, return as-is
  if (/RTX|GTX|Radeon|Intel.*(?:UHD|HD|Iris|Arc)|Apple M\d|GeForce/i.test(renderer)) {
    return renderer;
  }

  return '';
}

/**
 * Check if a GPU name is generic/unhelpful (would benefit from WebGL enrichment).
 */
function _isGenericName(name) {
  if (!name) return true;
  const lower = name.toLowerCase();
  return lower === 'unknown gpu' || lower === 'unknown' ||
    lower.startsWith('gpu (') || // e.g. "GPU (gen-9.5)"
    /^[a-z]+ \([a-z0-9.-]+\)$/i.test(name); // e.g. "intel (gen-9.5)"
}

/**
 * Guess GPU type from renderer string.
 */
function _guessGPUType(renderer) {
  const r = (renderer || '').toLowerCase();
  if (/geforce|radeon|rx\s?\d|rtx|gtx/i.test(r)) return 'discrete';
  if (/intel|uhd|iris|hd graphics/i.test(r)) return 'integrated';
  if (/apple m\d/i.test(r)) return 'integrated';
  return 'unknown';
}

function _guessGPUTypeFromInfo(info) {
  const v = ((info.vendor || '') + ' ' + (info.device || '') + ' ' + (info.description || '')).toLowerCase();
  if (/geforce|radeon|rx\s?\d|rtx|gtx/i.test(v)) return 'discrete';
  if (/intel|uhd|iris|hd graphics/i.test(v)) return 'integrated';
  if (/apple m\d/i.test(v)) return 'integrated';
  if (/microsoft|basic render|swiftshader/i.test(v)) return 'software';
  return 'unknown';
}

function _formatGPUName(info) {
  if (info.device && info.device !== '') return info.device;
  if (info.description && info.description !== '') return info.description;
  if (info.architecture) return `${info.vendor || 'GPU'} (${info.architecture})`;
  return info.vendor || 'Unknown GPU';
}

/**
 * Check WebCodecs hardware encoding support for H.264 and HEVC.
 */
export async function detectWebCodecsCapabilities() {
  const caps = {
    h264HardwareEncode: false,
    hevcHardwareEncode: false,
    h264Decode: false,
    hevcDecode: false,
  };

  if (typeof VideoEncoder === 'undefined') return caps;

  // H.264 hardware encode check
  try {
    const h264Config = {
      codec: 'avc1.640028',
      width: 1920, height: 1080,
      bitrate: 5_000_000, framerate: 30,
      hardwareAcceleration: 'prefer-hardware',
    };
    const result = await VideoEncoder.isConfigSupported(h264Config);
    caps.h264HardwareEncode = result.supported === true;
  } catch { /* unsupported */ }

  // HEVC hardware encode check
  try {
    const hevcConfig = {
      codec: 'hev1.1.6.L120.B0',
      width: 1920, height: 1080,
      bitrate: 5_000_000, framerate: 30,
      hardwareAcceleration: 'prefer-hardware',
    };
    const result = await VideoEncoder.isConfigSupported(hevcConfig);
    caps.hevcHardwareEncode = result.supported === true;
  } catch { /* unsupported */ }

  // Decode checks
  try {
    const h264Dec = { codec: 'avc1.640028', hardwareAcceleration: 'prefer-hardware' };
    const result = await VideoDecoder.isConfigSupported(h264Dec);
    caps.h264Decode = result.supported === true;
  } catch { /* unsupported */ }

  try {
    const hevcDec = { codec: 'hev1.1.6.L120.B0', hardwareAcceleration: 'prefer-hardware' };
    const result = await VideoDecoder.isConfigSupported(hevcDec);
    caps.hevcDecode = result.supported === true;
  } catch { /* unsupported */ }

  return caps;
}

/**
 * Fetch server-detected GPUs from the backend and merge any that the browser
 * didn't find.  Server detection uses nvidia-smi, lspci, sysfs, WMIC, etc.
 * and is more reliable for enumerating all physical GPUs.
 */
async function _fetchAndMergeServerGPUs(browserGpus) {
  try {
    const res = await fetch('/api/gpu-acceleration');
    if (!res.ok) return;
    const data = await res.json();
    const serverGpus = data.detected?.gpus || data.gpus || [];
    if (!serverGpus.length) return;

    for (const sg of serverGpus) {
      const nameLower = (sg.name || '').toLowerCase().replace(/\(r\)/g, '').replace(/\(tm\)/g, '').replace(/\s+/g, ' ').trim();
      // Check if this server GPU is already in the browser list (fuzzy name match)
      const alreadyFound = browserGpus.some(bg => {
        const bgName = (bg.name || '').toLowerCase().replace(/\(r\)/g, '').replace(/\(tm\)/g, '').replace(/\s+/g, ' ').trim();
        // Match if one name contains the other, or they share significant words
        if (bgName.includes(nameLower) || nameLower.includes(bgName)) return true;
        const bgWords = bgName.split(/\s+/).filter(w => w.length > 2);
        const sgWords = nameLower.split(/\s+/).filter(w => w.length > 2);
        const shared = bgWords.filter(w => sgWords.includes(w));
        return shared.length >= 2;
      });
      if (alreadyFound) continue;

      browserGpus.push({
        id: `server-${sg.vendor || 'unknown'}-${sg.index || browserGpus.length}`,
        name: sg.name || 'Unknown GPU',
        vendor: sg.vendor || 'unknown',
        architecture: '',
        device: '',
        description: '',
        type: sg.type || (sg.vendor === 'intel' ? 'integrated' : 'discrete'),
        backend: 'server-detected',
        hasFP16: false,
        hasTimestampQuery: false,
        maxBufferSize: 0,
        maxStorageBufferBindingSize: 0,
        estimatedVRAM_MB: sg.vram_mb || 0,
        features: [],
        powerPreference: null,
        isFallbackAdapter: false,
        whisperCapable: false,
        serverIndex: sg.index,
        serverVendor: sg.vendor,
      });
    }
  } catch (e) {
    console.warn('Failed to fetch server GPU list:', e);
  }
}

/**
 * Full client GPU capability scan — call once on mount.
 */
export async function scanClientGPU() {
  const gpus = await detectClientGPUs();
  const scanErrors = gpus._errors || [];
  delete gpus._errors;

  // Merge server-detected GPUs that the browser didn't find
  await _fetchAndMergeServerGPUs(gpus);

  const webcodecs = await detectWebCodecsCapabilities();

  return {
    webgpuSupported: !!(typeof navigator !== 'undefined' && navigator.gpu),
    webcodecSupported: typeof VideoEncoder !== 'undefined',
    gpus,
    webcodecs,
    recommended: gpus.find(g => g.type === 'discrete' && g.backend === 'webgpu') || gpus.find(g => g.type === 'discrete') || gpus[0] || null,
    scanErrors: scanErrors.length > 0 ? scanErrors : undefined,
  };
}
