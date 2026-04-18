/**
 * Lightweight filmstrip thumbnail cache for the timeline.
 *
 * On long-zoom views the user wants to see *what* is in each clip, not
 * just a colored bar. This module:
 *
 *  1. Takes a clip ``mediaRef`` (the source video URL) and a list of
 *     timestamps to thumbnail.
 *  2. Lazily extracts JPEG-encoded thumbs via a hidden ``<video>``
 *     element seeking to each timestamp.
 *  3. Caches results in-memory keyed on ``(src, t, w, h)`` so subsequent
 *     redraws are O(1).
 *  4. Reports completion via a callback so the Timeline can repaint
 *     once the thumbs are ready.
 *
 * Designed to be the single, throttled thumbnail source for both the
 * canvas Timeline and any future overlay strip.
 */

const _CACHE = new Map();         // key -> ImageBitmap
const _PENDING = new Map();       // key -> Promise<ImageBitmap>
const _ELEMENTS = new Map();      // src -> shared <video> element

const _MAX_CACHE = 600;           // keep the cache bounded

function _key(src, t, w, h) {
  return `${src}|${t.toFixed(2)}|${w}x${h}`;
}

function _getVideoEl(src) {
  let el = _ELEMENTS.get(src);
  if (el) return el;
  el = document.createElement('video');
  el.crossOrigin = 'anonymous';
  el.preload = 'metadata';
  el.muted = true;
  el.playsInline = true;
  el.src = src;
  el.style.position = 'fixed';
  el.style.left = '-99999px';
  el.style.width = '1px';
  el.style.height = '1px';
  el.style.pointerEvents = 'none';
  document.body.appendChild(el);
  _ELEMENTS.set(src, el);
  return el;
}

// Cached feature support so we don't pay the typeof check per thumb.
const _SUPPORTS_OFFSCREEN = typeof OffscreenCanvas === 'function';
const _SUPPORTS_BITMAP = typeof globalThis.createImageBitmap === 'function';

function _seekAndCapture(el, t, w, h) {
  // Browsers without ``OffscreenCanvas`` (Safari < 16.4, older mobile)
  // would throw inside the seek handler and surface as silent thumbnail
  // gaps. Bail out cleanly so callers fall back to the colored bar.
  if (!_SUPPORTS_OFFSCREEN || !_SUPPORTS_BITMAP) {
    return Promise.reject(new Error('OffscreenCanvas/createImageBitmap unsupported'));
  }
  return new Promise((resolve, reject) => {
    let done = false;
    const onSeeked = () => {
      if (done) return;
      done = true;
      el.removeEventListener('seeked', onSeeked);
      el.removeEventListener('error', onError);
      try {
        const off = new OffscreenCanvas(w, h);
        const ctx = off.getContext('2d');
        if (!ctx) {
          reject(new Error('OffscreenCanvas 2D context unavailable'));
          return;
        }
        // ``cover``-style fit so the thumb shows what the user expects.
        const vw = el.videoWidth || w;
        const vh = el.videoHeight || h;
        const scale = Math.max(w / vw, h / vh);
        const drawW = vw * scale;
        const drawH = vh * scale;
        const dx = (w - drawW) / 2;
        const dy = (h - drawH) / 2;
        ctx.drawImage(el, dx, dy, drawW, drawH);
        off.convertToBlob({ type: 'image/jpeg', quality: 0.6 })
          .then(createImageBitmap)
          .then(resolve)
          .catch(reject);
      } catch (e) {
        reject(e);
      }
    };
    const onError = (e) => {
      if (done) return;
      done = true;
      el.removeEventListener('seeked', onSeeked);
      el.removeEventListener('error', onError);
      reject(e?.error || e || new Error('thumbnail seek failed'));
    };
    el.addEventListener('seeked', onSeeked, { once: true });
    el.addEventListener('error', onError, { once: true });
    try {
      el.currentTime = Math.max(0.01, t);
    } catch (e) {
      onError(e);
    }
  });
}

/** True iff this browser can actually generate filmstrip thumbnails. */
export function isFilmstripSupported() {
  return _SUPPORTS_OFFSCREEN && _SUPPORTS_BITMAP;
}

/**
 * Get a cached thumbnail synchronously, or ``null`` if it has to be
 * generated. The caller is expected to schedule generation via
 * ``ensureThumbnail`` and trigger a redraw on the returned promise.
 */
export function getCachedThumbnail(src, t, w, h) {
  if (!isFilmstripSupported()) return null;
  return _CACHE.get(_key(src, t, w, h)) || null;
}

/**
 * Lazily generate a thumbnail. Returns the same Promise for repeat
 * calls during generation so we don't seek the same time twice.
 */
export function ensureThumbnail(src, t, w, h) {
  if (!src) return Promise.reject(new Error('no src'));
  if (!isFilmstripSupported()) {
    return Promise.reject(new Error('filmstrip thumbnails unsupported on this browser'));
  }
  const key = _key(src, t, w, h);
  const hit = _CACHE.get(key);
  if (hit) return Promise.resolve(hit);
  const inflight = _PENDING.get(key);
  if (inflight) return inflight;

  const el = _getVideoEl(src);
  const ready = el.readyState >= 1
    ? Promise.resolve()
    : new Promise((res) => el.addEventListener('loadedmetadata', res, { once: true }));

  const p = ready.then(() => _seekAndCapture(el, t, w, h)).then((bitmap) => {
    _PENDING.delete(key);
    _CACHE.set(key, bitmap);
    if (_CACHE.size > _MAX_CACHE) {
      // Evict the oldest entry — Map preserves insertion order.
      const firstKey = _CACHE.keys().next().value;
      const oldBitmap = _CACHE.get(firstKey);
      _CACHE.delete(firstKey);
      try { oldBitmap.close && oldBitmap.close(); } catch {}
    }
    return bitmap;
  }).catch((err) => {
    _PENDING.delete(key);
    throw err;
  });
  _PENDING.set(key, p);
  return p;
}

/**
 * Drop every cached thumbnail for a source (e.g. when the source URL
 * changes or the editor unmounts).
 */
export function disposeFilmstrip(src) {
  for (const k of [..._CACHE.keys()]) {
    if (k.startsWith(`${src}|`)) {
      const bitmap = _CACHE.get(k);
      try { bitmap.close && bitmap.close(); } catch {}
      _CACHE.delete(k);
    }
  }
  const el = _ELEMENTS.get(src);
  if (el) {
    try { el.remove(); } catch {}
    _ELEMENTS.delete(src);
  }
}

/** How many timestamps to thumbnail across a clip given pixel width. */
export function computeThumbStops(clipStart, clipEnd, pps, thumbW) {
  const widthPx = (clipEnd - clipStart) * pps;
  const count = Math.max(2, Math.min(120, Math.floor(widthPx / Math.max(thumbW, 24))));
  const stops = [];
  for (let i = 0; i < count; i++) {
    const t = clipStart + ((i + 0.5) / count) * (clipEnd - clipStart);
    stops.push(t);
  }
  return stops;
}
