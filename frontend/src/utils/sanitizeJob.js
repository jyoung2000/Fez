/**
 * Deep-sanitize any value so it is safe to render as a React child.
 * Walks objects/arrays recursively and converts any nested object that
 * is NOT an array to a JSON string.  This is the catch-all defence
 * against React error #310 ("Objects are not valid as a React child").
 *
 * Allowed leaf types: string, number, boolean, null, undefined.
 * Arrays are recursed into.  Plain objects are recursed into but their
 * VALUES are coerced if they are themselves objects (unless whitelisted).
 *
 * @param {*} val           The value to sanitize.
 * @param {Set} [objectKeys] Keys whose object values should be preserved
 *                           (e.g. speakerColors, words, subtitle_settings).
 * @param {number} [depth]  Internal recursion guard.
 */
function deepSanitize(val, objectKeys, depth = 0) {
  if (val == null || typeof val !== 'object') return val;
  // At max depth, stringify any remaining objects instead of leaking them
  if (depth > 10) {
    try { return JSON.stringify(val); } catch { return String(val); }
  }
  if (Array.isArray(val)) {
    return val.map((item) => deepSanitize(item, objectKeys, depth + 1));
  }
  // Plain object — recurse into values
  const out = {};
  for (const [k, v] of Object.entries(val)) {
    if (v == null || typeof v !== 'object') {
      out[k] = v;
    } else if (Array.isArray(v)) {
      out[k] = v.map((item) => deepSanitize(item, objectKeys, depth + 1));
    } else if (objectKeys && objectKeys.has(k)) {
      // Preserve whitelisted object structure but still sanitize leaf values
      out[k] = deepSanitize(v, objectKeys, depth + 1);
    } else {
      // Recurse into nested objects
      out[k] = deepSanitize(v, objectKeys, depth + 1);
    }
  }
  return out;
}

// Keys whose object values are intentionally objects (not display text)
const PRESERVE_OBJECT_KEYS = new Set([
  'speakerColors', 'speaker_colors', 'words',
  'subtitle_settings', 'subtitle_style', 'position', 'size',
]);

/**
 * Sanitize job data from API to ensure no objects leak into React children.
 * React error #310 ("Objects are not valid as a React child") happens when
 * an object/array ends up inside JSX {} interpolation.
 *
 * Uses deep recursive sanitization as a catch-all, PLUS field-specific
 * coercions for known display fields that must be primitives.
 */
export default function sanitizeJob(job) {
  if (!job || typeof job !== 'object') return job;

  // Deep-sanitize the entire object tree first (catch-all)
  const safe = deepSanitize(job, PRESERVE_OBJECT_KEYS);

  // Scalar display fields — force to string/number/null
  const stringFields = ['filename', 'file_path', 'language', 'resolution', 'status',
    'progress_message', 'created_at', 'updated_at', 'analysis_started_at', 'error'];
  for (const f of stringFields) {
    if (safe[f] != null && typeof safe[f] === 'object') safe[f] = JSON.stringify(safe[f]);
  }

  // summary — ensure all leaves are strings
  if (safe.summary && typeof safe.summary === 'object') {
    const s = { ...safe.summary };
    for (const k of ['overview', 'tone', 'estimated_audience', 'content_category']) {
      if (s[k] != null && typeof s[k] !== 'string') s[k] = String(s[k]);
    }
    if (s.key_topics && Array.isArray(s.key_topics)) {
      s.key_topics = s.key_topics.map((t) => (typeof t === 'string' ? t : String(t)));
    }
    safe.summary = s;
  }

  // provider_used — ensure values are strings
  if (safe.provider_used && typeof safe.provider_used === 'object') {
    const pu = {};
    for (const [k, v] of Object.entries(safe.provider_used)) {
      pu[k] = typeof v === 'string' ? v : String(v);
    }
    safe.provider_used = pu;
  }

  // clips — sanitize display-text fields
  if (Array.isArray(safe.clips)) {
    safe.clips = safe.clips.map(sanitizeClip);
  }

  // transcript segments — ensure text and speaker are strings
  if (Array.isArray(safe.transcript)) {
    safe.transcript = safe.transcript.map((seg) => {
      if (!seg || typeof seg !== 'object') return seg;
      const ss = { ...seg };
      if (typeof ss.text !== 'string') ss.text = String(ss.text ?? '');
      if (typeof ss.speaker !== 'string') ss.speaker = String(ss.speaker ?? '');
      return ss;
    });
  }

  // translated_transcript — same sanitization as transcript
  if (safe.translated_transcript) {
    safe.translated_transcript = Array.isArray(safe.translated_transcript)
      ? safe.translated_transcript.map((seg) => {
          if (!seg || typeof seg !== 'object') return seg;
          const ss = { ...seg };
          if (typeof ss.text !== 'string') ss.text = String(ss.text ?? '');
          if (typeof ss.speaker !== 'string') ss.speaker = String(ss.speaker ?? '');
          return ss;
        })
      : [];
  }

  // scenes — ensure description is a string
  if (Array.isArray(safe.scenes)) {
    safe.scenes = safe.scenes.map((sc) => {
      if (!sc || typeof sc !== 'object') return sc;
      const s = { ...sc };
      if (typeof s.description !== 'string') s.description = String(s.description ?? '');
      return s;
    });
  }

  // exported_clips — ensure filename and other display fields are strings
  if (Array.isArray(safe.exported_clips)) {
    safe.exported_clips = safe.exported_clips.map((ec) => {
      if (!ec || typeof ec !== 'object') return ec;
      const s = { ...ec };
      if (s.filename != null && typeof s.filename !== 'string') s.filename = String(s.filename);
      if (s.title != null && typeof s.title !== 'string') s.title = String(s.title);
      if (s.export_quality != null && typeof s.export_quality !== 'string') s.export_quality = String(s.export_quality);
      if (s.exported_at != null && typeof s.exported_at !== 'string') s.exported_at = String(s.exported_at);
      return s;
    });
  }

  // speaker_names — ensure all values are strings
  if (safe.speaker_names && typeof safe.speaker_names === 'object') {
    const sn = {};
    for (const [k, v] of Object.entries(safe.speaker_names)) {
      sn[k] = typeof v === 'string' ? v : String(v);
    }
    safe.speaker_names = sn;
  }

  // subtitle_settings — ensure all scalar display fields are primitives
  // (objects like speakerColors are allowed to remain as objects)
  if (safe.subtitle_settings && typeof safe.subtitle_settings === 'object') {
    safe.subtitle_settings = sanitizeSubtitleSettings(safe.subtitle_settings);
  }

  return safe;
}

/**
 * Sanitize a single clip object to ensure all display fields are primitives.
 */
export function sanitizeClip(c) {
  if (!c || typeof c !== 'object') return c;
  const sc = { ...c };
  const textFields = ['title', 'viral_score_reasoning', 'clip_type', 'platform',
    'suggested_caption', 'hook_text', 'why_this_works', 'clip_focus',
    'seo_title', 'seo_description', 'seo_platform_tips',
    'shorts_description', 'longform_description'];
  for (const f of textFields) {
    if (sc[f] != null && typeof sc[f] !== 'string') sc[f] = String(sc[f]);
  }
  if (Array.isArray(sc.seo_tags)) {
    sc.seo_tags = sc.seo_tags.map((t) => (typeof t === 'string' ? t : String(t)));
  }
  // Numeric fields — coerce objects to number
  const numFields = ['viral_score', 'start_time', 'end_time', 'id'];
  for (const f of numFields) {
    if (sc[f] != null && typeof sc[f] === 'object') sc[f] = Number(sc[f]) || 0;
  }
  return sc;
}

/**
 * Sanitize subtitle settings to ensure no objects leak into JSX.
 * Fields like speakerColors are intentionally left as objects.
 */
export function sanitizeSubtitleSettings(ss) {
  if (!ss || typeof ss !== 'object' || Array.isArray(ss)) return {};
  const s = { ...ss };
  // All string fields that may be rendered in JSX
  const stringFields = [
    'subtitleFont', 'subtitleFontColor',
    'subtitleOutlineColor', 'subtitlePosition', 'activeWordColor',
    'activeWordOutlineColor', 'activeWordBgColor', 'subtitleBgColor',
    'aspectRatio', 'exportQuality',
    // snake_case variants from backend
    'font', 'font_color', 'outline_color', 'position',
    'active_word_color', 'active_word_outline_color', 'active_word_bg_color',
    'background_color',
  ];
  for (const f of stringFields) {
    if (s[f] != null && typeof s[f] === 'object') s[f] = String(s[f]);
  }
  // Numeric fields — coerce objects to number
  const numFields = [
    'subtitleSize', 'subtitleBgOpacity', 'subtitleBgRadius',
    'subtitleOutlineOpacity', 'subtitleOutlineWidth', 'subtitleMaxWidth',
    'subtitleOffsetV', 'subtitleMaxWords', 'activeWordBgOpacity', 'activeWordBgRadius',
    'playbackVolume', 'playbackSpeed',
    // snake_case variants
    'size', 'background_opacity', 'background_radius',
    'outline_opacity', 'outline_width', 'max_width', 'offset_v',
    'max_words', 'active_word_bg_opacity', 'active_word_bg_radius',
  ];
  for (const f of numFields) {
    if (s[f] != null && typeof s[f] === 'object') s[f] = Number(s[f]) || 0;
  }
  return s;
}

/**
 * Safe string helper for JSX rendering.
 * Converts any value to a string safe for React children.
 */
export function safeStr(val) {
  if (val == null) return '';
  if (typeof val === 'string') return val;
  if (typeof val === 'number' || typeof val === 'boolean') return String(val);
  return JSON.stringify(val);
}

/**
 * Wrap any value for safe JSX rendering.
 * Use in JSX: {safeRender(maybeObject)}
 */
export function safeRender(val) {
  if (val == null) return '';
  if (typeof val === 'string' || typeof val === 'number' || typeof val === 'boolean') return val;
  if (typeof val === 'object') {
    try { return JSON.stringify(val); } catch { return '[object]'; }
  }
  return String(val);
}
