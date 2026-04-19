const JUNK_PREFIX = [
  /^[\{\[]/,
  /^['"]?(raw_response|error|description|importance_score|subject_box|traceback)['"]?\s*[:=]/i,
  /^(Error|Traceback|Exception|Failed to|RuntimeError|undefined|null|NaN|<html|<!DOCTYPE)\b/i,
];

const REFUSAL = [
  /^I (can'?t|cannot|am unable|'m unable|am not able|don'?t see)\b/i,
  /^(As an AI|Sorry,|I'm sorry|Unfortunately, I)/i,
  /^\[?image (unavailable|not (loaded|provided|visible))\]?/i,
];

const SYNTHETIC_PREFIX_OK = [/^key moment /i, /^video frame at /i];

export function cleanDescription(s) {
  if (!s || typeof s !== 'string') return '';
  // eslint-disable-next-line no-control-regex
  return s.replace(/[\u0000-\u001F]/g, ' ').replace(/\s+/g, ' ').trim();
}

export function looksLikeJunk(s) {
  if (!s || typeof s !== 'string') return true;
  const t = s.trim();
  if (!t) return true;
  if (SYNTHETIC_PREFIX_OK.some(rx => rx.test(t))) return false;
  if (JUNK_PREFIX.some(rx => rx.test(t))) return true;
  if (REFUSAL.some(rx => rx.test(t))) return true;
  if (t.length < 10) return true;
  if (t.split(/\s+/).length < 3) return true;
  const nonAlpha = (t.match(/[^\w\s.,;:!?'\-\u2026\u2013\u2014]/g) || []).length;
  if (nonAlpha / t.length > 0.15) return true;
  return false;
}

function formatTime(seconds) {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60).toString().padStart(2, '0');
  return `${m}:${s}`;
}

export function renderDescription(desc, timestamp, index) {
  const cleaned = cleanDescription(desc);
  if (looksLikeJunk(cleaned)) {
    return `Key moment ${index + 1} at ${formatTime(timestamp)}`;
  }
  return cleaned;
}
