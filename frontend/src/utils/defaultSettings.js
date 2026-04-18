/**
 * Shared default clip settings — SINGLE SOURCE OF TRUTH.
 * Used by ClipSettingsPanel, Analysis, ViralClips, and anywhere
 * that needs default subtitle/clip settings.
 */
export const DEFAULT_CLIP_SETTINGS = {
  clipCount: 12,
  minDuration: 15,
  maxDuration: 600,
  aspectRatio: null,
  subtitlesEnabled: true,  // Subtitles ON by default when transcript exists
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
  exportQuality: '1080p',
  playbackVolume: 100,
  playbackSpeed: 1.0,
};

/**
 * Clip-generation panel defaults (Analysis page "Generate Clips" controls).
 * Persisted in localStorage under GEN_STORAGE_KEY by Analysis.jsx. Bug 6
 * in the clip-focus audit: keep minDuration/maxDuration matching
 * DEFAULT_CLIP_SETTINGS above so first-run users see one canonical value.
 */
export const DEFAULT_GEN_SETTINGS = {
  clipCount: 12,
  minDuration: 15,
  maxDuration: 600,
  viralScoreMin: 0,
  viralScoreMax: 100,
  // Focus mode — persisted so a refresh doesn't wipe the user's query
  clipFocusEnabled: false,
  clipFocusText: '',
  minRelevance: 50,
  // Focus query history — last 10 unique queries, MRU order
  focusHistory: [],
};

/**
 * Centralized control ranges so ClipSettingsPanel, PropertiesPanel,
 * EffectsPanel, ClipPreview, ExportDialog and TransitionPicker all
 * agree on the slider / input bounds. Adding a new control here
 * means every panel picks it up — eliminates the "panels disagree"
 * bug class flagged in the editor audit.
 */
export const SUBTITLE_RANGES = {
  // Numeric font size (pt). The S/M/L preset chips below map to
  // discrete values in this range.
  size: { min: 12, max: 72, step: 1 },
  // Discrete preset chips paired with the numeric size field.
  sizePreset: ['S', 'M', 'L'],
  fontWeight: { min: 100, max: 900, step: 100 },
  // Vertical offset is "% of frame height", same scale across panels.
  offsetV: { min: 0, max: 100, step: 1 },
  // Max words / line. When 0 we fall back to natural segmentation.
  maxWords: { min: 0, max: 30, step: 1 },
  maxWidth: { min: 30, max: 100, step: 1 },
  bgOpacity: { min: 0, max: 100, step: 1 },
  bgRadius: { min: 0, max: 24, step: 1 },
  outlineWidth: { min: 0, max: 8, step: 1 },
  outlineOpacity: { min: 0, max: 100, step: 1 },
};

/** Playback speed options shared by preview chrome + clip settings. */
export const SPEED_OPTIONS = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 3.0, 4.0];

/** Transition picker / properties-panel duration bounds. */
export const TRANSITION_RANGES = {
  duration: { min: 0.1, max: 3.0, step: 0.05 },
  // Quick-pick chips used by TransitionPicker.
  durationPresets: [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0],
};

/** Volume slider — 0..1 internally, 0..100 in any UI label. */
export const VOLUME_RANGE = { min: 0, max: 1, step: 0.01 };

/** Export quality presets — keep the dialog and the panel in sync. */
export const EXPORT_QUALITIES = ['720p', '1080p', '4k'];

/**
 * Single-source vertical-position alias table.
 *
 * The legacy data model has TWO fields with overlapping intent:
 *   * ``subtitlePosition`` — categorical: "top" | "center" | "bottom".
 *   * ``subtitleOffsetV``  — numeric 0–100, distance from bottom.
 *
 * To stop the panels from disagreeing (top pill highlighted while
 * preview shows bottom, etc.), every UI computes one from the other
 * via these helpers. Storage carries both for backward-compat with
 * existing job JSONs.
 */
export const SUBTITLE_POSITION_TO_OFFSET = {
  top: 80,        // 80% from bottom = top of frame
  center: 50,
  bottom: 4,      // matches DEFAULT_CLIP_SETTINGS.subtitleOffsetV
};

export function offsetVFromPosition(position) {
  if (position == null) return SUBTITLE_POSITION_TO_OFFSET.bottom;
  return SUBTITLE_POSITION_TO_OFFSET[position] ?? SUBTITLE_POSITION_TO_OFFSET.bottom;
}

export function positionFromOffsetV(offsetV) {
  // Buckets are tuned so a user typing 50% lands on "center" but a
  // value of 4–10 stays "bottom" (matches the default).
  if (offsetV == null) return 'bottom';
  if (offsetV >= 65) return 'top';
  if (offsetV >= 35) return 'center';
  return 'bottom';
}
