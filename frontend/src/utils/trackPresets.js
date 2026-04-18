/**
 * Track Presets — localStorage-based preset system for timeline item properties.
 *
 * Each preset is scoped to a track/item type (video, audio, text, shape,
 * subtitle, image, overlay). Presets saved for one type cannot be applied
 * to a different type, enforced both at save and load time.
 *
 * Storage key: 'clipai_track_presets'
 * Format: Array of { id, name, type, data, createdAt }
 */

const STORAGE_KEY = 'clipai_track_presets';

// Which properties to capture per item type.
// We exclude timing (start/end), id, trackId, mediaRef, and content fields
// that are instance-specific. We capture style/layout/effects properties
// that a user would want to reuse across items.
const CAPTURE_FIELDS = {
  video: [
    'position', 'size', 'transform', 'effects', 'opacity',
    'fadeIn', 'fadeOut', 'volume', 'speed', 'transition',
    'subjectX',
  ],
  audio: [
    'volume', 'speed', 'fadeIn', 'fadeOut',
  ],
  text: [
    'position', 'size', 'transform', 'effects', 'opacity',
    'fadeIn', 'fadeOut', 'textStyle',
  ],
  shape: [
    'position', 'size', 'transform', 'effects', 'opacity',
    'fadeIn', 'fadeOut', 'shapeType', 'shapeStyle',
  ],
  subtitle: [
    'position', 'transform', 'opacity', 'fadeIn', 'fadeOut',
  ],
  image: [
    'position', 'size', 'transform', 'effects', 'opacity',
    'fadeIn', 'fadeOut', 'transition',
  ],
  overlay: [
    'position', 'size', 'transform', 'effects', 'opacity',
    'fadeIn', 'fadeOut', 'transition',
  ],
};

// Also capture subtitle settings (global) when saving a subtitle preset
const SUBTITLE_SETTINGS_FIELDS = [
  'subtitleFont', 'subtitleSize', 'subtitleFontWeight', 'subtitleFontColor',
  'subtitlePosition', 'subtitleMaxWidth', 'subtitleOffsetV', 'subtitleMaxWords',
  'subtitleOutlineColor', 'subtitleOutlineOpacity', 'subtitleOutlineWidth',
  'subtitleBgEnabled', 'subtitleBgColor', 'subtitleBgOpacity',
  'useSpeakerColors', 'showSpeakerLabels',
  'activeWordEnabled', 'activeWordColor', 'activeWordOutlineColor',
  'activeWordBgColor', 'activeWordBgOpacity', 'activeWordBgRadius',
];

/**
 * Friendly display names for track types.
 */
export const TYPE_LABELS = {
  video: 'Video',
  audio: 'Audio',
  text: 'Text',
  shape: 'Shape',
  subtitle: 'Subtitle',
  image: 'Image',
  overlay: 'Overlay',
};

/**
 * Load all presets from localStorage.
 * @returns {Array} Array of preset objects
 */
export function loadAllPresets() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

/**
 * Load presets filtered by item type.
 * @param {string} type - Item type (video, audio, text, etc.)
 * @returns {Array} Array of presets matching the type
 */
export function loadPresetsForType(type) {
  return loadAllPresets().filter(p => p.type === type);
}

/**
 * Save a preset from a timeline item.
 * @param {string} name - User-chosen name for the preset
 * @param {string} type - Item type
 * @param {Object} item - The timeline item to capture properties from
 * @param {Object} [subtitleSettings] - Global subtitle settings (for subtitle presets)
 * @returns {Object} The saved preset object
 */
export function savePreset(name, type, item, subtitleSettings = null) {
  const fields = CAPTURE_FIELDS[type];
  if (!fields) throw new Error(`Unknown item type: ${type}`);

  const data = {};
  for (const field of fields) {
    if (item[field] !== undefined && item[field] !== null) {
      // Deep clone to avoid reference issues
      data[field] = JSON.parse(JSON.stringify(item[field]));
    }
  }

  // For subtitle presets, also capture the global subtitle styling settings
  if (type === 'subtitle' && subtitleSettings) {
    data._subtitleSettings = {};
    for (const key of SUBTITLE_SETTINGS_FIELDS) {
      if (subtitleSettings[key] !== undefined) {
        data._subtitleSettings[key] = subtitleSettings[key];
      }
    }
  }

  const preset = {
    id: `tp-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
    name: name.trim(),
    type,
    data,
    createdAt: new Date().toISOString(),
  };

  const presets = loadAllPresets();
  presets.push(preset);
  localStorage.setItem(STORAGE_KEY, JSON.stringify(presets));

  return preset;
}

/**
 * Apply a preset to a timeline item.
 * Returns the update object to pass to updateItem().
 * Throws if the preset type doesn't match the item type.
 *
 * @param {Object} preset - The preset to apply
 * @param {string} itemType - The item's type
 * @returns {{ itemUpdates: Object, subtitleSettings: Object|null }}
 */
export function applyPreset(preset, itemType) {
  if (preset.type !== itemType) {
    throw new Error(
      `Cannot apply ${TYPE_LABELS[preset.type] || preset.type} preset ` +
      `to ${TYPE_LABELS[itemType] || itemType} item`
    );
  }

  const fields = CAPTURE_FIELDS[itemType];
  if (!fields) throw new Error(`Unknown item type: ${itemType}`);

  const itemUpdates = {};
  for (const field of fields) {
    if (preset.data[field] !== undefined) {
      itemUpdates[field] = JSON.parse(JSON.stringify(preset.data[field]));
    }
  }

  // For subtitle presets, also return the global subtitle settings
  let subtitleSettings = null;
  if (itemType === 'subtitle' && preset.data._subtitleSettings) {
    subtitleSettings = { ...preset.data._subtitleSettings };
  }

  return { itemUpdates, subtitleSettings };
}

/**
 * Delete a preset by ID.
 * @param {string} presetId - The preset ID to delete
 * @returns {boolean} Whether the preset was found and deleted
 */
export function deletePreset(presetId) {
  const presets = loadAllPresets();
  const filtered = presets.filter(p => p.id !== presetId);
  if (filtered.length === presets.length) return false;
  localStorage.setItem(STORAGE_KEY, JSON.stringify(filtered));
  return true;
}

/**
 * Rename a preset.
 * @param {string} presetId - The preset ID
 * @param {string} newName - The new name
 * @returns {boolean} Whether the preset was found and renamed
 */
export function renamePreset(presetId, newName) {
  const presets = loadAllPresets();
  const preset = presets.find(p => p.id === presetId);
  if (!preset) return false;
  preset.name = newName.trim();
  localStorage.setItem(STORAGE_KEY, JSON.stringify(presets));
  return true;
}

/**
 * Delete all presets for a specific type.
 * @param {string} type - Item type
 * @returns {number} Number of presets deleted
 */
export function deletePresetsForType(type) {
  const presets = loadAllPresets();
  const filtered = presets.filter(p => p.type !== type);
  const count = presets.length - filtered.length;
  localStorage.setItem(STORAGE_KEY, JSON.stringify(filtered));
  return count;
}
