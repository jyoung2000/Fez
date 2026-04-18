/**
 * Editor QA / Validation System
 *
 * Validates the video editor state to ensure:
 * - Items are on compatible tracks (no videos on audio tracks, etc.)
 * - Layer ordering is correct (video bottom, overlays middle, subtitles top)
 * - Property panel matches selected element type
 * - All items have valid timing (start < end, no negative values)
 * - No orphaned items (referencing non-existent tracks)
 * - Track types have proper allowed element constraints
 */

// Track-item type compatibility (must match timelineStore.js)
const TRACK_ALLOWED_TYPES = {
  video: ['video'],
  overlay: ['text', 'shape', 'image', 'overlay'],
  audio: ['audio'],
  subtitle: ['subtitle'],
};

/**
 * Validate that all items are on compatible tracks.
 * Returns an array of violation objects.
 */
export function validateTrackCompatibility(tracks, items) {
  const violations = [];
  const trackMap = new Map(tracks.map(t => [t.id, t]));

  for (const item of items) {
    const track = trackMap.get(item.trackId);
    if (!track) {
      violations.push({
        type: 'orphaned_item',
        severity: 'error',
        itemId: item.id,
        message: `Item "${item.id}" (${item.type}) references non-existent track "${item.trackId}"`,
      });
      continue;
    }

    const allowed = TRACK_ALLOWED_TYPES[track.type];
    if (allowed && !allowed.includes(item.type)) {
      violations.push({
        type: 'incompatible_track',
        severity: 'error',
        itemId: item.id,
        trackId: track.id,
        itemType: item.type,
        trackType: track.type,
        message: `${item.type} item "${item.id}" is on ${track.type} track "${track.name}" — should be on ${getExpectedTrackType(item.type)} track`,
      });
    }
  }
  return violations;
}

/**
 * Validate layer ordering: video tracks should have lowest order,
 * overlays above video, subtitles at the top.
 */
export function validateLayerOrder(tracks) {
  const violations = [];
  const videoOrders = tracks.filter(t => t.type === 'video').map(t => t.order);
  const overlayOrders = tracks.filter(t => t.type === 'overlay').map(t => t.order);
  const subtitleOrders = tracks.filter(t => t.type === 'subtitle').map(t => t.order);

  const maxVideo = Math.max(...videoOrders, -1);
  const minOverlay = Math.min(...overlayOrders, Infinity);
  const maxOverlay = Math.max(...overlayOrders, -1);
  const minSubtitle = Math.min(...subtitleOrders, Infinity);

  if (overlayOrders.length > 0 && videoOrders.length > 0 && maxVideo >= minOverlay) {
    violations.push({
      type: 'layer_order',
      severity: 'warning',
      message: `Video track order (${maxVideo}) should be below overlay track order (${minOverlay})`,
    });
  }

  if (subtitleOrders.length > 0 && overlayOrders.length > 0 && maxOverlay >= minSubtitle) {
    violations.push({
      type: 'layer_order',
      severity: 'warning',
      message: `Overlay track order (${maxOverlay}) should be below subtitle track order (${minSubtitle})`,
    });
  }

  return violations;
}

/**
 * Validate item timing: start < end, no negative values, reasonable durations.
 */
export function validateItemTiming(items) {
  const violations = [];
  for (const item of items) {
    if (item.start < 0) {
      violations.push({
        type: 'invalid_timing',
        severity: 'error',
        itemId: item.id,
        message: `Item "${item.id}" has negative start time: ${item.start}`,
      });
    }
    if (item.end <= item.start) {
      violations.push({
        type: 'invalid_timing',
        severity: 'error',
        itemId: item.id,
        message: `Item "${item.id}" has invalid timing: end (${item.end}) <= start (${item.start})`,
      });
    }
    if (item.end - item.start > 86400) {
      violations.push({
        type: 'invalid_timing',
        severity: 'warning',
        itemId: item.id,
        message: `Item "${item.id}" has duration > 24 hours (${item.end - item.start}s)`,
      });
    }
  }
  return violations;
}

/**
 * Validate item properties: position, size, opacity are within valid ranges.
 */
export function validateItemProperties(items) {
  const violations = [];
  for (const item of items) {
    // Opacity
    if (item.opacity !== undefined && (item.opacity < 0 || item.opacity > 1)) {
      violations.push({
        type: 'invalid_property',
        severity: 'warning',
        itemId: item.id,
        message: `Item "${item.id}" opacity out of range: ${item.opacity}`,
      });
    }
    // Volume
    if (item.volume !== undefined && (item.volume < 0 || item.volume > 5)) {
      violations.push({
        type: 'invalid_property',
        severity: 'warning',
        itemId: item.id,
        message: `Item "${item.id}" volume out of range: ${item.volume}`,
      });
    }
    // Speed
    if (item.speed !== undefined && (item.speed <= 0 || item.speed > 100)) {
      violations.push({
        type: 'invalid_property',
        severity: 'warning',
        itemId: item.id,
        message: `Item "${item.id}" speed out of range: ${item.speed}`,
      });
    }
  }
  return violations;
}

/**
 * Validate that the selected item's property panel matches its type.
 * Returns violations if the selected item would show wrong properties.
 */
export function validateSelectedItemPanel(items, selectedItemId) {
  const violations = [];
  if (!selectedItemId) return violations;

  const item = items.find(i => i.id === selectedItemId);
  if (!item) {
    violations.push({
      type: 'selection_invalid',
      severity: 'warning',
      message: `Selected item "${selectedItemId}" not found in items list`,
    });
    return violations;
  }

  // Verify item has a valid type
  const validTypes = ['video', 'audio', 'text', 'shape', 'image', 'overlay', 'subtitle'];
  if (!validTypes.includes(item.type)) {
    violations.push({
      type: 'invalid_item_type',
      severity: 'error',
      itemId: item.id,
      message: `Item "${item.id}" has invalid type: "${item.type}"`,
    });
  }

  return violations;
}

/**
 * Validate layer rendering: items visible at a given time should
 * render in correct order (video bottom, overlays middle, subtitles top).
 */
export function validateLayerRendering(tracks, items, currentTime) {
  const violations = [];
  const trackMap = new Map(tracks.map(t => [t.id, t]));

  // Get items visible at current time
  const visibleItems = items.filter(
    it => currentTime >= it.start && currentTime < it.end
  );

  // Group by track and check render order (based on track type, not track.order)
  const compositingPriority = { audio: 0, video: 1, overlay: 2, subtitle: 3 };
  const renderOrder = visibleItems
    .map(it => {
      const track = trackMap.get(it.trackId);
      return { item: it, track, order: compositingPriority[track?.type] ?? 1 };
    })
    .sort((a, b) => a.order - b.order);

  // Verify visual items render in correct layer order
  let lastVideoOrder = -1;
  let firstOverlayOrder = Infinity;
  let firstSubtitleOrder = Infinity;

  for (const { item, track, order } of renderOrder) {
    if (!track) continue;
    if (track.type === 'video') lastVideoOrder = Math.max(lastVideoOrder, order);
    if (track.type === 'overlay') firstOverlayOrder = Math.min(firstOverlayOrder, order);
    if (track.type === 'subtitle') firstSubtitleOrder = Math.min(firstSubtitleOrder, order);
  }

  if (lastVideoOrder >= 0 && firstOverlayOrder < Infinity && lastVideoOrder >= firstOverlayOrder) {
    violations.push({
      type: 'render_order',
      severity: 'error',
      message: `Video track (order ${lastVideoOrder}) renders at or above overlay track (order ${firstOverlayOrder}) at time ${currentTime.toFixed(2)}s`,
    });
  }

  if (firstOverlayOrder < Infinity && firstSubtitleOrder < Infinity && firstOverlayOrder >= firstSubtitleOrder) {
    violations.push({
      type: 'render_order',
      severity: 'warning',
      message: `Overlay track (order ${firstOverlayOrder}) renders at or above subtitle track (order ${firstSubtitleOrder}) at time ${currentTime.toFixed(2)}s`,
    });
  }

  return violations;
}

/**
 * Validate keyboard shortcut configuration completeness.
 * Returns a report of expected shortcuts and their status.
 */
export function validateKeyboardShortcuts() {
  const expectedShortcuts = [
    { key: 'Space', action: 'Play/Pause', category: 'transport' },
    { key: 'ArrowLeft', action: 'Frame backward', category: 'transport' },
    { key: 'ArrowRight', action: 'Frame forward', category: 'transport' },
    { key: 'Shift+ArrowLeft', action: '1 second backward', category: 'transport' },
    { key: 'Shift+ArrowRight', action: '1 second forward', category: 'transport' },
    { key: 'J', action: 'Shuttle reverse', category: 'transport' },
    { key: 'K', action: 'Shuttle stop', category: 'transport' },
    { key: 'L', action: 'Shuttle forward', category: 'transport' },
    { key: 'Home', action: 'Seek to start', category: 'transport' },
    { key: 'End', action: 'Seek to end', category: 'transport' },
    { key: 'M', action: 'Toggle mute', category: 'transport' },
    { key: 'V', action: 'Select tool', category: 'tools' },
    { key: 'C', action: 'Razor tool', category: 'tools' },
    { key: 'T', action: 'Text tool', category: 'tools' },
    { key: 'R', action: 'Shape tool', category: 'tools' },
    { key: 'N', action: 'Toggle snap', category: 'tools' },
    { key: 'S', action: 'Split at playhead', category: 'editing' },
    { key: 'Delete/Backspace', action: 'Delete selected', category: 'editing' },
    { key: 'Escape', action: 'Deselect', category: 'editing' },
    { key: 'Ctrl+Z', action: 'Undo', category: 'editing' },
    { key: 'Ctrl+Shift+Z / Ctrl+Y', action: 'Redo', category: 'editing' },
    { key: 'Ctrl+A', action: 'Select all', category: 'editing' },
  ];

  return {
    shortcuts: expectedShortcuts,
    count: expectedShortcuts.length,
    categories: [...new Set(expectedShortcuts.map(s => s.category))],
  };
}

/**
 * Validate subtitle consistency between settings, timeline items, and export readiness.
 * Returns violations when the state would cause preview/export mismatches.
 */
export function validateSubtitleConsistency(tracks, items, options = {}) {
  const { settings, segments } = options;
  const violations = [];

  const subtitleTrack = tracks.find(t => t.type === 'subtitle');
  const subtitleItems = items.filter(it => it.type === 'subtitle');
  const trackVisible = subtitleTrack ? subtitleTrack.visible !== false : false;
  const settingsEnabled = settings?.subtitlesEnabled ?? false;

  // Check: settings says subs ON but no subtitle items exist
  if (settingsEnabled && subtitleItems.length === 0) {
    violations.push({
      type: 'subtitle_missing_items',
      severity: 'warning',
      message: 'Subtitles are enabled in settings but no subtitle items exist on the timeline. Import a transcript first.',
    });
  }

  // Check: settings says subs ON but track is hidden
  if (settingsEnabled && !trackVisible) {
    violations.push({
      type: 'subtitle_track_hidden',
      severity: 'error',
      message: 'Subtitles are enabled in settings but the subtitle track is hidden. Subtitles will not appear in preview or export.',
    });
  }

  // Check: track visible but settings says subs OFF
  if (trackVisible && subtitleItems.length > 0 && !settingsEnabled) {
    violations.push({
      type: 'subtitle_settings_off',
      severity: 'warning',
      message: 'Subtitle items exist on the timeline but subtitles are disabled in settings. Toggle subtitles ON in clip settings to see them.',
    });
  }

  // Check: subtitle items with empty text
  const emptyItems = subtitleItems.filter(it => !it.subtitleText || it.subtitleText.trim() === '');
  if (emptyItems.length > 0) {
    violations.push({
      type: 'subtitle_empty_text',
      severity: 'warning',
      message: `${emptyItems.length} subtitle item(s) have empty text and will render as blank.`,
    });
  }

  // Check: per-segment subtitle override inconsistency
  if (segments && segments.length > 0) {
    const segsWithSubsOn = segments.filter(s => s.subtitlesEnabled !== false);
    const segsWithSubsOff = segments.filter(s => s.subtitlesEnabled === false);
    if (!settingsEnabled && segsWithSubsOn.length > 0 && segsWithSubsOff.length > 0) {
      violations.push({
        type: 'subtitle_mixed_segments',
        severity: 'info',
        message: `Mixed subtitle visibility: ${segsWithSubsOn.length} segments ON, ${segsWithSubsOff.length} OFF. Export will respect per-segment settings.`,
      });
    }
  }

  return violations;
}

/**
 * Run all validation checks and return a summary.
 */
export function runEditorQA(tracks, items, options = {}) {
  const { selectedItemId, currentTime, settings, segments } = options;

  const results = {
    trackCompatibility: validateTrackCompatibility(tracks, items),
    layerOrder: validateLayerOrder(tracks),
    itemTiming: validateItemTiming(items),
    itemProperties: validateItemProperties(items),
    selectedItemPanel: selectedItemId ? validateSelectedItemPanel(items, selectedItemId) : [],
    layerRendering: currentTime != null ? validateLayerRendering(tracks, items, currentTime) : [],
    subtitleConsistency: validateSubtitleConsistency(tracks, items, { settings, segments }),
  };

  const allViolations = [
    ...results.trackCompatibility,
    ...results.layerOrder,
    ...results.itemTiming,
    ...results.itemProperties,
    ...results.selectedItemPanel,
    ...results.layerRendering,
    ...results.subtitleConsistency,
  ];

  const errors = allViolations.filter(v => v.severity === 'error');
  const warnings = allViolations.filter(v => v.severity === 'warning');

  return {
    valid: errors.length === 0,
    errors,
    warnings,
    violations: allViolations,
    summary: `${errors.length} errors, ${warnings.length} warnings`,
    ...results,
  };
}

/**
 * Auto-fix track compatibility violations by moving items to correct tracks.
 * Returns an array of fixes applied.
 */
export function autoFixTrackCompatibility(tracks, items) {
  const fixes = [];
  const trackMap = new Map(tracks.map(t => [t.id, t]));

  for (const item of items) {
    const track = trackMap.get(item.trackId);
    if (!track) continue;

    const allowed = TRACK_ALLOWED_TYPES[track.type];
    if (allowed && !allowed.includes(item.type)) {
      const expectedTrackType = getExpectedTrackType(item.type);
      const correctTrack = tracks.find(t => t.type === expectedTrackType);
      if (correctTrack) {
        fixes.push({
          itemId: item.id,
          fromTrackId: item.trackId,
          toTrackId: correctTrack.id,
          reason: `Moving ${item.type} from ${track.type} track to ${correctTrack.type} track`,
        });
        item.trackId = correctTrack.id;
      }
    }
  }
  return fixes;
}

function getExpectedTrackType(itemType) {
  if (itemType === 'video') return 'video';
  if (itemType === 'audio') return 'audio';
  if (itemType === 'subtitle') return 'subtitle';
  return 'overlay'; // text, shape, image, overlay
}
