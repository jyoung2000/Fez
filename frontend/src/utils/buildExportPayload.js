/**
 * Shared export payload builder — single source of truth for all export codepaths.
 *
 * Used by ExportDialog, Analysis, and ViralClips to ensure consistent:
 * - Overlay item filtering (primary audio excluded)
 * - Blob URL / empty src validation
 * - Timing conversion (clip-relative → absolute)
 * - Settings mapping (camelCase → snake_case)
 */

/**
 * Build overlay arrays for the export request payload.
 *
 * @param {Object} params
 * @param {Array}  params.timelineItems  - Items from useTimelineStore
 * @param {Array}  params.mediaLibrary   - Media library from useTimelineStore
 * @param {number} params.clipStart      - Clip start time in source video (absolute)
 * @param {Array}  [params.tracks]       - Tracks array for position-based compositing order
 * @returns {{ textOverlays: Array, imageOverlays: Array, shapeOverlays: Array, audioOverlays: Array, warnings: string[] }}
 */
export function buildOverlayPayload({ timelineItems, mediaLibrary, clipStart, tracks }) {
  const warnings = [];

  // Track-position sorting: items on higher tracks (lower index) should appear
  // later in the array so they render on top in FFmpeg filter chains.
  const getTrackOrder = (item) => {
    if (!tracks) return 0;
    const idx = tracks.findIndex(t => t.id === item.trackId);
    return idx >= 0 ? tracks.length - idx : 0;
  };

  // ── Text overlays ──
  const textItems = timelineItems.filter(it => it.type === 'text')
    .sort((a, b) => getTrackOrder(a) - getTrackOrder(b));
  const textOverlays = textItems.map(it => ({
    item_id: it.id,
    text: it.textContent || '',
    x: it.position?.x ?? 50,
    y: it.position?.y ?? 50,
    width: it.size?.w ?? 80,
    height: it.size?.h ?? 20,
    font_size: Math.round(it.textStyle?.fontSize || 48),
    font_color: it.textStyle?.color || '#FFFFFF',
    font_family: it.textStyle?.fontFamily || 'sans-serif',
    font_weight: Math.round(it.textStyle?.fontWeight || 400),
    background_color: it.textStyle?.bgColor || null,
    outline_width: Math.round(it.textStyle?.outlineWidth || 0),
    outline_color: it.textStyle?.outlineColor || '#000000',
    start_time: (it.start || 0) + clipStart,
    end_time: (it.end || 0) + clipStart,
    rotation: it.transform?.rotation || 0,
    opacity: it.opacity ?? 1,
    fade_in: it.fadeIn || 0,
    fade_out: it.fadeOut || 0,
    animation: it.textStyle?.animation || 'none',
    text_align: it.textStyle?.textAlign || 'center',
    shadow_blur: it.textStyle?.shadowBlur || 0,
    shadow_color: it.textStyle?.shadowColor || 'rgba(0,0,0,0.5)',
    shadow_offset_x: it.textStyle?.shadowOffsetX || 0,
    shadow_offset_y: it.textStyle?.shadowOffsetY || 0,
    bg_opacity: it.textStyle?.bgOpacity ?? 0,
    bg_padding: it.textStyle?.bgPadding ?? 8,
    bg_radius: it.textStyle?.bgRadius ?? 4,
  }));

  // ── Image overlays ──
  const imageItems = timelineItems.filter(it => it.type === 'image' || it.type === 'overlay')
    .sort((a, b) => getTrackOrder(a) - getTrackOrder(b));
  const imageOverlays = [];
  for (const it of imageItems) {
    let src = it.src || '';
    if (!src && it.mediaRef) {
      const mediaEntry = mediaLibrary.find(m => m.id === it.mediaRef);
      src = mediaEntry?.url || '';
    }
    if (!src || src.startsWith('blob:')) {
      warnings.push(`Image overlay "${it.id || 'unknown'}" skipped — source not uploaded`);
      continue;
    }
    const fx = it.effects || {};
    imageOverlays.push({
      item_id: it.id,
      src,
      x: it.position?.x ?? 50,
      y: it.position?.y ?? 50,
      width: it.size?.w ?? 30,
      height: it.size?.h ?? 30,
      start_time: (it.start || 0) + clipStart,
      end_time: (it.end || 0) + clipStart,
      opacity: it.opacity ?? 1,
      fade_in: it.fadeIn || 0,
      fade_out: it.fadeOut || 0,
      rotation: it.transform?.rotation || 0,
      brightness: fx.brightness || 0,
      contrast: fx.contrast || 0,
      saturation: fx.saturation || 0,
      blur: fx.blur || 0,
      hue_rotate: fx.hueRotate || 0,
      sepia: fx.sepia || 0,
    });
  }

  // ── Shape overlays ──
  const shapeItems = timelineItems.filter(it => it.type === 'shape')
    .sort((a, b) => getTrackOrder(a) - getTrackOrder(b));
  const shapeOverlays = shapeItems.map(it => ({
    item_id: it.id,
    shape_type: it.shapeType || 'rectangle',
    x: it.position?.x ?? 50,
    y: it.position?.y ?? 50,
    width: it.size?.w ?? 20,
    height: it.size?.h ?? 20,
    fill_color: it.shapeStyle?.fillColor || '#FF3B30',
    stroke_color: it.shapeStyle?.strokeColor || '#FFFFFF',
    stroke_width: it.shapeStyle?.strokeWidth || 2,
    corner_radius: it.shapeStyle?.cornerRadius || 0,
    start_time: (it.start || 0) + clipStart,
    end_time: (it.end || 0) + clipStart,
    rotation: it.transform?.rotation || 0,
    opacity: it.opacity ?? 1,
    fade_in: it.fadeIn || 0,
    fade_out: it.fadeOut || 0,
  }));

  // ── Audio overlays ──
  // CRITICAL: Exclude the primary audio track (a1) which shares mediaRef
  // with the video item. Including it would either fail resolution or
  // cause doubled/phased audio in the export.
  const videoItem = timelineItems.find(it => it.type === 'video');
  const primaryMediaRef = videoItem?.mediaRef;
  const audioItems = timelineItems.filter(it =>
    it.type === 'audio' &&
    it.trackId !== 'a1' &&
    it.mediaRef !== primaryMediaRef
  );
  const audioOverlays = [];
  for (const it of audioItems) {
    let src = it.src || '';
    if (!src && it.mediaRef) {
      const mediaEntry = mediaLibrary.find(m => m.id === it.mediaRef);
      src = mediaEntry?.url || '';
    }
    if (!src || src.startsWith('blob:')) {
      warnings.push(`Audio overlay "${it.id || 'unknown'}" skipped — source not uploaded`);
      continue;
    }
    // Look up the parent track so we can honor a track-level mute.
    // Without ``audio_muted`` the export silently mixed in muted
    // audio overlays (the backend had no concept of track mute on
    // the AudioOverlay model).
    const trackForItem = tracks
      ? tracks.find(t => t.id === it.trackId)
      : null;
    const trackAudioMuted = trackForItem
      ? !!(trackForItem.audioMuted !== undefined
          ? trackForItem.audioMuted
          : trackForItem.muted)
      : false;
    audioOverlays.push({
      src,
      start_time: (it.start || 0) + clipStart,
      end_time: (it.end || 0) + clipStart,
      volume: it.volume ?? 1,
      speed: it.speed ?? 1,
      fade_in: it.fadeIn || 0,
      fade_out: it.fadeOut || 0,
      audio_muted: trackAudioMuted,
    });
  }

  // Build a global compositing order that interleaves ALL visual overlay types
  // by their track position. The backend uses this to determine the correct
  // render order in the FFmpeg filter chain.
  // Drop items whose parent track is hidden in the editor (eye-icon
  // off → ``videoVisible === false`` or legacy ``visible === false``)
  // — those are gone from the live preview, so they should be gone
  // from the export too.
  const allVisualItems = timelineItems.filter((it) => {
    const isVisualType = it.type === 'text' || it.type === 'shape'
      || it.type === 'image' || it.type === 'overlay';
    if (!isVisualType) return false;
    const tr = tracks ? tracks.find((t) => t.id === it.trackId) : null;
    if (tr) {
      const visible = tr.videoVisible !== undefined
        ? tr.videoVisible
        : (tr.visible !== false);
      if (!visible) return false;
    }
    return true;
  });
  const compositingOrder = allVisualItems
    .map(it => {
      const trackIdx = tracks ? tracks.findIndex(t => t.id === it.trackId) : -1;
      return {
        type: it.type === 'overlay' ? 'image' : it.type,
        id: it.id,
        track_index: trackIdx,
        // Lower compositing_priority = renders first (below); higher = renders later (on top)
        compositing_priority: trackIdx >= 0 ? tracks.length - trackIdx : 0,
      };
    })
    .sort((a, b) => a.compositing_priority - b.compositing_priority);

  return { textOverlays, imageOverlays, shapeOverlays, audioOverlays, warnings, compositingOrder };
}

/**
 * Build video_effects payload from the video timeline item.
 *
 * @param {Array} timelineItems
 * @returns {Object|null} video_effects object or null if no effects
 */
export function buildVideoEffectsPayload(timelineItems) {
  const videoItem = timelineItems.find(it => it.type === 'video');
  if (!videoItem) return null;

  const fx = videoItem.effects || {};
  const pos = videoItem.position || {};
  const sz = videoItem.size || {};
  const rot = videoItem.transform?.rotation || 0;
  const fadeIn = videoItem.fadeIn || 0;
  const fadeOut = videoItem.fadeOut || 0;

  const hasEffects = (fx.brightness || 0) !== 0 || (fx.contrast || 0) !== 0 ||
    (fx.saturation || 0) !== 0 || (fx.blur || 0) > 0 ||
    (fx.hueRotate || 0) > 0 || (fx.sepia || 0) > 0 ||
    (videoItem.opacity ?? 1) < 1;
  const hasTransform = (pos.x != null && pos.x !== 50) || (pos.y != null && pos.y !== 50) ||
    (sz.w != null && sz.w !== 100) || (sz.h != null && sz.h !== 100) ||
    rot !== 0 || fadeIn > 0 || fadeOut > 0;

  if (!hasEffects && !hasTransform) return null;

  return {
    brightness: fx.brightness || 0,
    contrast: fx.contrast || 0,
    saturation: fx.saturation || 0,
    blur: fx.blur || 0,
    hue_rotate: fx.hueRotate || 0,
    sepia: fx.sepia || 0,
    opacity: videoItem.opacity ?? 1,
    position_x: pos.x ?? 50,
    position_y: pos.y ?? 50,
    width: sz.w ?? 100,
    height: sz.h ?? 100,
    rotation: rot,
    fade_in: fadeIn,
    fade_out: fadeOut,
  };
}

/**
 * Map camelCase clipSettings to snake_case subtitle_settings for the backend.
 *
 * @param {Object} cs - clipSettings object
 * @returns {Object} subtitle_settings for ExportRequest
 */
export function mapSubtitleSettings(cs) {
  if (!cs) return null;
  return {
    font: cs.subtitleFont || 'DM Sans',
    size: cs.subtitleSize ?? 30,
    font_weight: typeof cs.subtitleFontWeight === 'number' ? cs.subtitleFontWeight : (cs.subtitleFontWeight || 'bold'),
    font_color: cs.subtitleFontColor || '#FFFFFF',
    position: cs.subtitlePosition || 'bottom',
    speaker_colors: cs.speakerColors || {},
    use_speaker_colors: cs.useSpeakerColors ?? true,
    background_enabled: cs.subtitleBgEnabled ?? false,
    background_color: cs.subtitleBgColor || '#000000',
    background_opacity: cs.subtitleBgOpacity ?? 75,
    background_radius: cs.subtitleBgRadius ?? 0,
    outline_color: cs.subtitleOutlineColor || '#000000',
    outline_opacity: cs.subtitleOutlineOpacity ?? 100,
    outline_width: cs.subtitleOutlineWidth ?? 2,
    show_speaker_labels: cs.showSpeakerLabels ?? false,
    max_width: cs.subtitleMaxWidth ?? 90,
    offset_v: cs.subtitleOffsetV ?? 4,
    max_words: cs.subtitleMaxWords ?? 0,
    active_word_enabled: cs.activeWordEnabled ?? false,
    active_word_color: cs.activeWordColor || '#FFD700',
    active_word_outline_color: cs.activeWordOutlineColor || '#000000',
    active_word_bg_color: cs.activeWordBgColor || '#000000',
    active_word_bg_opacity: cs.activeWordBgOpacity ?? 0,
    active_word_bg_radius: cs.activeWordBgRadius ?? 4,
  };
}
