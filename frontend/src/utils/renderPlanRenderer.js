/**
 * RenderPlanRenderer: Canvas-based preview renderer consuming the same
 * RenderPlan JSON that the FFmpeg filter builder uses.
 *
 * Ensures the preview shows exactly what the export will produce.
 * Each draw method mirrors the corresponding FFmpeg filter template.
 *
 * Blur params must match backend/services/ffmpeg_filter_builder.py:
 *   - blur sigma: 50px (CSS filter) ↔ gblur=sigma=50
 *   - brightness: 0.9 (CSS) ↔ eq=brightness=-0.1
 */

// Material Design standard easing for transitions
const EASE_CUBIC_BEZIER = [0.4, 0, 0.2, 1];

/**
 * Evaluate cubic-bezier easing at progress t (0-1).
 * Approximation using De Casteljau's algorithm.
 */
function cubicBezierEase(t) {
  // Control points: (0,0), (0.4, 0), (0.2, 1), (1, 1)
  const [x1, y1, x2, y2] = EASE_CUBIC_BEZIER;
  // Newton-Raphson to find t parameter for given x
  let guess = t;
  for (let i = 0; i < 8; i++) {
    const x = 3 * (1 - guess) * (1 - guess) * guess * x1 +
              3 * (1 - guess) * guess * guess * x2 +
              guess * guess * guess;
    const dx = 3 * (1 - guess) * (1 - guess) * x1 +
               6 * (1 - guess) * guess * (x2 - x1) +
               3 * guess * guess * (1 - x2);
    if (Math.abs(dx) < 1e-6) break;
    guess -= (x - t) / dx;
    guess = Math.max(0, Math.min(1, guess));
  }
  // Evaluate y at the found parameter
  return 3 * (1 - guess) * (1 - guess) * guess * y1 +
         3 * (1 - guess) * guess * guess * y2 +
         guess * guess * guess;
}

/**
 * Convert a normalized Rect (0-1) to pixel values with even-dimension enforcement.
 * Must match backend Rect.to_pixels() exactly.
 */
function rectToPixels(rect, sourceW, sourceH) {
  let pw = Math.round(rect.w * sourceW);
  let ph = Math.round(rect.h * sourceH);
  pw = pw - (pw % 2);
  ph = ph - (ph % 2);
  let px = Math.round(rect.x * sourceW);
  let py = Math.round(rect.y * sourceH);
  px = Math.max(0, Math.min(px, sourceW - pw));
  py = Math.max(0, Math.min(py, sourceH - ph));
  return { x: px, y: py, w: pw, h: ph };
}

/**
 * Linearly interpolate between two Rects at progress t (0-1).
 */
function lerpRect(a, b, t) {
  return {
    x: a.x + (b.x - a.x) * t,
    y: a.y + (b.y - a.y) * t,
    w: a.w + (b.w - a.w) * t,
    h: a.h + (b.h - a.h) * t,
  };
}


export class RenderPlanRenderer {
  /**
   * @param {HTMLCanvasElement} canvas - The output canvas element.
   * @param {HTMLVideoElement} videoElement - Hidden video source.
   * @param {Object} renderPlan - RenderPlan JSON from the backend.
   */
  constructor(canvas, videoElement, renderPlan) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.video = videoElement;
    this.plan = renderPlan;
    this.targetW = renderPlan.target_width;
    this.targetH = renderPlan.target_height;
    this.sourceW = renderPlan.source_width;
    this.sourceH = renderPlan.source_height;

    // Set canvas dimensions to match target output
    canvas.width = this.targetW;
    canvas.height = this.targetH;

    // Create offscreen canvas for blur effects
    this._offscreenCanvas = document.createElement('canvas');
    this._offscreenCanvas.width = this.targetW;
    this._offscreenCanvas.height = this.targetH;
    this._offscreenCtx = this._offscreenCanvas.getContext('2d');

    // Create second offscreen canvas for transitions
    this._transitionCanvas = document.createElement('canvas');
    this._transitionCanvas.width = this.targetW;
    this._transitionCanvas.height = this.targetH;
    this._transitionCtx = this._transitionCanvas.getContext('2d');
  }

  /**
   * Update the render plan (e.g., when switching clips).
   */
  updatePlan(renderPlan) {
    this.plan = renderPlan;
    this.targetW = renderPlan.target_width;
    this.targetH = renderPlan.target_height;
    this.sourceW = renderPlan.source_width;
    this.sourceH = renderPlan.source_height;
    this.canvas.width = this.targetW;
    this.canvas.height = this.targetH;
    this._offscreenCanvas.width = this.targetW;
    this._offscreenCanvas.height = this.targetH;
    this._transitionCanvas.width = this.targetW;
    this._transitionCanvas.height = this.targetH;
  }

  /**
   * Find the RenderOp active at the given time.
   * Uses binary search for efficiency.
   */
  findOpAt(timeSec) {
    const ops = this.plan.ops;
    if (!ops || ops.length === 0) return null;

    // Clamp time to plan bounds
    timeSec = Math.max(0, Math.min(timeSec, this.plan.total_duration_sec));

    let lo = 0, hi = ops.length - 1;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      if (ops[mid].end_sec <= timeSec) {
        lo = mid + 1;
      } else if (ops[mid].start_sec > timeSec) {
        hi = mid - 1;
      } else {
        return ops[mid];
      }
    }
    // Fallback: return last op
    return ops[ops.length - 1];
  }

  /**
   * Called on every requestAnimationFrame. Draws the current frame
   * according to the active RenderOp.
   */
  draw(currentTimeSec) {
    const op = this.findOpAt(currentTimeSec);
    if (!op) return;

    const ctx = this.ctx;

    // Phase 5: multi-region → single-crop fade-out has priority over
    // the standard ease_in_ms cross-fade.
    const idx = this.plan.ops.indexOf(op);
    if (idx > 0) {
      const prevOp = this.plan.ops[idx - 1];
      if (this._multiToSingleFadeOut(ctx, prevOp, op, currentTimeSec)) {
        return;
      }
    }

    // Check if we're in a transition zone
    const transition = this._getTransitionState(currentTimeSec);

    if (transition) {
      // Draw outgoing frame to transition canvas
      this._drawOp(this._transitionCtx, transition.outgoingOp, currentTimeSec);
      // Draw incoming frame to main canvas
      this._drawOp(ctx, transition.incomingOp, currentTimeSec);
      // Blend
      const alpha = cubicBezierEase(transition.progress);
      ctx.save();
      ctx.globalAlpha = 1 - alpha;
      ctx.drawImage(this._transitionCanvas, 0, 0);
      ctx.globalAlpha = alpha;
      // incoming is already drawn, just restore
      ctx.restore();
    } else {
      this._drawOp(ctx, op, currentTimeSec);
    }
  }

  _drawOp(ctx, op, currentTimeSec) {
    switch (op.kind) {
      case 'crop':
        this._drawCrop(ctx, op);
        break;
      case 'tracking_crop':
        this._drawTrackingCrop(ctx, op, currentTimeSec);
        break;
      case 'wide_master':
        this._drawWideMaster(ctx, op);
        break;
      case 'blur_fill':
        this._drawBlurFill(ctx, op);
        break;
      case 'contextual_pan':
        this._drawContextualPan(ctx, op, currentTimeSec);
        break;
      case 'split_screen':
        this._drawSplitScreen(ctx, op);
        break;
      case 'stacked_gameplay':
        this._drawStackedGameplay(ctx, op);
        break;
      case 'grid_2x2':
        this._drawGrid(ctx, op);
        break;
      case 'hud_composite':
        this._drawHudComposite(ctx, op);
        break;
      default:
        this._drawCrop(ctx, op);
    }
  }

  /**
   * Resolve the dynamic primary fraction (top region height as a 0-1
   * fraction of total output height). Falls back to `defaultFraction`
   * when `op.primary_fraction` is null/undefined. Mirrors the Python
   * `_resolve_primary_fraction` helper in ffmpeg_filter_builder.py.
   */
  _resolvePrimaryFraction(op, defaultFraction) {
    const pf = op && op.primary_fraction;
    if (pf == null) return defaultFraction;
    return Math.max(0.05, Math.min(0.95, Number(pf)));
  }

  /**
   * Phase 5: paint a 2px (or `op.separator_px`) dark line at the
   * boundary between the top and bottom regions of a multi-region
   * composite. No-op when separator_px <= 0. Color locked to #333333
   * to match the FFmpeg drawbox `color=0x333333`.
   */
  _drawSeparator(ctx, topH, op) {
    const sep = (op && op.separator_px) || 0;
    if (sep <= 0) return;
    ctx.save();
    ctx.fillStyle = '#333333';
    const y = Math.max(0, topH - Math.floor(sep / 2));
    ctx.fillRect(0, y, this.targetW, sep);
    ctx.restore();
  }

  /**
   * CROP: static rectangle crop.
   * Mirrors FFmpeg: crop=W:H:X:Y, scale=tgt_w:tgt_h
   */
  _drawCrop(ctx, op) {
    const { x, y, w, h } = rectToPixels(op.primary_rect, this.sourceW, this.sourceH);
    ctx.drawImage(this.video, x, y, w, h, 0, 0, this.targetW, this.targetH);
  }

  /**
   * TRACKING_CROP: animated crop via motion_path keypoints.
   * Linear interpolation between keypoints (matches FFmpeg piecewise-linear).
   */
  _drawTrackingCrop(ctx, op, currentTimeSec) {
    const tRel = currentTimeSec - op.start_sec;
    const path = op.motion_path;

    if (!path || path.length === 0) {
      this._drawCrop(ctx, op);
      return;
    }

    // Find the two surrounding keypoints
    let rect;
    if (tRel <= path[0].t) {
      rect = path[0].rect;
    } else if (tRel >= path[path.length - 1].t) {
      rect = path[path.length - 1].rect;
    } else {
      // Find segment
      for (let i = 0; i < path.length - 1; i++) {
        if (tRel >= path[i].t && tRel < path[i + 1].t) {
          const dt = path[i + 1].t - path[i].t;
          const progress = dt > 0 ? (tRel - path[i].t) / dt : 0;
          rect = lerpRect(path[i].rect, path[i + 1].rect, progress);
          break;
        }
      }
      if (!rect) rect = path[path.length - 1].rect;
    }

    const { x, y, w, h } = rectToPixels(rect, this.sourceW, this.sourceH);
    ctx.drawImage(this.video, x, y, w, h, 0, 0, this.targetW, this.targetH);
  }

  /**
   * CONTEXTUAL_PAN: time-interpolated lateral pan (Ken Burns).
   *
   * Linear lerp between the first and last motion_path keypoints'
   * crop x. Crop dimensions are constant; x is clamped so the
   * window NEVER goes off-source (no black bars at the edges).
   * Mirrors backend ffmpeg_filter_builder._filter_contextual_pan.
   */
  _drawContextualPan(ctx, op, currentTimeSec) {
    const path = op.motion_path;
    if (!path || path.length < 2) {
      this._drawCrop(ctx, op);
      return;
    }
    const tRel = currentTimeSec - op.start_sec;
    const kp0 = path[0];
    const kp1 = path[path.length - 1];
    const dt = Math.max(1e-3, kp1.t - kp0.t);
    let progress = (tRel - kp0.t) / dt;
    if (progress < 0) progress = 0;
    if (progress > 1) progress = 1;
    const rect = lerpRect(kp0.rect, kp1.rect, progress);
    const { x, y, w, h } = rectToPixels(rect, this.sourceW, this.sourceH);
    ctx.drawImage(this.video, x, y, w, h, 0, 0, this.targetW, this.targetH);
  }

  /**
   * WIDE_MASTER: full source frame preserved with a centered "letterbox-
   * like" sharp content area, but the surrounding fill is a BLURRED,
   * cover-fit duplicate of the source — NEVER solid black bars.
   *
   * Phase 4 of the reframing overhaul redefined this op's semantics:
   * black-bar letterboxing was eliminated. Implementation mirrors
   * _drawBlurFill (the two ops are functionally identical at the
   * renderer level).
   *
   * Mirrors FFmpeg: split + gblur(sigma=50) + overlay (see
   * backend/services/ffmpeg_filter_builder._filter_wide_master).
   */
  _drawWideMaster(ctx, op) {
    this._drawBlurFill(ctx, op);
  }

  /**
   * BLUR_FILL: source centered, blurred duplicate as background.
   * Mirrors FFmpeg: gblur=sigma=50, eq=brightness=-0.1
   *
   * CSS blur(50px) ↔ FFmpeg gblur=sigma=50
   */
  _drawBlurFill(ctx, op) {
    const offCtx = this._offscreenCtx;

    // Draw blurred background
    offCtx.save();
    offCtx.filter = `blur(50px) brightness(0.9)`;
    // Scale source to cover entire target (force fill)
    const scaleX = this.targetW / this.sourceW;
    const scaleY = this.targetH / this.sourceH;
    const scale = Math.max(scaleX, scaleY);
    const dw = this.sourceW * scale;
    const dh = this.sourceH * scale;
    const dx = (this.targetW - dw) / 2;
    const dy = (this.targetH - dh) / 2;
    offCtx.drawImage(this.video, 0, 0, this.sourceW, this.sourceH, dx, dy, dw, dh);
    offCtx.restore();

    // Draw blurred background to main canvas
    ctx.drawImage(this._offscreenCanvas, 0, 0);

    // Draw foreground centered (scaled to target width, maintaining aspect)
    const fgH = Math.round(this.sourceH * (this.targetW / this.sourceW));
    const fgY = Math.round((this.targetH - fgH) / 2);
    ctx.drawImage(this.video, 0, 0, this.sourceW, this.sourceH,
                  0, fgY, this.targetW, fgH);
  }

  /**
   * SPLIT_SCREEN: two crops stacked vertically. Default 50/50;
   * `op.primary_fraction` overrides the top-region share for Phase-5
   * dynamic-region behavior. Draws a 2px separator at the seam when
   * `op.separator_px > 0`.
   * Mirrors FFmpeg: crop + scale + vstack + drawbox
   */
  _drawSplitScreen(ctx, op) {
    const frac = this._resolvePrimaryFraction(op, 0.5);
    let topH = Math.floor(this.targetH * frac);
    topH = Math.max(2, Math.min(topH, this.targetH - 2));

    // Top half
    const top = rectToPixels(op.primary_rect, this.sourceW, this.sourceH);
    ctx.drawImage(this.video, top.x, top.y, top.w, top.h,
                  0, 0, this.targetW, topH);

    // Bottom half
    const bot = rectToPixels(op.secondary_rect, this.sourceW, this.sourceH);
    ctx.drawImage(this.video, bot.x, bot.y, bot.w, bot.h,
                  0, topH, this.targetW, this.targetH - topH);

    this._drawSeparator(ctx, topH, op);
  }

  /**
   * STACKED_GAMEPLAY: gameplay on top, facecam on bottom. Default
   * 60/40; `op.primary_fraction` overrides for Phase-5 dynamic-region
   * behavior. Draws a 2px separator at the seam when
   * `op.separator_px > 0`.
   * Mirrors FFmpeg: 60/40 split with vstack + drawbox.
   */
  _drawStackedGameplay(ctx, op) {
    const frac = this._resolvePrimaryFraction(op, 0.6);
    let topH = Math.floor(this.targetH * frac);
    topH = Math.max(2, Math.min(topH, this.targetH - 2));
    const botH = this.targetH - topH;

    // Gameplay (top)
    const game = rectToPixels(op.primary_rect, this.sourceW, this.sourceH);
    ctx.drawImage(this.video, game.x, game.y, game.w, game.h,
                  0, 0, this.targetW, topH);

    // Facecam (bottom)
    const cam = rectToPixels(op.secondary_rect, this.sourceW, this.sourceH);
    ctx.drawImage(this.video, cam.x, cam.y, cam.w, cam.h,
                  0, topH, this.targetW, botH);

    this._drawSeparator(ctx, topH, op);
  }

  /**
   * HUD_COMPOSITE: gameplay viewport on top + horizontal HUD strip on
   * bottom (Phase 5). The viewport occupies (1 - hud_strip_fraction)
   * of the height (default 75%) and the HUD strip the remainder. Each
   * `hud_strip_rects[i]` source rect is scaled into a horizontal slot
   * inside the strip. When no HUD rects are provided, the strip falls
   * back to a blurred-cover duplicate of the source — never a black
   * bar (Phase 4 contract).
   * Mirrors FFmpeg: split + crop + scale + xstack + vstack.
   */
  _drawHudComposite(ctx, op) {
    const stripFrac = Math.max(0.10, Math.min(0.50,
      Number(op.hud_strip_fraction != null ? op.hud_strip_fraction : 0.25)));
    let stripH = Math.floor(this.targetH * stripFrac);
    stripH = Math.max(2, Math.min(stripH, this.targetH - 2));
    const viewH = this.targetH - stripH;

    // Viewport (top)
    const view = rectToPixels(op.primary_rect, this.sourceW, this.sourceH);
    ctx.drawImage(this.video, view.x, view.y, view.w, view.h,
                  0, 0, this.targetW, viewH);

    // HUD strip (bottom)
    const hudRects = (op.hud_strip_rects || []);
    if (hudRects.length === 0) {
      // Blurred-cover fallback (no black bar).
      const offCtx = this._offscreenCtx;
      offCtx.save();
      offCtx.clearRect(0, 0, this.targetW, this.targetH);
      offCtx.filter = 'blur(50px) brightness(0.9)';
      const scaleX = this.targetW / this.sourceW;
      const scaleY = stripH / this.sourceH;
      const scale = Math.max(scaleX, scaleY);
      const dw = this.sourceW * scale;
      const dh = this.sourceH * scale;
      const dx = (this.targetW - dw) / 2;
      const dy = viewH + (stripH - dh) / 2;
      offCtx.drawImage(this.video, 0, 0, this.sourceW, this.sourceH, dx, dy, dw, dh);
      offCtx.restore();
      ctx.drawImage(this._offscreenCanvas, 0, viewH, this.targetW, stripH,
                    0, viewH, this.targetW, stripH);
    } else {
      const n = hudRects.length;
      let baseW = Math.floor(this.targetW / n);
      baseW = baseW - (baseW % 2);
      let lastW = this.targetW - baseW * (n - 1);
      lastW = lastW - (lastW % 2);
      let xCursor = 0;
      for (let i = 0; i < n; i++) {
        const slotW = (i === n - 1) ? lastW : baseW;
        const src = rectToPixels(hudRects[i], this.sourceW, this.sourceH);
        ctx.drawImage(this.video, src.x, src.y, src.w, src.h,
                      xCursor, viewH, slotW, stripH);
        xCursor += slotW;
      }
    }

    this._drawSeparator(ctx, viewH, op);
  }

  /**
   * Phase 5: when transitioning FROM a multi-region op TO a single-crop
   * op AND the prior op carries `transition_fade_out_ms`, animate the
   * region heights so the secondary region + separator fade out smoothly
   * (alpha ramp) over that window. Returns true when the caller should
   * skip its own draw — `_drawOp` calls this from the transition path.
   */
  _multiToSingleFadeOut(ctx, prevOp, currOp, currentTimeSec) {
    const fadeMs = (prevOp && prevOp.transition_fade_out_ms) || 0;
    if (fadeMs <= 0) return false;
    if (!RenderPlanRenderer._isMultiRegion(prevOp)) return false;
    if (!RenderPlanRenderer._isSingleCrop(currOp)) return false;

    const fadeSec = fadeMs / 1000;
    const boundary = currOp.start_sec;
    const fadeStart = boundary - fadeSec / 2;
    const fadeEnd = boundary + fadeSec / 2;
    if (currentTimeSec < fadeStart || currentTimeSec > fadeEnd) return false;

    const progress = Math.max(0, Math.min(1,
      (currentTimeSec - fadeStart) / fadeSec));

    // Draw multi-region (prevOp) into transition canvas.
    this._drawOp(this._transitionCtx, prevOp, currentTimeSec);
    // Draw single-crop (currOp) into main canvas.
    this._drawOp(ctx, currOp, currentTimeSec);
    // Blend: prev fades out, curr fades in.
    ctx.save();
    ctx.globalAlpha = 1 - progress;
    ctx.drawImage(this._transitionCanvas, 0, 0);
    ctx.restore();
    return true;
  }

  static _isMultiRegion(op) {
    if (!op) return false;
    return ['split_screen', 'stacked_gameplay', 'hud_composite', 'grid_2x2']
      .indexOf(op.kind) >= 0;
  }

  static _isSingleCrop(op) {
    if (!op) return false;
    return [
      'crop', 'tracking_crop', 'contextual_pan',
      'motivated_push_in', 'motivated_pull_out',
      'wide_master', 'blur_fill',
    ].indexOf(op.kind) >= 0;
  }

  /**
   * GRID_2X2: 4 tiles in 2x2 layout.
   * Mirrors FFmpeg: xstack with 4 inputs
   */
  _drawGrid(ctx, op) {
    const tileW = Math.floor(this.targetW / 2);
    const tileH = Math.floor(this.targetH / 2);

    const rects = [
      op.primary_rect,
      op.secondary_rect,
      op.tertiary_rect,
      op.quaternary_rect,
    ];

    const positions = [
      [0, 0],           // top-left
      [tileW, 0],       // top-right
      [0, tileH],       // bottom-left
      [tileW, tileH],   // bottom-right
    ];

    for (let i = 0; i < 4; i++) {
      if (!rects[i]) continue;
      const src = rectToPixels(rects[i], this.sourceW, this.sourceH);
      ctx.drawImage(this.video, src.x, src.y, src.w, src.h,
                    positions[i][0], positions[i][1], tileW, tileH);
    }
  }

  /**
   * Check if we're currently in a transition zone between two ops.
   * Returns null if no active transition, or an object with transition info.
   */
  _getTransitionState(currentTimeSec) {
    const ops = this.plan.ops;
    if (!ops || ops.length < 2) return null;

    for (let i = 1; i < ops.length; i++) {
      const easeMs = ops[i].ease_in_ms;
      if (easeMs <= 0) continue;

      const easeSec = easeMs / 1000;
      const boundary = ops[i].start_sec;
      const transStart = boundary - easeSec / 2;
      const transEnd = boundary + easeSec / 2;

      if (currentTimeSec >= transStart && currentTimeSec <= transEnd) {
        const progress = (currentTimeSec - transStart) / easeSec;
        return {
          outgoingOp: ops[i - 1],
          incomingOp: ops[i],
          progress: Math.max(0, Math.min(1, progress)),
        };
      }
    }

    return null;
  }

  /**
   * Clean up resources.
   */
  dispose() {
    this._offscreenCanvas = null;
    this._offscreenCtx = null;
    this._transitionCanvas = null;
    this._transitionCtx = null;
  }
}

export default RenderPlanRenderer;
