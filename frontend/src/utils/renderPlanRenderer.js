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
      case 'split_screen':
        this._drawSplitScreen(ctx, op);
        break;
      case 'stacked_gameplay':
        this._drawStackedGameplay(ctx, op);
        break;
      case 'grid_2x2':
        this._drawGrid(ctx, op);
        break;
      default:
        this._drawCrop(ctx, op);
    }
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
   * WIDE_MASTER: letterbox with black bars.
   * Mirrors FFmpeg: scale=tgt_w:-1, pad=tgt_w:tgt_h:0:(oh-ih)/2:black
   */
  _drawWideMaster(ctx, op) {
    // Fill with black
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, this.targetW, this.targetH);

    // Scale source to target width, maintain aspect ratio
    const scaledH = Math.round(this.sourceH * (this.targetW / this.sourceW));
    const yOffset = Math.round((this.targetH - scaledH) / 2);

    ctx.drawImage(this.video, 0, 0, this.sourceW, this.sourceH,
                  0, yOffset, this.targetW, scaledH);
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
   * SPLIT_SCREEN: two crops stacked vertically, 50/50.
   * Mirrors FFmpeg: crop + scale + vstack
   */
  _drawSplitScreen(ctx, op) {
    const halfH = Math.floor(this.targetH / 2);

    // Top half
    const top = rectToPixels(op.primary_rect, this.sourceW, this.sourceH);
    ctx.drawImage(this.video, top.x, top.y, top.w, top.h,
                  0, 0, this.targetW, halfH);

    // Bottom half
    const bot = rectToPixels(op.secondary_rect, this.sourceW, this.sourceH);
    ctx.drawImage(this.video, bot.x, bot.y, bot.w, bot.h,
                  0, halfH, this.targetW, this.targetH - halfH);
  }

  /**
   * STACKED_GAMEPLAY: gameplay top 60%, facecam bottom 40%.
   * Mirrors FFmpeg: 60/40 split with vstack
   */
  _drawStackedGameplay(ctx, op) {
    const topH = Math.round(this.targetH * 0.6);
    const botH = this.targetH - topH;

    // Gameplay (top 60%)
    const game = rectToPixels(op.primary_rect, this.sourceW, this.sourceH);
    ctx.drawImage(this.video, game.x, game.y, game.w, game.h,
                  0, 0, this.targetW, topH);

    // Facecam (bottom 40%)
    const cam = rectToPixels(op.secondary_rect, this.sourceW, this.sourceH);
    ctx.drawImage(this.video, cam.x, cam.y, cam.w, cam.h,
                  0, topH, this.targetW, botH);
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
