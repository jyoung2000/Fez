import React, { useMemo } from 'react';
import useTimelineStore from '../stores/timelineStore';
import { hexToRgba } from '../utils/colorUtils';

/**
 * TimelineOverlay renders text, shape, and image items from the multi-track
 * timeline as absolutely-positioned DOM elements over the video viewport.
 *
 * Props:
 *   currentTime  – playhead position (seconds, relative to clip start)
 *   clipStart    – absolute start of the clip/full video
 *   playing      – whether the video is currently playing
 */
export default function TimelineOverlay({ currentTime = 0, clipStart = 0 }) {
  const items = useTimelineStore((s) => s.items);
  const tracks = useTimelineStore((s) => s.tracks);

  // currentTime is already relative to clipStart (passed as currentTime - clipStart)
  const absTime = currentTime;

  // Filter to visible text/shape/image/overlay items at current time.
  // Respects track visibility — items on hidden tracks are not rendered in preview.
  const visible = useMemo(() => {
    return items
      .filter((it) => {
        if (it.type === 'video' || it.type === 'audio' || it.type === 'subtitle') return false;
        if (!(absTime >= it.start && absTime < it.end)) return false;
        // Check track visibility
        const track = tracks.find((t) => t.id === it.trackId);
        if (track && track.visible === false) return false;
        return true;
      })
      // Sort by track position: items on higher tracks (lower index) render later (on top)
      .sort((a, b) => {
        const idxA = tracks.findIndex((t) => t.id === a.trackId);
        const idxB = tracks.findIndex((t) => t.id === b.trackId);
        return idxB - idxA;
      });
  }, [items, absTime, tracks]);

  if (visible.length === 0) return null;

  return (
    <div
      style={{
        position: 'absolute',
        inset: 0,
        pointerEvents: 'none',
        zIndex: 6,
        overflow: 'hidden',
      }}
    >
      {visible.map((item) => {
        const elapsed = absTime - item.start;
        const duration = item.end - item.start;

        if (item.type === 'text') return <TextOverlayItem key={item.id} item={item} elapsed={elapsed} duration={duration} />;
        if (item.type === 'shape') return <ShapeOverlayItem key={item.id} item={item} elapsed={elapsed} duration={duration} />;
        if (item.type === 'image' || item.type === 'overlay') return <ImageOverlayItem key={item.id} item={item} elapsed={elapsed} duration={duration} />;
        return null;
      })}
    </div>
  );
}


function TextOverlayItem({ item, elapsed, duration }) {
  const rawPos = item.position || { x: 50, y: 50 };
  const pos = (rawPos.x === 0 && rawPos.y === 0) ? { x: 50, y: 50 } : rawPos;
  const size = item.size || { w: 80, h: 20 };
  const style = item.textStyle || {};
  const text = item.textContent || '';
  const rotation = item.transform?.rotation || 0;

  // Animation
  let animAlpha = 1;
  let animTransform = '';
  const anim = style.animation || 'none';
  if (anim === 'fade-in' && elapsed < 0.5) {
    animAlpha = elapsed / 0.5;
  } else if (anim === 'slide-up' && elapsed < 0.5) {
    const t = elapsed / 0.5;
    animAlpha = t;
    animTransform = `translateY(${(1 - t) * 30}px)`;
  } else if (anim === 'pop' && elapsed < 0.3) {
    const t = elapsed / 0.3;
    const scale = 0.5 + 0.5 * (t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2);
    animTransform = `scale(${scale})`;
  } else if (anim === 'bounce' && elapsed < 0.6) {
    const t = elapsed / 0.6;
    // Bounce easing: overshoot then settle
    const bounce = t < 0.4
      ? (t / 0.4) * 1.2
      : t < 0.7
        ? 1.2 - (t - 0.4) / 0.3 * 0.3
        : t < 0.85
          ? 0.9 + (t - 0.7) / 0.15 * 0.1
          : 1.0;
    animAlpha = Math.min(1, t * 2);
    animTransform = `scale(${bounce}) translateY(${(1 - Math.min(1, t * 2)) * -20}px)`;
  }

  // Typewriter
  let displayText = text;
  if (anim === 'typewriter' && duration > 0) {
    const chars = Math.floor((elapsed / duration) * text.length);
    displayText = text.slice(0, chars);
  }

  // Fade in/out
  let fadeAlpha = 1;
  if (item.fadeIn > 0 && elapsed < item.fadeIn) fadeAlpha *= elapsed / item.fadeIn;
  if (item.fadeOut > 0 && (duration - elapsed) < item.fadeOut) fadeAlpha *= (duration - elapsed) / item.fadeOut;

  const finalOpacity = (item.opacity ?? 1) * animAlpha * fadeAlpha;

  // Build CSS filter from effects
  const effects = item.effects || {};
  const filters = [];
  if (effects.brightness) filters.push(`brightness(${1 + effects.brightness / 100})`);
  if (effects.contrast) filters.push(`contrast(${1 + effects.contrast / 100})`);
  if (effects.saturation) filters.push(`saturate(${1 + effects.saturation / 100})`);
  if (effects.blur) filters.push(`blur(${effects.blur}px)`);
  if (effects.hueRotate) filters.push(`hue-rotate(${effects.hueRotate}deg)`);
  if (effects.sepia) filters.push(`sepia(${effects.sepia / 100})`);

  // Text shadow
  const shadows = [];
  if (style.shadowBlur > 0 || style.shadowOffsetX || style.shadowOffsetY) {
    shadows.push(`${style.shadowOffsetX || 0}px ${style.shadowOffsetY || 0}px ${style.shadowBlur || 0}px ${style.shadowColor || 'rgba(0,0,0,0.5)'}`);
  }

  // Outline via text-stroke + paint-order
  const outlineW = style.outlineWidth || 0;
  const outlineC = style.outlineColor || '#000000';

  return (
    <div
      style={{
        position: 'absolute',
        left: `${pos.x}%`,
        top: `${pos.y}%`,
        width: `${size.w}%`,
        height: `${size.h}%`,
        transform: `translate(-50%, -50%) ${rotation ? `rotate(${rotation}deg)` : ''} ${animTransform}`,
        opacity: finalOpacity,
        filter: filters.length ? filters.join(' ') : undefined,
        overflow: 'hidden',
        display: 'flex',
        alignItems: 'center',
        justifyContent:
          style.textAlign === 'left' ? 'flex-start'
          : style.textAlign === 'right' ? 'flex-end'
          : 'center',
        pointerEvents: 'none',
        transition: 'opacity 0.05s linear',
      }}
    >
      {/* Background box */}
      {style.bgColor && style.bgOpacity > 0 && (
        <div
          style={{
            position: 'absolute',
            inset: `-${style.bgPadding || 8}px`,
            background: hexToRgba(style.bgColor, (style.bgOpacity || 75) / 100),
            borderRadius: style.bgRadius || 4,
            zIndex: -1,
          }}
        />
      )}
      <span
        style={{
          fontSize: `clamp(10px, ${(style.fontSize || 48) / 12}vw, ${style.fontSize || 48}px)`,
          fontFamily: `"${style.fontFamily || 'DM Sans'}", sans-serif`,
          fontWeight: style.fontWeight || 400,
          color: style.color || '#FFFFFF',
          textAlign: style.textAlign || 'center',
          WebkitTextStroke: outlineW > 0 ? `${outlineW}px ${outlineC}` : undefined,
          paintOrder: outlineW > 0 ? 'stroke fill' : undefined,
          textShadow: shadows.length ? shadows.join(', ') : undefined,
          lineHeight: 1.3,
          wordBreak: 'break-word',
          whiteSpace: 'pre-wrap',
        }}
      >
        {displayText}
      </span>
    </div>
  );
}


function ShapeOverlayItem({ item, elapsed = 0, duration = 1 }) {
  const rawPos = item.position || { x: 10, y: 10 };
  const pos = (rawPos.x === 0 && rawPos.y === 0) ? { x: 10, y: 10 } : rawPos;
  const size = item.size || { w: 20, h: 20 };
  const style = item.shapeStyle || {};
  const shapeType = item.shapeType || 'rectangle';
  const rotation = item.transform?.rotation || 0;

  // Fade in/out
  let fadeAlpha = 1;
  if (item.fadeIn > 0 && elapsed < item.fadeIn) fadeAlpha *= elapsed / item.fadeIn;
  if (item.fadeOut > 0 && (duration - elapsed) < item.fadeOut) fadeAlpha *= (duration - elapsed) / item.fadeOut;

  const finalOpacity = (item.opacity ?? 1) * fadeAlpha;

  // Effects
  const effects = item.effects || {};
  const filters = [];
  if (effects.brightness) filters.push(`brightness(${1 + effects.brightness / 100})`);
  if (effects.contrast) filters.push(`contrast(${1 + effects.contrast / 100})`);
  if (effects.saturation) filters.push(`saturate(${1 + effects.saturation / 100})`);
  if (effects.blur) filters.push(`blur(${effects.blur}px)`);
  if (effects.hueRotate) filters.push(`hue-rotate(${effects.hueRotate}deg)`);
  if (effects.sepia) filters.push(`sepia(${effects.sepia / 100})`);

  const common = {
    position: 'absolute',
    left: `${pos.x}%`,
    top: `${pos.y}%`,
    width: `${size.w}%`,
    height: `${size.h}%`,
    transform: `translate(-50%, -50%) ${rotation ? `rotate(${rotation}deg)` : ''}`,
    opacity: finalOpacity,
    filter: filters.length ? filters.join(' ') : undefined,
    pointerEvents: 'none',
  };

  if (shapeType === 'rectangle') {
    return (
      <div
        style={{
          ...common,
          background: style.fillColor || '#FF3B30',
          border: style.strokeWidth > 0 ? `${style.strokeWidth || 2}px solid ${style.strokeColor || '#FFFFFF'}` : 'none',
          borderRadius: style.cornerRadius || 0,
        }}
      />
    );
  }

  if (shapeType === 'circle' || shapeType === 'ellipse') {
    return (
      <div
        style={{
          ...common,
          background: style.fillColor || '#FF3B30',
          border: style.strokeWidth > 0 ? `${style.strokeWidth || 2}px solid ${style.strokeColor || '#FFFFFF'}` : 'none',
          borderRadius: '50%',
        }}
      />
    );
  }

  // For arrow and line, use inline SVG
  if (shapeType === 'line') {
    return (
      <svg
        style={{ ...common, overflow: 'visible' }}
        viewBox="0 0 100 100"
        preserveAspectRatio="none"
      >
        <line
          x1="0" y1="0" x2="100" y2="100"
          stroke={style.strokeColor || '#FFFFFF'}
          strokeWidth={style.strokeWidth || 2}
        />
      </svg>
    );
  }

  if (shapeType === 'arrow') {
    return (
      <svg
        style={{ ...common, overflow: 'visible' }}
        viewBox="0 0 100 100"
        preserveAspectRatio="none"
      >
        <polygon
          points="0,50 70,50 70,0 100,50 70,100 70,50"
          fill={style.fillColor || '#FF3B30'}
          stroke={style.strokeColor || '#FFFFFF'}
          strokeWidth={style.strokeWidth || 2}
        />
      </svg>
    );
  }

  return null;
}


function ImageOverlayItem({ item, elapsed = 0, duration = 1 }) {
  const rawPos = item.position || { x: 50, y: 50 };
  const pos = (rawPos.x === 0 && rawPos.y === 0) ? { x: 50, y: 50 } : rawPos;
  const size = item.size || { w: 30, h: 30 };
  const rotation = item.transform?.rotation || 0;
  const mediaLibrary = useTimelineStore((s) => s.mediaLibrary);

  // Fade in/out
  let fadeAlpha = 1;
  if (item.fadeIn > 0 && elapsed < item.fadeIn) fadeAlpha *= elapsed / item.fadeIn;
  if (item.fadeOut > 0 && (duration - elapsed) < item.fadeOut) fadeAlpha *= (duration - elapsed) / item.fadeOut;

  const finalOpacity = (item.opacity ?? 1) * fadeAlpha;

  const effects = item.effects || {};
  const filters = [];
  if (effects.brightness) filters.push(`brightness(${1 + effects.brightness / 100})`);
  if (effects.contrast) filters.push(`contrast(${1 + effects.contrast / 100})`);
  if (effects.saturation) filters.push(`saturate(${1 + effects.saturation / 100})`);
  if (effects.blur) filters.push(`blur(${effects.blur}px)`);
  if (effects.hueRotate) filters.push(`hue-rotate(${effects.hueRotate}deg)`);
  if (effects.sepia) filters.push(`sepia(${effects.sepia / 100})`);

  let mediaSrc = item.mediaSrc || '';
  if (!mediaSrc && item.mediaRef) {
    const mediaEntry = mediaLibrary.find((m) => m.id === item.mediaRef);
    mediaSrc = mediaEntry?.url || mediaEntry?.thumbnailUrl || item.mediaRef;
  }

  if (!mediaSrc) {
    return (
      <div
        style={{
          position: 'absolute',
          left: `${pos.x}%`,
          top: `${pos.y}%`,
          width: `${size.w}%`,
          height: `${size.h}%`,
          transform: `translate(-50%, -50%) ${rotation ? `rotate(${rotation}deg)` : ''}`,
          opacity: finalOpacity,
          background: 'rgba(128,128,128,0.3)',
          border: '1px dashed rgba(255,255,255,0.3)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          fontSize: 10,
          color: 'rgba(255,255,255,0.5)',
          pointerEvents: 'none',
        }}
      >
        Image
      </div>
    );
  }

  return (
    <img
      src={mediaSrc}
      alt=""
      style={{
        position: 'absolute',
        left: `${pos.x}%`,
        top: `${pos.y}%`,
        width: `${size.w}%`,
        height: `${size.h}%`,
        objectFit: 'contain',
        transform: `translate(-50%, -50%) ${rotation ? `rotate(${rotation}deg)` : ''}`,
        opacity: finalOpacity,
        filter: filters.length ? filters.join(' ') : undefined,
        pointerEvents: 'none',
      }}
    />
  );
}
