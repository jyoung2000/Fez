import React, { useState, useRef, useCallback, useMemo, useEffect } from 'react';
import useTimelineStore from '../stores/timelineStore';

/**
 * InteractiveOverlay renders selectable, draggable, resizable, and rotatable
 * handles over overlay items (text, shape, image) in the video preview viewport.
 *
 * Props:
 *   currentTime  – playhead position (seconds, relative to clip start)
 *   clipStart    – absolute start of the clip
 *   containerRef – ref to the viewport container div
 *   onInteraction – callback(bool) to pause/resume play-on-click behavior
 */
export default function InteractiveOverlay({ currentTime = 0, clipStart = 0, containerRef, onInteraction }) {
  const items = useTimelineStore((s) => s.items);
  const tracks = useTimelineStore((s) => s.tracks);
  const selectedItemId = useTimelineStore((s) => s.selectedItemId);
  const setSelectedItemId = useTimelineStore((s) => s.setSelectedItemId);
  const updateItem = useTimelineStore((s) => s.updateItem);

  // currentTime is already relative to clipStart (passed as currentTime - clipStart)
  const absTime = currentTime;

  // Filter to visible interactive items at current time.
  // Audio and subtitle items are excluded — subtitle items are rendered by SubtitleOverlay.
  // Respects track visibility — items on hidden tracks are not interactive.
  const visible = useMemo(() => {
    return items
      .filter((it) => {
        if (it.type === 'audio' || it.type === 'subtitle') return false;
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

  // Build a set of locked item IDs for quick lookup
  const lockedItemIds = useMemo(() => {
    const lockedTrackIds = new Set(tracks.filter((t) => t.locked).map((t) => t.id));
    return new Set(items.filter((it) => lockedTrackIds.has(it.trackId)).map((it) => it.id));
  }, [items, tracks]);

  if (visible.length === 0) return null;

  return (
    <div
      style={{
        position: 'absolute',
        inset: 0,
        zIndex: 10,
        // Container is transparent to clicks — only individual elements capture.
        // This lets clicks pass through to SubtitleOverlay and viewport below.
        pointerEvents: 'none',
      }}
    >
      {visible.map((item) => (
        <InteractiveElement
          key={item.id}
          item={item}
          isSelected={item.id === selectedItemId}
          isLocked={lockedItemIds.has(item.id)}
          containerRef={containerRef}
          onSelect={() => setSelectedItemId(item.id)}
          onUpdate={(updates) => updateItem(item.id, updates)}
          onInteraction={onInteraction}
        />
      ))}
    </div>
  );
}


// ── Corner / edge handle positions ────────────────────────────────
const HANDLES = [
  { id: 'nw', cursor: 'nwse-resize', x: 0, y: 0 },
  { id: 'ne', cursor: 'nesw-resize', x: 1, y: 0 },
  { id: 'sw', cursor: 'nesw-resize', x: 0, y: 1 },
  { id: 'se', cursor: 'nwse-resize', x: 1, y: 1 },
  { id: 'n',  cursor: 'ns-resize',   x: 0.5, y: 0 },
  { id: 's',  cursor: 'ns-resize',   x: 0.5, y: 1 },
  { id: 'w',  cursor: 'ew-resize',   x: 0, y: 0.5 },
  { id: 'e',  cursor: 'ew-resize',   x: 1, y: 0.5 },
];


function InteractiveElement({ item, isSelected, isLocked, containerRef, onSelect, onUpdate, onInteraction }) {
  const [isHovered, setIsHovered] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [isResizing, setIsResizing] = useState(false);
  const [isRotating, setIsRotating] = useState(false);
  const [isEditing, setIsEditing] = useState(false);
  const editRef = useRef(null);
  const dragState = useRef(null);

  // Use refs for callbacks/item to keep the useEffect stable during drag operations.
  // Without this, the mousemove/mouseup listeners get torn down and re-added on every
  // render (because inline onUpdate/onInteraction change identity each render), which
  // can drop mouse events mid-drag.
  const onUpdateRef = useRef(onUpdate);
  const onInteractionRef = useRef(onInteraction);
  const itemRef = useRef(item);
  onUpdateRef.current = onUpdate;
  onInteractionRef.current = onInteraction;
  itemRef.current = item;

  const isVideo = item.type === 'video';
  const isText = item.type === 'text';
  const isShape = item.type === 'shape';

  // Video items default to centered full-size; overlays default to smaller centered
  const defaultPos = isVideo ? { x: 50, y: 50 } : { x: 50, y: 50 };
  const defaultSize = isVideo ? { w: 100, h: 100 } : { w: 30, h: 30 };
  const pos = item.position || defaultPos;
  // For items with legacy {x:0,y:0}, treat as centered (viewport uses % from center)
  const effectivePos = (pos.x === 0 && pos.y === 0) ? { x: 50, y: 50 } : pos;

  // For text items, auto-measure the text to compute a tight bounding box
  const measuredSize = useMemo(() => {
    if (!isText || !containerRef?.current) return null;
    const text = item.textContent || '';
    if (!text) return null;
    const style = item.textStyle || {};
    const fontSize = style.fontSize || 48;
    const fontFamily = style.fontFamily || 'DM Sans';
    const fontWeight = style.fontWeight || 400;
    try {
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext && canvas.getContext('2d');
      // ``getContext`` can return null in restricted browser contexts
      // (private mode quirks, very old browsers, certain mobile WebViews).
      // Without this guard the next line throws and the entire overlay
      // fails to render.
      if (!ctx) return null;
      ctx.font = `${fontWeight} ${fontSize}px "${fontFamily}", sans-serif`;
      const lines = text.split('\n');
      const textW = Math.max(...lines.map((l) => ctx.measureText(l).width));
      const textH = fontSize * 1.4 * Math.max(1, lines.length);
      const rect = containerRef.current.getBoundingClientRect();
      if (rect.width > 0 && rect.height > 0) {
        const pad = 20; // generous padding for easier click targeting
        const wPct = ((textW + pad * 2) / rect.width) * 100;
        const hPct = ((textH + pad * 2) / rect.height) * 100;
        return { w: Math.max(12, Math.min(95, wPct)), h: Math.max(8, Math.min(80, hPct)) };
      }
    } catch { /* fallback to stored size */ }
    return null;
  }, [isText, item.textContent, item.textStyle?.fontSize, item.textStyle?.fontFamily, item.textStyle?.fontWeight, containerRef]);

  const size = measuredSize || item.size || defaultSize;
  const rotation = item.transform?.rotation || 0;

  // ── Get container dimensions ──
  const getContainerRect = useCallback(() => {
    if (containerRef?.current) {
      return containerRef.current.getBoundingClientRect();
    }
    return { width: 1, height: 1, left: 0, top: 0 };
  }, [containerRef]);

  // ── DRAG ──────────────────────────────────────────────
  const handleDragStart = useCallback((e) => {
    e.stopPropagation();
    e.preventDefault();
    onSelect();
    if (isLocked) return; // Locked items can be selected but not moved
    onInteraction?.(true);

    const rect = getContainerRect();
    // Pause undo history during drag so intermediate frames don't flood it
    useTimelineStore.temporal.getState().pause();
    dragState.current = {
      type: 'drag',
      startMouseX: e.clientX,
      startMouseY: e.clientY,
      startPosX: effectivePos.x,
      startPosY: effectivePos.y,
      containerW: rect.width,
      containerH: rect.height,
    };
    setIsDragging(true);
  }, [effectivePos, getContainerRect, onSelect, onInteraction, isLocked]);

  // ── RESIZE ────────────────────────────────────────────
  const handleResizeStart = useCallback((e, handleId) => {
    e.stopPropagation();
    e.preventDefault();
    if (isLocked) return; // Locked items cannot be resized
    onInteraction?.(true);

    const rect = getContainerRect();
    // Pause undo history during resize so intermediate frames don't flood it
    useTimelineStore.temporal.getState().pause();
    dragState.current = {
      type: 'resize',
      handle: handleId,
      startMouseX: e.clientX,
      startMouseY: e.clientY,
      startPosX: effectivePos.x,
      startPosY: effectivePos.y,
      startW: size.w,
      startH: size.h,
      containerW: rect.width,
      containerH: rect.height,
      startFontSize: item.textStyle?.fontSize || 48,
      isTextItem: item.type === 'text',
    };
    setIsResizing(true);
  }, [effectivePos, size, item, getContainerRect, onInteraction, isLocked]);

  // ── ROTATE ────────────────────────────────────────────
  const handleRotateStart = useCallback((e) => {
    e.stopPropagation();
    e.preventDefault();
    if (isLocked) return; // Locked items cannot be rotated
    onInteraction?.(true);

    const rect = getContainerRect();
    // Center of the element in viewport coordinates
    const centerX = rect.left + (effectivePos.x / 100) * rect.width;
    const centerY = rect.top + (effectivePos.y / 100) * rect.height;

    // Pause undo history during rotation so intermediate frames don't flood it
    useTimelineStore.temporal.getState().pause();
    dragState.current = {
      type: 'rotate',
      centerX,
      centerY,
      startAngle: Math.atan2(e.clientY - centerY, e.clientX - centerX) * (180 / Math.PI),
      startRotation: rotation,
    };
    setIsRotating(true);
  }, [effectivePos, rotation, getContainerRect, onInteraction, isLocked]);

  // ── Pointer move / up handlers ────────────────────────
  // Pointer events unify mouse + touch + pen so the same drag, resize
  // and rotate paths work on tablets / phones. Only depends on the
  // boolean flags so listeners stay stable during the gesture; accesses
  // latest ``item``/``onUpdate``/``onInteraction`` via refs to avoid
  // listener churn.
  useEffect(() => {
    if (!isDragging && !isResizing && !isRotating) return;

    const handleMouseMove = (e) => {
      const ds = dragState.current;
      if (!ds) return;

      if (ds.type === 'drag') {
        const dx = e.clientX - ds.startMouseX;
        const dy = e.clientY - ds.startMouseY;
        const newX = ds.startPosX + (dx / ds.containerW) * 100;
        const newY = ds.startPosY + (dy / ds.containerH) * 100;
        onUpdateRef.current({
          position: {
            x: Math.max(0, Math.min(100, Math.round(newX * 10) / 10)),
            y: Math.max(0, Math.min(100, Math.round(newY * 10) / 10)),
          },
        });
      }

      if (ds.type === 'resize') {
        const dx = ((e.clientX - ds.startMouseX) / ds.containerW) * 100;
        const dy = ((e.clientY - ds.startMouseY) / ds.containerH) * 100;
        const h = ds.handle;

        let newX = ds.startPosX;
        let newY = ds.startPosY;
        let newW = ds.startW;
        let newH = ds.startH;

        if (h.includes('e')) {
          newW = Math.max(2, ds.startW + dx);
          newX = ds.startPosX + dx / 2;
        }
        if (h.includes('w')) {
          newW = Math.max(2, ds.startW - dx);
          newX = ds.startPosX + dx / 2;
        }
        if (h.includes('s')) {
          newH = Math.max(2, ds.startH + dy);
          newY = ds.startPosY + dy / 2;
        }
        if (h.includes('n')) {
          newH = Math.max(2, ds.startH - dy);
          newY = ds.startPosY + dy / 2;
        }

        // Shift key = lock aspect ratio
        if (e.shiftKey && ds.startW > 0 && ds.startH > 0) {
          const aspect = ds.startW / ds.startH;
          if (h === 'e' || h === 'w') {
            newH = newW / aspect;
          } else if (h === 'n' || h === 's') {
            newW = newH * aspect;
          } else {
            const dw = Math.abs(newW - ds.startW);
            const dh = Math.abs(newH - ds.startH);
            if (dw > dh) {
              newH = newW / aspect;
            } else {
              newW = newH * aspect;
            }
          }
        }

        const updates = {
          position: {
            x: Math.round(newX * 10) / 10,
            y: Math.round(newY * 10) / 10,
          },
          size: {
            w: Math.round(Math.max(2, newW) * 10) / 10,
            h: Math.round(Math.max(2, newH) * 10) / 10,
          },
        };

        // Scale font size for text items on corner (diagonal) drags
        const currentItem = itemRef.current;
        if (ds.isTextItem && h.length === 2 && ds.startW > 0) {
          const scale = newW / ds.startW;
          const newFontSize = Math.max(8, Math.min(400, Math.round(ds.startFontSize * scale)));
          updates.textStyle = { ...(currentItem.textStyle || {}), fontSize: newFontSize };
        }

        onUpdateRef.current(updates);
      }

      if (ds.type === 'rotate') {
        const angle = Math.atan2(e.clientY - ds.centerY, e.clientX - ds.centerX) * (180 / Math.PI);
        let newRotation = ds.startRotation + (angle - ds.startAngle);

        if (e.shiftKey) {
          newRotation = Math.round(newRotation / 15) * 15;
        }
        newRotation = ((newRotation % 360) + 360) % 360;
        if (newRotation > 180) newRotation -= 360;

        const currentItem = itemRef.current;
        onUpdateRef.current({
          transform: { ...(currentItem.transform || {}), rotation: Math.round(newRotation) },
        });
      }
    };

    const handleMouseUp = () => {
      // Resume undo history so the final state is recorded as one snapshot
      useTimelineStore.temporal.getState().resume();
      dragState.current = null;
      setIsDragging(false);
      setIsResizing(false);
      setIsRotating(false);
      setTimeout(() => onInteractionRef.current?.(false), 100);
    };

    // Pointer events are a superset of mouse/touch/pen — registering
    // ``pointermove`` + ``pointerup`` (with ``pointercancel`` for the
    // touch-interrupted case) is enough to drive every input modality.
    window.addEventListener('pointermove', handleMouseMove);
    window.addEventListener('pointerup', handleMouseUp);
    window.addEventListener('pointercancel', handleMouseUp);
    return () => {
      window.removeEventListener('pointermove', handleMouseMove);
      window.removeEventListener('pointerup', handleMouseUp);
      window.removeEventListener('pointercancel', handleMouseUp);
      // Safety: resume undo history if component unmounts during drag
      if (isDragging || isResizing || isRotating) {
        useTimelineStore.temporal.getState().resume();
      }
    };
  }, [isDragging, isResizing, isRotating]);

  // ── Click to select ───────────────────────────────────
  const handleClick = useCallback((e) => {
    e.stopPropagation();
    onSelect();
  }, [onSelect]);

  // ── Double-click to edit text/subtitle ────────────────
  const handleDoubleClick = useCallback((e) => {
    e.stopPropagation();
    if (isLocked) return; // Locked items cannot be edited
    if (item.type === 'text' || item.type === 'subtitle') {
      setIsEditing(true);
      onInteraction?.(true);
      // Focus the input after render
      setTimeout(() => editRef.current?.focus(), 50);
    }
  }, [item.type, onInteraction, isLocked]);

  const handleEditBlur = useCallback(() => {
    setIsEditing(false);
    onInteraction?.(false);
  }, [onInteraction]);

  const handleEditChange = useCallback((e) => {
    const val = e.target.value;
    if (item.type === 'text') {
      onUpdate({ textContent: val });
    } else if (item.type === 'subtitle') {
      onUpdate({ subtitleText: val });
    }
  }, [item.type, onUpdate]);

  const handleEditKeyDown = useCallback((e) => {
    e.stopPropagation(); // prevent keyboard shortcuts
    if (e.key === 'Escape') {
      setIsEditing(false);
      onInteraction?.(false);
    }
  }, [onInteraction]);

  // Determine the bounding box style.
  // Elements use percentage-based position (center) and size.
  // Text items get extra padding so the bounding box isn't cramped
  const boxPad = 0;
  const boxStyle = {
    position: 'absolute',
    left: `${effectivePos.x}%`,
    top: `${effectivePos.y}%`,
    width: `${size.w}%`,
    height: `${size.h}%`,
    padding: boxPad > 0 ? boxPad : undefined,
    transform: `translate(-50%, -50%) ${rotation ? `rotate(${rotation}deg)` : ''}`,
    cursor: isLocked ? 'not-allowed' : isDragging ? 'grabbing' : (isVideo ? 'default' : 'grab'),
    // Video items: pointer-events none so they don't block overlays.
    // Exception: during active manipulation, allow events for smooth tracking.
    pointerEvents: isVideo
      ? (isDragging || isResizing || isRotating ? 'auto' : 'none')
      : 'auto',
    minWidth: 20,
    minHeight: 20,
    // Video items always sit behind overlays (z:1). Other items at z:10, selected at z:15.
    // This ensures text/shape/image overlays are always clickable above the video.
    zIndex: isVideo ? 1 : (isSelected ? 15 : 10),
  };

  const isActive = isDragging || isResizing || isRotating;

  return (
    <div
      style={{
        ...boxStyle,
        // ``touch-action: none`` lets us own the touch sequence so the
        // browser doesn't preempt the drag with scroll/zoom gestures.
        touchAction: isLocked ? 'auto' : 'none',
      }}
      onPointerDown={isEditing ? undefined : handleDragStart}
      onClick={handleClick}
      onDoubleClick={handleDoubleClick}
      onPointerEnter={() => setIsHovered(true)}
      onPointerLeave={() => setIsHovered(false)}
    >
      {/* Invisible hit area matching the element */}
      <div style={{
        position: 'absolute',
        inset: -4,
        borderRadius: 2,
      }} />

      {/* Inline text editing */}
      {isEditing && (item.type === 'text' || item.type === 'subtitle') && (
        <textarea
          ref={editRef}
          value={item.type === 'text' ? (item.textContent || '') : (item.subtitleText || '')}
          onChange={handleEditChange}
          onBlur={handleEditBlur}
          onKeyDown={handleEditKeyDown}
          onPointerDown={(e) => e.stopPropagation()}
          style={{
            position: 'absolute',
            inset: 0,
            width: '100%',
            minHeight: 40,
            background: 'rgba(0,0,0,0.6)',
            color: '#fff',
            border: '2px solid #0A84FF',
            borderRadius: 4,
            padding: '6px 8px',
            fontSize: 14,
            fontFamily: 'inherit',
            resize: 'none',
            outline: 'none',
            zIndex: 50,
            backdropFilter: 'blur(4px)',
          }}
        />
      )}

      {/* Hover outline (when not selected) */}
      {isHovered && !isSelected && (
        <div style={{
          position: 'absolute',
          inset: -1,
          border: '1px dashed rgba(10, 132, 255, 0.5)',
          borderRadius: 2,
          pointerEvents: 'none',
        }} />
      )}

      {/* Selection outline */}
      {isSelected && (
        <>
          {/* Selection border — interactive: enables drag-from-border for video items */}
          <div
            style={{
              position: 'absolute',
              inset: -2,
              border: '2px solid #0A84FF',
              borderRadius: 2,
              pointerEvents: 'auto',
              boxShadow: '0 0 0 1px rgba(10, 132, 255, 0.3)',
              cursor: isLocked ? 'not-allowed' : (isDragging ? 'grabbing' : 'grab'),
              background: 'transparent',
              touchAction: 'none',
            }}
            onPointerDown={isEditing ? undefined : handleDragStart}
          />

          {/* Resize handles */}
          {HANDLES.map((h) => (
            <div
              key={h.id}
              style={{
                position: 'absolute',
                left: `${h.x * 100}%`,
                top: `${h.y * 100}%`,
                width: 10,
                height: 10,
                transform: 'translate(-50%, -50%)',
                background: '#fff',
                border: '2px solid #0A84FF',
                borderRadius: h.id.length === 2 ? 2 : '50%', // corners = square, edges = round
                cursor: h.cursor,
                zIndex: 30,
                boxShadow: '0 1px 3px rgba(0,0,0,0.3)',
                pointerEvents: 'auto',
                touchAction: 'none',
              }}
              onPointerDown={(e) => handleResizeStart(e, h.id)}
            />
          ))}

          {/* Rotation handle (above the element) */}
          <div style={{
            position: 'absolute',
            left: '50%',
            top: -24,
            transform: 'translateX(-50%)',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            zIndex: 30,
            pointerEvents: 'auto',
          }}>
            {/* Stem line connecting to element */}
            <div style={{
              width: 1,
              height: 8,
              background: '#0A84FF',
              position: 'absolute',
              bottom: -8,
            }} />
            {/* Rotation circle handle */}
            <div
              style={{
                width: 16,
                height: 16,
                borderRadius: '50%',
                background: '#fff',
                border: '2px solid #0A84FF',
                cursor: 'grab',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                boxShadow: '0 1px 4px rgba(0,0,0,0.3)',
                pointerEvents: 'auto',
                touchAction: 'none',
              }}
              onPointerDown={handleRotateStart}
              title="Rotate"
            >
              {/* Rotation icon */}
              <svg width="10" height="10" viewBox="0 0 16 16" fill="none" stroke="#0A84FF" strokeWidth="2">
                <path d="M14 8A6 6 0 1 1 8 2" strokeLinecap="round" />
                <path d="M8 0l3 2-3 2" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            </div>
          </div>

          {/* Info badge showing position/size while dragging */}
          {isActive && (
            <div style={{
              position: 'absolute',
              left: '50%',
              bottom: -24,
              transform: 'translateX(-50%)',
              background: 'rgba(0,0,0,0.8)',
              color: '#fff',
              fontSize: 10,
              fontFamily: 'var(--ve-font-mono, monospace)',
              padding: '2px 6px',
              borderRadius: 3,
              whiteSpace: 'nowrap',
              pointerEvents: 'none',
              zIndex: 40,
              backdropFilter: 'blur(4px)',
            }}>
              {isDragging && `${effectivePos.x.toFixed(1)}%, ${effectivePos.y.toFixed(1)}%`}
              {isResizing && `${size.w.toFixed(1)}% x ${size.h.toFixed(1)}%`}
              {isRotating && `${rotation}°`}
            </div>
          )}
        </>
      )}
    </div>
  );
}
