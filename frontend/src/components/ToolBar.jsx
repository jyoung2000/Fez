import React from 'react';
import useTimelineStore from '../stores/timelineStore';

const TOOLS = [
  { id: 'select', label: 'Select', shortcut: 'V', icon: (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M3 3l7.07 16.97 2.51-7.39 7.39-2.51L3 3z" />
    </svg>
  )},
  { id: 'razor', label: 'Razor', shortcut: 'C', icon: (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <line x1="12" y1="2" x2="12" y2="22" />
      <path d="M6 12h12" />
    </svg>
  )},
  { id: 'text', label: 'Text', shortcut: 'T', icon: (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="4 7 4 4 20 4 20 7" />
      <line x1="9" y1="20" x2="15" y2="20" />
      <line x1="12" y1="4" x2="12" y2="20" />
    </svg>
  )},
  { id: 'shape', label: 'Shape', shortcut: 'R', icon: (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <rect x="3" y="3" width="18" height="18" rx="2" />
    </svg>
  )},
];

export default function ToolBar({ compact = false }) {
  const activeTool = useTimelineStore((s) => s.activeTool);
  const setActiveTool = useTimelineStore((s) => s.setActiveTool);
  const addItem = useTimelineStore((s) => s.addItem);
  const setSelectedItemId = useTimelineStore((s) => s.setSelectedItemId);
  const duration = useTimelineStore((s) => s.duration);
  const playhead = useTimelineStore((s) => s.playhead);

  const handleToolClick = (toolId) => {
    setActiveTool(toolId);

    // For text/shape, auto-create an item at the playhead
    if (toolId === 'text' || toolId === 'shape') {
      // Determine target overlay track dynamically
      const state = useTimelineStore.getState();
      let targetTrackId = null;

      // 1. If selected item is on an overlay track, use that track
      if (state.selectedItemId) {
        const selectedItem = state.items.find(i => i.id === state.selectedItemId);
        if (selectedItem) {
          const selectedTrack = state.tracks.find(t => t.id === selectedItem.trackId);
          if (selectedTrack?.type === 'overlay' && !selectedTrack.locked) {
            targetTrackId = selectedTrack.id;
          }
        }
      }

      // 2. Fallback: first unlocked overlay track
      if (!targetTrackId) {
        const overlayTrack = state.tracks.find(t => t.type === 'overlay' && !t.locked);
        targetTrackId = overlayTrack?.id || 'v2';
      }

      if (toolId === 'text') {
        const newId = addItem({
          trackId: targetTrackId,
          type: 'text',
          start: playhead,
          end: Math.min(playhead + 5, duration || playhead + 5),
          textContent: 'New Text',
          textStyle: {
            fontSize: 48,
            fontFamily: 'DM Sans',
            fontWeight: 700,
            color: '#FFFFFF',
            textAlign: 'center',
            outlineWidth: 2,
            outlineColor: '#000000',
            bgColor: null,
            bgOpacity: 0,
            bgPadding: 8,
            bgRadius: 4,
            shadowBlur: 0,
            shadowColor: 'rgba(0,0,0,0.5)',
            shadowOffsetX: 0,
            shadowOffsetY: 0,
            animation: 'none',
          },
          position: { x: 50, y: 50 },
          size: { w: 60, h: 15 },
        });
        setSelectedItemId(newId);
      } else {
        const newId = addItem({
          trackId: targetTrackId,
          type: 'shape',
          start: playhead,
          end: Math.min(playhead + 5, duration || playhead + 5),
          shapeType: 'rectangle',
          shapeStyle: {
            fillColor: '#FF3B30',
            strokeColor: '#FFFFFF',
            strokeWidth: 2,
            cornerRadius: 8,
          },
          position: { x: 30, y: 30 },
          size: { w: 40, h: 30 },
        });
        setSelectedItemId(newId);
      }
      setActiveTool('select');
    }
  };

  return (
    <div className={`ve-toolbar${compact ? ' ve-toolbar--compact' : ''}`}>
      {TOOLS.map((tool) => (
        <button
          key={tool.id}
          className={`ve-toolbar__btn${activeTool === tool.id ? ' ve-toolbar__btn--active' : ''}`}
          onClick={() => handleToolClick(tool.id)}
          title={`${tool.label} (${tool.shortcut})`}
        >
          {tool.icon}
          {!compact && <span className="ve-toolbar__label">{tool.label}</span>}
        </button>
      ))}

      <div className="ve-toolbar__divider" />

      {/* Undo/Redo */}
      <button
        className="ve-toolbar__btn"
        onClick={() => useTimelineStore.temporal.getState().undo()}
        title="Undo (Ctrl+Z)"
      >
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <polyline points="1 4 1 10 7 10" />
          <path d="M3.51 15a9 9 0 102.13-9.36L1 10" />
        </svg>
      </button>
      <button
        className="ve-toolbar__btn"
        onClick={() => useTimelineStore.temporal.getState().redo()}
        title="Redo (Ctrl+Shift+Z)"
      >
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <polyline points="23 4 23 10 17 10" />
          <path d="M20.49 15a9 9 0 11-2.13-9.36L23 10" />
        </svg>
      </button>
    </div>
  );
}
