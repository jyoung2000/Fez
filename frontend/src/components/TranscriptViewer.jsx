import React, { useState, useMemo, useCallback, useRef, useEffect } from 'react';
import useResponsive from '../hooks/useResponsive';
import { spokenWindow } from '../utils/subtitleTiming';

const SPEAKER_COLORS_LIST = [
  'var(--accent-cyan)',
  'var(--accent-amber)',
  'var(--success)',
  '#A78BFA',
];

// Must match the palette in VideoEditor / SubtitleOverlay / ClipPreview
const DEFAULT_SPEAKER_PALETTE = [
  '#00D9FF', '#F59E0B', '#10B981', '#A78BFA', '#EF4444', '#EC4899',
  '#06B6D4', '#8B5CF6', '#F97316', '#14B8A6', '#E879F9', '#84CC16',
  '#FB7185', '#38BDF8', '#FBBF24', '#34D399', '#C084FC', '#F472B6',
  '#22D3EE', '#A3E635', '#FB923C', '#2DD4BF', '#818CF8', '#F87171',
];

function formatTime(seconds) {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, '0')}`;
}

function toSRT(segments) {
  return segments
    .map((seg, i) => {
      const startH = Math.floor(seg.start / 3600);
      const startM = Math.floor((seg.start % 3600) / 60);
      const startS = Math.floor(seg.start % 60);
      const startMs = Math.floor((seg.start % 1) * 1000);
      const endH = Math.floor(seg.end / 3600);
      const endM = Math.floor((seg.end % 3600) / 60);
      const endS = Math.floor(seg.end % 60);
      const endMs = Math.floor((seg.end % 1) * 1000);
      return `${i + 1}\n${String(startH).padStart(2, '0')}:${String(startM).padStart(2, '0')}:${String(startS).padStart(2, '0')},${String(startMs).padStart(3, '0')} --> ${String(endH).padStart(2, '0')}:${String(endM).padStart(2, '0')}:${String(endS).padStart(2, '0')},${String(endMs).padStart(3, '0')}\n${seg.speaker}: ${seg.text}`;
    })
    .join('\n\n');
}

function toTXT(segments) {
  return segments.map((seg) => `[${formatTime(seg.start)}] ${seg.speaker}: ${seg.text}`).join('\n');
}

// After any wheel / touchmove / keydown on the scroll container we pause
// auto-centering for this many milliseconds so the user's scroll doesn't
// get yanked back by the animation. The next active-segment change past
// the quiet window resumes it.
const MANUAL_OVERRIDE_MS = 2500;

export default function TranscriptViewer({ transcript, onSeek, jobId, onSpeakerRenamed, onTranscriptUpdated, onSpeakerColorChanged, onSpeakerAdded, speakerColors, timeRange, currentTime, maxHeight }) {
  const { isMobile } = useResponsive();
  const scrollContainerRef = useRef(null);
  const activeSegRef = useRef(null);
  const [search, setSearch] = useState('');
  const [editingSpeaker, setEditingSpeaker] = useState(null);
  const [editValue, setEditValue] = useState('');
  const [renaming, setRenaming] = useState(false);
  const [editingSegIdx, setEditingSegIdx] = useState(null);
  const [editSegText, setEditSegText] = useState('');
  const [savingSegment, setSavingSegment] = useState(false);

  // Multi-select state
  const [selectedIndices, setSelectedIndices] = useState(new Set());
  const lastClickedIdx = useRef(null);
  const [bulkSaving, setBulkSaving] = useState(false);

  // Add speaker state
  const [addingSpeaker, setAddingSpeaker] = useState(false);
  const [newSpeakerName, setNewSpeakerName] = useState('');

  // Delete speaker state
  const [deletingSpeaker, setDeletingSpeaker] = useState(null);
  const [deleteReassignTo, setDeleteReassignTo] = useState('');
  const [deletingSpeakerInProgress, setDeletingSpeakerInProgress] = useState(false);

  // Insert segment state
  const [insertAfterIdx, setInsertAfterIdx] = useState(null);
  const [insertText, setInsertText] = useState('');
  const [insertSpeaker, setInsertSpeaker] = useState('');
  const [insertStart, setInsertStart] = useState('');
  const [insertEnd, setInsertEnd] = useState('');
  const [insertSaving, setInsertSaving] = useState(false);

  // Build ordered list of unique speakers and assign colors by position
  const speakers = useMemo(() => {
    const seen = [];
    transcript.forEach((seg) => {
      if (!seen.includes(seg.speaker)) seen.push(seg.speaker);
    });
    return seen;
  }, [transcript]);

  // Map speaker name -> color: use subtitle speakerColors if provided (matches subtitles),
  // otherwise fall back to DEFAULT_SPEAKER_PALETTE, then SPEAKER_COLORS_LIST
  const speakerColor = useCallback(
    (name) => {
      if (speakerColors?.[name]) return speakerColors[name];
      const idx = speakers.indexOf(name);
      return DEFAULT_SPEAKER_PALETTE[idx % DEFAULT_SPEAKER_PALETTE.length] || SPEAKER_COLORS_LIST[idx % SPEAKER_COLORS_LIST.length] || 'var(--text-secondary)';
    },
    [speakers, speakerColors]
  );

  const timeFiltered = useMemo(() => {
    if (!timeRange) return transcript;
    return transcript.filter((seg) =>
      seg.start < timeRange.end && seg.end > timeRange.start
    );
  }, [transcript, timeRange]);

  const filtered = useMemo(() => {
    if (!search.trim()) return timeFiltered;
    const q = search.toLowerCase();
    return timeFiltered.filter((seg) => seg.text.toLowerCase().includes(q) || seg.speaker.toLowerCase().includes(q));
  }, [timeFiltered, search]);

  // Determine which segment is currently playing (by original transcript index).
  //
  // Uses ``spokenWindow`` from ``utils/subtitleTiming`` so the active-line
  // highlight tracks Whisper's per-word timestamps when available — the
  // segment-level start/end that Whisper emits are ±100–300 ms boundary
  // estimates, which read as "highlight is a half beat late." Per-word
  // timestamps are much tighter. Sharing the helper with
  // ``SubtitleOverlay`` also guarantees the transcript highlight and the
  // burned-in subtitle agree on when a line is "being spoken."
  //
  // Gap-hold: when ``currentTime`` lands in a short gap between two
  // segments, we keep the most recently ended segment "active" for up to
  // ``GAP_HOLD_SEC``. Tuned down from the legacy 5.0 s (which read as
  // "stuck on an old line") to 0.75 s — that covers Whisper's typical
  // inter-segment gaps (50–400 ms) plus a breath without lingering past
  // when a listener would say the line is done.
  const activeOriginalIdx = useMemo(() => {
    if (currentTime == null || !transcript.length) return -1;
    const GAP_HOLD_SEC = 0.75;

    let lastEndedIdx = -1;
    let lastEndedTime = -Infinity;
    for (let i = 0; i < transcript.length; i++) {
      const seg = transcript[i];
      const win = spokenWindow(seg);
      // Direct hit on the spoken window.
      if (win.start <= currentTime && currentTime < win.end) return i;
      // Track the most recently *ended* spoken window for gap-hold
      // fallback. (Note: this is the window end, not ``seg.end``, so
      // the fallback honors per-word timestamps too.)
      if (win.end <= currentTime && win.end > lastEndedTime) {
        lastEndedTime = win.end;
        lastEndedIdx = i;
      }
    }
    // If we're sitting in a gap shorter than GAP_HOLD_SEC, hold the
    // previous line. If any segment had already *started* we'd have hit
    // the direct branch above, so reaching here means the next line
    // hasn't begun yet and holding is safe.
    if (lastEndedIdx >= 0 && currentTime - lastEndedTime <= GAP_HOLD_SEC) {
      return lastEndedIdx;
    }
    return -1;
  }, [currentTime, transcript]);

  // Auto-scroll to keep the active segment centered in the transcript view.
  // Uses per-frame lerp (exponential ease-out) instead of CSS smooth scroll
  // to avoid choppiness when segments change rapidly during playback.
  //
  // Scroll math: we compute the delta via ``getBoundingClientRect()`` so
  // the result is independent of which ancestor happens to be
  // ``position: relative`` (the legacy code used ``offsetTop`` which
  // reports distance to the nearest *positioned* ancestor — on this page
  // that's ``<body>``, not the scroll container, so the target
  // ``scrollTop`` had nothing to do with the actual row position inside
  // the container). The resulting ``targetTop`` is clamped to
  // ``[0, scrollHeight - clientHeight]`` so we never request an
  // impossible position.
  //
  // Manual-scroll override: see MANUAL_OVERRIDE_MS above. A wheel /
  // touchmove / keydown on the container pauses auto-centering and
  // immediately kills any running centering animation so the user's
  // input is not fought by the per-frame lerp. Initialize to
  // ``-Infinity`` so the FIRST active-segment change after mount is
  // never suppressed by a stale zero reference that happens to fall
  // within the override window.
  const scrollAnimRef = useRef(null);
  const lastUserScrollRef = useRef(-Infinity);
  useEffect(() => {
    const container = scrollContainerRef.current;
    if (!container) return;
    const onUserScroll = () => {
      lastUserScrollRef.current = performance.now();
      // Kill the running auto-centering animation right away — without
      // this the ~400 ms lerp would keep writing ``scrollTop`` on each
      // frame and visibly yank the user back while they're trying to
      // wheel. That's the "the transcript scroll doesn't work" symptom.
      if (scrollAnimRef.current) {
        cancelAnimationFrame(scrollAnimRef.current);
        scrollAnimRef.current = null;
      }
    };
    const wheelOpts = { passive: true };
    const touchOpts = { passive: true };
    container.addEventListener('wheel', onUserScroll, wheelOpts);
    container.addEventListener('touchmove', onUserScroll, touchOpts);
    container.addEventListener('keydown', onUserScroll);
    return () => {
      container.removeEventListener('wheel', onUserScroll, wheelOpts);
      container.removeEventListener('touchmove', onUserScroll, touchOpts);
      container.removeEventListener('keydown', onUserScroll);
    };
  }, []);

  useEffect(() => {
    if (activeOriginalIdx < 0) return;
    const container = scrollContainerRef.current;
    if (!container) return;

    // Resolve the active row. Prefer the callback ref, but fall back
    // to a DOM query on the ``data-active`` attribute — under rapid
    // re-renders (search typing, segment edits) the callback ref can
    // briefly be ``null`` even though the row is present in the DOM.
    const el =
      activeSegRef.current || container.querySelector('[data-active="true"]');
    if (!el) return;

    // Cancel any running animation
    if (scrollAnimRef.current) {
      cancelAnimationFrame(scrollAnimRef.current);
      scrollAnimRef.current = null;
    }

    // Manual-scroll override: user wheel/touch/key within the last
    // MANUAL_OVERRIDE_MS pauses auto-centering. The next
    // active-segment change past the quiet window will resume it.
    if (performance.now() - lastUserScrollRef.current < MANUAL_OVERRIDE_MS) {
      return;
    }

    // Target: center the active row in the container. Use bounding
    // rects so the math is ``offsetParent``-independent, then clamp
    // to the valid scroll range.
    const rowRect = el.getBoundingClientRect();
    const contRect = container.getBoundingClientRect();
    const rowCenterInContainer =
      (rowRect.top - contRect.top) + container.scrollTop + rowRect.height / 2;
    let targetTop = rowCenterInContainer - container.clientHeight / 2;
    const maxScroll = Math.max(0, container.scrollHeight - container.clientHeight);
    if (targetTop < 0) targetTop = 0;
    if (targetTop > maxScroll) targetTop = maxScroll;

    // Already centered — nothing to animate. Avoids pointless rAF work
    // and keeps the user free to scroll during long segments.
    if (Math.abs(targetTop - container.scrollTop) < 2) return;

    const animate = () => {
      // Per-frame user-input check: if the user wheeled / touched /
      // keyed between the previous frame and this one, bail out so
      // their scroll takes over cleanly. (Without this the animation
      // keeps writing ``scrollTop`` and fights user input for the
      // rest of the lerp.)
      if (performance.now() - lastUserScrollRef.current < MANUAL_OVERRIDE_MS) {
        scrollAnimRef.current = null;
        return;
      }
      const diff = targetTop - container.scrollTop;
      // If close enough, snap and stop
      if (Math.abs(diff) < 1) {
        container.scrollTop = targetTop;
        scrollAnimRef.current = null;
        return;
      }
      // Lerp: move 15% of remaining distance each frame (~60fps)
      // This gives smooth exponential ease-out without fighting CSS transitions
      container.scrollTop += diff * 0.15;
      scrollAnimRef.current = requestAnimationFrame(animate);
    };
    scrollAnimRef.current = requestAnimationFrame(animate);

    return () => {
      if (scrollAnimRef.current) {
        cancelAnimationFrame(scrollAnimRef.current);
        scrollAnimRef.current = null;
      }
    };
  }, [activeOriginalIdx]);

  const download = (content, filename) => {
    const blob = new Blob([content], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  };

  const copyAll = () => {
    const text = toTXT(timeFiltered);
    navigator.clipboard.writeText(text);
  };

  const startEditing = (speaker) => {
    setEditingSpeaker(speaker);
    setEditValue(speaker);
  };

  const cancelEditing = () => {
    setEditingSpeaker(null);
    setEditValue('');
  };

  const submitRename = async () => {
    const newName = editValue.trim();
    if (!newName || newName === editingSpeaker || !jobId) {
      cancelEditing();
      return;
    }
    setRenaming(true);
    try {
      const res = await fetch(`/api/jobs/${jobId}/speakers`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ speaker_names: { [editingSpeaker]: newName } }),
      });
      if (res.ok) {
        // Transfer the old speaker's subtitle color to the new name
        if (onSpeakerColorChanged && speakerColors?.[editingSpeaker]) {
          onSpeakerColorChanged(newName, speakerColors[editingSpeaker], editingSpeaker);
        }
        cancelEditing();
        if (onSpeakerRenamed) onSpeakerRenamed();
      }
    } catch (err) {
      console.error('Speaker rename failed:', err);
    } finally {
      setRenaming(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      submitRename();
    } else if (e.key === 'Escape') {
      cancelEditing();
    }
  };

  // --- Delete speaker ---
  const startDeletingSpeaker = (speaker) => {
    setDeletingSpeaker(speaker);
    setDeleteReassignTo('');
  };

  const cancelDeletingSpeaker = () => {
    setDeletingSpeaker(null);
    setDeleteReassignTo('');
  };

  const confirmDeleteSpeaker = async () => {
    if (!jobId || !deletingSpeaker) return;
    setDeletingSpeakerInProgress(true);
    try {
      const base = `/api/jobs/${jobId}/speakers/${encodeURIComponent(deletingSpeaker)}`;
      const url = deleteReassignTo
        ? `${base}?reassign_to=${encodeURIComponent(deleteReassignTo)}`
        : base;
      const res = await fetch(url, { method: 'DELETE' });
      if (res.ok) {
        cancelDeletingSpeaker();
        if (onTranscriptUpdated) onTranscriptUpdated();
      } else {
        console.error('Delete speaker failed:', res.status, await res.text());
      }
    } catch (err) {
      console.error('Delete speaker failed:', err);
    } finally {
      setDeletingSpeakerInProgress(false);
    }
  };

  // --- Segment text editing ---
  const startSegEdit = (originalIdx, text, e) => {
    e.stopPropagation();
    setEditingSegIdx(originalIdx);
    setEditSegText(text);
  };

  const cancelSegEdit = () => {
    setEditingSegIdx(null);
    setEditSegText('');
  };

  const saveSegEdit = async () => {
    const newText = editSegText.trim();
    if (editingSegIdx === null || !jobId) { cancelSegEdit(); return; }
    if (!newText || newText === transcript[editingSegIdx]?.text) { cancelSegEdit(); return; }
    setSavingSegment(true);
    try {
      const res = await fetch(`/api/jobs/${jobId}/transcript/${editingSegIdx}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: newText }),
      });
      if (res.ok) {
        cancelSegEdit();
        if (onTranscriptUpdated) onTranscriptUpdated();
      }
    } catch (err) {
      console.error('Segment update failed:', err);
    } finally {
      setSavingSegment(false);
    }
  };

  const handleSegKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      saveSegEdit();
    } else if (e.key === 'Escape') {
      cancelSegEdit();
    }
  };

  // --- Change speaker on a single segment ---
  const handleChangeSpeaker = async (originalIdx, newSpeaker) => {
    if (!jobId || !newSpeaker) return;
    try {
      const res = await fetch(`/api/jobs/${jobId}/transcript/${originalIdx}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ speaker: newSpeaker }),
      });
      if (res.ok && onTranscriptUpdated) onTranscriptUpdated();
    } catch (err) {
      console.error('Speaker change failed:', err);
    }
  };

  // --- Multi-select helpers ---
  const toggleSelect = useCallback((originalIdx, e) => {
    setSelectedIndices((prev) => {
      const next = new Set(prev);
      if (e?.shiftKey && lastClickedIdx.current != null) {
        // Range select: select all filtered segments between last click and this one
        const filteredOriginalIndices = filtered.map((seg) => transcript.indexOf(seg));
        const a = filteredOriginalIndices.indexOf(lastClickedIdx.current);
        const b = filteredOriginalIndices.indexOf(originalIdx);
        if (a >= 0 && b >= 0) {
          const [lo, hi] = a < b ? [a, b] : [b, a];
          for (let i = lo; i <= hi; i++) {
            next.add(filteredOriginalIndices[i]);
          }
        }
      } else {
        if (next.has(originalIdx)) next.delete(originalIdx);
        else next.add(originalIdx);
      }
      lastClickedIdx.current = originalIdx;
      return next;
    });
  }, [filtered, transcript]);

  const selectAllFiltered = useCallback(() => {
    const allIndices = filtered.map((seg) => transcript.indexOf(seg));
    setSelectedIndices(new Set(allIndices));
  }, [filtered, transcript]);

  const clearSelection = useCallback(() => {
    setSelectedIndices(new Set());
    lastClickedIdx.current = null;
  }, []);

  const handleBulkSpeakerChange = async (newSpeaker) => {
    if (!jobId || !newSpeaker || selectedIndices.size === 0) return;
    setBulkSaving(true);
    try {
      const res = await fetch(`/api/jobs/${jobId}/transcript/bulk-update-speaker`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ segment_indices: [...selectedIndices], speaker: newSpeaker }),
      });
      if (res.ok) {
        clearSelection();
        if (onTranscriptUpdated) onTranscriptUpdated();
      }
    } catch (err) {
      console.error('Bulk speaker change failed:', err);
    } finally {
      setBulkSaving(false);
    }
  };

  // --- Delete segment ---
  const handleDeleteSegment = async (originalIdx) => {
    if (!jobId) return;
    try {
      const res = await fetch(`/api/jobs/${jobId}/transcript/${originalIdx}`, { method: 'DELETE' });
      if (res.ok && onTranscriptUpdated) onTranscriptUpdated();
    } catch (err) {
      console.error('Segment delete failed:', err);
    }
  };

  // --- Insert segment ---
  const openInsertForm = (afterIdx) => {
    const seg = transcript[afterIdx];
    setInsertAfterIdx(afterIdx);
    setInsertText('');
    setInsertSpeaker(seg?.speaker || speakers[0] || 'Speaker 1');
    setInsertStart(seg ? seg.end.toFixed(1) : '0');
    setInsertEnd(seg ? (seg.end + 2).toFixed(1) : '2');
  };

  const cancelInsert = () => {
    setInsertAfterIdx(null);
    setInsertText('');
  };

  const saveInsert = async () => {
    const text = insertText.trim();
    if (!text || !jobId) { cancelInsert(); return; }
    setInsertSaving(true);
    try {
      const res = await fetch(`/api/jobs/${jobId}/transcript`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          start: parseFloat(insertStart) || 0,
          end: parseFloat(insertEnd) || 0,
          text,
          speaker: insertSpeaker || 'Speaker 1',
        }),
      });
      if (res.ok) {
        cancelInsert();
        if (onTranscriptUpdated) onTranscriptUpdated();
      }
    } catch (err) {
      console.error('Insert segment failed:', err);
    } finally {
      setInsertSaving(false);
    }
  };

  const btnStyle = {
    padding: '3px 8px', fontSize: 11, border: 'none', borderRadius: 'var(--radius-sm)', cursor: 'pointer',
  };

  // --- Insert form row ---
  const renderInsertForm = (afterIdx) => {
    if (insertAfterIdx !== afterIdx) return null;
    return (
      <div
        key={`insert-${afterIdx}`}
        style={{
          padding: '8px 0', borderBottom: '1px solid var(--border)',
          background: 'var(--bg-elevated)', margin: '0 -4px', paddingLeft: 4, paddingRight: 4,
          borderRadius: 'var(--radius-sm)',
        }}
      >
        <div style={{ fontSize: 10, color: 'var(--accent-cyan)', fontWeight: 600, marginBottom: 6 }}>
          + INSERT NEW LINE
        </div>
        <div style={{ display: 'flex', gap: 6, marginBottom: 6, flexWrap: 'wrap' }}>
          <select
            value={insertSpeaker}
            onChange={(e) => setInsertSpeaker(e.target.value)}
            style={{
              padding: '3px 6px', fontSize: 11, background: 'var(--bg-base)',
              border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)',
              color: 'var(--text-primary)', minWidth: 100,
            }}
          >
            {speakers.map((sp) => <option key={sp} value={sp}>{sp}</option>)}
            <option value="__new__">+ New Speaker</option>
          </select>
          {insertSpeaker === '__new__' && (
            <input
              type="text"
              placeholder="Speaker name"
              onChange={(e) => setInsertSpeaker(e.target.value)}
              style={{
                padding: '3px 6px', fontSize: 11, background: 'var(--bg-base)',
                border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)',
                color: 'var(--text-primary)', width: 100,
              }}
            />
          )}
          <input
            type="number" step="0.1" value={insertStart}
            onChange={(e) => setInsertStart(e.target.value)}
            style={{
              padding: '3px 6px', fontSize: 11, width: 60, background: 'var(--bg-base)',
              border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)',
              color: 'var(--text-primary)', fontFamily: 'var(--font-mono)',
            }}
            title="Start time (seconds)"
          />
          <span style={{ fontSize: 10, color: 'var(--text-muted)', alignSelf: 'center' }}>to</span>
          <input
            type="number" step="0.1" value={insertEnd}
            onChange={(e) => setInsertEnd(e.target.value)}
            style={{
              padding: '3px 6px', fontSize: 11, width: 60, background: 'var(--bg-base)',
              border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)',
              color: 'var(--text-primary)', fontFamily: 'var(--font-mono)',
            }}
            title="End time (seconds)"
          />
        </div>
        <textarea
          autoFocus
          value={insertText}
          onChange={(e) => setInsertText(e.target.value)}
          placeholder="Type the transcript line..."
          rows={2}
          onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); saveInsert(); } if (e.key === 'Escape') cancelInsert(); }}
          style={{
            width: '100%', fontSize: 13, lineHeight: 1.5, color: 'var(--text-primary)',
            background: 'var(--bg-base)', border: '1px solid var(--accent-cyan)',
            borderRadius: 'var(--radius-sm)', padding: '4px 8px', resize: 'vertical',
            outline: 'none', fontFamily: 'inherit', boxSizing: 'border-box', marginBottom: 6,
          }}
        />
        <div style={{ display: 'flex', gap: 4 }}>
          <button onClick={saveInsert} disabled={insertSaving || !insertText.trim()}
            style={{ ...btnStyle, background: 'var(--accent-cyan)', color: 'var(--bg-base)', fontWeight: 600, opacity: insertSaving || !insertText.trim() ? 0.5 : 1 }}>
            {insertSaving ? 'Saving...' : 'Insert'}
          </button>
          <button onClick={cancelInsert} style={{ ...btnStyle, background: 'none', color: 'var(--text-muted)', border: '1px solid var(--border)' }}>
            Cancel
          </button>
        </div>
      </div>
    );
  };

  return (
    <div>
      {/* Speaker legend — click to rename */}
      <div
        style={{
          display: 'flex',
          gap: 12,
          marginBottom: 16,
          padding: '8px 12px',
          background: 'var(--bg-elevated)',
          border: '1px solid var(--border)',
          flexWrap: 'wrap',
          alignItems: 'center',
        }}
      >
        {speakers.map((sp) => {
          const color = speakerColor(sp);
          const isEditing = editingSpeaker === sp;
          const isDeleting = deletingSpeaker === sp;
          const segmentCount = transcript.reduce(
            (n, seg) => (seg.speaker === sp ? n + 1 : n),
            0,
          );
          const otherSpeakers = speakers.filter((other) => other !== sp);
          if (isDeleting) {
            return (
              <div
                key={sp}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 6,
                  padding: '4px 8px',
                  background: 'var(--bg-base)',
                  border: `1px solid ${color}`,
                  borderRadius: 'var(--radius-sm)',
                  flexWrap: 'wrap',
                }}
              >
                <span style={{ fontSize: 11, color: 'var(--text-secondary)' }}>
                  Delete <strong style={{ color }}>{String(sp ?? '')}</strong>
                  {' ('}
                  {segmentCount}
                  {' segment'}
                  {segmentCount === 1 ? '' : 's'}
                  {')'}?
                </span>
                {otherSpeakers.length > 0 && (
                  <select
                    value={deleteReassignTo}
                    onChange={(e) => setDeleteReassignTo(e.target.value)}
                    disabled={deletingSpeakerInProgress}
                    style={{
                      fontSize: 11,
                      padding: '2px 4px',
                      background: 'var(--bg-elevated)',
                      border: '1px solid var(--border)',
                      borderRadius: 'var(--radius-sm)',
                      color: 'var(--text-primary)',
                    }}
                    title="Optionally reassign their segments to another speaker instead of deleting them"
                  >
                    <option value="">Delete segments</option>
                    {otherSpeakers.map((o) => (
                      <option key={o} value={o}>{`Reassign → ${o}`}</option>
                    ))}
                  </select>
                )}
                <button
                  onClick={confirmDeleteSpeaker}
                  disabled={deletingSpeakerInProgress}
                  style={{
                    padding: '2px 8px',
                    fontSize: 11,
                    background: deleteReassignTo ? 'var(--accent-cyan)' : '#EF4444',
                    color: '#fff',
                    border: 'none',
                    borderRadius: 'var(--radius-sm)',
                    cursor: deletingSpeakerInProgress ? 'wait' : 'pointer',
                    fontWeight: 600,
                  }}
                >
                  {deleteReassignTo ? 'Reassign' : 'Delete'}
                </button>
                <button
                  onClick={cancelDeletingSpeaker}
                  disabled={deletingSpeakerInProgress}
                  style={{
                    padding: '2px 8px',
                    fontSize: 11,
                    background: 'transparent',
                    color: 'var(--text-secondary)',
                    border: '1px solid var(--border)',
                    borderRadius: 'var(--radius-sm)',
                    cursor: 'pointer',
                  }}
                >
                  Cancel
                </button>
              </div>
            );
          }
          return (
            <div key={sp} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <div
                style={{
                  width: 10,
                  height: 10,
                  borderRadius: '50%',
                  background: color,
                  flexShrink: 0,
                }}
              />
              {isEditing ? (
                <input
                  autoFocus
                  value={editValue}
                  onChange={(e) => setEditValue(e.target.value)}
                  onKeyDown={handleKeyDown}
                  onBlur={submitRename}
                  disabled={renaming}
                  style={{
                    fontSize: 12,
                    color: color,
                    background: 'var(--bg-base)',
                    border: `1px solid ${color}`,
                    borderRadius: 3,
                    padding: '2px 6px',
                    width: Math.max(80, editValue.length * 8),
                    outline: 'none',
                    fontFamily: 'inherit',
                  }}
                />
              ) : (
                <>
                  <span
                    onClick={() => startEditing(sp)}
                    title="Click to rename speaker"
                    style={{
                      fontSize: 12,
                      color: color,
                      cursor: 'pointer',
                      borderBottom: '1px dashed transparent',
                      transition: 'border-color 0.15s',
                    }}
                    onMouseEnter={(e) => (e.target.style.borderBottomColor = color)}
                    onMouseLeave={(e) => (e.target.style.borderBottomColor = 'transparent')}
                  >
                    {String(sp ?? '')}
                  </span>
                  {jobId && (
                    <button
                      onClick={() => startDeletingSpeaker(sp)}
                      title={`Delete ${sp} (${segmentCount} segment${segmentCount === 1 ? '' : 's'})`}
                      aria-label={`Delete speaker ${sp}`}
                      style={{
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        width: 16,
                        height: 16,
                        padding: 0,
                        background: 'transparent',
                        color: 'var(--text-muted)',
                        border: '1px solid transparent',
                        borderRadius: '50%',
                        cursor: 'pointer',
                        fontSize: 13,
                        lineHeight: 1,
                      }}
                      onMouseEnter={(e) => {
                        e.currentTarget.style.color = '#EF4444';
                        e.currentTarget.style.borderColor = '#EF4444';
                      }}
                      onMouseLeave={(e) => {
                        e.currentTarget.style.color = 'var(--text-muted)';
                        e.currentTarget.style.borderColor = 'transparent';
                      }}
                    >
                      {'\u00D7'}
                    </button>
                  )}
                </>
              )}
            </div>
          );
        })}
        {/* Add speaker inline */}
        {addingSpeaker ? (
          <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
            <input
              autoFocus
              value={newSpeakerName}
              onChange={(e) => setNewSpeakerName(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && newSpeakerName.trim()) {
                  const name = newSpeakerName.trim();
                  // Assign the next palette color
                  const nextColor = DEFAULT_SPEAKER_PALETTE[speakers.length % DEFAULT_SPEAKER_PALETTE.length];
                  if (onSpeakerAdded) onSpeakerAdded(name, nextColor);
                  setAddingSpeaker(false);
                  setNewSpeakerName('');
                } else if (e.key === 'Escape') {
                  setAddingSpeaker(false);
                  setNewSpeakerName('');
                }
              }}
              onBlur={() => { setAddingSpeaker(false); setNewSpeakerName(''); }}
              placeholder="Speaker name"
              style={{
                fontSize: 12, padding: '2px 6px', width: 120,
                background: 'var(--bg-base)', border: '1px solid var(--accent-cyan)',
                borderRadius: 3, color: 'var(--text-primary)', outline: 'none',
              }}
            />
          </div>
        ) : (
          <button
            onClick={() => setAddingSpeaker(true)}
            style={{
              display: 'flex', alignItems: 'center', gap: 3,
              padding: '2px 8px', fontSize: 11, fontWeight: 600,
              background: 'transparent', color: 'var(--text-muted)',
              border: '1px dashed var(--border)', borderRadius: 3,
              cursor: 'pointer',
            }}
            title="Add a new speaker"
          >
            <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round"><line x1="12" y1="5" x2="12" y2="19" /><line x1="5" y1="12" x2="19" y2="12" /></svg>
            Add Speaker
          </button>
        )}
        {speakers.length > 0 && !addingSpeaker && (
          <span style={{ fontSize: 10, color: 'var(--text-muted)', marginLeft: 4 }}>
            click name to rename
          </span>
        )}
      </div>

      {/* Search + actions */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 16, flexWrap: 'wrap' }}>
        <input
          type="text"
          placeholder="Search transcript..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          style={{
            flex: 1,
            minWidth: 200,
            padding: '8px 12px',
            borderRadius: 'var(--radius-sm)',
          }}
        />
        <button
          onClick={() => download(toTXT(timeFiltered), 'transcript.txt')}
          style={{
            padding: '8px 12px',
            background: 'var(--bg-elevated)',
            color: 'var(--text-secondary)',
            border: '1px solid var(--border)',
            borderRadius: 'var(--radius-sm)',
            fontSize: 12,
          }}
        >
          .txt
        </button>
        <button
          onClick={() => download(toSRT(timeFiltered), 'transcript.srt')}
          style={{
            padding: '8px 12px',
            background: 'var(--bg-elevated)',
            color: 'var(--text-secondary)',
            border: '1px solid var(--border)',
            borderRadius: 'var(--radius-sm)',
            fontSize: 12,
          }}
        >
          .srt
        </button>
        <button
          onClick={copyAll}
          style={{
            padding: '8px 12px',
            background: 'var(--bg-elevated)',
            color: 'var(--text-secondary)',
            border: '1px solid var(--border)',
            borderRadius: 'var(--radius-sm)',
            fontSize: 12,
          }}
        >
          Copy All
        </button>
        {jobId && speakers.length > 0 && (
          <button
            onClick={selectedIndices.size > 0 && selectedIndices.size === filtered.reduce((s, seg) => { s.add(transcript.indexOf(seg)); return s; }, new Set()).size ? clearSelection : selectAllFiltered}
            style={{
              padding: '8px 12px',
              background: selectedIndices.size > 0 ? 'var(--accent-cyan-dim, rgba(0,217,255,0.08))' : 'var(--bg-elevated)',
              color: selectedIndices.size > 0 ? 'var(--accent-cyan)' : 'var(--text-secondary)',
              border: selectedIndices.size > 0 ? '1px solid var(--accent-cyan)' : '1px solid var(--border)',
              borderRadius: 'var(--radius-sm)',
              fontSize: 12,
              cursor: 'pointer',
              fontWeight: selectedIndices.size > 0 ? 600 : 400,
            }}
          >
            {selectedIndices.size > 0 ? `${selectedIndices.size} Selected` : 'Select All'}
          </button>
        )}
      </div>

      {/* Editable hint */}
      {jobId && (
        <div style={{ fontSize: 10, color: 'var(--text-muted)', marginBottom: 8, display: 'flex', alignItems: 'center', gap: 4, flexWrap: 'wrap' }}>
          <span style={{ color: 'var(--accent-cyan)' }}>{'\u270E'}</span>
          Click text to edit. Use speaker dropdown to reassign. Changes update subtitles for new and existing clip exports.
        </div>
      )}

      {/* Multi-select bulk action bar */}
      {jobId && selectedIndices.size > 0 && (
        <div style={{
          display: 'flex', alignItems: 'center', gap: 10, padding: '8px 12px',
          marginBottom: 8, background: 'var(--accent-cyan-dim, rgba(0,217,255,0.08))',
          border: '1px solid var(--accent-cyan)', borderRadius: 'var(--radius-sm)',
          flexWrap: 'wrap',
        }}>
          <span style={{ fontSize: 12, fontWeight: 600, color: 'var(--accent-cyan)' }}>
            {selectedIndices.size} selected
          </span>
          <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>|</span>
          <span style={{ fontSize: 11, color: 'var(--text-secondary)' }}>Set speaker:</span>
          {speakers.map((sp) => (
            <button
              key={sp}
              onClick={() => handleBulkSpeakerChange(sp)}
              disabled={bulkSaving}
              style={{
                ...btnStyle, fontSize: 11, padding: '3px 10px',
                background: 'var(--bg-elevated)', color: speakerColor(sp),
                border: `1px solid ${speakerColor(sp)}`, fontWeight: 600,
                opacity: bulkSaving ? 0.5 : 1, cursor: bulkSaving ? 'wait' : 'pointer',
              }}
            >
              {sp}
            </button>
          ))}
          <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>|</span>
          <button
            onClick={selectAllFiltered}
            style={{ ...btnStyle, fontSize: 10, background: 'var(--bg-elevated)', color: 'var(--text-secondary)', border: '1px solid var(--border)' }}
          >
            Select All
          </button>
          <button
            onClick={clearSelection}
            style={{ ...btnStyle, fontSize: 10, background: 'var(--bg-elevated)', color: 'var(--text-secondary)', border: '1px solid var(--border)' }}
          >
            Deselect
          </button>
        </div>
      )}

      {/* Segments */}
      <div
        ref={scrollContainerRef}
        tabIndex={0}
        style={{
          maxHeight: maxHeight || 500,
          overflowY: 'auto',
          overflowX: 'hidden',
          position: 'relative',
          scrollbarGutter: 'stable',
          // Stop wheel scrolls from chaining to the page when the
          // transcript reaches top/bottom — without this the page
          // scrolls instead and users perceive the transcript as "not
          // scrollable".
          overscrollBehavior: 'contain',
          // Make sure nothing (CSS resets, parent styles) accidentally
          // disables touch scrolling on the container.
          touchAction: 'pan-y',
          // Needed for keyboard scroll (PageUp/Down, arrows, space).
          outline: 'none',
        }}
      >
        {filtered.map((seg, i) => {
          const color = speakerColor(seg.speaker);
          const originalIdx = transcript.indexOf(seg);
          const isEditingSeg = editingSegIdx === originalIdx;
          const isActiveSeg = activeOriginalIdx === originalIdx;
          const highlightedText = search.trim()
            ? seg.text.replace(
                new RegExp(`(${search.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi'),
                '<mark style="background:var(--accent-cyan-dim);color:var(--accent-cyan)">$1</mark>'
              )
            : seg.text;

          const isSelected = selectedIndices.has(originalIdx);

          // Callback ref so the ref explicitly CLEARS when a row stops
          // being active (the legacy `ref={isActive ? activeSegRef : undefined}`
          // left `activeSegRef.current` pointing at a stale DOM node after
          // the row lost its active state, so the scroll effect
          // occasionally targeted a detached element during search-filter
          // changes).
          const setActiveRef = (node) => {
            if (isActiveSeg) {
              activeSegRef.current = node;
            } else if (activeSegRef.current && activeSegRef.current === node) {
              activeSegRef.current = null;
            }
          };

          return (
            <React.Fragment key={`seg-${originalIdx}`}>
              <div
                ref={setActiveRef}
                data-active={isActiveSeg ? 'true' : undefined}
                style={{
                  display: 'flex',
                  gap: isMobile ? 8 : 12,
                  padding: '8px 4px',
                  alignItems: 'flex-start',
                  borderRadius: 'var(--radius-sm)',
                  transition: 'background 0.2s, border-color 0.2s, box-shadow 0.2s',
                  ...(isActiveSeg && !isSelected ? {
                    background: 'rgba(10,132,255,0.18)',
                    borderLeft: '3px solid var(--accent-cyan)',
                    boxShadow: 'inset 0 0 0 1px rgba(10,132,255,0.35)',
                    paddingLeft: 5,
                  } : isSelected ? {
                    background: 'var(--accent-cyan-dim, rgba(0,217,255,0.12))',
                    borderLeft: '2px solid var(--accent-cyan)',
                    paddingLeft: 6,
                  } : {
                    borderLeft: '2px solid transparent',
                  }),
                }}
              >
                {/* Selection checkbox */}
                {jobId && (
                  <div
                    style={{ flexShrink: 0, display: 'flex', alignItems: 'center', paddingTop: 2 }}
                  >
                    <input
                      type="checkbox"
                      checked={isSelected}
                      onChange={(e) => toggleSelect(originalIdx, e.nativeEvent)}
                      onClick={(e) => e.stopPropagation()}
                      title="Select line (Shift+click to select range)"
                      style={{ cursor: 'pointer', accentColor: 'var(--accent-cyan)' }}
                    />
                  </div>
                )}

                {/* Speaker + time column */}
                <div style={{ minWidth: isMobile ? 80 : 110, maxWidth: isMobile ? 110 : 150, flexShrink: 0 }}>
                  {/* Speaker dropdown */}
                  {jobId && speakers.length > 1 ? (
                    <div style={{ position: 'relative', display: 'inline-block', maxWidth: '100%' }}>
                      <select
                        value={seg.speaker}
                        onChange={(e) => handleChangeSpeaker(originalIdx, e.target.value)}
                        title="Change speaker"
                        style={{
                          fontFamily: 'var(--font-mono)', fontSize: 11, color,
                          fontWeight: 600, background: 'transparent', border: 'none',
                          cursor: 'pointer', padding: '0 14px 0 0', appearance: 'none',
                          WebkitAppearance: 'none', MozAppearance: 'none',
                          maxWidth: '100%', outline: 'none',
                        }}
                      >
                        {speakers.map((sp) => <option key={sp} value={sp}>{sp}</option>)}
                      </select>
                      <span style={{
                        position: 'absolute', right: 0, top: '50%', transform: 'translateY(-50%)',
                        pointerEvents: 'none', fontSize: 8, color: 'var(--text-muted)', lineHeight: 1,
                      }}>&#9662;</span>
                    </div>
                  ) : (
                    <span
                      style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color, fontWeight: 600 }}
                    >
                      {String(seg.speaker ?? '')}
                    </span>
                  )}
                  <br />
                  <span
                    style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--text-muted)', cursor: 'pointer' }}
                    onClick={() => onSeek?.(seg.start)}
                    title="Seek to this time"
                  >
                    {formatTime(seg.start)}
                  </span>
                </div>

                {/* Text content */}
                {isEditingSeg ? (
                  <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 4 }}>
                    <textarea
                      autoFocus
                      value={editSegText}
                      onChange={(e) => setEditSegText(e.target.value)}
                      onKeyDown={handleSegKeyDown}
                      disabled={savingSegment}
                      rows={Math.max(2, Math.ceil(editSegText.length / 60))}
                      style={{
                        width: '100%', fontSize: 13, lineHeight: 1.5,
                        color: 'var(--text-primary)', background: 'var(--bg-base)',
                        border: '1px solid var(--accent-cyan)', borderRadius: 'var(--radius-sm)',
                        padding: '4px 8px', resize: 'vertical', outline: 'none',
                        fontFamily: 'inherit', boxSizing: 'border-box',
                      }}
                    />
                    <div style={{ display: 'flex', gap: 4 }}>
                      <button onClick={saveSegEdit} disabled={savingSegment}
                        style={{ ...btnStyle, background: 'var(--accent-cyan)', color: 'var(--bg-base)', fontWeight: 600 }}>
                        {savingSegment ? 'Saving...' : 'Save'}
                      </button>
                      <button onClick={cancelSegEdit}
                        style={{ ...btnStyle, background: 'none', color: 'var(--text-muted)', border: '1px solid var(--border)' }}>
                        Cancel
                      </button>
                      <span style={{ fontSize: 9, color: 'var(--text-muted)', alignSelf: 'center', marginLeft: 4 }}>
                        Enter to save, Esc to cancel
                      </span>
                    </div>
                  </div>
                ) : (
                  <div
                    onClick={(e) => startSegEdit(originalIdx, seg.text, e)}
                    title="Click to edit text"
                    className="transcript-seg-editable"
                    style={{
                      fontSize: 13, lineHeight: 1.5, color: 'var(--text-primary)',
                      cursor: 'pointer', flex: 1, display: 'flex', alignItems: 'flex-start',
                      gap: 6, padding: '2px 4px', borderRadius: 'var(--radius-sm)',
                      transition: 'background 0.15s',
                    }}
                    onMouseEnter={(e) => { e.currentTarget.style.background = 'var(--bg-elevated)'; }}
                    onMouseLeave={(e) => { e.currentTarget.style.background = 'transparent'; }}
                  >
                    <span style={{ flex: 1 }} dangerouslySetInnerHTML={{ __html: highlightedText }} />
                    <span style={{ fontSize: 10, color: 'var(--text-muted)', opacity: 0.4, flexShrink: 0, marginTop: 2 }}>
                      {'\u270E'}
                    </span>
                  </div>
                )}

                {/* Action buttons: insert + delete */}
                {jobId && (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 4, flexShrink: 0 }}>
                    <button
                      onClick={() => openInsertForm(originalIdx)}
                      title="Insert line below"
                      style={{
                        ...btnStyle, background: 'var(--bg-elevated)', color: 'var(--accent-cyan)',
                        border: '1px solid var(--border)', fontSize: 10, lineHeight: 1, padding: '3px 6px',
                        opacity: 0.7, transition: 'opacity 0.15s, background 0.15s',
                        whiteSpace: 'nowrap',
                      }}
                      onMouseEnter={(e) => { e.currentTarget.style.opacity = '1'; e.currentTarget.style.background = 'var(--accent-cyan)'; e.currentTarget.style.color = 'var(--bg-base)'; }}
                      onMouseLeave={(e) => { e.currentTarget.style.opacity = '0.7'; e.currentTarget.style.background = 'var(--bg-elevated)'; e.currentTarget.style.color = 'var(--accent-cyan)'; }}
                    >
                      + Add
                    </button>
                    <button
                      onClick={() => { if (window.confirm('Delete this transcript line?')) handleDeleteSegment(originalIdx); }}
                      title="Delete line"
                      style={{
                        ...btnStyle, background: 'transparent', color: 'var(--danger, #ff3b30)',
                        border: '1px solid var(--border)', fontSize: 12, lineHeight: 1, padding: '2px 5px',
                        opacity: 0.5, transition: 'opacity 0.15s',
                      }}
                      onMouseEnter={(e) => { e.currentTarget.style.opacity = '1'; }}
                      onMouseLeave={(e) => { e.currentTarget.style.opacity = '0.5'; }}
                    >
                      {'\u2715'}
                    </button>
                  </div>
                )}
              </div>

              {/* Hover divider to insert a line */}
              {jobId && insertAfterIdx !== originalIdx && (
                <div
                  className="transcript-insert-divider"
                  onClick={() => openInsertForm(originalIdx)}
                  style={{
                    position: 'relative',
                    height: 1,
                    background: 'var(--border)',
                    cursor: 'pointer',
                    margin: '0 -4px',
                    padding: '6px 4px',
                    backgroundClip: 'content-box',
                  }}
                >
                  <span
                    className="transcript-insert-divider-btn"
                    style={{
                      position: 'absolute', left: '50%', top: '50%',
                      transform: 'translate(-50%, -50%)',
                      fontSize: 9, fontWeight: 600, lineHeight: 1,
                      padding: '2px 10px', borderRadius: 'var(--radius-sm)',
                      background: 'var(--accent-cyan)', color: 'var(--bg-base)',
                      whiteSpace: 'nowrap', pointerEvents: 'none',
                      opacity: 0, transition: 'opacity 0.15s',
                    }}
                  >
                    + Add line
                  </span>
                </div>
              )}
              {/* Plain divider when insert form is shown for this segment */}
              {insertAfterIdx === originalIdx && (
                <div style={{ height: 1, background: 'var(--border)', margin: '0 -4px' }} />
              )}

              {/* Insert form appears after this segment */}
              {renderInsertForm(originalIdx)}
            </React.Fragment>
          );
        })}
        {filtered.length === 0 && (
          <div style={{ textAlign: 'center', padding: 24, color: 'var(--text-muted)' }}>
            No matching segments found.
          </div>
        )}

        {/* Insert at the very end */}
        {jobId && transcript.length > 0 && !timeRange && (
          <div style={{ padding: '8px 0', textAlign: 'center' }}>
            <button
              onClick={() => openInsertForm(transcript.length - 1)}
              style={{
                ...btnStyle, background: 'var(--bg-elevated)', color: 'var(--accent-cyan)',
                border: '1px solid var(--border)', fontSize: 11, padding: '4px 12px',
              }}
            >
              + Add Line at End
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
