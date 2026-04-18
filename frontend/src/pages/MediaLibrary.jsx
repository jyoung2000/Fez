import React, { useState, useCallback, useRef, useEffect } from 'react';
import useTimelineStore from '../stores/timelineStore';
import useResponsive from '../hooks/useResponsive';

const ACCEPTED_IMAGE = '.png,.jpg,.jpeg,.gif,.webp,.svg';
const ACCEPTED_AUDIO = '.mp3,.wav,.ogg,.aac,.flac,.m4a';
const ACCEPTED_VIDEO = '.mp4,.mov,.avi,.mkv,.webm';
const ACCEPTED_FONT = '.ttf,.otf,.woff,.woff2';

function formatDuration(s) {
  if (!s || isNaN(s) || s <= 0) return '--';
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return `${m}:${sec.toString().padStart(2, '0')}`;
}

function TrashIcon({ size = 14 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="3 6 5 6 21 6" />
      <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
      <line x1="10" y1="11" x2="10" y2="17" />
      <line x1="14" y1="11" x2="14" y2="17" />
    </svg>
  );
}

function SearchIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="11" cy="11" r="8" /><line x1="21" y1="21" x2="16.65" y2="16.65" />
    </svg>
  );
}

function FontIcon({ size = 28 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" style={{ opacity: 0.35 }}>
      <polyline points="4 7 4 4 20 4 20 7" />
      <line x1="9" y1="20" x2="15" y2="20" />
      <line x1="12" y1="4" x2="12" y2="20" />
    </svg>
  );
}

function VideoPlaceholder() {
  return (
    <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" style={{ opacity: 0.25 }}>
      <rect x="2" y="2" width="20" height="20" rx="2.18" ry="2.18" />
      <line x1="7" y1="2" x2="7" y2="22" />
      <line x1="17" y1="2" x2="17" y2="22" />
      <line x1="2" y1="12" x2="22" y2="12" />
      <line x1="2" y1="7" x2="7" y2="7" />
      <line x1="2" y1="17" x2="7" y2="17" />
      <line x1="17" y1="7" x2="22" y2="7" />
      <line x1="17" y1="17" x2="22" y2="17" />
    </svg>
  );
}

function AudioPlaceholder() {
  return (
    <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" style={{ opacity: 0.25 }}>
      <path d="M9 18V5l12-2v13" />
      <circle cx="6" cy="18" r="3" />
      <circle cx="18" cy="16" r="3" />
    </svg>
  );
}

function ImagePlaceholder() {
  return (
    <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" style={{ opacity: 0.25 }}>
      <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
      <circle cx="8.5" cy="8.5" r="1.5" />
      <polyline points="21 15 16 10 5 21" />
    </svg>
  );
}

export default function MediaLibrary() {
  const { isMobile } = useResponsive();
  const mediaLibrary = useTimelineStore((s) => s.mediaLibrary);
  const addMedia = useTimelineStore((s) => s.addMedia);
  const updateMedia = useTimelineStore((s) => s.updateMedia);
  const removeMedia = useTimelineStore((s) => s.removeMedia);
  const removeMediaBatch = useTimelineStore((s) => s.removeMediaBatch);
  const fileRef = useRef(null);
  const fontRef = useRef(null);
  const [filter, setFilter] = useState('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [selected, setSelected] = useState(new Set());
  const [selectMode, setSelectMode] = useState(false);
  const [fonts, setFonts] = useState([]);
  const [isLoading, setIsLoading] = useState(true);

  // Load media from backend global library on mount
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const res = await fetch('/api/media/list');
        if (!res.ok) throw new Error('Failed to load');
        const data = await res.json();
        if (cancelled) return;
        const existing = useTimelineStore.getState().mediaLibrary;
        const existingIds = new Set(existing.map(m => m.id));
        // Add any backend items not already in the store
        for (const item of data.items || []) {
          if (!existingIds.has(item.id)) {
            addMedia({
              id: item.id,
              type: item.type,
              filename: item.filename,
              url: item.url,
              thumbnailUrl: item.type === 'image' ? item.url : '',
              duration: 0,
            });
          }
        }
        // Remove any store items that are no longer on the backend
        // (blob URLs that were never persisted are cleared)
        const backendIds = new Set((data.items || []).map(i => i.id));
        const stale = existing.filter(m => !backendIds.has(m.id) && !m.url?.startsWith('blob:'));
        for (const m of stale) {
          removeMedia(m.id);
        }
      } catch (err) {
        console.warn('Failed to load media library from backend:', err);
      } finally {
        if (!cancelled) setIsLoading(false);
      }
    })();
    return () => { cancelled = true; };
  }, []);

  // Fetch uploaded fonts
  useEffect(() => {
    fetch('/api/fonts')
      .then((r) => r.ok ? r.json() : [])
      .then(setFonts)
      .catch(() => {});
  }, []);

  // Regenerate thumbnails for video items that are missing them
  useEffect(() => {
    const videosWithoutThumbs = mediaLibrary.filter(
      (m) => m.type === 'video' && !m.thumbnailUrl && m.url
    );
    videosWithoutThumbs.forEach((media) => {
      const video = document.createElement('video');
      video.muted = true;
      video.playsInline = true;
      video.preload = 'auto';
      let done = false;
      const captureFrame = () => {
        if (done) return false;
        try {
          const vw = video.videoWidth || 0;
          const vh = video.videoHeight || 0;
          if (vw === 0 || vh === 0) return false;
          const canvas = document.createElement('canvas');
          const w = Math.min(vw, 240);
          const h = Math.round(w * vh / vw);
          canvas.width = w;
          canvas.height = h;
          canvas.getContext('2d').drawImage(video, 0, 0, w, h);
          const dataUrl = canvas.toDataURL('image/jpeg', 0.7);
          if (dataUrl.length > 500) {
            done = true;
            updateMedia(media.id, { thumbnailUrl: dataUrl, duration: video.duration || media.duration });
            video.removeAttribute('src');
            video.load(); // free memory
            return true;
          }
        } catch { /* ignore */ }
        return false;
      };
      // Try multiple strategies for reliable frame capture
      video.onseeked = () => captureFrame();
      video.onloadeddata = () => {
        // Some browsers paint the first frame on loadeddata without needing a seek
        if (!done && video.readyState >= 2) {
          if (!captureFrame()) {
            // Seek to a later frame
            video.currentTime = Math.min(0.5, (video.duration || 1) * 0.1);
          }
        }
      };
      video.onloadedmetadata = () => {
        if (!media.duration && video.duration) {
          updateMedia(media.id, { duration: video.duration });
        }
        // Trigger seek for onseeked fallback
        video.currentTime = Math.min(0.5, (video.duration || 1) * 0.1);
      };
      video.onerror = () => { done = true; }; // give up on broken sources
      video.src = media.url;
      video.load();
      // Fallback: retry capture at intervals (1s, 3s, 6s)
      const retries = [1000, 3000, 6000];
      retries.forEach((delay) => {
        setTimeout(() => { if (!done) captureFrame(); }, delay);
      });
    });
  }, [mediaLibrary.length, updateMedia]); // re-run when library size changes

  // Apply filters: type + search query
  const filtered = (() => {
    let items = filter === 'all' || filter === 'font'
      ? mediaLibrary
      : mediaLibrary.filter((m) => m.type === filter);
    if (searchQuery.trim()) {
      const q = searchQuery.trim().toLowerCase();
      items = items.filter((m) => (m.filename || '').toLowerCase().includes(q));
    }
    return items;
  })();

  // Filtered fonts for the font tab
  const filteredFonts = (() => {
    if (filter !== 'all' && filter !== 'font') return [];
    let f = fonts;
    if (searchQuery.trim()) {
      const q = searchQuery.trim().toLowerCase();
      f = f.filter((font) => (font.name || '').toLowerCase().includes(q));
    }
    return f;
  })();

  const handleFiles = useCallback(async (fileList) => {
    const files = Array.from(fileList);
    for (const file of files) {
      const ext = file.name.split('.').pop().toLowerCase();
      let type = 'video';
      if (['png', 'jpg', 'jpeg', 'gif', 'webp', 'svg'].includes(ext)) type = 'image';
      else if (['mp3', 'wav', 'ogg', 'aac', 'flac', 'm4a'].includes(ext)) type = 'audio';

      // Upload to backend first so it persists
      let backendUrl = '';
      let mediaId = '';
      try {
        const formData = new FormData();
        formData.append('file', file);
        const res = await fetch('/api/media/upload', { method: 'POST', body: formData });
        if (res.ok) {
          const data = await res.json();
          backendUrl = data.url;
          mediaId = data.id;
        }
      } catch (err) {
        console.warn('Backend upload failed, using local blob:', err);
      }

      const url = backendUrl || URL.createObjectURL(file);

      // Generate thumbnail for video files
      if (type === 'video') {
        const video = document.createElement('video');
        video.muted = true;
        video.playsInline = true;
        video.preload = 'auto';
        let added = false;
        const addOnce = (thumbUrl, dur) => {
          if (added) return;
          added = true;
          addMedia({ id: mediaId || undefined, type, filename: file.name, url, thumbnailUrl: thumbUrl, duration: dur });
        };
        const captureFrame = () => {
          try {
            const canvas = document.createElement('canvas');
            const w = Math.min(video.videoWidth || 180, 240);
            const h = Math.round(w * (video.videoHeight || 100) / (video.videoWidth || 180));
            canvas.width = w;
            canvas.height = h;
            canvas.getContext('2d').drawImage(video, 0, 0, w, h);
            const dataUrl = canvas.toDataURL('image/jpeg', 0.7);
            if (dataUrl.length > 500) {
              addOnce(dataUrl, video.duration || 0);
              return true;
            }
          } catch { /* fall through */ }
          return false;
        };
        video.onseeked = () => {
          if (!captureFrame() && video.currentTime > 0) {
            video.currentTime = 0;
          } else if (!added) {
            addOnce('', video.duration || 0);
          }
        };
        video.onloadedmetadata = () => {
          const seekTime = Math.min(0.5, (video.duration || 1) * 0.1);
          video.currentTime = seekTime;
        };
        video.onerror = () => {
          addOnce('', 0);
        };
        video.src = url;
        video.load();
        setTimeout(() => {
          if (!added) {
            captureFrame();
            if (!added) addOnce('', video.duration || 0);
          }
        }, 10000);
      } else {
        addMedia({
          id: mediaId || undefined,
          type,
          filename: file.name,
          url,
          thumbnailUrl: type === 'image' ? url : '',
          duration: 0,
        });
      }
    }
  }, [addMedia]);

  const handleFontUpload = useCallback(async (files) => {
    for (const file of Array.from(files)) {
      const formData = new FormData();
      formData.append('file', file);
      try {
        const res = await fetch('/api/fonts/upload', { method: 'POST', body: formData });
        if (res.ok) {
          const updated = await fetch('/api/fonts').then(r => r.ok ? r.json() : fonts);
          setFonts(updated);
        }
      } catch { /* ignore */ }
    }
  }, [fonts]);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    handleFiles(e.dataTransfer.files);
  }, [handleFiles]);

  const handleDragStart = useCallback((e, media) => {
    if (selectMode) { e.preventDefault(); return; }
    e.dataTransfer.setData('application/x-clipai-media', JSON.stringify(media));
    e.dataTransfer.effectAllowed = 'copy';
  }, [selectMode]);

  const toggleSelect = useCallback((id) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }, []);

  const selectAll = useCallback(() => {
    setSelected(new Set(filtered.map((m) => m.id)));
  }, [filtered]);

  const deselectAll = useCallback(() => {
    setSelected(new Set());
  }, []);

  const deleteSelected = useCallback(async () => {
    if (selected.size === 0) return;
    const count = selected.size;
    if (!window.confirm(`Delete ${count} item${count > 1 ? 's' : ''} from the media library? Any timeline items referencing them will also be removed.`)) return;
    // Delete from backend
    for (const id of selected) {
      try {
        await fetch(`/api/media/${id}`, { method: 'DELETE' });
      } catch { /* ignore */ }
    }
    removeMediaBatch(Array.from(selected));
    setSelected(new Set());
    setSelectMode(false);
  }, [selected, removeMediaBatch]);

  const exitSelectMode = useCallback(() => {
    setSelectMode(false);
    setSelected(new Set());
  }, []);

  const enterSelectMode = useCallback(() => {
    setSelectMode(true);
    setSelected(new Set());
  }, []);

  const deleteFont = useCallback(async (fontName) => {
    if (!window.confirm(`Delete font "${fontName}"?`)) return;
    try {
      await fetch(`/api/fonts/${encodeURIComponent(fontName)}`, { method: 'DELETE' });
      setFonts(prev => prev.filter(f => f.name !== fontName));
    } catch { /* ignore */ }
  }, []);

  const allFilteredSelected = filtered.length > 0 && filtered.every((m) => selected.has(m.id));
  const showFonts = filter === 'all' || filter === 'font';

  return (
    <div style={{ maxWidth: isMobile ? '100%' : 960, margin: '0 auto' }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 16, flexWrap: 'wrap', gap: 8 }}>
        <h2 style={{ fontSize: isMobile ? 18 : 20, margin: 0 }}>Media Library</h2>
        <div style={{ display: 'flex', gap: 8 }}>
          {selectMode ? (
            <>
              <button
                onClick={allFilteredSelected ? deselectAll : selectAll}
                style={{
                  padding: '8px 14px',
                  background: 'var(--bg-elevated)',
                  color: 'var(--text-primary)',
                  border: '1px solid var(--border)',
                  borderRadius: 'var(--radius-sm)',
                  fontSize: 12,
                  cursor: 'pointer',
                }}
              >
                {allFilteredSelected ? 'Deselect All' : 'Select All'}
              </button>
              <button
                onClick={deleteSelected}
                disabled={selected.size === 0}
                style={{
                  padding: '8px 14px',
                  background: selected.size > 0 ? '#FF375F' : 'var(--bg-elevated)',
                  color: selected.size > 0 ? '#fff' : 'var(--text-muted)',
                  border: 'none',
                  borderRadius: 'var(--radius-sm)',
                  fontSize: 12,
                  fontWeight: 600,
                  cursor: selected.size > 0 ? 'pointer' : 'default',
                }}
              >
                Delete{selected.size > 0 ? ` (${selected.size})` : ''}
              </button>
              <button
                onClick={exitSelectMode}
                style={{
                  padding: '8px 14px',
                  background: 'var(--bg-elevated)',
                  color: 'var(--text-secondary)',
                  border: '1px solid var(--border)',
                  borderRadius: 'var(--radius-sm)',
                  fontSize: 12,
                  cursor: 'pointer',
                }}
              >
                Cancel
              </button>
            </>
          ) : (
            <>
              {mediaLibrary.length > 0 && (
                <button
                  onClick={enterSelectMode}
                  style={{
                    padding: '8px 14px',
                    background: 'var(--bg-elevated)',
                    color: 'var(--text-secondary)',
                    border: '1px solid var(--border)',
                    borderRadius: 'var(--radius-sm)',
                    fontSize: 12,
                    cursor: 'pointer',
                  }}
                >
                  Select
                </button>
              )}
              <button
                onClick={() => fileRef.current?.click()}
                style={{
                  padding: '8px 16px',
                  background: 'var(--accent-cyan)',
                  color: 'var(--bg-base)',
                  border: 'none',
                  borderRadius: 'var(--radius-sm)',
                  fontSize: 13,
                  fontWeight: 600,
                  cursor: 'pointer',
                }}
              >
                + Add Media
              </button>
            </>
          )}
        </div>
        <input
          ref={fileRef}
          type="file"
          accept={`${ACCEPTED_VIDEO},${ACCEPTED_IMAGE},${ACCEPTED_AUDIO}`}
          multiple
          onChange={(e) => handleFiles(e.target.files)}
          style={{ display: 'none' }}
        />
        <input
          ref={fontRef}
          type="file"
          accept={ACCEPTED_FONT}
          multiple
          onChange={(e) => handleFontUpload(e.target.files)}
          style={{ display: 'none' }}
        />
      </div>

      {/* Search bar */}
      <div style={{ position: 'relative', marginBottom: 12 }}>
        <div style={{
          position: 'absolute', left: 14, top: '50%', transform: 'translateY(-50%)',
          color: 'var(--text-muted)', fontSize: 15, pointerEvents: 'none', lineHeight: 1,
        }}>
          <SearchIcon />
        </div>
        <input
          type="text"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          placeholder="Search media by name..."
          style={{
            width: '100%',
            padding: '10px 14px 10px 40px',
            fontSize: 14,
            background: 'var(--bg-panel)',
            border: '1px solid var(--border)',
            borderRadius: 'var(--radius-md)',
            color: 'var(--text-primary)',
            outline: 'none',
            boxSizing: 'border-box',
            transition: 'border-color 0.2s ease, box-shadow 0.2s ease',
          }}
          onFocus={(e) => {
            e.target.style.borderColor = 'var(--accent-cyan)';
            e.target.style.boxShadow = '0 0 0 3px var(--accent-cyan-dim)';
          }}
          onBlur={(e) => {
            e.target.style.borderColor = 'var(--border)';
            e.target.style.boxShadow = 'none';
          }}
        />
        {searchQuery && (
          <button
            onClick={() => setSearchQuery('')}
            style={{
              position: 'absolute', right: 10, top: '50%', transform: 'translateY(-50%)',
              background: 'none', border: 'none', color: 'var(--text-muted)',
              fontSize: 16, cursor: 'pointer', padding: '2px 6px',
            }}
          >
            x
          </button>
        )}
      </div>

      {/* Filter tabs */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 16 }}>
        {['all', 'video', 'audio', 'image', 'font'].map((f) => (
          <button
            key={f}
            onClick={() => setFilter(f)}
            style={{
              padding: '5px 12px',
              fontSize: 12,
              fontWeight: filter === f ? 600 : 400,
              background: filter === f ? 'var(--accent-cyan-dim)' : 'var(--bg-elevated)',
              color: filter === f ? 'var(--accent-cyan)' : 'var(--text-secondary)',
              border: `1px solid ${filter === f ? 'var(--accent-cyan)' : 'var(--border)'}`,
              borderRadius: 'var(--radius-sm)',
              cursor: 'pointer',
              textTransform: 'capitalize',
            }}
          >
            {f === 'font' ? 'Fonts' : f}
          </button>
        ))}
      </div>

      {/* Unified media + fonts grid */}
      {(
        <div
          onDragOver={(e) => { e.preventDefault(); e.dataTransfer.dropEffect = 'copy'; }}
          onDrop={handleDrop}
          style={{
            display: 'grid',
            gridTemplateColumns: isMobile ? 'repeat(2, 1fr)' : 'repeat(auto-fill, minmax(180px, 1fr))',
            gap: 12,
            minHeight: 200,
          }}
        >
          {filtered.length === 0 && !(showFonts && filteredFonts.length > 0) && (
            <div style={{
              gridColumn: '1 / -1',
              textAlign: 'center',
              padding: '48px 24px',
              color: 'var(--text-muted)',
              fontSize: 13,
              border: '2px dashed var(--border)',
              borderRadius: 'var(--radius-lg)',
            }}>
              {searchQuery ? 'No media matching your search.' : 'No media files. Drag and drop files here or click "Add Media".'}
            </div>
          )}
          {filtered.map((media) => {
            const isSelected = selected.has(media.id);
            return (
              <div
                key={media.id}
                draggable={!selectMode}
                onDragStart={(e) => handleDragStart(e, media)}
                onClick={selectMode ? () => toggleSelect(media.id) : undefined}
                style={{
                  background: 'var(--bg-panel)',
                  border: `2px solid ${isSelected ? 'var(--accent-cyan)' : 'var(--border)'}`,
                  borderRadius: 'var(--radius-sm)',
                  overflow: 'hidden',
                  cursor: selectMode ? 'pointer' : 'grab',
                  transition: 'border-color 0.15s, box-shadow 0.15s',
                  boxShadow: isSelected ? '0 0 0 2px rgba(0, 200, 255, 0.2)' : 'none',
                  position: 'relative',
                }}
              >
                {/* Selection checkbox overlay */}
                {selectMode && (
                  <div style={{
                    position: 'absolute',
                    top: 6,
                    left: 6,
                    zIndex: 2,
                    width: 20,
                    height: 20,
                    borderRadius: 4,
                    background: isSelected ? 'var(--accent-cyan)' : 'rgba(0,0,0,0.5)',
                    border: `2px solid ${isSelected ? 'var(--accent-cyan)' : 'rgba(255,255,255,0.4)'}`,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    color: '#fff',
                    fontSize: 12,
                    fontWeight: 700,
                    pointerEvents: 'none',
                  }}>
                    {isSelected ? '\u2713' : ''}
                  </div>
                )}

                {/* Thumbnail area */}
                <div style={{
                  height: 110,
                  background: 'var(--bg-elevated)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  overflow: 'hidden',
                }}>
                  {media.thumbnailUrl ? (
                    <img src={media.thumbnailUrl} alt="" style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
                  ) : (
                    media.type === 'video' ? <VideoPlaceholder /> :
                    media.type === 'audio' ? <AudioPlaceholder /> :
                    <ImagePlaceholder />
                  )}
                </div>

                {/* Info area */}
                <div style={{ padding: '8px 10px' }}>
                  <div style={{
                    fontSize: 11,
                    fontWeight: 600,
                    color: 'var(--text-primary)',
                    whiteSpace: 'nowrap',
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    marginBottom: 4,
                  }} title={media.filename}>
                    {media.filename}
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                    <span style={{
                      fontSize: 10,
                      color: 'var(--text-muted)',
                      textTransform: 'uppercase',
                      fontFamily: 'var(--font-mono)',
                    }}>
                      {media.type} {media.duration > 0 ? `| ${formatDuration(media.duration)}` : ''}
                    </span>
                    {!selectMode && (
                      <button
                        onClick={async (e) => {
                          e.stopPropagation();
                          if (window.confirm(`Delete "${media.filename}" from the media library?`)) {
                            try { await fetch(`/api/media/${media.id}`, { method: 'DELETE' }); } catch { /* ignore */ }
                            removeMedia(media.id);
                          }
                        }}
                        style={{
                          background: 'none',
                          border: 'none',
                          color: 'var(--text-muted)',
                          cursor: 'pointer',
                          padding: '2px 4px',
                          borderRadius: 'var(--radius-xs)',
                          display: 'flex',
                          alignItems: 'center',
                        }}
                        title="Delete"
                      >
                        <TrashIcon size={13} />
                      </button>
                    )}
                  </div>
                </div>
              </div>
            );
          })}

          {/* Font cards — inline alongside media items */}
          {showFonts && (
            <>
              {/* Upload font card */}
              <div
                onClick={() => fontRef.current?.click()}
                style={{
                  background: 'var(--bg-panel)',
                  border: '2px dashed var(--border)',
                  borderRadius: 'var(--radius-sm)',
                  overflow: 'hidden',
                  cursor: 'pointer',
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  justifyContent: 'center',
                  padding: '24px 12px',
                  gap: 8,
                  minHeight: 120,
                  transition: 'border-color 0.15s',
                }}
              >
                <span style={{ fontSize: 24, color: 'var(--text-muted)' }}>+</span>
                <span style={{ fontSize: 11, color: 'var(--text-muted)', textAlign: 'center' }}>
                  Upload Font
                </span>
                <span style={{ fontSize: 9, color: 'var(--text-muted)', opacity: 0.6 }}>
                  TTF, OTF, WOFF, WOFF2
                </span>
              </div>

              {filteredFonts.map((font) => (
                <div
                  key={`font-${font.name}`}
                  style={{
                    background: 'var(--bg-panel)',
                    border: '2px solid var(--border)',
                    borderRadius: 'var(--radius-sm)',
                    overflow: 'hidden',
                    position: 'relative',
                  }}
                >
                  <div style={{
                    height: 110,
                    background: 'var(--bg-elevated)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    overflow: 'hidden',
                  }}>
                    <FontIcon size={32} />
                  </div>
                  <div style={{ padding: '8px 10px' }}>
                    <div style={{
                      fontSize: 11,
                      fontWeight: 600,
                      color: 'var(--text-primary)',
                      whiteSpace: 'nowrap',
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                      marginBottom: 4,
                    }} title={font.name}>
                      {font.name}
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                      <span style={{
                        fontSize: 10,
                        color: 'var(--text-muted)',
                        textTransform: 'uppercase',
                        fontFamily: 'var(--font-mono)',
                      }}>
                        Font
                      </span>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          deleteFont(font.name);
                        }}
                        style={{
                          background: 'none',
                          border: 'none',
                          color: 'var(--text-muted)',
                          cursor: 'pointer',
                          padding: '2px 4px',
                          borderRadius: 'var(--radius-xs)',
                          display: 'flex',
                          alignItems: 'center',
                        }}
                        title="Delete font"
                      >
                        <TrashIcon size={13} />
                      </button>
                    </div>
                  </div>
                </div>
              ))}
            </>
          )}
        </div>
      )}

      <p style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 16, lineHeight: 1.5 }}>
        Drag items from here onto the multi-track timeline in the video editor.
        Supported formats: video (MP4, MOV, WebM), audio (MP3, WAV, OGG), images (PNG, JPG, GIF, WebP, SVG).
      </p>
    </div>
  );
}
