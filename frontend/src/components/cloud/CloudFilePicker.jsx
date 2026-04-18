/**
 * Modal file picker for a connected cloud provider.
 *
 * Behavior:
 *   - Shows a breadcrumb trail rooted at "root".
 *   - Debounced search box. Empty query -> folder browse; non-empty -> search.
 *   - Clicking a folder navigates into it. Clicking a file calls
 *     onPick(file) and the caller decides what to do with the import.
 *   - Pagination is a "Load more" button that sends the provider's
 *     next_page_token back to the server.
 *
 * The component is provider-agnostic: every request goes to
 * `/api/cloud/{provider}/browse` or `/api/cloud/{provider}/search`,
 * which return the same uniform shape regardless of Google vs Box.
 */
import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';

const SEARCH_DEBOUNCE_MS = 300;

function formatBytes(bytes) {
  if (bytes == null) return '';
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  return `${(bytes / 1024 / 1024 / 1024).toFixed(2)} GB`;
}

// Defense-in-depth — the server already filters to video MIMEs/extensions,
// but we double-check so a bug on the server side can never surface docs
// in the picker.
const VIDEO_EXT_RE = /\.(mp4|mov|mkv|webm|avi|m4v|3gp|mpe?g)$/i;

function isVideo(file) {
  if (!file) return false;
  if (typeof file.mime_type === 'string' && file.mime_type.toLowerCase().startsWith('video/')) {
    return true;
  }
  if (typeof file.name === 'string' && VIDEO_EXT_RE.test(file.name)) {
    return true;
  }
  return false;
}

export default function CloudFilePicker({ provider, providerLabel, onPick, onClose }) {
  const [path, setPath] = useState([{ id: null, name: 'Root' }]);
  const [query, setQuery] = useState('');
  const [debouncedQuery, setDebouncedQuery] = useState('');
  const [entries, setEntries] = useState({ folders: [], files: [] });
  const [nextPageToken, setNextPageToken] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [busyFileId, setBusyFileId] = useState('');
  const reqSeq = useRef(0);

  const currentFolder = path[path.length - 1];

  // Debounce the search box
  useEffect(() => {
    const handle = setTimeout(() => setDebouncedQuery(query.trim()), SEARCH_DEBOUNCE_MS);
    return () => clearTimeout(handle);
  }, [query]);

  const load = useCallback(
    async ({ append = false } = {}) => {
      const mySeq = ++reqSeq.current;
      setLoading(true);
      setError(null);
      try {
        let url;
        if (debouncedQuery) {
          const params = new URLSearchParams({ q: debouncedQuery });
          if (append && nextPageToken) params.set('page_token', nextPageToken);
          url = `/api/cloud/${provider}/search?${params.toString()}`;
        } else {
          const params = new URLSearchParams();
          if (currentFolder.id) params.set('folder_id', currentFolder.id);
          if (append && nextPageToken) params.set('page_token', nextPageToken);
          const qs = params.toString();
          url = `/api/cloud/${provider}/browse${qs ? `?${qs}` : ''}`;
        }

        const resp = await fetch(url);
        if (!resp.ok) {
          const body = await resp.json().catch(() => ({}));
          throw new Error(body.detail || `HTTP ${resp.status}`);
        }
        const page = await resp.json();
        if (mySeq !== reqSeq.current) return; // a newer request superseded this one

        const filteredFiles = (page.files || []).filter(isVideo);
        setEntries((prev) =>
          append
            ? {
                folders: [...prev.folders, ...(page.folders || [])],
                files: [...prev.files, ...filteredFiles],
              }
            : { folders: page.folders || [], files: filteredFiles }
        );
        setNextPageToken(page.next_page_token || null);
      } catch (err) {
        if (mySeq !== reqSeq.current) return;
        setError(err.message || 'Failed to load');
      } finally {
        if (mySeq === reqSeq.current) setLoading(false);
      }
    },
    [provider, currentFolder.id, debouncedQuery, nextPageToken]
  );

  // Reload when the folder or the debounced query changes.
  useEffect(() => {
    setEntries({ folders: [], files: [] });
    setNextPageToken(null);
    load({ append: false });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [provider, currentFolder.id, debouncedQuery]);

  const handleFolderClick = useCallback((folder) => {
    setQuery('');
    setPath((prev) => [...prev, { id: folder.id, name: folder.name }]);
  }, []);

  const handleCrumbClick = useCallback((index) => {
    setQuery('');
    setPath((prev) => prev.slice(0, index + 1));
  }, []);

  const handleFilePick = useCallback(
    async (file) => {
      setBusyFileId(file.id);
      try {
        await onPick(file);
      } finally {
        setBusyFileId('');
      }
    },
    [onPick]
  );

  const overlayStyle = useMemo(
    () => ({
      position: 'fixed',
      inset: 0,
      background: 'rgba(0, 0, 0, 0.65)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      zIndex: 10000,
      padding: 16,
    }),
    []
  );

  return (
    <div style={overlayStyle} onClick={(e) => e.target === e.currentTarget && onClose?.()}>
      <div
        style={{
          background: 'var(--bg-panel)',
          border: '1px solid var(--border)',
          borderRadius: 'var(--radius-lg)',
          width: 'min(720px, 100%)',
          maxHeight: '85vh',
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden',
        }}
      >
        {/* Header */}
        <div
          style={{
            padding: '12px 16px',
            borderBottom: '1px solid var(--border)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
          }}
        >
          <div style={{ fontWeight: 600 }}>
            {providerLabel || provider}
            <span style={{ color: 'var(--text-muted)', fontWeight: 400, marginLeft: 8, fontSize: 12 }}>
              videos only
            </span>
          </div>
          <button
            type="button"
            onClick={onClose}
            style={{
              background: 'transparent',
              border: 'none',
              color: 'var(--text-secondary)',
              fontSize: 18,
              cursor: 'pointer',
            }}
            aria-label="Close"
          >
            &times;
          </button>
        </div>

        {/* Breadcrumbs + search */}
        <div
          style={{
            padding: '10px 16px',
            borderBottom: '1px solid var(--border)',
            display: 'flex',
            flexDirection: 'column',
            gap: 8,
          }}
        >
          <nav
            aria-label="breadcrumb"
            style={{
              fontSize: 12,
              color: 'var(--text-secondary)',
              display: 'flex',
              gap: 4,
              flexWrap: 'wrap',
            }}
          >
            {path.map((crumb, idx) => (
              <React.Fragment key={`${crumb.id || 'root'}-${idx}`}>
                {idx > 0 && <span>/</span>}
                <button
                  type="button"
                  onClick={() => handleCrumbClick(idx)}
                  style={{
                    background: 'transparent',
                    border: 'none',
                    color:
                      idx === path.length - 1
                        ? 'var(--text-primary)'
                        : 'var(--accent-cyan)',
                    cursor: idx === path.length - 1 ? 'default' : 'pointer',
                    padding: 0,
                    fontSize: 12,
                  }}
                  disabled={idx === path.length - 1}
                >
                  {crumb.name}
                </button>
              </React.Fragment>
            ))}
          </nav>
          <input
            type="text"
            placeholder="Search videos..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            style={{
              padding: '8px 10px',
              fontSize: 13,
              background: 'var(--bg-input, var(--bg-panel))',
              color: 'var(--text-primary)',
              border: '1px solid var(--border)',
              borderRadius: 'var(--radius-sm)',
              width: '100%',
              boxSizing: 'border-box',
            }}
          />
        </div>

        {/* Body */}
        <div style={{ padding: 12, overflowY: 'auto', flex: 1 }}>
          {error && (
            <div style={{ padding: 8, color: 'var(--accent-red, #f66)', fontSize: 13 }}>
              {error}
            </div>
          )}

          {!loading && !error && entries.folders.length === 0 && entries.files.length === 0 && (
            <div style={{ padding: 16, textAlign: 'center', color: 'var(--text-muted)', fontSize: 13 }}>
              {debouncedQuery ? 'No matching videos.' : 'This folder has no videos.'}
            </div>
          )}

          {/* Folders */}
          {entries.folders.map((folder) => (
            <button
              key={`folder-${folder.id}`}
              type="button"
              onClick={() => handleFolderClick(folder)}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 10,
                padding: '8px 10px',
                width: '100%',
                background: 'transparent',
                border: 'none',
                borderRadius: 'var(--radius-sm)',
                cursor: 'pointer',
                color: 'var(--text-primary)',
                textAlign: 'left',
                fontSize: 13,
              }}
            >
              <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--accent-cyan)' }}>[+]</span>
              <span style={{ fontWeight: 500 }}>{folder.name}</span>
            </button>
          ))}

          {/* Files */}
          {entries.files.map((file) => (
            <div
              key={`file-${file.id}`}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 10,
                padding: '8px 10px',
                borderTop: '1px solid var(--border)',
              }}
            >
              {file.thumbnail_url ? (
                <img
                  src={file.thumbnail_url}
                  alt=""
                  style={{
                    width: 48,
                    height: 32,
                    objectFit: 'cover',
                    borderRadius: 2,
                    background: '#000',
                  }}
                />
              ) : (
                <div
                  style={{
                    width: 48,
                    height: 32,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    background: 'var(--bg-hover, #111)',
                    borderRadius: 2,
                    fontFamily: 'var(--font-mono)',
                    color: 'var(--text-muted)',
                    fontSize: 11,
                  }}
                >
                  VID
                </div>
              )}
              <div style={{ flex: 1, minWidth: 0 }}>
                <div
                  style={{
                    fontSize: 13,
                    color: 'var(--text-primary)',
                    whiteSpace: 'nowrap',
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                  }}
                  title={file.name}
                >
                  {file.name}
                </div>
                <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                  {formatBytes(file.size)}
                  {file.modified_at ? ` • ${new Date(file.modified_at).toLocaleDateString()}` : ''}
                </div>
              </div>
              <button
                type="button"
                onClick={() => handleFilePick(file)}
                disabled={busyFileId === file.id}
                style={{
                  padding: '6px 12px',
                  fontSize: 12,
                  background: 'var(--accent-cyan)',
                  color: '#000',
                  border: 'none',
                  borderRadius: 'var(--radius-sm)',
                  cursor: busyFileId === file.id ? 'default' : 'pointer',
                  fontWeight: 600,
                }}
              >
                {busyFileId === file.id ? 'Importing...' : 'Select'}
              </button>
            </div>
          ))}

          {nextPageToken && !loading && (
            <div style={{ textAlign: 'center', marginTop: 12 }}>
              <button
                type="button"
                onClick={() => load({ append: true })}
                style={{
                  padding: '8px 16px',
                  fontSize: 12,
                  background: 'transparent',
                  color: 'var(--text-secondary)',
                  border: '1px solid var(--border)',
                  borderRadius: 'var(--radius-sm)',
                  cursor: 'pointer',
                }}
              >
                Load more
              </button>
            </div>
          )}

          {loading && (
            <div style={{ padding: 12, textAlign: 'center', color: 'var(--text-muted)', fontSize: 12 }}>
              Loading...
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
