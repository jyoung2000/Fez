import React, { useEffect, useState } from 'react';
import { Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import Dashboard from './pages/Dashboard';
import Upload from './pages/Upload';
import Analysis from './pages/Analysis';
import ViralClips from './pages/ViralClips';
import Logs from './pages/Logs';
import Settings from './pages/Settings';
import ClipSEO from './pages/ClipSEO';
import MediaLibrary from './pages/MediaLibrary';
import Login from './pages/Login';
import SharedView from './pages/SharedView';
import ToastContainer from './components/Toast';
import { EncodingProvider } from './hooks/useEncodingManager';
import { AuthProvider, RequireAuth } from './auth/AuthContext';
import { PlayerProvider } from './contexts/PlayerContext';
// Import installs localStorage monkey-patches for auto-sync
import { pullFromCloud } from './utils/cloudSync';

// ── Error Boundary — prevents blank page on unhandled errors ──
class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }
  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }
  componentDidCatch(error, info) {
    console.error('[ErrorBoundary] Uncaught error:', error, info?.componentStack);
  }
  render() {
    if (this.state.hasError) {
      return (
        <div style={{ padding: 48, textAlign: 'center', fontFamily: 'system-ui, sans-serif', color: '#ccc', background: '#0a0a0f', minHeight: '100vh' }}>
          <h2 style={{ color: '#ef4444', marginBottom: 16 }}>Something went wrong</h2>
          <p style={{ fontSize: 14, marginBottom: 12, color: '#888' }}>
            {String(this.state.error?.message || 'An unexpected error occurred.')}
          </p>
          <button
            onClick={() => { this.setState({ hasError: false, error: null }); window.location.reload(); }}
            style={{ padding: '8px 24px', fontSize: 14, background: '#00D9FF', color: '#000', border: 'none', borderRadius: 6, cursor: 'pointer' }}
          >
            Reload Page
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}

// ── Page-level Error Boundary — keeps Layout visible when a page crashes ──
class PageErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }
  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }
  componentDidCatch(error, info) {
    console.error('[PageError] Uncaught error in page:', error, info?.componentStack);
  }
  render() {
    if (this.state.hasError) {
      return (
        <div style={{ padding: 48, textAlign: 'center' }}>
          <h2 style={{ color: 'var(--danger, #ef4444)', marginBottom: 16, fontSize: 18 }}>This page encountered an error</h2>
          <p style={{ fontSize: 13, marginBottom: 16, color: 'var(--text-muted, #888)', maxWidth: 500, margin: '0 auto 16px' }}>
            {String(this.state.error?.message || 'An unexpected error occurred.')}
          </p>
          <div style={{ display: 'flex', gap: 12, justifyContent: 'center' }}>
            <button
              onClick={() => this.setState({ hasError: false, error: null })}
              style={{ padding: '8px 20px', fontSize: 13, background: 'var(--bg-elevated, #1a1a2e)', color: 'var(--text-primary, #ccc)', border: '1px solid var(--border, #333)', borderRadius: 6, cursor: 'pointer' }}
            >
              Try Again
            </button>
            <button
              onClick={() => window.location.reload()}
              style={{ padding: '8px 20px', fontSize: 13, background: 'var(--accent-cyan, #00D9FF)', color: '#000', border: 'none', borderRadius: 6, cursor: 'pointer' }}
            >
              Reload Page
            </button>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}

function Wrapped({ children }) {
  return <PageErrorBoundary>{children}</PageErrorBoundary>;
}

export default function App() {
  const [cloudReady, setCloudReady] = useState(false);

  // Pull cloud settings before rendering pages so localStorage is populated
  // Uses a hard timeout so the app never stays blank indefinitely
  useEffect(() => {
    const timeout = setTimeout(() => setCloudReady(true), 3000);
    pullFromCloud().finally(() => {
      clearTimeout(timeout);
      setCloudReady(true);
    });
  }, []);

  // Apply site customisation (title, favicon) on load
  useEffect(() => {
    fetch('/api/site-config')
      .then((r) => r.json())
      .then((cfg) => {
        if (cfg.title) document.title = cfg.title;
        if (cfg.favicon) {
          let link = document.querySelector("link[rel~='icon']");
          if (!link) { link = document.createElement('link'); link.rel = 'icon'; document.head.appendChild(link); }
          link.href = `/api/site-uploads/${cfg.favicon}`;
        }
      })
      .catch(() => {});
  }, []);

  // Wait for cloud settings before rendering pages that read localStorage
  if (!cloudReady) return null;

  return (
    <ErrorBoundary>
      <AuthProvider>
        <PlayerProvider>
        <Routes>
          {/* The login page renders outside Layout and outside RequireAuth
              so unauthenticated users can actually reach it. */}
          <Route path="/login" element={<Wrapped><Login /></Wrapped>} />
          {/* Public share-link views: no login required. The token in
              the URL is the credential. Clip-scoped variant lets the
              owner share one clip rather than the whole analysis. */}
          <Route path="/share/:token" element={<Wrapped><SharedView /></Wrapped>} />
          <Route path="/share/:token/clip/:clipId" element={<Wrapped><SharedView /></Wrapped>} />
          <Route
            path="*"
            element={
              <RequireAuth>
                <EncodingProvider>
                  <ToastContainer />
                  <Layout>
                    <Routes>
                      <Route path="/" element={<Wrapped><Dashboard /></Wrapped>} />
                      <Route path="/upload" element={<Wrapped><Upload /></Wrapped>} />
                      <Route path="/clips" element={<Wrapped><ViralClips /></Wrapped>} />
                      <Route path="/logs" element={<Wrapped><Logs /></Wrapped>} />
                      <Route path="/analysis/:jobId" element={<Wrapped><Analysis /></Wrapped>} />
                      <Route path="/seo/:jobId/:clipId" element={<Wrapped><ClipSEO /></Wrapped>} />
                      <Route path="/media" element={<Wrapped><MediaLibrary /></Wrapped>} />
                      <Route path="/settings" element={<Wrapped><Settings /></Wrapped>} />
                    </Routes>
                  </Layout>
                </EncodingProvider>
              </RequireAuth>
            }
          />
        </Routes>
        </PlayerProvider>
      </AuthProvider>
    </ErrorBoundary>
  );
}
