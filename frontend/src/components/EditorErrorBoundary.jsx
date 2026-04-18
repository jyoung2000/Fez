import React from 'react';

/**
 * EditorErrorBoundary
 *
 * Reusable error boundary for every major editor surface
 * (VideoPlayer, VideoEditor, Timeline, ClipSettingsPanel,
 * PropertiesPanel, EffectsPanel, ExportDialog, ClipPreview).
 *
 * Goals:
 *   1. Never let one panel's crash blank out the whole page.
 *   2. Show the user a clear "what broke + retry" UI instead of a
 *      silent white square.
 *   3. Surface the failing component name + first stack frame so it
 *      ends up in the browser console (and a future telemetry sink)
 *      with context, not as an anonymous "x is undefined".
 *
 * Usage:
 *
 *   <EditorErrorBoundary name="VideoEditor">
 *     <VideoEditor ... />
 *   </EditorErrorBoundary>
 *
 * Pass an ``onReset`` callback when the parent has state that should
 * be cleared on retry (e.g. drop the broken clip selection).
 */
export default class EditorErrorBoundary extends React.Component {
  state = { hasError: false, error: null, retryKey: 0 };

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, info) {
    const name = this.props.name || 'EditorComponent';
    // Log a single, scoped line so the panel name + stack are easy to
    // find. We deliberately avoid console.error (which the dev tools
    // mark as "uncaught" + adds noise) and use console.warn with the
    // panel name as the leading tag.
    // eslint-disable-next-line no-console
    console.warn(
      `[EditorErrorBoundary:${name}] ${error?.message || 'render failed'}`,
      (info && info.componentStack && info.componentStack.split('\n').slice(0, 5).join('\n')) || '',
    );
  }

  handleRetry = () => {
    // Bump retryKey so the children remount. Calls ``onReset`` first so
    // the parent can scrub any state that triggered the crash (e.g.
    // a corrupted clip).
    try { this.props.onReset && this.props.onReset(); } catch { /* noop */ }
    this.setState((s) => ({ hasError: false, error: null, retryKey: s.retryKey + 1 }));
  };

  render() {
    if (!this.state.hasError) {
      // Wrap children in a fragment with a key so retry forces remount
      // without changing the parent tree.
      return (
        <React.Fragment key={this.state.retryKey}>
          {this.props.children}
        </React.Fragment>
      );
    }
    const name = this.props.name || 'this panel';
    const msg = String(this.state.error?.message || 'Unknown error');
    return (
      <div
        role="alert"
        style={{
          padding: this.props.compact ? 12 : 24,
          margin: 8,
          background: 'var(--bg-elevated, rgba(255,159,10,0.06))',
          border: '1px solid var(--accent-amber, #FF9F0A)',
          borderRadius: 8,
          color: 'var(--text-primary, #eee)',
          fontSize: this.props.compact ? 12 : 13,
          textAlign: 'center',
          maxWidth: 560,
          marginLeft: 'auto',
          marginRight: 'auto',
        }}
      >
        <div style={{ fontWeight: 600, marginBottom: 8 }}>
          {name} failed to render
        </div>
        <div
          style={{
            fontFamily: 'var(--font-mono)',
            fontSize: 11,
            color: 'var(--text-muted)',
            marginBottom: 12,
            wordBreak: 'break-word',
          }}
        >
          {msg}
        </div>
        <button
          type="button"
          onClick={this.handleRetry}
          style={{
            padding: '6px 16px',
            fontSize: 12,
            fontWeight: 600,
            background: 'var(--accent-cyan, #0A84FF)',
            color: '#fff',
            border: 'none',
            borderRadius: 6,
            cursor: 'pointer',
          }}
        >
          Retry
        </button>
        {this.props.fallbackHint && (
          <div style={{ marginTop: 10, fontSize: 11, color: 'var(--text-muted)' }}>
            {this.props.fallbackHint}
          </div>
        )}
      </div>
    );
  }
}
