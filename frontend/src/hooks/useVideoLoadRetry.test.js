// @vitest-environment jsdom
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import useVideoLoadRetry from './useVideoLoadRetry';

/**
 * Tests for the HEAD-probe behavior added in the preview-retry fix.
 *
 * The hook depends on ``fetch`` + a media element's event loop. We
 * stub both: ``fetch`` mocks the HEAD probe and the video ref is
 * wired to a stub object we can nudge events on.
 */

function _mockVideoRef(initialReadyState = 0) {
  const listeners = Object.create(null);
  const el = {
    readyState: initialReadyState,
    addEventListener(event, fn) {
      (listeners[event] ||= new Set()).add(fn);
    },
    removeEventListener(event, fn) {
      listeners[event]?.delete(fn);
    },
    load: vi.fn(),
  };
  el.__fire = (event) => {
    (listeners[event] || new Set()).forEach((fn) => fn(new Event(event)));
  };
  return { current: el };
}

describe('useVideoLoadRetry', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    globalThis.fetch = vi.fn();
  });

  afterEach(() => {
    vi.useRealTimers();
    delete globalThis.fetch;
  });

  it('enters preparing state on a 503 HEAD probe without waiting for error events', async () => {
    globalThis.fetch.mockResolvedValue({
      ok: false,
      status: 503,
      headers: new Map([['Retry-After', '5']]),
    });
    // Map.prototype.get exists, but useVideoLoadRetry calls
    // resp.headers.get — Map's get signature matches, so the mock
    // works as-is.

    const ref = _mockVideoRef();
    const { result } = renderHook(() => useVideoLoadRetry(ref, '/api/files/j/video.mp4'));
    // Let the HEAD fetch resolve.
    await act(async () => { await Promise.resolve(); });
    expect(result.current.preparing).toBe(true);
    expect(result.current.authError).toBe(false);
  });

  it('sets authError on a 401 probe and does not retry', async () => {
    globalThis.fetch.mockResolvedValue({
      ok: false,
      status: 401,
      headers: new Map(),
    });

    const ref = _mockVideoRef();
    const { result } = renderHook(() => useVideoLoadRetry(ref, '/api/files/j/video.mp4'));
    await act(async () => { await Promise.resolve(); });
    expect(result.current.authError).toBe(true);
    expect(result.current.preparing).toBe(false);
    // Advance time past any possible retry — load() should NOT be called.
    await act(async () => { vi.advanceTimersByTime(30_000); });
    expect(ref.current.load).not.toHaveBeenCalled();
  });

  it('uses Retry-After for the first retry delay', async () => {
    globalThis.fetch.mockResolvedValue({
      ok: false,
      status: 503,
      headers: new Map([['Retry-After', '7']]),
    });

    const ref = _mockVideoRef();
    renderHook(() => useVideoLoadRetry(ref, '/api/files/j/video.mp4'));
    await act(async () => { await Promise.resolve(); });

    // 6s later → no load yet (Retry-After says 7).
    await act(async () => { vi.advanceTimersByTime(6000); });
    expect(ref.current.load).not.toHaveBeenCalled();
    // 1.5s more → load fires.
    await act(async () => { vi.advanceTimersByTime(1500); });
    expect(ref.current.load).toHaveBeenCalled();
  });

  it('stalled without a progress within debounce triggers a retry', async () => {
    globalThis.fetch.mockResolvedValue({
      ok: true, status: 200, headers: new Map(),
    });

    const ref = _mockVideoRef();
    renderHook(() => useVideoLoadRetry(ref, '/api/files/j/video.mp4'));
    await act(async () => { await Promise.resolve(); });

    // Fire stalled — retry should be scheduled once the debounce elapses.
    await act(async () => { ref.current.__fire('stalled'); });
    expect(ref.current.load).not.toHaveBeenCalled();
    await act(async () => { vi.advanceTimersByTime(3500); });
    // Retry was scheduled; the first delay is 500ms.
    await act(async () => { vi.advanceTimersByTime(600); });
    expect(ref.current.load).toHaveBeenCalled();
  });

  it('stalled followed by progress within debounce does NOT retry', async () => {
    globalThis.fetch.mockResolvedValue({
      ok: true, status: 200, headers: new Map(),
    });

    const ref = _mockVideoRef();
    renderHook(() => useVideoLoadRetry(ref, '/api/files/j/video.mp4'));
    await act(async () => { await Promise.resolve(); });

    await act(async () => { ref.current.__fire('stalled'); });
    // Progress inside the 3s window cancels the retry.
    await act(async () => { vi.advanceTimersByTime(1500); });
    await act(async () => { ref.current.__fire('progress'); });
    await act(async () => { vi.advanceTimersByTime(5000); });
    expect(ref.current.load).not.toHaveBeenCalled();
  });
});
