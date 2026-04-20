import { describe, it, expect, beforeEach, vi, afterEach } from 'vitest';
import {
  getSignedMediaUrl,
  invalidateSignedMediaUrl,
  _resetCacheForTests,
} from './signedMediaUrl';

function _mockFetchResponse(body) {
  return {
    ok: true,
    status: 200,
    json: async () => body,
  };
}

describe('getSignedMediaUrl', () => {
  beforeEach(() => {
    _resetCacheForTests();
    globalThis.fetch = vi.fn();
  });
  afterEach(() => {
    delete globalThis.fetch;
  });

  it('mints once and caches the URL', async () => {
    globalThis.fetch.mockResolvedValue(_mockFetchResponse({
      url: '/api/files/job-1/video.mp4?exp=9999999999&sig=abc&u=u-1',
      expires_at: 9999999999,
    }));
    const a = await getSignedMediaUrl('job-1', 'video.mp4');
    const b = await getSignedMediaUrl('job-1', 'video.mp4');
    expect(a).toBe('/api/files/job-1/video.mp4?exp=9999999999&sig=abc&u=u-1');
    expect(b).toBe(a);
    // Only one network call — the second read came from the cache.
    expect(globalThis.fetch).toHaveBeenCalledTimes(1);
  });

  it('dedups two simultaneous requests for the same key', async () => {
    let resolveFn;
    globalThis.fetch.mockImplementation(() => new Promise((resolve) => {
      resolveFn = resolve;
    }));
    const p1 = getSignedMediaUrl('job-1', 'video.mp4');
    const p2 = getSignedMediaUrl('job-1', 'video.mp4');
    // One call in flight even with two callers.
    expect(globalThis.fetch).toHaveBeenCalledTimes(1);
    resolveFn(_mockFetchResponse({
      url: '/api/files/job-1/video.mp4?exp=9999999999&sig=abc&u=u-1',
      expires_at: 9999999999,
    }));
    const [a, b] = await Promise.all([p1, p2]);
    expect(a).toBe(b);
  });

  it('refetches an expired cache entry', async () => {
    const past = Math.floor(Date.now() / 1000) - 10;
    globalThis.fetch
      .mockResolvedValueOnce(_mockFetchResponse({
        url: `/api/files/j/v.mp4?exp=${past}&sig=one&u=u`,
        expires_at: past,
      }))
      .mockResolvedValueOnce(_mockFetchResponse({
        url: '/api/files/j/v.mp4?exp=9999999999&sig=two&u=u',
        expires_at: 9999999999,
      }));
    const a = await getSignedMediaUrl('j', 'v.mp4');
    const b = await getSignedMediaUrl('j', 'v.mp4');
    expect(a).toContain('sig=one');
    expect(b).toContain('sig=two');
    expect(globalThis.fetch).toHaveBeenCalledTimes(2);
  });

  it('strips leading slash from path for the cache key', async () => {
    globalThis.fetch.mockResolvedValue(_mockFetchResponse({
      url: '/api/files/job-1/video.mp4?exp=9999999999&sig=abc&u=u-1',
      expires_at: 9999999999,
    }));
    const a = await getSignedMediaUrl('job-1', 'video.mp4');
    const b = await getSignedMediaUrl('job-1', '/video.mp4');
    expect(a).toBe(b);
    expect(globalThis.fetch).toHaveBeenCalledTimes(1);
  });

  it('throws on empty args', async () => {
    await expect(() => getSignedMediaUrl('', 'p')).rejects.toThrow();
    await expect(() => getSignedMediaUrl('j', '')).rejects.toThrow();
  });

  it('surfaces server errors', async () => {
    globalThis.fetch.mockResolvedValue({ ok: false, status: 500 });
    await expect(() => getSignedMediaUrl('j', 'v.mp4')).rejects.toThrow(/500/);
  });

  it('invalidateSignedMediaUrl forces a re-mint', async () => {
    globalThis.fetch
      .mockResolvedValueOnce(_mockFetchResponse({
        url: '/api/files/j/v.mp4?exp=9999999999&sig=one&u=u',
        expires_at: 9999999999,
      }))
      .mockResolvedValueOnce(_mockFetchResponse({
        url: '/api/files/j/v.mp4?exp=9999999999&sig=two&u=u',
        expires_at: 9999999999,
      }));
    await getSignedMediaUrl('j', 'v.mp4');
    invalidateSignedMediaUrl('j', 'v.mp4');
    const after = await getSignedMediaUrl('j', 'v.mp4');
    expect(after).toContain('sig=two');
  });

  it('propagates abort signals', async () => {
    const ctrl = new AbortController();
    globalThis.fetch.mockImplementation((url, opts) =>
      new Promise((_, reject) => {
        opts.signal?.addEventListener('abort', () => reject(new DOMException('abort', 'AbortError')));
      }),
    );
    const p = getSignedMediaUrl('j', 'v.mp4', { signal: ctrl.signal });
    ctrl.abort();
    await expect(p).rejects.toThrow();
  });
});
