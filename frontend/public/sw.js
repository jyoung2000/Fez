// ClipAI Service Worker — Background Fetch support for Android uploads.
// Scope: / — but does NOT intercept regular fetch/API calls.

// Early return for all fetch events — we do not want to cache API responses
// or interfere with auth/SPA routing.
self.addEventListener('fetch', () => {
  // No-op: let the browser handle all fetches normally.
  // We only use this SW for Background Fetch events.
});

// ── Background Fetch handlers ──────────────────────────────────────────────

self.addEventListener('backgroundfetchsuccess', (event) => {
  event.waitUntil((async () => {
    // All chunk uploads succeeded — notify the client
    try {
      const clients = await self.clients.matchAll({ type: 'window' });
      for (const client of clients) {
        client.postMessage({ type: 'bg-fetch-complete', id: event.registration.id });
      }
    } catch { /* best-effort */ }
    event.updateUI({ title: 'Upload complete' });
  })());
});

self.addEventListener('backgroundfetchfail', (event) => {
  event.waitUntil((async () => {
    try {
      const clients = await self.clients.matchAll({ type: 'window' });
      for (const client of clients) {
        client.postMessage({ type: 'bg-fetch-fail', id: event.registration.id });
      }
    } catch { /* best-effort */ }
    event.updateUI({ title: 'Upload failed — tap to retry' });
  })());
});

self.addEventListener('backgroundfetchabort', (event) => {
  // User or system cancelled the background fetch
  console.log('[SW] Background fetch aborted:', event.registration.id);
});

self.addEventListener('backgroundfetchclick', (event) => {
  // User tapped the OS notification — open the upload page
  event.waitUntil((async () => {
    const clients = await self.clients.matchAll({ type: 'window' });
    // Try to focus an existing window
    for (const client of clients) {
      if (client.url.includes('/') && 'focus' in client) {
        return client.focus();
      }
    }
    // No existing window — open a new one
    const uploadId = event.registration.id.replace('clipai-upload-', '');
    return self.clients.openWindow(`/?resume=${uploadId}`);
  })());
});

// Activate immediately — no need to wait for old SW
self.addEventListener('activate', (event) => {
  event.waitUntil(self.clients.claim());
});
