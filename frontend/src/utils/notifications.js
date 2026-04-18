/**
 * Send a browser notification if the page is not visible and permission has
 * been granted.  Silently no-ops when the Notification API is unavailable or
 * the user has denied permission.
 *
 * @param {string} title - Notification title
 * @param {object} [options] - Standard Notification options (body, icon, tag…)
 */
export function sendNotification(title, options = {}) {
  if (!('Notification' in window)) return;
  if (document.visibilityState === 'visible') return;

  if (Notification.permission === 'granted') {
    new Notification(title, options);
  }
}

/**
 * Request notification permission.  Should be called from a user gesture
 * (e.g. button click) to avoid browser blocking.  Returns the permission
 * state string ('granted', 'denied', 'default').
 */
export async function requestNotificationPermission() {
  if (!('Notification' in window)) return 'denied';
  if (Notification.permission === 'granted') return 'granted';
  if (Notification.permission === 'denied') return 'denied';
  return Notification.requestPermission();
}
