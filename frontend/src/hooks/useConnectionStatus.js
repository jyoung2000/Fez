import { useState, useEffect, useCallback, useRef } from 'react';

/**
 * Global connection health monitor.
 * Pings the backend every few seconds and tracks:
 *  - online/offline/reconnecting state
 *  - latency
 *  - consecutive failures (for stuck detection)
 *  - browser online/offline events
 */

const PING_INTERVAL = 8000;       // Normal poll interval
const FAST_PING_INTERVAL = 3000;  // When recovering
const MAX_FAILURES_BEFORE_DISCONNECTED = 2;

export default function useConnectionStatus() {
  const [status, setStatus] = useState('connecting'); // 'connected' | 'disconnected' | 'reconnecting' | 'connecting'
  const [latency, setLatency] = useState(null);
  const [lastSeen, setLastSeen] = useState(null);
  const failCountRef = useRef(0);
  const wasConnectedRef = useRef(false);

  const ping = useCallback(async () => {
    const start = Date.now();
    try {
      const res = await fetch('/api/providers/status', {
        signal: AbortSignal.timeout(5000),
      });
      if (res.ok) {
        const ms = Date.now() - start;
        setLatency(ms);
        setLastSeen(Date.now());
        failCountRef.current = 0;
        wasConnectedRef.current = true;
        setStatus('connected');
        return true;
      }
      throw new Error('not ok');
    } catch {
      failCountRef.current += 1;
      if (failCountRef.current >= MAX_FAILURES_BEFORE_DISCONNECTED) {
        setStatus(wasConnectedRef.current ? 'reconnecting' : 'disconnected');
      }
      return false;
    }
  }, []);

  useEffect(() => {
    // Initial ping
    ping();

    const id = setInterval(() => {
      ping();
    }, status === 'reconnecting' ? FAST_PING_INTERVAL : PING_INTERVAL);

    // Browser online/offline events
    const handleOnline = () => { ping(); };
    const handleOffline = () => {
      failCountRef.current = MAX_FAILURES_BEFORE_DISCONNECTED;
      setStatus('disconnected');
    };

    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    return () => {
      clearInterval(id);
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, [ping, status]);

  return { status, latency, lastSeen };
}
