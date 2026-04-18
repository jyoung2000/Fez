/**
 * Small hook that fetches /api/cloud/providers and exposes a refresh()
 * function. Used by both the Settings page and the Upload page so they
 * always agree about which providers are connected.
 */
import { useCallback, useEffect, useRef, useState } from 'react';

export default function useCloudProviders() {
  const [state, setState] = useState({
    loading: true,
    enabled: false,
    providers: [],
    error: null,
  });
  const mountedRef = useRef(true);

  const refresh = useCallback(async () => {
    try {
      const resp = await fetch('/api/cloud/providers');
      if (!resp.ok) {
        throw new Error(`HTTP ${resp.status}`);
      }
      const data = await resp.json();
      if (!mountedRef.current) return;
      setState({
        loading: false,
        enabled: !!data.enabled,
        providers: Array.isArray(data.providers) ? data.providers : [],
        error: null,
      });
    } catch (err) {
      if (!mountedRef.current) return;
      setState((prev) => ({
        ...prev,
        loading: false,
        error: err.message || 'Failed to load providers',
      }));
    }
  }, []);

  useEffect(() => {
    mountedRef.current = true;
    refresh();
    return () => {
      mountedRef.current = false;
    };
  }, [refresh]);

  return { ...state, refresh };
}
