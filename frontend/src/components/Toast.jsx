import React, { useState, useEffect, useCallback } from 'react';
import useResponsive from '../hooks/useResponsive';

let _addToast = null;

export function showToast(message, type = 'info') {
  // Ensure message is always a string to prevent React error #310
  const safeMessage = typeof message === 'string' ? message : String(message ?? '');
  if (_addToast) _addToast({ message: safeMessage, type, id: Date.now() });
}

export default function ToastContainer() {
  const [toasts, setToasts] = useState([]);
  const { isMobile } = useResponsive();

  const addToast = useCallback((toast) => {
    setToasts((prev) => [...prev, toast]);
    setTimeout(() => {
      setToasts((prev) => prev.filter((t) => t.id !== toast.id));
    }, 5000);
  }, []);

  useEffect(() => {
    _addToast = addToast;
    return () => { _addToast = null; };
  }, [addToast]);

  if (toasts.length === 0) return null;

  return (
    <div
      style={{
        position: 'fixed',
        ...(isMobile
          ? { bottom: 'calc(76px + env(safe-area-inset-bottom, 0px))', left: 16, right: 16 }
          : { top: 16, right: 16 }
        ),
        zIndex: 9999,
        display: 'flex',
        flexDirection: 'column',
        gap: 8,
        alignItems: isMobile ? 'stretch' : 'flex-end',
      }}
    >
      {toasts.map((toast) => {
        const colors = {
          info: { bg: 'var(--accent-cyan-dim)', border: 'var(--accent-cyan)', text: 'var(--accent-cyan)' },
          success: { bg: 'var(--success-dim)', border: 'var(--success)', text: 'var(--success)' },
          error: { bg: 'var(--danger-dim)', border: 'var(--danger)', text: 'var(--danger)' },
          warning: { bg: 'var(--amber-dim)', border: 'var(--accent-amber)', text: 'var(--accent-amber)' },
        };
        const c = colors[toast.type] || colors.info;

        return (
          <div
            key={toast.id}
            className="slide-in"
            style={{
              background: c.bg,
              border: `1px solid ${c.border}`,
              borderRadius: 'var(--radius-md)',
              padding: isMobile ? '12px 16px' : '10px 16px',
              fontSize: 13,
              color: c.text,
              maxWidth: isMobile ? '100%' : 360,
              backdropFilter: 'blur(20px) saturate(180%)',
              WebkitBackdropFilter: 'blur(20px) saturate(180%)',
              boxShadow: 'var(--shadow-md)',
            }}
          >
            {toast.message}
          </div>
        );
      })}
    </div>
  );
}
