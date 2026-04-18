import { useState, useCallback } from 'react';

const STORAGE_KEY = 'clipai-theme';

function getInitialTheme() {
  return document.documentElement.dataset.theme || 'light';
}

export default function useTheme() {
  const [theme, setTheme] = useState(getInitialTheme);

  const toggleTheme = useCallback(() => {
    setTheme((prev) => {
      const next = prev === 'dark' ? 'light' : 'dark';
      document.documentElement.dataset.theme = next;
      localStorage.setItem(STORAGE_KEY, next);
      return next;
    });
  }, []);

  return { theme, toggleTheme, isDark: theme === 'dark' };
}
