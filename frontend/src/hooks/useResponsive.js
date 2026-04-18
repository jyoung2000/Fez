import { useState, useEffect } from 'react';

export default function useResponsive() {
  const [state, setState] = useState(() => {
    const w = typeof window !== 'undefined' ? window.innerWidth : 1024;
    return { isMobile: w < 768, isTablet: w >= 768 && w < 1024, isDesktop: w >= 1024 };
  });

  useEffect(() => {
    const mq = window.matchMedia('(max-width: 767px)');
    const tq = window.matchMedia('(min-width: 768px) and (max-width: 1023px)');
    const update = () => {
      setState({
        isMobile: mq.matches,
        isTablet: tq.matches,
        isDesktop: !mq.matches && !tq.matches,
      });
    };
    mq.addEventListener('change', update);
    tq.addEventListener('change', update);
    return () => {
      mq.removeEventListener('change', update);
      tq.removeEventListener('change', update);
    };
  }, []);

  return state;
}
