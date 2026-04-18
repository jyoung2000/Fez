import React, { useRef, useState, useEffect, useMemo } from 'react';
import { processKeyframes, interpolateSubjectX, isDynamic, subjectXToCenterPct, safeSubjectX } from '../utils/subjectTracking';
import useResponsive from '../hooks/useResponsive';
import { usePlayer } from '../contexts/PlayerContext';

const ASPECT_RATIO_VALUES = {
  '16:9': 16 / 9,
  '9:16': 9 / 16,
  '1:1': 1.0,
  '4:5': 4 / 5,
};

// subjectXToCenterPct imported from subjectTracking.js (shared with ClipPreview + backend)

function formatTime(seconds) {
  if (!seconds || isNaN(seconds)) return '0:00';
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, '0')}`;
}

export default function VideoPlayer({ src, clipStart, clipEnd, onTimeUpdate, aspectRatio, sourceWidth = 1920, sourceHeight = 1080, subjectX = 50, scenes, sceneCuts = null, initialTime }) {
  const { isMobile } = useResponsive();
  const videoRef = useRef(null);
  const containerRef = useRef(null);
  const [playing, setPlaying] = useState(false);
  const [displayTime, setDisplayTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [volume, setVolume] = useState(1);
  const [speed, setSpeed] = useState(1);
  const [hovered, setHovered] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);

  // Store callback in a ref to avoid re-registering event listeners when
  // the parent passes a new function reference on each render.
  const onTimeUpdateRef = useRef(onTimeUpdate);
  onTimeUpdateRef.current = onTimeUpdate;

  // Throttled display time updater — avoids re-rendering on every timeupdate
  // event (~4x/sec). Instead, updates the display ~4x/sec via rAF batching.
  const lastDisplayUpdateRef = useRef(0);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    // Time reporting uses a rAF loop during playback so consumers
    // (``SubtitleOverlay``, ``TranscriptViewer``) see a ~60 Hz clock
    // instead of the browser's ~4 Hz ``timeupdate`` event. Without this,
    // ``videoCurrentTime`` lags real audio by 0–250 ms and the active
    // transcript line / burned-in subtitles appear to trail the speaker.
    //
    // Guards:
    //  - ``lastReportedT`` dedupes identical values so a paused (or
    //    off-screen) player doesn't spam ``setVideoCurrentTime`` at
    //    60 Hz with the same number — same pattern as ``VideoEditor``'s
    //    tick (commit 1ad9593). Any second writer racing on the same
    //    parent state is a bug we don't want to re-enable.
    //  - ``timeupdate`` is kept as a fallback heartbeat for background
    //    tabs: browsers throttle rAF to ~1 Hz when the tab isn't
    //    visible, but ``timeupdate`` still fires, so consumers don't
    //    freeze when the user comes back.
    //  - ``seeked`` and the initial mount each fire an immediate report
    //    so the overlay / highlight snaps to the correct spot after a
    //    scrub instead of waiting for the next natural tick.
    let rafId = null;
    let lastReportedT = -1;

    const report = (t) => {
      if (t === lastReportedT) return;
      lastReportedT = t;
      onTimeUpdateRef.current?.(t);
    };

    const tick = () => {
      const t = video.currentTime;
      report(t);
      // Auto-stop at clip end in preview mode
      if (clipEnd && t >= clipEnd) {
        video.pause();
        setPlaying(false);
        setDisplayTime(t);
        rafId = null;
        return;
      }
      // Throttled display update (~4 Hz) — the HUD clock doesn't need
      // 60 fps precision and re-rendering VideoPlayer that often is
      // wasteful.
      const now = performance.now();
      if (now - lastDisplayUpdateRef.current > 250) {
        lastDisplayUpdateRef.current = now;
        setDisplayTime(t);
      }
      rafId = requestAnimationFrame(tick);
    };

    const startRaf = () => {
      if (rafId == null) rafId = requestAnimationFrame(tick);
    };
    const stopRaf = () => {
      if (rafId != null) {
        cancelAnimationFrame(rafId);
        rafId = null;
      }
      // One final sample so consumers land exactly on the paused /
      // ended position instead of whatever the last rAF frame saw.
      report(video.currentTime);
      setDisplayTime(video.currentTime);
    };

    // Fallback heartbeat — fires even if rAF is throttled in background
    // tabs. When the rAF loop is active we let it drive reports; when
    // it's not (paused, or the tab is backgrounded and rAF has been
    // choked), ``timeupdate`` still delivers fresh values.
    const onTime = () => {
      if (rafId == null) report(video.currentTime);
    };
    const onPlay = () => startRaf();
    const onPause = () => stopRaf();
    const onEnded = () => stopRaf();
    const onSeeked = () => report(video.currentTime);
    const onDur = () => setDuration(video.duration);
    const onError = () => {
      // Retry once on load error (handles transient partial content failures)
      if (!video._retried) {
        video._retried = true;
        video.load();
      }
    };

    video.addEventListener('timeupdate', onTime);
    video.addEventListener('play', onPlay);
    video.addEventListener('pause', onPause);
    video.addEventListener('ended', onEnded);
    video.addEventListener('seeked', onSeeked);
    video.addEventListener('loadedmetadata', onDur);
    video.addEventListener('error', onError);

    // Initial push so the overlay / transcript highlight has a value on
    // mount (otherwise they stay at the default ``currentTime=0`` until
    // the user presses play).
    report(video.currentTime);
    if (!video.paused) startRaf();

    return () => {
      if (rafId != null) {
        cancelAnimationFrame(rafId);
        rafId = null;
      }
      video.removeEventListener('timeupdate', onTime);
      video.removeEventListener('play', onPlay);
      video.removeEventListener('pause', onPause);
      video.removeEventListener('ended', onEnded);
      video.removeEventListener('seeked', onSeeked);
      video.removeEventListener('loadedmetadata', onDur);
      video.removeEventListener('error', onError);
    };
  }, [clipEnd]);

  // Track fullscreen changes
  useEffect(() => {
    const onFsChange = () => setIsFullscreen(!!document.fullscreenElement);
    document.addEventListener('fullscreenchange', onFsChange);
    return () => document.removeEventListener('fullscreenchange', onFsChange);
  }, []);

  const togglePlay = () => {
    const video = videoRef.current;
    if (!video) return;
    if (video.paused) {
      video.play();
      setPlaying(true);
    } else {
      video.pause();
      setPlaying(false);
      setDisplayTime(video.currentTime);
    }
  };

  const seek = (e) => {
    const video = videoRef.current;
    if (!video) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const pct = (e.clientX - rect.left) / rect.width;
    video.currentTime = pct * duration;
    setDisplayTime(pct * duration);
  };

  const seekTo = (time) => {
    const video = videoRef.current;
    if (!video) return;
    video.currentTime = time;
    setDisplayTime(time);
  };

  // Expose seekTo and pause/getTime via window for cross-component control.
  // Register this player instance with the PlayerContext. The context
  // also mirrors the API onto ``window.__clipai_*`` for backward
  // compat, so callers that haven't migrated still work.
  const player = usePlayer();
  useEffect(() => {
    const api = {
      seek: seekTo,
      pause: () => {
        const video = videoRef.current;
        if (video && !video.paused) {
          video.pause();
          setPlaying(false);
        }
      },
      play: () => {
        const video = videoRef.current;
        if (video && video.paused) {
          video.play().then(() => setPlaying(true)).catch(() => {});
        }
      },
      getTime: () => videoRef.current?.currentTime ?? 0,
    };
    player.register(api);
    return () => player.unregister();
    // ``seekTo`` is stable enough for a one-time register; player.* are
    // memoized in the provider.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Seek to initialTime once when the player mounts (no auto-play)
  const initialTimeApplied = useRef(false);
  useEffect(() => {
    const video = videoRef.current;
    if (!video || initialTime == null || initialTimeApplied.current) return;
    initialTimeApplied.current = true;
    const doSeek = () => {
      video.currentTime = initialTime;
      setDisplayTime(initialTime);
    };
    if (video.readyState >= 1) doSeek();
    else video.addEventListener('loadedmetadata', doSeek, { once: true });
  }, [initialTime]);

  // Auto-seek and auto-play when clip preview changes
  useEffect(() => {
    const video = videoRef.current;
    if (!video || clipStart === undefined || clipStart === null) return;
    video.currentTime = clipStart;
    setDisplayTime(clipStart);
    video.play().then(() => setPlaying(true)).catch(() => {});
  }, [clipStart, clipEnd]);

  const changeSpeed = () => {
    const speeds = [0.5, 1, 1.5, 2];
    const idx = speeds.indexOf(speed);
    const next = speeds[(idx + 1) % speeds.length];
    setSpeed(next);
    if (videoRef.current) videoRef.current.playbackRate = next;
  };

  const toggleFullscreen = () => {
    const el = containerRef.current;
    if (!el) return;
    if (document.fullscreenElement) {
      document.exitFullscreen();
    } else {
      el.requestFullscreen?.();
    }
  };

  const progress = duration ? (displayTime / duration) * 100 : 0;

  // Aspect ratio awareness — match ClipPreview / ClipSEO behavior
  const srcRatio = sourceWidth / sourceHeight;
  const targetRatio = useMemo(() => {
    if (aspectRatio && ASPECT_RATIO_VALUES[aspectRatio]) {
      return ASPECT_RATIO_VALUES[aspectRatio];
    }
    return srcRatio;
  }, [aspectRatio, srcRatio]);
  const isCrop = useMemo(() => {
    if (!aspectRatio || !ASPECT_RATIO_VALUES[aspectRatio]) return false;
    return Math.abs(srcRatio - ASPECT_RATIO_VALUES[aspectRatio]) > 0.01;
  }, [aspectRatio, srcRatio]);

  // Dynamic subject tracking keyframes (full pipeline matching backend)
  const subjectKeyframes = useMemo(
    () => {
      if (!scenes?.length || clipStart == null || clipEnd == null) return null;
      const _isCrop = isCrop;
      const processed = processKeyframes(scenes, clipStart, clipEnd, _isCrop ? srcRatio : null, _isCrop ? targetRatio : null, null, sceneCuts || null);
      if (processed && isDynamic(processed)) {
        console.log(`[SubjectTracking] VideoPlayer: ${processed.length} keyframes (pipeline: build→cuts→compress→deadzone→smooth→holds) (${clipStart.toFixed(1)}s-${clipEnd.toFixed(1)}s), x range: ${Math.min(...processed.map(k=>k.x))}-${Math.max(...processed.map(k=>k.x))}`);
      }
      return processed;
    },
    // ``sceneCuts`` is consumed by ``processKeyframes`` (passed in
    // above). Without it in the dep list, scene-cut updates left the
    // subject track stale until another dep changed.
    [scenes, clipStart, clipEnd, isCrop, srcRatio, targetRatio, sceneCuts],
  );
  const hasDynamicSubject = useMemo(
    () => isCrop && subjectKeyframes && isDynamic(subjectKeyframes),
    [isCrop, subjectKeyframes],
  );

  // Update objectPosition dynamically via rAF for instant ~60fps snaps
  const lastAppliedPctRef = useRef(null);
  useEffect(() => {
    if (!hasDynamicSubject) return;
    const video = videoRef.current;
    if (!video) return;
    // Apply initial position synchronously to eliminate 1-2 frame gap
    {
      const initRel = video.currentTime - (clipStart || 0);
      const initSx = interpolateSubjectX(subjectKeyframes, initRel);
      const initPct = subjectXToCenterPct(initSx, srcRatio, targetRatio);
      video.style.objectPosition = `${initPct}% 50%`;
      lastAppliedPctRef.current = initPct;
    }
    let animId;
    let lastPct = null;
    const tick = () => {
      const relTime = video.currentTime - (clipStart || 0);
      const sx = interpolateSubjectX(subjectKeyframes, relTime);
      const centerPct = subjectXToCenterPct(sx, srcRatio, targetRatio);
      // Only update DOM if value actually changed (avoid layout thrashing)
      const rounded = Math.round(centerPct * 10000) / 10000;
      if (rounded !== lastPct) {
        video.style.objectPosition = `${centerPct}% 50%`;
        lastPct = rounded;
        lastAppliedPctRef.current = centerPct;
      }
      animId = requestAnimationFrame(tick);
    };
    animId = requestAnimationFrame(tick);
    return () => {
      cancelAnimationFrame(animId);
      // Do NOT clear video.style.objectPosition here — the cleanup runs
      // after React's DOM commit, so clearing would overwrite the correct
      // static objectPosition that React just applied.
    };
  }, [hasDynamicSubject, subjectKeyframes, clipStart, srcRatio, targetRatio]);

  // Constrain portrait containers so they don't take full width
  const videoMaxWidth = useMemo(() => {
    if (targetRatio < 1) {
      const maxHpx = (typeof window !== 'undefined' ? window.innerHeight : 900) * 0.4;
      return Math.min(Math.round(maxHpx * targetRatio), 400);
    }
    return undefined;
  }, [targetRatio]);

  return (
    <div
      ref={containerRef}
      style={isFullscreen ? {
        position: 'relative',
        background: 'var(--video-bg)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        width: '100vw',
        height: '100vh',
      } : {
        position: 'relative',
        background: 'var(--video-bg)',
        borderBottom: '1px solid var(--border)',
        maxWidth: videoMaxWidth,
        margin: videoMaxWidth ? '0 auto' : undefined,
      }}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
    >
      <video
        ref={videoRef}
        src={src}
        preload="auto"
        style={{
          display: 'block',
          ...(isFullscreen ? {
            maxWidth: '100vw',
            maxHeight: '100vh',
            width: 'auto',
            height: '100vh',
          } : {
            width: '100%',
            maxHeight: '40vh',
          }),
          ...(aspectRatio ? { aspectRatio: `${targetRatio}` } : {}),
          objectFit: isCrop ? 'cover' : 'contain',
          objectPosition: isCrop ? `${subjectXToCenterPct(hasDynamicSubject ? subjectKeyframes[0].x : subjectX, srcRatio, targetRatio)}% 50%` : undefined,
        }}
        onClick={togglePlay}
      />

      {/* Aspect ratio badge */}
      {aspectRatio && (
        <div style={{
          position: 'absolute',
          top: 6,
          right: 6,
          fontSize: 10,
          fontFamily: 'var(--font-mono)',
          color: 'var(--accent-amber)',
          background: 'var(--badge-overlay-bg)',
          padding: '2px 6px',
          borderRadius: 3,
          pointerEvents: 'none',
          zIndex: 2,
        }}>
          {aspectRatio}
        </div>
      )}

      {/* Controls overlay */}
      <div
        style={{
          position: 'absolute',
          bottom: 0,
          left: 0,
          right: 0,
          background: 'var(--video-gradient)',
          padding: '24px 16px 12px',
          opacity: hovered ? 1 : 0,
          transition: 'opacity 0.2s ease',
        }}
      >
        {/* Seek bar */}
        <div
          onClick={seek}
          style={{
            height: isMobile ? 8 : 4,
            background: 'var(--bg-elevated)',
            cursor: 'pointer',
            position: 'relative',
            marginBottom: 8,
            borderRadius: 4,
          }}
        >
          {/* Clip markers */}
          {clipStart !== undefined && clipEnd !== undefined && duration > 0 && (
            <div
              style={{
                position: 'absolute',
                left: `${(clipStart / duration) * 100}%`,
                width: `${((clipEnd - clipStart) / duration) * 100}%`,
                height: '100%',
                background: 'var(--amber-dim)',
                borderLeft: '2px solid var(--accent-amber)',
                borderRight: '2px solid var(--accent-amber)',
              }}
            />
          )}
          <div
            style={{
              height: '100%',
              width: `${progress}%`,
              background: 'var(--accent-cyan)',
              position: 'relative',
            }}
          >
            <div
              style={{
                position: 'absolute',
                right: -4,
                top: -4,
                width: 12,
                height: 12,
                borderRadius: '50%',
                background: 'var(--accent-cyan)',
              }}
            />
          </div>
        </div>

        {/* Control buttons */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <button
            onClick={togglePlay}
            style={{
              background: 'none',
              border: 'none',
              color: 'var(--video-controls-text)',
              fontSize: isMobile ? 22 : 18,
              padding: isMobile ? 8 : 4,
              minHeight: isMobile ? 44 : undefined,
            }}
          >
            {playing ? '\u23F8' : '\u25B6'}
          </button>

          <span style={{ fontFamily: 'var(--font-mono)', fontSize: 12, color: 'var(--text-secondary)' }}>
            {formatTime(displayTime)} / {formatTime(duration)}
          </span>

          <div style={{ flex: 1 }} />

          {/* Volume */}
          <input
            type="range"
            min="0"
            max="1"
            step="0.1"
            value={volume}
            onChange={(e) => {
              const v = parseFloat(e.target.value);
              setVolume(v);
              if (videoRef.current) videoRef.current.volume = v;
            }}
            style={{ width: 60, accentColor: 'var(--accent-cyan)' }}
          />

          {/* Speed */}
          <button
            onClick={changeSpeed}
            style={{
              background: 'var(--bg-elevated)',
              border: '1px solid var(--border)',
              color: 'var(--text-secondary)',
              padding: isMobile ? '6px 12px' : '2px 8px',
              borderRadius: 'var(--radius-sm)',
              fontSize: isMobile ? 13 : 11,
              fontFamily: 'var(--font-mono)',
            }}
          >
            {speed}x
          </button>

          {/* Fullscreen */}
          <button
            onClick={toggleFullscreen}
            style={{
              background: 'none',
              border: 'none',
              color: 'var(--text-secondary)',
              fontSize: isMobile ? 20 : 16,
              padding: isMobile ? 8 : 4,
              minHeight: isMobile ? 44 : undefined,
            }}
          >
            {isFullscreen ? '\u2715' : '\u26F6'}
          </button>
        </div>
      </div>
    </div>
  );
}
