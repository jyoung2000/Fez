/**
 * PlaybackEngine — Master clock, media element sync, audio routing.
 *
 * Manages playback state and synchronizes hidden <video> elements
 * used as frame sources for RenderEngine. Also handles per-clip
 * volume via Web Audio API GainNode chains.
 */

export default class PlaybackEngine {
  constructor(options = {}) {
    this.isPlaying = false;
    this.currentTime = 0;
    this.playbackRate = 1;
    this.duration = 0;
    this._rafId = null;
    this._lastTimestamp = null;

    // Media element pool: assetId → { element, source?, gainNode? }
    this._mediaPool = new Map();

    // Audio routing
    this._audioCtx = null;
    this._masterGain = null;

    // Callbacks
    this.onFrame = options.onFrame || null;
    this.onTimeUpdate = options.onTimeUpdate || null;
    this.onPlayStateChange = options.onPlayStateChange || null;
    this.onEnded = options.onEnded || null;

    // Shuttle speed for J/K/L control
    this._shuttleSpeed = 0;
    this._reverseInterval = null;
  }

  /**
   * Initialize audio context (must be called on user gesture)
   */
  initAudio() {
    if (this._audioCtx) return;
    try {
      this._audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      this._masterGain = this._audioCtx.createGain();
      this._masterGain.connect(this._audioCtx.destination);
    } catch {
      // Web Audio not available
    }
  }

  /**
   * Resume audio context (required after user interaction due to autoplay policy)
   */
  resumeAudio() {
    if (this._audioCtx?.state === 'suspended') {
      this._audioCtx.resume().catch(() => {});
    }
  }

  /**
   * Register a media element as a frame source
   */
  registerMedia(assetId, element) {
    const existing = this._mediaPool.get(assetId);
    if (existing?.element === element) return;

    // Set up audio routing if needed
    let source = null;
    let gainNode = null;

    if (this._audioCtx && element instanceof HTMLVideoElement) {
      try {
        source = this._audioCtx.createMediaElementSource(element);
        gainNode = this._audioCtx.createGain();
        source.connect(gainNode);
        gainNode.connect(this._masterGain);
      } catch {
        // Element may already be connected or Web Audio unavailable
      }
    }

    this._mediaPool.set(assetId, { element, source, gainNode });
  }

  /**
   * Unregister a media element
   */
  unregisterMedia(assetId) {
    const entry = this._mediaPool.get(assetId);
    if (!entry) return;

    try {
      entry.element.pause();
      if (entry.gainNode) entry.gainNode.disconnect();
      if (entry.source) entry.source.disconnect();
    } catch {}

    this._mediaPool.delete(assetId);
  }

  /**
   * Get the media element map for RenderEngine
   */
  getMediaElements() {
    const map = new Map();
    for (const [id, entry] of this._mediaPool) {
      map.set(id, entry.element);
    }
    return map;
  }

  /**
   * Set volume for a specific media element
   */
  setClipVolume(assetId, volume, fadeTime = 0.05) {
    const entry = this._mediaPool.get(assetId);
    if (!entry) return;

    if (entry.gainNode && this._audioCtx) {
      const now = this._audioCtx.currentTime;
      entry.gainNode.gain.cancelScheduledValues(now);
      entry.gainNode.gain.setValueAtTime(entry.gainNode.gain.value, now);
      entry.gainNode.gain.linearRampToValueAtTime(volume, now + fadeTime);
    } else {
      entry.element.volume = Math.min(1, volume);
    }
  }

  /**
   * Set master volume
   */
  setMasterVolume(volume) {
    if (this._masterGain) {
      this._masterGain.gain.value = volume;
    }
  }

  /**
   * Start playback
   */
  play() {
    if (this.isPlaying) return;
    this.isPlaying = true;
    this.resumeAudio();

    // Start all relevant video elements
    for (const [, entry] of this._mediaPool) {
      if (entry.element instanceof HTMLVideoElement) {
        entry.element.playbackRate = this.playbackRate;
        entry.element.play().catch(() => {});
      }
    }

    this._lastTimestamp = performance.now();
    this._tick();
    this.onPlayStateChange?.(true);
  }

  /**
   * Pause playback
   */
  pause() {
    if (!this.isPlaying) return;
    this.isPlaying = false;

    if (this._rafId) {
      cancelAnimationFrame(this._rafId);
      this._rafId = null;
    }

    // Pause all video elements
    for (const [, entry] of this._mediaPool) {
      if (entry.element instanceof HTMLVideoElement) {
        entry.element.pause();
      }
    }

    this._clearReversePlayback();
    this.onPlayStateChange?.(false);
  }

  /**
   * Toggle play/pause
   */
  togglePlay() {
    if (this.isPlaying) this.pause();
    else this.play();
  }

  /**
   * Seek to a specific time
   */
  seek(time) {
    this.currentTime = Math.max(0, Math.min(this.duration, time));

    // Seek all media elements to their clip-relative time
    // (caller should handle clip offset mapping)
    for (const [, entry] of this._mediaPool) {
      if (entry.element instanceof HTMLVideoElement) {
        entry.element.currentTime = this.currentTime;
      }
    }

    this.onTimeUpdate?.(this.currentTime);
    this.onFrame?.(this.currentTime);
  }

  /**
   * Seek a specific media element to a clip-relative time
   */
  seekMedia(assetId, time) {
    const entry = this._mediaPool.get(assetId);
    if (entry?.element) {
      entry.element.currentTime = time;
    }
  }

  /**
   * Set playback rate
   */
  setPlaybackRate(rate) {
    this.playbackRate = rate;
    for (const [, entry] of this._mediaPool) {
      if (entry.element instanceof HTMLVideoElement) {
        entry.element.playbackRate = rate;
      }
    }
  }

  /**
   * J/K/L shuttle control
   */
  setShuttleSpeed(speed) {
    this._shuttleSpeed = speed;
    this._clearReversePlayback();

    if (speed === 0) {
      this.pause();
      return;
    }

    if (speed > 0) {
      this.playbackRate = speed;
      for (const [, entry] of this._mediaPool) {
        if (entry.element instanceof HTMLVideoElement) {
          entry.element.playbackRate = speed;
        }
      }
      if (!this.isPlaying) this.play();
    } else {
      // Reverse playback via frame stepping
      this.isPlaying = true;
      this.onPlayStateChange?.(true);

      // Pause native elements
      for (const [, entry] of this._mediaPool) {
        if (entry.element instanceof HTMLVideoElement) {
          entry.element.pause();
        }
      }

      this._reverseInterval = setInterval(() => {
        const step = (1 / 30) * Math.abs(speed);
        const newTime = Math.max(0, this.currentTime - step);
        this.currentTime = newTime;

        for (const [, entry] of this._mediaPool) {
          if (entry.element instanceof HTMLVideoElement) {
            entry.element.currentTime = newTime;
          }
        }

        this.onTimeUpdate?.(newTime);
        this.onFrame?.(newTime);

        if (newTime <= 0) {
          this._shuttleSpeed = 0;
          this.pause();
        }
      }, 1000 / 30);
    }
  }

  _clearReversePlayback() {
    if (this._reverseInterval) {
      clearInterval(this._reverseInterval);
      this._reverseInterval = null;
    }
  }

  /**
   * rAF tick — drives the master clock and triggers frame renders
   */
  _tick() {
    if (!this.isPlaying) return;

    // Use first video element as time source
    let primaryTime = null;
    for (const [, entry] of this._mediaPool) {
      if (entry.element instanceof HTMLVideoElement && !entry.element.paused) {
        primaryTime = entry.element.currentTime;
        break;
      }
    }

    if (primaryTime !== null) {
      this.currentTime = primaryTime;
    }

    // Check end
    if (this.duration > 0 && this.currentTime >= this.duration) {
      this.pause();
      this.onEnded?.();
      return;
    }

    this.onTimeUpdate?.(this.currentTime);
    this.onFrame?.(this.currentTime);

    this._rafId = requestAnimationFrame(() => this._tick());
  }

  /**
   * Clean up everything
   */
  destroy() {
    this.pause();
    this._clearReversePlayback();

    const ids = [...this._mediaPool.keys()];
    for (const id of ids) {
      this.unregisterMedia(id);
    }
    this._mediaPool.clear();

    if (this._audioCtx) {
      this._audioCtx.close().catch(() => {});
      this._audioCtx = null;
      this._masterGain = null;
    }
  }
}
