import React, { useEffect, useMemo, useRef, useState } from 'react';
import { useParams } from 'react-router-dom';
import VideoEditor from '../components/VideoEditor';
import SubtitleOverlay from '../components/SubtitleOverlay';
import EditorErrorBoundary from '../components/EditorErrorBoundary';

/**
 * Public, read-only view of a shared analysis or clip.
 *
 * Bound to ``/share/:token`` and ``/share/:token/clip/:clipId``. The
 * recipient does NOT need to sign in — the token itself is the
 * credential. The frontend only calls the public endpoints under
 * ``/api/share/public/<token>/...`` which the AuthMiddleware
 * allow-lists.
 *
 * The preview embeds the same ``<VideoEditor>`` the SEO page uses
 * (scrub / play / pause / subtitles / aspect-ratio preview) so the
 * recipient sees an interactive rendering of the clip rather than a
 * static stat card. Owner-only features — editing transcripts,
 * exporting, changing settings server-side — are not exposed because
 * the public share endpoints don't authenticate the recipient.
 */
export default function SharedView() {
  const { token, clipId: routeClipId } = useParams();
  const [info, setInfo] = useState(null);
  const [data, setData] = useState(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(true);

  // Lightweight editor state — the recipient CAN tweak these locally
  // (zoom the timeline, change subtitle size, toggle speakers) but the
  // changes are session-only and never persist back to the owner's job.
  const [currentTime, setCurrentTime] = useState(0);
  const [playbackVolume, setPlaybackVolume] = useState(100);
  const [playbackSpeed, setPlaybackSpeed] = useState(1);
  const [clipSettings, setClipSettings] = useState({});
  const clipSettingsInitialized = useRef(false);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      setLoading(true);
      setError('');
      try {
        const infoRes = await fetch(`/api/share/public/${encodeURIComponent(token)}`);
        if (!infoRes.ok) {
          const body = await infoRes.json().catch(() => ({}));
          throw new Error(body?.detail || `HTTP ${infoRes.status}`);
        }
        const infoBody = await infoRes.json();
        if (cancelled) return;
        setInfo(infoBody);

        const url = infoBody.scope === 'clip'
          ? `/api/share/public/${encodeURIComponent(token)}/clip`
          : `/api/share/public/${encodeURIComponent(token)}/job`;
        const dataRes = await fetch(url);
        if (!dataRes.ok) {
          const body = await dataRes.json().catch(() => ({}));
          throw new Error(body?.detail || `HTTP ${dataRes.status}`);
        }
        const body = await dataRes.json();
        if (cancelled) return;
        setData(body);
        // Seed subtitle / playback settings from the job's canonical
        // settings so the share view looks like the owner's preview.
        if (!clipSettingsInitialized.current && body?.subtitle_settings) {
          setClipSettings(body.subtitle_settings);
          clipSettingsInitialized.current = true;
        }
      } catch (e) {
        if (!cancelled) setError(e?.message || 'Failed to load share');
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    if (token) load();
    return () => { cancelled = true; };
  }, [token]);

  // ── Derived data for the VideoEditor ─────────────────────────────
  const isClip = info?.scope === 'clip';
  const clip = useMemo(() => {
    if (!data) return null;
    if (isClip) return data.clip || null;
    // Job scope but URL has /clip/:clipId — find that clip if it exists.
    if (routeClipId) {
      const cid = parseInt(routeClipId, 10);
      return (data.clips || []).find((c) => c.id === cid) || null;
    }
    return null;
  }, [data, isClip, routeClipId]);

  const videoSrc = useMemo(() => {
    if (!token) return null;
    return `/api/share/public/${encodeURIComponent(token)}/video`;
  }, [token]);

  // Source dimensions parsed from the "1920x1080" style resolution
  // string the backend returns. Fall back to 1920×1080 so the crop
  // math doesn't divide by zero — a wrong aspect ratio would just
  // render a fullscreen source instead of a crashed component.
  const sourceDims = useMemo(() => {
    const res = info?.resolution || data?.resolution || '';
    const m = /^(\d+)x(\d+)/.exec(res);
    if (m) return { w: parseInt(m[1], 10), h: parseInt(m[2], 10) };
    return { w: 1920, h: 1080 };
  }, [info, data]);

  const transcript = useMemo(() => {
    if (!data) return [];
    return Array.isArray(data.transcript) ? data.transcript : [];
  }, [data]);

  const scenes = useMemo(() => {
    if (!data) return [];
    return Array.isArray(data.scenes) ? data.scenes : [];
  }, [data]);

  const sceneCuts = useMemo(() => {
    if (!data) return null;
    return Array.isArray(data.scene_cut_timestamps) ? data.scene_cut_timestamps : null;
  }, [data]);

  const subjectTrack = useMemo(() => {
    if (!data) return null;
    return Array.isArray(data.subject_track) ? data.subject_track : null;
  }, [data]);

  const speakerNames = useMemo(() => {
    if (!data?.speaker_names) return {};
    return data.speaker_names;
  }, [data]);

  const speakers = useMemo(() => {
    const set = new Set();
    for (const s of transcript || []) {
      if (s?.speaker) set.add(s.speaker);
    }
    return Array.from(set);
  }, [transcript]);

  const clipStart = isClip || clip
    ? Number(clip?.start_time || 0)
    : 0;
  const clipEnd = isClip || clip
    ? Number(clip?.end_time || info?.duration || 0)
    : Number(info?.duration || data?.duration || 0);

  const aspectRatio = clipSettings?.aspectRatio || null;

  // ── Render states ────────────────────────────────────────────────
  if (loading) {
    return (
      <Frame>
        <div style={{ color: '#aaa', textAlign: 'center', padding: 40 }}>Loading share…</div>
      </Frame>
    );
  }
  if (error) {
    return (
      <Frame>
        <div style={errBlock}>
          <h2 style={{ marginTop: 0 }}>This share link can't be opened</h2>
          <p style={{ color: '#aaa', fontSize: 13 }}>{error}</p>
          <p style={{ color: '#777', fontSize: 12 }}>
            The owner may have revoked it, or it has expired.
          </p>
        </div>
      </Frame>
    );
  }
  if (!info || !data) return null;

  const filename = info.filename || data?.filename || 'Shared video';
  const title = clip?.title || filename;

  return (
    <Frame wide>
      <header style={{ marginBottom: 20 }}>
        <div style={{ fontSize: 11, letterSpacing: 1.5, textTransform: 'uppercase', color: '#777' }}>
          Shared {isClip ? 'clip' : 'analysis'} · read-only preview
        </div>
        <h1 style={{ margin: '6px 0 6px', fontSize: 22, color: '#eee' }}>{title}</h1>
        <div style={{ fontSize: 12, color: '#999' }}>
          {info.duration ? `${Math.round(info.duration)}s` : ''}
          {info.resolution ? `  ·  ${info.resolution}` : ''}
          {info.expires_at ? `  ·  expires ${info.expires_at.split('T')[0]}` : ''}
        </div>
      </header>

      {/* Interactive video editor preview — mirrors the SEO page layout
          so the recipient sees exactly what the owner's preview looks
          like, including subtitles, reframe window, and speaker
          coloring. Exports won't work (the endpoint requires auth)
          but every preview-side interaction does. */}
      {videoSrc && clipEnd > clipStart && (
        <section style={{ ...card, padding: 0, overflow: 'hidden' }}>
          <EditorErrorBoundary
            name="Shared preview"
            fallbackHint="Refresh the page to try again. If it keeps failing, ask the owner to re-share this clip."
          >
            <VideoEditor
              src={videoSrc}
              shareToken={token}
              clipStart={clipStart}
              clipEnd={clipEnd}
              title={title}
              aspectRatio={aspectRatio}
              sourceWidth={sourceDims.w}
              sourceHeight={sourceDims.h}
              scenes={scenes}
              sceneCuts={sceneCuts}
              subjectTrack={subjectTrack}
              transcript={transcript}
              speakers={speakers}
              speakerNames={speakerNames}
              initialVolume={playbackVolume}
              initialSpeed={playbackSpeed}
              onTimeUpdate={setCurrentTime}
              onVolumeChange={setPlaybackVolume}
              onSpeedChange={setPlaybackSpeed}
              settings={clipSettings}
              onSettingsChange={setClipSettings}
              jobId={data?.job_id || info?.job_id}
              clipId={clip?.id || null}
              onAspectRatioChange={(ar) =>
                setClipSettings((prev) => ({ ...prev, aspectRatio: ar }))
              }
              subtitleOverlay={
                <SubtitleOverlay
                  currentTime={currentTime}
                  transcript={transcript}
                  clipStart={clipStart}
                  clipEnd={clipEnd}
                  settings={clipSettings}
                  aspectRatio={aspectRatio}
                  sourceWidth={sourceDims.w}
                  sourceHeight={sourceDims.h}
                />
              }
              compact
            />
          </EditorErrorBoundary>
        </section>
      )}

      {/* Metadata + transcript download live BELOW the player so the
          recipient's eye lands on the video first. */}
      {isClip && clip && <ClipMetaBlock clip={clip} token={token} />}
      {!isClip && data && <JobMetaBlock job={data} token={token} />}

      <footer style={{
        marginTop: 32, padding: 16, borderTop: '1px solid #26263a',
        color: '#777', fontSize: 11, textAlign: 'center',
      }}>
        Powered by ClipAI · this is a shared read-only view
      </footer>
    </Frame>
  );
}

function ClipMetaBlock({ clip, token }) {
  return (
    <section style={card}>
      <h3 style={h3}>Clip details</h3>
      <Stat label="Title" value={clip.title || '—'} />
      <Stat label="Hook" value={clip.hook || '—'} />
      <Stat label="Window" value={clip.start_time != null ? `${clip.start_time.toFixed(2)}s → ${clip.end_time?.toFixed(2)}s` : '—'} />
      <Stat label="Duration" value={clip.duration ? `${clip.duration.toFixed(2)}s` : '—'} />
      <Stat label="Viral score" value={clip.viral_score != null ? clip.viral_score : '—'} />
      {clip.viral_reason && (
        <Stat label="Why" value={clip.viral_reason} />
      )}
      {(clip.hashtags?.length > 0) && (
        <Stat label="Hashtags" value={(clip.hashtags || []).map((t) => t.startsWith('#') ? t : `#${t}`).join(' ')} />
      )}
      <a
        href={`/api/share/public/${encodeURIComponent(token)}/transcript.srt`}
        download
        style={btnLink}
      >
        Download transcript (SRT)
      </a>
    </section>
  );
}

function JobMetaBlock({ job, token }) {
  return (
    <>
      {job.summary && (
        <section style={card}>
          <h3 style={h3}>Summary</h3>
          <p style={{ color: '#ccc', lineHeight: 1.6, fontSize: 14 }}>
            {job.summary.overview || ''}
          </p>
          {job.summary.key_points?.length > 0 && (
            <ul style={{ color: '#bbb', fontSize: 13 }}>
              {job.summary.key_points.map((p, i) => <li key={i}>{p}</li>)}
            </ul>
          )}
        </section>
      )}

      {job.clips?.length > 0 && (
        <section style={card}>
          <h3 style={h3}>Clips ({job.clips.length})</h3>
          <ul style={{ listStyle: 'none', padding: 0, margin: 0 }}>
            {job.clips.map((c) => (
              <li key={c.id} style={{ padding: '10px 0', borderTop: '1px solid #26263a' }}>
                <div style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between', gap: 8 }}>
                  <div style={{ fontWeight: 600, color: '#eee' }}>
                    Clip {c.id}: {c.title || c.hook || '(untitled)'}
                  </div>
                  <div style={{ fontSize: 11, color: '#888' }}>
                    {c.start_time != null && c.end_time != null
                      ? `${c.start_time.toFixed(1)}s → ${c.end_time.toFixed(1)}s`
                      : ''}
                  </div>
                </div>
                {c.viral_score != null && (
                  <div style={{ fontSize: 11, color: '#a855f7' }}>Score {c.viral_score}</div>
                )}
                {c.viral_reason && (
                  <div style={{ fontSize: 12, color: '#aaa', marginTop: 2 }}>{c.viral_reason}</div>
                )}
              </li>
            ))}
          </ul>
        </section>
      )}

      <a
        href={`/api/share/public/${encodeURIComponent(token)}/transcript.srt`}
        download
        style={{ ...btnLink, display: 'inline-block', marginTop: 0 }}
      >
        Download full transcript (SRT)
      </a>
    </>
  );
}

function Stat({ label, value }) {
  return (
    <div style={{ display: 'flex', gap: 12, padding: '6px 0', borderBottom: '1px solid #26263a' }}>
      <div style={{ width: 110, fontSize: 11, color: '#888', textTransform: 'uppercase', letterSpacing: 0.5 }}>{label}</div>
      <div style={{ flex: 1, fontSize: 13, color: '#ddd' }}>{String(value)}</div>
    </div>
  );
}

function Frame({ children, wide }) {
  return (
    <div style={{
      minHeight: '100vh',
      background: '#0a0a0f',
      color: '#eee',
      fontFamily: 'system-ui, -apple-system, sans-serif',
      padding: '32px 24px',
    }}>
      <div style={{ maxWidth: wide ? 1180 : 760, margin: '0 auto' }}>{children}</div>
    </div>
  );
}

const card = {
  background: '#15152a',
  border: '1px solid #26263a',
  borderRadius: 8,
  padding: 16,
  marginBottom: 16,
};
const h3 = { margin: '0 0 12px', fontSize: 15, color: '#eee' };
const errBlock = {
  background: '#15152a',
  border: '1px solid rgba(239, 68, 68, 0.4)',
  borderRadius: 8,
  padding: 24,
  textAlign: 'center',
  color: '#ffb4b4',
};
const btnLink = {
  display: 'inline-block',
  marginTop: 12,
  padding: '8px 14px',
  background: '#00D9FF',
  color: '#000',
  borderRadius: 6,
  fontSize: 12,
  fontWeight: 600,
  textDecoration: 'none',
};
