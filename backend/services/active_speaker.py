"""Active speaker detection via lip-audio cross-correlation.

Correlates lip aperture ratio (from FaceMesh) with speech timing
(from Whisper transcript) to determine which face is speaking at
each point in the video.

Pipeline:
  1. Face detection produces lip_aperture per face per frame
  2. Whisper produces word-level timestamps
  3. For each speech segment, find frames near that time
  4. The face with the highest lip aperture during speech = active speaker
  5. Map active speaker to face registry slot
  6. Build a timeline: [(timestamp, active_slot_id), ...]
"""
import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)

LAR_SPEAKING_THRESHOLD = 0.025
# Continuity bonus raised from 0.015 → 0.04 so it clears the speaking
# threshold. This makes speaker slots sticky across single-frame
# laughter/yawn artifacts from the listener.
CONTINUITY_BONUS = 0.04
MIN_DWELL_SECONDS = 0.6  # Collapse runs shorter than this
MIN_DWELL_WIN_MARGIN = 0.08  # ...unless a competing slot beats us by this
VAD_BRIDGE_GAP_SECONDS = 1.0  # Extend a SpeakerEvent into voiced VAD that starts within this gap


@dataclass
class SpeakerEvent:
    """A period where a specific face slot is the active speaker."""
    start: float
    end: float
    slot_id: int
    confidence: float
    # False when the speaker's face is not visible in the current shot
    # (e.g. over-the-shoulder reaction shot that spans a shot cut).
    on_screen: bool = True


def _compute_lip_motion_score(
    face_results: list,
    slot_id: int,
    start_time: float,
    end_time: float,
    face_registry=None,
) -> float:
    """Score how much a face's lips are MOVING (not just open).

    At >= 4fps: uses rate of change (derivative) to detect speech oscillation.
    At < 4fps: falls back to size-weighted aperture, because the motion
    derivative is unreliable when sampling below the Nyquist rate for
    speech cadence (4-8 Hz).
    """
    apertures = []
    for fr in face_results:
        if fr.timestamp < start_time - 0.5 or fr.timestamp > end_time + 0.5:
            continue
        for face in fr.faces:
            matched = False
            if face.identity_id >= 0 and face.identity_id == slot_id:
                matched = True
            elif face_registry:
                slot = face_registry.nearest_slot(face.nose_x)
                if slot and slot.slot_id == slot_id:
                    matched = True
            if matched:
                face_w = face.width if hasattr(face, 'width') else 8.0
                apertures.append((fr.timestamp, face.lip_aperture, face_w))
                break

    if not apertures:
        return 0.0
    if len(apertures) < 2:
        return sum(a for _, a, _ in apertures) / len(apertures)

    apertures.sort(key=lambda x: x[0])

    # Check effective sample rate
    total_dt = apertures[-1][0] - apertures[0][0]
    effective_fps = (len(apertures) - 1) / max(total_dt, 0.01) if total_dt > 0 else 0

    avg_aperture = sum(a for _, a, _ in apertures) / len(apertures)
    avg_face_width = sum(w for _, _, w in apertures) / len(apertures)

    if effective_fps < 4.0:
        # Low fps: lip motion derivative is noise.
        # Use size-weighted aperture — larger faces have more reliable
        # lip measurements and are more likely to be the actual speaker.
        size_weight = min(2.0, avg_face_width / 8.0)
        return min(1.0, avg_aperture * size_weight / 0.04)

    # High fps: use motion analysis
    deltas = []
    for i in range(1, len(apertures)):
        dt = apertures[i][0] - apertures[i - 1][0]
        if dt > 0:
            da = abs(apertures[i][1] - apertures[i - 1][1])
            deltas.append(da / dt)

    if not deltas:
        return 0.0

    avg_motion = sum(deltas) / len(deltas)
    motion_score = min(1.0, avg_motion / 0.08)
    aperture_score = min(1.0, avg_aperture / 0.04)

    return motion_score * 0.6 + aperture_score * 0.4


def _compute_lip_audio_sync(
    face_results: list,
    slot_id: int,
    transcript_segment,
    face_registry=None,
) -> float:
    """Score lip-audio synchronization for a face during a transcript segment.

    Computes correlation between word boundaries and lip motion.
    High sync = this face is actually producing the speech audio.

    Returns: sync score 0.0-1.0
    """
    words = getattr(transcript_segment, 'words', None)
    if not words or len(words) < 2:
        return 0.5  # Neutral — can't verify

    seg_start = transcript_segment.start if hasattr(transcript_segment, 'start') else transcript_segment.get('start', 0)
    seg_end = transcript_segment.end if hasattr(transcript_segment, 'end') else transcript_segment.get('end', 0)

    lip_timeline = []
    for fr in face_results:
        if fr.timestamp < seg_start - 0.2 or fr.timestamp > seg_end + 0.2:
            continue
        for face in fr.faces:
            matched = (face.identity_id == slot_id) if face.identity_id >= 0 else False
            if not matched and face_registry:
                slot = face_registry.nearest_slot(face.nose_x)
                matched = slot and slot.slot_id == slot_id
            if matched:
                lip_timeline.append((fr.timestamp, face.lip_aperture))
                break

    if len(lip_timeline) < 3:
        return 0.5

    word_sync_scores = []
    for w in words:
        w_start = w.start if hasattr(w, 'start') else w.get('start', 0)
        w_end = w.end if hasattr(w, 'end') else w.get('end', 0)

        during = [a for t, a in lip_timeline if w_start - 0.1 <= t <= w_end + 0.1]
        outside = [a for t, a in lip_timeline if t < w_start - 0.1 or t > w_end + 0.1]

        if during and outside:
            avg_during = sum(during) / len(during)
            avg_outside = sum(outside) / len(outside)
            if avg_outside > 0:
                ratio = avg_during / avg_outside
                word_sync_scores.append(min(1.0, ratio / 2.0))
            else:
                word_sync_scores.append(1.0 if avg_during > 0.02 else 0.0)

    return sum(word_sync_scores) / len(word_sync_scores) if word_sync_scores else 0.5


def build_vad_presence(
    audio_path: str,
    sample_rate: int = 16000,
) -> list[tuple[float, float]]:
    """Return a list of (start, end) voiced intervals in seconds.

    Prefers `webrtcvad` (mode=2, 30ms frames). Falls back to a pure-numpy
    energy + ZCR gate when webrtcvad is unavailable or audio loading fails.

    Used to bridge Whisper silence gaps (breaths, laughter, reaction beats)
    so SpeakerEvents can extend across the gap. Silently returns [] on any
    failure — callers must treat "no VAD data" as acceptable.
    """
    if not audio_path:
        return []

    # Load the audio → mono int16 at sample_rate using soundfile or wave.
    try:
        import numpy as np
        samples = None
        try:
            import soundfile as sf
            data, sr = sf.read(audio_path, dtype="int16", always_2d=False)
            if hasattr(data, "ndim") and data.ndim > 1:
                data = data.mean(axis=1).astype("int16")
            if sr != sample_rate:
                # Naive decimation/upsampling via numpy. Good enough for VAD.
                ratio = sr / sample_rate
                idx = (np.arange(int(len(data) / ratio)) * ratio).astype(int)
                idx = idx[idx < len(data)]
                data = data[idx].astype("int16")
            samples = data
        except Exception:
            import wave
            with wave.open(audio_path, "rb") as wf:
                nch = wf.getnchannels()
                sw = wf.getsampwidth()
                sr = wf.getframerate()
                raw = wf.readframes(wf.getnframes())
            if sw != 2:
                return []
            data = np.frombuffer(raw, dtype=np.int16)
            if nch > 1:
                data = data.reshape(-1, nch).mean(axis=1).astype(np.int16)
            if sr != sample_rate:
                ratio = sr / sample_rate
                idx = (np.arange(int(len(data) / ratio)) * ratio).astype(int)
                idx = idx[idx < len(data)]
                data = data[idx].astype(np.int16)
            samples = data
        if samples is None or len(samples) == 0:
            return []
    except Exception as e:
        logger.info("[SpeakerV2] VAD: audio load failed (%s) — skipping", e)
        return []

    frame_ms = 30
    frame_len = int(sample_rate * frame_ms / 1000)
    if frame_len <= 0 or len(samples) < frame_len:
        return []

    voiced_frames: list[bool] = []

    # ── Preferred: webrtcvad ──
    try:
        import webrtcvad  # type: ignore
        vad = webrtcvad.Vad(2)
        for i in range(0, len(samples) - frame_len + 1, frame_len):
            frame_bytes = samples[i:i + frame_len].tobytes()
            try:
                voiced_frames.append(vad.is_speech(frame_bytes, sample_rate))
            except Exception:
                voiced_frames.append(False)
    except Exception:
        # ── Fallback: numpy energy + ZCR gate ──
        try:
            import numpy as np
            # Work in float32 for RMS.
            f = samples.astype(np.float32) / 32768.0
            n_frames = (len(f) - frame_len) // frame_len + 1
            rms_all = np.empty(n_frames, dtype=np.float32)
            zcr_all = np.empty(n_frames, dtype=np.float32)
            for k in range(n_frames):
                chunk = f[k * frame_len:(k + 1) * frame_len]
                rms_all[k] = float(np.sqrt(np.mean(chunk * chunk) + 1e-12))
                # Zero-crossing rate
                signs = np.sign(chunk)
                zcr_all[k] = float(np.mean(np.abs(np.diff(signs)) > 0))
            median_rms = float(np.median(rms_all))
            thresh = max(1e-4, 0.5 * median_rms)
            for k in range(n_frames):
                voiced_frames.append(
                    bool(rms_all[k] > thresh and 0.01 <= zcr_all[k] <= 0.3)
                )
        except Exception as e:
            logger.info("[SpeakerV2] VAD fallback failed (%s) — skipping", e)
            return []

    # Collapse voiced frames into intervals.
    intervals: list[tuple[float, float]] = []
    in_voice = False
    v_start = 0.0
    for k, is_v in enumerate(voiced_frames):
        t = k * frame_ms / 1000.0
        if is_v and not in_voice:
            in_voice = True
            v_start = t
        elif not is_v and in_voice:
            in_voice = False
            intervals.append((v_start, t))
    if in_voice:
        intervals.append((v_start, len(voiced_frames) * frame_ms / 1000.0))

    # Merge intervals with gaps < 150ms (smooths out single non-voiced blips).
    merged: list[tuple[float, float]] = []
    for iv in intervals:
        if merged and iv[0] - merged[-1][1] < 0.15:
            merged[-1] = (merged[-1][0], iv[1])
        else:
            merged.append(iv)
    logger.info("[SpeakerV2] VAD: %d voiced intervals from %s", len(merged), audio_path)
    return merged


def _apply_vad_bridging(
    events: list[SpeakerEvent],
    vad_intervals: list[tuple[float, float]],
) -> tuple[list[SpeakerEvent], int]:
    """Extend each SpeakerEvent's end into VAD-voiced intervals that begin
    within VAD_BRIDGE_GAP_SECONDS of the event end, without overlapping a
    different-speaker event that follows. Returns (new_events, n_extended).
    """
    if not events or not vad_intervals:
        return events, 0
    extended = 0
    out: list[SpeakerEvent] = []
    for i, ev in enumerate(events):
        next_different = None
        for j in range(i + 1, len(events)):
            if events[j].slot_id != ev.slot_id and events[j].slot_id >= 0:
                next_different = events[j]
                break
        new_end = ev.end
        for (vs, ve) in vad_intervals:
            # Only VAD intervals that *start* within the gap window
            gap = vs - new_end
            if gap < 0 or gap > VAD_BRIDGE_GAP_SECONDS:
                continue
            target_end = ve
            if next_different is not None:
                target_end = min(target_end, next_different.start)
            if target_end > new_end:
                new_end = target_end
                extended += 1
        if new_end != ev.end:
            out.append(SpeakerEvent(
                start=ev.start, end=new_end,
                slot_id=ev.slot_id, confidence=ev.confidence,
                on_screen=getattr(ev, "on_screen", True),
            ))
        else:
            out.append(ev)
    return out, extended


def _collapse_short_runs(
    events: list[SpeakerEvent],
    min_dwell: float = MIN_DWELL_SECONDS,
    win_margin: float = MIN_DWELL_WIN_MARGIN,
) -> tuple[list[SpeakerEvent], int]:
    """Collapse runs of same-speaker events shorter than min_dwell into the
    neighbor on the higher-scoring side, unless the short run's speaker beats
    BOTH neighbors by at least win_margin.

    This is the "minimum-dwell" rule from the spec: once slot S becomes
    active, it stays active for ≥ min_dwell of subsequent segments unless
    another slot beats S by ≥ win_margin.

    Returns (collapsed_events, n_runs_collapsed).
    """
    if len(events) < 3:
        return events, 0

    # First merge consecutive same-slot events into runs.
    runs: list[SpeakerEvent] = [events[0]]
    for ev in events[1:]:
        prev = runs[-1]
        if ev.slot_id == prev.slot_id:
            runs[-1] = SpeakerEvent(
                start=prev.start, end=ev.end,
                slot_id=prev.slot_id,
                confidence=max(prev.confidence, ev.confidence),
                on_screen=getattr(prev, "on_screen", True) and getattr(ev, "on_screen", True),
            )
        else:
            runs.append(ev)

    if len(runs) < 3:
        return runs, 0

    collapsed_count = 0
    changed = True
    while changed and len(runs) >= 3:
        changed = False
        for i in range(1, len(runs) - 1):
            run = runs[i]
            duration = run.end - run.start
            if duration >= min_dwell:
                continue
            prev = runs[i - 1]
            nxt = runs[i + 1]
            # Keep the short run only if it beats BOTH neighbors by the margin.
            beats_prev = run.confidence - prev.confidence >= win_margin
            beats_next = run.confidence - nxt.confidence >= win_margin
            if beats_prev and beats_next:
                continue
            # Collapse into the higher-scoring neighbor.
            if prev.confidence >= nxt.confidence:
                runs[i - 1] = SpeakerEvent(
                    start=prev.start, end=run.end,
                    slot_id=prev.slot_id, confidence=prev.confidence,
                    on_screen=getattr(prev, "on_screen", True),
                )
                del runs[i]
            else:
                runs[i + 1] = SpeakerEvent(
                    start=run.start, end=nxt.end,
                    slot_id=nxt.slot_id, confidence=nxt.confidence,
                    on_screen=getattr(nxt, "on_screen", True),
                )
                del runs[i]
            collapsed_count += 1
            changed = True
            break
    # Final same-slot merge after collapses
    merged: list[SpeakerEvent] = [runs[0]]
    for ev in runs[1:]:
        prev = merged[-1]
        if ev.slot_id == prev.slot_id:
            merged[-1] = SpeakerEvent(
                start=prev.start, end=ev.end,
                slot_id=prev.slot_id,
                confidence=max(prev.confidence, ev.confidence),
                on_screen=getattr(prev, "on_screen", True) and getattr(ev, "on_screen", True),
            )
        else:
            merged.append(ev)
    return merged, collapsed_count


def _apply_shot_reverse_tolerance(
    events: list[SpeakerEvent],
    transcript_segments: list,
    face_results: list,
    shot_cuts: list,
    face_registry=None,
) -> tuple[list[SpeakerEvent], int]:
    """Mark events as on_screen=False when a shot-cut lands inside a
    transcript segment and the new shot has no face with lip_aperture > 0.03.

    This captures over-the-shoulder / reaction shots where the *listener*
    is visible while the previous speaker continues to talk. The crop then
    anchors on the listener (reaction-shot framing) — see the logic in
    required_regions.build_required_regions() — rather than fabricating a
    speaker anchor.

    Returns (new_events, n_offscreen).
    """
    if not events or not shot_cuts or not transcript_segments or not face_results:
        return events, 0

    sorted_cuts = sorted(float(c) for c in shot_cuts)

    def _has_active_lips_between(t0: float, t1: float) -> bool:
        for fr in face_results:
            if fr.timestamp < t0 or fr.timestamp > t1:
                continue
            for face in fr.faces:
                if getattr(face, "lip_aperture", 0.0) > 0.03:
                    return True
        return False

    # Build a timestamp → transcript segment lookup.
    def _seg_containing(t: float):
        for seg in transcript_segments:
            s = seg.start if hasattr(seg, "start") else seg.get("start", 0)
            e = seg.end if hasattr(seg, "end") else seg.get("end", 0)
            if s <= t <= e:
                return (s, e)
        return None

    offscreen = 0
    out: list[SpeakerEvent] = []
    for ev in events:
        on_screen = getattr(ev, "on_screen", True)
        if ev.slot_id < 0:
            out.append(ev)
            continue
        # Find any shot cut inside this event's time range.
        cuts_in = [c for c in sorted_cuts if ev.start < c < ev.end]
        for cut in cuts_in:
            seg_range = _seg_containing(cut)
            if seg_range is None:
                continue
            seg_s, seg_e = seg_range
            # Transcript segment must extend past the cut.
            if seg_e <= cut + 0.05:
                continue
            post_end = min(ev.end, seg_e)
            if not _has_active_lips_between(cut, post_end):
                on_screen = False
                offscreen += 1
                break
        if on_screen != getattr(ev, "on_screen", True):
            out.append(SpeakerEvent(
                start=ev.start, end=ev.end,
                slot_id=ev.slot_id, confidence=ev.confidence,
                on_screen=on_screen,
            ))
        else:
            out.append(ev)
    return out, offscreen


def build_active_speaker_timeline(
    face_results: list,
    transcript_segments: list,
    face_registry=None,
    window_seconds: float = 2.0,
    shot_cuts: list = None,
    audio_path: str = None,
) -> list[SpeakerEvent]:
    """Build a timeline of who is speaking when.

    For each transcript segment, finds nearby face detection frames
    and checks which face has the highest lip aperture. Maps that
    face to a face registry slot.
    """
    if not face_results or not transcript_segments:
        return []

    has_lip_data = any(
        f.lip_aperture > 0
        for fr in face_results
        for f in fr.faces
    )
    if not has_lip_data:
        logger.info("No lip aperture data available — skipping active speaker detection")
        return []

    frame_map = {}
    for fr in face_results:
        frame_map[fr.timestamp] = fr
    frame_times = sorted(frame_map.keys())

    events = []

    for seg in transcript_segments:
        seg_start = seg.start if hasattr(seg, 'start') else seg.get('start', 0)
        seg_end = seg.end if hasattr(seg, 'end') else seg.get('end', 0)

        nearby_frames = [
            frame_map[ft]
            for ft in frame_times
            if seg_start - window_seconds <= ft <= seg_end + window_seconds
        ]

        if not nearby_frames:
            events.append(SpeakerEvent(
                start=seg_start, end=seg_end, slot_id=-1, confidence=0.0
            ))
            continue

        if face_registry and face_registry.multi_speaker:
            slot_scores: dict[int, list[float]] = {}
            for fr in nearby_frames:
                for face in fr.faces:
                    slot = face_registry.nearest_slot(face.nose_x)
                    if slot:
                        sid = slot.slot_id
                        if sid not in slot_scores:
                            slot_scores[sid] = []
                        # Weight by face size — larger faces have more reliable lip data
                        size_weight = min(2.0, face.width / 8.0)
                        slot_scores[sid].append(face.lip_aperture * size_weight)

            if slot_scores:
                # Use lip motion score (rate of change) instead of raw aperture
                slot_motion_scores = {}
                for sid in slot_scores:
                    slot_motion_scores[sid] = _compute_lip_motion_score(
                        face_results, sid, seg_start, seg_end, face_registry
                    )

                # Audio sync: boost score for face whose lips correlate with words
                for sid in slot_motion_scores:
                    sync = _compute_lip_audio_sync(
                        face_results, sid, seg, face_registry
                    )
                    # Sync score 0.5 = neutral, >0.5 = boost, <0.5 = penalty
                    slot_motion_scores[sid] *= (0.7 + 0.6 * sync)

                # Continuity prior: bias toward previous speaker
                if events and events[-1].slot_id >= 0:
                    prev_sid = events[-1].slot_id
                    if prev_sid in slot_motion_scores:
                        slot_motion_scores[prev_sid] += CONTINUITY_BONUS

                best_slot_id = max(slot_motion_scores, key=slot_motion_scores.get)
                avg_lar = slot_motion_scores[best_slot_id]
                # Only assign a speaker if LAR is above the speaking threshold.
                # When no one is clearly speaking (both slots have near-zero LAR),
                # emit slot_id=-1 so the pipeline doesn't incorrectly override
                # the visual subject_x with a random slot pick.
                if avg_lar >= LAR_SPEAKING_THRESHOLD:
                    events.append(SpeakerEvent(
                        start=seg_start, end=seg_end,
                        slot_id=best_slot_id,
                        confidence=min(1.0, avg_lar / 0.05),
                    ))
                else:
                    events.append(SpeakerEvent(
                        start=seg_start, end=seg_end,
                        slot_id=-1, confidence=min(1.0, avg_lar / 0.05),
                    ))
                continue

        # Fallback: pick face with highest lip aperture
        best_face_x = None
        best_lar = 0.0
        for fr in nearby_frames:
            for face in fr.faces:
                if face.lip_aperture > best_lar:
                    best_lar = face.lip_aperture
                    best_face_x = face.nose_x
        if best_face_x is not None and best_lar >= LAR_SPEAKING_THRESHOLD and face_registry:
            slot = face_registry.nearest_slot(best_face_x)
            events.append(SpeakerEvent(
                start=seg_start, end=seg_end,
                slot_id=slot.slot_id if slot else -1,
                confidence=min(1.0, best_lar / 0.05),
            ))
        else:
            events.append(SpeakerEvent(
                start=seg_start, end=seg_end, slot_id=-1, confidence=0.0
            ))

    # Merge consecutive events with the same speaker
    if events:
        merged = [events[0]]
        for ev in events[1:]:
            if ev.slot_id == merged[-1].slot_id and ev.start - merged[-1].end < 1.0:
                merged[-1] = SpeakerEvent(
                    start=merged[-1].start, end=ev.end,
                    slot_id=ev.slot_id,
                    confidence=max(merged[-1].confidence, ev.confidence),
                )
            else:
                merged.append(ev)
        events = merged

    # ── Asymmetric hysteresis: fast attack, slower release ──
    # Replaces the old "dominant speaker momentum" symmetric smoothing.
    # Two-threshold state machine:
    #   Attack: confidence >= ATTACK_THRESHOLD → emit switch immediately
    #   Release: confidence < RELEASE_THRESHOLD → release lock
    # The event is back-dated to the attack-trigger timestamp so the crop
    # arrives *with* the speaker, not after.
    ATTACK_THRESHOLD = 0.7
    RELEASE_THRESHOLD = 0.3

    if events and len(events) >= 2:
        refined = [events[0]]
        for ev in events[1:]:
            prev = refined[-1]
            if ev.slot_id != prev.slot_id and ev.slot_id >= 0:
                # Attack: new speaker with high confidence → immediate switch
                if ev.confidence >= ATTACK_THRESHOLD:
                    refined.append(ev)
                # Low confidence switch → only accept if previous speaker
                # has released (low confidence in previous)
                elif prev.confidence < RELEASE_THRESHOLD:
                    refined.append(ev)
                else:
                    # Suppress: extend previous speaker through this segment
                    refined[-1] = SpeakerEvent(
                        start=prev.start, end=ev.end,
                        slot_id=prev.slot_id,
                        confidence=prev.confidence * 0.95,
                    )
            elif ev.slot_id == prev.slot_id:
                # Same speaker: merge
                refined[-1] = SpeakerEvent(
                    start=prev.start, end=ev.end,
                    slot_id=ev.slot_id,
                    confidence=max(prev.confidence, ev.confidence),
                )
            else:
                refined.append(ev)

        suppressed = len(events) - len(refined)
        if suppressed > 0:
            logger.info(
                "Asymmetric hysteresis: suppressed %d low-confidence switches "
                "(attack=%.1f, release=%.1f)",
                suppressed, ATTACK_THRESHOLD, RELEASE_THRESHOLD,
            )
        events = refined

    # ── Minimum-dwell collapse ──
    events, collapsed_runs = _collapse_short_runs(events)

    # ── Shot-reverse (off-screen speaker) detection ──
    events, offscreen_count = _apply_shot_reverse_tolerance(
        events, transcript_segments, face_results, shot_cuts or [], face_registry,
    )

    # ── VAD bridging ──
    if audio_path:
        try:
            vad_intervals = build_vad_presence(audio_path)
        except Exception as e:
            logger.info("[SpeakerV2] VAD bridging skipped (%s)", e)
            vad_intervals = []
        if vad_intervals:
            events, bridged_count = _apply_vad_bridging(events, vad_intervals)
            if bridged_count > 0:
                logger.info("[SpeakerV2] VAD bridged %d speaker event extensions", bridged_count)

    logger.info(
        "[SpeakerV2] offscreen_speaker_events=%d collapsed_short_runs=%d",
        offscreen_count, collapsed_runs,
    )

    logger.info(
        "Active speaker timeline: %d events, %d unique speakers, avg confidence=%.2f",
        len(events),
        len(set(e.slot_id for e in events if e.slot_id >= 0)),
        sum(e.confidence for e in events) / max(len(events), 1),
    )

    slot_times: dict[int, float] = {}
    for ev in events:
        if ev.slot_id >= 0:
            slot_times[ev.slot_id] = slot_times.get(ev.slot_id, 0) + (ev.end - ev.start)
    for sid, secs in sorted(slot_times.items()):
        logger.info("  Slot %d speaking: %.1fs (%.0f%%)",
                     sid, secs, secs / max(sum(slot_times.values()), 1) * 100)

    return events


def get_active_slot_at_time(events: list[SpeakerEvent], timestamp: float) -> int:
    """Get the active speaker slot ID at a given timestamp."""
    if not events:
        return -1
    for ev in events:
        if ev.start <= timestamp <= ev.end:
            return ev.slot_id
    prev = [ev for ev in events if ev.end <= timestamp]
    if prev:
        return prev[-1].slot_id
    return events[0].slot_id


def build_active_speaker_timeline_v2(
    face_results: list,
    transcript_segments: list,
    face_registry=None,
    window_seconds: float = 1.0,
    shot_cuts: list = None,
    audio_path: str = None,
) -> list[SpeakerEvent]:
    """V2: Uses face identity embeddings for accurate speaker tracking.

    Improvements over V1:
    1. Uses identity_id (from embeddings) instead of positional matching
    2. Considers lip aperture AND transcript timing simultaneously
    3. Handles speaker overlap (both speaking) by selecting the primary
    4. Produces events at higher temporal resolution (1s windows vs 2s)
    5. Outputs a confidence score per event based on lip-audio correlation
    """
    if not face_results or not transcript_segments:
        return []

    has_identity = any(
        f.identity_id >= 0
        for fr in face_results
        for f in fr.faces
    )
    if not has_identity:
        logger.info("No identity data — falling back to V1 active speaker detection")
        return build_active_speaker_timeline(
            face_results, transcript_segments, face_registry, window_seconds,
            shot_cuts=shot_cuts, audio_path=audio_path,
        )

    has_lip_data = any(
        f.lip_aperture > 0
        for fr in face_results
        for f in fr.faces
    )
    if not has_lip_data:
        logger.info("No lip aperture data available — skipping active speaker detection")
        return []

    frame_map = {}
    for fr in face_results:
        frame_map[fr.timestamp] = fr
    frame_times = sorted(frame_map.keys())

    events = []

    for seg in transcript_segments:
        seg_start = seg.start if hasattr(seg, 'start') else seg.get('start', 0)
        seg_end = seg.end if hasattr(seg, 'end') else seg.get('end', 0)

        nearby_frames = [
            frame_map[ft]
            for ft in frame_times
            if seg_start - window_seconds <= ft <= seg_end + window_seconds
        ]

        if not nearby_frames:
            events.append(SpeakerEvent(
                start=seg_start, end=seg_end, slot_id=-1, confidence=0.0
            ))
            continue

        # Score each identity by lip motion + size-weighted aperture
        identity_scores: dict[int, list[float]] = {}
        for fr in nearby_frames:
            for face in fr.faces:
                if face.identity_id < 0:
                    continue
                size_weight = min(2.0, face.width / 8.0)
                identity_scores.setdefault(face.identity_id, []).append(
                    face.lip_aperture * size_weight
                )

        if identity_scores:
            # Use lip motion score for each identity
            id_motion_scores = {}
            for iid in identity_scores:
                id_motion_scores[iid] = _compute_lip_motion_score(
                    face_results, iid, seg_start, seg_end, face_registry
                )

            # Audio sync: boost score for face whose lips correlate with words
            for iid in id_motion_scores:
                sync = _compute_lip_audio_sync(
                    face_results, iid, seg, face_registry
                )
                id_motion_scores[iid] *= (0.7 + 0.6 * sync)

            # Continuity prior
            if events and events[-1].slot_id >= 0:
                prev_sid = events[-1].slot_id
                if prev_sid in id_motion_scores:
                    id_motion_scores[prev_sid] += CONTINUITY_BONUS

            best_id = max(id_motion_scores, key=id_motion_scores.get)
            avg_lar = id_motion_scores[best_id]

            # Mark speaking faces
            for fr in nearby_frames:
                for face in fr.faces:
                    if face.identity_id == best_id:
                        face.is_speaking = True

            events.append(SpeakerEvent(
                start=seg_start, end=seg_end,
                slot_id=best_id,
                confidence=min(1.0, avg_lar / 0.05),
            ))
        else:
            events.append(SpeakerEvent(
                start=seg_start, end=seg_end, slot_id=-1, confidence=0.0
            ))

    # Merge consecutive events with the same speaker
    if events:
        merged = [events[0]]
        for ev in events[1:]:
            if ev.slot_id == merged[-1].slot_id and ev.start - merged[-1].end < 1.0:
                merged[-1] = SpeakerEvent(
                    start=merged[-1].start, end=ev.end,
                    slot_id=ev.slot_id,
                    confidence=max(merged[-1].confidence, ev.confidence),
                )
            else:
                merged.append(ev)
        events = merged

    # ── Dominant speaker momentum ──
    if events:
        cumulative_time: dict[int, float] = {}
        for ev in events:
            if ev.slot_id >= 0:
                cumulative_time[ev.slot_id] = cumulative_time.get(ev.slot_id, 0) + (ev.end - ev.start)

        if cumulative_time:
            total_time = sum(cumulative_time.values())
            dominant_slot = max(cumulative_time, key=cumulative_time.get)
            dominant_frac = cumulative_time[dominant_slot] / max(total_time, 0.1)

            num_speakers = len([s for s in cumulative_time if cumulative_time[s] > 5.0])
            if num_speakers >= 4:
                MIN_SWITCH_CONFIDENCE = 0.3 if dominant_frac > 0.70 else None
            elif dominant_frac > 0.4:
                MIN_SWITCH_CONFIDENCE = 0.5
            else:
                MIN_SWITCH_CONFIDENCE = None

            if MIN_SWITCH_CONFIDENCE is not None:
                reverted = 0
                for i in range(1, len(events)):
                    if (events[i - 1].slot_id == dominant_slot and
                            events[i].slot_id != dominant_slot and
                            events[i].confidence < MIN_SWITCH_CONFIDENCE):
                        events[i] = SpeakerEvent(
                            start=events[i].start, end=events[i].end,
                            slot_id=dominant_slot,
                            confidence=events[i - 1].confidence * 0.9,
                        )
                        reverted += 1
                if reverted > 0:
                    logger.info(
                        "V2 dominant speaker momentum: slot %d (%.0f%%, %d speakers), "
                        "reverted %d switches (threshold=%.1f)",
                        dominant_slot, dominant_frac * 100, num_speakers,
                        reverted, MIN_SWITCH_CONFIDENCE,
                    )
                    merged2 = [events[0]]
                    for ev in events[1:]:
                        if ev.slot_id == merged2[-1].slot_id and ev.start - merged2[-1].end < 1.0:
                            merged2[-1] = SpeakerEvent(
                                start=merged2[-1].start, end=ev.end,
                                slot_id=ev.slot_id,
                                confidence=max(merged2[-1].confidence, ev.confidence),
                            )
                        else:
                            merged2.append(ev)
                    events = merged2

    # ── Minimum-dwell collapse ──
    events, collapsed_runs = _collapse_short_runs(events)

    # ── Shot-reverse (off-screen speaker) detection ──
    events, offscreen_count = _apply_shot_reverse_tolerance(
        events, transcript_segments, face_results, shot_cuts or [], face_registry,
    )

    # ── VAD bridging ──
    if audio_path:
        try:
            vad_intervals = build_vad_presence(audio_path)
        except Exception as e:
            logger.info("[SpeakerV2] VAD bridging skipped (%s)", e)
            vad_intervals = []
        if vad_intervals:
            events, bridged_count = _apply_vad_bridging(events, vad_intervals)
            if bridged_count > 0:
                logger.info("[SpeakerV2] VAD bridged %d speaker event extensions (v2)", bridged_count)

    logger.info(
        "[SpeakerV2] offscreen_speaker_events=%d collapsed_short_runs=%d (v2)",
        offscreen_count, collapsed_runs,
    )

    logger.info(
        "Active speaker V2 timeline: %d events, %d unique speakers, avg confidence=%.2f",
        len(events),
        len(set(e.slot_id for e in events if e.slot_id >= 0)),
        sum(e.confidence for e in events) / max(len(events), 1),
    )
    slot_times: dict[int, float] = {}
    for ev in events:
        if ev.slot_id >= 0:
            slot_times[ev.slot_id] = slot_times.get(ev.slot_id, 0) + (ev.end - ev.start)
    for sid, secs in sorted(slot_times.items()):
        logger.info("  V2 Slot %d speaking: %.1fs (%.0f%%)",
                     sid, secs, secs / max(sum(slot_times.values()), 1) * 100)

    return events


def build_active_speaker_timeline_v3(
    face_results: list,
    transcript_segments: list,
    asd_scores: list,           # list[ASDResult] from light_asd.score_faces_for_clip
    face_registry=None,
    window_seconds: float = 0.5,
    shot_cuts: list = None,
    audio_path: str = None,
    p_speaking_threshold: float = 0.5,
) -> list:
    """V3: Audio-visual active speaker timeline driven by Light-ASD.

    Mirrors the v2 signature but consumes per-frame Light-ASD speaking
    probabilities instead of computing lip motion + audio sync from
    landmarks. Crucially handles two cases v2 cannot:
      - Overlapping speech: emits multiple co-active SpeakerEvents at
        the same timestamp when more than one face crosses the
        p_speaking threshold.
      - Off-camera speakers: returns slot_id=-1 for transcript segments
        where no on-screen face has p_speaking above threshold.

    Post-processing (dwell collapse, shot-reverse tolerance, VAD
    bridging) reuses the same helpers as v2 so the L1 solver sees a
    timeline shape it already understands.
    """
    if not face_results or not transcript_segments:
        return []

    if not asd_scores:
        logger.info("[SpeakerV3] No ASD scores — returning empty timeline")
        return []

    # Index ASD scores by (rounded timestamp, face_idx) for O(1) lookup.
    asd_by_key: dict = {}
    for s in asd_scores:
        key = (round(float(s.timestamp), 3), int(s.face_idx))
        asd_by_key[key] = float(s.p_speaking)

    # Per-frame index of the FrameFaces objects so we can resolve face_idx
    # → identity_id when emitting SpeakerEvents.
    frame_map = {}
    for fr in face_results:
        frame_map[round(float(fr.timestamp), 3)] = fr
    frame_times = sorted(frame_map.keys())

    events: list = []

    for seg in transcript_segments:
        seg_start = seg.start if hasattr(seg, "start") else seg.get("start", 0)
        seg_end = seg.end if hasattr(seg, "end") else seg.get("end", 0)

        nearby_times = [
            t for t in frame_times
            if seg_start - window_seconds <= t <= seg_end + window_seconds
        ]
        if not nearby_times:
            events.append(SpeakerEvent(
                start=seg_start, end=seg_end, slot_id=-1, confidence=0.0,
            ))
            continue

        # Aggregate p_speaking per identity_id over the segment window.
        id_scores: dict = {}  # identity_id → max p_speaking observed
        id_avg: dict = {}     # identity_id → list of p_speaking samples
        for t in nearby_times:
            fr = frame_map[t]
            for fi, face in enumerate(fr.faces):
                iid = getattr(face, "identity_id", -1)
                if iid < 0:
                    continue
                p = asd_by_key.get((t, fi), 0.0)
                if p > id_scores.get(iid, -1.0):
                    id_scores[iid] = p
                id_avg.setdefault(iid, []).append(p)

        if not id_scores:
            events.append(SpeakerEvent(
                start=seg_start, end=seg_end, slot_id=-1, confidence=0.0,
            ))
            continue

        # Co-active speakers: any identity whose max p_speaking crosses
        # the threshold gets its own SpeakerEvent at this segment's
        # timestamps. v2 could not represent overlapping speech.
        active_ids = [
            iid for iid, p in id_scores.items() if p >= p_speaking_threshold
        ]

        # XC.3: confidence floor. When EXACTLY ONE identity crosses the
        # speaking threshold but the runner-up's p_speaking is within
        # the margin of the winner's, v3 is "unsure" — the two faces
        # are too close to commit to one. Emit slot=-1 instead of
        # picking the wrong person. Prevents the wrong-speaker-framed
        # bug on multi-speaker panels where two faces have very similar
        # lip motion / audio energy at the same moment.
        #
        # Skips the floor when ≥2 identities ALREADY crossed threshold
        # — that's overlapping speech, not a close call. v3 emits one
        # SpeakerEvent per overlapping speaker as the test fixture
        # "test_v3_handles_overlapping_speech" pins.
        if len(active_ids) == 1:
            try:
                margin = float(os.environ.get(
                    "CLIPAI_ASD_CONFIDENCE_MARGIN", "0.15"
                ))
            except ValueError:
                margin = 0.15
            sorted_scores = sorted(id_scores.values(), reverse=True)
            if (
                margin > 0
                and len(sorted_scores) >= 2
                and (sorted_scores[0] - sorted_scores[1]) < margin
            ):
                events.append(SpeakerEvent(
                    start=seg_start, end=seg_end,
                    slot_id=-1, confidence=float(sorted_scores[0]),
                ))
                continue

        if not active_ids:
            # Off-camera speaker: transcript says someone spoke but no
            # face crosses threshold. Emit slot_id=-1 so downstream
            # handling can keep the camera on the previous active slot.
            events.append(SpeakerEvent(
                start=seg_start, end=seg_end, slot_id=-1, confidence=0.0,
            ))
            continue

        # Mark speaking faces for downstream consumers.
        for t in nearby_times:
            fr = frame_map[t]
            for face in fr.faces:
                if getattr(face, "identity_id", -1) in active_ids:
                    face.is_speaking = True

        for iid in active_ids:
            samples = id_avg.get(iid, [])
            avg_p = sum(samples) / len(samples) if samples else id_scores[iid]
            events.append(SpeakerEvent(
                start=seg_start, end=seg_end,
                slot_id=int(iid),
                confidence=float(min(1.0, max(0.0, avg_p))),
            ))

    if not events:
        return []

    # Sort by start so the merge / collapse helpers see a monotonic stream.
    events.sort(key=lambda e: (e.start, e.slot_id))

    # Same-speaker run merge (per slot_id), reused from v2's merge step.
    by_slot: dict = {}
    for ev in events:
        by_slot.setdefault(ev.slot_id, []).append(ev)
    merged_all: list = []
    for sid, evs in by_slot.items():
        evs.sort(key=lambda e: e.start)
        merged: list = [evs[0]]
        for ev in evs[1:]:
            if ev.start - merged[-1].end < 1.0:
                merged[-1] = SpeakerEvent(
                    start=merged[-1].start, end=ev.end,
                    slot_id=sid,
                    confidence=max(merged[-1].confidence, ev.confidence),
                )
            else:
                merged.append(ev)
        merged_all.extend(merged)
    events = sorted(merged_all, key=lambda e: (e.start, e.slot_id))

    # Reuse the v2 post-processing chain for solver-friendliness:
    events, collapsed_runs = _collapse_short_runs(events)
    events, offscreen_count = _apply_shot_reverse_tolerance(
        events, transcript_segments, face_results, shot_cuts or [], face_registry,
    )

    if audio_path:
        try:
            vad_intervals = build_vad_presence(audio_path)
        except Exception as e:
            logger.info("[SpeakerV3] VAD bridging skipped (%s)", e)
            vad_intervals = []
        if vad_intervals:
            events, bridged_count = _apply_vad_bridging(events, vad_intervals)
            if bridged_count > 0:
                logger.info(
                    "[SpeakerV3] VAD bridged %d speaker event extensions (v3)",
                    bridged_count,
                )

    logger.info(
        "[SpeakerV3] %d events, %d unique speakers (collapsed_runs=%d, "
        "offscreen=%d, threshold=%.2f)",
        len(events),
        len(set(e.slot_id for e in events if e.slot_id >= 0)),
        collapsed_runs, offscreen_count, p_speaking_threshold,
    )
    return events


def map_speakers_to_face_slots(
    transcript_segments: list,
    face_registry,
    face_results: list,
    scenes: list = None,
    original_scene_sx: list = None,
) -> dict[str, int]:
    """Map Whisper speaker labels to face registry slot IDs.

    Uses dense face detection data directly: for each transcript segment,
    find which face slot has the highest size-weighted lip aperture.
    This gives a direct speaker→slot mapping without relying on the AI
    vision model's scene subject_x (which clusters everything to center).

    Falls back to face size if no lip data available.
    """
    if not face_registry or not transcript_segments:
        return {}

    if not face_results:
        return {}

    frame_map = {}
    for fr in face_results:
        frame_map[fr.timestamp] = fr
    frame_times = sorted(frame_map.keys())

    # For each transcript segment, vote for which slot the speaker is at
    # using size-weighted lip aperture from face detection data
    speaker_slot_votes: dict[str, dict[int, float]] = {}

    for seg in transcript_segments:
        speaker = seg.speaker if hasattr(seg, 'speaker') else seg.get('speaker', '')
        if not speaker:
            continue
        seg_start = seg.start if hasattr(seg, 'start') else seg.get('start', 0)
        seg_end = seg.end if hasattr(seg, 'end') else seg.get('end', 0)
        seg_duration = max(seg_end - seg_start, 0.1)

        # Find face frames within this speech segment
        nearby_frames = [
            frame_map[ft] for ft in frame_times
            if seg_start - 0.5 <= ft <= seg_end + 0.5
        ]

        if not nearby_frames:
            continue

        # Find which slot has highest lip aperture during this segment
        # Weight by face size — larger faces have more reliable lip data
        slot_lip_scores: dict[int, float] = {}
        slot_size_scores: dict[int, float] = {}
        for fr in nearby_frames:
            for face in fr.faces:
                # Prefer identity_id (assigned during face registry build)
                # over nearest_slot (which can misassign nearby faces)
                if hasattr(face, 'identity_id') and face.identity_id >= 0:
                    sid = face.identity_id
                else:
                    slot = face_registry.nearest_slot(face.nose_x)
                    if not slot:
                        continue
                    sid = slot.slot_id
                size = face.width * (face.height if hasattr(face, 'height') else face.width)
                size_weight = min(2.0, face.width / 8.0)
                slot_lip_scores[sid] = slot_lip_scores.get(sid, 0) + face.lip_aperture * size_weight
                slot_size_scores[sid] = slot_size_scores.get(sid, 0) + size

        # Pick the slot with highest lip activity for this speaker
        # If no lip data, fall back to largest face
        best_slot = None
        if slot_lip_scores:
            max_lip = max(slot_lip_scores.values())
            if max_lip > 0.01:
                best_slot = max(slot_lip_scores, key=slot_lip_scores.get)

        if best_slot is None and slot_size_scores:
            best_slot = max(slot_size_scores, key=slot_size_scores.get)

        if best_slot is not None:
            if speaker not in speaker_slot_votes:
                speaker_slot_votes[speaker] = {}
            # Weight the vote by segment duration — longer segments are more reliable
            speaker_slot_votes[speaker][best_slot] = (
                speaker_slot_votes[speaker].get(best_slot, 0) + seg_duration
            )

    if not speaker_slot_votes:
        return {}

    # Fix 4: confident-vote fast path.
    # Before the hotfix, the greedy assignment dropped speakers to NONE
    # when their top-voted slot was already claimed by an earlier
    # speaker — even if the "claim" was a 0.4s vote and the dropped
    # speaker had a 10.4s vote with 25× margin over runner-up. Result:
    # two confident diarized speakers got "assigned slot NONE" in the
    # Verzuz run and their airtime vanished from downstream routing.
    #
    # Policy: if a speaker's top vote is >= 3s absolute AND >= 2×
    # runner-up, take it immediately — even if another speaker also
    # wants that slot. Two speakers CAN share a slot legitimately
    # (they sit close, or the registry fragmented real identities
    # into overlapping slots). Unconfident speakers still go through
    # the greedy path afterward with remaining slots.
    result: dict = {}
    used_slots: set = set()
    confident_assignments: list = []

    for speaker, slot_scores in speaker_slot_votes.items():
        ordered = sorted(slot_scores.items(), key=lambda x: -x[1])
        if not ordered:
            continue
        top_slot, top_score = ordered[0]
        runner_score = ordered[1][1] if len(ordered) > 1 else 0.0
        if top_score >= 3.0 and top_score >= 2.0 * runner_score:
            result[speaker] = top_slot
            used_slots.add(top_slot)
            confident_assignments.append(speaker)

    # Greedy assignment for the unconfident remainder: sort by total
    # evidence, assign the best unused slot. Speakers without any
    # unused slot stay unassigned (the original NONE behavior for the
    # truly ambiguous cases only).
    remaining = [sp for sp in speaker_slot_votes if sp not in result]
    sorted_remaining = sorted(
        remaining,
        key=lambda sp: sum(speaker_slot_votes[sp].values()),
        reverse=True,
    )
    for speaker in sorted_remaining:
        slot_scores = speaker_slot_votes[speaker]
        for slot_id, _score in sorted(slot_scores.items(), key=lambda x: -x[1]):
            if slot_id not in used_slots:
                result[speaker] = slot_id
                used_slots.add(slot_id)
                break

    logger.info(
        "Speaker-to-slot mapping (dense face lip+size): %s "
        "(confident=%d, greedy=%d)",
        result, len(confident_assignments),
        len(result) - len(confident_assignments),
    )
    for sp, votes in speaker_slot_votes.items():
        top = sorted(votes.items(), key=lambda x: -x[1])[:3]
        tag = "confident" if sp in confident_assignments else "greedy"
        logger.info(
            "  %s votes: %s → assigned slot %s (%s)",
            sp, top, result.get(sp, 'NONE'), tag,
        )
    return result


def build_transcript_speaker_timeline(
    transcript_segments: list,
    speaker_slot_map: dict,
    face_registry=None,
) -> list[SpeakerEvent]:
    """Build a speaker timeline directly from Whisper transcript speaker labels.

    More reliable than lip-aperture detection because Whisper uses audio
    features (voice embeddings, spectral analysis). Each transcript segment
    has a definitive speaker label from audio-based diarization.

    Gap filling: extends events to cover sub-second gaps between consecutive
    segments, preventing fallback to unreliable lip-based detection.
    """
    if not transcript_segments or not speaker_slot_map:
        return []

    events = []
    for seg in transcript_segments:
        speaker = seg.speaker if hasattr(seg, 'speaker') else seg.get('speaker', '')
        if not speaker or speaker not in speaker_slot_map:
            continue

        seg_start = seg.start if hasattr(seg, 'start') else seg.get('start', 0)
        seg_end = seg.end if hasattr(seg, 'end') else seg.get('end', 0)
        slot_id = speaker_slot_map[speaker]

        events.append(SpeakerEvent(
            start=seg_start,
            end=seg_end,
            slot_id=slot_id,
            confidence=0.95,
        ))

    if not events:
        return []

    # Sort by start time
    events.sort(key=lambda e: e.start)

    # Gap filling: merge same-speaker gaps < 3s, split different-speaker gaps < 1.5s
    filled = [events[0]]
    for ev in events[1:]:
        prev = filled[-1]
        gap = ev.start - prev.end

        if ev.slot_id == prev.slot_id and gap < 3.0:
            # Same speaker, small gap — merge
            filled[-1] = SpeakerEvent(
                start=prev.start, end=ev.end,
                slot_id=ev.slot_id,
                confidence=max(prev.confidence, ev.confidence),
            )
        elif gap < 1.5:
            # Different speaker but tiny gap — extend previous to midpoint
            midpoint = prev.end + gap / 2
            filled[-1] = SpeakerEvent(
                start=prev.start, end=midpoint,
                slot_id=prev.slot_id,
                confidence=prev.confidence,
            )
            filled.append(SpeakerEvent(
                start=midpoint, end=ev.end,
                slot_id=ev.slot_id,
                confidence=ev.confidence,
            ))
        else:
            filled.append(ev)
    events = filled

    if events:
        logger.info(
            "Transcript-driven speaker timeline: %d events, %d unique speakers",
            len(events),
            len(set(e.slot_id for e in events if e.slot_id >= 0)),
        )
        slot_times: dict[int, float] = {}
        for ev in events:
            if ev.slot_id >= 0:
                slot_times[ev.slot_id] = slot_times.get(ev.slot_id, 0) + (ev.end - ev.start)
        total = max(sum(slot_times.values()), 1)
        for sid, secs in sorted(slot_times.items()):
            logger.info("  Slot %d speaking: %.1fs (%.0f%%)", sid, secs, secs / total * 100)

    return events
