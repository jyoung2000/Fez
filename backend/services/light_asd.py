"""Light-ASD audio-visual active speaker detection.

Runs the Light-ASD ONNX model
(https://github.com/Junhua-Liao/Light-ASD) to produce a per-face
speaking probability at each sampled frame. The model takes (a) a
short audio MFCC window and (b) a sequence of mouth-region crops
from a single tracked face, and outputs a per-timestep probability
that *this* face is the speaker.

CPU-only via onnxruntime CPUExecutionProvider. Must NOT compete with
Whisper / Ollama on the GPU — the Jalon harness reserves the GTX 1650
(4 GB VRAM) for those workloads.

Routed via the CLIPAI_ASD_BACKEND=light_asd flag from
``backend/services/pipeline.py``. Default off; the legacy v2 lip-
aperture heuristic stays as the production path until validation
on Jalon's actual content corpus confirms parity or improvement.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# Audio: 13-coef MFCC at 100 Hz (10 ms hop), window = 40 frames (~0.4 s).
# Visual: grayscale mouth crop resized to 112x112, window = 5 frames (~0.2s @ 25fps).
LIGHT_ASD_URL = (
    "https://github.com/Junhua-Liao/Light-ASD/releases/download/v1.0/light_asd.onnx"
)


@dataclass
class ASDResult:
    """Per-face, per-timestamp speaking probability emitted by Light-ASD."""
    timestamp: float
    face_idx: int        # index into the FrameFaces.faces list at this timestamp
    p_speaking: float    # 0.0–1.0


_ASD_SESSION = None
# Once we've tried (and failed) to load the model, cache that fact so
# subsequent calls don't re-attempt the (~50 MB) download or re-log the
# same warning. The heuristic v2 fallback handles all callers cleanly.
_ASD_LOAD_FAILED = False


def _get_session():
    """Lazy-load the Light-ASD ONNX session.

    Returns None when onnxruntime isn't installed or the model can't
    be downloaded / loaded so callers can fall back to the v2 heuristic.
    Cached as a module-level singleton; failures are cached too so we
    don't retry the download on every call.
    """
    global _ASD_SESSION, _ASD_LOAD_FAILED
    if _ASD_SESSION is not None:
        return _ASD_SESSION
    if _ASD_LOAD_FAILED:
        return None
    try:
        import onnxruntime as ort
    except ImportError:
        logger.warning("onnxruntime not installed — Light-ASD unavailable")
        _ASD_LOAD_FAILED = True
        return None

    model_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "light_asd.onnx")
    # Allow operators / Docker images to pre-bundle the weights at a
    # custom path (e.g. baked into the image build). When set, this
    # path is preferred over the model_dir lookup so air-gapped
    # deployments don't have to hit the upstream release URL.
    bundled_path = os.environ.get("CLIPAI_LIGHT_ASD_MODEL_PATH", "").strip()
    if bundled_path and os.path.exists(bundled_path):
        model_path = bundled_path
    if not os.path.exists(model_path):
        try:
            import urllib.request
            logger.info("Downloading Light-ASD ONNX model...")
            urllib.request.urlretrieve(LIGHT_ASD_URL, model_path)
        except Exception as e:
            logger.warning(
                "Failed to download Light-ASD (%s) — falling back to "
                "heuristic v2 ASD for the rest of this run",
                e,
            )
            _ASD_LOAD_FAILED = True
            return None

    try:
        _ASD_SESSION = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )
        logger.info("Loaded Light-ASD ONNX session (CPU)")
        return _ASD_SESSION
    except Exception as e:
        logger.warning(
            "Light-ASD session init failed (%s) — falling back to "
            "heuristic v2 ASD for the rest of this run",
            e,
        )
        _ASD_LOAD_FAILED = True
        return None


def score_faces_for_clip(
    video_path: str,
    face_results: list,      # list[FrameFaces] with bbox info
    audio_wav_path: str,
    window_frames: int = 5,
) -> list:
    """Run Light-ASD over every face in every frame of face_results.

    For each face track (same identity_id across frames), builds a
    sliding window of mouth crops + MFCC audio, runs the ONNX model,
    and emits an ASDResult per (timestamp, face_idx).

    Returns an empty list on any failure (missing onnxruntime, missing
    librosa, no faces with identity_id, etc.) so the caller can fall
    back to the v2 heuristic timeline.
    """
    sess = _get_session()
    if sess is None:
        return []
    if not face_results:
        return []

    try:
        import cv2
        import numpy as np
    except ImportError as e:
        logger.warning("Light-ASD: cv2 / numpy not available: %s", e)
        return []

    # Extract MFCC from audio_wav_path. librosa is heavyweight but only
    # imported when the flag is on, so the default code path stays light.
    try:
        import librosa
        audio, sr = librosa.load(audio_wav_path, sr=16000, mono=True)
        mfcc = librosa.feature.mfcc(
            y=audio, sr=sr, n_mfcc=13, hop_length=160, n_fft=512,
        )
        # mfcc shape: (13, T) where T = len(audio)/160
    except Exception as e:
        logger.warning("MFCC extraction failed: %s — Light-ASD aborting", e)
        return []

    results: list[ASDResult] = []

    # Group face detections by identity_id (face track). Faces without
    # identity_id stay un-tracked; the v2 fallback handles them.
    tracks: dict = {}  # id → [(timestamp, frame_path, face_idx, face_info), ...]
    for fr in face_results:
        for fi, face in enumerate(fr.faces):
            if getattr(face, "identity_id", -1) < 0:
                continue
            tracks.setdefault(face.identity_id, []).append(
                (fr.timestamp, fr.frame_path, fi, face)
            )

    if not tracks:
        return []

    input_names = [inp.name for inp in sess.get_inputs()]

    for track_id, entries in tracks.items():
        entries.sort(key=lambda x: x[0])
        # Slide window_frames window; run once per center-frame
        for i in range(len(entries)):
            lo = max(0, i - window_frames // 2)
            hi = min(len(entries), lo + window_frames)
            if hi - lo < window_frames:
                continue

            mouth_crops = []
            valid = True
            for j in range(lo, hi):
                ts, fp, fidx, face = entries[j]
                img = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    valid = False
                    break
                h, w = img.shape
                # Crop mouth region — bottom half of face bbox.
                cx = face.x_center * w / 100
                fw = face.width * w / 100
                fh = face.height * h / 100
                my = (face.y_center + face.height * 0.15) * h / 100
                mh = fh * 0.5
                mx1 = int(max(0, cx - fw / 2))
                mx2 = int(min(w, cx + fw / 2))
                my1 = int(max(0, my - mh / 2))
                my2 = int(min(h, my + mh / 2))
                if mx2 - mx1 < 8 or my2 - my1 < 8:
                    valid = False
                    break
                crop = cv2.resize(img[my1:my2, mx1:mx2], (112, 112))
                mouth_crops.append(crop.astype(np.float32) / 255.0)
            if not valid or len(mouth_crops) < window_frames:
                continue

            # (1, 1, T, 112, 112) — batch, channels, time, H, W
            visual = np.stack(mouth_crops)[None, None, ...]

            # Audio window of 40 MFCC frames (~0.4s) centered on the i-th frame.
            ts_center = entries[i][0]
            mfcc_center = int(ts_center * 100)  # 100 Hz
            mfcc_lo = max(0, mfcc_center - 20)
            mfcc_hi = min(mfcc.shape[1], mfcc_lo + 40)
            if mfcc_hi - mfcc_lo < 40:
                continue
            audio_win = mfcc[:, mfcc_lo:mfcc_hi].T[None, ...].astype(np.float32)

            try:
                outputs = sess.run(
                    None,
                    {
                        input_names[0]: visual,
                        input_names[1]: audio_win,
                    },
                )
                # Last logit = sigmoid score for "speaking now."
                p_speak = float(outputs[0].flatten()[-1])
                p_speak = max(0.0, min(1.0, p_speak))
            except Exception as e:
                logger.debug("Light-ASD forward failed at t=%.2f: %s", ts_center, e)
                continue

            ts_i, _, fidx_i, _ = entries[i]
            results.append(ASDResult(
                timestamp=ts_i, face_idx=fidx_i, p_speaking=p_speak,
            ))

    logger.info(
        "[Light-ASD] scored %d (timestamp, face) pairs across %d tracks",
        len(results), len(tracks),
    )
    return results
