"""TACT Phase 2 Rung 5 — non-speech event classifier.

Wraps PANNs CNN14 (pretrained on AudioSet's 527-class taxonomy) to
classify a short audio interval as music / laughter / applause /
silence / noise / speech / other. The TACT use case: when Rungs 1-4
fail to transcribe a gap, this rung asks "is the gap actually music
or applause?" and emits a positive event tag rather than letting
Rung 6 mark it [unintelligible].

PANNs is pure-torch (no TF dependency that would conflict with the
torch CUDA stack already in the project). The CNN14 checkpoint is
~80 MB and downloads on first use; the production deployment should
vendor it into the Docker image.

Module is import-cheap. The PANNs / torch import + model load is
lazy (inside ``classify_interval``), guarded by a Lock so concurrent
jobs don't double-load.
"""
from __future__ import annotations

import logging
import os
import subprocess
import tempfile
import threading
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# Map AudioSet's 527 raw classes to a simplified 7-class taxonomy.
# Anything not in this dict maps to "other" — falls through to Rung 6.
AUDIOSET_TO_SIMPLIFIED: dict[str, str] = {
    # Music family.
    "Music": "music",
    "Musical instrument": "music",
    "Singing": "music",
    "Background music": "music",
    "Theme music": "music",
    "Soundtrack music": "music",
    "Lullaby": "music",
    # Speech family.
    "Speech": "speech",
    "Conversation": "speech",
    "Narration, monologue": "speech",
    "Male speech, man speaking": "speech",
    "Female speech, woman speaking": "speech",
    "Child speech, kid speaking": "speech",
    # Laughter.
    "Laughter": "laughter",
    "Giggle": "laughter",
    "Chuckle, chortle": "laughter",
    "Snicker": "laughter",
    # Applause / cheering.
    "Applause": "applause",
    "Clapping": "applause",
    "Cheering": "applause",
    "Crowd": "applause",
    # Silence.
    "Silence": "silence",
    # Noise.
    "Noise": "noise",
    "White noise": "noise",
    "Pink noise": "noise",
    "Static": "noise",
    "Hum": "noise",
    "Buzz": "noise",
    "Hiss": "noise",
}


@dataclass
class EventClassification:
    label: str        # "music" | "laughter" | "applause" | "silence"
                      # | "noise" | "speech" | "other"
    confidence: float  # softmax probability of the winning AudioSet class
    raw_class: str    # The AudioSet class name that won (for diagnostics)


_VALID_LABELS = (
    "music", "laughter", "applause", "silence", "noise", "speech", "other",
)

# Module-level model cache. PANNs takes ~3 s to instantiate.
_panns_model = None
_panns_lock = threading.Lock()


def _slice_to_32k_mono(
    audio_path: str, start_sec: float, end_sec: float, out_path: str,
) -> bool:
    """ffmpeg-slice the interval to 32 kHz mono PCM. PANNs CNN14 is
    trained on 32 kHz; resampling at the slice step avoids depending
    on whatever the parent loudnorm chain produced."""
    duration = max(0.0, end_sec - start_sec)
    if duration <= 0:
        return False
    try:
        cmd = [
            "ffmpeg", "-y", "-nostdin", "-loglevel", "error",
            "-ss", f"{start_sec:.3f}",
            "-i", audio_path,
            "-t", f"{duration:.3f}",
            "-ac", "1", "-ar", "32000",
            "-c:a", "pcm_s16le",
            out_path,
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=30)
        if result.returncode != 0:
            return False
        if not os.path.exists(out_path) or os.path.getsize(out_path) < 200:
            return False
        return True
    except Exception:
        return False


def _get_model():
    """Lazy-load PANNs CNN14. Returns None when the dep is missing or
    the checkpoint can't be loaded. Cached for the process lifetime."""
    global _panns_model
    with _panns_lock:
        if _panns_model is not None:
            return _panns_model
        try:
            from panns_inference import AudioTagging  # type: ignore
        except Exception as e:
            logger.info(
                "non_speech_events: panns_inference unavailable (%s) — "
                "Rung 5 will decline", e,
            )
            return None
        try:
            # checkpoint_path=None → use the bundled / cached default.
            _panns_model = AudioTagging(checkpoint_path=None, device="cpu")
            logger.info("non_speech_events: PANNs CNN14 loaded")
            return _panns_model
        except Exception as e:
            logger.warning(
                "non_speech_events: PANNs load failed (%s)", e,
            )
            return None


def classify_interval(
    audio_path: str, start_sec: float, end_sec: float,
) -> EventClassification:
    """Classify a single audio interval into the simplified taxonomy.

    Returns ``EventClassification(label="other", confidence=0.0,
    raw_class="")`` on any failure path so the caller (Rung 5
    wrapper) can fall through cleanly.
    """
    if end_sec <= start_sec:
        return EventClassification(label="other", confidence=0.0, raw_class="")

    model = _get_model()
    if model is None:
        return EventClassification(label="other", confidence=0.0, raw_class="")

    with tempfile.NamedTemporaryFile(
        suffix=".wav", delete=False, dir=tempfile.gettempdir(),
    ) as tmp:
        slice_path = tmp.name
    try:
        if not _slice_to_32k_mono(audio_path, start_sec, end_sec, slice_path):
            return EventClassification(label="other", confidence=0.0, raw_class="")
        try:
            import numpy as np
            import soundfile as sf
        except Exception:
            return EventClassification(label="other", confidence=0.0, raw_class="")
        try:
            audio, sr = sf.read(slice_path, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            # PANNs expects shape (1, num_samples).
            audio = np.expand_dims(audio.astype(np.float32), axis=0)
            clipwise_output, _ = model.inference(audio)
            # ``clipwise_output`` shape (1, 527). Pull labels from the
            # PANNs metadata module.
            try:
                from panns_inference import labels as _labels  # type: ignore
                class_names = list(_labels)
            except Exception:
                class_names = []

            probs = clipwise_output[0]
            top_idx = int(np.argmax(probs))
            raw_class = (
                class_names[top_idx] if top_idx < len(class_names) else f"class_{top_idx}"
            )
            label = AUDIOSET_TO_SIMPLIFIED.get(raw_class, "other")
            return EventClassification(
                label=label,
                confidence=float(probs[top_idx]),
                raw_class=raw_class,
            )
        except Exception as e:
            logger.info("non_speech_events: inference failed (%s)", e)
            return EventClassification(label="other", confidence=0.0, raw_class="")
    finally:
        try:
            os.unlink(slice_path)
        except OSError:
            pass
