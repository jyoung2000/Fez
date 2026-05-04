"""Parakeet TDT subprocess worker for the TACT consensus pass.

Mirrors ``whisper_worker.py``: subprocess that loads NeMo's Parakeet
model, transcribes a single audio file, writes JSON output to
``--output``, and exits. CUDA state is reclaimed by the OS when the
process terminates.

The whole module imports NeMo lazily inside ``main`` so it cannot be
loaded into a parent process by accident — the parent must
``subprocess.run`` it.

Output JSON schema matches the Whisper worker's:

    {
      "status": "ok" | "error",
      "error": str (only when status == "error"),
      "segments": [
        {
          "start": float, "end": float, "text": str,
          "avg_logprob": float | None,
          "no_speech_prob": float | None,
          "confidence": float | None,
          "words": [{"start": float, "end": float, "word": str}, ...]
        }, ...
      ],
      "info": {"language": str | None, "model": str}
    }
"""
from __future__ import annotations

import argparse
import json
import logging
import sys

logger = logging.getLogger(__name__)


def _segments_from_nemo(asr_results) -> list[dict]:
    """Translate a NeMo Parakeet ASR result (Hypothesis or list) into
    the Whisper-worker JSON schema. Best-effort across NeMo API
    versions: word timestamps come from the TDT decoder when
    available, segment-level timestamps from the ``timestamp`` dict.
    """
    out: list[dict] = []
    if asr_results is None:
        return out
    # NeMo returns either a list of Hypothesis objects (one per file)
    # or a single Hypothesis. Normalize to list.
    if not isinstance(asr_results, (list, tuple)):
        asr_results = [asr_results]
    for hyp in asr_results:
        text = getattr(hyp, "text", None) or (
            hyp.get("text") if isinstance(hyp, dict) else None
        ) or ""
        text = text.strip()
        if not text:
            continue
        # Word timestamps live under hyp.timestamp["word"] for newer
        # NeMo, hyp.word_timings for older. Tolerate both.
        words_raw = None
        ts = getattr(hyp, "timestamp", None)
        if ts is None and isinstance(hyp, dict):
            ts = hyp.get("timestamp")
        if isinstance(ts, dict):
            words_raw = ts.get("word")
        if words_raw is None:
            words_raw = getattr(hyp, "word_timings", None) or []
        words = []
        for w in words_raw or []:
            if isinstance(w, dict):
                ws = w.get("start") if "start" in w else w.get("start_time")
                we = w.get("end") if "end" in w else w.get("end_time")
                wt = w.get("word") or w.get("text")
            else:
                ws = getattr(w, "start", None)
                we = getattr(w, "end", None)
                wt = getattr(w, "word", None) or getattr(w, "text", None)
            if ws is None or we is None or not wt:
                continue
            words.append({"start": float(ws), "end": float(we), "word": str(wt)})
        # Segment bounds: prefer the segment-level timestamp dict;
        # fall back to first/last word.
        seg_start = None
        seg_end = None
        if isinstance(ts, dict):
            seg_ts = ts.get("segment") or {}
            if isinstance(seg_ts, list) and seg_ts:
                seg_ts = seg_ts[0]
            if isinstance(seg_ts, dict):
                seg_start = seg_ts.get("start") or seg_ts.get("start_time")
                seg_end = seg_ts.get("end") or seg_ts.get("end_time")
        if seg_start is None and words:
            seg_start = words[0]["start"]
        if seg_end is None and words:
            seg_end = words[-1]["end"]
        if seg_start is None or seg_end is None:
            continue
        out.append({
            "start": float(seg_start),
            "end": float(seg_end),
            "text": text,
            "words": words,
            "avg_logprob": None,
            "no_speech_prob": None,
            "confidence": None,
        })
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Parakeet transcription worker")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--model", default="nvidia/parakeet-tdt-0.6b-v3",
        help="HuggingFace / NeMo model id (default: parakeet-tdt-0.6b-v3)",
    )
    parser.add_argument("--language", default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [parakeet_worker] %(message)s",
        stream=sys.stderr,
    )

    try:
        # Import NeMo only inside the worker process. This guarantees
        # the ~1 GB import never lands in the parent.
        from nemo.collections.asr.models import ASRModel  # type: ignore

        logger.info("loading model: %s", args.model)
        asr = ASRModel.from_pretrained(args.model)
        try:
            asr.eval()
        except Exception:
            pass
        logger.info("transcribing: %s", args.audio)

        # Tolerate API-version drift: newer NeMo accepts a list, older
        # accepts a single path. Always request word timestamps.
        kwargs = {"timestamps": True}
        try:
            results = asr.transcribe([args.audio], **kwargs)
        except TypeError:
            # Older NeMo: positional single-path API.
            try:
                results = asr.transcribe(audio=[args.audio], **kwargs)
            except TypeError:
                results = asr.transcribe([args.audio])

        segments = _segments_from_nemo(results)
        out = {
            "status": "ok",
            "segments": segments,
            "info": {
                "language": args.language or "",
                "model": args.model,
            },
        }
        with open(args.output, "w") as f:
            json.dump(out, f)
        logger.info("done: %d segments", len(segments))
        return 0
    except Exception as e:
        logger.exception("parakeet worker failed")
        try:
            with open(args.output, "w") as f:
                json.dump({"status": "error", "error": str(e)}, f)
        except Exception:
            pass
        return 1


if __name__ == "__main__":
    sys.exit(main())
