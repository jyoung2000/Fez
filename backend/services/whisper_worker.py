"""Whisper transcription subprocess worker.

Runs faster-whisper in an isolated process so that CTranslate2's CUDA context
is fully released when transcription completes. Without subprocess isolation,
CTranslate2 holds ~1.6GB of VRAM indefinitely — leaving insufficient memory
for Ollama models on a 4GB GTX 1650.

The process writes transcription results to --output as JSON and exits.
All CUDA memory is reclaimed by the OS when the process terminates.
"""

import argparse
import gc
import json
import logging
import os
import subprocess
import sys
import tempfile

logger = logging.getLogger(__name__)


def _filter_segments(result_segments: list[dict], initial_prompt: str = "", task: str = "transcribe") -> list[dict]:
    """Remove hallucinated segments: ghosts, loops, backward jumps, duplicates.

    When task='translate', applies looser thresholds because English translations
    of non-English audio produce shorter text for the same audio duration.
    """
    is_translate = (task == "translate")

    # Build set of prompt fragments to detect echoing
    _prompt_fragments = set()
    if initial_prompt:
        for phrase in initial_prompt.replace(". ", ".\n").split("\n"):
            cleaned_phrase = phrase.strip().lower().rstrip(".")
            if len(cleaned_phrase) > 15:
                _prompt_fragments.add(cleaned_phrase)

    cleaned = []
    prev_text = ""
    for seg in result_segments:
        text = seg.get("text", "").strip()
        if not text:
            continue

        # Check for initial_prompt echo (Whisper hallucinates the prompt during silence)
        text_lower = text.lower().rstrip(".")
        is_prompt_echo = False
        for frag in _prompt_fragments:
            if frag in text_lower or text_lower in frag:
                is_prompt_echo = True
                break
        if is_prompt_echo:
            logger.warning("Filter: prompt echo at %.1fs: %s", seg["start"], text[:80])
            continue

        # Skip backward-jumping timestamps
        if cleaned and seg["start"] < cleaned[-1]["start"]:
            logger.warning("Filter: backward jump at %.1fs, skipping: %s", seg["start"], text[:60])
            continue

        # Skip runaway segments (>1500 chars)
        if len(text) > 1500:
            logger.warning("Filter: runaway at %.1fs (%d chars), skipping", seg["start"], len(text))
            continue

        # Skip ghost segments: long duration + short text + high no_speech
        # Translate: English output is shorter than source audio, and no_speech_prob
        # tends higher because Whisper is doing more internal processing.
        duration = seg["end"] - seg["start"]
        no_speech = seg.get("no_speech_prob", 0.0)
        ghost_ns_dur = 45 if is_translate else 30
        ghost_ns_textlen = 15 if is_translate else 30
        ghost_ns_prob = 0.7 if is_translate else 0.5
        if duration > ghost_ns_dur and len(text) < ghost_ns_textlen and no_speech > ghost_ns_prob:
            logger.warning("Filter: ghost (no_speech) at %.1fs (%.0fs, ns=%.2f): %s", seg["start"], duration, no_speech, text[:60])
            continue

        # Ghost check: text-to-duration ratio
        # Translate: English translations of Japanese are structurally shorter —
        # a 30s Japanese utterance may be just "Is that so?" (11 chars = 0.37 c/s).
        chars_per_sec = len(text) / max(duration, 0.1)
        ghost_ratio_threshold = 0.15 if is_translate else 1.0
        ghost_min_duration = 45 if is_translate else 15
        if duration > ghost_min_duration and chars_per_sec < ghost_ratio_threshold:
            logger.warning("Filter: ghost (ratio) at %.1fs (%.0fs, %.2f c/s): %s", seg["start"], duration, chars_per_sec, text[:60])
            continue

        # Mega-segments need proportional text
        # Translate: even short phrases like "Yes." are valid for long segments
        mega_min_text = 15 if is_translate else 200
        mega_min_duration = 240 if is_translate else 120
        if duration > mega_min_duration and len(text) < mega_min_text:
            logger.warning("Filter: mega-ghost at %.1fs (%.0fs, %d chars): %s", seg["start"], duration, len(text), text[:60])
            continue

        # Skip exact duplicate of previous segment
        if prev_text and text == prev_text and duration < 5.0:
            logger.warning("Filter: exact duplicate at %.1fs: %s", seg["start"], text[:60])
            continue

        # Near-duplicate of any recent segment (sliding window)
        # Translate: short backchannel responses ("Yes.", "Really?", "Is that so?")
        # repeat legitimately in Japanese conversation — only filter if both text
        # AND timestamp are nearly identical.
        if len(cleaned) >= 2:
            if is_translate:
                last = cleaned[-1]
                if (text.lower() == last.get("text", "").strip().lower()
                    and abs(seg["start"] - last["end"]) < 2.0
                    and duration < 3.0):
                    logger.warning("Filter: near-dup (translate) at %.1fs: %s", seg["start"], text[:60])
                    continue
            else:
                recent_texts = [c.get("text", "").strip().lower() for c in cleaned[-5:]]
                if text.lower() in recent_texts:
                    logger.warning("Filter: near-dup (window) at %.1fs: %s", seg["start"], text[:60])
                    continue

        # Repeated n-gram detection
        words = text.lower().split()
        if len(words) >= 9:
            trigrams = [tuple(words[i:i+3]) for i in range(len(words) - 2)]
            counts = {}
            for tg in trigrams:
                counts[tg] = counts.get(tg, 0) + 1
            max_rep = max(counts.values()) if counts else 0
            if max_rep >= 3 and len(trigrams) > 0 and max_rep / len(trigrams) > 0.4:
                logger.warning("Filter: looping at %.1fs: %s", seg["start"], text[:60])
                continue

        cleaned.append(seg)
        prev_text = text

    removed = len(result_segments) - len(cleaned)
    if removed > 0:
        logger.info("Hallucination filter removed %d/%d segments", removed, len(result_segments))
    return cleaned


def main():
    parser = argparse.ArgumentParser(description="Whisper transcription worker")
    parser.add_argument("--audio", required=False, default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--preflight", action="store_true", default=False,
                        help="Only load model and verify CUDA, then exit (no transcription)")
    parser.add_argument("--model", default="small")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--compute-type", default="float16")
    parser.add_argument("--language", default=None)
    parser.add_argument("--task", default="transcribe")
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--vad-filter", action="store_true", default=False)
    parser.add_argument("--word-timestamps", action="store_true", default=False)
    parser.add_argument("--initial-prompt", default=None)
    # Quality parameters (must match in-process transcription path)
    parser.add_argument("--best-of", type=int, default=3)
    parser.add_argument("--no-speech-threshold", type=float, default=0.8)
    parser.add_argument("--log-prob-threshold", type=float, default=-1.5)
    parser.add_argument("--compression-ratio-threshold", type=float, default=2.4)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=3)
    parser.add_argument("--condition-on-previous", action="store_true", default=True)
    parser.add_argument("--no-condition-on-previous", dest="condition_on_previous", action="store_false")
    parser.add_argument("--prompt-reset-on-temperature", type=float, default=0.5)
    parser.add_argument("--cjk", action="store_true", default=False)
    parser.add_argument("--vad-min-silence-ms", type=int, default=300)
    parser.add_argument("--vad-speech-pad-ms", type=int, default=600)
    parser.add_argument("--vad-onset", type=float, default=0.2)
    parser.add_argument("--vad-min-speech-ms", type=int, default=100)
    parser.add_argument("--audio-duration", type=float, default=0,
                        help="Total audio duration in seconds (for progress reporting)")
    # ── TACT Phase 3: disjoint-offset multi-pass ──
    # When non-zero, the worker trims the input audio to [offset, end]
    # via ffmpeg before transcription, then post-shifts every segment
    # and word timestamp by +offset before emitting JSON. This phase-
    # shifts Whisper's internal 30-s chunk grid relative to the primary
    # pass, so words at primary-pass chunk boundaries land in mid-chunk
    # in the offset pass.
    parser.add_argument(
        "--input-offset-sec", type=float, default=0.0,
        help="Trim audio to [offset, end] before Whisper; "
             "post-shift output timestamps by +offset.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    try:
        import time as _time
        from faster_whisper import WhisperModel

        model_kwargs = {
            "device": args.device,
            "compute_type": args.compute_type,
        }
        if args.device == "cuda":
            model_kwargs["device_index"] = args.device_index

        logger.info(
            "Loading Whisper model=%s device=%s device_index=%d compute_type=%s",
            args.model, args.device, args.device_index, args.compute_type,
        )

        # Emit progress so the frontend knows model loading has started
        _device_label = f"GPU ({args.compute_type})" if args.device == "cuda" else f"CPU ({args.compute_type})"
        print(f'PROGRESS:{json.dumps({"segments": 0, "pct": 0, "lang": "", "position_sec": 0, "eta_sec": 0, "last_text": "", "phase": "model_loading", "message": f"Loading Whisper model ({args.model}) into {_device_label}..."})}', file=sys.stderr, flush=True)

        _load_t0 = _time.monotonic()
        try:
            model = WhisperModel(args.model, **model_kwargs)
        except RuntimeError as cuda_err:
            # CUDA initialization can fail with "unknown error" when the GPU
            # driver is broken, VRAM is exhausted, or Docker GPU passthrough
            # isn't working properly. Fall back to CPU automatically.
            err_str = str(cuda_err).lower()
            if args.device == "cuda" and ("cuda" in err_str or "unknown error" in err_str):
                # Log detailed CUDA diagnostics before falling back
                _diag = []
                try:
                    import ctranslate2
                    _diag.append(f"ctranslate2_cuda_devices={ctranslate2.get_cuda_device_count()}")
                except Exception as _de:
                    _diag.append(f"ctranslate2_cuda_check_failed={_de}")
                try:
                    import glob as _gl
                    _nv = _gl.glob("/dev/nvidia[0-9]*")
                    _diag.append(f"nvidia_devices={_nv}")
                except Exception:
                    pass
                _diag.append(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}")
                logger.warning(
                    "CUDA failed (%s) — falling back to CPU (int8). "
                    "Diagnostics: %s. "
                    "This usually means another process (e.g. Ollama) is holding the GPU, "
                    "or GPU passthrough is broken in this container restart.",
                    cuda_err, ", ".join(_diag),
                )
                model_kwargs["device"] = "cpu"
                model_kwargs["compute_type"] = "int8"
                model_kwargs.pop("device_index", None)
                model = WhisperModel(args.model, **model_kwargs)
            else:
                raise
        _load_ms = int((_time.monotonic() - _load_t0) * 1000)
        _actual_device = model_kwargs.get("device", args.device)
        logger.info("Whisper model loaded in %dms (device=%s)", _load_ms, _actual_device)
        print(f'PROGRESS:{json.dumps({"segments": 0, "pct": 0, "lang": "", "position_sec": 0, "eta_sec": 0, "last_text": "", "phase": "model_loaded", "message": f"Whisper model loaded ({_load_ms}ms) — preprocessing audio..."})}', file=sys.stderr, flush=True)

        # ── Preflight mode: verify model loads then exit ──
        if args.preflight:
            actual_device = model_kwargs.get("device", args.device)
            logger.info("Preflight check passed: model=%s requested=%s actual=%s load_time=%dms", args.model, args.device, actual_device, _load_ms)
            with open(args.output, "w") as f:
                json.dump({"status": "ok", "load_time_ms": _load_ms, "actual_device": actual_device}, f)
            del model
            gc.collect()
            sys.exit(0)

        if not args.audio:
            logger.error("No audio file specified (--audio required for transcription)")
            with open(args.output, "w") as f:
                json.dump({"status": "error", "error": "No audio file specified"}, f)
            sys.exit(1)

        logger.info("Transcribing: %s", args.audio)

        # ── Audio preprocessing (matches in-process path in transcription.py) ──
        # Normalize volume so Whisper gets consistent input levels.
        # Whisper was trained on -20 LUFS audio; quiet/loud recordings degrade accuracy.
        # For translate tasks (e.g. Japanese→English), use gentler noise gate and
        # compression to preserve quiet backchannel responses, whispered asides,
        # and expressive speech that carries meaning in the source language.
        preprocessed_path = args.audio
        is_translate = args.task == "translate"
        # Gentler settings for translate: lower noise gate (-55dB vs -45dB), softer
        # compression (2:1 vs 4:1), more makeup gain to lift quiet speech.
        if is_translate:
            _af_base = (
                "highpass=f=50,"
                "acompressor=threshold=-35dB:ratio=2:attack=10:release=200:makeup=8dB,"
                "agate=threshold=-55dB:attack=10:release=100"
            )
        else:
            _af_base = (
                "highpass=f=50,"
                "acompressor=threshold=-30dB:ratio=4:attack=5:release=100:makeup=6dB,"
                "agate=threshold=-45dB:attack=5:release=50"
            )
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                preprocessed_path = tmp.name

            # Two-pass loudnorm for precise normalization
            # Pass 1: Measure loudness statistics
            measure_cmd = [
                "ffmpeg", "-y", "-i", args.audio,
                "-af", f"{_af_base},loudnorm=I=-20:TP=-1.5:LRA=7:print_format=json",
                "-f", "null", "-",
            ]
            measure_result = subprocess.run(measure_cmd, capture_output=True, text=True, timeout=120)

            loudnorm_stats = None
            if measure_result.returncode == 0 and measure_result.stderr:
                stderr_text = measure_result.stderr
                json_start = stderr_text.rfind('{')
                json_end = stderr_text.rfind('}')
                if json_start >= 0 and json_end > json_start:
                    try:
                        loudnorm_stats = json.loads(stderr_text[json_start:json_end + 1])
                    except (ValueError, KeyError):
                        pass

            if loudnorm_stats:
                # Pass 2: Apply measured corrections (precise normalization)
                normalize_filter = (
                    f"{_af_base},"
                    f"loudnorm=I=-20:TP=-1.5:LRA=7:linear=true"
                    f":measured_I={loudnorm_stats.get('input_i', '-24.0')}"
                    f":measured_TP={loudnorm_stats.get('input_tp', '-2.0')}"
                    f":measured_LRA={loudnorm_stats.get('input_lra', '7.0')}"
                    f":measured_thresh={loudnorm_stats.get('input_thresh', '-34.0')}"
                    f":offset={loudnorm_stats.get('target_offset', '0.0')}"
                )
                cmd = [
                    "ffmpeg", "-y", "-i", args.audio,
                    "-af", normalize_filter,
                    "-ar", "16000", "-ac", "1",
                    preprocessed_path,
                ]
                result = subprocess.run(cmd, capture_output=True, timeout=120)
                if result.returncode != 0:
                    logger.warning("Two-pass loudnorm failed, falling back to single-pass")
                    cmd_fallback = [
                        "ffmpeg", "-y", "-i", args.audio,
                        "-af", f"{_af_base},loudnorm=I=-20:TP=-1.5:LRA=7",
                        "-ar", "16000", "-ac", "1",
                        preprocessed_path,
                    ]
                    result = subprocess.run(cmd_fallback, capture_output=True, timeout=120)
                    if result.returncode != 0:
                        preprocessed_path = args.audio
                else:
                    logger.info("Audio preprocessed: two-pass loudnorm to -20 LUFS, 16kHz mono")
            else:
                # Fallback: single-pass if measurement failed
                cmd = [
                    "ffmpeg", "-y", "-i", args.audio,
                    "-af", f"{_af_base},loudnorm=I=-20:TP=-1.5:LRA=7",
                    "-ar", "16000", "-ac", "1",
                    preprocessed_path,
                ]
                result = subprocess.run(cmd, capture_output=True, timeout=120)
                if result.returncode != 0:
                    preprocessed_path = args.audio
                else:
                    logger.info("Audio preprocessed: single-pass loudnorm (measurement failed)")
        except Exception as e:
            logger.warning("Audio preprocessing skipped: %s", e)
            preprocessed_path = args.audio

        # All quality parameters matching the in-process path (transcription.py ~line 1076)
        if args.cjk and args.task != "translate":
            compression_ratio = 3.0  # CJK source, CJK output
        elif args.task == "translate":
            compression_ratio = 2.8  # Translated output has moderate compression
        else:
            compression_ratio = args.compression_ratio_threshold

        transcribe_kwargs = {
            "beam_size": args.beam_size,
            "best_of": args.best_of if args.beam_size > 1 else 1,
            "word_timestamps": args.word_timestamps,
            "vad_filter": args.vad_filter,
            "task": args.task,
            "condition_on_previous_text": args.condition_on_previous,
            "no_speech_threshold": args.no_speech_threshold,
            "log_prob_threshold": args.log_prob_threshold,
            "compression_ratio_threshold": compression_ratio,
            "repetition_penalty": args.repetition_penalty,
            "no_repeat_ngram_size": args.no_repeat_ngram_size,
            "temperature": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            "prompt_reset_on_temperature": args.prompt_reset_on_temperature,
        }

        if args.vad_filter:
            vad_params = {
                "min_silence_duration_ms": args.vad_min_silence_ms,
                "speech_pad_ms": args.vad_speech_pad_ms,
                "onset": args.vad_onset,
                "min_speech_duration_ms": args.vad_min_speech_ms,
            }
            # For translation tasks, ensure minimum speech padding
            # (parent process already sets good defaults — only enforce floor)
            # CJK translate gets wider padding and lower onset for soft speech
            if args.task == "translate":
                if args.cjk:
                    vad_params["speech_pad_ms"] = max(vad_params["speech_pad_ms"], 800)
                    vad_params["onset"] = min(vad_params["onset"], 0.08)
                else:
                    vad_params["speech_pad_ms"] = max(vad_params["speech_pad_ms"], 700)
            transcribe_kwargs["vad_parameters"] = vad_params

        if args.language:
            transcribe_kwargs["language"] = args.language

        if args.initial_prompt:
            transcribe_kwargs["initial_prompt"] = args.initial_prompt

        logger.info(
            "Transcribe params: beam=%d, best_of=%d, repetition_penalty=%.1f, "
            "compression_ratio=%.1f, no_speech=%.1f, cjk=%s",
            args.beam_size, transcribe_kwargs["best_of"],
            args.repetition_penalty, compression_ratio,
            args.no_speech_threshold, args.cjk,
        )

        # Emit progress: audio preprocessing done, starting VAD + transcription
        _vad_label = "with VAD filtering" if args.vad_filter else "without VAD"
        print(f'PROGRESS:{json.dumps({"segments": 0, "pct": 0, "lang": "", "position_sec": 0, "eta_sec": 0, "last_text": "", "phase": "vad_start", "message": f"Audio preprocessed — starting transcription {_vad_label}..."})}', file=sys.stderr, flush=True)

        # ── TACT Phase 3 offset trim ──
        # Trim preprocessed audio to [offset_sec, end] for the disjoint-
        # offset second pass. Done after preprocessing so the loudnorm /
        # noise-gate chain runs on the full file (consistent levels)
        # before slicing.
        offset_trim_path = None
        offset_sec = float(args.input_offset_sec or 0.0)
        if offset_sec > 0.0:
            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as _otmp:
                    offset_trim_path = _otmp.name
                _trim_cmd = [
                    "ffmpeg", "-y", "-nostdin", "-loglevel", "error",
                    "-ss", f"{offset_sec:.3f}",
                    "-i", preprocessed_path,
                    "-c:a", "pcm_s16le",
                    offset_trim_path,
                ]
                _trim_res = subprocess.run(_trim_cmd, capture_output=True, timeout=120)
                if _trim_res.returncode != 0 or not os.path.exists(offset_trim_path):
                    raise RuntimeError(
                        f"ffmpeg offset trim failed (rc={_trim_res.returncode})"
                    )
                logger.info(
                    "TACT: trimmed audio for offset pass at %.3fs", offset_sec,
                )
                preprocessed_path = offset_trim_path
            except Exception as _e:
                logger.warning("TACT offset trim failed: %s — continuing with offset=0", _e)
                offset_sec = 0.0
                if offset_trim_path:
                    try:
                        os.unlink(offset_trim_path)
                    except OSError:
                        pass
                    offset_trim_path = None

        segments_gen, info = model.transcribe(preprocessed_path, **transcribe_kwargs)

        # Materialize segments with real-time progress reporting to stderr.
        # The parent process reads PROGRESS:{json} lines to update the UI.
        result_segments = []
        _last_progress_pos = 0.0
        _audio_dur = args.audio_duration or 0
        _detected_lang = info.language or ""

        for seg in segments_gen:
            seg_data = {
                "id": seg.id,
                "start": seg.start,
                "end": seg.end,
                "text": seg.text,
                "avg_logprob": seg.avg_logprob,
                "no_speech_prob": seg.no_speech_prob,
            }
            if seg.words:
                seg_data["words"] = [
                    {"start": w.start, "end": w.end, "word": w.word, "probability": w.probability}
                    for w in seg.words
                ]
            result_segments.append(seg_data)

            # Emit progress every ~5 seconds of audio processed
            if seg.end - _last_progress_pos >= 5.0 or len(result_segments) == 1:
                _last_progress_pos = seg.end
                pct = min(99, int(seg.end / _audio_dur * 100)) if _audio_dur > 0 else 0
                progress_line = json.dumps({
                    "segments": len(result_segments),
                    "position_sec": round(seg.end, 1),
                    "pct": pct,
                    "lang": _detected_lang,
                })
                print(f"PROGRESS:{progress_line}", file=sys.stderr, flush=True)

        # ── TACT Phase 3 post-shift ──
        # When the offset trim above succeeded, every segment timestamp
        # is in trimmed-audio time. Shift back to original-audio time
        # by +offset_sec so downstream callers (the reconciler, the
        # ledger) see absolute timestamps.
        if offset_sec > 0.0:
            for s in result_segments:
                s["start"] = float(s.get("start", 0.0)) + offset_sec
                s["end"] = float(s.get("end", 0.0)) + offset_sec
                if s.get("words"):
                    for w in s["words"]:
                        w["start"] = float(w.get("start", 0.0)) + offset_sec
                        w["end"] = float(w.get("end", 0.0)) + offset_sec
            if offset_trim_path:
                try:
                    os.unlink(offset_trim_path)
                except OSError:
                    pass

        # Filter hallucinations (ghosts, loops, backward jumps, duplicates)
        result_segments = _filter_segments(result_segments, initial_prompt=args.initial_prompt or "", task=args.task)

        result = {
            "segments": result_segments,
            "info": {
                "language": info.language,
                "language_probability": info.language_probability,
                "duration": info.duration,
                "duration_after_vad": getattr(info, "duration_after_vad", info.duration),
            },
            "status": "ok",
        }

        # CTranslate2 silently returns 0 segments when cudaMalloc fails
        if len(result_segments) == 0 and info.duration > 10:
            logger.error(
                "WHISPER WORKER: 0 segments for %.1fs audio (language=%s, model=%s). "
                "CTranslate2 likely hit a silent CUDA OOM. "
                "Parent process should retry with a smaller model.",
                info.duration, info.language, args.model,
            )
            result["warning"] = "zero_segments_possible_oom"

        with open(args.output, "w") as f:
            json.dump(result, f)

        logger.info(
            "Transcription complete: %d segments, %.1fs, language=%s",
            len(result_segments), info.duration, info.language,
        )

        # Explicit cleanup before exit
        del model
        gc.collect()

        # Clean up preprocessed audio temp file
        if preprocessed_path != args.audio:
            try:
                os.unlink(preprocessed_path)
            except OSError:
                pass

    except Exception as e:
        logger.error("Whisper worker failed: %s", e, exc_info=True)
        with open(args.output, "w") as f:
            json.dump({"status": "error", "error": str(e)}, f)
        # Clean up preprocessed audio temp file on error
        try:
            if preprocessed_path != args.audio:
                os.unlink(preprocessed_path)
        except (OSError, NameError):
            pass
        sys.exit(1)

    # Process exits → OS reclaims ALL CUDA memory
    sys.exit(0)


if __name__ == "__main__":
    main()
