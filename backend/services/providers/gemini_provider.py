import asyncio
import json
import logging
import time
from typing import Optional

import google.generativeai as genai
from PIL import Image

from backend.config import settings
from backend.models import (
    FrameData, SceneDescription, TranscriptSegment, VideoSummary, ClipCandidate, ClipSEO,
)
from backend.services.providers.base import (
    AIProvider, ChunkedClipDetectionMixin, ProviderError, ProviderRateLimitError,
    extract_json, extract_description_fallback, normalize_seo_data,
    build_fallback_summary, has_real_summary_content, build_summary_from_transcript,
    CLIP_JSON_SCHEMA_FOUR_AXIS, parse_clip_dict,
)
from backend.services.scene_description_validator import (
    sanitize_description, is_valid_description,
)
from backend.services.prompts import DEFAULT_FRAME_ANALYSIS_PROMPT, DEFAULT_VIRAL_CLIP_PROMPT, DEFAULT_SEO_PROMPT, DEFAULT_SUMMARY_PROMPT
from backend.services.transcript_utils import analyze_transcript_energy, correlate_scenes_with_transcript, derive_content_guidance

logger = logging.getLogger(__name__)

_VISION_TIMEOUT = 120   # 2 min for vision calls
_TEXT_TIMEOUT = 180      # 3 min for text calls


def _deduplicate_clips(clips: list[ClipCandidate], max_overlap: float = 0.5) -> list[ClipCandidate]:
    """Remove clips that overlap by more than max_overlap fraction of the shorter clip."""
    if len(clips) <= 1:
        return clips

    sorted_clips = sorted(clips, key=lambda c: c.viral_score, reverse=True)
    kept = []

    for clip in sorted_clips:
        is_duplicate = False
        for existing in kept:
            overlap_start = max(clip.start_time, existing.start_time)
            overlap_end = min(clip.end_time, existing.end_time)
            overlap_duration = max(0, overlap_end - overlap_start)
            shorter_duration = min(clip.duration, existing.duration)

            if shorter_duration > 0 and overlap_duration / shorter_duration > max_overlap:
                is_duplicate = True
                logger.info(
                    "De-dup: dropping '%s' (%.0f-%.0fs, score=%d) — overlaps %.0f%% with '%s'",
                    clip.title, clip.start_time, clip.end_time, clip.viral_score,
                    (overlap_duration / shorter_duration) * 100, existing.title,
                )
                break

        if not is_duplicate:
            kept.append(clip)

    if len(kept) < len(clips):
        logger.info("De-duplication: kept %d of %d clips", len(kept), len(clips))
    return kept


class GeminiProvider(ChunkedClipDetectionMixin, AIProvider):

    def __init__(self):
        genai.configure(api_key=settings.GEMINI_API_KEY)
        # Per-user model overrides via the settings overlay
        # (``GEMINI_TEXT_MODEL`` / ``GEMINI_VIDEO_MODEL``). Falls back
        # to sensible defaults so older installs without the overlay
        # keys keep working.
        text_model_id = getattr(settings, "GEMINI_TEXT_MODEL", "") or "gemini-2.0-flash"
        video_model_id = getattr(settings, "GEMINI_VIDEO_MODEL", "") or "gemini-2.5-flash"
        self._model_name = text_model_id
        self._video_model_name = video_model_id
        self._model = genai.GenerativeModel(text_model_id)
        # Native-video model — used for full-clip multimodal reasoning.
        self._video_model = genai.GenerativeModel(video_model_id)
        self._total_tokens = 0
        self._native_video_enabled = settings.GEMINI_USE_NATIVE_VIDEO

    @property
    def supports_vision(self) -> bool:
        return True

    @property
    def provider_name(self) -> str:
        return "gemini"

    @property
    def text_model_name(self) -> str:
        return getattr(self, '_model_name', 'gemini/unknown')

    async def _upload_video_file(self, video_path: str) -> Optional[object]:
        """Upload a video file via the Gemini File API for native video analysis.

        Returns the uploaded file object, or None if upload fails.
        """
        import os
        file_size_mb = os.path.getsize(video_path) / (1024 * 1024)

        # Gemini File API has a 2GB limit; skip for very large files
        if file_size_mb > 2000:
            logger.warning("Video too large for Gemini File API (%.0fMB > 2GB limit)", file_size_mb)
            return None

        try:
            logger.info("Uploading video to Gemini File API (%.1fMB)...", file_size_mb)
            # genai.upload_file is synchronous, run in executor
            loop = asyncio.get_event_loop()
            video_file = await loop.run_in_executor(
                None, lambda: genai.upload_file(video_path)
            )

            # Wait for file to be processed (polling)
            import time as _time
            max_wait = 300  # 5 minutes
            start = _time.monotonic()
            while video_file.state.name == "PROCESSING":
                if _time.monotonic() - start > max_wait:
                    logger.warning("Gemini file processing timed out after %ds", max_wait)
                    return None
                await asyncio.sleep(5)
                video_file = await loop.run_in_executor(
                    None, lambda: genai.get_file(video_file.name)
                )

            if video_file.state.name == "ACTIVE":
                logger.info("Video uploaded and processed: %s", video_file.name)
                return video_file
            else:
                logger.warning("Gemini file upload failed with state: %s", video_file.state.name)
                return None

        except Exception as e:
            logger.warning("Gemini File API upload failed: %s", e)
            return None

    async def analyze_video_native(
        self,
        video_path: str,
        transcript: list[TranscriptSegment],
        scenes: list[SceneDescription],
        video_duration: float,
        custom_prompt: Optional[str] = None,
        cancel_check=None,
        clip_count: Optional[int] = None,
        min_duration: Optional[float] = None,
        max_duration: Optional[float] = None,
        video_summary: Optional[str] = None,
    ) -> tuple[list[SceneDescription], list[ClipCandidate]]:
        """Perform native video analysis using Gemini's video understanding.

        Uploads the video file and sends a single multimodal prompt that combines
        scene description and clip detection in one pass, giving far richer
        understanding than static frame analysis.

        Returns (scenes, clips) tuple.
        """
        video_file = await self._upload_video_file(video_path)
        if not video_file:
            raise ProviderError("Failed to upload video for native analysis")

        dur_min = int(min_duration) if min_duration else 30
        dur_max = int(max_duration) if max_duration else 300
        num_clips = clip_count or settings.MAX_CLIP_CANDIDATES

        content_guidance = derive_content_guidance(video_summary)
        energy_text = analyze_transcript_energy(transcript)

        transcript_text = "\n".join(
            f"[{s.start:.1f}-{s.end:.1f}] {s.speaker}: {s.text}" for s in transcript[:500]
        )
        if len(transcript_text) > 30000:
            transcript_text = transcript_text[:30000] + "\n... (truncated)"

        summary_section = ""
        if video_summary:
            summary_section = f"\nVIDEO SUMMARY:\n{video_summary}\n"

        prompt = (
            "You are analyzing a video file directly. You can see every frame, hear audio cues, "
            "and understand the full visual narrative. This gives you much richer context than "
            "static frame samples.\n\n"
            "Perform TWO tasks in a single response:\n\n"
            "TASK 1: SCENE ANALYSIS\n"
            "Identify the most important visual moments in the video. For each, provide:\n"
            "- timestamp (seconds from start)\n"
            "- description of what's happening visually\n"
            "- importance_score (1-10)\n"
            "- subject_x (0-100, horizontal position of main subject: 0=left, 50=center, 100=right)\n\n"
            "TASK 2: VIRAL CLIP DETECTION\n"
            f"{content_guidance}"
            "STRICT REQUIREMENTS:\n"
            f"- Each clip duration MUST be between {dur_min} and {dur_max} seconds\n"
            "- Start at natural speech boundaries\n"
            "- End at natural conclusions\n"
            "- Must work standalone without context from the full video\n"
            "- The main subject/speaker MUST remain in focus for the entire clip\n"
            "- When visual peaks coincide with strong transcript content, score higher\n\n"
            f"Video duration: {video_duration:.1f}s\n"
            f"{summary_section}"
            f"\nTRANSCRIPT:\n{transcript_text}\n"
            f"{energy_text}\n\n"
            f"Return ONLY valid JSON with this structure:\n"
            '{"scenes": [{"timestamp": 0.0, "description": "...", "importance_score": 7, "subject_x": 50}], '
            f'"clips": [{{"id": 1, "title": "...", "start_time": 0.0, "end_time": 0.0, '
            '"duration": 0.0, "viral_score": 80, "viral_score_reasoning": "...", '
            '"clip_type": "highlight", "platform": "both", "suggested_caption": "...", '
            f'"hook_text": "...", "why_this_works": "..."}}]}}\n\n'
            f"Return up to 20 key scenes and up to {num_clips} viral clips."
        )

        try:
            content = [video_file, prompt]
            raw = await self._call(content, max_tokens=16384, timeout=600)

            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
            data = json.loads(raw)

            # Parse scenes
            parsed_scenes = []
            for s in data.get("scenes", []):
                sx = s.get("subject_x", 50)
                sx = max(0, min(100, int(sx))) if sx is not None else 50
                parsed_scenes.append(SceneDescription(
                    timestamp=float(s.get("timestamp", 0)),
                    description=str(s.get("description", "")),
                    importance_score=max(1, min(10, int(s.get("importance_score", 5)))),
                    thumbnail_path="",  # No thumbnail for native video analysis
                    subject_x=sx,
                ))

            # Parse clips
            parsed_clips = []
            for c in data.get("clips", []):
                parsed, _ = parse_clip_dict(
                    c, fallback_id=len(parsed_clips) + 1,
                    min_duration=float(min_duration or 15),
                    max_duration=float(max_duration or 600),
                )
                if parsed is not None:
                    parsed_clips.append(parsed)

            parsed_clips = _deduplicate_clips(parsed_clips)
            logger.info(
                "Gemini native video analysis: %d scenes, %d clips",
                len(parsed_scenes), len(parsed_clips),
            )
            return parsed_scenes, parsed_clips

        except Exception as e:
            logger.error("Gemini native video analysis failed: %s", e)
            raise ProviderError(f"Native video analysis failed: {e}")
        finally:
            # Clean up uploaded file
            try:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, lambda: genai.delete_file(video_file.name))
                logger.info("Cleaned up Gemini uploaded file: %s", video_file.name)
            except Exception:
                pass

    async def _call(self, content, max_tokens: int = 4096, timeout: int = _TEXT_TIMEOUT) -> str:
        t0 = time.monotonic()
        try:
            response = await asyncio.wait_for(
                self._model.generate_content_async(
                    content,
                    generation_config=genai.types.GenerationConfig(
                        max_output_tokens=max_tokens,
                        temperature=0.3,
                    ),
                ),
                timeout=timeout,
            )
            elapsed = time.monotonic() - t0
            logger.info("Gemini call completed in %.1fs", elapsed)
            if response.usage_metadata:
                self._total_tokens += (
                    response.usage_metadata.prompt_token_count
                    + response.usage_metadata.candidates_token_count
                )
            return response.text or ""
        except asyncio.TimeoutError:
            logger.error("Gemini call timed out after %ds", timeout)
            raise ProviderError(f"Gemini timeout: no response in {timeout}s")
        except Exception as e:
            err_str = str(e).lower()
            if "429" in str(e) or "quota" in err_str or "rate" in err_str:
                raise ProviderRateLimitError(f"Gemini rate limited: {e}")
            raise ProviderError(f"Gemini error: {e}")

    async def text_complete(self, prompt: str, max_tokens: int = 4096, timeout: int | None = None) -> str:
        return await self._call(prompt, max_tokens=max_tokens)

    async def analyze_frames(
        self, frames: list[FrameData], custom_prompt: Optional[str] = None,
        cancel_check=None, progress_callback=None,
    ) -> list[SceneDescription]:
        instruction = custom_prompt if custom_prompt else DEFAULT_FRAME_ANALYSIS_PROMPT
        batch_size = 4
        total = len(frames)
        num_batches = (total + batch_size - 1) // batch_size
        batch_results: list[list[SceneDescription]] = [[] for _ in range(num_batches)]
        frames_completed = 0
        # Process up to 2 batches concurrently
        sem = asyncio.Semaphore(2)

        async def _process_batch(batch_idx: int):
            nonlocal frames_completed
            async with sem:
                if cancel_check:
                    cancel_check()
                start = batch_idx * batch_size
                batch = frames[start : start + batch_size]
                content = [
                    instruction + "\n\n"
                    "Return ONLY valid JSON array:\n"
                    '[{"timestamp": <float>, "description": "<text>", "importance_score": <1-10>, "subject_x": <0-100>, "active_speaker_x": <0-100 or null>}]\n'
                    "IMPORTANT: subject_x = horizontal position of the ACTIVE SPEAKER (the person "
                    "whose lips are moving or who is currently talking). If you can tell who is "
                    "speaking, use THEIR face position. If no one is clearly speaking, use the most "
                    "prominent person. active_speaker_x = same value if confident someone is speaking, "
                    "null otherwise. 0=left edge, 50=center, 100=right edge. "
                    "Do NOT default to 50 — carefully estimate the actual horizontal position."
                ]
                for frame in batch:
                    try:
                        img = Image.open(frame.path)
                        content.append(img)
                        content.append(f"[Frame at {frame.timestamp:.1f}s]")
                    except Exception:
                        content.append(f"[Frame at {frame.timestamp:.1f}s - could not load]")

                raw = await self._call(content, timeout=_VISION_TIMEOUT)
                try:
                    raw = raw.strip()
                    if raw.startswith("```"):
                        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
                    parsed = json.loads(raw)
                    if not isinstance(parsed, list):
                        parsed = [parsed]
                    for idx, item in enumerate(parsed):
                        frame_ref = batch[idx] if idx < len(batch) else batch[-1]
                        sx = item.get("subject_x")
                        if sx is None:
                            logger.warning(
                                "Batch %d frame %d: Gemini response missing subject_x field",
                                batch_idx, idx,
                            )
                            sx = 50
                        else:
                            sx = max(0, min(100, int(sx)))
                        _gem_desc = sanitize_description(item.get("description", ""))
                        if not is_valid_description(_gem_desc)[0]:
                            _gem_desc = ""
                        batch_results[batch_idx].append(SceneDescription(
                            timestamp=item.get("timestamp", frame_ref.timestamp),
                            description=_gem_desc,
                            importance_score=max(1, min(10, int(item.get("importance_score", 5)))),
                            thumbnail_path=frame_ref.path,
                            subject_x=sx,
                        ))
                except (json.JSONDecodeError, KeyError, IndexError) as e:
                    logger.warning(f"Failed to parse Gemini frame analysis: {e}")
                    fallback_desc = extract_description_fallback(raw) if raw else ""
                    clean_fb = sanitize_description(fallback_desc)
                    if not is_valid_description(clean_fb)[0]:
                        clean_fb = ""
                    for frame in batch:
                        batch_results[batch_idx].append(SceneDescription(
                            timestamp=frame.timestamp,
                            description=clean_fb,
                            importance_score=5,
                            thumbnail_path=frame.path,
                            subject_x=50,
                        ))
                frames_completed += len(batch)
                if progress_callback:
                    await progress_callback(min(frames_completed, total), total)

        await asyncio.gather(*[_process_batch(i) for i in range(num_batches)])
        scenes = []
        for batch_scene_list in batch_results:
            scenes.extend(batch_scene_list)
        return scenes

    async def generate_summary(
        self,
        transcript: list[TranscriptSegment],
        scenes: list[SceneDescription],
        cancel_check=None,
        custom_prompt=None,
    ) -> VideoSummary:
        instruction = custom_prompt if custom_prompt else DEFAULT_SUMMARY_PROMPT
        transcript_text = "\n".join(
            f"[{s.start:.1f}-{s.end:.1f}] {s.speaker}: {s.text}" for s in transcript
        )
        scene_text = "\n".join(
            f"[{s.timestamp:.1f}s] (importance: {s.importance_score}/10) {s.description}"
            for s in scenes
        )
        prompt = (
            f"{instruction}\n\n"
            f"TRANSCRIPT:\n{transcript_text}\n\n"
            f"SCENES:\n{scene_text}\n\n"
            "Return ONLY valid JSON:\n"
            '{"overview": "<paragraph>", "key_topics": ["<topic1>", ...], '
            '"tone": "<tone>", "estimated_audience": "<audience>", "content_category": "<category>"}'
        )
        raw = await self._call([prompt])
        try:
            data = extract_json(raw)
            if not has_real_summary_content(data):
                logger.warning("Summary JSON has placeholder values, trying fallback extraction")
                raise ValueError("Placeholder values detected in summary")
            return VideoSummary(**data)
        except Exception:
            logger.warning("Failed to parse summary JSON, using fallback extraction. Raw (first 300): %s", raw[:300])
            fb = build_fallback_summary(raw)
            if not has_real_summary_content(fb):
                logger.warning("Fallback extraction also produced placeholders, building from transcript")
                fb = build_summary_from_transcript(transcript, scenes)
            return VideoSummary(**fb)

    async def detect_viral_clips(
        self,
        transcript: list[TranscriptSegment],
        scenes: list[SceneDescription],
        video_duration: float,
        custom_prompt: Optional[str] = None,
        cancel_check=None,
        clip_count: Optional[int] = None,
        min_duration: Optional[float] = None,
        max_duration: Optional[float] = None,
        video_summary: Optional[str] = None,
        existing_clips: Optional[str] = None,
        hot_zones=None,
        progress_callback=None,
        tier=None,
        _partial_results: Optional[list] = None,
    ) -> list[ClipCandidate]:
        """Dispatch to multi-pass for long videos, single-pass for short."""
        if video_duration > 300:
            logger.info("Gemini: video %.0fs (>5min) — using multi-pass clip detection", video_duration)
            return await self._multi_pass_clip_detection(
                transcript, scenes, video_duration,
                tier=tier, sequential=False,
                custom_prompt=custom_prompt, cancel_check=cancel_check,
                clip_count=clip_count, min_duration=min_duration,
                max_duration=max_duration, video_summary=video_summary,
                existing_clips=existing_clips,
                hot_zones=hot_zones,
                progress_callback=progress_callback,
                _partial_results=_partial_results,
            )
        return await self._single_pass_clip_detection(
            transcript, scenes, video_duration,
            custom_prompt=custom_prompt, cancel_check=cancel_check,
            clip_count=clip_count, min_duration=min_duration,
            max_duration=max_duration, video_summary=video_summary,
            existing_clips=existing_clips,
            hot_zones=hot_zones,
        )

    async def _single_pass_clip_detection(
        self,
        transcript: list[TranscriptSegment],
        scenes: list[SceneDescription],
        video_duration: float,
        custom_prompt: Optional[str] = None,
        cancel_check=None,
        clip_count: Optional[int] = None,
        min_duration: Optional[float] = None,
        max_duration: Optional[float] = None,
        video_summary: Optional[str] = None,
        existing_clips: Optional[str] = None,
        hot_zones=None,
        **kwargs,
    ) -> list[ClipCandidate]:
        instruction = custom_prompt if custom_prompt else DEFAULT_VIRAL_CLIP_PROMPT

        # Derive content-type guidance from video summary
        content_guidance = derive_content_guidance(video_summary)

        # Pre-process transcript for energy signals
        energy_text = analyze_transcript_energy(transcript)

        # Correlate scenes with transcript for audio-visual peaks
        av_correlation = correlate_scenes_with_transcript(transcript, scenes)

        # Gemini has large context but cap for response speed
        max_transcript = 40000
        max_scenes = 12000
        transcript_text = "\n".join(
            f"[{s.start:.1f}-{s.end:.1f}] {s.speaker}: {s.text}" for s in transcript
        )
        if len(transcript_text) > max_transcript:
            transcript_text = transcript_text[:max_transcript] + f"\n... (truncated, {len(transcript)} total segments)"

        # Build scene text maintaining chronological order with importance flags
        scene_lines: list[str] = []
        for s in scenes:
            importance_flag = " ★" if s.importance_score >= 7 else ""
            desc_limit = 180 if s.importance_score >= 7 else 100
            desc = s.description[:desc_limit] if len(s.description) > desc_limit else s.description
            scene_lines.append(f"[{s.timestamp:.1f}s] ({s.importance_score}/10{importance_flag}) {desc}")
        scene_text = "\n".join(scene_lines)
        if len(scene_text) > max_scenes:
            scene_text = scene_text[:max_scenes] + f"\n... (truncated, {len(scenes)} total scenes)"

        dur_min = int(min_duration) if min_duration else 30
        dur_max = int(max_duration) if max_duration else 300
        num_clips = clip_count or settings.MAX_CLIP_CANDIDATES
        summary_section = ""
        if video_summary:
            summary_section = f"VIDEO SUMMARY:\n{video_summary}\n\n"

        existing_clips_section = ""
        if existing_clips:
            existing_clips_section = (
                f"\n\nALREADY IDENTIFIED CLIPS (find DIFFERENT moments, do not overlap):\n"
                f"{existing_clips}\n"
                f"Find clips that cover DIFFERENT timestamps and topics from the above."
            )

        prompt = (
            instruction + "\n\n"
            f"{content_guidance}"
            "STRICT REQUIREMENTS:\n"
            f"- Each clip duration MUST be between {dur_min} and {dur_max} seconds\n"
            "- Segments marked [LOW_CONF] have unreliable transcription — avoid clips where "
            "multiple [LOW_CONF] segments appear, as the actual dialogue may differ significantly\n"
            "- Start at natural speech boundaries — beginning of a sentence, after a pause, at a speaker change\n"
            "- End at natural conclusions — punchlines, resolved thoughts, scene transitions\n"
            "- Must work standalone without context from the full video\n"
            "- The main subject/speaker MUST remain in focus for the entire clip\n"
            "- Do NOT combine scenes from different settings or unrelated topics\n"
            "- When a visual peak (★ scene) coincides with strong transcript content, score that clip higher\n\n"
            f"Video duration: {video_duration:.1f}s\n\n"
            f"{summary_section}"
            f"TRANSCRIPT:\n{transcript_text}\n\n"
            f"SCENES:\n{scene_text}"
            f"{energy_text}"
            f"{av_correlation}"
            f"{existing_clips_section}\n\n"
            f"Return UP TO {num_clips} viral clip candidates, ranked by viral potential from highest to lowest. "
            f"Only return clips that genuinely score 40+ on viral potential. "
            f"It is better to return fewer high-quality clips than to pad with weak filler clips. "
            f"If the video has fewer than {num_clips} genuinely strong moments, return only the strong ones. "
            f"Each clip must be between {dur_min} and {dur_max} seconds long.\n\n"
            "Return ONLY valid JSON:\n"
            + CLIP_JSON_SCHEMA_FOUR_AXIS
        )

        # Store the original prompt for retries (don't mutate it)
        original_prompt = prompt

        for attempt in range(3):
            if cancel_check:
                cancel_check()
            raw = await self._call([prompt], max_tokens=8192)
            try:
                raw = raw.strip()
                if raw.startswith("```"):
                    raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
                data = json.loads(raw)
                clips = []
                for c in data.get("clips", []):
                    parsed, _ = parse_clip_dict(
                        c, fallback_id=len(clips) + 1,
                        min_duration=float(min_duration or 15),
                        max_duration=float(max_duration or 600),
                    )
                    if parsed is not None:
                        clips.append(parsed)
                if clips:
                    clips = _deduplicate_clips(clips)
                    logger.info(f"Parsed {len(clips)} valid clips after de-duplication")
                    return clips
                logger.warning(f"Attempt {attempt + 1}: All Gemini clips filtered out")
                continue
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Attempt {attempt + 1}: Failed to parse Gemini clips: {e}")
                # Reset prompt to original — don't keep appending
                prompt = original_prompt
                continue
        raise ProviderError("Failed to parse viral clips after 3 attempts")

    async def generate_seo(
        self, clip_title: str, clip_transcript: str, video_summary: str,
        platform: str, cancel_check=None, custom_prompt=None,
    ) -> ClipSEO:
        seo_instruction = custom_prompt if custom_prompt else DEFAULT_SEO_PROMPT
        is_description = video_summary.startswith("DESCRIPTION_OVERRIDE")
        if is_description:
            prompt = (
                f"{video_summary}\n\n"
                f"CLIP TITLE: {clip_title}\n"
                f"TARGET PLATFORM: {platform}\n\n"
                f"CLIP TRANSCRIPT:\n{clip_transcript}\n"
            )
        else:
            prompt = (
                f"{seo_instruction}\n\n"
                f"CLIP TITLE: {clip_title}\n"
                f"TARGET PLATFORM: {platform}\n\n"
                f"VIDEO SUMMARY:\n{video_summary}\n\n"
                f"CLIP TRANSCRIPT:\n{clip_transcript}\n"
            )
        tokens = 16384 if is_description else 4096
        raw = await self._call([prompt], max_tokens=tokens)
        try:
            data = normalize_seo_data(extract_json(raw))
            return ClipSEO(**data)
        except Exception:
            logger.warning(f"Failed to parse SEO JSON, using fallback. Raw (first 300): {raw[:300]}")
            desc = extract_description_fallback(raw) if is_description and raw else (raw[:300] if raw else "SEO generation failed")
            return ClipSEO(title=clip_title, description=desc, tags=[], platform_tips="")
