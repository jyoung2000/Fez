"""Two-pass visual verification for clip candidates.

After text-based clip detection, extract the first and last frame of each
candidate clip and send them to the vision model for verification. Adjusts
viral_score based on visual quality of the hook frame and conclusion frame.
"""

import asyncio
import logging
import os
from typing import Optional, Callable

from backend.models import ClipCandidate, FrameData

logger = logging.getLogger(__name__)


def _find_nearest_frame(frames: list[FrameData], target_time: float) -> Optional[FrameData]:
    """Find the frame closest to the target timestamp."""
    if not frames:
        return None
    return min(frames, key=lambda f: abs(f.timestamp - target_time))


async def verify_clips_visually(
    clips: list[ClipCandidate],
    frames: list[FrameData],
    vision_provider,
    cancel_check: Optional[Callable] = None,
    max_clips_to_verify: int = 8,
) -> list[ClipCandidate]:
    """Verify clip candidates by analyzing their hook and conclusion frames.

    For each clip, sends the first frame (hook) and last frame (conclusion) to
    the vision model for quality assessment. Adjusts viral_score based on:
    - Hook frame visual appeal (is it attention-grabbing?)
    - Conclusion frame quality (does it end on a strong visual?)
    - Visual continuity (do both frames belong to the same scene?)

    Returns clips with adjusted viral_scores.
    """
    if not clips or not frames:
        return clips

    # Only verify top clips to control API costs
    clips_to_verify = clips[:max_clips_to_verify]
    remaining_clips = clips[max_clips_to_verify:]

    verified = []
    for clip in clips_to_verify:
        if cancel_check:
            cancel_check()

        hook_frame = _find_nearest_frame(frames, clip.start_time)
        end_frame = _find_nearest_frame(frames, clip.end_time)

        if not hook_frame or not hook_frame.base64 or not end_frame or not end_frame.base64:
            verified.append(clip)
            continue

        try:
            # Build verification prompt
            prompt = (
                "You are evaluating two frames from a video clip for viral potential.\n\n"
                f"CLIP: \"{clip.title}\" ({clip.duration:.0f}s, score: {clip.viral_score})\n"
                f"Type: {clip.clip_type}\n\n"
                "Frame 1 is the HOOK (opening frame — first thing viewers see).\n"
                "Frame 2 is the CONCLUSION (final frame).\n\n"
                "Evaluate:\n"
                "1. Hook appeal (1-10): Is Frame 1 visually compelling enough to stop scrolling?\n"
                "2. Conclusion strength (1-10): Does Frame 2 end on a satisfying or impactful visual?\n"
                "3. Visual continuity (1-10): Do both frames look like they belong to the same scene/topic?\n"
                "4. Overall visual quality (1-10): Lighting, composition, clarity\n\n"
                "Return ONLY valid JSON:\n"
                '{"hook_appeal": 7, "conclusion_strength": 6, "visual_continuity": 8, '
                '"visual_quality": 7, "adjustment": 5, '
                '"reason": "Strong opening with clear subject, good resolution"}\n\n'
                "adjustment is a score modifier from -15 to +15 to apply to the viral score."
            )

            # Use the vision provider's analyze capability
            verification_frames = [hook_frame, end_frame]
            result = await vision_provider.analyze_frames(
                verification_frames, custom_prompt=prompt,
                cancel_check=cancel_check,
            )

            # Parse the result — it comes back as SceneDescriptions but we want the raw text
            if result and len(result) > 0:
                # The description field contains the analysis
                desc = result[0].description
                import json
                try:
                    # Try to extract JSON from the description
                    from backend.services.providers.base import extract_json
                    data = extract_json(desc)
                    adjustment = int(data.get("adjustment", 0))
                    adjustment = max(-15, min(15, adjustment))

                    original_score = clip.viral_score
                    clip.viral_score = max(1, min(100, clip.viral_score + adjustment))

                    if adjustment != 0:
                        reason = data.get("reason", "visual verification")
                        logger.info(
                            "Visual verification '%s': %d -> %d (%+d) — %s",
                            clip.title, original_score, clip.viral_score, adjustment, reason,
                        )
                except Exception:
                    pass  # Couldn't parse verification result, keep original score

        except Exception as e:
            logger.warning("Visual verification failed for '%s': %s", clip.title, e)

        verified.append(clip)

    # Re-sort by adjusted viral_score
    all_clips = verified + remaining_clips
    all_clips.sort(key=lambda c: c.viral_score, reverse=True)

    return all_clips
