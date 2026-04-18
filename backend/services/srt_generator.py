"""Generate SRT subtitle files from transcript segments with speaker labels."""

from backend.models import TranscriptSegment


def _format_srt_time(seconds: float) -> str:
    """Convert seconds to SRT timestamp format: HH:MM:SS,mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def generate_srt(
    segments: list[TranscriptSegment],
    include_speakers: bool = True,
) -> str:
    """Convert transcript segments to SRT format with optional speaker labels.

    Example output:
        1
        00:00:01,200 --> 00:00:04,800
        [Speaker 1] Hello everyone, welcome to the show.

        2
        00:00:05,000 --> 00:00:08,300
        [Speaker 2] Thanks for having me!
    """
    lines: list[str] = []
    for idx, seg in enumerate(segments, start=1):
        start = _format_srt_time(seg.start)
        end = _format_srt_time(seg.end)
        text = seg.text.strip()
        if not text:
            continue
        if include_speakers and seg.speaker:
            text = f"[{seg.speaker}] {text}"
        lines.append(f"{idx}")
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")  # blank line separator
    return "\n".join(lines)
