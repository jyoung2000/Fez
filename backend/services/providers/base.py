import asyncio
import json
import logging
import re
from abc import ABC, abstractmethod
from typing import Callable, Optional

from backend.models import FrameData, SceneDescription, TranscriptSegment, VideoSummary, ClipCandidate, ClipSEO

_mixin_logger = logging.getLogger(__name__)


def _fix_json_newlines(text: str) -> str:
    """Escape literal newlines that appear inside JSON string values.

    LLMs frequently return JSON with unescaped newlines inside string
    values (e.g. multi-paragraph descriptions).  ``json.loads`` rejects
    these, so we walk the text and replace literal ``\\n`` inside quoted
    strings with the escaped ``\\\\n`` sequence.
    """
    out: list[str] = []
    in_string = False
    escape_next = False
    for ch in text:
        if escape_next:
            out.append(ch)
            escape_next = False
            continue
        if ch == '\\' and in_string:
            out.append(ch)
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
        if in_string and ch == '\n':
            out.append('\\n')
        elif in_string and ch == '\r':
            out.append('\\r')
        elif in_string and ch == '\t':
            out.append('\\t')
        else:
            out.append(ch)
    return ''.join(out)


def extract_json(raw: str) -> dict:
    """Extract and parse JSON from an LLM response, handling thinking tags,
    markdown code blocks, literal newlines in strings, and other common
    wrappers."""
    text = raw.strip()
    # Strip <think>...</think> or <reasoning>...</reasoning> blocks (complete)
    text = re.sub(r'<(?:think|reasoning)>.*?</(?:think|reasoning)>', '', text, flags=re.DOTALL).strip()
    # Strip unclosed thinking tags (model timed out mid-think)
    # e.g. "<think>reasoning text\n\n{"clips": [...]}"
    text = re.sub(r'<(?:think|reasoning)>.*?(?=\{)', '', text, flags=re.DOTALL).strip()
    # Strip markdown code fences
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try to find JSON object within the text
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        snippet = text[start:end + 1]
        try:
            return json.loads(snippet)
        except json.JSONDecodeError:
            pass
        # LLMs often put literal newlines inside JSON string values —
        # escape them and retry.
        try:
            return json.loads(_fix_json_newlines(snippet))
        except json.JSONDecodeError:
            pass
    raise json.JSONDecodeError("No valid JSON found in response", text, 0)


def extract_description_fallback(raw: str) -> str:
    """Best-effort extraction of description text when JSON parsing fails.

    Looks for ``"description": "..."`` in the raw response and pulls out
    everything between the opening and closing quotes, handling the common
    case where the AI returns well-structured JSON but with literal
    newlines that break ``json.loads``.
    """
    # Strip thinking blocks first
    text = re.sub(r'<(?:think|reasoning)>.*?</(?:think|reasoning)>', '', raw, flags=re.DOTALL)
    m = re.search(r'"description"\s*:\s*"', text)
    if not m:
        return text[:500] if text else ""
    start = m.end()
    # Walk forward to find the unescaped closing quote
    i = start
    chars: list[str] = []
    while i < len(text):
        if text[i] == '\\' and i + 1 < len(text):
            # Escaped character — keep the real char
            nxt = text[i + 1]
            if nxt == 'n':
                chars.append('\n')
            elif nxt == 't':
                chars.append('\t')
            elif nxt == '"':
                chars.append('"')
            elif nxt == '\\':
                chars.append('\\')
            else:
                chars.append(nxt)
            i += 2
        elif text[i] == '"':
            break  # unescaped closing quote
        else:
            chars.append(text[i])
            i += 1
    return ''.join(chars) or text[:500]


def build_fallback_summary(raw: str) -> dict:
    """Extract a human-readable summary from raw AI text when JSON parsing fails.

    Instead of dumping raw AI output into the overview, tries to pull out
    meaningful content or generates a clean fallback.
    """
    if not raw or not raw.strip():
        return {
            "overview": "Could not generate a summary for this video.",
            "key_topics": [],
            "tone": "unknown",
            "estimated_audience": "general",
            "content_category": "uncategorized",
        }

    text = raw.strip()
    # Strip thinking/reasoning blocks
    text = re.sub(r'<(?:think|reasoning)>.*?</(?:think|reasoning)>', '', text, flags=re.DOTALL).strip()
    # Strip markdown code fences
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    # Try to extract individual fields from partial/broken JSON
    result = {}
    for field in ("overview", "tone", "estimated_audience", "content_category"):
        m = re.search(rf'"{field}"\s*:\s*"', text)
        if m:
            start = m.end()
            i = start
            chars = []
            while i < len(text):
                if text[i] == '\\' and i + 1 < len(text):
                    nxt = text[i + 1]
                    chars.append('\n' if nxt == 'n' else nxt)
                    i += 2
                elif text[i] == '"':
                    break
                else:
                    chars.append(text[i])
                    i += 1
            result[field] = ''.join(chars).strip()

    # Try to extract key_topics array (case-insensitive)
    m = re.search(r'"[Kk]ey_[Tt]opics"\s*:\s*\[', text)
    if not m:
        m = re.search(r'"KEY_TOPICS"\s*:\s*\[', text)
    if m:
        start = m.end()
        end = text.find(']', start)
        if end > start:
            topics_str = text[start:end]
            topics = re.findall(r'"([^"]+)"', topics_str)
            result["key_topics"] = topics

    # If we got an overview from partial JSON, use it
    if result.get("overview"):
        return {
            "overview": result["overview"],
            "key_topics": result.get("key_topics", []),
            "tone": result.get("tone", "unknown"),
            "estimated_audience": result.get("estimated_audience", "general"),
            "content_category": result.get("content_category", "uncategorized"),
        }

    # Last resort: strip JSON artifacts and use cleaned text as overview
    # Remove JSON syntax characters and clean up
    cleaned = re.sub(r'[{}\[\]"]', '', text)
    cleaned = re.sub(r'\b(?:overview|key_topics|tone|estimated_audience|content_category)\s*:', '', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    if cleaned and len(cleaned) > 20:
        return {
            "overview": cleaned[:500],
            "key_topics": [],
            "tone": "unknown",
            "estimated_audience": "general",
            "content_category": "uncategorized",
        }

    return {
        "overview": "Could not generate a summary for this video.",
        "key_topics": [],
        "tone": "unknown",
        "estimated_audience": "general",
        "content_category": "uncategorized",
    }


_PLACEHOLDER_PATTERNS = {
    "...", "…", "<paragraph>", "<topic1>", "<topic2>", "<topic3>",
    "<tone>", "<audience>", "<category>", "n/a", "N/A", "none",
    "placeholder", "undefined", "null",
}


def _is_placeholder(value: str) -> bool:
    """Check if a string value is a placeholder rather than real content."""
    stripped = value.strip()
    if not stripped or len(stripped) < 3:
        return True
    if stripped in _PLACEHOLDER_PATTERNS:
        return True
    # Strings that are all dots/ellipsis
    if all(c in '.…' for c in stripped):
        return True
    # Angle-bracket placeholders like <something>
    if stripped.startswith('<') and stripped.endswith('>'):
        return True
    return False


def has_real_summary_content(data: dict) -> bool:
    """Check if a parsed summary dict contains actual content, not placeholders."""
    overview = data.get("overview", "")
    if _is_placeholder(overview):
        return False
    key_topics = data.get("key_topics", [])
    # At minimum, overview must be substantial
    if len(overview.strip()) < 20:
        return False
    # Check that at least some topics exist and aren't placeholders
    real_topics = [t for t in key_topics if not _is_placeholder(t)]
    if not real_topics:
        return False
    return True


def build_summary_from_transcript(
    transcript: list["TranscriptSegment"],
    scenes: list["SceneDescription"],
) -> dict:
    """Build a basic human-readable summary directly from transcript and scene data.

    This is a deterministic last-resort fallback that always produces
    meaningful content without requiring an LLM call.
    """
    # Build overview from first several transcript segments
    overview_parts = []
    speakers_seen = set()
    total_chars = 0
    for seg in transcript:
        speakers_seen.add(seg.speaker)
        overview_parts.append(seg.text)
        total_chars += len(seg.text)
        if total_chars > 600:
            break

    num_speakers = len(speakers_seen)
    duration_mins = 0
    if transcript:
        duration_mins = round((transcript[-1].end - transcript[0].start) / 60)

    if overview_parts:
        combined_text = " ".join(overview_parts)
        # Truncate to ~300 chars at word boundary
        if len(combined_text) > 300:
            combined_text = combined_text[:300].rsplit(" ", 1)[0] + "..."
        if num_speakers > 1:
            overview = (
                f"A {duration_mins}-minute video featuring {num_speakers} speakers. "
                f"The conversation covers: {combined_text}"
            )
        else:
            overview = (
                f"A {duration_mins}-minute video. "
                f"The content covers: {combined_text}"
            )
    else:
        overview = "Video analysis completed but no transcript was available to generate a detailed summary."

    # Extract key topics from high-importance scenes (skip synthetic descriptions)
    _synthetic_prefixes = ("Frame at ", "Video frame at ", "Continuation of video")
    topics = []
    seen_topic_words = set()
    for scene in sorted(scenes, key=lambda s: s.importance_score, reverse=True):
        if len(topics) >= 5:
            break
        desc = scene.description.strip()
        if not desc or any(desc.startswith(p) for p in _synthetic_prefixes):
            continue
        if "unavailable" in desc.lower() or "analysis" in desc.lower():
            continue
        # Use first sentence or first 60 chars as topic
        topic = desc.split(".")[0].strip()
        if len(topic) > 60:
            topic = topic[:60].rsplit(" ", 1)[0]
        if len(topic) < 5:
            continue
        # Deduplicate by checking for word overlap
        topic_words = set(topic.lower().split())
        if topic_words & seen_topic_words and len(topic_words & seen_topic_words) > 2:
            continue
        seen_topic_words |= topic_words
        topics.append(topic)

    # If no scene topics, extract from transcript
    if not topics and transcript:
        # Use unique first words of segments as rough topics
        for seg in transcript[:20]:
            text = seg.text.strip()
            if len(text) > 15 and len(topics) < 4:
                topic = text.split(".")[0].strip()
                if len(topic) > 60:
                    topic = topic[:60].rsplit(" ", 1)[0]
                if topic and topic not in topics:
                    topics.append(topic)

    if not topics:
        topics = ["video content"]

    # Determine tone from scene importance scores
    if scenes:
        avg_score = sum(s.importance_score for s in scenes) / len(scenes)
        if avg_score >= 7:
            tone = "engaging and dynamic"
        elif avg_score >= 5:
            tone = "conversational"
        else:
            tone = "casual"
    else:
        tone = "conversational"

    audience = "general viewers"
    category = "video content"

    return {
        "overview": overview,
        "key_topics": topics,
        "tone": tone,
        "estimated_audience": audience,
        "content_category": category,
    }


def normalize_seo_data(data: dict) -> dict:
    """Merge legacy 'hashtags' field into 'tags' and ensure all tags have # prefix."""
    tags = list(data.get("tags", []))
    # Merge any separate hashtags field into tags
    hashtags = data.pop("hashtags", [])
    if hashtags:
        existing = {t.lower().lstrip("#") for t in tags}
        for h in hashtags:
            if h.lower().lstrip("#") not in existing:
                tags.append(h)
    # Ensure all tags have # prefix
    data["tags"] = [t if t.startswith("#") else f"#{t}" for t in tags]
    return data


def extract_partial_clips(raw: str) -> list[dict]:
    """Extract complete clip objects from a potentially truncated JSON response.

    When a timeout fires mid-generation, we may have a partial JSON response.
    This function extracts the complete clip objects and discards truncated ones.
    """
    text = re.sub(r'<(?:think|reasoning)>.*?</(?:think|reasoning)>', '', raw, flags=re.DOTALL)

    clips_match = re.search(r'"clips"\s*:\s*\[', text)
    if not clips_match:
        return []

    clips = []
    pos = clips_match.end()
    brace_depth = 0
    obj_start = None

    for i in range(pos, len(text)):
        ch = text[i]
        if ch == '{' and brace_depth == 0:
            obj_start = i
            brace_depth = 1
        elif ch == '{':
            brace_depth += 1
        elif ch == '}':
            brace_depth -= 1
            if brace_depth == 0 and obj_start is not None:
                try:
                    obj_text = text[obj_start:i + 1]
                    obj = json.loads(_fix_json_newlines(obj_text))
                    clips.append(obj)
                except json.JSONDecodeError:
                    pass
                obj_start = None
        elif ch == ']' and brace_depth == 0:
            break

    return clips


# ── Phase 1 (4-axis scoring) helpers ────────────────────────────────
# Every provider builds a JSON schema string for the LLM and parses
# the response back into ClipCandidate. We centralise the schema and
# parser here so adding axes / fields only needs one edit.

CLIP_JSON_SCHEMA_FOUR_AXIS = (
    '{"clips": [{"id": 1, '
    '"title": "SEO social media title about the topic (no speaker names)", '
    '"start_time": 45.2, "end_time": 112.8, "duration": 67.6, '
    '"hook_score": 88, "hook_reason": "opens with bold question", '
    '"flow_score": 75, "flow_reason": "single coherent exchange", '
    '"value_score": 80, "value_reason": "delivers a quotable hot take", '
    '"trend_score": 60, "trend_reason": "topic adjacent to trending", '
    '"clip_type": "informative|funny|emotional|shocking|tutorial|highlight|debate|reveal", '
    '"platform": "tiktok|youtube_shorts|both", '
    '"suggested_caption": "Caption with #hashtags", '
    '"hook_text": "Text overlay for opening frame", '
    '"why_this_works": "One sentence explanation"}], '
    '"total_candidates": 8, "best_clip_id": 1}'
)


def _coerce_axis_score(raw, default: int = 0) -> int:
    """Coerce an LLM-returned score field to an int 0-100.

    Handles ``None``, strings, floats, and out-of-range values. Returns
    ``default`` (typically 0 or 50) when the value is missing or
    unparseable.
    """
    if raw is None or raw == "":
        return default
    try:
        score = int(round(float(raw)))
    except (TypeError, ValueError):
        return default
    return max(0, min(100, score))


def parse_clip_dict(
    c: dict,
    fallback_id: int = 1,
    min_duration: float = 15.0,
    max_duration: float = 600.0,
) -> "tuple[ClipCandidate | None, str | None]":
    """Parse one LLM-returned clip dict into a ``ClipCandidate``.

    Returns ``(clip, None)`` on success or ``(None, reason)`` on
    validation failure. Centralises the boilerplate every provider
    used to repeat: duration validation, ID coercion, axis-score
    parsing, and backward compatibility with the legacy
    ``viral_score`` field.

    Backward compatibility: if the LLM did not return the four axes
    (older / fine-tuned models), we splat the legacy ``viral_score``
    across all four axes via ``fill_axes_from_legacy`` so downstream
    code paths still work.
    """
    try:
        start = float(c.get("start_time", 0))
        end = float(c.get("end_time", 0))
        duration = end - start
        if duration <= 0:
            duration = float(c.get("duration", 0))
        if duration < min_duration:
            return None, f"too short ({duration:.1f}s)"
        if duration > max_duration:
            return None, f"too long ({duration:.1f}s)"

        focus_relevance = c.get("focus_relevance")
        if focus_relevance is not None:
            try:
                focus_relevance = max(1, min(100, int(float(focus_relevance))))
            except (TypeError, ValueError):
                focus_relevance = None
        focus_tier = c.get("focus_tier")
        if focus_tier and focus_tier not in ("strong", "moderate", "weak"):
            focus_tier = None

        # Axis scores. ``trend_score`` defaults to 50 (neutral) when
        # the LLM omits it — we never want a missing trend signal to
        # drag the composite to 0.
        hook_score = _coerce_axis_score(c.get("hook_score"), default=0)
        flow_score = _coerce_axis_score(c.get("flow_score"), default=0)
        value_score = _coerce_axis_score(c.get("value_score"), default=0)
        trend_score = _coerce_axis_score(c.get("trend_score"), default=0)

        # Legacy viral_score is still accepted as a tiebreaker / backfill.
        legacy_viral = c.get("viral_score")
        if legacy_viral is not None:
            legacy_viral = _coerce_axis_score(legacy_viral, default=50)
        else:
            # No explicit viral_score — derive a temporary one from
            # whichever axes were populated. ``finalize_clip_scores``
            # will overwrite this with the genre-weighted composite.
            populated = [s for s in (hook_score, flow_score, value_score, trend_score) if s > 0]
            legacy_viral = int(sum(populated) / len(populated)) if populated else 50

        clip = ClipCandidate(
            id=int(c.get("id", fallback_id)),
            title=str(c.get("title", "Untitled")),
            start_time=start,
            end_time=end,
            duration=round(duration, 1),
            viral_score=legacy_viral,
            viral_score_reasoning=str(c.get("viral_score_reasoning", "")),
            clip_type=str(c.get("clip_type", "highlight")),
            platform=str(c.get("platform", "both")),
            suggested_caption=str(c.get("suggested_caption", "")),
            hook_text=str(c.get("hook_text", "")),
            why_this_works=str(c.get("why_this_works", "")),
            focus_relevance=focus_relevance,
            focus_tier=focus_tier,
            hook_score=hook_score,
            flow_score=flow_score,
            value_score=value_score,
            trend_score=trend_score,
            hook_reason=str(c.get("hook_reason", "")),
            flow_reason=str(c.get("flow_reason", "")),
            value_reason=str(c.get("value_reason", "")),
            trend_reason=str(c.get("trend_reason", "")),
        )

        # Backfill axes from the legacy score if the LLM ignored the
        # 4-axis schema entirely.
        if (hook_score, flow_score, value_score, trend_score) == (0, 0, 0, 0):
            from backend.services.clip_scoring import fill_axes_from_legacy
            fill_axes_from_legacy(clip)

        return clip, None
    except (TypeError, ValueError, KeyError) as err:
        return None, f"malformed: {err}"


# ── VLM upgrade Phase 1 — grounded subject schema + parser ─────────
# The new DEFAULT_SUBJECT_TRACKING_PROMPT asks for a normalized bbox
# instead of a 0-100 percentage estimate. ``SCENE_JSON_SCHEMA_GROUNDED``
# is the schema string we append to every vision request; each
# provider's analyze_frames wires it in once we're ready to flip over.
# ``parse_scene_dict`` is the back-compat-safe parser: if the LLM
# returns the new grounded fields, we use them; if it only returns the
# legacy ``subject_x`` (free-tier models, Ollama, fine-tunes), we
# synthesize a box from the percentage and mark confidence at 0.5.

SCENE_JSON_SCHEMA_GROUNDED = (
    '[{"timestamp": <float>, "description": "<text>", '
    '"importance_score": <1-10>, '
    '"subject_box": [<x1>, <y1>, <x2>, <y2>] or null, '
    '"subject_confidence": <0.0-1.0>, '
    '"secondary_subjects": ['
    '{"box": [<x1>, <y1>, <x2>, <y2>], "confidence": <0.0-1.0>, '
    '"label": "person|speaker|object|text"}], '
    '"no_subject_reason": '
    '"empty_frame|abstract|transition|occluded" or null, '
    '"active_face": <1-based index or 0>}]'
)


def _clamp_box(
    box,
) -> "tuple[list[float] | None, bool]":
    """Coerce and clamp a bbox to [0.0, 1.0] in (x1, y1, x2, y2) order.

    Returns ``(clamped_box, was_modified)``. A return of ``(None, False)``
    means the input was not a usable box (not a 4-element sequence or
    non-numeric). A return of ``(None, True)`` is never produced.

    Auto-fixes:
      - numeric coercion from strings
      - swaps x1/x2 and y1/y2 when out of order
      - clamps each coordinate to [0.0, 1.0]
      - rejects degenerate boxes (zero-area after clamp)
    """
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None, False
    try:
        coords = [float(v) for v in box]
    except (TypeError, ValueError):
        return None, False
    original = tuple(coords)
    x1, y1, x2, y2 = coords
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    x1 = max(0.0, min(1.0, x1))
    y1 = max(0.0, min(1.0, y1))
    x2 = max(0.0, min(1.0, x2))
    y2 = max(0.0, min(1.0, y2))
    # Degenerate (zero-area) after clamp → treat as missing.
    if x2 - x1 < 1e-6 or y2 - y1 < 1e-6:
        return None, True
    modified = tuple([x1, y1, x2, y2]) != original
    return [x1, y1, x2, y2], modified


def _coerce_confidence(raw, default: float = 0.0) -> float:
    """Coerce an LLM confidence field to a float in [0.0, 1.0]."""
    if raw is None:
        return default
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return default
    if v > 1.0 and v <= 100.0:
        # Some models return percentages instead of 0-1. Accept silently.
        v = v / 100.0
    return max(0.0, min(1.0, v))


def fill_box_from_legacy(
    subject_x: int | float,
) -> list[float]:
    """Synthesize a centered vertical strip from a legacy ``subject_x``.

    The legacy prompt asked for a 0-100 horizontal center; downstream
    code treats that as a face/head location. We derive a 10%-wide
    vertical strip centered on the given x and spanning the full frame
    height. The resulting box is deliberately tall — the downstream
    pipeline only uses the x-center for cropping, and Phase 5 crop QA
    re-scores the actual rendered frame so a loose box here has no
    downstream cost.
    """
    try:
        sx_frac = float(subject_x) / 100.0
    except (TypeError, ValueError):
        sx_frac = 0.5
    sx_frac = max(0.0, min(1.0, sx_frac))
    half_width = 0.05  # 10% total
    x1 = max(0.0, sx_frac - half_width)
    x2 = min(1.0, sx_frac + half_width)
    return [x1, 0.0, x2, 1.0]


def _box_center_x(box: list[float]) -> float:
    return (box[0] + box[2]) / 2.0


def _box_center_y(box: list[float]) -> float:
    return (box[1] + box[3]) / 2.0


def parse_scene_dict(d: dict, frame_timestamp: float | None = None) -> dict:
    """Parse one LLM-returned scene dict into a SceneDescription-ready kwargs dict.

    Back-compat contract:
    - When ``d`` contains a valid ``subject_box``, we parse the grounded
      fields and also derive the legacy ``subject_x`` / ``precise_x``
      so every existing pipeline consumer works unchanged.
    - When ``d`` only contains the legacy ``subject_x`` (older / weaker
      / offline models), we synthesize a box from the percentage and
      mark ``vlm_confidence = 0.5`` (legacy uncertainty default).
    - When neither is present we emit the safety-center defaults.

    Returns a dict with keys:
      timestamp, description, importance_score, subject_x, precise_x,
      precise_y, subject_box, vlm_confidence, secondary_subjects,
      no_subject_reason, active_speaker_x.

    The caller is responsible for merging face detection data,
    thumbnail_path, etc. and constructing the SceneDescription.
    """
    out: dict = {}

    # Timestamp (fall back to caller-provided frame timestamp).
    ts = d.get("timestamp")
    try:
        out["timestamp"] = float(ts) if ts is not None else float(frame_timestamp or 0.0)
    except (TypeError, ValueError):
        out["timestamp"] = float(frame_timestamp or 0.0)

    # Description / importance.
    out["description"] = str(d.get("description", "")) or ""
    try:
        imp = int(float(d.get("importance_score", 5)))
    except (TypeError, ValueError):
        imp = 5
    out["importance_score"] = max(1, min(10, imp))

    # Grounded fields.
    raw_box = d.get("subject_box")
    box, box_was_clamped = _clamp_box(raw_box)
    if box_was_clamped:
        _mixin_logger.warning(
            "parse_scene_dict: subject_box %s out of [0,1] bounds — clamped to %s",
            raw_box, box,
        )

    no_reason_raw = d.get("no_subject_reason")
    allowed_reasons = {"empty_frame", "abstract", "transition", "occluded"}
    no_reason = str(no_reason_raw) if no_reason_raw in allowed_reasons else None

    if box is not None:
        # Grounded path — model returned a bbox.
        vlm_conf = _coerce_confidence(
            d.get("subject_confidence"), default=0.0,
        )
        out["subject_box"] = box
        out["vlm_confidence"] = vlm_conf
        out["no_subject_reason"] = None  # box wins over a stale reason

        cx = _box_center_x(box) * 100.0
        cy = _box_center_y(box) * 100.0
        out["precise_x"] = cx
        out["precise_y"] = cy
        out["subject_x"] = int(round(cx))
    elif d.get("subject_x") is not None:
        # Legacy path — older model returned only 0-100 percent.
        raw_sx = d.get("subject_x")
        try:
            sx_val = float(str(raw_sx).strip().rstrip("%"))
            if sx_val > 100:  # some models return pixel coords
                sx_val = (sx_val / 1024.0) * 100.0
            sx_val = max(0.0, min(100.0, sx_val))
        except (TypeError, ValueError):
            sx_val = 50.0
        out["subject_box"] = fill_box_from_legacy(sx_val)
        out["vlm_confidence"] = 0.5  # legacy uncertainty default
        out["precise_x"] = sx_val
        out["precise_y"] = None
        out["subject_x"] = int(round(sx_val))
        out["no_subject_reason"] = no_reason
    else:
        # No usable subject signal at all.
        out["subject_box"] = None
        out["vlm_confidence"] = 0.0
        out["subject_x"] = 50
        out["precise_x"] = None
        out["precise_y"] = None
        out["no_subject_reason"] = no_reason or "empty_frame"

    # Secondary subjects (cap to top 3 by confidence).
    secs_raw = d.get("secondary_subjects") or []
    parsed_secs: list[dict] = []
    if isinstance(secs_raw, list):
        for s in secs_raw:
            if not isinstance(s, dict):
                continue
            sb, _ = _clamp_box(s.get("box"))
            if sb is None:
                continue
            parsed_secs.append({
                "box": sb,
                "confidence": _coerce_confidence(s.get("confidence"), 0.0),
                "label": str(s.get("label", "person"))[:32],
            })
    parsed_secs.sort(key=lambda s: s.get("confidence", 0.0), reverse=True)
    out["secondary_subjects"] = parsed_secs[:3]

    # active_speaker_x (0-100) — unchanged from legacy parser, kept here
    # so every provider gets it for free.
    asx_raw = d.get("active_speaker_x")
    if asx_raw is not None:
        try:
            out["active_speaker_x"] = max(0, min(100, int(float(str(asx_raw).strip()))))
        except (TypeError, ValueError):
            out["active_speaker_x"] = None
    else:
        out["active_speaker_x"] = None

    return out


class ProviderError(Exception):
    pass


class ProviderRateLimitError(ProviderError):
    pass


class AllProvidersFailedError(Exception):
    pass


class AIProvider(ABC):

    @abstractmethod
    async def analyze_frames(
        self,
        frames: list[FrameData],
        custom_prompt: Optional[str] = None,
        cancel_check: Optional[Callable] = None,
        progress_callback: Optional[Callable] = None,
    ) -> list[SceneDescription]:
        """Returns empty list if provider has no vision capability.
        progress_callback(frames_done, total_frames) is called after each batch."""
        pass

    @abstractmethod
    async def generate_summary(
        self,
        transcript: list[TranscriptSegment],
        scenes: list[SceneDescription],
        cancel_check: Optional[Callable] = None,
        custom_prompt: Optional[str] = None,
    ) -> VideoSummary:
        pass

    @abstractmethod
    async def detect_viral_clips(
        self,
        transcript: list[TranscriptSegment],
        scenes: list[SceneDescription],
        video_duration: float,
        custom_prompt: Optional[str] = None,
        cancel_check: Optional[Callable] = None,
        clip_count: Optional[int] = None,
        min_duration: Optional[float] = None,
        max_duration: Optional[float] = None,
        video_summary: Optional[str] = None,
        existing_clips: Optional[str] = None,
        hot_zones=None,
        progress_callback=None,
    ) -> list[ClipCandidate]:
        """All returned clips MUST have duration between min_duration and max_duration seconds."""
        pass

    @abstractmethod
    async def generate_seo(
        self,
        clip_title: str,
        clip_transcript: str,
        video_summary: str,
        platform: str,
        cancel_check: Optional[Callable] = None,
        custom_prompt: Optional[str] = None,
    ) -> ClipSEO:
        """Generate SEO-optimized title, description, and tags for a clip."""
        pass

    async def text_complete(self, prompt: str, max_tokens: int = 4096, timeout: int | None = None) -> str:
        """Generic text completion. Override in subclasses for provider-specific impl."""
        raise NotImplementedError(f"{self.provider_name} does not support text_complete")

    @property
    @abstractmethod
    def supports_vision(self) -> bool:
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        pass

    @property
    def total_tokens(self) -> int:
        """Return total tokens used by this provider instance."""
        return getattr(self, '_total_tokens', 0)

    @property
    def text_model_name(self) -> str:
        """Return the model ID used for text completion.

        Providers override this to return the actual model string
        (e.g. 'google/gemini-2.5-pro') so callers can log which
        model is handling their request and adapt timeouts.
        """
        return f"{self.provider_name}/unknown"

    @property
    def is_thinking_model(self) -> bool:
        """Return True if the text model is a 'thinking' model that needs extended timeouts.

        Thinking models (Gemini 2.5 Flash/Pro, Claude with extended thinking, o1/o3, etc.)
        internally reason before responding, routinely taking 60-120s+ on complex prompts.
        """
        model = self.text_model_name.lower()
        thinking_patterns = [
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "o1",
            "o3",
            "o4-mini",
            "deepseek-r1",
            "qwq",
            "qwen3",        # Qwen 3.x models use extended thinking
            "reka",         # Reka models may use reasoning blocks
        ]
        return any(pattern in model for pattern in thinking_patterns)


def _score_hook_strength(
    clip: ClipCandidate,
    transcript: list[TranscriptSegment],
    audio_moments: "list[dict] | None" = None,
    scenes: "list[SceneDescription] | None" = None,
) -> tuple[int, str]:
    """Score the hook (first 3 seconds) of a clip candidate.

    Returns (score 0-100, reason string).

    Phase 5 (OpusClip parity gap) extends the original transcript-only
    scorer with optional audio-attack and visual-importance signals.
    Both parameters are optional so existing callers keep working
    unchanged. When supplied, they nudge the score:

      * audio attack within first 500ms   → +10
      * sentiment tag (laughter/cheering) → +12
      * high importance scene at t=0      → +8
    """
    hook_start = clip.start_time
    hook_end = hook_start + 3.0

    hook_segs = [
        s for s in transcript
        if s.end > hook_start and s.start < hook_end
    ]

    if not hook_segs:
        # No speech in the first 3s — but a strong audio spike or a
        # visual peak can still pull viewers in (think a music drop or
        # a cinematic establishing shot). Don't auto-floor to 15.
        base = 15
        if audio_moments:
            attack = next(
                (m for m in audio_moments
                 if hook_start <= m.get("timestamp", -1) <= hook_start + 0.5),
                None,
            )
            if attack:
                base += 30
        if scenes:
            opener_scene = next(
                (s for s in scenes
                 if hook_start <= getattr(s, "timestamp", -1) <= hook_start + 1.0),
                None,
            )
            if opener_scene and getattr(opener_scene, "importance_score", 0) >= 8:
                base += 20
        return min(100, base), "dead_air_opening"

    first_seg = min(hook_segs, key=lambda s: s.start)
    first_text = first_seg.text.strip()
    first_words = first_text.split()[:6]
    first_word = first_words[0].lower().rstrip(".,!?") if first_words else ""

    score = 50
    reason_parts: list[str] = []

    # Mid-sentence start penalty
    if first_seg.start < hook_start - 0.5:
        score -= 25
        reason_parts.append("mid_sentence_start")

    # Filler word opener penalty
    filler_words = {"um", "uh", "like", "so", "and", "but", "well", "yeah"}
    if first_word in filler_words:
        score -= 20
        reason_parts.append(f"filler_opener_{first_word}")

    # Context-dependent opener penalty
    context_openers = {"that", "this", "it", "they", "he", "she", "those", "these"}
    if first_word in context_openers and len(first_words) > 1:
        second = first_words[1].lower().rstrip(".,!?")
        if second in {"is", "was", "were", "are", "'s", "thing"}:
            score -= 15
            reason_parts.append("context_dependent_opener")

    # Question opener bonus
    if "?" in " ".join(first_words):
        score += 25
        reason_parts.append("question_hook")

    # Reaction word opener bonus
    reaction_openers = {"wow", "oh", "wait", "no", "what", "holy", "damn", "yo", "bro"}
    if first_word in reaction_openers:
        score += 20
        reason_parts.append("reaction_hook")

    # Speaker change at clip start bonus
    pre_clip_segs = [s for s in transcript if s.end <= hook_start + 0.5 and s.end > hook_start - 2.0]
    if pre_clip_segs:
        prev_speaker = max(pre_clip_segs, key=lambda s: s.end).speaker
        if first_seg.speaker != prev_speaker:
            score += 15
            reason_parts.append("speaker_change_hook")

    # Exclamation / emphasis bonus
    if "!" in first_text:
        score += 10
        reason_parts.append("emphasis_hook")

    # Bold claim detection
    bold_markers = ["never", "always", "the truth", "the real", "actually", "honestly",
                    "people don't", "nobody", "everybody", "the best", "the worst"]
    first_text_lower = first_text.lower()
    if any(m in first_text_lower for m in bold_markers):
        score += 15
        reason_parts.append("bold_claim_hook")

    # ── Phase 5: audio attack within 500ms of t=0 ──
    if audio_moments:
        attack = None
        sentiment_attack = None
        for m in audio_moments:
            ts = m.get("timestamp", -1)
            if hook_start <= ts <= hook_start + 0.5:
                attack = m
            if hook_start <= ts <= hook_start + 1.5:
                sent = m.get("sentiment")
                if sent in {"laughter", "cheering", "shouting", "applause"}:
                    sentiment_attack = sent
                    break
        if attack:
            score += 10
            reason_parts.append("audio_attack_hook")
        if sentiment_attack:
            score += 12
            reason_parts.append(f"sentiment_{sentiment_attack}_hook")

    # ── Phase 5: visual peak in opening frame ──
    if scenes:
        opener_scene = next(
            (s for s in scenes
             if hook_start <= getattr(s, "timestamp", -1) <= hook_start + 1.0),
            None,
        )
        if opener_scene:
            importance = getattr(opener_scene, "importance_score", 0)
            if importance >= 8:
                score += 8
                reason_parts.append("visual_peak_hook")
            elif importance <= 3:
                score -= 8
                reason_parts.append("visual_dead_hook")

    return max(0, min(100, score)), "+".join(reason_parts) if reason_parts else "neutral"


def _find_better_hook(
    clip: ClipCandidate,
    transcript: list[TranscriptSegment],
) -> "float | None":
    """Try to find a better opening within 5 seconds of the original start."""
    search_start = clip.start_time
    search_end = clip.start_time + 5.0

    candidates: list[tuple[float, int]] = []
    for seg in transcript:
        if seg.start < search_start or seg.start > search_end:
            continue
        text = seg.text.strip()
        first_word = text.split()[0].lower().rstrip(".,!?") if text.split() else ""

        if first_word in {"um", "uh", "like", "so", "and", "but", "well"}:
            continue

        score = 0
        if "?" in text[:60]:
            score += 3
        if "!" in text[:60]:
            score += 2
        if first_word in {"wow", "oh", "wait", "no", "what", "holy"}:
            score += 3

        candidates.append((seg.start, score))

    if candidates:
        best = max(candidates, key=lambda c: c[1])
        if best[1] > 0:
            return best[0]
    return None


class ChunkedClipDetectionMixin:
    """Mixin providing multi-pass windowed clip detection for any provider.

    Providers must implement _single_pass_clip_detection() for a single window.
    This mixin handles windowing, gap sweeps, partial results, and merge.

    Supports two modes:
    - concurrent (cloud APIs): processes 2 windows at a time via semaphore
    - sequential (Ollama/local): processes 1 window at a time for VRAM safety
    """

    @staticmethod
    def _deduplicate_clips(clips: list[ClipCandidate], max_overlap: float = 0.5) -> list[ClipCandidate]:
        """Remove clips that overlap or are adjacent covering the same region.

        Only drops a clip when the overlap with an existing HIGHER-SCORED clip
        exceeds max_overlap (default 50%) of the shorter clip's duration.
        """
        if len(clips) <= 1:
            return clips

        # Normalize any inverted timestamps before dedup
        for clip in clips:
            if clip.end_time < clip.start_time:
                clip.start_time, clip.end_time = clip.end_time, clip.start_time
                if clip.duration <= 0:
                    clip.duration = round(clip.end_time - clip.start_time, 1)

        sorted_clips = sorted(clips, key=lambda c: c.viral_score, reverse=True)
        kept: list[ClipCandidate] = []

        for clip in sorted_clips:
            is_duplicate = False
            for existing in kept:
                overlap_start = max(clip.start_time, existing.start_time)
                overlap_end = min(clip.end_time, existing.end_time)
                overlap_duration = max(0, overlap_end - overlap_start)
                shorter_duration = min(clip.duration, existing.duration)

                if shorter_duration > 0 and overlap_duration / shorter_duration > max_overlap:
                    is_duplicate = True
                    _mixin_logger.info(
                        "De-dup: dropping '%s' (%.0f-%.0fs, score=%d) — overlaps %.0f%% with '%s'",
                        clip.title, clip.start_time, clip.end_time, clip.viral_score,
                        (overlap_duration / shorter_duration) * 100, existing.title,
                    )
                    break

                # Proximity check: only treat as duplicate if clips are nearly
                # identical in span (gap < 5s AND combined span barely exceeds
                # single clip). The old gap<10 + ratio<1.3 was too aggressive
                # for short videos where different clips naturally sit close.
                gap = min(
                    abs(clip.start_time - existing.end_time),
                    abs(existing.start_time - clip.end_time),
                )
                combined_span = max(clip.end_time, existing.end_time) - min(clip.start_time, existing.start_time)
                combined_dur = clip.duration + existing.duration
                if gap < 5 and combined_dur > 0 and combined_span / combined_dur < 1.15:
                    is_duplicate = True
                    _mixin_logger.info(
                        "De-dup (proximity): dropping '%s' (%.0f-%.0fs) — adjacent to '%s' (%.0f-%.0fs), gap=%.0fs",
                        clip.title, clip.start_time, clip.end_time,
                        existing.title, existing.start_time, existing.end_time, gap,
                    )
                    break

            if not is_duplicate:
                kept.append(clip)

        if len(kept) < len(clips):
            _mixin_logger.info("De-duplication: kept %d of %d clips", len(kept), len(clips))
        return kept

    @staticmethod
    def _condense_transcript_proportional(
        transcript: list[TranscriptSegment],
        max_chars: int = 12000,
        hot_zones=None,
    ) -> str:
        """Proportionally sample transcript so every time region gets representation.

        Hot zones get 2x weight so the AI gets more detail in high-potential regions.
        Never truncates mid-sentence — each bucket uses complete segments.
        """
        if not transcript:
            return "(no transcript)"

        # Merge consecutive same-speaker segments
        merged: list[tuple[float, float, str, str, float]] = []
        for seg in transcript:
            conf = getattr(seg, 'confidence', None) or 1.0
            if merged and merged[-1][3] == seg.speaker:
                prev = merged[-1]
                avg_conf = (prev[4] + conf) / 2
                merged[-1] = (prev[0], seg.end, prev[2] + " " + seg.text, seg.speaker, avg_conf)
            else:
                merged.append((seg.start, seg.end, seg.text, seg.speaker, conf))

        # Check if everything fits
        full_lines: list[str] = []
        for start, end, text, speaker, conf in merged:
            conf_marker = " [LOW_CONF]" if conf < 0.4 else ""
            full_lines.append(f"[{start:.0f}-{end:.0f}] {speaker}: {text}{conf_marker}")
        full_text = "\n".join(full_lines)
        if len(full_text) <= max_chars:
            return full_text

        # Build 60-second buckets
        if not merged:
            return "(no transcript)"
        video_end = merged[-1][1]
        bucket_size = 60.0
        num_buckets = max(1, int(video_end / bucket_size) + 1)

        buckets: list[list[tuple[float, float, str, str, float]]] = [[] for _ in range(num_buckets)]
        for seg in merged:
            bucket_idx = min(int(seg[0] / bucket_size), num_buckets - 1)
            buckets[bucket_idx].append(seg)

        # Score each bucket: 2x weight if overlapping hot zone
        hot_set: set[int] = set()
        if hot_zones:
            for z in hot_zones:
                z_start = getattr(z, 'start', 0)
                z_end = getattr(z, 'end', 0)
                for bi in range(max(0, int(z_start / bucket_size)), min(num_buckets, int(z_end / bucket_size) + 1)):
                    hot_set.add(bi)

        weights = [2.0 if i in hot_set else 1.0 for i in range(num_buckets)]
        total_weight = sum(weights)
        if total_weight == 0:
            total_weight = 1.0

        # Distribute char budget proportionally
        budget_per_bucket = [(max_chars * w / total_weight) for w in weights]

        # First and last bucket always get full detail
        if num_buckets >= 2:
            budget_per_bucket[0] = max(budget_per_bucket[0], max_chars * 0.08)
            budget_per_bucket[-1] = max(budget_per_bucket[-1], max_chars * 0.05)

        # Build output per bucket
        result_lines: list[str] = []
        for bi, bucket_segs in enumerate(buckets):
            budget = budget_per_bucket[bi]
            if not bucket_segs:
                continue

            bucket_lines: list[str] = []
            used = 0
            for start, end, text, speaker, conf in bucket_segs:
                conf_marker = " [LOW_CONF]" if conf < 0.4 else ""
                line = f"[{start:.0f}-{end:.0f}] {speaker}: {text}{conf_marker}"
                if used + len(line) + 1 <= budget:
                    bucket_lines.append(line)
                    used += len(line) + 1
                elif not bucket_lines:
                    # At least include one truncated line
                    avail = max(50, int(budget))
                    bucket_lines.append(line[:avail] + "...")
                    break
                else:
                    break

            if bucket_lines:
                result_lines.extend(bucket_lines)
            elif bucket_segs:
                # Bucket got no allocation — add a placeholder
                b_start = bucket_segs[0][0]
                b_end = bucket_segs[-1][1]
                result_lines.append(f"[{b_start:.0f}-{b_end:.0f}s] (content condensed)")

        return "\n".join(result_lines)

    @staticmethod
    def _condense_transcript_hot_zone_first(
        transcript: list[TranscriptSegment],
        max_chars: int = 12000,
        hot_zones=None,
    ) -> str:
        """Hot-zone-first condensation: full detail for hot zones, minimal for cold.

        Three tiers:
        - HOT (composite_score > 50): Full transcript lines, uncondensed
        - WARM (composite_score 20-50): Standard proportional condensation
        - COLD (composite_score < 20): Single-line timestamp range placeholder
        """
        if not transcript:
            return "(no transcript)"

        if not hot_zones:
            return ChunkedClipDetectionMixin._condense_transcript_proportional(
                transcript, max_chars=max_chars, hot_zones=hot_zones,
            )

        hot_threshold = 50
        warm_threshold = 20

        video_end = max(s.end for s in transcript)
        tier_map = {}
        hot_zones_sorted = sorted(hot_zones, key=lambda z: z.composite_score, reverse=True)
        for z in hot_zones_sorted:
            tier = "hot" if z.composite_score >= hot_threshold else ("warm" if z.composite_score >= warm_threshold else "cold")
            for t in range(int(z.start), min(int(z.end) + 1, int(video_end) + 1)):
                if t not in tier_map:
                    tier_map[t] = tier

        hot_segs = []
        warm_segs = []
        cold_ranges = []
        cold_start = None

        for seg in transcript:
            seg_tier = tier_map.get(int(seg.start), "cold")
            if seg_tier == "hot":
                if cold_start is not None:
                    cold_ranges.append((cold_start, seg.start))
                    cold_start = None
                hot_segs.append(seg)
            elif seg_tier == "warm":
                if cold_start is not None:
                    cold_ranges.append((cold_start, seg.start))
                    cold_start = None
                warm_segs.append(seg)
            else:
                if cold_start is None:
                    cold_start = seg.start
        if cold_start is not None:
            cold_ranges.append((cold_start, video_end))

        hot_budget = int(max_chars * 0.60)
        warm_budget = int(max_chars * 0.30)
        cold_budget = int(max_chars * 0.10)

        hot_lines = []
        hot_used = 0
        for seg in hot_segs:
            line = f"[{seg.start:.0f}-{seg.end:.0f}] {seg.speaker}: {seg.text}"
            if hot_used + len(line) + 1 <= hot_budget:
                hot_lines.append(line)
                hot_used += len(line) + 1

        warm_lines = []
        warm_used = 0
        for seg in warm_segs:
            text = seg.text[:100] if len(seg.text) > 100 else seg.text
            line = f"[{seg.start:.0f}-{seg.end:.0f}] {seg.speaker}: {text}"
            if warm_used + len(line) + 1 <= warm_budget:
                warm_lines.append(line)
                warm_used += len(line) + 1

        cold_lines = []
        for cs, ce in cold_ranges:
            line = f"[{cs:.0f}-{ce:.0f}s] (low-engagement region — {ce-cs:.0f}s of content condensed)"
            cold_lines.append(line)
        cold_text = "\n".join(cold_lines[:10])
        if len(cold_text) > cold_budget:
            cold_text = cold_text[:cold_budget]

        return "\n".join(hot_lines + warm_lines + [cold_text])

    @staticmethod
    def _stratified_top_n(
        clips: list[ClipCandidate],
        num_clips: int,
        video_duration: float,
    ) -> list[ClipCandidate]:
        """Select top-N clips while guaranteeing timeline coverage.

        A plain score-sort + slice collapses to whichever region the
        model scored highest — on long videos that is almost always the
        opening, silently dropping every late-video candidate.
        Here we bucket candidates across 4-6 time slices of the video,
        take the best clip from each non-empty bucket first, then fill
        remaining slots by global score. Result: the top-N spans the
        full duration instead of clustering in the first quarter.
        """
        if not clips:
            return []
        sorted_by_score = sorted(clips, key=lambda c: c.viral_score, reverse=True)
        if len(clips) <= num_clips or video_duration <= 0 or num_clips <= 1:
            return sorted_by_score[:num_clips]

        if video_duration < 900:
            bucket_count = 4
        elif video_duration < 3600:
            bucket_count = 5
        else:
            bucket_count = 6
        bucket_count = min(bucket_count, num_clips)

        bucket_size = video_duration / bucket_count
        buckets: list[list[ClipCandidate]] = [[] for _ in range(bucket_count)]
        for c in clips:
            idx = min(bucket_count - 1, max(0, int(c.start_time / bucket_size)))
            buckets[idx].append(c)
        for b in buckets:
            b.sort(key=lambda c: c.viral_score, reverse=True)

        selected: list[ClipCandidate] = []
        taken = [0] * bucket_count
        selected_ids: set[int] = set()

        for i, b in enumerate(buckets):
            if b and len(selected) < num_clips:
                selected.append(b[0])
                selected_ids.add(id(b[0]))
                taken[i] = 1

        soft_cap = max(2, num_clips // max(1, bucket_count - 1))
        while len(selected) < num_clips:
            made_progress = False
            for i, b in enumerate(buckets):
                if taken[i] < len(b) and taken[i] < soft_cap and len(selected) < num_clips:
                    pick = b[taken[i]]
                    selected.append(pick)
                    selected_ids.add(id(pick))
                    taken[i] += 1
                    made_progress = True
            if not made_progress:
                break

        if len(selected) < num_clips:
            for c in sorted_by_score:
                if len(selected) >= num_clips:
                    break
                if id(c) not in selected_ids:
                    selected.append(c)
                    selected_ids.add(id(c))

        selected.sort(key=lambda c: c.viral_score, reverse=True)

        _mixin_logger.info(
            "Stratified top-N: kept %d clips across %d time buckets (distribution: %s)",
            len(selected), bucket_count,
            [sum(1 for c in selected if int(c.start_time / bucket_size) == i) for i in range(bucket_count)],
        )
        return selected

    @staticmethod
    def _deduplicate_thematic(clips: list[ClipCandidate], max_similar: int = 2) -> list[ClipCandidate]:
        """Remove thematically duplicate clips that cover the same topic.

        Uses title similarity (Jaccard on words) to detect clips describing
        the same moment from different windows' perspectives.
        Keeps the highest-scoring version of each theme.
        """
        if len(clips) <= 1:
            return clips

        def _word_set(text: str) -> set:
            words = set()
            for w in text.lower().split():
                cleaned = ''.join(c for c in w if c.isalnum())
                if cleaned and len(cleaned) > 2:
                    words.add(cleaned)
            return words

        def _jaccard(a: set, b: set) -> float:
            if not a or not b:
                return 0.0
            return len(a & b) / len(a | b)

        sorted_clips = sorted(clips, key=lambda c: c.viral_score, reverse=True)
        kept = []

        for clip in sorted_clips:
            # Use TITLE ONLY for thematic similarity (hook_text dilutes Jaccard
            # and lets clips with identical titles but different hooks survive)
            clip_title_words = _word_set(clip.title)
            similar_count = 0
            for existing in kept:
                existing_title_words = _word_set(existing.title)
                title_sim = _jaccard(clip_title_words, existing_title_words)
                if title_sim > 0.75:
                    similar_count += 1
            if similar_count < max_similar:
                kept.append(clip)
            else:
                _mixin_logger.info(
                    "Thematic dedup: dropping '%s' (score=%d) — %d similar clips already kept",
                    clip.title, clip.viral_score, similar_count,
                )

        return kept

    @staticmethod
    def _snap_to_speech_boundaries(
        clip: ClipCandidate,
        transcript: list[TranscriptSegment],
        max_adjust: float = 3.0,
    ) -> ClipCandidate:
        """Snap clip start/end to natural speech boundaries using segment timestamps.

        - Start: snap to the beginning of the nearest segment start
        - End: snap to the end of the nearest segment end
        - Never adjust by more than max_adjust seconds
        """
        best_start = clip.start_time
        best_start_dist = float('inf')
        for seg in transcript:
            dist = abs(seg.start - clip.start_time)
            if dist < best_start_dist and dist <= max_adjust:
                if seg.start <= clip.start_time + 0.5:
                    best_start = seg.start
                    best_start_dist = dist

        best_end = clip.end_time
        best_end_dist = float('inf')
        for seg in transcript:
            dist = abs(seg.end - clip.end_time)
            if dist < best_end_dist and dist <= max_adjust:
                if seg.end >= clip.end_time - 0.5:
                    best_end = seg.end
                    best_end_dist = dist

        if best_start != clip.start_time or best_end != clip.end_time:
            clip.start_time = round(best_start, 2)
            clip.end_time = round(best_end, 2)
            clip.duration = round(clip.end_time - clip.start_time, 1)

        return clip

    @staticmethod
    def _score_retention_curve(
        clip: ClipCandidate,
        transcript: list[TranscriptSegment],
        audio_moments: "list[dict] | None" = None,
    ) -> tuple[int, str]:
        """Score a clip's retention curve — how well it sustains engagement.

        Divides the clip into thirds and scores each for energy distribution.
        Returns (score 0-100, curve_shape description).
        """
        dur = clip.end_time - clip.start_time
        if dur < 10:
            return 50, "too_short_to_measure"

        third = dur / 3
        boundaries = [
            (clip.start_time, clip.start_time + third),
            (clip.start_time + third, clip.start_time + 2 * third),
            (clip.start_time + 2 * third, clip.end_time),
        ]

        third_scores = []
        for t_start, t_end in boundaries:
            segs = [s for s in transcript if s.end > t_start and s.start < t_end]

            score = 0
            total_words = sum(len(s.text.split()) for s in segs)
            speech_duration = sum(
                min(s.end, t_end) - max(s.start, t_start) for s in segs
            )
            if speech_duration > 0:
                wps = total_words / speech_duration
                score += min(40, int(wps * 12))

            for seg in segs:
                text = seg.text
                if "!" in text:
                    score += 8
                if "?" in text:
                    score += 6

            speakers = set(s.speaker for s in segs)
            if len(speakers) >= 2:
                score += 10

            if audio_moments:
                moments_in_third = [
                    m for m in audio_moments
                    if t_start <= m.get("timestamp", 0) <= t_end
                ]
                score += len(moments_in_third) * 8

            third_scores.append(min(100, score))

        opening, middle, closing = third_scores

        if middle < 20 and opening > 40:
            curve = "energy_valley"
            overall = int((opening * 0.4 + middle * 0.3 + closing * 0.3) * 0.7)
        elif opening < 25:
            curve = "slow_start"
            overall = int((opening * 0.4 + middle * 0.3 + closing * 0.3) * 0.8)
        elif closing < 20 and opening > 40:
            curve = "fizzle_ending"
            overall = int((opening * 0.4 + middle * 0.3 + closing * 0.3) * 0.75)
        elif opening > 50 and middle > 30 and closing > 40:
            curve = "sustained_energy"
            overall = int(opening * 0.35 + middle * 0.30 + closing * 0.35)
        else:
            curve = "moderate"
            overall = int(opening * 0.35 + middle * 0.30 + closing * 0.35)

        return overall, curve

    async def _windowed_clip_detection(
        self,
        transcript: list[TranscriptSegment],
        scenes: list[SceneDescription],
        video_duration: float,
        window_duration: float = 600.0,
        overlap_duration: float = 120.0,
        _partial_results: Optional[list] = None,
        sequential: bool = False,
        **kwargs,
    ) -> list[ClipCandidate]:
        """Split long videos into overlapping windows and run clip detection on each.

        Args:
            sequential: If True, process windows one at a time (for Ollama/VRAM safety).
                        If False, process up to 2 concurrently.
        """
        import time as _time

        # Build window list
        windows: list[tuple[float, float]] = []
        window_start = 0.0
        step = window_duration - overlap_duration
        if step <= 0:
            step = window_duration
        while window_start < video_duration:
            window_end = min(window_start + window_duration, video_duration)
            windows.append((window_start, window_end))
            window_start += step

        # ── Cap window count for sequential (Ollama) mode ──
        # On CPU inference at 2-4 tok/s, each window takes 2-5 minutes.
        # 46 windows = 3.7 hours of clip detection (observed in production).
        # Scale cap by video length: ≤2hr→8, ≤4hr→10, >4hr→12.
        if video_duration <= 7200:
            MAX_SEQUENTIAL_WINDOWS = 8
        elif video_duration <= 14400:
            MAX_SEQUENTIAL_WINDOWS = 10
        else:
            MAX_SEQUENTIAL_WINDOWS = 12
        if sequential and len(windows) > MAX_SEQUENTIAL_WINDOWS:
            original = len(windows)
            step_size = max(1, len(windows) // MAX_SEQUENTIAL_WINDOWS)
            sampled = [windows[i] for i in range(0, len(windows), step_size)]
            if sampled[-1] != windows[-1]:
                sampled.append(windows[-1])
            windows = sampled[:MAX_SEQUENTIAL_WINDOWS]
            _mixin_logger.info(
                "Capped sequential windows: %d → %d (covering full %.0fs video)",
                original, len(windows), video_duration,
            )

        collected_clips: list[ClipCandidate] = []
        progress_callback = kwargs.get("progress_callback")

        # Emit accurate window count AFTER capping
        if progress_callback:
            try:
                await progress_callback("pass1_start", {"windows": len(windows)})
            except Exception:
                pass

        # Scale per-window clip request so total across windows reaches clip_count
        _requested_total = kwargs.get("clip_count", 12)
        _clips_per_window = max(5, min(12, (_requested_total // len(windows)) + 2))
        _mixin_logger.info(
            "Windowed detection: %d windows, requesting %d clips/window (target total: %d)",
            len(windows), _clips_per_window, _requested_total,
        )

        async def _process_window(idx: int, w_start: float, w_end: float):
            # Only pass kwargs that _single_pass_clip_detection accepts.
            # Strip orchestration-only params to avoid TypeError.
            _pass_through = {
                "custom_prompt", "cancel_check", "clip_count",
                "min_duration", "max_duration", "video_summary",
                "existing_clips", "hot_zones",
            }
            window_kwargs = {k: v for k, v in kwargs.items() if k in _pass_through}
            window_kwargs["clip_count"] = _clips_per_window
            _ws_lo = w_start - overlap_duration / 2
            _we_hi = w_end + overlap_duration / 2
            window_transcript = [
                seg for seg in transcript
                if seg.start < _we_hi and seg.end > _ws_lo
            ]
            _scene_lo = max(0.0, w_start - overlap_duration / 2)
            _scene_hi = w_end + overlap_duration / 2
            window_scenes = [
                s for s in scenes
                if _scene_lo <= s.timestamp <= _scene_hi
            ]

            _mixin_logger.info(
                "Window %d/%d: %.0f-%.0fs (%d segments, %d scenes)",
                idx + 1, len(windows), w_start, w_end,
                len(window_transcript), len(window_scenes),
            )

            # Retry once on transient failure so a single bad window
            # doesn't leave a 2-minute hole in the key-scene coverage.
            # Long videos are windowed to keep each LLM prompt under
            # the context limit — the downside is one flaky window
            # used to produce zero clips, and the clip selector had
            # no way to recover a highlight from that stretch of the
            # timeline. Second attempt runs with a gentler prompt
            # budget so a context-length issue doesn't re-fire.
            clips: list = []
            attempts = 0
            last_err: Optional[BaseException] = None
            while attempts < 2:
                attempts += 1
                try:
                    kwargs_for_try = dict(window_kwargs)
                    if attempts > 1:
                        # Smaller target on retry — reduces the
                        # chance of re-tripping a context / rate
                        # issue while still producing candidates.
                        _kw_target = kwargs_for_try.get("clip_count")
                        if isinstance(_kw_target, int) and _kw_target > 1:
                            kwargs_for_try["clip_count"] = max(
                                1, int(_kw_target // 2)
                            )
                    # Pass the ABSOLUTE video duration and an explicit
                    # window_range so the model understands the transcript
                    # / scene timestamps are global — not window-relative.
                    # Before this change we passed `w_end - w_start` as
                    # video_duration while transcripts still carried
                    # absolute timestamps like `[3600-3620]`, which led
                    # some models to emit window-relative output that
                    # landed at t≈0 after parsing, skewing every
                    # downstream selector toward the video opening.
                    kwargs_for_try["window_range"] = (w_start, w_end)
                    clips = await self._single_pass_clip_detection(
                        window_transcript, window_scenes, video_duration,
                        **kwargs_for_try,
                    )
                    break
                except Exception as e:
                    last_err = e
                    _mixin_logger.warning(
                        "Window %d attempt %d failed: %s",
                        idx + 1, attempts, e,
                    )
                    if attempts >= 2:
                        clips = []
                        break
                    try:
                        await asyncio.sleep(1.5)
                    except asyncio.CancelledError:
                        raise
            if not clips and last_err is not None:
                _mixin_logger.warning(
                    "Window %d/%d produced no clips after %d attempts "
                    "(last error: %s) — moving on",
                    idx + 1, len(windows), attempts, last_err,
                )
            collected_clips.extend(clips)
            if _partial_results is not None:
                _partial_results.extend(clips)
            _mixin_logger.info(
                "Window %d/%d found %d clips (total collected: %d)",
                idx + 1, len(windows), len(clips), len(collected_clips),
            )
            if progress_callback:
                try:
                    await progress_callback("pass1_window_done", {
                        "window_idx": idx + 1,
                        "window_total": len(windows),
                        "clips_so_far": len(collected_clips),
                    })
                except Exception:
                    pass
            return clips

        if sequential:
            # Process one window at a time (Ollama / VRAM safety)
            # Total elapsed guard: don't let clip detection run forever
            max_clip_phase_time = max(1200, video_duration * 2)  # 2x video length, min 20 min
            # Per-window timeout: generous but prevents individual windows from hanging
            per_window_timeout = 300.0  # 5 min per window
            clip_phase_start = _time.monotonic()

            for i, (ws, we) in enumerate(windows):
                cancel_check = kwargs.get("cancel_check")
                if cancel_check:
                    cancel_check()

                # Check total elapsed time
                elapsed = _time.monotonic() - clip_phase_start
                if elapsed > max_clip_phase_time:
                    _mixin_logger.warning(
                        "Clip detection phase exceeded %ds — stopping after %d/%d windows (%d clips)",
                        int(max_clip_phase_time), i, len(windows), len(collected_clips),
                    )
                    break

                # Adaptive per-window timeout: empty/sparse windows bail fast
                _ws_lo_check = ws - overlap_duration / 2
                _we_hi_check = we + overlap_duration / 2
                window_segs = [
                    seg for seg in transcript
                    if seg.start < _we_hi_check and seg.end > _ws_lo_check
                ]
                if len(window_segs) < 3:
                    this_window_timeout = 90.0  # 90s for near-empty windows
                    _mixin_logger.debug("Window %d: sparse (%d segs) — 90s timeout", i + 1, len(window_segs))
                else:
                    this_window_timeout = per_window_timeout

                try:
                    await asyncio.wait_for(
                        _process_window(i, ws, we),
                        timeout=this_window_timeout,
                    )
                except asyncio.TimeoutError:
                    _mixin_logger.warning(
                        "Window %d/%d timed out after %ds — moving to next window",
                        i + 1, len(windows), int(this_window_timeout),
                    )
                    continue
        else:
            # Concurrent with semaphore
            sem = asyncio.Semaphore(2)

            async def _concurrent_window(idx: int, ws: float, we: float):
                async with sem:
                    return await _process_window(idx, ws, we)

            try:
                results = await asyncio.gather(
                    *[_concurrent_window(i, ws, we) for i, (ws, we) in enumerate(windows)],
                    return_exceptions=True,
                )
                failed = sum(1 for r in results if isinstance(r, BaseException) or (isinstance(r, list) and not r))
                if failed == len(windows):
                    _mixin_logger.error("ALL %d windows failed in windowed detection", len(windows))
                elif failed > 0:
                    _mixin_logger.warning(
                        "%d/%d windows failed (%d clips from successful windows)",
                        failed, len(windows), len(collected_clips),
                    )
            except asyncio.CancelledError:
                _mixin_logger.warning(
                    "Windowed detection cancelled — returning %d clips from completed windows",
                    len(collected_clips),
                )

        # Don't dedup here — _multi_pass_clip_detection will dedup all clips
        # (windowed + pass 2) together. Deduping here kills clips that would
        # survive when considered alongside pass 2 results.
        return collected_clips

    async def _multi_pass_clip_detection(
        self,
        transcript: list[TranscriptSegment],
        scenes: list[SceneDescription],
        video_duration: float,
        tier=None,
        sequential: bool = False,
        custom_prompt: Optional[str] = None,
        cancel_check=None,
        clip_count: Optional[int] = None,
        min_duration: Optional[float] = None,
        max_duration: Optional[float] = None,
        video_summary: Optional[str] = None,
        existing_clips: Optional[str] = None,
        hot_zones=None,
        progress_callback=None,
        _partial_results: Optional[list] = None,
    ) -> list[ClipCandidate]:
        """Multi-pass clip detection for comprehensive coverage.

        Pass 1: Windowed detection across the full video
        Pass 2: Coverage sweep on under-represented regions
        Pass 3: Merge, deduplicate, score-sort

        Args:
            tier: VideoDurationTier controlling window sizes and gap limits.
            sequential: Process windows sequentially (for Ollama/local inference).
        """
        from backend.config import settings as _settings

        num_clips = clip_count or (tier.max_clip_candidates if tier else _settings.MAX_CLIP_CANDIDATES)

        # Window sizing — use tier if available, else adaptive
        # Short videos (<5 min): single pass, no windowing needed
        if video_duration <= 300:
            window_dur = video_duration
            overlap_dur = 0.0
            _mixin_logger.info("Short video (%.0fs) — single-pass detection, no windowing", video_duration)
        elif tier and tier.window_duration > 0:
            window_dur = tier.window_duration
            overlap_dur = tier.window_overlap
        elif video_duration > 1800:
            window_dur = 600.0
            overlap_dur = 120.0
        elif video_duration > 600:
            window_dur = 480.0
            overlap_dur = 90.0
        else:
            window_dur = video_duration
            overlap_dur = 0.0

        max_gaps = tier.max_gaps_pass2 if tier else 4

        _mixin_logger.info(
            "Multi-pass clip detection: %.0fs video, window=%.0fs, overlap=%.0fs, sequential=%s",
            video_duration, window_dur, overlap_dur, sequential,
        )

        all_clips: list[ClipCandidate] = []

        # Pass 1: windowed scan (pass1_start callback fires inside _windowed_clip_detection
        # after the window cap is applied, so it reports the accurate count)
        # Pass the full `num_clips` budget so `_windowed_clip_detection`
        # can size per-window quotas correctly. Previously we capped the
        # target at min(8, num_clips//3), which starved long-video
        # windows — a 90-minute run with ~11 windows ended up asking
        # for only ~5 clips per window regardless of tier budget, and
        # stratified selection had too few late-video candidates to
        # choose from.
        pass1_clips = await self._windowed_clip_detection(
            transcript, scenes, video_duration,
            window_duration=window_dur, overlap_duration=overlap_dur,
            _partial_results=_partial_results,
            sequential=sequential,
            custom_prompt=custom_prompt, cancel_check=cancel_check,
            clip_count=num_clips,
            min_duration=min_duration, max_duration=max_duration,
            video_summary=video_summary, existing_clips=existing_clips,
            hot_zones=hot_zones,
            progress_callback=progress_callback,
        )
        all_clips.extend(pass1_clips)
        _mixin_logger.info("Pass 1 found %d clips", len(pass1_clips))

        if progress_callback:
            await progress_callback("pass1_done", {"clips": len(pass1_clips)})

        # ── Early exit: skip Pass 2 if Pass 1 found enough quality clips ──
        # Quality alone is not enough for long videos — if every pass-1
        # clip clusters in the first quarter of the timeline, the late
        # video has zero coverage and no gap-fill ever runs. Require
        # BOTH count AND spread before we trust pass 1 is sufficient.
        high_quality_clips = [c for c in all_clips if c.viral_score >= 60]
        skip_pass2 = len(high_quality_clips) >= num_clips
        if skip_pass2 and high_quality_clips and video_duration > 600:
            hq_starts = [c.start_time for c in high_quality_clips]
            hq_spread = max(hq_starts) - min(hq_starts)
            hq_latest = max(c.end_time for c in high_quality_clips)
            required_spread = video_duration * 0.6
            required_reach = video_duration * 0.75
            if hq_spread < required_spread or hq_latest < required_reach:
                _mixin_logger.info(
                    "Pass 1 HQ clips cluster in %.0f%% of timeline "
                    "(latest end %.0fs / %.0fs total) — running Pass 2 anyway for coverage",
                    (hq_spread / video_duration) * 100 if video_duration > 0 else 0,
                    hq_latest, video_duration,
                )
                skip_pass2 = False
        if skip_pass2:
            _mixin_logger.info(
                "Pass 1 found %d high-quality clips (≥60 score) with broad coverage — skipping Pass 2 gap-fill",
                len(high_quality_clips),
            )

        # Pass 2: Coverage sweep — find regions with no clips.
        # Long videos can overshoot `num_clips` with clustered pass-1
        # output; still run the gap sweep so late regions get covered
        # before stratified top-N selects the final slate.
        _coverage_reach = max((c.end_time for c in all_clips), default=0.0)
        _needs_coverage_sweep = (
            video_duration > 600
            and _coverage_reach < video_duration * 0.75
        )
        if not skip_pass2 and (len(all_clips) < num_clips or _needs_coverage_sweep):
            from backend.services.hot_zone_scorer import get_coverage_gaps
            # If late regions are starving for coverage, ignore the hot-zone
            # filter so we don't drop a 20-minute late gap simply because it
            # lacks a pre-scored zone. Dialogue-light action / cinematic
            # content frequently has no hot zones in late acts.
            _gap_zones = [] if _needs_coverage_sweep else (hot_zones or [])
            gaps = get_coverage_gaps(
                _gap_zones, all_clips, video_duration,
                min_gap_duration=45.0, max_gaps=max_gaps,
            )

            if gaps:
                if progress_callback:
                    await progress_callback("pass2_start", {"gaps": len(gaps)})

                existing_desc = "\n".join(
                    f"  - '{c.title}' ({c.start_time:.0f}-{c.end_time:.0f}s)"
                    for c in all_clips
                )

                gap_sem = asyncio.Semaphore(1 if sequential else 2)

                async def _scan_gap(idx: int, gap_start: float, gap_end: float):
                    async with gap_sem:
                        if cancel_check:
                            cancel_check()
                        gap_duration = gap_end - gap_start
                        gap_transcript = [
                            s for s in transcript
                            if s.start >= gap_start - 15 and s.end <= gap_end + 15
                        ]
                        gap_scenes = [
                            s for s in scenes
                            if gap_start <= s.timestamp <= gap_end
                        ]
                        if not gap_transcript and not gap_scenes:
                            return []

                        if progress_callback:
                            await progress_callback("pass2_gap", {
                                "gap_idx": idx + 1,
                                "gap_total": len(gaps),
                                "start": gap_start,
                                "end": gap_end,
                            })

                        _mixin_logger.info(
                            "Pass 2: scanning gap %.0f-%.0fs (%d segments, %d scenes)",
                            gap_start, gap_end, len(gap_transcript), len(gap_scenes),
                        )

                        try:
                            if gap_duration > 600 and not sequential:
                                return await self._windowed_clip_detection(
                                    gap_transcript, gap_scenes, gap_duration,
                                    window_duration=480.0, overlap_duration=60.0,
                                    sequential=sequential,
                                    custom_prompt=custom_prompt, cancel_check=cancel_check,
                                    clip_count=max(3, num_clips // 2),
                                    min_duration=min_duration, max_duration=max_duration,
                                    video_summary=video_summary, existing_clips=existing_desc,
                                    hot_zones=hot_zones,
                                )
                            else:
                                return await self._single_pass_clip_detection(
                                    gap_transcript, gap_scenes, gap_duration,
                                    custom_prompt=custom_prompt, cancel_check=cancel_check,
                                    clip_count=3,
                                    min_duration=min_duration, max_duration=max_duration,
                                    video_summary=video_summary, existing_clips=existing_desc,
                                    hot_zones=hot_zones,
                                )
                        except Exception as e:
                            _mixin_logger.warning("Pass 2 gap scan failed: %s", e)
                            return []

                if sequential:
                    for i, (gs, ge) in enumerate(gaps):
                        result = await _scan_gap(i, gs, ge)
                        if result:
                            all_clips.extend(result)
                            _mixin_logger.info("Pass 2 found %d clips from gap", len(result))
                else:
                    gap_results = await asyncio.gather(
                        *[_scan_gap(i, gs, ge) for i, (gs, ge) in enumerate(gaps)],
                        return_exceptions=True,
                    )
                    for result in gap_results:
                        if isinstance(result, list):
                            all_clips.extend(result)
                        elif isinstance(result, BaseException):
                            _mixin_logger.warning("Pass 2 gap failed: %s", result)

        # Pass 3: Merge, deduplicate (time overlap + thematic), sort by score
        if progress_callback:
            await progress_callback("pass3_merge", {"raw": len(all_clips)})
        raw_count = len(all_clips)
        # Only drop clips that overlap >70% with a higher-scored clip.
        # The old 60% threshold was too aggressive for short videos where
        # clips from different windows naturally overlap.
        all_clips = self._deduplicate_clips(all_clips, max_overlap=0.7)
        after_overlap = len(all_clips)
        # Scale thematic dedup tolerance with video length.
        # Short videos naturally produce clips about the same topic — allow more.
        _thematic_max = 3 if video_duration < 1800 else 4 if video_duration < 5400 else 5
        all_clips = self._deduplicate_thematic(all_clips, max_similar=_thematic_max)
        after_thematic = len(all_clips)

        # ── Snap to speech boundaries ──
        for i, clip in enumerate(all_clips):
            all_clips[i] = self._snap_to_speech_boundaries(clip, transcript)

        # ── Hook strength validation ──
        # ``hot_zones`` is a list of HotZone objects, not raw audio
        # moments; we don't have direct access to ``audio_moments``
        # at this layer. The Phase 5 audio-attack signals fold in via
        # the hot-zone scoring earlier in the pipeline; here we still
        # get the transcript + scenes signals.
        for clip in all_clips:
            local_hook_score, hook_reason = _score_hook_strength(
                clip, transcript,
                audio_moments=None,
                scenes=scenes,
            )
            # Phase 1 — if the LLM already gave us a hook_score, blend
            # 60/40 toward the local signal when they disagree by >25.
            llm_hook = clip.hook_score
            if llm_hook > 0 and abs(local_hook_score - llm_hook) > 25:
                blended = int(round(local_hook_score * 0.6 + llm_hook * 0.4))
                clip.hook_score = max(0, min(100, blended))
                clip.hook_reason = (
                    f"{clip.hook_reason} | local: {hook_reason}"
                ).strip(" |")
            elif llm_hook == 0:
                # No LLM hook — fall back to the local signal entirely.
                clip.hook_score = local_hook_score
                clip.hook_reason = hook_reason

            hook_adjustment = int((local_hook_score - 50) * 0.2)
            clip.viral_score = max(1, min(100, clip.viral_score + hook_adjustment))
            clip.viral_score_reasoning += f" [Hook: {local_hook_score}/100 ({hook_reason})]"

            if local_hook_score < 30 and transcript:
                better_start = _find_better_hook(clip, transcript)
                if better_start is not None and better_start > clip.start_time:
                    old_start = clip.start_time
                    clip.start_time = better_start
                    clip.duration = round(clip.end_time - clip.start_time, 1)
                    _mixin_logger.info(
                        "Hook fix: '%s' start slid %.1f→%.1fs (hook was %s)",
                        clip.title, old_start, better_start, hook_reason,
                    )

        # Update audio component of hot-zone scoring with sentiment bonuses
        # (handled inline in hot_zone_scorer when sentiment tags are present).

        # ── Retention curve analysis ──
        for clip in all_clips:
            retention_score, curve_shape = self._score_retention_curve(
                clip, transcript, None,
            )
            retention_adj = int((retention_score - 50) * 0.15)
            clip.viral_score = max(1, min(100, clip.viral_score + retention_adj))
            clip.viral_score_reasoning += f" [Retention: {curve_shape}]"

        # ── Platform duration validation ──
        for clip in all_clips:
            dur = clip.duration
            platform = clip.platform.lower() if clip.platform else "both"

            if platform == "tiktok" and dur > 60:
                clip.platform = "youtube_shorts"
                if dur > 90:
                    clip.viral_score_reasoning += " [Re-platformed: too long for TikTok]"
            elif platform == "tiktok" and dur < 15:
                clip.viral_score = max(1, clip.viral_score - 10)
                clip.viral_score_reasoning += " [Warning: very short clip]"
            elif platform == "youtube_shorts" and dur > 180:
                clip.viral_score_reasoning += " [Warning: may exceed Shorts limit]"

        if len(all_clips) > num_clips:
            all_clips = self._stratified_top_n(all_clips, num_clips, video_duration)
        else:
            all_clips.sort(key=lambda c: c.viral_score, reverse=True)

        _mixin_logger.info(
            "Multi-pass complete: %d raw → %d overlap-dedup → %d thematic-dedup "
            "→ %d final (target: %d)",
            raw_count, after_overlap, after_thematic, len(all_clips), num_clips,
        )
        return all_clips
