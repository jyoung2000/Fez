"""Centralized AI prompt management for ClipAI.

Stores default prompts (extracted from the best OpenRouter versions) and
handles loading / saving user customizations from disk.
"""

import json
import logging
import os

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

PROMPTS_FILE = "/data/logs/custom_prompts.json"
MAX_PROMPT_LENGTH = 10_000

# ── Default Prompts ─────────────────────────────────────────────────
# Canonical instruction text shared by all providers.
# JSON output format and strict requirements are appended by each
# provider separately so users can't accidentally break them.

DEFAULT_FRAME_ANALYSIS_PROMPT = (
    "You are analyzing frames from a video to identify key visual moments.\n\n"
    "For each frame, describe:\n"
    "1. What is visually happening (people, actions, setting, on-screen text)\n"
    "2. Whether this is a visually striking or spectacle moment (dramatic visuals, "
    "reactions, reveals, cool effects, beautiful scenery, action sequences)\n"
    "3. Social media potential — would this frame make someone stop scrolling?\n\n"
    "Rate importance from 1-10 where:\n"
    "  1-3 = mundane (talking head, static scene)\n"
    "  4-6 = interesting (good visuals, clear action)\n"
    "  7-8 = compelling (strong emotion, visual spectacle, key reveal)\n"
    "  9-10 = viral-worthy (jaw-dropping moment, perfect reaction, stunning visual)"
)

# ── VLM upgrade Phase 1 — grounding-output prompt ──
# The legacy prompt asked for a 0-100 ``subject_x`` percentage, which
# modern vision models happily hallucinate and then get overridden by
# face detection anyway. The new prompt asks for a normalized bounding
# box with a confidence score — this unlocks the visual-grounding
# capability that Qwen3-VL and Gemini 3 are explicitly trained on,
# feeds Phase 3's confidence-weighted fusion, and gives Phase 5 a
# region to score the final crop against.
DEFAULT_SUBJECT_TRACKING_PROMPT = (
    "For each frame, return a JSON object identifying the primary subject — "
    "the person, character, or object the viewer's eye should follow if the "
    "frame is cropped to a vertical 9:16 window.\n\n"
    "Coordinate system: normalized bounding box [x1, y1, x2, y2] where (0, 0) "
    "is the top-left corner and (1, 1) is the bottom-right corner of the frame.\n\n"
    "Required fields:\n\n"
    "- subject_box: [x1, y1, x2, y2] for the primary subject's FACE (preferred) "
    "or HEAD-AND-SHOULDERS region. Tight box, not the whole body.\n"
    "- subject_confidence: 0.0-1.0. Use 0.9+ when a clearly visible face is "
    "centered and unambiguous; 0.6-0.8 when the face is visible but small, "
    "partially occluded, or one of several; 0.3-0.5 when you're inferring from "
    "body/posture without a clear face; below 0.3 when you're guessing.\n"
    "- secondary_subjects: list of 0-3 objects, each {\"box\": [...], "
    "\"confidence\": 0.0-1.0, \"label\": \"person\" | \"speaker\" | \"object\" | "
    "\"text\"}. Use this when multiple plausible subjects exist "
    "(panel discussions, crowd shots).\n"
    "- no_subject_reason: null when a subject is found, otherwise one of "
    "\"empty_frame\" (no people or focal objects), \"abstract\" (geometric or "
    "non-representational content), \"transition\" (motion blur, fade, dissolve), "
    "\"occluded\" (subject blocked by foreground or HUD).\n\n"
    "When the SPEAKER can be identified (mouth movement, body language, mic "
    "position, gaze of others), put their box in subject_box and tag them as "
    "\"speaker\" if also returning them in secondary_subjects. Otherwise put "
    "the most visually prominent person in subject_box.\n\n"
    "Do NOT return [0, 0, 1, 1] as a \"safe default.\" If you cannot identify "
    "a subject, set subject_box to null and populate no_subject_reason."
)

# Legacy prompt kept around for providers that still use the old 0-100
# subject_x schema (Ollama local fallback and any provider whose backend
# model doesn't support structured grounding output). parse_scene_dict()
# in backend.services.providers.base derives subject_box from subject_x
# when only the legacy field is returned, so the downstream pipeline
# stays uniform regardless of which prompt the provider used.
LEGACY_SUBJECT_TRACKING_PROMPT = (
    "For each frame, estimate where the CENTER of the main subject's FACE is located "
    "horizontally as a percentage from 0 to 100. This value is used to crop the video "
    "to portrait (9:16) by placing a narrow vertical window centered on the subject's face.\n\n"
    "COORDINATE SYSTEM — think of the frame as divided into zones:\n"
    "  0-10  = far left edge of the frame\n"
    "  25    = left quarter\n"
    "  33    = left third\n"
    "  50    = exact horizontal center of the frame\n"
    "  67    = right third\n"
    "  75    = right quarter\n"
    "  90-100 = far right edge of the frame\n\n"
    "CRITICAL: You MUST carefully look at each frame and determine the actual horizontal "
    "position of the subject. Do NOT default to 50 for every frame. Subjects are frequently "
    "positioned off-center — in interview setups, the subject is often at 35-45 or 55-65. "
    "Only return 50 if the subject's face is truly at the exact center of the frame.\n\n"
    "HOW TO ESTIMATE:\n"
    "1. Find the main subject's FACE (not body, not shoulders — the FACE/HEAD)\n"
    "2. Imagine a vertical line through the center of the face\n"
    "3. Estimate where that vertical line falls as a percentage of frame width\n"
    "4. Examples: face in left third = ~33. Face slightly right of center = ~58. Face in right third = ~67.\n\n"
    "PRIORITY:\n"
    "1. If a person's FACE is visible, use the horizontal center of their face\n"
    "2. If multiple people are visible, focus on the SPEAKING person or primary subject\n"
    "3. If no face is visible, use the center of the most important visual element\n\n"
    "Keep values in the 10-90 range to prevent the subject being cut off at frame edges."
)

# ── 4-axis scoring rubric (Phase 1 of OpusClip parity gap) ───────────
# Every genre prompt below shares this base. The LLM must return the
# four axes (hook / flow / value / trend) per clip; the composite
# ``viral_score`` is computed in Python from those axes (see
# clip_scoring.py).
FOUR_AXIS_RUBRIC = (
    "SCORING — return four independent axis scores (0-100) per clip:\n\n"
    "1. hook_score (0-100): Does the FIRST 3 SECONDS pull a scrolling viewer in?\n"
    "   90+ — bold question, jaw-dropping claim, visual spectacle, or a clean\n"
    "         emotional spike (laughter, shouting, gasp) lands inside the first 3s.\n"
    "   60-80 — clear topic introduction with a confident opener and no dead air.\n"
    "   30-50 — generic exposition, mid-thought entry, or weak energy at t=0.\n"
    "   0-25 — dead air, mid-sentence start, filler word opener (\"um\", \"so\", \"and\"),\n"
    "         or context-dependent pronoun opener (\"that was\", \"it's\").\n\n"
    "2. flow_score (0-100): Does the clip stay on ONE topic, ONE scene, ONE exchange,\n"
    "   with a setup → payoff arc?\n"
    "   90+ — single coherent moment with a clear beginning / middle / end.\n"
    "   60-80 — mostly cohesive; one or two minor digressions but the through-line is clear.\n"
    "   30-50 — drifts between sub-topics or cuts across scenes.\n"
    "   0-25 — incoherent: stitches unrelated moments or ends mid-thought.\n\n"
    "3. value_score (0-100): Does the clip RESOLVE — answer a question, reveal\n"
    "   something, land a punchline, or deliver an emotional payoff?\n"
    "   90+ — explicit payoff: punchline, shocking reveal, hot take, satisfying answer.\n"
    "   60-80 — solid takeaway viewer would remember.\n"
    "   ~50 — interesting but unresolved: the clip is engaging but doesn't actually\n"
    "         arrive anywhere.\n"
    "   0-30 — no payoff, filler dialogue, no quotable moment.\n\n"
    "4. trend_score (0-100): How closely does the topic/vibe match current short-form\n"
    "   patterns for the detected genre and platform?\n"
    "   - If a TREND CONTEXT block is provided below, use it. Phrases that overlap\n"
    "     a listed trending topic should score 70+, phrases adjacent to one 50-65,\n"
    "     and unrelated content 30-45.\n"
    "   - When NO trend context is provided, default to 50 (neutral). Do NOT guess.\n\n"
    "Each axis ALSO needs a ONE-SENTENCE reason field explaining the score:\n"
    "  hook_reason, flow_reason, value_reason, trend_reason.\n"
    "Be concrete and specific. Bad: \"good hook\". Good: \"opens with the question\n"
    "'why does nobody talk about this' — strong scroll-stopper\".\n"
)


_VIRAL_BASE_INSTRUCTIONS = (
    "DETECTION METHODOLOGY (follow this two-phase process):\n\n"
    "PHASE 1 — SCAN: Read through the transcript and scene descriptions chronologically. "
    "Identify ALL potential clip-worthy moments. Look for:\n"
    " - Transcript energy spikes (exclamations, questions, rapid exchanges, laughter)\n"
    " - High-importance visual moments (score 7+ in scene descriptions, marked with ★)\n"
    " - Moments where strong dialogue COINCIDES with strong visuals\n"
    " - Natural story arcs: setup → tension → payoff within a contained segment\n"
    " - Speaker changes that mark the start or end of a distinct exchange\n\n"
    "PHASE 2 — SCORE each candidate against the 4-axis rubric below. Do NOT\n"
    "blend the axes into a single number — return them separately and we will\n"
    "compose the final score in code.\n\n"
    "SCENE & SUBJECT COHERENCE (CRITICAL):\n"
    "- The main subject MUST stay in focus throughout the entire clip\n"
    "- NEVER cut across unrelated scenes or topics — the clip must feel like ONE moment\n"
    "- If a clip covers a conversation, keep it within the same exchange\n"
    "- Avoid clips that start on one topic/scene and drift into a completely different one\n"
    "- The visual setting should remain consistent — don't span across location changes\n"
    "- Prefer segments where the camera stays on the main action without jarring cuts\n"
    "- If scene descriptions show different settings at different timestamps, do NOT combine them into one clip\n\n"
    "TITLE RULES:\n"
    "- Write titles as SEO-optimized social media captions about the TOPIC/SUBJECT\n"
    "- NEVER include speaker names, 'Speaker 1', 'Speaker 2', or any speaker labels\n"
    "- NEVER include timestamps, internal IDs, or technical labels like [317-405]\n"
    "- Good: 'The Truth About Celebrity Gossip', 'This Reaction Was Priceless'\n"
    "- Bad: 'Speaker 1 reacts to news', '[317-405] Speaker 1: rapid_exchange'\n\n"
    "BOUNDARY RULES:\n"
    "- Start at natural speech boundaries — beginning of a sentence, after a pause, at a speaker change\n"
    "- End at natural conclusions — punchlines, resolved thoughts, scene transitions\n"
    "- Must work standalone without context from the full video\n"
)


def _build_viral_prompt(role_line: str, genre_block: str = "") -> str:
    """Compose a genre prompt from the shared base + a per-genre block."""
    parts = [role_line.rstrip(), "", _VIRAL_BASE_INSTRUCTIONS, FOUR_AXIS_RUBRIC]
    if genre_block:
        parts.append(genre_block.rstrip())
    return "\n".join(parts)


# Generic / fallback prompt — same role line as the legacy default.
_GENERIC_ROLE = (
    "You are an expert social media video strategist who identifies the most "
    "viral-worthy, attention-grabbing moments in long-form content. Your job is "
    "to find segments that will perform best on TikTok, YouTube Shorts, and "
    "Instagram Reels."
)

DEFAULT_VIRAL_CLIP_PROMPT = _build_viral_prompt(_GENERIC_ROLE)


# ── Genre-specific prompt variants (Phase 2) ────────────────────────
# Each variant adds a genre block that re-tunes what a 90+ score on
# each axis looks like for that content type and lists genre-specific
# examples. The shared 4-axis rubric still applies — the genre block
# just refines it.

VIRAL_PROMPT_TALKING_HEAD = _build_viral_prompt(
    "You are a podcast / interview / vlog clip editor finding the moments most "
    "likely to be reposted as standalone shorts. Focus on quotable hot takes, "
    "clean exchanges, and reaction-worthy answers. Visual spectacle is rare in "
    "this genre — judge clips on what is SAID, not what is shown.",
    genre_block=(
        "GENRE TUNING — TALKING HEAD / PODCAST / INTERVIEW / VLOG:\n"
        "- A 90+ HOOK is a question the audience wants answered or a confident\n"
        "  declarative claim. Mid-question entries are penalised hard.\n"
        "- A 90+ FLOW stays inside a single exchange between speakers — never\n"
        "  glue together two questions that cover different topics.\n"
        "- A 90+ VALUE is a quotable line: \"Most people get this wrong because\n"
        "  ___\", a confessional reveal, or a punchline that lands.\n"
        "- Prefer 30-60s clips that capture one full Q→A or one self-contained\n"
        "  monologue beat. Do not pad to fill duration.\n"
    ),
)

VIRAL_PROMPT_GAMEPLAY = _build_viral_prompt(
    "You are a gameplay highlight editor finding clutch plays, kill streaks, "
    "skill moments, funny deaths, and reaction-worthy commentary. Most viewers "
    "are scrolling — the first second of action decides whether they stop.",
    genre_block=(
        "GENRE TUNING — GAMEPLAY (FPS / MOBA / TPS / RACING):\n"
        "- A 90+ HOOK opens on a moment of high stakes or a kinetic action beat:\n"
        "  the start of a teamfight, a clutch round, an enemy contact, or a\n"
        "  funny fail moment. Static menu / loadout screens are dead air — score 0-25.\n"
        "- A 90+ FLOW is a single play that resolves in a clear win or loss.\n"
        "  Avoid stitching plays from different rounds together.\n"
        "- A 90+ VALUE has a clear payoff: the kill, the clutch, the joke. The\n"
        "  commentary reaction (\"NO WAY\", \"OH MY GOD\") is a reliable payoff signal.\n"
        "- De-emphasise long stretches of dialogue between fights — those are\n"
        "  filler in this genre.\n"
        "- Prefer 15-45s clips. Anything longer than 60s loses scrollers.\n"
    ),
)

VIRAL_PROMPT_SPORTS = _build_viral_prompt(
    "You are a sports highlight editor finding scoring plays, close calls, "
    "crowd reactions, and dramatic moments. Score boundaries by play "
    "completion, not sentence boundaries — the cheering after the play is part "
    "of the clip.",
    genre_block=(
        "GENRE TUNING — SPORTS:\n"
        "- A 90+ HOOK opens a few seconds before the decisive moment so viewers\n"
        "  feel the build-up. Cold opens on a celebration are weaker (40-60).\n"
        "- A 90+ FLOW is one continuous play from setup to result, ending after\n"
        "  the crowd reaction. Cutting before the cheer kills the payoff.\n"
        "- A 90+ VALUE is the play itself + the reaction (crowd, commentary,\n"
        "  player). A scoring play with no reaction shot is ~70.\n"
        "- Boundary rule: snap to play start / whistle, not to mid-sentence\n"
        "  commentary. The commentator can still be mid-word at the start.\n"
        "- Prefer 15-40s clips.\n"
    ),
)

VIRAL_PROMPT_MUSIC_VIDEO = _build_viral_prompt(
    "You are a music video editor finding beat drops, chorus moments, and "
    "iconic visual motifs. The audio waveform is the primary signal — find the "
    "moments where the music peaks and the visuals support it.",
    genre_block=(
        "GENRE TUNING — MUSIC VIDEO:\n"
        "- A 90+ HOOK is a beat drop, vocal entry, or a striking visual motif\n"
        "  in the first 1-2 seconds. Long instrumental intros without a payoff\n"
        "  are weaker (30-50).\n"
        "- A 90+ FLOW is bar-aligned: starts on a downbeat, ends on a phrase\n"
        "  resolution. Do NOT cut mid-bar.\n"
        "- A 90+ VALUE is the chorus or the most memorable visual sequence —\n"
        "  the part viewers would loop or duet with.\n"
        "- Boundary rule: snap to beat boundaries. Even a 0.3s offset feels wrong.\n"
        "- Trend axis matters more than usual here — match against what is\n"
        "  currently going viral on TikTok sound.\n"
        "- Prefer 15-30s clips.\n"
    ),
)

VIRAL_PROMPT_ANIMATION = _build_viral_prompt(
    "You are an animation / anime clip editor finding reaction shots, "
    "punchline frames, action peaks, and emotional beats. Cuts must respect "
    "shot boundaries — never combine two scenes that show different characters "
    "in different settings.",
    genre_block=(
        "GENRE TUNING — ANIMATION / ANIME / CARTOON:\n"
        "- A 90+ HOOK is a striking pose, a sudden expression, or a sharp\n"
        "  motion beat in the first second. Static establishing shots are weak\n"
        "  (30-45).\n"
        "- A 90+ FLOW is one scene with one character set. Hard cut to a new\n"
        "  scene = drop flow to 30 or below.\n"
        "- A 90+ VALUE is a punchline frame, a reveal expression, an action peak,\n"
        "  or an emotional climax. Talking heads with no expression change ~50.\n"
        "- Character consistency is mandatory: do not glue together two scenes\n"
        "  with different protagonists.\n"
        "- Prefer 15-40s clips.\n"
    ),
)

VIRAL_PROMPT_NARRATIVE = _build_viral_prompt(
    "You are a narrative film / TV clip editor finding cinematic dialogue, "
    "reveal moments, and emotional beats. Respect shot boundaries hard — a "
    "clip must live inside a single scene.",
    genre_block=(
        "GENRE TUNING — NARRATIVE / CINEMATIC DIALOGUE:\n"
        "- A 90+ HOOK opens on a clean line delivery or a striking visual.\n"
        "  Mid-line entries lose 25 points.\n"
        "- A 90+ FLOW lives entirely inside one scene with one set of\n"
        "  characters. Scene cuts inside the clip = drop flow hard.\n"
        "- A 90+ VALUE is a reveal, a confession, a punchline, or a line that\n"
        "  hits hard out of context.\n"
        "- Boundary rule: snap to shot transitions, not to mid-line audio.\n"
        "- Prefer 20-50s clips so the beat has room to breathe.\n"
    ),
)

VIRAL_PROMPT_GENERIC = DEFAULT_VIRAL_CLIP_PROMPT


def get_genre_prompt(content_type) -> str:
    """Return the right viral-clip prompt variant for a content type.

    Accepts a ``ClipContentType`` enum value (or anything with a
    ``.value`` attribute / a string). Always returns a non-empty
    string — falls back to ``DEFAULT_VIRAL_CLIP_PROMPT`` for unknown
    types so the caller never has to null-check.
    """
    # Defer the import so this module stays cheap to import. The
    # content_classifier module is several hundred lines of heuristics
    # we do not need just to look up an enum.
    try:
        from backend.services.content_classifier import ClipContentType
    except Exception:
        return DEFAULT_VIRAL_CLIP_PROMPT

    if content_type is None:
        return DEFAULT_VIRAL_CLIP_PROMPT

    if hasattr(content_type, "value"):
        key = content_type.value
    else:
        key = str(content_type)

    mapping = {
        ClipContentType.TALKING_HEAD.value: VIRAL_PROMPT_TALKING_HEAD,
        ClipContentType.MULTI_SPEAKER_PANEL.value: VIRAL_PROMPT_TALKING_HEAD,
        ClipContentType.GAMEPLAY.value: VIRAL_PROMPT_GAMEPLAY,
        ClipContentType.GAMEPLAY_MOBA.value: VIRAL_PROMPT_GAMEPLAY,
        ClipContentType.GAMEPLAY_TPS.value: VIRAL_PROMPT_GAMEPLAY,
        ClipContentType.GAMEPLAY_RACING.value: VIRAL_PROMPT_GAMEPLAY,
        ClipContentType.STREAM.value: VIRAL_PROMPT_GAMEPLAY,
        ClipContentType.SPORTS.value: VIRAL_PROMPT_SPORTS,
        ClipContentType.SPORTS_BASKETBALL.value: VIRAL_PROMPT_SPORTS,
        ClipContentType.SPORTS_RACING.value: VIRAL_PROMPT_SPORTS,
        ClipContentType.MUSIC_VIDEO.value: VIRAL_PROMPT_MUSIC_VIDEO,
        ClipContentType.ANIMATION.value: VIRAL_PROMPT_ANIMATION,
        ClipContentType.ANIMATION_DIALOGUE.value: VIRAL_PROMPT_ANIMATION,
        ClipContentType.CINEMATIC_DIALOGUE.value: VIRAL_PROMPT_NARRATIVE,
        ClipContentType.GENERIC.value: VIRAL_PROMPT_GENERIC,
    }
    return mapping.get(key, VIRAL_PROMPT_GENERIC)


DEFAULT_SUMMARY_PROMPT = (
    "You are summarizing a video for a human audience. Write like a real person — "
    "not a robot, not a press release. The summary should feel like something a "
    "friend would say if you asked 'what was that video about?'\n\n"
    "IMPORTANT: Do not reference or speculate about speakers unless speaker names are "
    "explicitly present in the transcript. Focus on WHAT is discussed and shown, "
    "not who is saying it.\n\n"
    "Based on the transcript and scene descriptions below, return ONLY valid JSON:\n"
    '{"overview": "<2-4 sentence paragraph>", "key_topics": ["topic1", "topic2", "topic3"], '
    '"tone": "<1-2 words>", "estimated_audience": "<who would watch>", "content_category": "<category>"}\n\n'
    "Field guidelines:\n"
    "- overview: Natural, conversational. Describe what happens in the video.\n"
    '   Good: "Two friends taste-test fast food burgers and debate whether In-N-Out is overrated."\n'
    '   Bad: "This video features content creators engaging in comparative analysis."\n'
    "- key_topics: 3-6 specific topics. Use natural phrases, not SEO keywords.\n"
    '   Good: ["fast food taste test", "In-N-Out vs Five Guys", "sauce disaster"]\n'
    '   Bad: ["food", "review", "content"]\n'
    '- tone: The vibe (e.g. "funny and casual", "serious", "educational")\n'
    '- estimated_audience: Be specific (e.g. "foodies and fast food fans")\n'
    '- content_category: Specific (e.g. "food review", "tech unboxing", "comedy sketch")\n'
)

# ── VLM upgrade Phase 5 — editorial crop QA prompt ────────────────
# Used by backend.services.crop_qa.score_crop_quality to re-score
# each rendered segment's output frames. The VLM's answer is parsed
# as JSON and fed into decide_recovery() to decide whether to
# re-solve the crop with looser constraints or fall through to a
# safety-center crop. Gated behind CLIPAI_CROP_QA; see
# docs/vlm_upgrade/PHASE_5_NOTES.md.
DEFAULT_CROP_QA_PROMPT = (
    "You are reviewing a vertical 9:16 crop of a horizontal source video. "
    "Look at this output frame and answer:\n\n"
    "- head_in_frame: bool — is the main subject's head fully inside the "
    "frame (not cut off at top/bottom/sides)?\n"
    "- awkward_crop: bool — is there an awkward edge cut (hand chopped "
    "mid-gesture, half a face, body cut at the neck)?\n"
    "- subject_partially_off_frame: bool — is the subject visible but "
    "partially clipped at a frame edge?\n"
    "- dead_space_dominant: bool — is more than 50% of the frame empty/"
    "non-subject space?\n"
    "- quality_score: 0-10. 10 = perfect editorial framing (subject "
    "well-placed with appropriate headroom, no awkward cuts). 7+ = "
    "acceptable. 5-6 = noticeable problems but watchable. <5 = "
    "unwatchable, must re-frame.\n\n"
    "Return JSON only."
)


DEFAULT_SEO_PROMPT = (
    "You write social media captions and tags like a real person — not a marketer, "
    "not a robot. Think of how popular creators actually post on YouTube, TikTok, "
    "Instagram, and Tumblr. The text should feel natural and authentic.\n\n"
    "Given a video clip, generate:\n\n"
    "1. TITLE — Write it like a real post title. Keep it under 100 characters. "
    "It should sound like something a person would actually type, not an ad. "
    "Use lowercase naturally. No clickbait, no ALL CAPS spam, no excessive punctuation.\n"
    "   Good: \"when the beat dropped and nobody was ready\"\n"
    "   Good: \"This changed how I think about cooking\"\n"
    "   Bad: \"YOU WON'T BELIEVE What Happens Next!!!\"\n\n"
    "2. DESCRIPTION — Write 1-3 casual sentences like a real caption someone would "
    "post. Can include personality, humor, or a brief thought. Keep it 100-250 chars. "
    "Don't stuff keywords or write like a press release.\n\n"
    "3. TAGS — 8-15 hashtags (with # prefix) that a real person would actually use. "
    "Mix popular broad tags with specific niche ones. Use lowercase. "
    "These are the tags people search and browse on social platforms.\n"
    "   Example: [\"#cooking\", \"#foodtok\", \"#recipe\", \"#homemade\", \"#fyp\"]\n\n"
    "4. PLATFORM_TIPS — One short sentence of posting advice for this specific clip.\n\n"
    "Return ONLY valid JSON:\n"
    '{"title": "...", "description": "...", "tags": ["#tag1", "#tag2", ...], '
    '"platform_tips": "..."}'
)


class PromptSet(BaseModel):
    frame_analysis: str = Field(default=DEFAULT_FRAME_ANALYSIS_PROMPT)
    viral_clip_detection: str = Field(default=DEFAULT_VIRAL_CLIP_PROMPT)
    subject_tracking: str = Field(default=DEFAULT_SUBJECT_TRACKING_PROMPT)
    summary: str = Field(default=DEFAULT_SUMMARY_PROMPT)
    seo: str = Field(default=DEFAULT_SEO_PROMPT)


def load_prompts() -> PromptSet:
    """Load custom prompts from disk, falling back to defaults."""
    if os.path.exists(PROMPTS_FILE):
        try:
            with open(PROMPTS_FILE, "r") as f:
                data = json.load(f)
            return PromptSet(**data)
        except Exception as e:
            logger.warning(f"Failed to load custom prompts: {e}")
    return PromptSet()


def save_prompts(prompts: PromptSet) -> None:
    """Persist custom prompts to disk."""
    os.makedirs(os.path.dirname(PROMPTS_FILE), exist_ok=True)
    with open(PROMPTS_FILE, "w") as f:
        json.dump(prompts.model_dump(), f, indent=2)


def get_defaults() -> PromptSet:
    """Return the hardcoded default prompts (for reset functionality)."""
    return PromptSet(
        frame_analysis=DEFAULT_FRAME_ANALYSIS_PROMPT,
        viral_clip_detection=DEFAULT_VIRAL_CLIP_PROMPT,
        subject_tracking=DEFAULT_SUBJECT_TRACKING_PROMPT,
        summary=DEFAULT_SUMMARY_PROMPT,
        seo=DEFAULT_SEO_PROMPT,
    )
