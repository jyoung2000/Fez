"""Test that the embedding-based face registry rejects slots with
x-span >60% over >30 frames as cross-shot merges, and falls back to
position-based clustering when >25% of slots fail this check.

Bug 3 from the Bungo Stray Dogs run: embedding registry produced 28
slots where many spanned 1-99% of frame width — obviously cross-shot
merges, not real identities. The position-based registry was correct
(4 speakers) but was discarded by the "max wins" rule.
"""

from dataclasses import dataclass, field

import numpy as np

from backend.services.face_registry import build_face_registry_with_embeddings


@dataclass
class _Face:
    nose_x: float
    nose_y: float = 50.0
    width: float = 10.0
    height: float = 12.0
    confidence: float = 0.9
    lip_aperture: float = 0.0
    identity_id: int = -1
    is_human: bool = True
    identity_embedding: list = None


@dataclass
class _FrameFaces:
    timestamp: float
    faces: list = field(default_factory=list)


def _make_similar_embedding(seed: int, noise: float = 0.01) -> list:
    """Generate an embedding that will look very similar to others with
    the same seed — simulates the anime-style face embedding failure
    where visually similar drawings produce near-identical embeddings
    across unrelated characters/shots."""
    rng = np.random.RandomState(seed)
    base = rng.randn(128).astype(np.float32)
    base = base / (np.linalg.norm(base) + 1e-8)
    # Add tiny noise so they're not identical but still within cosine
    # threshold (< 0.25)
    noise_rng = np.random.RandomState(seed + 999)
    base = base + noise_rng.randn(128).astype(np.float32) * noise
    base = base / (np.linalg.norm(base) + 1e-8)
    return base.tolist()


def test_wide_span_embedding_slot_rejected_and_falls_back():
    """Construct 40 frames where the embedding clustering would merge
    faces at x=1..99 into a single slot (because embeddings are nearly
    identical). Assert the wide-span sanity check fires and the
    position-based registry (4 clusters) replaces it.
    """
    # Shared embedding seed → all faces cluster as one identity by
    # embedding similarity.
    shared_emb_seed = 7

    frames = []
    # Frames 0-9: face at x=10 (shot 1, character A)
    for t in range(10):
        frames.append(_FrameFaces(
            timestamp=float(t),
            faces=[_Face(
                nose_x=10.0,
                identity_embedding=_make_similar_embedding(shared_emb_seed),
            )],
        ))
    # Frames 10-19: face at x=35 (shot 2, character A at different spot)
    for t in range(10, 20):
        frames.append(_FrameFaces(
            timestamp=float(t),
            faces=[_Face(
                nose_x=35.0,
                identity_embedding=_make_similar_embedding(shared_emb_seed),
            )],
        ))
    # Frames 20-29: face at x=65
    for t in range(20, 30):
        frames.append(_FrameFaces(
            timestamp=float(t),
            faces=[_Face(
                nose_x=65.0,
                identity_embedding=_make_similar_embedding(shared_emb_seed),
            )],
        ))
    # Frames 30-39: face at x=95
    for t in range(30, 40):
        frames.append(_FrameFaces(
            timestamp=float(t),
            faces=[_Face(
                nose_x=95.0,
                identity_embedding=_make_similar_embedding(shared_emb_seed),
            )],
        ))

    registry = build_face_registry_with_embeddings(
        frames, min_appearances=3, cosine_threshold=0.25,
    )

    # After the sanity check + position-based fallback, we should have
    # multiple distinct position-based slots rather than one giant slot
    # spanning 1-99%. We don't require an exact count (position-based
    # may cluster them differently) but no single slot may span the
    # whole frame.
    for s in registry.slots:
        span = s.x_max - s.x_min
        assert not (span > 60 and s.frame_count > 30), (
            f"Slot {s.slot_id} still has span={span:.0f}% across "
            f"{s.frame_count} frames — the wide-span sanity check did "
            f"not fire"
        )
