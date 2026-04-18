"""Week 2 Part A — unit tests for the dense-face anime augmentation.

Exercises ``_augment_dense_with_anime`` in isolation by mocking
``detect_anime_faces`` and feeding canned ``FrameFaces`` / frame-path
lists. The helper is the exact code path ``detect_faces_dense`` runs
when ``is_animated=True``, so these tests pin its contract without
needing ffmpeg / cv2 / a real video.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from unittest.mock import patch

import pytest


# Shared stubs. ``FrameFaces`` from the real module pulls in an
# ``identity_embedding`` field that requires numpy to round-trip. The
# test helpers only touch ``.faces`` (a mutable list), ``.timestamp``,
# and ``.frame_path``, so importing the real class is fine — we just
# don't populate the embedding.


@dataclass
class _StubFace:
    """Minimal FaceInfo-shaped stub for tests that don't need the full dataclass."""
    confidence: float = 0.0
    is_human: bool = True
    x_center: float = 50.0
    y_center: float = 50.0
    width: float = 10.0
    height: float = 12.0


@dataclass
class _StubFrameFaces:
    timestamp: float
    frame_path: str
    faces: list = field(default_factory=list)


@dataclass
class _StubAnimeDetection:
    x_center: float
    y_center: float
    width: float
    height: float
    confidence: float
    source: str = "lbpcascade_animeface"


@dataclass
class _StubAnimeResult:
    timestamp: float
    detections: list = field(default_factory=list)
    skipped_reason: str = ""

    @property
    def has_faces(self) -> bool:
        return bool(self.detections)


def _make_results_and_paths() -> tuple[list, list]:
    """Build three FrameFaces: weak, strong, empty."""
    results = [
        _StubFrameFaces(
            timestamp=0.0,
            frame_path="/tmp/frame_000.jpg",
            faces=[_StubFace(confidence=0.30)],  # weak — augment
        ),
        _StubFrameFaces(
            timestamp=0.5,
            frame_path="/tmp/frame_001.jpg",
            faces=[_StubFace(confidence=0.80)],  # strong — skip
        ),
        _StubFrameFaces(
            timestamp=1.0,
            frame_path="/tmp/frame_002.jpg",
            faces=[],                              # empty — augment
        ),
    ]
    frame_paths = [
        (fr.timestamp, fr.frame_path) for fr in results
    ]
    return results, frame_paths


def _fake_detect(frame_path: str, *, timestamp: float = 0.0):
    """Fake detect_anime_faces — returns two detections for every frame."""
    return _StubAnimeResult(
        timestamp=timestamp,
        detections=[
            _StubAnimeDetection(
                x_center=30.0, y_center=40.0,
                width=12.0, height=15.0, confidence=0.85,
            ),
            _StubAnimeDetection(
                x_center=70.0, y_center=55.0,
                width=10.0, height=13.0, confidence=0.90,
            ),
        ],
    )


def _fake_to_face_info(det):
    """Fake to_face_info — returns an object that walks like FaceInfo."""
    return _StubFace(
        confidence=float(det.confidence),
        is_human=False,     # Critical: anime detections must not trip the human verifier
        x_center=float(det.x_center),
        y_center=float(det.y_center),
        width=float(det.width),
        height=float(det.height),
    )


# ─────────────────────────────────────────────────────────────────────


def test_augment_appends_to_weak_and_empty_frames():
    """Augmentation touches frames 0 (weak) and 2 (empty), skips frame 1 (strong)."""
    from backend.services import face_detector

    results, frame_paths = _make_results_and_paths()

    with patch.object(
        face_detector, "_augment_dense_with_anime",
        wraps=face_detector._augment_dense_with_anime,
    ), patch(
        "backend.services.anime_face_detector.detect_anime_faces",
        side_effect=_fake_detect,
    ), patch(
        "backend.services.anime_face_detector.to_face_info",
        side_effect=_fake_to_face_info,
    ), patch(
        "backend.services.anime_face_detector.USE_ANIME_FACE_DETECTOR",
        True,
    ):
        n_frames, n_faces = face_detector._augment_dense_with_anime(
            results, frame_paths,
        )

    assert n_frames == 2, "weak + empty frames should be augmented"
    assert n_faces == 4, "two detections per augmented frame"

    # Frame 0 originally had one weak face; now has 1 weak + 2 anime = 3
    assert len(results[0].faces) == 3
    # Frame 1 was strong — untouched
    assert len(results[1].faces) == 1
    assert results[1].faces[0].confidence == 0.80
    # Frame 2 was empty; now has 2 anime faces
    assert len(results[2].faces) == 2

    # Every anime-origin face must be marked non-human so the pose
    # verifier downstream skips it.
    for anime_face in results[0].faces[1:]:  # all but the original weak face
        assert anime_face.is_human is False
    for anime_face in results[2].faces:
        assert anime_face.is_human is False


def test_flag_off_short_circuits():
    """USE_ANIME_FACE_DETECTOR=False → helper returns (0,0) without calling detector."""
    from backend.services import face_detector

    results, frame_paths = _make_results_and_paths()

    # Capture calls to detect_anime_faces so we can verify it's never hit.
    call_count = {"n": 0}

    def _spy(*args, **kwargs):
        call_count["n"] += 1
        return _StubAnimeResult(timestamp=0.0)

    with patch(
        "backend.services.anime_face_detector.USE_ANIME_FACE_DETECTOR",
        False,
    ), patch(
        "backend.services.anime_face_detector.detect_anime_faces",
        side_effect=_spy,
    ):
        n_frames, n_faces = face_detector._augment_dense_with_anime(
            results, frame_paths,
        )

    assert (n_frames, n_faces) == (0, 0)
    assert call_count["n"] == 0, "detector must not run when flag is OFF"
    # Results are untouched
    assert len(results[0].faces) == 1
    assert len(results[1].faces) == 1
    assert len(results[2].faces) == 0


def test_strong_conf_threshold_respected():
    """A face at exactly the threshold (0.55) counts as strong and skips augmentation."""
    from backend.services import face_detector

    results = [
        _StubFrameFaces(
            timestamp=0.0,
            frame_path="/tmp/f.jpg",
            faces=[_StubFace(confidence=0.55)],  # exactly at threshold
        ),
        _StubFrameFaces(
            timestamp=0.5,
            frame_path="/tmp/g.jpg",
            faces=[_StubFace(confidence=0.54)],  # just below
        ),
    ]
    frame_paths = [(r.timestamp, r.frame_path) for r in results]

    with patch(
        "backend.services.anime_face_detector.detect_anime_faces",
        side_effect=_fake_detect,
    ), patch(
        "backend.services.anime_face_detector.to_face_info",
        side_effect=_fake_to_face_info,
    ), patch(
        "backend.services.anime_face_detector.USE_ANIME_FACE_DETECTOR",
        True,
    ):
        n_frames, _ = face_detector._augment_dense_with_anime(
            results, frame_paths,
        )

    assert n_frames == 1, "only the sub-threshold frame gets augmented"
    # The 0.55 frame stayed at 1 face; the 0.54 frame got +2 anime.
    assert len(results[0].faces) == 1
    assert len(results[1].faces) == 3


def test_mismatched_frame_paths_are_safe():
    """Helper doesn't crash when frame_paths is shorter than results."""
    from backend.services import face_detector

    results = [
        _StubFrameFaces(timestamp=0.0, frame_path="/tmp/a.jpg", faces=[]),
        _StubFrameFaces(timestamp=0.5, frame_path="/tmp/b.jpg", faces=[]),
    ]
    frame_paths = [(0.0, "/tmp/a.jpg")]  # only one entry — second should be skipped

    with patch(
        "backend.services.anime_face_detector.detect_anime_faces",
        side_effect=_fake_detect,
    ), patch(
        "backend.services.anime_face_detector.to_face_info",
        side_effect=_fake_to_face_info,
    ), patch(
        "backend.services.anime_face_detector.USE_ANIME_FACE_DETECTOR",
        True,
    ):
        n_frames, n_faces = face_detector._augment_dense_with_anime(
            results, frame_paths,
        )

    assert n_frames == 1, "only index 0 was augmentable; index 1 out of range"
    assert n_faces == 2


def test_no_detections_returned_leaves_results_unchanged():
    """When detect_anime_faces returns no detections, the frame isn't touched."""
    from backend.services import face_detector

    results = [
        _StubFrameFaces(timestamp=0.0, frame_path="/tmp/a.jpg", faces=[]),
    ]
    frame_paths = [(0.0, "/tmp/a.jpg")]

    def _empty(frame_path, *, timestamp=0.0):
        return _StubAnimeResult(timestamp=timestamp, detections=[])

    with patch(
        "backend.services.anime_face_detector.detect_anime_faces",
        side_effect=_empty,
    ), patch(
        "backend.services.anime_face_detector.to_face_info",
        side_effect=_fake_to_face_info,
    ), patch(
        "backend.services.anime_face_detector.USE_ANIME_FACE_DETECTOR",
        True,
    ):
        n_frames, n_faces = face_detector._augment_dense_with_anime(
            results, frame_paths,
        )

    assert (n_frames, n_faces) == (0, 0)
    assert len(results[0].faces) == 0


def test_to_face_info_exception_is_swallowed():
    """A single conversion failure shouldn't kill the whole augmentation pass."""
    from backend.services import face_detector

    results = [
        _StubFrameFaces(timestamp=0.0, frame_path="/tmp/a.jpg", faces=[]),
    ]
    frame_paths = [(0.0, "/tmp/a.jpg")]

    calls = {"n": 0}

    def _flaky(det):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("simulated converter failure")
        return _fake_to_face_info(det)

    with patch(
        "backend.services.anime_face_detector.detect_anime_faces",
        side_effect=_fake_detect,
    ), patch(
        "backend.services.anime_face_detector.to_face_info",
        side_effect=_flaky,
    ), patch(
        "backend.services.anime_face_detector.USE_ANIME_FACE_DETECTOR",
        True,
    ):
        n_frames, n_faces = face_detector._augment_dense_with_anime(
            results, frame_paths,
        )

    # First detection was dropped but the second succeeded.
    assert n_frames == 1
    assert n_faces == 2  # count reflects attempted faces, not survivors
    assert len(results[0].faces) == 1  # only the successful conversion


# ─────────────────────────────────────────────────────────────────────
# Phase 2 — anime-priority cascade tests


def test_anime_priority_runs_cascade_on_strong_yunet_frames():
    """With ``anime_priority=True`` the cascade runs on every frame,
    even when YuNet locked in a strong (>0.55) detection on a mascot
    or eye-shaped patch.
    """
    from backend.services import face_detector

    # YuNet picked up a strong-confidence false positive (mascot/eye)
    # at x=20. The cascade finds the real character at x=70.
    results = [
        _StubFrameFaces(
            timestamp=0.0,
            frame_path="/tmp/f.jpg",
            faces=[_StubFace(confidence=0.85, x_center=20.0, is_human=True)],
        ),
    ]
    frame_paths = [(0.0, "/tmp/f.jpg")]

    def _fake_one_cascade(frame_path, *, timestamp=0.0):
        return _StubAnimeResult(
            timestamp=timestamp,
            detections=[
                _StubAnimeDetection(
                    x_center=70.0, y_center=50.0,
                    width=12.0, height=15.0, confidence=0.85,
                ),
            ],
        )

    with patch(
        "backend.services.anime_face_detector.detect_anime_faces",
        side_effect=_fake_one_cascade,
    ), patch(
        "backend.services.anime_face_detector.to_face_info",
        side_effect=_fake_to_face_info,
    ), patch(
        "backend.services.anime_face_detector.USE_ANIME_FACE_DETECTOR",
        True,
    ):
        n_frames, n_faces = face_detector._augment_dense_with_anime(
            results, frame_paths, anime_priority=True,
        )

    # The cascade must have run despite the strong YuNet face,
    # adding the real character.
    assert n_frames == 1
    assert n_faces == 1
    assert len(results[0].faces) == 2  # original YuNet + cascade

    # The YuNet false positive (no embedding → no match) should be
    # demoted (confidence × 0.3). However our stub face has no
    # identity_embedding so the demotion helper short-circuits and
    # leaves it at 0.85. That's the conservative behavior — without
    # embeddings we can't reliably tell true from false. The cascade
    # detection should be present at confidence 0.85 (from the
    # to_face_info stub).
    cascade_faces = [f for f in results[0].faces if not f.is_human]
    assert len(cascade_faces) == 1
    assert abs(cascade_faces[0].x_center - 70.0) < 1e-6
    assert cascade_faces[0].confidence == 0.85


def test_anime_priority_demotes_unmatched_yunet_face():
    """A live-action face whose embedding has NO cluster match within
    ±5 frames is demoted to 0.3× its confidence on anime-priority runs.
    """
    from backend.services import face_detector

    # We need real-ish embeddings for the demotion helper to fire.
    # Build two frames: one with a YuNet false positive that has a
    # unique embedding (no cluster), and one with a cascade detection
    # (no embedding, untouched).
    try:
        import numpy as np
    except ImportError:
        import pytest
        pytest.skip("numpy not available")

    @dataclass
    class _EmbedFace:
        confidence: float = 0.0
        is_human: bool = True
        x_center: float = 50.0
        y_center: float = 50.0
        width: float = 10.0
        height: float = 12.0
        identity_embedding: object = None

    rng = np.random.RandomState(0)
    yunet_face = _EmbedFace(
        confidence=0.85,
        is_human=True,
        x_center=20.0,
        identity_embedding=rng.randn(128).tolist(),
    )
    # A second YuNet hit in a far-away frame with a totally different
    # embedding — represents a different mascot / patch. Together they
    # form a "no two faces ever cluster" scenario, so both get demoted.
    other_face = _EmbedFace(
        confidence=0.85,
        is_human=True,
        x_center=80.0,
        identity_embedding=rng.randn(128).tolist(),
    )

    results = [
        _StubFrameFaces(
            timestamp=0.0,
            frame_path="/tmp/f.jpg",
            faces=[yunet_face],
        ),
        _StubFrameFaces(
            timestamp=10.0,
            frame_path="/tmp/g.jpg",
            faces=[other_face],
        ),
    ]
    frame_paths = [(0.0, "/tmp/f.jpg"), (10.0, "/tmp/g.jpg")]

    def _empty_cascade(frame_path, *, timestamp=0.0):
        return _StubAnimeResult(timestamp=timestamp, detections=[])

    with patch(
        "backend.services.anime_face_detector.detect_anime_faces",
        side_effect=_empty_cascade,
    ), patch(
        "backend.services.anime_face_detector.to_face_info",
        side_effect=_fake_to_face_info,
    ), patch(
        "backend.services.anime_face_detector.USE_ANIME_FACE_DETECTOR",
        True,
    ):
        face_detector._augment_dense_with_anime(
            results, frame_paths, anime_priority=True,
        )

    # The unmatched YuNet face should be demoted: 0.85 × 0.3 = 0.255
    assert abs(yunet_face.confidence - 0.255) < 1e-3, (
        f"Expected demoted confidence ≈ 0.255, got {yunet_face.confidence}"
    )


def test_anime_priority_keeps_clustered_live_action_face():
    """A live-action face that DOES match another live-action face in
    a nearby frame (same person, two consecutive frames) should NOT be
    demoted — it's a real human/character that the cascade missed.
    """
    from backend.services import face_detector

    try:
        import numpy as np
    except ImportError:
        import pytest
        pytest.skip("numpy not available")

    @dataclass
    class _EmbedFace:
        confidence: float = 0.0
        is_human: bool = True
        x_center: float = 50.0
        y_center: float = 50.0
        width: float = 10.0
        height: float = 12.0
        identity_embedding: object = None

    rng = np.random.RandomState(42)
    base_embed = rng.randn(128)
    base_embed = base_embed / float(np.linalg.norm(base_embed))
    # Two near-identical embeddings (same character across frames)
    embed_a = (base_embed + 0.05 * rng.randn(128)).tolist()
    embed_b = (base_embed + 0.05 * rng.randn(128)).tolist()

    face_a = _EmbedFace(confidence=0.85, x_center=50.0, identity_embedding=embed_a)
    face_b = _EmbedFace(confidence=0.85, x_center=50.0, identity_embedding=embed_b)

    results = [
        _StubFrameFaces(timestamp=0.0, frame_path="/tmp/a.jpg", faces=[face_a]),
        _StubFrameFaces(timestamp=0.5, frame_path="/tmp/b.jpg", faces=[face_b]),
    ]
    frame_paths = [(0.0, "/tmp/a.jpg"), (0.5, "/tmp/b.jpg")]

    def _empty_cascade(frame_path, *, timestamp=0.0):
        return _StubAnimeResult(timestamp=timestamp, detections=[])

    with patch(
        "backend.services.anime_face_detector.detect_anime_faces",
        side_effect=_empty_cascade,
    ), patch(
        "backend.services.anime_face_detector.to_face_info",
        side_effect=_fake_to_face_info,
    ), patch(
        "backend.services.anime_face_detector.USE_ANIME_FACE_DETECTOR",
        True,
    ):
        face_detector._augment_dense_with_anime(
            results, frame_paths, anime_priority=True,
        )

    # Neither face should be demoted — they cluster.
    assert face_a.confidence == 0.85
    assert face_b.confidence == 0.85


def test_log_line_fires_with_counts(caplog):
    """The summary log line must include both counts."""
    import logging
    from backend.services import face_detector

    results, frame_paths = _make_results_and_paths()

    with caplog.at_level(logging.INFO, logger="backend.services.face_detector"), \
         patch(
             "backend.services.anime_face_detector.detect_anime_faces",
             side_effect=_fake_detect,
         ), patch(
             "backend.services.anime_face_detector.to_face_info",
             side_effect=_fake_to_face_info,
         ), patch(
             "backend.services.anime_face_detector.USE_ANIME_FACE_DETECTOR",
             True,
         ):
        face_detector._augment_dense_with_anime(
            results, frame_paths, job_log_prefix="[job-xyz] ",
        )

    messages = [r.getMessage() for r in caplog.records]
    assert any(
        "Anime augmentation:" in m
        and "+4 faces" in m
        and "across 2 frames" in m
        and "[job-xyz]" in m
        for m in messages
    ), f"missing expected log line, got: {messages}"
