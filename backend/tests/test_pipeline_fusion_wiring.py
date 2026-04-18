"""Smoke tests that the pipeline imports the fusion helper and that
``fusion_enabled()`` is read at run time rather than cached at module
import. Covers Change 2 of the VLM-upgrade wiring prompt."""


def test_pipeline_imports_fusion_helper():
    from backend.services import pipeline as pipeline_mod
    assert hasattr(pipeline_mod, "apply_fusion_to_scene")
    assert hasattr(pipeline_mod, "fusion_enabled")


def test_fusion_gate_runtime_read(monkeypatch):
    """``fusion_enabled()`` must read ``CLIPAI_VLM_FUSION`` at call time
    so tests / runtime config flips work. The pipeline must NOT cache
    the env var at module import."""
    from backend.services.vlm_fusion import fusion_enabled
    monkeypatch.delenv("CLIPAI_VLM_FUSION", raising=False)
    assert fusion_enabled() is False
    monkeypatch.setenv("CLIPAI_VLM_FUSION", "weighted")
    assert fusion_enabled() is True
    monkeypatch.setenv("CLIPAI_VLM_FUSION", "hard")
    assert fusion_enabled() is False
