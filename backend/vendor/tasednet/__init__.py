"""Vendored TASED-Net hook.

This is a *placeholder* — the upstream MichiganCOG/TASED-Net repo
ships without an explicit license, so we don't ship the model
definition in this image. Operators who want the higher-quality
TASED-Net backend (vs. the spectral-residual fallback) supply their
own ``model.py`` here, with whatever attribution / licensing they
choose to comply with upstream terms.

When ``model.py`` is present and exports a ``TASED_v2`` PyTorch
``nn.Module``, ``backend.services.av_saliency._TasedNetAdapter``
imports and uses it. When it's absent, AvSaliency silently falls back
to the spectral-residual adapter — production stays functional either
way.

See ``infra/saliency/README.md`` for the operator-side setup.
"""

try:
    from .model import TASED_v2  # noqa: F401
except ImportError as _exc:  # pragma: no cover — operator-side path
    # Re-raise as ImportError so _TasedNetAdapter.load() can catch it
    # and trigger the spectral-residual fallback.
    raise ImportError(
        "TASED-Net model.py not vendored — see infra/saliency/README.md"
    ) from _exc
