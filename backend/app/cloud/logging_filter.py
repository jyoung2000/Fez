"""Log filter that redacts anything token-shaped before it hits the handlers.

This is strictly defensive — code in this package never logs raw tokens.
But any future contributor who calls ``logger.debug(response.json())`` on a
successful OAuth exchange would otherwise leak credentials straight to the
container logs. The filter runs at the root logger and scans formatted
records for ``access_token``, ``refresh_token``, and ``Authorization``
values, replacing them with ``<redacted>``.

Install with :func:`install_redaction_filter`. The function is idempotent
so it's safe to call from multiple startup hooks.
"""

from __future__ import annotations

import logging
import re
from typing import Final

_PATTERNS: Final = [
    # JSON form: "access_token": "abc..."
    re.compile(
        r'("(?:access_token|refresh_token|id_token)"\s*:\s*")[^"]*(")',
        re.IGNORECASE,
    ),
    # urlencoded form: access_token=abc...
    re.compile(
        r"((?:access_token|refresh_token|id_token)=)[^&\s\"']+",
        re.IGNORECASE,
    ),
    # Authorization header values
    re.compile(
        r"(Authorization['\"]?\s*[:=]\s*['\"]?(?:Bearer|Basic|Token)\s+)\S+",
        re.IGNORECASE,
    ),
    # Free-form "Bearer <token>" that isn't inside an Authorization header
    re.compile(r"(Bearer\s+)[A-Za-z0-9\-_\.]{8,}", re.IGNORECASE),
]

_REDACTED = "<redacted>"


def _scrub(text: str) -> str:
    for pat in _PATTERNS:
        text = pat.sub(
            lambda m: m.group(1) + _REDACTED + (m.group(2) if m.lastindex and m.lastindex >= 2 else ""),
            text,
        )
    return text


# Logger names whose ``record.args`` are consumed by a custom Formatter
# (e.g. uvicorn's ``AccessFormatter`` unpacks a 5-tuple) and MUST NOT be
# rewritten by this filter. Their args carry no tokens — only HTTP
# metadata — so skipping them is safe and avoids the ValueError flood
# that otherwise corrupts streaming responses (BaseHTTPMiddleware +
# StreamingResponse re-enters the access logger per chunk; emptying
# ``record.args`` makes uvicorn raise on every re-emit and breaks the
# response flow for long-running endpoints like ``/retranscribe``).
_PRESERVE_ARGS_LOGGERS: Final = (
    "uvicorn.access",
    "gunicorn.access",
    "hypercorn.access",
)


class TokenRedactionFilter(logging.Filter):
    """Scrub tokens from the rendered message and formatted args."""

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: D401 - Filter API
        try:
            # Access loggers use a custom Formatter that unpacks
            # ``record.args`` into a fixed-arity tuple. Rendering the
            # message early and clearing ``args`` would crash the
            # second formatting pass with "not enough values to
            # unpack". Their args never contain tokens, so leave them
            # entirely alone.
            if record.name in _PRESERVE_ARGS_LOGGERS:
                return True
            if record.args:
                # Render once up front so we can scrub a single string.
                rendered = record.getMessage()
                record.msg = _scrub(rendered)
                record.args = ()
            elif isinstance(record.msg, str):
                record.msg = _scrub(record.msg)
        except Exception:  # pragma: no cover - logging must never raise
            pass
        return True


_installed = False


def install_redaction_filter() -> None:
    """Attach the filter to the root logger exactly once."""
    global _installed
    if _installed:
        return
    root = logging.getLogger()
    root.addFilter(TokenRedactionFilter())
    # Also attach to any already-configured child loggers whose handlers
    # were installed before we were imported — filters on the root only
    # run if propagation reaches it.
    for name in ("uvicorn", "uvicorn.access", "uvicorn.error", "httpx"):
        logging.getLogger(name).addFilter(TokenRedactionFilter())
    _installed = True
