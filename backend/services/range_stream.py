"""Stream byte-ranges of on-disk files as HTTP 206 responses.

HTML5 ``<video>`` elements seek by issuing ``Range: bytes=start-end``
requests. Before this helper existed, the authenticated
``/api/files`` and public ``/api/share/public/{token}/video`` handlers
each implemented Range support by doing::

    with open(path, "rb") as f:
        f.seek(start)
        data = f.read(content_length)
    return Response(content=data, status_code=206, ...)

That works for small ranges but has two problems in production:

1.  A browser seeking into a large video can request megabytes of
    data per Range; the handler pins that whole slice in RAM until
    ``Response`` finishes writing it back.
2.  The read happens on the event-loop thread (it's blocking I/O
    inside an async handler). For multiple concurrent viewers this
    stalls unrelated requests.

``FileRangeStreamer`` replaces the pattern with a chunked async
generator backed by Starlette's ``StreamingResponse``. Chunks are
small (``_DEFAULT_CHUNK_BYTES``) so memory stays flat; reads happen
lazily as the client drains the socket, which also lets clients
abort mid-seek cheaply.

The helper preserves the exact header shape the previous
implementations used so nothing else changes — just the delivery
mechanism underneath.
"""

from __future__ import annotations

import os
from typing import AsyncIterator, Iterator, Optional

from fastapi.responses import StreamingResponse


# 256 KiB — large enough that we're not thrashing ``read()``/syscall
# boundaries, small enough that aborted seeks (the common case) free
# memory quickly. Tested against 4K H.264 playback: range fetches
# complete in a few chunks per seek.
_DEFAULT_CHUNK_BYTES = 256 * 1024


def parse_range_header(
    range_header: Optional[str],
    file_size: int,
) -> Optional[tuple[int, int]]:
    """Parse a single ``Range: bytes=start-end`` header.

    Returns ``(start, end)`` inclusive on success, or ``None`` when
    the header is missing. Raises ``ValueError`` on a malformed /
    unsatisfiable range so the caller can reply with 416 if desired.
    """
    if not range_header:
        return None
    try:
        units, _, spec = range_header.partition("=")
        if units.strip().lower() != "bytes":
            raise ValueError("non-bytes range")
        first = spec.split(",", 1)[0]
        start_s, _, end_s = first.partition("-")
        if start_s == "" and end_s == "":
            raise ValueError("empty range")
        # Suffix form ``bytes=-500`` → last 500 bytes.
        if start_s == "" and end_s != "":
            length = int(end_s)
            if length <= 0:
                raise ValueError("non-positive suffix length")
            start = max(0, file_size - length)
            end = file_size - 1
        else:
            start = int(start_s)
            end = int(end_s) if end_s else file_size - 1
    except (ValueError, IndexError) as e:
        raise ValueError(f"invalid range header: {e}") from e

    if start < 0 or start >= file_size or end < start:
        raise ValueError("range out of bounds")
    end = min(end, file_size - 1)
    return start, end


def _file_range_iter(
    path: str,
    start: int,
    length: int,
    chunk_size: int = _DEFAULT_CHUNK_BYTES,
) -> Iterator[bytes]:
    """Yield successive byte chunks of ``path`` starting at ``start``.

    StreamingResponse runs sync generators on the threadpool, so
    ``open`` / ``read`` blocking calls don't stall the event loop.
    """
    remaining = length
    with open(path, "rb") as f:
        f.seek(start)
        while remaining > 0:
            chunk = f.read(min(chunk_size, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
            yield chunk


def stream_file_range(
    file_path: str,
    file_size: int,
    start: int,
    end: int,
    content_type: str,
    extra_headers: Optional[dict] = None,
) -> StreamingResponse:
    """Build a 206 Partial Content response that streams the range.

    ``start`` and ``end`` are inclusive; the Content-Length header is
    computed from them, matching RFC 7233. ``extra_headers`` gets
    merged last so callers can add Cache-Control etc.
    """
    content_length = end - start + 1
    headers = {
        "Content-Range": f"bytes {start}-{end}/{file_size}",
        "Accept-Ranges": "bytes",
        "Content-Length": str(content_length),
        "Content-Type": content_type,
    }
    if extra_headers:
        headers.update(extra_headers)
    return StreamingResponse(
        _file_range_iter(file_path, start, content_length),
        status_code=206,
        media_type=content_type,
        headers=headers,
    )


__all__ = ["parse_range_header", "stream_file_range"]
