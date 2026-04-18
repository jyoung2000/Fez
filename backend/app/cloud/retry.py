"""Exponential backoff wrapper for outbound cloud-provider HTTP calls.

All provider implementations route their network access through
:func:`retry_request` so that 429 / 5xx responses are retried with
exponential backoff that honors ``Retry-After`` headers. We cap retries
at 3 and then bubble the failure up to the API layer, which translates
it into a user-visible error.
"""

from __future__ import annotations

import asyncio
import logging
import random
from typing import Awaitable, Callable, TypeVar

import httpx

logger = logging.getLogger(__name__)

T = TypeVar("T")

MAX_ATTEMPTS = 3
BASE_BACKOFF_SECONDS = 1.0
MAX_BACKOFF_SECONDS = 30.0


class CloudProviderError(Exception):
    """Raised when a provider call fails after all retries."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def _retry_after_seconds(response: httpx.Response) -> float | None:
    raw = response.headers.get("Retry-After")
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        # HTTP-date form — not worth parsing; fall back to exponential.
        return None


async def retry_request(
    send: Callable[[], Awaitable[httpx.Response]],
    *,
    description: str,
    max_attempts: int = MAX_ATTEMPTS,
) -> httpx.Response:
    """Invoke *send()* with retry on 429 / 5xx / transport errors.

    The caller is responsible for raising on non-retryable 4xx responses —
    ``retry_request`` only handles the "try again later" cases.
    """
    attempt = 0
    last_exc: Exception | None = None
    while attempt < max_attempts:
        attempt += 1
        try:
            response = await send()
        except (httpx.TransportError, httpx.TimeoutException) as exc:
            last_exc = exc
            if attempt >= max_attempts:
                raise CloudProviderError(
                    f"{description}: network error after {attempt} attempts: {exc}"
                ) from exc
            delay = _backoff_delay(attempt)
            logger.warning(
                "%s transport error (attempt %d/%d): %s — retrying in %.1fs",
                description,
                attempt,
                max_attempts,
                exc,
                delay,
            )
            await asyncio.sleep(delay)
            continue

        if response.status_code == 429 or 500 <= response.status_code < 600:
            if attempt >= max_attempts:
                return response  # caller will raise based on the status
            delay = _retry_after_seconds(response) or _backoff_delay(attempt)
            logger.warning(
                "%s got HTTP %d (attempt %d/%d) — retrying in %.1fs",
                description,
                response.status_code,
                attempt,
                max_attempts,
                delay,
            )
            await asyncio.sleep(delay)
            continue

        return response

    # Should be unreachable, but keep type checkers happy.
    raise CloudProviderError(
        f"{description}: exhausted retries without a response"
    ) from last_exc


def _backoff_delay(attempt: int) -> float:
    # Full jitter: random value in [0, base * 2**(attempt-1)], capped.
    ceiling = min(BASE_BACKOFF_SECONDS * (2 ** (attempt - 1)), MAX_BACKOFF_SECONDS)
    return random.uniform(0, ceiling)
