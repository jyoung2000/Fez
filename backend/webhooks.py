"""Webhook delivery utility for async operation callbacks.

Used by the v1 API to notify callers when async operations complete
(analysis, export, workflows).
"""

import asyncio
import logging
import os
from datetime import datetime, timezone

import httpx

logger = logging.getLogger(__name__)

WEBHOOK_TIMEOUT = int(os.getenv("WEBHOOK_TIMEOUT_SECONDS", "10"))
WEBHOOK_MAX_RETRIES = int(os.getenv("WEBHOOK_MAX_RETRIES", "3"))


async def send_webhook(callback_url: str, event: str, data: dict):
    """Fire-and-forget webhook notification with retries.

    Args:
        callback_url: URL to POST the payload to.
        event: Event type string (e.g. "analysis_complete").
        data: Event-specific payload data.
    """
    if not callback_url:
        return

    payload = {
        "event": event,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "data": data,
    }

    for attempt in range(WEBHOOK_MAX_RETRIES):
        try:
            async with httpx.AsyncClient(timeout=WEBHOOK_TIMEOUT) as client:
                resp = await client.post(callback_url, json=payload)
                if resp.status_code < 400:
                    logger.info(
                        "Webhook delivered: %s -> %s (status %d)",
                        event, callback_url, resp.status_code,
                    )
                    return
                logger.warning(
                    "Webhook %s to %s returned %d (attempt %d/%d)",
                    event, callback_url, resp.status_code, attempt + 1, WEBHOOK_MAX_RETRIES,
                )
        except Exception as e:
            logger.warning(
                "Webhook %s to %s failed (attempt %d/%d): %s",
                event, callback_url, attempt + 1, WEBHOOK_MAX_RETRIES, e,
            )
        if attempt < WEBHOOK_MAX_RETRIES - 1:
            await asyncio.sleep(2 ** attempt)
