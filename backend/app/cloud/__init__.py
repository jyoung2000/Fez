"""Cloud storage integration for ClipAI.

This package wires Google Drive and Box user accounts into ClipAI so videos
can be imported directly from a connected cloud provider, streamed
server-side into the same ingestion path as a local upload, and analyzed
without any local round-trip through the user's machine.

The public surface is intentionally small:

- :mod:`backend.app.cloud.base` — provider-agnostic Protocol and DTOs
- :mod:`backend.app.cloud.accounts` — JSON-backed store for encrypted tokens
- :mod:`backend.app.cloud.crypto` — Fernet wrapper for token at-rest encryption
- :mod:`backend.app.cloud.providers` — Google Drive + Box implementations
- :mod:`backend.app.cloud.service` — high-level helpers used by the router
"""
