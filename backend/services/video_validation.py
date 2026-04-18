"""Shared video file validation utilities.

Used by both the chunked upload and simple upload routers to avoid
duplicating magic-byte checks and integrity validation logic.
"""

import os

# Expected magic bytes at offset 0 for each format
_MAGIC = {
    "mkv": (0, b"\x1a\x45\xdf\xa3"),   # EBML header
    "webm": (0, b"\x1a\x45\xdf\xa3"),  # EBML header
    "avi": (0, b"RIFF"),               # RIFF container
}
# MP4/MOV: ftyp box -- first 4 bytes are box size, bytes 4-7 are "ftyp"
_FTYP_MAGIC = b"ftyp"


def validate_video_header(path: str, ext: str) -> str | None:
    """Check the first bytes of a video file. Returns an error message or None."""
    with open(path, "rb") as f:
        header = f.read(12)

    if len(header) < 8:
        return "File is too small to be a valid video"

    if header[:8] == b"\x00" * 8:
        return (
            "The file appears to be corrupt or an incomplete download — "
            "the first bytes are all zeros. Please verify the file plays "
            "correctly on your device before uploading."
        )

    if ext in _MAGIC:
        offset, magic = _MAGIC[ext]
        if header[offset:offset + len(magic)] != magic:
            return (
                f"File header does not match expected {ext.upper()} format. "
                "The file may be corrupt or mislabeled."
            )
    elif ext in ("mp4", "mov"):
        if header[4:8] != _FTYP_MAGIC:
            return (
                f"File header does not match expected {ext.upper()} format. "
                "The file may be corrupt or mislabeled."
            )

    return None


def validate_file_integrity(path: str, expected_size: int) -> dict:
    """Run QA checks on an assembled file. Returns a dict of check results."""
    checks = {}
    try:
        actual_size = os.path.getsize(path)
        checks["size_match"] = {
            "pass": actual_size == expected_size,
            "expected": expected_size,
            "actual": actual_size,
        }
    except OSError:
        checks["size_match"] = {"pass": False, "error": "File not found"}

    try:
        with open(path, "rb") as f:
            head = f.read(4096)
            f.seek(0, 2)
            tail_start = max(0, f.tell() - 4096)
            f.seek(tail_start)
            tail = f.read(4096)
        checks["readable"] = {"pass": True}
        checks["non_empty"] = {"pass": len(head) > 0}
        checks["tail_valid"] = {
            "pass": not all(b == 0 for b in tail[-512:]) if len(tail) >= 512 else True
        }
    except OSError as e:
        checks["readable"] = {"pass": False, "error": str(e)}

    return checks
