import logging
import os
import re
import subprocess

from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import FileResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["fonts"])

FONTS_DIR = "/data/fonts"
ALLOWED_EXTENSIONS = {"ttf", "otf", "woff", "woff2", "eot"}
MAX_FONT_SIZE = 20 * 1024 * 1024  # 20 MB

# System-installed fonts that we expose to the frontend for @font-face preview.
# These map a safe filename key → absolute path on the Docker container.
SYSTEM_FONT_PATHS = {
    "DMSans.ttf": "/usr/share/fonts/truetype/dmsans/DMSans.ttf",
    "LiberationSans-Regular.ttf": "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "LiberationSerif-Regular.ttf": "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
    "LiberationMono-Regular.ttf": "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    "DejaVuSans.ttf": "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "DejaVuSerif.ttf": "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    "DejaVuSansMono.ttf": "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "FreeSans.ttf": "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    # Google Fonts (installed in Dockerfile)
    "Montserrat.ttf": "/usr/share/fonts/truetype/google-fonts/Montserrat.ttf",
    "OpenSans.ttf": "/usr/share/fonts/truetype/google-fonts/OpenSans.ttf",
    "Roboto.ttf": "/usr/share/fonts/truetype/google-fonts/Roboto.ttf",
    "Poppins-Regular.ttf": "/usr/share/fonts/truetype/google-fonts/Poppins-Regular.ttf",
    "Inter.ttf": "/usr/share/fonts/truetype/google-fonts/Inter.ttf",
    "Nunito.ttf": "/usr/share/fonts/truetype/google-fonts/Nunito.ttf",
    "Lato-Regular.ttf": "/usr/share/fonts/truetype/google-fonts/Lato-Regular.ttf",
    "Oswald.ttf": "/usr/share/fonts/truetype/google-fonts/Oswald.ttf",
    "PlayfairDisplay.ttf": "/usr/share/fonts/truetype/google-fonts/PlayfairDisplay.ttf",
    "BebasNeue-Regular.ttf": "/usr/share/fonts/truetype/google-fonts/BebasNeue-Regular.ttf",
}

os.makedirs(FONTS_DIR, exist_ok=True)


def _safe_filename(name: str) -> str:
    """Sanitize filename to prevent path traversal."""
    name = os.path.basename(name)
    # Remove anything that isn't alphanumeric, dash, underscore, dot, or space
    name = re.sub(r'[^\w\s\-.]', '', name)
    return name.strip()


def _font_family_name(font_path: str) -> str | None:
    """Extract the real font family name using fc-query (fontconfig).

    This is the name that FFmpeg/libass uses to look up fonts, which may
    differ from the filename.  Returns None if fc-query is unavailable or
    the font can't be parsed.
    """
    try:
        result = subprocess.run(
            ["fc-query", "--format", "%{family}", font_path],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            # fc-query may return comma-separated family names; take the first
            return result.stdout.strip().split(",")[0].strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def _font_display_name(filename: str) -> str:
    """Derive a display name from a font filename (strip extension)."""
    return os.path.splitext(filename)[0].replace('-', ' ').replace('_', ' ')


# Formats that libass can read natively.
_LIBASS_NATIVE_EXTS = {"ttf", "otf"}


def _convert_to_ttf(src_path: str, dst_path: str) -> bool:
    """Convert a WOFF/WOFF2 font to TTF using fonttools.

    Returns True on success, False on failure (original file left in place).
    libass (used by FFmpeg for subtitle rendering) only supports TTF/OTF,
    so web font formats must be converted to ensure the exported video
    uses the same font the preview showed.
    """
    try:
        from fontTools.ttLib import TTFont

        font = TTFont(src_path)
        font.flavor = None  # remove WOFF/WOFF2 wrapper → raw TrueType/OpenType
        font.save(dst_path)
        font.close()
        logger.info(f"Converted {os.path.basename(src_path)} → {os.path.basename(dst_path)}")
        return True
    except Exception as e:
        logger.warning(f"Font conversion failed for {src_path}: {e}")
        return False


@router.post("/fonts/upload")
async def upload_font(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported font format. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    safe_name = _safe_filename(file.filename)
    if not safe_name:
        raise HTTPException(status_code=400, detail="Invalid filename")

    font_path = os.path.join(FONTS_DIR, safe_name)

    content = await file.read()
    if len(content) > MAX_FONT_SIZE:
        raise HTTPException(status_code=400, detail="Font file too large (max 20 MB)")
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Font file is empty")

    with open(font_path, "wb") as f:
        f.write(content)

    # Convert WOFF/WOFF2/EOT to TTF so libass can use the font during export.
    # The browser preview works with any format via @font-face, but FFmpeg's
    # libass only reads TTF/OTF natively.
    if ext not in _LIBASS_NATIVE_EXTS:
        ttf_name = os.path.splitext(safe_name)[0] + ".ttf"
        ttf_path = os.path.join(FONTS_DIR, ttf_name)
        if _convert_to_ttf(font_path, ttf_path):
            # Keep the original for browser serving, but the TTF is what
            # libass/fontconfig will discover.
            safe_name = ttf_name
            font_path = ttf_path
        else:
            logger.warning(
                f"Could not convert {safe_name} to TTF — subtitle export "
                f"may fall back to a different font"
            )

    # Use the real internal family name so the ASS file matches what
    # FFmpeg/libass will look up.  Fall back to filename-derived name.
    display_name = _font_family_name(font_path) or _font_display_name(safe_name)
    logger.info(f"Font uploaded: {safe_name} as '{display_name}' ({len(content)} bytes)")

    # Update fontconfig cache so libass can find this font immediately
    try:
        subprocess.run(["fc-cache", "-f", FONTS_DIR], capture_output=True, timeout=10)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return {
        "name": display_name,
        "filename": safe_name,
        "url": f"/api/fonts/file/{safe_name}",
    }


@router.get("/fonts")
async def list_fonts():
    """List custom (user-uploaded) fonts.  System font symlinks are excluded."""
    if not os.path.isdir(FONTS_DIR):
        return []
    fonts = []
    for fname in sorted(os.listdir(FONTS_DIR)):
        font_path = os.path.join(FONTS_DIR, fname)
        # Skip symlinks — those are system fonts symlinked for FFmpeg fontsdir
        if os.path.islink(font_path):
            continue
        ext = fname.rsplit(".", 1)[-1].lower() if "." in fname else ""
        if ext in ALLOWED_EXTENSIONS:
            name = _font_family_name(font_path) or _font_display_name(fname)
            fonts.append({
                "name": name,
                "filename": fname,
                "url": f"/api/fonts/file/{fname}",
            })
    return fonts


@router.get("/fonts/file/{filename}")
async def serve_font(filename: str):
    safe_name = _safe_filename(filename)
    font_path = os.path.join(FONTS_DIR, safe_name)
    if not os.path.isfile(font_path):
        raise HTTPException(status_code=404, detail="Font not found")

    ext = safe_name.rsplit(".", 1)[-1].lower() if "." in safe_name else ""
    content_types = {
        "ttf": "font/ttf",
        "otf": "font/otf",
        "woff": "font/woff",
        "woff2": "font/woff2",
        "eot": "application/vnd.ms-fontobject",
    }
    return FileResponse(font_path, media_type=content_types.get(ext, "application/octet-stream"))


@router.get("/fonts/builtin/{filename}")
async def serve_builtin_font(filename: str):
    """Serve a system-installed font so the browser preview can render it."""
    path = SYSTEM_FONT_PATHS.get(filename)
    if not path or not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Builtin font not found")
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "ttf"
    content_types = {
        "ttf": "font/ttf",
        "otf": "font/otf",
        "woff": "font/woff",
        "woff2": "font/woff2",
    }
    return FileResponse(path, media_type=content_types.get(ext, "font/ttf"))


@router.delete("/fonts/{filename}")
async def delete_font(filename: str):
    safe_name = _safe_filename(filename)
    font_path = os.path.join(FONTS_DIR, safe_name)
    if os.path.islink(font_path):
        raise HTTPException(status_code=400, detail="Cannot delete system font")
    if os.path.isfile(font_path):
        os.remove(font_path)
        logger.info(f"Font deleted: {safe_name}")
        return {"status": "deleted", "filename": safe_name}
    raise HTTPException(status_code=404, detail="Font not found")
