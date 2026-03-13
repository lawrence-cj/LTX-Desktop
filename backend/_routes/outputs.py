"""Route handler for serving generated output files over HTTP."""

from __future__ import annotations

import re
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from app_handler import AppHandler
from state import get_state_service

router = APIRouter(prefix="/api", tags=["outputs"])

_SAFE_FILENAME = re.compile(r"^[\w\-. ]+$")

_MIME_MAP: dict[str, str] = {
    ".mp4": "video/mp4",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}


@router.get("/outputs/{filename}")
def route_serve_output(
    filename: str,
    handler: AppHandler = Depends(get_state_service),
) -> FileResponse:
    """Serve a file from the outputs directory.

    Only filenames (no path separators) with safe characters are accepted
    to prevent directory traversal attacks.
    """
    if not _SAFE_FILENAME.match(filename) or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    file_path = handler.config.outputs_dir / filename
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    resolved = file_path.resolve()
    outputs_resolved = handler.config.outputs_dir.resolve()
    if not str(resolved).startswith(str(outputs_resolved)):
        raise HTTPException(status_code=403, detail="Access denied")

    suffix = Path(filename).suffix.lower()
    media_type = _MIME_MAP.get(suffix, "application/octet-stream")

    return FileResponse(resolved, media_type=media_type, filename=filename)
