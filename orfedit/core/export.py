"""Export a rendered image to a file (JPEG / PNG / TIFF)."""

from __future__ import annotations

import os
from typing import Optional

import numpy as np

from .image import RawImage
from .params import EditParams
from .pipeline import Pipeline

_FORMAT_BY_EXT = {
    ".jpg": "JPEG",
    ".jpeg": "JPEG",
    ".png": "PNG",
    ".tif": "TIFF",
    ".tiff": "TIFF",
}

SUPPORTED_EXPORT_EXTENSIONS = tuple(sorted(_FORMAT_BY_EXT))


def export_image(
    image: RawImage,
    params: EditParams,
    path: str,
    quality: int = 95,
    pipeline: Optional[Pipeline] = None,
) -> str:
    """Render ``image`` with ``params`` at full resolution and save it to ``path``.

    The output format is inferred from the file extension.  Returns the path
    written.  Raises :class:`ValueError` for unsupported extensions.
    """
    from PIL import Image

    ext = os.path.splitext(path)[1].lower()
    if ext not in _FORMAT_BY_EXT:
        raise ValueError(
            f"Unsupported export extension {ext!r}. "
            f"Choose one of: {', '.join(SUPPORTED_EXPORT_EXTENSIONS)}"
        )

    pipeline = pipeline or Pipeline()
    rgb = pipeline.render_uint8(image, params)
    pil = Image.fromarray(np.ascontiguousarray(rgb), mode="RGB")

    fmt = _FORMAT_BY_EXT[ext]
    save_kwargs = {}
    if fmt == "JPEG":
        save_kwargs.update(quality=int(quality), subsampling=0, optimize=True)
    elif fmt == "TIFF":
        save_kwargs.update(compression="tiff_lzw")

    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    pil.save(path, format=fmt, **save_kwargs)
    return path
