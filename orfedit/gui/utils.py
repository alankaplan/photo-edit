"""Small helpers shared across GUI widgets."""

from __future__ import annotations

import numpy as np
from PySide6.QtGui import QImage


def ndarray_to_qimage(rgb: np.ndarray) -> QImage:
    """Convert an (H, W, 3) ``uint8`` RGB array to a standalone :class:`QImage`.

    The returned image owns a copy of the pixel data, so the source array can be
    freed or mutated afterwards.
    """
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    h, w = rgb.shape[:2]
    image = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
    return image.copy()
