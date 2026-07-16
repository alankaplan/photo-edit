"""In-memory representation of a decoded raw image.

The rest of the core operates on :class:`RawImage`, never directly on a file or
on ``rawpy`` objects.  This keeps the pipeline and GUI decoupled from LibRaw and
makes it trivial to feed synthetic data in tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass
class RawImage:
    """A demosaiced raw frame in linear light.

    Attributes
    ----------
    linear_rgb:
        ``float32`` array of shape ``(H, W, 3)`` in linear-light sRGB
        primaries.  Values are normalised so that the sensor white point maps
        to ``1.0``; highlight-clipped or reconstructed data may exceed ``1.0``.
    metadata:
        Free-form dictionary of human-readable capture info (camera, ISO,
        shutter, aperture, focal length, timestamp, ...).
    source_path:
        Path the image was loaded from, if any.
    thumbnail:
        Optional small ``uint8`` preview extracted from the file.
    """

    linear_rgb: np.ndarray
    metadata: Dict[str, str] = field(default_factory=dict)
    source_path: Optional[str] = None
    thumbnail: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        arr = np.asarray(self.linear_rgb)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(
                f"linear_rgb must have shape (H, W, 3); got {arr.shape!r}"
            )
        self.linear_rgb = np.ascontiguousarray(arr, dtype=np.float32)

    # -- convenience ---------------------------------------------------------
    @property
    def size(self) -> Tuple[int, int]:
        """``(width, height)`` in pixels."""
        h, w = self.linear_rgb.shape[:2]
        return (w, h)

    @property
    def megapixels(self) -> float:
        h, w = self.linear_rgb.shape[:2]
        return (h * w) / 1e6

    def downscaled(self, max_edge: int) -> "RawImage":
        """Return a copy whose longest edge is at most ``max_edge`` pixels.

        Uses simple strided/area sampling -- fast and good enough for an
        interactive preview.  The full-resolution image is retained separately
        by the caller for export.
        """
        h, w = self.linear_rgb.shape[:2]
        longest = max(h, w)
        if longest <= max_edge:
            return RawImage(
                self.linear_rgb.copy(),
                dict(self.metadata),
                self.source_path,
                self.thumbnail,
            )
        scale = max_edge / float(longest)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        small = _resize_area(self.linear_rgb, new_w, new_h)
        return RawImage(small, dict(self.metadata), self.source_path, self.thumbnail)


def _resize_area(arr: np.ndarray, new_w: int, new_h: int) -> np.ndarray:
    """Downscale ``arr`` (H, W, C) to (new_h, new_w, C) by block averaging.

    Falls back to nearest sampling when up-scaling (not used for previews).
    """
    h, w = arr.shape[:2]
    if new_w >= w or new_h >= h:
        ys = np.clip((np.arange(new_h) * h / new_h).astype(int), 0, h - 1)
        xs = np.clip((np.arange(new_w) * w / new_w).astype(int), 0, w - 1)
        return arr[ys][:, xs].copy()

    # Area average: map each output pixel to a source block and take its mean.
    y_edges = np.linspace(0, h, new_h + 1).astype(int)
    x_edges = np.linspace(0, w, new_w + 1).astype(int)
    out = np.empty((new_h, new_w, arr.shape[2]), dtype=np.float32)
    # Reduce rows first, then columns, for reasonable speed without SciPy.
    row_reduced = np.empty((new_h, w, arr.shape[2]), dtype=np.float32)
    for i in range(new_h):
        y0, y1 = y_edges[i], max(y_edges[i] + 1, y_edges[i + 1])
        row_reduced[i] = arr[y0:y1].mean(axis=0)
    for j in range(new_w):
        x0, x1 = x_edges[j], max(x_edges[j] + 1, x_edges[j + 1])
        out[:, j] = row_reduced[:, x0:x1].mean(axis=1)
    return out


def synthetic_raw(width: int = 900, height: int = 600, seed: int = 0) -> RawImage:
    """Generate a colourful synthetic scene as a :class:`RawImage`.

    Useful for demos, headless GUI rendering and tests when no real ORF file is
    available.  The image contains smooth colour gradients, a bright highlight
    region and dark shadow region so tonal adjustments have something to act on.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    u = xx / max(1, width - 1)
    v = yy / max(1, height - 1)

    # Base gradients across the frame in linear light.
    r = 0.15 + 0.8 * u
    g = 0.15 + 0.8 * v
    b = 0.15 + 0.8 * (1.0 - 0.5 * (u + v))

    # A bright "sky" band near the top (highlight test region).
    sky = np.clip(1.4 - 3.0 * v, 0.0, 1.0) ** 2
    r = r + 0.9 * sky
    g = g + 0.95 * sky
    b = b + 1.1 * sky

    # A dark "shadow" corner (bottom-left) to exercise shadow lifting.
    shadow = np.clip(1.0 - 2.2 * np.hypot(u, 1.0 - v), 0.0, 1.0)
    for ch in (r, g, b):
        ch *= 1.0 - 0.85 * shadow

    # A saturated colour disc in the centre.
    cx, cy = 0.5, 0.55
    disc = np.clip(1.0 - 6.0 * np.hypot(u - cx, v - cy), 0.0, 1.0)
    r = r * (1 - disc) + disc * 0.95
    g = g * (1 - disc) + disc * 0.15
    b = b * (1 - disc) + disc * 0.35

    rgb = np.stack([r, g, b], axis=-1).astype(np.float32)
    rgb += rng.normal(0.0, 0.006, size=rgb.shape).astype(np.float32)  # sensor noise
    rgb = np.clip(rgb, 0.0, None)

    meta = {
        "Camera": "Synthetic (demo)",
        "Model": "OM-Demo",
        "ISO": "200",
        "Shutter": "1/250 s",
        "Aperture": "f/4.0",
        "Focal length": "25 mm",
        "Dimensions": f"{width} x {height}",
    }
    return RawImage(rgb, meta, source_path=None)
