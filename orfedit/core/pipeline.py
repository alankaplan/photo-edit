"""The edit pipeline: turn a :class:`RawImage` + :class:`EditParams` into pixels.

The pipeline is a pure function.  Given the same inputs it always produces the
same output, which makes it cheap to reason about and easy to test.  The GUI
runs it on a downscaled preview for interactivity and on the full-resolution
image for export -- the code path is identical.

Stage order (chosen to match how photographers expect the controls to behave)::

    linear  --WB--> --exposure-->  [encode to display]
    display --whites/blacks--> --highlights/shadows--> --contrast-->
            --brightness--> --saturation--> --vibrance--> --sharpen-->
            [clip 0..1] --> orient (rotate/flip)
"""

from __future__ import annotations

import numpy as np

from . import adjustments as adj
from .image import RawImage
from .params import EditParams


class Pipeline:
    """Stateless applicator for :class:`EditParams` over a :class:`RawImage`."""

    def render_linear(self, image: RawImage, params: EditParams) -> np.ndarray:
        """Apply the linear-light stages, returning a display-encoded array.

        The result is float RGB in display (sRGB) encoding, not yet clipped, so
        callers can build histograms of out-of-range data if they wish.
        """
        linear = image.linear_rgb
        linear = adj.white_balance(linear, params.temperature, params.tint)
        linear = adj.exposure(linear, params.exposure)
        display = adj.srgb_encode(linear)
        return display

    def render_display(self, image: RawImage, params: EditParams) -> np.ndarray:
        """Full pipeline through display-space edits, returned as unclipped float."""
        d = self.render_linear(image, params)
        d = adj.white_black_point(d, params.whites, params.blacks)
        d = adj.highlights_shadows(d, params.highlights, params.shadows)
        d = adj.contrast(d, params.contrast)
        d = adj.brightness(d, params.brightness)
        d = adj.saturation(d, params.saturation)
        d = adj.vibrance(d, params.vibrance)
        d = adj.unsharp_mask(d, params.sharpen)
        return d

    def render_float(self, image: RawImage, params: EditParams) -> np.ndarray:
        """Final float RGB in ``[0, 1]``, oriented, ready to quantise."""
        d = self.render_display(image, params)
        d = np.clip(d, 0.0, 1.0)
        d = adj.orient(d, params.rotation, params.flip_horizontal, params.flip_vertical)
        return d

    def render_uint8(self, image: RawImage, params: EditParams) -> np.ndarray:
        """Final 8-bit RGB image (H, W, 3) ``uint8`` for display/export."""
        d = self.render_float(image, params)
        return (d * 255.0 + 0.5).astype(np.uint8)


# A shared, stateless default instance plus function shortcuts.
_DEFAULT = Pipeline()


def process(image: RawImage, params: EditParams) -> np.ndarray:
    """Convenience: render ``image`` with ``params`` to a ``uint8`` RGB array."""
    return _DEFAULT.render_uint8(image, params)


def compute_histogram(rgb_uint8: np.ndarray, bins: int = 256):
    """Per-channel histogram of a ``uint8`` RGB image.

    Returns an array of shape ``(3, bins)`` (R, G, B) suitable for plotting.
    """
    if rgb_uint8.dtype != np.uint8:
        rgb_uint8 = np.clip(rgb_uint8, 0, 255).astype(np.uint8)
    hist = np.empty((3, bins), dtype=np.int64)
    for c in range(3):
        hist[c], _ = np.histogram(rgb_uint8[..., c], bins=bins, range=(0, 255))
    return hist
