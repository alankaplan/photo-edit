"""Edit parameters.

:class:`EditParams` is a plain dataclass holding every adjustable value in the
pipeline.  Every field defaults to a no-op, so ``EditParams()`` reproduces the
image as decoded from the raw file ("as shot").  All GUI sliders map onto these
fields and the pipeline is a pure function of ``(RawImage, EditParams)``.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, fields, replace
from typing import Any, Dict


@dataclass
class EditParams:
    """Non-destructive edit settings.

    Slider ranges (used by the GUI) are documented per field.  Internally the
    pipeline interprets these units; see :mod:`orfedit.core.adjustments`.
    """

    # --- White balance -----------------------------------------------------
    temperature: float = 0.0  # [-100, 100] negative=cooler/blue, positive=warmer
    tint: float = 0.0         # [-100, 100] negative=green, positive=magenta

    # --- Tone --------------------------------------------------------------
    exposure: float = 0.0     # [-5, 5] stops (EV)
    contrast: float = 0.0     # [-100, 100]
    highlights: float = 0.0   # [-100, 100] negative recovers highlights
    shadows: float = 0.0      # [-100, 100] positive lifts shadows
    whites: float = 0.0       # [-100, 100]
    blacks: float = 0.0       # [-100, 100]
    brightness: float = 0.0   # [-100, 100] simple additive lift after tone map

    # --- Colour ------------------------------------------------------------
    saturation: float = 0.0   # [-100, 100]
    vibrance: float = 0.0     # [-100, 100]

    # --- Detail ------------------------------------------------------------
    sharpen: float = 0.0      # [0, 100] unsharp-mask amount

    # --- Geometry ----------------------------------------------------------
    rotation: int = 0         # {0, 90, 180, 270} degrees, clockwise
    flip_horizontal: bool = False
    flip_vertical: bool = False

    def is_default(self) -> bool:
        """True when nothing has been changed from the as-shot state."""
        return self == EditParams()

    def copy(self) -> "EditParams":
        return replace(self)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EditParams":
        valid = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in valid})


# Metadata describing each adjustable slider for the GUI to build itself from.
# (name, label, group, minimum, maximum, step, default, decimals)
SLIDER_SPECS = [
    ("temperature", "Temperature", "White Balance", -100, 100, 1, 0, 0),
    ("tint", "Tint", "White Balance", -100, 100, 1, 0, 0),

    ("exposure", "Exposure", "Tone", -5.0, 5.0, 0.05, 0.0, 2),
    ("contrast", "Contrast", "Tone", -100, 100, 1, 0, 0),
    ("highlights", "Highlights", "Tone", -100, 100, 1, 0, 0),
    ("shadows", "Shadows", "Tone", -100, 100, 1, 0, 0),
    ("whites", "Whites", "Tone", -100, 100, 1, 0, 0),
    ("blacks", "Blacks", "Tone", -100, 100, 1, 0, 0),
    ("brightness", "Brightness", "Tone", -100, 100, 1, 0, 0),

    ("saturation", "Saturation", "Colour", -100, 100, 1, 0, 0),
    ("vibrance", "Vibrance", "Colour", -100, 100, 1, 0, 0),

    ("sharpen", "Sharpening", "Detail", 0, 100, 1, 0, 0),
]
