"""Pure image-adjustment primitives.

Every function here takes and returns a NumPy ``float32`` RGB array and has no
side effects, no GUI dependency and no global state.  Two colour spaces are
used:

* **linear** light -- where exposure and white balance are physically
  multiplicative;
* **display** (sRGB-encoded) -- where tone, contrast and colour edits behave
  perceptually, matching how sliders feel in a photo editor.

:func:`srgb_encode` / :func:`srgb_decode` convert between them.
"""

from __future__ import annotations

import numpy as np

# Rec.709 luma weights (sRGB primaries).
_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
_EPS = 1e-6


# --------------------------------------------------------------------------
# Colour space transfer functions
# --------------------------------------------------------------------------
def srgb_encode(linear: np.ndarray) -> np.ndarray:
    """Linear light -> sRGB display encoding. Handles values > 1 gracefully."""
    x = np.clip(linear, 0.0, None).astype(np.float32)
    a = 0.055
    return np.where(x <= 0.0031308, x * 12.92, (1 + a) * np.power(x, 1 / 2.4) - a)


def srgb_decode(display: np.ndarray) -> np.ndarray:
    """sRGB display encoding -> linear light."""
    x = np.clip(display, 0.0, None).astype(np.float32)
    a = 0.055
    return np.where(x <= 0.04045, x / 12.92, np.power((x + a) / (1 + a), 2.4))


def luminance(rgb: np.ndarray) -> np.ndarray:
    """Per-pixel luma (H, W) from an (H, W, 3) array."""
    return rgb @ _LUMA


def _smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - edge0) / (edge1 - edge0 + _EPS), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


# --------------------------------------------------------------------------
# Linear-space edits: white balance + exposure
# --------------------------------------------------------------------------
def white_balance(linear: np.ndarray, temperature: float, tint: float) -> np.ndarray:
    """Apply temperature/tint as per-channel gains in linear light.

    ``temperature`` and ``tint`` are in ``[-100, 100]``.  Positive temperature
    warms the image (more red, less blue); positive tint pushes toward magenta,
    negative toward green.
    """
    if temperature == 0 and tint == 0:
        return linear
    t = temperature / 100.0
    ti = tint / 100.0
    r_gain = 1.0 + 0.5 * t + 0.1 * ti
    g_gain = 1.0 - 0.3 * ti
    b_gain = 1.0 - 0.5 * t + 0.1 * ti
    gains = np.array(
        [max(r_gain, 0.0), max(g_gain, 0.0), max(b_gain, 0.0)], dtype=np.float32
    )
    return linear * gains


def exposure(linear: np.ndarray, ev: float) -> np.ndarray:
    """Multiply linear light by ``2**ev`` (stops of exposure)."""
    if ev == 0:
        return linear
    return linear * np.float32(2.0 ** ev)


# --------------------------------------------------------------------------
# Display-space tonal edits
# --------------------------------------------------------------------------
def white_black_point(display: np.ndarray, whites: float, blacks: float) -> np.ndarray:
    """Remap the white and black points (levels), amounts in ``[-100, 100]``.

    Positive ``whites`` brightens highlights (pulls the white point down);
    positive ``blacks`` lifts shadows.
    """
    if whites == 0 and blacks == 0:
        return display
    white_in = 1.0 - 0.4 * (whites / 100.0)
    black_in = -0.4 * (blacks / 100.0)
    white_in = max(white_in, black_in + 1e-3)
    return (display - black_in) / (white_in - black_in)


def highlights_shadows(
    display: np.ndarray, highlights: float, shadows: float
) -> np.ndarray:
    """Region-targeted tone recovery, amounts in ``[-100, 100]``.

    Negative ``highlights`` recovers (darkens) blown highlights; positive
    ``shadows`` lifts shadow detail.  Masks are derived from luma so mid-tones
    are largely preserved.
    """
    if highlights == 0 and shadows == 0:
        return display
    lum = luminance(np.clip(display, 0.0, 1.0))
    out = display
    if shadows != 0:
        shadow_mask = 1.0 - _smoothstep(0.0, 0.55, lum)
        out = out + (shadows / 100.0 * 0.5) * shadow_mask[..., None]
    if highlights != 0:
        highlight_mask = _smoothstep(0.45, 1.0, lum)
        out = out + (highlights / 100.0 * 0.5) * highlight_mask[..., None]
    return out


def contrast(display: np.ndarray, amount: float) -> np.ndarray:
    """Linear contrast about mid-grey (0.5), ``amount`` in ``[-100, 100]``."""
    if amount == 0:
        return display
    factor = 1.0 + amount / 100.0
    return 0.5 + (display - 0.5) * factor


def brightness(display: np.ndarray, amount: float) -> np.ndarray:
    """Additive brightness in display space, ``amount`` in ``[-100, 100]``."""
    if amount == 0:
        return display
    return display + (amount / 100.0) * 0.5


def saturation(display: np.ndarray, amount: float) -> np.ndarray:
    """Uniform saturation, ``amount`` in ``[-100, 100]`` (-100 = greyscale)."""
    if amount == 0:
        return display
    factor = 1.0 + amount / 100.0
    lum = luminance(display)[..., None]
    return lum + (display - lum) * factor


def vibrance(display: np.ndarray, amount: float) -> np.ndarray:
    """Saturation that spares already-saturated pixels, ``amount`` ``[-100,100]``."""
    if amount == 0:
        return display
    d = np.clip(display, 0.0, 1.0)
    mx = d.max(axis=2)
    mn = d.min(axis=2)
    cur_sat = (mx - mn) / (mx + _EPS)
    weight = 1.0 - cur_sat  # boost the least-saturated pixels most
    factor = 1.0 + (amount / 100.0) * weight[..., None]
    lum = luminance(display)[..., None]
    return lum + (display - lum) * factor


# --------------------------------------------------------------------------
# Detail
# --------------------------------------------------------------------------
def _gaussian_kernel1d(sigma: float):
    radius = max(1, int(round(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-(x ** 2) / (2.0 * sigma * sigma))
    return (k / k.sum()).astype(np.float32), radius


def _convolve_axis(img: np.ndarray, kernel: np.ndarray, radius: int, axis: int):
    pad = [(radius, radius) if a == axis else (0, 0) for a in range(img.ndim)]
    padded = np.pad(img, pad, mode="reflect")
    acc = np.zeros_like(img)
    n = img.shape[axis]
    for i, w in enumerate(kernel):
        sl = [slice(None)] * img.ndim
        sl[axis] = slice(i, i + n)
        acc += w * padded[tuple(sl)]
    return acc


def gaussian_blur(img: np.ndarray, sigma: float = 1.2) -> np.ndarray:
    """Separable Gaussian blur (no SciPy dependency)."""
    kernel, radius = _gaussian_kernel1d(sigma)
    out = _convolve_axis(img, kernel, radius, axis=0)
    out = _convolve_axis(out, kernel, radius, axis=1)
    return out


def unsharp_mask(display: np.ndarray, amount: float, sigma: float = 1.2) -> np.ndarray:
    """Unsharp-mask sharpening, ``amount`` in ``[0, 100]``."""
    if amount <= 0:
        return display
    blurred = gaussian_blur(display, sigma=sigma)
    strength = amount / 100.0 * 1.5
    return display + strength * (display - blurred)


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------
def orient(rgb: np.ndarray, rotation: int, flip_h: bool, flip_v: bool) -> np.ndarray:
    """Rotate (clockwise, multiples of 90) and flip an image."""
    out = rgb
    rot = rotation % 360
    if rot:
        # np.rot90 is counter-clockwise; negate to make the slider clockwise.
        out = np.rot90(out, k=(-rot // 90) % 4)
    if flip_h:
        out = out[:, ::-1]
    if flip_v:
        out = out[::-1, :]
    return np.ascontiguousarray(out)


# --------------------------------------------------------------------------
# Auto adjustments (suggest slider values from image statistics)
# --------------------------------------------------------------------------
def auto_exposure_ev(
    linear: np.ndarray, target: float = 0.11, damping: float = 0.8
) -> float:
    """Suggested exposure (stops) to place the median tone near ``target``.

    ``target`` is a linear-light value (0.11 ≈ a natural mid-tone once the sRGB
    curve is applied).  The correction is damped and clamped so a single bright
    or dark region can't drive the whole frame to a clipped extreme -- the old
    behaviour, which mapped the median straight onto mid-grey, tended to
    over-brighten well-lit scenes and under-brighten low-key ones.
    """
    lum = luminance(linear)
    median = float(np.median(lum))
    if median <= _EPS:
        return 0.0
    ev = float(np.log2(target / median)) * damping
    return float(np.clip(ev, -2.5, 2.5))


def auto_tone(linear: np.ndarray) -> dict:
    """Suggest a full set of tone adjustments (a Lightroom-style "Auto").

    Returns a dict of :class:`EditParams` field names to values covering
    exposure, contrast, highlights, shadows, whites and blacks.  Exposure is
    estimated first, then the resulting display-space tones are measured to
    decide how much to recover highlights, open shadows and set the black/white
    points -- so the whole tonal range is used instead of leaving the image flat
    and hazy.
    """
    ev = auto_exposure_ev(linear)

    # Evaluate the tones we'd actually see after this exposure, in display space.
    display = srgb_encode(np.clip(linear * (2.0 ** ev), 0.0, None))
    lum_d = luminance(np.clip(display, 0.0, 1.0))
    lo, hi = np.percentile(lum_d, [1.0, 99.0])
    frac_dark = float(np.mean(lum_d < 0.06))
    frac_bright = float(np.mean(lum_d > 0.94))

    # Scale the black/white-point moves by how much tonal range actually
    # exists: a nearly flat frame must not be stretched into pure black/white.
    range_factor = float(np.clip((hi - lo) / 0.25, 0.0, 1.0))

    # Blacks: deepen milky/lifted shadows toward a near-black point (~0.03).
    blacks = -float(np.clip((lo - 0.03) / 0.06, 0.0, 1.0)) * 45.0 * range_factor
    # Whites: add headroom when the brightest tones fall well short of white.
    whites = float(np.clip((0.92 - hi) / 0.15, 0.0, 1.0)) * 35.0 * range_factor
    # Highlights: recover when a meaningful area is near clipping.
    highlights = -float(np.clip(frac_bright / 0.10, 0.0, 1.0)) * 55.0
    # Shadows: open up when a meaningful area is crushed.
    shadows = float(np.clip(frac_dark / 0.12, 0.0, 1.0)) * 45.0

    return {
        "exposure": round(float(ev), 2),
        "contrast": 12,
        "highlights": round(highlights),
        "shadows": round(shadows),
        "whites": round(whites),
        "blacks": round(blacks),
    }


def auto_white_balance(linear: np.ndarray, damping: float = 0.85):
    """Estimate temperature/tint from near-neutral pixels (grey-world, robust).

    A plain full-image grey-world average is thrown off by scenes dominated by a
    saturated colour (foliage, sky, a red wall): it tries to neutralise the
    subject and pushes the opposite way.  Instead we estimate the illuminant
    from low-saturation, well-exposed pixels only, then recover both temperature
    and tint.  Returns ``(temperature, tint)`` in ``[-100, 100]``.
    """
    rgb = linear.reshape(-1, 3)
    lum = luminance(linear).reshape(-1)
    mx = rgb.max(axis=1)
    mn = rgb.min(axis=1)
    sat = (mx - mn) / (mx + _EPS)

    # Consider pixels that are neither near-black nor near-clipping.
    valid = (lum > 0.02) & (mx < 0.98)
    neutral = valid & (sat < 0.15)
    min_count = max(50, int(0.02 * rgb.shape[0]))
    if int(neutral.sum()) >= min_count:
        sample = rgb[neutral]
    else:
        # Fallback: the least-saturated 30% of the valid pixels.
        idx = np.where(valid)[0]
        if idx.size == 0:
            return 0.0, 0.0
        order = idx[np.argsort(sat[idx])]
        sample = rgb[order[: max(1, int(0.3 * idx.size))]]

    means = np.maximum(sample.mean(axis=0), _EPS)
    gains = means.mean() / means          # grey-world gains over neutral pixels
    gn = gains / gains.mean()             # normalise so the average gain is 1

    temperature = np.clip(100.0 * (gn[0] - gn[2]) * damping, -100.0, 100.0)
    tint = np.clip(100.0 * (1.0 - gn[1]) / 0.3 * damping, -100.0, 100.0)
    return float(temperature), float(tint)
