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
Region = "tuple[float, float, float, float]"  # normalized (x0, y0, x1, y1) in [0, 1]

# Linear-light metering targets.  Whole-frame metering aims at a conservative
# mid-tone; a user-selected region is treated as "the subject", so it is aimed
# a stop or so brighter (roughly zone VI) -- pointing at a light subject should
# make it look well-exposed, not drag it down to 18% grey.
WHOLE_FRAME_TARGET = 0.13
SUBJECT_TARGET = 0.22


def _resolve_region(shape, region):
    """Convert a normalized ``(x0, y0, x1, y1)`` region to pixel slices.

    Returns ``(row_slice, col_slice)`` or ``None`` when ``region`` is ``None``.
    Coordinates are fractions of width/height so a region selected on the preview
    maps identically onto the full-resolution image.
    """
    if region is None:
        return None
    h, w = shape[:2]
    x0, x1 = sorted((float(region[0]), float(region[2])))
    y0, y1 = sorted((float(region[1]), float(region[3])))
    px0 = int(np.clip(round(x0 * w), 0, w - 1))
    px1 = int(np.clip(round(x1 * w), px0 + 1, w))
    py0 = int(np.clip(round(y0 * h), 0, h - 1))
    py1 = int(np.clip(round(y1 * h), py0 + 1, h))
    return (slice(py0, py1), slice(px0, px1))


def _weighted_geomean(lum: np.ndarray, weight: np.ndarray) -> float:
    wsum = float(weight.sum()) + _EPS
    log_lum = np.log(np.clip(lum, 1e-4, None))
    return float(np.exp(float(np.sum(weight * log_lum)) / wsum))


def _metered_key(linear: np.ndarray, region: "Region | None" = None) -> float:
    """Highlight-protected key luminance for metering (geometric mean).

    Without a ``region`` the estimate is *centre-weighted*: it weights toward the
    middle of the frame and away from near-clipped highlights (typically sky or
    windows), which handles the common "subject in the middle" case but only
    guesses where the subject is.

    With a ``region`` (normalized bbox) the user has told us exactly what to
    expose for, so we meter only inside it -- true spot metering -- keeping just
    the highlight-protection weighting.
    """
    lum = luminance(linear)
    if region is not None:
        sl = _resolve_region(lum.shape, region)
        sub = lum[sl]
        w_tone = np.clip(1.0 - _smoothstep(0.7, 1.0, sub), 0.05, 1.0)
        return _weighted_geomean(sub, w_tone)

    h, w = lum.shape
    yy, xx = np.mgrid[0:h, 0:w]
    dy = (yy - (h - 1) / 2.0) / max(h / 2.0, 1.0)
    dx = (xx - (w - 1) / 2.0) / max(w / 2.0, 1.0)
    w_center = np.exp(-(dx * dx + dy * dy) / (2.0 * 0.5 ** 2))
    # De-weight pixels near clipping -- most often bright background, not subject.
    w_tone = np.clip(1.0 - _smoothstep(0.7, 1.0, lum), 0.05, 1.0)
    return _weighted_geomean(lum, w_center * w_tone)


def auto_exposure_ev(
    linear: np.ndarray,
    target: float = 0.13,
    damping: float = 0.85,
    region: "Region | None" = None,
) -> float:
    """Suggested exposure (stops) from (optionally spot-) metering.

    ``target`` is a linear-light key value (~0.13 ≈ a natural mid-tone once the
    sRGB curve is applied).  When ``region`` is given, exposure is metered from
    that region only (spot metering); otherwise metering is centre-weighted and
    ignores near-clipped highlights (see :func:`_metered_key`).  The result is
    damped and clamped for stability.
    """
    key = _metered_key(linear, region=region)
    if key <= _EPS:
        return 0.0
    ev = float(np.log2(target / key)) * damping
    return float(np.clip(ev, -2.5, 2.5))


def auto_tone(linear: np.ndarray, region: "Region | None" = None) -> dict:
    """Suggest a full set of tone adjustments (a Lightroom-style "Auto").

    Returns a dict of :class:`EditParams` field names to values covering
    exposure, contrast, highlights, shadows, whites and blacks.  Exposure comes
    from subject-weighted metering; the resulting display-space tones then decide
    how much to recover highlights, gently open shadows and set the black/white
    points.  The shaping is deliberately conservative -- a firm black point plus
    contrast, but only a light shadow/white lift -- to avoid a washed-out, hazy
    look on lifted frames.

    When a ``region`` is selected the user is pointing at the subject and means
    "expose *this* well", so we aim it at a brighter, subject-appropriate tone
    (``SUBJECT_TARGET``) rather than neutral 18% grey.  Plain mid-grey spot
    metering would darken a light subject (e.g. pale fur) toward grey and leave
    the frame dark, which is why metering a dark area used to work better.
    """
    target = SUBJECT_TARGET if region is not None else WHOLE_FRAME_TARGET
    ev = auto_exposure_ev(linear, target=target, region=region)

    # Evaluate the tones we'd actually see after this exposure, in display space.
    display = srgb_encode(np.clip(linear * (2.0 ** ev), 0.0, None))
    lum_d = luminance(np.clip(display, 0.0, 1.0))
    lo, hi = np.percentile(lum_d, [1.0, 99.0])
    frac_dark = float(np.mean(lum_d < 0.05))
    frac_bright = float(np.mean(lum_d > 0.94))

    # Scale the black/white-point moves by how much tonal range actually
    # exists: a nearly flat frame must not be stretched into pure black/white.
    range_factor = float(np.clip((hi - lo) / 0.25, 0.0, 1.0))

    # Blacks: set a firm black point for contrast (reliable, not hazy).
    blacks = -float(np.clip((lo - 0.02) / 0.05, 0.0, 1.0)) * 42.0 * range_factor
    # Whites: only a small lift, and only for genuinely dull frames.
    whites = float(np.clip((0.90 - hi) / 0.20, 0.0, 1.0)) * 15.0 * range_factor
    # Highlights: recover when a meaningful area is near clipping.
    highlights = -float(np.clip(frac_bright / 0.10, 0.0, 1.0)) * 45.0
    # Shadows: gentle lift only -- aggressive lifting is what washes images out.
    shadows = float(np.clip(frac_dark / 0.18, 0.0, 1.0)) * 28.0

    return {
        "exposure": round(float(ev), 2),
        "contrast": 16,
        "highlights": round(highlights),
        "shadows": round(shadows),
        "whites": round(whites),
        "blacks": round(blacks),
    }


def auto_white_balance(
    linear: np.ndarray, damping: float = 0.85, region: "Region | None" = None
):
    """Estimate temperature/tint to neutralise a colour cast.

    Without a ``region``, a plain full-image grey-world average is thrown off by
    scenes dominated by a saturated colour (foliage, sky, a red wall), so we
    estimate the illuminant from low-saturation, well-exposed pixels only.

    With a ``region``, the user has pointed at something that should be neutral
    (a grey card, a white shirt, a wall) -- the classic eyedropper -- so we use
    every pixel in that region as the reference and neutralise it directly.

    Returns ``(temperature, tint)`` in ``[-100, 100]``.
    """
    if region is not None:
        sl = _resolve_region(linear.shape, region)
        sample = linear[sl].reshape(-1, 3)
        if sample.size == 0:
            return 0.0, 0.0
    else:
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
