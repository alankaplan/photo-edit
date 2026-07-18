import numpy as np
import pytest

from orfedit.core import (
    EditParams,
    Pipeline,
    RawImage,
    compute_histogram,
    process,
    synthetic_raw,
)
from orfedit.core import adjustments as adj


@pytest.fixture
def image():
    return synthetic_raw(120, 80, seed=3)


def test_default_params_match_srgb_encode(image):
    """With no edits the output is just the sRGB encoding of the linear input."""
    out = Pipeline().render_float(image, EditParams())
    expected = np.clip(adj.srgb_encode(image.linear_rgb), 0, 1)
    assert np.allclose(out, expected, atol=1e-4)


def test_process_returns_uint8_rgb(image):
    out = process(image, EditParams(exposure=0.5))
    assert out.dtype == np.uint8
    assert out.ndim == 3 and out.shape[2] == 3
    assert 0 <= out.min() and out.max() <= 255


def test_exposure_brightens_output(image):
    base = process(image, EditParams()).astype(np.int32)
    brighter = process(image, EditParams(exposure=1.0)).astype(np.int32)
    assert brighter.mean() > base.mean()


def test_rotation_swaps_dimensions(image):
    out = process(image, EditParams(rotation=90))
    h, w = image.linear_rgb.shape[:2]
    assert out.shape[:2] == (w, h)


def test_grayscale_saturation(image):
    out = process(image, EditParams(saturation=-100))
    assert np.allclose(out[..., 0], out[..., 1], atol=1)
    assert np.allclose(out[..., 1], out[..., 2], atol=1)


def test_histogram_counts_all_pixels(image):
    out = process(image, EditParams())
    hist = compute_histogram(out)
    assert hist.shape == (3, 256)
    n = out.shape[0] * out.shape[1]
    for c in range(3):
        assert hist[c].sum() == n


def test_pipeline_is_deterministic(image):
    p = EditParams(exposure=0.3, contrast=15, vibrance=25, sharpen=20)
    assert np.array_equal(process(image, p), process(image, p))


def _percentiles(rgb_uint8):
    lum = rgb_uint8.astype(np.float32) @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    return np.percentile(lum, [1, 50, 99])


def test_auto_tone_improves_hazy_image():
    """Auto tone should expand a washed-out image's tonal range end-to-end."""
    rng = np.random.default_rng(5)
    disp = 0.4 + 0.5 * rng.random((80, 80, 3)).astype(np.float32)  # milky, low contrast
    hazy = RawImage(adj.srgb_decode(disp))

    before = process(hazy, EditParams())
    after = process(hazy, EditParams.from_dict(adj.auto_tone(hazy.linear_rgb)))

    lo_b, _, hi_b = _percentiles(before)
    lo_a, _, hi_a = _percentiles(after)
    # Deeper blacks and a wider overall spread => more contrast, less haze.
    assert lo_a < lo_b
    assert (hi_a - lo_a) > (hi_b - lo_b)


def test_auto_tone_brightens_dark_image():
    dark = RawImage(adj.srgb_decode(np.full((64, 64, 3), 0.05, np.float32)))
    before = process(dark, EditParams())
    after = process(dark, EditParams.from_dict(adj.auto_tone(dark.linear_rgb)))
    assert after.mean() > before.mean()


def test_auto_wb_neutralises_grey_scene():
    """Applying the suggested WB to a colour-cast neutral scene evens the channels."""
    cast = RawImage(np.full((48, 48, 3), 0.3, np.float32) * np.array([1.3, 1.0, 0.7], np.float32))
    temp, tint = adj.auto_white_balance(cast.linear_rgb)
    corrected = process(cast, EditParams(temperature=temp, tint=tint))
    channel_means = corrected.reshape(-1, 3).mean(axis=0)
    spread_before = 0.3 * (1.3 - 0.7)  # original R-B gap in linear
    spread_after = (channel_means.max() - channel_means.min()) / 255.0
    assert spread_after < spread_before


def test_preview_and_full_render_are_consistent(image):
    """A downscaled preview should look like the full render, just smaller."""
    p = EditParams(exposure=0.4, contrast=20, saturation=10)
    preview = process(image.downscaled(40), p)
    full = process(image, p)
    assert preview.shape[0] <= full.shape[0]
    # Global brightness should be close between preview and full.
    assert abs(int(preview.mean()) - int(full.mean())) < 12
