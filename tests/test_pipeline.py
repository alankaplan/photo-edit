import numpy as np
import pytest

from orfedit.core import (
    EditParams,
    Pipeline,
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


def test_preview_and_full_render_are_consistent(image):
    """A downscaled preview should look like the full render, just smaller."""
    p = EditParams(exposure=0.4, contrast=20, saturation=10)
    preview = process(image.downscaled(40), p)
    full = process(image, p)
    assert preview.shape[0] <= full.shape[0]
    # Global brightness should be close between preview and full.
    assert abs(int(preview.mean()) - int(full.mean())) < 12
