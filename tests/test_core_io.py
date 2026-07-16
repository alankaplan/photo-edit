import os

import numpy as np
import pytest

from orfedit.core import (
    EditParams,
    RawImage,
    SUPPORTED_EXTENSIONS,
    export_image,
    is_orf,
    load_orf,
    synthetic_raw,
)
from orfedit.core.export import SUPPORTED_EXPORT_EXTENSIONS


# -- params -----------------------------------------------------------------
def test_params_default_is_noop():
    assert EditParams().is_default()
    assert not EditParams(exposure=1).is_default()


def test_params_dict_roundtrip():
    p = EditParams(exposure=1.5, contrast=30, rotation=90, flip_horizontal=True)
    assert EditParams.from_dict(p.to_dict()) == p


def test_params_from_dict_ignores_unknown_keys():
    p = EditParams.from_dict({"exposure": 2.0, "bogus": 123})
    assert p.exposure == 2.0


# -- RawImage ---------------------------------------------------------------
def test_rawimage_rejects_bad_shape():
    with pytest.raises(ValueError):
        RawImage(np.zeros((4, 4)))  # missing channel axis


def test_rawimage_size_and_megapixels():
    img = synthetic_raw(200, 100)
    assert img.size == (200, 100)
    assert round(img.megapixels, 3) == round(200 * 100 / 1e6, 3)


def test_downscaled_limits_longest_edge():
    img = synthetic_raw(400, 200)
    small = img.downscaled(100)
    assert max(small.size) <= 100
    # No upscaling when already small enough.
    assert img.downscaled(10_000).size == img.size


def test_synthetic_has_dark_and_bright_regions():
    img = synthetic_raw(200, 150)
    lum = img.linear_rgb.mean(axis=2)
    assert lum.min() < 0.1        # a genuine shadow region exists
    assert lum.max() > 0.9        # and a genuine highlight region


# -- loader -----------------------------------------------------------------
def test_is_orf():
    assert is_orf("photo.ORF")
    assert is_orf("/a/b/c.orf")
    assert not is_orf("photo.jpg")


def test_supported_extensions_includes_orf():
    assert ".orf" in SUPPORTED_EXTENSIONS


def test_load_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_orf("/does/not/exist.orf")


def test_load_non_raw_file_raises(tmp_path):
    junk = tmp_path / "not_a_raw.orf"
    junk.write_bytes(b"this is not a raw file")
    with pytest.raises(ValueError):
        load_orf(str(junk))


# -- export -----------------------------------------------------------------
@pytest.mark.parametrize("ext", [".jpg", ".png", ".tif"])
def test_export_writes_file(tmp_path, ext):
    img = synthetic_raw(64, 48)
    out = tmp_path / f"result{ext}"
    path = export_image(img, EditParams(exposure=0.3), str(out))
    assert os.path.exists(path)
    assert os.path.getsize(path) > 0


def test_export_dimensions_match_render(tmp_path):
    from PIL import Image

    img = synthetic_raw(80, 60)
    out = tmp_path / "r.png"
    export_image(img, EditParams(rotation=90), str(out))
    with Image.open(out) as pil:
        # rotation swaps width/height
        assert pil.size == (60, 80)


def test_export_unsupported_extension_raises(tmp_path):
    img = synthetic_raw(32, 32)
    with pytest.raises(ValueError):
        export_image(img, EditParams(), str(tmp_path / "x.gif"))


def test_export_extensions_advertised():
    assert ".jpg" in SUPPORTED_EXPORT_EXTENSIONS
    assert ".png" in SUPPORTED_EXPORT_EXTENSIONS
