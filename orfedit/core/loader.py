"""Load Olympus ORF (and other LibRaw-supported) raw files.

Decoding is delegated to ``rawpy`` (a binding for LibRaw).  We ask LibRaw for a
*linear*, full-bit-depth, camera-white-balanced RGB image and hand it to the
pipeline for all further, user-controllable adjustments.  Doing white balance,
tone and colour work ourselves -- rather than letting LibRaw bake them in --
is what makes the editing non-destructive.
"""

from __future__ import annotations

import os
from typing import Dict

import numpy as np

from .image import RawImage

# ORF is the Olympus / OM System raw format.  We accept a few sibling formats
# that LibRaw decodes identically, so the app is useful beyond a single vendor.
SUPPORTED_EXTENSIONS = (".orf", ".raw", ".dng", ".rw2", ".nef", ".cr2", ".cr3", ".arw")


def is_orf(path: str) -> bool:
    """True if ``path`` looks like an Olympus ORF file by extension."""
    return os.path.splitext(path)[1].lower() == ".orf"


def _is_supported(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in SUPPORTED_EXTENSIONS


def load_orf(path: str, demosaic: str = "AHD") -> RawImage:
    """Decode a raw file at ``path`` into a linear-light :class:`RawImage`.

    Parameters
    ----------
    path:
        Filesystem path to the ORF (or other LibRaw-supported) file.
    demosaic:
        Demosaic algorithm name (see :func:`available_demosaic_algorithms`).

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If the file cannot be decoded as a raw image.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    try:
        import rawpy
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "rawpy is required to load raw files. Install it with "
            "`pip install rawpy`."
        ) from exc

    algo = _resolve_demosaic(demosaic)

    try:
        with rawpy.imread(path) as raw:
            rgb16 = raw.postprocess(
                gamma=(1, 1),              # linear output; we apply tone curves
                no_auto_bright=True,       # keep exposure under our control
                use_camera_wb=True,        # neutral starting white balance
                output_bps=16,
                output_color=rawpy.ColorSpace.sRGB,
                demosaic_algorithm=algo,
            )
            metadata = _extract_metadata(raw, path, rgb16.shape)
            thumb = _extract_thumbnail(raw)
    except Exception as exc:  # rawpy raises a variety of errors
        raise ValueError(f"Could not decode raw file {path!r}: {exc}") from exc

    linear = rgb16.astype(np.float32) / 65535.0
    return RawImage(linear, metadata=metadata, source_path=path, thumbnail=thumb)


def available_demosaic_algorithms() -> Dict[str, object]:
    """Map of friendly name -> ``rawpy.DemosaicAlgorithm`` for the UI.

    Only algorithms compiled into the installed LibRaw are returned.
    """
    try:
        import rawpy
    except ImportError:  # pragma: no cover
        return {}
    wanted = ["LINEAR", "VNG", "PPG", "AHD", "DCB", "DHT", "AAHD"]
    out: Dict[str, object] = {}
    for name in wanted:
        member = getattr(rawpy.DemosaicAlgorithm, name, None)
        if member is None:
            continue
        # not_supported() tells us whether this build of LibRaw includes it.
        try:
            if member.not_supported:
                continue
        except Exception:
            pass
        out[name] = member
    return out


def _resolve_demosaic(name: str):
    algos = available_demosaic_algorithms()
    if name in algos:
        return algos[name]
    if "AHD" in algos:
        return algos["AHD"]
    return next(iter(algos.values()), None)


def _extract_metadata(raw, path: str, shape) -> Dict[str, str]:
    meta: Dict[str, str] = {"File": os.path.basename(path)}
    h, w = shape[0], shape[1]
    meta["Dimensions"] = f"{w} x {h}"
    try:
        meta["Camera"] = f"{_clean(raw.camera_manufacturer)}".strip() or "Unknown"
    except Exception:
        pass
    try:
        meta["Model"] = _clean(raw.model) or "Unknown"
    except Exception:
        pass

    # rawpy exposes many capture fields via ``raw.metadata`` when built with it;
    # fall back gracefully when attributes are missing.
    try:
        m = raw.metadata  # type: ignore[attr-defined]
    except Exception:
        m = None
    if m is not None:
        iso = getattr(m, "iso_speed", None)
        if iso:
            meta["ISO"] = str(int(iso))
        shutter = getattr(m, "shutter", None)
        if shutter:
            meta["Shutter"] = _format_shutter(shutter)
        aperture = getattr(m, "aperture", None)
        if aperture:
            meta["Aperture"] = f"f/{aperture:.1f}"
        focal = getattr(m, "focal_len", None)
        if focal:
            meta["Focal length"] = f"{focal:.0f} mm"
    return meta


def _extract_thumbnail(raw):
    try:
        thumb = raw.extract_thumb()
    except Exception:
        return None
    try:
        import rawpy
        from io import BytesIO
        from PIL import Image

        if thumb.format == rawpy.ThumbFormat.JPEG:
            img = Image.open(BytesIO(thumb.data)).convert("RGB")
            return np.asarray(img, dtype=np.uint8)
        if thumb.format == rawpy.ThumbFormat.BITMAP:
            return np.asarray(thumb.data, dtype=np.uint8)
    except Exception:
        return None
    return None


def _format_shutter(seconds: float) -> str:
    if seconds <= 0:
        return "?"
    if seconds >= 1:
        return f"{seconds:.1f} s"
    return f"1/{round(1.0 / seconds)} s"


def _clean(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", "ignore").strip("\x00 ").strip()
    return str(value).strip("\x00 ").strip()
