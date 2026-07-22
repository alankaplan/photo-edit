import numpy as np
import pytest

from orfedit.core import adjustments as adj


@pytest.fixture
def img():
    rng = np.random.default_rng(1)
    return rng.random((16, 24, 3), dtype=np.float32) * 0.9 + 0.05


def test_srgb_roundtrip(img):
    back = adj.srgb_decode(adj.srgb_encode(img))
    assert np.allclose(back, img, atol=1e-4)


def test_srgb_encode_monotonic():
    x = np.linspace(0, 1, 50, dtype=np.float32).reshape(-1, 1, 1) * np.ones((1, 1, 3), np.float32)
    enc = adj.srgb_encode(x)[:, 0, 0]
    assert np.all(np.diff(enc) >= -1e-6)


def test_exposure_doubles_light(img):
    assert np.allclose(adj.exposure(img, 1.0), img * 2.0)
    assert np.allclose(adj.exposure(img, -1.0), img * 0.5)
    assert adj.exposure(img, 0.0) is img  # no-op short circuit


def test_white_balance_warm_boosts_red_cuts_blue(img):
    warm = adj.white_balance(img, temperature=100, tint=0)
    assert warm[..., 0].mean() > img[..., 0].mean()   # red up
    assert warm[..., 2].mean() < img[..., 2].mean()   # blue down


def test_white_balance_noop(img):
    assert adj.white_balance(img, 0, 0) is img


def test_saturation_minus_100_is_grayscale(img):
    gray = adj.saturation(img, -100)
    assert np.allclose(gray[..., 0], gray[..., 1], atol=1e-5)
    assert np.allclose(gray[..., 1], gray[..., 2], atol=1e-5)


def test_saturation_increases_spread(img):
    more = adj.saturation(img, 80)
    spread_before = (img.max(axis=2) - img.min(axis=2)).mean()
    spread_after = (more.max(axis=2) - more.min(axis=2)).mean()
    assert spread_after > spread_before


def test_contrast_pivots_around_mid():
    flat = np.full((4, 4, 3), 0.5, np.float32)
    assert np.allclose(adj.contrast(flat, 100), 0.5)  # mid grey is the pivot
    darker = adj.contrast(np.full((4, 4, 3), 0.2, np.float32), 50)
    assert darker.mean() < 0.2


def test_brightness_shifts_up(img):
    assert adj.brightness(img, 100).mean() > img.mean()


def test_highlights_recovery_darkens_bright_region():
    bright = np.full((8, 8, 3), 0.95, np.float32)
    recovered = adj.highlights_shadows(bright, highlights=-100, shadows=0)
    assert recovered.mean() < bright.mean()


def test_shadows_lift_brightens_dark_region():
    dark = np.full((8, 8, 3), 0.05, np.float32)
    lifted = adj.highlights_shadows(dark, highlights=0, shadows=100)
    assert lifted.mean() > dark.mean()


def test_white_black_point_noop(img):
    assert adj.white_black_point(img, 0, 0) is img


def test_vibrance_spares_saturated_pixels():
    # A less-saturated pixel and a highly-saturated pixel.
    px = np.array([[[0.6, 0.4, 0.4], [0.9, 0.1, 0.1]]], np.float32)
    out = adj.vibrance(px, 100)
    sat_before = px.max(axis=2) - px.min(axis=2)
    sat_after = out.max(axis=2) - out.min(axis=2)
    # Vibrance should raise the *relative* saturation of the less-saturated
    # pixel more than that of the already-saturated one.
    ratio_low = sat_after[0, 0] / sat_before[0, 0]
    ratio_high = sat_after[0, 1] / sat_before[0, 1]
    assert ratio_low > ratio_high


def test_gaussian_blur_preserves_mean(img):
    blurred = adj.gaussian_blur(img, sigma=1.5)
    # Reflect padding keeps the mean close but not exact near small borders.
    assert np.allclose(blurred.mean(), img.mean(), atol=5e-3)
    assert blurred.shape == img.shape


def test_unsharp_mask_increases_local_variance():
    x = np.zeros((16, 16, 3), np.float32)
    x[:, 8:] = 1.0  # a hard edge
    sharp = adj.unsharp_mask(x, 100)
    # Sharpening should overshoot at the edge, exceeding the original max.
    assert sharp.max() > 1.0


def test_orient_rotation_shape():
    x = np.zeros((10, 20, 3), np.float32)
    r90 = adj.orient(x, 90, False, False)
    assert r90.shape == (20, 10, 3)
    r180 = adj.orient(x, 180, False, False)
    assert r180.shape == (10, 20, 3)


def test_orient_flip():
    x = np.arange(2 * 3 * 3, dtype=np.float32).reshape(2, 3, 3)
    fh = adj.orient(x, 0, True, False)
    assert np.array_equal(fh, x[:, ::-1])
    fv = adj.orient(x, 0, False, True)
    assert np.array_equal(fv, x[::-1, :])


def test_auto_exposure_targets_midtone():
    dark = np.full((32, 32, 3), 0.05, np.float32)
    ev = adj.auto_exposure_ev(dark, target=0.18)
    assert ev > 0  # a dark image should be brightened
    # Applying the suggested exposure moves the median toward the target.
    boosted = adj.exposure(dark, ev)
    assert adj.luminance(boosted).mean() > adj.luminance(dark).mean()


def test_auto_white_balance_neutralises_cast():
    # Image with a strong red cast: grey-world should suggest cooling (temp<0).
    casted = np.full((16, 16, 3), 0.4, np.float32)
    casted[..., 0] = 0.8
    temp, tint = adj.auto_white_balance(casted)
    assert temp < 0


def test_auto_wb_corrects_blue_cast():
    img = np.full((40, 40, 3), 0.4, np.float32)
    img[..., 2] = 0.6  # blue-heavy -> auto WB should warm it (temp > 0)
    temp, _ = adj.auto_white_balance(img)
    assert temp > 0


def test_auto_wb_ignores_saturated_colours():
    # Mostly neutral grey (neutral illuminant) with a saturated green stripe.
    img = np.full((40, 40, 3), 0.3, np.float32)
    img[:, :10] = np.array([0.05, 0.6, 0.05], np.float32)
    temp, tint = adj.auto_white_balance(img)
    # The neutral pixels dominate the estimate, so the correction stays small
    # (a naive full-image grey-world would swing hard toward magenta here).
    assert abs(temp) < 15 and abs(tint) < 20


def test_auto_tone_returns_full_param_set():
    img = adj.srgb_decode(np.full((32, 32, 3), 0.5, np.float32))  # linear mid-grey
    tone = adj.auto_tone(img)
    assert set(tone) == {
        "exposure", "contrast", "highlights", "shadows", "whites", "blacks"
    }


def test_auto_exposure_brightens_backlit_subject():
    """Backlit group photo: near-blown sky, bright garden, shaded subjects.

    A global-average metering reads the frame as bright and darkens it, burying
    the subjects (the exact regression reported on the family photo).  Subject-
    weighted, highlight-protected metering must brighten instead.
    """
    lin = np.full((80, 80, 3), 0.45, np.float32)  # bright garden background
    lin[:25] = 0.95                                # near-blown sky band (top)
    lin[30:78, 8:72] = 0.08                         # shaded subjects fill the foreground
    assert adj.auto_exposure_ev(lin) > 0


def test_auto_exposure_spares_bright_subject_on_dark_surround():
    """Bright central subject on a dark surround must not be over-brightened.

    Mirrors the dog on the couch: keying off the subject avoids the washed-out
    over-exposure a global metering would apply to the dark room.
    """
    lin = np.full((80, 80, 3), 0.02, np.float32)  # dark surround
    lin[25:70, 15:65] = 0.28                        # bright central subject
    assert adj.auto_exposure_ev(lin) < 0.6


def test_spot_metering_exposes_for_selected_region():
    """Metering a dark corner brightens more than metering a bright corner."""
    lin = np.full((60, 60, 3), 0.5, np.float32)
    lin[:20, :20] = 0.04  # dark patch, top-left
    ev_dark = adj.auto_exposure_ev(lin, region=(0.0, 0.0, 0.33, 0.33))
    ev_bright = adj.auto_exposure_ev(lin, region=(0.66, 0.66, 1.0, 1.0))
    assert ev_dark > ev_bright
    assert ev_dark > 0        # expose the dark patch up
    assert ev_bright < 0.2    # the bright patch needs little/none


def test_spot_wb_neutralises_selected_patch():
    """WB from a region drives that patch's channels toward equal (neutral)."""
    lin = np.full((40, 40, 3), 0.4, np.float32)
    lin[:, :, 0] *= 1.4  # warm/red patch everywhere; select part of it
    temp, tint = adj.auto_white_balance(lin, region=(0.1, 0.1, 0.5, 0.5))
    assert temp < 0       # cool it to cancel the red
    # Applying the correction should even the region's channels.
    corrected = adj.white_balance(lin, temp, tint)
    patch = corrected[4:20, 4:20].reshape(-1, 3).mean(axis=0)
    assert (patch.max() - patch.min()) < (0.4 * 1.4 - 0.4)


def test_region_auto_tone_exposes_subject_brighter_than_midgrey():
    """Pointing at a dim subject brightens it more than neutral mid-grey would.

    This is the fix for "selecting the subject leaves it dark, but selecting a
    dark area works" -- a selected region is exposed as a subject, not as grey.
    """
    lin = np.full((60, 60, 3), 0.5, np.float32)
    lin[20:40, 20:40] = 0.10  # dim subject patch, centre
    region = (0.33, 0.33, 0.66, 0.66)
    ev_tone = adj.auto_tone(lin, region=region)["exposure"]
    ev_midgrey = adj.auto_exposure_ev(lin, region=region)  # neutral 0.13 target
    assert ev_tone > ev_midgrey


def test_region_resolves_and_clamps():
    sl = adj._resolve_region((100, 200), (0.25, 0.5, 0.75, 1.0))
    rows, cols = sl
    assert (rows.start, rows.stop) == (50, 100)
    assert (cols.start, cols.stop) == (50, 150)
    # Degenerate/zero-size region still yields at least one pixel.
    sl2 = adj._resolve_region((10, 10), (0.5, 0.5, 0.5, 0.5))
    assert sl2[0].stop > sl2[0].start and sl2[1].stop > sl2[1].start


def test_auto_tone_deepens_milky_blacks():
    """A washed-out, low-contrast frame should get its black point deepened."""
    rng = np.random.default_rng(0)
    disp = 0.35 + 0.55 * rng.random((64, 64, 3)).astype(np.float32)  # lifted blacks
    tone = adj.auto_tone(adj.srgb_decode(disp))
    assert tone["blacks"] < 0


def test_auto_tone_brightens_dark_scene():
    tone = adj.auto_tone(adj.srgb_decode(np.full((48, 48, 3), 0.03, np.float32)))
    assert tone["exposure"] > 0


def test_auto_tone_darkens_bright_scene():
    tone = adj.auto_tone(adj.srgb_decode(np.full((48, 48, 3), 0.85, np.float32)))
    assert tone["exposure"] < 0
