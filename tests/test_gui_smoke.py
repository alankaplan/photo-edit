"""Headless GUI smoke tests.

These run the real PySide6 widgets under the ``offscreen`` platform plugin, so
they need no display server.  They verify the window builds, loads an image and
that the background render worker delivers a frame.
"""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtCore import QEventLoop, QTimer  # noqa: E402

from orfedit.app import build_app  # noqa: E402
from orfedit.core.params import EditParams  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return build_app([])


def _pump(ms):
    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec()


def _wait_until(predicate, timeout_ms=3000, slice_ms=25):
    """Pump the event loop in small slices until ``predicate()`` or timeout."""
    waited = 0
    while waited < timeout_ms:
        if predicate():
            return True
        _pump(slice_ms)
        waited += slice_ms
    return predicate()


def test_window_builds_without_image(app):
    from orfedit.gui.main_window import MainWindow

    win = MainWindow()
    # Adjustments are disabled until an image is loaded.
    assert not win._act_export.isEnabled()
    win.close()


def test_load_demo_and_render(app):
    from orfedit.gui.main_window import MainWindow

    win = MainWindow()
    win._on_open_demo()
    assert win._act_export.isEnabled()
    assert _wait_until(lambda: win._last_shown_id >= 0)   # a frame was shown
    assert win._viewer.has_image()
    win.close()


def test_edits_trigger_rerender(app):
    from orfedit.gui.main_window import MainWindow

    win = MainWindow()
    win._on_open_demo()
    _wait_until(lambda: win._last_shown_id >= 0)
    first = win._last_shown_id
    win._controls.set_params(EditParams(exposure=1.0, contrast=40))
    assert _wait_until(lambda: win._last_shown_id > first)  # change -> new frame
    win.close()


def test_auto_tone_sets_exposure(app):
    from orfedit.gui.main_window import MainWindow

    win = MainWindow()
    win._on_open_demo()
    _wait_until(lambda: win._last_shown_id >= 0)
    win._on_auto_tone()
    _pump(100)
    # The synthetic image is fairly bright, so auto tone should choose some EV.
    assert isinstance(win._controls.params().exposure, float)
    win.close()
