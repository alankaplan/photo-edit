"""Headless smoke test: build the window, load the demo image, apply edits and
save a screenshot of the whole UI.  Runs under ``QT_QPA_PLATFORM=offscreen``.

Usage::

    QT_QPA_PLATFORM=offscreen python scripts/render_demo.py [out.png]
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QCoreApplication, QEventLoop, QTimer  # noqa: E402

from orfedit.app import build_app  # noqa: E402
from orfedit.core.params import EditParams  # noqa: E402
from orfedit.gui.main_window import MainWindow  # noqa: E402


def _pump(ms: int) -> None:
    """Process the Qt event loop for ``ms`` milliseconds."""
    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec()


def main() -> int:
    out = sys.argv[1] if len(sys.argv) > 1 else "ui_demo.png"
    app = build_app([])
    window = MainWindow()
    window.resize(1280, 820)
    window.show()

    # Load the built-in synthetic image (no ORF file required).
    window._on_open_demo()
    _pump(300)  # let the background render worker deliver a frame

    # Apply a representative edit and let it re-render.
    window._controls.set_params(
        EditParams(
            exposure=0.6,
            contrast=25,
            highlights=-45,
            shadows=35,
            whites=8,
            blacks=-6,
            temperature=18,
            tint=-6,
            saturation=12,
            vibrance=20,
            sharpen=35,
        )
    )
    _pump(400)

    pixmap = window.grab()
    ok = pixmap.save(out)
    QCoreApplication.processEvents()

    has_frame = window._last_shown_id >= 0
    print(f"render worker delivered frame: {has_frame}")
    print(f"screenshot saved: {ok} -> {out} ({pixmap.width()}x{pixmap.height()})")
    window.close()
    return 0 if (ok and has_frame) else 1


if __name__ == "__main__":
    raise SystemExit(main())
