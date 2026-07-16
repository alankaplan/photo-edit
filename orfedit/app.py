"""Application entry point: create the Qt app and show the main window."""

from __future__ import annotations

import sys
from typing import List, Optional

from PySide6.QtWidgets import QApplication

from .gui.main_window import MainWindow

_DARK_QSS = """
QMainWindow, QWidget { background-color: #202024; color: #e6e6ea; }
QToolBar { background: #26262b; border: 0; spacing: 4px; padding: 3px; }
QDockWidget { titlebar-close-icon: none; }
QDockWidget::title { background: #26262b; padding: 5px; }
QGroupBox {
    border: 1px solid #3a3a42; border-radius: 6px; margin-top: 10px; padding: 6px;
    font-weight: 600;
}
QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 3px; color: #b9b9c4; }
QPushButton {
    background: #34343c; border: 1px solid #45454f; border-radius: 5px; padding: 6px 10px;
}
QPushButton:hover { background: #3e3e48; }
QPushButton:disabled { color: #6a6a72; }
QComboBox { background: #34343c; border: 1px solid #45454f; border-radius: 4px; padding: 2px 6px; }
QSlider::groove:horizontal { height: 4px; background: #3a3a42; border-radius: 2px; }
QSlider::handle:horizontal {
    background: #d4d4dc; width: 14px; margin: -6px 0; border-radius: 7px;
}
QSlider::handle:horizontal:hover { background: #ffffff; }
QMenuBar, QMenu { background: #26262b; color: #e6e6ea; }
QMenu::item:selected, QMenuBar::item:selected { background: #3a3a44; }
QStatusBar { background: #26262b; color: #b9b9c4; }
QScrollArea { border: 0; }
"""


def build_app(argv: Optional[List[str]] = None) -> QApplication:
    """Return a configured :class:`QApplication` (creating one if needed)."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(argv if argv is not None else sys.argv)
    app.setApplicationName("ORF Photo Editor")
    app.setOrganizationName("orfedit")
    app.setStyle("Fusion")
    app.setStyleSheet(_DARK_QSS)
    return app


def main(argv: Optional[List[str]] = None) -> int:
    """Launch the GUI.  Optionally accepts a raw file path as the first arg."""
    argv = list(sys.argv if argv is None else argv)
    app = build_app(argv)
    window = MainWindow()
    window.show()

    # Allow `python -m orfedit <file.orf>` to open a file on startup.
    file_args = [a for a in argv[1:] if not a.startswith("-")]
    if file_args:
        window.load_path(file_args[0])

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
