"""PySide6 desktop interface for :mod:`orfedit.core`.

Importing this package pulls in PySide6.  If you only need the processing core
(e.g. for scripting or tests), import :mod:`orfedit.core` directly instead.
"""

from .main_window import MainWindow

__all__ = ["MainWindow"]
