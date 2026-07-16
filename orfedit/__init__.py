"""orfedit -- a raw image processing application for Olympus ORF files.

The package is split into two layers:

* :mod:`orfedit.core` -- a display-independent image processing core built on
  NumPy.  It can load ORF files (via ``rawpy``/LibRaw), apply a non-destructive
  edit pipeline and export the result.  It has no GUI dependency and is fully
  unit-testable.
* :mod:`orfedit.gui` -- a PySide6 desktop interface that drives the core.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
