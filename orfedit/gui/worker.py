"""Background threads so the UI never blocks on heavy NumPy / LibRaw work."""

from __future__ import annotations

from PySide6.QtCore import QObject, QThread, Signal, Slot

from ..core.image import RawImage
from ..core.loader import load_orf
from ..core.params import EditParams
from ..core.pipeline import Pipeline, compute_histogram


class ProcessingWorker(QObject):
    """Runs the edit pipeline off the GUI thread.

    Requests are tagged with a monotonically increasing id; the main window
    ignores results older than the newest request it issued, so a burst of
    slider moves never shows a stale frame.
    """

    ready = Signal(int, object, object)  # request_id, rgb_uint8, histogram

    def __init__(self):
        super().__init__()
        self._pipeline = Pipeline()

    @Slot(int, object, object)
    def process(self, request_id: int, image: RawImage, params: EditParams) -> None:
        rgb = self._pipeline.render_uint8(image, params)
        hist = compute_histogram(rgb)
        self.ready.emit(request_id, rgb, hist)


class LoaderThread(QThread):
    """Decodes a raw file on a worker thread and reports success or failure."""

    loaded = Signal(object)      # RawImage
    failed = Signal(str)         # error message

    def __init__(self, path: str, demosaic: str = "AHD", parent=None):
        super().__init__(parent)
        self._path = path
        self._demosaic = demosaic

    def run(self) -> None:  # noqa: D401
        try:
            image = load_orf(self._path, demosaic=self._demosaic)
        except Exception as exc:  # surface any decode error to the UI
            self.failed.emit(str(exc))
            return
        self.loaded.emit(image)
