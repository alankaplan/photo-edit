"""The application's main window."""

from __future__ import annotations

import os
from typing import Optional

from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QComboBox,
    QDockWidget,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ..core import adjustments as adj
from ..core.export import SUPPORTED_EXPORT_EXTENSIONS
from ..core.image import RawImage, synthetic_raw
from ..core.loader import SUPPORTED_EXTENSIONS, available_demosaic_algorithms
from ..core.params import EditParams
from .controls import ControlPanel
from .histogram import HistogramWidget
from .image_viewer import ImageViewer
from .utils import ndarray_to_qimage
from .worker import LoaderThread, ProcessingWorker

# Longest edge of the interactive preview.  The full-resolution image is kept
# for export so quality is never compromised.
PREVIEW_MAX_EDGE = 1600


class MainWindow(QMainWindow):
    """Load an ORF, adjust it live, and export the result."""

    # Emitted to the processing worker (queued across the thread boundary).
    _request_render = Signal(int, object, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("ORF Photo Editor")
        self.resize(1280, 820)

        self._full_image: Optional[RawImage] = None
        self._preview_image: Optional[RawImage] = None
        self._request_id = 0
        self._last_shown_id = -1
        self._loader: Optional[LoaderThread] = None
        self._demosaic = "AHD"

        self._build_central()
        self._build_dock()
        self._build_actions()
        self._build_worker()

        # Debounce timer coalesces rapid slider movements into one render.
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(30)
        self._debounce.timeout.connect(self._render_preview)

        self._controls.params_changed.connect(self._debounce.start)
        self._viewer.zoom_changed.connect(self._on_zoom_changed)

        self._set_image_loaded(False)
        self.statusBar().showMessage("Open an ORF file to begin  (File → Open)")

    # ------------------------------------------------------------------ UI
    def _build_central(self) -> None:
        self._viewer = ImageViewer(self)
        self.setCentralWidget(self._viewer)

    def _build_dock(self) -> None:
        dock = QDockWidget("Adjustments", self)
        dock.setAllowedAreas(Qt.DockWidgetArea.RightDockWidgetArea | Qt.DockWidgetArea.LeftDockWidgetArea)
        dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )

        container = QWidget()
        outer = QVBoxLayout(container)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(8)

        self._histogram = HistogramWidget()
        outer.addWidget(self._histogram)

        self._meta_label = QLabel("")
        self._meta_label.setWordWrap(True)
        self._meta_label.setTextFormat(Qt.TextFormat.RichText)
        self._meta_label.setStyleSheet("color:#b8b8c0; font-size:11px;")
        outer.addWidget(self._meta_label)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._controls = ControlPanel()
        scroll.setWidget(self._controls)
        outer.addWidget(scroll, 1)

        self._reset_button = QPushButton("Reset all adjustments")
        self._reset_button.clicked.connect(self._on_reset_all)
        outer.addWidget(self._reset_button)

        dock.setWidget(container)
        dock.setMinimumWidth(300)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
        self._dock = dock

    def _build_actions(self) -> None:
        menubar = self.menuBar()
        toolbar = self.addToolBar("Main")
        toolbar.setMovable(False)

        file_menu = menubar.addMenu("&File")
        self._act_open = QAction("&Open ORF…", self)
        self._act_open.setShortcut(QKeySequence.StandardKey.Open)
        self._act_open.triggered.connect(self._on_open)
        file_menu.addAction(self._act_open)

        self._act_demo = QAction("Open &Demo Image", self)
        self._act_demo.triggered.connect(self._on_open_demo)
        file_menu.addAction(self._act_demo)

        file_menu.addSeparator()
        self._act_export = QAction("&Export…", self)
        self._act_export.setShortcut(QKeySequence.StandardKey.Save)
        self._act_export.triggered.connect(self._on_export)
        file_menu.addAction(self._act_export)

        file_menu.addSeparator()
        act_quit = QAction("&Quit", self)
        act_quit.setShortcut(QKeySequence.StandardKey.Quit)
        act_quit.triggered.connect(self.close)
        file_menu.addAction(act_quit)

        edit_menu = menubar.addMenu("&Edit")
        self._act_auto_tone = QAction("Auto &Tone", self)
        self._act_auto_tone.triggered.connect(self._on_auto_tone)
        edit_menu.addAction(self._act_auto_tone)
        self._act_auto_wb = QAction("Auto &White Balance", self)
        self._act_auto_wb.triggered.connect(self._on_auto_wb)
        edit_menu.addAction(self._act_auto_wb)
        edit_menu.addSeparator()
        self._act_rotate = QAction("Rotate 90° CW", self)
        self._act_rotate.triggered.connect(self._on_rotate)
        edit_menu.addAction(self._act_rotate)
        self._act_flip_h = QAction("Flip Horizontal", self)
        self._act_flip_h.triggered.connect(lambda: self._toggle_flip("flip_horizontal"))
        edit_menu.addAction(self._act_flip_h)
        self._act_flip_v = QAction("Flip Vertical", self)
        self._act_flip_v.triggered.connect(lambda: self._toggle_flip("flip_vertical"))
        edit_menu.addAction(self._act_flip_v)
        edit_menu.addSeparator()
        self._act_reset = QAction("&Reset All", self)
        self._act_reset.triggered.connect(self._on_reset_all)
        edit_menu.addAction(self._act_reset)

        view_menu = menubar.addMenu("&View")
        self._act_fit = QAction("&Fit to Window", self)
        self._act_fit.setShortcut("Ctrl+0")
        self._act_fit.triggered.connect(self._viewer.fit_to_window)
        view_menu.addAction(self._act_fit)
        self._act_100 = QAction("Zoom &100%", self)
        self._act_100.setShortcut("Ctrl+1")
        self._act_100.triggered.connect(self._viewer.reset_zoom)
        view_menu.addAction(self._act_100)
        self._act_zoom_in = QAction("Zoom &In", self)
        self._act_zoom_in.setShortcut(QKeySequence.StandardKey.ZoomIn)
        self._act_zoom_in.triggered.connect(lambda: self._viewer.zoom_by(1.25))
        view_menu.addAction(self._act_zoom_in)
        self._act_zoom_out = QAction("Zoom &Out", self)
        self._act_zoom_out.setShortcut(QKeySequence.StandardKey.ZoomOut)
        self._act_zoom_out.triggered.connect(lambda: self._viewer.zoom_by(1 / 1.25))
        view_menu.addAction(self._act_zoom_out)

        help_menu = menubar.addMenu("&Help")
        act_about = QAction("&About", self)
        act_about.triggered.connect(self._on_about)
        help_menu.addAction(act_about)

        # Toolbar mirrors the most-used actions plus the demosaic selector.
        toolbar.addAction(self._act_open)
        toolbar.addAction(self._act_export)
        toolbar.addSeparator()
        toolbar.addAction(self._act_auto_tone)
        toolbar.addAction(self._act_auto_wb)
        toolbar.addAction(self._act_reset)
        toolbar.addSeparator()
        toolbar.addAction(self._act_fit)
        toolbar.addAction(self._act_100)
        toolbar.addWidget(QLabel("  Demosaic: "))
        self._demosaic_combo = QComboBox()
        algos = list(available_demosaic_algorithms().keys()) or ["AHD"]
        self._demosaic_combo.addItems(algos)
        if self._demosaic in algos:
            self._demosaic_combo.setCurrentText(self._demosaic)
        self._demosaic_combo.currentTextChanged.connect(self._on_demosaic_changed)
        toolbar.addWidget(self._demosaic_combo)

    def _build_worker(self) -> None:
        self._thread = QThread(self)
        self._worker = ProcessingWorker()
        self._worker.moveToThread(self._thread)
        self._request_render.connect(self._worker.process)
        self._worker.ready.connect(self._on_render_ready)
        self._thread.start()

    # -------------------------------------------------------------- loading
    def _on_open(self) -> None:
        patterns = " ".join(f"*{ext}" for ext in SUPPORTED_EXTENSIONS)
        path, _ = QFileDialog.getOpenFileName(
            self, "Open raw image", "", f"Raw images ({patterns});;All files (*)"
        )
        if path:
            self.load_path(path)

    def load_path(self, path: str) -> None:
        """Begin decoding ``path`` on a background thread."""
        if not os.path.exists(path):
            QMessageBox.warning(self, "Open", f"File not found:\n{path}")
            return
        self._act_open.setEnabled(False)
        self.statusBar().showMessage(f"Loading {os.path.basename(path)}…")
        self._loader = LoaderThread(path, demosaic=self._demosaic)
        self._loader.loaded.connect(self._on_loaded)
        self._loader.failed.connect(self._on_load_failed)
        self._loader.finished.connect(lambda: self._act_open.setEnabled(True))
        self._loader.start()

    def _on_loaded(self, image: RawImage) -> None:
        self.set_image(image)
        name = os.path.basename(image.source_path or "image")
        self.statusBar().showMessage(f"Loaded {name}  ({image.megapixels:.1f} MP)", 5000)

    def _on_load_failed(self, message: str) -> None:
        self.statusBar().showMessage("Load failed", 4000)
        QMessageBox.critical(self, "Could not open file", message)

    def _on_open_demo(self) -> None:
        self.set_image(synthetic_raw(1200, 800))
        self.statusBar().showMessage("Loaded synthetic demo image", 4000)

    def set_image(self, image: RawImage) -> None:
        """Adopt ``image`` as the current photo and render it."""
        self._full_image = image
        self._preview_image = image.downscaled(PREVIEW_MAX_EDGE)
        self._controls.blockSignals(True)
        self._controls.reset_all()
        self._controls.blockSignals(False)
        self._viewer.refit_on_next_image()
        self._update_metadata(image)
        self._set_image_loaded(True)
        self._render_preview()

    # ------------------------------------------------------------ rendering
    def _render_preview(self) -> None:
        if self._preview_image is None:
            return
        self._request_id += 1
        self._request_render.emit(
            self._request_id, self._preview_image, self._controls.params()
        )

    def _on_render_ready(self, request_id: int, rgb, hist) -> None:
        if request_id < self._last_shown_id:
            return  # a newer frame already superseded this one
        self._last_shown_id = request_id
        self._viewer.set_image(ndarray_to_qimage(rgb))
        self._histogram.set_histogram(hist)

    # ----------------------------------------------------------- edit tools
    def _on_reset_all(self) -> None:
        self._controls.reset_all()

    def _on_auto_tone(self) -> None:
        if self._preview_image is None:
            return
        ev = adj.auto_exposure_ev(self._preview_image.linear_rgb)
        self._controls.update_field("exposure", round(ev, 2))
        self.statusBar().showMessage(f"Auto tone: exposure {ev:+.2f} EV", 3000)

    def _on_auto_wb(self) -> None:
        if self._preview_image is None:
            return
        temperature, tint = adj.auto_white_balance(self._preview_image.linear_rgb)
        self._controls.update_field("tint", round(tint))
        self._controls.update_field("temperature", round(temperature))
        self.statusBar().showMessage(
            f"Auto white balance: temp {temperature:+.0f}, tint {tint:+.0f}", 3000
        )

    def _on_rotate(self) -> None:
        params = self._controls.params()
        self._controls.update_field("rotation", (params.rotation + 90) % 360)
        self._viewer.refit_on_next_image()

    def _toggle_flip(self, field: str) -> None:
        params = self._controls.params()
        self._controls.update_field(field, not getattr(params, field))

    def _on_demosaic_changed(self, name: str) -> None:
        self._demosaic = name
        # Re-decode the current file at the new demosaic quality, if any.
        if self._full_image is not None and self._full_image.source_path:
            self.load_path(self._full_image.source_path)

    # --------------------------------------------------------------- export
    def _on_export(self) -> None:
        if self._full_image is None:
            return
        default_name = "export.jpg"
        if self._full_image.source_path:
            stem = os.path.splitext(os.path.basename(self._full_image.source_path))[0]
            default_name = f"{stem}_edited.jpg"
        patterns = ";;".join(
            {
                ".jpg": "JPEG image (*.jpg *.jpeg)",
                ".png": "PNG image (*.png)",
                ".tif": "TIFF image (*.tif *.tiff)",
            }[ext]
            for ext in (".jpg", ".png", ".tif")
        )
        path, _ = QFileDialog.getSaveFileName(self, "Export image", default_name, patterns)
        if not path:
            return
        if os.path.splitext(path)[1].lower() not in SUPPORTED_EXPORT_EXTENSIONS:
            path += ".jpg"
        try:
            from ..core.export import export_image

            export_image(self._full_image, self._controls.params(), path)
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))
            return
        self.statusBar().showMessage(f"Exported {os.path.basename(path)}", 5000)

    # -------------------------------------------------------------- helpers
    def _update_metadata(self, image: RawImage) -> None:
        if not image.metadata:
            self._meta_label.setText("")
            return
        rows = "".join(
            f"<b>{k}:</b> {v}<br>" for k, v in image.metadata.items()
        )
        self._meta_label.setText(rows)

    def _set_image_loaded(self, loaded: bool) -> None:
        for widget in (
            self._controls,
            self._reset_button,
            self._act_export,
            self._act_reset,
            self._act_auto_tone,
            self._act_auto_wb,
            self._act_rotate,
            self._act_flip_h,
            self._act_flip_v,
            self._act_fit,
            self._act_100,
            self._act_zoom_in,
            self._act_zoom_out,
        ):
            widget.setEnabled(loaded)
        if not loaded:
            self._viewer.clear()
            self._histogram.set_histogram(None)
            self._meta_label.setText("")

    def _on_zoom_changed(self, scale: float) -> None:
        if self._viewer.has_image():
            self.statusBar().showMessage(f"Zoom: {scale * 100:.0f}%", 2000)

    def _on_about(self) -> None:
        QMessageBox.about(
            self,
            "About ORF Photo Editor",
            "<h3>ORF Photo Editor</h3>"
            "<p>A raw image processing tool for Olympus / OM System ORF files.</p>"
            "<p>Load a raw file, adjust exposure, colour, tone and detail "
            "non-destructively, then export to JPEG, PNG or TIFF.</p>"
            "<p>Built with NumPy, rawpy (LibRaw) and PySide6.</p>",
        )

    def closeEvent(self, event):  # noqa: N802
        self._thread.quit()
        self._thread.wait(2000)
        super().closeEvent(event)
