"""Zoomable / pannable image display built on ``QGraphicsView``."""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPainter, QPixmap
from PySide6.QtWidgets import (
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
)


class ImageViewer(QGraphicsView):
    """Displays a single image with fit-to-window, wheel zoom and drag-to-pan."""

    zoom_changed = Signal(float)

    MIN_SCALE = 0.02
    MAX_SCALE = 40.0

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._item: Optional[QGraphicsPixmapItem] = None
        self._fit_on_next = True

        self.setRenderHints(
            QPainter.RenderHint.SmoothPixmapTransform | QPainter.RenderHint.Antialiasing
        )
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setBackgroundBrush(Qt.GlobalColor.black)

    # -- image handling -----------------------------------------------------
    def set_image(self, image: QImage) -> None:
        """Show ``image``.  Preserves zoom/pan across live re-renders."""
        pixmap = QPixmap.fromImage(image)
        first = self._item is None
        size_changed = (
            self._item is not None
            and self._item.pixmap().size() != pixmap.size()
        )
        if self._item is None:
            self._item = QGraphicsPixmapItem(pixmap)
            self._item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
            self._scene.addItem(self._item)
        else:
            self._item.setPixmap(pixmap)
        self._scene.setSceneRect(self._item.boundingRect())
        if first or size_changed or self._fit_on_next:
            self.fit_to_window()
            self._fit_on_next = False

    def clear(self) -> None:
        if self._item is not None:
            self._scene.removeItem(self._item)
            self._item = None
        self._fit_on_next = True

    def has_image(self) -> bool:
        return self._item is not None

    # -- zoom ---------------------------------------------------------------
    def current_scale(self) -> float:
        return float(self.transform().m11())

    def fit_to_window(self) -> None:
        if self._item is None:
            return
        self.fitInView(self._item, Qt.AspectRatioMode.KeepAspectRatio)
        self.zoom_changed.emit(self.current_scale())

    def reset_zoom(self) -> None:
        """Show the image at 100% (one image pixel per screen pixel)."""
        if self._item is None:
            return
        self.setTransform(self.transform().fromScale(1.0, 1.0))
        self.zoom_changed.emit(self.current_scale())

    def zoom_by(self, factor: float) -> None:
        if self._item is None:
            return
        target = self.current_scale() * factor
        if target < self.MIN_SCALE or target > self.MAX_SCALE:
            return
        self.scale(factor, factor)
        self.zoom_changed.emit(self.current_scale())

    def refit_on_next_image(self) -> None:
        self._fit_on_next = True

    # -- events -------------------------------------------------------------
    def wheelEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        if self._item is None:
            return
        delta = event.angleDelta().y()
        if delta == 0:
            return
        self.zoom_by(1.25 if delta > 0 else 1 / 1.25)

    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event)
