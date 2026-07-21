"""Zoomable / pannable image display built on ``QGraphicsView``."""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QRect, QRectF, Qt, Signal
from PySide6.QtGui import QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
    QRubberBand,
)


class ImageViewer(QGraphicsView):
    """Displays a single image with fit-to-window, wheel zoom and drag-to-pan.

    A "region select" mode lets the user rubber-band a rectangle over the image;
    on release it emits :attr:`region_selected` with the selection as a
    normalized ``(x0, y0, x1, y1)`` box (fractions of the image), which the app
    uses for spot metering and eyedropper white balance.
    """

    zoom_changed = Signal(float)
    region_selected = Signal(float, float, float, float)

    MIN_SCALE = 0.02
    MAX_SCALE = 40.0

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._item: Optional[QGraphicsPixmapItem] = None
        self._fit_on_next = True

        self._select_mode = False
        self._rubber_band: Optional[QRubberBand] = None
        self._band_origin = None
        self._region_item: Optional[QGraphicsRectItem] = None

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
        self.clear_region()
        self._fit_on_next = True

    def has_image(self) -> bool:
        return self._item is not None

    # -- region selection ---------------------------------------------------
    def set_select_mode(self, enabled: bool) -> None:
        """Toggle rubber-band region selection (disables pan while active)."""
        self._select_mode = enabled
        self.setDragMode(
            QGraphicsView.DragMode.NoDrag
            if enabled
            else QGraphicsView.DragMode.ScrollHandDrag
        )
        self.setCursor(
            Qt.CursorShape.CrossCursor if enabled else Qt.CursorShape.ArrowCursor
        )

    def clear_region(self) -> None:
        if self._region_item is not None:
            self._scene.removeItem(self._region_item)
            self._region_item = None

    def _show_region_overlay(self, rect_scene: QRectF) -> None:
        self.clear_region()
        pen = QPen(Qt.GlobalColor.yellow)
        pen.setCosmetic(True)
        pen.setWidth(2)
        self._region_item = self._scene.addRect(rect_scene, pen)

    def _emit_region(self, view_rect: QRect) -> None:
        if self._item is None:
            return
        # Map the view-space rectangle into image (scene) pixel coordinates.
        scene_rect = self.mapToScene(view_rect).boundingRect()
        img_rect = scene_rect.intersected(self._item.boundingRect())
        w = self._item.boundingRect().width()
        h = self._item.boundingRect().height()
        if img_rect.width() < 2 or img_rect.height() < 2 or w <= 0 or h <= 0:
            return
        self._show_region_overlay(img_rect)
        self.region_selected.emit(
            img_rect.left() / w,
            img_rect.top() / h,
            img_rect.right() / w,
            img_rect.bottom() / h,
        )

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
    def mousePressEvent(self, event):  # noqa: N802
        if (
            self._select_mode
            and self._item is not None
            and event.button() == Qt.MouseButton.LeftButton
        ):
            self._band_origin = event.position().toPoint()
            if self._rubber_band is None:
                self._rubber_band = QRubberBand(QRubberBand.Shape.Rectangle, self.viewport())
            self._rubber_band.setGeometry(QRect(self._band_origin, self._band_origin))
            self._rubber_band.show()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):  # noqa: N802
        if self._select_mode and self._band_origin is not None and self._rubber_band:
            self._rubber_band.setGeometry(
                QRect(self._band_origin, event.position().toPoint()).normalized()
            )
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):  # noqa: N802
        if self._select_mode and self._band_origin is not None and self._rubber_band:
            rect = self._rubber_band.geometry()
            self._rubber_band.hide()
            self._band_origin = None
            self._emit_region(rect)
            return
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        if self._item is None:
            return
        delta = event.angleDelta().y()
        if delta == 0:
            return
        self.zoom_by(1.25 if delta > 0 else 1 / 1.25)

    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event)
