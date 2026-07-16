"""A small RGB histogram widget drawn with ``QPainter``."""

from __future__ import annotations

from typing import Optional

import numpy as np
from PySide6.QtCore import Qt, QPointF
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QWidget


class HistogramWidget(QWidget):
    """Draws per-channel histograms as translucent filled curves."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._hist: Optional[np.ndarray] = None
        self.setMinimumHeight(110)
        self.setMaximumHeight(150)

    def set_histogram(self, hist: Optional[np.ndarray]) -> None:
        self._hist = hist
        self.update()

    def paintEvent(self, event):  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect().adjusted(1, 1, -1, -1)
        painter.fillRect(self.rect(), QColor(24, 24, 26))

        # Grid lines (quarters) for orientation.
        painter.setPen(QPen(QColor(60, 60, 66), 1))
        for i in range(1, 4):
            x = rect.left() + rect.width() * i / 4
            painter.drawLine(int(x), rect.top(), int(x), rect.bottom())

        if self._hist is None or self._hist.size == 0:
            painter.setPen(QColor(130, 130, 138))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "No image")
            painter.end()
            return

        hist = self._hist.astype(np.float64)
        # Clip the very top so a single spike (e.g. pure black) doesn't flatten
        # everything else; scale to the 99.5th percentile of counts.
        scale = np.percentile(hist, 99.5)
        if scale <= 0:
            scale = hist.max() or 1.0
        norm = np.clip(hist / scale, 0.0, 1.0)

        bins = norm.shape[1]
        colours = [QColor(230, 80, 80), QColor(90, 200, 110), QColor(90, 150, 235)]
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Plus)
        for c in range(3):
            path = QPainterPath()
            path.moveTo(rect.left(), rect.bottom())
            for i in range(bins):
                x = rect.left() + rect.width() * i / (bins - 1)
                y = rect.bottom() - norm[c, i] * rect.height()
                path.lineTo(QPointF(x, y))
            path.lineTo(rect.right(), rect.bottom())
            path.closeSubpath()
            fill = QColor(colours[c])
            fill.setAlpha(120)
            painter.fillPath(path, fill)
        painter.end()
