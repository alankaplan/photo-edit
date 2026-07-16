"""The adjustment panel: one slider per :class:`EditParams` field.

The panel builds itself from :data:`orfedit.core.params.SLIDER_SPECS`, so adding
a new adjustment is a one-line change in the core plus its pipeline stage -- the
UI follows automatically.
"""

from __future__ import annotations

from typing import Dict, List

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from ..core.params import EditParams, SLIDER_SPECS


class LabeledSlider(QWidget):
    """A named slider with a live value read-out and double-click reset.

    Works in integer "ticks" internally (``QSlider`` is integer only) and maps
    to/from the real float value using the spec's ``step``.
    """

    value_changed = Signal(str, float)

    def __init__(self, name, label, minimum, maximum, step, default, decimals, parent=None):
        super().__init__(parent)
        self.name = name
        self.step = float(step)
        self.decimals = int(decimals)
        self._default = float(default)

        self._min_tick = round(minimum / self.step)
        self._max_tick = round(maximum / self.step)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(2)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        self._name_label = QLabel(label)
        self._value_label = QLabel(self._format(self._default))
        self._value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._value_label.setMinimumWidth(46)
        self._value_label.setStyleSheet("color: #cfcfd6;")
        header.addWidget(self._name_label)
        header.addStretch(1)
        header.addWidget(self._value_label)
        layout.addLayout(header)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setMinimum(self._min_tick)
        self._slider.setMaximum(self._max_tick)
        self._slider.setValue(self._to_tick(self._default))
        self._slider.valueChanged.connect(self._on_slider)
        layout.addWidget(self._slider)

    # -- conversions --------------------------------------------------------
    def _to_tick(self, value: float) -> int:
        return int(round(value / self.step))

    def _from_tick(self, tick: int) -> float:
        return tick * self.step

    def _format(self, value: float) -> str:
        if self.decimals == 0:
            text = f"{int(round(value))}"
        else:
            text = f"{value:.{self.decimals}f}"
        # Show an explicit + for positive adjustments (nicer for +/- sliders).
        if self._min_tick < 0 and value > 0:
            text = "+" + text
        return text

    # -- events -------------------------------------------------------------
    def _on_slider(self, tick: int) -> None:
        value = self._from_tick(tick)
        self._value_label.setText(self._format(value))
        self.value_changed.emit(self.name, value)

    def mouseDoubleClickEvent(self, event):  # noqa: N802
        self.reset()

    # -- public api ---------------------------------------------------------
    def value(self) -> float:
        return self._from_tick(self._slider.value())

    def set_value(self, value: float) -> None:
        blocked = self._slider.blockSignals(True)
        self._slider.setValue(self._to_tick(value))
        self._slider.blockSignals(blocked)
        self._value_label.setText(self._format(value))

    def reset(self) -> None:
        self.set_value(self._default)
        self.value_changed.emit(self.name, self._default)


class ControlPanel(QWidget):
    """Groups every :class:`LabeledSlider` and reports edits as :class:`EditParams`."""

    params_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._sliders: Dict[str, LabeledSlider] = {}
        self._params = EditParams()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        groups: Dict[str, List[tuple]] = {}
        order: List[str] = []
        for spec in SLIDER_SPECS:
            name, label, group, mn, mx, step, default, decimals = spec
            if group not in groups:
                groups[group] = []
                order.append(group)
            groups[group].append(spec)

        for group in order:
            box = QGroupBox(group)
            box_layout = QVBoxLayout(box)
            box_layout.setSpacing(4)
            for name, label, _g, mn, mx, step, default, decimals in groups[group]:
                slider = LabeledSlider(name, label, mn, mx, step, default, decimals)
                slider.value_changed.connect(self._on_value_changed)
                self._sliders[name] = slider
                box_layout.addWidget(slider)
            layout.addWidget(box)

        layout.addStretch(1)

    # -- state --------------------------------------------------------------
    def _on_value_changed(self, name: str, value: float) -> None:
        setattr(self._params, name, value)
        self.params_changed.emit()

    def params(self) -> EditParams:
        return self._params.copy()

    def set_params(self, params: EditParams) -> None:
        """Load ``params`` into the sliders without firing per-slider signals."""
        self._params = params.copy()
        for name, slider in self._sliders.items():
            slider.set_value(getattr(params, name))
        self.params_changed.emit()

    def update_field(self, name: str, value: float) -> None:
        """Set a single field (and its slider) programmatically."""
        setattr(self._params, name, value)
        if name in self._sliders:
            self._sliders[name].set_value(value)
        self.params_changed.emit()

    def reset_all(self) -> None:
        self.set_params(EditParams())
