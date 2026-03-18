from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .state import FittingViewerState


class ControlPanel(QWidget):
    dataset_index_changed = pyqtSignal(int)
    layer_index_changed = pyqtSignal(int)
    position_changed = pyqtSignal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(300)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        dataset_group = QFrame()
        dataset_layout = QFormLayout(dataset_group)
        dataset_layout.setContentsMargins(10, 10, 10, 10)
        self.dataset_selector = QComboBox()
        self.dataset_selector.currentIndexChanged.connect(self.dataset_index_changed.emit)
        dataset_layout.addRow("Dataset", self.dataset_selector)

        self.layer_slider = QSlider(Qt.Orientation.Horizontal)
        self.layer_slider.valueChanged.connect(self.layer_index_changed.emit)
        self.layer_label = QLabel("1 / 1")
        layer_row = QHBoxLayout()
        layer_row.addWidget(self.layer_slider, 1)
        layer_row.addWidget(self.layer_label, 0)
        dataset_layout.addRow("Layer", layer_row)
        layout.addWidget(dataset_group)

        coord_group = QFrame()
        coord_layout = QFormLayout(coord_group)
        coord_layout.setContentsMargins(10, 10, 10, 10)
        self.i_spin = QSpinBox()
        self.j_spin = QSpinBox()
        self.i_spin.valueChanged.connect(self._emit_position_changed)
        self.j_spin.valueChanged.connect(self._emit_position_changed)
        coord_layout.addRow("i", self.i_spin)
        coord_layout.addRow("j", self.j_spin)
        layout.addWidget(coord_group)

        info_group = QFrame()
        info_layout = QFormLayout(info_group)
        info_layout.setContentsMargins(10, 10, 10, 10)
        self.value_label = QLabel("-")
        self.shape_label = QLabel("-")
        self.units_label = QLabel("-")
        info_layout.addRow("Value", self.value_label)
        info_layout.addRow("Dataset Shape", self.shape_label)
        info_layout.addRow("Units", self.units_label)
        layout.addWidget(info_group)

        layout.addStretch()

    def _emit_position_changed(self, _value: int) -> None:
        self.position_changed.emit(self.i_spin.value(), self.j_spin.value())

    def sync_from_state(self, state: FittingViewerState) -> None:
        dataset_names = state.dataset_names
        i_max, j_max = state.bundle.spatial_shape
        layer_count = state.layer_count_for_dataset()

        self.dataset_selector.blockSignals(True)
        self.dataset_selector.clear()
        self.dataset_selector.addItems(dataset_names)
        self.dataset_selector.setCurrentIndex(state.selection.dataset_index)
        self.dataset_selector.blockSignals(False)

        self.layer_slider.blockSignals(True)
        self.layer_slider.setMinimum(0)
        self.layer_slider.setMaximum(max(layer_count - 1, 0))
        self.layer_slider.setValue(state.selection.layer_index)
        self.layer_slider.setEnabled(layer_count > 1)
        self.layer_slider.blockSignals(False)
        self.layer_label.setText(f"{state.selection.layer_index + 1} / {layer_count}")

        self.i_spin.blockSignals(True)
        self.i_spin.setRange(0, i_max - 1)
        self.i_spin.setValue(state.selection.i)
        self.i_spin.blockSignals(False)

        self.j_spin.blockSignals(True)
        self.j_spin.setRange(0, j_max - 1)
        self.j_spin.setValue(state.selection.j)
        self.j_spin.blockSignals(False)

        self.value_label.setText(f"{state.current_value():.6g}")
        self.shape_label.setText(" x ".join(str(part) for part in state.current_dataset.data.shape))
        x_unit = state.bundle.x_unit.strip() or "-"
        y_unit = state.bundle.y_unit.strip() or "-"
        self.units_label.setText(f"x: {x_unit}, y: {y_unit}")
