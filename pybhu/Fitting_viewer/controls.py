from __future__ import annotations

from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .state import FittingViewerState

COLORMAPS: list[str] = sorted([
    # Perceptual
    "viridis", "plasma", "inferno", "magma", "cividis", "turbo",
    # Sequential
    "Greys", "Purples", "Blues", "Greens", "Oranges", "Reds",
    "YlOrBr", "YlOrRd", "OrRd", "PuRd", "RdPu", "BuPu",
    "GnBu", "PuBu", "YlGnBu", "PuBuGn", "BuGn", "YlGn",
    # Sequential 2
    "binary", "gist_yarg", "gist_gray", "gray", "bone", "pink",
    "spring", "summer", "autumn", "winter", "cool", "Wistia",
    "hot", "afmhot", "gist_heat", "copper",
    # Diverging
    "PiYG", "PRGn", "BrBG", "PuOr", "RdGy", "RdBu",
    "RdYlBu", "RdYlGn", "Spectral", "coolwarm", "bwr", "seismic",
    # Cyclic
    "twilight", "twilight_shifted", "hsv",
    # Qualitative
    "Pastel1", "Pastel2", "Paired", "Accent",
    "Dark2", "Set1", "Set2", "Set3",
    "tab10", "tab20", "tab20b", "tab20c",
    # Misc
    "flag", "prism", "ocean", "gist_earth", "terrain", "gist_stern",
    "gnuplot", "gnuplot2", "CMRmap", "cubehelix", "brg",
    "gist_rainbow", "rainbow", "jet", "nipy_spectral", "gist_ncar",
], key=lambda x: x.lower())
# Wildcard * ensures ALL child widgets inherit the light background,
# preventing any dark-theme bleed-through.
_SS = """
* {
    background: #f7f7f7;
    color: #333;
}
QLabel {
    font-size: 11px;
    color: #555;
    border: none;
    padding: 0;
}
QPushButton {
    background: #fff;
    color: #333;
    border: 1px solid #c0c0c0;
    border-radius: 4px;
    padding: 4px 12px;
    font-size: 11px;
    font-weight: 500;
    min-height: 24px;
}
QPushButton:hover  { background: #eee; border-color: #999; }
QPushButton:pressed { background: #ddd; }
QComboBox {
    background: #fff;
    color: #333;
    border: 1px solid #c0c0c0;
    border-radius: 4px;
    padding: 3px 8px;
    font-size: 11px;
    min-height: 24px;
}
QComboBox::drop-down {
    width: 20px;
    border-left: 1px solid #ddd;
    background: #f5f5f5;
}
QComboBox::down-arrow {
    image: none;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 6px solid #555;
    width: 0; height: 0;
}
QComboBox:hover { border-color: #999; }
QComboBox QAbstractItemView {
    background: #fff;
    color: #333;
    selection-background-color: #dbeafe;
    selection-color: #333;
    border: 1px solid #ccc;
}
QSpinBox {
    background: #fff;
    color: #333;
    border: 1px solid #c0c0c0;
    border-radius: 4px;
    padding: 2px 4px;
    font-size: 12px;
    min-height: 26px;
}
QSpinBox:hover { border-color: #999; }
QSpinBox::up-button, QSpinBox::down-button {
    width: 20px;
    border: none;
    background: #f0f0f0;
}
QSpinBox::up-button   { subcontrol-position: top right; border-top-right-radius: 3px; border-bottom: 1px solid #ddd; }
QSpinBox::down-button { subcontrol-position: bottom right; border-bottom-right-radius: 3px; }
QSpinBox::up-button:hover, QSpinBox::down-button:hover { background: #e0e0e0; }
QSpinBox::up-button:pressed, QSpinBox::down-button:pressed { background: #ccc; }
QSpinBox::up-arrow {
    image: none;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-bottom: 6px solid #444;
    width: 0; height: 0;
}
QSpinBox::down-arrow {
    image: none;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 6px solid #444;
    width: 0; height: 0;
}
"""


def _label(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    return lbl


class ControlPanel(QWidget):
    open_file_requested = pyqtSignal()
    origin_changed = pyqtSignal(str)
    position_changed = pyqtSignal(int, int)
    add_point_requested = pyqtSignal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAutoFillBackground(True)

        # ── Single row: Open + Origin + i/j + Add + file label ──────────
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)

        self.open_button = QPushButton("Open File")
        self.open_button.clicked.connect(self.open_file_requested.emit)
        row.addWidget(self.open_button)

        row.addSpacing(4)
        row.addWidget(_label("Origin"))
        self.origin_selector = QComboBox()
        self.origin_selector.setFixedWidth(80)
        self.origin_selector.addItem("Upper", "upper")
        self.origin_selector.addItem("Lower", "lower")
        self.origin_selector.currentIndexChanged.connect(self._emit_origin_changed)
        row.addWidget(self.origin_selector)

        row.addSpacing(6)
        row.addWidget(_label("i"))
        self.i_spin = QSpinBox()
        self.i_spin.setFixedWidth(70)
        self.i_spin.valueChanged.connect(self._emit_position_changed)
        row.addWidget(self.i_spin)

        row.addSpacing(2)
        row.addWidget(_label("j"))
        self.j_spin = QSpinBox()
        self.j_spin.setFixedWidth(70)
        self.j_spin.valueChanged.connect(self._emit_position_changed)
        row.addWidget(self.j_spin)

        row.addSpacing(4)
        self.add_point_btn = QPushButton("+ Add")
        self.add_point_btn.setToolTip("Add current (i, j) as a spectrum comparison point")
        self.add_point_btn.clicked.connect(self._emit_add_point)
        row.addWidget(self.add_point_btn)

        row.addSpacing(6)
        self.file_label = QLabel("No file loaded")
        self.file_label.setStyleSheet("color: #888; font-size: 10px;")
        row.addWidget(self.file_label, 1)

        # ── Layout ──────────────────────────────────────────────────────
        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 6, 10, 6)
        outer.setSpacing(0)
        outer.addLayout(row)

        self.setStyleSheet(_SS)

    # ------------------------------------------------------------------ #

    def _emit_origin_changed(self, index: int) -> None:
        if index < 0:
            return
        origin = self.origin_selector.itemData(index)
        if origin is not None:
            self.origin_changed.emit(str(origin))

    def _emit_position_changed(self, _value: int) -> None:
        self.position_changed.emit(self.i_spin.value(), self.j_spin.value())

    def _emit_add_point(self) -> None:
        self.add_point_requested.emit(self.i_spin.value(), self.j_spin.value())

    # ------------------------------------------------------------------ #

    def sync_from_state(
        self,
        state: FittingViewerState | None,
        *,
        file_label: str | None = None,
        origin: str = "upper",
    ) -> None:
        self.origin_selector.blockSignals(True)
        idx = self.origin_selector.findData(origin)
        if idx >= 0:
            self.origin_selector.setCurrentIndex(idx)
        self.origin_selector.blockSignals(False)

        self.i_spin.blockSignals(True)
        self.j_spin.blockSignals(True)
        if state is None:
            self.i_spin.setRange(0, 0)
            self.i_spin.setValue(0)
            self.j_spin.setRange(0, 0)
            self.j_spin.setValue(0)
            self.i_spin.setEnabled(False)
            self.j_spin.setEnabled(False)
        else:
            ny, nx = state.bundle.spatial_shape
            self.i_spin.setRange(0, ny - 1)
            self.i_spin.setValue(state.selection.i)
            self.j_spin.setRange(0, nx - 1)
            self.j_spin.setValue(state.selection.j)
            self.i_spin.setEnabled(True)
            self.j_spin.setEnabled(True)
        self.i_spin.blockSignals(False)
        self.j_spin.blockSignals(False)

        self.file_label.setText(file_label or "No file loaded")
