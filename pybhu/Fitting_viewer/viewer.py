from __future__ import annotations

import os
import threading
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from .curve_panel import CurvePanel
from .loader import SUPPORTED_EXTENSIONS, coerce_fit_bundle
from .map_panel import MapPanel
from .state import FittingViewerState

# Key → (di, dj) for pixel-cursor movement
_ARROW_KEYS: dict[int, tuple[int, int]] = {
    Qt.Key.Key_Left:  (0, -1),
    Qt.Key.Key_Right: (0,  1),
    Qt.Key.Key_Up:    (-1, 0),
    Qt.Key.Key_Down:  (1,  0),
}

# Key → layer delta
_LAYER_KEYS: dict[int, int] = {
    Qt.Key.Key_PageUp:   +1,
    Qt.Key.Key_PageDown: -1,
}

_SLOT_LABELS = (
    ("slot_a", "A"),
    ("slot_b", "B"),
    ("slot_c", "C"),
    ("slot_d", "D"),
)

# Build the file-dialog filter from SUPPORTED_EXTENSIONS
_EXT_FILTER = (
    "Fitting Files ("
    + " ".join(f"*{ext}" for ext in SUPPORTED_EXTENSIONS)
    + ");;All Files (*)"
)


class FittingViewer(QMainWindow):
    """Main window: curve+controls on the left, 2×2 map grid on the right."""

    def __init__(self, bundle=None, *, title: str = "PyBHU Fitting Viewer"):
        self.app = QApplication.instance()
        self.owns_app = self.app is None
        if self.app is None:
            self.app = QApplication([])

        super().__init__()
        self.base_title = title
        self.current_path: Path | None = None
        self.map_origin = "upper"
        self.state: FittingViewerState | None = None
        self._refreshing = False

        # Force light palette on the whole window
        self.setAutoFillBackground(True)
        pal = self.palette()
        pal.setColor(QPalette.ColorRole.Window, QColor("#f0f0f0"))
        pal.setColor(QPalette.ColorRole.Base, QColor("#ffffff"))
        pal.setColor(QPalette.ColorRole.Text, QColor("#333333"))
        pal.setColor(QPalette.ColorRole.WindowText, QColor("#333333"))
        self.setPalette(pal)

        self.resize(1760, 1100)
        self._build_ui()
        self._connect_signals()

        self.map_panel.set_origin(self.map_origin)

        if bundle is not None:
            self.load_source(bundle)
        else:
            self._refresh_window_title()
            self.refresh_all()

    # ------------------------------------------------------------------ #
    #  UI construction
    # ------------------------------------------------------------------ #

    def _build_ui(self) -> None:
        self.curve_panel = CurvePanel(self)
        self.map_panel = MapPanel(self)

        self.status_bar = QStatusBar(self)
        self.status_bar.setStyleSheet(
            "QStatusBar { background: #f7f7f7; color: #555; font-size: 11px; "
            "border-top: 1px solid #e0e0e0; }"
        )
        self.setStatusBar(self.status_bar)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.curve_panel)    # LEFT
        splitter.addWidget(self.map_panel)      # RIGHT
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        splitter.setSizes([480, 1280])
        splitter.setHandleWidth(3)
        splitter.setStyleSheet("QSplitter::handle { background: #ddd; }")

        central = QWidget(self)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(splitter, 1)
        self.setCentralWidget(central)

    def _connect_signals(self) -> None:
        cp = self.curve_panel
        cp.open_file_requested.connect(self._on_open_file)
        cp.origin_changed.connect(self._on_origin_changed)
        cp.position_changed.connect(self._on_position_changed)
        cp.add_point_requested.connect(self._on_add_spectrum_point)

        mp = self.map_panel
        mp.selection_requested.connect(self._on_position_changed)
        mp.add_point_requested.connect(self._on_add_spectrum_point)
        mp.dataset_changed.connect(self._on_role_dataset_changed)
        mp.layer_changed.connect(self._on_role_layer_changed)

    # ------------------------------------------------------------------ #
    #  Public
    # ------------------------------------------------------------------ #

    def load_source(self, source) -> None:
        self.current_path = (
            Path(source) if isinstance(source, (str, os.PathLike)) else None
        )
        self.state = FittingViewerState(coerce_fit_bundle(source))
        self._refresh_window_title()
        self.refresh_all()

    def refresh_all(self) -> None:
        if self._refreshing:
            return
        self._refreshing = True
        try:
            self.curve_panel.sync_controls(
                self.state,
                file_label=self.current_path.name if self.current_path else None,
                origin=self.map_origin,
            )
            self.map_panel.refresh(self.state)
            self.curve_panel.refresh(self.state)
            self._refresh_status()
        finally:
            self._refreshing = False

    # ------------------------------------------------------------------ #
    #  Internal
    # ------------------------------------------------------------------ #

    def _refresh_window_title(self) -> None:
        suffix = f" — {self.current_path.name}" if self.current_path else ""
        self.setWindowTitle(f"{self.base_title}{suffix}")

    def _refresh_status(self) -> None:
        if self.state is None:
            self.status_bar.showMessage("Open a file to begin.")
            return
        i, j = self.state.position
        parts = [f"({i}, {j})"]
        for slot, label in _SLOT_LABELS:
            val = self.state.value_for_role(slot)
            lc = self.state.layer_count_for_role(slot)
            li = self.state.layer_index_for_role(slot)
            layer_s = f" l{li}" if lc > 1 else ""
            parts.append(f"{label}{layer_s}={val:.6g}")
        self.status_bar.showMessage("  |  ".join(parts))

    # ------------------------------------------------------------------ #
    #  Slots
    # ------------------------------------------------------------------ #

    def _on_open_file(self) -> None:
        start = self.current_path.parent if self.current_path else Path.cwd()
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Fitting Results", str(start), _EXT_FILTER,
        )
        if not path:
            return
        try:
            self.load_source(path)
        except Exception as exc:
            QMessageBox.critical(self, "Open Failed", f"Could not load:\n{exc}")

    def _on_origin_changed(self, origin: str) -> None:
        if origin not in ("upper", "lower"):
            return
        self.map_origin = origin
        self.map_panel.set_origin(origin)
        self.refresh_all()

    def _on_position_changed(self, i: int, j: int) -> None:
        if self.state is None:
            return
        self.state.set_position(i, j)
        self.refresh_all()

    def _on_add_spectrum_point(self, i: int, j: int) -> None:
        if self.state is None:
            return
        self.state.add_spectrum_point(i, j)
        self.refresh_all()

    def _on_role_dataset_changed(self, role: str, combo_index: int) -> None:
        if self.state is None or combo_index < 0:
            return
        selector = self.map_panel.slots[role].dataset_selector
        dataset_index = selector.currentData()
        if dataset_index is None:
            return
        self.state.set_role_dataset_index(role, int(dataset_index))
        self.refresh_all()

    def _on_role_layer_changed(self, role: str, index: int) -> None:
        if self.state is None:
            return
        self.state.set_role_layer_index(role, index)
        self.refresh_all()

    # ------------------------------------------------------------------ #
    #  Keyboard navigation
    # ------------------------------------------------------------------ #

    def keyPressEvent(self, event) -> None:
        if self.state is None:
            super().keyPressEvent(event)
            return

        key = event.key()

        move = _ARROW_KEYS.get(key)
        if move is not None:
            self.state.move_by(*move)
            self.refresh_all()
            return

        layer_delta = _LAYER_KEYS.get(key)
        if layer_delta is not None:
            self.state.move_layer_by(layer_delta)
            self.refresh_all()
            return

        if key == Qt.Key.Key_Escape:
            self.state.clear_spectrum_points()
            self.refresh_all()
            return

        super().keyPressEvent(event)


def run_viewer(viewer: FittingViewer, block: bool | None) -> FittingViewer:
    if block is None:
        block = viewer.owns_app

    if (
        block
        and threading.current_thread() is threading.main_thread()
        and viewer.app.thread().loopLevel() == 0
    ):
        viewer.app.exec()

    return viewer
