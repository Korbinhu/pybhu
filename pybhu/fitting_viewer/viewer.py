from __future__ import annotations

import threading

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QMainWindow, QSplitter, QStatusBar

from .bundle import ensure_fit_bundle
from .controls import ControlPanel
from .curve_panel import CurvePanel
from .map_panel import MapPanel
from .state import FittingViewerState


class FittingViewer(QMainWindow):
    def __init__(self, bundle, *, title: str = "PyBHU Fitting Viewer"):
        self.app = QApplication.instance()
        self.owns_app = self.app is None
        if self.app is None:
            self.app = QApplication([])

        super().__init__()
        self.bundle = ensure_fit_bundle(bundle)
        self.state = FittingViewerState(self.bundle)

        self.setWindowTitle(title)
        self.resize(1440, 860)

        self.curve_panel = CurvePanel(self)
        self.map_panel = MapPanel(self)
        self.control_panel = ControlPanel(self)
        self.status_bar = QStatusBar(self)
        self.setStatusBar(self.status_bar)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.curve_panel)
        splitter.addWidget(self.map_panel)
        splitter.addWidget(self.control_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        self.setCentralWidget(splitter)

        self.map_panel.selection_requested.connect(self._on_map_selection_requested)
        self.control_panel.dataset_index_changed.connect(self._on_dataset_changed)
        self.control_panel.layer_index_changed.connect(self._on_layer_changed)
        self.control_panel.position_changed.connect(self._on_position_changed)

        self.refresh_all()

    def refresh_all(self) -> None:
        self.control_panel.sync_from_state(self.state)
        self.map_panel.refresh(self.state)
        self.curve_panel.refresh(self.state)
        self._refresh_status()

    def _refresh_status(self) -> None:
        i, j = self.state.position
        self.status_bar.showMessage(
            f"Dataset: {self.state.current_dataset.name} | "
            f"Pixel: (i={i}, j={j}) | "
            f"Value: {self.state.current_value():.6g}"
        )

    def _on_map_selection_requested(self, i: int, j: int) -> None:
        self.state.set_position(i, j)
        self.refresh_all()

    def _on_dataset_changed(self, index: int) -> None:
        self.state.set_dataset_index(index)
        self.refresh_all()

    def _on_layer_changed(self, index: int) -> None:
        self.state.set_layer_index(index)
        self.refresh_all()

    def _on_position_changed(self, i: int, j: int) -> None:
        self.state.set_position(i, j)
        self.refresh_all()

    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key == Qt.Key.Key_Left:
            self.state.move_by(0, -1)
            self.refresh_all()
            return
        if key == Qt.Key.Key_Right:
            self.state.move_by(0, 1)
            self.refresh_all()
            return
        if key == Qt.Key.Key_Up:
            self.state.move_by(-1, 0)
            self.refresh_all()
            return
        if key == Qt.Key.Key_Down:
            self.state.move_by(1, 0)
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
