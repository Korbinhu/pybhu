from __future__ import annotations

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QVBoxLayout, QWidget

from .state import FittingViewerState


class MapPanel(QWidget):
    selection_requested = pyqtSignal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(facecolor="#ffffff", tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.axes = self.figure.add_subplot(111)
        self.image_artist = None
        self.hline = None
        self.vline = None
        self._dragging = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)

        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("button_release_event", self._on_release)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)

    def refresh(self, state: FittingViewerState) -> None:
        map_data = state.current_map()
        vmin = float(map_data.min())
        vmax = float(map_data.max())
        if vmin == vmax:
            vmax = vmin + 1.0

        if self.image_artist is None:
            self.axes.clear()
            self.image_artist = self.axes.imshow(
                map_data,
                cmap="viridis",
                origin="upper",
                interpolation="nearest",
                vmin=vmin,
                vmax=vmax,
            )
            self.axes.set_xlabel("j")
            self.axes.set_ylabel("i")
            self.hline = self.axes.axhline(state.selection.i, color="#ffffff", linestyle="--", linewidth=1.0)
            self.vline = self.axes.axvline(state.selection.j, color="#ffffff", linestyle="--", linewidth=1.0)
        else:
            self.image_artist.set_data(map_data)
            self.image_artist.set_clim(vmin, vmax)
            self.hline.set_ydata([state.selection.i, state.selection.i])
            self.vline.set_xdata([state.selection.j, state.selection.j])

        layer_text = (
            f"Layer {state.selection.layer_index + 1}/{state.layer_count_for_dataset()}"
            if state.layer_count_for_dataset() > 1
            else "Single Layer"
        )
        self.axes.set_title(f"{state.current_dataset.name} | {layer_text}")
        self.canvas.draw_idle()

    def _emit_selection(self, event) -> None:
        if event.inaxes != self.axes or event.xdata is None or event.ydata is None:
            return
        i = int(round(event.ydata))
        j = int(round(event.xdata))
        self.selection_requested.emit(i, j)

    def _on_press(self, event) -> None:
        self._dragging = True
        self._emit_selection(event)

    def _on_release(self, _event) -> None:
        self._dragging = False

    def _on_motion(self, event) -> None:
        if self._dragging:
            self._emit_selection(event)
