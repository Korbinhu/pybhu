from __future__ import annotations

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtWidgets import QVBoxLayout, QWidget

from .state import FittingViewerState


class CurvePanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(facecolor="#ffffff", tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.axes = self.figure.add_subplot(111)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)

    def refresh(self, state: FittingViewerState) -> None:
        i, j = state.position
        self.axes.clear()
        self.axes.plot(
            state.bundle.x_raw,
            state.raw_spectrum(),
            color="#1f77b4",
            linewidth=1.2,
            marker="o",
            markersize=2.5,
            label="Raw",
        )
        self.axes.plot(
            state.bundle.x_fit,
            state.fit_spectrum(),
            color="#d62728",
            linewidth=2.0,
            label="Fit",
        )
        self.axes.set_title(f"Spectra at (i={i}, j={j})")
        x_unit = state.bundle.x_unit.strip()
        y_unit = state.bundle.y_unit.strip()
        self.axes.set_xlabel(f"X ({x_unit})" if x_unit else "X")
        self.axes.set_ylabel(f"Signal ({y_unit})" if y_unit else "Signal")
        self.axes.grid(True, linestyle=":", alpha=0.5)
        self.axes.legend(loc="best")
        self.canvas.draw_idle()
