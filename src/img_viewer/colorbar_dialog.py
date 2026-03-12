from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from PyQt6.QtWidgets import QDialog, QVBoxLayout

from .colormaps import resolve_colormap


class ColorbarDialog(QDialog):
    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state
        self.figure = Figure(figsize=(1.5, 3.0))
        self.canvas = FigureCanvasQTAgg(self.figure)

        layout = QVBoxLayout(self)
        layout.addWidget(self.canvas)

        self.refresh()

    def refresh(self):
        self.figure.clear()
        axes = self.figure.add_subplot(111)
        cmap = resolve_colormap(self.state.colormap_name, self.state.inverted)
        vmin, vmax = self.state.visible_limits()
        mappable = ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=cmap)
        self.figure.colorbar(mappable, cax=axes)
        self.canvas.draw_idle()
