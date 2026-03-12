from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .colorbar_dialog import ColorbarDialog
from .colormaps import available_colormaps, resolve_colormap
from .histogram_dialog import HistogramDialog
from .state import ViewerState


class ImageStackViewer(QMainWindow):
    def __init__(self, data, **options):
        self.app = QApplication.instance()
        if self.app is None:
            self.app = QApplication([])
        super().__init__()
        initial_layer = options.pop("initial_layer", 0)
        if "colormap" in options and "colormap_name" not in options:
            options["colormap_name"] = options.pop("colormap")
        self.state = ViewerState(data, **options)
        self.state.set_current_layer(initial_layer)
        self.image_artist = None

        central = QWidget(self)
        layout = QVBoxLayout(central)
        controls = QHBoxLayout()
        self.layer_selector = QComboBox()
        self.layer_selector.addItems([str(i) for i in range(self.state.layer_count)])
        self.layer_selector.currentIndexChanged.connect(self.on_layer_changed)
        controls.addWidget(self.layer_selector)

        self.colormap_selector = QComboBox()
        self.colormap_selector.addItems(available_colormaps())
        self.colormap_selector.setCurrentText(self.state.colormap_name)
        self.colormap_selector.currentTextChanged.connect(self.on_colormap_changed)
        controls.addWidget(self.colormap_selector)

        self.invert_checkbox = QCheckBox("Invert")
        self.invert_checkbox.toggled.connect(self.toggle_inverted)
        controls.addWidget(self.invert_checkbox)

        histogram_button = QPushButton("Histogram")
        histogram_button.clicked.connect(self.open_histogram_dialog)
        controls.addWidget(histogram_button)

        colorbar_button = QPushButton("Colorbar")
        colorbar_button.clicked.connect(self.open_colorbar_dialog)
        controls.addWidget(colorbar_button)

        self.figure = Figure()
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.axes = self.figure.add_subplot(111)

        layout.addLayout(controls)
        layout.addWidget(self.canvas)
        self.setCentralWidget(central)

        self.layer_selector.setCurrentIndex(self.state.current_layer)
        self.refresh_image()

    def refresh_image(self):
        layer = self.state.data[:, :, self.state.current_layer]
        cmap = resolve_colormap(self.state.colormap_name, self.state.inverted)
        vmin, vmax = self.state.visible_limits()
        self.axes.clear()
        self.image_artist = self.axes.imshow(layer, cmap=cmap, vmin=vmin, vmax=vmax)
        self.axes.set_axis_off()
        self.canvas.draw_idle()

    def on_layer_changed(self, index: int):
        self.state.set_current_layer(index)
        self.refresh_image()
        self.refresh_aux_windows()

    def open_histogram_dialog(self):
        if getattr(self, "histogram_dialog", None) is None:
            self.histogram_dialog = HistogramDialog(
                self.state,
                on_limits_changed=self.refresh_image,
                parent=self,
            )
        self.histogram_dialog.show()
        self.histogram_dialog.raise_()
        self.histogram_dialog.activateWindow()

    def on_colormap_changed(self, name: str):
        self.state.set_colormap(name)
        self.refresh_image()
        self.refresh_aux_windows()

    def toggle_inverted(self, checked: bool = True):
        self.state.set_inverted(checked)
        self.refresh_image()
        self.refresh_aux_windows()

    def open_colorbar_dialog(self):
        if getattr(self, "colorbar_dialog", None) is None:
            self.colorbar_dialog = ColorbarDialog(self.state, parent=self)
        self.colorbar_dialog.refresh()
        self.colorbar_dialog.show()
        self.colorbar_dialog.raise_()
        self.colorbar_dialog.activateWindow()

    def refresh_aux_windows(self):
        if getattr(self, "colorbar_dialog", None) is not None:
            self.colorbar_dialog.refresh()
        if getattr(self, "histogram_dialog", None) is not None:
            self.histogram_dialog.refresh_from_state()
