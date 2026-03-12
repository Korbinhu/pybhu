from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class HistogramDialog(QDialog):
    def __init__(self, state, on_limits_changed=None, parent=None):
        super().__init__(parent)
        self.state = state
        self.on_limits_changed = on_limits_changed

        self.figure = Figure()
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.axes = self.figure.add_subplot(111)

        self.mode_selector = QComboBox()
        self.mode_selector.addItems(["global", "per_layer"])
        self.mode_selector.setCurrentText(self.state.color_limit_mode)
        self.mode_selector.currentTextChanged.connect(self.on_mode_changed)

        self.min_edit = QLineEdit()
        self.max_edit = QLineEdit()

        controls = QWidget(self)
        controls_layout = QFormLayout(controls)
        controls_layout.addRow("Mode", self.mode_selector)
        controls_layout.addRow("Minimum", self.min_edit)
        controls_layout.addRow("Maximum", self.max_edit)

        button_row = QHBoxLayout()
        self.apply_button = QPushButton("Apply")
        self.apply_button.clicked.connect(self.apply_changes)
        button_row.addWidget(self.apply_button)

        layout = QVBoxLayout(self)
        layout.addWidget(self.canvas)
        layout.addWidget(controls)
        layout.addLayout(button_row)

        self.refresh_from_state()

    def _seed_fields(self):
        vmin, vmax = self.state.visible_limits()
        self.min_edit.setText(str(vmin))
        self.max_edit.setText(str(vmax))

    def _draw_histogram(self):
        layer = self.state.data[:, :, self.state.current_layer].ravel()
        self.axes.clear()
        self.axes.hist(layer, bins=100)
        self.canvas.draw_idle()

    def refresh_from_state(self):
        self.mode_selector.setCurrentText(self.state.color_limit_mode)
        self._seed_fields()
        self._draw_histogram()

    def on_mode_changed(self, mode: str):
        self.state.set_color_limit_mode(mode)
        self.refresh_from_state()
        if self.on_limits_changed is not None:
            self.on_limits_changed()

    def apply_changes(self):
        self.state.set_limits(float(self.min_edit.text()), float(self.max_edit.text()))
        if self.on_limits_changed is not None:
            self.on_limits_changed()
