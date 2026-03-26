import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QCheckBox, QDialog, QFileDialog, QHBoxLayout, QLabel, QListWidget,
    QPushButton, QVBoxLayout, QWidget,
)

from .state import normalize_spectrum_axis

_PLOT_COLORS = [
    "#ff2d55", "#007aff", "#34c759", "#ff9500", "#af52de",
    "#ff3b30", "#5ac8fa", "#ffcc00", "#ff6482", "#30b0c7",
]


class SpectrumDialog(QDialog):
    def __init__(self, state, on_selection_cleared=None, on_layer_changed=None,
                 on_point_removed=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Spectrum Viewer")
        self.resize(900, 520)
        self.state = state
        self.points = []  # list of (x, y) tuples
        self.on_selection_cleared = on_selection_cleared
        self.on_layer_changed = on_layer_changed
        self.on_point_removed = on_point_removed
        self.spectrum_axis = None
        self.spectrum_axis_label = "Layer"

        # Draggable layer line state
        self._dragging_layer_line = False
        self._layer_line = None
        self._layer_markers = []

        self.figure = Figure(figsize=(7, 4), facecolor="#ffffff")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.axes = self.figure.add_subplot(111)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)

        # Mouse events for dragging the layer line
        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas.mpl_connect("button_release_event", self._on_release)

        self.info_label = QLabel("Click pixels in the image to add spectra.")

        # Point list
        self.point_list = QListWidget()
        self.point_list.setToolTip("Selected points — click to highlight, Delete to remove")
        self.point_list.currentRowChanged.connect(self.refresh_from_state)

        # Stack checkbox
        self.stack_checkbox = QCheckBox("Stack (offset)")
        self.stack_checkbox.setStyleSheet("QCheckBox { color: #000000; }")
        self.stack_checkbox.setToolTip("Vertically offset each spectrum for clarity")
        self.stack_checkbox.toggled.connect(self.refresh_from_state)

        # Buttons
        self.remove_button = QPushButton("Remove Selected")
        self.remove_button.setToolTip("Remove the highlighted point from the list")
        self.remove_button.clicked.connect(self._remove_selected_point)

        self.clear_button = QPushButton("Clear All")
        self.clear_button.clicked.connect(self.clear_selection)

        self.export_button = QPushButton("Export CSV")
        self.export_button.setToolTip("Save the plotted spectra as a CSV file")
        self.export_button.clicked.connect(self._export_csv)

        # --- Left: plot area ---
        left_panel = QVBoxLayout()
        left_panel.addWidget(self.toolbar)
        left_panel.addWidget(self.canvas, 1)
        left_panel.addWidget(self.info_label)
        left_widget = QWidget()
        left_widget.setLayout(left_panel)

        # --- Right: controls panel (hidden until points are added) ---
        right_panel = QVBoxLayout()
        right_panel.addWidget(QLabel("Points:"))
        right_panel.addWidget(self.point_list, 1)
        right_panel.addWidget(self.stack_checkbox)
        right_panel.addWidget(self.export_button)
        right_panel.addWidget(self.remove_button)
        right_panel.addWidget(self.clear_button)
        self._right_widget = QWidget()
        self._right_widget.setLayout(right_panel)
        self._right_widget.setFixedWidth(190)
        self._right_widget.setVisible(False)

        layout = QHBoxLayout(self)
        layout.addWidget(left_widget, 1)
        layout.addWidget(self._right_widget, 0)

        self.refresh_from_state()

    # ------------------------------------------------------------------
    # Public API called by the viewer
    # ------------------------------------------------------------------

    def set_state(self, state, spectrum_axis=None, spectrum_axis_label="Layer", points=None):
        self.state = state
        self.spectrum_axis = self._normalize_spectrum_axis(spectrum_axis)
        self.spectrum_axis_label = spectrum_axis_label if self.spectrum_axis is not None else "Layer"
        if points is not None:
            self.points = [(x, y) for x, y in points if self._point_in_bounds(x, y)]
        else:
            self.points = [(x, y) for x, y in self.points if self._point_in_bounds(x, y)]
        self._sync_point_list()
        self.refresh_from_state()

    def add_point(self, x, y):
        x, y = int(x), int(y)
        if not self._point_in_bounds(x, y):
            return
        if (x, y) in self.points:
            return
        self.points.append((x, y))
        self._sync_point_list()
        self.point_list.setCurrentRow(len(self.points) - 1)
        self.refresh_from_state()

    def clear_selection(self):
        self.points = []
        self._sync_point_list()
        if self.on_selection_cleared:
            self.on_selection_cleared()
        self.refresh_from_state()

    def closeEvent(self, event):
        self.points = []
        self._sync_point_list()
        if self.on_selection_cleared:
            self.on_selection_cleared()
        super().closeEvent(event)

    # ------------------------------------------------------------------
    # Draggable layer line
    # ------------------------------------------------------------------

    def _x_axis_values(self):
        """Return the x-axis array and the x-value for the current layer."""
        if self.spectrum_axis is None:
            x_axis = np.arange(1, self.state.layer_count + 1, dtype=np.float64)
        else:
            x_axis = self.spectrum_axis
        current_x = float(x_axis[self.state.current_layer])
        return x_axis, current_x

    def _x_to_layer_index(self, x_val):
        """Map an x-position on the spectrum plot back to the nearest layer index."""
        if self.state is None:
            return 0
        x_axis, _ = self._x_axis_values()
        idx = int(np.argmin(np.abs(x_axis - x_val)))
        return max(0, min(idx, self.state.layer_count - 1))

    def _on_press(self, event):
        if self.state is None or not self.points:
            return
        if event.inaxes != self.axes or event.xdata is None:
            return
        if getattr(self.toolbar, "mode", ""):
            return
        _, current_x = self._x_axis_values()
        x_range = self.axes.get_xlim()[1] - self.axes.get_xlim()[0]
        if x_range > 0 and abs(event.xdata - current_x) / x_range < 0.03:
            self._dragging_layer_line = True

    def _on_motion(self, event):
        if not self._dragging_layer_line:
            return
        if event.inaxes != self.axes or event.xdata is None:
            return
        new_index = self._x_to_layer_index(event.xdata)
        if new_index != self.state.current_layer:
            self.state.set_current_layer(new_index)
            self._update_layer_line_position()
            if self.on_layer_changed:
                self.on_layer_changed(new_index)

    def _on_release(self, event):
        self._dragging_layer_line = False

    def _update_layer_line_position(self):
        """Move the layer line and markers without a full replot."""
        if self._layer_line is None or self.state is None:
            return
        x_axis, current_x = self._x_axis_values()
        self._layer_line.set_xdata([current_x, current_x])

        # Update marker positions
        stacked = self.stack_checkbox.isChecked() and len(self.points) > 1
        for i, (marker, (px, py)) in enumerate(zip(self._layer_markers, self.points)):
            spectrum = self.state.data[py, px, :].astype(np.float64)
            if stacked:
                ranges = []
                for ppx, ppy in self.points:
                    s = self.state.data[ppy, ppx, :].astype(np.float64)
                    ranges.append(s.max() - s.min())
                offset = max(ranges) * 1.1 if max(ranges) > 0 else 1.0
                y_offset = i * offset
            else:
                y_offset = 0.0
            cur_val = float(spectrum[self.state.current_layer]) + y_offset
            marker.set_data([current_x], [cur_val])

        # Update info label
        highlight_row = self.point_list.currentRow()
        if 0 <= highlight_row < len(self.points):
            px, py = self.points[highlight_row]
            cur_val = float(self.state.data[py, px, self.state.current_layer])
            self.info_label.setText(
                f"Point ({px}, {py}) | {self.spectrum_axis_label}: {current_x:.6g} | Value: {cur_val:.6g}"
            )

        self.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _point_in_bounds(self, x, y):
        if self.state is None:
            return False
        return 0 <= x < self.state.data.shape[1] and 0 <= y < self.state.data.shape[0]

    def _normalize_spectrum_axis(self, spectrum_axis):
        if self.state is None:
            return None
        return normalize_spectrum_axis(spectrum_axis, self.state.layer_count)

    def _sync_point_list(self):
        self.point_list.blockSignals(True)
        self.point_list.clear()
        for i, (x, y) in enumerate(self.points):
            color = _PLOT_COLORS[i % len(_PLOT_COLORS)]
            self.point_list.addItem(f"\u2588 #{i+1}  ({x}, {y})")
            item = self.point_list.item(i)
            item.setForeground(QColor(color))
        self.point_list.blockSignals(False)

    def _remove_selected_point(self):
        row = self.point_list.currentRow()
        if 0 <= row < len(self.points):
            self.points.pop(row)
            self._sync_point_list()
            if self.on_point_removed:
                self.on_point_removed(list(self.points))
            self.refresh_from_state()

    def _export_csv(self):
        if not self.points or self.state is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Spectra as CSV", "spectra.csv", "CSV Files (*.csv);;All Files (*)"
        )
        if not path:
            return
        x_axis, _ = self._x_axis_values()
        spectra = [self.state.data[py, px, :] for px, py in self.points]
        header_parts = [self.spectrum_axis_label]
        for px, py in self.points:
            header_parts.append(f"({px},{py})")
        with open(path, "w") as f:
            f.write(",".join(header_parts) + "\n")
            for i, x_val in enumerate(x_axis):
                row = [str(x_val)]
                for spectrum in spectra:
                    row.append(str(float(spectrum[i])))
                f.write(",".join(row) + "\n")

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def refresh_from_state(self, _=None):
        self.axes.clear()
        self._layer_line = None
        self._layer_markers = []
        self._right_widget.setVisible(bool(self.points))

        if self.state is None:
            self.info_label.setText("No data loaded.")
            self.axes.set_axis_off()
            self.canvas.draw_idle()
            return

        if not self.points:
            self.info_label.setText("Click pixels in the image to add spectra.")
            self.axes.set_axis_off()
            self.axes.text(
                0.5, 0.5, "No points selected",
                ha="center", va="center", fontsize=14, color="#555555",
                transform=self.axes.transAxes,
            )
            self.canvas.draw_idle()
            return

        x_axis, current_x = self._x_axis_values()

        stacked = self.stack_checkbox.isChecked() and len(self.points) > 1
        highlight_row = self.point_list.currentRow()

        # Compute all spectra first to determine offset
        spectra = []
        for px, py in self.points:
            spectra.append(self.state.data[py, px, :].astype(np.float64))

        if stacked:
            ranges = [s.max() - s.min() for s in spectra]
            offset = max(ranges) * 1.1 if max(ranges) > 0 else 1.0
        else:
            offset = 0.0

        for i, ((px, py), spectrum) in enumerate(zip(self.points, spectra)):
            color = _PLOT_COLORS[i % len(_PLOT_COLORS)]
            y_offset = i * offset if stacked else 0.0
            shifted = spectrum + y_offset
            lw = 2.5 if i == highlight_row else 1.5
            alpha = 1.0 if i == highlight_row else 0.7
            self.axes.plot(
                x_axis, shifted,
                color=color, linewidth=lw, alpha=alpha,
            )
            # current layer marker
            cur_val = float(spectrum[self.state.current_layer]) + y_offset
            (marker,) = self.axes.plot(
                current_x, cur_val, marker="o", color=color, markersize=5, alpha=alpha,
            )
            self._layer_markers.append(marker)

        # Draggable current-layer vertical line
        self._layer_line = self.axes.axvline(
            current_x, color="#d83b01", linestyle="--", linewidth=1.5, alpha=0.8,
        )

        if len(self.points) == 1:
            self.axes.set_title(f"Spectrum at ({self.points[0][0]}, {self.points[0][1]})", fontweight="bold")
        else:
            self.axes.set_title(f"Spectra \u2014 {len(self.points)} points", fontweight="bold")

        self.axes.set_xlabel(self.spectrum_axis_label)
        self.axes.set_ylabel("Value" + (" + offset" if stacked else ""))
        self.axes.grid(True, linestyle=":", alpha=0.5)

        if self.state.layer_count == 1:
            if self.spectrum_axis is None:
                self.axes.set_xlim(0.5, 1.5)
            else:
                self.axes.set_xlim(current_x - 0.5, current_x + 0.5)

        # Info label shows highlighted point
        if 0 <= highlight_row < len(self.points):
            px, py = self.points[highlight_row]
            cur_val = float(spectra[highlight_row][self.state.current_layer])
            self.info_label.setText(
                f"Point ({px}, {py}) | {self.spectrum_axis_label}: {current_x:.6g} | Value: {cur_val:.6g}"
            )
        else:
            self.info_label.setText(f"{len(self.points)} points selected")

        self.figure.tight_layout()
        self.canvas.draw_idle()
