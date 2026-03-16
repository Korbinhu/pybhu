from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PyQt6.QtWidgets import (
    QComboBox, QDialog, QGridLayout, QHBoxLayout, QLineEdit,
    QPushButton, QVBoxLayout, QWidget, QLabel, QRadioButton, QButtonGroup, QMessageBox
)
import numpy as np

class HistogramDialog(QDialog):
    def __init__(self, state, on_limits_changed=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Histogram & Intensity Limits")
        self.resize(700, 600)
        self.state = state
        self.on_limits_changed = on_limits_changed

        # --- Plotting Setup ---
        self.figure = Figure(figsize=(7, 4), facecolor="#ffffff")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.axes = self.figure.add_subplot(111)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)

        self.dragging_line = None
        self.min_line = None
        self.max_line = None
        
        self.canvas.mpl_connect("button_press_event", self.on_press)
        self.canvas.mpl_connect("button_release_event", self.on_release)
        self.canvas.mpl_connect("motion_notify_event", self.on_motion)

        # --- Control Widgets Setup ---
        self.radio_layer = QRadioButton("Current Layer")
        self.radio_stack = QRadioButton("Entire Stack")
        self.radio_layer.setChecked(True)
        self.hist_range_group = QButtonGroup(self)
        self.hist_range_group.addButton(self.radio_layer)
        self.hist_range_group.addButton(self.radio_stack)
        self.hist_range_group.buttonClicked.connect(self._draw_histogram)

        self.mode_selector = QComboBox()
        self.mode_selector.addItems(["global", "per_layer"])
        self.mode_selector.setCurrentText(self.state.color_limit_mode)
        self.mode_selector.currentTextChanged.connect(self.on_mode_changed)

        self.min_edit = QLineEdit()
        self.max_edit = QLineEdit()
        self.min_edit.returnPressed.connect(self.apply_changes)
        self.max_edit.returnPressed.connect(self.apply_changes)

        self.apply_btn = QPushButton("Apply Typed Limits")
        self.apply_btn.clicked.connect(self.apply_changes)

        self.auto_btn = QPushButton("Auto Scale (Current Layer)")
        self.auto_btn.clicked.connect(self.auto_scale)

        # --- Layout Setup ---
        controls_widget = QWidget()
        layout_ctrl = QGridLayout(controls_widget)
        
        layout_ctrl.addWidget(QLabel("Data for Histogram:"), 0, 0)
        rad_layout = QHBoxLayout()
        rad_layout.addWidget(self.radio_layer)
        rad_layout.addWidget(self.radio_stack)
        rad_layout.addStretch()
        layout_ctrl.addLayout(rad_layout, 0, 1)

        layout_ctrl.addWidget(QLabel("Limit Mode:"), 0, 2)
        layout_ctrl.addWidget(self.mode_selector, 0, 3)

        layout_ctrl.addWidget(QLabel("Display Min:"), 1, 0)
        layout_ctrl.addWidget(self.min_edit, 1, 1)

        layout_ctrl.addWidget(QLabel("Display Max:"), 1, 2)
        layout_ctrl.addWidget(self.max_edit, 1, 3)
        
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.apply_btn)
        btn_layout.addWidget(self.auto_btn)
        btn_layout.addStretch()

        main_layout = QVBoxLayout(self)
        main_layout.addWidget(self.toolbar)
        main_layout.addWidget(self.canvas)
        main_layout.addWidget(controls_widget)
        main_layout.addLayout(btn_layout)

        # Draw the initial state
        self.refresh_from_state()

    def refresh_from_state(self):
        """Update the UI fields from current state and redraw."""
        self.mode_selector.setCurrentText(self.state.color_limit_mode)
        
        if self.state.color_limit_mode == "per_layer":
            # Force limits to update based on current layer explicitly when switching layers
            vmin, vmax = self.state._calculate_limits()
            self.state.set_limits(vmin, vmax)
            
        vmin, vmax = self.state.visible_limits()
        self.min_edit.setText(f"{vmin:.6g}")
        self.max_edit.setText(f"{vmax:.6g}")
        self._draw_histogram()

    def on_mode_changed(self, mode: str):
        """Handle limit mode switch (global vs per_layer)."""
        self.state.set_color_limit_mode(mode)
        self.refresh_from_state()
        if self.on_limits_changed:
            self.on_limits_changed()

    def apply_changes(self):
        """Read the Min/Max text fields and apply the limits to the application."""
        try:
            vmin = float(self.min_edit.text())
            vmax = float(self.max_edit.text())
            if vmin < vmax:
                self.state.set_limits(vmin, vmax)
                self._draw_histogram()
                if self.on_limits_changed:
                    self.on_limits_changed()
            else:
                QMessageBox.warning(self, "Invalid Input", "Display Min must be strictly less than Display Max.")
                self.refresh_from_state() # revert text fields
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Please enter valid numerical values.")
            self.refresh_from_state() # revert text fields

    def auto_scale(self):
        """Automatically scale to the min and max of the current visible layer."""
        layer = self.state.data[:, :, self.state.current_layer]
        vmin, vmax = float(layer.min()), float(layer.max())
        if vmin == vmax:
            vmax = vmin + 1.0 # Ensure some range exists
        self.min_edit.setText(f"{vmin:.6g}")
        self.max_edit.setText(f"{vmax:.6g}")
        self.apply_changes()

    def _draw_histogram(self, _=None):
        """Redraw the matplotlib histogram based on current data and limits."""
        self.axes.clear()
        
        # Decide which data to bin
        if self.radio_layer.isChecked():
            data_to_hist = self.state.data[:, :, self.state.current_layer].ravel()
            title = f"Histogram of Layer {self.state.current_layer + 1}"
        else:
            data_to_hist = self.state.data.ravel()
            title = "Histogram of Entire Data Stack"
            
        # Strip infinites or nans that break the histogram function
        data_to_hist = data_to_hist[np.isfinite(data_to_hist)]
        if len(data_to_hist) == 0:
            self.canvas.draw_idle()
            return

        # Plot Histogram
        self.axes.hist(data_to_hist, bins=100, color='#0078d7', alpha=0.7, log=True)
        
        # Plot Red Limit Lines
        vmin, vmax = self.state.visible_limits()
        self.min_line = self.axes.axvline(vmin, color='red', linestyle='--', linewidth=2)
        self.max_line = self.axes.axvline(vmax, color='red', linestyle='--', linewidth=2)
        
        # Format axes
        self.axes.set_title(title, fontweight='bold')
        self.axes.set_ylabel("Frequency (Log Scale)")
        self.axes.grid(True, linestyle=':', alpha=0.5)

        # Scale axes strictly to fit both the physical data AND the user's custom limits
        data_min, data_max = data_to_hist.min(), data_to_hist.max()
        plot_min = min(data_min, vmin)
        plot_max = max(data_max, vmax)
        
        if plot_min == plot_max:
            plot_max = plot_min + 1.0
            
        padding = (plot_max - plot_min) * 0.05
        self.axes.set_xlim(plot_min - padding, plot_max + padding)

        self.figure.tight_layout()
        self.canvas.draw_idle()

    def on_press(self, event):
        if event.inaxes != self.axes or event.xdata is None: return
        vmin, vmax = self.state.visible_limits()
        
        # Check if click is near MIN line (within 2% of the total x-axis range)
        x_range = self.axes.get_xlim()[1] - self.axes.get_xlim()[0]
        tolerance = x_range * 0.02
        
        if abs(event.xdata - vmin) < tolerance:
            self.dragging_line = 'min'
        elif abs(event.xdata - vmax) < tolerance:
            self.dragging_line = 'max'

    def on_release(self, event):
        self.dragging_line = None
        # Once they finish dragging, trigger a full redraw to scale axes perfectly
        self._draw_histogram()

    def on_motion(self, event):
        if self.dragging_line is None or event.inaxes != self.axes or event.xdata is None: return
        
        vmin, vmax = self.state.visible_limits()
        new_val = event.xdata
        
        if self.dragging_line == 'min':
            if new_val < vmax:
                self.state.set_limits(new_val, vmax)
                self.min_edit.setText(f"{new_val:.6g}")
                if self.min_line:
                    self.min_line.set_xdata([new_val, new_val])
        elif self.dragging_line == 'max':
            if new_val > vmin:
                self.state.set_limits(vmin, new_val)
                self.max_edit.setText(f"{new_val:.6g}")
                if self.max_line:
                    self.max_line.set_xdata([new_val, new_val])
                    
        self.canvas.draw_idle()
        if self.on_limits_changed:
            self.on_limits_changed()
