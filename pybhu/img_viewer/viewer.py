from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from .colorbar_dialog import ColorbarDialog
from .colormaps import available_colormaps, resolve_colormap
from .histogram_dialog import HistogramDialog
from .fft_dialog import FFTDialog
from .state import ViewerState
from .loader import load_data, get_archive_contents, find_showable_data, to_numpy


class ArchiveSelectionDialog(QDialog):
    def __init__(self, items, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Dataset from Archive")
        self.resize(450, 350)
        self.setStyleSheet("color: #000000; background-color: #ffffff;")
        
        layout = QVBoxLayout(self)
        label = QLabel("Multiple datasets found. Choose one to load:")
        label.setStyleSheet("font-weight: bold; margin-bottom: 5px; color: #000000;")
        layout.addWidget(label)
        
        self.list_widget = QListWidget()
        self.list_widget.setStyleSheet("border: 1px solid #888888; color: #000000; background-color: #ffffff;")
        for name, data in items:
            shape_str = f" (shape: {data.shape})"
            self.list_widget.addItem(f"{name}{shape_str}")
        
        layout.addWidget(self.list_widget)
        
        btn_layout = QHBoxLayout()
        self.ok_btn = QPushButton("Load Selected")
        self.ok_btn.setStyleSheet("color: white; background-color: #007bff; padding: 10px;")
        self.ok_btn.clicked.connect(self.accept)
        btn_layout.addWidget(self.ok_btn)
        
        layout.addLayout(btn_layout)
        self.selected_index = -1
        
    def accept(self):
        self.selected_index = self.list_widget.currentRow()
        if self.selected_index >= 0:
            super().accept()


class ImageStackViewer(QMainWindow):
    def __init__(self, data=None, **options):
        self.app = QApplication.instance()
        if self.app is None:
            self.app = QApplication([])
        super().__init__()
        self.setWindowTitle("PyBHU Scientific Image Viewer")
        self.resize(1200, 850)
        
        self.available_datasets = [] 
        
        # UI Styling - Force High Visibility Black on White
        self.setStyleSheet("""
            QMainWindow, QDialog {
                background-color: #ffffff;
                color: #000000;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QLabel {
                color: #000000;
            }
            .ControlPanel {
                background-color: #f0f0f0;
                border-left: 2px solid #cccccc;
            }
            .ControlGroup {
                border: 1px solid #cccccc;
                border-radius: 4px;
                background-color: #ffffff;
                padding: 10px;
                margin-bottom: 10px;
            }
            .GroupTitle {
                font-weight: bold;
                color: #000000;
                margin-bottom: 5px;
                border-bottom: 1px solid #aaaaaa;
            }
            QPushButton {
                background-color: #e1e1e1;
                color: #000000;
                border: 1px solid #adadad;
                border-radius: 3px;
                padding: 6px;
                min-height: 20px;
            }
            QPushButton:hover {
                background-color: #e5f1fb;
                border: 1px solid #0078d7;
            }
            QPushButton#LoadButton {
                background-color: #d4edda;
                border: 1px solid #c3e6cb;
            }
            QPushButton#LoadButton:hover {
                background-color: #c3e6cb;
            }
            QComboBox {
                border: 1px solid #cccccc;
                background-color: white;
                color: black;
                padding: 3px;
                min-height: 25px;
            }
            QComboBox QAbstractItemView {
                background-color: white;
                color: black;
                selection-background-color: #0078d7;
            }
            QSlider::groove:horizontal {
                border: 1px solid #999999;
                height: 8px;
                background: #eeeeee;
                margin: 2px 0;
            }
            QSlider::handle:horizontal {
                background: #0078d7;
                border: 1px solid #005a9e;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
            QStatusBar {
                border-top: 1px solid #cccccc;
                background-color: #f0f0f0;
                color: #000000;
            }
            QToolBar {
                background-color: #f0f0f0;
                border-bottom: 1px solid #cccccc;
            }
            QToolButton {
                background-color: transparent;
                border: 1px solid transparent;
                border-radius: 3px;
            }
            QToolButton:hover {
                background-color: #e5f1fb;
                border: 1px solid #0078d7;
            }
        """)

        initial_layer = options.pop("initial_layer", 0)
        if "colormap" in options and "colormap_name" not in options:
            options["colormap_name"] = options.pop("colormap")
        
        if data is None:
            data = np.zeros((256, 256))
        
        self.state = ViewerState(to_numpy(data), **options)
        self.state.set_current_layer(initial_layer)
            
        self.image_artist = None

        # Main Layout using Splitter
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # --- Left Area: Canvas and Navigation ---
        canvas_container = QWidget()
        canvas_layout = QVBoxLayout(canvas_container)
        canvas_layout.setContentsMargins(5, 5, 5, 5)

        # Force a distinct style for the toolbar container
        self.figure = Figure(facecolor="#ffffff", tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.axes = self.figure.add_subplot(111)
        
        # Navigation Toolbar
        self.nav_toolbar = NavigationToolbar2QT(self.canvas, self)
        canvas_layout.addWidget(self.nav_toolbar)
        canvas_layout.addWidget(self.canvas)

        # --- Right Area: Controls Sidebar ---
        sidebar = QWidget()
        sidebar.setObjectName("sidebar")
        sidebar.setProperty("class", "ControlPanel")
        sidebar.setFixedWidth(300)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        sidebar_layout.setContentsMargins(15, 15, 15, 15)
        sidebar_layout.setSpacing(15)

        # Data Management Group
        data_group = QFrame()
        data_group.setProperty("class", "ControlGroup")
        data_layout = QVBoxLayout(data_group)
        data_layout.setContentsMargins(10, 10, 10, 10)
        data_layout.setSpacing(8)
        title = QLabel("DATA MANAGEMENT")
        title.setProperty("class", "GroupTitle")
        data_layout.addWidget(title)
        
        self.load_button = QPushButton("Open File (.npy, .pkl, .npz)")
        self.load_button.setObjectName("LoadButton")
        self.load_button.clicked.connect(self.open_file_dialog)
        data_layout.addWidget(self.load_button)
        
        self.dataset_selector = QComboBox()
        self.dataset_selector.currentIndexChanged.connect(self.on_dataset_switched)
        self.dataset_selector.setEnabled(False)
        data_layout.addWidget(QLabel("Select Dataset:"))
        data_layout.addWidget(self.dataset_selector)
        
        sidebar_layout.addWidget(data_group)

        # Navigation Group (Moved from Bottom to Sidebar)
        nav_group = QFrame()
        nav_group.setProperty("class", "ControlGroup")
        nav_layout = QVBoxLayout(nav_group)
        nav_layout.setContentsMargins(10, 10, 10, 10)
        nav_layout.setSpacing(8)
        title = QLabel("NAVIGATION")
        title.setProperty("class", "GroupTitle")
        nav_layout.addWidget(title)
        
        self.layer_slider = QSlider(Qt.Orientation.Horizontal)
        self.layer_slider.valueChanged.connect(self.on_layer_changed)
        nav_layout.addWidget(QLabel("Layer Slider:"))
        nav_layout.addWidget(self.layer_slider)

        self.layer_label = QLabel("0 / 0")
        self.layer_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layer_label.setStyleSheet("font-weight: bold; font-size: 14px; margin-top: 5px;")
        nav_layout.addWidget(self.layer_label)
        
        sidebar_layout.addWidget(nav_group)

        # Appearance Group
        cmap_group = QFrame()
        cmap_group.setProperty("class", "ControlGroup")
        cmap_layout = QVBoxLayout(cmap_group)
        cmap_layout.setContentsMargins(10, 10, 10, 10)
        cmap_layout.setSpacing(8)
        title = QLabel("VISUALIZATION")
        title.setProperty("class", "GroupTitle")
        cmap_layout.addWidget(title)
        
        cmap_layout.addWidget(QLabel("Colormap Palette:"))
        self.colormap_selector = QComboBox()
        self.colormap_selector.addItems(available_colormaps())
        self.colormap_selector.setCurrentText(self.state.colormap_name)
        self.colormap_selector.currentTextChanged.connect(self.on_colormap_changed)
        cmap_layout.addWidget(self.colormap_selector)

        invert_layout = QHBoxLayout()
        invert_layout.addWidget(QLabel("Invert Colormap:"))
        self.invert_checkbox = QCheckBox()
        self.invert_checkbox.setChecked(self.state.inverted)
        self.invert_checkbox.toggled.connect(self.toggle_inverted)
        invert_layout.addWidget(self.invert_checkbox)
        invert_layout.addStretch()
        cmap_layout.addLayout(invert_layout)
        
        sidebar_layout.addWidget(cmap_group)

        # Analysis Group
        analysis_group = QFrame()
        analysis_group.setProperty("class", "ControlGroup")
        analysis_layout = QVBoxLayout(analysis_group)
        analysis_layout.setContentsMargins(10, 10, 10, 10)
        analysis_layout.setSpacing(8)
        title = QLabel("SCIENTIFIC TOOLS")
        title.setProperty("class", "GroupTitle")
        analysis_layout.addWidget(title)

        self.histogram_button = QPushButton("Analyze Histogram")
        self.histogram_button.clicked.connect(self.open_histogram_dialog)
        analysis_layout.addWidget(self.histogram_button)

        self.colorbar_button = QPushButton("Show Intensity Bar")
        self.colorbar_button.clicked.connect(self.open_colorbar_dialog)
        analysis_layout.addWidget(self.colorbar_button)

        self.fft_button = QPushButton("Analyze FFT")
        self.fft_button.clicked.connect(self.open_fft_dialog)
        analysis_layout.addWidget(self.fft_button)
        
        sidebar_layout.addWidget(analysis_group)

        sidebar_layout.addStretch()

        # Final Assembly
        main_splitter.addWidget(canvas_container)
        main_splitter.addWidget(sidebar)
        main_splitter.setStretchFactor(0, 1)
        main_splitter.setStretchFactor(1, 0)
        main_splitter.setHandleWidth(4)

        self.setCentralWidget(main_splitter)

        # Status Bar
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.canvas.mpl_connect("motion_notify_event", self.on_mouse_move)

        self.update_ui_from_state()
        self.refresh_image()

    def set_controls_enabled(self, enabled):
        self.colormap_selector.setEnabled(enabled)
        self.invert_checkbox.setEnabled(enabled)
        self.histogram_button.setEnabled(enabled)
        self.colorbar_button.setEnabled(enabled)
        self.fft_button.setEnabled(enabled)
        self.layer_slider.setEnabled(enabled)

    def update_ui_from_state(self):
        if not self.state: return
        self.layer_slider.setMinimum(0)
        self.layer_slider.setMaximum(self.state.layer_count - 1)
        self.layer_slider.setValue(self.state.current_layer)
        
        if self.state.layer_count <= 1:
            self.layer_label.setText("1 / 1")
            self.layer_slider.setEnabled(False)
        else:
            self.layer_label.setText(f"{self.state.current_layer + 1} / {self.state.layer_count}")
            self.layer_slider.setEnabled(True)
            
        self.colormap_selector.setCurrentText(self.state.colormap_name)
        self.invert_checkbox.setChecked(self.state.inverted)
        self.set_controls_enabled(True)

    def open_file_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Data File", "", "Data Files (*.npy *.pkl *.npz);;All Files (*)"
        )
        if not file_path: return
        
        try:
            if file_path.lower().endswith((".pkl", ".npz")):
                obj = get_archive_contents(file_path)
                showable = find_showable_data(obj)
                
                if not showable:
                    QMessageBox.warning(self, "No Showable Data", "No 2D/3D datasets found in archive.")
                    return
                
                self.available_datasets = showable
                self.dataset_selector.blockSignals(True)
                self.dataset_selector.clear()
                for name, data in showable:
                    self.dataset_selector.addItem(f"{name} ({'x'.join(map(str, data.shape))})")
                self.dataset_selector.blockSignals(False)
                self.dataset_selector.setEnabled(True)
                
                if len(showable) > 1:
                    dlg = ArchiveSelectionDialog(showable, self)
                    if dlg.exec():
                        self.dataset_selector.setCurrentIndex(dlg.selected_index)
                        data = showable[dlg.selected_index][1]
                    else:
                        return
                else:
                    self.dataset_selector.setCurrentIndex(0)
                    data = showable[0][1]
            else:
                self.dataset_selector.clear()
                self.dataset_selector.setEnabled(False)
                self.available_datasets = []
                data = load_data(file_path)
            
            self.load_data(data)
            self.statusBar.showMessage(f"File Loaded: {file_path}", 5000)
            
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load file:\n{str(e)}")

    def on_dataset_switched(self, index):
        if 0 <= index < len(self.available_datasets):
            data = self.available_datasets[index][1]
            self.load_data(data)

    def load_data(self, data):
        old_cmap = self.state.colormap_name if self.state else "viridis"
        old_inv = self.state.inverted if self.state else False
        
        self.state = ViewerState(to_numpy(data))
        self.state.set_colormap(old_cmap)
        self.state.set_inverted(old_inv)
        
        # Reset color limits flag specifically
        self.state.custom_limits = False
        
        # Force a brand new image artist by clearing axes
        self.axes.clear()
        self.image_artist = None 
        
        self.update_ui_from_state()
        self.refresh_image()
        
        if getattr(self, "histogram_dialog", None):
            self.histogram_dialog.state = self.state
            self.histogram_dialog.refresh_from_state()
        if getattr(self, "colorbar_dialog", None):
            self.colorbar_dialog.state = self.state
            self.colorbar_dialog.refresh()

    def refresh_image(self):
        if not self.state: return
        layer = self.state.data[:, :, self.state.current_layer]
        cmap = resolve_colormap(self.state.colormap_name, self.state.inverted)
        vmin, vmax = self.state.visible_limits()
        
        if self.image_artist is None:
            self.image_artist = self.axes.imshow(layer, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest', origin='lower')
            self.axes.set_axis_off()
        else:
            self.image_artist.set_data(layer)
            self.image_artist.set_cmap(cmap)
            self.image_artist.set_clim(vmin, vmax)
        
        self.canvas.draw_idle()

    def on_layer_changed(self, index: int):
        if not self.state: return
        self.state.set_current_layer(index)
        if self.state.layer_count > 1:
            self.layer_label.setText(f"{index + 1} / {self.state.layer_count}")
        self.refresh_image()
        self.refresh_aux_windows()

    def on_mouse_move(self, event):
        if self.state and event.inaxes == self.axes:
            x, y = int(event.xdata + 0.5), int(event.ydata + 0.5)
            if 0 <= x < self.state.data.shape[1] and 0 <= y < self.state.data.shape[0]:
                val = self.state.data[y, x, self.state.current_layer]
                self.statusBar.showMessage(f"Cursor: ({x}, {y}) | Value: {val:.6g}")
        else:
            self.statusBar.clearMessage()

    def open_histogram_dialog(self):
        if not self.state: return
        if getattr(self, "histogram_dialog", None) is None:
            self.histogram_dialog = HistogramDialog(
                self.state,
                on_limits_changed=self._on_limits_changed,
                parent=self,
            )
        self.histogram_dialog.show()
        self.histogram_dialog.raise_()
        self.histogram_dialog.activateWindow()

    def _on_limits_changed(self):
        self.refresh_image()
        if getattr(self, "colorbar_dialog", None) is not None:
            self.colorbar_dialog.refresh()

    def open_fft_dialog(self):
        if not self.state: return
        if getattr(self, "fft_dialog", None) is None:
            self.fft_dialog = FFTDialog(self.state, parent=self)
        self.fft_dialog.show()
        self.fft_dialog.raise_()
        self.fft_dialog.activateWindow()

    def on_colormap_changed(self, name: str):
        if self.state:
            self.state.set_colormap(name)
            self.refresh_image()
            self.refresh_aux_windows()

    def toggle_inverted(self, checked: bool = True):
        if self.state:
            self.state.set_inverted(checked)
            self.refresh_image()
            self.refresh_aux_windows()

    def open_colorbar_dialog(self):
        if not self.state: return
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
