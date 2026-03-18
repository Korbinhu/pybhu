import os

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from PyQt6.QtCore import QSettings, QStandardPaths, QTimer, Qt
from PyQt6.QtGui import QDragEnterEvent, QDropEvent
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

from .colormaps import available_colormaps, resolve_colormap
from .histogram_dialog import HistogramDialog
from .fft_dialog import FFTDialog
from .state import ViewerState
from .loader import (
    ARCHIVE_EXTENSIONS,
    SUPPORTED_EXTENSIONS,
    find_showable_data,
    get_archive_contents,
    load_data,
    to_numpy,
)


def create_viewer_settings():
    settings_path = os.environ.get("IMG_VIEWER_SETTINGS_PATH")
    if not settings_path:
        config_root = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppConfigLocation)
        if not config_root:
            config_root = os.path.expanduser("~")
        settings_path = os.path.join(config_root, "PyBHU", "img_viewer.ini")

    settings_dir = os.path.dirname(settings_path)
    if settings_dir:
        os.makedirs(settings_dir, exist_ok=True)

    return QSettings(settings_path, QSettings.Format.IniFormat)


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


class ViewerNavigationToolbar(NavigationToolbar2QT):
    def __init__(self, canvas, viewer):
        self.viewer = viewer
        super().__init__(canvas, viewer)

    def save_figure(self, *args):
        return self.viewer.export_current_view()


class ImageStackViewer(QMainWindow):
    def __init__(self, data=None, **options):
        self.app = QApplication.instance()
        self.owns_app = self.app is None
        if self.app is None:
            self.app = QApplication([])
        self.settings = create_viewer_settings()
        self.settings.sync()
        super().__init__()
        self.setWindowTitle("PyBHU Scientific Image Viewer")
        self.resize(1200, 850)
        self._is_centered = False
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

        if "initial_layer" in options:
            initial_layer = options.pop("initial_layer")
            options.pop("current_layer", None)  # discard alias if both supplied
        else:
            initial_layer = options.pop("current_layer", 0)
        if "colormap" in options and "colormap_name" not in options:
            options["colormap_name"] = options.pop("colormap")

        self.default_colormap_name = options.pop(
            "colormap_name",
            self.settings.value("viewer/colormap_name", "viridis"),
        )
        self.default_inverted = bool(
            options.pop("inverted", self.settings.value("viewer/inverted", False))
        )
        self.default_color_limit_mode = options.pop(
            "color_limit_mode",
            self.settings.value("viewer/color_limit_mode", "global"),
        )
        self.default_origin = options.pop(
            "origin",
            self.settings.value("viewer/origin", "lower"),
        )
        if self.default_color_limit_mode not in {"global", "per_layer"}:
            raise ValueError("color_limit_mode must be 'global' or 'per_layer'")
        if self.default_origin not in {"lower", "upper"}:
            raise ValueError("origin must be 'lower' or 'upper'")
        resolve_colormap(self.default_colormap_name, self.default_inverted)

        self.state = None
        if data is not None:
            self.state = ViewerState(
                to_numpy(data),
                current_layer=initial_layer,
                color_limit_mode=self.default_color_limit_mode,
                colormap_name=self.default_colormap_name,
                inverted=self.default_inverted,
                origin=self.default_origin,
                **options,
            )
             
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
        self.colorbar_figure = Figure(figsize=(0.8, 5), facecolor="#ffffff", tight_layout=True)
        self.colorbar_canvas = FigureCanvasQTAgg(self.colorbar_figure)
        self.colorbar_canvas.setFixedWidth(60)

        # Navigation Toolbar
        self.nav_toolbar = ViewerNavigationToolbar(self.canvas, self)
        canvas_layout.addWidget(self.nav_toolbar)
        plot_widget = QWidget()
        plot_layout = QHBoxLayout(plot_widget)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.setSpacing(8)
        plot_layout.addWidget(self.canvas, 1)
        plot_layout.addWidget(self.colorbar_canvas, 0, Qt.AlignmentFlag.AlignTop)
        canvas_layout.addWidget(plot_widget)

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
        
        self.load_button = QPushButton("Open File")
        self.load_button.setObjectName("LoadButton")
        self.load_button.clicked.connect(self.open_file_dialog)
        data_layout.addWidget(self.load_button)

        self.dataset_label = QLabel("Select Dataset:")
        self.dataset_label.setVisible(False)
        data_layout.addWidget(self.dataset_label)

        self.dataset_selector = QComboBox()
        self.dataset_selector.currentIndexChanged.connect(self.on_dataset_switched)
        self.dataset_selector.setEnabled(False)
        self.dataset_selector.setVisible(False)
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
        self.colormap_selector.setCurrentText(self.default_colormap_name)
        self.colormap_selector.currentTextChanged.connect(self.on_colormap_changed)
        cmap_layout.addWidget(self.colormap_selector)

        invert_layout = QHBoxLayout()
        invert_layout.addWidget(QLabel("Invert Colormap:"))
        self.invert_checkbox = QCheckBox()
        self.invert_checkbox.setChecked(self.default_inverted)
        self.invert_checkbox.toggled.connect(self.toggle_inverted)
        invert_layout.addWidget(self.invert_checkbox)
        invert_layout.addStretch()
        cmap_layout.addLayout(invert_layout)

        cmap_layout.addWidget(QLabel("Image Origin:"))
        self.origin_selector = QComboBox()
        self.origin_selector.addItem("Lower", userData="lower")
        self.origin_selector.addItem("Upper", userData="upper")
        self.origin_selector.setCurrentIndex(0 if self.default_origin == "lower" else 1)
        self.origin_selector.currentIndexChanged.connect(self.on_origin_changed)
        cmap_layout.addWidget(self.origin_selector)
        
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
        self.main_splitter = main_splitter

        # Status Bar
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.canvas.mpl_connect("motion_notify_event", self.on_mouse_move)

        self.restore_persisted_layout()

        self.update_ui_from_state()
        self.refresh_image()
        self.setAcceptDrops(True)

    def showEvent(self, event):
        super().showEvent(event)
        if not self._is_centered:
            self.center_on_screen()
            self._is_centered = True
        # Post to the next event-loop iteration so the window manager has
        # had a chance to actually map the window before we raise/activate it.
        QTimer.singleShot(0, self.ensure_window_visible)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if self._first_supported_url(event):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent):
        file_path = self._first_supported_url(event)
        if not file_path:
            event.ignore()
            return

        try:
            if self.load_path(file_path):
                event.acceptProposedAction()
            else:
                event.ignore()
        except Exception as exc:
            QMessageBox.critical(self, "Load Error", f"Failed to load file:\n{str(exc)}")
            event.ignore()

    def closeEvent(self, event):
        self.persist_settings()
        super().closeEvent(event)

    def center_on_screen(self):
        screen = self.screen() or self.app.primaryScreen()
        if screen is None:
            return
        available = screen.availableGeometry()
        frame = self.frameGeometry()
        frame.moveCenter(available.center())
        self.move(frame.topLeft())

    def ensure_window_visible(self):
        self.showNormal()
        if not self._is_centered:
            self.center_on_screen()
            self._is_centered = True
        self.raise_()
        self.activateWindow()

    def restore_persisted_layout(self):
        geometry = self.settings.value("window/geometry")
        if geometry is not None and self.restoreGeometry(geometry):
            self._is_centered = True

        splitter_state = self.settings.value("window/splitter_state")
        if splitter_state is not None:
            self.main_splitter.restoreState(splitter_state)

    def persist_settings(self):
        self.settings.setValue("window/geometry", self.saveGeometry())
        self.settings.setValue("window/splitter_state", self.main_splitter.saveState())
        self.settings.setValue("viewer/colormap_name", self.current_colormap_name())
        self.settings.setValue("viewer/inverted", self.current_inverted())
        self.settings.setValue("viewer/color_limit_mode", self.current_color_limit_mode())
        self.settings.setValue("viewer/origin", self.current_origin())
        self.settings.sync()

    def current_colormap_name(self):
        if self.state:
            return self.state.colormap_name
        return self.colormap_selector.currentText() or self.default_colormap_name

    def current_inverted(self):
        if self.state:
            return self.state.inverted
        return self.invert_checkbox.isChecked()

    def current_color_limit_mode(self):
        if self.state:
            return self.state.color_limit_mode
        return self.default_color_limit_mode

    def current_origin(self):
        if self.state:
            return self.state.origin
        return self.origin_selector.currentData() or self.default_origin

    def _first_supported_url(self, event):
        mime_data = event.mimeData()
        if not mime_data.hasUrls():
            return None

        for url in mime_data.urls():
            if not url.isLocalFile():
                continue
            local_path = url.toLocalFile()
            if self.is_supported_path(local_path):
                return local_path
        return None

    def set_controls_enabled(self, enabled):
        self.colormap_selector.setEnabled(enabled)
        self.invert_checkbox.setEnabled(enabled)
        self.origin_selector.setEnabled(enabled)
        self.histogram_button.setEnabled(enabled)
        self.fft_button.setEnabled(enabled)
        self.layer_slider.setEnabled(enabled)

    def _clear_dataset_selector(self):
        self.available_datasets = []
        self.dataset_selector.blockSignals(True)
        self.dataset_selector.clear()
        self.dataset_selector.blockSignals(False)
        self.dataset_selector.setEnabled(False)
        self.dataset_selector.setVisible(False)
        self.dataset_label.setVisible(False)

    def _populate_dataset_selector(self, showable, selected_index):
        self.available_datasets = list(showable)
        self.dataset_selector.blockSignals(True)
        self.dataset_selector.clear()
        for name, data in showable:
            self.dataset_selector.addItem(f"{name} ({'x'.join(map(str, data.shape))})")
        self.dataset_selector.setCurrentIndex(selected_index)
        self.dataset_selector.blockSignals(False)
        has_multiple = len(showable) > 1
        self.dataset_selector.setEnabled(has_multiple)
        self.dataset_selector.setVisible(has_multiple)
        self.dataset_label.setVisible(has_multiple)

    def update_ui_from_state(self):
        if not self.state:
            self.layer_slider.blockSignals(True)
            self.layer_slider.setMinimum(0)
            self.layer_slider.setMaximum(0)
            self.layer_slider.setValue(0)
            self.layer_slider.blockSignals(False)
            self.layer_label.setText("Open a file")
            self.colormap_selector.blockSignals(True)
            self.colormap_selector.setCurrentText(self.default_colormap_name)
            self.colormap_selector.blockSignals(False)
            self.invert_checkbox.blockSignals(True)
            self.invert_checkbox.setChecked(self.default_inverted)
            self.invert_checkbox.blockSignals(False)
            self.origin_selector.blockSignals(True)
            self.origin_selector.setCurrentIndex(0 if self.default_origin == "lower" else 1)
            self.origin_selector.blockSignals(False)
            self.set_controls_enabled(False)
            self.statusBar.showMessage("Open or drop a .npy, .pkl, .npz, .mat, .sxm, .3ds, .pt, or .pth file to begin.")
            return
        self.layer_slider.setMinimum(0)
        self.layer_slider.setMaximum(self.state.layer_count - 1)
        self.layer_slider.setValue(self.state.current_layer)
        self.set_controls_enabled(True)
        
        if self.state.layer_count <= 1:
            self.layer_label.setText("1 / 1")
            self.layer_slider.setEnabled(False)
        else:
            self.layer_label.setText(f"{self.state.current_layer + 1} / {self.state.layer_count}")
            self.layer_slider.setEnabled(True)
            
        self.colormap_selector.blockSignals(True)
        self.colormap_selector.setCurrentText(self.state.colormap_name)
        self.colormap_selector.blockSignals(False)
        self.invert_checkbox.blockSignals(True)
        self.invert_checkbox.setChecked(self.state.inverted)
        self.invert_checkbox.blockSignals(False)
        self.origin_selector.blockSignals(True)
        self.origin_selector.setCurrentIndex(0 if self.state.origin == "lower" else 1)
        self.origin_selector.blockSignals(False)

    def show_placeholder(self):
        self.axes.clear()
        self.axes.set_axis_off()
        self.axes.text(
            0.5,
            0.5,
            "Open or drop a .npy, .pkl, .npz, .mat, .sxm, .3ds, .pt, or .pth file to begin",
            ha="center",
            va="center",
            fontsize=14,
            color="#555555",
            transform=self.axes.transAxes,
        )
        self.canvas.draw_idle()

    def refresh_colorbar(self):
        self.colorbar_figure.clear()
        axes = self.colorbar_figure.add_subplot(111)

        if not self.state:
            axes.set_axis_off()
            axes.text(
                0.5,
                0.5,
                "No Data",
                ha="center",
                va="center",
                fontsize=10,
                color="#555555",
                rotation=90,
                transform=axes.transAxes,
            )
            self.colorbar_canvas.draw_idle()
            return

        cmap = resolve_colormap(self.state.colormap_name, self.state.inverted)
        vmin, vmax = self.state.visible_limits()
        if vmin == vmax:
            vmax = vmin + 1.0

        mappable = ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=cmap)
        colorbar = self.colorbar_figure.colorbar(mappable, cax=axes)
        colorbar.set_ticks([])
        colorbar.ax.tick_params(left=False, right=False, labelleft=False, labelright=False, size=0)
        self.colorbar_canvas.draw_idle()

    def export_current_view(self):
        if not self.state:
            QMessageBox.warning(self, "No Data", "Load a dataset before exporting an image.")
            return ""

        filetypes = self.canvas.get_supported_filetypes_grouped()
        sorted_filetypes = sorted(filetypes.items())
        default_filetype = self.canvas.get_default_filetype()
        start = self.canvas.get_default_filename()
        filters = []
        selected_filter = None

        for name, exts in sorted_filetypes:
            exts_list = " ".join(f"*.{ext}" for ext in exts)
            current_filter = f"{name} ({exts_list})"
            if default_filetype in exts:
                selected_filter = current_filter
            filters.append(current_filter)

        fname, _ = QFileDialog.getSaveFileName(
            self,
            "Choose a filename to save to",
            start,
            ";;".join(filters),
            selected_filter,
        )
        if not fname:
            return ""

        try:
            self._save_export_figure(fname)
            self.statusBar.showMessage(f"Exported image: {fname}", 5000)
        except Exception as exc:
            QMessageBox.critical(self, "Error saving file", str(exc))
            return ""

        return fname

    def _save_export_figure(self, path):
        layer = self.state.data[:, :, self.state.current_layer]
        cmap = resolve_colormap(self.state.colormap_name, self.state.inverted)
        vmin, vmax = self.state.visible_limits()
        if vmin == vmax:
            vmax = vmin + 1.0

        export_figure = Figure(figsize=(8.5, 6.5), facecolor="#ffffff", constrained_layout=True)
        grid = export_figure.add_gridspec(1, 2, width_ratios=[20, 1], wspace=0.04)
        image_axes = export_figure.add_subplot(grid[0, 0])
        colorbar_axes = export_figure.add_subplot(grid[0, 1])

        image = image_axes.imshow(
            layer,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
            origin=self.state.origin,
        )
        image_axes.set_axis_off()
        colorbar = export_figure.colorbar(image, cax=colorbar_axes)
        colorbar.set_ticks([])
        colorbar.ax.tick_params(left=False, right=False, labelleft=False, labelright=False, size=0)
        export_figure.savefig(path, dpi=300, facecolor="#ffffff")

    def supported_file_filter(self):
        patterns = " ".join(f"*{ext}" for ext in SUPPORTED_EXTENSIONS)
        return f"Data Files ({patterns});;All Files (*)"

    def is_supported_path(self, path):
        return os.path.isfile(path) and path.lower().endswith(SUPPORTED_EXTENSIONS)

    def _select_archive_dataset(self, showable, dataset_index=None, dataset_name=None):
        if dataset_index is not None and dataset_name is not None:
            raise ValueError("dataset_index and dataset_name cannot both be provided")

        if dataset_name is not None:
            for index, (name, data) in enumerate(showable):
                if name == dataset_name:
                    return index, data
            available = ", ".join(name for name, _ in showable)
            raise ValueError(
                f"Unknown dataset_name: {dataset_name!r}. Available datasets: {available}"
            )

        if dataset_index is not None:
            if not 0 <= dataset_index < len(showable):
                raise ValueError(
                    f"dataset_index {dataset_index} is out of range for {len(showable)} datasets"
                )
            return dataset_index, showable[dataset_index][1]

        if len(showable) > 1:
            dlg = ArchiveSelectionDialog(showable, self)
            if dlg.exec():
                return dlg.selected_index, showable[dlg.selected_index][1]
            return None, None

        return 0, showable[0][1]

    def load_path(self, file_path, dataset_index=None, dataset_name=None):
        if not self.is_supported_path(file_path):
            raise ValueError(
                f"Unsupported file extension: {os.path.splitext(file_path)[1].lower()}. "
                f"Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
            )

        if file_path.lower().endswith(ARCHIVE_EXTENSIONS):
            obj = get_archive_contents(file_path)
            showable = find_showable_data(obj)

            if not showable:
                QMessageBox.warning(self, "No Showable Data", "No 2D/3D datasets found in archive.")
                return False

            selected_index, data = self._select_archive_dataset(
                showable,
                dataset_index=dataset_index,
                dataset_name=dataset_name,
            )
            if data is None:
                return False
            self.load_data(data)
            self._populate_dataset_selector(showable, selected_index)
        else:
            if dataset_index is not None or dataset_name is not None:
                raise ValueError(
                    "dataset_index and dataset_name are only supported for archive-style files"
                )
            data = load_data(file_path)
            self.load_data(data)
            self._clear_dataset_selector()
        self.statusBar.showMessage(f"File Loaded: {file_path}", 5000)
        return True

    def open_file_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Data File", "", self.supported_file_filter()
        )
        if not file_path: return
        
        try:
            self.load_path(file_path)
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load file:\n{str(e)}")

    def on_dataset_switched(self, index):
        if 0 <= index < len(self.available_datasets):
            self.load_data(self.available_datasets[index][1])

    def load_data(self, data):
        old_cmap = self.state.colormap_name if self.state else self.colormap_selector.currentText()
        old_inv = self.state.inverted if self.state else self.invert_checkbox.isChecked()
        old_mode = self.state.color_limit_mode if self.state else self.default_color_limit_mode
        old_origin = self.state.origin if self.state else self.origin_selector.currentData()

        self.state = ViewerState(
            to_numpy(data),
            color_limit_mode=old_mode,
            colormap_name=old_cmap,
            inverted=old_inv,
            origin=old_origin,
        )
        self.default_colormap_name = old_cmap
        self.default_inverted = old_inv
        self.default_color_limit_mode = old_mode
        self.default_origin = old_origin
         
        # Force a brand new image artist by clearing axes
        self.axes.clear()
        self.image_artist = None 
        
        self.update_ui_from_state()
        self.refresh_image()
        
        if getattr(self, "histogram_dialog", None):
            self.histogram_dialog.state = self.state
            self.histogram_dialog.refresh_from_state()
        if getattr(self, "fft_dialog", None):
            self.fft_dialog.state = self.state

    def refresh_image(self):
        if not self.state:
            self.image_artist = None
            self.show_placeholder()
            self.refresh_colorbar()
            return
        layer = self.state.data[:, :, self.state.current_layer]
        cmap = resolve_colormap(self.state.colormap_name, self.state.inverted)
        vmin, vmax = self.state.visible_limits()
        origin = self.state.origin
        
        if self.image_artist is None or self.image_artist.origin != origin:
            self.axes.clear()
            self.image_artist = self.axes.imshow(
                layer,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                interpolation='nearest',
                origin=origin,
            )
            self.axes.set_axis_off()
        else:
            self.image_artist.set_data(layer)
            self.image_artist.set_cmap(cmap)
            self.image_artist.set_clim(vmin, vmax)
        
        self.canvas.draw_idle()
        self.refresh_colorbar()

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

    def on_origin_changed(self, index: int):
        origin = self.origin_selector.itemData(index)
        if origin is None:
            return
        if self.state:
            self.state.set_origin(origin)
            self.refresh_image()
        else:
            self.default_origin = origin

    def refresh_aux_windows(self):
        if getattr(self, "histogram_dialog", None) is not None:
            self.histogram_dialog.refresh_from_state()
