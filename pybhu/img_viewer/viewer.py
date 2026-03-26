import os

import matplotlib.patheffects as patheffects
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from PyQt6.QtCore import QSettings, QStandardPaths, QTimer, Qt
from PyQt6.QtGui import QDragEnterEvent, QDropEvent, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from .colormaps import available_colormaps, resolve_colormap
from .histogram_dialog import HistogramDialog
from .fft_dialog import FFTDialog
from .spectrum_dialog import SpectrumDialog
from .state import ViewerState, normalize_spectrum_axis
from .loader import (
    ARCHIVE_EXTENSIONS,
    SUPPORTED_EXTENSIONS,
    describe_supported_path,
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
        self.available_file_paths = []
        self.current_source_dir = ""

        self._apply_stylesheet()

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
        self.legacy_stm = options.pop("legacy_stm", "auto")
        if self.legacy_stm not in {"auto", "stm1", "stm2"}:
            raise ValueError("legacy_stm must be 'auto', 'stm1', or 'stm2'")
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
        self.spectrum_points = []
        self.spectrum_markers = []
        self.spectrum_axis = None
        self.spectrum_axis_label = "Layer"
        self.child_viewers = []
        self._limits_history = []
        self._limits_redo = []

        canvas_container = self._build_canvas_area()
        sidebar_scroll = self._build_sidebar()
        self._build_main_layout(canvas_container, sidebar_scroll)
        self._build_status_bar()
        self._connect_canvas_events()
        self._build_shortcuts()

        self.restore_persisted_layout()

        self.update_ui_from_state()
        self.refresh_image()
        self.setAcceptDrops(True)

    # ------------------------------------------------------------------
    # UI Building helpers (extracted from __init__)
    # ------------------------------------------------------------------

    def _apply_stylesheet(self):
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
            QPushButton:disabled {
                background-color: #f5f5f5;
                color: #aaaaaa;
                border: 1px solid #d9d9d9;
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
            QListWidget {
                background-color: #ffffff;
                color: #000000;
                border: 1px solid #cccccc;
                font-size: 12px;
            }
            QListWidget::item {
                padding: 3px 5px;
            }
            QListWidget::item:selected {
                background-color: #0078d7;
                color: #ffffff;
            }
            QListWidget::item:hover {
                background-color: #e5f1fb;
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

    def _build_canvas_area(self):
        canvas_container = QWidget()
        canvas_layout = QVBoxLayout(canvas_container)
        canvas_layout.setContentsMargins(5, 5, 5, 5)

        self.figure = Figure(facecolor="#ffffff", tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.axes = self.figure.add_subplot(111)

        self.nav_toolbar = ViewerNavigationToolbar(self.canvas, self)
        canvas_layout.addWidget(self.nav_toolbar)
        canvas_layout.addWidget(self.canvas)
        return canvas_container

    def _build_sidebar(self):
        sidebar_inner = QWidget()
        sidebar_inner.setObjectName("sidebar")
        sidebar_inner.setProperty("class", "ControlPanel")
        sidebar_layout = QVBoxLayout(sidebar_inner)
        sidebar_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        sidebar_layout.setContentsMargins(15, 15, 15, 15)
        sidebar_layout.setSpacing(15)

        sidebar_layout.addWidget(self._build_data_group())
        sidebar_layout.addWidget(self._build_nav_group())
        sidebar_layout.addWidget(self._build_visualization_group())
        sidebar_layout.addWidget(self._build_analysis_group())
        sidebar_layout.addStretch()

        sidebar_scroll = QScrollArea()
        sidebar_scroll.setWidget(sidebar_inner)
        sidebar_scroll.setWidgetResizable(True)
        sidebar_scroll.setFixedWidth(300)
        sidebar_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        sidebar_scroll.setStyleSheet("QScrollArea { border: none; background-color: #f0f0f0; }")
        return sidebar_scroll

    def _build_data_group(self):
        group = QFrame()
        group.setProperty("class", "ControlGroup")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        title = QLabel("DATA MANAGEMENT")
        title.setProperty("class", "GroupTitle")
        layout.addWidget(title)

        open_layout = QHBoxLayout()
        open_layout.setSpacing(8)
        self.load_button = QPushButton("Open File")
        self.load_button.setObjectName("LoadButton")
        self.load_button.setToolTip("Open a single data file (Ctrl+O)")
        self.load_button.clicked.connect(self.open_file_dialog)
        open_layout.addWidget(self.load_button)
        self.folder_button = QPushButton("Open Folder")
        self.folder_button.setToolTip("Open a folder and browse all supported files inside it")
        self.folder_button.clicked.connect(self.open_folder_dialog)
        open_layout.addWidget(self.folder_button)
        layout.addLayout(open_layout)

        self.file_label = QLabel("Files:")
        self.file_label.setVisible(False)
        layout.addWidget(self.file_label)
        self.file_selector = QListWidget()
        self.file_selector.currentRowChanged.connect(self.on_file_switched)
        self.file_selector.setEnabled(False)
        self.file_selector.setVisible(False)
        self.file_selector.setMaximumHeight(120)
        layout.addWidget(self.file_selector)

        self.dataset_label = QLabel("Channels:")
        self.dataset_label.setVisible(False)
        layout.addWidget(self.dataset_label)
        self.dataset_selector = QListWidget()
        self.dataset_selector.currentRowChanged.connect(self.on_dataset_switched)
        self.dataset_selector.setEnabled(False)
        self.dataset_selector.setVisible(False)
        self.dataset_selector.setMaximumHeight(160)
        layout.addWidget(self.dataset_selector)
        return group

    def _build_nav_group(self):
        group = QFrame()
        group.setProperty("class", "ControlGroup")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        self.nav_title = QLabel("NAVIGATION")
        self.nav_title.setProperty("class", "GroupTitle")
        layout.addWidget(self.nav_title)

        self.layer_slider = QSlider(Qt.Orientation.Horizontal)
        self.layer_slider.setToolTip("Navigate between layers (Left/Right arrow keys)")
        self.layer_slider.valueChanged.connect(self.on_layer_changed)
        self.layer_input = QLineEdit("—")
        self.layer_input.setFixedWidth(65)
        self.layer_input.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layer_input.setStyleSheet("font-weight: bold; font-size: 12px;")
        self.layer_input.setToolTip("Type a layer number and press Enter to jump")
        self.layer_input.returnPressed.connect(self._on_layer_input)

        slider_row = QHBoxLayout()
        slider_row.addWidget(self.layer_slider, 1)
        slider_row.addWidget(self.layer_input)
        layout.addLayout(slider_row)
        return group

    def _build_visualization_group(self):
        group = QFrame()
        group.setProperty("class", "ControlGroup")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        title = QLabel("VISUALIZATION")
        title.setProperty("class", "GroupTitle")
        layout.addWidget(title)

        self.colormap_selector = QComboBox()
        self.colormap_selector.setToolTip("Choose the color palette for image display")
        self.colormap_selector.addItems(available_colormaps())
        self.colormap_selector.setCurrentText(self.default_colormap_name)
        self.colormap_selector.currentTextChanged.connect(self.on_colormap_changed)
        cmap_row = QHBoxLayout()
        cmap_row.addWidget(QLabel("Colormap:"))
        cmap_row.addWidget(self.colormap_selector, 1)
        layout.addLayout(cmap_row)

        invert_layout = QHBoxLayout()
        invert_layout.addWidget(QLabel("Invert Colormap:"))
        self.invert_checkbox = QCheckBox()
        self.invert_checkbox.setToolTip("Reverse the direction of the color palette (I)")
        self.invert_checkbox.setChecked(self.default_inverted)
        self.invert_checkbox.toggled.connect(self.toggle_inverted)
        invert_layout.addWidget(self.invert_checkbox)
        invert_layout.addStretch()
        layout.addLayout(invert_layout)

        self.origin_selector = QComboBox()
        self.origin_selector.setToolTip("Set whether row 0 is at the bottom (Lower) or top (Upper)")
        self.origin_selector.addItem("Lower", userData="lower")
        self.origin_selector.addItem("Upper", userData="upper")
        self.origin_selector.setCurrentIndex(0 if self.default_origin == "lower" else 1)
        self.origin_selector.currentIndexChanged.connect(self.on_origin_changed)
        origin_row = QHBoxLayout()
        origin_row.addWidget(QLabel("Origin:"))
        origin_row.addWidget(self.origin_selector, 1)
        layout.addLayout(origin_row)

        self.auto_scale_button = QPushButton("Auto Scale")
        self.auto_scale_button.setToolTip("Reset color limits to fit the current layer's data range (A)")
        self.auto_scale_button.clicked.connect(self.auto_scale_current_layer)
        self.auto_scale_mode = QComboBox()
        self.auto_scale_mode.setToolTip("Choose how auto scale computes the limits")
        self.auto_scale_mode.addItem("Full Range", userData=100)
        self.auto_scale_mode.addItem("99.9%", userData=99.9)
        self.auto_scale_mode.addItem("99.5%", userData=99.5)
        self.auto_scale_mode.addItem("99%", userData=99)
        self.auto_scale_mode.addItem("98.5%", userData=98.5)
        self.auto_scale_mode.addItem("98%", userData=98)
        self.auto_scale_mode.addItem("97%", userData=97)
        self.auto_scale_mode.addItem("95%", userData=95)
        self.auto_scale_mode.addItem("90%", userData=90)
        auto_scale_row = QHBoxLayout()
        auto_scale_row.addWidget(self.auto_scale_button, 1)
        auto_scale_row.addWidget(self.auto_scale_mode)
        layout.addLayout(auto_scale_row)

        self.colorbar_figure = Figure(figsize=(5, 0.4), facecolor="#ffffff", tight_layout=True)
        self.colorbar_canvas = FigureCanvasQTAgg(self.colorbar_figure)
        self.colorbar_canvas.setFixedHeight(30)
        layout.addWidget(self.colorbar_canvas)
        return group

    def _build_analysis_group(self):
        group = QFrame()
        group.setProperty("class", "ControlGroup")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        title = QLabel("SCIENTIFIC TOOLS")
        title.setProperty("class", "GroupTitle")
        layout.addWidget(title)

        self.histogram_button = QPushButton("Analyze Histogram")
        self.histogram_button.setToolTip("View intensity histogram and adjust display limits (H)")
        self.histogram_button.clicked.connect(self.open_histogram_dialog)
        layout.addWidget(self.histogram_button)

        self.spectrum_button = QPushButton("Spectrum Viewer")
        self.spectrum_button.setToolTip("Inspect pixel values across all layers (S)")
        self.spectrum_button.clicked.connect(self.open_spectrum_dialog)
        layout.addWidget(self.spectrum_button)

        self.fft_button = QPushButton("Analyze FFT")
        self.fft_button.setToolTip("Compute 2D Fourier Transform of the data (F)")
        self.fft_button.clicked.connect(self.open_fft_dialog)
        layout.addWidget(self.fft_button)

        return group

    def _build_main_layout(self, canvas_container, sidebar_scroll):
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_splitter.addWidget(canvas_container)
        main_splitter.addWidget(sidebar_scroll)
        main_splitter.setStretchFactor(0, 1)
        main_splitter.setStretchFactor(1, 0)
        main_splitter.setHandleWidth(4)
        self.setCentralWidget(main_splitter)
        self.main_splitter = main_splitter

    def _build_status_bar(self):
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.cursor_label = QLabel("")
        self.statusBar.addPermanentWidget(self.cursor_label)
        self.dims_label = QLabel("")
        self.statusBar.addPermanentWidget(self.dims_label)

    def _connect_canvas_events(self):
        self.canvas.mpl_connect("motion_notify_event", self.on_mouse_move)
        self.canvas.mpl_connect("button_press_event", self.on_canvas_click)
        self.canvas.mpl_connect("scroll_event", self._on_scroll)

    def _build_shortcuts(self):
        QShortcut(QKeySequence("Ctrl+O"), self, activated=self.open_file_dialog)
        QShortcut(QKeySequence("H"), self, activated=self.open_histogram_dialog)
        QShortcut(QKeySequence("S"), self, activated=self.open_spectrum_dialog)
        QShortcut(QKeySequence("F"), self, activated=self.open_fft_dialog)
        QShortcut(QKeySequence("A"), self, activated=self.auto_scale_current_layer)
        QShortcut(QKeySequence("I"), self, activated=lambda: self.invert_checkbox.toggle())
        QShortcut(QKeySequence("Right"), self, activated=self._next_layer)
        QShortcut(QKeySequence("Left"), self, activated=self._prev_layer)
        QShortcut(QKeySequence("Ctrl+Z"), self, activated=self._undo_limits)
        QShortcut(QKeySequence("Ctrl+Y"), self, activated=self._redo_limits)

    # ------------------------------------------------------------------
    # Window management
    # ------------------------------------------------------------------

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
        self.auto_scale_button.setEnabled(enabled)
        self.histogram_button.setEnabled(enabled)
        self.spectrum_button.setEnabled(enabled)
        self.fft_button.setEnabled(enabled)
        self.layer_slider.setEnabled(enabled)

    def _clear_dataset_selector(self):
        self.available_datasets = []
        self.dataset_selector.blockSignals(True)
        self.dataset_selector.clear()
        self.dataset_selector.blockSignals(False)
        self.dataset_selector.setEnabled(False)
        self.dataset_selector.setVisible(False)
        self.dataset_label.setText("Channels:")
        self.dataset_label.setVisible(False)

    def _populate_dataset_selector(self, channels, selected_index):
        self.available_datasets = list(channels)
        self.dataset_selector.blockSignals(True)
        self.dataset_selector.clear()
        for label, *_ in channels:
            self.dataset_selector.addItem(label)
        if channels:
            self.dataset_selector.setCurrentRow(selected_index)
        self.dataset_selector.blockSignals(False)
        self.dataset_label.setText("Channels:")
        self.dataset_selector.setEnabled(bool(channels))
        self.dataset_selector.setVisible(bool(channels))
        self.dataset_label.setVisible(bool(channels))

    def _clear_file_selector(self):
        self.available_file_paths = []
        self.file_selector.blockSignals(True)
        self.file_selector.clear()
        self.file_selector.blockSignals(False)
        self.file_selector.setEnabled(False)
        self.file_selector.setVisible(False)
        self.file_label.setVisible(False)

    def _populate_file_selector(self, file_paths, selected_index):
        self.available_file_paths = list(file_paths)
        self.file_selector.blockSignals(True)
        self.file_selector.clear()
        for path in file_paths:
            self.file_selector.addItem(os.path.basename(path))
        self.file_selector.setCurrentRow(selected_index)
        self.file_selector.blockSignals(False)
        self.file_selector.setEnabled(bool(file_paths))
        self.file_selector.setVisible(bool(file_paths))
        self.file_label.setVisible(bool(file_paths))

    def _supported_paths_in_folder(self, folder_path):
        file_paths = []
        with os.scandir(folder_path) as entries:
            for entry in entries:
                if entry.is_file() and entry.name.lower().endswith(SUPPORTED_EXTENSIONS):
                    file_paths.append(entry.path)
        return sorted(file_paths, key=lambda path: os.path.basename(path).lower())

    def _single_channel_entries(self, file_path, data):
        channel_name = describe_supported_path(file_path, legacy_stm=self.legacy_stm)
        return [(channel_name, data, None, "Layer")]

    def _archive_channel_entries(self, file_path, archive_obj, showable):
        sweep_axis = None
        if isinstance(archive_obj, dict):
            try:
                candidate = np.asarray(archive_obj.get("sweep_signal"), dtype=np.float64)
            except (TypeError, ValueError):
                candidate = None
            if candidate is not None and candidate.ndim == 1 and candidate.size > 0 and np.isfinite(candidate).all():
                sweep_axis = candidate

        entries = []
        for name, data in showable:
            axis = None
            axis_label = "Layer"
            if sweep_axis is not None and getattr(data, "ndim", 0) == 3 and data.shape[2] == sweep_axis.size:
                axis = sweep_axis
                axis_label = "Sweep Signal"
            entries.append((f"{name} ({'x'.join(map(str, data.shape))})", data, axis, axis_label))
        return entries

    def _normalize_spectrum_axis(self, spectrum_axis, layer_count):
        return normalize_spectrum_axis(spectrum_axis, layer_count)

    def update_ui_from_state(self):
        if not self.state:
            self.layer_slider.blockSignals(True)
            self.layer_slider.setMinimum(0)
            self.layer_slider.setMaximum(0)
            self.layer_slider.setValue(0)
            self.layer_slider.blockSignals(False)
            self._update_layer_input_text()
            self.colormap_selector.blockSignals(True)
            self.colormap_selector.setCurrentText(self.default_colormap_name)
            self.colormap_selector.blockSignals(False)
            self.invert_checkbox.blockSignals(True)
            self.invert_checkbox.setChecked(self.default_inverted)
            self.invert_checkbox.blockSignals(False)
            self.origin_selector.blockSignals(True)
            self.origin_selector.setCurrentIndex(0 if self.default_origin == "lower" else 1)
            self.origin_selector.blockSignals(False)
            self.dims_label.setText("")
            self.set_controls_enabled(False)
            self.statusBar.showMessage(
                "Open or drop a file, or open a folder with supported files, to begin."
            )
            return
        self.layer_slider.setMinimum(0)
        self.layer_slider.setMaximum(self.state.layer_count - 1)
        self.layer_slider.setValue(self.state.current_layer)
        self.set_controls_enabled(True)
        shape = self.state.data.shape
        self.dims_label.setText(f"  {shape[1]}×{shape[0]}×{shape[2]}  ")
        self._update_layer_input_text()
        if self.state.layer_count <= 1:
            self.layer_slider.setEnabled(False)
        else:
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
            "Open or drop a file, or open a folder with supported files, to begin",
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
            self.colorbar_canvas.draw_idle()
            return

        cmap = resolve_colormap(self.state.colormap_name, self.state.inverted)
        vmin, vmax = self.state.visible_limits()
        if vmin == vmax:
            vmax = vmin + 1.0

        mappable = ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=cmap)
        colorbar = self.colorbar_figure.colorbar(mappable, cax=axes, orientation="horizontal")
        colorbar.set_ticks([])
        colorbar.ax.tick_params(top=False, bottom=False, labeltop=False, labelbottom=False, size=0)
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
        patterns = []
        for ext in SUPPORTED_EXTENSIONS:
            patterns.append(f"*{ext}")
            upper = ext.upper()
            if upper != ext:
                patterns.append(f"*{upper}")
        patterns = " ".join(dict.fromkeys(patterns))
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

        return 0, showable[0][1]

    def load_path(self, file_path, dataset_index=None, dataset_name=None, preserve_file_selector=False):
        if not self.is_supported_path(file_path):
            raise ValueError(
                f"Unsupported file extension: {os.path.splitext(file_path)[1].lower()}. "
                f"Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
            )
        self.current_source_dir = os.path.dirname(file_path)

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
            channels = self._archive_channel_entries(file_path, obj, showable)
            self.load_data(
                data,
                spectrum_axis=channels[selected_index][2],
                spectrum_axis_label=channels[selected_index][3],
            )
            self._populate_dataset_selector(
                channels,
                selected_index,
            )
        else:
            if dataset_index is not None or dataset_name is not None:
                raise ValueError(
                    "dataset_index and dataset_name are only supported for archive-style files"
                )
            data = load_data(file_path, legacy_stm=self.legacy_stm)
            channels = self._single_channel_entries(file_path, data)
            self.load_data(
                data,
                spectrum_axis=channels[0][2],
                spectrum_axis_label=channels[0][3],
            )
            self._populate_dataset_selector(channels, 0)
        if preserve_file_selector:
            try:
                selected_index = self.available_file_paths.index(file_path)
            except ValueError:
                selected_index = -1
            if selected_index >= 0 and self.file_selector.currentRow() != selected_index:
                self.file_selector.blockSignals(True)
                self.file_selector.setCurrentRow(selected_index)
                self.file_selector.blockSignals(False)
        else:
            self._populate_file_selector([file_path], 0)
        self.statusBar.showMessage(f"File Loaded: {file_path}", 5000)
        return True

    def open_file_dialog(self):
        start_dir = self.current_source_dir or ""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Data File", start_dir, self.supported_file_filter()
        )
        if not file_path: return
        
        try:
            self.load_path(file_path)
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load file:\n{str(e)}")

    def open_folder_dialog(self):
        start_dir = self.current_source_dir or ""
        folder_path = QFileDialog.getExistingDirectory(self, "Open Folder", start_dir)
        if not folder_path:
            return

        try:
            self.load_folder(folder_path)
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to open folder:\n{str(e)}")

    def load_folder(self, folder_path):
        file_paths = self._supported_paths_in_folder(folder_path)
        if not file_paths:
            QMessageBox.warning(
                self,
                "No Supported Files",
                "The selected folder does not contain supported data files.",
            )
            return False

        self.current_source_dir = folder_path
        self._populate_file_selector(file_paths, 0)
        return self.load_path(file_paths[0], preserve_file_selector=True)

    def on_file_switched(self, index):
        if not (0 <= index < len(self.available_file_paths)):
            return
        try:
            self.load_path(self.available_file_paths[index], preserve_file_selector=True)
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load file:\n{str(e)}")

    def on_dataset_switched(self, index):
        if 0 <= index < len(self.available_datasets):
            _, data, spectrum_axis, spectrum_axis_label = self.available_datasets[index]
            self.load_data(
                data,
                spectrum_axis=spectrum_axis,
                spectrum_axis_label=spectrum_axis_label,
            )

    def load_data(self, data, spectrum_axis=None, spectrum_axis_label="Layer"):
        old_cmap = self.state.colormap_name if self.state else self.colormap_selector.currentText()
        old_inv = self.state.inverted if self.state else self.invert_checkbox.isChecked()
        old_mode = self.state.color_limit_mode if self.state else self.default_color_limit_mode
        old_origin = self.state.origin if self.state else self.origin_selector.currentData()
        array = to_numpy(data)

        self.state = ViewerState(
            array,
            color_limit_mode=old_mode,
            colormap_name=old_cmap,
            inverted=old_inv,
            origin=old_origin,
        )
        self.default_colormap_name = old_cmap
        self.default_inverted = old_inv
        self.default_color_limit_mode = old_mode
        self.default_origin = old_origin
        self.spectrum_axis = self._normalize_spectrum_axis(spectrum_axis, self.state.layer_count)
        self.spectrum_axis_label = spectrum_axis_label if self.spectrum_axis is not None else "Layer"

        self.spectrum_points = [
            (x, y) for x, y in self.spectrum_points
            if self._is_valid_spectrum_point(x, y)
        ]

        # Force a brand new image artist by clearing axes
        self.axes.clear()
        self.image_artist = None
        self.spectrum_markers = []
        self._limits_history.clear()
        self._limits_redo.clear()

        self.update_ui_from_state()
        self.refresh_image()

        if getattr(self, "histogram_dialog", None):
            self.histogram_dialog.state = self.state
            self.histogram_dialog.refresh_from_state()
        if getattr(self, "fft_dialog", None):
            self.fft_dialog.state = self.state
        if getattr(self, "spectrum_dialog", None):
            self._sync_spectrum_dialog()

    def refresh_image(self):
        if not self.state:
            self.image_artist = None
            self.spectrum_markers = []
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

        self._refresh_spectrum_marker()

        self.canvas.draw_idle()
        self.refresh_colorbar()

    def on_layer_changed(self, index: int):
        if not self.state: return
        self.state.set_current_layer(index)
        self._update_layer_input_text()
        self.refresh_image()
        self.refresh_aux_windows()

    def _next_layer(self):
        if self.state and self.state.current_layer < self.state.layer_count - 1:
            self.layer_slider.setValue(self.state.current_layer + 1)

    def _prev_layer(self):
        if self.state and self.state.current_layer > 0:
            self.layer_slider.setValue(self.state.current_layer - 1)

    def _on_scroll(self, event):
        if not self.state or event.inaxes != self.axes:
            return
        if event.button == "up":
            self._next_layer()
        elif event.button == "down":
            self._prev_layer()

    def _on_layer_input(self):
        if not self.state:
            return
        text = self.layer_input.text().split("/")[0].strip()
        try:
            num = int(text)
        except ValueError:
            self._update_layer_input_text()
            return
        index = max(0, min(num - 1, self.state.layer_count - 1))
        self.layer_slider.setValue(index)

    def _update_layer_input_text(self):
        if not self.state:
            self.layer_input.setText("—")
            self.nav_title.setText("NAVIGATION")
        elif self.state.layer_count <= 1:
            self.layer_input.setText("1 / 1")
            self.nav_title.setText("NAVIGATION (Layer 1)")
        else:
            self.layer_input.setText(f"{self.state.current_layer + 1} / {self.state.layer_count}")
            self.nav_title.setText(f"NAVIGATION (Layer {self.state.current_layer + 1})")

    def auto_scale_current_layer(self):
        if not self.state:
            return

        layer = self.state.data[:, :, self.state.current_layer]
        pct = self.auto_scale_mode.currentData()
        if pct is None or pct >= 100:
            vmin, vmax = float(layer.min()), float(layer.max())
        else:
            lo = (100 - pct) / 2
            hi = 100 - lo
            vmin, vmax = float(np.percentile(layer, lo)), float(np.percentile(layer, hi))
        if vmin == vmax:
            vmax = vmin + 1.0
        self._push_limits_history()
        self.state.set_limits(vmin, vmax)
        self.refresh_image()
        self.refresh_aux_windows()

    def _push_limits_history(self):
        if not self.state:
            return
        self._limits_history.append(self.state.visible_limits())
        self._limits_redo.clear()
        if len(self._limits_history) > 50:
            self._limits_history.pop(0)

    def _undo_limits(self):
        if not self.state or not self._limits_history:
            return
        self._limits_redo.append(self.state.visible_limits())
        vmin, vmax = self._limits_history.pop()
        self.state.set_limits(vmin, vmax)
        self.refresh_image()
        self.refresh_aux_windows()

    def _redo_limits(self):
        if not self.state or not self._limits_redo:
            return
        self._limits_history.append(self.state.visible_limits())
        vmin, vmax = self._limits_redo.pop()
        self.state.set_limits(vmin, vmax)
        self.refresh_image()
        self.refresh_aux_windows()

    def on_mouse_move(self, event):
        if self.state and event.inaxes == self.axes:
            x, y = round(event.xdata), round(event.ydata)
            if 0 <= x < self.state.data.shape[1] and 0 <= y < self.state.data.shape[0]:
                val = self.state.data[y, x, self.state.current_layer]
                self.cursor_label.setText(f"  ({x}, {y}) | Value: {val:.6g}")
                return
        self.cursor_label.setText("")

    def on_canvas_click(self, event):
        if not self.state or event.inaxes != self.axes:
            return
        if getattr(self.nav_toolbar, "mode", ""):
            return
        if event.xdata is None or event.ydata is None:
            return

        x = round(event.xdata)
        y = round(event.ydata)
        if not self._is_valid_spectrum_point(x, y):
            return

        if getattr(self, "spectrum_dialog", None) is not None and self.spectrum_dialog.isVisible():
            if (x, y) in self.spectrum_points:
                return
            self.spectrum_points.append((x, y))
            self.spectrum_dialog.add_point(x, y)
            self.refresh_image()

    def _is_valid_spectrum_point(self, x, y):
        if self.state is None:
            return False
        return 0 <= x < self.state.data.shape[1] and 0 <= y < self.state.data.shape[0]

    _MARKER_COLORS = [
        "#ff2d55", "#007aff", "#34c759", "#ff9500", "#af52de",
        "#ff3b30", "#5ac8fa", "#ffcc00", "#ff6482", "#30b0c7",
    ]

    def _refresh_spectrum_marker(self):
        for m in self.spectrum_markers:
            try:
                m.remove()
            except (ValueError, TypeError):
                pass
        self.spectrum_markers = []

        for i, (x, y) in enumerate(self.spectrum_points):
            if not self._is_valid_spectrum_point(x, y):
                continue
            color = self._MARKER_COLORS[i % len(self._MARKER_COLORS)]
            (marker,) = self.axes.plot(
                [x],
                [y],
                marker="+",
                linestyle="None",
                color=color,
                markersize=18,
                markeredgewidth=3,
                zorder=5,
            )
            marker.set_path_effects(
                [patheffects.Stroke(linewidth=5, foreground="#000000"), patheffects.Normal()]
            )
            self.spectrum_markers.append(marker)

    def _sync_spectrum_dialog(self):
        if getattr(self, "spectrum_dialog", None) is None:
            return
        self.spectrum_dialog.set_state(
            self.state,
            spectrum_axis=self.spectrum_axis,
            spectrum_axis_label=self.spectrum_axis_label,
            points=list(self.spectrum_points),
        )

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
        self._push_limits_history()
        self.refresh_image()

    def open_spectrum_dialog(self):
        if not self.state:
            return
        if getattr(self, "spectrum_dialog", None) is None:
            self.spectrum_dialog = SpectrumDialog(
                self.state,
                on_selection_cleared=self._on_spectrum_selection_cleared,
                on_layer_changed=self._on_spectrum_layer_changed,
                on_point_removed=self._on_spectrum_point_removed,
                parent=self,
            )
        self._sync_spectrum_dialog()
        self.spectrum_dialog.show()
        self.spectrum_dialog.raise_()
        self.spectrum_dialog.activateWindow()

    def _on_spectrum_layer_changed(self, index):
        """Called when the user drags the layer line in the spectrum dialog."""
        self.layer_slider.setValue(index)

    def _on_spectrum_point_removed(self, remaining_points):
        self.spectrum_points = list(remaining_points)
        self.refresh_image()

    def _on_spectrum_selection_cleared(self):
        self.spectrum_points = []
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
        if getattr(self, "spectrum_dialog", None) is not None:
            self._sync_spectrum_dialog()
