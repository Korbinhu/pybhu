from __future__ import annotations

import os

import matplotlib
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PyQt6 import QtCore, QtGui
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QDragEnterEvent, QDropEvent
from PyQt6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from .loader import SUPPORTED_EXTENSIONS, SpectrumDataset, ensure_spectrum_dataset, load_dataset_candidates
from .state import DatasetViewerState

# ---------------------------------------------------------------------------
# Palette — all colour literals live here so a theme change is a one-stop edit
# ---------------------------------------------------------------------------
_CLR_BACKGROUND   = "#f3ede4"   # main window background
_CLR_SURFACE      = "#fffaf2"   # card / group surface
_CLR_SURFACE_ALT  = "#fffdf8"   # input / plot face
_CLR_PANEL        = "#efe5d4"   # sidebar panel
_CLR_BORDER       = "#cbb89c"   # primary border
_CLR_BORDER_LIGHT = "#d8c9b2"   # secondary / lighter border
_CLR_BORDER_INPUT = "#cfbda3"   # input field border
_CLR_TEXT         = "#1f2937"   # primary text
_CLR_TEXT_MID     = "#475569"   # secondary text
_CLR_TEXT_SUBTLE  = "#64748b"   # hint / placeholder text
_CLR_TEXT_DARK    = "#334155"   # dark text variant
_CLR_HEADING      = "#1f3a5f"   # hero title / chart title
_CLR_CAPTION      = "#786c5d"   # subtle captions
_CLR_BUTTON_BG    = "#f4ebdd"   # button background
_CLR_BUTTON_TEXT  = "#3f3428"   # button text
_CLR_BUTTON_HOV   = "#ead9bf"   # button hover
_CLR_PRIMARY      = "#c26a2d"   # primary action / line colour
_CLR_PRIMARY_DARK = "#a75922"   # primary border
_CLR_PRIMARY_TEXT = "#fff7ed"   # text on primary buttons
_CLR_ACCENT_BG    = "#dbe8f8"   # checked toolbar button fill
_CLR_ACCENT_BDR   = "#8db2da"   # checked toolbar button border
_CLR_HANDLE       = "#1f3a5f"   # slider handle
_CLR_GRID         = "#d4c1a5"   # axes grid lines
_CLR_TICK         = "#5b6470"   # axes tick labels / marks
_CLR_EMPTY_TEXT   = "#5f6673"   # empty-state placeholder text


def _group_frame(title: str) -> tuple[QFrame, QVBoxLayout]:
    frame = QFrame()
    frame.setProperty("class", "ControlGroup")
    frame.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(14, 14, 14, 14)
    layout.setSpacing(8)

    label = QLabel(title)
    label.setProperty("class", "GroupTitle")
    layout.addWidget(label)
    return frame, layout


def _format_float(value: float) -> str:
    return f"{float(value):.6g}"


def _parameter_decimals(lo: float, hi: float) -> int:
    span = abs(float(hi) - float(lo))
    if span >= 1000:
        return 0
    if span >= 100:
        return 1
    if span >= 10:
        return 2
    if span >= 1:
        return 3
    return 4


def _matplotlib_dialog_palette() -> QtGui.QPalette:
    palette = QtGui.QPalette()
    palette.setColor(QtGui.QPalette.ColorRole.Window,          QtGui.QColor(_CLR_SURFACE))
    palette.setColor(QtGui.QPalette.ColorRole.Base,            QtGui.QColor(_CLR_SURFACE_ALT))
    palette.setColor(QtGui.QPalette.ColorRole.AlternateBase,   QtGui.QColor(_CLR_SURFACE))
    palette.setColor(QtGui.QPalette.ColorRole.Text,            QtGui.QColor(_CLR_TEXT))
    palette.setColor(QtGui.QPalette.ColorRole.WindowText,      QtGui.QColor(_CLR_TEXT))
    palette.setColor(QtGui.QPalette.ColorRole.Button,          QtGui.QColor(_CLR_BUTTON_BG))
    palette.setColor(QtGui.QPalette.ColorRole.ButtonText,      QtGui.QColor(_CLR_BUTTON_TEXT))
    palette.setColor(QtGui.QPalette.ColorRole.Highlight,       QtGui.QColor(_CLR_ACCENT_BG))
    palette.setColor(QtGui.QPalette.ColorRole.HighlightedText, QtGui.QColor(_CLR_HEADING))
    palette.setColor(QtGui.QPalette.ColorRole.ToolTipBase,     QtGui.QColor(_CLR_SURFACE_ALT))
    palette.setColor(QtGui.QPalette.ColorRole.ToolTipText,     QtGui.QColor(_CLR_TEXT))
    return palette


def _style_matplotlib_form_dialog(dialog: QWidget) -> None:
    dialog.setPalette(_matplotlib_dialog_palette())
    dialog.setStyleSheet(
        f"""
        QDialog {{
            background-color: {_CLR_SURFACE};
            color: {_CLR_TEXT};
        }}
        QWidget {{
            color: {_CLR_TEXT};
        }}
        QLabel {{
            color: {_CLR_TEXT};
            background: transparent;
        }}
        QTabWidget::pane {{
            background-color: {_CLR_SURFACE};
            border: 1px solid {_CLR_BORDER};
            border-radius: 12px;
            top: -1px;
        }}
        QTabBar::tab {{
            background-color: {_CLR_BUTTON_BG};
            color: {_CLR_BUTTON_TEXT};
            border: 1px solid {_CLR_BORDER_INPUT};
            border-bottom: none;
            border-top-left-radius: 8px;
            border-top-right-radius: 8px;
            padding: 6px 14px;
            margin-right: 4px;
        }}
        QTabBar::tab:selected {{
            background-color: {_CLR_SURFACE};
            color: {_CLR_TEXT};
        }}
        QTabBar::tab:!selected {{
            margin-top: 3px;
        }}
        QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QDateEdit, QDateTimeEdit, QFontComboBox {{
            background-color: {_CLR_SURFACE_ALT};
            color: {_CLR_TEXT};
            border: 1px solid {_CLR_BORDER_INPUT};
            border-radius: 8px;
            padding: 4px 6px;
            min-height: 24px;
            selection-background-color: {_CLR_ACCENT_BG};
            selection-color: {_CLR_HEADING};
        }}
        QComboBox QAbstractItemView {{
            background-color: {_CLR_SURFACE_ALT};
            color: {_CLR_TEXT};
            selection-background-color: {_CLR_ACCENT_BG};
            selection-color: {_CLR_HEADING};
        }}
        QCheckBox {{
            color: {_CLR_TEXT};
        }}
        QPushButton {{
            background-color: {_CLR_BUTTON_BG};
            color: {_CLR_BUTTON_TEXT};
            border: 1px solid {_CLR_BORDER_INPUT};
            border-radius: 8px;
            padding: 6px 12px;
        }}
        QPushButton:hover {{
            background-color: {_CLR_BUTTON_HOV};
        }}
        """
    )
    # Qt propagates palette changes to child widgets automatically; no need to
    # walk findChildren manually.


class ViewerNavigationToolbar(NavigationToolbar2QT):
    ICON_COLOR = QtGui.QColor(_CLR_BUTTON_TEXT)

    def _icon(self, name):
        # matplotlib.get_data_path() is the public equivalent of cbook._get_data_path()
        data_dir = matplotlib.get_data_path()
        import pathlib
        path_regular = pathlib.Path(data_dir) / "images" / name
        path_large = path_regular.with_name(path_regular.name.replace(".png", "_large.png"))
        filename = str(path_large if path_large.exists() else path_regular)

        pm = QtGui.QPixmap(filename)
        pm.setDevicePixelRatio(self.devicePixelRatioF() or 1)
        mask = pm.createMaskFromColor(
            QtGui.QColor("black"),
            QtCore.Qt.MaskMode.MaskOutColor,
        )
        pm.fill(self.ICON_COLOR)
        pm.setMask(mask)
        return QtGui.QIcon(pm)

    def edit_parameters(self):
        super().edit_parameters()
        # _fedit_dialog is an undocumented internal attribute set by NavigationToolbar2QT
        # after it opens the "Edit axis parameters" dialog.  We use it here only to apply
        # our colour theme; if a future matplotlib version renames it the dialog will
        # simply not be themed (non-fatal).
        dialog = getattr(self, "_fedit_dialog", None)
        if dialog is not None:
            _style_matplotlib_form_dialog(dialog)


class DatasetSelectionDialog(QDialog):
    def __init__(self, items: list[tuple[str, SpectrumDataset]], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Dataset")
        self.resize(480, 360)
        self.selected_index = -1

        layout = QVBoxLayout(self)
        intro = QLabel("Multiple dataset candidates were detected. Choose one to load:")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        self.list_widget = QListWidget()
        for name, dataset in items:
            details = f"{name} ({dataset.sample_count} spectra x {dataset.point_count} points)"
            if dataset.parameter_count:
                details += f" [{dataset.parameter_count} params]"
            self.list_widget.addItem(details)
        self.list_widget.setCurrentRow(0)
        layout.addWidget(self.list_widget)

        buttons = QHBoxLayout()
        buttons.addStretch()
        load_button = QPushButton("Load Selected")
        load_button.clicked.connect(self.accept)
        buttons.addWidget(load_button)
        layout.addLayout(buttons)

    def accept(self) -> None:
        row = self.list_widget.currentRow()
        if row >= 0:
            self.selected_index = row
            super().accept()


class ParameterControl(QWidget):
    valueChanged = pyqtSignal(float)

    def __init__(self, name: str, lo: float, hi: float, value: float, parent=None):
        super().__init__(parent)
        self.setObjectName("ParameterControl")
        self.lo = float(lo)
        self.hi = float(hi)
        self._updating = False
        self.slider_steps = 1000

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        header = QHBoxLayout()
        self.name_label = QLabel(name)
        self.name_label.setStyleSheet("font-weight: 600; color: #1f2937;")
        header.addWidget(self.name_label)
        header.addStretch()

        self.spin = QDoubleSpinBox()
        self.spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.spin.setDecimals(_parameter_decimals(self.lo, self.hi))
        self.spin.setRange(self.lo, self.hi)
        self.spin.setSingleStep(max((self.hi - self.lo) / 200.0, 1e-6))
        self.spin.setValue(float(value))
        self.spin.valueChanged.connect(self._on_spin_changed)
        header.addWidget(self.spin)
        layout.addLayout(header)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, self.slider_steps)
        self.slider.setValue(self._value_to_slider(value))
        self.slider.setEnabled(self.hi > self.lo)
        self.slider.valueChanged.connect(self._on_slider_changed)
        layout.addWidget(self.slider)

        self.range_label = QLabel(f"{_format_float(self.lo)} to {_format_float(self.hi)}")
        self.range_label.setStyleSheet("color: #475569; font-size: 11px;")
        layout.addWidget(self.range_label)

    def _value_to_slider(self, value: float) -> int:
        if self.hi <= self.lo:
            return 0
        ratio = (float(value) - self.lo) / (self.hi - self.lo)
        return int(round(np.clip(ratio, 0.0, 1.0) * self.slider_steps))

    def _slider_to_value(self, slider_value: int) -> float:
        if self.hi <= self.lo:
            return self.lo
        ratio = slider_value / self.slider_steps
        return self.lo + ratio * (self.hi - self.lo)

    def _on_spin_changed(self, value: float) -> None:
        if self._updating:
            return
        self._updating = True
        self.slider.setValue(self._value_to_slider(value))
        self._updating = False
        self.valueChanged.emit(float(value))

    def _on_slider_changed(self, slider_value: int) -> None:
        if self._updating:
            return
        self._updating = True
        self.spin.setValue(self._slider_to_value(slider_value))
        self._updating = False
        self.valueChanged.emit(float(self.spin.value()))

    def value(self) -> float:
        return float(self.spin.value())

    def set_value(self, value: float) -> None:
        self._updating = True
        self.spin.setValue(float(value))
        self.slider.setValue(self._value_to_slider(value))
        self._updating = False


class SpectrumDatasetViewer(QMainWindow):
    def __init__(self, data=None, **options):
        self.app = QApplication.instance()
        self.owns_app = self.app is None
        if self.app is None:
            self.app = QApplication([])

        super().__init__()
        self.setWindowTitle(options.pop("window_title", "PyBHU Dataset Viewer"))
        self.resize(1280, 860)
        self.setAcceptDrops(True)
        self._is_centered = False

        self.state: DatasetViewerState | None = None
        self.available_datasets: list[tuple[str, SpectrumDataset]] = []
        self.line_artist = None
        self._parameter_controls: list[ParameterControl] = []
        self._syncing_parameters = False

        self.setStyleSheet(
            f"""
            QMainWindow, QDialog {{
                background-color: {_CLR_BACKGROUND};
                color: {_CLR_TEXT};
                font-family: 'Aptos', 'Segoe UI', sans-serif;
            }}
            .ControlPanel {{
                background-color: {_CLR_PANEL};
                border-left: 1px solid {_CLR_BORDER};
            }}
            .ControlGroup {{
                background-color: {_CLR_SURFACE};
                border: 2px solid {_CLR_BORDER};
                border-radius: 16px;
            }}
            .GroupTitle {{
                font-weight: 700;
                color: {_CLR_CAPTION};
                font-size: 11px;
                letter-spacing: 0.08em;
                padding-bottom: 4px;
                border-bottom: none;
            }}
            QLabel {{
                color: {_CLR_TEXT};
            }}
            QPushButton {{
                background-color: {_CLR_BUTTON_BG};
                border: 1px solid {_CLR_BORDER_INPUT};
                border-radius: 8px;
                padding: 7px 10px;
                min-height: 26px;
                color: {_CLR_BUTTON_TEXT};
            }}
            QPushButton:hover {{
                background-color: {_CLR_BUTTON_HOV};
            }}
            QPushButton#PrimaryButton {{
                background-color: {_CLR_PRIMARY};
                color: {_CLR_PRIMARY_TEXT};
                border: 1px solid {_CLR_PRIMARY_DARK};
            }}
            QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox {{
                background-color: {_CLR_SURFACE_ALT};
                border: 1px solid {_CLR_BORDER_INPUT};
                border-radius: 8px;
                padding: 4px 6px;
                min-height: 24px;
                color: {_CLR_TEXT};
                selection-background-color: {_CLR_ACCENT_BG};
                selection-color: {_CLR_HEADING};
            }}
            QStatusBar {{
                border-top: 1px solid {_CLR_BORDER_LIGHT};
                background-color: {_CLR_PANEL};
            }}
            QScrollArea {{
                background-color: {_CLR_SURFACE};
                border: 1px solid {_CLR_BORDER_LIGHT};
                border-radius: 10px;
            }}
            QScrollArea > QWidget > QWidget {{
                background-color: {_CLR_SURFACE};
                color: {_CLR_TEXT};
            }}
            QWidget#ParameterContainer {{
                background-color: {_CLR_SURFACE};
                color: {_CLR_TEXT};
            }}
            QWidget#ParameterControl {{
                background-color: {_CLR_SURFACE};
                color: {_CLR_TEXT};
            }}
            QSlider::groove:horizontal {{
                border: 1px solid {_CLR_BORDER_LIGHT};
                height: 8px;
                background: {_CLR_PANEL};
                border-radius: 4px;
            }}
            QSlider::handle:horizontal {{
                background: {_CLR_HANDLE};
                border: 1px solid {_CLR_HEADING};
                width: 18px;
                margin: -6px 0;
                border-radius: 9px;
            }}
            QFrame#HeroFrame {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #fff9ef, stop:0.55 #f1e1c6, stop:1 #ead2a9);
                border: 2px solid {_CLR_BORDER};
                border-radius: 18px;
            }}
            QFrame#TransportFrame {{
                background-color: {_CLR_SURFACE};
                border: 2px solid {_CLR_BORDER};
                border-radius: 16px;
            }}
            QLabel#HeroTitle {{
                color: {_CLR_HEADING};
                font-size: 22px;
                font-weight: 700;
            }}
            QLabel#HeroMeta {{
                color: {_CLR_TEXT_MID};
                font-size: 13px;
            }}
            QLabel#HeroSubtle {{
                color: {_CLR_CAPTION};
                font-size: 11px;
            }}
            QLabel#DataBadge {{
                background-color: rgba(31, 58, 95, 0.12);
                color: {_CLR_HEADING};
                border: 1px solid rgba(31, 58, 95, 0.2);
                border-radius: 12px;
                padding: 6px 10px;
                font-weight: 700;
            }}
            QLabel#SecondaryBadge {{
                background-color: rgba(194, 106, 45, 0.12);
                color: #9a531f;
                border: 1px solid rgba(194, 106, 45, 0.2);
                border-radius: 12px;
                padding: 6px 10px;
                font-weight: 600;
            }}
            QLabel#NavigatorValue {{
                color: {_CLR_HEADING};
                font-size: 15px;
                font-weight: 700;
            }}
            """
        )

        self._build_ui()
        self.refresh_from_state(sync_parameters=True)

        if data is not None:
            self.load_data(ensure_spectrum_dataset(data))

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if not self._is_centered:
            self.center_on_screen()
            self._is_centered = True

    def center_on_screen(self) -> None:
        screen = self.screen() or self.app.primaryScreen()
        if screen is None:
            return
        available = screen.availableGeometry()
        frame = self.frameGeometry()
        frame.moveCenter(available.center())
        self.move(frame.topLeft())

    def _build_ui(self) -> None:
        splitter = QSplitter(Qt.Orientation.Horizontal)

        plot_column = QWidget()
        plot_layout = QVBoxLayout(plot_column)
        plot_layout.setContentsMargins(10, 10, 10, 10)
        plot_layout.setSpacing(10)
        plot_layout.addWidget(self._build_hero_frame())
        plot_layout.addWidget(self._build_plot_frame(), 1)
        plot_layout.addWidget(self._build_transport_frame())

        splitter.addWidget(plot_column)
        splitter.addWidget(self._build_sidebar())
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        self.setCentralWidget(splitter)

        self.status_bar = QStatusBar(self)
        self.setStatusBar(self.status_bar)
        self.canvas.mpl_connect("motion_notify_event", self.on_mouse_move)

    def _build_hero_frame(self) -> QFrame:
        hero_frame = QFrame()
        hero_frame.setObjectName("HeroFrame")
        hero_frame.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        hero_layout = QHBoxLayout(hero_frame)
        hero_layout.setContentsMargins(18, 16, 18, 16)
        hero_layout.setSpacing(14)

        hero_text = QVBoxLayout()
        hero_text.setSpacing(4)
        self.hero_name_label = QLabel("Dataset Viewer")
        self.hero_name_label.setObjectName("HeroTitle")
        hero_text.addWidget(self.hero_name_label)
        self.hero_meta_label = QLabel("Open a torch or numpy-style dataset file to begin.")
        self.hero_meta_label.setObjectName("HeroMeta")
        self.hero_meta_label.setWordWrap(True)
        hero_text.addWidget(self.hero_meta_label)
        self.hero_path_label = QLabel("")
        self.hero_path_label.setObjectName("HeroSubtle")
        self.hero_path_label.setWordWrap(True)
        hero_text.addWidget(self.hero_path_label)
        hero_layout.addLayout(hero_text, 1)

        badge_column = QVBoxLayout()
        badge_column.setSpacing(8)
        badge_column.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignRight)
        self.backend_badge = QLabel("TORCH")
        self.backend_badge.setObjectName("DataBadge")
        badge_column.addWidget(self.backend_badge, 0, Qt.AlignmentFlag.AlignRight)
        self.visible_badge = QLabel("0 spectra")
        self.visible_badge.setObjectName("SecondaryBadge")
        badge_column.addWidget(self.visible_badge, 0, Qt.AlignmentFlag.AlignRight)
        hero_layout.addLayout(badge_column)

        return hero_frame

    def _build_plot_frame(self) -> QFrame:
        plot_frame = QFrame()
        plot_frame.setProperty("class", "ControlGroup")
        plot_frame.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        plot_frame_layout = QVBoxLayout(plot_frame)
        plot_frame_layout.setContentsMargins(10, 10, 10, 10)
        plot_frame_layout.setSpacing(6)

        self.figure = Figure(facecolor=_CLR_SURFACE_ALT, tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.axes = self.figure.add_subplot(111)
        self._configure_axes()

        self.nav_toolbar = ViewerNavigationToolbar(self.canvas, self)
        self.nav_toolbar.setStyleSheet(
            f"""
            QToolBar {{
                background-color: {_CLR_SURFACE};
                border: 1px solid {_CLR_BORDER_LIGHT};
                border-radius: 12px;
                spacing: 4px;
                padding: 6px;
            }}
            QToolButton {{
                background-color: transparent;
                border: 1px solid transparent;
                border-radius: 8px;
                padding: 6px;
            }}
            QToolButton:hover {{
                background-color: {_CLR_BUTTON_HOV};
                border: 1px solid {_CLR_BORDER_LIGHT};
            }}
            QToolButton:checked {{
                background-color: {_CLR_ACCENT_BG};
                border: 1px solid {_CLR_ACCENT_BDR};
            }}
            """
        )
        plot_frame_layout.addWidget(self.nav_toolbar)
        plot_frame_layout.addWidget(self.canvas)

        return plot_frame

    def _configure_axes(self) -> None:
        """Apply static axes styling once after creation or after axes.clear()."""
        self.axes.set_facecolor(_CLR_SURFACE_ALT)
        self.axes.grid(True, linewidth=0.35, alpha=0.55, color=_CLR_GRID)
        self.axes.spines["top"].set_visible(True)
        self.axes.spines["right"].set_visible(True)
        for spine in self.axes.spines.values():
            spine.set_color(_CLR_BORDER)
        self.axes.tick_params(
            colors=_CLR_TICK,
            top=False,
            right=False,
            labeltop=False,
            labelright=False,
            direction="out",
        )

    def _build_transport_frame(self) -> QFrame:
        transport_frame = QFrame()
        transport_frame.setObjectName("TransportFrame")
        transport_frame.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        transport_layout = QGridLayout(transport_frame)
        transport_layout.setContentsMargins(14, 12, 14, 12)
        transport_layout.setHorizontalSpacing(12)
        transport_layout.setVerticalSpacing(8)

        transport_layout.addWidget(QLabel("Visible Split"), 0, 0)
        self.split_selector = QComboBox()
        self.split_selector.currentIndexChanged.connect(self.on_split_changed)
        transport_layout.addWidget(self.split_selector, 0, 1)

        self.sample_label = QLabel("0 / 0")
        self.sample_label.setObjectName("NavigatorValue")
        self.sample_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        transport_layout.addWidget(self.sample_label, 0, 2)

        self.prev_button = QPushButton("Prev")
        self.prev_button.clicked.connect(self.on_prev)
        transport_layout.addWidget(self.prev_button, 0, 3)

        self.random_button = QPushButton("Random")
        self.random_button.clicked.connect(self.on_random)
        transport_layout.addWidget(self.random_button, 0, 4)

        self.next_button = QPushButton("Next")
        self.next_button.clicked.connect(self.on_next)
        transport_layout.addWidget(self.next_button, 0, 5)

        transport_layout.addWidget(QLabel("Visible Position"), 1, 0)
        self.sample_slider = QSlider(Qt.Orientation.Horizontal)
        self.sample_slider.valueChanged.connect(self.on_sample_slider_changed)
        transport_layout.addWidget(self.sample_slider, 1, 1, 1, 5)

        transport_layout.addWidget(QLabel("Global Index"), 2, 0)
        self.index_spin = QSpinBox()
        self.index_spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.index_spin.valueChanged.connect(self.on_index_changed)
        transport_layout.addWidget(self.index_spin, 2, 1)

        transport_layout.addWidget(QLabel("Y Minimum"), 2, 2)
        self.ymin_edit = QLineEdit()
        self.ymin_edit.setPlaceholderText("auto")
        self.ymin_edit.returnPressed.connect(self.apply_ymin)
        transport_layout.addWidget(self.ymin_edit, 2, 3)

        self.apply_ymin_button = QPushButton("Apply")
        self.apply_ymin_button.clicked.connect(self.apply_ymin)
        transport_layout.addWidget(self.apply_ymin_button, 2, 4)

        self.auto_ymin_button = QPushButton("Auto")
        self.auto_ymin_button.clicked.connect(self.reset_ymin)
        transport_layout.addWidget(self.auto_ymin_button, 2, 5)

        return transport_frame

    def _build_sidebar(self) -> QWidget:
        sidebar = QWidget()
        sidebar.setProperty("class", "ControlPanel")
        sidebar.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        sidebar.setFixedWidth(400)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(14, 14, 14, 14)
        sidebar_layout.setSpacing(14)

        data_group, data_layout = _group_frame("DATASET")
        self.load_button = QPushButton("Open Dataset")
        self.load_button.setObjectName("PrimaryButton")
        self.load_button.clicked.connect(self.open_file_dialog)
        data_layout.addWidget(self.load_button)

        self.dataset_label = QLabel("Candidate Dataset")
        self.dataset_label.setVisible(False)
        data_layout.addWidget(self.dataset_label)

        self.dataset_selector = QComboBox()
        self.dataset_selector.currentIndexChanged.connect(self.on_dataset_switched)
        self.dataset_selector.setVisible(False)
        self.dataset_selector.setEnabled(False)
        data_layout.addWidget(self.dataset_selector)

        self.summary_label = QLabel()
        self.summary_label.setWordWrap(True)
        self.summary_label.setStyleSheet(f"color: {_CLR_TEXT_DARK};")
        data_layout.addWidget(self.summary_label)
        sidebar_layout.addWidget(data_group)

        info_group, info_layout = _group_frame("CURRENT SAMPLE")
        self.current_info_label = QLabel()
        self.current_info_label.setWordWrap(True)
        self.current_info_label.setStyleSheet(f"color: {_CLR_TEXT_DARK};")
        info_layout.addWidget(self.current_info_label)
        sidebar_layout.addWidget(info_group)

        param_group, param_layout = _group_frame("PARAMETER MATCH")
        self.parameter_hint = QLabel("No parameter channels detected in the current dataset.")
        self.parameter_hint.setWordWrap(True)
        self.parameter_hint.setStyleSheet(f"color: {_CLR_TEXT_SUBTLE};")
        param_layout.addWidget(self.parameter_hint)

        self.parameter_scroll = QScrollArea()
        self.parameter_scroll.setWidgetResizable(True)
        self.parameter_scroll.setVisible(False)
        self.parameter_scroll.setStyleSheet(
            f"QScrollArea {{ background-color: {_CLR_SURFACE}; border: 1px solid {_CLR_BORDER_LIGHT}; border-radius: 10px; }}"
        )
        self.parameter_scroll.viewport().setStyleSheet(f"background-color: {_CLR_SURFACE};")
        self.parameter_container = QWidget()
        self.parameter_container.setObjectName("ParameterContainer")
        self.parameter_container.setStyleSheet(f"background-color: {_CLR_SURFACE}; color: {_CLR_TEXT};")
        self.parameter_layout = QVBoxLayout(self.parameter_container)
        self.parameter_layout.setContentsMargins(0, 0, 0, 0)
        self.parameter_layout.setSpacing(10)
        self.parameter_layout.addStretch()
        self.parameter_scroll.setWidget(self.parameter_container)
        param_layout.addWidget(self.parameter_scroll)

        self.match_label = QLabel()
        self.match_label.setWordWrap(True)
        self.match_label.setStyleSheet(f"color: {_CLR_TEXT_DARK};")
        param_layout.addWidget(self.match_label)
        sidebar_layout.addWidget(param_group)

        sidebar_layout.addStretch()

        return sidebar

    def supported_file_filter(self) -> str:
        patterns = " ".join(f"*{ext}" for ext in SUPPORTED_EXTENSIONS)
        return f"Dataset Files ({patterns});;All Files (*)"

    def is_supported_path(self, path: str) -> bool:
        return os.path.isfile(path) and path.lower().endswith(SUPPORTED_EXTENSIONS)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if self._first_supported_url(event):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:
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
            QMessageBox.critical(self, "Load Error", f"Failed to load file:\n{exc}")
            event.ignore()

    def _first_supported_url(self, event) -> str | None:
        mime_data = event.mimeData()
        if not mime_data.hasUrls():
            return None
        for url in mime_data.urls():
            if not url.isLocalFile():
                continue
            file_path = url.toLocalFile()
            if self.is_supported_path(file_path):
                return file_path
        return None

    def _clear_dataset_selector(self) -> None:
        self.available_datasets = []
        self.dataset_selector.blockSignals(True)
        self.dataset_selector.clear()
        self.dataset_selector.blockSignals(False)
        self.dataset_selector.setVisible(False)
        self.dataset_selector.setEnabled(False)
        self.dataset_label.setVisible(False)

    def _populate_dataset_selector(self, items: list[tuple[str, SpectrumDataset]], selected_index: int) -> None:
        self.available_datasets = list(items)
        self.dataset_selector.blockSignals(True)
        self.dataset_selector.clear()
        for name, dataset in items:
            label = f"{name} ({dataset.sample_count} x {dataset.point_count})"
            self.dataset_selector.addItem(label)
        self.dataset_selector.setCurrentIndex(selected_index)
        self.dataset_selector.blockSignals(False)
        has_multiple = len(items) > 1
        self.dataset_selector.setVisible(has_multiple)
        self.dataset_selector.setEnabled(has_multiple)
        self.dataset_label.setVisible(has_multiple)

    def _clear_parameter_controls(self) -> None:
        while self.parameter_layout.count():
            item = self.parameter_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._parameter_controls = []

    def _rebuild_parameter_controls(self) -> None:
        self._clear_parameter_controls()
        if self.state is None or self.state.dataset.labels is None:
            self.parameter_scroll.setVisible(False)
            self.parameter_hint.setVisible(True)
            return

        dataset = self.state.dataset
        current_labels = self.state.current_labels()
        for index, name in enumerate(dataset.param_names):
            lo = float(dataset.param_ranges[index, 0])
            hi = float(dataset.param_ranges[index, 1])
            value = float(current_labels[index])
            control = ParameterControl(name, lo, hi, value)
            control.valueChanged.connect(self.on_parameter_control_changed)
            self.parameter_layout.addWidget(control)
            self._parameter_controls.append(control)

        self.parameter_layout.addStretch()
        self.parameter_hint.setVisible(False)
        self.parameter_scroll.setVisible(True)

    def _set_controls_enabled(self, enabled: bool) -> None:
        self.sample_slider.setEnabled(enabled)
        self.prev_button.setEnabled(enabled)
        self.next_button.setEnabled(enabled)
        self.random_button.setEnabled(enabled)
        self.index_spin.setEnabled(enabled)
        self.split_selector.setEnabled(enabled)
        self.ymin_edit.setEnabled(enabled)
        self.apply_ymin_button.setEnabled(enabled)
        self.auto_ymin_button.setEnabled(enabled)

    def _select_archive_dataset(
        self,
        items: list[tuple[str, SpectrumDataset]],
        dataset_index: int | None = None,
        dataset_name: str | None = None,
    ) -> int | None:
        if dataset_index is not None and dataset_name is not None:
            raise ValueError("dataset_index and dataset_name cannot both be provided")

        if dataset_name is not None:
            for index, (name, _dataset) in enumerate(items):
                if name == dataset_name:
                    return index
            available = ", ".join(name for name, _dataset in items)
            raise ValueError(f"Unknown dataset_name: {dataset_name!r}. Available datasets: {available}")

        if dataset_index is not None:
            if not 0 <= dataset_index < len(items):
                raise ValueError(f"dataset_index {dataset_index} is out of range for {len(items)} datasets")
            return dataset_index

        if len(items) > 1:
            dialog = DatasetSelectionDialog(items, self)
            if dialog.exec():
                return dialog.selected_index
            return None

        return 0

    def load_path(self, file_path: str, dataset_index: int | None = None, dataset_name: str | None = None) -> bool:
        if not self.is_supported_path(file_path):
            ext = os.path.splitext(file_path)[1].lower()
            raise ValueError(
                f"Unsupported file extension: {ext}. Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
            )

        candidates = load_dataset_candidates(file_path)
        selected = self._select_archive_dataset(candidates, dataset_index=dataset_index, dataset_name=dataset_name)
        if selected is None:
            return False

        self._populate_dataset_selector(candidates, selected)
        self.load_data(candidates[selected][1])
        self.status_bar.showMessage(f"Loaded file: {file_path}", 5000)
        return True

    def open_file_dialog(self) -> None:
        file_path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "Open Dataset File",
            "",
            self.supported_file_filter(),
        )
        if not file_path:
            return
        try:
            self.load_path(file_path)
        except Exception as exc:
            QMessageBox.critical(self, "Load Error", f"Failed to load file:\n{exc}")

    def on_dataset_switched(self, index: int) -> None:
        if 0 <= index < len(self.available_datasets):
            self.load_data(self.available_datasets[index][1])

    def load_data(self, dataset: SpectrumDataset) -> None:
        previous_ymin = self.state.y_min if self.state is not None else None
        self.state = DatasetViewerState(dataset, y_min=previous_ymin)
        self.line_artist = None
        if not self.available_datasets:
            self._clear_dataset_selector()
        self._rebuild_parameter_controls()
        self.refresh_from_state(sync_parameters=True)

    def apply_ymin(self) -> None:
        if self.state is None:
            return
        text = self.ymin_edit.text().strip().lower()
        if text in ("", "auto", "a"):
            self.state.y_min = None
            self.refresh_from_state(sync_parameters=False)
            return
        try:
            self.state.y_min = float(text)
        except ValueError:
            QMessageBox.warning(self, "Invalid y minimum", "Enter a numeric lower limit or 'auto'.")
            return
        self.refresh_from_state(sync_parameters=False)

    def reset_ymin(self) -> None:
        if self.state is None:
            return
        self.state.y_min = None
        self.refresh_from_state(sync_parameters=False)

    def on_prev(self) -> None:
        if self.state is None:
            return
        self.state.move(-1)
        self.state.sync_target_to_current()
        self.refresh_from_state(sync_parameters=True)

    def on_next(self) -> None:
        if self.state is None:
            return
        self.state.move(+1)
        self.state.sync_target_to_current()
        self.refresh_from_state(sync_parameters=True)

    def on_random(self) -> None:
        if self.state is None:
            return
        self.state.randomize_current()
        self.state.sync_target_to_current()
        self.refresh_from_state(sync_parameters=True)

    def on_split_changed(self, _index: int) -> None:
        if self.state is None:
            return
        split_filter = self.split_selector.currentData()
        if split_filter is None:
            split_filter = self.split_selector.currentText()
        self.state.set_split_filter(str(split_filter))
        self.state.sync_target_to_current()
        self.refresh_from_state(sync_parameters=True)

    def on_sample_slider_changed(self, value: int) -> None:
        if self.state is None:
            return
        self.state.set_current_pool_position(value)
        self.state.sync_target_to_current()
        self.refresh_from_state(sync_parameters=True)

    def on_index_changed(self, value: int) -> None:
        if self.state is None:
            return
        self.state.jump_to_global_index(value)
        self.state.sync_target_to_current()
        self.refresh_from_state(sync_parameters=True)

    def on_parameter_control_changed(self, _value: float) -> None:
        if self.state is None or self._syncing_parameters or not self._parameter_controls:
            return
        target = np.asarray([control.value() for control in self._parameter_controls], dtype=float)
        self.state.set_parameter_target(target)
        self.refresh_from_state(sync_parameters=False)

    def _sync_parameter_controls(self) -> None:
        if self.state is None or self.state.dataset.labels is None:
            return
        labels = self.state.current_labels()
        if labels is None:
            return
        self._syncing_parameters = True
        for control, value in zip(self._parameter_controls, labels, strict=False):
            control.set_value(float(value))
        self._syncing_parameters = False

    def _summary_text(self) -> str:
        if self.state is None:
            return (
                "<b>No dataset loaded.</b><br>"
                "<span style='color:#64748b;'>Open a torch or numpy-style dataset and the viewer will detect the axis, "
                "spectra, split channels, and parameter channels automatically.</span>"
            )
        dataset = self.state.dataset
        x0 = _format_float(dataset.x[0])
        x1 = _format_float(dataset.x[-1])
        backend = dataset.spectra_backend.upper()
        lines = [
            f"<b>{dataset.sample_count:,}</b> spectra with <b>{dataset.point_count:,}</b> points each",
            f"<b>Axis</b>: {dataset.x_name} from {x0} to {x1}",
            f"<b>Signal</b>: {dataset.y_name}",
            f"<b>Backend</b>: {backend}",
        ]
        if dataset.parameter_count:
            lines.append(f"<b>Parameter channels</b>: {dataset.parameter_count}")
        if dataset.split_tags is not None:
            unique, counts = np.unique(dataset.split_tags, return_counts=True)
            split_summary = ", ".join(f"{name}={count}" for name, count in zip(unique, counts, strict=False))
            lines.append(f"<b>Splits</b>: {split_summary}")
        return "<br>".join(lines)

    def _current_sample_text(self) -> str:
        if self.state is None:
            return "<span style='color:#64748b;'>No sample selected.</span>"
        dataset = self.state.dataset
        lines = [
            f"<b>{dataset.name}</b>",
            f"Global index <b>{self.state.global_index}</b>",
            f"Visible position <b>{self.state.current_pool_position + 1} / {self.state.visible_count}</b>",
            f"Split <b>{self.state.current_split_tag()}</b>",
        ]
        labels = self.state.current_labels()
        if labels is not None:
            formatted = ", ".join(
                f"{name}={_format_float(value)}"
                for name, value in zip(dataset.param_names, labels, strict=False)
            )
            lines.append(f"<span style='color:#475569;'>{formatted}</span>")
        return "<br>".join(lines)

    def _match_text(self) -> str:
        if self.state is None or self.state.dataset.labels is None:
            return "<span style='color:#64748b;'>Parameter matching is unavailable for this dataset.</span>"
        if self.state.match_residuals is None:
            return "<span style='color:#64748b;'>Adjust the detected parameter channels to jump to the nearest spectrum.</span>"
        residuals = ", ".join(
            f"d{name}={_format_float(value)}"
            for name, value in zip(self.state.dataset.param_names, self.state.match_residuals, strict=False)
        )
        return f"<b>Nearest-match residuals</b><br><span style='color:#475569;'>{residuals}</span>"

    def update_controls_from_state(self, sync_parameters: bool) -> None:
        if self.state is None:
            self.hero_name_label.setText("Dataset Viewer")
            self.hero_meta_label.setText("Spectrum browsing for torch and numpy-style datasets.")
            self.hero_path_label.setText("Open a .pt, .pth, .npy, or .npz dataset to start.")
            self.backend_badge.setText("READY")
            self.visible_badge.setText("0 spectra")
            self.summary_label.setText(self._summary_text())
            self.current_info_label.setText(self._current_sample_text())
            self.match_label.setText(self._match_text())
            self.sample_slider.blockSignals(True)
            self.sample_slider.setMinimum(0)
            self.sample_slider.setMaximum(0)
            self.sample_slider.setValue(0)
            self.sample_slider.blockSignals(False)
            self.sample_label.setText("0 / 0")
            self.index_spin.blockSignals(True)
            self.index_spin.setRange(0, 0)
            self.index_spin.setValue(0)
            self.index_spin.blockSignals(False)
            self.split_selector.blockSignals(True)
            self.split_selector.clear()
            self.split_selector.addItem("all", userData="all")
            self.split_selector.blockSignals(False)
            self.ymin_edit.setText("auto")
            self._set_controls_enabled(False)
            return

        dataset = self.state.dataset
        source_path = str(dataset.metadata.get("source_path", ""))
        self.hero_name_label.setText(dataset.name)
        self.hero_meta_label.setText(
            f"{dataset.sample_count:,} spectra, {dataset.point_count:,} points, "
            f"{dataset.parameter_count} parameter channels"
        )
        self.hero_path_label.setText(source_path)
        self.backend_badge.setText(dataset.spectra_backend.upper())
        self.visible_badge.setText(f"{self.state.visible_count:,} visible")
        self.summary_label.setText(self._summary_text())
        self.current_info_label.setText(self._current_sample_text())
        self.match_label.setText(self._match_text())

        self.split_selector.blockSignals(True)
        self.split_selector.clear()
        filters = self.state.available_split_filters()
        for split_name in filters:
            self.split_selector.addItem(split_name, userData=split_name)
        current_split_index = filters.index(self.state.split_filter)
        self.split_selector.setCurrentIndex(current_split_index)
        self.split_selector.blockSignals(False)

        self.sample_slider.blockSignals(True)
        self.sample_slider.setMinimum(0)
        self.sample_slider.setMaximum(max(self.state.visible_count - 1, 0))
        self.sample_slider.setValue(self.state.current_pool_position)
        self.sample_slider.setEnabled(self.state.visible_count > 1)
        self.sample_slider.blockSignals(False)
        self.sample_label.setText(f"{self.state.current_pool_position + 1} / {self.state.visible_count}")

        self.index_spin.blockSignals(True)
        self.index_spin.setRange(0, max(dataset.sample_count - 1, 0))
        self.index_spin.setValue(self.state.global_index)
        self.index_spin.blockSignals(False)

        self.ymin_edit.setText("auto" if self.state.y_min is None else _format_float(self.state.y_min))
        self._set_controls_enabled(True)
        self.parameter_scroll.setVisible(bool(self._parameter_controls))
        self.parameter_hint.setVisible(not bool(self._parameter_controls))

        if sync_parameters:
            self._sync_parameter_controls()

    def refresh_plot(self) -> None:
        if self.state is None:
            self.axes.clear()
            self._configure_axes()
            self.axes.set_axis_off()
            self.axes.text(
                0.5,
                0.5,
                "Open or drop a torch or numpy dataset file",
                ha="center",
                va="center",
                fontsize=15,
                color=_CLR_EMPTY_TEXT,
                transform=self.axes.transAxes,
            )
            self.canvas.draw_idle()
            return

        dataset = self.state.dataset
        x = dataset.x
        y = self.state.current_spectrum()
        self.state.warm_visible_neighbors(radius=1)

        if self.line_artist is None:
            self.axes.clear()
            self._configure_axes()
            (self.line_artist,) = self.axes.plot(x, y, color=_CLR_PRIMARY, linewidth=1.7)
        else:
            self.line_artist.set_data(x, y)
            self.line_artist.set_color(_CLR_PRIMARY)
            self.axes.relim()
            self.axes.autoscale_view()

        self.axes.set_xlabel(dataset.x_name)
        self.axes.set_ylabel(dataset.y_name)
        title = (
            f"{dataset.name} | sample {self.state.global_index} | "
            f"visible {self.state.current_pool_position + 1}/{self.state.visible_count} | "
            f"split {self.state.current_split_tag()}"
        )
        self.axes.set_title(title, fontsize=11, pad=10, color=_CLR_HEADING)

        if self.state.y_min is not None:
            _bottom, top = self.axes.get_ylim()
            new_bottom = float(self.state.y_min)
            if top <= new_bottom:
                top = new_bottom + 1.0
            self.axes.set_ylim(new_bottom, top)

        self.canvas.draw_idle()

    def refresh_from_state(self, sync_parameters: bool) -> None:
        self.update_controls_from_state(sync_parameters=sync_parameters)
        self.refresh_plot()
        if self.state is None:
            self.status_bar.showMessage("Open or drop a supported dataset file to begin.")
        else:
            self.status_bar.showMessage(
                f"{self.state.dataset.name} loaded: {self.state.dataset.sample_count} spectra, "
                f"{self.state.dataset.point_count} points",
                4000,
            )

    def on_mouse_move(self, event) -> None:
        if self.state is None or event.inaxes != self.axes or event.xdata is None:
            return

        dataset = self.state.dataset
        x = dataset.x
        nearest = dataset.nearest_x_index(float(event.xdata))
        y_value = self.state.current_spectrum()[nearest]
        self.status_bar.showMessage(
            f"Sample {self.state.global_index} | {dataset.x_name}={_format_float(x[nearest])} | "
            f"{dataset.y_name}={_format_float(y_value)}"
        )
