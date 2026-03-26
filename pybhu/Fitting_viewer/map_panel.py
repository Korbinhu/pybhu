from __future__ import annotations

import numpy as np
from matplotlib import colormaps as mpl_colormaps
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt, pyqtSignal, QRect
from PyQt6.QtGui import QColor, QLinearGradient, QPainter, QPen
from PyQt6.QtWidgets import (
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .controls import COLORMAPS
from .state import FittingViewerState

SLOT_LETTERS: dict[str, str] = {
    "slot_a": "A",
    "slot_b": "B",
    "slot_c": "C",
    "slot_d": "D",
}

_BADGE_SS: dict[str, str] = {
    "slot_a": "background:#dbeafe; color:#2563eb; border:1px solid #93c5fd;",
    "slot_b": "background:#fce7f3; color:#db2777; border:1px solid #f9a8d4;",
    "slot_c": "background:#d1fae5; color:#059669; border:1px solid #6ee7b7;",
    "slot_d": "background:#ede9fe; color:#7c3aed; border:1px solid #c4b5fd;",
}

_VALUE_COLOR: dict[str, str] = {
    "slot_a": "#2563eb",
    "slot_b": "#db2777",
    "slot_c": "#059669",
    "slot_d": "#7c3aed",
}

_MAP_BG = "#f0f0f0"

_CROSS_COLOR = "#ff3333"
_CROSS_WIDTH = 1.0
_CROSS_ALPHA = 0.75

# Header stylesheet
_HEADER_SS = """
* {
    background: #f7f7f7;
    color: #555;
}
QComboBox {
    background: #fff; color: #333; border: 1px solid #ccc;
    border-radius: 3px; padding: 1px 4px; font-size: 10px; min-height: 18px;
}
QComboBox:hover { border-color: #999; }
QComboBox::drop-down { width: 14px; border-left: 1px solid #ddd; background: #f0f0f0; }
QComboBox::down-arrow {
    image: none;
    border-left: 3px solid transparent; border-right: 3px solid transparent;
    border-top: 4px solid #555; width: 0; height: 0;
}
QSpinBox {
    background: #fff; color: #333; border: 1px solid #ccc;
    border-radius: 3px; padding: 1px 2px; font-size: 10px; min-height: 18px;
}
QSpinBox:hover { border-color: #999; }
QSpinBox::up-button, QSpinBox::down-button {
    width: 14px; border: none; background: #f0f0f0;
}
QSpinBox::up-button:hover, QSpinBox::down-button:hover { background: #ddd; }
QSpinBox::up-arrow {
    image: none;
    border-left: 3px solid transparent; border-right: 3px solid transparent;
    border-bottom: 4px solid #444; width: 0; height: 0;
}
QSpinBox::down-arrow {
    image: none;
    border-left: 3px solid transparent; border-right: 3px solid transparent;
    border-top: 4px solid #444; width: 0; height: 0;
}
QLabel { border: none; background: transparent; }
"""

_HANDLE_W = 8  # half-width of each slider handle in pixels


class _RangeSlider(QWidget):
    """Horizontal dual-handle range slider with colormap gradient background."""

    range_changed = pyqtSignal(float, float)  # (vmin, vmax)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(18)
        self.setMinimumWidth(60)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        self._data_min = 0.0
        self._data_max = 1.0
        self._lo = 0.0        # current low value (normalised 0..1)
        self._hi = 1.0        # current high value (normalised 0..1)
        self._cmap_name = "viridis"
        self._cmap_colors: list[QColor] = []
        self._dragging: str | None = None  # "lo", "hi", "bar", None
        self._drag_offset = 0.0
        self._build_gradient_colors()

    # ── public API ───────────────────────────────────────────────────────

    def set_data_range(self, dmin: float, dmax: float, *, reset: bool = True) -> None:
        """Set the full data range. If *reset*, handles snap to full span."""
        if dmax <= dmin:
            dmax = dmin + 1.0
        if dmin == self._data_min and dmax == self._data_max:
            return
        self._data_min = dmin
        self._data_max = dmax
        if reset:
            self._lo = 0.0
            self._hi = 1.0
        self.update()

    def set_clim(self, vmin: float, vmax: float) -> None:
        """Set handle positions from absolute values."""
        span = self._data_max - self._data_min
        if span <= 0:
            return
        self._lo = max(0.0, min(1.0, (vmin - self._data_min) / span))
        self._hi = max(0.0, min(1.0, (vmax - self._data_min) / span))
        if self._lo > self._hi:
            self._lo, self._hi = self._hi, self._lo
        self.update()

    def clim(self) -> tuple[float, float]:
        span = self._data_max - self._data_min
        return (
            self._data_min + self._lo * span,
            self._data_min + self._hi * span,
        )

    def set_cmap(self, name: str) -> None:
        self._cmap_name = name
        self._build_gradient_colors()
        self.update()

    # ── internal ─────────────────────────────────────────────────────────

    def _build_gradient_colors(self) -> None:
        try:
            cm = mpl_colormaps[self._cmap_name]
        except KeyError:
            cm = mpl_colormaps["viridis"]
        self._cmap_colors = []
        for i in range(64):
            r, g, b, _ = cm(i / 63.0)
            self._cmap_colors.append(QColor(int(r * 255), int(g * 255), int(b * 255)))

    def _track_rect(self) -> QRect:
        return QRect(_HANDLE_W, 2, self.width() - 2 * _HANDLE_W, self.height() - 4)

    def _val_to_x(self, v: float) -> int:
        tr = self._track_rect()
        return tr.left() + int(v * tr.width())

    def _x_to_val(self, x: int) -> float:
        tr = self._track_rect()
        return max(0.0, min(1.0, (x - tr.left()) / max(tr.width(), 1)))

    # ── painting ─────────────────────────────────────────────────────────

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        tr = self._track_rect()

        # Draw colormap gradient background
        grad = QLinearGradient(tr.left(), 0, tr.right(), 0)
        n = len(self._cmap_colors)
        for i, c in enumerate(self._cmap_colors):
            grad.setColorAt(i / max(n - 1, 1), c)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(grad)
        p.drawRoundedRect(tr, 3, 3)

        # Dim regions outside the selected range
        lo_x = self._val_to_x(self._lo)
        hi_x = self._val_to_x(self._hi)
        dim = QColor(240, 240, 240, 180)
        p.setBrush(dim)
        if lo_x > tr.left():
            p.drawRect(QRect(tr.left(), tr.top(), lo_x - tr.left(), tr.height()))
        if hi_x < tr.right():
            p.drawRect(QRect(hi_x, tr.top(), tr.right() - hi_x, tr.height()))

        # Draw handles
        for hx in (lo_x, hi_x):
            p.setBrush(QColor("#ffffff"))
            p.setPen(QPen(QColor("#888888"), 1))
            p.drawRoundedRect(
                QRect(hx - _HANDLE_W // 2, 0, _HANDLE_W, self.height()), 2, 2,
            )

        p.end()

    # ── mouse interaction ────────────────────────────────────────────────

    def mousePressEvent(self, event) -> None:
        x = event.pos().x()
        lo_x = self._val_to_x(self._lo)
        hi_x = self._val_to_x(self._hi)

        if abs(x - lo_x) <= _HANDLE_W:
            self._dragging = "lo"
        elif abs(x - hi_x) <= _HANDLE_W:
            self._dragging = "hi"
        elif lo_x < x < hi_x:
            self._dragging = "bar"
            self._drag_offset = self._x_to_val(x) - self._lo
        else:
            # Click outside: snap nearest handle
            if x < lo_x:
                self._lo = self._x_to_val(x)
            else:
                self._hi = self._x_to_val(x)
            self._emit()
            self.update()

    def mouseMoveEvent(self, event) -> None:
        if self._dragging is None:
            return
        v = self._x_to_val(event.pos().x())
        if self._dragging == "lo":
            self._lo = min(v, self._hi - 0.005)
        elif self._dragging == "hi":
            self._hi = max(v, self._lo + 0.005)
        elif self._dragging == "bar":
            span = self._hi - self._lo
            new_lo = v - self._drag_offset
            new_lo = max(0.0, min(1.0 - span, new_lo))
            self._lo = new_lo
            self._hi = new_lo + span
        self._lo = max(0.0, self._lo)
        self._hi = min(1.0, self._hi)
        self._emit()
        self.update()

    def mouseReleaseEvent(self, _event) -> None:
        self._dragging = None

    def mouseDoubleClickEvent(self, _event) -> None:
        """Double-click resets to full range."""
        self._lo = 0.0
        self._hi = 1.0
        self._emit()
        self.update()

    def _emit(self) -> None:
        vmin, vmax = self.clim()
        self.range_changed.emit(vmin, vmax)


# =====================================================================


class _MapView(QWidget):
    selection_requested = pyqtSignal(int, int)
    add_point_requested = pyqtSignal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.origin = "upper"
        self.cmap = "viridis"
        self.figure = Figure(facecolor=_MAP_BG)
        self.figure.subplots_adjust(left=0, right=1, top=1, bottom=0)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.axes = self.figure.add_subplot(111)
        self.axes.set_facecolor(_MAP_BG)
        self.image_artist = None
        self.hline = None
        self.vline = None
        self._point_markers = []
        self._dragging = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)

        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("button_release_event", self._on_release)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)

    def set_origin(self, origin: str) -> None:
        if origin == self.origin:
            return
        self.origin = origin
        self._clear_artists()

    def set_cmap(self, cmap: str) -> None:
        self.cmap = cmap
        if self.image_artist is not None:
            self.image_artist.set_cmap(cmap)
            self.canvas.draw_idle()

    def _clear_artists(self) -> None:
        self.axes.clear()
        self.axes.set_facecolor(_MAP_BG)
        self.image_artist = None
        self.hline = None
        self.vline = None
        self._point_markers = []

    def _style_axes(self) -> None:
        self.axes.set_xticks([])
        self.axes.set_yticks([])
        for sp in self.axes.spines.values():
            sp.set_visible(False)

    def clear(self) -> None:
        self._clear_artists()
        self._style_axes()
        self.axes.text(
            0.5, 0.5, "No data",
            ha="center", va="center", transform=self.axes.transAxes,
            color="#bbb", fontsize=10,
        )
        self.canvas.draw_idle()

    def refresh(self, map_data: np.ndarray, *, i: int, j: int,
                 spectrum_points: list[tuple[int, int]] | None = None,
                 clim: tuple[float, float] | None = None) -> None:
        if clim is not None:
            vmin, vmax = clim
        else:
            vmin = float(np.nanmin(map_data))
            vmax = float(np.nanmax(map_data))
        if vmin >= vmax:
            vmax = vmin + 1.0

        if self.image_artist is None:
            self.axes.clear()
            self.axes.set_facecolor(_MAP_BG)
            self.image_artist = self.axes.imshow(
                map_data,
                cmap=self.cmap, origin=self.origin,
                interpolation="nearest", aspect="equal",
                vmin=vmin, vmax=vmax,
            )
            self._style_axes()
            self.hline = self.axes.axhline(
                i, color=_CROSS_COLOR, linestyle="--",
                linewidth=_CROSS_WIDTH, alpha=_CROSS_ALPHA,
            )
            self.vline = self.axes.axvline(
                j, color=_CROSS_COLOR, linestyle="--",
                linewidth=_CROSS_WIDTH, alpha=_CROSS_ALPHA,
            )
        else:
            self.image_artist.set_data(map_data)
            self.image_artist.set_clim(vmin, vmax)
            self.hline.set_ydata([i, i])
            self.vline.set_xdata([j, j])

        # Point markers
        for m in self._point_markers:
            try:
                m.remove()
            except (ValueError, TypeError):
                pass
        self._point_markers = []

        if spectrum_points:
            from .curve_panel import _MULTI_COLORS
            for idx, (pi, pj) in enumerate(spectrum_points):
                color = _MULTI_COLORS[(idx + 1) % len(_MULTI_COLORS)]
                (marker,) = self.axes.plot(
                    pj, pi, marker="+", color=color,
                    markersize=10, markeredgewidth=2, zorder=6,
                )
                self._point_markers.append(marker)

        self.canvas.draw_idle()

    # ── Mouse interaction ───────────────────────────────────────────────

    def _emit_selection(self, event, *, shift=False) -> None:
        if event.inaxes is not self.axes or event.xdata is None:
            return
        i, j = int(round(event.ydata)), int(round(event.xdata))
        if shift:
            self.add_point_requested.emit(i, j)
        else:
            self.selection_requested.emit(i, j)

    def _on_press(self, event) -> None:
        if event.inaxes is not self.axes or event.xdata is None:
            return
        shift = bool(event.guiEvent and hasattr(event.guiEvent, 'modifiers')
                     and event.guiEvent.modifiers() & Qt.KeyboardModifier.ShiftModifier)
        if shift:
            self._emit_selection(event, shift=True)
        else:
            self._dragging = True
            self._emit_selection(event)

    def _on_release(self, _event) -> None:
        self._dragging = False

    def _on_motion(self, event) -> None:
        if self._dragging:
            self._emit_selection(event)


# =====================================================================


class _MapSlot(QWidget):
    dataset_changed = pyqtSignal(int)
    layer_changed = pyqtSignal(int)
    selection_requested = pyqtSignal(int, int)
    add_point_requested = pyqtSignal(int, int)

    def __init__(self, role: str, parent=None):
        super().__init__(parent)
        self.role = role
        self._custom_clim: tuple[float, float] | None = None

        self.badge = QLabel(SLOT_LETTERS[role])
        self.badge.setFixedSize(20, 20)
        self.badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.badge.setStyleSheet(
            _BADGE_SS[role]
            + "border-radius: 3px; font-weight: 700; font-size: 10px;"
        )

        self.dataset_selector = QComboBox()
        self.dataset_selector.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon,
        )
        self.dataset_selector.setMinimumWidth(60)
        self.dataset_selector.currentIndexChanged.connect(self.dataset_changed.emit)

        self.cmap_selector = QComboBox()
        self.cmap_selector.setFixedWidth(72)
        for cmap in COLORMAPS:
            self.cmap_selector.addItem(cmap, cmap)
        self.cmap_selector.currentIndexChanged.connect(self._emit_cmap_changed)

        # ── slice group ─────────────────────────────────────────────────
        self.slice_widget = QWidget()
        sl = QHBoxLayout(self.slice_widget)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.setSpacing(2)

        lbl = QLabel("l")
        lbl.setStyleSheet("color:#999; font-size:9px;")
        lbl.setFixedWidth(8)
        self.layer_spin = QSpinBox()
        self.layer_spin.setFixedWidth(48)
        self.layer_spin.valueChanged.connect(self.layer_changed.emit)
        self.layer_of_label = QLabel("/ 1")
        self.layer_of_label.setStyleSheet("color:#aaa; font-size:9px;")

        sl.addWidget(lbl)
        sl.addWidget(self.layer_spin)
        sl.addWidget(self.layer_of_label)

        # ── value ───────────────────────────────────────────────────────
        val_clr = _VALUE_COLOR.get(role, "#333")
        self.value_label = QLabel("–")
        self.value_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
        )
        self.value_label.setStyleSheet(
            f"color:{val_clr}; font-size:10px; font-family:monospace; font-weight:600;"
        )

        # ── range slider for clim ───────────────────────────────────────
        self.range_slider = _RangeSlider()
        self.range_slider.range_changed.connect(self._on_slider_changed)

        # ── header row 1: badge + dataset + cmap + layer + value ─────
        hdr = QHBoxLayout()
        hdr.setContentsMargins(4, 2, 4, 2)
        hdr.setSpacing(3)
        hdr.addWidget(self.badge)
        hdr.addWidget(self.dataset_selector, 1)
        hdr.addWidget(self.cmap_selector)
        hdr.addWidget(self.slice_widget)
        hdr.addWidget(self.value_label)

        header = QWidget()
        header.setLayout(hdr)
        header.setFixedHeight(26)
        header.setAutoFillBackground(True)
        header.setStyleSheet(_HEADER_SS)

        # ── map canvas ──────────────────────────────────────────────────
        self.view = _MapView(self)
        self.view.selection_requested.connect(self.selection_requested.emit)
        self.view.add_point_requested.connect(self.add_point_requested.emit)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(header)
        layout.addWidget(self.view, 1)
        layout.addWidget(self.range_slider)

    def _on_slider_changed(self, vmin: float, vmax: float) -> None:
        """User dragged slider handles — apply custom clim."""
        self._custom_clim = (vmin, vmax)
        self._request_refresh()

    def _request_refresh(self) -> None:
        parent = self.parent()
        if parent is not None and hasattr(parent, '_refresh_slot'):
            parent._refresh_slot(self.role)

    def _emit_cmap_changed(self, index: int) -> None:
        cmap = self.cmap_selector.itemData(index)
        if cmap is not None:
            self.view.set_cmap(str(cmap))
            self.range_slider.set_cmap(str(cmap))

    def set_origin(self, o: str) -> None:
        self.view.set_origin(o)

    def set_cmap(self, c: str) -> None:
        self.cmap_selector.blockSignals(True)
        idx = self.cmap_selector.findData(c)
        if idx >= 0:
            self.cmap_selector.setCurrentIndex(idx)
        self.cmap_selector.blockSignals(False)
        self.view.set_cmap(c)
        self.range_slider.set_cmap(c)

    def clear(self) -> None:
        self.dataset_selector.blockSignals(True)
        self.dataset_selector.clear()
        self.dataset_selector.blockSignals(False)
        self.dataset_selector.setEnabled(False)

        self.layer_spin.blockSignals(True)
        self.layer_spin.setRange(0, 0)
        self.layer_spin.setValue(0)
        self.layer_spin.blockSignals(False)

        self.slice_widget.setVisible(False)
        self.layer_of_label.setText("/ 1")
        self.value_label.setText("–")
        self.view.clear()

    def sync_from_state(self, state: FittingViewerState) -> None:
        role = self.role

        options = state.available_dataset_indices_for_role(role)
        cur = state.dataset_index_for_role(role)
        self.dataset_selector.blockSignals(True)
        self.dataset_selector.clear()
        for di in options:
            self.dataset_selector.addItem(state.datasets[di].name, di)
        self.dataset_selector.setCurrentIndex(
            options.index(cur) if cur in options else 0,
        )
        self.dataset_selector.blockSignals(False)
        self.dataset_selector.setEnabled(True)

        lc = state.layer_count_for_role(role)
        self.layer_spin.blockSignals(True)
        self.layer_spin.setRange(0, max(lc - 1, 0))
        self.layer_spin.setValue(state.layer_index_for_role(role))
        self.layer_spin.blockSignals(False)
        self.slice_widget.setVisible(lc > 1)
        self.layer_of_label.setText(f"/ {lc}")

        self.value_label.setText(f"{state.value_for_role(role):.5g}")

        map_data = state.map_for_role(role)
        data_min = float(np.nanmin(map_data))
        data_max = float(np.nanmax(map_data))

        # Update slider data range (without emitting);
        # don't reset handles when a custom clim is active
        self.range_slider.blockSignals(True)
        has_custom = self._custom_clim is not None
        self.range_slider.set_data_range(data_min, data_max, reset=not has_custom)
        if has_custom:
            self.range_slider.set_clim(*self._custom_clim)
        self.range_slider.blockSignals(False)

        clim = self._custom_clim

        self.view.refresh(
            map_data,
            i=state.selection.i, j=state.selection.j,
            spectrum_points=state.spectrum_points,
            clim=clim,
        )


class MapPanel(QWidget):
    selection_requested = pyqtSignal(int, int)
    add_point_requested = pyqtSignal(int, int)
    dataset_changed = pyqtSignal(str, int)
    layer_changed = pyqtSignal(str, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._last_state: FittingViewerState | None = None
        self.slots: dict[str, _MapSlot] = {
            r: _MapSlot(r, self) for r in SLOT_LETTERS
        }

        grid = QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(2)
        grid.setVerticalSpacing(2)
        grid.addWidget(self.slots["slot_a"], 0, 0)
        grid.addWidget(self.slots["slot_b"], 0, 1)
        grid.addWidget(self.slots["slot_c"], 1, 0)
        grid.addWidget(self.slots["slot_d"], 1, 1)
        for c in (0, 1):
            grid.setColumnStretch(c, 1)
        for r in (0, 1):
            grid.setRowStretch(r, 1)

        for role, slot in self.slots.items():
            slot.selection_requested.connect(self.selection_requested.emit)
            slot.add_point_requested.connect(self.add_point_requested.emit)
            slot.dataset_changed.connect(
                lambda idx, r=role: self.dataset_changed.emit(r, idx),
            )
            slot.layer_changed.connect(
                lambda idx, r=role: self.layer_changed.emit(r, idx),
            )

        self.clear()

    def set_origin(self, o: str) -> None:
        for s in self.slots.values():
            s.set_origin(o)

    def clear(self) -> None:
        for s in self.slots.values():
            s.clear()

    def _refresh_slot(self, role: str) -> None:
        if self._last_state is not None and role in self.slots:
            self.slots[role].sync_from_state(self._last_state)

    def refresh(self, state: FittingViewerState | None) -> None:
        self._last_state = state
        if state is None:
            self.clear()
            return
        for s in self.slots.values():
            s.sync_from_state(state)
