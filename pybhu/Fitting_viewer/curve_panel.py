from __future__ import annotations

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import QFrame, QVBoxLayout, QWidget

from .controls import ControlPanel
from .state import FittingViewerState

# ── Light palette ───────────────────────────────────────────────────────
_FIG_BG   = "#f7f7f7"
_AXES_BG  = "#ffffff"
_TEXT     = "#333333"
_TEXT_SEC = "#777777"
_GRID     = "#ececec"
_SPINE    = "#cccccc"

_MULTI_COLORS = [
    "#1f77b4", "#e63946", "#2ca02c", "#ff7f0e", "#9467bd",
    "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#d62728",
]


class CurvePanel(QWidget):
    """Left panel: control bar on top, spectrum plot below."""

    open_file_requested = pyqtSignal()
    origin_changed = pyqtSignal(str)
    position_changed = pyqtSignal(int, int)
    add_point_requested = pyqtSignal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)

        # Force a light background on the entire widget — prevents dark
        # OS themes from bleeding through gaps between child widgets.
        self.setAutoFillBackground(True)
        pal = self.palette()
        pal.setColor(QPalette.ColorRole.Window, QColor("#f7f7f7"))
        self.setPalette(pal)

        # ── Control bar ─────────────────────────────────────────────────
        self.control_panel = ControlPanel(self)
        self.control_panel.open_file_requested.connect(self.open_file_requested.emit)
        self.control_panel.origin_changed.connect(self.origin_changed.emit)
        self.control_panel.position_changed.connect(self.position_changed.emit)
        self.control_panel.add_point_requested.connect(self.add_point_requested.emit)

        # ── Separator ───────────────────────────────────────────────────
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("background: #e0e0e0; border: none;")
        sep.setFixedHeight(1)

        # ── Matplotlib plot ─────────────────────────────────────────────
        self.figure = Figure(facecolor=_FIG_BG, dpi=100)
        self.figure.subplots_adjust(
            left=0.13, right=0.97, top=0.93, bottom=0.18,
        )
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.axes = self.figure.add_subplot(111)

        # ── Layout ──────────────────────────────────────────────────────
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.control_panel, 0)
        layout.addWidget(sep, 0)
        layout.addWidget(self.canvas, 1)

        self._style_axes()
        self.clear()

    # ------------------------------------------------------------------ #

    def sync_controls(
        self,
        state: FittingViewerState | None,
        *,
        file_label: str | None = None,
        origin: str = "upper",
    ) -> None:
        self.control_panel.sync_from_state(
            state, file_label=file_label, origin=origin,
        )

    # ------------------------------------------------------------------ #

    def _style_axes(self) -> None:
        ax = self.axes
        ax.set_facecolor(_AXES_BG)
        for sp in ax.spines.values():
            sp.set_color(_SPINE)
            sp.set_linewidth(0.5)
        ax.tick_params(
            colors=_TEXT, labelsize=9,
            direction="in", length=3, width=0.5, which="both",
        )
        ax.xaxis.label.set_color(_TEXT)
        ax.yaxis.label.set_color(_TEXT)
        ax.title.set_color(_TEXT)

    def _reset_axes(self) -> None:
        self.axes.clear()
        self._style_axes()

    # ------------------------------------------------------------------ #

    def clear(self) -> None:
        self._reset_axes()
        ax = self.axes
        ax.set_title("Spectra", fontsize=11, fontweight="medium")
        ax.set_xlabel("Energy", fontsize=9, color=_TEXT_SEC)
        ax.set_ylabel("Signal", fontsize=9, color=_TEXT_SEC)
        ax.text(
            0.5, 0.5, "Open a file to begin",
            ha="center", va="center", transform=ax.transAxes,
            color="#bbb", fontsize=12,
        )
        self.canvas.draw_idle()

    def refresh(self, state: FittingViewerState | None) -> None:
        if state is None:
            self.clear()
            return

        i, j = state.position
        self._reset_axes()
        ax = self.axes

        # All points to plot: current position + any extra spectrum_points
        all_points = [(i, j)]
        for pt in state.spectrum_points:
            if pt != (i, j) and pt not in all_points:
                all_points.append(pt)

        has_data = False
        for idx, (pi, pj) in enumerate(all_points):
            color = _MULTI_COLORS[idx % len(_MULTI_COLORS)]
            alpha = 1.0 if idx == 0 else 0.5
            lw = 1.5 if idx == 0 else 1.0

            # Slot A (raw) spectrum
            a_ds = state.dataset_for_role("slot_a")
            if a_ds.data.ndim == 3:
                a_x = state.x_axis_for_role("slot_a")
                a_y = a_ds.data[pi, pj, :]
                ax.scatter(
                    a_x, a_y,
                    color=color, s=6 if idx == 0 else 3, linewidths=0, alpha=alpha * 0.65,
                    label=f"({pi},{pj}) A" if len(all_points) > 1 else f"A: {a_ds.name}",
                    zorder=3,
                )
                has_data = True

            # Slot B (fit) spectrum
            b_ds = state.dataset_for_role("slot_b")
            if b_ds.data.ndim == 3:
                b_x = state.x_axis_for_role("slot_b")
                b_y = b_ds.data[pi, pj, :]
                ax.plot(
                    b_x, b_y,
                    color=color, linewidth=lw, alpha=alpha * 0.85,
                    linestyle="-" if idx == 0 else "--",
                    label=f"({pi},{pj}) B" if len(all_points) > 1 else f"B: {b_ds.name}",
                    zorder=4,
                )
                has_data = True

        if not has_data:
            ax.text(
                0.5, 0.5, "Select 3-D datasets for A / B",
                ha="center", va="center", transform=ax.transAxes,
                color="#bbb", fontsize=10,
            )

        if len(all_points) == 1:
            ax.set_title(f"Spectra at ({i}, {j})", fontsize=11, fontweight="medium")
        else:
            ax.set_title(f"Spectra — {len(all_points)} points", fontsize=11, fontweight="medium")

        x_u = state.bundle.x_unit.strip()
        y_u = state.bundle.y_unit.strip()
        ax.set_xlabel(f"Energy ({x_u})" if x_u else "Energy", fontsize=9, color=_TEXT_SEC)
        ax.set_ylabel(f"Signal ({y_u})" if y_u else "Signal", fontsize=9, color=_TEXT_SEC)
        ax.grid(True, linestyle="-", linewidth=0.3, color=_GRID, alpha=0.8)

        if has_data:
            ncol = min(len(all_points) * 2, 4)
            ax.legend(
                loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=ncol,
                fontsize=7, frameon=True, fancybox=False,
                framealpha=0.9, facecolor="#fff", edgecolor="#ddd",
            )

        self.canvas.draw_idle()
