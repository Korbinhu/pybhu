"""
Bi₂Sr₂CaCu₂O₈₊δ Tight Binding Parameter Predictor
Linear interpolation between adjacent doping levels.
PyQt6 + Matplotlib
"""

from __future__ import annotations

import sys
import threading
import types
from dataclasses import dataclass
from typing import Optional

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QTableWidget, QTableWidgetItem,
    QHeaderView, QFrame, QSizePolicy, QGridLayout, QScrollArea,
)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QFont, QColor, QDoubleValidator
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


# ── Domain Model ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Sample:
    name: str
    tc: float
    mu: float
    t: float
    tp: float
    tpp: float
    group: str  # "UD", "OD", or "BOTH"

    def param(self, key: str) -> float:
        return getattr(self, key)


SAMPLES = [
    Sample("UD30", 30, 340, 390, 120, 45, "UD"),
    Sample("UD78", 78, 330, 350, 100, 45, "UD"),
    Sample("OD91", 91, 405, 360, 108, 36, "BOTH"),
    Sample("OD67", 67, 430, 360, 108, 36, "OD"),
    Sample("OD35", 35, 445, 360, 108, 36, "OD"),
    Sample("OD0",  0,  467, 360, 108, 36, "OD"),
]

# Sets sorted by Tc ascending for interpolation
UD_SET = [SAMPLES[0], SAMPLES[1], SAMPLES[2]]         # 30, 78, 91
OD_SET = [SAMPLES[5], SAMPLES[4], SAMPLES[3], SAMPLES[2]]  # 0, 35, 67, 91

PARAMS = ("mu", "t", "tp", "tpp")
PARAM_LABELS = {"mu": "μ", "t": "t", "tp": "t′", "tpp": "t″"}
TABLE_FIELDS = ("name", "tc", "mu", "t", "tp", "tpp")
TABLE_HEADERS = ("Sample", "Tc (K)", "μ (meV)", "t (meV)", "t′ (meV)", "t″ (meV)")

TC_MATCH_THRESHOLD = 0.05


@dataclass
class Prediction:
    """Result of interpolation between two samples."""
    values: dict[str, float]
    bracket_lo: Sample
    bracket_hi: Sample
    tc: float
    regime: str

    @property
    def label(self) -> str:
        return f"◆ {self.regime}{self.tc}"


def interpolate(data_set: list[Sample], tc: float) -> tuple[dict[str, float], Sample, Sample]:
    """Linear interpolation between the two nearest bracketing samples."""
    n = len(data_set)
    if tc <= data_set[0].tc:
        lo, hi = 0, 1
    elif tc >= data_set[-1].tc:
        lo, hi = n - 2, n - 1
    else:
        lo, hi = 0, 1
        for i in range(n - 1):
            if data_set[i].tc <= tc <= data_set[i + 1].tc:
                lo, hi = i, i + 1
                break

    pt_a, pt_b = data_set[lo], data_set[hi]
    result: dict[str, float] = {}
    for param in PARAMS:
        if pt_b.tc == pt_a.tc:
            result[param] = float(pt_a.param(param))
        else:
            frac = (tc - pt_a.tc) / (pt_b.tc - pt_a.tc)
            result[param] = round(pt_a.param(param) + (pt_b.param(param) - pt_a.param(param)) * frac, 1)
    return result, pt_a, pt_b


def find_matching_sample(tc: float) -> Optional[Sample]:
    """Return a sample if tc is within threshold, else None."""
    for sample in SAMPLES:
        if abs(sample.tc - tc) < TC_MATCH_THRESHOLD:
            return sample
    return None


def build_display_rows(
    regime: str, tc: float, pred_values: dict[str, float],
    pt_a: Sample, pt_b: Sample, match: Optional[Sample],
) -> list[dict]:
    """Build table rows in original order with predicted row inserted."""
    rows = []
    for sample in SAMPLES:
        rows.append({
            "sample": sample, "predicted": False,
            "matched": sample is match,
            "bracket": sample.name in (pt_a.name, pt_b.name),
        })

    if match is not None:
        return rows

    pred_row = {
        "sample": None, "predicted": True, "matched": False, "bracket": False,
        "label": f"◆ {regime}{tc}", "tc": tc,
        "values": pred_values, "group": regime,
    }

    # Find correct insertion point based on table order within the active section
    ud_indices = {0, 1, 2}  # UD30, UD78, OD91
    od_indices = {2, 3, 4, 5}  # OD91, OD67, OD35, OD0
    active_indices = ud_indices if regime == "UD" else od_indices

    result = []
    inserted = False
    for i, row in enumerate(rows):
        sample = row["sample"]
        if not inserted and i in active_indices:
            if regime == "UD" and sample.tc > tc:
                result.append(pred_row)
                inserted = True
            elif regime == "OD" and sample.tc < tc:
                result.append(pred_row)
                inserted = True
        result.append(row)
        # If we've passed the last active index without inserting, append after
        if not inserted and i in active_indices:
            remaining_active = [j for j in active_indices if j > i]
            if not remaining_active:
                result.append(pred_row)
                inserted = True

    if not inserted:
        result.append(pred_row)

    return result


# ── UI Constants ─────────────────────────────────────────────────────────

BUTTON_HEIGHT = 44
INPUT_HEIGHT = 44
TABLE_ROW_HEIGHT = 38
TABLE_HEADER_HEIGHT = 40
CHART_HEIGHT = 100
MARGIN = 32
SPACING_SM = 8
SPACING_MD = 16
SPACING_LG = 24

FONT_BODY = "sans-serif"
FONT_MONO = "Consolas, Courier New, monospace"

class Colors:
    UD = "#534AB7"
    UD_BG = "#EEEDFE"
    UD_MID = "#AFA9EC"
    OD = "#0F6E56"
    OD_BG = "#E1F5EE"
    OD_MID = "#9FE1CB"
    BG = "#F7F7F5"
    PANEL_BG = "#EEEEEB"
    PANEL_BORDER = "#D5D3CC"
    TABLE_BORDER = "#C0BEB8"
    TABLE_GRID = "#E5E3DE"
    TEXT_PRIMARY = "#222"
    TEXT_BODY = "#333"
    TEXT_MUTED = "#888"
    TEXT_FAINT = "#999"


def regime_colors(regime: str) -> tuple[str, str, str]:
    """Return (primary, background, midtone) for the given regime."""
    if regime == "UD":
        return Colors.UD, Colors.UD_BG, Colors.UD_MID
    return Colors.OD, Colors.OD_BG, Colors.OD_MID


# ── Chart Widget ─────────────────────────────────────────────────────────

class ParamChart(FigureCanvas):
    def __init__(self) -> None:
        self.fig = Figure(figsize=(2.5, 1.0), dpi=100)
        self.fig.patch.set_facecolor(Colors.PANEL_BG)
        super().__init__(self.fig)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setFixedHeight(CHART_HEIGHT)

    def update_plot(
        self, param: str, pt_a: Sample, pt_b: Sample,
        tc: float, pred_val: float, color: str, line_color: str,
    ) -> None:
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_facecolor(Colors.PANEL_BG)

        xs = [pt_a.tc, pt_b.tc]
        ys = [pt_a.param(param), pt_b.param(param)]

        # Interpolation line + data points + predicted diamond
        ax.plot(xs, ys, "--", color=line_color, linewidth=1.5, zorder=1)
        ax.scatter(xs, ys, c=color, s=30, zorder=3, edgecolors="white", linewidths=1)
        ax.scatter([tc], [pred_val], c="white", s=40, zorder=4,
                   edgecolors=color, linewidths=1.5, marker="D")

        # Sample labels
        for sample, x, y in [(pt_a, xs[0], ys[0]), (pt_b, xs[1], ys[1])]:
            ax.annotate(
                sample.name, (x, y), textcoords="offset points",
                xytext=(0, 7), ha="center", fontsize=6.5, color="#444",
                fontweight="bold",
            )

        # Axis limits with padding
        all_x = xs + [tc]
        all_y = ys + [pred_val]
        x_pad = max((max(all_x) - min(all_x)) * 0.3, 8)
        y_pad = max((max(all_y) - min(all_y)) * 0.35, 4)
        ax.set_xlim(min(all_x) - x_pad, max(all_x) + x_pad)
        ax.set_ylim(min(all_y) - y_pad, max(all_y) + y_pad)

        ax.tick_params(labelsize=6, colors=Colors.TEXT_MUTED, length=2, pad=2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#ccc")
        ax.spines["bottom"].set_color("#ccc")
        ax.grid(True, alpha=0.2, linewidth=0.5)
        self.fig.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.18)
        self.draw()


# ── Main Window ──────────────────────────────────────────────────────────

class PredictorApp(QMainWindow):
    def __init__(self) -> None:
        self.app = QApplication.instance()
        self.owns_app = self.app is None
        if self.app is None:
            self.app = QApplication(sys.argv)
            self.app.setStyle("Fusion")

        super().__init__()
        self.setWindowTitle("Bi₂Sr₂CaCu₂O₈₊δ  Tight Binding Predictor")
        self.setMinimumSize(QSize(860, 700))
        self.resize(900, 920)
        self.setStyleSheet(f"QMainWindow {{ background: {Colors.BG}; }}")

        self._regime = "UD"
        self._tc = 50.0
        self._updating = False  # guard against double-fire

        self._setup_layout()
        self._build_header()
        self._build_controls()
        self._build_info()
        self._build_charts()
        self._build_table()
        self._root.addSpacing(SPACING_LG)

        self._refresh()

    def _setup_layout(self) -> None:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet(f"QScrollArea {{ background: {Colors.BG}; border: none; }}")
        self.setCentralWidget(scroll)

        container = QWidget()
        container.setStyleSheet(f"background: {Colors.BG};")
        scroll.setWidget(container)

        self._root = QVBoxLayout(container)
        self._root.setContentsMargins(MARGIN, MARGIN, MARGIN, MARGIN)
        self._root.setSpacing(0)

    def _build_header(self) -> None:
        title = QLabel("Bi₂Sr₂CaCu₂O₈₊δ")
        title.setFont(QFont(FONT_BODY, 18, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {Colors.TEXT_PRIMARY}; margin-bottom: 4px;")
        self._root.addWidget(title)

        subtitle = QLabel("Tight binding parameter predictor · linear interpolation")
        subtitle.setFont(QFont(FONT_BODY, 11))
        subtitle.setStyleSheet(f"color: {Colors.TEXT_MUTED}; margin-bottom: 22px;")
        self._root.addWidget(subtitle)

    def _build_controls(self) -> None:
        row = QHBoxLayout()
        row.setSpacing(SPACING_LG)

        # Regime toggle
        regime_col = QVBoxLayout()
        regime_col.setSpacing(SPACING_SM)
        regime_label = QLabel("REGIME")
        regime_label.setFont(QFont(FONT_BODY, 10, QFont.Weight.Bold))
        regime_label.setStyleSheet(f"color: {Colors.TEXT_MUTED};")
        regime_col.addWidget(regime_label)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(0)
        self._ud_btn = QPushButton("  UD · underdoped  ")
        self._od_btn = QPushButton("  OD · overdoped  ")
        for btn in (self._ud_btn, self._od_btn):
            btn.setFixedHeight(BUTTON_HEIGHT)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setFont(QFont(FONT_BODY, 13, QFont.Weight.Bold))
        self._ud_btn.clicked.connect(lambda: self._set_regime("UD"))
        self._od_btn.clicked.connect(lambda: self._set_regime("OD"))
        btn_row.addWidget(self._ud_btn)
        btn_row.addWidget(self._od_btn)
        regime_col.addLayout(btn_row)
        row.addLayout(regime_col, stretch=3)

        # Tc input
        tc_col = QVBoxLayout()
        tc_col.setSpacing(SPACING_SM)
        tc_label = QLabel("Tc (K)")
        tc_label.setFont(QFont(FONT_BODY, 10, QFont.Weight.Bold))
        tc_label.setStyleSheet(f"color: {Colors.TEXT_MUTED};")
        tc_col.addWidget(tc_label)

        input_row = QHBoxLayout()
        input_row.setSpacing(SPACING_SM)
        self._tc_input = QLineEdit("50")
        self._tc_input.setFixedSize(120, INPUT_HEIGHT)
        self._tc_input.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._tc_input.setValidator(QDoubleValidator(0.0, 999.0, 1))
        self._tc_input.setFont(QFont(FONT_BODY, 26, QFont.Weight.Bold))
        self._tc_input.editingFinished.connect(self._on_tc_changed)
        input_row.addWidget(self._tc_input)
        unit_label = QLabel("K")
        unit_label.setFont(QFont(FONT_BODY, 16, QFont.Weight.Bold))
        unit_label.setStyleSheet(f"color: {Colors.TEXT_MUTED};")
        input_row.addWidget(unit_label)
        tc_col.addLayout(input_row)
        row.addLayout(tc_col, stretch=1)

        self._root.addLayout(row)
        self._root.addSpacing(SPACING_MD)

    def _build_info(self) -> None:
        self._info_label = QLabel()
        self._info_label.setFont(QFont(FONT_BODY, 12))
        self._info_label.setTextFormat(Qt.TextFormat.RichText)
        self._info_label.setStyleSheet(f"color: #555; margin-bottom: {SPACING_MD}px;")
        self._info_label.setWordWrap(True)
        self._root.addWidget(self._info_label)

    def _build_charts(self) -> None:
        grid = QGridLayout()
        grid.setSpacing(10)

        self._charts: dict[str, ParamChart] = {}
        self._value_labels: dict[str, QLabel] = {}
        self._name_labels: dict[str, QLabel] = {}

        for idx, param in enumerate(PARAMS):
            row, col = divmod(idx, 2)

            panel = QFrame()
            panel.setObjectName(f"panel_{param}")
            panel.setStyleSheet(f"""
                QFrame#panel_{param} {{
                    background: {Colors.PANEL_BG};
                    border: 2px solid {Colors.PANEL_BORDER};
                    border-radius: 8px;
                }}
            """)
            panel_layout = QVBoxLayout(panel)
            panel_layout.setContentsMargins(10, 6, 10, 2)
            panel_layout.setSpacing(0)

            header = QHBoxLayout()
            name_label = QLabel(f"{PARAM_LABELS[param]} (meV)")
            name_label.setFont(QFont(FONT_BODY, 11, QFont.Weight.Bold))
            self._name_labels[param] = name_label
            header.addWidget(name_label)
            header.addStretch()
            value_label = QLabel("—")
            value_label.setFont(QFont(FONT_BODY, 18, QFont.Weight.Bold))
            value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self._value_labels[param] = value_label
            header.addWidget(value_label)
            panel_layout.addLayout(header)

            chart = ParamChart()
            panel_layout.addWidget(chart)
            self._charts[param] = chart

            grid.addWidget(panel, row, col)

        self._root.addLayout(grid)
        self._root.addSpacing(SPACING_LG)

    def _build_table(self) -> None:
        header_row = QHBoxLayout()
        title = QLabel("REFERENCE DATA")
        title.setFont(QFont(FONT_BODY, 11, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {Colors.TEXT_MUTED};")
        header_row.addWidget(title)
        header_row.addStretch()
        legend = QLabel()
        legend.setFont(QFont(FONT_BODY, 11))
        legend.setTextFormat(Qt.TextFormat.RichText)
        legend.setText(
            f'<span style="color:{Colors.UD}; font-size:16px;">■</span> UD    '
            f'<span style="color:{Colors.OD}; font-size:16px;">■</span> OD'
        )
        legend.setStyleSheet(f"color: {Colors.TEXT_MUTED};")
        header_row.addWidget(legend)
        self._root.addLayout(header_row)
        self._root.addSpacing(SPACING_SM)

        self._table = QTableWidget()
        self._table.setColumnCount(len(TABLE_HEADERS))
        self._table.setHorizontalHeaderLabels(TABLE_HEADERS)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._table.horizontalHeader().setFont(QFont(FONT_MONO, 10, QFont.Weight.Bold))
        self._table.horizontalHeader().setFixedHeight(TABLE_HEADER_HEIGHT)
        self._table.horizontalHeader().setStyleSheet(f"""
            QHeaderView::section {{
                background: #F0F0ED;
                color: #666;
                border: none;
                border-bottom: 2px solid {Colors.TABLE_BORDER};
                padding: 8px;
            }}
        """)
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self._table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._table.setFont(QFont(FONT_MONO, 12))
        self._table.setShowGrid(False)
        self._table.setStyleSheet(f"""
            QTableWidget {{
                background: white;
                border: 2px solid {Colors.TABLE_BORDER};
                border-radius: 10px;
                outline: none;
            }}
            QTableWidget::item {{
                padding: 6px 12px;
                border-bottom: 1px solid {Colors.TABLE_GRID};
            }}
        """)
        self._root.addWidget(self._table)

    # ── Actions ──────────────────────────────────────────────────────────

    def _set_regime(self, regime: str) -> None:
        self._regime = regime
        self._tc = 50.0
        self._tc_input.setText("50")
        self._refresh()

    def _on_tc_changed(self) -> None:
        if self._updating:
            return
        try:
            raw = float(self._tc_input.text())
            self._tc = round(max(0.0, raw), 1)  # clamp negatives
        except ValueError:
            pass
        self._tc_input.setText(str(self._tc))
        self._refresh()

    def _refresh(self) -> None:
        """Recompute predictions and update all UI elements."""
        self._updating = True
        try:
            self._refresh_inner()
        finally:
            self._updating = False

    def _refresh_inner(self) -> None:
        primary, bg, midtone = regime_colors(self._regime)
        data_set = UD_SET if self._regime == "UD" else OD_SET

        # Buttons
        self._style_buttons(primary)

        # Input
        self._tc_input.setStyleSheet(f"""
            QLineEdit {{
                color: {primary}; background: white;
                border: 2px solid {primary}; border-radius: 8px;
            }}
        """)

        # Interpolate
        predictions, pt_a, pt_b = interpolate(data_set, self._tc)
        match = find_matching_sample(self._tc)

        # Info label
        if match:
            self._info_label.setText(
                f'<span style="color:#555;">Tc = {self._tc} K matches </span>'
                f'<b style="color:{primary}; font-size:14px;">{match.name}</b>'
                f'<span style="color:#555;"> — exact values shown</span>'
            )
        else:
            self._info_label.setText(
                f'<span style="color:#555;">Interpolated between </span>'
                f'<b style="color:{primary}; font-size:14px;">{pt_a.name}</b>'
                f'<span style="color:#555;"> (Tc={pt_a.tc}) and </span>'
                f'<b style="color:{primary}; font-size:14px;">{pt_b.name}</b>'
                f'<span style="color:#555;"> (Tc={pt_b.tc})</span>'
            )

        # Charts + value labels
        for param in PARAMS:
            self._name_labels[param].setStyleSheet(f"color: {primary};")
            self._value_labels[param].setText(str(predictions[param]))
            self._value_labels[param].setStyleSheet(f"color: {primary};")
            self._charts[param].update_plot(
                param, pt_a, pt_b, self._tc, predictions[param], primary, midtone,
            )

        # Table
        rows = build_display_rows(self._regime, self._tc, predictions, pt_a, pt_b, match)
        self._populate_table(rows, match, primary)

    def _style_buttons(self, active_color: str) -> None:
        for which, btn in [("UD", self._ud_btn), ("OD", self._od_btn)]:
            is_active = self._regime == which
            color, bg_color, _ = regime_colors(which)
            is_left = which == "UD"
            rad = (
                f"border-top-left-radius:{'8' if is_left else '0'}px; "
                f"border-bottom-left-radius:{'8' if is_left else '0'}px; "
                f"border-top-right-radius:{'0' if is_left else '8'}px; "
                f"border-bottom-right-radius:{'0' if is_left else '8'}px;"
            )
            if is_active:
                btn.setStyleSheet(
                    f"QPushButton {{ background:{bg_color}; color:{color}; "
                    f"border:2px solid {color}; {rad} }}"
                )
            else:
                btn.setStyleSheet(
                    f"QPushButton {{ background:white; color:{Colors.TEXT_FAINT}; "
                    f"border:2px solid #CCC; {rad} }} "
                    f"QPushButton:hover {{ background:#F0F0ED; }}"
                )

    def _populate_table(
        self, rows: list[dict], match: Optional[Sample], active_color: str,
    ) -> None:
        self._table.setRowCount(len(rows))

        for i, row in enumerate(rows):
            self._table.setRowHeight(i, TABLE_ROW_HEIGHT)
            is_predicted = row["predicted"]
            is_matched = row["matched"]
            is_bracket = row["bracket"] and match is None

            # Determine colors
            if is_predicted:
                bg = QColor(active_color)
                bg.setAlpha(55)
                fg = QColor(active_color)
                bold = True
            elif is_matched:
                bg = QColor(active_color)
                bg.setAlpha(65)
                fg = QColor(active_color)
                bold = True
            elif is_bracket:
                bg = QColor(active_color)
                bg.setAlpha(45)
                fg = QColor(Colors.TEXT_PRIMARY)
                bold = True
            else:
                sample = row["sample"]
                is_ud = sample.group == "UD" or (
                    sample.group == "BOTH" and self._regime == "UD"
                )
                bg = QColor(Colors.UD_BG if is_ud else Colors.OD_BG)
                fg = QColor(Colors.TEXT_BODY)
                bold = False

            # Build cell values
            if is_predicted:
                values = [
                    row["label"], row["tc"],
                    row["values"]["mu"], row["values"]["t"],
                    row["values"]["tp"], row["values"]["tpp"],
                ]
            else:
                sample = row["sample"]
                values = [sample.name, sample.tc, sample.mu, sample.t, sample.tp, sample.tpp]

            for col, val in enumerate(values):
                item = QTableWidgetItem(str(val))
                align = (
                    Qt.AlignmentFlag.AlignLeft if col == 0
                    else Qt.AlignmentFlag.AlignRight
                )
                item.setTextAlignment(align | Qt.AlignmentFlag.AlignVCenter)
                item.setBackground(bg)
                item.setForeground(fg)
                font = QFont(FONT_MONO, 12)
                if bold:
                    font.setWeight(QFont.Weight.Bold)
                item.setFont(font)
                self._table.setItem(i, col, item)

        total_height = len(rows) * TABLE_ROW_HEIGHT + TABLE_HEADER_HEIGHT + SPACING_SM
        self._table.setMinimumHeight(total_height)
        self._table.setMaximumHeight(total_height)


# ── Entry Point ──────────────────────────────────────────────────────────

def BSCCO_TB(block: bool | None = None) -> PredictorApp:
    """
    Launch the BSCCO tight-binding predictor window.

    By default the helper starts the Qt event loop only when it had to create
    the QApplication itself.
    """
    window = PredictorApp()
    window.show()

    if block is None:
        block = window.owns_app

    if (
        block
        and threading.current_thread() is threading.main_thread()
        and window.app.thread().loopLevel() == 0
    ):
        window.app.exec()

    return window


def main() -> None:
    BSCCO_TB(block=True)


__all__ = [
    "BSCCO_TB",
    "PredictorApp",
    "Prediction",
    "Sample",
    "SAMPLES",
    "UD_SET",
    "OD_SET",
    "PARAMS",
    "TABLE_FIELDS",
    "TABLE_HEADERS",
    "TC_MATCH_THRESHOLD",
    "build_display_rows",
    "find_matching_sample",
    "interpolate",
    "main",
]


class _CallableModule(types.ModuleType):
    def __call__(self, *args, **kwargs):
        return BSCCO_TB(*args, **kwargs)


sys.modules[__name__].__class__ = _CallableModule


if __name__ == "__main__":
    main()
