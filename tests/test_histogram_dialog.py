import numpy as np

from stack_viewer.histogram_dialog import HistogramDialog
from stack_viewer.state import ViewerState


def test_histogram_dialog_starts_with_visible_limits(qapp):
    state = ViewerState(np.arange(24, dtype=float).reshape(2, 3, 4))
    dialog = HistogramDialog(state)
    assert float(dialog.min_edit.text()) == state.visible_limits()[0]
    assert float(dialog.max_edit.text()) == state.visible_limits()[1]


def test_histogram_dialog_apply_updates_state(qapp):
    state = ViewerState(np.arange(24, dtype=float).reshape(2, 3, 4))
    dialog = HistogramDialog(state)
    dialog.min_edit.setText("1.5")
    dialog.max_edit.setText("9.5")
    dialog.apply_changes()
    assert state.visible_limits() == (1.5, 9.5)


def test_histogram_dialog_mode_change_updates_state(qapp):
    state = ViewerState(np.arange(24, dtype=float).reshape(2, 3, 4))
    dialog = HistogramDialog(state)
    dialog.mode_selector.setCurrentText("per_layer")
    assert state.color_limit_mode == "per_layer"
