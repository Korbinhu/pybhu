import numpy as np
import pytest

from stack_viewer.state import ViewerState


def test_rejects_non_numpy_input():
    with pytest.raises(TypeError):
        ViewerState([[1, 2], [3, 4]])


def test_rejects_non_3d_input():
    with pytest.raises(ValueError):
        ViewerState(np.ones((4, 4)))


def test_rejects_empty_input():
    with pytest.raises(ValueError):
        ViewerState(np.empty((0, 4, 2)))


def test_rejects_non_finite_values():
    data = np.ones((4, 4, 2))
    data[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        ViewerState(data)


def test_initializes_global_and_per_layer_limits_from_data():
    data = np.arange(24, dtype=float).reshape(2, 3, 4)
    state = ViewerState(data)
    assert state.layer_count == 4
    assert state.current_layer == 0
    assert state.global_limits == (0.0, 23.0)
    assert state.per_layer_limits[0] == (0.0, 20.0)


def test_updates_only_active_layer_in_per_layer_mode():
    state = ViewerState(np.arange(24, dtype=float).reshape(2, 3, 4))
    state.color_limit_mode = "per_layer"
    state.set_current_layer(2)
    state.set_limits(100.0, 200.0)
    assert state.per_layer_limits[2] == (100.0, 200.0)
    assert state.per_layer_limits[0] != (100.0, 200.0)


def test_updates_all_layers_in_global_mode():
    state = ViewerState(np.arange(24, dtype=float).reshape(2, 3, 4))
    state.set_limits(-1.0, 5.0)
    assert state.global_limits == (-1.0, 5.0)
    assert all(limits == (-1.0, 5.0) for limits in state.per_layer_limits)


def test_switching_to_per_layer_copies_global_limits_to_all_layers():
    state = ViewerState(np.arange(24, dtype=float).reshape(2, 3, 4))
    state.set_limits(-2.0, 9.0)
    state.set_color_limit_mode("per_layer")
    assert all(limits == (-2.0, 9.0) for limits in state.per_layer_limits)


def test_switching_to_global_promotes_active_layer_limits():
    state = ViewerState(np.arange(24, dtype=float).reshape(2, 3, 4))
    state.set_color_limit_mode("per_layer")
    state.set_current_layer(1)
    state.set_limits(11.0, 12.0)
    state.set_color_limit_mode("global")
    assert state.global_limits == (11.0, 12.0)
    assert all(limits == (11.0, 12.0) for limits in state.per_layer_limits)
