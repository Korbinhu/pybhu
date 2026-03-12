import numpy as np

from stack_viewer.viewer import ImageStackViewer


def test_viewer_displays_first_layer_with_state_limits(qapp):
    data = np.arange(24, dtype=float).reshape(2, 3, 4)
    viewer = ImageStackViewer(data)
    viewer.refresh_image()
    array = viewer.image_artist.get_array()
    assert array.shape == (2, 3)
    assert tuple(viewer.image_artist.get_clim()) == viewer.state.visible_limits()


def test_viewer_refreshes_when_current_layer_changes(qapp):
    data = np.arange(24, dtype=float).reshape(2, 3, 4)
    viewer = ImageStackViewer(data)
    viewer.state.set_current_layer(3)
    viewer.refresh_image()
    assert viewer.image_artist.get_array()[0, 0] == data[0, 0, 3]


def test_colormap_change_updates_viewer_state(qapp):
    viewer = ImageStackViewer(np.arange(24, dtype=float).reshape(2, 3, 4))
    viewer.on_colormap_changed("plasma")
    assert viewer.state.colormap_name == "plasma"


def test_invert_toggle_updates_viewer_state(qapp):
    viewer = ImageStackViewer(np.arange(24, dtype=float).reshape(2, 3, 4))
    viewer.toggle_inverted()
    assert viewer.state.inverted is True


def test_open_colorbar_dialog_creates_single_window(qapp):
    viewer = ImageStackViewer(np.arange(24, dtype=float).reshape(2, 3, 4))
    viewer.open_colorbar_dialog()
    first = viewer.colorbar_dialog
    viewer.open_colorbar_dialog()
    assert viewer.colorbar_dialog is first


def test_open_histogram_dialog_tracks_active_layer(qapp):
    data = np.arange(24, dtype=float).reshape(2, 3, 4)
    viewer = ImageStackViewer(data, color_limit_mode="per_layer")
    viewer.open_histogram_dialog()
    viewer.state.set_limits(1.0, 2.0)
    viewer.on_layer_changed(3)
    expected = viewer.state.visible_limits()
    assert float(viewer.histogram_dialog.min_edit.text()) == expected[0]
    assert float(viewer.histogram_dialog.max_edit.text()) == expected[1]
