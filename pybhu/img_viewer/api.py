import os
import threading

from .viewer import ImageStackViewer


def img_viewer(data=None, block=None, **options):
    """
    Launch the ImageStackViewer.
    data can be a numpy array, a path to a .npy, .pkl, .npz, .mat, .sxm, .3ds, .1fl, .2fl, .ffl, .tfr, .1fr, .ffr, .pt, or .pth file,
    or None to start with an empty viewer and open files from the UI.
    Use origin='lower' or origin='upper' to control image orientation.
    Use legacy_stm='stm1' or legacy_stm='stm2' to disambiguate legacy `.1FL`
    files; the default auto mode falls back to STM1 unless sibling files imply
    the STM2 layout.
    For archive-style files with multiple datasets, use dataset_index or
    dataset_name to select one programmatically.
    block controls whether this helper runs the Qt event loop. By default it
    blocks only when it had to create the QApplication itself.
    """
    path_to_load = None
    initial_layer = options.get("initial_layer", options.get("current_layer", 0))
    dataset_index = options.pop("dataset_index", None)
    dataset_name = options.pop("dataset_name", None)
    if isinstance(data, (str, bytes, os.PathLike)):
        path_to_load = os.fspath(data)
        data = None
    elif dataset_index is not None or dataset_name is not None:
        raise ValueError("dataset_index and dataset_name are only supported when data is a file path")

    viewer = ImageStackViewer(data, **options)
    viewer.show()

    if path_to_load is not None:
        try:
            loaded = viewer.load_path(
                path_to_load,
                dataset_index=dataset_index,
                dataset_name=dataset_name,
            )
            if loaded and viewer.state is not None and initial_layer != viewer.state.current_layer:
                viewer.state.set_current_layer(initial_layer)
                viewer.update_ui_from_state()
                viewer.refresh_image()
        except Exception:
            viewer.close()
            raise

    if block is None:
        block = viewer.owns_app

    if (
        block
        and threading.current_thread() is threading.main_thread()
        and viewer.app.thread().loopLevel() == 0
    ):
        viewer.app.exec()

    return viewer
