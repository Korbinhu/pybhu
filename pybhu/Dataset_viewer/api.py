from __future__ import annotations

import os
import threading

from .viewer import SpectrumDatasetViewer


def dataset_viewer(data=None, block=None, **options):
    """
    Launch the spectrum dataset viewer.

    Parameters
    ----------
    data : SpectrumDataset | np.ndarray | Mapping | str | os.PathLike | None
        Data to display.  Can be a ``SpectrumDataset``, a numpy array, a mapping
        with x/y/labels keys, a supported dataset file path (.pt/.pth/.npy/.npz/
        .mat/.pkl), or ``None`` to open an empty viewer.
    block : bool | None
        Whether to run the Qt event loop before returning.

        * ``True``  — always block (call ``app.exec()``).
        * ``False`` — never block; the caller is responsible for running the loop.
        * ``None``  — block only if *this call* created the ``QApplication``
          (i.e. no Qt application was running before).  This is the safe default
          for scripts: interactive sessions that already have an event loop are
          not blocked, while standalone scripts get one automatically.
    **options
        ``initial_index`` / ``current_index`` : int
            Pool position to jump to after loading.
        ``dataset_index`` : int
            Which dataset to select when the file contains multiple candidates.
            Only valid when *data* is a file path.
        ``dataset_name`` : str
            Select a dataset by name when the file contains multiple candidates.
            Only valid when *data* is a file path.
        Any remaining keyword arguments are forwarded to
        ``SpectrumDatasetViewer.__init__``.

    Returns
    -------
    SpectrumDatasetViewer
    """
    path_to_load = None
    initial_index = options.pop("initial_index", options.pop("current_index", 0))
    dataset_index = options.pop("dataset_index", None)
    dataset_name = options.pop("dataset_name", None)

    if isinstance(data, (str, bytes, os.PathLike)):
        path_to_load = os.fspath(data)
        data = None
    elif dataset_index is not None or dataset_name is not None:
        raise ValueError("dataset_index and dataset_name are only supported when data is a file path")

    viewer = SpectrumDatasetViewer(data, **options)
    viewer.show()

    if path_to_load is not None:
        try:
            loaded = viewer.load_path(
                path_to_load,
                dataset_index=dataset_index,
                dataset_name=dataset_name,
            )
            if loaded and viewer.state is not None and initial_index != viewer.state.global_index:
                viewer.state.jump_to_global_index(initial_index)
                viewer.state.sync_target_to_current()
                viewer.refresh_from_state(sync_parameters=True)
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
