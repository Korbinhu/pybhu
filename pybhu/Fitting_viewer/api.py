from __future__ import annotations

from .bundle import FitBundle
from .viewer import FittingViewer, run_viewer


def fit_viewer(bundle=None, block: bool | None = None, **bundle_kwargs) -> FittingViewer:
    if bundle is None:
        if bundle_kwargs:
            bundle = FitBundle(**bundle_kwargs)
    elif bundle_kwargs:
        raise ValueError("Pass either a bundle or bundle fields, not both")

    viewer = FittingViewer(bundle)
    viewer.show()
    return run_viewer(viewer, block)
