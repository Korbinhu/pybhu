"""Interactive viewer for spatially resolved fitting results."""

from .api import fit_viewer
from .bundle import FitBundle, MapDataset, ensure_fit_bundle
from .state import FittingViewerState, SelectionState
from .viewer import FittingViewer

__all__ = [
    "FitBundle",
    "MapDataset",
    "SelectionState",
    "FittingViewerState",
    "FittingViewer",
    "ensure_fit_bundle",
    "fit_viewer",
]
