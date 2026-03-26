"""Interactive viewer for spatially resolved fitting results."""

from .api import fit_viewer
from .bundle import FitBundle, MapDataset, ensure_fit_bundle
from .loader import bundle_from_results_mapping, coerce_fit_bundle, load_fit_bundle
from .state import FittingViewerState, SelectionState
from .viewer import FittingViewer, run_viewer

__all__ = [
    "FitBundle",
    "MapDataset",
    "SelectionState",
    "FittingViewerState",
    "FittingViewer",
    "bundle_from_results_mapping",
    "coerce_fit_bundle",
    "ensure_fit_bundle",
    "fit_viewer",
    "load_fit_bundle",
    "run_viewer",
]
