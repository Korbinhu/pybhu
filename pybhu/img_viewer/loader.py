import numpy as np
import pickle
import os
from scipy.io import loadmat

from . import nanonis_read


TORCH_EXTENSIONS = (".pt", ".pth")
SUPPORTED_EXTENSIONS = (".npy", ".pkl", ".npz", ".mat", ".sxm", ".3ds", *TORCH_EXTENSIONS)
ARCHIVE_EXTENSIONS = (".pkl", ".npz", ".mat", ".sxm", ".3ds", *TORCH_EXTENSIONS)

# Kept for backward compatibility — other modules in the package import this name.
# All formats are now supported by default; this variable no longer gates loading.
UNSAFE_DESERIALIZATION_ENV = "IMG_VIEWER_ALLOW_UNSAFE_DESERIALIZATION"


def _unsafe_deserialization_allowed() -> bool:
    """Deprecated — no longer used internally. Kept for backward compatibility."""
    value = os.environ.get(UNSAFE_DESERIALIZATION_ENV, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _get_torch_module():
    try:
        import torch
    except ImportError:
        return None
    return torch


def _as_numpy_array(data):
    if isinstance(data, np.ndarray):
        return data

    torch = _get_torch_module()
    if torch is not None and isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()

    return None


def _convert_torch_structure(data):
    array = _as_numpy_array(data)
    if array is not None:
        return array
    if isinstance(data, dict):
        return {key: _convert_torch_structure(value) for key, value in data.items()}
    if isinstance(data, list):
        return [_convert_torch_structure(value) for value in data]
    if isinstance(data, tuple):
        return tuple(_convert_torch_structure(value) for value in data)
    return data


def _validate_showable_array(data: np.ndarray) -> np.ndarray:
    if not np.issubdtype(data.dtype, np.number):
        raise ValueError(f"Data must be numeric, got dtype {data.dtype}")
    if np.iscomplexobj(data):
        raise ValueError("Data must be real-valued, got complex data")
    if data.ndim not in [2, 3]:
        raise ValueError(f"Data must be 2D or 3D, got {data.ndim}D")
    if 0 in data.shape:
        raise ValueError("Data must not be empty")
    if not np.isfinite(data).all():
        raise ValueError(
            "Data must contain only finite values (no NaN or Inf). "
            "Pre-process with numpy.nan_to_num() to replace them before loading."
        )
    return data


def _is_showable_array(data: np.ndarray) -> bool:
    try:
        _validate_showable_array(data)
    except (TypeError, ValueError):
        return False
    return True


def _load_torch_archive(path: str):
    torch = _get_torch_module()
    if torch is None:
        raise ImportError("Loading .pt/.pth files requires PyTorch to be installed")

    try:
        loaded = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        # Older PyTorch versions do not have the weights_only parameter
        loaded = torch.load(path, map_location="cpu")
    return _convert_torch_structure(loaded)


def _load_pickle_archive(path: str):
    with open(path, 'rb') as f:
        return pickle.load(f)


def _load_npz_archive(path: str):
    with np.load(path, allow_pickle=True) as data:
        return dict(data)

def load_data(path: str) -> np.ndarray:
    """
    Load data from a .npy, .pkl, .npz, .mat, .sxm, .3ds, .pt, or .pth file.
    Returns a numpy.ndarray.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    
    ext = os.path.splitext(path)[1].lower()
    
    if ext == '.npy':
        data = np.load(path)
        return to_numpy(data)
    elif ext in ARCHIVE_EXTENSIONS:
        obj = get_archive_contents(path)
        showable = find_showable_data(obj)
        if showable:
            return to_numpy(showable[0][1])
        raise ValueError(f"No showable data found in {ext} file.")
    else:
        raise ValueError(
            f"Unsupported file extension: {ext}. Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

def to_numpy(data) -> np.ndarray:
    """Convert input data to a showable numpy array."""
    array = _as_numpy_array(data)
    if array is not None:
        data = array
    elif not isinstance(data, np.ndarray):
        data = np.array(data)

    return _validate_showable_array(data)

def get_archive_contents(path: str):
    """Load archive and return the raw object or dict."""
    ext = os.path.splitext(path)[1].lower()
    if ext == '.pkl':
        return _load_pickle_archive(path)
    elif ext == '.npz':
        return _load_npz_archive(path)
    elif ext == '.mat':
        return {
            key: value
            for key, value in loadmat(path, simplify_cells=True).items()
            if not key.startswith("__")
        }
    elif ext == '.sxm':
        scan = nanonis_read.Scan(path)
        return {
            channel: dict(direction_map)
            for channel, direction_map in scan.signals.items()
        }
    elif ext == '.3ds':
        grid = nanonis_read.Grid(path)
        return {
            name: value
            for name, value in grid.signals.items()
            if name != "params"
        }
    elif ext in TORCH_EXTENSIONS:
        return _load_torch_archive(path)
    raise ValueError(f"Unsupported archive extension: {ext}")

def find_showable_data(obj, path_prefix=""):
    """
    Recursively find keys/indices in an object that contain showable data.
    Returns a list of tuples (display_name, actual_data).
    """
    showable = []
    array = _as_numpy_array(obj)
    
    if array is not None:
        if _is_showable_array(array):
            showable.append((path_prefix or "Root Array", array))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            new_prefix = f"{path_prefix}/{k}" if path_prefix else str(k)
            showable.extend(find_showable_data(v, new_prefix))
    elif hasattr(obj, 'keys') and hasattr(obj, '__getitem__'): 
        # Support for NpzFile or similar dict-like
        for k in obj.keys():
            v = obj[k]
            new_prefix = f"{path_prefix}/{k}" if path_prefix else str(k)
            showable.extend(find_showable_data(v, new_prefix))
    elif isinstance(obj, (list, tuple)):
        try:
            arr = np.array(obj)
            if _is_showable_array(arr):
                showable.append((path_prefix or "List Data", arr))
                return showable
        except Exception:
            pass

        for i, v in enumerate(obj):
            new_prefix = f"{path_prefix}[{i}]" if path_prefix else f"[{i}]"
            showable.extend(find_showable_data(v, new_prefix))
                    
    return showable
