import numpy as np
import pickle
import os
from scipy.io import loadmat

from . import nanonis_read


TORCH_EXTENSIONS = (".pt", ".pth")
LEGACY_STM_EXTENSIONS = (".1fl", ".2fl", ".ffl", ".tfr", ".1fr", ".ffr")
SUPPORTED_EXTENSIONS = (
    ".npy",
    ".pkl",
    ".npz",
    ".mat",
    ".sxm",
    ".3ds",
    *LEGACY_STM_EXTENSIONS,
    *TORCH_EXTENSIONS,
)
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


def _read_binary_scalar(handle, offset: int, dtype):
    dtype = np.dtype(dtype)
    handle.seek(offset, os.SEEK_SET)
    raw = handle.read(dtype.itemsize)
    if len(raw) != dtype.itemsize:
        raise ValueError("Unexpected end of file while reading STM header")
    return np.frombuffer(raw, dtype=dtype, count=1)[0]


def _read_legacy_stm_header(handle) -> dict[str, float | int]:
    li_expand = int(_read_binary_scalar(handle, 1356 + 29 - 1, "<u2"))
    if li_expand == 0:
        li_expand = 1

    return {
        "irows": int(_read_binary_scalar(handle, 256 + 151 - 1, "<u2")),
        "icols": int(_read_binary_scalar(handle, 256 + 155 - 1, "<u2")),
        "w_factor": float(_read_binary_scalar(handle, 256 + 179 - 1, "<f4")),
        "w_zero": float(_read_binary_scalar(handle, 256 + 183 - 1, "<f4")),
        "li_amp": float(_read_binary_scalar(handle, 1356 + 5 - 1, "<f4")),
        "li_sens": float(_read_binary_scalar(handle, 1356 + 13 - 1, "<f4")),
        "li_expand": li_expand,
        "li_foffset": float(_read_binary_scalar(handle, 1356 + 31 - 1, "<f4")),
    }


def _detrend_legacy_topography(layer: np.ndarray) -> np.ndarray:
    sy, sx = layer.shape
    coord_x = np.tile(np.arange(1, sy + 1, dtype=np.float64), sx)
    coord_y = np.repeat(np.arange(1, sx + 1, dtype=np.float64), sy)
    values = np.asarray(layer, dtype=np.float64).ravel(order="F")

    design = np.column_stack(
        (
            np.ones(values.size, dtype=np.float64),
            coord_x,
            coord_y,
            coord_x * coord_y,
            coord_x * coord_x,
            coord_y * coord_y,
            coord_x * coord_x * coord_x,
            coord_y * coord_y * coord_y,
        )
    )
    coeffs, *_ = np.linalg.lstsq(design, values, rcond=None)
    corrected = values - design @ coeffs
    return corrected.reshape((sy, sx), order="F")


def _normalize_legacy_stm_mode(value: str) -> str:
    mode = str(value).strip().lower()
    if mode not in {"auto", "stm1", "stm2"}:
        raise ValueError("legacy_stm must be 'auto', 'stm1', or 'stm2'")
    return mode


def _has_same_stem_sibling(path: str, suffix: str) -> bool:
    stem, _ = os.path.splitext(path)
    return os.path.exists(stem + suffix)


def _resolve_legacy_stm_kind(path: str, legacy_stm: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".2fl":
        return "conductance"
    if ext == ".ffl":
        return "current"
    if ext == ".tfr":
        return "topography"
    if ext in {".1fr", ".ffr"}:
        return "feedback"
    if ext != ".1fl":
        raise ValueError(f"Unsupported legacy STM extension: {ext}")

    mode = _normalize_legacy_stm_mode(legacy_stm)
    if mode == "stm1":
        return "conductance"
    if mode == "stm2":
        return "current"

    # `.1FL` is ambiguous: STM1 uses it for conductance, STM2 for current.
    # In auto mode prefer STM2 only when sibling files with the same stem
    # explicitly indicate that layout.
    if _has_same_stem_sibling(path, ".2fl") or _has_same_stem_sibling(path, ".1fr"):
        return "current"
    return "conductance"


def describe_supported_path(path: str, legacy_stm: str = "auto") -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in LEGACY_STM_EXTENSIONS:
        labels = {
            "conductance": "Conductance",
            "current": "Current",
            "topography": "Topography",
            "feedback": "Feedback Current",
        }
        return labels[_resolve_legacy_stm_kind(path, legacy_stm)]
    if ext == ".3ds":
        return "Nanonis Grid"
    if ext == ".sxm":
        return "Nanonis Scan"
    if ext == ".npy":
        return "NumPy Array"
    if ext == ".npz":
        return "NumPy Archive"
    if ext == ".mat":
        return "MAT Archive"
    if ext == ".pkl":
        return "Pickle Archive"
    if ext in TORCH_EXTENSIONS:
        return "Torch Archive"
    return ext.lstrip(".").upper() or "Data"


def _load_legacy_stm_map(path: str, legacy_stm: str = "auto") -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    kind = _resolve_legacy_stm_kind(path, legacy_stm)
    with open(path, "rb") as handle:
        header = _read_legacy_stm_header(handle)
        handle.seek(2112, os.SEEK_SET)
        payload = handle.read()

    if len(payload) % 2:
        payload = payload[:-1]
    raw = np.frombuffer(payload, dtype="<u2")

    size = max(header["irows"], header["icols"])
    if size <= 0:
        raise ValueError("Legacy STM file header has invalid image dimensions")

    layer_size = size * size
    if ext in {".tfr", ".ffr", ".1fr"}:
        raw = raw[:layer_size]
    else:
        if raw.size <= 1300:
            raise ValueError("Legacy STM file is too small to contain map data")
        raw = raw[:-1300]

    layer_count = raw.size // layer_size
    if layer_count <= 0:
        raise ValueError("Legacy STM file does not contain a complete image layer")
    raw = raw[: layer_count * layer_size]

    # Match the original MATLAB loader: reshape in column-major order, then
    # transpose each layer into the displayed image orientation.
    map_data = np.reshape(raw, (size, size, layer_count), order="F")
    map_data = np.transpose(map_data, (1, 0, 2)).astype(np.float64, copy=False)
    map_data = map_data * header["w_factor"] + header["w_zero"]

    if kind == "conductance":
        map_data = (map_data / (header["li_expand"] * 10.0) + header["li_foffset"] / 100.0) * header["li_sens"]
        if header["li_amp"] == 0:
            raise ValueError("Legacy STM conductance map has zero lock-in amplitude")
        map_data = map_data / (header["li_amp"] / 1000.0)
    elif kind == "current":
        map_data = -map_data

    if kind in {"topography", "feedback"}:
        corrected = np.stack(
            [_detrend_legacy_topography(map_data[:, :, index]) for index in range(map_data.shape[2])],
            axis=2,
        )
        return _validate_showable_array(corrected)

    return _validate_showable_array(map_data)


def _load_torch_archive(path: str):
    torch = _get_torch_module()
    if torch is None:
        raise ImportError("Loading .pt/.pth files requires PyTorch to be installed")

    import warnings
    warnings.warn(
        f"Loading PyTorch file '{os.path.basename(path)}' with weights_only=False — "
        "this can execute arbitrary code. Only open files you trust.",
        stacklevel=3,
    )
    try:
        loaded = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        # Older PyTorch versions do not have the weights_only parameter
        loaded = torch.load(path, map_location="cpu")
    return _convert_torch_structure(loaded)


def _load_pickle_archive(path: str):
    import warnings
    warnings.warn(
        f"Loading pickle file '{os.path.basename(path)}' — pickle files can execute "
        "arbitrary code. Only open files you trust.",
        stacklevel=3,
    )
    with open(path, 'rb') as f:
        return pickle.load(f)


def _load_npz_archive(path: str):
    try:
        with np.load(path, allow_pickle=False) as data:
            return dict(data)
    except ValueError:
        import warnings
        warnings.warn(
            f"Loading npz file '{os.path.basename(path)}' with allow_pickle=True — "
            "pickled data can execute arbitrary code. Only open files you trust.",
            stacklevel=3,
        )
        with np.load(path, allow_pickle=True) as data:
            return dict(data)

def load_data(path: str, legacy_stm: str = "auto") -> np.ndarray:
    """
    Load data from a .npy, .pkl, .npz, .mat, .sxm, .3ds, .1fl, .2fl, .ffl, .tfr,
    .1fr, .ffr,
    .pt, or .pth file.
    Returns a numpy.ndarray.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    
    ext = os.path.splitext(path)[1].lower()
    
    if ext == '.npy':
        data = np.load(path)
        return to_numpy(data)
    elif ext in LEGACY_STM_EXTENSIONS:
        return _load_legacy_stm_map(path, legacy_stm=legacy_stm)
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
