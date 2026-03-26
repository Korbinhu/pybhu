from __future__ import annotations

import os
import pickle
import warnings
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .bundle import FitBundle, MapDataset

SUPPORTED_EXTENSIONS = (".pkl", ".pickle", ".mat", ".pt", ".pth", ".npy", ".npz")

# ====================================================================== #
#  Flexible key alias lists (inspired by pybhu.img_viewer.loader)
# ====================================================================== #
# When loading generic files (.mat, .pt, etc.) that don't use the fixed
# pybhu fitting-results keys, we search for matching keys by priority.

X_KEYS = (
    # fitting-viewer legacy names first
    "raw_bias", "model_energy_meV",
    # common aliases
    "x", "energy", "energies", "axis", "bias", "biases",
    "grid", "omega", "frequency", "f", "e",
    "x_raw", "x_fit",
)

RAW_CUBE_KEYS = (
    "raw_y", "spectra_preprocessed",
    "y", "spectra", "signals", "curves", "data", "dos", "values",
    "raw_cube",
)

FIT_CUBE_KEYS = (
    "fitted_curves", "fit_cube", "fitted", "fit", "pred",
    "predictions", "reconstructed", "recon",
)

PARAM_STACK_KEYS = (
    "pred_params_final", "pred_params",
    "params", "parameters", "labels", "targets", "features",
)

PARAM_NAMES_KEYS = (
    "param_names", "parameter_names", "label_names",
    "channels", "channel_names",
)

PARAM_UNITS_KEYS = ("param_units", "parameter_units")

GRID_SHAPE_KEYS = ("grid_shape", "spatial_shape", "shape", "image_shape")


# ====================================================================== #
#  Helpers
# ====================================================================== #

def _find_key(values: Mapping, candidates: tuple[str, ...]) -> tuple[str | None, Any]:
    """Return the first matching (key, value) from *values* by alias priority."""
    lower_map = {str(k).lower(): k for k in values}
    for alias in candidates:
        real_key = lower_map.get(alias.lower())
        if real_key is not None and values[real_key] is not None:
            return real_key, values[real_key]
    return None, None


def _as_numpy(data: Any) -> np.ndarray | None:
    """Convert data to numpy, handling torch tensors transparently."""
    if isinstance(data, np.ndarray):
        return data
    try:
        import torch
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
    except ImportError:
        pass
    try:
        arr = np.asarray(data)
        if arr.dtype == object:
            return None
        return arr
    except Exception:
        return None


def _coerce_axis(data: Any) -> np.ndarray | None:
    """Try to coerce *data* into a 1-D numeric axis."""
    arr = _as_numpy(data)
    if arr is None:
        return None
    if arr.ndim == 1:
        return arr.astype(float)
    if arr.ndim == 2 and 1 in arr.shape:
        return arr.reshape(-1).astype(float)
    return None


def _reshape_cube(name: str, data, spatial_shape: tuple[int, int] | None) -> np.ndarray:
    array = np.asarray(data, dtype=float)
    if array.ndim == 3:
        return array
    if (
        array.ndim == 2
        and spatial_shape is not None
        and array.shape[0] == spatial_shape[0] * spatial_shape[1]
    ):
        return array.reshape(spatial_shape[0], spatial_shape[1], array.shape[1])
    raise ValueError(f"{name} must be a 3D array or a flattened 2D array")


def _reshape_map(name: str, data, spatial_shape: tuple[int, int] | None):
    array = np.asarray(data, dtype=float)
    if (
        array.ndim == 2
        and spatial_shape is not None
        and array.shape[0] == spatial_shape[0] * spatial_shape[1]
    ):
        return array.reshape(spatial_shape[0], spatial_shape[1], array.shape[1])
    if (
        array.ndim == 1
        and spatial_shape is not None
        and array.shape[0] == spatial_shape[0] * spatial_shape[1]
    ):
        return array.reshape(spatial_shape)
    if array.ndim in (2, 3):
        return array
    raise ValueError(f"{name} must be a 2D/3D array or a flattened spatial array")


def _infer_spatial_shape(values: Mapping) -> tuple[int, int] | None:
    # Explicit grid_shape key
    _, gs = _find_key(values, GRID_SHAPE_KEYS)
    if gs is not None:
        gs_arr = _as_numpy(gs)
        if gs_arr is not None:
            flat = gs_arr.reshape(-1)
            if flat.size >= 2:
                return int(flat[0]), int(flat[1])
            if flat.size == 1:
                return 1, int(flat[0])

    # Infer from any 3-D array
    for candidates in (RAW_CUBE_KEYS, FIT_CUBE_KEYS, PARAM_STACK_KEYS):
        _, data = _find_key(values, candidates)
        if data is None:
            continue
        arr = _as_numpy(data)
        if arr is not None and arr.ndim >= 2:
            return int(arr.shape[0]), int(arr.shape[1])

    return None


def _axis_in_mev(raw_bias, bias_to_mev):
    return np.asarray(raw_bias, dtype=float) * float(bias_to_mev)


def _build_map_datasets(
    values: Mapping, spatial_shape: tuple[int, int] | None
) -> tuple[MapDataset, ...]:
    datasets: list[MapDataset] = []
    seen_names: set[str] = set()
    seen_aliases: set[str] = set()

    def add_dataset(name: str, data, *, kind: str = "aux") -> None:
        if data is None:
            return
        normalized_name = str(name).strip()
        alias = normalized_name.split(" (", 1)[0].strip().casefold()
        if not normalized_name or normalized_name in seen_names or alias in seen_aliases:
            return
        try:
            datasets.append(
                MapDataset(
                    normalized_name,
                    _reshape_map(normalized_name, data, spatial_shape),
                    kind=kind,
                )
            )
            seen_names.add(normalized_name)
            seen_aliases.add(alias)
        except (ValueError, TypeError) as exc:
            warnings.warn(
                f"Skipping map dataset {normalized_name!r}: {exc}", stacklevel=3
            )

    _, param_stack = _find_key(values, PARAM_STACK_KEYS)
    _, param_names_raw = _find_key(values, PARAM_NAMES_KEYS)
    _, param_units_raw = _find_key(values, PARAM_UNITS_KEYS)

    param_names = _parse_string_list(param_names_raw)
    param_units = _parse_string_list(param_units_raw)

    if param_stack is not None:
        param_stack = _reshape_map("predicted parameters", param_stack, spatial_shape)
        add_dataset("Predicted Parameters", param_stack)
        arr = np.asarray(param_stack)
        if arr.ndim == 3:
            for idx in range(arr.shape[2]):
                pname = param_names[idx] if idx < len(param_names) else f"param_{idx}"
                unit = param_units[idx].strip() if idx < len(param_units) else ""
                label = f"{pname} ({unit})" if unit else pname
                add_dataset(label, arr[:, :, idx])

    # Named parameter sub-dict
    params_named = values.get("params_named")
    if isinstance(params_named, Mapping):
        for name, data in params_named.items():
            add_dataset(name, data)

    # Known scalar map names
    for name in ("Delta", "Gamma", "v_large", "v_small", "A", "vs_amp", "r2"):
        if name in values:
            add_dataset(name, values[name])

    return tuple(datasets)


def _parse_string_list(value: Any) -> list[str]:
    """Extract a list of strings from various storage formats."""
    if value is None:
        return []
    if isinstance(value, str):
        return [p.strip() for p in value.split(",") if p.strip()]
    arr = _as_numpy(value)
    if arr is not None and arr.ndim <= 1:
        return [str(item) for item in arr.reshape(-1)]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return []


# ====================================================================== #
#  bundle_from_results_mapping — flexible key matching
# ====================================================================== #

def bundle_from_results_mapping(values: Mapping) -> FitBundle:
    spatial_shape = _infer_spatial_shape(values)

    # --- raw cube (required) ---
    _, raw_cube_source = _find_key(values, RAW_CUBE_KEYS)
    if raw_cube_source is None:
        raise KeyError(
            "Could not find raw spectra. Expected one of: "
            + ", ".join(repr(k) for k in RAW_CUBE_KEYS)
        )

    # --- fit cube (optional, falls back to raw) ---
    _, fit_cube_source = _find_key(values, FIT_CUBE_KEYS)
    if fit_cube_source is None:
        warnings.warn(
            "No fitted curves found; using raw data as fit fallback.",
            stacklevel=2,
        )
        fit_cube_source = raw_cube_source

    # --- x axis ---
    x_raw = None
    # Try raw_bias + bias_to_mev conversion first
    if "raw_bias" in values and values.get("bias_to_mev") is not None:
        x_raw = _axis_in_mev(values["raw_bias"], values["bias_to_mev"])
    if x_raw is None:
        _, x_raw_source = _find_key(values, X_KEYS)
        if x_raw_source is not None:
            x_raw = _coerce_axis(x_raw_source)
    if x_raw is None:
        # Last resort: generate an integer axis matching the spectral dimension
        arr = _as_numpy(raw_cube_source)
        if arr is not None and arr.ndim >= 2:
            n_points = arr.shape[-1]
            x_raw = np.arange(n_points, dtype=float)
            warnings.warn(
                f"No x-axis found; using integer indices 0..{n_points - 1}.",
                stacklevel=2,
            )
        else:
            raise KeyError(
                "Could not find an x-axis. Expected one of: "
                + ", ".join(repr(k) for k in X_KEYS)
            )

    # Try a separate fit x-axis
    x_fit_source = values.get("model_energy_meV")
    if x_fit_source is None:
        x_fit_source = values.get("x_fit")
    x_fit = _coerce_axis(x_fit_source) if x_fit_source is not None else None
    if x_fit is None:
        x_fit = x_raw

    # --- display names ---
    raw_name = values.get("raw_name")
    raw_name = str(raw_name).strip() if raw_name is not None else "dI/dV"
    if not raw_name:
        raw_name = "dI/dV"
    fit_name = values.get("fit_name")
    fit_name = str(fit_name).strip() if fit_name is not None else "Fitted Curve"
    if not fit_name:
        fit_name = "Fitted Curve"

    # --- x / y units ---
    x_unit_val = values.get("x_unit")
    if x_unit_val is None:
        x_unit_val = values.get("x_name")
    x_unit = str(x_unit_val).strip() if x_unit_val is not None else "meV"

    y_unit_val = values.get("y_unit")
    if y_unit_val is None:
        y_unit_val = values.get("y_name")
    y_unit = str(y_unit_val).strip() if y_unit_val is not None else "DOS"

    return FitBundle(
        x_raw=np.asarray(x_raw, dtype=float),
        raw_cube=_reshape_cube("raw_cube", raw_cube_source, spatial_shape),
        x_fit=np.asarray(x_fit, dtype=float),
        fit_cube=_reshape_cube("fit_cube", fit_cube_source, spatial_shape),
        map_datasets=_build_map_datasets(values, spatial_shape),
        x_unit=x_unit,
        y_unit=y_unit,
        raw_name=raw_name,
        fit_name=fit_name,
    )


# ====================================================================== #
#  File-format loaders
# ====================================================================== #

def _tensor_to_numpy(obj: Any) -> Any:
    """Recursively convert torch Tensors to numpy arrays."""
    try:
        import torch
    except ImportError:
        return obj

    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy()
    if isinstance(obj, dict):
        return {k: _tensor_to_numpy(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_tensor_to_numpy(v) for v in obj)
    return obj


def _load_pickle(path: Path) -> object:
    with path.open("rb") as fh:
        return pickle.load(fh)  # noqa: S301


def _load_mat(path: Path) -> dict:
    try:
        import scipy.io as sio
    except ImportError as exc:
        raise ImportError(
            "scipy is required to load .mat files.  pip install scipy"
        ) from exc

    try:
        mat = sio.loadmat(str(path), squeeze_me=True)
    except NotImplementedError:
        try:
            import h5py
        except ImportError as exc2:
            raise ImportError(
                "h5py is required to load v7.3 .mat files.  pip install h5py"
            ) from exc2
        mat = {}
        with h5py.File(str(path), "r") as f:
            for key in f:
                obj = f[key]
                if hasattr(obj, "shape"):
                    mat[key] = np.asarray(obj)
                elif hasattr(obj, "keys"):
                    mat[key] = {
                        k: np.asarray(obj[k])
                        for k in obj
                        if hasattr(obj[k], "shape")
                    }

    return {k: v for k, v in mat.items() if not k.startswith("__")}


def _load_torch(path: Path) -> object:
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required to load .pt/.pth files.  pip install torch"
        ) from exc

    loaded = torch.load(str(path), map_location="cpu", weights_only=False)
    return _tensor_to_numpy(loaded)


def _load_numpy(path: Path) -> object:
    ext = path.suffix.lower()
    if ext == ".npy":
        return {"data": np.load(str(path), allow_pickle=False)}
    # .npz
    npz = np.load(str(path), allow_pickle=False)
    return dict(npz)


_LOADERS = {
    ".pkl":    _load_pickle,
    ".pickle": _load_pickle,
    ".mat":    _load_mat,
    ".pt":     _load_torch,
    ".pth":    _load_torch,
    ".npy":    _load_numpy,
    ".npz":    _load_numpy,
}


def load_fit_bundle(path) -> FitBundle:
    """Load a FitBundle from any supported file format."""
    path = Path(path)
    ext = path.suffix.lower()

    loader = _LOADERS.get(ext)
    if loader is None:
        raise ValueError(
            f"Unsupported file extension: {ext or '<none>'}. "
            f"Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    loaded = loader(path)

    if isinstance(loaded, FitBundle):
        return loaded
    if not isinstance(loaded, Mapping):
        raise TypeError(
            f"Unsupported object in {path.name}: {type(loaded)!r}. "
            f"Expected a dict/mapping with fitting results."
        )

    if {"x_raw", "raw_cube", "x_fit", "fit_cube"} <= set(loaded):
        return FitBundle.from_mapping(loaded)

    return bundle_from_results_mapping(loaded)


def coerce_fit_bundle(source) -> FitBundle:
    if isinstance(source, FitBundle):
        return source
    if isinstance(source, Mapping):
        if {"x_raw", "raw_cube", "x_fit", "fit_cube"} <= set(source):
            return FitBundle.from_mapping(source)
        return bundle_from_results_mapping(source)
    if isinstance(source, (str, os.PathLike)):
        return load_fit_bundle(source)
    raise TypeError(
        "bundle must be a FitBundle, mapping-like object, or supported file path"
    )
