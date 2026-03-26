from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from pybhu.img_viewer.loader import get_archive_contents

# UNSAFE_DESERIALIZATION_ENV is the shared env-var name used across pybhu loaders.
# It is re-imported from img_viewer.loader so both subpackages honour the same variable.
# If this cross-package coupling becomes a problem, promote this constant to pybhu.io.
from pybhu.img_viewer.loader import UNSAFE_DESERIALIZATION_ENV

SUPPORTED_EXTENSIONS = (".pt", ".pth", ".npy", ".npz", ".mat", ".pkl")
# All supported extensions except .npy, which is loaded directly via np.load rather than
# through the archive helper used for the rest.
ARCHIVE_EXTENSIONS = (".pt", ".pth", ".npz", ".mat", ".pkl")

SPLIT_NAMES = ("train", "val", "test")
X_KEYS = ("x", "energy", "energies", "axis", "bias", "biases", "grid", "omega", "frequency", "f", "e")
Y_KEYS = ("y", "spectra", "signals", "curves", "data", "dos", "values")
LABEL_KEYS = ("labels", "params", "parameters", "targets", "features", "conditions", "metadata")
PARAM_NAMES_KEYS = (
    "param_names",
    "parameter_names",
    "label_names",
    "channels",
    "channel_names",
    "input_channels",
    "parameter_channels",
)
PARAM_RANGES_KEYS = ("param_ranges", "parameter_ranges", "label_ranges", "ranges")
TITLE_KEYS = ("title", "name")
X_NAME_KEYS = ("x_name", "x_label", "axis_name")
Y_NAME_KEYS = ("y_name", "y_label", "signal_name", "spectrum_name")
SPLIT_TAG_KEYS = ("split_tags", "splits", "split", "subset", "subsets", "partition")

# Keys that have well-known roles and should never be auto-detected as an axis or spectra
# candidate during heuristic mapping traversal.
_RESERVED_KEYS: frozenset[str] = frozenset(
    X_KEYS + Y_KEYS + LABEL_KEYS + PARAM_NAMES_KEYS + PARAM_RANGES_KEYS
    + TITLE_KEYS + X_NAME_KEYS + Y_NAME_KEYS + SPLIT_TAG_KEYS
)


def _get_torch_module():
    try:
        import torch
    except ImportError:
        return None
    return torch


def _unsafe_deserialization_allowed() -> bool:
    value = os.environ.get(UNSAFE_DESERIALIZATION_ENV, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _load_torch_archive_raw(path: str):
    torch = _get_torch_module()
    if torch is None:
        raise ImportError("Loading .pt/.pth files requires PyTorch to be installed")

    if _unsafe_deserialization_allowed():
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(path, map_location="cpu")

    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as exc:
        raise RuntimeError(
            "Safe loading of .pt/.pth files requires a PyTorch version that supports weights_only. "
            f"Update PyTorch or set {UNSAFE_DESERIALIZATION_ENV}=1 to load trusted files."
        ) from exc
    except Exception as exc:
        raise ValueError(
            "Failed to load this .pt/.pth file in safe mode. Only tensor-style PyTorch archives are "
            f"supported by default. Set {UNSAFE_DESERIALIZATION_ENV}=1 to load trusted files. "
            f"Original error: {exc}"
        ) from exc


def _is_torch_tensor(data: Any) -> bool:
    torch = _get_torch_module()
    return bool(torch is not None and isinstance(data, torch.Tensor))


def _as_numpy_array(data: Any) -> np.ndarray | None:
    if isinstance(data, np.ndarray):
        return data

    if _is_torch_tensor(data):
        return data.detach().cpu().numpy()

    return None


def _coerce_small_numeric_array(data: Any, require_finite: bool = True) -> np.ndarray | None:
    array = _as_numpy_array(data)
    if array is None:
        try:
            array = np.asarray(data)
        except Exception:
            return None

    if not isinstance(array, np.ndarray):
        return None
    if array.dtype == object:
        return None
    if not np.issubdtype(array.dtype, np.number):
        return None
    if np.iscomplexobj(array):
        return None
    if array.size == 0:
        return None
    array = np.asarray(array, dtype=float)
    if require_finite and not np.isfinite(array).all():
        return None
    return array


def _coerce_axis(data: Any) -> np.ndarray | None:
    array = _coerce_small_numeric_array(data)
    if array is None:
        return None
    if array.ndim == 1:
        return array
    if array.ndim == 2 and 1 in array.shape:
        return array.reshape(-1)
    return None


def _mapping_items(data: Any) -> list[tuple[Any, Any]]:
    if isinstance(data, Mapping):
        return list(data.items())
    if hasattr(data, "keys") and hasattr(data, "__getitem__"):
        return [(key, data[key]) for key in data.keys()]
    return []


def _key_name(key: Any) -> str:
    return str(key)


def _key_lower(key: Any) -> str:
    return _key_name(key).lower()


def _priority_key_match(items: list[tuple[Any, Any]], candidates: Iterable[str]) -> list[tuple[Any, Any]]:
    matched: list[tuple[Any, Any]] = []
    used: set[Any] = set()
    for candidate in candidates:
        for key, value in items:
            if key in used:
                continue
            if _key_lower(key) == candidate:
                matched.append((key, value))
                used.add(key)
    return matched


def _priority_prefixed_key_match(
    items: list[tuple[Any, Any]],
    suffixes: Iterable[str],
) -> list[tuple[str, Any, Any]]:
    matched: list[tuple[str, Any, Any]] = []
    used: set[Any] = set()
    for suffix in suffixes:
        token = f"_{suffix}"
        for key, value in items:
            if key in used:
                continue
            key_lower = _key_lower(key)
            if not key_lower.endswith(token):
                continue
            prefix = key_lower[: -len(token)]
            if not prefix:
                continue
            matched.append((prefix, key, value))
            used.add(key)
    return matched


def _shape_of(data: Any) -> tuple[int, ...] | None:
    if isinstance(data, np.ndarray):
        return tuple(int(dim) for dim in data.shape)
    if _is_torch_tensor(data):
        return tuple(int(dim) for dim in data.shape)
    try:
        shape = data.shape
    except Exception:
        return None
    try:
        return tuple(int(dim) for dim in shape)
    except Exception:
        return None


def _transpose_source(data: Any):
    if isinstance(data, np.ndarray):
        return data.T
    if _is_torch_tensor(data):
        return data.transpose(0, 1)
    return np.asarray(data).T


def _coerce_spectra_source(data: Any, point_count: int | None = None):
    shape = _shape_of(data)
    if shape is None:
        try:
            data = np.asarray(data)
        except Exception:
            return None
        shape = tuple(int(dim) for dim in data.shape)

    if len(shape) == 1:
        if isinstance(data, np.ndarray):
            data = data.reshape(1, -1)
        elif _is_torch_tensor(data):
            data = data.reshape(1, -1)
        else:
            data = np.asarray(data, dtype=float).reshape(1, -1)
        shape = _shape_of(data)
    elif len(shape) != 2:
        return None

    if isinstance(data, np.ndarray):
        if data.dtype == object or not np.issubdtype(data.dtype, np.number) or np.iscomplexobj(data):
            return None
    elif _is_torch_tensor(data):
        if data.is_complex():
            return None
    else:
        try:
            data = np.asarray(data, dtype=float)
        except Exception:
            return None
        if data.ndim != 2:
            return None

    if point_count is not None:
        shape = _shape_of(data)
        if shape is None:
            return None
        if shape[1] == point_count:
            pass
        elif shape[0] == point_count:
            data = _transpose_source(data)
        else:
            return None

    shape = _shape_of(data)
    if shape is None or 0 in shape:
        return None
    return data


def _coerce_labels_array(data: Any, sample_count: int) -> np.ndarray | None:
    # Labels may legitimately contain NaN for masked/missing values, so we do not
    # require finiteness here (unlike axis values, which must always be finite).
    array = _coerce_small_numeric_array(data, require_finite=False)
    if array is None:
        return None

    if array.ndim == 1:
        if sample_count == 1:
            array = array.reshape(1, -1)
        elif array.size == sample_count:
            array = array.reshape(-1, 1)
        else:
            return None
    elif array.ndim == 2:
        if array.shape[0] == sample_count:
            pass
        elif array.shape[1] == sample_count:
            array = array.T
        else:
            return None
    else:
        return None

    if 0 in array.shape:
        return None
    return array


def _normalize_param_names(names: list[str], count: int) -> list[str]:
    normalized = [name.strip() for name in names[:count]]
    while len(normalized) < count:
        normalized.append(f"param_{len(normalized)}")
    return [name or f"param_{index}" for index, name in enumerate(normalized)]


def _parse_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",")]
        return [part for part in parts if part]
    if isinstance(value, Mapping):
        return [_key_name(key) for key in value.keys()]

    array = _as_numpy_array(value)
    if array is not None and array.ndim <= 1:
        flat = array.reshape(-1)
        return [_key_name(item) for item in flat]

    if isinstance(value, (list, tuple)):
        return [_key_name(item) for item in value]
    return []


def _parse_param_ranges(value: Any, param_names: list[str], labels: np.ndarray) -> np.ndarray | None:
    if value is None:
        return None

    if isinstance(value, Mapping):
        ordered: list[tuple[float, float]] = []
        if param_names and all(name in value for name in param_names):
            keys = param_names
        else:
            keys = list(value.keys())
        for key in keys:
            pair = _coerce_small_numeric_array(value[key])
            if pair is None:
                return None
            flat = pair.reshape(-1)
            if flat.size < 2:
                return None
            ordered.append((float(flat[0]), float(flat[1])))
        ranges = np.asarray(ordered, dtype=float)
    else:
        array = _coerce_small_numeric_array(value)
        if array is None:
            return None
        if array.ndim == 1 and array.size == 2:
            ranges = array.reshape(1, 2)
        elif array.ndim == 2 and array.shape[1] == 2:
            ranges = array
        else:
            return None

    if ranges.shape[0] != labels.shape[1]:
        return None
    return ranges


def _split_counts_to_tags(split_mapping: Mapping[Any, Any], sample_count: int) -> np.ndarray | None:
    ordered_items = []
    seen = set()
    for split_name in SPLIT_NAMES:
        if split_name in split_mapping:
            ordered_items.append((split_name, split_mapping[split_name]))
            seen.add(split_name)
    for key, value in split_mapping.items():
        if key in seen:
            continue
        ordered_items.append((_key_name(key), value))

    tags: list[str] = []
    for name, count in ordered_items:
        if not isinstance(count, (int, np.integer)):
            return None
        if count < 0:
            return None
        tags.extend([str(name)] * int(count))

    if not tags or len(tags) > sample_count:
        return None
    if len(tags) < sample_count:
        tags.extend(["other"] * (sample_count - len(tags)))
    return np.asarray(tags, dtype=object)


def _extract_split_tags(mapping: Mapping[Any, Any], sample_count: int) -> np.ndarray | None:
    items = _mapping_items(mapping)
    for _key, value in _priority_key_match(items, SPLIT_TAG_KEYS):
        if isinstance(value, Mapping):
            tags = _split_counts_to_tags(value, sample_count)
            if tags is not None:
                return tags
            continue

        string_values = _parse_string_list(value)
        if len(string_values) == sample_count:
            return np.asarray(string_values, dtype=object)

        numeric = _coerce_small_numeric_array(value)
        if numeric is not None and numeric.ndim == 1 and numeric.size == sample_count:
            return np.asarray([str(item) for item in numeric], dtype=object)

    return None


def _select_metadata_text(mapping: Mapping[Any, Any], keys: Iterable[str], default: str) -> str:
    items = _mapping_items(mapping)
    for _key, value in _priority_key_match(items, keys):
        if isinstance(value, str) and value.strip():
            return value.strip()
    return default


def _axis_candidates(mapping: Mapping[Any, Any], excluded_keys: set[Any] | None = None) -> list[tuple[Any, np.ndarray]]:
    items = _mapping_items(mapping)
    excluded_keys = excluded_keys or set()
    candidates: list[tuple[Any, np.ndarray]] = []
    used: set[Any] = set()

    for key, value in _priority_key_match(items, X_KEYS):
        if key in excluded_keys:
            continue
        axis = _coerce_axis(value)
        if axis is not None:
            candidates.append((key, axis))
            used.add(key)

    for key, value in items:
        if key in used or key in excluded_keys:
            continue
        key_lower = _key_lower(key)
        if key_lower in _RESERVED_KEYS or key_lower.startswith(("train_", "val_", "test_")):
            continue
        axis = _coerce_axis(value)
        if axis is not None:
            candidates.append((key, axis))

    return candidates


def _select_spectra_candidate(mapping: Mapping[Any, Any]) -> tuple[Any | None, Any | None]:
    items = _mapping_items(mapping)
    for key, value in _priority_key_match(items, Y_KEYS):
        spectra = _coerce_spectra_source(value)
        if spectra is not None:
            return key, spectra

    # Y_KEYS are excluded here because we are looking for spectra candidates;
    # everything else that has a known role should be skipped.
    spectra_reserved = _RESERVED_KEYS - frozenset(Y_KEYS)
    for key, value in items:
        key_lower = _key_lower(key)
        if key_lower in spectra_reserved or key_lower.startswith(("train_", "val_", "test_")):
            continue
        spectra = _coerce_spectra_source(value)
        if spectra is None:
            continue
        shape = _shape_of(spectra)
        if shape is None:
            continue
        is_single = int(shape[0] == 1)
        fallbacks.append((is_single, -(shape[0] * shape[1]), key, spectra))

    if not fallbacks:
        return None, None

    # Sort preference: multi-sample tensors over single-row arrays (is_single=0 first),
    # then larger arrays (higher area = more likely to be the primary spectra), then key name.
    fallbacks.sort(key=lambda item: (item[0], item[1], _key_name(item[2])))
    _, _, key, spectra = fallbacks[0]
    return key, spectra


def _align_spectra_with_axis(spectra: Any, axis_candidates: list[tuple[Any, np.ndarray]]) -> tuple[Any, np.ndarray | None]:
    shape = _shape_of(spectra)
    if shape is None:
        return spectra, None
    for _key, axis in axis_candidates:
        if axis.size == shape[1]:
            return spectra, axis
        if axis.size == shape[0]:
            return _transpose_source(spectra), axis
    return spectra, None


def _select_labels(mapping: Mapping[Any, Any], sample_count: int, excluded_keys: set[Any] | None = None) -> tuple[Any | None, np.ndarray | None]:
    items = _mapping_items(mapping)
    excluded_keys = excluded_keys or set()

    for key, value in _priority_key_match(items, LABEL_KEYS):
        if key in excluded_keys:
            continue
        labels = _coerce_labels_array(value, sample_count)
        if labels is not None:
            return key, labels

    fallbacks: list[tuple[int, int, int, Any, np.ndarray]] = []
    for key, value in items:
        if key in excluded_keys:
            continue
        key_lower = _key_lower(key)
        if key_lower.startswith(("train_", "val_", "test_")):
            continue
        labels = _coerce_labels_array(value, sample_count)
        if labels is None:
            continue
        has_hint = int(any(token in key_lower for token in ("label", "param", "target", "feature", "condition")))
        width_penalty = int(labels.shape[1] > 64)
        fallbacks.append((-has_hint, width_penalty, labels.shape[1], key, labels))

    if not fallbacks:
        return None, None

    fallbacks.sort(key=lambda item: (item[0], item[1], item[2], _key_name(item[3])))
    _, _, _, key, labels = fallbacks[0]
    return key, labels


def _select_param_names(mapping: Mapping[Any, Any], count: int) -> list[str]:
    items = _mapping_items(mapping)
    for _key, value in _priority_key_match(items, PARAM_NAMES_KEYS):
        names = _parse_string_list(value)
        if names:
            return _normalize_param_names(names, count)
    return [f"param_{index}" for index in range(count)]


def _select_param_ranges(mapping: Mapping[Any, Any], param_names: list[str], labels: np.ndarray) -> np.ndarray:
    items = _mapping_items(mapping)
    for _key, value in _priority_key_match(items, PARAM_RANGES_KEYS):
        ranges = _parse_param_ranges(value, param_names, labels)
        if ranges is not None:
            return ranges
    return np.column_stack((labels.min(axis=0), labels.max(axis=0)))


def _nearest_sorted_index(axis: np.ndarray, value: float) -> int:
    index = int(np.searchsorted(axis, value))
    if index <= 0:
        return 0
    if index >= axis.size:
        return int(axis.size - 1)
    before = float(axis[index - 1])
    after = float(axis[index])
    if abs(before - value) <= abs(after - value):
        return int(index - 1)
    return index


def _axis_lookup_direction(axis: np.ndarray) -> int:
    if axis.size <= 1:
        return 1
    steps = np.diff(axis)
    if np.all(steps >= 0):
        return 1
    if np.all(steps <= 0):
        return -1
    return 0


def _row_to_numpy(data: Any, index: int) -> np.ndarray:
    if isinstance(data, np.ndarray):
        row = data[index]
    elif _is_torch_tensor(data):
        row = data[index].detach().cpu().numpy()
    else:
        row = np.asarray(data[index])
    return np.asarray(row).reshape(-1)


@dataclass
class SpectrumBlock:
    name: str
    spectra: Any
    point_count_hint: int | None = None
    sample_count: int = field(init=False)
    point_count: int = field(init=False)

    def __post_init__(self) -> None:
        prepared = _coerce_spectra_source(self.spectra, point_count=self.point_count_hint)
        if prepared is None:
            raise ValueError("Invalid spectra block")
        shape = _shape_of(prepared)
        if shape is None:
            raise ValueError("Invalid spectra block shape")
        self.spectra = prepared
        self.sample_count = int(shape[0])
        self.point_count = int(shape[1])


@dataclass
class SpectrumDataset:
    x: np.ndarray
    # NOTE: `spectra` holds the raw user-supplied value after construction so that
    # serialisation round-trips are not surprised.  Internal code must use
    # `_spectra_blocks` (always a list[SpectrumBlock]) — never `self.spectra`.
    spectra: Any
    labels: np.ndarray | None = None
    param_names: list[str] = field(default_factory=list)
    param_ranges: np.ndarray | None = None
    split_tags: np.ndarray | None = None
    name: str = "Dataset"
    x_name: str = "x"
    y_name: str = "signal"
    metadata: dict[str, Any] = field(default_factory=dict)
    _spectra_blocks: list[SpectrumBlock] = field(init=False, repr=False)
    _sample_count: int = field(init=False, repr=False)
    _point_count: int = field(init=False, repr=False)
    _block_offsets: np.ndarray = field(init=False, repr=False)
    _x_lookup_direction: int = field(init=False, repr=False)
    _x_lookup_axis: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        axis = _coerce_axis(self.x)
        if axis is None:
            raise ValueError("x must be a finite numeric 1D axis")

        blocks: list[SpectrumBlock]
        if isinstance(self.spectra, (list, tuple)) and self.spectra and all(isinstance(item, SpectrumBlock) for item in self.spectra):
            blocks = list(self.spectra)
        else:
            blocks = [SpectrumBlock("data", self.spectra, point_count_hint=axis.size)]

        point_count = blocks[0].point_count
        if point_count != axis.size:
            raise ValueError("spectra must be aligned with x")
        for block in blocks[1:]:
            if block.point_count != point_count:
                raise ValueError("all spectra blocks must share the same point count")

        sample_count = sum(block.sample_count for block in blocks)

        labels = None
        param_names: list[str] = []
        param_ranges = None
        if self.labels is not None:
            labels = _coerce_labels_array(self.labels, sample_count=sample_count)
            if labels is None:
                raise ValueError("labels must align with the number of spectra")
            param_names = _normalize_param_names(list(self.param_names), labels.shape[1])
            param_ranges = _parse_param_ranges(self.param_ranges, param_names, labels)
            if param_ranges is None:
                param_ranges = np.column_stack((labels.min(axis=0), labels.max(axis=0)))

        split_tags = None
        if self.split_tags is not None:
            raw_split_tags = np.asarray(self.split_tags, dtype=object).reshape(-1)
            split_tags = np.asarray([str(tag) for tag in raw_split_tags], dtype=object)
            if split_tags.size != sample_count:
                raise ValueError("split_tags must align with the number of spectra")

        offsets = []
        cursor = 0
        for block in blocks:
            offsets.append(cursor)
            cursor += block.sample_count

        self.x = axis
        # self.spectra is intentionally left as-is (the raw user-supplied value).
        # All internal access goes through self._spectra_blocks.
        self.labels = labels
        self.param_names = param_names
        self.param_ranges = param_ranges
        self.split_tags = split_tags
        self.name = self.name or "Dataset"
        self.x_name = (self.x_name or "x").strip()
        self.y_name = (self.y_name or "signal").strip()
        self._spectra_blocks = blocks
        self._sample_count = sample_count
        self._point_count = point_count
        self._block_offsets = np.asarray(offsets, dtype=int)
        self._x_lookup_direction = _axis_lookup_direction(self.x)
        self._x_lookup_axis = self.x if self._x_lookup_direction >= 0 else self.x[::-1]

    @property
    def sample_count(self) -> int:
        return self._sample_count

    @property
    def point_count(self) -> int:
        return self._point_count

    @property
    def parameter_count(self) -> int:
        return 0 if self.labels is None else int(self.labels.shape[1])

    @property
    def spectra_backend(self) -> str:
        return "torch" if any(_is_torch_tensor(block.spectra) for block in self._spectra_blocks) else "numpy"

    def spectrum_at(self, index: int) -> np.ndarray:
        if not 0 <= index < self.sample_count:
            raise IndexError("spectrum index out of range")
        block_index = int(np.searchsorted(self._block_offsets, index, side="right") - 1)
        block = self._spectra_blocks[block_index]
        local_index = index - int(self._block_offsets[block_index])
        return _row_to_numpy(block.spectra, local_index)

    def nearest_x_index(self, value: float) -> int:
        if self.point_count <= 1:
            return 0

        x_value = float(value)
        if self._x_lookup_direction > 0:
            return _nearest_sorted_index(self._x_lookup_axis, x_value)
        if self._x_lookup_direction < 0:
            reverse_index = _nearest_sorted_index(self._x_lookup_axis, x_value)
            return int(self.point_count - 1 - reverse_index)
        return int(np.argmin(np.abs(self.x - x_value)))


def _build_dataset_from_array(data: Any, name: str) -> SpectrumDataset | None:
    spectra = _coerce_spectra_source(data)
    if spectra is None:
        return None
    shape = _shape_of(spectra)
    if shape is None:
        return None
    x = np.arange(shape[1], dtype=float)
    return SpectrumDataset(x=x, spectra=spectra, name=name)


def _build_dataset_from_split_mapping(mapping: Mapping[Any, Any], name: str) -> SpectrumDataset | None:
    items = _mapping_items(mapping)
    axis_candidates = _axis_candidates(mapping)
    spectra_blocks: list[SpectrumBlock] = []
    label_parts: list[np.ndarray] = []
    split_tags: list[str] = []
    labels_complete = True
    labels_width: int | None = None
    axis: np.ndarray | None = None

    spectra_sources: dict[str, Any] = {}
    discovered_splits: list[str] = []
    for split_name, _key, value in _priority_prefixed_key_match(items, Y_KEYS):
        if split_name in spectra_sources:
            continue
        raw = _coerce_spectra_source(value)
        if raw is None:
            continue
        spectra_sources[split_name] = raw
        discovered_splits.append(split_name)

    if not spectra_sources:
        return None

    label_sources: dict[str, Any] = {}
    for split_name, _key, value in _priority_prefixed_key_match(items, LABEL_KEYS):
        if split_name not in spectra_sources or split_name in label_sources:
            continue
        label_sources[split_name] = value

    split_order = [split_name for split_name in SPLIT_NAMES if split_name in spectra_sources]
    split_order.extend(split_name for split_name in discovered_splits if split_name not in split_order)

    for split_name in split_order:
        raw = spectra_sources[split_name]

        aligned, matched_axis = _align_spectra_with_axis(raw, axis_candidates)
        if matched_axis is not None:
            axis = matched_axis
        point_hint = axis.size if axis is not None else None
        block = SpectrumBlock(split_name, aligned, point_count_hint=point_hint)
        spectra_blocks.append(block)
        split_tags.extend([split_name] * block.sample_count)

        label_value = label_sources.get(split_name)
        if label_value is None:
            labels_complete = False
            continue

        labels = _coerce_labels_array(label_value, block.sample_count)
        if labels is None:
            labels_complete = False
            continue
        if labels_width is None:
            labels_width = labels.shape[1]
        elif labels.shape[1] != labels_width:
            labels_complete = False
            continue
        label_parts.append(labels)

    if not spectra_blocks:
        return None

    if axis is None:
        axis = np.arange(spectra_blocks[0].point_count, dtype=float)

    labels = None
    if labels_complete and label_parts and len(label_parts) == len(spectra_blocks):
        labels = np.concatenate(label_parts, axis=0)

    param_names = _select_param_names(mapping, labels.shape[1]) if labels is not None else []
    param_ranges = _select_param_ranges(mapping, param_names, labels) if labels is not None else None
    title = _select_metadata_text(mapping, TITLE_KEYS, name)
    x_name = _select_metadata_text(mapping, X_NAME_KEYS, "x")
    y_name = _select_metadata_text(mapping, Y_NAME_KEYS, "signal")

    metadata = {
        "source": "split_mapping",
        "backend": "torch" if any(_is_torch_tensor(block.spectra) for block in spectra_blocks) else "numpy",
        "split_counts": {split: split_tags.count(split) for split in dict.fromkeys(split_tags)},
    }

    return SpectrumDataset(
        x=axis,
        spectra=spectra_blocks,
        labels=labels,
        param_names=param_names,
        param_ranges=param_ranges,
        split_tags=np.asarray(split_tags, dtype=object),
        name=title,
        x_name=x_name,
        y_name=y_name,
        metadata=metadata,
    )


def _build_dataset_from_mapping(mapping: Mapping[Any, Any], name: str) -> SpectrumDataset | None:
    spectra_key, raw_spectra = _select_spectra_candidate(mapping)
    if raw_spectra is None:
        return None

    axis_candidates = _axis_candidates(mapping, excluded_keys={spectra_key} if spectra_key is not None else set())
    spectra, axis = _align_spectra_with_axis(raw_spectra, axis_candidates)
    shape = _shape_of(spectra)
    if shape is None:
        return None
    if axis is None:
        axis = np.arange(shape[1], dtype=float)

    excluded = {spectra_key} if spectra_key is not None else set()
    labels_key, labels = _select_labels(mapping, shape[0], excluded_keys=excluded)
    if labels is not None and labels.shape[0] != shape[0]:
        labels = None
        labels_key = None

    param_names = _select_param_names(mapping, labels.shape[1]) if labels is not None else []
    param_ranges = _select_param_ranges(mapping, param_names, labels) if labels is not None else None
    split_tags = _extract_split_tags(mapping, shape[0])
    title = _select_metadata_text(mapping, TITLE_KEYS, name)
    x_name = _select_metadata_text(mapping, X_NAME_KEYS, "x")
    y_name = _select_metadata_text(mapping, Y_NAME_KEYS, "signal")

    metadata = {
        "source": "mapping",
        "backend": "torch" if _is_torch_tensor(spectra) else "numpy",
        "spectra_key": _key_name(spectra_key) if spectra_key is not None else None,
        "labels_key": _key_name(labels_key) if labels_key is not None else None,
    }

    return SpectrumDataset(
        x=axis,
        spectra=spectra,
        labels=labels,
        param_names=param_names,
        param_ranges=param_ranges,
        split_tags=split_tags,
        name=title,
        x_name=x_name,
        y_name=y_name,
        metadata=metadata,
    )


def _build_spectrum_dataset(data: Any, name: str) -> SpectrumDataset | None:
    if isinstance(data, SpectrumDataset):
        return data

    dataset = _build_dataset_from_array(data, name)
    if dataset is not None:
        return dataset

    mapping_items = _mapping_items(data)
    if mapping_items:
        mapping = dict(mapping_items)
        dataset = _build_dataset_from_split_mapping(mapping, name)
        if dataset is not None:
            return dataset
        return _build_dataset_from_mapping(mapping, name)

    return None


def find_spectrum_datasets(data: Any, path_prefix: str = "") -> list[tuple[str, SpectrumDataset]]:
    display_name = path_prefix or "Dataset"
    dataset = _build_spectrum_dataset(data, display_name)
    if dataset is not None:
        return [(display_name, dataset)]

    results: list[tuple[str, SpectrumDataset]] = []
    mapping_items = _mapping_items(data)
    if mapping_items:
        for key, value in mapping_items:
            child_prefix = f"{path_prefix}/{_key_name(key)}" if path_prefix else _key_name(key)
            results.extend(find_spectrum_datasets(value, child_prefix))
        return results

    if isinstance(data, (list, tuple)):
        for index, value in enumerate(data):
            child_prefix = f"{path_prefix}[{index}]" if path_prefix else f"[{index}]"
            results.extend(find_spectrum_datasets(value, child_prefix))
    return results


def ensure_spectrum_dataset(data: Any, name: str = "Dataset") -> SpectrumDataset:
    dataset = _build_spectrum_dataset(data, name or "Dataset")
    if dataset is None:
        raise ValueError("Could not detect a spectrum dataset from the provided input")
    return dataset


def load_dataset_candidates(path: str) -> list[tuple[str, SpectrumDataset]]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    ext = file_path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file extension: {ext}. Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    if ext == ".npy":
        loaded = np.load(file_path, allow_pickle=False)
    elif ext in {".pt", ".pth"}:
        loaded = _load_torch_archive_raw(str(file_path))
    else:
        loaded = get_archive_contents(str(file_path))

    candidates = find_spectrum_datasets(loaded)
    if not candidates:
        raise ValueError(
            "No spectrum-like datasets were detected. Expected a 1D/2D signal array or a mapping with x/y and optional labels."
        )

    base_name = file_path.stem
    results: list[tuple[str, SpectrumDataset]] = []
    for display_name, dataset in candidates:
        if display_name == "Dataset":
            final_name = base_name
        else:
            final_name = f"{base_name}/{display_name}"
        dataset.name = final_name
        dataset.metadata["source_path"] = str(file_path)
        results.append((final_name, dataset))
    return results
