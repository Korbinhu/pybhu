from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np


def _coerce_real_array(name: str, data, *, ndim: int) -> np.ndarray:
    array = np.asarray(data)
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must be numeric, got dtype {array.dtype}")
    if array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}D, got {array.ndim}D")
    if 0 in array.shape:
        raise ValueError(f"{name} must not be empty")
    if np.iscomplexobj(array):
        raise ValueError(f"{name} must be real-valued")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return np.array(array, copy=False)


def _coerce_map_array(name: str, data) -> np.ndarray:
    array = np.asarray(data)
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must be numeric, got dtype {array.dtype}")
    if array.ndim not in (2, 3):
        raise ValueError(f"{name} must be 2D or 3D, got {array.ndim}D")
    if 0 in array.shape:
        raise ValueError(f"{name} must not be empty")
    if np.iscomplexobj(array):
        raise ValueError(f"{name} must be real-valued")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return np.array(array, copy=False)


@dataclass(slots=True)
class MapDataset:
    name: str
    data: np.ndarray
    kind: str = "aux"

    def __post_init__(self) -> None:
        self.name = str(self.name).strip()
        if not self.name:
            raise ValueError("MapDataset.name must not be empty")
        if self.kind not in {"raw_map", "fit_map", "aux"}:
            raise ValueError("MapDataset.kind must be 'raw_map', 'fit_map', or 'aux'")
        self.data = _coerce_map_array(f"map dataset {self.name!r}", self.data)

    @property
    def layer_count(self) -> int:
        """1 for 2-D data, shape[2] for 3-D data."""
        if self.data.ndim == 2:
            return 1
        return int(self.data.shape[2])

    @property
    def shape_str(self) -> str:
        """Human-readable shape, e.g. '48×48' or '48×48×256'."""
        return "×".join(str(s) for s in self.data.shape)


@dataclass(slots=True)
class FitBundle:
    x_raw: np.ndarray
    raw_cube: np.ndarray
    x_fit: np.ndarray
    fit_cube: np.ndarray
    map_datasets: tuple[MapDataset, ...] = field(default_factory=tuple)
    x_unit: str = "mV"
    y_unit: str = "a.u."
    raw_name: str = "Raw Data"
    fit_name: str = "Fitted Data"

    def __post_init__(self) -> None:
        self.x_raw = _coerce_real_array("x_raw", self.x_raw, ndim=1)
        self.raw_cube = _coerce_real_array("raw_cube", self.raw_cube, ndim=3)
        self.x_fit = _coerce_real_array("x_fit", self.x_fit, ndim=1)
        self.fit_cube = _coerce_real_array("fit_cube", self.fit_cube, ndim=3)

        if self.raw_cube.shape[:2] != self.fit_cube.shape[:2]:
            raise ValueError("raw_cube and fit_cube must share the same spatial shape")
        if self.raw_cube.shape[2] != self.x_raw.shape[0]:
            raise ValueError("len(x_raw) must match raw_cube.shape[2]")
        if self.fit_cube.shape[2] != self.x_fit.shape[0]:
            raise ValueError("len(x_fit) must match fit_cube.shape[2]")

        normalized_maps: list[MapDataset] = []
        for item in self.map_datasets:
            if isinstance(item, MapDataset):
                dataset = item
            elif isinstance(item, Mapping):
                dataset = MapDataset(**dict(item))
            else:
                raise TypeError(
                    "map_datasets entries must be MapDataset or mapping-like objects"
                )
            if dataset.data.shape[:2] != self.raw_cube.shape[:2]:
                raise ValueError(
                    f"map dataset {dataset.name!r} must match spatial shape"
                    f" {self.raw_cube.shape[:2]}"
                )
            normalized_maps.append(dataset)

        self.map_datasets = tuple(normalized_maps)
        self.x_unit = str(self.x_unit)
        self.y_unit = str(self.y_unit)
        self.raw_name = str(self.raw_name).strip() or "Raw Data"
        self.fit_name = str(self.fit_name).strip() or "Fitted Data"

    @classmethod
    def from_mapping(cls, values: Mapping) -> "FitBundle":
        return cls(
            x_raw=values["x_raw"],
            raw_cube=values["raw_cube"],
            x_fit=values["x_fit"],
            fit_cube=values["fit_cube"],
            map_datasets=tuple(values.get("map_datasets", ())),
            x_unit=values.get("x_unit", "mV"),
            y_unit=values.get("y_unit", "a.u."),
            raw_name=values.get("raw_name", "Raw Data"),
            fit_name=values.get("fit_name", "Fitted Data"),
        )

    @property
    def spatial_shape(self) -> tuple[int, int]:
        return int(self.raw_cube.shape[0]), int(self.raw_cube.shape[1])

    @property
    def all_map_datasets(self) -> tuple[MapDataset, ...]:
        return (
            MapDataset(self.raw_name, self.raw_cube, kind="raw_map"),
            MapDataset(self.fit_name, self.fit_cube, kind="fit_map"),
            *self.map_datasets,
        )


def ensure_fit_bundle(bundle) -> FitBundle:
    if isinstance(bundle, FitBundle):
        return bundle
    if isinstance(bundle, Mapping):
        return FitBundle.from_mapping(bundle)
    raise TypeError("bundle must be a FitBundle or mapping-like object")
