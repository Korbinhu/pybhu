from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bundle import FitBundle, MapDataset, ensure_fit_bundle


@dataclass(slots=True)
class SelectionState:
    i: int
    j: int
    dataset_index: int
    layer_index: int


class FittingViewerState:
    def __init__(self, bundle, selection: SelectionState | None = None):
        self.bundle: FitBundle = ensure_fit_bundle(bundle)
        self.datasets: tuple[MapDataset, ...] = self.bundle.all_map_datasets
        ny, nx = self.bundle.spatial_shape

        if selection is None:
            dataset_index = 0
            layer_index = self.layer_count_for_dataset(dataset_index) // 2
            selection = SelectionState(
                i=ny // 2,
                j=nx // 2,
                dataset_index=dataset_index,
                layer_index=layer_index,
            )

        self.selection = selection
        self._clamp_selection()

    def _clamp_selection(self) -> None:
        ny, nx = self.bundle.spatial_shape
        self.selection.i = min(max(int(self.selection.i), 0), ny - 1)
        self.selection.j = min(max(int(self.selection.j), 0), nx - 1)
        self.selection.dataset_index = min(
            max(int(self.selection.dataset_index), 0),
            len(self.datasets) - 1,
        )
        max_layer = self.layer_count_for_dataset(self.selection.dataset_index) - 1
        self.selection.layer_index = min(max(int(self.selection.layer_index), 0), max_layer)

    @property
    def dataset_names(self) -> list[str]:
        return [dataset.name for dataset in self.datasets]

    @property
    def current_dataset(self) -> MapDataset:
        return self.datasets[self.selection.dataset_index]

    @property
    def position(self) -> tuple[int, int]:
        return self.selection.i, self.selection.j

    def layer_count_for_dataset(self, dataset_index: int | None = None) -> int:
        if dataset_index is None:
            dataset_index = self.selection.dataset_index
        return self.datasets[int(dataset_index)].layer_count

    def current_map(self) -> np.ndarray:
        data = self.current_dataset.data
        if data.ndim == 2:
            return data
        return data[:, :, self.selection.layer_index]

    def current_value(self) -> float:
        map_data = self.current_map()
        return float(map_data[self.selection.i, self.selection.j])

    def raw_spectrum(self) -> np.ndarray:
        return self.bundle.raw_cube[self.selection.i, self.selection.j, :]

    def fit_spectrum(self) -> np.ndarray:
        return self.bundle.fit_cube[self.selection.i, self.selection.j, :]

    def set_dataset_index(self, index: int) -> None:
        self.selection.dataset_index = int(index)
        self._clamp_selection()

    def set_layer_index(self, index: int) -> None:
        self.selection.layer_index = int(index)
        self._clamp_selection()

    def set_position(self, i: int, j: int) -> None:
        self.selection.i = int(i)
        self.selection.j = int(j)
        self._clamp_selection()

    def move_by(self, di: int, dj: int) -> None:
        self.set_position(self.selection.i + int(di), self.selection.j + int(dj))
