from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bundle import FitBundle, MapDataset, ensure_fit_bundle

# Maps role names to (dataset_attr, layer_attr) on SelectionState
ROLE_ATTRS = {
    "slot_a": ("slot_a_dataset_index", "slot_a_layer_index"),
    "slot_b": ("slot_b_dataset_index", "slot_b_layer_index"),
    "slot_c": ("slot_c_dataset_index", "slot_c_layer_index"),
    "slot_d": ("slot_d_dataset_index", "slot_d_layer_index"),
}

# Default dataset kind preferred for each slot's initial selection
ROLE_PREFERRED_KINDS = {
    "slot_a": "raw_map",
    "slot_b": "fit_map",
    "slot_c": "aux",
    "slot_d": "aux",
}


@dataclass(slots=True, init=False)
class SelectionState:
    i: int
    j: int
    slot_a_dataset_index: int
    slot_a_layer_index: int
    slot_b_dataset_index: int
    slot_b_layer_index: int
    slot_c_dataset_index: int
    slot_c_layer_index: int
    slot_d_dataset_index: int
    slot_d_layer_index: int

    def __init__(
        self,
        i: int,
        j: int,
        slot_a_dataset_index: int = 0,
        slot_a_layer_index: int = 0,
        slot_b_dataset_index: int | None = None,
        slot_b_layer_index: int = 0,
        slot_c_dataset_index: int | None = None,
        slot_c_layer_index: int = 0,
        slot_d_dataset_index: int | None = None,
        slot_d_layer_index: int = 0,
    ) -> None:
        self.i = int(i)
        self.j = int(j)
        self.slot_a_dataset_index = int(slot_a_dataset_index)
        self.slot_a_layer_index = int(slot_a_layer_index)
        self.slot_b_dataset_index = int(
            self.slot_a_dataset_index if slot_b_dataset_index is None else slot_b_dataset_index
        )
        self.slot_b_layer_index = int(slot_b_layer_index)
        self.slot_c_dataset_index = int(
            self.slot_a_dataset_index if slot_c_dataset_index is None else slot_c_dataset_index
        )
        self.slot_c_layer_index = int(slot_c_layer_index)
        self.slot_d_dataset_index = int(
            self.slot_c_dataset_index if slot_d_dataset_index is None else slot_d_dataset_index
        )
        self.slot_d_layer_index = int(slot_d_layer_index)


class FittingViewerState:
    def __init__(self, bundle, selection: SelectionState | None = None):
        self.bundle: FitBundle = ensure_fit_bundle(bundle)
        self.datasets: tuple[MapDataset, ...] = self.bundle.all_map_datasets
        ny, nx = self.bundle.spatial_shape

        if selection is None:
            defaults = {role: self._default_dataset_index(role) for role in ROLE_ATTRS}
            selection = SelectionState(
                i=ny // 2,
                j=nx // 2,
                slot_a_dataset_index=defaults["slot_a"],
                slot_a_layer_index=self._default_layer_index(defaults["slot_a"]),
                slot_b_dataset_index=defaults["slot_b"],
                slot_b_layer_index=self._default_layer_index(defaults["slot_b"]),
                slot_c_dataset_index=defaults["slot_c"],
                slot_c_layer_index=self._default_layer_index(defaults["slot_c"]),
                slot_d_dataset_index=self._default_secondary_param_index(defaults["slot_c"]),
                slot_d_layer_index=self._default_layer_index(
                    self._default_secondary_param_index(defaults["slot_c"])
                ),
            )

        self.selection = selection
        self.spectrum_points: list[tuple[int, int]] = []
        self._clamp_selection()

    # ------------------------------------------------------------------ #
    #  Internal helpers
    # ------------------------------------------------------------------ #

    def _selection_attrs(self, role: str) -> tuple[str, str]:
        if role not in ROLE_ATTRS:
            raise KeyError(f"Unknown role {role!r}. Expected one of {list(ROLE_ATTRS)}")
        return ROLE_ATTRS[role]

    def available_dataset_indices_for_role(self, role: str) -> list[int]:
        """All slots see all datasets — no filtering by kind or spectral length."""
        return list(range(len(self.datasets)))

    def _default_dataset_index(self, role: str) -> int:
        preferred_kind = ROLE_PREFERRED_KINDS[role]
        for index, dataset in enumerate(self.datasets):
            if dataset.kind == preferred_kind:
                return index
        return 0

    def _default_secondary_param_index(self, primary_index: int) -> int:
        """Pick an aux dataset different from the one already used by slot_c."""
        for index, dataset in enumerate(self.datasets):
            if index != primary_index and dataset.kind == "aux":
                return index
        for index in range(len(self.datasets)):
            if index != primary_index:
                return index
        return primary_index

    def _default_layer_index(self, dataset_index: int) -> int:
        return self.layer_count_for_dataset(dataset_index) // 2

    def _clamp_selection(self) -> None:
        ny, nx = self.bundle.spatial_shape
        self.selection.i = min(max(int(self.selection.i), 0), ny - 1)
        self.selection.j = min(max(int(self.selection.j), 0), nx - 1)
        n = len(self.datasets)
        for role in ROLE_ATTRS:
            dataset_attr, layer_attr = self._selection_attrs(role)
            current_index = min(max(int(getattr(self.selection, dataset_attr)), 0), n - 1)
            setattr(self.selection, dataset_attr, current_index)
            max_layer = self.layer_count_for_dataset(current_index) - 1
            current_layer = min(max(int(getattr(self.selection, layer_attr)), 0), max_layer)
            setattr(self.selection, layer_attr, current_layer)

    # ------------------------------------------------------------------ #
    #  Read-only accessors
    # ------------------------------------------------------------------ #

    @property
    def dataset_names(self) -> list[str]:
        return [ds.name for ds in self.datasets]

    @property
    def position(self) -> tuple[int, int]:
        return self.selection.i, self.selection.j

    def dataset_index_for_role(self, role: str) -> int:
        dataset_attr, _ = self._selection_attrs(role)
        return int(getattr(self.selection, dataset_attr))

    def layer_index_for_role(self, role: str) -> int:
        _, layer_attr = self._selection_attrs(role)
        return int(getattr(self.selection, layer_attr))

    def dataset_for_role(self, role: str) -> MapDataset:
        return self.datasets[self.dataset_index_for_role(role)]

    def layer_count_for_dataset(self, dataset_index: int) -> int:
        return self.datasets[int(dataset_index)].layer_count

    def layer_count_for_role(self, role: str) -> int:
        return self.layer_count_for_dataset(self.dataset_index_for_role(role))

    def map_for_role(self, role: str) -> np.ndarray:
        """
        Return the 2-D spatial map for *role* at the currently selected layer.

        Works uniformly for all data shapes:
          - 2-D [nx, ny]        → returned as-is
          - 3-D [nx, ny, l]     → sliced as [:, :, l_i]
        """
        data = self.dataset_for_role(role).data
        if data.ndim == 2:
            return data
        return data[:, :, self.layer_index_for_role(role)]

    def value_for_role(self, role: str) -> float:
        return float(self.map_for_role(role)[self.selection.i, self.selection.j])

    def x_axis_for_role(self, role: str) -> np.ndarray:
        """
        Return the x-axis that matches the selected dataset.
        Fit-kind datasets use x_fit; everything else uses x_raw.
        """
        kind = self.dataset_for_role(role).kind
        return self.bundle.x_fit if kind == "fit_map" else self.bundle.x_raw

    def spectrum_for_role(self, role: str) -> np.ndarray | None:
        """Return the full 1-D spectrum at (i, j) if the dataset is 3-D, else None."""
        dataset = self.dataset_for_role(role)
        if dataset.data.ndim != 3:
            return None
        return dataset.data[self.selection.i, self.selection.j, :]

    def raw_spectrum(self) -> np.ndarray:
        return self.bundle.raw_cube[self.selection.i, self.selection.j, :]

    def fit_spectrum(self) -> np.ndarray:
        return self.bundle.fit_cube[self.selection.i, self.selection.j, :]

    # ------------------------------------------------------------------ #
    #  Mutation
    # ------------------------------------------------------------------ #

    def set_role_dataset_index(self, role: str, index: int) -> None:
        dataset_attr, _ = self._selection_attrs(role)
        setattr(self.selection, dataset_attr, int(index))
        self._clamp_selection()

    def set_role_layer_index(self, role: str, index: int) -> None:
        _, layer_attr = self._selection_attrs(role)
        setattr(self.selection, layer_attr, int(index))
        self._clamp_selection()

    def set_position(self, i: int, j: int) -> None:
        self.selection.i = int(i)
        self.selection.j = int(j)
        self._clamp_selection()

    def move_by(self, di: int, dj: int) -> None:
        self.set_position(self.selection.i + int(di), self.selection.j + int(dj))

    # ------------------------------------------------------------------ #
    #  Multi-point spectrum positions
    # ------------------------------------------------------------------ #

    def add_spectrum_point(self, i: int, j: int) -> None:
        ny, nx = self.bundle.spatial_shape
        i, j = min(max(i, 0), ny - 1), min(max(j, 0), nx - 1)
        if (i, j) not in self.spectrum_points:
            self.spectrum_points.append((i, j))

    def remove_spectrum_point(self, index: int) -> None:
        if 0 <= index < len(self.spectrum_points):
            self.spectrum_points.pop(index)

    def clear_spectrum_points(self) -> None:
        self.spectrum_points.clear()

    def move_layer_by(self, delta: int) -> None:
        """Advance the layer index on every slot that has more than one layer."""
        for role in ROLE_ATTRS:
            if self.layer_count_for_role(role) > 1:
                new_idx = self.layer_index_for_role(role) + int(delta)
                self.set_role_layer_index(role, new_idx)
