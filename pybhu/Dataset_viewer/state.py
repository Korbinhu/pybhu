from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np

from .loader import SpectrumDataset


@dataclass
class DatasetViewerState:
    dataset: SpectrumDataset
    current_index: int = 0
    split_filter: str = "all"
    y_min: float | None = None
    target_parameters: np.ndarray | None = None
    match_residuals: np.ndarray | None = None
    pool_indices: np.ndarray = field(init=False, repr=False)
    labels_norm: np.ndarray | None = field(init=False, repr=False)
    label_offsets: np.ndarray | None = field(init=False, repr=False)
    label_span: np.ndarray | None = field(init=False, repr=False)
    current_pool_position: int = field(init=False, default=0)
    spectrum_cache_size: int = 8
    _spectrum_cache: OrderedDict[int, np.ndarray] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.dataset, SpectrumDataset):
            raise TypeError("dataset must be a SpectrumDataset")

        self._spectrum_cache = OrderedDict()

        if self.dataset.labels is None:
            self.labels_norm = None
            self.label_offsets = None
            self.label_span = None
        else:
            ranges = np.asarray(self.dataset.param_ranges, dtype=float)
            offsets = ranges[:, 0]
            span = np.clip(ranges[:, 1] - ranges[:, 0], 1e-12, None)
            self.label_offsets = offsets
            self.label_span = span
            self.labels_norm = (self.dataset.labels - offsets) / span

        self.set_split_filter(self.split_filter)
        self.jump_to_global_index(self.current_index)
        self.sync_target_to_current()

    @property
    def global_index(self) -> int:
        return int(self.pool_indices[self.current_pool_position])

    @property
    def visible_count(self) -> int:
        return int(self.pool_indices.size)

    def available_split_filters(self) -> list[str]:
        if self.dataset.split_tags is None:
            return ["all"]

        ordered = ["all"]
        seen = set()
        for tag in self.dataset.split_tags:
            tag_str = str(tag)
            if tag_str not in seen:
                ordered.append(tag_str)
                seen.add(tag_str)
        return ordered

    def _pool_for_filter(self, split_filter: str) -> np.ndarray:
        if split_filter == "all" or self.dataset.split_tags is None:
            return np.arange(self.dataset.sample_count, dtype=int)

        indices = np.flatnonzero(np.asarray(self.dataset.split_tags, dtype=object) == split_filter)
        if indices.size == 0:
            raise ValueError(f"Unknown split filter: {split_filter}")
        return indices.astype(int)

    def _nearest_pool_position(self, global_index: int) -> int:
        if self.pool_indices.size == 0:
            return 0
        pool = self.pool_indices
        loc = int(np.searchsorted(pool, global_index))
        if loc >= pool.size:
            return int(pool.size - 1)
        if loc == 0:
            return 0
        before = pool[loc - 1]
        after = pool[loc]
        if abs(before - global_index) <= abs(after - global_index):
            return int(loc - 1)
        return int(loc)

    def set_split_filter(self, split_filter: str) -> None:
        # During __post_init__ pool_indices does not exist yet, so we cannot call
        # self.global_index (which reads it).  Use an explicit guard instead of
        # relying on AttributeError propagating up through the property.
        if hasattr(self, "pool_indices"):
            current_global = self.global_index
        else:
            current_global = int(np.clip(self.current_index, 0, self.dataset.sample_count - 1))
        self.pool_indices = self._pool_for_filter(split_filter)
        self.split_filter = split_filter
        self.current_pool_position = self._nearest_pool_position(current_global)

    def set_current_pool_position(self, position: int) -> None:
        if self.pool_indices.size == 0:
            raise ValueError("No visible spectra in the current filter")
        clipped = int(np.clip(position, 0, self.pool_indices.size - 1))
        self.current_pool_position = clipped

    def move(self, delta: int) -> None:
        if self.pool_indices.size == 0:
            return
        size = self.pool_indices.size
        self.current_pool_position = (self.current_pool_position + delta) % size

    def jump_to_global_index(self, global_index: int) -> None:
        if self.dataset.sample_count == 0:
            self.current_pool_position = 0
            return
        index = int(np.clip(global_index, 0, self.dataset.sample_count - 1))
        self.current_pool_position = self._nearest_pool_position(index)

    def randomize_current(self) -> None:
        if self.pool_indices.size == 0:
            return
        self.current_pool_position = int(np.random.randint(self.pool_indices.size))

    def spectrum_at(self, global_index: int) -> np.ndarray:
        cached = self._spectrum_cache.get(global_index)
        if cached is not None:
            self._spectrum_cache.move_to_end(global_index)
            return cached

        spectrum = self.dataset.spectrum_at(global_index)
        self._spectrum_cache[global_index] = spectrum
        if len(self._spectrum_cache) > self.spectrum_cache_size:
            self._spectrum_cache.popitem(last=False)
        return spectrum

    def current_spectrum(self) -> np.ndarray:
        return self.spectrum_at(self.global_index)

    def warm_visible_neighbors(self, radius: int = 1) -> None:
        if radius <= 0 or self.pool_indices.size <= 1:
            return

        start = max(0, self.current_pool_position - radius)
        stop = min(self.pool_indices.size - 1, self.current_pool_position + radius)
        for pool_position in range(start, stop + 1):
            self.spectrum_at(int(self.pool_indices[pool_position]))

    def current_labels(self) -> np.ndarray | None:
        if self.dataset.labels is None:
            return None
        return self.dataset.labels[self.global_index]

    def current_split_tag(self) -> str:
        if self.dataset.split_tags is None:
            return "all"
        return str(self.dataset.split_tags[self.global_index])

    def sync_target_to_current(self) -> None:
        labels = self.current_labels()
        if labels is None:
            self.target_parameters = None
            self.match_residuals = None
            return
        self.target_parameters = np.asarray(labels, dtype=float)
        self.match_residuals = np.zeros_like(self.target_parameters)

    def find_nearest_in_pool(self, target: np.ndarray) -> int:
        if self.labels_norm is None or self.label_offsets is None or self.label_span is None:
            raise ValueError("This dataset does not have parameter labels")
        normalized_target = (np.asarray(target, dtype=float) - self.label_offsets) / self.label_span
        distances = np.sum((self.labels_norm[self.pool_indices] - normalized_target) ** 2, axis=1)
        return int(np.argmin(distances))

    def set_parameter_target(self, target: np.ndarray) -> None:
        if self.dataset.labels is None:
            raise ValueError("This dataset does not have parameter labels")
        target = np.asarray(target, dtype=float)
        if target.shape != (self.dataset.parameter_count,):
            raise ValueError("Parameter target shape does not match dataset labels")
        self.target_parameters = target
        self.current_pool_position = self.find_nearest_in_pool(target)
        self.match_residuals = np.abs(self.current_labels() - target)
