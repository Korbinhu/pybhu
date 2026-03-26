from dataclasses import dataclass, field

import numpy as np


@dataclass
class ViewerState:
    data: np.ndarray
    current_layer: int = 0
    color_limit_mode: str = "global"
    colormap_name: str = "viridis"
    inverted: bool = False
    origin: str = "lower"
    global_limits: tuple[float, float] = field(init=False)
    per_layer_limits: list[tuple[float, float]] = field(init=False)

    def __post_init__(self):
        if not isinstance(self.data, np.ndarray):
            raise TypeError("data must be a numpy.ndarray")
        if self.data.ndim == 2:
            self.data = self.data[:, :, np.newaxis]
        elif self.data.ndim != 3:
            raise ValueError("data must be a 2D or 3D numpy.ndarray")
        if np.iscomplexobj(self.data):
            raise ValueError("data must be real-valued")
        if 0 in self.data.shape:
            raise ValueError("data must not be empty")
        if np.any(np.isnan(self.data)) or np.any(np.isinf(self.data)):
            raise ValueError("data must contain only finite values")

        self.per_layer_limits = []
        for index in range(self.data.shape[2]):
            layer = self.data[:, :, index]
            self.per_layer_limits.append((float(layer.min()), float(layer.max())))

        self.global_limits = (float(self.data.min()), float(self.data.max()))
        self.set_current_layer(self.current_layer)
        self.set_color_limit_mode(self.color_limit_mode)
        self.set_colormap(self.colormap_name)
        self.set_inverted(self.inverted)
        self.set_origin(self.origin)

    @property
    def layer_count(self) -> int:
        return int(self.data.shape[2])

    def set_current_layer(self, index: int) -> None:
        if not 0 <= index < self.layer_count:
            raise ValueError("initial_layer is out of range")
        self.current_layer = index

    def set_limits(self, vmin: float, vmax: float) -> None:
        if vmin >= vmax:
            raise ValueError("vmin must be less than vmax")
        pair = (float(vmin), float(vmax))
        if self.color_limit_mode == "global":
            self.global_limits = pair
            self.per_layer_limits = [pair for _ in range(self.layer_count)]
        else:
            self.per_layer_limits[self.current_layer] = pair

    def set_color_limit_mode(self, mode: str) -> None:
        if mode not in {"global", "per_layer"}:
            raise ValueError("color_limit_mode must be 'global' or 'per_layer'")
        if mode == self.color_limit_mode:
            return
        if mode == "per_layer":
            self.per_layer_limits = [self.global_limits for _ in range(self.layer_count)]
        else:
            promoted = self.per_layer_limits[self.current_layer]
            self.global_limits = promoted
            self.per_layer_limits = [promoted for _ in range(self.layer_count)]
        self.color_limit_mode = mode

    def visible_limits(self) -> tuple[float, float]:
        if self.color_limit_mode == "global":
            return self.global_limits
        return self.per_layer_limits[self.current_layer]

    def set_colormap(self, name: str) -> None:
        from .colormaps import resolve_colormap

        resolve_colormap(name, self.inverted)
        self.colormap_name = name

    def set_inverted(self, value: bool) -> None:
        self.inverted = bool(value)

    def set_origin(self, value: str) -> None:
        if value not in {"lower", "upper"}:
            raise ValueError("origin must be 'lower' or 'upper'")
        self.origin = value


def normalize_spectrum_axis(spectrum_axis, layer_count: int):
    """Validate and convert a spectrum axis to a 1D float64 array, or return None."""
    if spectrum_axis is None:
        return None
    try:
        axis = np.asarray(spectrum_axis, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if axis.ndim != 1 or axis.size != layer_count or not np.isfinite(axis).all():
        return None
    return axis
