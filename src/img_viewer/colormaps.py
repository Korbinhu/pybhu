SUPPORTED_COLORMAPS = [
    "viridis",
    "plasma",
    "inferno",
    "magma",
    "cividis",
    "gray",
    "turbo",
]


def available_colormaps() -> list[str]:
    return list(SUPPORTED_COLORMAPS)


def resolve_colormap(name: str, inverted: bool = False) -> str:
    if name not in SUPPORTED_COLORMAPS:
        raise ValueError(f"unsupported colormap: {name}")
    return f"{name}_r" if inverted else name
