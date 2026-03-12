from stack_viewer.colormaps import available_colormaps, resolve_colormap


def test_available_colormaps_contains_expected_defaults():
    names = available_colormaps()
    assert "viridis" in names
    assert "plasma" in names


def test_resolve_colormap_adds_reverse_suffix_when_inverted():
    assert resolve_colormap("viridis", inverted=False) == "viridis"
    assert resolve_colormap("viridis", inverted=True) == "viridis_r"
