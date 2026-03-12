import numpy as np
import os
import pytest
import subprocess
import sys
from pathlib import Path

from stack_viewer import ImageStackViewer, view_stack


def test_public_symbols_are_importable():
    assert callable(view_stack)
    assert ImageStackViewer.__name__ == "ImageStackViewer"


def test_view_stack_returns_viewer_instance(qapp):
    viewer = view_stack(np.arange(24, dtype=float).reshape(2, 3, 4))
    assert viewer.__class__.__name__ == "ImageStackViewer"


def test_initial_layer_option_is_applied(qapp):
    viewer = view_stack(np.arange(24, dtype=float).reshape(2, 3, 4), initial_layer=2)
    assert viewer.state.current_layer == 2


def test_invalid_initial_layer_raises_value_error():
    with pytest.raises(ValueError):
        view_stack(np.arange(24, dtype=float).reshape(2, 3, 4), initial_layer=10)


def test_initial_colormap_option_is_applied(qapp):
    viewer = view_stack(np.arange(24, dtype=float).reshape(2, 3, 4), colormap="plasma")
    assert viewer.state.colormap_name == "plasma"


def test_initial_color_limit_mode_option_is_applied(qapp):
    viewer = view_stack(
        np.arange(24, dtype=float).reshape(2, 3, 4),
        color_limit_mode="per_layer",
    )
    assert viewer.state.color_limit_mode == "per_layer"


def test_view_stack_creates_qapplication_in_plain_python_process():
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    env["QT_QPA_PLATFORM"] = "offscreen"
    code = (
        "import numpy as np; "
        "from stack_viewer import view_stack; "
        "viewer = view_stack(np.arange(24, dtype=float).reshape(2, 3, 4)); "
        "print(type(viewer).__name__)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ImageStackViewer"
