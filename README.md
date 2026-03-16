# pybhu

`pybhu` currently exposes a desktop scientific image viewer for 2D and 3D array data. This README is intentionally focused on the public `img_viewer` interface.

## Highlights

- Interactive PyQt6 viewer for 2D and 3D image stacks
- Support for `.npy`, `.npz`, and `.pkl` datasets
- Layer navigation, colormap selection, histogram-based contrast control, colorbar display, and FFT analysis

## Repository Structure

```text
.
|-- __init__.py
|-- img_viewer/
|   |-- api.py
|   |-- viewer.py
|   |-- loader.py
|   |-- state.py
|   |-- histogram_dialog.py
|   |-- colorbar_dialog.py
|   |-- fft.py
|   `-- fft_dialog.py
`-- README.md
```

## Public Interface

### `img_viewer`

The main application in this repository is a PyQt6 + Matplotlib scientific viewer for array-based data. It is designed for quick inspection of 2D images and 3D stacks, with built-in tools for:

- browsing layers in a stack
- switching colormaps and inverting palettes
- adjusting intensity limits from a histogram window
- displaying a separate colorbar window
- computing forward or inverse 2D FFTs with selectable window functions

## Requirements

The repository currently ships as source code and does not yet include packaging metadata such as `pyproject.toml`. Install dependencies manually before use.

The examples below assume the package is available as `pybhu`, either because the project folder is named `pybhu` or because you install/package it that way.

Core viewer dependencies:

```bash
pip install numpy scipy matplotlib pyqt6
```

## Quick Start

### Launch the image viewer from a NumPy array

```python
import numpy as np
from pybhu import img_viewer

data = np.random.rand(256, 256, 16)
viewer = img_viewer(data, colormap_name="viridis", initial_layer=0)
viewer.app.exec()
```

If you are already inside an environment that owns the Qt event loop, you may not need to call `viewer.app.exec()`.

### Open data directly from file

```python
from pybhu import img_viewer

viewer = img_viewer("example_stack.npz")
viewer.app.exec()
```

Supported viewer file types:

- `.npy`
- `.npz`
- `.pkl`

For `.npz` and `.pkl`, the loader searches recursively for 2D or 3D arrays and lets you choose among multiple datasets when needed.


