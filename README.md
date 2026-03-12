# stack-viewer

```python
import numpy as np
from stack_viewer import view_stack

data = np.random.rand(64, 64, 10)
view_stack(data)
```

## Supported in 0.1.0

- 3D NumPy arrays only
- Layer browsing
- Colormap selection
- Colormap inversion
- Separate colorbar window
- Global and per-layer color limits

## Not yet supported

- `.mat` loading
- Spectrum viewers
- Analysis tools from the MATLAB application
