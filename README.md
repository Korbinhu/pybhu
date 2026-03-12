# pybhu

```python
import numpy as np
import pybhu

data = np.random.rand(64, 64, 10)
pybhu.img_viewer(data)
```

## Supported in 0.1.1

- 3D NumPy arrays only
- Layer browsing
- Colormap selection
- Colormap inversion
- Separate colorbar window
- Global and per-layer color limits

## Not yet supported

- `.mat` loading
- Img viewers
- Analysis tools STM
