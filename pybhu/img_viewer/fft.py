import numpy as np
import scipy.signal.windows as windows

def create_window_2d(shape, window_type):
    """Create a 2D window for filtering prior to FFT."""
    nr, nc = shape
    if window_type == 'none':
        return np.ones((nr, nc))
    elif window_type == 'sine':
        x = np.sin(np.linspace(0, np.pi, nr))
        y = np.sin(np.linspace(0, np.pi, nc))
        return np.outer(x, y)
    elif window_type == 'kaiser':
        w_r = windows.kaiser(nr, beta=6)
        w_c = windows.kaiser(nc, beta=6)
        return np.outer(w_r, w_c)
    elif window_type == 'gauss':
        # MATLAB's gausswin uses alpha=2.5 as default, std = (N-1)/(2*alpha)
        std_r = (nr - 1) / 5.0 if nr > 1 else 1
        std_c = (nc - 1) / 5.0 if nc > 1 else 1
        w_r = windows.gaussian(nr, std=max(std_r, 1e-8))
        w_c = windows.gaussian(nc, std=max(std_c, 1e-8))
        return np.outer(w_r, w_c)
    elif window_type == 'blackmanharris':
        w_r = windows.blackmanharris(nr)
        w_c = windows.blackmanharris(nc)
        return np.outer(w_r, w_c)
    else:
        return np.ones((nr, nc))

def apply_fft(data, window_type='none', output_type='amplitude', direction='ft'):
    """
    Applies 2D Fourier Transform iteratively across the 3D data stack.
    data: shape (nr, nc, nz)
    window_type: 'none', 'sine', 'kaiser', 'gauss', 'blackmanharris'
    output_type: 'amplitude', 'phase', 'real', 'imaginary', 'complex'
    direction: 'ft' (Forward), 'ift' (Inverse)
    """
    if data.ndim == 2:
        data = data[:, :, np.newaxis]
        
    nr, nc, nz = data.shape
    z = create_window_2d((nr, nc), window_type)
    
    tmp_data = np.nan_to_num(data).astype(complex)
    if direction == 'ft':
        tmp_data = tmp_data / (nr * nc)
        
    out_data = np.zeros_like(tmp_data, dtype=complex)
    
    for k in range(nz):
        layer = tmp_data[:, :, k] * z
        if direction == 'ft':
            out_data[:, :, k] = np.fft.fftshift(np.fft.fft2(layer))
        else:
            out_data[:, :, k] = np.fft.ifft2(np.fft.ifftshift(layer))
            
    if output_type == 'real':
        return np.real(out_data)
    elif output_type == 'imaginary':
        return np.imag(out_data)
    elif output_type == 'phase':
        return np.angle(out_data)
    elif output_type == 'amplitude':
        return np.abs(out_data)
    elif output_type == 'complex':
        return out_data
    else:
        return np.abs(out_data)
