import numpy as np
from scipy.signal import correlate2d

def Cal_Vs_from_Phi(phase_map, dx=1.0, dy=1.0):
    ## Central difference pre-factors
    pre_factor_x = 1 / (2 * dx)  # dx is the pixel length in the x-direction
    pre_factor_y = 1 / (2 * dy)  # dy is the pixel length in the y-direction

    # Kernels defined intuitively (Left to Right, Bottom to Top)
    kx = pre_factor_x * np.array([
        [ 0, 0, 0],
        [-1, 0, 1],
        [ 0, 0, 0],
    ])

    ky = pre_factor_y * np.array([
        [0, -1, 0],
        [0,  0, 0],
        [0,  1, 0],
    ])

    ## Complex order parameter
    psi = np.exp(1j * phase_map)

    # Cross-correlation to preserve spatial alignment (no flipping)
    dpsi_dx = correlate2d(psi, kx, mode="same", boundary="fill")
    dpsi_dy = correlate2d(psi, ky, mode="same", boundary="fill")

    # Superfluid velocity formula: v = Im(psi* \nabla psi)
    vx = np.imag(np.conj(psi) * dpsi_dx)
    vy = np.imag(np.conj(psi) * dpsi_dy)

    # Extended velocity metrics
    vs     = np.sqrt(vx**2 + vy**2)
    vsmall = np.minimum(np.abs(vx), np.abs(vy))
    vlarge = np.maximum(np.abs(vx), np.abs(vy))

    return vx, vy, vs, vsmall, vlarge









