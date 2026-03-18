import PhysicalConstant
import torch
import numpy as np

def Cal_Vs_from_Phi(phase_map, dx=1.0, dy=1.0, boundary="fill"):
    """
    Compute superfluid velocity from a 2D phase map using the lattice-gauge method
    
    Parameters
    ----------
    phase_map : torch.Tensor
        2D tensor of phase values in radians.
    dx, dy : float
        Lattice spacing in x and y.
    boundary : {"periodic", "fill"}
        Boundary condition.

    Returns
    -------
    vx, vy, vs, vsmall, vlarge : torch.Tensor
    """
    
    hbar = PhysicalConstant.hbar_unit_Js.value
    m_e  = PhysicalConstant.m_e.value
    K    = hbar / (2 *m_e)

    phi = phase_map.to(torch.float32)
    psi = torch.exp(1j * phi)

    psi_r = torch.roll(psi, shifts=-1, dims=1) # roll along x (row) direction
    psi_l = torch.roll(psi, shifts=+1, dims=1) # roll along x (row) direction
    psi_u = torch.roll(psi, shifts=-1, dims=0) # roll along y (col) direction
    psi_d = torch.roll(psi, shifts=+1, dims=0) # roll along y (col) direction

    dphi_x_r = torch.angle(torch.conj(psi)   * psi_r) / dx
    dphi_x_l = torch.angle(torch.conj(psi_l) * psi)   / dx
    dphi_y_u = torch.angle(torch.conj(psi)   * psi_u) / dy
    dphi_y_d = torch.angle(torch.conj(psi_d) * psi)   / dy

    dphi_dx = 0.5 * (dphi_x_r + dphi_x_l)
    dphi_dy = 0.5 * (dphi_y_u + dphi_y_d)

    if boundary == "fill":
        dphi_dx[ :,  0] = torch.zeros(1)
        dphi_dx[ :, -1] = torch.zeros(1)
        dphi_dy[ 0,  :] = torch.zeros(1)
        dphi_dy[-1,  :] = torch.zeros(1)
        
    elif boundary != "periodic":
        raise ValueError("boundary must be 'periodic' or 'fill'")

    vx = K * dphi_dx
    vy = K * dphi_dy

    return vx, vy


#%% Get the maximal and minimal values of the absolute v
def vl_vs(vx, vy):
    vsmall = torch.minimum(torch.abs(vx), torch.abs(vy))
    vlarge = torch.maximum(torch.abs(vx), torch.abs(vy))
    return vsmall, vlarge



#%% in-file test
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    device = "cuda" if torch.cuda.is_available() else "cpu"

    nx, ny = 256, 256
    dx, dy = 1.0, 1.0

    x = torch.arange(nx, device=device, dtype=torch.float32) * dx
    y = torch.arange(ny, device=device, dtype=torch.float32) * dy
    X, Y = torch.meshgrid(x, y, indexing="xy")

    x0 = x[nx // 2]
    y0 = y[ny // 2]

    # Single-vortex phase map
    phi = torch.atan2(Y - y0, X - x0)

    vx, vy, = Cal_Vs_from_Phi(phi, dx=dx, dy=dy, boundary="fill")

    # Downsample arrows for plotting
    step = 8
    Xq = X[::step, ::step]
    Yq = Y[::step, ::step]
    VXq = vx[::step, ::step]
    VYq = vy[::step, ::step]

    # Normalize arrows for display only
    mag = torch.sqrt(VXq**2 + VYq**2)
    eps = 1e-30
    VXq_plot = VXq / (mag + eps)
    VYq_plot = VYq / (mag + eps)

    mask = torch.isfinite(VXq_plot) & torch.isfinite(VYq_plot)

    # Move to CPU for plotting
    x_np   = x.detach().cpu().numpy()
    y_np   = y.detach().cpu().numpy()
    phi_np = phi.detach().cpu().numpy()

    Xq_np  = Xq[mask].detach().cpu().numpy()
    Yq_np  = Yq[mask].detach().cpu().numpy()
    VXq_np = VXq_plot[mask].detach().cpu().numpy()
    VYq_np = VYq_plot[mask].detach().cpu().numpy()

    x0_np = x0.detach().cpu().item()
    y0_np = y0.detach().cpu().item()

    fig, ax = plt.subplots(figsize=(7, 7))

    im = ax.imshow(phi_np, origin="lower", cmap="twilight",
                   extent=[x_np.min(), x_np.max(), y_np.min(), y_np.max()],
                   vmin=-np.pi,vmax=np.pi,)

    ax.quiver( Xq_np, Yq_np, VXq_np, VYq_np, angles="xy", scale_units="xy", scale=0.25, pivot="mid", color="k")

    ax.plot(x0_np, y0_np, "wo", ms=5, mec="k")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Single vortex: phase and velocity arrows")
    ax.set_aspect("equal")

    cbar = fig.colorbar(im, ax=ax, ticks=[-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    cbar.ax.set_yticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])
    cbar.set_label("phi (rad)")
    plt.tight_layout()

    plt.show()