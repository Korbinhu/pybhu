import numpy as np
from scipy import interpolate
from scipy.signal import convolve

kB = 0.08617  # Boltzmann constant in meV/K

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def FermiDirac(Bias, Dos, T):
    """
    1D Fermi-Dirac broadening.

    If input is torch.Tensor, convert to numpy for computation,
    then convert outputs back to torch.Tensor.

    Parameters
    ----------
    Bias : 1D numpy array or torch tensor
    Dos  : 1D numpy array or torch tensor
    T    : float
        Temperature in Kelvin

    Returns
    -------
    conv_data : same type as input
    dosr      : same type as input
    """

    input_is_torch = TORCH_AVAILABLE and isinstance(Bias, torch.Tensor)

    if input_is_torch:
        device = Bias.device
        dtype  = Bias.dtype

        Bias_np = Bias.detach().cpu().numpy()
        Dos_np  = Dos.detach().cpu().numpy()
    else:
        Bias_np = np.asarray(Bias)
        Dos_np  = np.asarray(Dos)

    if Bias_np.ndim != 1 or Dos_np.ndim != 1:
        raise ValueError("Bias and Dos must both be 1D arrays.")
    if len(Bias_np) != len(Dos_np):
        raise ValueError("Bias and Dos must have the same length.")

    # define the dos range for convolution, 50 points on each side of the bias range
    dE          = np.mean(np.diff(Bias_np))
    dos_r_start = np.min(Bias_np) - 100 * dE
    dos_r_end   = np.max(Bias_np) + 100 * dE
    dosr = np.arange(dos_r_start, dos_r_end + 1e-12, dE)

    # define the Fermi-Dirac distribution derivative
    beta = 1 / (kB * T)
    fdd  = -(beta * np.exp(Bias_np * beta)) / ((1 + np.exp(Bias_np * beta)) ** 2)
    fdd_norm = fdd / np.sum(fdd)

    conv_data_full = convolve(Dos_np, fdd_norm, mode='same')

    # interpolate back to original Bias points
    f_interp = interpolate.interp1d(Bias_np, conv_data_full, kind='cubic')
    conv_data = f_interp(Bias_np)

    if input_is_torch:
        conv_data = torch.as_tensor(conv_data, dtype=dtype, device=device)
        dosr = torch.as_tensor(dosr, dtype=dtype, device=device)

    return conv_data, dosr


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # test data
    bias = np.linspace(-5, 5, 401)
    dos = (
        0.8 * np.exp(-(bias - 1.2)**2 / (2 * 0.35**2))
        + 1.0 * np.exp(-(bias + 1.0)**2 / (2 * 0.5**2))
        + 0.08
    )
    T = 4.2

    conv_data, dosr = FermiDirac(bias, dos, T)

    print("Bias shape :", bias.shape)
    print("DOS shape  :", dos.shape)
    print("Output shape:", conv_data.shape)
    print("dosr range :", dosr[0], "to", dosr[-1])

    plt.figure(figsize=(8, 5))
    plt.plot(bias, dos, label='Original DOS', linewidth=2)
    plt.plot(bias, conv_data, label=f'Fermi-Dirac broadened (T={T} K)', linewidth=2)
    plt.xlabel('Bias')
    plt.ylabel('DOS')
    plt.title('Fermi-Dirac Broadening Check')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()