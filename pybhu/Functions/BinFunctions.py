import numpy as np
from scipy.signal import correlate2d
from . import PhysicalConstant
import scipy.io as sio

##
def load_data_from_mat(mat_file, symmetric = 'no', key=None):
    """
    x : array, the .e field from the .mat file
    y : array, the .map field from the .mat file
    """
    mat = sio.loadmat(mat_file, struct_as_record=False, squeeze_me=True)

    if key is None:
        key = next(k for k in mat.keys() if not k.startswith("__"))

    D = mat[key]

    x = np.asarray(D.e).squeeze()   # remove singleton dimensions if present
    y = np.asarray(D.map).squeeze() # remove singleton dimensions if present

    if symmetric in ['yes']:
        y = y/2 + y[:,:,::-1]/2 

    return x, y

#%% compute the correlation coefficient of two array
def CorrCoef(a, b):
    a = np.ravel(a).astype(np.float64) - np.mean(a)
    b = np.ravel(b).astype(np.float64) - np.mean(b)
    return float(np.sum(a * b) / (np.sqrt(np.sum(a * a) * np.sum(b * b))))

#%% compute the R2 for fitting analysis
def Compute_R2(fitted_data, raw_data, axis=-1):

    fitted_data = np.asarray(fitted_data)
    raw_data    = np.asarray(raw_data)

    if fitted_data.shape != raw_data.shape:
        raise ValueError("Input shapes must match")

    # Residual sum of squares
    ss_res = np.sum((raw_data - fitted_data) ** 2, axis=axis)

    # Total variance
    mean_raw = np.mean(raw_data, axis=axis, keepdims=True)
    ss_tot   = np.sum((raw_data - mean_raw) ** 2, axis=axis)

    # R² calculation
    r2 = 1 - (ss_res / ss_tot)

    return r2


