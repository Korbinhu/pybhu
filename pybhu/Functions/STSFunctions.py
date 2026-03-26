import numpy as np
from scipy import interpolate

#%% The function to compute STS convolution, consider the tip is a functional tip
def STS_Conv(sts1,sts2):
    '''
    Parameters
    ----------
    sts1 : numpy array
        the convolve kernels 1, should be odd length, if not will interpolate
    sts2 : numpy array
        the convolve kernels 2,should be odd length, if not will interpolate
    note : the length of sts1 and sts2 should be equal here
    Returns
    the convolution of the sts1 and sts2 based on the priciple of STM 
    '''
    N = len(sts1)
    if N%2 == 0:
        f1 = interpolate.interp1d(np.linspace(0,N,N),sts1, kind='cubic')
        f2 = interpolate.interp1d(np.linspace(0,N,N),sts2, kind='cubic')
        xint = np.linspace(0,N,N+1)
        sts1 = f1(xint)
        sts2 = f2(xint)
        
    sts_conved = np.zeros_like(sts1)
    M = int(N/2)
    
    for i in range(M):
        a = np.sum(sts1[M:M+i+1]*sts2[M-i:M+1])
        sts_conved[M+i+1] = a
    for j in range(M):
        b = np.sum(sts2[M:M+j+1]*sts1[M-j:M+1])
        sts_conved[M-j-1] = -b
        
    sts_conved = np.gradient(sts_conved)
    if N%2 == 0:
        fconv = interpolate.interp1d(np.linspace(0,N+1,N+1),sts_conved, kind='cubic')
        xconv = np.linspace(0,N+1,N)
        sts_conved = fconv(xconv)
        
    return sts_conved

#%%








