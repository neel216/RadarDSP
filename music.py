import numpy as np

def MUSIC(X, K):
    """
    Implements MUSIC algorithm for direction of arrival estimation.
    
    Parameters:
    X - signal matrix (sensor data matrix)
    K - number of sources/signals to be estimated
    
    Returns:
    Smusic - MUSIC spectrum
    """
    # wavelength:
    lambda_ = 2
    # sensor separation:
    dist = 1
    # angles to scan:
    theta_scan_deg = np.arange(-90, 90.05, 0.05)
    theta_scan = np.deg2rad(theta_scan_deg)
    num_angles = len(theta_scan)
    
    # get dimensions of X:
    N, T = X.shape
    
    # estimate autocorrelation matrix:
    Rx = (1 / T) * (X @ X.conj().T)
    
    # perform eigendecomposition of autocorrelation matrix:
    D, V = np.linalg.eig(Rx)
    # sort eigenvectors in decreasing order of their eigenvalues:
    indices = np.argsort(D)[::-1]
    V_sorted = V[:, indices]
    # extract last N-K eigenvectors (columns):
    V2 = V_sorted[:, K:]
    
    # calculate spectrum:
    Smusic = np.zeros(num_angles)
    for k in range(num_angles):
        # construct a(theta_k):
        a = np.exp(1j * (-2 * np.pi) * (dist / lambda_) * np.arange(N) * np.sin(theta_scan[k]))
        
        Smusic[k] = 1 / np.linalg.norm((V2.conj().T @ a))
    
    return Smusic, theta_scan_deg

# Example usage:
# X = ... (your sensor data matrix)
# K = ... (number of sources you expect)
# Smusic, theta_scan_deg = MUSIC(X, K)

# You can then plot the result with matplotlib, for example:
# import matplotlib.pyplot as plt
# plt.plot(theta_scan_deg, 10 * np.log10(Smusic))
# plt.xlabel('Angle (degrees)')
# plt.ylabel('Spatial Spectrum (dB)')
# plt.title('MUSIC Spectrum')
# plt.show()
