import numpy as np
from scipy.optimize import newton

# Try out the evenly spaced spiral of Koay:
# http://www.sciencedirect.com/science/article/pii/S1877750311000615?via%3Dihub


def solve_nspiral(npts):
    """
    Find the number of spirals we need given the total number of points

    Parameters
    ----------
    npts : int

    Returns
    -------
    n : float
        The number of latitude circles
    """
    # Eq 4 from Koay 2011. Initial guess for number of latitude stripes
    guess = np.sqrt(npts*np.pi/8.)
    # Eq 2 from Koay 2011.
    func = lambda nlat: npts/2.*np.sin(np.pi/(4.*nlat)) - nlat
    result = newton(func, guess)
    return result


def k5_i(npts):
    """
    Based on equation 5 from Koay 2011.
    """

    nspiral = solve_nspiral(npts)

    i_vals = np.arange(1, np.round(nspiral)+1)
    theta_i = (i_vals-0.5)*(np.pi/2.)/nspiral
    denom = np.pi/np.sin((np.pi/(4.*np.round(nspiral))))
    result = np.round((2.*np.pi*np.sin(theta_i))/denom*npts)
    # need to override the last value
    result[-1] = npts - np.sum(result[0:-1])
    return result


def npts2thetaphi(npts):
    """
    npts must be even.
    """

    # I'm going to say theta goes from 0-2pi. Because I am a moster who switches convention from the 
    # paper I'm using

    phi = []
    theta = []

    n = np.round(solve_nspiral(npts/2))

    k_i = k5_i(npts/2)

    # ugh, stoopid loop. So unpythonic.
    for i in np.arange(1, n+1, dtype=int):
        temp_phi = (i - 0.5)*(np.pi/2.)/n
        for j in np.arange(1, k_i[i-1]+1, dtype=int):
            theta.append((j - 0.5)*2*np.pi/k_i[i-1])
            phi.append(temp_phi)

    # make the bottom half of the sphere.
    theta = np.array(theta)
    theta = np.concatenate((theta, (theta+np.pi) % (2.*np.pi) ))
    phi = np.array(phi)
    phi = np.concatenate((phi, -phi+np.pi))

    return theta, phi


