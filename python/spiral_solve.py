import numpy as np
from scipy.special import ellipe, ellipk, ellipeinc


def spiral_on_sphere(n, ttol=1e-10, niter=20):
    """
    Find a solution for drawing a spiral of evenly spaced points on a sphere.

    Parameters
    ----------
    n : int
        How many points to put on a sphere

    Returns
    -------
    ra, dec: floats
        The RA and Dec for the points. (radians)

    Based on:
    Koay, Cheng Guan. “Analytically Exact Spiral Scheme for
    Generating Uniformly Distributed Points on the Unit Sphere.”
    Journal of computational science 2.1 (2011): 88–91. PMC. Web. 1 June 2018.
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3079956/
    """

    j = np.arange(n) + 1
    # Initial guess
    m = np.sqrt(n*np.pi)

    # iterate to solve for m
    for i in range(niter):
        m_old = m + 0
        m = k_m(m, n)
    # if we haven't reached specified tolerance
    while np.abs(m_old-m) > ttol:
        m_old = m + 0
        m = k_m(m, n)

    theta = np.arccos(1.-(2*j-1)/n)
    for i in range(niter):
        theta_old = theta + 0
        theta = k_theta(theta, m, j)
    while np.max(np.abs(theta-theta_old)) > ttol:
        theta_old = theta + 0
        theta = k_theta(theta, m, j)

    ra = m*theta % (2.*np.pi)
    dec = np.pi/2. - theta
    return ra, dec


def k_m(m, n):
    """
    Appendix A

    Parameters
    ----------
    m : float
        The value to iterate
    """
    numerator = m*np.pi*n
    numerator *= (2.*ellipe(-m**2) - ellipk(-m**2))
    denom = n*np.pi*ellipe(-m**2)
    denom -= n*np.pi*ellipk(-m**2)
    denom += m*ellipe(-m**2)**2
    result = numerator/denom
    return result


def S_length(theta, m):
    """
    Where theta is an array
    """
    case1 = np.where((theta > np.pi/2.) & (theta <= np.pi))
    case2 = np.where((theta >= 0.) & (theta <= np.pi/2.))

    result = theta*0
    result[case1] = 2.*ellipe(-m**2) - ellipeinc(np.pi-theta[case1], -m**2)
    result[case2] = ellipeinc(theta[case2], -m**2)
    return result


def k_theta(theta, m, j):
    """
    Parameters
    """
    result = theta+0
    numerator = (2.*j-1)*np.pi - m*S_length(theta, m)
    denom = m*np.sqrt(1.+m**2*np.sin(theta)**2)
    result += numerator/denom
    # Make sure we don't wrap around in crazy ways
    result = result % (np.pi)
    return result