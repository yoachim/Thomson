import numpy as np
from scipy.optimize import minimize
from lsst.sims.utils import _angularSeparation
from scipy.optimize import OptimizeResult

def thetaphi2xyz(theta, phi):
    x = np.sin(phi)*np.cos(theta)
    y = np.sin(phi)*np.sin(theta)
    z = np.cos(phi)
    return x, y, z


def xyz2thetaphi(x, y, z):
    phi = np.arccos(z)
    theta = np.arctan2(y, x)
    return theta, phi


def elec_potential(x0):
    """
    Compute the potential energy for electrons on a sphere

    Parameters
    ----------
    x0 : array
       First half of x0 are theta values, second half phi

    Returns
    -------
    Potential energy
    """

    theta = x0[0:x0.size/2]
    phi = x0[x0.size/2:]

    x, y, z = thetaphi2xyz(theta, phi)
    # Distance squared
    dsq = 0.

    indices = np.triu_indices(x.size, k=1)

    for coord in [x, y, z]:
        coord_i = np.tile(coord, (coord.size, 1))
        coord_j = coord_i.T
        d = (coord_i[indices]-coord_j[indices])**2
        dsq += d

    U = np.sum(1./np.sqrt(dsq))
    return U


def x0_split(x0):
    size = x0.size
    x = x0[0:int(size/3)]
    y = x0[int(size/3):int(2*size/3)]
    z = x0[int(2*size/3):]

    return x, y, z


def elec_potential_xyz(x0):
    """
    same as elec_potential, but pass in x,y,z coords
    """

    x, y, z = x0_split(x0)

    # Enforce on a sphere?
    r_sq = x**2 + y**2 + z**2
    r = np.sqrt(r_sq)
    x /= r
    y /= r
    z /= r


    # Distance squared
    dsq = 0.

    indices = np.triu_indices(x.size, k=1)

    for coord in [x, y, z]:
        coord_i = np.tile(coord, (coord.size, 1))
        coord_j = coord_i.T
        d = (coord_i[indices]-coord_j[indices])**2
        dsq += d

    U = np.sum(1./np.sqrt(dsq))
    return U


def potential_single(coord0, x, y, z):
    """
    Find the potential contribution from a single point.
    """

    x0 = coord0[0]
    y0 = coord0[1]
    z0 = coord0[2]
    # Enforce point has to be on a sphere
    rsq = x0**2+y0**2 + z0**2
    r = np.sqrt(rsq)
    x0 = x0/r
    y0 = y0/r
    z0 = z0/r

    dsq = (x-x0)**2+(y-y0)**2+(z-z0)**2
    U = np.sum(1./np.sqrt(dsq))
    return U


class elec_sphere(object):
    def __init__(self, x0, y0, z0, x, y, z, fx, fy, fz):
        """
        find the potential by moving around a single electron
        """

        self.x0 = x0
        self.y0 = y0
        self.z0 = z0

        self.x = x
        self.y = y
        self.z = z

        self.fx = fx
        self.fy = fy
        self.fz = fz

    def move_point(self, a, fraction=1.):
        x = self.x + a*self.fx*fraction
        y = self.y + a*self.fy*fraction
        z = self.z + a*self.fz*fraction

        # Force back to the sphere
        d = np.sqrt(x**2 + y**2 + z**2)
        x /= d
        y /= d
        z /= d

        return x, y, z

    def potential(self, a):
        """
        Calculate the potential moving the given point magnitude a
        """
        x, y, z = self.move_point(a)
        dsq = (x-self.x0)**2 + (y-self.y0)**2 + (z-self.z0)**2
        U = np.sum(1./np.sqrt(dsq))
        return U


def minimize_single(x0, y0, z0, index, fraction=1.):
    """
    minimize the potential by moving a single electron

    # XXX--should this be vectorized? get the force vector for all the electrons simultaneously?
    Then fit the amplitude for everything simultaneously? This might be easier to parallelize.
    """

    npts = x0.size
    sq_deg_on_sky = 360.**2/np.pi

    # How many square degrees per point
    res_sq_deg = sq_deg_on_sky/npts

    # Expected distance between points
    typical_dist_deg = np.sqrt(res_sq_deg)

    # Dist between points. Shift size should be small compared to this.
    typical_dist = np.radians(typical_dist_deg)

    all_ind = np.arange(x0.size)
    good = np.where(all_ind != index)

    ex = x0[index] + 0
    ey = y0[index] + 0
    ez = z0[index] + 0

    x0 = x0[good]
    y0 = y0[good]
    z0 = z0[good]

    # Find the force on the single electon

    dx = x0 - ex
    dy = y0 - ey
    dz = z0 - ez

    dsq = dx**2 + dy**2 + dz**2
    dist = np.sqrt(dsq)

    # Force from each other electron
    forces_mag = 1./dsq

    fx = np.sum(forces_mag * dx/dist)
    fy = np.sum(forces_mag * dy/dist)
    fz = np.sum(forces_mag * dz/dist)

    # subtract off the radial component of the force vector
    fx -= fx*ex
    fy -= fy*ey
    fz -= fz*ez

    # normalize so the vector offset is similar to typical distance between points
    f_tot = np.sqrt(fx**2+fy**2+fz**2) / typical_dist

    fx /= f_tot
    fy /= f_tot
    fz /= f_tot

    my_sphere = elec_sphere(x0, y0, z0, ex, ey, ez, fx, fy, fz)

    # Need to make sure this doesn't go crazy and finds only the local minimum of shifting.
    fit_amplitude = minimize(my_sphere.potential, 0.1, method='CG')

    result = my_sphere.move_point(fit_amplitude.x, fraction=fraction)
    # return the new x,y,z coord
    return result

def minimize_global(fun, x0, args=(), maxfev=None, stepsize=0.5, maxiter=100, callback=None, **options):
    """

    Parameters
    ----------
    stepsize : float (0.5)
        The fraction of the bestfit amplitude to move each electron (guessing should be close to 0.5
        to prevent overshooting solution)

    # Can have this return a scipy OptimizeResult object. Could fire up some ipyparallel
    # engines and scatter-gather to speed things up.
    https://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html

    """

    bestx = x0
    besty = fun(x0)
    funcalls = 1
    niter = 0
    improved = True
    stop = False

    while improved and not stop and niter < maxiter:
        improved = False
        niter += 1
        x, y, z = x0_split(bestx)

        # Note, this is a brutal loop over each electron, figuring out how to move it.
        # Should be easy to scatter-gather this in parallel to get a big speedup.
        # could potentially slowly crank down the stepsize as it iterates.
        new_coords = [minimize_single(x, y, z, index, fraction=stepsize) for index in range(x.size)]
        # reshape to be a single vector again
        testx = np.array(new_coords).T.ravel()
        testy = fun(testx)
        if testy < besty:
            besty = testy
            bestx = testx
            improved = True
        if callback is not None:
            callback(bestx)
        if maxfev is not None and funcalls >= maxfev:
            stop = True
            break

    return OptimizeResult(fun=besty, x=bestx, nit=niter, nfev=funcalls, success=(niter > 1))



def xyz2U(x, y, z):
    """
    compute the potential
    """
    dsq = 0.

    indices = np.triu_indices(x.size, k=1)

    for coord in [x, y, z]:
        coord_i = np.tile(coord, (coord.size, 1))
        coord_j = coord_i.T
        dsq += (coord_i[indices]-coord_j[indices])**2

    d = np.sqrt(dsq)
    U = np.sum(1./d)
    return U


def iterate_potential_smart(x0, stepfrac=0.1):
    """
    Calculate the change in potential by shifting points in theta and phi directions
    # wow, that sure didn't work at all.
    """

    theta = x0[0:x0.size/2]
    phi = x0[x0.size/2:]
    x, y, z = thetaphi2xyz(theta, phi)
    
    U_input = xyz2U(x, y, z)

    # Now to loop over each point, and find where it's potenital minimum would be, and move it 
    # half-way there.
    xyz_new = np.zeros((x.size, 3), dtype=float)
    mask = np.ones(x.size, dtype=bool)
    for i in np.arange(x.size):
        mask[i] = 0
        fit = minimize(potential_single, [x[i], y[i], z[i]], args=(x[mask], y[mask], z[mask]))
        mask[i] = 1
        xyz_new[i] = fit.x/np.sqrt(np.sum(fit.x**2))

    xyz_input = np.array((x, y, z)).T
    diff = xyz_input - xyz_new

    # Move half way in x-y-z space
    xyz_out = xyz_input + stepfrac*diff
    # Project back onto sphere
    xyz_out = xyz_out.T/np.sqrt(np.sum(xyz_out**2, axis=1))
    U_new = xyz2U(xyz_out[0, :], xyz_out[1, :], xyz_out[2, :])
    theta, phi = xyz2thetaphi(xyz_out[0, :], xyz_out[1, :], xyz_out[2, :])
    return np.concatenate((theta, phi)), U_new


def iterate_potential_random(x0, stepsize=.05):
    """
    Given a bunch of theta,phi values, shift things around to minimize potential
    """

    theta = x0[0:x0.size/2]
    phi = x0[x0.size/2:]

    x, y, z = thetaphi2xyz(theta, phi)
    # Distance squared
    dsq = 0.

    indices = np.triu_indices(x.size, k=1)

    for coord in [x, y, z]:
        coord_i = np.tile(coord, (coord.size, 1))
        coord_j = coord_i.T
        d = (coord_i[indices]-coord_j[indices])**2
        dsq += d

    d = np.sqrt(dsq)

    U_input = 1./d

    # offset everything by a random ammount
    x_new = x + np.random.random(theta.size) * stepsize
    y_new = y + np.random.random(theta.size) * stepsize
    z_new = z + np.random.random(theta.size) * stepsize

    r = (x_new**2 + y_new**2 + z_new**2)**0.5
    # put back on the sphere
    x_new = x_new/r
    y_new = y_new/r
    z_new = z_new/r

    dsq_new = 0
    for coord, coord_new in zip([x, y, z], [x_new, y_new, z_new]):
        coord_i_new = np.tile(coord_new, (coord_new.size, 1))
        coord_j = coord_i_new.T
        d_new = (coord_i_new[indices]-coord_j[indices])**2
        dsq_new += d_new
    U_new = 1./np.sqrt(dsq_new)

    U_diff = np.sum(U_new)-np.sum(U_input)
    if U_diff > 0:
        return x0, 0.
    else:
        theta, phi = xyz2thetaphi(x_new, y_new, z_new)
        return np.concatenate((theta, phi)), U_diff


def ang_potential(x0):
    """
    If distance is computed along sphere rather than through 3-space.
    """
    theta = x0[0:x0.size/2]
    phi = np.pi/2-x0[x0.size/2:]

    indices = np.triu_indices(theta.size, k=1)

    theta_i = np.tile(theta, (theta.size, 1))
    theta_j = theta_i.T
    phi_i = np.tile(phi, (phi.size, 1))
    phi_j = phi_i.T
    d = _angularSeparation(theta_i[indices], phi_i[indices], theta_j[indices], phi_j[indices])
    U = np.sum(1./d)
    return U


def fib_sphere_grid(npoints):
    """
    Use a Fibonacci spiral to distribute points uniformly on a sphere.

    based on https://people.sc.fsu.edu/~jburkardt/py_src/sphere_fibonacci_grid/sphere_fibonacci_grid_points.py

    Returns theta and phi in radians
    """

    phi = (1.0 + np.sqrt(5.0)) / 2.0

    i = np.arange(npoints, dtype=float)
    i2 = 2*i - (npoints-1)
    theta = (2.0*np.pi * i2/phi) % (2.*np.pi)
    sphi = i2/npoints
    phi = np.arccos(sphi)
    return theta, phi


def even_points(npts, use_fib_init=True, method='CG', potential_func=elec_potential, maxiter=None):
    """
    Distribute npts over a sphere and minimize their potential, making them
    "evenly" distributed

    Starting with the Fibonacci spiral speeds things up by ~factor of 2.
    """

    if use_fib_init:
        # Start with fibonacci spiral guess
        theta, phi = fib_sphere_grid(npts)
    else:
        # Random on a sphere
        theta = np.random.rand(npts)*np.pi*2.
        phi = np.arccos(2.*np.random.rand(npts)-1.)

    x = np.concatenate((theta, phi))
    # XXX--need to check if this is the best minimizer
    min_fit = minimize(potential_func, x, method='CG', options={'maxiter': maxiter})

    x = min_fit.x
    theta = x[0:x.size/2]
    phi = x[x.size/2:]
    # Looks like I get the same energy values as https://en.wikipedia.org/wiki/Thomson_problem
    return theta, phi


def x02sphere(x0):
    x0 = x0.reshape(3, int(x0.size/3))
    x = x0[0, :]
    y = x0[1, :]
    z = x0[2, :]

    r = np.sqrt(x**2 + y**2 + z**2)
    x = x/r
    y = y/r
    z = z/r

    return np.concatenate((x, y, z))


def even_points_xyz(npts, use_fib_init=True, method='CG', potential_func=elec_potential_xyz, maxiter=None,
                    callback=None):
    """
    Distribute npts over a sphere and minimize their potential, making them
    "evenly" distributed

    Starting with the Fibonacci spiral speeds things up by ~factor of 2.
    """

    if use_fib_init:
        # Start with fibonacci spiral guess
        theta, phi = fib_sphere_grid(npts)
    else:
        # Random on a sphere
        theta = np.random.rand(npts)*np.pi*2.
        phi = np.arccos(2.*np.random.rand(npts)-1.)

    x = np.concatenate(thetaphi2xyz(theta, phi))
    # XXX--need to check if this is the best minimizer
    min_fit = minimize(potential_func, x, method='CG', options={'maxiter': maxiter}, callback=callback)

    x = x02sphere(min_fit.x)

    # Looks like I get the same energy values as https://en.wikipedia.org/wiki/Thomson_problem
    return x


