import numpy as np
from thomson import *


npts = 600

theta, phi = fib_sphere_grid(npts)
# Note, declination = phi - pi/2, RA = theta
x, y, z = thetaphi2xyz(theta, phi)
x0 = np.concatenate((x,y,z))
init_potential = elec_potential_xyz(x0)
print('%i points, initial potential = %.2f' % (npts, init_potential))

fit_result = minimize_global(elec_potential_xyz, x0, maxiter=600, stepsize=0.8)
print('final potential = %.2f' % fit_result.fun)
print('%i iterations of solver' % fit_result.nit)
np.savez('%i_min.npz' % npts, fit_result=fit_result)


# Timing, 320 iterations of npts=500, 4m2.6s
# 