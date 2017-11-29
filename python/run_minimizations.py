#!/Users/yoachim/lsstp3/DarwinX86/miniconda3/4.2.12.lsst1/bin/python
import thomson as th
import numpy as np
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("npts", type=int)
    parser.add_argument("--maxiter", type=int, default=300)
    args = parser.parse_args()

    seed = 42
    npts = args.npts
    maxiter = args.maxiter

    theta, phi = th.fib_sphere_grid(npts)
    # Note, declination = phi - pi/2, RA = theta
    x, y, z = th.thetaphi2xyz(theta, phi)
    noise_string=''
    add_noise = True
    if add_noise:
        # Maybe add in some randomness? Getting stripes in result that I think are remnants of spiral.
        # set a seed so it's repeatable?
        np.random.seed(seed)

        # check how close things are
        distances = np.sqrt((x-x[0])**2 + (y-y[0])**2 + (z-z[0])**2)
        typical_dist = np.min(distances[1:])
        scale = 0.9

        x += (np.random.rand(x.size)-0.5)*typical_dist*scale * np.sqrt(2)
        y += (np.random.rand(x.size)-0.5)*typical_dist*scale * np.sqrt(2)
        z += (np.random.rand(x.size)-0.5)*typical_dist*scale * np.sqrt(2)

        # Snap back to sphere
        r = np.sqrt(x**2+y**2+z**2)

        x /= r
        y /= r
        z /= r
        noise_string='noise%.2f' % scale

    x0 = np.concatenate((x, y, z))
    init_potential = th.elec_potential_xyz(x0)
    print('%i points, initial potential = %.2f' % (npts, init_potential))

    fit_result = th.minimize_global(th.elec_potential_xyz, x0, maxiter=maxiter, stepsize=0.8)
    print('final potential = %.2f' % fit_result.fun)
    print('final stepsize = %.2f' % fit_result.stepsize)
    print('%i iterations of solver' % fit_result.nit)
    np.savez('%i_min%s.npz' % (npts, noise_string), fit_result=fit_result)


    # Timing, 320 iterations of npts=500, 4m2.6s
    # 