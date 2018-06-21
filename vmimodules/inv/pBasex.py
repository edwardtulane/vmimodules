# -*- coding: utf-8 -*-
"""
Created on Thu Sep 11 16:14:59 2014

@author: felix

2014-11-13: Checked and corrected both state-variable Abel Transform and
basis set generation
-- Shifted Transform by one entry, gave smaller norms vs. quadrature;
stops at entry N-1, entry N is guessed by differencing
-- Radial basis functions now contain contributions from both pos. and neg.
radii; also, the radial functions are now dumped as r_funs
-- Added a numerical implementation for discrete data via spline interpolation;
it works, but is rather messy and extremely slow

2015-05-05: Vectorised Abel transformation and basis set generation.
-- Added interpolation to increase the fidelity of the state variable transform.
For large basis sets the vectorised trans. may easily run out of the memory.
-- Removed unnecessary dependencies.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import sys, os
sys.path.insert(0, os.path.realpath(os.path.pardir))

import numpy as np
import scipy.special as legfuns

from scipy import integrate
import scipy.interpolate as intpol

import numexpr as ne


def gen_bas(rad, sig, lev, lodd, blowup=1, cos2=False):
    """Generate a basis set of radial Gaussians and Legendre pols.
    sig determines the step width, l the number of polynomials;
    Returns the upper right corner with even ls.
    'blowup' multiplies the grid in the x-direction.
    TODO implement odd ls properly"""

    # evaluate radial basis set
    n_funs = np.int(rad / sig)
    sig_sq = sig ** 2
    XY = np.linspace(-1 * rad, rad, (2 * rad * blowup) + 1)
    ZZ = np.linspace(-1 * rad, rad, (2 * rad) + 1) 
    diam, zdim = XY.shape[0], ZZ.shape[0]
    R = np.sqrt(XY ** 2 + ZZ[:, None] ** 2)[rad:, (rad * blowup):]
    r_basis = np.zeros([n_funs, rad + 1, (rad * blowup) + 1])
    r_funs = np.zeros([n_funs, (rad * blowup)  + 1])

    Rn = np.arange(n_funs + 1, dtype=np.float_)
    Rn *= sig
    Rn = Rn[:, None, None]

    r_basis = np.exp(-1 * ((R - Rn) ** 2) / sig_sq) #/ R2
    r_basis += np.exp(-1 * ((R + Rn) ** 2) / sig_sq)
    r_funs = r_basis[:, 0, :]

    # evaluate angular basis set
    th = np.arctan2(XY, ZZ[:,None])[rad:, (rad * blowup):]
    n_lev = np.arange(lev + 1)
    n_lodd = np.arange(lodd + 1)

    n_l = np.hstack((n_lev[0::2], n_lodd[1::2]))
    n_l = n_l[:, None, None]

    if cos2:
        ang_basis = np.zeros([1, rad +1, rad + 1])
        ang_basis[0] = np.cos(th) ** 2
        polar_basis = np.zeros([n_funs, rad +1, rad + 1])
    else:
        ang_basis = np.zeros([n_l.shape[0], rad +1, (rad * blowup)+ 1])
        ang_basis = legfuns.eval_legendre(n_l, np.cos(th))

    # multiply them
    r_basis.shape = (1, n_funs + 1, rad + 1, (rad * blowup) + 1)
    ang_basis.shape = (n_l.shape[0], 1, rad + 1, (rad * blowup) + 1)

    r_funs[0] /= 2
#   r_basis[0,0] /= 2
#   polar_basis = r_basis * ang_basis
#   polar_basis /=2

    return r_basis, ang_basis, r_funs 
#   return polar_basis, r_funs

### state-variable Abel transform

# Eigenvalues and expansion coefficients
h = np.array([0.318, 0.19, 0.35, 0.82, 1.80, 3.90, 8.30, 19.60, 48.30])
lam = np.array([0.0, -2.1, -6.2, -22.4, -92.5, \
        -414.5, -1889.4, -8990.9, -47391.1])

### state-variable propagator
def propag(N, n, lmb, h):
    """Returns the propagator elements Phi and Gamma"""
    frac = (N - n) / (N - n - 1)
    ph = frac ** lmb
    lmb1 = lmb + 1
    gam = (2 * (N - n - 1) / (lmb1)) * (1 - frac ** (lmb1))
    GAM = -1 * h * gam
    return np.diag(ph), GAM[:,None]

def prop_vec(N, lam, h):
    """Vector form of propag()"""
    n = np.arange(N +1)
    num = N - n
    den = N - n - 1
    frac = num / den.astype(np.float_)
    lam = lam[:, None]
    phi = frac ** lam
    phi_diag = np.zeros((lam.shape[0] * lam.shape[0], int(N + 1)))
    phi_diag[::lam.shape[0] + 1,:] = phi
    phi_diag.shape = (lam.shape[0], lam.shape[0], int(N + 1))
    lam1 = lam + 1
    gam = (2 * den / lam1) * (1 - frac ** lam1)
    gam = -1 * h[:, None] * gam

    return phi_diag, gam[:, None]

### 2014-09-18 Normalisation confirmed
C = np.ones(9) * (np.pi)
# TIP omit step width delta

def AbelTrans(f):
    """Transforms an image's right half by flipping and recursive computation
    of the impulse response. See Hansen and Law, JOSA A 1984"""
    f = f[:,::-1]
    x = np.zeros([lam.shape[0], f.shape[0]])
    g = np.zeros(f.shape)
    N = np.float(f.shape[1])
    Phi, Gam = prop_vec(N, lam, h)

    for i in range(int(N)):
        g[:,i] = np.dot(C, x)
#       Phi, Gam = propag(N, i, lam, h)
        x =  np.dot(Phi[:,:,i], x) + Gam[:,:,i] * f[:,i]
    g[:,0] = 2 * g[:,-1] - g[:,-2]

    return np.roll(g[:,::-1],1, axis=1 )

def AbelVec(f):
    """Vectorised Abel Transform"""
    f = f[:,:,::-1]
    x = np.zeros([f.shape[0], lam.shape[0], f.shape[1]])
    g = np.zeros(f.shape)
    N = np.float(f.shape[2])
    Phi, Gam = prop_vec(N, lam, h)

    for i in range(int(N)):
        g[:,:,i] = np.dot(C, x)
        x =  np.swapaxes(np.dot(Phi[:,:,i], x), 0, 1) + Gam[None,:,:,i] * f[:,None,:,i]
        print('.', end='')
    g[:,:,0] = 2 * g[:,:,-1] - g[:,:,-2]

    return np.roll(g[:,:,::-1],1, axis=2 )

def AbelInt(f):
    """Does the Abel integral numerically and linewise. n.b.: super slow!"""
    int = np.zeros(f.shape)
    err = np.zeros(f.shape)
    r = np.arange(f.shape[-1])
    for i, line in enumerate(f):
        print(i)
        ck = intpol.splrep(r, line)
        def integrand(r, x):
            return (intpol.splev(r, ck) * r) / np.sqrt(r**2 - x**2)
        for x, el in enumerate(line):
            int[i, x], err[i,x] = integrate.quad(integrand, x, r[-1], args = x)
    return int, err

def project_polar(img, r_max, sigma):
    from scipy.ndimage import interpolation as ndipol
    import scipy.signal as sig

    ck = sig.cspline2d(img, 0)

    rr = np.linspace(0, r_max, r_max + 1)
    thth = np.linspace(0, np.pi / 2, 257)
    pol_coord, rad_coord = np.meshgrid(thth, rr)
    dx = np.pi / (257 - 1)
    
    x_coord = rad_coord * np.sin(pol_coord) #- 0.5
    y_coord = rad_coord * np.cos(pol_coord) #- 0.5

    polar = ndipol.map_coordinates(ck, [y_coord, x_coord], prefilter=False,
                                   output=np.float_)

    return polar

if __name__ == '__main__':
#    pass
#else:
    r_max = 300
    sigma = 1.00
    n_even = 6
    n_odd = 0

    blowup = 4

    store_path = os.path.join(sys.path[0], 'storage')
    ext = '-' + str(r_max)+'-'+str(n_even)

    r_bas, ang_bas, r_funs = gen_bas(r_max, sigma, n_even, n_odd, blowup, cos2=False)
    bs = np.zeros((ang_bas.shape[0], r_bas.shape[1], r_bas.shape[2], r_bas.shape[2]))
    ab = np.zeros((ang_bas.shape[0], r_bas.shape[1], r_bas.shape[2], r_bas.shape[2]))
    ab_p = np.zeros((ang_bas.shape[0], r_bas.shape[1], r_bas.shape[2], 257))

    for l in range(ang_bas.shape[0]):
        tmp_bs = r_bas * ang_bas[l]
        tmp_bs = np.reshape(tmp_bs, (-1, r_bas.shape[2], r_bas.shape[3]))
        bs[l] = tmp_bs[:,:,::blowup]
        ab[l] = AbelVec(tmp_bs)[:,:,::blowup] / np.float(blowup)
        
        for m, img in enumerate(ab[l]):
            ab_p[l,m] = project_polar(img, r_max, sigma)

        print('\n Transformed l = %d' % (2*l))

    np.save(store_path + '/bs' + ext, bs.reshape(bs.shape[0] * bs.shape[1], bs.shape[2] ** 2).T)
    np.save(store_path + '/rf' + ext, r_funs[:, ::blowup].T)
    del bs, r_funs, r_bas, ang_bas

    ab = ab.reshape(ab.shape[0] * ab.shape[1], ab.shape[2] ** 2)
    np.save(store_path + '/ab' + ext, ab)
    FtF = np.dot(ab, ab.T)
    np.save(store_path + '/FtF' + ext, FtF)

