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
"""

import sys, os
sys.path.insert(0, os.path.realpath(os.path.pardir))

import numpy as np
import pylab as pl
import scipy.special as legfuns
import scipy.fftpack as ft

from scipy import integrate
import scipy.interpolate as intpol



def gen_bas(rad, sig, lev, lodd, blowup=1, cos2=False):
    """Generate a basis set of radial Gaussians and Legendre pols.
    sig determines the step width, l the number of polynomials;
    Returns the upper right corner with even ls
    TODO implement odd ls properly"""

    # evaluate radial basis set
    n_funs = np.int(rad / sig)
    sig_sq = sig ** 2
    XY = np.arange(-1 * rad * blowup, (rad * blowup) +1,dtype='float64')
    diam = XY.shape[0]
    R = np.sqrt(XY ** 2 + XY[:, None] ** 2)[rad:, rad:]
#    R2 = R ** 2
    r_basis = np.zeros([n_funs, rad + 1, rad + 1])
    r_funs = np.zeros([n_funs, rad + 1])
    for n in np.arange(n_funs):
        Rn = n * sig
        r_basis[n] = np.exp(-1 * ((R - Rn) ** 2) / sig_sq) #/ R2
        if Rn:
            r_basis[n] += np.exp(-1 * ((R + Rn) ** 2) / sig_sq)
        r_funs[n] = r_basis[n, 0, :]

    # evaluate angular basis set

    th = np.arctan2(XY, XY[:,None])[rad:,rad:]
    n_lev = np.arange(lev + 1)
    n_lodd = np.arange(lodd + 1)

    n_l = np.hstack((n_lev[0::2], n_lodd[1::2]))

    if cos2:
        ang_basis = np.zeros([1, rad +1, rad + 1])
        ang_basis[0] = np.cos(th) ** 2
        polar_basis = np.zeros([n_funs, rad +1, rad + 1])
    else:
        ang_basis = np.zeros([n_l.shape[0], rad +1, rad + 1])
        for i, k in enumerate(n_l):
            ang_basis[i] = legfuns.eval_legendre(k, np.cos(th))
        polar_basis = np.zeros([n_funs * n_l.shape[0], rad +1, rad + 1])

    # multiply them
    for i, r_im in enumerate(r_basis):
        for j, th_im in enumerate(ang_basis):
            polar_basis[i + j * n_funs] = r_im * th_im
    return polar_basis, r_funs

### 2015-05-04: reworking things a bit with an interpolated basis set
### later that night: this is useless

def gen_rad_bas(rad, sig, blowup=1):
    n_funs = np.int(rad / sig)
    sig_sq = sig ** 2
    XY = np.linspace(0, rad, (rad * blowup) + 1)
    diam = XY.shape[0]
    R = np.sqrt(XY ** 2 + XY[:, None] ** 2)#[rad:, rad:]
    r_basis = np.zeros([n_funs, diam, diam])
    r_funs = np.zeros([n_funs, diam])
    Rn = np.arange(n_funs + 1)
    Rn *= sig
    Rn = Rn[:, None, None]
    r_basis = np.exp(-1 * ((R - Rn) ** 2) / sig_sq) #/ R2
    r_basis += np.exp(-1 * ((R + Rn) ** 2) / sig_sq)
    r_funs = r_basis[:, 0, :]
    return r_basis, r_funs

def gen_ang_bas(rad, lev, lodd=0, blowup=1):
    XY = np.linspace(0, rad, (rad * blowup) + 1)
    diam = XY.shape[0]
    th = np.arctan2(XY, XY[:,None])#[rad:,rad:]
    n_lev = np.arange(lev + 1)
    n_lodd = np.arange(lodd + 1)

    n_l = np.hstack((n_lev[0::2], n_lodd[1::2]))
    n_l = n_l[:, None, None]

    ang_basis = np.zeros([n_l.shape[0], diam, diam])
    ang_basis = legfuns.eval_legendre(n_l, np.cos(th))
    return  ang_basis



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
    n = np.arange(N +1)
    num = N - n
    den = N - n - 1
    frac = num / den.astype(np.float_)
    lam = lam[:, None]
    phi = frac ** lam
    lam1 = lam + 1
    gam = (2 * den / lam1) * (1 - frac ** lam1)
    gam = -1 * h[:, None] * gam

    return np.diag(phi), gam[:, None]

def abel_vec(f):
    f = f[:,::-1]
    x = np.zeros([lam.shape[0], f.shape[0]])
    g = np.zeros(f.shape)
    N = np.float(f.shape[1])
    Phi, Gam = prop_vec(N, lam, h)

    for i in np.arange(N):
        g[:, i] = np.dot(C, x)
        x =  np.dot(Phi[i], x) + Gam[i] * f[:,i]
    g[:,0] = 2 * g[:,-1] - g[:,-2]

    return np.roll(g[:,::-1],1, axis=1 )

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

    for i in np.arange(N):
        g[:,i] = np.dot(C, x)
        Phi, Gam = propag(N, i, lam, h)
        x =  np.dot(Phi, x) + Gam * f[:,i]
    g[:,0] = 2 * g[:,-1] - g[:,-2]

    return np.roll(g[:,::-1],1, axis=1 )

def AbelInt(f):
    """Does the Abel integral numerically and linewise. n.b.: super slow!"""
    int = np.zeros(f.shape)
    err = np.zeros(f.shape)
    r = np.arange(f.shape[-1])
    for i, line in enumerate(f):
        print i
        ck = intpol.splrep(r, line)
        def integrand(r, x):
            return (intpol.splev(r, ck) * r) / np.sqrt(r**2 - x**2)
        for x, el in enumerate(line):
            int[i, x], err[i,x] = integrate.quad(integrand, x, r[-1], args = x)
    return int, err



if __name__ == '__main__':
    pass
else:
    r_max = 150
    sigma = 2.00
    n_even = 8
    n_odd = 0

    condition = 1E-7

    store_path = os.path.join(sys.path[0], 'storage')
    ext = '-' + str(r_max)+'-'+str(n_even)

    bs, r_funs = gen_bas(r_max, sigma, n_even, n_odd, cos2=False)
    np.save(store_path + '/bs' + ext, bs.reshape(bs.shape[0],-1).T)
    np.save(store_path + '/rf' + ext, r_funs.T)

    ab = np.zeros((bs.shape[0], bs.shape[1] * bs.shape[2]))

    for i, img in enumerate(bs):
        print 'Transforming img', i
        ab[i] = AbelTrans(img).ravel()

    np.save(store_path + '/ab' + ext, ab)
    del bs, r_funs
    FtF = np.dot(ab, ab.T)
    np.save(store_path + '/FtF' + ext, FtF)
    del FtF
#   inv = sp.linalg.pinv2(ab, rcond=condition)
#   np.save('./storage/inv' + ext + '-E7', inv)


