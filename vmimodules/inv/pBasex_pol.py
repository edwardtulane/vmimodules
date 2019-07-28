# -*- coding: utf-8 -*-
"""
Created on Thu Sep 11 16:14:59 2014

@author: felix

"""
from __future__ import print_function

import sys, os
sys.path.insert(0, os.path.realpath(os.path.pardir))

from glob import glob

import numpy as np
import scipy.special as legfuns

from scipy import integrate as integ
import scipy.interpolate as intpol

import ctypes
dll = sorted(glob('abel.cpython*so'))
dll = dll[-1]
lib = ctypes.CDLL('./%s' % dll)

kpol = lib.kernel_pol
kpol.restype = ctypes.c_double
kpol.argtypes = (ctypes.c_int, ctypes.c_double)



def gen_bas(rad, polN, sig, lev, lodd):
    """Generate a basis set of radial Gaussians and Legendre pols.
    sig determines the step width, l the number of polynomials;
    Returns the upper right corner with even ls.
    'blowup' multiplies the grid in the x-direction.
    TODO implement odd ls properly"""

    # evaluate radial basis set
    n_funs = np.int(rad / sig)
    sig_sq = sig ** 2
    XY = np.linspace(-1 * rad, rad, (2 * rad) + 1)
    ZZ = np.linspace(-1 * rad, rad, (2 * rad) + 1) 
    diam, zdim = XY.shape[0], ZZ.shape[0]
    R = np.sqrt(XY ** 2 + ZZ[:, None] ** 2)[rad:, (rad):]
    r_basis = np.zeros([n_funs, rad + 1, (rad) + 1])
    r_funs = np.zeros([n_funs, (rad)  + 1])

    Rn = np.arange(n_funs + 1, dtype=np.float_)
    Rn *= sig
    Rn = Rn[:, None, None]

    r_basis = np.exp(-1 * ((R - Rn) ** 2) / sig_sq) #/ R2
    r_basis += np.exp(-1 * ((R + Rn) ** 2) / sig_sq)
    r_funs = r_basis[:, 0, :]

    # evaluate angular basis set
    th = np.arctan2(XY, ZZ[:,None])[rad:, (rad):]
    n_lev = np.arange(lev + 1)
    n_lodd = np.arange(lodd + 1)

    n_l = np.hstack((n_lev[0::2], n_lodd[1::2]))
    n_l = n_l[:, None, None]

    ang_basis = np.zeros([n_l.shape[0], rad +1, (rad)+ 1])
    ang_basis = legfuns.eval_legendre(n_l, np.cos(th))

    # multiply them
    r_basis.shape = (1, n_funs + 1, rad + 1, (rad) + 1)
    ang_basis.shape = (n_l.shape[0], 1, rad + 1, (rad) + 1)

    r_funs[0] /= 2
#   r_basis[0,0] /= 2
    polar_basis = r_basis * ang_basis
#   polar_basis /=2

    ab = np.zeros(list(polar_basis.shape[:3]) + [polN])
                   
    for i, l in enumerate(n_l.ravel()):
        for j, rk in enumerate(Rn.ravel()):
            ab[i,j] = abel_integrate(rad+1, polN, rk, sig, l)
            print('.', end='')
        print('\n Transformed l = %d' % (l))

#   return r_basis, ang_basis, r_funs 
    return polar_basis, ab, r_funs

def abel_integrate(radN, polN, rk, sig, l):
    ab_pl = np.zeros([radN, polN])
    angs = np.linspace(0, np.pi/2, polN)  
    
    for i in range(radN):
        for j,th in enumerate(angs):
            ab_pl[i,j], _ = integ.quad(kpol, i, radN+20,
                                       args=(th, i, rk, sig, l)
                                      )
                                       
    return ab_pl

if __name__ == '__main__':
#    pass
#else:
    r_max = 150
    polN = 257
    sigma = 2.00
    n_even = 2
    n_odd = 0

    store_path = os.path.join(sys.path[0], 'storage')
    ext = '-' + str(r_max)+'-'+str(n_even)

    bs, ab, r_funs = gen_bas(r_max, polN, sigma, n_even, n_odd)

    np.save(store_path + '/bs' + ext, bs.reshape(bs.shape[0] * bs.shape[1], bs.shape[2] ** 2).T)
    np.save(store_path + '/rf' + ext, r_funs.T)
    del bs, r_funs

    ab = ab.reshape(ab.shape[0] * ab.shape[1], 
                    ab.shape[2] * ab.shape[3])
    np.save(store_path + '/ab' + ext, ab)
    FtF = np.dot(ab, ab.T)
    np.save(store_path + '/FtF' + ext, FtF)

