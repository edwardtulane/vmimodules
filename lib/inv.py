# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 20:19:00 2014

@author: brausse
"""

import sys, os
mod_home = os.path.realpath(os.path.pardir)
stor_dir = os.path.join(mod_home, 'storage')
sys.path.insert(0, mod_home)

import numpy as np
import vmiproc as vmp

M1, M2 = vmp.iniBasex(stor_dir)
MTM1, MTM2 = np.dot(M1.T, M1), np.dot(M2.T, M2)

ab = np.load('/home/brausse/program/vmimodules/storage/ab-250-8.npy'#, mmap_mode='r'
            )
bs = np.load('/home/brausse/program/vmimodules/storage/bs-250-8.npy'#, mmap_mode='r'
            )
FtF = np.load('/home/brausse/program/vmimodules/storage/FtF-250-8.npy'#, mmap_mode='r'
            )
rf = np.load('/home/brausse/program/vmimodules/storage/rf-250-8.npy'#, mmap_mode='r'
            )


#def invertFourierHankel(arr):
#    dim = (arr.shape[1] - 1) /2
#    arr._shift = np.append(arr.rect[:,dim:], arr.rect[:,:dim],axis=1)
#    arr.fourt = fft.fft(arr._shift,axis=1)
#    ft_freqs = fft.fftfreq(arr.fourt.shape[1])
####
#    jn = bessel.jn_zeros(0, dim + 2)
#    S, R1 = jn[-1], ft_freqs.max()
#    R2 = S / (2 * np.pi * R1)
#    print R1, R2, S
#    jn = jn[:-1]
#    F1_arg = jn / (2 * np.pi * R2)
#    F1_arg *= (ft_freqs.shape[0] - 1) / (2 * R1)
#    J1_vec = abs(bessel.j1(jn) ** -1)
#    if not arr.__Cmn.shape == (dim + 1, dim + 1):
#        arr.__jn_mat = (jn * jn[:, None]) / S
#        arr.__J1_mat =  J1_vec * J1_vec[:, None]
#        arr.__Cmn = (2 / S) * bessel.j0(arr.__jn_mat) * arr.__J1_mat
#    else:
#        pass
#
#    F1 = np.empty((arr.fourt.shape[0], dim + 1), dtype='complex')
#    arr.FHT = np.empty((arr.fourt.shape[0], dim + 1), dtype='complex')
#
#    for i, line in enumerate(arr.fourt):
#        ft_cR = sig.cspline1d(line.real)
#        ft_cI = sig.cspline1d(line.imag)
#        F1[i] = ( sig.cspline1d_eval(ft_cR, F1_arg) \
#                        + 1j * sig.cspline1d_eval(ft_cI, F1_arg) \
#                        ) * J1_vec * R1
#
#    arr.FHT = np.dot(F1, arr.__Cmn)
#    arr.FHT /= (R2 * J1_vec)
#    arr.F2_arg = jn / (2 * np.pi * R1)
#    arr.orig = np.dot(arr.FHT, arr.__Cmn)

def invertBasex(arr):
    bsx = vmp.Basex(arr, 10, 0, M1, M2,
                         MTM1, MTM2)
    return bsx

def invertPolBasex(arr):
        arr = arr.ravel()
        pbsx = np.dot(np.linalg.inv(FtF + 1 * np.eye(FtF.shape[0])), np.dot(ab, arr))
        return pbsx

def pbsx2fold(pbsx):
        fold = np.dot(pbsx, bs.T)
        fold.shape = (251, 251)
        return fold

def pbsx2rad(pbsx):
        int = np.dot(pbsx[:166], rf.T)
        beta = np.dot(pbsx[166:2 * 166], rf.T)
        return int, beta

#    def pbsx2fold(pbsx):
#            inv = np.dot(pbsx, bs.T)
#        inv.shape = (251,251)
#        return inv
