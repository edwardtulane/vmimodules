# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 20:19:00 2014

@author: brausse
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import sys, os

import numpy as np
import scipy as sp
import scipy.special as spc
import scipy.integrate as integ
import scipy.signal as sig
import scipy.fftpack as fft
import scipy.special as bessel
from . import proc as vmp

# import vmimodules.conf
from . import mod_home
# mod_home = vmimodules.conf.mod_home
stor_dir = os.path.join(mod_home, 'storage')

class Inverter(object):
    """
    Docstring
    """
    def __init__(self, r_max=250, n_even=8, dir=stor_dir, dryrun=0):
        self.__ext = '-'.join(('',  str(r_max), str(n_even))) #'-' + str(r_max)+'-'+str(n_even)

        self.dim = r_max + 1
        self.__dim2 = np.linspace(0, 1, self.dim) ** 2

        if not dryrun:
            self.ab = np.load(stor_dir + '/ab' + self.__ext + '.npy')
            self.bs = np.load(stor_dir + '/bs' + self.__ext + '.npy')
            self.rf = np.load(stor_dir + '/rf' + self.__ext + '.npy')
#           self.btb = self.bs.T.dot(self.bs)
            self.FtF = np.load(stor_dir + '/FtF' + self.__ext + '.npy')
            self.__M1, self.__M2 = vmp.iniBasex(stor_dir + '/')
            self.__MTM1, self.__MTM2 = np.dot(self.__M1.T, self.__M1), np.dot(
                    self.__M2.T, self.__M2),

            self.lvals = (n_even / 2) + 1
            self.n_funs = self.bs.shape[1] / self.lvals
            self.polN = self.ab.shape[1] / self.dim
#       self.th, self.lfuns = self.gen_lfuns(self.lvals)

#   set up a context manager?
    def __enter__(self):
        return self

    def __exit__(self):
        pass

#==============================================================================

    def invertFourierHankel(self, arr):
        dim = (arr.shape[1] - 1) /2
        shift = np.append(arr[:,dim:], arr[:,:dim],axis=1)
        fourt = fft.fft(shift,axis=1)
        ft_freqs = fft.fftfreq(fourt.shape[1])
    ###
        jn = bessel.jn_zeros(0, dim + 2)
        S, R1 = jn[-1], ft_freqs.max()
        R2 = S / (2 * np.pi * R1)
        print(R1, R2, S)
        jn = jn[:-1]
        F1_arg = jn / (2 * np.pi * R2)
        F1_arg *= (ft_freqs.shape[0] - 1) / (2 * R1)
        J1_vec = abs(bessel.j1(jn) ** -1)
#       if not arr.__Cmn.shape == (dim + 1, dim + 1):
        jn_mat = (jn * jn[:, None]) / S
        J1_mat =  J1_vec * J1_vec[:, None]
        Cmn = (2 / S) * bessel.j0(jn_mat) * J1_mat
#       else:
#           pass

        F1 = np.zeros((fourt.shape[0], dim + 1), dtype='complex')
        F2 = np.zeros_like(F1)
#       FHT = np.zeros((fourt.shape[0], dim + 1), dtype='complex')

        for i, line in enumerate(fourt):
            ft_cR = sig.cspline1d(line.real)
            ft_cI = sig.cspline1d(line.imag)
            F1[i] = ( sig.cspline1d_eval(ft_cR, F1_arg) \
                            + 1j * sig.cspline1d_eval(ft_cI, F1_arg) \
                            ) * J1_vec * R1

        FHT = np.dot(F1, Cmn)
        FHT /= (R2 * J1_vec)
        F2_arg = jn / (2 * np.pi * R1)
        orig = np.dot(FHT, Cmn)

        for i, line in enumerate(FHT):
            ft_cR = sig.cspline1d(line.real)
            ft_cI = sig.cspline1d(line.imag)
            F2[i] = ( sig.cspline1d_eval(ft_cR, F2_arg) \
                            + 1j * sig.cspline1d_eval(ft_cI, F2_arg) \
                            ) * J1_vec * R1
        return FHT, F2, orig

#==============================================================================

    def invertMaxEnt(self, arr, T=0, P=2):
        import shutil as sh
        arr_path = os.path.join(mod_home, 'inv', 'bin')
        cur_path = os.path.abspath(os.curdir)
        tmp_path = os.path.join('/tmp', '%i' % os.getpid())

        if not os.path.exists(tmp_path):
            os.mkdir(tmp_path)

        os.chdir(tmp_path)
        sh.copy(os.path.join(arr_path, 'MEVIR.elf'), tmp_path)

        np.savetxt('tmp_arr' , arr)

        os.system('./MEVIR.elf -S1 -R2 -T%d -P%d -I70 tmp_arr' % (T, P))

        if os.system('grep "Time" MaxAbel.log') >0:
            raise Exception('Maximum Entropy reconstruction failed!')

        os.system('sed -e "s/D/e/g" -i MXLeg.dat')
        leg, invmap, res = (np.loadtxt('MXLeg.dat').T[1:], 
                            np.loadtxt('MXmap.dat', delimiter=','), 
                            np.loadtxt('MXsim.dat', delimiter=',')
                           )
        os.chdir(cur_path)
        return leg, invmap, arr - res

    def invertMaxLeg(self, arr, T=0, P=2):
        import shutil as sh
        arr_path = os.path.join(mod_home, 'inv', 'bin')
        cur_path = os.path.abspath(os.curdir)
        tmp_path = os.path.join('/tmp', '%i' % os.getpid())

        if not os.path.exists(tmp_path):
            os.mkdir(tmp_path)

        os.chdir(tmp_path)
        sh.copy(os.path.join(arr_path, "Meveler2beta.elf"), tmp_path)

        np.savetxt('tmp_arr', arr, fmt='%i')

        os.system('./Meveler2beta.elf -S1 -R2 -T%d -P%d -I70 tmp_arr' % (T, P))

        if os.system('grep "Time" Meveler.log') >0:
            raise Exception('Maximum Entropy reconstruction failed!')

        os.system('sed -e "s/D/e/g" -i MEXdis.dat')
        leg, invmap, res = (np.loadtxt('MEXdis.dat').T[1:], 
                            np.loadtxt('MEXmap.dat', delimiter=','), 
                            np.loadtxt('MEXsim.dat', delimiter=',')
                           )
        os.chdir(cur_path)
        return leg, invmap, arr - res

    def invertBasex(self, arr):
        bsx, res = vmp.Basex(arr, 10, 0, self.__M1, self.__M2,
                            self.__MTM1, self.__MTM2)
        leg_p = vmp.get_raddist(bsx, arr.shape[0])
#       leg_p = leg_p * self.__dim2[None,:]
        return leg_p, bsx, res

    def invertPolBasex(self, arr, reg=1, get_pbsx=False):
        arr = arr.ravel()
#           pbsx = np.dot(np.linalg.inv(self.FtF + reg * np.eye(self.FtF.shape[0])), 
#                  np.dot(self.ab, arr))
        rhs = np.dot(self.ab, arr)
        lhs = self.FtF + reg * np.eye(self.FtF.shape[0])
        pbsx = np.linalg.solve(lhs, rhs)

        if get_pbsx:
            return pbsx

        leg = self.pbsx2rad(pbsx)
        leg *= self.__dim2
        pbsx = pbsx.ravel()
        inv_map = self.pbsx2fold(pbsx)
        res = arr - self.pbsx2ab(pbsx).ravel()
        res.shape = (self.dim, self.polN)
        return leg, inv_map, res

    def get_raddist_bs(self, arr):
        arr = arr.ravel()
        bta = self.bs.T.dot(arr.ravel())
        dist = np.linalg.solve(self.btb, bta)
        bta.shape = (-1, self.n_funs)
        dist = np.dot(bta, self.rf.T)

        return dist

    def invertImage(self, img,  radN, order=8):

        legq = np.zeros([4, order / 2 + 1, radN])

        qu = vmp.quadrants(img)
        arrq = np.zeros_like(qu)
        resq = arrq.copy()

        for i in range(4):
            q = qu[i]
            _, arr, res = self.invertMaxEnt(q)
            arrq[i], resq[i] = arr, res
            legq[i] = self.get_raddist(arr, radN, order)

        arrs = vmp.compose(arrq)
        ress = vmp.compose(resq)

        return legq, arrs, ress

#==============================================================================

    def pbsx2fold(self, pbsx):
            fold = np.dot(pbsx, self.bs.T)
            fold.shape = (self.dim, self.dim)
            return fold

    def pbsx2ab(self, pbsx):
            fold = np.dot(pbsx, self.ab)
            fold.shape = (self.dim, self.polN)
            return fold

    def pbsx2rad(self, pbsx):
            pbsx.shape = (-1, self.n_funs)
            dist = np.dot(pbsx, self.rf.T)
            return dist

    def gen_beta_vec(self, lvals):
        ll = np.arange(lvals) * 2
        th = np.linspace(0, np.pi, 2**20+1)
        beta_vec = np.zeros(lvals)
        for i, l in enumerate(ll): 
            up = spc.eval_legendre(l, np.cos(th)) ** 2 * np.sin(th) * np.cos(th) ** 2
            lo = spc.eval_legendre(l, np.cos(th)) ** 2 * np.sin(th)
            beta_vec[i] = integ.romb(up) / integ.romb(lo)
        return beta_vec

    def gen_lfuns(self, lvals):
        ll = np.arange(lvals) * 2
        th = np.linspace(0, np.pi, 2**9+1)
#       lfuns = np.zeros((lvals, 2**15+1))
        lfuns = spc.eval_legendre(ll[:, None], np.cos(th))
#       for i, l in enumerate(ll): 
#           lfuns[i] = spc.eval_legendre(l, np.cos(th))
        return th, lfuns

    def get_beta_map(self, pbsx):
        pbsx.shape = (pbsx.shape[0], self.lvals, -1)
        dists = np.dot(pbsx.swapaxes(1,2), self.lfuns)
        lo = dists ** 2 * np.sin(self.th)
        up = lo * np.cos(self.th) ** 2
        return integ.romb(up, axis=2) / integ.romb(lo, axis=2)

if __name__ == '__main__':
    from vis import *
    from vmiclass import Frame, RawImage
    t = RawImage(mod_home + '/TWtest.raw', xcntr=249, ycntr=234, radius=200)
    f = t.crop_square(-51)
    f.interpol()
    r = f.eval_rect(501)
    r_qu = vmp.quadrants(r)
    r_qu = r_qu.mean(0)
    r = vmp.unfold(r_qu, 1,1)
    inv = Inverter(250, 16)

#   i1 = inv.invertBasex(r)
    i2 = inv.invertMaxEnt(r_qu)
    i3 = inv.invertPolBasex(r_qu)

#   plt.plot(i1[0].T)
    plt.figure()
    plt.plot(i2[0].T)
    plt.show()
#   bsx = r.pBasex()
#   fold = vminv.pbsx2fold(bsx)
#   inv = vmp.unfold(fold, 1,1)
#   logplot(inv)
#    a = r.view(Frame)
#    a.interpol()
#    a.eval_polar()
