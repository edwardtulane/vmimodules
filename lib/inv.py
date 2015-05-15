# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 20:19:00 2014

@author: brausse
"""

import sys, os

import numpy as np
import scipy.special as spc
import scipy.integrate as integ
import proc as vmp

import vmimodules.conf
mod_home = vmimodules.conf.mod_home
stor_dir = os.path.join(mod_home, 'storage')

class Inverter(object):
    """
    Docstring
    """
    def __init__(self, r_max=250, n_even=16, dir=stor_dir, dryrun=0):
        self.__ext = '-'.join(('',  str(r_max), str(n_even))) #'-' + str(r_max)+'-'+str(n_even)
        if not dryrun:
            self.ab = np.load(stor_dir + '/ab' + self.__ext + '.npy')
            self.FtF = np.load(stor_dir + '/FtF' + self.__ext + '.npy')
            self.__M1, self.__M2 = vmp.iniBasex(stor_dir + '/')
            self.__MTM1, self.__MTM2 = np.dot(self.__M1.T, self.__M1), np.dot(
                    self.__M2.T, self.__M2),
        self.bs = np.load(stor_dir + '/bs' + self.__ext + '.npy')
        self.rf = np.load(stor_dir + '/rf' + self.__ext + '.npy')
        self.lvals = (n_even / 2) + 1
        self.n_funs = self.bs.shape[1] / self.lvals
        self.dim = r_max + 1
        self.__dim2 = np.arange(self.dim) ** 2
#       self.beta_vec = self.gen_beta_vec(self.lvals)
        self.th = np.linspace(0, 2*np.pi, 513)
#       self.th, self.lfuns = self.gen_lfuns(self.lvals)

#   set up a context manager?
    def __enter__(self):
        return self

    def __exit__(self):
        pass

#==============================================================================

    def invertFourierHankel(self, arr):
        dim = (arr.shape[1] - 1) /2
        arr._shift = np.append(arr.rect[:,dim:], arr.rect[:,:dim],axis=1)
        arr.fourt = fft.fft(arr._shift,axis=1)
        ft_freqs = fft.fftfreq(arr.fourt.shape[1])
    ###
        jn = bessel.jn_zeros(0, dim + 2)
        S, R1 = jn[-1], ft_freqs.max()
        R2 = S / (2 * np.pi * R1)
        print R1, R2, S
        jn = jn[:-1]
        F1_arg = jn / (2 * np.pi * R2)
        F1_arg *= (ft_freqs.shape[0] - 1) / (2 * R1)
        J1_vec = abs(bessel.j1(jn) ** -1)
        if not arr.__Cmn.shape == (dim + 1, dim + 1):
            arr.__jn_mat = (jn * jn[:, None]) / S
            arr.__J1_mat =  J1_vec * J1_vec[:, None]
            arr.__Cmn = (2 / S) * bessel.j0(arr.__jn_mat) * arr.__J1_mat
        else:
            pass

        F1 = np.empty((arr.fourt.shape[0], dim + 1), dtype='complex')
        arr.FHT = np.empty((arr.fourt.shape[0], dim + 1), dtype='complex')

        for i, line in enumerate(arr.fourt):
            ft_cR = sig.cspline1d(line.real)
            ft_cI = sig.cspline1d(line.imag)
            F1[i] = ( sig.cspline1d_eval(ft_cR, F1_arg) \
                            + 1j * sig.cspline1d_eval(ft_cI, F1_arg) \
                            ) * J1_vec * R1

        arr.FHT = np.dot(F1, arr.__Cmn)
        arr.FHT /= (R2 * J1_vec)
        arr.F2_arg = jn / (2 * np.pi * R1)
        arr.orig = np.dot(arr.FHT, arr.__Cmn)

#==============================================================================

    def invertMaxEnt(self, arr):
        arr_path = os.path.join(mod_home, 'inv')
        os.chdir(arr_path)
        np.savetxt('tmp_arr' , arr)

        os.system('./MEVIR.elf < tmp_arr')
        os.system('sed -e "s/D/e/g" -i MXLeg.dat')

        return (np.loadtxt('MXLeg.dat').T[1:], 
                np.loadtxt('MXmap.dat', delimiter=','), 
                np.loadtxt('MXres.dat', delimiter=',')
                )

    def invertBasex(self, arr):
        bsx, res = vmp.Basex(arr, 10, 0, self.__M1, self.__M2,
                            self.__MTM1, self.__MTM2)
        leg_p, leg_res = self.get_raddist(bsx)
        leg_p = np.array(leg_p).T * self.__dim2
        return leg_p[::2], bsx, res

    def invertPolBasex(self, arr, reg=1, get_pbsx=False):
            arr = arr.ravel()
            pbsx = np.dot(np.linalg.inv(self.FtF + reg * np.eye(self.FtF.shape[0])), 
                   np.dot(self.ab, arr))
            if get_pbsx:
                return pbsx

            leg = self.pbsx2rad(pbsx)
            leg *= self.__dim2
            pbsx = pbsx.ravel()
            inv_map = self.pbsx2fold(pbsx)
            res = arr - self.pbsx2ab(pbsx).ravel()
            res.shape = (self.dim, self.dim)
            return leg, inv_map, res

    def get_raddist(self, arr):
            from numpy.polynomial import Legendre as Leg
            fr = Frame(arr)
            fr.interpol()
            polar = fr.eval_polar(self.dim)
            polar = polar[:, :258]
#           polar *= self.__dim2[:, None]
            th = self.th[:258]
            leg, res = np.zeros((polar.shape[0], 9)), np.zeros(polar.shape[0])
            for i in xrange(polar.shape[0]):
                l, r = Leg.fit(th, polar[i] # * np.sin(th)
                        , 8,
                                       domain=[0, np.pi], full=True)
                leg[i], res[i] = l.coef, res[0]
            return leg, res

#==============================================================================

    def pbsx2fold(self, pbsx):
            fold = np.dot(pbsx, self.bs.T)
            fold.shape = (self.dim, self.dim)
            return fold

    def pbsx2ab(self, pbsx):
            fold = np.dot(pbsx, self.ab)
            fold.shape = (self.dim, self.dim)
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
        th = np.linspace(0, np.pi, 2**15+1)
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

    i1 = inv.invertBasex(r)
    i2 = inv.invertMaxEnt(r_qu)
    i3 = inv.invertPolBasex(r_qu)
#   bsx = r.pBasex()
#   fold = vminv.pbsx2fold(bsx)
#   inv = vmp.unfold(fold, 1,1)
#   logplot(inv)
#    a = r.view(Frame)
#    a.interpol()
#    a.eval_polar()
