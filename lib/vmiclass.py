# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 16:19:01 2015

@author: felix

Classes for VMI images

STATUS 2015-02-12
DONE:
TODO:

"""

import sys, os, warnings
mod_home = os.path.realpath(os.path.curdir)
stor_dir = os.path.join(mod_home, 'storage')
sys.path.insert(0, mod_home)

import numpy as np
import scipy as sp

import pylab as pl

import scipy.fftpack as fft
import scipy.ndimage as ndimg
import scipy.ndimage.interpolation as ndipol
import scipy.optimize as opt
import scipy.signal as sig
# import scipy.special as bessel
# import copy as cp
#
import proc as vmp
import inv as vminv
#import matplotlib.gridspec as gridspec
#from matplotlib.widgets import Slider, Button, RadioButtons
import vmimodules.conf
global_dens = vmimodules.conf.global_dens


class RawImage(np.ndarray):
    """
    VMI class for reading and manipulating single frames
    Contains methods for orienting and interpolating a single frame,
    also operator overloading for manipulations between images
    Inversion methods are called from an Inversion class object (TODO)
    """

    def __new__(self, file=[], xcntr=0, ycntr=0, radius=0, hotspots=[]):

        if type(file) is str:
            raw = vmp.rawread(file)
        else:
            raw = file

        self.cx, self.cy = xcntr, ycntr
        self.rad_sq = radius
        self.clrmap = pl.cm.gnuplot2

        if hotspots:
            raw = vmp.hot_spots(raw, hotspots)
        else:
            pass

        if not self.cx or not self.cy:
            self.cy, self.cx = (np.asarray(raw.shape) - 1) / 2
            warnings.warn(u'No valid center given. Using (%d, %d)' % (self.cx,
                          self.cy))

        if not radius:
            size = raw.shape
            dx, dy = np.min([self.cx, size[1] - self.cx - 1]), np.min([self.cy,
                        size[0] - self.cy - 1])
            self.rad_sq = np.min([dx, dy])
            warnings.warn(u'No valid radius given. Using %d' % (self.rad_sq))

        return np.ndarray.__new__(self, shape=raw.shape, dtype='int32',
                                  buffer=raw.copy(), order='C')

    def cropsquare(self, offset=0):
        cropd = vmp.centre_crop(self, self.cx, self.cy, self.rad_sq)
        return Frame(cropd, offset)

#==============================================================================

class Frame(np.ndarray):
    """
    Docstring
    """
    def __new__(self, frame, offs=0):

        self.cy, self.cx = (np.asarray(frame.shape) - 1) / 2
        size = frame.shape
        dx, dy = np.min([self.cx, size[1] - self.cx - 1]), np.min([self.cy,
                        size[0] - self.cy - 1])
        self.rad_sq = np.min([dx, dy])
        self.diam = size[0]
        self.offset = offs
        self.disp = [0.0, 0.0, 0.0]
        self.clrmap = pl.cm.gnuplot2

#       if do_intpol:
#           self.interpol(self)
#       This doesn't work, requires a Frame instance. __init__() breaks __new__

        return np.ndarray.__new__(self, shape=size, buffer=frame.copy().data,
                                  dtype=frame.dtype.name, order='C')
###

    def interpol(self, smooth=0.0):
        """ Cubic spline interpolation with zero smoothing """
        self.__ck = sig.cspline2d(self, smooth)

    def __rotateframe(self, phi=0):
        """ Rotate the coefficient matrix in degs, cutting off the corners """
        if not phi:
            self.ck = self.__ck
        else:
            self.ck = ndimg.rotate(self.__ck, phi,
                               reshape=False, prefilter=False)

    def evalrect(self, density=global_dens, displace=[0., 0.], phi=0):
        """ Project the image onto a rectangular grid with given spacing """
        self.__rotateframe(self.offset + self.disp[0] +  phi)
        coords = vmp.gen_rect(self.diam, density, self.disp[1:] + displace)
        rect = ndipol.map_coordinates(self.ck, coords, prefilter=False,
                                      output=np.float_)
        return RectImg(rect)

    def evalpolar(self, radN=251, polN=513):
        """ Project the image onto a polar grid with radial and polar denss."""
        self.__rotateframe(self.offset + self.disp[0])
        coords = vmp.gen_polar(self.rad_sq, radN, polN, self.disp[1:])
        polar = ndipol.map_coordinates(self.ck, coords, prefilter=False)
        return PolarImg(polar)

    def raddist(self):
        X = np.abs(np.arange(self.diam) - self.cx)
        radint = Frame(self * X)
        radint.interpol()
        polar = radint.evalpolar()
        return sp.integrate.romb(polar, axis=1)

#    def invertedpolar(self, radN, polN, inv='basex'):
#        """ Project the inverted image onto a polar grid."""
#        coords = vmp.gen_polar((self.dens_rect - 1) / 2, radN, polN, [0, 0, 0])
#        ck = sig.cspline2d(self.bsx, 0.0)
#        self.bsx_pol = ndipol.map_coordinates(ck, coords, prefilter=False)

### Finding the centre point and offset angle

    def __eval_sym(self, delta):
        """ Returns the total imaginary part of the 2D FFT of self.rect """
        delta[0] = 0.0
        rect_in = self.evalrect(global_dens, delta[1:], phi=delta[0])
        ft = fft.fft2(rect_in)
        return abs(ft.imag).sum()

    def __eval_sym2(self, delta):
        """ Use Bordas' criterion. I found it to be inferior """
        delta[0] = 0.0
        rect_in = self.evalrect(global_dens, delta[1:], phi=delta[0])
#       rect_in = self.rect.copy()
        Tstar = np.flipud(np.fliplr(rect_in))
        return -1 * np.sum(rect_in * Tstar)

    def __eval_sym3(self, delta, inv):
        rect_in = self.evalrect(301,  delta[1:], phi=delta[0])
        quads = vmp.quadrants(rect_in)
        pb = np.zeros((4, inv.FtF.shape[0]))
        for k, img in enumerate(quads):
            pb[k] = inv.invertPolBasex(img)
        dev = pb[:3].std(axis=0).sum()
        return dev


    def centre_pbsx(self, cntr=True, ang=True):
        """ Brute Force centering with the pBasex method """
        init_vec = [0, 0, 0]
        inv = vminv.Inverter(150, 8)
        domain = np.tile([-15, 15], 3).reshape(3, 2)
        if not cntr:
            domain[1:] = 0.0
        if not ang:
            domain[0] = 0.0
        self.res = opt.minimize(self.__eval_sym3, init_vec, args=inv,
                                method='L-BFGS-B', bounds=domain,#'L-BFGS-B'
                                tol=1E-5, options={'disp': True})
        if self.res.success:
            print 'Writing optimised centre and angular offset'
            self.disp += self.res.x
        del inv

    def find_centre(self, method=2, cntr=True, ang=True):
        """ Iterate 'eval_sym' with a bound BFGS alg. verbosely ('disp') 
            More or less deprecated """

        init_vec = [0, 0, 0]
        meth_d = {1: self.__eval_sym, 2: self.__eval_sym2}
        domain = np.tile([-10, 10], 3).reshape(3, 2)
        if not cntr:
            domain[1:] = 0.0
        if not ang:
            domain[0] = 0.0
        self.res = opt.minimize(meth_d[method], init_vec,
                                method='L-BFGS-B', bounds=domain,#'L-BFGS-B'
                                tol=1E-5, options={'disp': True})
        if self.res.success:
            print 'Writing optimized centre and angular offset'
            self.disp += self.res.x
        # Final evaluation
        self.__rotateframe(self.offset + self.disp[0])
        self.evalrect(global_dens)

class RectImg(Frame):
    """
    Docstring
    """
    def __new__(self, frame):

        self.cy, self.cx = (np.asarray(frame.shape) - 1) / 2
        size = frame.shape
        dx, dy = np.min([self.cx, size[1] - self.cx - 1]), np.min([self.cy,
                        size[0] - self.cy - 1])
        self.rad_sq = np.min([dx, dy])
        self.diam = size[0]
        self.clrmap = pl.cm.gnuplot2

        return np.ndarray.__new__(self, shape=size, dtype=np.float_,
                                  buffer=frame.copy().data, order='C')

    def __eval_bg_fac(self, fac, bg, inv):
        frame = Rect(self - fac * bg)

        quads = vmp.quadrants(frame)
        pb = np.zeros((4, inv.FtF.shape[0]))
        for k, img in enumerate(quads):
            pb[k] = inv.invertPolBasex(img)
        dev = pb[:3].std(axis=0).sum()
        return dev


    def find_bg_fac(self, bg):
        """ Subtract a background image with an optimised factor """
        init_vec = [0]
        inv = vminv.Inverter(150, 8)
        domain = [0, 2]
        self.bg_fac = opt.minimize(self.__eval_bg_fac, init_vec, args=(bg, inv),
                                method='L-BFGS-B', bounds=domain,#'L-BFGS-B'
                                tol=1E-5, options={'disp': True})
        if self.res.success:
            print 'Found optimum factor: ', self.bg_fac.x
        del inv
        return Rect(self - bg * self.bg_fac.x)

#==============================================================================

class PolarImg(np.ndarray):
    """
    Docstring
    """
    def __new__(self, frame):

        size = frame.shape
        self.cy, self.cx = (0, 0)
        drad, dphi = frame.shape
        self.clrmap = pl.cm.gnuplot2

        return np.ndarray.__new__(self, shape=size, dtype=np.float_,
                                  buffer=frame.copy().data, order='C')

#==============================================================================
#==============================================================================



if __name__ == '__main__':
    from vis import *
    t = RawImage(mod_home + '/ati-calibration.raw', xcntr=512, ycntr=465, radius=250)
    f = t.cropsquare()
    f.interpol()
    r = f.evalrect()
#   bsx = r.pBasex()
#   fold = vminv.pbsx2fold(bsx)
#   inv = vmp.unfold(fold, 1,1)
#   logplot(inv)
#    a = r.view(Frame)
#    a.interpol()
#    a.evalpolar()
