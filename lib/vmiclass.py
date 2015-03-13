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
mod_home = os.path.realpath(os.path.pardir)
stor_dir = os.path.join(mod_home, 'storage')
sys.path.insert(0, mod_home)

import numpy as np
import scipy as sp

import pylab as pl

# import scipy.fftpack as fft
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

    def cropsquare(self):
        cropd = vmp.centre_crop(self, self.cx, self.cy, self.rad_sq)
        return Frame(cropd)

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

    def evalrect(self, density=vmp.global_dens, displace=[0., 0.], phi=0):
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
#        self.rotateframe(self.offset + self.disp[0] + delta[0])
        self.evalrect(self.dens_rect, delta[1:], phi=delta[0])
        ft = fft.fft2(self.rect)
        return abs(ft.imag).sum()

    def __eval_sym2(self, delta):
        """ Use Bordas' criterion. I found it to be inferior """
#        self.rotateframe(self.offset + self.disp[0] + delta[0])
        self.evalrect(self.dens_rect, delta[1:], phi=0)
        rect_in = cp.copy(self.rect)
        Tstar = np.flipud(np.fliplr(rect_in))
        return -1 * np.sum(rect_in * Tstar)

    def __eval_sym3(self, delta):
        pinv = np.load('storage/inv-200-cos2.npy', mmap_mode='r')
        ab = np.load('storage/ab-200-cos2.npy', mmap_mode='r')
        self.rotateframe(self.offset + self.disp[0] + delta[0])
        self.evalrect(self.dens_rect, delta[1:])
        rfold = vmp.fold(self.rect, h=1)
        a_cos2 = np.dot(pinv.T, rfold.ravel())
        recon = np.dot(ab, a_cos2)
        return   np.linalg.norm(rfold.ravel() - recon)

    def find_centre(self, method=2):
        """ Iterate 'eval_sym' with a bound BFGS alg. verbosely ('disp') """
        init_vec = [0, 0, 0]
        meth_d = {1: self.__eval_sym, 2: self.__eval_sym2, 3: self.__eval_sym3}
        domain = np.tile([-90, 90], 3).reshape(3, 2)
        self.res = opt.minimize(meth_d[method], init_vec,
                init_vec,
                                method='L-BFGS-B', bounds=domain,#'L-BFGS-B'
                                tol=1E-5, options={'disp': True})
        if self.res.success:
            print 'Writing optimized centre and angular offset'
            self.disp += self.res.x
        # Final evaluation
        self.rotateframe(self.offset + self.disp[0])
        self.evalrect(self.dens_rect)

class RectImg(np.ndarray):
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

#   def pBasex(self):
#       fold = vmp.fold(self, 1, 1)
#       return vminv.invertPolBasex(fold)

#   def Basex(self):
#       frame = vmp.crop_circle(self, rmax=self.rad_sq)
#       bsx = vminv.invertBasex(frame)
#       return Frame(bsx)

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

class Plotter(object):
    """
    Docstring
    """
    def __init__(self):
        from vis import *


class Inverter(object):
    """
    Docstring
    """
    def __init__(self, rmax=250, sigma=1.00, n_even=50):
        self.n_funs = np.int(rad / sig)
        self.dim = rmax + 1
        self.lvals = n_even
        ext = '-' + str(r_max)+'-'+str(n_even)


if __name__ == '__main__':
    from vmivis import *
    t = RawImage(mod_home + '/ati-calibration.raw', xcntr=512, ycntr=465, radius=250)
    f = t.cropsquare()
    f.interpol()
    r = f.evalrect()
    bsx = r.pBasex()
    fold = vminv.pbsx2fold(bsx)
    inv = vmp.unfold(fold, 1,1)
    logplot(inv)
#    a = r.view(Frame)
#    a.interpol()
#    a.evalpolar()