# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 16:19:01 2015

@author: felix

Classes for VMI images

STATUS 2015-02-12
DONE:
TODO:

"""

import sys, os, warnings, re, time

import numpy as np
import scipy as sp
import pandas as pd

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
mod_home = vmimodules.conf.mod_home
stor_dir = os.path.join(mod_home, 'storage')

if 'GLOBAL_DENS' in os.environ:
    global_dens = int(os.environ['GLOBAL_DENS'])
else:
    global_dens = vmimodules.conf.global_dens

mm_to_fs = 6671.28190396304

class RawImage(np.ndarray):
    """
    VMI class for reading and manipulating single frames
    Contains methods for orienting and interpolating a single frame,
    also operator overloading for manipulations between images
    Inversion methods are called from an Inversion class object (TODO)

    Methods:
    -- crop_square: cuts out a square and may be supplied with the rotation angle
    """

    def __new__(self, file=[], xcntr=0, ycntr=0, radius=0, hotspots=[]):

        if type(file) is str:
            raw = vmp.rawread(file)
        else:
            raw = file

        self.cx, self.cy = xcntr, ycntr
        self.rad_sq = radius

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

    def crop_square(self, offset=0):
        cropd = vmp.centre_crop(self, self.cx, self.cy, self.rad_sq)
        return Frame(cropd, offset)

#==============================================================================

class Frame(np.ndarray):
    """
    Manipulation of square images incl. interpolation and rotation
    Methods:
    -- interpol
    -- eval_rect
    -- eval_polar
    -- centre_pbsx
    (-- rad_dist)
    (-- find_centre)
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

    def eval_rect(self, density=global_dens, displace=[0., 0.], phi=0):
        """ Project the image onto a rectangular grid with given spacing """
        self.__rotateframe(self.offset + self.disp[0] +  phi)
        coords = vmp.gen_rect(self.diam, density, self.disp[1:] + displace)
        rect = ndipol.map_coordinates(self.ck, coords, prefilter=False,
                                      output=np.float_)
        return RectImg(rect)

    def eval_polar(self, radN=251, polN=513):
        """ Project the image onto a polar grid with radial and polar denss."""
        self.__rotateframe(self.offset + self.disp[0])
        coords = vmp.gen_polar(self.rad_sq, radN, polN, self.disp[1:])
        polar = ndipol.map_coordinates(self.ck, coords, prefilter=False)
        return PolarImg(polar)

    def rad_dist(self):
        X = np.abs(np.arange(self.diam) - self.cx)
        radint = Frame(self * X)
        radint.interpol()
        polar = radint.eval_polar()
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
        rect_in = self.eval_rect(global_dens, delta[1:], phi=delta[0])
        ft = fft.fft2(rect_in)
        return abs(ft.imag).sum()

    def __eval_sym2(self, delta):
        """ Use Bordas' criterion. I found it to be inferior """
        delta[0] = 0.0
        rect_in = self.eval_rect(global_dens, delta[1:], phi=delta[0])
#       rect_in = self.rect.copy()
        Tstar = np.flipud(np.fliplr(rect_in))
        return -1 * np.sum(rect_in * Tstar)

    def __eval_sym3(self, delta, inv, dens):
        rect_in = self.eval_rect(501,  delta[1:], phi=delta[0])
        quads = vmp.quadrants(rect_in)
        pb = np.zeros((4, inv.FtF.shape[0]))
        for k, img in enumerate(quads):
            pb[k] = inv.invertPolBasex(img)
        dev = pb[:3].std(axis=0).sum()
        return dev


    def centre_pbsx(self, cntr=True, ang=False, dens=501):
        """ Brute Force centering with the pBasex method """
        init_vec = [0, 0, 0]
        inv = vminv.Inverter(250, 8)
        domain = np.tile([-15, 15], 3).reshape(3, 2)
        if not cntr:
            domain[1:] = 0.0
        if not ang:
            domain[0] = 0.0
        self.res = opt.minimize(self.__eval_sym3, init_vec, args=(inv, dens),
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
        self.eval_rect(global_dens)

#===============================================================================

class RectImg(Frame):
    """
    Processed image after interpolation and recasting.
    Methods:
    -- TODO: return (weighted) quadrant(s)
    -- find_bg_factor: early stage of bg subtraction improvement
    """
    def __new__(self, frame):

        size = frame.shape[-2:]
        self.cy, self.cx = (np.asarray(size) - 1) / 2
        dx, dy = np.min([self.cx, size[1] - self.cx - 1]), np.min([self.cy,
                        size[0] - self.cy - 1])
        self.rad_sq = np.min([dx, dy])
        self.diam = size[0]

        return np.ndarray.__new__(self, shape=frame.shape, dtype=np.float_,
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
    This is still a dummy for manipulations in polar coords.
    """
    def __new__(self, frame):

        size = frame.shape
        self.cy, self.cx = (0, 0)
        drad, dphi = frame.shape

        return np.ndarray.__new__(self, shape=size, dtype=np.float_,
                                  buffer=frame.copy().data, order='C')

#==============================================================================
#==============================================================================

# all the parameters that come to my mind. may be extended
header_keys = ['name', 'date', 'mode', 'seqNo', 
               'path', 'index', 'length', 'access', 'background']
time_keys = ['t start', 't end', 'delta t']
meta_keys = ['particle', 'Rep', 'Ext', 'MCP', 'Phos', 
                    'probe wavelength', 'pump wavelength', 
                    'molecule', 'acqNo', 'background']

frame_keys = ['center x', 'center y', 'offset angle', 'rmax',
              'mesh density', 'disp alpha', 'disp x', 'disp y']

center_keys = ['centering method', 'opt disp alpha', 'opt disp x', 'opt disp y', 'fun(min)']

inv_keys = ['inversion method', 'l max', 'odd l', 'sigma', 'total basis set size', 'RSS']

class ParseExperiment(object):

    def __init__(self, date, seqNo=None, inx=None, setup='tw', 
                 meta_dict={}, frame_dict={}):
        global header_keys, mm_to_fs
        vmi_dir = vmimodules.conf.vmi_dir

        self.date = pd.Timestamp(date)
        self.access = pd.Timestamp(time.asctime())
        self.seqNo, self.inx = seqNo, inx
        
        self.basedir = self.path = os.path.join(vmi_dir, setup, date)
        self.hdf = os.path.join(vmi_dir, 'procd', setup, date)

        if hasattr(frame_dict, 'mesh density'):
            self.dens = frame_dict['mesh density']
        else:
            frame_dict['mesh density'] = global_dens

        self.meta_dict, self.frame_dict = meta_dict, frame_dict

        if self.seqNo is None:
            if self.inx is None:
                raise Exception("Hand over indices if the measurement is not a sequence")
            self.mode = 'raw'
            self.index = inx[0]
            self.length = len(inx)
            self.name = '-'.join((date, 'raw', self.index))

            filelist = os.listdir(self.path)
            raws = [l for l in filelist if getint(l) in inx]
            raws.sort(key=getint)
            assert self.length == len(raws), 'Some rawfiles must be missing'

        else:
            self.mode = 'seq'
            seqdir = '-'.join((date, str(seqNo)))
            if setup == 'sf':
                seqdir = os.path.join('Seq', seqdir)
            self.path = os.path.join(self.path, seqdir)
            self.name = '-'.join((date, 'seq', str(seqNo)))

            filelist = os.listdir(self.path)
            regex = re.compile('raw$')
            raws = [l for l in filelist 
                    for m in [regex.search(l)] if m]
            raws.sort(key=getint)
            self.length = len(raws)

            metafile = os.path.join(self.basedir, '%s-%s.m' % (date, self.seqNo))
            sffile = os.path.join(self.path, 'StgPositions.npy')
            if os.path.exists(metafile):
                self.times = self.get_times(metafile)
            elif os.path.exists(sffile):
                self.times = np.load(sffile)
                self.times *= mm_to_fs
            else:
                self.times = np.linspace(time_dict['t start'],
                                         time_dict['t end'], time_dict['delta t'])

        self.hdf = os.path.join(self.hdf, self.name)
        hdf_dir = os.path.dirname(self.hdf)
        if not os.path.exists(hdf_dir):
            os.mkdir(hdf_dir)
        self.inx = raws

    def read_data(self):

        if not self.frame_dict.has_key('hot_spots'):
            self.frame_dict['hot_spots'] = []
        if not self.frame_dict.has_key('rmax'):
            self.frame_dict['rmax'] = 0
        if not self.frame_dict.has_key('offset angle'):
            self.frame_dict['offset angle'] = 0

        img, frames = {}, {}
        d = self.frame_dict
        for i in xrange(self.length):
            f = os.path.join(self.path, self.inx[i])
            img[i] = RawImage(f, d['center x'], d['center y'],
                              d['rmax'], d['hot_spots'])
            frames[i] = img[i].crop_square(d['offset angle'])

        self.raw_data = np.asarray(img.values())
        self.frames = np.asarray(frames.values())

    def get_header(self):
        
        header = {}
        for k in header_keys:
            if hasattr(self, k):
                header[k] = getattr(self, k)
            else:
                header[k] = None
        header.update(self.meta_dict)
        header.update(self.frame_dict)

        return pd.Series(header)

    def get_times(self, path):

        from StringIO import StringIO

        for line in open(path):
            if line.startswith('MBES_DELAY'):
                    line = line.split('[')[-1]
                    line = line.split(']')[0]
                    time_string = line
        s = StringIO(time_string)
        return np.loadtxt(s)

    def get_props(self):

        self.ints = self.frames.astype(np.float_).sum((-1, -2))
        dists = self.frames - self.frames.astype(np.float_).mean(0)
        self.diff = dists.sum((-1, -2))
        self.dist = (dists ** 2).sum((-1, -2))

        props = {'intensity': pd.Series(self.ints),
                 'difference':pd.Series(self.diff),
                 'distance' :pd.Series(self.dist)
                }

        return pd.DataFrame(props)

    def store(self):

        with pd.HDFStore('%s.h5' % self.hdf) as store:
            header = self.get_header()
            props = self.get_props()

            store['header'] = header
            store['props'] = props
            store['raw'] = pd.Panel(self.raw_data)
            store['frames'] = pd.Panel(self.frames)
            if hasattr(self, 'times'):
                store['times'] = pd.Series(self.times)

    def retrieve(self, key):

        with pd.HDFStore('%s.h5' % self.hdf) as store:
            return store['%s' % key]




def getint(name):
        basename = name.partition('.')[0]
        num = basename.split('-')[-1]
        num = num.split('q')[-1]
        return int(num)


if __name__ == '__main__':
    from vis import *
    t = RawImage(mod_home + '/ati-calibration.raw', xcntr=512, ycntr=465, radius=250)
    f = t.crop_square()
    f.interpol()
    r = f.eval_rect()
#   bsx = r.pBasex()
#   fold = vminv.pbsx2fold(bsx)
#   inv = vmp.unfold(fold, 1,1)
#   logplot(inv)
#    a = r.view(Frame)
#    a.interpol()
#    a.eval_polar()
