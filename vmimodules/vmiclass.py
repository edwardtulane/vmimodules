# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 16:19:01 2015

@author: felix

Classes for VMI images

STATUS 2015-02-12
DONE:
TODO:

"""
from __future__ import absolute_import, division, print_function, unicode_literals


import sys, os, warnings, re, time

import numpy as np
import scipy as sp

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize, SymLogNorm
from matplotlib.cm import gnuplot2 as gp


import scipy.fftpack as fft
import scipy.ndimage as ndimg
import scipy.integrate as integ
import scipy.ndimage as im
import scipy.ndimage.interpolation as ndipol
import scipy.optimize as opt
import scipy.signal as sig

from . import proc as vmp
from . import hitdet as hd
from . import inv as vminv
from .vis import Plotter
import seaborn.apionly as sb
sb.set_context('talk')

try:
    import bottleneck as bn
except ImportError as er:
    print('Could not find bottleneck. Using numpy instead.')
    bn = np
    def ss(arr):
        return np.power(arr, 2).sum()
    bn.ss = ss

try:
    from progressbar import ProgressBar
except ImportError as er:
    print('Could not find ProgressBar. Creating dummy instead.')
    class ProgressBar(object):
        def start(self): pass
        def update(self, *v): pass


# import vmimodules.conf
from . import mod_home, global_dens
# mod_home = vmimodules.conf.mod_home
stor_dir = os.path.join(mod_home, 'storage')

if 'GLOBAL_DENS' in os.environ:
    global_dens = int(os.environ['GLOBAL_DENS'])
else:
    global_dens = global_dens

mm_to_fs = 6671.28190396304

class RawImage(np.ndarray):
    """
    VMI class for reading and manipulating single frames
    Contains methods for orienting and interpolating a single frame,
    also operator overloading for manipulations between images

    Methods:
    -- crop_square: cuts out a square and may be supplied with the rotation angle
    """
    
    def __new__(self, file=[], xcntr=0, ycntr=0, radius=0, hotspots=[]):

        if type(file) is str:
            raw = vmp.rawread(file)
        else:
            raw = file

        if hotspots:
            raw = vmp.hot_spots(raw, hotspots)
        else:
            pass

        return np.ndarray.__new__(self, shape=raw.shape, dtype='int32',
                                  buffer=raw.copy(), order='C')

    def __init__(self, file=[], xcntr=0, ycntr=0, radius=0, hotspots=[]):

        self.cx, self.cy = xcntr, ycntr
        self.rad_sq = radius

        if not self.cx or not self.cy:
            self.cy, self.cx = (np.asarray(raw.shape) - 1) / 2
            warnings.warn('No valid center given. Using (%d, %d)' % (self.cx,
                          self.cy))

        if not radius:
            size = raw.shape
            dx, dy = np.min([self.cx, size[1] - self.cx - 1]), np.min([self.cy,
                             size[0] - self.cy - 1])
            self.rad_sq = np.min([dx, dy])
            warnings.warn('No valid radius given. Using %d' % (self.rad_sq))

    def crop_square(self, offset=0):
        cropd = vmp.centre_crop(self, self.cx, self.cy, self.rad_sq)
        return Frame(cropd, offset)

#==============================================================================

class Frame(np.ndarray):
    """
    Manipulation of square images incl. interpolation and rotation
    Arguments: frame array, offset angle
    Methods:
    -- interpol(sm)
    -- eval_rect(dens, disp, phi)
    -- eval_polar(radN, polN)
    -- centre_pbsx(cntr, ang, dens)
    deprecated:
    (-- rad_dist)
    (-- find_centre)
    """

    def __new__(self, frame, offs=0):

        size = frame.shape

        return np.ndarray.__new__(self, shape=size, buffer=frame.copy().data,
                                  dtype=frame.dtype.name, order='C')

    def __init__(self, frame, offs=0):

        size = frame.shape
        self.cy, self.cx = (np.asarray(frame.shape) - 1) / 2
        dx, dy = np.min([self.cx, size[1] - self.cx - 1]), np.min([self.cy,
                         size[0] - self.cy - 1])
        self.rad_sq = np.min([dx, dy])
        self.diam = size[0]
        self.offset = offs
        self.disp = np.array([0.0, 0.0, 0.0])

        self.interpol()

#       if do_intpol:
#       This doesn't work, requires a Frame instance. __init__() breaks __new__ (?)
#       Explanation: __init__ is called after __new__. The difference is that __new__ is
#       a static method of the class creating the instance, wheras __init__ initialises
#       the instance

###

    def interpol(self, smooth=0.0):
        """ Cubic spline interpolation with zero smoothing """
        self.ck = sig.cspline2d(self, smooth)

    def __rotateframe(self, phi=0):
        """ Rotate the coefficient matrix in degs, cutting off the corners 
            2015-12-09: This routine introduces an undesired offset! 
            DO NOT USE """
        if not phi:
            self.ck = self.__ck
        else:
            self.ck = ndimg.rotate(self.__ck, phi,
                                   reshape=False, prefilter=False)

    def eval_rect(self, density=global_dens, displace=[0., 0.], phi=0):
        """Deprecated. Superseded by eval_cart."""
        self.eval_rect(density, displace, phi)

    def eval_cart(self, density=global_dens, displace=[0., 0.], phi=0):
        """ Project the image onto a rectangular grid with given spacing """
        coords = vmp.gen_rect(self.diam, density, self.disp[1:] + displace, 
                              phi = self.offset + self.disp[0] + phi)
        rect = ndipol.map_coordinates(self.ck, coords, prefilter=False,
                                      output=np.float_)
        return CartImg(rect)

    def eval_polar(self, radN=251, polN=1025):
        """ Project the image onto a polar grid with radial and polar denss."""
        coords = vmp.gen_polar(self.rad_sq, radN, polN, self.disp[1:],
                              phi = self.offset + self.disp[0])
        polar = ndipol.map_coordinates(self.ck, coords, prefilter=False)
        return PolarImg(polar)

### Finding the centre point and offset angle

    def __sym_objective(self, disp):
        img = self.eval_cart(self.diam, disp[1:], phi=disp[0])
        img = vmp.crop_circle(img, (self.diam -1) / 2)
        qu = vmp.quadrants(img)
        qm = q[qsel].mean(0)
        return np.power(qm[None,:,:] - rect, 2).sum()


    def find_centre(self, cntr=True, ang=True):
        init_vec = [0, 0, 0]

        if not cntr:
            domain[1:] = 0.0
        if not ang:
            domain[0] = 0.0

        res = opt.minimize(self.__sym_objective, init_vec,
                           method='Nelder-Mead', #method='L-BFGS-B', bounds=domain,#'L-BFGS-B'
                           tol=1E-5, options={'disp': True})
        if res.success:
            print('Writing optimized centre and angular offset')
            return res
        else:
            warnings.warn('Centering failed!')

#===============================================================================

class CartImg(Frame):
    """
    Processed image after interpolation and recasting.
    Methods:
    -- TODO: return (weighted) quadrant(s)
    -- find_bg_factor: early stage of bg subtraction improvement
    """
    def __new__(self, frame):
        size = frame.shape[-2:]

        return np.ndarray.__new__(self, shape=frame.shape, dtype=np.float_,
                                  buffer=frame.copy().data, order='C')

    def __init__(self, frame):
        size = frame.shape[-2:]
        self.cy, self.cx = (np.asarray(size) - 1) / 2
        dx, dy = np.min([self.cx, size[1] - self.cx - 1]), np.min([self.cy,
                         size[0] - self.cy - 1])
        self.rad_sq = np.min([dx, dy])
        self.diam = size[0]

    def quadrants(self):
        return vmp.quadrants(self)

    def rad_dist(self, radN, polN=257, order=8):
        qu = vmp.quadrants(self)
        rd = np.zeros([4, (order/2)+1, radN])
        for i, q in enumerate(qu):
            rd[i] = vmp.get_raddist(q, radN, polN, order)
        return rd


    def __eval_bg_fac(self, fac, bg, inv):
        frame = self - fac * bg
        frame[frame < 0] = 0.

        quads = vmp.quadrants(frame)
        pb = np.zeros(quads.shape)
        for k, img in enumerate(quads):
            pb[k] = inv.invertMaxEnt(img)[2]
#       dev = bn.nansum(bn.nanstd(pb[:,0], axis=0))
        dev = bn.ss(pb)
#       ints = pb[:,:inv.n_funs]
#       dev += bn.ss(ints[ints < 0]) * inv.dim
        return dev


    def find_bg_fac(self, bg, dens=501):
        """ Subtract a background image with an optimised factor """
        init_vec = [0.5]
        rad = (dens - 1) / 2
        inv = vminv.Inverter(rad, 8, dryrun=0)
        domain = [[0, 2]]
        self.bg_fac = opt.minimize(self.__eval_bg_fac, init_vec, args=(bg, inv),
                                   method='L-BFGS-B', bounds=domain,#'L-BFGS-B'
                                   tol=1E-5, options={'disp': True})
        if self.bg_fac.success:
            print('Found optimum factor: ', self.bg_fac.x)
        del inv
        return self.bg_fac.x, self.bg_fac.fun

#==============================================================================

class PolarImg(np.ndarray):
    """
    This is still a dummy for manipulations in polar coords.
    """
    def __new__(self, frame):

        size = frame.shape

        return np.ndarray.__new__(self, shape=size, dtype=np.float_,
                                  buffer=frame.copy().data, order='C')
    def __init__(self, frame):
        self.cy, self.cx = (0, 0)
        drad, dphi = frame.shape



#==============================================================================
#==============================================================================

# all the parameters that come to my mind. may be extended
header_keys = ['name', 'date', 'setup', 'status', 'kind', 'seqNo', 
               'index', 'length', 'access', 'background', 'path', 'hdf']
time_keys = ['t start', 't end', 'delta t']
meta_keys = ['particle', 'Rep', 'Ext', 'MCP', 'Phos', 
             'probe wavelength', 'pump wavelength', 
             'probe power', 'pump power', 'pump pol',
             'molecule', 'fragment', 'charge', 'dilution', 'acqNo', 'mode',
             'gate', 'valve', 'p.w.']

frame_keys = ['center x', 'center y', 'offset angle', 'rmax',
              'mesh density', 'disp alpha', 'disp x', 'disp y', 'hot_spots']

center_keys = ['centering method', 'fun(min)', 'opt disp alpha', 'opt disp x', 'opt disp y', 'bg fac med']

inv_keys = ['inversion method', 'l max', 'odd l', 'sigma', 'total basis set size', 'mask']

singleshot_keys = ['comp_cntr', 'comp_strgth', 'no_levels', 'rmax_imax']

#==============================================================================

class CommonMethods(object):
    """ """
    def retrieve(self, key):

        with pd.HDFStore('%s.h5' % self.hdf) as store:
            return store['%s' % key]

    def push_fig(self, tag='frame', mode='lin'):
        """ """
        plt.savefig('%s/%s-%s-%s.png' % (self.hdf, self.name, tag, mode),
                    bbox_inches='tight')
        plt.close()

    def get_times(self, path):

        try:
            from io import StringIO
        except ImportError as er:
            from StringIO import StringIO

        for line in open(path):
            if line.startswith('MBES_DELAY'):
                    line = line.split('[')[-1]
                    line = line.split(']')[0]
                    time_string = line
        time_string = time_string.replace(',', '.')
        s = StringIO(time_string)
        return np.loadtxt(s)

#==============================================================================

class ParseExperiment(CommonMethods):

    def __init__(self, date, seqNo=None, inx=None, setup='tw', 
                 meta_dict={}, frame_dict={}, time_dict={}):
        global header_keys, meta_keys, frame_keys, time_keys, mm_to_fs
        vmi_dir = vmimodules.conf.vmi_dir

        self.date, self.setup = pd.Timestamp(date), setup
        self.access = pd.Timestamp(time.asctime())
        self.seqNo, self.inx = seqNo, inx
        self.particle = meta_dict['particle'][:3]

        if setup == 'fel':
            ext_date = '-'.join((date, self.particle))
        else:
            ext_date = date
        
        self.basedir = self.path = os.path.join(vmi_dir, setup, ext_date)
        self.hdf = os.path.join(vmi_dir, 'procd', setup, ext_date)
        self.cols = header_keys + meta_keys + frame_keys
        
        self.pl = Plotter()

        if 'mesh density' in frame_dict:
            self.dens = frame_dict['mesh density']
        else:
            frame_dict['mesh density'] = global_dens

        self.meta_dict, self.frame_dict = meta_dict, frame_dict

        if self.seqNo is None:
            if self.inx is None:
                raise Exception("Hand over indices if the measurement is not a sequence")
            self.kind = 'raw'
            self.index = inx[0]
            self.length = len(inx)
            self.name = '-'.join((date, 'raw', '%02d' % self.index))

            filelist = os.listdir(self.path)
            raws = [l for l in filelist if getint(l) in inx]
            regex = re.compile('raw$')
            raws = [l for l in raws 
                    for m in [regex.search(l)] if m]
            raws.sort(key=getint)
            self.length = len(raws)
            try:
                assert self.length == len(raws)
            except:    
                print('Some rawfiles must be missing')

        else:
            self.kind = 'seq'
            seqdir = '-'.join((date, str(seqNo)))
            if setup == 'sf':
                seqdir = os.path.join('Seq', seqdir)
            self.path = os.path.join(self.path, seqdir)
            self.name = '-'.join((date, 'seq', '%02d' % seqNo))

            filelist = os.listdir(self.path)
            regex = re.compile('raw$')
            raws = [l for l in filelist 
                    for m in [regex.search(l)] if m]
            raws.sort(key=getint)
            self.length = len(raws)

            if setup == 'sf':
                metadir = os.path.join(self.basedir, 'Seq')
            else:
                metadir = self.basedir
            metafile = os.path.join(metadir, '%s-%s.m' % (date, self.seqNo))
            sffile = os.path.join(self.path, 'TgtPositions.npy')
            if os.path.exists(metafile):
                times = self.get_times(metafile)
#               self.times = pd.MultiIndex.from_arrays([np.arange(self.length),
#                                                       times])
            elif os.path.exists(sffile):
                times = np.load(sffile)
                times *= mm_to_fs
                self.times = pd.MultiIndex.from_arrays([np.arange(self.length),
                                                        times])
            else:
                self.times = np.linspace(time_dict['t start'],
                                         time_dict['t end'], time_dict['delta t'])
            self.cols += time_keys

        hdf_dir = self.hdf
        self.hdf = os.path.join(self.hdf, self.name)
        if not os.path.exists(hdf_dir):
            os.mkdir(hdf_dir)
        if not os.path.exists(self.hdf):
            os.mkdir(self.hdf)
        self.inx = raws
        self.status = 'raw'

    def read_data(self):

        if not 'hot_spots' in self.frame_dict:
            self.frame_dict['hot_spots'] = []
        if not 'rmax' in self.frame_dict:
            self.frame_dict['rmax'] = 0
        if not 'offset angle' in self.frame_dict:
            self.frame_dict['offset angle'] = 0

        img, frames = {}, {}
        d = self.frame_dict
        for i in range(self.length):
            f = os.path.join(self.path, self.inx[i])
            img[i] = RawImage(f, d['center x'], d['center y'],
                              d['rmax'], d['hot_spots'])
            frames[i] = img[i].crop_square(d['offset angle'])

        self.raw_data = pd.Panel(img)
        self.frames = pd.Panel(frames)

    def get_header(self):
        
        header = {}
        for k in header_keys:
            if hasattr(self, k):
                header[k] = getattr(self, k)
            else:
                header[k] = None
        header.update(self.meta_dict)
        header.update(self.frame_dict)

        if hasattr(self, 'times'):
            tmin, tmax = self.times.levels[-1].min(), self.times.levels[-1].max()
            header['t start'] = tmin
            header['t end'] = tmax
            header['delta t'] = (tmax - tmin) / np.float(self.length)

        return pd.Series(header)

    def get_props(self):

        self.ints = bn.nansum(self.frames.astype(np.float_), axis=(-1, -2))
        self.meds = bn.nanmedian(self.frames.astype(np.float_), axis=(-1, -2))
        dists = self.frames.values - bn.nanmean(self.frames.astype(np.float_),
                axis=0)
        self.dist = bn.ss(dists, axis=(-1, -2))

        if self.meta_dict['mode'] == 'counting':
            f = self.frames.values
            for i in range(f.shape[0]):
                v = f[i]
                v = v[v >0]
                self.frames[i] /= np.float(v.min())
        else:
            nrm = np.prod(self.frames[0].shape) * 100
            self.frames = pd.Panel(self.frames.values / self.ints[:, None, None] * nrm)
        dists = self.frames.values - bn.nanmean(self.frames, axis=0)
        self.norm_dist = bn.ss(dists, axis=(-1, -2))

        props = {'intensity': pd.Series(self.ints),
                 'distance': pd.Series(self.dist),
                 'norm distance': pd.Series(self.norm_dist),
                 'median': pd.Series(self.meds)
                }
        props = pd.DataFrame(props)
        if hasattr(self, 'times'):
            props.index = self.times

        return props[['intensity', 'distance', 'norm distance', 'median']]

    def store(self):

        with pd.HDFStore('%s.h5' % self.hdf) as store:
            header = self.get_header()
            props = self.get_props()

            store['header'] = header[self.cols]
            store['props'] = props
#           raws = pd.Panel(self.raw_data)
            frames = self.frames

            if hasattr(self, 'times'):
#               raws.items = self.times
                frames.items = self.times

            store['frames'] = frames
#           store.append('frames', frames)

            fr_mean = bn.nanmean(self.frames, 0)
            with sb.axes_style('white'):
                vmax = np.percentile(fr_mean, 99)
                f, a = self.pl.linplot(fr_mean, vmax=vmax)
                self.push_fig(mode='lin')
                f, a = self.pl.logplot(fr_mean, vmax=vmax)
                self.push_fig(mode='log')

            if self.length > 1:
                for k, v in props.items(): # should be backwards compatible to Python 2
                    fig, axs = plt.subplots(2, 1, sharex=True)
                    axs[0].set_title = '%s distribution' % k
                    sb.kdeplot(v.values, ax=axs[0])
                    sb.boxplot(v.values, ax=axs[1], vert=False)
                    plt.draw()
                    self.push_fig(tag=k, mode='dist')

                sb.jointplot(props['norm distance'], props['intensity'],
                            kind='kde', stat_func=None)
                self.push_fig(tag='int', mode='corr')



#==============================================================================

class ProcessExperiment(CommonMethods):
    """ """
    def __init__(self, date, seqNo=None, index=None, setup='tw',
                 cpi=[0, 0, 0],
                 center_dict={}, inv_dict={}, particle=None):

        global header_keys, meta_keys, frame_keys
        global center_keys, inv_keys
        vmi_dir = vmimodules.conf.vmi_dir

        self.center_dict, self.inv_dict = center_dict, inv_dict

        self.access, self.cpi = pd.Timestamp(time.asctime()), cpi[:]
        self.seqNo, self.index = seqNo, index

        if setup == 'fel' and particle:
            particle = particle[:3]
            ext_date = '-'.join((date, particle))
        else:
            ext_date = date
        
        self.hdf = os.path.join(vmi_dir, 'procd', setup, ext_date)
        self.cols = header_keys + meta_keys + frame_keys

        self.pl = Plotter()

        if seqNo is None:
            if index is None:
                raise Exception("Hand over first index if the measurement \
                                 is not a sequence")
            self.mode = 'raw'
            self.name = '-'.join((date, 'raw', '%02d' % self.index))

        else:
            self.mode = 'seq'
            self.name = '-'.join((date, 'seq', '%02d' % self.seqNo))

        self.hdf_dir = self.hdf

        self.hdf_in = os.path.join(self.hdf, self.name)
        self.hdf_in = self.__get_ext(self.hdf_in, cpi)

        self.__recover_data()


    def __get_ext(self, path, cpi):
        flags = ['c', 'p', 'i']
        z = zip(flags, cpi)
        ext = ['%s%02d' % (f, i) for (f, i) in z if i]
        ext.insert(0, path)
        return '-'.join(ext)

    def __propagate_ext(self, ix, overwrite):
        self.cpi[ix] += 1

        self.hdf = os.path.join(self.hdf, self.name)
        self.hdf = self.__get_ext(self.hdf, self.cpi)

        if not overwrite:
            while os.path.exists('%s.h5' % self.hdf):
                self.cpi[ix] += 1
                self.hdf = os.path.join(self.hdf_dir, self.name)
                self.hdf = self.__get_ext(self.hdf, self.cpi)

        if not os.path.exists(self.hdf):
            os.mkdir(self.hdf)

    def __recover_data(self):
        with pd.HDFStore('%s.h5' % self.hdf_in) as store:
            self.header = store['header']
            self.data = store['frames']
            self.times = self.data.items

            if hasattr(store, 'bg_fac'):
                self.bg_fac = store['bg_fac'].values


    def __update_header(self):
        for k in self.cols:
            if hasattr(self, k):
                self.header[k] = getattr(self, k)

    def center(self, cntr=True, ang=False, overwrite=True):
        h = self.header
        dens, offs = h['mesh density'], h['offset angle']
        length = h['length']

        self.opt = np.zeros((length, 4))
        self.bg_fac = np.zeros((length, 2))

        pbar = ProgressBar().start()
        pbar.maxval = length
        for i in range(length):
            fr = Frame(self.data.values[i], offs)
            if hasattr(self, 'bg'):
                fr.interpol()
                r = fr.eval_rect(151)
                if not isinstance(self.bg, Frame):
                    bg = Frame(self.bg, offs)
                    bg.interpol()
                bg = bg.eval_rect(151)
                self.bg_fac[i] = r.find_bg_fac(bg, 151)
                fr = Frame(self.data.values[i] - self.bg_fac[i,0] * self.bg, offs)
#           fr.interpol()
            fr.centre_pbsx(cntr, ang, dens)
            self.opt[i,0], self.opt[i,1:] = fr.res.fun, fr.res.x
            pbar.update(i)

        self.header['disp alpha'], self.header[
                'disp x'], self.header['disp y'] = np.median(self.opt, axis=0)[1:]
        self.header['bg fac med'] = np.median(self.bg_fac[:,0])
        self.__propagate_ext(0, overwrite)
        self.status = 'centered'
        self.opt = zip(center_keys[1:], self.opt.T)
        self.opt = dict(self.opt)
        self.frames = self.data.values

        o = pd.DataFrame(self.opt)
        if length > 1:
            sb.kdeplot(o['opt disp x'], o['opt disp y'], shade=True, cmap='Blues')
            plt.scatter(self.header['disp x'], self.header['disp y'], 80, 'r', 'x')
            self.push_fig(tag='centre', mode='kde')
            if hasattr(self, 'bg'):
                fig, axs = plt.subplots(2, 1, sharex=True)
                axs[0].set_title = 'background factor distribution'
                sb.kdeplot(self.bg_fac[:,0], ax=axs[0])
                sb.boxplot(self.bg_fac[:,0], ax=axs[1], vert=False)
                plt.draw()
                self.push_fig(tag='bg-fac', mode='dist')

                sb.kdeplot(self.bg_fac[:,0], self.bg_fac[:,1], shade=True, cmap='Blues')
                self.push_fig(tag='bg_fac', mode='corr')

    def process(self, overwrite=True, crop=0):
        h = self.header
        dens, offs = h['mesh density'], h['offset angle']
        rmax = (dens - 1) / 2
        disp = [h['disp alpha'], h['disp x'], h['disp y']]
        length = h['length']

        self.frames = np.zeros((length, dens, dens))
        self.intmap = np.zeros((length, rmax + 1))

        self.inv = vminv.Inverter(rmax, 8)

        pbar = ProgressBar().start()
        pbar.maxval = length

        radint =  np.linspace(0, 1, self.inv.dim) ** 2

        for i in range(length):
            if hasattr(self, 'bg'):
                fr = Frame(self.data.values[i] - self.bg_fac[i,0] * self.bg, offs)
            else:
                fr = Frame(self.data.values[i], offs)
            fr.disp = disp
#           fr.interpol()
            r = fr.eval_rect(dens)
            if crop:
                rect = vmp.crop_circle(r, crop)
            self.frames[i] = r
            self.intmap[i] = self.inv.get_raddist(fr)[0] / radint
            pbar.update(i)

        self.__propagate_ext(1, overwrite)
        self.status = 'processed'

        fr_mean = bn.nanmean(self.frames, 0)
        self.intmap = np.nan_to_num(self.intmap)

        with sb.axes_style('white'):
            vmax = np.percentile(fr_mean, 99)
            f, a = self.pl.linplot(fr_mean, vmax=vmax)
            self.push_fig(mode='lin')
            f, a = self.pl.logplot(fr_mean + 1E-5, vmax=vmax)
            self.push_fig(mode='log')

            if length > 1:
                intmax = np.percentile(self.intmap, 99)
                self.pl.linplot(self.intmap, aspect='auto', vmax=intmax)
                self.push_fig(tag='intmap', mode='lin')
                sb.tsplot(self.intmap, err_style=['unit_traces', 'ci_band'],
                          err_palette='YlOrRd_d')
                self.push_fig(tag='intmap', mode='ts')
                self.pl.logplot(self.intmap + 1E-5, aspect='auto', vmax=intmax)
                self.push_fig(tag='intmap', mode='log')
            else:
                sb.plt.plot(self.intmap[0])
                self.push_fig(tag='raddist', mode='lin')

            self.pl.plotCentre(bn.nanmean(self.frames, 0), vmax=vmax)
            self.push_fig(tag='centre', mode='crosshair')

    def invert(self, overwrite=True):
        h = self.header
        dens, length = h['mesh density'], h['length']
        rmax = (dens - 1) / 2
        self.inv = vminv.Inverter(rmax, self.inv_dict['l max'])
        dim = self.inv.dim

        methods = {'pBasex': (self.inv.invertPolBasex, [4, self.inv.lvals, dim], 
                              [4, dim, dim]),
                   'MaxEnt': (self.inv.invertMaxEnt, [4, 4, dim], 
                             [ 4, dim, dim]),
                   'Basex' : (self.inv.invertBasex, [1, 5, dim], 
                              [1, dens, dens])
                  }

        m = methods[self.inv_dict['inversion method']]
        self.leg = np.zeros([length] + m[1])
        self.inv_map = np.zeros([length] + m[2])
        self.inv_res = self.inv_map.copy()

        if m[2][0] > 0:
            prep = vmp.quadrants
        else:
            prep = lambda a: a
        pbar = ProgressBar().start()
        pbar.maxval = length
        for i in range(length):
            rect = self.data.values[i]
#           rect *= 1000
            rect = prep(rect)
            for j in range(rect.shape[0]):
                self.leg[i,j], self.inv_map[i,j], self.inv_res[i,j] = m[0](rect[j])
            pbar.update(i)

        self.__propagate_ext(2, overwrite)
        self.status = 'inverted'
        self.cols += inv_keys
        self.header = self.header.append(pd.Series(self.inv_dict))

        fr_mean = bn.nanmean(self.inv_map, (0))
        fr_mean = vmp.compose(fr_mean)
        vmax = np.percentile(fr_mean, 99)
        res_mean = bn.nanmean(self.inv_res, (0))
        res_mean = vmp.compose(res_mean)
        resmax = np.percentile(res_mean, 99)
        rss = bn.ss(self.inv_res, (2,3)) / (self.inv.dim ** 2)
        leg_mean = bn.nanmean(self.leg, axis=1)

        with sb.axes_style('white'):
            f, a = self.pl.linplot(fr_mean, vmax=vmax)
            self.push_fig(mode='lin')
            f, a = self.pl.logplot(fr_mean.clip(0) + 1E-5, vmax=vmax)
            self.push_fig(mode='log')

            f, a = self.pl.lindiff(res_mean, vsym=resmax)
            f.colorbar(a)
            self.push_fig(tag='res', mode='lin')
            
            if length > 1:
                vmax = np.percentile(leg_mean[:,0], 99)
                self.pl.linplot(leg_mean[:,0], vmax=vmax, aspect='auto')
                self.push_fig(tag='intmap', mode='lin')
                self.pl.lindiff(leg_mean[:,1], vsym=vmax, aspect='auto')
                self.push_fig(tag='beta', mode='lin')

                fig, axs = plt.subplots(2, 1, sharex=True)
                sb.tsplot(leg_mean[:,0], err_style=['unit_traces', 'ci_band'],
                          err_palette='YlOrRd_d', ax=axs[0])
                sb.tsplot(leg_mean[:,1], err_style=['unit_traces', 'ci_band'],
                          err_palette='YlOrRd_d', ax=axs[1])
                self.push_fig(tag='dist', mode='ts')

            else:
                f = plt.figure()
                f.add_subplot(211)
                sb.plt.plot(leg_mean[0,0])
                f.add_subplot(212)
                sb.plt.plot(leg_mean[0,1])
#               self.push_fig(tag='beta', mode='lin')
                self.push_fig(tag='raddist', mode='lin')

        if length > 1:
            sb.violinplot(bn.nansum(self.inv_map, axis=(2,3)), names=np.arange(4)+1)
            self.push_fig(tag='int', mode='grouped')
            sb.violinplot(rss, names=np.arange(4)+1)
            self.push_fig(tag='rss', mode='grouped')


    def store(self):
        with pd.HDFStore('%s.h5' % self.hdf) as store:
            prop_1D = ['opt', 'bg_fac']
            prop_2D = ['intmap', 'betamap']
            prop_3D = ['res', 'frames']
            prop_4D = ['inv_map', 'inv_res', 'leg']

            self.__update_header()
            store['header'] = self.header[self.cols]


            for k in prop_1D:
                if hasattr(self, k):
                    v = pd.DataFrame(getattr(self,k))
                    if hasattr(self, 'times'): v.index = self.times
                    store[k] = v

            for k in prop_2D:
                if hasattr(self, k):
                    v = pd.DataFrame(getattr(self,k))
                    if hasattr(self, 'times'): v.index = self.times
                    store[k] = v

            for k in prop_3D:
                if hasattr(self, k):
                    v = pd.Panel(getattr(self,k))
                    if hasattr(self, 'times'): v.items = self.times
                    store[k] = v

            for k in prop_4D:
                if hasattr(self, k):
                    v = pd.Panel4D(getattr(self,k))
                    if hasattr(self, 'times'): v.labels = self.times
                    store.append(k, v)

#==============================================================================
#==============================================================================

if os.system('ps -aux | grep ipyparall | grep -v grep') == 0:
    from ipyparallel import Client, interactive
    cl = Client(timeout=2)
    view = cl[:]

    @view.parallel()
    @interactive
    def hitfind(img):
        from vmimodules import detect_hits_img
        sgl, mlt, ot, = detect_hits_img(img, comp_c, comp_s, levels=levels, thr=thr,
                                        imax=i_max, dilate=True)
        return sgl, mlt, ot

    @view.parallel()
    @interactive
    def hitglob(img):
        from vmimodules import detect_hits_img
        hits = detect_hits_img(img, comp_c, comp_s,
                            levels=levels, imax=i_max, dilate=True, global_analysis=True)
        return hits

else:
    print('No ipyparallel client found. Running in serial execution.')
    view = None

#==============================================================================

class ParseSingleShots(CommonMethods):

    def __init__(self, date, seqNo=None, inx=None, setup='fel', 
                 meta_dict={}, singleshot_dict={}, time_dict={}):
#       global header_keys, meta_keys, frame_keys, time_keys, mm_to_fs

        vmi_dir = vmimodules.conf.vmi_dir
        self.date, self.setup = pd.Timestamp(date), setup
        self.access = pd.Timestamp(time.asctime())
        self.seqNo, self.inx = seqNo, inx
        
        self.particle = meta_dict['particle'][:3]
        self.comp_c, self.comp_s = singleshot_dict['comp_cntr'], singleshot_dict['comp_strgth']
        self.no_levels = singleshot_dict['no_levels']

        self.basedir = self.path = os.path.join(vmi_dir, setup, '-'.join((date, self.particle)))
        self.hdf = os.path.join(vmi_dir, 'procd', setup, '-'.join((date, self.particle)))
        self.cols = header_keys + meta_keys + frame_keys
        
        self.meta_dict, self.singleshot_dict = meta_dict, singleshot_dict

        if self.seqNo is None:
            if self.inx is None:
                raise Exception("Hand over indices if the measurement is not a sequence")
            self.kind = 'raw'
            self.index = inx[0]
            self.name = '-'.join((date, 'raw', '%02d' % self.index))

            filelist = os.listdir(self.path)
            raws = [l for l in filelist if getint(l) in inx]
            regex = re.compile('raw.sss$')
            raws = [l for l in raws 
                    for m in [regex.search(l)] if m]
            raws.sort(key=getint)
            self.length = len(raws)


        else:
            self.kind = 'seq'
            seqdir = '-'.join((date, str(seqNo)))
            self.path = os.path.join(self.path, seqdir)
            self.name = '-'.join((date, 'seq', '%02d' % seqNo))

            filelist = os.listdir(self.path)
            regex = re.compile('raw.sss$')
            raws = [l for l in filelist 
                    for m in [regex.search(l)] if m]
            raws.sort(key=getint)
            self.length = len(raws)

            metadir = self.basedir
            metafile = os.path.join(metadir, '%s-%s.m' % (date, self.seqNo))

            if os.path.exists(metafile):
                self.times = self.get_times(metafile)
#               self.times = pd.MultiIndex.from_arrays([np.arange(self.length),
#                                                       times])
            self.cols += time_keys

        hdf_dir = self.hdf
        self.hdf = os.path.join(self.hdf, self.name)
        if not os.path.exists(hdf_dir):
            os.mkdir(hdf_dir)
#       if not os.path.exists(self.hdf):
#           os.mkdir(self.hdf)
        self.inx = raws
        self.status = 'raw'

    def detect_hits(self):

        if not hasattr(self, 'frames'):
            self.read_data()
  
        maxpix = np.zeros_like(self.frames[0][1], dtype=np.float_)

        pbar = ProgressBar().start()
        pbar.maxval = self.dimd

        for i in range(self.dimd):
            np.max([maxpix, self.frames[i][1]], axis=0, out=maxpix)
            pbar.update(i)

        print('Normalisation finished')

        self.i_max = im.percentile_filter(maxpix, 95, (70,70))
        rmax = self.singleshot_dict['rmax_imax']
        self.i_max = vmp.crop_circle(self.i_max, rmax)
        
        self.levels = np.linspace(0, 1, self.no_levels + 1)[:-1]
        chc = np.random.choice(self.dimd, 100, replace=False)
        glob_ana, locl_ana = list(), list()

        if view is None:
            self.parallel = False

            pbar = ProgressBar().start()
            pbar.maxval = 100

            for i, img in enumerate(self.frames[chc]):
                glob_ana.append(hd.detect_hits_img(img, self.comp_c, self.comp_s,
                                levels=self.levels, imax=self.i_max, dilate=True, global_analysis=True))
                pbar.update(i)
                
        else:
            self.parallel = True

            view['comp_c'], view['comp_s'] = self.comp_c, self.comp_s
            view['i_max'] = self.i_max
            view['levels'] = self.levels

            res = hitglob.map(self.frames[chc][1])
            res.wait()
#           glob_ana = res.result
            print('Global analysis finished. Took %i seconds.' % (res.wall_time))

#       hitdist = pd.concat([v[0] for v in glob_ana])
        cmpdist = pd.concat(res.result)

#       quants = np.linspace(0, 1, self.no_levels + 1)[:-1]
#       self.levels = np.percentile(hitdist, 100 * quants)

        self.thr = hd.find_otsus_thr(cmpdist)
        print("Otsu's threshold used: ", self.thr)

#       del hitdist, cmpdist
        cl.purge_results('all')

        if view is None:
            print('Starting hit detection in serial execution.')
            pbar = ProgressBar().start()
            pbar.maxval = self.dimd

            for i, img in enumerate(self.frames):
                locl_ana.append(hd.detect_hits_img(img, self.comp_c, self.comp_s, thr=self.thr,
                                levels=self.levels, imax=self.i_max, dilate=True))
                pbar.update(i)

            self.sgl = pd.Panel({i: r[0] for i,r in enumerate(locl_ana)}, dtype=np.float_)
            self.mlt = pd.Panel({i: r[1] for i,r in enumerate(locl_ana)}, dtype=np.float_)

        else:
            print('Starting hit detection in parallel mode running on %i cores.' % len(view))
            view['levels'] = self.levels
            view['thr'] = self.thr

            ind = np.arange(self.dimd)
            no_chunks = self.dimd / 1200
            chunks = np.split(ind, 1200 * (np.arange(no_chunks) + 1))
            
            for i, chunk in enumerate(chunks):
                id, ar = self.frames[chunk]
                res = hitfind.map(ar)
                res.wait()

#               locl_ana = res.result
                sgl = pd.Panel({i: r[0] for i,r in enumerate(res.result)}, dtype=np.float_)
                mlt = pd.Panel({i: r[1] for i,r in enumerate(res.result)}, dtype=np.float_)

                sgl.items, mlt.items = id, id 

                with pd.HDFStore('%s-ss.h5' % self.hdf) as store:
                    store['sgl%02d' % i] = sgl
                    store['mlt%02d' % i] = mlt
#                   print(store.get('mlt%02d' % i ) )
                
                print('%i images processed. Last chunk took %.1f minutes.' % (chunk[-1] + 1, (res.wall_time / 60)))
                cl.purge_results('all')

#       del self.frames

    def read_data(self):

        flist = [os.path.join(self.path, ix) for ix in self.inx]
        self.frames = vmp.SingleShotExtractor(flist)
        self.dimd = len(self.frames)
#       if self.length > 1:
#           for i in range(self.length):
#               f = os.path.join(self.path, self.inx[i])
#               pid, ss_img = vmp.read_singleshots(f)

#               if not hasattr(self, 'frames'):
#                   self.dimd, dimy, dimx = ss_img.shape
#                   self.frames = np.zeros((self.length * self.dimd, dimy, dimx), dtype=ss_img.dtype)
#                   self.pid_ar = np.zeros((self.length * self.dimd), dtype=np.int_)

#               self.frames[i * self.dimd : (i+1) * self.dimd] = ss_img
#               self.pid_ar[i * self.dimd : (i+1) * self.dimd] = pid

#       else:
#           f = os.path.join(self.path, self.inx[0])
#           self.pid_ar, self.frames = vmp.read_singleshots(f)
#           self.dimd = self.frames.shape[0]


    def get_header(self):
        
        header = {}
        for k in header_keys:
            if hasattr(self, k):
                header[k] = getattr(self, k)
            else:
                header[k] = None
        header.update(self.meta_dict)

        if hasattr(self, 'times'):
            tmin, tmax = self.times.min(), self.times.max()
            header['t start'] = tmin
            header['t end'] = tmax
            header['delta t'] = (tmax - tmin) / np.float(self.length)

        return pd.Series(header)

    def get_props(self):

#       with pd.HDFStore(vmimodules.conf.gmd_loc) as st:
#           gmd = st.joint_df[st.joint_df.pulse_id.isin(self.pid_ar)]

        counts = pd.DataFrame({k: [v[(v.ls_rank == 5) & (v['mask'] == True)].x_gau.dropna().shape[0],
                                   v.x_cntr.dropna().shape[0]] 
                               for k,v in self.mlt.items()},
                               index=['count_gau', 'count_com']).T

        return counts

    def store(self):

        with pd.HDFStore('%s-ss.h5' % self.hdf) as store:
            header = self.get_header()
#           counts = self.get_props()
#           counts = counts.join(self.ot)

#           counts.index = self.index


            if self.parallel:
                mlt = pd.concat([store.get(k) for k in store 
                                              if k.startswith('/mlt')])
                sgl = pd.concat([store.get(k) for k in store 
                                              if k.startswith('/sgl')])
                [store.remove(k) for k in store]

                store['mlt'] = mlt
                store['sgl'] = sgl
                self.pid_ar = mlt.items

            else:
                store['mlt'] = self.mlt
                store['sgl'] = self.sgl
                self.pid_ar = self.mlt.items

            store['header'] = header[self.cols]
            store['imax'] = pd.DataFrame(self.i_max)
#           store['counts'] = counts

            if hasattr(self, 'times'):
                long_times = np.repeat(self.times, self.dimd / len(self.times))
                index = pd.MultiIndex.from_arrays([self.pid_ar, long_times])

            else:
                index = self.pid_ar

            store.sgl.items, store.mlt.items = index, index
            cl.purge_everything()

#==============================================================================

def getint(name):
        basename = name.partition('.')[0]
        num = basename.split('-')[-1]
        num = num.split('q')[-1]
        try: return int(num)
        except: return None


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
