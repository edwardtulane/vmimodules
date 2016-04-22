# -*- coding: utf-8 -*-
"""
Refurbished on Wed Aug 20 2014

@author: brausse

Basic VMI data processsing functions

STATUS 2014-08-20
DONE: check and document all used functions
TODO: deliberately delete commented blocks

"""

import numpy as np
import scipy as sp

import math
import scipy.signal as sig
import scipy.ndimage.interpolation as ndipol
import matplotlib.pyplot as pl

def rawread(filename):
    """
    Read rawfile named 'filename'
    Its first 2 numbers are the image dimensions
    nx, ny, ntot are the number of points in x, y, and total
    14-08-20: Revisited
    """
    M = np.fromfile(filename, dtype=np.dtype('<i4'))
    nx, ny = M[0], M[1]
    ntot = nx * ny
    M = np.reshape(M[2:ntot + 2], (ny, nx))

    return M

def read_singleshots(fname):
    """
    """
    arr = np.memmap(fname, dtype=np.int32,mode='r')
    x_dim, y_dim, seqlen = arr[0:3]
    slice_len = (x_dim * y_dim) / 4
    ids = np.zeros(seqlen, dtype=np.int32)
    ss_arr = np.zeros((seqlen, y_dim, x_dim), dtype=np.int8)
    
    for i in range(seqlen):
        ids[i] = arr[3 + i * (slice_len + 1)]
        int8view = arr[3 + 1 + i * (slice_len + 1) 
                      :3 + (i+1) * (slice_len + 1)].view(np.int8)
        ss_arr[i] = int8view.reshape((y_dim, x_dim))
        
    return ids, ss_arr

class SingleShotExtractor():

    def __init__(self, flist):

        file_count = len(flist)
        self.arrs = list()
        for f in flist:
            self.arrs.append(np.memmap(f, dtype=np.int32, mode='r'))
        self.xdim, self.ydim = np.zeros(file_count, dtype=np.int32), np.zeros(file_count, dtype=np.int32)
        self.seqlen = np.zeros(file_count, dtype=np.int32)

        for i, ar in enumerate(self.arrs):
            self.xdim[i], self.ydim[i], self.seqlen[i] = ar[:3]

        assert np.all(self.ydim[1:] ==  self.ydim[:-1])
        assert np.all(self.xdim[1:] ==  self.xdim[:-1])

        self.slice_len = (self.xdim[0] * self.ydim[0]) / 4
        self.locs = np.repeat(np.arange(file_count), self.seqlen)
        self.indx = np.array([list(range(l)) for l in self.seqlen]).ravel()

        assert self.locs.shape == self.indx.shape



#       self.flist, self.file_id = flist, 0
#       self.arr = np.memmap(flist[self.file_id], dtype=np.int32, mode='r')
#       self.x_dim, self.y_dim, self.seqlen = self.arr[0:3]
#       self.slice_len = (self.x_dim * self.y_dim) / 4

#       ids = np.zeros(seqlen, dtype=np.int32)
#       self.ss_arr = np.zeros((self.y_dim, self.x_dim), dtype=np.int8)

#       self.index = 0

    def __len__(self):
        return np.sum(self.seqlen)

    def get_frame(self, index):
        ar_id = self.locs[index]
        ix = self.indx[index]
        ar = self.arrs[ar_id]

        pid = ar[3 + ix * (self.slice_len + 1)]
        int8view = ar[3 + 1 + ix * (self.slice_len + 1) 
                     :3 + (ix + 1) * (self.slice_len + 1)].view(np.int8)
        int8view.shape = (self.ydim[0], self.xdim[0])

        return pid, int8view

    def __getitem__(self, index):

        if isinstance(index, np.ndarray):
            dim = len(index)
            ids = np.zeros(dim, dtype=np.int32)
            frames = np.zeros((dim, self.ydim[0], self.xdim[0]), dtype=np.int8)

            for i,ix in enumerate(index):
                ids[i], frames[i] = self.get_frame(ix)

            return ids, frames

        else:
            id, ar = self.get_frame(index)
            ar.shape = (self.ydim[0], self.xdim[0])
            return id, np.array(ar)
#   def __iter__(self):
#       return self

#   def next(self):

#       if self.index == self.seqlen:
#           self.file_id += 1

#           if self.file_id  ==  len(self.flist):
#               raise StopIteration

#           self.arr = np.memmap(self.flist[self.file_id], dtype=np.int32, mode='r')
#           self.seqlen = self.arr[2]
#           self.index = 0

#       i, arr, sl_len = self.index, self.arr, self.slice_len
#       pid = arr[3 + i * (sl_len + 1)]
#       int8view = arr[3 + 1 + i * (sl_len + 1) 
#                      :3 + (i+1) * (sl_len + 1)].view(np.int8)
#       self.ss_arr[:,:] = int8view.reshape((self.y_dim, self.x_dim))

#       self.index += 1

#       return pid, self.ss_arr



def centre_crop(M, cx, cy, crop=0):
    """
    Centre and crop image to a square
    cx, cy: centre point; sx, sy: image size; dx, dy, dd: radial dimensions
    Restricted to 'dd' if the given crop value exceeds the image size
    14-08-20: Checked again and approved

    """
    sy, sx = M.shape
    dx, dy = min(cx, sx-cx-1), min(cy, sy-cy-1)
    dd = min(dx, dy)
    if crop and crop <= dd:
        dd = crop
    elif crop and crop > dd:
        print('Given crop size %d exceeds maximum of %d' % (crop, dd))
    M_centrd = M[(cy - dd):(cy + dd + 1), (cx - dd):(cx + dd + 1)]
    # (the stop is not inclusive)

    return M_centrd


def hot_spots(M, spots):
    """
    Replace hot spot pixel value with average of surrounding pixel values
    14-08-20: Revisited

    """
    if spots:
        for i, px in enumerate(spots):
            x = px[0]
            y = px[1]
            M[y, x] = (M[y+1, x] + M[y-1, x] + M[y, x+1] + M[y, x-1]) / 4
    else:
        print('No hot spots given!')

    return M


# Remove all pixel values above the threshold rmax
# 14-08-20: Maybe useful
def crop_circle(frame, rmax):
    """
    Remove all pixel values above the threshold rmax
    14-08-20: Maybe useful, not checked
    15-03-13: Looks fine, put it into Basex routine
    """
    cy, cx = (np.asarray(frame.shape) - 1) / 2
    rad = frame.shape[0]
    XY = np.arange(rad,dtype='float64')
    R = np.sqrt((XY - cx) ** 2 + (XY - cy)[:, None] ** 2)
#   rim = frame[(R>rmax-2) & (R<rmax+2)]
#   frame -= np.mean(rim)
    frame[R > rmax] = 0.0
#   frame[frame < 0] = 0
    return frame

def getintens(frame, cx, cy, rmax):
    """
    Sums up all pixels on the rim. Probably useless.
    """
    rad = frame.shape[0]
    XY = np.arange(rad,dtype='float64')
    R = np.sqrt((XY - cx) ** 2 + (XY - cy)[:, None] ** 2)
    msk = np.ma.masked_less(R, rmax)
    msk = np.ma.getmask(msk)
    return np.ma.array(frame, mask=msk)

def getintensdiff(lam, frame1, frame2, cx, cy, rmax):
    """
    Difference between two rims. Probably also useless.
    """
    intens = getintens(frame1 - lam * frame2, cx, cy, rmax)
    return abs(np.sum(intens))


# Calculate <cos**2(theta)> w.r.t. reference axis theta0
def meancos2(M,theta0,rmin=0.):
    """
    15-03-13: Deprecated.
    """
    sxy=M.shape[0]
    [xpix,ypix]=np.meshgrid(np.arange(0,sxy),np.arange(0,sxy))
    cxy=sxy/2
    # Calculate pixel position vectors and their length
    xpix=np.float_(xpix-cxy)
    ypix=np.float_(ypix-cxy)
    rpix=np.sqrt(xpix**2+ypix**2)
    # Take care of singularity at center pixel
    wM=M.copy()
    wM[rpix==0]=0.
    rpix[rpix==0]=1.
    # Zero out all values below rmin
    if rmin>0.:
        wM[rpix<=rmin]=0.

def iniBasex(pathname):
    """
    Load BASEX matrices
    14-08-20: Obsolete with introduction of the Inversion class
    """
    basexM = np.mat(np.loadtxt(pathname+'VMI_basexM.asc'))
    basexMc = np.mat(np.loadtxt(pathname+'VMI_basexMc.asc'))

    return basexM, basexMc


# BASEX transformation with two regularization factors
def Basex(IM,q1,q2, M, Mc, MTM, MTMc):
    """
    BASEX transformation with two regularization factors
    14-08-20: to be ported to the Inversion class and REVISITED
    14-11-10: what a mess
    """
    NBF = min(np.shape(M))
    N = max(np.shape(M))
    cN = (N - 1) / 2
    sz = np.shape(IM)
    sx = (sz[1]-1)/2
    sy = (sz[0]-1)/2
    zIM = np.mat(np.zeros([N,N]))
    zIM[cN-sy:cN+sy+1,cN-sx:cN+sx+1]=np.mat(IM)
    Ci = np.linalg.inv(MTMc+q2*np.eye(NBF,NBF))*Mc.T*zIM*M*np.linalg.inv(MTM+q1*np.eye(NBF,NBF))
    zIMr = np.dot(Mc,np.dot(Ci,Mc.T))
    ab = np.dot(Mc, Ci).dot(M.T)
    res = (zIM - ab)[cN-sy:cN+sy+1,cN-sx:cN+sx+1]
    IMr = zIMr[cN-sy:cN+sy+1,cN-sx:cN+sx+1]
    IMr=np.array(IMr)

    return IMr, np.array(res)

###
###
###


def gen_rect(diam, dens, disp, phi=0):
    """
    Generate a square grid of given diameter and density,
    whose centre is shifted by the displacement vector
    2015-11-26: corrected for bspline displacement of [0.5, 0.5]
    2015-12-09: included rotation and omitted displacement, which was caused by
                ndimage.rotate
    """

    r0 = (diam - 1) / 2.
    spc1D = np.linspace(0, diam - 1, dens)

    x_coord, y_coord = np.meshgrid(spc1D, spc1D)
    
    x_coord -= r0 #+ disp[0]
    y_coord -= r0 #+ disp[1]
    
    cosphi = math.cos(math.radians(phi))
    sinphi = math.sin(math.radians(phi))
    
    y_rot = cosphi * y_coord - sinphi * x_coord
    x_rot = sinphi * y_coord + cosphi * x_coord
    
    x_rot += r0 - disp[0]
    y_rot += r0 - disp[1]

    #return [y_coord + disp[1], x_coord + disp[0]]
    return [y_rot, x_rot]

def gen_polar(radius, radN, polN, disp, phi=0):
    """
    Generate a polar grid of given radius and density in both radius and angles
    14-08-20: Approved
    2015-12-09: included rotation and omitted displacement, which was caused by
                ndimage.rotate
    """

    radii = np.linspace(0, radius, radN)
    angles = np.linspace(0, 2*np.pi, polN)
    angles += math.radians(phi)
    pol_coord, rad_coord = np.meshgrid(angles, radii)
    x_coord = rad_coord * np.sin(pol_coord) + radius 
    y_coord = rad_coord * np.cos(pol_coord) + radius

    return [y_coord - disp[1], x_coord - disp[0]]

def gen_qrs_grid(radius, radN, polN, alpha):
    """
    Generate a polar grid of displaced rescattering circles.
    """
    radii = -np.linspace(-1.26, +1.26, radN) * 1.1
#   radii = np.abs(radii)
    offs = np.linspace(-1, +1, radN) * 1.1
    angles = np.linspace(0, 2*np.pi, polN)
    pol_coord, rad_coord = np.meshgrid(angles, radii)
    x_coord = rad_coord * np.sin(pol_coord)
    y_coord = rad_coord * np.cos(pol_coord) + offs[:, None]

    return [radii, y_coord * alpha + radius, x_coord * alpha + radius]


def plot_circles(axes, x_cntr, y_cntr, fro=5, to=120, Ncirc=7):
    """
    Draw red, equidistant circles into the plot 'axes'
    fro, to; Ncirc: lower and upper radius bounds; number of circles
    14-08-20: Approved
    """
    for rad in np.linspace(fro, to, Ncirc):
        circle = pl.Circle([x_cntr, y_cntr], radius=rad,
                           color='red', fill=False)
        axes.add_artist(circle)

#-----------------------------------------------------------------------------#
#-----------------------------------------------------------------------------#

### Utilities

def resolve_path(homedir, date, seqNr, setup, mode):
        if type(date) is not str:
            raise TypeError('Please give the date as a string')

        if setup == 'TW':
            path = homedir + 'terawatt/' + date + '/'
        elif setup == 'HE':
            path = homedir + 'strongfield/' + date + '/'
        else:
            raise ValueError('Setup must be HE or TW')

        if mode == 'raw':
            pass
        elif mode == 'seq' and seqNr:
            if setup == 'HE':
                path += 'Seq/'
            else:
                pass
            path += date + '-' + str(seqNr) + '/'
        else:
            raise ValueError('Mode must be raw or seq (+ seqNr)')

        return path

def fold(img_in, v=False, h=False):
    img = img_in.copy()
    cntr_v, cntr_h = (img.shape[0] - 1) / 2,  (img.shape[1] - 1) / 2
    if not (v or h):
        raise ValueError('Where shall I fold? (v, h) = bool')

    if v:
        img[cntr_v:,:] += img[cntr_v::-1,:]
        img[cntr_v:,:] /= 2
        slc = img[cntr_v:,:]

    if h:
        img[:,cntr_h:] += img[:,cntr_h::-1]
        img[:,cntr_h:] /= 2
        slc = img[:,cntr_h:]

    if v and h:
        slc = img[cntr_v:,cntr_h:]

    return slc

def unfold(img_in, v=False, h=False):
    img = img_in.copy()
    cntr_v, cntr_h = (img.shape[0] - 1) / 2,  (img.shape[1] - 1) / 2

    if any(isinstance(cntr_v, float), isinstance(cntr_h, float)):
        assert cntr_v.is_integer() and cntr_h.is_integer()
    cntr_h, cntr_v = int(cntr_h), int(cntr_v)
    if not (v or h):
        raise ValueError('Where shall I unfold? (v, h) = bool')

    if v:
        img = np.vstack((img[:0:-1], img))

    if h:
        img = np.hstack((img[:,:0:-1], img))

    return img

def quadrants(img_in):
    img = img_in.copy()
    cntr_h, cntr_v = (img.shape[0] - 1) / 2,  (img.shape[1] - 1) / 2

    if any(isinstance(cntr_v, float), isinstance(cntr_h, float)):
        assert cntr_v.is_integer() and cntr_h.is_integer()
    cntr_h, cntr_v = int(cntr_h), int(cntr_v)

    q1, q2 = img[cntr_h:,cntr_v:], img[cntr_h:,cntr_v::-1]
    q3, q4 = img[cntr_h::-1,cntr_v::-1], img[cntr_h::-1,cntr_v:]
    return np.asarray([q1, q2, q3, q4])

def compose(qu):
    cmps = np.zeros((np.asarray(qu[0].shape) * 2 - 1))
    cntry = qu[0].shape[0] - 1
    cntrx = qu[0].shape[1] - 1
    cmps[cntry:, cntrx:], cmps[cntry:,cntrx::-1] = qu[0],  qu[1]
    cmps[cntry::-1,cntrx::-1], cmps[cntry::-1,cntrx:] = qu[2], qu[3]
    return cmps

def halves(img_in):
    img = img_in.copy()
    cntr_h, cntr_v = (img.shape[0] - 1) / 2,  (img.shape[1] - 1) / 2
    h1 = img[:,cntr_v:]
    h2 = img[:,cntr_v::-1]
    return np.asarray([h1, h2])

def map_quadrant_polar(qu, radN=251, polN=257, smooth=0.0):

    # make a single quadrant iterable
    if len(qu.shape) == 2:
        qu = [qu]
        pol = np.zeros([radN, polN])
        pol = [qu]
    else:
        pol = np.zeros([qu.shape[0], radN, polN])

    radius = qu[0].shape[0] - 1
    radii = np.linspace(0, radius, radN)
    angles = np.linspace(0, 0.5*np.pi, polN)

    pol_coord, rad_coord = np.meshgrid(angles, radii)
    x_coord = rad_coord * np.sin(pol_coord)
    y_coord = rad_coord * np.cos(pol_coord)

    coords = [y_coord, x_coord]

    for i, q in enumerate(qu):
       ck = sig.cspline2d(q, smooth) 
       pol[i] = ndipol.map_coordinates(ck, coords, prefilter=False)

    if isinstance(qu, list):
        return pol[0]
    else:
        return pol

def get_raddist(qu, radN, polN=257, order=8, cov=False):
    pol = map_quadrant_polar(qu, radN)

    th = np.linspace(0, 0.5*np.pi, polN)
    rad2 = np.arange(radN)**2
#   kern = pol * np.sin(th) * rad2[:,None]
#   dist = integrate.romb(kern, axis=1, dx=np.pi/(polN-1))

    legvan = np.polynomial.legendre.legvander(np.cos(th), order)
    legvan = legvan[:,::2]
    x, res, rank, cond = np.linalg.lstsq(legvan, pol.T)

    return x * rad2[None,:]

def get_raddist_weighted(qu, sig, radN, polN=257, order=8):
    pol = map_quadrant_polar(qu, radN)
    sig_p = map_quadrant_polar(sig, radN)
    th = np.linspace(0, 0.5*np.pi, polN)
    rad2 = np.arange(radN)**2
#   kern = pol * np.sin(th) * rad2[:,None]
#   dist = integrate.romb(kern, axis=1, dx=np.pi/(polN-1))
    x = np.zeros([order/2+1, radN])
    sig_out = np.zeros_like(x)

    for i in range(radN):
        l = pol[i]
        w = 1 / sig_p[i]
        l *= w
        legvan = np.polynomial.legendre.legvander(np.cos(th), order)
        legvan = legvan[:,::2]
        legvan *= w[:,None]
        x[:,i], res, rank, cond = np.linalg.lstsq(legvan, l)

        cov_m = np.linalg.inv(legvan.T.dot(legvan))
        sig_out[:,i] = np.sqrt(np.diag(cov_m))

        

    return x * rad2[None,:], sig_out

#-----------------------------------------------------------------------------#
#-----------------------------------------------------------------------------#

#def find_bg_fac(fac, fra, frb):
#    sub = fra - fac * frb
#    return quadrants(sub).std(axis=(0)).sum()
    
    
if __name__ == '__main__':
    pl.close('all')
