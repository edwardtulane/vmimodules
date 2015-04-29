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
import pylab as pl

import copy as cp

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
        print 'Given crop size %d exceeds maximum of %d' % (crop, dd)
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
        print 'No hot spots given!'

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
    frame[R > rmax] = 0.0
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
    14-11-10: scheissegal
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
    IMr = zIMr[cN-sy:cN+sy+1,cN-sx:cN+sx+1]
    IMr=np.array(IMr)

    return IMr

###
###
###


def gen_rect(diam, dens, disp):
    """
    Generate a square grid of given diameter and density,
    whose centre is shifted by the displacement vector
    """
    spc1D = np.linspace(0, diam, dens)
    x_coord, y_coord = np.meshgrid(spc1D, spc1D)

    return [y_coord + disp[1], x_coord + disp[0]]


def gen_polar(radius, radN, polN, disp):
    """
    Generate a polar grid of given radius and density in both radius and angles
    14-08-20: Approved
    """
    radii = np.linspace(0, radius, radN)
    angles = np.linspace(0, 2*np.pi, polN)
    pol_coord, rad_coord = np.meshgrid(angles, radii)
    x_coord = rad_coord * np.sin(pol_coord) + radius
    y_coord = rad_coord * np.cos(pol_coord) + radius

    return [y_coord + disp[1], x_coord + disp[0]]


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
    img = cp.deepcopy(img_in)
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
    img = cp.deepcopy(img_in)
    cntr_v, cntr_h = (img.shape[0] - 1) / 2,  (img.shape[1] - 1) / 2
    if not (v or h):
        raise ValueError('Where shall I unfold? (v, h) = bool')

    if v:
        img = np.vstack((img[:0:-1], img))

    if h:
        img = np.hstack((img[:,:0:-1], img))

    return img

def quadrants(img_in):
    img = cp.deepcopy(img_in)
    cntr_h, cntr_v = (img.shape[0] - 1) / 2,  (img.shape[1] - 1) / 2
    q1, q2 = img[cntr_h:,cntr_v:], img[cntr_h:,cntr_v::-1]
    q3, q4 = img[cntr_h::-1,cntr_v::-1], img[cntr_h::-1,cntr_v:]
    return np.asarray([q1, q2, q3, q4])

def halves(img_in):
    img = cp.deepcopy(img_in)
    cntr_h, cntr_v = (img.shape[0] - 1) / 2,  (img.shape[1] - 1) / 2
    h1 = img[:,cntr_v:]
    h2 = img[:,cntr_v::-1]
    return np.asarray([h1, h2])

#-----------------------------------------------------------------------------#
#-----------------------------------------------------------------------------#

#def find_bg_fac(fac, fra, frb):
#    sub = fra - fac * frb
#    return quadrants(sub).std(axis=(0)).sum()
    
    
if __name__ == '__main__':
    pl.close('all')
