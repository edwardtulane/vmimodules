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

global_dens = 501

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


#
##
### not yet checked

#def beta_vec(phi,p):
#    x = np.cos(phi)
#    P = np.empty([2*(len(p)-2)+1+1,len(phi)])
#    P[0,:]=x
#    P[1,:]=0.5*(3*(x**2)-1)
#    I=1+p[1]*P[1]
#    for i in np.arange(3,len(p)+1):
#        n=2*(i-2)
#        P[n+1-1,:]=(2*n+1)*x*P[n-1,:]-n*P[n-1-1,:]
#        n=2*(i-2)+1
#        P[n+1-1,:]=(2*n+1)*x*P[n-1,:]-n*P[n-1-1,:]
#        I=I+p[i-1]*P[n+1-1,:]
#    I=I*p[0]/(4.*np.pi)
#
#    return I
#
#
#def beta(theta,p1,p2):
#    x=np.cos(theta)
#    P=np.empty([2,len(theta)])
#    P[0,:]=x
#    P[1,:]=0.5*(3*(x**2)-1)
#    I=1+p2*P[1]
#    I=I*p1/(4.*np.pi)
#
#    return I
#
#
#def beta_fit_func(p,phi,distr):
#    erf=beta_vec(phi,p)-distr
#    return erf
#
#def beta_r(img):
#    [sx,sy]=img.shape
#    radius=(sx+1)/2
#    [M,r,phi]=cart2polar(img,radius,360)
#    betas=[]
#    for i in range(len(r)):
#        cut=M[:,i]
#        v,v2=leastsq(beta_fit_func,[1000,1.],args=(phi,cut))
#        betas.append(v)
#    betas=np.array(betas)
#    betas=np.vstack((r,betas.transpose()))
#    return betas
#
#
#
#
#### not yet checked
###
##

# Remove all pixel values above the threshold rmax
# 14-08-20: Maybe useful
def crop_circle(frame, rmax):
    """
    Remove all pixel values above the threshold rmax
    14-08-20: Maybe useful, not checked
    """
    cy, cx = (np.asarray(frame.shape) - 1) / 2
    rad = frame.shape[0]
    XY = np.arange(rad,dtype='float64')
    R = np.sqrt((XY - cx) ** 2 + (XY - cy)[:, None] ** 2)
    frame[R > rmax] = 0.0
    return frame

def getintens(frame, cx, cy, rmax):
    rad = frame.shape[0]
    XY = np.arange(rad,dtype='float64')
    R = np.sqrt((XY - cx) ** 2 + (XY - cy)[:, None] ** 2)
    msk = np.ma.masked_less(R, rmax)
    msk = np.ma.getmask(msk)
    return np.ma.array(frame, mask=msk)

def getintensdiff(lam, frame1, frame2, cx, cy, rmax):
    intens = getintens(frame1 - lam * frame2, cx, cy, rmax)
    return abs(np.sum(intens))


# Calculate <cos**2(theta)> w.r.t. reference axis theta0
def meancos2(M,theta0,rmin=0.):
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
    # Get the cos between reference axis (theta0) and each pixel vector
    x0=np.cos(theta0)
    y0=np.sin(theta0)
    # cos(theta)=vec{a}*vec{b}/(a*b)
    costheta=(xpix*x0+ypix*y0)/(rpix)
    # Finally calculate <cos**2(theta)>
    return np.sum(wM*(costheta)**2)/np.sum(wM)
#
#
## Radial distribution (Normalized by number of Pixels)
#def radial_distr(M):
#	sx=M.shape[0]
#	x0=np.ceil(M.shape[0]/2)
#	y0=np.ceil(M.shape[1]/2)
#	R=np.zeros((np.ceil(x0),2))
#	R[:,0]=np.arange(1,np.ceil(x0)+1)
#
#	[x,y]=np.meshgrid(np.arange(1,sx+1),np.arange(1,sx+1))
#
#	r_mat = np.ceil(np.sqrt((x-x0-1)**2+(y-y0-1)**2))
#	r_mat[r_mat==0]=M[x0,y0]
#
#	for r in np.arange(1,R.shape[0]+1):
#		r_mat_r=(r_mat==r)
#		temp=r_mat[r_mat_r]
#		Nr=max(1,temp.shape[0])
#        R[r-1,1]=(np.sum(M[r_mat_r])/Nr)*r
#
#	return R
#
## Radial Distribution between angles
#def radial_distr_angle(M,phi1,phi2):
#    [sx,sy]=M.shape
#    x0,y0=(sx+1)/2,(sy+1)/2
#    [y,x]=np.meshgrid(np.arange(1,sx+1),np.arange(1,sx+1))
#    r_mat=np.ceil(np.sqrt((x-x0)**2+(y-y0)**2))
#    phi_mat=np.arctan2(x-x0,y-y0)+np.pi/2
#
#    #reduce intergration area
#    rad_arr=np.array([r_mat.flatten(),phi_mat.flatten(),M.flatten()])
#    rad_arr=rad_arr.transpose()
#    rad_arr_red=rad_arr[(rad_arr[:,1]<phi2) & (rad_arr[:,1]>=phi1)]
#
#    #Integrate for different r
#    R=np.zeros((x0,2))
#    for r in np.arange(1,R.shape[0]+1):
#        temp=rad_arr_red[rad_arr_red[:,0]==r]
#        Nr=max(1,temp.shape[0])
#        temp=(np.sum(temp,0)/Nr)
#        R[r-1,0]=temp[0]
#        R[r-1,1]=(temp[2])*r
#    return R
#
#def radial_distr_2(M):
#    [M,r,phi]=cart2polar(M,250,360)
#    R=np.array([r,np.sum(M,0)])
#    R=R.T
#    return R
#
#def radial_distr_angle_2(M,phi1,phi2):
#    [M,r,phi]=cart2polar(M,250,120)
#    M[phi1>phi]=0
#    M[phi>phi2]=0
#    R=np.array([r,np.sum(M,0)])
#    R=R
#    return R

###
###
###

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


if __name__ == '__main__':
    pl.close('all')
