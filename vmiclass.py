# -*- coding: utf-8 -*-
"""
Created on Mon Aug 11 14:46:59 2014

@author: brausse

Classes for VMI images

STATUS 2014-08-20
DONE:
TODO:

"""

import numpy as np
import scipy as sp

import pylab as pl

import scipy.fftpack as fft
import scipy.ndimage as ndimg
import scipy.ndimage.interpolation as ndipol
import scipy.optimize as opt
import scipy.signal as sig
import scipy.special as bessel
import copy as cp

import lib.proc as vmp
import lib.inv as inv
import lib.vis as vmv

#import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, Button, RadioButtons

glob_dens = 501

global M1, M2
class VMIimage(object):
    """
    VMI class for reading and manipulating single frames
    Contains methods for orienting and interpolating a single frame,
    also operator overloading for manipulations between images
    Inversion methods are called from an Inversion class object (TODO)
    """

    def __init__(self, file=[], xcntr=0, ycntr=0, radius=0, hotspots=[]):
        global glob_dens
        if type(file) is str:
            self.readfile(file)
        else:
            self.frame = file
            self.prist = cp.copy(self.frame)

        self.__set_centre(xcntr, ycntr)
        self.disp = [0, 0, 0]
        self.dens_rect = glob_dens
        self.clrmap = 'jet'

        self.rect = np.empty(0)
        self.polar = np.empty(0)

        self.__Cmn = np.empty(0)

        if hotspots:
            self.__rm_hotspot(hotspots)
        else:
            pass

        if not self.cx or not self.cy:
            print 'No valid center given'
        elif radius:
            self.rad_sq = radius
            self.cropsquare(self.rad_sq)
        else:
            pass

 #
### Addition and subtraction of frames and spline coefficients
 #

    def __add__(self, other):
        third = cp.deepcopy(self)
        third.ck = self.ck + other.ck
        third.frame = self.frame + other.frame
        return third

    def __iadd__(self, other):
        self.ck += other.ck
        self.frame += other.frame

    def __sub__(self, other):
        third = cp.deepcopy(self)
        third.ck = self.ck - other.ck
        third.frame = self.frame - other.frame
        return third

    def __isub__(self, other):
        self.ck -= other.ck
        self.frame -= other.frame
 #
### Multiplication and division by a float or an integer
 #

    def __rmul__(self, other):
        third = cp.deepcopy(self)
        third.ck = self.ck * other
        third.frame = self.frame * other
        return third

    def __truediv__(self, other):
        third = cp.deepcopy(self)
        third.ck = self.ck / other
        third.frame = self.frame / other
        return third

    def __str__(self):
        ret_str = ''
        if self.frame.any():
            ret_str += "VMIimage of dimension %r from a raw file of dimension \
%r and centre (%d, %d)\n" % \
                (self.frame.shape, self.prist.shape, self.cx, self.cy)
        if self.rect.any():
            ret_str += "Square projection of dimension %r \n" % \
                (self.rect.shape,)
        if self.polar.any():
            ret_str += "Polar projection of dimension %r \n" % \
                (self.polar.shape,)
        if not ret_str:
            ret_str = 'Empty VMI image'

        return ret_str

 #
### Raw image loading and manipulation
 #

    def readfile(self, filename):
        """ Read the file 'filename' and make a copy of the original image"""
        self.frame = vmp.rawread(filename)
        self.prist = cp.copy(self.frame)

    def __set_centre(self,  xcntr, ycntr):
        self.cx, self.cy = xcntr, ycntr

    def __rm_hotspot(self, hotspots):
        self.frame = vmp.hot_spots(self.prist, hotspots)

    def cropsquare(self, rad):
        self.frame = vmp.centre_crop(self.frame, self.cx, self.cy, rad)
        self.diam = self.frame.shape[0]

 #
### Interpolation and remapping
 #

    def interpol(self):
        """ Cubic spline interpolation with zero smoothing """
        self.__ck_frame = sig.cspline2d(self.frame, 0.0)

    def rotateframe(self, phi=0):
        """ Rotate the coefficient matrix in degs, cutting off the corners """
        self.ck = ndimg.rotate(self.__ck_frame, phi,
                               reshape=False, prefilter=False)

    def evalrect(self, density, displace=[0., 0.], phi=0):
        """ Project the image onto a rectangular grid with given spacing """
        self.rotateframe(self.offset + self.disp[0] +  phi)
        coords = vmp.gen_rect(self.diam, density, self.disp[1:] + displace)
        self.rect = ndipol.map_coordinates(self.ck, coords, prefilter=False)

    def evalpolar(self, radN, polN):
        """ Project the image onto a polar grid with radial and polar denss."""
        self.rotateframe(self.offset + self.disp[0])
        coords = vmp.gen_polar(self.rad_sq, radN, polN, self.disp[1:])
        self.polar = ndipol.map_coordinates(self.ck, coords, prefilter=False)

    def invertedpolar(self, radN, polN, inv='basex'):
        """ Project the inverted image onto a polar grid."""
        coords = vmp.gen_polar((self.dens_rect - 1) / 2, radN, polN, [0, 0, 0])
        ck = sig.cspline2d(self.bsx, 0.0)
        self.bsx_pol = ndipol.map_coordinates(ck, coords, prefilter=False)
 #
### Finding the centre point and offset angle
 #

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

    def find_centre(self):
        """ Iterate 'eval_sym' with a bound BFGS alg. verbosely ('disp') """
        init_vec = [0, 0, 0]
        domain = np.tile([-90, 90], 3).reshape(3, 2)
        self.res = opt.minimize(self.__eval_sym, init_vec,
                                method='L-BFGS-B', bounds=domain,#'L-BFGS-B'
                                tol=1E-5, options={'disp': True})
        if self.res.success:
            print 'Writing optimized centre and angular offset'
            self.disp += self.res.x
        # Final evaluation
        self.rotateframe(self.offset + self.disp[0])
        self.evalrect(self.dens_rect)

### inversion methods. TO BE ported to Inversion class

    def invertFourierHankel(self):
        dim = (self.rect.shape[1] - 1) /2
        self._shift = np.append(self.rect[:,dim:], self.rect[:,:dim],axis=1)
        self.fourt = fft.fft(self._shift,axis=1)
        ft_freqs = fft.fftfreq(self.fourt.shape[1])
###
        jn = bessel.jn_zeros(0, dim + 2)
        S, R1 = jn[-1], ft_freqs.max()
        R2 = S / (2 * np.pi * R1)
        print R1, R2, S
        jn = jn[:-1]
        F1_arg = jn / (2 * np.pi * R2)
        F1_arg *= (ft_freqs.shape[0] - 1) / (2 * R1)
        J1_vec = abs(bessel.j1(jn) ** -1)
        if not self.__Cmn.shape == (dim + 1, dim + 1):
            self.__jn_mat = (jn * jn[:, None]) / S
            self.__J1_mat =  J1_vec * J1_vec[:, None]
            self.__Cmn = (2 / S) * bessel.j0(self.__jn_mat) * self.__J1_mat
        else:
            pass

        F1 = np.empty((self.fourt.shape[0], dim + 1), dtype='complex')
        self.FHT = np.empty((self.fourt.shape[0], dim + 1), dtype='complex')

        for i, line in enumerate(self.fourt):
            ft_cR = sig.cspline1d(line.real)
            ft_cI = sig.cspline1d(line.imag)
            F1[i] = ( sig.cspline1d_eval(ft_cR, F1_arg) \
                            + 1j * sig.cspline1d_eval(ft_cI, F1_arg) \
                            ) * J1_vec * R1

        self.FHT = np.dot(F1, self.__Cmn)
        self.FHT /= (R2 * J1_vec)
        self.F2_arg = jn / (2 * np.pi * R1)
        self.orig = np.dot(self.FHT, self.__Cmn)

    def invertBasex(self):
        self.bsx = vmp.Basex(self.rect, 10, 0, inv.M1, inv.M2,
                             inv.MTM1, inv.MTM2)

 #
### Plenty of plotting methods
 #

    def plotplain(self, img):
        self.__crfig, self.__crax = vmv.vmiplot(img)

    def plotFrame(self):
        self.plotplain(self.frame)

    def plotPrist(self):
        self.plotplain(self.prist)

    def plotRect(self):
        self.plotplain(self.rect)

    def plotPolar(self):
        self.plotplain(self.polar)

    def plotFHT(self):
        self.plotplain(abs(self.FHT))

    def plotBsx(self):
        self.plotplain(self.bsx)

    def plotCentre(self):
        """ Plot a cross to chosen centre and a number of circles around it """
        if self.prist.any():
            self.plotPrist()
            pl.scatter(self.cx, self.cy, c='r', marker='+')
            vmp.plot_circles(self.__crax, self.cx, self.cy)

        self.plotRect()
        self._cntr = (self.rect.shape[0] - 1) / 2
        cntr = self._cntr
        pl.scatter(cntr, cntr, c='r', marker='+')
        vmp.plot_circles(self.__crax, cntr, cntr)
        line1 = pl.Line2D([cntr, cntr], [0, 2 * cntr], c='r')
        line2 = pl.Line2D([0, 2 * cntr], [cntr, cntr], c='r')
        self.__crax.add_artist(line1)
        self.__crax.add_artist(line2)


#=============================================================================#
#=============================================================================#

class TWimage(VMIimage):
    """ Subclass for images taken at the Terawatt VMI """

    def __init__(self, file, xcntr=249, ycntr=234, hotspots=[], radius=220
                #229
                ):
        #old centre: 247, 233
        VMIimage.__init__(self, file, xcntr, ycntr, radius)
        self.offset = -52.5
        self.disp = [ 0.4777369 ,  0.13077729,  1.13548382]

        self.interpol()
#        self.rotateframe()
#        self.evalrect(self.dens_rect)
#        self.evalpolar(500,800)


class HEimage(VMIimage):
    """ Subclass for images taken at the high energy VMI """

    def __init__(self, file, xcntr=483, ycntr=467,
                 hotspots=[[325, 480], [45, 610]], radius=330):
        VMIimage.__init__(self, file, xcntr, ycntr, radius, hotspots)
        self.offset = 88.0
        self.disp = [0,0,0] #[-0.93814201, -0.60886962, -1.28898686]

        self.interpol()
        self.rotateframe(self.offset+self.disp[0])
        self.evalrect(self.dens_rect, self.disp)
#        self.evalpolar(500,800)

#=============================================================================#
#=============================================================================#

class VMIseries(TWimage, HEimage):
    """ Import a series of VMI imgs. and merge them into a single frame """

    def __init__(self, date, indices, seqNr=None, setup='TW', mode='raw',
                 shutter=False):

#        global VMIdir
        VMIdir = '/home/brausse/vmi/'
        self.serFrame = {}
        self.serCk = {}
        self.serPrist = {}
#        self.Series = {}
        self.indx = indices



        path = vmp.resolve_path(VMIdir, date, seqNr, setup, mode)

        if type(self.indx) == int:
            self.indx = np.arange(self.indx) + 1
        elif type(self.indx) == np.ndarray:
            pass
        else:
            raise TypeError('Indices must be an integer or an array')

        self.coeffs = (np.arange(len(self.indx)) + 0.5) * (np.pi / 2)
        self.coeffs = np.sign(np.cos(self.coeffs)) #+ -0.02

        for i, n in enumerate(self.indx):
            if mode == 'raw':
                filename = path + date + '-' + str(n) + '.raw'
            elif mode == 'seq':

                if shutter == False:
                    filename = path + 'seq' + str(n).zfill(4) + '.raw'
                elif shutter == True and i % 2 :
                    filename = path + 'seq' + str(n).zfill(4) + '_closed.raw'
                else:
                    filename = path + 'seq' + str(n).zfill(4) + '_open.raw'

#            if setup == 'TW':
#                TWimage.__init__(self, filename)
#            elif setup == 'HE':
#                HEimage.__init__(self, filename)
            raw = vmp.rawread(filename)

            print 'Importing ' + filename
#            self.serFrame[i] = self.frame
#            self.serCk[i] = self.ck
            self.serPrist[i] = raw

        self.clrmap = 'seismic'

    def sumSeries(self):
        ckpos = np.zeros(self.serPrist[0].shape)
        ckneg = np.zeros(self.serPrist[0].shape)
        for i in np.where(self.coeffs < 0)[0]:
            ckneg += self.serPrist[i]
        for j in np.where(self.coeffs > 0)[0]:
            ckpos += self.serPrist[j]

        return ckpos / (i+1 /2.), ckneg / (j+1 /2.)

class VMIpair(VMIseries):
    def __init__(self, date, indices, setup='TW', mode='raw'):
        VMIseries.__init__(self, date, indices, setup='TW', mode='raw')
        self.dfcoeff = 1.0
        self.subtractPair(self.dfcoeff)

    def subtractPair(self, coeff):
        self.frame = self.serFrame[0] - coeff * self.serFrame[1]
        self.ck = self.serCk[0] -coeff * self.serCk[1]

    def diffWidget(self):
        fig, ax = vmv.plot(self.ck,limits=[-6000,6000],cm='seismic')
        axcolor = 'lightgoldenrodyellow'
        axdiff = pl.axes([0.15, 0.01, 0.65, 0.03], axisbg=axcolor)

        sdiff = Slider(axdiff, 'Difference Coefficient', 0.5, 1.5, valinit=1.0)

        def update(val):
            diff = sdiff.val
            self.subtractPair(diff)
            print self.ck.sum()
            ax.imshow(self.ck, vmin=-6000,vmax=6000,
                      cmap='seismic',origin='lower')
            fig.canvas.draw_idle()
        sdiff.on_changed(update)




#=============================================================================#
#=============================================================================#
#%%
if __name__ == '__main__':
    pl.close('all')
    glob_dens = 501
    he = HEimage('HEtest.raw')
    tw = TWimage('TWtest.raw')

    VMIdir = '/home/brausse/vmi/'
    tw.disp=[+1.2,0,0]
    tw.find_centre()
#    tw.evalrect(501)
#    img=vmp.fold(tw.rect,h=True,v=True)
#    img = img.ravel()
#    res=np.dot(pinv,img.ravel())
#
#    M1, M2 = vmp.iniBasex('storage/')

#    ai = np.ones(1232)*50
    its = 1000
    he.evalrect(501)
#    rc = he.rect.ravel()
#    ab = []
#    bs =np.load('storage/bs-170-19.npy', mmap_mode='r')
#    if not np.any(ab):
#    ab = np.load('storage/ab-170-12.npy', mmap_mode='r')
#    ab = ab.reshape(1750,-1)
#    FtF = np.load('storage/FtF-170-12.npy')
#%%
#print 'I am here'
#res2 = np.dot(np.linalg.inv(FtF + 1 * np.eye(1232)), np.dot(h1,ab).T)
#res2 = np.dot(np.linalg.inv(FtF + 1 * np.eye(1750)), np.dot(ab,img.ravel()))
#    b = fft.fft(img)
#    A = fft.fft(ab,axis=1)
#    Anorm = np.linalg.norm(A, axis = 0)
#%%
#    Anorm = np.sum(np.abs(A)**2, axis = 0)
#%%
#    lam = 1
#
#    lam = 10 ** np.arange(-10,10)
#    for i in lam:
#        den = (Anorm + lam[i]**2)
#        nom = np.sum((b / den)*(b / den).conj())
#        den2 = np.sum((1 / den)*(1 / den).conj())
#        print nom / den2
#

##%%
#h1, h2 = vmp.halves(tw.rect)
##h1, h1 = h1 * (1-1.5*drift[:,None]), h2 * (1-5*drift[:,None])
#h1, h2 = h1.ravel(), h2.ravel()
##h = h1 + h2
#print 'I am here'
#Fp = np.dot(ab.T, h1)
##tau2 = 5.59E-9
#tau2 = 1
##alpha = 5E7
#
##%%
##ai = np.ones(1584) * 251
#its = 100000
#cond = [[0, 0]]
#alpha = 1E5
##tau2 = 3E-9
#D = np.linalg.inv(FtF + alpha * np.eye(1232))
#reg_step =  np.dot(D, Fp)
##%%
#for i in np.arange(its):
#    a_0 = reg_step[:176]
#    a_I = np.tile(a_0, 7)
#    a_norm = np.reshape((reg_step / a_I),(7,176))
#
#    neg_I = np.where(a_0 < 0)[0]
#    reg_fac = (-1 * np.array([4, 1, 1, 1, 1, 1, 1]) / 2.)
#    a_norm[:, neg_I] *= reg_fac[:, None]
##    norm_sum = a_norm[1:7+1,:].sum(axis=0)
#
##    up_bnd = np.where(norm_sum > 2.2)[0]
##    lo_bnd = np.where(norm_sum < -1.2)[0]
##    a_norm[1:7+1,up_bnd] = 2.2 * a_norm[1:7+1,up_bnd] / norm_sum[up_bnd]
##    a_norm[1:7+1,lo_bnd] = -1.2 * a_norm[1:7+1,lo_bnd] / norm_sum[lo_bnd]
#
#    ai =  a_I * np.ravel(a_norm)
#    reg_step = ai + tau2 * np.dot(D, (Fp - np.dot(FtF, ai)))
#
#    if i % 1000 == 0:#  and i > 40000:# and i < 2500:
#
#        nrm = sp.linalg.norm(h2 - np.dot(ab, ai))
#        cond.append([i,nrm])
#        print '***', i * 100. / its, 'per cent', '***'
#        print 'Norm:', nrm, 'Gradient:', nrm - cond[-2][1]
#        print neg_I.shape[0], 'negative intenss. and'
#        print up_bnd.shape[0] + lo_bnd.shape[0], 'beta values constrained'
#        print '\n'
#
##%%
#
#c =np.asarray(cond)
#cx, cy = c.T[0],c.T[1]
#pl.plot(cx[3:],cy[3:])
##recon = np.dot(bs.T,ai)
##pl.figure()
##pl.imshow(vmp.unfold(recon.T,v=False,h=True))
#pl.figure()
#pl.imshow((h2 - np.dot(ab, ai)).reshape(501,251))
#
##pl.imshow(vmp.unfold((recon<0).T,v=False,h=True))
##
##    he.find_centre()
##    he.invertBasex()
##    he.invertFourierHankel()
##    cmp = np.append(he.bsx[:,:250], abs(he.FHT), axis=1)
##    pl.imshow(cmp, origin='lower')
