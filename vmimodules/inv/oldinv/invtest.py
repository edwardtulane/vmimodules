# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 14:35:59 2014

@author: brausse
"""

import sys, os
sys.path.insert(0, os.path.realpath(os.path.pardir))

import numpy as np
import scipy as sp
import pylab as pl

from scipy import integrate

import lib.proc as vmp
import lib.vminewclass as vmic
import lib.vis as vmv

from scipy.ndimage.interpolation import map_coordinates

from pBasex import AbelTrans


def testimage(r, the):
    I = \
    2000 * (7 * np.exp(-1*(r - 10)**2/4) * np.sin(the) ** 2 + \
    3 * np.exp(-1*(r - 15) ** 2 / 4) + \
    5 * np.exp(-1*(r - 20)**2/4) * np.cos(the) ** 2) + \
    200 * (np.exp(-1*(r - 70)**2/4) + 2 * np.exp(-1*(r - 85)**2/4) * np.cos(the) ** 2 + \
    np.exp(-1*(r - 100)**2/4) * np.sin(the) ** 2) + \
    50 * (2 * np.exp(-1*(r-145)**2/4) * np.sin(the) ** 2  + \
    np.exp(-1*(r-150)**2/4) + 3 * np.exp(-1*(r-155)**2/4)* np.cos(the) ** 2)# + \
#    20 * np.exp(-1*(r-45)**2/3600)

    return I


rad=250  
XY = np.arange(-1 * rad, rad +1,dtype='float64')
diam = XY.shape[0]
R = np.sqrt(XY ** 2 + XY[:, None] ** 2)#[rad:,rad:]
th = sp.arctan2(XY, XY[:,None])
#th = np.nan_to_num(th)#[rad:,rad:]
#th[rad+1:,:] -= np.pi

def test_integ(the, r):
    return testimage(r, the) * np.abs(np.sin(the)) * r ** 2
    
I_ref = np.zeros(251)
for i, r  in enumerate(XY[rad:]):
    I_ref[i], er = integrate.quad(test_integ, 0, pi, args=r)
#    print er

test_org = testimage(R, th)

test = vmp.fold(test_org, 0 , 1)
forward = AbelTrans(test)

img = vmic.VMIimage(vmp.unfold(forward, 0, 1), xcntr=250, ycntr=250, radius=250)
img.rect = img.prist
img.invertBasex()
bsx = vmic.VMIimage(img.bsx, xcntr=250, ycntr=250, radius=250)
bsx.interpol()
bsx.offset = 0
bsx.rotateframe()
#img.invertFourierHankel()

def bsx_integ(the, r):
    return map_coordinates(bsx.ck, [[r * cos(the) + 250], [r * sin(the) + 250]]) * np.abs(np.sin(the)) * r ** 2
I_quadbsx = np.zeros(251)
for i, r  in enumerate(XY[rad:]):
    I_quadbsx[i], er = integrate.quad(bsx_integ, 0, pi, args=r)
    print er

fld = np.ravel(vmp.fold(forward,1,0))
pbsx = np.dot(np.linalg.inv(FtF + 1 * np.eye(2500)), np.dot(ab, fld))
rec = np.dot(bs, pbsx)
#rec = vmp.unfold(rec, 1, 1)
