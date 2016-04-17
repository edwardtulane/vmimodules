# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 14:31:32 2014

@author: brausse
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import scipy as sp
import pylab as pl
import vmiproc as vmp

pl.close('all')

img = np.load('img/bb3al.npy')
img = frames[0]
if not 'ab' in dir():
    ab = np.load('storage/ab-250-18.npy')
if not 'FtF' in dir():
    FtF =np.load('storage/FtF-250-18.npy')

R =np.arange(250)
R = np.tile(R, 7)
R2 = R**2

#%%
h1, h2, q3, q4 = vmp.quadrants(img)
#h1, h1 = h1 * (1-1.5*drift[:,None]), h2 * (1-5*drift[:,None])
h1, h2 = h1.ravel(), h2.ravel()
#h = h1 + h2
print('I am here')
Fp = np.dot(ab, h2)
#tau2 = 5.59E-9
#tau2 = 1

#%%
its = 600000
cond = [[0, 0]]
alpha = 5E12
tau2 = 7.5E-9
#D = np.linalg.inv(FtF + alpha * np.eye(2000))
#reg_step =  np.dot(D, Fp)
reg_step = np.ones(2500) * 14.5
break_flag = False
#%%
for i in np.arange(its):
#    a_0 = reg_step[:200]
#    a_I = np.tile(a_0, 11)
#    a_norm = np.reshape((reg_step / a_I),(11,200))

#    neg_I = np.where(a_0 < 0)[0]
#    reg_fac = (-1 * np.array([4, 1, 1, 1, 1, 1, -2, -2, -2, -2, -2]) / 2.)
#    a_norm[:, neg_I] *= reg_fac[:, None]

#    norm_sum = a_norm[1:6+1,:].sum(axis=0)
#    lo_bnd = np.where(norm_sum < -1.5)[0]
#    a_norm[1:6+1,lo_bnd] = -1.5 * a_norm[1:6+1,lo_bnd] / norm_sum[lo_bnd]
#
#    a_norm = np.ravel(a_norm)
#    up_bnd = np.where(a_norm[:200 * 6 + 1] > 3.0)[0]
#    a_norm[up_bnd] = 2.5

    ai = reg_step#a_I * a_norm
    reg_step = ai + tau2 *(Fp - np.dot(FtF, ai))

    if i % 1000 == 0:
        nrm = np.linalg.norm(h1 - np.dot(ai, ab).T)
        cond.append([i,nrm])
        grd = nrm - cond[-2][1]
        print('***', i * 100. / its, 'per cent', '***')
        print('Norm:', nrm, 'Gradient:', grd)

        if np.abs(grd / nrm) < 1E-5:
            pass
#            break_flag = True

    if break_flag:
        break

#        print(neg_I.shape[0], 'negative intenss. and')
#        print(up_bnd.shape[0], 'and', lo_bnd.shape[0], 'beta values constrained')
        print('\n')

#%%
pl.plot(ai)
pl.figure()
c =np.asarray(cond)
cx, cy = c.T[0],c.T[1]
pl.plot(cx[3:],cy[3:])
#recon = np.dot(bs,ai)
#recon = recon.reshape(501,251)
#pl.figure()
#pl.imshow(vmp.unfold(recon,v=False,h=True))
pl.figure()
pl.imshow((h2 - np.dot(ab.T, ai)).reshape(501,251))

#pl.imshow(vmp.unfold((recon<0).T,v=False,h=True))
