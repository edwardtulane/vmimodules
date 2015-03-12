# -*- coding: utf-8 -*-
"""
Created on Mon Feb 16 05:37:45 2015

@author: felix
"""

import numpy as np
import matplotlib.pyplot as plt


path1 = '/home/felix/vmi/procd/2015-02-09/'
path2 = '/home/felix/vmi/procd/2015-02-10/'

i2 = np.load(path1 + '2015-02-09-5/intmap.npy')

eleframes = np.load(path2 + '2015-02-10-6/frames.npy')
eleinv = np.load(path2 + '2015-02-10-6/inverts.npy')

p = np.linspace(-1.757, 1.757, 501)


pos = np.linspace(2.32, 2.45, 61)
pos *= 6671

R = np.arange(251)
E = R**2 /2900.

plt.figure()
cont = plt.contourf(E[:130]*1., pos, (i2*R)[:,:130], 400, cmap=plt.cm.gnuplot2,vmax=4000)

plt.xlabel('Kinetic Energy [eV]',fontsize=14)
plt.ylabel('t [fsec]',fontsize=14)
plt.colorbar()

###
###

plt.figure()
cont = plt.contourf(p,p, eleinv[4,0] - 1.07 * eleinv[4,1], 400, cmap=plt.cm.seismic,vmin=-750,vmax=750)

plt.xlabel('px [a. u.]',fontsize=14)
plt.ylabel('py [a. u.]',fontsize=14)
plt.colorbar()

###
###

plt.figure()
cont = plt.contourf(p,p, eleinv[3,0] - 1.07 * eleinv[3,1], 400, cmap=plt.cm.seismic,vmin=-750,vmax=750)

plt.xlabel('px [a. u.]',fontsize=14)
plt.ylabel('py [a. u.]',fontsize=14)
plt.colorbar()

###
###

plt.figure()
cont = plt.contourf(p,p, eleinv[4,0] - 1.01 * eleinv[3,0], 400, cmap=plt.cm.seismic,vmin=-250,vmax=250)

plt.xlabel('px [a. u.]',fontsize=14)
plt.ylabel('py [a. u.]',fontsize=14)
plt.colorbar()

plt.figure()
cont = plt.imshow((eleinv[4,0] - 1.01 * eleinv[3,0])/(eleinv[4,0] + 1.01 * eleinv[3,0]), cmap=plt.cm.seismic,vmin=-0.05,vmax=0.05)


plt.xlabel('px [a. u.]',fontsize=14)
plt.ylabel('py [a. u.]',fontsize=14)
plt.colorbar()