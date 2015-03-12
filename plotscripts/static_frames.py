# -*- coding: utf-8 -*-
"""
Created on Mon Feb 16 00:03:57 2015

@author: felix
"""

from lib.vmivis import *

import matplotlib.pyplot as plt
import lib.vmiproc as vmp
from matplotlib.colors import LogNorm

#plt.style.use('dark_background')

path = '/home/felix/vmi/procd/2015-02-05/'
fra1 = np.load(path+'Iplusinvert.npy')
fra2 = np.load(path+'I2plusinverts.npy')
fra3 = np.load(path+'I3plusinverts.npy')

int1 = np.load(path+'Iplusints.npy')
int2 = np.load(path+'I2plusints.npy')
int3 = np.load(path+'I3plusints.npy')

R = np.arange(251)
E = R**2 /2900.


#plt.plot(E*1.,(int1[3,0]*np.arange(251)).T, label='I+', lw=2.0)
#plt.plot(E*2.,(int2[3,0]*np.arange(251)).T, label='I2+', lw=2.0)
#plt.plot(E*3.,5*(int3[3,0]*np.arange(251)).T, label='I3+ * 5', lw=2.0)
#
#plt.xlim([0,15])
#plt.ylim([0,300])
#
#plt.ylabel('Intensity [arb. u.]', fontsize=14)
#plt.xlabel('Kinetic Energy [eV]', fontsize=14)
#plt.legend(fontsize=14)
#plt.savefig('ion-raddist.svg')

###
###
###

#iplusses = (int2[:,0] *R).T
##iplusses /= iplusses[17]
#
#plt.figure()
#plt.plot(E*3.,iplusses, label='I+', lw=2.0)
#
#plt.xlim([0,15])
#plt.ylim([0,300])
#
#plt.ylabel('Intensity [arb. u.]', fontsize=14)
#plt.xlabel('Kinetic Energy [eV]', fontsize=14)
#plt.legend(fontsize=14)
p = np.linspace(-1.757, 1.757, 501) / 2.
plt.figure()
cont = plt.contourf(p,p,vmp.unfold(fra1[3,0],1,1), 950, cmap=cm.gnuplot2,vmin=0,vmax=120)


plt.xlabel('px [a. u.]',fontsize=14)
plt.ylabel('py [a. u.]',fontsize=14)
plt.colorbar()

plt.figure()
cont = plt.contourf(p*2.,p*2.,vmp.unfold(fra2[1,0],1,1), 950, cmap=cm.gnuplot2,vmin= -10,vmax=40)


plt.xlabel('px [a. u.]',fontsize=14)
plt.ylabel('py [a. u.]',fontsize=14)
plt.colorbar()

plt.figure()
cont = plt.contourf(p*3.,p*3.,vmp.unfold(fra3[3,0],1,1), 950, cmap=cm.gnuplot2,vmin=0, vmax=3)


plt.xlabel('px [a. u.]',fontsize=14)
plt.ylabel('py [a. u.]',fontsize=14)
plt.colorbar()
