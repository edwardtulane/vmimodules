# -*- coding: utf-8 -*-
"""
Created on Mon Feb 16 04:26:15 2015

@author: felix
"""

import numpy as np
import matplotlib.pyplot as plt


path = '/home/brausse/vmi/sf/procd/2015-03-01/2015-03-01'

iplusint, iplusbeta  = np.load(path + '-1/intmap.npy'), np.load(path + '-1/betamap.npy')
#i2plusint, i2plusbeta  = np.load(path + '-4/intmap.npy'), np.load(path + '-4/betamap.npy')
#i3plusint, i3plusbeta  = np.load(path + '-5/intmap.npy'), np.load(path + '-5/betamap.npy')
#revivalint, revivalbeta = np.load(path + '-6/intmap.npy'), np.load(path + '-6/betamap.npy')

pos = np.linspace(-.5, .5, 151)
pos *= 6671

R = np.arange(251)
E = R**2 /2900.

cont = plt.contourf(E[:130]*1., pos, (iplusint*R)[:,:130], 800, cmap=plt.cm.gnuplot2,vmax=80, vmin=0)

plt.xlabel('Kinetic Energy [eV]',fontsize=14)
plt.ylabel('t [fsec]',fontsize=14)
plt.colorbar()

###
###
plt.figure()
cont = plt.contourf(E[:130]*1., pos, (iplusbeta*R)[:,:130], 800, cmap=plt.cm.seismic,vmin =-160, vmax=160)

plt.xlabel('Kinetic Energy [eV]',fontsize=14)
plt.ylabel('t [fsec]',fontsize=14)
plt.colorbar()

plt.show()
