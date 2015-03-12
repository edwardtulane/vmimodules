# -*- coding: utf-8 -*-
"""
Created on Sun Feb 15 21:54:51 2015

@author: felix
"""

import matplotlib.pyplot as plt

path = '/home/felix/vmi/2015-02-11/Seq/2015-02-11-4/'
tof = np.load(path+'TOFtraces.npy')
pos = np.load(path+'TgtPositions.npy')
tof1 = tof[::2]
tof2 = tof[::2] - tof[1::2]
pos *= 6671


moz = (np.arange(700)**2) * 1.19E-3 -9.3

cont = plt.contourf(moz[:500], pos[4:58], tof1[4:58,:500], 400, cmap=plt.cm.gnuplot2,vmin=-15,vmax=60)

plt.xlabel('m/z [amu]',fontsize=14)
plt.ylabel('t [fsec]',fontsize=14)
plt.colorbar()
