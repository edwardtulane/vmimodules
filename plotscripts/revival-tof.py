# -*- coding: utf-8 -*-
"""
Created on Sun Feb 15 21:54:51 2015

@author: felix
"""

path = '/home/felix/vmi/2015-02-10/Seq/2015-02-10-4/'
tof = np.load(path+'TOFtraces.npy')
pos = np.load(path+'TgtPositions.npy')

t1, t2, t3 = np.split(tof[:3*91],3)

tof2 = (t1 + t2[::-1] + t3) / 3
pos *= 6671


moz = (np.arange(700)**2) * 9.765E-4 -1.06E1

contourf(moz[:600], pos[:90], np.diff(tof2[:,:600],axis=0), 400, cmap=cm.seismic,vmin=-4,vmax=4)
plt.colorbar()

