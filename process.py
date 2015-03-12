commpath='/home/krecinic/Documents/Host Documents/Python/Common/'

import sys
sys.path.append(commpath)
import numpy as np
import vmiproc as vmp
import pylab as pyl
import scipy.ndimage as ndim

imbase='seq';
datadir='/home/krecinic/Documents/Host Documents/Raw data/2014-06-12/'
dir1='Seq/2014-06-12-1/'
imnrs1=np.arange(1,21);
framenr=20

proclab='20140612'
# Image center
center=[515,477];
# MCP radius on image
rmax=100
# Camera 'hotspot' pixels
hotspots=[[325,490],[45,620]]
# Angle of laser w.r.t. camera axis
rotoff=0.
# Reference axis for the cos**2 calculation
theta0=90.

# Read in all images and average frames 
frames=[]
for i in np.arange(0,framenr):
    frames.append(vmp.rawread(datadir+dir1+imbase + '{0:04}'.format(i+1) + '.raw'))

# Remove hotspots
for i in np.arange(0,framenr):
    frames[i]=vmp.hot_spots(frames[i],hotspots)
# Center, crop, rotate
for i in np.arange(0,framenr):
        # Center and crop
        frames[i]=vmp.center(frames[i],center[0],center[1])
        frames[i]=vmp.crop_square(frames[i],rmax)
        # Rotate frames
        if (rotoff != 0.):
            frames[i]=ndim.interpolation.rotate(frames[i],rotoff,reshape=False)
        frames[i]=np.rot90(frames[i])

# Calculate alignment values
avcos2=[]
for i in np.arange(0,framenr):
    avcos2.append(vmp.meancos2(frames[i],theta0,rmin=10.))
np.save(proclab+'-avcos2',avcos2)
# Save the frames corresponding to maximum alignment, anti-alignment and average
np.save(proclab+'-unaligned', frames[0])
np.save(proclab+'-alignment', frames[np.argmax(avcos2)])
np.save(proclab+'-anti-alignment', frames[np.argmin(avcos2)])
