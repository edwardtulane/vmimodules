commpath='/home/krecinic/Documents/Host Documents/Python/Common/'

import sys
sys.path.append(commpath)
import numpy as np
import vmiproc as vmp
import vmivis as vmv
import pylab as pyl

# Data parameters
proclab='20140612'
figext='.png'

# Plot the alignment degree <cos^2(\theta)>_{2D}
avcos2=np.load(proclab+'-avcos2.npy')
fig=pyl.figure()
ax=fig.add_subplot(111)
ax.plot(avcos2,'b.-')
pyl.xlabel('Raw image nr.')
pyl.ylabel(r'$<cos^2(\theta)>_{2D}$')
pyl.savefig(proclab+'-avcos2'+figext)

# Load the unaligned, alignment, anti-alignment frames
funalign=np.load(proclab+'-unaligned.npy')
falign=np.load(proclab+'-alignment.npy')
fantialign=np.load(proclab+'-anti-alignment.npy')

# Plot frames and save figures
fig=pyl.figure()
ax=fig.add_subplot(111)
ax.imshow(funalign,interpolation='none')
pyl.title('Un-aligned')
pyl.savefig(proclab+'-unaligned')

fig=pyl.figure()
ax=fig.add_subplot(111)
ax.imshow(falign,interpolation='none')
pyl.title('Maximum alignment')
pyl.savefig(proclab+'-alignment')

fig=pyl.figure()
ax=fig.add_subplot(111)
ax.imshow(fantialign,interpolation='none')
pyl.title('Maxumum anti-alignment')
pyl.savefig(proclab+'-anti-alignment')

pyl.show()
