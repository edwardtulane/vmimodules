import numpy as np
import vmimodules.lib as vm

devs = np.zeros((10,10))
inv = vm.Inverter(250, 8)
raw = vm.RawImage('../TWtest.raw', 249, 233)
f = raw.cropsquare(offset=-45)
f.interpol()
f.centre_pbsx(ang=False)
f.centre_pbsx(cntr=False)

#for i in np.arange(-5, 5, 1):
#    for j in np.arange(-5, 5, 1):
#        rect = f.evalrect(501, displace=np.array([i, j]))
#        quads = vm.proc.quadrants(rect)
#        pb = np.zeros((4, inv.FtF.shape[0]))
#        for k, img in enumerate(quads):
#            pb[k] = inv.invertPolBasex(img)
#        dev = pb.std(axis=0).sum()
#        devs[i+5, j+5] = dev
#
#        print i, j, dev
