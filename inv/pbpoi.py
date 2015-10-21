import numpy as np, scipy as sp
import vmimodules as vm
from matplotlib.colors import LogNorm
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
def test_integ(the, r):
        return testimage(r, the) * np.abs(np.sin(the)) * r ** 2
# I_ref = np.zeros(251)
# for i, r  in enumerate(XY[rad:]):
#     I_ref[i], er = integrate.quad(test_integ, 0, pi, args=r)
#    print er
test_org = testimage(R, th)
test = vm.proc.fold(test_org, 0 , 1)
forward = AbelTrans(test)
def like_gauss(img, vec, inv):
        sol = vec.dot(inv.ab)
        return np.sum((img - sol) ** 2 * 0.5)
def like_poiss(vec, img, inv):
        sol = vec.dot(inv.ab)
        thr = np.percentile(img[img>0], 1)
        ma = (img > thr) & (sol > thr)
    #     
        lk = sol[ma].sum() - (img * np.log(sol))[ma].sum() # + sp.special.gammaln(img)[ma].sum()
    #     
        der1 = img / sol - 1
        der2 = inv.ab[:,ma].dot(der1[ma])
    # 
        return lk#, -der2
from scipy.optimize import minimize
i = vm.proc.fold(forward,1,0).ravel()
inv = vm.Inverter(250, 8)
pbsx = inv.invertPolBasex(vm.proc.fold(forward,1,0), get_pbsx=True)
x = minimize(like_poiss, pbsx, args=(i, inv), jac=False, options={'disp':True}, method='CG')
