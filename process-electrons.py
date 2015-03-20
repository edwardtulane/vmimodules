# -*- coding: utf-8 -*-
"""
Created on Fri Feb 13 04:49:20 2015

@author: felix
"""

import os
import re

import numpy as np
import scipy.optimize as opt
from progressbar import ProgressBar

import lib.vmiclass as vmc
import lib.proc as vmp
#from lib.vmiproc import global_dens as gd
from lib.vis import *
import lib.inv as vminv

def getint(name):
        basename = name.partition('.')[0]
        num = basename.split('-')[-1]
        return int(num)

image_dir = os.path.expanduser('~/vmi/tw')
date = '2014-07-28'
seq_no = '-5'

inpath = os.path.join(image_dir, date)
outpath = os.path.join(image_dir, 'procd', date)

if not os.path.exists(outpath):
    os.mkdir(outpath)

filelist = os.listdir(inpath)
regex = re.compile('raw$')
raws = [l for l in filelist for m in [regex.search(l)] if m]
raws.sort(key=getint)
seq_len = len(raws[29:])
gd = 301

frames = np.ndarray((seq_len, gd, gd))
intmap = np.ndarray((seq_len))
disps = np.zeros((seq_len, 3))
facs = np.zeros(seq_len)

for i, file in enumerate(raws[29:]):
    filepath = os.path.join(inpath, file)
    rawfile = vmc.RawImage(filepath, xcntr=230, ycntr=235, radius=150
                           )
    fr = rawfile.cropsquare()
    fr.interpol()
    frames[i] = fr


    print file, ' processed and inverted.'

bg = frames[-1]
rects = np.zeros((seq_len-1, 4, 151, 151))
pbsx = np.zeros((seq_len-1, 4, 3900))
inv = vminv.Inverter(150, 50)
coeffs = (np.arange(seq_len-2) + 0.5) * (np.pi / 2)
coeffs = np.sign(np.cos(coeffs))
pb = ProgressBar().start()
for i, fr in enumerate(frames[:-1]):
    o = opt.minimize(vmp.find_bg_fac, [2.0], bounds=[[0.1, 5.0]], args=(fr, bg), method='L-BFGS-B')
    facs[i] = o.x
    fr = vmc.Frame(fr - o.x * bg)
    fr.offset = -55
# run 1   fr.disp = np.array([0.0, -1.045, -0.3825])
# run 2   fr.disp = np.array([0.0, -2.1983, 1.462])
    fr.disp = np.array([0, -1.1615, -0.58898])
#   fr /= fr.sum()
    fr.interpol()
#   fr.find_centre()
#   disps[i] = fr.disp
    r = fr.evalrect(301)
    rects[i] = vmp.quadrants(r)
    for j in np.arange(4):
        pbsx[i,j] = inv.invertPolBasex(rects[i,j])
    pb.update((i+1) * 2)

al = pbsx[coeffs.clip(0).astype(bool)]
an = pbsx[coeffs.clip(-1, 0).astype(bool)]
#
#
print 'Done. Saving...'
#
np.save(outpath+'/frames2', rects)
np.save(outpath+'/inverts2', pbsx)
np.save(outpath+'/coeffs2', coeffs)
np.save(outpath+'/al2', al)
np.save(outpath+'/an2', an)
