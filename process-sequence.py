# -*- coding: utf-8 -*-
"""
Created on Fri Feb 13 04:49:20 2015

@author: felix
"""

import os
import re

import numpy as np

import lib.vmiclass as vmc
import lib.proc as vmp
#from lib.vmiproc import global_dens as gd
from lib.vis import *
import lib.inv as vminv

image_dir = '/home/brausse/vmi/tw/'
date = '2014-07-25'
seq_no = '-5'

inpath = os.path.join(image_dir, date, 
# 'Seq', 
date + seq_no)
outdate = os.path.join(image_dir, 'procd', date)
outpath = os.path.join(image_dir, 'procd', date, date + seq_no)

if not os.path.exists(outdate):
    os.mkdir(outdate)
if not os.path.exists(outpath):
    os.mkdir(outpath)

filelist = os.listdir(inpath)
regex = re.compile('raw$')
raws = [l for l in filelist for m in [regex.search(l)] if m]
raws.sort()
seq_len = len(raws)
gd = 301

frames = np.ndarray((seq_len, gd, gd))
rects = np.zeros((seq_len, 4, 151, 151))
pbsx = np.zeros((seq_len, 4, 3900))
inv = vminv.Inverter(150, 50)
disps = np.zeros((seq_len, 3))

for i, file in enumerate(raws):
    filepath = os.path.join(inpath, file)
    rawfile = vmc.RawImage(filepath, xcntr=225, ycntr=234, radius=150,
                           )
    fr = rawfile.cropsquare()
    fr.interpol()
    fr.offset=-55
#   fr.find_centre()
#   disps[i] = fr.disp
    fr.disp = [0,-1.04, 1.605]
    r = fr.evalrect(301)
    rects[i] = vmp.quadrants(r)
    for j in np.arange(4):
        pbsx[i,j] = inv.invertPolBasex(rects[i,j])


    print file, ' processed and inverted.'

# print 'Done. Saving...'

#for i, j in enumerate(np.arange(8) + 7):
#    frames[i] = rect[j]
#    fr = vmc.RectImg(frames[i])
#    bsx = fr.Basex()
#    intmap[i] = bsx.raddist()

np.save(outpath+'/frames', rects)
np.save(outpath+'/inverts', pbsx)
#np.save(outpath+'/intmap', intmap)
#np.save(outpath+'/betamap', betamap)

#ifra = np.save(outdate + '/Iplusframes.npy')
#iint = np.save(outdate + '/Iplusints.npy')
#iinv = np.save(outdate + '/Iplusinvert.npy')
