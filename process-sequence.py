# -*- coding: utf-8 -*-
"""
Created on Fri Feb 13 04:49:20 2015

@author: felix
"""

import os
import re

import numpy as np

import lib.vminewclass as vmc
import lib.vmiproc as vmp
from lib.vmiproc import global_dens as gd
from lib.vmivis import *
import lib.vminv as vminv

image_dir = '/home/brausse/vmi/sf/'
date = '2015-03-01'
seq_no = '-1'

inpath = os.path.join(image_dir, date, 'Seq', date + seq_no)
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

frames, inverts = np.ndarray((seq_len, gd, gd)), np.ndarray((seq_len, 251, 251))
intmap = np.ndarray((seq_len, 251))
betamap = intmap.copy()

rect = np.zeros((seq_len, 501,501))
for i, file in enumerate(raws):
    filepath = os.path.join(inpath, file)
    rawfile = vmc.RawImage(filepath, xcntr=514, ycntr=468, radius=250,
                           hotspots=[[325, 480], [45, 610], [501,703]])
    fr = rawfile.cropsquare()
    fr.interpol()
    rect = fr.evalrect()
    bsx = rect.pBasex()
#    fold = vminv.pbsx2fold(bsx)
#    frames[i], inverts[i] = rect, fold

    intmap[i], betamap[i] = vminv.pbsx2rad(bsx)

    print file, ' processed and inverted.'

# print 'Done. Saving...'

#for i, j in enumerate(np.arange(8) + 7):
#    frames[i] = rect[j]
#    fr = vmc.RectImg(frames[i])
#    bsx = fr.Basex()
#    intmap[i] = bsx.raddist()

# np.save(outpath+'/frames', frames)
# np.save(outpath+'/inverts', inverts)
np.save(outpath+'/intmap', intmap)
np.save(outpath+'/betamap', betamap)

#ifra = np.save(outdate + '/Iplusframes.npy')
#iint = np.save(outdate + '/Iplusints.npy')
#iinv = np.save(outdate + '/Iplusinvert.npy')
