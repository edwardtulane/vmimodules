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

def getint(name):
        basename = name.partition('.')[0]
        num = basename.split('-')[-1]
        return int(num)

image_dir = os.path.expanduser('~/vmi/tw')
date = '2014-07-25'
seq_no = '-5'

inpath = os.path.join(image_dir, date)
outpath = os.path.join(image_dir, 'procd', date)

if not os.path.exists(outpath):
    os.mkdir(outpath)

filelist = os.listdir(inpath)
regex = re.compile('raw$')
raws = [l for l in filelist for m in [regex.search(l)] if m]
raws.sort(key=getint)
seq_len = len(raws)
gd = 301

frames, inverts = np.ndarray((seq_len, gd, gd)), np.ndarray((seq_len, 251, 251))
intmap = np.ndarray((seq_len))

for i, file in enumerate(raws):
    filepath = os.path.join(inpath, file)
    rawfile = vmc.RawImage(filepath, xcntr=229, ycntr=235, radius=150,
                           )
    fr = rawfile.cropsquare()
#   fr.interpol()
#   rect = fr.evalrect()
#    bsx = rect.pBasex()
#    fold = vminv.pbsx2fold(bsx)
    frames[i] = fr

#    intmap[i] = fr.sum()

    print file, ' processed and inverted.'

# print 'Done. Saving...'

#img = np.zeros((28, gd, gd))
#for i in np.arange(28):
#    img[i] = frames[i:238:28].sum(axis=0)
#
#
#
#
#frames, inverts = np.ndarray((seq_len, gd, gd)), np.ndarray((seq_len, 251, 251))
#intmap = np.ndarray((seq_len))
#
#for i, file in enumerate(raws):
#    filepath = os.path.join(inpath, file)
#np.save(outpath+'/frames', rects)
#np.save(outpath+'/inverts', inverts)
#np.save(outpath+'/intmap', intmap)
