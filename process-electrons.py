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

image_dir = '/home/felix/vmi/'
date = '2015-02-10'
seq_no = '-5'

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
intmap = np.ndarray((seq_len))

for i, file in enumerate(raws):
    filepath = os.path.join(inpath, file)
    rawfile = vmc.RawImage(filepath, xcntr=513, ycntr=468, radius=250,
                           hotspots=[[325, 480], [45, 610], [501,703]])
    fr = rawfile.cropsquare()
    fr.interpol()
    rect = fr.evalrect()
#    bsx = rect.pBasex()
#    fold = vminv.pbsx2fold(bsx)
    frames[i] = rect

    intmap[i] = fr.sum()

    print file, ' processed and inverted.'

# print 'Done. Saving...'

img = np.zeros((28, gd, gd))
for i in np.arange(28):
    img[i] = frames[i:238:28].sum(axis=0)


seq_no = '-6'

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
intmap = np.ndarray((seq_len))

for i, file in enumerate(raws):
    filepath = os.path.join(inpath, file)
    rawfile = vmc.RawImage(filepath, xcntr=513, ycntr=468, radius=250,
                           hotspots=[[325, 480], [45, 610], [501,703]])
    fr = rawfile.cropsquare()
    fr.interpol()
    rect = fr.evalrect()
#    bsx = rect.pBasex()
#    fold = vminv.pbsx2fold(bsx)
    frames[i] = rect

    intmap[i] = fr.sum()

    print file, ' processed and inverted.'

for i in np.arange(28):
    img[i] = frames[i:238:28].sum(axis=0)

a, b = img[0::2], img[1::2]

rects = np.zeros((7,2,501,501))
for i in np.arange(7):
    rects[i] = [a[i], b[i]]
    rects[i] += [a[13-i], b[13-i]]

inverts = np.zeros((7,2,501,501))
intmap = np.zeros((7,2,251))


for i in np.arange(7):
    on, off = rects[i]
    on, off = vmc.RectImg(on), vmc.RectImg(off)
    bsx1, bsx2 = on.Basex(), off.Basex()
#    fold1, fold2 = vminv.pbsx2fold(bsx1), vminv.pbsx2fold(bsx2)
    inverts[i] = [bsx1, bsx2]
    intmap[i] = [bsx1.raddist(), bsx2.raddist()]

np.save(outpath+'/frames', rects)
np.save(outpath+'/inverts', inverts)
np.save(outpath+'/intmap', intmap)
