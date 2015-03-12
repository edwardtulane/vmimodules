# -*- coding: utf-8 -*-
"""
Created on Fri Feb 13 11:38:25 2015

@author: felix
"""

import os
import re

import numpy as np

#import lib.vminewclass as vmc
import lib.vmiproc as vmp
#from lib.vmiproc import global_dens as gd
from lib.vmivis import *
import lib.vminv as vminv

image_dir = '/home/felix/vmi/'
date = '2015-02-05'
seq_no = '-6'

inpath = os.path.join(image_dir, date)
outdate = os.path.join(image_dir, 'procd', date)
outpath = os.path.join(image_dir, 'procd', date, date + seq_no)

ifra = np.load(outdate + '/Iplusframes.npy')
iint = np.load(outdate + '/Iplusints.npy')
iinv = np.load(outdate + '/Iplusinvert.npy')