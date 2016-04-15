# -*- coding: utf-8 -*-
import sys, os, warnings
mod_home = os.path.expanduser('~/program/vmimodules')
sys.path.insert(0, mod_home)

vmi_dir = os.path.expanduser('~/vmi')
gmd_loc = os.path.expanduser('~/vmi/procd/fel/gmd/gmd.h5')

global_dens = 301

from lib import *
