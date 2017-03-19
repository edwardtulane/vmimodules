# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import sys, os, warnings
mod_home = os.path.dirname(__file__)
sys.path.insert(0, mod_home)

vmi_dir = os.path.expanduser('~/vmi')
gmd_loc = os.path.expanduser('~/vmi/procd/fel/gmd/gmd.h5')

global_dens = 301

from .vmiclass import RawImage, Frame, CartImg, PolarImg, ParseExperiment, ProcessExperiment, ParseSingleShots
from .vmiclass import header_keys, meta_keys, frame_keys, time_keys, inv_keys, singleshot_keys
from .inv import Inverter
from .vis import Plotter

from .hitdet import *

