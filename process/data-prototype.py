# This is a protoype data model for working with VMI Data

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

seq_dict = {'date' : pd.Timestamp(),
            'seq no' : # < 0 means top-level rawfiles
            'file'
            'frame index' : 
            }

raw_dict = {'Rep' :
            'Ext' :
            'MCP' :
            'Phos':
            'probe wavelength' :
            'pump wavelength'  :
            'molecule' :
            'acq no' :
            'background' :
            }

center_dict = {'coarse center x'
               'coarse center y'
               'offset angle'
               'rmax' :
               'mesh density'
               'centering method'
               'opt disp alpha'
               'opt disp x'
               'opt disp y'
               'fun(min)'
               }

frame_dict = { 'coarse center x'
               'coarse center y'
               'offset angle'
               'rmax' :
               'mesh density'
               'disp alpha'
               'disp x'
               'disp y'
               }

inv_dict = { 'inversion method'
            'l max'
            'odd l'
            'sigma'
            'total basis set size'
            'RSS'
            }
