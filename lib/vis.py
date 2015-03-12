# Basic VMI data visualization

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import LogNorm, Normalize, SymLogNorm
import matplotlib.style as style
from matplotlib.widgets import Slider, Button

# Scientific constants
H=27.211; #Hartree in [eV]

def vmiplot(frame):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    fig.subplots_adjust(left=0.25, bottom=0.25)
    vmin, vmax = frame.min(), frame.max()

    clrmap = plt.cm.gnuplot2

    img = ax.imshow(frame, cmap=clrmap, origin='lower',
                     interpolation='sinc', norm=Normalize(vmin=vmin, vmax=vmax))
    cbar = fig.colorbar(img)

    axcolor = 'lightgoldenrodyellow'
    axmin = fig.add_axes([0.25, 0.1, 0.65, 0.03], axisbg=axcolor)
    axmax = fig.add_axes([0.25, 0.15, 0.65, 0.03], axisbg=axcolor)

    smin = Slider(axmin, 'vmin', vmin, vmax, valinit=vmin)
    smax = Slider(axmax, 'vmax', vmin , vmax, valinit=vmax)

    def update(val):
        upd = ax.imshow(frame, cmap=clrmap, origin='lower',
                     interpolation='sinc', norm=Normalize(vmin=smin.val, vmax=smax.val))
        fig.canvas.update()
    smin.on_changed(update)
    smax.on_changed(update)

    return fig, ax

def logplot(frame, vmin=0, vmax=0):
    fig = plt.figure()
    ax = fig.add_subplot(111)
#    fig.subplots_adjust(left=0.25, bottom=0.25)
    if not vmin: vmin = frame.min().clip(min=1)
    if not vmax: vmax = frame.max()

    clrmap = plt.cm.gnuplot2

    img = ax.imshow(frame, cmap=clrmap, origin='lower',
                     interpolation='sinc', norm=LogNorm(vmin=vmin, vmax=vmax))
    cbar = fig.colorbar(img)

def diffplot(fra, frb, fac=1.0):
    frame = fra - fac * frb
    fig = plt.figure()
    ax = fig.add_subplot(111)
    fig.subplots_adjust(left=0.25, bottom=0.25)
    vsym = np.max([np.abs(frame.min()), np.abs(frame.max())])
    clrmap = plt.cm.seismic

    img = ax.imshow(frame, cmap=clrmap, origin='lower',
                     interpolation='sinc', norm=Normalize(vmin=-1*vsym, vmax=vsym))
    cbar = fig.colorbar(img)

    axcolor = 'lightgoldenrodyellow'
    axfac = fig.add_axes([0.25, 0.1, 0.65, 0.03], axisbg=axcolor)
    axsym = fig.add_axes([0.25, 0.15, 0.65, 0.03], axisbg=axcolor)

    sfac = Slider(axfac, 'factor', 0.8, 1.2, valinit=fac)
    ssym = Slider(axsym, 'scale', 0.5 * vsym , 1.5 * vsym, valinit=vsym)

    def update(val):
        diff = (fra - sfac.val * frb) #/ (fra + sfac.val * frb)
        upd = ax.imshow(diff, cmap=clrmap, origin='lower',
                     interpolation='sinc', norm=Normalize(vmin=-1*vsym, vmax=vsym))
        fig.canvas.update()
    sfac.on_changed(update)
    ssym.on_changed(update)

def difflog(fra, frb, fac=1.0, thresh=10):
    frame = fra - fac * frb
    fig = plt.figure()
    ax = fig.add_subplot(111)
    fig.subplots_adjust(left=0.25, bottom=0.25)
    vsym = np.max([np.abs(frame.min()), np.abs(frame.max())])
    clrmap = plt.cm.seismic

    img = ax.imshow(frame, cmap=clrmap, origin='lower',
                     interpolation='sinc', norm=SymLogNorm(linthresh=thresh, vmin=0, vmax=vsym))
    cbar = fig.colorbar(img)

    axcolor = 'lightgoldenrodyellow'
    axfac = fig.add_axes([0.25, 0.1, 0.65, 0.03], axisbg=axcolor)
    axsym = fig.add_axes([0.25, 0.15, 0.65, 0.03], axisbg=axcolor)

    sfac = Slider(axfac, 'factor', 0.8, 1.2, valinit=fac)
    ssym = Slider(axsym, 'scale', 0.5 * vsym , 1.5 * vsym, valinit=vsym)

    def update(val):
        diff = fra - sfac.val * frb
        upd = ax.imshow(diff, cmap=clrmap, origin='lower',
                     interpolation='sinc', norm=SymLogNorm(linthresh=thresh, vmin=0, vmax=1))
        fig.canvas.update()
    sfac.on_changed(update)
    ssym.on_changed(update)



