# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 11:01:45 2014

@author: krecinic
"""

viewerpath='/home/brausse/program/lab/RawViewer'

from PyQt4 import QtGui, QtCore 

import matplotlib
matplotlib.use("Qt4Agg")
import matplotlib.pyplot as p
    
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
#from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
from matplotlib.figure import Figure
#from matplotlib.colors import Normalize, LogNorm

import sys
sys.path.append(viewerpath)
import pdb
import pylab as pyl
import numpy as np
import vmiproc as vmp
import scipy.ndimage as ndim

class RawDisplayWindow(QtGui.QDialog):
    
    userMessage = QtCore.pyqtSignal(str, int)
    
    def __init__(self, parent=None):
        QtGui.QDialog.__init__(self, parent)
        self.main_frame = QtGui.QWidget()
        self.setWindowTitle('RawDisplay')

        # Create the figure
        self.dpi = 100
        self.fig = Figure((10.0, 10.0), dpi=self.dpi)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setParent(self.main_frame)

        # Create the navigation toolbar, tied to the canvas
#        self.mpl_toolbar = NavigationToolbar(self.canvas, self.main_frame)
        self.axes = self.fig.add_subplot(111)
        vbox = QtGui.QVBoxLayout()
        vbox.addWidget(self.canvas)
        #vbox.addWidget(self.mpl_toolbar)
        self.setLayout(vbox)
                
        # Initialize the processing parameters
        self.center = False
        self.centerX = 0
        self.centerY = 0
        
        self.crop = False
        self.cropR = 0
        
        self.rotate = False
        self.rot90deg = False
        self.rotateDegree = 0.0
        
        self.invert = False
        self.M,self.Mc = vmp.iniBasex(viewerpath+'/')
        
        self.cmname = 'jet'
        self.cmnorm = False
        self.cmlog = False
        self.csmanual = False
        self.csmin = 0.
        self.csmax = 1.
        
        self.grids = False
        self.xylines = False
        self.xygrid = False
        self.calibc = 1000.
        self.Egridvals = np.array([])
        self.ptickvals = np.array([])        
        self.hartree=27.211; #Hartree in [eV] 

        self.holdaxlim = False

        # Create empty image axis and plot initializing image
        self.imaxes = []
        self.cbar = []
        self.rawimg = []
        self.procimg = []
        self.initImage()

    def initImage(self):
        self.rawimg=np.zeros([200,200])
        for p in range(self.rawimg.shape[0]):
            self.rawimg[p]=p
        # Make sure we zoom to new image extent
        self.resetaxlim = True
        self.processRawImage()
    
    def openFile(self, fname):
        # Check file extentions and load 
        fname = str(fname)
        ext = fname.split('.')[-1];
        if ext == 'raw':
            self.rawimg = vmp.rawread(fname)
        elif ext == 'npy':
            self.rawimg = np.load(fname)
        else:
            self.emitUserMessage('Invalid file extension or type.',2)
            return
        # Process image and show
        self.processRawImage()
        # Make sure the window is visible again in case user closed it
        self.show()
    
    # --- Process the raw image with specified parameters 
    #     (e.g. center, rotate, etc.) ---
    def processRawImage(self):
        # Make a copy of the original image
        self.procimg=np.float_(self.rawimg.copy())
        # Apply processing
        self.centerImage()
        self.cropImage()
        self.rotateImage()
        self.invertImage()
        self.normImage()
        # Show the processed image
        self.showImage()
        
    def centerImage(self):
        if self.center == True and self.centerX != 0 and self.centerY != 0:
            self.procimg = vmp.center(self.procimg,self.centerX,self.centerY)
    
    def cropImage(self):
        if self.crop == True and self.cropR != 0:
            self.procimg = vmp.crop_square(self.procimg,self.cropR)
            self.procimg = vmp.crop_circle(self.procimg,self.cropR)
        
    def rotateImage(self):
        if self.rotate == True:
            if self.rot90deg == True:
                self.procimg = np.rot90(self.procimg)
            if self.rotateDegree != 0.0:
                self.procimg = ndim.interpolation.rotate(self.procimg,self.rotateDegree,reshape=False)

    def invertImage(self):
        if self.invert == True:
            self.procimg = vmp.Basex(self.procimg,10,10,self.M,self.Mc)
            #invavbg=vmp.Basex(avbg,1,1,M,Mc)
        
    def normImage(self):
        self.procimgmin=np.amin(self.procimg)
        self.procimgmax=np.amax(self.procimg)
        self.procimgnorm = self.procimg/(self.procimgmax-self.procimgmin)
        # Normalize and log scale
        self.procimglog = np.log10((self.procimgnorm-np.amin(self.procimgnorm)))

    def emitUserMessage(self,message,level=0):
        self.userMessage.emit(message,level)    
    
    def showImage(self):
        # --- DEBUG --- 
#        QtCore.pyqtRemoveInputHook()
#        pdb.set_trace()
        # Draw the image
        # Store original axis limits and clear axis
        xlim=self.axes.get_xlim()
        ylim=self.axes.get_ylim()
        self.axes.clear()
        # Plot the new image data
        if self.csmanual == False:
            if self.cmnorm:
                self.imaxes = self.axes.imshow(self.procimgnorm, cmap=p.cm.gnuplot2,
                                               interpolation='none')
            elif self.cmlog:
                self.imaxes = self.axes.imshow(self.procimglog, cmap=p.cm.gnuplot2,
                                               interpolation='none')
            else:
                self.imaxes = self.axes.imshow(self.procimg, cmap=p.cm.gnuplot2,
                                               interpolation='none')
        else:
            if self.cmnorm:
                self.imaxes = self.axes.imshow(self.procimgnorm, cmap=p.cm.gnuplot2,
                                               vmin=self.csmin,vmax=self.csmax,interpolation='none')
            elif self.cmlog:
                self.imaxes = self.axes.imshow(self.procimglog, cmap=p.cm.gnuplot2,
                                               vmin=self.csmin,vmax=self.csmax,interpolation='none')
            else:
                self.imaxes = self.axes.imshow(self.procimg, cmap=p.cm.gnuplot2,
                                               vmin=self.csmin,vmax=self.csmax,interpolation='none')
            # Set over/under colors for the manual colorscale ranges
            cmap=self.imaxes.get_cmap()
            cmap.set_over('w')
            cmap.set_under('k')
        # Sync the colorbar
        if self.cbar == []:
            self.cbar = Figure.colorbar(self.fig,self.imaxes)
        self.cbar.set_cmap(self.imaxes.get_cmap())
        self.cbar.set_clim(self.imaxes.get_clim())
        self.cbar.update_normal(self.imaxes)
        # Restore axis limits 
        if self.holdaxlim:
            self.axes.set_xlim(xlim)
            self.axes.set_ylim(ylim)
        # Plot in further figure elements
        self.axes.set_autoscale_on(False)
        if self.grids:
            self.plotXYCenter()
            self.plotEgrid()
            self.plotpTicks()
            self.axes.grid(self.xygrid)
        if self.ptickvals.size == 0:
            self.axes.set_xlabel(r'X (pixel)')
            self.axes.set_ylabel(r'Y (pixel)')
        # Update the figure plot
        self.fig.canvas.draw()

    def plotXYCenter(self):
        # --- DEBUG --- 
#        QtCore.pyqtRemoveInputHook()
#        pdb.set_trace()
        if self.xylines:
            self.axes.hold(True)
            xs=np.shape(self.procimg)[0]
            ys=np.shape(self.procimg)[1]
            xc=(xs-1)/2
            yc=(ys-1)/2
            self.axes.plot([0,xs],[yc,yc],'k-')
            self.axes.plot([xc,xc],[0,ys],'k-')
            self.axes.hold(False)

    def plotEgrid(self):
        if self.Egridvals.size > 0:
            # Center circles on image
            xs=np.shape(self.procimg)[0]
            ys=np.shape(self.procimg)[1]
            xc=xs/2.
            yc=ys/2.
            # Generate the circular grid for iso-energy lines
            gridl=[np.cos(np.linspace(0,np.pi*2,100)), np.sin(np.linspace(0,np.pi*2,100))]
            # Plot circle for each energy value
            self.axes.hold(True)
            for E in self.Egridvals:
                if E==0.:
                    self.axes.plot(xc,yc,'k+')
                else:
                    xgrid=gridl[0]*np.sqrt(self.calibc*E)+xc
                    ygrid=gridl[1]*np.sqrt(self.calibc*E)+yc
                    self.axes.plot(xgrid,ygrid, 'k:')
                    self.axes.text(np.sqrt(self.calibc*E)+xc,yc,'{}'.format(E))
            self.axes.hold(False)
    
    def plotpTicks(self):
        if self.ptickvals.size > 0:
            # Center circles on image
            xs=np.shape(self.procimg)[0]
            ys=np.shape(self.procimg)[1]
            xc=xs/2.
            yc=ys/2.
            # Calculate momentum tick positions in image coordinates and set 
            # tick labels
            ptickpos=self.ptickvals*np.sqrt(self.calibc*self.hartree/2.)
            self.axes.set_xticks(ptickpos+xc)
            self.axes.set_xticklabels([str(x) for x in self.ptickvals])
            self.axes.set_yticks(ptickpos+yc)
            self.axes.set_yticklabels([str(x) for x in self.ptickvals])
            # Set the axis labels
            self.axes.set_xlabel(r'$p_x$ (a.u.)')
            self.axes.set_ylabel(r'$p_y$ (a.u.)')
