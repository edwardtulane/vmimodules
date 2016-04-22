#!/usr/bin/python

# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 11:22:54 2014

@author: krecinic

Use RawViewer to quickly process and evaluate VMI images.
"""

from PyQt4 import QtGui, QtCore
from MainWindow import Ui_MainWindow
from RawDisplayWindow import RawDisplayWindow
import sys,pdb

import numpy as np

class MainUI(QtGui.QMainWindow):
    def __init__(self):
        QtGui.QWidget.__init__(self)
        self.UI = Ui_MainWindow()
        self.UI.setupUi(self)
        
        # Initialize class variables
        self.fname = ''
        self.fdir = QtCore.QDir()
        self.dname = ''
                
        # Initialize statusbar so we can display user messages
        self.status_text = QtGui.QLabel("Rawviewer v1.0, written by Faruk Krecinic")
        self.status_text.setWordWrap(True)
        self.statusBar().addWidget(self.status_text, 1)
        
        # Initialize image display window
        self.initRawDispWindow()        
        # Initialize main window buttons etc.
        self.initControlWindow()
        # Initialize menu actions
        self.initMenu()        

    def initRawDispWindow(self):
        # Create image viewing window
        self.rawdisplay = RawDisplayWindow(parent=self)
        self.rawdisplay.userMessage.connect(self.showUserMessage)

        # Retrieve settings from previous session and apply values
        settings=QtCore.QSettings('MBI','RawViewer')
        
        settings.beginGroup('rawdispwindow')
        v=settings.value('size',QtCore.QSize(1022,800)).toSize()
        self.rawdisplay.resize(v)
        v=settings.value('pos',QtCore.QPoint(0,0)).toPoint()
        self.rawdisplay.move(v)
        settings.endGroup()
        
        # Show window
        self.rawdisplay.show()
        self.rawdisplay.activateWindow()


    def initMenu(self):
        self.UI.action_Open.triggered.connect(self.openFile)
        self.UI.actionOpen_Next.triggered.connect(self.clickedPushNext)
        self.UI.actionOpen_Previous.triggered.connect(self.clickedPushPrev)
        self.UI.action_Quit.triggered.connect(self.close)
    
    def initControlWindow(self):
                # --- DEBUG --- 
#        QtCore.pyqtRemoveInputHook()
#        pdb.set_trace()

        # Retrieve settings from previous session and apply values
        settings=QtCore.QSettings('MBI','RawViewer')
        
        settings.beginGroup('mainwindow')        
        v=settings.value('size',QtCore.QSize(0,0)).toSize()
        self.resize(v)
        v=settings.value('pos',QtCore.QPoint(0,0)).toPoint()
        self.move(v)
        settings.endGroup()
        
        settings.beginGroup('center')        
        v=settings.value('X',0).toInt()
        if v[1]==True:
            self.rawdisplay.centerX=v[0]
            self.UI.spinCenterX.setValue(v[0])
        else:
            self.UI.spinCenterX.setValue(self.rawdisplay.centerX)
        v=settings.value('Y',0).toInt()
        if v[1]==True:
            self.rawdisplay.centerY=v[0]
            self.UI.spinCenterY.setValue(v[0])
        else:
            self.UI.spinCenterY.setValue(self.rawdisplay.centerY)
        settings.endGroup()
        
        settings.beginGroup('crop')        
        v=settings.value('R',0).toInt()
        if v[1]==True:
            self.rawdisplay.cropR=v[0]
            self.UI.spinRcrop.setValue(v[0])
        else:
            self.UI.spinRcrop.setValue(self.rawdisplay.cropR)
        settings.endGroup()
        
        settings.beginGroup('rotate')
        v=settings.value('90deg',False).toBool()
        self.rawdisplay.rot90deg=v
        self.UI.checkRot90.setChecked(v)
        v=settings.value('degree',0.0).toDouble()
        if v[1]==True:
            self.rawdisplay.rotateDegree=v[0]
            self.UI.spinRotateDeg.setValue(v[0])
        else:
            self.UI.spinRotateDeg.setValue(self.rawdisplay.rotateDegree)
        settings.endGroup()

        settings.beginGroup('colormap')
        v=settings.value('cmindex',0).toInt()
        if v[1]==True:
            self.UI.comboColormap.setCurrentIndex(v[0])
            self.rawdisplay.cmname=str(self.UI.comboColormap.currentText().toLower())
        v=settings.value('cmnorm',False).toBool()
        self.rawdisplay.cmnorm=v
        self.UI.checkNormColormap.setChecked(v)
        if v==True:
            self.UI.checkLogColormap.setEnabled(False)
        v=settings.value('cmlog',False).toBool()
        self.rawdisplay.cmlog=v
        self.UI.checkLogColormap.setChecked(v)
        settings.endGroup()
        
        settings.beginGroup('colorscale')
        v=settings.value('csmin',0.0).toDouble()
        if v[1]==True:
            self.rawdisplay.csmin=v[0]
            self.UI.spinCSmin.setValue(v[0])
        else:
            self.UI.spinCSmin.setValue(self.rawdisplay.csmin)
        v=settings.value('csmax',0.0).toDouble()
        if v[1]==True:
            self.rawdisplay.csmax=v[0]
            self.UI.spinCSmax.setValue(v[0])
        else:
            self.UI.spinCSmax.setValue(self.rawdisplay.csmax)
        settings.endGroup()
        
        settings.beginGroup('grids')
        v=settings.value('centerlines',False).toBool()
        self.rawdisplay.xylines=v
        self.UI.checkCenterXYLines.setChecked(v)
        v=settings.value('xygrid',False).toBool()
        self.rawdisplay.xygrid=v
        self.UI.checkXYgrid.setChecked(v)
        v=settings.value('calibc',0.0).toDouble()
        if v[1]==True:
            self.rawdisplay.calibc=v[0]
            self.UI.spinCalibConst.setValue(v[0])
        else:
            self.UI.spinCalibConst.setValue(self.rawdisplay.calibc)
        v=settings.value('egrid',' ').toString()
        self.UI.lineEGridVals.setText(v)
        v=settings.value('pticks',' ').toString()
        self.UI.linepTickVals.setText(v)
        settings.endGroup()
        
        # Redraw plot to apply changes
        self.rawdisplay.showImage()
        
        # Connect all signals!
        self.UI.lineFileName.returnPressed.connect(self.returnpressLineFname)
        self.UI.pushNext.clicked.connect(self.clickedPushNext)
        self.UI.pushPrevious.clicked.connect(self.clickedPushPrev)
        
        self.UI.groupCenter.toggled.connect(self.checkCenter)
        self.UI.spinCenterX.editingFinished.connect(self.checkCenter)
        self.UI.spinCenterY.editingFinished.connect(self.checkCenter)

        self.UI.groupCrop.toggled.connect(self.checkCrop)
        self.UI.spinRcrop.editingFinished.connect(self.checkCrop)
        
        self.UI.groupRotate.toggled.connect(self.checkRotate)
        self.UI.checkRot90.toggled.connect(self.checkRotate)
        self.UI.spinRotateDeg.editingFinished.connect(self.checkRotate)
        
        self.UI.groupInvert.toggled.connect(self.checkInvert)
        
        self.UI.checkAxisLim.toggled.connect(self.checkAxisLim)
        
        self.UI.comboColormap.activated.connect(self.activatedColormap)
        self.UI.checkNormColormap.toggled.connect(self.checkNormColormap)
        self.UI.checkLogColormap.toggled.connect(self.checkLogColormap)
        
        self.UI.groupColorscale.toggled.connect(self.checkColorscale)
        self.UI.spinCSmin.editingFinished.connect(self.checkColorscale)
        self.UI.spinCSmax.editingFinished.connect(self.checkColorscale)
        
        self.UI.groupGrids.toggled.connect(self.checkGrids)
        self.UI.checkCenterXYLines.toggled.connect(self.checkXYLines)
        self.UI.checkXYgrid.toggled.connect(self.checkXYGrid)
        self.UI.spinCalibConst.editingFinished.connect(self.setCalibc)
        self.UI.lineEGridVals.editingFinished.connect(self.setEGridVals)
        self.UI.linepTickVals.editingFinished.connect(self.setpTickVals)
        

    # --- Centering box has been checked ---
    # on is dummy argument, since we connect 2 different signals to 
    # the same slot: toggled(bool) and editingFinished()
    def checkCenter(self,on=False):
        self.rawdisplay.center = self.UI.groupCenter.isChecked()
        self.rawdisplay.centerX = self.UI.spinCenterX.value()
        self.rawdisplay.centerY = self.UI.spinCenterY.value()
        self.rawdisplay.processRawImage()

    def checkCrop(self,on=False):
        self.rawdisplay.crop = self.UI.groupCrop.isChecked()
        self.rawdisplay.cropR = self.UI.spinRcrop.value()
        self.rawdisplay.resetaxlim = True
        self.rawdisplay.processRawImage()
        
    # --- Rotate check box ---
    # on is dummy argument
    def checkRotate(self,on=False):
        self.rawdisplay.rotate = self.UI.groupRotate.isChecked()
        self.rawdisplay.rot90deg = self.UI.checkRot90.isChecked()
        self.rawdisplay.rotateDegree = self.UI.spinRotateDeg.value()
        self.rawdisplay.processRawImage()
        
    def checkInvert(self,on=False):
        self.rawdisplay.invert = self.UI.groupInvert.isChecked()
        self.rawdisplay.processRawImage()

    def checkAxisLim(self,on):
        self.rawdisplay.holdaxlim=on
    
    @QtCore.pyqtSlot(int)
    def activatedColormap(self,cmindex):
        self.rawdisplay.cmname = str(self.UI.comboColormap.currentText())
        self.rawdisplay.showImage()
    
    def checkNormColormap(self,on):
        self.rawdisplay.cmnorm=on
        # Logscale implies normalizing the data
        if on:
            self.UI.checkLogColormap.setEnabled(False)
        else:
            self.UI.checkLogColormap.setEnabled(True)
        self.rawdisplay.showImage()
    
    def checkLogColormap(self,on):
        self.rawdisplay.cmlog=on
        self.rawdisplay.showImage()
    
    def checkColorscale(self,on=False):
        self.rawdisplay.csmanual = self.UI.groupColorscale.isChecked()
        self.rawdisplay.csmin = self.UI.spinCSmin.value()
        self.rawdisplay.csmax = self.UI.spinCSmax.value()
        self.rawdisplay.showImage()
    
    def checkGrids(self,on):
        self.rawdisplay.grids=on
        if on:
            self.rawdisplay.xylines=self.UI.checkCenterXYLines.isChecked()
            self.rawdisplay.xygrid=self.UI.checkXYgrid.isChecked()
            self.setEGridVals()
            self.setpTickVals()
            # setEGridVals() takes care of replotting the image
        else:
            # Make sure the image is replotted without grids
            self.rawdisplay.showImage()
        
    def checkXYLines(self,on):
        self.rawdisplay.xylines=on
        self.rawdisplay.showImage()
    
    def checkXYGrid(self,on):
        self.rawdisplay.xygrid=on
        self.rawdisplay.showImage()
    
    def setCalibc(self):
        self.rawdisplay.calibc = self.UI.spinCalibConst.value()
        self.rawdisplay.showImage()
    
    # Read out the Egrid values show grid
    def setEGridVals(self):
        # Get text from input line and parse
        Egridstr = str(self.UI.lineEGridVals.text())
        self.rawdisplay.Egridvals=self.parseStringToArray(Egridstr)
        self.rawdisplay.showImage()
    
    # Read out ptick values and show pticks
    def setpTickVals(self):
        # Get text from input line and parse
        ptickstr = str(self.UI.linepTickVals.text())
        self.rawdisplay.ptickvals=self.parseStringToArray(ptickstr)
        self.rawdisplay.showImage()
        
    # Parse a line of input text into a numpy array    
    # Format: 
    # '1.2 2.4 2.3:2:5.3 6:9' -> [1.2,2.4,2.3,4.3,6.,7.,8.]
    def parseStringToArray(self, instr):
        # Split input string by 
        instr = filter(None,instr.split(' '))
        outarr=[]
        # Parse the input string
        for s in instr:
            if ':' in s:
                try:
                    s=np.float_(s.split(':'))
                except:
                    self.showUserMessage('Invalid syntax parsing string to array.')
                    continue
                if s.size == 2:
                    v = np.arange(s[0],s[1],1.)
                    outarr.extend(v)
                elif s.size == 3:
                    v = np.arange(s[0],s[2],s[1])
                    outarr.extend(v)
                else:
                    self.showUserMessage('Invalid syntax parsing string to array.')
            else:
                try:
                    v=np.float_(s)
                except ValueError:
                    self.showUserMessage('Invalid syntax parsing string to array.',1)
                    continue                    
                outarr.append(v)
        return np.array(outarr)
    
    # --- Open directly from lineedit without showing the dialog ---
    def returnpressLineFname(self):
        lineFname = self.UI.lineFileName.text()
        # Check if a file is listed in the line-edit
        if lineFname == '':
            self.openFile()
        else:
            # Check if the line-edit file exists
            if QtCore.QFile.exists(lineFname) == True:
                if lineFname != self.fname:
                    self.fname = lineFname
                    # Get the file directory and set it as working directory
                    self.fdir = QtCore.QFileInfo(self.fname).absoluteDir()
                    QtCore.QDir.setCurrent(self.fdir.path())
                    self.UI.lineFileName.setText(self.fname)
                    # Try opening the image file
                    self.rawdisplay.openFile(self.fname)
            else:
                self.showUserMessage('Given file does not exist.',2)
            
    # --- Next (file) button in main window ---
    def clickedPushNext(self):        
        # Get list of files in current directory
        flist = self.fdir.entryInfoList(QtCore.QDir.Files, QtCore.QDir.Type)
        # Check the file list and get the .raw and .npy files
        frawlist=[]
        for f in flist:
            if f.suffix() == 'raw' or f.suffix() == 'npy':
                frawlist.append(f.absoluteFilePath())
        if frawlist == []:
            self.showUserMessage('No raw or npy files in current directory.',2)
            return
        
        # Search for current file in the file list
        for i,f in enumerate(frawlist):
            if f == self.fdir.absoluteFilePath(self.fname):
                # Open the next file in list and return
                if i < len(frawlist)-1:
                    self.UI.lineFileName.setText(frawlist[i+1])
                else:
                    self.UI.lineFileName.setText(frawlist[0])
                # Open the file and exit from function
                self.returnpressLineFname()
                return
        # Did not find current file in list -> open first file in list
        self.UI.lineFileName.setText(frawlist[0])
        self.returnpressLineFname()

    # --- Previous (file) button in main window ---
    def clickedPushPrev(self):
        # Get list of files in current directory
        flist = self.fdir.entryInfoList(QtCore.QDir.Files, QtCore.QDir.Type)
        # Check the file list and get the .raw and .npy files
        frawlist=[]
        for f in flist:
            if f.suffix() == 'raw' or f.suffix() == 'npy':
                frawlist.append(f.absoluteFilePath())
        if frawlist == []:
            self.showUserMessage('No raw or npy files in current directory.',2)
            return
        
        # Search for current file in the file list
        for i,f in enumerate(frawlist):
            if f == self.fdir.absoluteFilePath(self.fname):
                # Open the next file in list and return
                if i > 0:
                    self.UI.lineFileName.setText(frawlist[i-1])
                else:
                    self.UI.lineFileName.setText(frawlist[len(frawlist)-1])
                # Open the file and exit from function
                self.returnpressLineFname()
                return
        # Did not find current file in list -> open first file in list
        self.UI.lineFileName.setText(frawlist[0])
        self.returnpressLineFname()
    

    # --- Show open file dialog and open specified file ---
    def openFile(self):
        file_choices = "Raw images (*.raw *.npy)"
        self.fname = unicode(QtGui.QFileDialog.getOpenFileName(self, 
                            'Open file', self.fdir.path(), file_choices))
        # Get the file directory and set it as working directory
        self.fdir = QtCore.QFileInfo(self.fname).absoluteDir()
        QtCore.QDir.setCurrent(self.fdir.path())
        # Print in filename line
        self.UI.lineFileName.setText(self.fname)
        # Try opening the image file
        self.rawdisplay.openFile(self.fname)
            
    # --- Receive user messages from child widgts and show in status bar ---
    @QtCore.pyqtSlot(str,int)
    def showUserMessage(self,message,level=0):
        if (level == 1):
            self.status_text.setText('WARNING: '+ message)
        elif (level == 2):
            self.status_text.setText('ERROR: ' + message)
        else:
            self.status_text.setText('INFO: ' + message)
    
    # --- Put any un-initialization stuff in here ---
    def Quit(self):
        # --- DEBUG --- 
#        QtCore.pyqtRemoveInputHook()
#        pdb.set_trace()

        # Store application window settings before exiting
        settings=QtCore.QSettings('MBI','RawViewer')

        settings.beginGroup('rawdispwindow')
        settings.setValue('size',self.rawdisplay.size())
        settings.setValue('pos',self.rawdisplay.pos())
        settings.endGroup()
        
        self.rawdisplay.close()
        
        settings.beginGroup('mainwindow')
        settings.setValue('size',self.size())
        settings.setValue('pos',self.pos())
        settings.endGroup()
                
        settings.beginGroup('center')
        settings.setValue('X',self.UI.spinCenterX.value())
        settings.setValue('Y',self.UI.spinCenterY.value())
        settings.endGroup()
        
        settings.beginGroup('crop')
        settings.setValue('R',self.UI.spinRcrop.value())
        settings.endGroup()
        
        settings.beginGroup('rotate')
        settings.setValue('90deg',self.UI.checkRot90.isChecked())
        settings.setValue('degree',self.UI.spinRotateDeg.value())
        settings.endGroup()
        
        settings.beginGroup('colormap')
        settings.setValue('cmindex',self.UI.comboColormap.currentIndex())
        settings.setValue('cmnorm',self.UI.checkNormColormap.isChecked())
        settings.setValue('cmlog',self.UI.checkLogColormap.isChecked())
        settings.endGroup()
        
        settings.beginGroup('colorscale')
        settings.setValue('csmin',self.UI.spinCSmin.value())
        settings.setValue('csmax',self.UI.spinCSmax.value())
        settings.endGroup()
        
        settings.beginGroup('grids')
        settings.setValue('centerlines',self.UI.checkCenterXYLines.isChecked())
        settings.setValue('xygrid',self.UI.checkXYgrid.isChecked())
        settings.setValue('calibc',self.UI.spinCalibConst.value())
        settings.setValue('egrid',self.UI.lineEGridVals.text())
        settings.setValue('pticks',self.UI.linepTickVals.text())
        settings.endGroup()
        
        self.close()



if __name__ == "__main__":
    App = QtGui.QApplication(sys.argv)
    Window = MainUI()
    App.aboutToQuit.connect(Window.Quit)
    Window.show()
    App.exec_()
