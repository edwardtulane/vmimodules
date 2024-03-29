# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'MainWindow.ui'
#
# Created by: PyQt5 UI code generator 5.5.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(390, 859)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.fileParameters = QtWidgets.QGroupBox(self.centralwidget)
        self.fileParameters.setEnabled(True)
        self.fileParameters.setFlat(False)
        self.fileParameters.setCheckable(False)
        self.fileParameters.setChecked(False)
        self.fileParameters.setObjectName("fileParameters")
        self.gridLayout = QtWidgets.QGridLayout(self.fileParameters)
        self.gridLayout.setObjectName("gridLayout")
        self.lineFileName = QtWidgets.QLineEdit(self.fileParameters)
        self.lineFileName.setObjectName("lineFileName")
        self.gridLayout.addWidget(self.lineFileName, 0, 0, 1, 2)
        self.pushNext = QtWidgets.QPushButton(self.fileParameters)
        self.pushNext.setObjectName("pushNext")
        self.gridLayout.addWidget(self.pushNext, 1, 0, 1, 1)
        self.pushPrevious = QtWidgets.QPushButton(self.fileParameters)
        self.pushPrevious.setObjectName("pushPrevious")
        self.gridLayout.addWidget(self.pushPrevious, 1, 1, 1, 1)
        self.verticalLayout.addWidget(self.fileParameters)
        self.EditingParamters = QtWidgets.QGroupBox(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.EditingParamters.sizePolicy().hasHeightForWidth())
        self.EditingParamters.setSizePolicy(sizePolicy)
        self.EditingParamters.setFlat(False)
        self.EditingParamters.setCheckable(False)
        self.EditingParamters.setObjectName("EditingParamters")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.EditingParamters)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.groupCenter = QtWidgets.QGroupBox(self.EditingParamters)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupCenter.sizePolicy().hasHeightForWidth())
        self.groupCenter.setSizePolicy(sizePolicy)
        self.groupCenter.setCheckable(True)
        self.groupCenter.setChecked(False)
        self.groupCenter.setObjectName("groupCenter")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.groupCenter)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.label = QtWidgets.QLabel(self.groupCenter)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        self.label.setObjectName("label")
        self.horizontalLayout_3.addWidget(self.label)
        self.spinCenterX = QtWidgets.QSpinBox(self.groupCenter)
        self.spinCenterX.setMaximum(9999)
        self.spinCenterX.setObjectName("spinCenterX")
        self.horizontalLayout_3.addWidget(self.spinCenterX)
        self.label_2 = QtWidgets.QLabel(self.groupCenter)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_2.sizePolicy().hasHeightForWidth())
        self.label_2.setSizePolicy(sizePolicy)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_3.addWidget(self.label_2)
        self.spinCenterY = QtWidgets.QSpinBox(self.groupCenter)
        self.spinCenterY.setMaximum(9999)
        self.spinCenterY.setObjectName("spinCenterY")
        self.horizontalLayout_3.addWidget(self.spinCenterY)
        self.verticalLayout_3.addWidget(self.groupCenter)
        self.groupCrop = QtWidgets.QGroupBox(self.EditingParamters)
        self.groupCrop.setToolTip("Crop image")
        self.groupCrop.setCheckable(True)
        self.groupCrop.setChecked(False)
        self.groupCrop.setObjectName("groupCrop")
        self.formLayout_4 = QtWidgets.QFormLayout(self.groupCrop)
        self.formLayout_4.setObjectName("formLayout_4")
        self.label_6 = QtWidgets.QLabel(self.groupCrop)
        self.label_6.setObjectName("label_6")
        self.formLayout_4.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_6)
        self.spinRcrop = QtWidgets.QSpinBox(self.groupCrop)
        self.spinRcrop.setMaximum(9999)
        self.spinRcrop.setObjectName("spinRcrop")
        self.formLayout_4.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.spinRcrop)
        self.verticalLayout_3.addWidget(self.groupCrop)
        self.groupRotate = QtWidgets.QGroupBox(self.EditingParamters)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupRotate.sizePolicy().hasHeightForWidth())
        self.groupRotate.setSizePolicy(sizePolicy)
        self.groupRotate.setCheckable(True)
        self.groupRotate.setChecked(False)
        self.groupRotate.setObjectName("groupRotate")
        self.formLayout_2 = QtWidgets.QFormLayout(self.groupRotate)
        self.formLayout_2.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        self.formLayout_2.setObjectName("formLayout_2")
        self.checkRot90 = QtWidgets.QCheckBox(self.groupRotate)
        self.checkRot90.setObjectName("checkRot90")
        self.formLayout_2.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.checkRot90)
        self.label_3 = QtWidgets.QLabel(self.groupRotate)
        self.label_3.setObjectName("label_3")
        self.formLayout_2.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_3)
        self.spinRotateDeg = QtWidgets.QDoubleSpinBox(self.groupRotate)
        self.spinRotateDeg.setDecimals(1)
        self.spinRotateDeg.setMinimum(-360.0)
        self.spinRotateDeg.setMaximum(360.0)
        self.spinRotateDeg.setObjectName("spinRotateDeg")
        self.formLayout_2.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.spinRotateDeg)
        self.verticalLayout_3.addWidget(self.groupRotate)
        self.groupInvert = QtWidgets.QGroupBox(self.EditingParamters)
        self.groupInvert.setCheckable(True)
        self.groupInvert.setChecked(False)
        self.groupInvert.setObjectName("groupInvert")
        self.verticalLayout_3.addWidget(self.groupInvert)
        self.verticalLayout.addWidget(self.EditingParamters)
        self.ViewingParameters = QtWidgets.QGroupBox(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.ViewingParameters.sizePolicy().hasHeightForWidth())
        self.ViewingParameters.setSizePolicy(sizePolicy)
        self.ViewingParameters.setObjectName("ViewingParameters")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.ViewingParameters)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.checkAxisLim = QtWidgets.QCheckBox(self.ViewingParameters)
        self.checkAxisLim.setObjectName("checkAxisLim")
        self.verticalLayout_2.addWidget(self.checkAxisLim)
        self.groupColormap = QtWidgets.QGroupBox(self.ViewingParameters)
        self.groupColormap.setObjectName("groupColormap")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.groupColormap)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.comboColormap = QtWidgets.QComboBox(self.groupColormap)
        self.comboColormap.setObjectName("comboColormap")
        self.comboColormap.addItem("")
        self.comboColormap.addItem("")
        self.comboColormap.addItem("")
        self.comboColormap.addItem("")
        self.comboColormap.addItem("")
        self.comboColormap.addItem("")
        self.comboColormap.addItem("")
        self.comboColormap.addItem("")
        self.comboColormap.addItem("")
        self.comboColormap.addItem("")
        self.comboColormap.addItem("")
        self.comboColormap.addItem("")
        self.comboColormap.addItem("")
        self.comboColormap.addItem("")
        self.comboColormap.addItem("")
        self.comboColormap.addItem("")
        self.horizontalLayout.addWidget(self.comboColormap)
        self.checkNormColormap = QtWidgets.QCheckBox(self.groupColormap)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.checkNormColormap.sizePolicy().hasHeightForWidth())
        self.checkNormColormap.setSizePolicy(sizePolicy)
        self.checkNormColormap.setObjectName("checkNormColormap")
        self.horizontalLayout.addWidget(self.checkNormColormap)
        self.checkLogColormap = QtWidgets.QCheckBox(self.groupColormap)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.checkLogColormap.sizePolicy().hasHeightForWidth())
        self.checkLogColormap.setSizePolicy(sizePolicy)
        self.checkLogColormap.setObjectName("checkLogColormap")
        self.horizontalLayout.addWidget(self.checkLogColormap)
        self.verticalLayout_2.addWidget(self.groupColormap)
        self.groupColorscale = QtWidgets.QGroupBox(self.ViewingParameters)
        self.groupColorscale.setCheckable(True)
        self.groupColorscale.setChecked(False)
        self.groupColorscale.setObjectName("groupColorscale")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.groupColorscale)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_7 = QtWidgets.QLabel(self.groupColorscale)
        self.label_7.setObjectName("label_7")
        self.horizontalLayout_2.addWidget(self.label_7)
        self.spinCSmin = QtWidgets.QDoubleSpinBox(self.groupColorscale)
        self.spinCSmin.setDecimals(3)
        self.spinCSmin.setMinimum(-999999.0)
        self.spinCSmin.setMaximum(999999.0)
        self.spinCSmin.setObjectName("spinCSmin")
        self.horizontalLayout_2.addWidget(self.spinCSmin)
        self.label_8 = QtWidgets.QLabel(self.groupColorscale)
        self.label_8.setObjectName("label_8")
        self.horizontalLayout_2.addWidget(self.label_8)
        self.spinCSmax = QtWidgets.QDoubleSpinBox(self.groupColorscale)
        self.spinCSmax.setDecimals(3)
        self.spinCSmax.setMinimum(-999999.0)
        self.spinCSmax.setMaximum(999999.0)
        self.spinCSmax.setObjectName("spinCSmax")
        self.horizontalLayout_2.addWidget(self.spinCSmax)
        self.verticalLayout_2.addWidget(self.groupColorscale)
        self.groupGrids = QtWidgets.QGroupBox(self.ViewingParameters)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupGrids.sizePolicy().hasHeightForWidth())
        self.groupGrids.setSizePolicy(sizePolicy)
        self.groupGrids.setCheckable(True)
        self.groupGrids.setChecked(False)
        self.groupGrids.setObjectName("groupGrids")
        self.formLayout_3 = QtWidgets.QFormLayout(self.groupGrids)
        self.formLayout_3.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        self.formLayout_3.setObjectName("formLayout_3")
        self.checkCenterXYLines = QtWidgets.QCheckBox(self.groupGrids)
        self.checkCenterXYLines.setObjectName("checkCenterXYLines")
        self.formLayout_3.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.checkCenterXYLines)
        self.label_4 = QtWidgets.QLabel(self.groupGrids)
        self.label_4.setObjectName("label_4")
        self.formLayout_3.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_4)
        self.spinCalibConst = QtWidgets.QDoubleSpinBox(self.groupGrids)
        self.spinCalibConst.setMaximum(99999.0)
        self.spinCalibConst.setProperty("value", 1000.0)
        self.spinCalibConst.setObjectName("spinCalibConst")
        self.formLayout_3.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.spinCalibConst)
        self.label_5 = QtWidgets.QLabel(self.groupGrids)
        self.label_5.setObjectName("label_5")
        self.formLayout_3.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.label_5)
        self.lineEGridVals = QtWidgets.QLineEdit(self.groupGrids)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lineEGridVals.sizePolicy().hasHeightForWidth())
        self.lineEGridVals.setSizePolicy(sizePolicy)
        self.lineEGridVals.setObjectName("lineEGridVals")
        self.formLayout_3.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.lineEGridVals)
        self.label_9 = QtWidgets.QLabel(self.groupGrids)
        self.label_9.setObjectName("label_9")
        self.formLayout_3.setWidget(4, QtWidgets.QFormLayout.LabelRole, self.label_9)
        self.linepTickVals = QtWidgets.QLineEdit(self.groupGrids)
        self.linepTickVals.setObjectName("linepTickVals")
        self.formLayout_3.setWidget(4, QtWidgets.QFormLayout.FieldRole, self.linepTickVals)
        self.checkXYgrid = QtWidgets.QCheckBox(self.groupGrids)
        self.checkXYgrid.setObjectName("checkXYgrid")
        self.formLayout_3.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.checkXYgrid)
        self.verticalLayout_2.addWidget(self.groupGrids)
        self.verticalLayout.addWidget(self.ViewingParameters)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 390, 20))
        self.menubar.setObjectName("menubar")
        self.menuF_ile = QtWidgets.QMenu(self.menubar)
        self.menuF_ile.setObjectName("menuF_ile")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.action_Open = QtWidgets.QAction(MainWindow)
        self.action_Open.setText("&Open")
        self.action_Open.setShortcut("Ctrl+O")
        self.action_Open.setObjectName("action_Open")
        self.action_Close = QtWidgets.QAction(MainWindow)
        self.action_Close.setObjectName("action_Close")
        self.action_Center = QtWidgets.QAction(MainWindow)
        self.action_Center.setCheckable(True)
        self.action_Center.setObjectName("action_Center")
        self.action_Rotate = QtWidgets.QAction(MainWindow)
        self.action_Rotate.setCheckable(True)
        self.action_Rotate.setObjectName("action_Rotate")
        self.action_Save = QtWidgets.QAction(MainWindow)
        self.action_Save.setObjectName("action_Save")
        self.action_Close_2 = QtWidgets.QAction(MainWindow)
        self.action_Close_2.setObjectName("action_Close_2")
        self.action_Quit = QtWidgets.QAction(MainWindow)
        self.action_Quit.setObjectName("action_Quit")
        self.action_Grid = QtWidgets.QAction(MainWindow)
        self.action_Grid.setObjectName("action_Grid")
        self.actionOpen_Next = QtWidgets.QAction(MainWindow)
        self.actionOpen_Next.setText("Open &Next")
        self.actionOpen_Next.setShortcut("Ctrl+N")
        self.actionOpen_Next.setObjectName("actionOpen_Next")
        self.actionOpen_Previous = QtWidgets.QAction(MainWindow)
        self.actionOpen_Previous.setText("Open &Previous")
        self.actionOpen_Previous.setShortcut("Ctrl+P")
        self.actionOpen_Previous.setObjectName("actionOpen_Previous")
        self.actionCro_p = QtWidgets.QAction(MainWindow)
        self.actionCro_p.setObjectName("actionCro_p")
        self.menuF_ile.addAction(self.action_Open)
        self.menuF_ile.addAction(self.actionOpen_Next)
        self.menuF_ile.addAction(self.actionOpen_Previous)
        self.menuF_ile.addAction(self.action_Save)
        self.menuF_ile.addSeparator()
        self.menuF_ile.addAction(self.action_Quit)
        self.menubar.addAction(self.menuF_ile.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        MainWindow.setTabOrder(self.lineFileName, self.pushNext)
        MainWindow.setTabOrder(self.pushNext, self.pushPrevious)
        MainWindow.setTabOrder(self.pushPrevious, self.groupCenter)
        MainWindow.setTabOrder(self.groupCenter, self.spinCenterX)
        MainWindow.setTabOrder(self.spinCenterX, self.spinCenterY)
        MainWindow.setTabOrder(self.spinCenterY, self.groupCrop)
        MainWindow.setTabOrder(self.groupCrop, self.spinRcrop)
        MainWindow.setTabOrder(self.spinRcrop, self.groupRotate)
        MainWindow.setTabOrder(self.groupRotate, self.checkRot90)
        MainWindow.setTabOrder(self.checkRot90, self.spinRotateDeg)
        MainWindow.setTabOrder(self.spinRotateDeg, self.checkAxisLim)
        MainWindow.setTabOrder(self.checkAxisLim, self.comboColormap)
        MainWindow.setTabOrder(self.comboColormap, self.checkNormColormap)
        MainWindow.setTabOrder(self.checkNormColormap, self.checkLogColormap)
        MainWindow.setTabOrder(self.checkLogColormap, self.groupColorscale)
        MainWindow.setTabOrder(self.groupColorscale, self.spinCSmin)
        MainWindow.setTabOrder(self.spinCSmin, self.spinCSmax)
        MainWindow.setTabOrder(self.spinCSmax, self.groupGrids)
        MainWindow.setTabOrder(self.groupGrids, self.checkCenterXYLines)
        MainWindow.setTabOrder(self.checkCenterXYLines, self.checkXYgrid)
        MainWindow.setTabOrder(self.checkXYgrid, self.spinCalibConst)
        MainWindow.setTabOrder(self.spinCalibConst, self.lineEGridVals)
        MainWindow.setTabOrder(self.lineEGridVals, self.linepTickVals)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "RawViewer"))
        self.fileParameters.setTitle(_translate("MainWindow", "File"))
        self.pushNext.setText(_translate("MainWindow", "&Next"))
        self.pushPrevious.setText(_translate("MainWindow", "&Previous"))
        self.EditingParamters.setTitle(_translate("MainWindow", "Edit"))
        self.groupCenter.setTitle(_translate("MainWindow", "&Center"))
        self.label.setText(_translate("MainWindow", "X="))
        self.label_2.setText(_translate("MainWindow", "Y="))
        self.groupCrop.setTitle(_translate("MainWindow", "Cro&p"))
        self.label_6.setText(_translate("MainWindow", "R="))
        self.groupRotate.setTitle(_translate("MainWindow", "Rotate"))
        self.checkRot90.setText(_translate("MainWindow", "90 deg"))
        self.label_3.setText(_translate("MainWindow", "Degree"))
        self.groupInvert.setTitle(_translate("MainWindow", "In&vert"))
        self.ViewingParameters.setTitle(_translate("MainWindow", "Plot"))
        self.checkAxisLim.setText(_translate("MainWindow", "Hold view"))
        self.groupColormap.setTitle(_translate("MainWindow", "Color map"))
        self.comboColormap.setItemText(0, _translate("MainWindow", "jet"))
        self.comboColormap.setItemText(1, _translate("MainWindow", "seismic"))
        self.comboColormap.setItemText(2, _translate("MainWindow", "coolwarm"))
        self.comboColormap.setItemText(3, _translate("MainWindow", "RdBu"))
        self.comboColormap.setItemText(4, _translate("MainWindow", "RdYlBu"))
        self.comboColormap.setItemText(5, _translate("MainWindow", "bone"))
        self.comboColormap.setItemText(6, _translate("MainWindow", "gray"))
        self.comboColormap.setItemText(7, _translate("MainWindow", "hot"))
        self.comboColormap.setItemText(8, _translate("MainWindow", "copper"))
        self.comboColormap.setItemText(9, _translate("MainWindow", "pink"))
        self.comboColormap.setItemText(10, _translate("MainWindow", "hsv"))
        self.comboColormap.setItemText(11, _translate("MainWindow", "cool"))
        self.comboColormap.setItemText(12, _translate("MainWindow", "spring"))
        self.comboColormap.setItemText(13, _translate("MainWindow", "summer"))
        self.comboColormap.setItemText(14, _translate("MainWindow", "autumn"))
        self.comboColormap.setItemText(15, _translate("MainWindow", "winter"))
        self.checkNormColormap.setText(_translate("MainWindow", "Norm"))
        self.checkLogColormap.setText(_translate("MainWindow", "Log"))
        self.groupColorscale.setTitle(_translate("MainWindow", "Co&lor scale"))
        self.label_7.setText(_translate("MainWindow", "min="))
        self.label_8.setText(_translate("MainWindow", "max="))
        self.groupGrids.setTitle(_translate("MainWindow", "Grids"))
        self.checkCenterXYLines.setText(_translate("MainWindow", "Center lines"))
        self.label_4.setText(_translate("MainWindow", "Calib. const."))
        self.label_5.setText(_translate("MainWindow", "Egrid="))
        self.label_9.setText(_translate("MainWindow", "pticks="))
        self.checkXYgrid.setText(_translate("MainWindow", "X/Y grid"))
        self.menuF_ile.setTitle(_translate("MainWindow", "&File"))
        self.action_Close.setText(_translate("MainWindow", "&Close"))
        self.action_Center.setText(_translate("MainWindow", "&Center"))
        self.action_Rotate.setText(_translate("MainWindow", "&Rotate"))
        self.action_Save.setText(_translate("MainWindow", "&Save"))
        self.action_Close_2.setText(_translate("MainWindow", "&Close"))
        self.action_Quit.setText(_translate("MainWindow", "&Quit"))
        self.action_Quit.setShortcut(_translate("MainWindow", "Ctrl+Q"))
        self.action_Grid.setText(_translate("MainWindow", "&Grid"))
        self.actionCro_p.setText(_translate("MainWindow", "Cro&p"))

