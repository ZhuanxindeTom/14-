# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'yolov5.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1100, 700)
        MainWindow.setMinimumSize(QtCore.QSize(1100, 700))
        MainWindow.setMaximumSize(QtCore.QSize(1100, 700))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setSpacing(0)
        self.gridLayout.setObjectName("gridLayout")
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setStyleSheet("#frame{\n"
"background-image: url(:/win/background.jpg);\n"
"}\n"
"")
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.label_1 = QtWidgets.QLabel(self.frame)
        self.label_1.setGeometry(QtCore.QRect(20, 70, 731, 441))
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(192, 192, 192))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(192, 192, 192))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(192, 192, 192))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(192, 192, 192))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(192, 192, 192))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(192, 192, 192))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(192, 192, 192))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(192, 192, 192))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(192, 192, 192))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Window, brush)
        self.label_1.setPalette(palette)
        self.label_1.setStyleSheet("background-color: rgb(192, 192, 192);")
        self.label_1.setText("")
        self.label_1.setAlignment(QtCore.Qt.AlignCenter)
        self.label_1.setObjectName("label_1")
        self.PB1 = QtWidgets.QPushButton(self.frame)
        self.PB1.setGeometry(QtCore.QRect(800, 70, 271, 60))
        font = QtGui.QFont()
        font.setFamily("楷体")
        font.setPointSize(13)
        self.PB1.setFont(font)
        self.PB1.setStyleSheet("background-color: rgb(192, 192, 192);")
        self.PB1.setObjectName("PB1")
        self.PB2 = QtWidgets.QPushButton(self.frame)
        self.PB2.setGeometry(QtCore.QRect(800, 190, 271, 60))
        font = QtGui.QFont()
        font.setFamily("楷体")
        font.setPointSize(13)
        self.PB2.setFont(font)
        self.PB2.setStyleSheet("background-color: rgb(192, 192, 192);")
        self.PB2.setObjectName("PB2")
        self.PB3 = QtWidgets.QPushButton(self.frame)
        self.PB3.setGeometry(QtCore.QRect(800, 310, 271, 60))
        font = QtGui.QFont()
        font.setFamily("楷体")
        font.setPointSize(13)
        self.PB3.setFont(font)
        self.PB3.setStyleSheet("background-color: rgb(192, 192, 192);")
        self.PB3.setObjectName("PB3")
        self.PB4 = QtWidgets.QPushButton(self.frame)
        self.PB4.setGeometry(QtCore.QRect(800, 430, 271, 60))
        font = QtGui.QFont()
        font.setFamily("楷体")
        font.setPointSize(13)
        self.PB4.setFont(font)
        self.PB4.setStyleSheet("background-color: rgb(192, 192, 192);")
        self.PB4.setObjectName("PB4")
        self.label_4 = QtWidgets.QLabel(self.frame)
        self.label_4.setGeometry(QtCore.QRect(270, 10, 561, 51))
        font = QtGui.QFont()
        font.setFamily("楷体")
        font.setPointSize(17)
        self.label_4.setFont(font)
        self.label_4.setStyleSheet("color:rgb(255, 255, 255)")
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_4.setObjectName("label_4")
        self.textBrowser = QtWidgets.QTextBrowser(self.frame)
        self.textBrowser.setGeometry(QtCore.QRect(20, 520, 1051, 171))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.textBrowser.setFont(font)
        self.textBrowser.setStyleSheet("background-color: rgb(192, 192, 192);")
        self.textBrowser.setObjectName("textBrowser")
        self.gridLayout.addWidget(self.frame, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.PB1.setText(_translate("MainWindow", "打开图片/视频"))
        self.PB2.setText(_translate("MainWindow", "打开/关闭摄像头"))
        self.PB3.setText(_translate("MainWindow", "开始检测"))
        self.PB4.setText(_translate("MainWindow", "结束检测"))
        self.label_4.setText(_translate("MainWindow", "基于YOLOv5算法的车牌检测系统"))
import win.qt
