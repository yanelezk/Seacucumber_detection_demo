# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mainwindow.ui'
#
# Created by: PyQt5 UI code generator 5.12.2
#
# WARNING! All changes made in this file will be lost!
from PyQt5 import QtCore, QtGui, QtWidgets, QtNetwork
import datetime
from PyQt5.QtCore import pyqtSlot,pyqtSignal,QByteArray,QDataStream,qUncompress,QTimer
from PyQt5.QtGui import QPixmap,QImage
from PyQt5.QtWidgets import QApplication,QMessageBox
import sys
from df import Ui_Form
from detect import deteTrepang
from pyserial_demo import Serial_Port
import math
import zlib
import numpy as np
import cv2
import qimage2ndarray
import time

class TcpServer(Serial_Port):

    def __init__(self,parent=None):
        super(TcpServer, self).__init__()
        #监听的端口
        self.tcpSocketServerPort = 55555
        #图像接受相关
        self.OneFrameAllDataSize = 0
        self.OneImageSize = 0
        self.dtype = 'array'
        #连接成功标准位
        self.connectFlag = False
        self.disconnectFlag = True
        self.fps_count = 0
        self.socket = QtNetwork.QTcpSocket()
        self.server = QtNetwork.QTcpServer()
        #初始化开始检测的按钮
        self.stopTcpconnect.clicked.connect(self.stopTCP)
        #在ＴＣＰ连接成功之前将停止按钮设为不可处罚
        self.stopTcpconnect.setEnabled(False)
        self.startTcpconnect.clicked.connect(self.startTCP)
        self.startTcpconnect.setEnabled(False)
        #检测本地视频
        # 初始化定时器
        self.timer_1s = QTimer(self)
        self.timer_1s.timeout.connect(self.FlagEvent)
        # 打开串口接收定时器，周期为ms
        self.timer_1s.start(100)
        # 初始化服务器端
        self.initServer()

    # 定时１s检查标志位回调函数
    def FlagEvent(self):
        # 检测方式
        self.deteWay = str(self.s1__box_9.currentText())
        # print(self.deteWay)
        if self.deteWay == str("检测本地图像"):
            if self.flag_sureViewVideo == True:
                self.cap.release()
                self.timer_video.stop()
                self.flag_sureViewVideo = False
            self.flag_detImage = True
            self.flag_timerDet = False
            self.flag_detVIdeo = False
            self.flag_showIamge = False
            # 本地图像检测按钮
            self.open_button_2.setEnabled(True)
            # 本地视频检测按钮
            self.close_button_2.setEnabled(False)
            # 预览按钮
            self.whatchImagrbutton.setEnabled(True)
            self.whatchVideogrbutton.setEnabled(False)
            # 设置ＴＣＰ连接按钮不可触发
            self.stopTcpconnect.setEnabled(False)
            self.startTcpconnect.setEnabled(False)
            # 先延时５毫秒
            if self.server.isListening() == True:
                print("正在断开tcp连接")
                time.sleep(5)
                # 关闭服务器
                self.server.close()
                self.socket.close()
                # 清空画面
                self.label_3.clear()
                self.lineEdit_4.setText(str("网络连接断开"))

        elif self.deteWay == str("检测本地视频"):
            self.flag_detVIdeo = True
            self.flag_detImage = False
            self.flag_timerDet = False
            self.flag_showIamge = False
            self.open_button_2.setEnabled(False)
            self.close_button_2.setEnabled(True)
            # 预览按钮
            self.whatchImagrbutton.setEnabled(False)
            self.whatchVideogrbutton.setEnabled(True)
            # 设置ＴＣＰ连接按钮不可触发
            self.stopTcpconnect.setEnabled(False)
            self.startTcpconnect.setEnabled(False)

            if self.server.isListening() == True:
                print("正在断开tcp连接")
                # 先延时５毫秒
                time.sleep(5)
                # 关闭服务器
                self.server.close()
                self.socket.close()
                # 清空画面
                self.label_3.clear()
                self.lineEdit_4.setText(str("网络连接断开"))

        elif self.deteWay == str("实时检测"):
            if self.flag_sureViewVideo == True:
                self.cap.release()
                self.timer_video.stop()
                self.flag_sureViewVideo = False
            self.flag_timerDet = True
            self.flag_detImage = False
            self.flag_detVIdeo = False
            self.flag_showIamge = False
            self.open_button_2.setEnabled(False)
            self.close_button_2.setEnabled(False)
            # 预览按钮
            self.whatchImagrbutton.setEnabled(False)
            self.whatchVideogrbutton.setEnabled(False)
            # 设置ＴＣＰ连接按钮不可触发
            if self.server.isListening() == False:
                self.stopTcpconnect.setEnabled(False)
                self.startTcpconnect.setEnabled(True)

        elif self.deteWay == str("画面传输不检测"):
            if self.flag_sureViewVideo == True:
                self.cap.release()
                self.timer_video.stop()
                self.flag_sureViewVideo = False
            self.flag_showIamge = True
            self.flag_detImage = False
            self.flag_timerDet = False
            self.flag_detVIdeo = False
            self.open_button_2.setEnabled(False)
            self.close_button_2.setEnabled(False)
            # 预览按钮
            self.whatchImagrbutton.setEnabled(False)
            self.whatchVideogrbutton.setEnabled(False)
            if self.server.isListening() == False:
                self.stopTcpconnect.setEnabled(False)
                self.startTcpconnect.setEnabled(True)
            self.lineEdit_3.setText(str(0))

    @pyqtSlot()
    def initServer(self):
        self.socket = QtNetwork.QTcpSocket()
        self.server = QtNetwork.QTcpServer()
        if self.server != None:
            if self.server.listen(QtNetwork.QHostAddress.Any,self.tcpSocketServerPort) == True:
                print("已监听到端口")
                self.server.newConnection.connect(self.TcpClientConnected)

    def TcpClientConnected(self):
        print("tcp连接成功")
        self.lineEdit_4.setText(str("网络连接成功"))
        self.stopTcpconnect.setEnabled(True)
        self.startTcpconnect.setEnabled(False)
        self.connectFlag = True
        self.disconnectFlag = False
        self.OneFrameAllDataSize = 0
        self.OneImageSize = 0
        self.socket = self.server.nextPendingConnection()
        self.socket.readyRead.connect(self.TcpClientReadyToRead)
        self.socket.disconnected.connect(self.TcpClientDisconnected)

    def TcpClientDisconnected(self):
        print("tcp断开")
        self.lineEdit_4.setText(str("网络连接断开"))

    def TcpClientReadyToRead(self):

        recv = QDataStream(self.socket)
        recv.setVersion(QDataStream.Qt_5_3)
        self.ImageMessage = QByteArray().clear()
        if self.OneFrameAllDataSize == 0 and self.OneImageSize ==0 :
            if self.socket.bytesAvailable() < 24:
                #清空缓冲区
                self.socket.readAll()
                return

            self.OneFrameAllDataSize = recv.readUInt64()
            self.OneImageSize = recv.readUInt64()
            #print(self.OneImageSize, self.OneFrameAllDataSize)

        if self.OneFrameAllDataSize > 500000 or self.OneFrameAllDataSize < 10000:

            self.OneFrameAllDataSize = 0
            self.OneImageSize = 0
            self.ImageMessage = QByteArray().clear()
            #清空缓存
            self.socket.readAll()
            return

        if self.socket.bytesAvailable() < self.OneFrameAllDataSize:
            return
        #print("读图像数据和音频数据")
        #清空之后再接收

        self.ImageMessage = QByteArray().clear()
        self.ImageMessage = recv.readBytes()

        self.fps_count = self.fps_count + 1
        if self.fps_count == 15:
            self.socket.readAll()
            self.fps_count = 0
            self.OneFrameAllDataSize = 0
            self.OneImageSize = 0
            return

        self.showImage()
        self.OneFrameAllDataSize = 0
        self.OneImageSize = 0
        #发送数据回去
        if self.flag_router == True and self.ser.isOpen():
            self.socket.write(self.routerData)
        elif self.flag_key == True:
            self.socket.write(self.controlData)
        else:
            print("不发送数据")

    def showImage(self):
        #显示图像
        self.img_bytearray = QByteArray()
        self.imgdata = QByteArray()
        self.img_bytearray = QByteArray.fromBase64(self.ImageMessage)
        self.imgdata = qUncompress(self.img_bytearray)
        self.image = QImage()
        self.image.loadFromData(self.imgdata)
        #实时检测
        if self.flag_timerDet == True and self.flag_showIamge == False:
            self.imageArray = qimage2ndarray.rgb_view(self.image)
            self.im_in = self.imageArray[:,:,::-1]
            im2show, numTre = self.det()
            img_rgb = cv2.cvtColor(im2show, cv2.COLOR_BGRA2RGB)
            QTimage = QImage(img_rgb.data, img_rgb.shape[1], img_rgb.shape[0], QImage.Format_RGB888)
            self.label_3.setPixmap(QPixmap.fromImage(QTimage))
            self.label_3.setScaledContents(True)
            self.lineEdit_3.setText(str(numTre))
            self.update()
        elif self.flag_timerDet == False and self.flag_showIamge == True:
            #只显示图像不进行检测
            self.label_3.setPixmap(QPixmap.fromImage(self.image))
            self.label_3.setScaledContents(True)
            self.update()
        else:
            print("error")


    def stopTCP(self):
        #先延时５毫秒
        time.sleep(2)
        print("停止检测")
        if self.server.isListening() == True:
            #关闭服务器
            self.server.close()
            self.socket.close()
            # 清空画面
            self.label_3.clear()
            self.stopTcpconnect.setEnabled(False)
            self.lineEdit_4.setText(str("网络连接断开"))
            #延时２毫秒　　确定网络断开了
            time.sleep(2)
            self.startTcpconnect.setEnabled(True)

    def startTCP(self):
        print("TCP初始化")
        if self.server.isListening() == False:
            #初始化网络
            self.initServer()

if __name__ == '__main__':

    app = QtWidgets.QApplication(sys.argv)
    tcpderver = TcpServer()
    tcpderver.show()
    sys.exit(app.exec_())


