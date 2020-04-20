import sys
import serial
import serial.tools.list_ports
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtCore import QTimer
from df import Ui_Form
import math
import os
import numpy as np
import cv2
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtGui import QKeyEvent
from df import Ui_MainWindow
from PyQt5.QtCore import *
from PyQt5.QtNetwork import *
import time
from PyQt5 import QtWidgets,QtGui,QtCore,QtNetwork
from detect import deteTrepang
import qimage2ndarray

class Serial_Port(deteTrepang):
    def __init__(self):
        super(Serial_Port, self).__init__()
        self.init()
        self.ser = serial.Serial()
        self.port_check()
        # 接收数据和发送数据数目置零
        self.data_num_received = 0
        self.lineEdit.setText(str(self.data_num_received))
        self.data_num_sended = 0
        self.lineEdit_2.setText(str(self.data_num_sended))
        # 标志位
        self.flag_detImage = False
        self.flag_detVIdeo = False
        self.flag_showIamge = False
        #默认实时检测
        self.flag_timerDet = True
        #　刚开始设置检测视频标志位为false
        self.flag_dertvideo = False
        self.flag_sureViewVideo = False
        #tcp发送的数据
        self.controlData = bytearray(39)
        self.initTcpSend()

    def init(self):
        # 串口检测按钮
        self.s1__box_1.clicked.connect(self.port_check)
        # 打开串口按钮
        self.open_button.clicked.connect(self.port_open)
        # 关闭串口按钮
        self.close_button.clicked.connect(self.port_close)
        # 清除接收按钮
        self.s2__clear_button.clicked.connect(self.receive_data_clear)
        # 清除发送按钮
        self.s3__clear_button.clicked.connect(self.send_data_clear)
        # 发送数据
        self.s3__send_button.clicked.connect(self.data_send)
        # 检测本地图像
        self.whatchImagrbutton.clicked.connect(self.viewImage)
        # 检测本地图像
        self.open_button_2.clicked.connect(self.detImage)
        # 预览本地视频
        self.whatchVideogrbutton.clicked.connect(self.viewVideoTimer)
        # 检测本地视频
        self.close_button_2.clicked.connect(self.detVideo)
        # 初始化定时器
        self.timer_video = QTimer(self)
        self.timer_video.timeout.connect(self.viewVideo)
        # 定时器接收数据
        self.timer_serial = QTimer(self)
        self.timer_serial.timeout.connect(self.data_receive)
        self.timer_ControlWay = QTimer(self)
        self.timer_ControlWay.timeout.connect(self.control_way)
        self.timer_ControlWay.start(100)
        self.flag_router = False
        self.flag_key = False
        # 加载图像
        self.loadImage()
        # 加载视频
        self.loadVideo()

    #初始化TCP发送
    def initTcpSend(self):
        '''
                    前进(0024)                        　　　上浮(4089)
           左滚(4081)         右滚(0010)           左转(0000) 　　　　右转(4067)
                    后退(4084)                        　　　下潜(0005)
        '''
        #角度控制
        a = '+'.encode()
        self.controlData[0] = 0x83
        self.controlData[1] = 0x83
        self.controlData[2] = 0x83
        #前后运动
        self.controlData[3] = 0x39
        self.controlData[4] = 0x36
        self.controlData[5] = 0x31
        self.controlData[6] = 0x32
        # 左右横滚运动
        self.controlData[7] = 0x32
        self.controlData[8] = 0x33
        self.controlData[9] = 0x39
        self.controlData[10] = 0x31
        # 左右运动
        self.controlData[11] = 0x37
        self.controlData[12] = 0x33
        self.controlData[13] = 0x31
        self.controlData[14] = 0x32
        # 上浮个下潜
        self.controlData[15] = 0x38
        self.controlData[16] = 0x37
        self.controlData[17] = 0x31
        self.controlData[18] = 0x32
        # 灯
        self.controlData[19] = 0x30
        self.controlData[20] = 0x30
        self.controlData[21] = 0x30
        self.controlData[22] = 0x30
        self.controlData[23] = 0x30
        self.controlData[24] = 0x30
        self.controlData[25] = 0x30
        self.controlData[26] = 0x30
        self.controlData[27] = 0x30
        self.controlData[28] = 0x30
        self.controlData[29] = 0x30
        self.controlData[30] = 0x30
        self.controlData[31] = int.from_bytes(a,'big')
        self.controlData[32] = 0x30
        self.controlData[33] = 0x30
        self.controlData[34] = 0x30
        self.controlData[35] = int.from_bytes(a,'big')
        self.controlData[36] = 0x30
        self.controlData[37] = 0x30
        self.controlData[38] = 0x30
        #print(self.controlData)
        self.jointData = bytes("0000+000+000",'utf-8')
        self.routerData = bytes(0)

    def keyPressEvent(self, QKeyEvent):
        if self.s1__box_8.currentText() == str("按键控制"):
            print("按键控制")
            #　上键按下
            if QKeyEvent.key() == Qt.Key_Up:
                print("前进")
                self.controlData[3] = 0x30
                self.controlData[4] = 0x30
                self.controlData[5] = 0x39
                self.controlData[6] = 0x30
            # 下键按下
            if QKeyEvent.key() == Qt.Key_Down:
                print("后退")
                self.controlData[3] = 0x30
                self.controlData[4] = 0x30
                self.controlData[5] = 0x35
                self.controlData[6] = 0x33
            # 左键按下
            if QKeyEvent.key() == Qt.Key_Left:
                print("左转")
                self.controlData[11] = 0x30
                self.controlData[12] = 0x30
                self.controlData[13] = 0x31
                self.controlData[14] = 0x30
            # 右键按下
            if QKeyEvent.key() == Qt.Key_Right:
                print("右转")
                self.controlData[11] = 0x30
                self.controlData[12] = 0x30
                self.controlData[13] = 0x35
                self.controlData[14] = 0x33
        else:
            print("远程遥控")

    def loadImage(self):
        # 加载图像
        self.jpg_dir = './images'
        self.image_format = ".jpg"
        local_jpgs = self.listJpgFile()
        self.s1__box_10.clear()
        for jpg in local_jpgs:
            self.s1__box_10.addItem(jpg)

    def listJpgFile(self):
        fs = os.listdir(self.jpg_dir)
        for i in range(len(fs) - 1, -1, -1):
            # 如果后缀不是.txt就将该文件删除掉
            if not fs[i].endswith(self.image_format):
                del fs[i]
        return fs

    def loadVideo(self):
        # 加载本地视频
        self.video_dir = './videos'
        self.video_format = ".MP4"
        local_videos = self.listMp4File()
        self.s1__box_11.clear()
        for jpg in local_videos:
            self.s1__box_11.addItem(jpg)

    def listMp4File(self):
        fs = os.listdir(self.video_dir)
        for i in range(len(fs) - 1, -1, -1):
            # 如果后缀不是.txt就将该文件删除掉
            if not fs[i].endswith(self.video_format):
                del fs[i]
        return fs

    #预览一张照片
    def viewImage(self):
        print("预览一张图像")
        image_name = str(self.s1__box_10.currentText())
        self.im_file = './images/' + image_name
        image = QPixmap(self.im_file)
        self.label_3.setPixmap(image)
        self.label_3.setScaledContents(True)
        self.lineEdit_3.setText("")

    def viewVideoTimer(self):
        self.flag_sureViewVideo = True
        print("预览一个视频")
        video_name = str(self.s1__box_11.currentText())
        self.video_file = './videos/' + video_name
        self.cap = cv2.VideoCapture(self.video_file)
        self.timer_video.start(40)

    def viewVideo(self):
        ret, frame = self.cap.read()
        if (self.cap.isOpened()):
            ret, frame = self.cap.read()
            if ret:
                if self.flag_dertvideo:
                    self.im_in = frame
                    im2show, numTre = self.det()
                    img_rgb = cv2.cvtColor(im2show, cv2.COLOR_BGRA2RGB)
                    self.lineEdit_3.setText(str(numTre))
                else:
                    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
                QTimage = QImage(img_rgb.data, img_rgb.shape[1], img_rgb.shape[0], QImage.Format_RGB888)
                self.label_3.setPixmap(QPixmap.fromImage(QTimage))
                self.label_3.setScaledContents(True)
            else:
                self.cap.release()
                self.timer_video.stop()
        else:
            print("open file error,init again")

    # 检测视频按钮回调函数
    def detVideo(self):
        #设置标志位
        self.flag_dertvideo = not self.flag_dertvideo

    # 串口检测
    def port_check(self):
        # 检测所有存在的串口，将信息存储在字典中
        self.Com_Dict = {}
        port_list = list(serial.tools.list_ports.comports())
        self.s1__box_2.clear()
        for port in port_list:
            self.Com_Dict["%s" % port[0]] = "%s" % port[1]
            self.s1__box_2.addItem(port[0])

    # 打开串口
    def port_open(self):
        self.ser.port = self.s1__box_2.currentText()
        self.ser.baudrate = int(self.s1__box_3.currentText())
        self.ser.bytesize = int(self.s1__box_4.currentText())
        self.ser.stopbits = int(self.s1__box_6.currentText())
        self.ser.parity = self.s1__box_5.currentText()
        try:
            self.ser.open()
        except:
            QMessageBox.critical(self, "Port Error", "此串口不能被打开！")
            return None
        # 打开串口接收定时器，周期为2ms
        self.timer_serial.stop()
        self.timer_serial.start(4)
        if self.ser.isOpen():
            self.open_button.setEnabled(False)
            self.close_button.setEnabled(True)
        #设置缓冲区大小
        self.ser.reset_input_buffer()

    # 关闭串口
    def port_close(self):
        #关闭定时器
        self.timer_serial.stop()
        try:
            self.ser.close()
        except:
            pass
        self.open_button.setEnabled(True)
        self.close_button.setEnabled(False)
        # 接收数据和发送数据数目置零
        self.data_num_received = 0
        self.lineEdit.setText(str(self.data_num_received))
        self.data_num_sended = 0
        self.lineEdit_2.setText(str(self.data_num_sended))

    # 发送数据
    def data_send(self):
        if self.ser.isOpen():
            sendData = str("RecordRouterData")
            num = self.ser.write(sendData.encode())
            self.data_num_sended += num
            self.lineEdit_2.setText(str(self.data_num_sended))
        else:
            pass

    # 接收数据
    def data_receive(self):
        try:
            num = self.ser.inWaiting()
        except:
            self.port_close()
            return None
        if num > 0:
            data = self.ser.read(num)
            num = len(data)
            self.routerData = data + self.jointData
            #print(len(self.routerData))
            # 串口接收到的字符串为b'123',要转化成unicode字符串才能输出到窗口中去
            self.s2__receive_text.setText("")
            self.s2__receive_text.insertPlainText(data.decode('iso-8859-1'))
            # 统计接收字符的数量
            #self.data_num_received += num
            self.lineEdit.setText("")
            self.lineEdit.setText(str(num))
        else:
            pass
    #控制方式
    def control_way(self):
        if self.s1__box_8.currentText() == str("远程遥控"):
            self.flag_router = True
            self.flag_key = False
        else:
            self.flag_router = False
            self.flag_key = True
    # 清除显示
    def send_data_clear(self):
        self.s3__send_text.setText("")

    def receive_data_clear(self):
        self.s2__receive_text.setText("")
        self.data_num_received = 0

    #检测一张图像
    def detImage(self):
        self.im_in = np.array(cv2.imread(self.im_file))
        im2show,numberTrepang = self.det()
        #检测之后输出的也是bgr格式
        img_rgb = cv2.cvtColor(im2show, cv2.COLOR_BGRA2RGB)
        QTimage = QImage(img_rgb.data, img_rgb.shape[1], img_rgb.shape[0], QImage.Format_RGB888)
        self.label_3.setPixmap(QPixmap.fromImage(QTimage))
        self.label_3.setScaledContents(True)
        self.lineEdit_3.setText(str(numberTrepang))

if __name__ == '__main__':

    app = QtWidgets.QApplication(sys.argv)
    detTR = Serial_Port()
    detTR.show()
    sys.exit(app.exec_())
