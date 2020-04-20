# -*- coding: utf-8 -*-
import cv2
import time

class doubleCam():
    def __init__(self):
        super(doubleCam, self).__init__()
        '''
        设置分辨率　３－－宽　４－－高
        '''
        cap1 = cv2.VideoCapture(1)
        cap1.set(3,640)
        cap1.set(4,420)
        cap0 = cv2.VideoCapture(0)
        cap0.set(3, 640)
        cap0.set(4, 420)
        #延时两毫秒，确保摄像头打开
        time.sleep(2)
        if cap1.isOpened():
            print("第一个摄像头已打开")
        else:
            print("第一个摄像头已打开")
        if cap0.isOpened():
            print("第二个摄像头已打开")
        else:
            print("第二个摄像头已打开")
        while (True):
            ret1, frame1 = cap1.read()
            ret0, frame0 = cap0.read()
            cv2.imshow('frame1', frame1)
            cv2.imshow('frame0', frame0)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break





if __name__ == '__main__':

    camerTwo = doubleCam()

