# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
import cv2
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
import torchvision.datasets as dset
from scipy.misc import imread
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
from model.nms.nms_wrapper import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections
from model.utils.blob import im_list_to_blob
from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet
import pdb
import datetime
import sys
from df import Ui_Form
from PyQt5 import QtCore, QtGui, QtWidgets, QtNetwork
import datetime
from PyQt5.QtCore import pyqtSlot,pyqtSignal,QByteArray,QDataStream,qUncompress
from PyQt5.QtGui import QPixmap,QImage
from PyQt5.QtWidgets import QApplication
from df import Ui_Form

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3

class deteTrepang(QtWidgets.QWidget,Ui_Form):
    #初始化函数
    def __init__(self):
        super(deteTrepang, self).__init__()
        #界面初始化
        self.setupUi(self)
        # 锁住按键
        self.grabKeyboard()
        #'cfgs/res50.yml'
        cfg_from_file('cfgs/res50.yml')
        cfg.USE_GPU_NMS = 1
        # 随机种子设置为3
        np.random.seed(cfg.RNG_SEED)
        self.input_dir = 'models/res50/pascal_voc'
        self.load_name = 'models/res50/pascal_voc/faster_rcnn_1_18_2720.pth'
        self.pascal_classes = np.asarray(['__background__', 'trepang'])
        self.fasterRCNN = resnet(self.pascal_classes, 50, pretrained=False, class_agnostic=False)
        self.fasterRCNN.create_architecture()
        self.checkpoint = torch.load(self.load_name)
        self.fasterRCNN.load_state_dict(self.checkpoint['model'])
        if 'pooling_mode' in self.checkpoint.keys():  # crop
            cfg.POOLING_MODE = self.checkpoint['pooling_mode']
        print('load model successfully!')
        print("load checkpoint %s" % (self.load_name))
        # initilize the tensor holder here.
        self.im_data = torch.FloatTensor(1)
        self.im_info = torch.FloatTensor(1)
        self.num_boxes = torch.LongTensor(1)
        self.gt_boxes = torch.FloatTensor(1)
        self.im_data = self.im_data.cuda()
        self.im_info = self.im_info.cuda()
        self.num_boxes = self.num_boxes.cuda()
        self.gt_boxes = self.gt_boxes.cuda()
        # make variable
        with torch.no_grad():
            self.im_data = Variable(self.im_data)
            self.im_info = Variable(self.im_info)
            self.num_boxes = Variable(self.num_boxes)
            self.gt_boxes = Variable(self.gt_boxes)
        #使用显卡
        cfg.CUDA = True
        self.fasterRCNN.cuda()
        self.fasterRCNN.eval()
        self.max_per_image = 100
        self.thresh = 0.05
        self.im_file = './images/trepang.jpg'
        #函数借口
        self.im_in = None
    #检测的程序
    def det(self):
        #imread读取的是ｂｇｒ格式
        #self.im_in = np.array(imread(self.im_file))
        if len(self.im_in.shape) == 2:
            self.im_in = self.im_in[:, :, np.newaxis]
            self.im_in = np.concatenate((self.im_in, self.im_in, self.im_in), axis=2)
        self.im = self.im_in
        blobs, im_scales = self._get_image_blob()
        assert len(im_scales) == 1, "Only single-image batch implemented"
        im_blob = blobs
        im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)
        im_data_pt = torch.from_numpy(im_blob)
        im_data_pt = im_data_pt.permute(0, 3, 1, 2)
        im_info_pt = torch.from_numpy(im_info_np)

        self.im_data.data.resize_(im_data_pt.size()).copy_(im_data_pt)
        self.im_info.data.resize_(im_info_pt.size()).copy_(im_info_pt)
        self.gt_boxes.data.resize_(1, 1, 5).zero_()
        self.num_boxes.data.resize_(1).zero_()

        rois, cls_prob, bbox_pred, \
        rpn_loss_cls, rpn_loss_box, \
        RCNN_loss_cls, RCNN_loss_bbox, \
        rois_label = self.fasterRCNN(self.im_data, self.im_info, self.gt_boxes, self.num_boxes)

        scores = cls_prob.data
        boxes = rois.data[:, :, 1:5]

        if cfg.TEST.BBOX_REG:
            # Apply bounding-box regression deltas
            box_deltas = bbox_pred.data
            if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                # Optionally normalize targets by a precomputed mean and stdev
                box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                             + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                box_deltas = box_deltas.view(1, -1, 4 * len(self.pascal_classes))

            pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
            pred_boxes = clip_boxes(pred_boxes, self.im_info.data, 1)

        pred_boxes /= im_scales[0]

        scores = scores.squeeze()
        pred_boxes = pred_boxes.squeeze()

        im2show = np.copy(self.im)
        for j in xrange(1, len(self.pascal_classes)):
            inds = torch.nonzero(scores[:, j] > self.thresh).view(-1)
            # if there is det
            if inds.numel() > 0:
                cls_scores = scores[:, j][inds]
                _, order = torch.sort(cls_scores, 0, True)

                cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]

                cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
                cls_dets = cls_dets[order]
                keep = nms(cls_dets, cfg.TEST.NMS, force_cpu=not cfg.USE_GPU_NMS)
                cls_dets = cls_dets[keep.view(-1).long()]

                im2show, number_tre = vis_detections(im2show, self.pascal_classes[j], cls_dets.cpu().numpy(), 0.5)
        return im2show, number_tre
        #cv2.imshow('test', im2show)
        #cv2.waitKey(0)
        #img_rgb = cv2.cvtColor(im2show, cv2.COLOR_BGRA2RGB)
        #QTimage = QImage(img_rgb.data, img_rgb.shape[1], img_rgb.shape[0], QImage.Format_RGB888)
        #self.label_3.setPixmap(QPixmap.fromImage(QTimage))
        #self.label_3.setScaledContents(True)

    def _get_image_blob(self):
        """Converts an image into a network input.
        Arguments:
          im (ndarray): a color image in BGR order
        Returns:
          blob (ndarray): a data blob holding an image pyramid
          im_scale_factors (list): list of image scales (relative to im) used
            in the image pyramid
        """
        im_orig = self.im.astype(np.float32, copy=True)
        im_orig -= cfg.PIXEL_MEANS

        im_shape = im_orig.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])

        processed_ims = []
        im_scale_factors = []

        for target_size in cfg.TEST.SCALES:
            im_scale = float(target_size) / float(im_size_min)
            # Prevent the biggest axis from being more than MAX_SIZE
            if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
                im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
            im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                            interpolation=cv2.INTER_LINEAR)
            im_scale_factors.append(im_scale)
            processed_ims.append(im)

        # Create a blob to hold the input images
        blob = im_list_to_blob(processed_ims)

        return blob, np.array(im_scale_factors)

if __name__ == '__main__':

    app = QtWidgets.QApplication(sys.argv)
    detTR = deteTrepang()
    detTR.show()
    detTR.det()
    sys.exit(app.exec_())





