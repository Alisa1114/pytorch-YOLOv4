# -*- coding: utf-8 -*-
'''
@Time          : 2020/04/26 15:48
@Author        : Tianxiaomo
@File          : camera.py
@Noice         :
@Modificattion :
    @Author    :
    @Time      :
    @Detail    :

'''
from __future__ import division
import cv2
from models import *
#from tool.darknet2pytorch import Darknet
import argparse
from tool.utils import *
import time


def arg_parse():
    """
    Parse arguements to the detect module

    """

    parser = argparse.ArgumentParser(description='YOLO v4 Cam Demo')
    parser.add_argument("--confidence", dest="confidence", help="Object Confidence to filter predictions", default=0.25)
    parser.add_argument("--nms_thresh", dest="nms_thresh", help="NMS Threshhold", default=0.4)
    parser.add_argument("--reso", dest='reso', help=
    "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default="160", type=str)
    parser.add_argument("--num_classes", dest="num_classes", help="Number of classes", default=2)
    return parser.parse_args()


if __name__ == '__main__':
    cfgfile = "cfg/yolov4.cfg"
    weightsfile = "checkpoints/Yolov4_epoch300.pth"

    args = arg_parse()
    confidence = float(args.confidence)
    nms_thesh = float(args.nms_thresh)
    #CUDA = torch.cuda.is_available() #原本的
    num_classes = int(args.num_classes)
    bbox_attrs = 5 + num_classes
    class_names = load_class_names("data/_classes.txt")
    
    model = Yolov4(n_classes=num_classes)
    pretrained_dict = torch.load(weightsfile, map_location=torch.device('cuda'))
    model.load_state_dict(pretrained_dict)
    #model = Darknet(cfgfile) #原本的
    #model.load_weights(weightsfile) #原本的

    if torch.cuda.is_available():
        model.cuda()

    model.eval()
    cap = cv2.VideoCapture(0)

    assert cap.isOpened(), 'Cannot capture source'

    frames = 0
    start = time.time()
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            #sized = cv2.resize(frame, (model.width, model.height)) #原本的
            sized = cv2.resize(frame, (608, 608))
            #sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB) #用攝影機捕捉的影像原本就是RGB了，大概
            boxes = do_detect(model, sized, 0.5, num_classes, 0.4)
            #boxes = do_detect(model, sized, 0.5, 0.4, CUDA) #原本的

            orig_im = plot_boxes_cv2(frame, boxes, class_names=class_names)

            cv2.imshow("frame", orig_im)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            frames += 1
            print("FPS of the video is {:5.2f}".format(frames / (time.time() - start)))
        else:
            break
