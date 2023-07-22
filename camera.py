import cv2
import torch
import numpy as np
import os
# from yolov5 import load

# weights = torch.load('best.pt')
# basedir = os.path.dirname(os.path.realpath(__file__))
# model = torch.load(basedir+'/best.pt')
# path = f"{basedir}+'/best.pt'"
model = torch.hub.load("ultralytics/yolov5", "custom", path='best.pt', force_reload=True)

# # set model parameters
# model.conf = 0.25  # NMS confidence threshold
# model.iou = 0.45  # NMS IoU threshold
# model.agnostic = False  # NMS class-agnostic
# model.multi_label = False  # NMS multiple labels per box
# model.max_det = 1000  # maximum number of detections per image
# Create a YOLOv5 model
# model = Model()  # specify the number of classes cfg='path/to/yolov5s.yaml', nc=80
# model.load_state_dict(weights)
# model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
      
class VideoCamera(object):
    def __init__(self):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        self.video = cv2.VideoCapture(0)
#        self.video = cv2.resize(self.video,(840,640))
        # If you decide to use video.mp4, you must have this file in the folder
        # as the main.py.
        # self.video = cv2.VideoCapture('video.mp4')
    
    def __del__(self):
        self.video.release()
    
    def get_frame(self):
        success, image = self.video.read()
        image = cv2.flip(image, 1)
        results = model(image)
        a = np.squeeze(results.render())

        
       
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()