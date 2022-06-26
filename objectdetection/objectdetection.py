#python3.8.5
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw
from random import randint
import sys

COCO_NAMES = './objectdetection/coco.names'
YOLOV3_WEIGHTS = './objectdetection/yolov3.weights'
YOLOV3_CFG = './objectdetection/yolov3.cfg'

def color_generator():
    color = []
    for i in range(3):
        color.append(randint(0,255)-randint(0,255)%10)
    return tuple(color)

def get_people_coordinates(image_path):
    #'coco.names', 'yolo3.weights', 'yolov3.cfg' must exist in cwd
    
    try:
        f = open(COCO_NAMES)
        f = open(YOLOV3_WEIGHTS)
        f = open(YOLOV3_CFG)
    except IOError:
        print('File from CWD missing')
        exit()
    classes = None

    image = cv2.imread(image_path)
    height, width, ch = image.shape
    with open(COCO_NAMES, 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    # read pre-trained model and config file
    net = cv2.dnn.readNet(YOLOV3_WEIGHTS, YOLOV3_CFG)
    net.setInput(cv2.dnn.blobFromImage(image, 0.00392, (416,416), (0,0,0), True, crop=False))
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    outs = net.forward(output_layers)


    boxes = []
    class_ids = []
    confidences = []
    boxes = []
    #puts results in different lists (linked by order)
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.1:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    #finds the best-fitting box
    
    indices_1 = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.3)
    indices_2 = cv2.dnn.NMSBoxes(boxes, confidences, 0, 0)
    if len(indices_1) >= len(indices_2):
        indices = indices_1
    else:
        indices = indices_2

    correct_boxes = []
    for i in indices:
        i = i[0]
        box = boxes[i]
        if class_ids[i]==0:
            x,y,w,h = box
            correct_boxes.append([int(x),int(y),int(x+w),int(y+h)])
    
    return correct_boxes