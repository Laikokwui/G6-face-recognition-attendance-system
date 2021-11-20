# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 19:56:53 2021

@author: Asus
"""
import cv2
import numpy as np

# detect faces using Yolov4
def detect_face(frame, net, ln, min_conf, nms_thresh, objIdx):
    results = []
    boxes = []
    centroids = []
    confidences = []
    
    (H, W) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)
    
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            
            if classID == objIdx and confidence > min_conf:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                centroids.append((centerX, centerY))
                confidences.append(float(confidence))
                
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, min_conf, nms_thresh)
    
    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            r = (confidences[i], (x, y, x + w, y + h), centroids[i])
            results.append(r)
            
    return results


def Yolov4Setup(labelsPath, weightsPath, configPath, use_gpu):
    # Set parameters for yoloV4
    labels = open(labelsPath).read().strip().split("\n")
    
    # Load YoloV4
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    
    # use gpu
    if use_gpu:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    
    # get darknet layers
    ln = net.getLayerNames()
    ln = [ln[x - 1] for x in net.getUnconnectedOutLayers()]
    
    return net, ln, labels
