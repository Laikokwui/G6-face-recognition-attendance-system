# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 15:18:54 2021

@author: Asus
"""
import os
from glob import glob
import sys
from tqdm import tqdm
import cv2
import utilities

ROOT = "C:/Users/Asus/Documents/G6-face-recognition-attendance-system/"

# Set parameters for yoloV4
labelsPath = "C:/Users/Asus/Documents/G6-face-recognition-attendance-system/yolov4/obj.names"
weightsPath = "C:/Users/Asus/Documents/G6-face-recognition-attendance-system/yolov4/yolov4-obj_last.weights"
configPath = "C:/Users/Asus/Documents/G6-face-recognition-attendance-system/yolov4/yolov4-obj.cfg"
net, ln, labels = utilities.Yolov4Setup(labelsPath, weightsPath, configPath, True)

# Create new sub folders for all the classes
class_list = os.listdir(ROOT+"Datasets/classification_data/train_data")

# loop through all the classes
for class_name in tqdm(range(len(class_list))):
    for file in glob(ROOT + "Datasets/classification_data/train_data/" + class_list[class_name] +"/*.jpg"):
        # read images
        img = cv2.imread(file, 1)
        
        # detect face location and how many are there
        results = utilities.detect_face(img, net, ln, 0.3, 0.3, objIdx=labels.index("face"))
        
        # if there are only one face in the image
        if len(results) == 1:
            # loop through results to get the face location
            for (i, (prob, bbox, centroid)) in enumerate(results):
                # get the coordinate from bbox
                (startX, startY, endX, endY) = bbox
                
                # crop the image
                crop_img = img[startY:endY, startX:endX]
                
                # try save the image if not pass
                try:
                    path = ROOT + "Datasets/Database/"
                    imagename = class_list[class_name]+'.jpg'
                    cv2.imwrite(os.path.join(path,imagename), crop_img)
                except:
                    pass
        break