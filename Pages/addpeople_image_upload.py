# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 09:06:35 2021

@author: Asus
"""

import streamlit as st
import cv2
import os
from PIL import Image
import numpy as np
from numpy import asarray
from Code import utilities

ROOT = "C:/Users/Asus/Documents/G6-face-recognition-attendance-system/"

def app():
    st.title('Add People Through Image Upload')
    
    with st.form(key='add_people'):
        name = st.text_input('Name')
        image_file = st.file_uploader('Upload photos',type=['jpg'])
        submit_button = st.form_submit_button('Upload New Person')
    
    if submit_button:
        if name == "":
            st.error("Name cannot be empty!")
        if image_file is None:
            st.error("Must upload an image!")
        else:
            # Set parameters for yoloV4
            labelsPath = "C:/Users/Asus/Documents/G6-face-recognition-attendance-system/yolov4/obj.names"
            weightsPath = "C:/Users/Asus/Documents/G6-face-recognition-attendance-system/yolov4/yolov4-obj_last.weights"
            configPath = "C:/Users/Asus/Documents/G6-face-recognition-attendance-system/yolov4/yolov4-obj.cfg"
            net, ln, labels = utilities.Yolov4Setup(labelsPath, weightsPath, configPath, True)
            
            # read images
            img = Image.open(image_file)
            img = asarray(img)
            
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
                    crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
                    
                    # try save the image if not pass
                    try:
                        path = r"C:\Users\Asus\Documents\G6-face-recognition-attendance-system\Datasets\Database"
                        imagename = name + ".jpg"
                        imagepath = os.path.join(path,imagename)
                        cv2.imwrite(imagepath, crop_img)
                    except:
                        pass
            
                st.success('Upload Successful!')
            else:
                st.error("There are more than 1 faces in the picture!")
    