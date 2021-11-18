# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 17:55:07 2021

@author: Asus
"""

import streamlit as st
import cv2
import imutils
import uuid
import os
from Code import utilities

def app():
    st.title('Home Page')
    
    st.text('Face Recognition Attendance System')
    text = st.empty()
    camera_button = st.button("Add through Camera")
    with st.form(key='add_people'):
        name = st.text_input('Name')
        image_upload = st.file_uploader('Upload a photo',type=['jpg'])
        submit_button = st.form_submit_button('Upload New Person')
    
    image_placeholder = st.empty()
    cancel_btn = st.empty()
    takepicture_btn = st.empty()
        
    if submit_button:
        if name == "":
            st.text("Name cannot be empty!")
        else:
            st.text('Upload Successful!')
            
    if camera_button:
        cancel_button = cancel_btn.button("Cancel")
        takepicture_button = takepicture_btn.button("Take Picture")
        labelsPath = "C:/Users/Asus/Documents/G6-face-recognition-attendance-system/yolov4/obj.names"
        weightsPath = "C:/Users/Asus/Documents/G6-face-recognition-attendance-system/yolov4/yolov4-obj_last.weights"
        configPath = "C:/Users/Asus/Documents/G6-face-recognition-attendance-system/yolov4/yolov4-obj.cfg"
        net, ln, labels = utilities.Yolov4Setup(labelsPath, weightsPath, configPath, True)
        
        vid_obj = cv2.VideoCapture(0) 
        success = True
        
        while success:
          success, img = vid_obj.read()
          img = imutils.resize(img,width=700)
          results = utilities.detect_face(img, net, ln, 0.3, 0.3, objIdx=labels.index("face"))
          
          for (i, (prob, bbox, centroid)) in enumerate(results):
              # get the coordinate from bbox
              (startX, startY, endX, endY) = bbox
              (cX, cY) = centroid
              color = (255, 255, 0)
              
              # try save the image if not pass
              try:
                  if takepicture_button:
                      crop_img = img[startY:endY, startX:endX]
                      path = "C:/Users/Asus/Documents/G6-face-recognition-attendance-system/Datasets"
                      cv2.imwrite(os.path.join(path , '{}.jpg'.format(uuid.uuid1())), crop_img)
              except:
                  pass
                  
              cv2.putText(img, "Face_Detect %s" % ("{:.3f}".format(prob)), (startX, startY-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            		
              cv2.rectangle(img, (startX, startY), (endX, endY), color, 2)
              
          img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
          image_placeholder.image(img)
          
          if cancel_button:
              break

