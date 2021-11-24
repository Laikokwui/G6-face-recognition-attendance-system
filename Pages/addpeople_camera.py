# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 09:11:18 2021

@author: Asus
"""
import streamlit as st
import cv2
import imutils
import os
from Code import utilities

def app():
    st.title('Add People by Using Webcam')
    
    with st.form(key='add_people'):
        name = st.text_input('Name')
        submit_button = st.form_submit_button('Upload New Person')
    
    if submit_button:
        if name == "":
            st.error("Name cannot be empty!")
        else:
            labelsPath = "C:/Users/Asus/Documents/G6-face-recognition-attendance-system/yolov4/obj.names"
            weightsPath = "C:/Users/Asus/Documents/G6-face-recognition-attendance-system/yolov4/yolov4-obj_last.weights"
            configPath = "C:/Users/Asus/Documents/G6-face-recognition-attendance-system/yolov4/yolov4-obj.cfg"
            net, ln, labels = utilities.Yolov4Setup(labelsPath, weightsPath, configPath, True)
            
            capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
           
            while capture.isOpened():
                stop = False
                grabbed, frame = capture.read()
                if not grabbed:
                    break
                frame = imutils.resize(frame,width=700)
                results = utilities.detect_face(frame, net, ln, 0.3, 0.3, objIdx=labels.index("face"))
                
                if len(results) == 1:
                    for (i, (prob, bbox, centroid)) in enumerate(results):
                        # get the coordinate from bbox
                        (startX, startY, endX, endY) = bbox
                        (cX, cY) = centroid
                        color = (255, 255, 0)
                        
                        img = frame
                        crop_img = img[startY:endY, startX:endX]
                        
                        if cv2.waitKey(1) & 0xFF == ord('s'):
                            path = r"C:\Users\Asus\Documents\G6-face-recognition-attendance-system\Datasets\Database"
                            imagename = name + ".jpg"
                            imagepath = os.path.join(path,imagename)
                            cv2.imwrite(imagepath, crop_img)
                            stop = True
                        
                        cv2.putText(img, "Face_Detect %s" % ("{:.3f}".format(prob)), (startX, startY-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    		
                        cv2.rectangle(img, (startX, startY), (endX, endY), color, 2)
                    
                    
                    cv2.imshow("Image Collection", img)
                    
                if stop:
                    break
                    
                                     
            capture.release()
            cv2.destroyAllWindows()
            st.success('Upload Successful!')