# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 09:11:18 2021

@author: Asus
"""
import streamlit as st
import cv2
import imutils
import uuid
import os
from Code import utilities

def app():
    st.title('Add People by Using Webcam')
    
    camera_btn = st.button("Open Camera")
    
    if camera_btn:
        labelsPath = "C:/Users/Asus/Documents/G6-face-recognition-attendance-system/yolov4/obj.names"
        weightsPath = "C:/Users/Asus/Documents/G6-face-recognition-attendance-system/yolov4/yolov4-obj_last.weights"
        configPath = "C:/Users/Asus/Documents/G6-face-recognition-attendance-system/yolov4/yolov4-obj.cfg"
        net, ln, labels = utilities.Yolov4Setup(labelsPath, weightsPath, configPath, True)
        
        capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
       
        while capture.isOpened():
            grabbed, img = capture.read()
            if not grabbed:
                break
            img = imutils.resize(img,width=700)
            results = utilities.detect_face(img, net, ln, 0.3, 0.3, objIdx=labels.index("face"))
          
            for (i, (prob, bbox, centroid)) in enumerate(results):
                # get the coordinate from bbox
                (startX, startY, endX, endY) = bbox
                (cX, cY) = centroid
                color = (255, 255, 0)
                crop_img = img[startY:endY, startX:endX]
                
                if cv2.waitKey(1) & 0xFF == ord('s'):
                    path = "C:/Users/Asus/Documents/G6-face-recognition-attendance-system/Datasets"
                    imgname = os.path.join(path , '{}.jpg'.format(uuid.uuid1()))
                    cv2.imwrite(imgname, crop_img)
                
                cv2.putText(img, "Face_Detect %s" % ("{:.3f}".format(prob)), (startX, startY-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            		
                cv2.rectangle(img, (startX, startY), (endX, endY), color, 2)
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2.imshow("Image Collection", img)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
          
        capture.release()
        cv2.destroyAllWindows()