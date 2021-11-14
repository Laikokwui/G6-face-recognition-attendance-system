# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 16:14:12 2021

@author: Asus
"""
import cv2
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--display", type=int, default=1, help="whether or not output frame should be displayed")
args = vars(ap.parse_args())

# Load the cascade
face_cascade = cv2.CascadeClassifier('C:/Users/Asus/anaconda3/envs/tensorgpu/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')

# To capture video from webcam. 
cap = cv2.VideoCapture(0)

while True:
    # Read the frame
    _, img = cap.read()
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minSize=(30, 30), minNeighbors=7)
    
    # Draw the rectangle around each face
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
    text = "Press \"Q\" to stop" + "Number of face " + str(len(faces))
    cv2.putText(img, text, (10, img.shape[0] - 25),
		cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 2)
    
    if args["display"] > 0:
        #display
        cv2.imshow("Image Collection", img)
        
        # Stop if escape key is pressed
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord("q"):
            break

# Release the VideoCapture object
cap.release()
cv2.destroyAllWindows()