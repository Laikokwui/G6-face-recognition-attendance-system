# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 09:06:35 2021

@author: Asus
"""

import streamlit as st
import cv2
from Code import utilities, yasir_utilities

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
            file_details = {"FileName":image_file.name,"FileType":image_file.type}
            st.success('Upload Successful!')
    