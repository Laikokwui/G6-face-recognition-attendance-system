# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 17:25:05 2021

@author: Asus
"""
import numpy as np
import streamlit as st

from multipage import MultiPage
from Pages import home, about, addpeople_image_upload, addpeople_camera

st.set_page_config(layout="wide")
app = MultiPage()

# Title of the main page
st.title("Face Recognition Attendance System")

# Add all your application here
app.add_page("Home", home.app)
app.add_page("Add People (Image Upload)", addpeople_image_upload.app)
app.add_page("Add People (Camera Input)", addpeople_camera.app)
app.add_page("About Us", about.app)

# The main app
app.run()

