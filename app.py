# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 17:25:05 2021

@author: Asus
"""
import numpy as np
import streamlit as st

from multipage import MultiPage
from Pages import home, about

app = MultiPage()

# Title of the main page
st.title("Group 7 - ML Assignment")

# Add all your application here
app.add_page("Home", home.app)
app.add_page("About Us", about.app)

# The main app
app.run()

