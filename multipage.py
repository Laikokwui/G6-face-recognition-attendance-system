# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 17:45:22 2021

@author: Asus

This file is the framework for generating multiple Streamlit applications 
through an object oriented framework. 
"""

import streamlit as st

# Define the multipage class to manage the multiple apps in our program 
class MultiPage:
    def __init__(self) -> None:
        self.pages = []
    
    def add_page(self, title, func) -> None: 
        self.pages.append({
            "title": title, 
            "function": func
        })

    def run(self):
        page = st.sidebar.selectbox(
            'App Navigation', 
            self.pages, 
            format_func=lambda page: page['title']
        )

        page['function']()


