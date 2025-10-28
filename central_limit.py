#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 28 14:20:24 2025

@author: syed
"""

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px

def main():
   
    uploaded_file = st.sidebar.file_uploader("Upload CSV DATA", ['csv'])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df)
        
        st.subheader("Revenue by Product Type")
        revenue_by_type = df.groupby('Product type')['Revenue generated'].sum().reset_index()
        fig = px.pie(revenue_by_type, values='Revenue generated', names='Product type', hole=0.4)
        st.plotly_chart(fig, use_container_width=True)
        
        st.sidebar.radio("Navigations", ['Home', 'Executive Summary', 'Customer Analysis'])
       
        
if __name__ == "__main__":
    main()
