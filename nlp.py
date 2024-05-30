import pandas as pd
import streamlit as st
from dtale.views import startup
from dtale.app import get_instance

def eda_dtale(df):
    # Start the D-Tale server
    data_id = startup(df)
    
    # Get the D-Tale instance
    instance = get_instance(data_id)
    
    # Open the D-Tale app in the user's default web browser
    instance.open_browser()

    st.write("Exploratory Data Analysis (EDA) using D-Tale:")
    st.markdown("D-Tale is now open in your default web browser. Explore your data using the interactive interface.")
    