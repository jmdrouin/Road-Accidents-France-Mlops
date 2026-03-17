# ----------------------------------------------
# import modules
# ----------------------------------------------

# classic packages
import pandas as pd
import numpy as np

# io, load, dump
import io
import joblib
from joblib import dump, load
from pathlib import Path

# visualization
import matplotlib.pyplot as plt
import seaborn as sns

# helper
import time
import random
import json

# custom libraries
import sys
#sys.path.append('../../library')

#streamlit
import streamlit as st
#import streamlit_functions as st_funct

# warnings
import warnings
warnings.filterwarnings("ignore")

# ----------------------------------------------
# function, attributes
# ----------------------------------------------

# raw data #q:use sample data ...

# ----------------------------------------------
# Page: Dataset
# ----------------------------------------------

def run_page_0():
    
    # ------------------------------------------
    # load data ...
    # ------------------------------------------
    
    # ------------------------------------------
    # page header, tabs
    # ------------------------------------------

    st.title("Dataset")
    tab1, tab2 = st.tabs(["Raw Data", "Consolidated Data"])

    # ------------------------------------------
    # Tab: Raw data
    # ------------------------------------------
    with tab1:
        st.subheader("Raw Data")
    
    # ------------------------------------------
    # Tab: consolidated data
    # ------------------------------------------
    with tab2:
        st.subheader("Consolidated Data")
    
    # ------------------------------------------
    #spacer at page bottom
    # ------------------------------------------
    st.write(' ')
    st.write(' ')
    st.write(' ')
    st.write(' ')
