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

# streamlit
import streamlit as st
#import streamlit_functions as st_funct

# warnings
import warnings
warnings.filterwarnings("ignore")


# ----------------------------------------------
# Homepage: Context & Objectives
# ----------------------------------------------
def run_home_page():
    st.title("Road Accidents In France")
    st.subheader(
        "Predicting Road-Accidents Injury Severity:"
        + " "
        + "A Multi-Class Classification Problem"
        )
    
    st.write( "#### Context" )
    st.markdown(
        """
        **Objective**: Predicting the severity of injury for individuals involved in road accidents.  
        **Problem Classification**: Supervised multi-class classification problem
        """
    )
    
    #st.image("art__road-accidents_longlat.png")
    
    # ------------------------------------------
    # spacer at page bottom
    # ------------------------------------------
    st.write( '' )
    st.write( '' )
    st.write( '' )
    st.write( '' )
