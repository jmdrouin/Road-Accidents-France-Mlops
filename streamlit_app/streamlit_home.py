import streamlit as st
import streamlit_app.components.clock as clock

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
