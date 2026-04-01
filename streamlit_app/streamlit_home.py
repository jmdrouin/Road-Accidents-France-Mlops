import streamlit as st
import streamlit_app.components.clock as clock
import os

# warnings
import warnings
warnings.filterwarnings("ignore")

# ----------------------------------------------
# Homepage: Context & Objectives
# ----------------------------------------------
def run_home_page():
    st.title("Road Accidents In France")
    st.subheader("An MLOps Architecture for Road Safety Analytics")
    
    st.write( "#### Context" )
    st.markdown(
        """
        **Objective**: Building an MLOps framework to predict accident severity.  
        **Focus**: Integration of scheduling, experiment tracking, and containerizing for a multi-class classification model.  

        **Team**: Jérôme Morin-Drouin, Alke Simmler  
        **Mentor**: Nicolas Fradin  
        """
    )
    st.divider()
    
    current_dir = os.path.dirname(__file__)
    img_path = os.path.join(current_dir, "pics", "highway_edit.png")
    
    col1, col2, col3 = st.columns([1, 3, 1]) #st.columns([1, 2, 1])
    with col2:
        if os.path.exists(img_path):
            st.image(img_path) #use_container_width=True #width=400
        else:
            st.error(f"Image not found: {img_path}")
    
    # ------------------------------------------
    # spacer at page bottom
    # ------------------------------------------
    st.write( '' )
    st.write( '' )
    st.write( '' )
    st.write( '' )
