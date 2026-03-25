# warnings
import warnings
warnings.filterwarnings("ignore")

#streamlit
import streamlit as st

from streamlit_home import run_home_page
from streamlit_0 import run_page_0
from predict_demo import predict_demo

from components.clock import display_simulated_date

# ----------------------------------------------
# Configuration, CSS
# ----------------------------------------------

st.set_page_config(page_title="Road Accidents: Multi-Class classification project") #, layout="wide"

st.markdown("""
<style>
    .block-container {
        padding-top: 1rem; /* Adjust this value to control the space (e.g., 0rem for minimal) */
        padding-bottom: 0rem;
        padding-left: 1rem;
        padding-right: 1rem;
    
    .single-space {
        line-height: 1.5; /* Adjust this value as needed */
        margin-top: 5px; /* Removes top margin of the paragraph */
        margin-bottom: 5px; /* Removes bottom margin of the paragraph */
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------
# Navigation
#
# Define pages using st.Page with custom titles and icons
# ----------------------------------------------

#material icons: eg 
# analytics bar_chart code computer insights database
# arrow_forward_ios trending_up

# Draft pages
def run_page_accidents():
    st.write("\n\n\n")
    st.code("""
        Predicting Accidents severity.
            
        DATA
        - Source: ahmedlahlou/accidents-in-france-from-2005-to-2016
        - TABLES:
            - caracteristics
                - accident id
                - date
                - ...
            - users
                - severity:  1 - Unscathed 2 - Killed 3 - Hospitalized wounded 4 - Light injury
                - ...
            - places
            - vehicles
            - holidays
        
        MODEL
        - Predict severity class of an accident
        - Main challenge: Imbalance for the 4 classes
        - Model chosen:
            - LightGBM classifier
            - Borderline SMOTE (boost to 50% of majority class)
    """)

def run_page_simulating():
    st.write("\n\n\n")
    st.code("""
        Simulating continuous data
            
        [show distribution of data through time]
        [clock for simulated time]
            
        update_data.py (POST/UPDATEDATA):
            - accidents_full.db -> accidents_2010XXXXXXXX.db
        
        [If possible show live demo]
    """)

def run_page_architecture():
    st.write("\n\n\n")
    st.code("""
        DOCKER compose:
            - api
            - streamlit
            - MLFlow
            - scheduler

        ARCHITECTURE/API
        
        TRAINING PIPELINE:
            [accidents_full.db]
        *UPDATE DATA*
            [accidents_{timestamp}.db]
        *BUILD FEATURES*
            [processed/accidents_{timestamp}.db]
        *TRAIN MODEL*
            [model_{timestamp}.pkl]
        *TRACK MODEL*
            (send to MLFlow, Dagshub)
        *SELECT MODEL* (f1)
            [best_model_bundle.pkl]

        OTHER API calls
        - run pipeline
        - predict_model
    """)

def run_page_mlflow():
    st.write("\n\n\n")
    st.code("""
        MLFLOW
    """)

#Sections
pages = [
    #st.Page(run_home_page, title="Home", icon=":material/home:"),
    #st.Page(run_page_0, title="Dataset", icon=":material/database:"),

    st.Page(run_page_accidents, title="Accidents"),
    st.Page(run_page_simulating, title="Simulating Continuous Data"),
    st.Page(run_page_architecture, title="Architecture"),
    st.Page(run_page_mlflow, title="MLFlow and tracking"),
    st.Page(predict_demo, title="Demo")
]

# Create the navigation menu
pg = st.navigation(pages, position="sidebar") # Position can be "sidebar" or "hidden"

with st.sidebar:
    display_simulated_date()

# Run the selected page
pg.run()