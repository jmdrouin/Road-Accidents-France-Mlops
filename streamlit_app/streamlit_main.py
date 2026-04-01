# ----------------------------------------------
# Run streamlit app with:

# python -m streamlit run streamlit_app/streamlit_main.py

# .venv\Scripts\activate
# streamlit run streamlit_app/streamlit_main.py

# uv run streamlit run streamlit_app/streamlit_main.py

# $env:PYTHONPATH = "."; uv run streamlit run streamlit_app/streamlit_main.py
#
# ----------------------------------------------

# warnings
import warnings
warnings.filterwarnings("ignore")

#streamlit
import streamlit as st

from streamlit_home import run_home_page
#from streamlit_0 import run_page_0
from streamlit_diagram import streamlit_diagram
from streamlit_data import streamlit_data
from streamlit_schedule import streamlit_schedule
from streamlit_mlflow import streamlit_mlflow
from predict_demo import predict_demo
from streamlit_next import streamlit_next


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
# home database bar_chart insights model_training analytics trending_up
# code computer
# arrow_forward_ios
# history schedule description event_note terminal

#Sections
pages = [
    st.Page(run_home_page, title="Home", icon=":material/home:"),
    st.Page(streamlit_diagram, title="Architecture", icon=":material/insights:"),
    st.Page(streamlit_data, title="Data processing", icon=":material/database:"),
    #st.Page(streamlit_schedule, title="Scheduler", icon=":material/schedule:"),
    st.Page(streamlit_mlflow, title="MLflow", icon=":material/model_training:"),
    st.Page(predict_demo, title="Prediction", icon=":material/analytics:"),
    st.Page(streamlit_next, title="Next Steps", icon=":material/trending_up:")
]

# Create the navigation menu
pg = st.navigation(pages, position="sidebar") # Position can be "sidebar" or "hidden"

with st.sidebar:
    display_simulated_date()

# Run the selected page
pg.run()
