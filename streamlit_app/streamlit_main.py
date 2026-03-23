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

#Sections
pages = [
    st.Page(run_home_page, title="Home", icon=":material/home:"),
    st.Page(run_page_0, title="Dataset", icon=":material/database:"),
    st.Page(predict_demo, title="Demo", icon=":material/database:")
]

# Create the navigation menu
pg = st.navigation(pages, position="sidebar") # Position can be "sidebar" or "hidden"

with st.sidebar:
    display_simulated_date()

# Run the selected page
pg.run()
