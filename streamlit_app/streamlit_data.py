import streamlit as st
import os
import json

# warnings
import warnings
warnings.filterwarnings("ignore")

# ----------------------------------------------
# function, attributes
# ----------------------------------------------

# raw data #q:use sample data ...

# ----------------------------------------------
# Page: Data processing
# ----------------------------------------------

def streamlit_data():

    # ------------------------------------------
    # page header, tabs
    # ------------------------------------------

    st.title("Data processing")
    st.write(' ')

    current_dir = os.path.dirname(__file__)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Pipeline", "Dataframes", "Data Processing", "Steps", "Time Machine"])

    # ------------------------------------------
    # Tab: Pipeline
    # ------------------------------------------
    with tab1:
        st.subheader("Pipeline")

        img_path_pipelie = os.path.join(current_dir, "pics", "pipeline_steps.png")

        if os.path.exists(img_path_pipelie):
            st.image(img_path_pipelie, caption="Pipeline")
        else:
            st.error(f"Diagram not found: {img_path_pipelie}")
    
    # ------------------------------------------
    # Tab: Dataframes
    # ------------------------------------------
    with tab2:
        st.subheader("Dataframes")
        st.write('Raw data consist of 5 dataframes:')
        
        spacer1, col1, col2, spacer2 = st.columns([1, 3, 4, 3])

        with col1:
            st.markdown("""
            - **Accidents** (master)
            - **Places**
            - **Vehicles**
            - **Users**
            - **Holidays**
            """)
        
        with col2:
            img_path_dataframes = os.path.join(current_dir, "pics", "dataframes.png")
            st.image(img_path_dataframes, use_container_width=True)
    
    # ------------------------------------------
    # Tab: Data Processing
    # ------------------------------------------
    with tab3:
        st.subheader("Data Processing")
        
        img_path = os.path.join(current_dir, "pics", "fetch_process_white.png")

        if os.path.exists(img_path):
            st.image(img_path, caption="Data Processing Diagram")
        else:
            st.error(f"Diagram not found: {img_path}")
    
    # ------------------------------------------
    # Tab: Data Processing Steps
    # ------------------------------------------
    with tab4:
        st.subheader("Data loading and processing steps")

        st.write('Fetching and processing of data contains these steps:')
        '''
        st.markdown("""
            ### 1. Fetch Data
            * **First, raw data are fetched from a raw DB file until a cut-off date:**
                * Since no new data are available, the full dataset (stored locally) is accessed until a cut-off date; the cut-off date is calculated using a **Time Machine** (see below).
            * **The selected data are then written to a new DB file with the timestamp of the cut-off date:**
                * `data/raw/accidents_<timestamp>.db`
                    
            ### 2. Build Features
            * **Data are being cleaned (e.g., null rows are being dropped) and aggregated:**
                * Dependent dataframes (**Places, Vehicles, Users, and Holidays**) are merged into the **Accidents** (master) dataframe.
            * **The processed and aggregated data are then stored locally to another DB file:**
                * `data/processed/accidents_<timestamp>.db`
            """)
        '''
        with st.expander("Step 1: Fetch Data", expanded=True):
            st.write("**Raw data extraction:**")
            st.info("The full dataset is fetched from a raw DB file until cut-off date and filtered via a **Time Machine** logic to simulate real-time data arrival.")
            st.code("Output: data/raw/accidents_<timestamp>.db")
        
        with st.expander("Step 2: Build Features", expanded=False):
            st.write("**Cleaning & Merging:**")
            st.info("""
            **Data Transformation Steps:**
            - **Cleaning:** e.g. dropping of NULL values
            - **Merging:** Places, Vehicles, Users, Holidays → Accidents
            """)
            st.code("Output: data/processed/accidents_<timestamp>.db")
        
    # ------------------------------------------
    # Tab: Time Machine
    # ------------------------------------------
    with tab5:
        st.subheader("The Time Machine Concept")
        
        # Info
        st.info("""
        **Goal:** To simulate a moving "Present" within a historical dataset.  
        It maps the **real-world timeframe** of the experiment to a **simulated timeframe**, ensuring the experiment progresses at the same relative pace.
        """)

        # Logic
        with st.expander("How the Mapping Logic works", expanded=False):
            st.markdown("""
            *   **Normalization (Ratio):** Calculates the progress of the real experiment (e.g., halfway between `real_start` and `real_end` = **0.5**).
            *   **Linear Mapping:** Applies that same ratio to the simulated range. A ratio of 0.5 returns a timestamp exactly halfway between `sim_start` and `sim_end`.
            *   **DB Filtering (Cut-off):** The resulting `sim_time` acts as a moving cut-off. We query: `timestamp <= sim_time`.
            *   **In short:** It answers: *"How much time has passed since I started the script?"* vs. *"What date would that be in my historical data?"*
            """)
        
        #st.caption
        st.write("This concept allows to hide future data that 'hasn't happened yet' in the simulation.")

        # Config File Simulation
        st.write("Current Time Machine Configuration:")
        config_data = {
            "experiment": {
                "real_start": "2026-03-20 00:00:00",
                "real_end": "2026-04-01 00:00:00",
                "sim_start": "2005-01-01 00:00:00",
                "sim_end": "2017-01-01 00:00:00 "
            }
        }
        #st.json(config_data)
        st.code(json.dumps(config_data, indent=2), language="json")

    # ------------------------------------------
    #spacer at page bottom
    # ------------------------------------------
    st.write(' ')
    st.write(' ')
    st.write(' ')
    st.write(' ')
