import streamlit as st

# warnings
import warnings
warnings.filterwarnings("ignore")

# ----------------------------------------------
# Page: Next steps
# ----------------------------------------------

def streamlit_next():
    
    # ------------------------------------------
    # page header
    # ------------------------------------------

    st.title("Next steps")

    #Here are some suggestions for further improvements to this project:
    #suggestions
    #proposed improvements
    st.write("Below are some proposed improvements to transition this MLOps project to a full production environment:")
    st.write(" ")

    with st.expander("🛠️ 1. Infrastructure & Security", expanded=True):
        st.markdown("""
        *   **Scheduling**: Replace the current scheduler script with an orchestrator like **Airflow**. This allows for automatic retries and provides a visual interface to monitor pipeline health.
        *   **API Security**: Implement **OAuth2 or API Key authentication** in FastAPI to ensure that only authorized services can trigger retraining or access the prediction engine.
        *   **Database Migration**: Transition from SQLite to a production-grade database like **PostgreSQL** for better concurrency and scalability.
        """)
    
    with st.expander("🧪 2. Automated Testing & CI/CD Integration"):
        st.markdown("""
        *   **Unit Testing for API Logic**: Implement **pytest** to automatically verify that endpoints handle data correctly.
        *   **Automated Testing in CI/CD**: Set up **GitHub Actions** to run the test suite automatically on every push to prevent "broken" code from reaching production. 
        *   **CI/CD for Docker**: Automate the Docker build and deployment process. Every time the code changes, the API and Scheduler containers will be automatically built, tested, and updated.
        """)
    
    with st.expander("📊 3. Model Management (ML Life Cycle)"):
        st.markdown("""
        *   **Dynamic Training Pipeline**: Shift from a "hard-coded" best model (based on a previous Data Science project) to a flexible training module.
        *   **Experiment Tracking**: Use **MLflow** to track and compare hyperparameter sweeps and different model architectures within the pipeline.
        """)
    
    # Conclusion
    st.divider()
    st.info("**Goal:** Moving from a functional prototype to a secure, scalable, and self-healing MLOps ecosystem.")

    '''
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
    '''

    # ------------------------------------------
    #spacer at page bottom
    # ------------------------------------------
    st.write(' ')
    st.write(' ')
    st.write(' ')
    st.write(' ')
