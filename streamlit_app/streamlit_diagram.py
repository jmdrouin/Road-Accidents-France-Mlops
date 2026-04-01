import streamlit as st
import os
import time

# warnings
import warnings
warnings.filterwarnings("ignore")

# ----------------------------------------------
# function, attributes
# ----------------------------------------------

# ----------------------------------------------
# Page: Architecture
# ----------------------------------------------

def streamlit_diagram():

    # ------------------------------------------
    # page header
    # ------------------------------------------

    st.title("Architecture")
    st.write(' ')

    current_dir = os.path.dirname(__file__)

    #tab2 "Pipeline", 
    tab1, tab2, tab3, tab4 = st.tabs(["Diagram", "Structure", "Configuration", "Schedule Log"])

    # ------------------------------------------
    # Tab: Architecture Diagram
    # ------------------------------------------
    with tab1:

        st.subheader("Components")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            symbol_path_1 = os.path.join(current_dir, "symbols", "docker.png")
            st.image(symbol_path_1, width=100)
            with st.popover("Docker", use_container_width=True):
                st.markdown("##### Docker: Isolation and Virtualization")
                st.markdown("""
                * **Standardization**: Guarantees a consistent environment across development, testing, and production. 
                * **Encapsulation**: Packages the ML model, code, and specific library versions into a portable unit.  
                * **Efficiency**: Provides process-level isolation that is faster and uses fewer resources than Virtual Machines.  
                """)

        with col2:
            symbol_path_2 = os.path.join(current_dir, "symbols", "fastapi.png")
            st.image(symbol_path_2, width=100)
            with st.popover("FastAPI", use_container_width=True):
                st.markdown("##### FastAPI: Deployment in a microservice architecture")
                st.markdown("""
                * **Deployment**: Wraps service methods into a web API providing endpoints for clients to interact with the model.  
                * **Decoupling**: Provides an API contract decoupling the service from clients.  
                * **Security**: Provides authentication and authorization.  
                """)

        with col3:
            symbol_path_3 = os.path.join(current_dir, "symbols", "mlflow.png")
            st.image(symbol_path_3, width=100)
            with st.popover("MLflow", use_container_width=True):
                st.markdown("##### MLflow with DagsHub: Remote experiment tracking & model versioning")
                st.markdown("""
                * **Experiment Tracking**: Logs parameters, metrics, and artifacts during training runs. 
                * **Model Registry**: Serves as a central hub to version models.  
                * **Access Control**: Provides built-in, team-based access controls.  
                * **DagsHub Server**: Integrates Git, and provides a hosted, remote MLflow tracking server offering a web-based UI to visualize MLflow experiment runs side-by-side.  
                """)

        with col4:
            symbol_path_4 = os.path.join(current_dir, "symbols", "scheduler.png")
            st.image(symbol_path_4, width=100)
            with st.popover("Scheduler", use_container_width=True):
                st.markdown("##### Scheduler: Automated, reliable workflow")
                st.markdown("""
                * **Automation of Pipelines**: Executes ETL (Extract, Transform, Load), model training, tracking, and evaluation at specific intervals
Dependency Management: Ensures tasks happen in a defined order.  
                * **Error Handling and logging**: Encapsulates error handling and logging.  
                """)

        st.divider()
        st.subheader("Architecture Diagram")
        
        img_path = os.path.join(current_dir, "pics", "mlops_accidents_arch_diagram.png")

        if os.path.exists(img_path):
            st.image(img_path, caption="MLOps Architecture Diagram")
        else:
            st.error(f"Diagram not found: {img_path}")
    
    # ------------------------------------------
    # Tab: Project Structure
    # ------------------------------------------
    with tab2:
        st.subheader("Tree structure")

        file_path = "project_structure.txt"

        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-16") as f:
                tree_content = f.read()
            
            st.write("Project tree structure of Road-Accidents-France-Mlops:")
            st.code(tree_content, language="text")
        else:
            st.error(f"File not found: {file_path}")
    
    # ------------------------------------------
    # Tab: Configuration files
    # ------------------------------------------
    with tab3:
        st.subheader("Configuration files")

        def display_file_content(file_path, language="yaml"):
        
            label = f"Show content of {os.path.basename(file_path)}"
            if st.checkbox(label, key=file_path): 
                if os.path.exists(file_path):
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()
                        st.code(content, language=language)
                    except Exception as e:
                        st.error(f"Error: {e}")
                else:
                    st.warning(f"File not found: {file_path}")
        
        display_file_content("docker-compose.yml", language="yaml")
        #st.divider()
        display_file_content("requirements.txt", language="text")
    
    # ------------------------------------------
    # Tab: Schedule Log Monitor
    # ------------------------------------------
    with tab4:
        st.subheader("Schedule Logfile")
        
        #current_dir = os.path.dirname(os.path.abspath(__file__))
        LOG_FILE = os.path.join(current_dir, "..", "logs", "scheduler.log")

        def get_last_n_lines(file_path, n=10):
            if not os.path.exists(file_path):
                return ["Log file not yet created."]
            with open(file_path, "r") as f:
                # Read rows and take last n
                return f.readlines()[-n:]
        
        # placeholder for logs (to stay in same place)
        log_placeholder = st.empty()

        # checkbock on/off live updates
        auto_refresh = st.checkbox("Show Logfile Tail", value=False) #Live-Update (every 5 sec)

        while auto_refresh:
            # get 10 rows
            lines = get_last_n_lines(LOG_FILE, n=10)
            
            # format text and show in placeholder
            # .code() goes well for logs (Monospace)
            log_placeholder.code("".join(lines), language="text")
            
            time.sleep(5) # wait for next update
            
            # in case user deactivates checkbox
            if not auto_refresh:
                break

    # ------------------------------------------
    #spacer at page bottom
    # ------------------------------------------
    st.write(' ')
    st.write(' ')
    st.write(' ')
    st.write(' ')
