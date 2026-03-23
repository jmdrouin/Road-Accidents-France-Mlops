from src.util.time_machine import simulated_time
import streamlit as st

def display_simulated_date():
    sim_now = simulated_time()

    st.markdown("### Simulated date")

    col1, col2 = st.columns([5,1])
    with col2:
        st.button("↻", key="refresh_sim_date", use_container_width=True)
    with col1:
        st.markdown(
            f"""
            <div style="
                padding: 0.8rem 1.2rem;
                border-radius: 10px;
                background-color: #111111;
                font-size: 1.4rem;
                font-weight: 600;
                text-align: center;
            ">
                {sim_now.strftime("%B %d %Y • %H:%M")}
            </div>
            """,
            unsafe_allow_html=True,
        )