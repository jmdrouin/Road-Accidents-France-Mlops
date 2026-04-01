
import streamlit as st
import subprocess
import os
import signal
import time

def streamlit_schedule():

    st.title("Log Monitor")

    # Pfad zu deiner Log-Datei
    current_dir = os.path.dirname(os.path.abspath(__file__))
    LOG_FILE = os.path.join(current_dir, "..", "logs", "scheduler.log")

    def get_last_n_lines(file_path, n=10):
        if not os.path.exists(file_path):
            return ["Log-Datei noch nicht erstellt."]
        with open(file_path, "r") as f:
            # Liest alle Zeilen und nimmt die letzten n
            return f.readlines()[-n:]
    
    # Platzhalter für die Logs (damit sie an der gleichen Stelle bleiben)
    log_placeholder = st.empty()

    # Checkbox zum Ein-/Ausschalten des Live-Updates
    auto_refresh = st.checkbox("Live-Update (alle 5 Sek)", value=False)

    while auto_refresh:
        # Letzte 10 Zeilen holen
        lines = get_last_n_lines(LOG_FILE, n=10)
        
        # Text formatieren und im Platzhalter anzeigen
        # .code() eignet sich super für Logs (Monospace-Schrift)
        log_placeholder.code("".join(lines), language="text")
        
        time.sleep(5)  # Wartezeit bis zum nächsten Update
        
        # Falls der User die Checkbox während der Schleife deaktiviert
        if not auto_refresh:
            break

    '''
    st.title("Scheduler xxx")

    # Wir nutzen den Session State, um den Prozess über Reruns hinweg zu speichern
    if "scheduler_process" not in st.session_state:
        st.session_state.scheduler_process = None

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🚀 Start Scheduler"):
            if st.session_state.scheduler_process is None:
                # Startet den Prozess im Hintergrund
                proc = subprocess.Popen(
                    ["uv", "run", "scripts/run_scheduler.py", "5"],
                    creationflags=subprocess.CREATE_NEW_CONSOLE # Öffnet ein eigenes Fenster unter Windows
                )
                st.session_state.scheduler_process = proc
                st.success(f"Started (PID: {proc.pid})")
            else:
                st.warning("Scheduler already running.")

    with col2:
        if st.button("🛑 Stop Scheduler"):
            if st.session_state.scheduler_process is not None:
                proc = st.session_state.scheduler_process
                # Beendet den Prozess und alle Kind-Prozesse (uv/python)
                os.system(f"taskkill /F /T /PID {proc.pid}")
                st.session_state.scheduler_process = None
                st.error("Scheduler stopped.")
            else:
                st.info("No running Scheduler found.")
    '''
