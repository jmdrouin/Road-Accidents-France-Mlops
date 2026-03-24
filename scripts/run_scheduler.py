import os
import subprocess
import sys
import logging
import argparse
from apscheduler.schedulers.blocking import BlockingScheduler
from datetime import datetime
from src.util.config import CONFIG
from src.models.track_model_remote import main as run_track_model_pipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("logs/scheduler.log"), # Saves to file
        logging.StreamHandler(sys.stdout) # Prints to terminal
    ]
)
logger = logging.getLogger(__name__)

default_interval = CONFIG["runtime"]["scheduler_interval"]

parser = argparse.ArgumentParser(description="Road Accident Pipeline Scheduler")
parser.add_argument(
    "interval", 
    type=int, 
    nargs="?", 
    default=default_interval, 
    help="Interval in minutes (default: 5)"
)
args = parser.parse_args()

def run_my_pipeline():
    logger.info(f"[{datetime.now()}] Starting pipeline")
    
    try:
        run_track_model_pipeline()
        print(f"[{datetime.now()}] Pipeline finished successfully.")
        logger.info(f"[{datetime.now()}] Pipeline finished successfully.")
        
    except subprocess.CalledProcessError as e:
        print(f"[{datetime.now()}] ERROR: Pipeline failed.")
        logger.error(f"PIPELINE FAILED (Exit Code {e.returncode})")

        if e.stdout:
            logger.info(f"Last Output before crash:\n{e.stdout}")
        
        if e.stderr:
            #Traceback
            logger.error(f"TRACEBACK / ERROR DETAILS:\n{e.stderr}")

    except FileNotFoundError:
        print(f"ERROR: Could not find {script_path}. Check your folder structure!")
        logger.info(f"ERROR: Could not find {script_path}. Check your folder structure!")
    
    except Exception as e:
        print(f"Unexpected Scheduler Error: {e}")
        logger.error(f"Unexpected Scheduler Error: {e}")

scheduler = BlockingScheduler()

# Schedule
# eg:
#scheduler.add_job(run_my_pipeline, 'cron', hour=2, minute=0)
#scheduler.add_job(run_my_pipeline, 'interval', hours=6)
#scheduler.add_job(run_my_pipeline, 'interval', minutes=5)

scheduler.add_job(
    run_my_pipeline, 
    'interval', 
    minutes=args.interval, 
    next_run_time=datetime.now() # Starts immediately
)

print("--- Road Accidents France Scheduler ---")
print(f"Target: {os.path.join('scripts', 'run_pipeline.py')}")
print("Status: Running... (Press Ctrl+C to stop)")

logger.info("--- Road Accidents France Scheduler ---")
logger.info(f"Target: {os.path.join('scripts', 'run_pipeline.py')}")
logger.info("Status: Running...")

try:
    scheduler.start()
except (KeyboardInterrupt, SystemExit):
    pass
