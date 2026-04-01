import os
import subprocess
import sys
import logging
import argparse
from apscheduler.schedulers.blocking import BlockingScheduler
from datetime import datetime
from datetime import timedelta

from src.util.config import CONFIG
#from src.models.track_model import main as run_track_model_pipeline
from src.models.track_model_remote import main as run_track_model_remote_pipeline

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

#new
parser.add_argument(
    "--hours", 
    type=int, 
    default=None, 
    help="Run for hours"
)

'''
parser.add_argument(
    "--minutes_total", 
    type=int, 
    default=None, 
    help="Run for minutes"
)
'''

args = parser.parse_args()

def run_my_pipeline():
    logger.info(f"[{datetime.now()}] Starting pipeline")
    
    try:
        #run_track_model_pipeline()
        run_track_model_remote_pipeline()
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
        print(f"ERROR: Could not find script. Check your folder structure!")
        logger.info(f"ERROR: Could not find script. Check your folder structure!")
    
    except Exception as e:
        print(f"Unexpected Scheduler Error: {e}")
        logger.error(f"Unexpected Scheduler Error: {e}")

scheduler = BlockingScheduler()

# calculate end date #new
#stop_time = datetime.now() + timedelta(hours=1)
stop_time = None
if args.hours is not None:
    stop_time = datetime.now() + timedelta(hours=args.hours)
    print(f"Auto-Stop scheduled at: {stop_time.strftime('%H:%M:%S')}")
    logger.info(f"Auto-Stop scheduled at: {stop_time}")
else:
    print("Mode: Run with no endtime set")
    logger.info("Mode: Run with no endtime set")

# Schedule
# eg:
#scheduler.add_job(run_my_pipeline, 'cron', hour=2, minute=0)
#scheduler.add_job(run_my_pipeline, 'interval', hours=6)
#scheduler.add_job(run_my_pipeline, 'interval', minutes=5)

scheduler.add_job(
    run_my_pipeline, 
    'interval', 
    minutes=args.interval, 
    next_run_time=datetime.now(), # starts immediately
    end_date=stop_time            # end job #new
)

print("--- Road Accidents France Scheduler ---")
print(f"Target: {os.path.join('scripts', 'run_pipeline.py')}")
print(f"Interval: {args.interval} minutes") #new
print("Status: Running... (Press Ctrl+C to stop)")

logger.info("--- Road Accidents France Scheduler ---")
logger.info(f"Target: {os.path.join('scripts', 'run_pipeline.py')}")
logger.info(f"Interval: {args.interval} minutes") #new
logger.info("Status: Running...")

try:
    scheduler.start()
    if stop_time is not None: #new
        logger.info("Scheduler reached end_date and stopped automatically.")
except (KeyboardInterrupt, SystemExit):
    pass
