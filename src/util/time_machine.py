
from datetime import datetime
from src.util.config import CONFIG

def simulated_time() -> datetime:
    experiment = CONFIG["experiment"]

    return map_to_simulated_time(
        real_time = datetime.now(),
        real_start = experiment["real_start"],
        real_end = experiment["real_end"],
        sim_start = experiment["sim_start"],
        sim_end = experiment["sim_end"]
    )

def map_to_simulated_time(
    real_time: datetime,
    real_start: datetime,
    real_end: datetime,
    sim_start: datetime,
    sim_end: datetime,
) -> datetime:
    # normalize to [0, 1]
    ratio = (real_time - real_start) / (real_end - real_start)

    # map to simulated range
    sim_time = sim_start + ratio * (sim_end - sim_start)

    return sim_time