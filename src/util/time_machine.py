
from datetime import datetime


def simulated_time() -> datetime:
    return map_to_simulated_time(
        real_time = datetime.now(),
        real_start = datetime(2026, 3, 20),
        real_end = datetime(2026, 4, 1),
        sim_start = datetime(2005, 1, 1),
        sim_end = datetime(2017,1,1)
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