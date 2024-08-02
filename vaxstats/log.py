import threading
import time
from math import floor

from loguru import logger


def log_progress(stop_event):
    intervals = [(15, 5), (60, 15), (180, 30), (float("inf"), 60)]

    start_time = time.time()
    interval_index = 0
    next_log_time = start_time + intervals[interval_index][1]

    while not stop_event.is_set():
        current_time = time.time()
        if current_time >= next_log_time:
            elapsed_time = current_time - start_time
            logger.info(f"Elapsed time: {floor(elapsed_time):.0f} seconds")

            if (
                elapsed_time >= intervals[interval_index][0]
                and interval_index < len(intervals) - 1
            ):
                interval_index += 1

            next_log_time = current_time + intervals[interval_index][1]

        time.sleep(1)
    logger.info(f"Finished in {elapsed_time:.2f} seconds")


def run_with_progress_logging(func, *args, **kwargs):
    stop_event = threading.Event()
    progress_thread = threading.Thread(target=log_progress, args=(stop_event,))

    try:
        progress_thread.start()
        result = func(*args, **kwargs)
    finally:
        stop_event.set()
        progress_thread.join()

    return result
