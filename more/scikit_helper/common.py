# For Time Logging
import time
from contextlib import contextmanager
import logging


@contextmanager
# Timing Function
def time_usage(name=""):
    """
    log the time usage in a code block
    """
    # print ("In time_usage runID = {}".format(runID))
    start = time.time()
    yield
    end = time.time()
    elapsed_seconds = float("%.10f" % (end - start))
    logging.info('%s: Time Taken (seconds): %s', name, elapsed_seconds)
