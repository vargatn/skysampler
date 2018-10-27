"""
Default Logger setup
"""

import sys
import logging
from . import paths


logging_levels = { 0: logging.CRITICAL,
                   1: logging.WARNING,
                   2: logging.INFO,
                   3: logging.DEBUG }


def setup_logger(logger_tag, logfile=None, level=1):
    """
    Setup Logger

    Parameters
    ----------
    logfile: string
        Name of logfile to write to. If `None` then writes to `stdout`, else the file is created in the default
        logger directory
    level: int
        0: logging.CRITICAL,
        1: logging.WARNING,
        2: logging.INFO,
        3: logging.DEBUG

    Returns
    -------
    logger
    """

    if logfile is None:
        logging.basicConfig(format="%(asctime)s %(name)s %(message)s", level=logging_levels[level], stream=sys.stdout)
    else:
        logging.basicConfig(format="%(asctime)s %(name)s %(message)s", level=logging_levels[level],
                            filename=paths.default_logger_path + logfile)

    return logging.getLogger(logger_tag)
