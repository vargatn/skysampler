"""

"""

import os
import yaml
import logging
import time
import sys


def get_poject_path(user_cfg):
    """Loads the absolute path to the cluster pipeline"""
    path = os.path.expanduser('~') + '/' + user_cfg
    project_path = None
    if os.path.isfile(path):
        with open(path)as file:
            cfg = yaml.safe_load(file)
            project_path = cfg["project_path"]
    return project_path


def read_yaml(cfg):
    """read directory tree form config file"""
    with open(cfg)as file:
         cfg = yaml.safe_load(file)
    return cfg


def read_paths(cfg, project_path):
    """replaces PACKAGE with [project_path]"""
    _settings = read_yaml(cfg)
    settings = {}
    for key in _settings.keys():
        if _settings[key] is not None:
            settings.update({key:_settings[key].replace("PACKAGE", project_path)})
        else:
            settings.update({key:_settings[key]})
    return settings


preset_logging_levels = {
    0: logging.CRITICAL,
    1: logging.WARNING,
    2: logging.INFO,
    3: logging.DEBUG
}

def setup_logger(tag, level, logfile_info):
    """
    Setup Logger

    Parameters
    ----------
    tag: string
        tag
    level: int
        0: logging.CRITICAL,
        1: logging.WARNING,
        2: logging.INFO,
        3: logging.DEBUG

    logfile_info: dict
        Contains stream, or filename

    Returns
    -------
    logger
    """
    tag = tag.upper()

    logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s %(message)s",
                        level=preset_logging_levels[level], stream=sys.stdout)
    # TODO add file logging here

    return logging.getLogger(tag)


def assign_logfile(settings):

    logfile_info = {
        "stream": sys.stdout,
    }
    if settings["logger_path"] is not None:
        logfile_info = {
            "filename": settings["logger_path"].replace("STAMP", time.strftime("%Y%m%d-%H%M%S")),
        }

    return logfile_info


user_project_file = '.skysampler.yaml'
project_path = get_poject_path(user_project_file)

default_paths_file = project_path + "/settings/default_paths.yaml"
custom_paths_file = project_path + "/settings/paths.yaml"

if project_path is not None:
    config = read_paths(default_paths_file, project_path)
    custom_settings = read_paths(custom_paths_file, project_path)
    config.update(custom_settings)


    config.update(read_yaml(config["config_path"]))
    logfile_info = assign_logfile(config)

    logger = setup_logger("PATHS", level=config["logging_level"], logfile_info=logfile_info)
    logger.info("config read from file: " + config["config_path"])





